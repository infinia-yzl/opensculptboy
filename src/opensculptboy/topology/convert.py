# OpenSculptBoy
# Apache License, Version 2.0
"""
Carry per-vertex fields of a detail level between anny's mesh and the ``sculpt`` mesh.

A field is any per-vertex array of a level (``opensculptboy.detail.subdivision.level``):
detail offsets, masks, per-vertex maps. :func:`to_sculpt` and :func:`to_anny` keep the
value of every vertex that both meshes share and transfer the others barycentrically from the
closest point of the other mesh's patched region:

- level 0: anny's vertex i is sculpt's vertex i; the new vertices of the patches and anny's
  vertices that ``sculpt`` no longer uses (inside the removed faces) are transferred;
- levels 1 and 2: a vertex of a level is the vertex, edge or face point of an element of the
  level above; two vertices are the same when they come from the same element (the same
  vertex, the same edge, the same face), recursively down to level 0, and the others are
  transferred.

The closest points are taken on the rest mesh of ``vertices`` (anny's (N, 3) rest vertices at
level 0, by default anny's default body), subdivided as the level is.
"""

from __future__ import annotations

import pathlib

import numpy as np

from opensculptboy.detail.subdivision import level, mixed_edges, split_faces


def _keys(levels_faces, base_keys, table):
    """The element key of every vertex of each level, interned in ``table``."""
    keys = [np.asarray(base_keys, dtype=np.int64)]

    def intern(key):
        return table.setdefault(key, len(table))

    for faces in levels_faces:
        prev = keys[-1]
        edges, face_edges = mixed_edges(faces)
        tri = faces[:, 2] == faces[:, 3]
        vertex_keys = [intern(("v", int(k))) for k in prev]
        edge_keys = [
            intern(("e",) + tuple(sorted((int(prev[a]), int(prev[b])))))
            for a, b in edges
        ]
        face_keys = [
            intern(("f",) + tuple(sorted(int(prev[v]) for v in (f[:3] if t else f))))
            for f, t in zip(faces, tri)
        ]
        keys.append(np.array(vertex_keys + edge_keys + face_keys, dtype=np.int64))
        del face_edges
    return keys


def _closest_barycentric(points, triangles, chunk: int = 2048):
    """For each point, the index of the closest triangle (T, 3, 3) and the barycentric
    coordinates (3,) of its closest point (Ericson, Real-Time Collision Detection 5.1.5)."""
    points = np.asarray(points, dtype=np.float64)
    a, b, c = (triangles[:, i] for i in range(3))
    best_t = np.zeros(len(points), dtype=np.int64)
    best_w = np.zeros((len(points), 3))
    best_d = np.full(len(points), np.inf)
    for start in range(0, len(points), max(1, chunk)):
        p = points[start : start + chunk, None, :]
        ab, ac, ap = b - a, c - a, p - a
        d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
        bp = p - b
        d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
        cp = p - c
        d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = va + vb + vc
            v = np.where(denom != 0, vb / denom, 0.0)
            w = np.where(denom != 0, vc / denom, 0.0)
            u = 1.0 - v - w
            # regions of the vertices and edges
            t_ab = np.where(d1 - d3 != 0, d1 / (d1 - d3), 0.0)
            t_ac = np.where(d2 - d6 != 0, d2 / (d2 - d6), 0.0)
            t_bc = np.where(
                (d4 - d3) + (d5 - d6) != 0, (d4 - d3) / ((d4 - d3) + (d5 - d6)), 0.0
            )
        cases = [
            ((d1 <= 0) & (d2 <= 0), (1.0, 0.0, 0.0)),
            ((d3 >= 0) & (d4 <= d3), (0.0, 1.0, 0.0)),
            ((d6 >= 0) & (d5 <= d6), (0.0, 0.0, 1.0)),
            ((vc <= 0) & (d1 >= 0) & (d3 <= 0), (1 - t_ab, t_ab, 0.0 * t_ab)),
            ((vb <= 0) & (d2 >= 0) & (d6 <= 0), (1 - t_ac, 0.0 * t_ac, t_ac)),
            (
                (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
                (0.0 * t_bc, 1 - t_bc, t_bc),
            ),
        ]
        bu, bv, bw = u.copy(), v.copy(), w.copy()
        done = np.zeros(u.shape, bool)
        for mask, (cu, cv, cw) in cases:
            sel = mask & ~done
            bu = np.where(sel, cu, bu)
            bv = np.where(sel, cv, bv)
            bw = np.where(sel, cw, bw)
            done |= sel
        q = bu[..., None] * a + bv[..., None] * b + bw[..., None] * c
        dist = ((q - p) ** 2).sum(-1)
        t = dist.argmin(1)
        rows = np.arange(len(t))
        s = slice(start, start + len(t))
        best_t[s] = t
        best_d[s] = dist[rows, t]
        best_w[s] = np.stack([bu[rows, t], bv[rows, t], bw[rows, t]], 1)
    return best_t, best_w


class LevelCorrespondence:
    """The shared vertices of level ``subdiv`` of anny and sculpt, and the transfer of the
    others (see the module docstring)."""

    def __init__(self, subdiv: int, vertices=None, data=None):
        from opensculptboy.topology.patch import base_quad_mesh
        from opensculptboy.topology.sculpt import sculpt_data, use_sculpt_data

        data = sculpt_data() if data is None else data
        with use_sculpt_data(data):
            sculpt_levels = [level("sculpt", k) for k in range(subdiv + 1)]
        anny_levels = [level("anny", k) for k in range(subdiv + 1)]
        n, m = data.vertex_count, data.new_vertex_count
        table: dict = {}
        anny_keys = _keys(
            [lv.quads for lv in anny_levels[:-1]],
            [table.setdefault(("v0", i), len(table)) for i in range(n)],
            table,
        )[-1]
        sculpt_keys = _keys(
            [lv.quads for lv in sculpt_levels[:-1]],
            [table.setdefault(("v0", i), len(table)) for i in range(n + m)],
            table,
        )[-1]
        a, s = anny_levels[-1], sculpt_levels[-1]
        index = {int(k): i for i, k in enumerate(anny_keys)}
        self.sculpt_to_anny = np.array(
            [index.get(int(k), -1) for k in sculpt_keys], dtype=np.int64
        )
        referenced = np.zeros(s.vertex_count, bool)
        referenced[np.unique(s.quads)] = True
        self.anny_to_sculpt = -np.ones(a.vertex_count, dtype=np.int64)
        shared = (self.sculpt_to_anny >= 0) & referenced
        self.anny_to_sculpt[self.sculpt_to_anny[shared]] = np.nonzero(shared)[0]

        base = base_quad_mesh()
        v0 = base.vertices if vertices is None else np.asarray(vertices, np.float64)
        self.anny_positions = _apply(a.operators, v0)
        self.sculpt_positions = _apply(s.operators, data.positions(v0))
        kept = int(data.kept_faces().sum())
        removed = np.zeros(data.quad_count, bool)
        removed[data.removed_faces] = True
        new = np.arange(len(sculpt_levels[0].quads)) >= kept
        # the patched regions with a ring of faces around them: anny's faces from the removed
        # quads, sculpt's from the new faces, and the faces that touch them at level 0
        anny_region = _ring(anny_levels[0].quads, removed)
        sculpt_region = _ring(sculpt_levels[0].quads, new)
        self.anny_triangles = split_faces(a.quads[anny_region[a.base_faces]])
        self.sculpt_triangles = split_faces(s.quads[sculpt_region[s.base_faces]])
        self.subdiv = subdiv

    def to_sculpt(self, field):
        field = np.asarray(field)
        out = np.empty((len(self.sculpt_to_anny),) + field.shape[1:], field.dtype)
        known = self.sculpt_to_anny >= 0
        out[known] = field[self.sculpt_to_anny[known]]
        out[~known] = _transfer(
            self.sculpt_positions[~known],
            self.anny_positions,
            self.anny_triangles,
            field,
        )
        return out

    def to_anny(self, field):
        field = np.asarray(field)
        out = np.empty((len(self.anny_to_sculpt),) + field.shape[1:], field.dtype)
        known = self.anny_to_sculpt >= 0
        out[known] = field[self.anny_to_sculpt[known]]
        out[~known] = _transfer(
            self.anny_positions[~known],
            self.sculpt_positions,
            self.sculpt_triangles,
            field,
        )
        return out


def _ring(faces, mask):
    """The faces of ``mask`` and every face that shares a vertex with them."""
    touched = np.zeros(int(faces.max()) + 1, bool)
    touched[faces[mask].ravel()] = True
    return mask | touched[faces].any(1)


def _apply(operators, x):
    for op in operators:
        x = op.apply(x)
    return np.asarray(x)


def _transfer(points, positions, triangles, field):
    if len(points) == 0:
        return field[:0]
    if len(triangles) == 0:
        raise ValueError("No patched region to transfer from.")
    t, w = _closest_barycentric(points, positions[triangles])
    corners = triangles[t]
    flat = field.reshape(len(field), -1).astype(np.float64)
    values = (w[:, :, None] * flat[corners]).sum(1)
    return values.reshape((len(points),) + field.shape[1:]).astype(field.dtype)


def to_sculpt(field, subdiv: int = 0, vertices=None, data=None) -> np.ndarray:
    """A per-vertex field of level ``subdiv`` of anny, on level ``subdiv`` of sculpt."""
    return LevelCorrespondence(subdiv, vertices, data).to_sculpt(field)


def to_anny(field, subdiv: int = 0, vertices=None, data=None) -> np.ndarray:
    """A per-vertex field of level ``subdiv`` of sculpt, on level ``subdiv`` of anny."""
    return LevelCorrespondence(subdiv, vertices, data).to_anny(field)


def convert_offsets(
    source: str | pathlib.Path,
    target: str | pathlib.Path,
    topology: str,
    vertices=None,
    data=None,
) -> pathlib.Path:
    """
    Write the detail offsets file ``source`` (of either topology) as a file of ``topology``
    (``"anny"`` or ``"sculpt"``) at ``target``, with the same level and metadata.
    """
    from safetensors import safe_open

    from opensculptboy.detail.layer import load_offsets, save_offsets
    from opensculptboy.topology.sculpt import is_sculpt, use_sculpt_data, sculpt_data

    with safe_open(str(source), framework="numpy") as f:
        meta = dict(f.metadata() or {})
    subdiv, source_topology = int(meta["subdiv"]), meta["topology"]
    data = sculpt_data() if data is None else data
    with use_sculpt_data(data):
        offsets = load_offsets(source, subdiv, source_topology)
        if is_sculpt(source_topology) == is_sculpt(topology):
            converted = offsets
        elif is_sculpt(topology):
            converted = to_sculpt(offsets, subdiv, vertices, data)
        else:
            converted = to_anny(offsets, subdiv, vertices, data)
        extra = {
            k: v
            for k, v in meta.items()
            if k not in ("format", "subdiv", "topology", "vertices", "frame")
        }
        extra["converted_from"] = source_topology
        return save_offsets(target, converted, subdiv, topology, extra)


def _carry(correspondence, field, to_sculpt: bool) -> np.ndarray:
    """A field (P, V, ...) of P poses on a level carried to the other mesh: (P, V', ...)."""
    field = np.moveaxis(np.asarray(field, dtype=np.float64), 0, 1)
    if to_sculpt:
        out = correspondence.to_sculpt(field)
    else:
        out = correspondence.to_anny(field)
    return np.moveaxis(out, 1, 0)


def convert_eye_poses(
    source, target, topology: str, correspondence=None
) -> pathlib.Path:
    """
    Write the eye poses file ``source`` (:class:`~opensculptboy.eyes.poses.EyePoses`, whose
    offsets hold the rows of level 0 of the card's mesh) as a file of ``topology`` at
    ``target``: each pose's offsets carried as a field of level 0 and each detail level's
    corrections as a field of that level (``correspondence``, a function of the level to its
    :class:`LevelCorrespondence`; None keeps them, for the same mesh). The lashes' offsets
    hold the rows of the card's lashes and stay as they are.
    """
    from opensculptboy.eyes.poses import EyePoses
    from opensculptboy.topology.sculpt import is_sculpt

    poses = EyePoses.load(source)
    if correspondence is not None:
        to_sculpt = is_sculpt(topology)
        poses.offsets = _carry(correspondence(0), poses.offsets, to_sculpt)
        levels = {}
        for subdiv, (rows, values) in poses.levels.items():
            c = correspondence(subdiv)
            count = len(c.anny_to_sculpt) if to_sculpt else len(c.sculpt_to_anny)
            full = np.zeros((len(values), count, 3))
            full[:, rows] = values
            moved = _carry(c, full, to_sculpt)
            kept = np.flatnonzero(np.abs(moved).max(axis=(0, 2)) > 0.0)
            levels[subdiv] = (kept, moved[:, kept])
        poses.levels = levels
    poses.save(target)
    return pathlib.Path(target)


def convert_mouth_poses(
    source, target, topology: str, correspondence=None
) -> pathlib.Path:
    """
    Write the mouth poses file ``source`` (:class:`~opensculptboy.mouth.poses.MouthPoses`)
    as a file of ``topology`` at ``target``: the skin offsets carried as a field of level 0
    and each detail level's corrections as a field of that level (``correspondence``, a
    function of the level to its :class:`LevelCorrespondence`; None keeps them, for the same
    mesh). The points of the inner mouth stay as they are.
    """
    from opensculptboy.mouth.poses import MouthPoses
    from opensculptboy.topology.sculpt import is_sculpt

    poses = MouthPoses.load(source)
    if correspondence is not None:
        to_sculpt = is_sculpt(topology)
        poses.skin = _carry(correspondence(0), poses.skin, to_sculpt)
        levels = {}
        for subdiv, (rows, values) in poses.levels.items():
            c = correspondence(subdiv)
            count = len(c.anny_to_sculpt) if to_sculpt else len(c.sculpt_to_anny)
            full = np.zeros((len(values), count, 3))
            full[:, rows] = values
            moved = _carry(c, full, to_sculpt)
            kept = np.flatnonzero(np.abs(moved).max(axis=(0, 2)) > 0.0)
            levels[subdiv] = (kept, moved[:, kept])
        poses.levels = levels
    poses.save(target)
    return pathlib.Path(target)


CARD_TOPOLOGIES = ("anny", "sculpt")


def _card_rest(character) -> np.ndarray:
    """Anny's rest vertices (N, 3) of a card's parameters, float64."""
    import dataclasses

    import torch

    anny_card = dataclasses.replace(character, topology="anny", pose={})
    model = anny_card.build_model(dtype=torch.float64)
    kwargs = anny_card.model_kwargs()
    kwargs.pop("facial_actions", None)
    with torch.no_grad():
        return model(**kwargs)["rest_vertices"][0].numpy()


def _relocated(value, source_dir: pathlib.Path, target_dir: pathlib.Path):
    """A card's relative file path (or a nested section of them) seen from another folder."""
    import os

    if isinstance(value, dict):
        return {k: _relocated(v, source_dir, target_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [_relocated(v, source_dir, target_dir) for v in value]
    if isinstance(value, str) and not pathlib.Path(value).is_absolute():
        if (source_dir / value).is_file():
            return pathlib.Path(
                os.path.relpath(source_dir / value, target_dir)
            ).as_posix()
    return value


def convert_card(
    source: str | pathlib.Path,
    target: str | pathlib.Path,
    topology: str,
    data=None,
    body: bool = True,
) -> dict:
    """
    Write the character card ``source`` (of ``anny`` or ``sculpt``) as a card of ``topology``
    at ``target``: the same parameters, its detail offsets converted level by level
    (:func:`convert_offsets`, written next to the new card as ``<card>.detail<N>.safetensors``)
    and the ear tile's UVs of its skin material converted as a field of their level (UVs whose
    transfer touches a vertex off the ears are left off them), and the reference's eye poses
    (``material.reference_eyes.poses``) and mouth poses (``material.mouth.poses``) carried with
    the same row maps (:func:`convert_eye_poses`, :func:`convert_mouth_poses`, written next to
    the new card). The textures stay on anny's atlas, which ``sculpt`` keeps, and the other
    files of the card are named from the new card's folder. With ``body`` (the default), the
    closest points of the transfer are taken on the card's own rest body, else on anny's
    default body. Returns a summary.
    """
    import json

    from opensculptboy.character import Character
    from opensculptboy.topology.sculpt import is_sculpt, sculpt_data, use_sculpt_data

    if topology not in CARD_TOPOLOGIES:
        raise ValueError(
            f"Cards convert between {' and '.join(CARD_TOPOLOGIES)}; got {topology!r}."
        )
    source, target = pathlib.Path(source), pathlib.Path(target)
    character = Character.from_dict(json.loads(source.read_text(encoding="utf-8")))
    if character.topology not in CARD_TOPOLOGIES:
        raise ValueError(
            f"{source} has the topology {character.topology!r}; cards convert between "
            f"{' and '.join(CARD_TOPOLOGIES)}."
        )
    data = sculpt_data() if data is None else data
    source_dir, target_dir = source.resolve().parent, target.resolve().parent
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = target.name[: -len(target.suffix)] if target.suffix else target.name
    material_in = character.material or {}
    eye_poses = (material_in.get("reference_eyes") or {}).get("poses")
    mouth_poses = (material_in.get("mouth") or {}).get("poses")
    carried = character.detail or eye_poses or mouth_poses
    vertices = _card_rest(character) if body and carried else None
    same = is_sculpt(character.topology) == is_sculpt(topology)
    correspondences: dict = {}

    def correspondence(subdiv: int) -> LevelCorrespondence:
        if subdiv not in correspondences:
            with use_sculpt_data(data):
                correspondences[subdiv] = LevelCorrespondence(subdiv, vertices, data)
        return correspondences[subdiv]

    summary = {"card": str(target), "from": character.topology, "to": topology}
    detail = dict(character.detail)
    if detail:
        levels = []
        for entry in detail.get("levels", []):
            subdiv = int(entry["subdiv"])
            path = pathlib.Path(entry["offsets"])
            path = path if path.is_absolute() else source_dir / path
            name = f"{stem}.detail{subdiv}.safetensors"
            convert_offsets(path, target_dir / name, topology, vertices, data)
            levels.append(dict(entry, offsets=name))
        rest = {k: v for k, v in detail.items() if k != "levels"}
        detail = {**_relocated(rest, source_dir, target_dir), "levels": levels}
        summary["levels"] = [e["subdiv"] for e in levels]
    material = _relocated(dict(character.material), source_dir, target_dir)
    ears = (character.material or {}).get("ears") or {}
    if ears.get("uvs"):
        from opensculptboy.texture.ears import EarUVs

        path = pathlib.Path(ears["uvs"])
        path = path if path.is_absolute() else source_dir / path
        with use_sculpt_data(data):
            table = EarUVs.load(path, character.topology)
            uv = table.uv
            if not same:
                c = LevelCorrespondence(table.subdiv, vertices, data)
                uv = c.to_sculpt(uv) if is_sculpt(topology) else c.to_anny(uv)
            name = f"{stem}.ear_uvs.safetensors"
            EarUVs(uv, table.subdiv).save(target_dir / name, topology)
        material["ears"] = dict(material["ears"], uvs=name)
        summary["ear_uvs"] = name
    for section, kind, name in (
        ("reference_eyes", "eye_poses", eye_poses),
        ("mouth", "mouth_poses", mouth_poses),
    ):
        if not name:
            continue
        path = pathlib.Path(name)
        path = path if path.is_absolute() else source_dir / path
        out_name = f"{stem}.{kind}.safetensors"
        with use_sculpt_data(data):
            c = None if same else correspondence
            if kind == "eye_poses":
                convert_eye_poses(path, target_dir / out_name, topology, c)
            else:
                convert_mouth_poses(path, target_dir / out_name, topology, c)
        material[section] = dict(material[section], poses=out_name)
        summary[kind] = out_name
    import dataclasses

    converted = dataclasses.replace(
        character, topology=topology, detail=detail, material=material
    )
    target.write_text(
        json.dumps(converted.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return summary

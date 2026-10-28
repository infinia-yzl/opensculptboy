# OpenSculptBoy
# Apache License, Version 2.0
"""
A reference's closed eyes carried to a fitted card (docs/eyes.md, "Blinks").

A handoff may carry its reference's eye poses under ``eyes.poses`` of appearance.json, in the
form of the mouth's poses (docs/mouth.md, "Mouth expressions from the reference"): per pose, a
``joints`` object of joint deltas of the GLB file's skins, and ``known: false`` for a guess.
The poses ``close_l`` and ``close_r`` close the figure's left eye (+X) and right eye, and
``half_l`` and ``half_r`` hold them half closed (the reference's blink half way).

:func:`eye_poses` poses the reference with each of them and carries the motion to level 0 of
the fitted card: each vertex near an eye moves by the reference's own blended skin transform
at its closest point on the reference, applied to its own position, so the lid's rows that lie
off the reference's surface turn with the lid as the reference's do. The motion fades out
between :data:`FADE_IN` and :data:`FADE_OUT` eyeball radii from the eye's centre and is turned
into the card's rest frame by the rigid motion of the fitted head. The detail levels up to
:data:`LEVEL_CARRY` take their own carry of each pose, the same motion on the level's own
vertices, stored as the difference to level 0's pose subdivided: a level's detail offsets lie
in the rest frame and would keep their rest direction on a lid that turns about the eye. The
reference's lashes
(the parts with the role ``lashes``, in the order of
:func:`opensculptboy.eyes.lashes.reference_lashes`) take the same poses with their own skin,
so the lids and the lashes close as one piece. The exports take these poses as the blink
targets (:func:`opensculptboy.eyes.blink.blink_offsets`).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

FORMAT = "opensculptboy/eye-poses@1"
# the reference's poses that close each eye, and that hold it half closed
CLOSED = {"L": "close_l", "R": "close_r"}
HALF = {"L": "half_l", "R": "half_r"}
# the reference's motion reaches the card's vertices within FADE_IN eyeball radii of an eye's
# centre in full, and fades out to none at FADE_OUT
FADE_IN = 1.8
FADE_OUT = 2.4
# a vertex whose carried motion stands out of its neighbours' mean by more than ROUGH
# (metres) plus ROUGH_SHARE of the size of that mean is a rough spot: next to a fold of the
# reference's skin (the crease of the upper lid) two neighbouring vertices can find their
# closest points on either side of the fold, which move apart; the rough spots and the ring
# around them take the harmonic extension of the rest
ROUGH = 0.001
ROUGH_SHARE = 0.5
ROUGH_ROUNDS = 6
# the carried motion is smoothed by this many steps (:func:`smooth_motion`)
SMOOTH_PASSES = 3
# a vertex takes the motion of the closest point of the reference's skin that faces the same
# way (the cosine of the angle between the normals at least FACING), among the CANDIDATES
# nearest triangles: the skin on either side of a fold faces apart
FACING = 0.3
CANDIDATES = 24
# the detail levels from 1 up to this one take their own carry of each pose (the reference's
# skin transforms applied to the level's own vertices, its detail offsets included), stored
# as the difference to level 0's pose subdivided (:meth:`EyePoses.level_correction`): the
# detail offsets lie in the rest frame and do not turn with level 0's pose, so a closed lid
# showed the rest's relief of its rim and its crease as lumps and a deep fold
LEVEL_CARRY = 2
# a level's correction keeps the rows that it moves by more than this (metres)
LEVEL_TOLERANCE = 1e-7


@dataclasses.dataclass
class BlinkPose:
    """One eye's blink from a reference: the rest-frame offsets of the closed pose (N, 3) and
    of the half-closed pose (or None) of a mesh, and those of the reference's lashes (M, 3),
    rows of the card's lashes, or None."""

    closed: np.ndarray
    half: np.ndarray | None = None
    lash_closed: np.ndarray | None = None
    lash_half: np.ndarray | None = None


@dataclasses.dataclass
class EyePoses:
    """The rest-frame offsets (P, V0, 3) of a fitted card's level 0 under the reference's eye
    poses (``names``), the offsets (P, M, 3) of the reference's lashes under them or None,
    what the carry measured (``stats``), and per detail level (``levels``) the rows (n,) of
    the full level and the offsets (P, n, 3) that each pose adds there to level 0's pose
    subdivided (:data:`LEVEL_CARRY`)."""

    names: tuple
    offsets: np.ndarray
    stats: dict = dataclasses.field(default_factory=dict)
    lashes: np.ndarray | None = None
    levels: dict = dataclasses.field(default_factory=dict)

    def save(self, path) -> None:
        from safetensors.numpy import save_file

        tensors = {"offsets": np.ascontiguousarray(self.offsets, np.float32)}
        if self.lashes is not None:
            tensors["lashes"] = np.ascontiguousarray(self.lashes, np.float32)
        for level, (rows, values) in sorted(self.levels.items()):
            tensors[f"level{level}_rows"] = np.ascontiguousarray(rows, np.int64)
            tensors[f"level{level}_offsets"] = np.ascontiguousarray(values, np.float32)
        save_file(
            tensors,
            str(path),
            metadata={
                "format": FORMAT,
                "names": json.dumps(list(self.names)),
                "stats": json.dumps(self.stats),
            },
        )

    @classmethod
    def load(cls, path) -> "EyePoses":
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as f:
            meta = f.metadata() or {}
            if meta.get("format") != FORMAT:
                raise ValueError(f"{path} is not a {FORMAT} file.")
            lashes = f.get_tensor("lashes") if "lashes" in f.keys() else None
            levels = {}
            for key in f.keys():
                if key.startswith("level") and key.endswith("_rows"):
                    level = int(key[len("level") : -len("_rows")])
                    levels[level] = (
                        f.get_tensor(key).astype(np.int64),
                        f.get_tensor(f"level{level}_offsets").astype(np.float64),
                    )
            return cls(
                tuple(json.loads(meta["names"])),
                f.get_tensor("offsets").astype(np.float64),
                json.loads(meta.get("stats", "{}")),
                None if lashes is None else lashes.astype(np.float64),
                levels,
            )

    def level_correction(self, name: str, topology) -> np.ndarray | None:
        """The offsets (N, 3) that the pose ``name`` adds on the vertices of a detail mesh
        (``topology``, a :class:`~opensculptboy.detail.subdivision.DetailTopology`) to level
        0's pose subdivided, or None when the poses hold none for its level."""
        entry = self.levels.get(int(getattr(topology, "subdiv", 0)))
        if entry is None or name not in self.names:
            return None
        rows, values = entry
        mesh_rows = np.asarray(topology.rows)
        full = np.zeros(
            (max(int(mesh_rows.max(initial=0)), int(rows.max(initial=0))) + 1, 3)
        )
        full[rows] = values[self.names.index(name)]
        return full[mesh_rows]

    def closed(self) -> dict:
        """The closed pose's offsets (V0, 3) by side, for the sides it holds."""
        return {
            side: self.offsets[self.names.index(name)]
            for side, name in CLOSED.items()
            if name in self.names
        }

    def blink_poses(self) -> dict:
        """The :class:`BlinkPose` of each side whose closed pose it holds."""
        out = {}
        for side, name in CLOSED.items():
            if name not in self.names:
                continue
            k = self.names.index(name)
            h = self.names.index(HALF[side]) if HALF[side] in self.names else None
            lashes = self.lashes
            out[side] = BlinkPose(
                self.offsets[k],
                None if h is None else self.offsets[h],
                None if lashes is None else lashes[k],
                None if lashes is None or h is None else lashes[h],
            )
        return out


def known_eye_poses(appearance: dict) -> dict:
    """The poses of a handoff's ``eyes.poses`` (name to joint deltas), those marked known
    only; empty when it has none."""
    poses = (appearance.get("eyes") or {}).get("poses") or {}
    out = {}
    for name, pose in poses.items():
        if isinstance(pose, dict) and pose.get("known", True) is not False:
            joints = pose.get("joints") or {}
            if isinstance(joints, dict) and joints:
                out[name] = joints
    return out


def eye_poses(
    handoff,
    reference,
    fitted: np.ndarray,
    rest: np.ndarray,
    head: np.ndarray,
    eyes: dict,
    triangles: np.ndarray,
    poses: dict | None = None,
    levels=None,
) -> EyePoses | None:
    """
    The :class:`EyePoses` of a fitted level 0 (``fitted`` as the fit placed it on the
    reference, ``rest`` its rest vertices, ``head`` the rows that turn the reference's frame
    into the rest frame, ``eyes`` the rows of each eyeball by side, ``triangles`` its
    triangles) from the reference's eye poses (``poses``: name to joint deltas, the handoff's
    known ``close_l``, ``close_r``, ``half_l`` and ``half_r`` by default), with the
    reference's lashes under each pose. ``levels``, a list of (level, its
    :class:`~opensculptboy.detail.subdivision.DetailTopology`, its vertices as the fit placed
    them, the rows of each eyeball on it), adds each level's own carry as its correction to
    level 0's pose subdivided (:data:`LEVEL_CARRY`). None when the handoff has no closed eye.
    """
    from opensculptboy.eyes.geometry import fit_sphere
    from opensculptboy.mouth.poses import reference_pose_transforms
    from opensculptboy.teeth.mesh import rigid

    if poses is None:
        known = known_eye_poses(handoff.appearance)
        wanted = list(CLOSED.values()) + list(HALF.values())
        poses = {n: known[n] for n in wanted if n in known}
    if not any(n in poses for n in CLOSED.values()):
        return None
    transforms = reference_pose_transforms(handoff, reference, poses)
    if not transforms:
        return None
    fitted = np.asarray(fitted, dtype=np.float64)
    R, _ = rigid(fitted[head], np.asarray(rest)[head])
    spheres = [fit_sphere(fitted[rows]) for rows in eyes.values()]
    carry = _Carry(fitted, triangles, eyes, spheres, reference)
    fine = [
        (
            level,
            topology,
            _Carry(
                np.asarray(f, np.float64),
                topology.triangles,
                e,
                spheres,
                reference,
                by_transforms=True,
            ),
        )
        for level, topology, f, e in (levels or ())
    ]
    names, offsets, stats, lashes = [], [], {}, []
    corrections = {level: [] for level, _, _ in fine}
    lash_parts = [
        k
        for k, p in enumerate(handoff.parts)
        if p.role == "lashes" and len(p.triangles) and p.uvs is not None
    ]
    for name in poses:
        if name not in transforms:
            continue
        lashes.append(_lash_offsets(handoff, lash_parts, poses[name], R))
        values = carry.motion(transforms[name]) @ R.T
        for level, topology, level_carry in fine:
            own = level_carry.motion(transforms[name]) @ R.T
            corrections[level].append(own - np.asarray(topology.apply(values)))
        size = np.linalg.norm(values, axis=1) * 1000.0
        names.append(name)
        offsets.append(values)
        stats[name] = {
            "moved_vertices": int((size > 0.05).sum()),
            "max_mm": round(float(size.max()), 3),
        }
    if not names:
        return None
    stored = {}
    for level, topology, _ in fine:
        values = np.stack(corrections[level])  # (P, N, 3) on the level's mesh
        moved = np.abs(values).max(axis=(0, 2)) > LEVEL_TOLERANCE
        mesh_rows = np.asarray(topology.rows)
        stored[level] = (mesh_rows[moved], values[:, moved])
        for k, name in enumerate(names):
            stats[name][f"level{level}_mm"] = round(
                float(np.linalg.norm(values[k], axis=1).max(initial=0.0) * 1000.0), 3
            )
    with_lashes = all(m is not None for m in lashes) and bool(lash_parts)
    return EyePoses(
        tuple(names),
        np.stack(offsets),
        stats,
        np.stack(lashes) if with_lashes else None,
        stored,
    )


class _Carry:
    """The reference's motion of a pose carried to a mesh (``fitted`` (V, 3) as the fit
    placed it, its ``triangles``, the rows of its eyeballs ``eyes``): each vertex near an eye
    (``spheres``, the centre and radius of each eyeball on level 0) moves by the reference's
    blended skin transform at its closest facing point on the reference, applied to its own
    position; the rough spots are smoothed, then the motion, which fades out between
    :data:`FADE_IN` and :data:`FADE_OUT` eyeball radii. With ``by_transforms`` (the detail
    levels), the transforms themselves are smoothed, by as many steps, and then applied to
    the vertices: a smoothed motion would take the relief of the level's detail offsets out
    of the turn, and the level's own carry exists to turn that relief."""

    def __init__(
        self, fitted, triangles, eyes, spheres, reference, by_transforms=False
    ):
        self.by_transforms = by_transforms
        from opensculptboy.export.body import vertex_normals

        self.fitted = fitted
        self.triangles = np.asarray(triangles)
        on_eye = np.zeros(len(fitted), bool)
        for rows in eyes.values():
            on_eye[rows] = True
        fade = np.zeros(len(fitted))
        for centre, radius in spheres:
            d = np.linalg.norm(fitted - centre, axis=1) / radius
            fade = np.maximum(
                fade, np.clip((FADE_OUT - d) / (FADE_OUT - FADE_IN), 0.0, 1.0)
            )
        self.fade = fade
        self.near = np.flatnonzero((fade > 0) & ~on_eye)
        normals = vertex_normals(fitted, self.triangles)
        self.reference = reference
        self.tri, self.bary = closest_facing(
            fitted[self.near],
            normals[self.near],
            reference.vertices,
            reference.triangles,
        )

    def motion(self, transforms) -> np.ndarray:
        """The motion (V, 3) in the reference's frame under a pose's skin transforms."""
        from opensculptboy.mouth.poses import _transformed

        near = self.near
        if self.by_transforms:
            return self._by_transforms(transforms) * self.fade[:, None]
        motion = np.zeros_like(self.fitted)
        motion[near] = _transformed(
            self.fitted[near], self.reference, transforms, self.tri, self.bary
        )
        motion = smooth_rough(motion, self.triangles, near)
        motion = smooth_motion(motion, self.triangles, near)
        return motion * self.fade[:, None]

    def _by_transforms(self, transforms) -> np.ndarray:
        """The motion (V, 3) of the vertices under their blended transforms, smoothed over
        the near rows by :data:`SMOOTH_PASSES` steps (the other rows hold the identity)."""
        R, t = transforms
        corners = np.asarray(self.reference.triangles)[self.tri]
        field = np.zeros((len(self.fitted), 12))
        field[:, :9] = np.eye(3).ravel()
        field[self.near, :9] = np.einsum(
            "nk,nkij->nij", self.bary, np.asarray(R)[corners]
        ).reshape(-1, 9)
        field[self.near, 9:] = np.einsum(
            "nk,nki->ni", self.bary, np.asarray(t)[corners]
        )
        field = smooth_motion(field, self.triangles, self.near)
        x = self.fitted
        moved = (
            np.einsum("nij,nj->ni", field[:, :9].reshape(-1, 3, 3), x) + field[:, 9:]
        )
        out = np.zeros_like(x)
        out[self.near] = (moved - x)[self.near]
        return out


def _lash_offsets(handoff, rows, deltas, R) -> np.ndarray | None:
    """The rest-frame offsets (M, 3) of the reference's lash parts (``rows``, indices into
    ``handoff.parts``) under a pose's joint deltas, from their own skin; None when a part has
    no skin in the file."""
    from opensculptboy.fit.poses import posed_parts

    if not rows:
        return None
    posed = posed_parts(handoff, deltas)
    out = []
    for k in rows:
        if posed[k] is None:
            return None
        out.append((np.asarray(posed[k]) - handoff.parts[k].vertices) @ R.T)
    return np.concatenate(out)


def closest_facing(points, normals, vertices, triangles):
    """
    For each point (P, 3) with its unit normal: the closest triangle of a mesh among those
    whose normal agrees with the point's (:data:`FACING`, among the :data:`CANDIDATES`
    triangles whose centres lie nearest; the closest of all when none agrees), and the
    barycentric coordinates of the closest point. Returns (triangles (P,), bary (P, 3)).
    """
    from scipy.spatial import cKDTree

    from opensculptboy.mouth.poses import _closest_on_triangles

    points = np.asarray(points, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    face = np.cross(b - a, c - a)
    face /= np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-30)
    k = min(CANDIDATES, len(triangles))
    _, cand = cKDTree((a + b + c) / 3.0).query(points, k=k)
    cand = np.asarray(cand).reshape(len(points), k)
    best = np.full(len(points), np.inf)
    best_any = np.full(len(points), np.inf)
    tri = np.zeros(len(points), np.int64)
    tri_any = np.zeros(len(points), np.int64)
    bary = np.zeros_like(points)
    bary_any = np.zeros_like(points)
    for j in range(k):
        t = cand[:, j]
        q, bj = _closest_on_triangles(points, a[t], b[t], c[t])
        d = np.linalg.norm(points - q, axis=1)
        agrees = (face[t] * normals).sum(1) >= FACING
        better = agrees & (d < best)
        best[better], tri[better], bary[better] = d[better], t[better], bj[better]
        better = d < best_any
        best_any[better], tri_any[better], bary_any[better] = (
            d[better],
            t[better],
            bj[better],
        )
    none = ~np.isfinite(best)
    tri[none], bary[none] = tri_any[none], bary_any[none]
    return tri, bary


def smooth_rough(motion, triangles, rows) -> np.ndarray:
    """A carried motion (V, 3) with its rough spots among ``rows`` (:data:`ROUGH`) and the
    ring around them replaced by the harmonic extension of the others."""
    from opensculptboy.eyes.blink import neighbours
    from opensculptboy.fit.register import smooth_mouth

    A = neighbours(triangles, len(motion))
    graph = (A > 0).astype(float)
    inside = np.zeros(len(motion), bool)
    inside[rows] = True
    out = np.array(motion, dtype=np.float64, copy=True)
    for _ in range(ROUGH_ROUNDS):
        mean = A @ out
        limit = ROUGH + ROUGH_SHARE * np.linalg.norm(mean, axis=1)
        rough = inside & (np.linalg.norm(out - mean, axis=1) > limit)
        if not rough.any():
            break
        rough |= inside & (np.asarray(graph @ rough.astype(float)).ravel() > 0)
        out = smooth_mouth(out, triangles, rough)
    return out


def smooth_motion(motion, triangles, rows, passes: int = SMOOTH_PASSES) -> np.ndarray:
    """A carried motion (V, 3) smoothed over ``rows`` by ``passes`` steps that take each row
    half way to its neighbours' mean: a fitted skin holds small creases of its own, which a
    motion that differs from row to row deepens, and the lids' seal reshapes their rims."""
    from opensculptboy.eyes.blink import neighbours

    A = neighbours(triangles, len(motion))
    inside = np.zeros(len(motion), bool)
    inside[rows] = True
    out = np.array(motion, dtype=np.float64, copy=True)
    for _ in range(passes):
        step = 0.5 * out + 0.5 * (A @ out)
        out[inside] = step[inside]
    return out


def write_card_eye_poses(card_path, handoff, reference=None) -> pathlib.Path | None:
    """
    Compute the closed eyes of a fitted card (its level 0, from its offsets and placement,
    with the corrections of its detail levels up to :data:`LEVEL_CARRY`) from its handoff and
    write them as ``<card>.eye_poses.safetensors``, named under
    ``material.reference_eyes.poses`` of the card, which is written again. Returns the path,
    or None (and changes nothing) when the handoff has no closed eye or the card no eyes.
    """
    from opensculptboy.character import Character
    from opensculptboy.eyes.mesh import eyeball_rows
    from opensculptboy.fit.reference import Reference
    from opensculptboy.mouth.inner import fitted_meshes
    from opensculptboy.teeth.mesh import mouth_rows

    card_path = pathlib.Path(card_path)
    character = Character.from_dict(json.loads(card_path.read_text(encoding="utf-8")))
    if character.eye_material() is None:
        return None
    known = known_eye_poses(handoff.appearance)
    if not any(n in known for n in CLOSED.values()):
        return None
    reference = reference or Reference.from_handoff(handoff)
    model, topology, posed, rest = fitted_meshes(card_path, reference, 0)
    head, _ = mouth_rows(model, topology)
    levels = []
    for level in sorted(character.detail_levels()):
        if 1 <= level <= LEVEL_CARRY:
            _, level_topology, level_posed, _ = fitted_meshes(
                card_path, reference, level
            )
            levels.append(
                (
                    level,
                    level_topology,
                    level_posed,
                    eyeball_rows(model, level_topology),
                )
            )
    result = eye_poses(
        handoff,
        reference,
        posed,
        rest,
        head,
        eyeball_rows(model, topology),
        np.asarray(topology.triangles),
        levels=levels,
    )
    if result is None:
        return None
    stem = (
        card_path.name[: -len(card_path.suffix)] if card_path.suffix else card_path.name
    )
    path = card_path.parent / f"{stem}.eye_poses.safetensors"
    result.save(path)
    material = dict(character.material or {})
    material["reference_eyes"] = dict(
        material.get("reference_eyes") or {}, poses=path.name
    )
    character.material = material
    card_path.write_text(
        json.dumps(character.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return path


def card_eye_poses(character, detail_dir=None) -> EyePoses | None:
    """The :class:`EyePoses` that a card's ``material.reference_eyes.poses`` names, or
    None."""
    name = ((character.material or {}).get("reference_eyes") or {}).get("poses")
    if not name:
        return None
    return EyePoses.load(pathlib.Path(detail_dir or ".") / name)


def card_blink_poses(character, detail=None, detail_dir=None, rest=None) -> dict | None:
    """
    The reference's blinks of a card on the mesh of a file: side to :class:`BlinkPose`, with
    offsets of the file's base mesh, or None for a card without them. On a detail level
    (``detail``, a resolved detail level), level 0's offsets go to the level's vertices by
    the level's subdivision, and the level's own correction (:data:`LEVEL_CARRY`) completes
    them; for poses written before the corrections existed, the level's detail offsets turn
    with the lid (:func:`~opensculptboy.eyes.blink.turn_relief`) when the file's rest
    vertices (``rest``, (N, 3)) are given.
    """
    poses = card_eye_poses(character, detail_dir)
    if poses is None:
        return None
    blinks = poses.blink_poses()
    if not blinks:
        return None
    if detail is None:
        return blinks
    topology = detail.topology
    operators = topology.operators
    if operators:  # the poses hold the rows of level 0
        check_rows(blinks, operators[0].shape[1])
    relief = None
    for side, pose in blinks.items():
        for name, attr in ((CLOSED[side], "closed"), (HALF[side], "half")):
            value = getattr(pose, attr)
            if value is None:
                continue
            fine = detail.targets(value[None])[0]
            if operators:
                extra = poses.level_correction(name, topology)
                if extra is not None:
                    fine = fine + extra
                elif rest is not None:
                    if relief is None:
                        relief = level_relief(character, detail, detail_dir)
                    from opensculptboy.eyes.blink import turn_relief

                    fine = turn_relief(rest, fine, relief, topology.triangles)
            setattr(pose, attr, fine)
    return blinks


def level_relief(character, detail, detail_dir=None) -> np.ndarray:
    """The part (N, 3) of a detail level's offsets that level 0's eye poses do not turn: the
    level's offsets less level 0's offsets subdivided (the eye poses were carried on level 0
    with its own offsets)."""
    from opensculptboy.detail import character_offsets

    topology = detail.topology
    out = np.zeros((topology.vertex_count, 3))
    if detail.offsets is not None:
        out += np.asarray(detail.offsets, dtype=np.float64)
    level0 = character_offsets(character, 0, detail_dir)
    if level0 is not None:
        out -= np.asarray(topology.apply(np.asarray(level0, dtype=np.float64)))
    return out


def check_rows(blinks: dict, count: int) -> None:
    """Raise ValueError when a blink (side to :class:`BlinkPose`) holds another number of
    rows than ``count``, the rows of the mesh it applies to: the card's eye poses then belong
    to another topology, as after ``opensculptboy convert``."""
    for side, pose in blinks.items():
        if pose is not None and len(pose.closed) != count:
            raise ValueError(
                f"The reference's blink of the {side} eye holds {len(pose.closed)} rows, "
                f"and the mesh {count}: the card's eye poses belong to another topology "
                "(bake the card again after a conversion)."
            )

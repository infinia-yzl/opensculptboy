# OpenSculptBoy
# Apache License, Version 2.0
"""
The data that the web viewer needs to show a character card as its exports show it
(docs/viewer.md, "A card's look").

The page draws the fine body of :func:`anny.viewer.geometry.fine_body`: the mixed subdivision of
Anny's body quads, with three levels on the head and the neck and two elsewhere. A card's baked
textures lie on Anny's atlas, and its detail layer holds offsets on full levels of Anny's quad
mesh (:mod:`opensculptboy.detail`). This module gives the page:

- :func:`fine_uvs`: the UV of every corner of the fine body's triangles, by the face-varying
  bilinear subdivision of the detail layer (:func:`opensculptboy.detail.subdivision.subdivide_uvs`),
  so that the page samples the textures where the exports do;
- :func:`seam_split`: the fine vertices on the seams of the atlas, one copy per side, so that each
  copy holds the UV of its side;
- :func:`head_tile`: the fine vertices on the head tile's island and the tile's texture transform
  (:mod:`opensculptboy.texture.tiles`);
- :func:`detail_rows`: for the rows of the page's levels 1 and 2 (the first two operators of its
  subdivision) the row of the same point on the full levels of each card topology (``anny`` and
  ``sculpt``), so that the page adds a card's offsets of a level after that level, and the Anny
  vertex of each coarse vertex of the page.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# the topologies whose cards the page shows
TOPOLOGIES = ("anny", "sculpt")


@dataclasses.dataclass
class FineUVs:
    """The UVs of the fine body's triangles: ``uv_ids`` (T, 3) index ``uvs`` (U, 2, OBJ
    convention, v = 0 at the bottom); ``base_faces`` (T,) is the input quad of each triangle."""

    uv_ids: np.ndarray
    uvs: np.ndarray
    base_faces: np.ndarray


def fine_uvs(subdivision, uv_quads: np.ndarray, uvs: np.ndarray) -> FineUVs:
    """
    The UVs of the triangles of a :class:`anny.utils.subdivision.MixedSubdivision`, in the order of
    its ``triangles``, for UV quads (F, 4) of its input quads and their UVs (OBJ convention).

    The levels repeat the construction of the subdivision in UV space: the region's faces take the
    top level, the faces outside it the base level, and a base face that borders the region takes
    the UV of the top level's edge point on each split edge, in the same fan.
    """
    from anny.utils.subdivision import build_edges, catmull_clark, quads_to_triangles

    from opensculptboy.detail.subdivision import subdivide_uvs

    n = subdivision.input_vertex_count
    quads = np.asarray(subdivision.input_quads, dtype=np.int64)
    face_region = np.asarray(subdivision.region, dtype=bool)
    base_face = np.arange(len(quads))
    uq, uv = np.asarray(uv_quads, dtype=np.int64), np.asarray(uvs, dtype=np.float64)
    q = quads
    for _ in range(subdivision.base_level):
        level = catmull_clark(n, q)
        face_region = face_region[level.parent_faces]
        base_face = base_face[level.parent_faces]
        uq, uv = subdivide_uvs(uq, uv)
        n, q = level.operator.shape[0], level.quads
    base_count, base_quads, base_region = n, q, face_region
    top = catmull_clark(n, q)
    top_uq, top_uv = subdivide_uvs(uq, uv)
    top_region = base_region[top.parent_faces]
    triangles = [quads_to_triangles(top.quads[top_region])]
    uv_triangles = [quads_to_triangles(top_uq[top_region])]
    faces = [np.tile(base_face[top.parent_faces][top_region], 2)]
    edges, face_edges = top.edges, top.face_edges
    uv_edges, uv_face_edges = build_edges(uq)
    del uv_edges
    region_side = np.zeros(len(edges), int)
    other_side = np.zeros(len(edges), int)
    np.add.at(region_side, face_edges[base_region].ravel(), 1)
    np.add.at(other_side, face_edges[~base_region].ravel(), 1)
    split = (region_side > 0) & (other_side > 0)
    outer = np.nonzero(~base_region)[0]
    has_split = split[face_edges[outer]]
    plain = outer[~has_split.any(1)]
    triangles.append(quads_to_triangles(base_quads[plain]))
    uv_triangles.append(quads_to_triangles(uq[plain]))
    faces.append(np.tile(base_face[plain], 2))
    fan, uv_fan, fan_faces = [], [], []
    n_uv = len(uv)
    for f in outer[has_split.any(1)]:
        ring, uv_ring = [], []
        for k in range(4):
            ring.append(base_quads[f, k])
            uv_ring.append(uq[f, k])
            if split[face_edges[f, k]]:
                ring.append(base_count + face_edges[f, k])
                uv_ring.append(n_uv + uv_face_edges[f, k])
        k0 = next(i for i, v in enumerate(ring) if v >= base_count)
        ring, uv_ring = ring[k0:] + ring[:k0], uv_ring[k0:] + uv_ring[:k0]
        for i in range(1, len(ring) - 1):
            fan.append([ring[0], ring[i], ring[i + 1]])
            uv_fan.append([uv_ring[0], uv_ring[i], uv_ring[i + 1]])
            fan_faces.append(base_face[f])
    triangles.append(np.asarray(fan, dtype=np.int64).reshape(-1, 3))
    uv_triangles.append(np.asarray(uv_fan, dtype=np.int64).reshape(-1, 3))
    faces.append(np.asarray(fan_faces, dtype=np.int64))
    triangles = np.concatenate(triangles)
    used = np.asarray(subdivision.used_top_vertices)
    fine = np.searchsorted(used, triangles)
    if not np.array_equal(fine, subdivision.triangles):
        raise ValueError(
            "The UV triangles do not follow the triangles of the subdivision."
        )
    return FineUVs(np.concatenate(uv_triangles), top_uv, np.concatenate(faces))


@dataclasses.dataclass
class Seams:
    """The fine body with one copy of each seam vertex per side: ``source`` (N + D,) the fine
    vertex of each vertex (its own index for the first N), ``uvs`` (N + D, 2) in the glTF
    convention (v = 0 at the top), ``triangles`` (T, 3) on these vertices, and ``base_faces``
    (T,) the input quad of each triangle."""

    source: np.ndarray
    uvs: np.ndarray
    triangles: np.ndarray
    base_faces: np.ndarray

    @property
    def copies(self) -> int:
        return len(self.source) - int(self.source.max(initial=-1)) - 1


def seam_split(triangles: np.ndarray, fine: FineUVs, vertex_count: int) -> Seams:
    """
    The vertices of a fine mesh (``triangles`` (T, 3) on ``vertex_count`` vertices) split along
    the seams of its UVs (:func:`fine_uvs`): a vertex keeps the UV of its first corner, and each
    other UV of its corners gets a copy of the vertex after the last one.
    """
    triangles = np.asarray(triangles, dtype=np.int64)
    corners = triangles.ravel()
    uv_corner = fine.uvs[fine.uv_ids.ravel()]
    # a corner's UV, rounded so that equal UVs reached by other paths meet
    key = np.round(uv_corner * 2**24).astype(np.int64)
    pairs = np.stack([corners, key[:, 0], key[:, 1]], 1)
    unique, first, inverse = np.unique(
        pairs, axis=0, return_index=True, return_inverse=True
    )
    inverse = inverse.ravel()
    # the first UV of each vertex (by the order of the corners) stays on the vertex
    order = np.argsort(first, kind="stable")
    seen = np.zeros(vertex_count, bool)
    target = np.empty(len(unique), np.int64)
    copies = []
    for u in order:
        v = int(unique[u, 0])
        if not seen[v]:
            seen[v] = True
            target[u] = v
        else:
            target[u] = vertex_count + len(copies)
            copies.append(u)
    copies = np.asarray(copies, dtype=np.int64)
    source = np.concatenate([np.arange(vertex_count), unique[copies, 0]]).astype(
        np.int64
    )
    uvs = np.zeros((len(source), 2))
    uv_of_unique = np.zeros((len(unique), 2))
    uv_of_unique[inverse] = uv_corner
    uvs[target] = uv_of_unique
    uvs[:, 1] = 1.0 - uvs[:, 1]
    return Seams(
        source, uvs, target[inverse].reshape(-1, 3), np.asarray(fine.base_faces)
    )


def head_tile(model, seams: Seams, body_faces: np.ndarray) -> dict:
    """
    The head tile on the page: ``flags`` (N + D,) 1 for the vertices of the triangles on the
    head's island of the atlas, and for each topology of :data:`TOPOLOGIES` the tile's
    ``KHR_texture_transform`` (offset and scale, glTF convention), or None without a head island.
    ``body_faces`` maps the input quads of the fine body to the faces of ``model`` (an
    ``anny-quads`` model).
    """
    from opensculptboy.texture.tiles import _head_island, head_texture_transform

    found = _head_island(model, "anny")
    flags = np.zeros(len(seams.source), np.uint8)
    out = {"flags": flags, "transforms": {}}
    if found is None:
        return out
    on_island = found[2][np.asarray(body_faces)[seams.base_faces]]
    flags[seams.triangles[on_island].ravel()] = 1
    for topology in TOPOLOGIES:
        try:
            out["transforms"][topology] = head_texture_transform(model, topology)
        except (ValueError, KeyError, FileNotFoundError):
            out["transforms"][topology] = None
    return out


def closest_on_triangles(
    points: np.ndarray,
    vertices: np.ndarray,
    triangles: np.ndarray,
    candidates: int = 24,
):
    """
    The closest point of a triangle mesh to each of ``points`` (M, 3), among the ``candidates``
    triangles whose centroids lie nearest to the point: the triangle (M,) and the barycentric
    weights (M, 3) of the closest point on it.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int64)
    corners = np.asarray(vertices, dtype=np.float64)[tri]
    k = min(candidates, len(tri))
    near = cKDTree(corners.mean(1)).query(points, k=k)[1].reshape(len(points), k)
    A, B, C = (corners[near, c] for c in range(3))
    p = points[:, None]
    best = np.full(near.shape, np.inf)
    weights = np.zeros(near.shape + (3,))
    # the closest point on each edge, then the projection inside the triangle where it falls there
    for a, b, ia, ib in ((A, B, 0, 1), (B, C, 1, 2), (C, A, 2, 0)):
        e = b - a
        t = np.clip(
            ((p - a) * e).sum(-1) / np.maximum((e * e).sum(-1), 1e-30), 0.0, 1.0
        )
        d = (((a + t[..., None] * e) - p) ** 2).sum(-1)
        better = d < best
        best = np.where(better, d, best)
        w = np.zeros(near.shape + (3,))
        w[..., ia], w[..., ib] = 1.0 - t, t
        weights[better] = w[better]
    ab, ac, ap = B - A, C - A, p - A
    d00, d01, d11 = (ab * ab).sum(-1), (ab * ac).sum(-1), (ac * ac).sum(-1)
    d20, d21 = (ap * ab).sum(-1), (ap * ac).sum(-1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-30)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    inside = np.stack([1.0 - v - w, v, w], -1)
    plane = (
        (inside[..., :1] * A + inside[..., 1:2] * B + inside[..., 2:] * C - p) ** 2
    ).sum(-1)
    use = (inside >= 0).all(-1) & (plane < best)
    best = np.where(use, plane, best)
    weights[use] = inside[use]
    pick = np.argmin(best, axis=1)
    rows = np.arange(len(points))
    return near[rows, pick], weights[rows, pick]


def _match(
    page: np.ndarray,
    target: np.ndarray,
    page_shape: np.ndarray,
    target_shape: np.ndarray,
    triangles: np.ndarray,
    tol=1e-9,
):
    """The row of ``target`` (M, 3) equal to each row of ``page`` (P, 3) (random values through the
    same stencil), else the row of ``target_shape`` nearest to the row of ``page_shape`` (a body
    through both stencils). For the rows without an equal row: their indices, the three rows and
    barycentric weights of the closest point of the target's ``triangles`` on that body, and the
    move from the page's row to that point."""
    from scipy.spatial import cKDTree

    dist, rows = cKDTree(target).query(page, k=1)
    missed = np.nonzero(dist > tol)[0]
    corners = np.zeros((len(missed), 3), np.int64)
    weights = np.zeros((len(missed), 3))
    delta = np.zeros((len(missed), 3))
    if len(missed):
        rows[missed] = cKDTree(target_shape).query(page_shape[missed], k=1)[1]
        face, weights = closest_on_triangles(
            page_shape[missed], target_shape, triangles
        )
        corners = np.asarray(triangles, dtype=np.int64)[face]
        point = np.einsum("mj,mjc->mc", weights, target_shape[corners])
        delta = point - page_shape[missed]
    return dict(
        rows=rows.astype(np.uint32),
        missed=missed.astype(np.uint32),
        corners=corners.astype(np.uint32),
        weights=weights.astype(np.float32),
        delta=delta.astype(np.float32),
    )


def detail_rows(
    compact_quads: np.ndarray,
    body_count: int,
    compact: np.ndarray,
    template: np.ndarray,
    seed: int = 0,
) -> dict:
    """
    The rows of a card's detail levels behind the page's rows.

    ``compact_quads`` (Q, 4) are the page's body quads on its coarse vertices, the first
    ``body_count`` of which it subdivides; ``compact`` (C,) is the Anny vertex of each coarse
    vertex, and ``template`` (V, 3) Anny's template. The page's level ``l`` (1 or 2) is ``l``
    full levels of Catmull-Clark subdivision of its quads (``viewer/src/subdivision.ts``).
    Returns ``{"coarse": compact, "<topology>": {"1": rows, "2": rows, "fallback": {...}}}``:
    for each page row of a level, the row of the full level of that topology whose stencil is the
    same, or, where the topology's patches changed the stencil, the nearest row on the template.
    ``"<level>_missed"`` lists those page rows; ``"<level>_corners"`` and ``"<level>_weights"``
    give the closest point of the level's surface on the template to each of them, and
    ``"<level>_delta"`` the move from the page's template position to that point. The page shows
    such a row at its own position, moved by that move and by the blend of the corners' offsets.
    """
    from anny.utils.subdivision import catmull_clark

    from opensculptboy.detail.subdivision import level

    compact = np.asarray(compact, dtype=np.int64)
    rng = np.random.default_rng(seed)
    noise = rng.random((len(template), 3))
    ops, n, q = [], body_count, np.asarray(compact_quads, dtype=np.int64)
    for _ in range(2):
        lv = catmull_clark(n, q)
        ops.append(lv.operator)
        n, q = lv.operator.shape[0], lv.quads
    body = compact[:body_count]
    out = {"coarse": compact.astype(np.uint32)}
    for topology in TOPOLOGIES:
        try:
            levels = [level(topology, k) for k in (0, 1, 2)]
        except (ValueError, KeyError, FileNotFoundError):
            continue
        count = levels[0].vertex_count
        extra_noise = rng.random((count - len(template), 3))
        x_t = np.concatenate([noise, extra_noise])
        shape_t = _topology_template(topology, template, count)
        x_p, s_p = noise[body], template[body]
        entry, missed = {}, {}
        for k in (1, 2):
            x_p, s_p = ops[k - 1].apply(x_p), ops[k - 1].apply(s_p)
            x_t, shape_t = (
                levels[k].operators[k - 1].apply(x_t),
                levels[k].operators[k - 1].apply(shape_t),
            )
            q = levels[k].quads
            m = _match(
                x_p,
                x_t,
                s_p,
                shape_t,
                np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]),
            )
            entry[str(k)] = m["rows"]
            for name in ("missed", "corners", "weights", "delta"):
                entry[f"{k}_{name}"] = m[name]
            missed[str(k)] = int(len(m["missed"]))
        entry["fallback"] = missed
        out[topology] = entry
    return out


def _topology_template(topology: str, template: np.ndarray, count: int) -> np.ndarray:
    """Anny's template on the vertices of a topology: the new vertices of the sculpt topology at
    the blends of their anchors."""
    if count == len(template):
        return np.asarray(template, dtype=np.float64)
    from opensculptboy.topology.sculpt import sculpt_data

    return np.asarray(
        sculpt_data().positions(np.asarray(template, dtype=np.float64)),
        dtype=np.float64,
    )


def sculpt_patch(
    level_k: int,
    compact: np.ndarray,
    coarse_page: np.ndarray,
    fine_page: np.ndarray,
    hidden: np.ndarray,
) -> dict | None:
    """
    The faces that the patches of the ``sculpt`` topology add (:mod:`opensculptboy.topology`), on
    the full level ``level_k`` of the sculpt mesh, for the page to draw in place of Anny's faces
    that they replace.

    ``compact`` (C,) is the Anny vertex of each coarse vertex of the page, ``coarse_page`` (C, 3)
    the page's coarse vertices of Anny's default body, ``fine_page`` (V, 3) the page's drawn
    vertices of that body and ``hidden`` (V,) the drawn vertices that the page leaves out on a
    sculpt card. Returns, per drawn vertex of the patch (one copy per side of a UV seam), the row
    of the level (where a card's offsets of that level apply), the sparse weights of the page's
    coarse vertices that give its parametric position (``start``, ``cols``, ``values``), its UV
    (glTF convention) and the nearest drawn vertex of the page that is not hidden (whose skin
    weights and shading fields it takes), and the triangles; or None without a sculpt topology.
    """
    from anny.utils.subdivision import SparseOperator
    from scipy.spatial import cKDTree

    from opensculptboy.detail.subdivision import level, split_quads
    from opensculptboy.topology.sculpt import sculpt_data

    try:
        data = sculpt_data()
        lv = level("sculpt", level_k)
    except (ValueError, KeyError, FileNotFoundError):
        return None
    n, m = data.vertex_count, data.new_vertex_count
    kept = int(data.kept_faces().sum())
    patch = lv.base_faces >= kept
    quads, uv_quads = lv.quads[patch], lv.uv_quads[patch]
    # the parametric rows: the level's operators from the sculpt mesh, whose new vertices blend Anny's
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.ones(n)]
    for k in range(data.anchors.shape[1]):
        keep = data.weights[:, k] > 0
        rows.append(n + np.nonzero(keep)[0])
        cols.append(data.anchors[keep, k])
        vals.append(data.weights[keep, k])
    blend = SparseOperator(
        np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), (n + m, n)
    )
    # the drawn vertices: one per vertex and UV of the patch's corners
    tris = split_quads(quads)
    uv_tris = split_quads(uv_quads)
    pairs = np.stack([tris.ravel(), uv_tris.ravel()], 1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    vertex_rows = unique[:, 0]
    # the rows of the level from the sculpt mesh, composed from the last operator back
    op = lv.operators[-1].select_rows(vertex_rows)
    for earlier in reversed(lv.operators[:-1]):
        op = op @ earlier
    op = op @ blend
    of_anny = -np.ones(n, np.int64)
    of_anny[np.asarray(compact)] = np.arange(len(compact))
    page_cols = of_anny[op.cols]
    if (page_cols < 0).any():
        raise ValueError(
            "A vertex of the patch blends Anny vertices that the page lacks."
        )
    start = np.concatenate(
        [[0], np.cumsum(np.bincount(op.rows, minlength=len(unique)))]
    )
    position = np.zeros((len(unique), 3))
    np.add.at(
        position, op.rows, op.values[:, None] * np.asarray(coarse_page)[page_cols]
    )
    visible = np.nonzero(~np.asarray(hidden, bool))[0]
    near = visible[cKDTree(np.asarray(fine_page)[visible]).query(position)[1]]
    uv = np.asarray(lv.uvs)[unique[:, 1]].copy()
    uv[:, 1] = 1.0 - uv[:, 1]
    return {
        "rows": vertex_rows.astype(np.uint32),
        "start": start.astype(np.uint32),
        "cols": page_cols.astype(np.uint32),
        "values": op.values.astype(np.float32),
        "uvs": uv.astype(np.float32),
        "near": near.astype(np.uint32),
        "triangles": inverse.reshape(-1, 3).astype(np.uint32),
    }


def pocket(seams: Seams, body_faces: np.ndarray) -> np.ndarray:
    """(N + D,) the drawn vertices of the page whose every triangle lies on a face of Anny that the
    patches of the sculpt topology replace; the page leaves them out on a sculpt card."""
    from opensculptboy.topology.sculpt import sculpt_data

    flags = np.zeros(len(seams.source), np.uint8)
    try:
        data = sculpt_data()
    except (ValueError, KeyError, FileNotFoundError):
        return flags
    removed = np.zeros(
        max(int(np.max(body_faces)), int(data.removed_faces.max())) + 1, bool
    )
    removed[data.removed_faces] = True
    on = removed[np.asarray(body_faces)[seams.base_faces]]
    flags[np.unique(seams.triangles[on])] = 1
    flags[np.unique(seams.triangles[~on])] = 0
    return flags


# ------------------------------------------------------------------ the faces
# The page plays the facial actions as the exports write them (docs/viewer.md, "Faces"): Anny's 52
# ARKit actions with the mouth cavity kept behind the skin in each target (opensculptboy.export.mouth),
# and jawOpen as the turn of the anny-mouth rig's jaw with its corrective (opensculptboy.mouth.jaw).


@dataclasses.dataclass
class FaceTables:
    """The faces of the page on Anny's default body, in Anny's frame (metres)."""

    labels: list  # the facial actions, in the order of model.facial_action_labels
    rest: np.ndarray  # (V, 3) the default body at rest
    offsets: np.ndarray  # (A, V, 3) each action's target, the cavity behind the skin
    cleared: list  # the actions whose targets the cavity's correction changed
    rotation: np.ndarray  # (3, 3) the jaw's turn for jawOpen at weight 1
    pivot: np.ndarray  # (3,) the head of the jaw bone at rest
    jaw_weights: np.ndarray  # (V,) the weight of the jaw and the tongue chain
    tongue: np.ndarray  # rows of Anny's tongue
    # per action that moves the tongue: ("rigid", R (3, 3), t (3,)) when the tongue moves rigidly,
    # else ("follow", offsets of the tongue's rows (n, 3))
    tongue_motion: dict
    eyes: dict  # "L" and "R": the rows of each eyeball
    eye_turns: (
        dict  # per action that turns an eyeball: {"L": rotation vector, "R": ...}
    )
    model: object = None


def face_model():
    """The model of the page's faces: Anny's quad mesh on the ``anny-mouth`` rig with every
    facial action, in float64."""
    import torch

    from opensculptboy.mouth.rig import mouth_model

    return mouth_model(
        topology="anny-quads", facial_actions="all", phenotypes="all"
    ).to(dtype=torch.float64)


def face_tables(model=None) -> FaceTables:
    """The facial actions of the exports on Anny's default body, the jaw's turn for jawOpen,
    the motion of Anny's tongue in each action and the turn of each eyeball in the actions
    that turn it."""
    import torch
    from scipy.spatial.transform import Rotation

    from opensculptboy.export.mouth import clear_vertices, mouth_interior
    from opensculptboy.export.vrm_tables import eyeball_vertices
    from opensculptboy.mouth.jaw import fit_jaw_open, jaw_weights
    from opensculptboy.teeth.mesh import RIGID_SHARE, mouth_rows, rigid

    model = model or face_model()
    labels = list(model.facial_action_labels)
    dtype = model.template_vertices.dtype
    rows = torch.zeros((len(labels) + 1, len(labels)), dtype=dtype)
    rows[1:] = torch.eye(len(labels), dtype=dtype)
    with torch.no_grad():
        rest = model(facial_actions=rows)["rest_vertices"].double().cpu().numpy()
    base = rest[0]
    offsets = rest[1:] - base
    # the exports keep the cavity behind the skin in each target at full weight
    interior = mouth_interior(model)
    cleared = []
    if interior is not None:
        for a in range(len(labels)):
            state, count = clear_vertices(base + offsets[a], interior)
            if count:
                offsets[a] = state - base
                cleared.append(labels[a])
    turn = fit_jaw_open(model)
    _, tongue = mouth_rows(model)
    motion = {}
    for a, name in enumerate(labels):
        d = offsets[a][tongue]
        size = float(np.linalg.norm(d, axis=1).mean()) if len(tongue) else 0.0
        if size < 1e-7:
            continue
        p = base[tongue]
        R, t = rigid(p, p + d)
        residual = float(np.linalg.norm(p @ R.T + t - (p + d), axis=1).mean())
        motion[name] = (
            ("rigid", R, t) if residual <= RIGID_SHARE * size else ("follow", d)
        )
    eyes = eyeball_vertices(model)
    turns = {}
    for a, name in enumerate(labels):
        entry = {}
        for side, ids in eyes.items():
            d = offsets[a][ids]
            if np.abs(d).max() < 1e-7:
                continue
            R, _ = rigid(base[ids], base[ids] + d)
            entry[side] = Rotation.from_matrix(R).as_rotvec()
        if entry:
            turns[name] = entry
    return FaceTables(
        labels=labels,
        rest=base,
        offsets=offsets,
        cleared=cleared,
        rotation=turn.rotation,
        pivot=turn.pivot,
        jaw_weights=jaw_weights(model),
        tongue=tongue,
        tongue_motion=motion,
        eyes=eyes,
        eye_turns=turns,
        model=model,
    )


def jaw_pivot_regressor(compact, count: int = 24, bodies: int = 48, seed: int = 0):
    """
    The head of the jaw bone as a blend of the page's coarse vertices: the ``count`` vertices of
    ``compact`` (Anny vertices) nearest to it on the default body, with weights that sum to 1 and
    fit the head over random bodies (phenotypes, face shapes and local changes) best. Returns
    (positions in ``compact``, weights, the largest error over the bodies in metres).
    """
    import torch

    from opensculptboy.mouth.rig import mouth_model

    model = mouth_model(
        topology="anny-quads",
        phenotypes="all",
        face_shapes="all",
        local_changes="all",
    ).to(dtype=torch.float64)
    jaw = list(model.bone_labels).index("jaw")
    rng = np.random.default_rng(seed)
    labels = list(model.phenotype_labels)
    phen = torch.as_tensor(rng.random((bodies, len(labels))))
    face = {}
    for name in model.face_shape_labels:
        lo, hi = model.face_shape_ranges[name]
        face[name] = torch.as_tensor(
            rng.uniform(lo, hi, bodies) * (rng.random(bodies) < 0.3)
        )
    local = {
        name: torch.as_tensor((rng.random(bodies) - 0.5) * (rng.random(bodies) < 0.1))
        for name in model.local_change_labels
    }
    phen[0] = 0.5
    for v in list(face.values()) + list(local.values()):
        v[0] = 0.0
    with torch.no_grad():
        out = model(
            phenotype_kwargs={k: phen[:, i] for i, k in enumerate(labels)},
            face_shape_kwargs=face,
            local_changes_kwargs=local,
        )
    V = out["rest_vertices"].double().cpu().numpy()[:, np.asarray(compact)]
    heads = out["rest_bone_heads"].double().cpu().numpy()[:, jaw]
    near = np.argsort(np.linalg.norm(V[0] - heads[0], axis=1))[:count]
    # least squares of sum_i a_i v_i = p over the bodies, with sum_i a_i = 1 (a heavy row) and a
    # small pull toward equal weights
    A = V[:, near].transpose(0, 2, 1).reshape(-1, count)
    b = heads.reshape(-1)
    k = 1e-3
    A = np.concatenate([A, 1e3 * np.ones((1, count)), k * np.eye(count)])
    b = np.concatenate([b, [1e3], k * np.full(count, 1.0 / count)])
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    err = float(
        np.linalg.norm(np.einsum("k,bkd->bd", w, V[:, near]) - heads, axis=1).max()
    )
    return near.astype(np.int64), w, err

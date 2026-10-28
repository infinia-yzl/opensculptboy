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


def _match(
    page: np.ndarray,
    target: np.ndarray,
    page_shape: np.ndarray,
    target_shape: np.ndarray,
    tol=1e-9,
):
    """The row of ``target`` (M, 3) equal to each row of ``page`` (P, 3) (random values through the
    same stencil), else the row of ``target_shape`` nearest to the row of ``page_shape`` (a body
    through both stencils); and the count of the rows found by the shape."""
    from scipy.spatial import cKDTree

    dist, rows = cKDTree(target).query(page, k=1)
    missed = dist > tol
    if missed.any():
        rows[missed] = cKDTree(target_shape).query(page_shape[missed], k=1)[1]
    return rows.astype(np.uint32), int(missed.sum())


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
            entry[str(k)], missed[str(k)] = _match(x_p, x_t, s_p, shape_t)
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

# OpenSculptBoy
# Apache License, Version 2.0
"""
The meshes of the detail layer: Catmull-Clark subdivision of anny's quad mesh, with UVs and
skin weights carried to every level.

Level 0 is anny's own mesh (its 13,718 vertices and its triangles). Level ``n`` is ``n`` levels
of Catmull-Clark subdivision of the quad mesh of the same vertices (``topology="anny-quads"``),
built with the sparse operators of :mod:`anny.utils.subdivision` (Apache-2.0, in this
repository), so that any per-vertex quantity of level 0 (rest vertices, morph targets, skin
weights) maps to level ``n`` by the same linear map:

- positions and morph-target offsets: exactly, by the operator (subdivision is linear);
- skin weights: by the operator, then the strongest bones of each vertex
  (:meth:`DetailTopology.skin_weights`); the rows still sum to 1;
- UVs: face-varying and bilinear inside each face of the previous level (the edge points take
  the midpoint of the edge's UVs, the face points the mean of the face's UVs), so that the UV
  seams of anny's atlas stay seams and every level shares its layout.

The ``sculpt`` topology (:mod:`opensculptboy.topology`) adds triangles ``[a, b, c, c]`` where
its patches join anny's quads. Level 1 of a mesh with triangles uses the general Catmull-Clark
rules (:func:`catmull_clark_mixed`: a face point is the mean of the face's corners, and each
face of n corners gives n quads), so levels 1 and 2 are quad meshes again; a quad mesh takes
:func:`anny.utils.subdivision.catmull_clark` unchanged.

A :class:`DetailTopology` is one mesh that an exporter writes: a full level (``region="full"``)
or a level whose head and neck alone take the last level of subdivision (``region="head"``),
joined to the coarser rest of the body without cracks as in
:class:`anny.utils.subdivision.MixedSubdivision`. Each mesh vertex is a vertex of the full
level (``rows``), so the offsets of a level apply to both.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import torch

from anny.utils.subdivision import (
    HEAD_REGION_BONES,
    SparseOperator,
    build_edges,
    catmull_clark,
    faces_weighted_to_bones,
)

LEVELS = (0, 1, 2)
REGIONS = ("full", "head")
# Bones kept per vertex while the skin weights pass through the levels; anny uses up to 9.
KEEP_BONES = 12


def split_quads(quads: np.ndarray) -> np.ndarray:
    """Triangles of quads [a, b, c, d]: [a, b, d] and [b, c, d], one quad after the other.
    Level n >= 1 splits every quad along the diagonal between its two edge points."""
    quads = np.asarray(quads, dtype=np.int64)
    return np.stack([quads[:, [0, 1, 3]], quads[:, [1, 2, 3]]], 1).reshape(-1, 3)


def is_triangle(faces: np.ndarray) -> np.ndarray:
    """(F,) mask of the triangles ``[a, b, c, c]`` of an (F, 4) face array."""
    faces = np.asarray(faces)
    return faces[:, 2] == faces[:, 3]


def split_faces(faces: np.ndarray) -> np.ndarray:
    """Triangles of mixed faces: quads as :func:`split_quads`, triangles ``[a, b, c, c]`` as
    ``[a, b, c]``, face after face (the same as :func:`split_quads` without triangles)."""
    faces = np.asarray(faces, dtype=np.int64)
    tri = is_triangle(faces)
    if not tri.any():
        return split_quads(faces)
    parts = [[f[:3]] if t else [f[[0, 1, 3]], f[[1, 2, 3]]] for f, t in zip(faces, tri)]
    return np.asarray([x for p in parts for x in p], dtype=np.int64).reshape(-1, 3)


def mixed_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Unique edges of a mesh of quads and triangles ``[a, b, c, c]``: ``(E, 2)`` sorted pairs and
    ``(F, 4)`` the edge from corner k to the next corner, -1 in the last slot of a triangle
    (whose edges are a-b, b-c and c-a). Without triangles, as
    :func:`anny.utils.subdivision.build_edges`.
    """
    faces = np.asarray(faces, dtype=np.int64)
    tri = is_triangle(faces)
    if not tri.any():
        return build_edges(faces)
    nxt = np.roll(faces, -1, axis=1)
    nxt[tri, 2] = faces[tri, 0]
    valid = np.ones(faces.shape, bool)
    valid[tri, 3] = False
    pairs = np.stack([faces[valid], nxt[valid]], -1)
    edges, inverse = np.unique(np.sort(pairs, 1), axis=0, return_inverse=True)
    face_edges = -np.ones(faces.shape, np.int64)
    face_edges[valid] = inverse.reshape(-1)
    return edges, face_edges


@dataclasses.dataclass
class MixedLevel:
    """One level of Catmull-Clark subdivision of a mesh of quads and triangles."""

    operator: SparseOperator  # (V + E + F, V)
    quads: np.ndarray  # (Q, 4) faces of the subdivided mesh, all quads
    parent_faces: np.ndarray  # (Q,) the input face of each new face


def catmull_clark_mixed(num_vertices: int, faces: np.ndarray):
    """
    One level of Catmull-Clark subdivision of a mesh of quads and triangles ``[a, b, c, c]``.
    A mesh of quads alone takes :func:`anny.utils.subdivision.catmull_clark` itself.

    The rules are those of ``catmull_clark`` for faces of n corners: the face point is the
    mean of the n corners, an inner edge point the mean of the edge's midpoint and of its two
    face points, an inner vertex point ``(F + 2 R + (n - 3) P) / n``, with the same boundary
    rules. New vertices come in the same order (vertex, edge, face points); the new faces come
    by corner (the corner-0 quad of every face, then corner 1, ...), a triangle giving three.
    """
    faces = np.asarray(faces, dtype=np.int64)
    if not is_triangle(faces).any():
        return catmull_clark(num_vertices, faces)
    return _catmull_clark_general(num_vertices, faces)


def _catmull_clark_general(num_vertices: int, faces: np.ndarray) -> MixedLevel:
    """The rules of :func:`catmull_clark_mixed` for any mix of quads and triangles (on quads
    alone, the operator and faces of ``catmull_clark``)."""
    tri = is_triangle(faces)
    n_vertices, n_faces = int(num_vertices), len(faces)
    corners = np.where(tri, 3, 4)
    edges, face_edges = mixed_edges(faces)
    n_edges = len(edges)
    valid = face_edges >= 0  # (F, 4) the real corners
    face_ids = np.nonzero(valid)[0]
    corner_vertices = faces[valid]  # a triangle's repeated corner left out
    inv = 1.0 / corners
    rows, cols, vals = [], [], []

    def add(r, c, v):
        r, c, v = np.broadcast_arrays(r, c, v)
        rows.append(r.ravel())
        cols.append(c.ravel())
        vals.append(np.asarray(v, dtype=np.float64).ravel())

    face_rows = n_vertices + n_edges + np.arange(n_faces)
    edge_rows = n_vertices + np.arange(n_edges)
    add(face_rows[face_ids], corner_vertices, inv[face_ids])

    faces_per_edge = np.bincount(face_edges[valid], minlength=n_edges)
    boundary_edge = faces_per_edge == 1
    add(edge_rows[boundary_edge, None], edges[boundary_edge], 0.5)
    inner = ~boundary_edge
    add(edge_rows[inner, None], edges[inner], 0.25)
    # each inner edge of a face takes 1/4 of the face point
    # (F, 4) share of each corner in its face point; a triangle's repeated corner takes 0
    corner_share = np.where(valid, inv[:, None], 0.0)
    fe_face, fe_k = np.nonzero(valid & inner[np.maximum(face_edges, 0)])
    add(
        edge_rows[face_edges[fe_face, fe_k]][:, None],
        faces[fe_face],
        0.25 * corner_share[fe_face],
    )

    valence = np.bincount(edges.ravel(), minlength=n_vertices).astype(np.float64)
    faces_per_vertex = np.bincount(corner_vertices, minlength=n_vertices).astype(
        np.float64
    )
    boundary_edges = edges[boundary_edge]
    boundary_count = np.bincount(boundary_edges.ravel(), minlength=n_vertices)
    is_boundary = boundary_count > 0
    smooth_boundary = is_boundary & (boundary_count == 2)
    corner_vertex = is_boundary & (boundary_count != 2)
    interior = ~is_boundary & (faces_per_vertex > 0)
    n = np.maximum(valence, 1)

    iv = np.nonzero(interior)[0]
    add(iv, iv, (n[iv] - 3) / n[iv])
    f_ids, k = np.nonzero(valid & interior[faces])
    v = faces[f_ids, k]
    add(
        v[:, None],
        faces[f_ids],
        (1.0 / n[v] / faces_per_vertex[v])[:, None] * corner_share[f_ids],
    )
    for side in (0, 1):
        e_ids = np.nonzero(interior[edges[:, side]])[0]
        v = edges[e_ids, side]
        add(v[:, None], edges[e_ids], (2.0 / n[v] / n[v] * 0.5)[:, None])

    sb = np.nonzero(smooth_boundary)[0]
    add(sb, sb, 6.0 / 8.0)
    for side in (0, 1):
        v = boundary_edges[:, side]
        keep = smooth_boundary[v]
        add(v[keep], boundary_edges[keep, 1 - side], 1.0 / 8.0)
    cv = np.nonzero(corner_vertex)[0]
    add(cv, cv, 1.0)
    lonely = np.nonzero(faces_per_vertex == 0)[0]
    add(lonely, lonely, 1.0)

    operator = SparseOperator(
        np.concatenate(rows),
        np.concatenate(cols),
        np.concatenate(vals),
        (n_vertices + n_edges + n_faces, n_vertices),
    )
    quads, parents = _corner_quads(faces, face_edges, n_vertices, n_edges)
    return MixedLevel(operator, quads, parents)


def _corner_quads(faces, face_edges, n_points, n_edges):
    """The quads [corner k, edge k, face point, edge k-1] of every face, by corner; the
    previous edge of a triangle's corner 0 is its edge c-a (slot 2)."""
    tri = is_triangle(faces)
    corners = np.where(tri, 3, 4)
    e = n_points + face_edges
    f = n_points + n_edges + np.arange(len(faces))
    quads, parents = [], []
    for k in range(4):
        sel = np.nonzero(corners > k)[0]
        prev = np.where(k == 0, corners[sel] - 1, k - 1)
        quads.append(
            np.stack(
                [faces[sel, k], e[sel, k], f[sel], e[sel, prev]],
                1,
            )
        )
        parents.append(sel)
    return np.concatenate(quads), np.concatenate(parents)


def subdivide_uvs(uv_quads: np.ndarray, uvs: np.ndarray):
    """
    One level of face-varying bilinear subdivision of a UV quad mesh, with the face order of
    :func:`anny.utils.subdivision.catmull_clark` (and of :func:`catmull_clark_mixed` for UV
    faces with triangles ``[a, b, c, c]``). Returns the new UV quads and UVs.
    """
    uv_quads = np.asarray(uv_quads, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float64)
    if is_triangle(uv_quads).any():
        edges, face_edges = mixed_edges(uv_quads)
        tri = is_triangle(uv_quads)
        corners = np.where(tri, 3, 4)
        centres = (
            uvs[uv_quads].sum(1) - np.where(tri, 1, 0)[:, None] * uvs[uv_quads[:, 3]]
        )
        new_uvs = np.concatenate(
            [
                uvs,
                0.5 * (uvs[edges[:, 0]] + uvs[edges[:, 1]]),
                centres / corners[:, None],
            ]
        )
        quads, _ = _corner_quads(uv_quads, face_edges, len(uvs), len(edges))
        return quads, new_uvs
    edges, face_edges = build_edges(uv_quads)
    n, n_edges, n_faces = len(uvs), len(edges), len(uv_quads)
    new_uvs = np.concatenate(
        [uvs, 0.5 * (uvs[edges[:, 0]] + uvs[edges[:, 1]]), uvs[uv_quads].mean(1)]
    )
    e = n + face_edges
    f = n + n_edges + np.arange(n_faces)
    u0, u1, u2, u3 = uv_quads.T
    new_quads = np.concatenate(
        [
            np.stack([u0, e[:, 0], f, e[:, 3]], 1),
            np.stack([u1, e[:, 1], f, e[:, 0]], 1),
            np.stack([u2, e[:, 2], f, e[:, 1]], 1),
            np.stack([u3, e[:, 3], f, e[:, 2]], 1),
        ]
    )
    return new_quads, new_uvs


@dataclasses.dataclass
class Level:
    """One full level of subdivision of a quad mesh."""

    subdiv: int
    operators: list  # SparseOperator per step, level 0 to this level
    quads: np.ndarray  # (F, 4) faces of this level
    uv_quads: np.ndarray | None  # (F, 4) UV indices of each face
    uvs: np.ndarray | None  # (U, 2) UVs, OBJ convention (v = 0 at the bottom)
    base_faces: np.ndarray  # (F,) the level-0 face of each face
    vertex_count: int


def _quad_spec(topology: str) -> str:
    return topology if topology.endswith("-quads") else f"{topology}-quads"


def _data_key(topology: str):
    """The digest of the current sculpt topology for the sculpt topologies (so that the cached
    levels follow :func:`opensculptboy.topology.use_sculpt_data`), None for anny's."""
    from opensculptboy.topology.sculpt import is_sculpt, sculpt_data

    return sculpt_data().digest if is_sculpt(topology) else None


@functools.lru_cache(maxsize=4)
def _quad_mesh(topology: str, data_key=None):
    """(vertex count, quads, UV quads, UVs) of the quad version of a topology"""
    from opensculptboy.topology.sculpt import build_model

    del data_key
    spec = _quad_spec(topology)
    model = build_model(
        topology=spec, local_changes="none", facial_actions="none", face_shapes="none"
    )
    faces = model.faces.cpu().numpy()
    if faces.ndim != 2 or faces.shape[1] != 4:
        raise ValueError(
            f"The detail layer needs a quad mesh; topology {spec!r} has faces with "
            f"{faces.shape[-1]} corners."
        )
    uv_faces = model.face_texture_coordinate_indices
    uvs = model.texture_coordinates
    if uv_faces is not None and uvs is not None:
        uv_faces = uv_faces.cpu().numpy().astype(np.int64)
        uvs = uvs.detach().cpu().double().numpy()
    else:
        uv_faces = uvs = None
    return len(model.template_vertices), faces.astype(np.int64), uv_faces, uvs


def level(topology: str, subdiv: int) -> Level:
    """The full level ``subdiv`` of the quad mesh of ``topology`` (cached). Level 0 of the
    ``sculpt`` topology may hold triangles ``[a, b, c, c]``; levels 1 and 2 are quads."""
    if subdiv not in LEVELS:
        raise ValueError(f"Detail levels are {LEVELS}; got {subdiv!r}.")
    return _level(_quad_spec(topology), subdiv, _data_key(topology))


@functools.lru_cache(maxsize=8)
def _level(topology: str, subdiv: int, data_key) -> Level:
    if subdiv == 0:
        count, quads, uv_quads, uvs = _quad_mesh(topology, data_key)
        return Level(0, [], quads, uv_quads, uvs, np.arange(len(quads)), count)
    previous = _level(topology, subdiv - 1, data_key)
    step = catmull_clark_mixed(previous.vertex_count, previous.quads)
    uv_quads, uvs = (None, None)
    if previous.uv_quads is not None:
        uv_quads, uvs = subdivide_uvs(previous.uv_quads, previous.uvs)
    return Level(
        subdiv,
        previous.operators + [step.operator],
        step.quads,
        uv_quads,
        uvs,
        previous.base_faces[step.parent_faces],
        step.operator.shape[0],
    )


@dataclasses.dataclass
class DetailTopology:
    """
    One mesh of the detail layer, as an exporter writes it.

    ``operators`` map a per-vertex quantity of level 0 to the mesh's vertices, applied in turn
    (none at level 0); ``rows`` gives the vertex of the full level ``subdiv`` behind each mesh
    vertex, where the offsets of that level apply. ``triangles`` index the mesh's vertices,
    ``uv_triangles`` index ``uvs`` (OBJ convention, v = 0 at the bottom).
    """

    subdiv: int
    region: str
    operators: list
    rows: np.ndarray  # (N,)
    triangles: np.ndarray  # (T, 3)
    uv_triangles: np.ndarray | None  # (T, 3)
    uvs: np.ndarray | None  # (U, 2)

    @property
    def vertex_count(self) -> int:
        return len(self.rows)

    def apply(self, x):
        """Map a level-0 quantity of shape (V0, ...) (NumPy or PyTorch) to (N, ...)."""
        for operator in self.operators:
            x = operator.apply(x)
        return x

    def apply_batch(self, x):
        """Map a batch (B, V0, D) (NumPy or PyTorch) to (B, N, D)."""
        if not self.operators:
            return x
        b, v, d = x.shape
        if isinstance(x, torch.Tensor):
            flat = x.permute(1, 0, 2).reshape(v, b * d)
            return self.apply(flat).reshape(-1, b, d).permute(1, 0, 2)
        flat = np.asarray(x).transpose(1, 0, 2).reshape(v, b * d)
        return self.apply(flat).reshape(-1, b, d).transpose(1, 0, 2)

    def skin_weights(self, weights, indices, keep: int = KEEP_BONES):
        """
        Skin weights (N, keep) and bone indices (N, keep) of the mesh from the (V0, K) weights
        of level 0, strongest first. Each level keeps the ``keep`` strongest bones of a vertex
        and renormalises them, which leaves out weights far below the 4 or 8 a file keeps.
        """
        weights = np.asarray(weights, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
        v, k = weights.shape
        bones = int(indices.max(initial=0)) + 1
        matrix = SparseOperator(
            np.repeat(np.arange(v), k), indices.ravel(), weights.ravel(), (v, bones)
        )
        for operator in self.operators:
            matrix = _strongest(operator @ matrix, keep)
        return _dense_rows(_strongest(matrix, keep), keep)


def _strongest(matrix: SparseOperator, keep: int) -> SparseOperator:
    """The ``keep`` largest entries of each row of a sparse matrix, renormalised to sum 1."""
    order = np.lexsort((-matrix.values, matrix.rows))
    rows, cols, values = matrix.rows[order], matrix.cols[order], matrix.values[order]
    starts = np.searchsorted(rows, rows, side="left")
    rank = np.arange(len(rows)) - starts
    kept = rank < keep
    rows, cols, values = rows[kept], cols[kept], values[kept]
    sums = np.bincount(rows, weights=values, minlength=matrix.shape[0])
    values = values / np.where(sums[rows] > 0, sums[rows], 1.0)
    return SparseOperator(rows, cols, values, matrix.shape)


def _dense_rows(matrix: SparseOperator, keep: int):
    """(N, keep) values and columns of a sparse matrix, each row sorted by decreasing value"""
    order = np.lexsort((-matrix.values, matrix.rows))
    rows, cols, values = matrix.rows[order], matrix.cols[order], matrix.values[order]
    rank = np.arange(len(rows)) - np.searchsorted(rows, rows, side="left")
    width = min(keep, int(rank.max(initial=0)) + 1)
    weights = np.zeros((matrix.shape[0], width))
    indices = np.zeros((matrix.shape[0], width), dtype=np.int64)
    weights[rows, rank] = values
    indices[rows, rank] = cols
    return weights, indices


def _anny_triangles(model):
    from opensculptboy.export.body import triangulated_faces

    return triangulated_faces(model)


def detail_topology(
    model, subdiv: int, region: str = "full", topology: str = "anny"
) -> DetailTopology:
    """
    The mesh of level ``subdiv`` of the detail layer for an Anny model of ``topology``.

    Level 0 is the model's own mesh with its triangles. ``region="head"`` gives the head and
    the neck (the faces weighted to :data:`~anny.utils.subdivision.HEAD_REGION_BONES`) level
    ``subdiv`` and the rest of the body level ``subdiv - 1``, joined without cracks: the coarse
    faces along the border take the edge points of the finer side and become fans.
    """
    if region not in REGIONS:
        raise ValueError(f"Unknown detail region {region!r}; use one of {REGIONS}.")
    full = level(topology, subdiv)
    base_count = level(topology, 0).vertex_count
    if base_count != len(model.template_vertices):
        raise ValueError(
            f"The quad mesh of {topology!r} has {base_count} vertices and the model "
            f"{len(model.template_vertices)}."
        )
    if subdiv == 0:
        if region != "full":
            raise ValueError("Level 0 has no region; it is anny's own mesh.")
        triangles, uv_triangles = _anny_triangles(model)
        uvs = model.texture_coordinates
        uvs = None if uvs is None else uvs.detach().cpu().double().numpy()
        return DetailTopology(
            0,
            "full",
            [],
            np.arange(full.vertex_count),
            np.asarray(triangles, dtype=np.int64),
            None if uv_triangles is None else np.asarray(uv_triangles, dtype=np.int64),
            uvs,
        )
    if region == "full":
        return DetailTopology(
            subdiv,
            "full",
            list(full.operators),
            np.arange(full.vertex_count),
            split_quads(full.quads),
            None if full.uv_quads is None else split_quads(full.uv_quads),
            full.uvs,
        )
    return _head_topology(model, topology, subdiv)


def _head_topology(model, topology: str, subdiv: int) -> DetailTopology:
    coarse = level(topology, subdiv - 1)
    fine = level(topology, subdiv)
    base = level(topology, 0)
    region0 = faces_weighted_to_bones(model, base.quads, HEAD_REGION_BONES)
    region = region0[coarse.base_faces]
    fine_region = region0[fine.base_faces]
    n = coarse.vertex_count
    edges, face_edges = mixed_edges(coarse.quads)
    corners = np.where(is_triangle(coarse.quads), 3, 4)
    has_uv = coarse.uv_quads is not None
    if has_uv:
        uv_edges, uv_face_edges = mixed_edges(coarse.uv_quads)
        n_uv = len(coarse.uvs)

    triangles = [split_quads(fine.quads[fine_region])]
    uv_triangles = [split_quads(fine.uv_quads[fine_region])] if has_uv else []
    region_side = np.zeros(len(edges), int)
    other_side = np.zeros(len(edges), int)
    region_edges = face_edges[region]
    other_edges = face_edges[~region]
    np.add.at(region_side, region_edges[region_edges >= 0], 1)
    np.add.at(other_side, other_edges[other_edges >= 0], 1)
    split = (region_side > 0) & (other_side > 0)
    outer = np.nonzero(~region)[0]
    has_split = split[face_edges[outer]] & (face_edges[outer] >= 0)
    plain = outer[~has_split.any(1)]
    triangles.append(split_faces(coarse.quads[plain]))
    if has_uv:
        uv_triangles.append(split_faces(coarse.uv_quads[plain]))
    fan, uv_fan = [], []
    for f in outer[has_split.any(1)]:
        ring, uv_ring = [], []
        for k in range(corners[f]):
            ring.append(coarse.quads[f, k])
            if has_uv:
                uv_ring.append(coarse.uv_quads[f, k])
            if split[face_edges[f, k]]:
                ring.append(n + face_edges[f, k])
                if has_uv:
                    uv_ring.append(n_uv + uv_face_edges[f, k])
        # a fan from the first inserted edge point keeps the triangles well shaped
        k0 = next(i for i, v in enumerate(ring) if v >= n)
        ring = ring[k0:] + ring[:k0]
        if has_uv:
            uv_ring = uv_ring[k0:] + uv_ring[:k0]
        for i in range(1, len(ring) - 1):
            fan.append([ring[0], ring[i], ring[i + 1]])
            if has_uv:
                uv_fan.append([uv_ring[0], uv_ring[i], uv_ring[i + 1]])
    triangles.append(np.asarray(fan, dtype=np.int64).reshape(-1, 3))
    triangles = np.concatenate(triangles)
    used = np.unique(triangles)
    remap = -np.ones(fine.vertex_count, np.int64)
    remap[used] = np.arange(len(used))
    uvs = uv_tris = None
    if has_uv:
        uv_triangles.append(np.asarray(uv_fan, dtype=np.int64).reshape(-1, 3))
        uv_tris = np.concatenate(uv_triangles)
        uv_used = np.unique(uv_tris)
        uv_remap = -np.ones(len(fine.uvs), np.int64)
        uv_remap[uv_used] = np.arange(len(uv_used))
        uvs, uv_tris = fine.uvs[uv_used], uv_remap[uv_tris]
    operators = list(fine.operators[:-1]) + [fine.operators[-1].select_rows(used)]
    return DetailTopology(
        subdiv, "head", operators, used, remap[triangles], uv_tris, uvs
    )

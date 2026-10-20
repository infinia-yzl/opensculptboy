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


def subdivide_uvs(uv_quads: np.ndarray, uvs: np.ndarray):
    """
    One level of face-varying bilinear subdivision of a UV quad mesh, with the face order of
    :func:`anny.utils.subdivision.catmull_clark`. Returns the new UV quads and UVs.
    """
    uv_quads = np.asarray(uv_quads, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float64)
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


@functools.lru_cache(maxsize=4)
def _quad_mesh(topology: str):
    """(vertex count, quads, UV quads, UVs) of the quad version of a topology"""
    import anny

    spec = topology if topology.endswith("-quads") else f"{topology}-quads"
    model = anny.Anny(
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


@functools.lru_cache(maxsize=8)
def level(topology: str, subdiv: int) -> Level:
    """The full level ``subdiv`` of the quad mesh of ``topology`` (cached)."""
    if subdiv not in LEVELS:
        raise ValueError(f"Detail levels are {LEVELS}; got {subdiv!r}.")
    if subdiv == 0:
        count, quads, uv_quads, uvs = _quad_mesh(topology)
        return Level(0, [], quads, uv_quads, uvs, np.arange(len(quads)), count)
    previous = level(topology, subdiv - 1)
    step = catmull_clark(previous.vertex_count, previous.quads)
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
    edges, face_edges = build_edges(coarse.quads)
    has_uv = coarse.uv_quads is not None
    if has_uv:
        uv_edges, uv_face_edges = build_edges(coarse.uv_quads)
        n_uv = len(coarse.uvs)

    triangles = [split_quads(fine.quads[fine_region])]
    uv_triangles = [split_quads(fine.uv_quads[fine_region])] if has_uv else []
    region_side = np.zeros(len(edges), int)
    other_side = np.zeros(len(edges), int)
    np.add.at(region_side, face_edges[region].ravel(), 1)
    np.add.at(other_side, face_edges[~region].ravel(), 1)
    split = (region_side > 0) & (other_side > 0)
    outer = np.nonzero(~region)[0]
    has_split = split[face_edges[outer]]
    plain = outer[~has_split.any(1)]
    triangles.append(split_quads(coarse.quads[plain]))
    if has_uv:
        uv_triangles.append(split_quads(coarse.uv_quads[plain]))
    fan, uv_fan = [], []
    for f in outer[has_split.any(1)]:
        ring, uv_ring = [], []
        for k in range(4):
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

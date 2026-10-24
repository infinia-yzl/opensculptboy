# OpenSculptBoy
# Apache License, Version 2.0
"""
Binding a garment to a body, and evaluating the binding on another body.

A **surface** binding stores, for each garment vertex, a triangle of Anny's template mesh (its
three corners), the barycentric coordinates of the bound point and the vertex's offset from
that point in the point's frame (:func:`opensculptboy.wearables.geometry.frames`: the tangent,
the bitangent and the blended vertex normal), divided by the frame's scale (the blended
length of the corners' edges). On another body the same point and frame give the vertex back,
so the offset turns and stretches with the skin; on the binding body the garment returns to
its own vertices. A **rigid** binding stores the vertices in the rest frame of one bone, for
pieces that move as one (a circlet, a buckle, a staff).
"""

from __future__ import annotations

import numpy as np

from opensculptboy.wearables.format import Binding, Garment
from opensculptboy.wearables.geometry import Surface, frames, unit_rows


class BodySurface:
    """
    The skin of a body on the rows of Anny's template mesh: rest vertices, the triangles that
    garments bind to, vertex normals and scales, the bone rest poses and the skin weights.

    Args:
        vertices: (N, 3) the rest vertices of Anny's rows (the first ``N`` rows of the model;
            the sculpt topology keeps anny's rows first).
        triangles: (T, 3) the triangles of the skin that garments bind to.
        weights, indices: (N, K) skin weights and bone indices of the rows.
        bone_poses: (J, 4, 4) the bones' rest world matrices.
        bone_labels: the bone names.
    """

    def __init__(self, vertices, triangles, weights, indices, bone_poses, bone_labels):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.bone_poses = np.asarray(bone_poses, dtype=np.float64)
        self.bone_labels = list(bone_labels)
        self._surface = None

    @property
    def surface(self) -> Surface:
        if self._surface is None:
            self._surface = Surface(self.vertices, self.triangles)
        return self._surface

    @property
    def normals(self) -> np.ndarray:
        return self.surface.normals

    @property
    def scales(self) -> np.ndarray:
        return self.surface.scales

    def with_vertices(self, vertices) -> "BodySurface":
        """The same skin with other rest vertices (the bones stay)."""
        return BodySurface(
            vertices,
            self.triangles,
            self.weights,
            self.indices,
            self.bone_poses,
            self.bone_labels,
        )


def bind_surface(
    garment: Garment,
    body: BodySurface,
    body_kind: str = "anny-template",
    corners: np.ndarray | None = None,
    barycentric: np.ndarray | None = None,
) -> Binding:
    """
    The surface binding of a garment on a body. By default each vertex binds to the closest
    point of the body; ``corners`` and ``barycentric`` (V, 3) give the bound points instead
    (for example points found through a fit's correspondence).
    """
    if corners is None or barycentric is None:
        found = body.surface.closest(garment.vertices)
        corners = body.triangles[found.triangle]
        barycentric = found.barycentric
    corners = np.asarray(corners, dtype=np.int64)
    barycentric = np.asarray(barycentric, dtype=np.float64)
    point, axes, scale = frames(
        body.vertices, body.normals, body.scales, corners, barycentric
    )
    scale = np.maximum(scale, 1e-9)
    offset = np.einsum("nij,nj->ni", axes, garment.vertices - point) / scale[:, None]
    normal = np.einsum("nij,nj->ni", axes, garment.normals)
    return Binding(
        method="surface",
        body=body_kind,
        body_vertices=body.vertices.copy(),
        corners=corners,
        barycentric=barycentric,
        offset=offset,
        normal=normal,
    )


def bind_rigid(
    garment: Garment, body: BodySurface, bone: str, body_kind: str = "anny-template"
) -> Binding:
    """The rigid binding of a garment to one bone of the body."""
    if bone not in body.bone_labels:
        raise ValueError(f"Unknown bone {bone!r}.")
    B = body.bone_poses[body.bone_labels.index(bone)]
    R, t = B[:3, :3], B[:3, 3]
    return Binding(
        method="rigid",
        body=body_kind,
        body_vertices=body.vertices.copy(),
        bone=bone,
        local=(garment.vertices - t) @ R,
        normal=garment.normals @ R,
    )


def evaluate(binding: Binding, body: BodySurface) -> tuple[np.ndarray, np.ndarray]:
    """The garment's rest vertices and unit normals (V, 3) on a body."""
    if binding.method == "rigid":
        B = body.bone_poses[body.bone_labels.index(binding.bone)]
        R, t = B[:3, :3], B[:3, 3]
        return binding.local @ R.T + t, unit_rows(binding.normal @ R.T)
    point, axes, scale = frames(
        body.vertices, body.normals, body.scales, binding.corners, binding.barycentric
    )
    vertices = point + scale[:, None] * np.einsum("nij,ni->nj", axes, binding.offset)
    normals = unit_rows(np.einsum("nij,ni->nj", axes, binding.normal))
    return vertices, normals


def body_weights(
    binding: Binding, body: BodySurface, max_influences: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """
    The skin weights of the garment's vertices: the body's weights blended at the bound
    points (surface), or the one bone (rigid); the strongest ``max_influences`` bones of each
    vertex, renormalised. Returns (V, k) float weights and int bone indices.
    """
    count = binding.vertex_count
    if binding.method == "rigid":
        bone = body.bone_labels.index(binding.bone)
        weights = np.zeros((count, max_influences))
        weights[:, 0] = 1.0
        indices = np.zeros((count, max_influences), np.int64)
        indices[:, 0] = bone
        return weights, indices
    J = len(body.bone_labels)
    dense = np.zeros((count, J))
    rows = np.arange(count)
    for c in range(3):
        w = body.weights[binding.corners[:, c]] * binding.barycentric[:, c, None]
        for k in range(body.weights.shape[1]):
            np.add.at(dense, (rows, body.indices[binding.corners[:, c], k]), w[:, k])
    order = np.argsort(-dense, axis=1, kind="stable")[:, :max_influences]
    weights = np.take_along_axis(dense, order, axis=1)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    indices = np.where(weights > 0, order, 0)
    return weights, indices

# OpenSculptBoy
# Apache License, Version 2.0
"""
MikkTSpace tangents in NumPy.

glTF 2.0 asks for MikkTSpace tangents when a file gives none, and Blender computes MikkTSpace
for every normal map; a normal map baked in the same tangent frames reads back the same in
both. :func:`mikktspace` follows the reference implementation of Morten S. Mikkelsen
(``mikktspace.c``, zlib licence; this is an independent port of its algorithm, not of its
code):

1. Vertices with the same position, normal and texture coordinate are one vertex.
2. Each triangle gives its tangent ``vOs`` (the derivative of the position along u) and its
   bitangent ``vOt``, unit length, flipped by the sign of the triangle's signed UV area; that
   sign is the triangle's orientation.
3. The corners of a vertex form groups: triangles that share the vertex and an edge, with
   the same winding and the same orientation, join one group (the fan around the vertex,
   split where the UV mapping mirrors).
4. The tangent of a group is the sum of the triangles' tangents, each projected onto the
   plane of the vertex normal and normalised, weighted by the angle of the triangle at the
   vertex (measured in the same plane), and normalised. The sign is the group's orientation.

The reference implementation compares the tangents within a group against an angular
threshold of 180 degrees, so that every group forms one subgroup; this port does the same.
Triangles of zero UV area take the orientation of the earliest-created group of a neighbour
(the reference creates its groups in the order of their first corner) and add their raw
tangents, and a group without a tangent takes (1, 0, 0). Degenerate triangles (two corners on
one vertex) add nothing. On a game character's head of 16,232 triangles, the tangents agree
with Blender's to 0.03 degrees (its float32 rounding) and the signs exactly, except next to
the 358 triangles of zero UV area, where Blender's own port is not consistent even within a
triangle; anny's meshes have no such triangles.

The texture coordinates are in the OBJ convention (v = 0 at the bottom), as Blender holds
them; ``w`` then gives the bitangent ``w * cross(normal, tangent)`` along increasing v, which
is up in the image, as glTF's normal textures expect. The tangent itself does not depend on
the direction of v.
"""

from __future__ import annotations

import numpy as np


def _normalise(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(length > 1e-30, v / np.where(length > 1e-30, length, 1.0), v)


def _weld(positions, normals, uvs) -> np.ndarray:
    """The index of each vertex among the vertices of the same position, normal and UV."""
    key = np.concatenate(
        [
            np.asarray(positions, np.float32),
            np.asarray(normals, np.float32),
            np.asarray(uvs, np.float32),
        ],
        axis=1,
    )
    key = np.ascontiguousarray(key).view(np.dtype((np.void, key.dtype.itemsize * 8)))
    _, first, inverse = np.unique(key.ravel(), return_index=True, return_inverse=True)
    return first[inverse.ravel()]


def _components(count: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Connected components of an undirected graph as labels 0..K-1 (NumPy alone: the
    exporters do not need SciPy). The fans around a vertex are short, so the smallest label
    spreads over each component in a few rounds."""
    label = np.arange(count)
    while True:
        low = np.minimum(label[src], label[dst])
        new = label.copy()
        np.minimum.at(new, src, low)
        np.minimum.at(new, dst, low)
        new = new[new]  # pointer jumping
        if np.array_equal(new, label):
            break
        label = new
    _, labels = np.unique(label, return_inverse=True)
    return labels.ravel()


def corner_tangents(
    positions: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    MikkTSpace tangents of each triangle corner: (T, 3, 4) tangents with the sign in ``w``,
    and the group (T, 3) of each corner (corners of one group have one tangent).

    Args:
        positions, normals: (N, 3) vertex positions and unit vertex normals.
        uvs: (N, 2) texture coordinates, OBJ convention (v = 0 at the bottom).
        triangles: (T, 3) vertex indices, counter-clockwise.
    """
    P = np.asarray(positions, dtype=np.float64)
    N = _normalise(np.asarray(normals, dtype=np.float64))
    UV = np.asarray(uvs, dtype=np.float64)
    F = np.asarray(triangles, dtype=np.int64)
    welded = _weld(P, N, UV)
    W = welded[F]  # (T, 3) welded vertex of each corner
    T = len(F)

    # 2. per-triangle tangent and bitangent, oriented by the signed UV area
    p0, p1, p2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    t0, t1, t2 = UV[F[:, 0]], UV[F[:, 1]], UV[F[:, 2]]
    d1, d2 = p1 - p0, p2 - p0
    t21, t31 = t1 - t0, t2 - t0
    area = t21[:, 0] * t31[:, 1] - t21[:, 1] * t31[:, 0]
    vos = t31[:, 1:2] * d1 - t21[:, 1:2] * d2
    vot = -t31[:, 0:1] * d1 + t21[:, 0:1] * d2
    orient = area > 0
    sign = np.where(orient, 1.0, -1.0)[:, None]
    nonzero = np.abs(area) > np.finfo(np.float32).tiny
    vos = np.where(nonzero[:, None], _normalise(vos) * sign, vos)
    vot = np.where(nonzero[:, None], _normalise(vot) * sign, vot)
    any_group = ~nonzero  # no UV area: takes the orientation of a neighbour's group
    degenerate = (W[:, 0] == W[:, 1]) | (W[:, 1] == W[:, 2]) | (W[:, 0] == W[:, 2])

    # 3. groups: a corner joins the corner of the same vertex across its outgoing edge, when
    # that triangle holds the edge reversed (the same winding) and has the same orientation
    corner = np.arange(3 * T).reshape(T, 3)
    a, b = W, np.roll(W, -1, axis=1)  # the outgoing edge (a -> b) of each corner
    size = int(W.max(initial=0)) + 1
    edge_key = (a * size + b).ravel()
    order = np.argsort(edge_key, kind="stable")
    sorted_keys = edge_key[order]
    twin_key = (b * size + a).ravel()
    pos = np.searchsorted(sorted_keys, twin_key)
    pos = np.minimum(pos, len(sorted_keys) - 1)
    found = sorted_keys[pos] == twin_key
    # the twin edge (b -> a) starts at corner c' of triangle g, so a is the next corner of g
    twin_corner = order[pos]
    g = twin_corner // 3
    k_next = (twin_corner % 3 + 1) % 3
    neighbour = g * 3 + k_next
    f_of = np.repeat(np.arange(T), 3)
    edge = found & ~degenerate[f_of] & ~degenerate[g] & (g != f_of)
    if any_group.any():
        # a triangle without UV area takes the orientation of the first group that reaches
        # it; the reference creates its groups in the order of their first corner
        link = edge & (orient[f_of] == orient[g]) & ~any_group[f_of] & ~any_group[g]
        first = _components(3 * T, corner.ravel()[link], neighbour[link])
        created = np.full(first.max(initial=-1) + 1, 3 * T)
        np.minimum.at(created, first, np.arange(3 * T))
        orient = orient.copy()
        resolved = ~any_group.copy()
        rank = np.where(resolved, created[first.reshape(T, 3)].min(1), 3 * T)
        for _ in range(8):  # chains of triangles without UV area
            pairs = np.concatenate(
                [np.stack([f_of, g], 1)[edge], np.stack([g, f_of], 1)[edge]]
            )
            pairs = pairs[~resolved[pairs[:, 0]] & resolved[pairs[:, 1]]]
            if len(pairs) == 0:
                break
            pairs = pairs[np.lexsort((rank[pairs[:, 1]], pairs[:, 0]))]
            t, pick = np.unique(pairs[:, 0], return_index=True)
            orient[t] = orient[pairs[pick, 1]]
            rank[t] = rank[pairs[pick, 1]]
            resolved[t] = True
    link = edge & (orient[f_of] == orient[g])
    group = _components(3 * T, corner.ravel()[link], neighbour[link]).reshape(T, 3)

    # 4. the angle-weighted sum of the projected tangents of each group
    n = N[F]  # (T, 3, 3) the normal of each corner's vertex
    prev = P[np.roll(F, 1, axis=1)]
    nxt = P[np.roll(F, -1, axis=1)]
    here = P[F]
    v1 = prev - here
    v2 = nxt - here
    v1 = _normalise(v1 - (v1 * n).sum(-1, keepdims=True) * n)
    v2 = _normalise(v2 - (v2 * n).sum(-1, keepdims=True) * n)
    angle = np.arccos(np.clip((v1 * v2).sum(-1), -1.0, 1.0))
    os_ = np.repeat(vos[:, None], 3, axis=1)
    os_ = _normalise(os_ - (os_ * n).sum(-1, keepdims=True) * n)
    contribution = angle[..., None] * os_
    contribution[degenerate] = 0.0
    total = np.zeros((group.max(initial=-1) + 1, 3))
    np.add.at(total, group.ravel(), contribution.reshape(-1, 3))
    empty = np.linalg.norm(total, axis=1) <= 1e-30
    total = _normalise(total)
    total[empty] = (1.0, 0.0, 0.0)  # as the reference gives a group without a tangent
    tangents = np.zeros((T, 3, 4))
    tangents[..., :3] = total[group]
    tangents[..., 3] = np.where(orient, 1.0, -1.0)[:, None]
    return tangents, group


def mikktspace(
    positions: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """
    MikkTSpace tangents (N, 4) of the vertices of a triangle mesh, the sign in ``w`` (see the
    module documentation). A vertex whose corners fall into several groups (a fan split by a
    mirrored mapping without a seam) takes the tangent of its largest group; vertices of no
    triangle get (1, 0, 0, 1).
    """
    tangents, group = corner_tangents(positions, normals, uvs, triangles)
    F = np.asarray(triangles, dtype=np.int64).ravel()
    flat = tangents.reshape(-1, 4)
    g = group.ravel()
    counts = np.bincount(g)[g]
    # per vertex, the corner of the largest group (ties: the first corner)
    order = np.lexsort((np.arange(len(F)), -counts, F))
    vertex, first = np.unique(F[order], return_index=True)
    out = np.tile(np.array([1.0, 0.0, 0.0, 1.0]), (len(positions), 1))
    out[vertex] = flat[order[first]]
    return out


def split_mesh(triangles: np.ndarray, uv_triangles: np.ndarray):
    """
    The vertices of a mesh split at UV seams, as the exporters write them: (source vertex,
    source UV, triangles of the split vertices), the pairs in sorted order (as
    :func:`opensculptboy.export.body.build_body` sorts them).
    """
    triangles = np.asarray(triangles, dtype=np.int64)
    pairs = np.stack(
        [triangles.reshape(-1), np.asarray(uv_triangles, np.int64).reshape(-1)], axis=1
    )
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return unique[:, 0], unique[:, 1], inverse.reshape(-1, 3)

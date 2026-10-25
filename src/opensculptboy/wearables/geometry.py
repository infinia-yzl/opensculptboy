# OpenSculptBoy
# Apache License, Version 2.0
"""
Surface queries of the wearables: the closest point of a triangle mesh, signed distances,
vertex normals and scales, triangle frames and ray hits.

All functions work on NumPy arrays in metres. The closest-point query finds candidate
triangles through a k-d tree of the triangle centroids (SciPy) and solves the exact point to
triangle problem on each candidate (Ericson, *Real-Time Collision Detection*, 5.1.5).
"""

from __future__ import annotations

import dataclasses

import numpy as np

# candidate triangles per query point
CANDIDATES = 24
# query points per chunk of the vectorised solve
CHUNK = 20_000


def vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals of a triangle mesh."""
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    face = np.cross(b - a, c - a)
    normals = np.zeros_like(vertices, dtype=np.float64)
    for i in range(3):
        np.add.at(normals, triangles[:, i], face)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(length > 0, length, 1.0)


def vertex_scales(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """The mean length of the edges at each vertex (0 for a vertex of no triangle)."""
    total = np.zeros(len(vertices))
    count = np.zeros(len(vertices))
    for i, j in ((0, 1), (1, 2), (2, 0)):
        length = np.linalg.norm(
            vertices[triangles[:, i]] - vertices[triangles[:, j]], axis=1
        )
        for k in (i, j):
            np.add.at(total, triangles[:, k], length)
            np.add.at(count, triangles[:, k], 1.0)
    return total / np.maximum(count, 1.0)


def _closest_on_triangles(p, a, b, c):
    """Barycentric coordinates (N, 3) of the closest point of each triangle (a, b, c) to each
    point p; all arrays (N, 3)."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        v = np.where(denom != 0, vb / denom, 0.0)
        w = np.where(denom != 0, vc / denom, 0.0)
    out = np.stack([1.0 - v - w, v, w], axis=1)
    # the regions of the vertices and edges, in Ericson's order (later ones win last)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        e_ac = d2 / (d2 - d6)
        e_ab = d1 / (d1 - d3)
    region_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    out[region_bc] = np.stack(
        [np.zeros(region_bc.sum()), 1 - e_bc[region_bc], e_bc[region_bc]], 1
    )
    region_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    out[region_ac] = np.stack(
        [1 - e_ac[region_ac], np.zeros(region_ac.sum()), e_ac[region_ac]], 1
    )
    region_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    out[region_ab] = np.stack(
        [1 - e_ab[region_ab], e_ab[region_ab], np.zeros(region_ab.sum())], 1
    )
    region_c = (d6 >= 0) & (d5 <= d6)
    out[region_c] = [0.0, 0.0, 1.0]
    region_b = (d3 >= 0) & (d4 <= d3)
    out[region_b] = [0.0, 1.0, 0.0]
    region_a = (d1 <= 0) & (d2 <= 0)
    out[region_a] = [1.0, 0.0, 0.0]
    return np.nan_to_num(out)


@dataclasses.dataclass
class Closest:
    """The closest surface point of each query point."""

    triangle: np.ndarray  # (N,) index into the surface's triangles
    barycentric: np.ndarray  # (N, 3)
    point: np.ndarray  # (N, 3)
    distance: np.ndarray  # (N,)


class Surface:
    """
    A triangle mesh with its vertex normals and scales, for closest-point queries.

    Args:
        vertices: (V, 3) the vertices.
        triangles: (T, 3) the triangles that take part in the queries.
        normals: (V, 3) unit vertex normals; the area-weighted normals of ``triangles`` by
            default.
    """

    def __init__(self, vertices, triangles, normals=None):
        from scipy.spatial import cKDTree

        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        self.normals = (
            vertex_normals(self.vertices, self.triangles)
            if normals is None
            else np.asarray(normals, dtype=np.float64)
        )
        self.scales = vertex_scales(self.vertices, self.triangles)
        corners = self.vertices[self.triangles]
        self.centroids = corners.mean(axis=1)
        self.radius = float(
            np.linalg.norm(corners - self.centroids[:, None], axis=2).max(initial=0.0)
        )
        self.tree = cKDTree(self.centroids)

    def closest(self, points, candidates: int = CANDIDATES) -> Closest:
        """The closest point of the surface to each of ``points`` (N, 3)."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        k = int(min(candidates, len(self.triangles)))
        best_t = np.zeros(len(points), np.int64)
        best_b = np.zeros((len(points), 3))
        best_d = np.full(len(points), np.inf)
        for start in range(0, len(points), CHUNK):
            p = points[start : start + CHUNK]
            _, near = self.tree.query(p, k=k)
            near = np.asarray(near).reshape(len(p), k)
            tri = self.triangles[near.reshape(-1)]
            q = np.repeat(p, k, axis=0)
            a, b, c = (self.vertices[tri[:, i]] for i in range(3))
            bary = _closest_on_triangles(q, a, b, c)
            foot = bary[:, :1] * a + bary[:, 1:2] * b + bary[:, 2:] * c
            dist = np.linalg.norm(q - foot, axis=1).reshape(len(p), k)
            # the nearest triangle; of triangles equally near (within 1e-12 m), the one of the
            # lowest index, so that the k-d tree's order of ties on another CPU changes nothing
            pick = np.lexsort((near, np.round(dist, 12)), axis=-1)[:, 0]
            rows = np.arange(len(p)) * k + pick
            sl = slice(start, start + len(p))
            best_t[sl] = near[np.arange(len(p)), pick]
            best_b[sl] = bary[rows]
            best_d[sl] = dist[np.arange(len(p)), pick]
        return Closest(best_t, best_b, self.point(best_t, best_b), best_d)

    def corners(self, triangle) -> np.ndarray:
        return self.triangles[np.asarray(triangle)]

    def point(self, triangle, barycentric) -> np.ndarray:
        return interpolate(self.vertices, self.corners(triangle), barycentric)

    def normal(self, triangle, barycentric) -> np.ndarray:
        n = interpolate(self.normals, self.corners(triangle), barycentric)
        return unit_rows(n)

    def signed_distance(self, points, candidates: int = CANDIDATES):
        """(distance along the surface normal at the closest point, :class:`Closest`)."""
        found = self.closest(points, candidates)
        n = self.normal(found.triangle, found.barycentric)
        signed = np.einsum("ij,ij->i", np.asarray(points) - found.point, n)
        return signed, found, n


def interpolate(values: np.ndarray, corners: np.ndarray, barycentric: np.ndarray):
    """Barycentric blends of per-vertex ``values`` over triangle ``corners`` (N, 3)."""
    values = np.asarray(values, dtype=np.float64)
    out = barycentric[:, 0, None] * values[corners[:, 0]]
    out = out + barycentric[:, 1, None] * values[corners[:, 1]]
    return out + barycentric[:, 2, None] * values[corners[:, 2]]


def unit_rows(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(length > 0, length, 1.0)


def frames(
    vertices: np.ndarray,
    normals: np.ndarray,
    scales: np.ndarray,
    corners: np.ndarray,
    barycentric: np.ndarray,
):
    """
    The frame of a bound point: its position, the unit axes (tangent, bitangent, normal) as
    (N, 3, 3) rows and its scale. The normal blends the corners' vertex normals, the tangent
    is the first edge of the triangle made orthogonal to it, and the scale blends the
    corners' edge lengths, so that offsets turn and stretch with the skin.
    """
    point = interpolate(vertices, corners, barycentric)
    n = unit_rows(interpolate(normals, corners, barycentric))
    edge = vertices[corners[:, 1]] - vertices[corners[:, 0]]
    edge = edge - np.einsum("ij,ij->i", edge, n)[:, None] * n
    t = unit_rows(edge)
    b = np.cross(n, t)
    scale = interpolate(scales[:, None], corners, barycentric)[:, 0]
    return point, np.stack([t, b, n], axis=1), scale


def ray_hits(
    origins: np.ndarray,
    directions: np.ndarray,
    vertices: np.ndarray,
    triangles: np.ndarray,
    reach: float,
) -> np.ndarray:
    """
    Whether the ray from each origin along its unit direction meets one of the triangles
    within ``reach`` metres (Moller-Trumbore on the triangles near each origin).
    """
    from scipy.spatial import cKDTree

    origins = np.asarray(origins, dtype=np.float64)
    directions = unit_rows(np.asarray(directions, dtype=np.float64))
    hit = np.zeros(len(origins), bool)
    if len(triangles) == 0 or len(origins) == 0:
        return hit
    corners = vertices[triangles]
    centroids = corners.mean(axis=1)
    radius = float(np.linalg.norm(corners - centroids[:, None], axis=2).max())
    tree = cKDTree(centroids)
    near = tree.query_ball_point(origins, reach + radius)
    rows = np.repeat(np.arange(len(origins)), [len(n) for n in near])
    if len(rows) == 0:
        return hit
    cols = np.concatenate([np.asarray(n, dtype=np.int64) for n in near])
    for start in range(0, len(rows), 400_000):
        r = rows[start : start + 400_000]
        c = cols[start : start + 400_000]
        a, b, cc = (corners[c, i] for i in range(3))
        d = directions[r]
        e1, e2 = b - a, cc - a
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = origins[r] - a
        u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, e1)
        v = np.einsum("ij,ij->i", d, q) * inv
        t = np.einsum("ij,ij->i", e2, q) * inv
        good = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0) & (t <= reach)
        hit[r[good]] = True
    return hit


def adjacency(count: int, triangles: np.ndarray):
    """The vertex adjacency of a triangle mesh as a sparse (count, count) matrix of ones."""
    from scipy.sparse import coo_matrix

    t = np.asarray(triangles, dtype=np.int64)
    rows = np.concatenate([t[:, 0], t[:, 1], t[:, 2], t[:, 1], t[:, 2], t[:, 0]])
    cols = np.concatenate([t[:, 1], t[:, 2], t[:, 0], t[:, 0], t[:, 1], t[:, 2]])
    matrix = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(count, count))
    matrix = matrix.tocsr()
    matrix.data[:] = 1.0
    return matrix


def line_partners(origins, directions, vertices, triangles, reach: float):
    """
    The partner of each origin on the line along its unit direction, within ``reach`` on
    either side: the nearest point of the triangles ahead of the origin, else the nearest
    point behind it. Returns (hit (N,), triangle (N,), barycentric (N, 3), along (N,): the
    signed distance of the partner along the direction, negative behind the origin),
    Moller-Trumbore on the triangles near each origin.
    """
    from scipy.spatial import cKDTree

    origins = np.asarray(origins, dtype=np.float64)
    directions = unit_rows(np.asarray(directions, dtype=np.float64))
    count = len(origins)
    best = np.full(count, np.inf)
    triangle = np.zeros(count, np.int64)
    barycentric = np.zeros((count, 3))
    signed = np.zeros(count)
    if not len(triangles) or not count:
        return np.isfinite(best), triangle, barycentric, signed
    corners = vertices[triangles]
    centroids = corners.mean(axis=1)
    radius = float(np.linalg.norm(corners - centroids[:, None], axis=2).max())
    near = cKDTree(centroids).query_ball_point(origins, reach + radius)
    rows = np.repeat(np.arange(count), [len(n) for n in near])
    if not len(rows):
        return np.isfinite(best), triangle, barycentric, signed
    cols = np.concatenate([np.asarray(n, dtype=np.int64) for n in near])
    for start in range(0, len(rows), 400_000):
        r, c = rows[start : start + 400_000], cols[start : start + 400_000]
        a, b, cc = (corners[c, i] for i in range(3))
        d = directions[r]
        e1, e2 = b - a, cc - a
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = origins[r] - a
        u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, e1)
        v = np.einsum("ij,ij->i", d, q) * inv
        along = np.einsum("ij,ij->i", e2, q) * inv
        # the rank of a hit: the distance ahead of the origin, or the reach plus the
        # distance behind it, so that every hit ahead ranks before every hit behind
        t = np.where(along >= 0.0, along, reach - along)
        good = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (np.abs(along) <= reach)
        k = np.flatnonzero(good)
        k = k[np.lexsort((c[k], t[k], r[k]))]  # per origin, the best hit first
        first = np.ones(len(k), bool)
        first[1:] = r[k][1:] != r[k][:-1]
        k = k[first]
        k = k[t[k] < best[r[k]]]  # nearer than a hit of an earlier chunk
        best[r[k]] = t[k]
        triangle[r[k]] = c[k]
        barycentric[r[k]] = np.stack([1.0 - u[k] - v[k], u[k], v[k]], axis=1)
        signed[r[k]] = along[k]
    return np.isfinite(best), triangle, barycentric, signed

# OpenSculptBoy
# Apache License, Version 2.0
"""
The slide of the skin along a posed surface, for the corrective shapes.

Plain skinning can stretch a narrow band of the skin far more than the skin around it, as in
the armpit of a raised arm. Real skin glides over the tissue under it, so the stretch spreads
over a wide area (Li, Sueda, Neog and Pai, "Thin Skin Elastodynamics", 2013). The slide does
the same on a mesh: the posed surface keeps its shape, and the vertices of a free zone move
along it until every triangle is as close to its rest shape as the zone allows.

The slide alternates the two steps of Sorkine and Alexa, "As-Rigid-As-Possible Surface
Modeling" (2007), on the triangles of the zone: the best turn of each rest triangle onto its
posed triangle, then the positions that fit those turned triangles in the least-squares sense.
After each round the free vertices are put back on the posed surface. It needs SciPy.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree


GIVE = 0.9


def closest_on_triangles(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """
    The closest point of each triangle (corners ``a``, ``b``, ``c``) to each point ``p``;
    the arrays broadcast to (..., 3).
    """
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    def ratio(top, bottom):
        safe = np.where(np.abs(bottom) < 1e-300, 1.0, bottom)
        return np.clip(np.where(np.abs(bottom) < 1e-300, 0.0, top / safe), 0.0, 1.0)

    total = va + vb + vc
    safe = np.where(np.abs(total) < 1e-300, 1.0, total)
    out = a + ab * (vb / safe)[..., None] + ac * (vc / safe)[..., None]
    cases = (
        (
            (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),
            b,
            c - b,
            (d4 - d3, d4 - d3 + d5 - d6),
        ),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a, ac, (d2, d2 - d6)),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a, ab, (d1, d1 - d3)),
    )
    for mask, origin, direction, (top, bottom) in cases:
        on_edge = origin + direction * ratio(top, bottom)[..., None]
        out = np.where(mask[..., None], on_edge, out)
    out = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, out)
    out = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, out)
    return np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, out)


class Surface:
    """A triangle surface that gives the closest point on it to a point near it."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, neighbours: int = 24):
        self.a = vertices[faces[:, 0]]
        self.b = vertices[faces[:, 1]]
        self.c = vertices[faces[:, 2]]
        self.tree = cKDTree((self.a + self.b + self.c) / 3.0)
        self.neighbours = min(neighbours, len(faces))

    def project(self, points: np.ndarray) -> np.ndarray:
        """
        The closest point of the surface to each of ``points`` (N, 3), among the triangles
        whose centres lie nearest to the point.
        """
        _, index = self.tree.query(points, k=self.neighbours)
        index = index.reshape(len(points), -1)
        found = closest_on_triangles(
            points[:, None, :], self.a[index], self.b[index], self.c[index]
        )
        best = np.linalg.norm(found - points[:, None, :], axis=-1).argmin(1)
        return found[np.arange(len(points)), best]


def free_zone(
    rest: np.ndarray,
    faces: np.ndarray,
    centre: np.ndarray,
    radius: float,
    allowed: np.ndarray | None = None,
) -> np.ndarray:
    """
    The vertices that may slide (V,): those of the triangles that lie whole within
    ``radius`` of ``centre`` on the rest surface, without the rim of that patch, which
    stays fixed and ties the zone to the skin around it. ``allowed`` (V,) leaves out
    vertices of other parts of the body that lie within the radius.
    """
    inside = np.linalg.norm(rest - centre, axis=1) < radius
    if allowed is not None:
        inside &= allowed
    whole = inside[faces].all(1)
    free = np.zeros(len(rest), dtype=bool)
    free[np.unique(faces[whole])] = True
    partly = free[faces].any(1) & ~free[faces].all(1)
    free[np.unique(faces[partly])] = False
    return free


def _rest_triangles(rest: np.ndarray, faces: np.ndarray):
    a0 = rest[faces[:, 1]] - rest[faces[:, 0]]
    b0 = rest[faces[:, 2]] - rest[faces[:, 0]]
    e1 = a0 / np.linalg.norm(a0, axis=1, keepdims=True)
    normal = np.cross(a0, b0)
    e2 = np.cross(normal / np.linalg.norm(normal, axis=1, keepdims=True), e1)
    flat = np.zeros((len(faces), 3, 2))
    flat[:, 1, 0] = (a0 * e1).sum(1)
    flat[:, 1, 1] = (a0 * e2).sum(1)
    flat[:, 2, 0] = (b0 * e1).sum(1)
    flat[:, 2, 1] = (b0 * e2).sum(1)
    return flat


def _narrowest(points: np.ndarray, zone: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    edges = np.stack(
        [
            points[zone[:, 1]] - points[zone[:, 0]],
            points[zone[:, 2]] - points[zone[:, 0]],
        ],
        axis=2,
    )
    return np.linalg.svd(edges @ inverse, compute_uv=False)[:, 1]


def _normals(points: np.ndarray, zone: np.ndarray) -> np.ndarray:
    return np.cross(
        points[zone[:, 1]] - points[zone[:, 0]], points[zone[:, 2]] - points[zone[:, 0]]
    )


def _guarded(out, ids, moved, zone, column, inverse, floor, surface) -> np.ndarray:
    before = out[ids]
    facing = _normals(out, zone)
    share = np.ones(len(ids))
    trial = out.copy()
    for attempt in range(6 + len(ids)):
        placed = before + share[:, None] * (moved - before)
        partial = (share > 0) & (share < 1)
        if partial.any():
            placed[partial] = surface.project(placed[partial])
        trial[ids] = placed
        broken = (_narrowest(trial, zone, inverse) < floor) | (
            (_normals(trial, zone) * facing).sum(1) <= 0
        )
        if not broken.any():
            return placed
        touched = column[zone[broken]].ravel()
        touched = np.unique(touched[(touched >= 0)])
        touched = touched[share[touched] > 0]
        if not len(touched):
            break
        share[touched] = 0.0 if attempt >= 5 else 0.5 * share[touched]
    return before


def slide(
    rest: np.ndarray,
    posed: np.ndarray,
    faces: np.ndarray,
    free: np.ndarray,
    iterations: int = 60,
    hold: float = 0.02,
    guard: float | None = 0.25,
) -> np.ndarray:
    """
    The posed surface with the vertices of ``free`` (V,) slid along it.

    Each round turns every rest triangle of the zone onto its present posed triangle, solves
    for the free vertices that fit those turned triangles best (cotangent weights of the
    rest triangles; ``hold`` ties each vertex to its place of the round before, as a share
    of the mean weight) and puts them back on the posed surface: on the triangles of the
    zone alone, so that a vertex never lands on another part of the body that the pose
    brings near. The other vertices stay.

    ``guard`` keeps the slide from trading stretch for squeezed skin: after a round, the
    narrower width of a triangle against its rest shape stays above :data:`GIVE` times
    the lower of ``guard`` and its value before the slide, and the triangle may not turn
    over. The vertices of a triangle that breaks
    this take half of their move of the round, again and again, and stay in place when
    that does not help, as the line searches of flip-free mappings do (Smith and Schaefer,
    "Bijective Parameterization with Free Boundaries", 2015). ``None`` switches it off. A quarter of the rest
    width stops triangles from collapsing; a higher guard holds more of the stretch back.
    The result has the shape of ``posed`` within the size of its triangles, and less uneven
    stretch against ``rest`` in the zone.
    """
    zone = faces[free[faces].any(1)]
    surface = Surface(posed, zone)
    flat = _rest_triangles(rest, zone)
    inverse = np.linalg.inv(np.stack([flat[:, 1], flat[:, 2]], axis=2))
    count = len(zone)
    cot = np.empty((count, 3))
    for i in range(3):
        u = flat[:, (i + 1) % 3] - flat[:, i]
        v = flat[:, (i + 2) % 3] - flat[:, i]
        cot[:, i] = (u * v).sum(1) / np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    weight = np.sqrt(0.5 * np.clip(cot, 0.05, 20.0))
    ids = np.where(free)[0]
    column = -np.ones(len(rest), dtype=np.int64)
    column[ids] = np.arange(len(ids))
    sides = [((i + 1) % 3, (i + 2) % 3, weight[:, i]) for i in range(3)]
    rows, cols, values = [], [], []
    for e, (j, k, w) in enumerate(sides):
        base = e * count + np.arange(count)
        for corner, sign in ((j, 1.0), (k, -1.0)):
            unknown = column[zone[:, corner]]
            mask = unknown >= 0
            rows.append(base[mask])
            cols.append(unknown[mask])
            values.append(sign * w[mask])
    matrix = sp.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * count, len(ids)),
    )
    normal = matrix.T @ matrix
    held = hold * normal.diagonal().mean()
    solve = spla.factorized((normal + held * sp.identity(len(ids))).tocsc())
    out = posed.copy()
    floor = None
    if guard is not None:
        floor = GIVE * np.minimum(_narrowest(out, zone, inverse), guard)
    for _ in range(iterations):
        edges = np.stack(
            [out[zone[:, 1]] - out[zone[:, 0]], out[zone[:, 2]] - out[zone[:, 0]]],
            axis=2,
        )
        u, _, vt = np.linalg.svd(edges @ inverse, full_matrices=False)
        turn = u @ vt
        right = np.zeros((3 * count, 3))
        for e, (j, k, w) in enumerate(sides):
            target = np.einsum("tij,tj->ti", turn, flat[:, j] - flat[:, k])
            for corner, sign in ((j, 1.0), (k, -1.0)):
                fixed = column[zone[:, corner]] < 0
                target = target - np.where(
                    fixed[:, None], sign * out[zone[:, corner]], 0.0
                )
            right[e * count : (e + 1) * count] = w[:, None] * target
        b = matrix.T @ right + held * out[ids]
        moved = np.stack([solve(b[:, axis]) for axis in range(3)], axis=1)
        moved = surface.project(moved)
        if guard is None:
            out[ids] = moved
            continue
        out[ids] = _guarded(out, ids, moved, zone, column, inverse, floor, surface)
    return out

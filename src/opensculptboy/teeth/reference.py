# OpenSculptBoy
# Apache License, Version 2.0
"""
The teeth of a reference character, in a fitted card's rest frame.

An appearance handoff gives the inside of the mouth parts with the role ``mouth``
(:func:`opensculptboy.fit.handoff.name_role`: teeth, gums, the tongue, the mouth's lining).
The fit leaves them out, and :func:`reference_teeth` takes the teeth among them: every
connected piece of those parts no larger than :data:`TOOTH_SIZE_MM` across, as a game draws
one tooth (a crown, or a crown and its root). The larger pieces (the gums, the tongue, the
lining) stay out, since anny's own mouth holds a cavity and a tongue. The pieces fall into two
arches by their heights (two clusters: the upper teeth and the lower teeth), and move from the
reference's frame into the card's rest frame by the rigid motion (Kabsch) that takes the
fitted head, as the fit placed and posed it, back to its rest pose.
"""

from __future__ import annotations

import numpy as np

from opensculptboy.teeth.mesh import TeethData, rigid

# the largest piece of a mouth part that counts as a tooth: a molar's crown and root
TOOTH_SIZE_MM = 15.0


def _pieces(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """The connected piece of each vertex, with vertices at one position welded."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    _, weld = np.unique(np.round(vertices, 7), axis=0, return_inverse=True)
    weld = np.asarray(weld).reshape(-1)
    n = int(weld.max(initial=-1)) + 1
    t = weld[triangles]
    graph = coo_matrix(
        (np.ones(t.size), (t.ravel(), np.roll(t, 1, axis=1).ravel())), shape=(n, n)
    )
    return connected_components(graph, directed=False)[1][weld]


def _two_arches(heights: np.ndarray) -> np.ndarray:
    """1 for the pieces of the lower arch, 0 for the upper: two clusters of the pieces'
    heights (k-means from the lowest and the highest); one arch (all upper) when the
    clusters lie closer than 1 mm."""
    lo, hi = float(heights.min()), float(heights.max())
    for _ in range(50):
        lower = np.abs(heights - lo) < np.abs(heights - hi)
        if lower.all() or not lower.any():
            break
        new_lo, new_hi = heights[lower].mean(), heights[~lower].mean()
        if np.isclose(new_lo, lo) and np.isclose(new_hi, hi):
            break
        lo, hi = new_lo, new_hi
    if hi - lo < 0.001:
        return np.zeros(len(heights), np.uint8)
    return lower.astype(np.uint8)


def reference_teeth(parts, posed: np.ndarray, rest: np.ndarray, head: np.ndarray):
    """
    The teeth of a reference (``parts``, :class:`~opensculptboy.fit.handoff.HandoffPart`
    of any role; those with the role ``mouth`` count) in the rest frame of a fitted mesh, with
    their colour and roughness: (:class:`~opensculptboy.teeth.mesh.TeethData`, ``{"color":
    linear RGB, "roughness": r}`` with the values the reference gives), or None without
    teeth. ``posed`` and ``rest`` are the fitted mesh's vertices as the fit placed it on the
    reference and at rest, and ``head`` the rows of its head
    (:func:`opensculptboy.teeth.mesh.mouth_rows`).
    """
    from opensculptboy.export.body import vertex_normals

    points, normals, uvs, triangles, heights, sizes = [], [], [], [], [], []
    piece_of, colours, roughness = [], [], []
    count = 0
    for part in parts:
        if part.role != "mouth" or len(part.triangles) == 0:
            continue
        v = np.asarray(part.vertices, dtype=np.float64)
        t = np.asarray(part.triangles, dtype=np.int64)
        piece = _pieces(v, t)
        n = (
            np.asarray(part.normals, dtype=np.float64)
            if part.normals is not None
            else vertex_normals(v, t)
        )
        uv = (
            np.c_[part.uvs[:, 0], 1.0 - part.uvs[:, 1]]
            if part.uvs is not None
            else np.zeros((len(v), 2))
        )
        took = False
        for p in np.unique(piece):
            rows = np.flatnonzero(piece == p)
            span = np.ptp(v[rows], axis=0).max() * 1000.0
            tris = t[np.isin(t, rows).all(1)]
            if span > TOOTH_SIZE_MM or len(tris) == 0:
                continue
            remap = np.full(len(v), -1)
            remap[rows] = np.arange(len(rows)) + count
            points.append(v[rows])
            normals.append(n[rows])
            uvs.append(uv[rows])
            triangles.append(remap[tris])
            heights.append(v[rows, 2].mean())
            sizes.append(len(rows))
            piece_of.append(np.full(len(rows), len(heights) - 1))
            count += len(rows)
            took = True
        if took:
            factor = getattr(part, "base_color_factor", None)
            if factor is not None:
                colours.append(np.asarray(factor, dtype=np.float64)[:3])
            if part.roughness is not None:
                roughness.append(float(np.median(part.roughness)))
    if not points:
        return None
    arch_of_piece = _two_arches(np.asarray(heights))
    piece_of = np.concatenate(piece_of)
    R, t = rigid(posed[head], rest[head])
    points = np.concatenate(points) @ R.T + t
    normals = np.concatenate(normals) @ R.T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    triangles = np.concatenate(triangles)
    data = TeethData(
        points=points,
        normals=normals,
        uvs=np.concatenate(uvs),
        triangles=triangles,
        arch=arch_of_piece[piece_of],
        part=np.zeros(len(triangles), np.uint8),
    )
    values = {}
    if colours:
        values["color"] = [round(float(c), 4) for c in np.median(colours, axis=0)]
    if roughness:
        values["roughness"] = round(float(np.median(roughness)), 4)
    return data, values

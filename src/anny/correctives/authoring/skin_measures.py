# OpenSculptBoy
# Apache License, Version 2.0
"""
Measures of the posed skin for the corrective shapes, on plain arrays.

Two properties of a posed surface are measured against its rest surface:

- the stretch of the skin: the principal stretches of each triangle (the singular values of
  its deformation gradient), as the texture stretch of Sander, Snyder, Gortler and Hoppe,
  "Texture Mapping Progressive Meshes" (2001). A value of 1 is the rest length;
- the hollow of a raised armpit: the depth of the dip between the front ridge (the edge of
  the chest muscle) and the back ridge (the edge of the back muscle) on a cross-section of
  the armpit. Anatomy texts describe this hollow as the pull of the suspensory ligament of
  the axilla on the armpit's floor when the arm is raised.

Lengths are in the units of the vertices.
"""

from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def principal_stretches(rest: np.ndarray, posed: np.ndarray, faces: np.ndarray):
    """
    The larger and the smaller principal stretch of each triangle from ``rest`` (V, 3) to
    ``posed`` (V, 3), and the rest area of each triangle: three arrays (T,).
    """
    a0 = rest[faces[:, 1]] - rest[faces[:, 0]]
    b0 = rest[faces[:, 2]] - rest[faces[:, 0]]
    normal = np.cross(a0, b0)
    area = 0.5 * np.linalg.norm(normal, axis=1)
    e1 = _unit(a0)
    e2 = np.cross(_unit(normal), e1)
    flat = np.empty((len(faces), 2, 2))
    flat[:, 0, 0] = (a0 * e1).sum(1)
    flat[:, 0, 1] = (b0 * e1).sum(1)
    flat[:, 1, 0] = (a0 * e2).sum(1)
    flat[:, 1, 1] = (b0 * e2).sum(1)
    edges = np.stack(
        [
            posed[faces[:, 1]] - posed[faces[:, 0]],
            posed[faces[:, 2]] - posed[faces[:, 0]],
        ],
        axis=2,
    )
    singular = np.linalg.svd(edges @ np.linalg.inv(flat), compute_uv=False)
    return singular[:, 0], singular[:, 1], area


def area_quantile(values: np.ndarray, area: np.ndarray, q: float) -> float:
    """The value that the share ``q`` of the area stays at or below."""
    order = np.argsort(values)
    share = np.cumsum(area[order]) / area.sum()
    return float(values[order][min(np.searchsorted(share, q), len(values) - 1)])


def stretch_summary(
    larger: np.ndarray, smaller: np.ndarray, area: np.ndarray, mask: np.ndarray
) -> dict:
    """
    The stretch of the triangles of ``mask``: the area-weighted mean, the value that 95 % of
    the area stays below, the peak, the area stretched over 2 times, the strongest
    compression and the mean change of area.
    """
    s1, s2, a = larger[mask], smaller[mask], area[mask]
    return {
        "mean": float((s1 * a).sum() / a.sum()),
        "p95": area_quantile(s1, a, 0.95),
        "peak": float(s1.max()),
        "area_over_2": float(a[s1 > 2.0].sum()),
        "least": float(s2.min()),
        "area_ratio": float((s1 * s2 * a).sum() / a.sum()),
    }


def armpit_triangles(
    rest: np.ndarray, faces: np.ndarray, shoulder: np.ndarray, up=(0.0, 0.0, 1.0)
) -> np.ndarray:
    """
    The triangles of the armpit on the rest surface: those within 7.5 % of the body height
    of the shoulder joint ``shoulder`` (3,) and more than 1.2 % of the height below it.
    """
    up = np.asarray(up, dtype=float)
    heights = rest @ up
    height = heights.max() - heights.min()
    centres = rest[faces].mean(1)
    near = np.linalg.norm(centres - shoulder, axis=1) < 0.075 * height
    return near & ((centres - shoulder) @ up < -0.012 * height)


def armpit_frame(
    shoulder: np.ndarray, elbow: np.ndarray, centre: np.ndarray, up=(0.0, 0.0, 1.0)
):
    """
    The frame of a raised armpit: its long axis (from the flank toward the inner arm), the
    outward direction (out of the hollow, away from the shoulder joint) and the direction
    from the front ridge to the back ridge, as three unit vectors. ``centre`` is the middle
    of the armpit's skin in the pose.
    """
    axis = _unit(_unit(elbow - shoulder) + np.asarray(up, dtype=float))
    outward = centre - shoulder
    outward = _unit(outward - (outward @ axis) * axis)
    return axis, outward, np.cross(axis, outward)


def section(
    vertices: np.ndarray, faces: np.ndarray, point: np.ndarray, normal: np.ndarray
):
    """
    The cut of a surface by the plane through ``point`` with the normal ``normal``: the
    segments (S, 2, 3) and the triangle of each segment (S,). A vertex that lies in the
    plane counts as just past it.
    """
    distance = (vertices - point) @ normal
    distance = np.where(distance == 0.0, np.finfo(float).tiny, distance)
    d = distance[faces]
    crossing = (d.min(1) < 0) & (d.max(1) > 0)
    rows = np.where(crossing)[0]
    d = d[rows]
    corners = vertices[faces[rows]]
    ends = np.zeros((len(rows), 2, 3))
    count = np.zeros(len(rows), dtype=np.int64)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        cut = (d[:, i] * d[:, j] < 0) & (count < 2)
        t = d[cut, i] / (d[cut, i] - d[cut, j])
        ends[cut, count[cut]] = corners[cut, i] + t[:, None] * (
            corners[cut, j] - corners[cut, i]
        )
        count[cut] += 1
    whole = count == 2
    return ends[whole], rows[whole]


def hollow_depth(
    vertices: np.ndarray,
    faces: np.ndarray,
    centre: np.ndarray,
    axis: np.ndarray,
    outward: np.ndarray,
    half_width: float,
    reach: float,
    bins: int = 48,
) -> dict:
    """
    The hollow on the cross-section of a surface through ``centre``, cut across ``axis``.

    The profile is the outermost surface that faces ``outward``, as its height along
    ``outward`` over the position from front to back (within ``half_width`` of the centre,
    heights within ``reach``). ``depth`` is the height of the lower of the two ridges (the
    highest point of the front half and of the back half) above the lowest point between
    them; a surface without a hollow gives a depth near 0. The result also holds the
    positions of both ridges and of the floor, and the profile itself.
    """
    across = np.cross(axis, outward)
    segments, rows = section(vertices, faces, centre, axis)
    corners = vertices[faces[rows]]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    facing = normals @ outward > 0
    points = segments[facing].reshape(-1, 3) - centre
    u, v = points @ across, points @ outward
    keep = (np.abs(u) <= half_width) & (np.abs(v) <= reach)
    u, v = u[keep], v[keep]
    edges = np.linspace(-half_width, half_width, bins + 1)
    index = np.clip(np.digitize(u, edges) - 1, 0, bins - 1)
    profile = np.full(bins, -np.inf)
    np.maximum.at(profile, index, v)
    middle = 0.5 * (edges[:-1] + edges[1:])
    filled = np.isfinite(profile)
    if (
        filled.sum() < 4
        or not filled[: bins // 2].any()
        or not filled[bins // 2 :].any()
    ):
        return {
            "depth": 0.0,
            "front": None,
            "back": None,
            "floor": None,
            "profile": None,
        }
    profile = np.interp(middle, middle[filled], profile[filled])
    front = int(np.argmax(profile[: bins // 2]))
    back = bins // 2 + int(np.argmax(profile[bins // 2 :]))
    floor = front + int(np.argmin(profile[front : back + 1]))
    depth = min(profile[front], profile[back]) - profile[floor]
    return {
        "depth": float(max(depth, 0.0)),
        "front": float(middle[front]),
        "back": float(middle[back]),
        "floor": float(middle[floor]),
        "profile": np.stack([middle, profile], axis=1),
    }

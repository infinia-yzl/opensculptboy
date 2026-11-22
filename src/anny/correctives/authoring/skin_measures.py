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
    shoulder: np.ndarray,
    elbow: np.ndarray,
    centre: np.ndarray,
    back=(0.0, 1.0, 0.0),
    up=(0.0, 0.0, 1.0),
):
    """
    The frame of a raised armpit from the skeleton: its long axis (from the flank toward
    the inner arm, across the torso's front-to-back direction), the outward direction (out
    of the hollow, away from the shoulder joint) and the direction from the front ridge to
    the back ridge, as three unit vectors. ``back`` and ``up`` are the torso's directions in
    the pose and ``centre`` is the middle of the armpit's skin.
    """
    back = _unit(np.asarray(back, dtype=float))
    axis = _unit(elbow - shoulder) + _unit(np.asarray(up, dtype=float))
    axis = _unit(axis - (axis @ back) * back)
    outward = np.cross(back, axis)
    if outward @ (centre - shoulder) < 0:
        outward = -outward
    return axis, outward, np.cross(axis, outward)


def section(
    vertices: np.ndarray, faces: np.ndarray, point: np.ndarray, normal: np.ndarray
):
    """
    The cut of a surface by the plane through ``point`` with the normal ``normal``: the
    segments (S, 2, 3), the triangle of each segment (S,) and the mesh edge under each end
    of a segment as a sorted pair of vertices (S, 2, 2). Two segments that meet share that
    edge. A vertex that lies in the plane counts as just past it.
    """
    distance = (vertices - point) @ normal
    distance = np.where(distance == 0.0, np.finfo(float).tiny, distance)
    d = distance[faces]
    rows = np.where((d.min(1) < 0) & (d.max(1) > 0))[0]
    d = d[rows]
    cut_faces = faces[rows]
    corners = vertices[cut_faces]
    ends = np.zeros((len(rows), 2, 3))
    edges = np.zeros((len(rows), 2, 2), dtype=np.int64)
    count = np.zeros(len(rows), dtype=np.int64)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        cut = (d[:, i] * d[:, j] < 0) & (count < 2)
        t = d[cut, i] / (d[cut, i] - d[cut, j])
        ends[cut, count[cut]] = corners[cut, i] + t[:, None] * (
            corners[cut, j] - corners[cut, i]
        )
        edges[cut, count[cut]] = np.sort(cut_faces[cut][:, [i, j]], axis=1)
        count[cut] += 1
    whole = count == 2
    return ends[whole], rows[whole], edges[whole]


def section_line(segments: np.ndarray, edges: np.ndarray, start: int):
    """
    The connected line of a cut through the segment ``start``, followed from segment to
    segment over the mesh edges that they share: its points in order (N, 3), the index of
    the first end of ``start`` among them (its second end follows it) and whether the line
    is a closed loop. A closed loop holds each point once and has ``start`` at its middle.
    """
    meeting: dict = {}
    for s in range(len(segments)):
        for e in range(2):
            meeting.setdefault((int(edges[s, e, 0]), int(edges[s, e, 1])), []).append(s)

    def follow(end: int):
        points, current, used = [], start, {start}
        while True:
            key = (int(edges[current, end, 0]), int(edges[current, end, 1]))
            points.append(segments[current, end])
            others = [s for s in meeting[key] if s not in used]
            if not others:
                return points, key
            current = others[0]
            used.add(current)
            same = (edges[current, 0] == np.array(key)).all()
            end = 1 if same else 0

    before, last = follow(0)
    closed = len(before) > 2 and last == (
        int(edges[start, 1, 0]),
        int(edges[start, 1, 1]),
    )
    if closed:
        line = np.array(before[::-1])
        shift = len(line) // 2
        return np.roll(line, shift, axis=0), shift - 1, True
    line = np.array(before[::-1] + follow(1)[0])
    seed = len(before) - 1
    closed = False
    return line, seed, bool(closed)


def hollow_depth(
    vertices: np.ndarray,
    faces: np.ndarray,
    centre: np.ndarray,
    axis: np.ndarray,
    outward: np.ndarray,
    half_width: float,
) -> dict:
    """
    The hollow on the cross-section of a surface through ``centre``, cut across ``axis``.

    The profile starts on the cut where it passes nearest to ``centre`` on a part of the
    surface that faces ``outward``. From there it follows the connected line of the cut
    toward the front and toward the back, as long as the line keeps going that way and
    stays within ``half_width`` of the centre; the line of a limb's cut turns back where
    it goes round the limb, and the profile ends there. Heights are measured along
    ``outward``. ``depth`` is the height of the lower of the two ridges (the highest point
    of the front part and of the back part) above the lowest point between them: 0 on a
    surface that bulges outward, and the dip of a groove between two ridges. The result
    also holds the positions of both ridges and of the floor from front to back, and the
    profile as (position, height) pairs from the front to the back.
    """
    across = np.cross(axis, outward)
    segments, rows, edges = section(vertices, faces, centre, axis)
    empty = {"depth": 0.0, "front": None, "back": None, "floor": None, "profile": None}
    if not len(segments):
        return empty
    corners = vertices[faces[rows]]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    distance = np.linalg.norm(segments.mean(1) - centre, axis=1)
    distance = np.where(normals @ outward > 0, distance, np.inf)
    start = int(np.argmin(distance))
    if not np.isfinite(distance[start]):
        return empty
    line, seed, closed = section_line(segments, edges, start)
    line = line - centre
    u, v = line @ across, line @ outward
    count = len(u)
    slack = 1e-6 * half_width
    toward_back = 1 if u[(seed + 1) % count] >= u[seed] else -1

    def walk(first: int, step: int, sign: float) -> list:
        reach = count // 2 if closed else count
        taken = [first]
        for _ in range(reach):
            following = taken[-1] + step
            if closed:
                following %= count
            elif following < 0 or following >= count:
                break
            if sign * (u[following] - u[taken[-1]]) < -slack:
                break
            if abs(u[following]) > half_width:
                break
            taken.append(following)
        return taken

    second = (seed + 1) % count
    back_part = walk(second if toward_back == 1 else seed, toward_back, 1.0)
    front_part = walk(seed if toward_back == 1 else second, -toward_back, -1.0)
    order = front_part[::-1] + back_part
    u, v = u[order], v[order]
    split = len(front_part)
    front = int(np.argmax(v[:split]))
    back = split + int(np.argmax(v[split:]))
    floor = front + int(np.argmin(v[front : back + 1]))
    depth = min(v[front], v[back]) - v[floor]
    return {
        "depth": float(max(depth, 0.0)),
        "front": float(u[front]),
        "back": float(u[back]),
        "floor": float(u[floor]),
        "profile": np.stack([u, v], axis=1),
    }

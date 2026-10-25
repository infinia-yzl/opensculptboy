# OpenSculptBoy
# Apache License, Version 2.0
"""
Orthographic z-buffer pictures of triangle meshes: the nearest triangle and its depth at each
pixel, for the visibility weights and the quality measures of
:mod:`opensculptboy.export.fit_budget`.

:func:`views` gives the directions around a figure that stands along +Y (a VRM file's frame),
each with a weight for how often a viewer sees the figure from there: VTuber avatars face the
camera, so the front weighs the most. :func:`render` draws the triangles of a mesh from one
direction into an id buffer (the index of the nearest triangle, -1 where empty) with its depth
and the barycentric coordinates of the pixel's centre, in NumPy alone.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class Camera:
    """An orthographic camera: ``right``, ``up`` and ``toward`` (unit vectors from the figure
    toward the camera), the picture's centre and the metres per pixel."""

    right: np.ndarray
    up: np.ndarray
    toward: np.ndarray
    centre: np.ndarray
    scale: float
    size: tuple
    weight: float = 1.0
    name: str = ""


# (name, yaw degrees from the front about +Y, pitch degrees, weight)
VIEWS = (
    ("front", 0, 0, 1.0),
    ("front-left", 35, 0, 0.8),
    ("front-right", -35, 0, 0.8),
    ("left", 90, 0, 0.5),
    ("right", -90, 0, 0.5),
    ("back-left", 145, 0, 0.3),
    ("back-right", -145, 0, 0.3),
    ("back", 180, 0, 0.3),
    ("front-above", 0, 30, 0.6),
    ("front-below", 0, -20, 0.4),
)


def views(
    vertices: np.ndarray, forward, size: int = 512, margin: float = 0.04, names=None
) -> list[Camera]:
    """The cameras of :data:`VIEWS` (or those named) around a figure along +Y facing
    ``forward`` (a unit vector in the horizontal plane), each framing every vertex."""
    V = np.asarray(vertices, np.float64)
    lo, hi = V.min(0), V.max(0)
    centre = 0.5 * (lo + hi)
    extent = float(np.linalg.norm(hi - lo)) * (1 + margin)
    f = np.asarray(forward, np.float64)
    f = f / np.linalg.norm(f)
    up_axis = np.array([0.0, 1.0, 0.0])
    out = []
    for name, yaw, pitch, weight in VIEWS:
        if names is not None and name not in names:
            continue
        a, b = np.radians(yaw), np.radians(pitch)
        side = np.cross(up_axis, f)  # the figure's left for a figure facing f
        toward = np.cos(a) * f + np.sin(a) * side
        toward = np.cos(b) * toward + np.sin(b) * up_axis
        toward /= np.linalg.norm(toward)
        right = np.cross(up_axis, toward)
        right /= np.linalg.norm(right)
        up = np.cross(toward, right)
        out.append(
            Camera(right, up, toward, centre, extent / size, (size, size), weight, name)
        )
    return out


def fragments(
    vertices: np.ndarray, triangles: np.ndarray, camera: Camera, chunk=400_000
):
    """
    Every fragment of a mesh seen by ``camera``: (pixel, depth, triangle, bary), sorted by
    pixel (row-major index) and then by depth (nearest first). Every triangle draws whichever
    way it faces, as double-sided materials draw.
    """
    V = np.asarray(vertices, np.float64) - camera.centre
    T = np.asarray(triangles, np.int64).reshape(-1, 3)
    w, h = camera.size
    x = V @ camera.right / camera.scale + w / 2.0 - 0.5
    y = h / 2.0 - 0.5 - V @ camera.up / camera.scale
    z = -(V @ camera.toward)
    X, Y, Z = x[T], y[T], z[T]
    x0 = np.clip(np.floor(X.min(1)), 0, w - 1).astype(np.int64)
    x1 = np.clip(np.ceil(X.max(1)), 0, w - 1).astype(np.int64)
    y0 = np.clip(np.floor(Y.min(1)), 0, h - 1).astype(np.int64)
    y1 = np.clip(np.ceil(Y.max(1)), 0, h - 1).astype(np.int64)
    inside = (
        (X.max(1) >= 0) & (X.min(1) <= w - 1) & (Y.max(1) >= 0) & (Y.min(1) <= h - 1)
    )
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    order = np.flatnonzero(inside)
    cells = (bw * bh)[order]
    out_p, out_d, out_t, out_b = [], [], [], []
    start = 0
    while start < len(order):
        stop = start + max(1, int(np.searchsorted(np.cumsum(cells[start:]), chunk)))
        part = order[start:stop]
        start = stop
        count = bw[part] * bh[part]
        t = np.repeat(part, count)
        local = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        px = x0[t] + local % bw[t]
        py = y0[t] + local // bw[t]
        ax, ay, bx, by, cx, cy = X[t, 0], Y[t, 0], X[t, 1], Y[t, 1], X[t, 2], Y[t, 2]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        valid = np.abs(det) > 1e-12
        det = np.where(valid, det, 1.0)
        l0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
        l1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
        l2 = 1.0 - l0 - l1
        hit = valid & (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        t, px, py = t[hit], px[hit], py[hit]
        bary = np.stack([l0[hit], l1[hit], l2[hit]], 1)
        out_p.append(py * w + px)
        out_d.append((bary * Z[t]).sum(1))
        out_t.append(t)
        out_b.append(bary)
    if not out_p:
        return (
            np.zeros(0, np.int64),
            np.zeros(0),
            np.zeros(0, np.int64),
            np.zeros((0, 3)),
        )
    pixel, depth = np.concatenate(out_p), np.concatenate(out_d)
    tri, bary = np.concatenate(out_t), np.concatenate(out_b)
    order = np.lexsort((depth, pixel))
    return pixel[order], depth[order], tri[order], bary[order]


def nearest(pixel, depth, tri, bary, size, keep=None):
    """(ids, depth, bary) images of the nearest fragment of each pixel among those that
    ``keep`` (a mask of the fragments) keeps."""
    w, h = size
    if keep is not None:
        pixel, depth, tri, bary = pixel[keep], depth[keep], tri[keep], bary[keep]
    first = np.r_[True, pixel[1:] != pixel[:-1]] if len(pixel) else np.zeros(0, bool)
    ids = np.full(w * h, -1, np.int64)
    best = np.full(w * h, np.inf)
    out_b = np.zeros((w * h, 3))
    ids[pixel[first]] = tri[first]
    best[pixel[first]] = depth[first]
    out_b[pixel[first]] = bary[first]
    return ids.reshape(h, w), best.reshape(h, w), out_b.reshape(h, w, 3)


def render(vertices: np.ndarray, triangles: np.ndarray, camera: Camera):
    """
    (ids, depth, bary) of a mesh seen by ``camera``: ids (H, W) the nearest triangle at each
    pixel centre (-1 where empty), depth (H, W) its distance along the view (larger is
    farther, inf where empty) and bary (H, W, 3) the pixel's barycentric coordinates in it.
    """
    return nearest(*fragments(vertices, triangles, camera), camera.size)


def visible_pixels(vertices, triangles, cameras) -> np.ndarray:
    """(T,) the weighted number of pixels where each triangle is the nearest, over the
    cameras (each pixel counts its camera's weight)."""
    T = np.asarray(triangles).reshape(-1, 3)
    out = np.zeros(len(T))
    for camera in cameras:
        ids, _, _ = render(vertices, T, camera)
        seen = ids[ids >= 0]
        out += camera.weight * np.bincount(seen, minlength=len(T))
    return out

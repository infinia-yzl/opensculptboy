# OpenSculptBoy
# Apache License, Version 2.0
"""
The eye texture: the eyeballs' region of anny's UV atlas, given a texture of its own.

Anny's eyeballs cover about 52 by 52 texels each of a 2048-texel atlas, too few for an iris. The
texture bake (:func:`opensculptboy.texture.bake.bake`) can therefore also write an eye texture
that holds that region alone: the bounding box of the eyeballs' UVs (both eyes), padded and
widened to the texture's aspect of :data:`EYE_TEXTURE_ASPECT`. The exporters draw the eyeballs
with a material of their own whose texture carries the transform of :func:`eye_texture_transform`
(``KHR_texture_transform`` in glTF and VRM 1.0, the ST vector of ``_MainTex`` in VRM 0.x), so
that the eyeballs keep anny's UVs and the files need no second UV set.
"""

from __future__ import annotations

import numpy as np

# width : height of the eye texture; the two eyes sit side by side in anny's atlas
EYE_TEXTURE_ASPECT = 2
# padding of the box on each side, as a share of its larger side
EYE_BOX_PADDING = 0.04


def eyeball_rows(model) -> np.ndarray | None:
    """The vertex rows of both eyeballs of an Anny model, or None for a mesh without them."""
    from opensculptboy.export.vrm_tables import eyeball_vertices

    try:
        return np.concatenate(list(eyeball_vertices(model).values()))
    except ValueError:
        return None


def eye_uv_box(model) -> tuple[np.ndarray, np.ndarray] | None:
    """
    The box (lower and upper corners, OBJ convention: v = 0 at the bottom) of anny's atlas that
    the eye texture covers, or None for a mesh without eyeballs or UVs.
    """
    rows = eyeball_rows(model)
    uvs = model.texture_coordinates
    uv_faces = model.face_texture_coordinate_indices
    if rows is None or uvs is None or uv_faces is None:
        return None
    faces = model.faces.cpu().numpy()
    uv_faces = uv_faces.cpu().numpy()
    uvs = uvs.detach().cpu().double().numpy()
    eye = np.zeros(len(model.template_vertices), bool)
    eye[rows] = True
    points = uvs[np.unique(uv_faces[eye[faces].all(1)])]
    lo, hi = points.min(0), points.max(0)
    pad = EYE_BOX_PADDING * float((hi - lo).max())
    lo, hi = lo - pad, hi + pad
    centre, size = 0.5 * (lo + hi), hi - lo
    # widen the short side to the aspect of the texture
    width = max(size[0], EYE_TEXTURE_ASPECT * size[1])
    size = np.array([width, width / EYE_TEXTURE_ASPECT])
    return centre - 0.5 * size, centre + 0.5 * size


def eye_texture_transform(model) -> dict | None:
    """
    The ``KHR_texture_transform`` (``{"offset": [u, v], "scale": [u, v]}``, glTF convention:
    v = 0 at the top) that takes anny's glTF UVs of the eye box to the eye texture's [0, 1]
    square, or None for a mesh without eyeballs.
    """
    box = eye_uv_box(model)
    if box is None:
        return None
    (u0, v0), (u1, v1) = box
    su, sv = 1.0 / (u1 - u0), 1.0 / (v1 - v0)
    # glTF v = 1 - v_obj: the box runs from 1 - v1 to 1 - v0
    return {
        "offset": [float(-u0 * su), float(-(1.0 - v1) * sv)],
        "scale": [float(su), float(sv)],
    }


def to_eye_uvs(uvs: np.ndarray, box) -> np.ndarray:
    """UVs of anny's atlas (OBJ convention) in the eye texture's square (OBJ convention)."""
    lo, hi = (np.asarray(b, np.float64) for b in box)
    return (np.asarray(uvs, np.float64) - lo) / (hi - lo)


# the layout bake: rays reach this far on each side of the fitted eyeball (metres), each texel
# takes SUPERSAMPLE by SUPERSAMPLE rays, and the gaze turns toward the reference's iris by at
# most this angle (degrees)
LAYOUT_REACH = 0.004
LAYOUT_SUPERSAMPLE = 3
MAX_GAZE_TURN = 15.0


def turned_axes(axes: np.ndarray, gaze: np.ndarray) -> np.ndarray:
    """An eye's axes (columns: +X, up, gaze) turned by the smallest rotation that takes its
    gaze to ``gaze`` (a direction)."""
    axes = np.asarray(axes, dtype=np.float64)
    g = np.asarray(gaze, dtype=np.float64)
    g = g / np.linalg.norm(g)
    a = axes[:, 2]
    v = np.cross(a, g)
    s, c = np.linalg.norm(v), float(a @ g)
    if s < 1e-12:
        return axes.copy()
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    return R @ axes


def _layout_directions(size, offset):
    """The unit directions in the eye's frame (gaze +Z) of the texels of the eye texture's
    layout (:mod:`opensculptboy.eyes.texture`), each moved by ``offset`` (x, y) of a texel,
    their angle about the gaze and the side of each texel (True for the left eye's half)."""
    from opensculptboy.eyes.texture import FRONT_SHARE

    w, h = size
    v, u = np.mgrid[0:h, 0:w]
    u = (u + 0.5 + offset[0]) / w
    v = 1.0 - (v + 0.5 + offset[1]) / h
    left = (np.mgrid[0:h, 0:w][1] + 0.5) / w < 0.5
    x = (u - np.where(left, 0.25, 0.75)) * 2.0
    y = v - 0.5
    t = np.hypot(x, y)
    front = 0.5 * FRONT_SHARE
    s = np.where(t <= front, t / front, 1.0 + (t - front) / (0.5 * (1 - FRONT_SHARE)))
    s = np.minimum(s, 2.0)
    phi = np.arctan2(y, x)
    rho = np.where(s <= 1.0, s, 2.0 - s)
    z = np.sqrt(np.clip(1.0 - rho**2, 0.0, 1.0)) * np.where(s <= 1.0, 1.0, -1.0)
    d = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], -1)
    return d, phi, left


def bake_eye_layout(target, frames: dict, size=None, gaze: bool = True):
    """
    The reference's eyes baked straight onto the eye texture's layout of
    :mod:`opensculptboy.eyes.texture`, with no detour through anny's atlas, whose eyeballs are
    coarse polygons with little room for the iris. ``target`` is the reference's eyes (a
    :class:`~opensculptboy.fit.reference.Reference`, ``Reference.eye_reference``), placed as
    the fitted mesh; ``frames`` gives each fitted eye's frame there (``"L"``, ``"R"``:
    :class:`~opensculptboy.eyes.geometry.EyeFrame`). Each texel's direction from the eye's
    centre gives a point of the fitted eyeball's sphere, and a ray along that direction takes
    the nearest hit on the reference's eyes within :data:`LAYOUT_REACH` (both ways):
    its base colour, and its shading normal in the frame (E1, E2, N) of the sphere there
    (:func:`opensculptboy.eyes.texture.carry_baked_normals`). With ``gaze``, each eye's gaze
    first turns toward the centre of the reference's pupil (by at most
    :data:`MAX_GAZE_TURN` degrees), so that the iris lies on the gaze as the reference's
    does. Returns the colours (H, W, 3, 8-bit sRGB), the normals (H, W, 3, 8-bit, ``n * 0.5
    + 0.5``) and each eye's axes as baked (columns: +X, up, gaze), or None when no ray
    reaches the reference's eyes.
    """
    from opensculptboy.eyes import texture as tex
    from opensculptboy.texture.bake import (
        _colours,
        _fill,
        _hits,
        _intersector,
        _shading_normals,
    )

    size = tex.EYE_TEXTURE_SIZE if size is None else size
    w, h = size
    rays = _intersector(target.vertices, target.triangles)
    k = LAYOUT_SUPERSAMPLE
    offsets = [
        ((i + 0.5) / k - 0.5, (j + 0.5) / k - 0.5) for j in range(k) for i in range(k)
    ]
    samples = [_layout_directions(size, o) for o in offsets]
    left = samples[0][2]

    def bake_side(frame, axes, half):
        colour = np.zeros((h, w, 3))
        normal = np.zeros((h, w, 3))
        count = np.zeros((h, w))
        for d, phi, _ in samples:
            dirs = d[half] @ axes.T
            points = frame.centre + frame.radius * dirs
            ids, tri, bary = _hits(points, dirs, target, LAYOUT_REACH, rays)
            if len(ids) == 0:
                continue
            values, known = _colours(target, tri, bary)
            shade = _shading_normals(target, tri, bary)
            n = dirs[ids]
            angle = phi[half][ids]
            e1 = (
                np.stack([-np.sin(angle), np.cos(angle), np.zeros(len(ids))], 1)
                @ axes.T
            )
            e1 -= (e1 * n).sum(1, keepdims=True) * n
            e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
            e2 = np.cross(n, e1)
            local = np.stack(
                [(shade * e1).sum(1), (shade * e2).sum(1), (shade * n).sum(1)], 1
            )
            rows, cols = np.nonzero(half)
            r, c = rows[ids], cols[ids]
            np.add.at(colour, (r[known], c[known]), values[known])
            np.add.at(normal, (r, c), local)
            np.add.at(count, (r, c), 1.0)
        return colour, normal, count

    colour = np.zeros((h, w, 3))
    normal = np.zeros((h, w, 3))
    count = np.zeros((h, w))
    out_axes = {}
    s, phi, _ = tex.texel_polar(size)
    for side, half in (("L", left), ("R", ~left)):
        frame = frames.get(side)
        if frame is None:
            continue
        axes = np.asarray(frame.axes, dtype=np.float64)
        c, n, k_ = bake_side(frame, axes, half)
        if gaze and (k_[half] > 0).any():
            found = c[half] / np.maximum(k_[half], 1.0)[:, None]
            hit = tex.find_iris(tex.linear(found), s[half], phi[half])
            if hit is not None:
                cx, cy = hit[0]
                rho = float(np.hypot(cx, cy))
                turn = np.degrees(np.arcsin(min(rho, 1.0)))
                if 1e-6 < turn <= MAX_GAZE_TURN:
                    d = np.array([cx, cy, np.sqrt(max(1.0 - rho * rho, 0.0))])
                    axes = turned_axes(axes, axes @ d)
                    c, n, k_ = bake_side(frame, axes, half)
        colour[half], normal[half], count[half] = c[half], n[half], k_[half]
        out_axes[side] = axes
    known = count > 0
    if not known.any():
        return None
    colour = _fill(colour / np.maximum(count, 1.0)[..., None], known)
    normal = normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
    normal[~known] = (0.0, 0.0, 1.0)
    normal = _fill(normal, known)
    return (
        np.clip(np.round(colour), 0, 255).astype(np.uint8),
        np.clip(np.round((normal * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8),
        out_axes,
    )

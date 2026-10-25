# OpenSculptBoy
# Apache License, Version 2.0
"""
Pictures of a dressed body for review: the body and its pieces in their own colours, with
markers on chosen points, from fixed views.

:func:`draw` renders meshes (:class:`Mesh`: vertices in Anny's frame, triangles and one sRGB
colour per triangle) with an orthographic camera (:class:`opensculptboy.render.flat.View`):
each pixel takes the colour of its nearest triangle, lit by a soft key light from the upper
left, both sides of a triangle alike, with a dark line where the depth jumps (the contours).
:class:`Marker` sets draw dots on the points that the camera sees (the depth test keeps a
point behind the figure out of the picture). :func:`labelled` adds a caption band and
:func:`grid` lays pictures out in rows. The pictures are made with NumPy and Pillow alone, so
that a person and a vision model can judge them on any machine.
"""

from __future__ import annotations

import dataclasses
import io

import numpy as np

from opensculptboy.render.flat import View

SKIN = (226, 188, 163)
BACKGROUND = (236, 238, 242)
LIGHT = (-0.45, 0.7, 0.55)  # picture right, up, toward the camera
VIEWS = {
    "front": View(0.0),
    "side": View(-90.0),  # the figure's left side
    "back": View(180.0),
    "three-quarter": View(-30.0, -8.0),
}


@dataclasses.dataclass
class Mesh:
    """Vertices (V, 3) in Anny's frame, triangles (T, 3) and colours: one sRGB triple (0 to
    255) for every triangle (T, 3), or one for the whole mesh."""

    vertices: np.ndarray
    triangles: np.ndarray
    colours: np.ndarray | tuple = SKIN

    def triangle_colours(self) -> np.ndarray:
        c = np.asarray(self.colours, dtype=np.float64)
        if c.ndim == 1:
            c = np.tile(c, (len(self.triangles), 1))
        return c.reshape(-1, 3)


@dataclasses.dataclass
class Marker:
    """Dots of ``radius`` pixels in an sRGB ``colour`` on ``points`` (N, 3), drawn where
    the camera sees them."""

    points: np.ndarray
    colour: tuple = (220, 30, 30)
    radius: float = 2.5


def _frame(points: np.ndarray, view: View, size, margin: float):
    """The scale and the offset that put the projected ``points`` in a picture of ``size``
    with ``margin`` pixels, centred: pixel = (x * s + ox, -y * s + oy)."""
    xy, _ = view.project(points)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    w, h = size[0] - 2 * margin, size[1] - 2 * margin
    span = np.maximum(hi - lo, 1e-6)
    s = min(w / span[0], h / span[1])
    ox = margin + (w - span[0] * s) / 2 - lo[0] * s
    oy = margin + (h - span[1] * s) / 2 + hi[1] * s
    return s, ox, oy


def draw(
    meshes,
    view: View = VIEWS["front"],
    size=(256, 384),
    margin: float = 10.0,
    frame_points: np.ndarray | None = None,
    markers=(),
    background=BACKGROUND,
    supersample: int = 2,
):
    """
    A PIL picture (RGB) of ``meshes`` from ``view``. ``frame_points`` (N, 3) sets the part of
    the scene that fills the picture (every vertex by default), so that a crop around one
    piece keeps the scale of its own extent.
    """
    from PIL import Image, ImageDraw

    from opensculptboy.render.flat import _rasters

    meshes = [m for m in meshes if len(m.triangles)]
    vertices = np.concatenate([np.asarray(m.vertices, np.float64) for m in meshes])
    starts = np.cumsum([0] + [len(m.vertices) for m in meshes])[:-1]
    triangles = np.concatenate(
        [np.asarray(m.triangles, np.int64) + s for m, s in zip(meshes, starts)]
    )
    # a floor under the colours, so that the light still shows the shape of black cloth
    colours = 24.0 + 0.9 * np.concatenate([m.triangle_colours() for m in meshes])
    points = vertices if frame_points is None else np.asarray(frame_points)
    s, ox, oy = _frame(points, view, size, margin)
    xy, depth = view.project(vertices)
    px = np.stack([xy[:, 0] * s + ox, -xy[:, 1] * s + oy], axis=1)
    # draw only the triangles that touch the picture
    inside = (px[:, 0] > -2) & (px[:, 0] < size[0] + 2)
    inside &= (px[:, 1] > -2) & (px[:, 1] < size[1] + 2)
    keep = inside[triangles].any(axis=1)
    ids = np.flatnonzero(keep)
    d, face = _rasters(
        px, depth, triangles[ids], size, supersample, shade=(np.arange(len(ids)) + 1.0)
    )
    face = np.rint(face).astype(np.int64) - 1
    filled = np.isfinite(d) & (face >= 0)
    right, up, toward = view.basis()
    light = LIGHT[0] * right + LIGHT[1] * up + LIGHT[2] * toward
    light /= np.linalg.norm(light)
    v = vertices[triangles[ids]]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-18)
    facing = n @ toward
    n[facing < 0] *= -1.0  # both sides of a triangle alike
    shade = 0.42 + 0.58 * np.clip(n @ light, 0.0, 1.0)
    rgb = np.empty(d.shape + (3,))
    rgb[:] = np.asarray(background, np.float64)
    hit = face[filled]
    rgb[filled] = colours[ids][hit] * shade[hit, None]
    # contours: a jump in depth of more than 2 cm between neighbouring pixels
    dd = np.where(np.isfinite(d), d, 1e3)
    jump = np.zeros(d.shape, bool)
    for axis in (0, 1):
        step = np.abs(np.diff(dd, axis=axis)) > 0.02
        if axis == 0:
            jump[1:] |= step
            jump[:-1] |= step
        else:
            jump[:, 1:] |= step
            jump[:, :-1] |= step
    rgb[jump & filled] *= 0.45
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    if supersample > 1:
        image = image.resize(tuple(size), Image.LANCZOS)
    paint = ImageDraw.Draw(image)
    for marker in markers:
        p = np.asarray(marker.points, np.float64).reshape(-1, 3)
        if not len(p):
            continue
        mxy, mdepth = view.project(p)
        mpx = np.stack([mxy[:, 0] * s + ox, -mxy[:, 1] * s + oy], axis=1)
        h, w = d.shape
        i = np.clip((mpx[:, 1] * supersample).astype(int), 0, h - 1)
        j = np.clip((mpx[:, 0] * supersample).astype(int), 0, w - 1)
        seen = mdepth <= dd[i, j] + 0.015
        seen &= (mpx[:, 0] >= 0) & (mpx[:, 0] < size[0])
        seen &= (mpx[:, 1] >= 0) & (mpx[:, 1] < size[1])
        r = marker.radius
        for x, y in mpx[seen]:
            paint.ellipse([x - r, y - r, x + r, y + r], fill=tuple(marker.colour))
    return image


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow before 10.1
        return ImageFont.load_default()


def labelled(image, lines, colour=(40, 40, 40), band: int | None = None, fill=None):
    """``image`` with a band of text ``lines`` under it (the first line bold in colour)."""
    from PIL import Image, ImageDraw

    lines = [str(t) for t in lines if t]
    size = max(11, image.size[0] // 22)
    band = band or (len(lines) * (size + 3) + 6)
    out = Image.new(
        "RGB", (image.size[0], image.size[1] + band), fill or (255, 255, 255)
    )
    out.paste(image, (0, 0))
    paint = ImageDraw.Draw(out)
    font = _font(size)
    y = image.size[1] + 3
    for k, text in enumerate(lines):
        paint.text((5, y), text, colour if k == 0 else (90, 90, 90), font)
        y += size + 3
    return out


def grid(rows, title_lines=(), background=(255, 255, 255)):
    """One picture of rows of pictures (a list of lists of PIL images), with a title band."""
    from PIL import Image, ImageDraw

    widths = [sum(i.size[0] for i in row) for row in rows]
    heights = [max(i.size[1] for i in row) for row in rows]
    width = max(widths + [640])
    size = max(14, width // 60)
    title = len(title_lines) * (size + 4) + (10 if title_lines else 0)
    sheet = Image.new("RGB", (width, title + sum(heights)), background)
    paint = ImageDraw.Draw(sheet)
    font = _font(size)
    y = 6
    for k, text in enumerate(title_lines):
        paint.text((8, y), str(text), (20, 20, 20) if k == 0 else (70, 70, 70), font)
        y += size + 4
    y = title
    for row, height in zip(rows, heights):
        x = 0
        for image in row:
            sheet.paste(image, (x, y))
            x += image.size[0]
        y += height
    return sheet


def png(image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=True)
    return stream.getvalue()

# OpenSculptBoy
# Apache License, Version 2.0
"""
The eye texture of the exported eyes: its layout, a procedural iris and the baked eyes of a
fitted card carried over to it.

Layout (:func:`polar_to_uv`): the texture is :data:`EYE_TEXTURE_SIZE` texels, one square half
per eye (the left eye, +X, in the left half), as the eye texture of
:mod:`opensculptboy.texture.eyes`. Within its square, a point of the eye lies at the angle
about the gaze at which it lies on the eye (from +X toward up, so that the square shows the eye
as it looks from the front) and at a distance from the square's centre that grows with its
distance from the gaze axis: the front half of the eyeball fills :data:`FRONT_SHARE` of the
square's half-width, so that the iris spans 220 of its 512 texels, and the back half the thin
ring outside it. The iris disc and the corneal bulge take the texels of their distance from the
axis, so the iris looks the same size from the front on every kind of eyeball.

:func:`procedural_eye_pixels` draws an eye from :class:`~opensculptboy.material.EyeMaterial`:
the pupil, an iris of radial fibres with a collarette and a darker rim, the limbal ring, and a
warm sclera with faint veins. :func:`baked_eye_pixels` carries a baked eye texture (on anny's
atlas, :mod:`opensculptboy.texture.eyes`) over to the layout through the eyeball of the mesh it
was baked on, finds its pupil and its iris and stretches them radially to the card's sizes, and
can tint the iris toward the card's colour: a game's iris textures are often grey, coloured by
the game's shader, and their pupils drawn wide.

:func:`matcap_pixels` draws the highlight that the MToon eyes of a VRM file take as a matcap: a
soft window light above and to the left of the view, and a faint fill below.
"""

from __future__ import annotations

import numpy as np

from opensculptboy.eyes.geometry import REFERENCE_EYEBALL_MM

# width, height of the eye texture in texels: one square half per eye
EYE_TEXTURE_SIZE = (1024, 512)
# the front half of the eyeball fills this share of each square's half-width
FRONT_SHARE = 0.9
MATCAP_SIZE = 256


def polar_to_uv(polar: np.ndarray, side: str) -> np.ndarray:
    """UVs (M, 2), OBJ convention (v = 0 at the bottom), of polar eye coordinates (M, 2)."""
    s, phi = np.asarray(polar, dtype=np.float64).T
    t = np.where(
        s <= 1.0,
        0.5 * FRONT_SHARE * s,
        0.5 * FRONT_SHARE + 0.5 * (1 - FRONT_SHARE) * (s - 1),
    )
    t = np.minimum(t, 0.5)
    cu = 0.25 if side == "L" else 0.75
    return np.stack([cu + 0.5 * t * np.cos(phi), 0.5 + t * np.sin(phi)], 1)


def texel_polar(size: tuple[int, int] = EYE_TEXTURE_SIZE):
    """
    The polar eye coordinates of every texel of the eye texture: ``s`` and ``phi`` (H, W) and
    the side of each texel (True for the left eye's half). Row 0 is the top of the image.
    """
    w, h = size
    v, u = np.mgrid[0:h, 0:w]
    u = (u + 0.5) / w
    v = 1.0 - (v + 0.5) / h  # OBJ convention
    left = u < 0.5
    x = (u - np.where(left, 0.25, 0.75)) * 2.0  # square units, -0.5 to 0.5
    y = v - 0.5
    t = np.hypot(x, y)
    front = 0.5 * FRONT_SHARE
    s = np.where(t <= front, t / front, 1.0 + (t - front) / (0.5 * (1 - FRONT_SHARE)))
    return np.minimum(s, 2.0), np.arctan2(y, x), left


def srgb(linear: np.ndarray) -> np.ndarray:
    """8-bit sRGB of linear colours in [0, 1]."""
    c = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    e = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
    return np.round(e * 255.0).astype(np.uint8)


def linear(pixels: np.ndarray) -> np.ndarray:
    """Linear colours of 8-bit sRGB pixels."""
    e = np.asarray(pixels, dtype=np.float64) / 255.0
    return np.where(e <= 0.04045, e / 12.92, ((e + 0.055) / 1.055) ** 2.4)


def _smooth(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _luminance(c: np.ndarray) -> np.ndarray:
    return c @ np.array([0.2126, 0.7152, 0.0722])


def _fibres(phi: np.ndarray, q: np.ndarray, rng, count: int, low: int, high: int):
    """A sum of random radial waves about the pupil, about -1 to 1."""
    out = np.zeros_like(phi)
    n = rng.integers(low, high, count)
    phase = rng.uniform(0, 2 * np.pi, count)
    twist = rng.normal(0.0, 1.2, count)
    reach = rng.uniform(0.3, 1.0, count)
    for k in range(count):
        wave = np.cos(n[k] * phi + phase[k] + twist[k] * q)
        out += wave * _smooth(reach[k], reach[k] - 0.25, q) / np.sqrt(n[k])
    return out / max(1e-9, np.sqrt((1.0 / n).sum()))


def _sized(eye):
    """``eye`` with numbers for its iris and pupil sizes (the defaults for ``"auto"``)."""
    sized = getattr(eye, "sized", None)
    return sized() if sized is not None and eye.wants_measure() else eye


def _radii(radii) -> dict[str, float]:
    if radii is None:
        return {"L": REFERENCE_EYEBALL_MM / 2.0, "R": REFERENCE_EYEBALL_MM / 2.0}
    return dict(radii)


def procedural_eye_pixels(
    eye, radii=None, size: tuple[int, int] = EYE_TEXTURE_SIZE
) -> np.ndarray:
    """The eye texture (H, W, 3) as 8-bit sRGB pixels for an
    :class:`~opensculptboy.material.EyeMaterial` on eyeballs of ``radii`` (mm by side; a human
    eyeball's when None)."""
    return srgb(procedural_eye_linear(eye, radii, size))


def procedural_eye_linear(
    eye, radii=None, size: tuple[int, int] = EYE_TEXTURE_SIZE
) -> np.ndarray:
    """The eye texture (H, W, 3) in linear colours, as :func:`procedural_eye_pixels`."""
    s, phi, left = texel_polar(size)
    radii = _radii(radii)
    eye = _sized(eye)
    rho_l, rho_p = eye.iris_mm / 2.0, eye.pupil_mm / 2.0
    out = np.empty(s.shape + (3,))
    for side, half, seed in (("L", left, eye.seed), ("R", ~left, eye.seed + 1)):
        rng = np.random.default_rng(seed)
        r, p = s[half] * radii[side], phi[half]  # mm from the gaze axis (front)
        q = (r - rho_p) / max(rho_l - rho_p, 1e-6)  # 0 at the pupil, 1 at the limbus
        # sclera: warm white, a little greyer toward the back and redder at the periphery
        sclera = np.array(eye.sclera_color, dtype=np.float64)
        shade = 1.0 - 0.18 * _smooth(0.85, 1.4, s[half])
        col = sclera[None] * shade[:, None]
        col = (
            col * (1.0 - 0.06 * _smooth(0.6, 1.0, s[half]))[:, None]
            + np.array([0.03, 0.005, 0.0]) * _smooth(0.6, 1.0, s[half])[:, None]
        )
        if eye.veins > 0:
            col = _veins(col, s[half], p, rng, eye.veins)
        # iris: radial fibres over the base colour, the collarette and its inner zone, a
        # darker outer band; the limbal ring, a dark band across the iris's edge, narrower
        # into the sclera; drawn on the texels near the iris alone
        width = max(eye.limbal_ring_mm, 1e-3) / max(rho_l - rho_p, 1e-6)
        near = q < 1.0 + 4.0 * width
        qi, pi = q[near], p[near]
        outer = np.array(eye.iris_color, dtype=np.float64)
        inner = np.array(eye.inner_color(), dtype=np.float64)
        fine = _fibres(pi, qi, rng, 90, 40, 180) * _smooth(0.0, 0.25, qi)
        coarse = _fibres(pi, qi, rng, 24, 6, 40)
        wobble = 0.03 * np.sin(7 * pi + rng.uniform(0, 6.3)) + 0.015 * np.sin(
            13 * pi + rng.uniform(0, 6.3)
        )
        collarette = 0.33 + wobble
        zone = _smooth(collarette + 0.08, collarette - 0.08, qi)  # 1 inside it
        iris = outer[None] * (1 - zone[:, None]) + inner[None] * zone[:, None]
        light = 1.0 + 0.2 * fine + 0.16 * coarse
        light *= 1.0 + 0.15 * np.exp(-(((qi - collarette) / 0.05) ** 2))  # the ridge
        light *= 1.0 - 0.3 * _smooth(0.7, 1.0, qi)  # the darker outer band
        iris = iris * np.clip(light, 0.2, 2.5)[:, None]
        spread = np.where(qi < 1.0, width, width / 3.0)
        ring = eye.limbal_ring * np.exp(-(((qi - 1.0) / spread) ** 2))
        edge = _smooth(1.0 - 0.1 * width, 1.0 + 0.1 * width, qi)  # iris to sclera
        mixed = iris * (1 - edge[:, None]) + col[near] * edge[:, None]
        col[near] = mixed * (1.0 - 0.85 * ring)[:, None]
        # the pupil, with a soft edge of 0.08 mm
        pupil = _smooth(rho_p + 0.04, rho_p - 0.04, r)
        col = col * (1 - pupil[:, None]) + 0.004 * pupil[:, None]
        # behind the eye: the plain sclera
        back = s[half] > 1.0
        col[back] = sclera * 0.8
        out[half] = col
    return out


def _veins(col, s, phi, rng, strength):
    """Thin red veins that run in from the periphery of the visible sclera."""
    vein = np.array([0.55, 0.06, 0.05])
    density = np.zeros_like(s)
    ring = (s > 0.55) & (s < 1.02)
    for _ in range(28):
        start = rng.uniform(0, 2 * np.pi)
        end = rng.uniform(0.62, 0.85)
        steps = 24
        ts = np.linspace(1.02, end, steps)
        ph = start + np.cumsum(rng.normal(0, 0.035, steps))
        xs, ys = ts * np.cos(ph), ts * np.sin(ph)
        sx, sy = s * np.cos(phi), s * np.sin(phi)
        box = ring & (sx > xs.min() - 0.02) & (sx < xs.max() + 0.02)
        box &= (sy > ys.min() - 0.02) & (sy < ys.max() + 0.02)
        x, y = sx[box], sy[box]
        best = np.full(x.shape, np.inf)
        along = np.zeros(x.shape)
        for i in range(steps - 1):
            ax, ay, bx, by = xs[i], ys[i], xs[i + 1], ys[i + 1]
            dx, dy = bx - ax, by - ay
            t = np.clip(((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy), 0, 1)
            d = np.hypot(x - ax - t * dx, y - ay - t * dy)
            closer = d < best
            best = np.where(closer, d, best)
            along = np.where(closer, (i + t) / (steps - 1), along)
        width = 0.006 * (1.0 - 0.6 * along)
        density[box] = np.maximum(
            density[box], np.exp(-((best / width) ** 2)) * (1.0 - along) ** 0.7
        )
    amount = np.clip(strength * density, 0.0, 1.0)[:, None]
    return col * (1 - amount) + vein[None] * amount


def raster(
    px: np.ndarray, triangles: np.ndarray, values: np.ndarray, size: tuple[int, int]
):
    """
    Values (H, W, D) of a texture whose texels (centres at integer pixel coordinates, row 0 at
    the top) are covered by triangles (T, 3) of points at pixel coordinates ``px`` (M, 2),
    interpolated linearly from per-point ``values`` (M, D), and the mask (H, W) of the texels
    covered.
    """
    w, h = size
    out = np.zeros((h, w, values.shape[1]))
    covered = np.zeros((h, w), bool)
    for tri in triangles:
        a, b, c = px[tri]
        lo = np.maximum(np.floor(np.minimum(np.minimum(a, b), c)).astype(int), 0)
        hi = np.minimum(
            np.ceil(np.maximum(np.maximum(a, b), c)).astype(int), [w - 1, h - 1]
        )
        if (hi < lo).any():
            continue
        m = np.array([b - a, c - a]).T
        if abs(np.linalg.det(m)) < 1e-12:
            continue
        ys, xs = np.mgrid[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
        p = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
        lam = np.linalg.solve(m, (p - a).T).T
        bary = np.c_[1.0 - lam.sum(1), lam]
        inside = (bary >= -1e-6).all(1)
        if not inside.any():
            continue
        yy, xx = ys.ravel()[inside], xs.ravel()[inside]
        out[yy, xx] = bary[inside] @ values[tri]
        covered[yy, xx] = True
    return out, covered


def rasterise_uvs(
    polar: np.ndarray,
    triangles: np.ndarray,
    values: np.ndarray,
    side: str,
    size: tuple[int, int] = EYE_TEXTURE_SIZE,
):
    """
    Values (H, W, D) of the eye texture's texels of one side, interpolated linearly over
    triangles (T, 3) of points with polar eye coordinates (M, 2) and per-point values (M, D),
    and the mask (H, W) of the texels covered. Triangles that reach far behind the eye (to its
    back pole, where the layout folds) are left out.
    """
    w, h = size
    uv = polar_to_uv(polar, side)
    px = np.stack([uv[:, 0] * w - 0.5, (1.0 - uv[:, 1]) * h - 0.5], 1)
    usable = (polar[triangles, 0] < 1.6).all(1)
    return raster(px, triangles[usable], values, size)


def _fill(values: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Texels outside ``covered`` take the value of the nearest covered texel."""
    from scipy import ndimage

    if covered.all() or not covered.any():
        return values
    _, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
    return values[iy, ix]


def _bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear samples of an image (H, W, D) at pixel coordinates (texel centres at
    integers), clamped to the edges."""
    h, w = image.shape[:2]
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def carry_baked(
    baked: np.ndarray,
    box,
    eyeballs: dict,
    size: tuple[int, int] = EYE_TEXTURE_SIZE,
) -> np.ndarray:
    """
    A baked eye texture (H, W, 3, linear colours, on anny's atlas inside ``box``,
    :func:`opensculptboy.texture.eyes.eye_uv_box`) carried over to the eye texture's layout
    (linear colours). ``eyeballs`` maps each side to the eyeball of the mesh the texture was
    baked on: (points in the eye's frame (M, 3), triangles (T, 3), UVs on anny's atlas (M, 2)
    or (T, 3, 2) per corner).
    """
    from opensculptboy.texture.eyes import to_eye_uvs

    w, h = size
    out = np.zeros((h, w, 3))
    covered = np.zeros((h, w), bool)
    bh, bw = baked.shape[:2]
    for side, (points, triangles, uvs) in eyeballs.items():
        # every corner of a triangle with its own UV, so that UV seams stay sharp
        corners = triangles.reshape(-1)
        uv = np.asarray(uvs, dtype=np.float64)
        uv = uv.reshape(-1, 2) if uv.ndim == 3 else uv[corners]
        directions = points[corners] / np.linalg.norm(
            points[corners], axis=1, keepdims=True
        )
        # the point of the sphere on each corner's ray, by its planar distance from the axis
        from opensculptboy.eyes.geometry import _polar

        polar = _polar(directions)
        fine = np.arange(len(corners)).reshape(-1, 3)
        box_uv = to_eye_uvs(uv, box)
        values, mask = rasterise_uvs(polar, fine, box_uv, side, size)
        x = values[..., 0] * bw - 0.5
        y = (1.0 - values[..., 1]) * bh - 0.5
        sample = _bilinear(baked, x, y)
        out[mask] = sample[mask]
        covered |= mask
    return _fill(out, covered)


def _profile(lum, s, phi, centre, step=0.004, reach=0.8):
    """The median luminance of rings about ``centre`` (square units of the texel layout)."""
    x, y = s * np.cos(phi) - centre[0], s * np.sin(phi) - centre[1]
    r = np.hypot(x, y)
    edges = np.arange(0.0, reach + step, step)
    which = np.digitize(r, edges) - 1
    prof = np.full(len(edges) - 1, np.nan)
    for i in range(len(prof)):
        sel = which == i
        if sel.sum() > 3:
            prof[i] = np.median(lum[sel])
    return 0.5 * (edges[:-1] + edges[1:]), prof


# a pale outer ring of the iris (as game textures often draw it) is at most this wide (eyeball
# radii), and the dark limbal ring beyond it falls back by at least this share of the step from
# the iris to the ring
PALE_RING_WIDTH = 0.15
PALE_RING_DROP = 0.3


def _pale_ring(radii, prof, valid, first: int, iris: float) -> float:
    """
    The limbus beyond the first ring ``first`` that is lighter than the iris: when a pale ring
    follows there and a darker limbal ring after it (the profile falls back by
    :data:`PALE_RING_DROP` of its rise within :data:`PALE_RING_WIDTH`, then rises again to the
    sclera), the pale ring is the iris's outer band and the limbus lies half way down that
    fall; otherwise the first ring itself.
    """
    start = float(radii[first])
    span = np.flatnonzero(valid & (radii >= start) & (radii <= start + PALE_RING_WIDTH))
    if len(span) < 4:
        return start
    values = prof[span]
    # the band's top: the last ring before the profile first falls back by a tenth of its
    # rise; the dip: the lowest ring after it before the profile climbs over the band again
    tops = np.maximum.accumulate(values)
    falling = np.flatnonzero(values < tops - 0.1 * max(float(tops[-1]) - iris, 1e-9))
    if len(falling) == 0:
        return start
    top = int(np.argmax(values[: falling[0]])) if falling[0] > 0 else 0
    peak = float(values[top])
    rise = peak - iris
    over = np.flatnonzero(values[top + 1 :] > peak)
    end = top + 1 + (int(over[0]) if len(over) else len(values) - top - 1)
    low = top + int(np.argmin(values[top:end]))
    dip = float(values[low])
    if rise <= 0 or peak - dip < PALE_RING_DROP * rise or low == len(values) - 1:
        return start
    # the sclera beyond rises again above the dip
    later = valid & (radii > radii[span[low]]) & (radii <= radii[span[low]] + 0.1)
    if not later.any() or np.nanmax(prof[later]) < dip + 0.5 * (peak - dip):
        return start
    half = 0.5 * (peak + dip)
    fall = np.flatnonzero(values[top : low + 1] < half)
    return float(radii[span[top + fall[0]]]) if len(fall) else start


def find_iris(colours: np.ndarray, s: np.ndarray, phi: np.ndarray):
    """
    The pupil and the iris of one eye of a texture in the eye texture's layout: the centre of
    the pupil (in units of ``s``), the pupil's radius, the limbus radius, and the median colour
    of the iris (linear), or None when the texture shows no dark pupil inside a darker iris.
    The pupil's edge lies half way from the pupil to the iris, and the limbus half way from
    the iris to the sclera, or, when a pale outer band of the iris and a darker limbal ring
    follow there, half way down from that band to the ring (:func:`_pale_ring`).
    ``colours`` (N, 3) are the linear colours of the texels at polar coordinates ``s``, ``phi``
    (N,), the front of the eye.
    """
    lum = _luminance(colours)
    front = s < 0.75
    if front.sum() < 100:
        return None
    lum_f = lum[front]
    dark = lum_f <= np.quantile(lum_f, 0.01)
    xs, ys = s[front] * np.cos(phi[front]), s[front] * np.sin(phi[front])
    centre = np.array([xs[dark].mean(), ys[dark].mean()])
    radii, prof = _profile(lum, s, phi, centre)
    valid = ~np.isnan(prof)
    if valid.sum() < 20:
        return None
    sclera = (
        np.nanmedian(lum[(s > 0.8) & (s < 0.95)])
        if ((s > 0.8) & (s < 0.95)).any()
        else np.nanmax(prof)
    )
    pupil = np.nanmin(prof[: max(5, valid.sum() // 4)])
    # the pupil's edge: the first ring halfway from the pupil to the iris median
    inner = prof[(radii > 0.05) & (radii < 0.45) & valid]
    if len(inner) == 0:
        return None
    iris = float(np.median(inner))
    if not (pupil < 0.6 * iris and iris < 0.85 * sclera):
        return None
    above = np.flatnonzero(valid & (prof > pupil + 0.5 * (iris - pupil)))
    if len(above) == 0:
        return None
    rho_p = float(radii[above[0]])
    # the limbus: beyond the pupil, the first ring halfway from the iris to the sclera
    beyond = np.flatnonzero(
        valid & (radii > rho_p + 0.05) & (prof > iris + 0.5 * (sclera - iris))
    )
    if len(beyond) == 0:
        return None
    rho_l = _pale_ring(radii, prof, valid, beyond[0], iris)
    x, y = s * np.cos(phi) - centre[0], s * np.sin(phi) - centre[1]
    r = np.hypot(x, y)
    ring = (r > rho_p + 0.2 * (rho_l - rho_p)) & (r < rho_l - 0.1 * (rho_l - rho_p))
    colour = np.median(colours[ring], axis=0) if ring.any() else colours[front].mean(0)
    return centre, rho_p, rho_l, colour


def fit_baked_iris(
    image: np.ndarray,
    eye,
    radii=None,
    size: tuple[int, int] = EYE_TEXTURE_SIZE,
    normals: np.ndarray | None = None,
):
    """
    A baked eye in the eye texture's layout (H, W, 3, linear) with each eye's pupil and iris
    centred on the gaze and stretched radially to the sizes of ``eye`` (an
    :class:`~opensculptboy.material.EyeMaterial`), and its iris tinted toward
    ``eye.iris_color`` by ``eye.iris_tint``. The pupil keeps the bake's own texels, stretched
    with the iris: a reference's pupil is often not black (a game's eye shader shades it), and
    its look is the reference's. Returns the image and, per side, what was found
    (:func:`find_iris`; ``"mirrored"`` when that side takes the other eye's mirror image; None
    when neither eye shows an iris, and the procedural eye replaces both). With ``normals``
    (H, W, 3, the baked eye's normals in the layout, :func:`carry_baked_normals`), they move
    with the texels and come back as a third value (flat where the procedural eye stands in).
    """
    s, phi, left = texel_polar(size)
    w, h = size
    eye = _sized(eye)
    out = image.copy()
    bent = None if normals is None else normals.copy()
    found = {}
    radii = _radii(radii)
    procedural = None
    for side, half in (("L", left), ("R", ~left)):
        target_p = eye.pupil_mm / 2.0 / radii[side]
        target_l = eye.iris_mm / 2.0 / radii[side]
        hit = find_iris(image[half], s[half], phi[half])
        found[side] = hit
        if hit is None:
            if procedural is None:
                procedural = procedural_eye_linear(eye, radii, size)
            out[half] = procedural[half]
            if bent is not None:
                bent[half] = FLAT_NORMAL
            continue
        centre, rho_p, rho_l, colour = hit
        # the source radius of each target radius, piecewise linear, then back to identity
        r = s[half]
        src = np.where(
            r < target_p,
            r * rho_p / target_p,
            rho_p + (r - target_p) * (rho_l - rho_p) / (target_l - target_p),
        )
        far = _smooth(target_l, target_l + 0.25, r)
        src = np.where(r > target_l, rho_l + (r - target_l), src)
        src = src * (1 - far) + r * far
        shift = (1 - far)[:, None] * centre[None]
        x = src * np.cos(phi[half]) + shift[:, 0]
        y = src * np.sin(phi[half]) + shift[:, 1]
        # back to texel coordinates of the same half
        rs = np.hypot(x, y)
        ps = np.arctan2(y, x)
        uv = polar_to_uv(np.stack([np.minimum(rs, 1.95), ps], 1), side)
        px, py = uv[:, 0] * w - 0.5, (1.0 - uv[:, 1]) * h - 0.5
        sample = _bilinear(image, px, py)
        if bent is not None:
            moved = _bilinear(normals, px, py)
            bent[half] = moved / np.maximum(
                np.linalg.norm(moved, axis=1, keepdims=True), 1e-9
            )
        if eye.iris_tint > 0:
            iris_zone = _smooth(target_p, target_p + 0.01, r) * _smooth(
                target_l + 0.01, target_l - 0.01, r
            )
            lum = _luminance(sample) / max(float(_luminance(colour)), 1e-6)
            tinted = np.array(eye.iris_color)[None] * lum[:, None]
            k = (eye.iris_tint * iris_zone)[:, None]
            sample = sample * (1 - k) + tinted * k
        out[half] = sample
    # one eye found, the other not: the other takes its mirror image
    half_width = w // 2
    for have, miss in (("L", "R"), ("R", "L")):
        if found[have] is not None and found[miss] is None:
            src_cols = slice(0, half_width) if have == "L" else slice(half_width, w)
            dst_cols = slice(half_width, w) if have == "L" else slice(0, half_width)
            out[:, dst_cols] = out[:, src_cols][:, ::-1]
            if bent is not None:
                # a mirror image turns the eye about the gaze the other way round
                bent[:, dst_cols] = bent[:, src_cols][:, ::-1] * np.array([-1.0, 1, 1])
            found[miss] = "mirrored"
            break
    if normals is None:
        return out, found
    return out, found, bent


# the normal of the eye's own surface in the frame of :func:`carry_baked_normals`
FLAT_NORMAL = np.array([0.0, 0.0, 1.0])


def _surface_frame(normals: np.ndarray):
    """The frame (E1, E2, N) of unit normals (M, 3) in an eye's frame: E1 runs about the gaze
    (+Z) in the sense of the layout's angle, E2 = N x E1; E1 is +Y where N lies on the
    axis."""
    e1 = np.stack([-normals[:, 1], normals[:, 0], np.zeros(len(normals))], 1)
    length = np.linalg.norm(e1, axis=1, keepdims=True)
    e1 = np.where(length > 1e-6, e1 / np.maximum(length, 1e-12), [0.0, 1.0, 0.0])
    e1 = e1 - (e1 * normals).sum(1, keepdims=True) * normals
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    return e1, np.cross(normals, e1), normals


def _uv_frames(points, triangles, uvs):
    """Per corner (T * 3) the unit normal (the direction from the eye's centre), the tangent
    along u and the sign of the bitangent along v (OBJ) of triangles (T, 3) of ``points``
    with per-corner UVs (T * 3, 2), as MikkTSpace gives them on a smooth sphere."""
    p = points[triangles]  # (T, 3, 3)
    uv = uvs.reshape(-1, 3, 2)
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    d1, d2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    det = np.where(np.abs(det) < 1e-18, 1e-18, det)
    t = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / det[:, None]
    b = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / det[:, None]
    n = p.reshape(-1, 3)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    t = np.repeat(t, 3, axis=0)
    b = np.repeat(b, 3, axis=0)
    t = t - (t * n).sum(1, keepdims=True) * n
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    sign = np.where((np.cross(n, t) * b).sum(1) < 0, -1.0, 1.0)
    return n, t, sign


def carry_baked_normals(
    normal_map: np.ndarray,
    eyeballs: dict,
    size: tuple[int, int] = EYE_TEXTURE_SIZE,
    box=None,
) -> np.ndarray:
    """
    The baked normals of the eyeballs (``normal_map``: a tangent-space normal map of anny's
    whole atlas, or of its eye texture's ``box`` when given, (H, W, 3) in [0, 1], encoded in
    the MikkTSpace frames of the eyeballs it was baked on) carried over to the eye texture's
    layout, as :func:`carry_baked` carries the colours: (H, W, 3) per texel, the normal in
    the frame (E1, E2, N) of the eye's own surface there (:func:`_surface_frame`), so that
    :func:`encode_normals` lays it on any eye surface; :data:`FLAT_NORMAL` is the surface's
    own normal. A reference's eye shader often bends its normals (a concave iris, a soft
    limbus), and its highlights and shading follow them.
    """
    from opensculptboy.eyes.geometry import _polar
    from opensculptboy.texture.eyes import to_eye_uvs

    w, h = size
    out = np.zeros((h, w, 3))
    covered = np.zeros((h, w), bool)
    bh, bw = normal_map.shape[:2]
    for side, (points, triangles, uvs) in eyeballs.items():
        corners = triangles.reshape(-1)
        uv = np.asarray(uvs, dtype=np.float64)
        uv = uv.reshape(-1, 2) if uv.ndim == 3 else uv[corners]
        n, t, sign = _uv_frames(points, triangles, uv)
        b = sign[:, None] * np.cross(n, t)
        polar = _polar(n)
        fine = np.arange(len(corners)).reshape(-1, 3)
        where = uv if box is None else to_eye_uvs(uv, box)
        values = np.concatenate([where, t, b, n], axis=1)
        grid, mask = rasterise_uvs(polar, fine, values, side, size)
        x = grid[..., 0] * bw - 0.5
        y = (1.0 - grid[..., 1]) * bh - 0.5
        code = 2.0 * _bilinear(normal_map, x, y) - 1.0
        frame_t, frame_b, frame_n = grid[..., 2:5], grid[..., 5:8], grid[..., 8:11]
        bent = (
            code[..., 0:1] * frame_t
            + code[..., 1:2] * frame_b
            + code[..., 2:3] * frame_n
        )[mask]
        frame_n = frame_n[mask]
        frame_n /= np.maximum(np.linalg.norm(frame_n, axis=1, keepdims=True), 1e-12)
        e1, e2, nn = _surface_frame(frame_n)
        local = np.stack(
            [(bent * e1).sum(1), (bent * e2).sum(1), (bent * nn).sum(1)], 1
        )
        local /= np.maximum(np.linalg.norm(local, axis=1, keepdims=True), 1e-12)
        out[mask] = local
        covered |= mask
    out = _fill(out, covered)
    out[~covered & (np.linalg.norm(out, axis=-1) < 0.5)] = FLAT_NORMAL
    return out


def encode_normals(
    bent: np.ndarray, surfaces: list, size: tuple[int, int] = EYE_TEXTURE_SIZE
) -> np.ndarray:
    """
    A tangent-space normal map (H, W, 3, 8-bit) of the eye texture's layout for the normals
    ``bent`` (:func:`carry_baked_normals`) on the surfaces of a file: each a tuple of UVs
    (M, 2, OBJ convention), triangles (T, 3), unit normals (M, 3) and MikkTSpace tangents
    (M, 4, the bitangent's sign in w) in the eye's frame (gaze +Z). Texels that no surface
    covers are flat. Triangles that reach far behind the eye (to its back pole, where the
    layout folds and a fan of triangles spans the whole square) are left out, as in
    :func:`rasterise_uvs`.
    """
    w, h = size
    out = np.zeros((h, w, 3))
    out[...] = [0.5, 0.5, 1.0]
    # the layout's distance from each square's centre at s = 1.6
    reach = 0.5 * FRONT_SHARE + 0.5 * (1 - FRONT_SHARE) * 0.6
    for uvs, triangles, normals, tangents in surfaces:
        cu = np.where(uvs[:, 0] < 0.5, 0.25, 0.75)
        t = np.hypot((uvs[:, 0] - cu) * 2.0, uvs[:, 1] - 0.5)
        triangles = triangles[(t[triangles] < reach).all(1)]
        px = np.stack([uvs[:, 0] * w - 0.5, (1.0 - uvs[:, 1]) * h - 0.5], 1)
        values = np.concatenate([normals, tangents], axis=1)
        grid, mask = raster(px, triangles, values, size)
        n = grid[mask][:, 0:3]
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        t = grid[mask][:, 3:6]
        t = t - (t * n).sum(1, keepdims=True) * n
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        b = np.where(grid[mask][:, 6:7] < 0, -1.0, 1.0) * np.cross(n, t)
        e1, e2, _ = _surface_frame(n)
        d = bent[mask]
        target = d[:, 0:1] * e1 + d[:, 1:2] * e2 + d[:, 2:3] * n
        code = np.stack(
            [(target * t).sum(1), (target * b).sum(1), (target * n).sum(1)], 1
        )
        code /= np.maximum(np.linalg.norm(code, axis=1, keepdims=True), 1e-12)
        out[mask] = 0.5 * code + 0.5
    return np.round(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def matcap_pixels(size: int = MATCAP_SIZE) -> np.ndarray:
    """A matcap (size, size, 3, 8-bit sRGB) for MToon eyes: black, with a soft window light
    above and to the left of the view and a faint fill light below and to the right."""
    v, u = np.mgrid[0:size, 0:size]
    x = (u + 0.5) / size * 2 - 1
    y = 1 - (v + 0.5) / size * 2
    key = np.exp(-((((x + 0.32) / 0.11) ** 2 + ((y - 0.38) / 0.09) ** 2) ** 2))
    fill = 0.18 * np.exp(-(((x - 0.35) / 0.12) ** 2 + ((y + 0.42) / 0.07) ** 2))
    value = np.clip(key + fill, 0, 1)
    return srgb(np.repeat(value[..., None], 3, axis=2))

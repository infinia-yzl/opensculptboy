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


def find_iris(colours: np.ndarray, s: np.ndarray, phi: np.ndarray):
    """
    The pupil and the iris of one eye of a texture in the eye texture's layout: the centre of
    the pupil (in units of ``s``), the pupil's radius, the limbus radius, and the median colour
    of the iris (linear), or None when the texture shows no dark pupil inside a darker iris.
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
    rho_l = float(radii[beyond[0]])
    x, y = s * np.cos(phi) - centre[0], s * np.sin(phi) - centre[1]
    r = np.hypot(x, y)
    ring = (r > rho_p + 0.2 * (rho_l - rho_p)) & (r < rho_l - 0.1 * (rho_l - rho_p))
    colour = np.median(colours[ring], axis=0) if ring.any() else colours[front].mean(0)
    return centre, rho_p, rho_l, colour


def fit_baked_iris(
    image: np.ndarray, eye, radii=None, size: tuple[int, int] = EYE_TEXTURE_SIZE
):
    """
    A baked eye in the eye texture's layout (H, W, 3, linear) with each eye's pupil and iris
    centred on the gaze and stretched radially to the sizes of ``eye`` (an
    :class:`~opensculptboy.material.EyeMaterial`), its iris tinted toward ``eye.iris_color``
    by ``eye.iris_tint``, and the pupil drawn dark. Returns the image and, per side, what was
    found (:func:`find_iris`, or None, when the procedural eye replaces that side).
    """
    s, phi, left = texel_polar(size)
    w, h = size
    out = image.copy()
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
        sample = _bilinear(image, uv[:, 0] * w - 0.5, (1.0 - uv[:, 1]) * h - 0.5)
        if eye.iris_tint > 0:
            iris_zone = _smooth(target_p, target_p + 0.01, r) * _smooth(
                target_l + 0.01, target_l - 0.01, r
            )
            lum = _luminance(sample) / max(float(_luminance(colour)), 1e-6)
            tinted = np.array(eye.iris_color)[None] * lum[:, None]
            k = (eye.iris_tint * iris_zone)[:, None]
            sample = sample * (1 - k) + tinted * k
        pupil = _smooth(target_p + 0.004, target_p - 0.004, r)
        sample = sample * (1 - pupil[:, None]) + 0.004 * pupil[:, None]
        out[half] = sample
    return out, found


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

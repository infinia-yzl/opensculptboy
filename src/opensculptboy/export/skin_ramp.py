# OpenSculptBoy
# Apache License, Version 2.0
"""
A baked skin ramp for MToon: the shade colour and the shading shift and toony factors that
imitate the warmth of subsurface scattering.

MToon (VRM's toon shader) lights a surface as ``mix(shade, lit, s(N.L))`` with one ramp
``s(x) = linearstep(-1 + toony, 1 - toony, x + shift)`` for every channel and a shade colour
per channel. Skin under real light scatters its red light further than its green and blue,
so its terminator turns warm and soft: a pre-integrated skin lookup (Penner 2011) gives each
channel its own falloff of diffuse light against ``N.L`` for a curvature. :func:`toon_ramp`
computes those falloffs and fits MToon's single ramp to them:

1. For each RGB channel, the diffuse falloff ``D_c(theta)`` on a sphere of radius
   ``curvature_radius_mm``: the clamped cosine averaged along the surface with the radial
   distribution of Burley's normalised diffusion profile, ``P(r) = (exp(-r / d) +
   exp(-r / 3d)) / 4d`` with ``d`` the channel's ``radius_mm`` (Blender's subsurface radius,
   a mean free path), over the arc length ``r = R |x|``.
2. The falloff of the material is ``(1 - w) max(cos theta, 0) + w D_c(theta)`` for the
   subsurface weight ``w``, relative to its value at ``theta = 0``.
3. A grid search over ``shift`` and ``toony`` (with the shade ratio of each channel solved by
   least squares in closed form, clamped to [0, 1]) fits ``ratio_c + (1 - ratio_c) s(cos
   theta)`` to it, uniformly over theta in [0, pi].

The fitted shade ratio multiplies the base colour (the export uses the base colour texture as
the shade texture too), so the shade colour factor is the ratio itself. It is an
approximation: MToon has one ramp shape for all channels, no light bleeding through thin
parts and no blur of the texture, and the curvature is one number for the whole skin (20 mm
in :func:`skin_toon_ramp`, the scale of the nose, the lips and the ears; over a cheek of 80
mm the falloff is a Lambert ramp in every channel). Scattering of a few millimetres changes
the falloff of a face little: a skin of subsurface weight 0.1 and radii (5.4, 2.1, 1.2) mm
gets the shade ratio (0.011, 0.003, 0.001), shift -0.5 and toony 0.5, which is a Lambert
ramp with a faint red terminator; weight 1 gives (0.13, 0.02, 0.00). The ramp follows the
material and does not add the softer, lighter shade of a stylised toon skin.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class ToonRamp:
    """MToon factors of a skin ramp, and the fit's RMS error per channel."""

    shade_ratio: tuple[float, float, float]
    shading_shift: float
    shading_toony: float
    rms: tuple[float, float, float]


def diffuse_falloff(
    radius_mm: float, weight: float, curvature_radius_mm: float, theta: np.ndarray
) -> np.ndarray:
    """The diffuse falloff of one channel against the angle ``theta`` between the normal
    and the light, 1 at ``theta = 0`` (see the module documentation)."""
    x = np.linspace(-np.pi, np.pi, 2001)
    r = curvature_radius_mm * np.abs(x)
    d = max(float(radius_mm), 1e-6)
    p = (np.exp(-r / d) + np.exp(-r / (3.0 * d))) / (4.0 * d)
    p /= p.sum()
    lit = np.maximum(np.cos(theta[:, None] + x[None, :]), 0.0)
    scattered = lit @ p
    lambert = np.maximum(np.cos(theta), 0.0)
    falloff = (1.0 - weight) * lambert + weight * scattered
    top = (1.0 - weight) + weight * float(np.maximum(np.cos(x), 0.0) @ p)
    return falloff / top


def _linearstep(a, b, x):
    return np.clip((x - a) / np.maximum(b - a, 1e-6), 0.0, 1.0)


def toon_ramp(
    radius_mm=(1.0, 0.2, 0.1),
    weight: float = 1.0,
    curvature_radius_mm: float = 80.0,
    samples: int = 181,
) -> ToonRamp:
    """The MToon ramp that best imitates a skin's subsurface falloff (see the module
    documentation)."""
    theta = np.linspace(0.0, np.pi, samples)
    target = np.stack(
        [diffuse_falloff(r, weight, curvature_radius_mm, theta) for r in radius_mm], 1
    )
    cos = np.cos(theta)
    best = None
    for toony in np.linspace(0.0, 0.95, 20):
        for shift in np.linspace(-0.6, 0.6, 61):
            s = _linearstep(-1.0 + toony, 1.0 - toony, cos + shift)
            # target ~ ratio + (1 - ratio) s = s + ratio (1 - s)
            g = 1.0 - s
            denom = float(g @ g)
            ratio = (
                np.zeros(3) if denom < 1e-12 else ((target - s[:, None]).T @ g) / denom
            )
            ratio = np.clip(ratio, 0.0, 1.0)
            fit = s[:, None] + ratio[None, :] * g[:, None]
            rms = np.sqrt(((fit - target) ** 2).mean(0))
            if best is None or rms.sum() < best[0]:
                best = (rms.sum(), ratio, shift, toony, rms)
    _, ratio, shift, toony, rms = best
    return ToonRamp(
        tuple(float(c) for c in ratio),
        float(shift),
        float(toony),
        tuple(float(c) for c in rms),
    )


def skin_toon_ramp(skin, curvature_radius_mm: float = 20.0) -> ToonRamp:
    """The ramp of a :class:`~opensculptboy.material.SkinMaterial`, at the curvature of the
    features where skin's scattering shows (the nose, the lips, the ears: about 20 mm)."""
    s = skin.subsurface
    return toon_ramp(s["radius_mm"], float(s["weight"]), curvature_radius_mm)

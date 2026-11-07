# OpenSculptBoy
# Apache License, Version 2.0
"""
The light passes of a layered portrait pack (``squire.portraits/2`` with ``passes``), for an app
that relights the stills live.

Each layer of the pack can come with one file per light pass: the light of the key alone, of the
fill alone (the environment: the sky, the ground and the soft boxes, with a garment's own light)
and of the rim alone. Each pass holds linear light with straight alpha, so an app sums the passes
with the strengths and colours it wants and applies the viewer's display pass (the tone curve, the
saturation and the contrast) to the sum. At the strengths of 1 the sum gives the layer itself.

The pass files are WebP, which holds 8 bits a channel. A pass stores its linear colour divided by
the pack's scale for that pass (the light that reaches the top of its range), encoded with the sRGB
transfer curve: the steps of the curve follow the eye's, so the shadows keep their precision, and an
app's graphics card decodes the curve for free when it reads the file as an sRGB texture, filtering
in linear light. The alpha is the layer's own, the same in every pass, and lossless.

The rigs: ``studio`` is the viewer's studio preset (the look of the bust pack), and ``stage`` is
the stage's light: the key from the open side of the frame (the
right of a figure on the left), 40 degrees off the camera and 20 degrees up, as loop lighting; a
large, soft fill from the camera's side; and the rim from behind on the side away from the key.
"""

from __future__ import annotations

import copy
import math

import numpy as np

# The passes a pack may hold: the key, the fill and the rim, or the key and the fill together
# ("base") with the rim, when two passes must do.
PASS_NAMES = ("key", "fill", "rim", "base")
PASS_SETS = (("key", "fill", "rim"), ("base", "rim"))
# Each pass's colour is stored over its scale; the scale is this share above the brightest light
# of the first body layer (the 99.9th percentile of its covered pixels), so a rare highlight clips.
SCALE_PERCENTILE = 99.9
SCALE_HEADROOM = 1.15
SCALE_LEAST = 1e-3
# The colour of the transparent pixels around the figure: the nearest covered colour, for this
# many pixels, so a resampled edge keeps its colour.
BLEED = 4
# The pixels the reproduction counts: those at least this opaque (0 to 255).
FIGURE_ALPHA = 128

# The viewer's studio preset (viewer/src/main.ts, PRESETS.studio): the look of the bust pack.
STUDIO = {
    "label": "Studio",
    "exposure": 0.45,
    "sky": [0.022, 0.022, 0.025],
    "horizon": [0.014, 0.013, 0.013],
    "ground": [0.006, 0.0055, 0.005],
    "boxes": [
        {"az": 78, "el": 5, "w": 30, "h": 40, "c": [0.12, 0.123, 0.135]},
        {"az": 180, "el": 40, "w": 20, "h": 12, "c": [0.30, 0.31, 0.34]},
        {"az": 0, "el": 70, "w": 30, "h": 30, "c": [0.03, 0.03, 0.032]},
    ],
    "key": {"az": -22, "el": 26, "size": 9, "c": [4.4, 4.2, 4.0]},
    "rim": {"az": 140, "el": 35, "c": [1.8, 1.85, 2.0]},
    "bg": [[0.050, 0.052, 0.058], [0.005, 0.0053, 0.0062]],
}
# The stage's light: the studio's lights and exposure, turned so that the key comes from the
# open side of the frame (positive azimuths lie on the figure's left, the frame's right), the
# fill box sits beside the camera on the side away from the key, and the rim comes from behind on
# that side. The rim renders neutral, since the stage colours it.
STAGE = copy.deepcopy(STUDIO)
STAGE.update(
    label="Stage",
    boxes=[
        {"az": -18, "el": 6, "w": 40, "h": 45, "c": [0.12, 0.123, 0.135]},
        {"az": 180, "el": 40, "w": 20, "h": 12, "c": [0.30, 0.31, 0.34]},
        {"az": 0, "el": 70, "w": 30, "h": 30, "c": [0.03, 0.03, 0.032]},
    ],
    key={"az": 40, "el": 20, "size": 9, "c": [4.4, 4.2, 4.0]},
    rim={"az": -140, "el": 35, "c": [1.9, 1.9, 1.9]},
)
RIGS = {"studio": STUDIO, "stage": STAGE}


def parse_passes(text: str | None) -> list[str] | None:
    """A comma-separated list of passes, one of :data:`PASS_SETS`; None or empty gives None."""
    if not text:
        return None
    names = [n.strip() for n in text.split(",") if n.strip()]
    for s in PASS_SETS:
        if sorted(names) == sorted(s):
            return list(s)
    raise ValueError(
        f"The passes {names} are not a set the stage sums; take {' or '.join(','.join(s) for s in PASS_SETS)}."
    )


def rig(name: str) -> dict:
    """A rig by name, as the viewer's ``window.portrait.light`` takes it."""
    if name not in RIGS:
        raise ValueError(f"Unknown light rig {name!r}; the rigs are {list(RIGS)}.")
    return copy.deepcopy(RIGS[name])


def key_side(lights: dict) -> str:
    """The side of the frame the key light comes from: a positive azimuth lies on the figure's left,
    which a camera in front of the figure sees on the frame's right."""
    return "right" if lights["key"]["az"] > 0 else "left"


def describe(name: str, lights: dict, display: dict, face: dict | None = None) -> dict:
    """The manifest's description of the light: the rig, the side of the key, the display pass the
    app repeats on the sum, the encoding of the passes, the lights, and each pass's mean light on the
    lit face of at_ease (the app's measure of the light ratio)."""
    out = {
        "rig": name,
        "key_side": key_side(lights),
        "exposure": round(float(display["exposure"]), 5),
        "tone": {
            "curve": display.get("tone", "agx"),
            "saturation": round(float(display["saturation"]), 4),
            "contrast": round(float(display["contrast"]), 4),
        },
        "encoding": "srgb",
        "lights": {
            "key": {
                k: lights["key"][k]
                for k in ("az", "el", "size", "c")
                if k in lights["key"]
            },
            "rim": {k: lights["rim"][k] for k in ("az", "el", "c")},
            "fill": {"boxes": [[b["az"], b["el"]] for b in lights["boxes"]]},
        },
    }
    if face:
        out["face"] = {k: round(float(v), 6) for k, v in face.items()}
    return out


# ------------------------------------------------------------------ the encoding of a pass
def srgb_encode(x: np.ndarray) -> np.ndarray:
    """The sRGB transfer curve (linear 0 to 1 to encoded 0 to 1)."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_decode(e: np.ndarray) -> np.ndarray:
    """The inverse of :func:`srgb_encode`."""
    e = np.clip(e, 0.0, 1.0)
    return np.where(e <= 0.04045, e / 12.92, np.power((e + 0.055) / 1.055, 2.4))


def pass_scale(rgb: np.ndarray, alpha: np.ndarray) -> float:
    """The scale of a pass from one of its layers: its light at the 99.9th percentile of the
    covered pixels, with headroom."""
    cov = (
        alpha >= FIGURE_ALPHA
        if alpha.dtype == np.uint8
        else alpha >= FIGURE_ALPHA / 255
    )
    if not cov.any():
        return 1.0
    top = float(np.percentile(rgb[cov].max(-1), SCALE_PERCENTILE))
    return max(SCALE_LEAST, top * SCALE_HEADROOM)


def bleed(rgb: np.ndarray, alpha: np.ndarray, steps: int = BLEED) -> np.ndarray:
    """The colour of the transparent pixels near the figure: the mean of their covered neighbours,
    grown outward one pixel a step, so a filter that reads across the edge keeps the edge's colour."""
    out = rgb.astype(np.float32, copy=True)
    have = alpha > 0
    out[~have] = 0
    for _ in range(steps):
        if have.all():
            break
        acc = np.zeros_like(out)
        n = np.zeros(have.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.roll(have, (dy, dx), (0, 1))
            sc = np.roll(out, (dy, dx), (0, 1))
            # no wrap across the picture's edges
            if dy == 1:
                sh[0] = False
            elif dy == -1:
                sh[-1] = False
            if dx == 1:
                sh[:, 0] = False
            elif dx == -1:
                sh[:, -1] = False
            acc += sc * sh[..., None]
            n += sh
        grow = ~have & (n > 0)
        out[grow] = acc[grow] / n[grow, None]
        have = have | grow
    return out


def encode_pass(rgb: np.ndarray, alpha: np.ndarray, scale: float) -> np.ndarray:
    """A pass's linear colour (H, W, 3, straight) and the layer's alpha (H, W, uint8) as RGBA bytes:
    the colour over the scale through the sRGB curve, the alpha as it is, the colour bled a few pixels
    into the transparent surround and zero beyond."""
    c = bleed(rgb, alpha)
    e = np.round(srgb_encode(c / scale) * 255).astype(np.uint8)
    out = np.dstack([e, alpha.astype(np.uint8)])
    return out


def decode_pass(rgba: np.ndarray, scale: float) -> np.ndarray:
    """The linear colour (H, W, 3) of a stored pass."""
    return srgb_decode(rgba[..., :3].astype(np.float32) / 255) * scale


# ------------------------------------------------------------------ the display pass
# The viewer's sheet display (viewer/src/main.ts, sheetDisplay): three.js's AgX on the exposed
# linear colour, the saturation about the Rec. 709 luminance, the sRGB curve and the contrast.
_SRGB_TO_2020 = np.array(
    [[0.6274, 0.0691, 0.0164], [0.3293, 0.9195, 0.0880], [0.0433, 0.0113, 0.8956]]
)
_2020_TO_SRGB = np.array(
    [[1.6605, -0.1246, -0.0182], [-0.5876, 1.1329, -0.1006], [-0.0728, -0.0083, 1.1187]]
)
_INSET = np.array(
    [
        [0.856627153315983, 0.137318972929847, 0.11189821299995],
        [0.0951212405381588, 0.761241990602591, 0.0767994186031903],
        [0.0482516061458583, 0.101439036467562, 0.811302368396859],
    ]
)
_OUTSET = np.array(
    [
        [1.1271005818144368, -0.1413297634984383, -0.14132976349843826],
        [-0.11060664309660323, 1.157823702216272, -0.11060664309660294],
        [-0.016493938717834573, -0.016493938717834257, 1.2519364065950405],
    ]
)
AGX_MIN_EV = -12.47393
AGX_MAX_EV = 4.026069
LUMA = np.array([0.2126, 0.7152, 0.0722])


def _mat(v: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """GLSL's ``mat3(c0, c1, c2) * v`` for the rows of ``cols`` as the matrix's columns."""
    return v @ cols


def agx(rgb: np.ndarray) -> np.ndarray:
    """three.js's AgX tone curve (exposure already applied): linear sRGB in, linear sRGB out."""
    c = _mat(np.asarray(rgb, np.float64), _SRGB_TO_2020)
    c = _mat(c, _INSET)
    c = np.log2(np.maximum(c, 1e-10))
    c = np.clip((c - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV), 0.0, 1.0)
    x2 = c * c
    x4 = x2 * x2
    c = (
        15.5 * x4 * x2
        - 40.14 * x4 * c
        + 31.96 * x4
        - 6.868 * x2 * c
        + 0.4298 * x2
        + 0.1191 * c
        - 0.00232
    )
    c = _mat(c, _OUTSET)
    c = np.power(np.maximum(c, 0.0), 2.2)
    c = _mat(c, _2020_TO_SRGB)
    return np.clip(c, 0.0, 1.0)


def display(rgb: np.ndarray, saturation: float, contrast: float) -> np.ndarray:
    """The viewer's display pass on exposed linear colour: display values 0 to 1."""
    c = agx(rgb)
    lum = (c @ LUMA)[..., None]
    c = np.maximum(lum + (c - lum) * saturation, 0.0)
    c = np.where(
        c <= 0.0031308, c * 12.92, np.power(np.maximum(c, 0), 0.41666) * 1.055 - 0.055
    )
    return np.clip((c - 0.5) * contrast + 0.5, 0.0, 1.0)


def display8(rgb: np.ndarray, saturation: float, contrast: float) -> np.ndarray:
    """:func:`display` as bytes."""
    return np.round(display(rgb, saturation, contrast) * 255).astype(np.uint8)


# ------------------------------------------------------------------ the colour difference
_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_WHITE = np.array([0.95047, 1.0, 1.08883])


def lab(rgb8: np.ndarray) -> np.ndarray:
    """CIELAB (D65) of sRGB bytes."""
    lin = srgb_decode(np.asarray(rgb8, np.float64) / 255)
    xyz = lin @ _SRGB_TO_XYZ.T / _WHITE
    d = 6 / 29
    f = np.where(xyz > d**3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    return np.stack(
        [
            116 * f[..., 1] - 16,
            500 * (f[..., 0] - f[..., 1]),
            200 * (f[..., 1] - f[..., 2]),
        ],
        -1,
    )


def delta_e2000(a8: np.ndarray, b8: np.ndarray) -> np.ndarray:
    """CIEDE2000 between two arrays of sRGB bytes (..., 3)."""
    return delta_e2000_lab(lab(a8), lab(b8))


def delta_e2000_lab(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 between two arrays of CIELAB colours (..., 3) (Sharma, Wu and Dalal, 2005)."""
    L1, a1, b1 = np.moveaxis(np.asarray(lab1, np.float64), -1, 0)
    L2, a2, b2 = np.moveaxis(np.asarray(lab2, np.float64), -1, 0)
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm**7 / (Cm**7 + 25.0**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lmp = (L1 + L2) / 2
    Cmp = (C1p + C2p) / 2
    hs = h1p + h2p
    hmp = np.where(
        C1p * C2p == 0,
        hs,
        np.where(
            np.abs(h1p - h2p) <= 180,
            hs / 2,
            np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2),
        ),
    )
    T = (
        1
        - 0.17 * np.cos(np.radians(hmp - 30))
        + 0.24 * np.cos(np.radians(2 * hmp))
        + 0.32 * np.cos(np.radians(3 * hmp + 6))
        - 0.20 * np.cos(np.radians(4 * hmp - 63))
    )
    dtheta = 30 * np.exp(-(((hmp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cmp**7 / (Cmp**7 + 25.0**7))
    Sl = 1 + 0.015 * (Lmp - 50) ** 2 / np.sqrt(20 + (Lmp - 50) ** 2)
    Sc = 1 + 0.045 * Cmp
    Sh = 1 + 0.015 * Cmp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )


class Reproduction:
    """The colour difference between each layer as rendered and the sum of its stored passes
    through the display pass, over the figure's pixels; also the stored layer's own difference
    (its lossy WebP) as the baseline."""

    BINS = np.linspace(0, 20, 401)

    def __init__(self):
        self.kinds: dict[str, dict] = {}

    def add(self, kind: str, rendered: np.ndarray, summed8: np.ndarray, stored=None):
        cov = rendered[..., 3] >= FIGURE_ALPHA
        if not cov.any():
            return
        d = delta_e2000(rendered[..., :3][cov], summed8[cov])
        k = self.kinds.setdefault(
            kind,
            {
                "n": 0,
                "sum": 0.0,
                "hist": np.zeros(len(self.BINS) - 1),
                "base": 0.0,
                "max": 0.0,
            },
        )
        k["n"] += int(d.size)
        k["sum"] += float(d.sum())
        k["max"] = max(k["max"], float(d.max()))
        k["hist"] += np.histogram(np.clip(d, 0, 19.999), self.BINS)[0]
        if stored is not None:
            k["base"] += float(
                delta_e2000(rendered[..., :3][cov], stored[..., :3][cov]).sum()
            )

    @staticmethod
    def _p95(hist) -> float:
        c = np.cumsum(hist)
        if not c[-1]:
            return 0.0
        i = int(np.searchsorted(c, 0.95 * c[-1]))
        return float(Reproduction.BINS[i + 1])

    def report(self) -> dict:
        out = {}
        tot = {"n": 0, "sum": 0.0, "hist": 0, "base": 0.0, "max": 0.0}
        for kind, k in self.kinds.items():
            out[kind] = {
                "pixels": k["n"],
                "mean": round(k["sum"] / k["n"], 3),
                "p95": round(self._p95(k["hist"]), 2),
                "max": round(k["max"], 2),
                "webp_mean": round(k["base"] / k["n"], 3),
            }
            tot["n"] += k["n"]
            tot["sum"] += k["sum"]
            tot["hist"] = tot["hist"] + k["hist"]
            tot["base"] += k["base"]
            tot["max"] = max(tot["max"], k["max"])
        if tot["n"]:
            out["all"] = {
                "pixels": tot["n"],
                "mean": round(tot["sum"] / tot["n"], 3),
                "p95": round(self._p95(tot["hist"]), 2),
                "max": round(tot["max"], 2),
                "webp_mean": round(tot["base"] / tot["n"], 3),
            }
        return out


def face_light(passes: dict, alpha: np.ndarray, rect: dict | None) -> dict:
    """Each pass's mean luminance in linear light over the lit side of the face: the opaque pixels of
    a rectangle (the face) where the key (or the base pass) gives at least its median light. The app
    reads the light ratio of the pack from it, (key + fill) / fill."""
    if rect:
        sl = (
            slice(rect["y"], rect["y"] + rect["height"]),
            slice(rect["x"], rect["x"] + rect["width"]),
        )
    else:
        sl = (slice(None), slice(None))
    cov = alpha[sl] >= 250
    if not cov.any():
        return {name: 0.0 for name in passes}
    lum = {name: rgb[sl] @ LUMA for name, rgb in passes.items()}
    key = lum.get("key", lum.get("base"))
    lit = cov & (key >= np.median(key[cov])) if key is not None else cov
    return {name: float(v[lit].mean()) for name, v in lum.items()}


def is_finite_positive(v) -> bool:
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(v)
        and v > 0
    )

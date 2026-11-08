# OpenSculptBoy
# Apache License, Version 2.0
"""
Render a layered portrait pack (``squire.portraits/2``, docs/portrait-pack.md) with the web
viewer's page in headless Chromium.

A layered pack shows the character in poses and at several framings, from a close-up of the
head and shoulders to the whole body, for an app that stages the character as the subject of a
window. Each framing has its own scale in pixels per head (the height from the chin to the
crown), so the app never shows a frame larger than its pixels. The layers:

- a body layer for each pose and framing (``body.<pose>.<framing>.webp``), with the neutral face;
- a face layer for each framing and face (``face.<framing>.<expression>.<mouth>.<eyes>.webp``):
  only the rectangle of the face that changes, placed at the framing's face rectangle;
- a front layer for a pose whose hand passes in front of that rectangle
  (``front.<pose>.<framing>.webp``), drawn over the face layer;
- the leap of joy as a short sequence of key poses (``leap.<key>.webp``) at the full framing,
  each with its own face.

The camera of a framing follows the head: it keeps the head at the same pixels in every pose,
looking at the plane through the eyes, and every pose keeps the head's turn of ``at_ease``
(:mod:`opensculptboy.portrait_pack.poses`). The face layers of a framing, rendered once in
``at_ease``, therefore fit every pose of that framing to the pixel. The camera, the lights and the
look are those of the bust pack (``squire.portraits/1``): the sheet's long lens from the front, the
page's lights and exposure, straight alpha on a transparent background.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import time

import numpy as np

from opensculptboy.portrait_pack import contract as C
from opensculptboy.portrait_pack import layered_contract as LC
from opensculptboy.portrait_pack import light as light_passes
from opensculptboy.portrait_pack import poses as stage_poses
from opensculptboy.portrait_pack.faces import frame_actions, pack_faces
from opensculptboy.portrait_pack.render import (
    DEFAULT_LICENCE,
    PortraitError,
    PortraitPage,
    _one_line,
    _version,
    provenance,
    text_problem,
)

# The framings: pixels per head, the room kept above the crown and the cut below it in heads
# (None: the ground), and whether raised hands lift the frame's top. Each framing renders through
# the camera of one shot (``--shots-file``): the close framing for the close-up, the waist framing
# for the medium close-up, the thigh framing for the two-shot and the full framing for the whole
# body and the leap. The thought, the square-on stern close-up and the low, far shot have
# framings of their own (``thought``, ``stern`` and ``low``), which need the shots' cameras.
# The scales cover a window of 1440 x 900 pixels at device scale 2: a close-up with the face at
# 45 % of the height asks for about 810 pixels per head, a medium close-up at 120 % about 655,
# a two-shot at 100 % about 320, and the full shot about 150 at rest and about 210 at the top of
# the leap, where the stage pushes in until the figure takes 90 % of the height.
FRAMINGS = {
    "close": dict(scale=810, above=0.5, below=2.2, hands=False),
    "waist": dict(scale=660, above=0.5, below=3.6, hands=True),
    "thigh": dict(scale=330, above=0.5, below=6.0, hands=True),
    "full": dict(scale=160, above=0.5, below=None, hands=True),
    "thought": dict(scale=810, above=0.5, below=2.2, hands=False),
    "stern": dict(scale=810, above=0.5, below=2.2, hands=False),
    "low": dict(scale=330, above=0.5, below=7.8, hands=True),
}
# The framings every pack may hold without the shots' cameras; at_ease shows at each of them.
BASE_FRAMINGS = ("close", "waist", "thigh", "full")
# The leap's key poses hold more pixels per head than the full framing: the stage pushes in at the
# top of the leap until the figure takes 88 % of the height, which asks for about 210.
LEAP_SCALE = 230
# Fewer samples where the face is small on the screen (the two-shot, the low shot's framing is its
# own, and the full shot); every other framing and the leap take the pack's samples (16).
FRAMING_SAMPLES = {"thigh": 8, "full": 8}
# The poses each framing holds: the poses its shot shows. Raised hands leave the close framings;
# a pose with a face set of its own shows where its shot needs it (the chin in the thought, the
# bow of hand_on_chest in the medium close-up and the two-shot, the lowered head in the low shot).
_MAIN = ("at_ease", "relaxed", "hands_behind_head", "scratch_head", "arms_crossed")
_RAISED = ("wave", "fist", "cheer")
FRAMING_POSES = {
    "close": _MAIN,
    "waist": _MAIN + ("hand_on_chest",) + _RAISED,
    "thigh": _MAIN + ("hand_on_chest",) + _RAISED,
    "full": _MAIN + _RAISED,
    "thought": ("chin",),
    "stern": ("at_ease", "arms_crossed"),
    "low": ("downcast",),
}
# The poses of the close framing (kept for callers of the first layered pack).
CLOSE_POSES = FRAMING_POSES["close"]
# The expressions of the main face set at a framing whose shot shows only some (all elsewhere), from
# the stage's rules (docs/portrait-pack.md, "Expressions by framing"): the stern close-up
# shows neutral and stern; the full shot shows only the celebration (joy), with neutral and smile
# for the moment before and after it; the close-up leaves stern to the stern framing when the pack
# holds it.
FRAMING_EXPRESSIONS = {
    "stern": ("neutral", "stern"),
    "full": ("neutral", "joy", "smile"),
}
# Expressions a framing leaves to another framing of the pack: framing -> {expression: the other}.
FRAMING_LEAVES = {"close": {"stern": "stern"}}
# The margins: around the figure's sides and top, and around the face's changing rectangle, in heads.
SIDE_MARGIN = 0.08
TOP_MARGIN = 0.06
FACE_MARGIN = 0.05
# The probe of each pose's outline: a low picture of the whole figure.
PROBE_SIZE = 640
# The rectangle of the face layers snaps outward to this grid.
GRID = 8
# The pixels rendered around a face rectangle and cut off again, so the screen-space skin pass
# sees the same neighbours as in the whole frame.
FACE_PAD = 24
# The front layer: a pixel of the face rectangle belongs to the pose's front when it differs
# from at_ease's by more than this (0 to 255), with at least this share of the rectangle.
FRONT_THRESHOLD = 40
FRONT_MIN_SHARE = 0.003
# The face layer paints only the face's own pixels: those that any face changes, grown by this
# share of a head, with an edge this soft.
MASK_GROW = 0.04
MASK_SOFT = 0.02
DEFAULT_QUALITY = {"body": 88, "face": 90}
# The joints that are points of the head, which the page projects by these names.
POINTS = ("eyes", "crown", "chin")
FORMATS = ("webp", "png")


def parse_framing_samples(text: str | None) -> dict | None:
    """``--framing-samples``: "thigh=8,full=8" as a mapping, "none" as an empty one, None (the
    default, :data:`FRAMING_SAMPLES`) when not given."""
    if text is None:
        return None
    if text.strip().lower() == "none":
        return {}
    out = {}
    for part in text.split(","):
        if not part.strip():
            continue
        name, _, value = part.partition("=")
        name = name.strip()
        if name not in FRAMINGS or not value.strip().isdigit() or int(value) < 1:
            raise ValueError(
                f"--framing-samples takes framing=samples pairs with framings of {list(FRAMINGS)} "
                f"and samples of 1 or more, not {part.strip()!r}."
            )
        out[name] = int(value)
    return out


def layer_ok(path: pathlib.Path, size) -> bool:
    """A layer file that an earlier run finished: it decodes whole, with alpha, at ``size``
    (width, height)."""
    from PIL import Image

    try:
        with Image.open(path) as img:
            img.load()
            return img.size == tuple(size) and "A" in img.getbands()
    except (OSError, ValueError, SyntaxError):
        return False


def load_layer(path: pathlib.Path) -> np.ndarray:
    """A layer file as RGBA bytes."""
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGBA"))


def save_layer(path: pathlib.Path, rgba: np.ndarray, quality: int | None) -> None:
    """One layer: lossy WebP with alpha kept losslessly at ``quality``, or lossless PNG when
    ``quality`` is None."""
    from PIL import Image

    img = Image.fromarray(rgba, "RGBA")
    if path.suffix == ".webp":
        img.save(
            path, format="WEBP", quality=int(quality or 90), method=6, alpha_quality=100
        )
    else:
        img.save(path, format="PNG", optimize=True, compress_level=9)


def _mask_rows(alpha: np.ndarray, threshold: int = 24):
    """For each row of a mask, the first and last covered column (or -1)."""
    cov = alpha > threshold
    any_ = cov.any(1)
    first = np.where(any_, cov.argmax(1), -1)
    last = np.where(any_, cov.shape[1] - 1 - cov[:, ::-1].argmax(1), -1)
    return first, last


def changed_box(a: np.ndarray, b: np.ndarray, threshold: int = 16, least: int = 3):
    """The box (x0, y0, x1, y1) of the pixels that differ by more than ``threshold``, over the
    rows and columns with at least ``least`` of them (so a few stray pixels do not widen it), or
    None."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1) > threshold
    rows = np.nonzero(d.sum(1) >= least)[0]
    cols = np.nonzero(d.sum(0) >= least)[0]
    if not len(rows) or not len(cols):
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def snap_rect(x0, y0, x1, y1, width, height, grid=GRID) -> dict:
    """A rectangle snapped outward to the grid and kept inside the frame."""
    x0 = max(0, int(np.floor(x0 / grid)) * grid)
    y0 = max(0, int(np.floor(y0 / grid)) * grid)
    x1 = min(width, int(np.ceil(x1 / grid)) * grid)
    y1 = min(height, int(np.ceil(y1 / grid)) * grid)
    return {"x": x0, "y": y0, "width": max(grid, x1 - x0), "height": max(grid, y1 - y0)}


def changed_pixels(a: np.ndarray, b: np.ndarray, threshold: int = 16) -> np.ndarray:
    """The pixels where two pictures differ by more than ``threshold`` in a channel."""
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1) > threshold


def _grow(m: np.ndarray, r: int) -> np.ndarray:
    """A mask grown by ``r`` pixels (a square neighbourhood, by doubling steps)."""
    out = m.copy()
    step = 1
    left = r
    while left > 0:
        k = min(step, left)
        g = out.copy()
        g[k:] |= out[:-k]
        g[:-k] |= out[k:]
        g[:, k:] |= g[:, :-k].copy()
        g[:, :-k] |= g[:, k:].copy()
        out = g
        left -= k
        step *= 2
    return out


def _blur(a: np.ndarray, r: int) -> np.ndarray:
    """A box blur of radius ``r`` along both axes."""
    out = a.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
        n = out.shape[axis]
        hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        out = (hi - lo) / (2 * r + 1)
    return out


def soft_mask(changed: np.ndarray, grow: int, soft: int) -> np.ndarray:
    """The face's own pixels as weights 0 to 1: the changed pixels grown by ``grow`` pixels, with
    the edge softened over ``soft`` pixels."""
    m = _grow(changed, grow)
    w = np.clip(_blur(m, soft) * 2 - 0.5, 0, 1)
    return w * m


def front_mask(
    pose_crop: np.ndarray, base_crop: np.ndarray, within: np.ndarray | None = None
):
    """The pixels of a pose's face rectangle that lie in front of the face (a hand), as alpha
    0 to 255, inside the face's own pixels ``within`` (weights 0 to 1) when given; None when the
    pose leaves them as at_ease shows them."""
    d = np.abs(pose_crop.astype(np.int16) - base_crop.astype(np.int16)).max(-1)
    m = d > FRONT_THRESHOLD
    if within is not None:
        m &= within > 0.01
    if m.mean() < FRONT_MIN_SHARE:
        return None
    # two pixels of growth and a soft edge of one pixel
    grown = m.copy()
    for _ in range(2):
        g = grown.copy()
        g[1:] |= grown[:-1]
        g[:-1] |= grown[1:]
        g[:, 1:] |= grown[:, :-1]
        g[:, :-1] |= grown[:, 1:]
        grown = g
    soft = grown.astype(np.float32)
    s = soft.copy()
    s[1:-1, 1:-1] = (
        soft[1:-1, 1:-1] * 4
        + soft[:-2, 1:-1]
        + soft[2:, 1:-1]
        + soft[1:-1, :-2]
        + soft[1:-1, 2:]
    ) / 8
    return np.clip(s * 255, 0, 255).astype(np.uint8)


class Framer:
    """The framings of one character: the scale, the frame and the camera of each, from the
    head's size and the outlines of the poses."""

    def __init__(
        self, figures: dict, probes: dict, framings: dict, poses_of: dict, leap=()
    ):
        self.figures = figures
        self.probes = probes
        self.framings = framings
        self.poses_of = poses_of
        # the leap's frames take the full framing's camera of at_ease, so they count in its frame
        # as they stand, measured from at_ease's head
        self.leap = [k for k in leap if k in probes]
        base = figures[stage_poses.HEAD_POSE]
        self.H = float(base["crown"][1] - base["chin"][1])
        self.ground_below = float(base["crown"][1] - base["ground"])
        self.geo = {name: self._geometry(name) for name in framings}

    def _rel(self, pose, head_of=None):
        """A pose's outline relative to its head (or to the head of the pose ``head_of``): rows
        as (crown - y) / H, columns as (x - eyes_x) / H, with the covered span of each row."""
        pr = self.probes[pose]
        f = self.figures[head_of or pose]
        rows = (f["crown"][1] - pr["y"]) / self.H
        x0 = (pr["x0"] - f["eyes"][0]) / self.H
        x1 = (pr["x1"] - f["eyes"][0]) / self.H
        return rows, x0, x1, pr["covered"]

    def _geometry(self, name):
        F = self.framings[name]
        H = self.H
        s = F["scale"]
        below = F["below"] if F["below"] is not None else self.ground_below / H + 0.12
        # the top: the room above the crown, or the highest point of the poses (the hair, a
        # headpiece, raised hands), whichever is higher
        top = F["above"]
        left = right = 0.0
        outlines = [self._rel(pose) for pose in self.poses_of[name]]
        if name in ("full", "leap"):
            outlines += [self._rel(k, stage_poses.HEAD_POSE) for k in self.leap]
        for rows, x0, x1, cov in outlines:
            if F["hands"]:
                up = -rows[cov].min() if cov.any() else 0.0
            else:
                near = cov & (rows > -1.2)
                up = -rows[near].min() if near.any() else 0.0
            top = max(top, up + TOP_MARGIN)
        for rows, x0, x1, cov in outlines:
            inside = cov & (rows >= -top) & (rows <= below)
            if inside.any():
                left = max(left, -x0[inside].min())
                right = max(right, x1[inside].max())
        left += SIDE_MARGIN
        right += SIDE_MARGIN
        width = int(np.ceil((left + right) * s / GRID)) * GRID
        height = int(np.ceil((top + below) * s / GRID)) * GRID
        return {
            "scale": s,
            "top": top,
            "below": below,
            "left": left,
            "right": right,
            "width": width,
            "height": height,
        }

    def alias(self, name: str, of: str, scale: float) -> None:
        """A framing ``name`` that is ``of`` at another scale (the leap at its own pixels per head):
        the same camera, poses and outlines."""
        self.framings[name] = dict(self.framings[of], scale=scale)
        self.poses_of[name] = self.poses_of[of]
        for table in ("outlines", "figures_of", "cameras"):
            if hasattr(self, table):
                getattr(self, table)[name] = getattr(self, table)[of]
        self.geo[name] = self._geometry(name)

    def view(self, name: str, pose: str) -> dict:
        """The camera of a framing for a pose: the head at the same pixels in every pose."""
        g = self.geo[name]
        f = self.figures[pose]
        m = g["scale"] / self.H  # pixels per metre
        half_h = g["height"] / (2 * m)
        half_w = g["width"] / (2 * m)
        x_left = f["eyes"][0] - g["left"] * self.H
        top = f["crown"][1] + g["top"] * self.H
        return {
            "cx": x_left + half_w,
            "cy": top - half_h,
            "cz": f["eyes"][2],
            "halfH": half_h,
            "width": g["width"],
            "height": g["height"],
        }

    def head_rows(self, name: str) -> dict:
        """The head's place in a framing's frames: the anchor (between the eyes), the crown's row
        and the chin's row, in whole pixels."""
        g = self.geo[name]
        f = self.figures[stage_poses.HEAD_POSE]
        m = g["scale"] / self.H
        crown = g["top"] * self.H * m
        eyes_y = crown + (f["crown"][1] - f["eyes"][1]) * m
        chin = crown + (f["crown"][1] - f["chin"][1]) * m
        return {
            "x": int(round(g["left"] * self.H * m)),
            "y": int(round(eyes_y)),
            "top": int(round(crown)),
            "chin": int(round(chin)),
            **({"size": round(stage_unit(f) * m, 2)} if "joints" in f else {}),
        }


def stage_unit(figure: dict) -> float:
    """The stage's head height of a figure (metres): the hips to the eyes over 3.5, as the
    stage measures a head."""
    from opensculptboy.portrait_pack.shots import STAGE_HEAD_UNITS

    return float(figure["eyes"][1] - figure["joints"]["root"][1]) / STAGE_HEAD_UNITS


# ------------------------------------------------------------------ the shots' cameras
def shot_camera(cam, figure: dict, eyes_of: dict | None = None) -> dict:
    """The camera of a shot (:class:`opensculptboy.portrait_pack.shots.ShotCamera`) for a figure
    as it stands (``window.portrait.figure``): level with the eyes, the chest or the hips, at the
    distance the lens and the shot's window give, pitched by the shot's tilt. The camera follows
    the head of ``eyes_of`` (default: the figure itself): it keeps the same place relative to the
    eyes in every pose, so the head lands on the same pixels."""
    from opensculptboy.portrait_pack.shots import STAGE_HEAD_UNITS

    ref = eyes_of or figure
    eyes = np.asarray(ref["eyes"], dtype=np.float64)
    hu = (eyes[1] - float(ref["joints"]["root"][1])) / STAGE_HEAD_UNITS
    d = cam.distance(hu)
    t = np.radians(cam.tilt)
    eye = eyes + np.array([0.0, -cam.below_eyes * hu, d * np.cos(t)])
    return {"eye": eye, "pitch": float(cam.tilt), "hu": float(hu), "distance": float(d)}


def project_points(camera: dict, points, focal: float) -> np.ndarray:
    """Points (N, 3) through a shot camera with its principal point at (0, 0): pixels to the
    right and down, and the depth, (N, 3)."""
    t = np.radians(camera["pitch"])
    fwd = np.array([0.0, -np.sin(t), -np.cos(t)])
    up = np.array([0.0, np.cos(t), -np.sin(t)])
    d = np.atleast_2d(np.asarray(points, dtype=np.float64)) - camera["eye"]
    depth = d @ fwd
    return np.stack([focal * d[:, 0] / depth, -focal * (d @ up) / depth, depth], axis=1)


class ShotFramer(Framer):
    """The framings with the cameras of their shots (``--shots-file``): each framing renders
    through the camera of the shot it serves (its lens, height, tilt and the body's turn), and
    the outlines come from probes through that camera. The geometry is that of :class:`Framer`,
    in head heights (chin to crown) measured at the eyes' depth."""

    def __init__(self, figures, outlines, framings, poses_of, cameras, leap=()):
        # figures[name][pose], outlines[name][pose]: rows, x0, x1, covered, in heads from the
        # crown and the eyes; cameras[name]: the shot's ShotCamera
        self.figures_of = figures
        self.outlines = outlines
        self.framings = framings
        self.poses_of = poses_of
        self.cameras = cameras
        self.leap = [k for k in leap if any(k in o for o in outlines.values())]
        base = next(iter(figures.values()))[stage_poses.HEAD_POSE]
        self.H = float(base["crown"][1] - base["chin"][1])
        # the ground lies below the lowest point of at_ease's outline through each camera
        lowest = []
        for o in outlines.values():
            rows, _, _, cov = o[stage_poses.HEAD_POSE]
            if cov.any():
                lowest.append(rows[cov].max())
        self.ground_below = float(max(lowest) if lowest else 7.0) * self.H
        self._current = next(iter(framings))
        self.geo = {name: self._geometry(name) for name in framings}

    def _geometry(self, name):
        self._current = name
        return super()._geometry(name)

    def _rel(self, pose, head_of=None):
        return self.outlines[self._current][pose]

    def camera(self, name: str, pose: str) -> dict:
        """The shot camera of a framing for a pose (leap keys take at_ease's)."""
        figs = self.figures_of[name]
        ref = figs[pose] if pose in stage_poses.POSES else figs[stage_poses.HEAD_POSE]
        return shot_camera(self.cameras[name], figs.get(pose, ref), eyes_of=ref)

    def _focal(self, name: str, camera: dict, figure: dict) -> float:
        """The focal length that gives the framing's pixels per head at the eyes' depth."""
        depth = project_points(camera, figure["eyes"], 1.0)[0, 2]
        return self.geo[name]["scale"] * depth / self.H

    def head_rows(self, name: str) -> dict:
        g = self.geo[name]
        f = self.figures_of[name][stage_poses.HEAD_POSE]
        cam = self.camera(name, stage_poses.HEAD_POSE)
        focal = self._focal(name, cam, f)
        p = project_points(cam, [f["crown"], f["eyes"], f["chin"]], focal)
        crown = g["top"] * g["scale"]
        # the head's size as the camera projects it: one stage head height, upright at the eyes'
        # depth (the crown's and the chin's rows come closer together under a tilted camera)
        return {
            "x": int(round(g["left"] * g["scale"])),
            "y": int(round(crown + p[1, 1] - p[0, 1])),
            "top": int(round(crown)),
            "chin": int(round(crown + p[2, 1] - p[0, 1])),
            "size": round(focal * stage_unit(f) / p[1, 2], 2),
        }

    def view(self, name: str, pose: str) -> dict:
        g = self.geo[name]
        figs = self.figures_of[name]
        ref = figs[pose] if pose in stage_poses.POSES else figs[stage_poses.HEAD_POSE]
        cam = self.camera(name, pose)
        focal = self._focal(name, cam, ref)
        head = self.head_rows(name)
        u, v, _ = project_points(cam, ref["eyes"], focal)[0]
        return {
            "camera": {
                "eye": [float(x) for x in cam["eye"]],
                "pitch": cam["pitch"],
                "focal": float(focal),
                "px": float(head["x"] - u),
                "py": float(head["y"] - v),
            },
            "turn": float(self.cameras[name].turn),
            "halfH": float(g["height"] / 2 * self.H / g["scale"]),
            "width": g["width"],
            "height": g["height"],
        }


def probe_outline(
    alpha: np.ndarray, camera: dict, focal: float, pp, figure: dict, H: float
):
    """A probe's outline in heads (chin to crown) from the crown's row and the eyes' column, as
    :meth:`Framer._rel` gives it: rows, x0, x1, covered."""
    x0, x1 = _mask_rows(alpha)
    p = project_points(camera, [figure["crown"], figure["eyes"]], focal)
    unit = focal * H / p[1, 2]  # pixels per head at the eyes' depth
    ys = np.arange(alpha.shape[0]) + 0.5
    rows = (ys - (pp[1] + p[0, 1])) / unit
    ex = pp[0] + p[1, 0]
    return rows, (x0 - ex) / unit, (x1 + 1 - ex) / unit, x0 >= 0


def _shot_framer(
    show, render, pp, cameras, defs, framings, poses_of, leap, neutral, log
):
    """The framings through their shots' cameras: each pose's figure with the shot's turn, and
    its outline from a probe through the shot's camera."""
    figs, outs = {}, {}
    for name in framings:
        cam = cameras[name]
        # at_ease gives every framing its head and its ground, whether or not the framing shows it
        keys = list(dict.fromkeys([stage_poses.HEAD_POSE, *poses_of[name]])) + (
            list(stage_poses.LEAP) if leap and name == "full" else []
        )
        # the body's turn of the shot, for the figures and the probes
        pp.tab.evaluate(
            "(v) => window.portrait.view(v)",
            {
                "cx": 0,
                "cy": 1,
                "cz": 0,
                "halfH": 1,
                "width": PROBE_SIZE,
                "height": PROBE_SIZE,
                "turn": cam.turn,
            },
        )
        figs[name] = {pose: show(pose) for pose in keys}
        outs[name] = {}
        for pose in keys:
            t = time.time()
            f = figs[name][pose]
            ref = f if pose in stage_poses.POSES else figs[name][stage_poses.HEAD_POSE]
            H = ref["crown"][1] - ref["chin"][1]
            camera = shot_camera(cam, f, eyes_of=ref)
            top, bottom = ref["crown"][1] + 2.4 * H, ref["ground"] - 0.1 * H
            depth = project_points(camera, ref["eyes"], 1.0)[0, 2]
            focal = 0.9 * PROBE_SIZE * depth / (top - bottom)
            mid = [ref["eyes"][0], (top + bottom) / 2, ref["eyes"][2]]
            u, v, _ = project_points(camera, mid, focal)[0]
            pp_xy = (PROBE_SIZE / 2 - u, PROBE_SIZE / 2 - v)
            show(pose)
            a = render(
                {
                    "camera": {
                        "eye": [float(x) for x in camera["eye"]],
                        "pitch": camera["pitch"],
                        "focal": float(focal),
                        "px": float(pp_xy[0]),
                        "py": float(pp_xy[1]),
                    },
                    "turn": cam.turn,
                    "halfH": (top - bottom) / 2,
                    "width": PROBE_SIZE,
                    "height": PROBE_SIZE,
                },
                neutral,
                1,
            )[..., 3]
            outs[name][pose] = probe_outline(a, camera, focal, pp_xy, ref, H)
            if (a[:, 0] > 24).any() or (a[:, -1] > 24).any() or (a[0] > 24).any():
                log(f"{name} {pose}: the outline reaches the probe's edge")
            log(
                f"{name} {pose}: posed and outlined through {cam.shot}'s camera ({time.time() - t:.1f} s)"
            )
    return ShotFramer(
        figs, outs, defs, poses_of, cameras, stage_poses.LEAP if leap else ()
    )


class Clock:
    """The seconds a pack's render spends in each phase, for its log and its result. The writing
    runs on a pool of threads beside the page, so its seconds overlap the page's."""

    def __init__(self):
        import threading

        self.seconds: dict[str, float] = {}
        self.lock = threading.Lock()

    def add(self, phase: str, seconds: float) -> None:
        with self.lock:
            self.seconds[phase] = self.seconds.get(phase, 0.0) + seconds

    def time(self, phase: str):
        import contextlib

        @contextlib.contextmanager
        def timer():
            t = time.time()
            try:
                yield
            finally:
                self.add(phase, time.time() - t)

        return timer()

    def report(self) -> dict:
        with self.lock:
            return {k: round(v, 1) for k, v in sorted(self.seconds.items())}


# The writing of the layers runs on this many threads while the page renders the next layer, with
# at most this many layers waiting.
WRITERS = 4
WAITING = 6
# The pass files' WebP effort (0 to 6): 4 encodes about seven times faster than 6 at the same size.
PASS_METHOD = 4


class Lit:
    """The light passes of a pack being written (:mod:`opensculptboy.portrait_pack.light`): the rig
    on the page, each layer's passes in linear light, their scales (from the first body layer), the
    layer itself as the passes' sum through the display pass, the pass files beside each layer, and
    the reproduction of each layer by its stored passes' sum. Every layer, with passes or without,
    is written on a pool of threads while the page renders the next one."""

    def __init__(
        self, pp, passes, rig, rig_name, quality, pass_quality, log, clock=None
    ):
        from concurrent.futures import ThreadPoolExecutor
        import threading

        self.pp = pp
        self.passes = list(passes or [])
        self.on = bool(self.passes)
        self.rig, self.rig_name = rig, rig_name
        self.log = log
        self.quality = quality or DEFAULT_QUALITY
        self.pass_quality = pass_quality
        self.scales: dict[str, float] = {}
        self.face: dict | None = None
        self.files = 0
        self.seconds = 0.0
        self.clock = clock or Clock()
        self.repro = light_passes.Reproduction()
        self.pool = ThreadPoolExecutor(max_workers=WRITERS)
        self.waiting: list = []
        self.lock = threading.Lock()
        self.display = (
            pp.tab.evaluate("(r) => window.portrait.light(r)", rig) if rig else None
        )
        if self.on and self.display.get("tone") != "agx":
            raise PortraitError("The light passes need the page's AgX tone curve.")

    def render(self, view: dict, actions: dict, samples: int) -> dict:
        """The passes of one picture: pass name to (H, W, 4) float32, straight linear colour
        times the exposure, and the coverage."""
        import base64

        t = time.time()
        self.pp.tab.evaluate("(v) => window.portrait.view(v)", view)
        region = view.get("region")
        w, h = (region["w"], region["h"]) if region else (view["width"], view["height"])
        out = {}
        for name in self.passes:
            self.pp.tab.evaluate("(n) => window.portrait.pass(n)", name)
            b64 = self.pp.tab.evaluate(
                "([a, s]) => window.portrait.linear(a, s)", [actions, samples]
            )
            out[name] = (
                np.frombuffer(base64.b64decode(b64), dtype="<f2")
                .reshape(h, w, 4)
                .astype(np.float32)
            )
        self.pp.tab.evaluate("() => window.portrait.pass('all')")
        self.seconds += time.time() - t
        self.clock.add("render passes", time.time() - t)
        return out

    def compose(self, passes: dict) -> np.ndarray:
        """The layer as the sum of its passes through the display pass (RGBA bytes): the passes add
        up to the whole rig in linear light, so the sum gives the layer without a picture of its
        own. The coverage comes from the passes, and the colour is zero where it is zero."""
        t = time.time()
        first = next(iter(passes.values()))
        total = sum(p[..., :3] for p in passes.values())
        alpha = np.round(np.clip(first[..., 3], 0, 1) * 255).astype(np.uint8)
        d = self.display
        rgb = light_passes.display8(total, d["saturation"], d["contrast"])
        rgb[alpha == 0] = 0
        self.clock.add("compose", time.time() - t)
        return np.ascontiguousarray(np.dstack([rgb, alpha]))

    def ensure_scales(self, passes: dict, alpha: np.ndarray, face=None, stored=None):
        """The pack's scales from its first body layer's passes. ``stored``: the scales an earlier
        run stored its passes with (resume), which the pack keeps when they agree within 2 %.
        Returns False when they disagree, so the earlier run's files cannot be kept."""
        if self.scales:
            return True
        self.scales = {
            n: round(light_passes.pass_scale(p[..., :3], alpha), 5)
            for n, p in passes.items()
        }
        agree = True
        if stored is not None:
            agree = all(
                n in stored and abs(stored[n] / self.scales[n] - 1) <= 0.02
                for n in self.scales
            )
            if agree:
                self.scales = {n: round(stored[n], 5) for n in self.scales}
        self.face = light_passes.face_light(
            {n: p[..., :3] for n, p in passes.items()}, alpha, face
        )
        self.log(
            "light passes: scales "
            + ", ".join(f"{n} {s:.3f}" for n, s in self.scales.items())
            + (
                ""
                if stored is None
                else " (the earlier run's)"
                if agree
                else " (the earlier run's differ: its layers are rendered again)"
            )
        )
        return agree

    def submit(self, folder, stem, kind, rgba, quality, passes=None, ext=".webp"):
        """Write a layer and its passes (pass name to linear colour, cut to the layer) on the pool."""
        rgba = np.ascontiguousarray(rgba)
        if passes is not None:
            passes = {n: np.ascontiguousarray(p[..., :3]) for n, p in passes.items()}

        def task():
            t = time.time()
            save_layer(folder / f"{stem}{ext}", rgba, quality)
            if self.on and passes is not None:
                self._write(folder, stem, kind, passes, rgba)
            self.clock.add("write (on the pool)", time.time() - t)

        with self.clock.time("wait for the pool"):
            while len(self.waiting) >= WAITING:
                self.waiting.pop(0).result()
        self.waiting.append(self.pool.submit(task))

    def drain(self):
        """Wait for every layer on the pool; an error in a writer is raised here."""
        with self.clock.time("wait for the pool"):
            while self.waiting:
                self.waiting.pop(0).result()

    def close(self):
        try:
            self.drain()
        finally:
            self.pool.shutdown(wait=True)

    def _write(self, folder, stem, kind, passes, rgba):
        from PIL import Image

        alpha = rgba[..., 3]
        q = self.pass_quality or min(
            100, self.quality["face" if kind in ("face", "front") else "body"] + 4
        )
        total = np.zeros(rgba.shape[:2] + (3,), np.float32)
        for name, rgb in passes.items():
            enc = light_passes.encode_pass(rgb, alpha, self.scales[name])
            path = folder / f"{stem}.{name}.webp"
            Image.fromarray(np.ascontiguousarray(enc), "RGBA").save(
                path,
                format="WEBP",
                quality=int(q),
                method=PASS_METHOD,
                alpha_quality=100,
                exact=True,
            )
            with self.lock:
                self.files += 1
            back = np.asarray(Image.open(path).convert("RGBA"))
            total += light_passes.decode_pass(back, self.scales[name])
        d = self.display
        summed = light_passes.display8(total, d["saturation"], d["contrast"])
        stored = np.asarray(Image.open(folder / f"{stem}.webp").convert("RGBA"))
        self.repro.add(kind, rgba, summed, stored)

    def have(self, folder, stem, size) -> bool:
        """Every pass file of a layer decodes whole at its size (True without passes)."""
        return all(
            layer_ok(folder / f"{stem}.{name}.webp", size) for name in self.passes
        )

    def manifest_passes(self) -> list:
        return [
            {"name": n, "scale": round(self.scales.get(n, 1.0), 5)} for n in self.passes
        ]

    def manifest_light(self) -> dict:
        return light_passes.describe(self.rig_name, self.rig, self.display, self.face)

    def result(self) -> dict:
        return {
            "rig": self.rig_name,
            "passes": self.manifest_passes(),
            "reproduction": self.repro.report(),
            "seconds": round(self.seconds, 1),
        }


def stored_scales(folder: pathlib.Path, stem: str, passes: dict, alpha: np.ndarray):
    """The scales an earlier run stored a layer's pass files with, estimated from the same layer's
    passes rendered again: the median ratio of the linear light to the stored light over the lit,
    covered pixels. None when a pass file is missing or cut short."""
    from PIL import Image

    out = {}
    cov = alpha >= 250
    for name, p in passes.items():
        path = folder / f"{stem}.{name}.webp"
        if not layer_ok(path, (alpha.shape[1], alpha.shape[0])):
            return None
        with Image.open(path) as img:
            stored = light_passes.decode_pass(np.asarray(img.convert("RGBA")), 1.0)
        a = p[..., :3].max(-1)
        b = stored.max(-1)
        use = cov & (b > 0.1) & (a > 0)
        if use.sum() < 100:
            return None
        out[name] = float(np.median(a[use] / b[use]))
    return out


def write_layered_pack(
    out,
    card=None,
    expressions=None,
    poses=None,
    framings=None,
    leap=True,
    licence: str | None = None,
    companion: str | None = None,
    samples: int = 16,
    fmt: str = "webp",
    quality: dict | None = None,
    scale: float = 1.0,
    page=None,
    log=None,
    preview: bool = False,
    poses_file=None,
    shots_file=None,
    passes=None,
    light_rig: str | None = None,
    pass_quality: int | None = None,
    resume: bool = False,
    framing_samples: dict | None = None,
) -> dict:
    """
    Render the layered portrait pack of a character card (None for the default character) into
    the folder ``out`` and check it. ``poses`` and ``framings`` choose from
    :data:`opensculptboy.portrait_pack.poses.POSES` and :data:`FRAMINGS`; ``scale`` multiplies
    every framing's pixels per head (below 1 for quick trials); ``fmt`` is ``"webp"`` (lossy
    with alpha) or ``"png"``. With ``preview``, only the body layers are written, for a look at
    the poses. ``poses_file`` is the companion's shared pose file
    (:mod:`opensculptboy.portrait_pack.shared_poses`; its copy in data/portrait/ by default), and
    ``shots_file`` the stage's shots (:mod:`opensculptboy.portrait_pack.shots`): with it,
    each framing renders through the camera of the shot it serves. ``passes`` (a set of
    :data:`opensculptboy.portrait_pack.light.PASS_SETS`) adds the light passes of every layer, lit by
    ``light_rig`` (:data:`opensculptboy.portrait_pack.light.RIGS`; the stage's rig with passes, the
    page's own preset without), each stored at ``pass_quality`` (the layer's quality and 4 more by
    default). With ``resume``, the folder may hold an unfinished pack of the same settings: each
    layer whose file and pass files decode whole at their size is kept (the first body layer is
    rendered again, since its passes set the scales), and the rest is rendered. Returns the
    manifest, the check's report, the time taken and, with passes, how closely the passes' sum
    reproduces each layer (``light``). ``framing_samples`` gives some framings fewer samples than
    ``samples`` (:data:`FRAMING_SAMPLES` by default; a framing never takes more than ``samples``).
    The result's ``timing`` gives the seconds of each phase.
    """
    from opensculptboy.portrait_pack import shared_poses
    from opensculptboy.portrait_pack import shots as shot_specs

    log = log or (lambda m: None)
    quality = {**DEFAULT_QUALITY, **(quality or {})}
    if fmt not in FORMATS:
        raise ValueError(f"The format {fmt!r} is not one of {FORMATS}.")
    expressions = list(expressions or C.EXPRESSIONS)
    if expressions[0] != C.REQUIRED_EXPRESSION:
        raise ValueError("The expression 'neutral' is required, and comes first.")
    poses = list(poses or stage_poses.POSES)
    unknown = [p for p in poses if p not in stage_poses.POSES]
    if unknown:
        raise ValueError(
            f"Unknown poses {unknown}; the poses are {list(stage_poses.POSES)}."
        )
    if stage_poses.HEAD_POSE in poses:
        poses.remove(stage_poses.HEAD_POSE)
    poses.insert(0, stage_poses.HEAD_POSE)
    shared = shared_poses.load(poses_file)
    wanted_shared = [stage_poses.shared_name(p) for p in poses] + (
        [stage_poses.shared_name(k) for k in stage_poses.LEAP] if leap else []
    )
    lacking = [p for p in wanted_shared if p not in shared.bones]
    if lacking:
        raise ValueError(f"The pose file {shared.source} lacks the poses {lacking}.")
    cameras = shot_specs.load(shots_file) if shots_file else None
    passes = list(passes) if passes else None
    if passes and tuple(passes) not in light_passes.PASS_SETS:
        raise ValueError(
            f"The passes {passes} are not one of {light_passes.PASS_SETS}."
        )
    if passes and fmt != "webp":
        raise ValueError("The light passes are WebP files: take --format webp.")
    rig_name = light_rig or ("stage" if passes else None)
    light_rig_spec = light_passes.rig(rig_name) if rig_name else None
    if framings:
        framings = list(framings)
    elif cameras is not None:
        framings = [f for f in FRAMINGS if f in cameras]
    else:
        framings = list(BASE_FRAMINGS)
    unknown = [f for f in framings if f not in FRAMINGS]
    if unknown:
        raise ValueError(
            f"Unknown framings {unknown}; the framings are {list(FRAMINGS)}."
        )
    if cameras is None:
        alone = [f for f in framings if f not in BASE_FRAMINGS]
        if alone:
            raise ValueError(
                f"The framings {alone} render through their shots' cameras: give --shots-file."
            )
    defs = {n: dict(FRAMINGS[n], scale=FRAMINGS[n]["scale"] * scale) for n in framings}
    poses_of = {n: [p for p in poses if p in FRAMING_POSES[n]] for n in framings}
    empty = [n for n in framings if not poses_of[n]]
    if empty:
        raise ValueError(f"The framings {empty} hold none of the poses {poses}.")
    # the face sets: the main set, which every pose that keeps at_ease's head shares, and a set
    # of its own for each pose that turns its head, with the expressions that suit it
    own = {
        p: [x for x in expressions if x in shared.faces.get(p, ())] or ["neutral"]
        for p in poses
        if p in shared.own_head
    }
    main_xs = {
        n: [
            x
            for x in expressions
            if x in FRAMING_EXPRESSIONS.get(n, expressions)
            and FRAMING_LEAVES.get(n, {}).get(x) not in framings
        ]
        for n in framings
    }
    framing_samples = FRAMING_SAMPLES if framing_samples is None else framing_samples

    def samples_at(name: str) -> int:
        return max(1, min(samples, int(framing_samples.get(name, samples))))

    card_name = None
    if card is not None:
        data = json.loads(pathlib.Path(card).read_text(encoding="utf-8"))
        card_name = (
            data.get("name")
            if data.get("name") not in (None, "", "character")
            else None
        )
    if licence is None:
        if card is not None:
            raise ValueError(
                "Cards hold no licence of their pictures: give the licence of this character's "
                "images with --licence."
            )
        licence = DEFAULT_LICENCE
    if companion is None and card_name:
        companion = _one_line(card_name, C.TEXT_LIMITS["companion"])
    for key, value in (("licence", licence), ("companion", companion)):
        if value is None:
            continue
        why = text_problem(value)
        if why or not 1 <= len(value) <= C.TEXT_LIMITS[key]:
            raise ValueError(f"The {key} {why or 'is too long'}.")
    folder = prepare_layered_folder(out, keep=resume)
    kept = {"layers": 0, "on": resume, "removed": 0}
    clock = Clock()
    ext = "." + fmt
    q_body = quality["body"] if fmt == "webp" else None
    q_face = quality["face"] if fmt == "webp" else None
    t0 = time.time()
    with contextlib.ExitStack() as stack:
        pp = stack.enter_context(PortraitPage(card=card, page=page, log=log))
        info = pp.begin(PROBE_SIZE, PROBE_SIZE)
        lit = Lit(
            pp,
            passes,
            light_rig_spec,
            rig_name,
            quality if fmt == "webp" else None,
            pass_quality,
            log,
            clock,
        )
        # the writers finish (and raise any error) before the page closes
        stack.callback(lit.close)
        known = set(info["actions"])
        unknown = sorted({k for *_, a in pack_faces(expressions) for k in a} - known)
        if unknown:
            raise PortraitError(f"The page does not know the facial actions {unknown}.")
        bones = pp.tab.evaluate("() => window.portrait.bones()")
        if not bones:
            raise PortraitError("The page holds no rig for the poses.")
        neutral = frame_actions("neutral", "closed", "open")
        # the card's rig and its T-pose turns, for the shared poses
        rig = shared_poses.card_rig(card)
        # the joints the manifest records, by the page's names for them
        humanoid = shared_poses.bone_map()
        joint_names = [n for n in LC.JOINT_NAMES if n in humanoid or n in POINTS]
        page_names = [n if n in POINTS else humanoid[n] for n in joint_names]

        def done(stem: str, size) -> bool:
            """With resume: the layer and its passes are already in the folder."""
            return (
                kept["on"]
                and layer_ok(folder / f"{stem}{ext}", size)
                and lit.have(folder, stem, size)
            )

        previous = {}
        if resume and (folder / C.MANIFEST).is_file():
            try:
                previous = json.loads((folder / C.MANIFEST).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous = {}

        def previous_face(name: str, pose: str):
            """The face rectangle an earlier run's manifest gives a framing's main set (at_ease) or
            a pose's own set, when the folder holds one."""
            f = (previous.get("framings") or {}).get(name) or {}
            r = (
                f.get("face")
                if pose == stage_poses.HEAD_POSE
                else ((f.get("sets") or {}).get(pose) or {}).get("face")
            )
            ok = isinstance(r, dict) and all(
                isinstance(r.get(k), int) for k in ("x", "y", "width", "height")
            )
            return r if ok else None

        def joints_now() -> list:
            """The joints of the picture just rendered, in its pixels: x, y for each name."""
            got = pp.tab.evaluate("(n) => window.portrait.project(n)", page_names)
            missing = [n for n, p in zip(joint_names, page_names) if p not in got]
            if missing:
                raise PortraitError(f"The page has no joints {missing}.")
            return [round(float(v), 1) for p in page_names for v in got[p][:2]]

        if cameras is not None:
            lacking = [f for f in framings if f not in cameras]
            if lacking:
                raise ValueError(
                    f"The shots file renders no shot for the framings {lacking}."
                )

        def show(pose: str) -> dict:
            pp_pose = shared_poses.page_pose(
                shared,
                stage_poses.shared_name(pose),
                bones,
                rig.labels,
                rig.parents,
                rig.turns,
            )
            return pp.tab.evaluate(
                "([q, r, l]) => window.portrait.pose(q, r, l)",
                [pp_pose["q"], pp_pose["root"], pp_pose["label"]],
            )

        def render(view: dict, actions: dict, n: int) -> np.ndarray:
            pp.tab.evaluate("(v) => window.portrait.view(v)", view)
            region = view.get("region")
            pp.size = (
                (region["w"], region["h"])
                if region
                else (view["width"], view["height"])
            )
            return pp.frame(actions, n)

        def picture(view: dict, actions: dict, n: int):
            """A layer's picture and its passes: with passes, the layer is their sum, so the
            page renders the passes alone; without, the layer itself (passes None)."""
            if lit.on:
                passes_now = lit.render(view, actions, n)
                return lit.compose(passes_now), passes_now
            with clock.time("render layers"):
                return render(view, actions, n), None

        # each pose: the figure and its outline
        t_probes = time.time()
        figures, probes = {}, {}
        for pose in (
            []
            if cameras is not None
            else poses + (list(stage_poses.LEAP) if leap and "full" in framings else [])
        ):
            t = time.time()
            f = show(pose)
            figures[pose] = f
            H = f["crown"][1] - f["chin"][1]
            top = f["crown"][1] + 2.4 * H
            bottom = f["ground"] - 0.1 * H
            half = (top - bottom) / 2
            view = {
                "cx": f["eyes"][0],
                "cy": (top + bottom) / 2,
                "cz": f["eyes"][2],
                "halfH": half,
                "width": PROBE_SIZE,
                "height": PROBE_SIZE,
            }
            a = render(view, neutral, 1)[..., 3]
            x0, x1 = _mask_rows(a)
            m = 2 * half / PROBE_SIZE
            ys = top - (np.arange(PROBE_SIZE) + 0.5) * m
            left = view["cx"] - half
            probes[pose] = {
                "y": ys,
                "x0": left + x0 * m,
                "x1": left + (x1 + 1) * m,
                "covered": x0 >= 0,
                "edge": bool(
                    (a[:, 0] > 24).any() or (a[:, -1] > 24).any() or (a[0] > 24).any()
                ),
            }
            if probes[pose]["edge"]:
                log(f"{pose}: the outline reaches the probe's edge")
            log(f"{pose}: posed and outlined ({time.time() - t:.1f} s)")
        if cameras is None:
            framer = Framer(
                figures, probes, defs, poses_of, stage_poses.LEAP if leap else ()
            )
        else:
            framer = _shot_framer(
                show, render, pp, cameras, defs, framings, poses_of, leap, neutral, log
            )
        if leap and "full" in framings:
            # the leap's own pixels per head, with the full framing's camera
            framer.alias("leap", "full", LEAP_SCALE * scale)
        clock.add("probes and framings", time.time() - t_probes)
        for n, g in framer.geo.items():
            log(
                f"framing {n}: {g['width']} x {g['height']} pixels, {g['scale']:.0f} pixels per head"
                + (f", {samples_at(n)} samples" if n in framings else "")
            )

        # the half-open eyes, measured in at_ease
        show(stage_poses.HEAD_POSE)
        halves = {} if preview else pp.half_blinks(expressions)
        faces = pack_faces(expressions, halves)

        def face_rect(name: str, pose: str, xs: list) -> tuple:
            """The face rectangle of a face set at a framing (the pose shown): the box of every
            pixel that one of the set's faces changes, and the face's own pixels as weights."""
            with clock.time("face rectangles"):
                return _face_rect(name, pose, xs)

        def _face_rect(name: str, pose: str, xs: list) -> tuple:
            g = framer.geo[name]
            head = framer.head_rows(name)
            view = framer.view(name, pose)
            unit = head["chin"] - head["top"]
            hx0 = max(0, head["x"] - int(0.75 * unit))
            hy0 = max(0, head["top"] - int(0.15 * unit))
            hx1 = min(g["width"], head["x"] + int(0.75 * unit))
            hy1 = min(g["height"], head["chin"] + int(0.4 * unit))
            hreg = {"x": hx0, "y": hy0, "w": hx1 - hx0, "h": hy1 - hy0}
            # each expression's open mouth with closed eyes against the neutral face, rendered
            # around the head at one pass; the first picture after a pose settles the page
            render({**view, "region": hreg}, neutral, 1)
            base = render({**view, "region": hreg}, neutral, 1)
            # what differs between two pictures of the same face is noise (fur and hair drawn
            # with a dither), which no face changes
            noise = _grow(
                changed_pixels(base, render({**view, "region": hreg}, neutral, 1)), 3
            )
            box = None
            changed = np.zeros(base.shape[:2], bool)
            for x in xs:
                for mouth, eyes in (("open", "closed"), ("closed", "closed")):
                    other = render(
                        {**view, "region": hreg},
                        frame_actions(x, mouth, eyes, halves.get(x)),
                        1,
                    )
                    changed |= changed_pixels(base, other) & ~noise
                    b = changed_box(base, np.where(noise[..., None], base, other))
                    if b:
                        box = (
                            b
                            if box is None
                            else (
                                min(box[0], b[0]),
                                min(box[1], b[1]),
                                max(box[2], b[2]),
                                max(box[3], b[3]),
                            )
                        )
            if box is None:
                raise PortraitError(f"No face of {pose} changes the frame at {name}.")
            mg = FACE_MARGIN * unit
            face = snap_rect(
                hx0 + box[0] - mg,
                hy0 + box[1] - mg,
                hx0 + box[2] + mg,
                hy0 + box[3] + mg,
                g["width"],
                g["height"],
            )
            # resume: the earlier run's rectangle when it lies within a few pixels (the noise of the
            # measure moves the box by a step of the grid), so its face layers stay valid
            old = previous_face(name, pose)
            if (
                old is not None
                and all(
                    abs(old[k] - face[k]) <= 3 * GRID
                    for k in ("x", "y", "width", "height")
                )
                and old["x"] + old["width"] <= g["width"]
                and old["y"] + old["height"] <= g["height"]
            ):
                face = dict(old)
            full = np.zeros((g["height"], g["width"]), bool)
            full[hy0:hy1, hx0:hx1] = changed
            mask = soft_mask(
                full[
                    face["y"] : face["y"] + face["height"],
                    face["x"] : face["x"] + face["width"],
                ],
                max(2, int(round(MASK_GROW * unit))),
                max(1, int(round(MASK_SOFT * unit))),
            )
            return face, mask

        def write_faces(
            name: str, pose: str, face: dict, mask, xs: list, stem: str
        ) -> int:
            """A face set's layers, rendered in its pose around the face rectangle."""
            g = framer.geo[name]
            show(pose)
            view = framer.view(name, pose)
            px0 = max(0, face["x"] - FACE_PAD)
            py0 = max(0, face["y"] - FACE_PAD)
            px1 = min(g["width"], face["x"] + face["width"] + FACE_PAD)
            py1 = min(g["height"], face["y"] + face["height"] + FACE_PAD)
            reg = {"x": px0, "y": py0, "w": px1 - px0, "h": py1 - py0}
            t = time.time()
            n = 0
            ns = samples_at(name)
            for x, mouth, eyes, actions in faces:
                if x not in xs:
                    continue
                if done(
                    f"face.{stem}.{x}.{mouth}.{eyes}", (face["width"], face["height"])
                ):
                    kept["layers"] += 1
                    continue
                rgba, whole = picture({**view, "region": reg}, actions, ns)
                ox, oy = face["x"] - px0, face["y"] - py0
                cut = (slice(oy, oy + face["height"]), slice(ox, ox + face["width"]))
                patch = rgba[cut].copy()
                patch[..., 3] = np.round(patch[..., 3] * mask).astype(np.uint8)
                patch[patch[..., 3] == 0] = 0
                lit.submit(
                    folder,
                    f"face.{stem}.{x}.{mouth}.{eyes}",
                    "face",
                    patch,
                    q_face,
                    None if whole is None else {k: w[cut] for k, w in whole.items()},
                    ext,
                )
                n += 1
            clock.add("faces", time.time() - t)
            log(f"{n} faces of {stem} at {ns} samples ({time.time() - t:.1f} s)")
            return n

        manifest_framings = {}
        manifest_poses = {p: {"framings": [], "front": []} for p in poses}
        body_joints: dict = {}
        leap_joints: dict = {}
        written = 0
        for name in framings:
            g = framer.geo[name]
            head = framer.head_rows(name)
            main_poses = [p for p in poses_of[name] if p not in own]
            # the main face set, rendered in at_ease, where a pose of the framing keeps its head
            face = mask = None
            if main_poses and not preview:
                show(stage_poses.HEAD_POSE)
                face, mask = face_rect(name, stage_poses.HEAD_POSE, main_xs[name])
            sets = {}
            if not preview:
                for pose in poses_of[name]:
                    if pose in own:
                        show(pose)
                        sets[pose] = (*face_rect(name, pose, own[pose]), own[pose])
            # the body layers, and the at_ease face rectangle for the front layers
            base_crop = None
            if face is not None and stage_poses.HEAD_POSE not in poses_of[name]:
                show(stage_poses.HEAD_POSE)
                with clock.time("render layers"):
                    rgba = render(
                        framer.view(name, stage_poses.HEAD_POSE),
                        neutral,
                        samples_at(name),
                    )
                base_crop = rgba[
                    face["y"] : face["y"] + face["height"],
                    face["x"] : face["x"] + face["width"],
                ].copy()
            for pose in poses_of[name]:
                t = time.time()
                show(pose)
                view = framer.view(name, pose)
                body_passes = None
                # the first body layer's passes set the pack's scales, so it is rendered again
                if done(f"body.{pose}.{name}", (g["width"], g["height"])) and (
                    not lit.on or lit.scales
                ):
                    pp.tab.evaluate("(v) => window.portrait.view(v)", view)
                    rgba = load_layer(folder / f"body.{pose}.{name}{ext}")
                    kept["layers"] += 1
                else:
                    first = lit.on and not lit.scales
                    rgba, body_passes = picture(view, neutral, samples_at(name))
                    if first:
                        # resume: the earlier run's scales, from this layer's pass files
                        old = (
                            stored_scales(
                                folder, f"body.{pose}.{name}", body_passes, rgba[..., 3]
                            )
                            if kept["on"]
                            else None
                        )
                        agree = lit.ensure_scales(
                            body_passes, rgba[..., 3], face, old if kept["on"] else None
                        )
                        if kept["on"] and (old is None or not agree):
                            kept["on"] = False
                            log(
                                "resume: the earlier run's scales cannot be read back or "
                                "differ; every layer is rendered"
                            )
                    lit.submit(
                        folder,
                        f"body.{pose}.{name}",
                        "body",
                        rgba,
                        q_body,
                        body_passes,
                        ext,
                    )
                    written += 1
                body_joints.setdefault(pose, {})[name] = joints_now()
                manifest_poses[pose]["framings"].append(name)
                if face is not None and pose not in own:
                    crop = rgba[
                        face["y"] : face["y"] + face["height"],
                        face["x"] : face["x"] + face["width"],
                    ].copy()
                    if pose == stage_poses.HEAD_POSE:
                        base_crop = crop
                    elif base_crop is not None and done(
                        f"front.{pose}.{name}", (face["width"], face["height"])
                    ):
                        kept["layers"] += 1
                        manifest_poses[pose]["front"].append(name)
                    elif base_crop is not None:
                        m = front_mask(crop, base_crop, mask)
                        if m is not None and body_passes is None and lit.on:
                            # a kept body layer: its passes again, for the front's
                            body_passes = lit.render(view, neutral, samples_at(name))
                        if m is not None:
                            front = crop.copy()
                            front[..., 3] = (
                                front[..., 3].astype(np.uint16) * m // 255
                            ).astype(np.uint8)
                            front[front[..., 3] == 0] = 0
                            cut = (
                                slice(face["y"], face["y"] + face["height"]),
                                slice(face["x"], face["x"] + face["width"]),
                            )
                            lit.submit(
                                folder,
                                f"front.{pose}.{name}",
                                "front",
                                front,
                                q_face,
                                None
                                if body_passes is None
                                else {k: b[cut] for k, b in body_passes.items()},
                                ext,
                            )
                            written += 1
                            manifest_poses[pose]["front"].append(name)
                clock.add("bodies and fronts", time.time() - t)
                log(f"body.{pose}.{name} ({time.time() - t:.1f} s)")
            # the face layers: the main set in at_ease, each own set in its pose
            if face is not None:
                written += write_faces(
                    name, stage_poses.HEAD_POSE, face, mask, main_xs[name], name
                )
            for pose, (rect, m, xs) in sets.items():
                written += write_faces(name, pose, rect, m, xs, f"{name}.{pose}")
            entry = {
                "frame": {"width": g["width"], "height": g["height"]},
                "head": head,
            }
            if cameras is not None:
                entry["camera"] = cameras[name].manifest()
            if face is not None:
                entry["face"] = face
                if main_xs[name] != expressions:
                    entry["expressions"] = main_xs[name]
            if sets:
                entry["sets"] = {
                    p: {"face": rect, "expressions": xs}
                    for p, (rect, _, xs) in sets.items()
                }
            if name == "full":
                entry["ground"] = int(
                    round(
                        (g["top"] * framer.H + framer.ground_below)
                        * g["scale"]
                        / framer.H
                    )
                )
            manifest_framings[name] = entry
        # the leap: the key poses with the full framing's camera of at_ease, the face of joy
        leap_keys = []
        leap_geo = None
        if leap and "full" in framings:
            t = time.time()
            view = framer.view("leap", stage_poses.HEAD_POSE)
            g_leap = framer.geo["leap"]
            leap_geo = {
                "frame": {"width": g_leap["width"], "height": g_leap["height"]},
                "head": framer.head_rows("leap"),
            }
            for key in stage_poses.LEAP:
                show(key)
                mouth = "open" if key in ("rise", "peak") else "half"
                if done(f"leap.{key}", (g_leap["width"], g_leap["height"])):
                    pp.tab.evaluate("(v) => window.portrait.view(v)", view)
                    leap_joints[key] = joints_now()
                    leap_keys.append(key)
                    kept["layers"] += 1
                    continue
                rgba, ps = picture(
                    view,
                    frame_actions(
                        "joy" if "joy" in expressions else "neutral",
                        mouth,
                        "open",
                        halves.get("joy"),
                    ),
                    samples,
                )
                leap_joints[key] = joints_now()
                lit.submit(folder, f"leap.{key}", "leap", rgba, q_body, ps, ext)
                written += 1
                leap_keys.append(key)
            clock.add("leap", time.time() - t)
            log(
                f"the leap: {len(leap_keys)} frames at {g_leap['scale']:.0f} pixels per head"
            )
        pp.end()
        renderer = pp.renderer
        if pp.errors:
            raise PortraitError(f"The page reported errors: {pp.errors[:3]}")

    # what the model adds above the crown and to the sides of the shoulders, in heads, for the
    # app's envelopes of the figure
    rows, x0, x1, cov = framer._rel(stage_poses.HEAD_POSE)
    above = float(max(0.0, -rows[cov].min())) if cov.any() else 0.0
    manifest = {"schema": LC.SCHEMA_V2}
    if companion:
        manifest["companion"] = companion
    manifest.update(
        format=fmt,
        expressions=expressions,
        framings=manifest_framings,
        poses={
            p: {k: v for k, v in d.items() if v or k == "framings"}
            for p, d in manifest_poses.items()
            if d["framings"]
        },
        extra={"above": round(min(3.0, above), 3), "width": 0},
        generator={"name": "OpenSculptBoy", "version": _version()},
        licence=licence,
        provenance=provenance(card_name),
    )
    if leap_keys:
        manifest["leap"] = {"framing": "full", "frames": leap_keys, **(leap_geo or {})}
    if body_joints:
        manifest["joints"] = {
            "names": joint_names,
            "poses": body_joints,
            **({"leap": leap_joints} if leap_joints else {}),
        }
    if lit.on:
        manifest["passes"] = lit.manifest_passes()
        manifest["light"] = lit.manifest_light()
    if preview:
        manifest["preview"] = True
    (folder / C.MANIFEST).write_text(manifest_text(manifest), encoding="utf-8")
    if resume:
        # the earlier run's layers that this pack no longer names (an expression a framing no
        # longer holds, a layer of another size) leave the folder
        wanted = set(LC.wanted_layers(manifest)) | {C.MANIFEST}
        for f in folder.iterdir():
            if f.name not in wanted and LC.is_layer_name(f.name):
                f.unlink()
                kept["removed"] += 1
    timing = {**clock.report(), "total": round(time.time() - t0, 1)}
    log("timing: " + ", ".join(f"{k} {v:.0f} s" for k, v in timing.items()))
    report = (
        C.check_pack(folder)
        if not preview
        else {
            "pass": True,
            "problems": [],
            "frames": written,
            "bytes": sum(p.stat().st_size for p in folder.iterdir()),
        }
    )
    return {
        "manifest": manifest,
        "check": report,
        "seconds": round(time.time() - t0, 1),
        "samples": samples,
        "renderer": renderer,
        "half_blinks": halves,
        "files": written + 1 + lit.files,
        "timing": timing,
        **({"kept": kept["layers"], "removed": kept["removed"]} if resume else {}),
        **({"light": lit.result()} if lit.on else {}),
    }


def manifest_text(manifest: dict) -> str:
    """The manifest as JSON with an indent of 2, each list of numbers on one line (the joints)."""
    import re

    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    number_list = re.compile(r"\[\s*(-?[\d.e+-]+(?:,\s*-?[\d.e+-]+)*)\s*\]")
    return (
        number_list.sub(
            lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]",
            text,
        )
        + "\n"
    )


def prepare_layered_folder(out, keep: bool = False) -> pathlib.Path:
    """The pack's folder, emptied of a previous layered pack (kept with ``keep``, for a run that
    resumes it); refuses a folder that holds anything else."""
    out = pathlib.Path(out)
    if out.exists() and not out.is_dir():
        raise ValueError(f"{out} is a file; name a folder for the pack.")
    if out.is_dir() and any(out.iterdir()):
        for p in out.iterdir():
            if not p.is_file() or not (
                p.name == C.MANIFEST or LC.is_layer_name(p.name)
            ):
                raise ValueError(
                    f"{out} holds files that are not a portrait pack ({p.name}); "
                    "name an empty folder or a previous pack."
                )
        for p in out.iterdir():
            if not keep:
                p.unlink()
    out.mkdir(parents=True, exist_ok=True)
    return out

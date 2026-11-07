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

import json
import pathlib
import time

import numpy as np

from opensculptboy.portrait_pack import contract as C
from opensculptboy.portrait_pack import layered_contract as LC
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
# (None: the ground), and whether raised hands lift the frame's top. The close framing holds the
# close-up and the thought (the head and the chest); the waist framing the medium close-up; the
# thigh framing the two-shot and the low, far shot; the full framing the whole body and the leap.
# The scales cover a window of 1440 x 900 pixels at device scale 2: a close-up with the face at
# 45 % of the height asks for about 810 pixels per head, a medium close-up at 120 % about 655,
# a two-shot at 100 % about 320 and the full shot about 150.
FRAMINGS = {
    "close": dict(scale=810, above=0.5, below=2.2, hands=False),
    "waist": dict(scale=660, above=0.5, below=3.6, hands=True),
    "thigh": dict(scale=330, above=0.5, below=6.0, hands=True),
    "full": dict(scale=160, above=0.5, below=None, hands=True),
}
# The poses of each framing: raised hands leave the close framing.
CLOSE_POSES = (
    "at_ease",
    "relaxed",
    "hands_behind_head",
    "chin",
    "scratch_head",
    "arms_crossed",
    "hand_on_chest",
)
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
FORMATS = ("webp", "png")


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
    """The face's own pixels as weights 0 to 1: the changed pixels grown by ``grow`` pixels, their
    holes filled across each row and column, and the edge softened over ``soft`` pixels."""
    m = _grow(changed, grow)
    # fill the holes between the eyes, the nose and the mouth: what lies between covered pixels on
    # both its row and its column
    rows = np.maximum.accumulate(m, 1) & np.maximum.accumulate(m[:, ::-1], 1)[:, ::-1]
    cols = np.maximum.accumulate(m, 0) & np.maximum.accumulate(m[::-1], 0)[::-1]
    m = m | (rows & cols)
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
        if name == "full":
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
        }


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
) -> dict:
    """
    Render the layered portrait pack of a character card (None for the default character) into
    the folder ``out`` and check it. ``poses`` and ``framings`` choose from
    :data:`opensculptboy.portrait_pack.poses.POSES` and :data:`FRAMINGS`; ``scale`` multiplies
    every framing's pixels per head (below 1 for quick trials); ``fmt`` is ``"webp"`` (lossy
    with alpha) or ``"png"``. With ``preview``, only the body layers are written, for a look at
    the poses. Returns the manifest, the check's report and the time taken.
    """
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
    if stage_poses.HEAD_POSE not in poses:
        poses.insert(0, stage_poses.HEAD_POSE)
    framings = list(framings or FRAMINGS)
    unknown = [f for f in framings if f not in FRAMINGS]
    if unknown:
        raise ValueError(
            f"Unknown framings {unknown}; the framings are {list(FRAMINGS)}."
        )
    defs = {n: dict(FRAMINGS[n], scale=FRAMINGS[n]["scale"] * scale) for n in framings}
    poses_of = {
        n: [p for p in poses if n != "close" or p in CLOSE_POSES] for n in framings
    }
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
    folder = prepare_layered_folder(out)
    ext = "." + fmt
    q_body = quality["body"] if fmt == "webp" else None
    q_face = quality["face"] if fmt == "webp" else None
    t0 = time.time()
    with PortraitPage(card=card, page=page, log=log) as pp:
        info = pp.begin(PROBE_SIZE, PROBE_SIZE)
        known = set(info["actions"])
        unknown = sorted({k for *_, a in pack_faces(expressions) for k in a} - known)
        if unknown:
            raise PortraitError(f"The page does not know the facial actions {unknown}.")
        bones = pp.tab.evaluate("() => window.portrait.bones()")
        if not bones:
            raise PortraitError("The page holds no rig for the poses.")
        neutral = frame_actions("neutral", "closed", "open")

        def show(pose: str) -> dict:
            pp_pose = stage_poses.page_pose(pose, bones)
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

        # each pose: the figure and its outline
        figures, probes = {}, {}
        for pose in poses + (
            list(stage_poses.LEAP) if leap and "full" in framings else []
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
        framer = Framer(
            figures, probes, defs, poses_of, stage_poses.LEAP if leap else ()
        )
        for n, g in framer.geo.items():
            log(
                f"framing {n}: {g['width']} x {g['height']} pixels, {g['scale']:.0f} pixels per head"
            )

        # the half-open eyes, measured in at_ease
        show(stage_poses.HEAD_POSE)
        halves = {} if preview else pp.half_blinks(expressions)
        faces = pack_faces(expressions, halves)

        manifest_framings = {}
        manifest_poses = {p: {"framings": [], "front": []} for p in poses}
        written = 0
        for name in framings:
            g = framer.geo[name]
            head = framer.head_rows(name)
            # the face's changing rectangle, from at_ease: each expression's open mouth with
            # closed eyes against the neutral face, rendered around the head at one pass
            show(stage_poses.HEAD_POSE)
            view0 = framer.view(name, stage_poses.HEAD_POSE)
            unit = head["chin"] - head["top"]
            hx0 = max(0, head["x"] - int(0.75 * unit))
            hy0 = max(0, head["top"] - int(0.15 * unit))
            hx1 = min(g["width"], head["x"] + int(0.75 * unit))
            hy1 = min(g["height"], head["chin"] + int(0.4 * unit))
            hreg = {"x": hx0, "y": hy0, "w": hx1 - hx0, "h": hy1 - hy0}
            face = None
            if not preview:
                render(
                    {**view0, "region": hreg}, neutral, 1
                )  # the first picture after a pose settles the page
                base = render({**view0, "region": hreg}, neutral, 1)
                box = None
                changed = np.zeros(base.shape[:2], bool)
                for x in expressions:
                    for mouth, eyes in (("open", "closed"), ("closed", "closed")):
                        other = render(
                            {**view0, "region": hreg},
                            frame_actions(x, mouth, eyes, halves.get(x)),
                            1,
                        )
                        changed |= changed_pixels(base, other)
                        b = changed_box(base, other)
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
                    raise PortraitError("No face changes the frame.")
                mg = FACE_MARGIN * unit
                face = snap_rect(
                    hx0 + box[0] - mg,
                    hy0 + box[1] - mg,
                    hx0 + box[2] + mg,
                    hy0 + box[3] + mg,
                    g["width"],
                    g["height"],
                )
                # the face's own pixels inside the rectangle: what any face changes, grown and softened
                full = np.zeros((g["height"], g["width"]), bool)
                full[hy0:hy1, hx0:hx1] = changed
                face_mask = soft_mask(
                    full[
                        face["y"] : face["y"] + face["height"],
                        face["x"] : face["x"] + face["width"],
                    ],
                    max(2, int(round(MASK_GROW * unit))),
                    max(1, int(round(MASK_SOFT * unit))),
                )
            # the body layers, and the at_ease face rectangle for the front layers
            base_crop = None
            for pose in poses_of[name]:
                t = time.time()
                show(pose)
                rgba = render(framer.view(name, pose), neutral, samples)
                save_layer(folder / f"body.{pose}.{name}{ext}", rgba, q_body)
                written += 1
                manifest_poses[pose]["framings"].append(name)
                if face is not None:
                    crop = rgba[
                        face["y"] : face["y"] + face["height"],
                        face["x"] : face["x"] + face["width"],
                    ].copy()
                    if pose == stage_poses.HEAD_POSE:
                        base_crop = crop
                    elif base_crop is not None:
                        m = front_mask(crop, base_crop, face_mask)
                        if m is not None:
                            front = crop.copy()
                            front[..., 3] = (
                                front[..., 3].astype(np.uint16) * m // 255
                            ).astype(np.uint8)
                            front[front[..., 3] == 0] = 0
                            save_layer(
                                folder / f"front.{pose}.{name}{ext}", front, q_face
                            )
                            written += 1
                            manifest_poses[pose]["front"].append(name)
                log(f"body.{pose}.{name} ({time.time() - t:.1f} s)")
            # the face layers, rendered in at_ease around the rectangle
            if face is not None:
                show(stage_poses.HEAD_POSE)
                px0 = max(0, face["x"] - FACE_PAD)
                py0 = max(0, face["y"] - FACE_PAD)
                px1 = min(g["width"], face["x"] + face["width"] + FACE_PAD)
                py1 = min(g["height"], face["y"] + face["height"] + FACE_PAD)
                reg = {"x": px0, "y": py0, "w": px1 - px0, "h": py1 - py0}
                t = time.time()
                for x, mouth, eyes, actions in faces:
                    rgba = render({**view0, "region": reg}, actions, samples)
                    ox, oy = face["x"] - px0, face["y"] - py0
                    patch = rgba[
                        oy : oy + face["height"], ox : ox + face["width"]
                    ].copy()
                    patch[..., 3] = np.round(patch[..., 3] * face_mask).astype(np.uint8)
                    patch[patch[..., 3] == 0] = 0
                    save_layer(
                        folder / f"face.{name}.{x}.{mouth}.{eyes}{ext}",
                        np.ascontiguousarray(patch),
                        q_face,
                    )
                    written += 1
                log(f"{len(faces)} faces at {name} ({time.time() - t:.1f} s)")
            manifest_framings[name] = {
                "frame": {"width": g["width"], "height": g["height"]},
                "head": head,
                **({"face": face} if face is not None else {}),
            }
            if name == "full":
                manifest_framings[name]["ground"] = int(
                    round(
                        (g["top"] * framer.H + framer.ground_below)
                        * g["scale"]
                        / framer.H
                    )
                )
        # the leap: the key poses with the full framing's camera of at_ease, the face of joy
        leap_keys = []
        if leap and "full" in framings:
            view = framer.view("full", stage_poses.HEAD_POSE)
            for key in stage_poses.LEAP:
                show(key)
                mouth = "open" if key in ("rise", "peak") else "half"
                rgba = render(
                    view,
                    frame_actions(
                        "joy" if "joy" in expressions else "neutral",
                        mouth,
                        "open",
                        halves.get("joy"),
                    ),
                    samples,
                )
                save_layer(folder / f"leap.{key}{ext}", rgba, q_body)
                written += 1
                leap_keys.append(key)
            log(f"the leap: {len(leap_keys)} frames")
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
        manifest["leap"] = {"framing": "full", "frames": leap_keys}
    if preview:
        manifest["preview"] = True
    (folder / C.MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
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
        "files": written + 1,
    }


def prepare_layered_folder(out) -> pathlib.Path:
    """The pack's folder, emptied of a previous layered pack; refuses a folder that holds
    anything else."""
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
            p.unlink()
    out.mkdir(parents=True, exist_ok=True)
    return out

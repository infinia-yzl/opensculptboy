# OpenSculptBoy
# Apache License, Version 2.0
"""
The stage's shots and their cameras (``squire.shots/1``), for the cameras of the layered pack's
framings.

Each shot of the stage names a camera: the lens (the focal length in millimetres, full-frame
equivalent over the window's height), the body point the camera sits level with (the eyes, the
chest or the hips) and how far above it, its tilt (degrees, positive looks down) and the body's
turn toward the open side of the frame (degrees, positive toward the figure's left). The file's
``renders`` names, for each framing of the layered pack, the shot whose camera renders its
stills, so a framing is rendered for the shot it serves.

The stage measures the body in head heights of its own: the distance from the hips to the eyes
divided by 3.5 (:data:`STAGE_HEAD_UNITS`), and the shot's ``span`` and ``share`` say how many of
them the window's height spans. With the lens's field of view, that sets the camera's distance:
the window's height at the eyes' distance is that many head heights.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib

FORMAT = "squire.shots/1"
HEIGHTS = ("eyes", "chest", "hips")
# The stage's head height: the hips-to-eyes distance divided by this.
STAGE_HEAD_UNITS = 3.5


class ShotsError(ValueError):
    """A shots file that breaks the format."""


@dataclasses.dataclass(frozen=True)
class ShotCamera:
    """The camera of one shot."""

    shot: str
    lens: float  # millimetres, full-frame equivalent over the window's height
    below_eyes: float  # the camera's height, in the stage's head heights below the eyes
    tilt: float  # degrees, positive looks down
    turn: float  # degrees, the body's turn
    window: float  # the stage's head heights that the window's height spans
    sensor: float = 24.0

    @property
    def fov(self) -> float:
        """The vertical field of view over the window, in degrees."""
        return math.degrees(2 * math.atan(self.sensor / 2 / self.lens))

    def distance(self, head_unit: float) -> float:
        """The camera's distance from the eyes (metres) for a body whose stage head height is
        ``head_unit`` metres: the window's height there spans :attr:`window` head heights."""
        return self.window * head_unit / (2 * math.tan(math.radians(self.fov) / 2))


def _middle(v) -> float:
    """A tunable's value for intensity 2 (the middle one), or the value itself."""
    if isinstance(v, list):
        if len(v) != 3:
            raise ShotsError("A tunable takes one number or three.")
        return float(v[1])
    return float(v)


def window_heads(spec: dict) -> float:
    """The stage's head heights that the window's height spans in a shot, at intensity 2: the
    shot's span over its share, or for a shot that fits the whole body, its span over the share
    of the height between its top and its bottom."""
    if spec.get("fit") == "body":
        return float(spec["span"]) / (
            float(spec.get("bottom", 0.97)) - _middle(spec["top"])
        )
    return float(spec["span"]) / _middle(spec["share"])


def parse(data: dict, source: str = "") -> dict:
    """The camera of each framing that the file renders: {framing: ShotCamera}."""
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ShotsError(f"The shots file's format is not {FORMAT}.")
    shots = data.get("shots")
    renders = data.get("renders")
    heights = data.get("heights", {})
    if not isinstance(shots, dict) or not isinstance(renders, dict):
        raise ShotsError("The shots file needs 'shots' and 'renders'.")
    sensor = float(data.get("sensor", 24))
    out = {}
    for framing, shot in renders.items():
        spec = shots.get(shot)
        if not isinstance(spec, dict):
            raise ShotsError(
                f"The framing {framing} renders for an unknown shot {shot!r}."
            )
        if spec.get("stills") != framing:
            raise ShotsError(
                f"The framing {framing} renders for {shot}, whose stills are {spec.get('stills')!r}."
            )
        c = spec.get("camera")
        if not isinstance(c, dict):
            raise ShotsError(f"The shot {shot} has no camera.")
        if c.get("height") not in HEIGHTS or c["height"] not in heights:
            raise ShotsError(
                f"The shot {shot}'s camera height must be one of {HEIGHTS}."
            )
        try:
            lens, above = float(c["lens"]), float(c.get("above", 0))
            tilt, turn = float(c.get("tilt", 0)), float(c.get("turn", 0))
        except (KeyError, TypeError, ValueError) as error:
            raise ShotsError(f"The shot {shot}'s camera is broken: {error}") from error
        if not 14 <= lens <= 300 or abs(tilt) > 20 or abs(turn) > 60:
            raise ShotsError(f"The shot {shot}'s lens, tilt or turn is out of range.")
        out[framing] = ShotCamera(
            shot=shot,
            lens=lens,
            below_eyes=float(heights[c["height"]]) - above,
            tilt=tilt,
            turn=turn,
            window=window_heads(spec),
            sensor=sensor,
        )
    return out


def load(path) -> dict:
    """The camera of each framing, from a shots file."""
    path = pathlib.Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ShotsError(f"{path} is not JSON: {error}") from error
    return parse(data, str(path))

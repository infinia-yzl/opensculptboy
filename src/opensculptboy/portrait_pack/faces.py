# OpenSculptBoy
# Apache License, Version 2.0
"""
The faces of a portrait pack as weights of Anny's 52 ARKit facial actions.

Each expression mixes the VRM 1.0 presets that the VRM export writes
(``data/vrm/expressions.json``, the look-at presets as the eye-look actions of
:data:`opensculptboy.export.vrm_tables.LOOK_ACTIONS`), scaled by its weights, with facial actions
of its own on top. The mouth and the eyes of each frame then add to that face: the vowels ``aa``,
``ih`` and ``oh`` for the mouth, and ``eyeBlinkLeft`` and ``eyeBlinkRight`` for the eyes. Since
the frames set the actions themselves, no preset can block the blink or the vowels, and every one
of the nine frames of an expression shows its own mouth and eyes. Each action is clamped to
[0, 1].
"""

from __future__ import annotations

import dataclasses
import functools
import json
import pathlib

from opensculptboy.portrait_pack.contract import EXPRESSIONS, EYES, MOUTHS

EXPRESSIONS_FILE = (
    pathlib.Path(__file__).resolve().parents[1] / "data" / "vrm" / "expressions.json"
)
BLINKS = ("eyeBlinkLeft", "eyeBlinkRight")
WIDE = ("eyeWideLeft", "eyeWideRight")


@dataclasses.dataclass(frozen=True)
class Expression:
    """One expression: VRM presets with their weights, facial actions added on top, and the
    blink of its open eyes (``lids``, added to the blink that its presets hold)."""

    presets: dict
    actions: dict = dataclasses.field(default_factory=dict)
    lids: float = 0.0


# The expressions, tuned by eye on the default character from the suggested weights of the
# contract. ``joy`` closes its eyes a little in the smile; ``worried`` raises the inner brows on
# top of a light ``sad``; ``drowsy`` has heavy lids on its open eyes. The model has no blush, so
# ``embarrassed`` holds the presets alone.
EXPRESSION_MIXES = {
    "neutral": Expression({}),
    "smile": Expression({"happy": 0.6}),
    "joy": Expression({"happy": 1.0}, lids=0.25),
    "thinking": Expression(
        {"relaxed": 0.4, "lookUp": 0.35},
        {"mouthPressLeft": 0.2, "mouthPressRight": 0.2},
    ),
    "surprised": Expression({"surprised": 1.0}),
    "worried": Expression(
        {"sad": 0.45},
        {
            "browInnerUp": 0.45,
            "eyeWideLeft": 0.15,
            "eyeWideRight": 0.15,
            "mouthPressLeft": 0.25,
            "mouthPressRight": 0.25,
        },
    ),
    "sad": Expression({"sad": 0.9}),
    "stern": Expression({"angry": 0.6}),
    "embarrassed": Expression({"happy": 0.35, "lookDown": 0.3}, {"browInnerUp": 0.2}),
    "drowsy": Expression({"relaxed": 0.8}, lids=0.4),
}
# The mouths as vowels with their weights.
MOUTH_MIXES = {
    "closed": {},
    "half": {"aa": 0.4, "ih": 0.2},
    "open": {"aa": 0.9, "oh": 0.2},
}
# How far each eyes state closes the lids from the expression's own open eyes toward a full
# blink: "half" lies half way, so it differs from "open" and from "closed" for every
# expression, also one with heavy lids of its own.
EYE_CLOSE = {"open": 0.0, "half": 0.5, "closed": 1.0}


@functools.lru_cache(maxsize=1)
def preset_mixes() -> dict:
    """The VRM 1.0 presets as mixes of facial actions: the presets of ``expressions.json`` and
    the look-at presets ``lookUp``, ``lookDown``."""
    from opensculptboy.export.vrm_tables import LOOK_ACTIONS

    data = json.loads(EXPRESSIONS_FILE.read_text(encoding="utf-8"))
    mixes = {name: dict(p["mix"]) for name, p in data["presets"].items()}
    for name in ("lookUp", "lookDown"):
        prefix = LOOK_ACTIONS[name]
        mixes[name] = {prefix + "Left": 1.0, prefix + "Right": 1.0}
    return mixes


def _add(into: dict, mix: dict, weight: float = 1.0) -> None:
    for k, v in mix.items():
        into[k] = into.get(k, 0.0) + weight * v


def frame_actions(expression: str, mouth: str, eyes: str) -> dict:
    """The facial actions of one frame: the action names with weights in (0, 1], every other
    action at 0."""
    if expression not in EXPRESSION_MIXES:
        raise ValueError(
            f"Unknown expression {expression!r}; the expressions are {list(EXPRESSIONS)}."
        )
    if mouth not in MOUTHS:
        raise ValueError(f"Unknown mouth {mouth!r}; the mouths are {list(MOUTHS)}.")
    if eyes not in EYES:
        raise ValueError(f"Unknown eyes {eyes!r}; the eyes are {list(EYES)}.")
    mixes = preset_mixes()
    x = EXPRESSION_MIXES[expression]
    w: dict = {}
    for name, weight in x.presets.items():
        _add(w, mixes[name], weight)
    _add(w, x.actions)
    for name, weight in MOUTH_MIXES[mouth].items():
        _add(w, mixes[name], weight)
    close = EYE_CLOSE[eyes]
    for blink in BLINKS:
        own = min(1.0, w.get(blink, 0.0) + x.lids)
        w[blink] = own + close * (1.0 - own)
    # the blink takes over the widened eyes as it closes them
    for wide in WIDE:
        if wide in w:
            w[wide] *= 1.0 - close
    out = {}
    for k in sorted(w):
        v = round(min(1.0, max(0.0, w[k])), 4)
        if v > 0:
            out[k] = v
    return out


def pack_faces(expressions=EXPRESSIONS) -> list[tuple[str, str, str, dict]]:
    """Every frame of these expressions as ``(expression, mouth, eyes, actions)``, in the order
    of the contract."""
    return [
        (x, m, e, frame_actions(x, m, e))
        for x in expressions
        for m in MOUTHS
        for e in EYES
    ]

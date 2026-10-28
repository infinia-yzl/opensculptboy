# OpenSculptBoy
# Apache License, Version 2.0
"""
The faces of the expression review: Anny's 52 facial actions at weights 1.0 and 0.5, the VRM
presets and visemes of ``data/vrm/expressions.json``, 20 common expressions mixed from the facial
actions, and mixes whose shapes collide. :mod:`opensculptboy.expressions.authoring.blender_faces`
renders them.

Usage::

    python -m opensculptboy.expressions.authoring.faces --out faces.json [--mouth-poses a,b,...]

``--mouth-poses`` adds a face per mouth pose of a fitted card (the morph target
``mouth.<pose>`` of its GLB file at weight 1).
"""

from __future__ import annotations

import argparse
import json
import pathlib

from anny.models.facial_actions import FACIAL_ACTION_LABELS

EXPRESSIONS_FILE = (
    pathlib.Path(__file__).resolve().parents[2] / "data" / "vrm" / "expressions.json"
)


def both(name: str, weight: float) -> dict[str, float]:
    """The left and right actions of ``name`` at one weight."""
    return {name + "Left": weight, name + "Right": weight}


def mix(*parts: dict, **single: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in parts:
        out.update(part)
    out.update(single)
    return {k: v for k, v in out.items() if v}


# Common expressions as mixes of the facial actions (Left and Right are the character's own
# sides): a first draft for the expression library, judged on the review grids.
EXPRESSIONS: dict[str, dict[str, float]] = {
    "grin": mix(
        both("mouthSmile", 1.0),
        both("cheekSquint", 0.6),
        both("mouthUpperUp", 0.35),
        both("mouthLowerDown", 0.3),
        both("mouthStretch", 0.2),
        both("eyeSquint", 0.3),
        jawOpen=0.08,
    ),
    "laugh": mix(
        both("mouthSmile", 1.0),
        both("cheekSquint", 0.8),
        both("eyeSquint", 0.6),
        both("mouthUpperUp", 0.3),
        both("mouthLowerDown", 0.4),
        jawOpen=0.45,
        browInnerUp=0.2,
    ),
    "smirk": mix(
        mouthSmileLeft=0.7,
        mouthDimpleLeft=0.3,
        cheekSquintLeft=0.3,
        mouthLeft=0.15,
        eyeSquintLeft=0.2,
        mouthPressRight=0.2,
    ),
    "wink": mix(
        eyeBlinkLeft=1.0,
        cheekSquintLeft=0.5,
        mouthSmileLeft=0.45,
        browDownLeft=0.3,
        mouthSmileRight=0.15,
    ),
    "worried": mix(
        both("mouthStretch", 0.3),
        both("mouthFrown", 0.2),
        both("mouthPress", 0.2),
        both("eyeWide", 0.2),
        browInnerUp=0.9,
    ),
    "afraid": mix(
        both("browOuterUp", 0.5),
        both("eyeWide", 1.0),
        both("mouthStretch", 0.7),
        both("mouthLowerDown", 0.3),
        browInnerUp=1.0,
        jawOpen=0.25,
    ),
    "disgusted": mix(
        both("noseSneer", 0.9),
        both("mouthUpperUp", 0.6),
        both("browDown", 0.5),
        both("cheekSquint", 0.4),
        both("mouthFrown", 0.3),
        both("eyeSquint", 0.3),
        mouthShrugLower=0.3,
    ),
    "contemptuous": mix(
        mouthSmileRight=0.4,
        mouthDimpleRight=0.5,
        noseSneerRight=0.2,
        mouthRight=0.2,
        eyeSquintRight=0.15,
        eyeSquintLeft=0.1,
    ),
    "sulking": mix(
        both("mouthFrown", 0.6),
        both("browDown", 0.4),
        both("mouthPress", 0.3),
        both("eyeLookDown", 0.3),
        mouthShrugLower=0.6,
        browInnerUp=0.3,
        mouthPucker=0.2,
    ),
    "crying": mix(
        both("browDown", 0.3),
        both("eyeSquint", 0.6),
        both("eyeBlink", 0.3),
        both("mouthFrown", 0.8),
        both("mouthStretch", 0.5),
        both("mouthLowerDown", 0.3),
        both("cheekSquint", 0.4),
        browInnerUp=1.0,
        jawOpen=0.2,
        mouthShrugLower=0.4,
    ),
    "curious": mix(
        both("eyeWide", 0.3),
        browOuterUpLeft=0.7,
        browInnerUp=0.4,
        mouthPucker=0.2,
        browOuterUpRight=0.2,
    ),
    "doubtful": mix(
        both("mouthPress", 0.4),
        browDownRight=0.5,
        browOuterUpLeft=0.6,
        eyeSquintRight=0.3,
        mouthLeft=0.3,
        mouthShrugLower=0.3,
    ),
    "focused": mix(
        both("browDown", 0.5),
        both("eyeSquint", 0.5),
        both("mouthPress", 0.4),
        mouthRollLower=0.1,
    ),
    "confused": mix(
        browInnerUp=0.5,
        browDownLeft=0.5,
        browOuterUpRight=0.5,
        mouthLeft=0.3,
        mouthFrownLeft=0.2,
        eyeSquintLeft=0.3,
        jawLeft=0.1,
    ),
    "sleepy": mix(
        both("eyeBlink", 0.6), both("eyeLookDown", 0.2), browInnerUp=0.2, jawOpen=0.06
    ),
    "pout": mix(
        both("mouthFrown", 0.3),
        mouthPucker=0.8,
        mouthShrugLower=0.5,
        mouthShrugUpper=0.3,
        browInnerUp=0.3,
    ),
    "puffed_cheeks": mix(cheekPuff=1.0, mouthPucker=0.2),
    "tongue_out": mix(both("mouthLowerDown", 0.2), jawOpen=0.35, tongueOut=1.0),
    "raised_brow": mix(
        browOuterUpLeft=1.0, browInnerUp=0.3, eyeWideLeft=0.3, browDownRight=0.2
    ),
    "shy": mix(
        both("mouthSmile", 0.35),
        both("mouthPress", 0.3),
        both("eyeLookDown", 0.6),
        both("cheekSquint", 0.2),
        both("eyeBlink", 0.2),
        browInnerUp=0.4,
    ),
}

# Mixes whose shapes collide or need each other (candidates for correctives)
COMBOS: dict[str, dict[str, float]] = {
    "jawOpen+mouthClose": mix(jawOpen=0.5, mouthClose=0.5),
    "jawOpen+tongueOut": mix(jawOpen=0.4, tongueOut=1.0),
    "jawOpen+mouthFunnel": mix(jawOpen=0.6, mouthFunnel=1.0),
    "smile+squint": mix(
        both("mouthSmile", 1.0), both("eyeSquint", 1.0), both("cheekSquint", 1.0)
    ),
    "blink+squint": mix(both("eyeBlink", 1.0), both("eyeSquint", 1.0)),
    "jawOpen+smile": mix(both("mouthSmile", 1.0), jawOpen=0.6),
}


def review_faces(mouth_poses=()) -> list[dict]:
    """The faces of the review, each ``{"name": ..., "mix": {target: weight}}``: ``neutral``,
    ``a1/<action>``, ``a05/<action>``, ``p/<preset>``, ``v/<viseme>``, ``x/<expression>``,
    ``c/<combination>`` and ``m/<mouth pose>``."""
    vrm = json.loads(EXPRESSIONS_FILE.read_text(encoding="utf-8"))
    faces = [{"name": "neutral", "mix": {}}]
    for weight, tag in ((1.0, "a1"), (0.5, "a05")):
        faces += [
            {"name": f"{tag}/{n}", "mix": {n: weight}} for n in FACIAL_ACTION_LABELS
        ]
    faces += [{"name": f"p/{n}", "mix": e["mix"]} for n, e in vrm["presets"].items()]
    faces += [{"name": f"v/{n}", "mix": e["mix"]} for n, e in vrm["customs"].items()]
    faces += [{"name": f"x/{n}", "mix": m} for n, m in EXPRESSIONS.items()]
    faces += [{"name": f"c/{n}", "mix": m} for n, m in COMBOS.items()]
    faces += [{"name": f"m/{p}", "mix": {f"mouth.{p}": 1.0}} for p in mouth_poses]
    return faces


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="write the faces of the expression review"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--mouth-poses", default="")
    args = parser.parse_args(argv)
    poses = [p for p in args.mouth_poses.split(",") if p]
    faces = review_faces(poses)
    pathlib.Path(args.out).write_text(
        json.dumps({"faces": faces}, indent=1), encoding="utf-8"
    )
    print(len(faces), "faces")


if __name__ == "__main__":
    main()

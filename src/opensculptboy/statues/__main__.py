# OpenSculptBoy
# Apache License, Version 2.0
"""
``python -m opensculptboy.statues SCAN --out FOLDER``: a statue scan as an appearance handoff
folder (see :mod:`opensculptboy.statues.handoff`). It prints one JSON result: the frame it
found, the triangles per role and the landmarks in the handoff's frame.
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m opensculptboy.statues",
        description="Write an appearance handoff folder from a statue scan: upright, "
        "facing +Z, the plinth cut off, scaled to a stature and the masked pieces taken out.",
    )
    parser.add_argument("scan", help="the scan file (STL, OBJ, PLY or GLB)")
    parser.add_argument("--out", required=True, help="the handoff folder to write")
    parser.add_argument(
        "--stature",
        type=float,
        default=1.8,
        help="the figure's height above the plinth in metres (default 1.8)",
    )
    parser.add_argument("--up", help="the scan's up axis (+x ... -z; default: guessed)")
    parser.add_argument("--front", help="the scan's front axis (default: guessed)")
    parser.add_argument(
        "--masks",
        help="a JSON file: a list of masks (see opensculptboy.statues.scan.Mask)",
    )
    parser.add_argument(
        "--landmarks",
        help='a JSON file {"landmarks": {name: [x, y, z]}, "frame": "handoff" or "scan"}',
    )
    parser.add_argument(
        "--keep-plinth", action="store_true", help="do not cut the plinth off"
    )
    parser.add_argument("--source", help="the scan's page")
    parser.add_argument("--licence", help="the scan's licence (an SPDX name)")
    parser.add_argument("--credit", help="the credit line")
    parser.add_argument("--title", help="the statue's name")
    parser.add_argument(
        "--lost",
        default="",
        help="the parts the statue lost, by commas: hand.l, forearm.r, arm.l, foot.r, "
        "lower_leg.l, leg.r, nose",
    )
    args = parser.parse_args(argv)

    from opensculptboy.fit.handoff import GLTF_TO_ANNY
    from opensculptboy.statues.handoff import scan_to_handoff

    masks = []
    if args.masks:
        with open(args.masks, encoding="utf-8") as f:
            masks = json.load(f)
    landmarks, frame = None, "handoff"
    if args.landmarks:
        with open(args.landmarks, encoding="utf-8") as f:
            data = json.load(f)
        landmarks = data.get("landmarks", data)
        frame = data.get("frame", "handoff")
    result = scan_to_handoff(
        args.scan,
        args.out,
        stature=args.stature,
        up=args.up,
        front=args.front,
        masks=masks,
        landmarks=landmarks,
        landmark_frame=frame,
        plinth=not args.keep_plinth,
        source=args.source,
        licence=args.licence,
        credit=args.credit,
        title=args.title,
        lost=[n for n in args.lost.split(",") if n.strip()],
    )
    print(
        json.dumps(
            {
                "folder": str(result.folder),
                "scale": result.scale,
                "up": result.up.tolist(),
                "front": result.front.tolist(),
                "up_confidence": round(result.up_confidence, 4),
                "front_score": round(result.front_score, 4),
                "plinth": result.plinth,
                "triangles": result.triangles,
                "landmarks": {
                    k: [round(float(x), 5) for x in v @ GLTF_TO_ANNY]
                    for k, v in result.landmarks.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

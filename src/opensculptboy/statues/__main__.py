# OpenSculptBoy
# Apache License, Version 2.0
"""
The statue commands, each printing one JSON result:

- ``python -m opensculptboy.statues run CONFIG... --scans DIR --out DIR``: the pipeline of
  :mod:`opensculptboy.statues.pipeline` for each statue config (``--steps convert,preview``
  for a first look, the default runs every step).
- ``python -m opensculptboy.statues convert SCAN --out FOLDER``: a statue scan as an
  appearance handoff folder (see :mod:`opensculptboy.statues.handoff`), with the frame it
  found, the triangles per role and the landmarks in the handoff's frame.
- ``python -m opensculptboy.statues starter CONFIG --run DIR --out starters/ID.zip``: a fitted
  statue (the run folder of ``run``) as a finished starter of the web viewer (see
  :mod:`opensculptboy.statues.starter`).
"""

from __future__ import annotations

import argparse
import json
import sys

from opensculptboy.statues.pipeline import STEPS


def main(argv=None) -> int:
    words = sys.argv[1:] if argv is None else list(argv)
    if words[:1] == ["run"]:
        return run(words[1:])
    if words[:1] == ["starter"]:
        return starter(words[1:])
    if words[:1] == ["convert"]:
        words = words[1:]
    return convert(words)


def run(argv) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m opensculptboy.statues run",
        description="Run the statue pipeline (convert, preview, fit, measure) for each "
        "statue config (opensculptboy.statue/1).",
    )
    parser.add_argument("configs", nargs="+", help="statue config files (JSON)")
    parser.add_argument("--out", required=True, help="the folder of the results")
    parser.add_argument(
        "--scans", help="the folder of the scans (default: the config's)"
    )
    parser.add_argument(
        "--steps",
        default=",".join(STEPS),
        help=f"the steps to run, by commas (default {','.join(STEPS)})",
    )
    parser.add_argument("--detail", type=int, choices=(0, 1, 2), help="override detail")
    parser.add_argument(
        "--fit",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="a FitOptions field over the config's (VALUE in JSON), for variants",
    )
    parser.add_argument(
        "--variant",
        help="a name for the variant: its results go to <out>/<name>.<variant>",
    )
    parser.add_argument(
        "--hold",
        metavar="EYES,LIPS",
        help="the rings around the eyes' and the mouth's openings that keep anny's shape",
    )
    parser.add_argument(
        "--cut",
        metavar="RINGS,METRES",
        help="the band along the scan's cut by a hat or locks whose skin takes no data",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="a statue config field over the config's (VALUE in JSON), for variants",
    )
    parser.add_argument(
        "--from",
        dest="stage1",
        metavar="RUN",
        help="an earlier run's folder of the statue (<out>/<name>): take over its handoff "
        "and its sliders, pose and placement, and run the detail levels and the measures",
    )
    parser.add_argument(
        "--reconvert",
        action="store_true",
        help="with --from: convert the scan again (new masks) and prune it with the earlier "
        "run's first fit, its chains fitted again",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="the fit's progress lines"
    )
    args = parser.parse_args(argv)
    import dataclasses

    if args.stage1 and len(args.configs) > 1:
        parser.error("--from takes one config")
    if args.reconvert and not args.stage1:
        parser.error("--reconvert goes with --from")

    from opensculptboy.statues.pipeline import StatueConfig, run_statue

    results = []
    for path in args.configs:
        config = StatueConfig.load(path)
        if args.detail is not None:
            config = dataclasses.replace(config, detail=args.detail)
        if args.fit:
            fit = dict(config.fit)
            for item in args.fit:
                key, _, value = item.partition("=")
                fit[key.strip()] = json.loads(value)
            config = StatueConfig.from_dict(
                {**_as_dict(config), "fit": fit}, config.path
            )
        if args.hold:
            config = dataclasses.replace(
                config, hold=[int(x) for x in args.hold.split(",")]
            )
        if args.set:
            fields = _as_dict(config)
            for item in args.set:
                key, _, value = item.partition("=")
                fields[key.strip()] = json.loads(value)
            config = StatueConfig.from_dict(fields, config.path)
        if args.cut:
            rings, reach = args.cut.split(",")
            config = dataclasses.replace(config, cut=[int(rings), float(reach)])
        if args.variant:
            config = dataclasses.replace(config, name=f"{config.name}.{args.variant}")
        steps = [s for s in args.steps.split(",") if s]
        results.append(
            run_statue(
                config,
                args.out,
                args.scans,
                steps,
                verbose=args.verbose,
                stage1=args.stage1,
                reconvert=args.reconvert,
            )
        )
    print(json.dumps(results, indent=2, default=float))
    return 0


def starter(argv) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m opensculptboy.statues starter",
        description="A fitted statue as a finished starter of the web viewer: a .zip of "
        "its card (with the marble look and the scan's credit), its detail files, its "
        "pieces and a thumbnail.",
    )
    parser.add_argument("config", help="the statue config (JSON)")
    parser.add_argument(
        "--run", required=True, help="the --out folder of the statue's pipeline run"
    )
    parser.add_argument("--out", required=True, help="the .zip to write")
    parser.add_argument(
        "--levels", default="0,1,2", help="the detail levels to pack (default 0,1,2)"
    )
    parser.add_argument(
        "--piece", action="append", default=[], help="a file of a piece to pack"
    )
    parser.add_argument("--thumbnail", help="a PNG for the thumbnail (default: drawn)")
    parser.add_argument(
        "--props",
        help="the handoff folder with props.glb (default: the run's handoff)",
    )
    parser.add_argument(
        "--no-props", action="store_true", help="leave the props out of the zip"
    )
    parser.add_argument(
        "--variant", help="the variant of the run (<run>/<name>.<variant>)"
    )
    args = parser.parse_args(argv)
    import pathlib

    from opensculptboy.statues.pipeline import StatueConfig
    from opensculptboy.statues.starter import build_starter

    config = StatueConfig.load(args.config)
    name = config.name + (f".{args.variant}" if args.variant else "")
    levels = tuple(int(x) for x in args.levels.split(",") if x)
    result = build_starter(
        pathlib.Path(args.run) / name,
        config,
        args.out,
        levels,
        args.piece,
        args.thumbnail,
        props=args.props,
        with_props=not args.no_props,
        log=lambda m: print(m, file=sys.stderr),
    )
    print(json.dumps(result, indent=2))
    return 0


def _as_dict(config) -> dict:
    import dataclasses

    data = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
    data.pop("path")
    return data


def convert(argv) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m opensculptboy.statues convert",
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

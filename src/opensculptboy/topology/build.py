# OpenSculptBoy
# Apache License, Version 2.0
"""
Build the ``sculpt`` topology file from the patch modules.

    python -m opensculptboy.topology.build                     # the ENABLED patches, into the package data
    python -m opensculptboy.topology.build --with placeholder  # add patches that are off by default
    python -m opensculptboy.topology.build --only eye_corner --out /tmp/sculpt.safetensors

The modules of :mod:`opensculptboy.topology.patches` are composed in their ``ORDER``; the
result is deterministic and is written as ``sculpt.safetensors`` with its ``sculpt.json``.
"""

from __future__ import annotations

import argparse
import pathlib

from opensculptboy.topology.sculpt import DEFAULT_PATH, compose


def main(argv=None) -> pathlib.Path:
    parser = argparse.ArgumentParser(
        prog="python -m opensculptboy.topology.build",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument(
        "--with",
        dest="include",
        action="append",
        default=[],
        metavar="PATCH",
        help="add a patch that is off by default (repeatable)",
    )
    parser.add_argument(
        "--without",
        dest="exclude",
        action="append",
        default=[],
        metavar="PATCH",
        help="leave out a patch (repeatable)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="PATCH",
        help="use only these patches (repeatable)",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=DEFAULT_PATH,
        help=f"the .safetensors file to write (default {DEFAULT_PATH})",
    )
    args = parser.parse_args(argv)
    data = compose(names=args.only, include=args.include, exclude=args.exclude)
    path = data.write(args.out)
    for record in data.patches:
        a, b = record["vertices"]
        print(
            f"{record['name']}: {record['removed_faces']} faces removed, {b - a} vertices "
            f"and {record['faces'][1] - record['faces'][0]} faces added"
        )
    print(
        f"wrote {path} and {path.with_suffix('.json').name} ({len(data.patches)} patches)"
    )
    return path


if __name__ == "__main__":
    main()

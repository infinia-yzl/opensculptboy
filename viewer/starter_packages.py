# OpenSculptBoy
# Apache License, Version 2.0
"""
Pin the statue starters' packages in the page: write viewer/src/starter_packages.json and viewer/src/starter_thumbs.ts.

    python viewer/starter_packages.py FOLDER [--release TAG]

FOLDER holds the packages that `python -m opensculptboy.statues starter` wrote (`<statue>.zip`). The packages are too
large for the repository: they go out as files of a GitHub release (tag TAG, `starters-1` by default, or the tag of
the manifest written before), and the person puts them in the folder `starters/` beside the page. The manifest holds
each package's file name, size and SHA-256, which the page checks when it reads the package (viewer/src/starters.ts);
the New screen's picture of each statue is the package's own `thumbnail.png`, carried into the page as a WebP data
URL. Run it after a package changes, then build the page again (cd viewer && node build.mjs --reuse-data) and upload
the packages to the release.
"""

import argparse
import base64
import hashlib
import io
import json
import pathlib
import sys
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST = HERE / "src" / "starter_packages.json"
THUMBS = HERE / "src" / "starter_thumbs.ts"
REPOSITORY = "https://github.com/infinia-yzl/opensculptboy"
FORMAT = "opensculptboy/starter-packages@1"
# the statues of the New screen in their order (viewer/src/starters.ts names and describes them)
STATUES = ("doryphoros", "david_goliath")


def picture(png: bytes) -> str:
    """the package's picture as a data URL: WebP when Pillow is there, else the PNG itself"""
    try:
        from PIL import Image
    except ImportError:
        return "data:image/png;base64," + base64.b64encode(png).decode()
    out = io.BytesIO()
    Image.open(io.BytesIO(png)).save(out, "WEBP", quality=86, method=6)
    return "data:image/webp;base64," + base64.b64encode(out.getvalue()).decode()


def pin(path: pathlib.Path) -> dict:
    data = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        if "character.json" not in names:
            raise SystemExit(f"{path.name} holds no character.json")
        card = json.loads(z.read("character.json"))
        png = z.read("thumbnail.png") if "thumbnail.png" in names else None
    return {
        "entry": {
            "file": path.name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "card": card.get("name", path.stem),
        },
        "picture": picture(png) if png else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=pathlib.Path)
    parser.add_argument(
        "--release",
        default=None,
        help="the release tag (default: the manifest's, or starters-1)",
    )
    args = parser.parse_args(argv)
    old = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    tag = args.release or old.get("tag") or "starters-1"
    packages, pictures = {}, {}
    for statue in STATUES:
        path = args.folder / f"{statue}.zip"
        if not path.exists():
            print(
                f"{path.name} is missing; the New screen leaves {statue} out",
                file=sys.stderr,
            )
            continue
        p = pin(path)
        packages[statue] = p["entry"]
        if p["picture"]:
            pictures[statue] = p["picture"]
        print(
            statue, p["entry"]["bytes"], "bytes", p["entry"]["sha256"][:12], flush=True
        )
    manifest = {
        "format": FORMAT,
        "tag": tag,
        "release": f"{REPOSITORY}/releases/tag/{tag}",
        "download": f"{REPOSITORY}/releases/download/{tag}/",
        "packages": packages,
    }
    # LF line endings on every system, as the repository keeps them
    MANIFEST.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    lines = [f"  {k}: {json.dumps(v)}," for k, v in pictures.items()]
    THUMBS.write_bytes(
        (
            "// OpenSculptBoy\n// Apache License, Version 2.0\n//\n"
            "// The New screen's pictures of the statue starters: each package's own thumbnail.png (the\n"
            "// statue in its stone and pose), as WebP data URLs. Written by viewer/starter_packages.py\n"
            "// with the packages' pins (starter_packages.json); run it again after a package changes.\n\n"
            "export const STARTER_THUMBS: Record<string, string> = {\n"
            + "\n".join(lines)
            + "\n};\n"
        ).encode("utf-8")
    )
    print(
        "wrote",
        MANIFEST.name,
        "and",
        THUMBS.name,
        sum(len(u) for u in pictures.values()) // 1024,
        "KB of pictures",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

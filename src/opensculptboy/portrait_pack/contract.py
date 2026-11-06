# OpenSculptBoy
# Apache License, Version 2.0
"""
The contract of a portrait pack (``squire.portraits/1``) and its check.

A portrait pack is one flat folder that holds ``manifest.json`` and one PNG per frame, named
``<expression>.<mouth>.<eyes>.png``. An app that shows a character as a 2D portrait that talks
and blinks, as visual novels do, swaps the frames in place: the mouth while the text types, the
eyes for a blink, the expression for each message. :func:`check_pack` applies every rule of the
contract and gives a reason code for each failure, so the export checks its own output and a pack
can be checked before an app imports it. The check reads the files alone: it needs neither the
model nor a browser.
"""

from __future__ import annotations

import json
import pathlib
import struct
import unicodedata

SCHEMA = "squire.portraits/1"
MANIFEST = "manifest.json"

# The parts of a frame's name, each in the order of the contract.
EXPRESSIONS = (
    "neutral",
    "smile",
    "joy",
    "thinking",
    "surprised",
    "worried",
    "sad",
    "stern",
    "embarrassed",
    "drowsy",
)
MOUTHS = ("closed", "half", "open")
EYES = ("open", "half", "closed")
REQUIRED_EXPRESSION = "neutral"

# The limits of a pack. MAX_PACK_BYTES matches the consuming app's contract: a full pack of 90
# frames at the default size of 768 by 1024 pixels keeps within it, also for a card with its own
# textures (about 80 MiB).
MAX_FILES = 100
MAX_SIDE = 1024
MAX_MANIFEST_BYTES = 64 * 1024
MAX_PACK_BYTES = 128 * 1024 * 1024

# The fields of the manifest: the required ones, then the optional one.
REQUIRED_FIELDS = (
    "schema",
    "frame",
    "anchor",
    "baseline",
    "expressions",
    "generator",
    "licence",
    "provenance",
)
OPTIONAL_FIELDS = ("companion",)
# The lengths of the text fields, in characters.
TEXT_LIMITS = {"companion": 64, "licence": 500, "provenance": 2000}

# The reason codes of the check.
CODES = (
    "extra_file",
    "missing_frame",
    "wrong_size",
    "not_rgba",
    "interlaced",
    "unknown_field",
    "subfolder",
    "too_many_files",
    "too_large",
    "bad_manifest",
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def frame_name(expression: str, mouth: str, eyes: str) -> str:
    """The file name of one frame."""
    return f"{expression}.{mouth}.{eyes}.png"


def frame_names(expressions) -> list[str]:
    """The file names of every frame of these expressions, in the order of the contract."""
    return [frame_name(x, m, e) for x in expressions for m in MOUTHS for e in EYES]


def is_frame_name(name: str) -> bool:
    """Whether a file name names a frame of any expression of the contract."""
    parts = name.split(".")
    return (
        len(parts) == 4
        and parts[3] == "png"
        and parts[0] in EXPRESSIONS
        and parts[1] in MOUTHS
        and parts[2] in EYES
    )


def text_problem(value) -> str | None:
    """Why a text field's value breaks the contract (a string without control characters and
    line breaks), or None."""
    if not isinstance(value, str):
        return "is not a string"
    for ch in value:
        if unicodedata.category(ch) in ("Cc", "Zl", "Zp"):
            return f"holds the control character or line break U+{ord(ch):04X}"
    return None


def png_header(data: bytes) -> dict | None:
    """The header (IHDR) of a PNG file: width, height, bit depth, colour type and interlace
    method; None when the bytes are not a PNG file."""
    if len(data) < 33 or data[:8] != PNG_SIGNATURE or data[12:16] != b"IHDR":
        return None
    width, height, depth, colour, _, _, interlace = struct.unpack(
        ">IIBBBBB", data[16:29]
    )
    return dict(
        width=width,
        height=height,
        bit_depth=depth,
        colour_type=colour,
        interlace=interlace,
    )


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def check_manifest(manifest) -> list[dict]:
    """The problems of a manifest (the parsed JSON value), each ``{"code", "message"}``."""
    problems = []

    def bad(message, code="bad_manifest"):
        problems.append({"code": code, "message": message, "file": MANIFEST})

    if not isinstance(manifest, dict):
        bad("The manifest is not a JSON object.")
        return problems
    for key in manifest:
        if key not in REQUIRED_FIELDS and key not in OPTIONAL_FIELDS:
            bad(f"The manifest has the unknown field {key!r}.", "unknown_field")
    for key in REQUIRED_FIELDS:
        if key not in manifest:
            bad(f"The manifest has no {key!r} field.")
    if "schema" in manifest and manifest["schema"] != SCHEMA:
        bad(f"The schema is {manifest['schema']!r}; it must be {SCHEMA!r}.")

    for key, limit in TEXT_LIMITS.items():
        if key not in manifest:
            continue
        value = manifest[key]
        why = text_problem(value)
        if why:
            bad(f"The field {key!r} {why}.")
        elif not 1 <= len(value) <= limit:
            bad(
                f"The field {key!r} has {len(value)} characters; it takes 1 to {limit}."
            )

    size = None
    frame = manifest.get("frame")
    if "frame" in manifest:
        if not isinstance(frame, dict):
            bad("The field 'frame' is not an object with a width and a height.")
        else:
            for key in frame:
                if key not in ("width", "height"):
                    bad(
                        f"The field 'frame' has the unknown field {key!r}.",
                        "unknown_field",
                    )
            w, h = frame.get("width"), frame.get("height")
            if not (_is_int(w) and _is_int(h)):
                bad("The frame's width and height must be whole numbers.")
            elif not (1 <= w <= MAX_SIDE and 1 <= h <= MAX_SIDE):
                bad(
                    f"The frame is {w} by {h} pixels; each side takes 1 to {MAX_SIDE}.",
                    "too_large" if max(w, h) > MAX_SIDE else "bad_manifest",
                )
            else:
                size = (w, h)

    anchor = manifest.get("anchor")
    anchor_y = None
    if "anchor" in manifest:
        if not isinstance(anchor, dict):
            bad("The field 'anchor' is not an object with an x and a y.")
        else:
            for key in anchor:
                if key not in ("x", "y"):
                    bad(
                        f"The field 'anchor' has the unknown field {key!r}.",
                        "unknown_field",
                    )
            x, y = anchor.get("x"), anchor.get("y")
            if not (_is_number(x) and _is_number(y)):
                bad("The anchor's x and y must be numbers.")
            elif size and not (0 <= x <= size[0] and 0 <= y <= size[1]):
                bad(f"The anchor ({x}, {y}) lies outside the frame.")
            else:
                anchor_y = y

    if "baseline" in manifest:
        b = manifest["baseline"]
        if not _is_number(b):
            bad("The baseline must be a number.")
        elif size and b > size[1]:
            bad(f"The baseline {b} lies below the frame's height {size[1]}.")
        elif anchor_y is not None and b <= anchor_y:
            bad(f"The baseline {b} must lie below the anchor's y {anchor_y}.")

    if "expressions" in manifest:
        xs = manifest["expressions"]
        if not isinstance(xs, list) or not xs:
            bad("The field 'expressions' must be a list of expressions.")
        else:
            for x in xs:
                if x not in EXPRESSIONS:
                    bad(
                        f"Unknown expression {x!r}; the expressions are {list(EXPRESSIONS)}."
                    )
            if len(set(map(str, xs))) != len(xs):
                bad("The expressions must be listed once each.")
            if xs[0] != REQUIRED_EXPRESSION:
                bad(
                    f"The expression {REQUIRED_EXPRESSION!r} is required, and comes first."
                )

    if "generator" in manifest:
        g = manifest["generator"]
        if not isinstance(g, dict):
            bad("The field 'generator' is not an object with a name and a version.")
        else:
            for key in g:
                if key not in ("name", "version"):
                    bad(
                        f"The field 'generator' has the unknown field {key!r}.",
                        "unknown_field",
                    )
            for key in ("name", "version"):
                why = text_problem(g.get(key))
                if why or not g.get(key):
                    bad(f"The generator's {key} must be a text of one line.")
    return problems


def check_pack(folder) -> dict:
    """
    Check a pack folder against the contract. Returns ``{"pass", "problems", "frames",
    "bytes", "largest_frame", "smallest_frame"}``, with each problem ``{"code", "message",
    "file"}`` and its code one of :data:`CODES`. Raises ``FileNotFoundError`` or
    ``NotADirectoryError`` when the folder cannot be read as a folder.
    """
    folder = pathlib.Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"{folder} does not exist.")
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a folder.")
    problems: list[dict] = []

    def bad(code, message, file=None):
        problems.append({"code": code, "message": message, "file": file})

    entries = sorted(folder.iterdir(), key=lambda p: p.name)
    files = []
    for p in entries:
        if p.is_dir():
            bad(
                "subfolder",
                f"{p.name} is a subfolder; a pack is one flat folder.",
                p.name,
            )
        elif not p.is_file():
            bad("extra_file", f"{p.name} is not a regular file.", p.name)
        else:
            files.append(p)
    if len(entries) > MAX_FILES:
        bad(
            "too_many_files",
            f"The pack holds {len(entries)} entries; at most {MAX_FILES}.",
        )
    sizes = {p.name: p.stat().st_size for p in files}
    total = sum(sizes.values())
    if total > MAX_PACK_BYTES:
        bad(
            "too_large",
            f"The pack holds {total} bytes; at most {MAX_PACK_BYTES} ({MAX_PACK_BYTES // 2**20} MiB).",
        )

    manifest = None
    if MANIFEST not in sizes:
        bad("bad_manifest", f"The pack has no {MANIFEST}.", MANIFEST)
    elif sizes[MANIFEST] > MAX_MANIFEST_BYTES:
        bad(
            "too_large",
            f"{MANIFEST} holds {sizes[MANIFEST]} bytes; at most {MAX_MANIFEST_BYTES}.",
            MANIFEST,
        )
    else:
        try:
            manifest = json.loads((folder / MANIFEST).read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            bad("bad_manifest", f"{MANIFEST} is not JSON in UTF-8: {error}", MANIFEST)
        else:
            problems.extend(check_manifest(manifest))

    listed = []
    size = None
    if isinstance(manifest, dict):
        xs = manifest.get("expressions")
        if isinstance(xs, list):
            listed = [x for x in dict.fromkeys(xs) if x in EXPRESSIONS]
        f = manifest.get("frame")
        if isinstance(f, dict) and _is_int(f.get("width")) and _is_int(f.get("height")):
            size = (f["width"], f["height"])
    wanted = frame_names(listed)
    for name in wanted:
        if name not in sizes:
            bad("missing_frame", f"The frame {name} is missing.", name)
    for name in sizes:
        if name != MANIFEST and name not in wanted:
            why = (
                "is not a frame of a listed expression"
                if is_frame_name(name)
                else "is not a frame of the contract"
            )
            bad("extra_file", f"{name} {why}.", name)

    frame_bytes = []
    for name in wanted:
        if name not in sizes:
            continue
        frame_bytes.append(sizes[name])
        with open(folder / name, "rb") as fh:
            head = png_header(fh.read(33))
        if head is None:
            bad("not_rgba", f"{name} is not a PNG file.", name)
            continue
        if head["colour_type"] != 6 or head["bit_depth"] != 8:
            bad(
                "not_rgba",
                f"{name} has colour type {head['colour_type']} at {head['bit_depth']} bits; "
                "frames are 8-bit RGBA (colour type 6).",
                name,
            )
        if head["interlace"] != 0:
            bad(
                "interlaced", f"{name} is interlaced; frames have no interlacing.", name
            )
        w, h = head["width"], head["height"]
        if max(w, h) > MAX_SIDE:
            bad(
                "too_large",
                f"{name} is {w} by {h} pixels; at most {MAX_SIDE} on each side.",
                name,
            )
        elif size and (w, h) != size:
            bad(
                "wrong_size",
                f"{name} is {w} by {h} pixels; the manifest says {size[0]} by {size[1]}.",
                name,
            )
    return {
        "pass": not problems,
        "problems": problems,
        "frames": len(frame_bytes),
        "bytes": total,
        "largest_frame": max(frame_bytes) if frame_bytes else 0,
        "smallest_frame": min(frame_bytes) if frame_bytes else 0,
    }

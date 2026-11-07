# OpenSculptBoy
# Apache License, Version 2.0
"""
The contract of a layered portrait pack (``squire.portraits/2``) and its check.

A layered pack is one flat folder that holds ``manifest.json`` and the layers of
:mod:`opensculptboy.portrait_pack.layers`: a body layer per pose and framing, a face layer per
framing and face, a face layer per face of each face set of its own (a pose that turns its head),
a front layer where a pose's hand passes in front of the face, and the key frames of the leap.
:func:`check_layered_pack` applies every rule and gives a reason code for each failure, with
the codes of the bust pack's check (``contract.CODES``).
"""

from __future__ import annotations

import json
import pathlib
import re
import struct

from opensculptboy.portrait_pack.contract import (
    EXPRESSIONS,
    EYES,
    MANIFEST,
    MAX_MANIFEST_BYTES,
    MOUTHS,
    REQUIRED_EXPRESSION,
    TEXT_LIMITS,
    png_header,
    text_problem,
)

SCHEMA_V2 = "squire.portraits/2"
FRAMING_NAMES = ("close", "waist", "thigh", "full", "thought", "stern", "low")
# The framings at which at_ease shows whenever the pack holds them; the others hold the poses of
# their shot alone.
BASE_FRAMINGS = ("close", "waist", "thigh", "full")
CAMERA_HEIGHTS = ("eyes", "chest", "hips")
LEAP_KEYS = ("crouch", "rise", "peak", "land")
BASE_POSE = "at_ease"
POSE_ID = re.compile(r"^[a-z][a-z_]{0,31}$")
FORMATS = ("webp", "png")
MAX_FILES_V2 = 600
MAX_PACK_BYTES_V2 = 256 * 1024 * 1024
MAX_SIDE_V2 = 4096
MAX_POSES = 16

REQUIRED = (
    "schema",
    "format",
    "expressions",
    "framings",
    "poses",
    "generator",
    "licence",
    "provenance",
)
OPTIONAL = ("companion", "leap", "extra")
# The light passes (opensculptboy.portrait_pack.light): each layer may come with one file per pass,
# named after the layer with the pass before the format (``body.at_ease.close.key.webp``). The pack
# then lists its passes with their scales and describes its light; its bound on files grows by the
# files of the passes.
OPTIONAL_LIGHT = ("passes", "light")
LIGHT_PASSES = (("key", "fill", "rim"), ("base", "rim"))
PASS_FORMAT = "webp"
# The joints of each body layer and each leap frame (``joints``), in the layer's own pixels, so an
# app can check where its frame's edges cut the figure:
# the point between the eyes, the crown and the chin, and VRM humanoid bones by their
# names (each bone's head: the elbow for the lower arm, the wrist for the hand, the knee for the
# lower leg, the ankle for the foot).
OPTIONAL_JOINTS = ("joints",)
JOINT_NAMES = (
    "eyes",
    "crown",
    "chin",
    "head",
    "neck",
    "upperChest",
    "chest",
    "hips",
    "leftUpperArm",
    "rightUpperArm",
    "leftLowerArm",
    "rightLowerArm",
    "leftHand",
    "rightHand",
    "leftMiddleDistal",
    "rightMiddleDistal",
    "leftUpperLeg",
    "rightUpperLeg",
    "leftLowerLeg",
    "rightLowerLeg",
    "leftFoot",
    "rightFoot",
)
# A joint may lie outside its layer (a hand the frame cuts off), within this many pixels.
JOINT_REACH = 3 * 4096


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def layer_kind(name: str):
    """The kind and the parts of a layer's file name: ("body", pose, framing, ext),
    ("face", framing, expression, mouth, eyes, ext), ("face", framing, set, expression, mouth,
    eyes, ext) for a pose's own face set, ("front", pose, framing, ext) or ("leap", key, ext);
    None for any other name."""
    parts = name.split(".")
    if len(parts) < 2 or parts[-1] not in FORMATS:
        return None
    ext = parts[-1]
    if (
        parts[0] == "body"
        and len(parts) == 4
        and POSE_ID.match(parts[1])
        and parts[2] in FRAMING_NAMES
    ):
        return ("body", parts[1], parts[2], ext)
    if (
        parts[0] == "front"
        and len(parts) == 4
        and POSE_ID.match(parts[1])
        and parts[2] in FRAMING_NAMES
    ):
        return ("front", parts[1], parts[2], ext)
    if (
        parts[0] == "face"
        and len(parts) == 6
        and parts[1] in FRAMING_NAMES
        and parts[2] in EXPRESSIONS
        and parts[3] in MOUTHS
        and parts[4] in EYES
    ):
        return ("face", parts[1], parts[2], parts[3], parts[4], ext)
    if (
        parts[0] == "face"
        and len(parts) == 7
        and parts[1] in FRAMING_NAMES
        and POSE_ID.match(parts[2])
        and parts[3] in EXPRESSIONS
        and parts[4] in MOUTHS
        and parts[5] in EYES
    ):
        return ("face", parts[1], parts[2], parts[3], parts[4], parts[5], ext)
    if parts[0] == "leap" and len(parts) == 3 and parts[1] in LEAP_KEYS:
        return ("leap", parts[1], ext)
    return None


def is_layer_name(name: str) -> bool:
    return layer_kind(name) is not None or pass_of(name) is not None


def pass_of(name: str):
    """The layer and the pass of a pass file's name (``body.at_ease.close.key.webp`` gives
    ``("body.at_ease.close.webp", "key")`` for a pack in WebP); None for any other name."""
    parts = name.split(".")
    if len(parts) < 3 or parts[-1] != PASS_FORMAT:
        return None
    if parts[-2] not in {p for s in LIGHT_PASSES for p in s}:
        return None
    layer = ".".join(parts[:-2] + [PASS_FORMAT])
    return (layer, parts[-2]) if layer_kind(layer) is not None else None


def check_light(m) -> list[str]:
    """The problems of a manifest's light passes (``passes`` and ``light``), as messages."""
    out = []
    if ("passes" in m) != ("light" in m):
        return ["The fields 'passes' and 'light' come together."]
    if "passes" not in m:
        return out
    ps = m["passes"]
    names = (
        [p.get("name") for p in ps]
        if isinstance(ps, list) and all(isinstance(p, dict) for p in ps)
        else None
    )
    if names is None or tuple(names) not in LIGHT_PASSES:
        out.append(
            "The field 'passes' must list the passes key, fill and rim, or base and rim, "
            "each {name, scale}."
        )
    else:
        for p in ps:
            if set(p) != {"name", "scale"} or not (
                _num(p["scale"]) and 0 < p["scale"] <= 64
            ):
                out.append(
                    f"The pass {p['name']!r} must hold its scale, above 0 and at most 64."
                )
    if m.get("format") != PASS_FORMAT:
        out.append("A pack with light passes is in WebP.")
    lt = m["light"]
    tone = lt.get("tone") if isinstance(lt, dict) else None
    if not (
        isinstance(lt, dict)
        and set(lt)
        <= {"rig", "key_side", "exposure", "tone", "encoding", "lights", "face"}
        and isinstance(lt.get("rig"), str)
        and lt.get("key_side") in ("left", "right")
        and _num(lt.get("exposure"))
        and lt["exposure"] > 0
        and lt.get("encoding") == "srgb"
        and isinstance(tone, dict)
        and tone.get("curve") == "agx"
        and _num(tone.get("saturation"))
        and _num(tone.get("contrast"))
        and isinstance(lt.get("lights", {}), dict)
        and isinstance(lt.get("face", {}), dict)
        and all(_num(v) and v >= 0 for v in lt.get("face", {}).values())
    ):
        out.append(
            "The field 'light' must hold the rig, the key's side (left or right), the exposure, the tone "
            "(curve agx, saturation, contrast) and the encoding srgb, and may hold the lights and the face's light."
        )
    return out


def check_joints(m) -> list[str]:
    """The problems of a manifest's joints (``joints``), as messages: the names, a list of numbers
    (x, y for each name) for each body layer of a pose at a framing it shows at, and for each leap
    frame."""
    if "joints" not in m:
        return []
    j = m["joints"]
    shape = (
        "The field 'joints' must hold the joints' names, 'poses' (pose to framing to the list of x, y "
        "for each name) and may hold 'leap' (leap frame to the list)."
    )
    if not (
        isinstance(j, dict)
        and set(j) <= {"names", "poses", "leap"}
        and isinstance(j.get("names"), list)
        and isinstance(j.get("poses"), dict)
        and isinstance(j.get("leap", {}), dict)
    ):
        return [shape]
    names = j["names"]
    if (
        not names
        or any(n not in JOINT_NAMES for n in names)
        or len(set(names)) != len(names)
    ):
        return [f"The joints' names must be distinct names of {list(JOINT_NAMES)}."]
    out = []

    def coords(v) -> bool:
        return (
            isinstance(v, list)
            and len(v) == 2 * len(names)
            and all(_num(x) and abs(x) <= JOINT_REACH for x in v)
        )

    ps = m.get("poses") if isinstance(m.get("poses"), dict) else {}
    for pose, by in j["poses"].items():
        shows = (
            (ps.get(pose) or {}).get("framings")
            if isinstance(ps.get(pose), dict)
            else None
        )
        if not isinstance(by, dict) or not isinstance(shows, list):
            out.append(f"The joints of {pose!r} must belong to a pose of the pack.")
            continue
        for f, v in by.items():
            if f not in shows:
                out.append(
                    f"The joints of {pose!r} at {f!r}: the pose does not show there."
                )
            elif not coords(v):
                out.append(
                    f"The joints of {pose!r} at {f!r} must be x, y for each name."
                )
    frames = (
        (m.get("leap") or {}).get("frames") if isinstance(m.get("leap"), dict) else None
    )
    for k, v in j.get("leap", {}).items():
        if not isinstance(frames, list) or k not in frames:
            out.append(
                f"The joints of the leap frame {k!r}: the pack has no such frame."
            )
        elif not coords(v):
            out.append(
                f"The joints of the leap frame {k!r} must be x, y for each name."
            )
    return out


def webp_header(data: bytes) -> dict | None:
    """The size of a WebP file and whether it holds alpha; None when the bytes are not WebP."""
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    chunk = data[12:16]
    if chunk == b"VP8X":
        flags = data[20]
        w = 1 + int.from_bytes(data[24:27], "little")
        h = 1 + int.from_bytes(data[27:30], "little")
        return {"width": w, "height": h, "alpha": bool(flags & 0x10)}
    if chunk == b"VP8L":
        b = data[21:25]
        bits = struct.unpack("<I", b)[0]
        w = (bits & 0x3FFF) + 1
        h = ((bits >> 14) & 0x3FFF) + 1
        return {"width": w, "height": h, "alpha": bool((bits >> 28) & 1)}
    if chunk == b"VP8 ":
        w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return {"width": w, "height": h, "alpha": False}
    return None


def image_header(data: bytes, ext: str) -> dict | None:
    """Width, height and alpha of a layer's file in its format; None when it is not one."""
    if ext == "webp":
        return webp_header(data)
    h = png_header(data)
    if h is None:
        return None
    ok = h["colour_type"] == 6 and h["bit_depth"] == 8 and h["interlace"] == 0
    return {"width": h["width"], "height": h["height"], "alpha": ok}


def _rect(r, w, h) -> bool:
    return (
        isinstance(r, dict)
        and set(r) == {"x", "y", "width", "height"}
        and all(_int(r[k]) for k in r)
        and r["x"] >= 0
        and r["y"] >= 0
        and r["width"] >= 1
        and r["height"] >= 1
        and r["x"] + r["width"] <= w
        and r["y"] + r["height"] <= h
    )


def _expression_list(xs, pack) -> bool:
    """A face set's expressions: expressions of the pack, each once."""
    return (
        isinstance(xs, list)
        and bool(xs)
        and all(isinstance(x, str) and x in pack for x in xs)
        and len(set(xs)) == len(xs)
    )


def _camera(c) -> bool:
    """A framing's camera: the shot, an optional expression, the lens, the height and the
    distance above it, the tilt and the body's turn."""
    if not isinstance(c, dict) or not {
        "shot",
        "lens",
        "height",
        "above",
        "tilt",
        "turn",
    } <= set(c):
        return False
    if any(
        k not in ("shot", "expression", "lens", "height", "above", "tilt", "turn")
        for k in c
    ):
        return False
    if not isinstance(c["shot"], str) or not POSE_ID.match(c["shot"]):
        return False
    if "expression" in c and c["expression"] not in EXPRESSIONS:
        return False
    return (
        c["height"] in CAMERA_HEIGHTS
        and _num(c["lens"])
        and 14 <= c["lens"] <= 300
        and _num(c["above"])
        and abs(c["above"]) <= 10
        and _num(c["tilt"])
        and abs(c["tilt"]) <= 20
        and _num(c["turn"])
        and abs(c["turn"]) <= 60
    )


def check_layered_manifest(m) -> list[dict]:
    """The problems of a layered pack's manifest, each ``{"code", "message", "file"}``."""
    problems = []

    def bad(message, code="bad_manifest"):
        problems.append({"code": code, "message": message, "file": MANIFEST})

    if not isinstance(m, dict):
        bad("The manifest is not a JSON object.")
        return problems
    for k in m:
        if (
            k not in REQUIRED
            and k not in OPTIONAL
            and k not in OPTIONAL_LIGHT
            and k not in OPTIONAL_JOINTS
        ):
            bad(f"The manifest has the unknown field {k!r}.", "unknown_field")
    for message in check_light(m) + check_joints(m):
        bad(message)
    for k in REQUIRED:
        if k not in m:
            bad(f"The manifest has no {k!r} field.")
    if m.get("schema") != SCHEMA_V2:
        bad(f"The schema must be {SCHEMA_V2!r}.")
    if "format" in m and m["format"] not in FORMATS:
        bad(f"The format must be one of {list(FORMATS)}.")
    for key, limit in TEXT_LIMITS.items():
        if key in m:
            why = text_problem(m[key])
            if why or not 1 <= len(m[key]) <= limit:
                bad(
                    f"The field {key!r} must be a text of 1 to {limit} characters on one line."
                )
    g = m.get("generator")
    if "generator" in m and not (
        isinstance(g, dict)
        and set(g) == {"name", "version"}
        and all(isinstance(g[k], str) and g[k] and not text_problem(g[k]) for k in g)
    ):
        bad("The field 'generator' must be {name, version}, each a text of one line.")
    xs = m.get("expressions")
    if "expressions" in m:
        if (
            not isinstance(xs, list)
            or not xs
            or any(x not in EXPRESSIONS for x in xs)
            or len(set(map(str, xs))) != len(xs)
        ):
            bad("The field 'expressions' must list known expressions, each once.")
        elif xs[0] != REQUIRED_EXPRESSION:
            bad("The expression 'neutral' is required, and comes first.")
    fr = m.get("framings")
    if "framings" in m:
        if not isinstance(fr, dict) or not fr:
            bad("The field 'framings' must be an object of framings.")
            fr = {}
        for name, f in fr.items():
            if name not in FRAMING_NAMES:
                bad(
                    f"Unknown framing {name!r}; the framings are {list(FRAMING_NAMES)}."
                )
                continue
            if (
                not isinstance(f, dict)
                or any(
                    k
                    not in (
                        "frame",
                        "head",
                        "face",
                        "ground",
                        "camera",
                        "expressions",
                        "sets",
                    )
                    for k in f
                )
                or "frame" not in f
                or "head" not in f
            ):
                bad(
                    f"The framing {name!r} must hold frame and head, and may hold face, expressions, "
                    "sets, camera and ground."
                )
                continue
            fw = f["frame"]
            if not (
                isinstance(fw, dict)
                and set(fw) == {"width", "height"}
                and _int(fw["width"])
                and _int(fw["height"])
            ):
                bad(f"The frame of {name!r} must be {{width, height}} in whole pixels.")
                continue
            w, h = fw["width"], fw["height"]
            if not (1 <= w <= MAX_SIDE_V2 and 1 <= h <= MAX_SIDE_V2):
                bad(
                    f"The frame of {name!r} is {w} by {h}; each side takes 1 to {MAX_SIDE_V2}.",
                    "too_large",
                )
            hd = f["head"]
            if not (
                isinstance(hd, dict)
                and {"x", "y", "top", "chin"}
                <= set(hd)
                <= {"x", "y", "top", "chin", "size"}
                and all(_int(hd[k]) for k in ("x", "y", "top", "chin"))
                and (
                    "size" not in hd
                    or (_num(hd["size"]) and 1 <= hd["size"] <= MAX_SIDE_V2)
                )
            ):
                bad(
                    f"The head of {name!r} must be {{x, y, top, chin}} in whole pixels, with an "
                    "optional size in pixels."
                )
            elif not (
                0 <= hd["x"] <= w
                and hd["top"] < hd["y"] < hd["chin"]
                and -h <= hd["top"]
                and hd["chin"] <= 2 * h
            ):
                bad(
                    f"The head of {name!r} must have its crown above the anchor and the anchor above "
                    "the chin, inside the frame."
                )
            if "face" in f and not _rect(f["face"], w, h):
                bad(
                    f"The face of {name!r} must be a rectangle {{x, y, width, height}} inside the frame."
                )
            if "ground" in f and not (_int(f["ground"]) and 0 <= f["ground"] <= h):
                bad(f"The ground of {name!r} must be a row of the frame.")
            if "camera" in f and not _camera(f["camera"]):
                bad(
                    f"The camera of {name!r} must be {{shot, expression (optional), lens, height, "
                    "above, tilt, turn}} within their ranges."
                )
            pack_xs = xs if isinstance(xs, list) else []
            if "expressions" in f and not (
                "face" in f
                and _expression_list(f["expressions"], pack_xs)
                and f["expressions"][0] == REQUIRED_EXPRESSION
            ):
                bad(
                    f"The expressions of {name!r} must list expressions of the pack, neutral first, "
                    "each once, for its face."
                )
            sets = f.get("sets", {})
            if not isinstance(sets, dict) or any(
                not (
                    POSE_ID.match(str(set_name))
                    and isinstance(fs, dict)
                    and set(fs) == {"face", "expressions"}
                    and _rect(fs["face"], w, h)
                    and _expression_list(fs["expressions"], pack_xs)
                )
                for set_name, fs in sets.items()
            ):
                bad(
                    f"The sets of {name!r} must map a pose to {{face, expressions}}: a rectangle "
                    "inside the frame and expressions of the pack, each once."
                )
    ps = m.get("poses")
    if "poses" in m:
        if not isinstance(ps, dict) or not ps:
            bad("The field 'poses' must be an object of poses.")
            ps = {}
        if len(ps) > MAX_POSES:
            bad(f"The pack holds {len(ps)} poses; at most {MAX_POSES}.")
        if BASE_POSE not in ps:
            bad(f"The pose {BASE_POSE!r} is required.")
        for name, p in ps.items():
            if not POSE_ID.match(str(name)):
                bad(
                    f"The pose id {name!r} must be lower case letters and underscores, 1 to 32."
                )
                continue
            if (
                not isinstance(p, dict)
                or any(k not in ("framings", "front") for k in p)
                or "framings" not in p
            ):
                bad(f"The pose {name!r} must hold framings, and may hold front.")
                continue
            pf = p["framings"]
            if (
                not isinstance(pf, list)
                or not pf
                or any(x not in fr for x in pf)
                or len(set(pf)) != len(pf)
            ):
                bad(
                    f"The framings of {name!r} must list framings of the pack, each once."
                )
            ft = p.get("front", [])
            if not isinstance(ft, list) or any(
                x not in (pf or []) or "face" not in fr.get(x, {}) for x in ft
            ):
                bad(
                    f"The front of {name!r} must list framings of the pose that have a face."
                )
            if not isinstance(pf, list):
                continue
            own = [x for x in pf if name in ((fr.get(x) or {}).get("sets") or {})]
            if own and len(own) != len(pf):
                bad(
                    f"The pose {name!r} has a face set of its own at {own}; it needs one at each of "
                    "its framings."
                )
            if own and isinstance(ft, list) and ft:
                bad(f"The pose {name!r} has a face set of its own, so it has no front.")
        if isinstance(fr, dict):
            for n, f in fr.items():
                for set_name in (f.get("sets") or {}) if isinstance(f, dict) else ():
                    if n not in ((ps.get(set_name) or {}).get("framings") or []):
                        bad(
                            f"The framing {n!r} holds a face set for {set_name!r}, which does not show there."
                        )
        if isinstance(fr, dict) and isinstance(ps.get(BASE_POSE), dict):
            missing = [
                n
                for n in fr
                if n in BASE_FRAMINGS and n not in (ps[BASE_POSE].get("framings") or [])
            ]
            if missing:
                bad(
                    f"The pose {BASE_POSE!r} must show in every framing of {list(BASE_FRAMINGS)} the "
                    f"pack holds; it lacks {missing}."
                )
    lp = m.get("leap")
    if "leap" in m:
        ok = (
            isinstance(lp, dict)
            and set(lp) == {"framing", "frames"}
            and lp["framing"] in (fr or {})
        )
        ok = (
            ok
            and isinstance(lp["frames"], list)
            and lp["frames"]
            and all(k in LEAP_KEYS for k in lp["frames"])
        )
        ok = ok and [k for k in LEAP_KEYS if k in lp["frames"]] == lp["frames"]
        if not ok:
            bad(
                f"The leap must be {{framing, frames}}, its frames among {list(LEAP_KEYS)} in that order."
            )
    ex = m.get("extra")
    if "extra" in m and not (
        isinstance(ex, dict)
        and set(ex) <= {"above", "width"}
        and all(_num(v) and 0 <= v <= 3 for v in ex.values())
    ):
        bad("The field 'extra' must hold above and width, each 0 to 3 heads.")
    return problems


def wanted_layers(m) -> dict:
    """Every layer a manifest names: file name to the size it must have."""
    out = {}
    try:
        ext = m["format"]
        fr = m["framings"]
        for pose, p in m["poses"].items():
            for f in p["framings"]:
                fw = fr[f]["frame"]
                out[f"body.{pose}.{f}.{ext}"] = (fw["width"], fw["height"])
            for f in p.get("front", []):
                r = fr[f]["face"]
                out[f"front.{pose}.{f}.{ext}"] = (r["width"], r["height"])
        for f, d in fr.items():
            if "face" in d:
                r = d["face"]
                for x in d.get("expressions", m["expressions"]):
                    for mo in MOUTHS:
                        for e in EYES:
                            out[f"face.{f}.{x}.{mo}.{e}.{ext}"] = (
                                r["width"],
                                r["height"],
                            )
            for pose, fs in (d.get("sets") or {}).items():
                r = fs["face"]
                for x in fs["expressions"]:
                    for mo in MOUTHS:
                        for e in EYES:
                            out[f"face.{f}.{pose}.{x}.{mo}.{e}.{ext}"] = (
                                r["width"],
                                r["height"],
                            )
        if "leap" in m:
            fw = fr[m["leap"]["framing"]]["frame"]
            for k in m["leap"]["frames"]:
                out[f"leap.{k}.{ext}"] = (fw["width"], fw["height"])
        for p in m.get("passes") or []:
            for name, size in list(out.items()):
                if pass_of(name) is None:
                    out[f"{name[: -len(ext) - 1]}.{p['name']}.{ext}"] = size
    except (KeyError, TypeError, AttributeError):
        return out
    return out


def _pass_count(folder: pathlib.Path) -> int:
    """The number of light passes a pack's manifest lists (0 when it lists none or cannot be read)."""
    try:
        ps = json.loads((folder / MANIFEST).read_bytes().decode("utf-8-sig")).get(
            "passes"
        )
        return min(len(ps), 3) if isinstance(ps, list) else 0
    except (OSError, ValueError, AttributeError):
        return 0


def check_layered_pack(folder) -> dict:
    """Check a layered pack folder: ``{"pass", "problems", "frames", "bytes", "largest_frame",
    "smallest_frame"}`` as :func:`contract.check_pack` gives it."""
    folder = pathlib.Path(folder)
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
    max_files = MAX_FILES_V2 * (1 + _pass_count(folder))
    if len(entries) > max_files:
        bad(
            "too_many_files",
            f"The pack holds {len(entries)} entries; at most {max_files}.",
        )
    sizes = {p.name: p.stat().st_size for p in files}
    total = sum(sizes.values())
    if total > MAX_PACK_BYTES_V2:
        bad(
            "too_large",
            f"The pack holds {total} bytes; at most {MAX_PACK_BYTES_V2 // 2**20} MiB.",
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
            manifest = json.loads((folder / MANIFEST).read_bytes().decode("utf-8-sig"))
        except (UnicodeDecodeError, ValueError) as error:
            bad("bad_manifest", f"{MANIFEST} is not JSON in UTF-8: {error}", MANIFEST)
        else:
            problems.extend(check_layered_manifest(manifest))
    wanted = wanted_layers(manifest) if isinstance(manifest, dict) else {}
    for name in wanted:
        if name not in sizes:
            bad("missing_frame", f"The layer {name} is missing.", name)
    for name in sizes:
        if name != MANIFEST and name not in wanted:
            bad("extra_file", f"{name} is not a layer the manifest names.", name)
    layer_bytes = []
    for name, want in wanted.items():
        if name not in sizes:
            continue
        layer_bytes.append(sizes[name])
        with open(folder / name, "rb") as fh:
            head = image_header(fh.read(64), name.rsplit(".", 1)[1])
        if head is None or not head["alpha"]:
            bad("not_rgba", f"{name} is not an image of its format with alpha.", name)
            continue
        if (head["width"], head["height"]) != want:
            bad(
                "wrong_size",
                f"{name} is {head['width']} by {head['height']} pixels; the manifest says {want[0]} by {want[1]}.",
                name,
            )
    return {
        "pass": not problems,
        "problems": problems,
        "frames": len(layer_bytes),
        "bytes": total,
        "largest_frame": max(layer_bytes, default=0),
        "smallest_frame": min(layer_bytes, default=0),
    }

# OpenSculptBoy
# Apache License, Version 2.0
"""
Outfit files ``opensculptboy.outfit/1`` (docs/wearables.md, "Outfit files"): an outfit apart
from any card, so that it can be shared.

An outfit file ``<name>.outfit.json`` names the outfit (``id``, ``label``, ``description``,
``tags``), lists its pieces as wearable packages with paths relative to the file (``piece``,
``file``, ``enabled`` and ``overrides`` of the package's ``layer`` and ``label``), adds hides
rules between pieces (``{"piece", "target", "groups"}``) and sets the spring chains of
pieces (``springs``, by piece id, in the form of a package's ``springs`` field). The JSON
Schema is :data:`SCHEMA_PATH`.

:func:`export_outfit` writes the file of one outfit of a card (``wearables outfit export``);
:func:`add_outfit` puts a file on a card (``wearables add CARD FILE.outfit.json``), with the
paths made relative to the card, the overrides, rules and springs kept on the outfit's pieces
(:func:`opensculptboy.wearables.wardrobe.check_piece_settings`), and the same guarantees as
the other card commands: a second run changes nothing, ``dry_run`` writes nothing, and an
outfit of the same id with other contents stays unless ``replace``. :func:`read_outfit`
checks a file and raises :class:`OutfitError` with a stable code and a hint.
"""

from __future__ import annotations

import copy
import json
import os
import pathlib

from opensculptboy.wearables.format import (
    PRIVATE,
    PUBLIC,
    ProvenanceError,
    WearableError,
    check_destination,
    repository_root,
)

FORMAT = "opensculptboy.outfit/1"
SUFFIX = ".outfit.json"
SCHEMA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "data"
    / "wearables"
    / "outfit-1.schema.json"
)
KEYS = {
    "format",
    "id",
    "label",
    "description",
    "tags",
    "provenance",
    "pieces",
    "hides",
    "springs",
    "credits",
    "extras",
}


class OutfitError(WearableError):
    """An outfit file that cannot be read, written or put on a card as asked."""

    code = "invalid_outfit"
    hint = "Fix the outfit file; src/opensculptboy/data/wearables/outfit-1.schema.json has its form."


def is_outfit_file(path) -> bool:
    return str(path).endswith(SUFFIX)


def _fail(where: str, message: str, hint: str | None = None):
    raise OutfitError(f"{where}: {message}", "invalid_outfit", hint)


def check_outfit(data, where: str = "outfit") -> None:
    """Raise :class:`OutfitError` (``invalid_outfit``) for a file that does not follow the
    format; the message names the field."""
    from opensculptboy.wearables.wardrobe import NO_OUTFIT, check_piece_settings

    if not isinstance(data, dict):
        _fail(where, "an outfit file is a JSON object.")
    if data.get("format") != FORMAT:
        _fail(
            where,
            f"format {data.get('format')!r}; this version reads {FORMAT}.",
            f'Set "format": "{FORMAT}".',
        )
    unknown = sorted(set(data) - KEYS)
    if unknown:
        _fail(where, f"unknown keys {unknown}.", f"The keys are {sorted(KEYS)}.")
    oid = data.get("id")
    if not isinstance(oid, str) or not oid or "/" in oid or oid == NO_OUTFIT:
        _fail(where, f"id {oid!r}: an outfit id is text without '/' and is not 'none'.")
    for key in ("label", "description"):
        if key in data and not isinstance(data[key], str):
            _fail(where, f"{key} must be text.")
    if "\n" in data.get("description", ""):
        _fail(where, "description must be one line.")
    tags = data.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) and t for t in tags):
        _fail(where, "tags must be a list of words.")
    if data.get("provenance", PUBLIC) not in (PUBLIC, PRIVATE):
        _fail(where, "provenance must be 'public' or 'private'.")
    pieces = data.get("pieces")
    if not isinstance(pieces, list) or not pieces:
        _fail(where, "pieces must list one or more pieces.")
    ids = []
    for k, piece in enumerate(pieces):
        at = f"pieces[{k}]"
        if not isinstance(piece, dict):
            _fail(where, f"{at} must be an object.")
        unknown = sorted(set(piece) - {"piece", "file", "enabled", "overrides"})
        if unknown:
            _fail(where, f"{at}: unknown keys {unknown}.")
        pid = piece.get("piece")
        if not isinstance(pid, str) or not pid or "/" in pid:
            _fail(where, f"{at}: piece {pid!r} must be an id without '/'.")
        if pid in ids:
            _fail(where, f"{at}: the piece {pid!r} comes twice.")
        ids.append(pid)
        file = piece.get("file")
        if not isinstance(file, str) or not file.endswith(".wearable.json"):
            _fail(where, f"{at}: file {file!r} must name a .wearable.json package.")
        try:
            check_piece_settings(piece, f"{data['id']}/{pid}")
        except ValueError as error:
            _fail(where, f"{at}: {error}")
    hides = data.get("hides", [])
    if not isinstance(hides, list):
        _fail(where, "hides must be a list of rules.")
    for k, rule in enumerate(hides):
        at = f"hides[{k}]"
        if not isinstance(rule, dict) or set(rule) != {"piece", "target", "groups"}:
            _fail(where, f"{at} must be {{'piece', 'target', 'groups'}}: {rule!r}.")
        if rule["piece"] not in ids:
            raise OutfitError(
                f"{where}: {at}: the piece {rule['piece']!r} is not in the outfit.",
                "unknown_piece",
                f"Name one of the outfit's pieces: {ids}.",
            )
        groups = rule["groups"]
        if (
            not isinstance(rule["target"], str)
            or not rule["target"]
            or not isinstance(groups, list)
            or not groups
            or not all(isinstance(g, str) and g for g in groups)
        ):
            _fail(where, f"{at}: target is a piece and groups a list of group names.")
    springs = data.get("springs", {})
    if not isinstance(springs, dict):
        _fail(where, "springs must map piece ids to lists of springs.")
    for pid, chains in springs.items():
        if pid not in ids:
            raise OutfitError(
                f"{where}: springs: the piece {pid!r} is not in the outfit.",
                "unknown_piece",
                f"Name one of the outfit's pieces: {ids}.",
            )
        if not isinstance(chains, list) or not all(
            isinstance(c, dict) and isinstance(c.get("name"), str) for c in chains
        ):
            _fail(where, f"springs[{pid!r}] must be a list of named spring entries.")
    credits = data.get("credits", [])
    if not isinstance(credits, list) or not all(isinstance(c, str) for c in credits):
        _fail(where, "credits must be a list of lines.")
    if not isinstance(data.get("extras", {}), dict):
        _fail(where, "extras must be an object.")


def read_outfit(path) -> tuple[pathlib.Path, dict]:
    """The path and the checked contents of an outfit file."""
    path = pathlib.Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise OutfitError(
            f"{path}: no such outfit file.",
            "outfit_not_found",
            "Name an existing .outfit.json file; 'wearables outfit export' writes one.",
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise OutfitError(
            f"{path}: not a JSON file ({error}).",
            "invalid_outfit",
            "Write the outfit file as UTF-8 JSON.",
        ) from error
    check_outfit(data, str(path))
    return path, data


def _relative(target: pathlib.Path, folder: pathlib.Path) -> str:
    return pathlib.Path(os.path.relpath(target.resolve(), folder.resolve())).as_posix()


def _resolve(file: str, folder: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(file)
    return path if path.is_absolute() else folder / path


def export_outfit(
    card,
    outfit: str,
    out,
    *,
    label: str | None = None,
    description: str | None = None,
    tags=(),
    replace: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Write the outfit ``outfit`` of a card as an outfit file ``out``: its pieces with paths
    relative to the file, ``enabled`` as the card's overrides leave it, the pieces'
    ``overrides``, their hides rules (as ``{"piece", "target", "groups"}``) and springs, the
    credits of the packages, and ``provenance`` private when a package is private (such a file
    never goes into a checkout of the repository). An existing file with other contents stays
    unless ``replace``.
    """
    from opensculptboy.wearables.cardedit import CardError, _load_package, _read_card
    from opensculptboy.wearables.export import credit_line

    path, raw = _read_card(card)
    out = pathlib.Path(out)
    if not is_outfit_file(out):
        raise OutfitError(
            f"{out}: an outfit file's name ends in {SUFFIX}.",
            "bad_option",
            f"Pass --out NAME{SUFFIX}.",
        )
    entry = next((o for o in raw.get("outfits", []) if o.get("id") == outfit), None)
    if entry is None:
        ids = [o.get("id") for o in raw.get("outfits", [])]
        raise CardError(
            f"Unknown outfit {outfit!r}; the outfits are {ids}.",
            "unknown_outfit",
            "Name one of the card's outfits with --outfit.",
        )
    overrides = raw.get("wearable_overrides") or {}
    pieces, hides, springs, lines = [], [], {}, []
    provenance = PUBLIC
    for piece in entry.get("pieces", []):
        pid = piece["piece"]
        package = _resolve(piece["file"], path.parent)
        json_path, wearable = _load_package(package)
        if wearable.provenance == PRIVATE:
            provenance = PRIVATE
        enabled = bool(piece.get("enabled", True))
        for key in (f"{outfit}/{pid}",) + (
            (pid,) if raw.get("active_outfit") == outfit else ()
        ):
            if key in overrides:
                enabled = bool(overrides[key].get("enabled", enabled))
        item = {
            "piece": pid,
            "file": _relative(json_path, out.parent),
            "enabled": enabled,
        }
        if piece.get("overrides"):
            item["overrides"] = dict(piece["overrides"])
        pieces.append(item)
        for rule in piece.get("hides", []):
            hides.append(
                {"piece": pid, "target": rule["piece"], "groups": list(rule["groups"])}
            )
        if "springs" in piece:
            springs[pid] = [dict(s) for s in piece["springs"]]
        line = credit_line(wearable)
        if line and line not in lines:
            lines.append(line)
    if not pieces:
        raise CardError(
            f"The outfit {outfit!r} has no pieces.",
            "no_pieces",
            "Add pieces with 'wearables add' first.",
        )
    data = {
        "format": FORMAT,
        "id": outfit,
        "label": label or entry.get("label", outfit),
    }
    text = description if description is not None else entry.get("description")
    if text:
        data["description"] = text
    words = list(dict.fromkeys([*entry.get("tags", []), *tags]))
    if words:
        data["tags"] = words
    data["provenance"] = provenance
    data["pieces"] = pieces
    if hides:
        data["hides"] = hides
    if springs:
        data["springs"] = springs
    if lines:
        data["credits"] = lines
    check_outfit(data, str(out))
    check_destination(out, provenance)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    exists = out.is_file()
    same = exists and out.read_text(encoding="utf-8") == text
    if exists and not same and not replace:
        raise OutfitError(
            f"{out} exists with other contents.",
            "file_exists",
            "Pass --replace to write over it, or choose another --out.",
        )
    changed = not same
    if changed and not dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return {
        "ok": True,
        "command": "wearables outfit export",
        "card": str(path),
        "outfit_file": str(out),
        "dry_run": bool(dry_run),
        "changed": changed,
        "written": changed and not dry_run,
        "outfit": data,
    }


def add_outfit(
    card,
    file,
    *,
    outfit: str | None = None,
    label: str | None = None,
    activate: bool = False,
    replace: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Put an outfit file on a card as the outfit of its id (``outfit`` renames it, ``label``
    relabels it): the pieces with paths relative to the card, each with its ``overrides``,
    the hides rules of the file as the hiding piece's ``hides`` and its ``springs``, and the
    file as the outfit's ``source``. Each package is read and checked first; a private one
    never goes on a card inside a checkout of the repository. The same outfit again changes
    nothing; an outfit of that id with other contents is an error (``outfit_exists``) unless
    ``replace``, which also takes out the card's overrides of the old pieces. ``activate``
    makes it the worn outfit.
    """
    from opensculptboy.wearables.cardedit import (
        CardError,
        _drop_overrides,
        _finish,
        _load_package,
        _read_card,
        _upgrade,
    )
    from opensculptboy.wearables.wardrobe import NO_OUTFIT

    path, before = _read_card(card)
    source, data = read_outfit(file)
    oid = outfit or data["id"]
    if "/" in oid or oid == NO_OUTFIT or not oid:
        raise CardError(
            f"An outfit id holds no '/' and is not '{NO_OUTFIT}': {oid!r}.",
            "bad_outfit",
            "Choose another outfit name.",
        )
    root = repository_root(path)
    result = {
        "command": "wearables add",
        "outfit_file": str(source),
        "outfit": oid,
        "added": [],
        "updated": [],
        "removed": [],
        "unchanged": [],
        "warnings": [],
    }
    rules: dict[str, list] = {}
    ids = [p["piece"] for p in data["pieces"]]
    for rule in data.get("hides", []):
        target = rule["target"]
        if "/" not in target and target not in ids:
            result["warnings"].append(
                f"hides rule of {rule['piece']!r}: {target!r} is not a piece of the "
                "outfit; it names a loose piece of the card"
            )
        rules.setdefault(rule["piece"], []).append(
            {"piece": target, "groups": list(rule["groups"])}
        )
    pieces = []
    for piece in data["pieces"]:
        package = _resolve(piece["file"], source.parent)
        json_path, wearable = _load_package(package)
        if wearable.provenance == PRIVATE and root is not None:
            raise ProvenanceError(
                f"{json_path}: a private package never goes on a card inside the "
                f"repository tree ({root}).",
                hint="Keep the card and its private packages in a folder outside the "
                "checkout.",
            )
        for rule in rules.get(piece["piece"], []):
            other = rule["piece"]
            if other in ids:
                target_path = _resolve(
                    data["pieces"][ids.index(other)]["file"], source.parent
                )
                _, target = _load_package(target_path)
                missing = sorted(set(rule["groups"]) - set(target.garment.groups))
                if missing:
                    raise OutfitError(
                        f"{source}: the hides rule of {piece['piece']!r} names groups "
                        f"{missing} that {other!r} lacks.",
                        "unknown_group",
                        f"The groups of {other!r} are {target.garment.groups}.",
                    )
        entry = {
            "piece": piece["piece"],
            "file": _relative(json_path, path.parent),
            "enabled": bool(piece.get("enabled", True)),
        }
        if piece.get("overrides"):
            entry["overrides"] = dict(piece["overrides"])
        if piece["piece"] in rules:
            entry["hides"] = rules[piece["piece"]]
        if piece["piece"] in data.get("springs", {}):
            entry["springs"] = [dict(s) for s in data["springs"][piece["piece"]]]
        pieces.append(entry)
    new = {"id": oid, "label": label or data.get("label") or oid}
    if data.get("description"):
        new["description"] = data["description"]
    if data.get("tags"):
        new["tags"] = list(data["tags"])
    new["source"] = _relative(source, path.parent)
    new["pieces"] = pieces
    after = copy.deepcopy(before)
    outfits = after.setdefault("outfits", [])
    old = next((o for o in outfits if o.get("id") == oid), None)
    refs = [f"{oid}/{p['piece']}" for p in pieces]
    if old is None:
        outfits.append(new)
        result["added"] = refs
        result["created_outfit"] = oid
    elif old == new:
        result["unchanged"] = refs
    elif replace:
        old_refs = [f"{oid}/{p['piece']}" for p in old.get("pieces", [])]
        outfits[outfits.index(old)] = new
        result["removed"] = [r for r in old_refs if r not in refs]
        result["updated"] = [r for r in refs if r in old_refs]
        result["added"] = [r for r in refs if r not in old_refs]
        overrides = after.get("wearable_overrides") or {}
        for key in [k for k in overrides if k.startswith(f"{oid}/")]:
            del overrides[key]
            result.setdefault("removed_overrides", []).append(key)
    else:
        raise CardError(
            f"The card has an outfit {oid!r} with other contents.",
            "outfit_exists",
            "Pass --replace to put the file's outfit in its place, or --outfit NAME to "
            "add it under another id.",
        )
    if activate:
        after["active_outfit"] = oid
    _drop_overrides(before, after, result)
    _upgrade(after)
    return _finish(path, before, after, result, dry_run)

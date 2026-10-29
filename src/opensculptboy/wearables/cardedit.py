# OpenSculptBoy
# Apache License, Version 2.0
"""
Changes to the wardrobe of a character card (``opensculptboy wearables add``, ``remove`` and
``outfit``, docs/wearables.md, "Free outfits").

Each function reads the card as JSON, changes its wardrobe fields (``outfits``,
``active_outfit``, ``wearables`` and ``wearable_overrides``), keeps every other field as it
was, raises an older card to schema 4 when the change adds wardrobe fields, checks the result
as a :class:`~opensculptboy.character.Character`, and writes it unless ``dry_run``. Package
paths are written relative to the card. A function run twice with the same arguments changes
the card once: the second run reports ``changed: false``.

A private package never goes on a card stored inside a checkout of this repository (the
provenance guard of :mod:`opensculptboy.wearables.format`).
"""

from __future__ import annotations

import copy
import json
import os
import pathlib

from opensculptboy.wearables.format import (
    PRIVATE,
    ProvenanceError,
    WearableError,
    package_paths,
    read_wearable,
    repository_root,
)
from opensculptboy.wearables.wardrobe import NO_OUTFIT, resolve_ref

WARDROBE = ("outfits", "active_outfit", "wearables", "wearable_overrides")


class CardError(WearableError):
    """A card change that cannot be made as asked."""

    code = "invalid_card"


def _read_card(path) -> tuple[pathlib.Path, dict]:
    from opensculptboy.character import Character

    path = pathlib.Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CardError(
            f"{path}: no such card.",
            "card_not_found",
            "Name an existing card; 'opensculptboy character > card.json' writes one.",
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise CardError(
            f"{path}: not a JSON card ({error}).",
            "invalid_card",
            "Name a character card (JSON).",
        ) from error
    try:
        Character.from_dict(raw)
    except ValueError as error:
        raise CardError(
            f"{path}: {error}", "invalid_card", "Fix the card's fields first."
        ) from error
    return path, raw


def _finish(path: pathlib.Path, before: dict, after: dict, result: dict, dry_run: bool):
    """Check the changed card, write it when it changed and not ``dry_run``, and complete
    the result."""
    from opensculptboy.character import Character

    try:
        Character.from_dict(after)
    except ValueError as error:
        raise CardError(
            f"{path}: the change would make an invalid card: {error}",
            "invalid_card",
            "Check the packages and the outfit names.",
        ) from error
    changed = after != before
    result.update(
        {
            "ok": True,
            "card": str(path),
            "dry_run": bool(dry_run),
            "changed": changed,
            "written": changed and not dry_run,
            "schema": {"before": before.get("schema"), "after": after.get("schema")},
            "wardrobe": {k: after[k] for k in WARDROBE if k in after},
        }
    )
    if changed and not dry_run:
        path.write_text(json.dumps(after, indent=2) + "\n", encoding="utf-8")
    return result


def _upgrade(raw: dict) -> None:
    """Raise a card that sets wardrobe fields to schema 4."""
    if any(raw.get(k) for k in WARDROBE) and raw.get("schema", 4) < 4:
        raw["schema"] = 4


def _refs(raw: dict) -> list[str]:
    """The references of every piece of a card (as JSON)."""
    refs = [
        f"{o['id']}/{p['piece']}"
        for o in raw.get("outfits", [])
        for p in o.get("pieces", [])
    ]
    refs += [
        e.get("id") or pathlib.Path(e["file"]).name.split(".")[0]
        for e in raw.get("wearables", [])
    ]
    return refs


def _named(key: str, raw: dict) -> str | None:
    """The piece that an override key names on a card, or None."""
    active = raw.get("active_outfit") or None
    try:
        return resolve_ref(key, _refs(raw), None if active == NO_OUTFIT else active)
    except ValueError:
        return None


def _drop_overrides(before: dict, after: dict, result: dict) -> None:
    """Take out the overrides whose pieces the change took off, so that no override names
    a missing piece (an export would refuse the card)."""
    overrides = after.get("wearable_overrides") or {}
    for key in list(overrides):
        if _named(key, before) is not None and _named(key, after) is None:
            del overrides[key]
            result.setdefault("removed_overrides", []).append(key)


def _relative(package: pathlib.Path, card: pathlib.Path) -> str:
    return pathlib.Path(
        os.path.relpath(package.resolve(), card.resolve().parent)
    ).as_posix()


def _load_package(path):
    paths = package_paths(path)
    if not paths["json"].is_file():
        raise CardError(
            f"{paths['json']}: no such package.",
            "package_not_found",
            "Name the .wearable.json file of a package (import-mhclo and bind write it).",
        )
    try:
        return paths["json"], read_wearable(paths["json"])
    except WearableError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise CardError(
            f"{paths['json']}: not a valid package: {error}",
            "invalid_package",
            "Write the package again with import-mhclo or bind.",
        ) from error


def _slot_of(card: pathlib.Path, file: str) -> str | None:
    path = pathlib.Path(file)
    if not path.is_absolute():
        path = card.parent / path
    try:
        return read_wearable(path).slot
    except (OSError, ValueError, KeyError):
        return None


def add_pieces(
    card,
    packages,
    *,
    outfit: str | None = None,
    label: str | None = None,
    activate: bool = False,
    loose: bool = False,
    replace: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Put packages on a card: into ``outfit`` (created when missing, with ``label``), or as
    loose pieces (without ``outfit``, or with ``loose``). A piece takes the package's name as
    its id. A piece of that id with another file is an error unless ``replace``; ``replace``
    also takes out the pieces of the same slot in the same place (``accessory`` and ``prop`` pieces only
    by id) with their overrides. ``activate`` makes ``outfit`` the worn outfit.
    """
    path, before = _read_card(card)
    after = copy.deepcopy(before)
    if outfit is not None and loose:
        raise CardError(
            "--outfit and --loose name two places.",
            "bad_option",
            "Pass --outfit NAME for an outfit, or --loose for loose pieces.",
        )
    if activate and outfit is None:
        raise CardError(
            "--activate needs --outfit.",
            "bad_option",
            "Pass --outfit NAME with --activate.",
        )
    if outfit is not None and ("/" in outfit or outfit == NO_OUTFIT or not outfit):
        raise CardError(
            f"An outfit id holds no '/' and is not '{NO_OUTFIT}': {outfit!r}.",
            "bad_outfit",
            "Choose another outfit name.",
        )
    if not packages:
        raise CardError(
            "No packages to add.", "bad_option", "Name one or more packages."
        )
    root = repository_root(path)
    loaded = []
    for item in packages:
        json_path, wearable = _load_package(item)
        if wearable.provenance == PRIVATE and root is not None:
            raise ProvenanceError(
                f"{json_path}: a private package never goes on a card inside the "
                f"repository tree ({root}).",
                hint="Keep the card and its private packages in a folder outside the "
                "checkout.",
            )
        loaded.append((json_path, wearable))
    result = {
        "command": "wearables add",
        "added": [],
        "updated": [],
        "removed": [],
        "unchanged": [],
        "warnings": [],
    }
    if outfit is not None:
        outfits = after.setdefault("outfits", [])
        entry = next((o for o in outfits if o.get("id") == outfit), None)
        if entry is None:
            entry = {"id": outfit, "label": label or outfit, "pieces": []}
            outfits.append(entry)
            result["created_outfit"] = outfit
        elif label is not None and entry.get("label") != label:
            entry["label"] = label
        pieces = entry.setdefault("pieces", [])
        id_key = "piece"
    else:
        pieces = after.setdefault("wearables", [])
        id_key = "id"

    def piece_id(e):
        return e.get(id_key) or pathlib.Path(e["file"]).name.split(".")[0]

    for json_path, wearable in loaded:
        pid = wearable.name.replace("/", "_")
        ref = f"{outfit}/{pid}" if outfit is not None else pid
        rel = _relative(json_path, path)
        new = {id_key: pid, "file": rel, "enabled": True}
        if replace and wearable.slot not in ("accessory", "prop"):
            for other in list(pieces):
                oid = piece_id(other)
                if oid != pid and _slot_of(path, other["file"]) == wearable.slot:
                    pieces.remove(other)
                    result["removed"].append(f"{outfit}/{oid}" if outfit else oid)
        existing = next((e for e in pieces if piece_id(e) == pid), None)
        if existing is None:
            clash = [
                piece_id(e)
                for e in pieces
                if wearable.slot not in ("accessory", "prop")
                and _slot_of(path, e["file"]) == wearable.slot
            ]
            if clash:
                result["warnings"].append(
                    f"{ref} shares the slot {wearable.slot!r} with {clash}; --replace "
                    "takes them out"
                )
            pieces.append(new)
            result["added"].append({"ref": ref, "file": rel, "slot": wearable.slot})
        elif existing.get("file") == rel and existing.get("enabled", True):
            result["unchanged"].append(ref)
        elif existing.get("file") == rel or replace:
            existing.update(new)
            result["updated"].append({"ref": ref, "file": rel, "slot": wearable.slot})
        else:
            raise CardError(
                f"{ref} is already on the card with the file {existing.get('file')!r}.",
                "piece_exists",
                "Pass --replace to put the new package in its place.",
            )
    if activate:
        after["active_outfit"] = outfit
    _drop_overrides(before, after, result)
    _upgrade(after)
    if after.get("wearables") == [] and "wearables" not in before:
        del after["wearables"]
    return _finish(path, before, after, result, dry_run)


def remove_pieces(card, refs=(), *, outfit: str | None = None, dry_run: bool = False):
    """
    Take pieces off a card by reference (``<outfit>/<piece>``, a loose piece's id, or a bare
    id of a piece of the worn outfit), with their overrides; ``outfit`` takes a whole outfit
    off. References that the card lacks are reported in ``missing`` and change nothing.
    """
    path, before = _read_card(card)
    after = copy.deepcopy(before)
    result = {
        "command": "wearables remove",
        "removed": [],
        "missing": [],
        "warnings": [],
    }
    outfits = after.get("outfits", [])
    loose = after.get("wearables", [])
    overrides = after.get("wearable_overrides", {})
    active = after.get("active_outfit") or None
    # a bare id names a piece of the worn outfit; with no outfit worn, of any outfit
    worn = None if active == NO_OUTFIT else active

    def loose_id(e):
        return e.get("id") or pathlib.Path(e["file"]).name.split(".")[0]

    if outfit is not None:
        entry = next((o for o in outfits if o.get("id") == outfit), None)
        if entry is None:
            result["missing"].append(outfit)
        else:
            outfits.remove(entry)
            result["removed"].append(outfit)
            for key in [k for k in overrides if k.startswith(f"{outfit}/")]:
                del overrides[key]
            if active == outfit:
                after.pop("active_outfit", None)
    for ref in refs:
        found = None
        if "/" in ref:
            name, _, pid = ref.partition("/")
            entry = next((o for o in outfits if o.get("id") == name), None)
            if entry is not None:
                piece = next((p for p in entry["pieces"] if p["piece"] == pid), None)
                if piece is not None:
                    entry["pieces"].remove(piece)
                    found = ref
        else:
            piece = next((e for e in loose if loose_id(e) == ref), None)
            if piece is not None:
                loose.remove(piece)
                found = ref
            else:
                matches = [
                    (o, p)
                    for o in outfits
                    for p in o.get("pieces", [])
                    if p["piece"] == ref and (worn is None or o["id"] == worn)
                ]
                if len(matches) > 1:
                    raise CardError(
                        f"{ref!r} names a piece of several outfits: "
                        f"{[o['id'] + '/' + ref for o, _ in matches]}.",
                        "ambiguous_piece",
                        "Name the piece as OUTFIT/PIECE.",
                    )
                if matches:
                    o, p = matches[0]
                    o["pieces"].remove(p)
                    found = f"{o['id']}/{ref}"
        if found is None:
            result["missing"].append(ref)
            continue
        result["removed"].append(found)
        for key in (found, ref):
            overrides.pop(key, None)
    _drop_overrides(before, after, result)
    if result["missing"]:
        result["warnings"].append(
            f"not on the card (nothing to remove): {result['missing']}"
        )
    for key in ("wearable_overrides",):
        if key in after and not after[key] and key not in before:
            del after[key]
    return _finish(path, before, after, result, dry_run)


def activate_outfit(card, outfit: str, *, dry_run: bool = False) -> dict:
    """Make ``outfit`` the worn outfit (``none`` for no outfit)."""
    path, before = _read_card(card)
    after = copy.deepcopy(before)
    ids = [o.get("id") for o in after.get("outfits", [])]
    if outfit != NO_OUTFIT and outfit not in ids:
        raise CardError(
            f"Unknown outfit {outfit!r}; the outfits are {ids}.",
            "unknown_outfit",
            "Name one of the card's outfits, or 'none'.",
        )
    after["active_outfit"] = outfit
    _upgrade(after)
    return _finish(path, before, after, {"command": "wearables outfit"}, dry_run)

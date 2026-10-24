# OpenSculptBoy
# Apache License, Version 2.0
"""
What a character wears: the card's outfits, its loose wearables and the choices of one export.

A card (schema 4) lists loose pieces in ``wearables`` (``{"id", "file", "enabled"}``) and
outfits in ``outfits`` (``{"id", "label", "pieces": [{"piece", "file", "enabled"}]}``), names
the outfit it wears in ``active_outfit`` and may switch single pieces in
``wearable_overrides`` (``{"<outfit>/<piece>" or "<piece>": {"enabled": bool}}``). A piece is
named by its reference: ``<outfit>/<piece>`` for a piece of an outfit, the bare id for a
loose piece; a bare id also names the piece of that id in the worn outfit.

:func:`worn_pieces` resolves the choices in this order: the worn outfit's pieces as the card
enables them, the enabled loose pieces, the card's overrides, then the ``hide`` and ``show``
lists of the export. A piece shown from another outfit, and a loose piece, replace the pieces
of the same slot that come from elsewhere (slot replacement; ``accessory`` pieces never
replace). Last, the ``hides`` rules of the shown pieces hide the named groups of other shown
pieces; hiding the piece that hides them restores them.
"""

from __future__ import annotations

import dataclasses
import difflib
import pathlib

from opensculptboy.wearables.format import read_wearable

NO_OUTFIT = "none"


@dataclasses.dataclass
class WornPiece:
    """One piece of the wardrobe and whether it shows."""

    ref: str
    piece: str
    outfit: str | None
    path: str
    wearable: object
    shown: bool
    hidden_groups: list = dataclasses.field(default_factory=list)
    hideable_groups: list = dataclasses.field(default_factory=list)


def check_wardrobe(character) -> None:
    """Raise ValueError for wardrobe fields of a card that do not hold together."""
    seen = set()
    for entry in character.wearables:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise ValueError(f"Each wearables entry needs a 'file': {entry!r}.")
        unknown = set(entry) - {"id", "file", "enabled", "params"}
        if unknown:
            raise ValueError(f"Unknown keys {sorted(unknown)} in a wearables entry.")
        ref = entry.get("id") or pathlib.Path(entry["file"]).name.split(".")[0]
        if ref in seen:
            raise ValueError(f"Two loose wearables share the id {ref!r}.")
        seen.add(ref)
    outfits = set()
    for outfit in character.outfits:
        if not isinstance(outfit, dict) or not isinstance(outfit.get("id"), str):
            raise ValueError(f"Each outfit needs an 'id': {outfit!r}.")
        if "/" in outfit["id"] or outfit["id"] == NO_OUTFIT:
            raise ValueError(
                f"An outfit id holds no '/' and is not 'none': {outfit!r}."
            )
        if outfit["id"] in outfits:
            raise ValueError(f"Two outfits share the id {outfit['id']!r}.")
        outfits.add(outfit["id"])
        unknown = set(outfit) - {"id", "label", "pieces"}
        if unknown:
            raise ValueError(
                f"Unknown keys {sorted(unknown)} in outfit {outfit['id']!r}."
            )
        pieces = set()
        for piece in outfit.get("pieces", []):
            if not isinstance(piece, dict) or not isinstance(piece.get("piece"), str):
                raise ValueError(
                    f"Each piece of an outfit needs a 'piece' id: {piece!r}."
                )
            if not isinstance(piece.get("file"), str):
                raise ValueError(f"Piece {piece['piece']!r} needs a 'file'.")
            if piece["piece"] in pieces:
                raise ValueError(
                    f"Outfit {outfit['id']!r} names piece {piece['piece']!r} twice."
                )
            pieces.add(piece["piece"])
    active = character.active_outfit
    if active and active != NO_OUTFIT and active not in outfits:
        raise ValueError(
            f"active_outfit {active!r} is not one of the outfits {sorted(outfits)}."
        )
    for key, value in character.wearable_overrides.items():
        if not isinstance(value, dict) or set(value) - {"enabled"}:
            raise ValueError(
                f"wearable_overrides[{key!r}] must be {{'enabled': true or false}}."
            )


def _entries(character):
    """(ref, piece id, outfit id or None, file, enabled) of every piece of the card."""
    for outfit in character.outfits:
        for piece in outfit.get("pieces", []):
            yield (
                f"{outfit['id']}/{piece['piece']}",
                piece["piece"],
                outfit["id"],
                piece["file"],
                bool(piece.get("enabled", True)),
            )
    for entry in character.wearables:
        ref = entry.get("id") or pathlib.Path(entry["file"]).name.split(".")[0]
        yield ref, ref, None, entry["file"], bool(entry.get("enabled", True))


def resolve_ref(name: str, refs: list[str], outfit: str | None) -> str:
    """The reference that ``name`` gives: itself, or ``<outfit>/<name>`` for a bare id of a
    piece of the worn outfit. Raises ValueError with suggestions for an unknown name."""
    if name in refs:
        return name
    if outfit and f"{outfit}/{name}" in refs:
        return f"{outfit}/{name}"
    matches = [r for r in refs if r.split("/")[-1] == name]
    if len(matches) == 1:
        return matches[0]
    bare = {r.split("/")[-1]: r for r in refs}
    close = difflib.get_close_matches(name, refs, n=3) or [
        bare[m] for m in difflib.get_close_matches(name, list(bare), n=3)
    ]
    hint = f" Did you mean {close}?" if close else f" The pieces are {refs}."
    if len(matches) > 1:
        hint = f" It names a piece of several outfits: {matches}."
    raise ValueError(f"Unknown piece {name!r}.{hint}")


def worn_pieces(
    character,
    detail_dir=None,
    outfit: str | None = None,
    show=(),
    hide=(),
    every_outfit: bool = False,
) -> list[WornPiece]:
    """
    The pieces of a card with what shows (see the module documentation). ``outfit`` overrides
    the card's ``active_outfit`` (``"none"``: no outfit). ``every_outfit`` also returns the
    pieces that do not show (for a wardrobe file); otherwise only the shown ones.
    """
    entries = list(_entries(character))
    refs = [e[0] for e in entries]
    outfit_ids = [o["id"] for o in character.outfits]
    active = character.active_outfit if outfit is None else outfit
    if not active or active == NO_OUTFIT:  # a card of loose pieces names no outfit
        active = None
    if active is not None and active not in outfit_ids:
        close = difflib.get_close_matches(active, outfit_ids, n=3)
        raise ValueError(
            f"Unknown outfit {active!r}; the outfits are {outfit_ids}"
            + (f" (did you mean {close}?)." if close else ".")
        )
    shown = {}
    for ref, _, owner, _, enabled in entries:
        shown[ref] = enabled if (owner is None or owner == active) else False
    for key, value in character.wearable_overrides.items():
        shown[resolve_ref(key, refs, active)] = bool(value["enabled"])
    base = detail_dir or "."
    loaded = {}

    def load(ref):
        if ref not in loaded:
            entry = entries[refs.index(ref)]
            path = pathlib.Path(entry[3])
            if not path.is_absolute():
                path = pathlib.Path(base) / path
            loaded[ref] = (read_wearable(path), str(path))
        return loaded[ref][0]

    for name in hide:
        shown[resolve_ref(name, refs, active)] = False
    shown_from_elsewhere = []
    for name in show:
        ref = resolve_ref(name, refs, active)
        shown[ref] = True
        owner = entries[refs.index(ref)][2]
        if owner != active:
            shown_from_elsewhere.append(ref)
    # slot replacement: a piece shown from another source replaces the pieces of its slot
    replacers = shown_from_elsewhere + [
        e[0] for e in entries if e[2] is None and shown[e[0]]
    ]
    for ref in replacers:
        slot = load(ref).slot
        if slot == "accessory":
            continue
        source = entries[refs.index(ref)][2]
        for other, _, owner, _, _ in entries:
            if other != ref and shown[other] and owner != source:
                if load(other).slot == slot and other not in replacers:
                    shown[other] = False
    candidates = [e for e in entries if shown[e[0]] or every_outfit]
    pieces = []
    for ref, piece, owner, _, _ in candidates:
        wearable = load(ref)
        pieces.append(
            WornPiece(ref, piece, owner, loaded[ref][1], wearable, bool(shown[ref]))
        )
    by_ref = {p.ref: p for p in pieces}
    for p in pieces:
        for rule in p.wearable.hides:
            target = rule["piece"]
            if "/" not in target and p.outfit is not None:
                target = f"{p.outfit}/{target}"
            other = by_ref.get(target)
            if other is None:
                continue
            for group in rule.get("groups", []):
                if group not in other.hideable_groups:
                    other.hideable_groups.append(group)
                if p.shown and other.shown and group not in other.hidden_groups:
                    other.hidden_groups.append(group)
    return pieces

# OpenSculptBoy
# Apache License, Version 2.0
"""
A catalogue of the wearable packages in a folder (``opensculptboy wearables catalogue``,
docs/wearables.md, "Catalogue").

:func:`catalogue` walks a folder for ``*.wearable.json`` packages and, on request, for
MakeHuman clothes assets (``*.mhclo``) that no listed package was imported from, and returns
one entry per piece: its name, label, slot, layer, tags and description, the mean base colour
of each material as hex (:func:`material_colours`), the groups, the counts of vertices,
triangles and materials, a licence summary with the credit line of the exports, a quality
summary from the checks of :mod:`opensculptboy.wearables.quality` run on the package as it
stands, and the path of a thumbnail. :func:`thumbnail` renders a package on the default body
into a small PNG file next to it, which later runs reuse while it is newer than the package's
files. The entries are sorted by kind and by path, so that the order stays the same between
runs; the filters (slot, tags, words) keep the entries that match.
"""

from __future__ import annotations

import copy
import io
import json
import pathlib

import numpy as np

from opensculptboy.wearables.format import (
    PRIVATE,
    SLOTS,
    WearableError,
    check_destination,
    package_paths,
    read_wearable,
)

FORMAT = "opensculptboy.catalogue/1"
THUMBNAIL_SIZE = 256
THUMBNAIL_SUFFIX = ".thumb.png"
KINDS = ("package", "mhclo")


class CatalogueError(WearableError):
    """A catalogue that cannot be made as asked."""

    code = "invalid_input"


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def hex_colour(linear) -> str:
    """The sRGB hex code (``#rrggbb``) of a linear RGB colour."""
    srgb = np.round(linear_to_srgb(np.asarray(linear)[:3]) * 255).astype(int)
    return "#" + "".join(f"{int(v):02x}" for v in srgb)


def mean_colour(material) -> np.ndarray:
    """The mean base colour of a material in linear RGB: its base colour factor times the
    mean of its base colour texture (sRGB, weighted by the texture's alpha where it has
    one)."""
    factor = np.asarray(material.base_color[:3], dtype=np.float64)
    data = material.textures.get("base_color")
    if not data:
        return factor
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(data))
        image.thumbnail((256, 256))
        pixels = np.asarray(image.convert("RGBA"), dtype=np.float64) / 255.0
    except (OSError, ValueError):
        return factor
    pixels = pixels.reshape(-1, 4)
    weight = pixels[:, 3] if pixels[:, 3].min() < 1.0 else np.ones(len(pixels))
    if weight.sum() <= 0:
        weight = np.ones(len(pixels))
    linear = srgb_to_linear(pixels[:, :3])
    mean = (linear * weight[:, None]).sum(axis=0) / weight.sum()
    return factor * mean


def material_colours(garment) -> list[dict]:
    """``[{"material", "hex"}]``: the mean base colour of each material that a triangle
    uses, in the order of the materials."""
    used = set(np.unique(garment.triangle_material).tolist())
    return [
        {"material": m.name, "hex": hex_colour(mean_colour(m))}
        for k, m in enumerate(garment.materials)
        if k in used
    ]


def triangle_colours(garment) -> np.ndarray:
    """(T, 3) the sRGB colour (0 to 255) of each triangle, from its material's mean."""
    table = np.array(
        [linear_to_srgb(mean_colour(m)) * 255.0 for m in garment.materials]
    ).reshape(-1, 3)
    return table[garment.triangle_material]


def licence_summary(wearable) -> dict:
    """The licence as recorded (``spdx``, ``declared``, ``author``, ``source``) and the credit
    line that the exports write (``credit``, null without one)."""
    from opensculptboy.wearables.export import credit_line

    licence = wearable.licence or {}
    out = {k: licence.get(k) for k in ("spdx", "declared", "author", "source")}
    out["credit"] = credit_line(wearable)
    return out


def quality_summary(wearable, body=None) -> dict:
    """The quality checks of :func:`opensculptboy.wearables.quality.check_garment` on a copy
    of the package's garment: the counts by severity, the codes of each severity, and the
    items of error and warning severity. A package keeps the fixes that its import made, so
    a check of the package finds what remains."""
    from opensculptboy.wearables.quality import check_garment

    _, _, quality = check_garment(copy.deepcopy(wearable.garment), body=body)
    items = [i.to_dict() for i in quality.items]
    codes = {
        s: sorted({i["code"] for i in items if i["severity"] == s})
        for s in ("error", "warning", "info")
    }
    return {
        "errors": len(quality.of("error")),
        "warnings": len(quality.of("warning")),
        "infos": len(quality.of("info")),
        "codes": codes,
        "items": [i for i in items if i["severity"] != "info"],
    }


def package_entry(path: pathlib.Path, root: pathlib.Path, quality=True) -> dict:
    """The catalogue entry of one package (see the module documentation)."""
    wearable = read_wearable(path)
    garment = wearable.garment
    entry = {
        "kind": "package",
        "path": _rel(path, root),
        "name": wearable.name,
        "label": wearable.label or wearable.name,
        "slot": wearable.slot,
        "layer": int(wearable.layer),
        "tags": list(wearable.tags),
        "description": wearable.description,
        "provenance": wearable.provenance,
        "colours": material_colours(garment),
        "groups": list(garment.groups),
        "vertices": int(len(garment.vertices)),
        "triangles": int(len(garment.triangles)),
        "materials": int(len(garment.materials)),
        "binding": {"body": wearable.binding.body, "method": wearable.binding.method},
        "weights": wearable.weights,
        "hides": list(wearable.hides),
        "licence": licence_summary(wearable),
        "quality": quality_summary(wearable) if quality else None,
        "thumbnail": None,
    }
    source = (wearable.extras or {}).get("mhclo")
    if isinstance(source, dict):
        entry["mhclo"] = {k: source.get(k) for k in ("name", "uuid", "file")}
    return entry


def mhclo_entry(path: pathlib.Path, root: pathlib.Path) -> dict:
    """The catalogue entry of a MakeHuman clothes asset that is not imported yet: what its
    header, its pack metadata and its material say, and the command that imports it."""
    from opensculptboy.wearables.mhclo import (
        DEFAULT_LAYER,
        find_pack_entry,
        licence_block,
        read_mhclo,
        read_mhmat,
        slot_of,
    )
    from opensculptboy.wearables.make import clean_tags
    from opensculptboy.wearables.quality import Quality

    asset = read_mhclo(path)
    pack, _ = find_pack_entry(path)
    name = asset.name or path.stem
    slot, reason = slot_of(asset, path, pack)
    tags = clean_tags([*asset.tags, *(pack.category_tags() if pack else [])])
    licence = licence_block(asset, name, pack=pack)
    colours = []
    if asset.material is not None and asset.material.is_file():
        try:
            material = read_mhmat(asset.material, name, Quality())
            colours = [
                {"material": material.name, "hex": hex_colour(mean_colour(material))}
            ]
        except (OSError, ValueError, WearableError):
            colours = []
    triangles = None
    obj = asset.obj_file or path.with_suffix(".obj")
    if obj.is_file():
        triangles = _obj_triangles(obj)
    stem = path.with_suffix("")
    return {
        "kind": "mhclo",
        "path": _rel(path, root),
        "name": name,
        "label": name,
        "slot": slot,
        "slot_from": reason,
        "layer": int(asset.z_depth) if asset.z_depth is not None else DEFAULT_LAYER,
        "tags": tags,
        "description": (pack.get("description") if pack else None)
        or asset.description
        or "",
        "provenance": "public",
        "colours": colours,
        "groups": [],
        "vertices": None if asset.refs is None else int(len(asset.refs)),
        "triangles": triangles,
        "materials": len(colours),
        "licence": {
            "spdx": licence.get("spdx"),
            "declared": licence.get("declared"),
            "author": licence.get("author"),
            "source": licence.get("source"),
            "credit": licence.get("attribution"),
        },
        "quality": None,
        "thumbnail": None,
        "mhclo": {"name": asset.name, "uuid": asset.uuid, "file": path.name},
        "imported": False,
        "import": [
            "opensculptboy",
            "wearables",
            "import-mhclo",
            str(path),
            "--out",
            str(stem.parent / "wear" / stem.name),
        ],
    }


def _obj_triangles(path: pathlib.Path) -> int:
    count = 0
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("f "):
                count += max(len(line.split()) - 3, 0)
    return count


def _rel(path: pathlib.Path, root: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _words(text: str) -> list[str]:
    import re

    return [w for w in re.split(r"[^\w]+", text.casefold()) if w]


def matches(entry: dict, slots=(), tags=(), query=()) -> bool:
    """Whether an entry passes the filters: one of ``slots``, every tag of ``tags`` (case
    folded), and every word of ``query`` inside its name, label, tags or description (a word
    of more than three letters that ends in ``s`` also matches without it)."""
    if slots and entry.get("slot") not in slots:
        return False
    own = {t.casefold() for t in entry.get("tags", [])}
    if any(t.casefold() not in own for t in tags):
        return False
    text = " ".join(
        [
            entry.get("name") or "",
            entry.get("label") or "",
            " ".join(entry.get("tags", [])),
            entry.get("description") or "",
        ]
    ).casefold()
    for word in query:
        word = word.casefold()
        forms = [word] + ([word[:-1]] if len(word) > 3 and word.endswith("s") else [])
        if not any(f in text for f in forms):
            return False
    return True


def find(folder: pathlib.Path, pattern: str) -> list[pathlib.Path]:
    """The files of ``folder`` (and its sub-folders) that match ``pattern``, sorted by path."""
    return sorted(
        (p for p in folder.rglob(pattern) if p.is_file()),
        key=lambda p: p.relative_to(folder).as_posix(),
    )


def thumbnail_path(package: pathlib.Path) -> pathlib.Path:
    paths = package_paths(package)
    return paths["stem"].parent / f"{paths['stem'].name}{THUMBNAIL_SUFFIX}"


def thumbnail_fresh(package: pathlib.Path) -> bool:
    """Whether the package's thumbnail exists and is newer than each of its files."""
    thumb = thumbnail_path(package)
    if not thumb.is_file():
        return False
    paths = package_paths(package)
    newest = max(
        paths[k].stat().st_mtime
        for k in ("json", "glb", "binding")
        if paths[k].exists()
    )
    return thumb.stat().st_mtime >= newest


def thumbnail(package, body, size: int = THUMBNAIL_SIZE) -> bytes:
    """A PNG picture (``size`` square) of a package dressed on ``body`` (a
    :class:`~opensculptboy.wearables.bind.BodySurface`) from the front and a little to the
    figure's right, framed on the piece, with the skin that the piece covers left out."""
    from opensculptboy.render.dressed import VIEWS, Mesh, draw, png
    from opensculptboy.wearables.cover import outfit_cover
    from opensculptboy.wearables.fit import dress
    from opensculptboy.wearables.wardrobe import WornPiece

    wearable = read_wearable(package)
    worn = WornPiece(wearable.name, wearable.name, None, str(package), wearable, True)
    (piece,) = dress(body, [worn])
    covered = outfit_cover(body, [piece])
    keep = ~covered[body.triangles].all(axis=1)
    tris = piece.visible_triangles()
    colours = triangle_colours(piece.garment)
    visible = np.ones(len(piece.garment.triangles), bool)
    for g, group in enumerate(piece.garment.groups):
        if group in piece.hidden_groups:
            visible &= piece.garment.triangle_group != g
    meshes = [
        Mesh(body.vertices, body.triangles[keep], (205, 196, 190)),
        Mesh(piece.vertices, tris, colours[visible]),
    ]
    used = np.unique(tris)
    points = piece.vertices[used]
    lo, hi = points.min(axis=0), points.max(axis=0)
    centre, half = (lo + hi) / 2, np.maximum((hi - lo).max() * 0.6, 0.18)
    corners = centre + half * np.array([[-1, -1, -1], [1, 1, 1]], dtype=np.float64)
    image = draw(
        meshes,
        VIEWS["three-quarter"],
        size=(size, size),
        margin=4,
        frame_points=np.concatenate([corners, centre + half * np.eye(3)]),
    )
    return png(image)


def catalogue(
    folder,
    *,
    mhclo_folders=(),
    slots=(),
    tags=(),
    query=(),
    quality: bool = True,
    thumbnails: bool = False,
    thumbnail_size: int = THUMBNAIL_SIZE,
    refresh: bool = False,
    log=None,
) -> dict:
    """
    The catalogue of ``folder`` (see the module documentation). ``mhclo_folders`` adds the
    MakeHuman assets of those folders that no package of ``folder`` was imported from (by the
    asset's uuid, else its name). ``thumbnails`` renders the missing or stale thumbnails
    (all of them with ``refresh``). A package that cannot be read goes to ``skipped`` with
    its error code.
    """
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        raise CatalogueError(
            f"{folder}: no such folder.",
            "folder_not_found",
            "Name a folder that holds .wearable.json packages.",
        )
    for slot in slots:
        if slot not in SLOTS:
            raise CatalogueError(
                f"Unknown slot {slot!r}.",
                "unknown_slot",
                f"Use one of {', '.join(SLOTS)}.",
            )
    entries, skipped = [], []
    for path in find(folder, "*.wearable.json"):
        try:
            entries.append(package_entry(path, folder, quality))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            code = getattr(error, "code", None) or "invalid_package"
            skipped.append(
                {"path": _rel(path, folder), "code": code, "message": str(error)}
            )
    imported_uuids = {
        (e.get("mhclo") or {}).get("uuid") for e in entries if e.get("mhclo")
    } - {None}
    imported_names = {
        (e.get("mhclo") or {}).get("name") for e in entries if e.get("mhclo")
    } - {None}
    seen_assets = set()
    for extra in mhclo_folders:
        extra = pathlib.Path(extra)
        if not extra.is_dir():
            raise CatalogueError(
                f"{extra}: no such folder.",
                "folder_not_found",
                "Name a folder that holds MakeHuman clothes assets (.mhclo).",
            )
        for path in find(extra, "*.mhclo"):
            if path.resolve() in seen_assets:
                continue
            seen_assets.add(path.resolve())
            try:
                entry = mhclo_entry(path, folder)
            except (OSError, ValueError, KeyError) as error:
                code = getattr(error, "code", None) or "invalid_asset"
                skipped.append(
                    {"path": _rel(path, folder), "code": code, "message": str(error)}
                )
                continue
            uuid, name = entry["mhclo"]["uuid"], entry["mhclo"]["name"]
            if (uuid and uuid in imported_uuids) or (
                not uuid and name in imported_names
            ):
                continue
            entries.append(entry)
    entries.sort(key=lambda e: (KINDS.index(e["kind"]), e["path"]))
    kept = [e for e in entries if matches(e, slots, tags, query)]
    counts = {"rendered": 0, "cached": 0}
    if thumbnails:
        body = None
        for entry in kept:
            if entry["kind"] != "package":
                continue
            package = folder / entry["path"]
            thumb = thumbnail_path(package)
            if not refresh and thumbnail_fresh(package):
                counts["cached"] += 1
            else:
                if entry["provenance"] == PRIVATE:
                    check_destination(thumb, PRIVATE)
                if body is None:
                    body = _default_body()
                if log:
                    log(f"thumbnail {entry['path']}")
                thumb.write_bytes(thumbnail(package, body, thumbnail_size))
                counts["rendered"] += 1
            entry["thumbnail"] = _rel(thumb, folder)
    else:
        for entry in kept:
            if entry["kind"] == "package":
                thumb = thumbnail_path(folder / entry["path"])
                if thumb.is_file():
                    entry["thumbnail"] = _rel(thumb, folder)
    return {
        "format": FORMAT,
        "folder": str(folder),
        "filters": {"slot": list(slots), "tag": list(tags), "query": list(query)},
        "count": len(kept),
        "total": len(entries),
        "pieces": kept,
        "skipped": skipped,
        "thumbnails": counts if thumbnails else None,
    }


def _default_body():
    import torch

    from opensculptboy.character import Character
    from opensculptboy.wearables.fit import card_body

    character = Character()
    return card_body(character, character.build_model(dtype=torch.float64))

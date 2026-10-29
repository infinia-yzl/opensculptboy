# OpenSculptBoy
# Apache License, Version 2.0
"""
The blinks of a card for the web viewer (docs/viewer.md, "A card's blinks").

The page draws a card's body at the level of its bake and has no facial actions of its own,
so it takes the eye targets of the GLB export from a file: :func:`page_blinks` builds the
card's file mesh at that level with its eyes, lashes and brows as
:func:`opensculptboy.export.gltf.export_glb` builds it, and keeps the targets :data:`TARGETS`
as the export writes them (the reference's blinks brought to meet, their in-between
correctives, and the squints), for the rows of the level that they move, the points of the
reference's lashes and the points of its brows. :func:`write_page_blinks` writes them next
to the card as ``<card>.blinks.safetensors`` (:data:`FORMAT`) and names the file under
``material.reference_eyes.blinks``.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

FORMAT = "opensculptboy/page-blinks@1"
# the facial actions whose targets the page plays, and the in-between correctives after them
ACTIONS = ("eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight")
# a row moves in a target when its offset reaches this length (metres)
MOVED = 1e-7


def page_blinks(character, detail_dir=None, level: int | None = None):
    """
    The eye targets of a card's GLB export at ``level`` (the bake's level by default):
    ``{"names", "level", "rows" (K,), "offsets" (P, K, 3), "lashes" (P, L, 3) or None,
    "brows" (P, B, 3) or None}``, the rows of the level's mesh, the rows of the card's lashes
    file and the rows of its brows file; None for a card without the eyes of
    :mod:`opensculptboy.eyes`.
    """
    from opensculptboy.export.body import morph_target_rows, rest_rows
    from opensculptboy.export.detail import ear_mesh, texture_images
    from opensculptboy.export.detail import resolve as resolve_detail
    from opensculptboy.eyes.blink import (
        blink_lash_offsets,
        blink_offsets,
        blink_targets,
    )
    from opensculptboy.eyes.lashes import eyed_lashes
    from opensculptboy.eyes.mesh import baked_skin, eyed_mesh
    from opensculptboy.eyes.poses import card_blink_poses
    from opensculptboy.groom.export import card_groom

    eye = character.eye_material()
    if eye is None:
        return None
    if level is None:
        level = (character.detail.get("textures") or {}).get("subdiv")
        if level is None:
            level = max(character.detail_levels(), default=0)
    model = character.build_model()
    names = [a for a in ACTIONS if a in model.facial_action_labels]
    targets, facial_rows, face_rows = morph_target_rows(model, character, names)
    rest_vertices, _ = rest_rows(model, character, facial_rows, face_rows)
    base = rest_vertices[0]
    detail = resolve_detail(model, character, int(level), detail_dir, "full")
    images = texture_images(character, None, detail_dir)
    mesh, _ = ear_mesh(character, character.skin_material(), images, detail, detail_dir)
    vertices, offsets = base, rest_vertices[1:] - base
    if detail is not None:
        vertices, offsets = detail.vertices(base), detail.targets(offsets)
    baked = None if "eye_base_color" in images else images.get("eyes")
    layout = None
    if "eye_layout" in images and "eye_base_color" not in images:
        layout = (images["eye_layout"], images.get("eye_layout_normal"))
    lashes, _ = eyed_lashes(character, detail_dir)
    brows = card_groom(character, "brows", detail_dir)
    eyed = eyed_mesh(
        model,
        mesh,
        vertices,
        eye,
        "gltf",
        baked,
        skin_baked=baked_skin(images),
        lashes=lashes,
        layout=layout,
        brows=None if brows is None else brows[0],
    )
    offsets, blinks = blink_offsets(
        eyed,
        vertices,
        offsets,
        [t.name for t in targets],
        card_blink_poses(character, detail, detail_dir, vertices),
    )
    if blinks is not None:
        targets = targets + blink_targets(blinks)
    rest = vertices
    offsets = eyed.targets(offsets)
    if blinks is not None:
        offsets = blink_lash_offsets(eyed, blinks, rest, offsets)
    count = eyed.base_count
    moved = np.linalg.norm(offsets[:, :count], axis=2).max(axis=0) >= MOVED
    rows = np.flatnonzero(moved)
    out = {
        "names": [t.name for t in targets],
        "level": int(level),
        "rows": rows,
        "offsets": offsets[:, rows],
        "lashes": None,
        "brows": None,
    }
    for name, size in (
        ("lashes", None if eyed.lashes is None else len(eyed.lashes.points)),
        ("brows", None if brows is None else len(brows[0].points)),
    ):
        if size is None:
            continue
        part = np.zeros((len(offsets), size, 3))
        start = count
        for p in eyed.parts:
            if p.name == name and p.points is not None:
                part[:, p.points] = offsets[:, start : start + len(p.rest)]
            start += len(p.rest)
        out[name] = part
    return out


def save(path, blinks: dict) -> None:
    """Write :func:`page_blinks` as a file of :data:`FORMAT`."""
    from safetensors.numpy import save_file

    tensors = {
        "rows": np.asarray(blinks["rows"], np.int64),
        "offsets": np.ascontiguousarray(blinks["offsets"], np.float32),
    }
    for name in ("lashes", "brows"):
        if blinks.get(name) is not None:
            tensors[name] = np.ascontiguousarray(blinks[name], np.float32)
    save_file(
        tensors,
        str(path),
        metadata={
            "format": FORMAT,
            "names": json.dumps(list(blinks["names"])),
            "level": json.dumps(int(blinks["level"])),
        },
    )


def load(path) -> dict:
    """A file of :data:`FORMAT` as :func:`page_blinks` gives it."""
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        meta = f.metadata() or {}
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file.")
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    return {
        "names": json.loads(meta["names"]),
        "level": json.loads(meta["level"]),
        "rows": tensors["rows"],
        "offsets": tensors["offsets"].astype(np.float64),
        "lashes": tensors.get("lashes"),
        "brows": tensors.get("brows"),
    }


def write_page_blinks(card_path) -> pathlib.Path | None:
    """
    Write the page's blinks of a fitted card (:func:`page_blinks`) as
    ``<card>.blinks.safetensors``, named under ``material.reference_eyes.blinks`` of the card,
    which is written again. Returns the path, or None (and changes nothing) for a card without
    the eyes.
    """
    from opensculptboy.character import Character

    card_path = pathlib.Path(card_path)
    character = Character.from_dict(json.loads(card_path.read_text(encoding="utf-8")))
    blinks = page_blinks(character, card_path.parent)
    if blinks is None:
        return None
    stem = (
        card_path.name[: -len(card_path.suffix)] if card_path.suffix else card_path.name
    )
    path = card_path.parent / f"{stem}.blinks.safetensors"
    save(path, blinks)
    material = dict(character.material or {})
    material["reference_eyes"] = dict(
        material.get("reference_eyes") or {}, blinks=path.name
    )
    character.material = material
    card_path.write_text(
        json.dumps(character.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return path

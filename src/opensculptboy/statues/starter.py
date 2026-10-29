# OpenSculptBoy
# Apache License, Version 2.0
"""
A fitted statue as a finished starter of the web viewer's New screen: one ``.zip`` beside the
page (``starters/<id>.zip``) that the page opens through its Open path.

The zip holds the statue's card as ``character.json`` with the detail files it names, its
pieces (a hat, locks) as wearable packages when they are given, and ``thumbnail.png``. The
card is the fitted card of :func:`opensculptboy.statues.pipeline.run_statue` with:

- ``name``: the statue's title;
- ``look`` (card schema 5, the web viewer's block): marble skin (``stone``), no page hair, as
  the carved hair is in the sculpted surface, and stone-coloured eyes;
- ``detail.fit.reference``: the scan's title, source, licence and credit, so that the credit
  travels with every copy of the card.

:func:`build_starter` writes the zip and returns what it holds; ``python -m
opensculptboy.statues starter`` is its command.
"""

from __future__ import annotations

import io
import json
import pathlib
import zipfile

import numpy as np

# the web viewer's stone for marble and its colour for the hair and the eyes
# (viewer/src/starters.ts, STONE_HAIR)
STONE = "carrara"
STONE_COLOUR = "#dcd7cf"
# the card schema that carries the look block
LOOK_SCHEMA = 5
THUMBNAIL_SIZE = (256, 384)


def statue_look(stone: str = STONE, colour: str = STONE_COLOUR) -> dict:
    """The look block of a marble statue: the stone, no page hair, stone-coloured eyes."""
    return {
        "skin": {"tone": 0.35, "undertone": 0.0, "stone": stone, "polish": 0.5},
        "hair": {"style": "bald", "color": colour},
        "eyes": {"color": colour},
    }


def starter_card(card: dict, config, levels=(0, 1, 2)) -> dict:
    """The card of the starter: the fitted ``card`` (a dict) with the statue's name, the
    marble look, the scan's credit and only the detail ``levels`` asked for."""
    out = json.loads(json.dumps(card))
    out["schema"] = LOOK_SCHEMA
    out["name"] = config.title or config.name
    detail = out.setdefault("detail", {})
    detail["levels"] = [
        e for e in detail.get("levels", []) if int(e.get("subdiv", -1)) in set(levels)
    ]
    fit = detail.setdefault("fit", {})
    fit["reference"] = {
        "title": config.title or config.name,
        "source": config.source,
        "licence": config.licence,
        "credit": config.credit,
        "scan": config.scan,
    }
    out["look"] = statue_look()
    # the schema first, as the web viewer writes its cards
    return {"schema": out.pop("schema"), **out}


def thumbnail(vertices: np.ndarray, triangles: np.ndarray, size=THUMBNAIL_SIZE):
    """A picture (PIL, RGB) of the fitted statue from the front three-quarter, in marble."""
    from opensculptboy.render.dressed import Mesh, View, draw

    return draw(
        [Mesh(np.asarray(vertices), np.asarray(triangles), (214, 210, 202))],
        View(-25.0),
        size,
        12.0,
        background=(46, 47, 50),
    )


def build_starter(
    root,
    config,
    out,
    levels=(0, 1, 2),
    pieces=(),
    picture=None,
) -> dict:
    """
    Write the starter zip ``out`` of a statue whose pipeline wrote ``root`` (the folder
    ``<out>/<name>/`` of :func:`~opensculptboy.statues.pipeline.run_statue`, with
    ``card/<name>.json`` and its detail files): the card (:func:`starter_card`) with the
    detail ``levels``, the wearable package files ``pieces`` (paths, with the card's
    ``wearables`` naming them already) and ``thumbnail.png`` (``picture``, a PNG file, or a
    drawing of the finest fitted level). Returns the entries and their sizes in bytes.
    """
    root = pathlib.Path(root)
    card_path = root / "card" / f"{root.name}.json"
    card = starter_card(
        json.loads(card_path.read_text(encoding="utf-8")), config, levels
    )
    files = {}
    for entry in card["detail"]["levels"]:
        files[entry["offsets"]] = (card_path.parent / entry["offsets"]).read_bytes()
    for piece in pieces:
        piece = pathlib.Path(piece)
        files[piece.name] = piece.read_bytes()
    if picture is not None:
        png = pathlib.Path(picture).read_bytes()
    else:
        import trimesh

        from opensculptboy.fit.handoff import GLTF_TO_ANNY

        finest = (
            max(int(e["subdiv"]) for e in card["detail"]["levels"])
            if card["detail"]["levels"]
            else 0
        )
        mesh = trimesh.load(
            str(root / "meshes" / f"fitted_level{finest}.ply"), process=False
        )
        buffer = io.BytesIO()
        thumbnail(np.asarray(mesh.vertices) @ GLTF_TO_ANNY.T, mesh.faces).save(
            buffer, "PNG"
        )
        png = buffer.getvalue()
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = {"character.json": (json.dumps(card, indent=2) + "\n").encode("utf-8")}
    entries.update(files)
    entries["thumbnail.png"] = png
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in entries.items():
            z.writestr(name, data)
    out.with_suffix(".png").write_bytes(png)
    return {
        "zip": str(out),
        "bytes": out.stat().st_size,
        "entries": {name: len(data) for name, data in entries.items()},
        "levels": [int(e["subdiv"]) for e in card["detail"]["levels"]],
    }

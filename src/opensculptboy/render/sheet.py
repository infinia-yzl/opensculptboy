# OpenSculptBoy
# Apache License, Version 2.0
"""
Pre-rendered pictures of a character for chat avatars: a portrait and an expression sheet.

:func:`expression_portraits` renders the character's head and shoulders with each expression of
:data:`EXPRESSIONS` (neutral, smile, surprised, sad and angry: the mixes of anny's ARKit facial
actions that the VRM presets ``neutral``, ``happy``, ``surprised``, ``sad`` and ``angry`` use,
from ``data/vrm/expressions.json``), in the flat toon shading of the VRM thumbnail
(:func:`opensculptboy.export.vrm.portrait_png`, :func:`opensculptboy.render.flat.shaded_png`),
on the card's level-0 mesh with its level-0 offsets. :func:`expression_sheet` lays them out in
one picture with their names. The colour of the skin is the mean of the card's base colour
texture, else its base colour factor, else the default skin.

The pictures are light: no path tracer is involved. For offline renders, the master tier's GLB
file (:mod:`opensculptboy.tiers`) imports into Blender with its subsurface skin set by
``src/opensculptboy/export/blender_skin.py``, and Cycles renders it.
"""

from __future__ import annotations

import io

import numpy as np

# (file name, VRM 1.0 preset of data/vrm/expressions.json, label)
EXPRESSIONS = (
    ("neutral", "neutral", "Neutral"),
    ("smile", "happy", "Smile"),
    ("surprised", "surprised", "Surprised"),
    ("sad", "sad", "Sad"),
    ("angry", "angry", "Angry"),
)


def expression_mix(preset: str) -> dict[str, float]:
    """The facial actions of a VRM 1.0 preset of ``data/vrm/expressions.json``."""
    from opensculptboy.export.vrm_tables import _expression_table

    return dict(_expression_table()["presets"][preset]["mix"])


def skin_colour(character, detail_dir=None) -> tuple[float, float, float, float]:
    """The linear RGBA colour of a character's skin for the flat pictures."""
    from opensculptboy.export.vrm import SKIN_COLOUR

    skin = character.skin_material()
    if skin is None:
        return SKIN_COLOUR
    factor = np.asarray(skin.base_color_factor[:3], np.float64)
    textures = skin.textures()
    if "base_color" in textures:
        from opensculptboy.texture.transfer import decode_image

        data = _path(textures, detail_dir).read_bytes()
        pixels = decode_image(data).reshape(-1, 3) / 255.0
        srgb = np.median(pixels, axis=0)
        linear = np.where(
            srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4
        )
        factor = factor * linear
    return (*(float(c) for c in factor), 1.0)


def _path(textures, detail_dir):
    import pathlib

    path = pathlib.Path(textures["base_color"])
    if not path.is_absolute() and detail_dir is not None:
        path = pathlib.Path(detail_dir) / path
    return path


def expression_portraits(
    character=None,
    model=None,
    size: int = 512,
    expressions=EXPRESSIONS,
    detail_dir=None,
    background: str = "#dfe3ea",
) -> dict[str, bytes]:
    """
    Square portraits (PNG files of ``size`` texels) of a character by expression name: the
    character's own facial actions under each expression's mix (the mix wins where both set
    an action).
    """
    import torch

    from opensculptboy.character import Character
    from opensculptboy.detail import character_offsets
    from opensculptboy.export.vrm import portrait_png

    character = character or Character()
    if model is None:
        model = character.build_model()
    offsets = character_offsets(character, 0, detail_dir)
    colour = skin_colour(character, detail_dir)
    pictures = {}
    for name, preset, _ in expressions:
        kwargs = character.model_kwargs()
        actions = dict(character.facial_actions)
        actions.update(expression_mix(preset))
        kwargs["facial_actions"] = actions or None
        with torch.no_grad():
            out = model(**kwargs)
        vertices = out["rest_vertices"][0].detach().double().cpu().numpy()
        if offsets is not None:
            vertices = vertices + offsets
        joints = out["rest_bone_poses"][0, :, :3, 3].detach().double().cpu().numpy()
        pictures[name] = portrait_png(
            model, vertices, joints, size, colour=colour, background=background
        )
    return pictures


def expression_sheet(
    portraits: dict[str, bytes],
    expressions=EXPRESSIONS,
    columns: int | None = None,
    label: bool = True,
) -> bytes:
    """One PNG file with the portraits of :func:`expression_portraits` side by side (one row
    by default), each with its label underneath."""
    from PIL import Image, ImageDraw, ImageFont

    names = [(n, text) for n, _, text in expressions if n in portraits]
    images = [Image.open(io.BytesIO(portraits[n])).convert("RGB") for n, _ in names]
    side = images[0].size[0]
    columns = columns or len(images)
    rows = -(-len(images) // columns)
    band = max(side // 10, 16) if label else 0
    sheet = Image.new("RGB", (columns * side, rows * (side + band)), "#ffffff")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=max(band * 6 // 10, 10))
    except TypeError:  # Pillow before 10.1
        font = ImageFont.load_default()
    for k, (image, (_, text)) in enumerate(zip(images, names)):
        x, y = (k % columns) * side, (k // columns) * (side + band)
        sheet.paste(image, (x, y))
        if label:
            width = draw.textlength(text, font=font)
            draw.text(
                (x + (side - width) / 2, y + side + band * 0.15), text, "#333333", font
            )
    stream = io.BytesIO()
    sheet.save(stream, format="PNG")
    return stream.getvalue()


def write_portraits(
    folder, character=None, size: int = 512, detail_dir=None, model=None
) -> dict:
    """Write ``portrait.png`` (the neutral face at twice ``size``), ``<expression>.png`` for each
    expression and ``expressions.png`` (the sheet) into ``folder``; returns their paths."""
    import pathlib

    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pictures = expression_portraits(character, model, size, detail_dir=detail_dir)
    written = {}
    for name, data in pictures.items():
        path = folder / f"{name}.png"
        path.write_bytes(data)
        written[name] = str(path)
    portrait = expression_portraits(
        character, model, 2 * size, EXPRESSIONS[:1], detail_dir
    )["neutral"]
    (folder / "portrait.png").write_bytes(portrait)
    written["portrait"] = str(folder / "portrait.png")
    (folder / "expressions.png").write_bytes(expression_sheet(pictures))
    written["sheet"] = str(folder / "expressions.png")
    return written

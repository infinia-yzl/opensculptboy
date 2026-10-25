# OpenSculptBoy
# Apache License, Version 2.0
"""
Rendering-budget tiers: one master at the highest quality, and cheaper tiers derived from it.

A character is authored once, as its master: the finest level of the card's detail layer with
the card's own maps (the textures of ``opensculptboy fit --textures`` or ``opensculptboy bake``).
Each cheaper tier only ever cuts down from the master: a coarser mesh, smaller textures, fewer
maps and a cheaper skin. The maps that hold a property of the surface (the base colour, the
roughness, the thickness, the curvature and the ambient occlusion) lay out anny's atlas alike on
every level, so a tier takes them resized; the normal maps hold the master's shading normals in
the tangent frames of a mesh, so each tier with a coarser mesh takes normal maps baked from the
master onto its own mesh (:func:`bake_tiers`, :mod:`opensculptboy.texture.transfer`), which keep
the master's pores and folds on the coarser surface.

========  ===========================  =====================================================
Tier      Mesh                         Skin
========  ===========================  =====================================================
master    level 2                      full subsurface scattering (Cycles), 4096 textures
high      level 2 head, level 1 body   engine subsurface (Unreal profile, HDRP diffusion)
medium    level 1 head, level 0 body   normal and roughness maps, thickness and curvature
low       level 0                      toon (MToon) with the skin ramp and baked occlusion
========  ===========================  =====================================================

:data:`TIERS` holds the presets (:class:`Tier`); :func:`tier_mesh` gives the mesh of a tier for
a card (never finer than the card's master), :func:`tier_textures` the textures of a tier,
:func:`tier_budget` the counts of a tier, and :func:`export_tier` writes a tier as a GLB or VRM
file. ``docs/budget-tiers.md`` describes the tiers, their budgets and the engine notes.
"""

from __future__ import annotations

import dataclasses
import io
import json
import pathlib
import warnings

import numpy as np

# the meshes of the detail layer, from the coarsest to the finest: (subdiv, region)
MESHES = ((0, "full"), (1, "head"), (1, "full"), (2, "head"), (2, "full"))
# the triangles and vertices of each mesh of anny's topology
MESH_COUNTS = {
    (0, "full"): (27_420, 13_718),
    (1, "head"): (55_272, 27_644),
    (1, "full"): (109_680, 54_848),
    (2, "head"): (220_992, 110_504),
    (2, "full"): (438_720, 219_368),
}
# the slots of anny's atlas, the head tile and the ear tile
ATLAS_SLOTS = (
    "base_color",
    "normal",
    "roughness",
    "thickness",
    "curvature",
    "occlusion",
)
HEAD_SLOTS = ("head_base_color", "head_normal", "head_roughness")
EAR_SLOTS = (
    "ears_base_color",
    "ears_normal",
    "ears_roughness",
    "ears_thickness",
    "ears_curvature",
)
SHADINGS = ("path_traced", "subsurface_profile", "preintegrated", "toon")
# the strength of the baked occlusion in a toon tier's shade texture
SHADE_OCCLUSION = 0.8


@dataclasses.dataclass(frozen=True)
class Tier:
    """
    A rendering-budget tier.

    ``subdiv`` and ``region`` name the mesh (see :data:`MESHES`); ``atlas_size`` is the side of
    the atlas textures, ``head_tile_size`` the height of the head tile (0: none, the head on
    the atlas), ``ear_tile_size`` the width of the ear tile (0: none) and ``eye_size`` the width
    of the eye texture. ``maps`` lists the atlas maps the tier keeps (the tiles keep the same
    kinds of maps). ``shading`` names the skin shading the tier is made for: ``path_traced``
    (Cycles random-walk subsurface), ``subsurface_profile`` (an engine's screen-space
    subsurface: Unreal's Subsurface Profile, Unity HDRP's Diffusion Profile),
    ``preintegrated`` (a pre-integrated skin lookup by curvature, with translucency by
    thickness) or ``toon`` (MToon with the skin ramp and baked occlusion). ``morph_targets`` is
    ``"facial"`` (the 52 ARKit facial actions) or ``"expressions"`` (the facial actions that the
    VRM expression presets mix); ``budget`` caps the triangles and the largest texture.
    """

    name: str
    subdiv: int
    region: str
    atlas_size: int
    head_tile_size: int
    ear_tile_size: int
    eye_size: int
    maps: tuple
    shading: str
    morph_targets: str
    max_influences: int
    format: str  # the default format: "glb" or "vrm"
    vrm_version: str
    budget: dict
    summary: str

    @property
    def mesh(self) -> tuple[int, str]:
        return (self.subdiv, self.region)

    def extras(self, mesh: tuple[int, str] | None = None) -> dict:
        """The tier as the exporters record it (``tier`` in the scene extras)."""
        subdiv, region = mesh or self.mesh
        return {
            "name": self.name,
            "subdiv": int(subdiv),
            "region": region,
            "shading": self.shading,
            "atlas_size": int(self.atlas_size),
            "head_tile_size": int(self.head_tile_size),
            "ear_tile_size": int(self.ear_tile_size),
            "maps": list(self.maps),
            "morph_targets": self.morph_targets,
            "max_influences": int(self.max_influences),
        }


TIERS = {
    "master": Tier(
        "master",
        2,
        "full",
        4096,
        4096,
        2048,
        1024,
        ("base_color", "normal", "roughness", "thickness", "curvature"),
        "path_traced",
        "facial",
        8,
        "glb",
        "1.0",
        {"triangles": 440_000, "texture_size": 4096},
        "offline renders: level 2, every map at full size, Cycles random-walk subsurface",
    ),
    "high": Tier(
        "high",
        2,
        "head",
        2048,
        2048,
        1024,
        1024,
        ("base_color", "normal", "roughness", "thickness"),
        "subsurface_profile",
        "facial",
        8,
        "glb",
        "1.0",
        {"triangles": 225_000, "texture_size": 2048},
        "real-time close-ups: the head at level 2, engine subsurface skin",
    ),
    "medium": Tier(
        "medium",
        1,
        "head",
        1024,
        1024,
        0,
        512,
        ("base_color", "normal", "roughness", "thickness", "curvature"),
        "preintegrated",
        "facial",
        4,
        "glb",
        "1.0",
        {"triangles": 60_000, "texture_size": 1024},
        "real-time scenes: the head at level 1, a pre-integrated skin from the baked maps",
    ),
    "low": Tier(
        "low",
        0,
        "full",
        1024,
        0,
        0,
        512,
        ("base_color", "normal", "occlusion"),
        "toon",
        "expressions",
        4,
        "vrm",
        "1.0",
        {"triangles": 32_000, "texture_size": 1024},
        "avatars and crowds: level 0, MToon with the skin ramp and baked occlusion",
    ),
}


def get_tier(tier) -> Tier:
    """A :class:`Tier` from its name (or the tier itself)."""
    if isinstance(tier, Tier):
        return tier
    if tier not in TIERS:
        raise ValueError(f"Unknown tier {tier!r}; the tiers are {list(TIERS)}.")
    return TIERS[tier]


def master_mesh(character) -> tuple[int, str]:
    """The master's mesh: the finest full level that the card's detail layer holds."""
    return (max([0, *character.detail_levels()]), "full")


def tier_mesh(character, tier) -> tuple[int, str]:
    """The mesh (subdiv, region) of a tier for a card: the tier's own mesh, or the master's
    when that is coarser (a tier only ever cuts down from the master)."""
    tier = get_tier(tier)
    return min(tier.mesh, master_mesh(character), key=MESHES.index)


def _head_level(mesh) -> int:
    return mesh[0]


def _body_level(mesh) -> int:
    return mesh[0] - (1 if mesh[1] == "head" else 0)


def needs_bake(character, tier) -> dict[str, bool]:
    """Which normal maps of a tier differ from the master's because its mesh does: the
    atlas's (``normal``, when the tier's body or head is coarser) and the tiles' (``head_normal``,
    ``ears_normal``, when its head is coarser and it keeps that tile)."""
    tier = get_tier(tier)
    mesh, master = tier_mesh(character, tier), master_mesh(character)
    coarser_head = _head_level(mesh) < _head_level(master)
    return {
        "normal": mesh != master,
        "head_normal": bool(tier.head_tile_size) and coarser_head,
        "ears_normal": bool(tier.ear_tile_size) and mesh[0] > 0 and coarser_head,
    }


def wearable_counts(
    character, detail_dir=None, dressing=None, bare: bool = False
) -> dict:
    """
    The pieces, triangles, vertices and materials of the wearables that a file of the card
    holds, and the size of each of their textures (by ``<ref>:<material>:<slot>``): the pieces
    that ``dressing`` (:class:`~opensculptboy.wearables.export.Dressing`) shows, without the
    groups that other pieces hide, or every piece of a wardrobe file. A card with wardrobe
    fields wears its active outfit when ``dressing`` is None, as the exports do; ``bare``
    (a VRM file without wearables) and a card without wardrobe fields hold none.
    """
    from opensculptboy.export.budget import image_size

    out = {"pieces": 0, "triangles": 0, "vertices": 0, "materials": 0, "textures": {}}
    if bare or (dressing is None and not character.has_wardrobe()):
        return out
    from opensculptboy.wearables.export import Dressing
    from opensculptboy.wearables.wardrobe import worn_pieces

    dressing = dressing or Dressing()
    pieces = worn_pieces(
        character,
        detail_dir,
        dressing.outfit,
        dressing.show,
        dressing.hide,
        every_outfit=dressing.wardrobe,
    )
    for piece in pieces:
        if not (piece.shown or dressing.wardrobe):
            continue
        garment = piece.wearable.garment
        keep = np.ones(len(garment.triangles), bool)
        if not dressing.wardrobe:
            for g, group in enumerate(garment.groups):
                if group in piece.hidden_groups:
                    keep &= garment.triangle_group != g
        if not keep.any():  # other pieces hide every group of this one
            continue
        out["pieces"] += 1
        out["triangles"] += int(keep.sum())
        out["vertices"] += len(garment.vertices)
        out["materials"] += len(garment.materials)
        for material in garment.materials:
            for slot, data in material.textures.items():
                size = image_size(data)
                if size is not None:
                    out["textures"][f"{piece.ref}:{material.name}:{slot}"] = list(size)
    return out


def tier_budget(
    character,
    tier,
    images: dict | None = None,
    detail_dir=None,
    dressing=None,
    bare: bool = False,
) -> dict:
    """The counts of a tier for a card: triangles and vertices of its mesh and of the
    wearables that the file holds (:func:`wearable_counts`, also under ``wearables``), and the
    size of each texture (``images``: slot -> PNG or JPEG bytes, as :func:`tier_textures` gives
    them, and the wearables' textures) with their texels and the bytes they take on a GPU
    (RGBA8 with mipmaps, uncompressed)."""
    from opensculptboy.export.budget import image_size

    tier = get_tier(tier)
    mesh = tier_mesh(character, tier)
    triangles, vertices = MESH_COUNTS[mesh]
    wear = wearable_counts(character, detail_dir, dressing, bare)
    textures = {}
    for slot, data in (images or {}).items():
        size = image_size(data)
        if size is not None:
            textures[slot] = list(size)
    textures.update(wear["textures"])
    texels = sum(w * h for w, h in textures.values())
    return {
        "tier": tier.name,
        "mesh": {"subdiv": mesh[0], "region": mesh[1]},
        "triangles": triangles + wear["triangles"],
        "vertices": vertices + wear["vertices"],
        "wearables": {k: v for k, v in wear.items() if k != "textures"},
        "textures": textures,
        "texels": int(texels),
        "gpu_bytes": int(texels * 4 * 4 / 3),
        "max_image_size": max([0, *(max(s) for s in textures.values())]),
    }


def check_budget(found: dict, tier) -> list[str]:
    """Messages for each count of a file (``budget.counts``) over the tier's budget."""
    tier = get_tier(tier)
    messages = []
    for key, counted in (
        ("triangles", "triangles"),
        ("texture_size", "max_image_size"),
    ):
        value, limit = int(found.get(counted, 0)), tier.budget[key]
        if value > limit:
            messages.append(
                f"{key}: {value:,} over the {tier.name} budget of {limit:,}"
            )
    return messages


def _resize(data: bytes, size: tuple[int, int], normal: bool = False) -> bytes:
    """A PNG file of an image shrunk to ``size`` (width, height), or the file itself when it
    is not larger; a normal map is averaged as vectors and normalised again."""
    from PIL import Image

    from opensculptboy.export.document import png_bytes

    image = Image.open(io.BytesIO(data))
    if image.size[0] <= size[0] and image.size[1] <= size[1]:
        return data
    if not normal:
        mode = "L" if image.mode in ("L", "I", "I;16", "F") else "RGB"
        out = image.convert(mode).resize(size, Image.Resampling.BOX)
        return png_bytes(np.asarray(out))
    pixels = np.asarray(image.convert("RGB"), np.float64) / 127.5 - 1.0
    channels = [
        np.asarray(
            Image.fromarray(pixels[..., k].astype(np.float32), "F").resize(
                size, Image.Resampling.BOX
            )
        )
        for k in range(3)
    ]
    n = np.stack(channels, -1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    return png_bytes(np.clip(np.round((n + 1.0) * 127.5), 0, 255).astype(np.uint8))


def _shade(base: bytes, occlusion: bytes, strength: float = SHADE_OCCLUSION) -> bytes:
    """The shade texture of a toon tier: the base colour times the occlusion (at the base
    colour's size), mixed in by ``strength``."""
    from PIL import Image

    from opensculptboy.export.document import png_bytes

    colour = Image.open(io.BytesIO(base)).convert("RGB")
    ao = (
        Image.open(io.BytesIO(occlusion))
        .convert("L")
        .resize(colour.size, Image.Resampling.BILINEAR)
    )
    factor = 1.0 - strength * (1.0 - np.asarray(ao, np.float64) / 255.0)
    shade = np.asarray(colour, np.float64) * factor[..., None]
    return png_bytes(np.clip(np.round(shade), 0, 255).astype(np.uint8))


def _read(path, base) -> bytes:
    path = pathlib.Path(path)
    if not path.is_absolute() and base is not None:
        path = pathlib.Path(base) / path
    return path.read_bytes()


def tier_textures(
    character, tier, detail_dir=None, format: str | None = None
) -> dict[str, bytes]:
    """
    The textures of a tier by slot (:data:`opensculptboy.export.detail.TEXTURE_SLOTS`), as PNG
    or JPEG bytes: the card's textures (``export.detail.texture_images``) that the tier keeps,
    with the normal maps that :func:`bake_tiers` baked for the tier's mesh in place of the
    master's, shrunk to the tier's sizes. A VRM file of a tier with an occlusion map gets the
    ``shade`` texture (the base colour darkened by the occlusion). A normal map baked for
    another mesh than the tier's own is used with a warning.
    """
    from opensculptboy.export.detail import texture_images

    tier = get_tier(tier)
    format = format or tier.format
    images = texture_images(character, None, detail_dir)
    mesh = tier_mesh(character, tier)
    entry = (character.material or {}).get("tiers", {}).get(tier.name, {})
    baked = entry and (entry.get("subdiv"), entry.get("region", "full")) == mesh
    if entry and not baked:
        warnings.warn(
            f"the {tier.name} tier's normal maps were baked for level {entry.get('subdiv')} "
            f"({entry.get('region', 'full')}), not for its mesh (level {mesh[0]}, "
            f"{mesh[1]}); run opensculptboy bake-tiers again",
            UserWarning,
            stacklevel=2,
        )
    for slot, wanted in needs_bake(character, tier).items():
        if baked and slot in entry:
            images[slot] = _read(entry[slot], detail_dir)
        elif wanted and slot in images:
            warnings.warn(
                f"the {tier.name} tier takes the master's {slot} map, baked for another mesh; "
                "opensculptboy bake-tiers bakes one for the tier's own mesh",
                UserWarning,
                stacklevel=2,
            )
    keep = {}
    for slot, data in images.items():
        if slot in ATLAS_SLOTS:
            if slot not in tier.maps:
                continue
            size = (tier.atlas_size, tier.atlas_size)
        elif slot in HEAD_SLOTS:
            if not tier.head_tile_size or slot[5:] not in tier.maps:
                continue
            size = (tier.head_tile_size // 2, tier.head_tile_size)
        elif slot in EAR_SLOTS:
            if not tier.ear_tile_size or mesh[0] == 0 or slot[5:] not in tier.maps:
                continue
            size = (tier.ear_tile_size, tier.ear_tile_size // 2)
        elif slot in ("eyes", "eyes_normal"):
            size = (tier.eye_size, tier.eye_size // 2)
        else:
            continue
        keep[slot] = _resize(data, size, normal=slot.endswith("normal"))
    if format == "vrm" and "base_color" in keep and "occlusion" in keep:
        keep["shade"] = _shade(keep["base_color"], keep["occlusion"])
    return keep


def tier_morph_targets(model, tier) -> list[str]:
    """The morph targets of a tier's GLB file: the 52 facial actions, or the facial actions
    that the VRM expression presets mix (in the model's order)."""
    tier = get_tier(tier)
    labels = list(model.facial_action_labels)
    if tier.morph_targets == "facial":
        return labels
    from opensculptboy.export.vrm_tables import _expression_table

    used = set()
    for entry in _expression_table()["presets"].values():
        used |= set(entry["mix"])
    return [label for label in labels if label in used]


def export_tier(
    path,
    character=None,
    tier="low",
    detail_dir=None,
    model=None,
    format: str | None = None,
    **kwargs,
) -> dict:
    """
    Write a tier of a character as a GLB or VRM file.

    The suffix of ``path`` picks the format (``.vrm``, else glTF); a path without one takes
    the tier's default (:attr:`Tier.format`) and its suffix. The tier sets the detail level and
    region (:func:`tier_mesh`), the textures (:func:`tier_textures`), the morph targets and
    the skin weights per vertex of a GLB file, and records itself in the file's extras;
    ``kwargs`` pass on to :func:`~opensculptboy.export.gltf.export_glb` or
    :func:`~opensculptboy.export.vrm.export_vrm` and override the tier's choices
    (``morph_targets``, ``max_influences``, ``version``...). A tier whose mesh and wearable
    pieces (:func:`wearable_counts`) are over the triangle budget of a VRM version cannot be
    written as VRM (ValueError). The summary adds
    the tier's counts (:func:`tier_budget`) and the messages of a file over the tier's
    budget, which also warn.
    """
    from opensculptboy.character import Character
    from opensculptboy.export import budget as budgets

    tier = get_tier(tier)
    character = character or Character()
    path = pathlib.Path(path)
    if format is None:
        format = (
            "vrm"
            if path.suffix.lower() == ".vrm"
            else ("glb" if path.suffix else tier.format)
        )
    if not path.suffix:
        path = path.with_suffix("." + format)
    mesh = tier_mesh(character, tier)
    dressing, bare = kwargs.get("dressing"), bool(kwargs.get("bare"))
    if format == "vrm":
        version = kwargs.pop("version", tier.vrm_version)
        target = "vrm1" if version in ("1.0", "1") else "vrm0"
        limit = budgets.BUDGETS[target]
        if kwargs.get("budget", "warn") != "off":
            wear = wearable_counts(character, detail_dir, dressing, bare)
            triangles = MESH_COUNTS[mesh][0] + wear["triangles"]
            worn = f" with {wear['pieces']} wearable pieces" if wear["pieces"] else ""
            if triangles > limit["triangles"]:
                raise ValueError(
                    f"the {tier.name} tier{worn} has {triangles:,} triangles, over the "
                    f"{budgets.LABELS[target]} budget of {limit['triangles']:,}: write a "
                    ".glb file, wear fewer pieces, or use the "
                    + ("medium or low tier." if target == "vrm1" else "low tier.")
                )
    images = tier_textures(character, tier, detail_dir, format)
    extras = tier.extras(mesh)
    if format == "vrm":
        from opensculptboy.export.vrm import export_vrm

        summary = export_vrm(
            path,
            character,
            model,
            version="1.0" if version in ("1.0", "1") else "0.x",
            detail_level=mesh[0],
            detail_region=mesh[1],
            detail_dir=detail_dir,
            textures=images,
            tier=extras,
            **kwargs,
        )
    else:
        from opensculptboy.export.gltf import export_glb

        if model is None:
            model = character.build_model(face_shapes=bool(character.face_shapes))
        kwargs.setdefault("morph_targets", tier_morph_targets(model, tier))
        kwargs.setdefault("max_influences", tier.max_influences)
        summary = export_glb(
            path,
            character,
            model,
            detail_level=mesh[0],
            detail_region=mesh[1],
            detail_dir=detail_dir,
            textures=images,
            tier=extras,
            **kwargs,
        )
    found = file_counts(path)
    messages = check_budget(found, tier)
    for message in messages:
        warnings.warn(f"{path.name}: {message}", UserWarning, stacklevel=2)
    summary["path"] = str(path)
    summary["tier_budget"] = tier_budget(
        character, tier, images, detail_dir, dressing, bare and format == "vrm"
    )
    summary["tier_messages"] = messages
    return summary


def file_counts(path) -> dict:
    """The counts of a GLB or VRM file (:func:`opensculptboy.export.budget.counts`)."""
    import struct

    from opensculptboy.export import budget as budgets

    data = pathlib.Path(path).read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20 : 20 + length])
    binary = data[20 + length + 8 :]
    return budgets.counts(gltf, binary)


def missing_maps(character, tier, detail_dir=None) -> list[str]:
    """The atlas maps of a tier that the card does not hold (the base colour and the
    roughness come from a texture bake of the reference; :func:`bake_tiers` adds the
    thickness, the curvature and the occlusion)."""
    tier = get_tier(tier)
    from opensculptboy.export.detail import texture_images

    held = set(texture_images(character, None, detail_dir))
    return [slot for slot in tier.maps if slot not in held]


def _stem(card: pathlib.Path) -> str:
    return card.name[: -len(card.suffix)] if card.suffix else card.name


def bake_tiers(
    card_path,
    tiers=("high", "medium", "low"),
    out=None,
    rays: int = 32,
    occlusion_distance: float = 0.05,
    max_distance: float = 0.006,
    supersample: int = 1,
    progress=None,
):
    """
    Bake the maps of the cheaper tiers of a card from its master, write them next to the card
    (``out``, the card itself by default, in the same folder) and record them in its skin
    material.

    For each tier whose mesh is coarser than the master's (:func:`needs_bake`), the master's
    shading normals (its geometry, bent by its own normal maps) are baked onto the tier's mesh
    as tangent-space normal maps at the tier's sizes (``<card>.<tier>.normal.png``, and the
    head and ear tiles' when the tier keeps them), recorded under ``material.tiers.<tier>``
    with the mesh they belong to. A tier that keeps the occlusion, the thickness or the
    curvature and finds the card without it gets the map baked from the master's geometry
    (``<card>.occlusion.png`` and so on, under ``material``): those maps hold the surface, so
    one serves every tier. Returns the updated :class:`~opensculptboy.character.Character`
    and a report per tier.
    """
    import time

    import torch

    from opensculptboy.character import Character
    from opensculptboy.detail import detail_topology
    from opensculptboy.export.detail import resolve, texture_images
    from opensculptboy.export.document import png_bytes
    from opensculptboy.texture import transfer

    say = progress or (lambda message: None)
    card = pathlib.Path(card_path)
    out = pathlib.Path(out) if out else card
    folder = card.parent
    if out.parent.resolve() != folder.resolve():
        raise ValueError(
            "the card that bake-tiers writes must lie in the folder of the card it reads, "
            "next to the files that both name."
        )
    character = Character.from_dict(json.loads(card.read_text(encoding="utf-8")))
    tiers = [get_tier(t) for t in tiers]
    model = character.build_model(dtype=torch.float64)
    with torch.no_grad():
        base = model(**character.model_kwargs())["rest_vertices"][0].numpy()

    def mesh_of(mesh):
        found = resolve(model, character, mesh[0], folder, mesh[1])
        if found is None:
            return detail_topology(model, 0, "full", character.topology), base
        return found.topology, found.vertices(base)

    master = master_mesh(character)
    master_topology, master_vertices = mesh_of(master)
    images = texture_images(character, None, folder)
    ear_table = None
    if (character.material or {}).get("ears"):
        from opensculptboy.texture.ears import card_ear_uvs

        ear_table = card_ear_uvs(character, folder)
    reference = None
    material = dict(character.material or {})
    entries = dict(material.get("tiers", {}))
    stem = _stem(out)
    report = {}

    def write(name: str, pixels) -> str:
        path = out.parent / f"{stem}.{name}.png"
        path.write_bytes(png_bytes(pixels))
        return path.name

    for tier in tiers:
        mesh = tier_mesh(character, tier)
        wanted = needs_bake(character, tier)
        if not any(wanted.values()):
            report[tier.name] = {"mesh": list(mesh), "baked": []}
            continue
        start = time.time()
        if reference is None:
            say(f"master: level {master[0]}, {len(master_vertices):,} vertices")
            reference = transfer.master_reference(
                master_vertices,
                master_topology,
                {k: v for k, v in images.items() if k.endswith("normal")},
                model,
                character.topology,
                ear_table,
            )
        topology, vertices = mesh_of(mesh)
        maps = transfer.bake_normals(
            reference,
            vertices,
            topology,
            tier.atlas_size,
            model,
            character.topology,
            tier.head_tile_size if wanted["head_normal"] else 0,
            ear_table if wanted["ears_normal"] else None,
            max_distance,
            supersample,
        )
        entry = {"subdiv": int(mesh[0]), "region": mesh[1]}
        for slot in ("normal", "head_normal", "ears_normal"):
            if wanted[slot] and slot in maps:
                entry[slot] = write(f"{tier.name}.{slot}", maps[slot])
        entries[tier.name] = entry
        report[tier.name] = {
            "mesh": list(mesh),
            "baked": sorted(k for k in entry if k.endswith("normal")),
            "covered": round(float(maps["covered"]), 4),
            "seconds": round(time.time() - start, 1),
        }
        say(f"{tier.name}: {report[tier.name]}")

    # the maps of the surface, from the master's geometry, where a tier keeps them
    held = set(images)
    sizes = {}
    for tier in tiers:
        for slot in ("occlusion", "thickness", "curvature"):
            if slot in tier.maps and slot not in held:
                sizes[slot] = max(sizes.get(slot, 0), tier.atlas_size)
    if sizes:
        from opensculptboy.texture.bake import (
            eyeball_triangles,
            vertex_curvature,
            vertex_thickness,
        )

        V, T = master_vertices, np.asarray(master_topology.triangles)
        for slot, size in sorted(sizes.items()):
            start = time.time()
            if slot == "occlusion":
                values = transfer.vertex_occlusion(V, T, rays, occlusion_distance)
            elif slot == "thickness":
                mm = float(material.get("thickness_mm", 30.0))
                eyes = eyeball_triangles(model, master_topology)
                values = vertex_thickness(V, T, mm / 1000.0, eyes) * 1000.0 / mm
                material["thickness_mm"] = mm
            else:
                per_m = float(material.get("curvature_per_m", 100.0))
                curvature = vertex_curvature(V, T)
                values = 0.5 + 0.5 * np.clip(curvature / per_m, -1.0, 1.0)
                material["curvature_per_m"] = per_m
            image = transfer.atlas_map(values, master_topology, size)
            material[slot] = write(slot, transfer.grey(image))
            report.setdefault("maps", {})[slot] = {
                "size": int(size),
                "seconds": round(time.time() - start, 1),
            }
            say(f"{slot}: {report['maps'][slot]}")
    if entries:
        material["tiers"] = entries
    character.material = material
    out.write_text(json.dumps(character.to_dict(), indent=2) + "\n", encoding="utf-8")
    return character, report

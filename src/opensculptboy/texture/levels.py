# OpenSculptBoy
# Apache License, Version 2.0
"""
Normal maps baked on one level of the detail layer, re-encoded for another level.

A tangent-space normal map holds each normal in the MikkTSpace frame of the mesh it was baked
on (:mod:`opensculptboy.texture.bake`): its tangent, bitangent and interpolated normal at the
texel. The fit bakes the card's maps on its finest level (``detail.textures.subdiv``), and the
coarser levels share the atlas, the head tile and the ear tile, but their surfaces turn
differently, most of all at a pointed ear's tip, where level 1 folds by tens of degrees
between triangles that level 2 rounds. Read in a coarser level's frames, the finest level's
map turns its normals by those differences: a crease along the folds of the coarse mesh.

:func:`level_normals` re-encodes the maps for the exported mesh: each texel of the map that a
triangle of the exported mesh covers is decoded in the frame of the baked level at the same UV
(the world normal that the finest level shows there) and encoded in the frame of the exported
mesh at that UV, so that both levels shade alike at rest. Texels outside the triangles of both
meshes keep their values. Both frames are those of the rest meshes, as the exporters write
them.
"""

from __future__ import annotations

import io

import numpy as np

# the normal maps by slot, and the texture whose UVs they use: anny's atlas, the head tile
# (opensculptboy.texture.tiles) and the ear tile (opensculptboy.texture.ears)
NORMAL_SLOTS = ("normal", "head_normal", "ears_normal")


def baked_level(character) -> int | None:
    """The level of the detail layer that a card's textures were baked on, or None."""
    level = (character.detail or {}).get("textures", {}).get("subdiv")
    return None if level is None else int(level)


def _slot_triangles(model, topology: str, mesh, ears, slot: str):
    """The triangles (mask) that sample a slot's texture and their UVs (OBJ convention) in
    that texture, or (None, None) when the mesh has none."""
    from opensculptboy.texture.tiles import head_uv_box, on_head_tile, to_tile_uvs

    uv_triangles, uvs = np.asarray(mesh.uv_triangles), np.asarray(mesh.uvs)
    on_ears = np.zeros(len(uv_triangles), bool) if ears is None else np.asarray(ears)
    if slot == "ears_normal":
        return on_ears, uvs
    if slot == "normal":
        return ~on_ears, uvs
    head = on_head_tile(model, uvs, uv_triangles, topology)
    box = head_uv_box(model, topology)
    if head is None or box is None:
        return None, None
    return head & ~on_ears, to_tile_uvs(uvs, box)


def _frames(vertices, mesh, uvs, pick, size, chunk=1 << 20):
    """The texels (rows, cols) that the picked triangles cover on a texture of ``size``
    (width, height), and the MikkTSpace frames (t, b, n) of the mesh there."""
    from opensculptboy.texture.bake import _Surface, rasterise

    triangles = np.asarray(mesh.triangles)
    uv_triangles = np.asarray(mesh.uv_triangles)
    surface = _Surface.of(
        np.asarray(vertices, np.float64), triangles, uv_triangles, np.asarray(mesh.uvs)
    )
    index = np.nonzero(pick)[0]
    rows, cols, tri, bary = rasterise(uv_triangles[index], uvs, size)
    tri = index[tri]
    frames = np.zeros((len(rows), 3, 3), np.float32)
    for start in range(0, len(rows), chunk):
        sel = slice(start, start + chunk)
        _, n, t, b = surface.at(tri[sel], bary[sel])
        frames[sel] = np.stack([t, b, n], 1)
    return rows, cols, frames


def reencode(image: np.ndarray, source, target) -> np.ndarray:
    """
    A normal map (H, W, 3) uint8 baked in the frames of ``source`` and re-encoded in the
    frames of ``target``; each is (rows, cols, frames (n, 3, 3) of t, b and n) as
    :func:`_frames` gives them.
    """
    h, w = image.shape[:2]
    rs, cs, fs = source
    rt, ct, ft = target
    world = np.zeros((h, w, 3), np.float32)
    known = np.zeros((h, w), bool)
    local = image[rs, cs].astype(np.float32) / 127.5 - 1.0
    world[rs, cs] = np.einsum("nk,nkd->nd", local, fs)
    known[rs, cs] = True
    take = known[rt, ct]
    rt, ct, ft = rt[take], ct[take], ft[take]
    n = world[rt, ct]
    out = np.einsum("nkd,nd->nk", ft, n)
    out[:, 2] = np.maximum(out[:, 2], 0.0)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    result = image.copy()
    result[rt, ct] = np.clip(np.round((out * 0.5 + 0.5) * 255.0), 0, 255).astype(
        np.uint8
    )
    return result


def level_normals(
    model,
    character,
    images: dict,
    detail,
    mesh,
    ears,
    base: np.ndarray,
    detail_dir=None,
) -> dict:
    """
    ``images`` with the normal maps re-encoded for the exported mesh ``mesh`` (the detail
    mesh ``detail`` with the ear tile's UVs, the mask ``ears`` of its ear triangles) when the
    card's textures were baked on another level or region; else ``images`` unchanged.
    ``base`` is Anny's rest mesh of the character.
    """
    from PIL import Image

    from opensculptboy.export.detail import ear_mesh, resolve
    from opensculptboy.export.document import png_bytes

    level = baked_level(character)
    slots = [s for s in NORMAL_SLOTS if s in images]
    if detail is None or level is None or not slots or mesh.uv_triangles is None:
        return images
    if detail.subdiv == level and detail.region == "full":
        return images
    try:
        source_detail = resolve(model, character, level, detail_dir, "full")
    except ValueError:  # the card no longer holds the level it was baked on
        return images
    if source_detail is None:
        return images
    skin = character.skin_material()
    source_mesh, source_ears = ear_mesh(
        character, skin, images, source_detail, detail_dir
    )
    source_vertices = source_detail.vertices(base)
    target_vertices = detail.vertices(base)
    topology = character.topology
    out = dict(images)
    for slot in slots:
        image = np.asarray(Image.open(io.BytesIO(images[slot])).convert("RGB"))
        size = (image.shape[1], image.shape[0])
        pick_s, uvs_s = _slot_triangles(model, topology, source_mesh, source_ears, slot)
        pick_t, uvs_t = _slot_triangles(model, topology, mesh, ears, slot)
        if pick_s is None or pick_t is None or not pick_s.any() or not pick_t.any():
            continue
        source = _frames(source_vertices, source_mesh, uvs_s, pick_s, size)
        target = _frames(target_vertices, mesh, uvs_t, pick_t, size)
        out[slot] = png_bytes(reencode(image, source, target))
    return out

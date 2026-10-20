# OpenSculptBoy
# Apache License, Version 2.0
"""
The head tile: the head's UV island of anny's atlas, given textures of their own.

On a 4096-texel atlas, anny's head holds about 4.6 texels per millimetre of skin, as much as a
game character's 2048-texel head texture, so a bake of such a reference resamples its detail
at about the rate it has and blurs it. The head tile gives the head (the face, the ears, the
scalp and the upper neck: the UV island of anny's atlas that lies on the head) a texture of its
own: the bounding box of the island, padded and widened to the tile's aspect of
:data:`HEAD_TILE_ASPECT` (width : height), so that a 2048 by 4096 tile holds about 6.4
texels per millimetre. As with the eye texture (:mod:`opensculptboy.texture.eyes`), the
exporters draw the island as a primitive of its own whose material's textures carry the box as
a ``KHR_texture_transform``, so that the mesh keeps anny's UVs and needs no second UV set. The
island ends at UV seams, so the border between the two materials follows a seam of the atlas.

The triangles of a mesh that lie on the island are found by their UV centroids
(:func:`on_head_tile`): the islands of an atlas do not overlap, and a subdivided triangle lies
inside a face of the base mesh, so the test holds on every level of the detail layer.
"""

from __future__ import annotations

import numpy as np

# width : height of the head tile; the head's island is about half as wide as it is tall
HEAD_TILE_ASPECT = 0.5
# padding of the box on each side, as a share of its larger side
HEAD_BOX_PADDING = 0.01
# resolution of the island lookup of on_head_tile (texels along each side of the atlas)
_LOOKUP_SIZE = 2048


def uv_islands(uv_faces: np.ndarray, uv_count: int) -> np.ndarray:
    """The island (connected component through shared UV vertices) of each UV face."""
    label = np.arange(uv_count)
    faces = np.asarray(uv_faces, dtype=np.int64)
    while True:  # the smallest UV vertex of each island spreads through its faces
        low = label[faces].min(1)
        new = label.copy()
        np.minimum.at(new, faces.ravel(), np.repeat(low, faces.shape[1]))
        new = new[new]
        if np.array_equal(new, label):
            break
        label = new
    _, island = np.unique(label[faces[:, 0]], return_inverse=True)
    return island.ravel()


_ISLANDS: dict = {}


def _head_island(model, topology: str = "anny"):
    """(UV faces, UVs, the mask of the head island's faces) of a topology's quad mesh, or
    None; the head is the faces weighted to the head and the neck bones of ``model``."""
    if topology in _ISLANDS:
        return _ISLANDS[topology]
    from anny.utils.subdivision import HEAD_REGION_BONES, faces_weighted_to_bones
    from opensculptboy.detail.subdivision import level

    try:
        base = level(topology, 0)
    except (ValueError, KeyError, FileNotFoundError):
        return None
    if base.uv_quads is None or base.vertex_count != len(model.template_vertices):
        return None
    head = faces_weighted_to_bones(model, base.quads, HEAD_REGION_BONES)
    island = uv_islands(base.uv_quads, len(base.uvs))
    # the largest island that lies wholly on the head
    counts = np.bincount(island)
    on_head = np.bincount(island, weights=head.astype(float))
    whole = np.nonzero(on_head == counts)[0]
    if len(whole) == 0:
        return None
    pick = whole[np.argmax(counts[whole])]
    found = base.uv_quads, np.asarray(base.uvs, np.float64), island == pick
    _ISLANDS[topology] = found
    return found


def head_uv_box(model, topology: str = "anny"):
    """
    The box (lower and upper corners, OBJ convention: v = 0 at the bottom) of anny's atlas
    that the head tile covers, or None for a mesh without a head island.
    """
    found = _head_island(model, topology)
    if found is None:
        return None
    uv_faces, uvs, mask = found
    points = uvs[np.unique(uv_faces[mask])]
    lo, hi = points.min(0), points.max(0)
    pad = HEAD_BOX_PADDING * float((hi - lo).max())
    lo, hi = lo - pad, hi + pad
    centre, size = 0.5 * (lo + hi), hi - lo
    # widen the short side to the tile's aspect, so that its texels are square
    if size[0] / size[1] < HEAD_TILE_ASPECT:
        size = np.array([HEAD_TILE_ASPECT * size[1], size[1]])
    else:
        size = np.array([size[0], size[0] / HEAD_TILE_ASPECT])
    return centre - 0.5 * size, centre + 0.5 * size


def tile_transform(box) -> dict:
    """The ``KHR_texture_transform`` (glTF convention: v = 0 at the top) that takes anny's
    glTF UVs of a box of the atlas (OBJ convention) to a tile's [0, 1] square."""
    (u0, v0), (u1, v1) = box
    su, sv = 1.0 / (u1 - u0), 1.0 / (v1 - v0)
    return {
        "offset": [float(-u0 * su), float(-(1.0 - v1) * sv)],
        "scale": [float(su), float(sv)],
    }


def head_texture_transform(model, topology: str = "anny") -> dict | None:
    """The ``KHR_texture_transform`` of the head tile, or None without a head island."""
    box = head_uv_box(model, topology)
    return None if box is None else tile_transform(box)


def to_tile_uvs(uvs: np.ndarray, box) -> np.ndarray:
    """UVs of anny's atlas (OBJ convention) in a tile's square (OBJ convention)."""
    lo, hi = (np.asarray(b, np.float64) for b in box)
    return (np.asarray(uvs, np.float64) - lo) / (hi - lo)


_LOOKUPS: dict = {}


def _lookup(model, topology: str):
    """An image of the atlas: 1 on the head island, 0 on the other islands, and the empty
    texels the value of a neighbour within a few texels (-1 further away)."""
    if topology in _LOOKUPS:
        return _LOOKUPS[topology]
    from opensculptboy.texture.bake import rasterise

    found = _head_island(model, topology)
    if found is None:
        return None
    uv_faces, uvs, mask = found
    triangles = np.concatenate([uv_faces[:, [0, 1, 2]], uv_faces[:, [0, 2, 3]]])
    inside = np.concatenate([mask, mask])
    rows, cols, tri, _ = rasterise(triangles, uvs, _LOOKUP_SIZE)
    image = np.full((_LOOKUP_SIZE, _LOOKUP_SIZE), -1, np.int8)
    image[rows, cols] = inside[tri].astype(np.int8)
    for _ in range(
        4
    ):  # the texels along the islands' borders whose centres no face holds
        empty = image < 0
        if not empty.any():
            break
        grown = image.copy()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            shifted = np.roll(image, (dy, dx), axis=(0, 1))
            take = (grown < 0) & (shifted >= 0)
            grown[take] = shifted[take]
        image = grown
    _LOOKUPS[topology] = image
    return image


def on_head_tile(
    model, uvs: np.ndarray, uv_triangles: np.ndarray, topology: str = "anny"
):
    """
    The mask (T,) of the triangles whose UV centroid lies on the head's island, for UVs in
    the OBJ convention (v = 0 at the bottom), or None without a head island.
    """
    image = _lookup(model, topology)
    if image is None:
        return None
    centre = np.asarray(uvs, np.float64)[np.asarray(uv_triangles)].mean(1)
    size = image.shape[0]
    col = np.clip((centre[:, 0] * size).astype(np.int64), 0, size - 1)
    row = np.clip(((1.0 - centre[:, 1]) * size).astype(np.int64), 0, size - 1)
    return image[row, col] == 1

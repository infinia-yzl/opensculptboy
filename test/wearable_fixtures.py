# OpenSculptBoy
# Apache License, Version 2.0
"""
Synthetic garments for the wearable tests: a tunic (an offset shell of anny's torso), a rigid
circlet around the head and a cape panel behind the back, all made in the test from anny's
template, and a synthetic reference character with a clothing part.
"""

from __future__ import annotations

import functools

import numpy as np

from opensculptboy import Character
from opensculptboy.wearables import (
    Garment,
    Material,
    Wearable,
    bind_rigid,
    bind_surface,
    card_body,
)

LICENCE = {
    "spdx": "CC0-1.0",
    "author": "OpenSculptBoy tests",
    "source": "test/wearable_fixtures.py",
    "design_source": "synthetic test garment, made in the test",
}


@functools.lru_cache(maxsize=1)
def template_model():
    import torch

    return Character().build_model(dtype=torch.float64)


@functools.lru_cache(maxsize=1)
def template_body():
    return card_body(Character(), template_model())


def _region_shell(body, keep_vertex: np.ndarray, offset: float):
    """The triangles of the body whose corners all lie in ``keep_vertex``, moved out along
    the vertex normals by ``offset``: (vertices, triangles, body rows)."""
    tris = body.triangles[keep_vertex[body.triangles].all(1)]
    rows = np.unique(tris)
    remap = np.full(len(body.vertices), -1)
    remap[rows] = np.arange(len(rows))
    vertices = body.vertices[rows] + offset * body.normals[rows]
    return vertices, remap[tris], rows


def torso_mask(body) -> np.ndarray:
    v = body.vertices
    low, h = float(v[:, 2].min()), float(np.ptp(v[:, 2]))
    hips, chest = low + 0.52 * h, low + 0.72 * h
    return (v[:, 2] > hips) & (v[:, 2] < chest) & (np.abs(v[:, 0]) < 0.17)


def tunic(body=None, offset: float = 0.006) -> Wearable:
    """A tunic: anny's torso between the hips and the chest, 6 mm off the skin, with two
    groups (the front ``front`` and the back ``back``), each with the skin under it as its
    cover mask (``group_covers``), so that a hidden group shows its skin again."""
    body = body or template_body()
    keep = torso_mask(body)
    vertices, triangles, rows = _region_shell(body, keep, offset)
    centre = vertices[triangles].mean(axis=1)
    group = (centre[:, 1] > 0).astype(np.int64)  # anny faces -Y: y > 0 is the back
    uvs = np.stack([vertices[:, 0] + 0.5, 1.0 - vertices[:, 2] / 2.0], axis=1)
    garment = Garment(
        vertices,
        triangles,
        uvs=uvs,
        triangle_group=group,
        groups=["front", "back"],
        materials=[Material("tunic", (0.2, 0.3, 0.6, 1.0), 0.0, 0.9)],
    )
    covers = np.zeros(len(body.vertices), bool)
    covers[rows] = True
    # the two rings along the border of the shell are not under the garment
    for _ in range(2):
        border = np.zeros(len(body.vertices), bool)
        outside = ~covers[body.triangles]
        border[body.triangles[outside.any(1)].ravel()] = True
        covers &= ~border
    return Wearable(
        name="tunic",
        slot="torso",
        layer=50,
        garment=garment,
        binding=bind_surface(garment, body),
        licence=dict(LICENCE),
        covers=np.zeros(len(body.vertices), bool),
        group_covers={
            "front": covers & (body.vertices[:, 1] <= 0),
            "back": covers & (body.vertices[:, 1] > 0),
        },
    )


def cover_of(wearable) -> np.ndarray:
    """The skin that all the groups of a piece cover."""
    mask = np.asarray(wearable.covers, bool).copy()
    for extra in wearable.group_covers.values():
        mask |= extra
    return mask


def circlet(body=None) -> Wearable:
    """A rigid circlet: a torus of 64 by 8 vertices around the head at the brow."""
    body = body or template_body()
    labels = body.bone_labels
    head = body.bone_poses[labels.index("head"), :3, 3]
    v = body.vertices
    near = np.linalg.norm(v[:, :2] - head[:2], axis=1) < 0.12
    top = float(v[near, 2].max())
    z = top - 0.06
    ring = near & (np.abs(v[:, 2] - z) < 0.01)
    radius = float(np.linalg.norm(v[ring, :2] - head[:2], axis=1).max()) + 0.012
    n, m, r = 64, 8, 0.004
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    b = np.linspace(0, 2 * np.pi, m, endpoint=False)
    A, B = np.meshgrid(a, b, indexing="ij")
    R = radius + r * np.cos(B)
    vertices = np.stack(
        [head[0] + R * np.cos(A), head[1] + R * np.sin(A), z + r * np.sin(B)], -1
    ).reshape(-1, 3)
    normals = np.stack(
        [np.cos(B) * np.cos(A), np.cos(B) * np.sin(A), np.sin(B)], -1
    ).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    p = i * m + j
    q = ((i + 1) % n) * m + j
    pr = i * m + (j + 1) % m
    qr = ((i + 1) % n) * m + (j + 1) % m
    triangles = np.concatenate(
        [
            np.stack([p, q, qr], -1).reshape(-1, 3),
            np.stack([p, qr, pr], -1).reshape(-1, 3),
        ]
    )
    garment = Garment(
        vertices,
        triangles,
        normals=normals,
        materials=[Material("gold", (0.9, 0.7, 0.3, 1.0), 1.0, 0.3)],
    )
    return Wearable(
        name="circlet",
        slot="head",
        layer=60,
        garment=garment,
        binding=bind_rigid(garment, body, "head"),
        weights="rigid",
        licence=dict(LICENCE),
        covers=np.zeros(len(body.vertices), bool),
    )


def bump_normal_map(size: int = 64) -> bytes:
    """A PNG tangent-space normal map (glTF convention) of a grid of round bumps."""
    from opensculptboy.export.document import png_bytes

    u = (np.arange(size) + 0.5) / size
    U, V = np.meshgrid(u, u)
    x = 0.4 * np.cos(8 * np.pi * U) * np.sin(8 * np.pi * V)
    y = 0.4 * np.sin(8 * np.pi * U) * np.cos(8 * np.pi * V)
    n = np.stack([x, y, np.ones_like(x)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return png_bytes(np.round((n * 0.5 + 0.5) * 255).astype(np.uint8))


def cape(body=None, normal_map: bool = False) -> Wearable:
    """A cape: a panel of 12 by 20 vertices behind the back, from the shoulders to the knees,
    4 cm off the skin, which hides the tunic's ``back`` group. With ``normal_map`` the panel
    has UVs and its material a normal map (:func:`bump_normal_map`)."""
    body = body or template_body()
    v = body.vertices
    low, h = float(v[:, 2].min()), float(np.ptp(v[:, 2]))
    shoulder, knee = low + 0.80 * h, low + 0.28 * h
    back = float(v[(np.abs(v[:, 0]) < 0.2) & (v[:, 2] > knee), 1].max()) + 0.04
    xs = np.linspace(-0.22, 0.22, 12)
    zs = np.linspace(shoulder, knee, 20)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    vertices = np.stack([X, np.full_like(X, back), Z], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(11), np.arange(19), indexing="ij")
    p = i * 20 + j
    triangles = np.concatenate(
        [
            np.stack([p, p + 1, p + 21], -1).reshape(-1, 3),
            np.stack([p, p + 21, p + 20], -1).reshape(-1, 3),
        ]
    )
    wool = Material("wool", (0.5, 0.1, 0.1, 1.0), 0.0, 1.0, double_sided=True)
    uvs = None
    if normal_map:
        uvs = np.stack([(X - xs[0]) / np.ptp(xs), (zs[-1] - Z) / np.ptp(zs)], -1)
        uvs = uvs.reshape(-1, 2)
        wool.textures["normal"] = bump_normal_map()
    garment = Garment(
        vertices,
        triangles,
        normals=np.tile([0.0, 1.0, 0.0], (len(vertices), 1)),
        uvs=uvs,
        materials=[wool],
    )
    return Wearable(
        name="cape",
        slot="cloak",
        layer=80,
        garment=garment,
        binding=bind_surface(garment, body),
        licence=dict(LICENCE),
        covers=np.zeros(len(body.vertices), bool),
        hides=[{"piece": "tunic", "groups": ["back"]}],
    )


PIECES = (tunic, circlet, cape)


def outfit_card() -> Character:
    """A card that wears the three pieces as the outfit ``test_outfit`` (files next to it)."""
    names = [make.__name__ for make in PIECES]
    pieces = [
        {"piece": n, "file": f"{n}.wearable.json", "enabled": True} for n in names
    ]
    return Character(
        name="dressed",
        outfits=[{"id": "test_outfit", "label": "Test outfit", "pieces": pieces}],
        active_outfit="test_outfit",
    )


def write_outfit(folder, body=None, normal_map: bool = False) -> Character:
    """Write the three pieces into ``folder`` and return :func:`outfit_card`; with
    ``normal_map`` the cape has a normal map."""
    import pathlib

    from opensculptboy.wearables import write_wearable

    folder = pathlib.Path(folder)
    for make in PIECES:
        w = make(body, normal_map=True) if make is cape and normal_map else make(body)
        write_wearable(folder / f"{w.name}.wearable.json", w)
    return outfit_card()

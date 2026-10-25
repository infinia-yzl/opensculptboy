# OpenSculptBoy
# Apache License, Version 2.0
"""
Synthetic garments for the wearable tests: a tunic (an offset shell of anny's torso), a rigid
circlet around the head and a cape panel behind the back, all made in the test from anny's
template, and a synthetic reference character with a clothing part. :func:`long_cape` and
:func:`skirt` are loose garments with springs (test_wearable_springs).
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


def _grid_triangles(columns: int, rows: int, wrap: bool = False) -> np.ndarray:
    """The triangles of a grid of ``columns`` by ``rows`` vertices (column-major), the last
    column joined to the first with ``wrap``."""
    out = []
    for i in range(columns if wrap else columns - 1):
        n = (i + 1) % columns
        for j in range(rows - 1):
            p, q = i * rows + j, n * rows + j
            out += [[p, p + 1, q + 1], [p, q + 1, q]]
    return np.array(out, np.int64)


def _dominant(body) -> np.ndarray:
    return body.indices[np.arange(len(body.vertices)), np.argmax(body.weights, axis=1)]


def long_cape(body=None, springs=None) -> Wearable:
    """
    A long cape: a panel of 15 by 26 vertices behind the back from the shoulders to the
    calves, 1.5 cm off the back at the top and 6.5 cm at the hem, hanging straight down from
    the widest point of the back, with one ``auto`` spring (the cape preset).
    """
    body = body or template_body()
    v = body.vertices
    low, h = float(v[:, 2].min()), float(np.ptp(v[:, 2]))
    zs = np.linspace(low + 0.81 * h, low + 0.12 * h, 26)
    xs = np.linspace(-0.24, 0.24, 15)
    middle = np.abs(v[:, 0]) < 0.2
    back = []
    for z in zs:
        band = middle & (np.abs(v[:, 2] - z) < 0.02)
        back.append(float(v[band, 1].max()))
    back = np.maximum.accumulate(np.array(back))  # straight down from the widest point
    t = np.linspace(0.0, 1.0, len(zs))
    ys = back + 0.015 + 0.05 * t**1.5
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    Y = np.broadcast_to(ys, X.shape)
    vertices = np.stack([X, Y, Z], -1).reshape(-1, 3)
    garment = Garment(
        vertices,
        _grid_triangles(len(xs), len(zs)),
        normals=np.tile([0.0, 1.0, 0.0], (len(vertices), 1)),
        materials=[Material("wool", (0.3, 0.1, 0.4, 1.0), 0.0, 1.0, double_sided=True)],
    )
    return Wearable(
        name="long_cape",
        slot="cloak",
        layer=80,
        garment=garment,
        binding=bind_surface(garment, body),
        licence=dict(LICENCE),
        covers=np.zeros(len(body.vertices), bool),
        springs=[{"name": "back"}] if springs is None else springs,
    )


ARM_PARTS = ("clavicle", "shoulder", "arm", "wrist", "finger", "metacarpal")


def skirt(body=None, springs=None) -> Wearable:
    """
    A skirt: a ring of 40 by 16 vertices from the waist to above the knees, 1 cm outside the
    hips and the thighs and flared by 16 cm at the hem, with one ``auto`` spring (the skirt
    preset of the ``legs`` slot).
    """
    body = body or template_body()
    v = body.vertices
    labels = body.bone_labels
    low, h = float(v[:, 2].min()), float(np.ptp(v[:, 2]))
    centre = body.bone_poses[labels.index("root"), :2, 3]
    arms = [k for k, b in enumerate(labels) if any(part in b for part in ARM_PARTS)]
    allowed = ~np.isin(_dominant(body), arms)  # the trunk and the legs
    zs = np.linspace(low + 0.58 * h, low + 0.33 * h, 16)
    n = 40
    angles = np.linspace(-np.pi, np.pi, n, endpoint=False)
    rel = v[:, :2] - centre
    az = np.arctan2(rel[:, 1], rel[:, 0])
    radius = np.linalg.norm(rel, axis=1)
    profile = np.zeros((n, len(zs)))
    for i, a in enumerate(angles):
        wedge = allowed & (np.abs(np.angle(np.exp(1j * (az - a)))) < np.pi / n * 1.5)
        for j, z in enumerate(zs):
            band = wedge & (np.abs(v[:, 2] - z) < 0.02)
            profile[i, j] = radius[band].max() if band.any() else 0.0
    profile = np.maximum.accumulate(profile, axis=1) + 0.01
    t = np.linspace(0.0, 1.0, len(zs))
    R = profile + 0.16 * t**1.3
    A = np.broadcast_to(angles[:, None], R.shape)
    Z = np.broadcast_to(zs[None], R.shape)
    vertices = np.stack(
        [centre[0] + R * np.cos(A), centre[1] + R * np.sin(A), Z], -1
    ).reshape(-1, 3)
    normals = np.stack([np.cos(A), np.sin(A), np.zeros_like(A)], -1).reshape(-1, 3)
    garment = Garment(
        vertices,
        _grid_triangles(n, len(zs), wrap=True),
        normals=normals,
        materials=[
            Material("linen", (0.7, 0.6, 0.4, 1.0), 0.0, 0.9, double_sided=True)
        ],
    )
    return Wearable(
        name="skirt",
        slot="legs",
        layer=30,
        garment=garment,
        binding=bind_surface(garment, body),
        licence=dict(LICENCE),
        covers=np.zeros(len(body.vertices), bool),
        springs=[{"name": "hem"}] if springs is None else springs,
    )


SLEEVE_BONES = ("upperarm01", "upperarm02", "lowerarm01", "lowerarm02", "wrist")


def robe(body=None) -> Wearable:
    """
    A robe as a reference handoff gives it: a long back panel (the long cape's) and two wide
    sleeves, 3.5 cm off the skin of the upper arms, the forearms and the wrists, in one piece
    with its own skin weights by bone name (``weights: own``): the body's weights at each
    vertex's closest skin point, smoothed over the piece, the strongest four. The weights of
    the sleeves lie on the upper arm, forearm and hand bones and their twist bones
    (``upperarm02``, ``lowerarm02``). One ``auto`` spring, as reference pieces with physics
    get (:func:`opensculptboy.wearables.reference.reference_springs`).
    """
    from opensculptboy.wearables.bind import (
        dense_body_weights,
        smooth_weights,
        strongest,
    )

    body = body or template_body()
    labels = body.bone_labels
    panel = long_cape(body).garment
    arms = [labels.index(f"{b}.{s}") for b in SLEEVE_BONES for s in ("L", "R")]
    keep = np.isin(_dominant(body), arms)
    shell, shell_triangles, _ = _region_shell(body, keep, 0.035)
    vertices = np.concatenate([panel.vertices, shell])
    triangles = np.concatenate([panel.triangles, shell_triangles + len(panel.vertices)])
    group = np.r_[
        np.zeros(len(panel.triangles), np.int64),
        np.ones(len(shell_triangles), np.int64),
    ]
    garment = Garment(
        vertices,
        triangles,
        triangle_group=group,
        groups=["back", "sleeves"],
        materials=[Material("silk", (0.2, 0.2, 0.5, 1.0), 0.0, 0.8, double_sided=True)],
    )
    binding = bind_surface(garment, body)
    dense = smooth_weights(dense_body_weights(binding, body), vertices, triangles, 10)
    order = strongest(dense, 4)
    w = np.take_along_axis(dense, order, axis=1)
    w = w / w.sum(axis=1, keepdims=True)
    used = np.unique(order[w > 0])
    remap = np.zeros(len(labels), np.int64)
    remap[used] = np.arange(len(used))
    binding.skin_weights = w
    binding.skin_joints = remap[np.where(w > 0, order, used[0])]
    binding.joint_names = [labels[b] for b in used]
    return Wearable(
        name="robe",
        slot="torso",
        layer=50,
        garment=garment,
        binding=binding,
        weights="own",
        licence=dict(LICENCE),
        covers=np.zeros(len(body.vertices), bool),
        springs=[{"name": "loose"}],
    )


def write_loose(folder, body=None, pieces=None) -> Character:
    """Write the long cape and the skirt into ``folder`` and return a card that wears them
    as the outfit ``loose``."""
    import pathlib

    from opensculptboy.wearables import write_wearable

    folder = pathlib.Path(folder)
    made = pieces or [long_cape(body), skirt(body)]
    for w in made:
        write_wearable(folder / f"{w.name}.wearable.json", w)
    return Character(
        name="loose",
        outfits=[
            {
                "id": "loose",
                "label": "Loose clothes",
                "pieces": [
                    {
                        "piece": w.name,
                        "file": f"{w.name}.wearable.json",
                        "enabled": True,
                    }
                    for w in made
                ],
            }
        ],
        active_outfit="loose",
    )


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

# OpenSculptBoy
# Apache License, Version 2.0
"""
A synthetic master for the tests of the rendering-budget tiers (``opensculptboy.tiers``): a card
with offsets on some of levels 0, 1 and 2 and a skin material with its own maps.

The offsets follow one smooth displacement field on every level (a few millimetres, as the
residuals of a fit), and the finest level (the master) adds what only it holds: fine bumps of
about 0.25 mm along the normal, smoothed over one ring (pores and folds a few millimetres
across on level 2). The skin material's normal map adds bumps of its own on anny's atlas, as a
bake of a reference's normal map does. The master's base colour, roughness and normal maps are
512 texels square, so that the tests stay small.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

from opensculptboy import Character
from opensculptboy.detail import detail_topology, save_offsets
from opensculptboy.export.document import png_bytes

MAP_SIZE = 512
PHENOTYPE = {"age": 0.45, "weight": 0.55, "gender": 0.4}


def _field(points: np.ndarray) -> np.ndarray:
    """The smooth displacement field of every level (metres), a function of position."""
    k = 2.0 * np.pi / np.array([0.21, 0.17, 0.25])
    return 0.002 * np.stack(
        [
            np.sin(k[0] * points[:, 1] + 0.3),
            np.sin(k[1] * points[:, 2] + 1.1),
            np.sin(k[2] * points[:, 0] + 2.0),
        ],
        1,
    )


def _smooth(values: np.ndarray, triangles: np.ndarray, rounds: int = 1) -> np.ndarray:
    T = np.asarray(triangles)
    edges = np.unique(
        np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0
    )
    i, j = edges[:, 0], edges[:, 1]
    count = np.bincount(i, minlength=len(values)) + np.bincount(
        j, minlength=len(values)
    )
    for _ in range(rounds):
        spread = np.bincount(i, values[j], len(values)) + np.bincount(
            j, values[i], len(values)
        )
        values = 0.5 * values + 0.5 * spread / np.maximum(count, 1)
    return values


def normal_map(
    size: int = MAP_SIZE, cycles: int = 48, tilt: float = 0.12
) -> np.ndarray:
    """A tangent-space normal map of bumps on anny's atlas, (size, size, 3) uint8."""
    u = (np.arange(size) + 0.5) / size
    x = tilt * np.sin(2.0 * np.pi * cycles * u)[None, :].repeat(size, 0)
    y = tilt * np.cos(2.0 * np.pi * cycles * 0.8 * u)[:, None].repeat(size, 1)
    n = np.stack([x, y, np.ones_like(x)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.clip(np.round((n + 1.0) * 127.5), 0, 255).astype(np.uint8)


def tiered_card(folder: pathlib.Path, levels=(0, 1, 2), seed: int = 0) -> pathlib.Path:
    """Write a master card (``card.json``) with offsets on ``levels`` and a skin material
    into ``folder``; returns the card's path."""
    from opensculptboy.fit.reference import vertex_normals

    folder = pathlib.Path(folder)
    character = Character(name="tiered", phenotype=dict(PHENOTYPE))
    model = character.build_model(dtype=torch.float64)
    with torch.no_grad():
        base = model(**character.model_kwargs())["rest_vertices"][0].numpy()
    rng = np.random.default_rng(seed)
    entries = []
    for subdiv in levels:
        topology = detail_topology(model, subdiv, "full")
        rest = np.asarray(topology.apply(base))
        offsets = _field(rest)
        if subdiv == max(levels):
            bumps = _smooth(
                0.0004 * rng.standard_normal(len(rest)), topology.triangles, 1
            )
            offsets = offsets + bumps[:, None] * vertex_normals(
                rest + offsets, topology.triangles
            )
        name = f"card.detail{subdiv}.safetensors"
        save_offsets(folder / name, offsets, subdiv)
        entries.append({"subdiv": subdiv, "offsets": name})
    textures = {
        "base_color": np.tile(
            np.array([205, 160, 140], np.uint8), (MAP_SIZE, MAP_SIZE, 1)
        ),
        "normal": normal_map(),
        "roughness": np.full((MAP_SIZE, MAP_SIZE), 140, np.uint8),
    }
    material = {
        "subsurface": {"weight": 0.1, "radius_mm": [5.4, 2.1, 1.2]},
        "roughness_factor": 1.0,
    }
    for slot, pixels in textures.items():
        name = f"card.{slot}.png"
        (folder / name).write_bytes(png_bytes(pixels))
        material[slot] = name
    character.detail = {"levels": entries}
    character.material = material
    path = folder / "card.json"
    import json

    path.write_text(json.dumps(character.to_dict(), indent=2) + "\n")
    return path

# OpenSculptBoy
# Apache License, Version 2.0
"""
The detail layer in the exporters: which mesh of :mod:`opensculptboy.detail` a file takes, and
the rest vertices, morph targets and skin weights of that mesh.

:func:`resolve` picks the mesh for a requested level: the card's offsets of that level on the
full subdivided mesh. For a VRM file, :func:`resolve_within_budget` keeps the triangles within
the budget of the version (:data:`opensculptboy.export.budget.BUDGETS`): when the full level
is over it, the head and the neck alone take the level (``region="head"``), and when that is
still over it, the level steps down. Level 0 of a card without detail is no detail layer, and
the exporters then write what they wrote before the detail layer existed.
"""

from __future__ import annotations

import dataclasses
import pathlib
import warnings

import numpy as np

from opensculptboy.export import budget as budgets


@dataclasses.dataclass
class DetailMesh:
    """A mesh of the detail layer with the card's offsets of its level."""

    topology: object  # opensculptboy.detail.subdivision.DetailTopology
    offsets: (
        np.ndarray | None
    )  # (N, 3) offsets of the mesh's vertices, Anny's rest frame
    requested: int  # the level asked for

    @property
    def subdiv(self) -> int:
        return self.topology.subdiv

    @property
    def region(self) -> str:
        return self.topology.region

    def vertices(self, base: np.ndarray) -> np.ndarray:
        """The detailed rest mesh (N, 3) of Anny's rest vertices (V0, 3)."""
        fine = np.asarray(self.topology.apply(np.asarray(base, dtype=np.float64)))
        return fine if self.offsets is None else fine + self.offsets

    def targets(self, offsets: np.ndarray) -> np.ndarray:
        """Morph-target offsets (R, N, 3) of Anny's (R, V0, 3); the detail adds nothing."""
        offsets = np.asarray(offsets, dtype=np.float64)
        if len(offsets) == 0:
            return offsets.reshape(0, self.topology.vertex_count, 3)
        return np.asarray(self.topology.apply_batch(offsets))

    def skin_weights(self, weights, indices):
        return self.topology.skin_weights(weights, indices)

    def extras(self) -> dict:
        return {
            "subdiv": int(self.subdiv),
            "region": self.region,
            "requested": int(self.requested),
            "offsets": self.offsets is not None,
        }


def _check_level(character, level: int) -> None:
    if level not in (0, 1, 2) or isinstance(level, bool):
        raise ValueError(f"The detail level must be 0, 1 or 2; got {level!r}.")
    if level > 0 and level not in character.detail_levels():
        have = character.detail_levels()
        raise ValueError(
            f"The character has no detail level {level}"
            + (f" (its levels are {have})." if have else " (it has no detail layer).")
        )


def resolve(
    model,
    character,
    level: int = 0,
    detail_dir: str | pathlib.Path | None = None,
    region: str = "full",
) -> DetailMesh | None:
    """
    The detail mesh of a file at ``level``, or None for level 0 of a card without level-0
    offsets (the exporters then keep their output as it was). Raises ValueError for a level
    the card does not hold.
    """
    from opensculptboy.detail import character_offsets, detail_topology

    _check_level(character, level)
    offsets = character_offsets(character, level, detail_dir)
    if level == 0 and offsets is None:
        return None
    topology = detail_topology(model, level, region, character.topology)
    rows = None if offsets is None else offsets[topology.rows]
    return DetailMesh(topology, rows, level)


def resolve_within_budget(
    model,
    character,
    level: int,
    target: str,
    mode: str = "warn",
    detail_dir: str | pathlib.Path | None = None,
) -> DetailMesh | None:
    """
    The finest detail mesh up to ``level`` whose triangles fit the budget ``target``
    (``"vrm1"`` or ``"vrm0"``): the full level, else the level on the head and the neck, else
    the next level down that the card holds. With ``mode="off"`` the full level is kept. A
    lowered level issues a :class:`~opensculptboy.export.budget.BudgetWarning`.
    """
    from opensculptboy.detail import detail_topology

    _check_level(character, level)
    if mode == "off" or level == 0:
        return resolve(model, character, level, detail_dir)
    limit = budgets.BUDGETS[budgets.TARGETS[target]]["triangles"]
    held = set(character.detail_levels()) | {0}
    for subdiv in range(level, -1, -1):
        if subdiv not in held:
            continue
        regions = ("full", "head") if subdiv > 0 else ("full",)
        for region in regions:
            triangles = len(
                detail_topology(model, subdiv, region, character.topology).triangles
            )
            if subdiv == 0 or triangles <= limit:
                if (subdiv, region) != (level, "full"):
                    warnings.warn(
                        f"detail level {level} has more triangles than the "
                        f"{budgets.LABELS[budgets.TARGETS[target]]} budget of {limit:,}; "
                        f"the file takes level {subdiv}"
                        + (" on the head and the neck" if region == "head" else ""),
                        budgets.BudgetWarning,
                        stacklevel=3,
                    )
                found = resolve(model, character, subdiv, detail_dir, region)
                if found is not None:
                    found.requested = level
                return found
    return None


TEXTURE_SLOTS = ("base_color", "normal")


def texture_images(
    character, textures=None, detail_dir: str | pathlib.Path | None = None
) -> dict[str, bytes]:
    """
    The PNG or JPEG files of the skin textures of a file, by slot (``"base_color"``,
    ``"normal"``).

    ``textures`` is None (the textures that the card's detail layer names, under
    ``detail.textures``; none for a card without them), False (no texture), or a mapping of
    slots to file paths or bytes. Relative paths of the card resolve against ``detail_dir``.
    """
    if textures is False:
        return {}
    if textures is None:
        textures = {
            k: v
            for k, v in character.detail.get("textures", {}).items()
            if k in TEXTURE_SLOTS
        }
        base = None if detail_dir is None else pathlib.Path(detail_dir)
    else:
        base = None
    unknown = sorted(set(textures) - set(TEXTURE_SLOTS))
    if unknown:
        raise ValueError(f"Unknown texture slots {unknown}; use {list(TEXTURE_SLOTS)}.")
    images = {}
    for slot, value in textures.items():
        if isinstance(value, (bytes, bytearray)):
            data = bytes(value)
        else:
            path = pathlib.Path(value)
            if not path.is_absolute() and base is not None:
                path = base / path
            data = path.read_bytes()
        if budgets.image_size(data) is None:
            raise ValueError(f"The {slot} texture is neither a PNG nor a JPEG file.")
        images[slot] = data
    return images


def mime_type(data: bytes) -> str:
    return "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"

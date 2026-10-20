# OpenSculptBoy
# Apache License, Version 2.0
"""
The offsets of the detail layer and the model that applies them.

Each level of a character's detail layer stores one offset per vertex of the full level
(:func:`opensculptboy.detail.subdivision.level`) in a ``.safetensors`` sidecar file next to the
card: rest-frame vectors in Anny's frame (metres, Z up, the figure facing -Y). The offsets
add to the subdivided rest mesh after the blend shapes (phenotype, local changes, face shapes,
facial actions) and before the skinning, as :class:`DetailModel` does. They belong to the
card's own parameter values: a fit stores what the parametric body of the card misses, so a
changed slider keeps the offsets where they were in the rest frame.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

from opensculptboy.detail.subdivision import (
    KEEP_BONES,
    DetailTopology,
    detail_topology,
    level,
)

OFFSETS_FORMAT = "opensculptboy/detail-offsets@1"


def save_offsets(
    path: str | pathlib.Path,
    offsets: np.ndarray,
    subdiv: int,
    topology: str = "anny",
    metadata: dict | None = None,
) -> pathlib.Path:
    """Write the (N, 3) offsets of level ``subdiv`` as a ``.safetensors`` file."""
    from safetensors.numpy import save_file

    offsets = np.ascontiguousarray(offsets, dtype=np.float32)
    expected = level(topology, subdiv).vertex_count
    if offsets.shape != (expected, 3):
        raise ValueError(
            f"Level {subdiv} of {topology!r} has {expected} vertices; got offsets of "
            f"shape {offsets.shape}."
        )
    meta = {
        "format": OFFSETS_FORMAT,
        "subdiv": str(subdiv),
        "topology": topology,
        "vertices": str(expected),
        "frame": "Anny rest frame: metres, Z up, the figure faces -Y",
    }
    meta.update({k: str(v) for k, v in (metadata or {}).items()})
    path = pathlib.Path(path)
    save_file({"offsets": offsets}, str(path), metadata=meta)
    return path


def load_offsets(
    path: str | pathlib.Path, subdiv: int, topology: str = "anny"
) -> np.ndarray:
    """The (N, 3) float64 offsets of level ``subdiv`` from a file of :func:`save_offsets`."""
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        meta = f.metadata() or {}
        offsets = f.get_tensor("offsets")
    if meta.get("format") != OFFSETS_FORMAT:
        raise ValueError(f"{path} is not a detail offsets file ({OFFSETS_FORMAT}).")
    if int(meta.get("subdiv", -1)) != subdiv or meta.get("topology") != topology:
        raise ValueError(
            f"{path} holds level {meta.get('subdiv')} of {meta.get('topology')!r}; "
            f"level {subdiv} of {topology!r} was asked for."
        )
    expected = level(topology, subdiv).vertex_count
    if offsets.shape != (expected, 3):
        raise ValueError(
            f"{path}: offsets of shape {offsets.shape}; level {subdiv} has {expected} vertices."
        )
    return offsets.astype(np.float64)


def character_offsets(
    character, subdiv: int, base_dir: str | pathlib.Path | None = None
) -> np.ndarray | None:
    """
    The offsets of level ``subdiv`` of a character's detail layer, or None when the card has no
    such level. Relative paths resolve against ``base_dir`` (the folder of the card; the
    current folder when None).
    """
    for entry in character.detail.get("levels", []):
        if int(entry["subdiv"]) == subdiv:
            path = pathlib.Path(entry["offsets"])
            if not path.is_absolute() and base_dir is not None:
                path = pathlib.Path(base_dir) / path
            return load_offsets(path, subdiv, character.topology)
    return None


def blend_skinning(
    rest: torch.Tensor,
    transforms: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Linear blend skinning of rest vertices (B, N, 3) by bone transforms (B, J, 4, 4) (posed
    world matrix times the inverse rest matrix) with weights and indices (N, K).
    """
    flat = transforms[..., :3, :].reshape(transforms.shape[0], -1, 12)
    blended = 0.0
    for k in range(weights.shape[1]):  # one bone slot at a time keeps the memory small
        blended = blended + weights[None, :, k, None] * flat[:, indices[:, k]]
    blended = blended.reshape(*blended.shape[:2], 3, 4)
    return torch.einsum("bnij, bnj -> bni", blended[..., :3], rest) + blended[..., 3]


def bone_transforms(output: dict) -> torch.Tensor:
    """Transforms (B, J, 4, 4) that skin rest vertices: bone_poses times rest_bone_poses^-1."""
    rest = output["rest_bone_poses"]
    inverse = torch.linalg.inv(rest)
    return output["bone_poses"] @ inverse


class DetailModel:
    """
    An Anny model with a level of the detail layer.

    Calling it runs the Anny model with the same keyword arguments, subdivides the rest mesh,
    adds the offsets (the rest frame, after the blend shapes) and skins the result with the
    subdivided skin weights by linear blend skinning. Without offsets, level 0 gives Anny's own
    ``vertices`` up to rounding (for an Anny model with ``skinning_method`` "lbs" or
    "warp_lbs").

    Args:
        model: the Anny model (its topology's quad mesh must have the same vertices).
        subdiv: the level, 0, 1 or 2.
        offsets: (N, 3) offsets of the full level in Anny's rest frame, or None.
        region: ``"full"`` or ``"head"`` (see :func:`detail_topology`).
        topology: the model's topology name (the card's ``topology``).
    """

    def __init__(
        self,
        model,
        subdiv: int = 0,
        offsets: np.ndarray | None = None,
        region: str = "full",
        topology: str = "anny",
        keep: int = KEEP_BONES,
    ):
        self.model = model
        self.topology: DetailTopology = detail_topology(model, subdiv, region, topology)
        dtype, device = model.template_vertices.dtype, model.template_vertices.device
        weights, indices = self.topology.skin_weights(
            model.vertex_bone_weights.detach().cpu().double().numpy(),
            model.vertex_bone_indices.detach().cpu().numpy(),
            keep,
        )
        self.weights = torch.as_tensor(weights, dtype=dtype, device=device)
        self.indices = torch.as_tensor(indices, device=device)
        self.offsets = None
        if offsets is not None:
            offsets = np.asarray(offsets)[self.topology.rows]
            self.offsets = torch.as_tensor(offsets, dtype=dtype, device=device)

    @property
    def triangles(self) -> np.ndarray:
        return self.topology.triangles

    def rest(self, rest_vertices: torch.Tensor) -> torch.Tensor:
        """The detailed rest mesh (B, N, 3) of Anny's rest vertices (B, V0, 3)."""
        fine = self.topology.apply_batch(rest_vertices)
        if self.offsets is not None:
            fine = fine + self.offsets.to(fine.dtype)
        return fine

    def __call__(self, **kwargs) -> dict:
        output = self.model(**kwargs)
        rest = self.rest(output["rest_vertices"])
        vertices = blend_skinning(
            rest,
            bone_transforms(output),
            self.weights.to(rest.dtype),
            self.indices,
        )
        return dict(
            rest_vertices=rest,
            vertices=vertices,
            bone_poses=output["bone_poses"],
            rest_bone_poses=output["rest_bone_poses"],
            anny_output=output,
        )

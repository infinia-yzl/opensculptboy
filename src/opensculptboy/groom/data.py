# OpenSculptBoy
# Apache License, Version 2.0
"""
The groom files of a fitted card: the reference's hair or brows in the card's rest frame.

A :class:`GroomData` holds one kind of groom (``"hair"`` or ``"brows"``): its points, normals,
UVs (glTF convention, v = 0 at the top) and triangles, the material of each triangle, the
vertex colours where the reference gives them, and the materials as the reference draws them
(:class:`GroomMaterial`: base colour factor, roughness, alpha mode, cutoff and sides, and the
names of their textures). The hair also holds its skin on Anny's bones (``bones``,
``bone_slots``, ``bone_weights``), the spring chains of the reference's chain joints in the
card's rest frame (``chains``) and each point's share of weight on those joints
(``chain_share``). The file ``<card>.<kind>.safetensors`` (:data:`FORMAT`) holds the arrays,
with the kind, the materials, the bones and the chains in its metadata; the card's
``material.hair`` or ``material.brows`` section names it and the textures.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np

FORMAT = "opensculptboy/groom@1"
KINDS = ("hair", "brows")


def file_stem(name: str) -> str:
    """A material name as a part of a file name: letters, digits, ``-`` and ``_``."""
    import re

    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "material"


@dataclasses.dataclass
class GroomMaterial:
    """A material of a groom, as the reference draws it: the base colour factor (linear
    RGBA), the roughness, the alpha mode (``"OPAQUE"``, ``"MASK"`` or ``"BLEND"``), the alpha
    cutoff, whether both sides draw, and the slot names of its textures (``"base_color"``,
    ``"normal"``) that the card's section maps to files."""

    name: str
    base_color: tuple = (1.0, 1.0, 1.0, 1.0)
    roughness: float = 0.6
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    normal_scale: float = 1.0
    textures: tuple = ()

    def json(self) -> dict:
        return {
            "name": self.name,
            "base_color": [round(float(c), 6) for c in self.base_color],
            "roughness": round(float(self.roughness), 6),
            "alpha_mode": self.alpha_mode,
            "alpha_cutoff": round(float(self.alpha_cutoff), 6),
            "double_sided": bool(self.double_sided),
            "normal_scale": round(float(self.normal_scale), 6),
            "textures": list(self.textures),
        }

    @classmethod
    def from_json(cls, data: dict) -> "GroomMaterial":
        return cls(
            name=str(data["name"]),
            base_color=tuple(float(c) for c in data.get("base_color", (1, 1, 1, 1))),
            roughness=float(data.get("roughness", 0.6)),
            alpha_mode=str(data.get("alpha_mode", "OPAQUE")),
            alpha_cutoff=float(data.get("alpha_cutoff", 0.5)),
            double_sided=bool(data.get("double_sided", False)),
            normal_scale=float(data.get("normal_scale", 1.0)),
            textures=tuple(data.get("textures", ())),
        )


@dataclasses.dataclass
class GroomData:
    """A groom in a card's rest frame (see the module documentation)."""

    kind: str
    points: np.ndarray  # (M, 3)
    normals: np.ndarray  # (M, 3)
    uvs: np.ndarray  # (M, 2), glTF convention
    triangles: np.ndarray  # (T, 3)
    material: np.ndarray  # (T,) index into materials
    materials: list
    colors: np.ndarray | None = None  # (M, 4) linear RGBA
    bones: tuple = ("head",)
    bone_slots: np.ndarray | None = None  # (M, K) indices into bones
    bone_weights: np.ndarray | None = None  # (M, K)
    chains: list = dataclasses.field(default_factory=list)  # [(n, 3) points]
    chain_share: np.ndarray | None = None  # (M,)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"A groom's kind is one of {KINDS} (got {self.kind!r}).")
        self.points = np.asarray(self.points, dtype=np.float64)
        self.normals = np.asarray(self.normals, dtype=np.float64)
        self.uvs = np.asarray(self.uvs, dtype=np.float64)
        self.triangles = np.asarray(self.triangles, dtype=np.int64)
        self.material = np.asarray(self.material, dtype=np.int64)
        if self.colors is not None:
            self.colors = np.asarray(self.colors, dtype=np.float64)
        self.bones = tuple(str(b) for b in self.bones)
        if self.bone_slots is None:
            self.bone_slots = np.zeros((len(self.points), 1), np.int64)
            self.bone_weights = np.ones((len(self.points), 1))
        self.bone_slots = np.asarray(self.bone_slots, dtype=np.int64)
        self.bone_weights = np.asarray(self.bone_weights, dtype=np.float64)
        self.chains = [
            np.asarray(c, dtype=np.float64).reshape(-1, 3) for c in self.chains
        ]
        if self.chain_share is None:
            self.chain_share = np.zeros(len(self.points))
        self.chain_share = np.asarray(self.chain_share, dtype=np.float64)
        self.materials = [
            m if isinstance(m, GroomMaterial) else GroomMaterial.from_json(m)
            for m in self.materials
        ]

    def save(self, path) -> None:
        from safetensors.numpy import save_file

        tensors = {
            "points": self.points,
            "normals": self.normals,
            "uvs": self.uvs,
            "triangles": self.triangles,
            "material": self.material,
            "bone_slots": self.bone_slots,
            "bone_weights": self.bone_weights,
            "chain_share": self.chain_share,
        }
        if self.colors is not None:
            tensors["colors"] = self.colors
        save_file(
            {k: np.ascontiguousarray(v) for k, v in tensors.items()},
            str(path),
            metadata={
                "format": FORMAT,
                "kind": self.kind,
                "materials": json.dumps([m.json() for m in self.materials]),
                "bones": json.dumps(list(self.bones)),
                "chains": json.dumps(
                    [[[round(float(x), 7) for x in p] for p in c] for c in self.chains]
                ),
            },
        )

    @classmethod
    def load(cls, path) -> "GroomData":
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as f:
            meta = f.metadata() or {}
            if meta.get("format") != FORMAT:
                raise ValueError(f"{path} is not a groom file ({FORMAT}).")
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        return cls(
            kind=meta["kind"],
            materials=json.loads(meta.get("materials", "[]")),
            bones=tuple(json.loads(meta.get("bones", '["head"]'))),
            chains=json.loads(meta.get("chains", "[]")),
            **tensors,
        )

    def material_triangles(self, k: int) -> np.ndarray:
        """(t, 3) the triangles of material ``k``, in their drawing order."""
        return self.triangles[self.material == k]

    def mean_color(self, k: int) -> np.ndarray:
        """(4,) the mean vertex colour of material ``k``'s points, or white without
        vertex colours."""
        if self.colors is None:
            return np.ones(4)
        rows = np.unique(self.material_triangles(k))
        if len(rows) == 0:
            return np.ones(4)
        return self.colors[rows].mean(axis=0)

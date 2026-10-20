# OpenSculptBoy
# Apache License, Version 2.0
"""
The skin material of a character card (``Character.material``, character schema 3).

The section describes the skin on anny's UV atlas in renderer-neutral terms, so that each
exporter can rebuild it in its own way: the glTF export as a metallic-roughness material with
``KHR_materials_specular``, ``KHR_materials_ior`` and the subsurface settings in the material's
extras (:mod:`opensculptboy.export.gltf`), Blender as a Principled BSDF with random-walk
subsurface scattering (:mod:`opensculptboy.export.blender_skin`), and VRM as an MToon ramp
that imitates the warmth of the scattering (:func:`opensculptboy.export.skin_ramp.toon_ramp`).

Every key is optional::

    "material": {
        "base_color": "card.base_color.png",          # sRGB texture on anny's atlas, or a
        "base_color_factor": [1, 1, 1, 1],            #   linear RGBA factor that multiplies it
        "normal": "card.normal.png",                  # tangent space, MikkTSpace frames of the
        "normal_scale": 1.0,                          #   exported mesh, +Y up (OpenGL)
        "roughness": "card.roughness.png",            # a grey texture, or a number
        "roughness_factor": 1.0,                      #   that multiplies it
        "specular": 1.0,                              # KHR_materials_specular specularFactor
        "specular_color": [1, 1, 1],                  #   and specularColorFactor (linear)
        "ior": 1.5,                                   # the index of refraction of the surface
        "subsurface": {
            "weight": 0.1,                            # 0 (none) to 1 (all diffuse light scatters)
            "radius_mm": [5.4, 2.1, 1.2],             # mean free path per RGB channel, mm
            "color": [0.8, 0.5, 0.4],                 # optional: the scattering colour (linear);
                                                      #   without it, the base colour
            "ior": 1.4,                               # the index of refraction of the volume
            "anisotropy": 0.0,                        # -1 (back) to 1 (forward) scattering
            "method": "random_walk_skin"              # "random_walk_skin", "random_walk" or
        },                                            #   "burley" (Blender's methods)
        "thickness": "card.thickness.png",            # grey: thickness / thickness_mm
        "thickness_mm": 30.0,
        "curvature": "card.curvature.png",            # grey: 0.5 + 0.5 * k / curvature_per_m,
        "curvature_per_m": 100.0,                     #   k the mean curvature (1 / radius)
        "head": {                                     # the head tile: textures of the head's
            "base_color": "card.head_base_color.png", #   UV island alone, at a finer texel
            "normal": "card.head_normal.png",         #   (opensculptboy.texture.tiles)
            "roughness": "card.head_roughness.png"
        }
    }

Texture values are paths relative to the card, as the detail layer's files are. The thickness
and curvature maps are baked from the fitted mesh (:mod:`opensculptboy.texture.bake`) for cheap
real-time subsurface scattering: the thickness drives translucency through thin parts such as
the ears, and the curvature picks the row of a pre-integrated skin lookup (Penner 2011). A card
without the section keeps the material it had before (the detail layer's textures, else a
plain skin colour).
"""

from __future__ import annotations

import dataclasses
from typing import Any

SUBSURFACE_METHODS = ("random_walk_skin", "random_walk", "burley")
TEXTURE_KEYS = ("base_color", "normal", "roughness", "thickness", "curvature")
HEAD_TEXTURE_KEYS = ("base_color", "normal", "roughness")
DEFAULTS = {
    "base_color_factor": [1.0, 1.0, 1.0, 1.0],
    "normal_scale": 1.0,
    "roughness": 0.5,
    "roughness_factor": 1.0,
    "specular": 1.0,
    "specular_color": [1.0, 1.0, 1.0],
    "ior": 1.5,
    "thickness_mm": 30.0,
    "curvature_per_m": 100.0,
}
SUBSURFACE_DEFAULTS = {
    "weight": 0.0,
    "radius_mm": [1.0, 0.2, 0.1],
    "ior": 1.4,
    "anisotropy": 0.0,
    "method": "random_walk_skin",
}
_KEYS = set(TEXTURE_KEYS) | set(DEFAULTS) | {"subsurface", "head"}
_SUBSURFACE_KEYS = set(SUBSURFACE_DEFAULTS) | {"color"}


def _number(value, name: str, low=None, high=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"material {name} must be a number (got {value!r}).")
    value = float(value)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"material {name} must lie in [{low}, {high}] (got {value}).")
    return value


def _colour(value, name: str, size: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"material {name} must be a list of {size} numbers.")
    return [_number(v, name, 0.0) for v in value]


@dataclasses.dataclass
class SkinMaterial:
    """The resolved skin material: every value present, textures as card-relative paths."""

    base_color: str | None = None
    base_color_factor: list = dataclasses.field(
        default_factory=lambda: list(DEFAULTS["base_color_factor"])
    )
    normal: str | None = None
    normal_scale: float = 1.0
    roughness: float | str = 0.5  # a number, or the path of a grey texture
    roughness_factor: float = 1.0
    specular: float = 1.0
    specular_color: list = dataclasses.field(default_factory=lambda: [1.0, 1.0, 1.0])
    ior: float = 1.5
    subsurface: dict = dataclasses.field(
        default_factory=lambda: dict(SUBSURFACE_DEFAULTS)
    )
    thickness: str | None = None
    thickness_mm: float = 30.0
    curvature: str | None = None
    curvature_per_m: float = 100.0
    head: dict = dataclasses.field(
        default_factory=dict
    )  # slot -> path of the head tile

    @classmethod
    def from_dict(cls, data: dict | None) -> "SkinMaterial":
        """The material of a card's ``material`` section; ValueError for an invalid one."""
        data = check_material(data or {})
        values = {k: v for k, v in data.items() if k != "subsurface"}
        subsurface = dict(SUBSURFACE_DEFAULTS)
        subsurface.update(data.get("subsurface", {}))
        return cls(subsurface=subsurface, **values)

    def textures(self) -> dict[str, str]:
        """The texture paths by slot (``base_color``, ``normal``, ``roughness``, ...; the head
        tile's as ``head_base_color``, ``head_normal`` and ``head_roughness``)."""
        out = {}
        for key in TEXTURE_KEYS:
            value = getattr(self, key)
            if isinstance(value, str):
                out[key] = value
        for key, value in self.head.items():
            out[f"head_{key}"] = value
        return out

    @property
    def roughness_value(self) -> float:
        """The roughness factor of a glTF material: the number, or the texture's multiplier."""
        if isinstance(self.roughness, str):
            return self.roughness_factor
        return self.roughness * self.roughness_factor

    def subsurface_extras(self) -> dict[str, Any]:
        """The subsurface settings as the glTF export stores them in the material's extras."""
        s = self.subsurface
        out = {
            "weight": float(s["weight"]),
            "radius_mm": [float(x) for x in s["radius_mm"]],
            "ior": float(s["ior"]),
            "anisotropy": float(s["anisotropy"]),
            "method": s["method"],
        }
        if s.get("color") is not None:
            out["color"] = [float(x) for x in s["color"]]
        return out


def check_material(data: dict) -> dict:
    """Raise ValueError for a ``material`` section that is not valid; returns it."""
    if not isinstance(data, dict):
        raise ValueError("The material field must be a JSON object.")
    unknown = sorted(set(data) - _KEYS)
    if unknown:
        raise ValueError(
            f"Unknown material keys {unknown}; the keys are {sorted(_KEYS)}."
        )
    for key in TEXTURE_KEYS:
        if key in data and key != "roughness" and not isinstance(data[key], str):
            raise ValueError(f"material {key} must be the path of a texture.")
    if "roughness" in data and not isinstance(data["roughness"], str):
        _number(data["roughness"], "roughness", 0.0, 1.0)
    if "base_color_factor" in data:
        _colour(data["base_color_factor"], "base_color_factor", 4)
    if "specular_color" in data:
        _colour(data["specular_color"], "specular_color", 3)
    for key, low, high in (
        ("normal_scale", 0.0, None),
        ("roughness_factor", 0.0, None),
        ("specular", 0.0, 1.0),
        ("ior", 1.0, None),
        ("thickness_mm", 0.0, None),
        ("curvature_per_m", 0.0, None),
    ):
        if key in data:
            _number(data[key], key, low, high)
    head = data.get("head", {})
    if not isinstance(head, dict) or set(head) - set(HEAD_TEXTURE_KEYS):
        raise ValueError(
            f"material head must map some of {list(HEAD_TEXTURE_KEYS)} to texture paths."
        )
    if not all(isinstance(v, str) for v in head.values()):
        raise ValueError("material head textures must be paths.")
    subsurface = data.get("subsurface", {})
    if not isinstance(subsurface, dict):
        raise ValueError("material subsurface must be a JSON object.")
    unknown = sorted(set(subsurface) - _SUBSURFACE_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown subsurface keys {unknown}; the keys are {sorted(_SUBSURFACE_KEYS)}."
        )
    if "weight" in subsurface:
        _number(subsurface["weight"], "subsurface weight", 0.0, 1.0)
    if "radius_mm" in subsurface:
        _colour(subsurface["radius_mm"], "subsurface radius_mm", 3)
    if subsurface.get("color") is not None:
        _colour(subsurface["color"], "subsurface color", 3)
    if "ior" in subsurface:
        _number(subsurface["ior"], "subsurface ior", 1.0, 4.0)
    if "anisotropy" in subsurface:
        _number(subsurface["anisotropy"], "subsurface anisotropy", -1.0, 1.0)
    if subsurface.get("method", "random_walk_skin") not in SUBSURFACE_METHODS:
        raise ValueError(
            f"subsurface method must be one of {list(SUBSURFACE_METHODS)} "
            f"(got {subsurface['method']!r})."
        )
    return data

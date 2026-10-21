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
        },
        "ears": {                                     # the ear tile: the outer ears unwrapped
            "uvs": "card.ear_uvs.safetensors",        #   again on levels 1 and 2, with maps of
            "base_color": "card.ears_base_color.png", #   their own (opensculptboy.texture.ears)
            "normal": "card.ears_normal.png",
            "roughness": "card.ears_roughness.png",
            "thickness": "card.ears_thickness.png",
            "curvature": "card.ears_curvature.png"
        },
        "eyes": {                                     # the eyes of opensculptboy.eyes (see
            "iris_color": [0.2, 0.1, 0.04],           #   EyeMaterial): the eyeball, the cornea,
            "pupil_mm": 3.5                           #   the caruncle and their materials
        }
    }

Texture values are paths relative to the card, as the detail layer's files are. The thickness
and curvature maps are baked from the fitted mesh (:mod:`opensculptboy.texture.bake`) for cheap
real-time subsurface scattering: the thickness drives translucency through thin parts such as
the ears, and the curvature picks the row of a pre-integrated skin lookup (Penner 2011). A card
without the section keeps the material it had before (the detail layer's textures, else a
plain skin colour). A section that holds nothing but ``eyes`` keeps the exporters' own skin
as well, and changes the eyes alone.
"""

from __future__ import annotations

import dataclasses
from typing import Any

SUBSURFACE_METHODS = ("random_walk_skin", "random_walk", "burley")
TEXTURE_KEYS = ("base_color", "normal", "roughness", "thickness", "curvature")
HEAD_TEXTURE_KEYS = ("base_color", "normal", "roughness")
EAR_TEXTURE_KEYS = ("base_color", "normal", "roughness", "thickness", "curvature")
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
_KEYS = set(TEXTURE_KEYS) | set(DEFAULTS) | {"subsurface", "head", "ears", "eyes"}
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
    # the head tile: slot -> path; the ear tile: "uvs" and slot -> path
    head: dict = dataclasses.field(default_factory=dict)
    ears: dict = dataclasses.field(default_factory=dict)
    # the eyes (EyeMaterial), or None for anny's own eyeballs
    eyes: "EyeMaterial | None" = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "SkinMaterial":
        """The material of a card's ``material`` section; ValueError for an invalid one."""
        data = check_material(data or {})
        values = {k: v for k, v in data.items() if k not in ("subsurface", "eyes")}
        subsurface = dict(SUBSURFACE_DEFAULTS)
        subsurface.update(data.get("subsurface", {}))
        eyes = EyeMaterial.from_dict(data["eyes"]) if "eyes" in data else None
        return cls(subsurface=subsurface, eyes=eyes, **values)

    def textures(self) -> dict[str, str]:
        """The texture paths by slot (``base_color``, ``normal``, ``roughness``, ...; the head
        tile's as ``head_base_color``, ``head_normal`` and ``head_roughness``, the ear tile's
        as ``ears_base_color`` and so on)."""
        out = {}
        for key in TEXTURE_KEYS:
            value = getattr(self, key)
            if isinstance(value, str):
                out[key] = value
        for key, value in self.head.items():
            out[f"head_{key}"] = value
        for key, value in self.ears.items():
            if key != "uvs":
                out[f"ears_{key}"] = value
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
    ears = data.get("ears", {})
    if not isinstance(ears, dict) or set(ears) - set(EAR_TEXTURE_KEYS) - {"uvs"}:
        raise ValueError(
            f"material ears must map 'uvs' and some of {list(EAR_TEXTURE_KEYS)} to paths."
        )
    if not all(isinstance(v, str) for v in ears.values()):
        raise ValueError("material ears must hold paths.")
    if ears and "uvs" not in ears:
        raise ValueError("material ears needs its 'uvs' file.")
    if "eyes" in data:
        check_eyes(data["eyes"])
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


# The eyes (``material.eyes``, :mod:`opensculptboy.eyes`): every key is optional. Colours are
# linear RGB, sizes millimetres (anny's eyeball is 29.9 mm across, a human's about 24 mm).
EYE_DEFAULTS = {
    "source": "auto",  # "auto" (the baked eye texture when the file has one), "procedural",
    #                    "baked"
    "base_color": None,  # an eye texture of its own, in the layout of opensculptboy.eyes.texture
    "iris_color": [0.20, 0.10, 0.04],  # the outer iris: a medium brown
    "iris_color_inner": None,  # the zone inside the collarette; None: a lighter amber of it
    "iris_tint": 0.0,  # 0 to 1: how far a baked iris turns to iris_color (keeping its pattern)
    "iris_mm": 11.7,  # the iris (limbus) diameter
    "pupil_mm": 3.5,  # the pupil diameter, in neutral light 3 to 4 mm
    "limbal_ring": 0.6,  # 0 to 1: the darkening of the ring around the iris
    "limbal_ring_mm": 0.6,  # its width
    "sclera_color": [0.80, 0.74, 0.68],  # a slightly warm white
    "veins": 0.15,  # 0 to 1: faint red veins at the periphery of the sclera
    "seed": 0,  # of the procedural pattern
    "sclera_roughness": 0.45,  # the eyeball under the cornea
    "cornea": True,  # glTF: a refracting cornea shell over a recessed iris; False: one
    #                  glossy surface with the iris on its bulge (as the VRM files have)
    "cornea_ior": 1.376,
    "cornea_roughness": 0.02,
    "bulge": 1.0,  # the corneal bulge, a share of its anatomical height (lowered to clear
    #                the lids)
    "caruncle": True,  # the mound in the inner corner and the pink, wet skin of the socket
    "caruncle_color": [0.42, 0.15, 0.13],  # both of them (the conjunctiva)
    "highlight": 1.0,  # VRM: the strength of the matcap highlight
}
EYE_SOURCES = ("auto", "procedural", "baked")


@dataclasses.dataclass
class EyeMaterial:
    """The resolved eyes of a card: every value of :data:`EYE_DEFAULTS` present."""

    source: str = "auto"
    base_color: str | None = None
    iris_color: list = dataclasses.field(
        default_factory=lambda: list(EYE_DEFAULTS["iris_color"])
    )
    iris_color_inner: list | None = None
    iris_tint: float = 0.0
    iris_mm: float = 11.7
    pupil_mm: float = 3.5
    limbal_ring: float = 0.6
    limbal_ring_mm: float = 0.6
    sclera_color: list = dataclasses.field(
        default_factory=lambda: list(EYE_DEFAULTS["sclera_color"])
    )
    veins: float = 0.15
    seed: int = 0
    sclera_roughness: float = 0.45
    cornea: bool = True
    cornea_ior: float = 1.376
    cornea_roughness: float = 0.02
    bulge: float = 1.0
    caruncle: bool = True
    caruncle_color: list = dataclasses.field(
        default_factory=lambda: list(EYE_DEFAULTS["caruncle_color"])
    )
    highlight: float = 1.0

    @classmethod
    def from_dict(cls, data: dict | None) -> "EyeMaterial":
        """The eyes of a card's ``material.eyes`` section; ValueError for an invalid one."""
        data = check_eyes(data or {})
        return cls(
            **{k: (list(v) if isinstance(v, list) else v) for k, v in data.items()}
        )

    def inner_color(self) -> list:
        """The colour inside the collarette: ``iris_color_inner``, or a lighter amber of the
        iris colour."""
        if self.iris_color_inner is not None:
            return list(self.iris_color_inner)
        r, g, b = self.iris_color
        return [min(1.0, 1.5 * r + 0.015), min(1.0, 1.45 * g + 0.01), min(1.0, 1.1 * b)]

    def extras(self) -> dict[str, Any]:
        """The settings that the glTF export keeps in the eye materials' extras."""
        return {
            "iris_mm": float(self.iris_mm),
            "pupil_mm": float(self.pupil_mm),
            "cornea_ior": float(self.cornea_ior),
            "cornea_roughness": float(self.cornea_roughness),
            "sclera_roughness": float(self.sclera_roughness),
            "cornea": bool(self.cornea),
        }


def check_eyes(data: dict) -> dict:
    """Raise ValueError for a ``material.eyes`` section that is not valid; returns it."""
    if not isinstance(data, dict):
        raise ValueError("material eyes must be a JSON object.")
    unknown = sorted(set(data) - set(EYE_DEFAULTS))
    if unknown:
        raise ValueError(
            f"Unknown eye keys {unknown}; the keys are {sorted(EYE_DEFAULTS)}."
        )
    if data.get("source", "auto") not in EYE_SOURCES:
        raise ValueError(
            f"eyes source must be one of {list(EYE_SOURCES)} (got {data['source']!r})."
        )
    if data.get("base_color") is not None and not isinstance(data["base_color"], str):
        raise ValueError("eyes base_color must be the path of a texture.")
    for key in ("iris_color", "sclera_color", "caruncle_color"):
        if key in data:
            _colour(data[key], f"eyes {key}", 3)
    if data.get("iris_color_inner") is not None:
        _colour(data["iris_color_inner"], "eyes iris_color_inner", 3)
    for key, low, high in (
        ("iris_tint", 0.0, 1.0),
        ("iris_mm", 6.0, 16.0),
        ("pupil_mm", 1.0, 9.0),
        ("limbal_ring", 0.0, 1.0),
        ("limbal_ring_mm", 0.0, 3.0),
        ("veins", 0.0, 1.0),
        ("sclera_roughness", 0.0, 1.0),
        ("cornea_ior", 1.0, 2.0),
        ("cornea_roughness", 0.0, 1.0),
        ("bulge", 0.0, 1.0),
        ("highlight", 0.0, 4.0),
    ):
        if key in data:
            _number(data[key], f"eyes {key}", low, high)
    if data.get("pupil_mm", 3.5) >= data.get("iris_mm", 11.7) - 1.0:
        raise ValueError("eyes pupil_mm must be at least 1 mm below iris_mm.")
    for key in ("cornea", "caruncle"):
        if key in data and not isinstance(data[key], bool):
            raise ValueError(f"eyes {key} must be true or false.")
    if "seed" in data and (
        isinstance(data["seed"], bool) or not isinstance(data["seed"], int)
    ):
        raise ValueError("eyes seed must be an integer.")
    return data

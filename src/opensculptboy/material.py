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
        "occlusion": "card.occlusion.png",            # grey: ambient occlusion (1 open, 0 shut)
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
        },
        "tiers": {                                    # the normal maps of the cheaper tiers,
            "low": {                                  #   baked from the master onto the mesh
                "subdiv": 0, "region": "full",        #   of each tier (opensculptboy.tiers)
                "normal": "card.low.normal.png"
            }
        }
    }

Texture values are paths relative to the card, as the detail layer's files are. The thickness
and curvature maps are baked from the fitted mesh (:mod:`opensculptboy.texture.bake`) for cheap
real-time subsurface scattering: the thickness drives translucency through thin parts such as
the ears, and the curvature picks the row of a pre-integrated skin lookup (Penner 2011). The
occlusion map is baked from the master mesh (:mod:`opensculptboy.texture.transfer`) for the toon
tier. ``tiers`` holds, per rendering-budget tier of :mod:`opensculptboy.tiers`, the normal maps
baked from the master's surface onto that tier's coarser mesh (its ``subdiv`` and ``region``),
which take the place of the master's maps when the tier is exported. A card without the section
keeps the material it had before (the detail layer's textures, else a plain skin colour). A
section that holds nothing but ``eyes`` keeps the exporters' own skin as well, and changes the
eyes alone; ``teeth`` (:data:`TEETH_DEFAULTS`) likewise adds teeth.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

SUBSURFACE_METHODS = ("random_walk_skin", "random_walk", "burley")
TEXTURE_KEYS = (
    "base_color",
    "normal",
    "roughness",
    "thickness",
    "curvature",
    "occlusion",
)
HEAD_TEXTURE_KEYS = ("base_color", "normal", "roughness")
EAR_TEXTURE_KEYS = ("base_color", "normal", "roughness", "thickness", "curvature")
# the tiers of opensculptboy.tiers, and the maps that a tier's entry may hold
TIER_NAMES = ("master", "high", "medium", "low")
TIER_TEXTURE_KEYS = ("normal", "head_normal", "ears_normal")
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
_KEYS = set(TEXTURE_KEYS) | set(DEFAULTS)
_KEYS |= {
    "subsurface",
    "head",
    "ears",
    "eyes",
    "reference_eyes",
    "teeth",
    "mouth",
    "tiers",
    "lashes",
    "hair",
    "brows",
}
_SUBSURFACE_KEYS = set(SUBSURFACE_DEFAULTS) | {"color"}


def _is_real(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
    occlusion: str | None = None
    # the head tile: slot -> path; the ear tile: "uvs" and slot -> path
    head: dict = dataclasses.field(default_factory=dict)
    ears: dict = dataclasses.field(default_factory=dict)
    # the eyes (EyeMaterial), or None for anny's own eyeballs
    eyes: "EyeMaterial | None" = None
    # tier name -> {"subdiv", "region", and some of TIER_TEXTURE_KEYS -> path}
    tiers: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict | None) -> "SkinMaterial":
        """The material of a card's ``material`` section; ValueError for an invalid one."""
        data = check_material(data or {})
        values = {
            k: v
            for k, v in data.items()
            if k
            not in (
                "subsurface",
                "eyes",
                "reference_eyes",
                "teeth",
                "mouth",
                "lashes",
                "hair",
                "brows",
            )
        }
        subsurface = dict(SUBSURFACE_DEFAULTS)
        subsurface.update(data.get("subsurface", {}))
        eyes = None
        if isinstance(data.get("eyes"), dict):
            eyes = EyeMaterial.from_dict(data["eyes"], data.get("reference_eyes"))
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
    if "eyes" in data and data["eyes"] is not False:
        check_eyes(data["eyes"])
    if "reference_eyes" in data:
        check_reference_eyes(data["reference_eyes"])
    if "teeth" in data:
        check_teeth(data["teeth"])
    if "lashes" in data and data["lashes"] is not False:
        check_lashes(data["lashes"])
    if "mouth" in data:
        check_mouth(data["mouth"])
    for kind in GROOM_KINDS:
        if kind in data and data[kind] is not False:
            check_groom(data[kind], kind)
    _check_tiers(data.get("tiers", {}))
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
    "iris_mm": "auto",  # the iris (limbus) diameter: mm, "baked" (the size found in the
    #                     baked eye texture) or "auto" (that size for baked eyes, else 11.7)
    "pupil_mm": "auto",  # the pupil diameter, likewise (3.5 when not baked: 3 to 4 mm in
    #                      neutral light)
    "limbal_ring": 0.6,  # 0 to 1: the darkening of the ring around the iris
    "limbal_ring_mm": 0.6,  # its width
    "sclera_color": [0.80, 0.74, 0.68],  # a slightly warm white
    "veins": 0.15,  # 0 to 1: faint red veins at the periphery of the sclera
    "seed": 0,  # of the procedural pattern
    "sclera_roughness": 0.45,  # the eyeball's own surface
    "cornea": "auto",  # glTF: True, a refracting cornea shell over a recessed iris; False,
    #                    one surface with a clear coat and the iris on its bulge (as the VRM
    #                    files have, and as a game's eyes are drawn); "auto": False when the
    #                    eye texture comes from a bake (the reference's eyes), else True
    # the cornea's IOR, roughness and radius of curvature of its front: numbers, or "auto"
    # (the values of the card's reference_eyes, which the bake measures on the reference's
    # eyes, else those of EYE_REFERENCE_DEFAULTS)
    "cornea_ior": "auto",
    "cornea_roughness": "auto",
    "cornea_radius_mm": "auto",
    "bulge": 1.0,  # the corneal bulge, a share of its anatomical height (lowered to clear
    #                the lids)
    "caruncle": True,  # the mound in the inner corner and the pink, wet skin of the socket
    "caruncle_color": [0.42, 0.15, 0.13],  # both of them (the conjunctiva)
    "highlight": 1.0,  # VRM: the strength of the matcap highlight
    "gaze": "auto",  # "auto": each eye looks where the reference's does (the gaze of the
    #                  card's reference_eyes, which the bake finds), else straight ahead;
    #                  "straight": along -Y
}
EYE_SOURCES = ("auto", "procedural", "baked")
# the cornea's values of eyes without a reference: a human cornea's IOR and front (7.8 mm),
# and a wet, smooth surface
EYE_REFERENCE_DEFAULTS = {
    "cornea_ior": 1.376,
    "cornea_roughness": 0.02,
    "cornea_radius_mm": 7.8,
}
# the ranges of the cornea's values, in material.eyes and in material.reference_eyes
_CORNEA_RANGES = {
    "cornea_ior": (1.0, 2.0),
    "cornea_roughness": (0.0, 1.0),
    "cornea_radius_mm": (5.0, 20.0),
}
# the iris and pupil sizes of eyes that do not take them from a bake (mm)
EYE_SIZES = {"iris_mm": 11.7, "pupil_mm": 3.5}
EYE_SIZE_WORDS = ("auto", "baked")


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
    iris_mm: float | str = "auto"
    pupil_mm: float | str = "auto"
    limbal_ring: float = 0.6
    limbal_ring_mm: float = 0.6
    sclera_color: list = dataclasses.field(
        default_factory=lambda: list(EYE_DEFAULTS["sclera_color"])
    )
    veins: float = 0.15
    seed: int = 0
    sclera_roughness: float = 0.45
    cornea: bool | str = "auto"
    cornea_ior: float = 1.376
    cornea_roughness: float = 0.02
    cornea_radius_mm: float = 7.8
    bulge: float = 1.0
    caruncle: bool = True
    caruncle_color: list = dataclasses.field(
        default_factory=lambda: list(EYE_DEFAULTS["caruncle_color"])
    )
    highlight: float = 1.0
    # "straight", or each eye's gaze at rest ({"L": [x, y, z], "R": ...}, Anny's frame)
    gaze: str | dict = "straight"

    @classmethod
    def from_dict(
        cls, data: dict | None, reference: dict | None = None
    ) -> "EyeMaterial":
        """
        The eyes of a card's ``material.eyes`` section; ValueError for an invalid one. The
        cornea's values that are ``"auto"`` or missing take those of ``reference`` (the card's
        ``material.reference_eyes``, measured on the reference's eyes), else
        :data:`EYE_REFERENCE_DEFAULTS`.
        """
        data = dict(check_eyes(data or {}))
        reference = check_reference_eyes(reference or {})
        for key, default in EYE_REFERENCE_DEFAULTS.items():
            if data.get(key, "auto") == "auto":
                data[key] = float(reference.get(key, default))
        gaze = data.get("gaze", "auto")
        data["gaze"] = (
            {k: list(v) for k, v in reference["gaze"].items()}
            if gaze == "auto" and reference.get("gaze")
            else "straight"
        )
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

    def sized(self, measured: dict | None = None) -> "EyeMaterial":
        """
        These eyes with numbers for ``iris_mm`` and ``pupil_mm``: ``measured`` (the sizes
        found in a baked eye texture, ``{"iris_mm", "pupil_mm"}``, or None) for ``"baked"``
        and ``"auto"``, else :data:`EYE_SIZES` for ``"auto"``. ValueError for ``"baked"``
        without measured sizes.
        """
        values = {}
        for key in ("iris_mm", "pupil_mm"):
            value = getattr(self, key)
            if value == "baked" and not measured:
                raise ValueError(
                    f"eyes {key} is 'baked', but no iris was found in a baked eye texture."
                )
            if value in EYE_SIZE_WORDS:
                value = measured[key] if measured else EYE_SIZES[key]
            values[key] = float(value)
        if values["pupil_mm"] >= values["iris_mm"] - 1.0:
            # a measured size against a given one: the given one with the other default
            for key in ("iris_mm", "pupil_mm"):
                if getattr(self, key) in EYE_SIZE_WORDS:
                    values[key] = EYE_SIZES[key]
        return dataclasses.replace(self, **values)

    def resolved(self, baked: bool) -> "EyeMaterial":
        """These eyes with a boolean ``cornea``: ``"auto"`` is a refracting cornea unless the
        eye texture comes from a bake (``baked``), whose reference draws its iris on one
        surface under a clear coat."""
        if isinstance(self.cornea, bool):
            return self
        return dataclasses.replace(self, cornea=not baked)

    def wants_measure(self) -> bool:
        """Whether a size comes from the baked eye texture when there is one."""
        return self.iris_mm in EYE_SIZE_WORDS or self.pupil_mm in EYE_SIZE_WORDS

    def extras(self) -> dict[str, Any]:
        """The settings that the glTF export keeps in the eye materials' extras."""
        return {
            "iris_mm": float(self.iris_mm),
            "pupil_mm": float(self.pupil_mm),
            "cornea_ior": float(self.cornea_ior),
            "cornea_roughness": float(self.cornea_roughness),
            "cornea_radius_mm": float(self.cornea_radius_mm),
            "sclera_roughness": float(self.sclera_roughness),
            "cornea": self.cornea is not False,
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
        ("limbal_ring", 0.0, 1.0),
        ("limbal_ring_mm", 0.0, 3.0),
        ("veins", 0.0, 1.0),
        ("sclera_roughness", 0.0, 1.0),
        ("bulge", 0.0, 1.0),
        ("highlight", 0.0, 4.0),
    ):
        if key in data:
            _number(data[key], f"eyes {key}", low, high)
    for key, (low, high) in _CORNEA_RANGES.items():
        if key in data and data[key] != "auto":
            if isinstance(data[key], str):
                raise ValueError(
                    f"eyes {key} must be a number or 'auto' (got {data[key]!r})."
                )
            _number(data[key], f"eyes {key}", low, high)
    for key, low, high in (("iris_mm", 6.0, 16.0), ("pupil_mm", 1.0, 9.0)):
        if key in data and data[key] not in EYE_SIZE_WORDS:
            if isinstance(data[key], str):
                raise ValueError(
                    f"eyes {key} must be a number of millimetres, 'auto' or 'baked' "
                    f"(got {data[key]!r})."
                )
            _number(data[key], f"eyes {key}", low, high)
    iris, pupil = data.get("iris_mm", "auto"), data.get("pupil_mm", "auto")
    if (
        iris not in EYE_SIZE_WORDS
        and pupil not in EYE_SIZE_WORDS
        and pupil >= iris - 1.0
    ):
        raise ValueError("eyes pupil_mm must be at least 1 mm below iris_mm.")
    if "cornea" in data and not (
        isinstance(data["cornea"], bool) or data["cornea"] == "auto"
    ):
        raise ValueError("eyes cornea must be true, false or 'auto'.")
    if "caruncle" in data and not isinstance(data["caruncle"], bool):
        raise ValueError("eyes caruncle must be true or false.")
    if data.get("gaze", "auto") not in ("auto", "straight"):
        raise ValueError("eyes gaze must be 'auto' or 'straight'.")
    if "seed" in data and (
        isinstance(data["seed"], bool) or not isinstance(data["seed"], int)
    ):
        raise ValueError("eyes seed must be an integer.")
    return data


def eyes_section(material: dict | None, detail: dict | None = None) -> dict | None:
    """
    The eyes section that a card's ``material`` turns on, or None for anny's own eyeballs:
    ``material.eyes`` itself; ``{}`` (the defaults) for a fitted card without it, whose bake
    measured the reference's eyes (``material.reference_eyes``) or baked its eye texture
    (``textures.eyes`` of the card's ``detail``); None for ``"eyes": false`` and for other
    cards without it.
    """
    material = material or {}
    eyes = material.get("eyes")
    if eyes is False:
        return None
    if eyes is not None:
        return eyes
    baked = ((detail or {}).get("textures") or {}).get("eyes")
    if material.get("reference_eyes") or baked:
        return {}
    return None


def check_reference_eyes(data: dict) -> dict:
    """
    Raise ValueError for a ``material.reference_eyes`` section that is not valid; returns it.
    The section holds the values of a reference's eyes that the bake measured
    (:meth:`opensculptboy.fit.reference.Reference.eye_values`): some of ``cornea_radius_mm``,
    ``cornea_roughness`` and ``cornea_ior``, as numbers, and ``gaze``, the direction in which
    each eye (``"L"``, ``"R"``) looks at rest (:func:`opensculptboy.texture.eyes.
    bake_eye_layout`), ``poses``, the file of the reference's blinks
    (:mod:`opensculptboy.eyes.poses`), and ``blinks``, the file of the exports' blink and squint
    targets that the web viewer plays (:mod:`opensculptboy.eyes.page`). The eyes' ``"auto"``
    values take
    them, and the section turns the eyes on for a card without ``material.eyes``
    (:func:`eyes_section`).
    """
    if not isinstance(data, dict):
        raise ValueError("material reference_eyes must be a JSON object.")
    keys = set(_CORNEA_RANGES) | {"gaze", "poses", "blinks"}
    unknown = sorted(set(data) - keys)
    if unknown:
        raise ValueError(
            f"Unknown reference_eyes keys {unknown}; the keys are {sorted(keys)}."
        )
    gaze = data.get("gaze", {})
    if not isinstance(gaze, dict) or set(gaze) - {"L", "R"}:
        raise ValueError("reference_eyes gaze must map 'L' and 'R' to directions.")
    for side, value in gaze.items():
        direction = value
        if (
            not isinstance(direction, (list, tuple))
            or len(direction) != 3
            or not all(_is_real(c) for c in direction)
            or float(np.linalg.norm(direction)) < 0.5
            or float(direction[1]) / float(np.linalg.norm(direction)) > -0.8
        ):
            raise ValueError(
                f"reference_eyes gaze {side} must be a direction of 3 numbers within "
                "about 37 degrees of -Y."
            )
    for key in ("poses", "blinks"):
        value = data.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"reference_eyes {key} must name a file (a string).")
    for key, value in data.items():
        if key in ("gaze", "poses", "blinks"):
            continue
        if isinstance(value, (str, bool)):
            raise ValueError(f"reference_eyes {key} must be a number (got {value!r}).")
        _number(value, f"reference_eyes {key}", *_CORNEA_RANGES[key])
    return data


# The lashes (``material.lashes``, :mod:`opensculptboy.eyes.lashes`): every key is optional.
# They come with the eyes of opensculptboy.eyes; ``"lashes": false`` leaves them out.
# Colours are linear RGB, lengths millimetres.
LASH_DEFAULTS = {
    "source": "auto",  # "auto" ("reference" with a mesh, else "generic"), "reference",
    #                    "generic"
    "mesh": None,  # the reference's lashes in the card's rest frame (a .safetensors file that
    #                the bake writes, opensculptboy.eyes.lashes.LashData), relative to the card
    "base_color": None,  # their texture (RGBA), relative to the card
    "color": [0.018, 0.011, 0.007],  # generic lashes: a dark brown
    "weight": "fine",  # generic lashes: a row of LASH_WEIGHTS, which sets the four below
    "length_mm": 6.5,  # generic upper lashes at their longest
    "lower": True,  # generic lashes on the lower lids too
    "lower_length_mm": 2.5,
    "curl": 0.45,  # how far the generic lashes curl away from the eye, a share of their length
    "density": 1.0,  # the share of the generic texture's strands
    "opacity": 1.0,  # the gain of each generic strand's alpha (above 1, a wider opaque core)
    "roughness": 1.0,  # a matt strand: a shiny card reads grey at a grazing angle
    "seed": 0,  # of the generic strands
}
LASH_SOURCES = ("auto", "reference", "generic")
# the weights of the generic lashes: the defaults of their density, lengths (mm) and opacity;
# a key of the card's section overrides its weight's value
LASH_WEIGHTS = {
    "fine": {"density": 1.0, "length_mm": 6.5, "lower_length_mm": 2.5, "opacity": 1.0},
    "medium": {
        "density": 1.5,
        "length_mm": 7.2,
        "lower_length_mm": 3.0,
        "opacity": 1.3,
    },
    "full": {"density": 2.0, "length_mm": 8.0, "lower_length_mm": 3.5, "opacity": 1.6},
}


@dataclasses.dataclass
class LashMaterial:
    """The resolved lashes of a card: every value of :data:`LASH_DEFAULTS` present, and
    ``source`` resolved to ``"reference"`` or ``"generic"``."""

    source: str = "generic"
    mesh: str | None = None
    base_color: str | None = None
    color: list = dataclasses.field(
        default_factory=lambda: list(LASH_DEFAULTS["color"])
    )
    weight: str = "fine"
    length_mm: float = 6.5
    lower: bool = True
    lower_length_mm: float = 2.5
    curl: float = 0.45
    density: float = 1.0
    opacity: float = 1.0
    roughness: float = 1.0
    seed: int = 0

    @classmethod
    def from_dict(cls, data: dict | None) -> "LashMaterial":
        data = check_lashes(dict(data or {}))
        weight = data.get("weight", LASH_DEFAULTS["weight"])
        values = {**LASH_DEFAULTS, **LASH_WEIGHTS[weight], **data}
        source = values["source"]
        if source == "auto":
            source = "reference" if values["mesh"] else "generic"
        if source == "reference" and not values["mesh"]:
            raise ValueError("lashes source 'reference' needs the mesh of the bake.")
        values["source"] = source
        values["color"] = [float(c) for c in values["color"]]
        return cls(**values)

    def extras(self) -> dict[str, Any]:
        """What the files record of the lashes."""
        out = {"source": self.source, "roughness": float(self.roughness)}
        if self.source == "generic":
            out.update(
                color=[float(c) for c in self.color],
                weight=self.weight,
                length_mm=float(self.length_mm),
                lower=bool(self.lower),
                density=float(self.density),
                opacity=float(self.opacity),
            )
        return out


def check_lashes(data: dict) -> dict:
    """Raise ValueError for a ``material.lashes`` section that is not valid; returns it."""
    if not isinstance(data, dict):
        raise ValueError("material lashes must be a JSON object or false.")
    unknown = sorted(set(data) - set(LASH_DEFAULTS))
    if unknown:
        raise ValueError(
            f"Unknown lashes keys {unknown}; the keys are {sorted(LASH_DEFAULTS)}."
        )
    if data.get("source", "auto") not in LASH_SOURCES:
        raise ValueError(
            f"lashes source must be one of {list(LASH_SOURCES)} (got {data['source']!r})."
        )
    if data.get("weight", "fine") not in LASH_WEIGHTS:
        raise ValueError(
            f"lashes weight must be one of {list(LASH_WEIGHTS)} (got {data['weight']!r})."
        )
    for key in ("mesh", "base_color"):
        if data.get(key) is not None and not isinstance(data[key], str):
            raise ValueError(f"lashes {key} must be a path.")
    if "color" in data:
        _colour(data["color"], "lashes color", 3)
    for key, low, high in (
        ("length_mm", 1.0, 20.0),
        ("lower_length_mm", 0.5, 15.0),
        ("curl", 0.0, 1.0),
        ("density", 0.05, 3.0),
        ("opacity", 0.1, 3.0),
        ("roughness", 0.0, 1.0),
    ):
        if key in data:
            _number(data[key], f"lashes {key}", low, high)
    if "lower" in data and not isinstance(data["lower"], bool):
        raise ValueError("lashes lower must be true or false.")
    if "seed" in data and (
        isinstance(data["seed"], bool) or not isinstance(data["seed"], int)
    ):
        raise ValueError("lashes seed must be an integer.")
    return data


# The teeth (``material.teeth``, :mod:`opensculptboy.teeth`): every key is optional. Colours
# are linear RGB.
TEETH_DEFAULTS = {
    "source": "auto",  # "auto" ("reference" with a mesh, else "generic"), "reference",
    #                    "generic"
    "mesh": None,  # the reference's teeth in the card's rest frame (a .safetensors file that
    #                the bake writes, opensculptboy.teeth.mesh.TeethData), relative to the card
    "color": [0.72, 0.66, 0.56],  # the enamel: a warm off-white
    "roughness": 0.35,
    "gums": True,  # generic teeth: the gums along the roots
    "gum_color": [0.45, 0.13, 0.12],
}
TEETH_SOURCES = ("auto", "reference", "generic")


@dataclasses.dataclass
class TeethMaterial:
    """The resolved teeth of a card: every value of :data:`TEETH_DEFAULTS` present, and
    ``source`` resolved to ``"reference"`` or ``"generic"``."""

    source: str = "generic"
    mesh: str | None = None
    color: list = dataclasses.field(
        default_factory=lambda: list(TEETH_DEFAULTS["color"])
    )
    roughness: float = 0.35
    gums: bool = True
    gum_color: list = dataclasses.field(
        default_factory=lambda: list(TEETH_DEFAULTS["gum_color"])
    )

    @classmethod
    def from_dict(cls, data: dict | None) -> "TeethMaterial":
        data = check_teeth(dict(data or {}))
        values = {**TEETH_DEFAULTS, **data}
        source = values["source"]
        if source == "auto":
            source = "reference" if values["mesh"] else "generic"
        if source == "reference" and not values["mesh"]:
            raise ValueError("teeth source 'reference' needs the mesh of the bake.")
        return cls(
            source=source,
            mesh=values["mesh"],
            color=[float(c) for c in values["color"]],
            roughness=float(values["roughness"]),
            gums=bool(values["gums"]),
            gum_color=[float(c) for c in values["gum_color"]],
        )

    def extras(self) -> dict[str, Any]:
        """What the files record of the teeth."""
        return {
            "source": self.source,
            "color": [float(c) for c in self.color],
            "roughness": float(self.roughness),
            "gums": bool(self.gums and self.source == "generic"),
        }


def check_teeth(data: dict) -> dict:
    """Raise ValueError for a ``material.teeth`` section that is not valid; returns it."""
    if not isinstance(data, dict):
        raise ValueError("material teeth must be a JSON object.")
    unknown = sorted(set(data) - set(TEETH_DEFAULTS))
    if unknown:
        raise ValueError(
            f"Unknown teeth keys {unknown}; the keys are {sorted(TEETH_DEFAULTS)}."
        )
    if data.get("source", "auto") not in TEETH_SOURCES:
        raise ValueError(
            f"teeth source must be one of {list(TEETH_SOURCES)} (got {data['source']!r})."
        )
    if data.get("mesh") is not None and not isinstance(data["mesh"], str):
        raise ValueError("teeth mesh must be the path of the bake's teeth file.")
    for key in ("color", "gum_color"):
        if key in data:
            _colour(data[key], f"teeth {key}", 3)
    if "roughness" in data:
        _number(data["roughness"], "teeth roughness", 0.0, 1.0)
    if "gums" in data and not isinstance(data["gums"], bool):
        raise ValueError("teeth gums must be true or false.")
    return data


# The inner mouth of a reference (opensculptboy.mouth.inner): the bake writes it, with the
# reference's converted mouth maps, as paths relative to the card.
MOUTH_DEFAULTS = {
    "mesh": None,  # the reference's inner mouth in the card's rest frame (MouthData)
    "base_color": None,  # the mouth's maps on the mesh's UVs
    "normal": None,
    "roughness": None,  # its G channel times roughness_factor
    "color": [0.62, 0.2, 0.18],  # the base colour without a map
    "roughness_factor": 1.0,
    "clearcoat": 0.6,  # the wet surface
    # the reference's mouth poses as rest-frame offsets (opensculptboy.mouth.poses)
    "poses": None,
}


@dataclasses.dataclass
class MouthMaterial:
    """The resolved inner mouth of a card: every value of :data:`MOUTH_DEFAULTS`."""

    mesh: str
    base_color: str | None = None
    normal: str | None = None
    roughness: str | None = None
    color: list = dataclasses.field(
        default_factory=lambda: list(MOUTH_DEFAULTS["color"])
    )
    roughness_factor: float = 1.0
    clearcoat: float = 0.6
    poses: str | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "MouthMaterial":
        data = check_mouth(dict(data or {}))
        if not data.get("mesh"):
            raise ValueError("material mouth needs the mesh of the bake.")
        values = {**MOUTH_DEFAULTS, **data}
        return cls(
            mesh=values["mesh"],
            base_color=values["base_color"],
            normal=values["normal"],
            roughness=values["roughness"],
            color=[float(c) for c in values["color"]],
            roughness_factor=float(values["roughness_factor"]),
            clearcoat=float(values["clearcoat"]),
            poses=values["poses"],
        )

    def textures(self) -> dict[str, str]:
        """The maps by slot (``base_color``, ``normal``, ``roughness``) that the card names."""
        return {
            slot: path
            for slot in ("base_color", "normal", "roughness")
            if (path := getattr(self, slot))
        }

    def extras(self) -> dict[str, Any]:
        """What the files record of the inner mouth."""
        return {
            "source": "reference",
            "textures": sorted(self.textures()),
            "color": [float(c) for c in self.color],
            "roughness_factor": float(self.roughness_factor),
            "clearcoat": float(self.clearcoat),
        }


def check_mouth(data: dict) -> dict:
    """Raise ValueError for a ``material.mouth`` section that is not valid; returns it."""
    if not isinstance(data, dict):
        raise ValueError("material mouth must be a JSON object.")
    unknown = sorted(set(data) - set(MOUTH_DEFAULTS))
    if unknown:
        raise ValueError(
            f"Unknown mouth keys {unknown}; the keys are {sorted(MOUTH_DEFAULTS)}."
        )
    for key in ("mesh", "base_color", "normal", "roughness", "poses"):
        if data.get(key) is not None and not isinstance(data[key], str):
            raise ValueError(f"material mouth {key} must be a path.")
    if "color" in data:
        _colour(data["color"], "mouth color", 3)
    if "roughness_factor" in data:
        _number(data["roughness_factor"], "mouth roughness_factor", 0.0, None)
    if "clearcoat" in data:
        _number(data["clearcoat"], "mouth clearcoat", 0.0, 1.0)
    return data


# The reference's hair and brows of a fitted card (``material.hair``, ``material.brows``,
# :mod:`opensculptboy.groom`), which the bake writes: ``mesh`` names the groom file and
# ``textures`` maps each of its materials to its texture files by slot (``base_color``,
# ``normal``), all relative to the card. ``false`` leaves the groom out. The brows come with the
# eyes of opensculptboy.eyes, as the lashes do.
GROOM_KINDS = ("hair", "brows")
GROOM_KEYS = ("source", "mesh", "textures")
GROOM_TEXTURE_SLOTS = ("base_color", "normal")


@dataclasses.dataclass
class GroomSection:
    """A card's ``material.hair`` or ``material.brows``: the groom file and the textures of its
    materials."""

    kind: str
    mesh: str
    textures: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict, kind: str) -> "GroomSection":
        data = check_groom(dict(data or {}), kind)
        if not data.get("mesh"):
            raise ValueError(f"material {kind} needs the mesh of the bake.")
        return cls(
            kind,
            data["mesh"],
            {k: dict(v) for k, v in (data.get("textures") or {}).items()},
        )

    def extras(self) -> dict[str, Any]:
        """What the files record of the groom."""
        return {"source": "reference", "materials": sorted(self.textures)}


def check_groom(data: dict, kind: str = "hair") -> dict:
    """Raise ValueError for a ``material.hair`` or ``material.brows`` section that is not
    valid; returns it."""
    if not isinstance(data, dict):
        raise ValueError(f"material {kind} must be a JSON object or false.")
    unknown = sorted(set(data) - set(GROOM_KEYS))
    if unknown:
        raise ValueError(
            f"Unknown {kind} keys {unknown}; the keys are {sorted(GROOM_KEYS)}."
        )
    if data.get("source", "reference") != "reference":
        raise ValueError(f"material {kind} source must be 'reference'.")
    if data.get("mesh") is not None and not isinstance(data["mesh"], str):
        raise ValueError(f"material {kind} mesh must be a path.")
    textures = data.get("textures", {})
    if not isinstance(textures, dict):
        raise ValueError(f"material {kind} textures must map materials to their files.")
    for name, slots in textures.items():
        if not isinstance(slots, dict) or set(slots) - set(GROOM_TEXTURE_SLOTS):
            raise ValueError(
                f"material {kind} textures {name!r} must map some of "
                f"{list(GROOM_TEXTURE_SLOTS)} to paths."
            )
        if not all(isinstance(v, str) for v in slots.values()):
            raise ValueError(f"material {kind} textures {name!r} must hold paths.")
    return data


def _check_tiers(tiers) -> None:
    if not isinstance(tiers, dict):
        raise ValueError("material tiers must be a JSON object keyed by tier name.")
    unknown = sorted(set(tiers) - set(TIER_NAMES))
    if unknown:
        raise ValueError(f"Unknown tiers {unknown}; the tiers are {list(TIER_NAMES)}.")
    for name, entry in tiers.items():
        if not isinstance(entry, dict):
            raise ValueError(f"material tiers {name} must be a JSON object.")
        extra = sorted(set(entry) - set(TIER_TEXTURE_KEYS) - {"subdiv", "region"})
        if extra:
            raise ValueError(
                f"Unknown keys {extra} in material tiers {name}; the keys are "
                f"subdiv, region and {list(TIER_TEXTURE_KEYS)}."
            )
        if entry.get("subdiv") not in (0, 1, 2) or isinstance(
            entry.get("subdiv"), bool
        ):
            raise ValueError(f"material tiers {name} needs subdiv 0, 1 or 2.")
        if entry.get("region", "full") not in ("full", "head"):
            raise ValueError(f"material tiers {name} region must be 'full' or 'head'.")
        for key in TIER_TEXTURE_KEYS:
            if key in entry and not isinstance(entry[key], str):
                raise ValueError(f"material tiers {name} {key} must be a path.")

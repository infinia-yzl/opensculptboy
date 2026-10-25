# OpenSculptBoy
# Apache License, Version 2.0
"""A character: the settings of the Anny model that describe one humanoid character."""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

# Schema 2 adds the optional ``detail`` field, schema 3 the optional ``material`` field and
# schema 4 the wardrobe fields (``wearables``, ``outfits``, ``active_outfit`` and
# ``wearable_overrides``). A card is written with the lowest schema that holds its fields, so
# that a card without them stays byte for byte what earlier versions wrote and read.
SCHEMA_VERSION = 4
SCHEMA_VERSIONS = (1, 2, 3, 4)
WARDROBE_FIELDS = ("wearables", "outfits", "active_outfit", "wearable_overrides")


@dataclasses.dataclass
class Character:
    """
    The settings that make one character of the Anny model.

    Every mapping takes parameter names of the model: ``phenotype`` takes
    ``model.phenotype_labels`` (0 to 1, default 0.5), ``local_changes`` takes
    ``model.local_change_labels`` (-1 to 1), ``face_shapes`` takes ``model.face_shape_labels``
    (see ``model.face_shape_ranges``) and ``facial_actions`` takes
    ``model.facial_action_labels`` (0 to 1, the ARKit blend shape names). Missing names keep
    their defaults.

    ``rig`` is ``"anny"`` (anny's default rig) or ``"anny-mouth"``, anny's rig with MakeHuman's
    jaw bone and tongue chain (:mod:`opensculptboy.mouth`), whose GLB files also open the mouth
    by a turn of the jaw and whose VRM files map the humanoid jaw.

    ``pose`` holds an optional still pose in Anny's ``local-ref`` parameters: ``"bones"`` maps
    bone labels (``model.bone_labels``) to unit quaternions (x, y, z, w), each the bone's turn
    relative to its parent, and ``"root"`` moves the root bone (x, y, z in metres, Anny's
    frame). Missing bones keep the rest pose, and an empty ``pose`` is the rest pose.

    ``detail`` holds an optional detail layer (schema 2): per-vertex offsets beyond the
    parametric model, such as the residuals of a fit to a reference mesh
    (:mod:`opensculptboy.fit`). ``"levels"`` lists ``{"subdiv": 0 | 1 | 2, "offsets": PATH}``,
    one entry per level of Catmull-Clark subdivision of anny's quad mesh, where ``PATH`` is a
    ``.safetensors`` sidecar file relative to the card (see :mod:`opensculptboy.detail`). Other
    keys (``"textures"``, ``"fit"``) describe baked textures and the fit that made the layer.
    An empty ``detail`` is no detail layer.

    ``material`` holds an optional skin material (schema 3): textures on anny's atlas, the
    roughness, the specular reflection and the subsurface scattering, and the thickness and
    curvature maps of real-time skin shading (see :mod:`opensculptboy.material`). An empty
    ``material`` keeps the exporters' own skin.

    The wardrobe fields (schema 4, :mod:`opensculptboy.wearables.wardrobe`) list what the
    character can wear: ``wearables`` holds loose pieces (``{"id", "file", "enabled"}``, each
    file a wearable package), ``outfits`` holds outfits (``{"id", "label", "pieces":
    [{"piece", "file", "enabled"}]}``), ``active_outfit`` names the outfit worn and
    ``wearable_overrides`` switches single pieces (``{"<outfit>/<piece>": {"enabled":
    false}}``).
    """

    name: str = "character"
    rig: str = "anny"
    topology: str = "anny"
    phenotypes: str = "default"
    phenotype: dict[str, float] = dataclasses.field(default_factory=dict)
    local_changes: dict[str, float] = dataclasses.field(default_factory=dict)
    face_shapes: dict[str, float] = dataclasses.field(default_factory=dict)
    facial_actions: dict[str, float] = dataclasses.field(default_factory=dict)
    pose: dict[str, Any] = dataclasses.field(default_factory=dict)
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)
    material: dict[str, Any] = dataclasses.field(default_factory=dict)
    wearables: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    outfits: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    active_outfit: str = ""
    wearable_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)

    def skin_material(self):
        """The :class:`~opensculptboy.material.SkinMaterial` of ``material``, or None (also
        for a section that holds the eyes, the values of the reference's eyes or the teeth
        alone, which keeps the exporters' own skin)."""
        if not self.material or set(self.material) <= {
            "eyes",
            "reference_eyes",
            "teeth",
            "mouth",
            "lashes",
        }:
            return None
        from opensculptboy.material import SkinMaterial

        return SkinMaterial.from_dict(self.material)

    def eye_material(self):
        """
        The :class:`~opensculptboy.material.EyeMaterial` of ``material.eyes``, or None for
        anny's own eyeballs. A fitted card turns the eyes on without the section: a card whose
        bake measured the reference's eyes (``material.reference_eyes``) or baked its eye
        texture (``detail.textures.eyes``) takes the eyes' defaults. ``"eyes": false`` keeps
        anny's own eyeballs.
        """
        from opensculptboy.material import EyeMaterial, eyes_section

        eyes = eyes_section(self.material, self.detail)
        if eyes is None:
            return None
        return EyeMaterial.from_dict(eyes, self.material.get("reference_eyes"))

    def lash_material(self):
        """The :class:`~opensculptboy.material.LashMaterial` of ``material.lashes``, or None
        for no lashes: the lashes come with the eyes of :meth:`eye_material` unless the
        section is ``false``."""
        lashes = self.material.get("lashes", {})
        if lashes is False or self.eye_material() is None:
            return None
        from opensculptboy.material import LashMaterial

        return LashMaterial.from_dict(lashes)

    def teeth_material(self):
        """The :class:`~opensculptboy.material.TeethMaterial` of ``material.teeth``, or None
        for a mouth without teeth (anny has none)."""
        if "teeth" not in self.material:
            return None
        from opensculptboy.material import TeethMaterial

        return TeethMaterial.from_dict(self.material["teeth"])

    def mouth_material(self):
        """The :class:`~opensculptboy.material.MouthMaterial` of ``material.mouth`` (the inner
        mouth of the reference, which takes the place of anny's cavity and tongue and of the
        teeth), or None."""
        if "mouth" not in self.material:
            return None
        from opensculptboy.material import MouthMaterial

        return MouthMaterial.from_dict(self.material["mouth"])

    def detail_levels(self) -> list[int]:
        """The subdivision levels of the detail layer, in increasing order."""
        return sorted(int(entry["subdiv"]) for entry in self.detail.get("levels", []))

    def build_model(self, face_shapes: bool | None = None, dtype=torch.float32):
        """
        The Anny model for this character. ``face_shapes`` loads the face-shape parameters;
        by default they load when the character sets any of them. The ``sculpt`` and
        ``sculpt-quads`` topologies build on anny's mesh with the patches of
        :mod:`opensculptboy.topology`.
        """
        import anny

        if face_shapes is None:
            face_shapes = bool(self.face_shapes)
        kwargs = dict(
            rig=self.rig,
            topology=self.topology,
            phenotypes=self.phenotypes,
            local_changes="all" if self.local_changes else "none",
            facial_actions="all",
            face_shapes="all" if face_shapes else "none",
        )
        from opensculptboy.mouth.rig import is_mouth_rig

        if self.topology.split("-")[0] == "sculpt":
            from opensculptboy.topology import sculpt_model

            model = sculpt_model(**kwargs)
        elif is_mouth_rig(self.rig):
            from opensculptboy.mouth.rig import mouth_model

            model = mouth_model(**kwargs)
        else:
            model = anny.Anny(**kwargs)
        return model.to(dtype=dtype)

    def model_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of ``model.forward`` for this character."""
        return dict(
            phenotype_kwargs=dict(self.phenotype) or None,
            local_changes_kwargs=dict(self.local_changes) or None,
            face_shape_kwargs=dict(self.face_shapes) or None,
            facial_actions=dict(self.facial_actions) or None,
        )

    def pose_parameters(self, model) -> torch.Tensor | None:
        """``local-ref`` pose parameters (1, bones, 4, 4) of ``pose`` for ``model``, or None
        when the character has no pose"""
        if not self.pose:
            return None
        import roma

        labels = list(model.bone_labels)
        bones = self.pose.get("bones", {})
        unknown = sorted(set(bones) - set(labels))
        if unknown:
            raise ValueError(f"Unknown bones in the pose: {unknown}.")
        dtype = model.template_vertices.dtype
        params = torch.eye(4, dtype=dtype).repeat(1, len(labels), 1, 1)
        for bone, quaternion in bones.items():
            q = torch.as_tensor(quaternion, dtype=torch.float64)
            params[0, labels.index(bone), :3, :3] = roma.unitquat_to_rotmat(
                q / q.norm()
            ).to(dtype)
        params[0, 0, :3, 3] = torch.as_tensor(
            self.pose.get("root", (0, 0, 0)), dtype=dtype
        )
        from opensculptboy.mouth.rig import hold_mouth_bones, is_mouth_rig

        if is_mouth_rig(self.rig):  # the mouth stays shut unless the pose moves it
            params = hold_mouth_bones(model, self, params)
        return params

    @staticmethod
    def pose_from_parameters(model, pose_parameters, tolerance: float = 1e-6) -> dict:
        """the ``pose`` field for ``local-ref`` pose parameters (bones, 4, 4) or (1, bones, 4,
        4); bones within ``tolerance`` of the rest pose are left out"""
        import roma

        params = (
            torch.as_tensor(pose_parameters).detach().double().cpu().reshape(-1, 4, 4)
        )
        quaternions = roma.rotmat_to_unitquat(params[:, :3, :3])
        bones = {}
        for label, q, R in zip(model.bone_labels, quaternions, params[:, :3, :3]):
            if (R - torch.eye(3, dtype=R.dtype)).abs().max() > tolerance:
                bones[label] = [round(float(x), 7) for x in q]
        return {
            "bones": bones,
            "root": [round(float(x), 6) for x in params[0, :3, 3]],
        }

    def has_wardrobe(self) -> bool:
        """Whether the card sets any of the wardrobe fields."""
        return any(getattr(self, name) for name in WARDROBE_FIELDS)

    def to_dict(self) -> dict[str, Any]:
        """The card as a JSON object in the lowest schema that holds it: a card without
        detail, material and wardrobe keeps schema 1, one without material and wardrobe
        schema 2, and one without wardrobe schema 3"""
        data = dataclasses.asdict(self)
        schema = 1
        if self.detail:
            schema = 2
        else:
            del data["detail"]
        if self.material:
            schema = 3
        else:
            del data["material"]
        if self.has_wardrobe():
            schema = 4
        for name in WARDROBE_FIELDS:
            if not getattr(self, name):
                del data[name]
        return {"schema": schema, **data}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Character":
        if not isinstance(data, dict):
            raise ValueError(
                f"A character card must be a JSON object (got {type(data).__name__})."
            )
        schema = data.get("schema", SCHEMA_VERSION)
        if schema not in SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported character schema {schema}; this version reads schemas "
                f"{', '.join(map(str, SCHEMA_VERSIONS))}."
            )
        if schema == 1 and "detail" in data:
            raise ValueError("The detail field needs character schema 2.")
        if schema in (1, 2) and "material" in data:
            raise ValueError("The material field needs character schema 3.")
        wardrobe = sorted(set(data) & set(WARDROBE_FIELDS))
        if schema in (1, 2, 3) and wardrobe:
            raise ValueError(f"The fields {wardrobe} need character schema 4.")
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - fields - {"schema"})
        if unknown:
            raise ValueError(
                f"Unknown character fields {unknown}; the fields are {sorted(fields)}."
            )
        character = cls(**{k: v for k, v in data.items() if k in fields})
        character.check_detail()
        if character.material:
            from opensculptboy.material import check_material

            check_material(character.material)
        if character.has_wardrobe():
            from opensculptboy.wearables.wardrobe import check_wardrobe

            check_wardrobe(character)
        return character

    def check_detail(self) -> None:
        """Raise ValueError for a ``detail`` field that is not a valid detail layer."""
        if not self.detail:
            return
        if not isinstance(self.detail, dict):
            raise ValueError("The detail field must be a JSON object.")
        levels = self.detail.get("levels", [])
        if not isinstance(levels, list):
            raise ValueError("detail.levels must be a list.")
        seen = set()
        for entry in levels:
            if not isinstance(entry, dict) or set(entry) - {
                "subdiv",
                "offsets",
                "error_mm",
            }:
                raise ValueError(
                    "Each entry of detail.levels holds 'subdiv', 'offsets' and optionally "
                    f"'error_mm' (got {entry!r})."
                )
            subdiv = entry.get("subdiv")
            if subdiv not in (0, 1, 2) or isinstance(subdiv, bool):
                raise ValueError(f"detail subdiv must be 0, 1 or 2 (got {subdiv!r}).")
            if subdiv in seen:
                raise ValueError(f"detail.levels names subdiv {subdiv} twice.")
            seen.add(subdiv)
            if not isinstance(entry.get("offsets"), str):
                raise ValueError(
                    "detail offsets must be the path of a .safetensors file."
                )

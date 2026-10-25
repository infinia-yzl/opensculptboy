# OpenSculptBoy
# Apache License, Version 2.0
"""
Blender helper: rebuild the subsurface scattering of an OpenSculptBoy GLB file's skin and the
refraction of its eyes.

Blender's glTF importer builds a Principled BSDF from the file's metallic-roughness material,
``KHR_materials_specular``, ``KHR_materials_ior``, ``KHR_texture_transform`` and the normal
map, but glTF has no ratified subsurface extension, so the export keeps the subsurface
settings in each skin material's extras under ``opensculptboy_skin`` (format
``opensculptboy/skin@1``). This script reads them (from the material's custom properties,
where the importer puts the extras, or from the file itself) and sets the Principled BSDF:

- ``subsurface.method`` -> ``subsurface_method``: ``random_walk_skin`` -> ``RANDOM_WALK_SKIN``
  (Cycles' random walk with the IOR and the anisotropy), ``random_walk`` -> ``RANDOM_WALK``,
  ``burley`` -> ``BURLEY``;
- ``subsurface.weight`` -> Subsurface Weight;
- ``subsurface.radius_mm`` -> Subsurface Radius (normalised to its largest channel) and
  Subsurface Scale (the largest channel in scene units), so that radius times scale is the
  mean free path per channel;
- ``subsurface.ior`` and ``subsurface.anisotropy`` -> Subsurface IOR and Anisotropy.

Blender 4 and later take the subsurface colour from the base colour; a ``subsurface.color``
of the file is then left aside (Blender 3 sets Subsurface Color). The thickness and curvature
maps of the extras serve real-time engines and are not used here.

The eyes of :mod:`opensculptboy.eyes` keep their settings under ``opensculptboy_eye`` (format
``opensculptboy/eye@1``) with a ``role`` per material, and :func:`apply_eye_materials` sets:

- ``cornea``: a clear refracting surface (Transmission Weight 1 at the cornea's IOR, 1.376 by
  default, and its roughness; no coat, sheen or subsurface) whose shadow rays pass through a
  Transparent BSDF (a Light Path node's Is Shadow Ray mixes them), so that Cycles lights the
  iris behind it: a refracting surface blocks the direct light of a path tracer otherwise. The
  importer's Volume Absorption of ``KHR_materials_volume`` (density 0) leaves the material
  output, since the open shell of the cornea encloses no volume. EEVEE's refraction is turned
  on for the material where the Blender version has the setting;
- ``eyeball``: the sclera's roughness, no subsurface scattering (a millimetre of it blurs the
  iris and greys the pupil) and, under a cornea, no specular reflection of its own (Specular
  IOR Level 0, as the file's ``KHR_materials_specular`` says): its rough reflection would lay a
  milky veil over the eye;
- ``conjunctiva``: the caruncle and the skin of the socket, with subsurface scattering;
- ``lashes``: the lashes keep the file's colour and alpha, and their shadow rays pass
  through them, as the cornea's do.

A wardrobe file (:mod:`opensculptboy.wearables`) holds every piece of every outfit as its own
node. Blender's importer reads every node and ignores ``KHR_node_visibility``, so
:func:`apply_wearable_visibility` hides the pieces and groups that the file's extras mark as
not shown (``opensculptboy_wearable.shown`` and ``opensculptboy_wearable_group.shown``), in
the viewport and in renders; showing them again is a click in the outliner.

The script needs nothing but Blender's own Python. Run it on a file with::

    blender --python blender_skin.py -- character.glb

which imports the file and sets its skin, its eyes and the visibility of its wearables, or,
inside Blender, ``exec`` it and call ``apply_skin_materials()`` and ``apply_eye_materials()``
after an import
(``apply_skin_materials("character.glb")`` reads the settings from the file when the importer
left out the extras).
"""

from __future__ import annotations

import json
import struct
import sys

EXTRAS_KEY = "opensculptboy_skin"
EYE_EXTRAS_KEY = "opensculptboy_eye"
METHODS = {
    "random_walk_skin": "RANDOM_WALK_SKIN",
    "random_walk": "RANDOM_WALK",
    "burley": "BURLEY",
}


def read_skin_settings(path: str, key: str = EXTRAS_KEY) -> dict:
    """The ``opensculptboy_skin`` extras (or those under ``key``) of a GLB file's materials,
    by material name."""
    with open(path, "rb") as f:
        magic, version, _ = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67 or version != 2:
            raise ValueError(f"{path} is not a glTF 2.0 binary file.")
        length, _ = struct.unpack("<II", f.read(8))
        gltf = json.loads(f.read(length))
    out = {}
    for material in gltf.get("materials", []):
        extras = (material.get("extras") or {}).get(key)
        if isinstance(extras, dict):
            out[material.get("name", "")] = extras
    return out


def principled_values(settings: dict, unit_scale: float = 1.0) -> dict:
    """
    The Principled BSDF values of a material's ``opensculptboy_skin`` extras: the subsurface
    method and the inputs by name. ``unit_scale`` is the scene's metres per unit
    (``scene.unit_settings.scale_length``).
    """
    s = settings.get("subsurface", {})
    radius = [float(x) for x in s.get("radius_mm", [1.0, 0.2, 0.1])]
    largest = max(max(radius), 1e-9)
    values = {
        "method": METHODS.get(s.get("method", "random_walk_skin"), "RANDOM_WALK_SKIN"),
        "inputs": {
            "Subsurface Weight": float(s.get("weight", 0.0)),
            "Subsurface Radius": [r / largest for r in radius],
            "Subsurface Scale": largest / 1000.0 / unit_scale,
            "Subsurface IOR": float(s.get("ior", 1.4)),
            "Subsurface Anisotropy": float(s.get("anisotropy", 0.0)),
        },
    }
    if s.get("color") is not None:
        values["color"] = [float(c) for c in s["color"]] + [1.0]
    return values


def eye_principled_values(settings: dict, unit_scale: float = 1.0) -> dict:
    """
    The Principled BSDF inputs of an eye material's ``opensculptboy_eye`` extras, by name,
    whether its shadow rays pass (``"shadow_transparent"``, the cornea) and whether the
    importer's volume goes (``"drop_volume"``, the cornea: ``KHR_materials_volume`` gives it a
    Volume Absorption node of density 0, and the open shell of the cornea encloses no volume).
    """
    role = settings.get("role")
    mm = 1.0 / 1000.0 / unit_scale
    if role == "cornea":
        inputs = {
            "Base Color": [1.0, 1.0, 1.0, 1.0],
            "Metallic": 0.0,
            "Roughness": float(settings.get("cornea_roughness", 0.02)),
            "IOR": float(settings.get("cornea_ior", 1.376)),
            "Specular IOR Level": 0.5,  # the reflection of its IOR, neither more nor less
            "Transmission Weight": 1.0,
            "Coat Weight": 0.0,
            "Sheen Weight": 0.0,
            "Subsurface Weight": 0.0,
            "Alpha": 1.0,
        }
        return {"inputs": inputs, "shadow_transparent": True, "drop_volume": True}
    if role == "eyeball":
        # no subsurface scattering: a millimetre of it blurs the iris and greys the pupil.
        # Under a cornea (the default) the eyeball reflects nothing of its own: its specular
        # layer at the sclera's roughness would lay a milky veil over the iris and the sclera
        inputs = {
            "Roughness": float(settings.get("sclera_roughness", 0.45)),
            "Subsurface Weight": 0.0,
        }
        if settings.get("cornea", True):
            inputs["Specular IOR Level"] = 0.0
            inputs["Coat Weight"] = 0.0
        return {"inputs": inputs, "shadow_transparent": False, "drop_volume": False}
    if role == "lashes":
        # the lashes keep the file's colour and alpha, and cast no shadow (a game's lash
        # material casts none either; their shadow darkens the whole eye)
        return {"inputs": {}, "shadow_transparent": True, "drop_volume": False}
    if role == "conjunctiva":
        inputs = {
            "Subsurface Weight": 0.35,
            "Subsurface Radius": [1.0, 0.3, 0.2],
            "Subsurface Scale": 2.0 * mm,
        }
        return {"inputs": inputs, "shadow_transparent": False, "drop_volume": False}
    return {"inputs": {}, "shadow_transparent": False, "drop_volume": False}


# Principled inputs of Blender 4 and later, and their names in Blender 3
BLENDER3_INPUTS = {
    "Transmission Weight": "Transmission",
    "Specular IOR Level": "Specular",
    "Coat Weight": "Clearcoat",
    "Sheen Weight": "Sheen",
    "Subsurface Weight": "Subsurface",
}


def _principled(material):
    if not material.use_nodes:
        return None
    return next(
        (
            n
            for n in material.node_tree.nodes
            if n.bl_idname == "ShaderNodeBsdfPrincipled"
        ),
        None,
    )


def apply_eye(material, settings: dict, unit_scale: float = 1.0) -> bool:
    """Set a Blender material from its eye settings; False when it has no Principled BSDF."""
    bsdf = _principled(material)
    if bsdf is None:
        return False
    values = eye_principled_values(settings, unit_scale)
    for name, value in values["inputs"].items():
        socket = bsdf.inputs.get(name)
        if socket is None and name in BLENDER3_INPUTS:  # Blender 3
            socket = bsdf.inputs.get(BLENDER3_INPUTS[name])
        if socket is not None:
            socket.default_value = value
    tree = material.node_tree
    output = next(
        (n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None
    )
    if values.get("drop_volume") and output is not None:
        for link in list(output.inputs["Volume"].links):
            tree.links.remove(link)
    if values["shadow_transparent"]:
        if output is not None:
            path = tree.nodes.new("ShaderNodeLightPath")
            clear = tree.nodes.new("ShaderNodeBsdfTransparent")
            mix = tree.nodes.new("ShaderNodeMixShader")
            tree.links.new(path.outputs["Is Shadow Ray"], mix.inputs[0])
            tree.links.new(bsdf.outputs[0], mix.inputs[1])
            tree.links.new(clear.outputs[0], mix.inputs[2])
            tree.links.new(mix.outputs[0], output.inputs["Surface"])
        for name in ("use_screen_refraction", "use_raytrace_refraction"):
            if hasattr(material, name):  # EEVEE's refraction, by Blender version
                setattr(material, name, True)
        if hasattr(material, "blend_method") and material.blend_method == "BLEND":
            material.blend_method = "HASHED"
    return True


def _settings_of(material, key: str = EXTRAS_KEY) -> dict | None:
    value = material.get(key) if hasattr(material, "get") else None
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, dict) else None


def apply_skin(material, settings: dict, unit_scale: float = 1.0) -> bool:
    """Set the Principled BSDF of a Blender material from its skin settings; False when the
    material has no Principled BSDF."""
    bsdf = _principled(material)
    if bsdf is None:
        return False
    values = principled_values(settings, unit_scale)
    bsdf.subsurface_method = values["method"]
    for name, value in values["inputs"].items():
        socket = bsdf.inputs.get(name)
        if socket is not None:
            socket.default_value = value
    if "color" in values and bsdf.inputs.get("Subsurface Color") is not None:
        bsdf.inputs["Subsurface Color"].default_value = values["color"]  # Blender 3
    return True


def apply_skin_materials(path: str | None = None) -> list[str]:
    """
    Apply the skin settings to every material of the open file that carries them (from its
    custom properties, or from the GLB file ``path`` by material name); returns the names of
    the materials set.
    """
    import bpy

    from_file = read_skin_settings(path) if path else {}
    unit_scale = bpy.context.scene.unit_settings.scale_length or 1.0
    done = []
    for material in bpy.data.materials:
        settings = _settings_of(material)
        if settings is None:
            base = material.name.split(".")[0]
            settings = from_file.get(material.name) or from_file.get(base)
        if settings and apply_skin(material, settings, unit_scale):
            done.append(material.name)
    return done


def apply_eye_materials(path: str | None = None) -> list[str]:
    """
    Apply the eye settings to every material of the open file that carries them (from its
    custom properties, or from the GLB file ``path`` by material name); returns the names of
    the materials set.
    """
    import bpy

    from_file = read_skin_settings(path, EYE_EXTRAS_KEY) if path else {}
    unit_scale = bpy.context.scene.unit_settings.scale_length or 1.0
    done = []
    for material in bpy.data.materials:
        settings = _settings_of(material, EYE_EXTRAS_KEY)
        if settings is None:
            base = material.name.split(".")[0]
            settings = from_file.get(material.name) or from_file.get(base)
        if settings and apply_eye(material, settings, unit_scale):
            done.append(material.name)
    return done


def read_wearable_visibility(path: str) -> dict[str, bool]:
    """Node name -> shown, for the wearable pieces and groups of a GLB file."""
    with open(path, "rb") as f:
        head = f.read(20)
        if len(head) < 20 or head[:4] != b"glTF":
            return {}
        length = struct.unpack("<I", head[12:16])[0]
        document = json.loads(f.read(length))
    out = {}
    for node in document.get("nodes", []):
        extras = node.get("extras") or {}
        for key in ("opensculptboy_wearable", "opensculptboy_wearable_group"):
            entry = extras.get(key)
            if isinstance(entry, dict) and "shown" in entry and node.get("name"):
                out[node["name"]] = bool(entry["shown"])
    return out


def apply_wearable_visibility(path: str) -> list[str]:
    """Hide the objects of the wearable pieces and groups that a GLB file marks as not shown;
    returns the names of the objects hidden."""
    import bpy

    shown = read_wearable_visibility(path)
    hidden = []
    for obj in bpy.data.objects:
        name = obj.name if obj.name in shown else obj.name.rsplit(".", 1)[0]
        if shown.get(name) is False:
            obj.hide_viewport = obj.hide_render = True
            try:
                obj.hide_set(True)
            except RuntimeError:  # an object outside the view layer: hidden above
                pass
            hidden.append(obj.name)
    return hidden


def main(argv: list[str]) -> None:
    import bpy

    args = argv[argv.index("--") + 1 :] if "--" in argv else []
    for path in args:
        bpy.ops.import_scene.gltf(filepath=path)
        print("skin set on", apply_skin_materials(path))
        print("eyes set on", apply_eye_materials(path))
        print("wearables hidden", apply_wearable_visibility(path))


if __name__ == "__main__" and "bpy" in sys.modules:
    main(sys.argv)

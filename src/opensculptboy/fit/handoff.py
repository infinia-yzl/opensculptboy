# OpenSculptBoy
# Apache License, Version 2.0
"""
The appearance handoff: a folder that describes a reference character to fit, as the
Dragon's Dogma 2 tools write it (draft contract ``docs/contracts/APPEARANCE-HANDOFF.md`` of
DD2DA)::

    <handoff>/
        replica.glb        skinned parts in their edited bind pose; metres, Y up, glTF axes
        textures/*.png     the textures of the parts
        appearance.json    {"landmarks": {name: [x, y, z]}, "parts": ..., "materials": ...}

:func:`load_handoff` reads the folder into :class:`Handoff`: every part of the GLB file as a
triangle mesh in Anny's frame (metres, Z up, the figure facing -Y; glTF's (x, y, z) becomes
(x, -z, y)), with its UVs and base colour and normal textures where the file has them, the
role of each part (``"skin"`` or another role such as ``"hair"``), and the landmarks in
Anny's frame. The landmarks are optional: without them the fit places the reference by its
bounding geometry (:func:`opensculptboy.fit.reference.auto_landmarks`).

Each primitive of the GLB file is one part, named after its mesh (the node's name when
``parts`` names the node) and carrying its material's name. ``parts`` may be a list of objects
or an object keyed by part name (the contract's ``{"body": {...}, "hair": {...}}``); each entry
names a mesh or node of the GLB file (``"name"``, ``"mesh"``, ``"node"`` or ``"node_name"``,
which also matches the nodes whose names start with it and an underscore, as the DD2DA tools
split a part into ``<node_name>_group<N>_...`` nodes) and may give its ``"role"`` (or
``"kind"``, ``"type"``, ``"category"``); without one, the key of an object names the role.
``materials`` entries (``{"name": ..., "role": ...}``) may give a role per material. The role of
a primitive is, in this order: its material's declared role; the role that its material's name
names (:func:`name_role`: hair, lashes, brows, beard, teeth and the mouth, eyes, nails,
clothing), so that the eyes, lashes and teeth inside a head part leave the skin; the declared
role of its part; and the role that its own name names, else skin. Only the skin takes part in
the fit, and the other parts attach later.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re

import numpy as np

# glTF (x, y, z) -> Anny (x, -z, y): the inverse of opensculptboy.export.body.ANNY_TO_GLTF.
GLTF_TO_ANNY = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
SKIN_ROLES = ("skin", "body", "head", "face", "hands", "hand", "feet", "foot")
# The roles that a part or material name names, in order: the first match wins, so that
# "eyelash" is lashes and "eyebrow" brows rather than eyes.
NAMED_ROLES = (
    ("lashes", r"lash"),
    ("brows", r"brow"),
    ("beard", r"beard|moustache|mustache"),
    ("hair", r"hair"),
    ("mouth", r"teeth|tooth|tongue|gum|mouth_?(interior|inside|cavity)"),
    ("eyes", r"eyeball|cornea|iris|pupil|sclera|(^|[^a-z])eyes?([^a-z]|$)"),
    ("nails", r"(^|[^a-z])(finger|toe)?nails?([^a-z]|$)"),
    ("clothing", r"cloth|armou?r|outfit|weapon|accessor"),
)
_NAMED_ROLES = [(role, re.compile(rx, re.IGNORECASE)) for role, rx in NAMED_ROLES]


@dataclasses.dataclass
class HandoffPart:
    """One part of the replica in Anny's frame."""

    name: str
    role: str
    vertices: np.ndarray  # (V, 3) metres, Anny's frame
    triangles: np.ndarray  # (T, 3)
    uvs: np.ndarray | None = None  # (V, 2), glTF convention (v = 0 at the top)
    base_color: np.ndarray | None = None  # (H, W, 3 or 4) uint8, glTF row order
    normal_map: np.ndarray | None = None  # (H, W, 3) uint8, tangent space
    material: str | None = None  # the glTF material's name
    # the file's vertex normals (NORMAL) in Anny's frame, unit length, or None
    normals: np.ndarray | None = None  # (V, 3)
    # the roughness: the G channel of the metallic-roughness texture times the material's
    # roughness factor, (H, W) float in 0..1, or the factor alone (a number), or None
    roughness: np.ndarray | float | None = None
    normal_scale: float = (
        1.0  # the scale of the normal texture (glTF normalTexture.scale)
    )
    # the vertices of the outer ear by side ("l", "r"), from appearance.json's masks.ear
    ear_vertices: dict | None = None

    @property
    def is_skin(self) -> bool:
        return self.role == "skin"


@dataclasses.dataclass
class Handoff:
    """A loaded appearance handoff."""

    path: pathlib.Path
    parts: list[HandoffPart]
    landmarks: dict[str, np.ndarray]  # name -> (3,) in Anny's frame, as given
    appearance: dict  # appearance.json as read

    def skin_parts(self) -> list[HandoffPart]:
        return [p for p in self.parts if p.is_skin]


def _declared_role(entry: dict | None) -> str | None:
    for key in ("role", "kind", "type", "category"):
        value = (entry or {}).get(key)
        if isinstance(value, str) and value:
            value = value.lower()
            return "skin" if value in SKIN_ROLES else value
    if (entry or {}).get("skin") is True:
        return "skin"
    return None


def _part_entries(parts) -> dict[str, dict]:
    """The ``parts`` of appearance.json by the mesh or node name they describe. An object
    keyed by part name gives its key as the role of entries that declare none."""
    if isinstance(parts, dict):
        items = []
        for k, v in parts.items():
            v = dict(v) if isinstance(v, dict) else {}
            if _declared_role(v) is None:
                v["role"] = str(k)
            items.append(dict(v, name=v.get("name", k)))
    elif isinstance(parts, list):
        items = [p if isinstance(p, dict) else {"name": str(p)} for p in parts]
    else:
        items = []
    entries = {}
    for item in items:
        for key in ("mesh", "node", "node_name", "name"):
            if isinstance(item.get(key), str) and item[key]:
                entries.setdefault(item[key], item)
    return entries


def _material_entries(materials) -> dict[str, dict]:
    """The ``materials`` of appearance.json by material name."""
    if isinstance(materials, dict):
        return {k: v for k, v in materials.items() if isinstance(v, dict)}
    if isinstance(materials, list):
        return {
            m["name"]: m
            for m in materials
            if isinstance(m, dict) and isinstance(m.get("name"), str)
        }
    return {}


def _find_entry(entries: dict, *names: str) -> dict | None:
    """The entry of a node or mesh name, exactly or as the prefix ``<entry>_`` of it."""
    for name in names:
        if name in entries:
            return entries[name]
    best = None
    for key, entry in entries.items():
        for name in names:
            if name.startswith(key + "_") and (best is None or len(key) > len(best[0])):
                best = (key, entry)
    return best[1] if best else None


def name_role(name: str | None) -> str | None:
    """The role of :data:`NAMED_ROLES` that a name names, or None."""
    for role, pattern in _NAMED_ROLES:
        if name and pattern.search(name):
            return role
    return None


def part_role(
    name: str,
    entry: dict | None = None,
    material: str | None = None,
    material_entry: dict | None = None,
) -> str:
    """``"skin"`` or another role of a part (see the module documentation)."""
    return (
        _declared_role(material_entry)
        or name_role(material)
        or _declared_role(entry)
        or name_role(name)
        or "skin"
    )


def _texture_array(image) -> np.ndarray | None:
    if image is None:
        return None
    try:
        return np.asarray(
            image.convert("RGBA" if image.mode in ("RGBA", "LA") else "RGB")
        )
    except AttributeError:
        return None


def _visual(geometry):
    """UVs (glTF convention), base colour, normal texture and roughness (see
    :class:`HandoffPart`) of a trimesh geometry."""
    visual = getattr(geometry, "visual", None)
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    base = normal = roughness = None
    if material is not None:
        base = _texture_array(getattr(material, "baseColorTexture", None))
        if base is None:
            base = _texture_array(getattr(material, "image", None))
        normal = _texture_array(getattr(material, "normalTexture", None))
        factor = getattr(material, "roughnessFactor", None)
        factor = 1.0 if factor is None else float(factor)
        texture = _texture_array(getattr(material, "metallicRoughnessTexture", None))
        if texture is not None and texture.ndim == 3:
            roughness = texture[..., 1].astype(np.float32) / 255.0 * factor
        elif getattr(material, "roughnessFactor", None) is not None:
            roughness = factor
    if uv is not None:
        uv = np.asarray(uv, dtype=np.float64).copy()
        uv[:, 1] = 1.0 - uv[:, 1]  # trimesh flips v to the OBJ convention on load
    return uv, base, normal, roughness


def _file_normals(geometry, rotation) -> np.ndarray | None:
    """The NORMAL attribute of a trimesh geometry loaded from glTF, rotated, or None."""
    cache = getattr(geometry, "_cache", None)
    normals = None
    try:
        normals = cache["vertex_normals"] if cache is not None else None
    except (KeyError, TypeError):
        normals = None
    if normals is None:
        return None
    normals = np.asarray(normals, dtype=np.float64)
    if normals.shape != np.shape(geometry.vertices):
        return None
    normals = normals @ np.asarray(rotation).T
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(length > 0, length, 1.0)


def load_glb_parts(
    path: str | pathlib.Path,
    entries: dict | None = None,
    materials: dict | None = None,
):
    """The parts of a GLB file in Anny's frame (node transforms applied, skinning not), one
    per primitive."""
    import trimesh

    entries = entries or {}
    materials = materials or {}
    scene = trimesh.load(str(path), force="scene", process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        geometry = scene.geometry[geometry_name]
        if not hasattr(geometry, "faces") or len(geometry.faces) == 0:
            continue
        vertices = np.asarray(geometry.vertices, dtype=np.float64)
        vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
        entry = _find_entry(entries, node, geometry_name)
        # trimesh names the nodes of a mesh's further primitives with a random suffix, so
        # the part takes the geometry's name unless the entry names the node itself
        name = node if node in entries else geometry_name
        material = getattr(getattr(geometry, "visual", None), "material", None)
        material = getattr(material, "name", None)
        uv, base, normal, roughness = _visual(geometry)
        parts.append(
            HandoffPart(
                name=name,
                role=part_role(name, entry, material, materials.get(material)),
                vertices=vertices @ GLTF_TO_ANNY.T,
                triangles=np.asarray(geometry.faces, dtype=np.int64),
                uvs=uv,
                base_color=base,
                normal_map=normal,
                material=material,
                normals=_file_normals(geometry, GLTF_TO_ANNY @ transform[:3, :3]),
                roughness=roughness,
            )
        )
    return parts


def load_handoff(folder: str | pathlib.Path) -> Handoff:
    """Read an appearance handoff folder (see the module documentation)."""
    folder = pathlib.Path(folder)
    if folder.is_file():
        folder = folder.parent
    glb = folder / "replica.glb"
    if not glb.exists():
        raise FileNotFoundError(f"{folder} holds no replica.glb.")
    appearance_path = folder / "appearance.json"
    appearance = (
        json.loads(appearance_path.read_text()) if appearance_path.exists() else {}
    )
    if not isinstance(appearance, dict):
        raise ValueError(f"{appearance_path} must hold a JSON object.")
    parts = load_glb_parts(
        glb,
        _part_entries(appearance.get("parts")),
        _material_entries(appearance.get("materials")),
    )
    if not any(p.is_skin for p in parts):
        raise ValueError(f"{glb} has no skin part to fit (see the parts' roles).")
    landmarks = {}
    for name, value in (appearance.get("landmarks") or {}).items():
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f"Landmark {name!r} must be [x, y, z]; got {value!r}.")
        landmarks[name] = GLTF_TO_ANNY @ point
    _attach_ear_masks(parts, (appearance.get("masks") or {}).get("ear"))
    return Handoff(folder, parts, landmarks, appearance)


def _attach_ear_masks(parts: list[HandoffPart], entries) -> None:
    """
    Attach the optional ``masks.ear`` entries of appearance.json to their parts: each entry
    names the part's mesh or node (``"mesh"``, ``"node"``; ``"primitive"`` picks a primitive
    of a mesh, ``"material"`` the primitive of that material), its ``"side"`` (``"l"`` or
    ``"r"``) and the indices of the outer ear's ``"vertices"`` in that primitive. Entries that
    name no loaded part, or indices out of range, are ignored.
    """
    if isinstance(entries, dict):
        entries = [dict(v, side=v.get("side", k)) for k, v in entries.items()]
    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        side = str(entry.get("side", "")).lower()[:1]
        vertices = entry.get("vertices")
        if side not in ("l", "r") or not isinstance(vertices, list):
            continue
        names = [
            entry.get(k) for k in ("mesh", "node") if isinstance(entry.get(k), str)
        ]
        primitive = entry.get("primitive")
        candidates = [p for p in parts if p.name in names]
        if isinstance(entry.get("material"), str):
            by_material = [p for p in candidates if p.material == entry["material"]]
            candidates = by_material or candidates
        elif isinstance(primitive, int) and 0 <= primitive < len(candidates):
            candidates = [candidates[primitive]]
        if not candidates:
            continue
        part = candidates[0]
        index = np.asarray([v for v in vertices if isinstance(v, int)], dtype=np.int64)
        index = index[(index >= 0) & (index < len(part.vertices))]
        if len(index):
            part.ear_vertices = dict(part.ear_vertices or {}, **{side: index})


def write_handoff(
    folder: str | pathlib.Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
    landmarks: dict[str, np.ndarray] | None = None,
    name: str = "body",
) -> pathlib.Path:
    """
    Write a one-part handoff folder from a mesh in Anny's frame (for tests and examples): the
    mesh as replica.glb in glTF's frame and the landmarks in appearance.json.
    """
    import trimesh

    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    gltf = np.asarray(vertices, dtype=np.float64) @ GLTF_TO_ANNY  # inverse rotation
    mesh = trimesh.Trimesh(gltf, np.asarray(triangles), process=False)
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name=name, geom_name=name)
    scene.export(str(folder / "replica.glb"))
    appearance = {
        "landmarks": {
            k: [float(x) for x in np.asarray(v) @ GLTF_TO_ANNY]
            for k, v in (landmarks or {}).items()
        },
        "parts": [{"name": name, "role": "skin"}],
        "materials": {},
    }
    (folder / "appearance.json").write_text(json.dumps(appearance, indent=2) + "\n")
    return folder

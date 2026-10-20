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

``parts`` may be a list of objects or an object keyed by part name; each entry names a mesh or
node of the GLB file (``"name"``, ``"mesh"`` or ``"node"``) and may give its ``"role"`` (or
``"kind"``, ``"type"``, ``"category"``). Parts without a role are skin unless their name says
otherwise (hair, lashes, brows, beard, teeth, tongue, eyeballs, clothing): only the skin takes
part in the fit, and the other parts attach later.
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
NOT_SKIN = re.compile(
    r"hair|lash|brow|beard|moustache|mustache|teeth|tooth|tongue|eyeball|cornea|iris|"
    r"pupil|(^|[^a-z])eyes?([^a-z]|$)|cloth|armou?r|outfit|weapon|accessor",
    re.IGNORECASE,
)


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


def _part_entries(parts) -> dict[str, dict]:
    """The ``parts`` of appearance.json by the mesh or node name they describe."""
    if isinstance(parts, dict):
        items = [dict(v, name=v.get("name", k)) for k, v in parts.items()]
    elif isinstance(parts, list):
        items = [p if isinstance(p, dict) else {"name": str(p)} for p in parts]
    else:
        items = []
    entries = {}
    for item in items:
        for key in ("mesh", "node", "name"):
            if isinstance(item.get(key), str):
                entries[item[key]] = item
    return entries


def part_role(name: str, entry: dict | None = None) -> str:
    """``"skin"`` or the declared role of a part; a part without a role is skin unless its
    name names another kind of part."""
    entry = entry or {}
    for key in ("role", "kind", "type", "category"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            value = value.lower()
            return "skin" if value in SKIN_ROLES else value
    if entry.get("skin") is True:
        return "skin"
    return "other" if NOT_SKIN.search(name) else "skin"


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
    """UVs (glTF convention), base colour and normal textures of a trimesh geometry."""
    visual = getattr(geometry, "visual", None)
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    base = normal = None
    if material is not None:
        base = _texture_array(getattr(material, "baseColorTexture", None))
        if base is None:
            base = _texture_array(getattr(material, "image", None))
        normal = _texture_array(getattr(material, "normalTexture", None))
    if uv is not None:
        uv = np.asarray(uv, dtype=np.float64).copy()
        uv[:, 1] = 1.0 - uv[:, 1]  # trimesh flips v to the OBJ convention on load
    return uv, base, normal


def load_glb_parts(path: str | pathlib.Path, entries: dict | None = None):
    """The parts of a GLB file in Anny's frame (node transforms applied, skinning not)."""
    import trimesh

    entries = entries or {}
    scene = trimesh.load(str(path), force="scene", process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        geometry = scene.geometry[geometry_name]
        if not hasattr(geometry, "faces") or len(geometry.faces) == 0:
            continue
        vertices = np.asarray(geometry.vertices, dtype=np.float64)
        vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
        entry = entries.get(node) or entries.get(geometry_name)
        name = (
            node if node in entries or geometry_name not in entries else geometry_name
        )
        uv, base, normal = _visual(geometry)
        parts.append(
            HandoffPart(
                name=name,
                role=part_role(name, entry),
                vertices=vertices @ GLTF_TO_ANNY.T,
                triangles=np.asarray(geometry.faces, dtype=np.int64),
                uvs=uv,
                base_color=base,
                normal_map=normal,
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
    parts = load_glb_parts(glb, _part_entries(appearance.get("parts")))
    if not any(p.is_skin for p in parts):
        raise ValueError(f"{glb} has no skin part to fit (see the parts' roles).")
    landmarks = {}
    for name, value in (appearance.get("landmarks") or {}).items():
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f"Landmark {name!r} must be [x, y, z]; got {value!r}.")
        landmarks[name] = GLTF_TO_ANNY @ point
    return Handoff(folder, parts, landmarks, appearance)


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

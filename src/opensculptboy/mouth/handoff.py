# OpenSculptBoy
# Apache License, Version 2.0
"""
The inner mouth of an appearance handoff (docs/reference-fit.md, "Mouth").

``appearance.json`` may describe the reference's inner mouth under ``mouth``: one entry per
piece in ``pieces`` (``role``: ``gums_upper``, ``gums_lower``, ``palate``, ``tongue``,
``mouth_floor``, ``throat`` or ``tooth_<arch>_<side><n>``), each naming its primitive of
``replica.glb`` (``mesh`` and ``primitive``) and, for a piece that is part of a primitive, its
``vertices``; and the mouth's material under ``materials``, whose ``converted`` maps
(``base_color``, ``normal``, ``roughness``, paths in the handoff folder) draw it.

:func:`mouth_parts` reads each piece straight from the GLB file, with the skin weights that
the trimesh loader leaves out: a :class:`~opensculptboy.fit.handoff.HandoffPart` of role
``"mouth"`` per piece, in Anny's frame, whose ``mouth_role`` names its piece and whose
``joints``, ``joint_slots`` and ``joint_weights`` give the joint names of its skin and each
vertex's strongest joints. :func:`mouth_textures` gives the paths of the converted maps.
"""

from __future__ import annotations

import json
import pathlib
import struct

import numpy as np

# glTF accessor component types and their element counts
_COMPONENTS = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
# The pieces of the contract and the part of the exported mouth that draws each
PIECE_PARTS = {
    "gums_upper": "gums",
    "gums_lower": "gums",
    "palate": "palate",
    "tongue": "tongue",
    "mouth_floor": "floor",
    "throat": "throat",
}
# The pieces that follow the lower jaw (the rest follow the head)
LOWER_PIECES = ("gums_lower", "tongue", "mouth_floor")


def piece_part(role: str) -> str | None:
    """The exported part of a piece's role (``"teeth"`` for the separate teeth), or None for
    a role this reader does not know."""
    if role.startswith("tooth_"):
        return "teeth"
    return PIECE_PARTS.get(role)


def piece_arch(role: str) -> int:
    """1 for a piece that follows the lower jaw, 0 for one that follows the head."""
    if role.startswith("tooth_"):
        return 1 if role.startswith("tooth_lower") else 0
    return 1 if role in LOWER_PIECES else 0


def read_glb(path: str | pathlib.Path) -> tuple[dict, bytes]:
    """The JSON and the binary chunk of a GLB file."""
    data = pathlib.Path(path).read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"{path} is not a GLB file.")
    offset, gltf, binary = 12, None, b""
    while offset + 8 <= len(data):
        length, kind = struct.unpack("<II", data[offset : offset + 8])
        chunk = data[offset + 8 : offset + 8 + length]
        if kind == 0x4E4F534A:
            gltf = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = chunk
        offset += 8 + length
    if gltf is None:
        raise ValueError(f"{path} has no JSON chunk.")
    return gltf, binary


def accessor(gltf: dict, binary: bytes, index: int) -> np.ndarray:
    """An accessor's values as an (N, k) array (float for normalized integers)."""
    a = gltf["accessors"][index]
    dtype = np.dtype(_COMPONENTS[a["componentType"]])
    k = _SIZES[a["type"]]
    count = a["count"]
    if "bufferView" not in a:
        return np.zeros((count, k), dtype)
    view = gltf["bufferViews"][a["bufferView"]]
    start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = view.get("byteStride") or k * dtype.itemsize
    raw = np.frombuffer(
        binary, np.uint8, count=(count - 1) * stride + k * dtype.itemsize, offset=start
    )
    rows = np.lib.stride_tricks.as_strided(
        raw, (count, k * dtype.itemsize), (stride, 1)
    )
    out = np.ascontiguousarray(rows).view(dtype).reshape(count, k)
    if a.get("normalized"):
        out = out.astype(np.float64) / np.iinfo(dtype).max
    return out


def _world_matrices(gltf: dict) -> list[np.ndarray]:
    """The world matrix of each node (the scene's node hierarchy, TRS or matrix)."""
    nodes = gltf.get("nodes", [])
    local = []
    for node in nodes:
        if "matrix" in node:
            local.append(np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T)
            continue
        m = np.eye(4)
        x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
        m[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
        m[:3, :3] *= np.asarray(node.get("scale", [1.0, 1.0, 1.0]))[None, :]
        m[:3, 3] = node.get("translation", [0.0, 0.0, 0.0])
        local.append(m)
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    world: list = [None] * len(nodes)

    def resolve(i):
        if world[i] is None:
            world[i] = local[i] if i not in parent else resolve(parent[i]) @ local[i]
        return world[i]

    return [resolve(i) for i in range(len(nodes))]


def _mesh_node(gltf: dict, mesh: int) -> int | None:
    for i, node in enumerate(gltf.get("nodes", [])):
        if node.get("mesh") == mesh:
            return i
    return None


def _mesh_index(gltf: dict, name: str) -> int | None:
    for i, mesh in enumerate(gltf.get("meshes", [])):
        if mesh.get("name") == name:
            return i
    for i, node in enumerate(gltf.get("nodes", [])):
        if node.get("name") == name and "mesh" in node:
            return node["mesh"]
    return None


def mouth_parts(folder: str | pathlib.Path, appearance: dict, max_joints: int = 4):
    """
    The pieces of ``appearance["mouth"]`` as parts of role ``"mouth"`` (see the module
    documentation), or an empty list without the section. Pieces whose primitive the GLB file
    lacks, or whose role this reader does not know, are left out.
    """
    from opensculptboy.fit.handoff import GLTF_TO_ANNY, HandoffPart

    pieces = ((appearance or {}).get("mouth") or {}).get("pieces") or []
    if not pieces:
        return []
    gltf, binary = read_glb(pathlib.Path(folder) / "replica.glb")
    world = _world_matrices(gltf)
    parts = []
    for entry in pieces:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role", ""))
        if piece_part(role) is None:
            continue
        mesh = _mesh_index(gltf, str(entry.get("mesh") or entry.get("node") or ""))
        if mesh is None:
            continue
        primitives = gltf["meshes"][mesh]["primitives"]
        index = entry.get("primitive", 0)
        if not isinstance(index, int) or not 0 <= index < len(primitives):
            continue
        prim = primitives[index]
        attributes = prim["attributes"]
        positions = accessor(gltf, binary, attributes["POSITION"]).astype(np.float64)
        n = len(positions)
        triangles = (
            accessor(gltf, binary, prim["indices"]).reshape(-1, 3).astype(np.int64)
            if "indices" in prim
            else np.arange(n).reshape(-1, 3)
        )
        node = _mesh_node(gltf, mesh)
        matrix = world[node] if node is not None else np.eye(4)
        positions = positions @ matrix[:3, :3].T + matrix[:3, 3]
        normals = None
        if "NORMAL" in attributes:
            normals = accessor(gltf, binary, attributes["NORMAL"]).astype(np.float64)
            normals = normals @ np.linalg.inv(matrix[:3, :3])
        uvs = None
        if "TEXCOORD_0" in attributes:
            uvs = accessor(gltf, binary, attributes["TEXCOORD_0"]).astype(np.float64)
        joints = slots = weights = None
        skin = None if node is None else gltf["nodes"][node].get("skin")
        if skin is not None and "JOINTS_0" in attributes and "WEIGHTS_0" in attributes:
            names = [
                gltf["nodes"][j].get("name", f"joint{j}")
                for j in gltf["skins"][skin]["joints"]
            ]
            j_all = [accessor(gltf, binary, attributes["JOINTS_0"]).astype(np.int64)]
            w_all = [accessor(gltf, binary, attributes["WEIGHTS_0"]).astype(np.float64)]
            if "JOINTS_1" in attributes and "WEIGHTS_1" in attributes:
                j_all.append(
                    accessor(gltf, binary, attributes["JOINTS_1"]).astype(np.int64)
                )
                w_all.append(
                    accessor(gltf, binary, attributes["WEIGHTS_1"]).astype(np.float64)
                )
            j_all, w_all = np.concatenate(j_all, 1), np.concatenate(w_all, 1)
            order = np.argsort(-w_all, axis=1, kind="stable")[:, :max_joints]
            slots = np.take_along_axis(j_all, order, 1)
            weights = np.take_along_axis(w_all, order, 1)
            total = weights.sum(1, keepdims=True)
            weights = np.where(
                total > 0, weights / np.where(total > 0, total, 1.0), 0.0
            )
            joints = names
        rows = entry.get("vertices")
        if isinstance(rows, list) and rows:
            rows = np.unique(
                np.asarray([r for r in rows if isinstance(r, int)], np.int64)
            )
            rows = rows[(rows >= 0) & (rows < n)]
            keep = np.zeros(n, bool)
            keep[rows] = True
            triangles = triangles[keep[triangles].all(1)]
            remap = np.full(n, -1)
            remap[rows] = np.arange(len(rows))
            triangles = remap[triangles]
        else:
            rows = np.arange(n)
        if len(triangles) == 0:
            continue

        def take(a):
            return None if a is None else a[rows]

        vertices = positions[rows] @ GLTF_TO_ANNY.T
        part_normals = take(normals)
        if part_normals is not None:
            part_normals = part_normals @ GLTF_TO_ANNY.T
            length = np.linalg.norm(part_normals, axis=1, keepdims=True)
            part_normals = part_normals / np.where(length > 0, length, 1.0)
        parts.append(
            HandoffPart(
                name=f"mouth/{role}",
                role="mouth",
                vertices=vertices,
                triangles=triangles,
                uvs=take(uvs),
                material=entry.get("material"),
                normals=part_normals,
                mouth_role=role,
                joints=joints,
                joint_slots=take(slots),
                joint_weights=take(weights),
            )
        )
    return parts


def mouth_textures(
    folder: str | pathlib.Path, appearance: dict
) -> dict[str, pathlib.Path]:
    """The converted maps of the mouth's material by slot (``base_color``, ``normal``,
    ``roughness``: the G channel), as paths in the handoff folder that exist; empty
    without them."""
    folder = pathlib.Path(folder)
    out = {}
    for material in ((appearance or {}).get("mouth") or {}).get("materials") or []:
        converted = (material or {}).get("converted") or {}
        for slot in ("base_color", "normal", "roughness"):
            value = converted.get(slot)
            if not isinstance(value, str) or slot in out:
                continue
            path = folder / value.split(" (")[0].strip()
            if path.is_file():
                out[slot] = path
    return out

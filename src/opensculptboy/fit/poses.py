# OpenSculptBoy
# Apache License, Version 2.0
"""
A reference posed by its own rig: the skinned primitives of a handoff's ``replica.glb`` under
the joint deltas of a pose of ``appearance.json``'s ``mouth.poses`` (the contract in
docs/mouth.md, "Reference poses").

A pose names joints of the file's skins with a ``translation`` and a ``rotation`` (a quaternion
x, y, z, w). Each joint's posed local transform is its own: its translation plus the delta's,
its rotation times the delta's. The skins then move the vertices by linear blend skinning with
their inverse bind matrices, as a glTF viewer does. :func:`posed_parts` gives the posed
vertices of every part of a loaded handoff, in Anny's frame and in the order of
``Handoff.parts``.
"""

from __future__ import annotations

import json
import pathlib
import struct

import numpy as np

from opensculptboy.fit.handoff import GLTF_TO_ANNY

_COMPONENT = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _quaternion_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class GlbRig:
    """The nodes, skins and skinned primitives of a GLB file."""

    def __init__(self, path: str | pathlib.Path):
        data = pathlib.Path(path).read_bytes()
        if data[:4] != b"glTF":
            raise ValueError(f"{path} is not a GLB file.")
        length = struct.unpack("<I", data[12:16])[0]
        self.document = json.loads(data[20 : 20 + length])
        rest = 20 + length
        self.binary = b""
        if len(data) > rest + 8:
            size = struct.unpack("<I", data[rest : rest + 4])[0]
            self.binary = data[rest + 8 : rest + 8 + size]
        self.nodes = self.document.get("nodes", [])
        self.parent = {}
        for i, node in enumerate(self.nodes):
            for child in node.get("children", []):
                self.parent[child] = i
        self.names = {n.get("name"): i for i, n in enumerate(self.nodes) if n.get("name")}

    def accessor(self, index: int) -> np.ndarray:
        a = self.document["accessors"][index]
        view = self.document["bufferViews"][a["bufferView"]]
        dtype = np.dtype(_COMPONENT[a["componentType"]])
        width = _WIDTH[a["type"]]
        offset = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = view.get("byteStride") or dtype.itemsize * width
        count = a["count"]
        raw = np.frombuffer(
            self.binary, np.uint8, stride * (count - 1) + dtype.itemsize * width, offset
        )
        rows = np.lib.stride_tricks.as_strided(
            raw, (count, dtype.itemsize * width), (stride, 1)
        )
        out = np.ascontiguousarray(rows).view(dtype).reshape(count, width)
        if a.get("normalized") and dtype.kind in "iu":
            out = out.astype(np.float64) / np.iinfo(dtype).max
        return out

    def local(self, index: int, delta: dict | None = None) -> np.ndarray:
        """The local matrix (4, 4) of a node, with a pose's delta applied."""
        node = self.nodes[index]
        M = np.eye(4)
        if "matrix" in node:
            M = np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
        else:
            R = _quaternion_matrix(node.get("rotation", [0, 0, 0, 1]))
            M[:3, :3] = R @ np.diag(node.get("scale", [1, 1, 1]))
            M[:3, 3] = node.get("translation", [0, 0, 0])
        if delta:
            M[:3, 3] = M[:3, 3] + np.asarray(delta.get("translation", [0, 0, 0]))
            M[:3, :3] = M[:3, :3] @ _quaternion_matrix(
                delta.get("rotation", [0, 0, 0, 1])
            )
        return M

    def world(self, deltas: dict | None = None) -> list[np.ndarray]:
        """The world matrix of every node under a pose's deltas (by joint name)."""
        deltas = deltas or {}
        out: list[np.ndarray | None] = [None] * len(self.nodes)

        def solve(i):
            if out[i] is None:
                name = self.nodes[i].get("name")
                local = self.local(i, deltas.get(name))
                parent = self.parent.get(i)
                out[i] = local if parent is None else solve(parent) @ local
            return out[i]

        for i in range(len(self.nodes)):
            solve(i)
        return out

    def skinned_primitives(self):
        """(mesh name, material name, primitive index, skin index, vertices (V, 3) in the
        file's frame, joints (V, K), weights (V, K)) of every skinned primitive."""
        materials = self.document.get("materials", [])
        meshes = self.document.get("meshes", [])
        for node in self.nodes:
            if "mesh" not in node or "skin" not in node:
                continue
            mesh = meshes[node["mesh"]]
            for k, primitive in enumerate(mesh["primitives"]):
                attributes = primitive["attributes"]
                if "JOINTS_0" not in attributes:
                    continue
                joints = [self.accessor(attributes["JOINTS_0"]).astype(np.int64)]
                weights = [self.accessor(attributes["WEIGHTS_0"]).astype(np.float64)]
                if "JOINTS_1" in attributes:
                    joints.append(self.accessor(attributes["JOINTS_1"]).astype(np.int64))
                    weights.append(
                        self.accessor(attributes["WEIGHTS_1"]).astype(np.float64)
                    )
                material = primitive.get("material")
                yield (
                    mesh.get("name"),
                    materials[material].get("name") if material is not None else None,
                    k,
                    node["skin"],
                    self.accessor(attributes["POSITION"]).astype(np.float64),
                    np.concatenate(joints, 1),
                    np.concatenate(weights, 1),
                )

    def skin_matrices(self, skin: int, world: list[np.ndarray]) -> np.ndarray:
        """The skinning matrices (J, 4, 4) of a skin: world times inverse bind."""
        entry = self.document["skins"][skin]
        joints = entry["joints"]
        if "inverseBindMatrices" in entry:
            inverse = self.accessor(entry["inverseBindMatrices"]).astype(np.float64)
            inverse = inverse.reshape(-1, 4, 4).transpose(0, 2, 1)
        else:
            inverse = np.repeat(np.eye(4)[None], len(joints), 0)
        return np.stack([world[j] for j in joints]) @ inverse

    def pose(self, deltas: dict | None = None):
        """For every skinned primitive: (mesh, material, primitive, rest (V, 3), posed
        (V, 3)), both by the skins (the rest pose skins the bind pose) in the file's frame."""
        rest_world = self.world()
        posed_world = self.world(deltas)
        for mesh, material, k, skin, vertices, joints, weights in (
            self.skinned_primitives()
        ):
            weights = weights / np.maximum(weights.sum(1, keepdims=True), 1e-12)
            out = []
            for world in (rest_world, posed_world):
                S = self.skin_matrices(skin, world)
                M = np.einsum("vk,vkij->vij", weights, S[joints])
                out.append(np.einsum("vij,vj->vi", M[:, :3, :3], vertices) + M[:, :3, 3])
            yield mesh, material, k, out[0], out[1]


def mouth_poses(appearance: dict) -> dict:
    """The poses of a handoff's ``mouth.poses`` (name to joint deltas), those marked known
    only; empty when it has none."""
    poses = (appearance.get("mouth") or {}).get("poses") or {}
    out = {}
    for name, pose in poses.items():
        if isinstance(pose, dict) and pose.get("known", True) is not False:
            joints = pose.get("joints") or {}
            if isinstance(joints, dict):
                out[name] = joints
    return out


def posed_parts(handoff, deltas: dict, tolerance: float = 1e-5) -> list:
    """
    The vertices (V, 3) of each part of a loaded handoff (``Handoff.parts`` order) under a
    pose's joint deltas, in Anny's frame; None for a part that no skinned primitive of the
    file matches (by vertex count and the rest positions within ``tolerance`` metres).
    """
    rig = GlbRig(handoff.path / "replica.glb")
    posed = [
        (rest @ GLTF_TO_ANNY.T, moved @ GLTF_TO_ANNY.T)
        for _, _, _, rest, moved in rig.pose(deltas)
    ]
    out = []
    for part in handoff.parts:
        match = None
        for rest, moved in posed:
            if len(rest) == len(part.vertices) and (
                np.abs(rest - part.vertices).max() < tolerance
            ):
                match = moved
                break
        out.append(match)
    return out

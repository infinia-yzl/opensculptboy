# OpenSculptBoy
# Apache License, Version 2.0
"""
The hair and the brows of an appearance handoff, read with what the trimesh loader leaves out.

:func:`groom_parts` reads every primitive of ``replica.glb`` whose role
(:func:`opensculptboy.fit.handoff.part_role`) is one of :data:`GROOM_ROLES` straight from the
GLB file: a :class:`~opensculptboy.fit.handoff.HandoffPart` per primitive, in Anny's frame, with
its UVs, its base colour and normal textures, its vertex colours (``COLOR_0``), its material's
alpha mode, cutoff and sides, and its skin: the joint names, each vertex's strongest joints
and weights, and the rest position and parent of each joint of the skin. The primitives that
``hidden`` of appearance.json names (parts that the game probably does not show) are left out.
"""

from __future__ import annotations

import io
import pathlib

import numpy as np

# the roles of the parts that a fitted card carries as grooms (opensculptboy.groom)
GROOM_ROLES = ("hair", "brows")


def _hidden(appearance: dict) -> set:
    """The (mesh, primitive) pairs that ``hidden`` of appearance.json names."""
    out = set()
    hidden = (appearance or {}).get("hidden")
    for entry in hidden.values() if isinstance(hidden, dict) else []:
        primitives = entry.get("primitives") if isinstance(entry, dict) else None
        for p in primitives if isinstance(primitives, list) else []:
            if isinstance(p, dict) and isinstance(p.get("mesh"), str):
                out.add((p["mesh"], int(p.get("primitive", 0))))
    return out


def _image(data: bytes | None) -> np.ndarray | None:
    if data is None:
        return None
    from PIL import Image

    image = Image.open(io.BytesIO(data))
    return np.asarray(image.convert("RGBA" if "A" in image.getbands() else "RGB"))


def _skin(gltf, binary, node: int, attributes: dict, world, max_joints: int):
    """(joint names, (V, k) slots, (V, k) weights, positions by name, parents by name) of a
    skinned node's primitive, or Nones."""
    from opensculptboy.fit.handoff import GLTF_TO_ANNY
    from opensculptboy.mouth.handoff import accessor

    skin = gltf["nodes"][node].get("skin")
    if skin is None or "JOINTS_0" not in attributes or "WEIGHTS_0" not in attributes:
        return None, None, None, None, None
    joints = gltf["skins"][skin]["joints"]
    names = [gltf["nodes"][j].get("name", f"joint{j}") for j in joints]
    j_all, w_all = [], []
    for s in range(4):
        if f"JOINTS_{s}" not in attributes or f"WEIGHTS_{s}" not in attributes:
            break
        j_all.append(accessor(gltf, binary, attributes[f"JOINTS_{s}"]).astype(np.int64))
        w_all.append(
            accessor(gltf, binary, attributes[f"WEIGHTS_{s}"]).astype(np.float64)
        )
    j_all, w_all = np.concatenate(j_all, 1), np.concatenate(w_all, 1)
    order = np.argsort(-w_all, axis=1, kind="stable")[:, :max_joints]
    slots = np.take_along_axis(j_all, order, 1)
    weights = np.take_along_axis(w_all, order, 1)
    total = weights.sum(1, keepdims=True)
    weights = np.where(total > 0, weights / np.where(total > 0, total, 1.0), 0.0)
    parent_of = {
        c: i for i, n in enumerate(gltf.get("nodes", [])) for c in n.get("children", [])
    }
    # the joints' bind pose: the inverse of the inverse bind matrices, else the nodes' pose
    bind = [world[j] for j in joints]
    ibm = gltf["skins"][skin].get("inverseBindMatrices")
    if ibm is not None:
        m = accessor(gltf, binary, ibm).astype(np.float64).reshape(-1, 4, 4)
        bind = [np.linalg.inv(x.T) for x in m]
    positions, parents = {}, {}
    for j, name, matrix in zip(joints, names, bind):
        positions[name] = GLTF_TO_ANNY @ matrix[:3, 3]
        p = parent_of.get(j)
        parents[name] = None if p is None else gltf["nodes"][p].get("name")
    return names, slots, weights, positions, parents


def groom_parts(folder: str | pathlib.Path, appearance: dict, max_joints: int = 4):
    """The primitives of the hair and the brows (see the module documentation), or an empty
    list when the handoff has none."""
    from opensculptboy.fit.handoff import (
        GLTF_TO_ANNY,
        HandoffPart,
        _find_entry,
        _material_entries,
        _part_entries,
        part_role,
    )
    from opensculptboy.mouth.handoff import _world_matrices, accessor, read_glb
    from opensculptboy.wearables.gltf import read_materials

    path = pathlib.Path(folder) / "replica.glb"
    gltf, binary = read_glb(path)
    entries = _part_entries((appearance or {}).get("parts"))
    material_entries = _material_entries((appearance or {}).get("materials"))
    hidden = _hidden(appearance)
    materials, index = read_materials(gltf, binary, path.parent)
    world = _world_matrices(gltf)
    parts = []
    for n, node in enumerate(gltf.get("nodes", [])):
        if "mesh" not in node:
            continue
        mesh = gltf["meshes"][node["mesh"]]
        mesh_name = mesh.get("name") or node.get("name") or f"mesh{node['mesh']}"
        node_name = node.get("name") or mesh_name
        entry = _find_entry(entries, node_name, mesh_name)
        primitives = mesh.get("primitives", [])
        for k, prim in enumerate(primitives):
            if prim.get("mode", 4) != 4 or (mesh_name, k) in hidden:
                continue
            m = prim.get("material")
            material = materials[index[m]] if m is not None else None
            name = material.name if material is not None else None
            label = node_name if node_name in entries else mesh_name
            role = part_role(label, entry, name, material_entries.get(name))
            if role not in GROOM_ROLES:
                continue
            attributes = prim["attributes"]
            positions = accessor(gltf, binary, attributes["POSITION"]).astype(
                np.float64
            )
            count = len(positions)
            triangles = (
                accessor(gltf, binary, prim["indices"]).reshape(-1, 3).astype(np.int64)
                if "indices" in prim
                else np.arange(count).reshape(-1, 3)
            )
            if len(triangles) == 0:
                continue
            matrix = world[n]
            positions = positions @ matrix[:3, :3].T + matrix[:3, 3]
            normals = None
            if "NORMAL" in attributes:
                normals = accessor(gltf, binary, attributes["NORMAL"]).astype(
                    np.float64
                )
                normals = normals @ np.linalg.inv(matrix[:3, :3])
                normals = normals @ GLTF_TO_ANNY.T
                length = np.linalg.norm(normals, axis=1, keepdims=True)
                normals = normals / np.where(length > 0, length, 1.0)
            uvs = None
            if "TEXCOORD_0" in attributes:
                uvs = accessor(gltf, binary, attributes["TEXCOORD_0"]).astype(
                    np.float64
                )
            colors = None
            if "COLOR_0" in attributes:
                colors = accessor(gltf, binary, attributes["COLOR_0"]).astype(
                    np.float64
                )
                if colors.shape[1] == 3:
                    colors = np.c_[colors, np.ones(count)]
            names, slots, weights, joint_positions, joint_parents = _skin(
                gltf, binary, n, attributes, world, max_joints
            )
            factor = None
            if material is not None:
                factor = tuple(
                    float(c) for c in (tuple(material.base_color) + (1.0,))[:4]
                )
            textures = material.textures if material is not None else {}
            parts.append(
                HandoffPart(
                    name=label if len(primitives) == 1 else f"{label}_{k}",
                    role=role,
                    vertices=positions @ GLTF_TO_ANNY.T,
                    triangles=triangles,
                    uvs=uvs,
                    base_color=_image(textures.get("base_color")),
                    normal_map=_image(textures.get("normal")),
                    material=name,
                    normals=normals,
                    roughness=None if material is None else float(material.roughness),
                    normal_scale=1.0 if material is None else material.normal_scale,
                    base_color_factor=factor,
                    joints=names,
                    joint_slots=slots,
                    joint_weights=weights,
                    colors=colors,
                    alpha_mode="OPAQUE" if material is None else material.alpha_mode,
                    alpha_cutoff=0.5 if material is None else material.alpha_cutoff,
                    double_sided=bool(material is not None and material.double_sided),
                    joint_positions=joint_positions,
                    joint_parents=joint_parents,
                )
            )
    return parts

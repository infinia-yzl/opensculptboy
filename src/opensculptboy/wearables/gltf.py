# OpenSculptBoy
# Apache License, Version 2.0
"""
Wearables in glTF files: their materials, and their meshes as nodes of a character's file.

:func:`add_materials` writes garment materials (glTF metallic-roughness, with their
textures) into a :class:`~opensculptboy.export.document.GltfDocument`, and
:func:`read_materials` reads them back from a file. :func:`add_pieces` writes dressed pieces
(:class:`~opensculptboy.wearables.fit.DressedPiece`) as meshes skinned by the character's
skin: one node per piece, and in a wardrobe file one more node per group that another piece
hides, so that a glTF tool can show and hide each piece and each such group.
"""

from __future__ import annotations

import pathlib

import numpy as np

from opensculptboy.export.document import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    GltfDocument,
)

VISIBILITY = "KHR_node_visibility"
TEXTURE_SLOTS = ("base_color", "metallic_roughness", "normal", "occlusion", "emissive")


def index_accessor(doc: GltfDocument, triangles: np.ndarray, vertex_count: int) -> int:
    index_type = np.uint16 if vertex_count < 65536 else np.uint32
    return doc.accessor(
        np.asarray(triangles).reshape(-1).astype(index_type), ELEMENT_ARRAY_BUFFER
    )


def garment_tangents(positions, normals, garment) -> np.ndarray:
    """MikkTSpace tangents (V, 4) float32 of a garment's vertices, for its normal maps."""
    from opensculptboy.texture.tangents import mikktspace

    uv = np.asarray(garment.uvs, dtype=np.float64)
    uv = np.stack([uv[:, 0], 1.0 - uv[:, 1]], axis=1)  # OBJ: v = 0 at the bottom
    return mikktspace(positions, normals, uv, garment.triangles).astype(np.float32)


def _mime(data: bytes) -> str:
    return "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"


def add_materials(doc: GltfDocument, materials, cache: dict | None = None) -> list[int]:
    """Write materials (:class:`~opensculptboy.wearables.format.Material`) and their
    textures; returns their material indices. ``cache`` maps image bytes to texture
    indices across calls, so that an image is written once."""
    cache = {} if cache is None else cache
    out = []
    for material in materials:

        def texture(slot):
            data = material.textures.get(slot)
            if data is None:
                return None
            key = hash(data)
            if key not in cache:
                cache[key] = doc.add_texture(
                    doc.add_image(data, _mime(data), f"{material.name}_{slot}")
                )
            return {"index": cache[key]}

        pbr = {
            "baseColorFactor": [float(c) for c in material.base_color],
            "metallicFactor": float(material.metallic),
            "roughnessFactor": float(material.roughness),
        }
        if texture("base_color"):
            pbr["baseColorTexture"] = texture("base_color")
        if texture("metallic_roughness"):
            pbr["metallicRoughnessTexture"] = texture("metallic_roughness")
        entry = {"name": material.name, "pbrMetallicRoughness": pbr}
        if texture("normal"):
            entry["normalTexture"] = dict(
                texture("normal"), scale=float(material.normal_scale)
            )
        if texture("occlusion"):
            entry["occlusionTexture"] = texture("occlusion")
        if texture("emissive"):
            entry["emissiveTexture"] = texture("emissive")
        if any(material.emissive) or texture("emissive"):
            entry["emissiveFactor"] = [float(c) for c in material.emissive] or [1, 1, 1]
        if material.alpha_mode != "OPAQUE":
            entry["alphaMode"] = material.alpha_mode
        if material.alpha_mode == "MASK":
            entry["alphaCutoff"] = float(material.alpha_cutoff)
        if material.double_sided:
            entry["doubleSided"] = True
        out.append(doc.add_material(entry))
    return out


def _image_bytes(gltf: dict, binary: bytes, index: int, folder) -> bytes | None:
    image = gltf.get("images", [])[index]
    if "bufferView" in image:
        view = gltf["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        return bytes(binary[start : start + view["byteLength"]])
    uri = image.get("uri")
    if not uri:
        return None
    if uri.startswith("data:"):
        import base64

        return base64.b64decode(uri.split(",", 1)[1])
    path = pathlib.Path(folder or ".") / uri
    return path.read_bytes() if path.exists() else None


def read_materials(gltf: dict, binary: bytes, folder=None, which=None):
    """
    The materials of a glTF file as :class:`~opensculptboy.wearables.format.Material`, and
    a map from glTF material index to position in the list. ``which`` limits the list to
    those glTF material indices (in that order).
    """
    from opensculptboy.wearables.format import Material

    entries = gltf.get("materials", [])
    which = list(range(len(entries))) if which is None else list(which)
    textures = gltf.get("textures", [])

    def image(info):
        if not info:
            return None
        source = textures[info["index"]].get("source")
        return None if source is None else _image_bytes(gltf, binary, source, folder)

    out, index = [], {}
    for k in which:
        entry = entries[k]
        pbr = entry.get("pbrMetallicRoughness", {})
        found = {
            "base_color": image(pbr.get("baseColorTexture")),
            "metallic_roughness": image(pbr.get("metallicRoughnessTexture")),
            "normal": image(entry.get("normalTexture")),
            "occlusion": image(entry.get("occlusionTexture")),
            "emissive": image(entry.get("emissiveTexture")),
        }
        index[k] = len(out)
        out.append(
            Material(
                name=entry.get("name", f"material{k}"),
                base_color=tuple(pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0))),
                metallic=float(pbr.get("metallicFactor", 1.0)),
                roughness=float(pbr.get("roughnessFactor", 1.0)),
                emissive=tuple(entry.get("emissiveFactor", (0.0, 0.0, 0.0))),
                alpha_mode=entry.get("alphaMode", "OPAQUE"),
                alpha_cutoff=float(entry.get("alphaCutoff", 0.5)),
                double_sided=bool(entry.get("doubleSided", False)),
                normal_scale=float(
                    (entry.get("normalTexture") or {}).get("scale", 1.0)
                ),
                textures={s: v for s, v in found.items() if v is not None},
            )
        )
    return out, index


def add_pieces(
    doc: GltfDocument,
    pieces,
    skin: int,
    frame: np.ndarray,
    joint_type=np.uint8,
    translation: np.ndarray | None = None,
    visibility: bool = False,
    all_groups: bool = False,
    cache: dict | None = None,
) -> list[tuple[int, bool]]:
    """
    Write dressed pieces as skinned meshes; returns each written node and whether it shows.

    Each piece's vertices and normals turn by ``frame`` (Anny's frame to the file's) and move
    by ``translation``. The groups that the worn pieces hide are left out, unless
    ``all_groups`` writes every group that a rule can hide (``piece.hideable_groups``) as a
    node of its own, hidden by ``KHR_node_visibility`` when ``visibility`` is set and the
    group is hidden. A piece that is not worn (``piece.shown`` False) is written only
    with ``all_groups`` and is hidden the same way.
    """
    cache = {} if cache is None else cache
    nodes = []
    for piece in pieces:
        if not piece.shown and not all_groups:
            continue
        garment = piece.garment
        positions = piece.vertices @ frame.T
        if translation is not None:
            positions = positions + translation
        attributes = {
            "POSITION": doc.accessor(
                positions.astype(np.float32), ARRAY_BUFFER, minmax=True
            ),
            "NORMAL": doc.accessor(
                (piece.normals @ frame.T).astype(np.float32), ARRAY_BUFFER
            ),
        }
        if garment.uvs is not None:
            attributes["TEXCOORD_0"] = doc.accessor(
                np.asarray(garment.uvs, dtype=np.float32), ARRAY_BUFFER
            )
            if any("normal" in m.textures for m in garment.materials):
                attributes["TANGENT"] = doc.accessor(
                    garment_tangents(positions, piece.normals @ frame.T, garment),
                    ARRAY_BUFFER,
                )
        attributes["JOINTS_0"] = doc.accessor(
            piece.joints.astype(joint_type), ARRAY_BUFFER
        )
        attributes["WEIGHTS_0"] = doc.accessor(
            piece.weights.astype(np.float32), ARRAY_BUFFER
        )
        materials = add_materials(doc, garment.materials, cache)

        def mesh_of(group_rows, name):
            primitives = []
            for m in range(len(garment.materials)):
                rows = group_rows & (garment.triangle_material == m)
                if not rows.any():
                    continue
                primitives.append(
                    {
                        "attributes": dict(attributes),
                        "indices": index_accessor(
                            doc, garment.triangles[rows], len(piece.vertices)
                        ),
                        "material": materials[m],
                        "mode": 4,
                    }
                )
            if not primitives:
                return None
            return doc.add_mesh({"name": name, "primitives": primitives})

        hidden = set(piece.hidden_groups)
        separate = set(piece.hideable_groups) | hidden if all_groups else set()
        main_rows = np.ones(len(garment.triangles), bool)
        for g, group in enumerate(garment.groups):
            if group in hidden or group in separate:
                main_rows &= garment.triangle_group != g
        children = []
        for g, group in enumerate(garment.groups):
            if group not in separate:
                continue
            mesh = mesh_of(garment.triangle_group == g, f"{piece.ref}.{group}")
            if mesh is None:
                continue
            node = {"name": f"{piece.ref}.{group}", "mesh": mesh, "skin": skin}
            node["extras"] = {
                "opensculptboy_wearable_group": {"piece": piece.ref, "group": group}
            }
            visible = piece.shown and group not in hidden
            if visibility and not visible:
                node["extensions"] = {VISIBILITY: {"visible": False}}
            children.append((doc.add_node(node), visible))
        mesh = mesh_of(main_rows, piece.ref)
        entry = {
            "name": piece.ref,
            "extras": {"opensculptboy_wearable": piece.extras()},
        }
        if mesh is not None:
            entry.update(mesh=mesh, skin=skin)
        if visibility and not piece.shown:
            entry["extensions"] = {VISIBILITY: {"visible": False}}
        nodes.append((doc.add_node(entry), bool(piece.shown)))
        nodes.extend(children)  # root nodes: glTF skins ignore the parents' transforms
    if visibility:
        doc.use_extension(VISIBILITY)
    return nodes

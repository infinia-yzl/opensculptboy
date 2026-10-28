# OpenSculptBoy
# Apache License, Version 2.0
"""
The pieces of a VRM file and what each one costs against the budget.

A VRM file of :mod:`opensculptboy.export.vrm` holds the body mesh, whose primitives draw the
skin (with its head and ear tiles), the eyes, the lashes, the brows and the teeth or the inner
mouth, and one mesh per shown wearable piece and for the hair. :func:`spec_pieces` names these pieces in a
:class:`~opensculptboy.export.vrm.VrmSpec`, and :func:`piece_report` counts for each one its
triangles, its vertices (those its triangles use), its materials, its textures with their
sizes and the joints that weight its vertices. :func:`spec_counts` gives the counts of the file
that the spec writes, as :func:`opensculptboy.export.budget.counts` reads them from the file,
without writing it. :func:`report_lines` prints a report in one line per piece, for the budget
messages and the command's output.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# the body mesh's materials by piece; other materials of the body mesh count as the skin
BODY_PIECES = {
    "skin": "body",
    "head": "body",
    "ears": "body",
    "eyes": "eyes",
    "conjunctiva": "eyes",
    "lashes": "lashes",
    "brows": "brows",
    "teeth": "teeth",
    "gums": "teeth",
    "mouth": "mouth",
}
# the texture slots of an MToon material
TEXTURE_FIELDS = (
    "base_texture",
    "shade_texture",
    "normal_texture",
    "matcap_texture",
    "emissive_texture",
    "rim_multiply_texture",
    "outline_width_texture",
)


@dataclasses.dataclass
class Piece:
    """A piece of a VRM file: a name, a kind (``body``, ``eyes``, ``lashes``, ``brows``,
    ``teeth``, ``mouth`` or ``wearable``, which the hair's mesh is too) and its primitives, as
    (mesh, primitive) indices of the spec."""

    name: str
    kind: str
    mesh: int
    primitives: list


def spec_pieces(spec) -> list[Piece]:
    """The pieces of a spec, in the order of its meshes and primitives."""
    out: dict[str, Piece] = {}
    for m, mesh in enumerate(spec.meshes):
        for p, primitive in enumerate(mesh.primitives):
            if m == 0 and mesh.name == "body":
                name = spec.materials[primitive.material].name
                kind = BODY_PIECES.get(name, "body")
                key = kind
            else:
                kind, key = "wearable", mesh.name
            if key not in out:
                out[key] = Piece(key, kind, m, [])
            out[key].primitives.append(p)
    return list(out.values())


def primitive_triangles(mesh, primitive) -> np.ndarray:
    """(T, 3) the triangles of a primitive of a spec mesh."""
    if primitive.triangles is None:
        return np.asarray(mesh.body.triangles, np.int64).reshape(-1, 3)
    return np.asarray(primitive.triangles, np.int64).reshape(-1, 3)


def texture_indices(material) -> dict:
    """The texture slots of an MToon material that hold a texture, by field."""
    return {
        f: getattr(material, f)
        for f in TEXTURE_FIELDS
        if getattr(material, f) is not None
    }


def image_sizes(spec) -> list:
    """(width, height) of each image of a spec (None where unknown)."""
    from opensculptboy.export.budget import image_size

    return [image_size(data) for _, data in spec.images]


def piece_report(spec) -> list[dict]:
    """
    Per piece: its ``name``, ``kind``, ``triangles``, ``vertices``, ``materials`` (names),
    ``textures`` (``{"name", "size": [w, h]}`` per distinct image) and ``joints`` (the joints
    that weight a vertex of its triangles).
    """
    sizes = image_sizes(spec)
    out = []
    for piece in spec_pieces(spec):
        mesh = spec.meshes[piece.mesh]
        tris = [primitive_triangles(mesh, mesh.primitives[p]) for p in piece.primitives]
        tris = np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)
        used = np.unique(tris)
        weights = np.asarray(mesh.body.weights)[used]
        joints = np.asarray(mesh.body.joints)[used]
        joint_set = np.unique(joints[weights > 0])
        materials, images = [], []
        for p in piece.primitives:
            material = spec.materials[mesh.primitives[p].material]
            if material.name not in materials:
                materials.append(material.name)
            for index in texture_indices(material).values():
                if index not in images:
                    images.append(index)
        textures = [
            {
                "name": spec.images[i][0],
                "size": None if sizes[i] is None else [int(x) for x in sizes[i]],
            }
            for i in images
            if i < len(spec.images)
        ]
        out.append(
            {
                "name": piece.name,
                "kind": piece.kind,
                "triangles": int(len(tris)),
                "vertices": int(len(used)),
                "materials": materials,
                "textures": textures,
                "joints": int(len(joint_set)),
            }
        )
    return out


def spec_counts(spec) -> dict:
    """The counts of :func:`opensculptboy.export.budget.counts` for the file of a spec:
    triangles and vertices (each primitive counts its mesh's vertices, as the file shares
    them), joints, materials, images (with the thumbnail), the largest image and the morph
    targets."""
    from opensculptboy.export.budget import image_size

    drawn = {node.mesh for node in spec.nodes if node.mesh is not None}
    triangles = vertices = 0
    for m in drawn:
        mesh = spec.meshes[m]
        for primitive in mesh.primitives:
            triangles += len(primitive_triangles(mesh, primitive))
            vertices += len(mesh.body.positions)
    largest = 0
    for size in image_sizes(spec):
        if size is not None:
            largest = max(largest, *size)
    images = len(spec.images)
    if spec.thumbnail is not None:
        images += 1
        size = image_size(spec.thumbnail)
        if size is not None:
            largest = max(largest, *size)
    joints = {j for skin in spec.skins for j in skin.joints}
    return {
        "triangles": int(triangles),
        "vertices": int(vertices),
        "joints": len(joints),
        "materials": len(spec.materials),
        "images": int(images),
        "max_image_size": int(largest),
        "morph_targets": int(sum(len(m.target_names) for m in spec.meshes)),
        "meshes": len(spec.meshes),
    }


def report_lines(report: list[dict]) -> list[str]:
    """One line per piece of :func:`piece_report`."""
    lines = []
    for piece in report:
        sizes = [t["size"] for t in piece["textures"] if t["size"]]
        largest = max((max(s) for s in sizes), default=0)
        text = (
            f"{piece['name']}: {piece['triangles']:,} triangles, "
            f"{piece['vertices']:,} vertices, {len(piece['materials'])} materials, "
            f"{len(piece['textures'])} textures"
        )
        if largest:
            text += f" (largest {largest})"
        text += f", {piece['joints']} joints"
        lines.append(text)
    return lines

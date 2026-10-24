# OpenSculptBoy
# Apache License, Version 2.0
"""
The wearables of a card in the GLB and VRM exports.

:class:`Dressing` holds the choices of one export: the outfit, the pieces to show or hide,
whether the covered skin is removed, and whether the file is a wardrobe (every piece of every
outfit as its own node, shown or hidden by the worn outfit). :func:`dress_card` resolves
them (:func:`opensculptboy.wearables.wardrobe.worn_pieces`) and dresses the card's body
(:func:`opensculptboy.wearables.fit.dress`). The exporters then remove the covered body
triangles (:func:`covered_body_triangles`) and write the pieces.
"""

from __future__ import annotations

import dataclasses

import numpy as np

COVERED_MODES = ("remove", "keep")


@dataclasses.dataclass
class Dressing:
    """
    The wearables of one export.

    Args:
        outfit: the outfit to wear (``"none"`` for none); the card's ``active_outfit`` when
            None.
        show, hide: piece references to show or hide (``<outfit>/<piece>``, or a bare piece
            id of the worn outfit or of a loose piece).
        covered: ``"remove"`` takes the body triangles that the shown pieces cover out of
            the file; ``"keep"`` keeps every triangle.
        wardrobe: write every piece of every outfit, each as its own node, with the body
            whole; the worn outfit decides what shows by default.
        visibility: in a wardrobe file, keep every piece in the default scene and mark the
            pieces and groups that do not show with ``KHR_node_visibility``. Without it (the
            default, since Blender's importer ignores the extension), the default scene holds
            what the worn outfit shows and a second scene holds every piece.
        push: run the push-out of :func:`opensculptboy.wearables.fit.dress`.
        springs: write the spring chains of the pieces that have them
            (:mod:`opensculptboy.wearables.springs`); False leaves every piece on the body's
            bones.
    """

    outfit: str | None = None
    show: tuple = ()
    hide: tuple = ()
    covered: str = "remove"
    wardrobe: bool = False
    visibility: bool = False
    push: bool = True
    springs: bool = True

    def check(self) -> None:
        if self.covered not in COVERED_MODES:
            raise ValueError(
                f"covered must be one of {COVERED_MODES}; got {self.covered!r}."
            )


def dress_card(
    character,
    model,
    detail_dir,
    dressing: Dressing,
    max_influences: int = 4,
    skin_weights=None,
):
    """(the card's body, the dressed pieces) of an export; pieces that do not show come too
    in a wardrobe file."""
    from opensculptboy.wearables.fit import card_body, dress
    from opensculptboy.wearables.wardrobe import worn_pieces

    dressing.check()
    pieces = worn_pieces(
        character,
        detail_dir,
        dressing.outfit,
        dressing.show,
        dressing.hide,
        every_outfit=dressing.wardrobe,
    )
    body = card_body(character, model, detail_dir)
    return body, dress(
        body, pieces, max_influences, dressing.push, skin_weights=skin_weights
    )


def covered_body_triangles(body, pieces, mesh, rows: int, model_rows: int, detail):
    """(T,) the triangles of the exported body mesh ``mesh`` that the shown pieces cover."""
    from opensculptboy.wearables.cover import covered_triangles, outfit_cover, row_cover

    mask = outfit_cover(body, pieces)
    return covered_triangles(mesh, row_cover(mask, rows, model_rows, detail))


def summary(pieces, removed: int, dressing: Dressing, springs=None) -> dict:
    """The ``wearables`` entry of a file's scene extras; ``springs`` is the file's
    :class:`~opensculptboy.wearables.springs.SpringRig`, or None."""
    out = {
        "format": "opensculptboy.wearable/1",
        "outfit": dressing.outfit,
        "wardrobe": bool(dressing.wardrobe),
        "covered": dressing.covered,
        "removed_body_triangles": int(removed),
        "pieces": [p.extras() for p in pieces],
    }
    if springs is not None:
        out["springs"] = springs.extras()
    return out


def piece_springs(body, pieces, model, dressing: Dressing, max_skin_joints=None):
    """The :class:`~opensculptboy.wearables.springs.SpringRig` of the pieces that a file
    writes (the shown ones; every one in a wardrobe), with their skin weights blended into
    the chains, or None (no springs, or ``dressing.springs`` off)."""
    if not dressing.springs:
        return None
    from opensculptboy.wearables.springs import rig_springs

    written = [p for p in pieces if p.shown or dressing.wardrobe]
    parents = [int(p) for p in model.bone_parents]
    return rig_springs(body, written, parents, max_skin_joints)


def lowest(pieces, frame: np.ndarray) -> float:
    """The lowest height (the file's Y) of the shown pieces, or +inf."""
    low = np.inf
    for piece in pieces:
        if piece.shown and len(piece.vertices):
            low = min(low, float((piece.vertices @ frame.T)[:, 1].min()))
    return low


def vrm_piece_meshes(pieces, rebind, frame: np.ndarray, transforms=None) -> list:
    """
    (piece, :class:`~opensculptboy.export.body.BodyMesh`) of each shown piece in a VRM
    file's T-pose: each vertex turns by the inverse of the blend of the bones' inverse turns
    with its four skin weights, so that the file, posed back to Anny's rest pose, shows the
    piece's rest vertices exactly (the inverse bind of
    :func:`opensculptboy.export.tpose.rebind`). The mesh holds the visible triangles, and
    ``triangle_faces`` holds each triangle's material. ``transforms`` (the bones' turns
    followed by those of the spring joints, each its chain's parent bone's) replaces
    ``rebind.transforms``.
    """
    from opensculptboy.export.body import BodyMesh
    from opensculptboy.export.tpose import blend, rigid_inverse
    from opensculptboy.wearables.geometry import unit_rows

    X = rebind.transforms if transforms is None else transforms
    out = []
    for piece in pieces:
        if not piece.shown:
            continue
        garment = piece.garment
        weights = piece.weights.astype(np.float64)
        N = blend(weights, piece.joints, rigid_inverse(X))
        linear = np.linalg.inv(N[:, :3, :3])
        bind = np.einsum("nij,nj->ni", linear, piece.vertices - N[:, :3, 3])
        normals = unit_rows(np.einsum("nij,nj->ni", linear, piece.normals))
        keep = np.ones(len(garment.triangles), bool)
        for g, group in enumerate(garment.groups):
            if group in piece.hidden_groups:
                keep &= garment.triangle_group != g
        if not keep.any():  # other pieces hide every group of this one
            continue
        count = len(bind)
        mesh = BodyMesh(
            positions=bind @ frame.T,
            normals=normals @ frame.T,
            uvs=None if garment.uvs is None else np.asarray(garment.uvs),
            source=np.arange(count),
            triangles=garment.triangles[keep],
            triangle_faces=garment.triangle_material[keep],
            joints=piece.joints.astype(np.int64),
            weights=piece.weights.astype(np.float32),
            targets=[],
            dropped_weight=0.0,
        )
        if garment.uvs is not None and any(
            "normal" in m.textures for m in garment.materials
        ):
            mesh.add_tangents()
        out.append((piece, mesh))
    return out


def vrm_piece_materials(piece, first_texture: int, images: list):
    """The MToon materials of a piece's garment materials, and the list of the file's
    garment images (slot, bytes) with this piece's added; ``first_texture`` is the texture
    index of the first garment image."""
    from opensculptboy.export import mtoon

    images = list(images)
    known = {hash(data): k for k, (_, data) in enumerate(images)}

    def texture(material, slot):
        data = material.textures.get(slot)
        if data is None:
            return None
        key = hash(data)
        if key not in known:
            known[key] = len(images)
            images.append((f"{piece.ref}:{material.name}:{slot}", data))
        return first_texture + known[key]

    out = []
    for material in piece.garment.materials:
        r, g, b, a = (tuple(material.base_color) + (1.0,))[:4]
        base = texture(material, "base_color")
        out.append(
            mtoon.ToonMaterial(
                name=f"{piece.ref}:{material.name}",
                base_color=(r, g, b, a),
                base_texture=base,
                shade_color=(0.7 * r, 0.62 * g, 0.6 * b),
                shade_texture=base,
                shading_shift=-0.1,
                shading_toony=0.5,
                gi_equalization=0.6,
                normal_texture=texture(material, "normal"),
                normal_scale=float(material.normal_scale),
                alpha_mode=material.alpha_mode,
                alpha_cutoff=float(material.alpha_cutoff),
                double_sided=bool(material.double_sided),
            )
        )
    return out, images

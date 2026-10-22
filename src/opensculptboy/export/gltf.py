# OpenSculptBoy
# Apache License, Version 2.0
"""
glTF 2.0 binary (GLB) export of an Anny character.

The file holds the character's mesh in its rest (bind) pose with normals and UVs, the bones of
the rig as joint nodes with inverse bind matrices, the skin weights (the strongest 4 or 8 bones
of each vertex), morph targets for the facial actions and, on request, for the face shapes, and
one animation for each pose or clip of ``anny.poses`` that the caller names, after an animation
named ``pose`` when the character holds its own pose. The phenotype, the
local changes and the face shapes that are not morph targets are baked into the mesh and the
skeleton. The settings of the character are stored in the ``extras`` of the scene, and
``read_character`` reads them back, from GLB and VRM files alike.

The exporter builds the file with :class:`~opensculptboy.export.document.GltfDocument` and the
body data of :mod:`opensculptboy.export.body`, which the VRM exporter shares.

Frame: Anny works in metres with Z up and the figure facing -Y. glTF works in metres with Y up
and the figure facing +Z. Every position p becomes C p, and every joint matrix M becomes
C M C^T, where C is the rotation (x, y, z) -> (x, z, -y). The skinned result is then C times
Anny's skinned vertices.
"""

from __future__ import annotations

import difflib
import pathlib
from typing import Iterable, Sequence

import numpy as np
import roma
import torch

from opensculptboy.character import Character
from opensculptboy.mouth.rig import is_mouth_rig
from opensculptboy.export.body import (
    ANNY_TO_GLTF,
    body_primitive,
    build_body,
    morph_target_rows,
    rest_rows,
)
from opensculptboy.export.detail import ear_mesh, mime_type, texture_images
from opensculptboy.export.detail import resolve as resolve_detail
from opensculptboy.export.document import GltfDocument, pbr_material, read_gltf_json

__all__ = ["C3", "C4", "export_glb", "read_character", "read_gltf_json"]

# (x, y, z) -> (x, z, -y): Anny's Z-up frame to glTF's Y-up frame.
C3 = ANNY_TO_GLTF.copy()
C4 = np.eye(4)
C4[:3, :3] = C3


def _check_names(names: Iterable[str], allowed: Sequence[str], what: str) -> None:
    for name in names:
        if name not in allowed:
            close = difflib.get_close_matches(name, allowed, n=3)
            hint = f" Did you mean {close}?" if close else ""
            raise ValueError(f"Unknown {what} {name!r}.{hint}")


def _to_gltf_matrices(matrices: np.ndarray) -> np.ndarray:
    """Joint matrices (..., 4, 4) of Anny's frame in glTF's frame: C M C^T."""
    return C4 @ matrices @ C4.T


def _local_transforms(world: np.ndarray, parents: Sequence[int]) -> np.ndarray:
    """Local matrices (..., J, 4, 4) of joints from their world matrices."""
    local = world.copy()
    for j, parent in enumerate(parents):
        if parent >= 0:
            local[..., j, :, :] = (
                np.linalg.inv(world[..., parent, :, :]) @ world[..., j, :, :]
            )
    return local


def _quaternions(rotations: np.ndarray) -> np.ndarray:
    """Unit quaternions (x, y, z, w) of rotation matrices (..., 3, 3)."""
    return roma.rotmat_to_unitquat(
        torch.from_numpy(np.ascontiguousarray(rotations))
    ).numpy()


def export_glb(
    path: str | pathlib.Path,
    character: Character | None = None,
    model=None,
    animations: Sequence[str] = (),
    morph_targets: Sequence[str] | None = None,
    max_influences: int = 4,
    ground: bool = True,
    base_color: Sequence[float] = (0.80, 0.62, 0.52, 1.0),
    detail_level: int = 0,
    detail_region: str = "full",
    detail_dir: str | pathlib.Path | None = None,
    textures=None,
) -> dict:
    """
    Write a character as a binary glTF 2.0 file.

    Args:
        path: the ``.glb`` file to write.
        character: the character; the default character when None.
        model: the Anny model to use; ``character.build_model()`` when None. It must have
            ``facial_actions="all"`` for facial-action morph targets, and face shapes for
            face-shape morph targets.
        animations: names of poses and clips of ``anny.poses`` (``anny.poses.names()``), one
            animation each. A pose is one keyframe; a clip plays at its own frame rate. A
            character with a ``pose`` also gets a one-keyframe animation named ``pose``, first.
        morph_targets: names of facial actions and face shapes to export as morph targets.
            None exports every facial action of the model. A facial action gives one target
            (weight 1 is the value 1). A face shape gives ``<name>.pos`` at +1 and, when the
            shape runs below 0, ``<name>.neg`` at -1. The character's values become the default
            weights, and the other face shapes are baked into the mesh.
        max_influences: bones per vertex, 4 (read by every engine) or 8 (a second
            JOINTS/WEIGHTS set). Anny uses up to 9; the weaker ones are dropped and the rest
            renormalised.
        ground: stand the rest pose on the floor (y = 0). The poses and clips of
            ``anny.poses`` stand on the floor in any case.
        base_color: the linear RGBA base colour of the material.
        detail_level: the level of the character's detail layer (``character.detail``, see
            :mod:`opensculptboy.detail`): 0 is anny's own mesh, 1 and 2 its Catmull-Clark
            subdivisions, with the card's offsets of that level added to the rest mesh and the
            skin weights, UVs and morph targets carried to the finer mesh. Level 0 of a card
            without detail writes the file as it was before the detail layer existed.
        detail_region: ``"full"`` (the whole body at ``detail_level``) or ``"head"`` (the head
            and the neck at ``detail_level``, the rest one level coarser).
        detail_dir: the folder that the paths of the detail layer are relative to (the card's
            folder); the current folder when None.
        textures: None for the skin textures that the card names (its detail layer's and its
            skin material's; none for most cards), False for none, or a mapping of texture
            slots (:data:`opensculptboy.export.detail.TEXTURE_SLOTS`) to PNG or JPEG files
            (paths or bytes) laid out on anny's UVs. A card with a ``material`` section gets
            the skin material of :func:`_skin_material`, with the head tile's textures on a
            primitive of their own; a file with a normal texture carries MikkTSpace
            ``TANGENT`` attributes.

    Returns:
        A summary: counts, the file size and the largest dropped skin weight.
    """
    if max_influences not in (4, 8):
        raise ValueError(f"max_influences must be 4 or 8, got {max_influences}.")
    character = character or Character()
    if model is None:
        wants_face = morph_targets is not None and bool(
            set(morph_targets) & set(_face_shape_names())
        )
        model = character.build_model(
            face_shapes=bool(character.face_shapes) or wants_face
        )
    facial_labels = list(model.facial_action_labels)
    face_labels = list(model.face_shape_labels)
    if morph_targets is None:
        morph_targets = facial_labels
    _check_names(morph_targets, facial_labels + face_labels, "morph target")
    _check_names(character.phenotype, list(model.phenotype_labels), "phenotype")
    _check_names(
        character.local_changes, list(model.local_change_labels), "local change"
    )
    _check_names(character.face_shapes, face_labels, "face shape")
    _check_names(character.facial_actions, facial_labels, "facial action")
    if animations:
        import anny.poses

        _check_names(animations, anny.poses.names(), "pose or clip")

    # Rows of parameter values: row 0 is the base mesh, one more row per morph target.
    targets, facial_rows, face_rows = morph_target_rows(model, character, morph_targets)
    rest_vertices, rest_bone_poses = rest_rows(model, character, facial_rows, face_rows)
    # On the anny-mouth rig: jawOpen as a turn of the jaw and the corrective target that
    # completes it (opensculptboy.mouth.jaw), played by the animation "jawOpen.jaw".
    jaw_open = None
    if is_mouth_rig(character.rig):
        from opensculptboy.mouth.jaw import with_jaw_corrective

        jaw_open, targets, rest_vertices = with_jaw_corrective(
            model, targets, rest_vertices, rest_bone_poses, max_influences
        )
    # A reference's mouth poses (opensculptboy.mouth.poses): a target "mouth.<pose>" each,
    # and on the anny-mouth rig a turn of the jaw with a corrective target, played by the
    # animation "mouth.<pose>.jaw".
    mouth_poses, pose_turns, inner_overrides = None, [], {}
    if character.mouth_material() is not None:
        from opensculptboy.mouth.poses import card_poses

        mouth_poses = card_poses(character, detail_dir)
    if mouth_poses is not None:
        from opensculptboy.export.body import MorphTarget
        from opensculptboy.mouth import poses as mouth_pose_data

        targets, rest_vertices, added, mouth_poses = mouth_pose_data.append_pose_rows(
            targets, rest_vertices, mouth_poses
        )
        for row, k in added.items():
            inner_overrides[row] = mouth_poses.inner[k]
        if "jaw" in model.bone_labels:
            pose_turns = mouth_pose_data.pose_turns(
                model,
                rest_vertices[0],
                mouth_poses,
                rest_bone_poses[list(model.bone_labels).index("jaw")],
                max_influences,
            )
            extra = []
            for k, turn in enumerate(pose_turns):
                name = mouth_pose_data.target_name(mouth_poses.names[k])
                inner_overrides[len(targets)] = ("corrective", k)
                targets.append(
                    MorphTarget(f"{name}.corrective", "corrective", -1, 0.0, 0.0)
                )
                extra.append((rest_vertices[0] + turn.corrective)[None])
            rest_vertices = np.concatenate([rest_vertices, *extra])
    base = rest_vertices[0]

    # Mesh: split the vertices at UV seams; normals come from the welded mesh.
    detail = resolve_detail(model, character, detail_level, detail_dir, detail_region)
    images = texture_images(character, textures, detail_dir)
    skin = character.skin_material()
    eye = character.eye_material()
    mesh, ears = ear_mesh(character, skin, images, detail, detail_dir)
    if textures is None:  # the card's maps, baked on its finest level
        from opensculptboy.texture.levels import level_normals

        images = level_normals(
            model, character, images, detail, mesh, ears, base, detail_dir
        )
    eyed = None
    teeth = character.teeth_material()
    mouth = character.mouth_material()
    if mouth is not None:  # the reference's inner mouth holds its own teeth
        teeth = None
    toothed = None
    if detail is None and eye is None and teeth is None and mouth is None:
        body = build_body(
            model, base, rest_vertices[1:] - base, ANNY_TO_GLTF, max_influences
        )
    else:
        weights = model.vertex_bone_weights.detach().cpu().double().numpy()
        indices = model.vertex_bone_indices.cpu().numpy()
        vertices, offsets = base, rest_vertices[1:] - base
        if detail is not None:
            weights, indices = detail.skin_weights(weights, indices)
            vertices, offsets = detail.vertices(base), detail.targets(offsets)
        normals = None
        rows_mesh = mesh  # the file's mesh before the eyes and the teeth
        if eye is not None:  # the eyes of opensculptboy.eyes replace anny's eyeballs
            from opensculptboy.eyes.mesh import baked_skin, eyed_mesh

            baked = None if "eye_base_color" in images else images.get("eyes")
            eyed = eyed_mesh(
                model, mesh, vertices, eye, "gltf", baked, skin_baked=baked_skin(images)
            )
            normals = eyed.normals(vertices)
            weights, indices = eyed.skin_weights(weights, indices)
            vertices, offsets = eyed.vertices(vertices), eyed.targets(offsets)
            ears = eyed.extend_mask(ears)
            mesh = eyed
        if teeth is not None or mouth is not None:  # opensculptboy.teeth, mouth.inner
            from opensculptboy.teeth.mesh import card_teeth, toothed_mesh

            toothed = toothed_mesh(
                model,
                None if detail is None else rows_mesh,
                mesh,
                vertices,
                card_teeth(model, character, detail_dir, base),
            )
            normals = toothed.normals(vertices, normals)
            weights, indices = toothed.skin_weights(weights, indices)
            overrides = None
            if inner_overrides:
                overrides = mouth_pose_data.resolve_overrides(
                    toothed.teeth, inner_overrides, mouth_poses, pose_turns
                )
            vertices = toothed.vertices(vertices)
            offsets = toothed.targets(offsets, overrides)
            ears = toothed.extend_mask(ears)
            if eyed is not None:
                eyed.trailing_triangles = len(toothed.teeth.triangles)
            mesh = toothed
        body = build_body(
            model,
            vertices,
            offsets,
            ANNY_TO_GLTF,
            max_influences,
            weights,
            indices,
            normals=normals,
            mesh=mesh,
        )
        if toothed is not None:
            source = toothed.anny_vertex(None if eyed is None else eyed.anny_vertex())
            body.anny_source = source[body.source]
        elif eyed is not None:
            body.anny_source = eyed.anny_vertex()[body.source]
    doc = GltfDocument(f"OpenSculptBoy glTF exporter (anny {_anny_version()})")
    head_material, head, ear_material = None, None, None
    if skin is not None:
        written: dict = {}  # slot -> texture index, so that each image is written once
        material = doc.add_material(_skin_material(doc, skin, images, written))
        head_material, head = _head_material(
            doc, model, body, skin, images, written, character.topology
        )
        if ears is not None:
            ear_material = doc.add_material(
                _skin_material(doc, skin, images, written, "ears", "ears_")
            )
    elif set(images) - EYE_SLOTS:
        material = doc.add_material(_textured_material(doc, images))
    else:
        material = doc.add_material(pbr_material("skin", base_color, 0.0, 0.6))
    if (
        "normal" in images
        or "head_normal" in images
        or (mouth is not None and mouth.normal)
    ):
        body.add_tangents()  # the MikkTSpace frames that the normal maps were baked in
    joint_type = np.uint8 if model.bone_count <= 256 else np.uint16
    eye_parts = []  # (material, triangle mask) of the eyes of opensculptboy.eyes
    if eyed is not None:
        eye_material, eyes = None, None
        eye_parts = _eye_part_materials(doc, model, body, eyed, eye, images)
    else:
        eye_material, eyes = _eye_material(
            doc, model, body, images, None if detail is None else detail.topology
        )
    teeth_parts = []  # (material, triangle mask) of the teeth of opensculptboy.teeth
    hidden = np.zeros(len(body.triangles), bool)
    if toothed is not None and mouth is not None:
        teeth_parts = _mouth_materials(doc, toothed, mouth, detail_dir)
        hidden = toothed.hidden_mask()
    elif toothed is not None:
        teeth_parts = _teeth_materials(doc, toothed, teeth)
    rest = ~hidden
    if eye_material is not None:
        rest &= ~eyes
    for _, part in eye_parts + teeth_parts:
        rest &= ~part
    if ear_material is not None:
        ears &= rest
        rest &= ~ears
    if head_material is not None:
        head &= rest
        rest &= ~head
    primitive = body_primitive(
        doc,
        body,
        material,
        triangles=None if rest.all() else body.triangles[rest],
        joint_type=joint_type,
        anny_vertex=detail is None or detail.subdiv == 0,
    )
    primitives = [primitive]
    if ear_material is not None:  # the ear tile: the outer ears on UVs of their own
        primitives.append(
            body_primitive(
                doc,
                body,
                ear_material,
                triangles=body.triangles[ears],
                attributes=primitive["attributes"],
                targets=primitive.get("targets"),
            )
        )
    if head_material is not None:  # the head tile: the same vertices, its own textures
        primitives.append(
            body_primitive(
                doc,
                body,
                head_material,
                triangles=body.triangles[head],
                attributes=primitive["attributes"],
                targets=primitive.get("targets"),
            )
        )
    if eye_material is not None:  # the eyeballs: the same vertices, their own texture
        primitives.append(
            body_primitive(
                doc,
                body,
                eye_material,
                triangles=body.triangles[eyes],
                attributes=primitive["attributes"],
                targets=primitive.get("targets"),
            )
        )
    for part_material, part in eye_parts + teeth_parts:  # the eyes and the teeth
        primitives.append(
            body_primitive(
                doc,
                body,
                part_material,
                triangles=body.triangles[part],
                attributes=primitive["attributes"],
                targets=primitive.get("targets"),
            )
        )
    mesh = {"name": character.name, "primitives": primitives}
    if targets:
        mesh["weights"] = [t.weight for t in targets]
        mesh["extras"] = {"targetNames": [t.name for t in targets]}

    # Skeleton: the bind pose is the rest pose of the base character.
    parents = [int(p) for p in model.bone_parents]
    labels = list(model.bone_labels)
    bind = _to_gltf_matrices(rest_bone_poses)
    lift = np.eye(4)
    if ground:
        # the lowest vertex as the file stores it (float32), on the floor
        lift[1, 3] = -float(body.positions[:, 1].astype(np.float32).min())
    default_local = _local_transforms(lift @ bind, parents)
    for j, label in enumerate(labels):
        node = {
            "name": label,
            "translation": default_local[j, :3, 3].tolist(),
            "rotation": _quaternions(default_local[j, :3, :3]).tolist(),
        }
        children = [c for c, p in enumerate(parents) if p == j]
        if children:
            node["children"] = children
        doc.add_node(node)
    mesh_index = doc.add_mesh(mesh)
    roots = [j for j, p in enumerate(parents) if p < 0]
    inverse_bind = np.linalg.inv(bind).transpose(0, 2, 1).reshape(len(labels), 16)
    skin = doc.add_skin(
        {
            "name": character.rig,
            "joints": list(range(len(labels))),
            "skeleton": roots[0],
            "inverseBindMatrices": doc.accessor(inverse_bind.astype(np.float32)),
        }
    )
    mesh_node = doc.add_node({"name": character.name, "mesh": mesh_index, "skin": skin})

    if character.pose:
        doc.add_animation(
            _character_pose(doc, model, character, parents, default_local)
        )
    for name in animations:
        doc.add_animation(
            _animation(doc, model, character, name, parents, default_local)
        )
    if jaw_open is not None:
        doc.add_animation(
            _jaw_open_animation(
                doc, jaw_open, labels, default_local, mesh_node, targets
            )
        )
    for k, turn in enumerate(pose_turns):
        name = mouth_pose_data.target_name(mouth_poses.names[k])
        doc.add_animation(
            _jaw_open_animation(
                doc,
                turn,
                labels,
                default_local,
                mesh_node,
                targets,
                name=f"{name}.jaw",
                corrective=f"{name}.corrective",
            )
        )

    doc.scene_name = character.name
    doc.scene_nodes = roots + [mesh_node]
    doc.scene_extras = {
        "opensculptboy": {
            "generator": "opensculptboy",
            "anny_version": _anny_version(),
            "character": character.to_dict(),
            "morph_targets": [t.name for t in targets],
            "animations": (["pose"] if character.pose else []) + list(animations),
            "max_influences": max_influences,
            "frame": "metres, Y up, the figure faces +Z",
        }
    }
    if detail is not None:
        doc.scene_extras["opensculptboy"]["detail"] = detail.extras()
    if images:
        doc.scene_extras["opensculptboy"]["textures"] = sorted(images)
    if head_material is not None:
        doc.scene_extras["opensculptboy"]["head_tile"] = True
    if ear_material is not None:
        doc.scene_extras["opensculptboy"]["ear_tile"] = True
    if eyed is not None:
        doc.scene_extras["opensculptboy"]["eyes"] = eyed.extras
    if jaw_open is not None:
        doc.scene_extras["opensculptboy"]["mouth"] = jaw_open.extras()
    if mouth_poses is not None:
        doc.scene_extras["opensculptboy"].setdefault("mouth", {})["poses"] = {
            name: dict(
                mouth_poses.stats.get(name, {}),
                target=mouth_pose_data.target_name(name),
                **(
                    {"jaw_degrees": round(pose_turns[k].angle_degrees, 3)}
                    if pose_turns
                    else {}
                ),
            )
            for k, name in enumerate(mouth_poses.names)
        }
    if toothed is not None:
        key = "teeth" if mouth is None else "inner_mouth"
        doc.scene_extras["opensculptboy"][key] = toothed.extras
    size = doc.write(path)
    summary = dict(
        path=str(path),
        bytes=size,
        vertices=int(len(body.source)),
        triangles=int(len(body.triangles)),
        joints=len(labels),
        morph_targets=len(targets),
        animations=len(doc.animations),
        max_influences=max_influences,
        largest_dropped_weight=body.dropped_weight,
    )
    if detail is not None:
        summary["detail"] = detail.extras()
    return summary


def _eye_material(doc: GltfDocument, model, body, images: dict, mesh=None):
    """
    The material of the eyeballs with the eye texture of ``images`` (its texture info carries
    the ``KHR_texture_transform`` of :func:`opensculptboy.texture.eyes.eye_texture_transform`)
    and the mask of the eyeball triangles, or (None, None) without an eye texture.
    """
    from opensculptboy.export.body import eyeball_triangles
    from opensculptboy.texture.eyes import eye_texture_transform

    if "eyes" not in images:
        return None, None
    transform = eye_texture_transform(model)
    eyes = eyeball_triangles(model, body, mesh)
    if transform is None or eyes is None or not eyes.any():
        return None, None
    data = images["eyes"]
    texture = doc.add_texture(doc.add_image(data, mime_type(data), "eyes"))
    material = pbr_material("eyes", (1.0, 1.0, 1.0, 1.0), 0.0, 0.3, texture=texture)
    material["pbrMetallicRoughness"]["baseColorTexture"]["extensions"] = {
        "KHR_texture_transform": transform
    }
    return doc.add_material(material), eyes


# the texture slots of the eyes alone, which leave the skin's material as it is
EYE_SLOTS = {"eyes", "eyes_normal", "eye_base_color"}
# the key of the eye materials' settings in their extras
EYE_EXTRAS = "opensculptboy_eye"
EYE_EXTRAS_FORMAT = "opensculptboy/eye@1"


def _eye_part_materials(doc: GltfDocument, model, body, eyed, eye, images: dict):
    """
    The materials and triangle masks of the eyes of :mod:`opensculptboy.eyes` (an
    :class:`~opensculptboy.eyes.mesh.EyedMesh` behind ``body``):

    - ``eyes``: the eye texture (:func:`opensculptboy.eyes.mesh.eye_texture`) on the eyeballs;
      under a cornea a diffuse surface without a specular reflection of its own
      (``KHR_materials_specular`` factor 0: the eyeball lies in the eye's own medium, and a
      reflection there reads as a milky veil over the iris), and without one (``eye.cornea``
      false) a clear coat (``KHR_materials_clearcoat``) at the cornea's IOR for the wet
      highlight;
    - ``cornea``: a clear, refracting shell (``KHR_materials_transmission``,
      ``KHR_materials_volume`` with the depth of the anterior chamber as its thickness,
      ``KHR_materials_ior``). As the transmission extension asks, it is OPAQUE: readers
      without the extension draw it white, so ``eye.cornea`` false suits them better;
    - ``conjunctiva``: the caruncle and the skin of the socket, pink and wet (a clear coat).

    The extras of each hold :data:`EYE_EXTRAS` (``opensculptboy/eye@1``) with its ``role``
    and the eye's settings, which :mod:`opensculptboy.export.blender_skin` reads.
    """
    from opensculptboy.eyes.geometry import CHAMBER_DEPTH_MM
    from opensculptboy.eyes.mesh import eye_texture

    eye = eyed.eye if eyed.eye is not None else eye  # with the sizes the mesh took
    settings = dict(eye.extras(), bulge={k: float(v) for k, v in eyed.bulge.items()})

    def extras(role):
        return {EYE_EXTRAS: {"format": EYE_EXTRAS_FORMAT, "role": role, **settings}}

    from opensculptboy.export.document import png_bytes

    bent = None
    if "eye_base_color" in images:
        data = images["eye_base_color"]
        source = {"source": "texture"}
    else:
        pixels, source, bent = eye_texture(
            eye,
            eyed,
            model,
            images.get("eyes"),
            images.get("eyes_normal", images.get("normal")),
            want_normals=True,
            normal_box="eyes_normal" in images,
        )
        data = png_bytes(pixels)
    normal_texture = None
    if bent is not None and body.tangents is not None:
        # the reference's eye normals (a concave iris, a soft limbus) in the file's frames
        normal_texture = doc.add_texture(
            doc.add_image(
                png_bytes(_eye_normal_pixels(body, eyed, bent)),
                "image/png",
                "eye_normal",
            )
        )
    eyed.extras = {
        "radius_mm": [float(f.radius * 1000) for f in eyed.frames.values()],
        "bulge": settings["bulge"],
        "cornea": bool(eye.cornea),
        "caruncle": bool(eye.caruncle),
        **source,
    }
    texture = doc.add_texture(doc.add_image(data, mime_type(data), "eye_base_color"))
    out = []
    eyeball = pbr_material(
        "eyes", (1.0, 1.0, 1.0, 1.0), 0.0, eye.sclera_roughness, texture=texture
    )
    if eye.cornea:
        # under the cornea the eyeball sits in the eye's own medium: no reflection of its
        # own, which a renderer would draw as a rough, milky layer over the iris (the cornea
        # gives the one sharp highlight)
        eyeball["extensions"] = {"KHR_materials_specular": {"specularFactor": 0.0}}
    else:
        eyeball["extensions"] = {
            "KHR_materials_clearcoat": {
                "clearcoatFactor": 1.0,
                "clearcoatRoughnessFactor": float(eye.cornea_roughness),
            },
            "KHR_materials_ior": {"ior": float(eye.cornea_ior)},
        }
    if normal_texture is not None:
        eyeball["normalTexture"] = {"index": normal_texture}
    eyeball["extras"] = extras("eyeball")
    out.append((doc.add_material(eyeball), eyed.part_mask("eyeball")))
    if eye.cornea:
        cornea = pbr_material("cornea", (1.0, 1.0, 1.0, 1.0), 0.0, eye.cornea_roughness)
        cornea["extensions"] = {
            "KHR_materials_transmission": {"transmissionFactor": 1.0},
            "KHR_materials_volume": {"thicknessFactor": CHAMBER_DEPTH_MM / 1000.0},
            "KHR_materials_ior": {"ior": float(eye.cornea_ior)},
        }
        cornea["extras"] = extras("cornea")
        out.append((doc.add_material(cornea), eyed.part_mask("cornea")))
    pink_mask = eyed.part_mask("caruncle", "socket")
    # a fitted character on sculpt keeps its corner in the skin's material (eyed_mesh)
    if eye.caruncle and pink_mask.any():
        pink = pbr_material("conjunctiva", tuple(eye.caruncle_color) + (1.0,), 0.0, 0.5)
        pink["extensions"] = {
            "KHR_materials_clearcoat": {
                "clearcoatFactor": 0.8,
                "clearcoatRoughnessFactor": 0.08,
            }
        }
        pink["extras"] = extras("conjunctiva")
        out.append((doc.add_material(pink), pink_mask))
    return out


# the key of the teeth materials' settings in their extras
TEETH_EXTRAS = "opensculptboy_teeth"
TEETH_EXTRAS_FORMAT = "opensculptboy/teeth@1"


def _teeth_materials(doc: GltfDocument, toothed, teeth):
    """
    The materials and triangle masks of the teeth of :mod:`opensculptboy.teeth` (a
    :class:`~opensculptboy.teeth.mesh.ToothedMesh` behind the body): ``teeth``, the enamel's
    colour and roughness with a clear coat for the wet surface, and ``gums`` (generic teeth),
    pink and wet. Their extras hold :data:`TEETH_EXTRAS` (``opensculptboy/teeth@1``) with the
    ``role`` and the teeth's settings.
    """
    settings = teeth.extras()
    toothed.extras = dict(
        settings,
        points=int(len(toothed.teeth.points)),
        triangles=int(len(toothed.teeth.triangles)),
    )

    def extras(role):
        return {TEETH_EXTRAS: {"format": TEETH_EXTRAS_FORMAT, "role": role, **settings}}

    out = []
    enamel = pbr_material(
        "teeth", tuple(teeth.color) + (1.0,), 0.0, float(teeth.roughness)
    )
    enamel["extensions"] = {
        "KHR_materials_clearcoat": {
            "clearcoatFactor": 0.5,
            "clearcoatRoughnessFactor": 0.1,
        }
    }
    enamel["extras"] = extras("teeth")
    out.append((doc.add_material(enamel), toothed.part_mask("teeth")))
    gums = toothed.part_mask("gums")
    if gums.any():
        gum = pbr_material("gums", tuple(teeth.gum_color) + (1.0,), 0.0, 0.45)
        gum["extensions"] = {
            "KHR_materials_clearcoat": {
                "clearcoatFactor": 0.8,
                "clearcoatRoughnessFactor": 0.08,
            }
        }
        gum["extras"] = extras("gums")
        out.append((doc.add_material(gum), gums))
    return out


def _mouth_materials(doc: GltfDocument, toothed, mouth, detail_dir):
    """
    The material and triangle mask of a reference's inner mouth (:mod:`opensculptboy.mouth.inner`
    behind the body): ``mouth``, the reference's own maps (base colour, normal, roughness in
    G) with a clear coat for the wet surface, over every part (teeth, gums, palate, tongue,
    floor, throat). Its extras hold :data:`TEETH_EXTRAS` (``opensculptboy/teeth@1``) with the
    role ``mouth`` and the mouth's settings.
    """
    from opensculptboy.teeth.mesh import PARTS

    folder = pathlib.Path(detail_dir or ".")
    settings = mouth.extras()
    toothed.extras = dict(
        settings,
        points=int(len(toothed.teeth.points)),
        triangles=int(len(toothed.teeth.triangles)),
        hidden=int(toothed.hidden_mask().sum()),
    )
    textures = {}
    for slot, name in mouth.textures().items():
        data = (folder / name).read_bytes()
        textures[slot] = doc.add_texture(
            doc.add_image(data, mime_type(data), f"mouth_{slot}")
        )
    material = pbr_material(
        "mouth",
        tuple(mouth.color) + (1.0,)
        if "base_color" not in textures
        else (1.0, 1.0, 1.0, 1.0),
        0.0,
        float(mouth.roughness_factor) if "roughness" in textures else 0.35,
        texture=textures.get("base_color"),
    )
    if "roughness" in textures:
        material["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {
            "index": textures["roughness"]
        }
    if "normal" in textures:
        material["normalTexture"] = {"index": textures["normal"]}
    if mouth.clearcoat > 0:
        material["extensions"] = {
            "KHR_materials_clearcoat": {
                "clearcoatFactor": float(mouth.clearcoat),
                "clearcoatRoughnessFactor": 0.1,
            }
        }
    material["extras"] = {
        TEETH_EXTRAS: {"format": TEETH_EXTRAS_FORMAT, "role": "mouth", **settings}
    }
    parts = toothed.part_mask(*PARTS)
    return [(doc.add_material(material), parts)]


def _eye_normal_pixels(body, eyed, bent) -> np.ndarray:
    """The tangent-space normal map of the eye texture's layout that lays ``bent`` (the baked
    eyes' normals, :func:`opensculptboy.eyes.texture.carry_baked_normals`) on the file's
    eyeballs, in their MikkTSpace frames (``body.tangents``)."""
    from opensculptboy.eyes.texture import encode_normals

    triangles = body.triangles[eyed.part_mask("eyeball")]
    uvs = np.stack([body.uvs[:, 0], 1.0 - body.uvs[:, 1]], 1)  # OBJ convention
    # glTF frame -> Anny's (rows: v_anny = v_gltf @ C) -> each eye's frame
    surfaces = []
    for side, frame in eyed.frames.items():
        mine = (uvs[triangles, 0] < 0.5).all(1) == (side == "L")
        tris = triangles[mine]
        used = np.unique(tris)
        local = np.full(len(body.uvs), -1)
        local[used] = np.arange(len(used))
        to_eye = ANNY_TO_GLTF @ frame.axes
        normals = body.normals[used] @ to_eye
        tangents = np.c_[body.tangents[used, :3] @ to_eye, body.tangents[used, 3]]
        surfaces.append((uvs[used], local[tris], normals, tangents))
    return encode_normals(bent, surfaces)


# the key of the skin's subsurface settings and real-time skin maps in a material's extras
SKIN_EXTRAS = "opensculptboy_skin"
SKIN_EXTRAS_FORMAT = "opensculptboy/skin@1"


def _texture(doc: GltfDocument, images: dict, slot: str, cache: dict) -> int | None:
    """The texture of an image slot, written once per document."""
    if slot not in images:
        return None
    if slot not in cache:
        data = images[slot]
        cache[slot] = doc.add_texture(doc.add_image(data, mime_type(data), slot))
    return cache[slot]


def _skin_material(
    doc: GltfDocument,
    skin,
    images: dict,
    cache: dict,
    name: str = "skin",
    prefix: str = "",
    transform: dict | None = None,
) -> dict:
    """
    The glTF material of a card's skin (:class:`~opensculptboy.material.SkinMaterial`): the
    base colour, roughness and normal textures of ``images`` (slots ``prefix + "base_color"``
    and so on) behind ``transform`` (a ``KHR_texture_transform``) when given, the specular
    reflection (``KHR_materials_specular``, ``KHR_materials_ior``), and the subsurface
    settings with the thickness and curvature maps in the extras under :data:`SKIN_EXTRAS`
    (glTF has no ratified subsurface extension). ``cache`` maps the slots already written
    to their textures.
    """

    def info(slot):
        index = _texture(doc, images, slot, cache)
        if index is None:
            return None
        entry = {"index": index}
        if transform is not None:
            entry["extensions"] = {"KHR_texture_transform": dict(transform)}
        return entry

    pbr = {"baseColorFactor": [float(c) for c in skin.base_color_factor]}
    base = info(prefix + "base_color")
    if base is not None:
        pbr["baseColorTexture"] = base
    pbr["metallicFactor"] = 0.0
    rough = info(prefix + "roughness")
    # a roughness texture is multiplied by the factor; without one, the number is the value
    pbr["roughnessFactor"] = float(
        skin.roughness_factor if rough is not None else skin.roughness_value
    )
    if rough is not None:
        pbr["metallicRoughnessTexture"] = rough
    material = {"name": name, "pbrMetallicRoughness": pbr}
    normal = info(prefix + "normal")
    if normal is not None:
        normal["scale"] = float(skin.normal_scale)
        material["normalTexture"] = normal
    material["extensions"] = {
        "KHR_materials_specular": {
            "specularFactor": float(skin.specular),
            "specularColorFactor": [float(c) for c in skin.specular_color],
        },
        "KHR_materials_ior": {"ior": float(skin.ior)},
    }
    extras = {"format": SKIN_EXTRAS_FORMAT, "subsurface": skin.subsurface_extras()}
    for slot, scale_key, scale in (
        ("thickness", "max_mm", skin.thickness_mm),
        ("curvature", "max_per_m", skin.curvature_per_m),
    ):
        # the head tile keeps anny's UVs and reads the atlas's maps; the ear tile has its own
        source = slot
        if prefix + slot in images or prefix == "ears_":
            source = prefix + slot
        index = _texture(doc, images, source, cache)
        if index is not None:
            extras[slot] = {"index": index, "texCoord": 0, scale_key: float(scale)}
    material["extras"] = {SKIN_EXTRAS: extras}
    return material


def _head_material(
    doc: GltfDocument, model, body, skin, images: dict, cache: dict, topology: str
):
    """
    The material of the head tile (:mod:`opensculptboy.texture.tiles`): the skin's settings
    with the head tile's textures behind the tile's ``KHR_texture_transform``, and the mask
    of the head tile's triangles; (None, None) when the card has no head tile textures.
    """
    from opensculptboy.export.body import head_tile_triangles
    from opensculptboy.texture.tiles import head_texture_transform

    if not any(slot.startswith("head_") for slot in images):
        return None, None
    transform = head_texture_transform(model, topology)
    mask = head_tile_triangles(model, body, topology)
    if transform is None or mask is None or not mask.any():
        return None, None
    material = _skin_material(doc, skin, images, cache, "head", "head_", transform)
    # the slots that the head tile lacks fall back to the atlas, without the transform
    for slot, key in (
        ("base_color", "baseColorTexture"),
        ("roughness", "metallicRoughnessTexture"),
    ):
        if "head_" + slot not in images and slot in images:
            material["pbrMetallicRoughness"][key] = {
                "index": _texture(doc, images, slot, cache)
            }
    if "head_normal" not in images and "normal" in images:
        material["normalTexture"] = {
            "index": _texture(doc, images, "normal", cache),
            "scale": float(skin.normal_scale),
        }
    return doc.add_material(material), mask


def _textured_material(doc: GltfDocument, images: dict) -> dict:
    """The skin material with the baked base colour and normal textures of ``images``."""
    texture = None
    if "base_color" in images:
        data = images["base_color"]
        texture = doc.add_texture(doc.add_image(data, mime_type(data), "base_color"))
    material = pbr_material("skin", (1.0, 1.0, 1.0, 1.0), 0.0, 0.6, texture=texture)
    if "normal" in images:
        data = images["normal"]
        normal = doc.add_texture(doc.add_image(data, mime_type(data), "normal"))
        material["normalTexture"] = {"index": normal}
    return material


def _animation(
    doc: GltfDocument, model, character: Character, name: str, parents, default_local
) -> dict:
    """One glTF animation for a pose or a clip of ``anny.poses``."""
    import anny.poses

    entry = anny.poses.pose_parameters(
        model,
        name,
        phenotype_kwargs=dict(character.phenotype) or None,
        local_changes_kwargs=dict(character.local_changes) or None,
    )
    return _keyframes(
        doc,
        model,
        character,
        name,
        entry["pose_parameters"],
        float(entry["fps"]),
        parents,
        default_local,
    )


def _jaw_open_animation(
    doc,
    jaw_open,
    labels,
    default_local,
    mesh_node,
    targets,
    name: str | None = None,
    corrective: str | None = None,
) -> dict:
    """The animation ``jawOpen.jaw``: over one second the jaw turns by the turn of
    :class:`~opensculptboy.mouth.jaw.JawOpen` and the target ``jawOpen.corrective`` goes from 0
    to 1, which together give anny's jawOpen; the other targets keep their weights."""
    from opensculptboy.mouth.jaw import CORRECTIVE, JAW_ANIMATION

    name = name or JAW_ANIMATION
    corrective = corrective or CORRECTIVE
    jaw = labels.index("jaw")
    C = ANNY_TO_GLTF
    local_turn = C @ jaw_open.local_matrix() @ C.T
    rest = default_local[jaw, :3, :3]
    quats = _quaternions(np.stack([rest, rest @ local_turn]))
    if np.dot(quats[0], quats[1]) < 0:
        quats[1] = -quats[1]
    times = doc.accessor(np.array([0.0, 1.0], np.float32), minmax=True)
    weights = np.array([[t.weight for t in targets]] * 2, dtype=np.float32)
    names = [t.name for t in targets]
    weights[:, names.index(corrective)] = [0.0, 1.0]
    samplers = [
        {
            "input": times,
            "output": doc.accessor(quats.astype(np.float32)),
            "interpolation": "LINEAR",
        },
        {
            "input": times,
            "output": doc.accessor(weights.reshape(-1)),
            "interpolation": "LINEAR",
        },
    ]
    channels = [
        {"sampler": 0, "target": {"node": jaw, "path": "rotation"}},
        {"sampler": 1, "target": {"node": mesh_node, "path": "weights"}},
    ]
    return {"name": name, "samplers": samplers, "channels": channels}


def _character_pose(
    doc: GltfDocument, model, character: Character, parents, default_local
) -> dict:
    """The character's own pose as a one-frame animation named ``pose``, on the floor."""
    import anny.poses

    params, _ = anny.poses.ground(
        model,
        character.pose_parameters(model),
        phenotype_kwargs=dict(character.phenotype) or None,
        local_changes_kwargs=dict(character.local_changes) or None,
    )
    return _keyframes(
        doc, model, character, "pose", params, 1.0, parents, default_local
    )


def _keyframes(
    doc: GltfDocument,
    model,
    character: Character,
    name: str,
    pose_parameters: torch.Tensor,
    fps: float,
    parents,
    default_local,
) -> dict:
    """One glTF animation for ``local-ref`` pose parameters (frames, bones, 4, 4)."""
    if is_mouth_rig(character.rig):  # the mouth stays shut unless the pose moves it
        from opensculptboy.mouth.rig import hold_mouth_bones

        pose_parameters = hold_mouth_bones(model, character, pose_parameters)
    with torch.no_grad():
        out = model(
            pose_parameters=pose_parameters,
            pose_parameterization="local-ref",
            **character.model_kwargs(),
        )
    world = _to_gltf_matrices(out["bone_poses"].double().cpu().numpy())
    local = _local_transforms(world, parents)
    frames = local.shape[0]
    times = (np.arange(frames) / fps).astype(np.float32)
    time_accessor = doc.accessor(times, minmax=True)
    samplers, channels = [], []
    for j in range(len(parents)):
        quats = _quaternions(local[:, j, :3, :3])
        for f in range(1, frames):  # the shortest path between keyframes
            if np.dot(quats[f], quats[f - 1]) < 0:
                quats[f] = -quats[f]
        tracks = [("rotation", quats)]
        translations = local[:, j, :3, 3]
        if np.abs(translations - default_local[j, :3, 3]).max() > 1e-6:
            tracks.append(("translation", translations))
        for path, values in tracks:
            samplers.append(
                {
                    "input": time_accessor,
                    "output": doc.accessor(values.astype(np.float32)),
                    "interpolation": "LINEAR",
                }
            )
            channels.append(
                {"sampler": len(samplers) - 1, "target": {"node": j, "path": path}}
            )
    return {"name": name, "samplers": samplers, "channels": channels}


def read_character(path: str | pathlib.Path) -> Character:
    """
    The character stored in a GLB or VRM file written by OpenSculptBoy (``export_glb`` or
    ``export_vrm``): the ``opensculptboy`` entry of the extras of a scene, of the root object
    or of the asset.
    """
    gltf = read_gltf_json(path)
    holders = [scene.get("extras") for scene in gltf.get("scenes", [])]
    holders += [gltf.get("extras"), gltf.get("asset", {}).get("extras")]
    for extras in holders:
        metadata = extras.get("opensculptboy") if isinstance(extras, dict) else None
        if isinstance(metadata, dict) and "character" in metadata:
            return Character.from_dict(metadata["character"])
    raise ValueError(
        f"{path} holds no OpenSculptBoy character (no 'opensculptboy' entry in its extras)."
    )


def _face_shape_names() -> list[str]:
    from anny.models.face_shapes import face_shape_spec

    return [p.name for p in face_shape_spec()]


def _anny_version() -> str:
    import anny

    return anny.__version__

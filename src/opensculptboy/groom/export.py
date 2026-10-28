# OpenSculptBoy
# Apache License, Version 2.0
"""
The grooms of a card in the GLB and VRM exports.

- The hair (:func:`hair_pieces`) is a mesh of its own, written as the dressed pieces of
  :mod:`opensculptboy.wearables` are: a :class:`~opensculptboy.wearables.fit.DressedPiece`
  bound rigidly to the head and skinned with the hair's own weights on Anny's bones, its vertex
  colours as ``COLOR_0`` in GLB files, and one ``chain`` spring per chain of the reference's
  chain joints (:data:`HAIR_PRESET`), whose joints the exports write as spring chains
  (``VRMC_springBone`` in VRM 1.0 files, ``secondaryAnimation`` in VRM 0.x). MToon draws no
  vertex colours in every app, so in VRM files each hair material takes the mean colour of its
  vertices as its base colour, and the blended cards write depth after the cap
  (``transparentWithZWrite``, a render queue offset per material in the groom's order).
- The brows (:func:`brow_parts`) are parts of the eyed mesh of :mod:`opensculptboy.eyes.mesh`:
  each brow point takes the closest point of the skin under it at rest (a triangle and its
  barycentric coordinates) and keeps its offset from that point, so it follows the skin's morph
  targets, its skin weights and the T-pose of a VRM file, as the lashes do.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

from opensculptboy.eyes.mesh import EyePart
from opensculptboy.groom.data import GroomData

# the spring of the hair's chains: short hair that keeps its shape and sways a little
HAIR_PRESET = "hair"
# a point weighs on the chains when this share of its reference weight lies on chain joints
CHAIN_SHARE = 0.05
# the material extras of the grooms in GLB files
GROOM_EXTRAS = "opensculptboy_groom"


def card_groom(character, kind: str, detail_dir=None):
    """The groom of a card's ``material.<kind>`` and its textures as bytes: (GroomData,
    {material: {slot: bytes}}), or None."""
    section = character.groom_section(kind)
    if section is None:
        return None
    folder = pathlib.Path(detail_dir or ".")
    data = GroomData.load(folder / section.mesh)
    textures = {
        name: {slot: (folder / path).read_bytes() for slot, path in slots.items()}
        for name, slots in section.textures.items()
    }
    return data, textures


def garment_materials(data: GroomData, textures: dict, vrm: bool = False) -> list:
    """The groom's materials as :class:`~opensculptboy.wearables.format.Material` items: for
    GLB files as the reference draws them (with the vertex colours), for VRM files with the
    mean vertex colour folded into the base colour and the MToon settings of the module
    documentation."""
    from opensculptboy.wearables.format import Material

    out = []
    for k, m in enumerate(data.materials):
        base = np.asarray(m.base_color, dtype=np.float64)
        if vrm:
            base = base * data.mean_color(k)
        toon = {}
        if vrm and m.alpha_mode == "BLEND":
            toon = {"transparent_with_zwrite": True, "render_queue_offset": min(k, 9)}
        out.append(
            Material(
                name=m.name,
                base_color=tuple(float(c) for c in base),
                metallic=0.0,
                roughness=float(m.roughness),
                alpha_mode=m.alpha_mode,
                alpha_cutoff=float(m.alpha_cutoff),
                double_sided=bool(m.double_sided),
                normal_scale=float(m.normal_scale),
                textures=dict(textures.get(m.name, {})),
                extras={GROOM_EXTRAS: {"kind": data.kind, "material": m.name}},
                toon=toon,
            )
        )
    return out


def hair_springs(data: GroomData) -> list:
    """The ``springs`` of the hair's package: one ``chain`` spring per chain, its points in
    glTF axes, hanging from the head and reaching the points of the group ``chain``."""
    from opensculptboy.wearables.springs import GLTF_TO_ANNY

    out = []
    for k, chain in enumerate(data.chains):
        out.append(
            {
                "name": f"chain{k}",
                "kind": "chain",
                "preset": HAIR_PRESET,
                "points": (np.asarray(chain) @ GLTF_TO_ANNY).round(7).tolist(),
                "parent": "head",
                "groups": ["chain"],
            }
        )
    return out


@dataclasses.dataclass
class GroomPiece:
    """The marker of a dressed piece that is a groom: what its node's extras record."""

    kind: str
    materials: list
    chains: int

    def extras(self) -> dict:
        return {
            "format": "opensculptboy/groom@1",
            "kind": self.kind,
            "source": "reference",
            "materials": list(self.materials),
            "chains": int(self.chains),
        }


def hair_pieces(character, body, detail_dir=None, max_influences=4, vrm=False) -> list:
    """
    The card's hair as dressed pieces on ``body`` (the export's
    :func:`~opensculptboy.wearables.fit.card_body`): one piece ``hair``, or none for a card
    without ``material.hair``. ``vrm`` gives the materials of a VRM file.
    """
    from opensculptboy.wearables.bind import body_weights, evaluate
    from opensculptboy.wearables.fit import DressedPiece
    from opensculptboy.wearables.format import PRIVATE, Binding, Garment, Wearable

    found = card_groom(character, "hair", detail_dir)
    if found is None:
        return []
    data, textures = found
    missing = sorted(set(data.bones) - set(body.bone_labels))
    if missing:
        raise ValueError(f"The card's rig lacks the hair's bones {missing}.")
    chain = data.chain_share > CHAIN_SHARE
    group = chain[data.triangles].any(1).astype(np.int64)
    garment = Garment(
        vertices=data.points,
        triangles=data.triangles,
        normals=data.normals,
        uvs=data.uvs,
        triangle_material=data.material,
        triangle_group=group,
        materials=garment_materials(data, textures, vrm),
        groups=["main", "chain"],
    )
    head = body.bone_poses[body.bone_labels.index("head")]
    R, t = head[:3, :3], head[:3, 3]
    binding = Binding(
        method="rigid",
        body="card",
        body_vertices=body.vertices.copy(),
        bone="head",
        local=(data.points - t) @ R,
        normal=data.normals @ R,
        skin_weights=data.bone_weights,
        skin_joints=data.bone_slots,
        joint_names=list(data.bones),
    )
    wearable = Wearable(
        name="hair",
        slot="accessory",
        layer=0,
        garment=garment,
        binding=binding,
        provenance=PRIVATE,
        weights="own",
        springs=hair_springs(data) if data.chains else [],
        label="Hair (reference)",
    )
    vertices, normals = evaluate(binding, body)
    weights, joints = body_weights(binding, body, max_influences)
    piece = DressedPiece(
        ref="hair",
        piece="hair",
        outfit=None,
        wearable=wearable,
        vertices=vertices,
        normals=normals,
        weights=weights,
        joints=joints,
        colors=None if vrm or data.colors is None else data.colors,
        groom=GroomPiece("hair", [m.name for m in data.materials], len(data.chains)),
    )
    return [piece]


# ------------------------------------------------------------------------------ the brows
@dataclasses.dataclass
class BrowPart(EyePart):
    """A part of the brows on an eyed mesh: an :class:`~opensculptboy.eyes.mesh.EyePart` whose
    points lie ``shift`` (M, 3) off the barycentric points of its anchors, and the groom
    material of its triangles."""

    shift: np.ndarray | None = None
    material: int = 0

    def positions(self, x: np.ndarray) -> np.ndarray:
        return self.operator @ x[self.anchors] + self.shift


def brow_parts(data: GroomData, rest, skin_triangles, frames) -> list:
    """
    The brows as parts of an eyed mesh (see the module documentation), one per eye's side and
    material: ``rest`` (N, 3) the mesh's rest vertices, ``skin_triangles`` its skin
    triangles, and ``frames`` the eyes' frames (side to frame), whose nearer centre gives each
    brow point its side.
    """
    from anny.hair import closest_triangles

    rest = np.asarray(rest, dtype=np.float64)
    tris = np.asarray(skin_triangles, dtype=np.int64)
    if len(data.points) == 0 or len(tris) == 0:
        return []
    tri, bary, _ = closest_triangles(data.points, rest, tris)
    corners = tris[tri]  # (M, 3)
    sides = list(frames)
    dist = np.stack(
        [np.linalg.norm(data.points - frames[s].centre, axis=1) for s in sides], 1
    )
    side_of = np.asarray(sides)[np.argmin(dist, axis=1)]
    uvs = np.c_[data.uvs[:, 0], 1.0 - data.uvs[:, 1]]  # OBJ, as the mesh's
    parts = []
    for side in sides:
        for k in range(len(data.materials)):
            pick = data.triangles[
                (data.material == k) & (side_of[data.triangles[:, 0]] == side)
            ]
            if len(pick) == 0:
                continue
            rows = np.unique(pick)
            local = np.full(len(data.points), -1)
            local[rows] = np.arange(len(rows))
            anchors, inverse = np.unique(corners[rows], return_inverse=True)
            inverse = np.asarray(inverse).reshape(len(rows), 3)
            operator = np.zeros((len(rows), len(anchors)))
            np.add.at(
                operator,
                (np.repeat(np.arange(len(rows)), 3), inverse.ravel()),
                bary[rows].ravel(),
            )
            shift = data.points[rows] - operator @ rest[anchors]
            nearest = anchors[inverse[np.arange(len(rows)), np.argmax(bary[rows], 1)]]
            parts.append(
                BrowPart(
                    name="brows",
                    side=side,
                    rest=data.points[rows],
                    normals=data.normals[rows],
                    uvs=uvs[rows],
                    triangles=local[pick],
                    anchors=anchors,
                    operator=operator,
                    blend=operator.copy(),
                    source=nearest,
                    points=rows,
                    shift=shift,
                    material=k,
                )
            )
    return parts


def brow_masks(eyed) -> dict:
    """The triangle mask of each brow material on an eyed mesh's triangles (with the
    triangles that later stages append), by material index."""
    out: dict = {}
    total = len(eyed.triangles) + eyed.trailing_triangles
    start = int(eyed.kept.sum())
    for p in eyed.parts:
        if p.name == "brows":
            mask = out.setdefault(p.material, np.zeros(total, bool))
            mask[start : start + len(p.triangles)] = True
        start += len(p.triangles)
    return out

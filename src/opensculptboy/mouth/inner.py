# OpenSculptBoy
# Apache License, Version 2.0
"""
The inner mouth of a reference character, in a fitted card's rest frame.

An appearance handoff may describe the whole inside of the reference's mouth
(:mod:`opensculptboy.mouth.handoff`): the gums with the back teeth, the palate, the tongue,
the floor of the mouth, the throat and the separate front teeth, each with the skin weights
of the reference's face rig. :func:`reference_mouth` takes them into the card's rest frame by
the rigid motion (Kabsch) that takes the fitted head, as the fit placed and posed it, back to
its rest pose, and maps their skin onto the ``anny-mouth`` rig (:mod:`opensculptboy.mouth.rig`):
the reference's jaw joint and the joints of the chin and the lower lip become ``jaw``, its
tongue chain becomes anny's tongue chain ``tongue00`` to ``tongue04`` by its place along the
chain, and every other joint becomes ``head``.

The result is a :class:`MouthData`, a :class:`~opensculptboy.teeth.mesh.TeethData` with the
skin weights and the flag ``interior``: the exporters append it as they append teeth
(:func:`opensculptboy.teeth.mesh.toothed_mesh`), give its vertices these weights on a rig
that has the mouth bones, and hide anny's own mouth cavity and tongue behind it.
"""

from __future__ import annotations

import dataclasses
import json
import re

import numpy as np

from opensculptboy.teeth.mesh import PARTS, TeethData, rigid

FORMAT = "opensculptboy/inner-mouth@1"
# anny's tongue chain along its middle, from the root to the tip
TONGUE_CHAIN = ("tongue00", "tongue01", "tongue02", "tongue03", "tongue04")
# the reference's joints that move with the lower jaw
_JAW = re.compile(r"jaw|chin|lowerlip|lower_lip", re.IGNORECASE)
_TONGUE = re.compile(r"tongue[_ ]?(\d+)", re.IGNORECASE)


@dataclasses.dataclass
class MouthData(TeethData):
    """A reference's inner mouth in a card's rest frame: the teeth's points, normals, UVs,
    triangles, arch of each point and part of each triangle (``teeth``, ``gums``,
    ``palate``, ``tongue``, ``floor``, ``throat``), with each point's skin on the
    ``anny-mouth`` rig (``bones`` names, ``bone_slots`` indices into it, ``bone_weights``)."""

    bones: tuple = ()
    bone_slots: np.ndarray | None = None  # (M, K) int
    bone_weights: np.ndarray | None = None  # (M, K)
    interior: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.bones = tuple(str(b) for b in self.bones)
        if self.bone_slots is not None:
            self.bone_slots = np.asarray(self.bone_slots, dtype=np.int64)
            self.bone_weights = np.asarray(self.bone_weights, dtype=np.float64)

    def save(self, path) -> None:
        from safetensors.numpy import save_file

        tensors = {
            "points": self.points,
            "normals": self.normals,
            "uvs": self.uvs,
            "triangles": self.triangles,
            "arch": self.arch,
            "part": self.part,
        }
        if self.bone_slots is not None:
            tensors["bone_slots"] = self.bone_slots
            tensors["bone_weights"] = self.bone_weights
        save_file(
            tensors,
            str(path),
            metadata={
                "format": FORMAT,
                "bones": json.dumps(list(self.bones)),
                "parts": json.dumps(list(PARTS)),
                "interior": json.dumps(bool(self.interior)),
            },
        )

    @classmethod
    def load(cls, path) -> "MouthData":
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as f:
            meta = f.metadata() or {}
            if meta.get("format") != FORMAT:
                raise ValueError(f"{path} is not a mouth file ({FORMAT}).")
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        return cls(
            **tensors,
            bones=tuple(json.loads(meta.get("bones", "[]"))),
            interior=bool(json.loads(meta.get("interior", "true"))),
        )

    def part_points(self, *names: str) -> np.ndarray:
        """The mask of the points on the triangles of the named parts."""
        mask = np.zeros(len(self.points), bool)
        for name in names:
            mask[np.unique(self.triangles[self.part == PARTS.index(name)])] = True
        return mask


def anny_bone(joint: str, tongue_joints: list[str]) -> str:
    """The ``anny-mouth`` bone of a reference joint (see the module documentation)."""
    if joint in tongue_joints:
        k = tongue_joints.index(joint)
        last = max(len(tongue_joints) - 1, 1)
        return TONGUE_CHAIN[int(round(k * (len(TONGUE_CHAIN) - 1) / last))]
    if _JAW.search(joint.replace(" ", "")):
        return "jaw"
    return "head"


def _tongue_joints(names) -> list[str]:
    """The reference's tongue chain from the root, by the number in each name."""
    found = []
    for name in names:
        m = _TONGUE.search(name)
        if m:
            found.append((int(m.group(1)), name))
    return [name for _, name in sorted(set(found))]


def map_skin(part, k: int = 4) -> tuple[list[str], np.ndarray, np.ndarray]:
    """A mouth part's skin on the ``anny-mouth`` rig: (bone names, (V, k) slots into them,
    (V, k) weights summing to 1). A part without a skin follows the jaw when its piece does,
    else the head."""
    from opensculptboy.mouth.handoff import piece_arch

    n = len(part.vertices)
    if part.joints is None or part.joint_slots is None:
        bone = "jaw" if piece_arch(part.mouth_role or "") else "head"
        slots = np.zeros((n, k), np.int64)
        weights = np.zeros((n, k))
        weights[:, 0] = 1.0
        return [bone], slots, weights
    names = list(part.joints)
    tongue = _tongue_joints(names)
    mapped = [anny_bone(name, tongue) for name in names]
    bones = sorted(set(mapped))
    index = {b: i for i, b in enumerate(bones)}
    target = np.array([index[b] for b in mapped])
    dense = np.zeros((n, len(bones)))
    np.add.at(
        dense,
        (
            np.repeat(np.arange(n), part.joint_slots.shape[1]),
            target[part.joint_slots].ravel(),
        ),
        part.joint_weights.ravel(),
    )
    order = np.argsort(-dense, axis=1, kind="stable")[:, :k]
    weights = np.take_along_axis(dense, order, 1)
    if weights.shape[1] < k:
        pad = k - weights.shape[1]
        weights = np.pad(weights, ((0, 0), (0, pad)))
        order = np.pad(order, ((0, 0), (0, pad)))
    total = weights.sum(1, keepdims=True)
    empty = total[:, 0] <= 0
    weights = np.where(total > 0, weights / np.where(total > 0, total, 1.0), 0.0)
    if empty.any():  # a vertex without weights follows its piece's bone
        fallback = index.get("jaw" if piece_arch(part.mouth_role or "") else "head", 0)
        order[empty] = fallback
        weights[empty] = 0.0
        weights[empty, 0] = 1.0
    return bones, order, weights


def reference_mouth(parts, posed: np.ndarray, rest: np.ndarray, head: np.ndarray):
    """
    The inner mouth of a reference (``parts``: the parts of role ``mouth`` whose
    ``mouth_role`` the handoff's ``mouth`` section names) in the rest frame of a fitted mesh,
    as a :class:`MouthData`, or None without such parts. ``posed`` and ``rest`` are the
    fitted mesh's vertices as the fit placed it on the reference and at rest, and ``head`` the
    rows of its head (:func:`opensculptboy.teeth.mesh.mouth_rows`).
    """
    from opensculptboy.export.body import vertex_normals
    from opensculptboy.mouth.handoff import piece_arch, piece_part

    pieces = [p for p in parts if getattr(p, "mouth_role", None) and len(p.triangles)]
    pieces = [p for p in pieces if piece_part(p.mouth_role) is not None]
    if not pieces:
        return None
    R, t = rigid(np.asarray(posed)[head], np.asarray(rest)[head])
    points, normals, uvs, triangles, arch, part = [], [], [], [], [], []
    skins = []
    count = 0
    for p in pieces:
        v = np.asarray(p.vertices, dtype=np.float64)
        tri = np.asarray(p.triangles, dtype=np.int64)
        n = p.normals if p.normals is not None else vertex_normals(v, tri)
        points.append(v @ R.T + t)
        normals.append(np.asarray(n, dtype=np.float64) @ R.T)
        uvs.append(
            np.c_[p.uvs[:, 0], 1.0 - p.uvs[:, 1]]
            if p.uvs is not None
            else np.zeros((len(v), 2))
        )
        triangles.append(tri + count)
        arch.append(np.full(len(v), piece_arch(p.mouth_role), np.uint8))
        part.append(np.full(len(tri), PARTS.index(piece_part(p.mouth_role)), np.uint8))
        skins.append(map_skin(p))
        count += len(v)
    bones = sorted({b for names, _, _ in skins for b in names})
    index = {b: i for i, b in enumerate(bones)}
    slots = np.concatenate(
        [np.array([index[b] for b in names])[s] for names, s, _ in skins]
    )
    weights = np.concatenate([w for _, _, w in skins])
    normals = np.concatenate(normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return MouthData(
        points=np.concatenate(points),
        normals=normals,
        uvs=np.concatenate(uvs),
        triangles=np.concatenate(triangles),
        arch=np.concatenate(arch),
        part=np.concatenate(part),
        bones=tuple(bones),
        bone_slots=slots,
        bone_weights=weights,
    )


def fitted_meshes(card_path, reference, subdiv: int | None = None):
    """The fitted mesh of a card at a level of its detail layer (its finest by default):
    (model, detail topology, vertices as the fit placed it on the reference, rest vertices),
    from the card's offsets and placement (``detail.fit.placement``), as the bake places it."""
    import json
    import pathlib

    import torch

    from opensculptboy.character import Character
    from opensculptboy.detail import DetailModel, character_offsets

    card_path = pathlib.Path(card_path)
    character = Character.from_dict(json.loads(card_path.read_text()))
    placement = character.detail.get("fit", {}).get("placement")
    if not placement:
        raise ValueError(f"{card_path} holds no fit placement (detail.fit.placement).")
    levels = character.detail_levels()
    subdiv = max(levels) if subdiv is None else subdiv
    offsets = character_offsets(character, subdiv, card_path.parent)
    model = character.build_model(dtype=torch.float64)
    detailed = DetailModel(model, subdiv, offsets, "full", character.topology)
    with torch.no_grad():
        out = detailed(
            pose_parameters=character.pose_parameters(model), **character.model_kwargs()
        )
    R = np.asarray(placement["rotation"], np.float64)
    t = np.asarray(placement["translation"], np.float64)
    posed = out["vertices"][0].numpy() @ R.T + t
    return model, detailed.topology, posed, out["rest_vertices"][0].numpy()

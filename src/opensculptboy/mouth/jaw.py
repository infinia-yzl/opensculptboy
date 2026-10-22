# OpenSculptBoy
# Apache License, Version 2.0
"""
Anny's mouth-open target as a turn of the jaw bone and a corrective shape.

Anny opens the mouth with the facial action ``jawOpen``: a fixed offset per vertex. On the
``anny-mouth`` rig (:mod:`opensculptboy.mouth.rig`) the same opening can come from the jaw:
:func:`fit_jaw_open` finds the turn ``R`` of the jaw bone about its head that reproduces the
target best by linear blend skinning with the rig's own weights (the jaw's and those of the
tongue chain, which hangs from the jaw), in the least-squares sense over the vertices that
either moves; with ``translate`` it also fits a shift ``t`` of the head, which the humanoid jaw
of a VRM file cannot take. The rest of the target is the corrective shape:

    v + jawOpen(v) = u + w_v ((R - I)(u - p) + t),  u = v + corrective(v),

with ``p`` the head of the jaw bone at rest and ``w_v`` the weight of the jaw and the tongue on
vertex ``v``: the corrective is a rest-frame offset, as a morph target is, and the skinned turn
of the corrected rest mesh is anny's open mouth exactly. :class:`JawOpen` reports how much of
the target the turn carries.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from opensculptboy.mouth.rig import MOUTH_BONES

JAW_OPEN = "jawOpen"
# The morph target of the GLB export that completes the jaw's turn to anny's jawOpen.
CORRECTIVE = "jawOpen.corrective"
# The animation of the GLB export that turns the jaw and plays the corrective.
JAW_ANIMATION = "jawOpen.jaw"


@dataclasses.dataclass
class JawOpen:
    """The jaw-open turn of a body on the ``anny-mouth`` rig (Anny's frame, metres)."""

    rotation: np.ndarray  # (3, 3) the turn R in Anny's world frame
    translation: np.ndarray  # (3,) the shift t of the jaw's head
    pivot: np.ndarray  # (3,) the head p of the jaw bone at rest
    local_rotation: (
        np.ndarray
    )  # (4,) the turn in the jaw bone's rest frame, quaternion xyzw
    corrective: np.ndarray  # (V, 3) the rest-frame offsets that complete the turn
    target: np.ndarray  # (V, 3) anny's jawOpen offsets
    residual: np.ndarray  # (V, 3) the target minus the skinned turn of the rest mesh

    def local_matrix(self) -> np.ndarray:
        """(3, 3) the turn in the jaw bone's rest frame."""
        return Rotation.from_quat(self.local_rotation).as_matrix()

    def extras(self) -> dict:
        """What a GLB file records of the turn (``opensculptboy.mouth`` in the scene extras)."""
        return dict(
            format="opensculptboy/mouth@1",
            jaw_open=dict(
                bone="jaw",
                local_rotation=[round(float(q), 8) for q in self.local_rotation],
                corrective=CORRECTIVE,
                animation=JAW_ANIMATION,
                **self.stats(),
            ),
        )

    @property
    def angle_degrees(self) -> float:
        return float(np.degrees(Rotation.from_matrix(self.rotation).magnitude()))

    def stats(self) -> dict:
        """The sizes of the target and of what the turn leaves of it, in millimetres, over
        the vertices that the target moves by more than 0.01 mm."""
        moved = np.linalg.norm(self.target, axis=1) > 1e-5
        t = np.linalg.norm(self.target[moved], axis=1) * 1000
        c = np.linalg.norm(self.residual[moved], axis=1) * 1000
        return dict(
            angle_degrees=round(self.angle_degrees, 3),
            translation_mm=round(float(np.linalg.norm(self.translation) * 1000), 3),
            moved_vertices=int(moved.sum()),
            target_mean_mm=round(float(t.mean()), 3),
            target_max_mm=round(float(t.max()), 3),
            residual_mean_mm=round(float(c.mean()), 3),
            residual_p95_mm=round(float(np.percentile(c, 95)), 3),
            residual_max_mm=round(float(c.max()), 3),
            # the share of the target's squared length that the turn carries
            explained=round(float(1 - (c**2).sum() / max((t**2).sum(), 1e-30)), 4),
        )


def jaw_weights(model, max_influences: int | None = None) -> np.ndarray:
    """(V,) the weight of the jaw and of the tongue chain on each vertex of ``model``; with
    ``max_influences``, of the strongest bones that a file keeps, renormalised as the
    exporters keep them."""
    labels = list(model.bone_labels)
    mouth = [labels.index(bone) for bone in MOUTH_BONES]
    w = model.vertex_bone_weights.detach().cpu().double().numpy()
    i = model.vertex_bone_indices.detach().cpu().numpy()
    if max_influences is not None:
        from opensculptboy.export.body import top_skin_weights

        w, i, _ = top_skin_weights(w, i, max_influences)
        w = w.astype(np.float64)
    return np.where(np.isin(i, mouth), w, 0.0).sum(axis=1)


def skinned_turn(vertices, weights, rotation, translation, pivot) -> np.ndarray:
    """(V, 3) the offsets that the turn gives the rest vertices by linear blend skinning."""
    moved = (vertices - pivot) @ (rotation - np.eye(3)).T + translation
    return weights[:, None] * moved


def corrective_for(
    vertices, target, weights, rotation, translation, pivot
) -> np.ndarray:
    """(V, 3) the corrective offsets ``c`` that a file adds before skinning, so that the
    skinned turn of the corrected vertices gives the target: ``v + c`` turned with weight
    ``w`` is ``v + target``. Per vertex ``(I + w (R - I)) c = target - turn(v)``."""
    rest = target - skinned_turn(vertices, weights, rotation, translation, pivot)
    A = np.eye(3)[None] + weights[:, None, None] * (rotation - np.eye(3))[None]
    return np.linalg.solve(A, rest[..., None])[..., 0]


def turn_for(vertices, target, weights, pivot, translate: bool = False):
    """The turn ``R`` (3, 3) and the shift ``t`` (3,) of the jaw that fit the offsets
    ``target`` (V, 3) of the rest vertices (V, 3) best by linear blend skinning with the
    weights (V,) of the jaw, about its head ``pivot``. Without ``translate`` the jaw only
    turns (``t`` is 0), as the humanoid jaw of a VRM file does."""
    use = (weights > 0) | (np.linalg.norm(target, axis=1) > 1e-7)
    count = 6 if translate else 3

    def residual(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        t = x[3:] if translate else np.zeros(3)
        fit = skinned_turn(vertices[use], weights[use], R, t, pivot)
        return (fit - target[use]).ravel()

    x = least_squares(residual, np.zeros(count), method="lm").x
    return Rotation.from_rotvec(x[:3]).as_matrix(), (
        x[3:] if translate else np.zeros(3)
    )


def jaw_open(vertices, target, weights, bone_pose, translate: bool = False) -> JawOpen:
    """The :class:`JawOpen` of rest vertices (V, 3) and anny's jawOpen offsets (V, 3), for the
    jaw's weights (V,) and its rest pose (4, 4)."""
    pivot = bone_pose[:3, 3]
    R, t = turn_for(vertices, target, weights, pivot, translate)
    corrective = corrective_for(vertices, target, weights, R, t, pivot)
    residual = target - skinned_turn(vertices, weights, R, t, pivot)
    frame = bone_pose[:3, :3]
    local = Rotation.from_matrix(frame.T @ R @ frame).as_quat()
    return JawOpen(R, t, pivot, local, corrective, target, residual)


def fit_jaw_open(model, character=None, translate: bool = False) -> JawOpen:
    """
    The jaw-open turn of ``character``'s body (the default body when None) on ``model``, a
    model of the ``anny-mouth`` rig with the facial action ``jawOpen``.
    """
    import torch

    from opensculptboy.character import Character

    character = character or Character(rig="anny-mouth")
    labels = list(model.facial_action_labels)
    if JAW_OPEN not in labels:
        raise ValueError(f"The model has no facial action {JAW_OPEN!r}.")
    if "jaw" not in model.bone_labels:
        raise ValueError("The model has no jaw bone; build it on the 'anny-mouth' rig.")
    dtype = model.template_vertices.dtype
    rows = torch.zeros((2, len(labels)), dtype=dtype)
    rows[1, labels.index(JAW_OPEN)] = 1.0
    kwargs = character.model_kwargs()
    kwargs["facial_actions"] = rows
    if kwargs.get("face_shape_kwargs") is not None:
        kwargs["face_shape_kwargs"] = {
            k: torch.full((2,), float(v), dtype=dtype)
            for k, v in kwargs["face_shape_kwargs"].items()
        }
    with torch.no_grad():
        out = model(**kwargs)
    rest = out["rest_vertices"].double().cpu().numpy()
    vertices, target = rest[0], rest[1] - rest[0]
    B = out["rest_bone_poses"][0].double().cpu().numpy()
    jaw = list(model.bone_labels).index("jaw")
    return jaw_open(vertices, target, jaw_weights(model), B[jaw], translate)


def with_jaw_corrective(
    model, targets, rest_vertices, rest_bone_poses, max_influences=4
):
    """
    The morph targets and rest rows of an export (``opensculptboy.export.body``'s
    ``morph_target_rows`` and ``rest_rows``) with the corrective of anny's jawOpen appended as
    the target ``jawOpen.corrective``, when the model has the jaw and the export has jawOpen.
    The corrective completes the turn for the weights that the file keeps
    (``max_influences``). Returns (the :class:`JawOpen` or None, targets, rest rows).
    """
    from opensculptboy.export.body import MorphTarget

    names = [t.name for t in targets]
    if "jaw" not in model.bone_labels or JAW_OPEN not in names:
        return None, targets, rest_vertices
    base = rest_vertices[0]
    target = rest_vertices[1 + names.index(JAW_OPEN)] - base
    jaw = list(model.bone_labels).index("jaw")
    # the turn fits the model's weights; the corrective completes it for the file's own
    # weights (the strongest max_influences bones), so the file's jaw gives jawOpen exactly
    turn = jaw_open(base, target, jaw_weights(model), rest_bone_poses[jaw])
    weights = jaw_weights(model, max_influences)
    turn.corrective = corrective_for(
        base, target, weights, turn.rotation, turn.translation, turn.pivot
    )
    corrective = MorphTarget(CORRECTIVE, "corrective", -1, 0.0, 0.0)
    rows = np.concatenate([rest_vertices, (base + turn.corrective)[None]])
    return turn, [*targets, corrective], rows

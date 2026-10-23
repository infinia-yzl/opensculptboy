# OpenSculptBoy
# Apache License, Version 2.0
"""
The jaw's skin weights of a fitted character, taken from its reference (docs/mouth.md, "The
jaw's weights from the reference").

A reference's rig need not give its lips to the jaw: a game's face rig moves the lips with
joints of their own, which its mouth poses turn together with the jaw. What carries over is
how far each point of the skin follows the jaw when the reference opens its mouth:
:func:`reference_jaw_share` poses the reference with its own mouth poses (``mouth.poses`` of
appearance.json) and measures, for each skin vertex, the share of the jaw's turn in its
motion. :func:`jaw_share` carries that share to the vertices of a fitted card, and
:func:`with_jaw_share` gives a model of the ``anny-mouth`` rig the jaw weights that follow it.
"""

from __future__ import annotations

import copy
import pathlib
import re

import numpy as np

MOUTH_FILE_FORMAT = "opensculptboy/mouth-weights@1"
# the poses that open the mouth by less than this turn of the jaw (degrees) carry too little
# of its motion to measure a share
MIN_TURN_DEG = 2.0
# skin farther than this from the jaw's motion (metres, summed over the poses) keeps no share
MIN_MOTION = 5e-4


def jaw_joint(appearance: dict, rig) -> int | None:
    """The node index of the reference's jaw: ``mouth.rig.jaw`` of appearance.json, else the
    one node whose name holds the word "jaw" or "mandible" with no side or action word beside
    it (``JAW_EXCLUDED``); None when no name or more than one name qualifies."""
    name = ((appearance.get("mouth") or {}).get("rig") or {}).get("jaw")
    if isinstance(name, str) and name in rig.names:
        return rig.names[name]
    found = [index for key, index in rig.names.items() if _is_jaw_name(key)]
    return found[0] if len(found) == 1 else None


# words that mark a jaw-related node as a side or an action of the jaw
JAW_EXCLUDED = {"l", "r", "left", "right", "open", "clench", "depress", "end", "tip"}


def _is_jaw_name(name: str) -> bool:
    words = [w for w in re.split(r"[^a-z0-9]+", name.lower()) if w]
    return bool({"jaw", "mandible"} & set(words)) and not (JAW_EXCLUDED & set(words))


def reference_jaw_share(handoff, poses: dict | None = None) -> dict:
    """
    The share (V,) of the jaw's turn in the motion of each vertex of each skin part of a
    loaded handoff, by part index, over the reference's own mouth poses (``poses``, name to
    joint deltas; the handoff's known ``mouth.poses`` when None). For a vertex with motion d
    and the motion j that the jaw's turn alone would give it, the share is the least-squares
    factor sum(d . j) / sum(j . j) over the poses, clipped to 0..1; vertices that the jaw's
    turn moves by less than :data:`MIN_MOTION` take 0; the poses that turn the jaw by less
    than :data:`MIN_TURN_DEG` count for nothing. The key ``"poses"`` names the poses that
    counted. Empty when the handoff has no poses or no jaw.
    """
    from opensculptboy.fit.handoff import GLTF_TO_ANNY
    from opensculptboy.fit.poses import GlbRig, mouth_poses, posed_parts

    poses = mouth_poses(handoff.appearance) if poses is None else poses
    rig = GlbRig(handoff.path / "replica.glb")
    jaw = jaw_joint(handoff.appearance, rig)
    if jaw is None or not poses:
        return {}
    rest_world = rig.world()
    C = np.eye(4)
    C[:3, :3] = GLTF_TO_ANNY
    num: dict[int, np.ndarray] = {}
    den: dict[int, np.ndarray] = {}
    used = []
    for name, deltas in poses.items():
        world = rig.world(deltas)
        turn = C @ world[jaw] @ np.linalg.inv(rest_world[jaw]) @ C.T
        angle = np.degrees(
            np.arccos(np.clip((np.trace(turn[:3, :3]) - 1) / 2, -1.0, 1.0))
        )
        if angle < MIN_TURN_DEG:
            continue
        used.append(name)
        posed = posed_parts(handoff, deltas)
        for k, part in enumerate(handoff.parts):
            if not part.is_skin or posed[k] is None:
                continue
            d = posed[k] - part.vertices
            j = part.vertices @ (turn[:3, :3] - np.eye(3)).T + turn[:3, 3]
            num[k] = num.get(k, 0.0) + (d * j).sum(1)
            den[k] = den.get(k, 0.0) + (j * j).sum(1)
    out = {}
    for k in num:
        share = num[k] / np.maximum(den[k], 1e-18)
        share[np.sqrt(den[k]) < MIN_MOTION] = 0.0
        out[k] = np.clip(share, 0.0, 1.0)
    if out:
        out["poses"] = tuple(used)
    return out


def reference_share_field(reference, handoff, shares: dict) -> np.ndarray:
    """The shares of :func:`reference_jaw_share` on the vertices of a
    :class:`~opensculptboy.fit.reference.Reference` (its skin parts laid end to end, as
    ``Reference.from_handoff`` lays them), 0 where no part gives one."""
    from scipy.spatial import cKDTree

    out = np.zeros(len(reference.vertices))
    tree = cKDTree(reference.vertices)
    for k, share in shares.items():
        if k == "poses":
            continue
        d, i = tree.query(handoff.parts[k].vertices)
        near = d < 1e-6
        out[i[near]] = np.maximum(out[i[near]], share[near])
    return out


def reference_border(reference, weld: float = 1e-6) -> np.ndarray:
    """Points (N, 3) along the reference's border edges (edges of one triangle, its vertices
    welded where a source splits them at UV seams), every quarter millimetre."""
    p = np.asarray(reference.vertices, np.float64)
    _, ids = np.unique(np.round(p / weld).astype(np.int64), axis=0, return_inverse=True)
    t = ids.reshape(-1)[np.asarray(reference.triangles)]
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), 1)
    edges, count = np.unique(e, axis=0, return_counts=True)
    border = edges[count == 1]
    if len(border) == 0:
        return np.zeros((0, 3))
    first = np.zeros(ids.max() + 1, np.int64)
    first[ids.reshape(-1)] = np.arange(len(p))
    a, b = p[first[border[:, 0]]], p[first[border[:, 1]]]
    n = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / 2.5e-4)).astype(int)
    return np.concatenate(
        [
            a[k] + np.linspace(0, 1, m + 1)[:, None] * (b[k] - a[k])
            for k, m in enumerate(n)
        ]
    )


def jaw_share(
    fitted: np.ndarray,
    triangles: np.ndarray,
    reference,
    field: np.ndarray,
    on_skin: np.ndarray,
    distance: float = 0.003,
    border: float = 0.0015,
) -> np.ndarray:
    """
    The share (V,) of a fitted level (``fitted``, posed and placed as the reference) from
    the reference's share ``field`` (one value per reference vertex): the vertices of
    ``on_skin`` within ``distance`` of the reference take the value at their closest point,
    interpolated over its triangle; the others (the inside of the mouth, the lips' hidden
    roll) take the harmonic extension of those, so that each lip's lining follows its lip.
    So do the vertices whose closest point lies within ``border`` of one of the reference's
    border edges: where a reference's closed lips are separate rows, the rows of both lips
    meet along its mouth opening, and the closest point there falls on either lip.
    """
    from scipy.spatial import cKDTree

    from opensculptboy.fit.register import smooth_mouth

    rows = np.nonzero(np.asarray(on_skin, bool))[0]
    closest, _, d, tri, bary = reference.closest(fitted[rows])[:5]
    values = (field[reference.triangles[tri]] * bary).sum(1)
    near = d < distance
    edge = reference_border(reference)
    if len(edge):
        near &= cKDTree(edge).query(closest)[0] > border
    share = np.zeros(len(fitted))
    share[rows[near]] = values[near]
    known = np.zeros(len(fitted), bool)
    known[rows[near]] = True
    filled = smooth_mouth(share[:, None], triangles, ~known)[:, 0]
    return np.clip(filled, 0.0, 1.0)


def attach_jaw_share(result, handoff, poses: dict | None = None) -> bool:
    """
    Give a :class:`~opensculptboy.fit.register.FitResult` the jaw share of its level 0 from
    its handoff's mouth poses (:func:`reference_jaw_share`, :func:`jaw_share`), which
    ``FitResult.write`` writes as ``<card>.jaw_share.safetensors`` under ``detail.mouth``. The
    skin of anny (outside :func:`~opensculptboy.fit.register.excluded_vertices`) takes the
    reference's share; the inside of the mouth and the other closed parts take its harmonic
    extension. Returns False, and changes nothing, when the handoff has no mouth poses or
    no jaw.
    """
    from opensculptboy.fit.poses import mouth_poses
    from opensculptboy.fit.reference import Reference
    from opensculptboy.fit.register import excluded_vertices, fit_model

    poses = mouth_poses(handoff.appearance) if poses is None else poses
    shares = reference_jaw_share(handoff, poses)
    if not shares or 0 not in result.meshes:
        return False
    reference = result.reference or Reference.from_handoff(handoff)
    field = reference_share_field(reference, handoff, shares)
    fitted, triangles = result.meshes[0]
    model = fit_model(result.character.topology)
    result.jaw_share = jaw_share(
        fitted, triangles, reference, field, ~excluded_vertices(model)
    )
    result.jaw_share_poses = tuple(shares["poses"])
    return True


def write_mouth_file(path: str | pathlib.Path, share: np.ndarray, poses) -> None:
    from safetensors.numpy import save_file

    save_file(
        {"jaw_share": np.asarray(share, np.float32)},
        str(path),
        metadata={"format": MOUTH_FILE_FORMAT, "poses": ",".join(poses)},
    )


def read_mouth_file(path: str | pathlib.Path) -> np.ndarray:
    from safetensors import safe_open

    with safe_open(str(path), "np") as f:
        if (f.metadata() or {}).get("format") != MOUTH_FILE_FORMAT:
            raise ValueError(f"{path} is not an {MOUTH_FILE_FORMAT} file.")
        return np.asarray(f.get_tensor("jaw_share"), np.float64)


def card_jaw_share(character, detail_dir=None) -> np.ndarray | None:
    """The jaw share of a card's ``detail.mouth.jaw_share`` file, or None."""
    name = (character.detail.get("mouth") or {}).get("jaw_share")
    if not name:
        return None
    path = pathlib.Path(name)
    if not path.is_absolute() and detail_dir is not None:
        path = pathlib.Path(detail_dir) / path
    return read_mouth_file(path)


def card_model(model, character, detail_dir=None):
    """The model of an export: on the ``anny-mouth`` rig with a card whose fit recorded a jaw
    share (``detail.mouth``), a copy whose jaw weights follow it (:func:`with_jaw_share`);
    otherwise ``model`` itself."""
    from opensculptboy.mouth.rig import is_mouth_rig

    if not is_mouth_rig(character.rig):
        return model
    share = card_jaw_share(character, detail_dir)
    return model if share is None else with_jaw_share(model, share)


def with_jaw_share(model, share: np.ndarray):
    """
    A copy of a model of the ``anny-mouth`` rig whose jaw weights follow ``share`` (V,): on
    each vertex that the head, the jaw or ``share`` moves, the head's and the jaw's weights
    together are split as (1 - share, share); the tongue's and every other bone's weights
    stay. The model itself is left unchanged.
    """
    import torch

    labels = list(model.bone_labels)
    if "jaw" not in labels:
        raise ValueError("with_jaw_share needs the 'anny-mouth' rig (a jaw bone).")
    share = np.asarray(share, np.float64)
    if len(share) != len(model.template_vertices):
        raise ValueError(
            f"The jaw share has {len(share)} values for a model of "
            f"{len(model.template_vertices)} vertices."
        )
    head, jaw = labels.index("head"), labels.index("jaw")
    w = model.vertex_bone_weights.detach().cpu().double().numpy()
    i = model.vertex_bone_indices.detach().cpu().numpy()
    V, K = w.shape
    dense = np.zeros((V, len(labels)))
    np.add.at(dense, (np.repeat(np.arange(V), K), i.reshape(-1)), w.reshape(-1))
    group = dense[:, head] + dense[:, jaw]
    rows = (group > 0) & ((dense[:, jaw] > 0) | (share > 0))
    dense[rows, jaw] = share[rows] * group[rows]
    dense[rows, head] = (1.0 - share[rows]) * group[rows]
    width = max(K, int((dense > 0).sum(1).max()))
    order = np.argsort(-dense, axis=1, kind="stable")[:, :width]
    weights = np.take_along_axis(dense, order, axis=1)
    weights /= weights.sum(1, keepdims=True)
    indices = np.where(weights > 0, order, 0)
    out = copy.copy(model)
    out._buffers = dict(model._buffers)
    out.vertex_bone_weights = torch.as_tensor(
        weights, dtype=model.vertex_bone_weights.dtype
    )
    out.vertex_bone_indices = torch.as_tensor(
        indices, dtype=model.vertex_bone_indices.dtype
    )
    return out

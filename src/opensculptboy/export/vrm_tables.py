# OpenSculptBoy
# Apache License, Version 2.0
"""
Tables of the VRM export for the ``anny`` rig: the humanoid bone map, the required bones of
each VRM version, the twist bones and their roll constraints, the node hierarchy of the file,
the eyeball vertices, the VRM skin weights, the expressions (``data/vrm/expressions.json``)
and the look-at ranges.

VRM apps turn the humanoid bones alone; every other node of the file keeps its rest place
under its parent. The twist bones of the forearms split the turn of the wrist over the length
of the forearm, so VRM 1.0 turns them with roll constraints (``VRMC_node_constraint``): each
follows half of the roll of the hand below it. Every node of a VRM file rests at the identity
rotation (normalised joints), so the roll axis of a constraint must be one of the axes X, Y
and Z of the file, and it must run along the limb in the T-pose: the forearms lie along X in
the T-pose to within a rounding error. In the rig the hand hangs from the forearm twist bone;
the hand would then turn by its own roll and again by the half roll of the twist bone above
it, so the file hangs the hand from the forearm, beside its twist bone, which becomes a leaf
(:func:`file_parents`).

The shin twist bones get no constraint: the T-posed shin runs off the vertical, by 5 degrees
on the default body and 3 to 8 degrees over the corners of the phenotypes (8, and 5 to 14
degrees, with ``keep_leg_spread``), so a roll about Y would swing the
vertices of the lower shin off the limb, by up to 5 to 12 mm, whenever the foot turns. Like
the twist bones of the upper arms and the thighs, they hang from their limb bone (the shin,
the upper arm and the thigh), which carries the roll of the knee, the shoulder or the hip in
every humanoid pose, and they follow it without a turn of their own. Their skin weights
therefore move to those bones in every VRM file (:func:`vrm_skin_weights`). This changes
nothing in the skinning of the file, and it frees an influence on the vertices that both
bones weigh (250 to 400 per limb), so the truncation to 4 bones drops about a fifth less
weight.

The eyeballs are the vertices of MakeHuman's eye helpers, found by their MakeHuman base-mesh
indices, so the VRM export supports the topologies that keep those indices
(:data:`VRM_TOPOLOGIES`); :func:`eyeball_vertices` refuses the others.

The inverse bind of the T-pose (:func:`opensculptboy.export.tpose.rebind`) folds the skin
where the T-pose straightens the fingers and the legs: at the webs of the fingers and the
thumb, the groin and the neck. ``data/vrm/skin_weights.safetensors`` holds new skin weights
for the 1124 vertices in and around those folds, made by
``python -m opensculptboy.export.authoring.skin_weights``: for each twist mode, the bones of
each edited vertex and logits that depend on the six default phenotypes, since the webs of the
fingers fold differently from one body to the next (:func:`edit_weights`). They hold every
gate on the bodies they were checked on, but they fail the hand gates of 36 of 82 bodies, so
no file uses them as they stand: they start the refinement of the weights for each body that
the VRM export runs with the inverse bind
(:func:`opensculptboy.export.authoring.refine.body_skin_weights`). :func:`skin_weight_data`
reads them when the digest of the file matches the model's mesh and skin weights
(:func:`skin_weight_digest`), and ``vrm_skin_weights(..., edits=True)`` applies them.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import pathlib
import warnings

import numpy as np
import torch

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "vrm"
# The edited skin weights of the fold regions, when the file exists (see the module
# docstring).
SKIN_WEIGHT_EDITS = DATA_DIR / "skin_weights.safetensors"
# The format of SKIN_WEIGHT_EDITS, in its metadata.
SKIN_WEIGHT_FORMAT = "opensculptboy/vrm-skin-weights@1"
# The phenotypes whose values the logits of the edits take (see edit_features).
EDIT_PHENOTYPES = ("gender", "age", "muscle", "weight", "height", "proportions")

VERSIONS = ("1.0", "0.x")

# VRM 1.0 humanoid bone -> anny bone. The right side mirrors the left (".L" -> ".R").
_LEFT = {
    "leftUpperLeg": "upperleg01.L",
    "leftLowerLeg": "lowerleg01.L",
    "leftFoot": "foot.L",
    "leftShoulder": "clavicle.L",
    "leftUpperArm": "upperarm01.L",
    "leftLowerArm": "lowerarm01.L",
    "leftHand": "wrist.L",
    "leftThumbMetacarpal": "finger1-1.L",
    "leftThumbProximal": "finger1-2.L",
    "leftThumbDistal": "finger1-3.L",
    "leftIndexProximal": "finger2-1.L",
    "leftIndexIntermediate": "finger2-2.L",
    "leftIndexDistal": "finger2-3.L",
    "leftMiddleProximal": "finger3-1.L",
    "leftMiddleIntermediate": "finger3-2.L",
    "leftMiddleDistal": "finger3-3.L",
    "leftRingProximal": "finger4-1.L",
    "leftRingIntermediate": "finger4-2.L",
    "leftRingDistal": "finger4-3.L",
    "leftLittleProximal": "finger5-1.L",
    "leftLittleIntermediate": "finger5-2.L",
    "leftLittleDistal": "finger5-3.L",
    "leftEye": "eye.L",
}
HUMANOID_1 = {
    "hips": "root",
    "spine": "spine04",
    "chest": "spine02",
    "upperChest": "spine01",
    "neck": "neck01",
    "head": "head",
    **_LEFT,
    **{
        "right" + k[len("left") :]: v[:-2] + ".R"
        for k, v in _LEFT.items()  # "leftFoot" -> "rightFoot", "foot.L" -> "foot.R"
    },
}
# VRM 0.x names the three thumb bones Proximal, Intermediate and Distal.
_THUMB_0 = {"ThumbMetacarpal": "ThumbProximal", "ThumbProximal": "ThumbIntermediate"}


def _name_0(name_1: str) -> str:
    for side in ("left", "right"):
        if name_1.startswith(side) and name_1[len(side) :] in _THUMB_0:
            return side + _THUMB_0[name_1[len(side) :]]
    return name_1


HUMANOID_0 = {_name_0(k): v for k, v in HUMANOID_1.items()}

REQUIRED_1 = {
    "hips",
    "spine",
    "head",
    *(
        side + part
        for side in ("left", "right")
        for part in (
            "UpperLeg",
            "LowerLeg",
            "Foot",
            "UpperArm",
            "LowerArm",
            "Hand",
        )
    ),
}
REQUIRED_0 = REQUIRED_1 | {"chest", "neck"}

# The expression presets of each VRM version (VRM 1.0 ``expressions.preset`` names, VRM 0.x
# ``presetName`` values other than "unknown"). No custom expression may take one of these
# names, compared without case.
PRESETS_1 = (
    "happy",
    "angry",
    "sad",
    "relaxed",
    "surprised",
    "aa",
    "ih",
    "ou",
    "ee",
    "oh",
    "blink",
    "blinkLeft",
    "blinkRight",
    "lookUp",
    "lookDown",
    "lookLeft",
    "lookRight",
    "neutral",
)
PRESETS_0 = (
    "neutral",
    "a",
    "i",
    "u",
    "e",
    "o",
    "blink",
    "joy",
    "angry",
    "sorrow",
    "fun",
    "lookup",
    "lookdown",
    "lookleft",
    "lookright",
    "blink_l",
    "blink_r",
)

# Twist bones that VRM 1.0 turns with roll constraints: twist bone -> (the bone whose roll it
# follows, the roll axis in the normalised T-pose of the file, the weight of the roll). The
# forearm twist bone takes half of the roll of the hand, as the forearm does when the hand
# turns. The shin twist bone has none: the T-posed shin runs off every axis of the file (see
# the module docstring), so its weights merge into the shin.
TWIST_BONES = {
    f"{bone}.{s}": (f"{source}.{s}", axis, 0.5)
    for s in ("L", "R")
    for bone, source, axis in (("lowerarm02", "wrist", "X"),)
}
# The twist bones and the mapped bones that take their weights. With twist="merge" every one
# of them moves; with twist="constraint" the bones of TWIST_BONES keep their weights.
TWIST_MERGE = {
    f"{bone}.{s}": f"{parent}.{s}"
    for s in ("L", "R")
    for bone, parent in (
        ("upperarm02", "upperarm01"),
        ("lowerarm02", "lowerarm01"),
        ("upperleg02", "upperleg01"),
        ("lowerleg02", "lowerleg01"),
    )
}
TWIST_MODES = ("constraint", "merge")

# The eyeballs: MakeHuman base-mesh vertex ranges of helper-l-eye and helper-r-eye, 72
# vertices each.
_EYE_RANGES = {"L": (14598, 14669), "R": (14670, 14741)}
EYEBALL_VERTEX_COUNT = 72
# The largest distance (metres) between the centroid of an eyeball and the head of its eye
# bone in the template; the MakeHuman eyes lie within a few micrometres.
EYEBALL_CENTRE_TOLERANCE = 0.02
# The topologies whose vertices keep their MakeHuman base-mesh indices, with the eyes.
VRM_TOPOLOGIES = (
    "anny",
    "anny-quads",
    "anny-full",
    "makehuman",
    "sculpt",
    "sculpt-quads",
)

# The facial actions of the look-at ranges: VRM look-at direction -> ARKit action prefix.
LOOK_ACTIONS = {
    "lookUp": "eyeLookUp",
    "lookDown": "eyeLookDown",
    "lookIn": "eyeLookIn",
    "lookOut": "eyeLookOut",
}


def _check_version(version: str) -> None:
    if version not in VERSIONS:
        raise ValueError(f"Unknown VRM version {version!r}; use '1.0' or '0.x'.")


def _check_twist(twist: str) -> None:
    if twist not in TWIST_MODES:
        raise ValueError(f"Unknown twist mode {twist!r}; use 'constraint' or 'merge'.")


def humanoid_bones(version: str) -> dict[str, str]:
    """VRM humanoid bone name -> anny bone label, for ``version`` "1.0" or "0.x"."""
    _check_version(version)
    return dict(HUMANOID_1 if version == "1.0" else HUMANOID_0)


def required_bones(version: str) -> set[str]:
    """The humanoid bones that a file of ``version`` must map."""
    _check_version(version)
    return set(REQUIRED_1 if version == "1.0" else REQUIRED_0)


def eyeball_vertices(model) -> dict[str, np.ndarray]:
    """
    The model vertex indices of each eyeball (``"L"`` and ``"R"``): the vertices of MakeHuman's
    eye helpers, found by their base-mesh indices (``model.base_mesh_vertex_indices``).

    Raises ValueError unless each side has exactly 72 vertices whose centroid lies within 2 cm
    of the head of its eye bone in the template (``model.template_bone_heads``). A
    retopologised mesh (``notoes``, ``soma``, ...) numbers its vertices its own way, so the
    MakeHuman indices would pick other vertices or none, and a mesh without eyes has none. The
    VRM export supports the MakeHuman-based topologies of :data:`VRM_TOPOLOGIES`.
    """
    supported = ", ".join(VRM_TOPOLOGIES)

    def refuse(problem: str):
        raise ValueError(
            f"The model's mesh has no MakeHuman eyeballs ({problem}). VRM export supports "
            f"the MakeHuman-based topologies {supported}."
        )

    labels = list(model.bone_labels)
    base = getattr(model, "base_mesh_vertex_indices", None)
    if base is None:
        refuse("it has no MakeHuman base-mesh vertex indices")
    base = torch.as_tensor(base).detach().cpu().numpy()
    template = model.template_vertices.detach().cpu().double().numpy()
    heads = model.template_bone_heads.detach().cpu().double().numpy()
    eyes = {}
    for side, (lo, hi) in _EYE_RANGES.items():
        bone = f"eye.{side}"
        if bone not in labels:
            refuse(f"the rig has no bone {bone!r}")
        name = {"L": "left", "R": "right"}[side]
        rows = np.flatnonzero((base >= lo) & (base <= hi))
        if len(rows) != EYEBALL_VERTEX_COUNT:
            refuse(
                f"{len(rows)} vertices on the {name} eyeball, where MakeHuman has "
                f"{EYEBALL_VERTEX_COUNT}"
            )
        gap = float(np.linalg.norm(template[rows].mean(0) - heads[labels.index(bone)]))
        if not gap <= EYEBALL_CENTRE_TOLERANCE:
            refuse(
                f"the centre of the {name} eyeball lies {100 * gap:.1f} cm from the head "
                f"of {bone}, where at most {100 * EYEBALL_CENTRE_TOLERANCE:g} cm is allowed"
            )
        eyes[side] = rows
    return eyes


def twist_constraints(version: str, twist: str) -> list[tuple[str, str, str, float]]:
    """
    The roll constraints of a file: (constrained bone, source bone, roll axis, weight). VRM 1.0
    with ``twist="constraint"`` carries them (``VRMC_node_constraint``); VRM 0.x has none.
    """
    _check_version(version)
    _check_twist(twist)
    if version != "1.0" or twist != "constraint":
        return []
    return [(bone, src, axis, w) for bone, (src, axis, w) in TWIST_BONES.items()]


def file_parents(model, version: str, twist: str) -> list[int]:
    """
    The parent of each bone in the node hierarchy of the file (-1 for the root).

    With roll constraints, the source of each constraint leaves the constrained twist bone for
    the twist bone's parent: the wrist hangs from lowerarm01. The forearm twist bones become
    leaves, and the hand does not turn twice. Every other bone keeps the rig's parent, the foot
    among them (it hangs from the shin twist bone lowerleg02, which carries no constraint);
    without constraints, the hierarchy is the rig's own. Every parent comes before its
    children, as in the rig.
    """
    labels = list(model.bone_labels)
    parents = [int(p) for p in model.bone_parents]
    for bone, source, _, _ in twist_constraints(version, twist):
        b, s = labels.index(bone), labels.index(source)
        child = s  # the child of the constrained bone on the way up from its source
        while parents[child] not in (b, -1):
            child = parents[child]
        if parents[child] == b:
            parents[child] = parents[b]
        if b in parents:
            raise ValueError(
                f"The constrained bone {bone!r} keeps children in the file: "
                f"{[labels[c] for c, p in enumerate(parents) if p == b]}."
            )
    return parents


def skin_weight_digest(model) -> str:
    """
    The SHA-256 digest (hex) of what the skin-weight edits rest on: the bone labels and
    parents, the vertices of the mesh (their MakeHuman base-mesh indices, which the faces
    stand in for on a model without them) and the model's skin weights (rounded to float32,
    so that models of either precision agree) and bone indices. The edits name vertices and
    bones, so a mesh with the same vertices and other faces (``anny-quads``) shares them.

    On the ``sculpt`` topologies the digest covers anny's own vertices (their rows, which
    ``sculpt`` keeps bit for bit, in anny's number of bone slots), so that the edits of anny
    carry over: they name anny's vertices, which keep their indices, and the new vertices of
    the patches take the blend of their anchors' rows (:func:`blend_new_rows`).
    """
    from opensculptboy.export.body import triangulated_faces

    base = getattr(model, "base_mesh_vertex_indices", None)
    bone_indices = model.vertex_bone_indices.detach().cpu().numpy()
    bone_weights = model.vertex_bone_weights.detach().cpu().numpy()
    if base is None:
        vertices = np.asarray(triangulated_faces(model)[0], dtype=np.int64)
    else:
        vertices = torch.as_tensor(base).detach().cpu().numpy().astype(np.int64)
    sculpt = getattr(model, "sculpt_data", None)
    if sculpt is not None:
        n = sculpt.vertex_count
        slots = getattr(model, "sculpt_bone_slots", bone_weights.shape[1])
        vertices = vertices[:n]
        bone_indices = bone_indices[:n, :slots]
        bone_weights = bone_weights[:n, :slots]
    digest = hashlib.sha256()
    digest.update("\n".join(model.bone_labels).encode())
    for array in (
        np.asarray([int(p) for p in model.bone_parents], dtype=np.int64),
        vertices,
        bone_indices.astype(np.int64),
        bone_weights.astype(np.float32),
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def skin_weight_data(model, path=None, warn_missing: bool = False) -> dict | None:
    """
    The edits of :data:`SKIN_WEIGHT_EDITS` (or ``path``) as stored: ``{twist: (vertices (E,),
    bones (E, 4), logits (E, 4, D))}``, with -1 for an empty slot.

    Returns None when the file is missing (with a warning when ``warn_missing``), and None
    with a warning when its digest differs from :func:`skin_weight_digest` of the model
    (another mesh, rig or skin weights): the edits name vertices and bones of the mesh they
    were made for.
    """
    path = pathlib.Path(path or SKIN_WEIGHT_EDITS)
    if not path.is_file():
        if warn_missing:
            warnings.warn(
                f"The VRM skin-weight data {path} is missing; the VRM export keeps the "
                "model's weights, and the T-pose of the file may fold at the fingers, the "
                "groin and the neck.",
                stacklevel=3,
            )
        return None
    metadata, tensors = _read_skin_weight_edits(str(path))
    if metadata.get("format") != SKIN_WEIGHT_FORMAT:
        raise ValueError(
            f"{path.name} has the format {metadata.get('format')!r}, where "
            f"{SKIN_WEIGHT_FORMAT!r} is expected."
        )
    if metadata.get("digest") != skin_weight_digest(model):
        warnings.warn(
            f"The VRM skin-weight edits of {path.name} were made for another mesh or other "
            "skin weights; the VRM export keeps the model's weights, and the T-pose of the "
            "file may fold at the fingers, the groin and the neck.",
            stacklevel=3,
        )
        return None
    return {
        twist: tuple(tensors[f"{twist}.{k}"] for k in ("vertices", "bones", "logits"))
        for twist in TWIST_MODES
    }


@functools.lru_cache(maxsize=4)
def _read_skin_weight_edits(path: str) -> tuple[dict, dict]:
    from safetensors import safe_open

    with safe_open(path, framework="numpy") as f:
        metadata = dict(f.metadata() or {})
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    return metadata, tensors


def edit_features(phenotype=None, count: int = 1 + len(EDIT_PHENOTYPES)) -> np.ndarray:
    """
    The first ``count`` features of a phenotype for the logits of the edits: 1, then each
    value x_i of :data:`EDIT_PHENOTYPES` less 0.5 (a missing value counts as its default,
    0.5), then the products x_i x_j (i <= j, in that order). The values may be numbers or
    tensors of one element.
    """
    phenotype = phenotype or {}
    x = [
        float(
            np.asarray(torch.as_tensor(phenotype[name]).detach().cpu()).reshape(-1)[0]
        )
        - 0.5
        if name in phenotype
        else 0.0
        for name in EDIT_PHENOTYPES
    ]
    products = [x[i] * x[j] for i in range(len(x)) for j in range(i, len(x))]
    features = np.array([1.0, *x, *products])
    if count > len(features):
        raise ValueError(f"The edits have {count} features; at most {len(features)}.")
    return features[:count]


def edit_weights(bones: np.ndarray, values: np.ndarray, phenotype=None) -> np.ndarray:
    """
    The weights (E, 4) of edited rows: ``values`` themselves when they are weights (E, 4),
    or, for logits (E, 4, D), the softmax over the slots that hold a bone (``bones`` >= 0) of
    the logits times :func:`edit_features` of ``phenotype``.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2:
        return values
    logits = values @ edit_features(phenotype, values.shape[-1])
    logits = np.where(np.asarray(bones) >= 0, logits, -np.inf)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def skin_weight_edits(
    model, twist: str = "constraint", path=None, phenotype=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    The edited rows of the VRM skin weights for ``twist`` and a body of ``phenotype`` (the
    default body by default): the vertex indices (E,) and, for each, its bones (E, 4) and
    weights (E, 4) (:func:`edit_weights`; a weight of 0 and the bone 0 pad the rows of fewer
    bones, and a bone may fill two slots), read from :data:`SKIN_WEIGHT_EDITS` (or ``path``).

    Returns None when the file is missing, and None with a warning when its digest differs
    from :func:`skin_weight_digest` of the model (another mesh, rig or skin weights): the
    edits name vertices and bones of the mesh they were made for.
    """
    _check_twist(twist)
    data = skin_weight_data(model, path)
    if data is None:
        return None
    vertices, bones, logits = data[twist]
    bones = bones.astype(np.int64)
    weights = edit_weights(bones, logits, phenotype)
    return (
        vertices.astype(np.int64),
        np.maximum(bones, 0),
        np.where(bones >= 0, weights, 0.0),
    )


def blend_new_rows(model, weights, indices, before=None):
    """
    Skin weights (V, K) and bone indices of a ``sculpt`` model with the rows of its new
    vertices blended again from their anchors' rows, as the model blends them: for every new
    vertex with an anchor whose row differs from ``before`` (a pair of the rows before an
    edit; all new vertices when None) and whose own row did not change. The rows keep K slots,
    the strongest bones first, and sum to 1. Other models pass through.
    """
    from opensculptboy.topology.lookup import model_sculpt_data

    data = model_sculpt_data(model)
    if data is None or data.new_vertex_count == 0:
        return weights, indices
    weights = np.asarray(weights, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    n, J = data.vertex_count, len(model.bone_labels)
    V, K = weights.shape

    def dense(w, i, rows):
        out = np.zeros((len(rows), J))
        np.add.at(
            out, (np.repeat(np.arange(len(rows)), K), i[rows].ravel()), w[rows].ravel()
        )
        return out

    new = np.arange(n, V)
    if before is None:
        rows = new
    else:
        bw = np.asarray(before[0], dtype=np.float64)
        bi = np.asarray(before[1], dtype=np.int64)
        if bw.shape[1] != K:
            return weights, indices
        anchors = np.unique(data.anchors[data.weights > 0])
        changed = np.zeros(V, bool)
        a = dense(weights, indices, anchors) - dense(bw, bi, anchors)
        changed[anchors] = np.abs(a).max(1) > 1e-12
        own = np.abs(dense(weights, indices, new) - dense(bw, bi, new)).max(1) > 1e-12
        hit = (changed[data.anchors] & (data.weights > 0)).any(1) & ~own
        rows = new[hit]
    if len(rows) == 0:
        return weights, indices
    anchor_dense = dense(weights, indices, np.arange(n))
    blended = np.einsum(
        "mk,mkj->mj", data.weights[rows - n], anchor_dense[data.anchors[rows - n]]
    )
    order = np.argsort(-blended, axis=1, kind="stable")[:, :K]
    w = np.take_along_axis(blended, order, axis=1)
    w /= w.sum(axis=1, keepdims=True)
    weights, indices = weights.copy(), indices.copy()
    weights[rows] = w
    indices[rows] = np.where(w > 0, order, 0)
    return weights, indices


def vrm_skin_weights(
    model, twist: str = "constraint", edits=False, phenotype=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    (V, K) skin weights and bone indices for a VRM file, before the truncation to 4 bones.

    - The eyelid vertices weighted to an eye bone (30 per eye, up to 10.5 %) move that weight
      to ``head``, so that look-at turns the eyeballs alone, and the eyeball vertices move
      their small ``head`` weight to their eye bone, so that each eyeball turns as one piece.
      The eye bones rest on the head in every pose but look-at, so neither move changes any
      other pose.
    - The twist bones of TWIST_MERGE move their weights to their mapped parents, except, with
      ``twist="constraint"`` (the VRM 1.0 default), the forearm twist bones of TWIST_BONES,
      which VRM 1.0 turns with roll constraints. A twist bone without a constraint (upper
      arm, thigh and shin) follows its parent in the file without a turn of its own, so the
      move leaves the skinning of the file as it was and frees influences for the
      truncation. ``twist="merge"`` (the VRM 0.x default) moves them all.
    - With ``edits``, rows for the body of ``phenotype`` (the character's phenotype values;
      the default body by default) then replace those of the vertices in and around the
      folds of the T-pose, with 4 bones or fewer each: True takes those of the data file
      (:func:`skin_weight_edits`, when it matches the model), and a dict takes edits of the
      authoring tools, ``{twist: (vertices, bones, weights or logits)}`` (see
      :func:`edit_weights`). False (the default) keeps the weights above: the VRM export
      starts from them, and refines the rows of the data file for each body
      (:func:`opensculptboy.export.authoring.refine.body_skin_weights`).

    Weights of the same bone on one vertex add up. Each row holds the model's K slots sorted
    from the strongest bone down; empty slots have weight 0 and bone 0. Rows sum to 1.
    """
    _check_twist(twist)
    labels = list(model.bone_labels)
    bone = {label: j for j, label in enumerate(labels)}
    weights = model.vertex_bone_weights.detach().cpu().double().numpy()
    indices = model.vertex_bone_indices.detach().cpu().numpy().astype(np.int64)
    V, K = weights.shape
    moved = indices.copy()
    head = bone["head"]
    for side, rows in eyeball_vertices(model).items():
        eye = bone[f"eye.{side}"]
        on_eyeball = np.zeros(V, dtype=bool)
        on_eyeball[rows] = True
        moved[(indices == eye) & ~on_eyeball[:, None]] = head
        moved[(indices == head) & on_eyeball[:, None]] = eye
    for twist_bone, parent in TWIST_MERGE.items():
        if twist == "constraint" and twist_bone in TWIST_BONES:
            continue
        moved[indices == bone[twist_bone]] = bone[parent]
    dense = np.zeros((V, len(labels)))
    np.add.at(
        dense, (np.repeat(np.arange(V), K), moved.reshape(-1)), weights.reshape(-1)
    )
    if edits is True:
        edits = skin_weight_edits(model, twist, phenotype=phenotype)
    elif isinstance(edits, dict):
        edits = edits.get(twist)
        if edits is not None:
            rows, bones, values = (np.asarray(a) for a in edits)
            values = np.where(bones >= 0, edit_weights(bones, values, phenotype), 0.0)
            edits = rows, np.maximum(bones, 0), values
    edited = edits is not None and edits is not False
    if edited:
        unedited = _sorted_rows(dense, K)
        rows, bones, values = (np.asarray(a) for a in edits)
        dense[rows] = 0.0
        np.add.at(
            dense,
            (np.repeat(rows, bones.shape[1]), bones.reshape(-1)),
            values.reshape(-1).astype(np.float64),
        )
    result, order = _sorted_rows(dense, K)
    if edited:
        # the new vertices of a sculpt model follow their edited anchors
        result, order = blend_new_rows(model, result, order, unedited)
    return result, order


def _sorted_rows(dense: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """The K strongest bones of dense rows (V, J), normalised, and their indices."""
    order = np.argsort(-dense, axis=1, kind="stable")[:, :K]
    result = np.take_along_axis(dense, order, axis=1)
    result /= result.sum(axis=1, keepdims=True)
    return result, np.where(result > 0, order, 0).astype(np.int64)


@dataclasses.dataclass
class Expression:
    """A VRM expression as a mix of ARKit facial actions (``model.facial_action_labels``)."""

    name: str  # the VRM 1.0 preset name, or the custom name
    mix: dict[str, float]  # facial action -> weight (0 to 1)
    preset: bool  # a VRM 1.0 preset (aa, blink, happy, ...)
    is_binary: bool = False
    override_blink: str = "none"  # "none", "block" or "blend"
    override_look_at: str = "none"
    override_mouth: str = "none"
    vrm0_preset: str | None = (
        None  # the VRM 0.x presetName; None: "unknown" (a custom group)
    )
    vrm0_name: str | None = None  # the VRM 0.x group name; None: the same as ``name``
    versions: tuple[str, ...] = ("1.0", "0.x")  # the VRM versions that carry it


@functools.lru_cache(maxsize=1)
def _expression_table() -> dict:
    with open(DATA_DIR / "expressions.json") as f:
        return json.load(f)


def expressions(version: str, facial_action_labels) -> list[Expression]:
    """
    The expressions of a VRM file: the presets, then the 52 perfect-sync customs (PascalCase
    ARKit names, ``EyeBlinkLeft``), then the further customs of ``version`` (the VSeeFace
    visemes SIL, CH, DD, ... in VRM 0.x). Raises ValueError when a mix names an action that
    ``facial_action_labels`` lacks.
    """
    _check_version(version)
    labels = list(facial_action_labels)
    table = _expression_table()
    result = []
    for name, entry in table["presets"].items():
        result.append(
            Expression(
                name=name,
                mix=dict(entry["mix"]),
                preset=True,
                is_binary=entry.get("isBinary", False),
                override_blink=entry.get("overrideBlink", "none"),
                override_look_at=entry.get("overrideLookAt", "none"),
                override_mouth=entry.get("overrideMouth", "none"),
                vrm0_preset=entry.get("vrm0Preset"),
                vrm0_name=entry.get("vrm0Name"),
                versions=tuple(entry.get("versions", VERSIONS)),
            )
        )
    if table.get("perfectSync", False):
        for label in labels:
            result.append(
                Expression(
                    name=label[0].upper() + label[1:], mix={label: 1.0}, preset=False
                )
            )
    for name, entry in table.get("customs", {}).items():
        result.append(
            Expression(
                name=name,
                mix=dict(entry["mix"]),
                preset=False,
                versions=tuple(entry.get("versions", VERSIONS)),
            )
        )
    known = set(labels)
    for e in result:
        unknown = sorted(set(e.mix) - known)
        if unknown:
            raise ValueError(
                f"The expression {e.name!r} uses facial actions that the model lacks: "
                f"{unknown}."
            )
    return [e for e in result if version in e.versions]


def eyeball_rotation(neutral: np.ndarray, moved: np.ndarray) -> np.ndarray:
    """
    The rotation (3, 3) that best turns the eyeball vertices ``neutral`` (N, 3) onto ``moved``
    (N, 3) about the eyeball centre (the centroid of the vertices): a Kabsch fit.
    """
    a = np.asarray(neutral, dtype=np.float64)
    b = np.asarray(moved, dtype=np.float64)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    flip = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, flip]) @ u.T


def rotation_angle(rotation: np.ndarray) -> float:
    """The angle of a rotation matrix, in degrees."""
    cos = (np.trace(rotation) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def look_at_ranges(model, rest_output=None) -> dict[str, float]:
    """
    The look-at range of each direction in degrees (``"lookUp"``, ``"lookDown"``, ``"lookIn"``,
    ``"lookOut"``): the eyeball rotation that the full ``eyeLook*`` facial action gives,
    fitted on the eyeball vertices (:func:`eyeball_rotation`) and averaged over the two eyes.

    ``rest_output`` is a model output whose ``rest_vertices`` hold the neutral face in row 0
    and facial action ``a`` at weight 1 in row ``1 + a`` (the rows of the VRM writer, with the
    character's shape); by default the model's default body is evaluated.
    """
    actions = list(model.facial_action_labels)
    names = [
        f"{prefix}{side}"
        for prefix in LOOK_ACTIONS.values()
        for side in ("Left", "Right")
    ]
    if rest_output is None:
        rows = torch.zeros(
            len(names) + 1, len(actions), dtype=model.dtype, device=model.device
        )
        for r, name in enumerate(names):
            rows[r + 1, actions.index(name)] = 1.0
        with torch.no_grad():
            rest = model(facial_actions=rows)["rest_vertices"]
        row = {name: r + 1 for r, name in enumerate(names)}
    else:
        rest = rest_output["rest_vertices"]
        row = {name: 1 + actions.index(name) for name in names}
    if isinstance(rest, torch.Tensor):
        rest = rest.detach().cpu().double().numpy()
    rest = np.asarray(rest, dtype=np.float64)
    eyes = eyeball_vertices(model)
    ranges = {}
    for direction, prefix in LOOK_ACTIONS.items():
        angles = []
        for side, eye in (("Left", "L"), ("Right", "R")):
            vertices = eyes[eye]
            turn = eyeball_rotation(
                rest[0, vertices], rest[row[prefix + side], vertices]
            )
            angles.append(rotation_angle(turn))
        ranges[direction] = float(np.mean(angles))
    return ranges


def lid_only(offsets: np.ndarray, eyeballs: dict[str, np.ndarray]) -> np.ndarray:
    """Morph target offsets (V, 3) with the eyeball rows zeroed (the VRM ``eyeLook*`` targets)."""
    offsets = np.array(offsets, dtype=np.float64, copy=True)
    for rows in eyeballs.values():
        offsets[rows] = 0.0
    return offsets

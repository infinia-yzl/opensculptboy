# OpenSculptBoy
# Apache License, Version 2.0
"""
The ``anny-mouth`` rig: anny's rig with MakeHuman's jaw bone and tongue chain kept.

Anny's default rig (``anny``) leaves out the bones of MakeHuman's facial rig with the
``nofacialexpression`` and ``notongue`` modifiers, and moves their skin weights to their
nearest kept ancestor, ``head``. The ``anny-mouth`` rig keeps ``jaw`` (child of ``head``) and
the tongue chain ``tongue00`` to ``tongue07.L/R`` (children of ``jaw``) with MakeHuman's own
weights, and leaves every other bone as it is in ``anny``. Only the head loses the weights
that move to the new bones.

The rig is built here, with anny's public model data and without changes to ``src/anny``:

- :func:`mouth_model_data` builds anny's model data with the mouth bones kept, then gives every
  bone its cached orientation. The 104 bones of ``anny`` take their rows of anny's own cache
  (``data/cached/anny.pth``), so they keep their rest frames bit for bit; ``head`` keeps the
  row that anny computed with the jaw's vertices merged into it. The 12 mouth bones take the
  rows of ``data/mouth/rig_cache.safetensors``, computed the way anny computes its cache
  (:mod:`opensculptboy.mouth.authoring.rig_cache`).
- :func:`mouth_model` builds the Anny model on ``anny`` or ``sculpt``.

A :class:`~opensculptboy.character.Character` takes the rig with ``rig="anny-mouth"``; cards
with ``rig="anny"`` build and export as before.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import pathlib

import torch

MOUTH_RIG = "anny-mouth"
TONGUE_BONES = (
    "tongue00",
    "tongue01",
    "tongue02",
    "tongue03",
    "tongue04",
    "tongue05.L",
    "tongue05.R",
    "tongue06.L",
    "tongue06.R",
    "tongue07.L",
    "tongue07.R",
)
MOUTH_BONES = ("jaw", *TONGUE_BONES)

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "mouth"
RIG_CACHE = DATA_DIR / "rig_cache.safetensors"
# The format of RIG_CACHE, in its metadata.
RIG_CACHE_FORMAT = "opensculptboy/mouth-rig-cache@1"
# The tensors of a cached orientation, per bone (anny's names, data/cached/anny.pth).
_PER_BONE = (
    "template_bone_heads",
    "bone_template_orientation_matrices",
    "reference_bone_orientations",
)
_PER_ROW = ("bone_heads_blendshapes", "bone_orientation_blendshapes")


def is_mouth_rig(rig) -> bool:
    """Whether ``rig`` names the ``anny-mouth`` rig."""
    return isinstance(rig, str) and rig == MOUTH_RIG


def mouth_rig_config(bone_orientation: str = "cached"):
    """The :class:`~anny.models.model_data.RigConfig` of ``anny-mouth``: anny's rig with the
    mouth bones kept. ``bone_orientation`` "cached" is the rig's own; the builder of the
    cache takes the bones' rest frames from their tails ("blender")."""
    from anny.models.model_data import RigConfig

    anny = RigConfig.from_string("anny")
    return dataclasses.replace(
        anny,
        bone_orientation=bone_orientation,
        bones_to_remove=frozenset(anny.bones_to_remove - set(MOUTH_BONES)),
    )


def _anny_cache() -> dict:
    from anny.paths import get_anny_root_dir

    return torch.load(
        os.path.join(get_anny_root_dir(), "data/cached/anny.pth"), weights_only=True
    )


def read_rig_cache(path: str | os.PathLike = RIG_CACHE) -> dict:
    """The rows of the mouth bones (``bone_labels``, ``blendshape_labels`` and the tensors of
    :data:`_PER_BONE` and :data:`_PER_ROW`), after checking that they were made for anny's
    current cache: the same blend-shape rows, in the same order."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    if metadata.get("format") != RIG_CACHE_FORMAT:
        raise ValueError(
            f"{path} is not a mouth rig cache ({metadata.get('format')!r})."
        )
    labels = json.loads(metadata["blendshape_labels"])
    anny_labels = [str(label) for label in _anny_cache()["blendshape_labels"]]
    if labels != anny_labels:
        raise ValueError(
            f"{path} was made for other blend shapes than data/cached/anny.pth; run "
            "python -m opensculptboy.mouth.authoring.rig_cache again."
        )
    return dict(
        tensors,
        bone_labels=json.loads(metadata["bone_labels"]),
        blendshape_labels=labels,
    )


@functools.lru_cache(maxsize=1)
def orientation_data() -> dict:
    """Anny's cached orientations (``data/cached/anny.pth``) with the rows of the mouth bones
    appended."""
    anny = _anny_cache()
    mouth = read_rig_cache()
    merged = dict(anny)
    merged["bone_labels"] = [str(b) for b in anny["bone_labels"]] + list(
        mouth["bone_labels"]
    )
    for key in _PER_BONE:
        merged[key] = torch.cat([anny[key], mouth[key].to(anny[key].dtype)])
    for key in _PER_ROW:
        merged[key] = torch.cat([anny[key], mouth[key].to(anny[key].dtype)], dim=1)
    return merged


def apply_cached_orientation(data, orientation: dict | None = None):
    """``data`` with the cached orientations of :func:`orientation_data`, as anny's
    ``apply_anny_cached_orientation`` gives them to the ``anny`` rig."""
    from anny.models.model_transforms import _select_blendshape_rows

    orientation = orientation_data() if orientation is None else orientation
    dtype, device = data.template_vertices.dtype, data.template_vertices.device
    source = {label: i for i, label in enumerate(orientation["bone_labels"])}
    labels = list(data.metadata.bone_labels)
    missing = [label for label in labels if label not in source]
    if missing:
        raise ValueError(
            f"The mouth rig's cached orientations lack the bones {missing}."
        )
    selection = [source[label] for label in labels]

    def per_bone(key):
        return orientation[key][selection].to(dtype=dtype, device=device)

    def per_row(key):
        return _select_blendshape_rows(
            orientation[key][:, selection],
            source_labels=orientation["blendshape_labels"],
            target_labels=data.metadata.blendshape_labels,
        ).to(dtype=dtype, device=device)

    return dataclasses.replace(
        data,
        template_bone_heads=per_bone("template_bone_heads"),
        bone_heads_blendshapes=per_row("bone_heads_blendshapes"),
        bone_template_orientation_matrices=per_bone(
            "bone_template_orientation_matrices"
        ),
        bone_orientation_blendshapes=per_row("bone_orientation_blendshapes"),
        reference_bone_orientations=per_bone("reference_bone_orientations"),
        bone_children_indices=None,
        bone_children_mask=None,
        bone_children_local_offsets=None,
        bone_nonzeroweight_mask=None,
        bone_vertex_indices=None,
        bone_vertex_weights=None,
        template_bone_vertices=None,
        template_bone_tails=None,
        bone_tails_blendshapes=None,
        bone_rolls_rotmat=None,
    )


def mouth_model_data(
    topology="anny",
    local_changes="none",
    facial_actions="none",
    face_shapes="none",
):
    """The model data of anny's ``topology`` (a topology spec of ``anny.Anny``) on the
    ``anny-mouth`` rig, with cached orientations."""
    from anny.models import build_model_data
    from anny.models.model_data import TopologyConfig

    topology = (
        TopologyConfig.from_string(topology) if isinstance(topology, str) else topology
    )
    data = build_model_data(
        rig=mouth_rig_config("blender"),
        topology=topology,
        local_changes=local_changes,
        facial_actions=facial_actions,
        face_shapes=face_shapes,
    )
    return apply_cached_orientation(data)


def mouth_model(
    topology: str = "anny",
    local_changes="none",
    facial_actions="none",
    phenotypes="default",
    extrapolate_phenotypes: bool = False,
    pose_parameterization="local-ref",
    skinning_method=None,
    face_shapes="none",
    scale_face_shapes: bool = True,
    rig: str = MOUTH_RIG,
):
    """An Anny model on the ``anny-mouth`` rig, with the arguments of :class:`anny.Anny`, on
    anny's topologies or on ``sculpt`` and ``sculpt-quads``."""
    import anny
    from anny.models.model_data import AnnyModelConfig

    if not is_mouth_rig(rig):
        raise ValueError(f"mouth_model builds the {MOUTH_RIG!r} rig; got {rig!r}.")
    from opensculptboy.topology.sculpt import is_sculpt, sculpt_model

    if is_sculpt(topology):
        return sculpt_model(
            rig=rig,
            topology=topology,
            local_changes=local_changes,
            facial_actions=facial_actions,
            phenotypes=phenotypes,
            extrapolate_phenotypes=extrapolate_phenotypes,
            pose_parameterization=pose_parameterization,
            skinning_method=skinning_method,
            face_shapes=face_shapes,
            scale_face_shapes=scale_face_shapes,
        )
    config = mouth_rig_config()
    data = mouth_model_data(
        topology=topology,
        local_changes=local_changes,
        facial_actions=facial_actions,
        face_shapes=face_shapes,
    )
    model = anny.Anny.from_model_data(
        data,
        skinning_method=skinning_method,
        pose_parameterization=pose_parameterization,
        bone_orientation=config.bone_orientation,
        root_identity_orientation=config.root_identity_orientation,
        phenotypes=phenotypes,
        extrapolate_phenotypes=extrapolate_phenotypes,
        scale_face_shapes=scale_face_shapes,
    )
    model.config = AnnyModelConfig(
        rig=rig,
        topology=topology,
        local_changes=local_changes,
        facial_actions=facial_actions,
        extrapolate_phenotypes=extrapolate_phenotypes,
        phenotypes=phenotypes,
        pose_parameterization=pose_parameterization,
        skinning_method=skinning_method,
        face_shapes=face_shapes,
        scale_face_shapes=scale_face_shapes,
    )
    return model


def split_head_weights(weights, indices, labels, model):
    """
    Skin weights of a file for ``model`` on the ``anny-mouth`` rig, from weights computed for
    the same body on the ``anny`` rig: ``weights`` and ``indices`` (V, K) index ``labels``,
    the bone labels of ``anny``.

    On ``anny`` the jaw's and the tongue's weights belong to ``head``. Each vertex here takes
    its weights as they are, except the weight of ``head``, which it shares among ``head``
    and the mouth bones in the proportions of ``model``'s own skin weights. Every vertex that
    the mouth bones do not move keeps its weights exactly, the edits of the VRM export (the
    eyelids, the twist bones, the refinement around the folds of the T-pose) among them.
    Returns the weights and indices (V, K + 12) on ``model``'s labels, the strongest first.
    """
    import numpy as np

    target = list(model.bone_labels)
    to_target = np.array([target.index(label) for label in labels])
    head = target.index("head")
    group = [head] + [target.index(bone) for bone in MOUTH_BONES]
    weights = np.asarray(weights, dtype=np.float64)
    indices = to_target[np.asarray(indices)]
    V, K = weights.shape
    dense = np.zeros((V, len(target)))
    np.add.at(
        dense, (np.repeat(np.arange(V), K), indices.reshape(-1)), weights.reshape(-1)
    )

    own_w = model.vertex_bone_weights.detach().cpu().double().numpy()
    own_i = model.vertex_bone_indices.detach().cpu().numpy()
    own = np.zeros((V, len(target)))
    np.add.at(
        own,
        (np.repeat(np.arange(V), own_w.shape[1]), own_i.reshape(-1)),
        own_w.reshape(-1),
    )
    share = own[:, group]
    total = share.sum(axis=1)
    moving = (total > 0) & (share[:, 1:].sum(axis=1) > 0)
    head_weight = dense[moving, head].copy()
    dense[moving, head] = 0.0
    dense[np.ix_(moving, group)] += head_weight[:, None] * (
        share[moving] / total[moving, None]
    )
    width = K + len(MOUTH_BONES)
    order = np.argsort(-dense, axis=1, kind="stable")[:, :width]
    result = np.take_along_axis(dense, order, axis=1)
    result /= result.sum(axis=1, keepdims=True)
    return result, np.where(result > 0, order, 0).astype(np.int64)

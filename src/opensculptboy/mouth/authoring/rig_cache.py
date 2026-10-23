# OpenSculptBoy
# Apache License, Version 2.0
"""
Build ``data/mouth/rig_cache.safetensors``: the cached orientations of the mouth bones of the
``anny-mouth`` rig (:mod:`opensculptboy.mouth.rig`).

The rows are computed as anny computes ``data/cached/anny.pth`` (``scripts/precompute_rig_caches.py``,
``main_anny``): on MakeHuman's rig with the same pruning and the mouth bones kept, with every
local change and face shape, the reference body of MakeHuman's defaults (``age=2/3``), the
squared skin weights of each bone and an aim at its own tail with the relative weight 0.5. The
blend-shape rows follow those of anny's cache, which the reader checks.

``--check BONE`` also computes a bone of anny's own rig the same way and prints its largest
difference from anny's cache, which shows that the computation matches anny's.

Run from the repository: ``python -m opensculptboy.mouth.authoring.rig_cache``.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import pathlib

import torch

from opensculptboy.mouth.rig import (
    MOUTH_BONES,
    RIG_CACHE,
    RIG_CACHE_FORMAT,
    _anny_cache,
)

REPOSITORY = pathlib.Path(__file__).resolve().parents[4]
AIM_WEIGHT = 0.5
STRATEGY = "skinning_squared"


def _precompute_module():
    path = REPOSITORY / "scripts" / "precompute_rig_caches.py"
    spec = importlib.util.spec_from_file_location("precompute_rig_caches", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_model():
    """MakeHuman's rig as anny's cache builder takes it, with the mouth bones kept."""
    import anny
    from anny.models.model_data import RigConfig

    rig = RigConfig.from_string(
        "makehuman-notongue-nobreasts-nofacialexpression-pruned"
    )
    rig = dataclasses.replace(
        rig, bones_to_remove=frozenset(rig.bones_to_remove - set(MOUTH_BONES))
    )
    return anny.Anny(
        rig=rig,
        topology="anny",
        local_changes="all",
        facial_actions="none",
        face_shapes="all",
    )


def compute(bones=MOUTH_BONES, model=None) -> dict:
    """The cached orientation rows of ``bones`` (labels of :func:`source_model`)."""
    pre = _precompute_module()
    model = source_model() if model is None else model
    labels = list(model.bone_labels)
    anny_labels = [str(label) for label in _anny_cache()["blendshape_labels"]]
    if list(model.blendshape_labels) != anny_labels:
        raise ValueError(
            "The source model's blend shapes differ from data/cached/anny.pth; append "
            "anny's cache first (scripts/precompute_rig_caches.py --append)."
        )
    picked = [labels.index(bone) for bone in bones]
    reference = model(phenotype_kwargs=dict(age=2 / 3))
    orientations = reference["rest_bone_poses"][0, :, :3, :3]
    weights = torch.stack(
        [pre._compute_bone_vertex_weights(model, j, STRATEGY) for j in picked]
    )
    rows = pre.compute_cached_orientation_data(
        template_vertices=model.template_vertices,
        blendshapes=model.blendshapes,
        template_bone_origins=model.template_bone_heads[picked],
        bone_origins_blendshapes=model.bone_heads_blendshapes[:, picked],
        bone_vertex_weights=weights,
        reference_vertices=reference["rest_vertices"][0],
        reference_bone_orientations=orientations[picked],
        reference_bone_origins=reference["rest_bone_heads"][0][picked],
        bone_labels=list(bones),
        aim_weight=AIM_WEIGHT,
        aim_target="tail",
        template_bone_tails=model.template_bone_tails[picked],
        bone_tails_blendshapes=model.bone_tails_blendshapes[:, picked],
        reference_bone_tails=reference["rest_bone_tails"][0][picked],
    )
    return dict(
        bone_labels=list(bones),
        blendshape_labels=anny_labels,
        template_bone_heads=model.template_bone_heads[picked].clone(),
        bone_heads_blendshapes=model.bone_heads_blendshapes[:, picked].clone(),
        reference_bone_orientations=orientations[picked].clone(),
        **rows,
    )


def write(rows: dict, path=RIG_CACHE) -> pathlib.Path:
    from safetensors.torch import save_file

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        key: value.detach().contiguous()
        for key, value in rows.items()
        if isinstance(value, torch.Tensor)
    }
    metadata = dict(
        format=RIG_CACHE_FORMAT,
        bone_labels=json.dumps(rows["bone_labels"]),
        blendshape_labels=json.dumps(rows["blendshape_labels"]),
        aim_weight=str(AIM_WEIGHT),
        strategy=STRATEGY,
    )
    save_file(tensors, str(path), metadata=metadata)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--out", default=str(RIG_CACHE))
    parser.add_argument(
        "--check",
        action="append",
        default=[],
        help="a bone of anny's rig to compute as a check against data/cached/anny.pth",
    )
    args = parser.parse_args(argv)
    model = source_model()
    if args.check:
        anny = _anny_cache()
        anny_bones = [str(b) for b in anny["bone_labels"]]
        rows = compute(args.check, model)
        for i, bone in enumerate(args.check):
            j = anny_bones.index(bone)
            for key in ("bone_template_orientation_matrices", "template_bone_heads"):
                diff = (rows[key][i] - anny[key][j].to(rows[key].dtype)).abs().max()
                print(f"{bone} {key}: largest difference {float(diff):.3e}")
            diff = (
                (
                    rows["bone_orientation_blendshapes"][:, i]
                    - anny["bone_orientation_blendshapes"][:, j].to(rows[key].dtype)
                )
                .abs()
                .max()
            )
            print(
                f"{bone} bone_orientation_blendshapes: largest difference {float(diff):.3e}"
            )
    rows = compute(MOUTH_BONES, model)
    print("wrote", write(rows, args.out))


if __name__ == "__main__":
    main()

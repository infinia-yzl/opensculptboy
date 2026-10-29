# OpenSculptBoy
# Apache License, Version 2.0
"""
The fit of a statue handoff: the reference fit with the statue's lost parts left without data.

A statue that lost a hand or its nose shows a break where the part was. The reference fit
would pull anny's hand onto the nearest marble (the thigh beside it, the stump of the wrist);
:func:`fit_statue` instead names the lost parts of the handoff's ``statue.lost`` list
(:data:`LOST_PARTS`, with the side: ``"hand.l"``, ``"forearm.r"``, ``"nose"``) to the fit as
vertices without data (``FitOptions.no_data``), so that anny keeps its own shape there and the
part follows the fitted skin around it. Where a support came off, the scan has an open border
along its cut; the refinement drops the pairs whose closest point lies on that border
(:func:`open_border`, the reference's ``"border"`` region), so that the skin does not crease
along the cut.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

# a lost part: the first bone of its chain in anny's rig (the part is every vertex whose
# skin weights lie mostly on that bone and the bones below it)
LOST_PARTS = {
    "hand": "wrist",
    "forearm": "lowerarm01",
    "arm": "upperarm01",
    "foot": "foot",
    "lower_leg": "lowerleg01",
    "leg": "upperleg01",
}
# the nose: the skin within this distance (metres) of the tip of anny's nose (pronasale)
NOSE_RADIUS = 0.014
# the rings of the scan's vertices around its open borders whose pairs the refinement drops
BORDER_RINGS = 2


def open_border(vertices: np.ndarray, triangles: np.ndarray, rings: int = BORDER_RINGS):
    """The vertices (V,) bool on the open borders of a mesh (the edges of one triangle, the
    cuts where supports came off and the holes of a scan) and ``rings`` rings around them,
    with the vertices at one position welded."""
    from opensculptboy.fit.register import welded_ids

    T = np.asarray(triangles)
    ids = welded_ids(np.asarray(vertices, dtype=np.float64))
    W = ids[T]
    edges = np.sort(np.concatenate([W[:, [0, 1]], W[:, [1, 2]], W[:, [2, 0]]]), 1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    on = np.zeros(ids.max() + 1, bool)
    on[unique[counts == 1].ravel()] = True
    for _ in range(rings):
        grown = on.copy()
        touched = on[W].any(1)
        grown[W[touched].ravel()] = True
        on = grown
    return on[ids]


def _subtree(model, first: str) -> np.ndarray:
    labels = list(model.bone_labels)
    parents = np.asarray(model.bone_parents)
    inside = np.zeros(len(labels), bool)
    inside[labels.index(first)] = True
    for _ in range(len(labels)):
        grown = inside | ((parents >= 0) & inside[np.clip(parents, 0, None)])
        if (grown == inside).all():
            break
        inside = grown
    return inside


def lost_mask(model, names, rest_vertices: np.ndarray | None = None) -> np.ndarray:
    """The vertices (V,) bool of the lost parts ``names`` on the fit model's level-0 mesh."""
    weights = model.vertex_bone_weights.detach().double().cpu().numpy()
    indices = model.vertex_bone_indices.cpu().numpy()
    mask = np.zeros(len(weights), bool)
    for name in names:
        key = str(name).strip().lower()
        if key == "nose":
            from opensculptboy.fit.register import _Craniofacial

            rest = (
                model.template_vertices.detach().double().cpu().numpy()
                if rest_vertices is None
                else np.asarray(rest_vertices, dtype=np.float64)
            )
            cranio = _Craniofacial(model)
            tip = (
                cranio.weights_matrix.detach()
                .double()
                .numpy()[cranio.labels.index("prn")]
            )
            mask |= np.linalg.norm(rest - tip @ rest, axis=1) < NOSE_RADIUS
            continue
        part, _, side = key.partition(".")
        if part not in LOST_PARTS or side not in ("l", "r"):
            raise ValueError(
                f"Unknown lost part {name!r}; use nose or one of "
                f"{sorted(LOST_PARTS)} with .l or .r."
            )
        bones = _subtree(model, f"{LOST_PARTS[part]}.{side.upper()}")
        share = (weights * bones[indices]).sum(1)
        mask |= share > 0.5
    return mask


def library_start(model, reference, names=None, samples: int = 6000, seed: int = 0):
    """
    The pose of anny's library (:mod:`anny.poses`) that comes closest to the reference, as
    the parameter stage's start: each pose on a body of the reference's height, standing on
    the floor and centred over the reference, scored by the mean distance both ways between
    its vertices and points of the reference (a chamfer distance, in metres). Returns the
    name, the (B, 3, 3) rotations and the scores by name. The reference stands upright in
    Anny's frame and faces -Y, as :func:`~opensculptboy.statues.scan_to_handoff` writes it.
    """
    import anny.poses
    import torch
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    R = reference.vertices
    target = R[rng.choice(len(R), min(samples, len(R)), replace=False)]
    target_tree = cKDTree(target)
    best_height, gap = 0.5, None
    with torch.no_grad():
        for h in np.linspace(0.0, 1.0, 11):
            rest = model(phenotype_kwargs={"height": float(h)})["rest_vertices"][0]
            g = abs(float(np.ptp(rest[:, 2].numpy())) - float(np.ptp(R[:, 2])))
            if gap is None or g < gap:
                best_height, gap = float(h), g
    phenotype = {"height": best_height}
    library = anny.poses.library()
    names = list(names or library.names("pose"))
    scores, poses = {}, {}
    for name in names:
        posed = anny.poses.pose_parameters(model, name, phenotype_kwargs=phenotype)
        V = posed["vertices"][0].double().numpy()
        if V.ndim == 3:
            V = V[0]
        V = V[rng.choice(len(V), min(samples, len(V)), replace=False)]
        V = V + [
            *(target[:, :2].mean(0) - V[:, :2].mean(0)),
            R[:, 2].min() - V[:, 2].min(),
        ]
        forward = cKDTree(V).query(target)[0].mean()
        backward = target_tree.query(V)[0].mean()
        scores[name] = float(forward + backward)
        poses[name] = posed["pose_parameters"][0, :, :3, :3].double().numpy()
    best = min(scores, key=scores.get)
    rest = np.eye(3)[None].repeat(len(model.bone_labels), 0)
    if "rest" not in scores:
        with torch.no_grad():
            V = model(phenotype_kwargs=phenotype)["vertices"][0].double().numpy()
        V = V[rng.choice(len(V), min(samples, len(V)), replace=False)]
        V = V + [
            *(target[:, :2].mean(0) - V[:, :2].mean(0)),
            R[:, 2].min() - V[:, 2].min(),
        ]
        scores["rest"] = float(
            cKDTree(V).query(target)[0].mean() + target_tree.query(V)[0].mean()
        )
        poses["rest"] = rest
        best = min(scores, key=scores.get)
    return best, poses[best], scores


def fit_statue(
    folder,
    detail: int = 2,
    name: str | None = None,
    options=None,
    start_pose: str | None = "auto",
):
    """
    Fit anny to a statue handoff folder (:func:`~opensculptboy.statues.scan_to_handoff`):
    :func:`~opensculptboy.fit.register.fit_reference` with the parts that ``statue.lost`` of
    appearance.json names left without data, the pairs on the scan's open borders dropped,
    and the parameter stage started from ``start_pose``: a pose of anny's library by name,
    ``"auto"`` for the closest one (:func:`library_start`) or None for the rest pose. Returns
    the :class:`~opensculptboy.fit.register.FitResult`.
    """
    from opensculptboy.fit.handoff import load_handoff
    from opensculptboy.fit.reference import Reference
    from opensculptboy.fit.register import FitOptions, fit_model, fit_reference

    folder = pathlib.Path(folder)
    appearance = json.loads((folder / "appearance.json").read_text(encoding="utf-8"))
    lost = list((appearance.get("statue") or {}).get("lost") or [])
    options = options or FitOptions()
    model = fit_model(options.topology)
    if lost:
        options = dataclasses.replace(options, no_data=lost_mask(model, lost))
    reference = Reference.from_handoff(load_handoff(folder))
    reference.regions["border"] = open_border(reference.vertices, reference.triangles)
    if start_pose and options.initial_pose is None:
        names = None if start_pose == "auto" else [start_pose]
        chosen, rotations, scores = library_start(model, reference, names)
        options = dataclasses.replace(options, initial_pose=rotations)
        if options.verbose:
            print(f"start pose: {chosen} ({1000 * scores[chosen]:.1f} mm)", flush=True)
    title = (appearance.get("statue") or {}).get("title")
    return fit_reference(
        reference,
        detail=detail,
        name=name or title or folder.resolve().name,
        options=options,
        model=model,
    )

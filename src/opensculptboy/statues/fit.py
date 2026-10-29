# OpenSculptBoy
# Apache License, Version 2.0
"""
The fit of a statue handoff: the reference fit with the statue's lost parts left without data.

A statue that lost a hand or its nose shows a break where the part was. The reference fit
would pull anny's hand onto the nearest marble (the thigh beside it, the stump of the wrist);
:func:`fit_statue` instead names the lost parts of the handoff's ``statue.lost`` list
(:data:`LOST_PARTS`, with the side: ``"hand.l"``, ``"forearm.r"``, ``"nose"``) to the fit as
vertices without data (``FitOptions.no_data``), so that anny keeps its own shape there and the
part follows the fitted skin around it.
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


def fit_statue(folder, detail: int = 2, name: str | None = None, options=None):
    """
    Fit anny to a statue handoff folder (:func:`~opensculptboy.statues.scan_to_handoff`):
    :func:`~opensculptboy.fit.register.fit_reference` with the parts that ``statue.lost`` of
    appearance.json names left without data. Returns the
    :class:`~opensculptboy.fit.register.FitResult`.
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
    title = (appearance.get("statue") or {}).get("title")
    return fit_reference(
        reference,
        detail=detail,
        name=name or title or folder.resolve().name,
        options=options,
        model=model,
    )

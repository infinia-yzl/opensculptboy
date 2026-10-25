# OpenSculptBoy
# Apache License, Version 2.0
"""
MakeHuman's base mesh on a card's body, for assets bound to its vertex indices.

Anny's template comes from MakeHuman's ``base.obj`` (19,158 vertices) in Anny's frame
(:data:`MH_TO_ANNY`: decimetres to metres, Y up to Z up). Anny's default topology keeps the
vertices that its faces use (13,718 rows: the body, the eyes and the tongue); it leaves out
the helper geometry (``helper-tights``, ``helper-skirt``, ``helper-hair`` and the others)
and the vertices of the nudity edits. ``model.base_mesh_vertex_indices`` gives the base-mesh
index of each row. The MPFB2 targets that Anny bundles move every base-mesh vertex, the
helpers included, so :func:`base_mesh_vertices` evaluates the rows that Anny lacks from the
same targets with the card's blend-shape coefficients, and takes Anny's rows from the card's
body (its detail layer included).
"""

from __future__ import annotations

import functools

import numpy as np

BASE_VERTEX_COUNT = 19158
# MakeHuman's frame (decimetres, Y up) to Anny's (metres, Z up): 0.1 times a turn of 90
# degrees about X, as anny.models.full_model.load_data applies to base.obj.
MH_TO_ANNY = 0.1 * np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def base_indices(model, count: int | None = None) -> np.ndarray:
    """(N,) the base-mesh index of each of Anny's rows of ``model`` (the first ``count``)."""
    base = model.base_mesh_vertex_indices.detach().cpu().numpy().astype(np.int64)
    return base if count is None else base[:count]


def rows_of_base(model, count: int | None = None) -> np.ndarray:
    """(19158,) the row of Anny's template for each base-mesh index, or -1."""
    base = base_indices(model, count)
    rows = np.full(BASE_VERTEX_COUNT, -1, np.int64)
    rows[base] = np.arange(len(base))
    return rows


@functools.lru_cache(maxsize=1)
def _full_targets(rows: tuple) -> tuple[np.ndarray, np.ndarray, tuple]:
    """The template rows and the blend shapes of the base-mesh vertices ``rows``, with the
    blend-shape labels, from the model data that Anny caches for its own rig."""
    from anny.models.full_model import load_data
    from anny.models.model_data import _parse_rig_spec

    rig, weights = _parse_rig_spec("anny").resolve_filenames()
    data = load_data(weights_filename=weights, rig_filename=rig, eyes=True, tongue=True)
    index = list(rows)
    template = data.template_vertices[index].detach().cpu().double().numpy()
    shapes = data.blendshapes[:, index].detach().cpu().double().numpy()
    labels = tuple(data.metadata.blendshape_labels)
    del data
    return template, shapes, labels


def blendshape_coefficients(character, model) -> np.ndarray:
    """(K,) the coefficients of ``model``'s blend shapes for the card's sliders."""
    import torch

    with torch.no_grad():
        _, phenotype, local, facial = model.get_tensor_inputs(
            None,
            dict(character.phenotype) or None,
            dict(character.local_changes) or None,
            None,
        )
        face = model._parse_parameter_kwargs(
            dict(character.face_shapes) or None,
            model.face_shape_labels,
            0.0,
            "face_shape_kwargs",
        )
        coeffs = model._get_phenotype_blendshape_coefficients(
            phenotype, local, facial, face
        )
    return coeffs[0].detach().cpu().double().numpy()


def target_rows(character, model, rows) -> np.ndarray:
    """(R, 3) the base-mesh vertices ``rows`` on the card's body at rest (Anny's frame),
    evaluated from MakeHuman's targets with the card's coefficients. The face shapes and the
    detail layer move only Anny's own rows, so they leave these rows as the targets give."""
    rows = tuple(int(r) for r in rows)
    if not rows:
        return np.zeros((0, 3))
    template, shapes, labels = _full_targets(rows)
    coeffs = blendshape_coefficients(character, model)
    index = {label: k for k, label in enumerate(labels)}
    out = template.copy()
    for k, label in enumerate(model.blendshape_labels):
        j = index.get(label)
        if j is not None and coeffs[k] != 0.0:
            out += coeffs[k] * shapes[j]
    return out


def base_mesh_vertices(
    character, model, body, needed=None
) -> tuple[np.ndarray, np.ndarray]:
    """
    (19158, 3) the vertices of MakeHuman's base mesh on the body of a card at rest (Anny's
    frame), and (19158,) True where the vertex came from the targets (a row that Anny's
    template lacks). ``body`` is the card's
    :class:`~opensculptboy.wearables.bind.BodySurface`; ``needed`` limits the evaluation from
    the targets to these base-mesh indices (the others stay NaN).
    """
    rows = rows_of_base(model, len(body.vertices))
    out = np.full((BASE_VERTEX_COUNT, 3), np.nan)
    present = rows >= 0
    out[present] = body.vertices[rows[present]]
    missing = np.flatnonzero(~present)
    if needed is not None:
        missing = np.intersect1d(missing, np.asarray(needed, dtype=np.int64))
    if len(missing):
        out[missing] = target_rows(character, model, missing)
    from_targets = np.zeros(BASE_VERTEX_COUNT, bool)
    from_targets[missing] = True
    return out, from_targets


def base_groups() -> dict[str, np.ndarray]:
    """The vertex indices of each face group of ``base.obj`` (``body``, ``helper-tights``,
    ``helper-skirt``, ...)."""
    return {k: v.copy() for k, v in _base_groups().items()}


@functools.lru_cache(maxsize=1)
def _base_groups() -> dict[str, np.ndarray]:
    import os

    import torch

    import anny.utils.obj_utils as obj_utils
    from anny.paths import get_anny_root_dir

    path = os.path.join(get_anny_root_dir(), "data/mpfb2/3dobjs/base.obj")
    _, _, groups = obj_utils.load_obj_file(path, dtype=torch.float64)
    return {
        name: np.unique(group["face_vertex_indices"].numpy().reshape(-1))
        for name, group in groups.items()
    }

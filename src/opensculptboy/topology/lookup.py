# OpenSculptBoy
# Apache License, Version 2.0
"""
Per-vertex lookups that work on every topology that keeps anny's vertices: ``anny``,
``anny-quads`` and the ``sculpt`` topologies.

Tables of MakeHuman's base mesh (face labels, keypoint weights, MediaPipe's map) are indexed by
``model.base_mesh_vertex_indices``. Anny's vertices keep their base-mesh index on ``sculpt``,
and the new vertices of the patches take indices from 19158 up (past every table), so a lookup
of a base-mesh table needs a value for them. Each new vertex is a fixed convex blend of anny's
vertices (its anchors, :mod:`opensculptboy.topology.sculpt`), and it takes its value through
them:

- :func:`base_mesh_rows`: the base-mesh index of each vertex, a new vertex taking that of its
  strongest anchor (for labels and other categorical tables);
- :func:`base_mesh_values`: a numeric table of base-mesh vertices on the model's vertices, a
  new vertex taking the weighted mean of its anchors' values (``blend=True``) or the value of
  its strongest anchor;
- :func:`patch_labels`: the patch label of each vertex (``""`` for anny's own vertices), such
  as ``"ear_root"`` or a finer label of the patch (``"ear_root/line"``);
- :func:`ear_root_line`: the mask of the root line of the outer ears when the ear-root patch
  marks one (its new vertices labelled :data:`EAR_ROOT_LINE_LABEL`), else None.

The model finds its sculpt data in the attribute ``sculpt_data`` that
:func:`~opensculptboy.topology.sculpt.sculpt_model` sets; a model of another topology holds
none, and every lookup is then the plain lookup.
"""

from __future__ import annotations

import numpy as np

# The vertex label of the new vertices that the ear-root patch puts on the line where the
# outer ear meets the head (``Patch.vertex_labels``). The fit's ear split parts the outer ear
# from the head along that line (opensculptboy.fit.register.pinna).
EAR_ROOT_LINE_LABEL = "ear_root/line"


def model_sculpt_data(model):
    """The :class:`~opensculptboy.topology.sculpt.SculptData` that a model was built with, or
    None for a model of another topology (anny's own vertex count)."""
    data = getattr(model, "sculpt_data", None)
    if data is not None:
        return data
    count = len(model.template_vertices)
    base = getattr(model, "base_mesh_vertex_indices", None)
    if base is None:
        return None
    from opensculptboy.topology.sculpt import MAKEHUMAN_VERTEX_COUNT

    if int(base.max()) < MAKEHUMAN_VERTEX_COUNT:
        return None
    # a model built from sculpt ModelData by other means: the current sculpt data, if it fits
    from opensculptboy.topology.sculpt import sculpt_data

    data = sculpt_data()
    if data.total_vertex_count != count:
        raise ValueError(
            f"The model has {count} vertices with new sculpt vertices, and the current "
            f"sculpt topology {data.total_vertex_count}."
        )
    return data


def vertex_anchors(model) -> tuple[np.ndarray, np.ndarray]:
    """(V, K) anchors and weights of every vertex of the model: identity rows for anny's
    vertices (K = 1 on other topologies) and the convex rows of the new vertices."""
    count = len(model.template_vertices)
    data = model_sculpt_data(model)
    if data is None:
        return np.arange(count)[:, None], np.ones((count, 1))
    n = data.vertex_count
    k = data.anchors.shape[1]
    anchors = np.zeros((count, k), np.int64)
    weights = np.zeros((count, k))
    anchors[:n, 0] = np.arange(n)
    weights[:n, 0] = 1.0
    anchors[n:] = data.anchors
    weights[n:] = data.weights
    return anchors, weights


def strongest_anchor(model) -> np.ndarray:
    """(V,) the anny vertex of each vertex: itself, or the strongest anchor of a new vertex
    (the first of equal weights)."""
    anchors, weights = vertex_anchors(model)
    return anchors[np.arange(len(anchors)), np.argmax(weights, 1)]


def base_mesh_rows(model) -> np.ndarray:
    """(V,) the MakeHuman base-mesh index of each vertex, a new vertex taking that of its
    strongest anchor; index base-mesh tables of labels with it."""
    base = model.base_mesh_vertex_indices.detach().cpu().numpy().astype(np.int64)
    if model_sculpt_data(model) is None:
        return base
    return base[strongest_anchor(model)]


def base_mesh_values(model, table, blend: bool = True) -> np.ndarray:
    """
    A table of MakeHuman's base-mesh vertices (B, ...) on the model's vertices (V, ...): anny's
    vertices read their row, and a new vertex takes the weighted mean of its anchors' rows
    (``blend``) or the row of its strongest anchor.
    """
    table = np.asarray(table)
    base = model.base_mesh_vertex_indices.detach().cpu().numpy().astype(np.int64)
    if model_sculpt_data(model) is None:
        return table[base]
    if not blend:
        return table[base[strongest_anchor(model)]]
    anchors, weights = vertex_anchors(model)
    rows = table[base[anchors]].astype(np.float64)  # (V, K, ...)
    w = weights.reshape(weights.shape + (1,) * (rows.ndim - 2))
    return (w * rows).sum(1)


def patch_labels(model) -> np.ndarray:
    """(V,) the patch label of each vertex: ``""`` for anny's vertices, else the label of the
    new vertex in the sculpt data (its patch name or a finer label such as
    ``"eye_corner/caruncle"``)."""
    count = len(model.template_vertices)
    out = np.full(count, "", dtype=object)
    data = model_sculpt_data(model)
    if data is None:
        return out
    labels = np.array(list(data.labels) or [""], dtype=object)
    out[data.vertex_count :] = labels[data.vertex_labels]
    return out


def labelled(model, prefixes) -> np.ndarray:
    """(V,) mask of the new vertices whose patch label equals one of ``prefixes`` or starts
    with one of them followed by ``/``."""
    labels = patch_labels(model)
    prefixes = (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
    out = np.zeros(len(labels), bool)
    for p in prefixes:
        out |= np.array([x == p or x.startswith(p + "/") for x in labels], bool)
    return out


def referenced_vertices(model) -> np.ndarray:
    """(V,) mask of the vertices that a face of the model uses: on ``sculpt``, anny's vertices
    inside the removed faces stay in the vertex list without a face."""
    count = len(model.template_vertices)
    out = np.zeros(count, bool)
    out[np.unique(model.faces.detach().cpu().numpy())] = True
    return out


def ear_root_line(model) -> np.ndarray | None:
    """(V,) mask of the root line of the outer ears (the new vertices labelled
    :data:`EAR_ROOT_LINE_LABEL`), or None when the model has none."""
    mask = labelled(model, EAR_ROOT_LINE_LABEL)
    return mask if mask.any() else None


def model_topology(model) -> str:
    """The topology name of a model (``"anny"`` when it does not record one)."""
    config = getattr(model, "config", None)
    topology = getattr(config, "topology", None)
    if isinstance(topology, str):
        return topology
    return "sculpt" if model_sculpt_data(model) is not None else "anny"

# OpenSculptBoy
# Apache License, Version 2.0
"""
Synthetic references for the tests of the reference fit (``opensculptboy.fit``).

:func:`synthetic_target` makes a figure that anny can only reproduce with its detail layer:
random anny parameters and a small pose, a smooth random offset field on top (a broad one on
level 0 and a finer one on level 1 of the subdivision), evaluated on level 2, posed and moved
by a random rigid placement. :func:`remesh` then gives it another topology: the vertices slide
within the surface, and vertex clustering on a grid with a random origin merges them, so that
no vertex or triangle of anny's levels survives. The eyeballs and the insides of the mouth and
the eye sockets are left out, as a game character's skin mesh has them as separate parts.
"""

from __future__ import annotations

import functools

import numpy as np
import torch

from opensculptboy.detail import DetailModel, detail_topology, level


def smooth_field(vertices, triangles, rng, iterations: int, amplitude: float):
    """A random vector field (N, 3) smoothed by Laplacian averaging, scaled to an RMS
    length of ``amplitude`` metres."""
    t = np.asarray(triangles)
    edges = np.unique(
        np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), 1), axis=0
    )
    degree = np.maximum(np.bincount(edges.ravel(), minlength=len(vertices)), 1)
    field = rng.standard_normal((len(vertices), 3))
    for _ in range(iterations):
        total = np.zeros_like(field)
        np.add.at(total, edges[:, 0], field[edges[:, 1]])
        np.add.at(total, edges[:, 1], field[edges[:, 0]])
        field = field + 0.5 * (total / degree[:, None] - field)
    field -= field.mean(0)
    return amplitude * field / np.sqrt((field**2).sum(1).mean())


def remesh(vertices, triangles, rng, cell: float = 0.005, jitter: float = 0.3):
    """
    Another triangulation of a surface: each vertex slides by up to ``jitter`` of the mean
    edge length within its tangent plane, then the vertices of each cell of a grid of size
    ``cell`` (random origin) merge into their mean, and degenerate triangles drop out.
    """
    from opensculptboy.export.body import vertex_normals

    V = np.asarray(vertices, dtype=np.float64)
    T = np.asarray(triangles)
    N = vertex_normals(V, T)
    edge = np.linalg.norm(V[T[:, 0]] - V[T[:, 1]], axis=1).mean()
    slide = rng.uniform(-1, 1, V.shape) * jitter * edge
    slide -= (slide * N).sum(1, keepdims=True) * N
    V = V + slide
    keys = np.floor((V + rng.uniform(0, cell, 3)) / cell).astype(np.int64)
    _, cluster, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    cluster = cluster.ravel()
    merged = np.zeros((len(counts), 3))
    np.add.at(merged, cluster, V)
    merged /= counts[:, None]
    tris = cluster[T]
    keep = (
        (tris[:, 0] != tris[:, 1])
        & (tris[:, 1] != tris[:, 2])
        & (tris[:, 0] != tris[:, 2])
    )
    tris = tris[keep]
    tris = np.unique(np.sort(tris, axis=1), axis=0, return_index=True)[1]
    tris = cluster[T][keep][np.sort(tris)]
    used = np.unique(tris)
    remap = -np.ones(len(merged), np.int64)
    remap[used] = np.arange(len(used))
    return merged[used], remap[tris]


@functools.lru_cache(maxsize=4)
def synthetic_target(seed: int = 0, cell: float = 0.005):
    """
    A synthetic reference and what made it: a dict with ``vertices`` and ``triangles`` (the
    remeshed target in Anny's frame), ``landmarks`` (the contract's names, with 1 mm of noise)
    and the ``truth`` (card values, placement and the level-2 surface before the remesh).
    """
    import anny
    import roma

    from opensculptboy.fit.reference import LANDMARKS
    from opensculptboy.fit.register import excluded_vertices

    rng = np.random.default_rng(seed)
    model = anny.Anny(local_changes="all", face_shapes="all", facial_actions="none").to(
        dtype=torch.float64
    )
    phenotype = {k: float(rng.uniform(0.3, 0.7)) for k in model.phenotype_labels}
    local = {
        k: float(rng.uniform(-0.4, 0.4))
        for k in rng.choice(model.local_change_labels, 12, replace=False)
    }
    face = {}
    for k in rng.choice(model.face_shape_labels, 10, replace=False):
        lo, hi = model.face_shape_ranges[k]
        face[str(k)] = float(np.clip(rng.uniform(-0.4, 0.4), lo, hi))
    labels = list(model.bone_labels)
    pose = torch.eye(4, dtype=torch.float64).repeat(1, len(labels), 1, 1)
    for bone, scale in (
        ("upperarm01.L", 0.15),
        ("upperarm01.R", 0.15),
        ("lowerarm01.L", 0.15),
        ("lowerarm01.R", 0.15),
        ("upperleg01.L", 0.05),
        ("upperleg01.R", 0.05),
        ("spine02", 0.05),
        ("neck02", 0.05),
    ):
        rotvec = torch.as_tensor(rng.uniform(-scale, scale, 3))
        pose[0, labels.index(bone), :3, :3] = roma.rotvec_to_rotmat(rotvec)
    kwargs = dict(
        pose_parameters=pose,
        phenotype_kwargs=phenotype,
        local_changes_kwargs=local,
        face_shape_kwargs=face,
    )

    # the offset field: broad on level 0, finer on level 1, carried to level 2
    l0, l1 = level("anny", 0), level("anny", 1)
    t0 = detail_topology(model, 0)
    t1 = detail_topology(model, 1)
    with torch.no_grad():
        rest0 = model(**kwargs)["rest_vertices"][0].numpy()
    broad = smooth_field(rest0, t0.triangles, rng, 30, 0.003)
    fine = smooth_field(t1.apply(rest0), t1.triangles, rng, 4, 0.0008)
    field1 = t1.apply(broad) + fine
    two = level("anny", 2)
    field2 = two.operators[-1].apply(field1)
    detailed = DetailModel(model, 2, offsets=field2)
    with torch.no_grad():
        posed = detailed(**kwargs)["vertices"][0].numpy()

    # the skin alone: no eyeballs, eye sockets or mouth cavity
    excluded = excluded_vertices(model)
    excluded2 = np.asarray(detailed.topology.apply(excluded.astype(float))) > 0.01
    triangles = detailed.triangles
    triangles = triangles[~excluded2[triangles].any(1)]

    angle = rng.uniform(-0.2, 0.2)
    rotation = roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, angle])).numpy()
    translation = rng.uniform(-0.3, 0.3, 3)
    surface = posed @ rotation.T + translation
    vertices, faces = remesh(surface, triangles, rng, cell)

    # landmarks of the true figure
    from anny.keypoints import KeypointsRegressor

    regressor = KeypointsRegressor.craniofacial(model)
    with torch.no_grad():
        out = model(**kwargs)
    cranio = regressor(out)[0].numpy()
    joints = out["bone_poses"][0, :, :3, 3].numpy()
    landmarks = {}
    for name, (kind, label, _) in LANDMARKS.items():
        point = (
            joints[labels.index(label)]
            if kind == "bone"
            else cranio[regressor.labels.index(label)]
        )
        noise = rng.normal(0.0, 0.001, 3)
        landmarks[name] = rotation @ point + translation + noise
    return dict(
        vertices=vertices,
        triangles=faces,
        landmarks=landmarks,
        truth=dict(
            phenotype=phenotype,
            local_changes=local,
            face_shapes=face,
            rotation=rotation,
            translation=translation,
            surface=surface,
            surface_triangles=triangles,
            levels=(l0.vertex_count, l1.vertex_count, two.vertex_count),
        ),
    )

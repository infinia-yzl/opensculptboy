# OpenSculptBoy
# Apache License, Version 2.0
"""
A synthetic statue scan for the tests of ``opensculptboy.statues``.

:func:`synthetic_statue` stands anny in the library's contrapposto (``weight_shift``) with a
heroic build, and gives it what a scan of a marble statue has: a plinth under the feet, a tree
stump fused to the right leg (the leg's skin inside the stump is gone, as a scan sees only the
outside), a lost left hand, and optionally a chipped nose, a sculpted surface (a smooth offset
field on level 1 of the detail layer) and a mass of short curls over the scalp. The scan is in
millimetres with +Y up and +X to the front, as many museum files are, and can be remeshed to
another topology with a little noise.
"""

from __future__ import annotations

import functools

import numpy as np
import torch

HEROIC = {
    "gender": 1.0,
    "age": 0.5,
    "muscle": 0.85,
    "weight": 0.5,
    "height": 0.7,
    "proportions": 0.85,
}
# anny's frame (Z up, facing -Y) to the scan's (+Y up, +X to the front), before the units
ANNY_TO_SCAN = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
# the stump: centre (x, y) in anny's frame, radius and height (metres)
STUMP = ((-0.2, 0.06), 0.075, 0.7)
PLINTH = (0.7, 0.7, 0.12)


@functools.lru_cache(maxsize=1)
def _model():
    from opensculptboy.fit.register import fit_model

    return fit_model()


def _cylinder(centre, radius, height, sections=48, rings=24):
    angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    heights = np.linspace(0.0, height, rings)
    ring = np.stack([np.cos(angles), np.sin(angles)], 1) * radius + np.asarray(centre)
    V = np.concatenate([np.c_[ring, np.full(sections, h)] for h in heights])
    T = []
    for r in range(rings - 1):
        for s in range(sections):
            a, b = r * sections + s, r * sections + (s + 1) % sections
            T += [(a, b, b + sections), (a, b + sections, a + sections)]
    top = len(V)
    V = np.vstack([V, [[centre[0], centre[1], height]]])
    last = (rings - 1) * sections
    T += [(last + s, last + (s + 1) % sections, top) for s in range(sections)]
    return V, np.asarray(T, np.int64)


def _box(size, top=0.0):
    import trimesh

    box = trimesh.creation.box(extents=size)
    V = np.asarray(box.vertices) + [0.0, 0.0, top - 0.5 * size[2]]
    return V, np.asarray(box.faces, np.int64)


def synthetic_statue(
    seed: int = 0,
    sculpt: float = 0.0,
    curls: float = 0.0,
    chip_nose: bool = False,
    cell: float | None = None,
    noise: float = 0.0,
    units: float = 1000.0,
) -> dict:
    """
    A synthetic statue scan and what made it: ``vertices`` and ``triangles`` (the scan, in its
    own frame and units), ``body`` (vertices, triangles: the whole figure before the stump,
    the plinth and the losses, in anny's frame, metres), ``kept`` (the triangles of ``body``
    that the scan shows), ``card`` (the phenotype and the pose), ``landmarks`` (contract names,
    anny's frame, on the figure as made), ``stump_mask`` (a box mask in the handoff's frame
    around the stump), ``up`` and ``front`` (the scan's axes).
    """
    import anny.poses
    from opensculptboy.detail import DetailModel, detail_topology, level
    from opensculptboy.fit.metrics import REGIONS, anny_regions
    from opensculptboy.fit.reference import LANDMARKS
    from opensculptboy.fit.register import _Craniofacial, excluded_vertices
    from test.fit_fixtures import remesh, smooth_field

    rng = np.random.default_rng(seed)
    model = _model()
    posed = anny.poses.pose_parameters(model, "weight_shift", phenotype_kwargs=HEROIC)
    pose = posed["pose_parameters"]
    kwargs = dict(pose_parameters=pose, phenotype_kwargs=HEROIC)
    with torch.no_grad():
        out = model(**kwargs)
    rest0 = out["rest_vertices"][0].double().numpy()
    excluded = excluded_vertices(model)
    regions0 = anny_regions(model, rest0)
    if sculpt > 0 or curls > 0:
        one = level("anny", 1)
        t1 = detail_topology(model, 1)
        field = np.zeros((one.vertex_count, 3))
        if sculpt > 0:
            field += smooth_field(t1.apply(rest0), t1.triangles, rng, 12, sculpt)
        if curls > 0:
            # short curls: the scalp lifted into a mass with a bumpy surface
            from opensculptboy.fit.reference import vertex_normals

            scalp = np.asarray(t1.apply((regions0 == REGIONS.index("scalp")) * 1.0))
            rest1 = np.asarray(t1.apply(rest0))
            normals = vertex_normals(rest1, t1.triangles)
            bumps = smooth_field(rest1, t1.triangles, rng, 2, 0.35 * curls)
            lift = np.clip(scalp, 0, 1)[:, None] * (
                curls * normals + (bumps * normals).sum(1, keepdims=True) * normals
            )
            field += lift
        detailed = DetailModel(model, 1, offsets=field)
        with torch.no_grad():
            V = detailed(**kwargs)["vertices"][0].double().numpy()
        T = detailed.triangles
        closed = np.asarray(detailed.topology.apply(excluded * 1.0)) > 0.01
        regions = (
            np.asarray(
                detailed.topology.apply(np.eye(len(REGIONS) + 1)[regions0 + 1])
            ).argmax(1)
            - 1
        )
    else:
        V = out["vertices"][0].double().numpy()
        T = model.faces.cpu().numpy()
        closed = excluded
        regions = regions0
    # the posed figure stands on z = 0 (anny.poses grounds it)
    V = V - [0.0, 0.0, V[:, 2].min()]
    T = T[~closed[T].any(1)]
    body = (V.copy(), T.copy())

    kept = np.ones(len(T), bool)
    centroid = V[T].mean(1)
    # the lost left hand (+X is the figure's left) and the stump's inside
    left_hand = (regions == REGIONS.index("hands")) & (V[:, 0] > 0)
    kept &= ~left_hand[T].any(1)
    (cx, cy), radius, height = STUMP
    inside = (np.hypot(centroid[:, 0] - cx, centroid[:, 1] - cy) < radius) & (
        centroid[:, 2] < height
    )
    kept &= ~inside
    cranio = _Craniofacial(model)
    with torch.no_grad():
        placed = out["vertices"][0].double().numpy() - [
            0.0,
            0.0,
            out["vertices"][0, :, 2].min().item(),
        ]
    points = cranio.weights_matrix.detach().double().numpy() @ placed
    named = dict(zip(cranio.labels, points))
    if chip_nose:
        kept &= np.linalg.norm(centroid - named["prn"], axis=1) > 0.008
    landmarks = {}
    joints = out["bone_poses"][0, :, :3, 3].double().numpy() - [
        0.0,
        0.0,
        out["vertices"][0, :, 2].min().item(),
    ]
    labels = list(model.bone_labels)
    for name, (kind, label, _) in LANDMARKS.items():
        if kind == "bone":
            landmarks[name] = joints[labels.index(label)]
        elif label in named:
            landmarks[name] = named[label]

    sv, st = V, T[kept]
    pieces = [(sv, st)]
    pieces.append(_cylinder((cx, cy), radius, height))
    pieces.append(_box(PLINTH, top=0.0))
    offsets = np.cumsum([0] + [len(p[0]) for p in pieces])
    S = np.concatenate([p[0] for p in pieces])
    F = np.concatenate([p[1] + offsets[k] for k, p in enumerate(pieces)])
    used = np.unique(F)
    remap = np.full(len(S), -1)
    remap[used] = np.arange(len(used))
    S, F = S[used], remap[F]
    if cell:
        S, F = remesh(S, F, rng, cell)
    if noise > 0:
        S = S + rng.normal(0.0, noise, S.shape)
    scan = (S @ ANNY_TO_SCAN.T) * units
    return {
        "vertices": scan,
        "triangles": F,
        "body": body,
        "kept": kept,
        "card": {"phenotype": HEROIC, "pose": pose},
        "landmarks": landmarks,
        "up": ANNY_TO_SCAN @ np.array([0.0, 0.0, 1.0]),
        "front": ANNY_TO_SCAN @ np.array([0.0, -1.0, 0.0]),
        "units": units,
    }


def to_scan(statue: dict, points) -> np.ndarray:
    """Points of the statue as made (anny's frame, metres) in the scan's frame."""
    return (np.asarray(points, dtype=np.float64) @ ANNY_TO_SCAN.T) * statue["units"]


def stump_mask(statue: dict, result) -> dict:
    """A box mask around the stump in the handoff's frame of a conversion ``result``, as a
    person would draw it on the converted figure."""
    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    (cx, cy), radius, height = STUMP
    centre = result.to_anny(to_scan(statue, [[cx, cy, 0.5 * height]]))[0]
    size = result.scale * statue["units"]
    return {
        "kind": "box",
        "role": "support",
        "centre": [float(x) for x in centre @ GLTF_TO_ANNY],
        # glTF's (x, y, z) sizes: anny's x, z and y
        "size": [2.4 * radius * size, (height + 0.03) * size, 2.4 * radius * size],
    }


def landmarks_in_handoff(statue: dict, result, names=None, jitter: float = 0.0, seed=0):
    """The statue's landmarks in the handoff's frame of ``result``, as a person would place
    them: ``names`` only, each moved by up to ``jitter`` metres."""
    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    rng = np.random.default_rng(seed)
    out = {}
    for name, point in statue["landmarks"].items():
        if names is not None and name not in names:
            continue
        p = result.to_anny(to_scan(statue, [point]))[0]
        p = p + rng.uniform(-jitter, jitter, 3)
        out[name] = [float(x) for x in p @ GLTF_TO_ANNY]
    return out

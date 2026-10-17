# OpenSculptBoy
# Apache License, Version 2.0
"""
The skin-weight edits of the VRM files (``data/vrm/skin_weights.safetensors``).

    python -m opensculptboy.export.authoring.skin_weights           # optimise, write, report
    python -m opensculptboy.export.authoring.skin_weights --report  # report on the data file

The inverse bind of the VRM T-pose (:func:`opensculptboy.export.tpose.rebind`) is the mesh
that the file's 4 weights skin back onto Anny's rest mesh. Where the T-pose straightens the
fingers and the legs, that mesh folds: at the webs of the fingers and the thumb, at the groin
and at the neck, edges bend by more than 120 degrees where Anny's rest mesh bends less
(:mod:`opensculptboy.export.mesh_checks`).

:func:`optimise` edits the weights of the vertices of those folds and of two rings around
them (:func:`fold_region`). Each vertex blends its candidate bones (its own and those of its
neighbours) by a softmax of logits that depend on the six default phenotypes through the
features of ``vrm_tables.edit_features`` (``Settings.features``: 1 for one weight set for
every body, 7 for logits linear in the phenotypes, 28 for quadratic ones), so the export
evaluates a few dot products and pays nothing more. The loss sums, over every body of
:func:`bodies`:

- the excess of each bend of the T-posed bind mesh over max(90 degrees, its rest bend),
  squared, and the triangles that shrink below 0.4 of their rest area;
- the same excess for the posed file in the hand poses (:data:`HAND_POSES` and the finger
  curls of :data:`CURLS`) over max(110 degrees, the bend of Anny's own posed mesh, the bend
  with the model's weights), and the triangles that shrink below 0.4 of their area on Anny's
  posed mesh (or below their share with the model's weights);
- the distances from Anny's posed mesh in the hand poses above caps of the model's weights:
  the largest distance of the hand in that pose, and the 99th percentile of the hand (or the
  vertex's own distance), and the increase of each distance;
- the mean squared distances of both binds from Anny's posed mesh over the library poses and
  clips (:data:`FRAMES`). The posed vertex is the product of a matrix of the frame and the
  weights times the bind vertex, so each mean is a quadratic form, whose matrix sums every
  frame once per body;
- the distances in the library frames above caps of the model's weights, on the bodies of
  :func:`cap_bodies`: the largest distance of the whole mesh in that entry, and in ``relaxed``
  and ``walk`` the 99th percentile (or the vertex's own distance). :func:`active_pairs` picks
  the pairs (frame, vertex) near their caps every few steps;
- small terms on the change of the weights and, before the second phase, on the weight
  beyond 4 bones.

The posed file vertex of the inverse bind is (sum_k w_k W_k T_k^-1)(sum_k w_k B_k T_k^-1)^-1 v,
with W_k Anny's posed bone poses, T_k the T-pose and B_k the rest pose, and torch
differentiates it as it stands. A second phase keeps the 4 bones of each vertex that the
bodies favour on average and optimises again, so that the file holds the weights exactly.
The forward bind keeps Anny's own weights; the edits change only the weights with which the
file is skinned.

:func:`gate_report` compares the edits with the model's weights (both twist modes) on these
gates, and the command writes the data file only when all of them hold (or with ``--force``):

- the T-pose: no new fold on the default and fixture bodies, and at most 5 % of the new folds
  of the model's weights over the 66 bodies of :func:`gate_bodies` (bodies drawn inside the
  phenotype cube are reported for information);
- the hand poses on the five bodies of :data:`HAND_BODIES`: no more new folds and collapsed
  triangles than the model's weights, and the largest and 99th-percentile distances of the
  hand from Anny's posed mesh within 10 % of theirs;
- :func:`~opensculptboy.export.tpose.bind_error` (both binds) and
  :func:`~opensculptboy.export.tpose.humanoid_error` on the three bodies of the tests within
  5 % of the model's weights, and the largest distance of every library pose and clip on
  those bodies within 10 %;
- the forward bind keeps Anny's own weights.

No weight set meets them all on its own. The hard-capped set below ships as the data file,
and the VRM export refines it for each body (:mod:`opensculptboy.export.authoring.refine`),
which checks the hand gates on the body itself. With the model's weights the inverse
bind folds 1293 edges over the 66 bodies (1000 at the webs of the fingers, 179 at the groin,
50 at the neck, 30 at the thumb) and collapses 274 triangles; the default body folds 8 edges
and the fixture body 11. The trade-off measured in October 2026:

- one weight set for every body, the T-pose alone: about 190 folds over the 68 bodies of
  the optimisation at best; logits linear in the phenotypes, the T-pose alone: about 60;
- every term, with the caps of the hand poses and the library held hard (``features=28``,
  ``cap_weight=3000``, ``library_weight=3000``, ``area_weight=1e5``): 266 folds (-79 %) and
  1 collapsed triangle over the 66 bodies, none on the default body and 1 on the fixture
  body; every hand, ``bind_error``, ``humanoid_error`` and library gate holds but one (a
  peace sign on corner 010101 folds 5 edges against 4);
- with the caps held softly (``cap_weight=100``, ``fold_weight=3``): 158 to 165 folds, but the
  99th percentile and the largest distance of the hand grow by 10 to 80 % in fists and finger
  curls on the corners. Unfolding a web moves its vertices in the bind, and the same vertices
  then land further from Anny in a fist;
- four rings of vertices around the folds instead of two: slower, and no better;
- the hand gates of the hard-capped set, checked on all 66 bodies and 16 random ones instead
  of the five of :data:`HAND_BODIES`, fail on 36 of the 82 (by one to five extra folds or
  collapsed triangles in a fist or a sign);
- a refinement of that set for each body (:mod:`opensculptboy.export.authoring.refine`),
  which checks the hand gates on the body itself: 232 folds (VRM 1.0) and 266 (VRM 0.x) over
  the 66 bodies, with every other gate held on every body, in 0.4 to 25 s per body (8 s at
  the median). The VRM export runs it by default.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import pathlib
import re
import time

import numpy as np
import torch

from opensculptboy.export import mesh_checks, tpose, vrm_tables

# The six default phenotypes, whose 64 corners the edits are made for.
PHENOTYPES = ("gender", "age", "muscle", "weight", "height", "proportions")
# The body of the shared test exports (test/vrm_fixtures.py), without its face shape.
FIXTURE = {"age": 0.3, "weight": 0.7, "muscle": 0.6}
# The bodies of the error reports of the tests (test_vrm_tpose.CORNERS and the default body).
TEST_BODIES = {
    "default": {},
    "heavy tall": dict(gender=1.0, weight=1.0, muscle=1.0, height=1.0),
    "heavy": dict(gender=0.0, weight=1.0),
}
# The bodies of the hand-pose gates.
HAND_BODIES = ("default", "fixture", "corner 010100", "corner 100101", "corner 010101")
# Library poses with fists and hand signs.
HAND_POSES = ("mh_guard", "mh_boxing", "mh_fight", "mh_peace", "mh_pointing")
# Anny's rest pose with every finger joint curled toward the palm: (fingers, thumb) degrees.
CURLS = {"curl 75": (75.0, 40.0), "curl 90": (90.0, 60.0)}
_FINGER_BONES = tuple(f"finger{k}-{i}" for k in (2, 3, 4, 5) for i in (1, 2, 3))
_CURLED_THUMB_BONES = ("finger1-2", "finger1-3")
# The bones of the hand: a vertex that any of them weighs is a vertex of the hand.
HAND_BONES = (
    _FINGER_BONES
    + ("finger1-1",)
    + _CURLED_THUMB_BONES
    + tuple(f"metacarpal{k}" for k in (1, 2, 3, 4))
)
# The library frames of the mean distance: (entry, frame step); None stands for every pose.
FRAMES = (
    (None, 1),
    ("walk", 1),
    ("idle", 6),
    ("run", 2),
    ("wave", 4),
    ("nod", 6),
    ("shrug", 4),
    ("jump", 4),
)
# The entries whose 99th percentile of the vertex distances is held as well as their largest
# distance (those of ``tpose.bind_error``).
P99_ENTRIES = ("relaxed", "walk")
# The twist modes and the VRM versions that use them by default.
VERSIONS = {"1.0": "constraint", "0.x": "merge"}


@dataclasses.dataclass
class Settings:
    """
    The settings of :func:`optimise`; the defaults are those of the best trade-off measured
    (see the module docstring).
    """

    # The rings of vertices around the folds that the edits may change.
    rings: int = 2
    # The candidate bones of each vertex.
    max_candidates: int = 8
    # The steps with every candidate, then with the strongest 4 bones.
    steps: tuple[int, int] = (600, 400)
    lr: float = 0.05
    # Degrees: the bend that any edge of the T-posed bind may reach.
    floor: float = 90.0
    # On the squared excess (degrees) of the bends of the T-posed bind.
    fold_weight: float = 1.0
    # Degrees: the bend that any edge of a hand pose may reach.
    hand_floor: float = 110.0
    # The share of the reference area below which a triangle costs.
    area_floor: float = 0.4
    area_weight: float = 1e5
    hand_fold_weight: float = 1.0
    # On the squared excess (mm) of the distance of a vertex in a hand pose above that of
    # the model's weights.
    worst_weight: float = 1.0
    # On the squared excess (mm) of a distance in a hand pose above its cap.
    cap_weight: float = 3000.0
    # On the mean squared distance (mm) over the library frames, of the inverse bind and of
    # the forward bind.
    mean_weight: float = 1.0
    forward_weight: float = 0.5
    # On the squared excess (mm) of a distance in a library frame above its cap.
    library_weight: float = 3000.0
    # The pairs (frame, vertex) of the library caps whose distances reach this share of
    # their caps, picked again every ``refresh`` steps.
    active_ratio: float = 0.8
    refresh: int = 50
    # On the squared change of the weights.
    change_weight: float = 10.0
    # On the squared weight beyond the 4 bones of each row, from this share of the first
    # phase on.
    sparse_weight: float = 100.0
    sparse_from: float = 0.5
    seed: int = 0
    # The features of the logits (vrm_tables.edit_features): 1 (one weight set for every
    # body), 7 (linear in the six default phenotypes) or 28 (quadratic).
    features: int = 28


def corner_name(values) -> str:
    """The name of a corner of the six phenotypes, "corner 010100" (in PHENOTYPES order)."""
    return "corner " + "".join(str(int(v)) for v in values)


def bodies() -> dict[str, dict[str, float]]:
    """
    The bodies of the edits, as phenotype values: the default body, the fixture body, the two
    other bodies of the tests, the 64 corners of :data:`PHENOTYPES`, the 12 centres of the
    faces of that cube (one phenotype at 0 or 1, the others at 0.5) and 12 bodies drawn
    inside it.
    """
    result = {"default": {}, "fixture": dict(FIXTURE)}
    result.update({k: dict(v) for k, v in TEST_BODIES.items() if k != "default"})
    for values in itertools.product((0.0, 1.0), repeat=len(PHENOTYPES)):
        result[corner_name(values)] = dict(zip(PHENOTYPES, values))
    for name in PHENOTYPES:
        for value in (0.0, 1.0):
            result[f"face {name} {value:g}"] = {name: value}
    rng = np.random.default_rng(2)
    for i, values in enumerate(rng.uniform(0.1, 0.9, size=(12, len(PHENOTYPES)))):
        result[f"inside {i}"] = dict(zip(PHENOTYPES, map(float, values)))
    return result


def cap_bodies() -> list[str]:
    """
    The bodies whose library frames hold caps in the loss: the bodies of the tests and of the
    hand-pose gates, and every fourth corner.
    """
    corners = [n for n in bodies() if n.startswith("corner")]
    return list(dict.fromkeys([*TEST_BODIES, *HAND_BODIES, *corners[::4]]))


def gate_bodies() -> list[str]:
    """The 66 bodies of the T-pose gate: the default body, the fixture body and the corners."""
    return [
        n for n in bodies() if n in ("default", "fixture") or n.startswith("corner")
    ]


@dataclasses.dataclass
class Body:
    """One body: its rest mesh, its rest bone poses B_j and its VRM T-pose T_j (float64)."""

    name: str
    phenotype: dict
    rest: np.ndarray
    rest_bone_poses: np.ndarray
    t_pose: np.ndarray

    @property
    def kwargs(self) -> dict:
        return {"phenotype_kwargs": dict(self.phenotype)} if self.phenotype else {}

    @property
    def unbind(self) -> np.ndarray:
        """X_j^-1 = B_j T_j^-1 (J, 4, 4): the turn of each bone from the T-pose to the rest."""
        return self.rest_bone_poses @ tpose.rigid_inverse(self.t_pose)


def make_body(model, name: str, phenotype: dict) -> Body:
    """The :class:`Body` of phenotype values (the default body for ``{}``)."""
    kwargs = {"phenotype_kwargs": dict(phenotype)} if phenotype else {}
    rest, B = tpose._rest_body(model, kwargs)
    T = tpose.vrm_t_pose(model, torch.from_numpy(B)).double().cpu().numpy()
    return Body(name, dict(phenotype), rest, B, T)


def inverse_bind(body: Body, weights: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """The inverse bind mesh (V, 3) of 4 weights: (sum_k w_k X_k^-1)^-1 v."""
    N = tpose.blend(weights, indices, body.unbind)
    return np.linalg.solve(N[:, :3, :3], (body.rest - N[:, :3, 3])[..., None])[..., 0]


def library_bone_poses(model, body: Body, name: str, frame_step: int = 1) -> np.ndarray:
    """Anny's world bone poses (F, J, 4, 4) of a library entry (``local-ref``, ungrounded)."""
    import anny.poses

    lib = anny.poses.library()
    rotations, roots = lib.frames(name)
    dtype = model.template_vertices.dtype
    B = torch.from_numpy(body.rest_bone_poses).to(dtype)
    rest = dict(
        rest_bone_poses=B[None],
        rest_bone_heads=B[None, :, :3, 3],
        rest_vertices=torch.from_numpy(body.rest).to(dtype)[None],
    )
    with torch.no_grad():
        params = anny.poses.to_local_ref(
            model,
            rotations[::frame_step],
            roots[::frame_step],
            lib.bones,
            rest_output=rest,
        )
        _, W = model.get_bone_transforms(
            params, B[None], pose_parameterization="local-ref"
        )
    return W.double().cpu().numpy()


def curl_bone_poses(model, body: Body, angle: float, thumb: float) -> np.ndarray:
    """
    Anny's world bone poses (1, J, 4, 4) of the rest pose with each joint of the four fingers
    turned by ``angle`` degrees and each of the two outer thumb joints by ``thumb`` degrees
    toward the palm, about the axis across the bone and the normal of the back of the hand.
    """
    from anny.utils.kinematics import parallel_forward_kinematic_absolute_orientations

    labels = list(model.bone_labels)
    parents = [int(p) for p in model.bone_parents]
    B = body.rest_bone_poses
    heads = B[:, :3, 3]
    turns = np.tile(np.eye(3), (len(labels), 1, 1))
    children = {
        j: [c for c, p in enumerate(parents) if p == j] for j in range(len(labels))
    }
    for j in tpose._parents_first(parents):
        p = parents[j]
        carry = turns[p] if p >= 0 else np.eye(3)
        base = labels[j][:-2]
        if base in _FINGER_BONES + _CURLED_THUMB_BONES:
            _, back = tpose.palm(heads, labels, labels[j][-1])
            if children[j]:
                direction = heads[children[j][0]] - heads[j]
            else:
                direction = B[j, :3, 1]
            axis = tpose._unit(np.cross(direction, back))
            turn = thumb if base in _CURLED_THUMB_BONES else angle
            turns[j] = tpose._axis_angle(carry @ axis, -np.radians(turn)) @ carry
        else:
            turns[j] = carry
    orientations = torch.from_numpy(turns @ B[:, :3, :3])
    poses, _ = parallel_forward_kinematic_absolute_orientations(
        model.kinematic_propagation_fronts,
        rest_bone_poses=torch.from_numpy(B)[None],
        absolute_orientations=orientations[None],
    )
    return poses.double().cpu().numpy()


def hand_bone_poses(model, body: Body) -> dict[str, np.ndarray]:
    """The hand poses of a body: name -> Anny's world bone poses (1, J, 4, 4)."""
    poses = {name: library_bone_poses(model, body, name) for name in HAND_POSES}
    for name, (angle, thumb) in CURLS.items():
        poses[name] = curl_bone_poses(model, body, angle, thumb)
    return poses


def _anny_weights(model):
    return (
        model.vertex_bone_weights.detach().cpu().double().numpy(),
        model.vertex_bone_indices.detach().cpu().numpy(),
    )


def anny_skin(model, body: Body, bone_poses: np.ndarray, rows=None) -> np.ndarray:
    """Anny's posed vertices (F, R, 3) for world bone poses (F, J, 4, 4)."""
    weights, indices = _anny_weights(model)
    rows = slice(None) if rows is None else rows
    return tpose.skin(
        body.rest[rows],
        weights[rows],
        indices[rows],
        bone_poses @ tpose.rigid_inverse(body.rest_bone_poses),
    )


def file_skin(body: Body, bind, weights, indices, bone_poses, rows=None) -> np.ndarray:
    """The posed vertices (F, R, 3) of a file with ``bind`` and 4 weights, for Anny's poses."""
    rows = slice(None) if rows is None else rows
    return tpose.skin(
        bind[rows],
        weights[rows],
        indices[rows],
        bone_poses @ tpose.rigid_inverse(body.t_pose),
    )


def hand_mask(model, dense: np.ndarray) -> np.ndarray:
    """The vertices (V,) that a bone of the hand weighs, in dense weights (V, J)."""
    labels = list(model.bone_labels)
    bones = [labels.index(f"{b}.{s}") for b in HAND_BONES for s in "LR"]
    return dense[:, bones].sum(axis=1) > 0.0


def dense_weights(weights: np.ndarray, indices: np.ndarray, bones: int) -> np.ndarray:
    """Dense (V, J) weights of (V, K) weights and indices."""
    weights = np.asarray(weights, dtype=np.float64)
    dense = np.zeros((len(weights), bones))
    np.add.at(
        dense,
        (np.repeat(np.arange(len(weights)), weights.shape[1]), np.ravel(indices)),
        weights.ravel(),
    )
    return dense


def region_names(model, triangles: np.ndarray, pairs: np.ndarray):
    """
    The part of the body of each edge (pair of triangles) and of each triangle: the family of
    the bone that Anny's weights favour on its vertices ("fingers", "thumb", "spine", ...).
    """
    labels = list(model.bone_labels)
    weights, indices = _anny_weights(model)
    dense = dense_weights(weights, indices, len(labels))

    def family(bone):
        name = re.sub(r"\.(L|R)$", "", labels[bone])
        if name.startswith("finger1"):
            return "thumb"
        if name.startswith("finger") or name.startswith("metacarpal"):
            return "fingers"
        return re.sub(r"\d+$", "", name)

    quads = np.concatenate([triangles[pairs[:, 0]], triangles[pairs[:, 1]]], axis=1)
    edge = np.array([family(b) for b in dense[quads].sum(axis=1).argmax(axis=1)])
    face = np.array([family(b) for b in dense[triangles].sum(axis=1).argmax(axis=1)])
    return edge, face


class Mesh:
    """The triangles of a model, their adjacent pairs and the vertex neighbours."""

    def __init__(self, model):
        from opensculptboy.export.body import triangulated_faces

        self.triangles = np.asarray(triangulated_faces(model)[0], dtype=np.int64)
        self.pairs = mesh_checks.adjacent_triangles(self.triangles)
        self.vertex_count = len(model.template_vertices)
        edges = np.sort(self.triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        edges = np.unique(edges, axis=0)
        self.neighbours = [[] for _ in range(self.vertex_count)]
        for a, b in edges:
            self.neighbours[a].append(b)
            self.neighbours[b].append(a)

    def dilate(self, mask: np.ndarray, rings: int) -> np.ndarray:
        mask = mask.copy()
        for _ in range(rings):
            grown = mask.copy()
            for v in np.flatnonzero(mask):
                grown[self.neighbours[v]] = True
            mask = grown
        return mask


def tpose_problems(mesh: Mesh, body: Body, bind: np.ndarray):
    """The new folds (E,) and the collapsed triangles (T,) of a bind mesh of a body."""
    new = mesh_checks.new_folds(
        mesh_checks.bends(bind, mesh.triangles, mesh.pairs),
        mesh_checks.bends(body.rest, mesh.triangles, mesh.pairs),
    )
    small = mesh_checks.collapsed(
        mesh_checks.areas(bind, mesh.triangles),
        mesh_checks.areas(body.rest, mesh.triangles),
    )
    return new, small


@dataclasses.dataclass
class Region:
    """The vertices that the edits may change, their candidate bones and their surroundings."""

    rows: np.ndarray  # (R,) the vertices
    candidates: np.ndarray  # (R, K) bone indices
    valid: np.ndarray  # (R, K) the slots that hold a candidate
    start: np.ndarray  # (R, K) the model's file weights in the candidate slots
    local: np.ndarray  # (L,) the vertices of the triangles whose bends the rows change
    place: np.ndarray  # (R,) the position of each row in ``local``
    triangles: np.ndarray  # (Tt, 3) those triangles, numbered in ``local``
    triangle_ids: np.ndarray  # (Tt,) their indices in the mesh
    # (E, 2) the edges whose bends the rows change, into ``triangles``
    pairs: np.ndarray
    pair_ids: np.ndarray  # (E,) their indices in ``Mesh.pairs``


def fold_region(
    model,
    mesh: Mesh,
    bodies: dict[str, Body],
    weights: np.ndarray,
    indices: np.ndarray,
    dense: np.ndarray,
    settings: Settings,
) -> Region:
    """
    The vertices of the new folds and the collapsed triangles of the inverse bind over
    ``bodies`` with the file weights (V, 4), and ``settings.rings`` rings around them. The
    candidate bones of a vertex are the bones of its dense row and the 4 bones of each
    neighbour, the strongest ``settings.max_candidates`` by their summed weight (its own
    count ten times).
    """
    seeds = np.zeros(mesh.vertex_count, dtype=bool)
    for body in bodies.values():
        new, small = tpose_problems(mesh, body, inverse_bind(body, weights, indices))
        seeds[mesh.triangles[mesh.pairs[new]].ravel()] = True
        seeds[mesh.triangles[small].ravel()] = True
    rows = np.flatnonzero(mesh.dilate(seeds, settings.rings))
    four = dense_weights(weights, indices, dense.shape[1])
    candidates = []
    for v in rows:
        score = 10.0 * dense[v] + four[mesh.neighbours[v]].sum(axis=0)
        order = [j for j in np.argsort(-score, kind="stable") if score[j] > 0.0]
        own = [j for j in order if four[v, j] > 0.0]
        others = [j for j in order if four[v, j] == 0.0]
        candidates.append((own + others)[: max(settings.max_candidates, len(own))])
    K = max(len(c) for c in candidates)
    cand = np.zeros((len(rows), K), dtype=np.int64)
    valid = np.zeros((len(rows), K), dtype=bool)
    start = np.zeros((len(rows), K))
    for r, (v, c) in enumerate(zip(rows, candidates)):
        cand[r, : len(c)] = c
        valid[r, : len(c)] = True
        start[r, : len(c)] = four[v, c]
    in_region = np.zeros(mesh.vertex_count, dtype=bool)
    in_region[rows] = True
    touched = in_region[mesh.triangles].any(axis=1)
    pair_ids = np.flatnonzero(touched[mesh.pairs].any(axis=1))
    triangle_ids = np.unique(mesh.pairs[pair_ids])
    local = np.unique(mesh.triangles[triangle_ids])
    number = np.full(mesh.vertex_count, -1)
    number[local] = np.arange(len(local))
    triangle_number = np.full(len(mesh.triangles), -1)
    triangle_number[triangle_ids] = np.arange(len(triangle_ids))
    return Region(
        rows=rows,
        candidates=cand,
        valid=valid,
        start=start,
        local=local,
        place=number[rows],
        triangles=number[mesh.triangles[triangle_ids]],
        triangle_ids=triangle_ids,
        pairs=triangle_number[mesh.pairs[pair_ids]],
        pair_ids=pair_ids,
    )


def library_frames(model, body: Body):
    """The library frames of :data:`FRAMES`: (entry name, bone poses (F, J, 4, 4)) pairs."""
    import anny.poses

    for name, step in FRAMES:
        for entry in anny.poses.names("pose") if name is None else [name]:
            yield entry, library_bone_poses(model, body, entry, step)


def forward_bind(model, body: Body) -> np.ndarray:
    """The forward bind mesh (V, 3): Anny's own skin of the rest mesh into the T-pose."""
    weights, indices = _anny_weights(model)
    X = body.t_pose @ tpose.rigid_inverse(body.rest_bone_poses)
    return tpose.skin(body.rest, weights, indices, X)


def body_terms(
    model,
    mesh: Mesh,
    region: Region,
    body: Body,
    weights: np.ndarray,
    indices: np.ndarray,
    hand: np.ndarray,
    caps: bool = True,
) -> dict[str, torch.Tensor]:
    """
    The tensors of the loss of one body, with the model's file weights (V, 4); ``caps``
    adds the caps of the library frames.
    """
    rows, local, K = region.rows, region.local, region.candidates.shape[1]
    bind = inverse_bind(body, weights, indices)
    terms = dict(
        rest=body.rest[rows],
        unbind=body.unbind,
        bind=bind[local],
        rest_bend=mesh_checks.bends(body.rest, mesh.triangles, mesh.pairs)[
            region.pair_ids
        ],
        rest_area=mesh_checks.areas(body.rest, mesh.triangles[region.triangle_ids]),
    )
    # The hand poses.
    poses = hand_bone_poses(model, body)
    W = np.concatenate(list(poses.values()))
    anny = anny_skin(model, body, W)
    file = file_skin(body, bind, weights, indices, W)
    error = np.linalg.norm(file - anny, axis=-1) * 1000.0  # (H, V) mm
    # The cap of a vertex of the hand: the largest distance of the hand in that pose, or the
    # 99th percentile of the distances of the hand (the vertex's own distance, when larger);
    # elsewhere, the largest distance of the whole mesh.
    cap_hand = np.minimum(
        error[:, hand].max(axis=1, keepdims=True),
        np.maximum(
            np.percentile(error[:, hand], 99.0, axis=1, keepdims=True), error[:, rows]
        ),
    )
    cap_all = error.max(axis=1, keepdims=True)
    terms.update(
        hand_transforms=W @ tpose.rigid_inverse(body.t_pose),
        hand_file=file[:, local],
        hand_anny=anny[:, rows],
        hand_bend=mesh_checks.bends(anny[:, local], region.triangles, region.pairs),
        hand_area=mesh_checks.areas(anny[:, local], region.triangles),
        hand_cap=np.where(hand[rows][None], cap_hand, cap_all),
    )
    # The library frames. The mean squared distance of the inverse bind is a quadratic form:
    # with z = vec(w_k [p; 1]) (4K) and G (3, 4K) the blocks [Y_k (3, 4)] of the frame,
    # |G z - a|^2 = z^T G^T G z - 2 a^T G z + |a|^2. That of the forward bind v is one in
    # the weights w (K): F = [Y_k [v; 1]] (3, K), |F w - a|^2.
    cand = torch.from_numpy(region.candidates)
    forward = forward_bind(model, body)
    fh = torch.from_numpy(np.append(forward[rows], np.ones((len(rows), 1)), axis=1))
    H = torch.zeros(len(rows), 4 * K, 4 * K, dtype=torch.float64)
    g = torch.zeros(len(rows), 4 * K, dtype=torch.float64)
    s = torch.zeros(len(rows), dtype=torch.float64)
    Hf = torch.zeros(len(rows), K, K, dtype=torch.float64)
    gf = torch.zeros(len(rows), K, dtype=torch.float64)
    sf = torch.zeros(len(rows), dtype=torch.float64)
    # The caps of the distances from Anny's posed mesh in each library frame: the largest
    # distance of the model's weights over the whole mesh in its entry and, in the entries of
    # P99_ENTRIES, the 99th percentile of the distances (or the row's own distance, when
    # larger). The forward bind has caps in the entries of P99_ENTRIES alone (infinite
    # elsewhere). :func:`active_pairs` picks the pairs (frame, row) near their caps.
    library = dict(Y=[], anny=[], cap=[], fwd_cap=[])
    count = 0
    for name, W in library_frames(model, body):
        Y = torch.from_numpy((W @ tpose.rigid_inverse(body.t_pose))[:, :, :3, :])
        anny = anny_skin(model, body, W)
        a = torch.from_numpy(anny[:, rows])
        for kind, mesh_bind in (("cap", bind), ("fwd_cap", forward)):
            cap = np.full((len(W), len(rows)), np.inf)
            if caps and (kind == "cap" or name in P99_ENTRIES):
                file = file_skin(body, mesh_bind, weights, indices, W)
                error = np.linalg.norm(file - anny, axis=-1) * 1000.0  # (F, V)
                cap[:] = error.max()
                if name in P99_ENTRIES:
                    cap = np.minimum(
                        cap, np.maximum(np.percentile(error, 99.0), error[:, rows])
                    )
            library[kind].append(cap)
        library["Y"].append(Y.numpy())
        library["anny"].append(anny[:, rows])
        for start in range(0, len(W), 16):
            G = Y[start : start + 16][:, cand]  # (f, R, K, 3, 4)
            Gf = torch.einsum("frkab,rb->frka", G, fh).permute(1, 0, 3, 2)
            Gf = Gf.reshape(len(rows), -1, K)  # (R, 3f, K)
            G = G.permute(0, 1, 3, 2, 4).reshape(G.shape[0], len(rows), 3, 4 * K)
            G = G.permute(1, 0, 2, 3).reshape(len(rows), -1, 4 * K)  # (R, 3f, 4K)
            A = a[start : start + 16].permute(1, 0, 2).reshape(len(rows), -1)
            H += G.transpose(1, 2) @ G
            g += (G.transpose(1, 2) @ A[..., None])[..., 0]
            s += (A**2).sum(dim=1)
            Hf += Gf.transpose(1, 2) @ Gf
            gf += (Gf.transpose(1, 2) @ A[..., None])[..., 0]
            sf += (A**2).sum(dim=1)
        count += len(W)
    if caps:
        for kind, arrays in library.items():
            terms[f"library_{kind}"] = np.concatenate(arrays)
    terms["forward"] = forward[rows]
    result = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in terms.items()}
    result.update(quad_H=H / count, quad_g=g / count, quad_s=s / count)
    result.update(fwd_H=Hf / count, fwd_g=gf / count, fwd_s=sf / count)
    empty = torch.zeros(0, dtype=torch.int64)
    result.update(active_frames=empty, active_rows=empty)
    result.update(active_fwd_frames=empty, active_fwd_rows=empty)
    return result


# The per-body terms that every body holds, and those of the library caps.
_BODY_TERMS = (
    "rest",
    "unbind",
    "bind",
    "rest_bend",
    "rest_area",
    "hand_file",
    "hand_anny",
    "hand_bend",
    "hand_area",
    "hand_cap",
    "quad_H",
    "quad_g",
    "quad_s",
    "fwd_H",
    "fwd_g",
    "fwd_s",
    "forward",
)
_LIBRARY_TERMS = ("library_Y", "library_anny", "library_cap", "library_fwd_cap")


def stack_terms(region: Region, terms: dict[str, dict]) -> dict[str, torch.Tensor]:
    """
    The terms of every body stacked along a first axis (B, ...), with the transforms of the
    hand poses gathered for the candidates of each row (B, H, R, K, 3, 4). The library caps
    stack over the bodies that hold them (C, ...); ``capped`` (C,) gives their places.
    """
    names = list(terms)
    cand = torch.from_numpy(region.candidates)
    batch = {k: torch.stack([terms[n][k] for n in names]) for k in _BODY_TERMS}
    batch["hand_Y"] = torch.stack(
        [terms[n]["hand_transforms"][:, cand, :3, :] for n in names]
    )
    # The bends, the areas and the distances of the model's weights in the hand poses.
    place = torch.from_numpy(region.place)
    bend, area = _angles(
        batch["hand_file"],
        torch.from_numpy(region.triangles),
        torch.from_numpy(region.pairs),
    )
    batch["hand_file_bend"] = bend
    batch["hand_file_ratio"] = area / batch["hand_area"]
    batch["hand_error"] = _distance(batch["hand_file"][:, :, place], batch["hand_anny"])
    capped = [i for i, n in enumerate(names) if "library_Y" in terms[n]]
    batch["capped"] = torch.tensor(capped, dtype=torch.int64)
    if capped:
        for k in _LIBRARY_TERMS:
            batch[k] = torch.stack([terms[names[i]][k] for i in capped])
    empty = torch.zeros(0, dtype=torch.int64)
    for kind in ("inverse", "forward"):
        batch[f"active_{kind}"] = (empty, empty, empty)
    return batch


def _angles(vertices, triangles, pairs):
    """The bends (..., E) in degrees, and the triangle areas (..., T), differentiable."""
    a, b, c = (vertices[..., triangles[:, k], :] for k in range(3))
    n = torch.cross(b - a, c - a, dim=-1)
    length = torch.sqrt((n**2).sum(dim=-1) + 1e-24)
    n = n / length[..., None]
    first, second = n[..., pairs[:, 0], :], n[..., pairs[:, 1], :]
    across = torch.cross(first, second, dim=-1)
    sin = torch.sqrt((across**2).sum(dim=-1) + 1e-18)
    cos = (first * second).sum(dim=-1)
    return torch.rad2deg(torch.atan2(sin, cos)), length / 2.0


def _distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The distances (mm) between the rows of two position arrays (metres), differentiable."""
    return torch.sqrt(((a - b) ** 2).sum(dim=-1) + 1e-18) * 1000.0


def _homogeneous(p: torch.Tensor) -> torch.Tensor:
    return torch.cat([p, torch.ones_like(p[..., :1])], dim=-1)


class Loss:
    """The loss of :func:`optimise` over a batch of :func:`stack_terms`."""

    def __init__(self, region: Region, settings: Settings):
        self.s = settings
        self.candidates = torch.from_numpy(region.candidates)
        self.start = torch.from_numpy(region.start)
        self.place = torch.from_numpy(region.place)
        self.triangles = torch.from_numpy(region.triangles)
        self.pairs = torch.from_numpy(region.pairs)

    def bind(self, w: torch.Tensor, t: dict) -> torch.Tensor:
        """The inverse bind (B, R, 3) of the rows for region weights ``w`` (B, R, K)."""
        N = torch.einsum("nrk,nrkij->nrij", w, t["unbind"][:, self.candidates])
        rhs = (t["rest"] - N[..., :3, 3])[..., None]
        return torch.linalg.solve(N[..., :3, :3], rhs)[..., 0]

    def __call__(self, w: torch.Tensor, t: dict) -> dict[str, torch.Tensor]:
        s = self.s
        p = self.bind(w, t)
        bind = t["bind"].index_copy(1, self.place, p)
        bend, area = _angles(bind, self.triangles, self.pairs)
        limit = torch.clamp(t["rest_bend"], min=s.floor)
        fold = torch.relu(bend - limit) ** 2
        if "fold_w" in t:
            # A factor per edge (B, E), as the refinement of one body may give.
            fold = t["fold_w"] * fold
        terms = dict(
            fold=s.fold_weight * fold.sum(),
            shrink=s.area_weight
            * (torch.relu(s.area_floor - area / t["rest_area"]) ** 2).sum(),
        )
        # The hand poses.
        ph = _homogeneous(p)
        M = torch.einsum("brk,bhrkij->bhrij", w, t["hand_Y"])
        q = torch.einsum("bhrij,brj->bhri", M, ph)
        posed = t["hand_file"].index_copy(2, self.place, q)
        bend, area = _angles(posed, self.triangles, self.pairs)
        # An edge may bend to max(hand_floor, Anny's bend), or as far as the model's weights
        # bend it, and a triangle may shrink to area_floor of Anny's area, or as far as the
        # model's weights shrink it.
        limit = torch.maximum(
            torch.clamp(t["hand_bend"], min=s.hand_floor), t["hand_file_bend"]
        )
        fold = torch.relu(bend - limit) ** 2  # (B, H, E)
        if "hand_pose_w" in t:
            # A factor per hand pose (H,).
            fold = t["hand_pose_w"][None, :, None] * fold
        terms["hand_fold"] = s.hand_fold_weight * fold.sum()
        floor = torch.clamp(t["hand_file_ratio"], max=s.area_floor)
        terms["hand_shrink"] = (
            s.area_weight * (torch.relu(floor - area / t["hand_area"]) ** 2).sum()
        )
        error = _distance(q, t["hand_anny"])  # (B, H, R)
        terms["worst"] = (
            s.worst_weight * (torch.relu(error - t["hand_error"]) ** 2).sum()
        )
        terms["cap"] = s.cap_weight * (torch.relu(error - t["hand_cap"]) ** 2).sum()
        # The mean squared distances over the library frames (mm^2).
        z = (w[..., None] * ph[:, :, None, :]).flatten(2)
        mean = (
            torch.einsum("bri,brij,brj->br", z, t["quad_H"], z)
            - 2.0 * (t["quad_g"] * z).sum(dim=-1)
            + t["quad_s"]
        )
        terms["mean"] = s.mean_weight * 1e6 * mean.sum()
        mean = (
            torch.einsum("brk,brkl,brl->br", w, t["fwd_H"], w)
            - 2.0 * (t["fwd_g"] * w).sum(dim=-1)
            + t["fwd_s"]
        )
        terms["forward"] = s.forward_weight * 1e6 * mean.sum()
        # The caps of the library frames, on the pairs of :func:`active_pairs`.
        terms["library"] = torch.zeros((), dtype=w.dtype)
        homogeneous = {"inverse": ph, "forward": _homogeneous(t["forward"])}
        caps = {"inverse": "library_cap", "forward": "library_fwd_cap"}
        for kind, h in homogeneous.items():
            c, f, r = t[f"active_{kind}"]
            if not len(c):
                continue
            Y = t["library_Y"][c[:, None], f[:, None], self.candidates[r]]
            b = t["capped"][c]
            q = torch.einsum("ak,akij,aj->ai", w[b, r], Y, h[b, r])
            distance = _distance(q, t["library_anny"][c, f, r])
            excess = torch.relu(distance - t[caps[kind]][c, f, r])
            terms["library"] = terms["library"] + s.library_weight * (excess**2).sum()
        change = ((w - self.start) ** 2).sum(dim=(1, 2)).mean()
        terms["change"] = s.change_weight * change
        return terms


def library_distances(loss: Loss, w: torch.Tensor, t: dict) -> torch.Tensor:
    """
    The distances (mm) of the inverse and the forward bind from Anny's posed mesh in every
    library frame of the capped bodies (C, 2, P, R), for region weights ``w`` (B, R, K),
    without gradients.
    """
    with torch.no_grad():
        capped = t["capped"]
        w = w[capped]
        p = loss.bind(w, {k: t[k][capped] for k in ("unbind", "rest")})
        h = torch.stack([_homogeneous(p), _homogeneous(t["forward"][capped])], dim=2)
        result = []
        for c in range(len(capped)):
            out = []
            for start in range(0, t["library_Y"].shape[1], 32):
                Y = t["library_Y"][c, start : start + 32][:, loss.candidates]
                q = torch.einsum("rk,frkab,rsb->sfra", w[c], Y, h[c])
                out.append(_distance(q, t["library_anny"][c, start : start + 32][None]))
            result.append(torch.cat(out, dim=1))
    return torch.stack(result)


def active_pairs(loss: Loss, w: torch.Tensor, t: dict, ratio: float) -> int:
    """
    Set the pairs (capped body, frame, row) of the batch ``t`` whose distances reach
    ``ratio`` of their caps; returns their count.
    """
    if not len(t["capped"]):
        return 0
    distances = library_distances(loss, w, t)
    count = 0
    for i, (kind, cap) in enumerate(
        (("inverse", "library_cap"), ("forward", "library_fwd_cap"))
    ):
        t[f"active_{kind}"] = torch.nonzero(
            distances[:, i] >= ratio * t[cap], as_tuple=True
        )
        count += len(t[f"active_{kind}"][0])
    return count


def _top_four(w: np.ndarray) -> np.ndarray:
    """A mask (R, K) of the strongest 4 slots of each row."""
    order = np.argsort(-w, axis=1, kind="stable")[:, :4]
    mask = np.zeros(w.shape, dtype=bool)
    np.put_along_axis(mask, order, True, axis=1)
    return mask


def phenotype_features(phenotype: dict, count: int) -> np.ndarray:
    """The first ``count`` features of a body (``vrm_tables.edit_features``)."""
    return vrm_tables.edit_features(phenotype, count)


@dataclasses.dataclass
class RegionWeights:
    """
    The weights of the region rows: for a body with features f (:func:`phenotype_features`),
    the softmax over the slots of ``mask`` of the logits sum_d L[..., d] f_d.
    """

    logits: np.ndarray  # (R, K, D)
    mask: np.ndarray  # (R, K)

    def weights(self, phenotype: dict) -> np.ndarray:
        """The weights (R, K) of a body."""
        f = phenotype_features(phenotype, self.logits.shape[-1])
        return _softmax(self.logits @ f, self.mask)


def _softmax(logits, mask):
    logits = np.where(mask, logits, -np.inf)
    logits = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def optimise(
    region: Region,
    terms: dict[str, dict[str, torch.Tensor]],
    phenotypes: dict[str, dict],
    settings: Settings,
    log=print,
    check=None,
    checkpoint=None,
) -> RegionWeights:
    """
    The optimised weights of the region, with 4 bones or fewer per row, over every body of
    ``terms`` (body name -> :func:`body_terms`; ``phenotypes`` gives each body's phenotype).
    ``check(weights)`` (optional) returns the new folds and collapsed triangles of the T-pose
    of each body, ``{name: (folds, collapsed)}``, for weights ``{name: (R, K)}``;
    ``checkpoint(result)`` (optional) receives the :class:`RegionWeights` at each check.
    """
    loss = Loss(region, settings)
    names = list(terms)
    batch = stack_terms(region, terms)
    features = torch.from_numpy(
        np.stack([phenotype_features(phenotypes[n], settings.features) for n in names])
    )
    mask = torch.from_numpy(region.valid.copy())
    logits = torch.zeros(
        *region.candidates.shape, features.shape[1], dtype=torch.float64
    )
    logits[..., 0] = torch.log(torch.clamp(torch.from_numpy(region.start), min=1e-4))
    logits.requires_grad_(True)
    negative = torch.tensor(-1e9, dtype=torch.float64)

    def weights():
        L = torch.einsum("rkd,bd->brk", logits, features)
        return torch.softmax(torch.where(mask, L, negative), dim=-1)

    started = time.perf_counter()
    for phase, steps in enumerate(settings.steps):
        if phase == 1:
            with torch.no_grad():
                mean = weights().mean(dim=0).numpy()
            mask = mask & torch.from_numpy(_top_four(mean))
        optimiser = torch.optim.Adam([logits], lr=settings.lr)
        # A cosine decay of the rate to a tenth over each phase.
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimiser,
            lambda step: 0.1 + 0.45 * (1.0 + np.cos(np.pi * step / max(steps, 1))),
        )
        for step in range(steps):
            last = step == steps - 1
            if settings.library_weight > 0 and step % settings.refresh == 0:
                with torch.no_grad():
                    count = active_pairs(loss, weights(), batch, settings.active_ratio)
                if step % 100 == 0:
                    log(f"  library pairs near their caps: {count}")
            optimiser.zero_grad()
            w = weights()
            parts = loss(w, batch)
            if phase == 0 and step >= settings.sparse_from * steps:
                # The weight beyond the 4 bones of each row that the bodies favour on
                # average, which the second phase drops.
                with torch.no_grad():
                    kept = torch.zeros_like(mask)
                    kept.scatter_(1, w.mean(dim=0).topk(4, dim=1).indices, True)
                extra = torch.where(kept, 0.0, w).sum(dim=-1)
                parts["sparse"] = settings.sparse_weight * (extra**2).sum()
            sum(parts.values()).backward()
            optimiser.step()
            schedule.step()
            if step % 50 == 0 or last:
                text = ", ".join(
                    f"{k} {float(v.detach()):.4g}" for k, v in parts.items()
                )
                log(
                    f"phase {phase + 1}, step {step}: {text} "
                    f"({time.perf_counter() - started:.0f} s)"
                )
            if check is not None and (step % 100 == 99 or last):
                with torch.no_grad():
                    w = weights().numpy()
                counts = check({n: _keep_four(w[i]) for i, n in enumerate(names)})
                folding = {n: c for n, c in counts.items() if c != (0, 0)}
                log(
                    f"  T-pose: {sum(c[0] for c in counts.values())} new folds, "
                    f"{sum(c[1] for c in counts.values())} collapsed on {len(folding)} "
                    f"bodies; default {counts.get('default')}, fixture "
                    f"{counts.get('fixture')}"
                )
                if checkpoint is not None:
                    checkpoint(
                        RegionWeights(logits.detach().numpy().copy(), mask.numpy())
                    )
    return RegionWeights(logits.detach().numpy(), mask.numpy())


def weight_rows(region: Region, w: np.ndarray, tolerance: float = 1e-6):
    """
    The rows (vertices, bones (E, 4), weights (E, 4)) of region weights (R, K) with 4 bones
    or fewer per row that differ from the model's by more than ``tolerance``, strongest bone
    first.
    """
    changed = np.abs(w - region.start).max(axis=1) > tolerance
    order = np.argsort(-w, axis=1, kind="stable")[:, :4]
    weights = np.take_along_axis(w, order, axis=1)
    bones = np.where(weights > 0, np.take_along_axis(region.candidates, order, 1), 0)
    return region.rows[changed], bones[changed], weights[changed]


def region_edits(
    region: Region, result: RegionWeights, phenotypes, tolerance: float = 1e-6
):
    """
    The edits (vertices (E,), bones (E, 4), logits (E, 4, D)) of optimised region weights:
    the rows whose weights differ from the model's by more than ``tolerance`` on any body of
    ``phenotypes``, with the bones of their slots (-1 for an empty slot), strongest first on
    average (see ``vrm_tables.edit_weights``).
    """
    weights = np.stack([result.weights(ph) for ph in phenotypes])  # (B, R, K)
    changed = np.abs(weights - region.start).max(axis=(0, 2)) > tolerance
    mean = np.where(result.mask, weights.mean(axis=0), -1.0)
    order = np.argsort(-mean, axis=1, kind="stable")[:, :4]
    held = np.take_along_axis(result.mask, order, axis=1)
    bones = np.where(held, np.take_along_axis(region.candidates, order, axis=1), -1)
    logits = np.take_along_axis(result.logits, order[..., None], axis=1)
    logits = np.where(held[..., None], logits, 0.0)
    return region.rows[changed], bones[changed], logits[changed]


def merge_edits(model, edits):
    """
    The edits of twist "merge" from those of "constraint": the slots of the twist bones that
    "merge" moves go to their mapped bones (a bone may then fill two slots, whose weights add
    up).
    """
    labels = list(model.bone_labels)
    rows, bones, logits = (np.array(a, copy=True) for a in edits)
    for bone, parent in vrm_tables.TWIST_MERGE.items():
        bones[bones == labels.index(bone)] = labels.index(parent)
    return rows, bones, logits


def save_edits(path, model, edits: dict, summary: dict | None = None) -> None:
    """Write edits ``{twist: (vertices, bones, logits)}`` with the model's digest."""
    from safetensors.numpy import save_file

    tensors = {}
    for twist, (rows, bones, logits) in edits.items():
        tensors[f"{twist}.vertices"] = np.asarray(rows, dtype=np.int32)
        tensors[f"{twist}.bones"] = np.asarray(bones, dtype=np.int32)
        tensors[f"{twist}.logits"] = np.asarray(logits, dtype=np.float32)
    metadata = dict(
        format=vrm_tables.SKIN_WEIGHT_FORMAT,
        digest=vrm_tables.skin_weight_digest(model),
        features=json.dumps(["1", *(f"{n} - 0.5" for n in vrm_tables.EDIT_PHENOTYPES)]),
        summary=json.dumps(summary or {}, sort_keys=True),
    )
    save_file(tensors, str(path), metadata=metadata)
    vrm_tables._read_skin_weight_edits.cache_clear()


def load_edits(path=None) -> dict:
    """The edits of the data file, ``{twist: (vertices, bones, logits)}``."""
    path = pathlib.Path(path or vrm_tables.SKIN_WEIGHT_EDITS)
    _, tensors = vrm_tables._read_skin_weight_edits(str(path))
    return {
        twist: tuple(tensors[f"{twist}.{k}"] for k in ("vertices", "bones", "logits"))
        for twist in vrm_tables.TWIST_MODES
    }


def authoring_model():
    """The model of the export (the anny rig and mesh) in float64."""
    from opensculptboy.character import Character

    return Character().build_model(face_shapes=False, dtype=torch.float64)


@dataclasses.dataclass
class Problem:
    """What :func:`optimise` needs: the bodies, the model's file weights, the region, terms."""

    mesh: Mesh
    bodies: dict[str, Body]
    weights: np.ndarray  # (V, 4) the model's file weights (twist "constraint")
    indices: np.ndarray
    region: Region
    terms: dict[str, dict[str, torch.Tensor]]

    def tpose_counts(self, w: dict[str, np.ndarray]) -> dict[str, tuple[int, int]]:
        """
        The new folds and collapsed triangles of each body with its region weights (R, K)
        of ``w`` (body name -> weights with 4 bones or fewer per row).
        """
        counts = {}
        for name, body in self.bodies.items():
            rows, bones, values = weight_rows(self.region, w[name])
            w4, i4 = self.weights.copy(), self.indices.copy()
            w4[rows], i4[rows] = values, bones
            new, small = tpose_problems(self.mesh, body, inverse_bind(body, w4, i4))
            counts[name] = (int(new.sum()), int(small.sum()))
        return counts


def problem(model, settings: Settings, log=print) -> Problem:
    """The bodies of :func:`bodies`, the fold region and the terms of every body."""
    mesh = Mesh(model)
    started = time.perf_counter()
    all_bodies = {n: make_body(model, n, ph) for n, ph in bodies().items()}
    raw = vrm_tables.vrm_skin_weights(model, "constraint", edits=False)
    weights, indices = tpose._strongest_four(*raw)
    dense = dense_weights(*raw, len(model.bone_labels))
    region = fold_region(model, mesh, all_bodies, weights, indices, dense, settings)
    log(
        f"region of {len(all_bodies)} bodies: {len(region.rows)} vertices, "
        f"{region.candidates.shape[1]} candidates, {len(region.pair_ids)} edges"
    )
    hand = hand_mask(model, dense)
    capped = cap_bodies()
    terms = {
        name: body_terms(
            model, mesh, region, body, weights, indices, hand, name in capped
        )
        for name, body in all_bodies.items()
    }
    log(f"terms of {len(terms)} bodies in {time.perf_counter() - started:.0f} s")
    return Problem(mesh, all_bodies, weights, indices, region, terms)


def author(
    model=None,
    settings: Settings | None = None,
    log=print,
    prepared=None,
    checkpoint=None,
) -> dict:
    """
    Optimise the edits of both twist modes; returns ``{twist: (vertices, bones, logits)}``.
    ``prepared`` is a :class:`Problem` of the same settings, to skip its preparation, and
    ``checkpoint(edits)`` (optional) receives the edits at each check of :func:`optimise`.
    """
    settings = settings or Settings()
    model = model if model is not None else authoring_model()
    torch.manual_seed(settings.seed)
    prepared = prepared or problem(model, settings, log)
    started = time.perf_counter()
    phenotypes = {n: b.phenotype for n, b in prepared.bodies.items()}

    def edits_of(result):
        constraint = region_edits(prepared.region, result, phenotypes.values())
        return {"constraint": constraint, "merge": merge_edits(model, constraint)}

    result = optimise(
        prepared.region,
        prepared.terms,
        phenotypes,
        settings,
        log=log,
        check=prepared.tpose_counts,
        checkpoint=None if checkpoint is None else lambda r: checkpoint(edits_of(r)),
    )
    log(f"optimised in {time.perf_counter() - started:.0f} s")
    return edits_of(result)


def _keep_four(w: np.ndarray) -> np.ndarray:
    """Region weights (R, K) cut to the strongest 4 slots of each row, renormalised."""
    w = np.where(_top_four(w), w, 0.0)
    return w / w.sum(axis=1, keepdims=True)


# The gate report.


def _ratio(after: float, before: float) -> float:
    return after / before if before > 0 else (1.0 if after == 0 else np.inf)


def tpose_report(model, mesh: Mesh, all_bodies: dict[str, Body], file_weights):
    """
    New folds and collapsed triangles of the inverse bind, per body and per part, with the
    file weights ``file_weights(body)`` (4 weights and bone indices) of each body.
    """
    edge_part, face_part = region_names(model, mesh.triangles, mesh.pairs)
    per_body, parts = {}, {}
    for name, body in all_bodies.items():
        bind = inverse_bind(body, *file_weights(body))
        new, small = tpose_problems(mesh, body, bind)
        per_body[name] = (int(new.sum()), int(small.sum()))
        for part in edge_part[new]:
            parts[part] = parts.get(part, 0) + 1
        for part in face_part[small]:
            parts["collapsed " + part] = parts.get("collapsed " + part, 0) + 1
    return per_body, parts


def hand_report(model, mesh: Mesh, body: Body, weights, indices, hand: np.ndarray):
    """
    Per hand pose: the distances (mm) of the hand from Anny's posed mesh (max, 99th
    percentile, mean), the new folds and collapsed triangles of the whole mesh and of the
    hand (the edges and triangles that touch it).
    """
    bind = inverse_bind(body, weights, indices)
    hand_pairs = hand[mesh.triangles[mesh.pairs]].any(axis=(1, 2))
    hand_triangles = hand[mesh.triangles].any(axis=1)
    report = {}
    for name, W in hand_bone_poses(model, body).items():
        anny = anny_skin(model, body, W)[0]
        file = file_skin(body, bind, weights, indices, W)[0]
        distance = np.linalg.norm(file - anny, axis=-1)[hand] * 1000.0
        new = mesh_checks.new_folds(
            mesh_checks.bends(file, mesh.triangles, mesh.pairs),
            mesh_checks.bends(anny, mesh.triangles, mesh.pairs),
        )
        small = mesh_checks.collapsed(
            mesh_checks.areas(file, mesh.triangles),
            mesh_checks.areas(anny, mesh.triangles),
        )
        report[name] = dict(
            max_mm=float(distance.max()),
            p99_mm=float(np.percentile(distance, 99.0)),
            mean_mm=float(distance.mean()),
            folds=int(new.sum()),
            collapsed=int(small.sum()),
            hand_folds=int((new & hand_pairs).sum()),
            hand_collapsed=int((small & hand_triangles).sum()),
        )
    return report


def library_worst(model, body: Body, weight_sets: dict) -> dict:
    """
    The largest vertex distance (mm) of the inverse bind from Anny's posed mesh over every
    frame of every library pose and clip, for each named set of 4 weights.
    """
    import anny.poses

    binds = {k: inverse_bind(body, *wi) for k, wi in weight_sets.items()}
    report = {k: {} for k in weight_sets}
    for name in anny.poses.names():
        W = library_bone_poses(model, body, name)
        worst = dict.fromkeys(weight_sets, 0.0)
        for start in range(0, len(W), 24):
            chunk = W[start : start + 24]
            anny = anny_skin(model, body, chunk)
            for k, (w, i) in weight_sets.items():
                file = file_skin(body, binds[k], w, i, chunk)
                d = float(np.linalg.norm(file - anny, axis=-1).max()) * 1000.0
                worst[k] = max(worst[k], d)
        for k in weight_sets:
            report[k][name] = worst[k]
    return report


def validation_bodies(count: int = 16, seed: int = 1) -> dict[str, dict[str, float]]:
    """Bodies inside the phenotype cube, off the bodies of the edits: random values."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.0, 1.0, size=(count, len(PHENOTYPES)))
    return {
        f"random {i}": dict(zip(PHENOTYPES, map(float, v)))
        for i, v in enumerate(values)
    }


def gate_report(model, edits, log=print) -> tuple[dict, bool]:
    """
    The gates of the edits ``{twist: (vertices, bones, logits)}`` against the model's
    weights, for both twist modes; returns the report and whether every gate holds.
    ``edits`` may instead be a function ``(version, twist, body) -> (weights, indices)``
    that gives the file's 4 weights of each :class:`Body`, such as a refinement per body.
    """
    mesh = Mesh(model)
    labels = list(model.bone_labels)
    all_bodies = {n: make_body(model, n, ph) for n, ph in bodies().items()}
    validation = {n: make_body(model, n, ph) for n, ph in validation_bodies().items()}
    gate = gate_bodies()
    report, passed = {}, True

    def verdict(name, ok, text):
        nonlocal passed
        passed &= bool(ok)
        log(f"[{'pass' if ok else 'FAIL'}] {name}: {text}")

    for version, twist in VERSIONS.items():
        log(f"== VRM {version} ({twist})")

        def weights_of(kind, version=version, twist=twist):
            if kind == "after" and callable(edits):
                return lambda body: edits(version, twist, body)
            chosen = False if kind == "before" else edits
            return lambda body: tpose.file_skin_weights(
                model, version, twist, edits=chosen, phenotype=body.phenotype
            )

        def sets_of(body):
            return {k: weights_of(k)(body) for k in ("before", "after")}

        dense = dense_weights(
            *vrm_tables.vrm_skin_weights(model, twist, edits=False), len(labels)
        )
        hand = hand_mask(model, dense)
        r = report[version] = {}
        # The T-pose.
        tp = {
            k: tpose_report(
                model, mesh, {n: all_bodies[n] for n in gate}, weights_of(k)
            )
            for k in ("before", "after")
        }
        r["tpose"] = tp
        total = {k: sum(f for f, _ in v[0].values()) for k, v in tp.items()}
        collapsed = {k: sum(c for _, c in v[0].values()) for k, v in tp.items()}
        for name in ("default", "fixture"):
            b, a = tp["before"][0][name], tp["after"][0][name]
            verdict(
                f"T-pose, {name} body",
                a[0] == 0,
                f"new folds {b[0]} -> {a[0]}, collapsed {b[1]} -> {a[1]}",
            )
        verdict(
            f"T-pose, {len(gate)} bodies",
            total["after"] <= 0.05 * total["before"],
            f"new folds {total['before']} -> {total['after']} (at most "
            f"{0.05 * total['before']:.0f}), collapsed {collapsed['before']} -> "
            f"{collapsed['after']}; by part before {tp['before'][1]}, after {tp['after'][1]}",
        )
        tv = {k: tpose_report(model, mesh, validation, weights_of(k)) for k in tp}
        r["validation"] = tv
        log(
            f"[info] T-pose, {len(validation)} random bodies: new folds "
            + " -> ".join(str(sum(f for f, _ in tv[k][0].values())) for k in tv)
            + ", collapsed "
            + " -> ".join(str(sum(c for _, c in tv[k][0].values())) for k in tv)
        )
        # The hand poses.
        r["hands"] = {}
        for name in HAND_BODIES:
            body = all_bodies[name]
            hr = {
                k: hand_report(model, mesh, body, *wi, hand)
                for k, wi in sets_of(body).items()
            }
            r["hands"][name] = hr
            for pose in hr["before"]:
                b, a = hr["before"][pose], hr["after"][pose]
                ok = (
                    a["folds"] <= b["folds"]
                    and a["collapsed"] <= b["collapsed"]
                    and a["max_mm"] <= 1.1 * b["max_mm"]
                    and a["p99_mm"] <= 1.1 * b["p99_mm"]
                )
                verdict(
                    f"{name}, {pose}",
                    ok,
                    f"max {b['max_mm']:.2f} -> {a['max_mm']:.2f} mm, p99 "
                    f"{b['p99_mm']:.2f} -> {a['p99_mm']:.2f}, mean {b['mean_mm']:.2f} -> "
                    f"{a['mean_mm']:.2f}; folds {b['folds']} -> {a['folds']} (hand "
                    f"{b['hand_folds']} -> {a['hand_folds']}), collapsed {b['collapsed']} "
                    f"-> {a['collapsed']}",
                )
        # The errors of the test bodies.
        r["bind_error"], r["humanoid_error"], r["library"] = {}, {}, {}
        for name, phenotype in TEST_BODIES.items():
            ph = phenotype or None
            sets = sets_of(all_bodies[name])
            for method in ("inverse", "forward"):
                be = {
                    k: tpose.bind_error(
                        model,
                        method,
                        phenotype_kwargs=ph,
                        file_weights=w,
                        file_indices=i,
                    )
                    for k, (w, i) in sets.items()
                }
                r["bind_error"][(name, method)] = be
                for pose in be["before"]:
                    b, a = be["before"][pose], be["after"][pose]
                    keys = ("max_mm", "p99_mm", "mean_mm", "normal_p99_deg")
                    ok = all(a[k] <= 1.05 * b[k] for k in keys)
                    verdict(
                        f"bind_error {name}, {method}, {pose}",
                        ok,
                        ", ".join(f"{k} {b[k]:.2f} -> {a[k]:.2f}" for k in keys),
                    )
            he = {
                k: tpose.humanoid_error(
                    model,
                    version,
                    twist,
                    phenotype_kwargs=ph,
                    file_weights=w,
                    file_indices=i,
                )
                for k, (w, i) in sets.items()
            }
            r["humanoid_error"][name] = he
            for pose in he["before"]:
                for variant in he["before"][pose]:
                    b, a = he["before"][pose][variant], he["after"][pose][variant]
                    keys = ("max_mm", "p99_mm", "mean_mm")
                    ok = all(a[k] <= 1.05 * b[k] for k in keys)
                    verdict(
                        f"humanoid_error {name}, {pose}, {variant}",
                        ok,
                        ", ".join(f"{k} {b[k]:.2f} -> {a[k]:.2f}" for k in keys),
                    )
            lw = library_worst(model, all_bodies[name], sets)
            r["library"][name] = lw
            worse = {
                pose: (lw["before"][pose], lw["after"][pose])
                for pose in lw["before"]
                if lw["after"][pose] > 1.1 * lw["before"][pose]
            }
            ratio = max(_ratio(lw["after"][p], lw["before"][p]) for p in lw["before"])
            verdict(
                f"library worst, {name}",
                not worse,
                f"{len(lw['before'])} entries, largest ratio {ratio:.3f}"
                + (f"; over 1.1: {worse}" if worse else ""),
            )
    # The forward bind keeps Anny's own weights.
    body = all_bodies["fixture"]
    zero = np.zeros((0, *body.rest.shape))
    forward = [
        tpose.rebind(
            model,
            body.rest,
            zero,
            body.rest_bone_poses,
            *(
                edits("1.0", "constraint", body)
                if callable(e)
                else tpose.file_skin_weights(
                    model, "1.0", "constraint", edits=e, phenotype=body.phenotype
                )
            ),
            method="forward",
        ).vertices
        for e in (False, edits)
    ]
    verdict(
        "forward bind keeps Anny's weights",
        np.array_equal(forward[0], forward[1]),
        "the forward bind mesh with and without the edits is "
        + ("the same" if np.array_equal(forward[0], forward[1]) else "different"),
    )
    return report, passed


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--report", action="store_true", help="report on the data file alone"
    )
    parser.add_argument(
        "--out", default=str(vrm_tables.SKIN_WEIGHT_EDITS), help="the data file"
    )
    parser.add_argument(
        "--settings",
        default="{}",
        help='other Settings as JSON, e.g. \'{"features": 7, "cap_weight": 100}\'',
    )
    parser.add_argument(
        "--force", action="store_true", help="write the edits even if a gate fails"
    )
    args = parser.parse_args(argv)
    model = authoring_model()
    if args.report:
        edits = load_edits(args.out)
    else:
        overrides = json.loads(args.settings)
        if "steps" in overrides:
            overrides["steps"] = tuple(overrides["steps"])
        edits = author(model, Settings(**overrides))
    report, passed = gate_report(model, edits)
    print(f"gates: {'all pass' if passed else 'some fail'}")
    if not args.report and (passed or args.force):
        save_edits(
            pathlib.Path(args.out),
            model,
            edits,
            summary={k: int(len(v[0])) for k, v in edits.items()},
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

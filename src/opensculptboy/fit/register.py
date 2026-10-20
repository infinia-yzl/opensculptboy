# OpenSculptBoy
# Apache License, Version 2.0
"""
Registration of anny to a reference mesh of any topology, in three stages.

1. **Placement.** With landmarks, a rigid alignment of anny's landmarks onto the reference's
   (Umeyama 1991, :func:`anny.faces.authoring.nicp.umeyama`); without them, the centres of
   the bounding landmarks of :func:`~opensculptboy.fit.reference.auto_landmarks`. The height
   phenotype starts at the value whose stature matches the reference.
2. **Parameters.** Adam over the phenotypes, the local changes, the face shapes, the pose of
   every bone but the root and the eyes, and a rigid placement, in rounds: each round finds the
   closest points again and runs a few steps on

   - point-to-surface distances both ways (anny's vertices to the reference surface, point to
     plane; a sample of the reference's vertices to anny's nearest vertices), with pairs too far
     apart or facing away from each other left out;
   - the landmarks (:data:`~opensculptboy.fit.reference.LANDMARKS`, weighted);
   - small ridge penalties on the local changes, the face shapes and the bone turns.

   The first rounds free the placement, the phenotypes and the large joints alone.
3. **Non-rigid refinement.** Anny's whole mesh moves onto the reference by a non-rigid ICP in
   the spirit of optimal-step NICP (Amberg et al., CVPR 2007), as in
   :func:`anny.faces.authoring.nicp.refine`: closest points pull the vertices, and a Laplacian
   term keeps the displacement smooth with a stiffness that decreases over the rounds. Region
   masks leave out the eyeballs, the eye sockets and the inside of the mouth, which follow the
   skin around them. Each level of the detail layer then repeats the refinement on its
   subdivided mesh, starting from the level below.

The residuals (:mod:`opensculptboy.fit.residual`) of each level become the card's detail layer:
:func:`fit_reference` returns a :class:`FitResult`, and :meth:`FitResult.write` writes the card
and its offsets files.

The fit needs SciPy (the ``viewer`` or ``examples`` extra).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time

import numpy as np
import torch

from opensculptboy.character import Character
from opensculptboy.fit import residual as R_
from opensculptboy.fit.reference import (
    LANDMARKS,
    Reference,
    auto_landmark_vertices,
    vertex_normals,
)

# Bones whose turns the parameter stage leaves at rest.
FIXED_BONES = ("root", "eye.L", "eye.R")
# Words of the bones that the first rounds free: the trunk and the large joints.
MAJOR_BONES = (
    "spine",
    "neck",
    "head",
    "clavicle",
    "shoulder",
    "upperarm",
    "lowerarm",
    "wrist",
    "upperleg",
    "lowerleg",
    "foot",
)
# Base-mesh face labels that the registration leaves out (with the eyeballs).
EXCLUDED_LABELS = ("eye_cavity.L", "eye_cavity.R", "mouth_cavity")


@dataclasses.dataclass
class FitOptions:
    """Settings of :func:`fit_reference`; the defaults suit a full figure in metres."""

    rounds: int = 10
    steps: int = 30
    major_rounds: int = (
        3  # rounds that free only the placement, phenotypes and large joints
    )
    reverse_samples: int = 20000
    landmark_weight: float = 1.0
    auto_landmark_weight: float = 0.2
    # ridge penalties (per squared unit, summed over the parameters)
    local_change_ridge: float = 1e-7
    face_shape_ridge: float = 1e-7
    pose_ridge: float = 1e-6
    # correspondences farther apart than this (metres) are left out, first and last round
    distance_limits: tuple = (0.05, 0.01)
    normal_threshold: float = 0.2
    # NICP stiffness per stage, level 0 and the subdivided levels
    stiffness: tuple = (10.0, 3.0, 1.0, 0.3)
    detail_stiffness: tuple = (1.0, 0.3, 0.1, 0.03)
    nicp_iterations: int = 3
    nicp_max_distance: float = 0.02
    seed: int = 0
    verbose: bool = False


@dataclasses.dataclass
class FitResult:
    """What a fit gives: the card, the residual offsets of each level and the errors."""

    character: Character
    rotation: np.ndarray  # (3, 3) anny -> reference: x_ref = R x + t
    translation: np.ndarray  # (3,)
    parametric_rest: np.ndarray  # (V0, 3) rest mesh of the card's parameters
    offsets: dict  # level -> (N_level, 3) rest-frame offsets
    errors: dict  # "parametric" and each level -> surface_error() dict
    seconds: dict  # time of each stage
    reference: Reference | None = None

    def write(self, card_path: str | pathlib.Path, textures: dict | None = None):
        """
        Write the card (JSON) and, next to it, ``<card>.detail<N>.safetensors`` for each level.
        ``textures`` maps texture slots to files already written next to the card.
        """
        from opensculptboy.detail import save_offsets

        card_path = pathlib.Path(card_path)
        stem = (
            card_path.name[: -len(card_path.suffix)]
            if card_path.suffix
            else card_path.name
        )
        levels = []
        for subdiv in sorted(self.offsets):
            name = f"{stem}.detail{subdiv}.safetensors"
            save_offsets(
                card_path.parent / name,
                self.offsets[subdiv],
                subdiv,
                self.character.topology,
                {"source": "opensculptboy fit"},
            )
            entry = {"subdiv": int(subdiv), "offsets": name}
            if subdiv in self.errors:
                entry["error_mm"] = round(self.errors[subdiv]["forward"]["mean"], 4)
            levels.append(entry)
        detail = {
            "levels": levels,
            "fit": {
                "placement": {
                    "rotation": np.round(self.rotation, 9).tolist(),
                    "translation": np.round(self.translation, 9).tolist(),
                    "frame": "x_reference = rotation @ x_anny + translation (Anny's axes)",
                },
                "errors_mm": {
                    str(k): {d: round(v[d]["mean"], 4) for d in ("forward", "reverse")}
                    for k, v in self.errors.items()
                },
                "seconds": {k: round(v, 1) for k, v in self.seconds.items()},
            },
        }
        if textures:
            detail["textures"] = dict(textures)
        self.character.detail = detail
        card_path.write_text(json.dumps(self.character.to_dict(), indent=2) + "\n")
        return card_path


def _log(options: FitOptions, message: str) -> None:
    if options.verbose:
        print(message, flush=True)


def fit_model():
    """The Anny model of the fit (float64, every local change and face shape, LBS)."""
    import anny

    return anny.Anny(
        local_changes="all",
        face_shapes="all",
        facial_actions="none",
        skinning_method="lbs",
    ).to(dtype=torch.float64)


def excluded_vertices(model) -> np.ndarray:
    """Mask (V,) of the vertices that the registration leaves out: the eyeballs, the eye
    sockets and the inside of the mouth."""
    from anny.faces.authoring.base_mesh import base_mesh
    from opensculptboy.export.vrm_tables import eyeball_vertices

    mesh = base_mesh()
    base = model.base_mesh_vertex_indices.cpu().numpy()
    out_base = np.zeros(len(mesh.V), bool)
    for label in EXCLUDED_LABELS:
        out_base[np.unique(mesh.quads[mesh.face_labels == label])] = True
    excluded = out_base[base]
    try:
        for rows in eyeball_vertices(model).values():
            excluded[rows] = True
    except ValueError:
        pass
    return excluded


def nicp(
    vertices: np.ndarray,
    triangles: np.ndarray,
    reference: Reference,
    weights: np.ndarray,
    stiffness=(10.0, 3.0, 1.0, 0.3),
    iterations: int = 3,
    max_distance: float = 0.02,
    normal_threshold: float = 0.2,
    anchor: float = 1e-4,
) -> np.ndarray:
    """
    Non-rigid ICP of a mesh onto the reference: in each round, solve

        min_X sum_i w_i |x_i - c_i|^2 + alpha |L (X - X_0)|^2 + anchor |X - X_0|^2

    with c_i the closest point of the reference, L the uniform Laplacian of the mesh and X_0
    the mesh at the start of the stiffness ``alpha``. Pairs farther than ``max_distance`` or
    whose normals disagree (dot product under ``normal_threshold``) drop out; ``weights`` of 0
    leave a vertex to the smoothness term. The small ``anchor`` keeps closed parts without
    data (the eyeballs) where they are, which :func:`follow_skin` then corrects.
    """
    import scipy.sparse
    import scipy.sparse.linalg

    from anny.faces.authoring.nicp import uniform_laplacian

    X = np.asarray(vertices, dtype=np.float64).copy()
    n = len(X)
    L = uniform_laplacian(n, np.asarray(triangles))
    LtL = (L.T @ L).tocsc()
    base = np.asarray(weights, dtype=np.float64)
    for alpha in stiffness:
        start = X.copy()
        smooth = alpha * (LtL @ start)
        for _ in range(iterations):
            C, N, d, _, _ = reference.closest(X)
            normals = vertex_normals(X, triangles)
            ok = (d < max_distance) & ((normals * N).sum(1) > normal_threshold)
            w = base * ok
            A = (scipy.sparse.diags(w + anchor) + alpha * LtL).tocsc()
            rhs = w[:, None] * C + anchor * start + smooth
            solve = scipy.sparse.linalg.factorized(A)
            X = np.stack([solve(rhs[:, k]) for k in range(3)], 1)
    return X


def follow_skin(
    before: np.ndarray,
    after: np.ndarray,
    excluded: np.ndarray,
    triangles: np.ndarray,
    radius: float = 0.02,
) -> np.ndarray:
    """
    Move each closed component of excluded vertices (the eyeballs) by the mean displacement
    of the fitted vertices within ``radius`` of its centre, so that it follows the skin.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(before)
    t = np.asarray(triangles)
    edges = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    out = after.copy()
    kept = np.nonzero(~excluded)[0]
    tree = cKDTree(before[kept])
    displacement = after - before
    for label in np.unique(labels[excluded]):
        component = labels == label
        if (component & ~excluded).any():
            continue  # joined to fitted skin: the Laplacian already carries it
        centre = before[component].mean(0)
        near = tree.query_ball_point(centre, radius)
        if near:
            out[component] = before[component] + displacement[kept[near]].mean(0)
    return out


class _Parameters(torch.nn.Module):
    """The free parameters of the parameter stage."""

    def __init__(self, model, rotation0, translation0, height0: float):
        super().__init__()
        dtype = model.template_vertices.dtype
        labels = list(model.bone_labels)
        self.free = [i for i, b in enumerate(labels) if b not in FIXED_BONES]
        self.major = [
            k
            for k, i in enumerate(self.free)
            if any(word in labels[i] for word in MAJOR_BONES)
        ]
        p = len(model.phenotype_labels)
        phenotype = torch.full((1, p), 0.5, dtype=dtype)
        phenotype[0, list(model.phenotype_labels).index("height")] = height0
        self.phenotype = torch.nn.Parameter(phenotype)
        self.local = torch.nn.Parameter(
            torch.zeros(1, len(model.local_change_labels), dtype=dtype)
        )
        self.face = torch.nn.Parameter(
            torch.zeros(1, len(model.face_shape_labels), dtype=dtype)
        )
        self.rotvec = torch.nn.Parameter(torch.zeros(len(self.free), 3, dtype=dtype))
        self.global_rotvec = torch.nn.Parameter(torch.zeros(3, dtype=dtype))
        self.global_translation = torch.nn.Parameter(torch.zeros(3, dtype=dtype))
        self.register_buffer("rotation0", torch.as_tensor(rotation0, dtype=dtype))
        self.register_buffer("translation0", torch.as_tensor(translation0, dtype=dtype))
        ranges = [model.face_shape_ranges[n] for n in model.face_shape_labels]
        self.register_buffer(
            "face_lo", torch.tensor([r[0] for r in ranges], dtype=dtype)
        )
        self.register_buffer(
            "face_hi", torch.tensor([r[1] for r in ranges], dtype=dtype)
        )
        self.bone_count = len(labels)

    def pose(self) -> torch.Tensor:
        import roma

        dtype = self.rotvec.dtype
        P = torch.eye(4, dtype=dtype).repeat(1, self.bone_count, 1, 1)
        P[0, self.free, :3, :3] = roma.rotvec_to_rotmat(self.rotvec)
        return P

    def placement(self):
        import roma

        R = roma.rotvec_to_rotmat(self.global_rotvec) @ self.rotation0
        return R, self.translation0 + self.global_translation

    @torch.no_grad()
    def project(self):
        self.phenotype.clamp_(0.0, 1.0)
        self.local.clamp_(-1.0, 1.0)
        self.face.copy_(
            torch.maximum(torch.minimum(self.face, self.face_hi), self.face_lo)
        )


def _initial_placement(model, reference: Reference, regressor, options):
    """Rotation, translation and height phenotype that place anny near the reference."""
    from anny.faces.authoring.nicp import umeyama

    labels = list(model.phenotype_labels)
    best, height0 = None, 0.5
    with torch.no_grad():
        for h in np.linspace(0.0, 1.0, 11):
            phenotype = torch.full((1, len(labels)), 0.5, dtype=model.dtype)
            phenotype[0, labels.index("height")] = float(h)
            rest = model(phenotype_kwargs=phenotype)["rest_vertices"][0].numpy()
            gap = abs(np.ptp(rest[:, 2]) - reference.height())
            if best is None or gap < best:
                best, height0 = gap, float(h)
        phenotype = torch.full((1, len(labels)), 0.5, dtype=model.dtype)
        phenotype[0, labels.index("height")] = height0
        out = model(phenotype_kwargs=phenotype)
    anny_points = _landmark_points(model, out, regressor, reference, None)
    names = [k for k in reference.landmarks if k in anny_points]
    src = np.stack([anny_points[k] for k in names])
    dst = np.stack([reference.landmarks[k] for k in names])
    if reference.auto or len(names) < 3:
        rotation = np.eye(3)
        translation = dst.mean(0) - src.mean(0)
    else:
        rotation, translation, _ = umeyama(src, dst)
    return rotation, translation, height0


def _landmark_points(model, out, regressor, reference, auto_vertices) -> dict:
    """anny's landmarks (numpy, Anny's frame, unplaced) for the names of the reference"""
    vertices = out["vertices"][0].detach().numpy()
    joints = out["bone_poses"][0, :, :3, 3].detach().numpy()
    labels = list(model.bone_labels)
    points = {}
    if reference.auto:
        indices = auto_vertices or auto_landmark_vertices(vertices)
        for name, rows in indices.items():
            points[name] = vertices[rows].mean(0)
        return points
    cranio = regressor.weights_matrix.detach().numpy() @ vertices
    for name in reference.landmarks:
        if name not in LANDMARKS:
            continue
        kind, label, _ = LANDMARKS[name]
        if kind == "bone":
            points[name] = joints[labels.index(label)]
        else:
            points[name] = cranio[regressor.labels.index(label)]
    return points


class _Craniofacial:
    """The craniofacial landmarks as a dense (K, V) matrix on the model's vertices."""

    def __init__(self, model):
        from anny.keypoints import KeypointsRegressor

        regressor = KeypointsRegressor.craniofacial(model)
        self.labels = list(regressor.labels)
        weights = getattr(regressor, "regression_weights", None)
        if weights is None or weights.dim() != 2:
            weights = torch.eye(len(model.template_vertices), dtype=model.dtype)
            weights = regressor({"vertices": weights[None]})[0]
        self.weights_matrix = weights.to(model.dtype)


def fit_parameters(model, reference: Reference, options: FitOptions):
    """Stages 1 and 2: the parameters and the placement. Returns the module and the output."""
    regressor = _Craniofacial(model)
    rotation0, translation0, height0 = _initial_placement(
        model, reference, regressor, options
    )
    params = _Parameters(model, rotation0, translation0, height0)
    excluded = excluded_vertices(model)
    fit_rows = np.nonzero(~excluded)[0]
    triangles = model.faces.cpu().numpy()
    rng = np.random.default_rng(options.seed)
    sample = rng.choice(
        len(reference.vertices),
        min(options.reverse_samples, len(reference.vertices)),
        replace=False,
    )
    sample_points = torch.as_tensor(reference.vertices[sample])
    sample_normals = torch.as_tensor(reference.normals[sample])

    # landmark targets and anny's side of each
    labels = list(model.bone_labels)
    names, kinds, rows, weights, targets = [], [], [], [], []
    auto_vertices = None
    if reference.auto:
        with torch.no_grad():
            out = model(phenotype_kwargs=params.phenotype)
        auto_vertices = auto_landmark_vertices(out["vertices"][0].numpy())
    for name, point in reference.landmarks.items():
        if reference.auto:
            if name not in auto_vertices:
                continue
            kinds.append("vertices")
            rows.append(torch.as_tensor(auto_vertices[name]))
            weights.append(options.auto_landmark_weight)
        elif name in LANDMARKS:
            kind, label, weight = LANDMARKS[name]
            kinds.append(kind)
            rows.append(
                labels.index(label) if kind == "bone" else regressor.labels.index(label)
            )
            weights.append(options.landmark_weight * weight)
        else:
            continue
        names.append(name)
        targets.append(point)
    lm_targets = torch.as_tensor(np.stack(targets)) if targets else None
    lm_weights = torch.as_tensor(weights, dtype=model.dtype)
    cranio = regressor.weights_matrix

    def forward():
        out = model(
            pose_parameters=params.pose(),
            phenotype_kwargs=params.phenotype,
            local_changes_kwargs=params.local,
            face_shape_kwargs=params.face,
        )
        R, t = params.placement()
        placed = out["vertices"][0] @ R.T + t
        return out, placed, R, t

    def landmark_loss(out, placed, R, t):
        if lm_targets is None:
            return placed.new_zeros(())
        joints = out["bone_poses"][0, :, :3, 3] @ R.T + t
        points = []
        for kind, row in zip(kinds, rows):
            if kind == "bone":
                points.append(joints[row])
            elif kind == "craniofacial":
                points.append(cranio[row] @ placed)
            else:
                points.append(placed[row].mean(0))
        points = torch.stack(points)
        sq = ((points - lm_targets) ** 2).sum(-1)
        return (sq * lm_weights).sum() / lm_weights.sum()

    groups = [
        dict(params=[params.phenotype], lr=0.01),
        dict(params=[params.local], lr=0.01),
        dict(params=[params.face], lr=0.01),
        dict(params=[params.rotvec], lr=0.005),
        dict(params=[params.global_rotvec], lr=0.002),
        dict(params=[params.global_translation], lr=0.002),
    ]
    optimizer = torch.optim.Adam(groups)
    lo, hi = options.distance_limits
    fit_rows_t = torch.as_tensor(fit_rows)
    from scipy.spatial import cKDTree

    for round_ in range(options.rounds):
        major_only = round_ < options.major_rounds
        limit = hi + (lo - hi) * max(0.0, 1.0 - round_ / max(options.rounds - 1, 1))
        with torch.no_grad():
            out, placed, R, t = forward()
        X = placed.numpy()
        C, N, d, _, _ = reference.closest(X[fit_rows])
        normals = vertex_normals(X, triangles)[fit_rows]
        ok = (d < limit) & ((normals * N).sum(1) > options.normal_threshold)
        fwd_rows = fit_rows_t[torch.as_tensor(ok)]
        fwd_points = torch.as_tensor(C[ok])
        fwd_normals = torch.as_tensor(N[ok])
        nearest_d, nearest = cKDTree(X[fit_rows]).query(reference.vertices[sample])
        rev_ok = nearest_d < limit
        rev_rows = fit_rows_t[torch.as_tensor(nearest[rev_ok])]
        rev_points = sample_points[torch.as_tensor(rev_ok)]
        rev_normals = sample_normals[torch.as_tensor(rev_ok)]
        lm_scale = 1.0 if round_ < options.major_rounds else 0.3
        for step in range(options.steps):
            optimizer.zero_grad()
            out, placed, R, t = forward()
            diff = placed[fwd_rows] - fwd_points
            plane = (diff * fwd_normals).sum(-1)
            data = (plane**2).mean() + 0.1 * (diff**2).sum(-1).mean()
            rdiff = placed[rev_rows] - rev_points
            data = data + 0.5 * ((rdiff * rev_normals).sum(-1) ** 2).mean()
            loss = data + lm_scale * landmark_loss(out, placed, R, t)
            loss = loss + options.local_change_ridge * (params.local**2).sum()
            loss = loss + options.face_shape_ridge * (params.face**2).sum()
            loss = loss + options.pose_ridge * (params.rotvec**2).sum()
            loss.backward()
            if major_only:
                params.local.grad = None
                params.face.grad = None
                mask = torch.zeros(len(params.free), 1, dtype=params.rotvec.dtype)
                mask[params.major] = 1.0
                params.rotvec.grad = params.rotvec.grad * mask
            optimizer.step()
            params.project()
        _log(
            options,
            f"round {round_}: {1000 * np.sqrt(np.mean(d[ok] ** 2)):.2f} mm rms over "
            f"{ok.mean():.0%} of the vertices (limit {1000 * limit:.0f} mm)",
        )
    with torch.no_grad():
        out, placed, R, t = forward()
    return params, out, R.numpy(), t.numpy()


def _character_from(model, params, name: str) -> Character:
    phenotype = {
        k: round(float(v), 6)
        for k, v in zip(model.phenotype_labels, params.phenotype[0].tolist())
    }
    local = {
        k: round(float(v), 6)
        for k, v in zip(model.local_change_labels, params.local[0].tolist())
        if abs(v) > 1e-6
    }
    face = {
        k: round(float(v), 6)
        for k, v in zip(model.face_shape_labels, params.face[0].tolist())
        if abs(v) > 1e-6
    }
    character = Character(
        name=name, phenotype=phenotype, local_changes=local, face_shapes=face
    )
    pose = Character.pose_from_parameters(model, params.pose().detach())
    if pose["bones"]:
        character.pose = pose
    return character


def _card_rest(model, character: Character) -> tuple[np.ndarray, np.ndarray]:
    """The parametric rest mesh (V0, 3) and the skinning transforms (J, 4, 4) of a card,
    from its rounded values, as the exporters evaluate them."""
    from opensculptboy.detail.layer import bone_transforms

    with torch.no_grad():
        out = model(
            pose_parameters=character.pose_parameters(model),
            phenotype_kwargs=dict(character.phenotype) or None,
            local_changes_kwargs=dict(character.local_changes) or None,
            face_shape_kwargs=dict(character.face_shapes) or None,
        )
    return (
        out["rest_vertices"][0].double().numpy(),
        bone_transforms(out)[0].double().numpy(),
    )


def fit_reference(
    reference: Reference,
    detail: int = 1,
    name: str = "fitted",
    options: FitOptions | None = None,
    model=None,
) -> FitResult:
    """
    Fit anny to a reference and measure the result (see the module documentation).

    Args:
        reference: the target surface in Anny's frame.
        detail: the finest level of the detail layer, 0, 1 or 2; the result holds the
            offsets of every level up to it.
        name: the character's name.
        options: the settings; :class:`FitOptions` by default.
        model: the fit model; :func:`fit_model` by default.
    """
    from opensculptboy.detail import detail_topology

    if detail not in (0, 1, 2):
        raise ValueError(f"detail must be 0, 1 or 2; got {detail!r}.")
    options = options or FitOptions()
    model = model or fit_model()
    seconds = {}
    clock = time.perf_counter()
    params, _, rotation, translation = fit_parameters(model, reference, options)
    character = _character_from(model, params, name)
    seconds["parameters"] = time.perf_counter() - clock

    rest0, transforms = _card_rest(model, character)
    excluded = excluded_vertices(model)
    errors, offsets = {}, {}
    weights0 = model.vertex_bone_weights.detach().double().numpy()
    indices0 = model.vertex_bone_indices.cpu().numpy()
    previous = np.zeros_like(rest0)
    previous_level = 0
    for subdiv in range(detail + 1):
        clock = time.perf_counter()
        topology = detail_topology(model, subdiv, "full", character.topology)
        w, i = topology.skin_weights(weights0, indices0)
        parametric = np.asarray(topology.apply(rest0))
        if subdiv == 0:
            start_rest = parametric
        else:
            up = detail_topology(model, previous_level, "full", character.topology)
            start_rest = parametric + _lift(previous, up, topology)
        posed = R_.pose_vertices(start_rest, w, i, transforms, rotation, translation)
        mask = np.asarray(topology.apply(excluded.astype(np.float64))) < 0.5
        if subdiv == 0:
            errors["parametric"] = R_.surface_error(
                posed, topology.triangles, reference, mask, seed=options.seed
            )
        fitted = nicp(
            posed,
            topology.triangles,
            reference,
            mask.astype(np.float64),
            options.stiffness if subdiv == 0 else options.detail_stiffness,
            options.nicp_iterations,
            options.nicp_max_distance,
            options.normal_threshold,
        )
        fitted = follow_skin(posed, fitted, ~mask, topology.triangles)
        registered = R_.unpose(fitted, w, i, transforms, rotation, translation)
        offsets[subdiv] = R_.rest_offsets(registered, parametric)
        errors[subdiv] = R_.surface_error(
            fitted, topology.triangles, reference, mask, seed=options.seed
        )
        previous, previous_level = offsets[subdiv], subdiv
        seconds[f"level{subdiv}"] = time.perf_counter() - clock
        _log(
            options,
            f"level {subdiv}: {errors[subdiv]['forward']['mean']:.3f} mm mean "
            f"({seconds[f'level{subdiv}']:.0f} s)",
        )
    return FitResult(
        character=character,
        rotation=rotation,
        translation=translation,
        parametric_rest=rest0,
        offsets=offsets,
        errors=errors,
        seconds=seconds,
        reference=reference,
    )


def _lift(offsets: np.ndarray, coarse, fine) -> np.ndarray:
    """Offsets of a coarser full level carried to a finer one by the subdivision."""
    steps = fine.operators[len(coarse.operators) :]
    x = offsets
    for operator in steps:
        x = operator.apply(x)
    return np.asarray(x)

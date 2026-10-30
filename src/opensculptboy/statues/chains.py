# OpenSculptBoy
# Apache License, Version 2.0
"""
The joints and the shape of a statue's hands and feet, fitted to the scan one limb at a time.

The parameter stage of the reference fit poses the whole figure at once: the trunk and the
large joints dominate its data, and the fingers and the toes stay near the start pose. A
statue's hand grips a hilt or rests on a hip with its fingers bent, and a foot stands on a
head or a base with its toes turned. With straight fingers the parameter stage shrinks the
hand and its fingers to hide them in the marble, and the detail levels copy the fist's
surface onto the palm and leave anny's fingers standing out of it.

:func:`refine_chains` fits each named chain (:data:`CHAINS`: the forearm's twist, the wrist,
the metacarpals and the fingers of a hand; the shin's twist, the foot and the toes of a foot)
to the scan with the rest of the figure held. The chain's own local changes
(:data:`SHAPES`: the hand's scale and its fingers' length, thickness and spread; the foot's
scale and place) start again from anny's own. A few starts of the pose (the stage's own pose
of the chain, and the fingers or the whole chain of some of anny's library poses) are fitted
with nearest-point pairs both ways in rounds of shrinking reach, the start that comes closest
wins, and its pose and the chain's local changes are then fitted together. The chain's
vertices pair with the scan's closest points whose normals agree (forward), and the scan's
points whose nearest vertex of the figure lies on the chain and faces the same way pair with
that vertex (reverse), so that a finger moves into the scan's finger rather than into the
palm or the hip it rests on.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# the bones of a chain by their words (with the side's suffix), and the bones whose skin
# anchors the chain's data (held, but measured)
CHAINS = {
    "hand": ("lowerarm02", "wrist", "metacarpal", "finger"),
    "foot": ("lowerleg02", "foot", "toe"),
}
ANCHORS = {"hand": ("lowerarm01",), "foot": ("lowerleg01",)}
# the bones whose turns the library starts take with the stage's other turns ("fingers"
# starts); a "whole" start takes the library's turns of every bone of the chain
FINGERS = {"hand": ("metacarpal", "finger"), "foot": ("toe",)}
# the chain's root, whose turn the search samples
ROOTS = {"hand": "wrist", "foot": "foot"}
# the scan's points of a chain lie within this distance (metres) of its root at the stage's pose
ROOT_REACH = {"hand": 0.14, "foot": 0.24}
# the library poses whose chains are tried as starts
STARTS = {
    "hand": (
        "relaxed",
        "hands_on_hips",
        "mh_hand_hip",
        "mh_boxing",
        "mh_fight",
        "mh_guard",
        "mh_thinking",
    ),
    "foot": ("relaxed", "mh_step", "mh_sit_ankle", "weight_shift", "mh_kick"),
}
# the chain's local changes ({s}: l or r), fitted with its pose
SHAPES = {
    "hand": (
        "{s}-hand-scale-incr",
        "{s}-hand-fingers-length-incr",
        "{s}-hand-fingers-diameter-incr",
        "{s}-hand-fingers-distance-incr",
        "{s}-hand-trans-out",
    ),
    "foot": (
        "{s}-foot-scale-incr",
        "{s}-foot-scale-depth-incr",
        "{s}-foot-scale-horiz-incr",
        "{s}-foot-scale-vert-incr",
        "{s}-foot-trans-forward",
        "{s}-foot-trans-out",
        "{s}-foot-trans-up",
    ),
}


@dataclasses.dataclass
class ChainOptions:
    """Settings of :func:`refine_chains`."""

    # the reach of the pairs (metres) in the first and the last round
    reach: tuple = (0.03, 0.005)
    rounds: int = 8
    steps: int = 25
    # the rounds of each start before the best start runs on
    trial_rounds: int = 3
    rate: float = 0.02
    # the forward pairs' normals agree to this (cosine)
    normal_threshold: float = 0.3
    # the pull of each bone's turn back to its start (per radian squared) and of each local
    # change back to anny's own
    prior: float = 1e-4
    shape_prior: float = 1e-5
    # the chain's local changes start again from anny's own (0) and are fitted with the pose
    shape: bool = True
    # the share of the distances that the score keeps (a trimmed mean)
    trim: float = 0.9
    # the scan's points that may pair with a chain lie within this distance (metres) of the
    # chain at the stage's pose
    region: float = 0.1
    # the weight of the joint hints (refine_chains' hints) against the surface's pairs, and
    # in the score (per metre)
    hint_weight: float = 10.0
    hint_score: float = 1.0
    # and lie further than this (metres) from the rest of the figure at the stage's pose
    explained: float = 0.012
    # the turns (degrees) about each axis that the search tries at the chain's root, and the
    # starts of the search that are fitted
    search: tuple = (-90.0, -45.0, 0.0, 45.0, 90.0)
    keep: int = 4
    # the weight of the chain's depth inside the rest of the figure in the score (0: none)
    depth: float = 0.0
    # the starts tried (names of the report's trials: "stage", "<pose>/fingers",
    # "<pose>/whole"); None: all of them
    starts: tuple | None = None
    verbose: bool = False


def chain_bones(model, chain: str) -> tuple[list, list]:
    """The bones (indices) of a chain such as ``"hand.l"`` and of its anchors."""
    part, side = _split(chain)
    suffix = "." + side.upper()
    labels = list(model.bone_labels)

    def pick(words):
        return [
            i
            for i, b in enumerate(labels)
            if b.endswith(suffix) and any(b.startswith(w) for w in words)
        ]

    return pick(CHAINS[part]), pick(ANCHORS[part])


def _split(chain: str) -> tuple[str, str]:
    part, _, side = chain.lower().partition(".")
    if part not in CHAINS or side not in ("l", "r"):
        raise ValueError(
            f"Unknown chain {chain!r}; use one of {sorted(CHAINS)} with .l or .r."
        )
    return part, side


def chain_shapes(model, chain: str) -> list:
    """The local changes of a chain that the model has."""
    part, side = _split(chain)
    labels = set(model.local_change_labels)
    return [n.format(s=side) for n in SHAPES[part] if n.format(s=side) in labels]


def _strongest(model) -> np.ndarray:
    weights = model.vertex_bone_weights.detach().cpu().numpy()
    indices = model.vertex_bone_indices.cpu().numpy()
    return indices[np.arange(len(indices)), weights.argmax(1)]


def chain_vertices(model, chain: str) -> tuple[np.ndarray, np.ndarray]:
    """Masks (V,) of the level-0 vertices whose strongest bone is in the chain, and of those
    whose strongest bone is one of its anchors."""
    bones, anchors = chain_bones(model, chain)
    strongest = _strongest(model)
    return np.isin(strongest, bones), np.isin(strongest, anchors)


def _identity_pose(model):
    import torch

    return torch.eye(4, dtype=model.template_vertices.dtype).repeat(
        1, len(model.bone_labels), 1, 1
    )


class _Skinner:
    """The posed and placed vertices (level 0) of a card for pose parameters, with its
    sliders held: the rest model is computed once."""

    def __init__(self, model, character, rotation, translation):
        import torch

        self.model = model
        with torch.no_grad():
            pose = character.pose_parameters(model)
            if pose is None:
                pose = _identity_pose(model)
            out = model(
                pose_parameters=pose,
                phenotype_kwargs=dict(character.phenotype) or None,
                local_changes_kwargs=dict(character.local_changes) or None,
                face_shape_kwargs=dict(character.face_shapes) or None,
            )
        self.pose = pose.detach().clone()
        self.rest_vertices = out["rest_vertices"].detach()
        self.rest_bone_poses = out["rest_bone_poses"].detach()
        dtype = self.rest_vertices.dtype
        self.R = torch.as_tensor(np.asarray(rotation), dtype=dtype)
        self.t = torch.as_tensor(np.asarray(translation), dtype=dtype)

    def __call__(self, pose):
        transforms, poses = self.model.get_bone_transforms(pose, self.rest_bone_poses)
        V = self.model.get_skinned_vertices(self.rest_vertices, transforms)[0]
        # the bones' heads, placed
        self.joints = poses[0, :, :3, 3] @ self.R.T + self.t
        return V @ self.R.T + self.t


class _ShapedSkinner:
    """The posed and placed vertices with the chain's local changes free: the whole model
    runs for each evaluation."""

    def __init__(self, model, character, rotation, translation):
        import torch

        self.model = model
        dtype = model.template_vertices.dtype
        self.labels = list(model.local_change_labels)
        values = dict(character.local_changes)
        self.local = torch.tensor(
            [[float(values.get(n, 0.0)) for n in self.labels]], dtype=dtype
        )
        self.phenotype = dict(character.phenotype) or None
        self.face = dict(character.face_shapes) or None
        self.R = torch.as_tensor(np.asarray(rotation), dtype=dtype)
        self.t = torch.as_tensor(np.asarray(translation), dtype=dtype)

    def __call__(self, pose, columns, values):
        local = self.local.clone()
        local[0, columns] = values
        out = self.model(
            pose_parameters=pose,
            phenotype_kwargs=self.phenotype,
            local_changes_kwargs=local,
            face_shape_kwargs=self.face,
        )
        self.joints = out["bone_poses"][0, :, :3, 3] @ self.R.T + self.t
        return out["vertices"][0] @ self.R.T + self.t


def _library_rotations(model, name: str) -> np.ndarray:
    import anny.poses

    posed = anny.poses.pose_parameters(model, name)
    return posed["pose_parameters"][0, :, :3, :3].detach().double().cpu().numpy()


def _local(reference, points: np.ndarray, reach: float):
    """The part of a reference within ``reach`` of ``points``: its triangles with a corner
    there, with their normals and the ``border`` region."""
    from scipy.spatial import cKDTree

    from opensculptboy.fit.reference import Reference

    near = cKDTree(points).query(reference.vertices)[0] < reach
    T = np.asarray(reference.triangles)
    T = T[near[T].any(1)]
    used = np.unique(T)
    remap = np.full(len(reference.vertices), -1)
    remap[used] = np.arange(len(used))
    local = Reference(reference.vertices[used], remap[T])
    local.normals = np.asarray(reference.normals)[used]
    border = reference.regions.get("border")
    if border is not None:
        local.regions["border"] = np.asarray(border, bool)[used]
    return local


def _subset(reference, keep: np.ndarray):
    """The triangles of a reference whose corners are all kept, as a reference."""
    from opensculptboy.fit.reference import Reference

    T = np.asarray(reference.triangles)
    T = T[np.asarray(keep, bool)[T].all(1)]
    used = np.unique(T)
    remap = np.full(len(reference.vertices), -1)
    remap[used] = np.arange(len(used))
    out = Reference(reference.vertices[used], remap[T])
    out.normals = np.asarray(reference.normals)[used]
    return out


def _trimmed(values: np.ndarray, share: float) -> float:
    values = np.sort(np.asarray(values, np.float64))
    if not len(values):
        return float("inf")
    return float(values[: max(1, int(share * len(values)))].mean())


def refine_chains(
    model,
    reference,
    character,
    rotation,
    translation,
    chains=("hand.l", "hand.r", "foot.l", "foot.r"),
    options: ChainOptions | None = None,
    no_data: np.ndarray | None = None,
    hints: dict | None = None,
):
    """
    Fit the bones and the local changes (:data:`SHAPES`) of each chain (``"hand.l"``,
    ``"foot.r"``; :data:`CHAINS`) of a fitted card (its sliders, pose and placement:
    ``x_reference = rotation @ x_anny + translation``) to ``reference``
    (:class:`~opensculptboy.fit.reference.Reference`), with every other bone and slider
    held. ``no_data`` (V,) bool: the level-0 vertices that take no pairs. ``hints`` maps
    bones of the chains (``"finger3-1.R"``) to where their heads lie in the reference's
    frame: a person's reading of a knuckle or a toe off pictures of the scan, which guides
    a hand whose fingers the scan's surface alone leaves ambiguous. Returns the card
    with the new pose and local changes, and a report per chain (the start that won, the
    scores in mm).
    """
    import roma
    import torch
    from scipy.spatial import cKDTree

    from opensculptboy.character import Character
    from opensculptboy.fit.register import excluded_vertices, vertex_normals

    options = options or ChainOptions()
    character = Character.from_dict(character.to_dict())
    triangles = model.faces.cpu().numpy()
    usable = ~excluded_vertices(model)
    if no_data is not None:
        usable &= ~np.asarray(no_data, bool)
    full = reference
    labels = list(model.bone_labels)
    library = {}
    report = {}
    pose = character.pose_parameters(model)
    pose = (_identity_pose(model) if pose is None else pose).detach().clone()
    stage_skin = _Skinner(model, character, rotation, translation)
    start_vertices = stage_skin(pose).numpy()
    stage_joints = stage_skin.joints.numpy()

    for chain in chains:
        part, side = _split(chain)
        bones, _ = chain_bones(model, chain)
        on_chain, on_anchor = chain_vertices(model, chain)
        rows = np.nonzero((on_chain | on_anchor) & usable)[0]
        chain_rows = np.nonzero(on_chain & usable)[0]
        if not len(chain_rows):
            continue
        # the scan near the chain alone, so that each closest-point query stays quick
        reference = _local(full, start_vertices[rows], options.region + 0.04)
        border = reference.regions.get("border")
        # the scan's points of the chain: near the chain at the stage's pose, and further
        # than ``explained`` from the rest of the figure (the hip under a hand, the thigh
        # beside it), which the parameter stage fitted well
        near = (
            cKDTree(start_vertices[rows]).query(reference.vertices)[0] < options.region
        )
        if border is not None:
            near &= ~np.asarray(border, bool)
        others = np.nonzero(usable & ~on_chain)[0]
        near[near] &= (
            cKDTree(start_vertices[others]).query(reference.vertices[near])[0]
            > options.explained
        )
        # and within reach of the chain's root (the wrist, the ankle)
        root_bone = [b for b in bones if labels[b].startswith(ROOTS[part])][0]
        near &= (
            np.linalg.norm(reference.vertices - stage_joints[root_bone], axis=1)
            < ROOT_REACH[part]
        )
        points = reference.vertices[near]
        point_normals = reference.normals[near]
        if not len(points):
            report[chain] = {"skipped": "no scan near the chain"}
            continue
        # the chain pairs with its own scan alone: a finger near the hip it rests on was
        # pulled flat onto the hip
        own = _subset(reference, near)
        # and the rest of the figure's scan near it, which the chain must not pass into
        rest = _subset(reference, ~near) if (~near).any() else None
        shapes = chain_shapes(model, chain) if options.shape else []
        before_shape = {n: character.local_changes.get(n, 0.0) for n in shapes}
        for n in shapes:
            character.local_changes.pop(n, None)
        skin = _Skinner(model, character, rotation, translation)
        fingers = [
            k
            for k, b in enumerate(bones)
            if any(labels[b].startswith(w) for w in FINGERS[part])
        ]
        stage = pose[0, bones, :3, :3].double().numpy()
        starts = {"stage": stage}
        for name in STARTS[part]:
            if name not in library:
                library[name] = _library_rotations(model, name)
            whole = library[name][bones]
            mixed = stage.copy()
            mixed[fingers] = whole[fingers]
            starts[f"{name}/fingers"] = mixed
            starts[f"{name}/whole"] = whole
        if options.starts is not None:
            starts = {k: v for k, v in starts.items() if k in set(options.starts)}
        index = torch.as_tensor(bones)

        def pairs(X, limit):
            C, N, d = own.closest(X[chain_rows])[:3]
            normals = vertex_normals(X, triangles)
            # a vertex inside the chain's own marble is left there (anny's fist lies inside
            # the statue's larger one; pulled out, the fingers opened)
            outside = ((X[chain_rows] - C) * N).sum(1) > -0.002
            ok = (
                (d < limit)
                & outside
                & ((normals[chain_rows] * N).sum(1) > options.normal_threshold)
            )
            dist, nearest = cKDTree(X[chain_rows]).query(points)
            mine = chain_rows[nearest]
            take = (dist < limit) & ((normals[mine] * point_normals).sum(1) > 0.0)
            return (
                torch.as_tensor(chain_rows[ok]),
                torch.as_tensor(C[ok]),
                torch.as_tensor(N[ok]),
                torch.as_tensor(mine[take]),
                torch.as_tensor(points[take]),
                torch.as_tensor(point_normals[take]),
            )

        def data_loss(
            V, fwd_rows, fwd_points, fwd_normals, rev_rows, rev_points, rev_n
        ):
            loss = V.new_zeros(())
            for r, p, n in (
                (fwd_rows, fwd_points, fwd_normals),
                (rev_rows, rev_points, rev_n),
            ):
                if len(r):
                    diff = V[r] - p
                    loss = loss + ((diff * n).sum(-1) ** 2).mean()
                    loss = loss + 0.1 * (diff**2).sum(-1).mean()
            return loss

        hinted = [
            (labels.index(b), np.asarray(p, np.float64))
            for b, p in (hints or {}).items()
            if b in labels and labels.index(b) in bones
        ]
        hint_rows = torch.as_tensor([h[0] for h in hinted], dtype=torch.long)
        hint_points = torch.as_tensor(
            np.asarray([h[1] for h in hinted]).reshape(-1, 3), dtype=pose.dtype
        )

        def hint_loss(J):
            if not len(hinted):
                return J.new_zeros(())
            return (
                options.hint_weight * ((J[hint_rows] - hint_points) ** 2).sum(-1).mean()
            )

        def hint_score(J):
            if not len(hinted):
                return 0.0
            d = np.linalg.norm(
                J.numpy()[hint_rows.numpy()] - hint_points.numpy(), axis=1
            )
            return options.hint_score * float(d.mean())

        def score(X):
            # trimmed distances both ways between the chain and the scan's points of the
            # chain, a point whose nearest vertex faces away at least 1 cm off; and the depth
            # of the chain's vertices inside the rest of the figure (fingers that pass into
            # the hip under them). Inside the chain's own marble a vertex costs its distance
            # alone: anny's fist lies inside the statue's larger one until the detail levels
            C, N, forward = own.closest(X[chain_rows])[:3]
            if rest is not None and options.depth > 0:
                C2, N2, d2 = rest.closest(X[chain_rows])[:3]
                depth = np.maximum(-((X[chain_rows] - C2) * N2).sum(1), 0.0)
                depth = np.where(d2 < 0.03, depth, 0.0)
                forward = forward + options.depth * depth
            normals = vertex_normals(X, triangles)
            dist, nearest = cKDTree(X[chain_rows]).query(points)
            facing = (normals[chain_rows[nearest]] * point_normals).sum(1) > 0.0
            dist = np.where(facing, dist, np.maximum(dist, 0.01))
            return _trimmed(forward, options.trim) + _trimmed(dist, options.trim)

        def limit_of(round_, rounds):
            hi, lo = options.reach
            return hi + (lo - hi) * round_ / max(rounds - 1, 1)

        def fit_pose(start, rounds):
            rotvec = torch.nn.Parameter(
                roma.rotmat_to_rotvec(torch.as_tensor(start, dtype=pose.dtype))
            )
            origin = rotvec.detach().clone()
            optimizer = torch.optim.Adam([rotvec], lr=options.rate)
            for round_ in range(rounds):
                with torch.no_grad():
                    current = pose.clone()
                    current[0, index, :3, :3] = roma.rotvec_to_rotmat(rotvec)
                    found = pairs(
                        skin(current).numpy(), limit_of(round_, options.rounds)
                    )
                for _ in range(options.steps):
                    optimizer.zero_grad()
                    current = pose.clone()
                    current[0, index, :3, :3] = roma.rotvec_to_rotmat(rotvec)
                    loss = data_loss(skin(current), *found) + hint_loss(skin.joints)
                    loss = loss + options.prior * ((rotvec - origin) ** 2).sum()
                    loss.backward()
                    optimizer.step()
            with torch.no_grad():
                current = pose.clone()
                current[0, index, :3, :3] = roma.rotvec_to_rotmat(rotvec)
                X = skin(current).numpy()
                return rotvec.detach(), score(X) + hint_score(skin.joints)

        with torch.no_grad():
            X = skin(pose).numpy()
            stage_score = score(X) + hint_score(skin.joints)
        # a search over the turn of the chain's root (the wrist, the foot) for each start,
        # scored as they are, then the best few fitted
        root = [k for k, b in enumerate(bones) if labels[b].startswith(ROOTS[part])]
        steps_ = np.radians(np.asarray(options.search, np.float64))
        offsets = roma.rotvec_to_rotmat(
            torch.as_tensor(
                np.stack(np.meshgrid(steps_, steps_, steps_), -1).reshape(-1, 3)
            )
        ).numpy()
        searched = []
        with torch.no_grad():
            for name, start in starts.items():
                for j, offset in enumerate(offsets):
                    trial = start.copy()
                    trial[root] = trial[root] @ offset
                    current = pose.clone()
                    current[0, index, :3, :3] = torch.as_tensor(trial, dtype=pose.dtype)
                    X = skin(current).numpy()
                    searched.append(
                        (score(X) + hint_score(skin.joints), f"{name}#{j}", trial)
                    )
        searched.sort(key=lambda item: item[0])
        top = {name: round(1000 * value, 3) for value, name, _ in searched[:8]}
        scores, results = {}, {}
        for value, name, start in searched[: options.keep]:
            results[name], scores[name] = fit_pose(start, options.trial_rounds)
        best = min(scores, key=scores.get)
        rotvec = torch.nn.Parameter(results[best].clone())
        origin = rotvec.detach().clone()
        shaped = _ShapedSkinner(model, character, rotation, translation)
        columns = torch.as_tensor(
            [shaped.labels.index(n) for n in shapes], dtype=torch.long
        )
        values = torch.nn.Parameter(torch.zeros(len(shapes), dtype=pose.dtype))
        optimizer = torch.optim.Adam(
            [{"params": [rotvec], "lr": options.rate}, {"params": [values], "lr": 0.02}]
        )
        for round_ in range(options.rounds):
            with torch.no_grad():
                current = pose.clone()
                current[0, index, :3, :3] = roma.rotvec_to_rotmat(rotvec)
                found = pairs(
                    shaped(current, columns, values).numpy(),
                    limit_of(round_, options.rounds),
                )
            for _ in range(options.steps):
                optimizer.zero_grad()
                current = pose.clone()
                current[0, index, :3, :3] = roma.rotvec_to_rotmat(rotvec)
                loss = data_loss(shaped(current, columns, values), *found)
                loss = loss + hint_loss(shaped.joints)
                loss = loss + options.prior * ((rotvec - origin) ** 2).sum()
                loss = loss + options.shape_prior * (values**2).sum()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    values.clamp_(-1.0, 1.0)
        with torch.no_grad():
            trial = pose.clone()
            trial[0, index, :3, :3] = roma.rotvec_to_rotmat(rotvec)
            X = shaped(trial, columns, values).numpy()
            final = score(X) + hint_score(shaped.joints)
            if final > scores[best]:
                # the joint fit came out worse than its start: keep the start
                rotvec = torch.nn.Parameter(results[best].clone())
                values.zero_()
                trial[0, index, :3, :3] = roma.rotvec_to_rotmat(rotvec)
                X = shaped(trial, columns, values).numpy()
                final = score(X) + hint_score(shaped.joints)
            pose[0, index, :3, :3] = trial[0, index, :3, :3]
            hint_mm = (
                1000 * hint_score(shaped.joints) / options.hint_score
                if hinted
                else None
            )
        fitted_shape = {n: round(float(v), 6) for n, v in zip(shapes, values.tolist())}
        for n, v in fitted_shape.items():
            if abs(v) > 1e-6:
                character.local_changes[n] = v
        report[chain] = {
            "start": best,
            "score_mm": round(1000 * final, 3),
            "stage_mm": round(1000 * stage_score, 3),
            "points": int(len(points)),
            "search_mm": top,
            "hints_mm": hint_mm,
            "trials_mm": {k: round(1000 * v, 3) for k, v in scores.items()},
            "shape_before": before_shape,
            "shape": fitted_shape,
        }
        if options.verbose:
            print(
                f"chain {chain}: start {best}, {1000 * final:.2f} mm (stage "
                f"{1000 * stage_score:.2f} mm, best trial {1000 * scores[best]:.2f} mm, "
                f"{len(points)} points)",
                flush=True,
            )
    character.pose = Character.pose_from_parameters(model, pose)
    return character, report

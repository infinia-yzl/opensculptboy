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
# a solid statue: the eyeballs' front takes the carved eye within this distance (metres)
SOLID_REACH = 0.015
# and seals the lips where anny's fitted lips meet when their rows move up to this distance
# (metres): a statue's lips are carved shut, with no crease for the fit to find
SOLID_LIP_SHIFT = 0.012
# and untangles the folds by the lids' and the lips' corners in this many rounds
SOLID_UNFOLD = 10
# and stitches the lips' rows that are each other's nearest within this distance (metres),
# as the seal left a slit in the middle of Doryphoros' mouth
SOLID_LIP_CONTACT = 0.006
# the rings of skin around the eyes' and the mouth's openings (the linings of the sockets and
# of the mouth) that keep anny's shape on a figure with a working face: on Doryphoros 2 and 3
# rings left dark gaps under the lids and between the lips, 4 a small one at the lips'
# corner, and 6 narrowed the eyes
HOLD_EYE_RINGS = 5
HOLD_LIP_RINGS = 5
# a working face's eyes grow this much against their lids, and the sockets' lining comes to
# this far (metres) outside them: on David the corners of the eyes showed the sockets dark
EYE_GROW = 1.05
EYE_HUG = 0.001
# the fitted skin's pits deeper than the scan's own by this much (metres) relax
PIT_EXCESS = 0.0005
# the rings of the scan's vertices around its open borders whose pairs the refinement drops
BORDER_RINGS = 2
# and the scan's vertices within this distance (metres) of them
BORDER_REACH = 0.012


def open_border(
    vertices: np.ndarray,
    triangles: np.ndarray,
    rings: int = BORDER_RINGS,
    reach: float = BORDER_REACH,
):
    """The vertices (V,) bool on the open borders of a mesh (the edges of one triangle, the
    cuts where supports and a hat came off and the holes of a scan), ``rings`` rings around
    them and every vertex within ``reach`` (metres) of them, with the vertices at one
    position welded: a dense scan has many rings within a millimetre."""
    from scipy.spatial import cKDTree

    from opensculptboy.fit.register import welded_ids

    V = np.asarray(vertices, dtype=np.float64)
    T = np.asarray(triangles)
    ids = welded_ids(V)
    W = ids[T]
    edges = np.sort(np.concatenate([W[:, [0, 1]], W[:, [1, 2]], W[:, [2, 0]]]), 1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    on = np.zeros(ids.max() + 1, bool)
    on[unique[counts == 1].ravel()] = True
    edge = on[ids]
    for _ in range(rings):
        grown = on.copy()
        touched = on[W].any(1)
        grown[W[touched].ravel()] = True
        on = grown
    out = on[ids]
    if reach > 0 and edge.any():
        d, _ = cKDTree(V[edge]).query(V, distance_upper_bound=reach)
        out |= np.isfinite(d)
    return out


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
    the parameter stage's start: each pose on anny's default body, standing on the floor,
    centred over the reference and scaled to its height, scored by the mean of the nearest
    90 % of the distances both ways between its vertices and points of the reference (a
    trimmed chamfer distance, in metres). Returns the
    name, the (B, 3, 3) rotations and the scores by name. The reference stands upright in
    Anny's frame and faces -Y, as :func:`~opensculptboy.statues.scan_to_handoff` writes it.
    """
    import anny.poses
    import torch
    from scipy.spatial import cKDTree

    from opensculptboy.statues.scan import surface_points

    R = reference.vertices
    # points spread by area: a scan packs its vertices where it has detail (the curls)
    target = surface_points(R, reference.triangles, samples, seed)[-samples:]
    target_tree = cKDTree(target)
    height = float(np.ptp(R[:, 2]))
    faces = model.faces.cpu().numpy()

    def centre(P):
        # the mean of the trunk's band, which arms, props and leftovers of supports move least
        z = (P[:, 2] - P[:, 2].min()) / max(np.ptp(P[:, 2]), 1e-9)
        band = P[(z > 0.5) & (z < 0.75)]
        return (band if len(band) else P)[:, :2].mean(0)

    def placed(V):
        # on the floor, centred over the reference and scaled to its height, so that the
        # score compares the poses alone
        V = surface_points(V, faces, samples, seed)[-samples:]
        V = V - [*centre(V), V[:, 2].min()]
        V = V * (height / max(np.ptp(V[:, 2]), 1e-9))
        return V + [*centre(target), R[:, 2].min()]

    def score(V):
        # the mean of the nearest 90 % of the distances both ways: a hat or a stump left in
        # the scan weighs little
        a = np.sort(cKDTree(V).query(target)[0])
        b = np.sort(target_tree.query(V)[0])
        return float(a[: int(0.9 * len(a))].mean() + b[: int(0.9 * len(b))].mean())

    library = anny.poses.library()
    names = list(names or library.names("pose"))
    scores, poses = {}, {}
    with torch.no_grad():
        V = model()["vertices"][0].double().numpy()
    scores["rest"] = score(placed(V))
    poses["rest"] = np.eye(3)[None].repeat(len(model.bone_labels), 0)
    for name in names:
        posed = anny.poses.pose_parameters(model, name)
        V = posed["vertices"][0].double().numpy()
        scores[name] = score(placed(V))
        poses[name] = posed["pose_parameters"][0, :, :3, :3].double().numpy()
    best = min(scores, key=scores.get)
    return best, poses[best], scores


def fit_statue(
    folder,
    detail: int = 2,
    name: str | None = None,
    options=None,
    start_pose: str | None = "auto",
    solid: bool = False,
    stage1: dict | None = None,
    hold: tuple = (HOLD_EYE_RINGS, HOLD_LIP_RINGS),
    chains: tuple = (),
    chain_options=None,
):
    """
    Fit anny to a statue handoff folder (:func:`~opensculptboy.statues.scan_to_handoff`):
    :func:`~opensculptboy.fit.register.fit_reference` with the parts that ``statue.lost`` of
    appearance.json names left without data, the pairs on the scan's open borders dropped,
    and the parameter stage started from ``start_pose``: a pose of anny's library by name,
    ``"auto"`` for the closest one (:func:`library_start`) or None for the rest pose. By
    default the figure keeps anny's working face: the skin within ``hold`` rings (eyes, lips)
    of the eyes' and the mouth's openings takes no data and follows the fitted skin around it
    (:func:`held_openings`), and the lips meet where anny's own edges meet
    (``FitOptions.lip_seal_self``), so that the lids and the lips keep anny's openings and
    close and open as anny's do. A ``solid`` statue has carved eyes and
    closed lips: the fit closes anny's eye and mouth openings on the scan and hides the
    eyeballs behind them (``FitOptions.close_openings``).
    With ``stage1`` (:func:`stage1_dict` of an earlier fit), the sliders, pose and placement
    are that fit's and only the detail levels run
    (:func:`~opensculptboy.fit.register.fit_levels`). ``chains`` (``"hand.l"``, ``"foot.r"``)
    are fitted one limb at a time after the parameters and before the detail levels
    (:func:`~opensculptboy.statues.chains.refine_chains`, with ``chain_options``); the
    result's ``chains`` holds their report. Returns the
    :class:`~opensculptboy.fit.register.FitResult`.
    """
    from opensculptboy.fit.handoff import load_handoff
    from opensculptboy.fit.reference import Reference
    from opensculptboy.fit.register import FitOptions, fit_model, fit_reference

    folder = pathlib.Path(folder)
    appearance = json.loads((folder / "appearance.json").read_text(encoding="utf-8"))
    statue = appearance.get("statue") or {}
    lost = list(statue.get("lost") or [])
    covered = list(statue.get("covered") or [])
    covered_levels = list(statue.get("covered_levels") or [])
    options = options or FitOptions()
    model = fit_model(options.topology)
    if lost or covered:
        no_data = lost_mask(model, lost)
        no_data[np.asarray(covered, np.int64)] = True
        options = dataclasses.replace(options, no_data=no_data, fill_no_data=True)
    if covered_levels:
        levels = np.zeros(model.template_vertices.shape[-2], bool)
        levels[np.asarray(covered_levels, np.int64)] = True
        options = dataclasses.replace(
            options,
            no_data_levels=levels,
            fill_no_data=True,
            held_bones=tuple(options.held_bones) + held_feet(model, levels),
        )
    if options.relax_dents == 0:
        # a dense fit of a coarse scan dents single vertices (a pit on a forehead)
        options = dataclasses.replace(options, relax_dents=PIT_EXCESS)
    if not solid:
        # anny's working face: the lids and the lips around their openings follow the fitted
        # skin (on the carved eyes and lips they tore into slits), and the lips meet where
        # anny's own edges meet (held apart, they showed a dark slit)
        held = held_openings(model, *hold)
        no_data = held if options.no_data is None else (options.no_data | held)
        options = dataclasses.replace(
            options,
            no_data=no_data,
            fill_no_data=True,
            lip_seal_self=True,
            unfold_rounds=options.unfold_rounds or SOLID_UNFOLD,
            eyes_to_lids=True,
            eyes_to_lids_scale=(
                EYE_GROW
                if options.eyes_to_lids_scale == 1.0
                else options.eyes_to_lids_scale
            ),
            eyes_lining=EYE_HUG if options.eyes_lining < 0 else options.eyes_lining,
        )
    if solid and options.close_openings == 0:
        # carved eyes, and closed lips sealed further than a character's would be
        options = dataclasses.replace(
            options,
            close_openings=SOLID_REACH,
            lip_seal_max_shift=max(options.lip_seal_max_shift, SOLID_LIP_SHIFT),
            lip_seal_self=True,
            unfold_rounds=SOLID_UNFOLD,
            lip_contact=max(options.lip_contact, SOLID_LIP_CONTACT),
        )
    handoff = load_handoff(folder)
    reference = Reference.from_handoff(handoff)
    reference.regions["border"] = open_border(reference.vertices, reference.triangles)
    locks = [p for p in handoff.parts if p.role == "hair"]
    if locks and options.cover_pieces is None and covered:
        # the cheek and the neck under fused locks come out to them
        base = np.cumsum([0] + [len(p.vertices) for p in locks[:-1]])
        options = dataclasses.replace(
            options,
            cover_pieces=(
                np.concatenate([p.vertices for p in locks]),
                np.concatenate([p.triangles + b for p, b in zip(locks, base)]),
            ),
        )
    title = (appearance.get("statue") or {}).get("title")
    name = name or title or folder.resolve().name
    if stage1 is None and not chains:
        if start_pose and options.initial_pose is None:
            options = _library_start(model, reference, start_pose, options)
        return fit_reference(
            reference, detail=detail, name=name, options=options, model=model
        )
    import time

    from opensculptboy.fit.register import (
        _character_from,
        fit_levels,
        fit_parameters,
    )

    clock = time.perf_counter()
    if stage1 is not None:
        character, rotation, translation = stage1_from(stage1)
    else:
        if start_pose and options.initial_pose is None:
            options = _library_start(model, reference, start_pose, options)
        params, _, rotation, translation = fit_parameters(model, reference, options)
        character = _character_from(model, params, name)
    seconds = {"parameters": time.perf_counter() - clock}
    report = {}
    if chains:
        from opensculptboy.statues.chains import ChainOptions, refine_chains

        clock = time.perf_counter()
        chain_options = chain_options or ChainOptions(verbose=options.verbose)
        character, report = refine_chains(
            model,
            reference,
            character,
            rotation,
            translation,
            tuple(chains),
            chain_options,
            no_data=options.no_data,
        )
        seconds["chains"] = time.perf_counter() - clock
    result = fit_levels(
        reference, character, rotation, translation, detail, options, model, seconds
    )
    result.chains = report
    return result


def _library_start(model, reference, start_pose, options):
    """The options with the start pose of the parameter stage from anny's library."""
    names = None if start_pose == "auto" else [start_pose]
    chosen, rotations, scores = library_start(model, reference, names)
    if options.verbose:
        print(f"start pose: {chosen} ({1000 * scores[chosen]:.1f} mm)", flush=True)
    return dataclasses.replace(options, initial_pose=rotations)


def held_openings(
    model, eye_rings: int = HOLD_EYE_RINGS, lip_rings: int = HOLD_LIP_RINGS
) -> np.ndarray:
    """Mask (V,) of the skin within ``eye_rings`` faces of the eye sockets' linings and within
    ``lip_rings`` faces of the mouth's, the linings themselves included."""
    from anny.faces.authoring.base_mesh import base_mesh
    from opensculptboy.fit.lips import rings_from
    from opensculptboy.topology.lookup import base_mesh_rows

    mesh = base_mesh()
    base = base_mesh_rows(model)
    faces = model.faces.cpu().numpy()
    n = model.template_vertices.shape[-2]
    out = np.zeros(n, bool)
    for labels, rings in (
        (("eye_cavity.L", "eye_cavity.R"), eye_rings),
        (("mouth_cavity",), lip_rings),
    ):
        lining = np.zeros(len(mesh.V), bool)
        for label in labels:
            lining[np.unique(mesh.quads[mesh.face_labels == label])] = True
        rows = np.nonzero(lining[base])[0]
        if len(rows):
            out |= rings_from(faces, rows, int(rings), n)
    return out


def stage1_dict(result) -> dict:
    """The sliders, pose and placement of a fit (JSON), from which :func:`fit_statue` can run
    the detail levels again."""
    card = result.character.to_dict()
    card.pop("detail", None)
    return {
        "format": "opensculptboy.statue-stage1/1",
        "character": card,
        "rotation": np.asarray(result.rotation, np.float64).tolist(),
        "translation": np.asarray(result.translation, np.float64).tolist(),
    }


def stage1_from(data: dict):
    """The card, rotation and translation of :func:`stage1_dict`."""
    from opensculptboy.character import Character

    if data.get("format") != "opensculptboy.statue-stage1/1":
        raise ValueError(f"Not a statue stage-1 file: {data.get('format')!r}.")
    return (
        Character.from_dict(data["character"]),
        np.asarray(data["rotation"], np.float64),
        np.asarray(data["translation"], np.float64),
    )


# the scan's triangles further than this (metres) from a first fit are not the figure's skin
PRUNE_DISTANCE = 0.025
# the regions of the first fit whose far triangles are hair (kept as a piece) and not props,
# and those that keep all of their skin
HAIR_REGIONS = ("scalp", "neck")
KEEP_REGIONS = ("face", "ears", "hands", "torso")


def prune_far(folder, result, model=None, distance: float = PRUNE_DISTANCE) -> dict:
    """
    Take out of a statue handoff the skin that a first fit (``result``, its level 0) does not
    reach: the triangles of the scan further than ``distance`` from the fitted skin, which are
    props, leftovers of supports and locks of hair standing off the head. Those nearest the
    head and the neck become a part of the role ``hair`` (``hair_cut``), the others go to
    ``removed.glb`` as ``support_cut``. Returns the triangle counts.
    """
    import trimesh
    from scipy.spatial import cKDTree

    from anny.hair import closest_triangles
    from opensculptboy.fit.handoff import GLTF_TO_ANNY, load_handoff
    from opensculptboy.fit.register import excluded_vertices, fit_model
    from opensculptboy.statues.scan import components

    folder = pathlib.Path(folder)
    model = model or fit_model()
    handoff = load_handoff(folder)
    skin = handoff.skin_parts()[0]
    V, T = result.meshes[0]
    excluded = excluded_vertices(model)
    S, ST = skin.vertices, skin.triangles
    centres = S[ST].mean(1)
    # against the whole fitted surface: a statue's carved eyes lie on anny's eyeballs
    _, _, d = closest_triangles(centres, V, T)
    far = d > distance
    from opensculptboy.statues.compare import STATUE_REGIONS, statue_regions

    regions = statue_regions(model)
    rows = np.nonzero(~excluded)[0]
    _, nearest = cKDTree(V[rows]).query(centres)
    region = regions[rows[nearest]]
    # the face, the hands and the trunk keep their skin: a first fit sits off a fist and off
    # what anny's body lacks (the genitals), and props by the trunk are masks' work
    far &= ~np.isin(region, [STATUE_REGIONS.index(r) for r in KEEP_REGIONS])
    head = np.isin(region, [STATUE_REGIONS.index(r) for r in HAIR_REGIONS])
    # the skin left must hold together: loose bits of it go with the far triangles
    near = ~far
    pieces = components(ST[near], len(S))
    p = S[ST[near]]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    sizes = np.bincount(pieces, area)
    loose = np.zeros(len(ST), bool)
    loose[np.nonzero(near)[0][sizes[pieces] < 0.01 * sizes.sum()]] = True
    far |= loose
    groups = {
        "statue": ST[~far],
        "hair_cut": ST[far & head],
        "support_cut": ST[far & ~head],
    }

    def mesh(triangles):
        used = np.unique(triangles)
        remap = np.full(len(S), -1)
        remap[used] = np.arange(len(used))
        return trimesh.Trimesh(S[used] @ GLTF_TO_ANNY, remap[triangles], process=False)

    scene = trimesh.Scene()
    for part in handoff.parts:
        if part.is_skin:
            continue
        scene.add_geometry(
            trimesh.Trimesh(
                part.vertices @ GLTF_TO_ANNY, part.triangles, process=False
            ),
            node_name=part.name,
            geom_name=part.name,
        )
    for name in ("statue", "hair_cut"):
        if len(groups[name]):
            scene.add_geometry(mesh(groups[name]), node_name=name, geom_name=name)
    scene.export(str(folder / "replica.glb"))
    removed_path = folder / "removed.glb"
    removed = (
        trimesh.load(str(removed_path), force="scene")
        if removed_path.exists()
        else trimesh.Scene()
    )
    if len(groups["support_cut"]):
        removed.add_geometry(
            mesh(groups["support_cut"]),
            node_name="support_cut",
            geom_name="support_cut",
        )
        removed.export(str(removed_path))
    appearance_path = folder / "appearance.json"
    appearance = json.loads(appearance_path.read_text(encoding="utf-8"))
    names = {e.get("name") for e in appearance.get("parts", [])}
    if len(groups["hair_cut"]) and "hair_cut" not in names:
        appearance["parts"].append({"name": "hair_cut", "role": "hair"})
    counts = {k: int(len(v)) for k, v in groups.items()}
    appearance.setdefault("statue", {})["pruned"] = {
        "distance": distance,
        "triangles": counts,
    }
    appearance_path.write_text(
        json.dumps(appearance, indent=2) + "\n", encoding="utf-8"
    )
    return counts


# the removed pieces' triangles within this distance (metres) of a first fit's surface, facing
# as it does, are the figure's skin that a coarse mask took (restore_near)
RESTORE_DISTANCE = 0.006
RESTORE_NORMAL = 0.5
# the regions of the first fit whose skin restore_near gives back: the hands stay out, as a
# hand's first fit wraps the grip it holds
RESTORE_REGIONS = ("torso", "arms", "legs", "feet", "neck")
# the removed roles that restore_near looks at
RESTORE_ROLES = ("support", "break", "debris")


def restore_near(
    folder,
    result,
    model=None,
    distance: float = RESTORE_DISTANCE,
    regions=RESTORE_REGIONS,
) -> int:
    """
    Give back to a statue handoff's skin the triangles of its removed pieces (supports,
    breaks, debris) that lie within ``distance`` of a first fit's level 0 (``result``) and
    face as it does, beside the fit's ``regions``: a box round a prop takes the skin of the
    foot that stands on it or of the thigh that it touches, and the first fit, which has no
    data there, still lies within a few millimetres of that skin. The triangles leave
    removed.glb and the props of props.glb and join the skin part of replica.glb. Returns
    the number of triangles given back.
    """
    import trimesh
    from scipy.spatial import cKDTree

    from anny.hair import closest_triangles
    from opensculptboy.fit.handoff import GLTF_TO_ANNY, load_handoff
    from opensculptboy.fit.register import excluded_vertices, fit_model, vertex_normals
    from opensculptboy.statues.compare import STATUE_REGIONS, statue_regions

    folder = pathlib.Path(folder)
    removed_path = folder / "removed.glb"
    if not removed_path.exists():
        return 0
    model = model or fit_model()
    V, T = (np.asarray(a) for a in result.meshes[0])
    V = np.asarray(V, np.float64)
    labels = statue_regions(model)[: len(V)]
    allowed = np.isin(labels, [STATUE_REGIONS.index(r) for r in regions])
    allowed &= ~excluded_vertices(model)[: len(V)]
    T_allowed = T[allowed[T].all(1)]
    normals = vertex_normals(V, T)
    removed = trimesh.load(str(removed_path), force="scene")
    back_v, back_t, count = [], [], 0
    kept = trimesh.Scene()
    taken_centres = []
    for name, geometry in removed.geometry.items():
        P = np.asarray(geometry.vertices, np.float64) @ GLTF_TO_ANNY.T
        F = np.asarray(geometry.faces)
        take = np.zeros(len(F), bool)
        if name in RESTORE_ROLES and len(F) and len(T_allowed):
            centres = P[F].mean(1)
            tri, bary, d = closest_triangles(centres, V, T_allowed)
            n_fit = np.einsum("nk, nkd -> nd", bary, normals[T_allowed[tri]])
            n_fit /= np.maximum(np.linalg.norm(n_fit, axis=1, keepdims=True), 1e-12)
            a, b, c = (P[F[:, k]] for k in range(3))
            n_tri = np.cross(b - a, c - a)
            n_tri /= np.maximum(np.linalg.norm(n_tri, axis=1, keepdims=True), 1e-18)
            take = (d < distance) & ((n_fit * n_tri).sum(1) > RESTORE_NORMAL)
        if take.any():
            used = np.unique(F[take])
            remap = np.full(len(P), -1)
            remap[used] = np.arange(len(used))
            back_v.append(P[used])
            back_t.append(remap[F[take]])
            taken_centres.append(P[F[take]].mean(1))
            count += int(take.sum())
        rest = F[~take]
        if len(rest):
            used = np.unique(rest)
            remap = np.full(len(P), -1)
            remap[used] = np.arange(len(used))
            kept.add_geometry(
                trimesh.Trimesh(P[used] @ GLTF_TO_ANNY, remap[rest], process=False),
                node_name=name,
                geom_name=name,
            )
    if not count:
        return 0
    kept.export(str(removed_path))
    handoff = load_handoff(folder)
    scene = trimesh.Scene()
    for part in handoff.parts:
        pv, pt = part.vertices, part.triangles
        if part.is_skin:
            base = np.cumsum([len(pv)] + [len(v) for v in back_v[:-1]])
            pv = np.concatenate([pv] + back_v)
            pt = np.concatenate([pt] + [t + b for t, b in zip(back_t, base)])
        scene.add_geometry(
            trimesh.Trimesh(pv @ GLTF_TO_ANNY, pt, process=False),
            node_name=part.name,
            geom_name=part.name,
        )
    scene.export(str(folder / "replica.glb"))
    props_path = folder / "props.glb"
    if props_path.exists():
        # the props lose the triangles that went back to the skin
        tree = cKDTree(np.concatenate(taken_centres))
        props = trimesh.load(str(props_path), force="scene")
        out = trimesh.Scene()
        for name, geometry in props.geometry.items():
            P = np.asarray(geometry.vertices, np.float64) @ GLTF_TO_ANNY.T
            F = np.asarray(geometry.faces)
            same = tree.query(P[F].mean(1))[0] < 1e-6
            rest = F[~same]
            if not len(rest):
                continue
            used = np.unique(rest)
            remap = np.full(len(P), -1)
            remap[used] = np.arange(len(used))
            out.add_geometry(
                trimesh.Trimesh(P[used] @ GLTF_TO_ANNY, remap[rest], process=False),
                node_name=name,
                geom_name=name,
            )
        out.export(str(props_path))
    appearance_path = folder / "appearance.json"
    appearance = json.loads(appearance_path.read_text(encoding="utf-8"))
    appearance.setdefault("statue", {})["restored"] = {
        "distance": distance,
        "triangles": count,
    }
    appearance_path.write_text(
        json.dumps(appearance, indent=2) + "\n", encoding="utf-8"
    )
    return count


# the regions whose skin a statue's hat may cover, and those that fused locks may cover: the
# face and the ears beside the locks keep their data, so that the cheeks meet the locks where
# the scan joins them
COVER_REGIONS = {
    "clothing": ("scalp", "ears", "neck", "face"),
    "hair": ("scalp", "ears", "neck"),
    # the pieces the conversion and the pruning took out (supports, the plinth, props): a foot
    # set on a removed head, a leg by a sword and a sole on the plinth keep anny's shape
    "removed": ("feet", "legs", "hands", "arms", "torso"),
}
# and the regions whose skin is covered where the first fit laid it on the scan's cut within
# COVER_CUT of a hat or locks (the face by the locks included)
CUT_REGIONS = {
    "clothing": ("scalp", "ears", "neck", "face"),
    "hair": ("scalp", "ears", "neck", "face"),
}
# the cut there: the scan's border, this many rings and this far (metres) around it
CUT_RINGS = 1
CUT_REACH = 0.003
# a vertex is covered when a hair or drapery piece lies this much (metres) nearer to it than
# the skin, and the skin is further than COVER_SKIN
COVER_MARGIN = 0.002
COVER_SKIN = 0.004
# or when its nearest skin lies along a cut of the scan and a piece lies within this distance
COVER_CUT = 0.02
# and the skin of the limbs within this many rings of the skin a removed piece covers
COVER_GROW = 3


def parametric_vertices(result, model) -> np.ndarray:
    """The posed and placed vertices (V, 3) of a fit's sliders and pose without its offsets,
    or its level 0 for a result without a character."""
    if getattr(result, "character", None) is None:
        return np.asarray(result.meshes[0][0], np.float64)
    import torch

    c = result.character
    with torch.no_grad():
        posed = model(
            pose_parameters=c.pose_parameters(model),
            phenotype_kwargs=dict(c.phenotype),
            local_changes_kwargs=dict(c.local_changes) or None,
            face_shape_kwargs=dict(c.face_shapes) or None,
        )["vertices"][0]
    return posed.double().cpu().numpy() @ np.asarray(result.rotation).T + np.asarray(
        result.translation
    )


# a foot whose skin a removed piece covers for more than this share keeps the start pose's turn
HELD_FOOT_SHARE = 0.4


def held_feet(model, covered: np.ndarray, share: float = HELD_FOOT_SHARE) -> tuple:
    """The bones of the feet (``foot``, its toes and the shin's lower twist ``lowerleg02``, by
    side) whose skin ``covered`` (a mask of the level-0 vertices) covers for more than
    ``share``: the parameter stage keeps their turns at the start pose's, as no skin turns a
    foot that stood on a removed head."""
    labels = list(model.bone_labels)
    weights = model.vertex_bone_weights.detach().double().cpu().numpy()
    indices = model.vertex_bone_indices.cpu().numpy()
    out = []
    for side in ("L", "R"):
        bones = [
            i
            for i, b in enumerate(labels)
            if b == f"foot.{side}" or (b.startswith("toe") and b.endswith(f".{side}"))
        ]
        if not bones:
            continue
        on = (np.isin(indices, bones) * weights).sum(1) > 0.5
        if on.any() and covered[on].mean() > share:
            out += [labels[i] for i in bones]
            # and the twist of the shin above it: free, it turned the foot's toes sideways
            if f"lowerleg02.{side}" in labels:
                out.append(f"lowerleg02.{side}")
    return tuple(out)


def _removed_pieces(folder) -> list:
    """The pieces of a handoff's removed.glb as ("removed", vertices, triangles) in anny's
    frame."""
    import trimesh

    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    path = pathlib.Path(folder) / "removed.glb"
    if not path.exists():
        return []
    scene = trimesh.load(str(path), force="scene")
    out = []
    for geometry in scene.geometry.values():
        if len(getattr(geometry, "faces", [])):
            V = np.asarray(geometry.vertices, np.float64) @ GLTF_TO_ANNY.T
            out.append(("removed", V, np.asarray(geometry.faces)))
    return out


def covered_skin(
    folder,
    result,
    model=None,
    cut: tuple = (CUT_RINGS, CUT_REACH),
    locks_face: bool = False,
    grow: int = 0,
    near_locks: float = 0.0,
    cover_removed: bool = True,
) -> int:
    """
    Mark in a statue handoff (``statue.covered`` in appearance.json) anny's level-0 vertices
    of the head that a hat or fused locks cover, by a first fit (``result``, its sliders and
    pose): those whose nearest piece of the statue is a hair or drapery piece rather than the
    skin.
    The scan has no skin under the hat and the locks, and the skin's cut at their edges would
    pull the skull flat onto it; :func:`fit_statue` gives those vertices no data, so that the
    skull keeps anny's own shape under them and follows the fitted skin around it. Returns
    the number of vertices marked. ``cut`` (rings, metres) is the band along the scan's cut
    by a hat or locks; ``locks_face`` lets locks cover the face beside them too; ``grow``
    widens the covered skin of the head by that many rings, so that the fill meets the
    fitted brow smoothly; ``near_locks`` (metres) covers the skin of the head that the first
    fit's level 0 laid within that distance of fused locks (the cheek and the neck beside
    them kept data from the cut and stayed narrow, which left a dark gap under the locks).
    ``cover_removed`` False leaves the skin by the removed pieces (supports, props, the
    plinth) with its data, for a statue whose masks keep the figure's skin whole.
    """
    from anny.hair import closest_triangles
    from opensculptboy.fit.handoff import load_handoff
    from opensculptboy.fit.register import excluded_vertices, fit_model
    from opensculptboy.statues.compare import STATUE_REGIONS, statue_regions

    folder = pathlib.Path(folder)
    model = model or fit_model()
    handoff = load_handoff(folder)
    skin = handoff.skin_parts()[0]
    covers = [
        (p.role, p.vertices, p.triangles)
        for p in handoff.parts
        if p.role in ("hair", "clothing")
    ]
    if cover_removed:
        covers += _removed_pieces(folder)
    # the sliders and the pose alone: the first fit's level 0 already lies flat on the skin's
    # cut under the hat, and its skull would read as uncovered skin
    V = parametric_vertices(result, model)
    # a limb by a removed piece takes the first fit's own level 0: its sliders and pose alone
    # left a foot on a removed head pointing elsewhere, all of it without data
    fitted0 = np.asarray(result.meshes[0][0], np.float64)
    marked = np.zeros(len(V), bool)
    marked_levels = np.zeros(len(V), bool)
    if covers:
        regions = statue_regions(model)
        excluded = excluded_vertices(model)[: len(V)]
        rows = np.nonzero(~excluded)[0]
        # the skin along its cuts: a vertex that the first fit laid on a cut beside a piece
        # is covered too (a foot flattened on the cut of the head it stood on)
        cut = open_border(skin.vertices, skin.triangles)
        skins = {}
        nearest = {}
        for role, pv, pt in covers:
            X = (fitted0 if role == "removed" else V)[rows]
            if role not in skins:
                tri, _, d_skin = closest_triangles(X, skin.vertices, skin.triangles)
                skins[role] = (d_skin, cut[skin.triangles[tri]].all(1))
            _, _, d = closest_triangles(X, pv, pt)
            nearest[role] = np.minimum(nearest.get(role, np.inf), d)
        for role, d_cover in nearest.items():
            d_skin, on_cut = skins[role]
            allowed = COVER_REGIONS[role]
            if role == "hair" and locks_face:
                allowed = allowed + ("face",)
            head = np.isin(regions[rows], [STATUE_REGIONS.index(r) for r in allowed])
            target = marked_levels if role == "removed" else marked
            target[rows] |= head & (
                ((d_cover + COVER_MARGIN < d_skin) & (d_skin > COVER_SKIN))
                | (on_cut & (d_cover < COVER_CUT))
            )
        if near_locks > 0:
            head = np.isin(
                regions[rows],
                [STATUE_REGIONS.index(r) for r in ("face", "ears", "neck", "scalp")],
            )
            for role, pv, pt in covers:
                if role == "hair":
                    _, _, d_lock = closest_triangles(fitted0[rows], pv, pt)
                    marked[rows] |= head & (d_lock < near_locks)
        # the skin that the first fit laid on the cut along a hat or locks: the brow pulled up
        # to the cut under the brim in a ridge, and the cheek by the locks folded onto their cut
        X0 = fitted0[rows]
        cut_covers = [c for c in covers if c[0] in CUT_REGIONS]
        if cut_covers:
            # the cut itself (a narrow band): the band of open_border reaches over the brows
            rim = open_border(
                skin.vertices, skin.triangles, rings=int(cut[0]), reach=float(cut[1])
            )
            tri0, _, _ = closest_triangles(X0, skin.vertices, skin.triangles)
            on_cut0 = rim[skin.triangles[tri0]].any(1)
        for role, pv, pt in cut_covers:
            _, _, d0 = closest_triangles(X0, pv, pt)
            near = np.isin(
                regions[rows], [STATUE_REGIONS.index(r) for r in CUT_REGIONS[role]]
            )
            marked[rows] |= near & on_cut0 & (d0 < COVER_CUT)
    path = folder / "appearance.json"
    appearance = json.loads(path.read_text(encoding="utf-8"))
    if marked_levels.any() and COVER_GROW > 0:
        # a few rings more around the skin by a removed piece: the skin along the scan's cut
        # there hung over the filled foot like a cuff
        from opensculptboy.fit.lips import rings_from

        triangles = model.faces.cpu().numpy()
        grown = rings_from(
            triangles, np.nonzero(marked_levels)[0], COVER_GROW, len(marked_levels)
        )
        regions = statue_regions(model)
        limbs = np.isin(
            regions,
            [STATUE_REGIONS.index(r) for r in ("feet", "legs", "hands", "arms")],
        )
        marked_levels |= grown & limbs[: len(grown)]
    if grow > 0 and marked.any():
        from opensculptboy.fit.lips import rings_from

        regions = statue_regions(model)
        head = np.isin(
            regions,
            [STATUE_REGIONS.index(r) for r in ("face", "scalp", "ears", "neck")],
        )
        grown = rings_from(
            model.faces.cpu().numpy(), np.nonzero(marked)[0], grow, len(marked)
        )
        marked |= grown & head[: len(grown)] & ~excluded_vertices(model)[: len(grown)]
    statue = appearance.setdefault("statue", {})
    statue["covered"] = np.nonzero(marked)[0].tolist()
    # by removed pieces: no data on the detail levels alone, so that the leg still poses
    statue["covered_levels"] = np.nonzero(marked_levels & ~marked)[0].tolist()
    path.write_text(json.dumps(appearance, indent=2) + "\n", encoding="utf-8")
    return int((marked | marked_levels).sum())


# a hat or a piece of hair seats its open border on the fitted skin within this distance
# (metres), sunk this far into it, and the move fades out over SEAT_BLEND from the border
# (by role: SEAT_BY_ROLE)
SEAT_REACH = 0.02
SEAT_SINK = 0.0005
SEAT_BLEND = 0.025
# by role, the reach and the blend (metres): a hat's brim was cut a slab above the brow, and
# its inner rim comes down onto the skin along a narrow band (at 2.5 cm a dark crescent showed
# under the brim, at 6 cm slits at its sides)
SEAT_BY_ROLE = {"clothing": (0.1, 0.015), "hair": (0.025, 0.025)}
# the border's sawtooth and the moves are averaged along the border within this distance
SEAT_SMOOTH = 0.01
# in this many rounds
SEAT_ROUNDS = 5


# a piece's triangles whose centres lie this far (metres) inside the fitted skin go (shards of
# the brow and the cheek that the masks cut with a hat and locks), and so do the pieces of it
# under TRIM_KEEP of its area
TRIM_INSIDE = 0.002
TRIM_KEEP = 0.02
# the rounds that take the teeth off a piece's open border
TRIM_TEETH = 4
# by role, how far inside the skin a piece's triangles go (None: they stay): the locks lie
# against the cheek and the brim over the skull, and trimmed they showed holes
TRIM_BY_ROLE = {"clothing": None, "hair": None}


def trim_piece(
    vertices: np.ndarray,
    triangles: np.ndarray,
    skin_vertices: np.ndarray,
    skin_triangles: np.ndarray,
    inside: float | None = TRIM_INSIDE,
    keep: float = TRIM_KEEP,
    teeth: int = TRIM_TEETH,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    A piece cut from a statue (a hat, locks of hair) without the triangles whose centres lie
    more than ``inside`` under the fitted skin and without its loose bits under ``keep`` of
    its area (``inside`` None keeps the triangles under the skin), and without the teeth of
    its open border (the triangles with two open edges, in up to ``teeth`` rounds): the masks
    cut whole triangles, and the shards of the brow and the cheek that came with the hat and
    the locks hung jagged under the brim and beside the face. Returns the vertices and the
    triangles that are left (compacted) and the number of triangles dropped.
    """
    from anny.hair import closest_triangles
    from opensculptboy.statues.scan import components

    V = np.asarray(vertices, np.float64)
    T = np.asarray(triangles)
    SV = np.asarray(skin_vertices, np.float64)
    ST = np.asarray(skin_triangles)
    centres = V[T].mean(1)
    tri, bary, _ = closest_triangles(centres, SV, ST)
    corners = SV[ST[tri]]
    on = np.einsum("nk, nkd -> nd", bary, corners)
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-18)
    # the skin's triangles turn outward (anny's mesh)
    kept = np.ones(len(T), bool)
    if inside is not None:
        kept = ((centres - on) * normal).sum(1) > -inside
    for _ in range(teeth):
        # a tooth of the sawtooth: a triangle with two edges on the open border
        edges = np.sort(T[kept][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        _, inverse, counts = np.unique(
            edges, axis=0, return_inverse=True, return_counts=True
        )
        open_edges = (counts[inverse.ravel()] == 1).reshape(-1, 3).sum(1)
        tooth = open_edges >= 2
        if not tooth.any():
            break
        rows = np.nonzero(kept)[0]
        kept[rows[tooth]] = False
    if kept.any():
        p = V[T[kept]]
        area = 0.5 * np.linalg.norm(
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1
        )
        labels = components(T[kept], len(V))
        sizes = np.bincount(labels, area)
        rows = np.nonzero(kept)[0]
        kept[rows[sizes[labels] < keep * sizes.sum()]] = False
    used = np.unique(T[kept])
    remap = np.full(len(V), -1)
    remap[used] = np.arange(len(used))
    return V[used], remap[T[kept]], int((~kept).sum())


# the triangles of a seated hat that lie on the fitted skin within this distance (metres) and
# face as it does go (the strips of the brow's fringe under the brim), by role
ON_SKIN_BY_ROLE = {"clothing": 0.004, "hair": None}


def on_skin(
    vertices: np.ndarray,
    triangles: np.ndarray,
    skin_vertices: np.ndarray,
    skin_triangles: np.ndarray,
    within: float,
    agree: float = 0.5,
) -> np.ndarray:
    """Mask (T,) of a piece's triangles that lie on the skin: their centres within ``within``
    of it, and their normals within acos(``agree``) of the skin's there (a brim over the brow
    faces down and stays)."""
    from anny.hair import closest_triangles

    V = np.asarray(vertices, np.float64)
    T = np.asarray(triangles)
    SV = np.asarray(skin_vertices, np.float64)
    ST = np.asarray(skin_triangles)
    centres = V[T].mean(1)
    tri, _, d = closest_triangles(centres, SV, ST)
    corners = SV[ST[tri]]
    skin_normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    skin_normal /= np.maximum(np.linalg.norm(skin_normal, axis=1, keepdims=True), 1e-18)
    p = V[T]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-18)
    return (d < within) & (np.abs((normal * skin_normal).sum(1)) > agree)


# a piece's holes (open border loops) up to this perimeter (metres) are filled
HOLE_PERIMETER = 0.2
# by role: a lock's larger holes filled as a fan showed a flat star on the lock
HOLE_BY_ROLE = {"clothing": HOLE_PERIMETER, "hair": 0.05}


def fill_holes(
    vertices: np.ndarray, triangles: np.ndarray, perimeter: float = HOLE_PERIMETER
) -> tuple[np.ndarray, np.ndarray, int]:
    """A piece with its holes up to ``perimeter`` filled: each such loop of open edges gets
    a vertex at its centre and a fan of triangles that turn as the piece's around the loop
    (the masks left holes in the brim, which showed dark from below). Returns the vertices,
    the triangles and the number of holes filled."""
    V = np.asarray(vertices, np.float64)
    T = np.asarray(triangles)
    directed = T[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    keys = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    open_edges = directed[counts[inverse.ravel()] == 1]
    following = {}
    for a, b in open_edges:
        following.setdefault(int(a), []).append(int(b))
    seen = set()
    new_V, new_T, filled = [V], [T], 0
    count = len(V)
    for start in list(following):
        if (start, following[start][0]) in seen:
            continue
        loop, a = [start], start
        ok = True
        while True:
            nexts = [b for b in following.get(a, []) if (a, b) not in seen]
            if not nexts:
                ok = False
                break
            b = nexts[0]
            seen.add((a, b))
            if b == start:
                break
            loop.append(b)
            a = b
            if len(loop) > 2000:
                ok = False
                break
        if not ok or len(loop) < 3:
            continue
        P = V[loop]
        length = np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1).sum()
        if length > perimeter:
            continue
        centre = P.mean(0)
        new_V.append(centre[None])
        # the open edge a -> b of a triangle runs the other way in the filling one
        fan = [(b, a, count) for a, b in zip(loop, loop[1:] + loop[:1])]
        new_T.append(np.asarray(fan, T.dtype))
        count += 1
        filled += 1
    return np.concatenate(new_V), np.concatenate(new_T), filled


def finish_piece(
    vertices: np.ndarray,
    triangles: np.ndarray,
    skin_vertices: np.ndarray,
    skin_triangles: np.ndarray,
    role: str,
    seat: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """A hat or locks cut from a statue made ready to wear on the fitted skin: trimmed
    (:func:`trim_piece`), seated (:func:`seat_piece`), and for a hat the strips that the seat
    laid on the skin taken out and the piece trimmed and seated again; ``seat`` (reach,
    blend) overrides :data:`SEAT_BY_ROLE`. Returns the vertices, the triangles and what was
    done."""
    reach, blend = seat or SEAT_BY_ROLE[role]
    V, T, dropped = trim_piece(
        vertices, triangles, skin_vertices, skin_triangles, inside=TRIM_BY_ROLE[role]
    )
    V, T, holes = fill_holes(V, T, HOLE_BY_ROLE[role])
    V, moved = seat_piece(V, T, skin_vertices, skin_triangles, reach=reach, blend=blend)
    stripped = 0
    within = ON_SKIN_BY_ROLE[role]
    if within:
        strip = on_skin(V, T, skin_vertices, skin_triangles, within)
        stripped = int(strip.sum())
        if stripped:
            V, T, again = trim_piece(
                V, T[~strip], skin_vertices, skin_triangles, inside=None
            )
            dropped += again
            V, moved = seat_piece(
                V, T, skin_vertices, skin_triangles, reach=reach, blend=blend
            )
    return (
        V,
        T,
        {
            "dropped": dropped,
            "holes_filled": holes,
            "on_skin": stripped,
            "border_moved": moved,
        },
    )


def seat_piece(
    vertices: np.ndarray,
    triangles: np.ndarray,
    skin_vertices: np.ndarray,
    skin_triangles: np.ndarray,
    reach: float = SEAT_REACH,
    sink: float = SEAT_SINK,
    blend: float = SEAT_BLEND,
    smooth: float = SEAT_SMOOTH,
    rounds: int = SEAT_ROUNDS,
) -> tuple[np.ndarray, int]:
    """
    The vertices (V, 3) of a piece cut from a statue (a hat, locks of hair) seated on the
    fitted skin, and the number of border vertices that moved. The piece was cut where it
    met the scan's skin, and the fitted skin under it keeps anny's shape there, so a gap
    showed between them. The piece's open border is first smoothed along itself (the mask
    cut whole triangles out and left a sawtooth); each border vertex then moves onto the
    skin, ``sink`` inside, by a share that falls from 1 at the skin to 0 at ``reach``; the
    moves are averaged along the border within ``smooth`` and fade out smoothly within
    ``blend`` of the border, so that the piece keeps its carving.
    """
    from scipy.spatial import cKDTree

    from anny.hair import closest_triangles

    V = np.asarray(vertices, np.float64).copy()
    T = np.asarray(triangles)
    SV = np.asarray(skin_vertices, np.float64)
    ST = np.asarray(skin_triangles)
    border = np.nonzero(open_border(V, T, rings=0, reach=0.0))[0]
    if len(border) == 0:
        return V, 0
    tree = cKDTree(V[border])
    groups = tree.query_ball_point(V[border], smooth)
    # the sawtooth of the cut, smoothed along the border
    B = V[border].copy()
    for _ in range(rounds):
        B = np.array([B[g].mean(0) for g in groups])
    tri, bary, d = closest_triangles(B, SV, ST)
    corners = SV[ST[tri]]
    on = np.einsum("nk, nkd -> nd", bary, corners)
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-18)
    share = np.clip((reach - d) / (0.5 * reach), 0.0, 1.0)
    move = B - V[border] + share[:, None] * (on - sink * normal - B)
    for _ in range(2):
        move = np.array([move[g].mean(0) for g in groups])
    r, k = tree.query(V, distance_upper_bound=blend)
    near = np.isfinite(r)
    t = np.zeros(len(V))
    t[near] = 1.0 - r[near] / blend
    w = t * t * (3.0 - 2.0 * t)
    V[near] += w[near, None] * move[k[near]]
    return V, int((share > 0).sum())

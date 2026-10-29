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
    solid: bool = True,
):
    """
    Fit anny to a statue handoff folder (:func:`~opensculptboy.statues.scan_to_handoff`):
    :func:`~opensculptboy.fit.register.fit_reference` with the parts that ``statue.lost`` of
    appearance.json names left without data, the pairs on the scan's open borders dropped,
    and the parameter stage started from ``start_pose``: a pose of anny's library by name,
    ``"auto"`` for the closest one (:func:`library_start`) or None for the rest pose. A
    ``solid`` statue has carved eyes and closed lips: the fit closes anny's eye and mouth
    openings on the scan and hides the eyeballs behind them (``FitOptions.close_openings``).
    Returns
    the :class:`~opensculptboy.fit.register.FitResult`.
    """
    from opensculptboy.fit.handoff import load_handoff
    from opensculptboy.fit.reference import Reference
    from opensculptboy.fit.register import FitOptions, fit_model, fit_reference

    folder = pathlib.Path(folder)
    appearance = json.loads((folder / "appearance.json").read_text(encoding="utf-8"))
    statue = appearance.get("statue") or {}
    lost = list(statue.get("lost") or [])
    covered = list(statue.get("covered") or [])
    options = options or FitOptions()
    model = fit_model(options.topology)
    if lost or covered:
        no_data = lost_mask(model, lost)
        no_data[np.asarray(covered, np.int64)] = True
        options = dataclasses.replace(options, no_data=no_data, fill_no_data=True)
    if options.relax_dents == 0:
        # a dense fit of a coarse scan dents single vertices (a pit on a forehead)
        options = dataclasses.replace(options, relax_dents=PIT_EXCESS)
    if solid and options.close_openings == 0:
        # carved eyes, and closed lips sealed further than a character's would be
        options = dataclasses.replace(
            options,
            close_openings=SOLID_REACH,
            lip_seal_max_shift=max(options.lip_seal_max_shift, SOLID_LIP_SHIFT),
            lip_seal_self=True,
            unfold_rounds=SOLID_UNFOLD,
        )
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
# a vertex is covered when a hair or drapery piece lies this much (metres) nearer to it than
# the skin, and the skin is further than COVER_SKIN
COVER_MARGIN = 0.002
COVER_SKIN = 0.004
# or when its nearest skin lies along a cut of the scan and a piece lies within this distance
COVER_CUT = 0.02


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


def covered_skin(folder, result, model=None) -> int:
    """
    Mark in a statue handoff (``statue.covered`` in appearance.json) anny's level-0 vertices
    of the head that a hat or fused locks cover, by a first fit (``result``, its sliders and
    pose): those whose nearest piece of the statue is a hair or drapery piece rather than the
    skin.
    The scan has no skin under the hat and the locks, and the skin's cut at their edges would
    pull the skull flat onto it; :func:`fit_statue` gives those vertices no data, so that the
    skull keeps anny's own shape under them and follows the fitted skin around it. Returns
    the number of vertices marked.
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
    covers += _removed_pieces(folder)
    # the sliders and the pose alone: the first fit's level 0 already lies flat on the skin's
    # cut under the hat, and its skull would read as uncovered skin
    V = parametric_vertices(result, model)
    # a limb by a removed piece takes the first fit's own level 0: its sliders and pose alone
    # left a foot on a removed head pointing elsewhere, all of it without data
    fitted0 = np.asarray(result.meshes[0][0], np.float64)
    marked = np.zeros(len(V), bool)
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
            head = np.isin(
                regions[rows], [STATUE_REGIONS.index(r) for r in COVER_REGIONS[role]]
            )
            marked[rows] |= head & (
                ((d_cover + COVER_MARGIN < d_skin) & (d_skin > COVER_SKIN))
                | (on_cut & (d_cover < COVER_CUT))
            )
    path = folder / "appearance.json"
    appearance = json.loads(path.read_text(encoding="utf-8"))
    appearance.setdefault("statue", {})["covered"] = np.nonzero(marked)[0].tolist()
    path.write_text(json.dumps(appearance, indent=2) + "\n", encoding="utf-8")
    return int(marked.sum())


# a hat or a piece of hair seats its open border on the fitted skin within this distance
# (metres), sunk this far into it, and the move fades out over SEAT_BLEND from the border
# (by role: SEAT_BY_ROLE)
SEAT_REACH = 0.02
SEAT_SINK = 0.0005
SEAT_BLEND = 0.025
# by role, the reach and the blend (metres): a hat's brim was cut a slab above the brow
SEAT_BY_ROLE = {"clothing": (0.025, 0.03), "hair": (0.015, 0.02)}
# the border's sawtooth and the moves are averaged along the border within this distance
SEAT_SMOOTH = 0.006


def seat_piece(
    vertices: np.ndarray,
    triangles: np.ndarray,
    skin_vertices: np.ndarray,
    skin_triangles: np.ndarray,
    reach: float = SEAT_REACH,
    sink: float = SEAT_SINK,
    blend: float = SEAT_BLEND,
    smooth: float = SEAT_SMOOTH,
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
    for _ in range(3):
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

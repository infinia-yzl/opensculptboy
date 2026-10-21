# OpenSculptBoy
# Apache License, Version 2.0
"""
The eye-corner patch of the ``sculpt`` topology: a caruncle and a fold of the conjunctiva in
the inner corner of each eye, in place of anny's pocket of skin there.

Anny's lids are concentric rings of quads about the gaze: the skin of the lids turns in at the
lid margins and goes on behind the eyeball as the lining of the socket. In the inner corner of
the eye the rings between the eyeball's sphere and the lid margins form a pocket, a wall of skin
that runs back into the head beside the eyeball. The view between the lids sees that wall, from
the front and more from the side of the temple, and a render shows it as a dark hole where a
real eye has the caruncle (a low pink mound) and the plica (a fold of the conjunctiva that lies
on the eyeball).

The patch finds the pocket on anny's quad mesh (``anny-quads``) from its geometry on the
default character (:func:`pocket_region`): the rings about the eye are the levels of a
breadth-first walk over the skin from the back of the socket, and the pocket is the band of
quads from the ring just inside the eyeball's sphere (:data:`INNER_RING`) out to the lid
margins (:data:`OUTER_RING`, one ring further in the canthus), over the angles about the gaze
that the corner spans (:data:`SECTOR`). It removes those quads and adds the same quads again
over new vertices in place of the band's inner vertices, so that the new faces join anny's own
along the band's border without a seam:

- each new vertex has coordinates (s, t) over the band (:func:`harmonic`): s runs from the
  inner ring (0) to the lid margins (1), and t across the corner from the upper lid (0) to the
  lower lid (1);
- its designed position (:func:`designed_point`, in the eye's frame of
  :mod:`opensculptboy.eyes.geometry`) follows a profile of s: from the lid margins the caruncle
  slopes inward and down to the crest of the plica, which lies just outside the eyeball's
  sphere a little toward the gaze from where anny's skin met the sphere, and the fold returns
  from the crest under the sphere to the inner ring, so that it closes the corner behind the
  crest. Toward both ends of the band (t near 0 and 1) the design blends back into anny's own
  lid margins;
- the new vertex is a convex combination of at most :data:`MAX_ANCHORS` anny vertices near it
  (:func:`convex_weights`: non-negative weights that sum to 1, fitted to the designed position
  on the default character), so that every phenotype, local change, face shape, facial action
  and the skin weights carry over to it exactly (``apply_retopology``), and its UV is the same
  combination of the anchors' UVs, inside the chart of the removed quads.

The right eye's patch is the mirror image of the left's across X = 0 (anny's template is
symmetric), with the same weights. Each new face is tagged ``"caruncle"`` (the mound),
``"conjunctiva"`` (the plica and the fold) or ``"skin"`` (the lid margins where the design
blends back into anny's), so that the eyes of :mod:`opensculptboy.eyes` can draw the first two
pink and wet.
"""

from __future__ import annotations

import numpy as np

from opensculptboy.topology.patch import BaseQuadMesh, Patch, blend_rows

NAME = "eye_corner"
ORDER = 100
ENABLED = True
OPTIONS: dict = {}
# new vertices are convex combinations of at most this many anny vertices
MAX_ANCHORS = 8
# the anchors come from the skin within this many rings of quads around the band
ANCHOR_RINGS = 3
# the rings of the band: the levels of the walk from the back of the socket (skin within
# RING_REACH eyeball radii of the eye's centre, starting at the skin RING_START behind the
# centre along the gaze). The band runs from INNER_RING (just inside the eyeball's sphere) to
# OUTER_RING, and to OUTER_RING_CANTHUS over the angles CANTHUS (degrees about the gaze, 180
# toward the nose)
RING_REACH = 1.8
RING_START = -0.3
INNER_RING = 7
OUTER_RING = 11
OUTER_RING_CANTHUS = 12
SECTOR = (135.0, 232.0)
CANTHUS = (172.0, 212.0)
# the design, in eyeball radii: the crest of the plica lies at s = CREST, PLICA_GAP outside the
# eyeball's sphere and PLICA_SHIFT nearer the gaze axis than anny's skin (over the middle of
# the band, SHIFT_SPAN of t); the fold returns FOLD_GAP inside the sphere; the caruncle rises
# from the crest to CARUNCLE_SINK under the level of the lids (the grey closing of a front view
# of the skin over LID_WINDOW, at LID_STEP a texel); the design blends into anny's lid margins
# over BLEND of t at both ends of the band
CREST = 0.35
PLICA_GAP = 0.06
PLICA_SHIFT = 0.08
SHIFT_SPAN = (0.1, 0.95)
FOLD_GAP = 0.03
CARUNCLE_SINK = 0.0
LID_WINDOW = 0.25
LID_STEP = 0.01
BLEND = 0.2
# faces with s above CREST are the caruncle's up to SKIN_S, and skin beyond it or where the
# design replaces less than SKIN_BLEND of anny's lid margins
SKIN_S = 0.8
SKIN_BLEND = 0.25
TAGS = ("caruncle", "conjunctiva", "skin")


def _edges(quads: np.ndarray) -> np.ndarray:
    quads = np.asarray(quads, dtype=np.int64)
    edges = np.concatenate([quads[:, [i, (i + 1) % 4]] for i in range(4)])
    return edges[edges[:, 0] != edges[:, 1]]


def _adjacency(count: int, quads: np.ndarray):
    import scipy.sparse as sp

    e = _edges(quads)
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(count, count))
    return ((a + a.T) > 0).astype(np.float64).tocsr()


def ring_levels(rest: np.ndarray, quads: np.ndarray, skin_quads: np.ndarray, frame):
    """The ring of each vertex about an eye (inf away from it): the level of a breadth-first
    walk over the skin's edges from the back of the socket."""
    from scipy.sparse.csgraph import shortest_path

    local = frame.to_local(rest)
    dist = np.linalg.norm(local, axis=1)
    used = np.zeros(len(rest), bool)
    used[skin_quads.ravel()] = True
    near = np.flatnonzero(used & (dist < RING_REACH))
    graph = _adjacency(len(rest), skin_quads)[near][:, near]
    start = np.flatnonzero(local[near, 2] < RING_START)
    levels = np.full(len(rest), np.inf)
    levels[near] = shortest_path(
        graph, unweighted=True, indices=start, directed=False
    ).min(0)
    return levels


def _angle(points: np.ndarray, side: str) -> np.ndarray:
    """The angle about the gaze in degrees, 180 toward the nose (the left eye's frame; the
    right eye mirrored)."""
    x = points[..., 0] if side == "L" else -points[..., 0]
    return np.degrees(np.arctan2(points[..., 1], x)) % 360.0


def pocket_region(rest, quads, skin_quads, frame, side: str = "L") -> np.ndarray:
    """The quads (indices) of the pocket in the inner corner of an eye."""
    levels = ring_levels(rest, quads, skin_quads, frame)
    local = frame.to_local(rest)
    centre = local[quads].mean(1)
    angle = _angle(centre, side)
    q_levels = levels[quads]
    outer = np.where(
        (angle > CANTHUS[0]) & (angle < CANTHUS[1]), OUTER_RING_CANTHUS, OUTER_RING
    )
    region = (q_levels.min(1) >= INNER_RING) & (q_levels.max(1) <= outer)
    region &= (angle > SECTOR[0]) & (angle < SECTOR[1])
    skin = np.zeros(len(quads), bool)
    # quads of the skin only
    skin_set = {tuple(q) for q in np.asarray(skin_quads).tolist()}
    skin[[i for i, q in enumerate(quads.tolist()) if tuple(q) in skin_set]] = True
    return np.flatnonzero(region & skin), levels


def harmonic(vertices: np.ndarray, quads: np.ndarray, fixed: dict) -> np.ndarray:
    """The harmonic function (uniform weights) over the graph of ``quads`` on ``vertices``
    with the values ``fixed`` ({vertex: value}); returns its values in ``vertices`` order."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    vertices = np.asarray(vertices)
    pos = {int(v): i for i, v in enumerate(vertices)}
    e = np.array([[pos[int(a)], pos[int(b)]] for a, b in _edges(quads)])
    n = len(vertices)
    w = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    w = ((w + w.T) > 0).astype(np.float64)
    lap = (sp.diags(np.asarray(w.sum(1)).ravel()) - w).tocsr()
    f = np.array([pos[int(v)] for v in fixed])
    val = np.array(list(fixed.values()), dtype=np.float64)
    free = np.setdiff1d(np.arange(n), f)
    x = np.zeros(n)
    x[f] = val
    x[free] = spsolve(lap[free][:, free].tocsc(), -(lap[free][:, f] @ val))
    return x


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def blend(t):
    """How far the design replaces anny's lid margins at t across the band."""
    return _smoothstep(np.asarray(t) / BLEND) * _smoothstep((1.0 - np.asarray(t)) / BLEND)


def lid_level(frame, rest, skin_triangles):
    """
    The level of the lids around an eye seen from the front: a function of x and y (eye's
    frame, eyeball radii) that gives the grey closing over :data:`LID_WINDOW` of the depth of
    the skin along the gaze, so that the valleys of the view (the pocket) take the level of
    the lids around them.
    """
    from scipy import ndimage

    from opensculptboy.eyes.geometry import _depth_raster

    local = frame.to_local(rest)
    near = (np.linalg.norm(local, axis=1) < 2.0) & (local[:, 2] > 0.0)
    tris = skin_triangles[near[skin_triangles].all(1)]
    step = LID_STEP
    xs = np.arange(-1.6, 1.6 + step / 2, step)
    ys = np.arange(-1.0, 1.0 + step / 2, step)
    depth = _depth_raster(local, tris, xs, ys)
    finite = np.isfinite(depth)
    _, (iy, ix) = ndimage.distance_transform_edt(~finite, return_indices=True)
    depth = depth[iy, ix]
    window = max(3, int(round(LID_WINDOW / step)))
    level = ndimage.grey_closing(depth, size=(window, window))

    def sample(x, y):
        i = np.clip(np.rint((y - ys[0]) / step).astype(int), 0, len(ys) - 1)
        j = np.clip(np.rint((x - xs[0]) / step).astype(int), 0, len(xs) - 1)
        return level[i, j]

    return sample


def shift(t):
    """The share of :data:`PLICA_SHIFT` at t across the band."""
    lo, hi = SHIFT_SPAN
    t = np.asarray(t)
    return _smoothstep(t / lo) * _smoothstep((1.0 - t) / (1.0 - hi))


def designed_point(p0, s, t, inner_point, lids):
    """
    The designed position (eye's frame, eyeball radii) of a vertex of the band at (s, t) whose
    anny position is ``p0``: ``inner_point`` is the inner ring's point at t and ``lids`` the
    level of the lids (:func:`lid_level`).
    """
    rho0 = float(np.hypot(p0[0], p0[1]))
    psi = float(np.arctan2(p0[1], p0[0]))
    direction = np.array([np.cos(psi), np.sin(psi)])
    moved = PLICA_SHIFT * float(shift(t))
    crest_rho = rho0 - moved
    shell = 1.0 + PLICA_GAP
    if s >= CREST:
        u = (s - CREST) / (1.0 - CREST)  # 0 at the crest, 1 at the lid margins
        rho = rho0 - moved * (1.0 - u)
        xy = rho * direction
        on_sphere = np.sqrt(max(shell**2 - rho**2, 0.0))
        top = float(lids(xy[0], xy[1])) - CARUNCLE_SINK
        rise = 1.0 - (1.0 - u) ** 3
        z = max(on_sphere + (top - on_sphere) * rise, on_sphere)
        return np.r_[xy, z]
    w = s / CREST  # 1 at the crest, 0 at the inner ring
    crest = np.r_[crest_rho * direction, np.sqrt(max(shell**2 - crest_rho**2, 0.0))]
    q = inner_point + (crest - inner_point) * np.sqrt(w)
    d = np.linalg.norm(q)
    if w < 0.9 and d > 1.0 - FOLD_GAP:
        q = q * (1.0 - FOLD_GAP) / d
    return q


def convex_weights(point, candidates, positions, k: int = MAX_ANCHORS, near=1e-3):
    """
    The convex combination of at most ``k`` of the ``candidates`` (rows of ``positions``)
    nearest ``point``: non-negative least squares with the sum held to 1 and a small penalty
    on far anchors (``near``). Returns the anchors, their weights and the distance of the
    combination from ``point``.
    """
    from scipy.optimize import nnls

    candidates = np.asarray(candidates)
    x = positions[candidates]
    d = np.linalg.norm(x - point, axis=1)

    def solve(rows):
        scale = d[rows].mean()
        a = np.vstack(
            [x[rows].T, 100.0 * np.ones(len(rows)), np.sqrt(near) * np.diag(d[rows])]
        )
        b = np.r_[point, 100.0, np.zeros(len(rows))]
        w, _ = nnls(a / scale, b / scale)
        return w

    w = solve(np.arange(len(candidates)))
    top = np.argsort(-w, kind="stable")[:k]
    top = np.sort(top[w[top] > 1e-9])
    w = solve(top)
    keep = w > 1e-12
    top, w = top[keep], w[keep] / w[keep].sum()
    return candidates[top], w, float(np.linalg.norm(w @ x[top] - point))


def _one_chart(base, faces, vertices) -> np.ndarray:
    """The vertices of ``vertices`` whose quads among ``faces`` give them one UV (away from
    the atlas's seams)."""
    seen: dict = {}
    for f in faces:
        for a, b in zip(base.quads[f].tolist(), base.uv_quads[f].tolist()):
            seen.setdefault(a, set()).add(b)
    return np.array(sorted(int(v) for v in vertices if len(seen.get(int(v), ())) == 1))


def one_eye(rest, quads, skin_mask, eyeball, side: str, base: BaseQuadMesh) -> dict:
    """The patch of one eye on anny's quads (``skin_mask``: the quads of the skin; ``eyeball``:
    the eyeball's rows): the removed quads, the new vertices (the anny vertex each replaces,
    anchors, weights, designed and combined positions, (s, t)), the new quads over anny
    indices (new vertex m as -1 - m), their tags and the quads of the anchors' UV chart."""
    from opensculptboy.detail.subdivision import split_quads
    from opensculptboy.eyes import geometry

    skin_quads = quads[skin_mask]
    frame = geometry.eye_frame(side, rest[eyeball])
    removed, levels = pocket_region(rest, quads, skin_quads, frame, side)
    region = quads[removed]
    kept = np.ones(len(quads), bool)
    kept[removed] = False
    band = np.unique(region)
    border = np.intersect1d(band, np.unique(quads[kept]))
    inside = np.setdiff1d(band, border)
    local = frame.to_local(rest)
    angle = _angle(local, side)
    # the border's sides: the kept quads along it inside the socket (s = 0), outside it on
    # the lids (s = 1), or at either end of the band (t = 0 on the upper lid, 1 on the lower)
    q_levels = levels[quads]
    middle = 0.5 * (SECTOR[0] + SECTOR[1])
    kinds: dict = {}
    for q in np.flatnonzero(kept & np.isin(quads, border).any(1)):
        finite = q_levels[q][np.isfinite(q_levels[q])]
        mean = finite.mean() if len(finite) else np.inf
        if mean < INNER_RING + 0.5:
            kind = "in"
        elif mean > OUTER_RING + 0.5:
            kind = "out"
        else:
            kind = "A" if _angle(local[quads[q]].mean(0), side) < middle else "B"
        for v in quads[q]:
            if v in border:
                kinds.setdefault(int(v), set()).add(kind)
    fixed_s = {v: 0.0 if "in" in k else 1.0 for v, k in kinds.items() if k & {"in", "out"}}
    fixed_t = {v: 0.0 if "A" in k else 1.0 for v, k in kinds.items() if k & {"A", "B"}}
    s_all = harmonic(band, region, fixed_s)
    t_all = harmonic(band, region, fixed_t)
    pos = {int(v): i for i, v in enumerate(band)}
    inner = np.array(sorted(v for v in fixed_s if fixed_s[v] == 0.0))
    i_t = np.array([t_all[pos[v]] for v in inner])
    i_order = np.argsort(i_t, kind="stable")

    # the candidates for anchors: the skin within ANCHOR_RINGS rings of quads of the band,
    # away from UV seams
    ring = band
    for _ in range(ANCHOR_RINGS):
        chart = np.flatnonzero(skin_mask & np.isin(quads, ring).any(1))
        ring = np.unique(quads[chart])
    candidates = _one_chart(base, chart, ring)

    lids = lid_level(frame, rest, split_quads(skin_quads))
    designed, combined, anchors, weights, residual, st = [], [], [], [], [], []
    for v in inside:
        s, t = s_all[pos[v]], t_all[pos[v]]
        inner_point = np.array(
            [np.interp(t, i_t[i_order], local[inner[i_order], k]) for k in range(3)]
        )
        target = designed_point(local[v], s, t, inner_point, lids)
        target = local[v] + blend(t) * (target - local[v])
        world = frame.to_world(target)
        a, w, r = convex_weights(world, candidates, rest)
        designed.append(world)
        combined.append(w @ rest[a])
        anchors.append(a)
        weights.append(w)
        residual.append(r)
        st.append((s, t))
    new_index = {int(v): -1 - m for m, v in enumerate(inside)}
    faces = np.array([[new_index.get(int(x), int(x)) for x in q] for q in region])
    st_of = {int(v): (s_all[pos[v]], t_all[pos[v]]) for v in band}
    face_s = np.array([np.mean([st_of[int(x)][0] for x in q]) for q in region])
    face_t = np.array([np.mean([st_of[int(x)][1] for x in q]) for q in region])
    tags = np.where(face_s >= CREST, "caruncle", "conjunctiva").astype(object)
    tags[(face_s >= SKIN_S) | ((blend(face_t) < SKIN_BLEND) & (face_s >= CREST))] = "skin"
    return dict(
        frame=frame,
        removed=removed,
        inside=inside,
        border=border,
        anchors=anchors,
        weights=weights,
        designed=np.array(designed),
        combined=np.array(combined),
        residual=np.array(residual),
        st=np.array(st),
        faces=faces,
        tags=tags,
        angle=angle,
        chart=chart[np.isin(quads[chart], candidates).all(1)],
    )


def eyeball_rows(base: BaseQuadMesh) -> dict[str, np.ndarray]:
    """The rows of each eyeball (``"L"``, ``"R"``) of the base mesh, by their MakeHuman
    base-mesh indices (:data:`opensculptboy.export.vrm_tables._EYE_RANGES`)."""
    from opensculptboy.export.vrm_tables import _EYE_RANGES, EYEBALL_VERTEX_COUNT

    index = np.asarray(base.base_mesh_vertex_indices)
    out = {}
    for side, (lo, hi) in _EYE_RANGES.items():
        rows = np.flatnonzero((index >= lo) & (index <= hi))
        if len(rows) != EYEBALL_VERTEX_COUNT:
            raise ValueError(f"The base mesh has no MakeHuman eyeball {side}.")
        out[side] = rows
    return out


def default_body(base: BaseQuadMesh) -> np.ndarray:
    """(N, 3) the rest vertices of anny's default character on the base mesh (the base holds
    anny's template, whose face differs from the default character's by up to 3 cm), where
    the design is fitted."""
    import anny
    import torch

    model = anny.Anny(
        topology="anny-quads", local_changes="none", facial_actions="none", face_shapes="none"
    ).to(dtype=torch.float64)
    if not np.array_equal(model.faces.cpu().numpy(), base.quads):
        raise ValueError("The default character's quads are not the base mesh's.")
    with torch.no_grad():
        return model()["rest_vertices"][0].cpu().numpy()


def left_eye(base: BaseQuadMesh) -> dict:
    """The patch of the left eye (x > 0) and what built it (:func:`one_eye`'s fields)."""
    vertices = default_body(base)
    quads = np.asarray(base.quads, dtype=np.int64)
    eyes = eyeball_rows(base)
    on_eye = np.zeros(len(vertices), bool)
    for rows in eyes.values():
        on_eye[rows] = True
    skin_mask = ~on_eye[quads].any(1)
    return one_eye(vertices, quads, skin_mask, eyes["L"], "L", base)


def build(base: BaseQuadMesh) -> Patch:
    """The eye-corner patch for both eyes: the left eye's, joined with its mirror image."""
    eye = left_eye(base)
    n = base.vertex_count
    removed = eye["removed"]
    m = len(eye["inside"])
    rows = [dict(zip(a.tolist(), w.tolist())) for a, w in zip(eye["anchors"], eye["weights"])]
    anchors, weights = blend_rows(rows)
    # the new faces keep the removed quads' UVs: the band's inner ring is a seam of the atlas,
    # whose UVs must stay on the border of the region, and the same combinations of the
    # anchors' UVs would fold the layout with the plica (flipped UV triangles)
    faces = eye["faces"]
    vertex_faces = np.where(faces < 0, n + (-1 - faces), faces)
    uv_faces = base.uv_quads[removed]
    uv_anchors, uv_weights = blend_rows([])
    labels = [NAME if t == "skin" else f"{NAME}/{t}" for t in eye["tags"]]
    left = Patch(
        name=NAME,
        removed_faces=removed,
        anchors=anchors,
        weights=weights,
        faces=vertex_faces,
        uv_faces=uv_faces,
        uv_anchors=uv_anchors,
        uv_weights=uv_weights,
        vertex_labels=[NAME] * m,
        face_labels=labels,
        options=constants(),
    )
    return left.with_mirror(base)


def constants() -> dict:
    """The constants the patch was built with, for the data file's record."""
    names = (
        "MAX_ANCHORS ANCHOR_RINGS RING_REACH RING_START INNER_RING OUTER_RING "
        "OUTER_RING_CANTHUS SECTOR CANTHUS CREST PLICA_GAP PLICA_SHIFT SHIFT_SPAN FOLD_GAP "
        "CARUNCLE_SINK LID_WINDOW LID_STEP BLEND SKIN_S SKIN_BLEND"
    ).split()
    g = globals()
    return {k.lower(): (list(g[k]) if isinstance(g[k], tuple) else g[k]) for k in names}

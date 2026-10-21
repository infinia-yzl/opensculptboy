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
    remeshed target in Anny's frame), ``eyes`` (the eyeballs by side, ``(vertices,
    triangles)``: spheres, not remeshed), ``landmarks`` (the contract's names, with 1 mm of noise)
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
    # the eyeballs, as separate parts of the handoff: level 2 of anny's eyeballs, not
    # remeshed, projected onto their least-squares spheres (the offset field bends them)
    from opensculptboy.export.vrm_tables import eyeball_vertices
    from opensculptboy.fit.reference import fit_sphere

    eyes = {}
    for side, rows in eyeball_vertices(model).items():
        mine = np.zeros(len(rest0))
        mine[rows] = 1.0
        on = np.asarray(detailed.topology.apply(mine)) > 0.99
        tris = detailed.triangles[on[detailed.triangles].all(1)]
        used = np.unique(tris)
        remap = np.full(len(on), -1)
        remap[used] = np.arange(len(used))
        centre, radius, _ = fit_sphere(surface[used])
        ray = surface[used] - centre
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        eyes[side] = (centre + radius * ray, remap[tris])

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
        eyes=eyes,
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


def _smoothed(values, triangles, iterations: int):
    """``values`` (N,) averaged with their neighbours ``iterations`` times."""
    t = np.asarray(triangles)
    edges = np.unique(
        np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), 1), axis=0
    )
    degree = np.maximum(np.bincount(edges.ravel(), minlength=len(values)), 1)
    out = np.asarray(values, dtype=np.float64).copy()
    for _ in range(iterations):
        total = np.bincount(edges[:, 0], out[edges[:, 1]], minlength=len(out))
        total += np.bincount(edges[:, 1], out[edges[:, 0]], minlength=len(out))
        out = 0.5 * out + 0.5 * total / degree
    return out


def point_ears(rest, ears, tip: float = 0.02, stand_off: float = 0.45):
    """
    Anny's rest vertices (N, 3) with pointed ears that stand off from the head. ``ears`` (N,)
    weighs each vertex: 1 on the outer ear (:func:`~opensculptboy.fit.register.pinna`),
    falling to 0 over the skin around its root. Each ear turns outward by ``stand_off`` radians
    about a vertical hinge at the front of its root, which opens a gap between the back of the
    ear and the head, and its upper half draws up and back into a tip, by ``tip`` metres at
    the top, narrowing as it goes. The weight carries both, so the skin around the root
    stretches smoothly, and the tip falls off faster toward the root, which stays in place.
    """
    V = np.asarray(rest, dtype=np.float64).copy()
    w = np.asarray(ears, dtype=np.float64)
    for s in (1.0, -1.0):
        side = (w > 0) & (np.sign(V[:, 0]) == s)
        core = side & (w > 0.9)
        if not core.any():
            continue
        C = V[core]
        hinge = np.array([s * np.abs(C[:, 0]).min(), C[:, 1].min(), 0.0])
        z_mid, z_top = np.median(C[:, 2]), C[:, 2].max()
        y_mid = np.median(C[:, 1])
        rows = np.nonzero(side)[0]
        x = V[rows]
        # outward about the hinge: the back of the ear (+Y) swings away from the head
        angle = -s * stand_off * w[rows]
        dx, dy = x[:, 0] - hinge[0], x[:, 1] - hinge[1]
        c, n = np.cos(angle), np.sin(angle)
        x[:, 0] = hinge[0] + c * dx - n * dy
        x[:, 1] = hinge[1] + n * dx + c * dy
        # the upper half draws into a tip, up and back, and narrows toward it; the root
        # stays on the head (w**4 falls faster than w over the skin around it)
        t = np.clip((x[:, 2] - z_mid) / (z_top - z_mid), 0.0, 1.0) * w[rows] ** 4
        x[:, 1] = y_mid + (x[:, 1] - y_mid) * (1.0 - 0.7 * t**2)
        direction = np.array([0.25 * s, 0.55, 1.0])
        direction /= np.linalg.norm(direction)
        x += tip * (t**2.5)[:, None] * direction
        V[rows] = x
    return V


def ear_outline(vertices, mask, rotation=None, translation=None) -> dict:
    """The ear outline landmarks of the handoff proposal (docs/reference-fit.md) from the
    vertices of an ear mask, in Anny's frame: highest, lowest, hindmost (+Y) and furthest from
    the midline, per side, for a figure placed by ``rotation`` and ``translation`` (whose own
    axes decide the sides and the directions)."""
    R = np.eye(3) if rotation is None else np.asarray(rotation)
    t = np.zeros(3) if translation is None else np.asarray(translation)
    V = np.asarray(vertices)
    own = (V - t) @ R  # the figure's own frame
    out = {}
    for side, s in (("l", 1.0), ("r", -1.0)):
        rows = np.nonzero(np.asarray(mask) & (np.sign(own[:, 0]) == s))[0]
        if len(rows) == 0:
            continue
        P = own[rows]
        for name, k in (
            ("top", np.argmax(P[:, 2])),
            ("lobe", np.argmin(P[:, 2])),
            ("back", np.argmax(P[:, 1])),
            ("lateral", np.argmax(np.abs(P[:, 0]))),
        ):
            out[f"ear_{name}_{side}"] = V[rows[k]]
    return out


def decimate(vertices, triangles, size, passes: int = 16, relax: int = 4):
    """
    A manifold decimation of a triangle mesh to an edge length of about ``size`` (per vertex,
    metres), as a game's mesh of another topology: each pass collapses a set of edges shorter
    than 0.7 of their size whose one-rings do not touch, to their midpoints moved onto the
    source surface, keeping each collapse that passes the link condition (no pinched
    neighbourhood) and turns no triangle around it by more than 60 degrees. Border edges and
    their vertices stay. ``relax`` rounds of tangential relaxation onto the source surface
    follow. Returns the vertices, the triangles and the new vertex of each old one.
    """
    import scipy.sparse

    from opensculptboy.fit.reference import Reference, vertex_normals

    source = Reference(vertices, triangles)
    V = np.asarray(vertices, dtype=np.float64).copy()
    T = np.asarray(triangles).copy()
    size = np.asarray(size, dtype=np.float64).copy()
    n = len(V)
    owner = np.arange(n)  # the vertex that each original vertex merged into

    def edges_of(T):
        e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
        e = np.sort(e, 1)
        u, counts = np.unique(e, axis=0, return_counts=True)
        return u, counts

    def normals_of(X, T):
        p = X[T]
        m = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-20)

    for _ in range(passes):
        E, counts = edges_of(T)
        border = np.zeros(n, bool)
        border[E[counts == 1].ravel()] = True
        a, b = E[:, 0], E[:, 1]
        length = np.linalg.norm(V[a] - V[b], axis=1)
        short = (counts == 2) & ~border[a] & ~border[b]
        short &= length < 0.7 * 0.5 * (size[a] + size[b])
        if not short.any():
            break
        A = scipy.sparse.coo_matrix(
            (np.ones(2 * len(E)), (np.r_[a, b], np.r_[b, a])), shape=(n, n)
        ).tocsr()
        A.data[:] = 1.0
        rows = np.nonzero(short)[0]
        common = np.asarray(A[a[rows]].multiply(A[b[rows]]).sum(1)).ravel()
        rows = rows[common == 2][np.argsort(length[rows[common == 2]])]
        locked = np.zeros(n, bool)
        chosen = []
        indptr, indices = A.indptr, A.indices
        for k in rows:
            i, j = a[k], b[k]
            if locked[i] or locked[j]:
                continue
            chosen.append(k)
            locked[indices[indptr[i] : indptr[i + 1]]] = True
            locked[indices[indptr[j] : indptr[j + 1]]] = True
            locked[i] = locked[j] = True
        chosen = np.asarray(chosen, np.int64)
        if len(chosen) == 0:
            break
        keep_i, gone_j = a[chosen], b[chosen]
        middle = 0.5 * (V[keep_i] + V[gone_j])
        middle, _, _, _, _ = source.closest(middle)
        collapse = np.full(n, -1)
        collapse[keep_i] = collapse[gone_j] = np.arange(len(chosen))
        touched = (collapse[T] >= 0).any(1)
        cid = collapse[T[touched]].max(1)
        remap = np.arange(n)
        remap[gone_j] = keep_i
        X = V.copy()
        X[keep_i] = middle
        old = normals_of(V, T[touched])
        newT = remap[T[touched]]
        alive = (newT[:, 0] != newT[:, 1]) & (newT[:, 1] != newT[:, 2])
        alive &= newT[:, 0] != newT[:, 2]
        new = normals_of(X, newT)
        bad = np.zeros(len(chosen), bool)
        flip = alive & ((old * new).sum(1) < 0.5)
        np.logical_or.at(bad, cid[flip], True)
        good = ~bad
        remap = np.arange(n)
        remap[gone_j[good]] = keep_i[good]
        V[keep_i[good]] = middle[good]
        size[keep_i[good]] = np.minimum(size[keep_i[good]], size[gone_j[good]])
        owner = remap[owner]
        T = remap[T]
        T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
    # tangential relaxation onto the source surface
    E, counts = edges_of(T)
    border = np.zeros(n, bool)
    border[E[counts == 1].ravel()] = True
    degree = np.maximum(np.bincount(E.ravel(), minlength=n), 1)
    for _ in range(relax):
        N = vertex_normals(V, T)
        mean = np.zeros_like(V)
        for k in range(3):
            mean[:, k] = np.bincount(E[:, 0], V[E[:, 1], k], minlength=n)
            mean[:, k] += np.bincount(E[:, 1], V[E[:, 0], k], minlength=n)
        step = mean / degree[:, None] - V
        step -= (step * N).sum(1, keepdims=True) * N
        moved = V + 0.5 * step
        used = np.unique(T)
        used = used[~border[used]]
        C, _, _, _, _ = source.closest(moved[used])
        trial = V.copy()
        trial[used] = C
        # keep the step where it turns no triangle over
        flipped = (normals_of(V, T) * normals_of(trial, T)).sum(1) < 0.5
        back = np.zeros(n, bool)
        back[T[flipped].ravel()] = True
        V = np.where(back[:, None], V, trial)
    used = np.unique(T)
    index = -np.ones(n, np.int64)
    index[used] = np.arange(len(used))
    return V[used], index[T], index[owner]


@functools.lru_cache(maxsize=2)
def pointed_ear_target(
    seed: int = 1,
    cell: float = 0.006,
    ear_cell: float = 0.002,
    mask_threshold: float = 0.5,
):
    """
    A synthetic reference with pointed ears that stand off from the head (:func:`point_ears`),
    as the first real replica has them: random anny parameters and a small pose, a broad
    random offset field of 1.5 mm RMS, the ears evaluated on level 2, posed, moved by a random
    placement and decimated (:func:`decimate`) to edges of about ``cell``, and ``ear_cell`` on
    and around the ears, into a manifold mesh of another topology in which the gap behind each
    ear stays open. Its ``masks`` hold the handoff's ``masks.ear``: the target
    vertices whose share of anny's outer ear is over ``mask_threshold`` (a different cut from
    the fit's own, as a real mask is) and the ``attachment``, those of them along the line
    where the ear meets the head. Its landmarks add the ear outline of the mask.
    """
    import anny
    import roma

    from opensculptboy.fit.reference import LANDMARKS
    from opensculptboy.fit.register import (
        ear_weights,
        excluded_vertices,
        mask_border,
        pinna,
    )

    rng = np.random.default_rng(seed)
    model = anny.Anny(local_changes="all", face_shapes="all", facial_actions="none").to(
        dtype=torch.float64
    )
    phenotype = {k: float(rng.uniform(0.35, 0.65)) for k in model.phenotype_labels}
    local = {
        k: float(rng.uniform(-0.3, 0.3))
        for k in rng.choice(model.local_change_labels, 8, replace=False)
        if "ear" not in k
    }
    face = {}
    for k in rng.choice(model.face_shape_labels, 8, replace=False):
        lo, hi = model.face_shape_ranges[k]
        face[str(k)] = float(np.clip(rng.uniform(-0.3, 0.3), lo, hi))
    labels = list(model.bone_labels)
    pose = torch.eye(4, dtype=torch.float64).repeat(1, len(labels), 1, 1)
    for bone, scale in (("upperarm01.L", 0.1), ("upperarm01.R", 0.1), ("neck02", 0.05)):
        rotvec = torch.as_tensor(rng.uniform(-scale, scale, 3))
        pose[0, labels.index(bone), :3, :3] = roma.rotvec_to_rotmat(rotvec)
    kwargs = dict(
        pose_parameters=pose,
        phenotype_kwargs=phenotype,
        local_changes_kwargs=local,
        face_shape_kwargs=face,
    )
    t0 = detail_topology(model, 0)
    two = detail_topology(model, 2)
    with torch.no_grad():
        rest0 = model(**kwargs)["rest_vertices"][0].numpy()
    broad = smooth_field(rest0, t0.triangles, rng, 30, 0.0015)
    outer = pinna(model, rest0).astype(np.float64)
    broad *= (1.0 - ear_weights(model))[:, None]  # the ears: point_ears alone
    rest2 = np.asarray(two.apply(rest0))
    ears2 = np.asarray(two.apply(outer))
    weight = _smoothed(ears2, two.triangles, 12)
    field2 = np.asarray(two.apply(broad)) + point_ears(rest2, weight) - rest2
    detailed = DetailModel(model, 2, offsets=field2)
    with torch.no_grad():
        posed = detailed(**kwargs)["vertices"][0].numpy()

    excluded = excluded_vertices(model)
    excluded2 = np.asarray(detailed.topology.apply(excluded.astype(float))) > 0.01
    triangles = detailed.triangles
    triangles = triangles[~excluded2[triangles].any(1)]

    angle = rng.uniform(-0.2, 0.2)
    rotation = roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, angle])).numpy()
    translation = rng.uniform(-0.3, 0.3, 3)
    surface = posed @ rotation.T + translation
    # finer on the ears and around them, as a game's head mesh has it
    sizes = np.where(_smoothed(ears2, two.triangles, 24) > 0.01, ear_cell, cell)
    vertices, faces, cluster = decimate(surface, triangles, sizes)
    kept = cluster >= 0
    share = np.bincount(cluster[kept], ears2[kept], minlength=len(vertices))
    share /= np.maximum(np.bincount(cluster[kept], minlength=len(vertices)), 1)
    ear = share > mask_threshold
    attachment = mask_border(faces, ear)

    from anny.keypoints import KeypointsRegressor

    regressor = KeypointsRegressor.craniofacial(model)
    with torch.no_grad():
        out = model(**kwargs)
    cranio = regressor(out)[0].numpy()
    joints = out["bone_poses"][0, :, :3, 3].numpy()
    landmarks = {}
    for name, (kind, label, _) in LANDMARKS.items():
        if name.startswith("ear_") and name not in ("ear_l", "ear_r"):
            continue  # the outline comes from the mask below
        point = (
            joints[labels.index(label)]
            if kind == "bone"
            else cranio[regressor.labels.index(label)]
        )
        landmarks[name] = rotation @ point + translation + rng.normal(0.0, 0.001, 3)
    landmarks.update(ear_outline(vertices, ear, rotation, translation))
    return dict(
        vertices=vertices,
        triangles=faces,
        landmarks=landmarks,
        ear=ear,
        attachment=attachment,
        truth=dict(
            phenotype=phenotype,
            local_changes=local,
            face_shapes=face,
            rotation=rotation,
            translation=translation,
        ),
    )


def stitched_ear_target(target: dict, grow: float = 0.004) -> dict:
    """
    The target of :func:`pointed_ear_target` written as a game writes its head: the ear mask
    grown over the skin around the root (the vertices within ``grow`` metres of it, reached
    along the mesh), as the real replica's ear piece holds its tragus and the flat front of its
    helix root, and the ear cut out as a piece of its own, stitched to the head by coincident
    vertices along the attachment line (the head keeps the line's vertices, the ear gets copies
    of them). The mask and the attachment are those of the ear piece.
    """
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree

    from opensculptboy.fit.register import mask_border

    V = np.asarray(target["vertices"], dtype=np.float64)
    T = np.asarray(target["triangles"])
    ear = np.asarray(target["ear"], bool).copy()
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V),) * 2).tocsr()
    A = A + A.T
    core = V[ear]
    while True:
        near = cKDTree(core).query(V)[0] < grow
        grown = ear | ((A @ ear.astype(np.float64) > 0) & near)
        if (grown == ear).all():
            break
        ear = grown
    on_ear = ear[T].all(1)
    line = mask_border(T, ear)
    line &= np.isin(np.arange(len(V)), T[on_ear]) & np.isin(
        np.arange(len(V)), T[~on_ear]
    )
    seam = np.nonzero(line)[0]
    copies = np.arange(len(V), len(V) + len(seam))
    remap = np.arange(len(V))
    remap[seam] = copies
    triangles = T.copy()
    triangles[on_ear] = remap[T[on_ear]]
    vertices = np.concatenate([V, V[seam]])
    mask = np.concatenate([ear, np.ones(len(seam), bool)])
    mask[seam] = False
    attachment = np.zeros(len(vertices), bool)
    attachment[copies] = True
    used = np.unique(triangles)
    keep = np.zeros(len(vertices), bool)
    keep[used] = True
    index = np.cumsum(keep) - 1
    return dict(
        target,
        vertices=vertices[keep],
        triangles=index[triangles],
        ear=mask[keep],
        attachment=attachment[keep],
    )

# OpenSculptBoy
# Apache License, Version 2.0
"""
A fitted character's mouth expressions from its reference's own mouth poses (docs/mouth.md,
"Mouth expressions from the reference").

A handoff may carry the reference's mouth poses (``mouth.poses`` of appearance.json: joint
deltas of its own rig). :func:`reference_pose_fields` poses the reference with each of them
and gives the motion of every point of its skin and of every piece of its inner mouth.
:func:`mouth_poses` carries that motion to a fitted card: each vertex of level 0 that lies on
the reference takes the motion at its closest point, and the inside of the mouth, the lips'
hidden roll and the rows along the reference's mouth opening (where the closest point falls
on either lip) take its harmonic extension, so that each lip's lining follows its own lip.
The motion is turned into the card's rest frame by the rigid motion of the fitted head, and
the pieces of the inner mouth take their own motion. The result, a :class:`MouthPoses`, is a
set of rest-frame offsets per pose: the exporters write each pose as a morph target
(``mouth.<pose>``) and the GLB export also splits it into a turn of the jaw and a corrective
(:func:`pose_turns`).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

POSES_FORMAT = "opensculptboy/mouth-poses@1"
# the morph targets of the poses are named "mouth.<pose>"
TARGET_PREFIX = "mouth."
# the reference's poses that the VRM presets play: the vowels of speech
VRM_VISEMES = {
    "aa": "viseme_a",
    "ih": "viseme_i",
    "ou": "viseme_u",
    "ee": "viseme_e",
    "oh": "viseme_o",
}
# a vertex of the fitted skin takes the reference's motion within this distance of it
ON_REFERENCE = 0.003
# ... unless its closest point lies within this distance of a border of the reference (the
# borders of a mouth opening, where the closest point can fall on either lip)
NEAR_BORDER = 0.0015
# the reference's skin that follows the jaw by this share or more counts to the lower lip
LOWER_SHARE = 0.35
# the poses are carried to the card's level FINE_LEVEL as well (when it has it) and level 0
# takes the offsets whose subdivision follows that level's motion best (:func:`_through_fine`),
# each row held to its own carried offset with the weight FINE_HOLD: level 0 has too few
# rows across the lips to follow the reference's roll of them, and the rounded mouth (the
# vowels O and U) opened too wide toward the corners
FINE_LEVEL = 1
FINE_HOLD = 1.0
# the lips' carried motion is smoothed over the lips and this many rings of faces around
# them, by this many Jacobi steps (:func:`smooth_lip_motion`)
LIP_SMOOTH_RINGS = 2
LIP_SMOOTH_ITERATIONS = 30
# the weight of the neighbours against the carried motion itself (a screened smoothing)
LIP_SMOOTH_WEIGHT = 4.0
# a lip's vertex whose carried motion stands out of its neighbours' mean by more than this
# (metres) is a rough spot
LIP_ROUGH = 0.0008
# with the reference's own skin transforms (:func:`reference_pose_transforms`), the weight of
# the neighbours in the smoothing of the lips' rough spots (the transforms leave less to smooth)
LIP_SMOOTH_WEIGHT_RIG = 1.0
# the lining off the skin within this many faces of a lip moves with that lip's transforms
LINING_RINGS = 3


def target_name(pose: str) -> str:
    return TARGET_PREFIX + pose


@dataclasses.dataclass
class MouthPoses:
    """The rest-frame offsets of a fitted card's mouth poses."""

    names: tuple  # the poses, in the handoff's order
    skin: np.ndarray  # (P, V0, 3) offsets of the vertices of anny's level 0
    inner: np.ndarray  # (P, M, 3) offsets of the points of the inner mouth (M may be 0)
    stats: dict = dataclasses.field(default_factory=dict)  # per pose: sizes, coverage

    def save(self, path) -> None:
        from safetensors.numpy import save_file

        save_file(
            {
                "skin": np.ascontiguousarray(self.skin, np.float32),
                "inner": np.ascontiguousarray(self.inner, np.float32),
            },
            str(path),
            metadata={
                "format": POSES_FORMAT,
                "names": json.dumps(list(self.names)),
                "stats": json.dumps(self.stats),
            },
        )

    @classmethod
    def load(cls, path) -> "MouthPoses":
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as f:
            meta = f.metadata() or {}
            if meta.get("format") != POSES_FORMAT:
                raise ValueError(f"{path} is not a {POSES_FORMAT} file.")
            return cls(
                names=tuple(json.loads(meta["names"])),
                skin=f.get_tensor("skin").astype(np.float64),
                inner=f.get_tensor("inner").astype(np.float64),
                stats=json.loads(meta.get("stats", "{}")),
            )

    def subset(self, names) -> "MouthPoses":
        keep = [self.names.index(n) for n in names if n in self.names]
        return MouthPoses(
            tuple(self.names[k] for k in keep),
            self.skin[keep],
            self.inner[keep],
            {self.names[k]: self.stats.get(self.names[k], {}) for k in keep},
        )


def reference_pose_fields(handoff, reference, poses: dict | None = None):
    """
    The motion of the reference under each of its mouth poses (``poses``, name to joint
    deltas; the handoff's known ``mouth.poses`` when None): a dict of name to (the offsets
    (V, 3) of ``reference.vertices``, its skin parts laid end to end, and a dict of handoff
    part index to the offsets of that part's vertices, for the pieces of the inner mouth).
    Poses that move nothing (the closed mouth) are left out.
    """
    from opensculptboy.fit.poses import mouth_poses, posed_parts

    poses = mouth_poses(handoff.appearance) if poses is None else poses
    skin = handoff.skin_parts()
    index = [next(k for k, p in enumerate(handoff.parts) if p is s) for s in skin]
    out = {}
    for name, deltas in poses.items():
        if not deltas:
            continue
        posed = posed_parts(handoff, deltas)
        field = []
        for k in index:
            part = handoff.parts[k]
            moved = posed[k]
            field.append(
                np.zeros_like(part.vertices) if moved is None else moved - part.vertices
            )
        pieces = {
            k: posed[k] - p.vertices
            for k, p in enumerate(handoff.parts)
            if getattr(p, "mouth_role", None) and posed[k] is not None
        }
        field = np.concatenate(field)
        if np.abs(field).max() < 1e-7:
            continue
        out[name] = (field, pieces)
    return out


def reference_pose_transforms(handoff, reference, poses: dict | None = None):
    """
    The reference's own skinning under each of its mouth poses (``poses``, name to joint
    deltas; the handoff's known ``mouth.poses`` when None): a dict of name to (R (V, 3, 3),
    t (V, 3)), the blended transform of each vertex of ``reference.vertices`` (its skin parts
    laid end to end) in Anny's frame, so that the posed vertex is ``R @ v + t``. It is the
    linear blend of the skin's joint matrices with the vertex's own weights, as a glTF viewer
    skins it, and it moves any point near the vertex as the reference's rig would. A vertex of
    a part that no skinned primitive matches keeps the identity. Empty when the file has no
    skinned primitive.
    """
    from opensculptboy.fit.handoff import GLTF_TO_ANNY
    from opensculptboy.fit.poses import GlbRig, _subset_rows
    from opensculptboy.fit.poses import mouth_poses as known_poses

    poses = known_poses(handoff.appearance) if poses is None else poses
    rig = GlbRig(handoff.path / "replica.glb")
    primitives = list(rig.skinned_primitives())
    if not primitives:
        return {}
    C = np.asarray(GLTF_TO_ANNY, np.float64)
    skin = handoff.skin_parts()
    rest_world = rig.world()
    rest = []
    for _, _, _, s, vertices, joints, weights in primitives:
        w = weights / np.maximum(weights.sum(1, keepdims=True), 1e-12)
        M = np.einsum("vk,vkij->vij", w, rig.skin_matrices(s, rest_world)[joints])
        rest.append((M, w, joints, s))
    rows = []
    for part in skin:
        found = None
        for k, (M, _, _, _) in enumerate(rest):
            vertices = primitives[k][4]
            at = (np.einsum("vij,vj->vi", M[:, :3, :3], vertices) + M[:, :3, 3]) @ C.T
            if (
                len(at) == len(part.vertices)
                and np.abs(at - part.vertices).max() < 1e-5
            ):
                found = (k, np.arange(len(at)))
            else:
                sub = _subset_rows(at, part.vertices, 1e-5)
                if sub is not None:
                    found = (k, sub)
            if found is not None:
                break
        rows.append((found, len(part.vertices)))
    out = {}
    for name, deltas in poses.items():
        posed_world = rig.world(deltas)
        blended = []
        for M, w, joints, s in rest:
            P = np.einsum("vk,vkij->vij", w, rig.skin_matrices(s, posed_world)[joints])
            blended.append(P @ np.linalg.inv(M))
        Rs, ts = [], []
        for found, n in rows:
            if found is None:
                Rs.append(np.repeat(np.eye(3)[None], n, 0))
                ts.append(np.zeros((n, 3)))
                continue
            A = blended[found[0]][found[1]]
            Rs.append(C @ A[:, :3, :3] @ C.T)
            ts.append(A[:, :3, 3] @ C.T)
        out[name] = (np.concatenate(Rs), np.concatenate(ts))
    return out


def lining_sides(
    triangles: np.ndarray, fitted_side: np.ndarray, off_skin: np.ndarray, rings: int
) -> np.ndarray:
    """
    The lip (1 upper, 2 lower) of the vertices off the skin (``off_skin``: the lining inside
    the mouth) within ``rings`` faces of either lip of ``fitted_side``, by the lip that
    reaches them first through the lining (the upper one on a tie); 0 elsewhere and on the
    lips themselves.
    """
    from scipy.sparse import coo_matrix

    T = np.asarray(triangles)
    n = len(fitted_side)
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    off = np.asarray(off_skin, bool)
    reached = {s: np.asarray(fitted_side) == s for s in (1, 2)}
    out = np.zeros(n, np.int8)
    for _ in range(max(int(rings), 0)):
        grown = {s: ((A @ reached[s].astype(np.float64)) > 0) & off for s in (1, 2)}
        for s in (1, 2):
            new = grown[s] & (out == 0) & ~reached[1] & ~reached[2]
            if s == 2:
                new &= ~grown[1]
            out[new] = s
        for s in (1, 2):
            reached[s] = reached[s] | (out == s)
    return out


def lip_sides(
    model,
    fitted: np.ndarray,
    rest: np.ndarray,
    triangles,
    reference,
    reference_share: np.ndarray | None = None,
):
    """
    The lip of each vertex of a fitted level 0 (``fitted``, as the fit placed it on the
    reference; ``rest`` and ``triangles`` its rest vertices and triangles) and of each vertex
    of the reference: 1 for the upper lip, 2 for the lower lip,
    0 elsewhere. Anny's lips are told apart by the jaw's opening (``fit.lips.lip_masks``);
    the reference's skin within the mouth's box by how far it follows the jaw in its own
    mouth poses (``reference_share``, :func:`~opensculptboy.mouth.weights.reference_jaw_share`
    on its vertices: lower from :data:`LOWER_SHARE`), else by the line where its lips end
    (``fit.lips.lip_line``: upper above the middle of that line, lower below it). Both zeros
    when the mouth's landmarks or the reference's line are missing.
    """
    from opensculptboy.fit import lips as L
    from opensculptboy.fit.metrics import mouth_zone
    from opensculptboy.fit.register import _Craniofacial, excluded_vertices, jaw_share

    fitted_side = np.zeros(len(fitted), np.int8)
    ref_side = np.zeros(len(reference.vertices), np.int8)
    cranio = _Craniofacial(model)
    points = cranio.weights_matrix.detach().double().cpu().numpy() @ fitted
    named = dict(zip(cranio.labels, points))
    if not all(k in named for k in ("ch.L", "ch.R", "ls", "li")):
        return fitted_side, ref_side
    box = L.MouthBox.from_landmarks(named)
    line = L.lip_line(reference.vertices, reference.triangles, box)
    if line is None:
        return fitted_side, ref_side
    roll = mouth_zone(model, rest) & ~excluded_vertices(model)
    T = np.asarray(triangles)
    edges = np.concatenate([T[:, [0, 1]], T[:, [1, 2]]])
    edges = edges[roll[edges].all(1)]
    spacing = float(
        np.median(np.linalg.norm(fitted[edges[:, 0]] - fitted[edges[:, 1]], axis=1))
    )
    upper, lower = L.lip_masks(fitted, roll, jaw_share(model), box, spacing)
    fitted_side[upper] = 1
    fitted_side[lower] = 2
    x0, x1 = box.x_range
    margin = 0.25 * box.width
    mid = 0.5 * (line.upper + line.lower)
    order = np.argsort(mid[:, 0])
    v = np.asarray(reference.vertices)
    zmid = np.interp(v[:, 0], mid[order, 0], mid[order, 2])
    lo, hi = box.z_range
    height = hi - lo
    inside = (
        (v[:, 0] > x0 - margin)
        & (v[:, 0] < x1 + margin)
        & (v[:, 2] > lo - height)
        & (v[:, 2] < hi + height)
    )
    if reference_share is not None:
        lower_lip = np.asarray(reference_share) >= LOWER_SHARE
    else:
        lower_lip = v[:, 2] < zmid
    ref_side[inside & ~lower_lip] = 1
    ref_side[inside & lower_lip] = 2
    return fitted_side, ref_side


def _closest_on_triangles(p, a, b, c):
    """The closest points (N, 3) of points p on triangles (a, b, c) (each (N, 3)) and their
    barycentric coordinates (N, 3) (Ericson, Real-Time Collision Detection, 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        bary = np.stack([1 - v - w, v, w], 1)
        # the regions of the vertices and the edges
        cases = [
            ((d1 <= 0) & (d2 <= 0), np.stack([np.ones_like(d1), 0 * d1, 0 * d1], 1)),
            ((d3 >= 0) & (d4 <= d3), np.stack([0 * d1, np.ones_like(d1), 0 * d1], 1)),
            ((d6 >= 0) & (d5 <= d6), np.stack([0 * d1, 0 * d1, np.ones_like(d1)], 1)),
        ]
        t = d1 / (d1 - d3)
        cases.append(
            ((vc <= 0) & (d1 >= 0) & (d3 <= 0), np.stack([1 - t, t, 0 * t], 1))
        )
        t = d2 / (d2 - d6)
        cases.append(
            ((vb <= 0) & (d2 >= 0) & (d6 <= 0), np.stack([1 - t, 0 * t, t], 1))
        )
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        cases.append(
            (
                (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
                np.stack([0 * t, 1 - t, t], 1),
            )
        )
    done = np.zeros(len(p), bool)
    out = bary.copy()
    for mask, value in cases:
        mask = mask & ~done
        out[mask] = value[mask]
        done |= mask
    out = np.nan_to_num(out)
    points = out[:, :1] * a + out[:, 1:2] * b + out[:, 2:] * c
    return points, out


def _sided_closest(points, reference, ref_side, side, k: int = 12):
    """The closest points of ``points`` on the reference's triangles whose three vertices
    lie on lip ``side``: (distance (N,), triangle (N,) into reference.triangles, barycentric
    (N, 3)), or None when there are none. The candidates are the triangles of the ``k``
    nearest vertices of that side."""
    from scipy.spatial import cKDTree

    V = np.asarray(reference.vertices)
    T = np.asarray(reference.triangles)
    keep = np.nonzero((ref_side[T] == side).all(1))[0]
    if len(keep) == 0:
        return None
    used = np.unique(T[keep])
    corner_of = [[] for _ in range(len(V))]
    for t in keep:
        for v in T[t]:
            corner_of[v].append(t)
    k = min(k, len(used))
    _, near = cKDTree(V[used]).query(points, k=k)
    near = used[np.atleast_2d(near).reshape(len(points), -1)]
    best = np.full(len(points), np.inf)
    tri = np.zeros(len(points), np.int64)
    bary = np.zeros((len(points), 3))
    candidates = [sorted({t for v in row for t in corner_of[v]}) for row in near]
    width = max(len(c) for c in candidates)
    for j in range(width):
        rows = np.array([i for i, c in enumerate(candidates) if j < len(c)])
        ts = np.array([candidates[i][j] for i in rows])
        q, b = _closest_on_triangles(
            points[rows], V[T[ts, 0]], V[T[ts, 1]], V[T[ts, 2]]
        )
        d = np.linalg.norm(q - points[rows], axis=1)
        better = d < best[rows]
        best[rows[better]] = d[better]
        tri[rows[better]] = ts[better]
        bary[rows[better]] = b[better]
    return best, tri, bary


def smooth_lip_motion(
    offsets: np.ndarray,
    triangles: np.ndarray,
    side: np.ndarray,
    rings: int = LIP_SMOOTH_RINGS,
    iterations: int = LIP_SMOOTH_ITERATIONS,
    known: np.ndarray | None = None,
    weight: float | None = None,
) -> np.ndarray:
    """
    A pose's offsets (V, 3) of level 0 with the lips' motion smoothed: the vertices of
    either lip (``side`` 1 or 2) and the skin within ``rings`` faces of them are solved for
    the motion closest to the carried one that is smooth over their neighbours, by
    ``iterations`` Jacobi steps of (I + w L) o = offsets (w ``weight``, by default
    :data:`LIP_SMOOTH_WEIGHT`). The carried motion holds the
    roughness of the reference's own triangles, which the detail levels fold into creases
    and torn spots on the lips. Edges between the upper and the lower lip are left out, so
    that each lip keeps its own motion, except within one face of the mouth's corners (where
    the lips' vertices meet), whose junction takes the mean of both. With ``known``, only
    the vertices that took the reference's motion directly are smoothed, among themselves:
    the lining's extension, filled in afterwards, would pull the lips' edges toward it.
    """
    from opensculptboy.fit.lips import _adjacency, rings_from

    side = np.asarray(side)
    lips = np.nonzero(side > 0)[0]
    if len(lips) == 0 or iterations <= 0:
        return offsets
    n = len(offsets)
    A = _adjacency(n, triangles).tocoo()
    i, j = A.row, A.col
    across = ((side[i] == 1) & (side[j] == 2)) | ((side[i] == 2) & (side[j] == 1))
    corners = (
        rings_from(triangles, np.unique(i[across]), 1, n) if across.any() else None
    )
    keep = ~across
    if corners is not None:
        keep |= corners[i] & corners[j]
    if known is not None:  # only the motion carried from the reference itself
        known = np.asarray(known, bool)
        keep &= known[i] & known[j]
    from scipy.sparse import coo_matrix

    W = coo_matrix((np.ones(int(keep.sum())), (i[keep], j[keep])), shape=(n, n)).tocsr()
    degree = np.asarray(W.sum(1)).ravel()
    region = rings_from(triangles, lips, rings, n) & (degree > 0)
    if known is not None:
        region &= known
    data = np.asarray(offsets, dtype=np.float64)
    out = data.copy()
    # only the rough spots: the vertices that stand out of their neighbours' mean by more
    # than LIP_ROUGH, and one ring around them, take the mean of their neighbours (a real
    # ramp of the motion, as where a lip's edge moves more than the skin outside it, stands
    # out by little and keeps its carried motion)
    safe = np.maximum(degree, 1.0)[:, None]
    rough = np.linalg.norm(data - (W @ data) / safe, axis=1) > LIP_ROUGH
    rough &= region
    if not rough.any():
        return out
    W1 = (W + W.T).astype(bool).astype(np.float64)
    near = rough | ((W1 @ rough.astype(np.float64)) > 0)
    rows = np.nonzero(near & region)[0]
    w = LIP_SMOOTH_WEIGHT if weight is None else float(weight)
    for _ in range(iterations):
        out[rows] = (data[rows] + w * (W @ out)[rows]) / (1.0 + w * degree[rows, None])
    return out


def carry_field(
    fitted: np.ndarray,
    triangles: np.ndarray,
    reference,
    field: np.ndarray,
    on_skin: np.ndarray,
    sides=None,
    jaw: np.ndarray | None = None,
    transforms=None,
    lining: int = LINING_RINGS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    A vector field (V, C) of a fitted level (``fitted``, as the fit placed it on the
    reference) from one (R, C) on the reference's vertices, and the mask of the vertices that
    took it directly: the vertices of ``on_skin`` within :data:`ON_REFERENCE` of the
    reference take the value at their closest point, interpolated over its triangle, unless
    that point lies within :data:`NEAR_BORDER` of one of the reference's borders; the others
    take the harmonic extension of those (``register.smooth_mouth``); with each vertex's
    share of the jaw (``jaw``), the extension from the vertices of its own jaw, mixed by that
    share. With the reference's own skin transforms of the pose (``transforms``, an (R, t)
    of :func:`reference_pose_transforms`) and ``sides``, each lip's vertices, and the lining
    off the skin within ``lining`` faces of it (:func:`lining_sides`), move by the transform
    of the closest point on the reference's same lip, applied to their own positions (the
    reference's rig carried to them, as a linear blend of its skin weights): a lip's inner
    rows and lining, which lie off the reference's surface, roll with that lip as the
    reference's do, and the rough spots are then smoothed with
    :data:`LIP_SMOOTH_WEIGHT_RIG`.
    """
    from scipy.spatial import cKDTree

    from opensculptboy.fit.register import smooth_mouth
    from opensculptboy.mouth.weights import reference_border

    rows = np.nonzero(np.asarray(on_skin, bool))[0]
    closest, _, d, tri, bary = reference.closest(fitted[rows])[:5]
    values = np.einsum("nk,nkc->nc", bary, field[reference.triangles[tri]])
    near = d < ON_REFERENCE
    edge = reference_border(reference)
    if len(edge):
        near &= cKDTree(edge).query(closest)[0] > NEAR_BORDER
    out = np.zeros((len(fitted), field.shape[1]))
    out[rows[near]] = values[near]
    known = np.zeros(len(fitted), bool)
    known[rows[near]] = True
    if sides is not None:  # each lip takes the motion of the reference's same lip
        fitted_side, ref_side = sides
        for side in (1, 2):
            mine = np.nonzero(fitted_side == side)[0]
            if len(mine) == 0:
                continue
            found = _sided_closest(fitted[mine], reference, ref_side, side)
            if found is None:
                continue
            distance, tri, bary = found
            if transforms is not None:
                out[mine] = _transformed(fitted[mine], reference, transforms, tri, bary)
                known[mine] = True
                continue
            ok = distance < ON_REFERENCE
            vals = np.einsum("nk,nkc->nc", bary, field[reference.triangles[tri]])
            out[mine[ok]] = vals[ok]
            known[mine[ok]] = True
            known[mine[~ok]] = False
            out[mine[~ok]] = 0.0
        weight = None
        if transforms is not None:
            weight = LIP_SMOOTH_WEIGHT_RIG
            sides_in = lining_sides(
                triangles, fitted_side, ~np.asarray(on_skin, bool), lining
            )
            for side in (1, 2):
                mine = np.nonzero(sides_in == side)[0]
                if len(mine) == 0:
                    continue
                found = _sided_closest(fitted[mine], reference, ref_side, side)
                if found is None:
                    continue
                _, tri, bary = found
                out[mine] = _transformed(fitted[mine], reference, transforms, tri, bary)
                known[mine] = True
        out = smooth_lip_motion(out, triangles, fitted_side, known=known, weight=weight)
    if jaw is None:
        return smooth_mouth(out, triangles, ~known), known
    # The lining inside the mouth: the extension from the lips of its own jaw (the lower lip
    # and the skin that follows the jaw, and the rest), mixed by each row's share of the jaw,
    # so that the lower lip's lining rolls with the lower lip.
    jaw = np.clip(np.asarray(jaw, dtype=np.float64), 0.0, 1.0)
    on_jaw = jaw >= 0.5
    low = smooth_mouth(out, triangles, ~(known & on_jaw))
    high = smooth_mouth(out, triangles, ~(known & ~on_jaw))
    filled = out.copy()
    rows = ~known
    filled[rows] = jaw[rows, None] * low[rows] + (1.0 - jaw[rows, None]) * high[rows]
    return filled, known


def _operator_matrix(topology):
    """The subdivision operators of a detail topology as one sparse matrix (N, V0)."""
    from scipy.sparse import csr_matrix

    M = None
    for op in topology.operators:
        S = csr_matrix((op.values, (op.rows, op.cols)), shape=op.shape)
        M = S if M is None else S @ M
    return M


def _fine_setup(fine, on_skin, sides, jaw):
    """The finer level's data for :func:`_through_fine`: (fitted, triangles, matrix, its
    skin mask, lip sides, jaw share)."""
    topology, fitted = fine
    S = _operator_matrix(topology)
    fitted_side, ref_side = sides
    up = S @ (np.asarray(fitted_side) == 1).astype(np.float64)
    lo = S @ (np.asarray(fitted_side) == 2).astype(np.float64)
    side = np.where(up > 0.5, 1, np.where(lo > 0.5, 2, 0)).astype(np.int8)
    skin = (S @ np.asarray(on_skin, np.float64)) > 0.5
    share = None if jaw is None else S @ np.asarray(jaw, np.float64)
    return (
        np.asarray(fitted),
        np.asarray(topology.triangles),
        S,
        skin,
        (side, ref_side),
        share,
    )


def _through_fine(values0, data, reference, field, transforms) -> np.ndarray:
    """A pose's offsets of level 0 whose subdivision best reproduces the reference's motion
    carried to a finer level (:func:`carry_field` on its vertices): least squares over the
    level-0 rows that the mouth's motion reaches, held to their own carried offsets by
    :data:`FINE_HOLD`. Level 0 alone has too few rows across the lips to follow the
    reference's roll of them."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    fitted, triangles, S, skin, sides, share = data
    values1, _ = carry_field(
        fitted,
        triangles,
        reference,
        field,
        skin,
        sides,
        share,
        transforms,
        lining=LINING_RINGS * 2,
    )
    moving = np.abs(values1).max(1) > 1e-6
    rows0 = np.unique(S[moving].indices)
    if len(rows0) == 0:
        return values0
    Sc = S.tocsc()
    keep = np.ones(S.shape[1], bool)
    keep[rows0] = False
    A = Sc[:, rows0]
    rhs = values1 - Sc[:, keep] @ values0[keep]
    lam = FINE_HOLD
    N = (A.T @ A + lam * diags(np.ones(len(rows0)))).tocsc()
    b = A.T @ rhs + lam * values0[rows0]
    out = np.asarray(values0, np.float64).copy()
    out[rows0] = np.asarray(spsolve(N, b)).reshape(len(rows0), -1)
    return out


def _transformed(points, reference, transforms, tri, bary) -> np.ndarray:
    """The offsets (N, 3) of ``points`` under the reference's skin transforms blended at
    their closest points (triangles ``tri`` with barycentric ``bary``)."""
    R, t = transforms
    corners = np.asarray(reference.triangles)[tri]
    Rb = np.einsum("nk,nkij->nij", bary, R[corners])
    tb = np.einsum("nk,nki->ni", bary, t[corners])
    return np.einsum("nij,nj->ni", Rb, points) + tb - points


def mouth_poses(
    handoff,
    reference,
    fitted: np.ndarray,
    triangles: np.ndarray,
    rest: np.ndarray,
    on_skin: np.ndarray,
    head: np.ndarray,
    mouth=None,
    poses: dict | None = None,
    sides=None,
    jaw: np.ndarray | None = None,
    fine=None,
) -> MouthPoses | None:
    """
    The :class:`MouthPoses` of a fitted level 0 (``fitted`` as the fit placed it on the
    reference, ``rest`` its rest vertices, ``head`` the rows of its head that turn the
    reference's frame into the rest frame, ``on_skin`` the vertices that can lie on the
    reference) from the reference's mouth poses, with the offsets of the points of the card's
    inner mouth (``mouth``, a :class:`~opensculptboy.mouth.inner.MouthData` whose points are
    the handoff's pieces in order); ``jaw`` is each vertex's share of the jaw
    (``register.jaw_share``), for the lining inside the mouth (:func:`carry_field`). None
    when the handoff has no moving pose.
    """
    from opensculptboy.teeth.mesh import rigid

    fields = reference_pose_fields(handoff, reference, poses)
    if not fields:
        return None
    transforms = {}
    if sides is not None:
        try:
            transforms = reference_pose_transforms(
                handoff, reference, {n: _pose_deltas(handoff, poses, n) for n in fields}
            )
        except (OSError, ValueError, KeyError):
            transforms = {}
    R, _ = rigid(np.asarray(fitted)[head], np.asarray(rest)[head])
    pieces = []
    if mouth is not None:
        from opensculptboy.mouth.handoff import piece_part

        pieces = [
            k
            for k, p in enumerate(handoff.parts)
            if getattr(p, "mouth_role", None)
            and len(p.triangles)
            and piece_part(p.mouth_role) is not None
        ]
    M = 0 if mouth is None else len(mouth.points)
    fine_data = None
    if fine is not None and sides is not None:
        fine_data = _fine_setup(fine, on_skin, sides, jaw)
    names, skin, inner, stats = [], [], [], {}
    for name, (field, moved) in fields.items():
        values, known = carry_field(
            fitted,
            triangles,
            reference,
            field,
            on_skin,
            sides,
            jaw,
            transforms.get(name),
        )
        if fine_data is not None:
            values = _through_fine(
                values, fine_data, reference, field, transforms.get(name)
            )
        offsets = values @ R.T
        piece = np.zeros((M, 3))
        if M:
            parts = [moved.get(k) for k in pieces]
            sizes = [len(handoff.parts[k].vertices) for k in pieces]
            if sum(sizes) == M:
                piece = (
                    np.concatenate(
                        [
                            np.zeros((n, 3)) if d is None else d
                            for d, n in zip(parts, sizes)
                        ]
                    )
                    @ R.T
                )
        size = np.linalg.norm(offsets, axis=1) * 1000
        names.append(name)
        skin.append(offsets)
        inner.append(piece)
        stats[name] = {
            "moved_vertices": int((size > 0.05).sum()),
            "max_mm": round(float(size.max()), 3),
            "inner_max_mm": round(
                float(np.linalg.norm(piece, axis=1).max() * 1000) if M else 0.0, 3
            ),
        }
    return MouthPoses(
        tuple(names),
        np.stack(skin),
        np.stack(inner) if M else np.zeros((len(names), 0, 3)),
        stats,
    )


def _pose_deltas(handoff, poses, name):
    from opensculptboy.fit.poses import mouth_poses as known_poses

    poses = known_poses(handoff.appearance) if poses is None else poses
    return poses[name]


def write_card_poses(card_path, handoff, reference=None) -> pathlib.Path | None:
    """
    Compute the mouth poses of a fitted card (its level 0, from its offsets and placement)
    from its handoff and write them as ``<card>.mouth_poses.safetensors``, named under
    ``material.mouth.poses`` of the card, which is written again. Returns the path, or None
    (and changes nothing) when the card has no inner mouth or the handoff no moving pose.
    """
    import torch

    from opensculptboy.character import Character
    from opensculptboy.fit.reference import Reference
    from opensculptboy.fit.register import excluded_vertices, jaw_share
    from opensculptboy.mouth.inner import MouthData, fitted_meshes
    from opensculptboy.teeth.mesh import mouth_rows

    card_path = pathlib.Path(card_path)
    character = Character.from_dict(json.loads(card_path.read_text()))
    material = dict(character.material or {})
    if "mouth" not in material:
        return None
    reference = reference or Reference.from_handoff(handoff)
    model, topology, posed, rest = fitted_meshes(card_path, reference, 0)
    head, _ = mouth_rows(model, topology)
    mouth = MouthData.load(card_path.parent / material["mouth"]["mesh"])
    with torch.no_grad():
        on_skin = ~excluded_vertices(model)
    fine = None
    if FINE_LEVEL and FINE_LEVEL in character.detail_levels():
        _, fine_topology, fine_fitted, _ = fitted_meshes(
            card_path, reference, FINE_LEVEL
        )
        fine = (fine_topology, fine_fitted)
    result = mouth_poses(
        handoff,
        reference,
        posed,
        np.asarray(topology.triangles),
        rest,
        on_skin,
        head,
        mouth,
        sides=lip_sides(
            model,
            posed,
            rest,
            topology.triangles,
            reference,
            _share(handoff, reference),
        ),
        jaw=np.asarray(jaw_share(model)),
        fine=fine,
    )
    if result is None:
        return None
    stem = (
        card_path.name[: -len(card_path.suffix)] if card_path.suffix else card_path.name
    )
    path = card_path.parent / f"{stem}.mouth_poses.safetensors"
    result.save(path)
    material["mouth"] = dict(material["mouth"], poses=path.name)
    character.material = material
    card_path.write_text(json.dumps(character.to_dict(), indent=2) + "\n")
    return path


def _share(handoff, reference) -> np.ndarray | None:
    """The jaw share of the reference's vertices from its mouth poses, or None."""
    from opensculptboy.mouth.weights import reference_jaw_share, reference_share_field

    shares = reference_jaw_share(handoff)
    if not shares:
        return None
    return reference_share_field(reference, handoff, shares)


def card_poses(character, detail_dir=None) -> MouthPoses | None:
    """The :class:`MouthPoses` that a card's ``material.mouth.poses`` names, or None."""
    mouth = (character.material or {}).get("mouth") or {}
    name = mouth.get("poses")
    if not name:
        return None
    return MouthPoses.load(pathlib.Path(detail_dir or ".") / name)


def append_pose_rows(targets, rest_vertices, poses: MouthPoses, names=None):
    """
    The morph targets and rest rows of an export (``export.body``'s ``morph_target_rows``
    and ``rest_rows``) with one target ``mouth.<pose>`` per pose (``names``: all by default)
    appended, its row the base mesh plus the pose's offsets. Returns (targets, rows, the
    pose of each appended target by its index in the targets).
    """
    from opensculptboy.export.body import MorphTarget

    poses = poses if names is None else poses.subset(names)
    base = rest_vertices[0]
    added = {}
    rows = [rest_vertices]
    targets = list(targets)
    for k, name in enumerate(poses.names):
        added[len(targets)] = k
        targets.append(MorphTarget(target_name(name), "mouth_pose", k, 1.0, 0.0))
        rows.append((base + poses.skin[k])[None])
    return targets, np.concatenate(rows), added, poses


def pose_turns(model, base, poses: MouthPoses, bone_pose, max_influences=4):
    """
    Per pose, the :class:`~opensculptboy.mouth.jaw.JawOpen` split of its offsets into a turn
    of the jaw (fitted with the model's jaw weights) and the corrective that completes it for
    the file's kept weights (``max_influences``), as for anny's jawOpen.
    """
    from opensculptboy.mouth.jaw import corrective_for, jaw_open, jaw_weights

    full = jaw_weights(model)
    kept = jaw_weights(model, max_influences)
    out = []
    for k in range(len(poses.names)):
        turn = jaw_open(base, poses.skin[k], full, bone_pose)
        turn.corrective = corrective_for(
            base, poses.skin[k], kept, turn.rotation, turn.translation, turn.pivot
        )
        out.append(turn)
    return out


def inner_corrective(points, weights, offsets, turn) -> np.ndarray:
    """The corrective offsets (M, 3) of the inner mouth's points for a pose's ``turn``: what
    completes the jaw's turn of the points, skinned with their jaw weights (M,), to the
    pose's offsets."""
    from opensculptboy.mouth.jaw import corrective_for

    return corrective_for(
        points, offsets, weights, turn.rotation, turn.translation, turn.pivot
    )


def resolve_overrides(mouth, overrides: dict, poses: MouthPoses, turns) -> dict:
    """The offsets (M, 3) of the inner mouth's points by target row: a pose's own offsets, or
    for an entry ``("corrective", k)`` the corrective of pose ``k``'s turn
    (:func:`inner_corrective`), with the points' weights of the jaw and the tongue."""
    from opensculptboy.mouth.rig import MOUTH_BONES

    out = {}
    weights = None
    for row, value in overrides.items():
        if isinstance(value, tuple):
            if weights is None:
                bones = list(getattr(mouth, "bones", ()))
                mouth_slots = [i for i, b in enumerate(bones) if b in MOUTH_BONES]
                slots = np.asarray(mouth.bone_slots)
                w = np.asarray(mouth.bone_weights, np.float64)
                weights = np.where(np.isin(slots, mouth_slots), w, 0.0).sum(1)
            k = value[1]
            out[row] = inner_corrective(mouth.points, weights, poses.inner[k], turns[k])
        else:
            out[row] = value
    return out

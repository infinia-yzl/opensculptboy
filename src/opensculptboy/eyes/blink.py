# OpenSculptBoy
# Apache License, Version 2.0
"""
Closing the eyes: the blink targets of the exported eyes (docs/eyes.md, "Blinks").

Anny closes each eye with one facial action (``eyeBlinkLeft``, ``eyeBlinkRight``), a fixed
offset per vertex. Played as a morph target, every vertex moves along a straight line, so the
lids cut into the round eyeball half way, and the upper lid closes a little past the lower lid.
A game character closes its eyes by turning lid joints, with the lashes skinned to the same
joints, so that the lids and the lashes move as one piece around the eye.

:func:`blink_offsets` sets the blink target of each eye in the same way:

- a card with a reference's blink (:mod:`opensculptboy.eyes.poses`) takes the reference's
  closed pose and half-closed pose as they come, the lids carried from the reference's lid
  joints (:func:`reference_blink`), and the reference's lashes keep the turn of those joints
  about their bases while each base goes with the card's lid margin (:func:`on_margin`);
- both are brought to meet: where the closed upper rim stays above the closed lower rim, or
  passes it, the upper lid turns about the eye's horizontal axis until its rim lies just past
  the lower rim (:func:`meet_rims`);
- any other card is built the same way from anny's own closed pose (:func:`rig_blink`), and
  its lashes go with their roots and turn by :data:`LASH_TURN` of their lid's turn, as the
  reference's lashes do (:func:`rig_lashes`);
- the slivers that a fitted mesh holds folded over at rest move as one piece
  (:func:`steady_folds`), so that a blink does not turn their corners' normals over;
- the in-between: the half-closed pose on the path around the eye, which GLB files carry as a
  corrective target (:data:`INBETWEEN_SUFFIX`) that an animation raises from 0 at the ends to
  1 half way; VRM files play the morph target alone, so there the closed pose keeps the
  straight line outside the eye (:func:`clear_line`).

The measures (:func:`measure_blink`) give, at each stage of a blink, how far the lids cross
each other (:func:`lid_crossing`), how deep lid skin lies inside the eye
(:func:`eye_penetration`), how deep lash points lie inside the lids (:func:`lash_penetration`)
and how far the upper lashes of a closed eye lie under the lower ones (:func:`lash_order`).
"""

from __future__ import annotations

import dataclasses

import numpy as np

# the stages of a blink that the measures look at
STAGES = (0.0, 0.25, 0.5, 0.75, 1.0)
# the lids stay this far (metres) outside the eye's outer surface along a straight line
LID_CLEARANCE = 0.0001
# the straight line of a morph target is checked at these shares of the way to the closed pose
LINE_CHECKS = (0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)
# the largest push (metres) out of the eye that the straight line gives a lid vertex
MAX_PUSH = 0.004
# the lid vertices that a blink moves: within this many eyeball radii of the eye's centre,
# in front of the plane this far behind it (eyeball radii)
LID_REACH = 1.6
LID_FRONT = -0.35
# the upper lid's rows take the turn that brings the rims together in full on the rim, and
# none from this distance (metres) of it; the turn is smoothed along the rim this many times
MEET_FALLOFF = 0.006
MEET_SMOOTH = 12
# the closed upper rim lies this far (metres) past the closed lower rim, in at most this many
# rounds (:func:`meet_rims`)
RIM_OVERLAP = 0.0002
MEET_ROUNDS = 4
# the turn that brings the rims together stays within this angle (degrees)
MAX_RIM_TURN = 20.0
# the turn that brings the rims together fades out over this share of the opening's length
# at each corner of the eye
MEET_TAPER = 0.1
# the smoothed turn can leave the rims apart between the upper rim's vertices, where a coarse
# lid holds few of them (10 to 15 on level 0): in at most SEAL_ROUNDS more rounds, the upper
# rim's vertices turn down by the least turns that close the gaps left (sampled SEAL_SAMPLES
# times along the opening), until the rims overlap by SEAL_OVERLAP (metres) everywhere from
# CORNER_MARGIN of each corner
SEAL_ROUNDS = 6
SEAL_SAMPLES = 256
SEAL_OVERLAP = 0.00002
# the lashes turn about the eye's horizontal axis by this share of their lid's turn: the
# upper lashes of a reference character turn 15 degrees while the upper rim turns about 41
LASH_TURN = 0.37
# a lash point may lie this far (metres) inside the lids beyond its depth at rest before it
# counts in the measures
LASH_TOLERANCE = 0.00005
# a closed eye's upper lashes lie at least this far (metres) outside the lower lashes where
# both cover the same spot, seen from the eye's centre within LASH_ORDER_REACH
LASH_ORDER_GAP = 0.00015
LASH_ORDER_REACH = 0.0008
# the in-between of a closed pose without a half-closed pose: the path around the eye in
# full for rows at least ARC_FULL eyeball radii from the eye's horizontal axis, none within
# ARC_FROM
ARC_FROM = 0.3
ARC_FULL = 0.7
# the in-between corrective of a blink target in GLB files
INBETWEEN_SUFFIX = ".inbetween"
# a margin vertex within this distance (eyeball radii along the eye's horizontal axis) of
# another that lies nearer the opening is left out of the margin's line
MARGIN_BIN = 0.025
# a triangle pressed to less than this share of its area at rest is flat, and does not count
# as folded
FLAT = 0.05
# a lid point that lies closer than this (metres) to the eye at rest already touches it: the
# straight line and the measures leave it aside
REST_CONTACT = 0.00005
# the lumps of a lid are its rises and sinks against the skin within about this distance
# (metres, the deviation of a Gaussian) (:func:`lid_lumps`)
LUMP_SIGMA = 0.001
# the corners of the eye: the crossing and the clipping measures leave out the ends of the
# margins within this distance (metres) along the eye's horizontal axis
CORNER_MARGIN = 0.002


# -------------------------------------------------------------------------- mesh helpers
def weld(vertices: np.ndarray, triangles: np.ndarray, tolerance: float = 1e-7):
    """The welded vertices and triangles of a mesh with split seams, and the welded index of
    each vertex."""
    vertices = np.asarray(vertices, dtype=np.float64)
    key = np.round(vertices / tolerance).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    return vertices[first], inverse[np.asarray(triangles, dtype=np.int64)], inverse


def closest_on_mesh(points, vertices, triangles, k: int = 12):
    """
    The closest points (P, 3) of ``points`` on a triangle mesh, their signed distances (P,)
    along the closest triangle's normal (by the triangles' winding; positive on the side the
    normal points to) and the closest triangle of each. The candidates are the ``k`` triangles
    whose centres lie nearest.
    """
    from scipy.spatial import cKDTree

    from opensculptboy.mouth.poses import _closest_on_triangles

    points = np.asarray(points, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if len(points) == 0 or len(triangles) == 0:
        return np.zeros((len(points), 3)), np.full(len(points), np.inf), None
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    centres = (a + b + c) / 3.0
    k = min(k, len(triangles))
    _, cand = cKDTree(centres).query(points, k=k)
    cand = np.asarray(cand).reshape(len(points), k)
    best = np.full(len(points), np.inf)
    out = np.zeros((len(points), 3))
    which = np.zeros(len(points), np.int64)
    for j in range(k):
        t = cand[:, j]
        q, _ = _closest_on_triangles(points, a[t], b[t], c[t])
        d = np.linalg.norm(points - q, axis=1)
        better = d < best
        best[better], out[better], which[better] = d[better], q[better], t[better]
    normal = np.cross(b[which] - a[which], c[which] - a[which])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    side = ((points - out) * normal).sum(1)
    # a point beyond an edge sees the edge's two triangles; the mean of their normals at the
    # closest point decides the side (the sign of the distance along it)
    signed = np.where(side >= 0, best, -best)
    return out, signed, which


def skin_side(points, vertices, triangles, k: int = 32):
    """
    The signed distance (P,) of ``points`` from a sheet of skin (a mesh of ``vertices`` and
    ``triangles`` wound outward, which need not be closed), negative under the skin; the
    skin's unit normal at each closest point (P, 3); and the closest points (P, 3). The side
    comes from the skin's vertex normals blended at the closest point, so a point beyond an
    edge or a corner, or in a fold of the skin, takes the side of the skin around it.
    """
    from opensculptboy.export.body import vertex_normals
    from opensculptboy.mouth.poses import _closest_on_triangles
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if len(points) == 0 or len(triangles) == 0:
        return np.full(len(points), np.inf), np.zeros_like(points), points.copy()
    normals = vertex_normals(vertices, triangles)
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    k = min(k, len(triangles))
    _, cand = cKDTree((a + b + c) / 3.0).query(points, k=k)
    cand = np.asarray(cand).reshape(len(points), k)
    best = np.full(len(points), np.inf)
    q = np.zeros_like(points)
    bary = np.zeros_like(points)
    which = np.zeros(len(points), np.int64)
    for j in range(k):
        t = cand[:, j]
        qj, bj = _closest_on_triangles(points, a[t], b[t], c[t])
        d = np.linalg.norm(points - qj, axis=1)
        better = d < best
        best[better], q[better], bary[better], which[better] = (
            d[better],
            qj[better],
            bj[better],
            t[better],
        )
    n = np.einsum("pk,pkd->pd", bary, normals[triangles[which]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    side = np.where(((points - q) * n).sum(1) < 0.0, -1.0, 1.0)
    return side * best, n, q


def neighbours(triangles: np.ndarray, count: int):
    """The adjacency (CSR, row-normalised) of a mesh's vertices."""
    from scipy.sparse import coo_matrix, diags

    t = np.asarray(triangles, dtype=np.int64)
    a = np.concatenate([t[:, 0], t[:, 1], t[:, 2], t[:, 1], t[:, 2], t[:, 0]])
    b = np.concatenate([t[:, 1], t[:, 2], t[:, 0], t[:, 0], t[:, 1], t[:, 2]])
    A = coo_matrix((np.ones(len(a)), (a, b)), shape=(count, count)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    return diags(1.0 / np.maximum(deg, 1.0)) @ A


def folded_at_rest(rest: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """The mask of the triangles whose normal points against the mean of their corners'
    normals in ``rest``: slivers that a mesh holds folded over."""
    from opensculptboy.export.body import vertex_normals

    x = np.asarray(rest, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)
    face = np.cross(x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]])
    return (face * vertex_normals(x, t)[t].sum(1)).sum(1) < 0.0


def steady_folds(rest, triangles, offsets, rows) -> np.ndarray:
    """
    The offsets (N, 3) with the corners of each group of triangles that are folded at rest
    (:func:`folded_at_rest`, touching ``rows``) moved together by their mean offset, so that
    the folded slivers keep their shape: a sliver squeezed in a blink can turn the normal of
    its corner over (Blender weighs the faces' normals by their angles), which shows as a
    dark speck.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    t = np.asarray(triangles, dtype=np.int64)
    out = np.array(offsets, dtype=np.float64, copy=True)
    mine = np.zeros(len(out), bool)
    mine[np.asarray(rows)] = True
    bad = t[folded_at_rest(rest, t) & mine[t].any(1)]
    if len(bad) == 0:
        return out
    corners = np.unique(bad)
    local = np.full(len(out), -1)
    local[corners] = np.arange(len(corners))
    e = np.concatenate([bad[:, [0, 1]], bad[:, [1, 2]]])
    n = len(corners)
    graph = coo_matrix(
        (np.ones(len(e)), (local[e[:, 0]], local[e[:, 1]])), shape=(n, n)
    )
    _, label = connected_components(graph, directed=False)
    for k in np.unique(label):
        group = corners[label == k]
        out[group] = out[group].mean(0)
    return out


# --------------------------------------------------------------------- the eye's frame
def cylinder(frame, points: np.ndarray):
    """Points in the eye's frame about its horizontal axis (eyeball radii): the place along
    the axis (``s``, +X of the frame), the angle about it from the gaze (``phi``, radians,
    up is positive) and the distance from it (``rho``)."""
    q = frame.to_local(np.asarray(points, dtype=np.float64))
    return q[:, 0], np.arctan2(q[:, 1], q[:, 2]), np.hypot(q[:, 1], q[:, 2])


def from_cylinder(frame, s, phi, rho) -> np.ndarray:
    local = np.stack([s, rho * np.sin(phi), rho * np.cos(phi)], 1)
    return frame.to_world(local)


@dataclasses.dataclass
class EyeSurface:
    """An eye's outer surface as a triangle mesh (Anny's frame, metres) whose normals point
    out of the eye."""

    vertices: np.ndarray
    triangles: np.ndarray

    def depth(self, points: np.ndarray) -> np.ndarray:
        """How far (metres) each point lies inside the surface (negative outside)."""
        _, signed, _ = closest_on_mesh(points, self.vertices, self.triangles)
        return -signed


def sphere_surface(frame, outer=None, segments: int = 48) -> EyeSurface:
    """A surface of revolution about the gaze of ``frame`` whose distance from the centre at
    an angle theta from the gaze is ``outer(theta)`` eyeball radii (1 for a sphere)."""
    from opensculptboy.eyes import geometry

    theta = np.radians(np.r_[np.arange(0.0, 100.0, 2.5), np.arange(100.0, 181.0, 10.0)])
    r = np.ones_like(theta) if outer is None else np.asarray(outer(theta))
    profile = np.stack([r * np.sin(theta), r * np.cos(theta)], 1)
    normals = np.stack([np.sin(theta), np.cos(theta)], 1)
    surface = geometry.revolve([np.c_[profile, normals]], segments)
    return EyeSurface(frame.to_world(surface.points), surface.triangles)


# ------------------------------------------------------------------------------ the lids
@dataclasses.dataclass
class EyeLids:
    """
    The lids of one eye on a mesh: its frame, the lid margins (``upper``, ``lower``: rows from
    the side of the nose to the temple), the rows of each lid's skin (``upper_rows``,
    ``lower_rows``: the outward-facing skin near the margin and the skin of the socket on its
    side), the rows that the shaping may move (``region``), and the outward-facing skin around
    the eye (``outer_triangles``), whose outside the lashes keep to.
    """

    side: str
    frame: object
    upper: np.ndarray
    lower: np.ndarray
    upper_rows: np.ndarray
    lower_rows: np.ndarray
    region: np.ndarray
    outer_triangles: np.ndarray
    lid_triangles: np.ndarray
    # each lid's outward-facing skin near its margin: rows and triangles
    upper_skin: np.ndarray | None = None
    lower_skin: np.ndarray | None = None
    upper_triangles: np.ndarray | None = None
    lower_triangles: np.ndarray | None = None
    # the triangles of the region around the eye
    region_triangles: np.ndarray | None = None
    # the rows of the skin of the socket (the lining of the lids, facing the eye)
    socket_rows: np.ndarray | None = None


def eye_lids(frame, rest, triangles, skin, normals=None) -> EyeLids | None:
    """
    The :class:`EyeLids` of the eye of ``frame`` on a welded mesh (``rest`` (N, 3), its
    ``triangles`` without the eyeballs', ``skin`` the mask of the vertices that are not on an
    eyeball), or None when its margins are not found.
    """
    from opensculptboy.export.body import vertex_normals
    from opensculptboy.eyes import geometry
    from opensculptboy.eyes.lashes import LOWER, SKIN_REACH, UPPER, lid_margins

    rest = np.asarray(rest, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if normals is None:
        normals = vertex_normals(rest, triangles)
    margins = lid_margins(frame, rest, normals, skin, triangles)
    if UPPER not in margins or LOWER not in margins:
        return None
    local = frame.to_local(rest)
    dist = np.linalg.norm(local, axis=1)
    socket = geometry.socket(frame, rest, normals, skin)
    region = skin & (dist < LID_REACH) & (local[:, 2] > LID_FRONT)
    # the side of each region vertex: above or below the line through the corners (in the
    # frame's x and y), the socket included
    up, lo = margins[UPPER], margins[LOWER]
    ends = np.concatenate([up.rows[[0, -1]], lo.rows[[0, -1]]])
    xs = local[ends, 0]
    a, b = ends[np.argmin(xs)], ends[np.argmax(xs)]
    line = local[a, 1] + (local[b, 1] - local[a, 1]) * (
        local[:, 0] - local[a, 0]
    ) / max(local[b, 0] - local[a, 0], 1e-12)
    above = local[:, 1] > line
    near = (dist < 1.35) & (local[:, 2] > -0.2)
    upper_rows = np.union1d(
        up.lid_rows, np.flatnonzero(socket & near & above & region)
    ).astype(np.int64)
    lower_rows = np.union1d(
        lo.lid_rows, np.flatnonzero(socket & near & ~above & region)
    ).astype(np.int64)
    outer = skin & ~socket & (dist < SKIN_REACH) & (local[:, 2] > -0.2)
    outer_tris = triangles[outer[triangles].all(1)]
    lid = np.zeros(len(rest), bool)
    lid[upper_rows] = True
    lid[lower_rows] = True
    lid_tris = triangles[(lid[triangles].all(1)) & ~socket[triangles].any(1)]

    def own(rows):
        mask = np.zeros(len(rest), bool)
        mask[rows] = True
        return triangles[mask[triangles].all(1)]

    return EyeLids(
        frame.side,
        frame,
        up.rows,
        lo.rows,
        upper_rows,
        lower_rows,
        np.flatnonzero(region),
        outer_tris,
        lid_tris,
        up.lid_rows,
        lo.lid_rows,
        own(up.lid_rows),
        own(lo.lid_rows),
        triangles[region[triangles].all(1)],
        np.flatnonzero(socket),
    )


def _margin_curve(lids: EyeLids, positions, rows, lowest: bool):
    """A lid margin's line as the angle about the eye's horizontal axis against the place
    along it: (s, phi), in eyeball radii and radians. A game-ready lid's rim can hold more
    than one row of margin vertices; the line is the rim's edge toward the opening: a margin
    vertex is left out when another lies within :data:`MARGIN_BIN` of it along the axis and
    nearer the opening (lower for the upper lid, ``lowest``; higher for the lower lid)."""
    s, phi, _ = cylinder(lids.frame, positions[rows])
    order = np.argsort(s)
    s, phi = s[order], phi[order]
    sense = 1.0 if lowest else -1.0
    keep = np.ones(len(s), bool)
    for i in range(len(s)):
        near = np.abs(s - s[i]) < MARGIN_BIN
        if (sense * phi[near] < sense * phi[i] - 1e-12).any():
            keep[i] = False
    return s[keep], phi[keep]


# ------------------------------------------------------------------------------ measures
def lid_crossing(lids: EyeLids, positions: np.ndarray) -> tuple[float, float]:
    """
    How far (mm, along the eye's surface) the upper lid's rim lies below the lower lid's rim
    (:func:`_margin_curve`), at its worst along the eye's horizontal axis away from the corners
    (:data:`CORNER_MARGIN`), 0 when it lies above everywhere; and the widest opening between
    the rims (mm, 0 when they meet or cross everywhere).
    """
    positions = np.asarray(positions, dtype=np.float64)
    R = lids.frame.radius
    s_u, p_u = _margin_curve(lids, positions, lids.upper, True)
    s_l, p_l = _margin_curve(lids, positions, lids.lower, False)
    lo = max(s_u[0], s_l[0]) + CORNER_MARGIN / R
    hi = min(s_u[-1], s_l[-1]) - CORNER_MARGIN / R
    if hi <= lo:
        return 0.0, 0.0
    _, _, rho = cylinder(lids.frame, positions[np.r_[lids.upper, lids.lower]])
    ss = np.linspace(lo, hi, 64)
    over = (np.interp(ss, s_l, p_l) - np.interp(ss, s_u, p_u)) * float(np.median(rho))
    over *= R * 1000.0
    return float(max(over.max(), 0.0)), float(max(-over.min(), 0.0))


def _segment_hits(p, q, a, b, c, eps: float = 1e-3) -> np.ndarray:
    """Whether segments p-q cross triangles a, b, c inside both (Moller-Trumbore), each
    (K, 3); a touch within ``eps`` of an end or an edge does not count."""
    d = q - p
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = (e1 * h).sum(1)
    ok = np.abs(det) > 1e-18
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    sv = p - a
    u = (sv * h).sum(1) * inv
    qv = np.cross(sv, e1)
    v = (d * qv).sum(1) * inv
    t = (e2 * qv).sum(1) * inv
    return ok & (u > eps) & (v > eps) & (u + v < 1 - eps) & (t > eps) & (t < 1 - eps)


def lid_lumps(lids: EyeLids, positions: np.ndarray, rest: np.ndarray) -> float:
    """
    How far (mm) the skin of the upper lid rises or sinks against the skin around it, at its
    worst away from the corners of the eye (:data:`CORNER_MARGIN`): for each row of the
    lid's outward skin, its distance from the eye's centre less the Gaussian mean
    (:data:`LUMP_SIGMA`, over the rows near it at ``rest``) of the distances of the rows
    around it. A lid that turns about the eye keeps every distance from the eye's centre, so
    a closed lid measures what it has of its rest's relief and of the lumps and folds that
    the closing gives it; compare it with the lid at rest, or with the closed lid of a
    smooth level.
    """
    from scipy.spatial import cKDTree

    rows = lids.upper_skin
    if rows is None or len(rows) == 0:
        return 0.0
    x = np.asarray(positions, dtype=np.float64)
    x0 = np.asarray(rest, dtype=np.float64)
    R = lids.frame.radius
    ends = []
    for margin in (lids.upper, lids.lower):
        s, _, _ = cylinder(lids.frame, x0[margin])
        ends.append((s.min(), s.max()))
    lo = max(ends[0][0], ends[1][0]) + CORNER_MARGIN / R
    hi = min(ends[0][1], ends[1][1]) - CORNER_MARGIN / R
    p0 = x0[rows]
    r = np.linalg.norm(x[rows] - lids.frame.centre, axis=1)
    s, _, _ = cylinder(lids.frame, p0)
    worst = 0.0
    for i, near in enumerate(cKDTree(p0).query_ball_point(p0, 3.0 * LUMP_SIGMA)):
        if not lo < s[i] < hi:
            continue
        near = np.asarray(near)
        w = np.exp(-0.5 * (np.linalg.norm(p0[near] - p0[i], axis=1) / LUMP_SIGMA) ** 2)
        worst = max(worst, abs(r[i] - (w * r[near]).sum() / w.sum()))
    return worst * 1000.0


def lid_clipping(lids: EyeLids, positions: np.ndarray) -> int:
    """The number of pairs of triangles of the upper lid's and the lower lid's
    outward-facing skin that cut through each other, away from the corners of the eye
    (:data:`CORNER_MARGIN` inside the ends of the margins), where both lids join in one fold
    (pairs that share a vertex left out)."""
    from scipy.spatial import cKDTree

    up, lo = lids.upper_triangles, lids.lower_triangles
    if up is None or lo is None or len(up) == 0 or len(lo) == 0:
        return 0
    x = np.asarray(positions, dtype=np.float64)
    R = lids.frame.radius
    ends = []
    for rows in (lids.upper, lids.lower):
        s, _, _ = cylinder(lids.frame, x[rows])
        ends.append((s.min(), s.max()))
    lo_s = max(ends[0][0], ends[1][0]) + CORNER_MARGIN / R
    hi_s = min(ends[0][1], ends[1][1]) - CORNER_MARGIN / R

    def inner(tris):
        s, _, _ = cylinder(lids.frame, x[tris].mean(1))
        return tris[(s > lo_s) & (s < hi_s)]

    up, lo = inner(up), inner(lo)
    if len(up) == 0 or len(lo) == 0:
        return 0
    cu, cl = x[up].mean(1), x[lo].mean(1)
    reach = 2.0 * max(
        np.linalg.norm(x[up] - cu[:, None], axis=2).max(),
        np.linalg.norm(x[lo] - cl[:, None], axis=2).max(),
    )
    pairs = cKDTree(cu).query_ball_tree(cKDTree(cl), reach)
    i = np.repeat(np.arange(len(up)), [len(p) for p in pairs])
    j = np.concatenate([np.asarray(p, np.int64) for p in pairs]) if len(i) else i
    if len(i) == 0:
        return 0
    shared = (up[i][:, :, None] == lo[j][:, None, :]).any((1, 2))
    i, j = i[~shared], j[~shared]
    hit = np.zeros(len(i), bool)
    for first, second, k1, k2 in ((up, lo, i, j), (lo, up, j, i)):
        for e in ((0, 1), (1, 2), (2, 0)):
            p, q = x[first[k1, e[0]]], x[first[k1, e[1]]]
            a, b, c = (x[second[k2, n]] for n in range(3))
            hit |= _segment_hits(p, q, a, b, c)
    return int(hit.sum())


def folded_triangles(
    lids: EyeLids, positions: np.ndarray, rest: np.ndarray, surface=None
) -> int:
    """The number of the lids' triangles (the skin of the region around the eye) that fold
    over: whose normal turns away from the mean of its corners' normals, beyond those folded
    at rest. With the eye's ``surface``, the triangles that lie on the eye at rest (the
    lining of the socket, hidden behind the lids) are left out."""

    tris = lids.region_triangles
    if tris is None or len(tris) == 0:
        return 0
    if surface is not None:
        rest = np.asarray(rest, dtype=np.float64)
        tris = tris[surface.depth(rest[tris].mean(1)) <= -REST_CONTACT]
    return int(_folded(tris, positions, rest, lids).sum())


def _folded(tris, positions, rest, lids) -> np.ndarray:
    """The mask of ``tris`` that fold over in ``positions`` and not at ``rest``, away from
    the corners of the eye (:data:`CORNER_MARGIN`); a triangle pressed flat (to less than
    :data:`FLAT` of its area at rest, as the lids' rows are where they meet) does not
    count."""
    from opensculptboy.export.body import vertex_normals

    x, x0 = np.asarray(positions), np.asarray(rest)

    def faces(x):
        a, b, c = (x[tris[:, i]] for i in range(3))
        return np.cross(b - a, c - a)

    def folded(x, face):
        normals = vertex_normals(x, tris)
        return (face * normals[tris].sum(1)).sum(1) < 0.0

    f, f0 = faces(x), faces(x0)
    out = folded(x, f) & ~folded(x0, f0)
    out &= np.linalg.norm(f, axis=1) > FLAT * np.linalg.norm(f0, axis=1)
    R = lids.frame.radius
    ends = []
    for rows in (lids.upper, lids.lower):
        s, _, _ = cylinder(lids.frame, x[rows])
        ends.append((s.min(), s.max()))
    lo = max(ends[0][0], ends[1][0]) + CORNER_MARGIN / R
    hi = min(ends[0][1], ends[1][1]) - CORNER_MARGIN / R
    s, _, _ = cylinder(lids.frame, x[tris].mean(1))
    return out & (s > lo) & (s < hi)


def eye_penetration(
    lids: EyeLids, positions: np.ndarray, rest: np.ndarray, surface: EyeSurface
) -> tuple[float, int]:
    """
    How deep (mm) the lids' outward skin lies inside the eye's outer surface: over its
    vertices and the centres of its triangles that lie outside the surface at rest (the
    lining of the socket, which slides on the eye behind the margins, left out). Returns the
    largest depth (0 when none) and the number of points more than 0.05 mm inside.
    """
    rows = np.union1d(lids.upper_skin, lids.lower_skin)
    tri = np.concatenate([lids.upper_triangles, lids.lower_triangles])
    at_rest = np.concatenate([rest[rows], rest[tri].mean(1)])
    now = np.concatenate([positions[rows], positions[tri].mean(1)])
    outside = surface.depth(at_rest) <= -REST_CONTACT
    if not outside.any():
        return 0.0, 0
    depth = surface.depth(now[outside]) * 1000.0
    return float(max(depth.max(initial=0.0), 0.0)), int((depth > 0.05).sum())


def lash_penetration(
    lids: EyeLids,
    positions: np.ndarray,
    rest: np.ndarray,
    lash_now: np.ndarray,
    lash_rest: np.ndarray,
) -> tuple[float, int]:
    """
    How deep (mm) lash points lie inside the lids' outward-facing skin beyond their depth at
    rest (the roots of a reference's lashes may sit in the skin), more than
    :data:`LASH_TOLERANCE`. Returns the largest depth (0 when none) and the number of points.
    """
    if len(lash_now) == 0 or len(lids.outer_triangles) == 0:
        return 0.0, 0
    at_rest = skin_side(lash_rest, rest, lids.outer_triangles)[0]
    now = skin_side(lash_now, positions, lids.outer_triangles)[0]
    floor = np.minimum(at_rest, 0.0) - LASH_TOLERANCE
    depth = (floor - now) * 1000.0
    return float(max(depth.max(initial=0.0), 0.0)), int((depth > 0.0).sum())


def lash_order(frame, upper: np.ndarray, lower: np.ndarray) -> tuple[float, int]:
    """
    How far (mm) a closed eye's upper lashes lie under its lower lashes: for each upper lash
    point, the lower lash points within :data:`LASH_ORDER_REACH` of its direction from the
    eye's centre (at its distance) should lie :data:`LASH_ORDER_GAP` closer to the centre.
    Returns the largest shortfall (0 when none) and the number of upper points short of it.
    """
    from scipy.spatial import cKDTree

    if len(upper) == 0 or len(lower) == 0:
        return 0.0, 0
    cu = np.asarray(upper) - frame.centre
    cl = np.asarray(lower) - frame.centre
    ru, rl = np.linalg.norm(cu, axis=1), np.linalg.norm(cl, axis=1)
    du, dl = cu / ru[:, None], cl / rl[:, None]
    reach = LASH_ORDER_REACH / max(float(np.median(ru)), 1e-9)
    tree = cKDTree(dl)
    worst, count = 0.0, 0
    for i, near in enumerate(tree.query_ball_point(du, reach)):
        if not near:
            continue
        short = (rl[near].max() + LASH_ORDER_GAP - ru[i]) * 1000.0
        if short > 0.01:
            worst = max(worst, float(short))
            count += 1
    return worst, count


def measure_blink(lids, rest, closed, surface, lash=None, half=None, stages=STAGES):
    """
    The measures of one eye's blink at ``stages``: ``closed`` (N, 3) is the closed pose; the
    pose at a stage t is the straight line ``rest + t (closed - rest)``, plus ``tent(t) half``
    when ``half`` (N, 3), an in-between corrective, is given (tent(t) = 1 - |2 t - 1|).
    ``lash`` is (lash points at rest (M, 3), closed (M, 3), the in-between corrective (M, 3)
    or None, the mask (M,) of the upper lashes). Returns {stage: {crossing_mm, ...}}.
    """
    rest = np.asarray(rest, dtype=np.float64)
    D = np.asarray(closed, dtype=np.float64) - rest
    out = {}
    for t in stages:
        tent = 1.0 - abs(2.0 * t - 1.0)
        x = rest + t * D + (tent * half if half is not None else 0.0)
        cross, opening = lid_crossing(lids, x)
        depth, n_depth = eye_penetration(lids, x, rest, surface)
        row = {
            "crossing_mm": round(cross, 3),
            "opening_mm": round(opening, 3),
            "clipping_pairs": lid_clipping(lids, x),
            "folded_triangles": folded_triangles(lids, x, rest, surface),
            "eye_mm": round(depth, 3),
            "eye_points": n_depth,
            "lumps_mm": round(lid_lumps(lids, x, rest), 3),
        }
        if lash is not None:
            l0, l1, lh, upper_mask = lash
            lx = l0 + t * (l1 - l0) + (tent * lh if lh is not None else 0.0)
            ld, nl = lash_penetration(lids, x, rest, lx, l0)
            row.update(lash_mm=round(ld, 3), lash_points=nl)
            if t == 1.0:
                od, no = lash_order(lids.frame, lx[upper_mask], lx[~upper_mask])
                row.update(lash_order_mm=round(od, 3), lash_order_points=no)
        out[f"{t:g}"] = row
    return out


# ------------------------------------------------------------------------ the eyed mesh
def eye_surfaces(eyed) -> dict:
    """The outer surface of each eye of an :class:`~opensculptboy.eyes.mesh.EyedMesh`
    (side to :class:`EyeSurface`, at rest): its cornea shell, else its eyeball."""
    out = {}
    for name in ("eyeball", "cornea"):  # the cornea, where there is one, wins
        for p in eyed.parts:
            if p.name == name:
                surface = EyeSurface(np.asarray(p.rest), np.asarray(p.triangles))
                if surface.depth(eyed.frames[p.side].centre[None])[0] < 0:
                    surface = EyeSurface(surface.vertices, surface.triangles[:, ::-1])
                out[p.side] = surface
    return out


def eyed_lids(eyed, rest: np.ndarray) -> dict:
    """The :class:`EyeLids` of each eye of an eyed mesh (side to lids; an eye whose margins
    are not found is left out) on its base mesh at rest (``rest`` (N, 3))."""
    rest = np.asarray(rest, dtype=np.float64)
    triangles = eyed.triangles[: int(eyed.kept.sum())]
    skin = np.zeros(len(rest), bool)
    skin[triangles] = True
    out = {}
    for side, frame in eyed.frames.items():
        lids = eye_lids(frame, rest, triangles, skin)
        if lids is not None:
            out[side] = lids
    return out


# ------------------------------------------------------------------------------ the rig
@dataclasses.dataclass
class BlinkShape:
    """One eye's blink: the closed pose's offsets (N, 3) for the eye's blink target, the
    in-between corrective (N, 3) that takes the half-way point of the target to the
    half-closed pose, the turn of each row about the eye's horizontal axis in the closed pose
    (N,), radians, up is positive), and what the closing did (``stats``)."""

    closed: np.ndarray
    inbetween: np.ndarray
    turn: np.ndarray
    stats: dict = dataclasses.field(default_factory=dict)
    # the extra turn (N,) that brought the rims together (:func:`meet_rims`), or None
    extra: np.ndarray | None = None


def _radial(frame, points):
    d = np.asarray(points) - frame.centre
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)


def _rim(lids: EyeLids, x, rows, lowest: bool) -> np.ndarray:
    """The rows of a lid margin on its rim, the edge toward the opening
    (:func:`_margin_curve`), in their order."""
    s, phi, _ = cylinder(lids.frame, x[rows])
    sense = 1.0 if lowest else -1.0
    keep = np.ones(len(rows), bool)
    for i in range(len(rows)):
        near = np.abs(s - s[i]) < MARGIN_BIN
        if (sense * phi[near] < sense * phi[i] - 1e-12).any():
            keep[i] = False
    return np.asarray(rows)[keep]


def lid_turns(lids: EyeLids, rest, offsets) -> np.ndarray:
    """The turn (N,) of each row of the lids' region about the eye's horizontal axis from
    ``rest`` to ``rest + offsets`` (radians, up is positive; 0 outside the region)."""
    rest = np.asarray(rest, dtype=np.float64)
    out = np.zeros(len(rest))
    r = np.asarray(lids.region)
    _, p0, _ = cylinder(lids.frame, rest[r])
    _, p1, _ = cylinder(lids.frame, rest[r] + np.asarray(offsets)[r])
    out[r] = np.angle(np.exp(1j * (p1 - p0)))
    return out


def line_push(
    x0, x1, surface: EyeSurface, frame, clearance=LID_CLEARANCE, checks=LINE_CHECKS
):
    """How far (metres, (n,)) to push each end point ``x1`` out from the eye's centre so that
    the straight line from ``x0`` stays ``clearance`` outside the eye's surface at the shares
    ``checks`` of the way (at most :data:`MAX_PUSH`)."""
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    n = _radial(frame, x1)
    ts = np.asarray(checks, dtype=np.float64)

    def depth(push, a, b, d):
        end = b + push[:, None] * d
        pts = a[None] + ts[:, None, None] * (end - a)[None]
        return surface.depth(pts.reshape(-1, 3)).reshape(len(ts), len(a)) + clearance

    need = depth(np.zeros(len(x0)), x0, x1, n).max(0) > 1e-7
    out = np.zeros(len(x0))
    if not need.any():
        return out
    lo = np.zeros(int(need.sum()))
    hi = np.full(int(need.sum()), MAX_PUSH)
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        ok = depth(mid, x0[need], x1[need], n[need]).max(0) <= 1e-7
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    out[need] = hi
    return out


def clear_line(lids: EyeLids, rest, closed, surface: EyeSurface, triangles):
    """
    A closed pose (N, 3) whose straight line from ``rest`` keeps the lids' outward skin
    :data:`LID_CLEARANCE` outside the eye's surface: each lid row that lies outside the eye
    at rest moves out from the eye's centre by what it needs (:func:`line_push`), and the
    push spreads smoothly over the lids. A morph target alone (VRM files) plays the straight
    line; the GLB files play the path with the in-between and take the pose as it is.
    """
    from opensculptboy.fit.register import smooth_mouth

    X0 = np.asarray(rest, dtype=np.float64)
    out = np.array(closed, dtype=np.float64, copy=True)
    region = np.zeros(len(X0), bool)
    region[lids.region] = True
    lining = np.zeros(len(X0), bool)
    if lids.socket_rows is not None:
        lining[lids.socket_rows] = True
    r = np.flatnonzero(region & ~lining)
    r = r[surface.depth(X0[r]) <= -REST_CONTACT]
    push = line_push(X0[r], X0[r] + out[r], surface, lids.frame)
    if push.max(initial=0.0) <= 0.0:
        return out
    moved = np.zeros_like(X0)
    moved[r] = push[:, None] * _radial(lids.frame, X0[r] + out[r])
    held = np.zeros(len(X0), bool)
    held[r[push > 0]] = True
    free = region & ~held
    out += smooth_mouth(moved, triangles, free)
    return out


def _arc_inbetween(lids: EyeLids, rest, closed, surface: EyeSurface) -> np.ndarray:
    """The in-between (N, 3) of a closed pose without a half-closed pose: half the turn of
    each row about the eye's horizontal axis, with its place along the axis and its distance
    from it half way, less the half-way point of the straight line."""
    X0 = np.asarray(rest, dtype=np.float64)
    X1 = X0 + np.asarray(closed, dtype=np.float64)
    region = np.zeros(len(X0), bool)
    region[lids.region] = True
    r = np.flatnonzero(region & (np.linalg.norm(closed, axis=1) > 1e-6))
    s0, p0, q0 = cylinder(lids.frame, X0[r])
    s1, p1, q1 = cylinder(lids.frame, X1[r])
    turn = np.angle(np.exp(1j * (p1 - p0)))
    half = from_cylinder(lids.frame, 0.5 * (s0 + s1), p0 + 0.5 * turn, 0.5 * (q0 + q1))
    # near the eye's horizontal axis the angle about it says little: the path keeps to the
    # straight line there
    mid = 0.5 * (X0[r] + X1[r])
    w = np.clip((np.minimum(q0, q1) - ARC_FROM) / (ARC_FULL - ARC_FROM), 0, 1)
    out = np.zeros_like(X0)
    out[r] = w[:, None] * (half - mid)
    return out


def _upper_weights(lids: EyeLids, rest, rims):
    """Each region row's share (N,) of the upper lid's extra turn: 1 on the upper rim
    (``rims``: the upper rim's rows, then the lower rim's), falling smoothly to 0 at
    :data:`MEET_FALLOFF` from it; 0 below the line half way between the rims at rest."""
    from scipy.spatial import cKDTree

    frame = lids.frame
    X0 = np.asarray(rest, dtype=np.float64)
    r = np.asarray(lids.region)
    ru, rl = rims
    s0, p0, _ = cylinder(frame, X0[r])
    curves = []
    for rows in (ru, rl):
        sr, pr, _ = cylinder(frame, X0[rows])
        o = np.argsort(sr)
        curves.append((sr[o], pr[o]))
    (su, pu), (sl, pl) = curves
    s = np.clip(s0, max(su[0], sl[0]), min(su[-1], sl[-1]))
    upper = p0 > 0.5 * (np.interp(s, su, pu) + np.interp(s, sl, pl))
    d, _ = cKDTree(X0[ru]).query(X0[r])
    t = np.clip(1.0 - d / MEET_FALLOFF, 0.0, 1.0)
    out = np.zeros(len(X0))
    out[r] = np.where(upper, t * t * (3.0 - 2.0 * t), 0.0)
    return out


def meet_rims(
    lids: EyeLids, rest, closed, half, triangles, overlap=RIM_OVERLAP, back=True
):
    """
    A closed pose (N, 3) whose rims meet, its half-closed pose (N, 3, or None) and the
    extra turn of each row (N,). A fitted lid lies a little off its reference's, so a carried
    pose can leave the rims apart, and anny's own blink closes the upper lid past the lower
    one: the upper lid turns about the eye's horizontal axis until its rim lies ``overlap``
    (metres) past the closed lower rim, as a game character's upper lid closes over its lower
    lid; with ``back`` false, a rim that passes further keeps its place. The turn fades out
    toward the corners of the eye (:data:`MEET_TAPER`) and stays within
    :data:`MAX_RIM_TURN`. Each row of the upper lid turns by its weight
    (:func:`_upper_weights`), and the half-closed pose by half as much. The smoothed turn
    can leave the rims apart between the upper rim's vertices of a coarse lid, so the last
    rounds seal them (:data:`SEAL_ROUNDS`): the upper rim's vertices turn down by the least
    turns that close every gap left (:func:`_seal_turns`).
    """
    frame = lids.frame
    R = frame.radius
    X0 = np.asarray(rest, dtype=np.float64)
    out = np.array(closed, dtype=np.float64, copy=True)
    half = None if half is None else np.array(half, dtype=np.float64, copy=True)
    extra = np.zeros(len(X0))
    ru = _rim(lids, X0, lids.upper, True)
    rl = _rim(lids, X0, lids.lower, False)
    w = _upper_weights(lids, X0, (ru, rl))
    r = np.flatnonzero(w > 0)
    sr, _, _ = cylinder(frame, X0[r])
    _, _, q = cylinder(frame, X0[lids.upper])
    rho = float(np.median(q))
    for _ in range(MEET_ROUNDS):
        X1 = X0 + out
        s_u, p_u = _margin_curve(lids, X1, lids.upper, True)
        s_l, p_l = _margin_curve(lids, X1, lids.lower, False)
        inside = (s_u > s_l[0]) & (s_u < s_l[-1])
        need = p_u - np.interp(s_u, s_l, p_l) + overlap / (R * rho)
        if not back:
            need = np.clip(need, 0.0, None)
        lo, hi = max(s_u[0], s_l[0]), min(s_u[-1], s_l[-1])
        taper = np.clip(np.minimum(s_u - lo, hi - s_u) / (MEET_TAPER * (hi - lo)), 0, 1)
        need = np.where(inside, need * taper * taper * (3.0 - 2.0 * taper), 0.0)
        if np.abs(need).max(initial=0.0) <= 1e-4:
            break
        # the closed rim's place along the axis at rest: the upper margin's rows in order
        su0 = np.interp(s_u, *_rest_along(lids, X0, X1))
        need = _smooth_along(need, MEET_SMOOTH)
        step = -w[r] * np.interp(sr, su0, need, left=0.0, right=0.0)
        limit = np.radians(MAX_RIM_TURN)
        step = np.clip(extra[r] + step, -limit, limit) - extra[r]
        extra[r] += step
        s1, p1, q1 = cylinder(frame, X1[r])
        out[r] = from_cylinder(frame, s1, p1 + step, q1) - X0[r]
        if half is not None:
            sh, ph, qh = cylinder(frame, X0[r] + half[r])
            half[r] = from_cylinder(frame, sh, ph + 0.5 * step, qh) - X0[r]
    for _ in range(SEAL_ROUNDS):
        X1 = X0 + out
        s_u, p_u = _margin_curve(lids, X1, lids.upper, True)
        s_l, p_l = _margin_curve(lids, X1, lids.lower, False)
        lo, hi = max(s_u[0], s_l[0]), min(s_u[-1], s_l[-1])
        if hi <= lo or len(s_u) < 2:
            break
        ss = np.linspace(lo, hi, SEAL_SAMPLES)
        gap = (
            np.interp(ss, s_u, p_u) - np.interp(ss, s_l, p_l) + SEAL_OVERLAP / (R * rho)
        )
        away = np.minimum(ss - lo, hi - ss) * R >= CORNER_MARGIN
        gap = np.where(away, np.clip(gap, 0.0, None), 0.0)
        if gap.max(initial=0.0) * R * rho <= 0.2 * SEAL_OVERLAP:
            break
        need = _seal_turns(s_u, ss, gap)
        su0 = np.interp(s_u, *_rest_along(lids, X0, X1))
        step = -w[r] * np.interp(sr, su0, need, left=0.0, right=0.0)
        limit = np.radians(MAX_RIM_TURN)
        step = np.clip(extra[r] + step, -limit, limit) - extra[r]
        extra[r] += step
        s1, p1, q1 = cylinder(frame, X1[r])
        out[r] = from_cylinder(frame, s1, p1 + step, q1) - X0[r]
        if half is not None:
            sh, ph, qh = cylinder(frame, X0[r] + half[r])
            half[r] = from_cylinder(frame, sh, ph + 0.5 * step, qh) - X0[r]
    return out, half, extra


def _seal_turns(s_u, ss, gap) -> np.ndarray:
    """The least turns (n,) of the upper rim's vertices at ``s_u`` (sorted) that cover the
    ``gap`` (S,) at the places ``ss`` along the opening, the rim running straight between its
    vertices: a linear programme, else each vertex takes the widest gap on its two
    segments."""
    from scipy.optimize import linprog

    n = len(s_u)
    j = np.clip(np.searchsorted(s_u, ss) - 1, 0, n - 2)
    a = np.clip((ss - s_u[j]) / np.maximum(s_u[j + 1] - s_u[j], 1e-12), 0.0, 1.0)
    open_ = gap > 0
    hat = np.zeros((int(open_.sum()), n))
    rows = np.arange(len(hat))
    hat[rows, j[open_]] = 1.0 - a[open_]
    hat[rows, j[open_] + 1] += a[open_]
    found = linprog(
        np.ones(n), A_ub=-hat, b_ub=-gap[open_], bounds=(0, None), method="highs"
    )
    if found.status == 0:
        return np.clip(found.x, 0.0, None)
    need = np.zeros(n)
    np.maximum.at(need, j[open_], gap[open_])
    np.maximum.at(need, j[open_] + 1, gap[open_])
    return need


def _rest_along(lids: EyeLids, rest, moved):
    """The upper margin's place along the eye's horizontal axis in ``moved`` against its
    place at rest, both sorted by the first (for :func:`numpy.interp`)."""
    s1, _, _ = cylinder(lids.frame, moved[lids.upper])
    s0, _, _ = cylinder(lids.frame, rest[lids.upper])
    o = np.argsort(s1)
    return s1[o], s0[o]


def _smooth_along(values, passes=3):
    out = np.asarray(values, dtype=np.float64).copy()
    for _ in range(passes):
        if len(out) > 2:
            out[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
    return out


def reference_blink(
    lids: EyeLids, rest, pose, surface: EyeSurface, triangles, clear: bool = False
) -> BlinkShape:
    """
    The :class:`BlinkShape` of one eye from a reference's blink (``pose``, a
    :class:`~opensculptboy.eyes.poses.BlinkPose`): its closed pose as it comes, with the rims
    brought together (:func:`meet_rims`), and its half-closed pose as the in-between
    (half the turn about the eye when it has none). With ``clear`` the closed pose keeps the
    straight line outside the eye (:func:`clear_line`).
    """
    rest = np.asarray(rest, dtype=np.float64)
    closed, half, extra = meet_rims(
        lids, rest, pose.closed, pose.half, triangles, back=False
    )
    pushed = clear_line(lids, rest, closed, surface, triangles) if clear else closed
    if half is not None:
        inbetween = half - 0.5 * pushed
    else:
        inbetween = _arc_inbetween(lids, rest, pushed, surface)
    stats = {
        "source": "reference",
        "rim_turn_deg": round(float(np.degrees(np.abs(extra).max(initial=0.0))), 2),
        "line_push_mm": round(
            float(np.linalg.norm(pushed - closed, axis=1).max() * 1000), 3
        ),
        "inbetween_max_mm": round(
            float(np.linalg.norm(inbetween, axis=1).max() * 1000), 3
        ),
    }
    return BlinkShape(pushed, inbetween, lid_turns(lids, rest, closed), stats, extra)


def rig_blink(
    lids: EyeLids, rest, offsets, surface: EyeSurface, triangles, clear: bool = False
) -> BlinkShape:
    """
    The :class:`BlinkShape` of one eye for a card without a reference's blink, built as a
    reference's is: the closed pose ``offsets`` (anny's own blink) with the rims brought
    together by one turn of the upper lid about the eye (:func:`meet_rims`), the half-closed
    pose half the turn about the eye (:func:`_arc_inbetween`), and lashes that follow their
    lid (:func:`rig_lashes`). With ``clear`` the closed pose keeps the straight line outside
    the eye (:func:`clear_line`).
    """
    X0 = np.asarray(rest, dtype=np.float64)
    D = np.asarray(offsets, dtype=np.float64)
    closed, _, extra = meet_rims(lids, X0, D, None, triangles)
    pushed = clear_line(lids, X0, closed, surface, triangles) if clear else closed
    inbetween = _arc_inbetween(lids, X0, pushed, surface)
    turn = lid_turns(lids, X0, closed)
    stats = {
        "source": "rig",
        "rim_turn_deg": round(float(np.degrees(np.abs(extra).max(initial=0.0))), 2),
        "line_push_mm": round(
            float(np.linalg.norm(pushed - closed, axis=1).max() * 1000), 3
        ),
        "inbetween_max_mm": round(
            float(np.linalg.norm(inbetween, axis=1).max() * 1000), 3
        ),
    }
    return BlinkShape(pushed, inbetween, turn, stats, extra)


# -------------------------------------------------------------------------------- lashes
def turn_about(frame, vectors, angles) -> np.ndarray:
    """``vectors`` (M, 3) turned about the eye's horizontal axis by ``angles`` (M,),
    radians, up is positive (the sense of :func:`cylinder`'s angle)."""
    q = np.asarray(vectors, dtype=np.float64) @ frame.axes
    c, n = np.cos(angles), np.sin(angles)
    y = q[:, 1] * c + q[:, 2] * n
    z = -q[:, 1] * n + q[:, 2] * c
    return np.stack([q[:, 0], y, z], 1) @ frame.axes.T


def rig_lashes(part, frame, rest, closed, inbetween, turn):
    """
    A lash part's offsets (M, 3) in the closed pose and its in-between corrective (M, 3),
    for lashes without a reference's pose: each point keeps its offset from its root on the
    lid margin, the root goes with the lid (closed ``rest + closed``, half closed ``rest +
    closed / 2 + inbetween``), and the offset turns about the eye's horizontal axis by
    :data:`LASH_TURN` of the root's turn (``turn``, (N,)), as the reference's lashes do.
    """
    follow = part.follow
    rows = follow.rows
    x0 = np.asarray(rest, dtype=np.float64)
    root0 = follow.root @ x0[rows]
    root1 = follow.root @ (x0 + closed)[rows]
    rooth = follow.root @ (x0 + 0.5 * closed + inbetween)[rows]
    angle = follow.root @ np.asarray(turn)[rows]
    v = part.rest - root0
    p1 = root1 + turn_about(frame, v, LASH_TURN * angle)
    ph = rooth + turn_about(frame, v, 0.5 * LASH_TURN * angle)
    return p1 - part.rest, ph - 0.5 * (part.rest + p1)


# ---------------------------------------------------------------------- the files' targets
# the blink target of each eye's side (the figure's left eye is +X)
BLINK_TARGETS = {"L": "eyeBlinkLeft", "R": "eyeBlinkRight"}


@dataclasses.dataclass
class EyeBlinks:
    """The blinks of a file's eyes: per side, the row of its blink target among the file's
    targets, its :class:`BlinkShape`, its lids and its eye's surface; the rows of the
    in-between correctives appended after the file's targets (side to row); the reference's
    blinks by side (:class:`~opensculptboy.eyes.poses.BlinkPose`), whose lashes the lash
    parts take; and the scene extras that record them."""

    rows: dict
    shapes: dict
    lids: dict
    surfaces: dict
    inbetween_rows: dict = dataclasses.field(default_factory=dict)
    extras: dict = dataclasses.field(default_factory=dict)
    poses: dict = dataclasses.field(default_factory=dict)


def blink_offsets(eyed, rest, offsets, names, poses=None, inbetween: bool = True):
    """
    A file's morph-target offsets (T, N, 3) of its base mesh (``rest`` (N, 3), the targets'
    ``names``) with each eye's blink target (:data:`BLINK_TARGETS`) set: from the reference's
    blink (``poses``, side to :class:`~opensculptboy.eyes.poses.BlinkPose`) by
    :func:`reference_blink`, else from the synthetic lid rig on the target itself by
    :func:`rig_blink`. With ``inbetween``, one more row per eye,
    the in-between corrective, is appended; without it (VRM files) the closed pose keeps the
    straight line outside the eye (:func:`clear_line`). Returns (offsets,
    :class:`EyeBlinks`), or (offsets unchanged, None) when the file has no blink target or no
    lid margins. Raises ValueError for poses of another mesh than ``rest``'s.
    """
    rest = np.asarray(rest, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    names = list(names)
    if poses:
        from opensculptboy.eyes.poses import check_rows

        check_rows(poses, len(rest))
    lids = eyed_lids(eyed, rest)
    surfaces = eye_surfaces(eyed)
    triangles = eyed.triangles[: int(eyed.kept.sum())]
    rows, shapes, used = {}, {}, {}
    out = offsets.copy()
    for side, name in BLINK_TARGETS.items():
        if name not in names or side not in lids or side not in surfaces:
            continue
        row = names.index(name)
        pose = poses.get(side) if poses else None
        if pose is not None:
            shape = reference_blink(
                lids[side], rest, pose, surfaces[side], triangles, not inbetween
            )
            used[side] = pose
        else:
            shape = rig_blink(
                lids[side], rest, offsets[row], surfaces[side], triangles, not inbetween
            )
        # the slivers folded at rest keep their shape
        moving = np.flatnonzero(np.linalg.norm(shape.closed, axis=1) > 1e-9)
        half = steady_folds(
            rest, triangles, 0.5 * shape.closed + shape.inbetween, moving
        )
        shape.closed = steady_folds(rest, triangles, shape.closed, moving)
        shape.inbetween = half - 0.5 * shape.closed
        out[row] = shape.closed
        rows[side], shapes[side] = row, shape
    if not rows:
        return offsets, None
    blinks = EyeBlinks(rows, shapes, lids, surfaces, poses=used)
    if inbetween:
        extra = []
        for side in rows:
            blinks.inbetween_rows[side] = len(out) + len(extra)
            extra.append(shapes[side].inbetween)
        out = np.concatenate([out, np.stack(extra)])
    blinks.extras = {
        BLINK_TARGETS[side]: dict(
            shapes[side].stats,
            **(
                {"inbetween": BLINK_TARGETS[side] + INBETWEEN_SUFFIX}
                if side in blinks.inbetween_rows
                else {}
            ),
        )
        for side in rows
    }
    return out, blinks


def lash_rows(eyed, side):
    """The lash parts of an eye of an eyed mesh: [(part, slice of its rows in the mesh)]."""
    start = eyed.base_count
    out = []
    for p in eyed.parts:
        if p.name == "lashes" and p.side == side:
            out.append((p, slice(start, start + len(p.rest))))
        start += len(p.rest)
    return out


def lash_bases(part) -> np.ndarray:
    """The base (M,) of each lash point of a part: the point of its own lash (a connected
    piece of the part) that lies nearest the point's root on the lid margin at rest, or that
    point's own base, so that every base is its own base."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    points = np.asarray(part.rest, dtype=np.float64)
    tris = np.asarray(part.triangles, dtype=np.int64)
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]]])
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(points), len(points)),
    )
    label = connected_components(graph, directed=False)[1]
    roots = part.follow.root @ part.follow.rest
    out = np.arange(len(points))
    for piece in np.unique(label):
        mine = np.flatnonzero(label == piece)
        _, near = cKDTree(points[mine]).query(roots[mine])
        out[mine] = mine[near]
    for _ in range(16):
        step = out[out]
        if (step == out).all():
            break
        out = step
    loose = np.flatnonzero(out[out] != out)
    out[loose] = loose
    return out


def on_margin(offsets, base, margin) -> np.ndarray:
    """Lash offsets (M, 3) of a reference's pose moved so that each lash's base (``base``,
    :func:`lash_bases`) moves as the lid margin under it (``margin``, (M, 3), the offset of
    each point's root): each lash keeps the reference's turn about its base. A fitted lid
    lies a little off its reference's, and its closed pose is brought to meet
    (:func:`meet_rims`), so the reference's own lash roots leave the lid."""
    offsets = np.asarray(offsets, dtype=np.float64)
    return offsets - offsets[base] + np.asarray(margin)[base]


# the upper lid's lashes (opensculptboy.eyes.lashes.UPPER)
UPPER_LID = 0
# the weights of a blink at which the upper lashes are kept in front of the lids, in at most
# FRONT_ROUNDS rounds (:func:`lashes_in_front`)
FRONT_CHECKS = (0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)
FRONT_ROUNDS = 4


def played(t: float, inbetween: bool) -> tuple[float, float]:
    """The weights of the half-closed pose and of the closed pose (absolute offsets) in a
    blink at ``t`` as a file plays it: with the in-between corrective, the path from rest
    through the half-closed pose to the closed pose; without it, the straight line."""
    if not inbetween:
        return 0.0, t
    tent = 1.0 - abs(2.0 * t - 1.0)
    return tent, t - 0.5 * tent


def lashes_in_front(part, rest, lid, lash, inbetween: bool):
    """
    A lash part's blink offsets (closed (M, 3), in-between corrective (M, 3)) with its
    points kept in front of the outward-facing skin around the eye at each weight of
    :data:`FRONT_CHECKS` as a file plays them (:func:`played`): as far from it as the
    lashes' binding keeps them (:data:`opensculptboy.eyes.lashes.CLEARANCE`, or their
    distance at rest when that is less), measured as :func:`skin_side` measures it. ``lid``
    holds the lids' closed offsets (N, 3) and their in-between corrective. A point that the
    closing lid would cover moves out along the skin's normal, and the half-closed pose and
    the closed pose share its push by their weights there. A reference's lashes keep its
    turn about their bases, and the card's lid can bulge over them where it lies off the
    reference's.
    """
    from opensculptboy.eyes.lashes import CLEARANCE

    follow = part.follow
    closed, between = lash
    if follow is None or follow.skin_rows is None or follow.skin_triangles is None:
        return closed, between
    rest = np.asarray(rest, dtype=np.float64)
    skin_tris = follow.skin_triangles
    floor = np.minimum(
        CLEARANCE, skin_side(part.rest, rest[follow.skin_rows], skin_tris)[0]
    )
    lid_closed, lid_between = (np.asarray(v, dtype=np.float64) for v in lid)
    skin_rows = follow.skin_rows
    lid_c = lid_closed[skin_rows]
    lid_h = (lid_between + 0.5 * lid_closed)[skin_rows]
    C = np.array(closed, dtype=np.float64, copy=True)
    H = np.array(between, dtype=np.float64, copy=True) + 0.5 * C
    for _ in range(FRONT_ROUNDS):
        moved = False
        for t in FRONT_CHECKS:
            a, b = played(t, inbetween)
            skin = rest[skin_rows] + a * lid_h + b * lid_c
            points = part.rest + a * H + b * C
            depth, normals, _ = skin_side(points, skin, skin_tris)
            need = np.clip(floor - depth, 0.0, None)
            if need.max(initial=0.0) <= 1e-7:
                continue
            moved = True
            share = need / (a * a + b * b)
            H += (share * a)[:, None] * normals
            C += (share * b)[:, None] * normals
        if not moved:
            break
    return C, H - 0.5 * C


def blink_lash_offsets(eyed, blinks: EyeBlinks, rest, targets):
    """
    The eyed mesh's targets (T, N + M, 3) with the lashes of each blink target and of its
    in-between corrective set: the reference's lashes under the reference's own blink when
    its :class:`~opensculptboy.eyes.poses.BlinkPose` holds them, each lash turned about its
    base as the reference's and its base moved with the card's lid margin
    (:func:`on_margin`), so the lids and the lashes close as one piece; else
    :func:`rig_lashes`. The other rows stay as
    :meth:`~opensculptboy.eyes.mesh.EyedMesh.targets` gave them. Records the measures of each
    eye's blink in ``blinks.extras``.
    """
    rest = np.asarray(rest, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64, copy=True)
    for side, row in blinks.rows.items():
        shape = blinks.shapes[side]
        frame = blinks.lids[side].frame
        pose = blinks.poses.get(side)
        parts = lash_rows(eyed, side)
        lash = None
        if parts:
            l0, l1, lc, upper = [], [], [], []
            for p, rows in parts:
                own = (
                    pose is not None
                    and pose.lash_closed is not None
                    and p.points is not None
                    and len(pose.lash_closed) > int(p.points.max())
                )
                if own:
                    # each lash keeps the reference's own turn about its base, and the
                    # base goes with the lid margin under it
                    base = lash_bases(p)
                    rows_f = p.follow.rows
                    closed = on_margin(
                        pose.lash_closed[p.points],
                        base,
                        p.follow.root @ shape.closed[rows_f],
                    )
                    # half way: the reference's half-closed lashes, else half its turn
                    # about the base; the base rides the margin of the lid's in-between
                    now = 0.5 * shape.closed + shape.inbetween
                    lash_half = (
                        pose.lash_half[p.points]
                        if pose.lash_half is not None
                        else 0.5 * pose.lash_closed[p.points]
                    )
                    half = on_margin(lash_half, base, p.follow.root @ now[rows_f])
                    between = half - 0.5 * closed
                else:
                    closed, between = rig_lashes(
                        p, frame, rest, shape.closed, shape.inbetween, shape.turn
                    )
                if p.lid == UPPER_LID:
                    # the upper lashes stay in front of the lids at every weight
                    closed, between = lashes_in_front(
                        p,
                        rest,
                        (shape.closed, shape.inbetween),
                        (closed, between),
                        side in blinks.inbetween_rows,
                    )
                targets[row, rows] = closed
                if side in blinks.inbetween_rows:
                    targets[blinks.inbetween_rows[side], rows] = between
                l0.append(p.rest)
                l1.append(p.rest + closed)
                lc.append(between)
                upper.append(np.full(len(p.rest), p.lid == 0))
            lash = (
                np.concatenate(l0),
                np.concatenate(l1),
                np.concatenate(lc) if side in blinks.inbetween_rows else None,
                np.concatenate(upper),
            )
        measures = measure_blink(
            blinks.lids[side],
            rest,
            rest + shape.closed,
            blinks.surfaces[side],
            lash,
            shape.inbetween if side in blinks.inbetween_rows else None,
        )
        blinks.extras[BLINK_TARGETS[side]]["measures"] = measures
    return targets


def blink_targets(blinks: EyeBlinks) -> list:
    """The morph targets (:class:`~opensculptboy.export.body.MorphTarget`) of the in-between
    correctives, in the order of their rows."""
    from opensculptboy.export.body import MorphTarget

    rows = sorted(blinks.inbetween_rows.items(), key=lambda kv: kv[1])
    return [
        MorphTarget(BLINK_TARGETS[side] + INBETWEEN_SUFFIX, "corrective", -1, 0.0, 0.0)
        for side, _ in rows
    ]


# the GLB animations of a blink: each eye's, and both eyes together
CLOSE_SUFFIX = ".close"
BOTH_EYES = "eyeBlink"


def blink_animations(blinks: EyeBlinks, targets) -> list:
    """
    The GLB animations of the blinks: ``<target>.close`` for each eye and ``eyeBlink.close``
    for both, over one second: the blink target goes from 0 to 1 and its in-between
    corrective from 0 to 1 half way and back to 0, so that the half-way point lies on the path
    around the eye. The other targets keep their weights. [(name, (times (K,), weights (K,
    T)))].
    """
    names = [t.name for t in targets]
    base = [float(t.weight) for t in targets]
    sets = [(BLINK_TARGETS[s] + CLOSE_SUFFIX, [s]) for s in sorted(blinks.rows)]
    if len(blinks.rows) == 2:
        sets.append((BOTH_EYES + CLOSE_SUFFIX, sorted(blinks.rows)))
    times = np.array([0.0, 0.5, 1.0])
    out = []
    for name, sides in sets:
        weights = np.array([base] * len(times))
        for side in sides:
            weights[:, names.index(BLINK_TARGETS[side])] = times
            corrective = BLINK_TARGETS[side] + INBETWEEN_SUFFIX
            if corrective in names:
                weights[:, names.index(corrective)] = [0.0, 1.0, 0.0]
        out.append((name, (times, weights)))
    return out

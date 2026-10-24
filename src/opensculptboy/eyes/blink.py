# OpenSculptBoy
# Apache License, Version 2.0
"""
Closing the eyes: the blink targets of the exported eyes (docs/eyes.md, "Blinks").

Anny closes each eye with one facial action (``eyeBlinkLeft``, ``eyeBlinkRight``), a fixed
offset per vertex. Played as a morph target, every vertex moves along a straight line, so the
lids cut into the round eyeball half way, and the upper lid closes a little past the lower lid.
A blink of an animated face follows the eyeball: the lids turn about the eye's centre, the upper
lid travels most of the way, and both lid margins meet along one line.

:func:`shape_blink` builds the closed pose of one eye from a closed pose that a card gives (a
reference's own closed eye carried to the card, :mod:`opensculptboy.eyes.poses`, else anny's
facial action) and the eye's outer surface:

- the seal: both lid margins move onto one line, where the closed margins meet (by default
  :data:`SEAL_SHARE` of the way from the upper margin to the lower one), and the lid skin
  around them follows smoothly (the lower lid's within :data:`LOWER_SPREAD` of its margin);
  the skin of the upper lid stays above that line, the skin of the lower lid below it;
- the clearance: every lid vertex that lies outside the eye at rest stays
  :data:`LID_CLEARANCE` outside its outer surface (the cornea's bulge included) in the closed
  pose and all along the straight line to it, so a morph target alone (VRM files, and any
  application that sets the weight) never cuts into the eye;
- the in-between: the half-closed pose on the path around the eye (the lid's angle about the
  eye's horizontal axis and its distance from the eye's centre blended, outside the eye's
  surface), which GLB files carry as a corrective target (:data:`INBETWEEN_SUFFIX`) that an
  animation raises from 0 at the ends to 1 half way.

The measures (:func:`measure_blink`) give, at each stage of a blink, how far the lids cross
each other (:func:`lid_crossing`), how deep lid skin lies inside the eye
(:func:`eye_penetration`), how deep lash points lie inside the lids (:func:`lash_penetration`)
and how far the upper lashes of a closed eye lie under the lower ones (:func:`lash_order`).
"""

from __future__ import annotations

import dataclasses

import numpy as np

# the stages of a blink that the measures and the lashes' correction look at
STAGES = (0.0, 0.25, 0.5, 0.75, 1.0)
# the lids stay this far (metres) outside the eye's outer surface
LID_CLEARANCE = 0.0001
# the margins of a closed eye meet this share of the way from the closed upper margin to the
# closed lower margin (angles about the eye's horizontal axis): the upper lid travels most of
# the way and comes down onto the lower lid
SEAL_SHARE = 0.85
# the straight line of a morph target is checked at these shares of the way to the closed pose
LINE_CHECKS = (0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)
# the largest push (metres) out of the eye that the clearance gives a lid vertex
MAX_PUSH = 0.004
# the lid vertices that the shaping moves: within this many eyeball radii of the eye's centre,
# in front of the plane this far behind it (eyeball radii)
LID_REACH = 1.6
LID_FRONT = -0.35
# the skin of each lid stays this far (metres, along the eye's surface) on its side of the
# seal, its margin's rows apart: two sheets of skin that meet at one line and lie on it
# side by side would cut through each other here and there
ORDER_GAP = 0.00015
# the seal's correction spreads over the lid with this many smoothing passes
SMOOTH_PASSES = 4
# the seal's correction of the lower margin fades out within this distance (metres) of the
# margin at rest: the lower lid presses forward where the upper lid meets it, and the cheek
# below keeps the closed pose's own motion
LOWER_SPREAD = 0.004
# a lash point may lie this far (metres) inside the lids beyond its depth at rest before it
# counts in the measures; the correction keeps the lashes LASH_CLEARANCE outside
LASH_TOLERANCE = 0.00005
LASH_CLEARANCE = 0.0001
# a closed eye's upper lashes lie at least this far (metres) outside the lower lashes where
# both cover the same spot, seen from the eye's centre within LASH_ORDER_REACH
LASH_ORDER_GAP = 0.00015
LASH_ORDER_REACH = 0.0008
# the in-between's path around the eye: in full for lid vertices at least ARC_FULL eyeball
# radii from the eye's horizontal axis, none within ARC_FROM, and at most ARC_SHARE of the
# vertex's motion away from the straight line (a turn of 60 degrees bows its path by 0.15
# of the chord)
ARC_FROM = 0.3
ARC_FULL = 0.7
ARC_SHARE = 0.3
# the in-between corrective of a blink target in GLB files
INBETWEEN_SUFFIX = ".inbetween"
# a margin vertex within this distance (eyeball radii along the eye's horizontal axis) of
# another that lies nearer the opening is left out of the margin's line
MARGIN_BIN = 0.025
# a triangle pressed to less than this share of its area at rest is flat, and does not count
# as folded
FLAT = 0.05
# the shaping relaxes the rows of folded triangles this many times
FOLD_PASSES = 8
# a lid point that lies closer than this (metres) to the eye at rest already touches it: the
# shaping and the measures leave it aside
REST_CONTACT = 0.00005
# the corners of the eye: the crossing and the clipping measures leaves out the ends of the margins within this
# distance (metres) along the eye's horizontal axis
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


def folded_triangles(lids: EyeLids, positions: np.ndarray, rest: np.ndarray) -> int:
    """The number of the lids' triangles (the skin of the region around the eye, the
    socket's included) that fold over: whose normal turns away from the mean of its corners'
    normals, beyond those folded at rest."""

    tris = lids.region_triangles
    if tris is None or len(tris) == 0:
        return 0
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
            "folded_triangles": folded_triangles(lids, x, rest),
            "eye_mm": round(depth, 3),
            "eye_points": n_depth,
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


# ------------------------------------------------------------------------------- shaping
@dataclasses.dataclass
class BlinkShape:
    """One eye's blink: the closed pose's offsets (N, 3) for the eye's blink target, the
    in-between corrective (N, 3) that takes the half-way point of the target to the half-closed
    pose on the path around the eye, and what the shaping did (``stats``)."""

    closed: np.ndarray
    inbetween: np.ndarray
    stats: dict = dataclasses.field(default_factory=dict)


def _radial(frame, points):
    d = np.asarray(points) - frame.centre
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)


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

    def worst(push):
        end = x1 + push[:, None] * n
        pts = x0[None] + ts[:, None, None] * (end - x0)[None]
        d = surface.depth(pts.reshape(-1, 3)).reshape(len(ts), len(x0)) + clearance
        return d.max(0)

    need = worst(np.zeros(len(x0))) > 1e-7
    out = np.zeros(len(x0))
    if not need.any():
        return out
    lo = np.zeros(int(need.sum()))
    hi = np.full(int(need.sum()), MAX_PUSH)
    sub0, sub1 = x0[need], x1[need]
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        end = sub1 + mid[:, None] * n[need]
        pts = sub0[None] + ts[:, None, None] * (end - sub0)[None]
        d = surface.depth(pts.reshape(-1, 3)).reshape(len(ts), len(sub0)) + clearance
        ok = d.max(0) <= 1e-7
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    out[need] = hi
    return out


def _smooth_along(values, passes=3):
    out = np.asarray(values, dtype=np.float64).copy()
    for _ in range(passes):
        if len(out) > 2:
            out[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
    return out


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


def _untangle(frame, sealed, n_upper, passes=24):
    """The sealed margins (the upper margin's rows, then the lower margin's) with each lid's
    rows moved apart where the polylines between them cross: the two margins hold different
    rows, so the straight edges between the rows of one cross the other's a little."""
    sealed = sealed.copy()
    for _ in range(passes):
        s, phi, rho = cylinder(frame, sealed)
        su, pu = s[:n_upper], phi[:n_upper]
        sl, pl = s[n_upper:], phi[n_upper:]
        ou, ol = np.argsort(su), np.argsort(sl)
        # the crossing at each row: the other margin's polyline above it (upper rows) or
        # below it (lower rows)
        cu = np.interp(su, sl[ol], pl[ol]) - pu
        cl = pl - np.interp(sl, su[ou], pu[ou])
        cu[(su <= sl.min()) | (su >= sl.max())] = 0.0
        cl[(sl <= su.min()) | (sl >= su.max())] = 0.0
        if max(cu.max(initial=0.0), cl.max(initial=0.0)) <= 1e-6:
            break
        phi[:n_upper] += 0.5 * np.clip(cu, 0.0, None)
        phi[n_upper:] -= 0.5 * np.clip(cl, 0.0, None)
        sealed = from_cylinder(frame, s, phi, rho)
    return sealed


def shape_blink(
    lids: EyeLids,
    rest: np.ndarray,
    offsets: np.ndarray,
    triangles: np.ndarray,
    surface: EyeSurface,
    seal_share: float = SEAL_SHARE,
    clearance: float = LID_CLEARANCE,
) -> BlinkShape:
    """
    The :class:`BlinkShape` of one eye from a closed pose (``offsets`` (N, 3) from ``rest``
    (N, 3), on a welded mesh with ``triangles``): the margins sealed along one line, the lids'
    skin kept on its side of it and outside the eye's ``surface`` along the straight line to
    the closed pose, and the in-between on the path around the eye (module docstring). Only
    the rows of ``lids.region`` change.
    """
    from scipy.spatial import cKDTree

    from opensculptboy.fit.register import smooth_mouth

    frame = lids.frame
    R = frame.radius
    X0 = np.asarray(rest, dtype=np.float64)
    X1 = X0 + np.asarray(offsets, dtype=np.float64)
    region = np.zeros(len(X0), bool)
    region[lids.region] = True
    # the rims: the margin rows at the edge toward the opening (a margin can hold more than
    # one row; the rows behind the rim follow as the lid's lining does)
    rim_upper = _rim(lids, X0, lids.upper, True)
    rim_lower = _rim(lids, X0, lids.lower, False)
    margins = np.r_[rim_upper, rim_lower]
    rows = np.flatnonzero(region)
    outside = np.zeros(len(X0), bool)
    outside[rows] = surface.depth(X0[rows]) <= -REST_CONTACT

    # the seal: where the closed margins meet, along the eye's horizontal axis
    s_u, p_u = _margin_curve(lids, X1, lids.upper, True)
    s_l, p_l = _margin_curve(lids, X1, lids.lower, False)
    grid = np.linspace(min(s_u[0], s_l[0]), max(s_u[-1], s_l[-1]), 48)
    phi_u, phi_l = np.interp(grid, s_u, p_u), np.interp(grid, s_l, p_l)
    seal_phi = _smooth_along(phi_u + seal_share * (phi_l - phi_u))
    _, _, rho_m = cylinder(frame, X1[margins])
    s_m, _, _ = cylinder(frame, X1[margins])
    order = np.argsort(s_m)
    seal_rho = _smooth_along(np.interp(grid, s_m[order], rho_m[order]), 4)

    def place(rho_extra):
        s, _, _ = cylinder(frame, X1[margins])
        return from_cylinder(
            frame,
            s,
            np.interp(s, grid, seal_phi),
            np.interp(s, grid, seal_rho + rho_extra),
        )

    extra = np.zeros_like(grid)
    for _ in range(
        3
    ):  # the seal's distance from the eye: clear along the straight line
        new = place(extra)
        push = line_push(X0[margins], new, surface, frame, clearance)
        if push.max(initial=0.0) <= 1e-7:
            break
        s, _, _ = cylinder(frame, new)
        bins = np.clip(np.searchsorted(grid, s), 0, len(grid) - 1)
        need = np.zeros_like(grid)
        np.maximum.at(need, bins, push / R)
        for k in range(len(grid)):  # each bin takes its neighbours' need too
            need[k] = need[max(k - 1, 0) : k + 2].max()
        extra += _smooth_along(need, 2) * 1.05
    sealed = place(extra)
    sealed = _untangle(frame, sealed, len(rim_upper))

    # the seal's correction spreads over the lids (their skin near the margins and their
    # lining), and the skin beyond keeps the closed pose's own motion
    delta = np.zeros_like(X0)
    delta[margins] = sealed - X1[margins]
    free = np.zeros(len(X0), bool)
    free[lids.upper_rows] = True
    lower_rows = np.asarray(lids.lower_rows)
    near = cKDTree(X0[lids.lower]).query(X0[lower_rows])[0] <= LOWER_SPREAD
    free[lower_rows[near]] = True
    free &= region
    free[margins] = False
    delta = smooth_mouth(delta, triangles, free)
    spread = X1 + delta

    # the outward skin of each lid keeps to its side (the lining behind the rims is free)
    upper = np.zeros(len(X0), bool)
    upper[lids.upper_skin] = True
    lower = np.zeros(len(X0), bool)
    lower[lids.lower_skin] = True
    upper[margins] = lower[margins] = False
    lo_s, hi_s = grid[0], grid[-1]
    lining = np.zeros(len(X0), bool)
    if lids.socket_rows is not None:
        lining[lids.socket_rows] = True
    # the sealed margins' lines: each lid's skin stays on its side of its own margin
    n_up = len(rim_upper)
    lines = {}
    for key, part in ((1.0, sealed[:n_up]), (-1.0, sealed[n_up:])):
        s, phi, _ = cylinder(frame, part)
        order = np.argsort(s)
        lines[key] = (s[order], phi[order])

    def clamp(x):
        x = x.copy()
        # each lid's skin on its side of the seal
        for mask, sense in ((upper, 1.0), (lower, -1.0)):
            r = np.flatnonzero(mask)
            s, phi, rho = cylinder(frame, x[r])
            line = np.interp(s, *lines[sense]) + sense * ORDER_GAP / (R * rho)
            bad = (s > lo_s) & (s < hi_s) & (np.abs(phi) < np.radians(80.0))
            bad &= sense * (phi - line) < 0.0
            if bad.any():
                x[r[bad]] = from_cylinder(frame, s[bad], line[bad], rho[bad])
        # outside the eye, all along the straight line from rest: the lids' outward skin
        # (the lining of the socket slides on the eye behind the margin, and a push out
        # would bring it in front of the margin)
        r = np.flatnonzero(outside & free & ~lining)
        push = line_push(X0[r], x[r], surface, frame, clearance)
        x[r] += push[:, None] * _radial(frame, x[r])
        return x

    A = neighbours(triangles, len(X0))
    x = clamp(spread)
    for _ in range(SMOOTH_PASSES):
        corr = x - spread
        smooth = 0.5 * corr + 0.5 * (A @ corr)
        corr[free] = smooth[free]
        x = clamp(spread + corr)
    # the triangles that fold over, closed or on the way: their rows take their neighbours'
    # mean motion
    tris = triangles[region[triangles].all(1)]
    for _ in range(FOLD_PASSES):
        rows_bad = np.zeros(len(X0), bool)
        for t in (1.0, 0.75, 0.5, 0.25):  # on the straight line too
            bad = _folded(tris, X0 + t * (x - X0), X0, lids)
            rows_bad[tris[bad].ravel()] = True
        if not rows_bad.any():
            break
        rows_bad &= free
        mean = A @ (x - X0)  # the neighbours' mean motion: the rest shape stays
        x[rows_bad] = X0[rows_bad] + mean[rows_bad]
        x = clamp(x)
    closed = x - X0

    # the in-between: half way along the path around the eye
    moving = region & (np.linalg.norm(closed, axis=1) > 1e-6)
    r = np.flatnonzero(moving)
    s0, p0, q0 = cylinder(frame, X0[r])
    s1, p1, q1 = cylinder(frame, x[r])
    turn = np.angle(np.exp(1j * (p1 - p0)))
    arc = from_cylinder(frame, 0.5 * (s0 + s1), p0 + 0.5 * turn, 0.5 * (q0 + q1))
    # near the eye's horizontal axis the angle about it says little: the path there keeps
    # to the straight line; and the path keeps within a share of the motion of the line
    mid = 0.5 * (X0[r] + x[r])
    near_axis = np.clip((np.minimum(q0, q1) - ARC_FROM) / (ARC_FULL - ARC_FROM), 0, 1)
    bow = near_axis[:, None] * (arc - mid)
    size = np.linalg.norm(bow, axis=1)
    cap = ARC_SHARE * np.linalg.norm(x[r] - X0[r], axis=1)
    bow *= np.minimum(1.0, cap / np.maximum(size, 1e-12))[:, None]
    half = mid + bow
    clear = outside[r]
    if lids.socket_rows is not None:
        clear &= ~np.isin(r, lids.socket_rows)
    if clear.any():
        depth = surface.depth(half[clear]) + clearance
        push = np.clip(depth, 0.0, MAX_PUSH)
        half[clear] += push[:, None] * _radial(frame, half[clear])
    inbetween = np.zeros_like(X0)
    inbetween[r] = half - 0.5 * (X0[r] + x[r])
    stats = {
        "seal_share": seal_share,
        "seal_lift_mm": round(float(extra.max() * R * 1000.0), 3),
        "margin_shift_mm": round(
            float(np.linalg.norm(delta[margins], axis=1).max() * 1000), 3
        ),
        "moved_rows": int(moving.sum()),
        "inbetween_max_mm": round(
            float(np.linalg.norm(inbetween, axis=1).max() * 1000), 3
        ),
    }
    return BlinkShape(closed, inbetween, stats)


# -------------------------------------------------------------------------------- lashes
def fix_lashes(lids, rest, closed, inbetween, l0, l1, lh, upper, rounds: int = 3):
    """
    One eye's lashes in its blink: ``l0`` (M, 3) at rest, ``l1`` in the closed pose and ``lh``
    in the half-closed pose of the path around the eye (the mesh at ``rest + closed / 2 +
    inbetween``), ``upper`` the mask of the upper lashes. Returns the closed lashes and the
    lashes' in-between corrective, (M, 3) each, with the upper lashes of the closed eye
    :data:`LASH_ORDER_GAP` outside the lower ones (:func:`lash_order`), and every point
    :data:`LASH_CLEARANCE` outside the lids' outward-facing skin (or as deep as at rest) at
    the stages of a blink, along the straight line to the closed pose and along the path with
    the in-between.
    """
    from scipy.spatial import cKDTree

    rest = np.asarray(rest, dtype=np.float64)
    closed = np.asarray(closed, dtype=np.float64)
    l0, l1 = np.asarray(l0, dtype=np.float64), np.asarray(l1, dtype=np.float64).copy()
    upper = np.asarray(upper, bool)
    frame = lids.frame
    tris = lids.outer_triangles
    at_rest = skin_side(l0, rest, tris)[0]
    floor = np.minimum(at_rest, 0.0) + LASH_CLEARANCE

    def push_out(points, x):
        """The move (M, 3) that takes each point below its floor out to it, along the skin's
        normal at its closest point (a root that sits in the skin at rest moves only by
        what it lies deeper than at rest)."""
        signed, n, _ = skin_side(points, x, tris)
        need = np.clip(floor - signed, 0.0, None)
        return need[:, None] * n

    def order(l1):
        if not upper.any() or upper.all():
            return l1
        cu = l1[upper] - frame.centre
        cl = l1[~upper] - frame.centre
        ru, rl = np.linalg.norm(cu, axis=1), np.linalg.norm(cl, axis=1)
        # a wider reach than the measure's: the measure's eye centre may lie a little apart
        reach = 1.5 * LASH_ORDER_REACH / max(float(np.median(ru)), 1e-9)
        tree = cKDTree(cl / rl[:, None])
        need = np.zeros(len(cu))
        for i, near in enumerate(tree.query_ball_point(cu / ru[:, None], reach)):
            if near:
                need[i] = max(0.0, rl[near].max() + LASH_ORDER_GAP - ru[i])
        out = l1.copy()
        out[upper] += need[:, None] * cu / ru[:, None]
        return out

    for _ in range(rounds):
        l1 = order(l1)
        for t in (1.0, 0.75, 0.5, 0.25):
            x = rest + t * closed
            l1 += push_out(l0 + t * (l1 - l0), x) / t
    # the push out of the lids can bring lower lashes over the upper ones: the order last
    # (it moves the upper lashes away from the eye, out of the lids)
    l1 = order(l1)
    lc = np.asarray(lh, dtype=np.float64) - 0.5 * (l0 + l1)
    for _ in range(rounds):
        for t in (0.5, 0.25, 0.75):
            tent = 1.0 - abs(2.0 * t - 1.0)
            x = rest + t * closed + tent * inbetween
            lc += push_out(l0 + t * (l1 - l0) + tent * lc, x) / tent
    return l1, lc


# ---------------------------------------------------------------------- the files' targets
# the blink target of each eye's side (the figure's left eye is +X)
BLINK_TARGETS = {"L": "eyeBlinkLeft", "R": "eyeBlinkRight"}


@dataclasses.dataclass
class EyeBlinks:
    """The blinks of a file's eyes: per side, the row of its blink target among the file's
    targets, its :class:`BlinkShape`, its lids and its eye's surface; the rows of the
    in-between correctives appended after the file's targets (side to row); and the scene
    extras that record them."""

    rows: dict
    shapes: dict
    lids: dict
    surfaces: dict
    inbetween_rows: dict = dataclasses.field(default_factory=dict)
    extras: dict = dataclasses.field(default_factory=dict)


def blink_offsets(eyed, rest, offsets, names, closed=None, inbetween: bool = True):
    """
    A file's morph-target offsets (T, N, 3) of its base mesh (``rest`` (N, 3), the targets'
    ``names``) with each eye's blink target (:data:`BLINK_TARGETS`) shaped by
    :func:`shape_blink`, from ``closed`` (side to the closed pose's offsets (N, 3), a
    reference's) or else from the target itself; with ``inbetween``, one more row per eye, the
    in-between corrective, appended. Returns (offsets, :class:`EyeBlinks`), or (offsets
    unchanged, None) when the file has no blink target or no lid margins.
    """
    rest = np.asarray(rest, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    names = list(names)
    lids = eyed_lids(eyed, rest)
    surfaces = eye_surfaces(eyed)
    triangles = eyed.triangles[: int(eyed.kept.sum())]
    rows, shapes = {}, {}
    out = offsets.copy()
    for side, name in BLINK_TARGETS.items():
        if name not in names or side not in lids or side not in surfaces:
            continue
        row = names.index(name)
        source = closed.get(side) if closed else None
        start = offsets[row] if source is None else np.asarray(source, np.float64)
        shape = shape_blink(lids[side], rest, start, triangles, surfaces[side])
        # the rows outside the eye's lids keep the closed pose as it came
        keep = np.ones(len(rest), bool)
        keep[lids[side].region] = False
        shape.closed[keep] = start[keep]
        out[row] = shape.closed
        rows[side], shapes[side] = row, shape
    if not rows:
        return offsets, None
    blinks = EyeBlinks(rows, shapes, lids, surfaces)
    if inbetween:
        extra = []
        for side in rows:
            blinks.inbetween_rows[side] = len(out) + len(extra)
            extra.append(shapes[side].inbetween)
        out = np.concatenate([out, np.stack(extra)])
    blinks.extras = {
        BLINK_TARGETS[side]: dict(
            shapes[side].stats,
            source="reference" if closed and closed.get(side) is not None else "target",
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


def blink_lash_offsets(eyed, blinks: EyeBlinks, rest, targets):
    """
    The eyed mesh's targets (T, N + M, 3) with the lashes of each blink target and its
    in-between corrective set by :func:`fix_lashes`; the other rows stay as
    :meth:`~opensculptboy.eyes.mesh.EyedMesh.targets` gave them. Records the measures of each
    eye's blink in ``blinks.extras``.
    """
    rest = np.asarray(rest, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64, copy=True)
    for side, row in blinks.rows.items():
        shape = blinks.shapes[side]
        parts = lash_rows(eyed, side)
        half_x = rest + 0.5 * shape.closed + shape.inbetween
        lash = None
        if parts:
            l0 = np.concatenate([p.positions(rest) for p, _ in parts])
            l1 = np.concatenate([p.positions(rest + shape.closed) for p, _ in parts])
            lh = np.concatenate([p.positions(half_x) for p, _ in parts])
            upper = np.concatenate([np.full(len(p.rest), p.lid == 0) for p, _ in parts])
            l1, lc = fix_lashes(
                blinks.lids[side],
                rest,
                shape.closed,
                shape.inbetween,
                l0,
                l1,
                lh,
                upper,
            )
            k = 0
            for p, rows in parts:
                n = len(p.rest)
                targets[row, rows] = l1[k : k + n] - p.rest
                if side in blinks.inbetween_rows:
                    targets[blinks.inbetween_rows[side], rows] = lc[k : k + n]
                k += n
            lash = (l0, l1, lc if side in blinks.inbetween_rows else None, upper)
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

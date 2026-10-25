# OpenSculptBoy
# Apache License, Version 2.0
"""
Mesh reduction by half-edge collapses with quadric error metrics.

:func:`decimate` removes vertices of a triangle mesh until it holds a target number of
triangles. Each collapse moves a vertex ``u`` onto a neighbour ``v`` and drops the triangles of
the edge, so the reduced mesh keeps a subset of the input vertices, each with its own position,
normal, UVs, skin weights, morph target offsets and wearable binding. Nothing is interpolated,
so a reduced piece of a VRM file keeps the inverse bind of every vertex it keeps.

The mesh is the vertex data of a glTF primitive: vertices split at UV seams and hard edges.
:func:`weld` joins the vertices at one position, and the collapses run on the welded surface
(after Garland and Heckbert 1997, with the vertex classes of meshoptimizer):

- an edge is a *feature* where the surface is open (one triangle), where the triangles on its
  two sides use other vertices of one corner (a UV seam or a hard edge), or other labels (a
  material border, or another budget group);
- a vertex with no feature edge collapses onto any neighbour; a vertex on one feature line (two
  feature edges) collapses only along that line, so that hems, seams and material borders keep
  their shape and the texture its layout; every other vertex stays (an end or a crossing of
  feature lines, or a vertex on a non-manifold edge);
- the cost of a collapse is the area-weighted quadric of the planes around both vertices at the
  kept vertex (the squared distance times the area that moves), in the rest shape and in each
  of the given morph target shapes, scaled by the removed vertex's importance; feature edges
  add planes along their lines;
- a collapse is refused when it breaks the link condition (it would join two sheets), turns a
  triangle by more than 60 degrees, flips or collapses a UV triangle, or makes a sliver.

Each pass takes the cheapest collapse of every vertex up to the median of those costs, then
in a few rounds the vertices whose collapse is the cheapest within two rings, and applies them
at once: their neighbourhoods do not overlap, so the checks of each still hold (after the
multiple-choice scheme of Wu and Kobbelt 2002). Budget groups (a piece of a file) keep at
least their floor of triangles.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# the weight of the planes along feature edges, per square metre of edge: open edges hold the
# silhouette of a hem; seams and material borders hold the texture's layout
BORDER_WEIGHT = 16.0
SEAM_WEIGHT = 4.0
# a collapse may turn a triangle's normal by at most this angle (cosine)
FLIP_COS = 0.5
# the normal term: a turn of 90 degrees between the normals of the two vertices costs as much
# as a move of this distance (metres) over the vertex's area
NORMAL_LENGTH = 0.003
# a mild preference for short edges, which keeps the triangles even on flat panels: an edge of
# length L costs as much as a distance of sqrt(LENGTH_WEIGHT) L over the vertex's area
LENGTH_WEIGHT = 1e-3
# each pass takes collapses up to this quantile of the cheapest collapse of every vertex, in
# rounds of independent sets
PASS_QUANTILE = 0.5
MIS_ROUNDS = 4


@dataclasses.dataclass
class Reduction:
    """
    The result of :func:`decimate`.

    ``triangles`` index the input vertices, and ``kept`` marks the vertices they use;
    ``labels`` and ``groups`` give each triangle's label and budget group, and ``origin`` the
    input triangle it comes from. ``error`` holds, per
    budget group, the largest RMS distance (metres) from the planes of the removed vertices to
    the kept vertex they moved onto, ``removed`` the vertices removed per group, and
    ``stopped`` why the reduction stopped (``target`` when it reached it).
    """

    triangles: np.ndarray
    labels: np.ndarray
    origin: np.ndarray
    groups: np.ndarray
    kept: np.ndarray
    error: dict
    removed: dict
    passes: int
    dropped_degenerate: int
    stopped: str = "target"

    def counts(self) -> dict:
        """Triangles per budget group."""
        values, number = np.unique(self.groups, return_counts=True)
        return {int(g): int(n) for g, n in zip(values, number)}


def weld(positions: np.ndarray, tolerance: float = 1e-5) -> np.ndarray:
    """
    (N,) the welded index of each vertex: vertices within about ``tolerance`` metres share
    one. Two grids, the second shifted by half a cell, catch the pairs that one grid splits.
    """
    P = np.asarray(positions, dtype=np.float64)
    n = len(P)
    if n == 0:
        return np.zeros(0, np.int64)
    a_parts, b_parts = [], []
    for shift in (0.0, 0.5):
        key = np.floor(P / tolerance + shift).astype(np.int64)
        _, inverse = np.unique(key, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        order = np.argsort(inverse, kind="stable")
        starts = np.searchsorted(inverse[order], np.arange(inverse.max() + 1))
        first = order[starts]
        a_parts.append(np.arange(n))
        b_parts.append(first[inverse])
    label = _components(n, np.concatenate(a_parts), np.concatenate(b_parts))
    _, ids = np.unique(label, return_inverse=True)
    return ids.ravel().astype(np.int64)


def _components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest member of the connected component of each of ``n`` nodes joined by the
    pairs (a, b)."""
    label = np.arange(n)
    while True:
        new = label.copy()
        np.minimum.at(new, a, label[b])
        np.minimum.at(new, b, label[a])
        while True:
            jumped = new[new]
            if np.array_equal(jumped, new):
                break
            new = jumped
        if np.array_equal(new, label):
            return label
        label = new


def _plane_quadrics(normal: np.ndarray, offset: np.ndarray, weight: np.ndarray):
    """(k, 10) quadrics of the planes ``normal . x + offset = 0``, scaled by ``weight``."""
    a, b, c = normal[:, 0], normal[:, 1], normal[:, 2]
    d = offset
    q = np.stack(
        [a * a, a * b, a * c, a * d, b * b, b * c, b * d, c * c, c * d, d * d], 1
    )
    return q * weight[:, None]


def _evaluate(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The quadrics (m, 10) at the points (m, 3)."""
    X, Y, Z = x[:, 0], x[:, 1], x[:, 2]
    return (
        q[:, 0] * X * X
        + 2 * q[:, 1] * X * Y
        + 2 * q[:, 2] * X * Z
        + 2 * q[:, 3] * X
        + q[:, 4] * Y * Y
        + 2 * q[:, 5] * Y * Z
        + 2 * q[:, 6] * Y
        + q[:, 7] * Z * Z
        + 2 * q[:, 8] * Z
        + q[:, 9]
    )


def _triangle_frame(P: np.ndarray, T: np.ndarray):
    """Unit normals, areas and plane offsets of triangles."""
    cross = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    length = np.linalg.norm(cross, axis=1)
    normal = cross / np.maximum(length, 1e-30)[:, None]
    offset = -(normal * P[T[:, 0]]).sum(1)
    return normal, 0.5 * length, offset


def _compactness(a, b, c):
    """4 sqrt(3) area / (sum of squared edge lengths): 1 for an equilateral triangle."""
    cross = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    edges = ((b - a) ** 2).sum(1) + ((c - b) ** 2).sum(1) + ((a - c) ** 2).sum(1)
    return 2.0 * np.sqrt(3.0) * cross / np.maximum(edges, 1e-30)


@dataclasses.dataclass
class _Edges:
    """The welded edges of the current triangles."""

    lo: np.ndarray
    hi: np.ndarray
    count: np.ndarray
    t0: np.ndarray  # first triangle
    t1: np.ndarray  # second triangle, -1 for none
    lo0: np.ndarray  # the vertex (input index) of lo in t0
    hi0: np.ndarray
    lo1: np.ndarray  # in t1, -1 for none
    hi1: np.ndarray
    feature: np.ndarray
    border: np.ndarray
    key: np.ndarray  # sorted lo * W + hi


def _edges(T, Tw, labels, W) -> _Edges:
    a = Tw[:, [0, 1, 2]].ravel()
    b = Tw[:, [1, 2, 0]].ravel()
    ca = T[:, [0, 1, 2]].ravel()
    cb = T[:, [1, 2, 0]].ravel()
    tri = np.repeat(np.arange(len(T)), 3)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    forward = a == lo  # the edge runs lo -> hi in this triangle
    key = lo * W + hi
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    unique, start, count = np.unique(key_sorted, return_index=True, return_counts=True)
    i0 = order[start]
    i1 = np.where(count >= 2, order[np.minimum(start + 1, len(order) - 1)], -1)
    lo_attr = np.where(forward, ca, cb)
    hi_attr = np.where(forward, cb, ca)
    has1 = i1 >= 0
    j1 = np.where(has1, i1, 0)
    t0, t1 = tri[i0], np.where(has1, tri[j1], -1)
    lo0, hi0 = lo_attr[i0], hi_attr[i0]
    lo1 = np.where(has1, lo_attr[j1], -1)
    hi1 = np.where(has1, hi_attr[j1], -1)
    two = count == 2
    # a manifold edge runs in opposite directions in its two triangles
    same_way = two & (forward[i0] == np.where(has1, forward[j1], ~forward[i0]))
    seam = two & ((lo0 != lo1) | (hi0 != hi1))
    other_label = two & (labels[t0] != labels[np.where(has1, t1, 0)])
    border = count == 1
    feature = border | seam | other_label | same_way | (count > 2)
    return _Edges(
        lo=lo[i0],
        hi=hi[i0],
        count=count,
        t0=t0,
        t1=t1,
        lo0=lo0,
        hi0=hi0,
        lo1=lo1,
        hi1=hi1,
        feature=feature,
        border=border,
        key=unique,
    )


def _csr(owner: np.ndarray, values: np.ndarray, n: int):
    """Offsets and values of a list per owner (0..n-1)."""
    order = np.argsort(owner, kind="stable")
    offsets = np.concatenate([[0], np.cumsum(np.bincount(owner, minlength=n))])
    return offsets, values[order]


def _common_neighbours(E: _Edges, W: int) -> np.ndarray:
    """The number of common neighbours of the two ends of each edge."""
    src = np.concatenate([E.lo, E.hi])
    dst = np.concatenate([E.hi, E.lo])
    offsets, nbr = _csr(src, dst, W)
    deg = np.diff(offsets)
    # every pair of neighbours (p, q) of each vertex, with p < q
    owner = np.repeat(np.arange(W), deg)
    repeat = deg[owner]
    rows = np.repeat(np.arange(len(nbr)), repeat)
    within = np.arange(len(rows)) - np.repeat(np.cumsum(repeat) - repeat, repeat)
    p = nbr[rows]
    q = nbr[offsets[owner[rows]] + within]
    keep = p < q
    pair_key = p[keep] * W + q[keep]
    position = np.minimum(np.searchsorted(E.key, pair_key), len(E.key) - 1)
    hit = E.key[position] == pair_key
    return np.bincount(position[hit], minlength=len(E.key))


def decimate(
    positions: np.ndarray,
    triangles: np.ndarray,
    target: int,
    labels: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    floors: dict | None = None,
    importance: np.ndarray | None = None,
    locked: np.ndarray | None = None,
    uvs: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    shapes=(),
    weld_tolerance: float = 1e-5,
    max_passes: int = 400,
) -> Reduction:
    """
    Reduce a triangle mesh to ``target`` triangles (or as near as its constraints allow).

    Args:
        positions: (N, 3) the vertices, split at seams as a glTF primitive holds them.
        triangles: (T, 3) indices into ``positions``.
        target: the number of triangles to keep, over every group.
        labels: (T,) a label per triangle (a material); borders between labels stay.
        groups: (T,) a budget group per triangle (a piece of a file); borders between groups
            stay, and ``floors`` gives the fewest triangles each group keeps (group -> count;
            a group at its count stays whole).
        importance: (N,) a factor on the cost of removing each vertex (1 by default).
        locked: (N,) vertices that stay.
        uvs: (N, 2) texture coordinates: a collapse keeps every UV triangle's orientation.
        normals: (N, 3) vertex normals, for the cost of a turn between the two vertices.
        shapes: (N, 3) offsets of morph target shapes, whose planes add to the cost.
        weld_tolerance: the distance (metres) within which vertices are one.
    """
    P = np.asarray(positions, dtype=np.float64)
    T = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    n = len(P)
    labels = (
        np.zeros(len(T), np.int64) if labels is None else np.asarray(labels, np.int64)
    )
    groups = (
        np.zeros(len(T), np.int64) if groups is None else np.asarray(groups, np.int64)
    )
    given = labels
    # a label per (label, group), so that group borders are features too
    _, labels = np.unique(np.stack([labels, groups], 1), axis=0, return_inverse=True)
    labels = labels.ravel()
    floors = dict(floors or {})
    wid = weld(P, weld_tolerance)
    W = int(wid.max(initial=-1)) + 1
    first = np.full(W, -1)
    first[wid[::-1]] = np.arange(n)[::-1]
    Pw = P[np.maximum(first, 0)]
    Tw = wid[T]
    good = (Tw[:, 0] != Tw[:, 1]) & (Tw[:, 1] != Tw[:, 2]) & (Tw[:, 2] != Tw[:, 0])
    dropped = int((~good).sum())
    T, Tw, labels, groups = T[good], Tw[good], labels[good], groups[good]
    given = given[good]
    origin = np.flatnonzero(good)
    imp = np.ones(W)
    if importance is not None:
        imp = np.zeros(W)
        np.maximum.at(imp, wid, np.asarray(importance, np.float64))
    lock_w = np.zeros(W, bool)
    if locked is not None:
        np.logical_or.at(lock_w, wid, np.asarray(locked, bool))
    shape_list = [np.zeros_like(Pw)] + [
        np.asarray(s, np.float64)[np.maximum(first, 0)] for s in shapes
    ]
    S = len(shape_list)
    UV = None if uvs is None else np.asarray(uvs, np.float64)
    NRM = None if normals is None else np.asarray(normals, np.float64)

    # quadrics of the triangles' planes in each shape, and of the feature edges' lines
    Q = np.zeros((S, W, 10))
    wsum = np.zeros(W)
    E = _edges(T, Tw, labels, W)
    for s, offset in enumerate(shape_list):
        Ps = Pw + offset
        normal, area, d = _triangle_frame(Ps, Tw)
        q = _plane_quadrics(normal, d, area)
        for k in range(3):
            np.add.at(Q[s], Tw[:, k], q)
        if s == 0:
            for k in range(3):
                np.add.at(wsum, Tw[:, k], area)
            f = np.flatnonzero(E.feature)
            edge = Ps[E.hi[f]] - Ps[E.lo[f]]
            length2 = (edge**2).sum(1)
            side = np.cross(edge, normal[E.t0[f]])
            side /= np.maximum(np.linalg.norm(side, axis=1), 1e-30)[:, None]
            weight = np.where(E.border[f], BORDER_WEIGHT, SEAM_WEIGHT) * length2
            q = _plane_quadrics(side, -(side * Ps[E.lo[f]]).sum(1), weight)
            np.add.at(Q[s], E.lo[f], q)
            np.add.at(Q[s], E.hi[f], q)
            np.add.at(wsum, E.lo[f], weight)
            np.add.at(wsum, E.hi[f], weight)
    group_ids = np.unique(groups)
    error = {int(g): 0.0 for g in group_ids}
    removed = {int(g): 0 for g in group_ids}
    passes = 0
    stopped = "target"
    while len(T) > target:
        if passes >= max_passes:
            stopped = "passes"
            break
        passes += 1
        E = _edges(T, Tw, labels, W) if passes > 1 else E
        count_g = {
            int(g): int(c) for g, c in zip(*np.unique(groups, return_counts=True))
        }
        allowance = {g: c - int(floors.get(g, 0)) for g, c in count_g.items()}
        # vertices of groups at their floor (or one triangle above it) stay
        frozen_groups = np.array(
            [g for g, a in allowance.items() if a < 2] or [-1], np.int64
        )
        frozen = np.zeros(W, bool)
        on_frozen = np.isin(groups, frozen_groups)
        frozen[Tw[on_frozen].ravel()] = True
        nonmanifold = np.zeros(W, bool)
        bad = E.count > 2
        nonmanifold[E.lo[bad]] = nonmanifold[E.hi[bad]] = True
        nfeat = np.bincount(E.lo[E.feature], minlength=W) + np.bincount(
            E.hi[E.feature], minlength=W
        )
        vlock = lock_w | nonmanifold | frozen | (nfeat == 1) | (nfeat > 2)

        # candidates: each edge in both directions, u -> v
        m = len(E.lo)
        u = np.concatenate([E.lo, E.hi])
        v = np.concatenate([E.hi, E.lo])
        e = np.concatenate([np.arange(m), np.arange(m)])
        rev = np.arange(2 * m) >= m
        au0 = np.where(rev, E.hi0[e], E.lo0[e])
        av0 = np.where(rev, E.lo0[e], E.hi0[e])
        au1 = np.where(rev, E.hi1[e], E.lo1[e])
        av1 = np.where(rev, E.lo1[e], E.hi1[e])
        ok = ~vlock[u] & ((nfeat[u] == 0) | E.feature[e]) & (E.count[e] <= 2)
        ok &= ~((au0 == au1) & (av0 != av1) & (au1 >= 0))
        common = _common_neighbours(E, W)
        ok &= common[e] == E.count[e]
        # a collapse removes the triangles of its edge: none of a group at its floor
        limited = np.zeros(int(groups.max(initial=0)) + 2, bool)
        for g, a in allowance.items():
            limited[g] = a < 2
        ok &= ~limited[groups[E.t0[e]]]
        ok &= ~((E.t1[e] >= 0) & limited[groups[np.maximum(E.t1[e], 0)]])
        if not ok.any():
            stopped = "no collapse is allowed"
            break
        cand = np.flatnonzero(ok)
        cu, cv, ce = u[cand], v[cand], e[cand]
        cost = np.zeros(len(cand))
        for s, offset in enumerate(shape_list):
            cost += _evaluate(Q[s][cu] + Q[s][cv], Pw[cv] + offset[cv])
        cost /= S
        edge_vec = Pw[cv] - Pw[cu]
        cost += LENGTH_WEIGHT * wsum[cu] * (edge_vec**2).sum(1)
        if NRM is not None:
            turn = 1.0 - (NRM[au0[cand]] * NRM[av0[cand]]).sum(1)
            cost += wsum[cu] * NORMAL_LENGTH**2 * np.maximum(turn, 0.0)
        cost = np.maximum(cost, 0.0) * imp[cu]
        cmap = (au0[cand], av0[cand], au1[cand], av1[cand])

        # the cheapest valid collapse of each vertex no dearer than the pass's threshold (a
        # quantile of every vertex's cheapest collapse), checked on its fan
        order = np.lexsort((cost, cu))
        su = cu[order]
        heads = np.r_[True, su[1:] != su[:-1]]
        start = np.flatnonzero(heads)
        length = np.diff(np.r_[start, len(order)])
        threshold = np.quantile(cost[order[start]], PASS_QUANTILE)
        best = np.full(W, -1)
        best_cost = np.full(W, np.inf)
        tri_offsets, tri_list = _csr(Tw.ravel(), np.repeat(np.arange(len(T)), 3), W)
        active = np.flatnonzero(cost[order[start]] <= threshold)
        pointer = np.zeros(len(start), np.int64)
        for _ in range(4):
            if len(active) == 0:
                break
            firsts = order[start[active] + pointer[active]]
            within = cost[firsts] <= threshold
            active, firsts = active[within], firsts[within]
            if len(active) == 0:
                break
            valid = _fan_ok(firsts, cu, cv, cmap, T, Tw, Pw, UV, tri_offsets, tri_list)
            good_c = firsts[valid]
            best[cu[good_c]] = good_c
            best_cost[cu[good_c]] = cost[good_c]
            pointer[active[~valid]] += 1
            active = active[~valid]
            active = active[pointer[active] < length[active]]
        finite = np.isfinite(best_cost)
        if not finite.any():
            stopped = "no collapse keeps its fan valid"
            break
        # winners: rounds of the cheapest within two rings, each round blocking the two rings
        # of its winners
        src = np.concatenate([E.lo, E.hi])
        dst = np.concatenate([E.hi, E.lo])
        open_ = finite.copy()
        order = np.lexsort((np.arange(W), best_cost))
        rank = np.empty(W, np.int64)
        rank[order] = np.arange(W)
        chosen = []
        for _ in range(MIS_ROUNDS):
            r = np.where(open_, rank, W + 1)
            m1 = r.copy()
            np.minimum.at(m1, src, r[dst])
            m2 = m1.copy()
            np.minimum.at(m2, src, m1[dst])
            found = np.flatnonzero((r == m2) & open_)
            if len(found) == 0:
                break
            chosen.append(found)
            block = np.zeros(W, bool)
            block[found] = True
            for _ring in range(2):
                grow = block.copy()
                grow[src[block[dst]]] = True
                block = grow
            open_ &= ~block
        win = np.concatenate(chosen)
        win = win[np.argsort(best_cost[win], kind="stable")]
        c = best[win]
        # each collapse removes the triangles of its edge: keep the groups and the target
        ec = ce[c]
        g0 = groups[E.t0[ec]]
        has1 = E.t1[ec] >= 0
        g1 = np.where(has1, groups[np.where(has1, E.t1[ec], 0)], -1)
        accept = np.ones(len(c), bool)
        for g, a in allowance.items():
            events = (g0 == g).astype(np.int64) + (g1 == g).astype(np.int64)
            accept &= (np.cumsum(events) <= a) | (events == 0)
        total = np.cumsum(1 + has1.astype(np.int64))
        accept &= total <= max(len(T) - target, 1)
        if not accept.any() and len(T) - target <= 1:
            break  # one triangle above the target: the last collapse would pass it
        if not accept.any():
            stopped = (
                f"the groups' floors ({len(c)} collapses, none within "
                f"{ {g: a for g, a in allowance.items() if a < 64} })"
            )
            break
        c, ec, g0, g1 = c[accept], ec[accept], g0[accept], g1[accept]
        uu, vv = cu[c], cv[c]
        for g in np.unique(g0):
            pick = g0 == g
            gap = _evaluate(Q[0][uu[pick]], Pw[vv[pick]]) / np.maximum(
                wsum[uu[pick]], 1e-30
            )
            error[int(g)] = max(error[int(g)], float(np.sqrt(max(gap.max(), 0.0))))
            removed[int(g)] += int(pick.sum())
        amap = np.arange(n)
        a0, b0, a1, b1 = (x[c] for x in cmap)
        amap[a0] = b0
        two = a1 >= 0
        amap[a1[two]] = b1[two]
        for s in range(S):
            np.add.at(Q[s], vv, Q[s][uu])
        np.add.at(wsum, vv, wsum[uu])
        T = amap[T]
        Tw = wid[T]
        keep = (Tw[:, 0] != Tw[:, 1]) & (Tw[:, 1] != Tw[:, 2]) & (Tw[:, 2] != Tw[:, 0])
        T, Tw, labels, groups = T[keep], Tw[keep], labels[keep], groups[keep]
        given = given[keep]
        origin = origin[keep]

    kept = np.zeros(n, bool)
    kept[T.ravel()] = True
    return Reduction(
        triangles=T,
        labels=given,
        origin=origin,
        groups=groups,
        kept=kept,
        error=error,
        removed=removed,
        passes=passes,
        dropped_degenerate=dropped,
        stopped=stopped,
    )


def _fan_ok(firsts, cu, cv, cmap, T, Tw, Pw, UV, tri_offsets, tri_list):
    """Whether each candidate collapse keeps its fan valid: every corner of ``u`` has a
    vertex of ``v`` to take, no triangle turns by more than :data:`FLIP_COS` or becomes a
    sliver, and no UV triangle flips."""
    u, v = cu[firsts], cv[firsts]
    a0, b0, a1, b1 = (x[firsts] for x in cmap)
    deg = tri_offsets[u + 1] - tri_offsets[u]
    row = np.repeat(np.arange(len(firsts)), deg)
    local = np.arange(len(row)) - np.repeat(np.cumsum(deg) - deg, deg)
    f = tri_list[tri_offsets[u][row] + local]
    tw = Tw[f]
    has_v = (tw == v[row][:, None]).any(1)
    keep = ~has_v
    row, f, tw = row[keep], f[keep], tw[keep]
    corner = np.argmax(tw == u[row][:, None], axis=1)
    attr = T[f, corner]
    mapped = np.where(attr == a0[row], b0[row], np.where(attr == a1[row], b1[row], -1))
    bad = np.zeros(len(firsts), bool)
    np.logical_or.at(bad, row, mapped < 0)
    p = Pw[tw]
    q = p.copy()
    q[np.arange(len(row)), corner] = Pw[v[row]]
    n_old = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n_new = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
    l_old = np.linalg.norm(n_old, axis=1)
    l_new = np.linalg.norm(n_new, axis=1)
    cos = (n_old * n_new).sum(1) / np.maximum(l_old * l_new, 1e-30)
    flip = (cos < FLIP_COS) | (l_new <= 1e-14)
    quality_old = _compactness(p[:, 0], p[:, 1], p[:, 2])
    quality_new = _compactness(q[:, 0], q[:, 1], q[:, 2])
    sliver = quality_new < np.minimum(0.1, 0.5 * quality_old)
    np.logical_or.at(bad, row, flip | sliver)
    if UV is not None:
        uv = UV[T[f]]
        uv_new = uv.copy()
        uv_new[np.arange(len(row)), corner] = UV[np.maximum(mapped, 0)]

        def signed(x):
            return (x[:, 1, 0] - x[:, 0, 0]) * (x[:, 2, 1] - x[:, 0, 1]) - (
                x[:, 2, 0] - x[:, 0, 0]
            ) * (x[:, 1, 1] - x[:, 0, 1])

        s_old, s_new = signed(uv), signed(uv_new)
        scale = np.abs(s_old)
        uv_flip = (scale > 1e-14) & (
            (np.sign(s_old) != np.sign(s_new)) | (np.abs(s_new) < 1e-3 * scale)
        )
        np.logical_or.at(bad, row, uv_flip)
    return ~bad

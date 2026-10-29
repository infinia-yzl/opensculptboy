# OpenSculptBoy
# Apache License, Version 2.0
"""
The closed lips of a fit (docs/mouth.md, "Sealed lips").

A reference's closed lips meet along a line: the bottom of the crease between them, or the
mouth's opening when its lips are separate rows. A fit to the reference's surface misses that
line: the points of a crease project onto its walls, so each lip of anny stops short of it and
the lips part by a slit through which the inside of the mouth shows, widest at the corners.

:func:`lip_line` finds the line on the reference from the front, :func:`lip_edges` the rows of
anny where each lip ends (the lowest points of the upper lip, the highest of the lower), and
:func:`seal_lips` moves both rows onto the line and carries the move smoothly to the lips
around them, so that the closed lips meet where the reference's lips meet. :func:`lip_slit`
measures what the front view shows between the lips.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# the front of anny's frame is -y; the views look along +y with x to the right and z up
_FRONT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def _front(points: np.ndarray) -> np.ndarray:
    """Points (N, 3) as (x, z, depth) of the front view, with depth -y (larger in front)."""
    return np.asarray(points, dtype=np.float64) @ _FRONT.T


@dataclasses.dataclass
class MouthBox:
    """The front view's window on the mouth, from anny's craniofacial landmarks: the mouth's
    corners (``ch.L``, ``ch.R``) and the midpoints of the lips' borders (``ls``, ``li``)."""

    left: np.ndarray
    right: np.ndarray
    upper: np.ndarray
    lower: np.ndarray

    @classmethod
    def from_landmarks(cls, points: dict) -> "MouthBox":
        return cls(
            *(np.asarray(points[k], np.float64) for k in ("ch.L", "ch.R", "ls", "li"))
        )

    @property
    def x_range(self) -> tuple[float, float]:
        a, b = sorted((float(self.left[0]), float(self.right[0])))
        return a, b

    @property
    def z_range(self) -> tuple[float, float]:
        return float(self.lower[2]), float(self.upper[2])

    @property
    def width(self) -> float:
        a, b = self.x_range
        return b - a


@dataclasses.dataclass
class LipLine:
    """Where a surface's lips end, ordered by x (N, 3) and in the surface's frame: the upper
    lip's edge and the lower lip's. They are one line where the lips meet; a mouth that
    stands open keeps them apart."""

    upper: np.ndarray
    lower: np.ndarray

    @property
    def ends(self) -> np.ndarray:
        """The mouth's corners (2, 3): where both edges end, at either side."""
        return 0.5 * (self.upper[[0, -1]] + self.lower[[0, -1]])

    def gap(self) -> np.ndarray:
        """The distance (N,) between the edges at each point of the line."""
        return np.linalg.norm(self.upper - self.lower, axis=1)


def _raster(points, triangles, xs, zs):
    from opensculptboy.eyes.geometry import _depth_raster

    p = _front(points)
    lo = np.array([xs[0], zs[0]]) - 0.01
    hi = np.array([xs[-1], zs[-1]]) + 0.01
    inside = ((p[:, :2] > lo) & (p[:, :2] < hi)).all(1)
    t = np.asarray(triangles)
    return _depth_raster(p, t[inside[t].any(1)], xs, zs)


def lip_line(
    vertices: np.ndarray,
    triangles: np.ndarray,
    box: MouthBox,
    resolution: float = 1e-4,
    margin: float = 0.004,
    min_strength: float = 2e-4,
    end_margin: float = 1e-3,
) -> LipLine | None:
    """
    Where the lips of a surface end (:class:`LipLine`), in the surface's frame (anny's axes,
    the front at -y), or None when the front view shows no lips.

    A surface whose mouth is an opening (its lips separate rows, as a game's head mesh has
    them, or a mesh without the inside of its mouth) gives the borders of that opening
    (:func:`_opening_line`): one line where they touch, two where the mouth stands open.
    Otherwise the lips meet in a crease, and each column of the front view between the
    mouth's corners (``box``, widened by
    ``margin``) is searched between the lips' fullest points above and below the mouth's
    middle: the line lies at the most recessed point between them, or in the middle of a run
    that the surface leaves open (a mouth opening whose lips are separate rows). A column
    belongs to the line while the crease there lies at least ``min_strength`` (metres) behind
    both lips; the line is the run of such columns that holds the middle of the mouth, which
    ends where the crease fades, and at most ``end_margin`` past the box's corners, beyond
    which the crease runs on into the folds of the cheeks.
    """
    x0, x1 = box.x_range
    z0, z1 = box.z_range
    opening = _opening_line(vertices, triangles, box, margin, end_margin)
    if opening is not None:
        return opening
    xs = np.arange(x0 - margin, x1 + margin, resolution)
    zs = np.arange(z0 - margin, z1 + margin, resolution)
    if len(xs) < 3 or len(zs) < 3:
        return None
    depth = _raster(vertices, triangles, xs, zs)
    middle = 0.5 * (z0 + z1)
    rows = []
    for ix, x in enumerate(xs):
        col = depth[:, ix]
        hit = np.isfinite(col)
        above, below = hit & (zs >= middle), hit & (zs < middle)
        if not above.any() or not below.any():
            rows.append(None)
            continue
        top = int(np.argmax(np.where(above, col, -np.inf)))
        bottom = int(np.argmax(np.where(below, col, -np.inf)))
        if top <= bottom + 1:
            rows.append(None)
            continue
        span = np.arange(bottom, top + 1)
        rim = min(col[top], col[bottom])
        gap = ~hit[span]
        if gap.any():
            # an opening: its middle, at the depth of its nearer rim
            run = span[gap]
            z = 0.5 * (zs[run[0]] + zs[run[-1]])
            near = [col[k] for k in (run[0] - 1, run[-1] + 1) if hit[k]]
            d = max(near) if near else rim
            strength = rim - d if near else np.inf
            strength = max(strength, resolution)  # an opening seals in any case
        else:
            k = span[int(np.argmin(col[span]))]
            z, d = zs[k], col[k]
            strength = rim - d
        rows.append((x, z, d, strength))
    strong = [r is not None and r[3] >= min_strength for r in rows]
    centre = (x0 + x1) / 2
    start = int(np.argmin(np.abs(xs - centre)))
    if not strong[start]:
        near = [i for i, s in enumerate(strong) if s]
        if not near:
            return None
        start = min(near, key=lambda i: abs(i - start))
    lo = hi = start
    while lo > 0 and strong[lo - 1]:
        lo -= 1
    while hi < len(xs) - 1 and strong[hi + 1]:
        hi += 1
    line = np.array([rows[i][:3] for i in range(lo, hi + 1)])
    # past the mouth's corners the crease runs on into the folds of the cheeks
    line = line[(line[:, 0] >= x0 - end_margin) & (line[:, 0] <= x1 + end_margin)]
    if len(line) < 3:
        return None
    # smooth the raster's steps along the line (about 0.5 mm)
    k = max(1, int(round(0.0005 / resolution)))
    if len(line) > 2 * k + 1:
        kernel = np.ones(2 * k + 1) / (2 * k + 1)
        pad = np.pad(line[:, 1:], ((k, k), (0, 0)), mode="edge")
        line[:, 1] = np.convolve(pad[:, 0], kernel, mode="valid")
        line[:, 2] = np.convolve(pad[:, 1], kernel, mode="valid")
    line = line @ _FRONT  # back to (x, y, z)
    return LipLine(line, line.copy())


def _opening_line(
    vertices: np.ndarray,
    triangles: np.ndarray,
    box: MouthBox,
    margin: float,
    end_margin: float,
    step: float = 2e-4,
    coverage: float = 0.6,
    split: float = 3e-4,
) -> LipLine | None:
    """
    The edges of a mouth opening, or None when the surface has none: the border edges
    (edges of one triangle) within the box, widened by ``margin``, sampled every
    ``step / 2``. In each column of width ``step`` the samples split at their largest step in
    height into the upper lip's border and the lower lip's, when that step exceeds ``split``
    (metres); otherwise the borders touch there and both edges take the column's median. The
    opening must span ``coverage`` of the mouth's width.
    """
    x0, x1 = box.x_range
    z0, z1 = box.z_range
    p = np.asarray(vertices, np.float64)
    # weld the vertices that a source splits at its UV seams, whose edges are no border
    _, weld = np.unique(
        np.round(p / 1e-6).astype(np.int64), axis=0, return_inverse=True
    )
    t = weld.reshape(-1)[np.asarray(triangles)]
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), 1)
    edges, count = np.unique(e, axis=0, return_counts=True)
    border = edges[count == 1]
    if len(border) == 0:
        return None
    first = np.zeros(weld.max() + 1, np.int64)
    first[weld.reshape(-1)] = np.arange(len(p))
    a, b = p[first[border[:, 0]]], p[first[border[:, 1]]]

    def inside(q):
        return (
            (q[:, 0] > x0 - margin)
            & (q[:, 0] < x1 + margin)
            & (q[:, 2] > z0 - margin)
            & (q[:, 2] < z1 + margin)
        )

    keep = inside(a) & inside(b)
    a, b = a[keep], b[keep]
    if len(a) == 0:
        return None
    pieces = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / (step / 2))).astype(
        int
    )
    samples = np.concatenate(
        [
            a[k] + np.linspace(0, 1, n + 1)[:, None] * (b[k] - a[k])
            for k, n in enumerate(pieces)
        ]
    )
    lo, hi = x0 - end_margin, x1 + end_margin
    samples = samples[(samples[:, 0] >= lo) & (samples[:, 0] <= hi)]
    bins = np.floor((samples[:, 0] - lo) / step).astype(int)
    filled = np.unique(bins)
    if len(filled) * step < coverage * (x1 - x0):
        return None
    upper, lower = [], []
    for k in filled:
        column = samples[bins == k]
        column = column[np.argsort(column[:, 2])]
        steps = np.diff(column[:, 2])
        cut = int(np.argmax(steps)) + 1 if len(steps) else 0
        if len(steps) and steps[cut - 1] > split:
            lower.append(np.median(column[:cut], 0))
            upper.append(np.median(column[cut:], 0))
        else:
            upper.append(np.median(column, 0))
            lower.append(upper[-1])
    upper, lower = np.array(upper), np.array(lower)
    order = np.argsort(upper[:, 0])
    return LipLine(_median_along(upper[order]), _median_along(lower[order]))


def _median_along(line: np.ndarray, half: int = 3) -> np.ndarray:
    """The line with y and z replaced by their median over ``half`` points either side."""
    out = line.copy()
    for k in range(len(line)):
        window = line[max(0, k - half) : k + half + 1]
        out[k, 1:] = np.median(window[:, 1:], 0)
    return out


def lip_edges(
    vertices: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    box: MouthBox,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The rows where anny's lips end, as vertex indices of ``vertices`` (the posed mesh without
    detail offsets): the vertices of the upper lip (``upper``, a mask) across the mouth
    (``box``, between the lips' borders) that no other vertex of the upper lip within
    ``spacing / 2`` of their x lies below, and the vertices of the lower lip that none of the lower lip lies above. The
    lips' rolls meet there when the mouth closes; the rolls behind them stay out, since the
    edge lies lower (higher) at the same x.
    """
    x0, x1 = box.x_range
    z0, z1 = box.z_range
    radius = 0.5 * spacing
    out = []
    for mask, sign in ((upper, 1.0), (lower, -1.0)):
        rows = np.nonzero(np.asarray(mask, bool))[0]
        x = vertices[rows, 0]
        z = vertices[rows, 2]
        inside = (x >= x0 - spacing) & (x <= x1 + spacing) & (z >= z0) & (z <= z1)
        rows, x = rows[inside], x[inside]
        order = np.argsort(x)
        rows, x = rows[order], x[order]
        z = sign * vertices[rows, 2]  # the edge has the smallest value
        lo = np.searchsorted(x, x - radius, "left")
        hi = np.searchsorted(x, x + radius, "right")
        keep = [r for k, r in enumerate(rows) if z[k] <= z[lo[k] : hi[k]].min() + 1e-9]
        out.append(np.array(sorted(keep), dtype=np.int64))
    return out[0], out[1]


def lip_masks(
    vertices: np.ndarray,
    roll: np.ndarray,
    share: np.ndarray,
    box: MouthBox,
    spacing: float,
    low: float = 0.3,
    high: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The upper and the lower lip (masks) among the vertices of the mouth (``roll``): the skin
    that the jaw's opening leaves (``share`` under ``low``) and the skin it carries (over
    ``high``). Around the corners the share runs between the two; there the vertices above
    the middle of the mouth (the line midway between the lips' edges, :func:`lip_edges` of
    the clear vertices, run out to the corners of :func:`lip_corners`) count to the upper
    lip, those below it to the lower lip.
    """
    roll = np.asarray(roll, bool)
    upper, lower = roll & (share < low), roll & (share > high)
    up, down = lip_edges(vertices, upper, lower, box, spacing)
    corners = lip_corners(vertices, roll, box)
    if len(up) < 2 or len(down) < 2:
        return upper, lower
    xs = np.unique(np.concatenate([vertices[up, 0], vertices[down, 0]]))

    def edge(rows):
        order = np.argsort(vertices[rows, 0])
        return np.interp(xs, vertices[rows, 0][order], vertices[rows, 2][order])

    mid = np.stack([xs, 0.5 * (edge(up) + edge(down))], 1)
    ends = vertices[corners][:, [0, 2]]
    mid = np.concatenate([mid, ends])
    mid = mid[np.argsort(mid[:, 0])]
    between = roll & ~upper & ~lower
    z = np.interp(vertices[between, 0], mid[:, 0], mid[:, 1])
    above = vertices[between, 2] > z
    rows = np.nonzero(between)[0]
    upper = upper.copy()
    lower = lower.copy()
    upper[rows[above]] = True
    lower[rows[~above]] = True
    return upper, lower


def on_line(points: np.ndarray, line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each point, the point of ``line`` (ordered by x) at its x, and whether its x lies
    within the line."""
    x = np.asarray(points)[:, 0]
    inside = (x >= line[0, 0]) & (x <= line[-1, 0])
    target = np.stack([np.interp(x, line[:, 0], line[:, k]) for k in range(3)], 1)
    return target, inside


def meeting_line(
    upper: np.ndarray,
    lower: np.ndarray,
    reference=None,
    step: float = 5e-4,
    snap: float = 0.003,
) -> LipLine | None:
    """
    A line where two lip edges (points (N, 3) and (M, 3), anny's frame, the front at -y) meet:
    the middle of the edges at each x that both span, every ``step`` metres, moved onto the
    nearest point of ``reference`` (anything with ``closest(points)``) where that lies within
    ``snap`` metres. A solid statue's lips are carved shut and leave no crease that
    :func:`lip_line` can find, so the fit closes anny's lips where its own fitted edges meet;
    None when the edges share less than three steps.
    """
    upper = np.asarray(upper, np.float64)
    lower = np.asarray(lower, np.float64)
    if len(upper) < 2 or len(lower) < 2:
        return None
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]
    x0 = max(upper[0, 0], lower[0, 0])
    x1 = min(upper[-1, 0], lower[-1, 0])
    xs = np.arange(x0, x1 + 0.5 * step, step)
    if len(xs) < 3:
        return None
    a = np.stack([np.interp(xs, upper[:, 0], upper[:, k]) for k in range(3)], 1)
    b = np.stack([np.interp(xs, lower[:, 0], lower[:, k]) for k in range(3)], 1)
    line = 0.5 * (a + b)
    if reference is not None:
        on = np.asarray(reference.closest(line)[0], np.float64)
        near = np.linalg.norm(on - line, axis=1) < snap
        line[near] = on[near]
    line[:, 0] = xs
    return LipLine(line, line.copy())


def _adjacency(n: int, triangles: np.ndarray):
    """The vertex adjacency (n, n) of a triangle mesh, as a CSR matrix of ones."""
    from scipy.sparse import coo_matrix

    t = np.asarray(triangles)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A


def harmonic_fill(
    values: np.ndarray, triangles: np.ndarray, free: np.ndarray
) -> np.ndarray:
    """The values (N, C) with the rows of ``free`` replaced by the harmonic extension of the
    others (uniform weights); a component of ``free`` that touches no other vertex keeps its
    values."""
    from scipy.sparse import diags
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import spsolve

    free = np.asarray(free, bool)
    out = np.array(values, dtype=np.float64, copy=True)
    if not free.any():
        return out
    A = _adjacency(len(out), triangles)
    rows = np.nonzero(free)[0]
    border = np.nonzero(~free)[0]
    touching = np.asarray(A[rows][:, border].sum(1)).ravel() > 0
    _, labels = connected_components(A[rows][:, rows], directed=False)
    keep = rows[np.isin(labels, np.unique(labels[touching]))]
    if len(keep) == 0:
        return out
    degree = np.asarray(A[keep].sum(1)).ravel()
    L = (diags(degree) - A[keep][:, keep]).tocsc()
    rhs = A[keep][:, border] @ out[border]
    out[keep] = np.asarray(spsolve(L, rhs)).reshape(len(keep), -1)
    return out


def rings_from(triangles: np.ndarray, start: np.ndarray, rings: int, n: int):
    """Mask (n,) of the vertices within ``rings`` faces of the vertices ``start``."""
    A = _adjacency(n, triangles)
    out = np.zeros(n, bool)
    out[np.asarray(start, np.int64)] = True
    for _ in range(rings):
        out |= A @ out.astype(np.float64) > 0
    return out


@dataclasses.dataclass
class Seal:
    """What :func:`seal_lips` did on a level: the rows moved onto the line and the vertices
    that followed them."""

    upper: np.ndarray  # rows of the upper lip on the line
    lower: np.ndarray  # rows of the lower lip on the line
    moved: np.ndarray  # mask of every vertex that moved
    shift_mm: float  # the mean distance the rows moved
    corners: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0, np.int64)
    )  # the corners moved onto the line's ends
    line: "LipLine | None" = None  # the line the rows moved onto


def lip_corners(vertices: np.ndarray, roll: np.ndarray, box: MouthBox) -> np.ndarray:
    """The vertex of the lips (``roll``, a mask) nearest each corner of the mouth (``box``,
    anny's landmarks ``ch.L`` and ``ch.R``): where the edges of both lips meet."""
    rows = np.nonzero(np.asarray(roll, bool))[0]
    out = []
    for corner in (box.left, box.right):
        out.append(
            rows[int(np.argmin(np.linalg.norm(vertices[rows] - corner, axis=1)))]
        )
    return np.unique(np.array(out, dtype=np.int64))


def seal_lips(
    offsets: np.ndarray,
    parametric: np.ndarray,
    triangles: np.ndarray,
    pose,
    unpose,
    line: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    region: np.ndarray,
    rings: int,
    corners: np.ndarray | None = None,
    held: np.ndarray | None = None,
    beyond: float = 0.0,
) -> tuple[np.ndarray, Seal]:
    """
    The rest-frame offsets (N, 3) of a level with the lips sealed: the rows ``upper`` and
    ``lower`` (:func:`lip_edges`) within the line's span move onto the reference's edges of
    the same lip (``line``, a :class:`LipLine` in the posed frame, from :func:`lip_line`) at
    their own x, so that both lips end where the reference's end and meet where they meet,
    the ``corners`` (:func:`lip_corners`) move onto the line's nearer end, and the vertices of
    ``region`` within ``rings`` faces of those rows, but those of ``held``, follow by the
    harmonic extension of the move; the rows of both edges within ``beyond`` metres past an
    end of the line move onto the line where the lips meet, carried on past that end, so that
    the corners close. The rest of the mesh stays. ``pose`` and ``unpose`` map
    rows of rest points to the posed frame and back, ``pose(points, rows)``.
    """
    n = len(parametric)
    rest = parametric + offsets
    rows, target = [], []
    ends = line.ends
    middle = 0.5 * (line.upper + line.lower)
    for edge, curve in ((upper, line.upper), (lower, line.lower)):
        at = pose(rest[edge], edge)
        goal, inside = on_line(at, curve)
        rows.append(edge[inside])
        target.append(goal[inside])
        # rows just past the line's ends close the mouth's corners: both lips move onto the
        # line where they meet, carried on past its end (a corner left open showed as a
        # dark notch)
        if beyond > 0 and len(middle) >= 2:
            for end, a, b in ((0, 0, 1), (1, -1, -2)):
                tangent = middle[a] - middle[b]
                if abs(tangent[0]) < 1e-9:
                    continue
                dx = at[:, 0] - ends[end, 0]
                past = ~inside & (np.sign(dx) == np.sign(tangent[0]))
                past &= np.abs(dx) <= beyond
                if past.any():
                    rows.append(edge[past])
                    target.append(ends[end] + dx[past, None] / tangent[0] * tangent)
    rows, target = np.concatenate(rows), np.concatenate(target)
    given = np.zeros(0, np.int64) if corners is None else np.asarray(corners)
    # a corner that is a row of an edge within the line's span keeps that row's target
    extra = np.setdiff1d(given, rows)
    if len(extra):
        at = pose(rest[extra], extra)
        pick = np.argmin(np.abs(at[:, None, 0] - ends[None, :, 0]), axis=1)
        rows = np.concatenate([rows, extra])
        target = np.concatenate([target, ends[pick]])
    if len(rows) == 0:
        return offsets, Seal(upper[:0], lower[:0], np.zeros(n, bool), 0.0)
    goal = unpose(target, rows)
    move = np.zeros((n, 3))
    move[rows] = goal - rest[rows]
    free = rings_from(triangles, rows, rings, n) & np.asarray(region, bool)
    if held is not None:
        free &= ~np.asarray(held, bool)
    free[rows] = False
    move = harmonic_fill(move, triangles, free)
    out = np.asarray(offsets, dtype=np.float64) + move
    moved = np.abs(move).max(1) > 0
    shift = float(np.linalg.norm(move[rows], axis=1).mean() * 1000)
    return out, Seal(
        np.intersect1d(upper, rows),
        np.intersect1d(lower, rows),
        moved,
        shift,
        np.intersect1d(given, rows),
        line,
    )


def lip_slit(
    vertices: np.ndarray,
    triangles: np.ndarray,
    reference_vertices: np.ndarray,
    reference_triangles: np.ndarray,
    box: MouthBox,
    resolution: float = 1e-4,
    depth: float = 5e-4,
    corner: float = 0.004,
) -> dict:
    """
    What the front view shows between the lips of a fit (``vertices``, posed and placed as
    the reference): the area (mm²) where the fit lies more than ``depth`` behind the
    reference within the lips' band (the box's height plus 3 mm above and below, its width
    plus 3 mm on either side), and the part of it within ``corner`` of either corner of the
    mouth. A fit whose lips meet where the reference's meet shows none: a slit or a hole at
    a corner lets the view in behind the lips.
    """
    x0, x1 = box.x_range
    z0, z1 = box.z_range
    xs = np.arange(x0 - 0.003, x1 + 0.003, resolution)
    zs = np.arange(z0 - 0.003, z1 + 0.003, resolution)
    fit = _raster(vertices, triangles, xs, zs)
    ref = _raster(reference_vertices, reference_triangles, xs, zs)
    seen = np.isfinite(ref)
    behind = seen & ((fit < ref - depth) | ~np.isfinite(fit))
    X = np.broadcast_to(xs[None, :], behind.shape)
    corners = (np.abs(X - x0) < corner) | (np.abs(X - x1) < corner)
    px = (resolution * 1000) ** 2
    return {
        "slit_mm2": round(float(behind.sum() * px), 3),
        "corners_mm2": round(float((behind & corners).sum() * px), 3),
        "middle_mm2": round(float((behind & ~corners).sum() * px), 3),
    }


def lip_gap(rest: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> dict:
    """The gap (mm) between the lips' edge rows in the rest frame (the closed mouth): from
    each row of ``upper`` to the lower lip's edge at its x (the rows of ``lower`` joined in
    order of x), within the span of both: mean, 95th percentile and largest."""
    empty = {"mean": None, "p95": None, "max": None}
    if len(upper) < 1 or len(lower) < 2:
        return empty
    order = np.argsort(rest[lower, 0])
    edge = rest[lower][order]
    x = rest[upper, 0]
    inside = (x >= edge[0, 0]) & (x <= edge[-1, 0])
    if not inside.any():
        return empty
    at = np.stack([np.interp(x[inside], edge[:, 0], edge[:, k]) for k in range(3)], 1)
    d = np.linalg.norm(rest[upper][inside] - at, axis=1) * 1000
    return {
        "mean": round(float(d.mean()), 3),
        "p95": round(float(np.percentile(d, 95)), 3),
        "max": round(float(d.max()), 3),
    }

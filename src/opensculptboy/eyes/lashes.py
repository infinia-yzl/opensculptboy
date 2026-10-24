# OpenSculptBoy
# Apache License, Version 2.0
"""
The lashes of the exported eyes.

A card with the eyes of :mod:`opensculptboy.eyes` also gets lashes along its lid margins
(``material.lashes``, :class:`~opensculptboy.material.LashMaterial`):

- a fitted card takes the lashes of its reference: ``opensculptboy fit --textures`` and
  ``opensculptboy bake`` find the parts with the role ``lashes`` in the handoff
  (:func:`reference_lashes`), move them into the card's rest frame by the rigid motion of the
  fitted head, and write them next to the card with their texture
  (``<card>.lashes.safetensors`` and ``<card>.lashes.png``);
- any other card takes generic lashes (:func:`generic_lashes`): one strip of cards along each
  lid margin, drawn with a texture of tapered strands (:func:`generic_texture`).

:func:`lash_parts` binds the lashes to the mesh of a file as parts of
:class:`~opensculptboy.eyes.mesh.EyedMesh`. Each lash vertex keeps its offset from its root
on the lid margin in the frame of the margin's tangent and the lid's direction away from the
margin there (:class:`PatchFollow`), so the lashes keep their shape and turn with the edge of
the lid in the blinks of the morph targets, the T-pose of a VRM file and every other target of
the file. As the upper lid closes, its lashes also turn down about the margin by 1.5 times the
angle that the opening between the lids closes (:data:`CLOSE_TURN`): the edge of a closing lid
rolls, and the lashes of a closed eye point forward and down along the lid line. They also stay
:data:`CLEARANCE` outside the outward-facing skin around the eye, so a closed lid lays them on
its skin. The lashes take the skin weights of their roots on the margin.

The lid margin (:func:`lid_margins`) is the line where the lids' skin turns from facing out
to facing the eyeball (the socket of :func:`opensculptboy.eyes.geometry.socket`), split into
the upper and the lower lid at the corners of the eye.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

FORMAT = "opensculptboy/lashes@1"
# the lids of a lash vertex
UPPER, LOWER = 0, 1
# the margin: outward-facing skin within this many eyeball radii of the eye's centre and
# within this angle of the gaze
MARGIN_REACH = 1.35
MARGIN_ANGLE = 75.0
# the lid of a margin: its outward-facing skin within this distance of the margin (metres)
LID_REACH = 0.005


@dataclasses.dataclass
class LashData:
    """
    A set of lashes in a card's rest frame (Anny's frame, metres): points, normals, UVs (OBJ
    convention, v = 0 at the bottom) into the lashes' own texture, triangles, and the lid of
    each point (:data:`UPPER` or :data:`LOWER`).
    """

    points: np.ndarray  # (M, 3)
    normals: np.ndarray  # (M, 3)
    uvs: np.ndarray  # (M, 2)
    triangles: np.ndarray  # (T, 3)
    lid: np.ndarray  # (M,) uint8

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float64)
        self.normals = np.asarray(self.normals, dtype=np.float64)
        self.uvs = np.asarray(self.uvs, dtype=np.float64)
        self.triangles = np.asarray(self.triangles, dtype=np.int64)
        self.lid = np.asarray(self.lid, dtype=np.uint8)

    def save(self, path) -> None:
        from safetensors.numpy import save_file

        save_file(
            {
                "points": self.points,
                "normals": self.normals,
                "uvs": self.uvs,
                "triangles": self.triangles,
                "lid": self.lid,
            },
            str(path),
            metadata={"format": FORMAT},
        )

    @classmethod
    def load(cls, path) -> "LashData":
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as f:
            if (f.metadata() or {}).get("format") != FORMAT:
                raise ValueError(f"{path} is not a lashes file ({FORMAT}).")
            return cls(**{k: f.get_tensor(k) for k in f.keys()})


# -------------------------------------------------------------------------- the lid margins
@dataclasses.dataclass
class Margin:
    """
    The margin of one lid: its vertex rows in order from the side of the nose to the temple
    (``rows``) with their rest positions (``points``), and the rows of the lid's outward-facing
    skin within :data:`LID_REACH` of it, the margin's own among them (``lid_rows``).
    """

    rows: np.ndarray
    points: np.ndarray
    lid_rows: np.ndarray


def _neighbours(triangles: np.ndarray, count: int):
    """The neighbours of each vertex (a CSR matrix) over the edges of ``triangles``."""
    from scipy.sparse import coo_matrix

    t = np.asarray(triangles, dtype=np.int64)
    a = np.concatenate([t[:, 0], t[:, 1], t[:, 2], t[:, 1], t[:, 2], t[:, 0]])
    b = np.concatenate([t[:, 1], t[:, 2], t[:, 0], t[:, 0], t[:, 1], t[:, 2]])
    return coo_matrix((np.ones(len(a)), (a, b)), shape=(count, count)).tocsr()


def lid_margins(frame, rest, normals, skin, skin_triangles) -> dict[int, Margin]:
    """
    The margins of an eye's lids (:data:`UPPER`, :data:`LOWER`) on a mesh: the vertices of
    the outward-facing skin of the lids (``skin``, a mask of the vertices that are not on an
    eyeball, within :data:`MARGIN_REACH` eyeball radii and :data:`MARGIN_ANGLE` of the gaze)
    that share a triangle of ``skin_triangles`` with the skin of the socket (the inside of the
    lids, which faces the eyeball). The corners of the eye are the margin's vertices farthest
    to each side; the margin and the lid skin above the line between them form the upper lid,
    those below it the lower lid. A lid of fewer than 3 margin vertices is left out.
    """
    from scipy.spatial import cKDTree

    from opensculptboy.eyes import geometry

    rest = np.asarray(rest, dtype=np.float64)
    local = frame.to_local(rest)
    dist = np.linalg.norm(local, axis=1)
    theta = np.degrees(
        np.arccos(np.clip(local[:, 2] / np.maximum(dist, 1e-12), -1.0, 1.0))
    )
    inner = geometry.socket(frame, rest, normals, skin)
    outer = skin & (dist < MARGIN_REACH) & (theta < MARGIN_ANGLE) & ~inner
    graph = _neighbours(skin_triangles, len(rest))
    touches = np.asarray(graph @ (inner & (dist < MARGIN_REACH)).astype(float)).ravel()
    margin = np.flatnonzero(outer & (touches > 0))
    out = {}
    if len(margin) < 6:
        return out
    x, y = local[margin, 0], local[margin, 1]
    a, b = np.argmin(x), np.argmax(x)

    def above(rows):  # above the line through the corners
        line = y[a] + (y[b] - y[a]) * (local[rows, 0] - x[a]) / max(x[b] - x[a], 1e-12)
        return local[rows, 1] > line

    skin_rows = np.flatnonzero(outer)
    for lid in (UPPER, LOWER):
        rows = margin[above(margin) == (lid == UPPER)]
        if len(rows) < 3:
            continue
        # from the side of the nose: toward the midline (-X for the left eye, +X for the
        # right)
        rows = rows[np.argsort(local[rows, 0] * (-1.0 if frame.side == "R" else 1.0))]
        mine = skin_rows[above(skin_rows) == (lid == UPPER)]
        gap, _ = cKDTree(rest[rows]).query(rest[mine])
        out[lid] = Margin(rows, rest[rows], np.union1d(mine[gap <= LID_REACH], rows))
    return out


# --------------------------------------------------------------------------------- binding
# each lash point follows the margin near its root: the root is a soft nearest point of the
# margin (weights exp(-d^2 / ROOT_SPREAD^2) over the margin vertices); the frame there is the
# margin's tangent and the lid's direction away from the margin (the lid's skin within
# ACROSS_REACH of the margin, less the margin), each weighted by exp(-d^2 / PATCH_SPREAD^2) of
# the distance from the root (metres)
ROOT_SPREAD = 0.0015
PATCH_SPREAD = 0.003
ACROSS_REACH = 0.0025
# as the upper lid closes, its lashes turn down about the margin by this share of the angle
# that the lids' opening closes (seen from the eye's centre), on top of the lid's own turn:
# the edge of a closing lid rolls, so that the lashes of a closed eye point forward and down
# and rest along the closed lid
CLOSE_TURN = 1.5
# the opening of the lids is measured in degrees, and a closed eye turns its lashes by at most
# this angle (degrees)
MAX_CLOSE_TURN = 70.0
# the lashes stay this far (metres) outside the outward-facing skin around the eye, or as far
# as they lie at rest when that is less: a closed lid lays them on its skin
CLEARANCE = 0.0003
# the skin around the eye that the lashes stay outside: within this many eyeball radii
SKIN_REACH = 1.8


@dataclasses.dataclass
class PatchFollow:
    """
    How the points of a part follow the lid margin of a mesh: each point's root, the
    margin's tangent there and the lid's direction away from the margin there are linear
    combinations of the rows ``rows`` (the rows of ``root``, ``tangent`` and ``across``,
    (M, n) each); their rest positions are ``rest``. A point keeps its offset from its root
    in the frame of the tangent, the lid's normal (the tangent across the lid's direction)
    and the third axis, so the lashes keep their shape and turn with the edge of the lid as
    it closes over the eyeball.
    """

    rows: np.ndarray  # (n,)
    rest: np.ndarray  # (n, 3)
    root: np.ndarray  # (M, n), rows sum to 1
    tangent: np.ndarray  # (M, n), rows sum to 0
    across: np.ndarray  # (M, n), rows sum to 0
    # the upper lashes: the eye's centre and the root's nearest point on the lower lid's
    # margin, (M, n) each with rows that sum to 1, and how far the lashes turn as the opening
    # between them closes (CLOSE_TURN; 0 for the lower lashes)
    centre: np.ndarray | None = None
    opposite: np.ndarray | None = None
    turn: np.ndarray | None = None  # (M,)
    # the outward-facing skin around the eye (rows of the mesh, and its triangles over them),
    # and how far each point may come to it (CLEARANCE, or its distance at rest)
    skin_rows: np.ndarray | None = None
    skin_triangles: np.ndarray | None = None
    floor: np.ndarray | None = None

    def place(self, points, positions):
        """``points`` (M, 3) at rest moved with the lid for ``positions`` (a function of
        rows, returning their positions), and kept outside the skin."""
        rotations, centres = self.motions(positions)
        moved = self.points(points, rotations, centres)
        if self.skin_rows is None:
            return moved
        depth, normals = self.skin_depth(moved, positions(self.skin_rows))
        push = np.clip(self.floor - depth, 0.0, None)
        return moved + push[:, None] * normals

    def skin_depth(self, points, skin):
        """The signed distance (M,) of ``points`` from the skin around the eye at ``skin``
        (its rows' positions), along the skin's normal at the nearest skin vertices (the
        mean of 4), and that normal (M, 3)."""
        from scipy.spatial import cKDTree

        from opensculptboy.export.body import vertex_normals

        normals = vertex_normals(skin, self.skin_triangles)
        dist, idx = cKDTree(skin).query(points, k=4)
        w = 1.0 / np.maximum(dist, 1e-6)
        w /= w.sum(1, keepdims=True)
        n = np.einsum("mk,mkd->md", w, normals[idx])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        base = np.einsum("mk,mkd->md", w, skin[idx])
        return ((points - base) * n).sum(1), n

    @staticmethod
    def _frames(x, tangent, across):
        t = tangent @ x
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        n = np.cross(t, across @ x)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        return np.stack([t, np.cross(n, t), n], 2)  # (M, 3, 3), columns t, b, n

    def _opening(self, x, axis):
        """The signed angle (M,) about ``axis`` from the root to its point on the lower
        margin, seen from the eye's centre."""
        c = self.centre @ x
        u = self.root @ x - c
        v = self.opposite @ x - c
        return np.arctan2((np.cross(u, v) * axis).sum(1), (u * v).sum(1))

    def motions(self, positions):
        """The rotation (M, 3, 3) of each point's frame, and its root at rest and in
        ``positions`` (M, 3) (a function of the rows, returning (n, 3))."""
        moved = np.asarray(positions(self.rows), dtype=np.float64)
        f0 = self._frames(self.rest, self.tangent, self.across)
        f1 = self._frames(moved, self.tangent, self.across)
        rotations = np.einsum("mij,mkj->mik", f1, f0)
        if self.turn is not None and self.centre is not None:
            # the extra turn of a closing lid: about the tangent, toward the lower lid
            axis = f1[:, :, 0]
            before = self._opening(self.rest, f0[:, :, 0])
            after = self._opening(moved, axis)
            sense = np.sign(before)
            closed = np.clip((before - after) * sense, 0.0, None)
            angle = sense * np.minimum(self.turn * closed, np.radians(MAX_CLOSE_TURN))
            rotations = np.einsum(
                "mij,mjk->mik", _axis_rotations(axis, angle), rotations
            )
        return rotations, (self.root @ self.rest, self.root @ moved)

    def points(self, points, rotations, centres) -> np.ndarray:
        """``points`` (M, 3) moved by the motions of their frames."""
        c0, c1 = centres
        return np.einsum("mij,mj->mi", rotations, points - c0) + c1

    def directions(self, vectors, rotations) -> np.ndarray:
        """Unit vectors (M, 3) turned by the rotations of their frames."""
        v = np.einsum("mij,mj->mi", rotations, vectors)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def _axis_rotations(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rotations (M, 3, 3) by ``angles`` (M,) about unit ``axes`` (M, 3) (Rodrigues)."""
    k = np.zeros((len(axes), 3, 3))
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -axes[:, 2], axes[:, 1], -axes[:, 0]
    k -= np.transpose(k, (0, 2, 1))
    s, c = np.sin(angles)[:, None, None], np.cos(angles)[:, None, None]
    return np.eye(3)[None] + s * k + (1 - c) * (k @ k)


def bind(points, lids, margins, rest, eyeball=None):
    """
    How lash ``points`` (M, 3) of one eye follow its lid margins: (anchors (n,), blend (M,
    n), source (M,), :class:`PatchFollow`). ``lids`` gives each point's lid, whose margin
    (``margins``, :func:`lid_margins`) it follows: its root is the soft nearest point of that
    margin (:data:`ROOT_SPREAD`); the margin's tangent and the lid's direction away from the
    margin are weighted by their distance from the root (:data:`PATCH_SPREAD`). blend @ the
    anchors' skin weights gives the points' skin weights (the root's, from the margin's
    vertices), and source is each point's nearest margin vertex. With ``eyeball`` (the rows
    of the eye's eyeball, whose mean is its centre), the upper lashes also turn down as the
    lids close (:data:`CLOSE_TURN`).
    """
    points = np.asarray(points, dtype=np.float64)
    rest = np.asarray(rest, dtype=np.float64)
    eyeball = np.zeros(0, np.int64) if eyeball is None else np.asarray(eyeball)
    anchors = np.unique(
        np.concatenate([m.lid_rows for m in margins.values()] + [eyeball])
    ).astype(np.int64)
    column = np.full(int(anchors.max()) + 1, -1)
    column[anchors] = np.arange(len(anchors))
    n = len(anchors)
    root_w = np.zeros((len(points), n))
    tangent_w = np.zeros((len(points), n))
    across_w = np.zeros((len(points), n))
    centre_w = np.zeros((len(points), n))
    opposite_w = np.zeros((len(points), n))
    turn = np.zeros(len(points))
    source = np.zeros(len(points), dtype=np.int64)
    if len(eyeball):
        centre_w[:, column[eyeball]] = 1.0 / len(eyeball)
    for lid in np.unique(lids):
        key = int(lid) if int(lid) in margins else next(iter(margins))
        margin = margins[key]
        mine = np.flatnonzero(lids == lid)
        cols = column[margin.rows]
        d2 = ((points[mine, None] - margin.points[None]) ** 2).sum(2)
        soft = np.exp(-(d2 - d2.min(1, keepdims=True)) / ROOT_SPREAD**2)
        soft /= soft.sum(1, keepdims=True)
        root = soft @ margin.points
        source[mine] = margin.rows[np.argmin(d2, axis=1)]
        root_w[np.ix_(mine, cols)] = soft
        # the tangent: each margin vertex's neighbours along the margin, weighted
        e2 = ((root[:, None] - margin.points[None]) ** 2).sum(2)
        w = np.exp(-(e2 - e2.min(1, keepdims=True)) / PATCH_SPREAD**2)
        w /= w.sum(1, keepdims=True)
        k = np.arange(len(margin.rows))
        ahead, behind = np.minimum(k + 1, len(k) - 1), np.maximum(k - 1, 0)
        for j in range(len(k)):
            tangent_w[mine, cols[ahead[j]]] += w[:, j]
            tangent_w[mine, cols[behind[j]]] -= w[:, j]
        # the lid's direction away from the margin: its skin near the margin, less the
        # margin, both weighted
        from scipy.spatial import cKDTree

        gap, _ = cKDTree(margin.points).query(rest[margin.lid_rows])
        near = margin.lid_rows[(gap > 1e-9) & (gap <= ACROSS_REACH)]
        if len(near) == 0:
            near = margin.lid_rows[gap > 1e-9]
        e2 = ((root[:, None] - rest[near][None]) ** 2).sum(2)
        u = np.exp(-(e2 - e2.min(1, keepdims=True)) / PATCH_SPREAD**2)
        u /= u.sum(1, keepdims=True)
        across_w[np.ix_(mine, column[near])] += u
        across_w[np.ix_(mine, cols)] -= w
        if key == UPPER and LOWER in margins and len(eyeball):
            lower = margins[LOWER]
            d2 = ((root[:, None] - lower.points[None]) ** 2).sum(2)
            soft = np.exp(-(d2 - d2.min(1, keepdims=True)) / ROOT_SPREAD**2)
            soft /= soft.sum(1, keepdims=True)
            opposite_w[np.ix_(mine, column[lower.rows])] = soft
            turn[mine] = CLOSE_TURN
    upper = turn > 0
    if not upper.any():
        opposite_w[:] = root_w  # unused: no lashes turn
    else:
        opposite_w[~upper] = root_w[~upper]
    return (
        anchors,
        root_w,
        source,
        PatchFollow(
            anchors,
            rest[anchors],
            root_w,
            tangent_w,
            across_w,
            centre_w if len(eyeball) else None,
            opposite_w if len(eyeball) else None,
            turn if len(eyeball) else None,
        ),
    )


# --------------------------------------------------------------------------- generic lashes
# the generic lashes: segments of each strip along the lid, rows from root to tip, and the
# share of the margin that they cover from the nasal corner (they start a little away from it)
GENERIC_SEGMENTS = 16
GENERIC_ROWS = 3
GENERIC_SPAN = (0.12, 0.98)
# the generic texture: the upper lashes' strands on its left half, the lower lashes' on its
# right half
GENERIC_TEXTURE = (2048, 128)
# the lower lashes against the upper ones: the share of strands, their width and their opacity
GENERIC_LOWER = (0.45, 0.75, 0.6)
# how far (metres) the generic lashes' roots sit from the lid margin toward the opening: the
# margin's vertices lie on the lid's outer skin a little away from its rim
GENERIC_ROOT_LIFT = 0.0008


def _smooth_curve(points: np.ndarray, passes: int = 3) -> np.ndarray:
    out = np.asarray(points, dtype=np.float64).copy()
    for _ in range(passes):
        out[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
    return out


def _resample(curve: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """Points of a polyline at shares (0 to 1) of its length."""
    step = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    along = np.r_[0.0, np.cumsum(step)]
    total = max(along[-1], 1e-12)
    return np.stack(
        [np.interp(shares * total, along, curve[:, k]) for k in range(3)], 1
    )


def generic_lashes(frames: dict, margins: dict, settings) -> LashData:
    """
    Generic lashes on the lid margins of each eye (``frames`` and ``margins`` by side, as
    :func:`lid_margins` finds them; ``settings`` a :class:`~opensculptboy.material.
    LashMaterial`): along each lid, a strip of cards that leaves the margin a little in front
    of it, runs outward from the eye and curls up (the upper lid) or down (the lower lid). Its
    rows carry the texture's strands from root (v = 0) to tip (v = 1); the strip is longest
    toward the temple and shorter at the nasal corner, and the lower lashes are
    ``lower_length_mm`` long.
    """
    points, normals, uvs, triangles, lids = [], [], [], [], []
    count = 0
    lengths = {UPPER: settings.length_mm, LOWER: settings.lower_length_mm}
    for side, frame in frames.items():
        for lid, margin in margins.get(side, {}).items():
            if lid == LOWER and not settings.lower:
                continue
            curve = _smooth_curve(margin.points)
            shares = np.linspace(*GENERIC_SPAN, GENERIC_SEGMENTS + 1)
            roots = _resample(curve, shares)
            tangent = np.gradient(roots, axis=0)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
            radial = roots - frame.centre
            radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
            up = frame.axes[:, 1] * (1.0 if lid == UPPER else -1.0)
            # the lid's own up: across the margin, in the plane of the eye's surface there
            across = up[None] - (up[None] * radial).sum(1, keepdims=True) * radial
            across -= (across * tangent).sum(1, keepdims=True) * tangent
            across /= np.maximum(np.linalg.norm(across, axis=1, keepdims=True), 1e-12)
            # longer toward the temple, short at the ends
            u = (shares - shares[0]) / (shares[-1] - shares[0])
            profile = np.sin(np.pi * np.clip(0.15 + 0.8 * u, 0, 1)) ** 0.6
            length = lengths[lid] / 1000.0 * (0.45 + 0.55 * profile)
            out_dir = radial * np.cos(np.radians(25.0)) + across * np.sin(
                np.radians(25.0)
            )
            # the roots sit on the rim of the lid: from the margin, a little toward the
            # opening and out of the skin
            base = roots - across * GENERIC_ROOT_LIFT + radial * 0.0003
            for row in range(GENERIC_ROWS + 1):
                f = row / GENERIC_ROWS
                curl = settings.curl * f * f
                p = base + length[:, None] * (f * out_dir + curl * across)
                points.append(p)
                # each lid's strands take their half of the texture, a texel in from its edges
                half = 0.0 if lid == UPPER else 0.5
                inset = 2.0 / GENERIC_TEXTURE[0]
                uvs.append(
                    np.stack(
                        [half + inset + (0.5 - 2 * inset) * u, np.full(len(u), f)], 1
                    )
                )
                lids.append(np.full(len(u), lid))
                normals.append(
                    np.cross(tangent, out_dir) * (1.0 if lid == UPPER else -1.0)
                )
            width = GENERIC_SEGMENTS + 1
            for row in range(GENERIC_ROWS):
                for k in range(GENERIC_SEGMENTS):
                    a = count + row * width + k
                    b, c, d = a + 1, a + width, a + width + 1
                    triangles += [[a, b, d], [a, d, c]]
            count += width * (GENERIC_ROWS + 1)
    if not points:
        return LashData(
            np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 3)), []
        )
    normals = np.concatenate(normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return LashData(
        np.concatenate(points),
        normals,
        np.concatenate(uvs),
        np.asarray(triangles),
        np.concatenate(lids),
    )


def generic_texture(settings, size=GENERIC_TEXTURE) -> np.ndarray:
    """
    The generic lashes' texture (H, W, 4, 8-bit sRGB and alpha): tapered strands from the
    bottom edge (the root, v = 0) toward the top (the tip) in ``settings.color``, each slanting
    a little and curving; the alpha falls to nothing at the tips. The left half holds the upper
    lashes' strands and the right half the lower lashes', fewer, finer and lighter
    (:data:`GENERIC_LOWER`).
    """
    from opensculptboy.eyes.texture import srgb

    w, h = size
    half = w // 2
    rng = np.random.default_rng(int(settings.seed))
    yy, xx = np.mgrid[0:h, 0:half]
    v = 1.0 - (yy + 0.5) / h  # 0 at the root (bottom), 1 at the tip
    halves = []
    for share, thin, opacity in ((1.0, 1.0, 1.0), GENERIC_LOWER):
        alpha = np.zeros((h, half))
        for _ in range(int(round(settings.density * share * half / 4.0))):
            x0 = rng.uniform(0, half)
            reach = rng.uniform(0.55, 1.0)
            slant = rng.normal(0.0, 0.08) * half / 10.0
            bend = rng.normal(0.0, 0.05) * half / 10.0
            width = rng.uniform(1.6, 2.6) * thin
            along = np.clip(v / reach, 0.0, 1.0)
            centre = x0 + slant * along + bend * along**2
            dx = np.abs(((xx - centre + half / 2) % half) - half / 2)
            taper = width * (1.0 - 0.85 * along)
            strand = np.clip(1.0 - dx / np.maximum(taper, 0.3), 0.0, 1.0)
            strand *= (v <= reach) * (1.0 - along**3)
            alpha = np.maximum(alpha, strand)
        halves.append(alpha * opacity)
    alpha = np.concatenate(halves, 1)
    colour = np.empty((h, w, 3))
    colour[...] = np.asarray(settings.color, dtype=np.float64)
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[..., :3] = srgb(colour)
    out[..., 3] = np.round(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    return out


# ------------------------------------------------------------------------- reference lashes
def reference_lashes(parts, posed: np.ndarray, rest: np.ndarray, head: np.ndarray):
    """
    The lashes of a reference (``parts``, :class:`~opensculptboy.fit.handoff.HandoffPart` of
    any role; those with the role ``lashes`` count) in the rest frame of a fitted mesh, with
    their texture: (:class:`LashData`, texture (H, W, 4) uint8, ``{"roughness": r}``), or
    None without lashes. The parts' textures are stacked from the top in the order of the
    parts, at the width of the widest, and their UVs follow them. ``posed`` and ``rest`` are
    the fitted mesh's vertices as the fit placed it on the reference and at rest, and
    ``head`` the rows of its head (:func:`opensculptboy.teeth.mesh.mouth_rows`). A part
    whose points lie mostly above the eyes' level (its median above the median of all lash
    points) is an upper lid's.
    """
    from PIL import Image

    from opensculptboy.export.body import vertex_normals
    from opensculptboy.teeth.mesh import rigid

    lashes = [
        p
        for p in parts
        if p.role == "lashes" and len(p.triangles) and p.uvs is not None
    ]
    if not lashes:
        return None
    images = []
    for p in lashes:
        image = p.base_color
        if image is None:
            image = np.zeros((4, 4, 4), np.uint8)
            image[..., 3] = 255
        image = np.asarray(image)
        if image.shape[-1] == 3:
            image = np.concatenate(
                [image, np.full(image.shape[:2] + (1,), 255, image.dtype)], -1
            )
        images.append(image.astype(np.uint8))
    width = max(i.shape[1] for i in images)
    scaled = []
    for image in images:
        if image.shape[1] != width:
            height = max(1, round(image.shape[0] * width / image.shape[1]))
            image = np.asarray(
                Image.fromarray(image, "RGBA").resize((width, height), Image.LANCZOS)
            )
        scaled.append(image)
    atlas = np.concatenate(scaled, axis=0)
    total = atlas.shape[0]
    R, t = rigid(posed[head], rest[head])
    points, normals, uvs, triangles, lids = [], [], [], [], []
    count, top = 0, 0
    heights = np.concatenate([p.vertices[:, 2] for p in lashes])
    level = float(np.median(heights))
    for p, image in zip(lashes, scaled):
        v = np.asarray(p.vertices, dtype=np.float64)
        n = (
            np.asarray(p.normals, dtype=np.float64)
            if p.normals is not None
            else vertex_normals(v, p.triangles)
        )
        uv = np.asarray(p.uvs, dtype=np.float64)  # glTF: v = 0 at the top
        v_top = (top + uv[:, 1] * image.shape[0]) / total
        uvs.append(np.stack([uv[:, 0], 1.0 - v_top], 1))
        points.append(v @ R.T + t)
        normals.append(n @ R.T)
        triangles.append(np.asarray(p.triangles, dtype=np.int64) + count)
        lid = UPPER if float(np.median(v[:, 2])) >= level else LOWER
        lids.append(np.full(len(v), lid))
        count += len(v)
        top += image.shape[0]
    normals = np.concatenate(normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    roughness = [
        float(np.median(p.roughness)) for p in lashes if p.roughness is not None
    ]
    values = {"roughness": round(float(np.median(roughness)), 4)} if roughness else {}
    data = LashData(
        np.concatenate(points),
        normals,
        np.concatenate(uvs),
        np.concatenate(triangles),
        np.concatenate(lids),
    )
    return data, atlas, values


# ---------------------------------------------------------------------------- the eye parts
def lash_parts(lashes, frames, rest, normals, skin, skin_triangles, eyeballs=None):
    """
    The lashes as parts of an eyed mesh (:class:`~opensculptboy.eyes.mesh.EyePart`, named
    ``"lashes"``, one per eye and lid) and the lashes they hold: ``lashes`` is (a
    :class:`~opensculptboy.material.LashMaterial`, the reference's :class:`LashData` or None
    for generic lashes); ``frames`` and ``eyeballs`` give each eye's frame and the rows of its
    eyeball, ``rest`` and ``normals`` the mesh's rest vertices and normals, ``skin`` the mask
    of its vertices that are not on an eyeball and ``skin_triangles`` its other triangles.
    An eye without a lid margin gets no lashes.
    """
    from opensculptboy.eyes.mesh import EyePart

    settings, data = lashes
    margins = {
        side: lid_margins(frame, rest, normals, skin, skin_triangles)
        for side, frame in frames.items()
    }
    if data is None:
        data = generic_lashes(frames, margins, settings)
    parts = []
    if len(data.points) == 0:
        return parts, data
    sides = side_of_points(data.points, frames)
    for side in frames:
        if not margins.get(side):
            continue
        for lid in (UPPER, LOWER):
            mine = (sides == side) & (data.lid == lid)
            tri = data.triangles[mine[data.triangles].all(1)]
            if len(tri) == 0:
                continue
            rows = np.unique(tri)
            local = np.full(len(data.points), -1)
            local[rows] = np.arange(len(rows))
            lids = np.full(len(rows), lid)
            if lid not in margins[side]:
                lids[:] = next(iter(margins[side]))
            anchors, blend, source, follow = bind(
                data.points[rows],
                lids,
                margins[side],
                rest,
                None if eyeballs is None else eyeballs.get(side),
            )
            if eyeballs is not None:
                _keep_outside(
                    follow,
                    data.points[rows],
                    frames[side],
                    rest,
                    normals,
                    skin,
                    skin_triangles,
                )
            parts.append(
                EyePart(
                    name="lashes",
                    side=side,
                    rest=data.points[rows],
                    normals=data.normals[rows],
                    uvs=data.uvs[rows],
                    triangles=local[tri],
                    anchors=anchors,
                    operator=None,
                    blend=blend,
                    source=source,
                    follow=follow,
                )
            )
    return parts, data


def _keep_outside(follow, points, frame, rest, normals, skin, skin_triangles) -> None:
    """Give ``follow`` the outward-facing skin around the eye of ``frame`` (within
    :data:`SKIN_REACH` eyeball radii, in front of the eye's centre, the socket left out) and
    each point's floor: :data:`CLEARANCE`, or its distance from that skin at rest."""
    from opensculptboy.eyes import geometry

    local = frame.to_local(rest)
    dist = np.linalg.norm(local, axis=1)
    inner = geometry.socket(frame, rest, normals, skin)
    outer = skin & ~inner & (dist < SKIN_REACH) & (local[:, 2] > -0.2)
    tris = np.asarray(skin_triangles)[outer[np.asarray(skin_triangles)].all(1)]
    if len(tris) < 4:
        return
    rows = np.unique(tris)
    local_index = np.full(len(rest), -1)
    local_index[rows] = np.arange(len(rows))
    follow.skin_rows = rows
    follow.skin_triangles = local_index[tris]
    depth, _ = follow.skin_depth(points, rest[rows])
    follow.floor = np.minimum(CLEARANCE, depth)


# --------------------------------------------------------------------------------- the card
def card_lashes(settings, detail_dir=None):
    """
    The reference's lashes of a card's lashes (a :class:`~opensculptboy.material.
    LashMaterial`) and their texture as PNG bytes, or (None, None) for generic lashes, which
    :func:`generic_lashes` lays on the lid margins of the file's mesh.
    """
    if settings.source != "reference":
        return None, None
    folder = pathlib.Path(detail_dir or ".")
    data = LashData.load(folder / settings.mesh)
    texture = (
        (folder / settings.base_color).read_bytes() if settings.base_color else None
    )
    return data, texture


def lash_texture(settings, texture: bytes | None = None) -> bytes:
    """The lashes' texture as PNG bytes: the reference's (``texture``), else the generic
    strands of :func:`generic_texture`."""
    if texture is not None:
        return texture
    from opensculptboy.export.document import png_bytes

    return png_bytes(generic_texture(settings))


def eyed_lashes(character, detail_dir=None):
    """The lashes of a card for :func:`~opensculptboy.eyes.mesh.eyed_mesh` and their
    texture's PNG bytes: ((settings, the reference's :class:`LashData` or None), bytes), or
    (None, None) for a card without lashes."""
    settings = character.lash_material()
    if settings is None:
        return None, None
    data, texture = card_lashes(settings, detail_dir)
    return (settings, data), lash_texture(settings, texture)


def side_of_points(points, frames) -> np.ndarray:
    """The side ("L" or "R") of each point: the eye whose centre lies nearest."""
    sides = list(frames)
    dist = np.stack(
        [np.linalg.norm(points - frames[s].centre, axis=1) for s in sides], 1
    )
    return np.asarray(sides)[np.argmin(dist, axis=1)]

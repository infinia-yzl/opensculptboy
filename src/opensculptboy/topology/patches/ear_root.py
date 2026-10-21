# OpenSculptBoy
# Apache License, Version 2.0
"""
The ear-root patch: a denser, regular band of quads along the root of each outer ear.

Anny's outer ear meets the head across a single ring of coarse quads, which must bend from the
cheek into the ear in one step. On a fitted character whose ear is a separate piece stitched to
the head, the fit parts the ear from the head along that ring, and the coarse quads leave a
crack along the front of the attachment line. This patch replaces the root with a band that
can bend smoothly and gives the fit a clean line to part along.

**The removed faces.** Anny's outer ear proper is :func:`pinna_mask`, by the rule of
:func:`opensculptboy.fit.register.pinna` on the default body: of the vertices that the ear
translation targets move in full, those more than 2 mm off a quadratic fitted to the head's
skin around the ear, with the hole they enclose (the concha). The *strip* is the ring of quads
that joins the pinna to the head (each has vertices on both), all the way around the ear: the
front of the helix root, the tragus, the lobe and the back. The patch removes the strip, one
ring of the head's quads around it (with the quads that fill the notches of that ring's outer
border) and one ring of the pinna's quads inside it (:func:`root_band`): on anny, 32 + 38 +
32 quads per ear, between the head's border loop (outside) and a loop of the pinna (inside).
Anny's vertices inside the band are left without faces.

**The band.** The band is all quads. Its middle row is the *root line*: 64 vertices evenly
spaced along the middle of the strip, smoothed over about two of anny's edges, where the
head's skin turns into the ear. Two rows run on each side of it, at 0.3 and 0.6 of the way to
the band's border in the same column, each of 64 vertices and joined by regular quads. Each
border loop then joins its outer row through transition units, chosen by a dynamic programme
so that the border vertices meet the row's columns: one border edge to one row edge (a quad),
one to three (four quads, two new vertices), or two to four (six quads, three new vertices)::

    a---b---c---d---e      [A, B, Y, X]  [B, C, Z, Y]
    |   |   |   |   |      [A, X, b, a]  [X, Y, c, b]
    |   X---Y---Z   |      [Y, Z, d, c]  [Z, C, e, d]
    |  /    |    \\  |
    A-------B-------C

Every vertex of the rows and the root line has four edges. The columns are the arc length
along each loop, in phase with the angle about the ear.

**On anny's surface.** Each new vertex is the closest point of anny's surface (the triangles
of the removed faces, split along the shorter diagonal as
:func:`anny.utils.mesh_utils.triangulate_faces` splits them) to its place in its column, then
relaxed toward its neighbours and moved, where a corner angle of its faces is poor, by a search
along the surface (:func:`_untangle`, in 3D and in the atlas). It is the barycentric
combination of the corners of its triangle: at most three anchors, so every blend shape, local
change, face shape and skin weight carries over. Its UV is the same combination of the
corners' UVs: the removed faces lie in one UV chart, so the band covers their region of the
atlas. The vertices lie on anny's surface; the band's faces are chords of it, which round
anny's sharpest creases at the root (folds of up to 100 degrees between its coarse quads) by
up to about 1.7 mm (0.24 mm at the 95th percentile on the default body).

**The root line.** The vertices of the root line carry the label :data:`ROOT_LABEL`
(``"ear_root/line"``, ``opensculptboy.topology.lookup.EAR_ROOT_LINE_LABEL``; the others
``"ear_root"``), stored in loop order, the left ear's first: :func:`root_lines` gives them as
the vertex loop of each ear, anny's side of the attachment line of the reference fit.

The patch is authored on the left ear (x > 0) and mirrored.
"""

from __future__ import annotations

import numpy as np

from opensculptboy.topology.patch import BaseQuadMesh, Patch, PatchError, blend_rows

NAME = "ear_root"
ORDER = 200
ENABLED = True
OPTIONS = {
    "pinna_height": 0.002,
    "core_weight": 0.99,
    "head_rings": 1,
    "pinna_rings": 1,
    "head_rows": [0.3, 0.6],
    "pinna_rows": [0.3, 0.6],
    "relax": 20,
    "unit_penalty": 1.0,
    "root_smoothing": 2.0,
    "untangle": 10,
    "root_across": 0.5,
}

# The label of the root-line vertices.
ROOT_LABEL = "ear_root/line"
# The sine of a corner angle above which the untangling leaves a vertex alone (about 37
# degrees).
GOOD_CORNER = 0.6
# Barycentric weights below this vanish (a vertex on an edge of anny keeps two anchors).
SNAP = 1e-9


def quad_triangles(vertices: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """(F, 2, 3) the two triangles of each quad, split along its shorter diagonal as anny's
    triangulation splits it (ties along b-d)."""
    from anny.utils.mesh_utils import DIAGONAL_TIE

    v, q = np.asarray(vertices, dtype=np.float64), np.asarray(quads)
    d1 = np.linalg.norm(v[q[:, 0]] - v[q[:, 2]], axis=1)
    d2 = np.linalg.norm(v[q[:, 1]] - v[q[:, 3]], axis=1)
    ac = d1 < d2 * (1.0 - DIAGONAL_TIE)
    split_ac = np.stack([q[:, [0, 1, 2]], q[:, [2, 3, 0]]], 1)
    split_bd = np.stack([q[:, [0, 1, 3]], q[:, [3, 1, 2]]], 1)
    return np.where(ac[:, None, None], split_ac, split_bd)


def ear_weights(base: BaseQuadMesh) -> np.ndarray:
    """(N,) the displacement of MakeHuman's ear translation targets (``l-ear-trans-up`` and
    ``r-ear-trans-up``) relative to their largest: 1 on the outer ear, falling to 0 over the
    skin around it (:func:`opensculptboy.fit.register.ear_weights` on a model)."""
    from anny.viewer.landmarks import read_target

    out = np.zeros(base.vertex_count)
    index = base.base_mesh_vertex_indices
    for side in "lr":
        idx, d = read_target(f"ears/{side}-ear-trans-up")
        m = np.linalg.norm(d, axis=1)
        full = np.zeros(int(max(index.max(), idx.max())) + 1)
        full[idx] = m / m.max()
        out = np.maximum(out, full[index])
    return out


def _vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    v, t = vertices, triangles
    n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    out = np.zeros_like(v)
    for k in range(3):
        np.add.at(out, t[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-20)


def default_body(base: BaseQuadMesh) -> np.ndarray:
    """(N, 3) the rest vertices of anny's default body (every phenotype at its default) on the
    quad mesh of ``base``, where the patch picks its faces and lays out its band."""
    import anny
    import torch

    model = anny.Anny(
        topology="anny-quads",
        local_changes="none",
        face_shapes="none",
        facial_actions="none",
    ).to(dtype=torch.float64)
    if not np.array_equal(model.faces.cpu().numpy(), base.quads):
        raise PatchError("anny-quads does not have the faces of the base mesh.")
    with torch.no_grad():
        return model(phenotype_kwargs={})["rest_vertices"][0].cpu().numpy()


def pinna_mask(
    base: BaseQuadMesh,
    vertices: np.ndarray,
    height: float = OPTIONS["pinna_height"],
    core_weight: float = OPTIONS["core_weight"],
) -> np.ndarray:
    """
    (N,) anny's outer ears proper on ``vertices`` (anny's quad mesh), by the rule of
    :func:`opensculptboy.fit.register.pinna`: of the vertices that the ear translation targets
    move in full (:func:`ear_weights` over ``core_weight``), those more than ``height`` off a
    quadratic fitted to the head's skin around the ear (weights between 0.2 and 0.95), with
    the holes that they enclose (the concha).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    v = np.asarray(vertices, dtype=np.float64)
    tri = quad_triangles(base.vertices, base.quads).reshape(-1, 3)
    w = ear_weights(base)
    core = w > core_weight
    normals = _vertex_normals(v, tri)
    high = np.zeros(len(v), bool)
    for s in (1.0, -1.0):
        side = np.sign(v[:, 0]) == s
        ring = side & (w > 0.2) & (w < 0.95)
        rows = np.nonzero(side & core)[0]
        if ring.sum() < 6 or len(rows) == 0:
            raise PatchError("The ear translation targets do not mark anny's ears.")
        centre = v[ring].mean(0)
        normal = normals[ring].mean(0)
        normal /= np.linalg.norm(normal)
        a = np.cross(normal, [0.0, 0.0, 1.0])
        a /= np.linalg.norm(a)
        b = np.cross(normal, a)

        def local(x):
            d = x - centre
            u, t = d @ a, d @ b
            return np.c_[np.ones_like(u), u, t, u * u, u * t, t * t], d @ normal

        m, z = local(v[ring])
        coef = np.linalg.lstsq(m, z, rcond=None)[0]
        m, z = local(v[rows])
        high[rows] = z - m @ coef > height
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    adjacency = coo_matrix(
        (np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(v), len(v))
    ).tocsr()
    low = np.nonzero(~high)[0]
    _, labels = connected_components(adjacency[low][:, low], directed=False)
    open_ = np.zeros(labels.max() + 1, bool)
    open_[labels[~core[low]]] = True
    out = high.copy()
    out[low[~open_[labels]]] = True
    return out


def _loops(quads: np.ndarray) -> list[list[int]]:
    """The boundary loops of a set of quads, each in the direction its faces run along it."""
    edges = {}
    for q in quads.tolist():
        for i in range(4):
            edges[(q[i], q[(i + 1) % 4])] = True
    nxt = {}
    for a, b in edges:
        if (b, a) not in edges:
            if a in nxt:
                raise PatchError("The ear band's border touches itself.")
            nxt[a] = b
    loops, seen = [], set()
    for start in sorted(nxt):
        if start in seen:
            continue
        loop, v = [], start
        while v not in seen:
            seen.add(v)
            loop.append(v)
            v = nxt[v]
        if v != start:
            raise PatchError("The ear band's border is not a set of loops.")
        loops.append(loop)
    return loops


def root_band(
    base: BaseQuadMesh,
    pinna: np.ndarray,
    head_rings: int = OPTIONS["head_rings"],
    pinna_rings: int = OPTIONS["pinna_rings"],
) -> dict:
    """
    The root of the left ear (x > 0): ``strip``, the quads that join the pinna to the head
    (each has vertices in both), with its borders ``strip_outer`` (on the head) and
    ``strip_inner`` (the pinna's border); ``faces``, the strip with ``head_rings`` rings of the
    head's quads around it and ``pinna_rings`` rings of the pinna's quads inside it, and their
    borders ``outer`` and ``inner``. Each loop runs in the direction its faces run along it.
    """
    q = base.quads
    side = (base.vertices[q][..., 0] > 0).all(1)
    on = pinna[q]
    strip = np.nonzero(side & on.any(1) & ~on.all(1))[0]

    def borders(faces):
        loops = _loops(q[faces])
        inner = [lp for lp in loops if pinna[lp].all()]
        outer = [lp for lp in loops if not pinna[lp].any()]
        if len(loops) != 2 or len(inner) != 1 or len(outer) != 1:
            raise PatchError(
                "The root of anny's ear is not a ring of faces around the pinna."
            )
        return np.array(outer[0]), np.array(inner[0])

    def fill(faces, allowed):
        # fill the notches of the border: add the allowed faces around each border vertex
        # that the band holds on two more faces than it leaves
        for _ in range(2):
            inside = np.zeros(len(q), bool)
            inside[faces] = True
            more = []
            for x in np.unique(q[faces]).tolist():
                around = base.vertex_faces[x]
                held = inside[around].sum()
                left = around[~inside[around]]
                if len(left) and held >= len(left) + 2 and allowed[left].all():
                    more += left.tolist()
            if not more:
                break
            faces = np.union1d(faces, more)
        return faces

    strip_outer, strip_inner = borders(strip)
    faces, outer, inner = strip, strip_outer, strip_inner
    for _ in range(head_rings):
        more = np.nonzero(side & np.isin(q, outer).any(1) & ~on.any(1))[0]
        faces = fill(np.union1d(faces, more), side & ~on.any(1))
        outer, inner = borders(faces)
    for _ in range(pinna_rings):
        more = np.nonzero(side & np.isin(q, inner).any(1) & on.all(1))[0]
        faces = np.union1d(faces, more)
        outer, inner = borders(faces)
    return {
        "faces": np.concatenate([strip, np.setdiff1d(faces, strip)]),
        "strip": strip,
        "outer": outer,
        "inner": inner,
        "strip_outer": strip_outer,
        "strip_inner": strip_inner,
    }


def _wrap(x):
    """Periodic differences (turns) into [-0.5, 0.5)."""
    return (np.asarray(x) + 0.5) % 1.0 - 0.5


def _unwrap(x):
    """Increasing periodic columns (turns) without their jumps."""
    x = np.asarray(x, dtype=np.float64)
    return x[0] + np.r_[0.0, np.cumsum(np.mod(np.diff(x), 1.0))]


def _turn(points, centre, axis) -> float:
    """The turns of a closed loop about ``axis`` through ``centre`` (+1 or -1 for a loop
    around the ear)."""
    a = np.cross(axis, [0.0, 0.0, 1.0])
    a /= np.linalg.norm(a)
    b = np.cross(axis, a)
    d = points - centre
    angle = np.arctan2(d @ b, d @ a) / (2 * np.pi)
    return float(np.sum(_wrap(np.diff(np.r_[angle, angle[:1]]))))


def _arc_columns(points: np.ndarray, centre, axis) -> np.ndarray:
    """
    The column (turns) of each vertex of a closed loop: its arc length along the loop over the
    loop's length, increasing along the loop, with the phase that makes the columns agree on
    average with the angle about ``axis`` through ``centre``.
    """
    seg = np.linalg.norm(np.roll(points, -1, 0) - points, axis=1)
    arc = np.r_[0.0, np.cumsum(seg)[:-1]] / seg.sum()
    a = np.cross(axis, [0.0, 0.0, 1.0])
    a /= np.linalg.norm(a)
    b = np.cross(axis, a)
    d = points - centre
    angle = np.arctan2(d @ b, d @ a) / (2 * np.pi)
    turn = np.sum(_wrap(np.diff(np.r_[angle, angle[:1]])))
    if abs(abs(turn) - 1.0) > 1e-6:
        raise PatchError("A loop of the ear band does not run once around the ear.")
    angle = np.sign(turn) * angle
    phase = np.angle(np.mean(np.exp(2j * np.pi * (angle - arc)))) / (2 * np.pi)
    return arc + phase


def _loop_at(points: np.ndarray, columns: np.ndarray, at) -> np.ndarray:
    """Points of a closed polyline whose vertices sit at increasing ``columns`` (period 1)."""
    c = columns[0] + np.r_[0.0, np.cumsum(np.mod(np.diff(columns), 1.0))]
    x = np.r_[c, c[0] + 1.0]
    pts = np.vstack([points, points[:1]])
    a = c[0] + np.mod(np.asarray(at, dtype=np.float64) - c[0], 1.0)
    return np.stack([np.interp(a, x, pts[:, d]) for d in range(3)], 1)


def _smooth_loop(points: np.ndarray, sigma: float, samples: int = 8) -> tuple:
    """A closed polyline resampled at ``samples`` points per edge by arc length and smoothed
    by a Gaussian of ``sigma`` edges: the points and their columns (arc length, period 1)."""
    n = len(points)
    cols = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    total = cols[-1] + np.linalg.norm(points[0] - points[-1])
    at = np.arange(n * samples) / (n * samples)
    dense = _loop_at(points, cols / total, at)
    k = np.arange(len(at))
    k = np.minimum(k, len(at) - k)
    g = np.exp(-0.5 * (k / (sigma * samples)) ** 2)
    g = np.fft.fft(g / g.sum())
    smooth = np.real(np.fft.ifft(np.fft.fft(dense, axis=0) * g[:, None], axis=0))
    return smooth, at


def _closest_on_triangles(points, vertices, triangles):
    """The closest point of a set of triangles to each point: the triangle (index), its
    barycentric weights (n, 3) and the distance (Ericson, *Real-Time Collision Detection*)."""
    p = np.asarray(points, dtype=np.float64)[:, None]
    a, b, c = (vertices[triangles[:, k]][None] for k in range(3))
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    def ratio(x, y):
        return x / np.where(y == 0, 1e-300, y)

    den = va + vb + vc
    w1, w2 = ratio(vb, den), ratio(vc, den)
    w = np.stack([1 - w1 - w2, w1, w2], -1)
    zero, one = np.zeros_like(d1), np.ones_like(d1)
    cases = [
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), ratio(d1, d1 - d3), "ab"),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), ratio(d2, d2 - d6), "ac"),
        (
            (va <= 0) & (d4 >= d3) & (d5 >= d6),
            ratio(d4 - d3, (d4 - d3) + (d5 - d6)),
            "bc",
        ),
    ]
    for mask, u, edge in cases:
        val = {
            "ab": np.stack([1 - u, u, zero], -1),
            "ac": np.stack([1 - u, zero, u], -1),
            "bc": np.stack([zero, 1 - u, u], -1),
        }[edge]
        w = np.where(mask[..., None], val, w)
    for mask, val in (
        ((d1 <= 0) & (d2 <= 0), np.stack([one, zero, zero], -1)),
        ((d3 >= 0) & (d4 <= d3), np.stack([zero, one, zero], -1)),
        ((d6 >= 0) & (d5 <= d6), np.stack([zero, zero, one], -1)),
    ):
        w = np.where(mask[..., None], val, w)
    q = w[..., 0:1] * a + w[..., 1:2] * b + w[..., 2:3] * c
    dist = np.linalg.norm(q - p, axis=-1)
    best = np.argmin(dist, 1)
    rows = np.arange(len(p))
    return best, w[rows, best], dist[rows, best]


def _on_surface(points, vertices, triangles):
    """The closest points of the triangles to the points."""
    best, bary, _ = _closest_on_triangles(points, vertices, triangles)
    return np.einsum("nk,nkd->nd", bary, vertices[triangles[best]])


def _corner_sines(q3, q2):
    """
    The sine of each corner angle of quads, (..., 4): in 3D about the normal of the quad (the
    cross product of its diagonals), and in the atlas; ``q3`` (..., 4, 3), ``q2`` (..., 4, 2).
    """
    normal = np.cross(q3[..., 2, :] - q3[..., 0, :], q3[..., 3, :] - q3[..., 1, :])
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-30)
    a3 = np.roll(q3, 1, axis=-2) - q3
    b3 = np.roll(q3, -1, axis=-2) - q3
    s3 = (np.cross(b3, a3) * normal[..., None, :]).sum(-1) / np.maximum(
        np.linalg.norm(a3, axis=-1) * np.linalg.norm(b3, axis=-1), 1e-30
    )
    a2 = np.roll(q2, 1, axis=-2) - q2
    b2 = np.roll(q2, -1, axis=-2) - q2
    s2 = (b2[..., 0] * a2[..., 1] - b2[..., 1] * a2[..., 0]) / np.maximum(
        np.linalg.norm(a2, axis=-1) * np.linalg.norm(b2, axis=-1), 1e-30
    )
    # an edge much shorter than the quad's mean edge counts as a poor corner too
    length = np.linalg.norm(b3, axis=-1)
    short = 4.0 * length / np.maximum(length.mean(-1, keepdims=True), 1e-30) - 1.0
    return np.minimum(np.minimum(s3, s2), short)


def _untangle(
    targets, faces, n_vertices, moving, vertices, triangles, uv_of_vertex, uvs, rounds
):
    """
    Raise the worst corner of the faces around each moving vertex: a search over steps along
    anny's surface (eight directions, shrinking) that keeps a step when it raises the
    smallest sine of a corner angle of those faces, in 3D and in the atlas.
    """
    p = np.array(targets)
    best, bary, _ = _closest_on_triangles(p, vertices, triangles)
    uv_tri = uvs[uv_of_vertex[triangles]]  # (T, 3, 2)
    puv = np.einsum("nk,nkd->nd", bary, uv_tri[best])
    P3 = np.vstack([vertices, p])
    P2 = np.vstack([uvs[uv_of_vertex], puv])
    incident = [[] for _ in range(len(p))]
    for f, face in enumerate(faces.tolist()):
        for x in face:
            if x >= n_vertices:
                incident[x - n_vertices].append(f)
    angles = np.arange(8) * np.pi / 4
    centroids = vertices[triangles].mean(1)
    for _ in range(rounds):
        changed = False
        for i in moving:
            row = n_vertices + i
            fq = faces[incident[i]]  # (k, 4)
            here = _corner_sines(P3[fq], P2[fq]).min()
            if here > GOOD_CORNER:
                continue
            ring = np.setdiff1d(np.unique(fq), [row])
            h0 = 0.3 * np.mean(np.linalg.norm(P3[ring] - P3[row], axis=1))
            tri = vertices[triangles[best[i]]]
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            normal /= np.linalg.norm(normal)
            e1 = np.cross(
                normal, [0.0, 0.0, 1.0] if abs(normal[2]) < 0.9 else [1.0, 0, 0]
            )
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(normal, e1)
            for h in (3 * h0, h0, h0 / 3, h0 / 9, h0 / 27):
                cand = P3[row] + h * (
                    np.cos(angles)[:, None] * e1 + np.sin(angles)[:, None] * e2
                )
                near = np.argsort(np.linalg.norm(centroids - P3[row], axis=1))[:32]
                cb, cw, _ = _closest_on_triangles(cand, vertices, triangles[near])
                cb = near[cb]
                c3 = np.einsum("ck,ckd->cd", cw, vertices[triangles[cb]])
                c2 = np.einsum("ck,ckd->cd", cw, uv_tri[cb])
                q3 = np.broadcast_to(P3[fq], (len(cand),) + fq.shape + (3,)).copy()
                q2 = np.broadcast_to(P2[fq], (len(cand),) + fq.shape + (2,)).copy()
                at = fq == row
                q3[:, at] = c3[:, None]
                q2[:, at] = c2[:, None]
                score = _corner_sines(q3, q2).reshape(len(cand), -1).min(1)
                k = int(np.argmax(score))
                if score[k] > here + 1e-9:
                    here, best[i], bary[i] = score[k], cb[k], cw[k]
                    P3[row], P2[row] = c3[k], c2[k]
                    changed = True
        if not changed:
            break
    return P3[n_vertices:]


def _template(A, B, C, X, Y, Z, a, b, c, d, e):
    """The six quads that join two border edges to four root-line edges."""
    return [
        [A, B, Y, X],
        [B, C, Z, Y],
        [A, X, b, a],
        [X, Y, c, b],
        [Y, Z, d, c],
        [Z, C, e, d],
    ]


def _plan_side(coarse, fine, penalty):
    """
    The transition units of one side: a split of the border loop (``n`` vertices at the
    increasing columns ``coarse``, unwrapped, period 1) into units of one edge to one root-line
    edge, one edge to three, or two edges to four, so that their root-line edges add up to the
    ``len(fine)`` edges of the root line (columns ``fine``, increasing, period 1) and the border
    vertices at the ends of the units meet root-line vertices in their columns. Returns the
    first root-line vertex (the one across from border vertex 0) and the units ``(kind, i, j)``
    (kind 1, 3 or 4; border vertex i and root-line vertex j, counted from that first one).
    """
    n, nf = len(coarse), len(fine)

    def fc(j):
        return fine[j % nf] + np.floor_divide(j, nf)

    best = (np.inf, None, None)
    for j0 in range(nf):
        offset = np.round(coarse[0] - fc(j0))
        if abs(coarse[0] - fc(j0) - offset) > 4.0 / nf:
            continue

        def miss(i, j):
            return (nf * (coarse[i % n] + i // n - fc(j0 + j) - offset)) ** 2

        cost = np.full((n + 1, nf + 1), np.inf)
        back = np.zeros((n + 1, nf + 1, 2), dtype=np.int64)
        cost[0, 0] = miss(0, 0)
        for i in range(n):
            for j in range(nf + 1):
                if not np.isfinite(cost[i, j]):
                    continue
                for kind, di, dj, extra in (
                    (1, 1, 1, penalty),
                    (3, 1, 3, penalty),
                    (4, 2, 4, 0.25 * miss(i + 1, j + 2)),
                ):
                    i2, j2 = i + di, j + dj
                    if i2 > n or j2 > nf:
                        continue
                    c = cost[i, j] + extra + miss(i2, j2)
                    if c < cost[i2, j2]:
                        cost[i2, j2] = c
                        back[i2, j2] = (i, j)
        if cost[n, nf] < best[0]:
            units, i, j = [], n, nf
            while i > 0:
                pi, pj = back[i, j]
                units.append((int(j - pj), int(pi), int(pj)))
                i, j = pi, pj
            best = (cost[n, nf], j0, units[::-1])
    if best[1] is None:
        raise PatchError("The ear band's border cannot be joined to the root line.")
    return best[1], best[2]


def _side_patch(
    base: BaseQuadMesh,
    pinna: np.ndarray,
    vertices: np.ndarray,
    head_rings: int = OPTIONS["head_rings"],
    pinna_rings: int = OPTIONS["pinna_rings"],
    head_rows=OPTIONS["head_rows"],
    pinna_rows=OPTIONS["pinna_rows"],
    relax: int = OPTIONS["relax"],
    unit_penalty: float = OPTIONS["unit_penalty"],
    untangle: int = OPTIONS["untangle"],
    root_across: float = OPTIONS["root_across"],
    root_smoothing: float = OPTIONS["root_smoothing"],
) -> tuple[Patch, dict]:
    """The patch of the left ear and its layout on ``vertices``."""
    band = root_band(base, pinna, head_rings, pinna_rings)
    v, n_vertices, n_uvs = vertices, base.vertex_count, base.uv_count
    outer, inner = band["outer"], band["inner"]
    n = max(len(outer), len(inner), len(band["strip_outer"]), len(band["strip_inner"]))
    nf = 2 * n  # vertices of the root line and of each row
    faces = band["faces"]
    triangles = quad_triangles(base.vertices, base.quads[faces]).reshape(-1, 3)
    centre = v[pinna & (v[:, 0] > 0)].mean(0)
    fn = np.cross(
        v[triangles[:, 1]] - v[triangles[:, 0]], v[triangles[:, 2]] - v[triangles[:, 0]]
    )
    axis = fn.sum(0) / np.linalg.norm(fn.sum(0))
    # columns: arc length along a loop over its length, in phase with the angle about the ear,
    # increasing the way the outer border runs (the inner border runs the other way)
    if _turn(v[outer], centre, axis) * _turn(v[inner], centre, axis) > 0:
        raise PatchError("The ear band's borders run the same way around the ear.")

    def loop_columns(loop):
        forward = _turn(v[loop], centre, axis) * _turn(v[outer], centre, axis) > 0
        lp = loop if forward else loop[::-1]
        return lp, _unwrap(_arc_columns(v[lp], centre, axis))

    out_lp, u_out = loop_columns(outer)
    in_lp, u_in = loop_columns(inner)  # inner border backward
    so_lp, u_so = loop_columns(band["strip_outer"])
    si_lp, u_si = loop_columns(band["strip_inner"])
    # the root curve: halfway across the strip in each column, on anny's surface, smoothed
    # over about one of anny's edges; the root line is nf vertices evenly spaced along it
    dense = np.arange(8 * n) / (8 * n)
    middle = (1.0 - root_across) * _loop_at(
        v[so_lp], u_so, dense
    ) + root_across * _loop_at(v[si_lp], u_si, dense)
    curve, curve_u = _smooth_loop(
        _on_surface(middle, v, triangles), OPTIONS["root_smoothing"]
    )
    curve = _on_surface(curve, v, triangles)
    curve_u = curve_u + np.angle(
        np.mean(np.exp(2j * np.pi * (_arc_columns(curve, centre, axis) - curve_u)))
    ) / (2 * np.pi)
    u_root = _unwrap(curve_u[0] + np.arange(nf) / nf)

    # new vertices, as (column, side, fraction of the way from the root curve to the border)
    column, sides, fraction = [], [], []

    def add(col, side, frac):
        column.append(float(col))
        sides.append(side)
        fraction.append(float(frac))
        return n_vertices + len(column) - 1

    root = [add(u_root[j], 0, 0.0) for j in range(nf)]
    rows_of = {0: root}
    rows_by_side = {1: tuple(head_rows), -1: tuple(pinna_rows)}
    for side in (1, -1):
        for k, frac in enumerate(rows_by_side[side]):
            rows_of[side * (k + 1)] = [add(u_root[j], side, frac) for j in range(nf)]
    new_faces = []
    # regular quads between consecutive rows: the outer side runs with increasing index,
    # the inner side with decreasing index
    for side in (1, -1):
        rows = rows_by_side[side]
        chain = [rows_of[side * (k + 1)] for k in range(len(rows))][::-1] + [root]
        for lower, upper in zip(chain[:-1], chain[1:]):
            for j in range(nf):
                j2 = (j + side) % nf
                new_faces.append([lower[j], lower[j2], upper[j2], upper[j]])
    # transitions from each border to its last row (the root line without rows)
    for side, loop in ((1, outer), (-1, inner)):
        rows = rows_by_side[side]
        mid = 0.5 * ((rows[-1] if len(rows) else 0.0) + 1.0)
        fine = rows_of[side * len(rows)] if len(rows) else root
        if side == 1:
            lp, uc = loop, _unwrap(_arc_columns(v[loop], centre, axis))
            if _turn(v[loop], centre, axis) * _turn(v[outer], centre, axis) < 0:
                raise PatchError("The ear band's outer border runs backward.")
            fine_seq = lambda j: j % nf  # noqa: E731
            fine_cols = u_root
        else:
            # the inner border as its faces run along it: columns decrease; negate them
            lp = loop
            uc = _unwrap(np.mod(-_arc_columns(v[loop[::-1]], centre, axis)[::-1], 1.0))
            fine_seq = lambda j: (-j) % nf  # noqa: E731
            fine_cols = _unwrap(np.mod(-u_root[(-np.arange(nf)) % nf], 1.0))
        j0, units = _plan_side(uc, fine_cols, unit_penalty)
        nl = len(lp)

        def fcol(j):
            return fine_cols[j % nf] + np.floor_divide(j, nf)

        def ccol(i):
            return uc[i % nl] + i // nl

        for unit, i, j in units:
            A, B, C = (int(lp[(i + d) % nl]) for d in range(3))
            f = [fine[fine_seq(j0 + j + d)] for d in range(5)]
            fc = [fcol(j0 + j + d) for d in range(5)]
            ca, cb, cc = ccol(i), ccol(i + 1), ccol(i + 2)
            if unit == 1:
                new_faces.append([A, B, f[1], f[0]])
            elif unit == 3:
                x = add(side * 0.5 * (ca + (cb - ca) / 3 + fc[1]), side, mid)
                y = add(side * 0.5 * (ca + 2 * (cb - ca) / 3 + fc[2]), side, mid)
                new_faces += [
                    [A, B, y, x],
                    [A, x, f[1], f[0]],
                    [x, y, f[2], f[1]],
                    [y, B, f[3], f[2]],
                ]
            else:
                x = add(side * 0.5 * (0.5 * (ca + cb) + fc[1]), side, mid)
                y = add(side * 0.5 * (cb + fc[2]), side, mid)
                z = add(side * 0.5 * (0.5 * (cb + cc) + fc[3]), side, mid)
                new_faces += _template(A, B, C, x, y, z, *f)
    new_faces = np.array(new_faces, dtype=np.int64)
    column, sides, fraction = np.array(column), np.array(sides), np.array(fraction)
    # first positions: in the column's cross-section, the fraction of the way from the root
    # curve to the border, on anny's surface
    root_at = _loop_at(curve, curve_u, column)
    border = np.where(
        (sides > 0)[:, None],
        _loop_at(v[out_lp], u_out, column),
        _loop_at(v[in_lp], u_in, column),
    )
    targets = _on_surface(
        root_at + fraction[:, None] * (border - root_at), v, triangles
    )
    # then relax all but the root line: each vertex halfway toward the mean of its
    # neighbours, back onto anny's surface each time
    neighbours = [set() for _ in range(len(targets))]
    for face in new_faces.tolist():
        for a_, b_ in zip(face, face[1:] + face[:1]):
            for x_, y_ in ((a_, b_), (b_, a_)):
                if x_ >= n_vertices:
                    neighbours[x_ - n_vertices].add(y_)
    neighbours = [np.array(sorted(x)) for x in neighbours]
    moving = np.nonzero(sides != 0)[0]
    for _ in range(relax):
        allp = np.vstack([v, targets])
        mean = np.array([allp[neighbours[i]].mean(0) for i in moving])
        moved = _on_surface(0.5 * (targets[moving] + mean), v, triangles)
        # a move that brings a vertex onto a neighbour (the surface's closest point clamps
        # at the band's corners) is refused
        for k, i in enumerate(moving):
            d = np.linalg.norm(allp[neighbours[i]] - moved[k], axis=1)
            if d.min() > 0.25 * d.mean():
                targets[i] = moved[k]
    # a vertex that sits on a neighbour (its first position clamped onto a corner of the
    # band) moves to the mean of its other neighbours
    for _ in range(5):
        allp = np.vstack([v, targets])
        for i in range(len(targets)):
            d = np.linalg.norm(allp[neighbours[i]] - targets[i], axis=1)
            if d.min() < 0.2 * d.mean():
                others = neighbours[i][d >= 0.2 * d.mean()]
                targets[i] = _on_surface(allp[others].mean(0)[None], v, triangles)[0]
    uv_of = base.face_uvs(faces)
    uv_of_vertex = np.zeros(n_vertices, dtype=np.int64)
    for a_, u_ in uv_of.items():
        uv_of_vertex[a_] = u_
    targets = _untangle(
        targets,
        new_faces,
        n_vertices,
        np.arange(len(targets)),
        v,
        triangles,
        uv_of_vertex,
        base.uvs,
        untangle,
    )
    best, bary, _ = _closest_on_triangles(targets, v, triangles)
    blend = []
    for k_tri, w in zip(best, bary):
        w = np.clip(w, 0.0, None)
        w[w < SNAP] = 0.0
        w /= w.sum()
        blend.append(
            {int(triangles[k_tri, k]): float(w[k]) for k in range(3) if w[k] > 0}
        )
    anchors, weights = blend_rows(blend)
    uv_anchors, uv_weights = blend_rows(
        [{uv_of[a]: w for a, w in b.items()} for b in blend]
    )
    uv_faces = np.where(
        new_faces >= n_vertices,
        new_faces - n_vertices + n_uvs,
        np.vectorize(lambda i: uv_of.get(int(i), -1))(new_faces),
    )
    labels = ["ear_root"] * len(blend)
    for i in root:
        labels[i - n_vertices] = ROOT_LABEL
    patch = Patch(
        name=NAME,
        removed_faces=faces,
        anchors=anchors,
        weights=weights,
        faces=new_faces,
        uv_faces=uv_faces,
        uv_anchors=uv_anchors,
        uv_weights=uv_weights,
        vertex_labels=labels,
    )
    layout = dict(band, column=column, sides=sides, fraction=fraction, curve=curve)
    return patch, layout


def build(base: BaseQuadMesh, **options) -> Patch:
    """The ear-root patch of both ears (the left authored, the right its mirror image);
    ``options`` override :data:`OPTIONS`."""
    unknown = sorted(set(options) - set(OPTIONS))
    if unknown:
        raise PatchError(f"Unknown options of the ear-root patch: {unknown}.")
    opts = {**OPTIONS, **options}
    rest = default_body(base)
    pinna = pinna_mask(base, rest, opts["pinna_height"], opts["core_weight"])
    side_opts = {
        k: x for k, x in opts.items() if k not in ("pinna_height", "core_weight")
    }
    left, _ = _side_patch(base, pinna, rest, **side_opts)
    left.options = {
        k: ([float(y) for y in x] if isinstance(x, (list, tuple)) else x)
        for k, x in opts.items()
    }
    return left.with_mirror(base)


def root_lines(vertex_labels, positions) -> dict[str, np.ndarray]:
    """
    The root line of each ear, ``{"l": (K,), "r": (K,)}``: the indices of the vertices labelled
    :data:`ROOT_LABEL`, in loop order, split by the sign of x (the left ear at x > 0).
    ``vertex_labels`` and ``positions`` (V, 3) cover the same vertices: a patch's new vertices
    (``patch.labels_of_vertices()``, ``patch.positions(base)``) or a whole patched mesh.
    """
    labels = np.asarray(list(vertex_labels))
    x = np.asarray(positions)[:, 0]
    root = labels == ROOT_LABEL
    return {
        "l": np.nonzero(root & (x > 0))[0],
        "r": np.nonzero(root & (x < 0))[0],
    }

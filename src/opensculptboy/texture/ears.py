# OpenSculptBoy
# Apache License, Version 2.0
"""
The ear tile: the outer ears unwrapped again for the detail levels, with a texture of their own.

Anny's atlas lays out its round ears; a fit that reaches a pointed ear stretches the tip's
texels several times over, and a bake then shows streaks there. For levels 1 and 2 the ear
tile gives each outer ear (the faces where MakeHuman's ear-translation targets move the ear in
full, :func:`opensculptboy.fit.register.ear_weights` over 0.5) a UV island of its own,
unwrapped from the fitted surface: a least-squares conformal map (Levy et al. 2002) refined by
as-rigid-as-possible iterations (Liu et al. 2008), so that the texels keep one size over the
whole ear, the tip included. Both islands share one scale, the left ear (+X) in the left half
of the tile and the right ear in the right half, each turned so that its long side runs along
v. Level 0 keeps anny's UVs.

:func:`ear_uv_table` unwraps the finest level of a card and returns one UV per vertex of that
level (NaN off the ears). The vertex points of Catmull-Clark come first in each level
(:mod:`anny.utils.subdivision`), so vertex ``i`` of a coarser level is vertex ``i`` of the
finer ones, and the same table gives the ear UVs of every level from 1 up; a mesh of the
region kind (the head at a finer level than the body) finds its vertices of the full level
through ``DetailTopology.rows``. :func:`with_ear_uvs` gives a mesh with the ear triangles on
the ear tile's UVs (appended after the atlas's UVs) and the mask of those triangles. The
table travels with the card as ``<card>.ear_uvs.safetensors`` (``material.ears.uvs``).
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

# width, height of the ear tile in texels: one square half per ear
EAR_TILE_SIZE = (2048, 1024)
EAR_PADDING = 0.03  # of each half, on every side
EAR_WEIGHT = 0.5  # the outer ear: ear weight above this
FORMAT = "opensculptboy/ear-uvs@1"


def _lscm(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """A least-squares conformal map (n, 2) of a disk-like triangle mesh."""
    import scipy.sparse
    import scipy.sparse.linalg

    n = len(vertices)
    p = vertices[triangles]
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    x = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-20)
    normal = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(normal, axis=1)
    y = np.cross(normal / np.maximum(2 * area[:, None], 1e-20), x)
    local = np.stack(
        [np.zeros(len(p)), (e1 * x).sum(1), (e2 * x).sum(1)], 1
    ) + 1j * np.stack([np.zeros(len(p)), np.zeros(len(p)), (e2 * y).sum(1)], 1)
    # the conformal energy of a triangle: sum_k W_k u_k = 0 with W_k = (w_{k+2} - w_{k+1})
    w = (
        np.stack(
            [
                local[:, 2] - local[:, 1],
                local[:, 0] - local[:, 2],
                local[:, 1] - local[:, 0],
            ],
            1,
        )
        / np.sqrt(np.maximum(2 * area, 1e-30))[:, None]
    )
    rows = np.repeat(np.arange(len(p)), 3)
    cols = triangles.ravel()
    M = scipy.sparse.csr_matrix((w.ravel(), (rows, cols)), shape=(len(p), n))
    # pin the two vertices furthest apart
    a = int(np.argmax(np.linalg.norm(vertices - vertices.mean(0), axis=1)))
    b = int(np.argmax(np.linalg.norm(vertices - vertices[a], axis=1)))
    free = np.setdiff1d(np.arange(n), [a, b])
    pinned = np.array([0.0 + 0j, np.linalg.norm(vertices[b] - vertices[a]) + 0j])
    A = M[:, free]
    rhs = -(M[:, [a, b]] @ pinned)
    # complex least squares as a real system
    Ar = scipy.sparse.bmat([[A.real, -A.imag], [A.imag, A.real]]).tocsr()
    br = np.concatenate([rhs.real, rhs.imag])
    sol = scipy.sparse.linalg.lsqr(Ar, br, atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    uv = np.zeros(n, complex)
    uv[free] = sol[: len(free)] + 1j * sol[len(free) :]
    uv[[a, b]] = pinned
    return np.stack([uv.real, uv.imag], 1)


def _arap(vertices, triangles, uv, iterations: int = 20) -> np.ndarray:
    """As-rigid-as-possible refinement (local rotations, global cotangent solve) of a map."""
    import scipy.sparse
    import scipy.sparse.linalg

    n = len(vertices)
    p = vertices[triangles]
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    x = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-20)
    normal = np.cross(e1, e2)
    y = np.cross(
        normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-20), x
    )
    flat = np.stack(
        [
            np.zeros((len(p), 2)),
            np.stack([(e1 * x).sum(1), (e1 * y).sum(1)], 1),
            np.stack([(e2 * x).sum(1), (e2 * y).sum(1)], 1),
        ],
        1,
    )  # (T, 3, 2) each triangle in its own plane
    # cotangent weights of the half-edges (opposite corner k)
    cot = np.zeros((len(p), 3))
    for k in range(3):
        a = flat[:, (k + 1) % 3] - flat[:, k]
        b = flat[:, (k + 2) % 3] - flat[:, k]
        cross = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        cot[:, k] = (a * b).sum(1) / np.maximum(cross, 1e-20)
    cot = np.clip(cot, 1e-6, 1e6)
    rows_, cols_, W = [], [], []
    for k in range(3):
        i, j = triangles[:, (k + 1) % 3], triangles[:, (k + 2) % 3]
        rows_ += [i, j, i, j]
        cols_ += [j, i, i, j]
        W += [-cot[:, k], -cot[:, k], cot[:, k], cot[:, k]]
    L = scipy.sparse.csr_matrix(
        (np.concatenate(W), (np.concatenate(rows_), np.concatenate(cols_))),
        shape=(n, n),
    )
    L = (L + scipy.sparse.identity(n) * 1e-9).tocsc()
    anchor = 0
    keep = np.setdiff1d(np.arange(n), [anchor])
    solve = scipy.sparse.linalg.factorized(L[keep][:, keep].tocsc())
    uv = np.asarray(uv, np.float64).copy()
    for _ in range(iterations):
        q = uv[triangles]
        # local step: the rotation that best maps each flat triangle onto its UV triangle
        S = np.zeros((len(p), 2, 2))
        for k in range(3):
            fa = flat[:, (k + 1) % 3] - flat[:, (k + 2) % 3]
            ua = q[:, (k + 1) % 3] - q[:, (k + 2) % 3]
            S += cot[:, k, None, None] * np.einsum("ti,tj->tij", ua, fa)
        U, _, Vt = np.linalg.svd(S)
        R = U @ Vt
        flip = np.linalg.det(R) < 0
        U[flip, :, 1] *= -1
        R = U @ Vt
        # global step: L uv = b
        b = np.zeros((n, 2))
        for k in range(3):
            i, j = triangles[:, (k + 1) % 3], triangles[:, (k + 2) % 3]
            d = np.einsum("tij,tj->ti", R, flat[:, (k + 1) % 3] - flat[:, (k + 2) % 3])
            np.add.at(b, i, cot[:, k, None] * d)
            np.add.at(b, j, -cot[:, k, None] * d)
        rhs = b[keep] - L[keep][:, [anchor]] @ uv[[anchor]]
        uv[keep] = np.stack([solve(rhs[:, 0]), solve(rhs[:, 1])], 1)
    return uv


def unwrap(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """A map (n, 2) in metres of a disk-like mesh: LSCM, then ARAP, centred and turned so that
    its long side runs along v."""
    uv = _lscm(vertices, triangles)

    # LSCM is conformal up to scale: bring it to the surface's own size before ARAP
    def area(points):
        q = points[triangles]
        a, b = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]
        return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()

    p = vertices[triangles]
    surface = (
        0.5
        * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum()
    )
    uv *= np.sqrt(surface / max(area(uv), 1e-30))
    uv = _arap(vertices, triangles, uv)
    # orientation: counter-clockwise like the mesh seen from outside
    q = uv[triangles]
    signed = (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1]) - (
        q[:, 1, 1] - q[:, 0, 1]
    ) * (q[:, 2, 0] - q[:, 0, 0])
    if signed.sum() < 0:
        uv[:, 0] = -uv[:, 0]
    uv -= uv.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(uv.T))
    major = evecs[:, np.argmax(evals)]
    angle = np.arctan2(major[0], major[1])  # turn the major axis onto +v
    c, s = np.cos(angle), np.sin(angle)
    return uv @ np.array([[c, -s], [s, c]]).T


@dataclasses.dataclass
class EarUVs:
    """Ear tile UVs of a level's vertices: (N_level, 2) in the tile's square (OBJ
    convention), NaN off the ears, and the level they were unwrapped on."""

    uv: np.ndarray
    subdiv: int

    def save(self, path, topology: str = "anny") -> None:
        from safetensors.numpy import save_file

        save_file(
            {"uv": self.uv.astype(np.float32)},
            str(path),
            {"format": FORMAT, "subdiv": str(self.subdiv), "topology": topology},
        )

    @classmethod
    def load(cls, path, topology: str = "anny") -> "EarUVs":
        from safetensors import safe_open

        with safe_open(str(path), "numpy") as f:
            meta = f.metadata() or {}
            if meta.get("format") != FORMAT:
                raise ValueError(f"{path} is not an ear UV file ({FORMAT}).")
            if meta.get("topology", topology) != topology:
                raise ValueError(
                    f"{path} belongs to topology {meta.get('topology')!r}, not {topology!r}."
                )
            return cls(f.get_tensor("uv").astype(np.float64), int(meta["subdiv"]))


def ear_triangles(model, topology, threshold: float = EAR_WEIGHT) -> np.ndarray:
    """The mask (T,) of a detail mesh's outer ear triangles (mean ear weight over the
    threshold); the weights need the model's ear local changes (``local_changes="all"``)."""
    from opensculptboy.fit.register import ear_weights

    weights = np.asarray(topology.apply(ear_weights(model)))
    return weights[np.asarray(topology.triangles)].mean(1) > threshold


def _largest_component(triangles: np.ndarray, n: int) -> np.ndarray:
    """The mask of the triangles of the largest part connected through shared vertices."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rows = np.repeat(np.arange(len(triangles)), 3)
    A = coo_matrix((np.ones(len(rows)), (rows, triangles.ravel())), (len(triangles), n))
    _, label = connected_components((A @ A.T).tocsr(), directed=False)
    return label == np.argmax(np.bincount(label))


def ear_uv_table(
    vertices: np.ndarray, topology, ears: np.ndarray, sides: np.ndarray
) -> EarUVs:
    """
    Unwrap the outer ears of a detail mesh (full level): ``vertices`` its fitted rest mesh,
    ``ears`` the mask of its ear triangles, ``sides`` the figure side (+1 left, -1 right) of
    each vertex. Returns the ear tile UVs of the level's vertices.
    """
    T = np.asarray(topology.triangles)
    n = len(vertices)
    table = np.full((n, 2), np.nan)
    islands = {}
    for side in (1, -1):
        pick = ears & (sides[T].mean(1) * side > 0)
        tris = T[pick]
        if len(tris) == 0:
            continue
        tris = tris[_largest_component(tris, n)]
        # close the set: every triangle of the island's vertices belongs to it, as
        # with_ear_uvs finds them on every level
        inside = np.zeros(n, bool)
        inside[tris] = True
        tris = T[inside[T].all(1)]
        tris = tris[_largest_component(tris, n)]
        used, local = np.unique(tris, return_inverse=True)
        islands[side] = (used, unwrap(vertices[used], local.reshape(-1, 3)))
    if not islands:
        return EarUVs(table, topology.subdiv)
    width, height = EAR_TILE_SIZE
    half = np.array([0.5, 1.0])  # each half of the tile, in tile UV units
    # one scale for both ears: metres to tile UV units, fitting the larger ear in its half
    extent = max(
        float(
            np.max(
                (uv.max(0) - uv.min(0)) / (half * np.array([width, height]) / height)
            )
        )
        for _, uv in islands.values()
    )
    scale_px = (1.0 - 2 * EAR_PADDING) * height / extent  # texels per metre
    for side, (used, uv) in islands.items():
        centre = np.array([0.25 if side > 0 else 0.75, 0.5])
        mid = 0.5 * (uv.max(0) + uv.min(0))
        texels = (uv - mid) * scale_px
        table[used] = centre + texels / np.array([width, height])
    return EarUVs(table, topology.subdiv)


def density(vertices, topology, table: EarUVs) -> dict:
    """Texels per millimetre on the ear tile at a level: area-weighted mean, 5th and 50th
    percentiles."""
    mesh, mask = with_ear_uvs(topology, table)
    if mask is None or not mask.any():
        return {}
    T = mesh.triangles[mask]
    UT = mesh.uv_triangles[mask]
    P = np.asarray(vertices)[T]
    a = (
        0.5
        * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
        * 1e6
    )
    q = mesh.uvs[UT] * np.array(EAR_TILE_SIZE, float)
    au = 0.5 * np.abs(
        (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
        - (q[:, 1, 1] - q[:, 0, 1]) * (q[:, 2, 0] - q[:, 0, 0])
    )
    d = np.sqrt(au / np.maximum(a, 1e-12))
    return {
        "mean": float(np.sqrt((au.sum()) / a.sum())),
        "p5": float(np.percentile(d, 5)),
        "p50": float(np.median(d)),
    }


def with_ear_uvs(topology, table: EarUVs | None):
    """
    A copy of a detail mesh whose outer ear triangles take the ear tile's UVs (appended after
    the atlas's), and the mask (T,) of those triangles; (topology, None) for level 0, a mesh
    without UVs or no table.
    """
    if table is None or topology.subdiv == 0 or topology.uv_triangles is None:
        return topology, None
    rows = np.asarray(topology.rows)
    if rows.max(initial=0) >= len(table.uv):
        return topology, None
    uv = table.uv[rows]
    valid = ~np.isnan(uv[:, 0])
    T = np.asarray(topology.triangles)
    mask = valid[T].all(1)
    if not mask.any():
        return topology, None
    ear_vertices = np.unique(T[mask])
    new_index = np.full(len(rows), -1)
    new_index[ear_vertices] = len(topology.uvs) + np.arange(len(ear_vertices))
    uv_triangles = np.asarray(topology.uv_triangles).copy()
    uv_triangles[mask] = new_index[T[mask]]
    uvs = np.concatenate([np.asarray(topology.uvs), uv[ear_vertices]])
    return dataclasses.replace(topology, uv_triangles=uv_triangles, uvs=uvs), mask


def card_ear_uvs(character, detail_dir=None) -> EarUVs | None:
    """The ear tile UVs that a card's skin material names (``material.ears.uvs``), or None."""
    ears = (character.material or {}).get("ears") or {}
    path = ears.get("uvs")
    if not path:
        return None
    path = pathlib.Path(path)
    if not path.is_absolute() and detail_dir is not None:
        path = pathlib.Path(detail_dir) / path
    return EarUVs.load(path, character.topology)

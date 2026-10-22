# OpenSculptBoy
# Apache License, Version 2.0
"""
The mouth cavity kept behind the skin of the face.

Anny's mouth cavity (the faces of the base-mesh label ``mouth_cavity``: the inner faces of the
lips, the gums and the palate) moves with its own rows of every blend shape, and some rows move
the outer skin further than the cavity. The face shapes ``nose-trans-backward-forward`` and
``mouth-trans-backward-forward`` at -1, for example, pull the skin above the upper lip back past
the front of the cavity, so that a renderer shows the cavity's red through the skin there, at
rest and more when the mouth opens.

:func:`clear_mouth` finds the vertices of the cavity that lie in front of the skin and pushes
them behind it, with a smooth correction that fades out over the cavity; the exporters run it on
the rest mesh and on the mesh of each morph target, so that each target at full weight keeps the
cavity behind the skin. A mesh without such vertices comes back unchanged, as do the tongue and
the lips' edge (the cavity's vertices on a face of the skin).

A vertex of the cavity lies in front of the skin when, seen from one of the :data:`VIEWS`, no skin
covers it and the skin lies right behind it, within :data:`REACH`; the cavity seen through the
open mouth has no skin right behind it, and the cavity behind the lips has the lips in front of
it, so both stay where they are. A vertex that the skin covers by less than :data:`GAP` is pushed
back too, since the cavity's triangles then cut through the skin's between their vertices,
except within :data:`LIP_EDGE` of the lips' edge, where the cavity meets the skin. The rays run
on Warp's BVH (``warp.mesh_query_ray``).
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np

CLEARANCE = 0.0005  # metres behind the skin for a pushed vertex
REACH = 0.008  # metres behind a vertex within which the skin must lie for a push
GAP = 0.0015  # metres of skin in front of the cavity below which the cavity is pushed back
LIP_EDGE = 0.003  # metres from the lips' edge within which GAP does not apply
FADE = 0.02  # metres from the pushed vertices beyond which the correction is zero
NEAR = 0.08  # metres from the cavity's middle within which the skin enters the test
ROUNDS = 4  # rounds of test and correction
# Topologies whose vertices are MakeHuman's base-mesh vertices (with their modifiers).
MAKEHUMAN_MESHES = ("anny", "makehuman", "sculpt", "head", "default")
# Directions toward a viewer in anny's rest frame (the face looks along -Y, Z is up): the front,
# 40 degrees to either side and 30 degrees above and below.
VIEWS = np.array(
    [
        [0.0, -1.0, 0.0],
        [np.sin(np.radians(40)), -np.cos(np.radians(40)), 0.0],
        [-np.sin(np.radians(40)), -np.cos(np.radians(40)), 0.0],
        [0.0, -np.cos(np.radians(30)), np.sin(np.radians(30))],
        [0.0, -np.cos(np.radians(30)), -np.sin(np.radians(30))],
    ]
)


@dataclasses.dataclass
class MouthInterior:
    """The mouth cavity in one mesh (anny's own, or a mesh of the detail layer)."""

    #: (V,) bool: the vertices of the cavity
    cavity: np.ndarray
    #: (V,) bool: the cavity's vertices on a face of the skin (the lips' edge)
    pinned: np.ndarray
    #: (S, 3) the triangles with a vertex off the cavity, the tongue's aside
    skin_triangles: np.ndarray
    #: (E, 2) the edges between vertices of the cavity
    edges: np.ndarray

    @property
    def probe(self) -> np.ndarray:
        """The cavity's vertices that the correction may move."""
        return np.flatnonzero(self.cavity & ~self.pinned)


def mouth_interior(model, mesh=None) -> MouthInterior | None:
    """
    The mouth cavity of ``model``'s mesh, or of ``mesh`` (a
    :class:`~opensculptboy.detail.subdivision.DetailTopology`); None for a mesh without one (a
    hand, or a mesh made by retopology). A vertex of a detail mesh belongs to the cavity when
    every level-0 vertex behind it does.
    """
    from anny.faces.authoring.base_mesh import base_mesh
    from opensculptboy.export.body import triangulated_faces
    from opensculptboy.topology.lookup import base_mesh_rows, model_topology

    if model_topology(model).split("-")[0] not in MAKEHUMAN_MESHES:
        return None  # a mesh made by retopology, without MakeHuman's labels
    source = base_mesh()
    rows = base_mesh_rows(model)
    if len(rows) == 0 or rows.min() < 0 or rows.max() >= len(source.V):
        return None
    cavity_base = np.zeros(len(source.V), bool)
    cavity_base[np.unique(source.quads[source.face_labels == "mouth_cavity"])] = True
    tongue_base = np.zeros(len(source.V), bool)
    tongue_rows = np.asarray(source.groups.get("helper-tongue", []), dtype=np.int64)
    tongue_base[tongue_rows] = True
    cavity, tongue = cavity_base[rows], tongue_base[rows]
    if not cavity.any():
        return None
    if mesh is not None and mesh.operators:
        both = np.stack([cavity, tongue], axis=1).astype(np.float64)
        fine = np.asarray(mesh.apply(both))
        cavity, tongue = fine[:, 0] > 1 - 1e-9, fine[:, 1] > 1 - 1e-9
    triangles = mesh.triangles if mesh is not None else triangulated_faces(model)[0]
    triangles = np.asarray(triangles, dtype=np.int64)
    cavity &= ~tongue
    inside = cavity[triangles]
    other = tongue[triangles].any(1)
    skin = ~inside.all(1) & ~other
    pinned = np.zeros(len(cavity), bool)
    pinned[np.unique(triangles[skin])] = True
    pinned &= cavity
    within = triangles[inside.all(1)]
    edges = np.concatenate([within[:, [0, 1]], within[:, [1, 2]], within[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return MouthInterior(
        cavity=cavity, pinned=pinned, skin_triangles=triangles[skin], edges=edges
    )


@functools.lru_cache(maxsize=1)
def _ray_kernel():
    import warp as wp

    wp.init()

    @wp.kernel
    def cast(
        mesh: wp.uint64,
        origins: wp.array(dtype=wp.vec3),
        directions: wp.array(dtype=wp.vec3),
        reach: float,
        out: wp.array(dtype=float),
    ):
        i = wp.tid()
        query = wp.mesh_query_ray(mesh, origins[i], directions[i], reach)
        if query.result:
            out[i] = query.t
        else:
            out[i] = -1.0

    return cast


def _cast(
    mesh, origins: np.ndarray, directions: np.ndarray, reach: float
) -> np.ndarray:
    """The distance (N,) along each ray to the first triangle of ``mesh``, -1 for none."""
    import warp as wp

    out = wp.zeros(len(origins), dtype=float, device="cpu")
    wp.launch(
        _ray_kernel(),
        dim=len(origins),
        inputs=[
            mesh.id,
            wp.array(origins.astype(np.float32), dtype=wp.vec3, device="cpu"),
            wp.array(directions.astype(np.float32), dtype=wp.vec3, device="cpu"),
            float(reach),
        ],
        outputs=[out],
        device="cpu",
    )
    return out.numpy()


def _skin_mesh(vertices: np.ndarray, interior: MouthInterior):
    """A Warp mesh of the skin triangles near the cavity, or None when there are none."""
    import warp as wp

    middle = vertices[interior.cavity].mean(0)
    tri = interior.skin_triangles
    near = np.linalg.norm(vertices[tri].mean(1) - middle, axis=1) < NEAR
    proper = (
        (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 2] != tri[:, 0])
    )
    tri = tri[near & proper]
    if len(tri) == 0:
        return None
    points = wp.array(vertices.astype(np.float32), dtype=wp.vec3, device="cpu")
    indices = wp.array(tri.reshape(-1).astype(np.int32), dtype=wp.int32, device="cpu")
    return wp.Mesh(points=points, indices=indices)


def _in_front(vertices: np.ndarray, interior: MouthInterior, probe: np.ndarray, deep):
    """For each probed vertex: whether it lies in front of the skin from one of the views (or,
    where ``deep``, less than :data:`GAP` behind it), and the point the correction moves it to:
    :data:`CLEARANCE` behind the skin along the nearest such view, or :data:`GAP` behind it."""
    mesh = _skin_mesh(vertices, interior)
    outside = np.zeros(len(probe), bool)
    target = vertices[probe].copy()
    if mesh is None or len(probe) == 0:
        return outside, target
    p = vertices[probe]
    eps = 1e-6
    best = np.full(len(probe), np.inf)
    for view in VIEWS:
        toward = np.repeat(view[None], len(p), axis=0)
        front = _cast(mesh, p + toward * eps, toward, 1.0)
        behind = _cast(mesh, p - toward * eps, -toward, REACH)
        hit = (front < 0) & (behind >= 0) & (behind < best)
        best[hit] = behind[hit]
        target[hit] = p[hit] - view * (behind[hit, None] + CLEARANCE)
        outside |= hit
        close = deep & ~outside & (front >= 0) & (front < GAP)
        target[close] = p[close] - view * (GAP - front[close, None])
        outside |= close
    return outside, target


def _fill(
    n, edges, pins: np.ndarray, values: np.ndarray, free: np.ndarray, iterations=400
):
    """A smooth field (n, 3): ``values`` on ``pins``, zero on the other fixed vertices, and the
    mean of the neighbours on ``free`` (Jacobi iterations)."""
    field = np.zeros((n, 3))
    field[pins] = values
    if not free.any() or len(edges) == 0:
        return field
    a, b = edges[:, 0], edges[:, 1]
    degree = np.zeros(n)
    np.add.at(degree, a, 1.0)
    np.add.at(degree, b, 1.0)
    rows = free & (degree > 0)
    for _ in range(iterations):
        total = np.zeros((n, 3))
        np.add.at(total, a, field[b])
        np.add.at(total, b, field[a])
        field[rows] = total[rows] / degree[rows, None]
    return field


def clear_vertices(
    vertices: np.ndarray, interior: MouthInterior
) -> tuple[np.ndarray, int]:
    """``vertices`` (V, 3) with the cavity behind the skin, and the number of the cavity's
    vertices that lay in front of it; the same array when none did."""
    vertices = np.asarray(vertices, dtype=np.float64)
    probe = interior.probe
    edge = vertices[interior.pinned]
    deep = np.ones(len(probe), bool)
    if len(edge):
        for start in range(0, len(probe), 2048):
            chunk = vertices[probe[start : start + 2048]]
            d = np.linalg.norm(chunk[:, None, :] - edge[None], axis=2).min(1)
            deep[start : start + 2048] = d > LIP_EDGE
    outside, target = _in_front(vertices, interior, probe, deep)
    if not outside.any():
        return vertices, 0
    count = int(outside.sum())
    n = len(vertices)
    pins = np.zeros(n, bool)
    push = np.zeros((n, 3))
    out = vertices
    for _ in range(ROUNDS):
        rows = probe[outside]
        push[rows] += target[outside] - out[rows]
        pins[rows] = True
        moved = vertices[pins]
        cand = probe[~pins[probe]]
        far = np.full(n, np.inf)
        for start in range(0, len(cand), 2048):
            chunk = cand[start : start + 2048]
            far[chunk] = np.min(
                np.linalg.norm(vertices[chunk][:, None, :] - moved[None], axis=2),
                axis=1,
            )
        free = np.zeros(n, bool)
        free[cand] = far[cand] < FADE
        field = _fill(n, interior.edges, pins, push[pins], free)
        out = vertices + field
        outside, target = _in_front(out, interior, probe, deep)
        outside &= ~pins[probe]
        if not outside.any():
            break
    return out, count


def clear_mouth(
    vertices: np.ndarray, offsets: np.ndarray, interior: MouthInterior | None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    The rest mesh (V, 3) and the morph-target offsets (R, V, 3) with the cavity behind the skin
    in the rest mesh and in the mesh of each target at full weight, and a report
    (``{"rest": count, "targets": [indices of corrected targets]}``). Without anything to
    correct, the arrays come back as they are.
    """
    report = {"rest": 0, "targets": []}
    if interior is None:
        return vertices, offsets, report
    base = np.asarray(vertices, dtype=np.float64)
    fixed, report["rest"] = clear_vertices(base, interior)
    offsets = np.asarray(offsets, dtype=np.float64)
    new = None
    for r in range(len(offsets)):
        state, count = clear_vertices(base + offsets[r], interior)
        if count == 0 and report["rest"] == 0:
            continue
        if new is None:
            new = offsets.copy()
        new[r] = state - fixed
        if count:
            report["targets"].append(r)
    if report["rest"] == 0 and new is None:
        return vertices, offsets, report
    return fixed, offsets if new is None else new, report


def cavity_in_front(vertices: np.ndarray, interior: MouthInterior | None) -> int:
    """The number of the cavity's vertices that lie in front of the skin in ``vertices`` (the
    first rule of :func:`clear_vertices`, without :data:`GAP`)."""
    if interior is None:
        return 0
    vertices = np.asarray(vertices, dtype=np.float64)
    probe = interior.probe
    outside, _ = _in_front(vertices, interior, probe, np.zeros(len(probe), bool))
    return int(outside.sum())

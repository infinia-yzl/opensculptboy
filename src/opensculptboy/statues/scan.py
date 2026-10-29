# OpenSculptBoy
# Apache License, Version 2.0
"""
A statue scan as a mesh that the reference fit can take: read, turned upright, scaled, its
plinth cut off and its marked pieces taken out.

A scan of a statue comes in any unit (millimetres for most museum files), with any axis up
and facing any way, and it carries what a character has no part for: the plinth, struts and
tree stumps fused to a leg, and breaks where an arm was lost. The functions here turn it into
Anny's frame (metres, Z up, the figure facing -Y, X toward the figure's left):

- :func:`load_scan` reads an STL, OBJ, PLY or GLB file as one triangle mesh with welded
  vertices.
- :func:`guess_up` takes the longest principal axis as the vertical, and the end with the
  wider footprint (the plinth and the feet) as the bottom.
- :func:`guess_front` looks for the nose: the sharpest point of the outline of the face in a
  horizontal band of the head.
- :func:`plinth_top` finds where the plinth ends: the first slab from the bottom whose
  footprint falls well under the plinth's.
- :func:`select_triangles` turns a mask (a box, a sphere, a half space or a list of vertices)
  into the triangles inside it.
"""

from __future__ import annotations

import dataclasses

import numpy as np

AXES = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}
# the share of the height at each end in which guess_up compares the footprints
END_BAND = 0.06
# the band of the nose in which guess_front looks for it: shares of the height below the
# highest point (the crown), and the reach (share of the height) of the head around the crown
FACE_BAND = (0.075, 0.105)
HEAD_REACH = 0.07
# the angle (degrees) on each side of a direction against which guess_front measures how far
# the outline stands out there
FRONT_SPREAD = 40.0
# a slab of plinth_top is the plinth while its footprint is over this share of the lowest
# slab's, and the plinth ends within this share of the height
PLINTH_SHARE = 0.45
PLINTH_REACH = 0.25


def axis_vector(name) -> np.ndarray:
    """A unit vector from an axis name (``"+z"``, ``"-y"``...) or from three numbers."""
    if isinstance(name, str):
        key = name.strip().lower()
        if key in ("x", "y", "z"):
            key = "+" + key
        if key not in AXES:
            raise ValueError(f"Unknown axis {name!r}; use one of {sorted(AXES)}.")
        return np.asarray(AXES[key])
    vector = np.asarray(name, dtype=np.float64).reshape(3)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length == 0:
        raise ValueError(f"The axis {name!r} has no direction.")
    return vector / length


def load_scan(path) -> tuple[np.ndarray, np.ndarray]:
    """The vertices (V, 3) and triangles (T, 3) of a scan file, all its meshes merged, with
    the vertices that an STL file repeats per triangle welded."""
    import trimesh

    loaded = trimesh.load(str(path), force="mesh", process=False)
    mesh = trimesh.Trimesh(
        np.asarray(loaded.vertices, dtype=np.float64),
        np.asarray(loaded.faces, dtype=np.int64),
        process=False,
    )
    mesh.merge_vertices(digits_vertex=6)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)


def _footprint(points: np.ndarray) -> float:
    """The area of the convex hull of 2D points (0 for fewer than three)."""
    from scipy.spatial import ConvexHull, QhullError

    if len(points) < 3:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except QhullError:
        return 0.0


def _plane_basis(up: np.ndarray) -> np.ndarray:
    """Two unit vectors (2, 3) across ``up``."""
    helper = (
        np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    a = np.cross(up, helper)
    a /= np.linalg.norm(a)
    return np.stack([a, np.cross(up, a)])


def guess_up(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """
    The up direction of a standing statue and a confidence from 0 to 1: the longest principal
    axis of its vertices, pointing away from the end with the wider footprint (the plinth and
    the feet stand wider than the head). The confidence is the share by which the footprints
    of both ends differ.
    """
    V = np.asarray(vertices, dtype=np.float64)
    centred = V - V.mean(0)
    _, _, vt = np.linalg.svd(centred[:: max(1, len(V) // 50000)], full_matrices=False)
    axis = vt[0]
    height = centred @ axis
    lo, hi = height.min(), height.max()
    band = END_BAND * (hi - lo)
    basis = _plane_basis(axis)
    low = _footprint(centred[height < lo + band] @ basis.T)
    high = _footprint(centred[height > hi - band] @ basis.T)
    up = axis if low >= high else -axis
    confidence = abs(low - high) / max(low, high, 1e-12)
    return up, float(confidence)


def guess_front(vertices: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, float]:
    """
    The direction the statue faces (across ``up``) and a score. The head lies under the
    highest point; in its band of the nose (:data:`FACE_BAND`, shares of the height below the
    highest point), the outline stands out most in the direction of the nose, beyond the
    outline :data:`FRONT_SPREAD` degrees to each side. The score is that lead over the
    head's radius (0.1 and more reads as a clear nose).
    """
    V = np.asarray(vertices, dtype=np.float64)
    up = axis_vector(up)
    height = V @ up
    lo, hi = height.min(), height.max()
    below = (hi - height) / max(hi - lo, 1e-12)
    basis = _plane_basis(up)
    crown = (V[np.argmax(height)] @ basis.T)[None]
    flat = V @ basis.T
    near = np.linalg.norm(flat - crown, axis=1) < HEAD_REACH * (hi - lo)
    band = flat[near & (below > FACE_BAND[0]) & (below < FACE_BAND[1])]
    if len(band) < 10:
        raise ValueError("The scan has too few vertices in the band of the face.")
    head = band - 0.5 * (band.max(0) + band.min(0))
    angles = np.radians(np.arange(0.0, 360.0, 2.0))
    directions = np.stack([np.cos(angles), np.sin(angles)], 1)
    reach = (head @ directions.T).max(0)
    step = int(round(FRONT_SPREAD / 2.0))
    beside = 0.5 * (np.roll(reach, step) + np.roll(reach, -step))
    lead = reach - beside
    best = int(lead.argmax())
    front = directions[best] @ basis
    radius = float(np.median(reach))
    return front / np.linalg.norm(front), float(lead[best] / max(radius, 1e-12))


def frame_rotation(up, front) -> np.ndarray:
    """The rotation (3, 3) from the scan's axes to Anny's frame: ``up`` to +Z and ``front``
    (made square to ``up``) to -Y, so that +X points to the figure's left."""
    z = axis_vector(up)
    f = axis_vector(front)
    f = f - (f @ z) * z
    if np.linalg.norm(f) < 1e-6:
        raise ValueError("The front direction runs along the up direction.")
    y = -f / np.linalg.norm(f)
    x = np.cross(y, z)
    return np.stack([x, y, z])


def surface_points(
    vertices: np.ndarray, triangles: np.ndarray, count: int = 200000, seed: int = 0
) -> np.ndarray:
    """``count`` points (N, 3) spread over a mesh by area, and the vertices of its triangles: a plinth's flat
    sides may be a few large triangles with no vertex between their corners."""
    V = np.asarray(vertices, dtype=np.float64)
    T = np.asarray(triangles)
    p = V[T]
    V = V[np.unique(T)]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    if area.sum() <= 0:
        return V
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(p), count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    points = p[pick, 0] + u[:, None] * (p[pick, 1] - p[pick, 0])
    points += v[:, None] * (p[pick, 2] - p[pick, 0])
    return np.concatenate([V, points])


def plinth_top(
    vertices: np.ndarray, triangles: np.ndarray | None = None, slabs: int = 100
) -> float | None:
    """
    The height (Anny's frame, Z up) where the plinth ends, or None when the bottom shows no
    plinth: walking up in ``slabs`` slabs, the first slab whose footprint falls under
    :data:`PLINTH_SHARE` of the lowest slab's, within :data:`PLINTH_REACH` of the height.
    With ``triangles``, the footprints take points spread over the surface as well.
    """
    V = np.asarray(vertices, dtype=np.float64)
    if triangles is not None:
        V = surface_points(V, triangles)
    z = V[:, 2]
    lo, hi = z.min(), z.max()
    edges = np.linspace(lo, hi, slabs + 1)
    which = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, slabs - 1)
    reach = int(np.ceil(PLINTH_REACH * slabs))
    areas = np.array([_footprint(V[which == k, :2]) for k in range(reach + 1)])
    base = areas[0]
    # a plinth stands wider than the body above it (the legs at a third of the height)
    middle = _footprint(V[(z > lo + 0.3 * (hi - lo)) & (z < lo + 0.4 * (hi - lo)), :2])
    if base <= 1.5 * middle:
        return None
    for k in range(1, reach + 1):
        if areas[k] < PLINTH_SHARE * base:
            return _plinth_surface(V, edges, k, hi - lo)
    return None


def _plinth_surface(V: np.ndarray, edges: np.ndarray, k: int, height: float) -> float:
    """The height of the plinth's top within the slabs around slab ``k``, where its
    footprint fell: the highest of the points below slab ``k + 1`` that lie away from the
    feet (the points of slabs ``k + 1`` and ``k + 2`` seen from above)."""
    from scipy.spatial import cKDTree

    z = V[:, 2]
    feet = V[(z >= edges[k + 1]) & (z < edges[min(k + 3, len(edges) - 1)]), :2]
    low = V[z < edges[k + 1]]
    if len(feet) == 0 or len(low) == 0:
        return float(edges[k])
    distance, _ = cKDTree(feet).query(low[:, :2])
    plinth = low[distance > 0.03 * height]
    if len(plinth) == 0:
        return float(edges[k])
    return float(np.percentile(plinth[:, 2], 99.5))


@dataclasses.dataclass
class Mask:
    """
    A region of the scan in Anny's frame (metres) and what it holds. ``kind`` is ``"box"``
    (``centre``, ``size``), ``"sphere"`` (``centre``, ``radius``), ``"half"`` (``point``,
    ``normal``: the side the normal points to) or ``"vertices"`` (``indices`` into the scan's
    welded vertices). ``role`` is ``"support"`` (a strut, a stump or the plinth: taken out),
    ``"break"`` (the face of a break where a part was lost: taken out, so that anny keeps its
    own shape there), ``"drapery"`` (kept as clothing, out of the fit) or ``"hair"`` (kept as a
    sculpted mass of its own, out of the fit).
    """

    kind: str
    role: str = "support"
    centre: tuple | None = None
    size: tuple | None = None
    radius: float | None = None
    point: tuple | None = None
    normal: tuple | None = None
    indices: tuple | None = None
    name: str | None = None

    ROLES = ("support", "break", "drapery", "hair")

    @classmethod
    def from_dict(cls, entry: dict) -> "Mask":
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(entry) - fields
        if unknown:
            raise ValueError(f"Unknown mask fields {sorted(unknown)}.")
        mask = cls(**entry)
        if mask.role not in cls.ROLES:
            raise ValueError(
                f"Unknown mask role {mask.role!r}; use one of {cls.ROLES}."
            )
        return mask

    def inside(self, points: np.ndarray) -> np.ndarray:
        """Whether each point (N, 3) lies in the region."""
        P = np.asarray(points, dtype=np.float64)
        if self.kind == "box":
            half = 0.5 * np.asarray(self.size, dtype=np.float64)
            return (np.abs(P - np.asarray(self.centre)) <= half).all(1)
        if self.kind == "sphere":
            return np.linalg.norm(P - np.asarray(self.centre), axis=1) <= self.radius
        if self.kind == "half":
            normal = axis_vector(self.normal)
            return (P - np.asarray(self.point)) @ normal >= 0.0
        raise ValueError(f"Unknown mask kind {self.kind!r}.")


def select_triangles(
    vertices: np.ndarray, triangles: np.ndarray, mask: Mask
) -> np.ndarray:
    """The triangles (T,) bool that ``mask`` holds: their centroids inside a shape, or all
    their corners among the listed vertices."""
    T = np.asarray(triangles)
    if mask.kind == "vertices":
        chosen = np.zeros(len(vertices), bool)
        chosen[np.asarray(mask.indices, dtype=np.int64)] = True
        return chosen[T].all(1)
    return mask.inside(np.asarray(vertices)[T].mean(1))


def components(triangles: np.ndarray, count: int) -> np.ndarray:
    """The connected component of each triangle (T,), numbered by size, largest first."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    T = np.asarray(triangles)
    rows = np.repeat(np.arange(len(T)), 3)
    # triangles that share a vertex share a component: join them through the vertices
    adjacency = coo_matrix(
        (
            np.ones(2 * len(rows)),
            (
                np.concatenate([rows, len(T) + T.ravel()]),
                np.concatenate([len(T) + T.ravel(), rows]),
            ),
        ),
        shape=(len(T) + count, len(T) + count),
    )
    _, labels = connected_components(adjacency, directed=False)
    labels = labels[: len(T)]
    _, inverse, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    order = np.argsort(-sizes)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[inverse.ravel()]

# OpenSculptBoy
# Apache License, Version 2.0
"""
The patches of the ``sculpt`` topology and the quad mesh they are written against.

A patch module of :mod:`opensculptboy.topology.patches` returns a :class:`Patch` from
``build(base, **options)``, where ``base`` is the :class:`BaseQuadMesh` of
:func:`base_quad_mesh` (anny's quad mesh, ``topology="anny-quads"``). See
``src/opensculptboy/topology/README.md`` for the contract.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
from collections import Counter
from typing import Sequence

import numpy as np

# Anchors per new vertex.
MAX_ANCHORS = 8
# Tolerance on the sum of the weights of a row.
WEIGHT_TOLERANCE = 1e-9


class PatchError(ValueError):
    """A patch or a composed set of patches that breaks the rules of the format."""


@dataclasses.dataclass(frozen=True)
class BaseQuadMesh:
    """
    Anny's quad mesh (``topology="anny-quads"``) at the default body, the base of every patch.

    ``vertices`` (N, 3) float64 in anny's frame (metres, Z up, the face toward -Y, X toward the
    figure's left); ``quads`` (F, 4) vertex indices, counter-clockwise seen from outside;
    ``uv_quads`` (F, 4) indices into ``uvs`` (U, 2), OBJ convention (v = 0 at the bottom);
    ``base_mesh_vertex_indices`` (N,) the MakeHuman ``base.obj`` index of each vertex.
    Vertex i of this mesh is vertex i of every anny model of the ``anny`` and ``anny-quads``
    topologies, and of ``sculpt``.
    """

    vertices: np.ndarray
    quads: np.ndarray
    uv_quads: np.ndarray
    uvs: np.ndarray
    base_mesh_vertex_indices: np.ndarray

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def uv_count(self) -> int:
        return len(self.uvs)

    @functools.cached_property
    def digest(self) -> str:
        """SHA-256 of the faces, the UV faces, and the vertices and UVs rounded to 10 µm and
        1e-6, so that every machine computes the same value."""
        h = hashlib.sha256()
        for array in (
            self.quads.astype(np.int64),
            self.uv_quads.astype(np.int64),
            np.round(self.vertices * 1e5).astype(np.int64),
            np.round(self.uvs * 1e6).astype(np.int64),
        ):
            h.update(np.ascontiguousarray(array).tobytes())
        return h.hexdigest()

    @functools.cached_property
    def mirror_vertices(self) -> np.ndarray:
        """(N,) the vertex at the mirror image (x -> -x) of each vertex (itself on X = 0)."""
        key = np.round(self.vertices * 1e7).astype(np.int64)
        index = {tuple(k): i for i, k in enumerate(key.tolist())}
        mirrored = key * np.array([-1, 1, 1])
        try:
            return np.array([index[tuple(k)] for k in mirrored.tolist()], np.int64)
        except KeyError:
            raise PatchError("anny's quad mesh is not mirror symmetric.") from None

    @functools.cached_property
    def _face_index(self) -> dict:
        return {frozenset(q.tolist()): f for f, q in enumerate(self.quads)}

    def face_of(self, vertices: Sequence[int]) -> int:
        """The index of the quad with these four vertices (in any order); KeyError if none."""
        return self._face_index[frozenset(int(v) for v in vertices)]

    @functools.cached_property
    def mirror_faces(self) -> np.ndarray:
        """(F,) the quad at the mirror image of each quad."""
        m = self.mirror_vertices
        return np.array([self.face_of(m[q]) for q in self.quads], dtype=np.int64)

    @functools.cached_property
    def mirror_uvs(self) -> np.ndarray:
        """(U,) the UV at the mirror image of each UV: the UV of the mirrored vertex in the
        mirrored face."""
        m = self.mirror_vertices
        out = -np.ones(self.uv_count, dtype=np.int64)
        for f, g in enumerate(self.mirror_faces):
            for k in range(4):
                v = m[self.quads[f, k]]
                k2 = int(np.nonzero(self.quads[g] == v)[0][0])
                out[self.uv_quads[f, k]] = self.uv_quads[g, k2]
        return out

    @functools.cached_property
    def vertex_faces(self) -> list[np.ndarray]:
        """The quads around each vertex."""
        order = np.argsort(self.quads.ravel(), kind="stable")
        counts = np.bincount(self.quads.ravel(), minlength=self.vertex_count)
        faces = order // 4
        return np.split(faces, np.cumsum(counts)[:-1])

    def face_uvs(self, faces: Sequence[int]) -> dict[int, int]:
        """
        The UV index of each vertex of the given quads. Raise :class:`PatchError` when a vertex
        takes two UVs among them, that is when the quads cross a seam of the atlas (a patch
        replaces faces of one UV chart).
        """
        result: dict[int, int] = {}
        for f in faces:
            for v, u in zip(self.quads[f].tolist(), self.uv_quads[f].tolist()):
                if result.setdefault(v, u) != u:
                    raise PatchError(
                        f"Vertex {v} has UVs {result[v]} and {u} among faces {list(faces)}: "
                        "the faces cross a UV seam."
                    )
        return result

    def face_labels(self) -> np.ndarray:
        """(F,) the label of the MakeHuman body segmentation of each quad (``anny.faces.
        authoring.base_mesh``), for patches that select their faces by region."""
        from anny.faces.authoring.base_mesh import base_mesh

        mesh = base_mesh()
        index = {frozenset(q.tolist()): i for i, q in enumerate(mesh.quads)}
        base = self.base_mesh_vertex_indices
        return np.array(
            [mesh.face_labels[index[frozenset(base[q].tolist())]] for q in self.quads]
        )


@functools.lru_cache(maxsize=1)
def base_quad_mesh() -> BaseQuadMesh:
    """The :class:`BaseQuadMesh` (anny's quad mesh at the default body)."""
    from anny.models import build_model_data
    from anny.models.model_data import RigConfig, TopologyConfig

    data = build_model_data(
        RigConfig.from_string("anny"),
        TopologyConfig.from_string("anny-quads"),
        "none",
        "none",
    )
    return base_from_model_data(data)


def base_from_model_data(data) -> BaseQuadMesh:
    """The :class:`BaseQuadMesh` of anny's quad :class:`~anny.models.model_data.ModelData`."""
    faces = data.faces.cpu().numpy().astype(np.int64)
    if faces.ndim != 2 or faces.shape[1] != 4:
        raise PatchError("The base of the sculpt topology is anny's quad mesh.")
    return BaseQuadMesh(
        vertices=data.template_vertices.detach().cpu().double().numpy(),
        quads=faces,
        uv_quads=data.face_texture_coordinate_indices.cpu().numpy().astype(np.int64),
        uvs=data.texture_coordinates.detach().cpu().double().numpy(),
        base_mesh_vertex_indices=data.base_mesh_vertex_indices.cpu().numpy(),
    )


def blend_rows(rows: Sequence[dict[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    """
    (M, 8) anchors and weights in canonical form from rows ``{anchor: weight}``: the weights
    of one anchor summed, zero weights dropped, anchors in increasing order, unused slots
    anchor 0 and weight 0. Raise :class:`PatchError` for a row with more than 8 anchors.
    """
    anchors = np.zeros((len(rows), MAX_ANCHORS), dtype=np.int64)
    weights = np.zeros((len(rows), MAX_ANCHORS), dtype=np.float64)
    for i, row in enumerate(rows):
        summed: dict[int, float] = {}
        for a, w in row.items():
            summed[int(a)] = summed.get(int(a), 0.0) + float(w)
        items = sorted((a, w) for a, w in summed.items() if w != 0.0)
        if len(items) > MAX_ANCHORS:
            raise PatchError(
                f"Row {i} has {len(items)} anchors; at most {MAX_ANCHORS}."
            )
        for k, (a, w) in enumerate(items):
            anchors[i, k], weights[i, k] = a, w
    return anchors, weights


def _empty_rows():
    return (
        np.zeros((0, MAX_ANCHORS), dtype=np.int64),
        np.zeros((0, MAX_ANCHORS), dtype=np.float64),
    )


@dataclasses.dataclass
class Patch:
    """
    One patch of the ``sculpt`` topology, in its own local numbering.

    - ``name``: the patch's label (``"eye_corner"``, ``"ear_root"``, ``"placeholder"``).
    - ``removed_faces`` (R,): indices into ``base.quads`` of the faces the patch replaces.
    - ``anchors`` (M, 8) int64 and ``weights`` (M, 8) float64: new vertex m is
      ``sum_k weights[m, k] * base.vertices[anchors[m, k]]``; weights non-negative, rows
      summing to 1, unused slots anchor 0 and weight 0 (:func:`blend_rows` builds them).
    - ``faces`` (F, 4) int64: the new faces, counter-clockwise seen from outside, over
      ``i < N`` (anny's vertex i) and ``N + m`` (this patch's new vertex m), N =
      ``base.vertex_count``; a triangle repeats its last index (``[a, b, c, c]``).
    - ``uv_faces`` (F, 4) int64: each corner's UV, ``j < U`` (anny's UV j) and ``U + n``
      (this patch's new UV n), U = ``base.uv_count``, with the same repeated last index.
    - ``uv_anchors`` (J, 8) and ``uv_weights`` (J, 8): new UV n is
      ``sum_k uv_weights[n, k] * base.uvs[uv_anchors[n, k]]``, by the same rules.
    - ``vertex_labels`` (M,) and ``face_labels`` (F,): optional finer labels (default
      ``name``), such as ``"eye_corner/caruncle"``.
    - ``symmetric``: the validator checks the mirror symmetry across X = 0.
    - ``options``: the options the module built the patch with (recorded in the file).
    """

    name: str
    removed_faces: np.ndarray
    anchors: np.ndarray
    weights: np.ndarray
    faces: np.ndarray
    uv_faces: np.ndarray
    uv_anchors: np.ndarray
    uv_weights: np.ndarray
    vertex_labels: list[str] | None = None
    face_labels: list[str] | None = None
    symmetric: bool = True
    options: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.removed_faces = np.asarray(self.removed_faces, dtype=np.int64).reshape(-1)
        self.anchors = np.asarray(self.anchors, dtype=np.int64).reshape(-1, MAX_ANCHORS)
        self.weights = np.asarray(self.weights, dtype=np.float64).reshape(
            -1, MAX_ANCHORS
        )
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 4)
        self.uv_faces = np.asarray(self.uv_faces, dtype=np.int64).reshape(-1, 4)
        self.uv_anchors = np.asarray(self.uv_anchors, dtype=np.int64).reshape(
            -1, MAX_ANCHORS
        )
        self.uv_weights = np.asarray(self.uv_weights, dtype=np.float64).reshape(
            -1, MAX_ANCHORS
        )

    @property
    def vertex_count(self) -> int:
        return len(self.anchors)

    def labels_of_vertices(self) -> list[str]:
        return list(self.vertex_labels or [self.name] * self.vertex_count)

    def labels_of_faces(self) -> list[str]:
        return list(self.face_labels or [self.name] * len(self.faces))

    def positions(self, base: BaseQuadMesh, vertices: np.ndarray | None = None):
        """(M, 3) positions of the new vertices on ``vertices`` (default ``base.vertices``)."""
        v = base.vertices if vertices is None else np.asarray(vertices)
        return np.einsum("mk,mkd->md", self.weights, v[self.anchors])

    def new_uvs(self, base: BaseQuadMesh) -> np.ndarray:
        """(J, 2) the new UVs."""
        return np.einsum("jk,jkd->jd", self.uv_weights, base.uvs[self.uv_anchors])

    def mirrored(self, base: BaseQuadMesh) -> "Patch":
        """The mirror image of the patch across X = 0 (faces reversed to stay outward)."""
        n, u = base.vertex_count, base.uv_count
        mv, mu = base.mirror_vertices, base.mirror_uvs

        def map_faces(faces, mirror, count):
            out = np.where(faces < count, mirror[np.minimum(faces, count - 1)], faces)
            tri = out[:, 2] == out[:, 3]
            # reverse the winding: [a, b, c, d] -> [a, d, c, b]; [a, b, c, c] -> [a, c, b, b]
            rev = out[:, [0, 3, 2, 1]]
            rev[tri] = out[tri][:, [0, 2, 1, 1]]
            return rev

        def map_rows(anchors, weights, mirror):
            rows = [
                {int(mirror[a]): w for a, w in zip(ra, rw) if w != 0.0}
                for ra, rw in zip(anchors, weights)
            ]
            return blend_rows(rows) if rows else _empty_rows()

        anchors, weights = map_rows(self.anchors, self.weights, mv)
        uv_anchors, uv_weights = map_rows(self.uv_anchors, self.uv_weights, mu)
        return Patch(
            name=self.name,
            removed_faces=base.mirror_faces[self.removed_faces],
            anchors=anchors,
            weights=weights,
            faces=map_faces(self.faces, mv, n),
            uv_faces=map_faces(self.uv_faces, mu, u),
            uv_anchors=uv_anchors,
            uv_weights=uv_weights,
            vertex_labels=self.vertex_labels,
            face_labels=self.face_labels,
            symmetric=self.symmetric,
            options=dict(self.options),
        )

    def union(self, other: "Patch", base: BaseQuadMesh) -> "Patch":
        """This patch and ``other`` (same name) as one patch; ``other``'s new vertices and UVs
        come after this patch's."""
        n, u = base.vertex_count, base.uv_count
        m, j = self.vertex_count, len(self.uv_anchors)
        faces = np.where(other.faces >= n, other.faces + m, other.faces)
        uv_faces = np.where(other.uv_faces >= u, other.uv_faces + j, other.uv_faces)
        return Patch(
            name=self.name,
            removed_faces=np.concatenate([self.removed_faces, other.removed_faces]),
            anchors=np.concatenate([self.anchors, other.anchors]),
            weights=np.concatenate([self.weights, other.weights]),
            faces=np.concatenate([self.faces, faces]),
            uv_faces=np.concatenate([self.uv_faces, uv_faces]),
            uv_anchors=np.concatenate([self.uv_anchors, other.uv_anchors]),
            uv_weights=np.concatenate([self.uv_weights, other.uv_weights]),
            vertex_labels=(
                None
                if self.vertex_labels is None and other.vertex_labels is None
                else self.labels_of_vertices() + other.labels_of_vertices()
            ),
            face_labels=(
                None
                if self.face_labels is None and other.face_labels is None
                else self.labels_of_faces() + other.labels_of_faces()
            ),
            symmetric=self.symmetric and other.symmetric,
            options=dict(self.options),
        )

    def with_mirror(self, base: BaseQuadMesh) -> "Patch":
        """This patch (one side) and its mirror image as one symmetric patch."""
        return self.union(self.mirrored(base), base)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _corners(faces: np.ndarray) -> list[list[int]]:
    """The corners of each face, a triangle's repeated last index dropped."""
    return [f[:3] if f[2] == f[3] else f for f in faces.tolist()]


def _directed_edges(polygons) -> Counter:
    edges = Counter()
    for p in polygons:
        for i in range(len(p)):
            edges[(p[i], p[(i + 1) % len(p)])] += 1
    return edges


def _boundary(polygons) -> Counter:
    """The directed edges of a set of faces that no other face of the set takes back."""
    edges = _directed_edges(polygons)
    return Counter(
        {e: c for e, c in edges.items() if edges.get((e[1], e[0]), 0) == 0 and c}
    )


def _check_rows(anchors, weights, count, what):
    if anchors.shape != weights.shape or anchors.shape[1:] != (MAX_ANCHORS,):
        raise PatchError(
            f"{what}: anchors and weights must both be (M, {MAX_ANCHORS})."
        )
    if len(anchors) == 0:
        return
    if anchors.min() < 0 or anchors.max() >= count:
        raise PatchError(f"{what}: anchors out of range [0, {count}).")
    if not np.isfinite(weights).all() or weights.min() < 0:
        raise PatchError(f"{what}: weights must be finite and non-negative.")
    error = np.abs(weights.sum(1) - 1.0).max()
    if error > WEIGHT_TOLERANCE:
        raise PatchError(f"{what}: weight rows must sum to 1 (off by {error:.3g}).")
    if (anchors[weights == 0] != 0).any():
        raise PatchError(f"{what}: unused slots must be anchor 0 with weight 0.")
    for i, (ra, rw) in enumerate(zip(anchors, weights)):
        used = ra[rw > 0]
        if len(set(used.tolist())) != len(used):
            raise PatchError(f"{what}: row {i} names an anchor twice.")


def _check_faces(faces, count, what):
    if faces.ndim != 2 or faces.shape[1] != 4:
        raise PatchError(f"{what}: faces must be (F, 4).")
    if len(faces) and (faces.min() < 0 or faces.max() >= count):
        raise PatchError(f"{what}: face indices out of range [0, {count}).")
    for i, f in enumerate(faces.tolist()):
        corners = f[:3] if f[2] == f[3] else f
        if len(set(corners)) != len(corners):
            raise PatchError(
                f"{what}: face {i} {f} repeats a vertex (a triangle repeats only its last)."
            )


def _signed_uv_areas(polygons, uvs):
    areas = []
    for p in polygons:
        q = uvs[p]
        x, y = q[:, 0], q[:, 1]
        areas.append(0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))
    return np.array(areas)


def _uv_triangle_areas(polygons, uvs):
    """Signed areas of the fan triangles of each face (any flipped triangle shows)."""
    out = []
    for p in polygons:
        q = uvs[p]
        for i in range(1, len(p) - 1):
            a, b, c = q[0], q[i], q[i + 1]
            out.append(
                0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            )
    return np.array(out)


def validate_patch(patch: Patch, base: BaseQuadMesh) -> None:
    """
    Raise :class:`PatchError` unless ``patch`` follows the rules of the format:

    - removed faces unique and in range; anchors in range, weights convex (at most 8 anchors,
      non-negative, summing to 1, canonical unused slots), and the same for the UV rows;
    - faces over valid indices, quads of four distinct vertices or triangles ``[a, b, c, c]``,
      every new vertex and new UV used, UV faces shaped as the faces;
    - no new face repeating a kept face or another new face;
    - the new faces close the hole exactly: their boundary is the boundary of the removed
      faces, edge for edge and in the same direction, in vertices and in UVs;
    - the new UV faces keep the orientation of the removed faces (no flipped triangle) and
      cover the same area of the atlas;
    - with ``symmetric``, the patch is its own mirror image across X = 0.
    """
    n, u = base.vertex_count, base.uv_count
    m, j = patch.vertex_count, len(patch.uv_anchors)
    what = f"patch {patch.name!r}"
    removed = patch.removed_faces
    if len(np.unique(removed)) != len(removed):
        raise PatchError(f"{what}: a removed face is named twice.")
    if len(removed) and (removed.min() < 0 or removed.max() >= len(base.quads)):
        raise PatchError(f"{what}: removed faces out of range.")
    _check_rows(patch.anchors, patch.weights, n, f"{what} vertices")
    _check_rows(patch.uv_anchors, patch.uv_weights, u, f"{what} UVs")
    _check_faces(patch.faces, n + m, f"{what} faces")
    _check_faces(patch.uv_faces, u + j, f"{what} UV faces")
    if patch.uv_faces.shape != patch.faces.shape:
        raise PatchError(f"{what}: uv_faces must match faces.")
    if not np.array_equal(
        patch.faces[:, 2] == patch.faces[:, 3],
        patch.uv_faces[:, 2] == patch.uv_faces[:, 3],
    ):
        raise PatchError(
            f"{what}: a triangle must be a triangle in faces and uv_faces."
        )
    if m and not np.isin(np.arange(n, n + m), patch.faces).all():
        raise PatchError(f"{what}: every new vertex must be used by a new face.")
    if j and not np.isin(np.arange(u, u + j), patch.uv_faces).all():
        raise PatchError(f"{what}: every new UV must be used by a new face.")
    for labels, count, kind in (
        (patch.vertex_labels, m, "vertex"),
        (patch.face_labels, len(patch.faces), "face"),
    ):
        if labels is not None and len(labels) != count:
            raise PatchError(f"{what}: {kind}_labels needs {count} entries.")

    new = _corners(patch.faces)
    kept_sets = {
        frozenset(q.tolist())
        for f, q in enumerate(base.quads)
        if f not in set(removed.tolist())
    }
    seen = set()
    for i, p in enumerate(new):
        key = frozenset(p)
        if key in kept_sets or key in seen:
            raise PatchError(f"{what}: face {i} {p} repeats an existing face.")
        seen.add(key)

    removed_polys = [base.quads[f].tolist() for f in removed]
    if _boundary(new) != _boundary(removed_polys):
        raise PatchError(
            f"{what}: the new faces must close the hole of the removed faces exactly (same "
            "boundary edges, same direction)."
        )
    if max(_directed_edges(new).values(), default=1) > 1:
        raise PatchError(
            f"{what}: a directed edge is used twice (flipped or doubled face)."
        )

    uvs = np.concatenate([base.uvs, patch.new_uvs(base)])
    new_uv = _corners(patch.uv_faces)
    removed_uv = [base.uv_quads[f].tolist() for f in removed]
    if _boundary(new_uv) != _boundary(removed_uv):
        raise PatchError(
            f"{what}: the new UV faces must cover the removed faces' region of the atlas "
            "(same UV boundary)."
        )
    if len(removed_uv):
        sign = np.sign(_signed_uv_areas(removed_uv, uvs))
        if not (sign == sign[0]).all() or sign[0] == 0:
            raise PatchError(f"{what}: the removed faces mix UV orientations.")
        tri = _uv_triangle_areas(new_uv, uvs) * sign[0]
        if len(tri) and tri.min() <= 0:
            raise PatchError(f"{what}: a new UV triangle is flipped or degenerate.")
        a_new = np.abs(_signed_uv_areas(new_uv, uvs)).sum()
        a_old = np.abs(_signed_uv_areas(removed_uv, uvs)).sum()
        if abs(a_new - a_old) > 1e-9 + 1e-6 * a_old:
            raise PatchError(
                f"{what}: the new UV faces cover {a_new:.6g} of the atlas, the removed "
                f"faces {a_old:.6g}."
            )
    if patch.symmetric:
        check_symmetry(patch, base)


def check_symmetry(patch: Patch, base: BaseQuadMesh, tolerance: float = 1e-9) -> None:
    """Raise :class:`PatchError` unless the patch is its own mirror image across X = 0."""
    what = f"patch {patch.name!r}"
    mirror = patch.mirrored(base)
    if set(mirror.removed_faces.tolist()) != set(patch.removed_faces.tolist()):
        raise PatchError(f"{what}: the removed faces are not mirror symmetric.")
    rows = {
        _row_key(a, w): i for i, (a, w) in enumerate(zip(patch.anchors, patch.weights))
    }
    if len(rows) != patch.vertex_count:
        raise PatchError(f"{what}: two new vertices have the same anchors and weights.")
    n = base.vertex_count
    to_self = np.empty(patch.vertex_count, dtype=np.int64)
    for i, (a, w) in enumerate(zip(mirror.anchors, mirror.weights)):
        key = _row_key(a, w)
        if key not in rows:
            raise PatchError(f"{what}: new vertex {i} has no mirror image.")
        to_self[i] = rows[key]
    # Faces of the mirror image, in this patch's numbering, as rotations of their corners.
    mapped = np.where(
        mirror.faces >= n, n + to_self[np.maximum(mirror.faces - n, 0)], mirror.faces
    )
    if {_cycle(p) for p in _corners(mapped)} != {
        _cycle(p) for p in _corners(patch.faces)
    }:
        raise PatchError(f"{what}: the new faces are not mirror symmetric.")
    del tolerance


def _row_key(anchors, weights):
    return tuple(
        (int(a), round(float(w), 12)) for a, w in sorted(zip(anchors, weights)) if w > 0
    )


def _cycle(p):
    k = p.index(min(p))
    return tuple(p[k:] + p[:k])

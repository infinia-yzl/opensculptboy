# OpenSculptBoy
# Apache License, Version 2.0
"""
The teeth on the mesh that an exporter writes.

:class:`TeethData` holds a set of teeth in a card's rest frame (Anny's frame, metres): points,
normals, UVs and triangles, the arch of each point (0 for the upper teeth, 1 for the lower)
and the part of each triangle (0 for the teeth, 1 for the gums). :func:`toothed_mesh` appends
them to the mesh of a file (anny's own mesh, a level of the detail layer, or either with the
eyes of :mod:`opensculptboy.eyes`) as a :class:`ToothedMesh`, which carries the file's rest
mesh, bind mesh, morph targets, normals and skin weights over to the teeth:

- positions (:meth:`ToothedMesh.vertices`): every tooth follows the rigid motion (Kabsch) of
  the head, the vertices that the ``head`` bone skins alone, so that the teeth keep their
  place in a VRM file's T-pose;
- morph targets (:meth:`ToothedMesh.targets`): the upper teeth take the rigid motion of the
  head's vertices that no target moves (none for the facial actions); the lower teeth take the
  rigid motion of the tongue in a target that moves it rigidly (anny's jaw actions turn the
  tongue with the jaw) and the head's otherwise (``tongueOut`` bends the tongue);
- skin weights: the mean weights of the head's vertices (the ``head`` bone).
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

FORMAT = "opensculptboy/teeth@1"
PARTS = ("teeth", "gums", "palate", "tongue", "floor", "throat")
# the tongue moves rigidly in a target when the residual of its rigid fit stays under this
# share of its mean motion (anny's jawOpen: 4 %; tongueOut: 23 %)
RIGID_SHARE = 0.1
# the head's vertices: the share of their skin weight on the head bone
HEAD_WEIGHT = 0.999


@dataclasses.dataclass
class TeethData:
    """A set of teeth in a card's rest frame (see the module documentation)."""

    points: np.ndarray  # (M, 3)
    normals: np.ndarray  # (M, 3)
    uvs: np.ndarray  # (M, 2), OBJ convention (v = 0 at the bottom)
    triangles: np.ndarray  # (T, 3)
    arch: np.ndarray  # (M,) uint8: 0 upper, 1 lower
    part: np.ndarray  # (T,) uint8: 0 teeth, 1 gums

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float64)
        self.normals = np.asarray(self.normals, dtype=np.float64)
        self.uvs = np.asarray(self.uvs, dtype=np.float64)
        self.triangles = np.asarray(self.triangles, dtype=np.int64)
        self.arch = np.asarray(self.arch, dtype=np.uint8)
        self.part = np.asarray(self.part, dtype=np.uint8)

    def save(self, path) -> None:
        from safetensors.numpy import save_file

        save_file(
            {
                "points": self.points,
                "normals": self.normals,
                "uvs": self.uvs,
                "triangles": self.triangles,
                "arch": self.arch,
                "part": self.part,
            },
            str(path),
            metadata={"format": FORMAT},
        )

    @classmethod
    def load(cls, path) -> "TeethData":
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as f:
            if (f.metadata() or {}).get("format") != FORMAT:
                raise ValueError(f"{path} is not a teeth file ({FORMAT}).")
            return cls(**{k: f.get_tensor(k) for k in f.keys()})


def rigid(rest: np.ndarray, moved: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rotation R and translation t of the least-squares rigid map (Kabsch) from
    ``rest`` (n, 3) to ``moved``: moved ~ rest @ R.T + t."""
    a0, b0 = rest.mean(0), moved.mean(0)
    u, _, vt = np.linalg.svd((rest - a0).T @ (moved - b0))
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    R = (u @ np.diag([1.0, 1.0, d]) @ vt).T
    return R, b0 - a0 @ R.T


def _components(model) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces = model.faces.cpu().numpy()
    n = len(model.template_vertices)
    edges = np.concatenate(
        [faces[:, [k, (k + 1) % faces.shape[1]]] for k in range(faces.shape[1])]
    )
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def mouth_rows(model, mesh=None) -> tuple[np.ndarray, np.ndarray]:
    """
    The head's rows and the tongue's rows of a mesh (anny's own when None, else a
    :class:`~opensculptboy.detail.subdivision.DetailTopology`): the vertices that the
    ``head`` bone skins alone (:data:`HEAD_WEIGHT`) on the body's surface, and the vertices of
    the tongue, the largest closed part apart from the body and the eyeballs (empty for a model
    without one). Raises ValueError for a rig without a ``head`` bone.
    """
    labels = list(model.bone_labels)
    if "head" not in labels:
        raise ValueError("The teeth need a rig with a 'head' bone.")
    weights = model.vertex_bone_weights.detach().cpu().double().numpy()
    indices = model.vertex_bone_indices.cpu().numpy()
    head_weight = (weights * (indices == labels.index("head"))).sum(1)
    comp = _components(model)
    counts = np.bincount(comp)
    body = np.argmax(counts)
    head = (head_weight >= HEAD_WEIGHT) & (comp == body)
    eyes = np.zeros(len(comp), bool)
    try:
        from opensculptboy.export.vrm_tables import eyeball_vertices

        for rows in eyeball_vertices(model).values():
            eyes[rows] = True
    except ValueError:
        pass
    tongue = np.zeros(len(comp), bool)
    for c in np.argsort(-counts):
        if c == body or counts[c] < 20:
            continue
        rows = comp == c
        if eyes[rows].any():
            continue
        tongue = rows
        break
    if mesh is not None and mesh.operators:
        head = np.asarray(mesh.apply(head.astype(np.float64))) >= HEAD_WEIGHT
        tongue = np.asarray(mesh.apply(tongue.astype(np.float64))) > 0.5
    return np.flatnonzero(head), np.flatnonzero(tongue)


def cavity_rows(model, mesh=None, keep_rings: int = 1) -> np.ndarray:
    """
    The rows of anny's mouth cavity that the inner mouth of a reference hides on a mesh
    (anny's own when None, else a detail-layer mesh): the vertices of the base mesh's
    ``mouth_cavity`` farther than ``keep_rings`` faces from the skin outside it, so that the
    inner margin of the lips stays.
    """
    from anny.faces.authoring.base_mesh import base_mesh
    from opensculptboy.topology.lookup import base_mesh_rows

    bm = base_mesh()
    cavity = np.zeros(len(bm.V), bool)
    cavity[np.unique(bm.quads[bm.face_labels == "mouth_cavity"])] = True
    cavity = cavity[base_mesh_rows(model)]
    faces = model.faces.cpu().numpy()
    inner = cavity.copy()
    for _ in range(keep_rings):
        border = inner[faces].any(1) & ~inner[faces].all(1)
        inner[np.unique(faces[border])] = False
    if mesh is not None and mesh.operators:
        inner = np.asarray(mesh.apply(inner.astype(np.float64))) > 0.999
    return np.flatnonzero(inner)


# the reach (metres) of a column of the inner mouth: a vertex of anny's cavity hides when the
# inner mouth lies in front of it within this distance across the depth
COLUMN = 0.004
# how far (metres) a vertex of the cavity may stand in front of the inner mouth and still hide
FRONT_MARGIN = 0.0005


def behind_mouth(rest: np.ndarray, rows: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    The rows of anny's cavity (``rows`` of ``rest``) that lie behind the front of an inner
    mouth (``points``, in the same frame; Anny faces -Y): those with a point of the mouth
    within :data:`COLUMN` across the depth (X and Z) whose depth is no more than
    :data:`FRONT_MARGIN` behind theirs. The cavity's lining in front of the mouth stays, so
    that it still closes the space between the lips and the front of the mouth.
    """
    from scipy.spatial import cKDTree

    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0 or len(points) == 0:
        return rows
    columns = cKDTree(points[:, [0, 2]]).query_ball_point(rest[rows][:, [0, 2]], COLUMN)
    front = np.array(
        [points[c, 1].min() if c else np.inf for c in columns], dtype=np.float64
    )
    return rows[rest[rows, 1] > front - FRONT_MARGIN]


def _base_mesh(model, base):
    """Triangles, UV triangles and UVs (OBJ convention) of the base mesh."""
    if base is not None:
        return base.triangles, base.uv_triangles, base.uvs
    from opensculptboy.export.body import triangulated_faces

    triangles, uv_triangles = triangulated_faces(model)
    uvs = model.texture_coordinates
    uvs = None if uvs is None else uvs.detach().cpu().double().numpy()
    return triangles, uv_triangles, uvs


@dataclasses.dataclass
class ToothedMesh:
    """
    A mesh of an export with its teeth appended after its ``base_count`` vertices and its
    triangles. ``triangles``, ``uv_triangles`` and ``uvs`` take the place of the base mesh's in
    :func:`~opensculptboy.export.body.build_body`; :meth:`part_mask` gives the triangles of
    the teeth and of the gums.
    """

    base_count: int
    base_triangle_count: int
    teeth: TeethData
    rest: np.ndarray  # (N, 3) the base mesh's rest vertices
    head: np.ndarray  # rows of the head
    tongue: np.ndarray  # rows of the tongue
    triangles: np.ndarray
    uv_triangles: np.ndarray
    uvs: np.ndarray
    base_triangles: np.ndarray
    extras: dict = dataclasses.field(default_factory=dict)
    # the rig's bone labels, for the skin of a reference's inner mouth
    # (opensculptboy.mouth.inner.MouthData)
    labels: list | None = None
    # the base mesh's triangles that the inner mouth hides (anny's cavity and tongue)
    hidden: np.ndarray | None = None

    @property
    def vertex_count(self) -> int:
        return self.base_count + len(self.teeth.points)

    def _head_motion(self, x: np.ndarray):
        return rigid(self.rest[self.head], np.asarray(x)[self.head])

    def vertices(self, x: np.ndarray) -> np.ndarray:
        """(N + M, 3) positions of the mesh's (N, 3), the teeth moved with the head."""
        x = np.asarray(x, dtype=np.float64)
        R, t = self._head_motion(x)
        return np.concatenate([x, self.teeth.points @ R.T + t])

    def targets(self, offsets: np.ndarray, overrides: dict | None = None) -> np.ndarray:
        """(R, N + M, 3) morph-target offsets of the mesh's (R, N, 3) (see the module
        documentation). ``overrides`` gives the offsets (M, 3) of the teeth's points in some
        targets by row, in place of those the mesh's offsets give them: a reference's mouth
        poses move its inner mouth by its own motion (:mod:`opensculptboy.mouth.poses`)."""
        offsets = np.asarray(offsets, dtype=np.float64)
        if offsets.ndim != 3 or len(offsets) == 0:
            return offsets
        points = self.teeth.points
        lower = self.teeth.arch == 1
        head = self.head
        still = head[np.abs(offsets[:, head]).max(axis=(0, 2)) < 1e-9]
        anchors = still if len(still) >= 8 else head
        out = np.zeros((len(offsets), len(points), 3))
        follow = self._tongue_points()
        for r, d in enumerate(offsets):
            R, t = rigid(self.rest[anchors], self.rest[anchors] + d[anchors])
            moved = points @ R.T + t - points
            if len(self.tongue) >= 4 and lower.any():
                jaw = self._jaw_motion(d)
                if jaw is not None:
                    moved[lower] = points[lower] @ jaw[0].T + jaw[1] - points[lower]
                elif follow is not None and follow.any():
                    moved[follow] = self._tongue_follow(d, follow)
            out[r] = moved
        for r, moved in (overrides or {}).items():
            out[r] = moved
        return np.concatenate([offsets, out], axis=1)

    def _tongue_points(self):
        """The points of a reference's tongue (MouthData), which follow anny's tongue in a
        target that bends it; None for teeth alone."""
        part_points = getattr(self.teeth, "part_points", None)
        return None if part_points is None else part_points("tongue")

    def _tongue_follow(self, d: np.ndarray, follow: np.ndarray) -> np.ndarray:
        """The offsets of the points ``follow`` in a target ``d`` of the base mesh: the
        inverse-distance blend of the displacements of anny's 4 nearest tongue vertices."""
        from scipy.spatial import cKDTree

        tongue = self.rest[self.tongue]
        k = min(4, len(tongue))
        dist, idx = cKDTree(tongue).query(self.teeth.points[follow], k=k)
        dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
        w = 1.0 / np.maximum(dist, 1e-5)
        w /= w.sum(1, keepdims=True)
        return np.einsum("nk,nkc->nc", w, d[self.tongue][idx])

    def _jaw_motion(self, d: np.ndarray):
        """The rigid motion of the tongue in one target (R, t), or None when the target does
        not move the tongue rigidly."""
        p = self.rest[self.tongue]
        q = p + d[self.tongue]
        motion = float(np.linalg.norm(d[self.tongue], axis=1).mean())
        if motion < 1e-7:
            return None
        R, t = rigid(p, q)
        residual = float(np.linalg.norm(p @ R.T + t - q, axis=1).mean())
        if residual > RIGID_SHARE * motion:
            return None
        return R, t

    def normals(self, x: np.ndarray, base_normals: np.ndarray | None = None):
        """(N + M, 3) unit normals for positions ``x`` (N, 3) of the mesh: the base mesh's
        (``base_normals``, else those of its triangles) and the teeth's turned with the
        head."""
        x = np.asarray(x, dtype=np.float64)
        if base_normals is None:
            from opensculptboy.export.body import vertex_normals

            base_normals = vertex_normals(x, self.base_triangles)
        R, _ = self._head_motion(x)
        return np.concatenate(
            [np.asarray(base_normals, dtype=np.float64), self.teeth.normals @ R.T]
        )

    def skin_weights(self, weights, indices) -> tuple[np.ndarray, np.ndarray]:
        """(N + M, K) skin weights and bone indices: the teeth take the mean weights of the
        head's vertices, its K strongest bones renormalised."""
        weights = np.asarray(weights, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
        k = weights.shape[1]
        own = self._own_skin(k)
        if own is not None:
            return (
                np.concatenate([weights, own[0]]),
                np.concatenate([indices, own[1]]),
            )
        bones = int(indices.max(initial=0)) + 1
        dense = np.zeros(bones)
        np.add.at(dense, indices[self.head].ravel(), weights[self.head].ravel())
        order = np.argsort(-dense, kind="stable")[:k]
        w = dense[order] / max(dense[order].sum(), 1e-12)
        i = np.where(w > 0, order, 0)
        m = len(self.teeth.points)
        return (
            np.concatenate([weights, np.tile(w, (m, 1))]),
            np.concatenate([indices, np.tile(i, (m, 1))]),
        )

    def _own_skin(self, k: int):
        """The skin of a reference's inner mouth on the file's rig, as (M, k) weights and bone
        indices, or None for teeth without a skin of their own or a rig without the head. A
        bone that the rig lacks gives its weight to its nearest kept ancestor: a tongue bone
        to ``jaw``, and ``jaw`` to ``head``."""
        slots = getattr(self.teeth, "bone_slots", None)
        if slots is None or not self.labels or "head" not in self.labels:
            return None
        labels = list(self.labels)

        def kept(name):
            while name not in labels:
                name = "jaw" if name.startswith("tongue") else "head"
            return labels.index(name)

        column = np.array([kept(b) for b in self.teeth.bones], dtype=np.int64)
        m = len(self.teeth.points)
        bones = np.unique(column)
        dense = np.zeros((m, len(bones)))
        where = np.searchsorted(bones, column[slots])
        np.add.at(
            dense,
            (np.repeat(np.arange(m), slots.shape[1]), where.ravel()),
            self.teeth.bone_weights.ravel(),
        )
        order = np.argsort(-dense, axis=1, kind="stable")[:, :k]
        w = np.take_along_axis(dense, order, 1)
        i = bones[order]
        if w.shape[1] < k:
            w = np.pad(w, ((0, 0), (0, k - w.shape[1])))
            i = np.pad(i, ((0, 0), (0, k - i.shape[1])))
        w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
        return w, np.where(w > 0, i, 0)

    def anny_vertex(self, base_source: np.ndarray | None = None) -> np.ndarray:
        """The Anny vertex (or the base mesh vertex) of each vertex of the mesh: the teeth's
        nearest head vertex."""
        base = np.arange(self.base_count) if base_source is None else base_source
        rest = self.rest[self.head]
        nearest = np.argmin(
            np.linalg.norm(self.teeth.points[:, None] - rest[None], axis=2), axis=1
        )
        return np.concatenate([base, base[self.head[nearest]]])

    def extend_mask(self, mask: np.ndarray | None) -> np.ndarray | None:
        """A mask over the base mesh's triangles as a mask over this mesh's triangles."""
        if mask is None:
            return None
        extra = len(self.triangles) - self.base_triangle_count
        return np.concatenate([np.asarray(mask, bool), np.zeros(extra, bool)])

    def hidden_mask(self) -> np.ndarray:
        """The mask of this mesh's triangles that the inner mouth hides (none for teeth)."""
        mask = np.zeros(len(self.triangles), bool)
        if self.hidden is not None:
            mask[: self.base_triangle_count] = self.hidden
        return mask

    def part_mask(self, *names: str) -> np.ndarray:
        """The mask of the triangles of the named parts (``"teeth"``, ``"gums"``)."""
        mask = np.zeros(len(self.triangles), bool)
        part = self.teeth.part
        for name in names:
            mask[self.base_triangle_count :] |= part == PARTS.index(name)
        return mask


def toothed_mesh(model, mesh, base, rest: np.ndarray, teeth: TeethData) -> ToothedMesh:
    """
    The mesh of a file with ``teeth`` appended: ``mesh`` is the file's mesh before the eyes
    (a :class:`~opensculptboy.detail.subdivision.DetailTopology`, or None for anny's own
    mesh), whose rows name the head and the tongue (:func:`mouth_rows`); ``base`` is the mesh
    that the teeth follow (``mesh``, or the :class:`~opensculptboy.eyes.mesh.EyedMesh` of the
    eyes, whose first rows are ``mesh``'s), and ``rest`` its rest vertices in Anny's frame.
    """
    rest = np.asarray(rest, dtype=np.float64)
    head, tongue = mouth_rows(model, mesh)
    if len(head) < 8:
        raise ValueError("The teeth need a mesh with a head.")
    triangles, uv_triangles, uvs = _base_mesh(model, base)
    if uvs is None or uv_triangles is None:
        raise ValueError("The teeth need a mesh with UVs.")
    count, uv_count = len(rest), len(uvs)
    hidden = None
    if getattr(teeth, "interior", False):  # a reference's inner mouth hides anny's own
        gone = np.zeros(count, bool)
        gone[behind_mouth(rest, cavity_rows(model, mesh), teeth.points)] = True
        gone[tongue] = True
        hidden = gone[np.asarray(triangles)].all(1)
    return ToothedMesh(
        labels=list(model.bone_labels),
        hidden=hidden,
        base_count=count,
        base_triangle_count=len(triangles),
        teeth=teeth,
        rest=rest,
        head=head,
        tongue=tongue,
        triangles=np.concatenate([triangles, teeth.triangles + count]).astype(np.int64),
        uv_triangles=np.concatenate([uv_triangles, teeth.triangles + uv_count]).astype(
            np.int64
        ),
        uvs=np.concatenate([uvs, teeth.uvs]),
        base_triangles=np.asarray(triangles, dtype=np.int64),
    )


def card_teeth(model, character, detail_dir=None, rest=None) -> TeethData | None:
    """
    The inner mouth of a card's ``material.mouth`` (:class:`~opensculptboy.mouth.inner.MouthData`),
    else the teeth of its ``material.teeth``, or None without either: the bake's mesh
    (``source`` ``"reference"``, a path relative to ``detail_dir``), or the generic teeth of
    :mod:`opensculptboy.teeth.generic` behind the lips of ``rest`` (the rest vertices of
    anny's own mesh for the character).
    """
    mouth = character.mouth_material()
    if mouth is not None:  # a reference's inner mouth (opensculptboy.mouth.inner)
        from opensculptboy.mouth.inner import MouthData

        return MouthData.load(pathlib.Path(detail_dir or ".") / mouth.mesh)
    material = character.teeth_material()
    if material is None:
        return None
    if material.source == "reference":
        path = pathlib.Path(detail_dir or ".") / material.mesh
        return TeethData.load(path)
    from opensculptboy.teeth.generic import generic_teeth

    return generic_teeth(model, rest, gums=material.gums)

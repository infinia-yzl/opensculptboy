# OpenSculptBoy
# Apache License, Version 2.0
"""
The eyes on the mesh that an exporter writes.

:func:`eyed_mesh` takes the mesh of a file (anny's own mesh, or a level of the detail layer)
and its rest vertices, and returns an :class:`EyedMesh`: the same mesh without anny's eyeball
triangles, with the eye parts of :mod:`opensculptboy.eyes.geometry` appended as new vertices
and triangles. Each part is a linear map of some of the mesh's own vertices, its anchors:

- the eyeball and the cornea follow the least-squares affine map of their eyeball's vertices
  (exact for the turns, moves and scales that anny's eye bones and eye morph targets give an
  eyeball), and take the mean skin weights of those vertices, so that they turn rigidly with
  the eye bone;
- the caruncle follows the affine map of the skin around the inner corner of the eye, and
  takes the skin weights of its nearest skin vertices.

Because the maps are linear, :meth:`EyedMesh.vertices` and :meth:`EyedMesh.targets` carry
any vertex positions and morph-target offsets of the mesh over to the parts: the rest mesh, the
morph targets, and the T-pose bind mesh and its targets of a VRM file. Anny's eyeball vertices
are left unused, so the file does not hold them. The parts' vertices name the nearest Anny
vertex of their anchors in ``_ANNY_VERTEX``.

The mesh's eyeball triangles also give :func:`eye_texture`, which draws the eye texture
(:mod:`opensculptboy.eyes.texture`) from the card's eye settings and, for a card with a baked
eye texture on anny's atlas, from that texture through those triangles.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from opensculptboy.eyes import geometry
from opensculptboy.eyes.texture import polar_to_uv

# the triangles of each part: the eyeball (iris and sclera), the cornea shell, the caruncle;
# the skin of the socket keeps its own triangles of the mesh
PARTS = ("eyeball", "cornea", "caruncle")


@dataclasses.dataclass
class EyePart:
    """
    A part of an eye: its rest points, normals and UVs (M, ...), its triangles (T, 3) over its
    own points, and the linear map from its ``anchors`` (rows of the mesh): positions =
    ``operator`` @ X[anchors]; skin weights = ``blend`` @ the anchors' weights.
    """

    name: str
    side: str
    rest: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    triangles: np.ndarray
    anchors: np.ndarray
    operator: np.ndarray
    blend: np.ndarray
    source: np.ndarray  # (M,) the anchor row nearest each point


def affine_operator(anchors: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    The (M, n) matrix W with W @ anchors = points that maps any positions of the anchors
    (n, 3) to the points (M, 3) by the least-squares affine map from the rest anchors: rows
    sum to 1, so it carries offsets (morph targets) as well as positions.
    """
    hom = np.c_[np.asarray(anchors, dtype=np.float64), np.ones(len(anchors))]
    return np.c_[
        np.asarray(points, dtype=np.float64), np.ones(len(points))
    ] @ np.linalg.pinv(hom)


def _rotation(rest: np.ndarray, moved: np.ndarray) -> np.ndarray:
    """The rotation of the least-squares rigid map (Kabsch) from ``rest`` to ``moved``."""
    a = rest - rest.mean(0)
    b = moved - moved.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    return (u @ np.diag([1.0, 1.0, d]) @ vt).T


@dataclasses.dataclass
class EyedMesh:
    """
    A mesh of an export with its eyes: the base mesh's triangles that remain (``kept``) and
    the eye parts appended after its ``base_count`` vertices. ``triangles``, ``uv_triangles``
    and ``uvs`` take the place of the base mesh's in
    :func:`~opensculptboy.export.body.build_body`; ``part_masks`` give the triangles of each
    part, and ``socket`` those of the skin of the socket (the conjunctiva).
    """

    base_count: int
    parts: list[EyePart]
    kept: np.ndarray  # (T0,) bool over the base mesh's triangles
    socket: np.ndarray  # (T0,) bool: the skin of the socket that shows at the lids
    triangles: np.ndarray
    uv_triangles: np.ndarray
    uvs: np.ndarray
    frames: dict
    bulge: dict  # side -> the share of the corneal bulge kept
    eyeballs: (
        dict  # side -> (points in the eye's frame, triangles, corner UVs), the base
    )
    #                 mesh's eyeball for the texture
    shapes: dict = dataclasses.field(default_factory=dict)  # side -> geometry.EyeShape
    extras: dict = dataclasses.field(default_factory=dict)  # what the file records
    # the eyes with the iris and pupil sizes they were built with (numbers), and the sizes
    # found in the baked eye texture by side (empty without one)
    eye: object = None
    measured: dict = dataclasses.field(default_factory=dict)

    def radii_mm(self) -> dict[str, float]:
        """The radius of each eyeball in millimetres."""
        return {side: f.radius * 1000.0 for side, f in self.frames.items()}

    @property
    def vertex_count(self) -> int:
        return self.base_count + sum(len(p.rest) for p in self.parts)

    def vertices(self, x: np.ndarray) -> np.ndarray:
        """(N + M, 3) positions of the mesh's (N, 3), the parts' carried over."""
        x = np.asarray(x, dtype=np.float64)
        return np.concatenate([x] + [p.operator @ x[p.anchors] for p in self.parts])

    def targets(self, offsets: np.ndarray) -> np.ndarray:
        """(R, N + M, 3) morph-target offsets of the mesh's (R, N, 3)."""
        offsets = np.asarray(offsets, dtype=np.float64)
        parts = [
            np.einsum("mn,rnd->rmd", p.operator, offsets[:, p.anchors])
            for p in self.parts
        ]
        return np.concatenate([offsets] + parts, axis=1)

    def normals(
        self, x: np.ndarray, base_normals: np.ndarray | None = None
    ) -> np.ndarray:
        """(N + M, 3) unit normals for positions ``x`` (N, 3) of the mesh: the base mesh's
        (``base_normals``, else those of its remaining triangles) and the parts' rest normals
        turned with their anchors."""
        x = np.asarray(x, dtype=np.float64)
        if base_normals is None:
            from opensculptboy.export.body import vertex_normals

            kept = self.triangles[: int(self.kept.sum())]
            base_normals = vertex_normals(x, kept)
        out = [np.asarray(base_normals, dtype=np.float64)]
        for p in self.parts:
            rest = p.operator @ x[p.anchors]
            turn = _rotation(p.rest, rest) if len(p.rest) > 2 else np.eye(3)
            out.append(p.normals @ turn.T)
        return np.concatenate(out)

    def skin_weights(self, weights, indices) -> tuple[np.ndarray, np.ndarray]:
        """(N + M, K) skin weights and bone indices of the mesh's (N, K): each part's blend
        of its anchors' weights, its K strongest bones renormalised."""
        weights = np.asarray(weights, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
        k = weights.shape[1]
        bones = int(indices.max(initial=0)) + 1
        out_w, out_i = [weights], [indices]
        for p in self.parts:
            dense = np.zeros((len(p.anchors), bones))
            np.add.at(
                dense,
                (np.repeat(np.arange(len(p.anchors)), k), indices[p.anchors].ravel()),
                weights[p.anchors].ravel(),
            )
            # each distinct row of the blend is mixed once, so that identical rows (the
            # eyeball's and the cornea's) give bit-identical weights on every BLAS
            rows, inverse = np.unique(p.blend, axis=0, return_inverse=True)
            mixed = (rows @ dense)[np.asarray(inverse).reshape(-1)]
            order = np.argsort(-mixed, axis=1, kind="stable")[:, :k]
            w = np.take_along_axis(mixed, order, axis=1)
            w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
            out_w.append(w)
            out_i.append(np.where(w > 0, order, 0))
        return np.concatenate(out_w), np.concatenate(out_i)

    def anny_vertex(self, base_source: np.ndarray | None = None) -> np.ndarray:
        """The Anny vertex (or the base mesh vertex) of each vertex of the mesh: the parts'
        nearest anchor."""
        base = np.arange(self.base_count) if base_source is None else base_source
        return np.concatenate([base] + [base[p.source] for p in self.parts])

    def extend_mask(self, mask: np.ndarray | None) -> np.ndarray | None:
        """A mask over the base mesh's triangles as a mask over this mesh's triangles."""
        if mask is None:
            return None
        extra = len(self.triangles) - int(self.kept.sum())
        return np.concatenate(
            [np.asarray(mask, bool)[self.kept], np.zeros(extra, bool)]
        )

    def part_mask(self, *names: str) -> np.ndarray:
        """The mask of the triangles of the named parts (``"socket"`` for the socket's)."""
        mask = np.zeros(len(self.triangles), bool)
        start = int(self.kept.sum())
        if "socket" in names:
            mask[:start] = self.socket[self.kept]
        for p in self.parts:
            if p.name in names:
                mask[start : start + len(p.triangles)] = True
            start += len(p.triangles)
        return mask


def eyeball_rows(model, mesh=None) -> dict[str, np.ndarray]:
    """The rows of each eyeball (``"L"``, ``"R"``) in the mesh (anny's own when None)."""
    from opensculptboy.export.vrm_tables import eyeball_vertices

    eyes = eyeball_vertices(model)
    if mesh is None or not mesh.operators:  # anny's own vertices
        return eyes
    out = {}
    for side, rows in eyes.items():
        indicator = np.zeros(len(model.template_vertices))
        indicator[rows] = 1.0
        out[side] = np.flatnonzero(np.asarray(mesh.apply(indicator)) > 0.5)
    return out


def _base_mesh(model, mesh):
    """Triangles, UV triangles and UVs (OBJ convention) of the base mesh."""
    if mesh is not None:
        return mesh.triangles, mesh.uv_triangles, mesh.uvs
    from opensculptboy.export.body import triangulated_faces

    triangles, uv_triangles = triangulated_faces(model)
    uvs = model.texture_coordinates
    uvs = None if uvs is None else uvs.detach().cpu().double().numpy()
    return triangles, uv_triangles, uvs


def measure_baked(model, eyeballs: dict, radii: dict, baked: bytes) -> dict:
    """The iris and pupil diameters (mm) by side that :func:`~opensculptboy.eyes.texture.
    find_iris` finds in a baked eye texture (PNG or JPEG bytes on anny's atlas) carried over
    through the mesh's ``eyeballs`` (:attr:`EyedMesh.eyeballs`); sides without an iris are
    left out."""
    import io

    from PIL import Image

    from opensculptboy.eyes import texture as tex
    from opensculptboy.texture.eyes import eye_uv_box

    image = tex.linear(np.asarray(Image.open(io.BytesIO(baked)).convert("RGB")))
    carried = tex.carry_baked(image, eye_uv_box(model), eyeballs)
    s, phi, left = tex.texel_polar()
    out = {}
    for side, half in (("L", left), ("R", ~left)):
        hit = tex.find_iris(carried[half], s[half], phi[half])
        if hit is not None:
            out[side] = {
                "pupil_mm": round(float(2.0 * hit[1] * radii[side]), 2),
                "iris_mm": round(float(2.0 * hit[2] * radii[side]), 2),
            }
    return out


def _mean_sizes(measured: dict) -> dict | None:
    if not measured:
        return None
    return {
        key: float(np.mean([m[key] for m in measured.values()]))
        for key in ("iris_mm", "pupil_mm")
    }


# the labels of the faces of the sculpt topology's eye-corner patch that take the conjunctiva's
# pink, wet material (opensculptboy.topology.patches.eye_corner)
CORNER_LABELS = ("eye_corner/caruncle", "eye_corner/conjunctiva")
# of those faces, only the triangles whose corners all lie within this many eyeball radii of
# the eye's centre take it: the tissue inside the lid opening lies on the eyeball or just over
# it, while the lids' outer skin and the face around the corner lie further out (the opening of
# geometry.caruncle uses the same reach)
CORNER_REACH = geometry.CARUNCLE_NEAR
# the texture slots of a baked skin: a file with one of them takes the corner's appearance from
# the bake (the reference's own caruncle and lids), not from the conjunctiva's material
BAKED_SKIN_SLOTS = ("base_color", "head_base_color")


def baked_skin(images) -> bool:
    """Whether the textures of a file (slot -> image) hold a baked skin."""
    return any(slot in images for slot in BAKED_SKIN_SLOTS)


def _sculpt_data(model):
    """The composed patches of a model on the ``sculpt`` topology, or None."""
    data = getattr(model, "sculpt_data", None)
    if data is not None:
        return data
    config = getattr(model, "config", None)
    topology = getattr(config, "topology", None)
    if not isinstance(topology, str) or not topology.startswith("sculpt"):
        return None
    from opensculptboy.topology import sculpt_data

    return sculpt_data()


def corner_triangles(
    model, rest, triangles, uv_triangles, uvs, frames
) -> np.ndarray | None:
    """
    The mask of the triangles of a mesh of the ``sculpt`` topology (level 0 or a detail level)
    that belong to the caruncle and the conjunctiva of the eye-corner patch inside the lid
    opening, or None for a model without them. A triangle belongs to them when its UV centroid
    lies in the UV quad of one of those faces (the patch's faces keep their own region of the
    atlas, and the detail levels subdivide the UVs inside each face) and its corners lie within
    :data:`CORNER_REACH` eyeball radii of an eye's centre: where the lids of a character meet
    further in than the default character's, the patch's faces outside the opening lie on the
    skin of the lids and the face, and keep the skin's material.
    """
    data = _sculpt_data(model)
    if data is None:
        return None
    pink = [i for i, label in enumerate(data.labels) if label in CORNER_LABELS]
    chosen = np.isin(data.face_labels, pink)
    if not chosen.any():
        return None
    atlas = model.texture_coordinates.detach().cpu().double().numpy()
    quads = atlas[data.uv_faces[chosen]]  # (P, 4, 2)
    centre = uvs[uv_triangles].mean(1)  # (T, 2)
    inside = np.zeros(len(triangles), bool)
    for corners in ([0, 1, 2], [0, 2, 3]):
        a, b, c = (quads[:, k] for k in corners)
        det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (
            b[:, 1] - a[:, 1]
        )
        ok = np.abs(det) > 1e-14
        for p, (a0, b0, c0, d0) in enumerate(zip(a, b, c, det)):
            if not ok[p]:
                continue
            q = centre - a0
            u = ((c0[1] - a0[1]) * q[:, 0] - (c0[0] - a0[0]) * q[:, 1]) / d0
            v = (-(b0[1] - a0[1]) * q[:, 0] + (b0[0] - a0[0]) * q[:, 1]) / d0
            inside |= (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    corners = rest[triangles]  # (T, 3, 3)
    near = np.zeros(len(triangles), bool)
    for frame in frames.values():
        reach = np.linalg.norm(corners - frame.centre, axis=2).max(1)
        near |= reach < CORNER_REACH * frame.radius
    return inside & near


def eyed_mesh(
    model,
    mesh,
    rest: np.ndarray,
    eye,
    kind: str = "gltf",
    baked: bytes | None = None,
    skin_baked: bool = False,
) -> EyedMesh:
    """
    The mesh of a file with the eyes of ``eye`` (an :class:`~opensculptboy.material.
    EyeMaterial`): ``mesh`` is the base mesh (a :class:`~opensculptboy.detail.subdivision.
    DetailTopology`, or None for anny's own mesh) and ``rest`` (N, 3) its rest vertices in
    Anny's frame. ``kind`` is ``"gltf"`` (a cornea shell over a recessed iris when
    ``eye.cornea``, which ``"auto"`` turns off for baked eyes) or ``"vrm"`` (the
    single-surface eyeball, at fewer segments). ``baked``
    is the file's baked eye texture (on anny's atlas), if any: an ``iris_mm`` or ``pupil_mm``
    of ``"auto"`` or ``"baked"`` takes the size found in it (:func:`measure_baked`, the mean
    of both eyes), and ``"auto"`` the default without it. ``skin_baked`` says that the file's
    skin carries a baked texture (:func:`baked_skin`): on the ``sculpt`` topology the eye
    corner and the skin of the socket then keep the skin's material, so that they show the
    reference's own appearance from the bake, and the conjunctiva's material is left to
    characters without one. Raises ValueError for a mesh without anny's eyeballs, and for
    ``"baked"`` sizes that no baked iris gives.
    """
    from opensculptboy.export.body import vertex_normals

    rest = np.asarray(rest, dtype=np.float64)
    triangles, uv_triangles, uvs = _base_mesh(model, mesh)
    if uvs is None or uv_triangles is None:
        raise ValueError("The eyes need a mesh with UVs.")
    rows = eyeball_rows(model, mesh)
    on_eye = np.zeros(len(rest), bool)
    for r in rows.values():
        on_eye[r] = True
    eyeball_tris = on_eye[triangles].all(1)
    kept = ~eyeball_tris
    skin_tris = triangles[kept]
    normals = vertex_normals(rest, skin_tris)
    used = np.zeros(len(rest), bool)
    used[skin_tris] = True
    skin = used & ~on_eye

    # a cornea of "auto": one surface for the eyes of a bake, as the reference draws them
    eye = eye.resolved(baked is not None and eye.source in ("auto", "baked"))
    two_surface = kind == "gltf" and eye.cornea
    segments = geometry.SEGMENTS[kind]
    parts: list[EyePart] = []
    socket_vertices = np.zeros(len(rest), bool)
    frames, bulges, eyeballs, shapes = {}, {}, {}, {}
    skin_rows = np.flatnonzero(skin)

    # the frames, and the bulge and the radius scale that clear the lids: the smaller of the
    # two eyes', so that both eyes match (the scale does not depend on the iris's size, the
    # bulge does)
    def clearance(iris_mm):
        clear = {}
        for side in ("L", "R"):
            frame = geometry.eye_frame(side, rest[rows[side]])
            shape = geometry.EyeShape.of(iris_mm, frame.radius, eye.cornea_radius_mm)
            shell = 1.0 + shape.gap if two_surface else 1.0
            frames[side] = frame
            clear[side] = geometry.bulge_clearance(
                frame, rest[skin_rows], normals[skin_rows], shape, shell
            )
        return clear

    nominal = eye.sized()  # the card's sizes, or the defaults for those from a bake
    clear = clearance(nominal.iris_mm)
    bulge = min(b for b, _ in clear.values()) * eye.bulge
    scale = min(c for _, c in clear.values())
    # on the sculpt topology the mesh has its own caruncle and conjunctival fold (the
    # eye-corner patch): no caruncle part, and the patch's faces take the pink material
    corner = corner_triangles(model, rest, triangles, uv_triangles, uvs, frames)
    # a fitted character's corner takes the reference's appearance from the bake
    follow_bake = skin_baked and _sculpt_data(model) is not None
    for side in ("L", "R"):
        frame = frames[side]
        frame.radius *= scale
        # the base mesh's eyeball, for the eye texture
        mine = np.zeros(len(rest), bool)
        mine[rows[side]] = True
        tris = eyeball_tris & mine[triangles].all(1)
        eyeballs[side] = (
            frame.to_local(rest),
            triangles[tris],
            uvs[uv_triangles[tris]],
        )
    # the sizes of the iris and the pupil: the card's, or those of the baked eyes
    measured = {}
    if eye.wants_measure() and baked is not None and eye.source in ("auto", "baked"):
        radii = {side: f.radius * 1000.0 for side, f in frames.items()}
        measured = measure_baked(model, eyeballs, radii, baked)
    eye = eye.sized(_mean_sizes(measured))
    if eye.iris_mm != nominal.iris_mm:
        clear = clearance(eye.iris_mm)
        bulge = min(b for b, _ in clear.values()) * eye.bulge
        for frame in frames.values():
            frame.radius *= scale
    for side in ("L", "R"):
        anchors = rows[side]
        frame = frames[side]
        shape = geometry.EyeShape.of(eye.iris_mm, frame.radius, eye.cornea_radius_mm)
        bulges[side], shapes[side] = bulge, shape
        if two_surface:
            surfaces = {
                "eyeball": geometry.revolve(geometry.eyeball_profile(shape), segments),
                "cornea": geometry.revolve(
                    geometry.cornea_profile(shape, bulge), segments
                ),
            }
        else:
            surfaces = {
                "eyeball": geometry.revolve(
                    geometry.single_profile(shape, bulge), segments
                )
            }
        blend = np.full((1, len(anchors)), 1.0 / len(anchors))
        for name, surface in surfaces.items():
            parts.append(_part(name, side, frame, surface, rest, anchors, blend=blend))
        if eye.caruncle:
            socket_vertices |= geometry.socket(frame, rest, normals, skin)
            if corner is not None:
                continue
            surface = geometry.caruncle(frame, rest, skin_tris, shape)
            if surface is None:
                continue
            points = frame.to_world(surface.points)
            centre = points.mean(0)
            dist = np.linalg.norm(rest[skin_rows] - centre, axis=1)
            near = skin_rows[dist < 0.45 * frame.radius]
            if len(near) < 8:
                near = skin_rows[np.argsort(dist)[:8]]
            # each point blends the skin weights of its 4 nearest anchors
            d = np.linalg.norm(points[:, None] - rest[near][None], axis=2)
            order = np.argsort(d, axis=1)[:, :4]
            inv = 1.0 / np.maximum(np.take_along_axis(d, order, 1), 1e-9)
            blend_c = np.zeros((len(points), len(near)))
            np.put_along_axis(blend_c, order, inv / inv.sum(1, keepdims=True), 1)
            parts.append(
                _part("caruncle", side, frame, surface, rest, near, blend=blend_c)
            )
    socket = (
        socket_vertices[triangles].all(1) & kept
        if eye.caruncle and not follow_bake
        else np.zeros(len(triangles), bool)
    )
    if corner is not None and eye.caruncle and not follow_bake:
        socket |= corner & kept
    # the combined mesh: the base triangles that remain, then the parts'
    count, uv_count = len(rest), len(uvs)
    all_tris, all_uv_tris, all_uvs = [triangles[kept]], [uv_triangles[kept]], [uvs]
    for p in parts:
        all_tris.append(p.triangles + count)
        all_uv_tris.append(p.triangles + uv_count)
        all_uvs.append(p.uvs)
        count += len(p.rest)
        uv_count += len(p.uvs)
    return EyedMesh(
        base_count=len(rest),
        parts=parts,
        kept=kept,
        socket=socket,
        triangles=np.concatenate(all_tris).astype(np.int64),
        uv_triangles=np.concatenate(all_uv_tris).astype(np.int64),
        uvs=np.concatenate(all_uvs),
        frames=frames,
        bulge=bulges,
        eyeballs=eyeballs,
        shapes=shapes,
        eye=eye,
        measured=measured,
    )


def _part(name, side, frame, surface, rest, anchors, blend) -> EyePart:
    points = frame.to_world(surface.points)
    normals = frame.directions(surface.normals)
    operator = affine_operator(rest[anchors], points)
    if blend.shape[0] == 1:
        blend = np.repeat(blend, len(points), axis=0)
    nearest = np.argmin(
        np.linalg.norm(points[:, None] - rest[anchors][None], axis=2), axis=1
    )
    return EyePart(
        name=name,
        side=side,
        rest=points,
        normals=normals,
        uvs=polar_to_uv(surface.polar, side),
        triangles=surface.triangles,
        anchors=np.asarray(anchors),
        operator=operator,
        blend=blend,
        source=np.asarray(anchors)[nearest],
    )


def eye_texture(
    eye,
    eyed: EyedMesh,
    model,
    baked: bytes | None = None,
    normal_map: bytes | None = None,
):
    """
    The eye texture (H, W, 3, 8-bit sRGB) of a file and what it was drawn from:
    ``"texture"`` (the card's own ``eye.base_color``, read by the caller), ``"baked"`` (the
    baked eye texture ``baked`` on anny's atlas, carried over and fitted, per side) or
    ``"procedural"``. With ``normal_map`` (the card's normal map on anny's atlas, which the
    bake fills on the eyeballs with the reference's eyes) a third value: the baked eyes'
    normals in the layout (:func:`opensculptboy.eyes.texture.carry_baked_normals`, moved
    with the iris), or None when the eyes are not baked.
    """
    from opensculptboy.eyes import texture as tex

    if (
        eyed.eye is not None and eye.wants_measure()
    ):  # the sizes the mesh was built with
        eye = dataclasses.replace(
            eye, iris_mm=eyed.eye.iris_mm, pupil_mm=eyed.eye.pupil_mm
        )
    eye = tex._sized(eye)
    use_baked = baked is not None and eye.source in ("auto", "baked")
    if eye.source == "baked" and baked is None:
        raise ValueError(
            "The eyes' source is 'baked', but the file has no eye texture."
        )
    radii = eyed.radii_mm()
    with_normals = normal_map is not None
    if not use_baked:
        pixels, info = tex.procedural_eye_pixels(eye, radii), {"source": "procedural"}
        return (pixels, info, None) if with_normals else (pixels, info)
    import io

    from PIL import Image

    from opensculptboy.texture.eyes import eye_uv_box

    image = tex.linear(np.asarray(Image.open(io.BytesIO(baked)).convert("RGB")))
    box = eye_uv_box(model)
    carried = tex.carry_baked(image, box, eyed.eyeballs)
    bent = None
    if with_normals:
        atlas = np.asarray(Image.open(io.BytesIO(normal_map)).convert("RGB"))
        bent = tex.carry_baked_normals(atlas / 255.0, eyed.eyeballs)
        fitted, found, bent = tex.fit_baked_iris(carried, eye, radii, normals=bent)
    else:
        fitted, found = tex.fit_baked_iris(carried, eye, radii)
    sides = {
        side: "procedural"
        if hit is None
        else "mirrored"
        if isinstance(hit, str)
        else "baked"
        for side, hit in found.items()
    }
    measured = {
        side: {
            "pupil_mm": round(float(2.0 * hit[1] * radii[side]), 2),
            "iris_mm": round(float(2.0 * hit[2] * radii[side]), 2),
        }
        for side, hit in found.items()
        if hit is not None and not isinstance(hit, str)
    }
    info = {"source": "baked", "sides": sides, "measured": measured}
    if with_normals:
        if all(v == "procedural" for v in sides.values()):
            bent = None
        return tex.srgb(fitted), info, bent
    return tex.srgb(fitted), info

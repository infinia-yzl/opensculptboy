# OpenSculptBoy
# Apache License, Version 2.0
"""
Texture bake: the reference's skin colour and surface normals carried onto anny's UV atlas.

For each texel of anny's atlas, :func:`bake` finds the point of the fitted mesh that the texel
covers (a rasterisation of the mesh's UV triangles), casts a ray along the mesh normal through
it, and takes the nearest hit on the reference's skin within ``max_distance``. The hit gives:

- ``base_color``: the reference's base colour texture at the hit's UVs (bilinear);
- ``normal``: the reference's shading normal at the hit (its interpolated vertex normals), in
  the tangent space of the fitted mesh at the texel, encoded as a glTF normal texture (+X along
  increasing u, +Y along increasing v of the OBJ convention, which is up in the image).

Texels of anny's eyeballs take the reference's eyes in the same way when the reference has them
(``Reference.eye_reference``, the parts of role ``"eyes"``), so that the iris comes along with
the skin; the eyes are also baked onto an eye texture of their own
(:mod:`opensculptboy.texture.eyes`), whose texels are about nine times finer (along each side)
than those of a 2048-texel atlas when it is 1024 texels wide. Texels that no ray reaches take
the value of the nearest texel that one did, so that the seams of the atlas do not show. The
rays use Embree through trimesh (``embreex``, the ``viewer`` extra), as :mod:`anny.viewer.bake`
does. The vertex colours of the reference are never read: the DD2DA tools keep the game's
masks in them (``_COLOR_0_GAME``), and the ``COLOR_0`` of a hair part holds the hair's albedo.

Left for later (TODO): the reference's own normal map is not yet carried (the bake uses the
geometry of the reference alone), the tangent frames are per triangle (MikkTSpace would match
engines exactly), and the other material maps (roughness, subsurface) are not baked.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class BakedTextures:
    """Baked textures of anny's atlas, row 0 at the top (as glTF and PNG files expect)."""

    base_color: np.ndarray | None  # (S, S, 3) uint8
    normal: np.ndarray  # (S, S, 3) uint8, tangent space
    coverage: np.ndarray  # (S, S) bool: texels that a ray reached
    covered_share: (
        float  # share of the atlas texels inside a UV triangle that a ray reached
    )
    # the base colour of the eyeballs on the eye texture (opensculptboy.texture.eyes), or None
    eyes: np.ndarray | None = None  # (S_eye / 2, S_eye, 3) uint8

    def png_files(self) -> dict[str, bytes]:
        """The textures as PNG files by slot (``"base_color"``, ``"normal"``, ``"eyes"``)."""
        from opensculptboy.export.document import png_bytes

        files = {"normal": png_bytes(self.normal)}
        if self.base_color is not None:
            files["base_color"] = png_bytes(self.base_color)
        if self.eyes is not None:
            files["eyes"] = png_bytes(self.eyes)
        return files


def rasterise(uv_triangles: np.ndarray, uvs: np.ndarray, size, chunk=200000):
    """
    The texels covered by UV triangles (OBJ convention, v = 0 at the bottom) on an atlas of
    ``size`` texels (one number for a square, or (width, height)): (rows, cols, triangle,
    barycentric coordinates (n, 3)); row 0 is the top of the image. A texel covered by two
    triangles keeps the first.
    """
    width, height = (size, size) if np.ndim(size) == 0 else size
    uv = np.asarray(uvs, dtype=np.float64)[np.asarray(uv_triangles)]  # (T, 3, 2)
    x = uv[..., 0] * width - 0.5  # texel centres at integers
    y = (1.0 - uv[..., 1]) * height - 0.5
    x0 = np.clip(np.floor(x.min(1)).astype(np.int64), 0, width - 1)
    x1 = np.clip(np.ceil(x.max(1)).astype(np.int64), 0, width - 1)
    y0 = np.clip(np.floor(y.min(1)).astype(np.int64), 0, height - 1)
    y1 = np.clip(np.ceil(y.max(1)).astype(np.int64), 0, height - 1)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    rows, cols, tris, barys = [], [], [], []
    order = np.arange(len(uv))
    for start in range(0, len(order), max(1, chunk // 16)):
        part = order[start : start + max(1, chunk // 16)]
        count = (w[part] * h[part]).astype(np.int64)
        t = np.repeat(part, count)
        local = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        px = x0[t] + local % w[t]
        py = y0[t] + local // w[t]
        ax, ay = x[t, 0], y[t, 0]
        bx, by = x[t, 1], y[t, 1]
        cx, cy = x[t, 2], y[t, 2]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        valid = np.abs(det) > 1e-12
        det = np.where(valid, det, 1.0)
        l0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
        l1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
        l2 = 1.0 - l0 - l1
        inside = valid & (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        rows.append(py[inside])
        cols.append(px[inside])
        tris.append(t[inside])
        barys.append(np.stack([l0, l1, l2], 1)[inside])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    tris, barys = np.concatenate(tris), np.concatenate(barys)
    _, first = np.unique(rows * width + cols, return_index=True)
    return rows[first], cols[first], tris[first], np.clip(barys[first], 0.0, 1.0)


def _tangent_frames(vertices, triangles, uv_triangles, uvs):
    """Per-triangle tangent (along u) and bitangent (along v) of a mesh, unit length."""
    p = vertices[triangles]
    t = uvs[uv_triangles]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    d1, d2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    det = np.where(np.abs(det) > 1e-14, det, 1e-14)
    tangent = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / det[:, None]
    bitangent = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / det[:, None]
    for a in (tangent, bitangent):
        a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-20)
    return tangent, bitangent


def _sample(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear samples (n, C) of an image (H, W, C) at UVs in the glTF convention (v = 0
    at the top), repeating outside [0, 1]."""
    h, w = image.shape[:2]
    x = (uv[:, 0] % 1.0) * w - 0.5
    y = (uv[:, 1] % 1.0) * h - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    img = image.astype(np.float64)

    def at(yy, xx):
        return img[yy % h, xx % w]

    top = at(y0, x0) * (1 - fx) + at(y0, x0 + 1) * fx
    bottom = at(y0 + 1, x0) * (1 - fx) + at(y0 + 1, x0 + 1) * fx
    return top * (1 - fy) + bottom * fy


def _fill(image: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Every unknown texel takes the value of the nearest known texel."""
    from scipy.ndimage import distance_transform_edt

    if known.all() or not known.any():
        return image
    _, (iy, ix) = distance_transform_edt(~known, return_indices=True)
    return image[iy, ix]


def _hits(points, normals, target, max_distance):
    """The nearest hit within ``max_distance`` of rays along ``normals`` through ``points``
    (both ways) on a target :class:`~opensculptboy.fit.reference.Reference` whose triangle
    faces the same way as the ray's normal: the rays that hit, their hit triangles and
    barycentric coordinates."""
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
    except ImportError as error:  # pragma: no cover - depends on the installed extras
        raise ImportError(
            "The texture bake needs Embree: install the viewer extra (embreex)."
        ) from error
    import trimesh

    empty = np.zeros(0, np.int64)
    if len(points) == 0:
        return empty, empty, np.zeros((0, 3))
    mesh = trimesh.Trimesh(target.vertices, target.triangles, process=False)
    rays = RayMeshIntersector(mesh)
    origins = points + normals * max_distance
    locations, ray_ids, hit_triangles = rays.intersects_location(
        origins, -normals, multiple_hits=True
    )
    gap = np.linalg.norm(locations - points[ray_ids], axis=1)
    # a hit counts when the target faces the same way there: on a thin part (an ear) the
    # nearest surface along the normal can be the other side
    facing = (target.face_normals[hit_triangles] * normals[ray_ids]).sum(1) > 0.0
    near = (gap <= max_distance) & facing
    locations, ray_ids, hit_triangles, gap = (
        locations[near],
        ray_ids[near],
        hit_triangles[near],
        gap[near],
    )
    order = np.lexsort((gap, ray_ids))
    ray_ids, keep = np.unique(ray_ids[order], return_index=True)
    hits = order[keep]
    locations, hit_triangles = locations[hits], hit_triangles[hits]
    hit_bary = trimesh.triangles.points_to_barycentric(
        target.vertices[target.triangles[hit_triangles]], locations
    )
    hit_bary = np.clip(hit_bary, 0.0, 1.0)
    hit_bary /= hit_bary.sum(1, keepdims=True)
    return ray_ids, hit_triangles, hit_bary


def _colours(target, hit_triangles, hit_bary):
    """Base colours (n, 3) at hits on a target's parts, and which of them are known."""
    colours = np.zeros((len(hit_triangles), 3))
    known = np.zeros(len(hit_triangles), bool)
    parts = list(getattr(target, "parts", []) or [])
    if not parts:
        return colours, known
    offsets = np.asarray(target.part_offsets)
    triangle_offsets = np.cumsum([0] + [len(p.triangles) for p in parts])
    part_of = np.searchsorted(triangle_offsets, hit_triangles, side="right") - 1
    corners = target.triangles[hit_triangles]
    for k, part in enumerate(parts):
        pick = part_of == k
        if part.base_color is None or part.uvs is None or not pick.any():
            continue
        local_corners = corners[pick] - offsets[k]
        uv = np.einsum("nk, nkd -> nd", hit_bary[pick], part.uvs[local_corners])
        colours[pick] = _sample(part.base_color, uv)[:, :3]
        known[pick] = True
    return colours, known


def bake(
    vertices: np.ndarray,
    triangles: np.ndarray,
    uv_triangles: np.ndarray,
    uvs: np.ndarray,
    reference,
    size: int = 1024,
    max_distance: float = 0.01,
    eye_triangles: np.ndarray | None = None,
    eye_box=None,
    eye_size: int = 1024,
) -> BakedTextures:
    """
    Bake the reference onto anny's atlas.

    Args:
        vertices, triangles: the fitted mesh, placed in the reference's frame (for example
            the detail-layer mesh of the fit, posed and placed as the fit left it).
        uv_triangles, uvs: its UV triangles and UVs (OBJ convention, v = 0 at the bottom),
            as :class:`~opensculptboy.detail.subdivision.DetailTopology` holds them.
        reference: the :class:`~opensculptboy.fit.reference.Reference`; its ``parts`` give
            the UVs and the base colour textures (none: the base colour is not baked).
        size: the side of the square textures, in texels.
        max_distance: rays reach this far on each side of the fitted surface (metres).
        eye_triangles: a mask (T,) of the fitted mesh's eyeball triangles, whose texels take
            the reference's eyes (``reference.eye_reference``) instead of its skin; without
            either, every texel takes the skin.
        eye_box: the box of the atlas that the eye texture covers
            (:func:`opensculptboy.texture.eyes.eye_uv_box`); with it, the eyeballs are also
            baked onto an eye texture of ``eye_size`` by ``eye_size / 2`` texels.
    """
    from opensculptboy.fit.reference import vertex_normals

    V = np.asarray(vertices, dtype=np.float64)
    T = np.asarray(triangles, dtype=np.int64)
    UT = np.asarray(uv_triangles, dtype=np.int64)
    UV = np.asarray(uvs, dtype=np.float64)
    rows, cols, tri, bary = rasterise(UT, UV, size)
    N = vertex_normals(V, T)
    points = np.einsum("nk, nkd -> nd", bary, V[T[tri]])
    normals = np.einsum("nk, nkd -> nd", bary, N[T[tri]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    eye_reference = getattr(reference, "eye_reference", None)
    on_eye = np.zeros(len(tri), bool)
    if eye_triangles is not None and eye_reference is not None:
        on_eye = np.asarray(eye_triangles, bool)[tri]
    groups = []
    for target, pick in ((reference, ~on_eye), (eye_reference, on_eye)):
        if target is None or not pick.any():
            continue
        texels = np.nonzero(pick)[0]
        ids, hit_triangles, hit_bary = _hits(
            points[texels], normals[texels], target, max_distance
        )
        groups.append((target, texels[ids], hit_triangles, hit_bary))

    coverage = np.zeros((size, size), bool)
    normal_image = np.zeros((size, size, 3))
    normal_image[..., 2] = 1.0
    colour_image = np.zeros((size, size, 3))
    colour_known = np.zeros((size, size), bool)
    any_colour = False
    tangent, bitangent = _tangent_frames(V, T, UT, UV)
    for target, ray_ids, hit_triangles, hit_bary in groups:
        # the normal: the target's shading normal in the fitted mesh's tangent space
        corners = target.triangles[hit_triangles]
        target_normals = np.einsum("nk, nkd -> nd", hit_bary, target.normals[corners])
        target_normals /= np.maximum(
            np.linalg.norm(target_normals, axis=1, keepdims=True), 1e-12
        )
        n = normals[ray_ids]
        t = tangent[tri[ray_ids]]
        t = t - (t * n).sum(1, keepdims=True) * n
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        b = np.cross(n, t)
        b *= np.sign((b * bitangent[tri[ray_ids]]).sum(1, keepdims=True) + 1e-20)
        local = np.stack(
            [
                (target_normals * t).sum(1),
                (target_normals * b).sum(1),
                (target_normals * n).sum(1),
            ],
            1,
        )
        local[:, 2] = np.maximum(local[:, 2], 0.0)
        local /= np.maximum(np.linalg.norm(local, axis=1, keepdims=True), 1e-12)
        coverage[rows[ray_ids], cols[ray_ids]] = True
        normal_image[rows[ray_ids], cols[ray_ids]] = local
        colours, known = _colours(target, hit_triangles, hit_bary)
        any_colour |= bool(getattr(target, "parts", None))
        colour_image[rows[ray_ids[known]], cols[ray_ids[known]]] = colours[known]
        colour_known[rows[ray_ids[known]], cols[ray_ids[known]]] = True
    normal_image = _fill(normal_image, coverage)
    normal_png = np.clip(np.round((normal_image * 0.5 + 0.5) * 255), 0, 255)

    base_png = None
    parts = list(getattr(reference, "parts", []) or [])
    if any_colour and any(
        p.base_color is not None and p.uvs is not None for p in parts
    ):
        colour_image = _fill(colour_image, colour_known)
        base_png = np.clip(np.round(colour_image), 0, 255).astype(np.uint8)

    eyes = None
    if (
        eye_box is not None
        and eye_triangles is not None
        and eye_reference is not None
        and eye_size > 0
    ):
        eyes = _bake_eyes(
            V,
            T,
            UT,
            UV,
            N,
            eye_triangles,
            eye_reference,
            eye_box,
            eye_size,
            max_distance,
        )

    reached = sum(len(g[1]) for g in groups)
    return BakedTextures(
        base_color=base_png,
        normal=normal_png.astype(np.uint8),
        coverage=coverage,
        covered_share=float(reached / max(len(rows), 1)),
        eyes=eyes,
    )


def _bake_eyes(V, T, UT, UV, N, eye_triangles, target, box, size, max_distance):
    """The base colour (size / 2, size, 3) of the eye texture: the eyeball triangles laid out
    by the box of the atlas, each texel taking the reference's eyes along the normal."""
    from opensculptboy.texture.eyes import EYE_TEXTURE_ASPECT, to_eye_uvs

    pick = np.nonzero(np.asarray(eye_triangles, bool))[0]
    width, height = int(size), int(size) // EYE_TEXTURE_ASPECT
    rows, cols, tri, bary = rasterise(UT[pick], to_eye_uvs(UV, box), (width, height))
    tri = pick[tri]
    points = np.einsum("nk, nkd -> nd", bary, V[T[tri]])
    normals = np.einsum("nk, nkd -> nd", bary, N[T[tri]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    ids, hit_triangles, hit_bary = _hits(points, normals, target, max_distance)
    colours, known = _colours(target, hit_triangles, hit_bary)
    image = np.zeros((height, width, 3))
    reached = np.zeros((height, width), bool)
    image[rows[ids[known]], cols[ids[known]]] = colours[known]
    reached[rows[ids[known]], cols[ids[known]]] = True
    if not reached.any():
        return None
    return np.clip(np.round(_fill(image, reached)), 0, 255).astype(np.uint8)


def bake_fit(result, subdiv: int | None = None, size: int = 1024, **kwargs):
    """
    Bake the reference of a :class:`~opensculptboy.fit.register.FitResult` onto the mesh of a
    level of its detail layer (its finest by default), posed and placed as the fit left it.
    """
    import torch

    from opensculptboy.detail import DetailModel

    subdiv = max(result.offsets) if subdiv is None else subdiv
    character = result.character
    model = character.build_model(dtype=torch.float64)
    detailed = DetailModel(
        model, subdiv, result.offsets[subdiv], "full", character.topology
    )
    with torch.no_grad():
        out = detailed(
            pose_parameters=character.pose_parameters(model), **character.model_kwargs()
        )
    vertices = out["vertices"][0].numpy() @ result.rotation.T + result.translation
    topology = detailed.topology
    if "eye_triangles" not in kwargs:
        kwargs["eye_triangles"] = eyeball_triangles(model, topology)
    if "eye_box" not in kwargs:
        from opensculptboy.texture.eyes import eye_uv_box

        kwargs["eye_box"] = eye_uv_box(model)
    return bake(
        vertices,
        topology.triangles,
        topology.uv_triangles,
        topology.uvs,
        result.reference,
        size,
        **kwargs,
    )


def eyeball_triangles(model, topology) -> np.ndarray | None:
    """The mask of the eyeball triangles of a detail-layer mesh, or None when the model's
    mesh has no eyeballs."""
    from opensculptboy.export.vrm_tables import eyeball_vertices

    try:
        rows = np.concatenate(list(eyeball_vertices(model).values()))
    except ValueError:
        return None
    eye = np.zeros(len(model.template_vertices))
    eye[rows] = 1.0
    fine = np.asarray(topology.apply(eye)) > 0.5
    return fine[np.asarray(topology.triangles)].all(1)

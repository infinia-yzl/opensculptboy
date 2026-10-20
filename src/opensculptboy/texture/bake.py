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

Texels that no ray reaches take the value of the nearest texel that one did, so that the seams
of the atlas do not show. The rays use Embree through trimesh (``embreex``, the ``viewer``
extra), as :mod:`anny.viewer.bake` does.

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

    def png_files(self) -> dict[str, bytes]:
        """The textures as PNG files by slot (``"base_color"``, ``"normal"``)."""
        from opensculptboy.export.document import png_bytes

        files = {"normal": png_bytes(self.normal)}
        if self.base_color is not None:
            files["base_color"] = png_bytes(self.base_color)
        return files


def rasterise(uv_triangles: np.ndarray, uvs: np.ndarray, size: int, chunk=200000):
    """
    The texels covered by UV triangles (OBJ convention, v = 0 at the bottom) on a square atlas
    of ``size`` texels: (rows, cols, triangle, barycentric coordinates (n, 3)); row 0 is the
    top of the image. A texel covered by two triangles keeps the first.
    """
    uv = np.asarray(uvs, dtype=np.float64)[np.asarray(uv_triangles)]  # (T, 3, 2)
    x = uv[..., 0] * size - 0.5  # texel centres at integers
    y = (1.0 - uv[..., 1]) * size - 0.5
    x0 = np.clip(np.floor(x.min(1)).astype(np.int64), 0, size - 1)
    x1 = np.clip(np.ceil(x.max(1)).astype(np.int64), 0, size - 1)
    y0 = np.clip(np.floor(y.min(1)).astype(np.int64), 0, size - 1)
    y1 = np.clip(np.ceil(y.max(1)).astype(np.int64), 0, size - 1)
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
    _, first = np.unique(rows * size + cols, return_index=True)
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


def bake(
    vertices: np.ndarray,
    triangles: np.ndarray,
    uv_triangles: np.ndarray,
    uvs: np.ndarray,
    reference,
    size: int = 1024,
    max_distance: float = 0.01,
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
    """
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
    except ImportError as error:  # pragma: no cover - depends on the installed extras
        raise ImportError(
            "The texture bake needs Embree: install the viewer extra (embreex)."
        ) from error
    import trimesh

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

    target = trimesh.Trimesh(reference.vertices, reference.triangles, process=False)
    rays = RayMeshIntersector(target)
    origins = points + normals * max_distance
    locations, ray_ids, hit_triangles = rays.intersects_location(
        origins, -normals, multiple_hits=True
    )
    gap = np.linalg.norm(locations - points[ray_ids], axis=1)
    near = gap <= max_distance
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
        reference.vertices[reference.triangles[hit_triangles]], locations
    )
    hit_bary = np.clip(hit_bary, 0.0, 1.0)
    hit_bary /= hit_bary.sum(1, keepdims=True)

    # the normal: the reference's shading normal in the fitted mesh's tangent space
    corners = reference.triangles[hit_triangles]
    target_normals = np.einsum("nk, nkd -> nd", hit_bary, reference.normals[corners])
    target_normals /= np.maximum(
        np.linalg.norm(target_normals, axis=1, keepdims=True), 1e-12
    )
    tangent, bitangent = _tangent_frames(V, T, UT, UV)
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

    coverage = np.zeros((size, size), bool)
    coverage[rows[ray_ids], cols[ray_ids]] = True
    normal_image = np.zeros((size, size, 3))
    normal_image[..., 2] = 1.0
    normal_image[rows[ray_ids], cols[ray_ids]] = local
    normal_image = _fill(normal_image, coverage)
    normal_png = np.clip(np.round((normal_image * 0.5 + 0.5) * 255), 0, 255)

    base_png = None
    parts = list(getattr(reference, "parts", []) or [])
    if parts and any(p.base_color is not None and p.uvs is not None for p in parts):
        offsets = np.asarray(reference.part_offsets)
        triangle_offsets = np.cumsum([0] + [len(p.triangles) for p in parts])
        part_of = np.searchsorted(triangle_offsets, hit_triangles, side="right") - 1
        colours = np.zeros((len(ray_ids), 3))
        known = np.zeros(len(ray_ids), bool)
        for k, part in enumerate(parts):
            pick = part_of == k
            if part.base_color is None or part.uvs is None or not pick.any():
                continue
            local_corners = corners[pick] - offsets[k]
            uv = np.einsum("nk, nkd -> nd", hit_bary[pick], part.uvs[local_corners])
            colours[pick] = _sample(part.base_color, uv)[:, :3]
            known[pick] = True
        colour_image = np.zeros((size, size, 3))
        colour_known = np.zeros((size, size), bool)
        colour_image[rows[ray_ids[known]], cols[ray_ids[known]]] = colours[known]
        colour_known[rows[ray_ids[known]], cols[ray_ids[known]]] = True
        colour_image = _fill(colour_image, colour_known)
        base_png = np.clip(np.round(colour_image), 0, 255).astype(np.uint8)

    return BakedTextures(
        base_color=base_png,
        normal=normal_png.astype(np.uint8),
        coverage=coverage,
        covered_share=float(len(ray_ids) / max(len(rows), 1)),
    )


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
    return bake(
        vertices,
        topology.triangles,
        topology.uv_triangles,
        topology.uvs,
        result.reference,
        size,
        **kwargs,
    )

# OpenSculptBoy
# Apache License, Version 2.0
"""
Texture bake: the reference's skin carried onto anny's UV atlas.

For each texel of anny's atlas, :func:`bake` finds the point of the fitted mesh that the texel
covers (a rasterisation of the mesh's UV triangles), casts a ray along the mesh normal through
it, and takes the nearest hit on the reference's skin within ``max_distance``, on a triangle
that faces the same way. The hit gives:

- ``base_color``: the reference's base colour texture at the hit's UVs (bilinear);
- ``normal``: the reference's shading normal at the hit, in the tangent space of the fitted
  mesh at the texel, encoded as a glTF normal texture. The shading normal is the reference's
  own: its file normals (glTF ``NORMAL``, which a game's artists smooth and shape, so that a
  surface reads differently from its bare geometry) with its normal map applied in its
  MikkTSpace frames, as Blender and glTF viewers apply it. The tangent space of the fitted
  mesh is MikkTSpace (:mod:`opensculptboy.texture.tangents`) on the mesh split at UV seams,
  with the normals of the welded mesh: the frames that the exporters write as ``TANGENT``
  and that Blender computes, so that the map reads back as it was baked;
- ``roughness``: the G channel of the reference's metallic-roughness texture times its
  roughness factor, as a grey texture.

From the fitted mesh alone, the bake also writes two maps of real-time skin shading, as
per-vertex values interpolated over the texels: ``thickness``, the mean distance through the
mesh along seven rays into it (the normal and six rays 30 degrees around it; misses count
as ``thickness_mm``), over ``thickness_mm``; and ``curvature``, the mean curvature (the
mean of ``(n_i - n_j) . (p_i - p_j) / |p_i - p_j|^2`` over a vertex's edges, 1 / radius,
positive where convex) as ``0.5 + 0.5 * k / curvature_per_m``.

Texels of anny's eyeballs take the reference's eyes in the same way when the reference has them
(``Reference.eye_reference``, the parts of role ``"eyes"``); the eyes are also baked onto an eye
texture of their own (:mod:`opensculptboy.texture.eyes`), and the head's UV island onto the
head tile (:mod:`opensculptboy.texture.tiles`, base colour, normal and roughness), whose
texels are finer than the atlas's. Texels that no ray reaches take the value of the nearest
texel that one did, so that the seams of the atlas do not show. The rays use Embree through
trimesh (``embreex``, the ``viewer`` extra), as :mod:`anny.viewer.bake` does. The vertex
colours of the reference are never read: a source may keep other data (masks) in them,
and the ``COLOR_0`` of a hair part holds the hair's albedo.
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
    roughness: np.ndarray | None = None  # (S, S) uint8, roughness * 255
    thickness: np.ndarray | None = None  # (S, S) uint8, thickness / thickness_mm * 255
    curvature: np.ndarray | None = None  # (S, S) uint8, (0.5 + 0.5 k / k_max) * 255
    thickness_mm: float = 30.0
    curvature_per_m: float = 100.0
    # the head tile (opensculptboy.texture.tiles) by slot: base_color, normal, roughness
    head: dict = dataclasses.field(default_factory=dict)
    # the ear tile (opensculptboy.texture.ears) by slot: base_color, normal, roughness,
    # thickness, curvature; and its UVs
    ears: dict = dataclasses.field(default_factory=dict)
    ear_uvs: object = None  # opensculptboy.texture.ears.EarUVs

    def png_files(self) -> dict[str, bytes]:
        """The textures as PNG files by slot (``"base_color"``, ``"normal"``, ``"eyes"``,
        ``"roughness"``, ``"thickness"``, ``"curvature"``, ``"head_base_color"``, ...)."""
        from opensculptboy.export.document import png_bytes

        files = {"normal": png_bytes(self.normal)}
        for slot in ("base_color", "eyes", "roughness", "thickness", "curvature"):
            image = getattr(self, slot)
            if image is not None:
                files[slot] = png_bytes(image)
        for slot, image in self.head.items():
            files[f"head_{slot}"] = png_bytes(image)
        for slot, image in self.ears.items():
            files[f"ears_{slot}"] = png_bytes(image)
        return files

    def material(self, names: dict[str, str]) -> dict:
        """The ``material`` section of a card for the files ``names`` (slot -> path, as
        :meth:`png_files` names the slots, and ``"ear_uvs"`` the ear tile's UV file), without
        the eye texture, which the detail layer names."""
        material = {}
        for slot in ("base_color", "normal", "roughness", "thickness", "curvature"):
            if slot in names:
                material[slot] = names[slot]
        if "thickness" in names:
            material["thickness_mm"] = self.thickness_mm
        if "curvature" in names:
            material["curvature_per_m"] = self.curvature_per_m
        head = {
            slot: names[f"head_{slot}"]
            for slot in ("base_color", "normal", "roughness")
            if f"head_{slot}" in names
        }
        if head:
            material["head"] = head
        ears = {
            slot: names[f"ears_{slot}"]
            for slot in ("base_color", "normal", "roughness", "thickness", "curvature")
            if f"ears_{slot}" in names
        }
        if ears and "ear_uvs" in names:
            material["ears"] = {"uvs": names["ear_uvs"], **ears}
        return material


def rasterise(
    uv_triangles: np.ndarray, uvs: np.ndarray, size, chunk=200000, shift=(0.0, 0.0)
):
    """
    The texels covered by UV triangles (OBJ convention, v = 0 at the bottom) on an atlas of
    ``size`` texels (one number for a square, or (width, height)): (rows, cols, triangle,
    barycentric coordinates (n, 3)); row 0 is the top of the image. A texel covered by two
    triangles keeps the first. ``shift`` moves the sample point of every texel from its
    centre by a share of a texel (x right, y down), for supersampling.
    """
    width, height = (size, size) if np.ndim(size) == 0 else size
    uv = np.asarray(uvs, dtype=np.float64)[np.asarray(uv_triangles)]  # (T, 3, 2)
    x = uv[..., 0] * width - 0.5 - shift[0]  # sample points at integers
    y = (1.0 - uv[..., 1]) * height - 0.5 - shift[1]
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
        rows.append(py[inside].astype(np.int32))
        cols.append(px[inside].astype(np.int32))
        tris.append(t[inside].astype(np.int32))
        barys.append(np.stack([l0, l1, l2], 1)[inside].astype(np.float32))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    tris, barys = np.concatenate(tris), np.concatenate(barys)
    _, first = np.unique(rows.astype(np.int64) * width + cols, return_index=True)
    return (
        rows[first].astype(np.int64),
        cols[first].astype(np.int64),
        tris[first].astype(np.int64),
        np.clip(barys[first].astype(np.float64), 0.0, 1.0),
    )


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _sample(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear samples (n, C) of an image (H, W) or (H, W, C) at UVs in the glTF
    convention (v = 0 at the top), repeating outside [0, 1]."""
    if image.ndim == 2:
        return _sample(image[..., None], uv)
    h, w = image.shape[:2]
    x = (uv[:, 0] % 1.0) * w - 0.5
    y = (uv[:, 1] % 1.0) * h - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]

    def at(yy, xx):
        return image[yy % h, xx % w].astype(np.float64)

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


def _intersector(vertices, triangles):
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
    except ImportError as error:  # pragma: no cover - depends on the installed extras
        raise ImportError(
            "The texture bake needs Embree: install the viewer extra (embreex)."
        ) from error
    import trimesh

    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    return RayMeshIntersector(mesh)


def _hits(points, normals, target, max_distance, rays=None):
    """The nearest hit within ``max_distance`` of rays along ``normals`` through ``points``
    (both ways) on a target :class:`~opensculptboy.fit.reference.Reference` whose triangle
    faces the same way as the ray's normal: the rays that hit, their hit triangles and
    barycentric coordinates."""
    import trimesh

    empty = np.zeros(0, np.int64)
    if len(points) == 0:
        return empty, empty, np.zeros((0, 3))
    if rays is None:
        rays = _intersector(target.vertices, target.triangles)
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


def _part_index(target, hit_triangles):
    parts = list(getattr(target, "parts", []) or [])
    triangle_offsets = np.cumsum([0] + [len(p.triangles) for p in parts])
    return parts, np.searchsorted(triangle_offsets, hit_triangles, side="right") - 1


def _parts_aligned(target) -> bool:
    """Whether the target's vertices are its parts' vertices, in order (the parts' own
    normals and tangents then apply to it as they are)."""
    parts = list(getattr(target, "parts", []) or [])
    offsets = getattr(target, "part_offsets", None)
    if not parts or offsets is None or int(offsets[-1]) != len(target.vertices):
        return False
    return all(
        np.allclose(target.vertices[offsets[k] : offsets[k] + 4], p.vertices[:4])
        for k, p in enumerate(parts)
    )


def target_shading(target) -> tuple[np.ndarray, np.ndarray | None]:
    """
    The shading frames of a target's vertices: unit normals (V, 3), the parts' own file
    normals where they have them, and MikkTSpace tangents (V, 4) of the parts that have a
    normal map (zero elsewhere; None when no part has one). Cached on the target.
    """
    cached = getattr(target, "_shading", None)
    if cached is not None:
        return cached
    from opensculptboy.texture.tangents import mikktspace

    normals = np.array(target.normals, dtype=np.float64)
    tangents = None
    if _parts_aligned(target):
        offsets = target.part_offsets
        for k, part in enumerate(target.parts):
            rows = slice(int(offsets[k]), int(offsets[k + 1]))
            if getattr(part, "normals", None) is not None:
                normals[rows] = part.normals
            if part.normal_map is not None and part.uvs is not None:
                if tangents is None:
                    tangents = np.zeros((len(target.vertices), 4))
                uv = np.stack([part.uvs[:, 0], 1.0 - part.uvs[:, 1]], 1)  # OBJ
                tangents[rows] = mikktspace(
                    part.vertices, normals[rows], uv, part.triangles
                )
    target._shading = (normals, tangents)
    return target._shading


def _shading_normals(target, hit_triangles, hit_bary):
    """The target's shading normals (n, 3) at hits: its vertex normals, bent by the normal
    map of each part that has one."""
    normals, tangents = target_shading(target)
    corners = target.triangles[hit_triangles]
    n = _unit(np.einsum("nk, nkd -> nd", hit_bary, normals[corners]))
    if tangents is None:
        return n
    parts, part_of = _part_index(target, hit_triangles)
    offsets = np.asarray(target.part_offsets)
    out = n.copy()
    for k, part in enumerate(parts):
        pick = part_of == k
        if part.normal_map is None or part.uvs is None or not pick.any():
            continue
        local = corners[pick] - offsets[k]
        uv = np.einsum("nk, nkd -> nd", hit_bary[pick], part.uvs[local])
        c = _sample(part.normal_map, uv)[:, :3] / 127.5 - 1.0
        c[:, :2] *= float(getattr(part, "normal_scale", 1.0))
        c = _unit(c)
        frame = tangents[corners[pick]]
        t = np.einsum("nk, nkd -> nd", hit_bary[pick], frame[..., :3])
        w = np.where(
            np.einsum("nk, nk -> n", hit_bary[pick], frame[..., 3]) < 0, -1.0, 1.0
        )
        nn = n[pick]
        t = _unit(t - (t * nn).sum(1, keepdims=True) * nn)
        b = w[:, None] * np.cross(nn, t)
        out[pick] = _unit(c[:, :1] * t + c[:, 1:2] * b + c[:, 2:3] * nn)
    return out


def _colours(target, hit_triangles, hit_bary):
    """Base colours (n, 3) at hits on a target's parts, and which of them are known."""
    colours = np.zeros((len(hit_triangles), 3))
    known = np.zeros(len(hit_triangles), bool)
    parts, part_of = _part_index(target, hit_triangles)
    if not parts:
        return colours, known
    offsets = np.asarray(target.part_offsets)
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


def _roughness(target, hit_triangles, hit_bary):
    """Roughness values (n,) at hits on a target's parts, and which of them are known."""
    values = np.zeros(len(hit_triangles))
    known = np.zeros(len(hit_triangles), bool)
    parts, part_of = _part_index(target, hit_triangles)
    if not parts:
        return values, known
    offsets = np.asarray(target.part_offsets)
    corners = target.triangles[hit_triangles]
    for k, part in enumerate(parts):
        pick = part_of == k
        rough = getattr(part, "roughness", None)
        if rough is None or not pick.any():
            continue
        if np.ndim(rough) == 0:
            values[pick] = float(rough)
        elif part.uvs is not None:
            local = corners[pick] - offsets[k]
            uv = np.einsum("nk, nkd -> nd", hit_bary[pick], part.uvs[local])
            values[pick] = _sample(np.asarray(rough), uv)[:, 0]
        else:
            continue
        known[pick] = True
    return values, known


@dataclasses.dataclass
class _Surface:
    """The fitted mesh with its welded normals and the MikkTSpace frames of its vertices
    split at UV seams, as the exporters write them."""

    vertices: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray
    split: np.ndarray  # (T, 3) split vertex of each corner
    tangents: np.ndarray  # (S, 4)

    @classmethod
    def of(cls, vertices, triangles, uv_triangles, uvs):
        from opensculptboy.fit.reference import vertex_normals
        from opensculptboy.texture.tangents import mikktspace, split_mesh

        N = vertex_normals(vertices, triangles)
        source, uv_source, split = split_mesh(triangles, uv_triangles)
        tangents = mikktspace(vertices[source], N[source], uvs[uv_source], split)
        return cls(vertices, triangles, N, split, tangents)

    def at(self, tri, bary):
        """Points, normals, tangents and bitangents (n, 3) of texels."""
        corners = self.triangles[tri]
        points = np.einsum("nk, nkd -> nd", bary, self.vertices[corners])
        n = _unit(np.einsum("nk, nkd -> nd", bary, self.normals[corners]))
        frame = self.tangents[self.split[tri]]
        t = np.einsum("nk, nkd -> nd", bary, frame[..., :3])
        w = np.where(np.einsum("nk, nk -> n", bary, frame[..., 3]) < 0, -1.0, 1.0)
        t = _unit(t - (t * n).sum(1, keepdims=True) * n)
        b = w[:, None] * np.cross(n, t)
        return points, n, t, b


def _accumulate(image, count, rows, cols, values) -> None:
    """Add values (n,) or (n, C) at texels into an image (H, W) or (H, W, C) and count them."""
    h, w = count.shape
    flat = rows * w + cols
    count += np.bincount(flat, minlength=h * w).reshape(h, w)
    if image.ndim == 2:
        image += np.bincount(flat, values, h * w).reshape(h, w)
    else:
        for k in range(image.shape[2]):
            image[..., k] += np.bincount(flat, values[:, k], h * w).reshape(h, w)


def _bake_maps(
    surface,
    uv_triangles,
    uvs,
    size,
    groups,
    max_distance,
    pick=None,
    chunk=1 << 20,
    supersample: int = 1,
):
    """
    Base colour, normal and roughness images of a texture of ``size`` (one number, or
    (width, height)) over the UV triangles ``pick`` (all by default). ``groups`` lists
    (target, triangle mask): each texel takes the target whose mask holds its triangle.
    ``supersample`` casts ``supersample`` by ``supersample`` rays per texel, spread over the
    texel, and averages what they hit, so that edges of colour and of the islands do not
    stair-step. Returns the images (H, W, 3) floats, (H, W) floats, and their known masks,
    the count of texels inside a triangle, and the texels of the central samples.
    """
    width, height = (size, size) if np.ndim(size) == 0 else size
    index = np.arange(len(uv_triangles)) if pick is None else np.nonzero(pick)[0]
    colour = np.zeros((height, width, 3))
    colour_count = np.zeros((height, width))
    normal = np.zeros((height, width, 3))
    normal_count = np.zeros((height, width))
    rough = np.zeros((height, width))
    rough_count = np.zeros((height, width))
    inside = np.zeros((height, width), bool)
    k = max(int(supersample), 1)
    offsets = [
        ((i + 0.5) / k - 0.5, (j + 0.5) / k - 0.5) for j in range(k) for i in range(k)
    ]
    centre = None
    rays = {}
    for shift in offsets:
        rows, cols, tri, bary = rasterise(
            uv_triangles[index], uvs, (width, height), shift=shift
        )
        tri = index[tri]
        inside[rows, cols] = True
        if centre is None or shift == (0.0, 0.0):
            centre = (rows, cols, tri, bary)
        for g, (target, mask) in enumerate(groups):
            if target is None:
                continue
            if g not in rays:
                rays[g] = _intersector(target.vertices, target.triangles)
            texels = (
                np.nonzero(mask[tri])[0] if mask is not None else np.arange(len(tri))
            )
            for start in range(0, len(texels), chunk):
                sel = texels[start : start + chunk]
                points, n, t, b = surface.at(tri[sel], bary[sel])
                ids, hit_tri, hit_bary = _hits(points, n, target, max_distance, rays[g])
                r, c = rows[sel[ids]], cols[sel[ids]]
                shade = _shading_normals(target, hit_tri, hit_bary)
                local = np.stack(
                    [
                        (shade * t[ids]).sum(1),
                        (shade * b[ids]).sum(1),
                        (shade * n[ids]).sum(1),
                    ],
                    1,
                )
                local[:, 2] = np.maximum(local[:, 2], 0.0)
                _accumulate(normal, normal_count, r, c, _unit(local))
                values, known = _colours(target, hit_tri, hit_bary)
                _accumulate(colour, colour_count, r[known], c[known], values[known])
                values, known = _roughness(target, hit_tri, hit_bary)
                _accumulate(rough, rough_count, r[known], c[known], values[known])
    reached = normal_count > 0
    normal[~reached] = (0.0, 0.0, 1.0)
    normal = _unit(normal)
    colour /= np.maximum(colour_count, 1.0)[..., None]
    rough /= np.maximum(rough_count, 1.0)
    return dict(
        colour=colour,
        colour_known=colour_count > 0,
        normal=normal,
        reached=reached,
        roughness=rough,
        roughness_known=rough_count > 0,
        inside=int(inside.sum()),
        texels=centre,
    )


def _to_png(maps, fill_colour: bool):
    normal = _fill(maps["normal"], maps["reached"])
    normal_png = np.clip(np.round((normal * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
    colour = None
    if fill_colour and maps["colour_known"].any():
        colour = np.clip(
            np.round(_fill(maps["colour"], maps["colour_known"])), 0, 255
        ).astype(np.uint8)
    rough = None
    if maps["roughness_known"].any():
        rough = np.clip(
            np.round(_fill(maps["roughness"], maps["roughness_known"]) * 255), 0, 255
        ).astype(np.uint8)
    return colour, normal_png, rough


def vertex_curvature(
    vertices: np.ndarray, triangles: np.ndarray, smoothing: int = 4
) -> np.ndarray:
    """The mean curvature (V,) of a mesh's vertices, 1 / metres, positive where convex: the
    mean over each vertex's edges of (n_i - n_j) . (p_i - p_j) / |p_i - p_j|^2, then
    averaged with the neighbours ``smoothing`` times (a fine mesh's curvature carries the
    ripples of its last refinement, which a lookup of skin shading should not)."""
    from opensculptboy.fit.reference import vertex_normals

    V = np.asarray(vertices, np.float64)
    N = vertex_normals(V, triangles)
    T = np.asarray(triangles, np.int64)
    edges = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    edges = np.unique(np.sort(edges, 1), axis=0)
    i, j = edges[:, 0], edges[:, 1]
    d = V[i] - V[j]
    k = ((N[i] - N[j]) * d).sum(1) / np.maximum((d * d).sum(1), 1e-18)
    total = np.bincount(i, k, len(V)) + np.bincount(j, k, len(V))
    count = np.bincount(i, minlength=len(V)) + np.bincount(j, minlength=len(V))
    curvature = total / np.maximum(count, 1)
    for _ in range(smoothing):
        spread = np.bincount(i, curvature[j], len(V)) + np.bincount(
            j, curvature[i], len(V)
        )
        curvature = 0.5 * curvature + 0.5 * spread / np.maximum(count, 1)
    return curvature


def vertex_thickness(
    vertices: np.ndarray,
    triangles: np.ndarray,
    max_thickness: float = 0.03,
    exclude: np.ndarray | None = None,
    cone_degrees: float = 30.0,
    rays: int = 6,
) -> np.ndarray:
    """
    The thickness (V,) of a closed mesh at its vertices, metres: the mean distance to the
    first surface along the inward normal and ``rays`` rays at ``cone_degrees`` around it,
    each capped at ``max_thickness`` (a miss counts as the cap). ``exclude`` masks the
    triangles that the rays pass through (the eyeballs, which do not stop light in skin).
    """
    from opensculptboy.fit.reference import vertex_normals

    V = np.asarray(vertices, np.float64)
    T = np.asarray(triangles, np.int64)
    N = vertex_normals(V, T)
    keep = np.ones(len(T), bool) if exclude is None else ~np.asarray(exclude, bool)
    intersector = _intersector(V, T[keep])
    a = np.abs(N).argmin(1)
    helper = np.eye(3)[a]
    u = _unit(np.cross(N, helper))
    v = np.cross(N, u)
    directions = [-N]
    c, s = np.cos(np.radians(cone_degrees)), np.sin(np.radians(cone_degrees))
    for k in range(rays):
        phi = 2.0 * np.pi * k / rays
        directions.append(-N * c + s * (np.cos(phi) * u + np.sin(phi) * v))
    total = np.zeros(len(V))
    eps = 1e-5
    for d in directions:
        origins = V + d * eps
        hit, ray_ids, _ = intersector.intersects_location(
            origins, d, multiple_hits=False
        )
        distance = np.full(len(V), max_thickness)
        gap = np.linalg.norm(hit - origins[ray_ids], axis=1)
        distance[ray_ids] = np.minimum(gap, max_thickness)
        total += distance
    return total / len(directions)


def _vertex_map(values, rows, cols, tri, bary, triangles, size):
    image = np.zeros((size, size))
    known = np.zeros((size, size), bool)
    image[rows, cols] = np.einsum("nk, nk -> n", bary, values[triangles[tri]])
    known[rows, cols] = True
    return _fill(image, known)


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
    head_triangles: np.ndarray | None = None,
    head_box=None,
    head_size: int = 0,
    skin_maps: bool = False,
    thickness_mm: float = 30.0,
    curvature_per_m: float = 100.0,
    ear_mesh=None,
    supersample: int = 2,
) -> BakedTextures:
    """
    Bake the reference onto anny's atlas.

    Args:
        vertices, triangles: the fitted mesh, placed in the reference's frame (for example
            the detail-layer mesh of the fit, posed and placed as the fit left it).
        uv_triangles, uvs: its UV triangles and UVs (OBJ convention, v = 0 at the bottom),
            as :class:`~opensculptboy.detail.subdivision.DetailTopology` holds them.
        reference: the :class:`~opensculptboy.fit.reference.Reference`; its ``parts`` give
            the UVs, the base colour, normal and roughness textures and the file normals
            (none: the base colour and the roughness are not baked, and the normals are the
            reference's geometric normals).
        size: the side of the square atlas textures, in texels.
        max_distance: rays reach this far on each side of the fitted surface (metres).
        eye_triangles: a mask (T,) of the fitted mesh's eyeball triangles, whose texels take
            the reference's eyes (``reference.eye_reference``) instead of its skin; without
            either, every texel takes the skin.
        eye_box: the box of the atlas that the eye texture covers
            (:func:`opensculptboy.texture.eyes.eye_uv_box`); with it, the eyeballs are also
            baked onto an eye texture of ``eye_size`` by ``eye_size / 2`` texels.
        head_triangles, head_box, head_size: the triangles of the head tile
            (:func:`opensculptboy.texture.tiles.on_head_tile`), its box of the atlas
            (:func:`~opensculptboy.texture.tiles.head_uv_box`) and its height in texels
            (0: no head tile); its width follows the box's aspect.
        skin_maps: also bake the thickness and curvature maps of the fitted mesh, with the
            scales ``thickness_mm`` and ``curvature_per_m``.
        ear_mesh: (UV triangles, UVs, triangle mask) of the mesh with its outer ears on the
            ear tile (:func:`opensculptboy.texture.ears.with_ear_uvs`); with it, the ears are
            baked onto the ear tile (:data:`~opensculptboy.texture.ears.EAR_TILE_SIZE`) in the
            tangent frames of those UVs.
        supersample: rays per texel along each side on the head tile (the ear tile takes one
            more); the atlas takes one.
    """
    V = np.asarray(vertices, dtype=np.float64)
    T = np.asarray(triangles, dtype=np.int64)
    UT = np.asarray(uv_triangles, dtype=np.int64)
    UV = np.asarray(uvs, dtype=np.float64)
    surface = _Surface.of(V, T, UT, UV)

    eye_reference = getattr(reference, "eye_reference", None)
    on_eye = np.zeros(len(T), bool)
    if eye_triangles is not None and eye_reference is not None:
        on_eye = np.asarray(eye_triangles, bool)
    groups = [(reference, ~on_eye), (eye_reference, on_eye if on_eye.any() else None)]
    if not on_eye.any():
        groups = [(reference, None)]
    maps = _bake_maps(surface, UT, UV, size, groups, max_distance)
    parts = list(getattr(reference, "parts", []) or [])
    has_colour = any(p.base_color is not None and p.uvs is not None for p in parts)
    base_png, normal_png, rough_png = _to_png(maps, has_colour)
    rows, cols, tri, bary = maps["texels"]
    coverage = maps["reached"]

    thickness = curvature = None
    if skin_maps:
        exclude = None if not on_eye.any() else on_eye
        thick = vertex_thickness(V, T, thickness_mm / 1000.0, exclude)
        image = _vertex_map(
            thick * 1000.0 / thickness_mm, rows, cols, tri, bary, T, size
        )
        thickness = np.clip(np.round(image * 255), 0, 255).astype(np.uint8)
        curv = vertex_curvature(V, T)
        image = _vertex_map(
            0.5 + 0.5 * np.clip(curv / curvature_per_m, -1, 1),
            rows,
            cols,
            tri,
            bary,
            T,
            size,
        )
        curvature = np.clip(np.round(image * 255), 0, 255).astype(np.uint8)

    head = {}
    if head_size and head_box is not None and head_triangles is not None:
        from opensculptboy.texture.tiles import HEAD_TILE_ASPECT, to_tile_uvs

        height = int(head_size)
        width = int(round(height * HEAD_TILE_ASPECT))
        tile = _bake_maps(
            surface,
            UT,
            to_tile_uvs(UV, head_box),
            (width, height),
            [(reference, None)],
            max_distance,
            pick=np.asarray(head_triangles, bool) & ~on_eye,
            supersample=supersample,
        )
        colour, normal, rough = _to_png(tile, has_colour)
        head = {"normal": normal}
        if colour is not None:
            head["base_color"] = colour
        if rough is not None:
            head["roughness"] = rough

    ears = {}
    if ear_mesh is not None and ear_mesh[2] is not None and np.any(ear_mesh[2]):
        from opensculptboy.texture.ears import EAR_TILE_SIZE

        ear_ut, ear_uv, ear_mask = (np.asarray(a) for a in ear_mesh)
        ear_surface = _Surface.of(V, T, ear_ut.astype(np.int64), ear_uv)
        tile = _bake_maps(
            ear_surface,
            ear_ut.astype(np.int64),
            ear_uv,
            EAR_TILE_SIZE,
            [(reference, None)],
            max_distance,
            pick=ear_mask.astype(bool),
            supersample=supersample + 1,
        )
        colour, normal, rough = _to_png(tile, has_colour)
        ears = {"normal": normal}
        if colour is not None:
            ears["base_color"] = colour
        if rough is not None:
            ears["roughness"] = rough
        if skin_maps:
            r, c, t_, b_ = tile["texels"]
            w, h = EAR_TILE_SIZE
            for slot, values in (
                ("thickness", thick * 1000.0 / thickness_mm),
                ("curvature", 0.5 + 0.5 * np.clip(curv / curvature_per_m, -1, 1)),
            ):
                image = np.zeros((h, w))
                known = np.zeros((h, w), bool)
                image[r, c] = np.einsum("nk, nk -> n", b_, values[T[t_]])
                known[r, c] = True
                image = _fill(image, known)
                ears[slot] = np.clip(np.round(image * 255), 0, 255).astype(np.uint8)

    eyes = None
    if (
        eye_box is not None
        and eye_triangles is not None
        and eye_reference is not None
        and eye_size > 0
    ):
        eyes = _bake_eyes(
            surface,
            UT,
            UV,
            eye_triangles,
            eye_reference,
            eye_box,
            eye_size,
            max_distance,
        )

    return BakedTextures(
        base_color=base_png,
        normal=normal_png,
        coverage=coverage,
        covered_share=float(coverage.sum() / max(maps["inside"], 1)),
        eyes=eyes,
        roughness=rough_png,
        thickness=thickness,
        curvature=curvature,
        thickness_mm=float(thickness_mm),
        curvature_per_m=float(curvature_per_m),
        head=head,
        ears=ears,
    )


def _bake_eyes(surface, UT, UV, eye_triangles, target, box, size, max_distance):
    """The base colour (size / 2, size, 3) of the eye texture: the eyeball triangles laid out
    by the box of the atlas, each texel taking the reference's eyes along the normal."""
    from opensculptboy.texture.eyes import EYE_TEXTURE_ASPECT, to_eye_uvs

    width, height = int(size), int(size) // EYE_TEXTURE_ASPECT
    maps = _bake_maps(
        surface,
        UT,
        to_eye_uvs(UV, box),
        (width, height),
        [(target, None)],
        max_distance,
        pick=np.asarray(eye_triangles, bool),
    )
    if not maps["colour_known"].any():
        return None
    return np.clip(
        np.round(_fill(maps["colour"], maps["colour_known"])), 0, 255
    ).astype(np.uint8)


def bake_fit(result, subdiv: int | None = None, size: int = 1024, **kwargs):
    """
    Bake the reference of a :class:`~opensculptboy.fit.register.FitResult` onto the mesh of a
    level of its detail layer (its finest by default), posed and placed as the fit left it.
    ``head_size`` (in ``kwargs``) also bakes the head tile at that height, and
    ``ear_tile=True`` unwraps the outer ears of a level from 1 up onto the ear tile
    (:mod:`opensculptboy.texture.ears`) and bakes it; ``BakedTextures.ear_uvs`` then holds
    its UVs.
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
    ear_uvs = None
    if "eye_triangles" not in kwargs:
        kwargs["eye_triangles"] = eyeball_triangles(model, topology)
    if "eye_box" not in kwargs:
        from opensculptboy.texture.eyes import eye_uv_box

        kwargs["eye_box"] = eye_uv_box(model)
    if kwargs.pop("ear_tile", False) and "ear_mesh" not in kwargs and subdiv > 0:
        from opensculptboy.texture.ears import ear_triangles, ear_uv_table, with_ear_uvs

        rest = out["rest_vertices"][0].numpy()
        table = ear_uv_table(
            rest, topology, ear_triangles(model, topology), np.sign(rest[:, 0])
        )
        mesh, mask = with_ear_uvs(topology, table)
        if mask is not None:
            kwargs["ear_mesh"] = (mesh.uv_triangles, mesh.uvs, mask)
            ear_uvs = table
    if kwargs.get("head_size") and "head_box" not in kwargs:
        from opensculptboy.texture.tiles import head_uv_box, on_head_tile

        kwargs["head_box"] = head_uv_box(model, character.topology)
        kwargs["head_triangles"] = on_head_tile(
            model, topology.uvs, topology.uv_triangles, character.topology
        )
    baked = bake(
        vertices,
        topology.triangles,
        topology.uv_triangles,
        topology.uvs,
        result.reference,
        size,
        **kwargs,
    )
    if baked.ears:
        baked.ear_uvs = ear_uvs
    return baked


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


def bake_card(
    card_path, handoff, subdiv: int | None = None, size: int = 1024, **kwargs
):
    """
    Bake the reference of a handoff folder (or a loaded
    :class:`~opensculptboy.fit.reference.Reference`) onto a fitted card again, without
    fitting: the card's own offsets and placement (``detail.fit.placement``) place the mesh
    as the fit left it. Returns the :class:`BakedTextures`.
    """
    import json
    import pathlib

    from opensculptboy.character import Character
    from opensculptboy.detail import character_offsets
    from opensculptboy.fit.reference import Reference

    card_path = pathlib.Path(card_path)
    character = Character.from_dict(json.loads(card_path.read_text()))
    placement = character.detail.get("fit", {}).get("placement")
    if not placement:
        raise ValueError(f"{card_path} holds no fit placement (detail.fit.placement).")
    if isinstance(handoff, Reference):
        reference = handoff
    else:
        from opensculptboy.fit.handoff import load_handoff

        reference = Reference.from_handoff(load_handoff(handoff))
    levels = character.detail_levels()
    subdiv = max(levels) if subdiv is None else subdiv
    offsets = {subdiv: character_offsets(character, subdiv, card_path.parent)}
    result = dataclasses.make_dataclass(
        "_Fitted", ["character", "rotation", "translation", "offsets", "reference"]
    )(
        character,
        np.asarray(placement["rotation"], np.float64),
        np.asarray(placement["translation"], np.float64),
        offsets,
        reference,
    )
    return bake_fit(result, subdiv, size, **kwargs)

# OpenSculptBoy
# Apache License, Version 2.0
"""
Texture atlases of the MToon materials of a VRM file.

:func:`atlas_spec` merges the materials of a :class:`~opensculptboy.export.vrm.VrmSpec` that
draw alike into atlas materials until the file holds no more than a given number of materials.
Two materials draw alike when they share the alpha mode (and the cutoff of ``MASK``), the
double-sided flag, the MToon shading, rim and outline settings and the ratio of the shade
colour to the base colour. When the exact groups are too many, the merge relaxes in steps: the
cutoffs of ``MASK`` materials join (the alpha of each tile is remapped so that its cutoff falls
on the group's), ``OPAQUE`` materials join ``MASK`` ones (their tiles take an alpha of 1), and
single-sided materials join double-sided ones. ``BLEND`` materials only join each other.

Each atlas page holds a tile per source region: the box of the UVs that the members draw on
their maps, snapped to the texels of the largest map, so that a tile at its native size copies
the texels exactly. A tile repeats its map outside [0, 1] as the sampler did. The base colour
factor and the normal scale are baked into the tiles (the base colour in linear light, with
the alpha premultiplied while it is filtered), the shade colour keeps its ratio to the base
colour, and the normal map is filtered as vectors. Every tile has a margin of ``padding``
texels from the same map, and the tiles start on multiples of 8 texels, so the first three mip
levels of a page never mix two tiles. A page larger than the size limit shrinks the tiles of
the densest texels first (the texels per metre of the surface), so the texel density evens
out before any tile loses its detail. The vertices that a tile's UVs move are split from those
of other materials, and the primitives of a mesh that share an atlas material join.

Only materials of the wearable meshes join an atlas: the skin, its tiles, the eyes, the lashes
and the mouth keep their materials, which carry texture transforms, matcaps and ramps.
"""

from __future__ import annotations

import dataclasses
import io

import numpy as np

PADDING = 8
ALIGN = 8
MIN_TILE = 8
# the tiles of a page fill at most this share of it before the next size is tried
FILL = 0.92


# ---------------------------------------------------------------------------------------------
# Colour and images


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(
        np.float32
    )


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, np.float32), 0.0, 1.0)
    return np.where(
        c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055
    ).astype(np.float32)


_LUT = _srgb_to_linear(np.arange(256) / 255.0)


def decode(data: bytes) -> np.ndarray:
    """(H, W, 4) uint8 pixels of a PNG or JPEG file (RGBA; alpha 255 where the file has
    none), row 0 at the top."""
    from PIL import Image

    image = Image.open(io.BytesIO(bytes(data)))
    return np.asarray(image.convert("RGBA"))


def has_alpha(data: bytes) -> bool:
    """Whether an image file stores an alpha channel."""
    from PIL import Image

    image = Image.open(io.BytesIO(bytes(data)))
    return image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info


def _encode_png(pixels: np.ndarray) -> bytes:
    from opensculptboy.export.document import png_bytes

    return png_bytes(np.ascontiguousarray(pixels))


def colour_linear(pixels: np.ndarray) -> np.ndarray:
    """(H, W, 4) float32: linear RGB premultiplied by alpha, and alpha."""
    out = np.empty(pixels.shape[:2] + (4,), np.float32)
    alpha = pixels[..., 3].astype(np.float32) / 255.0
    out[..., :3] = _LUT[pixels[..., :3]] * alpha[..., None]
    out[..., 3] = alpha
    return out


def colour_pixels(linear: np.ndarray, alpha_out: bool = True) -> np.ndarray:
    """uint8 sRGB pixels of premultiplied linear colour (RGBA, or RGB)."""
    alpha = linear[..., 3]
    rgb = linear[..., :3] / np.maximum(alpha, 1e-6)[..., None]
    rgb = np.where(alpha[..., None] > 1e-6, rgb, 0.0)
    srgb = np.round(_linear_to_srgb(rgb) * 255.0)
    if not alpha_out:
        return srgb.astype(np.uint8)
    a = np.round(np.clip(alpha, 0.0, 1.0) * 255.0)
    return np.concatenate([srgb, a[..., None]], -1).astype(np.uint8)


def normal_vectors(pixels: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """(H, W, 3) float32 unit vectors of a tangent-space normal map, with ``scale`` on X
    and Y (glTF normalTexture.scale)."""
    n = pixels[..., :3].astype(np.float32) / 127.5 - 1.0
    if scale != 1.0:
        n[..., :2] *= scale
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)


def normal_pixels(vectors: np.ndarray) -> np.ndarray:
    n = vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-6)
    return np.clip(np.round((n + 1.0) * 127.5), 0, 255).astype(np.uint8)


def _bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear samples of a float image (H, W, C) at texel coordinates (texel centres at
    integers), repeating outside the image."""
    h, w = image.shape[:2]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0).astype(np.float32)[..., None]
    fy = (y - y0).astype(np.float32)[..., None]
    x0 %= w
    y0 %= h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def sample_region(
    image: np.ndarray, box, size, padding: int, exact: bool = False
) -> np.ndarray:
    """
    The pixels (h + 2 padding, w + 2 padding, C) of a float image over the UV box
    ``(u0, v0, u1, v1)`` (glTF UVs, repeating outside [0, 1]) at ``size`` (w, h), with
    ``padding`` texels of the same map around it. ``exact`` copies the texels (the box on
    texel boundaries and one texel per pixel); otherwise each pixel averages a grid of
    bilinear samples that covers its footprint.
    """
    u0, v0, u1, v1 = box
    w, h = size
    H, Wd = image.shape[:2]
    j = np.arange(-padding, w + padding)
    i = np.arange(-padding, h + padding)
    if exact:
        x = np.round(u0 * Wd).astype(np.int64) + j
        y = np.round(v0 * H).astype(np.int64) + i
        return image[(y % H)[:, None], (x % Wd)[None, :]]
    step_x = (u1 - u0) * Wd / w  # source texels per pixel
    step_y = (v1 - v0) * H / h
    kx = max(1, int(np.ceil(step_x - 1e-6)))
    ky = max(1, int(np.ceil(step_y - 1e-6)))
    out = np.zeros((len(i), len(j), image.shape[2]), np.float32)
    rows = max(1, int(4_000_000 // max(len(j) * kx * ky, 1)))
    for start in range(0, len(i), rows):
        ii = i[start : start + rows]
        acc = np.zeros((len(ii), len(j), image.shape[2]), np.float32)
        for a in range(ky):
            for b in range(kx):
                u = u0 + (j + (b + 0.5) / kx) / w * (u1 - u0)
                v = v0 + (ii + (a + 0.5) / ky) / h * (v1 - v0)
                X = np.broadcast_to(u[None, :] * Wd - 0.5, (len(ii), len(j)))
                Y = np.broadcast_to(v[:, None] * H - 0.5, (len(ii), len(j)))
                acc += _bilinear(image, X, Y)
        out[start : start + rows] = acc / (kx * ky)
    return out


def resize_image(data: bytes, size, role: str = "colour") -> bytes:
    """A PNG file of an image resized to ``size`` (width, height): colour in linear light
    with premultiplied alpha (the alpha kept when the file has one), normal maps as vectors,
    grey or data maps as they are."""
    pixels = decode(data)
    w, h = int(size[0]), int(size[1])
    if role == "normal":
        n = sample_region(normal_vectors(pixels), (0, 0, 1, 1), (w, h), 0)
        return _encode_png(normal_pixels(n))
    if role == "data":
        n = sample_region(pixels.astype(np.float32), (0, 0, 1, 1), (w, h), 0)
        return _encode_png(np.clip(np.round(n), 0, 255).astype(np.uint8))
    lin = sample_region(colour_linear(pixels), (0, 0, 1, 1), (w, h), 0)
    return _encode_png(colour_pixels(lin, has_alpha(data)))


# ---------------------------------------------------------------------------------------------
# Packing


def pack(sizes, width: int, height: int):
    """
    Positions (x, y) of rectangles ``sizes`` [(w, h)] on a page (skyline, bottom-left, the
    tallest first), or None when they do not fit. Positions are multiples of :data:`ALIGN`
    when the sizes are.
    """
    order = sorted(range(len(sizes)), key=lambda k: (-sizes[k][1], -sizes[k][0]))
    skyline = [(0, 0, width)]  # (x, y, length)
    out = [None] * len(sizes)
    for k in order:
        w, h = sizes[k]
        best = None
        for s in range(len(skyline)):
            x = skyline[s][0]
            if x + w > width:
                break
            # the height of the segments under [x, x + w)
            y, covered, t = 0, 0, s
            while covered < w and t < len(skyline):
                y = max(y, skyline[t][1])
                covered = skyline[t][0] + skyline[t][2] - x
                t += 1
            if covered < w or y + h > height:
                continue
            if best is None or (y, x) < (best[1], best[0]):
                best = (x, y, s)
        if best is None:
            return None
        x, y, s = best
        out[k] = (x, y)
        # raise the skyline under the rectangle
        new = []
        for sx, sy, sl in skyline:
            end = sx + sl
            if end <= x or sx >= x + w:
                new.append((sx, sy, sl))
                continue
            if sx < x:
                new.append((sx, sy, x - sx))
            if end > x + w:
                new.append((x + w, sy, end - x - w))
        new.append((x, y + h, w))
        new.sort()
        merged = []
        for seg in new:
            if (
                merged
                and merged[-1][1] == seg[1]
                and merged[-1][0] + merged[-1][2] == seg[0]
            ):
                merged[-1] = (merged[-1][0], seg[1], merged[-1][2] + seg[2])
            else:
                merged.append(seg)
        skyline = merged
    return out


def _round_up(x: int, step: int = ALIGN) -> int:
    return int(-(-int(x) // step) * step)


# ---------------------------------------------------------------------------------------------
# Groups of materials


def shading_key(m) -> tuple:
    """The MToon settings that a material's atlas shares (besides the alpha mode, the cutoff
    and the double-sided flag)."""

    def r(values):
        return tuple(round(float(x), 4) for x in np.atleast_1d(values))

    base = np.asarray(m.base_color[:3], np.float64)
    ratio = np.asarray(m.shade_color, np.float64) / np.maximum(base, 1e-6)
    return (
        r(ratio),
        r(m.shading_shift),
        r(m.shading_toony),
        r(m.gi_equalization),
        r(m.rim_color),
        r(m.rim_fresnel_power),
        r(m.rim_lift),
        r(m.rim_lighting_mix),
        m.outline_width_mode,
        r(m.outline_width),
        r(m.outline_color),
        r(m.outline_lighting_mix),
        r(m.emissive),
        int(m.render_queue_offset),
        bool(m.transparent_with_zwrite),
    )


def atlasable(m) -> bool:
    """Whether a material can join an atlas: no texture transform, matcap, emissive, rim or
    outline texture, a shade texture equal to the base colour texture (or none), and a base
    colour whose zero channels have no shade."""
    if m.texture_transform is not None or m.matcap_texture is not None:
        return False
    if any(
        t is not None
        for t in (m.emissive_texture, m.rim_multiply_texture, m.outline_width_texture)
    ):
        return False
    if m.shade_texture is not None and m.shade_texture != m.base_texture:
        return False
    base = np.asarray(m.base_color[:3])
    shade = np.asarray(m.shade_color)
    return not np.any((base < 1e-6) & (shade > 1e-6))


def group_key(m, level: int) -> tuple:
    """The key of a material's group at a relaxation level (0: exact; 1: the cutoffs of MASK
    join; 2: OPAQUE joins MASK; 3: single-sided joins double-sided)."""
    mode = m.alpha_mode
    cutoff = round(float(m.alpha_cutoff), 4) if mode == "MASK" else None
    if level >= 1 and mode == "MASK":
        cutoff = None
    if level >= 2 and mode == "OPAQUE":
        mode, cutoff = "MASK", None
    sided = bool(m.double_sided)
    if level >= 3:
        sided = None
    return (mode, cutoff, sided, shading_key(m))


@dataclasses.dataclass
class _Source:
    """A tile: the region of one set of maps that members draw with one factor."""

    key: tuple
    group: int
    base: int | None
    normal: int | None
    factor: tuple
    normal_scale: float
    alpha: str  # "texture", "opaque" or the cutoff remap "c->t"
    cutoff: float
    target_cutoff: float
    uses: list  # (mesh, primitive)
    box: tuple = (0.0, 0.0, 1.0, 1.0)
    native: tuple = (MIN_TILE, MIN_TILE)  # (w, h) texels at the reference map's size
    reference: tuple = (MIN_TILE, MIN_TILE)  # the reference map's size
    area: float = 0.0  # surface (m^2)
    scale: float = 1.0
    size: tuple = (MIN_TILE, MIN_TILE)
    page: int = -1
    origin: tuple = (0, 0)


@dataclasses.dataclass
class AtlasReport:
    """What :func:`atlas_spec` did."""

    level: int = 0
    pages: list = dataclasses.field(default_factory=list)
    merged: int = 0
    materials_before: int = 0
    materials_after: int = 0
    messages: list = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _uv_box(uvs: np.ndarray, reference) -> tuple:
    """The box of UVs, snapped outward to the texels of a map of size ``reference``."""
    w, h = reference
    lo = uvs.min(0)
    hi = uvs.max(0)
    x0, y0 = np.floor(lo[0] * w), np.floor(lo[1] * h)
    x1, y1 = np.ceil(hi[0] * w), np.ceil(hi[1] * h)
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    return (x0 / w, y0 / h, x1 / w, y1 / h), (int(x1 - x0), int(y1 - y0))


def atlas_spec(
    spec,
    max_materials: int,
    max_size: int,
    padding: int = PADDING,
    materials: set | None = None,
) -> AtlasReport:
    """
    Merge the wearable meshes' materials of ``spec`` into atlases until it holds at most
    ``max_materials`` materials, with pages of at most ``max_size`` texels a side. Changes
    the spec in place; ``materials`` limits the merge to those material indices.
    """
    from opensculptboy.export.budget import image_size
    from opensculptboy.export.pieces import primitive_triangles

    _drop_unused(spec, spec.materials, spec.images)  # materials that no primitive draws
    report = AtlasReport(materials_before=len(spec.materials))
    if len(spec.materials) <= max_materials:
        report.materials_after = len(spec.materials)
        return report
    # the materials that may join: used by wearable meshes alone
    users: dict[int, list] = {}
    for m, mesh in enumerate(spec.meshes):
        for p, primitive in enumerate(mesh.primitives):
            users.setdefault(primitive.material, []).append((m, p))
    body = {primitive.material for primitive in spec.meshes[0].primitives}
    candidates = [
        k
        for k, mat in enumerate(spec.materials)
        if k in users
        and k not in body
        and atlasable(mat)
        and (materials is None or k in materials)
        and all(
            spec.meshes[m].body.uvs is not None or mat.base_texture is None
            for m, _ in users[k]
        )
    ]
    fixed = len(spec.materials) - len(candidates)
    slots = max_materials - fixed
    if slots < 1:
        report.messages.append(
            f"{fixed} materials cannot join an atlas, more than the {max_materials} allowed"
        )
        report.materials_after = len(spec.materials)
        return report
    level, groups = 0, {}
    for level in range(4):
        groups = {}
        for k in candidates:
            groups.setdefault(group_key(spec.materials[k], level), []).append(k)
        if len(groups) <= slots:
            break
    report.level = level
    if len(groups) > slots:
        report.messages.append(
            f"{len(groups)} groups of materials draw differently, more than the {slots} "
            "atlas materials allowed"
        )
    keys = list(groups)
    # each group's cutoff (the members' area-weighted median for MASK)
    sources: dict[tuple, _Source] = {}
    for g, key in enumerate(keys):
        members = groups[key]
        modes = {spec.materials[k].alpha_mode for k in members}
        cutoffs = [
            float(spec.materials[k].alpha_cutoff)
            for k in members
            if spec.materials[k].alpha_mode == "MASK"
        ]
        target = float(np.median(cutoffs)) if cutoffs else 0.5
        for k in members:
            mat = spec.materials[k]
            if mat.alpha_mode == "OPAQUE" and "MASK" in modes:
                alpha = "opaque"
            elif mat.alpha_mode == "OPAQUE":
                alpha = "opaque"
            elif mat.alpha_mode == "MASK" and abs(mat.alpha_cutoff - target) > 1e-4:
                alpha = "remap"
            else:
                alpha = "texture"
            factor = tuple(round(float(x), 6) for x in mat.base_color)
            skey = (
                g,
                mat.base_texture,
                mat.normal_texture,
                factor,
                round(float(mat.normal_scale), 4),
                alpha,
                round(float(mat.alpha_cutoff), 4),
            )
            if skey not in sources:
                sources[skey] = _Source(
                    key=skey,
                    group=g,
                    base=mat.base_texture,
                    normal=mat.normal_texture,
                    factor=tuple(float(x) for x in mat.base_color),
                    normal_scale=float(mat.normal_scale),
                    alpha=alpha,
                    cutoff=float(mat.alpha_cutoff),
                    target_cutoff=target,
                    uses=[],
                )
            sources[skey].uses.extend(users[k])
    sizes = [image_size(data) for _, data in spec.images]
    for source in sources.values():
        maps = [i for i in (source.base, source.normal) if i is not None]
        if not maps:
            source.reference = (MIN_TILE, MIN_TILE)
        else:
            source.reference = max(
                (tuple(sizes[i]) for i in maps), key=lambda s: s[0] * s[1]
            )
        uv_rows, area = [], 0.0
        for m, p in source.uses:
            mesh = spec.meshes[m]
            tris = primitive_triangles(mesh, mesh.primitives[p])
            if mesh.body.uvs is not None:
                uv_rows.append(np.asarray(mesh.body.uvs)[tris.ravel()])
            pos = np.asarray(mesh.body.positions)[tris]
            area += 0.5 * float(
                np.linalg.norm(
                    np.cross(pos[:, 1] - pos[:, 0], pos[:, 2] - pos[:, 0]), axis=1
                ).sum()
            )
        source.area = max(area, 1e-8)
        if maps and uv_rows:
            source.box, source.native = _uv_box(
                np.concatenate(uv_rows), source.reference
            )
        else:
            source.box, source.native = (0.0, 0.0, 1.0, 1.0), (MIN_TILE, MIN_TILE)
    # pages: one per group, more for groups that need them, while slots allow
    by_group: dict[int, list] = {}
    for source in sources.values():
        by_group.setdefault(source.group, []).append(source)
    capacity = float(max_size * max_size) * FILL
    demand = {
        g: sum(
            (s.native[0] + 2 * padding) * (s.native[1] + 2 * padding) for s in members
        )
        for g, members in by_group.items()
    }
    pages_of = {g: 1 for g in by_group}
    spare = slots - len(by_group)
    while spare > 0:
        g = max(by_group, key=lambda g: demand[g] / pages_of[g])
        if demand[g] / pages_of[g] <= capacity or pages_of[g] >= len(by_group[g]):
            break
        pages_of[g] += 1
        spare -= 1
    # split each group's tiles over its pages (the largest first, onto the emptiest page)
    page_sources: list[list] = []
    page_group: list[int] = []
    for g, members in by_group.items():
        first = len(page_sources)
        for _ in range(pages_of[g]):
            page_sources.append([])
            page_group.append(g)
        load = [0.0] * pages_of[g]
        for s in sorted(members, key=lambda s: -s.native[0] * s.native[1]):
            k = int(np.argmin(load))
            load[k] += s.native[0] * s.native[1]
            page_sources[first + k].append(s)
    # lay out each page: shrink the densest tiles until the page fits
    page_sizes = []
    for pk, members in enumerate(page_sources):
        layout = _layout(members, max_size, padding)
        page_sizes.append(layout)
        for s in members:
            s.page = pk
    # the page images and the new materials
    images = list(spec.images)
    new_materials = list(spec.materials)
    page_material = {}
    for pk, members in enumerate(page_sources):
        width, height = page_sizes[pk]
        key = keys[page_group[pk]]
        mats = [spec.materials[k] for k in groups[key]]
        alpha_out = any(s.alpha != "opaque" for s in members)
        base_page = np.zeros((height, width, 4), np.float32)
        normal_page = None
        if any(s.normal is not None for s in members):
            normal_page = np.zeros((height, width, 3), np.float32)
            normal_page[..., 2] = 1.0
        filled = np.zeros((height, width), bool)
        for s in members:
            _draw(s, spec.images, sizes, base_page, normal_page, padding)
            x, y = s.origin
            filled[y : y + s.size[1] + 2 * padding, x : x + s.size[0] + 2 * padding] = (
                True
            )
        if (~filled).any() and filled.any():  # the empty texels: the page's mean colour
            mean = base_page[filled].mean(0)
            base_page[~filled] = mean
        name = f"atlas{pk}"
        base_index = len(images)
        images.append(
            (f"{name}:base_color", _encode_png(colour_pixels(base_page, alpha_out)))
        )
        normal_index = None
        if normal_page is not None:
            normal_index = len(images)
            images.append((f"{name}:normal", _encode_png(normal_pixels(normal_page))))
        first = mats[0]
        mode = key[0]
        if mode == "MASK" and not alpha_out:
            mode = "OPAQUE"
        ratio = np.asarray(first.shade_color) / np.maximum(
            np.asarray(first.base_color[:3]), 1e-6
        )
        material = dataclasses.replace(
            first,
            name=f"{name}:" + "+".join(sorted({m.name.split(":")[0] for m in mats})),
            base_color=(1.0, 1.0, 1.0, 1.0),
            base_texture=base_index,
            shade_texture=base_index,
            shade_color=tuple(float(x) for x in ratio),
            normal_texture=normal_index,
            normal_scale=1.0,
            alpha_mode=mode,
            alpha_cutoff=float(members[0].target_cutoff),
            double_sided=bool(key[2])
            if key[2] is not None
            else any(m.double_sided for m in mats),
        )
        page_material[pk] = len(new_materials)
        new_materials.append(material)
        report.pages.append(
            {
                "name": material.name,
                "size": [width, height],
                "alpha_mode": material.alpha_mode,
                "double_sided": material.double_sided,
                "tiles": len(members),
                "materials": sorted(
                    {
                        spec.materials[k].name
                        for k in groups[key]
                        if any(u in src.uses for src in members for u in users[k])
                    }
                ),
                "smallest_scale": float(min(s.scale for s in members)),
            }
        )
    # move the UVs into the tiles, splitting the vertices that two tiles share
    tile_of = {}
    for s in sources.values():
        for use in s.uses:
            tile_of[use] = s
    for m, mesh in enumerate(spec.meshes):
        uses = [(m, p) for p in range(len(mesh.primitives))]
        if not any(u in tile_of for u in uses):
            continue
        _retile_mesh(mesh, m, tile_of, page_material, page_sizes, padding)
    report.merged = len(candidates)
    _drop_unused(spec, new_materials, images)
    report.materials_after = len(spec.materials)
    return report


def _layout(members, max_size: int, padding: int):
    """Place the tiles of a page: their sizes (shrunk by density where the largest page does
    not hold them at their native size) and origins; returns the page's (width, height)."""
    density = np.array(
        [np.sqrt(s.native[0] * s.native[1] / s.area) for s in members], np.float64
    )

    def sizes_at(cap):
        out = []
        for s, d in zip(members, density):
            scale = 1.0 if cap is None else min(1.0, cap / d)
            w = max(MIN_TILE // 2, int(round(s.native[0] * scale)))
            h = max(MIN_TILE // 2, int(round(s.native[1] * scale)))
            out.append((scale, (w, h)))
        return out

    def try_fit(sized, width, height):
        rects = [
            (_round_up(w + 2 * padding), _round_up(h + 2 * padding))
            for _, (w, h) in sized
        ]
        return pack(rects, width, height)

    candidates = []
    size = 64
    while size <= max_size:
        candidates.extend(
            [(size, size // 2), (size, size)] if size > 64 else [(size, size)]
        )
        size *= 2
    candidates = sorted(
        {c for c in candidates if c[0] <= max_size}, key=lambda c: c[0] * c[1]
    )

    def fit(cap):
        sized = sizes_at(cap)
        need = sum(
            _round_up(w + 2 * padding) * _round_up(h + 2 * padding)
            for _, (w, h) in sized
        )
        for width, height in candidates:
            if need > width * height:
                continue
            placed = try_fit(sized, width, height)
            if placed is not None:
                return sized, placed, (width, height)
        return None

    found = fit(None)
    if found is None:
        lo, hi = 0.0, float(density.max())
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if fit(mid) is not None:
                lo = mid
            else:
                hi = mid
        found = fit(lo) if lo > 0 else None
        if found is None:  # every tile at its smallest
            found = fit(1e-9)
    sized, placed, page = found
    for s, (scale, (w, h)), (x, y) in zip(members, sized, placed):
        s.scale = float(min(scale, w / s.native[0], h / s.native[1]))
        s.size = (w, h)
        s.origin = (x, y)
    return page


def _draw(source, images, sizes, base_page, normal_page, padding):
    """Draw a tile's maps into the page arrays."""
    x, y = source.origin
    w, h = source.size
    region = (slice(y, y + h + 2 * padding), slice(x, x + w + 2 * padding))
    factor = np.asarray(source.factor, np.float32)
    if source.base is not None:
        pixels = decode(images[source.base][1])
        exact = tuple(sizes[source.base]) == tuple(source.reference) and (
            (w, h) == tuple(source.native)
        )
        lin = sample_region(colour_linear(pixels), source.box, (w, h), padding, exact)
    else:
        lin = np.ones((h + 2 * padding, w + 2 * padding, 4), np.float32)
    alpha = lin[..., 3:4]
    rgb = lin[..., :3] / np.maximum(alpha, 1e-6)
    rgb = np.where(alpha > 1e-6, rgb, 0.0) * factor[:3]
    a = alpha[..., 0] * factor[3]
    if source.alpha == "opaque":
        a = np.ones_like(a)
    elif source.alpha == "remap":
        c, t = source.cutoff, source.target_cutoff
        a = np.where(
            a < c,
            a * (t / max(c, 1e-6)),
            t + (a - c) * ((1 - t) / max(1 - c, 1e-6)),
        )
    a = np.clip(a, 0.0, 1.0)
    base_page[region] = np.concatenate([rgb * a[..., None], a[..., None]], -1)
    if normal_page is not None:
        if source.normal is not None:
            pixels = decode(images[source.normal][1])
            exact = tuple(sizes[source.normal]) == tuple(source.reference) and (
                (w, h) == tuple(source.native)
            )
            n = sample_region(
                normal_vectors(pixels, source.normal_scale),
                source.box,
                (w, h),
                padding,
                exact,
            )
            normal_page[region] = n / np.maximum(
                np.linalg.norm(n, axis=-1, keepdims=True), 1e-6
            )
        else:
            normal_page[region] = (0.0, 0.0, 1.0)


def _retile_mesh(mesh, m, tile_of, page_material, page_sizes, padding):
    """Move a mesh's UVs into its tiles, split the vertices that two tiles (or a tile and a
    material outside the atlas) share, and join the primitives of one atlas material."""
    from opensculptboy.export.pieces import primitive_triangles
    from opensculptboy.export.vrm import SpecPrimitive

    body = mesh.body
    n = len(body.positions)
    prims = mesh.primitives
    tris = [primitive_triangles(mesh, p) for p in prims]
    # the user of each primitive: its tile, or None for a material outside the atlas
    owner = [tile_of.get((m, p)) for p in range(len(prims))]
    keys = [
        id(o) if o is not None else ("material", prims[p].material)
        for p, o in enumerate(owner)
    ]
    first_user: dict[int, object] = {}
    copies: dict[tuple, int] = {}
    rows = [np.arange(n)]
    count = n
    remap_tris = []
    extra_sources = []
    for p, t in enumerate(tris):
        flat = t.ravel()
        new = flat.copy()
        index = {}
        for v in np.unique(flat).tolist():
            if first_user.setdefault(v, keys[p]) == keys[p]:
                continue
            if (v, keys[p]) not in copies:
                copies[(v, keys[p])] = count
                extra_sources.append(np.array([v]))
                count += 1
            index[v] = copies[(v, keys[p])]
        if index:
            lookup = np.arange(n)
            for v, k in index.items():
                lookup[v] = k
            new = lookup[flat]
        remap_tris.append(new.reshape(-1, 3))
    source = np.concatenate(rows + extra_sources) if extra_sources else rows[0]
    if len(source) > n:
        _take_vertices(body, source)
    # a mesh without UVs gets them: each tile's vertices at the middle of their box (its
    # constant colour)
    flat = body.uvs is None
    uvs = np.full((len(source), 2), 0.5) if flat else np.array(body.uvs, np.float64)
    moved = np.zeros(len(source), bool)
    for p, t in enumerate(remap_tris):
        s = owner[p]
        if s is None:
            continue
        verts = np.unique(t)
        verts = verts[~moved[verts]]
        width, height = page_sizes[s.page]
        u0, v0, u1, v1 = s.box
        x, y = s.origin
        w, h = s.size
        uv = uvs[verts]
        uv[:, 0] = (x + padding + (uv[:, 0] - u0) / (u1 - u0) * w) / width
        uv[:, 1] = (y + padding + (uv[:, 1] - v0) / (v1 - v0) * h) / height
        uvs[verts] = uv
        moved[verts] = True
    body.uvs = uvs
    # one primitive per material
    joined: dict[int, list] = {}
    order = []
    for p, t in enumerate(remap_tris):
        s = owner[p]
        material = prims[p].material if s is None else page_material[s.page]
        if material not in joined:
            joined[material] = []
            order.append(material)
        joined[material].append(t)
    mesh.primitives = [
        SpecPrimitive(material, np.concatenate(joined[material])) for material in order
    ]
    body.triangles = np.concatenate([p.triangles for p in mesh.primitives])
    faces = np.asarray(body.triangle_faces)
    if len(faces) != len(body.triangles):
        body.triangle_faces = np.zeros(len(body.triangles), np.int64)


def _take_vertices(body, source: np.ndarray) -> None:
    """Make ``body`` hold the vertices ``source`` (indices into its current vertices)."""
    for field in (
        "positions",
        "normals",
        "uvs",
        "source",
        "joints",
        "weights",
        "tangents",
        "anny_source",
    ):
        value = getattr(body, field)
        if value is not None:
            setattr(body, field, np.asarray(value)[source])
    body.targets = [np.asarray(t)[source] for t in body.targets]


def _drop_unused(spec, materials, images) -> None:
    """Keep the materials that a primitive draws with and the images that a material uses,
    renumbered in their order."""
    from opensculptboy.export.pieces import TEXTURE_FIELDS

    used = []
    for mesh in spec.meshes:
        for p in mesh.primitives:
            if p.material not in used:
                used.append(p.material)
    keep_m = sorted(used)
    m_index = {old: new for new, old in enumerate(keep_m)}
    kept = [materials[k] for k in keep_m]
    used_images = sorted(
        {
            getattr(m, f)
            for m in kept
            for f in TEXTURE_FIELDS
            if getattr(m, f) is not None
        }
    )
    i_index = {old: new for new, old in enumerate(used_images)}
    for m in kept:
        for f in TEXTURE_FIELDS:
            if getattr(m, f) is not None:
                setattr(m, f, i_index[getattr(m, f)])
    for mesh in spec.meshes:
        for p in mesh.primitives:
            p.material = m_index[p.material]
    spec.materials = kept
    spec.images = [images[k] for k in used_images]

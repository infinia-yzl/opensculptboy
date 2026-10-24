# OpenSculptBoy
# Apache License, Version 2.0
"""
The quality checks of a garment before it becomes a package, shared by ``wearables bind``
and ``wearables import-mhclo`` (docs/wearables.md, "Quality").

:func:`check_garment` checks the geometry, the winding and the normals, the UVs, and the
materials and their textures, fixes what it can fix safely and returns the garment with a
:class:`Quality` report. Each item of the report has a stable ``code``, a ``severity``
(``error``, ``warning`` or ``info``), a message and a ``hint``; a fix is an ``info`` item with
``fixed`` set. An ``error`` item fails the command (:func:`require`) unless the user accepts
the quality as it is.

The fixes: degenerate triangles and unused vertices are removed, duplicate vertices (one
position, UV and normal) are merged, triangles wound against their neighbours are turned
over, missing normals are computed (weighted by the angle of each corner, welded by
position, so that UV seams and material borders stay smooth), and a normal map in DirectX
convention gets its green channel flipped (``normal_green``). The packages and the exports
write MikkTSpace tangents for every material with a normal map.
"""

from __future__ import annotations

import dataclasses
import io

import numpy as np

from opensculptboy.wearables.format import Garment, WearableError

SEVERITIES = ("error", "warning", "info")
TEXTURE_LIMIT = 4096  # pixels: a larger side of a texture gets a warning
NORMAL_GREEN = ("auto", "keep", "flip")
LINEAR_SLOTS = ("normal", "occlusion", "metallic_roughness")
WELD = 1e-7  # metres: vertices this close count as one position
UV_RASTER = 1024  # pixels of the UV-overlap raster
OVERLAP_SHARE = 0.01  # a share of UV area covered twice above this is reported
DENSITY_SPREAD = 4.0  # p95 / p5 texel density above this is reported
GREEN_CONFIDENCE = 0.25  # |correlation| from which the green convention counts as known
FRAGMENT_TRIANGLES = 4  # a loose part of fewer triangles is a fragment
# the triangles of the default body in an export, for the share of the VRM budgets
BODY_TRIANGLES = 27_420


class QualityError(WearableError):
    """A garment with an error-severity quality item."""

    code = "quality_error"


@dataclasses.dataclass
class Item:
    code: str
    severity: str
    message: str
    hint: str = ""
    count: int | None = None
    fixed: bool = False
    material: str | None = None
    slot: str | None = None
    value: float | None = None

    def to_dict(self) -> dict:
        out = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "hint": self.hint,
        }
        for key in ("count", "material", "slot", "value"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        if self.fixed:
            out["fixed"] = True
        return out


@dataclasses.dataclass
class Quality:
    """The items of the checks, in the order found."""

    items: list = dataclasses.field(default_factory=list)

    def add(self, code, severity, message, hint="", **kw) -> Item:
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}.")
        item = Item(code, severity, message, hint, **kw)
        self.items.append(item)
        return item

    def fix(self, code, message, hint="", **kw) -> Item:
        return self.add(code, "info", message, hint, fixed=True, **kw)

    def of(self, severity) -> list:
        return [i for i in self.items if i.severity == severity]

    def codes(self) -> list[str]:
        return [i.code for i in self.items]

    def to_dict(self) -> dict:
        return {
            "errors": len(self.of("error")),
            "warnings": len(self.of("warning")),
            "infos": len(self.of("info")),
            "fixes": [i.code for i in self.items if i.fixed],
            "items": [i.to_dict() for i in self.items],
        }


def require(quality: Quality, accept: bool = False) -> None:
    """Raise :class:`QualityError` for the first error item unless ``accept``."""
    errors = quality.of("error")
    if errors and not accept:
        first = errors[0]
        codes = ", ".join(sorted({e.code for e in errors}))
        raise QualityError(
            f"The garment has quality errors ({codes}): {first.message}",
            first.code,
            (first.hint + " " if first.hint else "")
            + "Or pass --accept-quality to write the package as it is.",
        )


# -- geometry ---------------------------------------------------------------------------


def _areas(vertices, triangles) -> np.ndarray:
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def _keep(garment: Garment, triangles_kept: np.ndarray, rows: np.ndarray) -> Garment:
    """The garment with the triangles ``triangles_kept`` (bool) and the vertices ``rows``
    (old indices, in their new order); triangles are remapped."""
    remap = np.full(len(garment.vertices), -1, np.int64)
    remap[rows] = np.arange(len(rows))
    return Garment(
        garment.vertices[rows],
        remap[garment.triangles[triangles_kept]],
        normals=garment.normals[rows],
        uvs=None if garment.uvs is None else garment.uvs[rows],
        triangle_material=garment.triangle_material[triangles_kept],
        triangle_group=garment.triangle_group[triangles_kept],
        materials=garment.materials,
        groups=garment.groups,
    )


def fix_geometry(garment: Garment, quality: Quality) -> tuple[Garment, np.ndarray]:
    """Remove degenerate triangles and unused vertices and merge duplicate vertices; returns
    the garment and the old index of each new vertex."""
    t = garment.triangles
    repeated = (t[:, 0] == t[:, 1]) | (t[:, 1] == t[:, 2]) | (t[:, 2] == t[:, 0])
    flat = _areas(garment.vertices, t) < 1e-12
    bad = repeated | flat
    if bad.any():
        quality.fix(
            "degenerate_triangles_removed",
            f"{int(bad.sum())} triangles without area removed",
            count=int(bad.sum()),
        )
    used = np.unique(t[~bad])
    unused = len(garment.vertices) - len(used)
    if unused:
        quality.fix(
            "unused_vertices_removed",
            f"{unused} vertices of no triangle removed",
            count=int(unused),
        )
    garment = _keep(garment, ~bad, used)
    rows = used
    # duplicate vertices: one position, UV and normal
    key = [np.round(garment.vertices / WELD)]
    if garment.uvs is not None:
        key.append(np.round(garment.uvs / 1e-7))
    key.append(np.round(garment.normals / 1e-5))
    key = np.concatenate(key, axis=1).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    if len(first) < len(garment.vertices):
        merged = len(garment.vertices) - len(first)
        order = np.sort(first)
        new_of_unique = np.empty(len(first), np.int64)
        new_of_unique[np.argsort(first)] = np.arange(len(first))
        remap = new_of_unique[inverse]
        triangles = remap[garment.triangles]
        garment = Garment(
            garment.vertices[order],
            triangles,
            normals=garment.normals[order],
            uvs=None if garment.uvs is None else garment.uvs[order],
            triangle_material=garment.triangle_material,
            triangle_group=garment.triangle_group,
            materials=garment.materials,
            groups=garment.groups,
        )
        rows = rows[order]
        quality.fix(
            "duplicate_vertices_merged",
            f"{merged} duplicate vertices (one position, UV and normal) merged",
            count=int(merged),
        )
        t = garment.triangles
        again = (t[:, 0] == t[:, 1]) | (t[:, 1] == t[:, 2]) | (t[:, 2] == t[:, 0])
        if again.any():
            used = np.unique(t[~again])
            garment = _keep(garment, ~again, used)
            rows = rows[used]
    return garment, rows


def _edges(triangles):
    """Directed edges (3T, 2), their triangle and the undirected key of each."""
    e = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    tri = np.tile(np.arange(len(triangles)), 3)
    key = np.sort(e, axis=1)
    return e, tri, key


def check_topology(garment: Garment, quality: Quality, weld_id: np.ndarray) -> None:
    """Non-manifold edges and loose parts, on the mesh welded by position."""
    t = weld_id[garment.triangles]
    _, _, key = _edges(t)
    _, counts = np.unique(key, axis=0, return_counts=True)
    nonmanifold = int((counts > 2).sum())
    if nonmanifold:
        quality.add(
            "non_manifold_edges",
            "warning",
            f"{nonmanifold} edges are shared by more than two triangles",
            "Separate the faces that meet at one edge; the push-out and the ray tests "
            "expect each edge between at most two triangles.",
            count=nonmanifold,
        )
    labels = _components(int(weld_id.max(initial=-1)) + 1, t)
    tri_label = labels[t[:, 0]]
    sizes = np.bincount(tri_label)
    parts = int((sizes > 0).sum())
    fragments = int(((sizes > 0) & (sizes < FRAGMENT_TRIANGLES)).sum())
    quality.add("parts", "info", f"{parts} connected parts", count=parts)
    if fragments:
        quality.add(
            "loose_fragments",
            "warning",
            f"{fragments} parts have fewer than {FRAGMENT_TRIANGLES} triangles",
            "Delete stray faces or join them to the garment.",
            count=fragments,
        )


def _components(count: int, triangles: np.ndarray) -> np.ndarray:
    from scipy.sparse.csgraph import connected_components

    from opensculptboy.wearables.geometry import adjacency

    _, labels = connected_components(adjacency(count, triangles), directed=False)
    return labels


def weld_ids(vertices: np.ndarray) -> np.ndarray:
    keys = np.round(np.asarray(vertices) / WELD).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    return np.asarray(inverse).reshape(-1)


def fix_winding(garment: Garment, quality: Quality, weld_id: np.ndarray) -> np.ndarray:
    """Turn over the triangles wound against the most of their part (on the mesh welded by
    position, across manifold edges); returns the triangles turned over (bool)."""
    t = weld_id[garment.triangles]
    e, tri, key = _edges(t)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks = key[order]
    same = (ks[1:] == ks[:-1]).all(axis=1)
    starts = np.flatnonzero(np.concatenate([[True], ~same]))
    lengths = np.diff(np.concatenate([starts, [len(ks)]]))
    pairs = starts[lengths == 2]
    a, b = order[pairs], order[pairs + 1]
    ta, tb = tri[a], tri[b]
    # the two triangles agree when they run their shared edge in opposite directions
    agree = (e[a] == e[b][:, ::-1]).all(axis=1)
    n = len(t)
    neighbours: list[list] = [[] for _ in range(n)]
    for x, y, ok in zip(ta.tolist(), tb.tolist(), agree.tolist()):
        if x != y:
            neighbours[x].append((y, ok))
            neighbours[y].append((x, ok))
    flip = np.zeros(n, bool)
    seen = np.zeros(n, bool)
    for seed in range(n):
        if seen[seed]:
            continue
        part = [seed]
        seen[seed] = True
        stack = [seed]
        while stack:
            x = stack.pop()
            for y, ok in neighbours[x]:
                if not seen[y]:
                    seen[y] = True
                    flip[y] = flip[x] if ok else not flip[x]
                    part.append(y)
                    stack.append(y)
        part = np.array(part)
        if flip[part].sum() * 2 > len(part):
            flip[part] = ~flip[part]
    if flip.any():
        garment.triangles[flip] = garment.triangles[flip][:, ::-1]
        quality.fix(
            "winding_fixed",
            f"{int(flip.sum())} triangles wound against their neighbours turned over",
            count=int(flip.sum()),
        )
    return flip


def angle_weighted_normals(
    vertices: np.ndarray, triangles: np.ndarray, weld_id: np.ndarray | None = None
) -> np.ndarray:
    """Unit vertex normals, each corner weighted by its angle, summed over the vertices of
    one position (``weld_id``), so that UV seams and material borders stay smooth."""
    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)
    face = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    length = np.linalg.norm(face, axis=1, keepdims=True)
    face = face / np.where(length > 0, length, 1.0)
    ids = np.arange(len(v)) if weld_id is None else weld_id
    total = np.zeros((int(ids.max(initial=-1)) + 1, 3))
    for i in range(3):
        p, q, r = v[t[:, i]], v[t[:, (i + 1) % 3]], v[t[:, (i + 2) % 3]]
        e1, e2 = q - p, r - p
        cos = np.einsum("ij,ij->i", e1, e2) / np.maximum(
            np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1), 1e-30
        )
        angle = np.arccos(np.clip(cos, -1.0, 1.0))
        np.add.at(total, ids[t[:, i]], face * angle[:, None])
    out = total[ids]
    length = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(length > 0, length, 1.0)


def check_normals(
    garment: Garment,
    quality: Quality,
    weld_id: np.ndarray,
    normals_from_file: bool,
    flipped: np.ndarray,
    body=None,
) -> None:
    """Compute missing normals; report hard edges, normals against the winding and parts
    that face into the body."""
    if not normals_from_file:
        garment.normals = angle_weighted_normals(
            garment.vertices, garment.triangles, weld_id
        )
        quality.fix(
            "normals_computed",
            "the file has no normals; computed from the triangles, weighted by angle, "
            "smooth across UV seams and material borders",
        )
    else:
        if flipped.any():
            # the triangles turned over keep the file's normals: they agree with them now
            pass
        groups = np.zeros((int(weld_id.max(initial=-1)) + 1, 3))
        np.add.at(groups, weld_id, garment.normals)
        count = np.bincount(weld_id).astype(np.float64)
        mean = groups[weld_id] / count[weld_id, None]
        spread = np.einsum("ij,ij->i", mean, garment.normals) / np.maximum(
            np.linalg.norm(mean, axis=1), 1e-12
        )
        hard = int(((count[weld_id] > 1) & (spread < 0.95)).sum())
        if hard:
            quality.add(
                "hard_edges",
                "info",
                f"{hard} split vertices carry differing normals (hard edges); kept",
                "Smooth the normals at UV seams and material borders in the modelling "
                "tool if those edges should look round.",
                count=hard,
            )
        v, t = garment.vertices, garment.triangles
        face = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
        corner = garment.normals[t].sum(axis=1)
        against = int((np.einsum("ij,ij->i", face, corner) < 0).sum())
        if against:
            quality.add(
                "normals_against_winding",
                "warning",
                f"{against} triangles are wound against their vertex normals",
                "Recalculate the normals outside in the modelling tool, or export "
                "without normals to have them computed.",
                count=against,
            )
    if body is not None and len(garment.triangles):
        found = body.surface.closest(garment.vertices)
        outward = garment.vertices - found.point
        length = np.linalg.norm(outward, axis=1, keepdims=True)
        near = length[:, 0] < 1e-4
        bn = body.normals[body.triangles[found.triangle]]
        bn = np.einsum("nk,nkd->nd", found.barycentric, bn)
        outward = np.where(near[:, None], bn, outward / np.maximum(length, 1e-12))
        v, t = garment.vertices, garment.triangles
        face = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
        score = np.einsum("ij,ij->i", face, outward[t].mean(axis=1))
        labels = _components(int(weld_id.max(initial=-1)) + 1, weld_id[t])[
            weld_id[t[:, 0]]
        ]
        inward_parts = 0
        inward_triangles = 0
        for label in np.unique(labels):
            rows = labels == label
            if score[rows].sum() < 0:
                inward_parts += 1
                inward_triangles += int(rows.sum())
        if inward_parts:
            quality.add(
                "part_faces_inward",
                "warning",
                f"{inward_parts} parts ({inward_triangles} triangles) face into the body",
                "Flip the normals of those parts in the modelling tool, unless they are an "
                "inner lining.",
                count=inward_parts,
            )


# -- UVs --------------------------------------------------------------------------------


def _textured(garment: Garment) -> np.ndarray:
    """(T,) True for triangles whose material has a texture."""
    textured = np.array([bool(m.textures) for m in garment.materials], bool)
    if not len(textured):
        return np.zeros(len(garment.triangles), bool)
    return textured[garment.triangle_material]


def check_uvs(garment: Garment, quality: Quality, weld_id: np.ndarray) -> None:
    """Missing UVs on textured materials (an error), UVs outside 0..1, flipped, zero-area and
    overlapping UV triangles, and the spread of the texel density."""
    textured = _textured(garment)
    if garment.uvs is None:
        if textured.any():
            names = sorted(
                {garment.materials[m].name for m in garment.triangle_material[textured]}
            )
            quality.add(
                "missing_uvs",
                "error",
                f"the materials {names} have textures and the mesh has no UVs",
                "Unwrap the mesh, or export it with UVs.",
            )
        return
    uv, t = garment.uvs, garment.triangles
    if textured.any():
        inside = (uv[t[textured]] >= -1e-4) & (uv[t[textured]] <= 1 + 1e-4)
        outside = int((~inside.all(axis=(1, 2))).sum())
        if outside:
            quality.add(
                "uv_outside_unit_square",
                "warning",
                f"{outside} textured triangles have UVs outside 0..1, where the texture "
                "repeats",
                "Keep the UVs in 0..1 unless the texture tiles.",
                count=outside,
            )
    a, b, c = uv[t[:, 0]], uv[t[:, 1]], uv[t[:, 2]]
    signed = 0.5 * ((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    zero = np.abs(signed) < 1e-10
    if zero.any():
        quality.add(
            "uv_degenerate",
            "warning",
            f"{int(zero.sum())} triangles have UV triangles without area",
            "Unwrap those faces; a texture shows one stretched texel on them.",
            count=int(zero.sum()),
        )
    # flipped UV triangles: the minority sign within a UV chart (mirrored charts keep theirs)
    charts = _components(len(uv), t)[t[:, 0]]
    sign = np.sign(signed)
    flipped = 0
    for chart in np.unique(charts):
        rows = (charts == chart) & ~zero
        pos, neg = int((sign[rows] > 0).sum()), int((sign[rows] < 0).sum())
        flipped += min(pos, neg)
    if flipped:
        quality.add(
            "uv_flipped",
            "warning",
            f"{flipped} UV triangles are mirrored against the rest of their UV island",
            "Unwrap those faces again; a normal map shows them inverted.",
            count=flipped,
        )
    _uv_overlap(garment, signed, zero, quality)
    _texel_density(garment, signed, textured, quality)


def _uv_overlap(garment, signed, zero, quality) -> None:
    from PIL import Image, ImageDraw

    uv, t = garment.uvs, garment.triangles
    size = UV_RASTER
    overlap = mirrored = total = 0.0
    for m in np.unique(garment.triangle_material):
        rows = (garment.triangle_material == m) & ~zero
        if not rows.any():
            continue
        masks = {}
        for name, which in (("pos", signed > 0), ("neg", signed < 0)):
            image = Image.new("1", (size, size), 0)
            draw = ImageDraw.Draw(image)
            for tri in uv[t[rows & which]]:
                p = np.mod(tri, 1.0) if np.ptp(tri) < 1 else tri
                draw.polygon([(float(x * size), float(y * size)) for x, y in p], fill=1)
            masks[name] = np.asarray(image, bool)
            area = float(np.abs(signed[rows & which]).sum()) * size * size
            union = float(masks[name].sum())
            overlap += max(
                area - union - 0.5 * _perimeter(uv, t[rows & which], size), 0.0
            )
            total += area
        mirrored += float((masks["pos"] & masks["neg"]).sum())
    if total <= 0:
        return
    share = overlap / total
    if share > OVERLAP_SHARE:
        quality.add(
            "uv_overlap",
            "warning",
            f"{share:.1%} of the UV area is covered twice in the same direction",
            "Lay the overlapping UV islands apart, unless they share a texture on purpose.",
            value=round(share, 4),
        )
    if mirrored / total > OVERLAP_SHARE:
        quality.add(
            "uv_mirrored_overlap",
            "info",
            f"{mirrored / total:.1%} of the UV area is shared by mirrored islands (a "
            "symmetric garment that reuses one half of the texture)",
            value=round(mirrored / total, 4),
        )


def _perimeter(uv, triangles, size) -> float:
    """The length (pixels) of the outer UV edges, which the raster counts twice."""
    if not len(triangles):
        return 0.0
    _, _, key = _edges(triangles)
    unique, counts = np.unique(key, axis=0, return_counts=True)
    border = unique[counts == 1]
    return (
        float(np.linalg.norm(uv[border[:, 0]] - uv[border[:, 1]], axis=1).sum()) * size
    )


def _texture_size(material) -> tuple[int, int] | None:
    data = material.textures.get("base_color") or next(
        iter(material.textures.values()), None
    )
    if data is None:
        return None
    try:
        from PIL import Image

        return Image.open(io.BytesIO(data)).size
    except Exception:  # noqa: BLE001 - an unreadable texture has an item of its own
        return None


def _texel_density(garment, signed, textured, quality) -> None:
    v, t = garment.vertices, garment.triangles
    area = _areas(v, t)
    density = np.full(len(t), np.nan)
    for m, material in enumerate(garment.materials):
        size = _texture_size(material) if material.textures else None
        if size is None:
            continue
        rows = (garment.triangle_material == m) & textured & (area > 0)
        density[rows] = np.sqrt(np.abs(signed[rows]) * size[0] * size[1] / area[rows])
    ok = np.isfinite(density) & (density > 0)
    if ok.sum() < 10:
        return
    p5, p50, p95 = np.percentile(density[ok], [5, 50, 95])
    spread = float(p95 / p5)
    message = (
        f"texel density {p50 / 100:.0f} texels/cm (median), p95/p5 spread {spread:.1f}"
    )
    if spread > DENSITY_SPREAD:
        quality.add(
            "texel_density_spread",
            "warning",
            message,
            "Scale the UV islands to one texel density, so that the texture looks equally "
            "sharp across the piece.",
            value=round(spread, 2),
        )
    else:
        quality.add("texel_density", "info", message, value=round(spread, 2))


# -- materials and textures -------------------------------------------------------------


def _decode(data: bytes):
    from PIL import Image

    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def _png(image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def green_convention(image) -> float:
    """The correlation of d(red)/d(up) with d(green)/d(right) of a tangent-space normal map:
    positive for the OpenGL convention (green points up the image, as glTF and MakeHuman
    use), negative for DirectX; near 0 when the map shows no relief."""
    rgb = image.convert("RGB")
    rgb.thumbnail((512, 512))
    a = np.asarray(rgb, dtype=np.float64) / 127.5 - 1.0
    nz = np.maximum(a[..., 2], 0.05)
    gx, gy = a[..., 0] / nz, a[..., 1] / nz
    up = -np.diff(gx, axis=0)[:, :-1]  # image rows run down
    right = np.diff(gy, axis=1)[:-1, :]
    up, right = up.ravel(), right.ravel()
    keep = (np.abs(up) + np.abs(right)) > 1e-3
    if keep.sum() < 50:
        return 0.0
    up, right = up[keep], right[keep]
    denom = np.sqrt((up**2).sum() * (right**2).sum())
    return float((up * right).sum() / denom) if denom > 0 else 0.0


def check_materials(
    garment: Garment,
    quality: Quality,
    texture_limit: int = TEXTURE_LIMIT,
    normal_green: str = "auto",
) -> None:
    """Check the textures of every material (readable, size, colour space, alpha, the
    normal map's green channel) and fix the green channel by ``normal_green``."""
    if normal_green not in NORMAL_GREEN:
        raise ValueError(f"normal_green must be one of {NORMAL_GREEN}.")
    from PIL import Image, ImageOps

    used = set(np.unique(garment.triangle_material).tolist())
    count = 0
    for m, material in enumerate(garment.materials):
        if m not in used:
            continue
        for slot in list(material.textures):
            data = material.textures[slot]
            label = dict(material=material.name, slot=slot)
            try:
                image = _decode(data)
            except Exception:  # noqa: BLE001 - any undecodable image is dropped
                del material.textures[slot]
                quality.add(
                    "texture_unreadable",
                    "warning",
                    f"{material.name}: the {slot} texture is not a readable image; dropped",
                    "Save the texture as PNG or JPEG.",
                    fixed=True,
                    **label,
                )
                continue
            count += 1
            w, h = image.size
            if w & (w - 1) or h & (h - 1):
                quality.add(
                    "texture_not_power_of_two",
                    "warning",
                    f"{material.name}: the {slot} texture is {w} x {h}, not a power of two",
                    "Resize it to a power of two (512, 1024, 2048) for mipmaps on every "
                    "engine.",
                    **label,
                )
            if max(w, h) > texture_limit:
                quality.add(
                    "texture_too_large",
                    "warning",
                    f"{material.name}: the {slot} texture is {w} x {h}, above the limit "
                    f"{texture_limit}",
                    "Resize the texture, or raise --texture-limit.",
                    value=max(w, h),
                    **label,
                )
            info = image.info
            tagged = "srgb" in info or "icc_profile" in info
            gamma = info.get("gamma")
            if slot in LINEAR_SLOTS and (tagged or (gamma and abs(gamma - 1.0) > 0.05)):
                quality.add(
                    "linear_texture_tagged_srgb",
                    "info",
                    f"{material.name}: the {slot} texture is tagged as sRGB; glTF reads it "
                    "as linear data",
                    "Save normal, roughness and occlusion maps without a colour profile.",
                    **label,
                )
            if slot == "base_color" and gamma and abs(gamma - 1.0) < 0.05:
                quality.add(
                    "base_colour_tagged_linear",
                    "warning",
                    f"{material.name}: the base colour texture is tagged as linear; glTF "
                    "reads it as sRGB",
                    "Save the base colour texture in sRGB.",
                    **label,
                )
            if slot == "base_color":
                has_alpha = image.mode in ("RGBA", "LA", "PA") or "transparency" in info
                low = False
                if has_alpha:
                    alpha = np.asarray(image.convert("RGBA"))[..., 3]
                    low = bool((alpha < 250).mean() > 0.001)
                if low and material.alpha_mode == "OPAQUE":
                    quality.add(
                        "alpha_unused",
                        "warning",
                        f"{material.name}: the base colour texture has alpha and the "
                        "material is opaque, so the alpha does not show",
                        "Make the material transparent (MASK for cut-outs, BLEND for "
                        "sheer cloth), or save the texture without alpha.",
                        **label,
                    )
            if slot == "normal":
                a = np.asarray(image.convert("RGB"), dtype=np.float64)
                if image.mode in ("L", "I", "F") or a[..., 2].mean() < 160:
                    quality.add(
                        "normal_map_not_tangent_space",
                        "warning",
                        f"{material.name}: the normal map is not a tangent-space map (its "
                        "blue channel is low or it is grey)",
                        "Bake a tangent-space normal map; a bump map has no glTF slot.",
                        **label,
                    )
                    continue
                c = green_convention(image)
                convention = (
                    "OpenGL"
                    if c > GREEN_CONFIDENCE
                    else "DirectX"
                    if c < -GREEN_CONFIDENCE
                    else "unknown"
                )
                flip = normal_green == "flip" or (
                    normal_green == "auto" and convention == "DirectX"
                )
                if flip:
                    r, g, b = image.convert("RGB").split()
                    flipped = Image.merge("RGB", (r, ImageOps.invert(g), b))
                    material.textures[slot] = _png(flipped)
                    quality.fix(
                        "normal_green_flipped",
                        f"{material.name}: the normal map's green channel flipped "
                        f"(found {convention}, correlation {c:+.2f}) to the OpenGL "
                        "convention of glTF",
                        "Pass --normal-green keep to leave it as it is.",
                        value=round(c, 3),
                        **label,
                    )
                elif convention == "DirectX":
                    quality.add(
                        "normal_map_directx",
                        "warning",
                        f"{material.name}: the normal map looks DirectX (green down; "
                        f"correlation {c:+.2f}); glTF expects OpenGL",
                        "Pass --normal-green flip, or flip the green channel.",
                        value=round(c, 3),
                        **label,
                    )
                else:
                    quality.add(
                        "normal_map_convention",
                        "info",
                        f"{material.name}: the normal map's green channel reads {convention}"
                        f" (correlation {c:+.2f})",
                        value=round(c, 3),
                        **label,
                    )
        if material.alpha_mode != "OPAQUE":
            data = material.textures.get("base_color")
            alpha_in_texture = False
            if data is not None:
                try:
                    image = _decode(data)
                    if (
                        image.mode in ("RGBA", "LA", "PA")
                        or "transparency" in image.info
                    ):
                        alpha_in_texture = bool(
                            (np.asarray(image.convert("RGBA"))[..., 3] < 250).any()
                        )
                except Exception:  # noqa: BLE001
                    pass
            if not alpha_in_texture and float(material.base_color[3]) >= 0.999:
                quality.add(
                    "alpha_missing",
                    "warning",
                    f"{material.name}: the material is {material.alpha_mode} and nothing "
                    "gives it alpha",
                    "Add an alpha map, or make the material opaque.",
                    material=material.name,
                )
    if count:
        quality.add(
            "textures_embedded",
            "info",
            f"{count} textures embedded in the package's GLB file",
            count=count,
        )


def check_tangents(garment: Garment, quality: Quality) -> None:
    """MikkTSpace tangents for the materials with a normal map: the packages and the exports
    write them; degenerate tangents (from zero-area UVs) are reported."""
    normal = [m for m, mat in enumerate(garment.materials) if "normal" in mat.textures]
    if not normal or garment.uvs is None:
        return
    from opensculptboy.wearables.gltf import garment_tangents

    tangents = garment_tangents(garment.vertices, garment.normals, garment)
    rows = np.unique(garment.triangles[np.isin(garment.triangle_material, normal)])
    bad = int((~np.isfinite(tangents[rows]).all(axis=1)).sum())
    quality.add(
        "tangents_mikktspace",
        "info",
        "MikkTSpace tangents written for the normal map (the package and the exports)",
    )
    if bad:
        quality.add(
            "tangents_degenerate",
            "warning",
            f"{bad} vertices of the normal-mapped faces have no tangent (UVs without area)",
            "Unwrap those faces.",
            count=bad,
        )


def check_budget(garment: Garment, quality: Quality) -> None:
    """The vertex and triangle counts, and the triangles against the VRM budgets with the
    default body's."""
    from opensculptboy.export.budget import BUDGETS, LABELS

    triangles = int(len(garment.triangles))
    quality.add(
        "counts",
        "info",
        f"{len(garment.vertices)} vertices, {triangles} triangles",
        count=triangles,
    )
    for target, budget in BUDGETS.items():
        total = BODY_TRIANGLES + triangles
        if total > budget["triangles"]:
            quality.add(
                "triangle_budget",
                "warning" if target == "vrm1" else "info",
                f"with the default body ({BODY_TRIANGLES} triangles) the piece makes "
                f"{total} triangles, above the {LABELS[target]} budget of "
                f"{budget['triangles']}",
                "Decimate the piece, or export a lower tier for that format.",
                count=total,
            )


def check_garment(
    garment: Garment,
    *,
    body=None,
    normals_from_file: bool = True,
    texture_limit: int = TEXTURE_LIMIT,
    normal_green: str = "auto",
    quality: Quality | None = None,
) -> tuple[Garment, np.ndarray, Quality]:
    """
    Check and fix a garment (Anny's frame; see the module documentation). ``body`` (a
    :class:`~opensculptboy.wearables.bind.BodySurface`) enables the check of parts that face
    into it. Returns the garment, the old index of each of its vertices (for per-vertex data
    such as own skin weights) and the report.
    """
    quality = Quality() if quality is None else quality
    garment, rows = fix_geometry(garment, quality)
    if not len(garment.triangles):
        quality.add(
            "no_triangles",
            "error",
            "no triangles with area are left",
            "Export the garment's faces.",
        )
        return garment, rows, quality
    weld_id = weld_ids(garment.vertices)
    check_topology(garment, quality, weld_id)
    flipped = fix_winding(garment, quality, weld_id)
    check_normals(garment, quality, weld_id, normals_from_file, flipped, body)
    check_uvs(garment, quality, weld_id)
    check_materials(garment, quality, texture_limit, normal_green)
    check_tangents(garment, quality)
    check_budget(garment, quality)
    return garment, rows, quality


def strip_textures_without_uvs(garment: Garment, quality: Quality) -> None:
    """Drop the textures of a garment without UVs (the fix of an accepted ``missing_uvs``),
    so that the package stays a valid glTF file."""
    if garment.uvs is not None:
        return
    for material in garment.materials:
        if material.textures:
            material.textures = {}
            quality.fix(
                "textures_dropped_without_uvs",
                f"{material.name}: the textures dropped, since the mesh has no UVs",
                material=material.name,
            )

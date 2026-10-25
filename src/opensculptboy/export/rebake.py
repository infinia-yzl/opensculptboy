# OpenSculptBoy
# Apache License, Version 2.0
"""
Normal maps of reduced wearable pieces, baked again from the full pieces.

A reduced piece (:mod:`opensculptboy.export.fit_budget`) keeps a subset of its vertices with
their normals and UVs, so its normal maps still fit its UV layout; where the reduction flattens
a fold or a rim, the interpolated normals of the coarser triangles drift from the full piece's
shading. :func:`rebake_spec` measures that drift per material: for texels of the reduced
primitive, rays along the reduced surface's normal find the nearest point of the full
primitive (:func:`_cast`, with Embree), and the angle between the full piece's shading normal
(its file normals bent by its normal map in its MikkTSpace frames) and the reduced piece's (its
normals bent by the same map in its own frames) is the error. Where the mean error exceeds
:data:`REBAKE_DEGREES`, the material's map is baked again: each reached texel takes the full
piece's shading normal in the reduced piece's MikkTSpace frame. Texels that several triangles
share (mirrored or repeated islands) take the mean of their normals where these agree within
:data:`SHARED_DEGREES`, and keep the map's value elsewhere, as do texels that no ray reaches.
A material without a normal map gets one (flat where nothing is baked) when the bake brings
its shading closer. The bake keeps a map only when it lowers the mean error.
"""

from __future__ import annotations

import numpy as np

REBAKE_DEGREES = 1.0
# the size of a new normal map: half the base colour's, within these bounds
NEW_MAP_SIZE = (256, 1024)
SAMPLES = 20_000
# the texels of mirrored or repeated islands take the mean of their triangles' normals when
# these agree within this angle
SHARED_DEGREES = 10.0


def _texels(uv_triangles: np.ndarray, width: int, height: int, chunk: int = 2_000_000):
    """Every texel centre inside each UV triangle (glTF UVs, v down; only texels within the
    image): rows, cols, triangle, barycentric coordinates and the texel's index, once per
    triangle that covers it."""
    uv = np.asarray(uv_triangles, np.float64)
    x = uv[..., 0] * width - 0.5
    y = uv[..., 1] * height - 0.5
    x0 = np.clip(np.floor(x.min(1)), 0, width - 1).astype(np.int64)
    x1 = np.clip(np.ceil(x.max(1)), 0, width - 1).astype(np.int64)
    y0 = np.clip(np.floor(y.min(1)), 0, height - 1).astype(np.int64)
    y1 = np.clip(np.ceil(y.max(1)), 0, height - 1).astype(np.int64)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    inside_image = (
        (x.max(1) >= 0)
        & (x.min(1) <= width - 1)
        & (y.max(1) >= 0)
        & (y.min(1) <= height - 1)
    )
    rows, cols, tris, barys = [], [], [], []
    order = np.flatnonzero(inside_image)
    cells = (w * h)[order]
    start = 0
    while start < len(order):
        stop = start + max(1, int(np.searchsorted(np.cumsum(cells[start:]), chunk)))
        part = order[start:stop]
        start = stop
        count = w[part] * h[part]
        t = np.repeat(part, count)
        local = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        px = x0[t] + local % w[t]
        py = y0[t] + local // w[t]
        ax, ay, bx, by, cx, cy = x[t, 0], y[t, 0], x[t, 1], y[t, 1], x[t, 2], y[t, 2]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        valid = np.abs(det) > 1e-12
        det = np.where(valid, det, 1.0)
        l0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
        l1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
        l2 = 1.0 - l0 - l1
        hit = valid & (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        rows.append(py[hit])
        cols.append(px[hit])
        tris.append(t[hit])
        barys.append(np.clip(np.stack([l0, l1, l2], 1)[hit], 0.0, 1.0))
    if not rows:
        empty = np.zeros(0, np.int64)
        return empty, empty, empty, np.zeros((0, 3)), empty
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    tris, barys = np.concatenate(tris), np.concatenate(barys)
    barys /= np.maximum(barys.sum(1, keepdims=True), 1e-12)
    return rows, cols, tris, barys, rows * width + cols


def _frames(positions, normals, uvs, triangles):
    """MikkTSpace tangents (N, 4) of a mesh with its file normals (glTF UVs)."""
    from opensculptboy.texture.tangents import mikktspace

    uv_obj = np.stack([uvs[:, 0], 1.0 - uvs[:, 1]], 1)
    return mikktspace(positions, normals, uv_obj, triangles)


def _frame_at(normals, tangents, triangles, tri, bary):
    """The interpolated normal, tangent and bitangent (n, 3) at points of a mesh."""
    from opensculptboy.texture.bake import _unit

    corners = triangles[tri]
    n = _unit(np.einsum("nk,nkd->nd", bary, normals[corners]))
    t = np.einsum("nk,nkd->nd", bary, tangents[corners][..., :3])
    w = np.where(np.einsum("nk,nk->n", bary, tangents[corners][..., 3]) < 0, -1.0, 1.0)
    t = _unit(t - (t * n).sum(1, keepdims=True) * n)
    return n, t, w[:, None] * np.cross(n, t)


def _shading(normals, tangents, triangles, tri, bary, normal_map, scale, uvs):
    """Shading normals (n, 3) at points of a mesh: interpolated normals bent by the map."""
    from opensculptboy.texture.bake import _sample, _unit

    n, t, b = _frame_at(normals, tangents, triangles, tri, bary)
    if normal_map is None:
        return n, t, b
    uv = np.einsum("nk,nkd->nd", bary, uvs[triangles[tri]])
    c = _sample(normal_map, uv)[:, :3] / 127.5 - 1.0
    c[:, :2] *= scale
    c = _unit(c)
    return _unit(c[:, :1] * t + c[:, 1:2] * b + c[:, 2:3] * n), t, b


def _reference(positions, normals, uvs, triangles, normal_map, scale):
    from opensculptboy.fit.handoff import HandoffPart
    from opensculptboy.fit.reference import Reference

    used, local = np.unique(triangles, return_inverse=True)
    local = local.reshape(-1, 3)
    part = HandoffPart(
        name="full",
        role="skin",
        vertices=positions[used],
        triangles=local,
        uvs=uvs[used],
        normal_map=normal_map,
        normals=normals[used],
        normal_scale=float(scale),
    )
    return Reference(
        part.vertices, local, parts=[part], part_offsets=np.array([0, len(used)])
    )


def _cast(points, normals, reference, max_distance, rays, chunk: int = 200_000):
    """The nearest point of ``reference`` along each normal, both ways, within
    ``max_distance``, on a triangle that faces the same way: the rays that hit, their
    triangles and barycentric coordinates. One first hit per direction keeps the cost and the
    memory of rays that cross many layers (fur cards) small."""
    import trimesh

    out_i, out_t, out_l, out_d = [], [], [], []
    for start in range(0, len(points), chunk):
        p = points[start : start + chunk]
        n = normals[start : start + chunk]
        for sign in (1.0, -1.0):
            origins = p - sign * n * 1e-5
            loc, ray, tri = rays.intersects_location(
                origins, sign * n, multiple_hits=False
            )
            gap = np.linalg.norm(loc - p[ray], axis=1)
            facing = (reference.face_normals[tri] * n[ray]).sum(1) > 0.0
            keep = (gap <= max_distance) & facing
            out_i.append(ray[keep] + start)
            out_t.append(tri[keep])
            out_l.append(loc[keep])
            out_d.append(gap[keep])
    if not out_i:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3))
    ids, tris = np.concatenate(out_i), np.concatenate(out_t)
    locs, gaps = np.concatenate(out_l), np.concatenate(out_d)
    order = np.lexsort((gaps, ids))
    ids, first = np.unique(ids[order], return_index=True)
    tris, locs = tris[order][first], locs[order][first]
    bary = trimesh.triangles.points_to_barycentric(
        reference.vertices[reference.triangles[tris]], locs
    )
    bary = np.clip(bary, 0.0, 1.0)
    bary /= np.maximum(bary.sum(1, keepdims=True), 1e-12)
    return ids, tris, bary


def _angles(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(1), -1.0, 1.0)))


def rebake_material(full, reduced, normal_map, scale, size, max_distance, seed=0):
    """
    Measure and bake one material. ``full`` and ``reduced`` are (positions, normals, uvs,
    triangles) of the material's primitive; ``normal_map`` (H, W, 3) uint8 or None, of
    ``size`` (w, h). Returns (the new map or None, the stats).
    """
    from opensculptboy.texture.bake import _intersector, _shading_normals, _unit

    P, N, UV, T = reduced
    reference = _reference(*full, normal_map, scale)
    tangents = _frames(P, N, UV, T)
    width, height = size
    rows, cols, tri, bary, texel = _texels(UV[T], width, height)
    if len(tri) == 0:
        return None, {"texels": 0}
    corners = T[tri]
    points = np.einsum("nk,nkd->nd", bary, P[corners])
    n = _unit(np.einsum("nk,nkd->nd", bary, N[corners]))
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(tri), min(SAMPLES, len(tri)), replace=False)
    rays = _intersector(reference.vertices, reference.triangles)
    ids, hit_tri, hit_bary = _cast(
        points[sample], n[sample], reference, max_distance, rays
    )
    if len(ids) == 0:
        return None, {"texels": int(len(np.unique(texel))), "reached": 0.0}
    target = _shading_normals(reference, hit_tri, hit_bary)
    current, _, _ = _shading(
        N, tangents, T, tri[sample][ids], bary[sample][ids], normal_map, scale, UV
    )
    before = _angles(current, target)
    sample_ids, sample_target = ids, target
    stats = {
        "texels": int(len(np.unique(texel))),
        "reached": float(len(ids) / len(sample)),
        "before_deg": float(before.mean()),
    }
    if before.mean() <= REBAKE_DEGREES:
        return None, stats
    # the full piece's normal in the reduced frame at every fragment that a ray reaches
    ids, hit_tri, hit_bary = _cast(points, n, reference, max_distance, rays)
    target = _shading_normals(reference, hit_tri, hit_bary)
    nn, t, b = _frame_at(N, tangents, T, tri[ids], bary[ids])
    c = _unit(
        np.stack([(target * t).sum(1), (target * b).sum(1), (target * nn).sum(1)], 1)
    )
    # a texel that several triangles cover (mirrored or repeated islands) takes their mean
    # where they agree within SHARED_DEGREES, and keeps its value where they do not
    key = texel[ids]
    unique, inverse = np.unique(key, return_inverse=True)
    mean = np.zeros((len(unique), 3))
    np.add.at(mean, inverse, c)
    mean = _unit(mean)
    agree = np.ones(len(unique), bool)
    np.logical_and.at(
        agree, inverse, (c * mean[inverse]).sum(1) >= np.cos(np.radians(SHARED_DEGREES))
    )
    covered = np.zeros(len(unique), np.int64)
    np.add.at(covered, inverse, 1)
    # every fragment of a texel must be reached: one unreached side keeps the texel
    all_fragments = np.zeros(len(unique), np.int64)
    pos = np.searchsorted(unique, texel)
    inside = (pos < len(unique)) & (unique[np.minimum(pos, len(unique) - 1)] == texel)
    np.add.at(all_fragments, pos[inside], 1)
    agree &= covered == all_fragments
    if normal_map is None:
        image = np.zeros((height, width, 3), np.uint8)
        image[...] = (128, 128, 255)
    else:
        image = np.array(normal_map[..., :3], np.uint8, copy=True)
        if scale != 1.0:  # the new map holds the true vectors: scale 1
            v = image.astype(np.float64) / 127.5 - 1.0
            v[..., :2] *= scale
            v /= np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
            image = np.clip(np.round((v + 1.0) * 127.5), 0, 255).astype(np.uint8)
    write = unique[agree]
    image[write // width, write % width] = np.clip(
        np.round((mean[agree] + 1.0) * 127.5), 0, 255
    ).astype(np.uint8)
    # the error after, at the same samples and their hits on the full piece
    after_n, _, _ = _shading(
        N,
        tangents,
        T,
        tri[sample][sample_ids],
        bary[sample][sample_ids],
        image,
        1.0,
        UV,
    )
    target2 = sample_target
    after = _angles(after_n, target2)
    stats["after_deg"] = float(after.mean())
    stats["baked_texels"] = int(agree.sum())
    if after.mean() >= 0.95 * before.mean():
        return None, stats
    return image, stats


def rebake_spec(full_spec, spec) -> dict:
    """
    Bake again the normal maps of the reduced wearable pieces of ``spec`` from ``full_spec``
    (the same spec before the reduction), per material. Changes ``spec`` in place (new images
    are added, and a material shared with another piece is not touched); returns per piece the
    stats of each material. Without Embree (the viewer extra) it measures nothing and says so.
    """
    from opensculptboy.export.document import png_bytes
    from opensculptboy.export.pieces import primitive_triangles
    from opensculptboy.texture.transfer import decode_image

    try:
        import trimesh.ray.ray_pyembree  # noqa: F401
    except ImportError:
        return {
            "": {"skipped": "the bake needs Embree: install the viewer extra (embreex)"}
        }
    from opensculptboy.export.budget import image_size

    users = {}
    for mesh in spec.meshes:
        for p in mesh.primitives:
            users.setdefault(p.material, set()).add(mesh.name)
    image_users = {}
    for k, m in enumerate(spec.materials):
        if m.normal_texture is not None:
            image_users.setdefault(m.normal_texture, set()).add(k)
    out = {}
    full_meshes = {m.name: m for m in full_spec.meshes}
    for mesh in spec.meshes[1:]:
        old = full_meshes.get(mesh.name)
        if old is None or old.body.uvs is None or mesh.body.uvs is None:
            continue
        stats = {}
        before_tris = {p.material: primitive_triangles(old, p) for p in old.primitives}
        for primitive in mesh.primitives:
            material = spec.materials[primitive.material]
            full_t = before_tris.get(primitive.material)
            red_t = primitive_triangles(mesh, primitive)
            if (
                full_t is None
                or len(full_t) == len(red_t)
                or material.texture_transform is not None
            ):
                continue
            if len(users.get(primitive.material, ())) > 1:
                continue
            if material.normal_texture is not None:
                data = spec.images[material.normal_texture][1]
                normal_map = decode_image(data)
                if normal_map.ndim == 2:
                    continue
                size = image_size(data)
            else:
                normal_map = None
                if material.base_texture is None:
                    continue
                bw, bh = image_size(spec.images[material.base_texture][1])
                lo, hi = NEW_MAP_SIZE
                size = (int(np.clip(bw // 2, lo, hi)), int(np.clip(bh // 2, lo, hi)))
            full = (
                np.asarray(old.body.positions),
                np.asarray(old.body.normals),
                np.asarray(old.body.uvs),
                full_t,
            )
            reduced = (
                np.asarray(mesh.body.positions),
                np.asarray(mesh.body.normals),
                np.asarray(mesh.body.uvs),
                red_t,
            )
            image, info = rebake_material(
                full, reduced, normal_map, float(material.normal_scale), size, 0.03
            )
            if image is not None:
                data = png_bytes(image)
                shared = (
                    material.normal_texture is not None
                    and len(image_users.get(material.normal_texture, ())) > 1
                )
                if material.normal_texture is None or shared:
                    material.normal_texture = len(spec.images)
                    spec.images.append(
                        (f"{mesh.name}:{material.name}:normal(baked)", data)
                    )
                else:
                    name = spec.images[material.normal_texture][0]
                    spec.images[material.normal_texture] = (name, data)
                material.normal_scale = 1.0
                info["baked"] = True
            stats[material.name] = info
        out[mesh.name] = {
            "materials": stats,
            "baked": any(v.get("baked") for v in stats.values()),
        }
    return out

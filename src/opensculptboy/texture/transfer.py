# OpenSculptBoy
# Apache License, Version 2.0
"""
Texture transfer from a master mesh onto the coarser meshes of the rendering-budget tiers.

The tiers of :mod:`opensculptboy.tiers` cut down from one master: the finest level of a card's
detail layer with its own maps. Every level of the detail layer lays out anny's atlas alike
(:mod:`opensculptboy.detail.subdivision`), so a map of the surface's colour or of a property of
the master's geometry (its base colour, roughness, thickness, curvature and ambient occlusion)
serves every tier as it is, resized. A normal map does not: it holds the shading normal in the
tangent frames of the mesh it was baked on, and a coarser mesh has other normals (and lacks the
master's pores and folds). :func:`bake_normals` therefore bakes the master's shading normal onto
the coarser mesh: for each texel of that mesh, a ray along its normal finds the master's surface
(the texture bake of :mod:`opensculptboy.texture.bake`, with the master as its reference), and
the master's shading normal there (its welded vertex normals, bent by the master's own normal
maps in its MikkTSpace frames) is written in the MikkTSpace frame of the coarser mesh, as the
exporters write its ``TANGENT``. :func:`master_reference` turns the master mesh and its maps
into that reference: one part per map (the atlas, the head tile and the ear tile, each on its
own UVs), so that the bake reads each map where the exporters draw it.

:func:`vertex_occlusion` gives the ambient occlusion of a mesh at its vertices (cosine-weighted
rays over the hemisphere, the share that leaves within ``distance`` falling linearly), and
:func:`atlas_map` lays any per-vertex value of a mesh onto anny's atlas, as the thickness and
curvature maps of the bake are laid.
"""

from __future__ import annotations

import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    """The pixels (H, W) or (H, W, 3) uint8 of a PNG or JPEG file, row 0 at the top."""
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(bytes(data)))
    if image.mode in ("L", "I", "I;16", "F"):
        return np.asarray(image.convert("L"))
    return np.asarray(image.convert("RGB"))


def _part(name, P, N, uv_obj, triangles, normal_map, normal_scale):
    """A skin part of the master's split mesh: the triangles' vertices, renumbered."""
    from opensculptboy.fit.handoff import HandoffPart

    used, local = np.unique(triangles, return_inverse=True)
    return HandoffPart(
        name=name,
        role="skin",
        vertices=P[used],
        triangles=local.reshape(-1, 3),
        uvs=np.stack([uv_obj[used, 0], 1.0 - uv_obj[used, 1]], 1),  # glTF: v down
        normal_map=normal_map,
        normals=N[used],
        normal_scale=float(normal_scale),
    )


def master_reference(
    vertices: np.ndarray,
    topology,
    images: dict | None = None,
    model=None,
    character_topology: str = "anny",
    ear_table=None,
    normal_scale: float = 1.0,
):
    """
    The master mesh as the reference of a bake (:class:`~opensculptboy.fit.reference.Reference`).

    Args:
        vertices: (N, 3) the master's rest mesh, Anny's frame.
        topology: its :class:`~opensculptboy.detail.subdivision.DetailTopology`.
        images: the master's maps by slot (PNG or JPEG bytes): ``normal`` on anny's atlas,
            ``head_normal`` on the head tile and ``ears_normal`` on the ear tile; each one
            bends the shading normal of its part. Without them, the shading normal is the
            master's welded vertex normal (its geometry alone).
        model: the Anny model (for the head tile's island); needed with ``head_normal``.
        character_topology: the card's topology name.
        ear_table: the ear tile's UVs (:class:`~opensculptboy.texture.ears.EarUVs`); needed
            with ``ears_normal``.
        normal_scale: the scale of the master's normal textures.
    """
    from opensculptboy.fit.reference import Reference, vertex_normals
    from opensculptboy.texture.tangents import split_mesh

    images = images or {}
    V = np.asarray(vertices, np.float64)
    T = np.asarray(topology.triangles, np.int64)
    UT = np.asarray(topology.uv_triangles, np.int64)
    UV = np.asarray(topology.uvs, np.float64)
    ears = np.zeros(len(T), bool)
    if "ears_normal" in images and ear_table is not None:
        from opensculptboy.texture.ears import with_ear_uvs

        mesh, mask = with_ear_uvs(topology, ear_table)
        if mask is not None:
            UT, UV, ears = (
                np.asarray(mesh.uv_triangles, np.int64),
                np.asarray(mesh.uvs, np.float64),
                mask,
            )
    head = np.zeros(len(T), bool)
    box = None
    if "head_normal" in images and model is not None:
        from opensculptboy.texture.tiles import head_uv_box, on_head_tile

        box = head_uv_box(model, character_topology)
        found = on_head_tile(model, UV, UT, character_topology)
        if box is not None and found is not None:
            head = found & ~ears
    N = vertex_normals(V, T)
    source, uv_source, split = split_mesh(T, UT)
    P, Nw, uv = V[source], N[source], UV[uv_source]

    def image(slot):
        return decode_image(images[slot]) if slot in images else None

    parts = []
    body = ~(head | ears)
    if body.any():
        parts.append(
            _part("atlas", P, Nw, uv, split[body], image("normal"), normal_scale)
        )
    if head.any():
        from opensculptboy.texture.tiles import to_tile_uvs

        parts.append(
            _part(
                "head",
                P,
                Nw,
                to_tile_uvs(uv, box),
                split[head],
                image("head_normal"),
                normal_scale,
            )
        )
    if ears.any():
        parts.append(
            _part("ears", P, Nw, uv, split[ears], image("ears_normal"), normal_scale)
        )
    offsets = np.cumsum([0] + [len(p.vertices) for p in parts])
    return Reference(
        np.concatenate([p.vertices for p in parts]),
        np.concatenate([p.triangles + offsets[k] for k, p in enumerate(parts)]),
        parts=parts,
        part_offsets=offsets,
    )


def bake_normals(
    reference,
    vertices: np.ndarray,
    topology,
    size: int,
    model=None,
    character_topology: str = "anny",
    head_size: int = 0,
    ear_table=None,
    max_distance: float = 0.006,
    supersample: int = 1,
) -> dict:
    """
    The master's shading normals (``reference``, :func:`master_reference`) baked onto a
    coarser mesh, as tangent-space normal maps by slot: ``normal`` (anny's atlas, ``size``
    texels square), ``head_normal`` (the head tile, ``head_size`` texels high, when given) and
    ``ears_normal`` (the ear tile of :mod:`opensculptboy.texture.ears`, when ``ear_table`` is
    given and the mesh is level 1 or 2). Each is (H, W, 3) uint8, row 0 at the top. Rays reach
    ``max_distance`` (metres) on each side of the coarser mesh; texels that none reaches take
    the nearest reached value. Also returns the share of the atlas's texels that a ray reached
    under ``"covered"``.
    """
    from opensculptboy.texture.bake import bake

    kwargs = {}
    if head_size and model is not None:
        from opensculptboy.texture.tiles import head_uv_box, on_head_tile

        kwargs.update(
            head_box=head_uv_box(model, character_topology),
            head_triangles=on_head_tile(
                model, topology.uvs, topology.uv_triangles, character_topology
            ),
            head_size=int(head_size),
        )
    if ear_table is not None and topology.subdiv > 0:
        from opensculptboy.texture.ears import with_ear_uvs

        mesh, mask = with_ear_uvs(topology, ear_table)
        if mask is not None:
            kwargs["ear_mesh"] = (mesh.uv_triangles, mesh.uvs, mask)
    baked = bake(
        np.asarray(vertices, np.float64),
        topology.triangles,
        topology.uv_triangles,
        topology.uvs,
        reference,
        int(size),
        max_distance=max_distance,
        supersample=supersample,
        **kwargs,
    )
    out = {"normal": baked.normal, "covered": baked.covered_share}
    if "normal" in baked.head:
        out["head_normal"] = baked.head["normal"]
    if "normal" in baked.ears:
        out["ears_normal"] = baked.ears["normal"]
    return out


def _hemisphere(count: int) -> np.ndarray:
    """``count`` cosine-weighted directions (count, 3) over the hemisphere about +Z: a
    Fibonacci spiral on the unit disk, lifted onto the hemisphere (Malley's method)."""
    k = np.arange(count) + 0.5
    radius = np.sqrt(k / count)
    angle = np.pi * (3.0 - np.sqrt(5.0)) * k
    x, y = radius * np.cos(angle), radius * np.sin(angle)
    return np.stack([x, y, np.sqrt(np.maximum(1.0 - x * x - y * y, 0.0))], 1)


def vertex_occlusion(
    vertices: np.ndarray,
    triangles: np.ndarray,
    rays: int = 32,
    distance: float = 0.05,
    chunk: int = 1 << 19,
    seed: int = 0,
) -> np.ndarray:
    """
    The ambient occlusion (V,) of a mesh at its vertices: 1 where the hemisphere above a
    vertex is open, 0 where it is shut. ``rays`` cosine-weighted directions about the vertex
    normal (one spiral, turned by a random angle per vertex so that the pattern does not
    band) leave from just above the surface; a ray that meets the mesh at ``t`` occludes by
    ``1 - t / distance`` within ``distance`` metres. Needs Embree (the ``viewer`` extra).
    """
    from opensculptboy.fit.reference import vertex_normals
    from opensculptboy.texture.bake import _intersector, _unit

    V = np.asarray(vertices, np.float64)
    T = np.asarray(triangles, np.int64)
    N = vertex_normals(V, T)
    helper = np.eye(3)[np.abs(N).argmin(1)]
    u = _unit(np.cross(N, helper))
    v = np.cross(N, u)
    turn = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, len(V))
    c, s = np.cos(turn)[:, None], np.sin(turn)[:, None]
    u, v = c * u + s * v, -s * u + c * v
    pattern = _hemisphere(int(rays))
    intersector = _intersector(V, T)
    occluded = np.zeros(len(V))
    eps = 1e-4
    per = max(1, chunk // len(pattern))
    for start in range(0, len(V), per):
        rows = np.arange(start, min(start + per, len(V)))
        d = (
            pattern[None, :, 0:1] * u[rows, None]
            + pattern[None, :, 1:2] * v[rows, None]
            + pattern[None, :, 2:3] * N[rows, None]
        ).reshape(-1, 3)
        origins = np.repeat(V[rows] + N[rows] * eps, len(pattern), axis=0)
        hit, ray_ids, _ = intersector.intersects_location(
            origins, d, multiple_hits=False
        )
        gap = np.linalg.norm(hit - origins[ray_ids], axis=1)
        amount = np.clip(1.0 - gap / distance, 0.0, 1.0)
        occluded += np.bincount(ray_ids // len(pattern) + start, amount, len(V))
    return 1.0 - occluded / len(pattern)


def atlas_map(values: np.ndarray, topology, size: int) -> np.ndarray:
    """A per-vertex value (V,) of a detail mesh laid onto anny's atlas: (size, size) floats,
    interpolated over each UV triangle, the texels outside every triangle taking the nearest
    value inside one."""
    from opensculptboy.texture.bake import _vertex_map, rasterise

    rows, cols, tri, bary = rasterise(topology.uv_triangles, topology.uvs, int(size))
    return _vertex_map(
        np.asarray(values, np.float64),
        rows,
        cols,
        tri,
        bary,
        np.asarray(topology.triangles),
        int(size),
    )


def grey(image: np.ndarray) -> np.ndarray:
    """A float image in [0, 1] as uint8 grey pixels."""
    return np.clip(np.round(np.asarray(image) * 255.0), 0, 255).astype(np.uint8)


def _sample_map(image: np.ndarray, uv_obj: np.ndarray) -> np.ndarray:
    """A tangent-space normal map (H, W, 3) uint8 at UVs (OBJ convention), bilinear, as unit
    vectors."""
    from opensculptboy.texture.bake import _sample, _unit

    uv = np.stack([uv_obj[:, 0], 1.0 - uv_obj[:, 1]], 1)
    return _unit(_sample(image, uv)[:, :3] / 127.5 - 1.0)


def normal_error(
    reference,
    vertices: np.ndarray,
    topology,
    maps: dict | None = None,
    model=None,
    character_topology: str = "anny",
    samples: int = 20000,
    exclude: np.ndarray | None = None,
    max_distance: float = 0.006,
    seed: int = 0,
) -> np.ndarray:
    """
    The angles (degrees) between the master's shading normals (``reference``,
    :func:`master_reference`) and those of a coarser mesh, at ``samples`` random points of the
    coarser mesh (uniform over its area, ``exclude`` masking triangles) whose ray along the
    normal meets the master within ``max_distance``. The coarser mesh's shading normal is its
    interpolated vertex normal, bent by its normal maps when ``maps`` gives them (``normal``
    on anny's atlas, ``head_normal`` on the head tile, as (H, W, 3) uint8 images) in its
    MikkTSpace frames, as an engine reads them from the exported file.
    """
    from opensculptboy.texture.bake import _hits, _shading_normals, _Surface, _unit

    maps = maps or {}
    V = np.asarray(vertices, np.float64)
    T = np.asarray(topology.triangles, np.int64)
    UT = np.asarray(topology.uv_triangles, np.int64)
    UV = np.asarray(topology.uvs, np.float64)
    surface = _Surface.of(V, T, UT, UV)
    p = V[T]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    if exclude is not None:
        area = np.where(np.asarray(exclude, bool), 0.0, area)
    rng = np.random.default_rng(seed)
    tri = rng.choice(len(T), samples, p=area / area.sum())
    r1, r2 = rng.random(samples), rng.random(samples)
    s = np.sqrt(r1)
    bary = np.stack([1.0 - s, s * (1.0 - r2), s * r2], 1)
    points, n, t, b = surface.at(tri, bary)
    ids, hit_tri, hit_bary = _hits(points, n, reference, max_distance)
    master = _shading_normals(reference, hit_tri, hit_bary)
    shading = n[ids]
    uv = np.einsum("nk, nkd -> nd", bary[ids], UV[UT[tri[ids]]])
    head = np.zeros(len(ids), bool)
    if "head_normal" in maps and model is not None:
        from opensculptboy.texture.tiles import head_uv_box, on_head_tile, to_tile_uvs

        box = head_uv_box(model, character_topology)
        on_head = on_head_tile(model, UV, UT, character_topology)
        if box is not None and on_head is not None:
            head = on_head[tri[ids]]
            c = _sample_map(maps["head_normal"], to_tile_uvs(uv[head], box))
            k = ids[head]
            shading[head] = _unit(c[:, :1] * t[k] + c[:, 1:2] * b[k] + c[:, 2:3] * n[k])
    if "normal" in maps:
        rest = ~head
        c = _sample_map(maps["normal"], uv[rest])
        k = ids[rest]
        shading[rest] = _unit(c[:, :1] * t[k] + c[:, 1:2] * b[k] + c[:, 2:3] * n[k])
    cosine = np.clip((shading * master).sum(1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

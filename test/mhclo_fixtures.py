# OpenSculptBoy
# Apache License, Version 2.0
"""
Synthetic MakeHuman clothes assets for the import tests, written by the tests: a shirt-like
tube around the torso bound to the triangles of the base mesh (three indices, weights and an
offset per vertex, with scale references), and a patch of the ``helper-tights`` geometry
bound vertex by vertex (single-index bindings), each with an ``.obj`` mesh and an ``.mhmat``
material with small textures. The defects of the quality tests are options of the writers.
``makeclothes`` writes the header as MakeClothes exports it, and :func:`write_pack` writes the
folder layout of a MakeHuman asset pack (``clothes/<item>/`` and ``packs/<pack>.json``).
"""

from __future__ import annotations

import io
import json
import pathlib

import numpy as np

from opensculptboy.wearables.basemesh import MH_TO_ANNY, _full_targets, base_groups
from opensculptboy.wearables.basemesh import base_indices
from opensculptboy.wearables.fit import skin_triangles
from opensculptboy.wearables.geometry import Surface
from test.wearable_fixtures import template_model

TO_MH = np.linalg.inv(MH_TO_ANNY)


def png(size=64, mode="RGB", colour=(180, 120, 60), alpha=None) -> bytes:
    from PIL import Image

    image = Image.new(mode, (size, size), colour if mode != "RGBA" else (*colour, 255))
    if alpha is not None:
        image.putalpha(alpha)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def normal_map(size=128, directx=False) -> bytes:
    """A tangent-space normal map of round bumps (OpenGL convention: green points up the
    image), or its DirectX form (green flipped)."""
    from PIL import Image

    y, x = np.mgrid[0:size, 0:size] / size
    height = np.sin(6 * np.pi * x) * np.sin(4 * np.pi * y) + 0.5 * np.sin(
        10 * np.pi * x * y
    )
    dx = np.gradient(height, axis=1) * size / 20
    d_up = -np.gradient(height, axis=0) * size / 20  # image rows run down
    n = np.stack([-dx, -d_up, np.ones_like(dx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    if directx:
        n[..., 1] *= -1
    rgb = np.clip((n * 0.5 + 0.5) * 255 + 0.5, 0, 255).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(out, format="PNG")
    return out.getvalue()


def height_map(size=128) -> bytes:
    """A grey height map of round bumps (white is high), as MakeHuman's bump maps."""
    from PIL import Image

    y, x = np.mgrid[0:size, 0:size] / size
    height = 0.5 + 0.25 * np.sin(6 * np.pi * x) * np.sin(4 * np.pi * y)
    grey = np.clip(height * 255 + 0.5, 0, 255).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(grey, "L").save(out, format="PNG")
    return out.getvalue()


def template_skin():
    """Anny's template skin (base.obj in Anny's frame) and the base-mesh index of each row."""
    model = template_model()
    vertices = model.template_vertices.detach().cpu().double().numpy()
    triangles = skin_triangles(model)
    n = int(triangles.max()) + 1
    return vertices[:n], triangles, base_indices(model, len(vertices))


def tube_on_template(rings=10, around=32, gap=0.012):
    """A tube around the template's torso, ``gap`` off its widest skin (Anny's frame)."""
    from test.test_wearables_bind import tube

    vertices, _, _ = template_skin()

    class Body:
        pass

    body = Body()
    body.vertices = vertices
    v, _, t, _ = tube(body, rings=rings, around=around, gap=gap)
    return v, t


def write_asset(
    folder,
    name,
    *,
    licence="CC0",
    author="OpenSculptBoy tests",
    tags=("shirt", "long sleeves"),
    z_depth=50,
    uvs=True,
    textures=True,
    flip_triangle=False,
    directx=False,
    missing_texture=False,
    alpha_map_opaque=False,
    texture_size=64,
    helper=False,
    delete_verts=True,
    description="A synthetic shirt for the tests",
    makeclothes=False,
    with_normal_map=True,
    bump=None,
    material_lines=(),
) -> pathlib.Path:
    """Write ``name.mhclo``, ``name.obj``, ``name.mhmat`` and textures into ``folder`` and
    return the ``.mhclo`` path. ``helper`` writes the helper-tights patch instead of the
    torso tube."""
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    skin, triangles, base = template_skin()
    if helper:
        rows, faces, positions = _tights_patch()
        lines = [str(int(r)) for r in rows]
        polygons = faces
        mh = positions
        scale_lines = []
        deleted = []
    else:
        vertices, tris = tube_on_template()
        found = Surface(skin, triangles).closest(vertices)
        corners = base[triangles[found.triangle]]
        offsets = (vertices - found.point) @ TO_MH.T
        lines = [
            " ".join(str(int(c)) for c in corners[k])
            + " "
            + " ".join(f"{w:.9f}" for w in found.barycentric[k])
            + " "
            + " ".join(f"{d:.9f}" for d in offsets[k])
            for k in range(len(vertices))
        ]
        mh = vertices @ TO_MH.T
        # open the tube along one line, so that its UVs unwrap without a seam
        around = 32
        column = np.arange(len(vertices)) % around
        cut = (column[tris] == around - 1).any(1) & (column[tris] == 0).any(1)
        polygons = [list(t) for t in tris[~cut]]
        h = skin @ TO_MH.T
        scale_lines = []
        for n, axis in enumerate("xyz"):
            a, b = int(np.argmin(h[:, n])), int(np.argmax(h[:, n]))
            scale_lines.append(
                f"{axis}_scale {base[a]} {base[b]} {abs(h[a, n] - h[b, n]):.6f}"
            )
        torso = np.linalg.norm(skin[:, :2] - vertices[:, :2].mean(0), axis=1) < 0.2
        torso &= (skin[:, 2] > vertices[:, 2].min()) & (
            skin[:, 2] < vertices[:, 2].max()
        )
        deleted = sorted(base[np.flatnonzero(torso)].tolist())
    if flip_triangle:
        polygons[3] = polygons[3][::-1]
    count = len(mh)
    span = np.ptp(mh[:, :2], axis=0)
    uv = 0.05 + 0.9 * (mh[:, :2] - mh[:, :2].min(axis=0)) / np.where(span > 0, span, 1)
    if not helper:  # a cylinder unwrap of the tube
        around = 32
        k = np.arange(count)
        uv = np.stack(
            [(k % around) / (around - 1) * 0.9 + 0.05, (k // around) / 9 * 0.9 + 0.05],
            1,
        )
    obj = [f"# {name}"] + [f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in mh]
    if uvs:
        obj += [f"vt {u:.6f} {v:.6f}" for u, v in uv]
        obj += ["f " + " ".join(f"{i + 1}/{i + 1}" for i in poly) for poly in polygons]
    else:
        obj += ["f " + " ".join(f"{i + 1}" for i in poly) for poly in polygons]
    (folder / f"{name}.obj").write_text("\n".join(obj) + "\n")
    mat = [f"# material of {name}", f"name {name}_material", "diffuseColor 0.8 0.7 0.6"]
    mat += ["shininess 0.3", "backfaceCull False"]
    if textures:
        (folder / f"{name}_diffuse.png").write_bytes(png(texture_size))
        mat.append(
            f"diffuseTexture {name}_missing.png"
            if missing_texture
            else f"diffuseTexture {name}_diffuse.png"
        )
        if with_normal_map:
            (folder / f"{name}_normal.png").write_bytes(normal_map(128, directx))
            mat += [f"normalmapTexture {name}_normal.png", "shaderConfig normal true"]
    if (
        bump
    ):  # the key of the bump map: bumpTexture as the real packs, or bumpmapTexture
        (folder / f"{name}-bump.png").write_bytes(height_map(128))
        mat += [f"{bump} {name}-bump.png", "shaderConfig bump true"]
    mat += list(material_lines)
    if alpha_map_opaque:
        (folder / f"{name}_alpha.png").write_bytes(png(64, "L", 128))
        mat.append(f"transparencymapTexture {name}_alpha.png")
    (folder / f"{name}.mhmat").write_text("\n".join(mat) + "\n")
    if makeclothes:  # the order of MakeClothes' header, without a description
        head = ["# Exported from MakeClothes (TM)"]
    else:
        head = ["# Synthetic clothes asset written by the OpenSculptBoy tests"]
    if author:
        head.append(f"# author {author}")
    if licence:
        head.append(f"# license {licence}")
    head.append("# homepage https://example.org/tests")
    uuid = f"uuid 00000000-0000-4000-8000-{count:012d}"
    if makeclothes:
        head += [uuid, "basemesh hm08"] + [f"tag {t}" for t in tags]
        head += [f"name {name}", f"obj_file {name}.obj"] + scale_lines
        head += [f"z_depth {z_depth if z_depth is not None else 50}", "max_pole 5"]
        head += [f"material {name}.mhmat"]
    else:
        head += ["basemesh hm08", f"name {name}", uuid]
        head += [f"tag {t}" for t in tags]
        if description:
            head.append(f"description {description}")
        head += [f"obj_file {name}.obj", f"material {name}.mhmat"]
        if z_depth is not None:
            head.append(f"z_depth {z_depth}")
        head += scale_lines
    head += ["verts 0"] + lines
    if delete_verts and deleted:
        ranges, start = [], deleted[0]
        prev = start
        for v in deleted[1:] + [None]:
            if v is not None and v == prev + 1:
                prev = v
                continue
            ranges.append(f"{start} - {prev}" if prev > start else str(start))
            if v is not None:
                start = prev = v
        head += ["", "delete_verts"] + [
            " ".join(ranges[i : i + 8]) for i in range(0, len(ranges), 8)
        ]
    path = folder / f"{name}.mhclo"
    path.write_text("\n".join(head) + "\n")
    return path


def write_pack(root, pack: str, items: dict) -> pathlib.Path:
    """Write ``root/packs/<pack>.json``, the metadata file of a MakeHuman asset pack: an
    object keyed by item folder name, each entry with the fields of the real files (author,
    category, changed, created, description, license, original_author, original_source,
    source, thumbnail, type). Each value of ``items`` replaces the defaults of its entry."""
    root = pathlib.Path(root)
    (root / "packs").mkdir(parents=True, exist_ok=True)
    (root / "clothes").mkdir(parents=True, exist_ok=True)
    data = {}
    for item, fields in items.items():
        entry = {
            "author": "pack tests",
            "category": "Shirt/T-shirt/Sweater",
            "changed": "2020-01-01",
            "created": "2019-01-01",
            "description": f"The pack's description of {item}",
            "license": "CC0",
            "original_author": "",
            "original_source": "",
            "source": f"https://example.org/assets/{item}",
            "thumbnail": f"https://example.org/assets/{item}.thumb.png",
            "type": "clothes",
        }
        entry.update(fields)
        data[item] = entry
    path = root / "packs" / f"{pack}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _tights_patch():
    """A patch of helper-tights faces around the left thigh: the base-mesh index of each
    patch vertex, the patch's faces on those vertices and their positions (MakeHuman's
    frame) on the base mesh."""
    import os

    import torch

    import anny.utils.obj_utils as obj_utils
    from anny.paths import get_anny_root_dir

    path = os.path.join(get_anny_root_dir(), "data/mpfb2/3dobjs/base.obj")
    _, _, groups = obj_utils.load_obj_file(path, dtype=torch.float64)
    faces = groups["helper-tights"]["face_vertex_indices"].numpy()
    members = base_groups()["helper-tights"]
    position = _full_targets(tuple(int(r) for r in members))[0]  # Anny's frame
    where = dict(zip(members.tolist(), position))
    centre = np.array([where[int(v)] for v in faces.reshape(-1)]).reshape(
        -1, faces.shape[1], 3
    )
    centre = centre.mean(axis=1)
    keep = (centre[:, 0] > 0.04) & (centre[:, 2] > -0.30) & (centre[:, 2] < -0.15)
    patch = faces[keep]
    rows = np.unique(patch)
    remap = {int(r): k for k, r in enumerate(rows)}
    polygons = [[remap[int(v)] for v in face] for face in patch]
    mh = np.array([where[int(r)] for r in rows]) @ TO_MH.T
    return rows, polygons, mh

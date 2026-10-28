# OpenSculptBoy
# Apache License, Version 2.0
"""
Synthetic grooms for the tests of ``opensculptboy.groom``.

:func:`groom_handoff` writes a small appearance handoff: a head of skin (a sphere), a hair cap
over its top, three hair cards behind it whose vertex colours hold the hair's colour and which
a chain of two joints swings, and a strip of brow over each eye, each with an alpha texture;
the hair and the brows are skinned to a rig of five joints. :func:`card_grooms` puts a hair cap
with two cards and a chain, and two brow strips, on the head of anny's default character and
writes them as a card's groom files.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from opensculptboy.export.document import GltfDocument, png_bytes

# the head of the handoff, glTF axes (Y up, the figure facing +Z)
HEAD_CENTRE = np.array([0.0, 1.6, 0.0])
HEAD_RADIUS = 0.09
JOINTS = {  # name: (parent, position)
    "root": (None, (0.0, 0.0, 0.0)),
    "NeckJoint": ("root", (0.0, 1.47, -0.01)),
    "HeadJoint": ("NeckJoint", (0.0, 1.55, 0.0)),
    "HairChain_0": ("HeadJoint", (0.0, 1.64, -0.085)),
    "HairChain_1": ("HairChain_0", (0.0, 1.58, -0.105)),
}


def sphere(centre, radius, rows=12, cols=24, top_only=False):
    """A UV sphere (or its top half): vertices (V, 3), triangles (T, 3), UVs (V, 2)."""
    vs, uv = [], []
    last = rows // 2 if top_only else rows
    for i in range(last + 1):
        theta = np.pi * i / rows
        for j in range(cols + 1):
            phi = 2 * np.pi * j / cols
            vs.append(
                centre
                + radius
                * np.array(
                    [
                        np.sin(theta) * np.cos(phi),
                        np.cos(theta),
                        np.sin(theta) * np.sin(phi),
                    ]
                )
            )
            uv.append((j / cols, i / rows))
    tris = []
    for i in range(last):
        for j in range(cols):
            a, b = i * (cols + 1) + j, (i + 1) * (cols + 1) + j
            tris += [(a, b, b + 1), (a, b + 1, a + 1)]
    return np.array(vs), np.array(tris), np.array(uv)


def card(top, bottom, width, normal_axis=2):
    """A vertical card of two quads from ``top`` to ``bottom``: vertices, triangles, UVs."""
    top, bottom = np.asarray(top, float), np.asarray(bottom, float)
    mid = 0.5 * (top + bottom)
    side = np.array([width / 2, 0.0, 0.0])
    v = np.array(
        [top - side, top + side, mid - side, mid + side, bottom - side, bottom + side]
    )
    t = np.array([[0, 2, 1], [1, 2, 3], [2, 4, 3], [3, 4, 5]])
    uv = np.array([[0, 0], [1, 0], [0, 0.5], [1, 0.5], [0, 1], [1, 1]], float)
    return v, t, uv


def strand_texture(size=16, colour=(255, 255, 255)) -> bytes:
    """An RGBA texture of vertical strands: alpha 1 on every other column."""
    pixels = np.zeros((size, size, 4), np.uint8)
    pixels[..., :3] = colour
    pixels[:, ::2, 3] = 255
    return png_bytes(pixels)


def _material(doc, name, texture=None, blend=False, double=False, factor=(1, 1, 1, 1)):
    entry = {
        "name": name,
        "pbrMetallicRoughness": {
            "baseColorFactor": list(factor),
            "metallicFactor": 0.0,
            "roughnessFactor": 0.7,
        },
    }
    if texture is not None:
        image = doc.add_image(texture, "image/png", name)
        entry["pbrMetallicRoughness"]["baseColorTexture"] = {
            "index": doc.add_texture(image)
        }
    if blend:
        entry["alphaMode"] = "BLEND"
    if double:
        entry["doubleSided"] = True
    return doc.add_material(entry)


def _primitive(doc, v, t, uv, material, joints, weights, colors=None):
    attributes = {
        "POSITION": doc.accessor(np.asarray(v, np.float32), minmax=True),
        "TEXCOORD_0": doc.accessor(np.asarray(uv, np.float32)),
        "JOINTS_0": doc.accessor(np.asarray(joints, np.uint8)),
        "WEIGHTS_0": doc.accessor(np.asarray(weights, np.float32)),
    }
    if colors is not None:
        attributes["COLOR_0"] = doc.accessor(np.asarray(colors, np.float32))
    return {
        "attributes": attributes,
        "indices": doc.accessor(np.asarray(t, np.uint32).reshape(-1)),
        "material": material,
        "mode": 4,
    }


def groom_handoff(folder) -> pathlib.Path:
    """Write the handoff of the module documentation into ``folder``."""
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    doc = GltfDocument("groom fixtures")
    names = list(JOINTS)
    world = {n: np.asarray(p, float) for n, (_, p) in JOINTS.items()}
    for n, (parent, _) in JOINTS.items():
        local = world[n] - (world[parent] if parent else 0.0)
        doc.add_node({"name": n, "translation": local.tolist()})
    for n, (parent, _) in JOINTS.items():
        if parent:
            doc.nodes[names.index(parent)].setdefault("children", []).append(
                names.index(n)
            )
    ibm = np.tile(np.eye(4), (len(names), 1, 1))
    for k, n in enumerate(names):
        ibm[k, :3, 3] = -world[n]
    skin = doc.add_skin(
        {
            "joints": list(range(len(names))),
            "inverseBindMatrices": doc.accessor(
                ibm.transpose(0, 2, 1).reshape(-1, 16).astype(np.float32)
            ),
        }
    )
    head = names.index("HeadJoint")

    def rigid(n, joint=head):
        j = np.zeros((n, 4), np.uint8)
        j[:, 0] = joint
        w = np.zeros((n, 4), np.float32)
        w[:, 0] = 1.0
        return j, w

    # the skin: a sphere
    v, t, uv = sphere(HEAD_CENTRE, HEAD_RADIUS)
    skin_material = _material(doc, "skin_material")
    j, w = rigid(len(v))
    head_mesh = doc.add_mesh(
        {
            "name": "head_skin",
            "primitives": [_primitive(doc, v, t, uv, skin_material, j, w)],
        }
    )
    # the hair: a cap over the top (opaque) and three cards behind (blended, coloured)
    cap_v, cap_t, cap_uv = sphere(HEAD_CENTRE, HEAD_RADIUS + 0.003, top_only=True)
    cap_material = _material(doc, "hair_cap", factor=(0.1, 0.06, 0.04, 1.0))
    j, w = rigid(len(cap_v))
    # the cap's lowest ring weighs on the neck a little
    low = cap_v[:, 1] < HEAD_CENTRE[1] + 0.01
    j[low, 1] = names.index("NeckJoint")
    w[low, 0], w[low, 1] = 0.7, 0.3
    hair_strandserial = _material(
        doc, "hair_strands", strand_texture(), blend=True, double=True
    )
    cards_v, cards_t, cards_uv = [], [], []
    for k, x in enumerate((-0.03, 0.0, 0.03)):
        cv, ct, cuv = card(
            (x, HEAD_CENTRE[1] + 0.05, -HEAD_RADIUS - 0.006),
            (x, HEAD_CENTRE[1] - 0.02, -HEAD_RADIUS - 0.02),
            0.02,
        )
        cards_t.append(ct + 6 * k)
        cards_v.append(cv)
        cards_uv.append(cuv)
    cards_v, cards_t, cards_uv = (
        np.concatenate(cards_v),
        np.concatenate(cards_t),
        np.concatenate(cards_uv),
    )
    jc = np.zeros((len(cards_v), 4), np.uint8)
    wc = np.zeros((len(cards_v), 4), np.float32)
    jc[:, 0], jc[:, 1] = head, names.index("HairChain_0")
    share = np.clip((HEAD_CENTRE[1] + 0.05 - cards_v[:, 1]) / 0.07, 0, 1)
    wc[:, 0], wc[:, 1] = 1 - 0.8 * share, 0.8 * share
    colors = np.c_[np.full((len(cards_v), 3), 0.1), np.ones(len(cards_v))]
    colors[:, 0] = 0.05 + 0.1 * share  # a root-to-tip gradient
    hair_mesh = doc.add_mesh(
        {
            "name": "hair_mesh",
            "primitives": [
                _primitive(doc, cap_v, cap_t, cap_uv, cap_material, j, w),
                _primitive(
                    doc, cards_v, cards_t, cards_uv, hair_strandserial, jc, wc, colors
                ),
            ],
        }
    )
    # the brows: a strip over each eye, 1 mm in front of the skin
    brow_material = _material(
        doc, "brow_strands", strand_texture(colour=(54, 19, 2)), blend=True, double=True
    )
    brow_v, brow_t, brow_uv = [], [], []
    for k, x in enumerate((-0.03, 0.03)):
        bv = []
        for row, dy in enumerate((0.0, 0.006)):
            for s in np.linspace(-0.015, 0.015, 4):
                p = np.array([x + s, HEAD_CENTRE[1] + 0.02 + dy, 0.0])
                d = np.sqrt(
                    max(HEAD_RADIUS**2 - (p[0]) ** 2 - (p[1] - HEAD_CENTRE[1]) ** 2, 0)
                )
                p[2] = d + 0.001
                bv.append(p)
        bv = np.array(bv)
        bt = []
        for c in range(3):
            a, b = c, c + 4
            bt += [(a, b, a + 1), (a + 1, b, b + 1)]
        brow_t.append(np.array(bt) + 8 * k)
        brow_v.append(bv)
        brow_uv.append(
            np.c_[np.tile(np.linspace(0, 1, 4), 2), np.repeat([0.0, 1.0], 4)]
        )
    brow_v, brow_t, brow_uv = (
        np.concatenate(brow_v),
        np.concatenate(brow_t),
        np.concatenate(brow_uv),
    )
    j, w = rigid(len(brow_v))
    brow_mesh = doc.add_mesh(
        {
            "name": "brow_mesh",
            "primitives": [
                _primitive(doc, brow_v, brow_t, brow_uv, brow_material, j, w)
            ],
        }
    )
    nodes = [
        doc.add_node({"name": name, "mesh": mesh, "skin": skin})
        for name, mesh in (
            ("head_skin", head_mesh),
            ("hair_mesh", hair_mesh),
            ("brow_mesh", brow_mesh),
        )
    ]
    doc.scene_nodes = [0] + nodes
    doc.write(folder / "replica.glb")
    appearance = {
        "parts": {
            "head": {"node_name": "head_skin", "role": "skin"},
            "hair": {"node_name": "hair_mesh"},
            "eyebrows": {"node_name": "brow_mesh"},
        },
        "materials": [],
        "landmarks": {},
    }
    (folder / "appearance.json").write_text(json.dumps(appearance), encoding="utf-8")
    return folder


# -------------------------------------------------------------------- grooms on anny's head
def card_grooms(model, rest, folder, stem="card"):
    """
    A hair cap with two cards and a chain, and two brow strips, on the head of ``rest`` (the
    rest vertices of ``model``'s default character), written into ``folder`` as a card's groom
    files: returns the ``material`` sections ``{"hair": ..., "brows": ...}`` and the two
    :class:`~opensculptboy.groom.data.GroomData`.
    """
    from anny.hair import closest_triangles

    from opensculptboy.export.body import vertex_normals
    from opensculptboy.eyes.mesh import eyeball_rows
    from opensculptboy.groom.data import GroomData, GroomMaterial
    from opensculptboy.teeth.mesh import mouth_rows

    folder = pathlib.Path(folder)
    rest = np.asarray(rest, np.float64)
    from opensculptboy.export.body import triangulated_faces

    triangles, _ = triangulated_faces(model)
    triangles = np.asarray(triangles)
    head, _ = mouth_rows(model, None)
    normals = vertex_normals(rest, triangles)
    top = rest[head, 2].max()
    scalp = np.zeros(len(rest), bool)
    scalp[head[rest[head, 2] > top - 0.05]] = True
    cap_t = triangles[scalp[triangles].all(1)]
    rows = np.unique(cap_t)
    remap = np.full(len(rest), -1)
    remap[rows] = np.arange(len(rows))
    cap_v = rest[rows] + 0.003 * normals[rows]
    cap_t = remap[cap_t]
    centre = rest[head].mean(0)
    back = rest[head, 1].max()
    cards_v, cards_t = [], []
    for k, x in enumerate((-0.02, 0.02)):
        v, t, _ = card((0, 0, 0), (0, 0, 0), 0.02)
        v = np.array(
            [
                [x - 0.01, back + 0.004, top - 0.03],
                [x + 0.01, back + 0.004, top - 0.03],
                [x - 0.01, back + 0.012, top - 0.07],
                [x + 0.01, back + 0.012, top - 0.07],
                [x - 0.01, back + 0.02, top - 0.11],
                [x + 0.01, back + 0.02, top - 0.11],
            ]
        )
        cards_v.append(v)
        cards_t.append(t + 6 * k)
    cards_v, cards_t = np.concatenate(cards_v), np.concatenate(cards_t)
    points = np.concatenate([cap_v, cards_v])
    tris = np.concatenate([cap_t, cards_t + len(cap_v)])
    material = np.r_[np.zeros(len(cap_t), int), np.ones(len(cards_t), int)]
    uvs = np.zeros((len(points), 2))
    uvs[len(cap_v) :] = np.tile(
        [[0, 0], [1, 0], [0, 0.5], [1, 0.5], [0, 1], [1, 1]], (2, 1)
    )
    colors = np.ones((len(points), 4))
    colors[:, :3] = 0.08
    colors[len(cap_v) :, 0] = np.linspace(0.05, 0.2, len(cards_v))
    share = np.zeros(len(points))
    share[len(cap_v) :] = np.clip((top - 0.03 - cards_v[:, 2]) / 0.08, 0, 1)
    labels = list(model.bone_labels)
    slots = np.zeros((len(points), 1), np.int64)
    weights = np.ones((len(points), 1))
    hair = GroomData(
        kind="hair",
        points=points,
        normals=vertex_normals(points, tris),
        uvs=uvs,
        triangles=tris,
        material=material,
        materials=[
            GroomMaterial("hair_cap", (0.1, 0.06, 0.04, 1.0), 0.7),
            GroomMaterial(
                "hair_strands",
                (1, 1, 1, 1),
                0.45,
                "BLEND",
                0.5,
                True,
                textures=("base_color",),
            ),
        ],
        colors=colors,
        bones=("head",),
        bone_slots=slots,
        bone_weights=weights,
        chains=[
            np.array(
                [[0.0, back + 0.004, top - 0.03], [0.0, back + 0.012, top - 0.07]]
                + [[0.0, back + 0.02, top - 0.11]]
            )
        ],
        chain_share=share,
    )
    assert "head" in labels
    # the brows: a strip over each eye, 0.5 mm off the skin
    eyes = eyeball_rows(model)
    brow_v, brow_t = [], []
    for k, side in enumerate(("L", "R")):
        c = rest[eyes[side]].mean(0)
        grid = []
        for dz in (0.016, 0.021):
            for s in np.linspace(-0.018, 0.018, 5):
                grid.append([c[0] + s, c[1] - 0.03, c[2] + dz])
        grid = np.array(grid)
        skin_tris = triangles[
            ~np.isin(triangles, np.concatenate(list(eyes.values()))).any(1)
        ]
        tri, bary, _ = closest_triangles(grid, rest, skin_tris)
        p = np.einsum("nk,nkd->nd", bary, rest[skin_tris[tri]])
        n = np.einsum("nk,nkd->nd", bary, normals[skin_tris[tri]])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        brow_v.append(p + 0.0005 * n)
        t = []
        for col in range(4):
            a, b = col, col + 5
            t += [(a, b, a + 1), (a + 1, b, b + 1)]
        brow_t.append(np.array(t) + 10 * k)
    brow_v, brow_t = np.concatenate(brow_v), np.concatenate(brow_t)
    brow_uv = np.tile(
        np.c_[np.tile(np.linspace(0, 1, 5), 2), np.repeat([0.0, 1.0], 5)], (2, 1)
    )
    brows = GroomData(
        kind="brows",
        points=brow_v,
        normals=vertex_normals(brow_v, brow_t),
        uvs=brow_uv,
        triangles=brow_t,
        material=np.zeros(len(brow_t), int),
        materials=[
            GroomMaterial(
                "brow_strands",
                (1, 1, 1, 1),
                0.9,
                "BLEND",
                0.5,
                True,
                textures=("base_color",),
            )
        ],
    )
    hair.save(folder / f"{stem}.hair.safetensors")
    brows.save(folder / f"{stem}.brows.safetensors")
    (folder / f"{stem}.hair.hair_strands.base_color.png").write_bytes(strand_texture())
    (folder / f"{stem}.brows.brow_strands.base_color.png").write_bytes(
        strand_texture(colour=(54, 19, 2))
    )
    sections = {
        "hair": {
            "source": "reference",
            "mesh": f"{stem}.hair.safetensors",
            "textures": {
                "hair_strands": {
                    "base_color": f"{stem}.hair.hair_strands.base_color.png"
                }
            },
        },
        "brows": {
            "source": "reference",
            "mesh": f"{stem}.brows.safetensors",
            "textures": {
                "brow_strands": {
                    "base_color": f"{stem}.brows.brow_strands.base_color.png"
                }
            },
        },
    }
    return sections, hair, brows, centre

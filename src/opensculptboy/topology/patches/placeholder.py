# OpenSculptBoy
# Apache License, Version 2.0
"""
A placeholder patch for the tests of the core: one quad on each side of the back of the torso,
each split into an inner quad, a ring of three quads and two triangles.

Off in the shipped file (``ENABLED = False``); the tests compose it in. Each new vertex lies
halfway between a corner of the quad and its centre, so the patch keeps the surface flat and
exercises every part of the format: anchors and weights, new faces over old and new
vertices, triangles where the patch joins anny's faces, new UVs and the mirror image.
"""

from __future__ import annotations

import numpy as np

from opensculptboy.topology.patch import BaseQuadMesh, Patch, PatchError, blend_rows

NAME = "placeholder"
ORDER = 900
ENABLED = False
OPTIONS = {"target": [0.05, 0.12, 0.35]}


def chosen_face(base: BaseQuadMesh, target) -> int:
    """The quad of the left side (x > 0) with an outward normal toward +Y (the back) whose
    centre lies nearest ``target`` (metres, anny's frame), away from UV seams."""
    v, q = base.vertices, base.quads
    centres = v[q].mean(1)
    normals = np.cross(v[q[:, 1]] - v[q[:, 0]], v[q[:, 3]] - v[q[:, 0]])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    candidates = np.nonzero((centres[:, 0] > 0.01) & (normals[:, 1] > 0.8))[0]
    order = candidates[np.argsort(np.linalg.norm(centres[candidates] - target, axis=1))]
    for f in order:
        ring = np.unique(np.concatenate([base.vertex_faces[i] for i in q[f]]))
        try:
            base.face_uvs(ring)
        except PatchError:
            continue
        return int(f)
    raise PatchError("No quad of the back fits the placeholder patch.")


def build(base: BaseQuadMesh, target=OPTIONS["target"]) -> Patch:
    """The placeholder patch: the chosen quad and its mirror image, split."""
    f = chosen_face(base, np.asarray(target, dtype=np.float64))
    n, u = base.vertex_count, base.uv_count
    corners, uv_corners = base.quads[f], base.uv_quads[f]

    def inset(ids):
        # p_i = (v_i + centre) / 2: 5/8 on its corner and 1/8 on each other corner
        return [
            {int(c): (0.625 if k == i else 0.125) for k, c in enumerate(ids)}
            for i in range(4)
        ]

    anchors, weights = blend_rows(inset(corners))
    uv_anchors, uv_weights = blend_rows(inset(uv_corners))

    def faces(outer, first_new):
        p = [first_new + i for i in range(4)]
        v = [int(x) for x in outer]
        return [
            [p[0], p[1], p[2], p[3]],
            [v[0], v[1], p[1], p[1]],  # the first ring quad as two triangles
            [v[0], p[1], p[0], p[0]],
            [v[1], v[2], p[2], p[1]],
            [v[2], v[3], p[3], p[2]],
            [v[3], v[0], p[0], p[3]],
        ]

    left = Patch(
        name=NAME,
        removed_faces=[f],
        anchors=anchors,
        weights=weights,
        faces=faces(corners, n),
        uv_faces=faces(uv_corners, u),
        uv_anchors=uv_anchors,
        uv_weights=uv_weights,
        options={"target": [float(x) for x in target]},
    )
    return left.with_mirror(base)

# OpenSculptBoy
# Apache License, Version 2.0
"""
The eye texture: the eyeballs' region of anny's UV atlas, given a texture of its own.

Anny's eyeballs cover about 52 by 52 texels each of a 2048-texel atlas, too few for an iris. The
texture bake (:func:`opensculptboy.texture.bake.bake`) can therefore also write an eye texture
that holds that region alone: the bounding box of the eyeballs' UVs (both eyes), padded and
widened to the texture's aspect of :data:`EYE_TEXTURE_ASPECT`. The exporters draw the eyeballs
with a material of their own whose texture carries the transform of :func:`eye_texture_transform`
(``KHR_texture_transform`` in glTF and VRM 1.0, the ST vector of ``_MainTex`` in VRM 0.x), so
that the eyeballs keep anny's UVs and the files need no second UV set.
"""

from __future__ import annotations

import numpy as np

# width : height of the eye texture; the two eyes sit side by side in anny's atlas
EYE_TEXTURE_ASPECT = 2
# padding of the box on each side, as a share of its larger side
EYE_BOX_PADDING = 0.04


def eyeball_rows(model) -> np.ndarray | None:
    """The vertex rows of both eyeballs of an Anny model, or None for a mesh without them."""
    from opensculptboy.export.vrm_tables import eyeball_vertices

    try:
        return np.concatenate(list(eyeball_vertices(model).values()))
    except ValueError:
        return None


def eye_uv_box(model) -> tuple[np.ndarray, np.ndarray] | None:
    """
    The box (lower and upper corners, OBJ convention: v = 0 at the bottom) of anny's atlas that
    the eye texture covers, or None for a mesh without eyeballs or UVs.
    """
    rows = eyeball_rows(model)
    uvs = model.texture_coordinates
    uv_faces = model.face_texture_coordinate_indices
    if rows is None or uvs is None or uv_faces is None:
        return None
    faces = model.faces.cpu().numpy()
    uv_faces = uv_faces.cpu().numpy()
    uvs = uvs.detach().cpu().double().numpy()
    eye = np.zeros(len(model.template_vertices), bool)
    eye[rows] = True
    points = uvs[np.unique(uv_faces[eye[faces].all(1)])]
    lo, hi = points.min(0), points.max(0)
    pad = EYE_BOX_PADDING * float((hi - lo).max())
    lo, hi = lo - pad, hi + pad
    centre, size = 0.5 * (lo + hi), hi - lo
    # widen the short side to the aspect of the texture
    width = max(size[0], EYE_TEXTURE_ASPECT * size[1])
    size = np.array([width, width / EYE_TEXTURE_ASPECT])
    return centre - 0.5 * size, centre + 0.5 * size


def eye_texture_transform(model) -> dict | None:
    """
    The ``KHR_texture_transform`` (``{"offset": [u, v], "scale": [u, v]}``, glTF convention:
    v = 0 at the top) that takes anny's glTF UVs of the eye box to the eye texture's [0, 1]
    square, or None for a mesh without eyeballs.
    """
    box = eye_uv_box(model)
    if box is None:
        return None
    (u0, v0), (u1, v1) = box
    su, sv = 1.0 / (u1 - u0), 1.0 / (v1 - v0)
    # glTF v = 1 - v_obj: the box runs from 1 - v1 to 1 - v0
    return {
        "offset": [float(-u0 * su), float(-(1.0 - v1) * sv)],
        "scale": [float(su), float(sv)],
    }


def to_eye_uvs(uvs: np.ndarray, box) -> np.ndarray:
    """UVs of anny's atlas (OBJ convention) in the eye texture's square (OBJ convention)."""
    lo, hi = (np.asarray(b, np.float64) for b in box)
    return (np.asarray(uvs, np.float64) - lo) / (hi - lo)

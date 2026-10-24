# OpenSculptBoy
# Apache License, Version 2.0
"""
The skin that wearables cover.

A wearable declares the skin it covers as a mask over the vertices of Anny's template
(``Wearable.covers``), or leaves it to :func:`ray_cover`: a body vertex is covered when the
ray along its normal meets the dressed piece within :data:`RAY_REACH`. An export removes a
body triangle when all three of its corners are covered by a shown piece
(:func:`covered_triangles`), or keeps every triangle on request.

:func:`coverage_gaps` measures a fit: the covered body vertices that lie outside the piece
by more than a tolerance (the E4 gate is 2 mm at rest).
"""

from __future__ import annotations

import numpy as np

from opensculptboy.wearables.geometry import Surface, ray_hits

RAY_REACH = 0.03  # metres


def ray_cover(body, piece, reach: float = RAY_REACH) -> np.ndarray:
    """(N,) the body vertices whose normal ray meets the visible triangles of a dressed
    piece within ``reach``."""
    return ray_hits(
        body.vertices,
        body.normals,
        piece.vertices,
        piece.visible_triangles(),
        reach,
    )


def piece_cover(body, piece) -> np.ndarray:
    """The cover mask of a dressed piece on Anny's rows: its package's mask, else the ray
    test. The masks of the piece's groups that show add to it. With ``cover_ray``, and when
    groups without masks of their own are hidden, the piece keeps only the vertices whose
    normal ray meets its visible triangles."""
    wearable = piece.wearable
    if wearable.covers is None:
        return ray_cover(body, piece)
    mask = np.asarray(wearable.covers, bool)[: len(body.vertices)].copy()
    for group, extra in wearable.group_covers.items():
        if group not in piece.hidden_groups:
            mask |= np.asarray(extra, bool)[: len(body.vertices)]
    hidden = set(piece.hidden_groups) - set(wearable.group_covers)
    if wearable.cover_ray:  # the mask holds where the piece itself is in the way
        mask &= ray_cover(body, piece, RAY_REACH)
    elif hidden:  # groups without masks of their own: keep what the rest still covers
        mask &= ray_cover(body, piece, 2 * RAY_REACH)
    return mask


def outfit_cover(body, pieces) -> np.ndarray:
    """The union of the cover masks of the shown pieces."""
    mask = np.zeros(len(body.vertices), bool)
    for piece in pieces:
        if piece.shown:
            mask |= piece_cover(body, piece)
    return mask


def row_cover(mask: np.ndarray, rows: int, model_rows: int, detail=None) -> np.ndarray:
    """
    The cover mask on the rows of an exported mesh: Anny's rows (``mask``), the rows that the
    model appends after them (never covered; ``model_rows`` in all), the vertices of a detail
    level (covered where every level-0 vertex under them is), and the rows that the eyes and
    teeth append (never covered). ``rows`` is the row count of the exported mesh.
    """
    base = np.zeros(model_rows)
    base[: len(mask)] = np.asarray(mask, bool)
    mask = base > 0.5
    if detail is not None:
        mask = np.asarray(detail.topology.apply(base)).reshape(-1) > 1 - 1e-9
    out = np.zeros(rows, bool)
    out[: min(rows, len(mask))] = mask[:rows]
    return out


def covered_triangles(body_mesh, rows_mask: np.ndarray) -> np.ndarray:
    """(T,) the triangles of a :class:`~opensculptboy.export.body.BodyMesh` whose three
    corners are covered."""
    return rows_mask[body_mesh.source[body_mesh.triangles]].all(axis=1)


def coverage_gaps(body, piece, mask: np.ndarray, tolerance: float = 0.002) -> dict:
    """
    The covered body vertices that lie outside a dressed piece by more than ``tolerance``
    (signed along the piece's normal at the closest point): their count, the largest
    distance and the share of the covered vertices that the piece reaches within 5 cm.
    """
    rows = np.flatnonzero(mask)
    if len(rows) == 0:
        return {"covered": 0, "outside": 0, "largest_mm": 0.0}
    surface = Surface(piece.vertices, piece.visible_triangles(), piece.normals)
    signed, found, _ = surface.signed_distance(body.vertices[rows])
    near = found.distance < 0.05
    outside = near & (signed > tolerance)
    return {
        "covered": int(len(rows)),
        "outside": int(outside.sum()),
        "largest_mm": round(float(signed[near].max(initial=0.0)) * 1000.0, 3),
    }

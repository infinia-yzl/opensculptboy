# OpenSculptBoy
# Apache License, Version 2.0
"""
Measures of a mesh that find folds and collapsed triangles: the pairs of triangles that share
an edge, the bend of each such edge (the dihedral angle between the two triangles) and the
area of each triangle.

A fold is an edge that bends by more than :data:`FOLD_ANGLE` (120 degrees) where a reference
mesh bends by that much or less; the reference is Anny's rest mesh for a bind mesh, and
Anny's posed mesh for a posed file. A collapsed triangle keeps less than
:data:`COLLAPSED_AREA` (a quarter) of its area on the reference mesh.
"""

from __future__ import annotations

import numpy as np

# The bend (degrees) above which an edge counts as a fold, where the reference bends less.
FOLD_ANGLE = 120.0
# The share of its reference area below which a triangle counts as collapsed.
COLLAPSED_AREA = 0.25


def adjacent_triangles(triangles: np.ndarray) -> np.ndarray:
    """The pairs of triangles (E, 2) that share an edge."""
    triangles = np.asarray(triangles)
    edges = np.sort(triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    faces = np.repeat(np.arange(len(triangles)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, faces = edges[order], faces[order]
    same = np.all(edges[1:] == edges[:-1], axis=1)
    return np.stack([faces[:-1][same], faces[1:][same]], axis=1)


def bends(vertices: np.ndarray, triangles: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """
    The dihedral angle of each pair of adjacent triangles, in degrees: the angle between their
    normals, 0 on a flat surface and 180 where the surface folds back onto itself. ``vertices``
    may carry leading batch dimensions (..., V, 3).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    a, b, c = (vertices[..., triangles[:, k], :] for k in range(3))
    normals = np.cross(b - a, c - a)
    first, second = normals[..., pairs[:, 0], :], normals[..., pairs[:, 1], :]
    return np.degrees(
        np.arctan2(
            np.linalg.norm(np.cross(first, second), axis=-1),
            np.sum(first * second, axis=-1),
        )
    )


def areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """The area of each triangle; ``vertices`` may carry leading batch dimensions."""
    vertices = np.asarray(vertices, dtype=np.float64)
    a, b, c = (vertices[..., triangles[:, k], :] for k in range(3))
    return np.linalg.norm(np.cross(b - a, c - a), axis=-1) / 2.0


def new_folds(bend: np.ndarray, reference_bend: np.ndarray) -> np.ndarray:
    """The edges that bend by more than :data:`FOLD_ANGLE` where the reference does not."""
    return (np.asarray(bend) > FOLD_ANGLE) & (np.asarray(reference_bend) <= FOLD_ANGLE)


def collapsed(area: np.ndarray, reference_area: np.ndarray) -> np.ndarray:
    """The triangles that keep less than :data:`COLLAPSED_AREA` of their reference area."""
    return np.asarray(area) < COLLAPSED_AREA * np.maximum(
        np.asarray(reference_area), 1e-12
    )

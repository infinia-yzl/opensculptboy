# OpenSculptBoy
# Apache License, Version 2.0
"""
Residuals of a fit: what the parametric body misses, as rest-frame offsets per vertex.

The registration lays anny's mesh (posed, and placed in the reference's frame) on the target.
Linear blend skinning turns each rest vertex v by its blended bone transform M_i, so the posed
vertex is M_i v; :func:`unpose` inverts that per vertex and brings the registered mesh back to
the rest frame, where :func:`rest_offsets` gives the offsets: registered rest minus parametric
rest. Added to the subdivided parametric rest mesh and skinned again, the offsets give back the
registered mesh exactly (:class:`opensculptboy.detail.DetailModel`).

:func:`surface_error` measures a fitted mesh against the reference in both directions.
"""

from __future__ import annotations

import numpy as np


def blended_transforms(
    weights: np.ndarray, indices: np.ndarray, transforms: np.ndarray
) -> np.ndarray:
    """Per-vertex blends (N, 4, 4) of bone transforms (J, 4, 4) with weights (N, K)."""
    weights = np.asarray(weights, dtype=np.float64)
    transforms = np.asarray(transforms, dtype=np.float64)
    blended = np.zeros((len(weights), 4, 4))
    for k in range(weights.shape[1]):
        blended += weights[:, k, None, None] * transforms[np.asarray(indices)[:, k]]
    return blended


def pose_vertices(rest, weights, indices, transforms, rotation=None, translation=None):
    """Skin rest vertices (N, 3) and place them by x -> R x + t."""
    M = blended_transforms(weights, indices, transforms)
    posed = np.einsum("nij, nj -> ni", M[:, :3, :3], rest) + M[:, :3, 3]
    if rotation is not None:
        posed = posed @ np.asarray(rotation).T
    if translation is not None:
        posed = posed + np.asarray(translation)
    return posed


def unpose(posed, weights, indices, transforms, rotation=None, translation=None):
    """
    Rest vertices (N, 3) that :func:`pose_vertices` takes to ``posed``: the placement undone,
    then the inverse of each vertex's blended transform.
    """
    posed = np.asarray(posed, dtype=np.float64)
    if translation is not None:
        posed = posed - np.asarray(translation)
    if rotation is not None:
        posed = posed @ np.asarray(rotation)
    M = blended_transforms(weights, indices, transforms)
    linear = np.linalg.inv(M[:, :3, :3])
    return np.einsum("nij, nj -> ni", linear, posed - M[:, :3, 3])


def rest_offsets(registered_rest: np.ndarray, parametric_rest: np.ndarray):
    """The detail offsets (N, 3): registered rest mesh minus the parametric rest mesh."""
    return np.asarray(registered_rest, dtype=np.float64) - np.asarray(
        parametric_rest, dtype=np.float64
    )


def surface_error(
    vertices: np.ndarray,
    triangles: np.ndarray,
    reference,
    mask: np.ndarray | None = None,
    reverse_samples: int = 20000,
    reverse_limit: float = 0.01,
    seed: int = 0,
) -> dict:
    """
    Distances in millimetres between a fitted mesh and a reference
    (:class:`~opensculptboy.fit.reference.Reference`).

    ``forward`` measures each fitted vertex (within ``mask``) to the reference surface;
    ``reverse`` measures a sample of reference vertices to the fitted surface, leaving out
    those further than ``reverse_limit`` (parts the fit does not cover, such as the inside
    of the mouth). Each gives the mean, the median, the 95th percentile and the maximum.
    """
    from anny.hair import closest_triangles

    vertices = np.asarray(vertices, dtype=np.float64)
    keep = np.ones(len(vertices), bool) if mask is None else np.asarray(mask, bool)
    _, _, forward, _, _ = reference.closest(vertices[keep])
    rng = np.random.default_rng(seed)
    count = min(reverse_samples, len(reference.vertices))
    sample = rng.choice(len(reference.vertices), count, replace=False)
    fitted_triangles = np.asarray(triangles)
    if mask is not None:
        fitted_triangles = fitted_triangles[keep[fitted_triangles].all(1)]
    _, _, reverse = closest_triangles(
        reference.vertices[sample], vertices, fitted_triangles
    )
    covered = reverse < reverse_limit

    def stats(d):
        d = 1000.0 * np.asarray(d)
        return {
            "mean": float(d.mean()),
            "median": float(np.median(d)),
            "p95": float(np.percentile(d, 95)),
            "max": float(d.max()),
        }

    return {
        "forward": stats(forward),
        "reverse": stats(reverse[covered]),
        "reverse_covered": float(covered.mean()),
    }

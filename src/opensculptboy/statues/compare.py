# OpenSculptBoy
# Apache License, Version 2.0
"""
How close a fitted statue comes to its scan, region by region.

:func:`statue_regions` labels anny's level-0 vertices with the regions of a statue: the face,
the scalp, the ears, the neck, the torso, the arms, the hands, the legs and the feet (the
regions of :func:`opensculptboy.fit.metrics.anny_regions`, with its ``body`` split by each
vertex's strongest bone). :func:`region_distances` measures a fitted mesh against a surface
both ways: from each fitted vertex to the surface (``forward``) and from points of the surface
to the fitted mesh (``reverse``, each point in the region of its nearest fitted vertex), in
millimetres. ``far`` is the share of the fitted vertices more than :data:`FAR` from the
surface: where the scan has nothing, such as a lost hand, which anny fills with its own shape.
"""

from __future__ import annotations

import numpy as np

STATUE_REGIONS = (
    "face",
    "scalp",
    "ears",
    "neck",
    "torso",
    "arms",
    "hands",
    "legs",
    "feet",
)
# a fitted vertex further than this (metres) from the scan has no scan under it
FAR = 0.005
_ARM_WORDS = ("clavicle", "shoulder", "upperarm", "lowerarm")
_LEG_WORDS = ("upperleg", "lowerleg")
_FOOT_WORDS = ("foot", "toe")


def statue_regions(model, rest_vertices: np.ndarray | None = None) -> np.ndarray:
    """The region (an index into :data:`STATUE_REGIONS`, or -1 for the closed parts and the
    cavities) of each vertex of anny's level-0 mesh."""
    from opensculptboy.fit.metrics import REGIONS, anny_regions

    base = anny_regions(model, rest_vertices)
    weights = model.vertex_bone_weights.detach().cpu().numpy()
    indices = model.vertex_bone_indices.cpu().numpy()
    bones = list(model.bone_labels)
    strongest = np.asarray(bones, dtype=object)[
        indices[np.arange(len(indices)), weights.argmax(1)]
    ]
    out = np.full(len(base), -1, np.int64)
    for name in ("face", "scalp", "ears", "neck", "hands"):
        out[base == REGIONS.index(name)] = STATUE_REGIONS.index(name)
    out[base == REGIONS.index("jaw")] = STATUE_REGIONS.index("face")
    body = base == REGIONS.index("body")

    def bone_has(words):
        return np.array([any(w in str(b) for w in words) for b in strongest])

    out[body] = STATUE_REGIONS.index("torso")
    out[body & bone_has(_ARM_WORDS)] = STATUE_REGIONS.index("arms")
    out[body & bone_has(_LEG_WORDS)] = STATUE_REGIONS.index("legs")
    out[body & bone_has(_FOOT_WORDS)] = STATUE_REGIONS.index("feet")
    return out


def carry_regions(topology, regions0: np.ndarray) -> np.ndarray:
    """The regions of level 0 carried to a level of the detail layer (the strongest share)."""
    regions0 = np.asarray(regions0)
    shares = np.zeros((len(regions0), len(STATUE_REGIONS) + 1))
    shares[np.arange(len(regions0)), regions0 + 1] = 1.0
    return np.asarray(topology.apply(shares)).argmax(1) - 1


def _stats(values: np.ndarray) -> dict | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    mm = 1000.0 * np.abs(values)
    return {
        "mean": round(float(mm.mean()), 3),
        "median": round(float(np.median(mm)), 3),
        "p95": round(float(np.percentile(mm, 95)), 3),
        "max": round(float(mm.max()), 3),
        "count": int(len(mm)),
    }


def region_distances(
    vertices: np.ndarray,
    triangles: np.ndarray,
    regions: np.ndarray,
    surface_vertices: np.ndarray,
    surface_triangles: np.ndarray,
    samples: int = 60000,
    seed: int = 0,
) -> dict:
    """
    The distances (mm) between a fitted mesh (``vertices``, ``triangles``, ``regions`` per
    vertex, -1 for parts left out) and a surface, per region and over all: ``forward`` from
    the fitted vertices, ``reverse`` from ``samples`` points spread over the surface by area,
    and ``far``, the share of the fitted vertices more than :data:`FAR` away.
    """
    from anny.hair import closest_triangles
    from scipy.spatial import cKDTree

    V = np.asarray(vertices, dtype=np.float64)
    T = np.asarray(triangles, dtype=np.int64)
    regions = np.asarray(regions)
    S = np.asarray(surface_vertices, dtype=np.float64)
    ST = np.asarray(surface_triangles, dtype=np.int64)
    measured = regions >= 0
    used = np.unique(ST)
    remap = np.full(len(S), -1, np.int64)
    remap[used] = np.arange(len(used))
    S, ST = S[used], remap[ST]
    _, _, forward = closest_triangles(V[measured], S, ST)
    forward_all = np.full(len(V), np.nan)
    forward_all[measured] = forward

    rng = np.random.default_rng(seed)
    p = S[ST]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    pick = rng.choice(len(ST), samples, p=area / area.sum())
    u, v = rng.random(samples), rng.random(samples)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    points = p[pick, 0] + u[:, None] * (p[pick, 1] - p[pick, 0])
    points += v[:, None] * (p[pick, 2] - p[pick, 0])
    kept = T[measured[T].all(1)]
    _, _, reverse = closest_triangles(points, V, kept)
    rows = np.nonzero(measured)[0]
    _, nearest = cKDTree(V[rows]).query(points)
    reverse_regions = regions[rows[nearest]]

    out = {"all": {}, "regions": {}}

    def entry(fwd, rev):
        far = np.asarray(fwd)[np.isfinite(fwd)]
        return {
            "forward": _stats(fwd),
            "reverse": _stats(rev),
            "far": round(float((far > FAR).mean()), 4) if len(far) else None,
        }

    out["all"] = entry(forward_all[measured], reverse)
    for k, name in enumerate(STATUE_REGIONS):
        out["regions"][name] = entry(
            forward_all[regions == k], reverse[reverse_regions == k]
        )
    return out

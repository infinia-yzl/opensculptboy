# OpenSculptBoy
# Apache License, Version 2.0
"""
Honest measures of a fitted surface against its reference, by region.

The registration ends by moving vertices onto the reference (``project_onto``), so a distance
from the fitted vertices to the reference says little about what the registration got right:
it is close to 0 by construction. :func:`measure` therefore measures a fitted mesh as it is
given, over the whole skin, with no mask, no visibility test and no distance cut-off:

- ``forward``: each skin vertex of the fitted mesh to the reference surface, signed along the
  reference's normal (positive: the fit lies outside the reference, too full; negative: inside);
- ``reverse``: every vertex of the reference skin to the fitted surface (every triangle of
  anny's body mesh, the inside of the mouth and the eye sockets included, the closed parts
  such as the eyeballs left out), signed the same way (positive: the fit lies outside);
- ``normal_angle``: the angle in degrees between each fitted vertex normal and the
  reference's interpolated normal at its closest point, both against the reference's
  geometric normals (``geometric``) and against the normals its file shades with (``file``,
  the same when the source has none).

``all`` holds every reference vertex for the reverse measure and every fitted vertex but those
of anny's cavities for the others: the inside of anny's mouth and eye sockets has no
counterpart in a skin that closes them, and its own region ``cavity`` reports it. Each measure
gives the mean of the absolute values, the signed mean, the 95th percentile and
the maximum of the absolute values, per region (:data:`REGIONS`) and over all. The regions come
from anny's mesh (:func:`anny_regions`: the base-mesh labels, the outer ears of
``ear_weights``, the dominant bones and the craniofacial landmarks of the rest mesh) and pass to
the reference by its nearest fitted vertex, with two exceptions taken from the source itself:
the reference's ``"ear"`` mask (``masks.ear`` of a handoff) makes the ears, and the reference
vertices within :data:`SEAM_RADIUS` of a border that two skin parts share (the seam of a head
part and a body part at the neck) make ``neck_seam``. A fitted vertex takes the region of its
closest reference vertex where that is the ears or the seam, and its own region otherwise.
Compute the measures on the geometry before and after any final projection; the two together
tell the registration from the projection.
"""

from __future__ import annotations

import dataclasses

import numpy as np

REGIONS = (
    "face",
    "jaw",
    "ears",
    "scalp",
    "neck",
    "neck_seam",
    "body",
    "hands",
    "cavity",
)
# reference vertices within this distance (metres) of a border shared by two skin parts
SEAM_RADIUS = 0.01


def anny_regions(model, rest_vertices: np.ndarray | None = None) -> np.ndarray:
    """
    The region (an index into :data:`REGIONS`, or -1 for the closed parts that no measure
    takes: the eyeballs and the tongue) of each vertex of anny's level-0 mesh, from its rest
    mesh (``rest_vertices``, the template by default; metres, Z up, facing -Y):

    - ``hands``: the base-mesh labels ``hand.L`` and ``hand.R``;
    - ``cavity``: the inside of the mouth and the eye sockets;
    - ``ears``: the outer ears (``ear_weights`` over 0.5);
    - ``neck``: the vertices whose strongest bone is a neck bone;
    - of the base-mesh label ``head``: ``jaw`` below the mouth corners (cheilion) and in front
      of 1 cm behind the tragion, which holds the jaw line, its angle and the skin under the
      chin; ``face`` above it and in front of the tragion, below 6 cm under the top of the head;
      ``scalp`` the rest;
    - ``body``: everything else.
    """
    from anny.faces.authoring.base_mesh import base_mesh
    from opensculptboy.fit.register import (
        EXCLUDED_LABELS,
        _Craniofacial,
        ear_weights,
        excluded_vertices,
    )

    if rest_vertices is None:
        rest_vertices = model.template_vertices.detach().double().cpu().numpy()
    V = np.asarray(rest_vertices, dtype=np.float64)
    mesh = base_mesh()
    base = model.base_mesh_vertex_indices.cpu().numpy()
    labels = np.full(len(mesh.V), "", dtype=object)
    for name in np.unique(mesh.face_labels):
        labels[np.unique(mesh.quads[mesh.face_labels == name])] = name
    labels = labels[base]
    region = np.full(len(V), REGIONS.index("body"), np.int64)
    weights = model.vertex_bone_weights.detach().cpu().numpy()
    indices = model.vertex_bone_indices.cpu().numpy()
    bones = list(model.bone_labels)
    strongest = indices[np.arange(len(indices)), weights.argmax(1)]
    neck = np.isin(strongest, [k for k, b in enumerate(bones) if b.startswith("neck")])
    cranio = _Craniofacial(model)
    points = cranio.weights_matrix.detach().double().numpy() @ V
    names = cranio.labels
    mouth_z = 0.5 * (points[names.index("ch.L"), 2] + points[names.index("ch.R"), 2])
    tragion_y = 0.5 * (points[names.index("t.L"), 1] + points[names.index("t.R"), 1])
    top = points[names.index("v"), 2]
    head = labels == "head"
    jaw = head & (V[:, 2] < mouth_z) & (V[:, 1] < tragion_y + 0.01)
    face = head & ~jaw & (V[:, 1] < tragion_y) & (V[:, 2] < top - 0.06)
    region[head] = REGIONS.index("scalp")
    region[face] = REGIONS.index("face")
    region[jaw] = REGIONS.index("jaw")
    region[neck & ~np.isin(labels, ["hand.L", "hand.R"])] = REGIONS.index("neck")
    region[ear_weights(model) > 0.5] = REGIONS.index("ears")
    region[np.isin(labels, ["hand.L", "hand.R"])] = REGIONS.index("hands")
    region[np.isin(labels, list(EXCLUDED_LABELS))] = REGIONS.index("cavity")
    closed = excluded_vertices(model) & ~np.isin(labels, list(EXCLUDED_LABELS))
    region[closed] = -1
    return region


def level_regions(topology, regions0: np.ndarray) -> np.ndarray:
    """The regions of level 0 carried to a level of the detail layer (the strongest share)."""
    regions0 = np.asarray(regions0)
    shares = np.zeros((len(regions0), len(REGIONS) + 1))
    shares[np.arange(len(regions0)), regions0 + 1] = 1.0  # column 0: the closed parts
    shares = np.asarray(topology.apply(shares))
    return shares.argmax(1) - 1


def seam_vertices(reference, tolerance: float = 5e-4) -> np.ndarray:
    """Mask (V,) of the reference vertices on a border of one skin part that lies on another
    part (within ``tolerance``): the seam of a head part and a body part."""
    from scipy.spatial import cKDTree

    offsets = getattr(reference, "part_offsets", None)
    out = np.zeros(len(reference.vertices), bool)
    if offsets is None or len(offsets) < 3:
        return out
    T = reference.triangles
    edges = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    border = np.unique(unique[counts == 1])
    part = np.searchsorted(offsets, np.arange(len(reference.vertices)), "right") - 1
    for k in range(len(offsets) - 1):
        own = border[part[border] == k]
        others = np.nonzero(part != k)[0]
        if len(own) == 0 or len(others) == 0:
            continue
        d, _ = cKDTree(reference.vertices[others]).query(reference.vertices[own])
        out[own[d < tolerance]] = True
    return out


def reference_regions(
    reference, fitted_vertices: np.ndarray, fitted_regions: np.ndarray
) -> np.ndarray:
    """The region of each reference vertex: that of its nearest fitted vertex (of the measured
    regions), the ears from the reference's ``"ear"`` mask when it has one, and the seam."""
    from scipy.spatial import cKDTree

    rows = np.nonzero(fitted_regions >= 0)[0]
    _, nearest = cKDTree(np.asarray(fitted_vertices)[rows]).query(reference.vertices)
    region = np.asarray(fitted_regions)[rows[nearest]]
    ears = REGIONS.index("ears")
    mask = reference.regions.get("ear") if hasattr(reference, "regions") else None
    if mask is not None:
        others = rows[fitted_regions[rows] != ears]
        _, near = cKDTree(np.asarray(fitted_vertices)[others]).query(
            reference.vertices[region == ears]
        )
        region[region == ears] = fitted_regions[others[near]]
        region[mask] = ears
    seam = seam_vertices(reference)
    if seam.any():
        distance, _ = cKDTree(reference.vertices[seam]).query(reference.vertices)
        region[distance < SEAM_RADIUS] = REGIONS.index("neck_seam")
    return region


@dataclasses.dataclass
class Measures:
    """Per-vertex measures of a fitted mesh, and their summary (:meth:`summary`)."""

    forward: (
        np.ndarray
    )  # (N,) signed distance (metres) of each fitted vertex; NaN if closed
    reverse: (
        np.ndarray
    )  # (R,) signed distance of each reference vertex to the fitted surface
    angle_geometric: (
        np.ndarray
    )  # (N,) degrees, against the reference's geometric normals
    angle_file: (
        np.ndarray
    )  # (N,) degrees, against the normals the reference shades with
    regions: np.ndarray  # (N,) region of each fitted vertex
    reference_regions: np.ndarray  # (R,) region of each reference vertex
    # (N,) the largest angle in degrees between the normals of two triangles that share an
    # edge at the vertex: folds and crumples show as large values
    fold: np.ndarray | None = None

    def summary(self) -> dict:
        """mean / signed mean / p95 / max per region and over all, in mm and degrees."""

        def stats(values, scale):
            values = np.asarray(values, dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                return None
            a = np.abs(values) * scale
            return {
                "mean": round(float(a.mean()), 4),
                "signed": round(float(values.mean() * scale), 4),
                "p95": round(float(np.percentile(a, 95)), 4),
                "max": round(float(a.max()), 4),
                "n": int(len(a)),
            }

        out = {}
        cavity = REGIONS.index("cavity")
        for k, name in enumerate(("all",) + REGIONS):
            fwd = (
                (self.regions >= 0) & (self.regions != cavity)
                if name == "all"
                else self.regions == k - 1
            )
            rev = (
                np.ones(len(self.reverse), bool)
                if name == "all"
                else self.reference_regions == k - 1
            )
            entry = {
                "forward_mm": stats(self.forward[fwd], 1000.0),
                "reverse_mm": stats(self.reverse[rev], 1000.0),
                "angle_geometric_deg": stats(self.angle_geometric[fwd], 1.0),
                "angle_file_deg": stats(self.angle_file[fwd], 1.0),
            }
            if self.fold is not None:
                folds = self.fold[fwd]
                folds = folds[np.isfinite(folds)]
                if len(folds):
                    entry["folded_over_60deg"] = round(float((folds > 60).mean()), 4)
            if any(v is not None for v in entry.values()):
                out[name] = entry
        return out


def measure(
    vertices: np.ndarray,
    triangles: np.ndarray,
    regions: np.ndarray,
    reference,
    reference_region: np.ndarray | None = None,
) -> Measures:
    """
    The measures of a fitted mesh (see the module documentation). ``regions`` (N,) gives the
    region of each fitted vertex (:func:`anny_regions`, carried by :func:`level_regions`; -1
    for the closed parts, which no measure takes), and ``reference_region`` the region of each
    reference vertex (:func:`reference_regions` of these vertices by default; pass one
    labelling to compare several meshes region by region).
    """
    from anny.hair import closest_triangles
    from opensculptboy.fit.reference import vertex_normals

    V = np.asarray(vertices, dtype=np.float64)
    T = np.asarray(triangles)
    regions = np.asarray(regions).copy()
    if reference_region is None:
        reference_region = reference_regions(reference, V, regions)
    measured = regions >= 0
    normals = vertex_normals(V, T)
    C, N, d, tri, bary = reference.closest(V[measured])
    corners = reference.triangles[tri]
    signed = ((V[measured] - C) * N).sum(1)
    forward = np.full(len(V), np.nan)
    forward[measured] = signed
    shading = np.einsum("nk, nkd -> nd", bary, reference.shading_normals[corners])
    shading /= np.maximum(np.linalg.norm(shading, axis=1, keepdims=True), 1e-12)

    def angle(a, b):
        return np.degrees(np.arccos(np.clip((a * b).sum(1), -1.0, 1.0)))

    angle_geometric = np.full(len(V), np.nan)
    angle_file = np.full(len(V), np.nan)
    angle_geometric[measured] = angle(normals[measured], N)
    angle_file[measured] = angle(normals[measured], shading)
    # the fitted vertices on the ears or the seam of the source take that region
    nearest = corners[np.arange(len(corners)), np.argmax(bary, 1)]
    source = reference_region[nearest]
    special = np.isin(source, [REGIONS.index("ears"), REGIONS.index("neck_seam")])
    rows = np.nonzero(measured)[0]
    regions[rows[special]] = source[special]

    fitted = T[measured[T].all(1)]
    tri_f, bary_f, _ = closest_triangles(reference.vertices, V, fitted)
    points = np.einsum("nk, nkd -> nd", bary_f, V[fitted[tri_f]])
    n_f = np.einsum("nk, nkd -> nd", bary_f, normals[fitted[tri_f]])
    n_f /= np.maximum(np.linalg.norm(n_f, axis=1, keepdims=True), 1e-12)
    reverse = -((reference.vertices - points) * n_f).sum(1)
    return Measures(
        forward,
        reverse,
        angle_geometric,
        angle_file,
        regions,
        reference_region,
        fold_angles(V, T),
    )


def fold_angles(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """For each vertex, the largest angle (degrees) between the normals of two triangles that
    share an edge at it (0 for a vertex on no shared edge)."""
    V = np.asarray(vertices, dtype=np.float64)
    T = np.asarray(triangles)
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    edges = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    owner = np.tile(np.arange(len(T)), 3)
    key = np.sort(edges, 1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, owner, edges = key[order], owner[order], edges[order]
    same = (key[1:] == key[:-1]).all(1)
    a, b = owner[:-1][same], owner[1:][same]
    angle = np.degrees(np.arccos(np.clip((n[a] * n[b]).sum(1), -1.0, 1.0)))
    out = np.zeros(len(V))
    np.maximum.at(out, key[:-1][same, 0], angle)
    np.maximum.at(out, key[:-1][same, 1], angle)
    return out

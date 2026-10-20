# OpenSculptBoy
# Apache License, Version 2.0
"""
The reference of a fit: a target surface in Anny's frame, its landmarks, and the closest-point
queries of the registration.

The landmarks of the handoff contract and their counterparts on anny (:data:`LANDMARKS`):

- the face, as craniofacial landmarks of :mod:`anny.faces.measurements` (surface points):
  ``eye_outer_l/r`` (exocanthion), ``eye_inner_l/r`` (endocanthion), ``nose_tip``
  (pronasale), ``mouth_corner_l/r`` (cheilion), ``chin`` (gnathion), ``ear_l/r`` (tragion)
  and ``crown`` (vertex, the top of the head);
- the body, as joint centres (the heads of anny's bones): ``neck_base`` (neck01),
  ``shoulder_l/r`` (upperarm01), ``elbow_l/r`` (lowerarm01), ``wrist_l/r`` (wrist),
  ``hip_l/r`` (upperleg01), ``knee_l/r`` (lowerleg01) and ``ankle_l/r`` (foot).

``l`` and ``r`` are the figure's own left and right (+X and -X in both frames). Names are
matched loosely (:func:`canonical_landmark`): ``LeftEyeOuter``, ``eye.outer.L`` and
``left_eye_outer`` all name ``eye_outer_l``. A ``skin`` or ``surface`` word marks a point
snapped to the skin (``mouth_corner_l_skin``, as the DD2DA tools write next to the joint
position ``mouth_corner_l``): when a source gives both, the surface point serves the face
landmarks, which are skin points on anny, and the other serves the joint centres. Joint centres
differ between rigs, so the fit weighs them less than the surface points of the face.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np

# name -> (kind, anny label, weight); the weight scales the landmark term of the fit.
LANDMARKS = {
    "eye_outer_l": ("craniofacial", "ex.L", 1.0),
    "eye_outer_r": ("craniofacial", "ex.R", 1.0),
    "eye_inner_l": ("craniofacial", "en.L", 1.0),
    "eye_inner_r": ("craniofacial", "en.R", 1.0),
    "nose_tip": ("craniofacial", "prn", 1.0),
    "mouth_corner_l": ("craniofacial", "ch.L", 1.0),
    "mouth_corner_r": ("craniofacial", "ch.R", 1.0),
    "chin": ("craniofacial", "gn", 1.0),
    "ear_l": ("craniofacial", "t.L", 0.5),
    "ear_r": ("craniofacial", "t.R", 0.5),
    "crown": ("craniofacial", "v", 0.5),
    "neck_base": ("bone", "neck01", 0.1),
    "shoulder_l": ("bone", "upperarm01.L", 0.1),
    "shoulder_r": ("bone", "upperarm01.R", 0.1),
    "elbow_l": ("bone", "lowerarm01.L", 0.2),
    "elbow_r": ("bone", "lowerarm01.R", 0.2),
    "wrist_l": ("bone", "wrist.L", 0.2),
    "wrist_r": ("bone", "wrist.R", 0.2),
    "hip_l": ("bone", "upperleg01.L", 0.1),
    "hip_r": ("bone", "upperleg01.R", 0.1),
    "knee_l": ("bone", "lowerleg01.L", 0.2),
    "knee_r": ("bone", "lowerleg01.R", 0.2),
    "ankle_l": ("bone", "foot.L", 0.2),
    "ankle_r": ("bone", "foot.R", 0.2),
}
_SYNONYMS = {
    "outer_eye": "eye_outer",
    "eye_corner_outer": "eye_outer",
    "eye_outer_corner": "eye_outer",
    "exocanthion": "eye_outer",
    "inner_eye": "eye_inner",
    "eye_corner_inner": "eye_inner",
    "eye_inner_corner": "eye_inner",
    "endocanthion": "eye_inner",
    "nose": "nose_tip",
    "pronasale": "nose_tip",
    "mouth": "mouth_corner",
    "mouth_corners": "mouth_corner",
    "lip_corner": "mouth_corner",
    "cheilion": "mouth_corner",
    "gnathion": "chin",
    "chin_tip": "chin",
    "tragion": "ear",
    "head_top": "crown",
    "top_of_head": "crown",
    "vertex": "crown",
    "neck": "neck_base",
    "upper_arm": "shoulder",
    "forearm": "elbow",
    "hand": "wrist",
    "thigh": "hip",
    "upper_leg": "hip",
    "shin": "knee",
    "lower_leg": "knee",
    "foot": "ankle",
}


SURFACE_WORDS = ("skin", "surface")


def _landmark_key(name: str) -> tuple[str | None, bool]:
    """The name of :data:`LANDMARKS` that a landmark name means (or None), and whether the
    name marks a point snapped to the skin."""
    words = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    tokens = [t for t in re.split(r"[^a-z0-9]+", words.lower()) if t]
    side = None
    surface = False
    rest = []
    for token in tokens:
        if token in ("l", "left"):
            side = "l"
        elif token in ("r", "right"):
            side = "r"
        elif token in SURFACE_WORDS:
            surface = True
        else:
            rest.append(token)
    base = "_".join(rest)
    base = _SYNONYMS.get(base, base)
    candidate = f"{base}_{side}" if side else base
    return (candidate if candidate in LANDMARKS else None), surface


def canonical_landmark(name: str) -> str | None:
    """The name of :data:`LANDMARKS` that a landmark name means, or None."""
    return _landmark_key(name)[0]


def canonical_landmarks(landmarks: dict) -> dict[str, np.ndarray]:
    """The landmarks under their names of :data:`LANDMARKS`; unknown names are left out.
    Of two points for one landmark, a face landmark takes the one snapped to the skin and a
    joint centre the other."""
    out, chosen = {}, {}
    for name, point in landmarks.items():
        key, surface = _landmark_key(name)
        if key is None:
            continue
        wanted = LANDMARKS[key][0] == "craniofacial"
        if key in chosen and chosen[key] == wanted:
            continue  # the preferred kind is already in
        if key in chosen and surface != wanted:
            continue  # keep the preferred one that came first
        out[key] = np.asarray(point, dtype=np.float64)
        chosen[key] = surface
    return out


def vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    from opensculptboy.export.body import vertex_normals as normals

    return normals(np.asarray(vertices, dtype=np.float64), triangles)


def auto_landmarks(vertices: np.ndarray) -> dict[str, np.ndarray]:
    """
    Landmarks from the bounding geometry of a standing figure in Anny's frame (Z up, facing
    -Y): ``crown`` (the highest point), ``fingertip_l/r`` (the points furthest along +X and
    -X) and ``sole_l/r`` (the mean of the points within 1 cm of the floor on each side).
    The same rule on anny's mesh gives the counterparts.
    """
    V = np.asarray(vertices, dtype=np.float64)
    centre_x = 0.5 * (V[:, 0].max() + V[:, 0].min())
    low = V[:, 2] < V[:, 2].min() + 0.01
    out = {
        "crown": V[np.argmax(V[:, 2])],
        "fingertip_l": V[np.argmax(V[:, 0])],
        "fingertip_r": V[np.argmin(V[:, 0])],
    }
    for side, mask in (("l", V[:, 0] >= centre_x), ("r", V[:, 0] < centre_x)):
        pick = low & mask
        if pick.any():
            out[f"sole_{side}"] = V[pick].mean(0)
    return out


def auto_landmark_vertices(vertices: np.ndarray) -> dict[str, np.ndarray]:
    """The vertex indices behind each landmark of :func:`auto_landmarks`."""
    V = np.asarray(vertices, dtype=np.float64)
    centre_x = 0.5 * (V[:, 0].max() + V[:, 0].min())
    low = V[:, 2] < V[:, 2].min() + 0.01
    out = {
        "crown": np.array([np.argmax(V[:, 2])]),
        "fingertip_l": np.array([np.argmax(V[:, 0])]),
        "fingertip_r": np.array([np.argmin(V[:, 0])]),
    }
    for side, mask in (("l", V[:, 0] >= centre_x), ("r", V[:, 0] < centre_x)):
        pick = np.nonzero(low & mask)[0]
        if len(pick):
            out[f"sole_{side}"] = pick
    return out


@dataclasses.dataclass
class Reference:
    """
    A target surface in Anny's frame (metres, Z up, facing -Y): the skin of a handoff, or any
    triangle mesh. ``landmarks`` use the names of :data:`LANDMARKS`; ``auto`` tells that they
    come from :func:`auto_landmarks` because the source gave none.
    """

    vertices: np.ndarray  # (V, 3)
    triangles: np.ndarray  # (T, 3)
    landmarks: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    auto: bool = False
    parts: list = dataclasses.field(default_factory=list)  # HandoffPart, skin only
    part_offsets: np.ndarray | None = None  # (P + 1,) first vertex of each part

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.triangles = np.asarray(self.triangles, dtype=np.int64)
        self.normals = vertex_normals(self.vertices, self.triangles)
        self._tree = None

    @classmethod
    def from_mesh(cls, vertices, triangles, landmarks: dict | None = None):
        """A reference from a mesh in Anny's frame; landmarks by any name of
        :func:`canonical_landmark`, or None for the automatic landmarks."""
        landmarks = canonical_landmarks(landmarks or {})
        if landmarks:
            return cls(vertices, triangles, landmarks)
        return cls(vertices, triangles, auto_landmarks(vertices), auto=True)

    @classmethod
    def from_handoff(cls, handoff) -> "Reference":
        """The skin parts of a :class:`~opensculptboy.fit.handoff.Handoff`, merged."""
        parts = handoff.skin_parts()
        offsets = np.cumsum([0] + [len(p.vertices) for p in parts])
        vertices = np.concatenate([p.vertices for p in parts])
        triangles = np.concatenate(
            [p.triangles + offsets[i] for i, p in enumerate(parts)]
        )
        landmarks = canonical_landmarks(handoff.landmarks)
        auto = not landmarks
        if auto:
            landmarks = auto_landmarks(vertices)
        return cls(vertices, triangles, landmarks, auto, list(parts), offsets)

    def transformed(self, rotation: np.ndarray, translation: np.ndarray):
        """The reference moved by x -> R x + t (the parts keep their own coordinates)."""
        R, t = np.asarray(rotation), np.asarray(translation)
        moved = Reference(
            self.vertices @ R.T + t,
            self.triangles,
            {k: R @ v + t for k, v in self.landmarks.items()},
            self.auto,
            self.parts,
            self.part_offsets,
        )
        return moved

    @property
    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.vertices)
        return self._tree

    def closest(self, points: np.ndarray):
        """
        For each point: the closest point of the surface, the interpolated unit normal there,
        the distance, the triangle and the barycentric coordinates.
        """
        from anny.hair import closest_triangles

        points = np.asarray(points, dtype=np.float64)
        tri, bary, distance = closest_triangles(points, self.vertices, self.triangles)
        corners = self.triangles[tri]
        closest = np.einsum("nk, nkd -> nd", bary, self.vertices[corners])
        normals = np.einsum("nk, nkd -> nd", bary, self.normals[corners])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        return closest, normals, distance, tri, bary

    def height(self) -> float:
        return float(np.ptp(self.vertices[:, 2]))

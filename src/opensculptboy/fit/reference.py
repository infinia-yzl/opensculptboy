# OpenSculptBoy
# Apache License, Version 2.0
"""
The reference of a fit: a target surface in Anny's frame, its landmarks, and the closest-point
queries of the registration.

The landmarks of the handoff contract and their counterparts on anny (:data:`LANDMARKS`):

- the face, as craniofacial landmarks of :mod:`anny.faces.measurements` (surface points):
  ``eye_outer_l/r`` (exocanthion), ``eye_inner_l/r`` (endocanthion), ``nose_tip``
  (pronasale), ``mouth_corner_l/r`` (cheilion), ``chin`` (gnathion), ``ear_l/r`` (tragion),
  ``crown`` (vertex, the top of the head), and the outline of the ear: ``ear_top_l/r``
  (superaurale, the highest point of the ear, the tip of a pointed ear), ``ear_lobe_l/r``
  (subaurale, the lowest point of the lobe), ``ear_back_l/r`` (postaurale, the hindmost point)
  and ``ear_lateral_l/r`` (the point furthest from the head);
- the body, as joint centres (the heads of anny's bones): ``neck_base`` (neck01),
  ``shoulder_l/r`` (upperarm01), ``elbow_l/r`` (lowerarm01), ``wrist_l/r`` (wrist),
  ``hip_l/r`` (upperleg01), ``knee_l/r`` (lowerleg01) and ``ankle_l/r`` (foot).

``l`` and ``r`` are the figure's own left and right (+X and -X in both frames). Names are
matched loosely (:func:`canonical_landmark`): ``LeftEyeOuter``, ``eye.outer.L`` and
``left_eye_outer`` all name ``eye_outer_l``. A ``skin`` or ``surface`` word marks a point
snapped to the skin (``mouth_corner_l_skin``, as a source may write next to the joint
position ``mouth_corner_l``): when a source gives both, the surface point serves the face
landmarks, which are skin points on anny, and the other serves the joint centres. Joint centres
differ between rigs, so the fit weighs them less than the surface points of the face, and the
neck base and the hips, which sit 20 to 30 mm apart between rigs, take no weight at all.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np

from opensculptboy.groom.handoff import GROOM_ROLES

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
    "ear_top_l": ("craniofacial", "sa.L", 0.2),
    "ear_top_r": ("craniofacial", "sa.R", 0.2),
    "ear_lobe_l": ("craniofacial", "sba.L", 0.2),
    "ear_lobe_r": ("craniofacial", "sba.R", 0.2),
    "ear_back_l": ("craniofacial", "pa.L", 0.2),
    "ear_back_r": ("craniofacial", "pa.R", 0.2),
    "ear_lateral_l": ("craniofacial", "ela.L", 0.2),
    "ear_lateral_r": ("craniofacial", "ela.R", 0.2),
    "crown": ("craniofacial", "v", 0.5),
    # joint centres that sit 20 to 30 mm apart between rigs: read, reported, not fitted
    "neck_base": ("bone", "neck01", 0.0),
    "shoulder_l": ("bone", "upperarm01.L", 0.1),
    "shoulder_r": ("bone", "upperarm01.R", 0.1),
    "elbow_l": ("bone", "lowerarm01.L", 0.2),
    "elbow_r": ("bone", "lowerarm01.R", 0.2),
    "wrist_l": ("bone", "wrist.L", 0.2),
    "wrist_r": ("bone", "wrist.R", 0.2),
    "hip_l": ("bone", "upperleg01.L", 0.0),
    "hip_r": ("bone", "upperleg01.R", 0.0),
    "knee_l": ("bone", "lowerleg01.L", 0.2),
    "knee_r": ("bone", "lowerleg01.R", 0.2),
    "ankle_l": ("bone", "foot.L", 0.2),
    "ankle_r": ("bone", "foot.R", 0.2),
    # the tip of the second toe (MediaPipe's foot_index): turns a foot without data
    "toe_l": ("bone", "toe2-3.L", 0.2),
    "toe_r": ("bone", "toe2-3.R", 0.2),
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
    "ear_tip": "ear_top",
    "helix_tip": "ear_top",
    "superaurale": "ear_top",
    "earlobe": "ear_lobe",
    "ear_lobe_bottom": "ear_lobe",
    "subaurale": "ear_lobe",
    "postaurale": "ear_back",
    "ear_outer": "ear_lateral",
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
    "foot_index": "toe",
    "toes": "toe",
    "toe_tip": "toe",
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


def fit_sphere(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least-squares sphere through points: centre, radius and the largest distance of a
    point from the sphere."""
    P = np.asarray(points, dtype=np.float64)
    A = np.c_[2.0 * P, np.ones(len(P))]
    x = np.linalg.lstsq(A, (P**2).sum(1), rcond=None)[0]
    centre = x[:3]
    radius = float(np.sqrt(max(x[3] + centre @ centre, 0.0)))
    spread = float(np.abs(np.linalg.norm(P - centre, axis=1) - radius).max())
    return centre, radius, spread


def eye_spheres(parts: list, tolerance: float = 0.15) -> list:
    """
    The eyeballs of a reference as spheres [(centre, radius)]: one sphere per part whose
    points lie within ``tolerance`` of a radius from it. Of spheres whose centres lie inside
    one another (an eyeball made of a sclera and a cornea), the largest is the eyeball.
    Parts of another shape (an eye made of flat cards) are left out.
    """
    spheres = []
    for vertices in parts:
        vertices = np.asarray(vertices, dtype=np.float64)
        if len(vertices) < 10:
            continue
        centre, radius, spread = fit_sphere(vertices)
        if not radius < 1.5 * np.linalg.norm(np.ptp(vertices, axis=0)):
            continue  # a flat or nearly flat part: no sphere
        if not spread < tolerance * radius:
            continue
        for k, (c, r) in enumerate(spheres):
            if np.linalg.norm(centre - c) < max(radius, r):
                if radius > r:
                    spheres[k] = (centre, radius)
                break
        else:
            spheres.append((centre, radius))
    return spheres


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
    # the eyeballs as spheres: [(centre (3,), radius)], from the parts of role "eyes"
    eyes: list = dataclasses.field(default_factory=list)
    # the parts of role "eyes" as a reference of their own, for the texture bake
    eye_reference: "Reference | None" = None
    # the source's own vertex normals (V, 3), its shading; None: the geometric normals
    file_normals: np.ndarray | None = None
    # named vertex masks (V,) of the source: "ear" (masks.ear of the handoff) and
    # "ear_attachment" (its vertices along the line where each ear meets the head)
    regions: dict = dataclasses.field(default_factory=dict)
    # the parts of role "mouth" (teeth, gums, tongue), for the teeth of the bake
    mouth_parts: list = dataclasses.field(default_factory=list)
    # the converted maps of the inner mouth by slot (paths), from the handoff's ``mouth``
    mouth_textures: dict = dataclasses.field(default_factory=dict)
    # the parts of role "lashes", for the lashes of the bake
    lash_parts: list = dataclasses.field(default_factory=list)
    # the parts of roles "hair" and "brows", for the grooms of the bake (opensculptboy.groom)
    groom_parts: list = dataclasses.field(default_factory=list)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.triangles = np.asarray(self.triangles, dtype=np.int64)
        self.normals = vertex_normals(self.vertices, self.triangles)
        if self.file_normals is not None:
            self.file_normals = np.asarray(self.file_normals, dtype=np.float64)
        self.regions = {k: np.asarray(v, bool) for k, v in (self.regions or {}).items()}
        self._tree = None
        self._folds = None

    @property
    def folds(self) -> np.ndarray:
        """The largest angle (degrees) between the normals of two triangles that share an edge
        at each vertex (``opensculptboy.fit.metrics.fold_angles``): the reference's own
        creases, such as the rim of a thin ear. The edges are those of the welded surface
        (``register.welded_ids``), so that the seams where the source splits its vertices (UV
        seams, an ear stitched to the head) fold as they look."""
        if self._folds is None:
            from opensculptboy.fit.metrics import fold_angles
            from opensculptboy.fit.register import welded_ids

            ids = welded_ids(self.vertices)
            _, first, joined = np.unique(ids, return_index=True, return_inverse=True)
            joined = joined.ravel()
            self._folds = fold_angles(self.vertices[first], joined[self.triangles])[
                joined
            ]
        return self._folds

    @property
    def shading_normals(self) -> np.ndarray:
        """The normals that the source shades with: its own where it has them."""
        return self.normals if self.file_normals is None else self.file_normals

    @property
    def face_normals(self) -> np.ndarray:
        """Unit normals (T, 3) of the triangles."""
        p = self.vertices[self.triangles]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)

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
        file_normals = None
        if any(p.normals is not None for p in parts):
            file_normals = np.concatenate(
                [
                    p.normals
                    if p.normals is not None
                    else vertex_normals(p.vertices, p.triangles)
                    for p in parts
                ]
            )
        regions = {}
        ear = handoff.mask("ear", list(parts)) if hasattr(handoff, "mask") else None
        if ear is not None and ear.any():
            regions["ear"] = ear
            attachment = handoff.mask("ear", list(parts), "attachment")
            if attachment is not None and attachment.any():
                regions["ear_attachment"] = attachment
        eye_parts = [p for p in handoff.parts if p.role == "eyes"]
        eyes = eye_spheres([p.vertices for p in eye_parts])
        eye_reference = None
        if eye_parts:
            eye_offsets = np.cumsum([0] + [len(p.vertices) for p in eye_parts])
            eye_reference = cls(
                np.concatenate([p.vertices for p in eye_parts]),
                np.concatenate(
                    [p.triangles + eye_offsets[i] for i, p in enumerate(eye_parts)]
                ),
                parts=eye_parts,
                part_offsets=eye_offsets,
            )
        return cls(
            vertices,
            triangles,
            landmarks,
            auto,
            list(parts),
            offsets,
            eyes,
            eye_reference,
            file_normals,
            regions,
            [p for p in handoff.parts if p.role == "mouth"],
            dict(getattr(handoff, "mouth_textures", None) or {}),
            [p for p in handoff.parts if p.role == "lashes"],
            [p for p in handoff.parts if p.role in GROOM_ROLES],
        )

    def eye_values(self) -> dict:
        """
        The values of the reference's eyes that a card's eyes can take
        (``material.reference_eyes``, :func:`opensculptboy.material.check_reference_eyes`):
        for each eyeball sphere, the eye part whose point lies farthest in front of it along
        the gaze is the eye's outer surface (a cornea, or an eyeball drawn as one surface);
        ``cornea_radius_mm`` is the radius of curvature of that surface's front
        (:func:`opensculptboy.eyes.geometry.front_curvature_mm`), ``cornea_roughness`` its
        roughness (the median of its texture, or its factor) and ``cornea_ior`` its index of
        refraction, each the mean of both eyes. Keys whose values the reference does not give
        are left out; empty without eyes.
        """
        from opensculptboy.eyes.geometry import GAZE, front_curvature_mm

        eyes = self.eye_reference
        if eyes is None or not self.eyes or eyes.part_offsets is None:
            return {}
        found = {"cornea_radius_mm": [], "cornea_roughness": [], "cornea_ior": []}
        offsets = eyes.part_offsets
        for centre, radius in self.eyes:
            best, front = None, -np.inf
            for k, part in enumerate(eyes.parts):
                points = eyes.vertices[offsets[k] : offsets[k + 1]]
                near = np.linalg.norm(points - centre, axis=1) < 1.5 * radius
                if not near.any():
                    continue
                ahead = float(((points[near] - centre) @ GAZE).max())
                if ahead > front:
                    best, front = k, ahead
            if best is None:
                continue
            part = eyes.parts[best]
            points = eyes.vertices[offsets[best] : offsets[best + 1]]
            curvature = front_curvature_mm(points, centre)
            if curvature is not None and 5.0 <= curvature <= 20.0:
                found["cornea_radius_mm"].append(curvature)
            roughness = getattr(part, "roughness", None)
            if roughness is not None:
                found["cornea_roughness"].append(float(np.median(roughness)))
            ior = getattr(part, "ior", None)
            if ior is not None and 1.0 <= ior <= 2.0:
                found["cornea_ior"].append(float(ior))
        return {
            key: round(float(np.mean(values)), 3)
            for key, values in found.items()
            if values
        }

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
            [(R @ c + t, r) for c, r in self.eyes],
            None
            if self.eye_reference is None
            else self.eye_reference.transformed(R, t),
            None if self.file_normals is None else self.file_normals @ R.T,
            self.regions,
            [
                dataclasses.replace(
                    p,
                    vertices=p.vertices @ R.T + t,
                    normals=None if p.normals is None else p.normals @ R.T,
                )
                for p in self.mouth_parts
            ],
            self.mouth_textures,
            [
                dataclasses.replace(
                    p,
                    vertices=p.vertices @ R.T + t,
                    normals=None if p.normals is None else p.normals @ R.T,
                )
                for p in self.lash_parts
            ],
            [
                dataclasses.replace(
                    p,
                    vertices=p.vertices @ R.T + t,
                    normals=None if p.normals is None else p.normals @ R.T,
                    joint_positions=None
                    if p.joint_positions is None
                    else {k: R @ v + t for k, v in p.joint_positions.items()},
                )
                for p in self.groom_parts
            ],
        )
        return moved

    @property
    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.vertices)
        return self._tree

    def closest(self, points: np.ndarray, smooth: float = 0.0, shading: bool = False):
        """
        For each point: the closest point of the surface, the interpolated unit normal there,
        the distance, the triangle and the barycentric coordinates.

        With ``smooth`` > 0 the surface is the reference's Phong tessellation (Boubekeur and
        Alexa 2008) instead of its flat triangles: the closest point c of a triangle, with
        barycentric coordinates b, moves to (1 - smooth) c + smooth sum_i b_i pi_i(c), where
        pi_i projects onto the tangent plane of corner i. The vertices stay where they are and
        the surface between them curves with the vertex normals, as the shading of a coarse
        mesh suggests; a fine mesh fitted to the flat triangles would copy their facets.
        ``shading`` takes the normals that the source shades with
        (:attr:`shading_normals`) for the tessellation and the returned normals. A triangle
        whose corner normals part by more than 60 degrees (the rim of a thin part such as an
        ear) stays flat, and one between 37 and 60 degrees curves less.
        """
        from anny.hair import closest_triangles

        points = np.asarray(points, dtype=np.float64)
        tri, bary, distance = closest_triangles(points, self.vertices, self.triangles)
        corners = self.triangles[tri]
        closest = np.einsum("nk, nkd -> nd", bary, self.vertices[corners])
        vertex_normals_ = self.shading_normals if shading else self.normals
        corner_normals = vertex_normals_[corners]
        normals = np.einsum("nk, nkd -> nd", bary, corner_normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        if smooth > 0:
            offsets = closest[:, None] - self.vertices[corners]
            heights = (offsets * corner_normals).sum(-1)
            # a triangle whose corner normals part by more than 60 degrees (the rim of a thin
            # part, a crease) stays flat: its tessellation would overshoot
            n0, n1, n2 = (corner_normals[:, k] for k in range(3))
            agree = np.minimum.reduce(
                [(n0 * n1).sum(1), (n1 * n2).sum(1), (n2 * n0).sum(1)]
            )
            share = smooth * np.clip((agree - 0.5) / 0.3, 0.0, 1.0)
            closest = closest - np.einsum(
                "n, nk, nk, nkd -> nd", share, bary, heights, corner_normals
            )
            distance = np.linalg.norm(points - closest, axis=1)
        return closest, normals, distance, tri, bary

    def subset(self, triangles: np.ndarray) -> "Reference":
        """The reference restricted to a mask of its triangles, with the normals of the whole
        surface, its landmarks and its regions (compacted), for closest-point queries within a
        region. ``source_vertices`` maps its vertices back."""
        triangles = self.triangles[np.asarray(triangles, bool)]
        used = np.unique(triangles)
        remap = np.full(len(self.vertices), -1, np.int64)
        remap[used] = np.arange(len(used))
        out = Reference(
            self.vertices[used],
            remap[triangles],
            dict(self.landmarks),
            self.auto,
            file_normals=None if self.file_normals is None else self.file_normals[used],
            regions={k: v[used] for k, v in self.regions.items()},
        )
        out.normals = self.normals[used]  # the normals of the whole surface
        out.source_vertices = used
        return out

    def height(self) -> float:
        return float(np.ptp(self.vertices[:, 2]))

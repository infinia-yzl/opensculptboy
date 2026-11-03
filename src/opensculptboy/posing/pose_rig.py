# OpenSculptBoy
# Apache License, Version 2.0
"""
An exact forward model of Anny's skinning in NumPy, for solves that turn a few bones many times.

Anny's ``local-ref`` pose gives each bone a rotation ``D`` in the axes of the reference pose. For
every bone below the root, the bone's transform is its parent's transform followed by a turn by
``D`` about the bone's head in the reference pose (``h``):

    G_b = G_parent(b) . Rot_h_b(D_b),   Rot_h(D): x -> D (x - h) + h

so a pose is a chain of turns about fixed points. The root's own transform and the card's
placement are one fixed affine map ``frame``. A vertex skins as
``x = frame( sum_k w_k G_k C_k [v; 1] )``, where ``C_k`` takes the rest bone frame to the
reference one (``ref_k rest_k^-1``). Every rigid transform here is linear in the homogeneous
vertex, so a point given by barycentric weights on a triangle is a sum of influences
``(bone, s, u)``: ``x = frame( sum_b R_b u_b + s_b t_b )`` with ``R_b, t_b`` the bone's turn
chain. :class:`PoseRig` holds the reference heads, the root frame and the skinning data;
:meth:`PoseRig.from_card` builds it from a card with Anny's own model (torch), and the rest of
the module needs NumPy alone, as the web page's twin (``viewer/src/pose_edit/solve.ts``) does.

The model gives the same vertices: :func:`PoseRig.vertices` matches Anny's posed level-0 mesh to
the rounding of float64 (``test_reference_points``).
"""

from __future__ import annotations

import dataclasses

import numpy as np


def _inverse_rigid(T: np.ndarray) -> np.ndarray:
    out = np.zeros_like(T)
    R = T[..., :3, :3]
    out[..., :3, :3] = np.swapaxes(R, -1, -2)
    out[..., :3, 3] = -np.einsum("...ji,...j->...i", R, T[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def topological_order(parents) -> np.ndarray:
    """The bones with every parent before its children."""
    parents = np.asarray(parents)
    order, seen = [], np.zeros(len(parents), bool)
    while len(order) < len(parents):
        for b, p in enumerate(parents):
            if not seen[b] and (p < 0 or seen[p]):
                order.append(b)
                seen[b] = True
    return np.asarray(order)


@dataclasses.dataclass
class PoseRig:
    """Anny's rig for one card (sliders and local changes held), in NumPy.

    ``heads`` (B, 3): each bone's head in the reference pose (the space of the turns).
    ``bind`` (B, 4, 4): ``C_b``, from the rest bone frame to the reference one. ``frame``
    (4, 4): the root bone's own transform followed by the card's placement, from the space of
    the turns to the output frame. ``rest_vertices`` (V, 3), ``weights`` and ``indices`` (V, K)
    and ``faces`` (F, 3): the level-0 mesh."""

    labels: list
    parents: np.ndarray
    heads: np.ndarray
    bind: np.ndarray
    frame: np.ndarray
    rest_vertices: np.ndarray
    weights: np.ndarray
    indices: np.ndarray
    faces: np.ndarray
    root: int = 0

    def __post_init__(self):
        self.parents = np.asarray(self.parents, np.int64)
        self.order = topological_order(self.parents)
        self.index = {n: i for i, n in enumerate(self.labels)}
        self.children = [[] for _ in self.labels]
        for b, p in enumerate(self.parents):
            if p >= 0:
                self.children[p].append(b)

    # ------------------------------------------------------------------ building
    @classmethod
    def from_arrays(
        cls,
        labels,
        parents,
        rest_bone_poses,
        ref_bone_poses,
        root_parameter,
        rest_vertices,
        weights,
        indices,
        faces,
        rotation=None,
        translation=None,
    ) -> "PoseRig":
        """The rig of Anny's arrays: the rest and reference bone poses (B, 4, 4), the root's
        ``local-ref`` parameter (4, 4) and the placement ``x -> rotation x + translation``."""
        rest = np.asarray(rest_bone_poses, np.float64)
        ref = np.asarray(ref_bone_poses, np.float64)
        parents = np.asarray(parents, np.int64)
        root = int(np.nonzero(parents < 0)[0][0])
        # the root: T_0 = Rr^T D Rr (translation Rr^T t), G_0 = T_0 ref_0^-1 (Anny's
        # parallel_forward_kinematic with the base transform ref_0^-1)
        D = np.asarray(root_parameter, np.float64)
        Rr = ref[root, :3, :3]
        T0 = np.eye(4)
        T0[:3, :3] = Rr.T @ D[:3, :3] @ Rr
        T0[:3, 3] = Rr.T @ D[:3, 3]
        G0 = T0 @ _inverse_rigid(ref[root])
        # below the root, G_b = G_0 K_b with K_b the chain of turns about the reference heads;
        # the root's own skinning matrix G_0 C_0 is K_root = identity
        place = np.eye(4)
        if rotation is not None:
            place[:3, :3] = np.asarray(rotation, np.float64)
        if translation is not None:
            place[:3, 3] = np.asarray(translation, np.float64)
        bind = ref @ np.linalg.inv(rest)
        return cls(
            labels=list(labels),
            parents=parents,
            heads=ref[:, :3, 3].copy(),
            bind=bind,
            frame=place @ G0,
            rest_vertices=np.asarray(rest_vertices, np.float64),
            weights=np.asarray(weights, np.float64),
            indices=np.asarray(indices, np.int64),
            faces=np.asarray(faces, np.int64),
            root=root,
        )

    @classmethod
    def from_card(cls, model, character, rotation=None, translation=None, pose=None):
        """The rig of a card on Anny's model, and the card's pose (or ``pose``, Anny's pose
        parameters (1, B, 4, 4)) as rotations (B, 3, 3)."""
        import torch

        from anny.utils import kinematics

        if pose is None:
            pose = character.pose_parameters(model)
        if pose is None:
            pose = torch.eye(4, dtype=model.template_vertices.dtype).repeat(
                1, len(model.bone_labels), 1, 1
            )
        with torch.no_grad():
            out = model(
                pose_parameters=pose,
                phenotype_kwargs=dict(character.phenotype) or None,
                local_changes_kwargs=dict(character.local_changes) or None,
                face_shape_kwargs=dict(character.face_shapes) or None,
            )
            rest_bones = out["rest_bone_poses"]
            if model.reference_bone_orientations is not None:
                ref, _ = kinematics.parallel_forward_kinematic_absolute_orientations(
                    model.kinematic_propagation_fronts,
                    rest_bone_poses=rest_bones,
                    absolute_orientations=model.reference_bone_orientations[None],
                )
            else:
                ref = rest_bones
        P = pose[0].detach().double().cpu().numpy()
        parents = np.asarray(model.bone_parents)
        root = int(np.nonzero(parents < 0)[0][0])
        rig = cls.from_arrays(
            model.bone_labels,
            parents,
            rest_bones[0].double().cpu().numpy(),
            ref[0].double().cpu().numpy(),
            P[root],
            out["rest_vertices"][0].double().cpu().numpy(),
            model.vertex_bone_weights.detach().double().cpu().numpy(),
            model.vertex_bone_indices.cpu().numpy(),
            model.faces.cpu().numpy(),
            rotation,
            translation,
        )
        return rig, P[:, :3, :3].copy()

    # ------------------------------------------------------------------ forward model
    def transforms(self, rotations: np.ndarray) -> np.ndarray:
        """The chains of turns K (B, 4, 4) in the space of the turns, for rotations (B, 3, 3);
        the root's is the identity (its own turn is in ``frame``)."""
        K = np.zeros((len(self.labels), 4, 4))
        K[:, 3, 3] = 1.0
        H = self.heads
        for b in self.order:
            p = self.parents[b]
            if p < 0:
                K[b, :3, :3] = np.eye(3)
                continue
            D = rotations[b]
            local = np.eye(4)
            local[:3, :3] = D
            local[:3, 3] = H[b] - D @ H[b]
            K[b] = K[p] @ local
        return K

    def joints(self, K: np.ndarray) -> np.ndarray:
        """The posed heads (B, 3) in the space of the turns."""
        return np.einsum("bij,bj->bi", K[:, :3, :3], self.heads) + K[:, :3, 3]

    def to_output(self, X: np.ndarray) -> np.ndarray:
        """Points of the space of the turns in the output frame (the card's placement)."""
        return X @ self.frame[:3, :3].T + self.frame[:3, 3]

    def from_output(self, X: np.ndarray) -> np.ndarray:
        A = self.frame[:3, :3]
        return (np.asarray(X, np.float64) - self.frame[:3, 3]) @ np.linalg.inv(A).T

    def vertices(self, rotations: np.ndarray, rows=None) -> np.ndarray:
        """The posed level-0 vertices (or the ``rows``) in the output frame."""
        K = self.transforms(rotations)
        M = K @ self.bind
        rows = slice(None) if rows is None else rows
        v, w, i = self.rest_vertices[rows], self.weights[rows], self.indices[rows]
        Mi = M[i]  # (V, K, 4, 4)
        X = np.einsum("vkij,vj->vki", Mi[..., :3, :3], v) + Mi[..., :3, 3]
        return self.to_output((w[..., None] * X).sum(1))

    # ------------------------------------------------------------------ points
    def influences(self, triangles, barycentric) -> list:
        """For points given by triangles (P,) and barycentric weights (P, 3), the influences
        of each: arrays ``bones`` (n,), ``s`` (n,) and ``u`` (n, 3), with which the point is
        ``sum R_b u + s t_b`` in the space of the turns."""
        out = []
        tri = np.asarray(triangles, np.int64)
        bary = np.asarray(barycentric, np.float64)
        for t, w3 in zip(tri, bary):
            acc = {}
            for vi, a in zip(self.faces[t], w3):
                v = np.append(self.rest_vertices[vi], 1.0)
                for b, w in zip(self.indices[vi], self.weights[vi]):
                    if w == 0.0:
                        continue
                    h = a * w * (self.bind[b] @ v)
                    acc[int(b)] = acc.get(int(b), 0.0) + h
            bones = np.asarray(sorted(acc), np.int64)
            H = np.stack([acc[b] for b in bones]) if len(bones) else np.zeros((0, 4))
            out.append({"bones": bones, "u": H[:, :3].copy(), "s": H[:, 3].copy()})
        return out

    def points(self, K: np.ndarray, influences: list) -> np.ndarray:
        """The points (P, 3) of ``influences`` in the space of the turns."""
        out = np.empty((len(influences), 3))
        for n, f in enumerate(influences):
            Kb = K[f["bones"]]
            out[n] = (
                np.einsum("bij,bj->i", Kb[:, :3, :3], f["u"]) + f["s"] @ Kb[:, :3, 3]
            )
        return out

    def subtree(self, bone: int) -> np.ndarray:
        """Mask (B,) of a bone and its descendants."""
        inside = np.zeros(len(self.labels), bool)
        stack = [bone]
        while stack:
            b = stack.pop()
            inside[b] = True
            stack.extend(self.children[b])
        return inside

    def to_json(self, influences=None) -> dict:
        """The rig as the page's solver reads it (``viewer/src/pose_edit/solve.ts``)."""
        data = {
            "labels": list(self.labels),
            "parents": self.parents.tolist(),
            "heads": self.heads.tolist(),
            "frame": self.frame.tolist(),
        }
        if influences is not None:
            data["influences"] = [
                {
                    "bones": f["bones"].tolist(),
                    "s": f["s"].tolist(),
                    "u": f["u"].tolist(),
                }
                for f in influences
            ]
        return data

# OpenSculptBoy
# Apache License, Version 2.0
"""
The hair and the brows of a reference character, carried onto a fitted card's head.

The fit leaves the parts of the roles ``hair`` and ``brows`` out (they are not skin), and the
bake carries them into the card's rest frame through the fit's correspondence
(:func:`carry`): each point moves with the fitted head as the fit placed and posed it, by the
rigid motion (Kabsch) that takes the head back to its rest pose, plus two fields of the head's
vertices blended at the point (Gaussian weights of its nearest head vertices on the placed
mesh): the head's own residual from that rigid motion, and the offset from the reference's skin
to the fitted surface, so that a brow strip that lies 0.5 mm over the reference's skin lies
0.5 mm over the card's. The hair, whose skin reaches the neck and the shoulders, takes the
offset alone and then the inverse of the fit's pose blended with its own skin on Anny's bones
(:meth:`Carrier.unpose`), so that the exports, which pose the hair with that skin, put the
strands over the nape where the reference has them.

:func:`reference_groom` builds the :class:`~opensculptboy.groom.data.GroomData` of one kind:
the parts' points, normals, UVs and vertex colours, one material per material of the parts,
and for the hair its skin on Anny's bones (:func:`anny_skin`): the reference's joint with the
most weight is the head, the joints below it that weight the hair are chain joints (their
weight goes to the head, and each chain becomes a spring chain from its first joint down to a
tail one segment past its last joint), and every other joint becomes Anny's nearest bone of the
torso, the neck or the head. The triangles of a blended material are sorted from the inside out
(by the distance of their centres from the centre of the head), so that an engine that draws
them in order draws the outer cards over the inner ones from any view outside the head.
"""

from __future__ import annotations

import numpy as np

from opensculptboy.groom.data import GroomData, GroomMaterial

# the Gaussian of the blended fields: the nearest head vertices, at a scale of at least this
# (metres) or 1.5 times the distance of the nearest one
NEAREST = 16
SPREAD = 0.004
# the head's vertices whose offset to the reference's skin stays within this (metres) give the
# offset field: the others lie inside the head (the mouth's cavity, the eye sockets), where the
# reference has no skin
SURFACE = 0.003
# the Anny bones that a joint of the hair may become
HAIR_BONES = (
    "spine03",
    "spine02",
    "spine01",
    "clavicle.L",
    "clavicle.R",
    "neck01",
    "neck02",
    "neck03",
    "head",
)
# a chain joint's tail lies this share of its last segment past its last joint
TAIL_SHARE = 1.0


class Carrier:
    """
    The carry of points of the reference's frame into the card's rest frame (see the module
    documentation): ``posed`` and ``rest`` are the fitted mesh's vertices as the fit placed it
    on the reference and at rest, ``head`` the rows of its head, and ``reference`` the
    :class:`~opensculptboy.fit.reference.Reference` whose skin gives the offset to the fitted
    surface (None: no offset). ``R`` and ``t`` are the rigid part: x -> R x + t.
    """

    def __init__(self, posed, rest, head, reference=None, motion=None):
        from scipy.spatial import cKDTree

        from opensculptboy.teeth.mesh import rigid

        hp = np.asarray(posed, dtype=np.float64)[head]
        hr = np.asarray(rest, dtype=np.float64)[head]
        self.R, self.t = rigid(hp, hr)
        self.residual = hr - (hp @ self.R.T + self.t)
        self.offset = np.zeros_like(hp)
        surface = np.arange(len(hp))
        if reference is not None:
            self.offset = hp - reference.closest(hp)[0]
            near = np.linalg.norm(self.offset, axis=1) <= SURFACE
            if near.sum() >= 8:
                surface = np.flatnonzero(near)
        self.tree = cKDTree(hp)
        self.count = len(hp)
        # the offset field from the head's surface alone
        self.surface = surface
        self.surface_tree = cKDTree(hp[surface])
        self.centre = hr.mean(0)
        # the fit's pose: (bone transforms (J, 4, 4) that take rest points to posed ones in
        # Anny's frame, the placement R_p, t_p that takes posed points to the reference's frame)
        self.motion = motion

    def _blend(self, points, surface: bool = False):
        """(P, k) indices of the nearest head vertices (of its surface, rows of the offset
        field) and their Gaussian weights."""
        tree = self.surface_tree if surface else self.tree
        k = min(NEAREST, len(self.surface) if surface else self.count)
        d, idx = tree.query(points, k=k)
        d, idx = d.reshape(len(points), k), idx.reshape(len(points), k)
        scale = np.maximum(1.5 * d[:, :1], SPREAD)
        w = np.exp(-((d / scale) ** 2))
        w /= np.maximum(w.sum(1, keepdims=True), 1e-300)
        return idx, w

    def unpose(self, points, normals, slots, weights):
        """
        Points and normals of the reference's frame in the card's rest frame by the inverse of
        the fit's pose blended with a skin on Anny's bones (``slots`` (P, k) bone indices,
        ``weights`` (P, k)): the offset to the fitted surface, then the placement undone, then
        the inverse of each point's blended bone transform. Without the fit's pose, as
        :meth:`points`.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self.motion is None:
            R = self.R
            return self.points(points), np.asarray(normals) @ R.T
        transforms, Rp, tp = self.motion
        x = points + self._shift(points)
        x = (x - tp) @ Rp  # the placement undone: Anny's frame, posed
        M = np.einsum("pk,pkij->pij", weights, transforms[slots])
        inverse = np.linalg.inv(M)
        rest = np.einsum("pij,pj->pi", inverse[:, :3, :3], x) + inverse[:, :3, 3]
        n = np.einsum("pij,pj->pi", inverse[:, :3, :3], np.asarray(normals) @ Rp)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        return rest, n

    def points(self, points) -> np.ndarray:
        """(P, 3) the points in the card's rest frame."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        idx, w = self._blend(points)
        lift = np.einsum("pk,pkd->pd", w, self.residual[idx])
        return (points + self._shift(points)) @ self.R.T + self.t + lift

    def _shift(self, points) -> np.ndarray:
        """(P, 3) the offset to the fitted surface, blended over the head's surface."""
        idx, w = self._blend(points, surface=True)
        return np.einsum("pk,pkd->pd", w, self.offset[self.surface][idx])


def carry(points, posed, rest, head, reference=None) -> np.ndarray:
    """Points (P, 3) of the reference's frame in the card's rest frame (:class:`Carrier`)."""
    return Carrier(posed, rest, head, reference).points(points)


def _descends(joint, ancestor, parents) -> bool:
    seen = 0
    while joint is not None and seen < 256:
        joint = parents.get(joint)
        if joint == ancestor:
            return True
        seen += 1
    return False


def anny_skin(parts, R, t, bone_heads, labels, k: int = 4):
    """
    The skin of the hair's points on Anny's bones (see the module documentation):
    (bone names, (M, k) slots into them, (M, k) weights, the chains [(n, 3) points in the
    card's rest frame], (M,) each point's share of weight on chain joints). ``parts`` are the
    hair's :class:`~opensculptboy.fit.handoff.HandoffPart` items in their order, ``R`` and
    ``t`` the rigid motion of the reference's frame into the card's rest frame, and
    ``bone_heads`` (J, 3) the rest heads of the bones ``labels``.
    """
    labels = list(labels)
    count = sum(len(p.vertices) for p in parts)
    # the total weight of each joint over the parts, and the joints' places and parents
    totals: dict = {}
    positions: dict = {}
    parents: dict = {}
    for p in parts:
        if p.joints is None or p.joint_slots is None:
            continue
        for slot, name in enumerate(p.joints):
            totals[name] = totals.get(name, 0.0) + float(
                p.joint_weights[p.joint_slots == slot].sum()
            )
        positions.update(p.joint_positions or {})
        parents.update(p.joint_parents or {})
    head_bone = labels.index("head") if "head" in labels else 0
    if not totals:
        slots = np.full((count, 1), 0, np.int64)
        return ["head"], slots, np.ones((count, 1)), [], np.zeros(count)
    head_joint = max(totals, key=totals.get)
    weighted = [j for j, w in totals.items() if w > 0]
    chain_joints = {j for j in weighted if _descends(j, head_joint, parents)}
    allowed = [labels.index(b) for b in HAIR_BONES if b in labels] or [head_bone]
    bone_of = {}
    for name in totals:
        if name == head_joint or name in chain_joints or name not in positions:
            bone_of[name] = head_bone
            continue
        at = R @ positions[name] + t
        d = np.linalg.norm(np.asarray(bone_heads)[allowed] - at, axis=1)
        bone_of[name] = allowed[int(np.argmin(d))]
    used = sorted(set(bone_of.values()) | {head_bone})
    local = {b: i for i, b in enumerate(used)}
    dense = np.zeros((count, len(used)))
    share = np.zeros(count)
    start = 0
    for p in parts:
        n = len(p.vertices)
        if p.joints is None or p.joint_slots is None:
            dense[start : start + n, local[head_bone]] = 1.0
        else:
            for slot, name in enumerate(p.joints):
                w = np.where(p.joint_slots == slot, p.joint_weights, 0.0).sum(1)
                dense[start : start + n, local[bone_of.get(name, head_bone)]] += w
                if name in chain_joints:
                    share[start : start + n] += w
            empty = dense[start : start + n].sum(1) <= 0
            dense[start : start + n][empty, local[head_bone]] = 1.0
        start += n
    order = np.argsort(-dense, axis=1, kind="stable")[:, :k]
    weights = np.take_along_axis(dense, order, 1)
    weights /= np.maximum(weights.sum(1, keepdims=True), 1e-12)
    slots = np.where(weights > 0, order, 0)
    # the chains: from each chain joint whose parent is no chain joint, down its first
    # children, with a tail one segment past the last joint
    children: dict = {}
    for name, parent in parents.items():
        children.setdefault(parent, []).append(name)
    chains = []
    for root in sorted(chain_joints):
        if parents.get(root) in chain_joints or root not in positions:
            continue
        line = [root]
        while children.get(line[-1]):
            nxt = sorted(c for c in children[line[-1]] if c in positions)
            if not nxt:
                break
            line.append(nxt[0])
        pts = [R @ positions[j] + t for j in line]
        if len(pts) >= 2:
            pts.append(pts[-1] + TAIL_SHARE * (pts[-1] - pts[-2]))
        else:
            continue
        chains.append(np.asarray(pts))
    return [labels[b] for b in used], slots, weights, chains, np.clip(share, 0.0, 1.0)


def _sorted_inside_out(points, triangles, centre) -> np.ndarray:
    """The triangles in increasing distance of their centres from ``centre``."""
    d = np.linalg.norm(points[triangles].mean(1) - centre, axis=1)
    return triangles[np.argsort(d, kind="stable")]


def reference_groom(
    parts,
    kind: str,
    posed,
    rest,
    head,
    reference=None,
    bone_heads=None,
    labels=None,
    carrier: Carrier | None = None,
):
    """
    The groom of ``kind`` (``"hair"`` or ``"brows"``) of a reference's parts (those of that
    role) in the rest frame of a fitted mesh (see the module documentation), and its textures:
    (:class:`~opensculptboy.groom.data.GroomData`, ``{material name: {slot: (H, W, C) uint8}}``),
    or None without such parts. ``posed``, ``rest``, ``head`` and ``reference`` are as for
    :func:`carry` (``carrier``, a :class:`Carrier` of them, saves building it again, and with
    the fit's pose undoes it for the hair with the hair's own skin, :meth:`Carrier.unpose`);
    ``bone_heads`` and ``labels`` (the rest heads and names of the card's bones) give the hair
    its skin.
    """
    from opensculptboy.export.body import vertex_normals

    mine = [p for p in parts if p.role == kind and len(p.triangles)]
    if not mine:
        return None
    if carrier is None:
        carrier = Carrier(posed, rest, head, reference)
    R, t, centre = carrier.R, carrier.t, carrier.centre
    points, normals, uvs, colors, triangles, material = [], [], [], [], [], []
    source, source_normals = [], []  # the reference's frame
    materials: list[GroomMaterial] = []
    textures: dict = {}
    names: dict = {}
    count = 0
    has_colors = any(p.colors is not None for p in mine)
    for p in mine:
        v = np.asarray(p.vertices, dtype=np.float64)
        tri = np.asarray(p.triangles, dtype=np.int64)
        n = (
            np.asarray(p.normals, dtype=np.float64)
            if p.normals is not None
            else vertex_normals(v, tri)
        )
        name = p.material or p.name
        if name not in names:
            names[name] = len(materials)
            slots = {}
            if p.base_color is not None:
                slots["base_color"] = np.asarray(p.base_color, np.uint8)
            if p.normal_map is not None:
                slots["normal"] = np.asarray(p.normal_map, np.uint8)
            roughness = p.roughness if isinstance(p.roughness, float) else None
            if roughness is None and p.roughness is not None:
                roughness = float(np.median(p.roughness))
            materials.append(
                GroomMaterial(
                    name=name,
                    base_color=tuple(p.base_color_factor or (1.0, 1.0, 1.0, 1.0)),
                    roughness=0.6 if roughness is None else roughness,
                    alpha_mode=p.alpha_mode or "OPAQUE",
                    alpha_cutoff=float(p.alpha_cutoff),
                    double_sided=bool(p.double_sided),
                    normal_scale=float(p.normal_scale),
                    textures=tuple(sorted(slots)),
                )
            )
            textures[name] = slots
        points.append(carrier.points(v))
        normals.append(n @ R.T)
        source.append(v)
        source_normals.append(n)
        uvs.append(
            np.asarray(p.uvs, np.float64)
            if p.uvs is not None
            else np.zeros((len(v), 2))
        )
        if has_colors:
            colors.append(
                np.asarray(p.colors, np.float64)
                if p.colors is not None
                else np.ones((len(v), 4))
            )
        triangles.append(tri + count)
        material.append(np.full(len(tri), names[name]))
        count += len(v)
    points = np.concatenate(points)
    normals = np.concatenate(normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    triangles = np.concatenate(triangles)
    material = np.concatenate(material)
    # each material's triangles together, in the parts' order, the blended ones inside out
    order_t, order_m = [], []
    for k, m in enumerate(materials):
        tri = triangles[material == k]
        if m.alpha_mode == "BLEND":
            tri = _sorted_inside_out(points, tri, centre)
        order_t.append(tri)
        order_m.append(np.full(len(tri), k))
    data = GroomData(
        kind=kind,
        points=points,
        normals=normals,
        uvs=np.concatenate(uvs),
        triangles=np.concatenate(order_t),
        material=np.concatenate(order_m),
        materials=materials,
        colors=np.concatenate(colors) if has_colors else None,
    )
    if kind == "hair" and bone_heads is not None and labels is not None:
        bones, slots, weights, chains, share = anny_skin(mine, R, t, bone_heads, labels)
        data.bones, data.bone_slots, data.bone_weights = tuple(bones), slots, weights
        data.chains, data.chain_share = chains, share
        if carrier.motion is not None:  # the fit's pose undone with the hair's own skin
            index = np.array([list(labels).index(b) for b in bones])
            rest_points, rest_normals = carrier.unpose(
                np.concatenate(source),
                np.concatenate(source_normals),
                index[slots],
                weights,
            )
            data.points, data.normals = rest_points, rest_normals
            data.triangles = _resorted(data, carrier.centre)
    return data, textures


def _resorted(data: GroomData, centre) -> np.ndarray:
    """The triangles of a groom with each blended material's sorted again from the inside out
    (after its points moved)."""
    out = data.triangles.copy()
    for k, m in enumerate(data.materials):
        if m.alpha_mode != "BLEND":
            continue
        pick = data.material == k
        out[pick] = _sorted_inside_out(data.points, data.triangles[pick], centre)
    return out

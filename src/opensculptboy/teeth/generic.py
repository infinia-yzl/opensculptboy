# OpenSculptBoy
# Apache License, Version 2.0
"""
Generic teeth: a procedural set for a character without a reference's teeth.

:func:`generic_teeth` builds two arches of seven teeth a side (incisors, canine, premolars and
molars), each crown a rounded box (a superellipse in section) that thins toward the edge of the
incisors and the canine, and a ridge of gums along the roots of each arch. The arches follow a
parabola behind the lips: their front lies :data:`LIP_CLEARANCE_MM` behind the inner margin of
anny's lips (the border of the base mesh's ``mouth_cavity``), at the height of its middle where
the lips meet, and the whole set scales with the width of the mouth. The upper incisors and
canines overlap the lower ones by :data:`OVERBITE_MM`, in front of them; the back teeth meet.
Everything here is drawn from these numbers, without third-party assets.
"""

from __future__ import annotations

import numpy as np

from opensculptboy.teeth.mesh import TeethData

# (width, depth, crown height) in millimetres and the kind of each tooth from the middle:
# central and lateral incisor, canine, two premolars, two molars
UPPER = [
    (8.5, 7.0, 10.5, "incisor"),
    (6.5, 6.0, 9.0, "incisor"),
    (7.5, 8.0, 10.0, "canine"),
    (7.0, 9.0, 8.0, "premolar"),
    (6.5, 9.0, 7.5, "premolar"),
    (10.0, 11.0, 7.0, "molar"),
    (9.0, 11.0, 6.5, "molar"),
]
LOWER = [
    (5.0, 6.0, 9.0, "incisor"),
    (5.5, 6.5, 9.0, "incisor"),
    (7.0, 7.5, 10.0, "canine"),
    (7.0, 7.5, 8.0, "premolar"),
    (7.0, 8.0, 7.5, "premolar"),
    (11.0, 10.5, 7.0, "molar"),
    (10.5, 10.0, 6.5, "molar"),
]
GAP_MM = 0.25  # between neighbouring crowns
ARCH_CURVE = (
    55.0  # the parabola y = a x^2 of the arch, in 1/m (in Anny's frame, -Y forward)
)
MOUTH_WIDTH_MM = 50.0  # the width of the inner lip margin that the sizes above suit
LIP_CLEARANCE_MM = 2.5  # the upper incisors' front behind the lips' inner margin
OVERJET_MM = 3.0  # the lower arch behind the upper
OVERBITE_MM = 2.0  # the vertical overlap of the front teeth
EMBED = 0.25  # the share of a crown's height that the gums cover
SEGMENTS = (12, 6)  # around and along each crown


def mouth_frame(model, rest: np.ndarray) -> dict:
    """
    The mouth of anny's own mesh at rest (``rest``, (V, 3) in Anny's frame): the inner
    margin of the lips (the vertices of the base mesh's ``mouth_cavity`` on a face that leaves
    it), its width, and the front of its middle where the lips meet (``front``: (x, y, z)).
    """
    from anny.faces.authoring.base_mesh import base_mesh
    from opensculptboy.topology.lookup import base_mesh_rows

    mesh = base_mesh()
    cavity = np.zeros(len(mesh.V), bool)
    cavity[np.unique(mesh.quads[mesh.face_labels == "mouth_cavity"])] = True
    cavity = cavity[base_mesh_rows(model)]
    faces = model.faces.cpu().numpy()
    mixed = cavity[faces].any(1) & ~cavity[faces].all(1)
    ring = np.zeros(len(cavity), bool)
    ring[np.unique(faces[mixed])] = True
    ring &= cavity
    if ring.sum() < 6:
        raise ValueError("The generic teeth need anny's mouth (its mouth cavity).")
    points = np.asarray(rest, dtype=np.float64)[ring]
    centre_x = float(np.median(points[:, 0]))
    width = float(np.ptp(points[:, 0]))
    middle = points[np.abs(points[:, 0] - centre_x) < 0.15 * width]
    front_y = float(middle[:, 1].min())
    return {
        "ring": np.flatnonzero(ring),
        "width": width,
        "front": np.array([centre_x, front_y, float(middle[:, 2].mean())]),
    }


def _arch_points(lengths: np.ndarray, curve: float) -> tuple[np.ndarray, np.ndarray]:
    """Points (n, 2) (x, y) at arc lengths along the parabola y = curve x^2 for x >= 0, and
    their unit tangents."""
    x = np.linspace(0.0, 0.08, 4001)
    y = curve * x**2
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    px = np.interp(lengths, s, x)
    tangent = np.c_[np.ones_like(px), 2.0 * curve * px]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return np.c_[px, curve * px**2], tangent


def _crown(width, depth, height, kind):
    """A crown in its own frame: x along the arch, y toward the lips, z from the biting edge
    (0) to the root (``height``); points (M, 3), triangles, and UVs (M, 2)."""
    nu, nv = SEGMENTS
    u = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
    v = np.linspace(0.0, 1.0, nv)
    power = 2.6
    cu, su = np.cos(u), np.sin(u)
    ex = np.sign(cu) * np.abs(cu) ** (2.0 / power)
    ey = np.sign(su) * np.abs(su) ** (2.0 / power)
    points, uvs = [], []
    for t in v:
        if kind in ("incisor", "canine"):
            thin = 0.3 + 0.7 * np.sqrt(t)  # thin at the biting edge
            narrow = 0.85 + 0.15 * np.sqrt(t) if kind == "incisor" else 0.6 + 0.4 * t
        else:
            thin = 0.85 + 0.15 * t
            narrow = 0.85 + 0.15 * t
        neck = 1.0 - 0.15 * max(0.0, t - 0.7) / 0.3  # narrower at the root
        a = 0.5 * width * narrow * neck
        b = 0.5 * depth * thin * neck
        for k in range(nu):
            points.append((a * ex[k], b * ey[k], t * height))
            uvs.append((k / nu, t))
    points.append((0.0, 0.0, -0.05 * height if kind == "molar" else 0.0))  # edge cap
    uvs.append((0.5, 0.0))
    points.append((0.0, 0.0, height))  # root cap
    uvs.append((0.5, 1.0))
    edge, root = len(points) - 2, len(points) - 1
    triangles = []
    for j in range(nv - 1):
        for k in range(nu):
            a0, a1 = j * nu + k, j * nu + (k + 1) % nu
            b0, b1 = a0 + nu, a1 + nu
            triangles += [(a0, b0, a1), (a1, b0, b1)]
    for k in range(nu):
        triangles.append((edge, k, (k + 1) % nu))
        top = (nv - 1) * nu
        triangles.append((root, top + (k + 1) % nu, top + k))
    return np.asarray(points), np.asarray(triangles), np.asarray(uvs)


def _outward(points, triangles, centre):
    """The triangles turned so that their normals point away from ``centre``."""
    p = points[triangles]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    inward = ((p.mean(1) - centre) * normal).sum(1) < 0
    triangles = triangles.copy()
    triangles[inward] = triangles[inward][:, ::-1]
    return triangles


def _arch(table, front, scale, upper: bool):
    """The crowns of one arch: points, triangles, UVs and the centres of the crowns' roots
    along the arch (for the gums)."""
    widths = np.array([w for w, *_ in table]) * scale / 1000.0
    gap = GAP_MM * scale / 1000.0
    centres = np.cumsum(widths + gap) - 0.5 * widths - 0.5 * gap
    xy, tangent = _arch_points(centres, ARCH_CURVE)
    sign = (
        1.0 if upper else -1.0
    )  # the crowns rise from the biting edge for the upper arch
    back = 0.0 if upper else OVERJET_MM * scale / 1000.0
    points, triangles, uvs, roots = [], [], [], []
    count = 0
    for side in (1.0, -1.0):
        for k, (w, d, h, kind) in enumerate(table):
            w, d, h = (v * scale / 1000.0 for v in (w, d, h))
            local, tris, uv = _crown(w, d, h, kind)
            # the crown's centre lies half its depth behind the arch's front line
            t = np.array([tangent[k, 0] * side, tangent[k, 1], 0.0])
            lip = np.array(
                [tangent[k, 1] * side, -tangent[k, 0], 0.0]
            )  # toward the lips
            up = np.array([0.0, 0.0, sign])
            front_teeth = kind in ("incisor", "canine")
            edge = (-0.5 if front_teeth else 0.0) * OVERBITE_MM * scale / 1000.0
            centre = (
                front
                + np.array([side * xy[k, 0], xy[k, 1] + back, 0.0])
                - lip * 0.5 * d
                + up * edge
            )
            world = centre + local[:, :1] * t + local[:, 1:2] * lip + local[:, 2:3] * up
            tris = _outward(world, tris, centre + up * 0.5 * h)
            points.append(world)
            triangles.append(tris + count)
            uvs.append(uv * [0.04, 0.5] + [0.02 + 0.06 * k, 0.5 if upper else 0.0])
            roots.append((centre + up * h * (1.0 - EMBED), lip, d))
            count += len(world)
    return np.concatenate(points), np.concatenate(triangles), np.concatenate(uvs), roots


def _gums(roots, upper: bool, scale: float):
    """A ridge of gums along the roots of one arch: a half-tube over the crowns' necks."""
    order = sorted(range(len(roots)), key=lambda i: roots[i][0][0])
    rings = []
    profile = np.linspace(0.0, np.pi, 9)
    sign = 1.0 if upper else -1.0
    height = 3.0 * scale / 1000.0
    for i in order:
        centre, lip, depth = roots[i]
        half = 0.5 * depth + 1.2 * scale / 1000.0
        ring = (
            centre
            + np.cos(profile)[:, None] * half * lip
            + (np.sin(profile)[:, None] * height - 0.4 * height) * sign * np.r_[0, 0, 1]
        )
        rings.append(ring)
    rings = np.asarray(rings)  # (n, p, 3)
    n, p = rings.shape[:2]
    points = rings.reshape(-1, 3)
    triangles = []
    for i in range(n - 1):
        for j in range(p - 1):
            a, b = i * p + j, i * p + j + 1
            c, d = a + p, b + p
            triangles += [(a, c, b), (b, c, d)]
    uvs = np.c_[np.repeat(np.linspace(0, 1, n), p), np.tile(np.linspace(0, 1, p), n)]
    uvs = uvs * [0.4, 0.2] + [0.5, 0.75 if upper else 0.55]
    centre = rings.mean((0, 1)) - sign * np.r_[0, 0, height]
    return points, _outward(points, np.asarray(triangles), centre), uvs


def generic_teeth(model, rest: np.ndarray, gums: bool = True) -> TeethData:
    """The generic teeth (see the module documentation) behind the lips of anny's own mesh
    at rest (``rest``)."""
    from opensculptboy.export.body import vertex_normals

    mouth = mouth_frame(model, rest)
    scale = float(np.clip(mouth["width"] * 1000.0 / MOUTH_WIDTH_MM, 0.75, 1.3))
    front = mouth["front"] + np.array([0.0, LIP_CLEARANCE_MM * scale / 1000.0, 0.0])
    points, normals, uvs, triangles, arch, part = [], [], [], [], [], []
    count = 0
    for upper, table in ((True, UPPER), (False, LOWER)):
        p, t, uv, roots = _arch(table, front, scale, upper)
        pieces = [(p, t, uv, 0)]
        if gums:
            gp, gt, guv = _gums(roots, upper, scale)
            pieces.append((gp, gt, guv, 1))
        for p, t, uv, kind in pieces:
            points.append(p)
            normals.append(vertex_normals(p, t))
            uvs.append(uv)
            triangles.append(t + count)
            arch.append(np.full(len(p), 0 if upper else 1, np.uint8))
            part.append(np.full(len(t), kind, np.uint8))
            count += len(p)
    return TeethData(
        points=np.concatenate(points),
        normals=np.concatenate(normals),
        uvs=np.concatenate(uvs),
        triangles=np.concatenate(triangles),
        arch=np.concatenate(arch),
        part=np.concatenate(part),
    )

# OpenSculptBoy
# Apache License, Version 2.0
"""
Contact helpers of the pose editor: Squeeze, which closes a hand's fingers together onto a held
prop, and the balance mark, the posed body's centre of mass over the support of its feet.
(Contact pins, which hold a skin point on a surface during a solve, are part of
:func:`opensculptboy.posing.point_solve.solve`.)

**Squeeze** takes one value from 0 to 1. Each finger closes along its own curl: its joints'
bend scaled together, from straight (0) through the hand's present curl (the finger's bends
now, or the library fist's when the finger is straight). The contact curl of a finger is the
least scale at which any of its skin comes within ``clearance`` of the prop (a scan along the
curl, then bisection), and Squeeze sets each finger to ``value`` times its contact curl: 1
closes every finger until it touches with 1 mm to spare, as GraspIt's AutoGrasp closes a hand
(A. T. Miller and P. K. Allen, "GraspIt!", IEEE Robotics and Automation Magazine, 2004), and
lower values open them all in proportion.

**Balance** treats the posed level-0 skin as a solid of uniform density: its centre of mass is
the volume-weighted mean of the tetrahedra between each triangle and a centre point (the
divergence theorem; C. Zhang and T. Chen, "Efficient feature extraction for 2D/3D objects in
mesh representation", 2001). The mark is its projection on the ground, and the support polygon
is the convex hull (Andrew's monotone chain) of the foot skin within ``contact`` of the ground.
"""

from __future__ import annotations

import numpy as np

# the clearance (metres) Squeeze keeps from the prop, the largest curl scale it tries, the scan's
# step and the bisection's steps
CLEARANCE = 0.001
MAX_SCALE = 1.6
SCAN_STEP = 0.04
BISECTION = 12
# a finger's bends (degrees, MCP, PIP, DIP; the thumb's CMC, MCP, IP) when it closes from
# straight: the library fist's (mh_fight)
FIST_BENDS = {
    "finger1": (26.0, 21.0, 54.3),
    "finger2": (71.5, 64.3, 78.2),
    "finger3": (59.1, 77.3, 82.1),
    "finger4": (59.6, 78.8, 76.6),
    "finger5": (46.5, 90.9, 56.0),
}
# the foot skin within this height (metres) of the ground supports the body
CONTACT = 0.01


def signed_distance(points, objects) -> np.ndarray:
    """The signed distance (n,) of points to the nearest of analytic objects in the same
    frame (negative inside): ``{"kind": "sphere", "centre", "radius"}``, ``{"kind":
    "capsule", "start", "end", "radius"}``, ``{"kind": "ellipsoid", "centre", "radii",
    "axis"}`` (as the statue hands' objects, in the rig's output frame), or callables
    ``points -> distances``."""
    P = np.asarray(points, np.float64)
    out = np.full(len(P), np.inf)
    for o in objects:
        if callable(o):
            d = np.asarray(o(P), np.float64)
        elif o["kind"] == "sphere":
            d = np.linalg.norm(P - np.asarray(o["centre"]), axis=1) - float(o["radius"])
        elif o["kind"] == "capsule":
            a, b = np.asarray(o["start"], np.float64), np.asarray(o["end"], np.float64)
            ab = b - a
            t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
            d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1) - float(o["radius"])
        elif o["kind"] == "ellipsoid":
            axis = np.asarray(o.get("axis", [1.0, 0.0, 0.0]), np.float64)
            axis /= np.linalg.norm(axis)
            other = (
                np.array([0.0, 0.0, 1.0])
                if abs(axis[2]) < 0.9
                else np.array([1.0, 0, 0])
            )
            second = np.cross(axis, other)
            second /= np.linalg.norm(second)
            frame = np.stack([axis, second, np.cross(axis, second)])
            radii = np.asarray(o["radii"], np.float64)
            local = (P - np.asarray(o["centre"])) @ frame.T
            d = (np.linalg.norm(local / radii, axis=1) - 1.0) * radii.min()
        else:
            raise ValueError(f"Unknown object {o.get('kind')!r}.")
        out = np.minimum(out, d)
    return out


def objects_from_masks(objects) -> list:
    """Objects in a statue config's masks' frame (y up) in Anny's frame."""
    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    out = []
    for o in objects:
        o = dict(o)
        for k in ("centre", "start", "end", "axis"):
            if k in o:
                o[k] = (np.asarray(o[k], np.float64) @ GLTF_TO_ANNY.T).tolist()
        out.append(o)
    return out


def _finger_rows(rig, finger: str, side: str) -> np.ndarray:
    bones = [rig.index[f"{finger}-{j}{side}"] for j in (1, 2, 3)]
    strongest = rig.indices[np.arange(len(rig.indices)), rig.weights.argmax(1)]
    return np.nonzero(np.isin(strongest, bones))[0]


def squeeze(
    rig,
    rotations,
    hand: str,
    objects,
    value: float = 1.0,
    clearance: float = CLEARANCE,
    fingers=("finger1", "finger2", "finger3", "finger4", "finger5"),
) -> tuple[np.ndarray, dict]:
    """Close a hand's fingers (``"hand.r"``) onto ``objects`` (:func:`signed_distance`) by
    one value from 0 to 1 (see the module). Returns the rotations and, per finger, its curl
    direction's scale at contact (``contact``; None when it closes fully without touching),
    the scale set and its clearance in mm."""
    from opensculptboy.posing.point_solve import _exp, _log, joint_frames

    side = "." + hand.split(".")[1].upper()
    value = float(np.clip(value, 0.0, 1.0))
    D = np.asarray(rotations, np.float64).copy()
    frames = joint_frames(rig)
    report = {}
    for finger in fingers:
        bones = [rig.index[f"{finger}-{j}{side}"] for j in (1, 2, 3)]
        coords = [frames[b]["sign"] * (frames[b]["axes"].T @ _log(D[b])) for b in bones]
        bends = np.array([c[0] for c in coords])
        if np.degrees(np.abs(bends)).sum() < 15.0:
            bends = np.radians(FIST_BENDS[finger])
        rows = _finger_rows(rig, finger, side)

        def posed(scale):
            trial = D.copy()
            for b, c, bend in zip(bones, coords, bends):
                c = c.copy()
                c[0] = scale * bend
                trial[b] = _exp(frames[b]["axes"] @ (frames[b]["sign"] * c))
            return trial

        def gap(scale):
            X = rig.vertices(posed(scale), rows)
            return float(signed_distance(X, objects).min())

        contact = None
        previous = 0.0
        if gap(0.0) <= clearance:
            contact = 0.0
        else:
            for scale in np.arange(SCAN_STEP, MAX_SCALE + 1e-9, SCAN_STEP):
                if gap(scale) <= clearance:
                    lo, hi = previous, scale
                    for _ in range(BISECTION):
                        mid = 0.5 * (lo + hi)
                        lo, hi = (mid, hi) if gap(mid) > clearance else (lo, mid)
                    contact = lo
                    break
                previous = scale
        chosen = value * (MAX_SCALE if contact is None else contact)
        new = posed(chosen)
        for b in bones:
            D[b] = new[b]
        report[finger] = {
            "contact": None if contact is None else round(float(contact), 4),
            "scale": round(float(chosen), 4),
            "clearance_mm": round(1000 * gap(chosen), 2),
            # a finger that lies inside the prop even when straight cannot clear it
            "inside": bool(contact == 0.0 and gap(0.0) < 0.0),
        }
    return D, report


def _hull(points: np.ndarray) -> np.ndarray:
    """The convex hull of 2D points, counter-clockwise (Andrew's monotone chain)."""
    P = np.unique(np.round(np.asarray(points, np.float64), 9), axis=0)
    if len(P) < 3:
        return P
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in P[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def centre_of_mass(vertices, triangles) -> tuple[np.ndarray, float]:
    """The centre of mass (3,) and the volume (m^3) of a closed surface of uniform density."""
    V = np.asarray(vertices, np.float64)
    T = np.asarray(triangles, np.int64)
    o = V.mean(0)
    a, b, c = V[T[:, 0]] - o, V[T[:, 1]] - o, V[T[:, 2]] - o
    vol = (a * np.cross(b, c)).sum(1) / 6.0
    total = float(vol.sum())
    centre = o + ((a + b + c) / 4.0 * vol[:, None]).sum(0) / total
    return centre, abs(total)


def balance(rig, rotations, up=(0.0, 0.0, 1.0), contact: float = CONTACT) -> dict:
    """The balance mark of a posed rig: the centre of mass, its point on the ground (the
    plane through the lowest skin, square to ``up``, in the rig's output frame), the support
    polygon of the feet there (counter-clockwise), whether the mark lies inside it and by how
    much (mm; negative outside)."""
    up = np.asarray(up, np.float64)
    up = up / np.linalg.norm(up)
    V = rig.vertices(np.asarray(rotations, np.float64))
    centre, volume = centre_of_mass(V, rig.faces)
    height = V @ up
    ground = float(height.min())
    strongest = rig.indices[np.arange(len(rig.indices)), rig.weights.argmax(1)]
    feet = np.array(
        [rig.labels[b].startswith(("foot", "toe")) for b in strongest], bool
    )
    support = feet & (height <= ground + contact)
    if not support.any():
        support = height <= ground + contact
    e1 = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(up, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    flat = np.stack([V[support] @ e1, V[support] @ e2], 1)
    hull = _hull(flat)
    mark2 = np.array([centre @ e1, centre @ e2])
    # the signed distance of the mark to the hull's edges (positive inside)
    margin = np.inf
    for i in range(len(hull)):
        p, q = hull[i], hull[(i + 1) % len(hull)]
        edge = q - p
        normal = np.array([edge[1], -edge[0]]) / max(np.linalg.norm(edge), 1e-12)
        # a counter-clockwise hull has its inside to the left of each edge
        margin = min(margin, -float((mark2 - p) @ normal))
    inside = bool(len(hull) >= 3 and margin >= 0)
    on_ground = centre - (centre @ up - ground) * up
    polygon = [(e1 * x + e2 * y + ground * up).tolist() for x, y in hull]
    return {
        "centre": centre.tolist(),
        "mark": on_ground.tolist(),
        "polygon": polygon,
        "inside": inside,
        "margin_mm": round(1000 * float(margin), 2) if len(hull) >= 3 else None,
        "volume_m3": round(volume, 6),
    }

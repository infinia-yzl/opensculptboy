# OpenSculptBoy
# Apache License, Version 2.0
"""
The first placement of the reference points on a scan, from a fitted card.

Each of Anny's points (:mod:`opensculptboy.posing.reference_points`) on the card's posed and
placed body goes to the scan along the skin's normal there: the nearest crossing of the scan's
surface on the line through the point (in front or behind), within ``reach``; where the line
meets no surface, the closest point of the scan takes its place (Project All in sculpting
apps projects along normals in the same way, with a nearest-point fallback).

Each placed point gets a confidence from 0 to 1, the product of three marks:

- **distance**: ``exp(-d / DISTANCE_SCALE)``, so a point that had to travel far is doubtful;
- **normals**: the scan's normal at the spot against the skin's (a point that lands on the far
  side of a thin part, or on a face turned away, scores low);
- **fused carving**: the parts of Anny around the spot (a finger, the palm, the forearm, the
  hip) as the card places them; when a part other than the point's own lies about as near to
  the spot as its own does (a fingertip pressed into the palm, a fist carved into the hip), the
  scan cannot tell them apart there, and the mark falls towards 0.

A point is on when its confidence reaches ``ON``; the page shows the others in amber.
"""

from __future__ import annotations

import numpy as np

from opensculptboy.posing.reference_points import influences_of, ray_hits

# the farthest (metres) a point may move along its normal, and the radius of the scan's
# triangles searched around it
REACH = 0.03
SEARCH = 0.05
# the confidence marks: the distance (metres) at which the distance mark is 1/e, the cosine
# below which the normals' mark is 0 and above which it is 1, and the margin (metres) by which
# a point's own part must be nearer than any other
DISTANCE_SCALE = 0.012
NORMAL_RANGE = (0.0, 0.7)
FUSED_MARGIN = 0.004
ON = 0.5

# the parts of the figure, by the bones' words (without the side's suffix)
PARTS = (
    ("palm", ("wrist", "metacarpal")),
    ("thumb", ("finger1-",)),
    ("index", ("finger2-",)),
    ("middle", ("finger3-",)),
    ("ring", ("finger4-",)),
    ("pinky", ("finger5-",)),
    ("forearm", ("lowerarm",)),
    ("upperarm", ("upperarm", "shoulder01")),
    ("thigh", ("upperleg",)),
    ("shin", ("lowerleg",)),
    ("foot", ("foot", "toe")),
    ("head", ("neck", "head", "eye")),
    ("torso", ("root", "spine", "pelvis", "clavicle")),
)
# a point's own parts: those of its joint, and the part each joint meets
MEETS = {
    "palm": ("forearm",),
    "thumb": ("palm",),
    "index": ("palm",),
    "middle": ("palm",),
    "ring": ("palm",),
    "pinky": ("palm",),
    "forearm": ("upperarm", "palm"),
    "upperarm": ("torso", "forearm"),
    "thigh": ("torso", "shin"),
    "shin": ("thigh", "foot"),
    "foot": ("shin",),
    "head": ("torso",),
}


def part_of(label: str) -> str:
    """The part (:data:`PARTS`) of a bone, with its side's suffix (``"index.R"``)."""
    name, side = label, ""
    for s in (".L", ".R"):
        if label.endswith(s):
            name, side = label[: -len(s)], s
    for part, words in PARTS:
        if name.startswith(words):
            return part + (side if part not in ("head", "torso") else "")
    return "torso"


def closest_on_triangles(points, vertices, triangles):
    """The closest point of a set of triangles to each point: the point (n, 3), the triangle
    (n,) and the distance (n,) (Ericson, "Real-Time Collision Detection", 5.1.5)."""
    P = np.asarray(points, np.float64)
    A, B, C = (vertices[triangles[:, k]] for k in range(3))
    best = np.full(len(P), np.inf)
    out = np.zeros_like(P)
    tri = np.zeros(len(P), np.int64)
    for n, p in enumerate(P):
        ab, ac, ap = B - A, C - A, p - A
        d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
        bp = p - B
        d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
        cp = p - C
        d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        denom = va + vb + vc
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        v = vb / denom
        w = vc / denom
        Q = A + ab * v[:, None] + ac * w[:, None]
        # the regions outside the face: vertices, then edges
        r_a = (d1 <= 0) & (d2 <= 0)
        r_b = (d3 >= 0) & (d4 <= d3)
        r_c = (d6 >= 0) & (d5 <= d6)
        r_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        r_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        r_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        t_ab = np.where(r_ab, d1 / np.where(d1 - d3 == 0, 1e-30, d1 - d3), 0)
        t_ac = np.where(r_ac, d2 / np.where(d2 - d6 == 0, 1e-30, d2 - d6), 0)
        t_bc = np.where(
            r_bc,
            (d4 - d3)
            / np.where((d4 - d3) + (d5 - d6) == 0, 1e-30, (d4 - d3) + (d5 - d6)),
            0,
        )
        Q = np.where(r_bc[:, None], B + (C - B) * t_bc[:, None], Q)
        Q = np.where(r_ac[:, None], A + ac * t_ac[:, None], Q)
        Q = np.where(r_ab[:, None], A + ab * t_ab[:, None], Q)
        Q = np.where(r_c[:, None], C, Q)
        Q = np.where(r_b[:, None], B, Q)
        Q = np.where(r_a[:, None], A, Q)
        d = np.linalg.norm(Q - p, axis=1)
        k = int(np.argmin(d))
        best[n], out[n], tri[n] = d[k], Q[k], k
    return out, tri, best


def _face_normals(vertices, triangles):
    n = np.cross(
        vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
        vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
    )
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)


def _vertex_normals(vertices, triangles):
    fn = np.cross(
        vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
        vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
    )
    out = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(out, triangles[:, k], fn)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-30)


def place_points(
    rig,
    rotations,
    points: list,
    scan_vertices,
    scan_triangles,
    *,
    reach: float = REACH,
    on: float = ON,
) -> list:
    """Place each of ``points`` (the stored set's entries) on a scan (vertices and triangles in
    the rig's output frame), from the rig posed by ``rotations``. Returns one dict per point:
    ``name``, ``position`` (3,), ``confidence``, ``on``, ``method`` (``"normal"`` or
    ``"closest"``), ``distance_mm`` and the three marks."""
    from scipy.spatial import cKDTree

    SV = np.asarray(scan_vertices, np.float64)
    ST = np.asarray(scan_triangles, np.int64)
    centres = SV[ST].mean(1)
    edge = (
        float(np.linalg.norm(SV[ST[:, 0]] - SV[ST[:, 1]], axis=1).max())
        if len(ST)
        else 0.0
    )
    tree = cKDTree(centres)
    face_n = _face_normals(SV, ST)
    rotations = np.asarray(rotations, np.float64)
    K = rig.transforms(rotations)
    X = rig.to_output(rig.points(K, influences_of(rig, points)))
    V = rig.vertices(rotations)
    VN = _vertex_normals(V, rig.faces)
    labels = rig.labels
    strongest = rig.indices[np.arange(len(rig.indices)), rig.weights.argmax(1)]
    vertex_part = np.array([part_of(labels[b]) for b in strongest])
    part_trees = {}
    for part in np.unique(vertex_part):
        part_trees[part] = cKDTree(V[vertex_part == part])
    out = []
    for p, x in zip(points, X):
        bary = np.asarray(p["barycentric"])
        normal = bary @ VN[np.asarray(p["vertices"])]
        normal /= np.linalg.norm(normal)
        near = np.asarray(tree.query_ball_point(x, SEARCH + edge), np.int64)
        position, method, scan_normal = None, None, None
        if len(near):
            T = ST[near]
            hits = []
            for sign in (1.0, -1.0):
                t, rows, _ = ray_hits(x, sign * normal, SV, T)
                if len(t) and t[0] <= reach:
                    hits.append((t[0], sign, rows[0]))
            if hits:
                t, sign, row = min(hits)
                position = x + sign * t * normal
                scan_normal = face_n[near[row]]
                method = "normal"
            else:
                Q, row, _ = closest_on_triangles(x[None], SV, T)
                position = Q[0]
                scan_normal = face_n[near[row[0]]]
                method = "closest"
        if position is None:
            out.append(
                {
                    "name": p["name"],
                    "position": x.tolist(),
                    "confidence": 0.0,
                    "on": False,
                    "method": "none",
                    "distance_mm": None,
                    "marks": {"distance": 0.0, "normals": 0.0, "fused": 0.0},
                }
            )
            continue
        d = float(np.linalg.norm(position - x))
        mark_d = float(np.exp(-d / DISTANCE_SCALE))
        cos = (
            abs(float(scan_normal @ normal))
            if method == "normal"
            else float(scan_normal @ normal)
        )
        lo, hi = NORMAL_RANGE
        mark_n = float(np.clip((cos - lo) / (hi - lo), 0.0, 1.0))
        # the fused carving: the nearest of the point's own parts against the nearest other
        own = {part_of(p["joint"])}
        base = part_of(p["joint"])
        side = base[base.index(".") :] if "." in base else ""
        word = base.split(".")[0]
        own |= {
            m + (side if m not in ("head", "torso") else "")
            for m in MEETS.get(word, ())
        }
        d_own = min(
            (part_trees[o].query(position)[0] for o in own if o in part_trees),
            default=np.inf,
        )
        d_other = min(
            (t.query(position)[0] for name, t in part_trees.items() if name not in own),
            default=np.inf,
        )
        mark_f = float(np.clip((d_other - d_own) / FUSED_MARGIN, 0.0, 1.0))
        confidence = mark_d * mark_n * mark_f
        out.append(
            {
                "name": p["name"],
                "position": position.tolist(),
                "confidence": round(confidence, 4),
                "on": bool(confidence >= on),
                "method": method,
                "distance_mm": round(1000 * d, 2),
                "marks": {
                    "distance": round(mark_d, 4),
                    "normals": round(mark_n, 4),
                    "fused": round(mark_f, 4),
                },
            }
        )
    return out


def placement_json(placed: list, chains=None) -> dict:
    """The placed points as a statue config's ``reference_points``: by chain, each point's
    name, position in the masks' frame (the scan's, y up), confidence and on."""
    from opensculptboy.fit.handoff import GLTF_TO_ANNY
    from opensculptboy.posing.reference_points import load_points

    chain_of = {p["name"]: p["chain"] for p in load_points()}
    out = {}
    for e in placed:
        chain = chain_of.get(e["name"], "body")
        if chains is not None and chain not in chains:
            continue
        out.setdefault(chain, []).append(
            {
                "name": e["name"],
                "position": (np.asarray(e["position"]) @ GLTF_TO_ANNY)
                .round(5)
                .tolist(),
                "confidence": e["confidence"],
                "on": e["on"],
            }
        )
    return {"format": "opensculptboy.reference-placement/1", "chains": out}


def place_on_handoff(folder, character, rotation, translation, chains=None) -> dict:
    """The reference points of a fitted card (its sliders, pose and placement
    ``x_scan = rotation x_anny + translation``) placed on a handoff's scan, as
    :func:`placement_json` gives them, with each point's details (Anny's frame) under
    ``points``."""
    from opensculptboy.fit.handoff import load_handoff
    from opensculptboy.fit.reference import Reference
    from opensculptboy.fit.register import fit_model
    from opensculptboy.posing.pose_rig import PoseRig
    from opensculptboy.posing.reference_points import load_points

    reference = Reference.from_handoff(load_handoff(folder))
    model = fit_model()
    rig, R = PoseRig.from_card(model, character, rotation, translation)
    points = load_points()
    if chains is not None:
        points = [p for p in points if p["chain"] in chains]
    placed = place_points(rig, R, points, reference.vertices, reference.triangles)
    out = placement_json(placed)
    out["points"] = placed
    return out


def main(argv=None):
    import argparse
    import json
    import pathlib

    parser = argparse.ArgumentParser(
        description="Place Anny's reference points on a statue's scan from a fitted card."
    )
    parser.add_argument("handoff", help="the statue's handoff folder")
    parser.add_argument(
        "--stage1", required=True, help="a fit's stage-1 file (stage1/main.json)"
    )
    parser.add_argument("--out", required=True, help="the placement (JSON)")
    parser.add_argument(
        "--chains", nargs="*", help="only these point chains (hand.r, ...)"
    )
    args = parser.parse_args(argv)
    from opensculptboy.statues.fit import stage1_from

    data = json.loads(pathlib.Path(args.stage1).read_text(encoding="utf-8"))
    character, rotation, translation = stage1_from(data)
    out = place_on_handoff(args.handoff, character, rotation, translation, args.chains)
    pathlib.Path(args.out).write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )
    low = [e["name"] for e in out["points"] if not e["on"]]
    print(f"{len(out['points'])} points placed; {len(low)} unsure: {', '.join(low)}")


if __name__ == "__main__":
    main()

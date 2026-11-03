# OpenSculptBoy
# Apache License, Version 2.0
"""
Reference points: fixed spots on Anny's skin that a person places on a scan, so that Anny's pose
can follow them (:mod:`opensculptboy.posing.point_solve`), as markers on a body drive a skeleton
in motion capture.

Each hand has 21 points, in the order and with the names of MediaPipe's hand landmarks (the
wrist; the thumb's CMC, MCP, IP and tip; each finger's MCP, PIP, DIP and tip). The body has the
shoulders, elbows, hips, knees, ankles, heels, the big toes and the top of the head. A point is
a triangle of Anny's level-0 mesh (topology ``anny``) and barycentric weights, so it lies on the
skin in every pose and on every body, and it records the joint it stands for.

A joint's point is the skin over the joint where it shows on a carved figure: on the back of
the hand and of the thumb for the hand's joints, the point of the elbow, the kneecap, the outer
ankle, the outer hip and the top of the shoulder. :func:`build_point_set` finds each by casting
a ray from the joint's head in the reference pose along that direction and taking the first
skin it meets (Moller and Trumbore's ray-triangle test); a fingertip's point is where the ray
along the end bone leaves the finger. The set is stored in ``data/posing/reference_points.json``
(``opensculptboy.reference-points/1``), which the web page reads too; ``python -m
opensculptboy.posing.reference_points --write`` builds it again from the default body.
"""

from __future__ import annotations

import functools
import json
import pathlib

import numpy as np

FORMAT = "opensculptboy.reference-points/1"
POINTS_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "data"
    / "posing"
    / "reference_points.json"
)
# MediaPipe's hand landmarks, in its order, and the finger each belongs to
HAND_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
FINGER_WORDS = {"thumb": 1, "index": 2, "middle": 3, "ring": 4, "pinky": 5}
BODY_NAMES = (
    "shoulder",
    "elbow",
    "hip",
    "knee",
    "ankle",
    "heel",
    "toe",
)


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def ray_hits(origin, direction, vertices, triangles):
    """The distances (n,) along a ray to the triangles it crosses in front of its origin, the
    triangles' rows and the barycentric weights (n, 3) of the hits (Moller and Trumbore, "Fast,
    minimum storage ray-triangle intersection", 1997)."""
    A, B, C = (vertices[triangles[:, k]] for k in range(3))
    e1, e2 = B - A, C - A
    p = np.cross(direction, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origin - A
    u = (s * p).sum(1) * inv
    q = np.cross(s, e1)
    v = (q @ direction) * inv
    t = (e2 * q).sum(1) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
    rows = np.nonzero(hit)[0]
    order = np.argsort(t[rows])
    rows = rows[order]
    bary = np.stack([1.0 - u[rows] - v[rows], u[rows], v[rows]], 1)
    return t[rows], rows, bary


def _first_hit(origin, direction, vertices, triangles, allowed=None):
    t, rows, bary = ray_hits(origin, direction, vertices, triangles)
    if allowed is not None:
        keep = allowed[rows]
        t, rows, bary = t[keep], rows[keep], bary[keep]
    if not len(rows):
        raise ValueError("A reference point's ray meets no skin.")
    return int(rows[0]), bary[0], float(t[0])


def _specs(rig, side: str) -> list:
    """The points of one side as (name, chain, joint bone, end, ray origin, direction, finger,
    bones whose skin the ray may meet)."""
    from opensculptboy.posing.point_solve import _hand_axes

    H = rig.heads
    ix = rig.index
    s = side
    lateral = np.array([1.0 if s == ".L" else -1.0, 0.0, 0.0])
    palm, little = _hand_axes(rig, s)
    dorsal = -palm
    thumb_dorsal = -_unit(0.5 * palm + 0.85 * little)
    out = []
    hand = "hand." + s[1].lower()

    def perp(d, axis):
        axis = _unit(axis)
        return _unit(d - (d @ axis) * axis)

    wrist = ix["wrist" + s]
    out.append(
        (
            "wrist",
            hand,
            "wrist" + s,
            False,
            H[wrist],
            perp(dorsal, H[ix["finger3-1" + s]] - H[wrist]),
            None,
        )
    )
    for word, k in FINGER_WORDS.items():
        finger = f"finger{k}{s}"
        bones = [ix[f"finger{k}-{j}{s}"] for j in (1, 2, 3)]
        names = ("cmc", "mcp", "ip") if k == 1 else ("mcp", "pip", "dip")
        up = thumb_dorsal if k == 1 else dorsal
        for j, (b, n) in enumerate(zip(bones, names)):
            nxt = bones[j + 1] if j < 2 else None
            axis = (H[nxt] - H[b]) if nxt is not None else (H[b] - H[bones[j - 1]])
            out.append(
                (
                    f"{word}_{n}",
                    hand,
                    rig.labels[b],
                    False,
                    H[b],
                    perp(up, axis),
                    finger,
                )
            )
        axis = _unit(H[bones[2]] - H[bones[1]])
        out.append(
            (f"{word}_tip", hand, rig.labels[bones[2]], True, H[bones[2]], axis, finger)
        )
    arm = "arm." + s[1].lower()
    leg = "leg." + s[1].lower()
    up = np.array([0.0, 0.0, 1.0])
    front = np.array([0.0, -1.0, 0.0])
    shoulder = ix["upperarm01" + s]
    out.append(
        (
            "shoulder",
            arm,
            rig.labels[shoulder],
            False,
            H[shoulder],
            _unit(up + 0.35 * lateral),
            None,
        )
    )
    elbow = ix["lowerarm01" + s]
    upper = H[elbow] - H[shoulder]
    out.append(
        ("elbow", arm, rig.labels[elbow], False, H[elbow], perp(-front, upper), None)
    )
    hip = ix["upperleg01" + s]
    knee = ix["lowerleg01" + s]
    out.append(
        (
            "hip",
            leg,
            rig.labels[hip],
            False,
            H[hip] + 0.0 * up,
            perp(lateral, H[knee] - H[hip]),
            None,
        )
    )
    out.append(
        (
            "knee",
            leg,
            rig.labels[knee],
            False,
            H[knee],
            perp(front, H[knee] - H[hip]),
            None,
        )
    )
    foot = ix["foot" + s]
    shin = H[foot] - H[knee]
    out.append(
        ("ankle", leg, rig.labels[foot], False, H[foot], perp(lateral, shin), None)
    )
    out.append(
        ("heel", leg, rig.labels[foot], False, H[foot], _unit(-front - 1.2 * up), None)
    )
    toe = ix["toe1-2" + s]
    out.append(
        (
            "toe",
            leg,
            rig.labels[toe],
            True,
            H[toe],
            _unit(H[toe] - H[ix["toe1-1" + s]]),
            None,
        )
    )
    return out


def build_point_set(rig) -> list:
    """The reference points of a rig's reference pose (:class:`~opensculptboy.posing.pose_rig.PoseRig`
    at the identity pose), as dicts: ``name`` (``"index_pip.R"``), ``chain`` (``"hand.r"``,
    ``"arm.r"``, ``"leg.r"``, ``"head"``), ``joint`` (the bone whose head, or end, it stands
    over), ``end``, ``finger``, ``mediapipe`` (the hand landmark's index), ``triangle``,
    ``vertices`` (the triangle's three rows) and ``barycentric``."""
    identity = np.eye(3)[None].repeat(len(rig.labels), 0)
    K = rig.transforms(identity)
    # the reference pose's mesh in the space of the turns (no placement)
    M = K @ rig.bind
    v, w, i = rig.rest_vertices, rig.weights, rig.indices
    X = (
        w[..., None]
        * (np.einsum("vkij,vj->vki", M[i][..., :3, :3], v) + M[i][..., :3, 3])
    ).sum(1)
    T = rig.faces
    out = []
    for side in (".R", ".L"):
        for name, chain, joint, end, origin, direction, finger in _specs(rig, side):
            tri, bary, dist = _first_hit(
                np.asarray(origin), np.asarray(direction), X, T
            )
            entry = {
                "name": name + side,
                "chain": chain,
                "joint": joint,
                "end": bool(end),
                "finger": finger,
                "triangle": tri,
                "vertices": [int(x) for x in T[tri]],
                "barycentric": [round(float(x), 6) for x in bary],
                "depth_mm": round(1000 * dist, 2),
            }
            if chain.startswith("hand"):
                entry["mediapipe"] = HAND_NAMES.index(name)
            out.append(entry)
    head = rig.index["head"]
    tri, bary, dist = _first_hit(rig.heads[head], np.array([0.0, 0.0, 1.0]), X, T)
    out.append(
        {
            "name": "head_top",
            "chain": "neck",
            "joint": "head",
            "end": True,
            "finger": None,
            "triangle": tri,
            "vertices": [int(x) for x in T[tri]],
            "barycentric": [round(float(x), 6) for x in bary],
            "depth_mm": round(1000 * dist, 2),
        }
    )
    return out


@functools.lru_cache(maxsize=1)
def _points_text() -> str:
    return POINTS_PATH.read_text(encoding="utf-8")


def load_points() -> list:
    """The stored point set (``data/posing/reference_points.json``)."""
    data = json.loads(_points_text())
    if data.get("format") != FORMAT:
        raise ValueError(f"Not a reference point set: {data.get('format')!r}.")
    return data["points"]


def points_of(chains, points=None) -> list:
    """The points that a solve of ``chains`` uses: an arm's are its hand's too, a leg's its
    foot's."""
    points = load_points() if points is None else points
    wanted = set()
    for chain in chains:
        part, _, side = chain.lower().partition(".")
        if part == "arm":
            wanted |= {f"arm.{side}", f"hand.{side}"}
        elif part in ("leg", "foot"):
            wanted.add(f"leg.{side}")
        else:
            wanted.add(chain.lower())
    return [p for p in points if p["chain"] in wanted]


def influences_of(rig, points: list) -> list:
    """Each point's skinning influences on a rig (:meth:`PoseRig.influences`), by its
    triangle's vertices and barycentric weights."""
    fake = np.asarray([p["vertices"] for p in points], np.int64)
    bary = np.asarray([p["barycentric"] for p in points], np.float64)
    saved = rig.faces
    try:
        rig.faces = fake
        return rig.influences(np.arange(len(points)), bary)
    finally:
        rig.faces = saved


def positions(rig, rotations, points: list) -> np.ndarray:
    """The points (P, 3) on a posed rig, in its output frame."""
    K = rig.transforms(np.asarray(rotations, np.float64))
    return rig.to_output(rig.points(K, influences_of(rig, points)))


def write(rig, path=POINTS_PATH) -> dict:
    data = {
        "format": FORMAT,
        "topology": "anny",
        "about": "Fixed spots on Anny's level-0 skin (topology anny) that stand for joints: "
        "21 per hand in MediaPipe's order, and the shoulders, elbows, hips, knees, ankles, "
        "heels, big toes and the top of the head. Each is a triangle (its three vertex rows) "
        "and barycentric weights; depth_mm is how far the skin lies from the joint's head on "
        "the default body.",
        "points": build_point_set(rig),
    }
    # one point per line
    head = {k: v for k, v in data.items() if k != "points"}
    lines = ",\n".join("  " + json.dumps(p) for p in data["points"])
    text = json.dumps(head, indent=1)[:-2] + ',\n "points": [\n' + lines + "\n ]\n}\n"
    json.loads(text)
    pathlib.Path(path).write_text(text, encoding="utf-8")
    return data


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="write the data file")
    parser.add_argument("--out", default=str(POINTS_PATH))
    args = parser.parse_args(argv)
    from opensculptboy.character import Character
    from opensculptboy.fit.register import fit_model
    from opensculptboy.posing.pose_rig import PoseRig

    model = fit_model()
    rig, _ = PoseRig.from_card(model, Character())
    points = build_point_set(rig)
    if args.write:
        write(rig, args.out)
    for p in points:
        print(f"{p['name']:>16} {p['joint']:>14} depth {p['depth_mm']:6.2f} mm")


if __name__ == "__main__":
    main()

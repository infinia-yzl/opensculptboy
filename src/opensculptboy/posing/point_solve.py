# OpenSculptBoy
# Apache License, Version 2.0
"""
Anny's pose solved to reference points: the bones of the chosen chains turn so that the skin's
points (:mod:`opensculptboy.posing.reference_points`) meet the places given for them.

The solve is damped least squares (Levenberg-Marquardt, with Nielsen's update of the damping;
K. Madsen, H. B. Nielsen and O. Tingleff, "Methods for non-linear least squares problems",
2004) over the ``local-ref`` turns of the chains' bones, as motion-capture solvers fit a
skeleton to markers. Each step linearises the exact forward model of :mod:`pose_rig`:

- a bone's turns are taken about the axes of its joint frame (bend, side and twist,
  :func:`joint_frames`, from the table ``data/posing/joint_ranges.json``), only those the
  table frees (a finger's middle joint bends and nothing else);
- a twist group (the forearm's, the upper arm's, the thigh's and the shin's) turns its bones
  together about the limb's axis, each by its share, so that a twist spreads along the limb as
  a forearm's skin turns along its length and no single bone pinches it;
- each point's residual is its distance to its place (metres, weighted); a point that is off
  has none, and a finger whose points are all off is pulled to a natural curl: its joints bend
  as its neighbours' do (the hand's synergy), or as the ``natural`` pose has them;
- each step stops at the ranges' ends (a projected step; a coordinate that starts outside its
  range may stay there or move back, never further out; ``limits=False`` switches the ranges
  off). Stiff springs at the ends trapped the solve far from reachable places;
- a small pull keeps every bone near its start, so that the pose stays the nearest one;
- contact pins hold chosen skin points on a point or a plane (a plane that follows a surface
  is taken again on each step: a palm that slides along a hip).

Every step's arithmetic is repeated in ``viewer/src/pose_edit/solve.ts``; a problem built by
:func:`build_problem` is plain JSON that both read, and ``test_reference_points`` checks that
they give the same pose and points.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import pathlib

import numpy as np

RANGES_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "data"
    / "posing"
    / "joint_ranges.json"
)
# the weight (metres per radian) of a turn away from its start
PRIOR_WEIGHT = 0.003
# the pull of a finger whose points are all off towards its natural curl (metres per radian)
NATURAL_WEIGHT = 0.03
# the weight of a contact pin against a point's (1)
PIN_WEIGHT = 10.0
# a turn's rotation vector on the left side, as the right side's mirror image has it
MIRROR = np.array([1.0, -1.0, -1.0])
DIRECTIONS = {
    "front": (0.0, -1.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
}


@functools.lru_cache(maxsize=1)
def _ranges_cached() -> str:
    return RANGES_PATH.read_text(encoding="utf-8")


def load_ranges() -> dict:
    """The joint range table (``data/posing/joint_ranges.json``)."""
    return json.loads(_ranges_cached())


def _split(label: str) -> tuple[str, str]:
    for s in (".L", ".R"):
        if label.endswith(s):
            return label[: -len(s)], s
    return label, ""


def _unit(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _hand_axes(rig, side: str):
    """The palm's normal and the direction towards the little finger of a hand in the
    reference pose (the space of the turns)."""
    H = rig.heads
    ix = rig.index
    wrist = H[ix["wrist" + side]]
    along = H[ix["finger3-1" + side]] - wrist
    across = H[ix["finger2-1" + side]] - H[ix["finger5-1" + side]]
    across = across - (across @ _unit(along)) * _unit(along)
    # the fingers curl towards -(along x across) on the right hand and the mirror on the left
    sign = 1.0 if side == ".R" else -1.0
    palm = -sign * _unit(np.cross(along, across))
    return palm, -_unit(across)


def joint_frames(rig, table: dict | None = None) -> dict:
    """Each bone's joint frame in the table: ``axes`` (3, 3), whose columns are the bend, side
    and twist axes in the space of the turns, ``sign`` (3,) (the left side's side and twist
    mirrored), the free ``dofs`` (3,) bool, and the ranges ``lo`` and ``hi`` (3,) in radians."""
    table = table or load_ranges()
    H = rig.heads
    ix = rig.index
    out = {}
    for b, label in enumerate(rig.labels):
        name, side = _split(label)
        spec = table["bones"].get(name)
        if spec is None:
            continue
        nxt = spec["next"]
        if nxt == "+z":
            t = np.array([0.0, 0.0, 1.0])
        elif nxt == "tip":
            p = rig.parents[b]
            t = _unit(H[b] - H[p])
        else:
            if nxt + side not in ix:
                continue
            t = _unit(H[ix[nxt + side]] - H[b])
        toward = spec["toward"]
        if toward in ("palm", "thumb"):
            palm, little = _hand_axes(rig, side)
            d = palm if toward == "palm" else _unit(0.5 * palm + 0.85 * little)
        else:
            d = np.asarray(DIRECTIONS[toward])
        if "hinge" in spec:
            # the library fist's axis (the right hand's; the left's is its mirror image)
            a = _unit(
                np.asarray(spec["hinge"], np.float64)
                * (MIRROR if side == ".L" else 1.0)
            )
            t = _unit(t - (t @ a) * a)
        else:
            d = d - (d @ t) * t
            a = _unit(np.cross(t, d))
        s = np.cross(t, a)
        sign = np.array([1.0, -1.0, -1.0]) if side == ".L" else np.ones(3)
        out[b] = {
            "axes": np.stack([a, s, t], 1),
            "sign": sign,
            "dofs": np.array([k in spec["dofs"] for k in "bst"]),
            "lo": np.radians([spec["bend"][0], spec["side"][0], spec["twist"][0]]),
            "hi": np.radians([spec["bend"][1], spec["side"][1], spec["twist"][1]]),
        }
    return out


def chain_bones(rig, chains, table: dict | None = None) -> list:
    """The bones (indices) of chains such as ``"arm.r"``, ``"hand.l"`` or ``"leg.r"`` (the
    table's ``chains``), parents first."""
    table = table or load_ranges()
    picked = set()
    for chain in chains:
        part, _, side = chain.lower().partition(".")
        if part not in table["chains"]:
            raise ValueError(
                f"Unknown chain {chain!r}; use one of {sorted(table['chains'])}."
            )
        suffix = "." + side.upper() if side else ""
        for b, label in enumerate(rig.labels):
            name, s = _split(label)
            if s != suffix:
                continue
            if any(name.startswith(w) for w in table["chains"][part]):
                picked.add(b)
    return [int(b) for b in rig.order if b in picked]


def _log(R: np.ndarray) -> np.ndarray:
    """The rotation vector of a rotation matrix."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-6:
        return 0.5 * w
    if angle > np.pi - 1e-6:
        # near a half turn: the axis from the symmetric part
        M = (R + R.T) / 2.0 - c * np.eye(3)
        k = int(np.argmax(np.diag(M)))
        axis = M[:, k] / np.sqrt(max(M[k, k], 1e-15))
        if axis @ w < 0:
            axis = -axis
        return angle * axis
    return angle / (2.0 * np.sin(angle)) * w


def _exp(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    if angle < 1e-9:
        return np.eye(3) + K
    return (
        np.eye(3)
        + np.sin(angle) / angle * K
        + (1.0 - np.cos(angle)) / angle**2 * (K @ K)
    )


def _coordinates(frame: dict, R: np.ndarray) -> np.ndarray:
    """A rotation's joint coordinates (bend, side, twist) in a joint frame."""
    return frame["sign"] * (frame["axes"].T @ _log(R))


def coordinates(rig, rotations, table: dict | None = None) -> dict:
    """Each bone's joint coordinates in degrees (bend, side, twist) for a pose (B, 3, 3), by
    bone label: the values the page's sliders show."""
    frames = joint_frames(rig, table)
    return {
        rig.labels[b]: np.degrees(_coordinates(f, np.asarray(rotations[b])))
        .round(3)
        .tolist()
        for b, f in frames.items()
    }


@dataclasses.dataclass
class SolveResult:
    """A solve's pose (B, 3, 3), the points in the output frame (P, 3), each point's distance
    to its place (mm, None where it is off), the steps taken and the final cost."""

    rotations: np.ndarray
    points: np.ndarray
    residual_mm: list
    iterations: int
    cost: float
    pins_mm: list = dataclasses.field(default_factory=list)


def build_problem(
    rig,
    rotations,
    points: list,
    chains,
    *,
    table: dict | None = None,
    limits: bool = True,
    natural=None,
    pins: list = (),
    prior: float = PRIOR_WEIGHT,
    natural_weight: float = NATURAL_WEIGHT,
    iterations: int = 40,
    bones=None,
) -> dict:
    """The solve as plain JSON (the page's solver reads the same).

    ``points``: dicts with ``influence`` (from :meth:`PoseRig.influences`), ``target`` (3,)
    in the output frame, optional ``weight`` (default 1), ``on`` (default True) and
    ``finger`` (``"finger2.R"``: the finger it lies on, for the natural curl). ``pins``:
    dicts with ``influence``, ``kind`` (``"point"`` with ``target``; ``"plane"`` with
    ``point`` and ``normal``, output frame) and an optional ``weight``. ``chains`` name the
    bones that turn (:func:`chain_bones`); ``bones`` adds bone indices. ``natural`` (B, 3, 3):
    the pose an off finger without neighbours on curls to (default: the start)."""
    table = table or load_ranges()
    rotations = np.asarray(rotations, np.float64)
    frames = joint_frames(rig, table)
    active = chain_bones(rig, chains, table) if chains else []
    if bones is not None:
        active = [int(b) for b in rig.order if b in set(active) | set(bones)]
    active_set = set(active)
    # the twist groups whose bones all turn
    groups = []
    grouped = set()
    for name, g in table["twist"].items():
        side = None
        for chain_side in (".L", ".R"):
            members = [rig.index.get(m + chain_side) for m in g["shares"]]
            if all(m is not None and m in active_set for m in members):
                side = chain_side
                groups.append(
                    {
                        "name": name + side,
                        "axis": [
                            rig.index[g["axis"][0] + side],
                            rig.index[g["axis"][1] + side],
                        ],
                        "members": members,
                        "shares": [float(v) for v in g["shares"].values()],
                        "read": rig.index[g["read"] + side],
                        "read_share": float(g["shares"][g["read"]]),
                        "lo": float(np.radians(g["range"][0])),
                        "hi": float(np.radians(g["range"][1])),
                    }
                )
                grouped.update(members)
    # the parameters: each active bone's free axes (a bone of an inactive group turns about its
    # own twist axis within its share of the group's range), then the groups
    columns = []
    bones_out = []
    for b in active:
        fr = frames.get(b)
        if fr is None:
            continue
        dofs = fr["dofs"].copy()
        lo, hi = fr["lo"].copy(), fr["hi"].copy()
        name, side = _split(rig.labels[b])
        for g in table["twist"].values():
            if name in g["shares"] and b not in grouped:
                dofs[2] = True
                share = float(g["shares"][name])
                lo[2], hi[2] = (
                    np.radians(g["range"][0]) * share,
                    np.radians(g["range"][1]) * share,
                )
        bones_out.append(
            {
                "bone": int(b),
                "axes": fr["axes"].tolist(),
                "sign": fr["sign"].tolist(),
                "dofs": [bool(x) for x in dofs],
                "lo": lo.tolist(),
                "hi": hi.tolist(),
                "start": _coordinates(fr, rotations[b]).tolist(),
            }
        )
        for k in range(3):
            if dofs[k]:
                columns.append({"bone": int(b), "axis": k})
    for gi, g in enumerate(groups):
        columns.append({"group": gi})
    S = float(np.cbrt(abs(np.linalg.det(rig.frame[:3, :3]))))
    pts = []
    for p in points:
        on = bool(p.get("on", True)) and float(p.get("weight", 1.0)) > 0
        f = p["influence"]
        pts.append(
            {
                "bones": [int(x) for x in f["bones"]],
                "s": [float(x) for x in f["s"]],
                "u": np.asarray(f["u"], np.float64).tolist(),
                "target": rig.from_output(np.asarray(p["target"], np.float64)[None])[
                    0
                ].tolist(),
                "weight": float(p.get("weight", 1.0)) if on else 0.0,
                "finger": p.get("finger"),
            }
        )
    pin_out = []
    for q in pins:
        f = q["influence"]
        entry = {
            "bones": [int(x) for x in f["bones"]],
            "s": [float(x) for x in f["s"]],
            "u": np.asarray(f["u"], np.float64).tolist(),
            "kind": q["kind"],
            "weight": float(q.get("weight", PIN_WEIGHT)),
        }
        if q["kind"] == "point":
            entry["target"] = rig.from_output(np.asarray(q["target"])[None])[0].tolist()
        elif q["kind"] == "plane":
            entry["point"] = rig.from_output(np.asarray(q["point"])[None])[0].tolist()
            n = np.asarray(q["normal"], np.float64) @ rig.frame[:3, :3]
            entry["normal"] = _unit(n).tolist()
        else:
            raise ValueError(f"A pin is a point or a plane; got {q['kind']!r}.")
        pin_out.append(entry)
    # the natural curl of fingers whose points are all off: by their neighbours, else the pose
    fingers = {}
    for p in pts:
        if p["finger"]:
            fingers.setdefault(p["finger"], []).append(p["weight"] > 0)
    off = sorted(f for f, flags in fingers.items() if not any(flags))
    on = sorted(f for f, flags in fingers.items() if any(flags))
    natural = rotations if natural is None else np.asarray(natural, np.float64)
    curls = []
    for f in off:
        name, side = _split(f)
        k = int(name[-1])
        joints = [rig.index.get(f"{name}-{j}{side}") for j in (1, 2, 3)]
        if any(j is None or j not in active_set for j in joints):
            continue
        neighbours = []
        if k > 1:
            for dk in (1, -1, 2, -2, 3, -3):
                n = k + dk
                if 2 <= n <= 5 and f"finger{n}{side}" in on:
                    neighbours.append(n)
                if len(neighbours) == 2:
                    break
        for j, b in enumerate(joints):
            curls.append(
                {
                    "bone": int(b),
                    "neighbours": [
                        rig.index[f"finger{n}-{j + 1}{side}"] for n in neighbours
                    ],
                    "natural": _coordinates(frames[b], natural[b]).tolist(),
                }
            )
    return {
        "format": "opensculptboy.point-solve/1",
        "parents": rig.parents.tolist(),
        "heads": rig.heads.tolist(),
        "order": [int(b) for b in rig.order],
        "scale": S,
        "frame": rig.frame.tolist(),
        "rotations": rotations.reshape(-1, 9).tolist(),
        "bones": bones_out,
        "groups": groups,
        "columns": columns,
        "points": pts,
        "pins": pin_out,
        "curls": curls,
        "limits": bool(limits),
        "weights": {
            "prior": float(prior),
            "natural": float(natural_weight),
        },
        "iterations": int(iterations),
    }


def _jl(v: np.ndarray) -> np.ndarray:
    """SO(3)'s left Jacobian at the rotation vector v: exp(v + d) = exp(Jl(v) d) exp(v)."""
    angle = float(np.linalg.norm(v))
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    if angle < 1e-6:
        return np.eye(3) + 0.5 * K + (K @ K) / 6.0
    return (
        np.eye(3)
        + (1.0 - np.cos(angle)) / angle**2 * K
        + (angle - np.sin(angle)) / angle**3 * (K @ K)
    )


class _Solver:
    """One problem's state and steps; :func:`solve_problem` runs it.

    The unknowns are each turning bone's joint coordinates ``c`` (bend, side, twist; radians),
    with ``D_b = exp(A_b (sign * c_b))``: a bone's locked coordinates keep their start values
    exactly, a twist group adds its share of one turn to each member's twist, the ranges are
    boxes in ``c``, and the pull to the start and the natural curls are linear in it."""

    def __init__(self, problem: dict, surfaces=None):
        P = problem
        self.P = P
        self.parents = np.asarray(P["parents"], np.int64)
        self.heads = np.asarray(P["heads"], np.float64)
        self.order = np.asarray(P["order"], np.int64)
        self.B = len(self.parents)
        self.fixed = np.asarray(P["rotations"], np.float64).reshape(-1, 3, 3)
        self.bones = {e["bone"]: e for e in P["bones"]}
        for e in P["bones"]:
            e["_A"] = np.asarray(e["axes"]) * np.asarray(e["sign"])[None]
        self.columns = P["columns"]
        self.groups = P["groups"]
        self.surfaces = surfaces or {}
        self.w = P["weights"]
        self.n = len(self.columns)
        # each column's (bone, coordinate, share)
        self.col_members = []
        for c in self.columns:
            if "group" in c:
                g = self.groups[c["group"]]
                self.col_members.append(
                    [(m, 2, s) for m, s in zip(g["members"], g["shares"])]
                )
            else:
                self.col_members.append([(c["bone"], c["axis"], 1.0)])
        self.ancestors = []
        for b in range(self.B):
            chain, p = set(), b
            while p >= 0:
                chain.add(p)
                p = self.parents[p]
            self.ancestors.append(chain)

    def start(self) -> dict:
        """The joint coordinates of the start pose."""
        return {
            b: np.asarray(e["start"], np.float64).copy() for b, e in self.bones.items()
        }

    def rotations(self, C: dict) -> np.ndarray:
        D = self.fixed.copy()
        for b, c in C.items():
            D[b] = _exp(self.bones[b]["_A"] @ c)
        return D

    def transforms(self, D):
        K = np.zeros((self.B, 4, 4))
        K[:, 3, 3] = 1.0
        H = self.heads
        for b in self.order:
            p = self.parents[b]
            if p < 0:
                K[b, :3, :3] = np.eye(3)
                continue
            K[b, :3, :3] = K[p, :3, :3] @ D[b]
            K[b, :3, 3] = K[p, :3, :3] @ (H[b] - D[b] @ H[b]) + K[p, :3, 3]
        return K

    def step(self, C: dict, delta: np.ndarray) -> dict:
        """The coordinates after a step, held within the ranges (a coordinate that starts
        outside its range may stay where it is, and move only back towards it)."""
        out = {b: c.copy() for b, c in C.items()}
        for ci, members in enumerate(self.col_members):
            for b, k, share in members:
                out[b][k] += share * delta[ci]
        if not self.P["limits"]:
            return out
        for b, e in self.bones.items():
            for k in range(3):
                if e["dofs"][k]:
                    lo, hi = min(e["lo"][k], C[b][k]), max(e["hi"][k], C[b][k])
                    out[b][k] = min(max(out[b][k], lo), hi)
        for g in self.groups:
            b, share = g["read"], g["read_share"]
            v, was = out[b][2] / share, C[b][2] / share
            lo, hi = min(g["lo"], was), max(g["hi"], was)
            clamped = min(max(v, lo), hi)
            if clamped != v:
                for m, s in zip(g["members"], g["shares"]):
                    out[m][2] += s * (clamped - v)
        return out

    def residuals(self, C: dict, jacobian: bool):
        D = self.rotations(C)
        K = self.transforms(D)
        J = np.einsum("bij,bj->bi", K[:, :3, :3], self.heads) + K[:, :3, 3]
        n = self.n
        S = self.P["scale"]
        axes = None
        if jacobian:
            # each column's members: the world axis of the turn d D_b / d c_bk about j_b
            axes = []
            for members in self.col_members:
                row = []
                for b, k, share in members:
                    A = self.bones[b]["_A"]
                    Rp = K[self.parents[b], :3, :3]
                    row.append((b, share * (Rp @ (_jl(A @ C[b]) @ A[:, k]))))
                axes.append(row)
        rows, jac = [], []

        def point(e):
            x = np.zeros(3)
            parts = []
            for b, s, u in zip(e["bones"], e["s"], e["u"]):
                y = K[b, :3, :3] @ np.asarray(u) + s * K[b, :3, 3]
                parts.append((b, s, y))
                x += y
            return x, parts

        def point_jacobian(parts, weight):
            Jx = np.zeros((3, n))
            for ci, row in enumerate(axes):
                for b, axis in row:
                    Y, Ssum = np.zeros(3), 0.0
                    for c, s, y in parts:
                        if b in self.ancestors[c]:
                            Y += y
                            Ssum += s
                    if Ssum != 0.0 or Y.any():
                        Jx[:, ci] += np.cross(axis, Y - Ssum * J[b])
            return weight * S * Jx

        for e in self.P["points"]:
            if e["weight"] <= 0:
                continue
            x, parts = point(e)
            rows.append(e["weight"] * S * (x - np.asarray(e["target"])))
            if jacobian:
                jac.append(point_jacobian(parts, e["weight"]))
        for e in self.P["pins"]:
            x, parts = point(e)
            if e["kind"] == "point":
                rows.append(e["weight"] * S * (x - np.asarray(e["target"])))
                if jacobian:
                    jac.append(point_jacobian(parts, e["weight"]))
            else:
                p, nrm = np.asarray(e["point"]), np.asarray(e["normal"])
                surface = self.surfaces.get(id(e))
                if surface is not None:
                    p, nrm = surface(x)
                rows.append(np.array([e["weight"] * S * float(nrm @ (x - p))]))
                if jacobian:
                    jac.append(nrm[None] @ point_jacobian(parts, e["weight"]))

        # rows linear in the coordinates: d c_bk / d column
        def coordinate_row(b, k):
            g = np.zeros(n)
            for ci, members in enumerate(self.col_members):
                for bb, kk, share in members:
                    if bb == b and kk == k:
                        g[ci] += share
            return g

        wp = self.w["prior"]
        for b, e in self.bones.items():
            start = np.asarray(e["start"])
            for k in range(3):
                g = coordinate_row(b, k)
                if not g.any():
                    continue
                rows.append(np.array([wp * (C[b][k] - start[k])]))
                if jacobian:
                    jac.append(wp * g[None])
        wn = self.w["natural"]
        for cdef in self.P["curls"]:
            b = cdef["bone"]
            if b not in self.bones:
                continue
            neighbours = [nb for nb in cdef["neighbours"] if nb in self.bones]
            # the neighbours' bend as they stand (a target for this step)
            target = (
                float(np.mean([C[nb][0] for nb in neighbours]))
                if neighbours
                else float(cdef["natural"][0])
            )
            rows.append(np.array([wn * (C[b][0] - target)]))
            if jacobian:
                jac.append(wn * coordinate_row(b, 0)[None])
        r = np.concatenate(rows) if rows else np.zeros(0)
        if not jacobian:
            return r
        Jm = np.concatenate(jac, 0) if jac else np.zeros((0, n))
        return r, Jm


def solve_problem(problem: dict, surfaces=None, iterations: int | None = None):
    """Run a problem of :func:`build_problem`; returns the rotations (B, 3, 3), the steps
    taken and the final cost (half the sum of squared residuals)."""
    solver = _Solver(problem, surfaces)
    C = solver.start()
    steps = int(problem["iterations"] if iterations is None else iterations)
    n = solver.n
    if n == 0:
        r = solver.residuals(C, False)
        return solver.rotations(C), 0, 0.5 * float(r @ r)
    r, Jm = solver.residuals(C, True)
    cost = 0.5 * float(r @ r)
    mu, nu = None, 2.0
    taken = 0
    for it in range(steps):
        H = Jm.T @ Jm
        g = Jm.T @ r
        if mu is None:
            mu = 1e-3 * max(float(np.diag(H).max()), 1e-12)
        delta = -_cholesky_solve(H + mu * np.eye(n), g)
        C_new = solver.step(C, delta)
        r_new = solver.residuals(C_new, False)
        cost_new = 0.5 * float(r_new @ r_new)
        predicted = 0.5 * float(delta @ (mu * delta - g))
        taken = it + 1
        if cost_new < cost:
            rho = (cost - cost_new) / max(predicted, 1e-300)
            mu *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            nu = 2.0
            improved = cost - cost_new
            C, cost = C_new, cost_new
            r, Jm = solver.residuals(C, True)
            if (
                improved <= 1e-12 * max(cost, 1e-30)
                or float(np.abs(delta).max()) < 1e-9
            ):
                break
        else:
            mu *= nu
            nu *= 2.0
            if mu > 1e10:
                break
    return solver.rotations(C), taken, cost


def _cholesky_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A x = b for a symmetric positive definite A, by Cholesky's factors (as the page does)."""
    n = len(b)
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                L[i, i] = np.sqrt(max(s, 1e-300))
            else:
                L[i, j] = s / L[j, j]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L[i + 1 :, i] @ x[i + 1 :]) / L[i, i]
    return x


def solve(
    rig,
    rotations,
    points: list,
    chains,
    *,
    pins: list = (),
    surfaces=None,
    **options,
) -> SolveResult:
    """Solve the chains' bones to the points (see :func:`build_problem` for the arguments):
    returns the pose, the points where they land and how far each is from its place."""
    problem = build_problem(rig, rotations, points, chains, pins=pins, **options)
    keyed = {}
    if surfaces:
        # a pin's surface (a callable x -> (point, normal) in the output frame), by pin index
        for i, fn in surfaces.items():
            entry = problem["pins"][i]

            def turned(x, fn=fn):
                q, nrm = fn(rig.to_output(x[None])[0])
                nrm = _unit(np.asarray(nrm) @ rig.frame[:3, :3])
                return rig.from_output(np.asarray(q)[None])[0], nrm

            keyed[id(entry)] = turned
    D, taken, cost = solve_problem(problem, keyed)
    K = rig.transforms(D)
    influences = [p["influence"] for p in points]
    X = rig.to_output(rig.points(K, influences)) if influences else np.zeros((0, 3))
    residual = [
        round(1000 * float(np.linalg.norm(x - np.asarray(p["target"]))), 3)
        if p.get("on", True) and float(p.get("weight", 1.0)) > 0
        else None
        for x, p in zip(X, points)
    ]
    pins_mm = []
    for q in pins:
        x = rig.to_output(rig.points(K, [q["influence"]]))[0]
        if q["kind"] == "point":
            d = np.linalg.norm(x - np.asarray(q["target"]))
        else:
            nrm = _unit(q["normal"])
            d = abs(nrm @ (x - np.asarray(q["point"])))
        pins_mm.append(round(1000 * float(d), 3))
    return SolveResult(D, X, residual, taken, cost, pins_mm)

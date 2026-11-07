# OpenSculptBoy
# Apache License, Version 2.0
"""
The poses of the layered portrait pack (``squire.portraits/2``): body language for a companion
who talks, as character games and visual novels show it.

Each pose is built with the posing tools of the pose library (:mod:`anny.poses.authoring`) on
the authoring rig: the legs and the spine of a standing stance, the arms placed by two-bone IK
on targets around the head and the chest, and the hands curled. The rotations follow the
library's convention (each bone at rest along the world axes, a rotation relative to the
parent), so they fit every body as the library's poses do, and the web viewer takes them as
they are (``window.portrait.pose``).

The head keeps one turn in the world in every pose of :data:`POSES`: the turn of ``at_ease``.
The pack's face layers are then rendered once for each framing and fit every pose of that
framing, since the camera of a framing follows the head (``opensculptboy.portrait_pack.layers``).
The frames of the leap (:data:`LEAP`) turn the head freely and carry their own face.

    python -m opensculptboy.portrait_pack.poses     # every pose's rotations as JSON, for a check
"""

from __future__ import annotations

import functools
import json

import numpy as np

# The poses, in the order of the pack, with what each shows.
POSES = {
    "at_ease": "standing straight, the arms down",
    "relaxed": "the weight on one leg, the shoulders loose",
    "hands_behind_head": "both hands behind the head, the elbows out",
    "chin": "one hand on the chin, the other arm across the waist",
    "scratch_head": "one hand scratching the side of the head",
    "arms_crossed": "the arms folded across the chest",
    "hand_on_chest": "a hand on the chest with a slight bow",
    "wave": "one hand raised in a wave",
    "fist": "one fist raised",
    "cheer": "both hands in the air",
}
# The key poses of the leap of joy, in the order they play: the crouch before it, the rise,
# the top with both hands in the air, and the landing.
LEAP = ("crouch", "rise", "peak", "land")
# The pose whose head every pose of POSES keeps.
HEAD_POSE = "at_ease"


@functools.lru_cache(maxsize=1)
def _tools():
    """The posing tools (imported late: they build the authoring rig with torch)."""
    from anny.poses.authoring import poselib as L
    from anny.poses.authoring import poses as PS
    from anny.poses.authoring import posing as P

    return P, L, PS


def _key_points(p):
    """The places the arms reach for, in the pose's world (the legacy frame, metres)."""
    P, _, _ = _tools()
    F = P.fk(p)
    Pw = F[1]
    eyes = 0.5 * (Pw[P.IDX["eye.L"]] + Pw[P.IDX["eye.R"]])
    head = Pw[P.IDX["head"]]
    up = F[0][P.IDX["head"]][:, 1]
    fwd = F[0][P.IDX["head"]][:, 2]
    chin = eyes - 0.095 * up + 0.035 * fwd
    crown = head + 0.19 * up
    chest = Pw[P.IDX["spine01"]]
    return dict(
        F=F,
        eyes=eyes,
        head=head,
        chin=chin,
        crown=crown,
        chest=chest,
        up=up,
        fwd=fwd,
        shoulder={s: Pw[P.IDX[f"upperarm01.{s}"]] for s in "LR"},
        elbow={s: Pw[P.IDX[f"lowerarm01.{s}"]] for s in "LR"},
    )


def _hang(p, s, amount=0.28):
    """A relaxed arm at the side, as the relaxed stance holds it."""
    _, L, _ = _tools()
    L.arm_fk(p, s, abd=7, flex=4, twist=-5, elbow=14, pron=-5)
    L.hands(p, s, amount)


def _legs():
    """The relaxed stance's legs, spine and head, with the arms at rest: each pose sets its arms
    once, since the posing tools turn the arm's bones from where they are."""
    _, L, PS = _tools()
    p, feet = PS.base_stance()
    PS.place_legs(p, feet)
    L.head(p, pitch=2)
    return p


def _stance(hang="LR"):
    """The relaxed stance with the arms of ``hang`` hanging at the sides."""
    p = _legs()
    for s in hang:
        _hang(p, s)
    return p


def at_ease():
    return _stance()


def relaxed():
    _, _, PS = _tools()
    return PS.weight_shift()


def arms_crossed():
    _, _, PS = _tools()
    return PS.arms_crossed()


def chin():
    """The right hand under the chin, its elbow on the left forearm, which crosses the waist."""
    P, L, _ = _tools()
    p = _stance("")
    L.spine(p, bend=2)
    k = _key_points(p)
    sg = -1  # the right side: the figure's right is -X
    wrist = k["chin"] + np.array([sg * 0.012, -0.105, 0.05])
    L.arm_ik(
        p,
        "R",
        wrist,
        pole=k["chest"] + np.array([sg * 0.12, -0.45, 0.35]),
        fingers=P.unit([0.15, 0.9, 0.25]),
        palm=np.array([0.2, 0.1, -1.0]),
    )
    L.hands(p, "R", 0.55, thumb=0.2)
    elbow = _key_points(p)["elbow"]["R"]
    L.arm_ik(
        p,
        "L",
        elbow + np.array([0.075, -0.035, 0.01]),
        pole=k["chest"] + np.array([0.5, -0.4, 0.1]),
        fingers=P.unit([-1.0, 0.0, 0.1]),
        palm=np.array([0.0, 1.0, 0.0]),
    )
    L.hands(p, "L", 0.4)
    return p


def scratch_head():
    """The right hand on the side of the head above the ear, the elbow out to the side."""
    P, L, _ = _tools()
    p = _stance("L")
    k = _key_points(p)
    sg = -1
    target = k["crown"] + np.array([sg * 0.075, -0.05, -0.04])
    L.arm_ik(
        p,
        "R",
        target - P.unit([-sg * 0.4, 0.75, 0.0]) * 0.07,
        pole=k["shoulder"]["R"] + np.array([sg * 0.7, 0.15, 0.25]),
        fingers=P.unit([-sg * 0.45, 0.85, -0.15]),
        palm=np.array([-sg, -0.2, 0.0]),
    )
    L.hands(p, "R", 0.45, thumb=0.25)
    return p


def hands_behind_head():
    """Both hands behind the head, the fingers laced, the elbows out to the sides."""
    P, L, _ = _tools()
    p = _stance("")
    L.spine(p, bend=-3)
    k = _key_points(p)
    for s in "LR":
        sg = L.side_sign(s)
        L.arm_ik(
            p,
            s,
            k["head"] + np.array([sg * 0.06, 0.1, -0.13]),
            pole=k["shoulder"][s] + np.array([sg * 0.7, 0.35, 0.05]),
            fingers=P.unit([-sg * 0.9, 0.35, -0.1]),
            palm=np.array([0.0, 0.0, 1.0]),
        )
        L.hands(p, s, 0.35, thumb=0.2)
    return p


def hand_on_chest():
    """The right hand flat on the left of the chest, with a slight bow."""
    P, L, _ = _tools()
    p = _stance("L")
    L.spine(p, bend=7)
    k = _key_points(p)
    fingers = P.unit([0.55, 0.8, 0.05])
    palm_centre = k["chest"] + np.array([0.045, 0.05, 0.165])
    L.arm_ik(
        p,
        "R",
        palm_centre - fingers * 0.075,
        pole=k["chest"] + np.array([-0.45, -0.6, 0.15]),
        fingers=fingers,
        palm=np.array([0.0, 0.0, -1.0]),
    )
    L.hands(p, "R", 0.12, thumb=0.1)
    return p


def wave():
    """The right hand raised beside the head, the palm forward, the fingers spread."""
    P, L, _ = _tools()
    p = _stance("L")
    L.spine(p, side=-2)
    k = _key_points(p)
    sg = -1
    L.arm_ik(
        p,
        "R",
        k["shoulder"]["R"] + np.array([sg * 0.32, 0.24, 0.08]),
        pole=k["shoulder"]["R"] + np.array([sg * 0.8, -0.2, -0.1]),
        fingers=P.unit([sg * 0.15, 1.0, 0.05]),
        palm=np.array([0.0, 0.0, 1.0]),
    )
    L.hands(p, "R", 0.05, thumb=0.0, spread=6)
    return p


def fist():
    """The right fist raised above the head, the elbow bent."""
    P, L, _ = _tools()
    p = _legs()
    _hang(p, "L", 0.6)
    L.spine(p, side=-3, bend=-2)
    k = _key_points(p)
    sg = -1
    L.arm_ik(
        p,
        "R",
        k["shoulder"]["R"] + np.array([sg * 0.12, 0.5, 0.08]),
        pole=k["shoulder"]["R"] + np.array([sg * 0.8, 0.1, 0.1]),
        fingers=P.unit([sg * 0.1, 1.0, 0.1]),
        palm=np.array([-sg * 0.6, 0.0, 0.8]),
    )
    L.hands(p, "R", 1.0, thumb=0.6)
    return p


def _arms_up(p, spread=0.27, lift=0.52, forward=0.06, fist_amount=0.15):
    P, L, _ = _tools()
    k = _key_points(p)
    for s in "LR":
        sg = L.side_sign(s)
        L.arm_ik(
            p,
            s,
            k["shoulder"][s] + np.array([sg * spread, lift, forward]),
            pole=k["shoulder"][s] + np.array([sg * 0.8, 0.0, -0.2]),
            fingers=P.unit([sg * 0.35, 1.0, 0.05]),
            palm=np.array([-sg * 0.2, 0.0, 1.0]),
        )
        L.hands(p, s, fist_amount, thumb=0.05, spread=4 if fist_amount < 0.5 else 0)


def cheer():
    """Both arms up in a V, the hands open."""
    _, L, _ = _tools()
    p = _stance("")
    L.spine(p, bend=-4)
    _arms_up(p)
    return p


# ------------------------------------------------------------------ the leap
def _jump_legs(bend, toes=0.0, tuck=0.0, pitch=0.0, spine_bend=0.0, head_pitch=0.0):
    """Legs for a frame of the leap: knees bent by `bend` degrees, the heels raised by `toes`
    (the feet on their balls), the lower legs tucked back by `tuck` metres."""
    P, L, PS = _tools()
    feet = {
        s: L.foot_on_floor(
            s,
            L.side_sign(s) * 0.11,
            P.HEADS[P.IDX[f"foot.{s}"]][2] - tuck,
            yaw=8,
            pitch=-toes,
            lift=tuck * 0.9,
        )
        for s in "LR"
    }
    p = P.Pose()
    L.pelvis(p, pitch=pitch)
    p.root[1] += PS.fit_pelvis(p, feet, bend)
    L.spine(p, bend=spine_bend)
    PS.place_legs(
        p,
        feet,
        poles={s: feet[s] + np.array([L.side_sign(s) * 0.1, 0.35, 0.6]) for s in "LR"},
        pitches={s: -toes for s in "LR"},
    )
    L.head(p, pitch=head_pitch)
    return p


def crouch():
    """Before the leap: the knees bent, the arms swung back."""
    _, L, _ = _tools()
    p = _jump_legs(70, pitch=18, spine_bend=14, head_pitch=-6)
    for s in "LR":
        L.arm_fk(p, s, abd=14, flex=-42, twist=-10, elbow=22, pron=-10)
        L.hands(p, s, 0.45)
    return p


def rise():
    """Leaving the ground: the body stretched up on the toes, the arms swinging up."""
    _, L, _ = _tools()
    p = _jump_legs(4, toes=38, spine_bend=-4, head_pitch=-8)
    _arms_up(p, spread=0.18, lift=0.5, forward=0.18, fist_amount=0.35)
    return p


def peak():
    """The top of the leap: both hands in the air, the lower legs tucked back."""
    _, L, _ = _tools()
    p = _jump_legs(38, toes=30, tuck=0.16, spine_bend=-6, head_pitch=-10)
    _arms_up(p, spread=0.3, lift=0.5, forward=0.04, fist_amount=0.15)
    return p


def land():
    """Landing: the knees bent, the arms still up and out."""
    _, L, _ = _tools()
    p = _jump_legs(46, pitch=10, spine_bend=8, head_pitch=-4)
    _arms_up(p, spread=0.36, lift=0.36, forward=0.08, fist_amount=0.25)
    return p


BUILDERS = dict(
    at_ease=at_ease,
    relaxed=relaxed,
    hands_behind_head=hands_behind_head,
    chin=chin,
    scratch_head=scratch_head,
    arms_crossed=arms_crossed,
    hand_on_chest=hand_on_chest,
    wave=wave,
    fist=fist,
    cheer=cheer,
    crouch=crouch,
    rise=rise,
    peak=peak,
    land=land,
)


def _lock_head(p, head_world):
    """Give the head this turn in the world; the neck keeps its own."""
    P, _, _ = _tools()
    P.set_world_rot(p, "head", head_world)


@functools.lru_cache(maxsize=None)
def built(name: str):
    """The pose ``name`` on the authoring rig (a posing.Pose), with the head of HEAD_POSE for
    the poses of POSES."""
    P, _, _ = _tools()
    if name not in BUILDERS:
        raise KeyError(f"Unknown pose {name!r}; the poses are {list(BUILDERS)}.")
    p = BUILDERS[name]()
    if name in POSES and name != HEAD_POSE:
        _lock_head(p, P.fk(built(HEAD_POSE))[0][P.IDX["head"]])
    return p


def head_turn(name: str) -> np.ndarray:
    """The head's turn in the world (3, 3) in a pose."""
    P, _, _ = _tools()
    return P.fk(built(name))[0][P.IDX["head"]]


def page_pose(name: str, bones) -> dict:
    """A pose for the web viewer's ``window.portrait.pose``: ``q`` one rotation (x, y, z, w) per
    bone in the order of ``bones`` (the page's rig), ``root`` the root's offset in hip heights."""
    P, _, _ = _tools()
    p = built(name)
    q = []
    for b in bones:
        if b in P.IDX:
            v = p.q[P.IDX[b]]
            v = v / np.linalg.norm(v)
            q.extend(float(x) for x in v)
        else:
            q.extend((0.0, 0.0, 0.0, 1.0))
    hip = P.RIG.hip_height * P.RIG.scale
    return {"q": q, "root": [float(x) for x in p.root / hip], "label": name}


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--bones", help="a JSON list of the page's bones (default: the library's)"
    )
    args = ap.parse_args(argv)
    if args.bones:
        bones = json.loads(args.bones)
    else:
        import anny.poses

        bones = anny.poses.library().bones
    out = {n: page_pose(n, bones) for n in BUILDERS}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

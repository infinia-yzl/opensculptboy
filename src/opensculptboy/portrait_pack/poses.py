# OpenSculptBoy
# Apache License, Version 2.0
"""
The poses of the layered portrait pack (``squire.portraits/2``): body language for a companion
who talks, as character games and visual novels show it.

The ten stage poses of :data:`POSES` come from the companion's shared pose file
(:mod:`opensculptboy.portrait_pack.shared_poses`): the same file that the stage's 3D
preview reads, so the stills and the 3D preview show the same body. That file keeps the head
and the neck of ``at_ease`` in every pose, so the pack's face layers, rendered once for each
framing in ``at_ease``, fit every pose of that framing (``opensculptboy.portrait_pack.layers``).

The key poses of the leap (:data:`LEAP`) are built here with the posing tools of the pose
library (:mod:`anny.poses.authoring`) on the authoring rig, until the shared file carries them:
the legs of a jump placed by the library's stance tools and the arms placed by two-bone IK on
targets around the shoulders. The rotations follow the library's convention (each bone at rest
along the world axes, a rotation relative to the parent), and the web viewer takes them as they
are (``window.portrait.pose``). The frames of the leap turn the head freely and carry their own
face.

    python -m opensculptboy.portrait_pack.poses     # the leap's rotations as JSON, for a check
"""

from __future__ import annotations

import functools
import json

import numpy as np

# The poses, in the order of the pack, with what each shows. Their rotations come from the shared
# pose file.
POSES = {
    "at_ease": "standing straight, the arms down",
    "relaxed": "the weight on one leg, a hand on the hip",
    "hands_behind_head": "both hands behind the head, the elbows out",
    "chin": "one hand under the chin, the other forearm across the waist",
    "scratch_head": "one hand on the back of the head",
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


BUILDERS = dict(crouch=crouch, rise=rise, peak=peak, land=land)


@functools.lru_cache(maxsize=None)
def built(name: str):
    """The leap's key pose ``name`` on the authoring rig (a posing.Pose)."""
    if name not in BUILDERS:
        raise KeyError(f"Unknown key pose {name!r}; the leap's are {list(BUILDERS)}.")
    return BUILDERS[name]()


def page_pose(name: str, bones) -> dict:
    """A key pose of the leap for the web viewer's ``window.portrait.pose``: ``q`` one rotation
    (x, y, z, w) per bone in the order of ``bones`` (the page's rig), ``root`` the root's offset
    in hip heights."""
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

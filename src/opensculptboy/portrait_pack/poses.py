# OpenSculptBoy
# Apache License, Version 2.0
"""
The poses of the layered portrait pack (``squire.portraits/2``): body language for a companion
who talks, as character games and visual novels show it.

Every pose comes from the companion's shared pose file
(:mod:`opensculptboy.portrait_pack.shared_poses`): the same file that the stage's 3D
preview reads, so the stills and the 3D preview show the same body. That file keeps the head and
the neck of ``at_ease`` in most poses, so the pack's main face set, rendered once for each
framing in ``at_ease``, fits every one of them (``opensculptboy.portrait_pack.layers``). A pose
that turns its own head (the thought's chin, the bow of ``hand_on_chest``, the lowered head of
``downcast``) has a face set of its own, with the expressions that suit it.

The key poses of the leap (:data:`LEAP`) come from the same file, under the names of
:data:`LEAP_POSES`; each leap frame carries its own face.
"""

from __future__ import annotations

# The poses, in the order of the pack, with what each shows. Their rotations come from the shared
# pose file.
POSES = {
    "at_ease": "standing at rest, the weight on one leg, the arms down",
    "relaxed": "the weight on one leg, a hand on the hip",
    "hands_behind_head": "both hands behind the head, the elbows out",
    "chin": "one hand under the chin, the head turned toward the thought",
    "scratch_head": "one hand on the back of the head",
    "arms_crossed": "the arms folded across the chest",
    "hand_on_chest": "a hand on the chest with a slight bow",
    "wave": "one hand raised in a wave",
    "fist": "one fist raised",
    "cheer": "both hands in the air",
    "downcast": "standing at rest with the head lowered and turned away",
}
# The key poses of the leap of joy, in the order they play: the crouch before it, the rise,
# the top with both fists in the air, and the landing.
LEAP = ("crouch", "rise", "peak", "land")
# The shared pose file's name for each key of the leap.
LEAP_POSES = {
    "crouch": "leap_crouch",
    "rise": "leap_rise",
    "peak": "cheer_peak",
    "land": "leap_land",
}
# The pose whose head every pose with the main face set keeps.
HEAD_POSE = "at_ease"


def shared_name(pose: str) -> str:
    """The shared pose file's name for a pose of the pack or a key of the leap."""
    return LEAP_POSES.get(pose, pose)

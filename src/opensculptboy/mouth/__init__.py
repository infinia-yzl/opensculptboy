# OpenSculptBoy
# Apache License, Version 2.0
"""
The mouth of the exports: the ``anny-mouth`` rig, which keeps MakeHuman's jaw bone and tongue
chain (:mod:`opensculptboy.mouth.rig`), and the jaw-open turn fitted to anny's mouth-open
target (:mod:`opensculptboy.mouth.jaw`). docs/mouth.md describes both.
"""

from opensculptboy.mouth.rig import (
    MOUTH_BONES,
    MOUTH_RIG,
    TONGUE_BONES,
    is_mouth_rig,
    mouth_model,
    mouth_model_data,
)

__all__ = [
    "MOUTH_BONES",
    "MOUTH_RIG",
    "TONGUE_BONES",
    "is_mouth_rig",
    "mouth_model",
    "mouth_model_data",
]

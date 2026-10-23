# OpenSculptBoy
# Apache License, Version 2.0
"""
The detail layer of a character: per-vertex offsets on Catmull-Clark subdivisions of anny's
quad mesh, which carry the shape that the parametric model cannot (the residuals of a fit to
a reference mesh, see :mod:`opensculptboy.fit`).

- :mod:`opensculptboy.detail.subdivision`: the meshes of each level, with UVs, skin weights
  and morph targets carried by the subdivision operators (:func:`detail_topology`).
- :mod:`opensculptboy.detail.layer`: the offsets files of the card (:func:`save_offsets`,
  :func:`character_offsets`) and :class:`DetailModel`, which applies them between the blend
  shapes and the skinning.
"""

from opensculptboy.detail.layer import (
    DetailModel,
    character_offsets,
    load_offsets,
    save_offsets,
)
from opensculptboy.detail.subdivision import DetailTopology, detail_topology, level

__all__ = [
    "DetailModel",
    "DetailTopology",
    "character_offsets",
    "detail_topology",
    "level",
    "load_offsets",
    "save_offsets",
]

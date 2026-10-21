# OpenSculptBoy
# Apache License, Version 2.0
"""
The ``sculpt`` topology: anny's mesh with patches where anny's surface cannot follow a
detailed character (docs/sculpt-topology.md).

Every vertex of anny keeps its index; a patch removes some of anny's quads, appends new
vertices as fixed convex blends of anny's vertices, and adds new faces. The patch modules live
in :mod:`opensculptboy.topology.patches` (contract in ``README.md`` next to this file).
"""

from opensculptboy.topology.patch import (
    MAX_ANCHORS,
    BaseQuadMesh,
    Patch,
    PatchError,
    base_quad_mesh,
    blend_rows,
    check_symmetry,
    validate_patch,
)
from opensculptboy.topology.convert import convert_offsets, to_anny, to_sculpt
from opensculptboy.topology.sculpt import (
    TOPOLOGIES,
    SculptData,
    build_model,
    compose,
    is_sculpt,
    read_sculpt_data,
    sculpt_data,
    sculpt_model,
    sculpt_model_data,
    use_sculpt_data,
)

__all__ = [
    "TOPOLOGIES",
    "SculptData",
    "build_model",
    "compose",
    "is_sculpt",
    "read_sculpt_data",
    "sculpt_data",
    "sculpt_model",
    "sculpt_model_data",
    "use_sculpt_data",
    "MAX_ANCHORS",
    "BaseQuadMesh",
    "Patch",
    "PatchError",
    "base_quad_mesh",
    "blend_rows",
    "check_symmetry",
    "convert_offsets",
    "to_anny",
    "to_sculpt",
    "validate_patch",
]

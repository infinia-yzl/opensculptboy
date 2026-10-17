# OpenSculptBoy
# Apache License, Version 2.0
"""
Authoring tools of the export: they build the data files of ``opensculptboy/data`` that the
export reads, and measure them. They run offline, once per change of the model or the rig,
and the export itself never imports them.

- :mod:`opensculptboy.export.authoring.skin_weights` optimises the skin weights of the VRM
  files around the folds of the T-pose (``data/vrm/skin_weights.safetensors``) and prints the
  gate report of the result.
- :mod:`opensculptboy.export.authoring.refine` refines such a data file for one body at a
  time and prints the gate report of the refined weights, to measure what a refinement at
  export time would give.
"""

# OpenSculptBoy
# Apache License, Version 2.0
"""
Authoring tools of the export: they build the data files of ``opensculptboy/data`` that the
export reads, and measure them. They run offline, once per change of the model or the rig.
The VRM export imports one of them, the refinement of the skin weights for each body
(:func:`opensculptboy.export.authoring.refine.body_skin_weights`), and with it the loss of
:mod:`~opensculptboy.export.authoring.skin_weights`.

- :mod:`opensculptboy.export.authoring.skin_weights` optimises the skin weights of the VRM
  files around the folds of the T-pose (``data/vrm/skin_weights.safetensors``) and prints the
  gate report of the result.
- :mod:`opensculptboy.export.authoring.refine` refines that data file for one body at a
  time, as the VRM export does by default, and prints the gate report of the refined
  weights.
"""

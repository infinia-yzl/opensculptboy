# OpenSculptBoy
# Apache License, Version 2.0
"""
Teeth for the exports. Anny's mouth has a cavity and a tongue but no teeth, so an open mouth
shows an empty dark hole. A card with a ``material.teeth`` section
(:class:`opensculptboy.material.TeethMaterial`) adds them to the GLB and VRM exports:

- :mod:`opensculptboy.teeth.reference` takes the teeth of a reference character (the parts
  of an appearance handoff with the role ``mouth`` that are tooth-sized) into the card's rest
  frame, as the bake leaves the fitted mesh: ``<card>.teeth.safetensors``;
- :mod:`opensculptboy.teeth.generic` builds a simple procedural set (two arches of rounded
  crowns, and gums along their roots) behind the lips of any character;
- :mod:`opensculptboy.teeth.mesh` appends the teeth to the mesh that an exporter writes. The
  upper teeth move rigidly with the head; the lower teeth move rigidly with the jaw. Anny's rig
  has no jaw bone: its jaw actions (``jawOpen``, ``jawLeft``, ``jawRight``, ``jawForward``)
  are morph targets, which turn the tongue with the jaw, so each lower tooth takes the rigid
  motion of the tongue in each morph target that moves the tongue rigidly.
"""

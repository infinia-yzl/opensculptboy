# OpenSculptBoy
# Apache License, Version 2.0
"""
Grooms: the hair and the brows of a reference character on a fitted card (docs/reference-fit.md,
"Hair and brows").

- :mod:`opensculptboy.groom.handoff` reads the hair and brow primitives of an appearance handoff
  with their skins, vertex colours and alpha;
- :mod:`opensculptboy.groom.reference` carries them into the card's rest frame through the fit's
  correspondence, with the hair's skin on Anny's bones and its spring chains;
- :mod:`opensculptboy.groom.data` holds them (``<card>.hair.safetensors`` and
  ``<card>.brows.safetensors``, ``opensculptboy/groom@1``), which the card's ``material.hair``
  and ``material.brows`` sections name with their textures;
- :mod:`opensculptboy.groom.export` writes them into the exports: the hair as a mesh of its own,
  skinned to the head, the neck and the spring chains of its chain joints (spring bones in VRM
  files), and the brows as parts of the eyed mesh (:mod:`opensculptboy.eyes.mesh`), bound to the
  skin under them so that they follow its morph targets.
"""

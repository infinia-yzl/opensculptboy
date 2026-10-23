# OpenSculptBoy
# Apache License, Version 2.0
"""
Eyes for the exports: a smooth eyeball fitted to anny's, a cornea shell, the caruncle and the
conjunctiva, their textures and how they attach to the exported mesh.

Anny's eyeballs are spheres of 72 vertices whose texture leaves no room for an iris, and the
skin of the eye socket that shows between the lids and the eyeball at the inner corner of the
eye reads as a dark hole. A card whose skin material holds an ``eyes`` section
(:class:`opensculptboy.material.EyeMaterial`) replaces them in the GLB and VRM exports:

- :mod:`opensculptboy.eyes.geometry` builds the eye parts in each eye's frame (the sphere of
  anny's eyeball, the gaze along -Y of Anny's rest frame): the eyeball with a recessed iris,
  the cornea shell over it, the single-surface eyeball of the VRM files, and the caruncle in
  the inner corner, and finds the lid margins and the skin of the socket;
- :mod:`opensculptboy.eyes.texture` lays out the eye texture (one square per eye, the front of
  the eyeball by its distance from the gaze axis) and draws it: a procedural iris, pupil,
  limbal ring and sclera, or the baked eye texture of the detail layer carried over from
  anny's atlas, with its pupil and iris brought to the card's sizes;
- :mod:`opensculptboy.eyes.mesh` appends the parts to the mesh that an exporter writes, as
  linear maps of the mesh's own vertices, so that the rest mesh, the morph targets, the skin
  weights and the T-pose of a VRM file carry over to them.
"""

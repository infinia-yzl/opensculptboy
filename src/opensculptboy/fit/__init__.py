# OpenSculptBoy
# Apache License, Version 2.0
"""
Fit anny to a reference mesh of any topology, and keep what anny misses as a detail layer.

- :mod:`opensculptboy.fit.handoff`: the appearance handoff folder (``replica.glb``,
  ``appearance.json`` with optional landmarks, ``textures/``), read into Anny's frame.
- :mod:`opensculptboy.fit.reference`: the target surface, its landmarks (given, or from the
  bounding geometry) and the closest-point queries.
- :mod:`opensculptboy.fit.register`: placement, the fit of anny's parameters and pose, and
  the non-rigid refinement of each level (:func:`fit_reference`).
- :mod:`opensculptboy.fit.residual`: the rest-frame offsets of the registered meshes and the
  surface errors.

Example::

    from opensculptboy.fit import fit_handoff

    result = fit_handoff("handoff/", detail=1)
    result.write("character.json")   # the card and character.detail0/1.safetensors

The fit needs SciPy (the ``viewer`` or ``examples`` extra); see ``docs/reference-fit.md``.
"""

from __future__ import annotations


def fit_handoff(folder, detail: int = 1, name: str | None = None, options=None):
    """Fit anny to the skin of an appearance handoff folder (see :func:`fit_reference`)."""
    import pathlib

    from opensculptboy.fit.handoff import load_handoff
    from opensculptboy.fit.reference import Reference
    from opensculptboy.fit.register import fit_reference

    handoff = load_handoff(folder)
    reference = Reference.from_handoff(handoff)
    return fit_reference(
        reference,
        detail=detail,
        name=name or pathlib.Path(handoff.path).resolve().name,
        options=options,
    )


__all__ = ["fit_handoff"]

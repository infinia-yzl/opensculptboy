# OpenSculptBoy
# Apache License, Version 2.0
"""
Statue starters: classical statues fitted from public-domain scans (docs/reference-fit.md, "Statue scans").

- :mod:`opensculptboy.statues.scan`: a scan read, turned upright and to face the front, its
  plinth found and its masked pieces selected.
- :mod:`opensculptboy.statues.handoff`: :func:`scan_to_handoff`, which writes an appearance
  handoff folder that ``opensculptboy fit`` takes.
- :mod:`opensculptboy.statues.fit`: :func:`fit_statue`, the reference fit with the statue's
  lost parts (a hand, the nose) left without data, so that anny keeps its own shape there.
- :mod:`opensculptboy.statues.compare`: the distances between a fitted card and its scan, per
  region of the figure.

Example::

    from opensculptboy.statues import fit_statue, scan_to_handoff

    scan_to_handoff("statue.stl", "handoff/", stature=1.8, licence="CC0-1.0", lost=["hand.l"])
    fit_statue("handoff/", detail=2).write("statue.json")

``python -m opensculptboy.statues SCAN --out FOLDER`` runs the conversion from the command line.
"""

from __future__ import annotations

from opensculptboy.statues.fit import fit_statue, lost_mask
from opensculptboy.statues.handoff import StatueHandoff, scan_to_handoff
from opensculptboy.statues.scan import Mask

__all__ = ["Mask", "StatueHandoff", "fit_statue", "lost_mask", "scan_to_handoff"]

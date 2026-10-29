# OpenSculptBoy
# Apache License, Version 2.0
"""
Statue starters: classical statues fitted from public-domain scans (docs/statues.md).

- :mod:`opensculptboy.statues.scan`: a scan read, turned upright and to face the front, its
  plinth found and its masked pieces selected.
- :mod:`opensculptboy.statues.handoff`: :func:`scan_to_handoff`, which writes an appearance
  handoff folder that ``opensculptboy fit`` takes.
- :mod:`opensculptboy.statues.compare`: the distances between a fitted card and its scan, per
  region of the figure.

Example::

    from opensculptboy.fit import fit_handoff
    from opensculptboy.statues import scan_to_handoff

    scan_to_handoff("doryphoros.stl", "handoff/", stature=1.8, licence="CC0-1.0")
    fit_handoff("handoff/", detail=2).write("doryphoros.json")

``python -m opensculptboy.statues SCAN --out FOLDER`` runs the conversion from the command line.
"""

from __future__ import annotations

from opensculptboy.statues.handoff import StatueHandoff, scan_to_handoff
from opensculptboy.statues.scan import Mask

__all__ = ["Mask", "StatueHandoff", "scan_to_handoff"]

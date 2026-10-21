# OpenSculptBoy
# Apache License, Version 2.0
"""
The patch modules of the ``sculpt`` topology, one module per patch.

The builder (:mod:`opensculptboy.topology.build`) imports every module of this package and
composes their patches in the order of their ``ORDER`` numbers (then their names). The
contract of a module is in ``src/opensculptboy/topology/README.md``.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType


def patch_modules() -> list[ModuleType]:
    """The patch modules of this package, in composition order (``ORDER``, then name)."""
    modules = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        if not callable(getattr(module, "build", None)):
            continue
        modules.append(module)
    return sorted(modules, key=lambda m: (int(getattr(m, "ORDER", 1000)), m.NAME))

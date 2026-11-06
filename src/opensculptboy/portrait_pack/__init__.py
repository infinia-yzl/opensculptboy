# OpenSculptBoy
# Apache License, Version 2.0
"""
The portrait pack (``squire.portraits/1``, docs/portrait-pack.md): a character rendered by the
web viewer into one flat folder of still frames and a manifest, for an app that shows the
character as a 2D portrait that talks and blinks, as visual novels do.

- :mod:`opensculptboy.portrait_pack.contract`: the contract and :func:`check_pack`, which needs
  neither the model nor a browser;
- :mod:`opensculptboy.portrait_pack.faces`: each frame's face as weights of Anny's facial actions;
- :mod:`opensculptboy.portrait_pack.render`: :func:`write_pack`, which renders the frames with the
  viewer's page in headless Chromium (the ``portrait`` extra).
"""

from opensculptboy.portrait_pack.contract import (
    CODES,
    EXPRESSIONS,
    EYES,
    MAX_PACK_BYTES,
    MOUTHS,
    SCHEMA,
    check_manifest,
    check_pack,
    frame_name,
    frame_names,
)
from opensculptboy.portrait_pack.faces import (
    EXPRESSION_MIXES,
    frame_actions,
    pack_faces,
)
from opensculptboy.portrait_pack.render import (
    DEFAULT_SAMPLES,
    DEFAULT_SIZE,
    PortraitError,
    PortraitPage,
    write_pack,
)

__all__ = [
    "CODES",
    "DEFAULT_SAMPLES",
    "DEFAULT_SIZE",
    "EXPRESSIONS",
    "EXPRESSION_MIXES",
    "EYES",
    "MAX_PACK_BYTES",
    "MOUTHS",
    "PortraitError",
    "PortraitPage",
    "SCHEMA",
    "check_manifest",
    "check_pack",
    "frame_actions",
    "frame_name",
    "frame_names",
    "pack_faces",
    "write_pack",
]

# OpenSculptBoy
# Apache License, Version 2.0
"""
Wearables: clothes, footwear, headgear, capes and accessories that fit any Anny body
(docs/wearables.md).

- :mod:`~opensculptboy.wearables.format`: the package ``opensculptboy.wearable/1`` (a
  description, the garment's geometry and materials as GLB, and its binding), with the
  ``provenance`` guard;
- :mod:`~opensculptboy.wearables.bind`: surface binding to Anny's template (triangle,
  barycentrics, offset in the triangle's frame) and rigid binding to one bone;
- :mod:`~opensculptboy.wearables.fit`: the body of a card, the evaluation of worn pieces,
  their skin weights from the body, the layers and the push-out;
- :mod:`~opensculptboy.wearables.cover`: the skin that pieces cover, and the triangles an
  export removes;
- :mod:`~opensculptboy.wearables.wardrobe`: the card's outfits, loose pieces and overrides,
  slot replacement and the ``hides`` rules between pieces;
- :mod:`~opensculptboy.wearables.gltf`: materials and pieces in glTF files;
- :mod:`~opensculptboy.wearables.make`: a package from a garment modelled on a body
  (``opensculptboy wearables bind``);
- :mod:`~opensculptboy.wearables.reference`: reference wearables, built from the
  ``equipment`` section of an appearance handoff and carried onto a fitted card;
- :mod:`~opensculptboy.wearables.mhclo`: MakeHuman clothes assets as packages
  (``opensculptboy wearables import-mhclo``), with :mod:`~opensculptboy.wearables.basemesh`
  for MakeHuman's base mesh on a card's body;
- :mod:`~opensculptboy.wearables.quality`: the quality checks and safe fixes of a garment;
- :mod:`~opensculptboy.wearables.cardedit`: changes to a card's wardrobe
  (``wearables add``, ``remove`` and ``outfit``);
- :mod:`~opensculptboy.wearables.catalogue`: the packages of a folder as JSON entries with
  colours, quality summaries and thumbnails (``wearables catalogue``);
- :mod:`~opensculptboy.wearables.check`: the quality and the fit of a card's outfit on test
  bodies at rest, in the T-pose and in the clips, with gates and a review sheet
  (``wearables check``);
- :mod:`~opensculptboy.wearables.outfitfile`: outfit files ``opensculptboy.outfit/1``
  (``wearables outfit export``, and ``wearables add`` of an outfit file).
"""

from opensculptboy.wearables.bind import (
    BodySurface,
    bind_rigid,
    bind_surface,
    body_weights,
    evaluate,
)
from opensculptboy.wearables.format import (
    FORMAT,
    PRIVATE,
    PUBLIC,
    Binding,
    Garment,
    Material,
    ProvenanceError,
    Wearable,
    WearableError,
    check_destination,
    read_wearable,
    write_wearable,
)
from opensculptboy.wearables.fit import DressedPiece, card_body, dress
from opensculptboy.wearables.wardrobe import WornPiece, worn_pieces

__all__ = [
    "FORMAT",
    "PRIVATE",
    "PUBLIC",
    "Binding",
    "BodySurface",
    "DressedPiece",
    "Garment",
    "Material",
    "ProvenanceError",
    "Wearable",
    "WearableError",
    "WornPiece",
    "bind_rigid",
    "bind_surface",
    "body_weights",
    "card_body",
    "check_destination",
    "dress",
    "evaluate",
    "read_wearable",
    "worn_pieces",
    "write_wearable",
]

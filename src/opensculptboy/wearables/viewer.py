# OpenSculptBoy
# Apache License, Version 2.0
"""
The data that the web viewer needs to dress a character (docs/viewer.md, "Wearables").

The page shapes Anny's body on the quads of the ``anny-quads`` topology, and garments bind to
the triangles of Anny's outer skin (:func:`opensculptboy.wearables.fit.skin_triangles`). Each
skin triangle is one half of a body quad, so the page rebuilds the skin triangles from its own
quads with one code per quad: which diagonal splits it and which of its halves belong to the
skin. :func:`skin_split_codes` computes the codes, and :func:`write_skin_split` writes them into
``viewer/src/wearables/skin_split.ts``::

    python -m opensculptboy.wearables.viewer
"""

from __future__ import annotations

import base64
import pathlib

import numpy as np

# the halves of a quad (a, b, c, d) for the two diagonals, in the winding of the quad
SPLITS = (((0, 1, 2), (0, 2, 3)), ((0, 1, 3), (1, 2, 3)))
TARGET = pathlib.Path("viewer/src/wearables/skin_split.ts")


def page_quads(quad_model=None) -> np.ndarray:
    """(Q, 4) the body quads of the page, in the order of the viewer build (``coarse_quads``)."""
    import torch

    import anny
    from anny.poses.authoring.rig import body_faces_mask

    if quad_model is None:
        quad_model = anny.Anny(topology="anny-quads").to(dtype=torch.float64)
    return quad_model.faces.numpy()[body_faces_mask(quad_model)].astype(np.int64)


def skin_split_codes(quads: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """
    (Q,) one code per quad: bit 0 picks the diagonal (0: a-c, 1: b-d), bit 1 keeps the first
    half and bit 2 the second. Raises ValueError when a skin triangle is not the half of a quad
    in the quad's winding.
    """
    rotations = {}
    for k, t in enumerate(np.asarray(skin, dtype=np.int64)):
        for r in range(3):
            rotations[(int(t[r]), int(t[(r + 1) % 3]), int(t[(r + 2) % 3]))] = k
    used = np.zeros(len(skin), bool)
    codes = np.zeros(len(quads), np.uint8)
    for q, quad in enumerate(np.asarray(quads, dtype=np.int64)):
        best = None
        for s, halves in enumerate(SPLITS):
            found = [rotations.get(tuple(int(quad[i]) for i in h)) for h in halves]
            score = sum(f is not None for f in found)
            if score and (best is None or score > best[0]):
                best = (score, s, found)
        if best is None:
            continue
        _, s, found = best
        codes[q] = (
            s | (2 if found[0] is not None else 0) | (4 if found[1] is not None else 0)
        )
        for f in found:
            if f is not None:
                used[f] = True
    if not used.all():
        raise ValueError(
            f"{int((~used).sum())} skin triangles are not halves of the page's quads."
        )
    return codes


def triangles_from_codes(quads: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """The skin triangles that the codes give (as the page builds them)."""
    out = []
    for quad, code in zip(np.asarray(quads), np.asarray(codes)):
        halves = SPLITS[int(code) & 1]
        for bit, h in ((2, halves[0]), (4, halves[1])):
            if int(code) & bit:
                out.append([int(quad[i]) for i in h])
    return np.asarray(out, dtype=np.int64).reshape(-1, 3)


def skin_split_source(codes: np.ndarray) -> str:
    """The TypeScript module that holds the codes, two per byte, as base64."""
    codes = np.asarray(codes, dtype=np.uint8)
    padded = np.concatenate([codes, np.zeros(len(codes) % 2, np.uint8)])
    packed = (padded[0::2] | (padded[1::2] << 4)).astype(np.uint8)
    text = base64.b64encode(packed.tobytes()).decode("ascii")
    lines = [text[i : i + 120] for i in range(0, len(text), 120)]
    body = "\n".join(f"  '{line}' +" for line in lines)
    return (
        "// OpenSculptBoy\n"
        "// Apache License, Version 2.0\n"
        "//\n"
        "// Written by `python -m opensculptboy.wearables.viewer`: the skin triangles that garments bind to\n"
        "// (opensculptboy.wearables.fit.skin_triangles) as halves of the page's body quads (coarse_quads). One code\n"
        "// per quad, two per byte: bit 0 picks the diagonal (0: a-c, 1: b-d), bit 1 keeps the first half, bit 2 the\n"
        "// second.\n\n"
        f"export const QUAD_COUNT = {len(codes)};\n"
        "export const SPLIT_CODES =\n"
        f"{body}\n  '';\n"
    )


def write_skin_split(path: str | pathlib.Path = TARGET) -> pathlib.Path:
    """Compute the codes from Anny's model and write the TypeScript module."""
    import torch

    from opensculptboy import Character
    from opensculptboy.wearables.fit import skin_triangles

    model = Character().build_model(dtype=torch.float64)
    quads = page_quads()
    codes = skin_split_codes(quads, skin_triangles(model))
    path = pathlib.Path(path)
    path.write_text(skin_split_source(codes), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_skin_split())

# OpenSculptBoy
# Apache License, Version 2.0
"""
Synthetic statues for the page tests of the pose editor: a character card of Anny's default
body with a light scan beside it (``scan.glb``, as a statue starter carries it, entry
``detail.fit.reference.ghost``), the scan being Anny's own skin in another pose, a little
outside the skin, so that the reference points placed on it pose the character towards that
pose.
"""

import json
import pathlib
import struct

import numpy as np


def scan_glb(vertices_anny, triangles) -> bytes:
    """A GLB of one mesh in glTF's axes (y up): positions quantised to 16 bits with the node's
    scale and translation (KHR_mesh_quantization), indices of 32 bits, no normals."""
    V = np.asarray(vertices_anny, np.float64)
    G = np.stack([V[:, 0], V[:, 2], -V[:, 1]], 1)
    lo, hi = G.min(0), G.max(0)
    scale = (hi - lo).max() / 65535.0
    q = np.round((G - lo) / scale).astype(np.uint16)
    pos = np.concatenate([q, np.zeros((len(q), 1), np.uint16)], 1).tobytes()
    idx = np.asarray(triangles, np.uint32).tobytes()
    bin_ = pos + idx
    gltf = {
        "asset": {"version": "2.0", "generator": "opensculptboy test"},
        "extensionsUsed": ["KHR_mesh_quantization"],
        "extensionsRequired": ["KHR_mesh_quantization"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "scale": [float(scale)] * 3, "translation": lo.tolist()}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}
        ],
        "buffers": [{"byteLength": len(bin_)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos), "byteStride": 8},
            {"buffer": 0, "byteOffset": len(pos), "byteLength": len(idx)},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5123,
                "count": len(q),
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": q.max(0).astype(int).tolist(),
            },
            {
                "bufferView": 1,
                "componentType": 5125,
                "count": int(np.size(triangles)),
                "type": "SCALAR",
            },
        ],
    }
    js = json.dumps(gltf).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    bin_ += b"\0" * ((4 - len(bin_) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_)
    return (
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(js), 0x4E4F534A)
        + js
        + struct.pack("<II", len(bin_), 0x004E4942)
        + bin_
    )


def points_statue(folder, chains=("arm.r", "arm.l"), seed=5, outside=0.001):
    """A card and its scan in ``folder``: the scan is the default body posed by a random pose of
    the chains (``test_reference_points._random_pose``), ``outside`` metres outside its skin.
    Returns the files, the rig, the rest and the true rotations."""
    from opensculptboy.posing.point_place import _vertex_normals
    from test.test_reference_points import _random_pose, _rig

    folder = pathlib.Path(folder)
    rig, rest = _rig()
    truth = _random_pose(rig, rest, list(chains), seed, share=(0.04, 0.15))
    V = rig.vertices(truth)
    V = V + outside * _vertex_normals(V, rig.faces)
    (folder / "scan.glb").write_bytes(scan_glb(V, rig.faces))
    card = {
        "schema": 5,
        "name": "Points statue",
        "phenotype": {},
        "detail": {
            "fit": {
                "reference": {
                    "ghost": {
                        "file": "scan.glb",
                        "frame": "card",
                        "triangles": int(len(rig.faces)),
                        "vertices": int(len(V)),
                        "licence": "CC0-1.0",
                        "credit": "A synthetic scan of the default character",
                        "source": "test",
                    }
                }
            }
        },
    }
    (folder / "character.json").write_text(json.dumps(card), encoding="utf-8")
    return [folder / "character.json", folder / "scan.glb"], rig, rest, truth

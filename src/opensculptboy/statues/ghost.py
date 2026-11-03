# OpenSculptBoy
# Apache License, Version 2.0
"""
A statue starter's light scan: the figure's scan as the studio's ghost (DESIGN-0013, "Reference
in the view"), a ``scan.glb`` of a few megabytes in the starter package.

The scan is the run's handoff ``replica.glb`` (the figure's skin with its carved hair and
drapery, such as a hat and locks; the removed pieces, such as a sword or a base, stay out),
taken into the frame of the starter's card: the frame of the card's
own posed body, as Anny's forward pass with the card's pose gives it, where the page draws the
skin and the props (``x_card = (x_scan - t) R`` with the card's ``detail.fit.placement``,
``x_scan = R x_card + t``). It is reduced to about :data:`TRIANGLES` triangles by quadric
collapses (:func:`opensculptboy.export.decimate.decimate`) and written in glTF axes with
``EXT_meshopt_compression`` and ``KHR_mesh_quantization`` (:mod:`opensculptboy.export.meshopt`):

- one mesh of one primitive, without normals or texture coordinates (the page computes smooth
  normals from the triangles);
- ``POSITION``: VEC3 of UNSIGNED_SHORT, not normalised, in a byte stride of 8, compressed in the
  mode ``ATTRIBUTES``; the mesh's node carries the dequantisation as ``translation`` and a
  uniform ``scale`` (position = scale q + translation);
- the indices: UNSIGNED_INT, compressed in the mode ``TRIANGLES``;
- buffer 1 is the extension's fallback buffer, which holds no data, so both extensions are
  required;
- ``asset.copyright`` and ``asset.extras`` carry the scan's licence, credit and source.

The card names the file in ``detail.fit.reference.ghost`` (:func:`ghost_entry`), beside the
scan's credit. :func:`build_ghost` writes the file, and :func:`add_ghost` puts one into an
existing starter zip (``python -m opensculptboy.statues ghost``).
"""

from __future__ import annotations

import io
import json
import pathlib
import struct
import zipfile

import numpy as np

FILE = "scan.glb"
TRIANGLES = 300_000
FRAME = "card"
COMPRESSION = "EXT_meshopt_compression"
GHOST_COLOUR = (0.80, 0.78, 0.74, 1.0)
EXTENSIONS = ["EXT_meshopt_compression", "KHR_mesh_quantization"]


def scan_mesh(handoff) -> tuple[np.ndarray, np.ndarray]:
    """The figure's scan of a handoff folder (``replica.glb``, every part) as vertices in
    Anny's axes (the handoff's frame) and triangles."""
    import trimesh

    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    scene = trimesh.load(
        str(pathlib.Path(handoff) / "replica.glb"), force="scene", process=False
    )
    vertices, triangles, base = [], [], 0
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name]
        V = np.asarray(mesh.vertices, np.float64)
        V = V @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]
        vertices.append(V)
        triangles.append(np.asarray(mesh.faces, np.int64) + base)
        base += len(V)
    V = np.concatenate(vertices) @ GLTF_TO_ANNY.T
    return V, np.concatenate(triangles)


def to_card(vertices: np.ndarray, placement: dict) -> np.ndarray:
    """Points of the scan (Anny's axes) in the frame of a card placed by ``placement``
    (``x_scan = rotation x_card + translation``)."""
    R = np.asarray(placement["rotation"], np.float64)
    t = np.asarray(placement["translation"], np.float64)
    return (np.asarray(vertices, np.float64) - t) @ R


def reduce(vertices: np.ndarray, triangles: np.ndarray, target: int = TRIANGLES):
    """The mesh welded, without degenerate triangles, and reduced to about ``target``
    triangles; vertices that no triangle uses are dropped."""
    from opensculptboy.export.decimate import decimate, weld

    V = np.asarray(vertices, np.float64)
    T = np.asarray(triangles, np.int64)
    ids = weld(V)
    first = np.zeros(int(ids.max()) + 1, np.int64)
    first[ids[::-1]] = np.arange(len(V))[::-1]
    V, T = V[first], ids[T]
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 2] != T[:, 0])]
    if len(T) > target:
        T = np.asarray(decimate(V, T, target).triangles, np.int64)
    used = np.unique(T)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[T]


def _pad(data: bytes, fill: bytes) -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


def ghost_glb(
    vertices: np.ndarray, triangles: np.ndarray, reference: dict | None = None
):
    """The GLB (bytes) of a mesh in the card's frame (Anny's axes) as the module describes it,
    and its numbers (vertices, triangles, bytes, the quantisation step in mm)."""
    from opensculptboy.export.meshopt import (
        encode_index_buffer,
        encode_vertex_buffer,
        order_triangles,
    )

    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    reference = reference or {}
    # Anny's axes to glTF's: x -> GLTF_TO_ANNY.T x
    G = np.asarray(vertices, np.float64) @ GLTF_TO_ANNY
    T = np.asarray(triangles, np.int64)
    T = T[order_triangles(T, G)]
    indices, remap, decoded = encode_index_buffer(T)
    order = np.empty(int(remap.max()) + 1, np.int64)
    order[remap[remap >= 0]] = np.nonzero(remap >= 0)[0]
    G = G[order]
    lo = G.min(0)
    step = max(float(np.ptp(G, 0).max()) / 65535.0, 1e-9)
    q = np.zeros((len(G), 4), np.uint16)
    q[:, :3] = np.clip(np.round((G - lo) / step), 0, 65535).astype(np.uint16)
    attributes = encode_vertex_buffer(q.view(np.uint8).reshape(len(G), 8))
    n_v, n_i = len(G), 3 * len(decoded)
    a_len = len(attributes)
    a_pad = (4 - a_len % 4) % 4
    binary = attributes + bytes(a_pad) + indices
    licence = reference.get("licence")
    credit = reference.get("credit")
    doc = {
        "asset": {
            "version": "2.0",
            "generator": "OpenSculptBoy statue ghost",
            **(
                {
                    "copyright": " ".join(
                        x for x in (credit, licence and f"({licence})") if x
                    )
                }
                if credit or licence
                else {}
            ),
            "extras": {
                "title": reference.get("title"),
                "source": reference.get("source"),
                "licence": licence,
                "credit": credit,
                "frame": FRAME,
            },
        },
        "extensionsUsed": EXTENSIONS,
        "extensionsRequired": EXTENSIONS,
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {
                "name": "scan",
                "mesh": 0,
                "translation": [float(x) for x in lo],
                "scale": [step, step, step],
            }
        ],
        "meshes": [
            {
                "name": "scan",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "scan",
                "pbrMetallicRoughness": {
                    "baseColorFactor": list(GHOST_COLOUR),
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.6,
                },
                "doubleSided": True,
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5123,
                "count": n_v,
                "type": "VEC3",
                "min": q[:, :3].min(0).tolist(),
                "max": q[:, :3].max(0).tolist(),
            },
            {"bufferView": 1, "componentType": 5125, "count": n_i, "type": "SCALAR"},
        ],
        "bufferViews": [
            {
                "buffer": 1,
                "byteOffset": 0,
                "byteLength": 8 * n_v,
                "byteStride": 8,
                "target": 34962,
                "extensions": {
                    COMPRESSION: {
                        "buffer": 0,
                        "byteOffset": 0,
                        "byteLength": a_len,
                        "byteStride": 8,
                        "count": n_v,
                        "mode": "ATTRIBUTES",
                    }
                },
            },
            {
                "buffer": 1,
                "byteOffset": 8 * n_v,
                "byteLength": 4 * n_i,
                "target": 34963,
                "extensions": {
                    COMPRESSION: {
                        "buffer": 0,
                        "byteOffset": a_len + a_pad,
                        "byteLength": len(indices),
                        "byteStride": 4,
                        "count": n_i,
                        "mode": "TRIANGLES",
                    }
                },
            },
        ],
        "buffers": [
            {"byteLength": len(binary)},
            {
                "byteLength": 8 * n_v + 4 * n_i,
                "extensions": {COMPRESSION: {"fallback": True}},
            },
        ],
    }
    text = _pad(json.dumps(doc, separators=(",", ":")).encode("utf-8"), b" ")
    blob = _pad(binary, b"\0")
    size = 12 + 8 + len(text) + 8 + len(blob)
    out = (
        struct.pack("<III", 0x46546C67, 2, size)
        + struct.pack("<II", len(text), 0x4E4F534A)
        + text
        + struct.pack("<II", len(blob), 0x004E4942)
        + blob
    )
    info = {
        "vertices": n_v,
        "triangles": len(decoded),
        "bytes": len(out),
        "step_mm": round(1000 * step, 4),
    }
    return out, info


def read_ghost(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """A ghost GLB's vertices (glTF axes, the card's frame) and triangles, decoded with this
    module's own reading of the formats (:func:`decode_vertex_buffer`,
    :func:`decode_index_buffer`): the check of a written file."""
    from opensculptboy.export.meshopt import decode_index_buffer, decode_vertex_buffer

    length = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20 : 20 + length])
    binary = data[20 + length + 8 :]
    views = doc["bufferViews"]
    a = views[0]["extensions"][COMPRESSION]
    i = views[1]["extensions"][COMPRESSION]
    raw = decode_vertex_buffer(
        binary[a["byteOffset"] : a["byteOffset"] + a["byteLength"]], a["count"], 8
    )
    q = raw.view(np.uint16).reshape(-1, 4)[:, :3].astype(np.float64)
    node = doc["nodes"][0]
    V = q * np.asarray(node["scale"]) + np.asarray(node["translation"])
    T = decode_index_buffer(
        binary[i["byteOffset"] : i["byteOffset"] + i["byteLength"]], i["count"]
    )
    return V, T.reshape(-1, 3)


def ghost_entry(info: dict, reference: dict, source_triangles: int) -> dict:
    """The card's ``detail.fit.reference.ghost``: the file, its frame, its numbers and the
    scan's licence and credit."""
    return {
        "file": FILE,
        "frame": FRAME,
        "triangles": int(info["triangles"]),
        "vertices": int(info["vertices"]),
        "bytes": int(info["bytes"]),
        "source_triangles": int(source_triangles),
        "compression": COMPRESSION,
        "licence": reference.get("licence"),
        "credit": reference.get("credit"),
        "source": reference.get("source"),
    }


def build_ghost(handoff, card: dict, target: int = TRIANGLES, log=None):
    """The ghost of a starter's card (a dict with ``detail.fit.placement`` and
    ``detail.fit.reference``) from the run's handoff folder: the GLB's bytes and the card's
    entry."""
    fit = card["detail"]["fit"]
    reference = fit.get("reference") or {}
    V, T = scan_mesh(handoff)
    V = to_card(V, fit["placement"])
    source_triangles = len(T)
    V, T = reduce(V, T, target)
    data, info = ghost_glb(V, T, reference)
    if log:
        log(
            f"ghost: {source_triangles} to {info['triangles']} triangles, "
            f"{info['bytes'] / 1e6:.2f} MB"
        )
    return data, ghost_entry(info, reference, source_triangles)


def add_ghost(zip_in, handoff, zip_out=None, target: int = TRIANGLES, log=None) -> dict:
    """Put the ghost of a starter zip's card (from the run's ``handoff``) into the zip as
    :data:`FILE`, with its entry in the card; ``zip_out`` defaults to the zip itself. Returns
    the entry and the zip's size."""
    zip_in = pathlib.Path(zip_in)
    zip_out = pathlib.Path(zip_out) if zip_out else zip_in
    with zipfile.ZipFile(zip_in) as z:
        entries = {n: z.read(n) for n in z.namelist() if n != FILE}
    card = json.loads(entries["character.json"])
    data, entry = build_ghost(handoff, card, target, log)
    card["detail"]["fit"]["reference"]["ghost"] = entry
    entries["character.json"] = (json.dumps(card, indent=2) + "\n").encode("utf-8")
    entries[FILE] = data
    write_zip(zip_out, entries)
    return {"zip": str(zip_out), "bytes": zip_out.stat().st_size, "ghost": entry}


def write_zip(path, entries: dict) -> None:
    """A starter zip of ``entries`` (name -> bytes), each entry deflated (meshopt's output is
    made to shrink further under a general compressor)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in entries.items():
            z.writestr(name, data)
    path.write_bytes(buffer.getvalue())

# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of a statue starter's light scan (``opensculptboy.statues.ghost``) and of the meshopt
codecs it is written with (``opensculptboy.export.meshopt``), with synthetic meshes: the round
trip of both codecs through this module's decoders, through the page's decoder
(``viewer/src/meshopt_ref.js``) and meshoptimizer's own (node), the GLB's structure, its frame,
and a starter zip that carries it.
"""

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest
import zipfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
NODE_MODULES = REPO / "viewer" / "node_modules"


def _sphere(n=40, m=80, centre=(0.0, 0.0, 1.0), radii=(0.3, 0.2, 0.5)):
    """A closed UV sphere (Anny's axes) with welded poles."""
    th = np.linspace(0, np.pi, n)[1:-1]
    ph = np.linspace(0, 2 * np.pi, m, endpoint=False)
    T_, P_ = np.meshgrid(th, ph, indexing="ij")
    ring = np.stack(
        [np.sin(T_) * np.cos(P_), np.sin(T_) * np.sin(P_), np.cos(T_)], -1
    ).reshape(-1, 3)
    V = np.concatenate([[[0, 0, 1]], ring, [[0, 0, -1]]]) * radii + centre
    idx = 1 + np.arange(len(th) * m).reshape(len(th), m)
    nxt = np.roll(idx, -1, 1)
    quads_a = np.stack([idx[:-1], nxt[:-1], nxt[1:]], -1).reshape(-1, 3)
    quads_b = np.stack([idx[:-1], nxt[1:], idx[1:]], -1).reshape(-1, 3)
    top = np.stack([np.zeros(m, int), nxt[0], idx[0]], -1)
    bottom = np.stack([np.full(m, len(V) - 1), idx[-1], nxt[-1]], -1)
    return V, np.concatenate([top, quads_a, quads_b, bottom])


def _rotations(T):
    return {tuple(np.roll(t, k)) for t in T for k in range(3)}


class TestMeshopt(unittest.TestCase):
    def test_index_codec_round_trip(self):
        from opensculptboy.export.meshopt import (
            decode_index_buffer,
            encode_index_buffer,
            order_triangles,
        )

        rng = np.random.default_rng(1)
        V, T = _sphere()
        cases = {
            "walk": T[order_triangles(T, V)],
            "shuffled": T[rng.permutation(len(T))],
            # triangles of three new vertices, of old ones and of far ones
            "scattered": rng.integers(0, 5000, (400, 3)),
            "one": np.array([[2, 0, 1]]),
        }
        for name, tris in cases.items():
            with self.subTest(name):
                data, remap, decoded = encode_index_buffer(tris)
                got = decode_index_buffer(data, 3 * len(tris)).reshape(-1, 3)
                np.testing.assert_array_equal(got, decoded)
                used = np.unique(tris)
                # the new numbers are 0 .. used - 1, once each
                np.testing.assert_array_equal(
                    np.sort(remap[used]), np.arange(len(used))
                )
                # each decoded triangle is a rotation of its input (the winding holds)
                for a, b in zip(remap[tris], decoded):
                    self.assertIn(tuple(b), {tuple(np.roll(a, k)) for k in range(3)})
        data, _, _ = encode_index_buffer(cases["walk"])
        # a walk over the sphere's edges costs under 2 bytes a triangle
        self.assertLess(len(data) / len(T), 2.0)

    def test_vertex_codec_round_trip(self):
        from opensculptboy.export.meshopt import (
            decode_vertex_buffer,
            encode_vertex_buffer,
        )

        rng = np.random.default_rng(2)
        smooth = np.cumsum(rng.integers(-3, 4, (700, 4)), 0).astype(np.int64) % 65536
        for name, data in {
            "smooth": smooth.astype(np.uint16).view(np.uint8).reshape(700, 8),
            "noise": rng.integers(0, 256, (300, 12), dtype=np.uint8),
            "constant": np.full((17, 4), 7, np.uint8),
            "one": np.arange(8, dtype=np.uint8)[None],
        }.items():
            with self.subTest(name):
                stream = encode_vertex_buffer(data)
                got = decode_vertex_buffer(stream, *data.shape)
                np.testing.assert_array_equal(got, data)
        self.assertLess(len(encode_vertex_buffer(np.zeros((256, 8), np.uint8))), 120)

    @unittest.skipUnless(
        shutil.which("node") and (NODE_MODULES / "meshoptimizer").exists(),
        "needs node and viewer/node_modules (cd viewer && npm ci)",
    )
    def test_the_page_and_meshoptimizer_decode_the_streams(self):
        from opensculptboy.export.meshopt import (
            encode_index_buffer,
            encode_vertex_buffer,
            order_triangles,
        )

        V, T = _sphere()
        T = T[order_triangles(T, V)]
        indices, remap, decoded = encode_index_buffer(T)
        q = np.zeros((len(V), 4), np.uint16)
        q[remap, :3] = np.round((V - V.min(0)) * 20000).astype(np.uint16)
        vertices = encode_vertex_buffer(q.view(np.uint8).reshape(len(V), 8))
        script = r"""
import fs from 'fs';
import { pathToFileURL } from 'url';
const [modules, ref, dir] = process.argv.slice(2);
const { MeshoptDecoder } = await import(pathToFileURL(modules + '/meshoptimizer/meshopt_decoder.mjs'));
globalThis.window = {};
new Function(fs.readFileSync(ref, 'utf8'))();
const meta = JSON.parse(fs.readFileSync(dir + '/meta.json'));
const ib = new Uint8Array(fs.readFileSync(dir + '/i.bin')), vb = new Uint8Array(fs.readFileSync(dir + '/v.bin'));
const out = {};
for (const [name, D] of [['page', window.MeshoptDecoderRef], ['meshoptimizer', MeshoptDecoder]]) {
  await D.ready;
  const ti = new Uint8Array(meta.indices * 4); D.decodeGltfBuffer(ti, meta.indices, 4, ib, 'TRIANGLES');
  const tv = new Uint8Array(meta.vertices * 8); D.decodeGltfBuffer(tv, meta.vertices, 8, vb, 'ATTRIBUTES');
  out[name] = { indices: Array.from(new Uint32Array(ti.buffer)), vertices: Array.from(new Uint16Array(tv.buffer)) };
}
console.log(JSON.stringify(out));
"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            (tmp / "decode.mjs").write_text(script, encoding="utf-8")
            (tmp / "i.bin").write_bytes(indices)
            (tmp / "v.bin").write_bytes(vertices)
            (tmp / "meta.json").write_text(
                json.dumps({"indices": decoded.size, "vertices": len(V)})
            )
            run = subprocess.run(
                [
                    "node",
                    str(tmp / "decode.mjs"),
                    str(NODE_MODULES),
                    str(REPO / "viewer" / "src" / "meshopt_ref.js"),
                    str(tmp),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            got = json.loads(run.stdout)
        for name in ("page", "meshoptimizer"):
            with self.subTest(name):
                np.testing.assert_array_equal(got[name]["indices"], decoded.ravel())
                np.testing.assert_array_equal(got[name]["vertices"], q.ravel())


class TestGhost(unittest.TestCase):
    REFERENCE = {
        "title": "Kouros",
        "licence": "PDM-1.0",
        "credit": "A museum: cast 1",
        "source": "https://example.org/kouros",
    }

    def test_the_glb_holds_the_mesh_in_glTF_axes(self):
        from scipy.spatial import cKDTree

        from opensculptboy.fit.handoff import GLTF_TO_ANNY
        from opensculptboy.statues.ghost import EXTENSIONS, ghost_glb, read_ghost

        V, T = _sphere()
        data, info = ghost_glb(V, T, self.REFERENCE)
        self.assertEqual(info["triangles"], len(T))
        self.assertEqual(info["vertices"], len(V))
        length = int.from_bytes(data[12:16], "little")
        doc = json.loads(data[20 : 20 + length])
        self.assertEqual(doc["extensionsRequired"], EXTENSIONS)
        self.assertEqual(doc["asset"]["extras"]["licence"], "PDM-1.0")
        self.assertIn("A museum: cast 1", doc["asset"]["copyright"])
        self.assertEqual(doc["accessors"][0]["componentType"], 5123)
        self.assertEqual(doc["bufferViews"][0]["byteStride"], 8)
        self.assertTrue(doc["buffers"][1]["extensions"]["EXT_meshopt_compression"])
        G, got = read_ghost(data)
        # glTF's axes: Anny's z up is glTF's y up
        A = G @ GLTF_TO_ANNY.T
        d, _ = cKDTree(V).query(A)
        self.assertLess(d.max(), info["step_mm"] / 1000)
        # every triangle is one of the input's, its winding kept
        nearest = cKDTree(V).query(A)[1]
        self.assertTrue(_rotations(nearest[got]) <= _rotations(T))

    def test_the_scan_goes_into_the_card_frame(self):
        from opensculptboy.statues.ghost import to_card

        angle = np.radians(20)
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        t = np.array([0.1, -0.2, 0.05])
        x = np.random.default_rng(0).normal(size=(10, 3))
        placed = x @ R.T + t
        np.testing.assert_allclose(
            to_card(placed, {"rotation": R.tolist(), "translation": t.tolist()}), x
        )

    def test_a_reduced_mesh_keeps_its_shape(self):
        from scipy.spatial import cKDTree

        from opensculptboy.statues.ghost import reduce

        V, T = _sphere(80, 160)
        W, U = reduce(V, T, 3000)
        self.assertLessEqual(len(U), 3100)
        self.assertEqual(len(np.unique(U)), len(W))
        d, _ = cKDTree(W).query(V)
        self.assertLess(np.median(d), 0.01)

    def test_a_starter_zip_takes_its_ghost(self):
        import trimesh

        from opensculptboy.fit.handoff import GLTF_TO_ANNY
        from opensculptboy.statues.ghost import FILE, add_ghost, read_ghost

        V, T = _sphere()
        R = np.eye(3)
        t = np.array([0.0, 0.1, 0.0])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            handoff = tmp / "handoff"
            handoff.mkdir()
            # the scan placed on the card: x_scan = R x_card + t, in glTF axes on file
            mesh = trimesh.Trimesh((V @ R.T + t) @ GLTF_TO_ANNY, T, process=False)
            mesh.export(str(handoff / "replica.glb"))
            card = {
                "schema": 5,
                "name": "Kouros",
                "detail": {
                    "levels": [],
                    "fit": {
                        "placement": {
                            "rotation": R.tolist(),
                            "translation": t.tolist(),
                        },
                        "reference": dict(self.REFERENCE),
                    },
                },
            }
            zip_in = tmp / "kouros.zip"
            with zipfile.ZipFile(zip_in, "w") as z:
                z.writestr("character.json", json.dumps(card))
                z.writestr("thumbnail.png", b"png")
            result = add_ghost(zip_in, handoff, tmp / "out.zip", target=1000)
            with zipfile.ZipFile(tmp / "out.zip") as z:
                names = set(z.namelist())
                written = json.loads(z.read("character.json"))
                data = z.read(FILE)
            self.assertEqual(names, {"character.json", "thumbnail.png", FILE})
            entry = written["detail"]["fit"]["reference"]["ghost"]
            self.assertEqual(entry, result["ghost"])
            self.assertEqual(entry["file"], FILE)
            self.assertEqual(entry["frame"], "card")
            self.assertEqual(entry["licence"], "PDM-1.0")
            self.assertEqual(entry["source_triangles"], len(T))
            self.assertLessEqual(entry["triangles"], 1100)
            self.assertEqual(entry["bytes"], len(data))
            G, _ = read_ghost(data)
            # in the card's frame: the placement taken off
            A = G @ GLTF_TO_ANNY.T
            self.assertAlmostEqual(A[:, 2].max(), 1.5, delta=0.002)
            self.assertAlmostEqual(A[:, 1].mean(), 0.0, delta=0.01)

    def test_the_starter_build_carries_the_ghost(self):
        import trimesh

        from opensculptboy.fit.handoff import GLTF_TO_ANNY
        from opensculptboy.statues.pipeline import StatueConfig
        from opensculptboy.statues.starter import build_starter

        config = StatueConfig.from_dict(
            {
                "format": "opensculptboy.statue/1",
                "name": "kouros",
                "scan": "kouros.stl",
                **self.REFERENCE,
            }
        )
        V, T = _sphere()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "kouros"
            for sub in ("card", "meshes", "handoff"):
                (root / sub).mkdir(parents=True)
            card = {
                "schema": 2,
                "name": "kouros",
                "rig": "anny",
                "topology": "anny",
                "detail": {
                    "levels": [{"subdiv": 0, "offsets": "kouros.detail0.safetensors"}],
                    "fit": {
                        "placement": {
                            "rotation": np.eye(3).tolist(),
                            "translation": [0, 0, 0],
                        }
                    },
                },
            }
            (root / "card" / "kouros.json").write_text(json.dumps(card))
            (root / "card" / "kouros.detail0.safetensors").write_bytes(b"0" * 10)
            trimesh.Trimesh(V @ GLTF_TO_ANNY, T, process=False).export(
                str(root / "handoff" / "replica.glb")
            )
            trimesh.creation.box(extents=(0.4, 0.3, 1.8)).export(
                str(root / "meshes" / "fitted_level0.ply")
            )
            out = pathlib.Path(tmp) / "kouros.zip"
            result = build_starter(
                root, config, out, levels=(0,), with_props=False, ghost=500
            )
            self.assertEqual(result["ghost"]["file"], "scan.glb")
            with zipfile.ZipFile(out) as z:
                self.assertIn("scan.glb", z.namelist())
                written = json.loads(z.read("character.json"))
            self.assertEqual(
                written["detail"]["fit"]["reference"]["ghost"]["licence"], "PDM-1.0"
            )
            light = json.loads(out.with_suffix(".light.json").read_text("utf-8"))
            self.assertNotIn("ghost", light["detail"]["fit"]["reference"])
            # 0 leaves the ghost out
            build_starter(root, config, out, levels=(0,), with_props=False, ghost=0)
            with zipfile.ZipFile(out) as z:
                self.assertNotIn("scan.glb", z.namelist())


if __name__ == "__main__":
    unittest.main()

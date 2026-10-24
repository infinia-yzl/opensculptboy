# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of ``opensculptboy wearables bind`` (``opensculptboy.wearables.make``,
docs/wearables.md, "Make a wearable") with synthetic garment files written by the test: a
tube around the torso in two materials and a ring around the head, in the frame of an export
of the default card. The checks cover the package's fields, the round trip on the template,
a dressed export on two random bodies, the licence refusal, the far-from-body warning and
refusal, own skin weights, and the command end to end.
"""

import contextlib
import io
import json
import pathlib
import tempfile
import unittest

import numpy as np

from opensculptboy import Character, export_glb
from opensculptboy.export.body import ANNY_TO_GLTF
from opensculptboy.export.document import ARRAY_BUFFER, GltfDocument
from opensculptboy.wearables import WornPiece, dress, evaluate, read_wearable
from opensculptboy.wearables import write_wearable
from opensculptboy.wearables.cover import coverage_gaps
from opensculptboy.wearables.export import Dressing
from opensculptboy.wearables.gltf import index_accessor
from opensculptboy.wearables.make import (
    GarmentError,
    ground_lift,
    make_wearable,
    parse_hides,
    to_anny,
)
from test.gltf_reader import GLB, evaluate_nodes, structure_errors
from test.wearable_fixtures import template_body, template_model

LICENCE = {
    "spdx": "CC0-1.0",
    "author": "OpenSculptBoy tests",
    "design_source": "synthetic test garment, made in the test",
}
NODE_SHIFT = np.array([0.0, 0.25, 0.0])  # the tube's node translation


def _to_file(points, lift):
    """Anny's frame to the frame of an export (glTF axes, on the floor)."""
    return np.asarray(points) @ ANNY_TO_GLTF.T + np.array([0.0, lift, 0.0])


def tube(body, rings=20, around=64, gap=0.012, bump=0.0):
    """A tube around the torso between the hips and the chest, ``gap`` off the widest skin
    in each direction; the lowest ring moves out by ``bump``. Returns vertices, outward
    normals and triangles in Anny's frame, and the front (y < 0) of each triangle."""
    v = body.vertices
    low, h = float(v[:, 2].min()), float(np.ptp(v[:, 2]))
    heights = np.linspace(low + 0.54 * h, low + 0.70 * h, rings)
    angles = np.linspace(0, 2 * np.pi, around, endpoint=False)
    direction = np.stack([np.cos(angles), np.sin(angles)], 1)
    points, normals = [], []
    for k, z in enumerate(heights):
        slab = v[(np.abs(v[:, 2] - z) < 0.02) & (np.abs(v[:, 0]) < 0.19)]
        centre = slab[:, :2].mean(axis=0)
        rel = slab[:, :2] - centre
        radial = rel @ direction.T  # (n, around)
        angle = np.arctan2(rel[:, 1], rel[:, 0])
        near = np.abs(np.angle(np.exp(1j * (angle[:, None] - angles[None])))) < 0.25
        radius = np.where(near, radial, -np.inf).max(axis=0) + gap
        radius += bump if k == 0 else 0.0
        xy = centre + radius[:, None] * direction
        points.append(np.column_stack([xy, np.full(around, z)]))
        normals.append(np.column_stack([direction, np.zeros(around)]))
    i, j = np.meshgrid(np.arange(rings - 1), np.arange(around), indexing="ij")
    p = i * around + j
    q = i * around + (j + 1) % around
    triangles = np.concatenate(
        [
            np.stack([p, q, q + around], -1).reshape(-1, 3),
            np.stack([p, q + around, p + around], -1).reshape(-1, 3),
        ]
    )
    vertices = np.concatenate(points)
    front = vertices[triangles].mean(axis=1)[:, 1] < 0
    return vertices, np.concatenate(normals), triangles, front


def ring(body, n=48, m=6, r=0.004):
    """A torus around the head at the brow, in Anny's frame."""
    head = body.bone_poses[body.bone_labels.index("head"), :3, 3]
    v = body.vertices
    near = np.linalg.norm(v[:, :2] - head[:2], axis=1) < 0.12
    z = float(v[near, 2].max()) - 0.06
    band = near & (np.abs(v[:, 2] - z) < 0.01)
    radius = float(np.linalg.norm(v[band, :2] - head[:2], axis=1).max()) + 0.012
    A, B = np.meshgrid(
        np.linspace(0, 2 * np.pi, n, endpoint=False),
        np.linspace(0, 2 * np.pi, m, endpoint=False),
        indexing="ij",
    )
    R = radius + r * np.cos(B)
    vertices = np.stack(
        [head[0] + R * np.cos(A), head[1] + R * np.sin(A), z + r * np.sin(B)], -1
    ).reshape(-1, 3)
    normals = np.stack(
        [np.cos(B) * np.cos(A), np.cos(B) * np.sin(A), np.sin(B)], -1
    ).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    p, q = i * m + j, ((i + 1) % n) * m + j
    pr, qr = i * m + (j + 1) % m, ((i + 1) % n) * m + (j + 1) % m
    triangles = np.concatenate(
        [
            np.stack([p, q, qr], -1).reshape(-1, 3),
            np.stack([p, qr, pr], -1).reshape(-1, 3),
        ]
    )
    return vertices, normals, triangles


def write_tube_glb(path, body, normals=True, shift=(0.0, 0.0, 0.0), lift=None, **kw):
    """The tube as a GLB file: one mesh node (translated by NODE_SHIFT) with two
    primitives that share one set of vertices, materials ``linen`` (front) and ``wool``
    (back), and UVs."""
    vertices, outward, triangles, front = tube(body, **kw)
    lift = ground_lift(body) if lift is None else lift
    positions = _to_file(vertices, lift) + np.asarray(shift) - NODE_SHIFT
    doc = GltfDocument("test")
    attributes = {
        "POSITION": doc.accessor(positions.astype(np.float32), ARRAY_BUFFER, True),
        "TEXCOORD_0": doc.accessor(
            np.column_stack([np.arange(len(positions)) % 64 / 64.0] * 2).astype(
                np.float32
            ),
            ARRAY_BUFFER,
        ),
    }
    if normals:
        attributes["NORMAL"] = doc.accessor(
            (outward @ ANNY_TO_GLTF.T).astype(np.float32), ARRAY_BUFFER
        )
    linen = doc.add_material({"name": "linen", "pbrMetallicRoughness": {}})
    wool = doc.add_material({"name": "wool", "pbrMetallicRoughness": {}})
    primitives = [
        {
            "attributes": dict(attributes),
            "indices": index_accessor(doc, triangles[rows], len(positions)),
            "material": material,
            "extras": {"group": group},
        }
        for rows, material, group in ((front, linen, "front"), (~front, wool, "back"))
    ]
    mesh = doc.add_mesh({"name": "tube", "primitives": primitives})
    node = doc.add_node(
        {"name": "Tube", "mesh": mesh, "translation": NODE_SHIFT.tolist()}
    )
    doc.scene_nodes = [node]
    doc.write(path)
    return vertices + to_anny(np.asarray(shift)[None], 0.0)


def write_ring_glb(path, body, skinned=False):
    """The ring as a GLB file; ``skinned`` binds it to a joint node ``head`` at the
    origin with an identity inverse bind matrix."""
    vertices, normals, triangles = ring(body)
    positions = _to_file(vertices, ground_lift(body))
    doc = GltfDocument("test")
    attributes = {
        "POSITION": doc.accessor(positions.astype(np.float32), ARRAY_BUFFER, True),
        "NORMAL": doc.accessor(
            (normals @ ANNY_TO_GLTF.T).astype(np.float32), ARRAY_BUFFER
        ),
    }
    nodes = []
    entry = {"name": "Ring"}
    if skinned:
        attributes["JOINTS_0"] = doc.accessor(
            np.zeros((len(positions), 4), np.uint8), ARRAY_BUFFER
        )
        attributes["WEIGHTS_0"] = doc.accessor(
            np.tile(np.float32([1, 0, 0, 0]), (len(positions), 1)), ARRAY_BUFFER
        )
        joint = doc.add_node({"name": "head"})
        nodes.append(joint)
        entry["skin"] = doc.add_skin({"joints": [joint]})
    gold = doc.add_material({"name": "gold", "pbrMetallicRoughness": {}})
    entry["mesh"] = doc.add_mesh(
        {
            "name": "ring",
            "primitives": [
                {
                    "attributes": attributes,
                    "indices": index_accessor(doc, triangles, len(positions)),
                    "material": gold,
                }
            ],
        }
    )
    nodes.append(doc.add_node(entry))
    doc.scene_nodes = nodes
    doc.write(path)
    return vertices


class TestBind(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.body = template_body()
        cls.tube_path = cls.folder / "tube_garment.glb"
        cls.tube_vertices = write_tube_glb(cls.tube_path, cls.body)
        cls.ring_path = cls.folder / "ring_garment.glb"
        cls.ring_vertices = write_ring_glb(cls.ring_path, cls.body)
        cls.tube, cls.tube_report = make_wearable(
            cls.tube_path,
            cls.body,
            "tube",
            "torso",
            50,
            licence=LICENCE,
            hides=["cape:back,cape:hood", "boots:top"],
        )
        cls.ring, cls.ring_report = make_wearable(
            cls.ring_path, cls.body, "ring", "head", 60, rigid="head", licence=LICENCE
        )
        for w in (cls.tube, cls.ring):
            write_wearable(cls.folder / f"{w.name}.wearable.json", w)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_package_fields(self):
        tube = read_wearable(self.folder / "tube.wearable.json")
        self.assertEqual((tube.slot, tube.layer, tube.weights), ("torso", 50, "body"))
        self.assertEqual(tube.binding.method, "surface")
        self.assertEqual(tube.binding.body, "anny-template")
        self.assertEqual(sorted(tube.garment.groups), ["linen", "wool"])
        self.assertEqual(sorted(tube.group_covers), ["linen", "wool"])
        self.assertTrue(tube.cover_ray)
        self.assertEqual(int(tube.covers.sum()), 0)
        self.assertEqual(
            tube.hides,
            [
                {"piece": "cape", "groups": ["back", "hood"]},
                {"piece": "boots", "groups": ["top"]},
            ],
        )
        self.assertEqual(tube.licence["spdx"], "CC0-1.0")
        self.assertIsNotNone(tube.garment.uvs)
        # the two primitives shared one set of vertices: each keeps the ones it uses
        self.assertGreaterEqual(len(tube.garment.vertices), len(self.tube_vertices))
        ring = read_wearable(self.folder / "ring.wearable.json")
        self.assertEqual((ring.binding.method, ring.binding.bone), ("rigid", "head"))
        self.assertEqual(ring.weights, "rigid")
        report = self.tube_report
        self.assertEqual(report["vertices"], len(tube.garment.vertices))
        self.assertLess(report["bind_distance_mm_max"], 50.0)
        self.assertEqual(report["far_vertices"], 0)
        self.assertGreater(report["covered_skin_vertices"], 300)
        self.assertEqual(report["warnings"], [])

    def test_round_trip_on_the_template(self):
        from scipy.spatial import cKDTree

        for name, source in (
            ("tube", self.tube_vertices),
            ("ring", self.ring_vertices),
        ):
            with self.subTest(name):
                w = read_wearable(self.folder / f"{name}.wearable.json")
                back, _ = evaluate(w.binding, self.body)
                # every vertex returns to its place in the garment file (float32)
                distance, _ = cKDTree(source).query(back)
                self.assertLess(float(distance.max()), 2e-6)
                np.testing.assert_allclose(back, w.garment.vertices, rtol=0, atol=1e-6)
        self.assertLess(self.tube_report["round_trip_mm_max"], 1e-6)
        self.assertLess(self.ring_report["round_trip_mm_max"], 1e-6)

    def test_frame_is_the_default_export(self):
        """The default card's export stands Anny's template on the floor by the lift that
        the command takes off."""
        path = self.folder / "base.glb"
        export_glb(path, Character(), template_model())
        glb = GLB(path)
        lift = ground_lift(self.body)
        expected = _to_file(self.body.vertices, lift)
        skinned = np.concatenate(
            [p.positions for prims in evaluate_nodes(glb).values() for p in prims]
        )
        self.assertAlmostEqual(float(skinned[:, 1].min()), 0.0, places=5)
        np.testing.assert_allclose(skinned.min(0), expected.min(0), atol=1e-5)
        np.testing.assert_allclose(skinned.max(0), expected.max(0), atol=1e-5)
        # the bind pose (the vertices as the file stores them) is Anny's centred frame
        attributes = glb.json["meshes"][0]["primitives"][0]["attributes"]
        accessor = glb.json["accessors"][attributes["POSITION"]]
        np.testing.assert_allclose(
            accessor["min"], expected.min(0) - [0, lift, 0], atol=1e-5
        )

    def test_dressed_export_on_two_random_bodies(self):
        rng = np.random.default_rng(11)
        model = template_model()
        worn = [WornPiece(w.name, w.name, None, "", w, True) for w in (self.tube,)]
        for k in range(2):
            phenotype = {n: float(rng.uniform(0, 1)) for n in model.phenotype_labels}
            with self.subTest(k=k, phenotype=phenotype):
                card = Character(
                    phenotype=phenotype,
                    wearables=[
                        {"id": n, "file": f"{n}.wearable.json", "enabled": True}
                        for n in ("tube", "ring")
                    ],
                )
                path = self.folder / f"dressed{k}.glb"
                summary = export_glb(
                    path, card, model, detail_dir=self.folder, dressing=Dressing()
                )
                glb = GLB(path)
                self.assertEqual(structure_errors(glb), [])
                names = [n.get("name") for n in glb.json["nodes"]]
                self.assertIn("tube", names)
                self.assertIn("ring", names)
                self.assertGreater(summary["removed_body_triangles"], 100)
                from opensculptboy.wearables import card_body

                body = card_body(card, model)
                piece = dress(body, worn)[0]
                mask = np.zeros(len(body.vertices), bool)
                for extra in self.tube.group_covers.values():
                    mask |= extra
                gaps = coverage_gaps(body, piece, mask, tolerance=0.002)
                self.assertGreater(gaps["covered"], 300)
                self.assertLess(gaps["outside"], 0.01 * gaps["covered"], gaps)

    def test_far_vertices_warn_and_a_garment_off_the_body_fails(self):
        from scipy.spatial import cKDTree

        path = self.folder / "bumped.glb"
        write_tube_glb(path, self.body, normals=False, bump=0.07)
        _, report = make_wearable(
            path, self.body, "bumped", "torso", 50, licence=LICENCE
        )
        self.assertGreaterEqual(report["far_vertices"], 64)
        self.assertTrue(any("more than 50 mm" in w for w in report["warnings"]))
        self.assertTrue(any("no normals" in w for w in report["warnings"]))
        path = self.folder / "away.glb"
        write_tube_glb(path, self.body, shift=(1.0, 0.0, 0.0))
        with self.assertRaisesRegex(GarmentError, "from the skin"):
            make_wearable(path, self.body, "away", "torso", 50, licence=LICENCE)
        path = self.folder / "centred.glb"  # the export's bind pose, or --no-ground
        source = write_tube_glb(path, self.body, lift=0.0)
        for frame in ("auto", "centred"):
            w, report = make_wearable(
                path, self.body, "centred", "torso", 50, licence=LICENCE, frame=frame
            )
            self.assertEqual(report["frame"], "centred")
            back, _ = evaluate(w.binding, self.body)
            distance, _ = cKDTree(source).query(back)
            self.assertLess(float(distance.max()), 2e-6)

    def test_own_weights_and_hides_items(self):
        path = self.folder / "ring_skinned.glb"
        write_ring_glb(path, self.body, skinned=True)
        w, _ = make_wearable(
            path, self.body, "ring", "head", 60, weights="own", licence=LICENCE
        )
        self.assertEqual(w.weights, "own")
        self.assertEqual(w.binding.joint_names, ["head"])
        with self.assertRaisesRegex(GarmentError, "skinned"):
            make_wearable(
                self.ring_path, self.body, "ring", "head", 60, weights="own",
                licence=LICENCE,
            )  # fmt: skip
        with self.assertRaisesRegex(GarmentError, "PIECE:GROUP"):
            parse_hides(["cape"])


class TestBindCommand(unittest.TestCase):
    def test_command(self):
        from opensculptboy.cli import main

        body = template_body()
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            garment = folder / "tube_garment.glb"
            write_tube_glb(garment, body)
            out = folder / "wear" / "tube"
            args = ["wearables", "bind", str(garment), "--out", str(out)]
            args += ["--slot", "torso", "--layer", "50", "--label", "Tube"]
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                main(args + ["--slot", "torsos"])  # an unknown slot
            printed = io.StringIO()
            extra = ["--licence", "CC0-1.0", "--author", "tests"]
            extra += ["--design-source", "synthetic", "--groups-from", "extras"]
            extra += ["--hides", "cape:back", "--tag", "Tube, test", "--tag", "linen"]
            extra += ["--description", "A tube around the torso."]
            with (
                contextlib.redirect_stdout(printed),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main(args + extra), 0)
            report = json.loads(printed.getvalue())
            for suffix in (".wearable.json", ".glb", ".binding.safetensors"):
                self.assertTrue((folder / "wear" / f"tube{suffix}").exists(), suffix)
            self.assertEqual(report["groups"], ["front", "back"])
            self.assertLess(report["round_trip_mm_max"], 1e-6)
            self.assertGreater(report["covered_skin_vertices"], 300)
            self.assertEqual(report["quality"]["errors"], 0)
            self.assertIn("counts", [i["code"] for i in report["quality"]["items"]])
            description = json.loads(
                (folder / "wear" / "tube.wearable.json").read_text()
            )
            self.assertEqual(description["label"], "Tube")
            self.assertEqual(description["licence"]["author"], "tests")
            self.assertEqual(description["tags"], ["tube", "test", "linen"])
            self.assertEqual(description["description"], "A tube around the torso.")
            self.assertEqual(
                description["hides"], [{"piece": "cape", "groups": ["back"]}]
            )


if __name__ == "__main__":
    unittest.main()

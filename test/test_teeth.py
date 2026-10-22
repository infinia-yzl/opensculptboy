# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the teeth of ``opensculptboy.teeth`` and their use: the card's ``material.teeth``
section and its validation, the generic teeth behind closed lips, the toothed mesh (the upper
teeth still in every facial action, the lower teeth rigid with the jaw in ``jawOpen`` and still
in ``tongueOut``), the teeth of a reference brought into the rest frame, the GLB file
(materials, extras, the teeth against the file's morph targets), the VRM files and the
command's ``--teeth``. The files of cards without the section stay as they were
(``TestGltfReference`` in ``test.test_gltf_export``).
"""

import json
import pathlib
import tempfile
import unittest
import warnings

import numpy as np
import torch

from opensculptboy import Character, export_glb
from opensculptboy.material import TeethMaterial, check_material
from opensculptboy.teeth.generic import generic_teeth, mouth_frame
from opensculptboy.teeth.mesh import TeethData, mouth_rows, rigid, toothed_mesh
from test.gltf_reader import GLB, evaluate_nodes, structure_errors


def _kabsch_residual(a: np.ndarray, b: np.ndarray) -> float:
    """The largest distance left after the best rigid map of ``a`` onto ``b``."""
    R, t = rigid(a, b)
    return float(np.linalg.norm(a @ R.T + t - b, axis=1).max())


def _hits(origins, direction, vertices, triangles, reach=0.05) -> np.ndarray:
    """Whether a ray from each origin along ``direction`` meets the triangles within
    ``reach`` metres (Moller-Trumbore)."""
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    e1, e2 = b - a, c - a
    p = np.cross(direction, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-14
    a, e1, e2, p, det = a[ok], e1[ok], e2[ok], p[ok], det[ok]
    hit = np.zeros(len(origins), bool)
    for i, o in enumerate(origins):
        s = o - a
        u = (s * p).sum(1) / det
        q = np.cross(s, e1)
        v = (q @ direction) / det
        t = (e2 * q).sum(1) / det
        inside = (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t < reach)
        hit[i] = inside.any()
    return hit


class TestTeethCard(unittest.TestCase):
    def test_section(self):
        self.assertIsNone(Character().teeth_material())
        card = Character(material={"teeth": {}})
        teeth = card.teeth_material()
        self.assertEqual(teeth.source, "generic")
        self.assertTrue(teeth.gums)
        # the section alone keeps the exporters' own skin
        self.assertIsNone(card.skin_material())
        reference = TeethMaterial.from_dict({"mesh": "c.teeth.safetensors"})
        self.assertEqual(reference.source, "reference")
        check_material({"teeth": {"color": [0.7, 0.6, 0.5], "roughness": 0.3}})
        for bad in (
            {"colour": [1, 1, 1]},
            {"source": "dentures"},
            {"roughness": 2.0},
            {"gums": "yes"},
            {"mesh": 3},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_material({"teeth": bad})
        with self.assertRaises(ValueError):
            TeethMaterial.from_dict({"source": "reference"})
        # the card round-trips
        self.assertEqual(Character.from_dict(card.to_dict()), card)
        # beside a skin material, as the bake writes them
        baked = Character(material={"roughness": 0.4, "teeth": {"mesh": "c.t"}})
        self.assertEqual(baked.skin_material().roughness, 0.4)
        self.assertEqual(baked.teeth_material().source, "reference")

    def test_data_file(self):
        rng = np.random.default_rng(0)
        data = TeethData(
            points=rng.normal(size=(5, 3)),
            normals=rng.normal(size=(5, 3)),
            uvs=rng.random((5, 2)),
            triangles=[[0, 1, 2], [2, 3, 4]],
            arch=[0, 0, 1, 1, 1],
            part=[0, 1],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "t.teeth.safetensors"
            data.save(path)
            back = TeethData.load(path)
        for key in ("points", "normals", "uvs", "triangles", "arch", "part"):
            np.testing.assert_array_equal(getattr(back, key), getattr(data, key))


class TestToothedMesh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from opensculptboy.export.body import morph_target_rows, rest_rows

        cls.card = Character()
        cls.model = cls.card.build_model(dtype=torch.float64)
        cls.names = ["jawOpen", "jawLeft", "tongueOut", "mouthSmileLeft"]
        targets, facial, faces = morph_target_rows(cls.model, cls.card, cls.names)
        rows, _ = rest_rows(cls.model, cls.card, facial, faces)
        cls.rest = rows[0]
        cls.offsets = rows[1:] - rows[0]
        cls.teeth = generic_teeth(cls.model, cls.rest)
        cls.toothed = toothed_mesh(cls.model, None, None, cls.rest, cls.teeth)

    def test_generic_structure(self):
        t = self.teeth
        self.assertEqual(set(np.unique(t.arch)), {0, 1})
        self.assertEqual(set(np.unique(t.part)), {0, 1})
        self.assertEqual(t.triangles.max(), len(t.points) - 1)
        self.assertTrue(np.isfinite(t.points).all() and np.isfinite(t.normals).all())
        # 14 crowns an arch, in the middle of the mouth, the upper arch above the lower
        mouth = mouth_frame(self.model, self.rest)
        upper, lower = t.points[t.arch == 0], t.points[t.arch == 1]
        self.assertGreater(upper[:, 2].mean(), lower[:, 2].mean())
        self.assertLess(abs(t.points[:, 0].mean() - mouth["front"][0]), 1e-3)
        width = np.ptp(t.points[:, 0])
        self.assertTrue(0.03 < width < 0.08, width)

    def test_behind_closed_lips(self):
        # no tooth lies in front of the lips' inner margin, and a ray from a tooth toward
        # the front meets the face, unless it leaves through the slit that anny's lips keep
        # between them at rest
        from opensculptboy.export.body import triangulated_faces

        triangles, _ = triangulated_faces(self.model)
        mouth = mouth_frame(self.model, self.rest)
        ring = self.rest[mouth["ring"]]
        crowns = np.unique(self.teeth.triangles[self.teeth.part == 0])
        points = self.teeth.points[crowns]
        for p in points:
            beside = ring[np.abs(ring[:, 0] - p[0]) < 0.003]
            if len(beside):
                self.assertGreater(p[1], beside[:, 1].min())
        near = (
            np.linalg.norm(self.rest[triangles].mean(1) - mouth["front"], axis=1) < 0.08
        )
        points = points[::3]
        hit = _hits(points, np.array([0.0, -1.0, 0.0]), self.rest, triangles[near], 0.1)
        middle = ring[np.abs(ring[:, 0] - mouth["front"][0]) < 0.005]
        low, high = middle[:, 2].min() - 0.001, middle[:, 2].max() + 0.001
        slit = (points[:, 2] > low) & (points[:, 2] < high)
        self.assertTrue((hit | slit).all(), f"{int((~(hit | slit)).sum())} in the open")
        self.assertGreater(hit.mean(), 0.9)

    def test_rest_and_head_motion(self):
        x = self.toothed.vertices(self.rest)
        np.testing.assert_allclose(x[len(self.rest) :], self.teeth.points, atol=1e-12)
        # a rigid motion of the whole mesh moves the teeth with it
        angle = 0.4
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        moved = self.toothed.vertices(self.rest @ R.T + 0.1)
        np.testing.assert_allclose(
            moved[len(self.rest) :], self.teeth.points @ R.T + 0.1, atol=1e-9
        )

    def test_targets(self):
        out = self.toothed.targets(self.offsets)
        n = len(self.rest)
        teeth = out[:, n:]
        upper = self.teeth.arch == 0
        lower = ~upper
        p = self.teeth.points
        for r, name in enumerate(self.names):
            with self.subTest(target=name):
                # the upper teeth stay with the skull in every facial action
                self.assertLess(np.abs(teeth[r, upper]).max(), 1e-9)
                if name.startswith("jaw"):
                    # the lower teeth turn rigidly, as anny's tongue does
                    self.assertGreater(np.abs(teeth[r, lower]).max(), 0.002)
                    self.assertLess(
                        _kabsch_residual(p[lower], p[lower] + teeth[r, lower]), 1e-9
                    )
                else:
                    self.assertLess(np.abs(teeth[r, lower]).max(), 1e-9)
        # jawOpen lowers the front teeth and keeps them behind the lower lip's motion
        jaw = teeth[self.names.index("jawOpen"), lower]
        self.assertLess(jaw[:, 2].mean(), -0.003)

    def test_skin_weights(self):
        weights = self.model.vertex_bone_weights.double().numpy()
        indices = self.model.vertex_bone_indices.numpy()
        w, i = self.toothed.skin_weights(weights, indices)
        labels = list(self.model.bone_labels)
        tail_w, tail_i = w[len(self.rest) :], i[len(self.rest) :]
        self.assertTrue((tail_w == tail_w[0]).all() and (tail_i == tail_i[0]).all())
        self.assertEqual(labels[tail_i[0, 0]], "head")
        np.testing.assert_allclose(tail_w.sum(1), 1.0)


class TestReferenceTeeth(unittest.TestCase):
    def _box(self, centre, size):
        corners = np.array(
            [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float
        )
        faces = np.array(
            [
                [0, 1, 3],
                [0, 3, 2],
                [4, 6, 7],
                [4, 7, 5],
                [0, 4, 5],
                [0, 5, 1],
                [2, 3, 7],
                [2, 7, 6],
                [0, 2, 6],
                [0, 6, 4],
                [1, 5, 7],
                [1, 7, 3],
            ]
        )
        return np.asarray(centre) + 0.5 * np.asarray(size) * corners, faces

    def test_teeth_of_a_reference(self):
        from opensculptboy.fit.handoff import HandoffPart
        from opensculptboy.teeth.reference import reference_teeth

        parts = []
        for k in range(4):  # two upper and two lower teeth, each its own part
            for z, name in ((1.462, "upper"), (1.452, "lower")):
                v, f = self._box([0.006 * k - 0.009, -0.07, z], [0.005, 0.006, 0.009])
                parts.append(
                    HandoffPart(
                        f"tooth_{name}_{k}",
                        "mouth",
                        v,
                        f,
                        base_color_factor=(0.8, 0.79, 0.7, 1.0),
                        roughness=0.3,
                    )
                )
        gum, f = self._box([0.0, -0.05, 1.47], [0.05, 0.05, 0.01])  # too big: lining
        parts.append(HandoffPart("mouth_lining", "mouth", gum, f))
        parts.append(HandoffPart("skin", "skin", gum + 1.0, f))
        # the fitted head, placed on the reference: turned and moved from its rest pose
        rng = np.random.default_rng(3)
        rest = rng.normal(size=(50, 3)) * 0.05 + [0.0, -0.05, 0.65]
        angle = 0.05
        R = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        )
        posed = rest @ R.T + [0.0, 0.0, 0.8]
        data, values = reference_teeth(parts, posed, rest, np.arange(50))
        self.assertEqual(len(data.points), 8 * 8)  # the lining stays out
        self.assertEqual(values, {"color": [0.8, 0.79, 0.7], "roughness": 0.3})
        # back in the rest frame: the inverse of the placement
        source = np.concatenate([p.vertices for p in parts[:8]])
        np.testing.assert_allclose(
            data.points, (source - [0.0, 0.0, 0.8]) @ R, atol=1e-9
        )
        # the arches by height
        heights = np.concatenate(
            [np.full(8, p.vertices[:, 2].mean()) for p in parts[:8]]
        )
        np.testing.assert_array_equal(data.arch, (heights < 1.457).astype(np.uint8))
        self.assertTrue((data.part == 0).all())


class TestExports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.model = Character().build_model(dtype=torch.float64)
        cls.card = Character(name="teeth", material={"teeth": {}})
        cls.path = cls.folder / "teeth.glb"
        export_glb(cls.path, cls.card, cls.model)
        cls.glb = GLB(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _primitives(self, glb):
        g = glb.json
        return {
            g["materials"][p["material"]]["name"]: p
            for p in g["meshes"][0]["primitives"]
        }

    def test_structure(self):
        g = self.glb.json
        self.assertEqual(structure_errors(self.glb), [])
        prims = self._primitives(self.glb)
        self.assertEqual(sorted(prims), ["gums", "skin", "teeth"])
        for p in prims.values():  # the teeth share the body's vertices and targets
            self.assertEqual(p["attributes"], prims["skin"]["attributes"])
            self.assertEqual(p.get("targets"), prims["skin"].get("targets"))
        materials = {m["name"]: m for m in g["materials"]}
        for name in ("teeth", "gums"):
            extras = materials[name]["extras"]["opensculptboy_teeth"]
            self.assertEqual(extras["format"], "opensculptboy/teeth@1")
            self.assertEqual(extras["role"], name)
        info = g["scenes"][0]["extras"]["opensculptboy"]["teeth"]
        self.assertEqual(info["source"], "generic")
        self.assertGreater(info["points"], 0)

    def test_teeth_follow_the_jaw(self):
        names = self.glb.json["meshes"][0]["extras"]["targetNames"]
        prims = self._primitives(self.glb)
        rows = np.unique(self.glb.accessor(prims["teeth"]["indices"]))
        rest = evaluate_nodes(self.glb)
        node = min(rest)
        p0 = rest[node][0].positions[rows]
        weights = np.zeros(len(names))
        weights[names.index("jawOpen")] = 1.0
        moved = evaluate_nodes(self.glb, weights=list(weights))[node][0].positions[rows]
        still = np.linalg.norm(moved - p0, axis=1) < 1e-6
        # the upper teeth stay, the lower teeth turn rigidly (float32 file)
        self.assertTrue(still.any() and (~still).any())
        self.assertLess(_kabsch_residual(p0[~still], moved[~still]), 2e-5)
        self.assertGreater(np.linalg.norm(moved - p0, axis=1).max(), 0.003)

    def test_with_eyes_and_detail(self):
        # the teeth after the eyes: every part keeps its own primitive
        path = self.folder / "eyes_teeth.glb"
        card = Character(material={"teeth": {"gums": False}, "eyes": {}})
        export_glb(path, card, self.model)
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        self.assertEqual(
            sorted(self._primitives(glb)),
            ["conjunctiva", "cornea", "eyes", "skin", "teeth"],
        )

    def test_vrm(self):
        from opensculptboy.export.vrm import export_vrm
        from test.test_vrm_schema import check_vrm0, check_vrm1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for version in ("1.0", "0.x"):
                with self.subTest(version=version):
                    path = self.folder / f"teeth{version}.vrm"
                    export_vrm(
                        path,
                        self.card,
                        self.model,
                        version=version,
                        author="tests",
                        refine_weights=False,
                    )
                    glb = GLB(path)
                    g = glb.json
                    self.assertEqual(structure_errors(glb), [])
                    check = check_vrm1 if version == "1.0" else check_vrm0
                    self.assertEqual(check(g), [])
                    self.assertEqual(
                        [m["name"] for m in g["materials"]], ["skin", "teeth", "gums"]
                    )
                    extras = g["scenes"][0]["extras"]["opensculptboy"]["teeth"]
                    self.assertEqual(extras["source"], "generic")

    def test_command(self):
        from opensculptboy.cli import main

        card = self.folder / "card.json"
        card.write_text(json.dumps(Character(name="cli").to_dict()))
        out = self.folder / "cli.glb"
        self.assertEqual(
            main(["export", str(out), "--character", str(card), "--teeth"]), 0
        )
        self.assertIn("teeth", self._primitives(GLB(out)))
        values = self.folder / "teeth.json"
        values.write_text(json.dumps({"gums": False, "color": [0.6, 0.6, 0.55]}))
        self.assertEqual(
            main(
                ["export", str(out), "--character", str(card), "--teeth", str(values)]
            ),
            0,
        )
        prims = self._primitives(GLB(out))
        self.assertIn("teeth", prims)
        self.assertNotIn("gums", prims)


if __name__ == "__main__":
    unittest.main()

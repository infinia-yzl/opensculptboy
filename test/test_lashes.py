# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the lashes of ``opensculptboy.eyes.lashes``: the card's ``material.lashes`` section,
the lid margins, the generic lashes and their texture, the lashes on the eyed mesh (back at
rest, with the closing lid in a blink, still with the other eye's blink and in a turn of the
eye), the lashes of a reference brought into the rest frame with their textures stacked, the
lashes file, and the GLB and VRM files with and without lashes.
"""

import io
import pathlib
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from opensculptboy import Character, export_glb
from opensculptboy.eyes.lashes import (
    LOWER,
    UPPER,
    LashData,
    card_lashes,
    generic_texture,
    reference_lashes,
)
from opensculptboy.eyes.mesh import eyed_mesh
from opensculptboy.material import EyeMaterial, LashMaterial, check_material
from test.gltf_reader import GLB, evaluate_nodes, structure_errors


class TestLashCard(unittest.TestCase):
    def test_section(self):
        # the lashes come with the eyes
        self.assertIsNone(Character().lash_material())
        self.assertIsNone(Character(material={"lashes": {}}).lash_material())
        lashes = Character(material={"eyes": {}}).lash_material()
        self.assertEqual(lashes.source, "generic")
        self.assertTrue(lashes.lower)
        off = Character(material={"eyes": {}, "lashes": False})
        self.assertIsNone(off.lash_material())
        self.assertIsNotNone(off.eye_material())
        # the section keeps the exporters' own skin
        self.assertIsNone(
            Character(material={"eyes": {}, "lashes": {}}).skin_material()
        )
        reference = LashMaterial.from_dict({"mesh": "c.lashes.safetensors"})
        self.assertEqual(reference.source, "reference")
        with self.assertRaises(ValueError):
            LashMaterial.from_dict({"source": "reference"})
        check_material({"lashes": {"color": [0.02, 0.02, 0.02], "length_mm": 9}})
        for bad in (
            {"colour": [0, 0, 0]},
            {"source": "mink"},
            {"length_mm": 40},
            {"lower": "yes"},
            {"mesh": 3},
            [],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_material({"lashes": bad})
        card = Character(material={"eyes": {}, "lashes": {"curl": 0.3}})
        self.assertEqual(Character.from_dict(card.to_dict()), card)

    def test_texture(self):
        image = generic_texture(LashMaterial())
        self.assertEqual(image.shape, (128, 1024, 4))
        alpha = image[..., 3] / 255.0
        self.assertTrue(0.03 < alpha.mean() < 0.5)
        # dense at the roots (the bottom rows), gone at the tips (the top row)
        self.assertGreater(alpha[-8:].mean(), 5 * alpha[:8].mean())
        self.assertLess(alpha[0].max(), 0.2)


class TestLashMesh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Character().build_model(dtype=torch.float64)
        labels = list(cls.model.facial_action_labels)
        names = ("eyeBlinkLeft", "eyeLookOutLeft", "eyeBlinkRight")
        actions = torch.zeros(len(names) + 1, len(labels), dtype=torch.float64)
        for k, name in enumerate(names):
            actions[k + 1, labels.index(name)] = 1.0
        with torch.no_grad():
            out = cls.model(facial_actions=actions)
        cls.rest = out["rest_vertices"][0].numpy()
        cls.targets = {
            name: out["rest_vertices"][k + 1].numpy() - cls.rest
            for k, name in enumerate(names)
        }
        cls.eyed = eyed_mesh(
            cls.model,
            None,
            cls.rest,
            EyeMaterial(),
            "gltf",
            lashes=(LashMaterial(), None),
        )

    def _lashes(self):
        start = self.eyed.base_count
        out = []
        for p in self.eyed.parts:
            if p.name == "lashes":
                out.append((p, slice(start, start + len(p.rest))))
            start += len(p.rest)
        return out

    def test_parts(self):
        lashes = self._lashes()
        self.assertEqual(len(lashes), 4)  # each eye, each lid
        self.assertEqual({p.side for p, _ in lashes}, {"L", "R"})
        x = self.eyed.vertices(self.rest)
        for p, rows in lashes:
            np.testing.assert_allclose(x[rows], p.rest, atol=1e-12)
            # the roots lie at the lid margin, within 1.5 mm
            near = np.linalg.norm(
                p.rest[:, None] - self.rest[p.follow.rows][None], axis=2
            ).min(1)
            self.assertLess(near.min(), 0.0015)
        # the eyes mirror each other
        left = [p for p, _ in lashes if p.side == "L"]
        right = [p for p, _ in lashes if p.side == "R"]
        for a, b in zip(left, right):
            np.testing.assert_allclose(
                np.sort(a.rest[:, 0]), np.sort(-b.rest[:, 0]), atol=2e-3
            )

    def test_blink(self):
        d = self.targets["eyeBlinkLeft"]
        moved = self.eyed.vertices(self.rest + d)
        offsets = self.eyed.targets(d[None])[0]
        np.testing.assert_allclose(
            self.eyed.vertices(self.rest) + offsets, moved, atol=1e-12
        )
        for p, rows in self._lashes():
            shift = np.linalg.norm(offsets[rows], axis=1) * 1000
            upper = p.rest[:, 2].mean() > self.eyed.frames[p.side].centre[2]
            if p.side == "R":
                self.assertLess(shift.max(), 1e-6)  # the other eye stays open
            elif upper:
                # the upper lashes come down with the lid, and keep their shape: the
                # distances between points within 3 mm of each other change by less than
                # 1.5 mm for 95 % of the pairs, and by at most 3 mm where the lid folds
                self.assertGreater(np.median(shift), 5.0)
                a, b = p.rest, moved[rows]
                da = np.linalg.norm(a[:, None] - a[None], axis=2)
                db = np.linalg.norm(b[:, None] - b[None], axis=2)
                change = np.abs(da - db)[da < 0.003]
                self.assertLess(np.percentile(change, 95), 0.0015)
                self.assertLess(change.max(), 0.003)
            else:
                self.assertLess(np.median(shift), 1.0)

    def test_closed_eye(self):
        # a closed eye shows its upper lashes along the lid line, pointing down and out of
        # the lid, with either eye's blink; half way, they point forward
        import trimesh

        skin_tris = self.eyed.triangles[: int(self.eyed.kept.sum())]
        for action, side in (("eyeBlinkLeft", "L"), ("eyeBlinkRight", "R")):
            for weight in (0.5, 1.0):
                with self.subTest(action=action, weight=weight):
                    x = self.rest + weight * self.targets[action]
                    moved = self.eyed.vertices(x)
                    skin = trimesh.Trimesh(x, skin_tris, process=False)
                    frame = self.eyed.frames[side]
                    for p, rows in self._lashes():
                        upper = p.rest[:, 2].mean() > frame.centre[2]
                        if p.side != side or not upper:
                            continue
                        local = frame.to_local(moved[rows])
                        tips, roots = p.uvs[:, 1] > 0.9, p.uvs[:, 1] < 0.1
                        drop = np.median(local[tips, 1]) - np.median(local[roots, 1])
                        ahead = np.median(local[tips, 2]) - np.median(local[roots, 2])
                        self.assertGreater(ahead, 0.15)  # out, in front of the lid
                        if weight == 1.0:
                            self.assertLess(drop, -0.1)  # down
                        # the outer half of each lash lies outside the skin
                        outer = moved[rows][p.uvs[:, 1] > 0.5]
                        cp, _, tid = trimesh.proximity.closest_point(skin, outer)
                        side_of = ((outer - cp) * skin.face_normals[tid]).sum(1)
                        self.assertGreater((side_of > 0).mean(), 0.95)

    def test_eye_turn(self):
        # a turn of the eye leaves the lashes on their lids (the lids barely move)
        offsets = self.eyed.targets(self.targets["eyeLookOutLeft"][None])[0]
        for p, rows in self._lashes():
            self.assertLess(np.linalg.norm(offsets[rows], axis=1).max(), 0.003)

    def test_skin_weights(self):
        weights = self.model.vertex_bone_weights.double().numpy()
        indices = self.model.vertex_bone_indices.numpy()
        w, i = self.eyed.skin_weights(weights, indices)
        head = list(self.model.bone_labels).index("head")
        for p, rows in self._lashes():
            np.testing.assert_allclose(w[rows].sum(1), 1.0, atol=1e-9)
            # the lashes take the weights of their lid's margin, mostly the head's
            self.assertTrue((i[rows][:, 0] == head).all() or (w[rows] > 0).any())
            self.assertGreater(float((w[rows] * (i[rows] == head)).sum(1).mean()), 0.5)


class TestReferenceLashes(unittest.TestCase):
    def test_stacked_textures_and_rest_frame(self):
        from opensculptboy.fit.handoff import HandoffPart

        def card(z, width):
            v = np.array(
                [[0.0, -0.07, z], [0.01, -0.07, z], [0.0, -0.075, z + 0.004]], float
            )
            uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            image = np.zeros((32, width, 4), np.uint8)
            image[..., 3] = 200
            return v, uv, image

        v1, uv1, im1 = card(1.52, 64)
        v2, uv2, im2 = card(1.50, 32)
        parts = [
            HandoffPart("upper", "lashes", v1, np.array([[0, 1, 2]]), uv1, im1),
            HandoffPart("lower", "lashes", v2, np.array([[0, 1, 2]]), uv2, im2),
            HandoffPart("skin", "skin", v1, np.array([[0, 1, 2]])),
        ]
        rng = np.random.default_rng(1)
        rest = rng.normal(size=(40, 3))
        angle = 0.1
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        posed = rest @ R.T + [0.0, 0.01, 0.02]
        data, atlas, values = reference_lashes(parts, posed, rest, np.arange(40))
        self.assertEqual(atlas.shape, (96, 64, 4))  # 32 + the second scaled to 64 wide
        self.assertEqual(len(data.points), 6)
        self.assertEqual(list(data.lid), [UPPER] * 3 + [LOWER] * 3)
        # the points come back to the rest frame by the inverse motion
        np.testing.assert_allclose(
            data.points[:3], (v1 - [0.0, 0.01, 0.02]) @ R, atol=1e-9
        )
        # the first part's texture is the top third of the atlas (OBJ v from the bottom)
        np.testing.assert_allclose(
            data.uvs[:3, 1], [1.0, 1.0, 1.0 - 32 / 96], atol=1e-9
        )
        np.testing.assert_allclose(data.uvs[3:, 1], [2 / 3, 2 / 3, 0.0], atol=1e-9)
        self.assertEqual(values, {})
        self.assertIsNone(reference_lashes(parts[2:], posed, rest, np.arange(40)))

    def test_file(self):
        data = LashData(
            np.zeros((3, 3)),
            np.tile([0.0, -1.0, 0.0], (3, 1)),
            np.zeros((3, 2)),
            [[0, 1, 2]],
            [0, 0, 0],
        )
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "c.lashes.safetensors"
            data.save(path)
            (pathlib.Path(folder) / "c.lashes.png").write_bytes(b"png")
            settings = LashMaterial.from_dict(
                {"mesh": path.name, "base_color": "c.lashes.png"}
            )
            back, texture = card_lashes(settings, folder)
            np.testing.assert_array_equal(back.triangles, data.triangles)
            self.assertEqual(texture, b"png")
            self.assertEqual(card_lashes(LashMaterial(), folder), (None, None))


class TestLashFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.model = Character().build_model(dtype=torch.float64)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _materials(self, glb):
        g = glb.json
        return {
            g["materials"][p["material"]]["name"]: p
            for p in g["meshes"][0]["primitives"]
        }

    def test_glb(self):
        path = self.folder / "lashes.glb"
        card = Character(material={"eyes": {"cornea": False}})
        export_glb(path, card, self.model)
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        g = glb.json
        material = next(m for m in g["materials"] if m["name"] == "lashes")
        self.assertEqual(material["alphaMode"], "BLEND")
        self.assertTrue(material["doubleSided"])
        self.assertEqual(material["extras"]["opensculptboy_eye"]["role"], "lashes")
        texture = g["textures"][
            material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        ]
        image = Image.open(io.BytesIO(glb.image_bytes(texture["source"])))
        self.assertEqual(image.mode, "RGBA")
        extras = g["scenes"][0]["extras"]["opensculptboy"]["eyes"]["lashes"]
        self.assertEqual(extras["source"], "generic")
        # the lashes of the file close with the lid of its own eyeBlinkLeft target
        names = g["meshes"][0]["extras"]["targetNames"]
        prim = self._materials(glb)["lashes"]
        rows = np.unique(glb.accessor(prim["indices"]))
        open_ = evaluate_nodes(glb)
        weights = [0.0] * len(names)
        weights[names.index("eyeBlinkLeft")] = 1.0
        closed = evaluate_nodes(glb, weights=weights)
        node = next(iter(open_))
        a = open_[node][0].positions[rows]
        b = closed[node][0].positions[rows]
        moved = np.linalg.norm(b - a, axis=1)
        left = a[:, 0] > 0
        self.assertGreater(np.percentile(moved[left], 75), 0.005)
        self.assertLess(moved[~left].max(), 1e-6)
        # "lashes": false leaves them out
        bare = self.folder / "bare.glb"
        export_glb(
            bare,
            Character(material={"eyes": {"cornea": False}, "lashes": False}),
            self.model,
        )
        self.assertEqual(
            sorted(self._materials(GLB(bare))), ["conjunctiva", "eyes", "skin"]
        )

    def test_vrm(self):
        from opensculptboy.export.vrm import export_vrm

        for version in ("1.0", "0.x"):
            with self.subTest(version=version):
                path = self.folder / f"lashes{version}.vrm"
                export_vrm(
                    path,
                    Character(material={"eyes": {}}),
                    self.model,
                    version=version,
                    author="tests",
                    refine_weights=False,
                    budget="strict",
                )
                g = GLB(path).json
                material = next(m for m in g["materials"] if m["name"] == "lashes")
                self.assertEqual(material["alphaMode"], "BLEND")
                self.assertTrue(material["doubleSided"])


if __name__ == "__main__":
    unittest.main()

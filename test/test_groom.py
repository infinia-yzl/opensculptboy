# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the grooms of ``opensculptboy.groom``: the hair and the brows of a synthetic handoff
read with their skins, vertex colours and alpha; the carry into a card's rest frame through the
fit's correspondence (rigid motion, the head's residual and the offset to the fitted surface);
the hair's skin on Anny's bones and its chains; the groom files and the card's sections; the GLB
file (the hair as a mesh of its own with COLOR_0 and its spring chain, the brows on the skin
under them through its morph targets); the VRM files of both versions (MToon materials, spring
bones, brows); and the page's blink targets against the GLB file's.
"""

import pathlib
import tempfile
import unittest
import warnings

import numpy as np
import torch

from opensculptboy import Character, export_glb
from opensculptboy.fit.handoff import load_handoff
from opensculptboy.groom.data import GroomData
from opensculptboy.groom.reference import Carrier, anny_skin, reference_groom
from opensculptboy.material import check_material
from test.gltf_reader import GLB, evaluate_nodes, structure_errors
from test.groom_fixtures import card_grooms, groom_handoff

C = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # Anny -> glTF


def _turn(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class TestGroomHandoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = groom_handoff(cls.tmp.name)
        cls.handoff = load_handoff(cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_parts(self):
        roles = sorted({p.role for p in self.handoff.parts})
        self.assertEqual(roles, ["brows", "hair", "skin"])
        hair = [p for p in self.handoff.parts if p.role == "hair"]
        self.assertEqual(sorted(p.material for p in hair), ["hair_strands", "hair_cap"])
        cards = next(p for p in hair if p.material == "hair_strands")
        self.assertEqual(cards.alpha_mode, "BLEND")
        self.assertTrue(cards.double_sided)
        self.assertEqual(cards.colors.shape, (len(cards.vertices), 4))
        self.assertEqual(cards.base_color.shape[-1], 4)
        self.assertIn("HairChain_0", cards.joints)
        self.assertEqual(cards.joint_parents["HairChain_1"], "HairChain_0")
        # the joints in Anny's frame: glTF (x, y, z) -> (x, -z, y)
        np.testing.assert_allclose(
            cards.joint_positions["HeadJoint"], [0.0, 0.0, 1.55], atol=1e-6
        )
        brows = [p for p in self.handoff.parts if p.role == "brows"]
        self.assertEqual(len(brows), 1)
        self.assertEqual(brows[0].alpha_mode, "BLEND")
        # the hair and the brows stay out of the fit
        self.assertTrue(
            all(p.material == "skin_material" for p in self.handoff.skin_parts())
        )

    def test_reference_keeps_them(self):
        from opensculptboy.fit.reference import Reference

        reference = Reference.from_handoff(self.handoff)
        self.assertEqual(
            sorted(p.role for p in reference.groom_parts), ["brows", "hair", "hair"]
        )
        moved = reference.transformed(_turn(0.3), np.array([0.1, 0.0, 0.0]))
        p = moved.groom_parts[0]
        np.testing.assert_allclose(
            p.vertices, reference.groom_parts[0].vertices @ _turn(0.3).T + [0.1, 0, 0]
        )
        head = p.joint_positions["HeadJoint"]
        np.testing.assert_allclose(head, _turn(0.3) @ [0, 0, 1.55] + [0.1, 0, 0])


class TestCarry(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        d = rng.standard_normal((400, 3))
        self.posed = 0.09 * d / np.linalg.norm(d, axis=1, keepdims=True)
        self.R, self.t = _turn(0.4), np.array([0.02, -0.01, 1.5])
        self.rest = self.posed @ self.R.T + self.t
        self.head = np.arange(400)

    def test_rigid(self):
        points = np.array([[0.0, 0.0, 0.095], [0.0, 0.1, 0.0]])
        carried = Carrier(self.posed, self.rest, self.head).points(points)
        np.testing.assert_allclose(carried, points @ self.R.T + self.t, atol=1e-9)

    def test_offset_to_the_fitted_surface(self):
        # the reference's skin lies 1 mm inside the fitted head: a point 0.5 mm over the
        # reference's skin lands 0.5 mm over the fitted head
        class Skin:
            def closest(self, x):
                return (0.089 * x / np.linalg.norm(x, axis=1, keepdims=True),)

        point = np.array([[0.0, 0.0, 0.0895]])
        carried = Carrier(self.posed, self.rest, self.head, Skin()).points(point)
        local = (carried - self.t) @ self.R
        self.assertAlmostEqual(float(np.linalg.norm(local)), 0.0905, delta=1e-4)


class TestReferenceGroom(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.handoff = load_handoff(groom_handoff(cls.tmp.name))
        skin = cls.handoff.skin_parts()[0]
        cls.posed = skin.vertices
        cls.R, cls.t = _turn(0.2), np.array([0.0, 0.01, -0.02])
        cls.rest = cls.posed @ cls.R.T + cls.t
        cls.head = np.arange(len(cls.posed))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_hair(self):
        labels = ["root", "neck01", "head"]
        heads = np.array([[0, 0, 0], [0, 0.01, 1.47], [0, 0, 1.55]]) @ self.R.T + self.t
        data, textures = reference_groom(
            self.handoff.parts,
            "hair",
            self.posed,
            self.rest,
            self.head,
            None,
            heads,
            labels,
        )
        self.assertEqual([m.name for m in data.materials], ["hair_cap", "hair_strands"])
        self.assertEqual(data.materials[1].alpha_mode, "BLEND")
        self.assertIn("base_color", textures["hair_strands"])
        # the points moved by the rigid motion of the head
        cap = next(p for p in self.handoff.parts if p.material == "hair_cap")
        np.testing.assert_allclose(
            data.points[: len(cap.vertices)],
            cap.vertices @ self.R.T + self.t,
            atol=1e-9,
        )
        # the blended triangles run from the inside out
        centre = self.rest.mean(0)
        tri = data.triangles[data.material == 1]
        d = np.linalg.norm(data.points[tri].mean(1) - centre, axis=1)
        self.assertTrue((np.diff(d) >= -1e-12).all())
        # the skin: the head, the neck's ring, and the chain's weight on the head
        self.assertEqual(set(data.bones), {"head", "neck01"})
        self.assertEqual(len(data.chains), 1)
        np.testing.assert_allclose(
            data.chains[0][0], self.R @ [0, 0.085, 1.64] + self.t, atol=1e-6
        )
        self.assertEqual(len(data.chains[0]), 3)  # HairChain_0, HairChain_1 and a tail
        self.assertGreater(data.chain_share.max(), 0.5)
        self.assertEqual(data.chain_share[: len(cap.vertices)].max(), 0.0)
        self.assertEqual(data.colors.shape, (len(data.points), 4))

    def test_brows(self):
        data, textures = reference_groom(
            self.handoff.parts, "brows", self.posed, self.rest, self.head
        )
        self.assertEqual(data.kind, "brows")
        self.assertEqual(len(data.points), 16)
        self.assertIn("brow_strands", textures)

    def test_none(self):
        skin = self.handoff.skin_parts()
        self.assertIsNone(
            reference_groom(skin, "hair", self.posed, self.rest, self.head)
        )

    def test_anny_skin_without_joints(self):
        part = next(p for p in self.handoff.parts if p.role == "brows")
        import dataclasses

        bare = dataclasses.replace(part, joints=None, joint_slots=None)
        bones, slots, weights, chains, share = anny_skin(
            [bare], np.eye(3), np.zeros(3), np.zeros((1, 3)), ["head"]
        )
        self.assertEqual(bones, ["head"])
        self.assertEqual(chains, [])
        np.testing.assert_allclose(weights.sum(1), 1.0)


class TestGroomFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.model = Character().build_model(dtype=torch.float64)
        with torch.no_grad():
            rest = cls.model()["rest_vertices"][0].numpy()
        cls.sections, cls.hair, cls.brows, _ = card_grooms(cls.model, rest, cls.folder)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_round_trip(self):
        back = GroomData.load(self.folder / "card.hair.safetensors")
        np.testing.assert_allclose(back.points, self.hair.points)
        np.testing.assert_array_equal(back.triangles, self.hair.triangles)
        self.assertEqual(back.materials, self.hair.materials)
        self.assertEqual(len(back.chains), 1)
        np.testing.assert_allclose(back.chains[0], self.hair.chains[0], atol=1e-7)
        np.testing.assert_allclose(back.colors, self.hair.colors)
        self.assertIsNone(GroomData.load(self.folder / "card.brows.safetensors").colors)

    def test_sections(self):
        check_material(dict(self.sections, eyes={}))
        card = Character(material=dict(self.sections, eyes={}))
        self.assertIsNone(card.skin_material())
        self.assertEqual(card.groom_section("hair").mesh, "card.hair.safetensors")
        # the brows come with the eyes
        self.assertIsNone(Character(material=self.sections).groom_section("brows"))
        self.assertIsNone(
            Character(material={"hair": False, "eyes": {}}).groom_section("hair")
        )
        for bad in (
            {"hair": {"mesh": 3}},
            {"hair": {"textures": {"m": {"albedo": "x.png"}}}},
            {"brows": {"source": "generic"}},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    check_material(bad)

    def test_bake_section(self):
        from opensculptboy.texture.bake import BakedTextures

        baked = BakedTextures(None, np.zeros((4, 4, 3), np.uint8), np.ones((4, 4)), 1.0)
        material = baked.material(
            {
                "hair": "c.hair.safetensors",
                "hair_textures": {"m": {"base_color": "a.png"}},
            }
        )
        self.assertEqual(
            material["hair"],
            {
                "source": "reference",
                "mesh": "c.hair.safetensors",
                "textures": {"m": {"base_color": "a.png"}},
            },
        )


class TestGroomExports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.model = Character().build_model(dtype=torch.float64)
        with torch.no_grad():
            rest = cls.model()["rest_vertices"][0].numpy()
        cls.sections, cls.hair, cls.brows, _ = card_grooms(cls.model, rest, cls.folder)
        cls.card = Character(material=dict(cls.sections, eyes={"cornea": False}))
        cls.path = cls.folder / "groom.glb"
        cls.summary = export_glb(cls.path, cls.card, cls.model, detail_dir=cls.folder)
        cls.glb = GLB(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _node(self, name):
        g = self.glb.json
        return next(i for i, n in enumerate(g["nodes"]) if n.get("name") == name)

    def test_structure(self):
        self.assertEqual(structure_errors(self.glb), [])
        extras = self.glb.json["scenes"][0]["extras"]["opensculptboy"]
        self.assertEqual(extras["hair"]["kind"], "hair")
        self.assertTrue(extras["hair"]["vertex_colors"])
        self.assertEqual(len(extras["hair"]["spring_chains"]), 1)
        self.assertEqual(extras["brows"]["parts"], 2)
        self.assertIn("springs", extras)

    def test_hair_node(self):
        g = self.glb.json
        node = g["nodes"][self._node("hair")]
        self.assertEqual(node["extras"]["opensculptboy_groom"]["kind"], "hair")
        prims = g["meshes"][node["mesh"]]["primitives"]
        self.assertIn("COLOR_0", prims[0]["attributes"])
        names = [g["materials"][p["material"]]["name"] for p in prims]
        self.assertEqual(names, ["hair_cap", "hair_strands"])
        blend = g["materials"][prims[1]["material"]]
        self.assertEqual(blend["alphaMode"], "BLEND")
        self.assertTrue(blend["doubleSided"])
        self.assertEqual(blend["extras"]["opensculptboy_groom"]["kind"], "hair")
        # the file's hair at rest is the groom's points, in glTF's frame on the floor
        evaluated = evaluate_nodes(self.glb)[self._node("hair")][0].positions
        body = evaluate_nodes(self.glb)[self._node(self.card.name)][0].positions
        lift = body[:, 1].min()
        self.assertAlmostEqual(lift, 0.0, places=5)
        expected = self.hair.points @ C.T
        expected[:, 1] -= (self.model_low() @ C.T)[1]
        np.testing.assert_allclose(evaluated, expected, atol=2e-5)

    def model_low(self):
        with torch.no_grad():
            rest = self.model()["rest_vertices"][0].numpy()
        return rest[np.argmin(rest[:, 2])]

    def test_chain_swings_the_cards(self):
        # the chain's first joint turns the cards' lower points and leaves the cap
        g = self.glb.json
        spring = next(
            i
            for i, n in enumerate(g["nodes"])
            if "opensculptboy_spring" in n.get("extras", {})
        )
        prim = g["meshes"][g["nodes"][self._node("hair")]["mesh"]]["primitives"][1]
        joints = self.glb.accessor(prim["attributes"]["JOINTS_0"])
        weights = self.glb.accessor(prim["attributes"]["WEIGHTS_0"])
        skin = g["skins"][0]["joints"]
        on_chain = np.isin(np.asarray(skin)[joints], [spring]) & (weights > 0)
        self.assertTrue(on_chain.any())
        cap = g["meshes"][g["nodes"][self._node("hair")]["mesh"]]["primitives"][0]
        rows = np.unique(self.glb.accessor(cap["indices"]))
        cap_on_chain = np.isin(np.asarray(skin)[joints[rows]], [spring]) & (
            weights[rows] > 0
        )
        self.assertFalse(cap_on_chain.any())

    def test_brows_follow_the_skin(self):
        g = self.glb.json
        mesh = g["meshes"][0]
        names = mesh["extras"]["targetNames"]
        prim = next(
            p
            for p in mesh["primitives"]
            if g["materials"][p["material"]]["name"].startswith("brows:")
        )
        material = g["materials"][prim["material"]]
        self.assertEqual(material["alphaMode"], "BLEND")
        rows = np.unique(self.glb.accessor(prim["indices"]))
        node = self._node(self.card.name)
        rest = evaluate_nodes(self.glb)[node][0].positions[rows]
        weights = [0.0] * len(names)
        weights[names.index("browInnerUp")] = 1.0
        up = evaluate_nodes(self.glb, weights=weights)[node][0].positions[rows]
        moved = np.linalg.norm(up - rest, axis=1)
        self.assertGreater(moved.max(), 0.001)
        # the brows' points at rest are the groom's
        lift = (self.model_low() @ C.T)[1]
        expected = self.brows.points @ C.T
        expected[:, 1] -= lift
        d = np.linalg.norm(rest[:, None] - expected[None], axis=2).min(1)
        self.assertLess(d.max(), 2e-5)

    def test_vrm(self):
        from opensculptboy.export.vrm import export_vrm

        for version in ("1.0", "0.x"):
            with self.subTest(version=version):
                path = self.folder / f"groom{version}.vrm"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    export_vrm(
                        path,
                        self.card,
                        self.model,
                        version=version,
                        author="tests",
                        refine_weights=False,
                        budget="warn",
                        detail_dir=self.folder,
                    )
                glb = GLB(path)
                g = glb.json
                self.assertEqual(structure_errors(glb), [])
                names = [m["name"] for m in g["materials"]]
                self.assertIn("brows", names)
                self.assertIn("hair:hair_strands", names)
                hair = g["materials"][names.index("hair:hair_strands")]
                self.assertEqual(hair["alphaMode"], "BLEND")
                if version == "1.0":
                    mtoon = hair["extensions"]["VRMC_materials_mtoon"]
                    self.assertTrue(mtoon["transparentWithZWrite"])
                    springs = g["extensions"]["VRMC_springBone"]["springs"]
                    self.assertTrue(any("hair" in s.get("name", "") for s in springs))
                else:
                    groups = g["extensions"]["VRM"]["secondaryAnimation"]["boneGroups"]
                    self.assertTrue(any("hair" in s.get("comment", "") for s in groups))
                # the hair's base colour holds its mean vertex colour
                factor = hair["pbrMetallicRoughness"]["baseColorFactor"]
                self.assertLess(factor[1], 0.5)


class TestPageBlinks(unittest.TestCase):
    def test_against_the_glb(self):
        from opensculptboy.eyes import page

        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            model = Character().build_model()
            with torch.no_grad():
                rest = model()["rest_vertices"][0].double().numpy()
            sections, _, brows, _ = card_grooms(model, rest, folder)
            card = Character(
                material={"eyes": {"cornea": False}, "brows": sections["brows"]}
            )
            blinks = page.page_blinks(card, folder)
            self.assertEqual(blinks["level"], 0)
            names = blinks["names"]
            for name in (
                "eyeBlinkLeft",
                "eyeBlinkRight",
                "eyeBlinkLeft.inbetween",
                "eyeSquintLeft",
            ):
                self.assertIn(name, names)
            self.assertGreater(len(blinks["rows"]), 100)
            self.assertEqual(blinks["lashes"].shape[0], len(names))
            self.assertEqual(blinks["brows"].shape[1], len(brows.points))
            path = folder / "blinks.safetensors"
            page.save(path, blinks)
            back = page.load(path)
            np.testing.assert_allclose(back["offsets"], blinks["offsets"], atol=1e-6)
            # the GLB file's targets at the skin's vertices of the same rows
            glb_path = folder / "blinks.glb"
            export_glb(glb_path, card, model, detail_dir=folder)
            glb = GLB(glb_path)
            g = glb.json
            mesh = g["meshes"][0]
            prim = mesh["primitives"][0]
            targets = mesh["extras"]["targetNames"]
            anny = glb.accessor(prim["attributes"]["_ANNY_VERTEX"]).astype(int)
            skin_rows = np.unique(glb.accessor(prim["indices"]))
            lookup = {int(r): k for k, r in enumerate(blinks["rows"])}
            for name in ("eyeBlinkLeft", "eyeBlinkLeft.inbetween", "eyeSquintRight"):
                with self.subTest(name=name):
                    delta = glb.accessor(
                        prim["targets"][targets.index(name)]["POSITION"]
                    )
                    p = names.index(name)
                    checked = 0
                    for v in skin_rows:
                        k = lookup.get(int(anny[v]))
                        if k is None:
                            continue
                        np.testing.assert_allclose(
                            delta[v], C @ blinks["offsets"][p, k], atol=2e-6
                        )
                        checked += 1
                    self.assertGreater(checked, 50)


if __name__ == "__main__":
    unittest.main()

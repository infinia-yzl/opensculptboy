# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the wearables (``opensculptboy.wearables``, docs/wearables.md) with synthetic
garments only (``test.wearable_fixtures``): the package format and its provenance guard, the
surface and rigid bindings, the fit on random bodies (the E4 gate: no covered body vertex more
than 2 mm outside the garment at rest), the push-out, the card's outfits with slot replacement
and ``hides`` rules, the GLB, wardrobe and VRM files, the command, and reference wearables
built from a synthetic reference character with a clothing part.
"""

import contextlib
import dataclasses
import io
import json
import pathlib
import re
import tempfile
import unittest

import numpy as np

from opensculptboy import Character, export_glb
from opensculptboy.wearables import (
    PRIVATE,
    Garment,
    ProvenanceError,
    WornPiece,
    card_body,
    dress,
    evaluate,
    read_wearable,
    worn_pieces,
    write_wearable,
)
from opensculptboy.wearables.cover import coverage_gaps
from opensculptboy.wearables.export import Dressing
from test.gltf_reader import GLB, evaluate_nodes, structure_errors
from test.wearable_fixtures import (
    bump_normal_map,
    cape,
    circlet,
    cover_of,
    template_body,
    template_model,
    tunic,
    write_outfit,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRIVATE_MARK = re.compile(rb'"provenance"\s*:\s*"private"')


def _worn(*wearables):
    return [WornPiece(w.name, w.name, None, "", w, True) for w in wearables]


class TestNoPrivatePackages(unittest.TestCase):
    """No file of the public tree declares the private provenance."""

    def test_source_and_test_data_hold_no_private_package(self):
        found = []
        for folder in (ROOT / "src", ROOT / "test" / "data"):
            for path in folder.rglob("*"):
                if path.is_file() and PRIVATE_MARK.search(path.read_bytes()):
                    found.append(str(path.relative_to(ROOT)))
        self.assertEqual(found, [], "private wearable packages in the public tree")


class TestFormat(unittest.TestCase):
    def test_package_round_trip(self):
        w = tunic()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_wearable(pathlib.Path(tmp) / "tunic", w)
            self.assertEqual(path.name, "tunic.wearable.json")
            for suffix in (".glb", ".binding.safetensors"):
                self.assertTrue((pathlib.Path(tmp) / f"tunic{suffix}").exists())
            back = read_wearable(path)
            self.assertEqual(
                GLB(pathlib.Path(tmp) / "tunic.glb").json["asset"]["version"], "2.0"
            )
        self.assertEqual(back.garment.groups, ["front", "back"])
        for g in range(2):  # the file holds the triangles group by group
            mine = w.garment.triangles[w.garment.triangle_group == g]
            theirs = back.garment.triangles[back.garment.triangle_group == g]
            np.testing.assert_array_equal(np.sort(theirs, 0), np.sort(mine, 0))
        np.testing.assert_allclose(back.garment.vertices, w.garment.vertices, atol=1e-6)
        np.testing.assert_array_equal(back.binding.corners, w.binding.corners)
        np.testing.assert_array_equal(back.binding.offset, w.binding.offset)
        np.testing.assert_array_equal(back.covers, w.covers)
        self.assertEqual(sorted(back.group_covers), ["back", "front"])
        np.testing.assert_array_equal(back.group_covers["back"], w.group_covers["back"])
        self.assertEqual(back.licence["spdx"], "CC0-1.0")
        self.assertEqual(back.provenance, "public")

    def test_public_package_needs_a_licence(self):
        w = circlet()
        w.licence = {}
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            write_wearable(pathlib.Path(tmp) / "circlet", w)

    def test_garment_without_triangles(self):
        w = circlet()
        w.garment = Garment(w.garment.vertices, np.zeros((0, 3)))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "at least one triangle"):
                write_wearable(pathlib.Path(tmp) / "circlet", w)
            self.assertFalse((pathlib.Path(tmp) / "circlet.glb").exists())

    def test_private_package_never_lands_in_the_repository(self):
        w = circlet()
        w.provenance = PRIVATE
        for folder in (ROOT / "src" / "opensculptboy", ROOT / "test" / "data", ROOT):
            with self.subTest(folder.name), self.assertRaises(ProvenanceError):
                write_wearable(folder / "circlet.wearable.json", w)
            self.assertFalse((folder / "circlet.wearable.json").exists())
        with tempfile.TemporaryDirectory() as tmp:  # outside the tree it is written
            path = write_wearable(pathlib.Path(tmp) / "circlet", w)
            self.assertTrue(PRIVATE_MARK.search(path.read_bytes()))
            self.assertTrue(
                PRIVATE_MARK.search(path.with_name("circlet.glb").read_bytes())
            )


class TestBinding(unittest.TestCase):
    def test_garments_return_to_their_vertices_on_the_template(self):
        body = template_body()
        for make in (tunic, circlet, cape):
            with self.subTest(make.__name__):
                w = make()
                vertices, normals = evaluate(w.binding, body)
                np.testing.assert_allclose(
                    vertices, w.garment.vertices, rtol=0, atol=1e-12
                )
                np.testing.assert_allclose(
                    normals, w.garment.normals, rtol=0, atol=1e-9
                )

    def test_ten_random_bodies_keep_the_covered_skin_inside(self):
        """The E4 gate: no covered body vertex lies more than 2 mm outside the tunic."""
        rng = np.random.default_rng(4)
        model = template_model()
        w = tunic()
        for k in range(10):
            phenotype = {n: float(rng.uniform(0, 1)) for n in model.phenotype_labels}
            with self.subTest(k=k, phenotype=phenotype):
                body = card_body(Character(phenotype=phenotype), model)
                piece = dress(body, _worn(w))[0]
                gaps = coverage_gaps(body, piece, cover_of(w), tolerance=0.002)
                self.assertGreater(gaps["covered"], 300)
                self.assertEqual(gaps["outside"], 0, gaps)

    def test_push_out_keeps_the_layers_apart(self):
        """A tunic 12 mm off the skin over a rigid plate 8 mm off it (a lower layer, on the
        chest bone): on a thinner body the skin draws the tunic into the plate, and the
        push-out brings it back over the plate by the 1 mm gap."""
        import dataclasses

        from opensculptboy.wearables import bind_rigid
        from opensculptboy.wearables.geometry import Surface

        body = template_body()
        plate = tunic(offset=0.008)
        plate = dataclasses.replace(
            plate,
            name="plate",
            layer=40,
            weights="rigid",
            binding=bind_rigid(plate.garment, body, "spine03"),
        )
        coat = tunic(offset=0.012)
        thin = card_body(
            Character(phenotype={"weight": 0.0, "muscle": 0.0}), template_model()
        )
        worn = [WornPiece(w.name, w.name, None, "", w, True) for w in (plate, coat)]
        plain = dress(thin, worn, push=False)
        pushed = dress(thin, worn)
        under = Surface(pushed[0].vertices, plate.garment.triangles, pushed[0].normals)
        before = under.signed_distance(plain[1].vertices)
        after = under.signed_distance(pushed[1].vertices)
        near = before[1].distance < 0.01
        self.assertGreater(
            int((before[0][near] < 0).sum()), 50, "the test needs a sunk coat"
        )
        self.assertGreater(pushed[1].pushed, 50)
        near = after[1].distance < 0.01
        self.assertGreater(float(np.percentile(after[0][near], 1)), 0.0008)
        self.assertGreater(float(after[0][near].min()), 0.0)

    def test_push_out_over_a_rigid_piece_on_other_bodies(self):
        """On a taller and on a shorter body the chest bone moves, and the rigid plate with
        it: the coat keeps the relation it had with the plate on the binding body (outside
        by the 1 mm gap, or by as little as it lay outside there)."""
        import dataclasses

        from opensculptboy.wearables import bind_rigid
        from opensculptboy.wearables.geometry import Surface

        body = template_body()
        plate = tunic(offset=0.008)
        plate = dataclasses.replace(
            plate,
            name="plate",
            layer=40,
            weights="rigid",
            binding=bind_rigid(plate.garment, body, "spine03"),
        )
        coat = tunic(offset=0.012)
        home = Surface(plate.garment.vertices, plate.garment.triangles)
        target = np.minimum(home.signed_distance(coat.garment.vertices)[0], 0.001)
        worn = [WornPiece(w.name, w.name, None, "", w, True) for w in (plate, coat)]
        for height in (0.0, 1.0):
            with self.subTest(height=height):
                other = card_body(
                    Character(phenotype={"height": height}), template_model()
                )
                pushed = dress(other, worn)
                under = Surface(
                    pushed[0].vertices, plate.garment.triangles, pushed[0].normals
                )
                signed, found, _ = under.signed_distance(pushed[1].vertices)
                near = found.distance < 0.01
                self.assertGreater(int(near.sum()), 50)
                self.assertGreater(float((signed - target)[near].min()), -1e-4)

    def test_weights(self):
        body = template_body()
        dressed = dress(body, _worn(tunic(), circlet()))
        for piece in dressed:
            np.testing.assert_allclose(piece.weights.sum(1), 1.0, atol=1e-9)
        head = body.bone_labels.index("head")
        self.assertTrue((dressed[1].joints[:, 0] == head).all())
        self.assertTrue((dressed[1].weights[:, 0] == 1.0).all())

    def test_see_through_materials_cover_no_skin(self):
        from opensculptboy.wearables.cover import ray_cover

        body = template_body()
        piece = dress(body, _worn(tunic()))[0]
        self.assertGreater(int(ray_cover(body, piece).sum()), 300)
        piece.garment.materials[0].alpha_mode = "MASK"  # lace: the skin may show
        self.assertEqual(int(ray_cover(body, piece).sum()), 0)

    def test_group_masks_without_a_mask_of_the_fixed_groups(self):
        """A package with group masks and no ``covers`` mask: the ray test covers the skin
        of its fixed groups, and each shown group adds its mask."""
        from opensculptboy.wearables.cover import piece_cover, ray_cover

        body = template_body()
        w = tunic()
        w.covers = None
        shins = body.vertices[:, 2] < body.vertices[:, 2].min() + 0.3
        w.group_covers["front"] = w.group_covers["front"] | shins
        piece = dress(body, _worn(w))[0]
        mask = piece_cover(body, piece)
        self.assertTrue(mask[shins].all())
        self.assertTrue(mask[ray_cover(body, piece)].all())
        piece.hidden_groups = ["front"]
        self.assertFalse(piece_cover(body, piece)[shins].any())

    def test_smoothed_weights(self):
        """Diffused weights keep their sums, join vertices at one position and blend
        neighbours alike."""
        from opensculptboy.wearables.bind import dense_body_weights, smooth_weights

        w = tunic()
        dense = dense_body_weights(w.binding, template_body())
        smooth = smooth_weights(dense, w.garment.vertices, w.garment.triangles, 40)
        np.testing.assert_allclose(smooth.sum(1), 1.0, atol=1e-9)
        t = w.garment.triangles

        def jump(x):
            return float(np.abs(x[t[:, 0]] - x[t[:, 1]]).sum(1).mean())

        self.assertLess(jump(smooth), 0.5 * jump(dense))

    def test_rigid_piece_follows_its_bone(self):
        model = template_model()
        w = circlet()
        body = card_body(Character(phenotype={"height": 1.0}), model)
        vertices, _ = evaluate(w.binding, body)
        B0 = template_body().bone_poses[body.bone_labels.index("head")]
        B1 = body.bone_poses[body.bone_labels.index("head")]
        expected = (w.garment.vertices - B0[:3, 3]) @ B0[:3, :3] @ B1[:3, :3].T + B1[
            :3, 3
        ]
        np.testing.assert_allclose(vertices, expected, atol=1e-12)


class TestWardrobe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.card = write_outfit(cls.folder)
        other = tunic()
        other.name = "vest"
        write_wearable(cls.folder / "vest.wearable.json", other)
        cls.card.outfits.append(
            {
                "id": "spare",
                "label": "Spare",
                "pieces": [{"piece": "vest", "file": "vest.wearable.json"}],
            }
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def shown(self, **kwargs):
        pieces = worn_pieces(self.card, self.folder, **kwargs)
        return {p.ref: p for p in pieces}

    def test_card_schema(self):
        data = self.card.to_dict()
        self.assertEqual(data["schema"], 4)
        self.assertEqual(Character.from_dict(data), self.card)
        self.assertEqual(Character().to_dict()["schema"], 1)
        with self.assertRaises(ValueError):
            Character.from_dict(dict(data, schema=3))
        with self.assertRaises(ValueError):
            Character.from_dict(dict(data, active_outfit="missing"))

    def test_active_outfit_and_hides(self):
        pieces = self.shown()
        self.assertEqual(
            sorted(pieces),
            ["test_outfit/cape", "test_outfit/circlet", "test_outfit/tunic"],
        )
        self.assertEqual(pieces["test_outfit/tunic"].hidden_groups, ["back"])
        # hiding the cape gives the tunic its back again
        pieces = self.shown(hide=["cape"])
        self.assertNotIn("test_outfit/cape", pieces)
        self.assertEqual(pieces["test_outfit/tunic"].hidden_groups, [])

    def test_slot_replacement_and_overrides(self):
        pieces = self.shown(show=["spare/vest"])
        self.assertIn("spare/vest", pieces)
        self.assertNotIn("test_outfit/tunic", pieces)  # the vest takes the torso slot
        self.assertIn("test_outfit/cape", pieces)
        card = Character.from_dict(
            dict(
                self.card.to_dict(), wearable_overrides={"circlet": {"enabled": False}}
            )
        )
        names = [p.ref for p in worn_pieces(card, self.folder)]
        self.assertNotIn("test_outfit/circlet", names)
        # a piece of another outfit that an override shows takes its slot as well
        card = Character.from_dict(
            dict(
                self.card.to_dict(),
                wearable_overrides={"spare/vest": {"enabled": True}},
            )
        )
        names = [p.ref for p in worn_pieces(card, self.folder)]
        self.assertIn("spare/vest", names)
        self.assertNotIn("test_outfit/tunic", names)
        self.assertEqual(worn_pieces(self.card, self.folder, outfit="none"), [])
        # a card with loose pieces alone, and no active outfit
        loose = Character(wearables=[{"id": "vest", "file": "vest.wearable.json"}])
        self.assertEqual([p.ref for p in worn_pieces(loose, self.folder)], ["vest"])
        with self.assertRaisesRegex(ValueError, "Did you mean"):
            worn_pieces(self.card, self.folder, hide=["capes"])
        every = worn_pieces(self.card, self.folder, every_outfit=True)
        self.assertEqual(len(every), 4)
        self.assertFalse(next(p for p in every if p.ref == "spare/vest").shown)


class TestDressedFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.card = write_outfit(cls.folder)
        cls.model = template_model()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def export(self, name, **dressing):
        path = self.folder / name
        summary = export_glb(
            path,
            self.card,
            self.model,
            detail_dir=self.folder,
            dressing=Dressing(**dressing),
        )
        return path, summary

    def test_dressed_glb(self):
        path, summary = self.export("dressed.glb")
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        names = [n.get("name") for n in glb.json["nodes"]]
        for ref in ("test_outfit/tunic", "test_outfit/circlet", "test_outfit/cape"):
            self.assertIn(ref, names)
        self.assertGreater(summary["removed_body_triangles"], 300)
        extras = glb.json["scenes"][0]["extras"]["opensculptboy"]["wearables"]
        tunic_extras = next(p for p in extras["pieces"] if p["piece"] == "tunic")
        self.assertEqual(tunic_extras["ref"], "test_outfit/tunic")
        # a CC0 piece is credited as any other piece with licence information
        self.assertIn(TUNIC_CREDIT, extras["credits"])
        self.assertEqual(len(extras["credits"]), 3)
        self.assertEqual(tunic_extras["hidden_groups"], ["back"])
        # the file's skinned pieces sit where the fit put them, on the floor
        dressed = dress(template_body(), worn_pieces(self.card, self.folder))
        evaluated = evaluate_nodes(glb)
        node = names_index(glb, "test_outfit/circlet")
        vertices = evaluated[node][0].positions
        C = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])
        expected = dressed[1].vertices @ C.T
        lift = np.median(vertices - expected, axis=0)
        np.testing.assert_allclose(vertices, expected + lift, atol=2e-5)
        np.testing.assert_allclose(
            lift[[0, 2]], 0.0, atol=1e-6
        )  # the ground lift alone
        _, kept = self.export("kept.glb", covered="keep")
        self.assertEqual(kept["removed_body_triangles"], 0)
        _, bare = self.export("bare.glb", outfit="none")
        self.assertEqual(bare["wearables"], [])

    def test_skin_goes_only_under_shown_pieces(self):
        """An export removes the body's skin only under the pieces it shows: a bare body
        (no outfit, or every piece hidden) keeps every body triangle, and hiding a piece
        brings back the skin that it alone covers."""

        def body_triangles(path):
            g = GLB(path).json
            body = next(n for n in g["nodes"] if "mesh" in n and "skin" in n)
            return sum(
                g["accessors"][p["indices"]]["count"] // 3
                for p in g["meshes"][body["mesh"]]["primitives"]
            )

        whole, _ = self.export("whole.glb", covered="keep")
        dressed, every = self.export("every.glb")
        no_tunic, some = self.export("no_tunic.glb", hide=("tunic",))
        hidden, none_shown = self.export(
            "hidden.glb", hide=("tunic", "circlet", "cape")
        )
        bare, no_outfit = self.export("none.glb", outfit="none")
        self.assertEqual(none_shown["removed_body_triangles"], 0)
        self.assertEqual(no_outfit["removed_body_triangles"], 0)
        self.assertEqual(body_triangles(hidden), body_triangles(whole))
        self.assertEqual(body_triangles(bare), body_triangles(whole))
        self.assertLess(some["removed_body_triangles"], every["removed_body_triangles"])
        self.assertEqual(
            body_triangles(dressed),
            body_triangles(whole) - every["removed_body_triangles"],
        )

    def test_dressed_glb_with_eight_influences(self):
        """Eight bones per vertex: the pieces hold them in two sets of four, as the body."""
        path = self.folder / "eight.glb"
        export_glb(
            path, self.card, self.model, detail_dir=self.folder, max_influences=8
        )
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        mesh = glb.json["meshes"][
            glb.json["nodes"][names_index(glb, "test_outfit/tunic")]["mesh"]
        ]
        attributes = mesh["primitives"][0]["attributes"]
        self.assertIn("JOINTS_1", attributes)
        weights = sum(glb.accessor(attributes[f"WEIGHTS_{s}"]).sum(1) for s in (0, 1))
        np.testing.assert_allclose(weights, 1.0, atol=1e-6)

    def test_skin_covered_whole(self):
        """A suit whose mask covers every skin vertex: the body's skin primitive leaves the
        GLB and VRM files (a glTF accessor holds at least one element), the eyeballs stay,
        and with nothing left of the body the export fails with a clear message."""
        from opensculptboy.export.document import png_bytes
        from opensculptboy.export.vrm import export_vrm

        suit = tunic()
        suit.name = "suit"
        suit.covers = np.ones(len(suit.covers), bool)
        write_wearable(self.folder / "suit.wearable.json", suit)
        card = Character(
            name="suited", wearables=[{"id": "suit", "file": "suit.wearable.json"}]
        )
        eyes = {"eyes": png_bytes(np.full((4, 8, 3), 40, np.uint8))}
        files = [self.folder / "suited.glb"]
        export_glb(files[0], card, self.model, textures=eyes, detail_dir=self.folder)
        for version in ("1.0", "0.x"):
            files.append(self.folder / f"suited_{version}.vrm")
            export_vrm(
                files[-1],
                card,
                author="test",
                version=version,
                textures=eyes,
                detail_dir=self.folder,
            )
        for path in files:
            with self.subTest(path.name):
                glb = GLB(path)
                self.assertEqual(structure_errors(glb), [])
                counts = [a["count"] for a in glb.json["accessors"]]
                self.assertGreater(min(counts), 0)
        with self.assertRaisesRegex(ValueError, "cover every triangle"):
            export_glb(
                self.folder / "bare.glb", card, self.model, detail_dir=self.folder
            )

    def test_piece_with_every_group_hidden_leaves_the_file(self):
        from opensculptboy.export.document import GltfDocument
        from opensculptboy.wearables.export import vrm_piece_meshes
        from opensculptboy.wearables.gltf import add_pieces

        piece = dress(template_body(), _worn(tunic()))[0]
        piece.hidden_groups = ["front", "back"]
        doc = GltfDocument("test")
        C = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])
        self.assertEqual(add_pieces(doc, [piece], 0, C), [])
        self.assertEqual(len(doc.buffers.accessors), 0)

        class Rebind:
            transforms = np.tile(np.eye(4), (len(template_body().bone_labels), 1, 1))

        self.assertEqual(vrm_piece_meshes([piece], Rebind(), C), [])

    def test_wardrobe_glb(self):
        path, _ = self.export(
            "wardrobe.glb", wardrobe=True, visibility=True, hide=("circlet",)
        )
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        self.assertIn("KHR_node_visibility", glb.json["extensionsUsed"])
        nodes = {n.get("name"): n for n in glb.json["nodes"]}
        self.assertTrue(hidden(nodes["test_outfit/circlet"]))
        self.assertFalse(hidden(nodes["test_outfit/cape"]))
        self.assertTrue(hidden(nodes["test_outfit/tunic.back"]))  # hidden by the cape
        self.assertFalse(hidden(nodes["test_outfit/tunic"]))
        scene = glb.json["scenes"][0]["nodes"]
        self.assertIn(names_index(glb, "test_outfit/circlet"), scene)
        from opensculptboy.export.blender_skin import read_wearable_visibility

        shown = read_wearable_visibility(path)
        self.assertFalse(shown["test_outfit/circlet"])
        self.assertFalse(shown["test_outfit/tunic.back"])
        self.assertTrue(shown["test_outfit/cape"])
        path, _ = self.export(
            "plain.glb", wardrobe=True, visibility=False, hide=("circlet",)
        )
        glb = GLB(path)
        self.assertNotIn("KHR_node_visibility", glb.json.get("extensionsUsed", []))
        self.assertNotIn(
            names_index(glb, "test_outfit/circlet"), glb.json["scenes"][0]["nodes"]
        )
        self.assertEqual(
            len(glb.json["scenes"]), 2
        )  # the second scene holds every piece
        for name in ("test_outfit/circlet", "test_outfit/tunic.back"):
            self.assertIn(names_index(glb, name), glb.json["scenes"][1]["nodes"])

    def test_vrm_files(self):
        from opensculptboy.export.vrm import export_vrm
        from test.test_vrm_schema import check_vrm0, check_vrm1

        for version, check in (("1.0", check_vrm1), ("0.x", check_vrm0)):
            with self.subTest(version):
                path = self.folder / f"dressed_{version}.vrm"
                summary = export_vrm(
                    path,
                    self.card,
                    author="test",
                    version=version,
                    detail_dir=self.folder,
                )
                refs = ["test_outfit/tunic", "test_outfit/circlet", "test_outfit/cape"]
                self.assertEqual(summary["wearables"], refs)
                pieces = glb_json_wearables(path)["pieces"]
                self.assertEqual([p["ref"] for p in pieces], refs)
                self.assertEqual(
                    [p["piece"] for p in pieces], ["tunic", "circlet", "cape"]
                )
                glb = GLB(path)
                self.assertEqual(structure_errors(glb), [])
                self.assertEqual(check(glb.json), [])
                names = [m["name"] for m in glb.json["meshes"]]
                self.assertIn("test_outfit/cape", names)
                if version == "1.0":
                    text = glb.json["extensions"]["VRMC_vrm"]["meta"][
                        "thirdPartyLicenses"
                    ]
                else:
                    text = glb.json["extensions"]["VRM"]["meta"]["reference"]
                self.assertIn("Wearable: " + TUNIC_CREDIT, text)

    def test_vrm_normal_maps_without_tangents(self):
        """A normal-mapped skin and piece keep their normal maps in both versions, without
        vertex tangents (test.test_vrm_three.TestThreeVrmNormalMaps draws them in three-vrm)."""
        from opensculptboy.export.vrm import export_vrm
        from test.test_vrm_schema import check_vrm0, check_vrm1

        folder = self.folder / "normal_map"
        card = write_outfit(folder, normal_map=True)
        for version, check in (("1.0", check_vrm1), ("0.x", check_vrm0)):
            with self.subTest(version):
                path = folder / f"normal_map_{version}.vrm"
                export_vrm(
                    path,
                    card,
                    author="test",
                    version=version,
                    detail_dir=folder,
                    refine_weights=False,
                    textures={"normal": bump_normal_map()},
                )
                glb = GLB(path)
                self.assertEqual(check(glb.json), [])
                mapped = [m for m in glb.json["materials"] if "normalTexture" in m]
                self.assertGreaterEqual(len(mapped), 2, "the skin and the cape")
                (mesh,) = [
                    m for m in glb.json["meshes"] if m["name"] == "test_outfit/cape"
                ]
                for primitive in mesh["primitives"]:
                    self.assertIn("TEXCOORD_0", primitive["attributes"])
                    material = glb.json["materials"][primitive["material"]]
                    self.assertIn("normalTexture", material)

    def test_command(self):
        from opensculptboy.cli import main

        card = self.folder / "card.json"
        card.write_text(json.dumps(self.card.to_dict()))
        out = self.folder / "cmd.glb"
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(
                main(["export", str(out), "--character", str(card), "--hide", "cape"]),
                0,
            )
        summary = _last_json(printed.getvalue())
        self.assertEqual(
            summary["wearables"], ["test_outfit/tunic", "test_outfit/circlet"]
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["export", str(out), "--character", str(card), "--hide", "hat"])
        # a tier export takes the same choices
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            command = ["export", str(out), "--character", str(card), "--tier", "high"]
            self.assertEqual(main(command + ["--hide", "cape"]), 0)
        summary = _last_json(printed.getvalue())
        self.assertEqual(
            summary["wearables"], ["test_outfit/tunic", "test_outfit/circlet"]
        )
        # a package file that is missing is a usage error of one line
        lost = self.folder / "lost.json"
        data = self.card.to_dict()
        data["wearables"] = [{"id": "ring", "file": "ring.wearable.json"}]
        lost.write_text(json.dumps(data))
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as exit:
            main(["export", str(out), "--character", str(lost)])
        self.assertEqual(exit.exception.code, 2)
        self.assertIn("ring.wearable.json", errors.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(main(["wearables", "list", "--character", str(card)]), 0)
        listing = _last_json(printed.getvalue())
        self.assertEqual(listing["active_outfit"], "test_outfit")
        self.assertEqual(len(listing["pieces"]), 3)


def hidden(node: dict) -> bool:
    """Whether KHR_node_visibility hides a node."""
    visibility = node.get("extensions", {}).get("KHR_node_visibility", {})
    return visibility.get("visible") is False


def _last_json(text: str):
    """The JSON object that a command printed last (warp may print its module loads first)."""
    return json.loads(text[text.index("{\n") :])


TUNIC_CREDIT = (
    '"tunic" by OpenSculptBoy tests, CC0-1.0, test/wearable_fixtures.py '
    "(design: synthetic test garment, made in the test)"
)


class TestCredits(unittest.TestCase):
    def test_credit_lines(self):
        """Every piece with licence, author or source information gets a credit line,
        whatever its licence; an attribution line stands as given."""
        from opensculptboy.wearables.export import credit_line

        w = tunic()
        self.assertEqual(credit_line(w), TUNIC_CREDIT)
        for licence, expected in (
            ({"spdx": "AGPL-3.0-only"}, '"tunic", AGPL-3.0-only'),
            ({"author": "Ada"}, '"tunic" by Ada'),
            ({"source": "https://example.org/t"}, '"tunic", https://example.org/t'),
            (
                {"spdx": "CC-BY-4.0", "attribution": "Tunic by Ada (CC BY 4.0)"},
                "Tunic by Ada (CC BY 4.0)",
            ),
            ({}, None),
        ):
            with self.subTest(licence=licence):
                self.assertEqual(
                    credit_line(dataclasses.replace(w, licence=licence)), expected
                )


def glb_json_wearables(path):
    """The ``wearables`` entry of the scene extras of a GLB or VRM file."""
    return GLB(path).json["scenes"][0]["extras"]["opensculptboy"]["wearables"]


def names_index(glb, name):
    return [n.get("name") for n in glb.json["nodes"]].index(name)


def _write_reference(v6: bool, covered: str = "named"):
    """A synthetic reference character: anny's default body posed and placed, as its skin,
    and a shirt (an offset shell of the posed torso, in two groups) and a cloak panel as
    clothing; and the card fitted to it (the same body, with the pose and the placement).
    ``covered`` is ``"named"`` (the pieces name the covered skin), ``"none"`` (no piece names
    covered skin) or ``"split"`` (the skin in two parts, a torso part that the cover entries
    name and a part of the rest, and a band around the left thigh as a third piece).
    Returns (handoff folder, card path, the shirt's posed vertices)."""
    import torch

    import anny.poses
    from opensculptboy.export.body import ANNY_TO_GLTF
    from opensculptboy.export.document import ARRAY_BUFFER, GltfDocument, png_bytes
    from opensculptboy.wearables.fit import skin_triangles
    from opensculptboy.wearables.geometry import vertex_normals
    from test.wearable_fixtures import torso_mask

    model = template_model()
    params = anny.poses.pose_parameters(model, "mh_thinking", grounded=False)
    pose = Character.pose_from_parameters(model, params["pose_parameters"][:1])
    card = Character(name="reference fit", pose=pose)
    with torch.no_grad():
        out = model(pose_parameters=card.pose_parameters(model))
    angle = np.radians(4.0)
    R = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    t = np.array([0.01, -0.02, 0.9])
    posed = out["vertices"][0].double().numpy() @ R.T + t
    triangles = skin_triangles(model)
    normals = vertex_normals(posed, triangles)
    body = template_body()
    keep = torso_mask(body)
    shell_tris = triangles[keep[triangles].all(1)]
    rows = np.unique(shell_tris)
    remap = np.full(len(posed), -1)
    remap[rows] = np.arange(len(rows))
    shirt = posed[rows] + 0.008 * normals[rows]
    shirt_tris = remap[shell_tris]
    back = posed[rows][:, 1].mean() < shirt[shirt_tris].mean(axis=1)[:, 1]
    cloak = np.array(
        [[-0.2, 0.3, 1.2], [0.2, 0.3, 1.2], [0.2, 0.3, 0.6], [-0.2, 0.3, 0.6]]
    )
    cloak = cloak @ R.T + np.array([0.0, 0.0, 0.0]) + np.array([t[0], t[1], 0.0])

    doc = GltfDocument("synthetic reference")
    texture = doc.add_texture(doc.add_png(np.full((4, 4, 3), 180, np.uint8)))
    skin_material = doc.add_material({"name": "skin_mat"})
    cloth = doc.add_material(
        {
            "name": "cloth_mat",
            "pbrMetallicRoughness": {"baseColorTexture": {"index": texture}},
        }
    )

    def node(name, vertices, groups, material):
        attributes = {
            "POSITION": doc.accessor(
                (vertices @ ANNY_TO_GLTF.T).astype(np.float32),
                ARRAY_BUFFER,
                minmax=True,
            )
        }
        primitives = []
        for group, tris in groups:
            primitives.append(
                {
                    "attributes": attributes,
                    "indices": doc.accessor(tris.reshape(-1).astype(np.uint32)),
                    "material": material,
                    "extras": {"group": group},
                }
            )
        mesh = doc.add_mesh({"name": name, "primitives": primitives})
        return doc.add_node(
            {"name": name, "mesh": mesh, "extras": {"part": "clothing"}}
        )

    torso_tris = keep[triangles].all(1)
    if covered == "split":
        nodes = [
            node("skin_body", posed, [(0, triangles[torso_tris])], skin_material),
            node("skin_rest", posed, [(0, triangles[~torso_tris])], skin_material),
        ]
        covered_tris = np.arange(int(torso_tris.sum())).tolist()
        band_rows, band_tris = _thigh_band(body, triangles)
        band = posed[band_rows] + 0.008 * normals[band_rows]
    else:
        nodes = [node("skin_body", posed, [(0, triangles)], skin_material)]
        covered_tris = np.flatnonzero(torso_tris).tolist()
    nodes += [
        node("shirt", shirt, [(0, shirt_tris[~back]), (5, shirt_tris[back])], cloth),
        node("cloak", cloak, [(0, np.array([[0, 2, 1], [0, 3, 2]]))], cloth),
    ]
    if covered == "split":
        nodes.append(node("band", band, [(0, band_tris)], cloth))
    _ = png_bytes
    doc.scene_nodes = nodes
    folder = pathlib.Path(tempfile.mkdtemp())
    handoff = folder / "handoff"
    handoff.mkdir()
    doc.write(handoff / "replica.glb")
    cover = {
        "body": [
            {
                "mesh": "skin_body",
                "node": "skin_body",
                "primitive": 0,
                "triangles": covered_tris,
            }
        ]
    }
    shirt_obj = {"name": "shirt", "slot": "torso", "layer": 50, "node": "shirt"}
    cloak_obj = {"name": "cloak", "slot": "cloak", "layer": 80, "node": "cloak"}
    if v6:
        shirt_obj["primitives"] = [{"index": 0, "group": 0}, {"index": 1, "group": 5}]
        cloak_obj["primitives"] = [{"index": 0, "group": 0}]
        equipment = {
            "default_outfit": "travel",
            "outfits": [
                {
                    "id": "travel",
                    "label": "Travel",
                    "default": True,
                    "pieces": [
                        {
                            "piece": "travel.shirt",
                            "slot": "torso",
                            "layer": 50,
                            "objects": [shirt_obj],
                            "hides": [],
                            "covered": cover,
                        },
                        {
                            "piece": "travel.cloak",
                            "slot": "cloak",
                            "layer": 80,
                            "objects": [cloak_obj],
                            "hides": [
                                {
                                    "piece": "travel.shirt",
                                    "object": "shirt",
                                    "groups": [5],
                                }
                            ],
                            "covered": {"body": []},
                        },
                    ],
                }
            ],
        }
    else:
        shirt_obj["primitives"] = [0, 1]
        cloak_obj["primitives"] = [0]
        equipment = {
            "objects": [shirt_obj, cloak_obj],
            "hides": [{"by": "cloak C1", "object": "shirt", "mask": "0x20"}],
            "covered": cover,
            "source": {"look": "Travel"},
        }
    if covered == "split":
        band_obj = {"name": "band", "slot": "legs", "layer": 30, "node": "band"}
        if v6:
            band_obj["primitives"] = [{"index": 0, "group": 0}]
            equipment["outfits"][0]["pieces"].append(
                {
                    "piece": "travel.band",
                    "slot": "legs",
                    "layer": 30,
                    "objects": [band_obj],
                    "hides": [],
                }
            )
        else:
            band_obj["primitives"] = [0]
            equipment["objects"].append(band_obj)
    if covered == "none":
        _drop_covered(equipment)
    appearance = {
        "format": "test",
        "parts": {
            "skin_body": {"role": "skin"},
            "skin_rest": {"role": "skin"},
            "band": {"role": "clothing"},
            "shirt": {"role": "clothing"},
            "cloak": {"role": "clothing"},
        },
        "equipment": equipment,
    }
    (handoff / "appearance.json").write_text(json.dumps(appearance))
    card.detail = {
        "fit": {"placement": {"rotation": R.tolist(), "translation": t.tolist()}}
    }
    card_path = folder / "card" / "card.json"
    card_path.parent.mkdir()
    card_path.write_text(json.dumps(card.to_dict()))
    return handoff, card_path, shirt


def _thigh_band(body, triangles):
    """The rows and triangles (renumbered) of a band of the skin around the left thigh."""
    v = body.vertices
    low, h = float(v[:, 2].min()), float(np.ptp(v[:, 2]))
    keep = (v[:, 2] > low + 0.33 * h) & (v[:, 2] < low + 0.40 * h) & (v[:, 0] > 0.02)
    tris = triangles[keep[triangles].all(1)]
    rows = np.unique(tris)
    remap = np.full(len(v), -1)
    remap[rows] = np.arange(len(rows))
    return rows, remap[tris]


def _drop_covered(value):
    """Remove every ``covered`` field of an equipment section, in place."""
    if isinstance(value, dict):
        value.pop("covered", None)
        for item in value.values():
            _drop_covered(item)
    elif isinstance(value, list):
        for item in value:
            _drop_covered(item)


class TestReferenceWearables(unittest.TestCase):
    def test_reference_outfits(self):
        from opensculptboy.wearables.reference import build_reference_wearables

        for v6 in (False, True):
            with self.subTest(v6=v6):
                handoff, card_path, shirt = _write_reference(v6)
                out = handoff.parent / "dressed"
                with self.assertRaises(ProvenanceError):
                    build_reference_wearables(
                        handoff, card_path, ROOT / "test" / "data" / "x"
                    )
                report = build_reference_wearables(
                    handoff, card_path, out, log=lambda m: None
                )
                pieces = {p["piece"]: p for p in report["outfits"][0]["pieces"]}
                self.assertEqual(sorted(pieces), ["cloak", "shirt"])
                self.assertLess(pieces["shirt"]["round_trip_mm_max"], 0.01)
                self.assertLess(pieces["shirt"]["skin_to_card_mm_p95"], 0.01)
                card = Character.from_dict(json.loads((out / "card.json").read_text()))
                self.assertEqual(card.active_outfit, "travel")
                listed = worn_pieces(card, out)
                by_ref = {p.ref: p for p in listed}
                self.assertEqual(by_ref["travel/shirt"].wearable.provenance, PRIVATE)
                self.assertEqual(by_ref["travel/shirt"].hidden_groups, ["shirt:group5"])
                hidden = worn_pieces(card, out, hide=["cloak"])
                self.assertEqual(hidden[0].hidden_groups, [])
                shirt_package = by_ref["travel/shirt"].wearable
                self.assertEqual(shirt_package.weights, "own")
                self.assertTrue(shirt_package.cover_ray)
                np.testing.assert_allclose(
                    shirt_package.binding.skin_weights.sum(1), 1.0, atol=1e-9
                )
                covers = shirt_package.covers
                self.assertGreater(int(covers.sum()), 300)
                # at rest, the shirt lies about 8 mm off the card's rest skin
                body = card_body(card, template_model(), out)
                rest, _ = evaluate(shirt_package.binding, body)
                signed, found, _ = body.surface.signed_distance(rest)
                self.assertLess(abs(float(np.median(signed)) - 0.008), 0.0015)
                path = out / "dressed.glb"
                summary = export_glb(path, card, template_model(), detail_dir=out)
                self.assertEqual(summary["wearables"], ["travel/shirt", "travel/cloak"])
                glb = GLB(path)
                self.assertEqual(structure_errors(glb), [])
                self.assertTrue(PRIVATE_MARK.search(path.read_bytes()))

    def test_skin_without_cover_entries(self):
        """A handoff whose pieces name no covered skin builds, and the clothing matches
        against every skin part of the handoff: a thigh band binds to the thigh although
        the cover entries name only the torso part."""
        from opensculptboy.wearables.reference import build_reference_wearables

        body = template_body()
        v = body.vertices
        low, h = float(v[:, 2].min()), float(np.ptp(v[:, 2]))
        for v6, covered in ((False, "none"), (True, "split")):
            with self.subTest(v6=v6, covered=covered):
                handoff, card_path, _ = _write_reference(v6, covered)
                out = handoff.parent / "dressed"
                report = build_reference_wearables(
                    handoff, card_path, out, log=lambda m: None
                )
                pieces = {p["piece"]: p for p in report["outfits"][0]["pieces"]}
                self.assertLess(pieces["shirt"]["round_trip_mm_max"], 0.01)
                self.assertLess(pieces["shirt"]["skin_to_card_mm_p95"], 0.01)
                card = Character.from_dict(
                    json.loads((out / "card.json").read_text(encoding="utf-8"))
                )
                by_ref = {p.ref: p for p in worn_pieces(card, out)}
                if covered == "none":
                    self.assertIsNone(by_ref["travel/shirt"].wearable.covers)
                    continue
                self.assertGreater(
                    int(by_ref["travel/shirt"].wearable.covers.sum()), 300
                )
                self.assertLess(pieces["band"]["gear_to_skin_mm_p95"], 9.0)
                corners = by_ref["travel/band"].wearable.binding.corners
                z = v[corners.ravel(), 2]
                self.assertLess(float(z.max()), low + 0.45 * h)


if __name__ == "__main__":
    unittest.main()

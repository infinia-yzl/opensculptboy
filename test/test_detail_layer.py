# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the detail layer (``opensculptboy.detail``) and of its use in the character card and
the exporters: schema 2 cards and their backwards-compatible loading, the subdivided meshes
(UVs, skin weights, the head region), :class:`DetailModel` against Anny, the offsets files, the
GLB file of level 1 against the detailed rest mesh and its morph targets, the VRM files within
their budgets, the baked textures in the glTF and MToon materials, and the files of a card
without detail, which stay byte for byte what they were.
"""

import json
import pathlib
import tempfile
import unittest
import warnings

import numpy as np
import torch

from opensculptboy import Character, export_glb
from opensculptboy.character import SCHEMA_VERSION
from opensculptboy.detail import (
    DetailModel,
    character_offsets,
    detail_topology,
    level,
    load_offsets,
    save_offsets,
)
from opensculptboy.export.budget import BUDGETS, BudgetWarning
from opensculptboy.export.document import png_bytes
from test.gltf_reader import GLB, evaluate, structure_errors


def detailed_card(folder: pathlib.Path, seed: int = 0) -> Character:
    """A card with random offsets (about 1 mm) on levels 0 and 1, written into ``folder``."""
    rng = np.random.default_rng(seed)
    levels = []
    for subdiv in (0, 1):
        name = f"card.detail{subdiv}.safetensors"
        count = level("anny", subdiv).vertex_count
        save_offsets(folder / name, 0.001 * rng.standard_normal((count, 3)), subdiv)
        levels.append({"subdiv": subdiv, "offsets": name})
    return Character(
        name="detailed",
        phenotype={"age": 0.4, "weight": 0.6},
        detail={"levels": levels},
    )


def _posed_parameters(model):
    import roma

    labels = list(model.bone_labels)
    pose = torch.eye(4, dtype=model.dtype).repeat(1, len(labels), 1, 1)
    for bone, rotvec in (
        ("upperarm01.L", [0.3, 0.5, 0.2]),
        ("lowerleg01.R", [0.6, 0.0, 0.0]),
        ("neck02", [0.1, -0.2, 0.0]),
    ):
        pose[0, labels.index(bone), :3, :3] = roma.rotvec_to_rotmat(
            torch.tensor(rotvec, dtype=model.dtype)
        )
    return pose


class TestCharacterDetail(unittest.TestCase):
    def test_card_without_detail_keeps_schema_1(self):
        data = Character(name="plain", phenotype={"age": 0.3}).to_dict()
        self.assertEqual(data["schema"], 1)
        self.assertNotIn("detail", data)
        self.assertEqual(
            Character.from_dict(data), Character(name="plain", phenotype={"age": 0.3})
        )

    def test_schema_1_cards_still_load(self):
        old = {
            "schema": 1,
            "name": "old",
            "rig": "anny",
            "topology": "anny",
            "phenotypes": "default",
            "phenotype": {"age": 0.2},
            "local_changes": {},
            "face_shapes": {},
            "facial_actions": {},
            "pose": {},
        }
        character = Character.from_dict(old)
        self.assertEqual(character.detail, {})
        self.assertEqual(character.to_dict(), old)

    def test_detail_round_trip(self):
        detail = {
            "levels": [
                {"subdiv": 0, "offsets": "a.detail0.safetensors", "error_mm": 0.3},
                {"subdiv": 1, "offsets": "a.detail1.safetensors"},
            ],
            "fit": {"errors_mm": {"1": {"forward": 0.03}}},
        }
        card = Character(name="a", detail=detail)
        data = json.loads(json.dumps(card.to_dict()))
        self.assertEqual(data["schema"], SCHEMA_VERSION)
        self.assertEqual(Character.from_dict(data), card)
        self.assertEqual(card.detail_levels(), [0, 1])

    def test_invalid_detail_is_refused(self):
        for detail in (
            {"levels": [{"subdiv": 3, "offsets": "x"}]},
            {"levels": [{"subdiv": 1}]},
            {"levels": [{"subdiv": 1, "offsets": "x"}, {"subdiv": 1, "offsets": "y"}]},
            {"levels": [{"subdiv": True, "offsets": "x"}]},
        ):
            with self.subTest(detail=detail), self.assertRaises(ValueError):
                Character.from_dict({"schema": 2, "detail": detail})
        with self.assertRaisesRegex(ValueError, "schema 2"):
            Character.from_dict({"schema": 1, "detail": {"levels": []}})
        with self.assertRaisesRegex(ValueError, "schema"):
            Character.from_dict({"schema": 3})


class TestSubdivision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Character().build_model(dtype=torch.float64)

    def test_level_sizes(self):
        sizes = [level("anny", s).vertex_count for s in (0, 1, 2)]
        self.assertEqual(sizes, [13718, 54848, 219368])
        one = detail_topology(self.model, 1)
        self.assertEqual(len(one.triangles), 8 * 13710)
        self.assertEqual(one.uv_triangles.shape, one.triangles.shape)
        self.assertLess(one.uv_triangles.max(), len(one.uvs))

    def test_level0_is_anny_mesh(self):
        from opensculptboy.export.body import triangulated_faces

        zero = detail_topology(self.model, 0)
        triangles, uv_triangles = triangulated_faces(self.model)
        np.testing.assert_array_equal(zero.triangles, triangles)
        np.testing.assert_array_equal(zero.uv_triangles, uv_triangles)
        self.assertEqual(zero.operators, [])

    def test_uvs_are_bilinear_in_each_face(self):
        # the corner of each level-1 face at a vertex point keeps the UV of its parent corner
        zero, one = level("anny", 0), level("anny", 1)
        faces = len(zero.quads)
        for k in range(4):
            block = one.uv_quads[k * faces : (k + 1) * faces]
            np.testing.assert_array_equal(block[:, 0], zero.uv_quads[:, k])
            mid = one.uvs[block[:, 1]]
            expected = 0.5 * (
                zero.uvs[zero.uv_quads[:, k]] + zero.uvs[zero.uv_quads[:, (k + 1) % 4]]
            )
            np.testing.assert_allclose(mid, expected, atol=1e-12)

    def test_skin_weights_sum_to_one(self):
        for subdiv, region in ((1, "full"), (1, "head"), (2, "head")):
            with self.subTest(subdiv=subdiv, region=region):
                topology = detail_topology(self.model, subdiv, region)
                w, i = topology.skin_weights(
                    self.model.vertex_bone_weights.double().numpy(),
                    self.model.vertex_bone_indices.numpy(),
                )
                self.assertEqual(len(w), topology.vertex_count)
                np.testing.assert_allclose(w.sum(1), 1.0, atol=1e-12)
                self.assertTrue((np.diff(w, axis=1) <= 1e-15).all())

    def test_head_region_joins_without_cracks(self):
        head = detail_topology(self.model, 1, "head")
        self.assertLessEqual(len(head.triangles), BUDGETS["vrm1"]["triangles"])
        # every edge of a closed part of the mesh borders two triangles
        t = head.triangles
        edges = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), 1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        full = detail_topology(self.model, 1).triangles
        full_edges = np.sort(
            np.concatenate([full[:, [0, 1]], full[:, [1, 2]], full[:, [2, 0]]]), 1
        )
        _, full_counts = np.unique(full_edges, axis=0, return_counts=True)
        self.assertEqual((counts == 1).sum(), (full_counts == 1).sum())
        self.assertTrue(
            np.isin(head.rows, np.arange(level("anny", 1).vertex_count)).all()
        )


class TestDetailModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Character().build_model(dtype=torch.float64)
        cls.pose = _posed_parameters(cls.model)

    def test_level0_without_offsets_is_anny(self):
        detailed = DetailModel(self.model, 0)
        kwargs = dict(pose_parameters=self.pose, phenotype_kwargs={"age": 0.3})
        with torch.no_grad():
            out = detailed(**kwargs)
            ref = self.model(**kwargs)
        torch.testing.assert_close(out["vertices"], ref["vertices"], atol=1e-12, rtol=0)

    def test_offsets_add_in_the_rest_frame(self):
        rng = np.random.default_rng(1)
        offsets = 0.002 * rng.standard_normal((level("anny", 1).vertex_count, 3))
        plain = DetailModel(self.model, 1)
        detailed = DetailModel(self.model, 1, offsets)
        rest_pose = torch.eye(4, dtype=self.model.dtype).repeat(
            1, self.model.bone_count, 1, 1
        )
        kwargs = dict(pose_parameters=rest_pose, phenotype_kwargs={"weight": 0.7})
        with torch.no_grad():
            a = plain(**kwargs)
            b = detailed(**kwargs)
        np.testing.assert_allclose(
            (b["rest_vertices"] - a["rest_vertices"])[0].numpy(), offsets, atol=1e-12
        )
        # the skinning turns each offset by the linear part of its blended bone transform,
        # and unpose() takes the posed mesh back to the detailed rest mesh
        from opensculptboy.detail.layer import bone_transforms
        from opensculptboy.fit.residual import blended_transforms, unpose

        X = bone_transforms(b)[0].numpy()
        w, i = detailed.weights.numpy(), detailed.indices.numpy()
        M = blended_transforms(w, i, X)
        moved = (b["vertices"] - a["vertices"])[0].numpy()
        np.testing.assert_allclose(
            moved, np.einsum("nij, nj -> ni", M[:, :3, :3], offsets), atol=1e-12
        )
        rest = unpose(b["vertices"][0].numpy(), w, i, X)
        np.testing.assert_allclose(rest, b["rest_vertices"][0].numpy(), atol=1e-10)

    def test_offsets_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            card = detailed_card(folder)
            offsets = character_offsets(card, 1, folder)
            self.assertEqual(offsets.shape, (54848, 3))
            self.assertIsNone(character_offsets(card, 2, folder))
            with self.assertRaisesRegex(ValueError, "level 1"):
                load_offsets(folder / "card.detail1.safetensors", 0)
            with self.assertRaises(ValueError):
                save_offsets(folder / "bad.safetensors", np.zeros((10, 3)), 1)


class TestDetailExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.card = detailed_card(cls.folder)
        cls.model = cls.card.build_model()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_card_without_detail_writes_the_same_files(self):
        character = Character(name="plain", phenotype={"age": 0.3})
        a, b = self.folder / "a.glb", self.folder / "b.glb"
        export_glb(a, character)
        export_glb(b, character, detail_level=0, detail_dir=self.folder)
        self.assertEqual(a.read_bytes(), b.read_bytes())
        self.assertNotIn("detail", GLB(a).json["scenes"][0]["extras"]["opensculptboy"])

    def test_missing_level_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no detail level 2"):
            export_glb(
                self.folder / "x.glb",
                self.card,
                self.model,
                detail_level=2,
                detail_dir=self.folder,
            )
        with self.assertRaisesRegex(ValueError, "no detail layer"):
            export_glb(self.folder / "x.glb", Character(), detail_level=1)

    def test_glb_level1_is_the_detailed_rest_mesh(self):
        from scipy.spatial import cKDTree

        from opensculptboy.export.gltf import C3

        path = self.folder / "level1.glb"
        summary = export_glb(
            path, self.card, self.model, detail_level=1, detail_dir=self.folder
        )
        self.assertEqual(summary["triangles"], 8 * 13710)
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        extras = glb.json["scenes"][0]["extras"]["opensculptboy"]
        self.assertEqual(extras["detail"]["subdiv"], 1)
        self.assertEqual(len(glb.json["meshes"][0]["primitives"][0]["targets"]), 52)

        detailed = DetailModel(
            self.model.double(), 1, character_offsets(self.card, 1, self.folder)
        )
        kwargs = self.card.model_kwargs()
        jaw = dict(kwargs, facial_actions={"jawOpen": 1.0})
        with torch.no_grad():
            rest = detailed(**kwargs)["rest_vertices"][0].numpy() @ C3.T
            open_ = detailed(**jaw)["rest_vertices"][0].numpy() @ C3.T
        self.model.float()
        positions, _ = evaluate(glb)
        lift = positions[:, 1].min() - rest[:, 1].min()
        distance, rows = cKDTree(rest).query(positions - [0.0, lift, 0.0])
        self.assertLess(distance.max(), 2e-6)
        names = glb.json["meshes"][0]["extras"]["targetNames"]
        weights = np.zeros(len(names))
        weights[names.index("jawOpen")] = 1.0
        opened, _ = evaluate(glb, weights=weights)
        np.testing.assert_allclose(
            opened - [0.0, lift, 0.0], open_[rows], atol=2e-6, rtol=0
        )

    def test_vrm_keeps_within_its_budget(self):
        from opensculptboy.export.vrm import export_vrm

        tiny = png_bytes(np.full((8, 8, 3), 200, np.uint8))
        normal = png_bytes(np.tile(np.array([128, 128, 255], np.uint8), (8, 8, 1)))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            one = export_vrm(
                self.folder / "one.vrm",
                self.card,
                self.model,
                author="tests",
                detail_level=1,
                detail_dir=self.folder,
                budget="strict",
                refine_weights=False,
                textures={"base_color": tiny, "normal": normal},
            )
        self.assertLessEqual(one["triangles"], BUDGETS["vrm1"]["triangles"])
        self.assertEqual(one["budget"], [])
        self.assertTrue(any(issubclass(w.category, BudgetWarning) for w in caught))
        glb = GLB(self.folder / "one.vrm")
        self.assertEqual(structure_errors(glb), [])
        extras = glb.json["scenes"][0]["extras"]["opensculptboy"]
        self.assertEqual(
            (extras["detail"]["subdiv"], extras["detail"]["region"]), (1, "head")
        )
        material = glb.json["materials"][0]
        mtoon = material["extensions"]["VRMC_materials_mtoon"]
        self.assertEqual(
            material["pbrMetallicRoughness"]["baseColorTexture"]["index"], 0
        )
        self.assertEqual(mtoon["shadeMultiplyTexture"]["index"], 0)
        self.assertEqual(material["normalTexture"]["index"], 1)
        self.assertEqual(extras["textures"], ["base_color", "normal"])

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            zero = export_vrm(
                self.folder / "zero.vrm",
                self.card,
                self.model,
                version="0.x",
                author="tests",
                detail_level=1,
                detail_dir=self.folder,
                budget="strict",
                refine_weights=False,
            )
        self.assertLessEqual(zero["triangles"], BUDGETS["vrm0"]["triangles"])
        extras = GLB(self.folder / "zero.vrm").json["scenes"][0]["extras"][
            "opensculptboy"
        ]
        self.assertEqual(extras["detail"]["subdiv"], 0)

    def test_textures_in_the_gltf_material(self):
        image = png_bytes(np.full((4, 4, 3), 90, np.uint8))
        path = self.folder / "textured.glb"
        export_glb(
            path,
            Character(),
            self.model,
            textures={"base_color": image, "normal": image},
        )
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        material = glb.json["materials"][0]
        self.assertIn("baseColorTexture", material["pbrMetallicRoughness"])
        self.assertIn("normalTexture", material)
        self.assertEqual(len(glb.json["images"]), 2)
        with self.assertRaisesRegex(ValueError, "Unknown texture slots"):
            export_glb(path, Character(), self.model, textures={"albedo": image})


if __name__ == "__main__":
    unittest.main()

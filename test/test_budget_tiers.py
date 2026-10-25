# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the rendering-budget tiers (``opensculptboy.tiers``): the presets and the mesh each
takes for a card (never finer than the master), the polygon and texture budgets against
``opensculptboy.export.budget.BUDGETS``, the card's ``material.tiers`` section, the bake of the
master's normals onto the tiers' meshes (``opensculptboy.texture.transfer``), which brings the
shading normals of a low tier close to the master's on the synthetic master of
``test/tier_fixtures.py``, the occlusion, thickness and curvature maps of every tier, the GLB and
VRM files of the tiers, and the portraits and the expression sheet. Cards and exports without
tiers stay as they were (``TestGltfReference`` in ``test.test_gltf_export``).
"""

import dataclasses
import io
import json
import pathlib
import tempfile
import unittest
import warnings

import numpy as np

from opensculptboy import Character
from opensculptboy.export.budget import BUDGETS
from opensculptboy.export.document import png_bytes
from opensculptboy.material import check_material
from opensculptboy.tiers import (
    MESH_COUNTS,
    MESHES,
    TIERS,
    _resize,
    check_budget,
    export_tier,
    file_counts,
    missing_maps,
    needs_bake,
    tier_mesh,
)
from test.gltf_reader import GLB, structure_errors

# the normal error of the low tier against the synthetic master on level 1 (degrees, mean over
# 20,000 points): without a normal map, with the master's map and with the baked map; measure
# again when a change moves them (docs/budget-tiers.md quotes them)
MEASURED_LOW_ERROR = {"none": 6.70, "master_map": 1.80, "baked": 0.48}


def _card(levels) -> Character:
    """A card whose detail layer names ``levels`` (no files: the tiers' meshes only)."""
    entries = [{"subdiv": s, "offsets": f"x{s}.safetensors"} for s in levels]
    return Character(detail={"levels": entries} if levels else {})


def _has_embree() -> bool:
    try:
        import embreex  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        return False
    return True


class TestPresets(unittest.TestCase):
    def test_tier_meshes_cut_down_from_the_master(self):
        expected = {
            (): {t: (0, "full") for t in TIERS},
            (0, 1): {
                "master": (1, "full"),
                "high": (1, "full"),
                "medium": (1, "head"),
                "low": (0, "full"),
            },
            (0, 1, 2): {t: TIERS[t].mesh for t in TIERS},
        }
        for levels, meshes in expected.items():
            card = _card(levels)
            for name, mesh in meshes.items():
                with self.subTest(levels=levels, tier=name):
                    self.assertEqual(tier_mesh(card, name), mesh)
        self.assertEqual(
            [TIERS[t].mesh for t in ("low", "medium", "high", "master")],
            [MESHES[0], MESHES[1], MESHES[3], MESHES[4]],
        )

    def test_which_maps_need_a_bake(self):
        card = _card((0, 1, 2))
        self.assertEqual(
            needs_bake(card, "master"),
            {"normal": False, "head_normal": False, "ears_normal": False},
        )
        # the high tier's head is the master's: its tiles keep the master's maps
        self.assertEqual(
            needs_bake(card, "high"),
            {"normal": True, "head_normal": False, "ears_normal": False},
        )
        self.assertEqual(
            needs_bake(card, "medium"),
            {"normal": True, "head_normal": True, "ears_normal": False},
        )
        self.assertFalse(any(needs_bake(_card(()), "low").values()))

    def test_mesh_counts(self):
        from opensculptboy.detail import detail_topology

        model = Character().build_model()
        for mesh, (triangles, vertices) in MESH_COUNTS.items():
            with self.subTest(mesh=mesh):
                topology = detail_topology(model, *mesh)
                self.assertEqual(len(topology.triangles), triangles)
                self.assertEqual(topology.vertex_count, vertices)

    def test_budgets(self):
        for name, tier in TIERS.items():
            with self.subTest(tier=name):
                triangles = MESH_COUNTS[tier.mesh][0]
                self.assertLessEqual(triangles, tier.budget["triangles"])
                self.assertLessEqual(tier.atlas_size, tier.budget["texture_size"])
                self.assertLessEqual(tier.head_tile_size, tier.budget["texture_size"])
                self.assertLessEqual(tier.ear_tile_size, tier.budget["texture_size"])
                self.assertIn(
                    tier.shading,
                    ("path_traced", "subsurface_profile", "preintegrated", "toon"),
                )
        low, medium = TIERS["low"], TIERS["medium"]
        for budget in BUDGETS.values():  # the low tier fits VRM 1.0 and 0.x
            self.assertLessEqual(MESH_COUNTS[low.mesh][0], budget["triangles"])
            self.assertLessEqual(low.atlas_size, budget["texture_size"])
        self.assertLessEqual(MESH_COUNTS[medium.mesh][0], BUDGETS["vrm1"]["triangles"])
        self.assertGreater(MESH_COUNTS[medium.mesh][0], BUDGETS["vrm0"]["triangles"])
        self.assertEqual(low.format, "vrm")
        self.assertEqual(
            check_budget({"triangles": 30_000, "max_image_size": 2048}, "low"),
            ["texture_size: 2,048 over the low budget of 1,024"],
        )

    def test_vrm_over_its_budget_is_refused(self):
        card = _card((0, 1, 2))
        with tempfile.TemporaryDirectory() as folder:
            for name, version in (
                ("high", "1.0"),
                ("master", "1.0"),
                ("medium", "0.x"),
            ):
                with self.subTest(tier=name, version=version):
                    with self.assertRaisesRegex(ValueError, "over the VRM"):
                        export_tier(
                            pathlib.Path(folder) / "x.vrm", card, name, version=version
                        )
            self.assertEqual(list(pathlib.Path(folder).iterdir()), [])

    def test_wearables_count_against_the_tier(self):
        """The triangles, materials and textures of the worn pieces count in the tier's
        budget, and in the triangle budget of a tier's VRM file."""
        from unittest import mock

        from opensculptboy.export import budget as budgets
        from opensculptboy.tiers import tier_budget, wearable_counts
        from opensculptboy.wearables.export import Dressing
        from test.wearable_fixtures import PIECES, write_outfit

        with tempfile.TemporaryDirectory() as folder:
            card = write_outfit(folder)
            pieces = [make() for make in PIECES]
            triangles = sum(len(w.garment.triangles) for w in pieces)
            # the cape hides the tunic's back group
            tunic = next(w for w in pieces if w.name == "tunic")
            back = tunic.garment.groups.index("back")
            hidden = int((tunic.garment.triangle_group == back).sum())
            counted = tier_budget(card, "low", detail_dir=folder)
            self.assertEqual(counted["wearables"]["pieces"], 3)
            self.assertEqual(counted["wearables"]["triangles"], triangles - hidden)
            self.assertEqual(counted["wearables"]["materials"], 3)
            self.assertEqual(
                counted["triangles"], MESH_COUNTS[(0, "full")][0] + triangles - hidden
            )
            wardrobe = wearable_counts(card, folder, Dressing(wardrobe=True))
            self.assertEqual(wardrobe["triangles"], triangles)
            self.assertEqual(wearable_counts(card, folder, bare=True)["pieces"], 0)
            self.assertEqual(tier_budget(_card(()), "low")["wearables"]["pieces"], 0)
            # a budget between the body and the dressed body refuses the dressed file alone
            limit = MESH_COUNTS[(0, "full")][0] + 10
            vrm0 = dict(budgets.BUDGETS["vrm0"], triangles=limit)
            with mock.patch.dict(budgets.BUDGETS, {"vrm0": vrm0}):
                with self.assertRaisesRegex(
                    ValueError, "3 wearable pieces .* over the VRM"
                ):
                    export_tier(
                        pathlib.Path(folder) / "x.vrm",
                        card,
                        "low",
                        folder,
                        version="0.x",
                    )
            self.assertFalse((pathlib.Path(folder) / "x.vrm").exists())

    def test_material_tiers(self):
        material = {
            "normal": "n.png",
            "occlusion": "ao.png",
            "tiers": {"low": {"subdiv": 0, "region": "full", "normal": "l.png"}},
        }
        card = Character(material=material)
        self.assertEqual(
            Character.from_dict(json.loads(json.dumps(card.to_dict()))), card
        )
        self.assertEqual(card.skin_material().textures()["occlusion"], "ao.png")
        for tiers in (
            {"tiny": {"subdiv": 0}},
            {"low": {"normal": "l.png"}},
            {"low": {"subdiv": 3}},
            {"low": {"subdiv": 0, "region": "body"}},
            {"low": {"subdiv": 0, "roughness": "r.png"}},
            {"low": {"subdiv": 0, "normal": 1}},
        ):
            with self.subTest(tiers=tiers), self.assertRaises(ValueError):
                check_material({"tiers": tiers})

    def test_resize(self):
        rng = np.random.default_rng(0)
        n = rng.normal(size=(64, 64, 3)) * [0.3, 0.3, 0.1] + [0, 0, 1]
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        data = png_bytes(np.clip(np.round((n + 1) * 127.5), 0, 255).astype(np.uint8))
        small = _resize(data, (16, 16), normal=True)
        from opensculptboy.texture.transfer import decode_image

        pixels = decode_image(small).astype(float) / 127.5 - 1.0
        self.assertEqual(pixels.shape, (16, 16, 3))
        np.testing.assert_allclose(np.linalg.norm(pixels, axis=-1), 1.0, atol=0.02)
        self.assertIs(_resize(data, (64, 64)), data)  # never upscaled


@unittest.skipUnless(_has_embree(), "the bake needs SciPy and embreex (viewer extra)")
class TestTierBake(unittest.TestCase):
    """The tiers of a master on level 1 (the CI's memory), with small textures."""

    @classmethod
    def setUpClass(cls):
        from opensculptboy.tiers import bake_tiers
        from test.tier_fixtures import tiered_card

        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.card_path = tiered_card(cls.folder, levels=(0, 1))
        # the low tier at the size of the master's maps (512), the others shrunk to 256
        cls.tiers = {}
        for name in ("high", "medium", "low"):
            size = 512 if name == "low" else 256
            cls.tiers[name] = dataclasses.replace(
                TIERS[name],
                atlas_size=size,
                head_tile_size=size if TIERS[name].head_tile_size else 0,
                eye_size=size,
                budget=dict(TIERS[name].budget, texture_size=size),
            )
        cls.character, cls.report = bake_tiers(
            cls.card_path, list(cls.tiers.values()), rays=16
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_maps_of_every_tier(self):
        material = self.character.material
        for slot in ("occlusion", "thickness", "curvature"):
            self.assertTrue((self.folder / material[slot]).exists(), slot)
        for name, tier in self.tiers.items():
            with self.subTest(tier=name):
                self.assertEqual(missing_maps(self.character, tier, self.folder), [])
        # level 1 is the master here: the high tier takes the master's mesh and maps
        self.assertEqual(self.report["high"]["baked"], [])
        self.assertNotIn("high", material["tiers"])
        # the medium tier's head (level 1) is the master's: it keeps the master's head maps
        self.assertEqual(material["tiers"]["medium"]["region"], "head")
        for name in ("medium", "low"):
            self.assertEqual(
                sorted(material["tiers"][name]), ["normal", "region", "subdiv"]
            )
        reloaded = Character.from_dict(json.loads(self.card_path.read_text()))
        self.assertEqual(reloaded.material, material)
        occlusion = _grey(self.folder / material["occlusion"])
        self.assertGreater(occlusion.mean(), 0.6)
        self.assertLess(occlusion.min(), 0.4)  # the mouth, the armpits, the folds

    def test_baked_normals_bring_the_low_tier_to_the_master(self):
        import torch

        from opensculptboy.export.detail import resolve, texture_images
        from opensculptboy.texture.bake import eyeball_triangles
        from opensculptboy.texture.transfer import (
            decode_image,
            master_reference,
            normal_error,
        )

        character, folder = self.character, self.folder
        model = character.build_model(dtype=torch.float64)
        with torch.no_grad():
            base = model(**character.model_kwargs())["rest_vertices"][0].numpy()
        master = resolve(model, character, 1, folder, "full")
        images = texture_images(character, None, folder)
        reference = master_reference(
            master.vertices(base), master.topology, {"normal": images["normal"]}, model
        )
        low = resolve(model, character, 0, folder, "full")
        exclude = eyeball_triangles(model, low.topology)
        baked = decode_image(
            (folder / character.material["tiers"]["low"]["normal"]).read_bytes()
        )
        errors = {
            key: normal_error(
                reference,
                low.vertices(base),
                low.topology,
                maps,
                model,
                exclude=exclude,
            )
            for key, maps in (
                ("none", None),
                ("master_map", {"normal": decode_image(images["normal"])}),
                ("baked", {"normal": baked}),
            )
        }
        means = {k: float(v.mean()) for k, v in errors.items()}
        self.assertGreater(len(errors["baked"]), 19_000)  # the rays meet the master
        self.assertLess(means["baked"], 0.4 * means["master_map"], means)
        self.assertLess(means["baked"], 0.1 * means["none"], means)
        for key, value in MEASURED_LOW_ERROR.items():
            self.assertLess(means[key], 1.2 * value, (key, means))

    def test_low_tier_files(self):
        tier = self.tiers["low"]
        glb_path = self.folder / "low.glb"
        summary = export_tier(glb_path, self.character, tier, self.folder)
        glb = GLB(glb_path)
        self.assertEqual(structure_errors(glb), [])
        self.assertEqual(summary["triangles"], MESH_COUNTS[(0, "full")][0])
        self.assertEqual(summary["tier_messages"], [])
        extras = glb.json["scenes"][0]["extras"]["opensculptboy"]
        self.assertEqual(extras["tier"]["name"], "low")
        skin = glb.json["materials"][0]
        self.assertIn("occlusionTexture", skin)
        self.assertEqual(skin["extras"]["opensculptboy_skin"]["shading"], "toon")
        self.assertNotIn("metallicRoughnessTexture", skin["pbrMetallicRoughness"])
        self.assertEqual(file_counts(glb_path)["max_image_size"], 512)
        names = glb.json["meshes"][0]["extras"]["targetNames"]
        self.assertTrue(
            set(names) < set(Character().build_model().facial_action_labels)
        )
        self.assertIn("mouthSmileLeft", names)

        vrm_path = self.folder / "low.vrm"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            summary = export_tier(
                vrm_path,
                self.character,
                tier,
                self.folder,
                author="test",
                refine_weights=False,
            )
        vrm = GLB(vrm_path)
        self.assertEqual(structure_errors(vrm), [])
        self.assertEqual(
            file_counts(vrm_path)["triangles"], MESH_COUNTS[(0, "full")][0]
        )
        mtoon = vrm.json["materials"][0]["extensions"]["VRMC_materials_mtoon"]
        base = vrm.json["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"][
            "index"
        ]
        self.assertNotEqual(mtoon["shadeMultiplyTexture"]["index"], base)
        self.assertEqual(
            vrm.json["scenes"][0]["extras"]["opensculptboy"]["tier"]["shading"], "toon"
        )

    def test_medium_tier_file(self):
        tier = self.tiers["medium"]
        path = self.folder / "medium.glb"
        summary = export_tier(path, self.character, tier, self.folder)
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        self.assertEqual(summary["triangles"], MESH_COUNTS[(1, "head")][0])
        self.assertEqual(summary["detail"]["region"], "head")
        self.assertEqual(check_budget(file_counts(path), tier), [])
        self.assertEqual([m["name"] for m in glb.json["materials"]], ["skin"])
        self.assertEqual(file_counts(path)["max_image_size"], 256)  # shrunk from 512
        skin = glb.json["materials"][0]["extras"]["opensculptboy_skin"]
        self.assertEqual(skin["shading"], "preintegrated")
        self.assertIn("thickness", skin)
        self.assertIn("curvature", skin)
        self.assertEqual(len(glb.json["meshes"][0]["extras"]["targetNames"]), 52)


class TestCommand(unittest.TestCase):
    def test_export_tier_and_portraits(self):
        import contextlib

        from opensculptboy.cli import main

        with tempfile.TemporaryDirectory() as folder:
            folder = pathlib.Path(folder)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    main(["export", str(folder / "a"), "--tier", "medium"]), 0
                )
            text = stdout.getvalue()  # Warp may print its banner first
            summary = json.loads(text[text.index("{\n") :])
            self.assertEqual(summary["tier"], "medium")
            self.assertTrue(summary["path"].endswith("a.glb"))  # the tier's format
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                main(["export", str(folder / "b"), "--tier", "low"])  # a VRM file
            self.assertIn("needs an author", stderr.getvalue())
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                main(
                    ["export", str(folder / "c.glb"), "--tier", "low", "--detail", "0"]
                )
            with contextlib.redirect_stdout(io.StringIO()):
                main(["portraits", "--out", str(folder / "p"), "--size", "48"])
            self.assertEqual(
                sorted(p.name for p in (folder / "p").iterdir()),
                sorted(
                    ["portrait.png", "expressions.png"]
                    + [
                        f"{n}.png"
                        for n in ("neutral", "smile", "surprised", "sad", "angry")
                    ]
                ),
            )

    def test_export_tier_with_eyes(self):
        """``--eyes`` and ``--teeth`` put the eyes of opensculptboy.eyes and the teeth of
        opensculptboy.teeth into a tier's file."""
        import contextlib

        from opensculptboy.cli import main

        def material_names(path):
            data = pathlib.Path(path).read_bytes()
            length = int.from_bytes(data[12:16], "little")
            return {
                m.get("name") for m in json.loads(data[20 : 20 + length])["materials"]
            }

        with tempfile.TemporaryDirectory() as folder:
            folder = pathlib.Path(folder)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "export",
                            str(folder / "m.glb"),
                            "--tier",
                            "medium",
                            "--eyes",
                            "--teeth",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "export",
                            str(folder / "l.vrm"),
                            "--tier",
                            "low",
                            "--eyes",
                            "--author",
                            "test",
                        ]
                    ),
                    0,
                )
            self.assertLessEqual(
                {"eyes", "cornea", "conjunctiva", "teeth", "gums"},
                material_names(folder / "m.glb"),
            )
            vrm = material_names(folder / "l.vrm")
            # the single-surface MToon eyes of VRM files, and the caruncle's material
            self.assertLessEqual({"eyes", "conjunctiva"}, vrm)
            self.assertNotIn("cornea", vrm)


def _grey(path) -> np.ndarray:
    from PIL import Image

    return (
        np.asarray(Image.open(io.BytesIO(pathlib.Path(path).read_bytes())).convert("L"))
        / 255.0
    )


class TestPortraits(unittest.TestCase):
    def test_expression_sheet(self):
        from PIL import Image

        from opensculptboy.render.sheet import (
            EXPRESSIONS,
            expression_portraits,
            expression_sheet,
        )

        pictures = expression_portraits(Character(), size=96)
        self.assertEqual(list(pictures), [name for name, _, _ in EXPRESSIONS])
        arrays = {
            k: np.asarray(Image.open(io.BytesIO(v)).convert("RGB"))
            for k, v in pictures.items()
        }
        for name, pixels in arrays.items():
            self.assertEqual(pixels.shape, (96, 96, 3))
            if name != "neutral":
                self.assertTrue(np.any(pixels != arrays["neutral"]), name)
        sheet = Image.open(io.BytesIO(expression_sheet(pictures)))
        self.assertEqual(sheet.size[0], 5 * 96)
        self.assertGreater(sheet.size[1], 96)


if __name__ == "__main__":
    unittest.main()

# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of ``opensculptboy wearables import-mhclo`` (``opensculptboy.wearables.mhclo``), the
quality checks (``opensculptboy.wearables.quality``) and the card commands ``wearables add``,
``remove`` and ``outfit`` (``opensculptboy.wearables.cardedit``), with synthetic MakeHuman
assets written by the tests (``test/mhclo_fixtures.py``): the header and binding formats, the
binding on the base mesh and on the default body, the helper geometry against MakeHuman's
targets on random bodies, the recorded licences and their credits, one known defect per
quality check, the JSON results and errors, idempotent card changes, and the package schema.
"""

import contextlib
import io
import json
import pathlib
import tempfile
import unittest

import numpy as np

from opensculptboy import Character, export_glb
from opensculptboy.wearables import read_wearable, write_wearable
from opensculptboy.wearables.export import Dressing, credits
from opensculptboy.wearables.format import PRIVATE, SCHEMA_PATH, WearableError
from opensculptboy.wearables.mhclo import (
    import_mhclo,
    infer_slot,
    read_mhclo,
    spdx_of,
)
from opensculptboy.wearables.quality import QualityError
from test.gltf_reader import GLB, structure_errors
from test.mhclo_fixtures import write_asset
from test.wearable_fixtures import circlet, template_body, template_model, write_outfit

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(args) -> tuple[int, dict, str]:
    """Run the command; returns the exit status, the JSON result and the error stream."""
    from opensculptboy.cli import main

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            status = main(args)
        except SystemExit as exit:
            status = exit.code
    return status, json.loads(out.getvalue()), err.getvalue()


def validate(test, description: dict) -> None:
    try:
        import jsonschema
    except ImportError:  # the dev extra holds it
        return
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(description, schema)


def imported(folder, name, **kw):
    """Write an asset and import it with the module API on the default body."""
    path = write_asset(folder, name, **{k: v for k, v in kw.items() if k in ASSET_KEYS})
    options = {k: v for k, v in kw.items() if k not in ASSET_KEYS}
    report = import_mhclo(
        path, template_body(), Character(), template_model(), **options
    )
    return report, path


ASSET_KEYS = {
    "licence",
    "author",
    "tags",
    "z_depth",
    "uvs",
    "textures",
    "flip_triangle",
    "directx",
    "missing_texture",
    "alpha_map_opaque",
    "texture_size",
    "helper",
    "delete_verts",
    "description",
}


def codes(report):
    return [i["code"] for i in report["quality"]["items"]]


class TestFormat(unittest.TestCase):
    def test_header_and_binding_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "skirt.mhclo"
            path.write_text(
                "# author: Some One\n# license CC-BY 4.0\nname Pleated skirt\n"
                "uuid abc\ntag Skirt\ntag long\nz_depth 30\nobj_file skirt\n"
                "material skirt.mhmat\nx_scale 1 2 3.0\ny_scale 4 5 6.0\n"
                "z_scale 7 8 9.0\nverts 0\n10\n1 2 3 0.2 0.3 0.5\n"
                "4 5 6 0.1 0.2 0.7 0.01 -0.02 0.03\n\ndelete_verts\n5 - 8 12 20-22\n"
            )
            asset = read_mhclo(path)
        self.assertEqual(asset.name, "Pleated skirt")
        self.assertEqual(asset.author, "Some One")
        self.assertEqual(asset.licence, "CC-BY 4.0")
        self.assertEqual(asset.tags, ["skirt", "long"])
        self.assertEqual(asset.z_depth, 30)
        self.assertEqual(asset.obj_file.name, "skirt.obj")
        self.assertEqual(asset.scale, [(1, 2, 3.0), (4, 5, 6.0), (7, 8, 9.0)])
        np.testing.assert_array_equal(asset.refs, [[10, 10, 10], [1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(asset.single, [True, False, False])
        np.testing.assert_allclose(asset.offsets[2], [0.01, -0.02, 0.03])
        np.testing.assert_array_equal(asset.offsets[1], [0, 0, 0])
        self.assertEqual(asset.delete_verts.tolist(), [5, 6, 7, 8, 12, 20, 21, 22])

    def test_description_in_a_comment_line(self):
        """MPFB2 writes '# author:', '# license:' and '# description:' comment lines."""
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "coat.mhclo"
            path.write_text(
                "# author: Some One\n# license: CC0\n# description: A long coat\n"
                "basemesh hm08\nname coat\nverts 0\n10\n"
            )
            self.assertEqual(read_mhclo(path).description, "A long coat")
            path.write_text(path.read_text() + "description The key's text\n")
            self.assertEqual(read_mhclo(path).description, "The key's text")

    def test_malformed_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "bad.mhclo"
            path.write_text("name bad\nverts 0\n1 2 3 0.5\n")
            with self.assertRaises(WearableError) as caught:
                read_mhclo(path)
        self.assertEqual(caught.exception.code, "malformed_binding")

    def test_malformed_files_give_coded_errors(self):
        """A header line with a missing value, a scale reference outside the base mesh, a
        zero scale length and an OBJ face on a missing UV are coded errors (JSON results of
        the command), never tracebacks."""
        import re

        with tempfile.TemporaryDirectory() as tmp:
            path = write_asset(tmp, "shirt")
            text = path.read_text()
            no_value = text.replace("z_depth 50", "z_depth")
            cases = [
                ("malformed_header", no_value),
                ("malformed_header", re.sub(r"(x_scale \d+ \d+) [\d.]+", r"\1", text)),
                ("unknown_basemesh", re.sub(r"x_scale \d+ ", "x_scale 99999 ", text)),
                (
                    "malformed_header",
                    re.sub(r"(x_scale \d+ \d+) [\d.]+", r"\1 0", text),
                ),
            ]
            for k, (code, body) in enumerate(cases):
                with self.subTest(case=k):
                    path.write_text(body)
                    with self.assertRaises(WearableError) as caught:
                        import_mhclo(
                            path, template_body(), Character(), template_model()
                        )
                    self.assertEqual(caught.exception.code, code)
            path.write_text(text)
            obj = path.with_suffix(".obj")
            obj.write_text(obj.read_text() + "f 1/1 2/2 3/999\n")
            with self.assertRaises(WearableError) as caught:
                import_mhclo(path, template_body(), Character(), template_model())
            self.assertEqual(caught.exception.code, "binding_mismatch")
            obj.write_text(obj.read_text() + "v 1.0 2.0\n")
            with self.assertRaises(WearableError) as caught:
                import_mhclo(path, template_body(), Character(), template_model())
            self.assertEqual(caught.exception.code, "malformed_obj")
            path.write_text(no_value)
            status, result, _ = run(["wearables", "import-mhclo", str(path)])
            self.assertEqual((status, result["error"]["code"]), (1, "malformed_header"))
            self.assertIn("line", result["error"]["message"])
            self.assertTrue(result["error"]["hint"])

    def test_licence_names_and_slots(self):
        self.assertEqual(spdx_of("CC0"), "CC0-1.0")
        self.assertEqual(spdx_of("cc0 1.0"), "CC0-1.0")
        self.assertEqual(spdx_of("CC-BY 4.0"), "CC-BY-4.0")
        self.assertEqual(spdx_of("CC BY-SA 3.0"), "CC-BY-SA-3.0")
        self.assertEqual(spdx_of("AGPL3"), "AGPL-3.0-only")
        self.assertIsNone(spdx_of("CC-BY"))  # no version: the text stays in 'declared'
        self.assertIsNone(spdx_of(None))
        self.assertEqual(infer_slot("female_casualsuit01", [])[0], "torso")
        self.assertEqual(infer_slot("boots02", [])[0], "feet")
        self.assertEqual(infer_slot("thing", ["Hat"])[0], "head")
        self.assertEqual(infer_slot("thing", ["clothes"]), (None, None))

    def test_slot_words(self):
        """The slot guess of the garment words, plurals and compound names."""
        cases = {
            "under": "underwear female_underwear01 underpants briefs bra sportsbra "
            "panties boxer",
            "feet": "socks kneesocks boots",
            "legs": "stockings tights fishnettights pantyhose jeans",
            "hands": "gloves glove mittens",
            "head": "hat hats tophat helmet hood",
            "cloak": "cape cloak hooded_cloak capes",
            "neck": "scarf scarves",
            "face": "glasses sunglasses mask facemask",
            "torso": "hoodedjacket zebra_dress tshirtlong",
            "accessory": "bracelet",
        }
        for slot, names in cases.items():
            for name in names.split():
                with self.subTest(name=name):
                    self.assertEqual(infer_slot(name, [])[0], slot)


class TestImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.shirt_path = write_asset(cls.folder, "shirt")
        status, cls.report, cls.err = run(
            ["wearables", "import-mhclo", str(cls.shirt_path), "--out"]
            + [str(cls.folder / "wear" / "shirt"), "--tag", "cotton"]
        )
        assert status == 0, cls.report
        (cls.tights, cls.tights_report), cls.tights_path = imported(
            cls.folder / "src", "tights", helper=True, tags=("tights",), licence="CC-BY"
        )
        write_wearable(cls.folder / "wear" / "tights", cls.tights)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_report_and_package(self):
        r = self.report
        self.assertTrue(r["ok"])
        self.assertEqual(r["slot"], "torso")
        self.assertEqual(r["slot_from"], "the word 'shirt'")
        self.assertEqual(r["layer"], 50)
        self.assertEqual(r["mhclo"]["scale"], "scale")
        self.assertLess(
            r["mhclo"]["source_fit_mm_max"], 1e-3
        )  # the formula and the frame
        self.assertLess(r["round_trip_mm_max"], 1e-6)
        self.assertLess(r["bind_distance_mm_max"], 50)
        self.assertEqual(r["covers_from"], "delete_verts")
        self.assertGreater(r["covered_skin_vertices"], 300)
        self.assertEqual(r["quality"]["errors"], 0)
        self.assertEqual(r["tags"], ["shirt", "long sleeves", "cotton"])
        description = json.loads(
            (self.folder / "wear" / "shirt.wearable.json").read_text()
        )
        validate(self, description)
        self.assertEqual(description["description"], "A synthetic shirt for the tests")
        self.assertEqual(description["licence"]["spdx"], "CC0-1.0")
        self.assertEqual(description["licence"]["author"], "OpenSculptBoy tests")
        self.assertEqual(description["licence"]["source"], "https://example.org/tests")
        glb = GLB(self.folder / "wear" / "shirt.glb")
        self.assertEqual(structure_errors(glb), [])
        attributes = glb.json["meshes"][0]["primitives"][0]["attributes"]
        self.assertIn("TANGENT", attributes)  # a normal map: MikkTSpace tangents
        material = glb.json["materials"][0]
        self.assertTrue(material.get("doubleSided"))
        self.assertAlmostEqual(material["pbrMetallicRoughness"]["roughnessFactor"], 0.7)
        self.assertIn("normalTexture", material)

    def test_helper_geometry_follows_the_targets(self):
        """The tights patch binds vertex by vertex to helper-tights, which Anny's template
        lacks: its vertices come from MakeHuman's targets on the binding body, and on random
        bodies the surface binding stays within millimetres of the targets' helper."""
        from scipy.spatial import cKDTree

        from opensculptboy.wearables import card_body, evaluate
        from opensculptboy.wearables.basemesh import MH_TO_ANNY, base_mesh_vertices
        from opensculptboy.wearables.mhclo import evaluate_binding, referenced

        r = self.tights_report
        self.assertEqual(r["slot"], "legs")
        self.assertEqual(r["mhclo"]["single_bindings"], r["mhclo"]["obj_vertices"])
        self.assertEqual(r["mhclo"]["helper_bound_vertices"]["helper-tights"], 79)
        self.assertEqual(r["mhclo"]["target_evaluated_vertices"], 79)
        self.assertLess(r["bind_distance_mm_max"], 15)
        asset = read_mhclo(self.tights_path)
        model = template_model()
        to_mh = np.linalg.inv(MH_TO_ANNY)

        def exact(card, body):
            h, _ = base_mesh_vertices(card, model, body, referenced(asset))
            return evaluate_binding(asset, h @ to_mh.T) @ MH_TO_ANNY.T

        start, _ = evaluate(self.tights.binding, template_body())
        order = cKDTree(exact(Character(), template_body())).query(start)[1]
        rng = np.random.default_rng(3)
        for k in range(3):
            phenotype = {n: float(rng.uniform(0, 1)) for n in model.phenotype_labels}
            with self.subTest(k=k):
                card = Character(phenotype=phenotype)
                body = card_body(card, model)
                vertices, _ = evaluate(self.tights.binding, body)
                error = np.linalg.norm(vertices - exact(card, body)[order], axis=1)
                self.assertLess(float(np.median(error)), 0.008)
                self.assertLess(float(error.max()), 0.02)

    def test_rows_from_the_targets_match_anny(self):
        """Anny's own rows evaluated from the targets equal the model's rest vertices."""
        from opensculptboy.wearables.basemesh import base_indices, target_rows

        model = template_model()
        card = Character(phenotype={"weight": 0.9, "gender": 0.2})
        rest = model(phenotype_kwargs=card.phenotype)["rest_vertices"][0]
        rows = np.arange(0, 13000, 97)
        base = base_indices(model)[rows]
        np.testing.assert_allclose(
            target_rows(card, model, base), rest.detach().numpy()[rows], atol=1e-9
        )

    def test_licences_are_recorded_as_found(self):
        for licence, spdx in (("AGPL3", "AGPL-3.0-only"), (None, None)):
            with self.subTest(licence=licence):
                (wearable, report), _ = imported(
                    self.folder / "lic", f"top_{licence}", licence=licence
                )
                self.assertEqual(wearable.licence.get("spdx", "absent"), spdx)
                self.assertNotIn("licence", " ".join(report["warnings"]))
                self.assertEqual(wearable.provenance, "public")
                # every licence, or none, with an author gets a credit line
                line = wearable.licence["attribution"]
                self.assertIn("by OpenSculptBoy tests", line)
                if licence:
                    self.assertIn(licence, line)
        self.assertIn("attribution", self.tights.licence)
        self.assertIn("by OpenSculptBoy tests", self.tights.licence["attribution"])

    def test_dressed_export_carries_the_credits(self):
        card = Character(
            outfits=[
                {
                    "id": "casual",
                    "pieces": [
                        {"piece": "shirt", "file": "wear/shirt.wearable.json"},
                        {"piece": "tights", "file": "wear/tights.wearable.json"},
                    ],
                }
            ],
            active_outfit="casual",
        )
        path = self.folder / "dressed.glb"
        export_glb(
            path, card, template_model(), detail_dir=self.folder, dressing=Dressing()
        )
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        names = [n.get("name") for n in glb.json["nodes"]]
        self.assertIn("casual/shirt", names)
        wear = glb.json["scenes"][0]["extras"]["opensculptboy"]["wearables"]
        shirt = read_wearable(self.folder / "wear" / "shirt.wearable.json")
        self.assertIn("CC0", shirt.licence["attribution"])
        self.assertEqual(
            wear["credits"],
            [shirt.licence["attribution"], self.tights.licence["attribution"]],
        )

        class Piece:
            wearable = self.tights
            shown = True

        self.assertEqual(
            credits([Piece(), Piece()]), [self.tights.licence["attribution"]]
        )
        from opensculptboy.export.vrm import _wear_credits

        self.assertEqual(
            _wear_credits([Piece()]),
            [f"Wearable: {self.tights.licence['attribution']}"],
        )

    def test_dry_run_writes_no_files(self):
        out = self.folder / "dry" / "shirt"
        status, result, _ = run(
            ["wearables", "import-mhclo", str(self.shirt_path), "--out", str(out)]
            + ["--dry-run"]
        )
        self.assertEqual(
            (status, result["dry_run"], result["written"]), (0, True, False)
        )
        self.assertEqual(result["slot"], "torso")
        self.assertFalse(out.parent.exists())
        self.assertEqual(
            (self.report["dry_run"], self.report["written"]), (False, True)
        )

    def test_command_errors_are_json(self):
        status, result, err = run(
            ["wearables", "import-mhclo", str(self.folder / "none.mhclo")]
        )
        self.assertEqual(status, 1)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "file_not_found")
        self.assertTrue(result["error"]["hint"])
        self.assertIn("file_not_found", err)
        path = write_asset(self.folder / "x", "thing", tags=())
        status, result, _ = run(["wearables", "import-mhclo", str(path)])
        self.assertEqual(result["error"]["code"], "unknown_slot")
        status, result, _ = run(["wearables", "import-mhclo", str(path), "--bogus"])
        self.assertEqual((status, result["error"]["code"]), (2, "usage"))


class TestCommandErrors(unittest.TestCase):
    def test_every_failure_is_one_json_error(self):
        """A file error and an unexpected error of the JSON commands each print one JSON
        error with a code and a hint, and no traceback; an unexpected one exits with 2."""
        from unittest import mock

        with tempfile.TemporaryDirectory() as folder:
            card = pathlib.Path(folder) / "card.json"
            card.write_text(json.dumps(Character().to_dict()), encoding="utf-8")
            asset = write_asset(pathlib.Path(folder), "shirt")
            cases = (
                (
                    "opensculptboy.wearables.cardedit.remove_pieces",
                    ["wearables", "remove", str(card), "x"],
                ),
                (
                    "opensculptboy.wearables.cardedit.activate_outfit",
                    ["wearables", "outfit", str(card), "--activate", "none"],
                ),
                (
                    "opensculptboy.wearables.mhclo.import_mhclo",
                    ["wearables", "import-mhclo", str(asset), "--dry-run"],
                ),
            )
            for target, args in cases:
                for raised, code, status in (
                    (PermissionError("denied"), "file_error", 1),
                    (RuntimeError("boom"), "internal_error", 2),
                    (ZeroDivisionError("division by zero"), "internal_error", 2),
                ):
                    with self.subTest(command=args[1], error=type(raised).__name__):
                        with mock.patch(target, side_effect=raised):
                            got, result, err = run(args)
                        self.assertEqual(got, status)
                        self.assertFalse(result["ok"])
                        self.assertEqual(result["error"]["code"], code)
                        self.assertIn(str(raised), result["error"]["message"])
                        self.assertTrue(result["error"]["hint"])
                        self.assertNotIn("Traceback", err)


class TestQuality(unittest.TestCase):
    """One synthetic asset per defect."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def item(self, report, code):
        found = [i for i in report["quality"]["items"] if i["code"] == code]
        self.assertTrue(found, f"{code} not in {codes(report)}")
        return found[0]

    def test_flipped_triangle_is_turned_over(self):
        (wearable, report), _ = imported(self.folder, "flip", flip_triangle=True)
        self.assertEqual(self.item(report, "winding_fixed")["count"], 1)
        (_, clean), _ = imported(self.folder, "clean")
        self.assertNotIn("winding_fixed", codes(clean))

    def test_missing_uvs_on_a_textured_material(self):
        path = write_asset(self.folder, "nouv", uvs=False)
        with self.assertRaises(QualityError) as caught:
            import_mhclo(path, template_body(), Character(), template_model())
        self.assertEqual(caught.exception.code, "missing_uvs")
        status, result, _ = run(["wearables", "import-mhclo", str(path)])
        self.assertEqual((status, result["error"]["code"]), (1, "missing_uvs"))
        self.assertIn("--accept-quality", result["error"]["hint"])
        wearable, report = import_mhclo(
            path, template_body(), Character(), template_model(), accept_quality=True
        )
        self.assertIn("textures_dropped_without_uvs", report["quality"]["fixes"])
        self.assertEqual(wearable.garment.materials[0].textures, {})

    def test_directx_normal_map(self):
        (wearable, report), _ = imported(self.folder, "dx", directx=True)
        item = self.item(report, "normal_green_flipped")
        self.assertLess(item["value"], -0.25)
        from PIL import Image

        from opensculptboy.wearables.quality import green_convention

        fixed = Image.open(io.BytesIO(wearable.garment.materials[0].textures["normal"]))
        self.assertGreater(green_convention(fixed), 0.25)
        (_, kept), _ = imported(self.folder, "dx2", directx=True, normal_green="keep")
        self.assertEqual(self.item(kept, "normal_map_directx")["severity"], "warning")

    def test_missing_texture(self):
        (_, report), _ = imported(self.folder, "missing", missing_texture=True)
        item = self.item(report, "texture_missing")
        self.assertEqual((item["severity"], item["slot"]), ("warning", "base_color"))

    def test_alpha_map_on_an_opaque_material(self):
        (wearable, report), _ = imported(self.folder, "alpha", alpha_map_opaque=True)
        self.assertEqual(
            self.item(report, "alpha_map_on_opaque")["severity"], "warning"
        )
        self.assertEqual(wearable.garment.materials[0].alpha_mode, "MASK")
        self.assertIn("alpha_map_merged", report["quality"]["fixes"])

    def test_texture_size(self):
        (_, report), _ = imported(self.folder, "big", texture_size=64, texture_limit=32)
        self.assertEqual(self.item(report, "texture_too_large")["value"], 64)
        (_, odd), _ = imported(self.folder, "odd", texture_size=48)
        self.assertEqual(
            self.item(odd, "texture_not_power_of_two")["slot"], "base_color"
        )

    def test_uv_overlap(self):
        """No overlap for a grid that fills the UV square up to 1.0, nor for 100 small
        islands; the overlap of 5 islands laid twice (4.8 % of the area) is found."""
        from opensculptboy.wearables.format import Garment
        from opensculptboy.wearables.quality import Quality, check_uvs, weld_ids

        def grid(n, u0, u1, v0, v1):
            u, v = np.meshgrid(np.linspace(u0, u1, n + 1), np.linspace(v0, v1, n + 1))
            uv = np.column_stack([u.ravel(), v.ravel()])
            a = (np.arange(n)[None, :] + (n + 1) * np.arange(n)[:, None]).ravel()
            t = np.concatenate(
                [np.c_[a, a + 1, a + n + 2], np.c_[a, a + n + 2, a + n + 1]]
            )
            return uv, t

        def overlap(islands):
            uvs, tris, offset = [], [], 0
            for uv, t in islands:
                uvs.append(uv)
                tris.append(t + offset)
                offset += len(uv)
            uv = np.concatenate(uvs)
            vertices = np.column_stack([uv, np.arange(len(uv)) * 1e-3]) * 0.3
            garment = Garment(vertices, np.concatenate(tris), uvs=uv)
            quality = Quality()
            check_uvs(garment, quality, weld_ids(garment.vertices))
            found = [i for i in quality.items if i.code == "uv_overlap"]
            return found[0].value if found else 0.0

        self.assertEqual(overlap([grid(8, 0, 1, 0, 1)]), 0.0)
        small = [
            grid(2, x / 10, x / 10 + 0.09, y / 10, y / 10 + 0.09)
            for x in range(10)
            for y in range(10)
        ]
        self.assertEqual(overlap(small), 0.0)
        self.assertAlmostEqual(overlap(small + small[:5]), 5 / 105, delta=0.01)

    def test_flipped_uvs(self):
        """Mirrored halves that meet at a seam are no flips; a fold of one vertex is."""
        from opensculptboy.wearables.format import Garment
        from opensculptboy.wearables.quality import Quality, check_uvs, weld_ids

        n = 8
        x, y = np.meshgrid(np.linspace(-1, 1, 2 * n + 1), np.linspace(0, 1, n + 1))
        a = (np.arange(2 * n)[None, :] + (2 * n + 1) * np.arange(n)[:, None]).ravel()
        w = 2 * n + 1
        t = np.concatenate([np.c_[a, a + 1, a + w + 1], np.c_[a, a + w + 1, a + w]])
        vertices = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)]) * 0.2

        def flipped(uv):
            quality = Quality()
            garment = Garment(vertices, t, uvs=uv)
            check_uvs(garment, quality, weld_ids(vertices))
            return sum(i.count for i in quality.items if i.code == "uv_flipped")

        mirrored = np.column_stack([np.abs(x.ravel()), y.ravel()]) * 0.9 + 0.05
        self.assertEqual(flipped(mirrored), 0)
        plain = np.column_stack([(x.ravel() + 1) / 2, y.ravel()]) * 0.9 + 0.05
        self.assertEqual(flipped(plain), 0)
        fold = plain.copy()
        fold[4 * w + 4, 0] += 3 * 0.9 / (2 * n)  # past its right neighbours
        self.assertGreater(flipped(fold), 0)
        self.assertLess(flipped(fold), 6)

    def test_geometry_fixes(self):
        from opensculptboy.wearables.format import Garment
        from opensculptboy.wearables.quality import check_garment

        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], float)
        v = np.vstack([v, [[0.5, 0.5, 1.0]]])
        t = np.array([[0, 1, 2], [1, 3, 2], [4, 3, 2], [0, 0, 1], [2, 1, 3], [1, 2, 5]])
        garment = Garment(v * 0.1, t, normals=np.tile([0, 0, 1.0], (6, 1)))
        fixed, rows, quality = check_garment(garment)
        found = {i.code: i for i in quality.items}
        self.assertIn("degenerate_triangles_removed", found)
        self.assertIn("duplicate_vertices_merged", found)
        # 4-3-2 is 1-3-2 once 4 merges into 1, and 2-1-3 is 1-3-2 in another turn
        self.assertEqual(found["duplicate_triangles_removed"].count, 2)
        self.assertEqual(len(fixed.triangles), 3)
        self.assertIn("non_manifold_edges", found)  # 1-2 shared by three triangles
        self.assertEqual(len(fixed.vertices), 5)
        self.assertEqual(sorted(rows.tolist()), [0, 1, 2, 3, 5])


class TestCardCommands(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        write_outfit(cls.folder / "wear")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def card(self, name="card.json", data=None) -> pathlib.Path:
        path = self.folder / name
        data = Character(name="ada").to_dict() if data is None else data
        path.write_text(json.dumps(data, indent=2) + "\n")
        return path

    def test_add_is_idempotent_and_keeps_the_card(self):
        card = self.card()
        wear = [
            str(self.folder / "wear" / f"{n}.wearable.json") for n in ("tunic", "cape")
        ]
        args = ["wearables", "add", str(card), *wear, "--outfit", "travel"]
        args += ["--label", "Travel clothes", "--activate"]
        status, dry, _ = run(args + ["--dry-run"])
        self.assertEqual((status, dry["changed"], dry["written"]), (0, True, False))
        self.assertEqual(json.loads(card.read_text())["schema"], 1)
        status, first, _ = run(args)
        self.assertEqual((status, first["changed"], first["written"]), (0, True, True))
        self.assertEqual(first["schema"], {"before": 1, "after": 4})
        data = json.loads(card.read_text())
        self.assertEqual(data["name"], "ada")
        self.assertEqual(data["active_outfit"], "travel")
        self.assertEqual(
            data["outfits"][0]["pieces"][0],
            {"piece": "tunic", "file": "wear/tunic.wearable.json", "enabled": True},
        )
        self.assertEqual(dry["wardrobe"], first["wardrobe"])
        before = card.read_text()
        status, second, _ = run(args)
        self.assertEqual((status, second["changed"]), (0, False))
        self.assertEqual(second["unchanged"], ["travel/tunic", "travel/cape"])
        self.assertEqual(card.read_text(), before)
        Character.from_dict(json.loads(before))

    def test_loose_replace_and_errors(self):
        card = self.card("loose.json")
        tunic = str(self.folder / "wear" / "tunic.wearable.json")
        status, result, _ = run(["wearables", "add", str(card), tunic])
        self.assertEqual(result["added"][0]["ref"], "tunic")
        other = self.folder / "other"
        w = read_wearable(tunic)
        write_wearable(other / "tunic", w)
        status, result, _ = run(
            ["wearables", "add", str(card), str(other / "tunic.wearable.json")]
        )
        self.assertEqual((status, result["error"]["code"]), (1, "piece_exists"))
        status, result, _ = run(
            [
                "wearables",
                "add",
                str(card),
                str(other / "tunic.wearable.json"),
                "--replace",
            ]
        )
        self.assertEqual(result["updated"][0]["file"], "other/tunic.wearable.json")
        status, result, _ = run(
            ["wearables", "add", str(card), tunic, "--loose", "--outfit", "x"]
        )
        self.assertEqual(result["error"]["code"], "bad_option")

    def test_remove_and_outfit(self):
        card = self.card("remove.json")
        wear = [
            str(self.folder / "wear" / f"{n}.wearable.json") for n in ("tunic", "cape")
        ]
        run(["wearables", "add", str(card), *wear, "--outfit", "travel", "--activate"])
        run(["wearables", "add", str(card), wear[0], "--outfit", "work"])
        status, result, _ = run(["wearables", "remove", str(card), "cape", "--dry-run"])
        self.assertEqual(
            (result["removed"], result["written"]), (["travel/cape"], False)
        )
        status, result, _ = run(["wearables", "remove", str(card), "cape"])
        self.assertEqual(result["removed"], ["travel/cape"])
        status, again, err = run(["wearables", "remove", str(card), "cape"])
        self.assertEqual(
            (status, again["changed"], again["missing"]), (0, False, ["cape"])
        )
        self.assertIn("not on the card", err)
        status, result, _ = run(
            ["wearables", "outfit", str(card), "--activate", "work"]
        )
        self.assertEqual(result["wardrobe"]["active_outfit"], "work")
        status, result, _ = run(
            ["wearables", "outfit", str(card), "--activate", "gala"]
        )
        self.assertEqual(result["error"]["code"], "unknown_outfit")
        status, result, _ = run(["wearables", "remove", str(card), "--outfit", "work"])
        self.assertNotIn("active_outfit", result["wardrobe"])
        Character.from_dict(json.loads(card.read_text()))

    def test_remove_bare_ids_with_no_outfit_worn(self):
        """With ``active_outfit`` set to none, a bare id names the piece of that id in any
        outfit, and a piece of several outfits needs its outfit."""
        card = self.card("none_worn.json")
        wear = [
            str(self.folder / "wear" / f"{n}.wearable.json") for n in ("tunic", "cape")
        ]
        run(["wearables", "add", str(card), *wear, "--outfit", "travel"])
        run(["wearables", "add", str(card), wear[0], "--outfit", "work"])
        run(["wearables", "outfit", str(card), "--activate", "none"])
        status, result, _ = run(["wearables", "remove", str(card), "cape"])
        self.assertEqual((status, result["removed"]), (0, ["travel/cape"]))
        status, result, _ = run(["wearables", "remove", str(card), "tunic"])
        self.assertEqual((status, result["error"]["code"]), (1, "ambiguous_piece"))
        status, result, _ = run(["wearables", "remove", str(card), "work/tunic"])
        self.assertEqual(result["removed"], ["work/tunic"])
        status, result, _ = run(["wearables", "remove", str(card), "tunic"])
        self.assertEqual(result["removed"], ["travel/tunic"])

    def test_overrides_of_removed_pieces_go_with_them(self):
        """An override that names a piece by its bare id goes when the piece goes, so that
        the card still exports."""
        from opensculptboy.wearables.wardrobe import worn_pieces

        card = self.card("overrides.json")
        wear = [
            str(self.folder / "wear" / f"{n}.wearable.json") for n in ("tunic", "cape")
        ]
        run(["wearables", "add", str(card), *wear, "--outfit", "travel", "--activate"])
        data = json.loads(card.read_text())
        data["wearable_overrides"] = {"cape": {"enabled": False}}
        card.write_text(json.dumps(data))
        status, result, _ = run(["wearables", "remove", str(card), "travel/cape"])
        self.assertEqual((status, result["removed_overrides"]), (0, ["cape"]))
        after = Character.from_dict(json.loads(card.read_text()))
        self.assertEqual(after.wearable_overrides, {})
        self.assertEqual(
            [p.ref for p in worn_pieces(after, self.folder)], ["travel/tunic"]
        )

    def test_card_with_non_ascii_text_under_an_ascii_locale(self):
        """``wearables add`` reads and writes a UTF-8 card with non-ASCII text in a process
        whose locale encoding is ASCII (C locale, no UTF-8 mode, no locale coercion)."""
        import os
        import subprocess
        import sys

        name = "Caf" + chr(0xE9) + " " + chr(0x2014) + " " + chr(0x65C5) + chr(0x306E)
        card = self.folder / "non_ascii.json"
        data = Character(name=name).to_dict()
        card.write_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"))
        tunic = self.folder / "wear" / "tunic.wearable.json"
        code = (
            "import locale, sys\n"
            "from opensculptboy.cli import main\n"
            "if 'utf' in locale.getpreferredencoding(False).lower().replace('-', ''):\n"
            "    print('SKIP'); sys.exit(0)\n"
            "sys.exit(main(['wearables', 'add', sys.argv[1], sys.argv[2]]))\n"
        )
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("LC_", "PYTHONIOENCODING", "PYTHONUTF8"))
        }
        env.update(
            LC_ALL="C",
            LANG="C",
            PYTHONCOERCECLOCALE="0",
            PYTHONUTF8="0",
            PYTHONPATH=os.pathsep.join(
                [str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")]
            ),
        )
        result = subprocess.run(
            [sys.executable, "-X", "utf8=0", "-c", code, str(card), str(tunic)],
            capture_output=True,
            env=env,
            cwd=ROOT,
            timeout=600,
        )
        out = result.stdout.decode("utf-8", "replace")
        if out.startswith("SKIP"):
            self.skipTest("the locale encoding stays UTF-8 here")
        self.assertEqual(
            result.returncode, 0, result.stderr.decode("utf-8", "replace")[-3000:]
        )
        written = json.loads(card.read_bytes().decode("utf-8"))
        self.assertEqual(written["name"], name)
        self.assertEqual(written["wearables"][0]["file"], "wear/tunic.wearable.json")

    def test_private_package_never_goes_on_a_card_in_the_repository(self):
        w = circlet()
        w.provenance = PRIVATE
        write_wearable(self.folder / "private" / "circlet", w)
        with tempfile.TemporaryDirectory(dir=ROOT) as inside:
            card = pathlib.Path(inside) / "card.json"
            card.write_text(json.dumps(Character().to_dict()))
            status, result, _ = run(
                ["wearables", "add", str(card)]
                + [str(self.folder / "private" / "circlet.wearable.json")]
            )
        self.assertEqual(
            (status, result["error"]["code"]), (1, "private_in_repository")
        )
        outside = self.card("outside.json")
        status, result, _ = run(
            ["wearables", "add", str(outside)]
            + [str(self.folder / "private" / "circlet.wearable.json")]
        )
        self.assertEqual(status, 0)

    def test_packages_follow_the_schema(self):
        for name in ("tunic", "cape", "circlet"):
            with self.subTest(name=name):
                data = json.loads(
                    (self.folder / "wear" / f"{name}.wearable.json").read_text()
                )
                validate(self, data)
                self.assertEqual(data["tags"], [])


if __name__ == "__main__":
    unittest.main()

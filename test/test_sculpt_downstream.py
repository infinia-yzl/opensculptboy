# OpenSculptBoy
# Apache License, Version 2.0
"""
The downstream parts on the ``sculpt`` topology (docs/sculpt-topology.md, workstream 4), with
the test-only placeholder patch composed in on the back of the head: the base-mesh lookups of
new vertices (``opensculptboy.topology.lookup``, the fit's masks and regions, the COCO
keypoints), the ear root line of the ear split, the GLB and VRM exports of a sculpt card (the
VRM structure, the budgets and the skin-weight edits of anny carried over), a synthetic
reference fit on sculpt, and ``opensculptboy convert``.
"""

import contextlib
import json
import pathlib
import tempfile
import unittest
import warnings

import numpy as np
import torch

from opensculptboy import Character, export_glb
from opensculptboy.topology import base_quad_mesh, compose, use_sculpt_data
from opensculptboy.topology.patches import placeholder
from opensculptboy.topology.sculpt import MAKEHUMAN_VERTEX_COUNT, compose_patches
from test.gltf_reader import GLB, structure_errors

# The placeholder patch on the back of the head (scalp), inside the head region of the detail
# layer, the head tile and the fitted skin.
HEAD_TARGET = [0.03, 0.12, 1.62]
LEVEL1_BOUND_MM = 0.5


def head_patch_data(labels=None):
    """The placeholder patch moved to the back of the head; ``labels`` sets the vertex labels
    of its new vertices."""
    base = base_quad_mesh()
    patch = placeholder.build(base, target=HEAD_TARGET)
    if labels is not None:
        patch.vertex_labels = np.array([labels] * patch.vertex_count, dtype=object)
    return compose_patches([patch], base, [placeholder])


class TestLookups(unittest.TestCase):
    """Base-mesh tables on the new vertices, through their anchors."""

    @classmethod
    def setUpClass(cls):
        import anny

        from opensculptboy.fit.register import fit_model

        cls.data = head_patch_data()
        cls.anny = fit_model("anny")
        cls.stack = contextlib.ExitStack()
        cls.stack.enter_context(use_sculpt_data(cls.data))
        cls.model = fit_model("sculpt")
        cls.plain = anny.Anny().to(torch.float64)
        cls.n = cls.data.vertex_count

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()

    def test_patch_is_on_the_head(self):
        centre = self.model.template_vertices[self.n :].double().mean(0).numpy()
        self.assertGreater(centre[2], 1.4)
        self.assertGreater(centre[1], 0.0)

    def test_base_mesh_rows(self):
        from opensculptboy.topology.lookup import (
            base_mesh_rows,
            base_mesh_values,
            model_topology,
            patch_labels,
            strongest_anchor,
        )

        base = self.model.base_mesh_vertex_indices.numpy()
        self.assertTrue((base[self.n :] >= MAKEHUMAN_VERTEX_COUNT).all())
        rows = base_mesh_rows(self.model)
        np.testing.assert_array_equal(rows[: self.n], base[: self.n])
        self.assertTrue((rows < MAKEHUMAN_VERTEX_COUNT).all())
        strongest = strongest_anchor(self.model)
        np.testing.assert_array_equal(rows[self.n :], base[strongest[self.n :]])
        # a numeric table blends: the base-mesh index itself as a table of positions
        table = np.zeros((MAKEHUMAN_VERTEX_COUNT, 3))
        table[base[: self.n]] = self.model.template_vertices[: self.n].double().numpy()
        values = base_mesh_values(self.model, table)
        np.testing.assert_allclose(
            values, self.model.template_vertices.double().numpy(), atol=1e-12
        )
        self.assertEqual(set(patch_labels(self.model)[self.n :]), {"placeholder"})
        self.assertEqual(model_topology(self.model), "sculpt")
        self.assertEqual(model_topology(self.anny), "anny")
        # anny's model is the plain lookup
        np.testing.assert_array_equal(
            base_mesh_rows(self.anny), self.anny.base_mesh_vertex_indices.numpy()
        )

    def test_keypoints(self):
        from anny.keypoints import KeypointsRegressor

        a = KeypointsRegressor.coco(self.anny).regression_weights
        s = KeypointsRegressor.coco(self.model).regression_weights
        torch.testing.assert_close(s[:, : self.n], a, rtol=0, atol=0)
        self.assertEqual(float(s[:, self.n :].abs().max()), 0.0)

    def test_fit_masks_and_regions(self):
        from opensculptboy.fit import metrics
        from opensculptboy.fit.register import excluded_vertices, lip_margin
        from opensculptboy.topology.lookup import referenced_vertices

        used = referenced_vertices(self.model)
        self.assertTrue(used.all(), "the placeholder keeps every corner of its quads")
        a, s = excluded_vertices(self.anny), excluded_vertices(self.model)
        np.testing.assert_array_equal(s[: self.n], a)
        self.assertFalse(s[self.n :].any(), "the scalp patch is fitted")
        np.testing.assert_array_equal(
            lip_margin(self.model)[: self.n], lip_margin(self.anny)
        )
        ra, rs = metrics.anny_regions(self.anny), metrics.anny_regions(self.model)
        np.testing.assert_array_equal(rs[: self.n], ra)
        self.assertEqual(set(rs[self.n :].tolist()), {metrics.REGIONS.index("scalp")})

    def test_patch_roles(self):
        from opensculptboy.fit import register

        register.PATCH_ROLES["placeholder"] = "exclude"
        try:
            s = register.excluded_vertices(self.model)
        finally:
            del register.PATCH_ROLES["placeholder"]
        self.assertTrue(s[self.n :].all())
        np.testing.assert_array_equal(
            s[: self.n], register.excluded_vertices(self.anny)
        )

    def test_pinna_and_root_line(self):
        from opensculptboy.fit.register import mask_border, pinna
        from opensculptboy.topology.lookup import ear_root_line

        rest = self.anny.template_vertices.double().numpy()
        ears = pinna(self.anny, rest)
        self.assertGreater(ears.sum(), 100)
        np.testing.assert_array_equal(
            pinna(self.model, self.model.template_vertices.double().numpy())[: self.n],
            ears,
        )
        # a root line cuts the ear off the head: the border of the ears gives the ears back
        T = self.anny.faces.numpy()
        root = mask_border(T, ears)
        np.testing.assert_array_equal(pinna(self.anny, rest, root=root), ears)
        # no line on this mesh; the ear-root patch's labelled vertices form one
        self.assertIsNone(ear_root_line(self.model))
        with use_sculpt_data(head_patch_data("ear_root/line")):
            labelled = Character(topology="sculpt").build_model()
            line = ear_root_line(labelled)
        self.assertEqual(int(line.sum()), self.data.new_vertex_count)
        self.assertTrue(line[self.n :].all())


class TestSculptExports(unittest.TestCase):
    """GLB and VRM files of a sculpt card, at level 0 and on a detail level."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.data = head_patch_data()
        cls.stack = contextlib.ExitStack()
        cls.stack.enter_context(use_sculpt_data(cls.data))
        cls.character = Character(
            name="sculpt",
            topology="sculpt",
            phenotype={"age": 0.3, "weight": 0.7},
            face_shapes={"nose-scale-horiz": 0.4},
        )
        cls.model = cls.character.build_model(face_shapes=True, dtype=torch.float64)
        cls.n = cls.data.vertex_count

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()
        cls.tmp.cleanup()

    def export_vrm(self, version, **options):
        from opensculptboy.export.vrm import export_vrm

        path = self.folder / f"sculpt_{version}_{len(options)}.vrm"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            summary = export_vrm(
                path,
                self.character,
                self.model,
                version=version,
                author="tests",
                budget="strict",
                **options,
            )
        return path, summary, caught

    def test_skin_weight_edits_carry_over(self):
        from opensculptboy.export import vrm_tables

        data = vrm_tables.skin_weight_data(self.model)
        self.assertIsNotNone(data, "the edits of anny match sculpt's anny rows")
        import anny

        plain = anny.Anny(face_shapes="all").to(torch.float64)
        self.assertEqual(
            vrm_tables.skin_weight_digest(self.model),
            vrm_tables.skin_weight_digest(plain),
        )
        # with the edits, new rows are the blends of their anchors' edited rows
        w, i = vrm_tables.vrm_skin_weights(self.model, edits=True)
        wa, ia = vrm_tables.vrm_skin_weights(plain, edits=True)
        k = wa.shape[1]
        np.testing.assert_array_equal(w[: self.n, :k], wa)
        np.testing.assert_array_equal(i[: self.n, :k], ia)
        self.assertFalse(w[: self.n, k:].any())
        np.testing.assert_allclose(w.sum(1), 1.0, atol=1e-12)

    def test_blend_new_rows(self):
        from opensculptboy.export.vrm_tables import blend_new_rows

        J = len(self.model.bone_labels)
        V = len(self.model.template_vertices)
        weights = np.zeros((V, 2))
        indices = np.zeros((V, 2), np.int64)
        weights[:, 0] = 1.0
        before = (weights.copy(), indices.copy())
        anchors = self.data.anchors[self.data.weights > 0]
        weights[anchors, 0] = 0.25
        weights[anchors, 1] = 0.75
        indices[anchors, 1] = J - 1
        w, i = blend_new_rows(self.model, weights, indices, before)
        np.testing.assert_array_equal(i[self.n :, 0], J - 1)
        np.testing.assert_allclose(w[self.n :], [[0.75, 0.25]] * (V - self.n))
        np.testing.assert_array_equal(w[: self.n], weights[: self.n])

    def test_glb(self):
        path = self.folder / "sculpt.glb"
        export_glb(path, self.character, self.model)
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])

    def test_vrm_files(self):
        from opensculptboy.export.budget import BUDGETS
        from opensculptboy.export.document import read_gltf_json
        from test.test_vrm_schema import check_vrm0, check_vrm1

        for version, check, budget in (
            ("1.0", check_vrm1, "vrm1"),
            ("0.x", check_vrm0, "vrm0"),
        ):
            with self.subTest(version=version):
                path, summary, caught = self.export_vrm(version)
                messages = [str(w.message) for w in caught]
                self.assertFalse([m for m in messages if "skin-weight" in m], messages)
                doc = read_gltf_json(path)
                self.assertEqual(check(doc), [])
                self.assertEqual(structure_errors(GLB(path)), [])
                self.assertLessEqual(summary["triangles"], BUDGETS[budget]["triangles"])
                extras = json.dumps(doc["scenes"][0].get("extras", {}))
                self.assertIn('"sculpt"', extras)
                self.assertNotIn("no data", extras)

    def test_eyes(self):
        # the eyes of opensculptboy.eyes on a sculpt mesh, at level 0 and on level 1
        from opensculptboy.detail import level, save_offsets
        from opensculptboy.export.document import read_gltf_json
        from test.test_vrm_schema import check_vrm1

        offsets = np.zeros((level("sculpt", 1).vertex_count, 3))
        save_offsets(self.folder / "e.detail1.safetensors", offsets, 1, "sculpt")
        card = Character.from_dict(
            {
                **self.character.to_dict(),
                "material": {"eyes": {}},
                "detail": {
                    "levels": [{"subdiv": 1, "offsets": "e.detail1.safetensors"}]
                },
            }
        )
        for detail_level in (0, 1):
            with self.subTest(detail_level=detail_level):
                path = self.folder / f"eyes{detail_level}.glb"
                export_glb(
                    path,
                    card,
                    self.model,
                    detail_level=detail_level,
                    detail_dir=self.folder,
                )
                self.assertEqual(structure_errors(GLB(path)), [])
        from opensculptboy.export.vrm import export_vrm

        path = self.folder / "eyes.vrm"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            export_vrm(path, card, self.model, author="tests", refine_weights=False)
        self.assertEqual(check_vrm1(read_gltf_json(path)), [])

    def test_detail_level_within_budget(self):
        from opensculptboy.detail import level, save_offsets
        from opensculptboy.export.budget import BUDGETS, BudgetWarning
        from opensculptboy.export.document import read_gltf_json
        from test.test_vrm_schema import check_vrm1

        offsets = np.zeros((level("sculpt", 1).vertex_count, 3))
        save_offsets(self.folder / "c.detail1.safetensors", offsets, 1, "sculpt")
        card = Character.from_dict(
            {
                **self.character.to_dict(),
                "detail": {
                    "levels": [{"subdiv": 1, "offsets": "c.detail1.safetensors"}]
                },
            }
        )
        from opensculptboy.export.vrm import export_vrm

        path = self.folder / "detail.vrm"
        with self.assertWarns(BudgetWarning):
            summary = export_vrm(
                path,
                card,
                self.model,
                author="tests",
                detail_level=1,
                detail_dir=self.folder,
                budget="strict",
                refine_weights=False,
            )
        self.assertLessEqual(summary["triangles"], BUDGETS["vrm1"]["triangles"])
        doc = read_gltf_json(path)
        self.assertEqual(check_vrm1(doc), [])
        detail = doc["scenes"][0]["extras"]["detail"]
        self.assertEqual((detail["subdiv"], detail["region"]), (1, "head"))
        glb = self.folder / "detail.glb"
        result = export_glb(
            glb, card, self.model, detail_level=1, detail_dir=self.folder
        )
        self.assertEqual(result["detail"]["subdiv"], 1)
        self.assertEqual(structure_errors(GLB(glb)), [])


class TestSculptFit(unittest.TestCase):
    """A synthetic reference fit on sculpt (levels 0 and 1)."""

    @classmethod
    def setUpClass(cls):
        from opensculptboy.fit.handoff import load_handoff
        from opensculptboy.fit.reference import Reference
        from opensculptboy.fit.register import FitOptions, fit_reference
        from test.fit_fixtures import synthetic_target
        from opensculptboy.fit.handoff import write_handoff

        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        target = synthetic_target(1)
        write_handoff(
            cls.folder / "handoff",
            target["vertices"],
            target["triangles"],
            dict(target["landmarks"]),
        )
        cls.reference = Reference.from_handoff(load_handoff(cls.folder / "handoff"))
        cls.data = head_patch_data()
        cls.stack = contextlib.ExitStack()
        cls.stack.enter_context(use_sculpt_data(cls.data))
        cls.result = fit_reference(
            cls.reference,
            detail=1,
            name="sculpt",
            options=FitOptions(topology="sculpt"),
        )
        cls.card_path = cls.result.write(cls.folder / "card.json")
        print(
            "\nsculpt fit: "
            + ", ".join(
                f"{k}: forward {v['forward']['mean']:.3f} reverse {v['reverse']['mean']:.3f} mm"
                for k, v in cls.result.errors.items()
            )
            + f" ({sum(cls.result.seconds.values()):.0f} s)"
        )

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()
        cls.tmp.cleanup()

    def test_card_is_on_sculpt(self):
        from opensculptboy.detail import character_offsets, level

        card = Character.from_dict(json.loads(self.card_path.read_text()))
        self.assertEqual(card.topology, "sculpt")
        for subdiv in (0, 1):
            offsets = character_offsets(card, subdiv, self.folder)
            self.assertEqual(len(offsets), level("sculpt", subdiv).vertex_count)

    def test_level1_error(self):
        errors = self.result.errors
        self.assertLess(errors[1]["forward"]["mean"], LEVEL1_BOUND_MM)
        self.assertLess(errors[1]["reverse"]["mean"], LEVEL1_BOUND_MM)
        self.assertLess(
            errors[1]["reverse"]["mean"], errors["parametric"]["reverse"]["mean"]
        )

    def test_patch_is_fitted(self):
        # the new vertices of level 0 lie on the reference as the skin around them does
        fitted, _ = self.result.meshes[0]
        n = self.data.vertex_count
        _, _, d, _, _ = self.reference.closest(fitted[n:])
        self.assertLess(float(np.max(d)), 0.002)

    def test_measures(self):
        final = self.result.measures[1]["final"]["all"]
        self.assertLess(final["reverse_mm"]["mean"], LEVEL1_BOUND_MM)

    def test_fitted_card_exports(self):
        card = Character.from_dict(json.loads(self.card_path.read_text()))
        path = self.folder / "fitted.glb"
        summary = export_glb(path, card, detail_level=1, detail_dir=self.folder)
        self.assertEqual(summary["detail"]["subdiv"], 1)
        self.assertEqual(structure_errors(GLB(path)), [])

    def test_wrong_model_topology_is_refused(self):
        from opensculptboy.fit.register import fit_levels, fit_model

        with self.assertRaises(ValueError):
            fit_levels(
                self.reference,
                self.result.character,
                self.result.rotation,
                self.result.translation,
                detail=0,
                model=fit_model("anny"),
            )


class TestConvertCommand(unittest.TestCase):
    """``opensculptboy convert`` between anny and sculpt cards."""

    @classmethod
    def setUpClass(cls):
        from opensculptboy.detail import level, save_offsets
        from opensculptboy.texture.ears import EarUVs

        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.data = head_patch_data()
        cls.stack = contextlib.ExitStack()
        cls.stack.enter_context(use_sculpt_data(cls.data))
        rng = np.random.default_rng(0)
        cls.offsets = {}
        levels = []
        for subdiv in (0, 1):
            offsets = rng.normal(0, 1e-3, (level("anny", subdiv).vertex_count, 3))
            name = f"src/card.detail{subdiv}.safetensors"
            (cls.folder / "src").mkdir(exist_ok=True)
            save_offsets(cls.folder / name, offsets, subdiv, "anny")
            cls.offsets[subdiv] = offsets
            levels.append(
                {"subdiv": subdiv, "offsets": f"card.detail{subdiv}.safetensors"}
            )
        uv = np.full((level("anny", 1).vertex_count, 2), np.nan)
        uv[:500] = rng.uniform(0, 1, (500, 2))
        EarUVs(uv, 1).save(cls.folder / "src" / "card.ear_uvs.safetensors", "anny")
        (cls.folder / "src" / "card.base_color.png").write_bytes(b"\x89PNG")
        card = Character(
            name="convert",
            phenotype={"age": 0.6},
            detail={"levels": levels},
            material={
                "base_color": "card.base_color.png",
                "ears": {"uvs": "card.ear_uvs.safetensors"},
            },
        )
        cls.source = cls.folder / "src" / "card.json"
        cls.source.write_text(json.dumps(card.to_dict(), indent=2))

    @classmethod
    def tearDownClass(cls):
        cls.stack.close()
        cls.tmp.cleanup()

    def run_cli(self, *argv):
        import contextlib as cl
        import io

        from opensculptboy.cli import main

        out = io.StringIO()
        with cl.redirect_stdout(out):
            code = main(list(argv))
        return code, out.getvalue()

    def test_round_trip(self):
        from opensculptboy.detail import character_offsets
        from opensculptboy.texture.ears import card_ear_uvs
        from opensculptboy.topology.convert import LevelCorrespondence

        target = self.folder / "out" / "sculpt.json"
        code, text = self.run_cli(
            "convert", str(self.source), "--to", "sculpt", "--out", str(target)
        )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(text)["levels"], [0, 1])
        card = Character.from_dict(json.loads(target.read_text()))
        self.assertEqual(card.topology, "sculpt")
        self.assertEqual(card.material["base_color"], "../src/card.base_color.png")
        n = self.data.vertex_count
        for subdiv in (0, 1):
            offsets = character_offsets(card, subdiv, target.parent)
            c = LevelCorrespondence(subdiv, data=self.data)
            shared = c.sculpt_to_anny >= 0
            np.testing.assert_allclose(
                offsets[shared],
                self.offsets[subdiv][c.sculpt_to_anny[shared]],
                atol=1e-9,
            )
            if subdiv == 0:
                self.assertEqual(len(offsets), n + self.data.new_vertex_count)
        table = card_ear_uvs(card, target.parent)
        self.assertEqual(len(table.uv), len(c.sculpt_to_anny))
        back = self.folder / "back" / "anny.json"
        code, _ = self.run_cli(
            "convert", str(target), "--to", "anny", "--out", str(back)
        )
        self.assertEqual(code, 0)
        card = Character.from_dict(json.loads(back.read_text()))
        self.assertEqual(card.topology, "anny")
        for subdiv in (0, 1):
            offsets = character_offsets(card, subdiv, back.parent)
            same = LevelCorrespondence(subdiv, data=self.data).anny_to_sculpt >= 0
            np.testing.assert_allclose(
                offsets[same], self.offsets[subdiv][same], atol=1e-9
            )
        # the converted card exports
        path = self.folder / "converted.glb"
        sculpt = Character.from_dict(json.loads(target.read_text()))
        sculpt.material = {}
        export_glb(path, sculpt, detail_level=1, detail_dir=target.parent)
        self.assertEqual(structure_errors(GLB(path)), [])

    def test_refusals(self):
        with self.assertRaises(SystemExit):
            self.run_cli("convert", str(self.source), "--to", "smpl", "--out", "x.json")
        with self.assertRaises(SystemExit):
            self.run_cli(
                "convert", str(self.source), "--to", "sculpt", "--out", str(self.source)
            )


class TestAnnyUnchanged(unittest.TestCase):
    def test_anny_card_keeps_its_topology(self):
        from opensculptboy.fit.register import FitOptions

        self.assertEqual(FitOptions().topology, "anny")
        self.assertEqual(Character().to_dict().get("topology", "anny"), "anny")

    def test_shipped_file_still_loads(self):
        # every test above composes its own data; the shipped file is untouched
        from opensculptboy.topology import sculpt_data

        self.assertEqual(sculpt_data().new_vertex_count, compose().new_vertex_count)


if __name__ == "__main__":
    unittest.main()

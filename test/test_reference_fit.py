# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the reference fit (``opensculptboy.fit``) on a synthetic target of another topology
(``test.fit_fixtures.synthetic_target``: random anny parameters, a small pose, smooth random
offsets on levels 0 and 1, remeshed by sliding and vertex clustering), read through an
appearance handoff folder as the Dragon's Dogma 2 tools write it.

``TestSyntheticFit`` fits once (about 70 s on two cores) and checks the mean surface error of
the detail layer's level 1 against the target in both directions (under 0.5 mm), the card
written with its offsets files, the reproduction of the fit from the card alone, and the
exports of the fitted card. ``TestAutoLandmarks`` fits the same target without landmarks.
The measured errors print to stderr; ``docs/reference-fit.md`` quotes them.
"""

import json
import pathlib
import sys
import tempfile
import unittest
import warnings

import numpy as np
import torch

from opensculptboy import Character, export_glb
from opensculptboy.detail import DetailModel, character_offsets, level
from opensculptboy.export.budget import BUDGETS
from opensculptboy.fit.handoff import load_handoff, part_role, write_handoff
from opensculptboy.fit.reference import Reference, canonical_landmark
from opensculptboy.fit.register import FitOptions, excluded_vertices, fit_reference
from opensculptboy.fit.residual import surface_error
from test.fit_fixtures import synthetic_target

# the bound of the mean surface error of level 1, both ways (mm)
LEVEL1_BOUND_MM = 0.5


def _report(title: str, errors: dict) -> None:
    lines = [title]
    for key, value in errors.items():
        lines.append(
            f"  {key}: forward {value['forward']['mean']:.3f} mm mean "
            f"(p95 {value['forward']['p95']:.3f}), reverse "
            f"{value['reverse']['mean']:.3f} mm mean (p95 {value['reverse']['p95']:.3f})"
        )
    print("\n" + "\n".join(lines), file=sys.stderr)


class TestLandmarkNames(unittest.TestCase):
    def test_loose_names(self):
        for name, expected in (
            ("eye_outer_l", "eye_outer_l"),
            ("LeftEyeOuter", "eye_outer_l"),
            ("eye.outer.L", "eye_outer_l"),
            ("right_mouth_corner", "mouth_corner_r"),
            ("nose", "nose_tip"),
            ("head_top", "crown"),
            ("Ankle.R", "ankle_r"),
            ("neck", "neck_base"),
            ("tail", None),
        ):
            with self.subTest(name=name):
                self.assertEqual(canonical_landmark(name), expected)

    def test_part_roles(self):
        self.assertEqual(part_role("body"), "skin")
        self.assertEqual(part_role("Hair_Long"), "other")
        self.assertEqual(part_role("eye_L"), "other")
        self.assertEqual(part_role("eyelid"), "skin")
        self.assertEqual(part_role("anything", {"role": "hair"}), "hair")
        self.assertEqual(part_role("hair_cap", {"role": "head"}), "skin")


class TestHandoff(unittest.TestCase):
    def test_round_trip_in_anny_frame(self):
        rng = np.random.default_rng(3)
        vertices = rng.normal(size=(30, 3))
        triangles = rng.integers(0, 30, size=(40, 3))
        triangles = triangles[(np.diff(np.sort(triangles, 1), axis=1) > 0).all(1)]
        landmarks = {"nose_tip": np.array([0.1, -0.2, 1.6])}
        with tempfile.TemporaryDirectory() as tmp:
            write_handoff(tmp, vertices, triangles, landmarks)
            # the files are in glTF's frame: Anny's Z up is glTF's Y up
            raw = json.loads((pathlib.Path(tmp) / "appearance.json").read_text())
            np.testing.assert_allclose(raw["landmarks"]["nose_tip"], [0.1, 1.6, 0.2])
            handoff = load_handoff(tmp)
        self.assertEqual(len(handoff.parts), 1)
        part = handoff.parts[0]
        self.assertTrue(part.is_skin)
        used = np.unique(triangles)
        np.testing.assert_allclose(
            part.vertices[np.unique(part.triangles)], vertices[used], atol=1e-6
        )
        np.testing.assert_allclose(handoff.landmarks["nose_tip"], landmarks["nose_tip"])

    def test_missing_replica(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError):
            load_handoff(tmp)


class TestSyntheticFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        target = synthetic_target(0)
        cls.target = target
        write_handoff(
            cls.folder / "handoff",
            target["vertices"],
            target["triangles"],
            {k: v for k, v in target["landmarks"].items()},
        )
        handoff = load_handoff(cls.folder / "handoff")
        cls.reference = Reference.from_handoff(handoff)
        cls.result = fit_reference(cls.reference, detail=1, name="synthetic")
        cls.card_path = cls.result.write(cls.folder / "card.json")
        _report(
            f"synthetic fit ({len(target['vertices'])} target vertices, "
            f"{sum(cls.result.seconds.values()):.0f} s: "
            + ", ".join(f"{k} {v:.0f} s" for k, v in cls.result.seconds.items())
            + ")",
            cls.result.errors,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_target_has_another_topology(self):
        counts = {level("anny", s).vertex_count for s in (0, 1, 2)}
        self.assertNotIn(len(self.target["vertices"]), counts)
        self.assertFalse(self.reference.auto)

    def test_level1_error_is_under_half_a_millimetre(self):
        errors = self.result.errors
        self.assertLess(errors[1]["forward"]["mean"], LEVEL1_BOUND_MM)
        self.assertLess(errors[1]["reverse"]["mean"], LEVEL1_BOUND_MM)
        self.assertGreater(errors[1]["reverse_covered"], 0.99)
        # each stage improves on the one before
        for coarse, fine in (("parametric", 0), (0, 1)):
            for side in ("forward", "reverse"):
                self.assertLess(
                    errors[fine][side]["mean"], errors[coarse][side]["mean"]
                )

    def test_card_reproduces_the_fit(self):
        card = Character.from_dict(json.loads(self.card_path.read_text()))
        self.assertEqual(card.detail_levels(), [0, 1])
        model = card.build_model(dtype=torch.float64)
        placement = card.detail["fit"]["placement"]
        R = np.array(placement["rotation"])
        t = np.array(placement["translation"])
        excluded = excluded_vertices(model).astype(np.float64)
        for subdiv in (0, 1):
            with self.subTest(subdiv=subdiv):
                offsets = character_offsets(card, subdiv, self.folder)
                detailed = DetailModel(model, subdiv, offsets)
                with torch.no_grad():
                    out = detailed(
                        pose_parameters=card.pose_parameters(model),
                        **card.model_kwargs(),
                    )
                vertices = out["vertices"][0].numpy() @ R.T + t
                mask = np.asarray(detailed.topology.apply(excluded)) < 0.5
                error = surface_error(
                    vertices, detailed.triangles, self.reference, mask
                )
                self.assertAlmostEqual(
                    error["forward"]["mean"],
                    self.result.errors[subdiv]["forward"]["mean"],
                    delta=0.01,
                )

    def test_fitted_card_exports(self):
        from opensculptboy.export.vrm import export_vrm

        card = Character.from_dict(json.loads(self.card_path.read_text()))
        summary = export_glb(
            self.folder / "fitted.glb", card, detail_level=1, detail_dir=self.folder
        )
        self.assertEqual(summary["detail"]["subdiv"], 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vrm = export_vrm(
                self.folder / "fitted.vrm",
                card,
                author="tests",
                detail_level=1,
                detail_dir=self.folder,
                budget="strict",
                refine_weights=False,
            )
        self.assertLessEqual(vrm["triangles"], BUDGETS["vrm1"]["triangles"])


class TestAutoLandmarks(unittest.TestCase):
    def test_fit_without_landmarks(self):
        target = synthetic_target(0)
        reference = Reference.from_mesh(target["vertices"], target["triangles"])
        self.assertTrue(reference.auto)
        result = fit_reference(
            reference, detail=0, options=FitOptions(rounds=8, steps=25)
        )
        _report("fit without landmarks", result.errors)
        self.assertLess(result.errors[0]["forward"]["mean"], 1.0)
        self.assertLess(result.errors[0]["reverse"]["mean"], 1.0)


def _has_embree() -> bool:
    try:
        import trimesh.ray.ray_pyembree  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_has_embree(), "the texture bake needs embreex (the viewer extra)")
class TestTextureBake(unittest.TestCase):
    def test_bake_onto_the_same_surface_copies_the_texture(self):
        import anny

        from opensculptboy.export.body import build_body, triangulated_faces
        from opensculptboy.fit.handoff import HandoffPart
        from opensculptboy.texture.bake import bake

        model = anny.Anny().to(torch.float64)
        with torch.no_grad():
            rest = model()["rest_vertices"][0].numpy()
        body = build_body(model, rest)
        size = 256
        yy, xx = np.mgrid[0:size, 0:size]
        texture = np.stack(
            [xx * 255 // size, yy * 255 // size, ((xx // 16 + yy // 16) % 2) * 255], -1
        ).astype(np.uint8)
        part = HandoffPart(
            "skin", "skin", rest[body.source], body.triangles, body.uvs, texture
        )
        reference = Reference(
            part.vertices,
            part.triangles,
            {},
            True,
            [part],
            np.array([0, len(part.vertices)]),
        )
        triangles, uv_triangles = triangulated_faces(model)
        baked = bake(
            rest,
            triangles,
            uv_triangles,
            model.texture_coordinates.double().numpy(),
            reference,
            size=size,
        )
        self.assertGreater(baked.covered_share, 0.99)
        covered = baked.coverage
        difference = np.abs(baked.base_color.astype(int) - texture.astype(int))[covered]
        self.assertLessEqual(np.percentile(difference, 99), 2)
        normal = baked.normal[covered].astype(float) / 127.5 - 1.0
        # the same surface gives flat normals; the target's vertices split at the UV seams,
        # whose unwelded normals tilt a few texels along the seams
        self.assertGreater(np.median(normal[:, 2]), 0.999)
        self.assertGreater(np.percentile(normal[:, 2], 1), 0.95)
        self.assertEqual(set(baked.png_files()), {"base_color", "normal"})


if __name__ == "__main__":
    unittest.main()

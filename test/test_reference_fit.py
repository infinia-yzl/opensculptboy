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
from opensculptboy.fit.handoff import (
    _find_entry,
    _part_entries,
    load_handoff,
    part_role,
    write_handoff,
)
from opensculptboy.fit.reference import (
    LANDMARKS,
    Reference,
    canonical_landmark,
    canonical_landmarks,
    eye_spheres,
)
from opensculptboy.fit.register import (
    FitOptions,
    ear_weights,
    excluded_vertices,
    fit_model,
    fit_reference,
    follow_skin,
    lip_margin,
    project_onto,
    relax_along,
    shared_target_controls,
)
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
            ("ear_tip_l", "ear_top_l"),
            ("RightEarLobe", "ear_lobe_r"),
            ("postaurale.L", "ear_back_l"),
            ("ear_l_rig", None),
            ("tail", None),
        ):
            with self.subTest(name=name):
                self.assertEqual(canonical_landmark(name), expected)

    def test_surface_variants(self):
        # a face landmark takes the point snapped to the skin, a joint centre the other
        points = {
            "mouth_corner_l": [1.0, 0.0, 0.0],
            "mouth_corner_l_skin": [2.0, 0.0, 0.0],
            "elbow_l": [3.0, 0.0, 0.0],
            "elbow_l_skin": [4.0, 0.0, 0.0],
            "nose_tip_joint": [5.0, 0.0, 0.0],
        }
        for order in (list(points), list(reversed(points))):
            with self.subTest(order=order[0]):
                found = canonical_landmarks({k: points[k] for k in order})
                self.assertEqual(sorted(found), ["elbow_l", "mouth_corner_l"])
                self.assertEqual(found["mouth_corner_l"][0], 2.0)
                self.assertEqual(found["elbow_l"][0], 3.0)

    def test_rig_dependent_joints_take_no_weight(self):
        # the hips and the neck base sit 20 to 30 mm apart between rigs
        for name in ("hip_l", "hip_r", "neck_base"):
            self.assertEqual(LANDMARKS[name][2], 0.0)
        for name in ("ear_top_l", "ear_lobe_r", "ear_back_l", "ear_lateral_r"):
            self.assertEqual(LANDMARKS[name][0], "craniofacial")

    def test_part_roles(self):
        self.assertEqual(part_role("body"), "skin")
        self.assertEqual(part_role("Hair_Long"), "hair")
        self.assertEqual(part_role("eye_L"), "eyes")
        self.assertEqual(part_role("eyelid"), "skin")
        self.assertEqual(part_role("eyebrows_mesh"), "brows")
        self.assertEqual(part_role("anything", {"role": "hair"}), "hair")
        self.assertEqual(part_role("hair_cap", {"role": "head"}), "skin")
        # the material of a primitive overrides the role of its part
        self.assertEqual(part_role("head_0", {"role": "head"}, "teeth_mat"), "mouth")
        self.assertEqual(part_role("head_0", {"role": "head"}, "l_eye_mat"), "eyes")
        self.assertEqual(part_role("body_0", {"role": "body"}, "nail_mat"), "nails")
        self.assertEqual(part_role("body_0", {"role": "body"}, "body_mat"), "skin")
        self.assertEqual(
            part_role("head_0", {"role": "head"}, "x", {"role": "eyes"}), "eyes"
        )

    def test_contract_parts(self):
        # the contract keys parts by kind, and the tools name nodes <node_name>_group<N>_...
        entries = _part_entries(
            {"hair": {"node_name": "hair_a"}, "head": {"node_name": "head_b"}}
        )
        self.assertEqual(
            part_role(
                "hair_a_group0_band_mat",
                _find_entry(entries, "hair_a_group0_band_mat"),
                "band_mat",
            ),
            "hair",
        )
        self.assertEqual(
            part_role("head_b_group0", _find_entry(entries, "head_b_group0")),
            "skin",
        )
        self.assertIsNone(_find_entry(entries, "torso_0"))


def _sphere(centre, radius, cap=1.0, count=400, seed=0):
    """Points of a sphere, or of its cap toward -Y when ``cap`` < 1 (a share of the polar
    angle)."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(count, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d[np.arccos(-d[:, 1]) <= cap * np.pi]
    return np.asarray(centre) + radius * d


class TestEyes(unittest.TestCase):
    def test_eye_spheres(self):
        left = _sphere([0.03, -0.05, 1.5], 0.012, cap=0.4)
        cornea = _sphere([0.03, -0.055, 1.5], 0.008, cap=0.2, seed=1)
        right = _sphere([-0.03, -0.05, 1.5], 0.012, cap=0.4, seed=2)
        card = np.random.default_rng(3).uniform(-0.01, 0.01, (50, 3)) * [1, 0, 1]
        spheres = eye_spheres([left, cornea, right, card + [0, -0.1, 1.5]])
        self.assertEqual(len(spheres), 2)
        for (centre, radius), x in zip(
            sorted(spheres, key=lambda e: -e[0][0]), (1, -1)
        ):
            np.testing.assert_allclose(centre, [0.03 * x, -0.05, 1.5], atol=5e-4)
            self.assertAlmostEqual(radius, 0.012, delta=3e-4)

    def test_eyeball_follows_reference_eye(self):
        # a closed component (an eyeball) moves and scales onto the reference's eye sphere
        import trimesh

        ball = trimesh.creation.icosphere(2, 0.014)
        skin = trimesh.creation.box((0.2, 0.01, 0.2))
        vertices = np.concatenate([ball.vertices, skin.vertices + [0, 0.05, 0]])
        triangles = np.concatenate([ball.faces, skin.faces + len(ball.vertices)])
        excluded = np.arange(len(vertices)) < len(ball.vertices)
        eye = (np.array([0.004, -0.002, 0.001]), 0.012)
        out = follow_skin(vertices, vertices, excluded, triangles, eyes=[eye])
        radii = np.linalg.norm(out[excluded] - eye[0], axis=1)
        np.testing.assert_allclose(radii, eye[1], atol=1e-4)
        np.testing.assert_allclose(out[~excluded], vertices[~excluded])


class TestFitRegions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = fit_model()

    def test_shared_targets(self):
        # a face shape and a local change on the same MakeHuman target: the face shape alone
        # is fitted, and the local changes of the body stay free
        local, face = shared_target_controls(self.model)
        labels = list(self.model.local_change_labels)
        self.assertTrue(local[labels.index("chin-prominent-incr")])
        self.assertTrue(local[labels.index("l-ear-shape-triangle")])
        self.assertFalse(local[labels.index("l-upperarm-fat-incr")])
        faces = list(self.model.face_shape_labels)
        self.assertTrue(face[faces.index("chin-prominent")])
        self.assertFalse(face[faces.index("detail-1")])
        with self.assertRaisesRegex(ValueError, "shared_targets"):
            fit_reference(
                Reference.from_mesh(np.eye(3), np.array([[0, 1, 2]])),
                detail=0,
                options=FitOptions(shared_targets="neither", rounds=1, steps=1),
                model=self.model,
            )

    def test_closed_parts_are_excluded(self):
        # the eyeballs and the tongue take no data
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        faces = self.model.faces.numpy()
        n = len(self.model.template_vertices)
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]]])
        graph = coo_matrix((np.ones(len(edges)), edges.T), shape=(n, n))
        _, labels = connected_components(graph, directed=False)
        body = np.argmax(np.bincount(labels))
        excluded = excluded_vertices(self.model)
        self.assertTrue(excluded[labels != body].all())
        self.assertGreater((labels != body).sum(), 144)  # more than the eyeballs
        margin = lip_margin(self.model, excluded)
        self.assertTrue(margin.any())
        self.assertFalse((margin & excluded).any())

    def test_ear_weights(self):
        w = ear_weights(self.model)
        rest = self.model.template_vertices.numpy()
        ears = rest[w > 0.99]
        self.assertGreater(len(ears), 100)
        # both sides, far from the midline, at the height of the head
        self.assertGreater(np.abs(ears[:, 0]).min(), 0.05)
        top = rest[:, 2].max()
        self.assertGreater(ears[:, 2].min(), top - 0.3)
        self.assertEqual(w[rest[:, 2] < top - 0.5].max(), 0.0)

    def test_relax_and_project(self):
        # a stretched patch of a plane spreads out evenly and stays on the plane
        import trimesh

        plane = trimesh.creation.box((0.2, 0.2, 0.01))
        reference = Reference(plane.vertices, plane.faces)
        grid = trimesh.creation.box((0.1, 0.1, 0.01)).subdivide().subdivide()
        V = np.asarray(grid.vertices, dtype=np.float64)
        top = V[:, 2] > 0
        V[top, :2] = np.sign(V[top, :2]) * np.abs(V[top, :2]) ** 2 * 10  # bunch them up
        relaxed = relax_along(
            V, grid.faces, reference, top.astype(float), iterations=30
        )
        np.testing.assert_allclose(relaxed[top, 2], 0.005, atol=1e-9)
        edges = grid.edges_unique
        on_top = top[edges].all(1)

        def spread(X):
            length = np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1)[on_top]
            return length.std() / length.mean()

        self.assertLess(spread(relaxed), 0.6 * spread(V))
        lifted = V + [0, 0, 0.001]
        projected = project_onto(lifted, grid.faces, reference, top, 0.002)
        np.testing.assert_allclose(projected[top, 2], 0.005, atol=1e-9)
        np.testing.assert_allclose(projected[~top], lifted[~top])


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

    def test_vertex_colours_stay_out_of_the_skin_colour(self):
        # the DD2DA tools keep the game's masks in vertex colours, and a hair part's COLOR_0
        # holds its albedo: the base colour of a part is its texture alone
        import trimesh
        from PIL import Image

        texture = np.zeros((8, 8, 3), np.uint8)
        texture[..., 1] = 200
        box = trimesh.creation.box((0.1, 0.1, 0.1))
        uv = np.random.default_rng(0).uniform(0.1, 0.9, (len(box.vertices), 2))
        box.visual = trimesh.visual.TextureVisuals(
            uv=uv, image=Image.fromarray(texture)
        )
        box.visual.vertex_attributes["color"] = np.tile(
            [255, 0, 0, 255], (len(box.vertices), 1)
        ).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            scene = trimesh.Scene()
            scene.add_geometry(box, node_name="body", geom_name="body")
            scene.export(str(pathlib.Path(tmp) / "replica.glb"))
            raw = (pathlib.Path(tmp) / "replica.glb").read_bytes()
            self.assertIn(b"COLOR_0", raw)
            handoff = load_handoff(tmp)
        (part,) = handoff.parts
        np.testing.assert_array_equal(part.base_color[..., :3], texture)

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

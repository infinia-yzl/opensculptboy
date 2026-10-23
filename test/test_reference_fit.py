# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the reference fit (``opensculptboy.fit``) on a synthetic target of another topology
(``test.fit_fixtures.synthetic_target``: random anny parameters, a small pose, smooth random
offsets on levels 0 and 1, remeshed by sliding and vertex clustering), read through an
appearance handoff folder (``opensculptboy.fit.handoff``).

``TestSyntheticFit`` fits once (about 70 s) and checks the mean surface error of
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
from opensculptboy.detail import DetailModel, character_offsets, detail_topology, level
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
    EarSplit,
    FitOptions,
    Target,
    _reverse_pairs,
    ear_weights,
    excluded_vertices,
    fair_region,
    fair_skin,
    fit_levels,
    fit_model,
    fit_parameters,
    fit_reference,
    fold_defects,
    follow_skin,
    lip_margin,
    mouth_lining,
    pinna,
    project_onto,
    relabel_ears,
    relax_along,
    repair_folds,
    shared_target_controls,
    smooth_mouth,
)
from opensculptboy.fit.residual import surface_error
from test.fit_fixtures import (
    faceted_sphere,
    pointed_ear_target,
    stitched_ear_target,
    synthetic_target,
)

# the bound of the mean surface error of level 1, both ways (mm)
LEVEL1_BOUND_MM = 0.5
# the fine ripple (metres, metrics.ripple) of the synthetic fit's level-1 skin of the head:
# 124 micrometres projected onto the target's facets alone, 65 with the skin faired
LEVEL1_RIPPLE_BOUND = 90e-6


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
        # the contract keys parts by kind, and a source may split a node into <node_name>_...
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


def _hairpin(n_u: int = 41, n_v: int = 11, radius: float = 0.0015):
    """A strip folded back over itself, in metres: a flat "head" at z = 0 for x in [0, 0.01],
    a half turn of ``radius`` and a "flap" back over the head at z = 2 ``radius``, with a gap
    between them; and the mask of the flap (from the top of the turn on)."""
    turn = np.pi * radius
    u = np.linspace(0.0, 0.02 + turn, n_u)
    v = np.linspace(0.0, 0.01, n_v)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    angle = np.clip((uu - 0.01) / radius, 0.0, np.pi)
    x = np.where(uu <= 0.01, uu, 0.01 + radius * np.sin(angle))
    x = np.where(uu >= 0.01 + turn, 0.01 - (uu - 0.01 - turn), x)
    z = np.where(uu <= 0.01, 0.0, radius * (1.0 - np.cos(angle)))
    V = np.stack([x, vv, z], -1).reshape(-1, 3)
    k = np.arange(n_u * n_v).reshape(n_u, n_v)
    a, b, c, d = k[:-1, :-1], k[1:, :-1], k[1:, 1:], k[:-1, 1:]
    T = np.concatenate([np.stack([a, b, c], -1), np.stack([a, c, d], -1)]).reshape(
        -1, 3
    )
    flap = (uu >= 0.01 + 0.5 * turn).ravel()
    return V, T, flap


class TestMouthLining(unittest.TestCase):
    def test_lining_is_the_mouth_and_its_margin(self):
        model = fit_model()
        cavity = mouth_lining(model, 0)
        margin = mouth_lining(model, 1)
        self.assertGreater(cavity.sum(), 100)
        self.assertTrue((margin >= cavity).all())
        self.assertGreater(margin.sum(), cavity.sum())
        # the margin is lip_margin's, the eye sockets stay out
        np.testing.assert_array_equal(
            margin & ~cavity, lip_margin(model, np.zeros_like(cavity), 1) & ~cavity
        )

    def test_smooth_mouth_is_harmonic(self):
        # a grid whose middle is the lining: a linear field on its border extends linearly,
        # and noise inside the lining is gone
        n = 12
        x, y = np.meshgrid(
            np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64)
        )
        points = np.stack([x.ravel(), y.ravel(), np.zeros(n * n)], 1)
        quads = [
            (r * n + c, r * n + c + 1, (r + 1) * n + c + 1, (r + 1) * n + c)
            for r in range(n - 1)
            for c in range(n - 1)
        ]
        triangles = np.array(
            [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]
        )
        lining = ((x > 2) & (x < n - 3) & (y > 2) & (y < n - 3)).ravel()
        field = np.stack(
            [0.001 * points[:, 0], -0.002 * points[:, 1], 0 * points[:, 0]], 1
        )
        noisy = field.copy()
        noisy[lining] += np.random.default_rng(0).normal(0, 0.01, (lining.sum(), 3))
        smooth = smooth_mouth(noisy, triangles, lining)
        np.testing.assert_allclose(smooth, field, atol=1e-12)
        # a lining component that touches nothing keeps its offsets
        alone = np.zeros(n * n, bool)
        alone[:] = True
        np.testing.assert_array_equal(smooth_mouth(noisy, triangles, alone), noisy)


class TestEarParts(unittest.TestCase):
    """The pieces of the ear fit (EarSplit, pinna, fair_region, repair_folds) on small shapes."""

    @classmethod
    def setUpClass(cls):
        cls.model = fit_model()

    def test_pinna(self):
        # the outer ear proper: the vertices that the ear targets move in full, without the
        # strip of the head's skin around the root, with the concha
        rest = self.model.template_vertices.double().numpy()
        core = ear_weights(self.model) > 0.99
        outer = pinna(self.model, rest)
        self.assertFalse((outer & ~core).any())
        for side in (1.0, -1.0):
            on = np.sign(rest[:, 0]) == side
            self.assertGreater((outer & on).sum(), 0.8 * (core & on).sum())
            self.assertLess((outer & on).sum(), (core & on).sum())
        # mirror images: as many on each side
        self.assertEqual(
            (outer & (rest[:, 0] > 0)).sum(), (outer & (rest[:, 0] < 0)).sum()
        )

    def test_split_matches_ears_to_ears(self):
        V, T, flap = _hairpin()
        reference = Reference(V, T, regions={"ear": flap})
        split = EarSplit(reference)
        # the attachment defaults to the border of the mask, in the turn
        self.assertTrue(split.attachment.any())
        self.assertTrue((V[split.attachment, 2] > 0).all())
        # a point in the gap, nearer the flap: as head it lands on the head, as ear on the flap
        point = np.array([[0.005, 0.005, 0.002]])
        for is_ear, height in ((False, 0.0), (True, 0.003)):
            target = Target(reference, split=split, anny_ear=np.array([is_ear]))
            C, _, _ = target.closest(point)
            self.assertAlmostEqual(C[0, 2], height, delta=1e-9)
        unsplit = Target(reference).closest(point)[0]
        self.assertAlmostEqual(unsplit[0, 2], 0.003, delta=1e-9)
        # the attachment line: the closest point of the turn's border
        a, d = split.attach(np.array([[0.0115, 0.005, 0.0015]]))
        self.assertLess(d[0], 0.0005)
        # reverse pairs: no vertex of the flap pairs with a head vertex of the fit
        fit = np.array([[0.005, 0.005, 0.0012], [0.005, 0.005, 0.0025]])
        normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        target = Target(reference, split=split, anny_ear=np.array([False, True]))
        hit, mean = _reverse_pairs(fit, normals, np.arange(2), target, 0.02, -1.0)
        self.assertTrue(hit.all())
        self.assertLess(mean[0, 2], 0.001)  # the head's, the start of the turn with it
        self.assertGreater(mean[1, 2], 0.0025)  # the flap's

    def test_stitched_ear_keeps_its_cut(self):
        # a game writes its ear as a piece of its own, stitched to the head by coincident
        # vertices: the cut, and the folds across it, are those of the welded surface
        V, T, flap = _hairpin()
        joined = EarSplit(Reference(V, T, regions={"ear": flap}))
        on_flap = flap[T].all(1)
        seam = np.unique(T[on_flap])
        seam = seam[np.isin(seam, T[~on_flap])]
        copy = np.arange(len(V), len(V) + len(seam))
        Vs = np.concatenate([V, V[seam]])
        Ts = T.copy()
        remap = np.arange(len(V))
        remap[seam] = copy
        Ts[on_flap] = remap[T[on_flap]]
        flap_s = np.concatenate([flap, np.ones(len(seam), bool)])
        flap_s[seam] = False
        stitched = Reference(Vs, Ts, regions={"ear": flap_s})
        split = EarSplit(stitched)
        for a, b in ((joined.ear, split.ear), (joined.head, split.head)):
            self.assertGreater(b.cut.sum(), 0)
            np.testing.assert_allclose(
                np.sort(a.vertices[a.cut], 0), np.sort(b.vertices[b.cut], 0)
            )
        np.testing.assert_allclose(stitched.folds[: len(V)], Reference(V, T).folds)

    def test_relabel_ears_follows_the_attachment(self):
        # a flat sheet whose ear part (x > 0.01) holds more than the fitted mesh's ear mask
        # (x > 0.014): the mask moves out to the line, within its band
        n = 41
        u = np.linspace(0.0, 0.02, n)
        xx, yy = np.meshgrid(u, u, indexing="ij")
        V = np.stack([xx, yy, np.zeros_like(xx)], -1).reshape(-1, 3)
        k = np.arange(n * n).reshape(n, n)
        a, b, c, d = k[:-1, :-1], k[1:, :-1], k[1:, 1:], k[:-1, 1:]
        T = np.concatenate([np.stack([a, b, c], -1), np.stack([a, c, d], -1)])
        T = T.reshape(-1, 3)
        split = EarSplit(Reference(V, T, regions={"ear": V[:, 0] > 0.01 - 1e-9}))
        mask = V[:, 0] > 0.014 - 1e-9
        moved = relabel_ears(V, T, mask, split, rings=10)
        inside = V[:, 0] > 0.0102
        outside = V[:, 0] < 0.0098
        self.assertTrue(moved[inside].all())
        self.assertFalse(moved[outside].any())
        # outside its band the mask stays as it was
        kept = relabel_ears(V, T, mask, split, 0)
        self.assertFalse(kept[V[:, 0] < 0.012].any())

    def test_fair_and_repair_keep_a_crease(self):
        # a tent (a crease of 90 degrees) is the reference's own fold: no defect; a crumple of
        # the fit on it is one, and the repair takes it out and lays the sheet on the reference
        n = 21
        u = np.linspace(-0.02, 0.02, n)
        xx, yy = np.meshgrid(u, u, indexing="ij")
        V = np.stack([xx, yy, 0.5 * (0.02 - np.abs(xx))], -1).reshape(-1, 3)
        k = np.arange(n * n).reshape(n, n)
        a, b, c, d = k[:-1, :-1], k[1:, :-1], k[1:, 1:], k[:-1, 1:]
        T = np.concatenate([np.stack([a, b, c], -1), np.stack([a, c, d], -1)])
        T = T.reshape(-1, 3)
        reference = Reference(V, T)
        inner = (np.abs(V[:, :2]) < 0.015).all(1)
        self.assertFalse(fold_defects(V, T, reference, inner).any())
        crumpled = V.copy()
        pit = np.nonzero((np.abs(V[:, 0] - 0.01) < 1e-9) & (np.abs(V[:, 1]) < 0.003))[0]
        crumpled[pit, 2] -= 0.004  # pits that fold the sheet
        found = fold_defects(crumpled, T, reference, inner)
        self.assertTrue(found[pit].any())
        repaired, first = repair_folds(crumpled, T, reference, inner)
        self.assertGreater(first, 0)
        self.assertFalse(fold_defects(repaired, T, reference, inner).any())
        _, _, dist, _, _ = reference.closest(repaired[inner])
        self.assertLess(dist.max(), 2e-4)
        faired = fair_region(crumpled, T, reference, inner)
        _, _, dist, _, _ = reference.closest(faired[inner])
        self.assertLess(dist.max(), 5e-4)
        # the crease stays where it is
        ridge = inner & (np.abs(V[:, 0]) < 1e-9)
        np.testing.assert_allclose(repaired[ridge, 2], V[ridge, 2], atol=2e-4)


class TestSkinFairing(unittest.TestCase):
    def test_fairing_drops_the_facets(self):
        # a fine sphere projected onto a coarse one's flat triangles copies their facets;
        # faired on them, it keeps to the reference and lies close to the true sphere
        from opensculptboy.fit.metrics import ripple

        RV, RT, X, T = faceted_sphere()
        reference = Reference(RV, RT)
        Y = fair_skin(X, T, reference, np.ones(len(X), bool), 160.0, 2, 0.01)
        radius = 0.05

        def spread(points):
            return float((np.linalg.norm(points, axis=1) - radius).std())

        # measured: ripple 34 -> 11 micrometres, spread of the radii 49 -> 12
        self.assertLess(ripple(Y, T), 0.4 * ripple(X, T))
        self.assertLess(spread(Y), 0.4 * spread(X))
        self.assertLess(reference.closest(Y)[2].mean(), 1e-4)

    def test_region_leaves_ears_and_held_out(self):
        from opensculptboy.fit.register import skin_fair_region

        class Flat:
            # a strip of 6 vertices: 0-1-2-3-4-5 joined by triangles
            triangles = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]])

            @staticmethod
            def apply(values):
                return values

        skin0 = np.ones(6)
        data = np.ones(6)
        ears = np.zeros(6, bool)
        ears[0] = True
        held = np.zeros(6, bool)
        held[5] = True
        region = skin_fair_region(Flat, skin0, data, ears, 1, held)
        # vertex 0 is ear, 1 and 2 lie one face from it, 5 is held
        np.testing.assert_array_equal(region, [False, False, False, True, True, False])

    def test_options_per_level(self):
        # the mouth's repair and seal reach further on level 2 than on levels 0 and 1
        from opensculptboy.fit.register import FitOptions, _as_levels, _per_level

        options = FitOptions()
        reach = [
            _per_level(_as_levels(options.mouth_repair_reach), k) for k in range(3)
        ]
        self.assertEqual(reach, [5, 5, 8])
        seal = [_per_level(_as_levels(options.lip_seal_reach), k) for k in range(3)]
        self.assertEqual(seal, [0.0015, 0.0015, 0.003])
        blend = [
            _per_level(_as_levels(options.skin_fair_lid_blend), k) for k in range(3)
        ]
        self.assertEqual(blend, [0.0, 0.0, 50.0])
        # one value serves every level, and so does None
        self.assertEqual(_per_level(_as_levels(0.002), 2), 0.002)
        self.assertIsNone(_per_level(_as_levels(None), 1))

    def test_lid_blend_falls_off(self):
        from opensculptboy.fit.register import lid_blend

        # a strip of 8 vertices, each pair of neighbours joined: 0-1-...-7
        triangles = np.array([[k, k + 1, k + 2] for k in range(6)])
        held = np.zeros(8, bool)
        held[0] = True
        band, anchor = lid_blend(triangles, held, 3, 50.0)
        # vertices 1 and 2 share a face with 0; the next rings follow
        self.assertTrue(band[1] and band[2] and not band[0])
        self.assertEqual(anchor[0], 0.0)
        self.assertEqual(anchor[1], 50.0)
        # the weights fall with each ring and end at 0 past the band
        rings = sorted({float(a) for a in anchor[band]}, reverse=True)
        self.assertEqual(rings[0], 50.0)
        self.assertTrue(all(a > b for a, b in zip(rings, rings[1:])))
        self.assertTrue((anchor[~band] == 0).all())

    def test_band_joins_held_skin_without_a_step(self):
        # a cap of a faceted sphere held as projected (the seated lids), the rest faired:
        # held fast, a band around the cap keeps the facets and ends in a step; faired with a
        # falling hold, the band joins the cap and the faired skin smoothly
        from opensculptboy.fit.metrics import ripple
        from opensculptboy.fit.register import _adjacency, lid_blend

        RV, RT, X, T = faceted_sphere()
        reference = Reference(RV, RT)
        cap = X[:, 2] > 0.035
        adjacency = _adjacency(len(X), T)
        ring = cap.copy()
        for _ in range(3):
            ring |= adjacency @ ring.astype(np.float64) > 0
        fast = fair_skin(X, T, reference, ~ring, 160.0, 2, 0.01)
        band, anchor = lid_blend(T, cap, 3, 50.0)
        blended = fair_skin(X, T, reference, ~cap, 160.0, 2, 0.01, anchor=anchor)
        # measured: ripple over the band 32 -> 19 micrometres; the cap stays where it was
        self.assertLess(ripple(blended, T, band), 0.7 * ripple(fast, T, band))
        np.testing.assert_allclose(blended[cap], X[cap])
        self.assertLess(reference.closest(blended[band])[2].mean(), 1e-4)

    def test_corners_faired_around_held_rows(self):
        # the mouth's corners: a crumpled disc around each of two points of a faceted sphere,
        # with a row of held vertices through it (the sealed lips) and a part left out (the
        # inside of the mouth); the disc lies smooth on the reference again, the held row and
        # the left-out part stay near where they were, and the skin outside the discs stays
        from opensculptboy.fit.metrics import ripple
        from opensculptboy.fit.register import _fair_corners

        RV, RT, X, T = faceted_sphere()
        reference = Reference(RV, RT)
        front = X[np.argmax(X[:, 2])]
        corners = np.stack([front + [0.006, 0.0, 0.0], front - [0.006, 0.0, 0.0]])
        near = np.min(np.linalg.norm(X[:, None] - corners[None], axis=2), axis=1)
        disc = near < 0.008
        rng = np.random.default_rng(7)
        crumpled = X.copy()
        crumpled[disc] += rng.normal(scale=2e-4, size=(int(disc.sum()), 3))
        held = np.nonzero(disc & (np.abs(X[:, 1] - front[1]) < 5e-4))[0]
        cavity = disc & (X[:, 1] < front[1] - 0.002)
        offsets, count = _fair_corners(
            crumpled - X,
            X,
            T,
            reference,
            corners,
            cavity,
            (held,),
            lambda p, r: p,
            lambda p, r: p,
            160.0,
            0.008,
            1000.0,
            2,
            0.01,
        )
        Y = X + offsets
        near = np.min(np.linalg.norm(crumpled[:, None] - corners[None], axis=2), axis=1)
        faired = (near < 0.008) & ~cavity
        self.assertEqual(count, int(faired.sum()))
        # measured: the ripple falls to about a third, the held row moves 15 micrometres; the faired disc
        # stays within the noise of the reference (off it by 117 micrometres on average, the faceted sphere's own
        # skin lying under the smooth surface the fairing gives)
        self.assertLess(ripple(Y, T, faired), 0.4 * ripple(crumpled, T, faired))
        self.assertLess(reference.closest(Y[faired])[2].mean(), 2e-4)
        np.testing.assert_array_equal(Y[~faired], crumpled[~faired])
        np.testing.assert_array_equal(Y[cavity], crumpled[cavity])
        moved = np.linalg.norm(Y[held] - crumpled[held], axis=1)
        self.assertLess(moved.mean(), 5e-5)


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
        # a source may keep masks in its vertex colours, and a hair part's COLOR_0
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

    def test_ear_mask_and_attachment(self):
        rng = np.random.default_rng(4)
        vertices = rng.normal(size=(30, 3))
        triangles = np.array([[0, 1, 2], [2, 3, 4], [5, 6, 7]])
        ear = np.zeros(30, bool)
        ear[[1, 2, 3]] = True
        attachment = np.zeros(30, bool)
        attachment[2] = True
        with tempfile.TemporaryDirectory() as tmp:
            write_handoff(
                tmp,
                vertices,
                triangles,
                {"nose_tip": [0.0, -0.1, 1.6]},
                masks={"ear": {"vertices": ear, "attachment": attachment}},
            )
            reference = Reference.from_handoff(load_handoff(tmp))
        np.testing.assert_array_equal(reference.regions["ear"], ear[: len(ear)])
        np.testing.assert_array_equal(reference.regions["ear_attachment"], attachment)

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
        cls.model = fit_model()
        cls.result = fit_reference(
            cls.reference, detail=1, name="synthetic", model=cls.model
        )
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

    def test_honest_measures(self):
        # unmasked, signed, before the final projection: the projection cannot hide errors
        measures = self.result.measures
        self.assertEqual(set(measures), {"parametric", 0, 1})
        before = measures[1]["before_projection"]["all"]
        final = measures[1]["final"]["all"]
        print(
            f"\nhonest level 1: before projection forward {before['forward_mm']['mean']:.3f}"
            f" reverse {before['reverse_mm']['mean']:.3f}; final forward "
            f"{final['forward_mm']['mean']:.3f} reverse {final['reverse_mm']['mean']:.3f} mm",
            file=sys.stderr,
        )
        for entry in (before, final):
            self.assertLess(entry["forward_mm"]["mean"], LEVEL1_BOUND_MM)
            self.assertLess(entry["reverse_mm"]["mean"], LEVEL1_BOUND_MM)
        self.assertLess(
            measures[1]["final"]["all"]["reverse_mm"]["mean"],
            measures["parametric"]["all"]["reverse_mm"]["mean"],
        )

    def test_default_fit_without_ear_mask(self):
        # the ear options act through masks.ear alone: without it, the refinement with them
        # is the refinement without them, bit for bit; with it, the ears fit apart
        self.assertNotIn("ear", self.reference.regions)
        card, R, t = (
            self.result.character,
            self.result.rotation,
            self.result.translation,
        )
        kwargs = dict(detail=0, model=self.model, options=FitOptions(measure=False))
        split = fit_levels(self.reference, card, R, t, **kwargs)
        kwargs["options"] = FitOptions(measure=False, ear_split=False)
        plain = fit_levels(self.reference, card, R, t, **kwargs)
        np.testing.assert_array_equal(split.offsets[0], plain.offsets[0])
        np.testing.assert_array_equal(split.offsets[0], self.result.offsets[0])

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

    def test_skin_keeps_no_facets(self):
        # the target's triangles are 5 mm cells, larger than level 1's faces on the face;
        # projected onto them alone, the skin of the head copies them as steps
        # (FitOptions.skin_fairness, register.fair_skin)
        from opensculptboy.fit.metrics import (
            REGIONS,
            anny_regions,
            level_regions,
            ripple,
        )

        vertices, triangles = self.result.meshes[1]
        topology = detail_topology(self.model, 1)
        regions = level_regions(topology, anny_regions(self.model))
        head = np.isin(regions, [REGIONS.index(k) for k in ("face", "jaw", "neck")])
        value = ripple(vertices, triangles, head)
        print(f"\nlevel-1 skin ripple {1e6 * value:.1f} micrometres", file=sys.stderr)
        self.assertLess(value, LEVEL1_RIPPLE_BOUND)

    def test_mouth_opens_without_tears(self):
        # the offsets of the inside of the mouth and the lips' inner margin are smooth
        # (smooth_mouth), so the mouth's own shapes open it without spikes, and the closed
        # lips keep no crumple the card without offsets lacks
        from opensculptboy.fit.metrics import mouth_tears

        card = Character.from_dict(json.loads(self.card_path.read_text()))
        model = card.build_model(dtype=torch.float64)
        for subdiv in (0, 1):
            offsets = character_offsets(card, subdiv, self.folder)
            tears = mouth_tears(model, card, offsets, subdiv)
            print(f"\nmouth tears, level {subdiv}: {tears}", file=sys.stderr)
            for state in ("closed", "open"):
                with self.subTest(subdiv=subdiv, state=state):
                    entry = tears[state]
                    self.assertGreater(entry["edges"], 500)
                    # the mouth's old handling (the lining with the nearest skin in space,
                    # no repair) tore 1.9 % and 4.6 % of these edges on levels 0 and 1; the
                    # smooth lining and the repair keep them under 1 %. The target's mouth is
                    # cut open at the cavity, so its lips stand apart and the fit leaves them
                    # unsealed (FitOptions.lip_seal_max_gap)
                    bound = 0.01
                    self.assertLessEqual(entry["torn"], bound * entry["edges"])

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


class TestPointedEars(unittest.TestCase):
    """
    The fit of pointed ears that stand off from the head (``pointed_ear_target``: a gap behind
    each ear, a tip 20 mm up and back, ``masks.ear`` with its attachment line), with the ear
    split (the default with masks.ear) and without it, from the same fitted parameters. Without
    it, the skin of the head behind the ear stretches up to the tip, which stays uncovered;
    with it, the tip is reached, the gap stays open and the ears fold only where the reference
    does. The measures are those of ``metrics.measure`` on the ears of the mask.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        folder = pathlib.Path(cls.tmp.name) / "handoff"
        target = pointed_ear_target()
        write_handoff(
            folder,
            target["vertices"],
            target["triangles"],
            target["landmarks"],
            masks={
                "ear": {"vertices": target["ear"], "attachment": target["attachment"]}
            },
        )
        cls.reference = Reference.from_handoff(load_handoff(folder))
        model = fit_model()
        options = FitOptions()
        params, _, R, t = fit_parameters(model, cls.reference, options)
        from opensculptboy.fit.register import _character_from

        card = _character_from(model, params, "pointed")
        cls.after = fit_levels(cls.reference, card, R, t, 1, options, model)
        before = FitOptions(ear_split=False)
        cls.before = fit_levels(cls.reference, card, R, t, 1, before, model)
        # the same target written as a game writes its head: a wider ear mask, cut out as a
        # piece of its own and stitched to the head by coincident vertices
        stitched = stitched_ear_target(target)
        folder = pathlib.Path(cls.tmp.name) / "stitched"
        write_handoff(
            folder,
            stitched["vertices"],
            stitched["triangles"],
            target["landmarks"],
            masks={
                "ear": {
                    "vertices": stitched["ear"],
                    "attachment": stitched["attachment"],
                }
            },
        )
        cls.stitched_reference = Reference.from_handoff(load_handoff(folder))
        cls.stitched = fit_levels(cls.stitched_reference, card, R, t, 1, options, model)
        lines = ["pointed ears (ears of the mask; before -> after, mm and degrees):"]

        def root(result, subdiv):
            return result.measures[subdiv]["final"]["ears_root"]

        for subdiv in (0, 1):
            b, a = (
                r.measures[subdiv]["final"]["ears"] for r in (cls.before, cls.after)
            )
            sb, sa = (
                r.measures[subdiv]["final"]["ears_surround"]["reverse_mm"]
                for r in (cls.before, cls.after)
            )
            lines.append(
                f"  level {subdiv}: reverse mean {b['reverse_mm']['mean']:.3f} -> "
                f"{a['reverse_mm']['mean']:.3f}, p95 {b['reverse_mm']['p95']:.3f} -> "
                f"{a['reverse_mm']['p95']:.3f}, max {b['reverse_mm']['max']:.2f} -> "
                f"{a['reverse_mm']['max']:.2f}; faces over 1 mm {b['faces_over_1mm']:.4f} "
                f"-> {a['faces_over_1mm']:.4f}; folds {b['folded_over_60deg']:.4f} -> "
                f"{a['folded_over_60deg']:.4f}, not in the reference "
                f"{b['folds_not_in_reference']:.4f} -> {a['folds_not_in_reference']:.4f}; "
                f"normal angle p95 {b['angle_geometric_deg']['p95']:.1f} -> "
                f"{a['angle_geometric_deg']['p95']:.1f}, max "
                f"{b['angle_geometric_deg']['max']:.1f} -> "
                f"{a['angle_geometric_deg']['max']:.1f}, over 90 degrees "
                f"{b['facing_away']:.4f} -> {a['facing_away']:.4f}; around the ears "
                "reverse mean "
                f"{sb['mean']:.3f} -> {sa['mean']:.3f}, p95 {sb['p95']:.3f} -> "
                f"{sa['p95']:.3f}; root defects "
                f"{root(cls.before, subdiv)['defects']:.4f} -> "
                f"{root(cls.after, subdiv)['defects']:.4f}"
            )
        for subdiv in (0, 1):
            m = cls.stitched.measures[subdiv]["final"]
            lines.append(
                f"  stitched, level {subdiv}: ears reverse mean "
                f"{m['ears']['reverse_mm']['mean']:.3f}, folds not in the reference "
                f"{m['ears']['folds_not_in_reference']:.4f}, over 90 degrees "
                f"{m['ears']['facing_away']:.4f}; root defects "
                f"{m['ears_root']['defects']:.4f} of {m['ears_root']['n']}"
            )
        print("\n" + "\n".join(lines), file=sys.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def ears(self, result, level):
        return result.measures[level]["final"]["ears"]

    def test_target(self):
        self.assertIn("ear", self.reference.regions)
        self.assertIn("ear_attachment", self.reference.regions)
        self.assertIn("ear_top_l", self.reference.landmarks)

    def test_tip_is_reached(self):
        # the reference's ears lie on the fit: 0.15 and 0.45 mm (mean, p95) on level 1, where
        # the fit without the split leaves the tip uncovered (0.36 and 1.8 mm)
        for subdiv, mean, p95 in ((0, 0.65, 2.2), (1, 0.2, 0.6)):
            with self.subTest(level=subdiv):
                after = self.ears(self.after, subdiv)["reverse_mm"]
                before = self.ears(self.before, subdiv)["reverse_mm"]
                self.assertLess(after["mean"], mean)
                self.assertLess(after["p95"], p95)
                self.assertLess(after["p95"], 0.6 * before["p95"])
                self.assertLess(after["max"], before["max"])

    def test_no_skin_across_the_gap(self):
        # a triangle that bridges the gap behind the ear has its middle in the air (0.4 % of
        # the ears' triangles over 1 mm without the split, 0.08 % with it, on level 1); the head
        # around the ear stays covered (0.08 and 0.27 mm, mean and p95)
        after, before = self.ears(self.after, 1), self.ears(self.before, 1)
        self.assertLess(after["faces_over_1mm"], 0.003)
        self.assertLess(after["faces_over_1mm"], 0.5 * before["faces_over_1mm"])
        around = self.after.measures[1]["final"]["ears_surround"]["reverse_mm"]
        self.assertLess(around["mean"], 0.12)
        self.assertLess(around["p95"], 0.4)

    def test_folds_and_normals(self):
        # the ears fold where the reference folds (the rim of the tip) and hardly anywhere
        # else: 1.5 % of their vertices fold by 30 degrees more than the reference near them on
        # level 0 (3.2 % without the split) and 0.9 % on level 1 (0.8 %); a vertex whose normal
        # turns from the reference's by more than 90 degrees is a flipped sheet or a spike, or
        # sits on the rim of the tip with its closest point on the other side: 0.06 % of them
        # on level 1 (0.12 %), and the normals of 95 % agree to 15 degrees
        for subdiv, share in ((0, 0.025), (1, 0.015)):
            with self.subTest(level=subdiv):
                after = self.ears(self.after, subdiv)
                before = self.ears(self.before, subdiv)
                self.assertLess(after["folds_not_in_reference"], share)
                self.assertLessEqual(
                    after["folds_not_in_reference"],
                    1.2 * before["folds_not_in_reference"],
                )
                self.assertLess(after["facing_away"], 0.005)
        self.assertLess(self.ears(self.after, 1)["angle_geometric_deg"]["p95"], 15.0)

    def test_root_blends_into_the_head(self):
        # the strip within 3 mm of the attachment line, where the ear meets the head: a crack
        # along the root shows there as folded and flipped slivers (on level 1, 1.1 % of its
        # vertices, 3.3 % before relabel_ears and the reverse term's cut). A game's ear,
        # stitched to the head by coincident vertices and holding the flat skin in front of the
        # root, fits as cleanly: none (1.4 % when the stitched cut was empty)
        root = self.after.measures[1]["final"]["ears_root"]
        self.assertGreater(root["n"], 100)
        self.assertLess(root["defects"], 0.02)
        root = self.stitched.measures[1]["final"]["ears_root"]
        self.assertGreater(root["n"], 100)
        self.assertLess(root["defects"], 0.007)
        stitched = self.stitched.measures[1]["final"]["ears"]
        self.assertLess(stitched["reverse_mm"]["p95"], 0.6)
        self.assertLess(stitched["facing_away"], 0.005)

    def test_stitched_ear_has_its_cut(self):
        split = EarSplit(self.stitched_reference)
        self.assertGreater(split.ear.cut.sum(), 0)
        self.assertGreater(split.head.cut.sum(), 0)

    def test_rest_of_the_figure(self):
        # the ear split leaves the rest of the fit as it was
        for subdiv in (0, 1):
            after = self.after.measures[subdiv]["final"]["all"]["reverse_mm"]
            before = self.before.measures[subdiv]["final"]["all"]["reverse_mm"]
            self.assertLess(after["mean"], before["mean"] * 1.02)
        self.assertLess(
            self.after.measures[1]["final"]["all"]["reverse_mm"]["mean"],
            LEVEL1_BOUND_MM,
        )


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

# OpenSculptBoy
# Apache License, Version 2.0
"""
The transfer of ICT-FaceKit's expression shapes onto Anny's face
(:mod:`opensculptboy.expressions.authoring.ict`): the name table, the side-aware
correspondences, the frame, the smooth extension and the model rows, with synthetic data; and
on the downloaded ICT data (local only), the shapes and their sides.
"""

import types
import unittest

import numpy as np
import torch

from opensculptboy.expressions.authoring import ict
from test.markers import local_only


def grid(nx: int, ny: int, z: float = 0.0, spacing: float = 0.01):
    """A flat quad grid in the xy plane: vertices (nx*ny, 3), quads, triangles."""
    xs, ys = np.meshgrid(
        np.arange(nx) * spacing, np.arange(ny) * spacing, indexing="ij"
    )
    V = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)], 1)
    idx = np.arange(nx * ny).reshape(nx, ny)
    quads = np.stack(
        [
            idx[:-1, :-1].ravel(),
            idx[1:, :-1].ravel(),
            idx[1:, 1:].ravel(),
            idx[:-1, 1:].ravel(),
        ],
        1,
    )
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    return V, quads, tris


class TestNames(unittest.TestCase):
    def test_every_action_but_the_tongue(self):
        from anny.models.facial_actions import FACIAL_ACTION_LABELS

        self.assertEqual(
            sorted(set(FACIAL_ACTION_LABELS) - set(ict.ARKIT_FROM_ICT)), ["tongueOut"]
        )
        self.assertEqual(set(ict.ARKIT_FROM_ICT) - set(FACIAL_ACTION_LABELS), set())
        shapes = [n for names in ict.ARKIT_FROM_ICT.values() for n in names]
        self.assertEqual(len(shapes), 53)
        self.assertEqual(len(set(shapes)), 53)
        self.assertEqual(
            ict.ARKIT_FROM_ICT["browInnerUp"], ("browInnerUp_L", "browInnerUp_R")
        )
        self.assertEqual(ict.ARKIT_FROM_ICT["mouthSmileLeft"], ("mouthSmile_L",))


class TestGeometry(unittest.TestCase):
    def test_share(self):
        d = np.zeros((100, 3))
        d[:10, 2] = np.linspace(0.001, 0.01, 10)
        s = ict.share(d)
        self.assertEqual(s[50], 0.0)
        self.assertAlmostEqual(s[9], 1.0)
        self.assertTrue((s <= 1.0).all())

    def test_side_correspondences_keep_the_lip(self):
        # two "lips": an upper sheet at z = +1 mm (share 0) and a lower sheet at z = -1 mm
        # (share 1), both 10 cm wide
        Vu, _, Tu = grid(11, 3, z=0.001)
        Vl, _, Tl = grid(11, 3, z=-0.001)
        V = np.concatenate([Vu, Vl])
        T = np.concatenate([Tu, Tl + len(Vu)])
        target_share = np.concatenate([np.zeros(len(Vu)), np.ones(len(Vl))])
        # upper-lip points that lie a little below the middle: their closest point is the lower lip
        points = np.array([[0.05, 0.01, -0.0004], [0.03, 0.01, -0.0002]])
        point_share = np.zeros(2)
        tri, bary, d = ict.side_correspondences(
            points, [point_share], V, T, [target_share]
        )
        hit = np.einsum("hk, hkd -> hd", bary, V[T[tri]])
        np.testing.assert_allclose(hit[:, 2], 0.001, atol=1e-9)
        np.testing.assert_allclose(d, [0.0014, 0.0012], atol=1e-9)
        # a lower-lip point stays on the lower lip
        tri, bary, _ = ict.side_correspondences(
            points[:1], [np.ones(1)], V, T, [target_share]
        )
        self.assertAlmostEqual(
            np.einsum("k, kd -> d", bary[0], V[T[tri[0]]])[2], -0.001
        )

    def test_sides_left_alone_when_they_agree(self):
        V, _, T = grid(5, 5)
        share = np.linspace(0, 1, len(V))
        points = V[[3, 12, 20]] + [0, 0, 0.002]
        tri, bary, d = ict.side_correspondences(
            points, [share[[3, 12, 20]]], V, T, [share]
        )
        np.testing.assert_allclose(d, 0.002, atol=1e-9)

    def test_to_anny_frame(self):
        rng = np.random.default_rng(0)
        A = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        F = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        d = rng.normal(size=(4, 3))
        out = ict.to_anny_frame(d, A, F, 2.0)
        np.testing.assert_allclose(out, 2.0 * (F.T @ A @ d.T).T)

    def test_sample(self):
        V, _, T = grid(3, 3)
        delta = np.zeros_like(V)
        delta[:, 1] = V[:, 0]  # an offset that grows along x
        out = ict.sample(delta, T, np.array([0]), np.array([[1 / 3, 1 / 3, 1 / 3]]))
        np.testing.assert_allclose(out[0, 1], V[T[0], 0].mean())


class TestExtend(unittest.TestCase):
    def setUp(self):
        self.V, self.quads, _ = grid(21, 5)
        self.edges = ict.edges_of(self.quads)
        self.n = len(self.V)

    def test_known_held_and_smooth_toward_fixed(self):
        x = self.V[:, 0]
        known = x <= 0.05 + 1e-9
        fixed = x >= 0.15 - 1e-9
        values = np.zeros((self.n, 3))
        values[:, 2] = 0.01
        field = ict.extend(self.n, self.edges, known, values, fixed=fixed, smooth=0.05)
        np.testing.assert_allclose(field[known, 2], 0.01, atol=5e-4)
        np.testing.assert_allclose(field[fixed], 0.0, atol=1e-6)
        # in between the field falls without overshoot, column by column
        middle = [
            field[np.isclose(x, xi), 2].mean() for xi in np.arange(0.05, 0.16, 0.01)
        ]
        self.assertTrue(np.all(np.diff(middle) <= 1e-9))
        self.assertTrue(0.0 < middle[5] < 0.01)

    def test_prior_kept_where_nothing_is_known(self):
        # a second, unconnected grid keeps its prior; the first follows its known values
        V2, quads2, _ = grid(3, 3)
        n2 = len(V2)
        edges = np.concatenate([self.edges, ict.edges_of(quads2) + self.n])
        n = self.n + n2
        prior = np.zeros((n, 3))
        prior[self.n :, 0] = 0.004
        known = np.zeros(n, bool)
        known[: self.n] = True
        values = np.zeros((n, 3))
        field = ict.extend(n, edges, known, values, prior=prior)
        np.testing.assert_allclose(field[self.n :, 0], 0.004, atol=1e-6)
        np.testing.assert_allclose(field[: self.n], 0.0, atol=1e-6)


class TestModelRows(unittest.TestCase):
    def test_apply_shapes(self):
        labels = ["browInnerUp", "jawOpen"]
        model = types.SimpleNamespace(
            _n_phenotype_blendshapes=3,
            facial_action_labels=labels,
            blendshapes=torch.zeros((5, 4, 3)),
        )
        model.blendshapes[4] = 1.0
        new = np.full((4, 3), 2.0)
        ict.apply_shapes(model, {"jawOpen": new}, strength=0.5)
        np.testing.assert_allclose(model.blendshapes[4].numpy(), 1.5)
        np.testing.assert_allclose(model.blendshapes[3].numpy(), 0.0)
        self.assertEqual(ict.facial_action_rows(model), (3, labels))


@local_only(
    "ICT-FaceKit's files in ANNY_CACHE_DIR/faces (python -m anny.faces.authoring.sources)"
)
class TestIctData(unittest.TestCase):
    def setUp(self):
        from anny.faces.authoring.sources import cache_dir

        root = cache_dir() / "ICT-FaceKit"
        if not (root / "FaceXModel").exists():
            self.skipTest("ICT-FaceKit is not downloaded")
        self.shapes = ict.load_expressions(root)

    def test_shapes_and_sides(self):
        names = {n for ns in ict.ARKIT_FROM_ICT.values() for n in ns}
        self.assertLessEqual(names, set(self.shapes.names))
        self.assertEqual(self.shapes.neutral.shape, (ict.SKIN_VERTICES, 3))
        sides = ict.check_sides(
            {
                a: self.shapes.arkit(a)
                for a in ("browDownLeft", "mouthSmileLeft", "eyeBlinkLeft")
            },
            self.shapes.neutral,
        )
        for name, x in sides.items():
            self.assertGreater(x, 0.0, name)
        # the shapes are a few millimetres to a few centimetres
        size = np.linalg.norm(self.shapes.arkit("jawOpen"), axis=1).max()
        self.assertTrue(0.005 < size < 0.06, size)


if __name__ == "__main__":
    unittest.main()

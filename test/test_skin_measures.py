# OpenSculptBoy
# Apache License, Version 2.0
"""
The measures of the posed skin (``anny.correctives.authoring.skin_measures``) on synthetic
surfaces: the stretch of a sheet pulled by a known factor, and the hollow of a surface with
two ridges of a known height.
"""

import unittest

import numpy as np

from anny.correctives.authoring import skin_measures as SM


def sheet(nx=41, ny=21, width=0.4, length=0.2):
    """A flat sheet in the plane z = 0 with normals along +z: vertices and triangles."""
    x = np.linspace(-width / 2, width / 2, nx)
    y = np.linspace(-length / 2, length / 2, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    a = (i * ny + j).ravel()
    b, c, d = a + ny, a + ny + 1, a + 1
    return V, np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])


class TestStretch(unittest.TestCase):
    def test_a_pulled_sheet(self):
        V, F = sheet()
        P = V * np.array([2.0, 0.8, 1.0])
        larger, smaller, area = SM.principal_stretches(V, P, F)
        np.testing.assert_allclose(larger, 2.0, atol=1e-9)
        np.testing.assert_allclose(smaller, 0.8, atol=1e-9)
        np.testing.assert_allclose(area.sum(), 0.4 * 0.2, atol=1e-12)

    def test_a_turn_stretches_nothing(self):
        V, F = sheet()
        t = 0.7
        R = np.array(
            [[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]]
        )
        larger, smaller, _ = SM.principal_stretches(V, V @ R.T + 0.3, F)
        np.testing.assert_allclose(larger, 1.0, atol=1e-9)
        np.testing.assert_allclose(smaller, 1.0, atol=1e-9)

    def test_summary_of_a_band(self):
        V, F = sheet()
        P = V.copy()
        right = V[:, 0] > 0.1
        P[right, 0] = 0.1 + 3.0 * (V[right, 0] - 0.1)
        larger, smaller, area = SM.principal_stretches(V, P, F)
        s = SM.stretch_summary(larger, smaller, area, np.ones(len(F), bool))
        self.assertAlmostEqual(s["peak"], 3.0, 9)
        self.assertAlmostEqual(s["area_over_2"], 0.1 * 0.2, 9)
        self.assertAlmostEqual(s["mean"], 0.75 * 1.0 + 0.25 * 3.0, 9)
        self.assertAlmostEqual(s["p95"], 3.0, 9)
        self.assertAlmostEqual(SM.area_quantile(larger, area, 0.5), 1.0, 9)

    def test_armpit_triangles_lie_below_the_joint(self):
        V, F = sheet()
        V = V[:, [0, 2, 1]] * np.array([1.0, 1.0, 8.0])
        shoulder = np.array([0.0, 0.0, 0.5])
        mask = SM.armpit_triangles(V, F, shoulder)
        centres = V[F].mean(1)
        self.assertTrue(mask.any())
        self.assertTrue((centres[mask, 2] < shoulder[2]).all())
        self.assertTrue(
            (np.linalg.norm(centres[mask] - shoulder, axis=1) < 0.075 * 1.6).all()
        )


class TestHollow(unittest.TestCase):
    FRAME = dict(
        centre=np.zeros(3),
        axis=np.array([0.0, 1.0, 0.0]),
        outward=np.array([0.0, 0.0, 1.0]),
        half_width=0.15,
    )

    def test_a_dome_has_no_hollow(self):
        V, F = sheet()
        V[:, 2] = -2.0 * V[:, 0] ** 2
        got = SM.hollow_depth(V, F, **self.FRAME)
        self.assertIsNotNone(got["profile"])
        self.assertLess(got["depth"], 5e-4)

    def test_two_ridges_with_a_floor(self):
        V, F = sheet(nx=161)
        x = V[:, 0]
        V[:, 2] = 0.03 * (1 - np.exp(-((x / 0.03) ** 2))) - 4.0 * np.maximum(
            np.abs(x) - 0.08, 0.0
        )
        got = SM.hollow_depth(V, F, **self.FRAME)
        want = 0.03 * (1 - np.exp(-((0.08 / 0.03) ** 2)))
        self.assertAlmostEqual(got["depth"], want, delta=0.002)
        self.assertAlmostEqual(got["floor"], 0.0, delta=0.01)
        self.assertAlmostEqual(got["front"], -0.08, delta=0.01)
        self.assertAlmostEqual(got["back"], 0.08, delta=0.01)

    def test_the_inner_side_is_ignored(self):
        V, F = sheet(nx=161)
        x = V[:, 0]
        V[:, 2] = 0.03 * (1 - np.exp(-((x / 0.03) ** 2)))
        back = V.copy()
        back[:, 2] = 0.1
        both_V = np.concatenate([V, back])
        both_F = np.concatenate([F, F[:, ::-1] + len(V)])
        got = SM.hollow_depth(both_V, both_F, **self.FRAME)
        alone = SM.hollow_depth(V, F, **self.FRAME)
        self.assertAlmostEqual(got["depth"], alone["depth"], 9)

    def test_the_line_of_a_cut_is_connected(self):
        V, F = sheet(nx=41)
        V[:, 2] = -2.0 * V[:, 0] ** 2
        segments, rows, edges = SM.section(V, F, np.zeros(3), np.array([0.0, 1.0, 0.0]))
        line, seed, closed = SM.section_line(segments, edges, len(segments) // 2)
        self.assertFalse(closed)
        self.assertEqual(len(line), len(segments) + 1)
        np.testing.assert_allclose(line[seed], segments[len(segments) // 2, 0])
        steps = np.diff(line[:, 0])
        self.assertTrue((steps >= -1e-12).all() or (steps <= 1e-12).all())
        np.testing.assert_allclose(line[:, 1], 0.0, atol=1e-12)
        self.assertAlmostEqual(abs(line[0, 0]), 0.2, 9)
        self.assertAlmostEqual(abs(line[-1, 0]), 0.2, 9)

    def test_a_limb_bulges_and_a_groove_in_it_dips(self):
        n, m, radius = 96, 21, 0.05
        angle = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        along = np.linspace(-0.1, 0.1, m)
        A, Y = np.meshgrid(angle, along, indexing="ij")
        dip = 0.012 * np.exp(-((np.angle(np.exp(1j * (A - np.pi / 2))) / 0.35) ** 2))
        R = radius - dip
        tube = np.stack([R * np.cos(A), Y, R * np.sin(A)], -1).reshape(-1, 3)
        round_tube = np.stack([radius * np.cos(A), Y, radius * np.sin(A)], -1).reshape(
            -1, 3
        )
        i, j = np.meshgrid(np.arange(n), np.arange(m - 1), indexing="ij")
        a = (i * m + j).ravel()
        b = (((i + 1) % n) * m + j).ravel()
        F = np.concatenate([np.stack([a, a + 1, b + 1], 1), np.stack([a, b + 1, b], 1)])
        frame = dict(
            centre=np.array([0.0, 0.0, radius]),
            axis=np.array([0.0, 1.0, 0.0]),
            outward=np.array([0.0, 0.0, 1.0]),
            half_width=0.04,
        )
        segments, rows, edges = SM.section(tube, F, frame["centre"], frame["axis"])
        line, seed, closed = SM.section_line(segments, edges, 0)
        self.assertTrue(closed)
        self.assertEqual(len(line), len(segments))
        self.assertEqual(seed, len(line) // 2 - 1)
        height = (radius - dip[:, 0]) * np.sin(angle)
        near = np.abs(radius * np.cos(angle)) < 0.04
        want = height[near & (height > 0)].max() - height[n // 4]
        for direction in (1.0, -1.0):
            with self.subTest(direction=direction):
                frame["axis"] = np.array([0.0, direction, 0.0])
                self.assertLess(SM.hollow_depth(round_tube, F, **frame)["depth"], 1e-4)
                got = SM.hollow_depth(tube, F, **frame)
                self.assertAlmostEqual(got["depth"], want, delta=5e-4)
                self.assertLess(abs(got["floor"]), 0.005)
                self.assertLess(got["front"], 0.0)
                self.assertGreater(got["back"], 0.0)
                self.assertTrue((np.diff(got["profile"][:, 0]) > -1e-9).all())

    def test_frame_of_a_raised_arm(self):
        shoulder = np.array([0.2, 0.0, 1.4])
        elbow = shoulder + 0.3 * np.array([0.5, 0.0, 0.866])
        centre = np.array([0.24, 0.01, 1.33])
        axis, outward, across = SM.armpit_frame(shoulder, elbow, centre)
        self.assertAlmostEqual(axis[1], 0.0, 12)
        M = np.stack([axis, outward, across])
        np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-12)
        self.assertGreater(outward @ (centre - shoulder), 0)
        self.assertGreater(abs(across[1]), 0.9)


if __name__ == "__main__":
    unittest.main()

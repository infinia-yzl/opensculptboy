# OpenSculptBoy
# Apache License, Version 2.0
"""
The slide of the skin along a posed surface (``anny.correctives.authoring.slide``) on
synthetic surfaces: closest points, the free zone, a flat sheet with a stretched band and a
sheet bent over a cylinder.
"""

import unittest

import numpy as np

from anny.correctives.authoring import skin_measures as SM
from anny.correctives.authoring import slide as SL


def sheet(nx=41, ny=11, width=0.4, length=0.1):
    x = np.linspace(-width / 2, width / 2, nx)
    y = np.linspace(-length / 2, length / 2, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    a = (i * ny + j).ravel()
    b, c, d = a + ny, a + ny + 1, a + 1
    return V, np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])


def uneven(x, width=0.4):
    """Positions along the sheet with the left half at half length and the right at 1.5."""
    return np.where(x < 0, -width / 4 + 0.5 * x, -width / 4 + 1.5 * x)


class TestSurface(unittest.TestCase):
    def test_closest_points_of_one_triangle(self):
        a, b, c = np.eye(3)[0] * 0, np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        points = np.array(
            [
                [0.2, 0.2, 0.5],
                [-1.0, -1.0, 0.3],
                [2.0, -0.5, 0.0],
                [0.5, -1.0, 2.0],
                [1.0, 1.0, 1.0],
            ]
        )
        want = np.array(
            [
                [0.2, 0.2, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
            ]
        )
        np.testing.assert_allclose(
            SL.closest_on_triangles(points, a, b, c), want, atol=1e-12
        )

    def test_projection_onto_a_sheet(self):
        V, F = sheet()
        rng = np.random.default_rng(3)
        points = rng.uniform([-0.19, -0.04, -0.01], [0.19, 0.04, 0.01], (200, 3))
        found = SL.Surface(V, F).project(points)
        np.testing.assert_allclose(found[:, :2], points[:, :2], atol=1e-12)
        np.testing.assert_allclose(found[:, 2], 0.0, atol=1e-12)
        outside = SL.Surface(V, F).project(np.array([[0.3, 0.0, 0.02]]))
        np.testing.assert_allclose(outside, [[0.2, 0.0, 0.0]], atol=1e-12)

    def test_own_skin_leaves_out_other_bones(self):
        indices = np.array([[0, 1, 2], [0, 3, 0], [4, 1, 0], [2, 2, 2]])
        weights = np.array(
            [[0.5, 0.5, 0.0], [0.96, 0.04, 0.0], [0.2, 0.8, 0.0], [0.4, 0.3, 0.3]]
        )
        own = SL.own_skin(indices, weights, [0, 1, 2])
        np.testing.assert_array_equal(own, [True, True, False, True])
        np.testing.assert_array_equal(
            SL.own_skin(indices, weights, [0, 1, 2], share=0.01),
            [True, False, False, True],
        )

    def test_free_zone_keeps_a_fixed_rim(self):
        V, F = sheet()
        free = SL.free_zone(V, F, np.zeros(3), 0.04)
        self.assertTrue(free.any())
        self.assertTrue((np.linalg.norm(V[free], axis=1) < 0.04).all())
        touching = F[free[F].any(1)]
        rim = np.unique(touching[~free[touching]])
        self.assertTrue(len(rim) > 0)
        self.assertTrue((np.linalg.norm(V[rim], axis=1) < 0.04).all())


class TestSlide(unittest.TestCase):
    def test_a_flat_sheet_gets_its_rest_layout_back(self):
        V, F = sheet()
        P = V.copy()
        P[:, 0] = uneven(V[:, 0])
        free = np.abs(V[:, 0]) < 0.2 - 1e-9
        before, _, area = SM.principal_stretches(V, P, F)
        self.assertAlmostEqual(before.max(), 1.5, 9)
        S = SL.slide(V, P, F, free, iterations=200)
        larger, smaller, _ = SM.principal_stretches(V, S, F)
        np.testing.assert_allclose(S[~free], P[~free], atol=0)
        np.testing.assert_allclose(S[:, 2], 0.0, atol=1e-12)
        self.assertLess(larger.max(), 1.02)
        self.assertGreater(smaller.min(), 0.98)

    def test_a_bent_sheet_stays_on_its_surface(self):
        V, F = sheet()
        radius = 0.15
        angle = uneven(V[:, 0]) / radius
        P = np.stack([radius * np.sin(angle), V[:, 1], radius * np.cos(angle)], 1)
        free = np.abs(V[:, 0]) < 0.2 - 1e-9
        before, _, area = SM.principal_stretches(V, P, F)
        S = SL.slide(V, P, F, free, iterations=200)
        after, _, _ = SM.principal_stretches(V, S, F)
        distance = np.hypot(S[:, 0], S[:, 2])
        self.assertLess(distance.max(), radius + 1e-9)
        self.assertGreater(distance.min(), radius * np.cos(1.5 * 0.01 / radius) - 1e-9)
        np.testing.assert_allclose(S[~free], P[~free], atol=0)
        self.assertGreater(SM.area_quantile(before, area, 0.95), 1.45)
        self.assertLess(SM.area_quantile(after, area, 0.95), 1.05)
        self.assertLess(np.abs(S[:, 1] - V[:, 1]).max(), 0.004)

    def test_a_near_part_of_the_body_is_left_alone(self):
        V, F = sheet()
        P = V.copy()
        P[:, 0] = uneven(V[:, 0])
        other = V + np.array([0.0, 0.0, 0.002])
        rest = np.concatenate([V, other + np.array([0.0, 0.0, 0.5])])
        posed = np.concatenate([P, other])
        faces = np.concatenate([F, F + len(V)])
        own = np.arange(len(rest)) < len(V)
        free = SL.free_zone(rest, faces, np.zeros(3), 0.19, allowed=own)
        self.assertFalse(free[~own].any())
        S = SL.slide(rest, posed, faces, free, iterations=100)
        np.testing.assert_allclose(S[own][:, 2], 0.0, atol=1e-12)
        np.testing.assert_allclose(S[~own], posed[~own], atol=0)
        before, _, area = SM.principal_stretches(rest, posed, faces)
        after, _, _ = SM.principal_stretches(rest, S, faces)
        zone = free[faces].any(1)
        self.assertLess(after[zone].max(), before[zone].max() - 0.2)

    def test_the_guard_halves_a_move_that_squeezes(self):
        V, F = sheet(nx=21, ny=11, width=0.2, length=0.1)
        free = (np.abs(V[:, 0]) < 0.1 - 1e-9) & (np.abs(V[:, 1]) < 0.05 - 1e-9)
        ids = np.where(free)[0]
        column = -np.ones(len(V), dtype=np.int64)
        column[ids] = np.arange(len(ids))
        flat = SL._rest_triangles(V, F)
        inverse = np.linalg.inv(np.stack([flat[:, 1], flat[:, 2]], axis=2))
        floor = SL.GIVE * np.minimum(SL._narrowest(V, F, inverse), 0.5)
        squeezed = int(np.argmin(np.linalg.norm(V - [-0.05, 0.0, 0.0], axis=1)))
        easy = int(np.argmin(np.linalg.norm(V - [0.05, 0.0, 0.0], axis=1)))
        moved = V[ids].copy()
        moved[column[squeezed]] += [0.0099, 0.0, 0.0]
        moved[column[easy]] += [0.002, 0.0, 0.0]
        whole = V.copy()
        whole[ids] = moved
        self.assertLess(SL._narrowest(whole, F, inverse).min(), 0.02)
        placed = SL._guarded(
            V.copy(), ids, moved, F, column, inverse, floor, SL.Surface(V, F)
        )
        kept = V.copy()
        kept[ids] = placed
        self.assertTrue((SL._narrowest(kept, F, inverse) >= floor).all())
        self.assertTrue((SL._normals(kept, F)[:, 2] > 0).all())
        np.testing.assert_allclose(placed[column[easy]], moved[column[easy]], atol=0)
        step = placed[column[squeezed], 0] - V[squeezed, 0]
        self.assertGreater(step, 0.002)
        self.assertLess(step, 0.006)
        rest_of_them = np.ones(len(ids), bool)
        rest_of_them[[column[squeezed], column[easy]]] = False
        np.testing.assert_allclose(placed[rest_of_them], V[ids][rest_of_them], atol=0)

    def test_the_guard_leaves_a_harmless_slide_alone(self):
        V, F = sheet()
        P = V.copy()
        P[:, 0] = uneven(V[:, 0])
        free = np.abs(V[:, 0]) < 0.2 - 1e-9
        with_guard = SL.slide(V, P, F, free, iterations=30)
        without = SL.slide(V, P, F, free, iterations=30, guard=None)
        np.testing.assert_allclose(with_guard, without, atol=1e-9)

    def test_only_the_free_zone_moves(self):
        V, F = sheet()
        P = V.copy()
        P[:, 0] = uneven(V[:, 0])
        free = SL.free_zone(V, F, np.zeros(3), 0.045)
        S = SL.slide(V, P, F, free, iterations=5)
        np.testing.assert_allclose(S[~free], P[~free], atol=0)
        self.assertGreater(np.abs(S[free] - P[free]).max(), 1e-3)
        even = SL.free_zone(V, F, np.array([0.1, 0.0, 0.0]), 0.03)
        same = SL.slide(V, P, F, even, iterations=5)
        np.testing.assert_allclose(same, P, atol=1e-9)


if __name__ == "__main__":
    unittest.main()

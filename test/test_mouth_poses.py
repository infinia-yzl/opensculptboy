# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of a fitted character's mouth expressions from its reference's mouth poses
(``opensculptboy.mouth.poses``, docs/mouth.md "Mouth expressions from the reference"): the
motion of a posed reference carried onto a fitted surface, the file of the poses, and the
targets, turns and visemes that the GLB and VRM exports make of them.
"""

import json
import pathlib
import tempfile
import unittest
import warnings

import numpy as np
from opensculptboy import Character, export_glb
from opensculptboy.fit.handoff import load_handoff
from opensculptboy.fit.reference import Reference
from opensculptboy.mouth import poses as P
from opensculptboy.mouth.rig import MOUTH_RIG
from test.gltf_reader import GLB, evaluate_nodes, structure_errors
from test.test_mouth_inner import _body, synthetic_mouth
from test.test_mouth_lips import _write_rig_glb


def _grid(n=41, size=0.06):
    """A grid of n by n points in glTF's frame (the xy plane at z = 0.05, y up) and its
    triangles; the jaw takes the lower half's weight, more toward the bottom."""
    u = np.linspace(-size / 2, size / 2, n)
    X, Y = np.meshgrid(u, u)
    V = np.c_[X.ravel(), Y.ravel(), np.full(n * n, 0.05)]
    k = np.arange(n * n).reshape(n, n)
    a, b, c, d = k[:-1, :-1], k[:-1, 1:], k[1:, :-1], k[1:, 1:]
    T = np.concatenate(
        [np.c_[a.ravel(), b.ravel(), d.ravel()], np.c_[a.ravel(), d.ravel(), c.ravel()]]
    )
    jaw = np.clip(-V[:, 1] / (size / 2), 0.0, 1.0)
    return V, T, jaw


class TestCarry(unittest.TestCase):
    def test_reference_motion_on_the_fitted_surface(self):
        V, T, jaw = _grid()
        soup = V[T.ravel()]
        J = np.zeros((len(soup), 4), int)
        J[:, 1] = 1
        W = np.zeros((len(soup), 4))
        W[:, 1] = jaw[T.ravel()]
        W[:, 0] = 1.0 - W[:, 1]
        turn = [np.sin(np.radians(5)), 0.0, 0.0, np.cos(np.radians(5))]
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            _write_rig_glb(folder / "replica.glb", soup, J, W, [0.0, 0.0, -0.05])
            appearance = {
                "parts": [{"name": "head", "role": "skin"}],
                "mouth": {
                    "rig": {"jaw": "lower_jaw"},
                    "poses": {
                        "closed": {"known": True, "joints": {}},
                        "open": {
                            "known": True,
                            "joints": {"lower_jaw": {"rotation": turn}},
                        },
                    },
                },
            }
            (folder / "appearance.json").write_text(json.dumps(appearance))
            handoff = load_handoff(folder)
            reference = Reference.from_handoff(handoff)
            fields = P.reference_pose_fields(handoff, reference)
            self.assertEqual(list(fields), ["open"])  # the closed pose moves nothing
            field = fields["open"][0]
            # the fitted surface: the grid itself, in Anny's frame
            from opensculptboy.fit.handoff import GLTF_TO_ANNY

            fitted = V @ GLTF_TO_ANNY.T
            values, known = P.carry_field(
                fitted, T, reference, field, np.ones(len(fitted), bool)
            )
        # each vertex of the soup that is a grid point carries its own motion
        from scipy.spatial import cKDTree

        d, i = cKDTree(reference.vertices).query(fitted)
        self.assertTrue((d < 1e-6).all())
        self.assertGreater(known.mean(), 0.8)  # the border's ring takes the extension
        np.testing.assert_allclose(values[known], field[i[known]], atol=1e-7)
        self.assertGreater(np.abs(values).max(), 1e-3)
        # the extension stays between its neighbours' values
        self.assertLess(
            np.abs(values[~known]).max(), np.abs(field).max() * 1.0001 + 1e-9
        )

    def test_lip_rows_off_the_surface_take_the_rig(self):
        """A lip's rows that lie off the reference's surface (its inner roll and lining)
        move as the reference's rig moves them: the blend of its joints' turns with the
        weights at their closest point on the same lip, applied to their own positions (the
        motion of the surface point alone stretched the lips' inner edge in wide poses)."""
        from opensculptboy.fit.handoff import GLTF_TO_ANNY
        from opensculptboy.fit.poses import _quaternion_matrix

        V, T, jaw = _grid()
        soup = V[T.ravel()]
        J = np.zeros((len(soup), 4), int)
        J[:, 1] = 1
        W = np.zeros((len(soup), 4))
        W[:, 1] = jaw[T.ravel()]
        W[:, 0] = 1.0 - W[:, 1]
        turn = [np.sin(np.radians(6)), 0.0, 0.0, np.cos(np.radians(6))]
        pivot = np.array([0.0, 0.0, -0.05])
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            _write_rig_glb(folder / "replica.glb", soup, J, W, pivot)
            appearance = {
                "parts": [{"name": "head", "role": "skin"}],
                "mouth": {
                    "rig": {"jaw": "lower_jaw"},
                    "poses": {
                        "open": {
                            "known": True,
                            "joints": {"lower_jaw": {"rotation": turn}},
                        },
                    },
                },
            }
            (folder / "appearance.json").write_text(json.dumps(appearance))
            handoff = load_handoff(folder)
            reference = Reference.from_handoff(handoff)
            field = P.reference_pose_fields(handoff, reference)["open"][0]
            transforms = P.reference_pose_transforms(handoff, reference)["open"]
        self.assertEqual(transforms[0].shape, (len(reference.vertices), 3, 3))
        # the transforms reproduce the reference's own motion at its vertices
        moved = np.einsum("nij,nj->ni", *transforms[:1], reference.vertices)
        np.testing.assert_allclose(
            moved + transforms[1] - reference.vertices, field, atol=1e-7
        )
        # the lower lip's rows, 4 mm behind the reference's surface
        lip = (jaw >= 0.5) & (np.abs(V[:, 0]) < 0.02)
        behind = V.copy()
        behind[lip, 2] -= 0.004
        fitted = behind @ GLTF_TO_ANNY.T
        fitted_side = np.where(lip, 2, 0).astype(np.int8)
        ref_jaw = W[:, 1]
        ref_side = np.where(ref_jaw >= P.LOWER_SHARE, 2, 1).astype(np.int8)
        sides = (fitted_side, ref_side)
        on_skin = np.ones(len(fitted), bool)
        with_rig, known = P.carry_field(
            fitted, T, reference, field, on_skin, sides, transforms=transforms
        )
        R = _quaternion_matrix(turn)
        # away from the lip's border, where the smoothing of rough spots meets the skin
        inside = lip & (jaw >= 0.6) & (np.abs(V[:, 0]) < 0.015)
        w = jaw[inside, None]
        p = behind[inside]
        want = (1 - w) * p + w * ((p - pivot) @ R.T + pivot)
        np.testing.assert_allclose(
            with_rig[inside], (want - p) @ GLTF_TO_ANNY.T, atol=1e-6
        )
        self.assertTrue(known[lip].all())
        # without the rig the rows lie too far from the surface to take its motion
        plain, known = P.carry_field(fitted, T, reference, field, on_skin, sides)
        self.assertFalse(known[lip].any())


class TestLiningSides(unittest.TestCase):
    def test_each_lip_claims_its_lining(self):
        # a strip: upper lip row, three lining rows, lower lip row
        rows, n = 5, 6
        V = np.array(
            [[x, 0.0, -r * 0.002] for r in range(rows) for x in range(n)], float
        )
        T = []
        for r in range(rows - 1):
            for c in range(n - 1):
                p, q = r * n + c, (r + 1) * n + c
                T += [[p, p + 1, q], [p + 1, q + 1, q]]
        side = np.zeros(len(V), np.int8)
        side[:n] = 1
        side[-n:] = 2
        off = np.zeros(len(V), bool)
        off[n:-n] = True
        got = P.lining_sides(np.asarray(T), side, off, 3).reshape(rows, n)
        self.assertTrue((got[0] == 0).all() and (got[-1] == 0).all())
        self.assertTrue((got[1] == 1).all())
        self.assertTrue((got[3] == 2).all())
        self.assertTrue((got[2] == 1).all())  # a tie goes to the upper lip
        self.assertTrue((P.lining_sides(np.asarray(T), side, off, 0) == 0).all())


class TestPiecesMove(unittest.TestCase):
    def test_pieces_cut_from_a_primitive_take_its_motion(self):
        """The inner mouth's pieces are rows of one primitive of the reference: each takes
        the posed rows of that primitive (a piece left at rest stood its tongue against the
        teeth in every open pose)."""
        from opensculptboy.fit.handoff import GLTF_TO_ANNY
        from opensculptboy.fit.poses import posed_parts

        V, T, jaw = _grid()
        soup = V[T.ravel()]
        J = np.zeros((len(soup), 4), int)
        J[:, 1] = 1
        W = np.zeros((len(soup), 4))
        W[:, 1] = jaw[T.ravel()]
        W[:, 0] = 1.0 - W[:, 1]
        turn = [np.sin(np.radians(8)), 0.0, 0.0, np.cos(np.radians(8))]
        lower = np.flatnonzero(W[:, 1] > 0.5)
        lower = lower[: len(lower) // 3 * 3]
        upper = np.flatnonzero(W[:, 1] <= 0.5)
        upper = upper[: len(upper) // 3 * 3]
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            _write_rig_glb(folder / "replica.glb", soup, J, W, [0.0, 0.0, -0.05])
            pieces = [
                {
                    "role": "tongue",
                    "mesh": "head",
                    "primitive": 0,
                    "vertices": lower.tolist(),
                },
                {
                    "role": "palate",
                    "mesh": "head",
                    "primitive": 0,
                    "vertices": upper.tolist(),
                },
            ]
            appearance = {
                "parts": [{"name": "head", "role": "skin"}],
                "mouth": {
                    "rig": {"jaw": "lower_jaw"},
                    "pieces": pieces,
                    "poses": {
                        "open": {
                            "known": True,
                            "joints": {"lower_jaw": {"rotation": turn}},
                        }
                    },
                },
            }
            (folder / "appearance.json").write_text(json.dumps(appearance))
            handoff = load_handoff(folder)
            whole = next(p for p in handoff.parts if not p.mouth_role)
            moved = posed_parts(handoff, {"lower_jaw": {"rotation": turn}})
        k_whole = handoff.parts.index(whole)
        full = moved[k_whole]
        self.assertIsNotNone(full)
        for role, rows in (("tongue", lower), ("palate", upper)):
            k = next(i for i, p in enumerate(handoff.parts) if p.mouth_role == role)
            self.assertIsNotNone(moved[k], role)
            piece = handoff.parts[k].vertices
            np.testing.assert_allclose(piece, (soup[rows] @ GLTF_TO_ANNY.T), atol=1e-6)
            np.testing.assert_allclose(moved[k], full[rows], atol=1e-9)
        # the tongue's piece follows the jaw's turn
        k = next(i for i, p in enumerate(handoff.parts) if p.mouth_role == "tongue")
        self.assertGreater(
            np.linalg.norm(moved[k] - handoff.parts[k].vertices, axis=1).max(), 1e-3
        )


class TestLipMotion(unittest.TestCase):
    """The lips' carried motion is smoothed within each lip, and the lining inside the mouth
    follows the lips of its own jaw."""

    @staticmethod
    def _strip(n=30):
        # two rows of lips (upper above, lower below) joined only at both ends (the corners)
        x = np.linspace(-0.02, 0.02, n)
        V = np.concatenate(
            [
                np.c_[x, 0 * x, 0.002 + 0 * x],
                np.c_[x, 0 * x, 0.004 + 0 * x],
                np.c_[x, 0 * x, -0.002 + 0 * x],
                np.c_[x, 0 * x, -0.004 + 0 * x],
            ]
        )
        T = []
        for a, b in ((0, 1), (2, 3)):
            for i in range(n - 1):
                p, q = a * n + i, b * n + i
                T += [[p, p + 1, q], [p + 1, q + 1, q]]
        for i in (0, n - 2):  # the corners: the lips meet
            T += [[i, i + 1, 2 * n + i], [i + 1, 2 * n + i + 1, 2 * n + i]]
        side = np.r_[np.ones(2 * n, np.int8), np.full(2 * n, 2, np.int8)]
        return V, np.asarray(T), side

    def test_each_lip_keeps_its_motion(self):
        V, T, side = self._strip()
        rng = np.random.default_rng(0)
        offsets = np.where(side[:, None] == 2, [0.0, 0.0, -0.02], [0.0, 0.0, -0.004])
        rough = offsets + rng.normal(0, 0.002, offsets.shape)
        out = P.smooth_lip_motion(rough, T, side)
        n = len(V) // 4
        middle = np.r_[n // 3 : 2 * n // 3]
        for lip, want in ((1, -0.004), (2, -0.02)):
            rows = np.flatnonzero(side == lip)
            rows = rows[np.isin(rows % n, middle)]
            # the noise is smoothed (by at least a third), and each lip's own motion stays
            self.assertLess(
                np.abs(out[rows, 2] - want).mean(),
                0.67 * np.abs(rough[rows, 2] - want).mean(),
            )
            self.assertLess(abs(out[rows, 2].mean() - want), 0.001)

    def test_a_smooth_ramp_keeps_its_motion(self):
        V, T, side = self._strip()
        ramp = np.zeros((len(V), 3))
        ramp[:, 2] = -0.5 * V[:, 0] - np.where(side == 2, 0.01, 0.002)
        n = len(V) // 4
        rows = np.flatnonzero(np.isin(np.arange(len(V)) % n, np.r_[4 : n - 4]))
        # away from the corners, where both lips meet, a ramp stands out of nothing
        np.testing.assert_allclose(
            P.smooth_lip_motion(ramp, T, side)[rows], ramp[rows], atol=1e-9
        )

    def test_lining_follows_its_own_jaw(self):
        from opensculptboy.fit.register import smooth_mouth

        # a lining strip between an upper lip (row 0) and a lower lip (last row)
        rows, n = 9, 5
        z = np.linspace(0.01, -0.01, rows)
        x = np.linspace(-0.01, 0.01, n)
        V = np.array([[xi, 0.0, zi] for zi in z for xi in x])
        T = []
        for r in range(rows - 1):
            for c in range(n - 1):
                p, q = r * n + c, (r + 1) * n + c
                T += [[p, p + 1, q], [p + 1, q + 1, q]]
        T = np.asarray(T)
        known = np.zeros(len(V), bool)
        known[:n] = known[-n:] = True
        out = np.zeros((len(V), 3))
        out[-n:, 2] = -0.02  # the lower lip moves down
        jaw = np.repeat(np.r_[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], n)
        plain = smooth_mouth(out, T, ~known)
        # the old extension: a ramp from the upper lip to the lower
        self.assertGreater(plain[2 * n, 2], -0.015)
        mixed = jaw[:, None] * smooth_mouth(out, T, ~(known & (jaw >= 0.5)))
        mixed += (1 - jaw[:, None]) * smooth_mouth(out, T, ~(known & (jaw < 0.5)))
        # the rows of the jaw roll with the lower lip, the others stay with the upper
        np.testing.assert_allclose(mixed[4 * n : 8 * n, 2], -0.02, atol=1e-9)
        np.testing.assert_allclose(mixed[n : 4 * n, 2], 0.0, atol=1e-9)


class TestSides(unittest.TestCase):
    def test_closest_point_on_triangles(self):
        rng = np.random.default_rng(0)
        a, b, c = rng.normal(size=(3, 40, 3))
        p = rng.normal(size=(40, 3)) * 2
        q, bary = P._closest_on_triangles(p, a, b, c)
        np.testing.assert_allclose(bary.sum(1), 1.0, atol=1e-12)
        self.assertGreaterEqual(bary.min(), 0.0)
        u, v = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201))
        keep = u + v <= 1
        u, v = u[keep], v[keep]
        for i in range(len(p)):
            dense = a[i] + u[:, None] * (b[i] - a[i]) + v[:, None] * (c[i] - a[i])
            best = np.linalg.norm(dense - p[i], axis=1).min()
            self.assertLessEqual(np.linalg.norm(q[i] - p[i]), best + 1e-12)

    def test_each_lip_takes_its_own_motion(self):
        # two lips that touch along y = 0: the upper one still, the lower one moving down;
        # the rows of both lips on the line take their own lip's motion
        from types import SimpleNamespace

        V, T, _ = _grid(n=21, size=0.02)
        upper = V[:, 1] >= 0
        lower_v = V[~upper] - [0.0, 0.0, 0.0]
        verts = np.concatenate([V[upper], lower_v])
        index = np.full(len(V), -1)
        index[upper] = np.arange(upper.sum())
        index[~upper] = upper.sum() + np.arange((~upper).sum())
        # the row y = 0 belongs to both lips: duplicate it for the lower lip
        row = np.nonzero(np.abs(V[:, 1]) < 1e-9)[0]
        verts = np.concatenate([verts, V[row]])
        dup = np.arange(len(verts) - len(row), len(verts))
        tri = []
        for t in T:
            if upper[t].all():
                tri.append(index[t])
            else:
                mapped = [
                    dup[np.nonzero(row == i)[0][0]] if i in row else index[i] for i in t
                ]
                tri.append(mapped)
        tri = np.array(tri)
        side = np.where(np.arange(len(verts)) < upper.sum(), 1, 2).astype(np.int8)
        side[dup] = 2
        field = np.zeros((len(verts), 3))
        field[side == 2] = [0.0, -0.004, 0.0]
        reference = SimpleNamespace(vertices=verts, triangles=tri)
        points = V[row]
        for s, expected in ((1, 0.0), (2, -0.004)):
            d, t, b = P._sided_closest(points, reference, side, s)
            self.assertLess(d.max(), 1e-9)
            got = np.einsum("nk,nkc->nc", b, field[tri[t]])
            np.testing.assert_allclose(got[:, 1], expected, atol=1e-12)


def _bilinear(n_coarse):
    """The operator from a coarse grid of n by n points (_grid) to the grid of 2n - 1 by
    2n - 1 points over the same square, as one SparseOperator."""
    from anny.utils.subdivision import SparseOperator

    n = 2 * n_coarse - 1
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(n):
            ii = [i // 2] if i % 2 == 0 else [i // 2, i // 2 + 1]
            jj = [j // 2] if j % 2 == 0 else [j // 2, j // 2 + 1]
            w = 1.0 / (len(ii) * len(jj))
            for a in ii:
                for b in jj:
                    rows.append(i * n + j)
                    cols.append(a * n_coarse + b)
                    vals.append(w)
    return SparseOperator(rows, cols, vals, (n * n, n_coarse * n_coarse))


class TestFineCarry(unittest.TestCase):
    def test_level0_takes_the_finer_motion(self):
        """Level 0's offsets are solved so that their subdivision follows the motion carried
        to the finer level, each row held to its own offset."""
        from types import SimpleNamespace
        from unittest import mock

        op = _bilinear(11)
        Vf, Tf, _ = _grid(n=21)
        rng = np.random.default_rng(1)
        truth = rng.normal(scale=1e-3, size=(op.shape[1], 3))
        fine_field = op.apply(truth)
        coarse = truth + rng.normal(scale=1e-3, size=truth.shape)
        topology = SimpleNamespace(operators=[op], triangles=Tf)
        n0 = op.shape[1]
        data = P._fine_setup(
            (topology, Vf),
            np.ones(n0, bool),
            (np.zeros(n0, np.int8), np.zeros(1, np.int8)),
            np.zeros(n0),
        )
        with mock.patch.object(P, "carry_field", return_value=(fine_field, None)):
            out = P._through_fine(coarse, data, None, None, None)
        before = np.abs(op.apply(coarse) - fine_field).mean()
        after = np.abs(op.apply(out) - fine_field).mean()
        self.assertLess(after, 0.6 * before)

    def test_rounded_pose_corners_follow(self):
        """A pose that moves only a band of the reference narrower than level 0's rows (the
        corners of a rounded mouth, pulled together in the vowels O and U): level 0 carried
        alone smears the band over its rows, and its offsets solved through the finer level
        follow the band's motion there."""
        from types import SimpleNamespace

        from opensculptboy.fit.handoff import GLTF_TO_ANNY

        size = 0.06
        Vf, Tf, _ = _grid(n=41, size=size)
        Vc, Tc, _ = _grid(n=21, size=size)
        # the jaw moves a band 1.5 mm below the lips' line, toward the corners
        band = (Vf[:, 1] < 0) & (Vf[:, 1] > -0.0016) & (np.abs(Vf[:, 0]) > 0.009)
        weight = np.where(band, 1.0, 0.0)
        soup = Vf[Tf.ravel()]
        J = np.zeros((len(soup), 4), int)
        J[:, 1] = 1
        W = np.zeros((len(soup), 4))
        W[:, 1] = weight[Tf.ravel()]
        W[:, 0] = 1.0 - W[:, 1]
        turn = [np.sin(np.radians(6)), 0.0, 0.0, np.cos(np.radians(6))]
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            _write_rig_glb(folder / "replica.glb", soup, J, W, [0.0, 0.0, -0.05])
            appearance = {
                "parts": [{"name": "head", "role": "skin"}],
                "mouth": {
                    "rig": {"jaw": "lower_jaw"},
                    "poses": {
                        "o": {
                            "known": True,
                            "joints": {"lower_jaw": {"rotation": turn}},
                        },
                    },
                },
            }
            (folder / "appearance.json").write_text(json.dumps(appearance))
            handoff = load_handoff(folder)
            reference = Reference.from_handoff(handoff)
            field = P.reference_pose_fields(handoff, reference)["o"][0]
            transforms = P.reference_pose_transforms(handoff, reference)["o"]
        coarse = Vc @ GLTF_TO_ANNY.T
        fine = Vf @ GLTF_TO_ANNY.T
        from scipy.spatial import cKDTree

        d, i = cKDTree(reference.vertices).query(fine)
        self.assertLess(d.max(), 1e-6)
        target = field[i]  # the reference's motion at the finer level's vertices
        ref_side = np.where(np.asarray(reference.vertices)[:, 2] < 0, 2, 1).astype(
            np.int8
        )  # Anny's z is glTF's y
        side_c = np.where(Vc[:, 1] < 0, 2, 1).astype(np.int8)
        on_skin = np.ones(len(coarse), bool)
        values0, _ = P.carry_field(
            coarse,
            Tc,
            reference,
            field,
            on_skin,
            (side_c, ref_side),
            transforms=transforms,
        )
        op = _bilinear(21)
        topology = SimpleNamespace(operators=[op], triangles=Tf)
        data = P._fine_setup((topology, fine), on_skin, (side_c, ref_side), None)
        out = P._through_fine(values0, data, reference, field, transforms)
        # the motion the finer level carries itself
        fine_values, _ = P.carry_field(
            fine,
            *data[1:2],
            reference,
            field,
            data[3],
            data[4],
            data[5],
            transforms,
            lining=P.LINING_RINGS * 2,
        )
        rows = np.abs(Vf[:, 1]) < 0.0046
        before = np.linalg.norm(op.apply(values0)[rows] - fine_values[rows], axis=1)
        after = np.linalg.norm(op.apply(out)[rows] - fine_values[rows], axis=1)
        self.assertGreater(np.abs(target[band]).max(), 1e-3)
        self.assertLess(after.mean(), 0.8 * before.mean())


class TestLevelCarry(unittest.TestCase):
    """Each detail level's own carry of a pose (:func:`opensculptboy.mouth.poses.level_corrections`):
    the reference's rig turns the level's own vertices, its rest detail included, where level
    0's pose subdivided leaves that detail fixed in the rest frame and folds it into a crease
    along the rolled lip."""

    def test_level_carry_turns_the_detail(self):
        from types import SimpleNamespace

        from opensculptboy.export.mesh_checks import adjacent_triangles, bends
        from opensculptboy.fit.handoff import GLTF_TO_ANNY

        size = 0.06
        Vf, Tf, jaw_f = _grid(n=41, size=size)
        Vc, Tc, _ = _grid(n=21, size=size)
        # the reference: the lower half turns with the jaw, fully below the lips' line
        weight = np.clip(-Vf[:, 1] / 0.004, 0.0, 1.0)
        soup = Vf[Tf.ravel()]
        J = np.zeros((len(soup), 4), int)
        J[:, 1] = 1
        W = np.zeros((len(soup), 4))
        W[:, 1] = weight[Tf.ravel()]
        W[:, 0] = 1.0 - W[:, 1]
        turn = [np.sin(np.radians(15)), 0.0, 0.0, np.cos(np.radians(15))]
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            _write_rig_glb(folder / "replica.glb", soup, J, W, [0.0, -0.01, 0.0])
            appearance = {
                "parts": [{"name": "head", "role": "skin"}],
                "mouth": {
                    "rig": {"jaw": "lower_jaw"},
                    "poses": {
                        "open": {
                            "known": True,
                            "joints": {"lower_jaw": {"rotation": turn}},
                        },
                    },
                },
            }
            (folder / "appearance.json").write_text(json.dumps(appearance))
            handoff = load_handoff(folder)
            reference = Reference.from_handoff(handoff)
            field = P.reference_pose_fields(handoff, reference)["open"][0]
            transforms = P.reference_pose_transforms(handoff, reference)["open"]
        coarse = Vc @ GLTF_TO_ANNY.T
        # the finer level carries rest detail the coarse level lacks: a ridge 0.8 mm high
        # across the lower lip, 11 mm below the lips' line
        ridge = 0.0008 * np.exp(-(((Vf[:, 1] + 0.011) / 0.0012) ** 2))
        fine_rest = Vf.copy()
        fine_rest[:, 2] += ridge
        fine = fine_rest @ GLTF_TO_ANNY.T
        ref_side = np.where(np.asarray(reference.vertices)[:, 2] < 0, 2, 1).astype(
            np.int8
        )
        side_c = np.where(Vc[:, 1] < 0, 2, 1).astype(np.int8)
        on_skin = np.ones(len(coarse), bool)
        values0, _ = P.carry_field(
            coarse,
            Tc,
            reference,
            field,
            on_skin,
            (side_c, ref_side),
            transforms=transforms,
        )
        op = _bilinear(21)
        topology = SimpleNamespace(
            operators=[op], triangles=Tf, subdiv=1, rows=np.arange(len(Vf))
        )
        topology.apply = op.apply
        data = P._fine_setup((topology, fine), on_skin, (side_c, ref_side), None)
        corr = P.level_corrections(
            {1: (data, np.ones(len(fine)))},
            reference,
            field,
            transforms,
            np.eye(3),
            values0,
        )
        poses = P.MouthPoses(
            ("open",),
            values0[None],
            np.zeros((1, 0, 3)),
            {},
            {1: (corr[1][0], corr[1][1][None])},
        )
        subdivided = fine + op.apply(values0)
        own = fine + poses.on_level(0, topology)
        # the ridge's cross-section in the middle of the lip keeps its bends with the level's
        # own carry, and loses them with level 0's pose subdivided
        pairs = adjacent_triangles(Tf)
        centre = fine_rest[Tf].mean(1)
        mid = centre[pairs[:, 0]]
        sel = (np.abs(mid[:, 0]) < 0.012) & (np.abs(mid[:, 1] + 0.011) < 0.003)
        rest_bends = bends(fine, Tf, pairs[sel])
        err_own = np.abs(bends(own, Tf, pairs[sel]) - rest_bends).max()
        err_sub = np.abs(bends(subdivided, Tf, pairs[sel]) - rest_bends).max()
        self.assertLess(err_own, 0.5)
        self.assertGreater(err_sub, 2.0 * err_own + 0.5)

    def test_corner_weight(self):
        corners = np.array([[-0.022, -0.15, 0.64], [0.022, -0.15, 0.64]])
        x = np.linspace(-0.03, 0.03, 61)
        points = np.c_[x, np.full_like(x, -0.15), np.full_like(x, 0.64)]
        w = P.corner_weight(points, corners)
        near = np.min(np.abs(np.abs(x)[:, None] - 0.022), 1) < P.LEVEL_CORNER_NEAR
        far = np.min(np.abs(np.abs(x)[:, None] - 0.022), 1) > P.LEVEL_CORNER_FAR
        self.assertTrue(np.all(w[near] == 0.0))
        self.assertTrue(np.all(w[far] == 1.0))
        self.assertAlmostEqual(float(w[30]), 1.0)  # the middle of the mouth
        middle = (x > -0.022) & (x < 0.0)
        self.assertTrue(np.all(np.diff(w[middle]) >= -1e-12))

    def test_corrections_reach_the_targets(self):
        """The exports add a level's correction to the pose's target and to its jaw
        corrective, and nothing on another level."""
        from types import SimpleNamespace

        rows = np.array([1, 3])
        values = np.array([[[1.0, 0, 0], [0, 2.0, 0]], [[0, 0, 3.0], [4.0, 0, 0]]])
        poses = P.MouthPoses(
            ("a", "b"),
            np.zeros((2, 2, 3)),
            np.zeros((2, 0, 3)),
            {},
            {2: (rows, values)},
        )
        topology = SimpleNamespace(subdiv=2, rows=np.array([3, 0, 1]))
        offsets = np.zeros((4, 3, 3))
        added = {1: 0, 2: 1}
        overrides = {3: ("corrective", 1), 0: np.zeros(3)}
        out = P.add_level_corrections(offsets, topology, poses, added, overrides)
        np.testing.assert_allclose(out[1], [[0, 2.0, 0], [0, 0, 0], [1.0, 0, 0]])
        np.testing.assert_allclose(out[2], [[4.0, 0, 0], [0, 0, 0], [0, 0, 3.0]])
        np.testing.assert_allclose(out[3], out[2])
        np.testing.assert_allclose(out[0], 0.0)
        other = SimpleNamespace(subdiv=1, rows=np.arange(3))
        self.assertIs(
            P.add_level_corrections(offsets, other, poses, added, overrides), offsets
        )


class TestFile(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(1)
        poses = P.MouthPoses(
            ("viseme_a", "open_wide"),
            rng.normal(size=(2, 7, 3)) * 1e-3,
            rng.normal(size=(2, 4, 3)) * 1e-3,
            {"viseme_a": {"max_mm": 1.0}},
            {2: (np.array([0, 5]), rng.normal(size=(2, 2, 3)) * 1e-3)},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "p.safetensors"
            poses.save(path)
            back = P.MouthPoses.load(path)
        self.assertEqual(back.names, poses.names)
        np.testing.assert_allclose(back.skin, poses.skin, atol=1e-9)
        np.testing.assert_allclose(back.inner, poses.inner, atol=1e-9)
        np.testing.assert_array_equal(back.levels[2][0], poses.levels[2][0])
        np.testing.assert_allclose(back.levels[2][1], poses.levels[2][1], atol=1e-9)
        sub = back.subset(["open_wide", "nothing"])
        self.assertEqual(sub.names, ("open_wide",))
        np.testing.assert_allclose(sub.skin[0], poses.skin[1], atol=1e-9)
        np.testing.assert_allclose(
            sub.levels[2][1][0], poses.levels[2][1][1], atol=1e-9
        )


class TestExports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        model, rest, offsets, names = _body()
        cls.model = model
        mouth = synthetic_mouth(model, rest)
        mouth.save(cls.folder / "c.inner_mouth.safetensors")
        # anny's own jawOpen at half weight stands in for a reference's pose
        open_half = 0.5 * offsets[names.index("jawOpen")]
        base = rest
        lower = mouth.arch == 1
        inner = np.zeros((2, len(mouth.points), 3))
        inner[0, lower] = [0.0, 0.0, -0.004]  # the lower pieces drop with the pose
        cls.poses = P.MouthPoses(
            ("viseme_a", "smile"),
            np.stack([open_half, np.zeros_like(open_half)]),
            inner,
            {},
        )
        cls.poses.save(cls.folder / "c.mouth_poses.safetensors")
        cls.card = Character(
            name="mouth",
            rig=MOUTH_RIG,
            material={
                "mouth": {
                    "mesh": "c.inner_mouth.safetensors",
                    "poses": "c.mouth_poses.safetensors",
                }
            },
        )
        cls.path = cls.folder / "mouth.glb"
        export_glb(cls.path, cls.card, cls.model, detail_dir=cls.folder)
        cls.glb = GLB(cls.path)
        cls.base = base

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _targets(self):
        return self.glb.json["scenes"][0]["extras"]["opensculptboy"]["morph_targets"]

    def test_targets_and_animations(self):
        g = self.glb.json
        self.assertEqual(structure_errors(self.glb), [])
        names = self._targets()
        for pose in ("viseme_a", "smile"):
            self.assertIn(f"mouth.{pose}", names)
            self.assertIn(f"mouth.{pose}.corrective", names)
        animations = [a["name"] for a in g["animations"]]
        self.assertIn("mouth.viseme_a.jaw", animations)
        info = g["scenes"][0]["extras"]["opensculptboy"]["mouth"]["poses"]
        self.assertGreater(info["viseme_a"]["jaw_degrees"], 5.0)
        self.assertLess(info["smile"]["jaw_degrees"], 1e-3)

    def test_turn_and_corrective_give_the_pose(self):
        # the animation's last frame (the jaw's turn and the corrective) shows the pose's
        # target, the skin and the inner mouth alike
        node = min(evaluate_nodes(self.glb))
        names = self._targets()
        weights = np.zeros(len(names))
        weights[names.index("mouth.viseme_a")] = 1.0
        target = evaluate_nodes(self.glb, weights={node: weights})[node]
        turned = evaluate_nodes(self.glb, animation="mouth.viseme_a.jaw", frame=1)[node]
        rest = evaluate_nodes(self.glb)[node]
        for k in range(len(target)):
            moved = np.linalg.norm(target[k].positions - rest[k].positions, axis=1)
            self.assertGreater(moved.max(), 0.002)
            np.testing.assert_allclose(
                turned[k].positions, target[k].positions, atol=2e-5
            )

    def test_vrm_visemes(self):
        from opensculptboy.export.vrm import export_vrm
        from test.test_vrm_schema import check_vrm0, check_vrm1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for version in ("1.0", "0.x"):
                with self.subTest(version=version):
                    path = self.folder / f"mouth{version}.vrm"
                    export_vrm(
                        path,
                        self.card,
                        self.model,
                        version=version,
                        author="tests",
                        refine_weights=False,
                        detail_dir=self.folder,
                    )
                    glb = GLB(path)
                    g = glb.json
                    self.assertEqual(structure_errors(glb), [])
                    check = check_vrm1 if version == "1.0" else check_vrm0
                    self.assertEqual(check(g), [])
                    names = g["scenes"][0]["extras"]["opensculptboy"]["morph_targets"]
                    self.assertIn("mouth.viseme_a", names)
                    self.assertNotIn("mouth.smile", names)  # the visemes alone
                    t = names.index("mouth.viseme_a")
                    if version == "1.0":
                        aa = g["extensions"]["VRMC_vrm"]["expressions"]["preset"]["aa"]
                        binds = aa["morphTargetBinds"]
                        self.assertEqual([b["index"] for b in binds], [t])
                        self.assertEqual(binds[0]["weight"], 1.0)
                    else:
                        groups = g["extensions"]["VRM"]["blendShapeMaster"][
                            "blendShapeGroups"
                        ]
                        a = next(x for x in groups if x["presetName"] == "a")
                        self.assertEqual([b["index"] for b in a["binds"]], [t])


if __name__ == "__main__":
    unittest.main()

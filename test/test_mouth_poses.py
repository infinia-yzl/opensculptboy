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


class TestFile(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(1)
        poses = P.MouthPoses(
            ("viseme_a", "open_wide"),
            rng.normal(size=(2, 7, 3)) * 1e-3,
            rng.normal(size=(2, 4, 3)) * 1e-3,
            {"viseme_a": {"max_mm": 1.0}},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "p.safetensors"
            poses.save(path)
            back = P.MouthPoses.load(path)
        self.assertEqual(back.names, poses.names)
        np.testing.assert_allclose(back.skin, poses.skin, atol=1e-9)
        np.testing.assert_allclose(back.inner, poses.inner, atol=1e-9)
        sub = back.subset(["open_wide", "nothing"])
        self.assertEqual(sub.names, ("open_wide",))
        np.testing.assert_allclose(sub.skin[0], poses.skin[1], atol=1e-9)


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

# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the closed lips of a fit and the jaw's weights from a reference (docs/mouth.md):
``opensculptboy.fit.lips`` finds where a reference's lips end (an opening whose borders touch
or stand apart, or a crease) and seals anny's lips onto it, so that the front view shows no
slit and no hole at the corners; ``opensculptboy.fit.poses`` poses a GLB file's skins with a
handoff's joint deltas, and ``opensculptboy.mouth.weights`` measures how far the skin follows
the jaw and gives a model of the ``anny-mouth`` rig the jaw weights that follow it.
"""

import json
import pathlib
import struct
import tempfile
import unittest

import numpy as np
import torch

from opensculptboy import Character
from opensculptboy.fit import lips
from opensculptboy.fit.handoff import GLTF_TO_ANNY, load_handoff
from opensculptboy.fit.metrics import mouth_zone
from opensculptboy.fit.register import jaw_share as anny_jaw_share
from opensculptboy.fit.register import mouth_lining


def _strip(z, y, x0=-0.03, x1=0.03, height=0.014, n=61, up=True):
    """A lip as a strip of quads in the plane y (anny's frame, the front at -y), its edge at
    height z, reaching ``height`` up (or down)."""
    xs = np.linspace(x0, x1, n)
    sign = 1.0 if up else -1.0
    rows = [z + sign * h for h in np.linspace(0.0, height, 7)]
    V = np.array([[x, y, r] for r in rows for x in xs])
    T = []
    for j in range(len(rows) - 1):
        for i in range(n - 1):
            a, b = j * n + i, j * n + i + 1
            c, d = a + n, b + n
            T += [[a, b, d], [a, d, c]]
    return V, np.array(T)


def _box(z=0.0, width=0.016, half=0.004):
    return lips.MouthBox(
        np.array([width, 0.0, z]),
        np.array([-width, 0.0, z]),
        np.array([0.0, 0.0, z + half]),
        np.array([0.0, 0.0, z - half]),
    )


def _join(*meshes):
    V, T, k = [], [], 0
    for v, t in meshes:
        V.append(v)
        T.append(t + k)
        k += len(v)
    return np.concatenate(V), np.concatenate(T)


class TestLipLine(unittest.TestCase):
    def test_touching_opening(self):
        # a game's head mesh: the lips are separate rows whose borders touch
        V, T = _join(_strip(0.0, 0.0, up=True), _strip(0.0, 0.0, up=False))
        V[: len(V) // 2, 1] += 2e-6  # rows that touch, not one welded row
        line = lips.lip_line(V, T, _box())
        self.assertIsNotNone(line)
        self.assertLess(line.gap().max(), 1e-4)
        np.testing.assert_allclose(line.upper[:, 2], 0.0, atol=1e-4)
        self.assertLess(line.upper[0, 0], -0.015)
        self.assertGreater(line.upper[-1, 0], 0.015)

    def test_open_mouth_keeps_both_edges(self):
        # a mouth that stands open by 2 mm: each lip keeps its own edge
        V, T = _join(_strip(0.001, 0.0, up=True), _strip(-0.001, 0.0, up=False))
        line = lips.lip_line(V, T, _box())
        self.assertIsNotNone(line)
        np.testing.assert_allclose(line.upper[:, 2], 0.001, atol=1e-4)
        np.testing.assert_allclose(line.lower[:, 2], -0.001, atol=1e-4)
        np.testing.assert_allclose(line.gap(), 0.002, atol=1e-4)

    def test_crease(self):
        # welded lips that meet in a crease 2 mm deep: the line runs along its bottom
        xs = np.linspace(-0.03, 0.03, 61)
        zs = np.linspace(-0.014, 0.014, 57)
        X, Z = np.meshgrid(xs, zs)
        Y = 0.002 * np.exp(-((Z / 0.0008) ** 2))  # the bottom lies 2 mm back (+y)
        V = np.stack([X, Y, Z], -1).reshape(-1, 3)
        n = len(xs)
        T = []
        for j in range(len(zs) - 1):
            for i in range(n - 1):
                a, b = j * n + i, j * n + i + 1
                T += [[a, b, b + n], [a, b + n, a + n]]
        line = lips.lip_line(V, np.array(T), _box())
        self.assertIsNotNone(line)
        self.assertLess(line.gap().max(), 1e-9)
        np.testing.assert_allclose(line.upper[:, 2], 0.0, atol=2e-4)
        np.testing.assert_allclose(line.upper[:, 1], 0.002, atol=2e-4)

    def test_no_lips(self):
        # a flat surface: no opening and no crease
        xs = np.linspace(-0.03, 0.03, 31)
        zs = np.linspace(-0.014, 0.014, 15)
        X, Z = np.meshgrid(xs, zs)
        V = np.stack([X, np.zeros_like(X), Z], -1).reshape(-1, 3)
        n = len(xs)
        T = []
        for j in range(len(zs) - 1):
            for i in range(n - 1):
                a, b = j * n + i, j * n + i + 1
                T += [[a, b, b + n], [a, b + n, a + n]]
        self.assertIsNone(lips.lip_line(V, np.array(T), _box()))


class TestSeal(unittest.TestCase):
    """anny's own lips end apart (its mouth is modelled a little open): sealed onto a line
    between them, they meet along it, and the skin away from them stays."""

    @classmethod
    def setUpClass(cls):
        card = Character()
        cls.model = card.build_model(dtype=torch.float64)
        with torch.no_grad():
            cls.rest = cls.model(**card.model_kwargs())["rest_vertices"][0].numpy()
        cls.T = cls.model.faces.cpu().numpy()

    def test_seal_closes_the_lips(self):
        from anny.keypoints import KeypointsRegressor

        model, rest, T = self.model, self.rest, self.T
        regressor = KeypointsRegressor.craniofacial(model)
        points = regressor({"vertices": torch.as_tensor(rest)[None]})[0].numpy()
        box = lips.MouthBox.from_landmarks(dict(zip(regressor.labels, points)))
        zone = mouth_zone(model, rest)
        cavity = mouth_lining(model, 0)
        jaw = anny_jaw_share(model)
        roll = zone & ~cavity
        e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]]])
        e = e[roll[e].all(1)]
        spacing = float(
            np.median(np.linalg.norm(rest[e[:, 0]] - rest[e[:, 1]], axis=1))
        )
        upper, lower = lips.lip_edges(
            rest, *lips.lip_masks(rest, roll, jaw, box, spacing), box, spacing
        )
        corners = lips.lip_corners(rest, roll, box)
        self.assertGreater(len(upper), 10)
        self.assertGreater(len(lower), 5)
        self.assertEqual(len(corners), 2)
        # the upper lip's edge lies above the lower lip's
        self.assertGreater(rest[upper, 2].mean(), rest[lower, 2].mean())

        # a line midway between anny's own edges
        xs = np.linspace(rest[corners, 0].min(), rest[corners, 0].max(), 60)

        def edge(rows):
            order = np.argsort(rest[rows, 0])
            return np.stack(
                [xs]
                + [
                    np.interp(xs, rest[rows, 0][order], rest[rows, k][order])
                    for k in (1, 2)
                ],
                1,
            )

        mid = 0.5 * (edge(upper) + edge(lower))
        line = lips.LipLine(mid, mid.copy())
        before = lips.lip_gap(rest, upper, lower)
        ident = lambda p, r: p  # noqa: E731
        offsets, seal = lips.seal_lips(
            np.zeros_like(rest),
            rest,
            T,
            ident,
            ident,
            line,
            upper,
            lower,
            zone,
            3,
            corners=corners,
        )
        after = rest + offsets
        gap = lips.lip_gap(after, seal.upper, seal.lower)
        print(f"\nseal of anny's lips: gap {before} -> {gap} mm")
        self.assertGreater(before["mean"], 1.0)
        # both lips' rows lie on the line; between its rows, the measure draws the lower
        # lip's edge straight, which the curved line leaves a little apart
        for rows in (seal.upper, seal.lower):
            on, _ = lips.on_line(after[rows], line.upper)
            np.testing.assert_allclose(after[rows], on, atol=1e-9)
        self.assertLess(gap["mean"], 0.15)
        # the corners end where the line ends
        self.assertEqual(len(seal.corners), 2)
        np.testing.assert_allclose(
            np.sort(after[seal.corners, 0]), np.sort(line.ends[:, 0]), atol=1e-9
        )
        # the skin farther from the rows than the seal's rings stays
        far = ~lips.rings_from(
            T, np.concatenate([seal.upper, seal.lower, seal.corners]), 3, len(rest)
        )
        np.testing.assert_array_equal(offsets[far], 0.0)


def _write_rig_glb(path, vertices, joints, weights, jaw_pivot):
    """A GLB file of one skinned triangle soup: a root joint ``Head_0`` and its child
    ``C_Jaw`` at ``jaw_pivot`` (glTF's frame), bound at the identity."""
    vertices = np.asarray(vertices, np.float32)
    J = np.asarray(joints, np.uint16)
    W = np.asarray(weights, np.float32)
    ibm = np.stack([np.eye(4), np.eye(4)]).astype(np.float32)
    ibm[1, :3, 3] = -np.asarray(jaw_pivot)
    ibm = ibm.transpose(0, 2, 1).reshape(2, 16)
    indices = np.arange(len(vertices) // 3 * 3, dtype=np.uint32)
    blobs = [
        vertices.tobytes(),
        J.tobytes(),
        W.tobytes(),
        ibm.tobytes(),
        indices.tobytes(),
    ]
    views, offset, data = [], 0, b""
    for blob in blobs:
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob)})
        data += blob + b"\0" * (-len(blob) % 4)
        offset = len(data)
    accessors = [
        {
            "bufferView": 0,
            "componentType": 5126,
            "count": len(vertices),
            "type": "VEC3",
            "min": vertices.min(0).tolist(),
            "max": vertices.max(0).tolist(),
        },
        {"bufferView": 1, "componentType": 5123, "count": len(J), "type": "VEC4"},
        {"bufferView": 2, "componentType": 5126, "count": len(W), "type": "VEC4"},
        {"bufferView": 3, "componentType": 5126, "count": 2, "type": "MAT4"},
        {
            "bufferView": 4,
            "componentType": 5125,
            "count": len(indices),
            "type": "SCALAR",
        },
    ]
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 2]}],
        "nodes": [
            {"name": "Head_0", "children": [1]},
            {"name": "C_Jaw", "translation": list(map(float, jaw_pivot))},
            {"name": "head", "mesh": 0, "skin": 0},
        ],
        "skins": [{"joints": [0, 1], "inverseBindMatrices": 3}],
        "meshes": [
            {
                "name": "head",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
                        "indices": 4,
                    }
                ],
            }
        ],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(data)}],
    }
    text = json.dumps(document).encode()
    text += b" " * (-len(text) % 4)
    body = struct.pack("<II", len(text), 0x4E4F534A) + text
    body += struct.pack("<II", len(data), 0x004E4942) + data
    pathlib.Path(path).write_bytes(
        struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body
    )


class TestJawShare(unittest.TestCase):
    def test_share_from_poses(self):
        from opensculptboy.mouth.weights import reference_jaw_share

        rng = np.random.default_rng(0)
        # glTF's frame: y up, z to the front; the jaw's pivot behind and above the chin
        V = rng.uniform([-0.03, -0.04, 0.0], [0.03, 0.0, 0.05], (300, 3))
        on_jaw = np.arange(300) < 100
        half = (np.arange(300) >= 100) & (np.arange(300) < 200)
        J = np.zeros((300, 4), int)
        J[:, 1] = 1
        W = np.zeros((300, 4))
        W[:, 0] = np.where(on_jaw, 0.0, np.where(half, 0.5, 1.0))
        W[:, 1] = 1.0 - W[:, 0]
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            _write_rig_glb(folder / "replica.glb", V, J, W, [0.0, 0.0, -0.05])
            turn = [np.sin(np.radians(4)), 0.0, 0.0, np.cos(np.radians(4))]
            appearance = {
                "parts": [{"name": "head", "role": "skin"}],
                "mouth": {
                    "rig": {"jaw": "C_Jaw"},
                    "poses": {
                        "open": {
                            "known": True,
                            "joints": {"C_Jaw": {"rotation": turn}},
                        },
                        "small": {
                            "known": True,
                            "joints": {"C_Jaw": {"rotation": [0, 0, 0, 1]}},
                        },
                        "guess": {
                            "known": False,
                            "joints": {"C_Jaw": {"rotation": turn}},
                        },
                    },
                },
            }
            (folder / "appearance.json").write_text(json.dumps(appearance))
            handoff = load_handoff(folder)
            np.testing.assert_allclose(
                handoff.parts[0].vertices, V @ GLTF_TO_ANNY.T, atol=1e-6
            )
            shares = reference_jaw_share(handoff)
        self.assertEqual(
            shares["poses"], ("open",)
        )  # the still pose and the guess count for nothing
        share = shares[0]
        moving = np.linalg.norm(V - [0.0, 0.0, -0.05], axis=1) > 0.01
        np.testing.assert_allclose(share[on_jaw & moving], 1.0, atol=1e-5)
        np.testing.assert_allclose(share[half & moving], 0.5, atol=1e-5)
        np.testing.assert_allclose(share[~on_jaw & ~half], 0.0, atol=1e-5)

    def test_weights_follow_the_share(self):
        from opensculptboy.mouth.weights import with_jaw_share

        card = Character(rig="anny-mouth")
        model = card.build_model(dtype=torch.float64)
        labels = list(model.bone_labels)
        head, jaw = labels.index("head"), labels.index("jaw")

        def dense(m):
            w = m.vertex_bone_weights.double().numpy()
            i = m.vertex_bone_indices.numpy()
            out = np.zeros((len(w), len(labels)))
            np.add.at(
                out, (np.repeat(np.arange(len(w)), w.shape[1]), i.ravel()), w.ravel()
            )
            return out

        before = dense(model)
        group = before[:, head] + before[:, jaw]
        share = np.where(group > 0, 0.25, 0.0)
        after = dense(with_jaw_share(model, share))
        rows = group > 0
        np.testing.assert_allclose(after[rows, jaw], 0.25 * group[rows], atol=1e-9)
        np.testing.assert_allclose(after[rows, head], 0.75 * group[rows], atol=1e-9)
        others = [k for k in range(len(labels)) if k not in (head, jaw)]
        np.testing.assert_allclose(after[:, others], before[:, others], atol=1e-9)
        # the model itself keeps its weights
        np.testing.assert_array_equal(dense(model), before)


if __name__ == "__main__":
    unittest.main()

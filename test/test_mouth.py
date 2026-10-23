# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of ``opensculptboy.export.mouth``: anny's default character keeps its mouth cavity behind
the skin under every mouth expression, and a mesh with nothing to correct comes back unchanged.
A character whose face shapes pull the nose and the mouth back (``nose-trans-backward-forward``
and ``mouth-trans-backward-forward`` at -1) shows the cavity in front of the skin above the upper
lip, and its GLB file holds the cavity behind the skin in the rest mesh and in the open mouth,
with the tongue and the lips' edge where they were.
"""

import pathlib
import tempfile
import unittest

import numpy as np
import torch

from opensculptboy import Character, export_glb
from opensculptboy.export import mouth as M
from opensculptboy.export.body import morph_target_rows, rest_rows
from test.gltf_reader import GLB, evaluate

MOUTH_ACTIONS = (
    "jawOpen",
    "jawForward",
    "jawLeft",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthLeft",
    "mouthRollUpper",
    "mouthRollLower",
    "mouthShrugUpper",
    "mouthShrugLower",
    "mouthSmileLeft",
    "mouthFrownLeft",
    "mouthPressLeft",
    "mouthStretchLeft",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "mouthLowerDownLeft",
)
# the open mouth of the review renders: jaw open with the lips apart
OPEN = {
    "jawOpen": 0.55,
    "mouthLowerDownLeft": 0.35,
    "mouthLowerDownRight": 0.35,
    "mouthUpperUpLeft": 0.2,
    "mouthUpperUpRight": 0.2,
}
BACKWARD = {"nose-trans-backward-forward": -1.0, "mouth-trans-backward-forward": -1.0}


def _rows(character):
    model = character.build_model(face_shapes=bool(character.face_shapes))
    model = model.to(dtype=torch.float64)
    labels = list(model.facial_action_labels)
    _, facial_rows, face_rows = morph_target_rows(model, character, labels)
    rest, _ = rest_rows(model, character, facial_rows, face_rows)
    return model, labels, rest[0], rest[1:] - rest[0]


def _mix(labels, offsets, mix):
    weights = np.zeros(len(labels))
    for name, value in mix.items():
        weights[labels.index(name)] = value
    return np.tensordot(weights, offsets, 1)


class TestDefaultMouth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.labels, cls.base, cls.offsets = _rows(Character())
        cls.interior = M.mouth_interior(cls.model)

    def test_cavity(self):
        self.assertIsNotNone(self.interior)
        self.assertGreater(self.interior.cavity.sum(), 300)
        self.assertGreater(self.interior.pinned.sum(), 10)
        self.assertFalse((self.interior.pinned & ~self.interior.cavity).any())

    def test_cavity_behind_the_skin(self):
        """No vertex of the cavity lies in front of the skin at rest or under a mouth
        expression at full weight, nor in the open mouth."""
        states = {
            "rest": self.base,
            "open": self.base + _mix(self.labels, self.offsets, OPEN),
        }
        for name in MOUTH_ACTIONS:
            states[name] = self.base + self.offsets[self.labels.index(name)]
        for name, vertices in states.items():
            with self.subTest(name):
                self.assertEqual(M.cavity_in_front(vertices, self.interior), 0)

    def test_nothing_to_correct_comes_back_unchanged(self):
        keep = [
            self.labels.index(n) for n in ("jawOpen", "mouthSmileLeft", "eyeBlinkLeft")
        ]
        vertices, offsets, report = M.clear_mouth(
            self.base, self.offsets[keep], self.interior
        )
        self.assertIs(vertices, self.base)
        self.assertEqual(report, {"rest": 0, "targets": []})
        np.testing.assert_array_equal(offsets, self.offsets[keep])

    def test_meshes_without_a_mouth(self):
        import anny

        hand = anny.Anny(rig="anny-hand.R", topology="hand.R")
        self.assertIsNone(M.mouth_interior(hand))


class TestBackwardMouth(unittest.TestCase):
    """The skin above the upper lip pulled back past the cavity: the nose and the mouth back."""

    @classmethod
    def setUpClass(cls):
        cls.character = Character(face_shapes=dict(BACKWARD))
        cls.model, cls.labels, cls.base, cls.offsets = _rows(cls.character)
        cls.interior = M.mouth_interior(cls.model)
        names = ["jawOpen", "mouthUpperUpLeft", "mouthUpperUpRight"]
        names += ["mouthLowerDownLeft", "mouthLowerDownRight", "mouthRollUpper"]
        cls.names = names
        cls.picked = [cls.labels.index(n) for n in names]
        cls.fixed, cls.fixed_offsets, cls.report = M.clear_mouth(
            cls.base, cls.offsets[cls.picked], cls.interior
        )

    def _state(self, vertices, offsets, mix):
        weights = np.zeros(len(self.names))
        for name, value in mix.items():
            if name in self.names:
                weights[self.names.index(name)] = value
        return vertices + np.tensordot(weights, offsets, 1)

    def test_the_cavity_shows_without_the_correction(self):
        self.assertGreater(M.cavity_in_front(self.base, self.interior), 0)
        opened = self._state(self.base, self.offsets[self.picked], OPEN)
        self.assertGreater(M.cavity_in_front(opened, self.interior), 0)
        self.assertGreater(self.report["rest"], 0)

    def test_the_correction_keeps_the_cavity_behind(self):
        self.assertTrue(self.report["targets"])
        mixes = {"rest": {}, "open": OPEN}
        mixes.update({name: {name: 1.0} for name in self.names})
        for name, mix in mixes.items():
            with self.subTest(name):
                state = self._state(self.fixed, self.fixed_offsets, mix)
                self.assertEqual(M.cavity_in_front(state, self.interior), 0)

    def test_only_the_cavity_moves(self):
        moved = np.linalg.norm(self.fixed - self.base, axis=1) > 0
        moved |= (
            np.linalg.norm(self.fixed_offsets - self.offsets[self.picked], axis=2) > 0
        ).any(0)
        self.assertTrue(moved.any())
        self.assertFalse((moved & ~self.interior.cavity).any())
        self.assertFalse((moved & self.interior.pinned).any())
        # a smooth push of at most a few millimetres
        self.assertLess(np.linalg.norm(self.fixed - self.base, axis=1).max(), 0.006)

    def test_glb_file(self):
        """The GLB file of the character holds the cavity behind the skin at rest and in the
        open mouth (its morph targets at the open mouth's weights)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "mouth.glb"
            export_glb(path, self.character, model=self.model)
            glb = GLB(path)
            target_names = glb.json["meshes"][0].get("extras", {}).get("targetNames")
            self.assertEqual(target_names, self.labels)
            weights = np.zeros(len(self.labels))
            for name, value in OPEN.items():
                weights[self.labels.index(name)] = value
            to_anny = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
            shift = None
            for mix_weights in (np.zeros(len(self.labels)), weights):
                positions, anny_vertex = evaluate(glb, weights=mix_weights)
                vertices = np.zeros_like(self.base)
                vertices[anny_vertex] = positions @ to_anny.T
                if shift is None:  # the file stands on the floor: undo that translation
                    shift = (self.fixed - vertices).mean(0)
                vertices += shift
                self.assertEqual(M.cavity_in_front(vertices, self.interior), 0)


if __name__ == "__main__":
    unittest.main()

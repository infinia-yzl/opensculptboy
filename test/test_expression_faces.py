# OpenSculptBoy
# Apache License, Version 2.0
"""
The faces of the expression review (:mod:`opensculptboy.expressions.authoring.faces`) and how
the Blender renderer plays a mix (:func:`opensculptboy.expressions.authoring.blender_faces.face_weights`).
"""

import json
import tempfile
import unittest

from anny.models.facial_actions import FACIAL_ACTION_LABELS
from opensculptboy.expressions.authoring import blender_faces, faces


class TestReviewFaces(unittest.TestCase):
    def test_counts_and_names(self):
        out = faces.review_faces(["laugh"])
        names = [f["name"] for f in out]
        self.assertEqual(len(names), len(set(names)))
        count = lambda p: sum(n.startswith(p) for n in names)  # noqa: E731
        self.assertEqual(count("a1/"), 52)
        self.assertEqual(count("a05/"), 52)
        self.assertEqual(count("p/"), 14)
        self.assertEqual(count("v/"), 10)
        self.assertEqual(count("x/"), 20)
        self.assertEqual(count("m/"), 1)
        self.assertEqual(out[-1]["mix"], {"mouth.laugh": 1.0})

    def test_mixes_use_known_actions_in_range(self):
        for face in faces.review_faces():
            for name, weight in face["mix"].items():
                self.assertIn(name, FACIAL_ACTION_LABELS, face["name"])
                self.assertGreater(weight, 0.0, face["name"])
                self.assertLessEqual(weight, 1.0, face["name"])

    def test_asymmetric_expressions(self):
        # a wink, a smirk and a raised brow move one side more than the other
        for name in ("wink", "smirk", "raised_brow", "contemptuous"):
            m = faces.EXPRESSIONS[name]
            left = sum(w for k, w in m.items() if k.endswith("Left"))
            right = sum(w for k, w in m.items() if k.endswith("Right"))
            self.assertNotAlmostEqual(left, right, msg=name)

    def test_command(self):
        with tempfile.TemporaryDirectory() as d:
            faces.main(["--out", f"{d}/faces.json", "--mouth-poses", "a,b"])
            data = json.load(open(f"{d}/faces.json", encoding="utf-8"))
            self.assertEqual(data["faces"][-1]["name"], "m/b")


class TestFaceWeights(unittest.TestCase):
    def test_jaw_turn_and_corrective(self):
        values, turns = blender_faces.face_weights(
            {"jawOpen": 0.4, "mouthFunnel": 1.0},
            rig_keys={"jawOpen"},
            key_names={"jawOpen", "jawOpen.corrective", "mouthFunnel"},
        )
        self.assertEqual(values, {"jawOpen.corrective": 0.4, "mouthFunnel": 1.0})
        self.assertEqual(turns, {"jawOpen": 0.4})

    def test_blink_inbetween(self):
        keys = {"eyeBlinkLeft", "eyeBlinkLeft.inbetween"}
        for w, expected in ((0.0, 0.0), (0.25, 0.5), (0.5, 1.0), (1.0, 0.0)):
            values, _ = blender_faces.face_weights({"eyeBlinkLeft": w}, set(), keys)
            self.assertAlmostEqual(values["eyeBlinkLeft.inbetween"], expected)
        # a file without the in-between plays the blink alone
        values, _ = blender_faces.face_weights(
            {"eyeBlinkLeft": 0.5}, set(), {"eyeBlinkLeft"}
        )
        self.assertEqual(values, {"eyeBlinkLeft": 0.5})

    def test_plain_morph_without_the_rig(self):
        values, turns = blender_faces.face_weights({"jawOpen": 1.0}, set(), {"jawOpen"})
        self.assertEqual(values, {"jawOpen": 1.0})
        self.assertEqual(turns, {})


if __name__ == "__main__":
    unittest.main()

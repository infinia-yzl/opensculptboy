# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of ``opensculptboy.statues`` on a synthetic statue scan (``test.statue_fixtures``):
anny in contrapposto on a plinth, with a tree stump fused to the right leg and a lost left
hand, in millimetres with +Y up and +X to the front. The conversion must find the up and front
axes, cut the plinth off, scale the figure, take the masked stump out and write a handoff
that the fit reads; the region measures must label every region and read zero on the figure
itself.
"""

import json
import pathlib
import tempfile
import unittest

import numpy as np

from opensculptboy.fit.handoff import GLTF_TO_ANNY, load_handoff
from opensculptboy.fit.reference import Reference
from opensculptboy.statues import Mask, scan_to_handoff
from opensculptboy.statues.compare import (
    STATUE_REGIONS,
    region_distances,
    statue_regions,
)
from opensculptboy.statues.scan import (
    axis_vector,
    frame_rotation,
    guess_front,
    guess_up,
    plinth_top,
)
from test.statue_fixtures import (
    PLINTH,
    _model,
    landmarks_in_handoff,
    stump_mask,
    synthetic_statue,
)


class TestScanFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.statue = synthetic_statue(0)

    def test_up_and_front_are_found(self):
        up, confidence = guess_up(self.statue["vertices"])
        self.assertGreater(up @ self.statue["up"], 0.99)
        self.assertGreater(confidence, 0.3)
        front, score = guess_front(self.statue["vertices"], up)
        angle = np.degrees(np.arccos(np.clip(front @ self.statue["front"], -1, 1)))
        self.assertLess(angle, 15.0)
        self.assertGreater(score, 0.05)

    def test_frame_rotation(self):
        R = frame_rotation("+y", "+x")
        np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(R @ [1, 0, 0], [0, -1, 0], atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(R), 1.0)
        with self.assertRaises(ValueError):
            frame_rotation("+z", "-z")
        with self.assertRaises(ValueError):
            axis_vector("up")

    def test_plinth_top(self):
        R = frame_rotation(self.statue["up"], self.statue["front"])
        X = self.statue["vertices"] @ R.T / self.statue["units"]
        top = plinth_top(X, self.statue["triangles"])
        self.assertIsNotNone(top)
        bottom = X[:, 2].min()
        self.assertAlmostEqual(top - bottom, PLINTH[2], delta=0.003)
        # a figure without a plinth has none
        T = self.statue["triangles"]
        figure = T[(X[T, 2] > top + 0.001).all(1)]
        self.assertIsNone(plinth_top(X, figure))

    def test_masks(self):
        box = Mask.from_dict({"kind": "box", "centre": [0, 0, 0], "size": [2, 2, 2]})
        self.assertTrue(box.inside(np.array([[0.9, -0.9, 0.0]]))[0])
        self.assertFalse(box.inside(np.array([[1.1, 0.0, 0.0]]))[0])
        half = Mask.from_dict({"kind": "half", "point": [0, 0, 1], "normal": "+z"})
        self.assertEqual(
            half.inside(np.array([[0, 0, 0.5], [0, 0, 1.5]])).tolist(), [False, True]
        )
        with self.assertRaises(ValueError):
            Mask.from_dict({"kind": "box", "role": "pedestal"})


class TestScanToHandoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.statue = synthetic_statue(0)
        scan = (cls.statue["vertices"], cls.statue["triangles"])
        # a first pass places the scan; the masks and landmarks are drawn on it
        cls.preview = scan_to_handoff(scan, cls.folder / "preview")
        cls.marks = landmarks_in_handoff(
            cls.statue, cls.preview, ["nose_tip", "chin", "knee_l", "knee_r"]
        )
        cls.result = scan_to_handoff(
            scan,
            cls.folder / "handoff",
            stature=1.8,
            masks=[stump_mask(cls.statue, cls.preview)],
            landmarks=cls.marks,
            licence="CC0-1.0",
            title="synthetic statue",
        )
        cls.handoff = load_handoff(cls.folder / "handoff")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_frame_and_scale(self):
        self.assertGreater(self.result.up @ self.statue["up"], 0.99)
        self.assertTrue(self.result.plinth)
        skin = self.handoff.skin_parts()
        self.assertEqual([p.name for p in skin], ["statue"])
        V = skin[0].vertices
        self.assertAlmostEqual(V[:, 2].max() - V[:, 2].min(), 1.8, delta=0.01)
        self.assertGreater(V[:, 2].min(), -0.002)
        # the figure faces -Y in anny's frame: the nose lies in front of the head's centre
        nose = np.asarray(self.marks["nose_tip"]) @ GLTF_TO_ANNY.T
        head = V[V[:, 2] > 1.6].mean(0)
        self.assertLess(nose[1], head[1] - 0.05)

    def test_masked_pieces_are_removed(self):
        counts = self.result.triangles
        self.assertGreater(counts.get("plinth", 0), 0)
        self.assertGreater(counts.get("support", 0), 0)
        V = self.handoff.skin_parts()[0].vertices
        # nothing of the stump is left: no skin within its radius below its top
        statue = self.statue
        from test.statue_fixtures import STUMP, to_scan

        (cx, cy), radius, height = STUMP
        centre = self.result.to_anny(to_scan(statue, [[cx, cy, 0.0]]))[0]
        r = radius * self.result.scale * statue["units"]
        h = height * self.result.scale * statue["units"]
        near = (np.hypot(V[:, 0] - centre[0], V[:, 1] - centre[1]) < 0.7 * r) & (
            V[:, 2] < 0.9 * h
        )
        self.assertFalse(near.any())
        self.assertTrue((self.folder / "handoff" / "removed.glb").exists())

    def test_appearance_records_the_statue(self):
        appearance = json.loads(
            (self.folder / "handoff" / "appearance.json").read_text(encoding="utf-8")
        )
        section = appearance["statue"]
        self.assertEqual(section["format"], "opensculptboy/statue-handoff@1")
        self.assertEqual(section["licence"], "CC0-1.0")
        # the transform carries the scan onto replica.glb's frame
        t = section["transform"]
        R, s, b = np.asarray(t["rotation"]), t["scale"], np.asarray(t["translation"])
        top = self.statue["vertices"][
            np.argmax(self.statue["vertices"] @ self.statue["up"])
        ]
        crown = appearance["landmarks"]["crown"]
        np.testing.assert_allclose(s * R @ top + b, crown, atol=0.02)
        self.assertIn("nose_tip", appearance["landmarks"])

    def test_round_trip(self):
        points = self.statue["vertices"][:50]
        np.testing.assert_allclose(
            self.result.to_scan(self.result.to_anny(points)), points, atol=1e-8
        )

    def test_reference_reads_the_landmarks(self):
        reference = Reference.from_handoff(self.handoff)
        self.assertFalse(reference.auto)
        self.assertIn("crown", reference.landmarks)
        self.assertIn("knee_l", reference.landmarks)

    def test_drapery_and_hair_masks_become_parts(self):
        scan = (self.statue["vertices"], self.statue["triangles"])
        top = self.preview.to_anny(self.statue["vertices"]).max(0)[2]
        hair = {
            "kind": "half",
            "role": "hair",
            "point": [0.0, float(top) - 0.05, 0.0],
            "normal": "+y",
        }
        drape = {
            "kind": "sphere",
            "role": "drapery",
            "centre": [0.0, 1.05, 0.15],
            "radius": 0.1,
        }
        result = scan_to_handoff(scan, self.folder / "parts", masks=[hair, drape])
        handoff = load_handoff(self.folder / "parts")
        roles = {p.role for p in handoff.parts}
        self.assertIn("clothing", roles)
        self.assertIn("hair", roles)
        self.assertGreater(result.triangles["hair"], 0)


class TestLostParts(unittest.TestCase):
    def test_lost_mask(self):
        from opensculptboy.fit.metrics import REGIONS, anny_regions
        from opensculptboy.statues import lost_mask

        model = _model()
        rest = model.template_vertices.detach().double().numpy()
        hand = lost_mask(model, ["hand.l"])
        self.assertGreater(hand.sum(), 100)
        # the figure's left is +X; the hand is the base mesh's left hand
        self.assertTrue((rest[hand, 0] > 0).all())
        regions = anny_regions(model)
        self.assertGreater((regions[hand] == REGIONS.index("hands")).mean(), 0.9)
        forearm = lost_mask(model, ["forearm.l"])
        self.assertTrue((forearm | ~hand).all())  # the forearm's chain holds the hand
        nose = lost_mask(model, ["nose"])
        self.assertTrue(0 < nose.sum() < 400)
        with self.assertRaises(ValueError):
            lost_mask(model, ["tail.l"])

    def test_fit_statue_leaves_lost_parts_without_data(self):
        from unittest import mock

        from opensculptboy.statues import fit_statue

        statue = synthetic_statue(0)
        with tempfile.TemporaryDirectory() as tmp:
            scan_to_handoff(
                (statue["vertices"], statue["triangles"]),
                pathlib.Path(tmp) / "h",
                lost=["hand.l"],
            )
            appearance = json.loads(
                (pathlib.Path(tmp) / "h" / "appearance.json").read_text("utf-8")
            )
            self.assertEqual(appearance["statue"]["lost"], ["hand.l"])
            with mock.patch("opensculptboy.fit.register.fit_reference") as fit:
                fit_statue(pathlib.Path(tmp) / "h", detail=0)
            options = fit.call_args.kwargs["options"]
            self.assertGreater(np.asarray(options.no_data).sum(), 100)
        with self.assertRaises(ValueError):
            scan_to_handoff(
                (statue["vertices"], statue["triangles"]), tmp, lost=["wing.l"]
            )


class TestRegionDistances(unittest.TestCase):
    def test_regions_and_zero_distance(self):
        model = _model()
        regions = statue_regions(model)
        for k, name in enumerate(STATUE_REGIONS):
            self.assertTrue((regions == k).any(), name)
        statue = synthetic_statue(0)
        V, T = statue["body"]
        result = region_distances(V, T, regions, V, T, samples=5000)
        self.assertLess(result["all"]["forward"]["max"], 1e-6)
        self.assertLess(result["all"]["reverse"]["p95"], 0.5)
        # the scan's skin lacks the left hand: its vertices lie far from it
        kept = T[statue["kept"]]
        cut = region_distances(V, T, regions, V, kept, samples=5000)
        self.assertGreater(cut["regions"]["hands"]["far"], 0.3)
        self.assertLess(cut["regions"]["torso"]["far"], 0.01)


if __name__ == "__main__":
    unittest.main()

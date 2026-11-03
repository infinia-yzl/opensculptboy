# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of a studio card applied to a statue config (``opensculptboy.statues.card_config``) and
of contact pins in a config's reference points (``opensculptboy.statues.points``), with a
synthetic card of the default character: the points in the masks' frame, the hands' keys with
the forearm's twist shared as the posing solve shares it, the hands' shape and timing, the
config that the pipeline reads, and a solve held by a pin.
"""

import json
import pathlib
import tempfile
import unittest

import numpy as np

from test.statue_fixtures import _model

PLACEMENT_ANGLE = 25.0
TRANSLATION = [0.05, -0.12, 0.31]
BASE = {"format": "opensculptboy.statue/1", "name": "kouros", "scan": "k.stl"}


def _placement():
    a = np.radians(PLACEMENT_ANGLE)
    R = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0, 0, 1]])
    return R, np.asarray(TRANSLATION)


def _rotations(character):
    from opensculptboy.posing.pose_rig import PoseRig

    return PoseRig.from_card(_model(), character)


def _turn(rig, label, bend=0.0, side=0.0, twist=0.0):
    from opensculptboy.posing.point_solve import _exp, joint_frames

    f = joint_frames(rig)[rig.index[label]]
    return _exp(f["axes"] @ (f["sign"] * np.radians([bend, side, twist])))


def _card(twists=None, shared=False):
    """A card of the default character with the forearms twisted in ``lowerarm02`` alone (or
    shared by the table's shares), a bent index finger, a hand shape, a placement and points
    and pins in the card's frame."""
    import torch

    from opensculptboy.character import Character
    from opensculptboy.posing.reference_points import points_of, positions

    model = _model()
    twists = twists or {".R": 60.0, ".L": -40.0}
    changes = {"r-hand-scale-incr": 0.3, "l-hand-fingers-length-incr": -0.2}
    shaped = Character(local_changes=changes)
    rig, R = _rotations(shaped)
    for side, total in twists.items():
        if shared:
            for bone, share in (
                ("lowerarm01", 0.4),
                ("lowerarm02", 0.5),
                ("wrist", 0.1),
            ):
                R[rig.index[bone + side]] = _turn(rig, bone + side, twist=share * total)
        else:
            R[rig.index["lowerarm02" + side]] = _turn(
                rig, "lowerarm02" + side, twist=total
            )
    R[rig.index["finger2-2.R"]] = _turn(rig, "finger2-2.R", bend=40.0)
    P = torch.eye(4, dtype=torch.float64).repeat(len(model.bone_labels), 1, 1)
    P[:, :3, :3] = torch.as_tensor(R)
    character = Character(
        local_changes=changes,
        pose=Character.pose_from_parameters(model, P),
    )
    use = points_of(["arm.r"])
    X = positions(rig, R, use)
    Rp, t = _placement()
    card = character.to_dict()
    card["schema"] = 5
    card["detail"] = {
        "levels": [],
        "fit": {
            "placement": {"rotation": Rp.tolist(), "translation": t.tolist()},
            "reference": {
                "title": "Kouros",
                "points": {
                    "format": "opensculptboy.card-points/1",
                    "frame": "card",
                    "chains": {
                        "hand.r": [
                            {
                                "name": p["name"],
                                "position": x.tolist(),
                                "on": p["name"] != "pinky_tip.R",
                                "weight": 1.0,
                                "confidence": 0.9,
                                "placed": "auto",
                            }
                            for p, x in zip(use, X)
                            if p["chain"] == "hand.r"
                        ],
                        "arm.r": [
                            {"name": p["name"], "position": x.tolist()}
                            for p, x in zip(use, X)
                            if p["chain"] == "arm.r"
                        ],
                    },
                },
                "pins": [
                    {
                        "name": "wrist.R",
                        "kind": "plane",
                        "point": X[0].tolist(),
                        "normal": [0.0, 0.0, 2.0],
                        "weight": 5.0,
                    }
                ],
            },
        },
    }
    return card, rig, R, X


class TestApplyCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from opensculptboy.statues.card_config import apply_card

        cls.card, cls.rig, cls.R, cls.X = _card()
        config = {
            **BASE,
            "chains": ["hand.l"],
            "posed_hands": {
                "hand.l": {
                    "clear_body": 0.002,
                    "local_changes": {"l-hand-scale-incr": 0.5},
                }
            },
        }
        cls.out, cls.report = apply_card(cls.card, config, model=_model())

    def test_points_go_to_the_masks_frame_by_solve_chain(self):
        from opensculptboy.statues.points import in_anny

        points = self.out["reference_points"]
        self.assertEqual(set(points), {"arm.r"})
        entries = [e for e in points["arm.r"] if "kind" not in e]
        self.assertEqual(len(entries), 23)
        by = {e["name"]: e for e in entries}
        self.assertFalse(by["pinky_tip.R"]["on"])
        self.assertTrue(by["elbow.R"]["on"])
        # back in Anny's frame they lie where the placed card puts its points
        Rp, t = _placement()
        anny = {e["name"]: e for e in in_anny(points)["arm.r"] if "kind" not in e}
        from opensculptboy.posing.reference_points import points_of

        for p, x in zip(points_of(["arm.r"]), self.X):
            np.testing.assert_allclose(
                anny[p["name"]]["position"], Rp @ x + t, atol=1e-5
            )

    def test_pins_join_their_points_chain(self):
        from opensculptboy.fit.handoff import GLTF_TO_ANNY

        pins = [e for e in self.out["reference_points"]["arm.r"] if "kind" in e]
        self.assertEqual(len(pins), 1)
        pin = pins[0]
        self.assertEqual(
            (pin["name"], pin["kind"], pin["weight"]), ("wrist.R", "plane", 5.0)
        )
        Rp, _ = _placement()
        np.testing.assert_allclose(
            np.asarray(pin["normal"]) @ GLTF_TO_ANNY.T, Rp @ [0, 0, 1.0], atol=1e-6
        )
        self.assertEqual(
            self.report["reference_points"]["arm.r"],
            {"points": 23, "on": 22, "pins": 1},
        )

    def test_the_forearm_twist_is_shared_by_the_table(self):
        import roma
        import torch

        from opensculptboy.posing.point_solve import _coordinates, joint_frames

        frames = joint_frames(self.rig)
        for hand, side, total in (("hand.r", ".R", 60.0), ("hand.l", ".L", -40.0)):
            bones = self.out["posed_hands"][hand]["key"]["bones"]
            twist = {}
            for name in ("lowerarm01", "lowerarm02", "wrist"):
                M = roma.rotvec_to_rotmat(
                    torch.as_tensor(np.radians(bones[name]))
                ).numpy()
                twist[name] = np.degrees(
                    _coordinates(frames[self.rig.index[name + side]], M)
                )[2]
            whole = sum(twist.values())
            with self.subTest(hand):
                # the shares hold; the whole twist moves a little to keep the hand in place
                self.assertAlmostEqual(whole, total, delta=5.0)
                self.assertAlmostEqual(twist["lowerarm01"], 0.4 * whole, places=2)
                self.assertAlmostEqual(twist["lowerarm02"], 0.5 * whole, places=2)
                self.assertAlmostEqual(twist["wrist"], 0.1 * whole, places=2)
                self.assertGreater(
                    min(self.report["hands"][hand]["twist_change_deg"].values()), 1.0
                )
                # and the hand's points stay where the card has them
                self.assertLess(self.report["hands"][hand]["points_moved_mm"], 0.5)

    def test_the_key_holds_every_bone_of_the_arm(self):
        import roma
        import torch

        bones = self.out["posed_hands"]["hand.r"]["key"]["bones"]
        for name in (
            "shoulder01",
            "upperarm01",
            "upperarm02",
            "wrist",
            "metacarpal1",
            "finger1-1",
            "finger5-3",
        ):
            self.assertIn(name, bones)
        self.assertEqual(len(bones), self.report["hands"]["hand.r"]["bones"])
        M = roma.rotvec_to_rotmat(
            torch.as_tensor(np.radians(bones["finger2-2"]))
        ).numpy()
        np.testing.assert_allclose(M, self.R[self.rig.index["finger2-2.R"]], atol=1e-6)
        np.testing.assert_allclose(bones["upperarm01"], [0, 0, 0], atol=1e-4)

    def test_the_key_poses_the_hand_as_the_card_does_apart_from_the_twist(self):
        import torch

        from opensculptboy.statues.chains import _identity_pose
        from opensculptboy.statues.hands import key_hand

        model = _model()
        pose = _identity_pose(model).clone()
        key = self.out["posed_hands"]["hand.r"]["key"]
        key_hand(model, pose, "hand.r", key)
        labels = list(model.bone_labels)
        for name in ("finger2-2", "finger3-1", "shoulder01"):
            b = labels.index(name + ".R")
            np.testing.assert_allclose(pose[0, b, :3, :3].numpy(), self.R[b], atol=1e-6)
        self.assertIsInstance(pose, torch.Tensor)

    def test_hands_keep_their_entry_and_get_their_shape_and_timing(self):
        hands = self.out["posed_hands"]
        self.assertFalse(hands["hand.r"]["after_fit"])
        self.assertTrue(hands["hand.l"]["after_fit"])
        self.assertEqual(hands["hand.l"]["clear_body"], 0.002)
        # the key's shape is the card's; the entry's own shape goes, so it cannot override it
        self.assertNotIn("local_changes", hands["hand.l"])
        self.assertEqual(
            hands["hand.r"]["key"]["local_changes"], {"r-hand-scale-incr": 0.3}
        )
        self.assertEqual(
            hands["hand.l"]["key"]["local_changes"],
            {"l-hand-fingers-length-incr": -0.2},
        )
        self.assertNotIn("clearance_mm", hands["hand.r"]["key"])
        self.assertEqual(sorted(self.out["chains"]), ["hand.l", "hand.r"])

    def test_the_pipeline_reads_the_config(self):
        from opensculptboy.statues.pipeline import StatueConfig

        config = StatueConfig.from_dict(self.out)
        self.assertIn("arm.r", config.reference_points)
        self.assertIn("key", config.posed_hands["hand.r"])

    def test_a_shared_pose_stays_as_it_is(self):
        from opensculptboy.statues.card_config import apply_card

        card, _, _, _ = _card(shared=True)
        _, report = apply_card(card, dict(BASE), model=_model())
        for hand in ("hand.r", "hand.l"):
            self.assertLess(report["hands"][hand]["points_moved_mm"], 0.01)
            self.assertLess(
                max(report["hands"][hand]["twist_change_deg"].values()), 0.01
            )

    def test_points_in_the_scan_frame_and_the_command(self):
        from opensculptboy.statues.card_config import card_points, main

        card = json.loads(json.dumps(self.card))
        block = card["detail"]["fit"]["reference"]["points"]
        block["frame"] = "scan"
        block["chains"] = {"neck": [{"name": "head_top", "position": [0.0, 1.7, 0.1]}]}
        card["detail"]["fit"]["reference"]["pins"] = []
        self.assertEqual(
            card_points(card),
            {
                "neck": [
                    {
                        "name": "head_top",
                        "position": [0.0, 1.7, 0.1],
                        "on": True,
                        "weight": 1.0,
                    }
                ]
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            (tmp / "card.json").write_text(json.dumps(self.card))
            (tmp / "config.json").write_text(json.dumps(BASE))
            self.assertEqual(
                main(
                    [
                        str(tmp / "card.json"),
                        str(tmp / "config.json"),
                        "--out",
                        str(tmp / "new.json"),
                        "--hands",
                        "hand.r",
                        "--after-fit",
                        "hand.r=true",
                    ]
                ),
                0,
            )
            written = json.loads((tmp / "new.json").read_text())
        self.assertEqual(list(written["posed_hands"]), ["hand.r"])
        self.assertTrue(written["posed_hands"]["hand.r"]["after_fit"])


class TestPins(unittest.TestCase):
    def test_a_config_checks_its_pins(self):
        from opensculptboy.statues.pipeline import StatueConfig

        good = {"name": "wrist.R", "kind": "point", "target": [0, 1, 0]}
        config = StatueConfig.from_dict({**BASE, "reference_points": {"arm.r": [good]}})
        self.assertEqual(config.reference_points["arm.r"][0]["kind"], "point")
        for bad in (
            {"name": "wrist.R", "kind": "line", "target": [0, 1, 0]},
            {"name": "wrist.R", "kind": "plane", "point": [0, 1, 0]},
        ):
            with self.assertRaises(ValueError):
                StatueConfig.from_dict({**BASE, "reference_points": {"arm.r": [bad]}})

    def test_a_pin_holds_its_point_in_the_solve(self):
        from opensculptboy.character import Character
        from opensculptboy.posing.reference_points import points_of, positions
        from opensculptboy.statues.chains import _identity_pose
        from opensculptboy.statues.points import solve_on_pose

        model = _model()
        character = Character()
        rig, R0 = _rotations(character)
        target = R0.copy()
        target[rig.index["lowerarm01.R"]] = _turn(rig, "lowerarm01.R", bend=40.0)
        target[rig.index["finger3-2.R"]] = _turn(rig, "finger3-2.R", bend=50.0)
        use = points_of(["arm.r"])
        X = positions(rig, target, use)
        entries = [{"name": p["name"], "position": x.tolist()} for p, x in zip(use, X)]
        wrist = X[[p["name"] for p in use].index("wrist.R")]
        # the wrist pinned 2 cm away: the pin wins over the points
        held = (wrist + [0.0, 0.0, 0.02]).tolist()
        pin = {"name": "wrist.R", "kind": "point", "target": held, "weight": 50.0}
        pose = _identity_pose(model).clone()
        _, free = solve_on_pose(
            model, character, np.eye(3), np.zeros(3), pose, "arm.r", entries
        )
        _, pinned = solve_on_pose(
            model, character, np.eye(3), np.zeros(3), pose, "arm.r", entries + [pin]
        )
        self.assertLess(free["max_mm"], 2.0)
        self.assertEqual(free["pins_mm"], {})
        self.assertLess(pinned["pins_mm"]["wrist.R"], 5.0)
        self.assertGreater(pinned["distance_mm"]["wrist.R"], 15.0)


if __name__ == "__main__":
    unittest.main()

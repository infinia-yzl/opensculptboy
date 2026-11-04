# OpenSculptBoy
# Apache License, Version 2.0
"""
Parity of the page's Squeeze and balance mark (``viewer/src/pose_edit/contact.ts``) with
``opensculptboy.posing.contact``: the page closes the same fingers onto a ball to the same
scale and clearance, and finds the same centre of mass, support polygon and margin. The
editor's Grip (``gripFrom``) starts from the hand as it stands: from straight fingers it closes
as Squeeze does; fingers clenched into the ball open first and then close to 1 mm; and fingers
inside a ball at every curl stay as they are, marked blocked.
"""

import json
import pathlib
import subprocess
import tempfile
import unittest

import numpy as np

from test.test_reference_points import _node_available, _rig

REPO = pathlib.Path(__file__).resolve().parents[1]
NODE_SCRIPT = REPO / "viewer" / "test" / "pose_edit_contact.mjs"


@unittest.skipUnless(_node_available(), "node 22 or later runs the page's TypeScript")
class TestPageContact(unittest.TestCase):
    def test_squeeze_and_balance_match_python(self):
        from opensculptboy.posing import contact
        from opensculptboy.posing import point_solve as ps
        from opensculptboy.posing.point_solve import _hand_axes

        rig, rest = _rig()
        J = rig.to_output(rig.joints(rig.transforms(rest)))
        ix = rig.index
        palm, _ = _hand_axes(rig, ".R")
        centre = 0.5 * (J[ix["wrist.R"]] + J[ix["finger3-1.R"]]) + palm * 0.042
        ball = {"kind": "sphere", "centre": centre.tolist(), "radius": 0.03}
        fingers = ["finger1", "finger2", "finger3", "finger4", "finger5"]
        frames = ps.joint_frames(rig)
        bent = rest.copy()
        for name in ("spine04", "spine03", "spine02"):
            f = frames[ix[name]]
            bent[ix[name]] = ps._exp(
                f["axes"] @ (f["sign"] * np.radians([30.0, 0.0, 0.0]))
            )
        cases = [
            {
                "kind": "squeeze",
                "rotations": rest,
                "hand": "hand.r",
                "objects": [ball],
                "value": 1.0,
                "fingers": fingers,
            },
            {
                "kind": "squeeze",
                "rotations": rest,
                "hand": "hand.r",
                "objects": [ball],
                "value": 0.5,
                "fingers": fingers,
            },
            {"kind": "balance", "rotations": rest},
            {"kind": "balance", "rotations": bent},
        ]
        # the fingers clenched beyond the ball (1.4 times the fist's bends), and a ball that holds the whole hand
        clenched = rest.copy()
        for f in fingers:
            for j in range(3):
                b = ix[f"{f}-{j + 1}.R"]
                fr = frames[b]
                bend = np.radians(contact.FIST_BENDS[f][j]) * 1.4
                clenched[b] = ps._exp(
                    fr["axes"] @ (fr["sign"] * np.array([bend, 0.0, 0.0]))
                )
        big = {"kind": "sphere", "centre": centre.tolist(), "radius": 0.2}
        grips = [
            {
                "kind": "grip",
                "rotations": rest,
                "objects": [ball],
                "value": 1.0,
                "expect": "squeeze",
            },
            {
                "kind": "grip",
                "rotations": clenched,
                "objects": [ball],
                "value": 1.0,
                "expect": "opened",
            },
            {
                "kind": "grip",
                "rotations": clenched,
                "objects": [ball],
                "value": 0.0,
                "expect": "start",
            },
            {
                "kind": "grip",
                "rotations": clenched,
                "objects": [big],
                "value": 1.0,
                "expect": "blocked",
            },
        ]
        cases += [
            {**g, "hand": "hand.r", "fingers": fingers, "rest": rest} for g in grips
        ]
        data = {
            "rig": {
                "labels": rig.labels,
                "parents": rig.parents.tolist(),
                "heads": rig.heads.tolist(),
                "frame": rig.frame.tolist(),
            },
            "rest": rig.rest_vertices.tolist(),
            "indices": rig.indices.tolist(),
            "weights": rig.weights.tolist(),
            "k": int(rig.indices.shape[1]),
            "faces": rig.faces.tolist(),
            "bind": rig.bind.reshape(-1, 16).tolist(),
            "cases": [
                {
                    **{k: v for k, v in c.items() if k != "rest"},
                    "rotations": np.asarray(c["rotations"]).reshape(-1, 9).tolist(),
                }
                for c in cases
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "input.json").write_text(json.dumps(data))
            subprocess.run(["node", str(NODE_SCRIPT), tmp], check=True)
            out = json.loads(pathlib.Path(tmp, "output.json").read_text())
        for c, page in zip(cases, out):
            with self.subTest(kind=c["kind"], value=c.get("value")):
                if c["kind"] == "grip":
                    self._check_grip(c, page, contact, rig, fingers)
                elif c["kind"] == "squeeze":
                    D, report = contact.squeeze(
                        rig,
                        c["rotations"],
                        c["hand"],
                        c["objects"],
                        c["value"],
                        fingers=fingers,
                    )
                    for f in fingers:
                        for key in ("contact", "scale"):
                            if report[f][key] is None:
                                self.assertIsNone(page["report"][f][key])
                            else:
                                self.assertAlmostEqual(
                                    report[f][key], page["report"][f][key], delta=2e-4
                                )
                        self.assertAlmostEqual(
                            report[f]["clearance_mm"],
                            page["report"][f]["clearance_mm"],
                            delta=0.02,
                        )
                        self.assertEqual(
                            report[f]["inside"], page["report"][f]["inside"]
                        )
                    Dp = np.asarray(page["rotations"]).reshape(-1, 3, 3)
                    self.assertLess(
                        np.abs(rig.vertices(D) - rig.vertices(Dp)).max(), 1e-5
                    )
                else:
                    b = contact.balance(rig, c["rotations"])
                    p = page["balance"]
                    np.testing.assert_allclose(b["centre"], p["centre"], atol=1e-9)
                    np.testing.assert_allclose(b["mark"], p["mark"], atol=1e-9)
                    self.assertEqual(b["inside"], p["inside"])
                    self.assertAlmostEqual(b["margin_mm"], p["margin_mm"], delta=0.011)
                    self.assertEqual(len(b["polygon"]), len(p["polygon"]))
                    np.testing.assert_allclose(b["polygon"], p["polygon"], atol=1e-8)
                print(f"page {c['kind']}: {page['ms']:.0f} ms")

    def _check_grip(self, c, page, contact, rig, fingers):
        report, expect = page["report"], c["expect"]
        Dp = np.asarray(page["rotations"]).reshape(-1, 3, 3)
        if expect == "blocked":
            # inside at every curl: each finger keeps its present curl and says so
            self.assertTrue(all(report[f]["blocked"] for f in fingers))
            np.testing.assert_allclose(
                Dp, np.asarray(c["rotations"]).reshape(-1, 3, 3), atol=1e-9
            )
            return
        # a finger inside the ball even straight (the thumb beside the palm) cannot grip it; the others do
        _, straight = contact.squeeze(
            rig, c["rest"], c["hand"], c["objects"], 0.0, fingers=fingers
        )
        for f in fingers:
            r = report[f]
            if straight[f]["inside"]:
                # it stays as it was inside, or keeps outside where it stood
                self.assertTrue(r["blocked"] or r["clearance_mm"] >= 0, f)
                continue
            self.assertFalse(r["blocked"], f)
            if c["value"] == 0.0:
                # Squeeze at 0: each finger at its start, clear of the ball
                self.assertGreater(r["clearance_mm"], 1.0, f)
            elif r["contact"] is not None or r["jointwise"]:
                # each finger at contact with 1 mm to spare (within the bisection's step)
                self.assertGreaterEqual(r["clearance_mm"], 1.0, f)
                self.assertLess(r["clearance_mm"], 1.5, f)
        if expect == "squeeze":
            # from straight fingers Grip closes as Squeeze does; a finger that the shared curl brings to no contact
            # closes on joint by joint and comes within the clearance
            D, sq = contact.squeeze(
                rig, c["rotations"], c["hand"], c["objects"], 1.0, fingers=fingers
            )
            for f in fingers:
                self.assertFalse(report[f]["opened"], f)
                if report[f]["blocked"]:
                    continue
                if report[f]["jointwise"]:
                    self.assertIsNone(sq[f]["contact"], f)
                    self.assertGreaterEqual(report[f]["clearance_mm"], 1.0, f)
                    self.assertLess(report[f]["clearance_mm"], 1.5, f)
                    continue
                self.assertAlmostEqual(
                    sq[f]["clearance_mm"], report[f]["clearance_mm"], delta=0.02
                )
            if not any(report[f]["jointwise"] or report[f]["blocked"] for f in fingers):
                self.assertLess(np.abs(rig.vertices(D) - rig.vertices(Dp)).max(), 1e-5)
        if expect == "opened":
            # the fingers clenched into the ball open before they close
            self.assertGreaterEqual(sum(report[f]["opened"] for f in fingers), 3)


if __name__ == "__main__":
    unittest.main()

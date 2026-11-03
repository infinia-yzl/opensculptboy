# OpenSculptBoy
# Apache License, Version 2.0
"""
Parity of the page's Squeeze and balance mark (``viewer/src/pose_edit/contact.ts``) with
``opensculptboy.posing.contact``: the page closes the same fingers onto a ball to the same
scale and clearance, and finds the same centre of mass, support polygon and margin.
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
                {**c, "rotations": np.asarray(c["rotations"]).reshape(-1, 9).tolist()}
                for c in cases
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "input.json").write_text(json.dumps(data))
            subprocess.run(["node", str(NODE_SCRIPT), tmp], check=True)
            out = json.loads(pathlib.Path(tmp, "output.json").read_text())
        for c, page in zip(cases, out):
            with self.subTest(kind=c["kind"], value=c.get("value")):
                if c["kind"] == "squeeze":
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


if __name__ == "__main__":
    unittest.main()

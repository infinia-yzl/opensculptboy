# OpenSculptBoy
# Apache License, Version 2.0
"""
Parity of the page's problem builder (``viewer/src/pose_edit/problem.ts``) with
``opensculptboy.posing.point_solve.build_problem``: the page builds the solve of reference
points from a rig in the space of the turns and the skin rows of the points' triangles, and it
must write the same problem, the same chains, joint coordinates and influences as Python does,
so that the page's solve (``solve.ts``) finds the pose that Python finds.
"""

import json
import pathlib
import subprocess
import tempfile
import unittest

import numpy as np

from test.test_reference_points import _library, _node_available, _random_pose, _rig

REPO = pathlib.Path(__file__).resolve().parents[1]
NODE_SCRIPT = REPO / "viewer" / "test" / "pose_edit_problem.mjs"
CHAINS = ["arm.r", "hand.l", "leg.l", "foot.r", "neck", "spine"]


def _close(test, a, b, path="problem", tol=1e-9):
    """Compare two JSON values: the same structure, numbers within tol."""
    if isinstance(a, dict):
        test.assertEqual(sorted(a), sorted(b), path)
        for k in a:
            _close(test, a[k], b[k], f"{path}.{k}", tol)
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b), path)
        for i, (x, y) in enumerate(zip(a, b)):
            _close(test, x, y, f"{path}[{i}]", tol)
    elif isinstance(a, bool) or a is None or isinstance(a, str):
        test.assertEqual(a, b, path)
    else:
        test.assertLess(abs(float(a) - float(b)), tol, f"{path}: {a} against {b}")


@unittest.skipUnless(_node_available(), "node 22 or later runs the page's TypeScript")
class TestPageProblem(unittest.TestCase):
    def test_the_page_builds_the_same_problem(self):
        from opensculptboy.character import Character
        from opensculptboy.posing import point_solve as ps
        from opensculptboy.posing.reference_points import (
            influences_of,
            points_of,
            positions,
        )

        # a body away from the default (its bind turns), placed, in a library pose
        card = Character(phenotype={"muscle": 0.8, "height": 0.3, "weight": 0.7})
        a = np.radians(25.0)
        Rz = np.array(
            [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0, 0, 1.0]]
        )
        rig, _ = _rig(card, 1.05 * Rz, np.array([0.2, -0.1, 0.4]))
        start = _library("mh_fight")
        truth = _random_pose(rig, start, ["arm.r", "leg.l"], 3)
        use = points_of(["arm.r", "leg.l"])
        places = positions(rig, truth, use)
        ring = {f"ring_{k}.R" for k in ("mcp", "pip", "dip", "tip")}
        points = [
            {
                "vertices": p["vertices"],
                "barycentric": p["barycentric"],
                "target": t.tolist(),
                "weight": 0.5 if p["name"] == "index_tip.R" else 1.0,
                "on": p["name"] not in ring,
                "finger": p["finger"],
            }
            for p, t in zip(use, places)
        ]
        stored = {p["name"]: p for p in use}
        pins = [
            {
                "vertices": stored["wrist.R"]["vertices"],
                "barycentric": stored["wrist.R"]["barycentric"],
                "kind": "plane",
                "point": (places[0] + [0.0, 0.0, 0.01]).tolist(),
                "normal": [0.1, 0.0, 1.0],
            },
            {
                "vertices": stored["toe.L"]["vertices"],
                "barycentric": stored["toe.L"]["barycentric"],
                "kind": "point",
                "target": places[-1].tolist(),
                "weight": 4.0,
            },
        ]
        rows = sorted({v for p in points + pins for v in p["vertices"]})
        skin = {
            "rest": {str(r): rig.rest_vertices[r].tolist() for r in rows},
            "indices": {str(r): rig.indices[r].tolist() for r in rows},
            "weights": {str(r): rig.weights[r].tolist() for r in rows},
            "bind": rig.bind.reshape(-1, 16).tolist(),
        }
        cases = [
            {"chains": ["arm.r"], "limits": True, "pins": False, "solve": True},
            {"chains": ["arm.r", "leg.l"], "limits": False, "pins": True, "solve": True},
            {"chains": ["hand.r"], "limits": True, "pins": False, "solve": False},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "input.json").write_text(
                json.dumps(
                    {
                        "rig": {
                            "labels": rig.labels,
                            "parents": rig.parents.tolist(),
                            "heads": rig.heads.tolist(),
                            "frame": rig.frame.tolist(),
                        },
                        "rotations": start.reshape(-1, 9).tolist(),
                        "skin": skin,
                        "points": points,
                        "pins": pins,
                        "cases": cases,
                        "chains": CHAINS,
                    }
                )
            )
            subprocess.run(["node", str(NODE_SCRIPT), tmp], check=True)
            out = json.loads(pathlib.Path(tmp, "output.json").read_text())
        # the influences of the points' skin
        expected = influences_of(rig, use)
        for f, page in zip(expected, out["influences"]):
            self.assertEqual(f["bones"].tolist(), page["bones"])
            np.testing.assert_allclose(f["s"], page["s"], atol=1e-12)
            np.testing.assert_allclose(f["u"], page["u"], atol=1e-12)
        # the chains and the joint coordinates
        for c in CHAINS:
            self.assertEqual(ps.chain_bones(rig, [c]), out["chains"][c], c)
        coords = ps.coordinates(rig, start)
        for label, deg in coords.items():
            np.testing.assert_allclose(
                np.degrees(out["coordinates"][label]), deg, atol=1e-3, err_msg=label
            )
        # each case's problem, field by field, and the page's solve of it against Python's
        inputs = [
            {**p, "influence": f}
            for p, f in zip(points, influences_of(rig, use))
        ]
        pin_in = [
            {**q, "influence": influences_of(rig, [stored["wrist.R"], stored["toe.L"]])[i]}
            for i, q in enumerate(pins)
        ]
        for case, page in zip(cases, out["cases"]):
            with self.subTest(chains=case["chains"]):
                problem = ps.build_problem(
                    rig,
                    start,
                    inputs,
                    case["chains"],
                    limits=case["limits"],
                    pins=pin_in if case["pins"] else (),
                )
                _close(self, json.loads(json.dumps(problem)), page["problem"])
                if case["solve"]:
                    D, _, _ = ps.solve_problem(json.loads(json.dumps(problem)))
                    Dp = np.asarray(page["solve"]["rotations"]).reshape(-1, 3, 3)
                    self.assertLess(
                        np.abs(rig.vertices(D) - rig.vertices(Dp)).max(), 1e-4
                    )


if __name__ == "__main__":
    unittest.main()

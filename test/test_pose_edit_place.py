# OpenSculptBoy
# Apache License, Version 2.0
"""
Parity of the page's first placement of the reference points on a scan
(``viewer/src/pose_edit/place.ts`` and its BVH, ``bvh.ts``) with
``opensculptboy.posing.point_place.place_points``: on a scan of a fist and on a scan that lies
off the skin, where some points fall back to the closest point, the page places every point at
the same spot with the same method, confidence and marks.
"""

import json
import pathlib
import subprocess
import tempfile
import unittest

import numpy as np

from test.test_reference_points import _library, _node_available, _rig

REPO = pathlib.Path(__file__).resolve().parents[1]
NODE_SCRIPT = REPO / "viewer" / "test" / "pose_edit_place.mjs"


def _case(rig, rotations, points, scan, triangles):
    """The page's inputs of a placement: each point's place and skin normal on the posed body,
    the posed vertices with their parts, and the scan."""
    from opensculptboy.posing.point_place import _vertex_normals, part_of
    from opensculptboy.posing.reference_points import influences_of

    K = rig.transforms(rotations)
    X = rig.to_output(rig.points(K, influences_of(rig, points)))
    V = rig.vertices(rotations)
    VN = _vertex_normals(V, rig.faces)
    strongest = rig.indices[np.arange(len(rig.indices)), rig.weights.argmax(1)]
    entries = []
    for p, x in zip(points, X):
        n = np.asarray(p["barycentric"]) @ VN[np.asarray(p["vertices"])]
        entries.append(
            {
                "name": p["name"],
                "joint": p["joint"],
                "position": x.tolist(),
                "normal": (n / np.linalg.norm(n)).tolist(),
            }
        )
    return {
        "points": entries,
        "body": {
            "vertices": V.tolist(),
            "parts": [part_of(rig.labels[b]) for b in strongest],
        },
        "scan": {
            "vertices": np.asarray(scan).tolist(),
            "triangles": np.asarray(triangles).tolist(),
        },
    }


@unittest.skipUnless(_node_available(), "node 22 or later runs the page's TypeScript")
class TestPagePlacement(unittest.TestCase):
    def test_the_page_places_the_points_as_python_does(self):
        from opensculptboy.posing.point_place import _vertex_normals, place_points
        from opensculptboy.posing.reference_points import load_points, points_of

        rig, rest = _rig()
        fist = rest.copy()
        library = _library("mh_fight")
        for b, label in enumerate(rig.labels):
            if label.endswith(".R") and label.startswith(("finger", "metacarpal")):
                fist[b] = library[b]
        V = rig.vertices(fist)
        N = _vertex_normals(V, rig.faces)
        # a scan of the fist 2 mm outside the skin; and a scan 6 mm inside it with a wavy offset and without the
        # left hand, where far and lost points fall back to the closest point or find nothing
        strongest = rig.indices[np.arange(len(rig.indices)), rig.weights.argmax(1)]
        left_hand = np.array(
            [
                rig.labels[b].endswith(".L")
                and rig.labels[b].startswith(("wrist", "metacarpal", "finger"))
                for b in strongest
            ]
        )
        wave = 0.006 + 0.03 * (np.sin(40 * V[:, 2]) > 0.95)
        cases = [
            (fist, points_of(["hand.r"]), V + 0.002 * N, rig.faces),
            (
                fist,
                load_points(),
                V - wave[:, None] * N,
                rig.faces[~left_hand[rig.faces].any(1)],
            ),
        ]
        inputs = [_case(rig, *c) for c in cases]
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "input.json").write_text(json.dumps({"cases": inputs}))
            subprocess.run(["node", str(NODE_SCRIPT), tmp], check=True)
            out = json.loads(pathlib.Path(tmp, "output.json").read_text())
        methods = set()
        for (rot, points, scan, T), page in zip(cases, out):
            expected = place_points(rig, rot, points, scan, T)
            for e, p in zip(expected, page["placed"]):
                with self.subTest(point=e["name"]):
                    self.assertEqual(e["name"], p["name"])
                    self.assertEqual(e["method"], p["method"])
                    self.assertEqual(e["on"], p["on"])
                    np.testing.assert_allclose(e["position"], p["position"], atol=1e-7)
                    self.assertAlmostEqual(e["confidence"], p["confidence"], delta=2e-4)
                    # a closest point on an edge or a corner of the scan has several triangles; the two pick
                    # different ones (Python's in its tree's order), so their normals' marks may differ there
                    marks = (
                        ("distance", "fused")
                        if e["method"] == "closest"
                        else ("distance", "normals", "fused")
                    )
                    for k in marks:
                        self.assertAlmostEqual(e["marks"][k], p["marks"][k], delta=2e-4)
                    methods.add(e["method"])
            print(
                f"page placement of {len(points)} points: BVH {page['build_ms']:.0f} ms, "
                f"placement {page['place_ms']:.0f} ms"
            )
        # both ways of placing a point were checked
        self.assertTrue({"normal", "closest"} <= methods, methods)


if __name__ == "__main__":
    unittest.main()

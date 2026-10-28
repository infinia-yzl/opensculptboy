# OpenSculptBoy
# Apache License, Version 2.0
"""
The faces of the viewer page (viewer/src/face/rig.ts) against Anny and the exports.

The page plays Anny's facial actions as the exports write them (``opensculptboy.viewer.face_tables``:
the mouth cavity behind the skin in each target) and ``jawOpen`` as the turn of the ``anny-mouth``
rig's jaw with its corrective, as the GLB animation ``jawOpen.jaw`` plays them. Node runs the page's
TypeScript on the coarse body and on its mixed subdivision, and the test compares both with the same
faces built in Python from Anny's model, ``anny.utils.subdivision.MixedSubdivision`` and
``opensculptboy.mouth.jaw``. The test skips when node is missing.
"""

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import anny
from anny.utils.subdivision import MixedSubdivision
from test.test_viewer_parity import node_available

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "viewer" / "test" / "face_parity.mjs"
STEP = 1e-5  # the offsets' step in the page's data, as the build writes them


def _vertex_normals(V, T):
    acc = np.zeros_like(V)
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    for k in range(3):
        np.add.at(acc, T[:, k], n)
    return acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-30)


@unittest.skipUnless(
    node_available(), "node 22 or later is needed to run the page's TypeScript"
)
class TestViewerExpressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from opensculptboy import viewer
        from opensculptboy.mouth.jaw import corrective_for

        cls.F = F = viewer.face_tables()
        cls.labels = F.labels
        page_model = anny.Anny(topology="anny-quads", phenotypes="all").to(
            dtype=torch.float64
        )
        base_index = page_model.base_mesh_vertex_indices.numpy()
        faces = page_model.faces.numpy()
        body = (base_index[faces] < 13380).all(1)
        cls.sub = MixedSubdivision.for_model(page_model, faces_mask=body)
        used = np.unique(faces[body])
        cls.used = used
        compact = -np.ones(len(F.rest), np.int64)
        compact[used] = np.arange(len(used))
        cls.dir = d = pathlib.Path(tempfile.mkdtemp())
        compact[faces[body]].astype(np.uint32).tofile(d / "quads.bin")
        cls.sub.renumbered_rows(compact, len(used)).astype(np.uint32).tofile(
            d / "rows.bin"
        )
        cls.sub.triangles.astype(np.uint32).tofile(d / "triangles.bin")
        rest = F.rest[used]
        rest.astype(np.float32).tofile(d / "coarse.bin")
        # the actions' offsets on the coarse body, in the page's records (x, y, z and a pad, steps of STEP)
        starts, counts, ids, recs = [], [], [], []
        off = F.offsets[:, used]
        for a in range(len(F.labels)):
            moved = np.nonzero(np.linalg.norm(off[a], axis=1) > 1e-6)[0]
            r = np.zeros((len(moved), 4), np.int16)
            r[:, :3] = np.round(off[a, moved] / STEP)
            starts.append(int(sum(counts)))
            counts.append(int(len(moved)))
            ids.append(moved)
            recs.append(r)
        np.concatenate(ids).astype(np.uint32).tofile(d / "ids.bin")
        np.concatenate(recs).tofile(d / "offsets.bin")
        cls.quantised = np.zeros_like(off)
        for a in range(len(F.labels)):
            cls.quantised[a, ids[a]] = recs[a][:, :3].astype(np.float64) * STEP
        jw = F.jaw_weights[used]
        on = np.nonzero(jw > 1e-6)[0]
        on.astype(np.uint32).tofile(d / "jaw_ids.bin")
        jw[on].astype(np.float32).tofile(d / "jaw_weights.bin")
        cls.jaw_weights = (
            np.where(jw > 1e-6, jw, 0.0).astype(np.float32).astype(np.float64)
        )
        # a pivot that four coarse vertices give exactly on this body (the page's regressor has its own test)
        near = np.argsort(np.linalg.norm(rest - F.pivot, axis=1))[:4]
        A = np.concatenate([rest[near].T, np.ones((1, 4))])
        w = np.linalg.solve(A, np.concatenate([F.pivot, [1.0]]))
        cls.rotvec = Rotation.from_matrix(F.rotation).as_rotvec()
        meta = dict(
            format="opensculptboy/page-faces@1",
            actions=F.labels,
            starts=starts,
            counts=counts,
            step=STEP,
            cleared=F.cleared,
            jaw=dict(
                action="jawOpen",
                rotation=cls.rotvec.tolist(),
                angle_degrees=float(np.degrees(np.linalg.norm(cls.rotvec))),
                pivot=F.pivot.tolist(),
                pivot_vertices=near.tolist(),
                pivot_weights=w.tolist(),
                pivot_error=0.0,
                vertices=int(len(on)),
            ),
            eye_turns={},
            tongue=dict(rest=[], motion={}),
            region=0,
            presets={},
            visemes={},
        )
        with open(REPO / "src/opensculptboy/data/vrm/expressions.json") as f:
            vrm = json.load(f)["presets"]
        cls.mixes = [
            {"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8, "browInnerUp": 0.5},
            {"jawOpen": 1.0},
            {"jawOpen": 0.5, "mouthFunnel": 0.3},
            vrm["aa"]["mix"],
            vrm["surprised"]["mix"],
            {
                "jawOpen": 0.7,
                "mouthClose": 0.4,
                "eyeBlinkLeft": 1.0,
                "noseSneerRight": 0.6,
            },
        ]
        inp = dict(
            meta=meta,
            n_body=int(len(used)),
            levels=int(cls.sub.base_level),
            mixes=cls.mixes,
        )
        with open(d / "input.json", "w") as f:
            json.dump(inp, f)
        subprocess.run(["node", str(SCRIPT), str(d)], check=True, cwd=REPO)
        cls.out = json.loads((d / "output.json").read_text())
        # the same faces in Python: the coarse body, the fine body, the jaw's weights and correctives
        cls.rest = rest
        cls.fine0 = cls.sub(F.rest)
        jfull = np.zeros((len(F.rest), 1))
        jfull[used, 0] = cls.jaw_weights
        cls.fine_jaw = cls.sub(jfull)[:, 0]
        ja = F.labels.index("jawOpen")
        cls.ja = ja
        R1 = F.rotation
        cls.coarse_corrective = corrective_for(
            rest, cls.quantised[ja], cls.jaw_weights, R1, np.zeros(3), F.pivot
        )
        full = np.zeros((len(F.rest), 3))
        full[used] = cls.quantised[ja]
        cls.fine_target = cls.sub(full)
        cls.fine_corrective = corrective_for(
            cls.fine0, cls.fine_target, cls.fine_jaw, R1, np.zeros(3), F.pivot
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _weights(self, mix):
        w = np.zeros(len(self.labels))
        for k, v in mix.items():
            w[self.labels.index(k)] = v
        return w

    def _turn(self, V, weights, wj):
        R = Rotation.from_rotvec(wj * self.rotvec).as_matrix()
        return V + weights[:, None] * ((V - self.F.pivot) @ (R - np.eye(3)).T)

    def expected_coarse(self, mix):
        w = self._weights(mix)
        V = self.rest.copy()
        for a in range(len(w)):
            if a != self.ja:
                V += w[a] * self.quantised[a]
        wj = w[self.ja]
        if wj:
            V = self._turn(V + wj * self.coarse_corrective, self.jaw_weights, wj)
        return V

    def expected_fine(self, mix):
        w = self._weights(mix)
        D = np.zeros((len(self.F.rest), 3))
        for a in range(len(w)):
            if a != self.ja and w[a]:
                D[self.used] += w[a] * self.quantised[a]
        P = self.fine0 + self.sub(D)
        wj = w[self.ja]
        if wj:
            P = self._turn(P + wj * self.fine_corrective, self.fine_jaw, wj)
        return P

    def test_tables_follow_anny(self):
        # the targets are Anny's facial actions, but where the exports keep the mouth cavity behind the skin
        model = self.F.model
        with torch.no_grad():
            rows = torch.zeros((2, len(self.labels)), dtype=torch.float64)
            rows[1, self.ja] = 1.0
            out = model(facial_actions=rows)["rest_vertices"].numpy()
        self.assertNotIn("jawOpen", self.F.cleared)
        self.assertLess(np.abs(out[1] - out[0] - self.F.offsets[self.ja]).max(), 1e-9)
        # about 22 degrees on the default body (docs/mouth.md)
        self.assertGreater(np.degrees(np.linalg.norm(self.rotvec)), 15)
        self.assertLess(np.degrees(np.linalg.norm(self.rotvec)), 30)

    def test_coarse_faces(self):
        for i, mix in enumerate(self.mixes):
            with self.subTest(mix=i):
                V = np.fromfile(self.dir / f"coarse_{i}.bin", np.float32).reshape(-1, 3)
                self.assertLess(np.abs(V - self.expected_coarse(mix)).max(), 2e-5)

    def test_jaw_open_is_anny(self):
        # at weight 1 the jaw's turn and its corrective give Anny's jawOpen, on the coarse body and the fine body
        V = np.fromfile(self.dir / "coarse_1.bin", np.float32).reshape(-1, 3)
        self.assertLess(np.abs(V - (self.rest + self.quantised[self.ja])).max(), 2e-5)
        P = np.fromfile(self.dir / "fine_1.bin", np.float32).reshape(-1, 3)
        self.assertLess(np.abs(P - (self.fine0 + self.fine_target)).max(), 3e-5)
        # half way the turn differs from half of the target: the jaw turns
        V = self.expected_coarse({"jawOpen": 0.5})
        linear = self.rest + 0.5 * self.quantised[self.ja]
        self.assertGreater(np.abs(V - linear).max(), 1e-4)

    def test_fine_faces(self):
        for i, mix in enumerate(self.mixes):
            with self.subTest(mix=i):
                P = np.fromfile(self.dir / f"fine_{i}.bin", np.float32).reshape(-1, 3)
                E = self.expected_fine(mix)
                self.assertEqual(P.shape, E.shape)
                self.assertLess(np.abs(P - E).max(), 3e-5)
                # the normals of the moved surface: those of its triangles
                N = np.fromfile(self.dir / f"normals_{i}.bin", np.float32).reshape(
                    -1, 3
                )
                NE = _vertex_normals(E, self.sub.triangles)
                cos = np.clip((N * NE).sum(1), -1, 1)
                self.assertLess(np.degrees(np.arccos(cos)).max(), 1.0)

    def test_sparse_subdivision(self):
        self.assertLess(self.out["sparse_error"], 1e-6)
        self.assertEqual(self.out["sparse_outside"], 0)

    def test_rest_again(self):
        self.assertLess(self.out["rest_error"], 1e-7)
        self.assertLess(self.out["rest_normal_error"], 1e-6)

    def test_eye_turns(self):
        # an eye-look action turns each eyeball rigidly: its rotation vector gives the eyeball's offsets
        F = self.F
        for name, entry in F.eye_turns.items():
            for side, rv in entry.items():
                with self.subTest(action=name, eye=side):
                    ids = F.eyes[side]
                    p = F.rest[ids]
                    c = p.mean(0)
                    moved = (p - c) @ Rotation.from_rotvec(rv).as_matrix().T + c
                    target = p + F.offsets[F.labels.index(name)][ids]
                    self.assertLess(np.abs(moved - target).max(), 2e-4)

    def test_tongue_motion(self):
        # the jaw actions turn Anny's tongue rigidly (the lower teeth follow that motion); tongueOut bends it
        self.assertEqual(self.F.tongue_motion["jawOpen"][0], "rigid")
        self.assertEqual(self.F.tongue_motion["tongueOut"][0], "follow")


class TestJawPivot(unittest.TestCase):
    def test_pivot_over_bodies(self):
        # the page's blend of coarse vertices gives the jaw's head on random bodies within half a millimetre
        from opensculptboy import viewer

        page_model = anny.Anny(topology="anny-quads", phenotypes="all")
        base_index = page_model.base_mesh_vertex_indices.numpy()
        faces = page_model.faces.numpy()
        used = np.unique(faces[(base_index[faces] < 13380).all(1)])
        near, weights, err = viewer.jaw_pivot_regressor(used, bodies=24)
        self.assertEqual(len(near), len(weights))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=4)
        self.assertLess(err, 5e-4)


if __name__ == "__main__":
    unittest.main()

# OpenSculptBoy
# Apache License, Version 2.0
"""
The page's view of a character card (docs/viewer.md, "A card's look"): the UVs and seams of the
fine body (opensculptboy.viewer) against the detail layer's UVs, the rows of a card's detail levels
behind the page's levels, and in node the eye layout, the eye axes, the eyeball's outer surface,
a level offset added inside the page's subdivision and MakeHuman's local changes
(viewer/test/card_parity.mjs), and the page's reading of a card's grooms and blinks, with the
weights of its blink targets (viewer/test/groom_parity.mjs).
"""

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import torch

import anny
from anny.utils.subdivision import (
    MixedSubdivision,
    catmull_clark,
    faces_weighted_to_bones,
)
from anny.viewer import export
from opensculptboy import viewer as card
from opensculptboy.detail.subdivision import level
from opensculptboy.eyes.geometry import EyeShape, outer_radius
from opensculptboy.eyes.texture import polar_to_uv
from opensculptboy.texture.eyes import turned_axes

from .test_viewer_parity import node_available

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "viewer" / "test" / "card_parity.mjs"
GROOM_SCRIPT = REPO / "viewer" / "test" / "groom_parity.mjs"
BODY = 13380  # anny's base-mesh vertices of the body (the page's fine body)


def _page_body(model):
    """the page's fine body: the mixed subdivision of anny's body quads, the body quads on their
    compact numbering, and the anny vertex of each compact vertex"""
    base_index = model.base_mesh_vertex_indices.numpy()
    faces = model.faces.numpy()
    body = (base_index[faces] < BODY).all(1)
    region = faces_weighted_to_bones(model, faces[body])
    sub = MixedSubdivision(model.template_vertices.shape[0], faces[body], region)
    used = np.unique(faces[body])
    return sub, np.searchsorted(used, faces[body]), used, np.nonzero(body)[0]


class TestCardRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = anny.Anny(topology="anny-quads").to(dtype=torch.float64)
        cls.template = cls.model.template_vertices.numpy()
        cls.sub, cls.quads, cls.used, cls.body_faces = _page_body(cls.model)
        cls.rows = card.detail_rows(
            cls.quads, len(cls.used), cls.used, cls.template, seed=0
        )

    def _page_levels(self, x):
        out, n, q = [], len(self.used), self.quads
        for _ in range(2):
            lv = catmull_clark(n, q)
            x = lv.operator.apply(x)
            out.append(x)
            n, q = lv.operator.shape[0], lv.quads
        return out

    def test_anny_rows_are_exact(self):
        entry = self.rows["anny"]
        self.assertEqual(entry["fallback"], {"1": 0, "2": 0})
        # a fresh field through both stencils meets at every row
        y = np.random.default_rng(7).random((len(self.template), 3))
        page = self._page_levels(y[self.used])
        for k in (1, 2):
            lv = level("anny", k)
            target = y
            for op in lv.operators:
                target = op.apply(target)
            rows = entry[str(k)].astype(np.int64)
            self.assertEqual(len(rows), len(page[k - 1]))
            self.assertLess(np.abs(target[rows] - page[k - 1]).max(), 1e-9)

    def test_sculpt_rows(self):
        if "sculpt" not in self.rows:
            self.skipTest("no sculpt topology in this installation")
        entry = self.rows["sculpt"]
        page = self._page_levels(self.template[self.used])
        shape = card._topology_template(
            "sculpt", self.template, level("sculpt", 0).vertex_count
        )
        for k in (1, 2):
            rows = entry[str(k)].astype(np.int64)
            missed = entry[f"{k}_missed"].astype(np.int64)
            # the patches change the stencil of a few rows about the eye corner alone
            self.assertGreater(len(missed), 0)
            self.assertLess(len(missed), 0.02 * len(rows))
            target = shape
            for op in level("sculpt", k).operators:
                target = op.apply(target)
            # the move takes each such row to the closest point of the level's surface, a blend of
            # three rows, within about a millimetre of the page's surface
            weights = entry[f"{k}_weights"].astype(np.float64)
            corners = entry[f"{k}_corners"].astype(np.int64)
            np.testing.assert_allclose(weights.sum(1), 1.0, atol=1e-6)
            self.assertGreaterEqual(weights.min(), -1e-6)
            point = np.einsum("mj,mjc->mc", weights, target[corners])
            moved = page[k - 1][missed] + entry[f"{k}_delta"]
            self.assertLess(np.abs(moved - point).max(), 1e-6)
            self.assertLess(np.linalg.norm(entry[f"{k}_delta"], axis=1).max(), 0.002)

    def test_closest_on_triangles(self):
        # points above a unit square of two triangles, and beside it
        V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
        T = np.array([[0, 1, 2], [0, 2, 3]])
        P = np.array(
            [[0.25, 0.75, 0.3], [0.8, 0.1, -0.2], [2.0, 0.5, 0.0], [-1, -1, 1]]
        )
        face, w = card.closest_on_triangles(P, V, T)
        Q = np.einsum("mj,mjc->mc", w, V[T[face]])
        np.testing.assert_allclose(
            Q, [[0.25, 0.75, 0], [0.8, 0.1, 0], [1, 0.5, 0], [0, 0, 0]], atol=1e-12
        )

    def test_uvs_and_seams(self):
        model = self.model
        uv_faces = model.face_texture_coordinate_indices.numpy()[self.body_faces]
        uvs = model.texture_coordinates.detach().numpy()
        fine = card.fine_uvs(self.sub, uv_faces, uvs)
        n = int(self.sub.triangles.max()) + 1
        seams = card.seam_split(self.sub.triangles, fine, n)
        # each drawn vertex is a fine vertex, and each corner keeps its own UV (glTF convention)
        self.assertTrue(
            np.array_equal(seams.source[seams.triangles], self.sub.triangles)
        )
        corner = fine.uvs[fine.uv_ids].reshape(-1, 2)
        drawn = seams.uvs[seams.triangles.ravel()]
        self.assertLess(np.abs(drawn[:, 0] - corner[:, 0]).max(), 1e-6)
        self.assertLess(np.abs(1 - drawn[:, 1] - corner[:, 1]).max(), 1e-6)
        self.assertGreater(seams.copies, 0)
        # outside the head's region the fine body has the faces of level 2: the UVs of an input
        # quad's corners are those of the detail layer's level 2 on that quad (with the edge points
        # of level 3 on the quads that border the region)
        lv = level("anny", 2)
        key = np.round(lv.uvs * 2**20).astype(np.int64)
        want = {}
        for f, uq in zip(lv.base_faces, lv.uv_quads):
            want.setdefault(int(f), set()).update(map(tuple, key[uq]))
        page_key = np.round(fine.uvs * 2**20).astype(np.int64)
        got = {}
        outside = ~np.asarray(self.sub.region, bool)[fine.base_faces]
        for t in np.nonzero(outside)[0]:
            f = int(self.body_faces[fine.base_faces[t]])
            got.setdefault(f, set()).update(map(tuple, page_key[fine.uv_ids[t]]))
        self.assertGreater(len(got), 5000)
        same = 0
        for f, uv in got.items():
            self.assertLessEqual(want[f], uv, f)
            same += uv == want[f]
        self.assertGreater(same, 0.8 * len(got))


@unittest.skipUnless(
    node_available(), "node 22 or later is needed to run the page's TypeScript"
)
class TestCardParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = pathlib.Path(tempfile.mkdtemp())
        rng = np.random.default_rng(5)
        # eyes
        polar = np.stack([rng.random(200) * 2.2, rng.random(200) * 2 * np.pi], 1)
        sides = rng.random(200) < 0.5
        gaze = rng.normal(size=(20, 3)) * [0.2, 1, 0.2] - [0, 2, 0]
        gaze[0] = [0, -1, 0]
        cls.shape = EyeShape.of(11.5, 0.012)
        cls.theta = np.linspace(0, np.pi, 181)
        cls.polar, cls.sides, cls.gaze = polar, sides, gaze
        # a level-1 offset on anny's level 1 and the page's first two levels
        model = anny.Anny(topology="anny-quads").to(dtype=torch.float64)
        template = model.template_vertices.numpy()
        sub, quads, used, _ = _page_body(model)
        rows = card.detail_rows(quads, len(used), used, template, seed=0)["anny"]
        lv1 = level("anny", 1)
        coarse = rng.random((len(template), 3)) * 0.1
        offsets = rng.normal(size=(lv1.vertex_count, 3)) * 1e-3
        n1 = catmull_clark(len(used), quads)
        lv2 = catmull_clark(n1.operator.shape[0], n1.quads)
        page_rows = np.arange(lv2.operator.shape[0], dtype=np.uint32)
        expected = lv2.operator.apply(
            n1.operator.apply(coarse[used]) + offsets[rows["1"].astype(np.int64)]
        )
        cls.expected_fine = expected
        # local changes on a few vertices
        local_model = anny.Anny(topology="anny-quads", local_changes="all").to(
            dtype=torch.float64
        )
        cls.local_model = local_model
        cls.vertex_ids = np.sort(rng.choice(len(template), 400, replace=False))
        lc = export.local_offsets(local_model, cls.vertex_ids)
        labels = lc["labels"]
        cls.local_values = []
        for i in range(6):
            pick = rng.choice(len(labels), 12, replace=False)
            cls.local_values.append(
                {labels[j]: float(rng.uniform(-1, 1)) for j in pick}
            )
        step = 1e-5
        q = np.round(lc["offsets"] / step)
        assert np.abs(q).max() < 32767
        for name, arr in (
            ("quads.bin", quads.astype(np.uint32)),
            ("rows.bin", page_rows),
            ("coarse.bin", coarse[used].astype(np.float32)),
            (
                "level1.bin",
                offsets[rows["1"].astype(np.int64)].astype(np.float32),
            ),
            ("local_ids.bin", lc["ids"]),
            ("local_offsets.bin", q.astype(np.int16)),
            ("local_bones.bin", lc["bone_deltas"].astype(np.float32)),
        ):
            arr.tofile(cls.dir / name)
        cls.n_bones = len(local_model.bone_labels)
        inp = dict(
            eyes=dict(
                polar=[
                    [float(s), float(p), bool(b)] for (s, p), b in zip(polar, sides)
                ],
                gaze=gaze.tolist(),
                theta=cls.theta.tolist(),
                iris=cls.shape.iris,
                cornea=cls.shape.cornea,
                bulge=0.8,
            ),
            subdivision=dict(n=int(len(used))),
            local=dict(
                tables=dict(
                    labels=labels,
                    starts=lc["starts"],
                    counts=lc["counts"],
                    bones=lc["bones"],
                    step=step,
                ),
                values=cls.local_values,
                vertices=int(len(cls.vertex_ids)),
                bones=cls.n_bones,
            ),
        )
        with open(cls.dir / "input.json", "w") as f:
            json.dump(inp, f)
        subprocess.run(["node", str(SCRIPT), str(cls.dir)], check=True, cwd=REPO)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_eye_layout(self):
        got = np.fromfile(self.dir / "eye_uv.bin", np.float64).reshape(-1, 2)
        for side, sel in (("L", self.sides), ("R", ~self.sides)):
            want = polar_to_uv(self.polar[sel], side)
            self.assertLess(np.abs(got[sel, 0] - want[:, 0]).max(), 1e-12)
            self.assertLess(np.abs(got[sel, 1] - (1 - want[:, 1])).max(), 1e-12)

    def test_eye_axes(self):
        got = np.fromfile(self.dir / "eye_axes.bin", np.float64).reshape(-1, 3, 3)
        axes = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
        for g, a in zip(self.gaze, got):
            self.assertLess(np.abs(a - turned_axes(axes, g)).max(), 1e-12)

    def test_outer_radius(self):
        got = np.fromfile(self.dir / "eye_radius.bin", np.float64)
        want = outer_radius(self.theta, self.shape, 0.8, 1.0)
        self.assertLess(np.abs(got - want).max(), 1e-12)

    def test_level_offsets(self):
        got = np.fromfile(self.dir / "fine.bin", np.float32).reshape(-1, 3)
        self.assertEqual(got.shape, self.expected_fine.shape)
        self.assertLess(np.abs(got - self.expected_fine).max(), 1e-5)

    def test_local_changes(self):
        V = np.fromfile(self.dir / "local_coarse.bin", np.float32).reshape(
            len(self.local_values), -1, 3
        )
        J = np.fromfile(self.dir / "local_joints.bin", np.float32).reshape(
            len(self.local_values), -1, 3
        )
        model = self.local_model
        with torch.no_grad():
            base = model()
        for i, values in enumerate(self.local_values):
            with torch.no_grad():
                out = model(local_changes_kwargs=values)
            dv = (out["rest_vertices"] - base["rest_vertices"])[0].numpy()
            dj = (out["rest_bone_heads"] - base["rest_bone_heads"])[0].numpy()
            # 16-bit offsets in steps of 1e-5 m, summed over a dozen changes
            self.assertLess(np.abs(V[i] - dv[self.vertex_ids]).max(), 1e-4, values)
            self.assertLess(np.abs(J[i] - dj).max(), 1e-5, values)


@unittest.skipUnless(
    node_available(), "node 22 or later is needed to run the page's TypeScript"
)
class TestGroomParity(unittest.TestCase):
    """The page reads the groom and blinks files as opensculptboy writes them."""

    @classmethod
    def setUpClass(cls):
        from opensculptboy.eyes import page
        from opensculptboy.groom.data import GroomData, GroomMaterial

        cls.dir = pathlib.Path(tempfile.mkdtemp())
        rng = np.random.default_rng(2)
        cls.groom = GroomData(
            kind="hair",
            points=rng.random((12, 3)),
            normals=rng.random((12, 3)),
            uvs=rng.random((12, 2)),
            triangles=np.arange(12).reshape(4, 3),
            material=np.array([0, 0, 1, 1]),
            materials=[
                GroomMaterial("cap", (0.1, 0.06, 0.04, 1.0), 0.7),
                GroomMaterial(
                    "strands",
                    (1, 1, 1, 1),
                    0.45,
                    "BLEND",
                    0.5,
                    True,
                    1.0,
                    ("base_color",),
                ),
            ],
            colors=rng.random((12, 4)),
        )
        cls.groom.save(cls.dir / "c.hair.safetensors")
        cls.names = [
            "eyeBlinkLeft",
            "eyeBlinkRight",
            "eyeSquintLeft",
            "eyeSquintRight",
            "eyeBlinkLeft.inbetween",
            "eyeBlinkRight.inbetween",
        ]
        cls.blinks = {
            "names": cls.names,
            "level": 2,
            "rows": np.array([3, 17, 40]),
            "offsets": rng.normal(size=(6, 3, 3)) * 1e-3,
            "lashes": rng.normal(size=(6, 5, 3)) * 1e-3,
            "brows": None,
        }
        page.save(cls.dir / "c.blinks.safetensors", cls.blinks)
        card = {
            "detail": {
                "levels": [{"subdiv": 2, "offsets": "d2"}],
                "textures": {"subdiv": 2, "eye_layout": "e.png"},
            },
            "material": {
                "reference_eyes": {"blinks": "c.blinks.safetensors"},
                "hair": {
                    "source": "reference",
                    "mesh": "c.hair.safetensors",
                    "textures": {"strands": {"base_color": "s.png"}},
                },
                "brows": {"source": "reference", "mesh": "c.brows.safetensors"},
            },
        }
        cls.closures = [
            [0, 0, 0, 0],
            [0.5, 0.25, 0.1, 0.0],
            [1, 1, 0, 1],
            [0.75, 0.0, 0.0, 0.0],
        ]
        cls.curve = [0.0, 0.1, 1 / 3, 0.5, 0.9, 1.0]
        (cls.dir / "input.json").write_text(
            json.dumps(
                {
                    "card": card,
                    "groom": "c.hair.safetensors",
                    "blinks": "c.blinks.safetensors",
                    "closures": cls.closures,
                    "curve": cls.curve,
                }
            )
        )
        subprocess.run(["node", str(GROOM_SCRIPT), str(cls.dir)], check=True, cwd=REPO)
        cls.out = json.loads((cls.dir / "output.json").read_text())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_plan(self):
        plan = self.out["plan"]
        self.assertEqual(plan["hair"]["mesh"], "c.hair.safetensors")
        self.assertEqual(plan["hair"]["textures"], {"strands": {"base_color": "s.png"}})
        self.assertEqual(plan["brows"], {"mesh": "c.brows.safetensors", "textures": {}})
        self.assertEqual(plan["blinks"], "c.blinks.safetensors")
        self.assertEqual(plan["shown"], 2)

    def test_groom(self):
        g = self.out["groom"]
        self.assertEqual(g["kind"], "hair")
        np.testing.assert_allclose(g["points"], self.groom.points.ravel()[:9])
        np.testing.assert_allclose(g["uvs"], self.groom.uvs.ravel()[:6], atol=1e-7)
        self.assertEqual(g["triangles"], 4)
        self.assertEqual(g["material"], [0, 0, 1, 1])
        np.testing.assert_allclose(
            g["colors"], self.groom.colors.ravel()[:8], atol=1e-7
        )
        strands = g["materials"][1]
        self.assertEqual(
            (strands["name"], strands["alpha"], strands["double"]),
            ("strands", "BLEND", True),
        )

    def test_blinks(self):
        b = self.out["blinks"]
        self.assertEqual(b["names"], self.names)
        self.assertEqual(b["level"], 2)
        self.assertEqual(b["rows"], [3, 17, 40])
        np.testing.assert_allclose(
            b["offsets"], self.blinks["offsets"].ravel()[:12], atol=1e-9
        )
        self.assertEqual(b["lashes"], 6 * 5 * 3)

    def test_weights(self):
        # the blink targets follow the closures, their in-between correctives rise to 1 half way
        # (the GLB animations "<target>.close"), and the squints follow theirs
        for (left, right, sl, sr), w in zip(self.closures, self.out["weights"]):
            want = dict(zip(self.names, w))
            self.assertAlmostEqual(want["eyeBlinkLeft"], left)
            self.assertAlmostEqual(want["eyeBlinkRight"], right)
            self.assertAlmostEqual(want["eyeSquintLeft"], sl, places=6)
            self.assertAlmostEqual(want["eyeSquintRight"], sr)
            for side, c in (("Left", left), ("Right", right)):
                tent = 0.0 if c in (0, 1) else 1 - abs(2 * c - 1)
                self.assertAlmostEqual(want[f"eyeBlink{side}.inbetween"], tent)

    def test_curve(self):
        c = self.out["curve"]
        self.assertEqual(c[0], 0.0)
        self.assertEqual(c[-1], 0.0)
        self.assertAlmostEqual(c[2], 1.0)
        self.assertTrue(0 < c[1] < 1 and 0 < c[4] < c[3] < 1)

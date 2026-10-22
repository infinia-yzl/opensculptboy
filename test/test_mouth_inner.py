# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of a reference's inner mouth (``opensculptboy.mouth.inner``, docs/mouth.md): the
handoff's ``mouth`` section read into parts with their roles and skins, the reference's joints
mapped onto the ``anny-mouth`` rig, the pieces brought into the card's rest frame, the file
of the bake, the card's ``material.mouth`` section, the mouth on the mesh of a file (anny's
cavity and tongue hidden, the skin of each piece, the pieces behind the skin at rest and with
the jaw open, the tongue following anny's) and the GLB and VRM files. Cards without the section
keep their files (``test.test_gltf_export.TestGltfReference``).
"""

import functools
import json
import pathlib
import tempfile
import unittest
import warnings

import numpy as np
import torch

from opensculptboy import Character, export_glb
from opensculptboy.material import MouthMaterial, check_material
from opensculptboy.mouth import MOUTH_RIG
from opensculptboy.mouth.inner import MouthData, anny_bone, map_skin, reference_mouth
from opensculptboy.teeth.generic import generic_teeth
from opensculptboy.teeth.mesh import PARTS, mouth_rows, rigid, toothed_mesh
from test.gltf_reader import GLB, evaluate_nodes, structure_errors


def _kabsch_residual(a, b) -> float:
    R, t = rigid(a, b)
    return float(np.linalg.norm(a @ R.T + t - b, axis=1).max())


def _crossings(vertices, triangles, points, mouth_triangles) -> int:
    """The number of edges of the mouth's triangles that cross the triangles of a mesh."""
    import trimesh

    edges = np.unique(
        np.sort(
            np.concatenate([mouth_triangles[:, [k, (k + 1) % 3]] for k in range(3)]), 1
        ),
        axis=0,
    )
    start, end = points[edges[:, 0]], points[edges[:, 1]]
    length = np.linalg.norm(end - start, axis=1)
    keep = length > 1e-9
    start, end, length = start[keep], end[keep], length[keep]
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    where, rays, _ = mesh.ray.intersects_location(
        start, (end - start) / length[:, None], multiple_hits=False
    )
    reach = np.linalg.norm(where - start[rays], axis=1)
    return int((reach < length[rays] - 1e-7).sum())


@functools.lru_cache(maxsize=1)
def _body():
    """The default body on the anny-mouth rig, its rest mesh and the targets of four mouth
    actions."""
    from opensculptboy.export.body import morph_target_rows, rest_rows

    card = Character(rig=MOUTH_RIG)
    model = card.build_model(dtype=torch.float64)
    names = ["jawOpen", "jawLeft", "tongueOut", "mouthSmileLeft"]
    targets, facial, faces = morph_target_rows(model, card, names)
    rows, _ = rest_rows(model, card, facial, faces)
    return model, rows[0], rows[1:] - rows[0], names


def synthetic_mouth(model, rest) -> MouthData:
    """A whole inner mouth for a model without a reference: the generic teeth and gums, and
    a tongue (anny's own, shrunk by 10 % about its centre), skinned as a reference's would be:
    the upper arch to the head, the lower arch to the jaw and the tongue to the tongue chain."""
    from opensculptboy.export.body import triangulated_faces, vertex_normals

    teeth = generic_teeth(model, rest)
    _, tongue = mouth_rows(model)
    triangles, _ = triangulated_faces(model)
    keep = np.zeros(len(rest), bool)
    keep[tongue] = True
    tri = triangles[keep[triangles].all(1)]
    rows = np.unique(tri)
    remap = np.full(len(rest), -1)
    remap[rows] = np.arange(len(rows))
    centre = rest[rows].mean(0)
    tongue_points = centre + 0.9 * (rest[rows] - centre)
    tri = remap[tri]
    n = len(teeth.points)
    points = np.concatenate([teeth.points, tongue_points])
    normals = np.concatenate([teeth.normals, vertex_normals(tongue_points, tri)])
    m = len(points)
    slots = np.zeros((m, 4), np.int64)
    weights = np.zeros((m, 4))
    weights[:, 0] = 1.0
    bones = ("head", "jaw", "tongue02")
    slots[:n, 0] = np.where(teeth.arch == 1, 1, 0)
    slots[n:, 0] = 2
    return MouthData(
        points=points,
        normals=normals,
        uvs=np.zeros((m, 2)),
        triangles=np.concatenate([teeth.triangles, tri + n]),
        arch=np.concatenate([teeth.arch, np.ones(len(rows), np.uint8)]),
        part=np.concatenate(
            [teeth.part, np.full(len(tri), PARTS.index("tongue"), np.uint8)]
        ),
        bones=bones,
        bone_slots=slots,
        bone_weights=weights,
    )


class TestHandoffMouth(unittest.TestCase):
    def test_pieces(self):
        from opensculptboy.fit.handoff import load_handoff, write_handoff
        from opensculptboy.fit.reference import Reference

        model, rest, _, _ = _body()
        from opensculptboy.export.body import triangulated_faces

        skin, _ = triangulated_faces(model)
        mouth = synthetic_mouth(model, rest)
        pieces = []
        for role, part in (("gums_upper", "gums"), ("tooth_lower_l1", "teeth")):
            tri = mouth.triangles[mouth.part == PARTS.index(part)]
            if role.startswith("tooth"):
                tri = tri[np.isin(tri, np.flatnonzero(mouth.arch == 1)).all(1)]
            pieces.append((role, mouth.points, tri))
        with tempfile.TemporaryDirectory() as tmp:
            write_handoff(tmp, rest, skin, mouth=pieces)
            handoff = load_handoff(tmp)
        parts = [p for p in handoff.parts if p.mouth_role]
        self.assertEqual(
            [p.mouth_role for p in parts], ["gums_upper", "tooth_lower_l1"]
        )
        for p in parts:
            self.assertEqual(p.role, "mouth")
            # the pieces keep only their own vertices, in Anny's frame
            self.assertEqual(p.triangles.max(), len(p.vertices) - 1)
            from scipy.spatial import cKDTree

            gap, _ = cKDTree(mouth.points).query(p.vertices)
            self.assertLess(gap.max(), 1e-6)  # float32 in the file
        reference = Reference.from_handoff(handoff)
        self.assertEqual(len(reference.mouth_parts), 2)
        # without a skin, each piece follows the bone of its arch
        data = reference_mouth(reference.mouth_parts, rest, rest, mouth_rows(model)[0])
        bones = np.asarray(data.bones)[data.bone_slots[:, 0]]
        np.testing.assert_array_equal(bones == "jaw", data.arch == 1)
        self.assertEqual(set(np.unique(data.part)), {0, 1})


class TestJoints(unittest.TestCase):
    def test_anny_bones(self):
        tongue = [f"Tongue_{k}" for k in range(6)]
        self.assertEqual(anny_bone("Jaw", tongue), "jaw")
        self.assertEqual(anny_bone("Chin_02_R", tongue), "jaw")
        self.assertEqual(anny_bone("LowerLip_Edge_L", tongue), "jaw")
        self.assertEqual(anny_bone("Head", tongue), "head")
        self.assertEqual(anny_bone("UpperLip", tongue), "head")
        # the chain by its place: root to root, tip to tip
        self.assertEqual(anny_bone("Tongue_0", tongue), "tongue00")
        self.assertEqual(anny_bone("Tongue_5", tongue), "tongue04")
        self.assertEqual(anny_bone("Tongue_2", tongue), "tongue02")

    def test_map_skin(self):
        from opensculptboy.fit.handoff import HandoffPart

        names = ["Head", "Jaw"] + [f"Tongue_{k}" for k in range(6)]
        slots = np.array([[0, 1, 0, 0], [1, 2, 3, 0], [7, 6, 1, 0]])
        weights = np.array([[0.7, 0.3, 0, 0], [0.5, 0.3, 0.2, 0], [0.6, 0.3, 0.1, 0]])
        part = HandoffPart(
            "t",
            "mouth",
            np.zeros((3, 3)),
            np.array([[0, 1, 2]]),
            mouth_role="tongue",
            joints=names,
            joint_slots=slots,
            joint_weights=weights,
        )
        bones, s, w = map_skin(part)
        np.testing.assert_allclose(w.sum(1), 1.0)
        got = [dict(zip(np.asarray(bones)[s[i]], w[i])) for i in range(3)]
        self.assertAlmostEqual(got[0]["head"], 0.7)
        self.assertAlmostEqual(got[0]["jaw"], 0.3)
        self.assertAlmostEqual(got[1]["jaw"], 0.5)
        self.assertAlmostEqual(got[1]["tongue00"], 0.3)
        # Tongue_5 and Tongue_4 both land near the tip
        self.assertAlmostEqual(got[2]["tongue04"], 0.6)


class TestReferenceMouth(unittest.TestCase):
    def test_placement_and_file(self):
        from opensculptboy.fit.handoff import HandoffPart

        rng = np.random.default_rng(5)
        rest = rng.normal(size=(40, 3)) * 0.05 + [0.0, -0.05, 0.65]
        angle = 0.07
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        posed = rest @ R.T + [0.01, 0.0, 0.9]
        box = rng.normal(size=(6, 3)) * 0.01 + [0.0, -0.06, 1.55]
        tri = np.array([[0, 1, 2], [3, 4, 5]])
        parts = [
            HandoffPart("a", "mouth", box, tri, mouth_role="palate"),
            HandoffPart("b", "mouth", box + 0.001, tri, mouth_role="gums_lower"),
            HandoffPart("c", "mouth", box, tri, mouth_role="unknown_piece"),
            HandoffPart("d", "mouth", box, tri),  # no role: the teeth path's
        ]
        data = reference_mouth(parts, posed, rest, np.arange(40))
        self.assertEqual(len(data.points), 12)
        np.testing.assert_allclose(
            data.points[:6], (box - [0.01, 0.0, 0.9]) @ R, atol=1e-9
        )
        np.testing.assert_array_equal(data.arch, [0] * 6 + [1] * 6)
        self.assertEqual(
            [PARTS[k] for k in data.part], ["palate", "palate", "gums", "gums"]
        )
        self.assertIsNone(reference_mouth(parts[2:], posed, rest, np.arange(40)))
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "c.inner_mouth.safetensors"
            data.save(path)
            back = MouthData.load(path)
        self.assertEqual(back.bones, data.bones)
        self.assertTrue(back.interior)
        for key in (
            "points",
            "triangles",
            "arch",
            "part",
            "bone_slots",
            "bone_weights",
        ):
            np.testing.assert_array_equal(getattr(back, key), getattr(data, key))


class TestCard(unittest.TestCase):
    def test_section(self):
        self.assertIsNone(Character().mouth_material())
        card = Character(material={"mouth": {"mesh": "c.mouth.safetensors"}})
        mouth = card.mouth_material()
        self.assertEqual(mouth.mesh, "c.mouth.safetensors")
        self.assertEqual(mouth.textures(), {})
        # the section alone keeps the exporters' own skin
        self.assertIsNone(card.skin_material())
        self.assertEqual(Character.from_dict(card.to_dict()), card)
        for bad in ({}, {"mesh": 3}, {"teeth": 1}, {"mesh": "m", "clearcoat": 2.0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MouthMaterial.from_dict(bad)
        check_material({"mouth": {"mesh": "m", "base_color": "b.png"}})


class TestMouthMesh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.rest, cls.offsets, cls.names = _body()
        cls.mouth = synthetic_mouth(cls.model, cls.rest)
        cls.mesh = toothed_mesh(cls.model, None, None, cls.rest, cls.mouth)

    def test_hidden(self):
        from opensculptboy.export.body import triangulated_faces

        hidden = self.mesh.hidden_mask()
        triangles, _ = triangulated_faces(self.model)
        self.assertEqual(len(hidden), len(self.mesh.triangles))
        self.assertFalse(hidden[len(triangles) :].any())  # the mouth's own stay
        _, tongue = mouth_rows(self.model)
        on_tongue = np.isin(triangles, tongue).all(1)
        self.assertTrue(hidden[: len(triangles)][on_tongue].all())
        self.assertGreater(
            hidden.sum(), on_tongue.sum()
        )  # and the cavity behind the lips
        # the lips' inner margin stays: every hidden triangle lies behind it
        self.assertLess(hidden.sum(), 0.1 * len(triangles))

    def test_skin(self):
        weights = self.model.vertex_bone_weights.double().numpy()
        indices = self.model.vertex_bone_indices.numpy()
        w, i = self.mesh.skin_weights(weights, indices)
        labels = list(self.model.bone_labels)
        n = len(self.rest)
        np.testing.assert_allclose(w[n:].sum(1), 1.0)
        first = np.asarray(labels)[i[n:, 0]]
        tongue = self.mouth.part_points("tongue")
        self.assertTrue((first[tongue] == "tongue02").all())
        self.assertTrue((first[~tongue & (self.mouth.arch == 1)] == "jaw").all())
        self.assertTrue((first[self.mouth.arch == 0] == "head").all())
        # on a rig without the mouth bones every piece follows the head
        anny = Character().build_model(dtype=torch.float64)
        plain = toothed_mesh(anny, None, None, self.rest, self.mouth)
        w, i = plain.skin_weights(
            anny.vertex_bone_weights.double().numpy(), anny.vertex_bone_indices.numpy()
        )
        heads = np.asarray(list(anny.bone_labels))[i[n:, 0]]
        self.assertTrue((heads == "head").all())

    def test_behind_the_skin(self):
        # nothing of the mouth goes through the skin that the file draws: at rest a ray from
        # each point toward the front meets the closed lips, unless it leaves through the slit
        # that anny's lips keep between them; at rest and with the jaw open no edge of the
        # mouth crosses the outer skin (the lining of anny's mouth faces into the mouth)
        import trimesh

        from opensculptboy.teeth.generic import mouth_frame
        from opensculptboy.teeth.mesh import cavity_rows

        drawn = ~self.mesh.hidden_mask()[: self.mesh.base_triangle_count]
        skin = self.mesh.base_triangles[drawn]
        mesh = trimesh.Trimesh(self.rest, skin, process=False)
        points = self.mouth.points
        hit = mesh.ray.intersects_any(
            points, np.tile([0.0, -1.0, 0.0], (len(points), 1))
        )
        ring = self.rest[mouth_frame(self.model, self.rest)["ring"]]
        middle = ring[np.abs(ring[:, 0] - ring[:, 0].mean()) < 0.005]
        slit = (points[:, 2] > middle[:, 2].min() - 0.001) & (
            points[:, 2] < middle[:, 2].max() + 0.001
        )
        self.assertTrue((hit | slit).all(), f"{int((~(hit | slit)).sum())} in the open")
        lining = np.zeros(len(self.rest), bool)
        lining[cavity_rows(self.model, None, keep_rings=0)] = True
        outer = skin[~lining[skin].any(1)]
        out = self.mesh.targets(self.offsets)
        n = len(self.rest)
        r = self.names.index("jawOpen")
        for name, x, moved in (
            ("rest", self.rest, points),
            ("jawOpen", self.rest + self.offsets[r], points + out[r, n:]),
        ):
            with self.subTest(pose=name):
                self.assertEqual(_crossings(x, outer, moved, self.mouth.triangles), 0)

    def test_targets(self):
        out = self.mesh.targets(self.offsets)
        n = len(self.rest)
        mouth = out[:, n:]
        upper = self.mouth.arch == 0
        tongue = self.mouth.part_points("tongue")
        lower = ~upper & ~tongue
        p = self.mouth.points
        r = self.names.index("jawOpen")
        self.assertLess(np.abs(mouth[r, upper]).max(), 1e-9)
        self.assertLess(_kabsch_residual(p[lower], p[lower] + mouth[r, lower]), 1e-9)
        # tongueOut bends anny's tongue: the reference's tongue follows it forward
        t = self.names.index("tongueOut")
        self.assertLess(mouth[t, tongue, 1].mean(), -0.003)
        self.assertLess(np.abs(mouth[t, lower]).max(), 1e-9)


class TestHeldMouth(unittest.TestCase):
    def test_pose_keeps_the_mouth_shut(self):
        # a posed card on anny-mouth keeps the jaw and the tongue at their rest turns, so its
        # skin matches the same card on anny's rig, whose head carries the mouth
        shape = {"mouth-trans-backward-forward": -1.0, "chin-prominent": 1.0}
        pose = {
            "bones": {"head": [0.08, 0.03, 0.0, 0.996], "neck01": [0.05, 0, 0, 0.9987]}
        }
        positions = {}
        with tempfile.TemporaryDirectory() as tmp:
            for rig in ("anny", MOUTH_RIG):
                card = Character(rig=rig, face_shapes=shape, pose=pose)
                path = pathlib.Path(tmp) / f"{rig}.glb"
                export_glb(path, card, max_influences=8)
                glb = GLB(path)
                node = min(evaluate_nodes(glb, animation="pose"))
                positions[rig] = evaluate_nodes(glb, animation="pose")[node][
                    0
                ].positions
        gap = np.linalg.norm(positions["anny"] - positions[MOUTH_RIG], axis=1)
        self.assertLess(gap.max(), 1e-4)


class TestExports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        model, rest, _, _ = _body()
        cls.model = model
        synthetic_mouth(model, rest).save(cls.folder / "c.inner_mouth.safetensors")
        cls.card = Character(
            name="mouth",
            rig=MOUTH_RIG,
            material={"mouth": {"mesh": "c.inner_mouth.safetensors"}},
        )
        cls.path = cls.folder / "mouth.glb"
        export_glb(cls.path, cls.card, cls.model, detail_dir=cls.folder)
        cls.glb = GLB(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _primitives(self, glb):
        g = glb.json
        return {
            g["materials"][p["material"]]["name"]: p
            for p in g["meshes"][0]["primitives"]
        }

    def test_structure(self):
        g = self.glb.json
        self.assertEqual(structure_errors(self.glb), [])
        prims = self._primitives(self.glb)
        self.assertEqual(sorted(prims), ["mouth", "skin"])
        extras = {m["name"]: m for m in g["materials"]}["mouth"]["extras"]
        self.assertEqual(extras["opensculptboy_teeth"]["role"], "mouth")
        info = g["scenes"][0]["extras"]["opensculptboy"]["inner_mouth"]
        self.assertGreater(info["hidden"], 0)
        self.assertNotIn("teeth", g["scenes"][0]["extras"]["opensculptboy"])
        # the skin's primitive leaves out anny's cavity and tongue
        from opensculptboy.export.body import triangulated_faces

        skin = len(self.glb.accessor(prims["skin"]["indices"])) // 3
        self.assertEqual(skin, len(triangulated_faces(self.model)[0]) - info["hidden"])

    def test_jaw_turn_moves_the_lower_pieces(self):
        from opensculptboy.mouth.jaw import JAW_ANIMATION

        prims = self._primitives(self.glb)
        rows = np.unique(self.glb.accessor(prims["mouth"]["indices"]))
        node = min(evaluate_nodes(self.glb))
        rest = evaluate_nodes(self.glb)[node][0].positions[rows]
        opened = evaluate_nodes(self.glb, animation=JAW_ANIMATION, frame=1)
        moved = opened[node][0].positions[rows]
        d = np.linalg.norm(moved - rest, axis=1)
        self.assertGreater(d.max(), 0.003)
        self.assertGreater((d < 1e-4).mean(), 0.2)  # the upper arch stays

    def test_vrm(self):
        from opensculptboy.export.vrm import export_vrm
        from test.test_vrm_schema import check_vrm0, check_vrm1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for version in ("1.0", "0.x"):
                with self.subTest(version=version):
                    path = self.folder / f"mouth{version}.vrm"
                    export_vrm(
                        path,
                        self.card,
                        self.model,
                        version=version,
                        author="tests",
                        refine_weights=False,
                        detail_dir=self.folder,
                    )
                    glb = GLB(path)
                    g = glb.json
                    self.assertEqual(structure_errors(glb), [])
                    check = check_vrm1 if version == "1.0" else check_vrm0
                    self.assertEqual(check(g), [])
                    self.assertEqual(
                        [m["name"] for m in g["materials"]], ["skin", "mouth"]
                    )
                    extras = g["scenes"][0]["extras"]["opensculptboy"]
                    self.assertGreater(extras["inner_mouth"]["hidden"], 0)

    def test_command(self):
        # the bake gives a card with an inner mouth the anny-mouth rig
        from opensculptboy.cli import _mouth_rig

        card = Character(material={"mouth": {"mesh": "m"}})
        _mouth_rig(card)
        self.assertEqual(card.rig, MOUTH_RIG)
        plain = Character()
        _mouth_rig(plain)
        self.assertEqual(plain.rig, "anny")
        out = self.folder / "cli.glb"
        path = self.folder / "card.json"
        path.write_text(json.dumps(self.card.to_dict()))
        from opensculptboy.cli import main

        self.assertEqual(main(["export", str(out), "--character", str(path)]), 0)
        self.assertIn("mouth", self._primitives(GLB(out)))


if __name__ == "__main__":
    unittest.main()

# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the lashes of ``opensculptboy.eyes.lashes``: the card's ``material.lashes`` section,
the lid margins, the generic lashes and their texture, the lashes on the eyed mesh (back at
rest, with the closing lid in a blink, still with the other eye's blink and in a turn of the
eye), the lashes of a reference brought into the rest frame with their textures stacked, the
lashes file, and the GLB and VRM files with and without lashes.
"""

import io
import pathlib
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from opensculptboy import Character, export_glb
from opensculptboy.eyes import blink as B
from opensculptboy.eyes.lashes import (
    LOWER,
    UPPER,
    LashData,
    card_lashes,
    generic_texture,
    reference_lashes,
)
from opensculptboy.eyes.mesh import eyed_mesh
from opensculptboy.material import (
    LASH_WEIGHTS,
    EyeMaterial,
    LashMaterial,
    check_material,
)
from test.gltf_reader import GLB, evaluate_nodes, structure_errors


class TestLashCard(unittest.TestCase):
    def test_section(self):
        # the lashes come with the eyes
        self.assertIsNone(Character().lash_material())
        self.assertIsNone(Character(material={"lashes": {}}).lash_material())
        lashes = Character(material={"eyes": {}}).lash_material()
        self.assertEqual(lashes.source, "generic")
        self.assertTrue(lashes.lower)
        off = Character(material={"eyes": {}, "lashes": False})
        self.assertIsNone(off.lash_material())
        self.assertIsNotNone(off.eye_material())
        # the section keeps the exporters' own skin
        self.assertIsNone(
            Character(material={"eyes": {}, "lashes": {}}).skin_material()
        )
        reference = LashMaterial.from_dict({"mesh": "c.lashes.safetensors"})
        self.assertEqual(reference.source, "reference")
        with self.assertRaises(ValueError):
            LashMaterial.from_dict({"source": "reference"})
        check_material({"lashes": {"color": [0.02, 0.02, 0.02], "length_mm": 9}})
        for bad in (
            {"colour": [0, 0, 0]},
            {"source": "mink"},
            {"length_mm": 40},
            {"lower": "yes"},
            {"mesh": 3},
            {"weight": "heavy"},
            {"opacity": 0},
            [],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_material({"lashes": bad})
        card = Character(material={"eyes": {}, "lashes": {"curl": 0.3}})
        self.assertEqual(Character.from_dict(card.to_dict()), card)

    def test_weights(self):
        # the weight sets the generic lashes' density, lengths and opacity, fine by default;
        # a key of the section overrides its weight's value
        self.assertEqual(list(LASH_WEIGHTS), ["fine", "medium", "full"])
        fine = LashMaterial.from_dict({})
        self.assertEqual(fine.weight, "fine")
        for weight, values in LASH_WEIGHTS.items():
            lashes = LashMaterial.from_dict({"weight": weight})
            for key, value in values.items():
                self.assertEqual(getattr(lashes, key), value, (weight, key))
        own = LashMaterial.from_dict({"weight": "medium", "length_mm": 9.0})
        self.assertEqual((own.length_mm, own.density), (9.0, 1.5))
        # each weight in turn: more strands, longer, more opaque; the middle lies between
        order = [LashMaterial.from_dict({"weight": w}) for w in LASH_WEIGHTS]
        for key in ("density", "length_mm", "lower_length_mm", "opacity"):
            values = [getattr(lashes, key) for lashes in order]
            self.assertEqual(values, sorted(set(values)), key)
        cover = [
            (generic_texture(lashes)[..., 3] / 255.0)[:, :1024].mean()
            for lashes in order
        ]
        self.assertLess(cover[0] * 1.3, cover[1])
        self.assertLess(cover[1] * 1.15, cover[2])
        self.assertEqual(
            LashMaterial.from_dict({"weight": "full"}).extras()["weight"], "full"
        )

    def test_texture(self):
        image = generic_texture(LashMaterial())
        self.assertEqual(image.shape, (128, 2048, 4))
        alpha = image[..., 3] / 255.0
        upper, lower = alpha[:, :1024], alpha[:, 1024:]
        self.assertTrue(0.03 < upper.mean() < 0.5)
        # dense at the roots (the bottom rows), gone at the tips (the top row)
        self.assertGreater(upper[-8:].mean(), 5 * upper[:8].mean())
        self.assertLess(alpha[0].max(), 0.2)
        # the lower lashes are sparser and lighter
        self.assertLess(lower.mean(), 0.5 * upper.mean())
        self.assertLess(lower.max(), 0.65)


class TestLashMesh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Character().build_model(dtype=torch.float64)
        labels = list(cls.model.facial_action_labels)
        names = (
            "eyeBlinkLeft",
            "eyeLookOutLeft",
            "eyeBlinkRight",
            "eyeSquintLeft",
            "eyeLookDownLeft",
        )
        actions = torch.zeros(len(names) + 1, len(labels), dtype=torch.float64)
        for k, name in enumerate(names):
            actions[k + 1, labels.index(name)] = 1.0
        with torch.no_grad():
            out = cls.model(facial_actions=actions)
        cls.rest = out["rest_vertices"][0].numpy()
        cls.targets = {
            name: out["rest_vertices"][k + 1].numpy() - cls.rest
            for k, name in enumerate(names)
        }
        cls.eyed = eyed_mesh(
            cls.model,
            None,
            cls.rest,
            EyeMaterial(),
            "gltf",
            lashes=(LashMaterial(), None),
        )

    def _lashes(self):
        start = self.eyed.base_count
        out = []
        for p in self.eyed.parts:
            if p.name == "lashes":
                out.append((p, slice(start, start + len(p.rest))))
            start += len(p.rest)
        return out

    def test_parts(self):
        lashes = self._lashes()
        self.assertEqual(len(lashes), 4)  # each eye, each lid
        self.assertEqual({p.side for p, _ in lashes}, {"L", "R"})
        x = self.eyed.vertices(self.rest)
        for p, rows in lashes:
            np.testing.assert_allclose(x[rows], p.rest, atol=1e-12)
            # the roots lie at the lid margin, within 1.5 mm
            near = np.linalg.norm(
                p.rest[:, None] - self.rest[p.follow.rows][None], axis=2
            ).min(1)
            self.assertLess(near.min(), 0.0015)
        # the eyes mirror each other
        left = [p for p, _ in lashes if p.side == "L"]
        right = [p for p, _ in lashes if p.side == "R"]
        for a, b in zip(left, right):
            np.testing.assert_allclose(
                np.sort(a.rest[:, 0]), np.sort(-b.rest[:, 0]), atol=2e-3
            )

    def test_blink(self):
        d = self.targets["eyeBlinkLeft"]
        moved = self.eyed.vertices(self.rest + d)
        offsets = self.eyed.targets(d[None])[0]
        np.testing.assert_allclose(
            self.eyed.vertices(self.rest) + offsets, moved, atol=1e-12
        )
        for p, rows in self._lashes():
            shift = np.linalg.norm(offsets[rows], axis=1) * 1000
            upper = p.rest[:, 2].mean() > self.eyed.frames[p.side].centre[2]
            if p.side == "R":
                self.assertLess(shift.max(), 1e-6)  # the other eye stays open
            elif upper:
                # the upper lashes come down with the lid, and keep their shape: the
                # distances between points within 3 mm of each other change by less than
                # 1.5 mm for 95 % of the pairs, and by at most 3 mm where the lid folds
                self.assertGreater(np.median(shift), 5.0)
                a, b = p.rest, moved[rows]
                da = np.linalg.norm(a[:, None] - a[None], axis=2)
                db = np.linalg.norm(b[:, None] - b[None], axis=2)
                change = np.abs(da - db)[da < 0.003]
                self.assertLess(np.percentile(change, 95), 0.0015)
                self.assertLess(change.max(), 0.003)
            else:
                self.assertLess(np.median(shift), 1.0)

    def test_closed_eye(self):
        # a closed eye of the files' blinks shows its upper lashes along the lid line,
        # pointing forward and out of the lid, with either eye's blink, and half way
        from scipy.spatial import cKDTree

        from opensculptboy.export.body import vertex_normals

        skin_tris = self.eyed.triangles[: int(self.eyed.kept.sum())]
        skin_rows = np.unique(skin_tris)
        names = ["eyeBlinkLeft", "eyeBlinkRight"]
        shaped, blinks = B.blink_offsets(
            self.eyed, self.rest, np.stack([self.targets[n] for n in names]), names
        )
        full = B.blink_lash_offsets(
            self.eyed, blinks, self.rest, self.eyed.targets(shaped)
        )
        at_rest = self.eyed.vertices(self.rest)

        for action, side in (("eyeBlinkLeft", "L"), ("eyeBlinkRight", "R")):
            row, extra = blinks.rows[side], blinks.inbetween_rows[side]
            for weight in (0.5, 1.0):
                with self.subTest(action=action, weight=weight):
                    tent = 1.0 - abs(2.0 * weight - 1.0)
                    x = self.rest + weight * shaped[row] + tent * shaped[extra]
                    moved = at_rest + weight * full[row] + tent * full[extra]
                    normals = vertex_normals(x, skin_tris)
                    tree = cKDTree(x[skin_rows])
                    frame = self.eyed.frames[side]
                    for p, rows in self._lashes():
                        upper = p.rest[:, 2].mean() > frame.centre[2]
                        if p.side != side or not upper:
                            continue
                        local = frame.to_local(moved[rows])
                        tips, roots = p.uvs[:, 1] > 0.9, p.uvs[:, 1] < 0.1
                        ahead = np.median(local[tips, 2]) - np.median(local[roots, 2])
                        self.assertGreater(ahead, 0.15)  # out, in front of the lid
                        # the outer half of each lash lies outside the skin
                        outer = moved[rows][p.uvs[:, 1] > 0.5]
                        _, near = tree.query(outer)
                        near = skin_rows[near]
                        side_of = ((outer - x[near]) * normals[near]).sum(1)
                        self.assertGreater((side_of > 0).mean(), 0.95)

    def test_upper_lashes_in_front(self):
        # the upper lashes stay in front of the lids at every weight of a blink, along the
        # path with the in-between (GLB) and along the straight line (VRM), as they lie at
        # rest (0.3 mm out, or nearer where their roots lie nearer)
        names = ["eyeBlinkLeft", "eyeBlinkRight"]
        offsets = np.stack([self.targets[n] for n in names])
        weights = np.round(np.arange(0.05, 1.0001, 0.05), 2)
        for inbetween in (True, False):
            shaped, blinks = B.blink_offsets(
                self.eyed, self.rest, offsets, names, None, inbetween
            )
            full = B.blink_lash_offsets(
                self.eyed, blinks, self.rest, self.eyed.targets(shaped)
            )
            for side in ("L", "R"):
                lids, row = blinks.lids[side], blinks.rows[side]
                shape = blinks.shapes[side]
                extra = blinks.inbetween_rows.get(side)
                for p, rows in B.lash_rows(self.eyed, side):
                    if p.lid != UPPER:
                        continue
                    for t in weights:
                        a, b = B.played(t, inbetween)
                        lid = self.rest + b * shape.closed
                        lash = p.rest + b * full[row, rows]
                        if extra is not None:
                            lid = lid + a * (shape.inbetween + 0.5 * shape.closed)
                            lash = lash + a * (
                                full[extra, rows] + 0.5 * full[row, rows]
                            )
                        depth, _ = B.lash_penetration(
                            lids, lid, self.rest, lash, p.rest
                        )
                        with self.subTest(inbetween=inbetween, side=side, t=t):
                            self.assertLessEqual(depth, 0.02)

    def test_lashes_in_front_pushes_out(self):
        # a lash point that a closing lid would cover moves out along the skin's normal,
        # in the poses that play it at that weight
        names = ["eyeBlinkLeft", "eyeBlinkRight"]
        shaped, blinks = B.blink_offsets(
            self.eyed, self.rest, np.stack([self.targets[n] for n in names]), names
        )
        shape = blinks.shapes["L"]
        part = next(p for p, _ in B.lash_rows(self.eyed, "L") if p.lid == UPPER)
        lids = blinks.lids["L"]
        # every lash point pushed 2 mm into the eye in the closed pose: the push takes them
        # out of the lids again
        inward = -B._radial(lids.frame, part.rest) * 0.002
        closed, between = B.lashes_in_front(
            part,
            self.rest,
            (shape.closed, shape.inbetween),
            (inward, -0.5 * inward),
            True,
        )
        lid = self.rest + shape.closed
        depth, _ = B.lash_penetration(
            lids, lid, self.rest, part.rest + closed, part.rest
        )
        self.assertLessEqual(depth, 0.02)
        before, _ = B.lash_penetration(
            lids, lid, self.rest, part.rest + inward, part.rest
        )
        self.assertGreater(before, 0.5)
        # the half-closed pose is untouched at rest: nothing moves at weight 0
        self.assertEqual(B.played(0.0, True), (0.0, 0.0))
        self.assertEqual(B.played(0.5, True), (1.0, 0.0))
        self.assertEqual(B.played(1.0, True), (0.0, 1.0))
        self.assertEqual(B.played(0.3, False), (0.0, 0.3))

    def test_squint_and_look_down(self):
        # a squint and a look down move the upper lid without rolling its edge: the lashes
        # turn with the margin's frame alone (the blinks set their own turn); measured as
        # the angle between each lash's offset from its root and that offset at rest turned
        # by the margin's frame, over the points 2 mm or more from their roots (the lashes'
        # clearance of the skin may tilt a few of them)
        for action in ("eyeSquintLeft", "eyeLookDownLeft"):
            d = self.targets[action]
            offsets = self.eyed.targets(d[None])[0]
            for p, rows in self._lashes():
                if p.side != "L" or p.lid != UPPER:
                    continue
                f = p.follow
                x1 = (self.rest + d)[f.rows]
                turn = np.einsum(
                    "mij,mkj->mik",
                    f._frames(x1, f.tangent, f.across),
                    f._frames(f.rest, f.tangent, f.across),
                )
                v0 = p.rest - f.root @ f.rest
                v1 = p.rest + offsets[rows] - f.root @ x1
                w = np.einsum("mij,mj->mi", turn, v0)
                far = np.linalg.norm(v0, axis=1) > 0.002
                cos = (w * v1).sum(1) / (
                    np.linalg.norm(w, axis=1) * np.linalg.norm(v1, axis=1)
                )
                angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))[far]
                with self.subTest(action=action):
                    self.assertLess(np.median(angle), 0.5)

    def test_eye_turn(self):
        # a turn of the eye leaves the lashes on their lids (the lids barely move)
        offsets = self.eyed.targets(self.targets["eyeLookOutLeft"][None])[0]
        for p, rows in self._lashes():
            self.assertLess(np.linalg.norm(offsets[rows], axis=1).max(), 0.003)

    def test_skin_weights(self):
        weights = self.model.vertex_bone_weights.double().numpy()
        indices = self.model.vertex_bone_indices.numpy()
        w, i = self.eyed.skin_weights(weights, indices)
        head = list(self.model.bone_labels).index("head")
        # the lid margins of the file's mesh, as the eyed mesh finds them
        from opensculptboy.export.body import triangulated_faces, vertex_normals
        from opensculptboy.eyes.lashes import lid_margins
        from opensculptboy.eyes.mesh import eyeball_rows

        triangles, _ = triangulated_faces(self.model)
        on_eye = np.zeros(len(self.rest), bool)
        for r in eyeball_rows(self.model).values():
            on_eye[r] = True
        skin_tris = triangles[~on_eye[triangles].all(1)]
        used = np.zeros(len(self.rest), bool)
        used[skin_tris] = True
        normals = vertex_normals(self.rest, skin_tris)
        bones = int(indices.max()) + 1
        dense = np.zeros((len(self.rest), bones))
        np.add.at(
            dense,
            (np.repeat(np.arange(len(self.rest)), indices.shape[1]), indices.ravel()),
            weights.ravel(),
        )
        for p, rows in self._lashes():
            frame = self.eyed.frames[p.side]
            margins = lid_margins(frame, self.rest, normals, used & ~on_eye, skin_tris)
            np.testing.assert_allclose(w[rows].sum(1), 1.0, atol=1e-9)
            # each lash's root lies on its own lid's margin: the rows its weights blend
            upper = frame.to_local(p.rest)[:, 1].mean() > 0
            margin = margins[UPPER if upper else LOWER].rows
            roots = p.anchors[(p.blend > 1e-12).any(0)]
            self.assertTrue(np.isin(roots, margin).all())
            # the lashes take the weights of their roots on the margin (the strongest
            # bones, renormalised), mostly the head's
            mixed = p.blend @ dense[p.anchors]
            got = np.zeros_like(mixed)
            np.add.at(got, (np.arange(len(got))[:, None], i[rows]), w[rows])
            for k in range(len(mixed)):
                strongest = np.argsort(-mixed[k], kind="stable")[: w.shape[1]]
                want = np.zeros(bones)
                want[strongest] = mixed[k, strongest] / mixed[k, strongest].sum()
                np.testing.assert_allclose(got[k], want, atol=1e-9)
            self.assertGreater(float((w[rows] * (i[rows] == head)).sum(1).mean()), 0.5)


class TestReferenceLashes(unittest.TestCase):
    def test_stacked_textures_and_rest_frame(self):
        from opensculptboy.fit.handoff import HandoffPart

        def card(z, width):
            v = np.array(
                [[0.0, -0.07, z], [0.01, -0.07, z], [0.0, -0.075, z + 0.004]], float
            )
            uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            image = np.zeros((32, width, 4), np.uint8)
            image[..., 3] = 200
            return v, uv, image

        v1, uv1, im1 = card(1.52, 64)
        v2, uv2, im2 = card(1.50, 32)
        parts = [
            HandoffPart("upper", "lashes", v1, np.array([[0, 1, 2]]), uv1, im1),
            HandoffPart("lower", "lashes", v2, np.array([[0, 1, 2]]), uv2, im2),
            HandoffPart("skin", "skin", v1, np.array([[0, 1, 2]])),
        ]
        rng = np.random.default_rng(1)
        rest = rng.normal(size=(40, 3))
        angle = 0.1
        R = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        posed = rest @ R.T + [0.0, 0.01, 0.02]
        data, atlas, values = reference_lashes(parts, posed, rest, np.arange(40))
        self.assertEqual(atlas.shape, (96, 64, 4))  # 32 + the second scaled to 64 wide
        self.assertEqual(len(data.points), 6)
        self.assertEqual(list(data.lid), [UPPER] * 3 + [LOWER] * 3)
        # the points come back to the rest frame by the inverse motion
        np.testing.assert_allclose(
            data.points[:3], (v1 - [0.0, 0.01, 0.02]) @ R, atol=1e-9
        )
        # the first part's texture is the top third of the atlas (OBJ v from the bottom)
        np.testing.assert_allclose(
            data.uvs[:3, 1], [1.0, 1.0, 1.0 - 32 / 96], atol=1e-9
        )
        np.testing.assert_allclose(data.uvs[3:, 1], [2 / 3, 2 / 3, 0.0], atol=1e-9)
        self.assertEqual(values, {})
        self.assertIsNone(reference_lashes(parts[2:], posed, rest, np.arange(40)))

    def test_one_mesh_with_both_lids(self):
        # one lash part holds the upper and the lower strips of both eyes: each strip takes
        # the lid on its side of the horizontal line through its eye's centre, whatever the
        # part's median, so the lower lashes follow the lower lid
        from opensculptboy.fit.handoff import HandoffPart

        rng = np.random.default_rng(2)
        sphere = rng.normal(size=(60, 3))
        sphere = 0.012 * sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
        centres = {
            "L": np.array([0.032, -0.07, 1.6]),
            "R": np.array([-0.032, -0.07, 1.6]),
        }
        head = rng.normal(scale=0.05, size=(40, 3)) + [0.0, 0.0, 1.6]
        rest = np.concatenate([head, sphere + centres["L"], sphere + centres["R"]])
        eyes = {"L": np.arange(40, 100), "R": np.arange(100, 160)}

        def strip(centre, up, count):
            # a row of cards along a lid: roots on the lid, tips away from the eye
            x = np.linspace(-0.012, 0.012, count)
            root = np.stack(
                [
                    centre[0] + x,
                    np.full(count, centre[1] - 0.011),
                    np.full(count, centre[2] + up * 0.005),
                ],
                1,
            )
            tip = root + [0.0, -0.004, up * 0.006]
            v = np.concatenate([root, tip])
            k = np.arange(count - 1)
            t = np.concatenate(
                [
                    np.stack([k, k + 1, k + count], 1),
                    np.stack([k + 1, k + count + 1, k + count], 1),
                ]
            )
            return v, t

        pieces = [
            strip(centres["L"], 1.0, 24),
            strip(centres["L"], -1.0, 8),
            strip(centres["R"], 1.0, 24),
            strip(centres["R"], -1.0, 8),
        ]
        v, t, lid, count = [], [], [], 0
        for k, (pv, pt) in enumerate(pieces):
            v.append(pv)
            t.append(pt + count)
            lid += [UPPER if k % 2 == 0 else LOWER] * len(pv)
            count += len(pv)
        v, t = np.concatenate(v), np.concatenate(t)
        uv = np.zeros((len(v), 2))
        image = np.full((8, 8, 4), 255, np.uint8)
        parts = [HandoffPart("lashes", "lashes", v, t, uv, image)]
        data, _, _ = reference_lashes(parts, rest, rest, np.arange(40), eyes)
        np.testing.assert_array_equal(data.lid, lid)
        # without the eyes, each strip still takes its own lid
        data, _, _ = reference_lashes(parts, rest, rest, np.arange(40))
        np.testing.assert_array_equal(data.lid, lid)

    def test_file(self):
        data = LashData(
            np.zeros((3, 3)),
            np.tile([0.0, -1.0, 0.0], (3, 1)),
            np.zeros((3, 2)),
            [[0, 1, 2]],
            [0, 0, 0],
        )
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "c.lashes.safetensors"
            data.save(path)
            (pathlib.Path(folder) / "c.lashes.png").write_bytes(b"png")
            settings = LashMaterial.from_dict(
                {"mesh": path.name, "base_color": "c.lashes.png"}
            )
            back, texture = card_lashes(settings, folder)
            np.testing.assert_array_equal(back.triangles, data.triangles)
            self.assertEqual(texture, b"png")
            self.assertEqual(card_lashes(LashMaterial(), folder), (None, None))


class TestLashFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.model = Character().build_model(dtype=torch.float64)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _materials(self, glb):
        g = glb.json
        return {
            g["materials"][p["material"]]["name"]: p
            for p in g["meshes"][0]["primitives"]
        }

    def test_command_weight(self):
        # --lashes sets the generic lashes' weight, and "off" leaves them out
        import contextlib

        from opensculptboy.cli import main

        out = self.folder / "medium.glb"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(["export", str(out), "--eyes", "--lashes", "medium"]), 0
            )
        g = GLB(out).json
        extras = g["scenes"][0]["extras"]["opensculptboy"]["eyes"]["lashes"]
        self.assertEqual(extras["weight"], "medium")
        self.assertEqual(extras["opacity"], LASH_WEIGHTS["medium"]["opacity"])
        off = self.folder / "off.glb"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["export", str(off), "--eyes", "--lashes", "off"]), 0)
        self.assertNotIn("lashes", self._materials(GLB(off)))

    def test_glb(self):
        path = self.folder / "lashes.glb"
        card = Character(material={"eyes": {"cornea": False}})
        export_glb(path, card, self.model)
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        g = glb.json
        material = next(m for m in g["materials"] if m["name"] == "lashes")
        self.assertEqual(material["alphaMode"], "BLEND")
        self.assertTrue(material["doubleSided"])
        self.assertEqual(material["extras"]["opensculptboy_eye"]["role"], "lashes")
        texture = g["textures"][
            material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        ]
        image = Image.open(io.BytesIO(glb.image_bytes(texture["source"])))
        self.assertEqual(image.mode, "RGBA")
        extras = g["scenes"][0]["extras"]["opensculptboy"]["eyes"]["lashes"]
        self.assertEqual(extras["source"], "generic")
        # the lashes of the file close with the lid of its own eyeBlinkLeft target
        names = g["meshes"][0]["extras"]["targetNames"]
        prim = self._materials(glb)["lashes"]
        rows = np.unique(glb.accessor(prim["indices"]))
        open_ = evaluate_nodes(glb)
        weights = [0.0] * len(names)
        weights[names.index("eyeBlinkLeft")] = 1.0
        closed = evaluate_nodes(glb, weights=weights)
        node = next(iter(open_))
        a = open_[node][0].positions[rows]
        b = closed[node][0].positions[rows]
        moved = np.linalg.norm(b - a, axis=1)
        left = a[:, 0] > 0
        self.assertGreater(np.percentile(moved[left], 75), 0.005)
        self.assertLess(moved[~left].max(), 1e-6)
        # "lashes": false leaves them out
        bare = self.folder / "bare.glb"
        export_glb(
            bare,
            Character(material={"eyes": {"cornea": False}, "lashes": False}),
            self.model,
        )
        self.assertEqual(
            sorted(self._materials(GLB(bare))), ["conjunctiva", "eyes", "skin"]
        )

    def test_vrm(self):
        from opensculptboy.export.vrm import export_vrm

        for version in ("1.0", "0.x"):
            with self.subTest(version=version):
                path = self.folder / f"lashes{version}.vrm"
                export_vrm(
                    path,
                    Character(material={"eyes": {}}),
                    self.model,
                    version=version,
                    author="tests",
                    refine_weights=False,
                    budget="strict",
                )
                g = GLB(path).json
                material = next(m for m in g["materials"] if m["name"] == "lashes")
                self.assertEqual(material["alphaMode"], "BLEND")
                self.assertTrue(material["doubleSided"])


if __name__ == "__main__":
    unittest.main()

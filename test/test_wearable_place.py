# OpenSculptBoy
# Apache License, Version 2.0
"""
The place of a rigid piece (a prop moved in the studio, ``opensculptboy.wearables.place``) with a
synthetic prop, the test circlet of ``test.wearable_fixtures``: the maths of a place and of a
change of the carrying bone, the card field (schema 6, old cards still load), the dressed
piece, the GLB at rest and posed (the piece follows its new bone), the VRM files of both
versions (the piece on that bone alone), and a statue's ``apply-card`` (the card's places as
the config's ``prop_places``, which the starter's build applies). The page's twin
(``viewer/src/wearables/place.ts``) runs in node against the same numbers
(``viewer/test/props.mjs``).
"""

import copy
import json
import pathlib
import subprocess
import tempfile
import unittest

import numpy as np

from opensculptboy import Character, export_glb
from opensculptboy.wearables import card_body, dress, worn_pieces
from opensculptboy.wearables.export import Dressing
from opensculptboy.wearables.place import (
    apply_place,
    check_place,
    place_matrix,
    place_of,
    quaternion,
    rehold,
    rotation_matrix,
)
from test.gltf_reader import GLB, evaluate_nodes, node_world_matrices, structure_errors
from test.test_viewer_card_file import node_available
from test.wearable_fixtures import template_body, template_model, write_outfit

REPO = pathlib.Path(__file__).resolve().parents[1]
C = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # Anny's frame to glTF's
CARRIER = "wrist.R"
TURN = [0.0, 0.0, float(np.sin(np.radians(15))), float(np.cos(np.radians(15)))]
PLACE = {"bone": CARRIER, "rotation": TURN, "translation": [0.01, 0.02, -0.03]}


def _quat(axis, degrees):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.radians(degrees) / 2
    return [*(a * np.sin(h)), float(np.cos(h))]


def placed_card(card: Character, place=PLACE, pose=None) -> Character:
    data = card.to_dict()
    for outfit in data["outfits"]:
        for piece in outfit["pieces"]:
            if piece["piece"] == "circlet":
                piece["place"] = copy.deepcopy(place)
    data["schema"] = 6
    if pose:
        data["pose"] = pose
    return Character.from_dict(data)


class TestPlaceMaths(unittest.TestCase):
    def test_quaternion_round_trip(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            back = quaternion(rotation_matrix(q))
            self.assertAlmostEqual(abs(float(np.dot(back, q))), 1.0, places=12)

    def test_place_round_trip(self):
        head = np.array([0.2, -0.05, 1.1])
        M = place_matrix(PLACE, head)
        again = place_of(M, head, CARRIER, digits=12)
        np.testing.assert_allclose(again["rotation"], PLACE["rotation"], atol=1e-12)
        np.testing.assert_allclose(
            again["translation"], PLACE["translation"], atol=1e-12
        )
        # the head stays put when only the turn changes, and moves by the translation
        np.testing.assert_allclose(
            M[:3, :3] @ head + M[:3, 3], head + PLACE["translation"], atol=1e-12
        )

    def test_rehold_keeps_the_piece_in_the_world(self):
        rng = np.random.default_rng(5)

        def rigid():
            M = np.eye(4)
            M[:3, :3] = rotation_matrix(rng.normal(size=4))
            M[:3, 3] = rng.normal(size=3) * 0.2
            return M

        S_old, S_new = rigid(), rigid()
        h_old, h_new = rng.normal(size=3), rng.normal(size=3)
        X = rng.normal(size=(30, 3)) * 0.1
        before = apply_place(PLACE, X, X, h_old)[0] @ S_old[:3, :3].T + S_old[:3, 3]
        new = rehold(PLACE, "head", S_old, S_new, h_old, h_new, "root")
        self.assertEqual(new["bone"], "root")
        after = apply_place(new, X, X, h_new)[0] @ S_new[:3, :3].T + S_new[:3, 3]
        np.testing.assert_allclose(after, before, atol=1e-6)

    def test_bad_places(self):
        check_place(PLACE, "circlet")
        for bad in (
            {"rotation": TURN},
            {"bone": CARRIER, "rotation": [0, 0, 0, 2]},
            {"bone": CARRIER, "translation": [0, 0]},
            {"bone": CARRIER, "scale": 2},
            [CARRIER],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_place(bad, "circlet")


class TestPlacedPiece(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.card = write_outfit(cls.folder)
        cls.placed = placed_card(cls.card)
        cls.model = template_model()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_card_schema(self):
        data = self.placed.to_dict()
        self.assertEqual(data["schema"], 6)
        self.assertEqual(Character.from_dict(json.loads(json.dumps(data))), self.placed)
        # a card without a place keeps its schema, and an older schema cannot hold one
        self.assertEqual(self.card.to_dict()["schema"], 4)
        with self.assertRaisesRegex(ValueError, "schema 6"):
            Character.from_dict(dict(data, schema=5))
        bad = copy.deepcopy(data)
        bad["outfits"][0]["pieces"][1]["place"] = {
            "bone": CARRIER,
            "rotation": [1, 1, 0, 0],
        }
        with self.assertRaisesRegex(ValueError, "unit quaternion"):
            Character.from_dict(bad)
        # a loose piece holds a place as well
        loose = Character.from_dict(
            {
                "schema": 6,
                "wearables": [
                    {"id": "circlet", "file": "circlet.wearable.json", "place": PLACE}
                ],
            }
        )
        self.assertEqual(worn_pieces(loose, self.folder)[0].place, PLACE)

    def dressed(self, card):
        body = template_body()
        return body, {p.ref: p for p in dress(body, worn_pieces(card, self.folder))}

    def test_dressed_piece_moves_about_its_bone(self):
        body, plain = self.dressed(self.card)
        _, moved = self.dressed(self.placed)
        ref = "test_outfit/circlet"
        b = body.bone_labels.index(CARRIER)
        head = body.bone_poses[b][:3, 3]
        Q = rotation_matrix(TURN)
        expected = head + (plain[ref].vertices - head) @ Q.T + PLACE["translation"]
        np.testing.assert_allclose(moved[ref].vertices, expected, atol=1e-12)
        np.testing.assert_allclose(
            moved[ref].normals, plain[ref].normals @ Q.T, atol=1e-12
        )
        self.assertTrue(np.all(moved[ref].joints[:, 0] == b))
        self.assertTrue(np.all(moved[ref].weights[:, 0] == 1.0))
        self.assertEqual(moved[ref].extras()["place"], PLACE)
        # the other pieces stay as they were
        np.testing.assert_array_equal(
            moved["test_outfit/tunic"].vertices, plain["test_outfit/tunic"].vertices
        )

    def test_place_on_a_surface_piece_is_refused(self):
        data = self.placed.to_dict()
        data["outfits"][0]["pieces"][0]["place"] = PLACE
        with self.assertRaisesRegex(ValueError, "rigidly bound"):
            dress(template_body(), worn_pieces(Character.from_dict(data), self.folder))

    def _export(self, card, name):
        path = self.folder / name
        export_glb(path, card, self.model, detail_dir=self.folder, dressing=Dressing())
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        animation = "pose" if card.pose else None
        names = [n.get("name") for n in glb.json["nodes"]]
        node = names.index("test_outfit/circlet")
        V = evaluate_nodes(glb, animation)[node][0].positions
        world = node_world_matrices(glb, animation)
        return glb, V, world, names

    def test_glb_at_rest_and_posed(self):
        """At rest the file holds the moved piece; posed, the piece keeps its place in the
        frame of its new bone (it follows the right wrist, no longer the head)."""
        _, plain = self.dressed(self.card)
        _, moved = self.dressed(self.placed)
        _, V, world, names = self._export(self.placed, "placed.glb")
        expected = moved["test_outfit/circlet"].vertices @ C.T
        lift = np.median(V - expected, axis=0)
        np.testing.assert_allclose(V, expected + lift, atol=2e-5)
        pose = {
            "bones": {
                "lowerarm01.R": _quat([1, 0, 0], 40),
                "wrist.R": _quat([0, 1, 1], 35),
                "neck01": _quat([0, 0, 1], 25),
            }
        }
        _, Vp, world_p, _ = self._export(
            placed_card(self.card, pose=pose), "placed_posed.glb"
        )

        def local(Vs, W, bone):
            M = np.linalg.inv(W[names.index(bone)])
            return Vs @ M[:3, :3].T + M[:3, 3]

        np.testing.assert_allclose(
            local(Vp, world_p, CARRIER), local(V, world, CARRIER), atol=2e-5
        )
        self.assertGreater(
            np.abs(local(Vp, world_p, "head") - local(V, world, "head")).max(), 0.05
        )

    def test_vrm_files(self):
        from opensculptboy.export.vrm import export_vrm
        from test.test_vrm_schema import check_vrm0, check_vrm1

        body, moved = self.dressed(self.placed)
        piece = moved["test_outfit/circlet"]
        head = body.bone_poses[body.bone_labels.index(CARRIER)][:3, 3]
        reach = np.linalg.norm(piece.vertices - head, axis=1)
        for version, check in (("1.0", check_vrm1), ("0.x", check_vrm0)):
            with self.subTest(version):
                path = self.folder / f"placed_{version}.vrm"
                export_vrm(
                    path,
                    self.placed,
                    author="test",
                    version=version,
                    detail_dir=self.folder,
                )
                glb = GLB(path)
                self.assertEqual(structure_errors(glb), [])
                self.assertEqual(check(glb.json), [])
                names = [n.get("name") for n in glb.json["nodes"]]
                node = names.index("test_outfit/circlet")
                skin = glb.json["skins"][glb.json["nodes"][node]["skin"]]
                prim = glb.json["meshes"][glb.json["nodes"][node]["mesh"]][
                    "primitives"
                ][0]
                joints = glb.accessor(prim["attributes"]["JOINTS_0"])
                weights = glb.accessor(prim["attributes"]["WEIGHTS_0"])
                used = {
                    skin["joints"][int(j)]
                    for j, w in zip(joints.reshape(-1), weights.reshape(-1))
                    if w > 0
                }
                self.assertEqual({names[j] for j in used}, {CARRIER})
                # the piece keeps its reach from the wrist in the file's T-pose
                V = evaluate_nodes(glb)[node][0].positions
                W = node_world_matrices(glb)[names.index(CARRIER)]
                np.testing.assert_allclose(
                    np.linalg.norm(V - W[:3, 3], axis=1), reach, atol=2e-4
                )

    def test_body_sliders_keep_the_place_on_the_bone(self):
        """On a taller body the bone's head moves, and the piece moves with it."""
        tall = Character.from_dict(
            dict(self.placed.to_dict(), phenotype={"height": 1.0})
        )
        body = card_body(tall, self.model)
        moved = {p.ref: p for p in dress(body, worn_pieces(tall, self.folder))}
        plain = {
            p.ref: p
            for p in dress(
                body,
                worn_pieces(
                    Character.from_dict(
                        dict(self.card.to_dict(), phenotype={"height": 1.0})
                    ),
                    self.folder,
                ),
            )
        }
        head = body.bone_poses[body.bone_labels.index(CARRIER)][:3, 3]
        expected = apply_place(
            PLACE,
            plain["test_outfit/circlet"].vertices,
            plain["test_outfit/circlet"].normals,
            head,
        )[0]
        np.testing.assert_allclose(
            moved["test_outfit/circlet"].vertices, expected, atol=1e-12
        )


class TestStatueProps(unittest.TestCase):
    def test_apply_card_and_the_starter_rest(self):
        from opensculptboy.statues.card_config import apply_card, card_places
        from opensculptboy.statues.pipeline import StatueConfig
        from opensculptboy.statues.starter import placed_rest, prop_place

        card = {
            "schema": 6,
            "outfits": [
                {
                    "id": "statue",
                    "pieces": [
                        {
                            "piece": "sword",
                            "file": "sword.wearable.json",
                            "place": PLACE,
                        },
                        {"piece": "base", "file": "base.wearable.json"},
                    ],
                }
            ],
            "active_outfit": "statue",
        }
        self.assertEqual(card_places(card), {"sword": PLACE})
        config = {
            "format": "opensculptboy.statue/1",
            "name": "synthetic",
            "scan": "scan.stl",
            "prop_bones": {"Sword": "metacarpal4.R"},
        }
        out, report = apply_card(card, config, hands=())
        self.assertEqual(out["prop_places"], {"sword": PLACE})
        self.assertEqual(report["prop_places"], {"sword": CARRIER})
        loaded = StatueConfig.from_dict(out)
        self.assertEqual(prop_place(loaded, "Sword"), PLACE)
        self.assertIsNone(prop_place(loaded, "Base"))
        with self.assertRaises(ValueError):
            StatueConfig.from_dict(
                dict(
                    out,
                    prop_places={"sword": {"bone": CARRIER, "rotation": [2, 0, 0, 0]}},
                )
            )
        # the starter's build moves the prop's rest and binds it to the place's bone
        body = template_body()
        rest = (
            np.random.default_rng(1).normal(size=(40, 3)) * 0.05
            + body.bone_poses[body.bone_labels.index("head")][:3, 3]
        )
        moved, bone = placed_rest(rest, PLACE, body, body.bone_labels)
        self.assertEqual(body.bone_labels[bone], CARRIER)
        head = body.bone_poses[bone][:3, 3]
        np.testing.assert_allclose(
            moved, apply_place(PLACE, rest, rest, head)[0], atol=1e-12
        )


@unittest.skipUnless(
    node_available() and (REPO / "viewer" / "node_modules" / "three").exists(),
    "node 22 or later and the viewer's node packages are needed",
)
class TestPlaceNode(unittest.TestCase):
    def test_page_twin(self):
        """The page's place maths, its card round trip and its dress of a placed piece
        (viewer/test/props.mjs) against this module's numbers."""
        head = [0.02, -0.04, 1.3]
        X = np.random.default_rng(2).normal(size=(5, 3)) * 0.1
        expected = apply_place(PLACE, X, X, head)
        case = {
            "place": PLACE,
            "head": head,
            "points": X.tolist(),
            "vertices": expected[0].tolist(),
            "normals": expected[1].tolist(),
        }
        out = subprocess.run(
            ["node", "test/props.mjs", json.dumps(case)],
            cwd=REPO / "viewer",
            capture_output=True,
            text=True,
        )
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        print(out.stdout.strip())


if __name__ == "__main__":
    unittest.main()

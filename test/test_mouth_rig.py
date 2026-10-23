# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of ``opensculptboy.mouth``: the ``anny-mouth`` rig keeps anny's bones and mesh and adds
MakeHuman's jaw and tongue chain; anny's jawOpen splits exactly into a turn of the jaw and a
corrective; the GLB file of the rig plays that turn; the VRM files map the humanoid jaw and keep
the weights of every vertex that the mouth bones leave alone; cards on the ``anny`` rig keep
their files (``test.test_gltf_export.TestGltfReference`` guards them byte for byte).
"""

import functools
import pathlib
import tempfile
import unittest

import numpy as np
import torch

from opensculptboy import Character, export_glb
from opensculptboy.export.vrm import export_vrm
from opensculptboy.mouth import MOUTH_BONES, MOUTH_RIG
from opensculptboy.mouth.jaw import (
    CORRECTIVE,
    JAW_ANIMATION,
    JAW_OPEN,
    fit_jaw_open,
    jaw_weights,
    skinned_turn,
)
from opensculptboy.mouth.rig import read_rig_cache
from test.gltf_reader import GLB, evaluate

# Measured on the default body (docs/mouth.md): the jaw turns by 21.8 degrees about its head,
# and the turn carries 88.8 % of the squared length of anny's jawOpen.
MEASURED_ANGLE = 21.76
MEASURED_EXPLAINED = 0.888


@functools.lru_cache(maxsize=1)
def _models():
    anny = Character().build_model(dtype=torch.float64)
    mouth = Character(rig=MOUTH_RIG).build_model(dtype=torch.float64)
    return anny, mouth


@functools.lru_cache(maxsize=1)
def _exports():
    folder = pathlib.Path(tempfile.mkdtemp())
    files = {}
    for rig in ("anny", MOUTH_RIG):
        character = Character(rig=rig)
        name = rig.replace("-", "_")
        files[rig, "glb"] = folder / f"{name}.glb"
        export_glb(files[rig, "glb"], character)
        for version in ("1.0", "0.x"):
            files[rig, version] = folder / f"{name}_{version}.vrm"
            export_vrm(files[rig, version], character, version=version, author="test")
    return files


class TestRig(unittest.TestCase):
    def test_bones(self):
        anny, mouth = _models()
        labels, mouth_labels = list(anny.bone_labels), list(mouth.bone_labels)
        self.assertEqual(len(mouth_labels), len(labels) + len(MOUTH_BONES))
        self.assertEqual(set(mouth_labels) - set(labels), set(MOUTH_BONES))
        parent = {
            b: mouth_labels[p]
            for b, p in zip(mouth_labels, mouth.bone_parents)
            if p >= 0
        }
        self.assertEqual(parent["jaw"], "head")
        for bone in MOUTH_BONES[1:]:
            self.assertTrue(
                parent[bone] == "jaw" or parent[bone].startswith("tongue"), bone
            )

    def test_shared_bones_and_mesh_unchanged(self):
        anny, mouth = _models()
        a, m = anny(), mouth()
        self.assertEqual(
            float((a["rest_vertices"] - m["rest_vertices"]).abs().max()), 0.0
        )
        pick = [list(mouth.bone_labels).index(b) for b in anny.bone_labels]
        diff = (a["rest_bone_poses"][0] - m["rest_bone_poses"][0][pick]).abs().max()
        self.assertLess(float(diff), 1e-12)

    def test_only_the_head_gives_weight(self):
        anny, mouth = _models()

        def dense(model):
            w = model.vertex_bone_weights.double().numpy()
            i = model.vertex_bone_indices.numpy()
            out = np.zeros((len(w), model.bone_count))
            np.add.at(
                out, (np.repeat(np.arange(len(w)), w.shape[1]), i.ravel()), w.ravel()
            )
            return out, list(model.bone_labels)

        a, la = dense(anny)
        m, lm = dense(mouth)
        mouth_cols = [lm.index(b) for b in MOUTH_BONES]
        merged = m[:, [lm.index(b) for b in la]]
        merged[:, la.index("head")] += m[:, mouth_cols].sum(axis=1)
        np.testing.assert_allclose(merged, a, atol=1e-12)
        self.assertGreater(m[:, lm.index("jaw")].sum(), 100)
        self.assertTrue((m[:, mouth_cols].sum(axis=1) > 0).sum() > 1000)

    def test_cache_matches_anny(self):
        rows = read_rig_cache()
        self.assertEqual(rows["bone_labels"], list(MOUTH_BONES))
        self.assertEqual(
            rows["bone_orientation_blendshapes"].shape[1], len(MOUTH_BONES)
        )


class TestJawOpen(unittest.TestCase):
    def test_turn_and_corrective_give_jaw_open(self):
        _, mouth = _models()
        turn = fit_jaw_open(mouth)
        base = mouth()["rest_vertices"][0].numpy()
        corrected = base + turn.corrective
        rebuilt = turn.corrective + skinned_turn(
            corrected, jaw_weights(mouth), turn.rotation, turn.translation, turn.pivot
        )
        np.testing.assert_allclose(rebuilt, turn.target, atol=1e-12)
        stats = turn.stats()
        self.assertAlmostEqual(stats["angle_degrees"], MEASURED_ANGLE, delta=0.5)
        self.assertGreater(stats["explained"], MEASURED_EXPLAINED - 0.01)
        # the target opens the mouth: the chin goes down (-Z in Anny's frame)
        self.assertLess(turn.target[:, 2].min(), -0.01)


class TestFiles(unittest.TestCase):
    def test_glb_plays_the_turn(self):
        glb = GLB(_exports()[MOUTH_RIG, "glb"])
        g = glb.json
        self.assertEqual(len(g["skins"][0]["joints"]), 116)
        names = g["meshes"][0]["extras"]["targetNames"]
        self.assertIn(CORRECTIVE, names)
        self.assertIn(JAW_ANIMATION, [a["name"] for a in g["animations"]])
        mouth = g["scenes"][0]["extras"]["opensculptboy"]["mouth"]
        self.assertEqual(mouth["format"], "opensculptboy/mouth@1")
        opened = [0.0] * len(names)
        opened[names.index(JAW_OPEN)] = 1.0
        node = next(i for i, n in enumerate(g["nodes"]) if "skin" in n)
        by_target, _ = evaluate(glb, weights={node: opened})
        by_jaw, _ = evaluate(glb, animation=JAW_ANIMATION, frame=1)
        rest, _ = evaluate(glb)
        self.assertGreater(np.abs(by_target - rest).max(), 0.01)
        # float32 file: the jaw's turn and the corrective meet anny's jawOpen to 0.05 mm
        self.assertLess(np.abs(by_jaw - by_target).max(), 5e-5)

    def test_anny_rig_files_have_no_jaw(self):
        g = GLB(_exports()["anny", "glb"]).json
        self.assertNotIn(CORRECTIVE, g["meshes"][0]["extras"]["targetNames"])
        self.assertNotIn("mouth", g["scenes"][0]["extras"]["opensculptboy"])

    def test_vrm_humanoid_jaw(self):
        files = _exports()
        for version in ("1.0", "0.x"):
            with self.subTest(version=version):
                g = GLB(files[MOUTH_RIG, version]).json
                nodes = g["nodes"]
                if version == "1.0":
                    bones = g["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"]
                    jaw, head = bones["jaw"]["node"], bones["head"]["node"]
                else:
                    bones = {
                        b["bone"]: b["node"]
                        for b in g["extensions"]["VRM"]["humanoid"]["humanBones"]
                    }
                    jaw, head = bones["jaw"], bones["head"]
                self.assertEqual(nodes[jaw]["name"], "jaw")
                self.assertIn(jaw, nodes[head].get("children", []))
                anny = GLB(files["anny", version]).json
                self.assertNotIn(
                    "jaw",
                    anny["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"]
                    if version == "1.0"
                    else [
                        b["bone"]
                        for b in anny["extensions"]["VRM"]["humanoid"]["humanBones"]
                    ],
                )

    def test_vrm_weights_away_from_the_mouth(self):
        files = _exports()
        for version in ("1.0", "0.x"):
            with self.subTest(version=version):
                a, m = GLB(files["anny", version]), GLB(files[MOUTH_RIG, version])
                ra, rm = evaluate(a)[0], evaluate(m)[0]
                np.testing.assert_allclose(rm, ra, atol=1e-6)

                def named(glb):
                    g = glb.json
                    node = next(i for i, n in enumerate(g["nodes"]) if "skin" in n)
                    prim = g["meshes"][g["nodes"][node]["mesh"]]["primitives"][0]
                    joints = g["skins"][g["nodes"][node]["skin"]]["joints"]
                    names = np.array([g["nodes"][j]["name"] for j in joints])
                    ids = glb.accessor(prim["attributes"]["JOINTS_0"]).astype(int)
                    w = glb.accessor(prim["attributes"]["WEIGHTS_0"])
                    return names[ids], w

                na, wa = named(a)
                nm, wm = named(m)
                mouth = np.isin(nm, list(MOUTH_BONES)) & (wm > 0)
                away = ~mouth.any(axis=1)
                self.assertGreater(away.sum(), 0.9 * len(away))
                key = lambda n, w: sorted(  # noqa: E731
                    (str(b), round(float(x), 6)) for b, x in zip(n, w) if x > 0
                )
                for v in np.flatnonzero(away)[::97]:
                    self.assertEqual(key(nm[v], wm[v]), key(na[v], wa[v]), v)


if __name__ == "__main__":
    unittest.main()

# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the skin weights of the VRM export with the inverse bind: the data file of the
edits (its format and digest, and the model's weights when it is missing or made for another
model), the refinement for each body (opensculptboy.export.authoring.refine), which the
export runs by default, and its effect: the folds and collapsed triangles of the T-posed
bind, and the fists and hand signs against Anny's own posed mesh, on the default and fixture
bodies and three corners of the six default phenotypes.
"""

import time
import types
import unittest
import warnings
from unittest import mock

import numpy as np
import torch

from opensculptboy.export import tpose, vrm_tables
from opensculptboy.export.authoring import refine, skin_weights
from test import vrm_fixtures

VERSIONS = {"1.0": "constraint", "0.x": "merge"}
# Three corners of the six default phenotypes (in the order of skin_weights.PHENOTYPES:
# gender, age, muscle, weight, height, proportions), among them the Compact corner 010100,
# whose fingers fold most with the model's own weights. Their refinements take 9 to 12 s
# each, so the test checks them in VRM 1.0 alone; the default and fixture bodies in both
# versions.
CORNERS = ("corner 010100", "corner 100101", "corner 010101")
# The new folds and collapsed triangles of the inverse bind with the refined weights, as
# measured over the gate report of the refinement (October 2026): (version, body) -> the
# largest count the test allows (the measured folds and 2 more for the corners; none on the
# default and fixture bodies). With the model's weights the same bodies fold 8, 11, 25, 35 and
# 31 edges.
MEASURED_FOLDS = {
    ("1.0", "default"): 0,
    ("1.0", "fixture"): 0,
    ("0.x", "default"): 0,
    ("0.x", "fixture"): 0,
    ("1.0", "corner 010100"): 5 + 2,
    ("1.0", "corner 100101"): 7 + 2,
    ("1.0", "corner 010101"): 3 + 2,
}


def bodies_of(model) -> dict:
    """The bodies of the checks: (phenotype, rest vertices, rest bone poses)."""
    kwargs = vrm_fixtures.CHARACTER.model_kwargs()
    result = {
        "default": ({}, *tpose._rest_body(model, {})),
        "fixture": (
            dict(vrm_fixtures.CHARACTER.phenotype),
            *tpose._rest_body(
                model,
                dict(
                    phenotype_kwargs=kwargs["phenotype_kwargs"],
                    face_shape_kwargs=kwargs["face_shape_kwargs"],
                ),
            ),
        ),
    }
    phenotypes = skin_weights.bodies()
    for name in CORNERS:
        result[name] = (
            phenotypes[name],
            *tpose._rest_body(model, {"phenotype_kwargs": phenotypes[name]}),
        )
    return result


class TestVrmSkinWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = vrm_fixtures.model()
        cls.labels = list(cls.model.bone_labels)
        cls.mesh = skin_weights.Mesh(cls.model)
        cls.bodies = bodies_of(cls.model)
        cls.cases = [(v, b) for v, b in MEASURED_FOLDS]
        # The file's weights of each case: refined for the body (the export's default), and
        # the model's weights (refine_weights=False).
        cls.weights, cls.seconds = {}, {}
        for version, name in cls.cases:
            phenotype, rest, B = cls.bodies[name]
            for refined in (True, False):
                start = time.perf_counter()
                cls.weights[(version, name, refined)] = tpose.body_file_weights(
                    cls.model,
                    version,
                    VERSIONS[version],
                    "inverse",
                    rest,
                    B,
                    phenotype=phenotype,
                    refine_weights=refined,
                )
                cls.seconds[(version, name, refined)] = time.perf_counter() - start

    def body(self, name) -> skin_weights.Body:
        phenotype, rest, B = self.bodies[name]
        T = tpose.vrm_t_pose(self.model, torch.from_numpy(B)).numpy()
        return skin_weights.Body(name, phenotype, rest, B, T)

    def test_tpose_folds(self):
        # The inverse bind of the refined weights folds no edge (a bend over 120 degrees
        # where Anny's rest mesh bends 120 degrees or less) and collapses no triangle (below
        # a quarter of its rest area) on the default and fixture bodies, and few on the
        # corners, where the model's weights fold the webs of the fingers and the groin.
        for version, name in self.cases:
            body = self.body(name)
            counts = {}
            for refined in (True, False):
                bind = skin_weights.inverse_bind(
                    body, *self.weights[(version, name, refined)]
                )
                new, small = skin_weights.tpose_problems(self.mesh, body, bind)
                counts[refined] = (int(new.sum()), int(small.sum()))
            print(
                f"\nVRM {version}, {name}: new folds and collapsed triangles of the inverse "
                f"bind {counts[False]} with the model's weights, {counts[True]} refined in "
                f"{self.seconds[(version, name, True)]:.1f} s"
            )
            with self.subTest(version=version, body=name):
                bound = MEASURED_FOLDS[(version, name)]
                self.assertLessEqual(counts[True][0], bound)
                self.assertEqual(counts[True][1], 0)
                self.assertGreater(counts[False][0], counts[True][0])

    def test_hand_poses(self):
        # Fists and hand signs (skin_weights.HAND_POSES, and Anny's rest pose with every
        # finger joint curled by 75 and 90 degrees, skin_weights.CURLS): the file posed
        # through W_j T_j^-1 against Anny's own posed mesh. With the refined weights it folds
        # no more edges and collapses no more triangles than with the model's weights, and
        # the distances of the hand from Anny's mesh stay within 10 % of theirs, at the
        # largest and at the 99th percentile.
        for version, name in self.cases:
            twist = VERSIONS[version]
            plain = vrm_tables.vrm_skin_weights(self.model, twist)
            hand = skin_weights.hand_mask(
                self.model, skin_weights.dense_weights(*plain, len(self.labels))
            )
            body = self.body(name)
            before, after = (
                skin_weights.hand_report(
                    self.model,
                    self.mesh,
                    body,
                    *self.weights[(version, name, refined)],
                    hand,
                )
                for refined in (False, True)
            )
            for pose, b in before.items():
                a = after[pose]
                print(
                    f"\nVRM {version}, {name}, {pose}: hand max {b['max_mm']:.2f} -> "
                    f"{a['max_mm']:.2f} mm, 99th percentile {b['p99_mm']:.2f} -> "
                    f"{a['p99_mm']:.2f}; new folds {b['folds']} -> {a['folds']}, "
                    f"collapsed {b['collapsed']} -> {a['collapsed']}"
                )
                with self.subTest(version=version, body=name, pose=pose):
                    self.assertLessEqual(a["folds"], b["folds"])
                    self.assertLessEqual(a["collapsed"], b["collapsed"])
                    self.assertLessEqual(a["max_mm"], 1.1 * b["max_mm"])
                    self.assertLessEqual(a["p99_mm"], 1.1 * b["p99_mm"])

    def test_off_keeps_the_model_weights(self):
        # Without the refinement, the file's weights are those of vrm_skin_weights (the
        # eyelids moved to the head, the twist bones merged), truncated to 4 bones, exactly.
        for version, name in self.cases:
            twist = VERSIONS[version]
            expected = tpose.file_skin_weights(self.model, version, twist)
            with self.subTest(version=version, body=name):
                for a, b in zip(self.weights[(version, name, False)], expected):
                    np.testing.assert_array_equal(a, b)
                phenotype, rest, B = self.bodies[name]
                *_, outcome = refine.body_skin_weights(
                    self.model, twist, rest, B, phenotype, refine=False
                )
                self.assertEqual(outcome, "off")

    def test_deterministic(self):
        # The same body gives the same weights, bit for bit, in a new refinement.
        phenotype, rest, B = self.bodies["fixture"]
        cached = refine.body_skin_weights(self.model, "constraint", rest, B, phenotype)
        with mock.patch.object(refine, "_WEIGHTS", {}):
            again = refine.body_skin_weights(
                self.model, "constraint", rest, B, phenotype
            )
        for a, b in zip(cached, again):
            if isinstance(a, str):
                self.assertEqual(a, b)
            else:
                np.testing.assert_array_equal(a, b)
        self.assertIn(cached[2], ("refined", "data file"))

    def test_data_file(self):
        # The data file of the edits carries the digest of the mesh, rig and skin weights it
        # was made for, which models of either precision share. A model with other weights,
        # or a missing file, gives the model's weights, with a warning.
        from safetensors import safe_open

        with safe_open(str(vrm_tables.SKIN_WEIGHT_EDITS), framework="numpy") as f:
            metadata = f.metadata()
        digest = vrm_tables.skin_weight_digest(self.model)
        self.assertEqual(metadata["format"], vrm_tables.SKIN_WEIGHT_FORMAT)
        self.assertEqual(metadata["digest"], digest)
        attributes = (
            "bone_labels",
            "bone_parents",
            "faces",
            "face_texture_coordinate_indices",
            "vertex_bone_indices",
            "vertex_bone_weights",
            "base_mesh_vertex_indices",
            "template_vertices",
            "template_bone_heads",
        )
        stub = {name: getattr(self.model, name) for name in attributes}
        single = types.SimpleNamespace(
            **{**stub, "vertex_bone_weights": stub["vertex_bone_weights"].float()}
        )
        self.assertEqual(vrm_tables.skin_weight_digest(single), digest)
        data = vrm_tables.skin_weight_data(self.model)
        for twist in vrm_tables.TWIST_MODES:
            with self.subTest(twist=twist):
                rows, bones, logits = data[twist]
                self.assertGreater(len(rows), 1000)
                self.assertEqual(bones.shape, (len(rows), 4))
                self.assertEqual(logits.shape, (len(rows), 4, 28))
                rows, bones, values = vrm_tables.skin_weight_edits(self.model, twist)
                np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-6)
                self.assertTrue((values >= 0).all())
        weights = stub["vertex_bone_weights"].clone()
        weights[0] = weights[0].flip(0)
        other = types.SimpleNamespace(**{**stub, "vertex_bone_weights": weights})
        phenotype, rest, B = self.bodies["default"]
        missing = vrm_tables.DATA_DIR / "missing.safetensors"
        for case, model, path in (
            ("another model", other, vrm_tables.SKIN_WEIGHT_EDITS),
            ("missing file", self.model, missing),
        ):
            with (
                self.subTest(case=case),
                mock.patch.object(vrm_tables, "SKIN_WEIGHT_EDITS", path),
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                w, i, outcome = refine.body_skin_weights(model, "constraint", rest, B)
                self.assertEqual(outcome, "no data")
                self.assertTrue(
                    any("model's weights" in str(x.message) for x in caught), case
                )
                expected = vrm_tables.vrm_skin_weights(model, "constraint")
                np.testing.assert_array_equal(w, expected[0])
                np.testing.assert_array_equal(i, expected[1])


if __name__ == "__main__":
    unittest.main()

# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the ``sculpt`` topology (``opensculptboy.topology``, docs/sculpt-topology.md) with the
test-only placeholder patch composed in: the patch format and its validator, the data file and
its loader, the patched ModelData (anny's rows exact for every blend shape, facial action and
face shape; the new rows exactly their convex blends), the mesh (manifold and watertight where
anny is, mirror symmetric, UVs not flipped and covering the removed faces), the detail levels on
mixed faces, a GLB export of a sculpt card, and the converter between both meshes.
"""

import json
import pathlib
import tempfile
import unittest
from collections import Counter

import numpy as np
import torch

from opensculptboy import Character, export_glb
from opensculptboy.topology import (
    Patch,
    PatchError,
    base_quad_mesh,
    blend_rows,
    compose,
    read_sculpt_data,
    sculpt_data,
    sculpt_model_data,
    use_sculpt_data,
    validate_patch,
)
from opensculptboy.topology.patches import placeholder
from opensculptboy.topology.sculpt import (
    DEFAULT_PATH,
    MAKEHUMAN_VERTEX_COUNT,
    compose_patches,
)
from test.gltf_reader import GLB, evaluate, structure_errors


def placeholder_data():
    return compose(include=["placeholder"])


def directed_edges(triangles) -> Counter:
    t = np.asarray(triangles)
    pairs = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    return Counter(map(tuple, pairs.tolist()))


def boundary(edges: Counter) -> set:
    return {e for e in edges if (e[1], e[0]) not in edges}


def signed_areas(uv_triangles, uvs):
    a, b, c = (uvs[uv_triangles[:, i]] for i in range(3))
    return 0.5 * (
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )


class TestPatchFormat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = base_quad_mesh()
        cls.patch = placeholder.build(cls.base, **placeholder.OPTIONS)

    def test_placeholder_is_valid_and_symmetric(self):
        validate_patch(self.patch, self.base)
        self.assertEqual(len(self.patch.removed_faces), 2)
        self.assertEqual(
            set(self.base.mirror_faces[self.patch.removed_faces]),
            set(self.patch.removed_faces),
        )
        # one triangle pair per side
        self.assertEqual(
            int((self.patch.faces[:, 2] == self.patch.faces[:, 3]).sum()), 4
        )

    def test_makehuman_vertex_count(self):
        from anny.faces.authoring.base_mesh import base_mesh

        self.assertEqual(MAKEHUMAN_VERTEX_COUNT, len(base_mesh().V))
        self.assertLess(
            self.base.base_mesh_vertex_indices.max(), MAKEHUMAN_VERTEX_COUNT
        )

    def broken(self, **changes) -> Patch:
        fields = {
            k: getattr(self.patch, k)
            for k in (
                "name",
                "removed_faces",
                "anchors",
                "weights",
                "faces",
                "uv_faces",
                "uv_anchors",
                "uv_weights",
            )
        }
        fields = {
            k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in fields.items()
        }
        fields.update(changes)
        return Patch(**fields)

    def assert_refused(self, patch, message):
        with self.assertRaisesRegex(PatchError, message):
            validate_patch(patch, self.base)

    def test_validator_refuses_broken_patches(self):
        w = self.patch.weights.copy()
        w[0, 0] += 0.1
        self.assert_refused(self.broken(weights=w), "sum to 1")
        w = self.patch.weights.copy()
        w[0, 0] += 1.0
        w[0, 1] -= 1.0
        self.assert_refused(self.broken(weights=w), "non-negative")
        a = self.patch.anchors.copy()
        a[0, 0] = self.base.vertex_count
        self.assert_refused(self.broken(anchors=a), "out of range")
        f = self.patch.faces.copy()
        f[0] = f[0, [0, 3, 2, 1]]  # the inner quad flipped
        self.assert_refused(self.broken(faces=f), "close the hole|twice")
        self.assert_refused(
            self.broken(faces=self.patch.faces[1:], uv_faces=self.patch.uv_faces[1:]),
            "close the hole",
        )
        u = self.patch.uv_faces.copy()
        u[0] = u[0, [0, 3, 2, 1]]
        self.assert_refused(self.broken(uv_faces=u), "UV")
        # one side alone is not symmetric
        n, u = self.base.vertex_count, self.base.uv_count
        half = self.broken(
            removed_faces=self.patch.removed_faces[:1],
            anchors=self.patch.anchors[:4],
            weights=self.patch.weights[:4],
            faces=self.patch.faces[:6],
            uv_faces=self.patch.uv_faces[:6],
            uv_anchors=self.patch.uv_anchors[:4],
            uv_weights=self.patch.uv_weights[:4],
        )
        self.assertTrue((half.faces < n + 4).all() and (half.uv_faces < u + 4).all())
        self.assert_refused(half, "mirror")
        half.symmetric = False
        validate_patch(half, self.base)

    def test_a_face_claimed_twice_is_refused(self):
        with self.assertRaisesRegex(PatchError, "removed by both"):
            compose_patches([self.patch, self.patch], self.base)

    def test_blend_rows_are_canonical(self):
        anchors, weights = blend_rows([{5: 0.5, 2: 0.5, 7: 0.0}])
        np.testing.assert_array_equal(anchors[0], [2, 5, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(weights[0], [0.5, 0.5, 0, 0, 0, 0, 0, 0])
        with self.assertRaises(PatchError):
            blend_rows([{i: 1 / 9 for i in range(9)}])


class TestDataFile(unittest.TestCase):
    def test_shipped_file_loads_without_the_placeholder(self):
        data = read_sculpt_data(DEFAULT_PATH)
        self.assertNotIn("placeholder", [p["name"] for p in data.patches])
        self.assertEqual(data.base_digest, base_quad_mesh().digest)

    def test_write_read_round_trip_and_checks(self):
        data = placeholder_data()
        with tempfile.TemporaryDirectory() as tmp:
            path = data.write(pathlib.Path(tmp) / "sculpt.safetensors")
            first = path.read_bytes()
            again = placeholder_data().write(path)
            self.assertEqual(first, again.read_bytes(), "the build is deterministic")
            loaded = read_sculpt_data(path)
            self.assertEqual(loaded.digest, data.digest)
            self.assertEqual(loaded.labels, ["placeholder"])
            sidecar = path.with_suffix(".json")
            header = json.loads(sidecar.read_text())
            header["base_digest"] = "0" * 64
            sidecar.write_text(json.dumps(header))
            with self.assertRaisesRegex(ValueError, "sidecar|another version"):
                read_sculpt_data(path)
            header = json.loads(sidecar.read_text())
            header["sha256"] = "0" * 64
            sidecar.write_text(json.dumps(header))
            with self.assertRaisesRegex(ValueError, "does not match"):
                read_sculpt_data(path)

    def test_use_sculpt_data_overrides_the_file(self):
        data = placeholder_data()
        with use_sculpt_data(data):
            self.assertIs(sculpt_data(), data)
        self.assertIsNot(sculpt_data(), data)


class TestModelData(unittest.TestCase):
    """The patched ModelData against anny's quad ModelData with every blend shape."""

    @classmethod
    def setUpClass(cls):
        from anny.models import build_model_data
        from anny.models.model_data import RigConfig, TopologyConfig

        cls.data = placeholder_data()
        base = build_model_data(
            rig=RigConfig.from_string("anny"),
            topology=TopologyConfig.from_string("anny-quads"),
            local_changes="all",
            facial_actions="all",
            face_shapes="all",
        )
        out = sculpt_model_data(base, cls.data)
        n = len(base.template_vertices)
        idx = torch.as_tensor(cls.data.anchors)
        w = torch.as_tensor(cls.data.weights)
        labels = base.metadata.blendshape_labels
        cls.blocks = {x.split(":")[0] for x in labels}
        cls.results = {
            "rows": len(labels),
            "same_labels": out.metadata.blendshape_labels == labels,
            "blendshapes_kept": torch.equal(out.blendshapes[:, :n], base.blendshapes),
            "template_kept": torch.equal(
                out.template_vertices[:n], base.template_vertices
            ),
            "new_blendshapes": torch.equal(
                out.blendshapes[:, n:],
                sum(
                    base.blendshapes[:, idx[:, k]] * w[None, :, k, None]
                    for k in range(4)
                ),
            ),
            "new_template": float(
                (
                    out.template_vertices[n:]
                    - (w[..., None] * base.template_vertices[idx]).sum(1)
                )
                .abs()
                .max()
            ),
            "weights_kept": torch.equal(
                out.vertex_bone_weights[:n, : base.vertex_bone_weights.shape[1]],
                base.vertex_bone_weights,
            )
            and bool(
                (
                    out.vertex_bone_weights[:n, base.vertex_bone_weights.shape[1] :]
                    == 0
                ).all()
            ),
            "indices_kept": torch.equal(
                out.vertex_bone_indices[:n, : base.vertex_bone_indices.shape[1]],
                base.vertex_bone_indices,
            ),
            "new_weight_sums": float(
                (out.vertex_bone_weights[n:].sum(1) - 1).abs().max()
            ),
            "bones_kept": all(
                torch.equal(getattr(out, f), getattr(base, f))
                for f in (
                    "template_bone_heads",
                    "bone_heads_blendshapes",
                    "bone_template_orientation_matrices",
                    "bone_orientation_blendshapes",
                    "craniofacial_landmarks",
                    "craniofacial_landmarks_blendshapes",
                )
            ),
            "base_index": out.base_mesh_vertex_indices.tolist(),
            "base_base_index": base.base_mesh_vertex_indices.tolist(),
            "uvs_kept": torch.equal(
                out.texture_coordinates[: len(base.texture_coordinates)],
                base.texture_coordinates,
            ),
        }
        # the new skin weights: the convex blend of the anchors' weights
        dense = torch.zeros(
            (len(base.template_vertices), len(base.metadata.bone_labels)),
            dtype=torch.float64,
        )
        dense.scatter_add_(1, base.vertex_bone_indices, base.vertex_bone_weights)
        expected = (w[..., None] * dense[idx]).sum(1)
        got = torch.zeros(
            (cls.data.new_vertex_count, dense.shape[1]), dtype=torch.float64
        )
        got.scatter_add_(1, out.vertex_bone_indices[n:], out.vertex_bone_weights[n:])
        cls.results["new_skin_weights"] = float(
            (got - expected / expected.sum(1, keepdim=True)).abs().max()
        )
        cls.n = n
        del base, out

    def test_every_block_is_present(self):
        self.assertLessEqual(
            {"local_change", "facial_action", "face_shape"}, self.blocks
        )
        self.assertTrue(self.results["same_labels"])

    def test_anny_rows_are_exact(self):
        r = self.results
        self.assertTrue(
            r["blendshapes_kept"], "every blend shape row of anny's vertices"
        )
        self.assertTrue(r["template_kept"])
        self.assertTrue(r["weights_kept"] and r["indices_kept"])
        self.assertTrue(r["bones_kept"])
        self.assertTrue(r["uvs_kept"])
        self.assertEqual(r["base_index"][: self.n], r["base_base_index"])

    def test_new_rows_are_their_blends(self):
        r = self.results
        self.assertTrue(r["new_blendshapes"])
        self.assertLess(r["new_template"], 1e-15)
        self.assertLess(r["new_weight_sums"], 1e-12)
        self.assertLess(r["new_skin_weights"], 1e-12)
        m = self.data.new_vertex_count
        self.assertEqual(
            r["base_index"][self.n :],
            list(range(MAKEHUMAN_VERTEX_COUNT, MAKEHUMAN_VERTEX_COUNT + m)),
        )


class TestSculptModel(unittest.TestCase):
    """Anny models on sculpt and anny, and the patched mesh."""

    @classmethod
    def setUpClass(cls):
        import anny

        cls.data = placeholder_data()
        kwargs = dict(local_changes="all", facial_actions="all")
        cls.anny = anny.Anny(**kwargs).to(torch.float64)
        with use_sculpt_data(cls.data):
            cls.character = Character(
                topology="sculpt",
                phenotype={"age": 0.3, "weight": 0.7},
                local_changes={cls.anny.local_change_labels[4]: 0.6},
                facial_actions={"jawOpen": 0.4, "eyeBlinkLeft": 0.5},
            )
            cls.model = cls.character.build_model(dtype=torch.float64)
            cls.quads = Character(topology="sculpt-quads").build_model(
                dtype=torch.float64
            )
        cls.n = len(cls.anny.template_vertices)
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_outputs_match_anny_and_the_blends(self):
        import anny.poses

        pose = anny.poses.pose_parameters(self.anny, "walk")["pose_parameters"][3:4]
        kwargs = self.character.model_kwargs()
        with torch.no_grad():
            a = self.anny(pose_parameters=pose, **kwargs)
            s = self.model(pose_parameters=pose, **kwargs)
        n = self.n
        for key in ("rest_vertices", "vertices"):
            self.assertTrue(torch.equal(s[key][0, :n], a[key][0]), key)
        rest = a["rest_vertices"][0].numpy()
        blend = np.einsum("mk,mkd->md", self.data.weights, rest[self.data.anchors])
        np.testing.assert_allclose(
            s["rest_vertices"][0, n:].numpy(), blend, rtol=0, atol=1e-12
        )
        self.assertTrue(torch.equal(s["bone_poses"], a["bone_poses"]))

    def test_faces_follow_anny(self):
        faces = self.model.faces.numpy()
        anny_faces = self.anny.faces.numpy()
        self.assertEqual(faces.shape[1], 3)
        quads = self.quads.faces.numpy()
        kept = int(self.data.kept_faces().sum())
        self.assertEqual(len(quads), kept + len(self.data.faces))
        # sculpt triangulates the kept quads as anny does
        removed_tris = 2 * len(self.data.removed_faces)
        new_tris = 2 * len(self.data.faces) - int(
            (self.data.faces[:, 2] == self.data.faces[:, 3]).sum()
        )
        self.assertEqual(len(faces), len(anny_faces) - removed_tris + new_tris)
        anny_set = Counter(map(frozenset, anny_faces.tolist()))
        common = sum((Counter(map(frozenset, faces.tolist())) & anny_set).values())
        self.assertEqual(common, len(anny_faces) - removed_tris)

    def test_manifold_and_watertight_where_anny_is(self):
        a = directed_edges(self.anny.faces.numpy())
        s = directed_edges(self.model.faces.numpy())
        self.assertEqual(max(s.values()), 1, "consistent orientation, no doubled face")
        self.assertEqual(boundary(s), boundary(a), "the same open borders as anny")

    def test_mirror_symmetry(self):
        v = self.model.template_vertices.numpy()
        key = np.round(v * 1e7).astype(np.int64)
        index = {tuple(k): i for i, k in enumerate(key.tolist())}
        mirror = np.array([index[(-k[0], k[1], k[2])] for k in key.tolist()])
        faces = self.model.faces.numpy()
        mirrored = {frozenset(f) for f in mirror[faces].tolist()}
        self.assertEqual(mirrored, {frozenset(f) for f in faces.tolist()})

    def test_uvs_not_flipped_and_cover_the_removed_faces(self):
        base = base_quad_mesh()
        uvs = self.model.texture_coordinates.numpy()
        uv_faces = self.model.face_texture_coordinate_indices.numpy()
        new = uv_faces[-self.data_new_triangles() :]
        removed = base.uv_quads[self.data.removed_faces]
        removed_tris = np.concatenate([removed[:, [0, 1, 2]], removed[:, [0, 2, 3]]])
        a_new, a_old = signed_areas(new, uvs), signed_areas(removed_tris, base.uvs)
        for side in (1, -1):  # both sides, which the atlas may mirror
            sel_new = np.sign(a_new) == side
            self.assertTrue(np.all(a_new != 0))
            self.assertAlmostEqual(
                float(a_new[sel_new].sum()),
                float(a_old[np.sign(a_old) == side].sum()),
                places=12,
            )
        # every new UV inside the box of its removed face's UVs
        new_uvs = uvs[len(base.uvs) :]
        lo, hi = base.uvs[removed.ravel()].min(0), base.uvs[removed.ravel()].max(0)
        self.assertTrue(np.all((new_uvs >= lo - 1e-12) & (new_uvs <= hi + 1e-12)))

    def data_new_triangles(self) -> int:
        return 2 * len(self.data.faces) - int(
            (self.data.faces[:, 2] == self.data.faces[:, 3]).sum()
        )

    def test_glb_export_of_a_sculpt_card(self):
        path = pathlib.Path(self.tmp.name) / "sculpt.glb"
        with use_sculpt_data(self.data):
            export_glb(path, self.character, self.model)
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        vertices, source = evaluate(glb)
        with torch.no_grad():
            rest = self.model(**self.character.model_kwargs())["rest_vertices"][
                0
            ].numpy()
        from opensculptboy.export.gltf import C3

        expected = (rest @ C3.T)[source]
        expected[:, 1] -= expected[:, 1].min()
        self.assertLess(np.abs(vertices - expected).max(), 1e-5)
        # the file holds the new vertices and only referenced vertices
        self.assertTrue(
            set(range(self.n, self.n + self.data.new_vertex_count))
            <= set(source.tolist())
        )
        self.assertLessEqual(
            set(source.tolist()), set(np.unique(self.model.faces.numpy()).tolist())
        )
        extras = glb.json["scenes"][0]["extras"]
        self.assertIn("sculpt", json.dumps(extras))

    def test_detail_levels_on_sculpt(self):
        from opensculptboy.detail import DetailModel, detail_topology, level

        with use_sculpt_data(self.data):
            l0, l1, l2 = (level("sculpt", k) for k in (0, 1, 2))
            self.assertEqual(l0.vertex_count, self.n + self.data.new_vertex_count)
            tri = int((l0.quads[:, 2] == l0.quads[:, 3]).sum())
            self.assertEqual(len(l1.quads), 4 * (len(l0.quads) - tri) + 3 * tri)
            self.assertFalse((l1.quads[:, 2] == l1.quads[:, 3]).any())
            self.assertEqual(len(l2.quads), 4 * len(l1.quads))
            for lv in (l1, l2):
                op = lv.operators[-1]
                sums = np.bincount(op.rows, weights=op.values, minlength=op.shape[0])
                self.assertLess(np.abs(sums - 1).max(), 1e-12)
            # the levels' UVs are not flipped in the patched region
            kept = int(self.data.kept_faces().sum())
            for lv in (l1, l2):
                from opensculptboy.detail.subdivision import split_quads

                patched = lv.uv_quads[lv.base_faces >= kept]
                areas = signed_areas(split_quads(patched), lv.uvs)
                self.assertTrue(np.all(areas != 0))
                parents = np.repeat(
                    l0.uv_quads[lv.base_faces[lv.base_faces >= kept]], 2, 0
                )
                parent_sign = np.sign(signed_areas(parents[:, :3], l0.uvs))
                self.assertTrue(np.all(np.sign(areas) == parent_sign))
            for region in ("full", "head"):
                top = detail_topology(self.model, 1, region, "sculpt")
                t = directed_edges(top.triangles)
                self.assertEqual(max(t.values()), 1, region)
                ref = directed_edges(detail_topology(self.anny, 1, region).triangles)
                self.assertEqual(len(boundary(t)), len(boundary(ref)), region)
            detail = DetailModel(self.model, 1, None, "full", "sculpt")
            with torch.no_grad():
                out = detail(**self.character.model_kwargs())
            self.assertEqual(out["vertices"].shape[1], l1.vertex_count)
        # anny's levels stay anny's
        self.assertEqual(level("anny", 0).vertex_count, self.n)


class TestMixedCatmullClark(unittest.TestCase):
    def test_general_rules_on_quads_are_anny_s(self):
        from anny.utils.subdivision import catmull_clark
        from opensculptboy.detail.subdivision import _catmull_clark_general

        base = base_quad_mesh()
        a = catmull_clark(base.vertex_count, base.quads)
        g = _catmull_clark_general(base.vertex_count, base.quads)
        np.testing.assert_array_equal(a.quads, g.quads)
        np.testing.assert_array_equal(a.operator.cols, g.operator.cols)
        np.testing.assert_array_equal(a.operator.rows, g.operator.rows)
        np.testing.assert_allclose(
            a.operator.values, g.operator.values, rtol=0, atol=1e-15
        )

    def test_mixed_prism_against_the_textbook_rules(self):
        from opensculptboy.detail.subdivision import catmull_clark_mixed

        # a closed triangular prism: two triangles and three quads, outward
        rng = np.random.default_rng(1)
        v = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], float
        ) + 0.05 * rng.standard_normal((6, 3))
        faces = np.array(
            [[0, 2, 1, 1], [3, 4, 5, 5], [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]]
        )
        level = catmull_clark_mixed(6, faces)
        got = level.operator.apply(v)
        polys = [list(f[:3]) if f[2] == f[3] else list(f) for f in faces]
        fp = {i: v[p].mean(0) for i, p in enumerate(polys)}
        edges = {}
        for i, p in enumerate(polys):
            for k in range(len(p)):
                edges.setdefault(tuple(sorted((p[k], p[(k + 1) % len(p)]))), []).append(
                    i
                )
        ordered = sorted(edges)
        expected = []
        for x in range(6):
            adj_f = [i for i, p in enumerate(polys) if x in p]
            adj_e = [e for e in ordered if x in e]
            nval = len(adj_e)
            F = np.mean([fp[i] for i in adj_f], 0)
            R = np.mean([(v[a] + v[b]) / 2 for a, b in adj_e], 0)
            expected.append((F + 2 * R + (nval - 3) * v[x]) / nval)
        for e in ordered:
            f1, f2 = edges[e]
            expected.append((v[e[0]] + v[e[1]] + fp[f1] + fp[f2]) / 4)
        expected += [fp[i] for i in range(len(polys))]
        np.testing.assert_allclose(got, np.array(expected), rtol=0, atol=1e-14)
        self.assertEqual(len(level.quads), 3 * 2 + 4 * 3)
        # the subdivided prism is closed and consistently oriented
        q = level.quads
        tris = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
        e = directed_edges(tris)
        self.assertEqual(max(e.values()), 1)
        self.assertEqual(boundary(e), set())


class TestConvert(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = placeholder_data()

    def test_round_trip_per_level(self):
        from opensculptboy.topology.convert import LevelCorrespondence

        for subdiv in (0, 1, 2):
            with self.subTest(subdiv=subdiv):
                c = LevelCorrespondence(subdiv, data=self.data)
                rng = np.random.default_rng(subdiv)
                field = rng.standard_normal((len(c.anny_to_sculpt), 3))
                s = c.to_sculpt(field)
                shared = c.sculpt_to_anny >= 0
                np.testing.assert_array_equal(
                    s[shared], field[c.sculpt_to_anny[shared]]
                )
                back = c.to_anny(s)
                same = c.anny_to_sculpt >= 0
                np.testing.assert_array_equal(back[same], field[same])
                # only the patched regions differ
                self.assertLess((~shared).sum(), 200)
                self.assertLess((~same).sum(), 40)
                # a linear field transfers nearly exactly (the patches lie close to anny)
                lin = c.to_sculpt(c.anny_positions)
                self.assertLess(np.abs(lin - c.sculpt_positions)[~shared].max(), 1e-3)

    def test_identity_on_anny_vertices_at_level_0(self):
        from opensculptboy.topology.convert import to_anny, to_sculpt

        n = self.data.vertex_count
        field = np.arange(n, dtype=np.float64)
        s = to_sculpt(field, 0, data=self.data)
        np.testing.assert_array_equal(s[:n], field)
        np.testing.assert_array_equal(to_anny(s, 0, data=self.data), field)

    def test_offsets_files(self):
        from opensculptboy.detail import level, load_offsets, save_offsets
        from opensculptboy.topology.convert import convert_offsets

        with tempfile.TemporaryDirectory() as tmp, use_sculpt_data(self.data):
            tmp = pathlib.Path(tmp)
            offsets = np.random.default_rng(0).normal(
                0, 1e-3, (level("anny", 1).vertex_count, 3)
            )
            save_offsets(tmp / "a.safetensors", offsets, 1, "anny")
            convert_offsets(tmp / "a.safetensors", tmp / "s.safetensors", "sculpt")
            s = load_offsets(tmp / "s.safetensors", 1, "sculpt")
            self.assertEqual(len(s), level("sculpt", 1).vertex_count)
            convert_offsets(tmp / "s.safetensors", tmp / "b.safetensors", "anny")
            b = load_offsets(tmp / "b.safetensors", 1, "anny")
            from opensculptboy.topology.convert import LevelCorrespondence

            same = LevelCorrespondence(1, data=self.data).anny_to_sculpt >= 0
            np.testing.assert_allclose(b[same], offsets[same], rtol=0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()

# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of ``--fit-budget`` (``opensculptboy.export.fit_budget``, docs/vrm.md) with synthetic
meshes and the synthetic garments of ``test.wearable_fixtures``:

- the reduction (``opensculptboy.export.decimate``): a closed surface stays closed, open edges,
  corners, UV seams and material borders stay, groups keep their floors, locked vertices stay,
  morph target shapes join the cost, and the kept vertices are input vertices;
- the atlas (``opensculptboy.export.atlas``): texels copied exactly at the native size, the base
  colour factor and the cutoff baked in, OPAQUE joining MASK, shared vertices split, the
  densest tile shrinking first, flat normals where a member has no normal map;
- the report per piece (``opensculptboy.export.pieces``) in the summary and the budget messages;
- the fitted VRM files of both versions: within the budget, valid, the kept vertices with their
  weights and morph target offsets, and files within the budget left as they are;
- the quality measures and the normal re-bake (with Embree), and the command's options.
"""

import contextlib
import copy
import dataclasses
import functools
import io
import pathlib
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np

from opensculptboy.export import budget
from opensculptboy.export.atlas import atlas_spec, decode, pack
from opensculptboy.export.decimate import decimate, weld
from opensculptboy.export.document import png_bytes

_TMP = tempfile.TemporaryDirectory()


def uv_sphere(nu=48, nv=24, radius=0.1):
    """A sphere with a UV seam (the first and last columns at one position) and poles of
    many vertices: (positions, triangles, UVs, normals)."""
    u = np.linspace(0, 1, nu + 1)
    v = np.linspace(0, 1, nv + 1)
    U, V = np.meshgrid(u, v, indexing="ij")
    theta, phi = U * 2 * np.pi, V * np.pi
    P = np.stack(
        [
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi),
        ],
        -1,
    ).reshape(-1, 3)
    UV = np.stack([U, V], -1).reshape(-1, 2)
    T = []
    for i in range(nu):
        for j in range(nv):
            a, b, c, d = (
                i * (nv + 1) + j,
                (i + 1) * (nv + 1) + j,
                (i + 1) * (nv + 1) + j + 1,
                i * (nv + 1) + j + 1,
            )
            if j > 0:
                T.append([c, b, a])
            if j < nv - 1:
                T.append([d, c, a])
    N = P / np.linalg.norm(P, axis=1, keepdims=True)
    return P, np.array(T), UV, N


def grid(n=24, bump=0.0):
    """A square panel of n by n vertices (1 m) with an optional bump: positions, triangles."""
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    z = bump * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.02)
    P = np.stack([x, y, z], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    p = (i * n + j).ravel()
    T = np.concatenate(
        [np.stack([p, p + n, p + n + 1], 1), np.stack([p, p + n + 1, p + 1], 1)]
    )
    return P, T


def welded_edge_counts(P, T):
    w = weld(P)[T]
    e = np.sort(np.concatenate([w[:, [0, 1]], w[:, [1, 2]], w[:, [2, 0]]]), 1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return counts


class TestDecimate(unittest.TestCase):
    def test_closed_surface_stays_closed(self):
        P, T, UV, N = uv_sphere()
        r = decimate(P, T, 600, uvs=UV, normals=N)
        self.assertLessEqual(len(r.triangles), 601)
        self.assertGreater(len(r.triangles), 500)
        np.testing.assert_array_equal(welded_edge_counts(P, r.triangles), 2)
        # the kept vertices are input vertices, on the sphere
        self.assertTrue(np.allclose(np.linalg.norm(P[r.kept], axis=1), 0.1))
        # no UV triangle flips: every one keeps the orientation of the input's
        uv = UV[r.triangles]
        signed = (uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1]) - (
            uv[:, 2, 0] - uv[:, 0, 0]
        ) * (uv[:, 1, 1] - uv[:, 0, 1])
        self.assertTrue((signed < 0).all() or (signed > 0).all())
        # every triangle spans less than half the UV width: none crosses the seam
        self.assertLess(float(np.ptp(uv[..., 0], axis=1).max()), 0.5)
        self.assertLess(max(r.error.values()), 0.01)

    def test_open_edges_and_corners_stay(self):
        P, T = grid(30, bump=0.05)
        r = decimate(P, T, 200, uvs=P[:, :2])
        n = 30
        self.assertTrue(r.kept[[0, n - 1, n * (n - 1), n * n - 1]].all())
        counts = welded_edge_counts(P, r.triangles)
        self.assertTrue(set(np.unique(counts)) <= {1, 2})
        # the open edges are the border of the square: their length stays 4 m
        w = weld(P)[r.triangles]
        e = np.sort(np.concatenate([w[:, [0, 1]], w[:, [1, 2]], w[:, [2, 0]]]), 1)
        key, number = np.unique(e, axis=0, return_counts=True)
        border = key[number == 1]
        length = np.linalg.norm(P[border[:, 0]] - P[border[:, 1]], axis=1).sum()
        self.assertAlmostEqual(float(length), 4.0, places=9)

    def test_material_borders_floors_and_locks(self):
        P, T = grid(25)  # a column of vertices at x = 0.5
        centre = P[T].mean(1)
        labels = (centre[:, 0] > 0.5).astype(int)
        locked = np.zeros(len(P), bool)
        locked[[5, 6, 7]] = True
        whole = int((labels == 1).sum())
        r = decimate(
            P,
            T,
            300,
            labels=labels,
            groups=labels,
            floors={1: whole},
            locked=locked,
        )
        self.assertEqual(int((r.groups == 1).sum()), whole)
        self.assertTrue(r.kept[[5, 6, 7]].all())
        # the border between the labels stays on the line x = 0.5
        both = np.intersect1d(
            np.unique(r.triangles[r.labels == 0]), np.unique(r.triangles[r.labels == 1])
        )
        self.assertGreater(len(both), 2)
        np.testing.assert_allclose(P[both, 0], 0.5, atol=1e-12)
        np.testing.assert_array_equal(r.labels, labels[r.origin])

    def test_small_parts_stay(self):
        """Cards of two triangles (fur, feathers) keep their triangles."""
        P, T = [], []
        for k in range(50):
            o, b = np.array([k * 0.05, 0.0, 0.0]), len(P)
            P += [o, o + [0.01, 0, 0], o + [0.01, 0.03, 0], o + [0, 0.03, 0]]
            T += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
        G, GT = grid(12)
        P = np.concatenate([np.array(P), G + [0, 1, 0]])
        T = np.concatenate([np.array(T), GT + 200])
        r = decimate(P, T, 120)
        self.assertEqual(int((r.triangles < 200).all(1).sum()), 100)
        self.assertLess(len(r.triangles), len(T))

    def test_shapes_join_the_cost(self):
        P, T = grid(24)
        offset = np.zeros_like(P)
        offset[:, 2] = 0.1 * np.exp(
            -((P[:, 0] - 0.5) ** 2 + (P[:, 1] - 0.5) ** 2) / 0.01
        )
        near = np.linalg.norm(P[:, :2] - 0.5, axis=1) < 0.15
        flat = decimate(P, T, 150)
        shaped = decimate(P, T, 150, shapes=[offset])
        self.assertGreater(int(shaped.kept[near].sum()), int(flat.kept[near].sum()))


def _texture(width, height, seed, alpha=None):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
    pixels[..., 3] = 255 if alpha is None else alpha
    return pixels


def _quad(x0, uv_box):
    """Two triangles of a 0.1 m square at x0 whose UVs span ``uv_box``."""
    from opensculptboy.export.body import BodyMesh

    u0, v0, u1, v1 = uv_box
    P = np.array(
        [[x0, 0, 0], [x0 + 0.1, 0, 0], [x0 + 0.1, 0.1, 0], [x0, 0.1, 0]], float
    )
    uv = np.array([[u0, v1], [u1, v1], [u1, v0], [u0, v0]], float)
    return BodyMesh(
        positions=P,
        normals=np.tile([0, 0, 1.0], (4, 1)),
        uvs=uv,
        source=np.arange(4),
        triangles=np.array([[0, 1, 2], [0, 2, 3]]),
        triangle_faces=np.zeros(2, np.int64),
        joints=np.zeros((4, 4), np.int64),
        weights=np.tile([1.0, 0, 0, 0], (4, 1)).astype(np.float32),
        targets=[],
        dropped_weight=0.0,
    )


def _toy_spec(materials, images, meshes):
    """A minimal spec with the fields that the atlas uses."""
    from opensculptboy.export.vrm import SpecMesh, SpecPrimitive

    spec_meshes = []
    for name, body, prims in meshes:
        spec_meshes.append(
            SpecMesh(
                name,
                body,
                np.zeros(3),
                [SpecPrimitive(m, t) for m, t in prims],
                [],
            )
        )
    return SimpleNamespace(materials=materials, images=images, meshes=spec_meshes)


def _sample_atlas(spec, mesh, vertex):
    """The atlas texel under a vertex's UV (its texel centre)."""
    material = spec.materials[spec.meshes[mesh].primitives[0].material]
    pixels = decode(spec.images[material.base_texture][1])
    h, w = pixels.shape[:2]
    u, v = spec.meshes[mesh].body.uvs[vertex]
    return pixels[min(int(v * h), h - 1), min(int(u * w), w - 1)]


class TestAtlas(unittest.TestCase):
    def _materials(self, **kw):
        from opensculptboy.export.mtoon import ToonMaterial

        return ToonMaterial(**kw)

    def test_two_materials_share_a_page_texel_for_texel(self):
        a, b = _texture(64, 64, 1), _texture(32, 64, 2)
        images = [
            ("skin", png_bytes(_texture(16, 16, 3)[..., :3])),
            ("a", png_bytes(a)),
            ("b", png_bytes(b)),
        ]
        skin = self._materials(name="skin", base_texture=0)
        ma = self._materials(
            name="p/a:a", base_texture=1, shade_texture=1, double_sided=True
        )
        mb = self._materials(
            name="p/b:b", base_texture=2, shade_texture=2, double_sided=True
        )
        spec = _toy_spec(
            [skin, ma, mb],
            images,
            [
                ("body", _quad(0, (0, 0, 1, 1)), [(0, None)]),
                ("p/a", _quad(1, (0.25, 0.25, 0.75, 0.75)), [(1, None)]),
                ("p/b", _quad(2, (0, 0, 1, 1)), [(2, None)]),
            ],
        )
        before_a = a[16:48, 16:48]
        report = atlas_spec(spec, 2, 4096)
        self.assertEqual(len(spec.materials), 2)
        self.assertEqual(len(report.pages), 1)
        self.assertEqual(report.pages[0]["smallest_scale"], 1.0)
        atlas = spec.materials[spec.meshes[1].primitives[0].material]
        self.assertTrue(atlas.name.startswith("atlas0"))
        pixels = decode(spec.images[atlas.base_texture][1])
        h, w = pixels.shape[:2]
        uv = spec.meshes[1].body.uvs
        x0, y0 = int(round(uv[:, 0].min() * w)), int(round(uv[:, 1].min() * h))
        np.testing.assert_array_equal(
            pixels[y0 : y0 + 32, x0 : x0 + 32, :3], before_a[..., :3]
        )
        # the tiles start on multiples of 8 texels with 8 texels of margin
        self.assertEqual((x0 - 8) % 8, 0)
        self.assertEqual((y0 - 8) % 8, 0)
        self.assertEqual(spec.materials[0].name, "skin")

    def test_factor_cutoff_and_opaque_join(self):
        alpha = np.full((16, 16), 102, np.uint8)  # 0.4
        images = [
            ("a", png_bytes(_texture(16, 16, 4, alpha))),
            ("b", png_bytes(_texture(16, 16, 5))),
        ]
        ma = self._materials(
            name="p:a",
            base_texture=0,
            shade_texture=0,
            alpha_mode="MASK",
            alpha_cutoff=0.3,
            base_color=(0.5, 1, 1, 1),
            shade_color=(0.3, 0.6, 0.6),  # the shade keeps the ratio of the others
            double_sided=True,
        )
        mb = self._materials(
            name="p:b",
            base_texture=1,
            shade_texture=1,
            alpha_mode="OPAQUE",
            double_sided=True,
        )
        mc = self._materials(
            name="p:c", alpha_mode="MASK", alpha_cutoff=0.6, double_sided=True
        )
        spec = _toy_spec(
            [ma, mb, mc],
            images,
            [
                ("body", _quad(5, (0, 0, 1, 1)), []),
                ("p", _quad(0, (0, 0, 1, 1)), [(0, None)]),
                ("q", _quad(1, (0, 0, 1, 1)), [(1, None)]),
                ("r", _quad(2, (0, 0, 1, 1)), [(2, None)]),
            ],
        )
        spec.meshes[0].primitives = []
        report = atlas_spec(spec, 1, 1024)
        self.assertEqual(report.level, 2)
        self.assertEqual(len(spec.materials), 1)
        m = spec.materials[0]
        self.assertEqual(m.alpha_mode, "MASK")
        a = _sample_atlas(spec, 1, 3) / 255.0  # the corner at UV (0, 0): texel (0, 0)
        b = _sample_atlas(spec, 2, 3) / 255.0
        self.assertGreater(
            a[3], m.alpha_cutoff
        )  # 0.4 passed the cutoff 0.3, and still does
        self.assertEqual(b[3], 1.0)  # the opaque member
        original = _texture(16, 16, 4)[0, 0, 0] / 255.0
        lin = (
            (original / 12.92)
            if original <= 0.04045
            else ((original + 0.055) / 1.055) ** 2.4
        )
        expected = 0.5 * lin
        got = a[0]
        got = got / 12.92 if got <= 0.04045 else ((got + 0.055) / 1.055) ** 2.4
        self.assertAlmostEqual(got, expected, delta=0.01)

    def test_shared_vertices_split_and_primitives_join(self):
        from opensculptboy.export.mtoon import ToonMaterial

        images = [
            ("a", png_bytes(_texture(16, 16, 6))),
            ("b", png_bytes(_texture(16, 16, 7))),
        ]
        mats = [
            ToonMaterial(name="p:a", base_texture=0, shade_texture=0),
            ToonMaterial(name="p:b", base_texture=1, shade_texture=1),
        ]
        body = _quad(0, (0, 0, 1, 1))
        spec = _toy_spec(
            mats,
            images,
            [
                ("body", _quad(5, (0, 0, 1, 1)), []),
                ("p", body, [(0, body.triangles[:1]), (1, body.triangles[1:])]),
            ],
        )
        spec.meshes[0].primitives = []
        atlas_spec(spec, 1, 1024)
        mesh = spec.meshes[1]
        self.assertEqual(len(mesh.primitives), 1)
        self.assertEqual(len(mesh.body.positions), 6)  # the two shared corners split
        t = mesh.primitives[0].triangles
        uv0, uv1 = mesh.body.uvs[t[0]], mesh.body.uvs[t[1]]
        self.assertFalse(np.allclose(uv0.mean(0), uv1.mean(0), atol=0.05))

    def test_densest_tile_shrinks_first(self):
        from opensculptboy.export.mtoon import ToonMaterial

        images = [
            ("dense", png_bytes(_texture(256, 256, 8))),
            ("plain", png_bytes(_texture(128, 128, 9))),
            ("n", png_bytes(np.full((32, 32, 3), (128, 128, 255), np.uint8))),
        ]
        mats = [
            ToonMaterial(name="p:dense", base_texture=0, shade_texture=0),
            ToonMaterial(
                name="q:plain", base_texture=1, shade_texture=1, normal_texture=2
            ),
        ]
        small = _quad(0, (0, 0, 1, 1))
        small.positions = small.positions * 0.1  # 1 cm across: dense texels
        spec = _toy_spec(
            mats,
            images,
            [
                ("body", _quad(5, (0, 0, 1, 1)), []),
                ("p", small, [(0, None)]),
                ("q", _quad(1, (0, 0, 1, 1)), [(1, None)]),
            ],
        )
        spec.meshes[0].primitives = []
        report = atlas_spec(spec, 1, 256)
        page = report.pages[0]
        self.assertLessEqual(max(page["size"]), 256)
        self.assertLess(page["smallest_scale"], 1.0)
        m = spec.materials[0]
        normal = decode(spec.images[m.normal_texture][1])
        u, v = spec.meshes[1].body.uvs.mean(0)
        h, w = normal.shape[:2]
        np.testing.assert_array_equal(
            normal[int(v * h), int(u * w), :3], (128, 128, 255)
        )

    def test_pack(self):
        sizes = [(64, 64)] * 4 + [(128, 32)]
        placed = pack(sizes, 256, 128)
        self.assertIsNotNone(placed)
        boxes = [(x, y, x + w, y + h) for (x, y), (w, h) in zip(placed, sizes)]
        for i, a in enumerate(boxes):
            self.assertTrue(a[2] <= 256 and a[3] <= 128)
            for b in boxes[i + 1 :]:
                self.assertTrue(
                    a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
                )
        self.assertIsNone(pack([(300, 10)], 256, 256))


@functools.lru_cache(maxsize=None)
def _dressed(version):
    """(card, folder, spec) of the synthetic dressed character, with the cape's normal map."""
    from opensculptboy.export.vrm import vrm_spec
    from test.wearable_fixtures import template_model, write_outfit

    folder = pathlib.Path(_TMP.name) / "wear"
    card = write_outfit(folder, normal_map=True)
    spec = vrm_spec(
        card,
        template_model(),
        version=version,
        author="t",
        detail_dir=folder,
        refine_weights=False,
    )
    return card, folder, spec


def _small(version, **limits):
    key = "vrm1" if version == "1.0" else "vrm0"
    return mock.patch.dict(budget.BUDGETS, {key: dict(budget.BUDGETS[key], **limits)})


class TestPieces(unittest.TestCase):
    def test_report_and_counts(self):
        from opensculptboy.export.pieces import piece_report, spec_counts
        from opensculptboy.export.vrm import write_vrm

        _, _, spec = _dressed("1.0")
        report = piece_report(spec)
        names = [p["name"] for p in report]
        self.assertEqual(names[0], "body")
        for ref in ("test_outfit/tunic", "test_outfit/circlet", "test_outfit/cape"):
            self.assertIn(ref, names)
        cape = report[names.index("test_outfit/cape")]
        self.assertEqual([t["size"] for t in cape["textures"]], [[64, 64]])
        self.assertGreater(cape["joints"], 0)
        path = pathlib.Path(_TMP.name) / "pieces.vrm"
        summary = write_vrm(copy.deepcopy(spec), path, budget="off")
        self.assertEqual(summary["pieces"], report)
        counts = spec_counts(spec)
        for key in (
            "triangles",
            "joints",
            "materials",
            "max_image_size",
            "morph_targets",
        ):
            self.assertEqual(counts[key], summary["counts"][key], key)
        self.assertEqual(sum(p["triangles"] for p in report), counts["triangles"])

    def test_budget_messages_name_the_pieces(self):
        from opensculptboy.export.vrm import write_vrm

        _, _, spec = _dressed("1.0")
        path = pathlib.Path(_TMP.name) / "over.vrm"
        with _small("1.0", triangles=1000):
            with self.assertRaisesRegex(
                budget.BudgetError, "test_outfit/cape: .* triangles"
            ):
                write_vrm(copy.deepcopy(spec), path, budget="strict")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                write_vrm(copy.deepcopy(spec), path, budget="warn")
        text = [str(w.message) for w in caught]
        self.assertTrue(
            any("the pieces of the file" in t and "body:" in t for t in text)
        )


class TestFitBudget(unittest.TestCase):
    def fit(self, version, options=None, **limits):
        from opensculptboy.export.fit_budget import fit_spec

        _, _, spec = _dressed(version)
        spec = copy.deepcopy(spec)
        with _small(version, **limits):
            report = fit_spec(spec, options)
        return spec, report

    def test_files_of_both_versions_fit(self):
        from opensculptboy.export.vrm import write_vrm
        from test.gltf_reader import GLB, structure_errors
        from test.test_vrm_schema import check_vrm0, check_vrm1

        for version, check in (("1.0", check_vrm1), ("0.x", check_vrm0)):
            with self.subTest(version):
                spec, report = self.fit(version, triangles=12000, materials=2)
                path = pathlib.Path(_TMP.name) / f"fit_{version}.vrm"
                with _small(version, triangles=12000, materials=2):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", budget.BudgetWarning)
                        summary = write_vrm(spec, path, budget="strict")
                self.assertLessEqual(summary["counts"]["triangles"], 12000)
                self.assertLessEqual(summary["counts"]["materials"], 2)
                self.assertEqual(report["messages"], [])
                self.assertEqual(
                    [s["step"] for s in report["steps"]], ["reduce", "rebake", "atlas"]
                )
                glb = GLB(path)
                self.assertEqual(structure_errors(glb), [])
                self.assertEqual(check(glb.json), [])
                self.assertIn(
                    "budget_fit", glb.json["scenes"][0]["extras"]["opensculptboy"]
                )
                quality = report["quality"]
                self.assertLess(quality["silhouette_share_max"], 0.02)
                self.assertIn("test_outfit/cape", quality["pieces"])

    def test_kept_vertices_keep_their_data(self):
        _, _, full = _dressed("1.0")
        spec, report = self.fit("1.0", triangles=12000)
        reduced = report["steps"][0]["pieces"]
        self.assertLess(reduced["body"]["after"], reduced["body"]["before"])
        for m, mesh in enumerate(spec.meshes):
            old = full.meshes[m].body
            new = mesh.body
            self.assertEqual(full.meshes[m].name, mesh.name)
            # each kept vertex is a vertex of the full mesh, with its weights and offsets
            key = {
                tuple(np.round(p, 9)): k
                for k, p in enumerate(np.asarray(old.positions))
            }
            rows = np.array(
                [key[tuple(np.round(p, 9))] for p in np.asarray(new.positions)]
            )
            np.testing.assert_array_equal(new.joints, np.asarray(old.joints)[rows])
            np.testing.assert_array_equal(new.weights, np.asarray(old.weights)[rows])
            np.testing.assert_array_equal(new.normals, np.asarray(old.normals)[rows])
            for t_new, t_old in zip(new.targets, old.targets):
                np.testing.assert_array_equal(t_new, np.asarray(t_old)[rows])

    def test_fixed_triangles_and_priorities(self):
        from opensculptboy.export.fit_budget import FitOptions

        options = FitOptions.parse(
            ["test_outfit/tunic=0"],
            ["test_outfit/cape=100"],
            rebake=False,
            quality=False,
        )
        self.assertEqual(options.triangles, {"test_outfit/cape": 100})
        spec, report = self.fit("1.0", options, triangles=12000)
        pieces = report["steps"][0]["pieces"]
        self.assertLessEqual(pieces["test_outfit/cape"]["after"], 100)
        self.assertTrue(pieces["test_outfit/cape"]["fixed"])
        with self.assertRaises(ValueError):
            FitOptions.parse(["cape"])
        with self.assertRaises(ValueError):
            FitOptions.parse([], ["cape=-1"])

    def test_within_budget_stays_byte_identical(self):
        from opensculptboy.export.fit_budget import fit_spec
        from opensculptboy.export.vrm import write_vrm

        _, _, spec = _dressed("1.0")
        plain = pathlib.Path(_TMP.name) / "plain.vrm"
        fitted = pathlib.Path(_TMP.name) / "fitted.vrm"
        write_vrm(copy.deepcopy(spec), plain)
        twin = copy.deepcopy(spec)
        report = fit_spec(twin)
        self.assertEqual(report["steps"], [])
        write_vrm(twin, fitted)
        self.assertEqual(plain.read_bytes(), fitted.read_bytes())

    def test_unweighted_joints_leave_the_skin(self):
        from opensculptboy.export.vrm import write_vrm
        from test.gltf_reader import GLB, structure_errors

        _, _, full = _dressed("0.x")
        weighted = set()
        for mesh in full.meshes:
            j, w = np.asarray(mesh.body.joints), np.asarray(mesh.body.weights)
            weighted |= {full.skins[0].joints[k] for k in np.unique(j[w > 0])}
        spec, report = self.fit("0.x", joints=len(weighted))
        self.assertEqual(report["steps"][-1]["step"], "joints")
        self.assertEqual(set(spec.skins[0].joints), weighted)
        path = pathlib.Path(_TMP.name) / "joints.vrm"
        write_vrm(spec, path, budget="off")
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        # every vertex keeps its bones: the joints name the same nodes as before
        for m, mesh in enumerate(spec.meshes):
            new = np.asarray(spec.skins[0].joints)[np.asarray(mesh.body.joints)]
            old = np.asarray(full.skins[0].joints)[
                np.asarray(full.meshes[m].body.joints)
            ]
            w = np.asarray(mesh.body.weights) > 0
            np.testing.assert_array_equal(new[w], old[w])

    def test_thinning_takes_the_least_seen_parts(self):
        from opensculptboy.export.fit_budget import _thin_parts

        P, T = [], []
        for k in range(6):  # six cards of two triangles
            o, b = np.array([k * 0.05, 0.0, 0.0]), len(P)
            P += [o, o + [0.01, 0, 0], o + [0.01, 0.03, 0], o + [0, 0.03, 0]]
            T += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
        P, T = np.array(P), np.array(T)
        seen = np.repeat([5.0, 0.0, 3.0, 0.0, 9.0, 1.0], 2)
        groups = np.zeros(len(T), int)
        rows = np.arange(len(T))
        args = (P, T, groups, rows, seen, np.ones(len(P)), {0})
        keep = _thin_parts(*args, 0, hidden_only=True)
        np.testing.assert_array_equal(keep.reshape(-1, 2).all(1), [1, 0, 1, 0, 1, 1])
        # three cards: the two hidden ones, then the least seen
        keep = _thin_parts(*args, 5)
        np.testing.assert_array_equal(keep.reshape(-1, 2).all(1), [1, 0, 1, 0, 1, 0])
        # at most half of each material goes
        keep = _thin_parts(*args, 5, labels=np.repeat([0, 0, 0, 1, 1, 1], 2))
        np.testing.assert_array_equal(keep.reshape(-1, 2).all(1), [1, 0, 1, 0, 1, 1])
        self.assertTrue(
            _thin_parts(P, T, groups, rows, seen, np.ones(len(P)), set(), 5).all()
        )

    def test_textures_shrink_to_the_limit(self):
        spec, report = self.fit("0.x", texture_size=32)
        sizes = [s["to"] for s in report["steps"][-1]["images"]]
        self.assertTrue(all(max(s) <= 32 for s in sizes))
        self.assertLessEqual(report["after"]["counts"]["max_image_size"], 32)


def _has_embree() -> bool:
    try:
        import embreex  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_has_embree(), "the bake needs Embree and SciPy (viewer extra)")
class TestRebake(unittest.TestCase):
    def test_folds_flattened_by_the_reduction_come_back(self):
        from opensculptboy.export.rebake import rebake_material

        n = 64
        x, y = np.meshgrid(
            np.linspace(0, 0.4, n), np.linspace(0, 0.4, n), indexing="ij"
        )
        z = 0.004 * np.sin(x * 2 * np.pi / 0.025)
        P = np.stack([x, y, z], -1).reshape(-1, 3)
        dz = 0.004 * 2 * np.pi / 0.025 * np.cos(x * 2 * np.pi / 0.025)
        N = np.stack([-dz, np.zeros_like(dz), np.ones_like(dz)], -1).reshape(-1, 3)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        UV = np.stack([x / 0.4, y / 0.4], -1).reshape(-1, 2)
        i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
        p = (i * n + j).ravel()
        T = np.concatenate(
            [np.stack([p, p + n, p + n + 1], 1), np.stack([p, p + n + 1, p + 1], 1)]
        )
        r = decimate(P, T, 300, uvs=UV)
        image, stats = rebake_material(
            (P, N, UV, T), (P, N, UV, r.triangles), None, 1.0, (256, 256), 0.02
        )
        self.assertIsNotNone(image)
        self.assertGreater(stats["before_deg"], 2.0)
        self.assertLess(stats["after_deg"], 0.5 * stats["before_deg"])


class TestCommand(unittest.TestCase):
    def test_options_need_fit_budget_and_a_vrm_file(self):
        from opensculptboy.cli import main

        for argv in (
            ["export", "x.vrm", "--author", "a", "--fit-priority", "body=2"],
            ["export", "x.glb", "--fit-budget"],
            [
                "export",
                "x.vrm",
                "--author",
                "a",
                "--fit-budget",
                "--fit-triangles",
                "body",
            ],
        ):
            with self.subTest(argv):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as exit_,
                ):
                    main(argv)
                self.assertEqual(exit_.exception.code, 2)

    def test_command_prints_what_it_did(self):
        import json

        from opensculptboy.cli import main
        from opensculptboy.export import vrm

        card, folder, _ = _dressed("1.0")
        path = folder / "card.json"
        path.write_text(json.dumps(card.to_dict()))
        out = pathlib.Path(_TMP.name) / "cmd.vrm"
        stdout, stderr = io.StringIO(), io.StringIO()
        real = vrm.export_vrm

        def export(*args, **kwargs):
            kwargs["refine_weights"] = False
            kwargs["fit_budget"] = dataclasses.replace(
                kwargs["fit_budget"], quality=False, rebake=False
            )
            return real(*args, **kwargs)

        with (
            _small("1.0", triangles=15000),
            mock.patch.object(vrm, "export_vrm", export),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertEqual(
                main(
                    [
                        "export",
                        str(out),
                        "--character",
                        str(path),
                        "--author",
                        "a",
                        "--fit-budget",
                    ]
                ),
                0,
            )
        lines = [
            line for line in stderr.getvalue().splitlines() if line.startswith("fit: ")
        ]
        self.assertTrue(
            lines[0].startswith("fit: fit to the vrm1 budget"), stderr.getvalue()
        )
        self.assertTrue(any(line.startswith("fit: reduced: ") for line in lines))
        text = stdout.getvalue()  # after Warp's own lines, if any
        summary = json.loads(text[text.index("{") :])
        self.assertLessEqual(summary["counts"]["triangles"], 15000)
        self.assertIn("pieces", summary)


if __name__ == "__main__":
    unittest.main()

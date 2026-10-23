# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the ear-root patch of the ``sculpt`` topology
(``opensculptboy.topology.patches.ear_root``): the faces it removes and the band it adds (all
quads, closing the hole, manifold, mirror symmetric, the root line as a labelled loop), the new
vertices on anny's surface, the band's shape and faces on the default body and on other bodies
(phenotype corners, face shapes, the ear's local changes), its UVs, its Catmull-Clark levels
against anny's, and its rows in the patched model.
"""

import unittest

import numpy as np
import torch

from anny.utils.subdivision import catmull_clark
from opensculptboy.topology import base_quad_mesh, compose, sculpt_model
from opensculptboy.topology.patches import ear_root


def quad_edges(quads):
    q = np.asarray(quads)
    return np.concatenate([np.stack([q[:, i], q[:, (i + 1) % 4]], 1) for i in range(4)])


def edge_counts(quads):
    keys, counts = np.unique(np.sort(quad_edges(quads), 1), axis=0, return_counts=True)
    return {tuple(k): c for k, c in zip(keys.tolist(), counts.tolist())}


def corner_sines(points):
    """The sine of each corner angle of quads (F, 4, 3) about their diagonals' normal."""
    normal = np.cross(points[:, 2] - points[:, 0], points[:, 3] - points[:, 1])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    a = np.roll(points, 1, 1) - points
    b = np.roll(points, -1, 1) - points
    return (np.cross(b, a) * normal[:, None]).sum(-1) / (
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    )


class EarRootCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = base_quad_mesh()
        cls.data = compose(["ear_root"])
        cls.patch = ear_root.build(cls.base)
        cls.rest = ear_root.default_body(cls.base)
        cls.n = cls.base.vertex_count
        removed = cls.patch.removed_faces
        cls.keep = np.setdiff1d(np.arange(len(cls.base.quads)), removed)
        region = np.unique(
            np.concatenate(
                [cls.base.vertex_faces[x] for x in np.unique(cls.base.quads[removed])]
            )
        )
        cls.anny_triangles = ear_root.quad_triangles(
            cls.base.vertices, cls.base.quads[region]
        ).reshape(-1, 3)

    def positions(self, rest):
        return np.vstack([rest, self.patch.positions(self.base, rest)])

    def surface_distance(self, rest, samples=4):
        """Distances (m) from points of the band's triangles to anny's surface."""
        template = self.positions(self.base.vertices)
        tri = ear_root.quad_triangles(template, self.patch.faces).reshape(-1, 3)
        g = (
            np.array(
                [(i, j) for i in range(samples + 1) for j in range(samples + 1 - i)]
            )
            / samples
        )
        w = np.c_[1 - g.sum(1), g]
        pts = np.einsum("sk,tkd->tsd", w, self.positions(rest)[tri]).reshape(-1, 3)
        return np.concatenate(
            [
                ear_root._closest_on_triangles(
                    pts[c : c + 500], rest, self.anny_triangles
                )[2]
                for c in range(0, len(pts), 500)
            ]
        )


class TestBand(EarRootCase):
    def test_removed_faces(self):
        # per ear: the strip at the root (32 quads), one ring of the head with its notches and
        # one ring of the pinna, all on one side, all in one UV chart
        removed = self.patch.removed_faces
        self.assertEqual(len(removed), 2 * 102)
        x = self.base.vertices[self.base.quads[removed]][..., 0]
        self.assertTrue(((x > 0).all(1) | (x < 0).all(1)).all())
        self.assertEqual(((x > 0).all(1)).sum(), 102)
        pinna = ear_root.pinna_mask(self.base, self.rest)
        band = ear_root.root_band(self.base, pinna)
        self.assertEqual(len(band["strip"]), 32)
        self.base.face_uvs(band["faces"])

    def test_all_quads_labelled(self):
        faces = self.patch.faces
        self.assertFalse((faces[:, 2] == faces[:, 3]).any())
        self.assertEqual(self.patch.vertex_count, 2 * 409)
        self.assertEqual(len(faces), 2 * 439)
        labels = set(self.patch.labels_of_vertices())
        self.assertEqual(labels, {"ear_root", ear_root.ROOT_LABEL})
        self.assertEqual(set(self.patch.labels_of_faces()), {"ear_root"})
        self.assertEqual(ear_root.ROOT_LABEL, "ear_root/line")
        self.assertIn(ear_root.ROOT_LABEL, self.data.labels)

    def test_manifold_without_t_junctions(self):
        # every edge of the patched mesh has at most two faces, and its border is anny's
        quads = np.vstack([self.base.quads[self.keep], self.patch.faces])
        counts = edge_counts(quads)
        self.assertLessEqual(max(counts.values()), 2)
        border = {e for e, c in counts.items() if c == 1}
        anny_border = {e for e, c in edge_counts(self.base.quads).items() if c == 1}
        self.assertEqual(border, anny_border)
        # consistent orientation: no directed edge twice
        directed = quad_edges(quads)
        self.assertEqual(len(np.unique(directed, axis=0)), len(directed))

    def test_root_line(self):
        # a closed loop of 64 vertices per ear, each joined to the next by an edge of the band,
        # each with four edges
        positions = self.patch.positions(self.base, self.rest)
        lines = ear_root.root_lines(self.patch.labels_of_vertices(), positions)
        edges = edge_counts(self.patch.faces)
        for side, line in lines.items():
            self.assertEqual(len(line), 64, side)
            idx = self.n + line
            for a, b in zip(idx, np.roll(idx, -1)):
                self.assertIn(tuple(sorted((int(a), int(b)))), edges)
            valence = [sum(1 for e in edges if int(i) in e) for i in idx]
            self.assertEqual(set(valence), {4})
        self.assertTrue((positions[lines["l"], 0] > 0).all())

    def test_symmetric(self):
        # the validator checks the rows and faces; the positions mirror on the default body
        mirror = self.patch.mirrored(self.base)
        rows = {
            tuple((int(a), round(float(w), 12)) for a, w in zip(ra, rw) if w > 0): i
            for i, (ra, rw) in enumerate(zip(self.patch.anchors, self.patch.weights))
        }
        to = [
            rows[tuple((int(a), round(float(w), 12)) for a, w in zip(ra, rw) if w > 0)]
            for ra, rw in zip(mirror.anchors, mirror.weights)
        ]
        p = self.patch.positions(self.base, self.rest)
        np.testing.assert_allclose(p[to] * [-1, 1, 1], p, atol=1e-9)

    def test_convex_rows(self):
        w = self.patch.weights
        self.assertTrue((w >= 0).all())
        np.testing.assert_allclose(w.sum(1), 1.0, atol=1e-12)
        self.assertLessEqual(int((w > 0).sum(1).max()), 3)

    def test_uvs(self):
        # every UV quad keeps its orientation along both diagonals (convex), and the band's
        # UV area is the removed faces' area (no overlap with a closed boundary)
        uvs = np.vstack([self.base.uvs, self.patch.new_uvs(self.base)])
        q = uvs[self.patch.uv_faces]

        def area(a, b, c):
            return 0.5 * (
                (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
            )

        old = self.base.uvs[self.base.uv_quads[self.patch.removed_faces]]
        sign = np.sign(area(old[:, 0], old[:, 1], old[:, 2]))
        self.assertEqual(len(set(sign.tolist())), 1)
        for i, j, k in ((0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3)):
            self.assertGreater((area(q[:, i], q[:, j], q[:, k]) * sign[0]).min(), 0)
        new_area = np.abs(
            area(q[:, 0], q[:, 1], q[:, 2]) + area(q[:, 0], q[:, 2], q[:, 3])
        )
        old_area = np.abs(
            area(old[:, 0], old[:, 1], old[:, 2])
            + area(old[:, 0], old[:, 2], old[:, 3])
        )
        self.assertAlmostEqual(new_area.sum(), old_area.sum(), delta=1e-9)


class TestShape(EarRootCase):
    def test_vertices_on_anny_surface(self):
        p = self.patch.positions(self.base, self.rest)
        d = ear_root._closest_on_triangles(p, self.rest, self.anny_triangles)[2]
        self.assertLess(d.max(), 1e-9)

    def test_default_body(self):
        # measured: mean 0.055 mm, p95 0.24 mm, max 1.67 mm (the band's chords round anny's
        # creases at the root, folds of up to 100 degrees between its coarse quads); smallest
        # corner 10.6 degrees; shortest edge 0.54 mm
        d = self.surface_distance(self.rest) * 1e3
        self.assertLess(d.mean(), 0.07)
        self.assertLess(np.percentile(d, 95), 0.28)
        self.assertLess(d.max(), 2.0)
        q = self.positions(self.rest)[self.patch.faces]
        sines = corner_sines(q)
        self.assertGreater(np.degrees(np.arcsin(sines.min())), 9.0)
        lengths = np.linalg.norm(np.roll(q, -1, 1) - q, axis=-1)
        self.assertGreater(lengths.min(), 4e-4)

    def test_other_bodies(self):
        # phenotype corners, face shapes and the ear's local changes move the band through its
        # anchors; it follows anny's surface, and where anny's own quads at the root do not
        # fold over, the band's faces keep their orientation but for a few slivers (measured:
        # 2 of 1756 triangles with ear-flap-incr at 1, none on the other bodies here)
        model = ear_root_model()
        rng = np.random.default_rng(3)
        bodies = [
            {
                "phenotype_kwargs": {
                    k: float(rng.integers(0, 2)) for k in model.phenotype_labels
                }
            }
            for _ in range(2)
        ]
        bodies.append(
            {
                "face_shape_kwargs": {
                    k: float(rng.uniform(*model.face_shape_ranges[k]))
                    for k in model.face_shape_labels
                }
            }
        )
        for name, value in (
            ("ear-scale-incr", 1.0),
            ("ear-flap-incr", 1.0),
            ("ear-wing-incr", 1.0),
        ):
            bodies.append(
                {"local_changes_kwargs": {f"l-{name}": value, f"r-{name}": value}}
            )
        template = self.positions(self.base.vertices)
        tri = ear_root.quad_triangles(template, self.patch.faces).reshape(-1, 3)
        anny_tri = ear_root.quad_triangles(
            self.base.vertices, self.base.quads[self.patch.removed_faces]
        ).reshape(-1, 3)

        def normals(v, t):
            n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
            return n / np.linalg.norm(n, axis=1, keepdims=True)

        n0 = normals(self.positions(self.rest), tri)
        a0 = normals(self.rest, anny_tri)
        for kwargs in bodies:
            with self.subTest(body=sorted(kwargs)[0]), torch.no_grad():
                rest = model(**kwargs)["rest_vertices"][0].numpy()
                d = self.surface_distance(rest) * 1e3
                self.assertLess(np.percentile(d, 95), 0.4)
                self.assertLess(d.max(), 3.0)
                if ((normals(rest, anny_tri) * a0).sum(1) > 0).all():
                    flipped = (normals(self.positions(rest), tri) * n0).sum(1) < 0
                    self.assertLessEqual(int(flipped.sum()), 2)


_MODEL = None


def ear_root_model():
    """anny-quads with every phenotype, the ear's local changes and every face shape."""
    global _MODEL
    if _MODEL is None:
        import anny

        _MODEL = anny.Anny(
            topology="anny-quads",
            phenotypes="all",
            local_changes=[
                f"{s}-ear-{k}"
                for s in "lr"
                for k in ("scale-incr", "flap-incr", "wing-incr")
            ],
            face_shapes="all",
            facial_actions="none",
        ).to(dtype=torch.float64)
    return _MODEL


class TestLevels(EarRootCase):
    def test_level_two_no_sharper_than_anny(self):
        # Catmull-Clark level 2 of the patched quad mesh: the folds between faces in the band
        # stay below anny's own around the root (measured p99 15.4 degrees against anny's
        # 21.3, max 30.1 against 34.9)
        def level2(n, quads, v):
            parents = np.arange(len(quads))
            for _ in range(2):
                step = catmull_clark(n, quads)
                v = step.operator.apply(v)
                quads, n = step.quads, step.operator.shape[0]
                parents = parents[step.parent_faces]
            return v, quads, parents

        def folds(v, quads, mask):
            nrm = np.cross(
                v[quads[:, 2]] - v[quads[:, 0]], v[quads[:, 3]] - v[quads[:, 1]]
            )
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
            e = np.sort(quad_edges(quads), 1)
            f = np.tile(np.arange(len(quads)), 4)
            order = np.lexsort(e.T[::-1])
            e, f = e[order], f[order]
            same = (e[1:] == e[:-1]).all(1)
            a, b = f[:-1][same], f[1:][same]
            sel = mask[a] | mask[b]
            cos = np.clip((nrm[a[sel]] * nrm[b[sel]]).sum(1), -1, 1)
            return np.degrees(np.arccos(cos))

        quads = np.vstack([self.base.quads[self.keep], self.patch.faces])
        new = np.r_[
            np.zeros(len(self.keep), bool), np.ones(len(self.patch.faces), bool)
        ]
        v, q, parents = level2(
            self.n + self.patch.vertex_count, quads, self.positions(self.rest)
        )
        ours = folds(v, q, new[parents])
        old = np.isin(np.arange(len(self.base.quads)), self.patch.removed_faces)
        va, qa, pa = level2(self.n, self.base.quads, self.rest)
        anny = folds(va, qa, old[pa])
        self.assertLess(np.percentile(ours, 99), np.percentile(anny, 99))
        self.assertLess(ours.max(), anny.max())


class TestModel(EarRootCase):
    def test_rows_in_the_patched_model(self):
        # the patched model's new vertices are the patch's blends of anny's rest vertices
        model = sculpt_model(topology="sculpt-quads", data=self.data).to(
            dtype=torch.float64
        )
        with torch.no_grad():
            rest = model(phenotype_kwargs={})["rest_vertices"][0].numpy()
        np.testing.assert_allclose(rest[: self.n], self.rest, atol=1e-6)
        mask = np.array([lab.startswith("ear_root") for lab in self.data.labels])
        self.assertTrue(mask.any())
        np.testing.assert_allclose(
            rest[self.n :], self.patch.positions(self.base, self.rest), atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()

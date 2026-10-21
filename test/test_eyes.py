# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the eyes of ``opensculptboy.eyes`` and their use: the card's ``material.eyes``
section and its validation, the surfaces of the eyeball, the cornea and the caruncle, the eye
texture's layout and the carried-over baked eyes, the eyed mesh's linear maps (rest, morph
targets, rigid skin weights), the caruncle over the pocket in the inner corner of the eye, the
GLB file (materials, extensions, the eyes against the file's skinning and morph targets), the
VRM files (MToon with the matcap, within budget), the Blender helper and the command's
``--eyes``. The files of cards without the section stay as they were (``TestGltfReference`` in
``test.test_gltf_export``).
"""

import json
import pathlib
import tempfile
import unittest
import warnings

import numpy as np
import torch

from opensculptboy import Character, export_glb
from opensculptboy.eyes import geometry
from opensculptboy.eyes import texture as tex
from opensculptboy.eyes.mesh import affine_operator, eye_texture, eyed_mesh
from opensculptboy.material import EyeMaterial, check_material
from test.gltf_reader import GLB, evaluate_nodes, structure_errors


def _kabsch_residual(a: np.ndarray, b: np.ndarray) -> float:
    """The largest distance left after the best rigid map of ``a`` onto ``b``."""
    ca, cb = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(ca.T @ cb)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = (u @ np.diag([1.0, 1.0, d]) @ vt).T
    return float(np.linalg.norm(ca @ rotation.T - cb, axis=1).max())


def synthetic_baked_eyes(
    model, eyed, centre=(0.03, -0.02), pupil_mm=2.8, iris_mm=11.2, pale_ring=False
):
    """A baked eye texture on anny's atlas (the box of opensculptboy.texture.eyes): a grey
    iris with a wide pupil, off the gaze by ``centre`` (eyeball radii), as a game's eye
    texture might come, with ``pale_ring`` a pale outer band of the iris (1.2 mm) and a
    darker limbal ring outside it (0.5 mm); returns the PNG bytes and the eyeball radius used
    (mm)."""
    from opensculptboy.export.document import png_bytes
    from opensculptboy.texture.eyes import eye_uv_box, to_eye_uvs

    box = eye_uv_box(model)
    w, h = tex.EYE_TEXTURE_SIZE
    image = np.full((h, w, 3), 0.5)
    radius = eyed.radii_mm()["L"]
    for points, triangles, uvs in eyed.eyeballs.values():
        corners = triangles.reshape(-1)
        uv = to_eye_uvs(uvs.reshape(-1, 2), box)
        px = np.stack([uv[:, 0] * w - 0.5, (1 - uv[:, 1]) * h - 0.5], 1)
        d = points[corners] / np.linalg.norm(points[corners], axis=1, keepdims=True)
        values, covered = tex.raster(
            px, np.arange(len(corners)).reshape(-1, 3), d, (w, h)
        )
        r = np.hypot(values[..., 0] - centre[0], values[..., 1] - centre[1]) * radius
        sclera = np.array([0.7, 0.68, 0.66])
        if pale_ring:
            band = r < iris_mm / 2 + 0.5
            sclera = np.where(band[..., None], 0.3, sclera)
            outer = np.where((r < iris_mm / 2)[..., None], 0.55, sclera)
            iris = np.where((r < iris_mm / 2 - 1.2)[..., None], 0.12, outer)
        else:
            iris = np.where((r < iris_mm / 2)[..., None], 0.12, sclera)
        colour = np.where((r < pupil_mm / 2)[..., None], 0.01, iris)
        colour = np.where((values[..., 2] > 0)[..., None], colour, [0.7, 0.68, 0.66])
        image[covered] = colour[covered]
    return png_bytes(tex.srgb(image)), radius


class TestEyeCard(unittest.TestCase):
    def test_round_trip_and_defaults(self):
        card = Character(name="eyes", material={"eyes": {"pupil_mm": 3.0}})
        data = json.loads(json.dumps(card.to_dict()))
        self.assertEqual(data["schema"], 3)
        self.assertEqual(Character.from_dict(data), card)
        eye = card.eye_material()
        self.assertEqual(eye.pupil_mm, 3.0)
        self.assertEqual(eye.iris_mm, "auto")  # the baked size, else 11.7
        self.assertEqual(eye.sized().iris_mm, 11.7)
        self.assertEqual(
            EyeMaterial(iris_mm="baked")
            .sized({"iris_mm": 11.0, "pupil_mm": 4.5})
            .pupil_mm,
            4.5,
        )
        self.assertTrue(eye.cornea and eye.caruncle)
        self.assertAlmostEqual(eye.cornea_ior, 1.376)
        # the eyes alone keep the exporters' own skin; with skin values, both
        self.assertIsNone(card.skin_material())
        both = Character(material={"ior": 1.4, "eyes": {}})
        self.assertEqual(both.skin_material().ior, 1.4)
        self.assertIsNotNone(both.skin_material().eyes)
        self.assertIsNone(Character().eye_material())

    def test_invalid_eyes_are_refused(self):
        for eyes in (
            {"pupil_mm": 0.5},
            {"pupil_mm": 11.0, "iris_mm": 11.7},
            {"iris_mm": "large"},
            {"iris_color": [1, 2]},
            {"source": "photo"},
            {"cornea": 1},
            {"seed": 1.5},
            {"glow": 1},
            [],
        ):
            with self.subTest(eyes=eyes), self.assertRaises(ValueError):
                check_material({"eyes": eyes})


class TestGeometry(unittest.TestCase):
    shape = geometry.EyeShape.of(11.7, 0.015)

    def test_surfaces_face_outward(self):
        for profile in (
            geometry.eyeball_profile(self.shape),
            geometry.cornea_profile(self.shape, 1.0),
            geometry.single_profile(self.shape, 0.5),
        ):
            surface = geometry.revolve(profile, 32)
            p, t = surface.points, surface.triangles
            face = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
            self.assertTrue(((face * surface.normals[t].sum(1)).sum(1) > 0).all())
            np.testing.assert_allclose(np.linalg.norm(surface.normals, axis=1), 1.0)

    def test_iris_recess_cornea_and_bulge(self):
        rho_l, z_l = geometry.limbus(self.shape.iris)
        eyeball = geometry.revolve(geometry.eyeball_profile(self.shape), 24)
        iris = np.hypot(*eyeball.points[:, :2].T) < rho_l * 0.99
        iris &= eyeball.points[:, 2] > 0
        # the iris lies behind the plane of the limbus, inside the cornea's chamber
        self.assertTrue((eyeball.points[iris, 2] < z_l).all())
        cornea = geometry.revolve(geometry.cornea_profile(self.shape, 1.0), 24)
        apex = cornea.points[:, 2].max()
        height = (apex - 1.0) * 15.0  # mm on a 15 mm eyeball
        self.assertGreater(height, 1.0)
        self.assertLess(height, 2.0)
        # the outer radius at the apex agrees with the surface, and drops with the bulge
        self.assertAlmostEqual(
            float(
                geometry.outer_radius(
                    np.array([0.0]), self.shape, 1.0, 1 + self.shape.gap
                )[0]
            ),
            apex,
            places=6,
        )
        low = geometry.revolve(geometry.cornea_profile(self.shape, 0.25), 24)
        self.assertLess(low.points[:, 2].max(), apex)

    def test_lids_lower_the_bulge(self):
        axes = np.stack([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]], 1)
        frame = geometry.EyeFrame("L", np.zeros(3), 0.015, axes)
        # a ring of lid skin 15 degrees from the gaze, just outside the sphere
        angle = np.radians(15.0)
        phi = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        local = (
            np.stack(
                [
                    np.sin(angle) * np.cos(phi),
                    np.sin(angle) * np.sin(phi),
                    np.full(40, np.cos(angle)),
                ],
                1,
            )
            * 1.03
        )
        skin = frame.to_world(local)
        normals = frame.directions(local / np.linalg.norm(local, axis=1, keepdims=True))
        bulge, scale = geometry.bulge_clearance(frame, skin, normals, self.shape, 1.0)
        self.assertEqual(scale, 1.0)
        self.assertGreater(bulge, 0.0)
        self.assertLess(bulge, 1.0)
        # without lids in the way, the full bulge
        far = frame.to_world(local * 1.3)
        self.assertEqual(
            geometry.bulge_clearance(frame, far, normals, self.shape, 1.0)[0], 1.0
        )


class TestTexture(unittest.TestCase):
    def test_layout_round_trip(self):
        s, phi, left = tex.texel_polar()
        h, w = s.shape
        inside = left & (s < 1.99)  # the corners of each square lie off the eye
        polar = np.stack([s[inside], phi[inside]], 1)
        uv = tex.polar_to_uv(polar, "L")
        v, u = np.nonzero(inside)
        np.testing.assert_allclose(uv[:, 0], (u + 0.5) / w, atol=1e-9)
        np.testing.assert_allclose(uv[:, 1], 1 - (v + 0.5) / h, atol=1e-9)

    def test_procedural_eye(self):
        eye = EyeMaterial(pupil_mm=3.5, iris_color=[0.05, 0.1, 0.2])
        pixels = tex.procedural_eye_pixels(eye, {"L": 12.0, "R": 12.0})
        self.assertEqual(pixels.shape, (512, 1024, 3))
        found = tex.find_iris(
            tex.linear(pixels)[tex.texel_polar()[2]],
            *(a[tex.texel_polar()[2]] for a in tex.texel_polar()[:2]),
        )
        self.assertIsNotNone(found)
        centre, rho_p, rho_l, colour = found
        self.assertLess(np.abs(centre).max(), 0.01)
        self.assertAlmostEqual(rho_p * 12.0 * 2, 3.5, delta=0.3)
        self.assertAlmostEqual(rho_l * 12.0 * 2, 11.7, delta=0.8)
        self.assertGreater(colour[2], colour[0])  # blue
        # the same seed, the same eye
        np.testing.assert_array_equal(
            pixels, tex.procedural_eye_pixels(eye, {"L": 12.0, "R": 12.0})
        )


class TestEyedMesh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Character().build_model(dtype=torch.float64)
        cls.rest = cls.model.template_vertices.double().numpy()
        cls.eye = EyeMaterial()
        cls.eyed = eyed_mesh(cls.model, None, cls.rest, cls.eye, "gltf")

    def test_parts_and_rest(self):
        names = sorted({(p.name, p.side) for p in self.eyed.parts})
        self.assertEqual(
            names,
            [(n, s) for n in ("caruncle", "cornea", "eyeball") for s in ("L", "R")],
        )
        x = self.eyed.vertices(self.rest)
        start = self.eyed.base_count
        for p in self.eyed.parts:
            np.testing.assert_allclose(
                x[start : start + len(p.rest)], p.rest, atol=1e-12
            )
            start += len(p.rest)
        # anny's eyeball triangles are gone; every other triangle is kept
        self.assertEqual(int((~self.eyed.kept).sum()), 280)
        self.assertEqual(len(self.eyed.triangles), len(self.eyed.uv_triangles))
        self.assertTrue(np.isfinite(self.eyed.uvs).all())

    def test_the_eye_turns_rigidly(self):
        # an eye morph target that turns anny's eyeball turns the eye parts the same way
        rows = {p.side: p.anchors for p in self.eyed.parts if p.name == "eyeball"}
        import roma

        turn = roma.rotvec_to_rotmat(torch.tensor([0.3, 0.0, 0.2])).double().numpy()
        frame = self.eyed.frames["L"]
        moved = self.rest.copy()
        moved[rows["L"]] = (self.rest[rows["L"]] - frame.centre) @ turn.T + frame.centre
        offsets = (moved - self.rest)[None]
        x = self.eyed.vertices(self.rest) + self.eyed.targets(offsets)[0]
        start = self.eyed.base_count
        for p in self.eyed.parts:
            part = x[start : start + len(p.rest)]
            if p.side == "L" and p.name in ("eyeball", "cornea"):
                expected = (p.rest - frame.centre) @ turn.T + frame.centre
                np.testing.assert_allclose(part, expected, atol=1e-9)
            elif p.side == "R":
                np.testing.assert_allclose(part, p.rest, atol=1e-12)
            start += len(p.rest)
        # one set of skin weights for each eyeball: the parts turn with the eye bone
        weights = self.model.vertex_bone_weights.double().numpy()
        indices = self.model.vertex_bone_indices.numpy()
        w, i = self.eyed.skin_weights(weights, indices)
        start = self.eyed.base_count
        labels = list(self.model.bone_labels)
        for p in self.eyed.parts:
            pw, pi = w[start : start + len(p.rest)], i[start : start + len(p.rest)]
            if p.name != "caruncle":
                self.assertTrue((pw == pw[0]).all() and (pi == pi[0]).all())
                self.assertEqual(labels[pi[0, 0]], f"eye.{p.side}")
            np.testing.assert_allclose(pw.sum(1), 1.0)
            start += len(p.rest)

    def test_affine_operator(self):
        rng = np.random.default_rng(1)
        anchors = rng.normal(size=(20, 3))
        points = rng.normal(size=(7, 3))
        op = affine_operator(anchors, points)
        np.testing.assert_allclose(op @ anchors, points, atol=1e-10)
        np.testing.assert_allclose(op.sum(1), 1.0)
        a = rng.normal(size=(3, 3))
        np.testing.assert_allclose(
            op @ (anchors @ a.T + 1.5), points @ a.T + 1.5, atol=1e-9
        )

    def test_caruncle_covers_the_pocket(self):
        # the front view between the lids shows the skin of the socket in the inner corner;
        # the caruncle hides most of it
        x = self.eyed.vertices(self.rest)
        skin = self.eyed.triangles[: int(self.eyed.kept.sum())]
        start = int(self.eyed.kept.sum())
        caruncles = {}
        for p in self.eyed.parts:
            if p.name == "caruncle":
                caruncles[p.side] = self.eyed.triangles[
                    start : start + len(p.triangles)
                ]
            start += len(p.triangles)
        for side, frame in self.eyed.frames.items():
            with self.subTest(side=side):
                before = _pocket(x, skin, frame)
                after = _pocket(x, skin, frame, caruncles[side])
                self.assertGreater(before, 50)
                self.assertLess(after, 0.3 * before)

    def test_baked_sizes(self):
        # "auto" (the default) and "baked" keep the sizes found in the baked eyes; a pale
        # outer band of the iris belongs to the iris (the limbus is its outer edge, before the
        # darker limbal ring), as a game's eye texture may draw it
        for pale in (False, True):
            with self.subTest(pale_ring=pale):
                baked, radius = synthetic_baked_eyes(
                    self.model, self.eyed, pupil_mm=4.6, iris_mm=11.4, pale_ring=pale
                )
                eyed = eyed_mesh(
                    self.model, None, self.rest, EyeMaterial(), "gltf", baked
                )
                self.assertAlmostEqual(eyed.eye.iris_mm, 11.4, delta=0.35)
                self.assertAlmostEqual(eyed.eye.pupil_mm, 4.6, delta=0.35)
                self.assertEqual(sorted(eyed.measured), ["L", "R"])
                limbus = eyed.shapes["L"].iris * eyed.frames["L"].radius * 2000
                self.assertAlmostEqual(limbus, eyed.eye.iris_mm, delta=1e-6)
                pixels, info = eye_texture(eyed.eye, eyed, self.model, baked)
                s, phi, left = tex.texel_polar()
                found = tex.find_iris(tex.linear(pixels)[left], s[left], phi[left])
                self.assertAlmostEqual(2 * found[1] * radius, 4.6, delta=0.4)
                self.assertAlmostEqual(2 * found[2] * radius, 11.4, delta=0.5)
        # without a bake, "auto" is the defaults and "baked" an error
        self.assertEqual(self.eyed.eye.iris_mm, 11.7)
        self.assertEqual(self.eyed.eye.pupil_mm, 3.5)
        with self.assertRaises(ValueError):
            eyed_mesh(self.model, None, self.rest, EyeMaterial(pupil_mm="baked"))
        # given sizes win over the bake
        fixed = eyed_mesh(
            self.model,
            None,
            self.rest,
            EyeMaterial(iris_mm=11.7, pupil_mm=3.5),
            "gltf",
            baked,
        )
        self.assertEqual((fixed.eye.iris_mm, fixed.eye.pupil_mm), (11.7, 3.5))

    def test_vrm_eyes_have_one_surface(self):
        eyed = eyed_mesh(self.model, None, self.rest, self.eye, "vrm")
        self.assertEqual(sorted({p.name for p in eyed.parts}), ["caruncle", "eyeball"])
        self.assertLess(len(eyed.triangles), 32_000)  # the VRM 0.x budget

    def test_baked_eyes_take_the_cards_sizes(self):
        baked, radius = synthetic_baked_eyes(self.model, self.eyed)
        eye = EyeMaterial(
            pupil_mm=3.5, iris_mm=11.7, iris_tint=1.0, iris_color=[0.05, 0.12, 0.2]
        )
        pixels, info = eye_texture(eye, self.eyed, self.model, baked)
        self.assertEqual(info["sides"], {"L": "baked", "R": "baked"})
        s, phi, left = tex.texel_polar()
        found = tex.find_iris(tex.linear(pixels)[left], s[left], phi[left])
        centre, rho_p, rho_l, colour = found
        self.assertLess(np.abs(centre).max(), 0.01)  # the iris moved onto the gaze
        self.assertAlmostEqual(2 * rho_p * radius, 3.5, delta=0.35)
        self.assertAlmostEqual(2 * rho_l * radius, 11.7, delta=0.8)
        self.assertGreater(colour[2], 2 * colour[0])  # tinted blue
        # without a baked texture, or with source "procedural", the procedural eye
        self.assertEqual(
            eye_texture(eye, self.eyed, self.model)[1]["source"], "procedural"
        )


def _pocket(points, triangles, frame, cover=None) -> int:
    """The pixels of the front view's pocket in the inner corner (as geometry.caruncle
    finds it) that no ``cover`` triangles hide."""
    from scipy import ndimage

    local = frame.to_local(points)
    near = (np.linalg.norm(local, axis=1) < 1.6) & (local[:, 2] > -0.2)
    step = geometry.CARUNCLE_STEP
    xs = np.arange(-1.3, 1.3 + step / 2, step)
    ys = np.arange(-0.8, 0.8 + step / 2, step)
    depth = geometry._depth_raster(local, triangles[near[triangles].all(1)], xs, ys)
    gx, gy = np.meshgrid(xs, ys)
    rr = gx**2 + gy**2
    front = np.where(rr < 1, np.sqrt(np.maximum(1 - rr, 0)), -np.inf)
    if cover is not None:
        front = np.maximum(front, geometry._depth_raster(local, cover, xs, ys))
    skin_first = depth > front
    radius = np.sqrt(rr + np.where(np.isfinite(depth), depth, 0) ** 2)
    opening = (~skin_first & np.isfinite(front)) | (skin_first & (radius < 1.12))
    labels, _ = ndimage.label(opening)
    under = np.maximum(depth, front)
    _, (iy, ix) = ndimage.distance_transform_edt(
        ~np.isfinite(under), return_indices=True
    )
    under = under[iy, ix]
    window = int(round(geometry.CARUNCLE_WINDOW / step))
    recessed = ndimage.grey_closing(under, size=(window, window)) - under > 0.03
    nasal = -1.0 if frame.centre[0] >= 0 else 1.0
    pocket = (labels == labels[len(ys) // 2, len(xs) // 2]) & skin_first & recessed
    return int((pocket & (nasal * gx > geometry.CARUNCLE_NASAL)).sum())


class TestExports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.model = Character().build_model(dtype=torch.float64)
        cls.card = Character(name="eyes", material={"eyes": {"pupil_mm": 3.2}})
        cls.path = cls.folder / "eyes.glb"
        export_glb(cls.path, cls.card, cls.model)
        cls.glb = GLB(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _materials(self, glb):
        g = glb.json
        return {
            g["materials"][p["material"]]["name"]: p
            for p in g["meshes"][0]["primitives"]
        }

    def test_structure(self):
        g = self.glb.json
        self.assertEqual(structure_errors(self.glb), [])
        prims = self._materials(self.glb)
        self.assertEqual(sorted(prims), ["conjunctiva", "cornea", "eyes", "skin"])
        for p in prims.values():  # the eye parts share the body's vertices and targets
            self.assertEqual(p["attributes"], prims["skin"]["attributes"])
            self.assertEqual(p.get("targets"), prims["skin"].get("targets"))
        materials = {m["name"]: m for m in g["materials"]}
        cornea = materials["cornea"]
        self.assertNotIn(
            "alphaMode", cornea
        )  # OPAQUE, as KHR_materials_transmission asks
        ext = cornea["extensions"]
        self.assertEqual(ext["KHR_materials_transmission"]["transmissionFactor"], 1.0)
        self.assertAlmostEqual(ext["KHR_materials_ior"]["ior"], 1.376)
        self.assertGreater(ext["KHR_materials_volume"]["thicknessFactor"], 0.0)
        self.assertIn("KHR_materials_clearcoat", materials["conjunctiva"]["extensions"])
        self.assertIn("baseColorTexture", materials["eyes"]["pbrMetallicRoughness"])
        # under the cornea the eyeball reflects nothing of its own (no milky veil)
        self.assertEqual(
            materials["eyes"]["extensions"]["KHR_materials_specular"]["specularFactor"],
            0.0,
        )
        for name in ("eyes", "cornea", "conjunctiva"):
            extras = materials[name]["extras"]["opensculptboy_eye"]
            self.assertEqual(extras["format"], "opensculptboy/eye@1")
            self.assertEqual(extras["pupil_mm"], 3.2)
        info = g["scenes"][0]["extras"]["opensculptboy"]["eyes"]
        self.assertEqual(info["source"], "procedural")
        self.assertTrue(info["cornea"] and info["caruncle"])
        from opensculptboy.export.gltf import read_character

        self.assertEqual(read_character(self.path), self.card)
        # the skin keeps the exporters' own material
        self.assertNotIn("extensions", materials["skin"])

    def test_eyes_follow_the_file(self):
        names = self.glb.json["meshes"][0]["extras"]["targetNames"]
        prims = self._materials(self.glb)
        indices = self.glb.accessor(prims["eyes"]["indices"]).reshape(-1, 3)
        rows = np.unique(indices)
        rest = evaluate_nodes(self.glb)
        mesh_node = min(rest)
        p0 = rest[mesh_node][0].positions[rows]
        # the left eyeball (+X) turns rigidly with eyeLook targets
        weights = np.zeros(len(names))
        weights[names.index("eyeLookUpLeft")] = 1.0
        moved = evaluate_nodes(self.glb, weights=list(weights))[mesh_node][0].positions[
            rows
        ]
        left = p0[:, 0] > 0
        self.assertGreater(np.abs(moved[left] - p0[left]).max(), 1e-3)
        # anny's own target turns its eyeball with 0.13 mm of other movement; the affine map
        # carries no more of it over
        self.assertLess(_kabsch_residual(p0[left], moved[left]), 1.3e-4)
        np.testing.assert_allclose(moved[~left], p0[~left], atol=1e-6)

    def test_without_cornea(self):
        path = self.folder / "single.glb"
        card = Character(material={"eyes": {"cornea": False, "caruncle": False}})
        export_glb(path, card, self.model)
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        self.assertEqual(sorted(self._materials(glb)), ["eyes", "skin"])
        eyes = next(m for m in glb.json["materials"] if m["name"] == "eyes")
        self.assertEqual(
            eyes["extensions"]["KHR_materials_clearcoat"]["clearcoatFactor"], 1.0
        )

    def test_vrm(self):
        from opensculptboy.export.budget import BUDGETS
        from opensculptboy.export.vrm import export_vrm
        from test.test_vrm_schema import check_vrm0, check_vrm1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for version in ("1.0", "0.x"):
                with self.subTest(version=version):
                    path = self.folder / f"eyes{version}.vrm"
                    summary = export_vrm(
                        path,
                        self.card,
                        self.model,
                        version=version,
                        author="tests",
                        refine_weights=False,
                        budget="strict",
                    )
                    glb = GLB(path)
                    g = glb.json
                    self.assertEqual(structure_errors(glb), [])
                    check = check_vrm1 if version == "1.0" else check_vrm0
                    self.assertEqual(check(g), [])
                    self.assertEqual(
                        [m["name"] for m in g["materials"]],
                        ["skin", "eyes", "conjunctiva"],
                    )
                    self.assertEqual(
                        [i["name"] for i in g["images"][:2]],
                        ["eye_base_color", "eye_matcap"],
                    )
                    budget = BUDGETS["vrm1" if version == "1.0" else "vrm0"]
                    self.assertLessEqual(summary["triangles"], budget["triangles"])
                    if version == "1.0":
                        mtoon = g["materials"][1]["extensions"]["VRMC_materials_mtoon"]
                        self.assertIn("matcapTexture", mtoon)
                    else:
                        props = g["extensions"]["VRM"]["materialProperties"][1]
                        self.assertIn("_SphereAdd", props["textureProperties"])
                    extras = g["scenes"][0]["extras"]["opensculptboy"]["eyes"]
                    self.assertFalse(extras["cornea"])

    def test_detail_level(self):
        # level 1 of a card with offsets and a baked eye texture: the eyes fit the level's
        # eyeballs, take the baked iris, and the VRM 1.0 file keeps within its budget
        import dataclasses

        from opensculptboy.export.vrm import export_vrm
        from test.test_detail_layer import detailed_card

        from opensculptboy.detail import detail_topology, load_offsets, save_offsets
        from opensculptboy.eyes.mesh import eyeball_rows

        folder = self.folder / "detail"
        folder.mkdir()
        card = detailed_card(folder)
        # the fit moves the eyeballs onto the reference's spheres: no noise on them
        for entry in card.detail["levels"]:
            path = folder / entry["offsets"]
            offsets = load_offsets(path, entry["subdiv"])
            mesh = detail_topology(self.model, entry["subdiv"])
            for rows in eyeball_rows(self.model, mesh).values():
                offsets[mesh.rows[rows]] = 0.0
            save_offsets(path, offsets, entry["subdiv"])
        eyed = eyed_mesh(
            self.model,
            None,
            self.model.template_vertices.double().numpy(),
            EyeMaterial(),
        )
        baked, _ = synthetic_baked_eyes(self.model, eyed)
        (folder / "card.eyes.png").write_bytes(baked)
        card = dataclasses.replace(
            card,
            material={"eyes": {}},
            detail=dict(card.detail, textures={"eyes": "card.eyes.png"}),
        )
        path = folder / "level1.glb"
        export_glb(path, card, self.model, detail_level=1, detail_dir=folder)
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        info = glb.json["scenes"][0]["extras"]["opensculptboy"]["eyes"]
        self.assertEqual(info["sides"], {"L": "baked", "R": "baked"})
        self.assertEqual(info["bulge"]["L"], info["bulge"]["R"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            export_vrm(
                folder / "level1.vrm",
                card,
                self.model,
                author="tests",
                refine_weights=False,
                detail_level=1,
                detail_dir=folder,
                budget="strict",
            )

    def test_blender_helper(self):
        from opensculptboy.export.blender_skin import (
            EYE_EXTRAS_KEY,
            eye_principled_values,
            read_skin_settings,
        )

        settings = read_skin_settings(str(self.path), EYE_EXTRAS_KEY)
        self.assertEqual(sorted(settings), ["conjunctiva", "cornea", "eyes"])
        cornea = eye_principled_values(settings["cornea"])
        self.assertTrue(cornea["shadow_transparent"])
        self.assertEqual(cornea["inputs"]["Transmission Weight"], 1.0)
        self.assertAlmostEqual(cornea["inputs"]["IOR"], 1.376)
        self.assertEqual(cornea["inputs"]["Coat Weight"], 0.0)
        self.assertTrue(cornea["drop_volume"])
        eyeball = eye_principled_values(settings["eyes"])
        self.assertFalse(eyeball["shadow_transparent"])
        # no scattering that blurs the iris, no reflection under the cornea
        self.assertEqual(eyeball["inputs"]["Subsurface Weight"], 0.0)
        self.assertEqual(eyeball["inputs"]["Specular IOR Level"], 0.0)
        single = eye_principled_values(dict(settings["eyes"], cornea=False))
        self.assertNotIn("Specular IOR Level", single["inputs"])
        # the skin settings are read apart from the eyes'
        self.assertEqual(read_skin_settings(str(self.path)), {})

    def test_command(self):
        import contextlib
        import io

        from opensculptboy.cli import main

        values = self.folder / "eyes.json"
        values.write_text(
            json.dumps({"iris_color": [0.1, 0.2, 0.1], "caruncle": False})
        )
        out = self.folder / "command.glb"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["export", str(out), "--eyes", str(values)]), 0)
        glb = GLB(out)
        self.assertEqual(sorted(self._materials(glb)), ["cornea", "eyes", "skin"])
        card = glb.json["scenes"][0]["extras"]["opensculptboy"]["character"]
        self.assertEqual(card["material"]["eyes"]["iris_color"], [0.1, 0.2, 0.1])


if __name__ == "__main__":
    unittest.main()

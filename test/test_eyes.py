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
    model,
    eyed,
    centre=(0.03, -0.02),
    pupil_mm=2.8,
    iris_mm=11.2,
    pale_ring=False,
    pupil=0.01,
):
    """A baked eye texture on anny's atlas (the box of opensculptboy.texture.eyes): a grey
    iris with a wide pupil, off the gaze by ``centre`` (eyeball radii), as a game's eye
    texture might come, with ``pale_ring`` a pale outer band of the iris (1.2 mm) and a
    darker limbal ring outside it (0.5 mm), and a pupil of linear grey ``pupil``; returns the
    PNG bytes and the eyeball radius used (mm)."""
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
        colour = np.where((r < pupil_mm / 2)[..., None], pupil, iris)
        colour = np.where((values[..., 2] > 0)[..., None], colour, [0.7, 0.68, 0.66])
        image[covered] = colour[covered]
    return png_bytes(tex.srgb(image)), radius


def bowl_normal_atlas(model, eyed, size=2048, tilt_deg=20.0, reach=0.35):
    """A tangent-space normal map (size, size, 3, in [0, 1]) of anny's atlas, flat but on the
    eyeballs, where it holds a concave iris as a bake writes it (in the MikkTSpace frames of
    the eyeballs): within ``reach`` eyeball radii of the gaze axis the normal tilts by
    ``tilt_deg`` toward the axis."""
    from opensculptboy.texture.eyes import eye_uv_box
    from opensculptboy.texture.tangents import mikktspace

    lo, hi = eye_uv_box(model)
    atlas = np.full((size, size, 3), 0.5, np.float32)
    atlas[..., 2] = 1.0
    x0, y0 = int(lo[0] * size) - 2, int((1 - hi[1]) * size) - 2
    x1, y1 = int(hi[0] * size) + 3, int((1 - lo[1]) * size) + 3
    for points, triangles, uvs in eyed.eyeballs.values():
        corners = triangles.reshape(-1)
        p = points[corners]
        n = p / np.linalg.norm(p, axis=1, keepdims=True)
        uv = uvs.reshape(-1, 2)
        fine = np.arange(len(corners)).reshape(-1, 3)
        tangents = mikktspace(p, n, uv, fine)
        b = tangents[:, 3:] * np.cross(n, tangents[:, :3])
        px = np.stack([uv[:, 0] * size - 0.5 - x0, (1 - uv[:, 1]) * size - 0.5 - y0], 1)
        values = np.concatenate([n, tangents[:, :3], b], 1)
        grid, mask = tex.raster(px, fine, values, (x1 - x0, y1 - y0))
        nn, tt, bb = grid[..., 0:3], grid[..., 3:6], grid[..., 6:9]
        nn = nn / np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-12)
        radial = np.concatenate([nn[..., :2], np.zeros(nn.shape[:2] + (1,))], -1)
        inward = -radial / np.maximum(
            np.linalg.norm(radial, axis=-1, keepdims=True), 1e-12
        )
        inward -= (inward * nn).sum(-1, keepdims=True) * nn
        inside = (np.hypot(nn[..., 0], nn[..., 1]) < reach) & (nn[..., 2] > 0)
        angle = np.radians(tilt_deg) * inside[..., None]
        bent = np.cos(angle) * nn + np.sin(angle) * inward
        code = np.stack(
            [(bent * tt).sum(-1), (bent * bb).sum(-1), (bent * nn).sum(-1)], -1
        )
        box = atlas[y0:y1, x0:x1]
        box[mask] = 0.5 * code[mask] + 0.5
    return atlas


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
        # cornea "auto": one coated surface for baked eyes, as the reference draws them
        self.assertEqual(eye.cornea, "auto")
        self.assertIs(eye.resolved(baked=True).cornea, False)
        self.assertIs(eye.resolved(baked=False).cornea, True)
        self.assertIs(EyeMaterial(cornea=True).resolved(baked=True).cornea, True)
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
            {"cornea": "yes"},
            {"cornea_radius_mm": 30.0},
            {"seed": 1.5},
            {"glow": 1},
            [],
        ):
            with self.subTest(eyes=eyes), self.assertRaises(ValueError):
                check_material({"eyes": eyes})


def _flat_front_eye(centre, radius=0.012, cornea=0.0093, limbus=0.005, subdivisions=5):
    """An eyeball sphere (Anny's frame, gaze -Y) whose front within ``limbus`` of the gaze
    axis is a flatter cap of radius ``cornea``, as a game's eye often is: (vertices,
    triangles)."""
    import trimesh

    ball = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    points = np.asarray(ball.vertices, dtype=np.float64).copy()
    rho = np.hypot(points[:, 0], points[:, 2])
    cap = (points[:, 1] < 0) & (rho < limbus)
    z0 = -np.sqrt(radius**2 - limbus**2) + np.sqrt(cornea**2 - limbus**2)
    points[cap, 1] = z0 - np.sqrt(cornea**2 - rho[cap] ** 2)
    return points + np.asarray(centre), np.asarray(ball.faces, dtype=np.int64)


class TestReferenceEyes(unittest.TestCase):
    """The values of the reference's eyes that the bake records in material.reference_eyes,
    and the cornea's "auto" values that take them."""

    measured = {"cornea_radius_mm": 9.3, "cornea_roughness": 0.05, "cornea_ior": 1.5}

    def test_auto_takes_the_reference_values(self):
        eye = EyeMaterial.from_dict({}, self.measured)
        self.assertEqual(eye.cornea_radius_mm, 9.3)
        self.assertEqual(eye.cornea_roughness, 0.05)
        self.assertEqual(eye.cornea_ior, 1.5)
        given = EyeMaterial.from_dict(
            {"cornea_radius_mm": 8.0, "cornea_ior": "auto"}, self.measured
        )
        self.assertEqual(given.cornea_radius_mm, 8.0)  # a number wins over the measure
        self.assertEqual(given.cornea_ior, 1.5)
        plain = EyeMaterial.from_dict({})
        self.assertEqual(
            (plain.cornea_radius_mm, plain.cornea_roughness, plain.cornea_ior),
            (7.8, 0.02, 1.376),
        )
        partial = EyeMaterial.from_dict({}, {"cornea_radius_mm": 9.3})
        self.assertEqual((partial.cornea_radius_mm, partial.cornea_ior), (9.3, 1.376))

    def test_card(self):
        card = Character(material={"eyes": {}, "reference_eyes": self.measured})
        self.assertEqual(
            Character.from_dict(json.loads(json.dumps(card.to_dict()))), card
        )
        self.assertEqual(card.eye_material().cornea_radius_mm, 9.3)
        self.assertEqual(card.eye_material().sized().extras()["cornea_radius_mm"], 9.3)
        self.assertIsNone(card.skin_material())
        # the measured values alone turn the eyes on (a fitted card) and keep the
        # exporters' own skin; "eyes": false keeps anny's own eyeballs
        alone = Character(material={"reference_eyes": self.measured})
        self.assertEqual(alone.eye_material().cornea_radius_mm, 9.3)
        self.assertIsNone(alone.skin_material())
        off = Character(material={"eyes": False, "reference_eyes": self.measured})
        self.assertIsNone(off.eye_material())
        self.assertIsNone(off.lash_material())
        baked = Character(detail={"textures": {"eyes": "card.eyes.png"}})
        self.assertIsNotNone(baked.eye_material())
        self.assertIsNone(Character().eye_material())
        skin = Character(
            material={"ior": 1.4, "eyes": {}, "reference_eyes": self.measured}
        )
        self.assertEqual(skin.skin_material().eyes.cornea_roughness, 0.05)
        for bad in (
            {"cornea_radius_mm": 30.0},
            {"cornea_ior": "auto"},
            {"cornea_roughness": True},
            {"iris_mm": 11.0},
            [],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_material({"reference_eyes": bad})
        with self.assertRaises(ValueError):
            check_material({"eyes": {"cornea_ior": "wet"}})

    def test_measured_on_a_handoff(self):
        from opensculptboy.fit.handoff import HandoffPart, load_handoff, write_handoff
        from opensculptboy.fit.reference import Reference
        import trimesh

        body = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
        eyes = [_flat_front_eye([x, -0.12, 0.0]) for x in (0.03, -0.03)]
        with tempfile.TemporaryDirectory() as folder:
            write_handoff(
                folder,
                np.asarray(body.vertices),
                np.asarray(body.faces),
                parts=[(f"eye_{k}", v, t, "eyes") for k, (v, t) in enumerate(eyes)],
            )
            reference = Reference.from_handoff(load_handoff(folder))
        self.assertEqual(len(reference.eyes), 2)
        values = reference.eye_values()
        self.assertAlmostEqual(values["cornea_radius_mm"], 9.3, delta=0.1)
        # placed elsewhere, the reference measures the same
        moved = reference.transformed(np.eye(3), np.array([0.0, 0.5, 1.0])).eye_values()
        self.assertAlmostEqual(moved["cornea_radius_mm"], values["cornea_radius_mm"])
        # the eye parts' material: the median of a roughness texture, and the IOR
        for part in reference.eye_reference.parts:
            part.roughness = np.full((4, 4), 0.05, np.float32)
            part.ior = 1.5
        values = reference.eye_values()
        self.assertEqual(
            (values["cornea_roughness"], values["cornea_ior"]), (0.05, 1.5)
        )
        self.assertIsInstance(reference.eye_reference.parts[0], HandoffPart)
        self.assertEqual(Reference(body.vertices, body.faces).eye_values(), {})

    def test_material_iors(self):
        from opensculptboy.fit.handoff import _material_iors

        document = {
            "asset": {"version": "2.0"},
            "materials": [
                {"name": "eye", "extensions": {"KHR_materials_ior": {"ior": 1.33}}},
                {"name": "skin"},
                {},
            ],
        }
        data = json.dumps(document).encode()
        data += b" " * (-len(data) % 4)
        glb = (
            b"glTF" + (2).to_bytes(4, "little") + (20 + len(data)).to_bytes(4, "little")
        )
        glb += len(data).to_bytes(4, "little") + b"JSON" + data
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "replica.glb"
            path.write_bytes(glb)
            self.assertEqual(_material_iors(path), {"eye": 1.33, "skin": 1.5})
            path.write_bytes(b"not a glb file")
            self.assertEqual(_material_iors(path), {})

    def test_baked_material_records_them(self):
        from opensculptboy.texture.bake import BakedTextures

        baked = BakedTextures(
            None, np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), bool), 1.0
        )
        self.assertNotIn("reference_eyes", baked.material({"normal": "n.png"}))
        baked.reference_eyes = dict(self.measured)
        material = baked.material({"normal": "n.png"})
        self.assertEqual(material["reference_eyes"], self.measured)
        check_material(material)


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

    def test_front_curvature(self):
        # the radius of curvature of a reference eye's front: a sphere of 12 mm with a cap of
        # 9.3 mm radius over its front 5 mm
        rng = np.random.default_rng(0)
        d = rng.normal(size=(4000, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        points = 0.012 * d
        rho = np.hypot(points[:, 0], points[:, 2])
        cap = (points[:, 1] < 0) & (rho < 0.005)
        z0 = -np.sqrt(0.012**2 - 0.005**2) + np.sqrt(0.0093**2 - 0.005**2)
        points[cap, 1] = z0 - np.sqrt(0.0093**2 - rho[cap] ** 2)
        radius = geometry.front_curvature_mm(points, np.zeros(3))
        self.assertAlmostEqual(radius, 9.3, delta=0.05)
        self.assertIsNone(geometry.front_curvature_mm(points[:5], np.zeros(3)))

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


class TestBakedNormals(unittest.TestCase):
    def test_baked_normals_move_with_the_iris(self):
        # a concave iris off the gaze: once fit_baked_iris moves it onto the gaze, its normals
        # tilt toward the gaze axis all round, even next to the pupil's centre
        s, phi, left = tex.texel_polar()
        x, y = s * np.cos(phi), s * np.sin(phi)
        centre = np.array([0.06, -0.04])
        dx, dy = x - centre[0], y - centre[1]
        r = np.hypot(dx, dy)
        lum = np.where(r < 0.15, 0.01, np.where(r < 0.45, 0.12, 0.7))
        image = np.repeat(lum[..., None], 3, axis=2)
        tilt = np.sin(np.radians(20.0)) * (r < 0.4)
        ux, uy = -dx / np.maximum(r, 1e-9), -dy / np.maximum(r, 1e-9)
        a = tilt * (ux * -np.sin(phi) + uy * np.cos(phi))  # along E1
        b = tilt * (ux * -np.cos(phi) + uy * -np.sin(phi))  # along E2 (toward the axis)
        normals = np.stack([a, b, np.sqrt(1 - a**2 - b**2)], -1)
        radius = 12.0
        eye = EyeMaterial(iris_mm=2 * 0.45 * radius, pupil_mm=2 * 0.15 * radius)
        _, found, moved = tex.fit_baked_iris(
            image, eye, {"L": radius, "R": radius}, normals=normals
        )
        self.assertEqual(
            {k: v is not None for k, v in found.items()}, {"L": True, "R": True}
        )
        for lo, hi in ((0.02, 0.08), (0.1, 0.3)):
            d = moved[left & (s > lo) & (s < hi)]
            self.assertGreater(np.median(d[:, 1]), np.sin(np.radians(15.0)))
            self.assertLess(np.percentile(np.abs(d[:, 0]), 90), 0.1)


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

    def test_baked_eyes_keep_their_pupil(self):
        # the pupil keeps the bake's own texels (a game's pupil is often shaded grey), and
        # baked eyes take one coated surface unless the card asks for a cornea
        baked, _ = synthetic_baked_eyes(self.model, self.eyed, pupil=0.05)
        eyed = eyed_mesh(self.model, None, self.rest, EyeMaterial(), "gltf", baked)
        self.assertEqual(sorted({p.name for p in eyed.parts}), ["caruncle", "eyeball"])
        self.assertIs(eyed.eye.cornea, False)
        with_cornea = eyed_mesh(
            self.model, None, self.rest, EyeMaterial(cornea=True), "gltf", baked
        )
        self.assertIn("cornea", {p.name for p in with_cornea.parts})
        pixels, _ = eye_texture(EyeMaterial(), eyed, self.model, baked)
        s, phi, left = tex.texel_polar()
        centre = tex.linear(pixels)[left & (s < 0.05)]
        np.testing.assert_allclose(centre, 0.05, atol=0.01)

    def test_baked_normals_carry_over(self):
        # the reference's eye normals, baked onto the eyeballs of the atlas normal map in
        # their MikkTSpace frames, reach the eye texture's layout: flat stays flat, and a
        # concave iris (its normals tilted toward the gaze axis) stays concave on both sides
        size = 2048
        flat = np.full((size, size, 3), 0.5, np.float32)
        flat[..., 2] = 1.0
        carried = tex.carry_baked_normals(flat, self.eyed.eyeballs)
        s, phi, left = tex.texel_polar()
        front = s < 0.9
        flat_front = np.broadcast_to([0.0, 0.0, 1.0], carried[front].shape)
        np.testing.assert_allclose(carried[front], flat_front, atol=1e-6)
        # a bowl: inside 0.35 eyeball radii the normal tilts 20 degrees toward the axis
        bowl = bowl_normal_atlas(self.model, self.eyed, size)
        carried = tex.carry_baked_normals(bowl, self.eyed.eyeballs)
        ring = (s > 0.12) & (s < 0.3)
        for side, half in (("L", left), ("R", ~left)):
            with self.subTest(side=side):
                d = carried[ring & half]
                # E2 = N x E1 points toward the gaze axis: the bowl tilts along it
                self.assertGreater(np.median(d[:, 1]), np.sin(np.radians(15.0)))
                self.assertLess(np.abs(np.median(d[:, 0])), 0.05)
        outside = (s > 0.5) & (s < 0.9)
        self.assertLess(np.median(np.abs(carried[outside][:, 1])), 0.05)
        # the same normals baked on the eye texture's box of the atlas (slot eyes_normal)
        from opensculptboy.texture.eyes import eye_uv_box

        lo, hi = eye_uv_box(self.model)
        v, u = np.mgrid[0:512, 0:1024]
        au = lo[0] + (u + 0.5) / 1024 * (hi[0] - lo[0])
        av = lo[1] + (1 - (v + 0.5) / 512) * (hi[1] - lo[1])
        boxed = tex._bilinear(bowl, au * size - 0.5, (1 - av) * size - 0.5)
        carried = tex.carry_baked_normals(boxed, self.eyed.eyeballs, box=(lo, hi))
        for half in (left, ~left):
            d = carried[ring & half]
            self.assertGreater(np.median(d[:, 1]), np.sin(np.radians(15.0)))
        # where the procedural eye stands in (no iris found), the normals are flat
        _, found, moved = tex.fit_baked_iris(
            np.full(carried.shape, 0.5),
            EyeMaterial(iris_mm=11.7, pupil_mm=3.5),
            normals=carried,
        )
        self.assertEqual(found, {"L": None, "R": None})
        np.testing.assert_allclose(moved[front], [[0.0, 0.0, 1.0]] * int(front.sum()))


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
        self.assertEqual(
            sorted(prims), ["conjunctiva", "cornea", "eyes", "lashes", "skin"]
        )
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
        self.assertEqual(sorted(self._materials(glb)), ["eyes", "lashes", "skin"])
        eyes = next(m for m in glb.json["materials"] if m["name"] == "eyes")
        self.assertEqual(
            eyes["extensions"]["KHR_materials_clearcoat"]["clearcoatFactor"], 1.0
        )

    def test_baked_eye_normals(self):
        # a file whose eyes and normal map come from a bake: one coated surface, and the
        # eye's normal texture holds the reference's concave iris in the file's own
        # MikkTSpace frames (decoded on the file's vertices, the normals tilt toward the
        # gaze axis)
        from PIL import Image

        from opensculptboy.export.document import png_bytes

        rest = self.model.template_vertices.double().numpy()
        eyed = eyed_mesh(self.model, None, rest, EyeMaterial())
        baked, _ = synthetic_baked_eyes(self.model, eyed)
        atlas = bowl_normal_atlas(self.model, eyed)
        normal = png_bytes(np.round(atlas * 255).astype(np.uint8))
        path = self.folder / "baked_normals.glb"
        card = Character(material={"eyes": {}})
        export_glb(path, card, self.model, textures={"eyes": baked, "normal": normal})
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        prims = self._materials(glb)
        self.assertNotIn("cornea", prims)
        eyes = next(m for m in glb.json["materials"] if m["name"] == "eyes")
        self.assertIn("KHR_materials_clearcoat", eyes["extensions"])
        self.assertFalse(eyes["extras"]["opensculptboy_eye"]["cornea"])
        texture = glb.json["textures"][eyes["normalTexture"]["index"]]
        image = glb.json["images"][texture["source"]]
        self.assertEqual(image["name"], "eye_normal")
        view = glb.json["bufferViews"][image["bufferView"]]
        data = glb.bin[
            view.get("byteOffset", 0) : view.get("byteOffset", 0) + view["byteLength"]
        ]
        import io

        pixels = np.asarray(Image.open(io.BytesIO(bytes(data))).convert("RGB")) / 255.0
        attributes = prims["eyes"]["attributes"]
        rows = np.unique(glb.accessor(prims["eyes"]["indices"]))
        position = glb.accessor(attributes["POSITION"])[rows]
        normals = glb.accessor(attributes["NORMAL"])[rows]
        tangents = glb.accessor(attributes["TANGENT"])[rows]
        uv = glb.accessor(attributes["TEXCOORD_0"])[rows]
        h, w = pixels.shape[:2]
        code = (
            2.0
            * pixels[
                np.clip((uv[:, 1] * h).astype(int), 0, h - 1),
                np.clip((uv[:, 0] * w).astype(int), 0, w - 1),
            ]
            - 1.0
        )
        bitangent = tangents[:, 3:] * np.cross(normals, tangents[:, :3])
        bent = (
            code[:, :1] * tangents[:, :3]
            + code[:, 1:2] * bitangent
            + code[:, 2:] * normals
        )
        for side, mine in (("L", position[:, 0] > 0), ("R", position[:, 0] < 0)):
            with self.subTest(side=side):
                p, n, b = position[mine], normals[mine], bent[mine]
                # the gaze runs along +Z of the file through the eye's centre
                centre = p.mean(0)
                radial = p[:, :2] - centre[:2]
                rho = np.linalg.norm(radial, axis=1)
                ring = (rho > 0.15 * rho.max()) & (rho < 0.28 * rho.max())
                ring &= n[:, 2] > 0.8
                inward = -radial[ring] / rho[ring, None]
                tilt = (b[ring, :2] * inward).sum(1) - (n[ring, :2] * inward).sum(1)
                self.assertGreater(np.median(tilt), 0.2)

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
                        ["skin", "eyes", "conjunctiva", "lashes"],
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
        self.assertEqual(sorted(settings), ["conjunctiva", "cornea", "eyes", "lashes"])
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
        self.assertEqual(
            sorted(self._materials(glb)), ["cornea", "eyes", "lashes", "skin"]
        )
        card = glb.json["scenes"][0]["extras"]["opensculptboy"]["character"]
        self.assertEqual(card["material"]["eyes"]["iris_color"], [0.1, 0.2, 0.1])

    def test_command_off(self):
        # --eyes off keeps anny's own eyeballs for a card whose reference eyes turn the
        # eyes on
        import contextlib
        import io

        from opensculptboy.cli import main

        fitted = Character(material={"reference_eyes": {"cornea_radius_mm": 7.6}})
        self.assertIsNotNone(fitted.eye_material())
        path = self.folder / "fitted.json"
        path.write_text(json.dumps(fitted.to_dict()), encoding="utf-8")
        out = self.folder / "off.glb"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(["export", str(out), "--character", str(path), "--eyes", "off"]), 0
            )
        glb = GLB(out)
        self.assertEqual(sorted(self._materials(glb)), ["skin"])
        extras = glb.json["scenes"][0]["extras"]["opensculptboy"]
        self.assertNotIn("eyes", extras)
        self.assertIs(extras["character"]["material"]["eyes"], False)
        # anny's eyeball triangles stay in the skin
        from opensculptboy.export.body import triangulated_faces

        triangles, _ = triangulated_faces(Character().build_model())
        prim = glb.json["meshes"][0]["primitives"][0]
        self.assertEqual(glb.accessor(prim["indices"]).size, triangles.size)


def _has_embree() -> bool:
    try:
        import trimesh.ray.ray_pyembree  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_has_embree(), "the eye bake needs embreex (the viewer extra)")
class TestEyeLayoutBake(unittest.TestCase):
    """The reference's eyes baked straight onto the eye texture's layout, with the gaze turned
    toward the reference's pupil."""

    def test_layout_and_gaze(self):
        from opensculptboy.eyes import geometry
        from opensculptboy.eyes import texture as tex
        from opensculptboy.fit.handoff import HandoffPart
        from opensculptboy.fit.reference import Reference
        from opensculptboy.texture.eyes import bake_eye_layout, turned_axes

        # a reference eye: a sphere of 12 mm whose texture draws a dark pupil and a grey iris
        # about a gaze 6 degrees toward +X from -Y
        radius, centre = 0.012, np.array([0.03, -0.05, 1.5])
        rings, segments = 48, 96
        theta = np.linspace(0, np.pi, rings + 1)
        phi = np.linspace(0, 2 * np.pi, segments + 1)
        t, p = np.meshgrid(theta, phi, indexing="ij")
        local = np.stack(
            [np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1
        ).reshape(-1, 3)
        uvs = np.stack([p / (2 * np.pi), t / np.pi], -1).reshape(-1, 2)
        quads = []
        for i in range(rings):
            for j in range(segments):
                a = i * (segments + 1) + j
                quads += [
                    [a, a + segments + 1, a + 1],
                    [a + 1, a + segments + 1, a + segments + 2],
                ]
        axes = np.stack([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]], 1)  # +X, up, gaze
        gaze = turned_axes(axes, [np.sin(np.radians(6)), -np.cos(np.radians(6)), 0.0])
        points = centre + radius * local @ gaze.T
        h, w = 256, 512
        v, u = np.mgrid[0:h, 0:w]
        angle = (v + 0.5) / h * 180.0  # from the gaze (the texture's v)
        image = np.full((h, w, 3), 230, np.uint8)
        image[angle < 25] = (110, 110, 100)
        image[angle < 9] = (5, 5, 5)
        part = HandoffPart("eye", "eyes", points, np.asarray(quads), uvs, image)
        target = Reference(
            points,
            np.asarray(quads),
            parts=[part],
            part_offsets=np.array([0, len(points)]),
        )
        frames = {"L": geometry.EyeFrame("L", centre, radius, axes)}
        colour, normal, out_axes = bake_eye_layout(target, frames)
        self.assertEqual(colour.shape, (512, 1024, 3))
        self.assertEqual(normal.shape, (512, 1024, 3))
        # the gaze turned toward the reference's pupil, within a degree
        turn = np.degrees(np.arccos(np.clip(out_axes["L"][:, 2] @ gaze[:, 2], -1, 1)))
        self.assertLess(turn, 1.0)
        # and the pupil sits on the gaze of the baked layout, with the iris round about it
        s, phi_, left = tex.texel_polar()
        hit = tex.find_iris(tex.linear(colour[left]), s[left], phi_[left])
        self.assertIsNotNone(hit)
        self.assertLess(np.hypot(*hit[0]), 0.02)
        self.assertAlmostEqual(hit[2], np.sin(np.radians(25)), delta=0.03)
        # the normals of a plain sphere are the sphere's own
        n = normal[left].astype(float) / 255.0 * 2.0 - 1.0
        front = s[left] < 0.9
        self.assertGreater(np.median(n[front, 2]), 0.99)
        # an eye frame turned to a recorded gaze looks along it
        frame = geometry.eye_frame("L", points, gaze[:, 2])
        np.testing.assert_allclose(frame.axes[:, 2], gaze[:, 2], atol=1e-9)
        # a part without a texture (a clear shell over the eye) gives no colour: where its
        # hits come nearest, the texels take the colour of the textured eye around them
        shell_points = centre + (radius + 0.0004) * local @ gaze.T
        cos = (local @ np.array([0.0, 0.0, 1.0]))[np.asarray(quads)].mean(1)
        band = np.asarray(quads)[(cos < np.cos(np.radians(40))) & (cos > 0.5)]
        shell = HandoffPart("shell", "eyes", shell_points, band, None, None)
        n = len(points)
        target = Reference(
            np.concatenate([points, shell_points]),
            np.concatenate([np.asarray(quads), band + n]),
            parts=[part, shell],
            part_offsets=np.array([0, n, 2 * n]),
        )
        frames = {"L": geometry.EyeFrame("L", centre, radius + 0.0005, gaze)}
        colour, _, _ = bake_eye_layout(target, frames, gaze=False)
        ring = left & (s > np.sin(np.radians(45))) & (s < np.sin(np.radians(55)))
        self.assertGreater(float(colour[ring].min()), 200)


if __name__ == "__main__":
    unittest.main()

# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the skin material (``Character.material``, ``opensculptboy.material``) and its use:
schema 3 cards and their backwards-compatible loading, the MikkTSpace port, the GLB skin
material (specular, IOR, subsurface extras, the thickness and curvature maps, the head tile and
its texture transform, the tangents), the Blender helper's reading of it, the VRM files with the
skin ramp and the head tile, the ramp itself, and the bake of the reference's own normal map in
matching tangent frames. The files of cards without the section stay as they were
(``TestGltfReference`` in ``test.test_gltf_export``).
"""

import json
import pathlib
import tempfile
import unittest
import warnings

import numpy as np
import torch

from opensculptboy import Character, export_glb
from opensculptboy.export.document import png_bytes
from opensculptboy.material import SkinMaterial, check_material
from opensculptboy.texture.tangents import corner_tangents, mikktspace
from test.gltf_reader import GLB, structure_errors

SKIN = {
    "roughness_factor": 0.9,
    "specular": 0.8,
    "specular_color": [1.0, 0.95, 0.9],
    "ior": 1.45,
    "normal_scale": 0.7,
    "subsurface": {
        "weight": 0.1,
        "radius_mm": [5.4, 2.1, 1.2],
        "ior": 1.4,
        "anisotropy": 0.0,
        "method": "random_walk_skin",
    },
    "thickness_mm": 25.0,
    "curvature_per_m": 80.0,
}


class TestCard(unittest.TestCase):
    def test_round_trip_and_schema(self):
        card = Character(name="skin", material=dict(SKIN, base_color="a.png"))
        data = json.loads(json.dumps(card.to_dict()))
        self.assertEqual(data["schema"], 3)
        self.assertEqual(Character.from_dict(data), card)
        skin = card.skin_material()
        self.assertEqual(skin.textures(), {"base_color": "a.png"})
        self.assertAlmostEqual(skin.roughness_value, 0.9 * 0.5)
        self.assertEqual(skin.subsurface["radius_mm"], [5.4, 2.1, 1.2])
        # without the section, the card keeps the schema it had
        self.assertEqual(Character(name="plain").to_dict()["schema"], 1)
        self.assertIsNone(Character().skin_material())
        detail = {"levels": [{"subdiv": 0, "offsets": "x.safetensors"}]}
        self.assertEqual(Character(detail=detail).to_dict()["schema"], 2)

    def test_defaults_and_head_tile(self):
        skin = SkinMaterial.from_dict(
            {"roughness": "r.png", "head": {"normal": "h.png"}}
        )
        self.assertEqual(skin.subsurface["method"], "random_walk_skin")
        self.assertEqual(skin.roughness_value, 1.0)  # the texture's multiplier
        self.assertEqual(
            skin.textures(), {"roughness": "r.png", "head_normal": "h.png"}
        )

    def test_invalid_material_is_refused(self):
        for material in (
            {"colour": [1, 1, 1]},
            {"base_color": [1, 1, 1, 1]},
            {"roughness": 2.0},
            {"specular": True},
            {"subsurface": {"weight": 1.5}},
            {"subsurface": {"radius_mm": [1, 2]}},
            {"subsurface": {"method": "christensen"}},
            {"subsurface": {"radius": [1, 1, 1]}},
            {"head": {"thickness": "t.png"}},
        ):
            with self.subTest(material=material), self.assertRaises(ValueError):
                check_material(material)
        with self.assertRaisesRegex(ValueError, "schema 3"):
            Character.from_dict({"schema": 2, "material": {"ior": 1.4}})
        with self.assertRaises(ValueError):
            Character.from_dict({"schema": 3, "material": {"ior": 0.5}})


def _grid(n=6, flip_u=False):
    """A flat grid in the XY plane with UVs along X and Y (U flipped on request)."""
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    positions = np.stack([x.ravel(), y.ravel(), np.zeros(n * n)], 1)
    uvs = positions[:, :2].copy()
    if flip_u:
        uvs[:, 0] = 1.0 - uvs[:, 0]
    quads = [
        (i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j)
        for i in range(n - 1)
        for j in range(n - 1)
    ]
    triangles = np.array(
        [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], np.int64
    )
    normals = np.tile([0.0, 0.0, 1.0], (n * n, 1))
    return positions, normals, uvs, triangles


class TestMikkTSpace(unittest.TestCase):
    def test_plane(self):
        positions, normals, uvs, triangles = _grid()
        tangents = mikktspace(positions, normals, uvs, triangles)
        np.testing.assert_allclose(
            tangents, np.tile([1.0, 0, 0, 1.0], (36, 1)), atol=1e-12
        )

    def test_mirrored_mapping_flips_the_sign(self):
        positions, normals, uvs, triangles = _grid(flip_u=True)
        tangents = mikktspace(positions, normals, uvs, triangles)
        np.testing.assert_allclose(
            tangents, np.tile([-1.0, 0, 0, -1.0], (36, 1)), atol=1e-12
        )

    def test_frame_invariance(self):
        import anny

        from opensculptboy.export.body import build_body

        model = anny.Anny().to(torch.float64)
        with torch.no_grad():
            rest = model()["rest_vertices"][0].numpy()
        body = build_body(model, rest)
        uv = np.stack([body.uvs[:, 0], 1.0 - body.uvs[:, 1]], 1)
        tangents = mikktspace(body.positions, body.normals, uv, body.triangles)
        used = np.unique(body.triangles)
        t = tangents[used]
        self.assertTrue(np.allclose(np.linalg.norm(t[:, :3], axis=1), 1.0))
        self.assertLess(np.abs((t[:, :3] * body.normals[used]).sum(1)).max(), 1e-9)
        self.assertTrue(set(np.unique(t[:, 3])) <= {-1.0, 1.0})
        # a rotation turns the tangents with the mesh; the direction of v flips the sign only
        angle = 0.7
        rotation = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        turned = mikktspace(
            body.positions @ rotation.T, body.normals @ rotation.T, uv, body.triangles
        )
        np.testing.assert_allclose(turned[used, :3], t[:, :3] @ rotation.T, atol=1e-9)
        flipped = mikktspace(
            body.positions, body.normals, body.uvs, body.triangles
        )  # glTF's v = 1 - v
        np.testing.assert_allclose(flipped[used, :3], t[:, :3], atol=1e-9)
        np.testing.assert_array_equal(flipped[used, 3], -t[:, 3])
        # anny's corners of one vertex share one group: one tangent per vertex
        _, group = corner_tangents(body.positions, body.normals, uv, body.triangles)
        groups_per_vertex = np.zeros(len(body.positions), int)
        pairs = np.unique(np.stack([body.triangles.ravel(), group.ravel()], 1), axis=0)
        np.add.at(groups_per_vertex, pairs[:, 0], 1)
        self.assertEqual(groups_per_vertex[used].max(), 1)


def _png(value, shape=(8, 8), channels=3):
    return png_bytes(
        np.full(shape + ((channels,) if channels > 1 else ()), value, np.uint8)
    )


class TestExports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.model = Character().build_model(dtype=torch.float64)
        cls.textures = {
            "base_color": _png(150),
            "normal": png_bytes(np.full((8, 8, 3), (128, 128, 255), np.uint8)),
            "roughness": _png(120, channels=1),
            "thickness": _png(255, channels=1),
            "curvature": _png(128, channels=1),
            "head_base_color": _png(160, (16, 8)),
            "head_normal": png_bytes(np.full((16, 8, 3), (128, 128, 255), np.uint8)),
            "head_roughness": _png(110, (16, 8), 1),
            "eyes": _png(40, (4, 8)),
        }
        cls.card = Character(name="skin", material=dict(SKIN))
        cls.path = cls.folder / "skin.glb"
        export_glb(cls.path, cls.card, cls.model, textures=cls.textures)
        cls.glb = GLB(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_structure(self):
        g = self.glb.json
        self.assertEqual(structure_errors(self.glb), [])
        for name in (
            "KHR_materials_specular",
            "KHR_materials_ior",
            "KHR_texture_transform",
        ):
            self.assertIn(name, g["extensionsUsed"])
        self.assertNotIn("extensionsRequired", g)
        names = [m["name"] for m in g["materials"]]
        self.assertEqual(names, ["skin", "head", "eyes"])
        skin = g["materials"][0]
        pbr = skin["pbrMetallicRoughness"]
        self.assertEqual(pbr["metallicFactor"], 0.0)
        self.assertAlmostEqual(pbr["roughnessFactor"], 0.9)
        self.assertIn("metallicRoughnessTexture", pbr)
        self.assertAlmostEqual(skin["normalTexture"]["scale"], 0.7)
        specular = skin["extensions"]["KHR_materials_specular"]
        self.assertAlmostEqual(specular["specularFactor"], 0.8)
        self.assertAlmostEqual(skin["extensions"]["KHR_materials_ior"]["ior"], 1.45)
        extras = skin["extras"]["opensculptboy_skin"]
        self.assertEqual(extras["format"], "opensculptboy/skin@1")
        self.assertEqual(extras["subsurface"]["radius_mm"], [5.4, 2.1, 1.2])
        self.assertEqual(extras["thickness"]["max_mm"], 25.0)
        self.assertEqual(extras["curvature"]["max_per_m"], 80.0)
        image_names = [
            g["images"][g["textures"][extras[k]["index"]]["source"]]["name"]
            for k in ("thickness", "curvature")
        ]
        self.assertEqual(image_names, ["thickness", "curvature"])
        # every image once
        self.assertEqual(len(g["images"]), len(self.textures))

    def test_head_tile(self):
        from opensculptboy.texture.tiles import head_texture_transform

        g = self.glb.json
        transform = head_texture_transform(self.model)
        head = g["materials"][1]
        for info in (
            head["pbrMetallicRoughness"]["baseColorTexture"],
            head["pbrMetallicRoughness"]["metallicRoughnessTexture"],
            head["normalTexture"],
        ):
            self.assertEqual(info["extensions"]["KHR_texture_transform"], transform)
        skin, tile, eyes = g["meshes"][0]["primitives"]
        self.assertEqual(skin["attributes"], tile["attributes"])
        indices = self.glb.accessor(tile["indices"]).reshape(-1, 3)
        uv = self.glb.accessor(tile["attributes"]["TEXCOORD_0"])[np.unique(indices)]
        mapped = uv * transform["scale"] + transform["offset"]
        self.assertTrue(((mapped >= 0) & (mapped <= 1)).all())
        # the head tile is the head's whole UV island: 2 triangles per quad of 3116 quads
        self.assertEqual(len(indices), 6232)
        counts = sum(
            len(self.glb.accessor(p["indices"])) // 3 for p in (skin, tile, eyes)
        )
        self.assertEqual(counts, 27420)

    def test_tangents(self):
        primitive = self.glb.json["meshes"][0]["primitives"][0]
        tangents = self.glb.accessor(primitive["attributes"]["TANGENT"])
        normals = self.glb.accessor(primitive["attributes"]["NORMAL"])
        self.assertTrue(
            np.allclose(np.linalg.norm(tangents[:, :3], axis=1), 1.0, atol=1e-5)
        )
        self.assertLess(np.abs((tangents[:, :3] * normals).sum(1)).max(), 1e-4)
        self.assertTrue(set(np.unique(tangents[:, 3])) <= {-1.0, 1.0})

    def test_blender_helper_reads_the_file(self):
        from opensculptboy.export.blender_skin import (
            principled_values,
            read_skin_settings,
        )

        settings = read_skin_settings(str(self.path))
        self.assertEqual(sorted(settings), ["head", "skin"])
        values = principled_values(settings["skin"])
        self.assertEqual(values["method"], "RANDOM_WALK_SKIN")
        inputs = values["inputs"]
        radius = np.array(inputs["Subsurface Radius"]) * inputs["Subsurface Scale"]
        np.testing.assert_allclose(radius, [0.0054, 0.0021, 0.0012])
        self.assertEqual(inputs["Subsurface Weight"], 0.1)

    def test_vrm_skin_ramp_and_head_tile(self):
        from opensculptboy.export.skin_ramp import skin_toon_ramp
        from opensculptboy.export.vrm import export_vrm
        from opensculptboy.texture.tiles import head_texture_transform

        ramp = skin_toon_ramp(self.card.skin_material())
        transform = head_texture_transform(self.model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for version in ("1.0", "0.x"):
                with self.subTest(version=version):
                    path = self.folder / f"skin{version}.vrm"
                    export_vrm(
                        path,
                        self.card,
                        self.model,
                        version=version,
                        author="tests",
                        refine_weights=False,
                        textures=self.textures,
                    )
                    glb = GLB(path)
                    self.assertEqual(structure_errors(glb), [])
                    g = glb.json
                    self.assertEqual(
                        [m["name"] for m in g["materials"]], ["skin", "eyes", "head"]
                    )
                    self.assertEqual(
                        [i["name"] for i in g["images"]][:5],
                        [
                            "base_color",
                            "normal",
                            "eyes",
                            "head_base_color",
                            "head_normal",
                        ],
                    )
                    primitive = g["meshes"][0]["primitives"][0]
                    self.assertIn("TANGENT", primitive["attributes"])
                    if version == "1.0":
                        mtoon = g["materials"][0]["extensions"]["VRMC_materials_mtoon"]
                        np.testing.assert_allclose(
                            mtoon["shadeColorFactor"], ramp.shade_ratio
                        )
                        self.assertAlmostEqual(
                            mtoon["shadingShiftFactor"], ramp.shading_shift
                        )
                        head = g["materials"][2]
                        info = head["pbrMetallicRoughness"]["baseColorTexture"]
                        self.assertEqual(
                            info["extensions"]["KHR_texture_transform"], transform
                        )


class TestSkinRamp(unittest.TestCase):
    def test_lambert_without_scattering(self):
        from opensculptboy.export.skin_ramp import toon_ramp

        ramp = toon_ramp((5.4, 2.1, 1.2), weight=0.0)
        self.assertEqual(ramp.shade_ratio, (0.0, 0.0, 0.0))
        self.assertLess(max(ramp.rms), 1e-9)  # MToon holds a Lambert ramp exactly

    def test_warm_with_scattering(self):
        from opensculptboy.export.skin_ramp import toon_ramp

        ramp = toon_ramp((5.4, 2.1, 1.2), weight=1.0, curvature_radius_mm=10.0)
        r, g, b = ramp.shade_ratio
        self.assertGreater(r, g)
        self.assertGreater(g, b)
        self.assertLess(max(ramp.rms), 0.06)


def _has_embree() -> bool:
    try:
        import trimesh.ray.ray_pyembree  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_has_embree(), "the texture bake needs embreex (the viewer extra)")
class TestNormalMapBake(unittest.TestCase):
    def test_the_reference_normal_map_survives_the_bake(self):
        # the reference is anny's own surface with a normal map that tilts every normal along
        # its tangent; baked onto the same surface in matching MikkTSpace frames, the map
        # comes back as it was, and the roughness with it
        import anny

        from opensculptboy.export.body import build_body, triangulated_faces
        from opensculptboy.fit.handoff import HandoffPart
        from opensculptboy.fit.reference import Reference
        from opensculptboy.texture.bake import bake
        from opensculptboy.texture.tiles import head_uv_box, on_head_tile

        model = anny.Anny().to(torch.float64)
        with torch.no_grad():
            rest = model()["rest_vertices"][0].numpy()
        body = build_body(model, rest, frame=np.eye(3))  # in Anny's frame, as the bake
        tilt = np.array([0.35, -0.2, 0.0])
        tilt[2] = np.sqrt(1.0 - (tilt[:2] ** 2).sum())
        encoded = np.round((tilt * 0.5 + 0.5) * 255).astype(np.uint8)
        part = HandoffPart(
            "skin",
            "skin",
            body.positions,
            body.triangles,
            body.uvs,
            np.full((64, 64, 3), 128, np.uint8),
            np.tile(encoded, (64, 64, 1)),
            normals=body.normals,  # the welded normals, as the file holds them
            roughness=np.full((64, 64), 0.4, np.float32),
        )
        reference = Reference(
            part.vertices,
            part.triangles,
            {},
            True,
            [part],
            np.array([0, len(part.vertices)]),
        )
        triangles, uv_triangles = triangulated_faces(model)
        uvs = model.texture_coordinates.double().numpy()
        size = 256
        baked = bake(
            rest,
            triangles,
            uv_triangles,
            uvs,
            reference,
            size=size,
            head_triangles=on_head_tile(model, uvs, uv_triangles),
            head_box=head_uv_box(model),
            head_size=128,
            skin_maps=True,
        )
        covered = baked.coverage
        decoded = baked.normal[covered].astype(float) / 127.5 - 1.0
        decoded /= np.linalg.norm(decoded, axis=1, keepdims=True)
        angle = np.degrees(np.arccos(np.clip(decoded @ tilt, -1.0, 1.0)))
        self.assertLess(np.median(angle), 1.0)
        self.assertLess(np.percentile(angle, 95), 3.0)
        self.assertLessEqual(
            np.abs(baked.roughness[covered].astype(int) - 102).max(), 1
        )
        self.assertEqual(baked.head["normal"].shape, (128, 64, 3))
        self.assertEqual(set(baked.head), {"base_color", "normal", "roughness"})
        self.assertEqual(baked.thickness.shape, (size, size))
        self.assertEqual(baked.curvature.shape, (size, size))
        material = baked.material({k: f"{k}.png" for k in baked.png_files()})
        check_material(material)
        self.assertEqual(material["head"]["normal"], "head_normal.png")


if __name__ == "__main__":
    unittest.main()


class TestEarTile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from opensculptboy.detail import DetailModel, save_offsets
        from opensculptboy.texture.ears import ear_triangles, ear_uv_table

        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.card = Character(name="ears", local_changes={"l-ear-trans-up": 0.0})
        cls.model = cls.card.build_model(dtype=torch.float64)
        detailed = DetailModel(cls.model, 1)
        with torch.no_grad():
            cls.rest = detailed()["rest_vertices"][0].numpy()
        cls.topology = detailed.topology
        cls.ears = ear_triangles(cls.model, cls.topology)
        cls.table = ear_uv_table(
            cls.rest, cls.topology, cls.ears, np.sign(cls.rest[:, 0])
        )
        cls.table.save(cls.folder / "ears.ear_uvs.safetensors")
        save_offsets(
            cls.folder / "ears.detail1.safetensors", np.zeros((len(cls.rest), 3)), 1
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_unwrap(self):
        from opensculptboy.texture.ears import EarUVs, density, with_ear_uvs

        uv = self.table.uv
        on = ~np.isnan(uv[:, 0])
        self.assertGreater(on.sum(), 1000)
        self.assertTrue(((uv[on] > 0) & (uv[on] < 1)).all())
        # the left ear (+X) in the left half, the right ear in the right half
        left = on & (self.rest[:, 0] > 0)
        self.assertTrue((uv[left, 0] < 0.5).all())
        self.assertTrue((uv[on & ~left, 0] > 0.5).all())
        stats = density(self.rest, self.topology, self.table)
        self.assertGreater(stats["p5"], 0.5 * stats["p50"])  # one texel size, about
        mesh, mask = with_ear_uvs(self.topology, self.table)
        self.assertGreaterEqual(mask.sum(), self.ears.sum())  # the closed set
        q = mesh.uvs[mesh.uv_triangles[mask]]
        signed = (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1]) - (
            q[:, 1, 1] - q[:, 0, 1]
        ) * (q[:, 2, 0] - q[:, 0, 0])
        self.assertLess((signed < 0).mean(), 0.005)
        loaded = EarUVs.load(self.folder / "ears.ear_uvs.safetensors")
        np.testing.assert_allclose(loaded.uv, uv, atol=1e-6)
        # level 0 keeps anny's UVs
        from opensculptboy.detail import detail_topology

        zero = detail_topology(self.model, 0)
        self.assertIsNone(with_ear_uvs(zero, self.table)[1])

    def test_pointed_ear_unwrap_has_no_flips(self):
        """A pointed tip is a cone far from developable: LSCM and ARAP flip triangles there
        (46 of 6,518 on this target, 179 of 25,968 on level 2 of the real replica), which
        overlap texels and turn the tangent frames over; ``untangle`` leaves none."""
        from opensculptboy.fit.register import pinna
        from opensculptboy.texture import ears as E
        from test.fit_fixtures import _smoothed, point_ears

        rest0 = self.model.template_vertices.detach().double().numpy()
        outer = np.asarray(self.topology.apply(pinna(self.model, rest0).astype(float)))
        weight = _smoothed(outer, self.topology.triangles, 12)
        pointed = point_ears(self.rest, weight, tip=0.02)

        def flips(table):
            mesh, mask = E.with_ear_uvs(self.topology, table)
            return int((E._signed_areas(mesh.uvs, mesh.uv_triangles[mask]) <= 0).sum())

        real = E.untangle
        try:
            E.untangle = lambda vertices, triangles, uv: uv
            plain = E.ear_uv_table(
                pointed, self.topology, self.ears, np.sign(pointed[:, 0])
            )
        finally:
            E.untangle = real
        table = E.ear_uv_table(
            pointed, self.topology, self.ears, np.sign(pointed[:, 0])
        )
        self.assertGreater(flips(plain), 10)  # the case the fix is for
        self.assertEqual(flips(table), 0)
        stats = E.density(pointed, self.topology, table)
        self.assertGreater(stats["p5"], 0.4 * stats["p50"])

    def test_export(self):
        card = Character(
            name="ears",
            detail={"levels": [{"subdiv": 1, "offsets": "ears.detail1.safetensors"}]},
            material={
                "subsurface": {"weight": 0.2},
                "ears": {
                    "uvs": "ears.ear_uvs.safetensors",
                    "base_color": "x.png",
                    "normal": "y.png",
                },
            },
        )
        textures = {
            "base_color": _png(150),
            "normal": png_bytes(np.full((8, 8, 3), (128, 128, 255), np.uint8)),
            "ears_base_color": _png(170, (8, 16)),
            "ears_normal": png_bytes(np.full((8, 16, 3), (128, 128, 255), np.uint8)),
            "ears_thickness": _png(30, (8, 16), 1),
        }
        path = self.folder / "ears.glb"
        export_glb(
            path,
            card,
            detail_level=1,
            detail_dir=self.folder,
            textures=textures,
        )
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        names = [m["name"] for m in glb.json["materials"]]
        self.assertEqual(names, ["skin", "ears"])
        ears = glb.json["materials"][1]
        self.assertNotIn("extensions", ears["pbrMetallicRoughness"]["baseColorTexture"])
        extras = ears["extras"]["opensculptboy_skin"]
        self.assertIn("thickness", extras)
        skin, tile = glb.json["meshes"][0]["primitives"]
        indices = glb.accessor(tile["indices"]).reshape(-1, 3)
        self.assertGreaterEqual(len(indices), self.ears.sum())
        uv = glb.accessor(tile["attributes"]["TEXCOORD_0"])[np.unique(indices)]
        self.assertTrue(((uv > 0) & (uv < 1)).all())
        self.assertEqual(
            glb.json["scenes"][0]["extras"]["opensculptboy"]["ear_tile"], True
        )

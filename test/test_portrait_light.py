# OpenSculptBoy
# Apache License, Version 2.0
"""
The light passes of a layered portrait pack (opensculptboy.portrait_pack.light, docs/portrait-pack.md).

In CI: the passes' names and sets, the rigs (the studio rig is the page's studio preset; the stage's
key comes from the frame's right and its rim from behind on the left), the encoding of a pass and its
precision, the colour bled around the figure, the display pass's tone curve, CIEDE2000 on the
published test pairs, the reproduction's report, and the check of a pack with passes (its files, its
fields and its bound on files). Local only (Playwright with the built page): the default character's
layered pack with passes at a fifth of its scale, its check, the sum of its passes against each
layer, and the key's side on the face.
"""

import json
import os
import pathlib
import re
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from opensculptboy.portrait_pack import contract, layered_contract as LC, layers, render
from opensculptboy.portrait_pack import light
from test.markers import local_only
from test.test_hair_page import CHROMIUM, PAGE, playwright_available
from test.test_portrait_layers import layered_manifest, run_cli, write_layered

MAIN_TS = pathlib.Path(__file__).resolve().parents[1] / "viewer" / "src" / "main.ts"


def lit_manifest():
    m = layered_manifest()
    m["passes"] = [
        {"name": "key", "scale": 2.5},
        {"name": "fill", "scale": 0.6},
        {"name": "rim", "scale": 1.2},
    ]
    m["light"] = light.describe(
        "stage",
        light.rig("stage"),
        {"exposure": 0.45, "tone": "agx", "saturation": 1.15, "contrast": 1.05},
        {"key": 0.3, "fill": 0.1, "rim": 0.01},
    )
    return m


class TestPassNames(unittest.TestCase):
    def test_sets(self):
        self.assertEqual(light.parse_passes("key,fill,rim"), ["key", "fill", "rim"])
        self.assertEqual(light.parse_passes("rim, key ,fill"), ["key", "fill", "rim"])
        self.assertEqual(light.parse_passes("base,rim"), ["base", "rim"])
        self.assertIsNone(light.parse_passes(""))
        for bad in ("key,rim", "key,fill,rim,base", "glow"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                light.parse_passes(bad)

    def test_pass_file_names(self):
        self.assertEqual(
            LC.pass_of("body.at_ease.close.key.webp"),
            ("body.at_ease.close.webp", "key"),
        )
        self.assertEqual(
            LC.pass_of("face.close.joy.open.half.rim.webp"),
            ("face.close.joy.open.half.webp", "rim"),
        )
        self.assertEqual(LC.pass_of("leap.peak.base.webp"), ("leap.peak.webp", "base"))
        for name in (
            "body.at_ease.close.webp",
            "body.at_ease.close.glow.webp",
            "body.at_ease.key.webp",
            "body.at_ease.close.key.png",
        ):
            with self.subTest(name=name):
                self.assertIsNone(LC.pass_of(name))
        self.assertTrue(LC.is_layer_name("front.chin.close.fill.webp"))


class TestRigs(unittest.TestCase):
    def test_studio_is_the_page_preset(self):
        """The studio rig repeats the viewer's PRESETS.studio, number for number."""
        text = MAIN_TS.read_text(encoding="utf-8")
        block = text[text.index("  studio: {") : text.index("  window: {")]
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", block)]
        s = light.STUDIO
        want = [s["exposure"], *s["sky"], *s["horizon"], *s["ground"]]
        for b in s["boxes"]:
            want += [b["az"], b["el"], b["w"], b["h"], *b["c"]]
        want += [s["key"][k] for k in ("az", "el", "size")] + s["key"]["c"]
        want += [s["rim"]["az"], s["rim"]["el"], *s["rim"]["c"]]
        want += [v for pair in s["bg"] for v in pair]
        # the comment above the preset holds numbers of its own; the preset's come last
        self.assertEqual(nums[-len(want) :], want)

    def test_stage_light_comes_from_the_open_side(self):
        st = light.rig("stage")
        self.assertEqual(light.key_side(st), "right")
        self.assertEqual(light.key_side(light.rig("studio")), "left")
        self.assertTrue(30 <= st["key"]["az"] <= 45)
        self.assertEqual(st["key"]["el"], 20)
        # the rim from behind, on the side away from the key
        self.assertGreater(abs(st["rim"]["az"]), 90)
        self.assertLess(st["rim"]["az"] * st["key"]["az"], 0)
        # the large fill box beside the camera, on the side away from the key
        fill = max(st["boxes"], key=lambda b: b["w"] * b["h"])
        self.assertLess(abs(fill["az"]), 30)
        self.assertLess(fill["az"] * st["key"]["az"], 0)
        # the same light as the studio's, turned: the exposure and the key's colour stay
        self.assertEqual(st["exposure"], light.STUDIO["exposure"])
        self.assertEqual(st["key"]["c"], light.STUDIO["key"]["c"])
        with self.assertRaises(ValueError):
            light.rig("disco")

    def test_description(self):
        d = lit_manifest()["light"]
        self.assertEqual(d["key_side"], "right")
        self.assertEqual(
            d["tone"], {"curve": "agx", "saturation": 1.15, "contrast": 1.05}
        )
        self.assertEqual(d["lights"]["key"]["az"], 40)
        self.assertEqual(LC.check_light({**lit_manifest()}), [])


class TestEncoding(unittest.TestCase):
    def test_round_trip_precision(self):
        """A pass keeps its light to about 1.2 % from a tenth of its scale up, 2.5 % from a fiftieth,
        and to a few thousandths of the scale below it."""
        scale = 3.0
        v = np.linspace(0, scale, 4001, dtype=np.float32)
        rgb = np.stack([v, v * 0.7, v * 0.4], -1)[None]
        alpha = np.full(rgb.shape[:2], 255, np.uint8)
        back = light.decode_pass(light.encode_pass(rgb, alpha, scale), scale)
        err = np.abs(back - rgb)
        high = rgb > scale / 50
        self.assertLess(float((err[high] / rgb[high]).max()), 0.025)
        top = rgb > scale / 10
        self.assertLess(float((err[top] / rgb[top]).max()), 0.012)
        self.assertLess(float(err[~high].max()), 0.0035 * scale)
        # light above the scale clips at the scale
        top = light.decode_pass(
            light.encode_pass(
                np.full((1, 1, 3), 2 * scale, np.float32), alpha[:1, :1], scale
            ),
            scale,
        )
        self.assertTrue(np.allclose(top, scale))

    def test_scale_and_alpha(self):
        rng = np.random.default_rng(1)
        rgb = rng.uniform(0, 2, (40, 40, 3)).astype(np.float32)
        alpha = np.zeros((40, 40), np.uint8)
        alpha[10:30, 10:30] = 255
        s = light.pass_scale(rgb, alpha)
        top = float(np.percentile(rgb[alpha > 0].max(-1), 99.9))
        self.assertAlmostEqual(s, top * light.SCALE_HEADROOM, places=5)
        enc = light.encode_pass(rgb, alpha, s)
        self.assertTrue(np.array_equal(enc[..., 3], alpha))
        self.assertEqual(light.pass_scale(rgb, np.zeros_like(alpha)), 1.0)

    def test_bleed(self):
        """The colour reaches a few pixels into the transparent surround, and no further."""
        rgb = np.zeros((20, 20, 3), np.float32)
        alpha = np.zeros((20, 20), np.uint8)
        rgb[8:12, 8:12] = (0.5, 0.25, 0.1)
        alpha[8:12, 8:12] = 255
        out = light.bleed(rgb, alpha)
        self.assertTrue(np.allclose(out[9, 8 - light.BLEED], (0.5, 0.25, 0.1)))
        self.assertTrue(np.allclose(out[10, 12 + light.BLEED - 1], (0.5, 0.25, 0.1)))
        self.assertTrue(np.allclose(out[0, 0], 0))
        self.assertTrue(np.allclose(out[10, 12 + light.BLEED + 1], 0))


class TestFaceLight(unittest.TestCase):
    def test_lit_side(self):
        """The face's light counts the lit side: the pixels where the key gives at least its median."""
        key = np.zeros((4, 4, 3), np.float32)
        key[:, 2:] = 1.0
        fill = np.full((4, 4, 3), 0.1, np.float32)
        fill[:, :2] = 0.3
        alpha = np.full((4, 4), 255, np.uint8)
        out = light.face_light(
            {"key": key, "fill": fill}, alpha, {"x": 0, "y": 0, "width": 4, "height": 4}
        )
        self.assertAlmostEqual(out["key"], 1.0, places=5)
        self.assertAlmostEqual(out["fill"], 0.1, places=5)
        self.assertEqual(
            light.face_light({"key": key}, np.zeros((4, 4), np.uint8), None),
            {"key": 0.0},
        )


class TestDisplay(unittest.TestCase):
    def test_agx(self):
        grey = light.agx(np.array([[0.18, 0.18, 0.18]]))[0]
        self.assertTrue(np.allclose(grey, grey[0], atol=2e-3), grey)
        self.assertTrue(0.15 < grey[0] < 0.25, grey)
        ramp = light.agx(np.linspace(0, 20, 200)[:, None].repeat(3, 1))[:, 1]
        self.assertTrue(np.all(np.diff(ramp) >= -1e-9))
        self.assertLessEqual(float(ramp.max()), 1.0)
        self.assertLess(float(ramp[0]), 0.01)

    def test_display_bytes(self):
        out = light.display8(
            np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]]), 1.15, 1.05
        )
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue((out[1] >= 250).all())
        self.assertTrue((out[0] <= 5).all())


class TestDeltaE(unittest.TestCase):
    def test_sharma_pairs(self):
        """Pairs from Sharma, Wu and Dalal's test data for CIEDE2000."""
        pairs = [
            ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
            ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
            ((50.0, 2.5, 0.0), (50.0, 0.0, -2.5), 4.3065),
            ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
            ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
            ((22.7233, 20.0904, -46.694), (23.0331, 14.973, -42.5619), 2.0373),
            ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
        ]
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        want = np.array([p[2] for p in pairs])
        self.assertTrue(np.allclose(light.delta_e2000_lab(a, b), want, atol=1e-3))
        same = np.array([[10, 200, 30]], np.uint8)
        self.assertAlmostEqual(float(light.delta_e2000(same, same)[0]), 0.0)

    def test_reproduction_report(self):
        rep = light.Reproduction()
        rendered = np.zeros((10, 10, 4), np.uint8)
        rendered[..., :3] = 120
        rendered[:5, :, 3] = 255
        summed = rendered[..., :3].copy()
        summed[0, 0] = 123
        rep.add("body", rendered, summed, rendered)
        r = rep.report()
        self.assertEqual(r["body"]["pixels"], 50)
        self.assertGreater(r["body"]["max"], 0.5)
        self.assertLess(r["body"]["mean"], r["body"]["max"])
        self.assertEqual(r["body"]["webp_mean"], 0.0)
        self.assertEqual(r["all"]["pixels"], 50)


class TestPassCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pack_with_passes(self):
        folder = write_layered(self.tmp / "p", lit_manifest())
        report = contract.check_pack(folder)
        self.assertTrue(report["pass"], report["problems"])
        # the layers of the plain pack (15) and three passes of each
        self.assertEqual(report["frames"], 15 * 4)
        self.assertTrue((folder / "face.close.neutral.half.open.rim.webp").is_file())

    def test_broken(self):
        def codes_of(change, m=None):
            folder = write_layered(
                self.tmp / f"b{len(list(self.tmp.iterdir()))}", m or lit_manifest()
            )
            change(folder)
            return sorted({p["code"] for p in contract.check_pack(folder)["problems"]})

        self.assertEqual(
            codes_of(lambda f: (f / "body.chin.close.fill.webp").unlink()),
            ["missing_frame"],
        )

        def wrong_size(f):
            layers.save_layer(
                f / "leap.peak.key.webp", np.zeros((5, 5, 4), np.uint8), 80
            )

        self.assertEqual(codes_of(wrong_size), ["wrong_size"])
        for change in (
            lambda m: m.pop("light"),
            lambda m: m.update(passes=[{"name": "key", "scale": 1}]),
            lambda m: m["passes"][0].update(scale=0),
            lambda m: m["light"].update(key_side="up"),
            lambda m: m["light"]["tone"].update(curve="aces"),
        ):
            m = lit_manifest()
            change(m)
            with self.subTest(m=m.get("passes")):
                problems = LC.check_layered_manifest(m)
                self.assertIn("bad_manifest", {p["code"] for p in problems}, m)

    def test_more_files_with_passes(self):
        """The bound on files grows with the passes: a pack with passes may hold four times the
        files of one without."""
        folder = write_layered(self.tmp / "big", lit_manifest())
        with mock.patch.object(LC, "MAX_FILES_V2", 20):
            self.assertTrue(contract.check_pack(folder)["pass"])
            folder2 = write_layered(self.tmp / "plain")
            for i in range(10):
                (folder2 / f"body.x{'abcdefghij'[i]}.close.webp").write_bytes(b"x")
            self.assertIn(
                "too_many_files",
                {p["code"] for p in contract.check_pack(folder2)["problems"]},
            )

    def test_command_options(self):
        code, _, err = run_cli(
            "portrait-pack", "--out", str(self.tmp / "x"), "--passes", "key,fill,rim"
        )
        self.assertEqual(code, 2)
        self.assertIn("take --layers", err)
        code, _, err = run_cli(
            "portrait-pack",
            "--layers",
            "--out",
            str(self.tmp / "y"),
            "--passes",
            "key,rim",
        )
        self.assertEqual(code, 2)
        self.assertIn("not a set", err)


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestDefaultPackWithPasses(unittest.TestCase):
    """The default character's layered pack with the light passes, at a fifth of its scale."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = pathlib.Path(tempfile.mkdtemp())
        cls.folder = cls.tmp / "pack"
        env = {render.CHROMIUM_ENV: str(CHROMIUM)} if CHROMIUM.exists() else {}
        with mock.patch.dict(os.environ, env):
            cls.result = run_cli(
                "portrait-pack",
                "--layers",
                "--out",
                str(cls.folder),
                "--scale",
                "0.2",
                "--samples",
                "2",
                "--expressions",
                "neutral,joy",
                "--poses",
                "at_ease,chin",
                "--framings",
                "close,full",
                "--passes",
                "key,fill,rim",
                "--page",
                str(PAGE),
                "--json",
            )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_pack_and_reproduction(self):
        code, out, err = self.result
        self.assertEqual(code, 0, err[-2000:])
        result = json.loads(out)
        self.assertTrue(result["check"]["pass"], result["check"]["problems"])
        m = result["manifest"]
        self.assertEqual([p["name"] for p in m["passes"]], ["key", "fill", "rim"])
        self.assertEqual(m["light"]["key_side"], "right")
        rep = result["light"]["reproduction"]
        self.assertLess(rep["all"]["mean"], 2.0, rep)
        for kind in ("body", "face", "front", "leap"):
            self.assertIn(kind, rep)

    def test_key_from_the_right(self):
        """The key pass lights the face's right half (the frame's) more than its left."""
        m = json.loads((self.folder / contract.MANIFEST).read_text())
        r = m["framings"]["close"]["face"]
        key = np.asarray(
            Image.open(self.folder / "body.at_ease.close.key.webp").convert("RGBA"),
            np.float32,
        )
        face = key[r["y"] : r["y"] + r["height"], r["x"] : r["x"] + r["width"]]
        lit = face[..., 3] > 250
        half = face.shape[1] // 2
        left = face[:, :half, :3][lit[:, :half]].mean()
        right = face[:, half:, :3][lit[:, half:]].mean()
        self.assertGreater(right, left * 1.1)


if __name__ == "__main__":
    unittest.main()

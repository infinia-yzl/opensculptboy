# OpenSculptBoy
# Apache License, Version 2.0
"""
The portrait pack (opensculptboy.portrait_pack, docs/portrait-pack.md).

In CI: the check with synthetic packs (a valid one, then each broken case with its reason code),
the manifest's rules, the faces of every frame, the folder rules and the command's exit codes.
Local only (Playwright with the built page): the default character's full pack and the pack of
--expressions neutral, with the frames' outline and the faces' changes. The full pack renders at
288 by 384 pixels with one pass per frame (about 20 minutes in software); set
OPENSCULPTBOY_PORTRAIT_PACK to a pack folder of the default character to check that pack instead.
"""

import contextlib
import io
import json
import os
import pathlib
import shutil
import struct
import tempfile
import unittest
import zlib
from unittest import mock

import numpy as np
from PIL import Image

from opensculptboy import cli
from opensculptboy.portrait_pack import contract, render
from opensculptboy.portrait_pack.contract import (
    EXPRESSIONS,
    EYES,
    MANIFEST,
    MOUTHS,
    SCHEMA,
    check_manifest,
    check_pack,
    frame_name,
    frame_names,
)
from opensculptboy.portrait_pack.faces import (
    EXPRESSION_MIXES,
    frame_actions,
    pack_faces,
)
from test.markers import local_only
from test.test_hair_page import CHROMIUM, PAGE, playwright_available

W, H = 12, 16


def manifest(expressions=("neutral",), w=W, h=H, **extra):
    m = {
        "schema": SCHEMA,
        "frame": {"width": w, "height": h},
        "anchor": {"x": w / 2, "y": h / 3},
        "baseline": h,
        "expressions": list(expressions),
        "generator": {"name": "OpenSculptBoy", "version": "test"},
        "licence": "CC0-1.0",
        "provenance": "A synthetic pack of the tests, rendered as the model is.",
    }
    m.update(extra)
    return m


def picture(w=W, h=H, mode="RGBA"):
    """A figure on a transparent background."""
    a = np.zeros((h, w, 4), np.uint8)
    a[h // 4 :, w // 4 : 3 * w // 4] = (200, 150, 120, 255)
    im = Image.fromarray(a, "RGBA")
    return im if mode == "RGBA" else im.convert(mode)


def write_pack(folder, expressions=("neutral",), **extra):
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in frame_names(expressions):
        picture().save(folder / name)
    (folder / MANIFEST).write_text(json.dumps(manifest(expressions, **extra)))
    return folder


def interlace(path):
    """Set the interlace method of a PNG's header to Adam7 (the check reads the header)."""
    data = bytearray(pathlib.Path(path).read_bytes())
    data[28] = 1
    crc = zlib.crc32(bytes(data[12:29])) & 0xFFFFFFFF
    data[29:33] = struct.pack(">I", crc)
    pathlib.Path(path).write_bytes(bytes(data))


def codes(report):
    return {p["code"] for p in report["problems"]}


def run_cli(*args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(list(args))
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), err.getvalue()


class TestCheck(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def pack(self, expressions=("neutral",), **extra):
        return write_pack(self.dir / "pack", expressions, **extra)

    def test_valid_packs(self):
        for xs in (("neutral",), EXPRESSIONS):
            with self.subTest(expressions=len(xs)):
                folder = write_pack(self.dir / str(len(xs)), xs)
                report = check_pack(folder)
                self.assertTrue(report["pass"], report["problems"])
                self.assertEqual(report["frames"], 9 * len(xs))
        self.assertEqual(len(frame_names(EXPRESSIONS)), 90)

    def test_broken_cases(self):
        def extra_file(f):
            (f / "notes.txt").write_text("x")

        def stray_frame(f):  # a frame of an expression the manifest does not list
            picture().save(f / "smile.open.open.png")

        def missing_frame(f):
            (f / "neutral.half.closed.png").unlink()

        def wrong_size(f):
            picture(W + 2, H).save(f / "neutral.open.open.png")

        def rgb(f):
            picture(mode="RGB").save(f / "neutral.open.open.png")

        def grey_alpha(f):
            picture(mode="LA").save(f / "neutral.open.open.png")

        def interlaced(f):
            interlace(f / "neutral.open.open.png")

        def not_png(f):
            (f / "neutral.open.open.png").write_bytes(b"not a picture")

        def unknown_field(f):
            (f / MANIFEST).write_text(json.dumps(manifest(colour="red")))

        def unknown_inner_field(f):
            m = manifest()
            m["frame"]["depth"] = 8
            (f / MANIFEST).write_text(json.dumps(m))

        def subfolder(f):
            (f / "more").mkdir()

        def too_many(f):
            for i in range(95):
                (f / f"x{i}.png").write_bytes(b"")

        def too_large_frame(f):
            m = manifest(w=2048, h=H)
            (f / MANIFEST).write_text(json.dumps(m))

        def no_manifest(f):
            (f / MANIFEST).unlink()

        def not_json(f):
            (f / MANIFEST).write_text("{")

        cases = {
            "extra_file": [extra_file, stray_frame],
            "missing_frame": [missing_frame],
            "wrong_size": [wrong_size],
            "not_rgba": [rgb, grey_alpha, not_png],
            "interlaced": [interlaced],
            "unknown_field": [unknown_field, unknown_inner_field],
            "subfolder": [subfolder],
            "too_many_files": [too_many],
            "too_large": [too_large_frame],
            "bad_manifest": [no_manifest, not_json],
        }
        self.assertEqual(set(cases), set(contract.CODES))
        for code, breakers in cases.items():
            for breaker in breakers:
                with self.subTest(code=code, case=breaker.__name__):
                    folder = write_pack(self.dir / breaker.__name__)
                    breaker(folder)
                    report = check_pack(folder)
                    self.assertFalse(report["pass"])
                    self.assertIn(code, codes(report), report["problems"])

    def test_total_size(self):
        folder = self.pack()
        self.assertEqual(contract.MAX_PACK_BYTES, 128 * 2**20)
        with mock.patch.object(contract, "MAX_PACK_BYTES", 100):
            self.assertIn("too_large", codes(check_pack(folder)))
        (folder / MANIFEST).write_text(
            json.dumps(manifest(provenance="x" * 1900)) + " " * 70000
        )
        self.assertIn("too_large", codes(check_pack(folder)))

    def test_folder_errors(self):
        with self.assertRaises(FileNotFoundError):
            check_pack(self.dir / "none")
        (self.dir / "file").write_text("x")
        with self.assertRaises(NotADirectoryError):
            check_pack(self.dir / "file")


class TestManifest(unittest.TestCase):
    def problems(self, **change):
        m = manifest(**{k: v for k, v in change.items() if v is not None})
        for k, v in change.items():
            if v is None:
                del m[k]
        return check_manifest(m)

    def assertBad(self, code="bad_manifest", **change):
        got = {p["code"] for p in self.problems(**change)}
        self.assertIn(code, got, change)

    def test_valid(self):
        self.assertEqual(check_manifest(manifest()), [])
        self.assertEqual(check_manifest(manifest(companion="Ada")), [])
        self.assertEqual(check_manifest(manifest(EXPRESSIONS)), [])

    def test_rules(self):
        self.assertBad(schema=SCHEMA.replace("/1", "/2"))
        for field in contract.REQUIRED_FIELDS:
            with self.subTest(missing=field):
                self.assertBad(**{field: None})
        self.assertBad(companion="")
        self.assertBad(companion="x" * 65)
        self.assertBad(licence="")
        self.assertBad(licence="x" * 501)
        self.assertBad(provenance="x" * 2001)
        self.assertBad(frame={"width": 0, "height": 10})
        self.assertBad(frame={"width": 10.5, "height": 10})
        self.assertBad(frame={"width": True, "height": 10})
        self.assertBad("too_large", frame={"width": 1025, "height": 10})
        self.assertBad(anchor={"x": 1, "y": H + 1})
        self.assertBad(anchor={"x": "1", "y": 2})
        self.assertBad(baseline=H + 1)
        self.assertBad(baseline=H / 3)
        self.assertBad(expressions=[])
        self.assertBad(expressions=["smile", "neutral"])
        self.assertBad(expressions=["neutral", "neutral"])
        self.assertBad(expressions=["neutral", "grin"])
        self.assertBad(generator={"name": "OpenSculptBoy"})
        self.assertBad(
            "unknown_field", generator={"name": "x", "version": "1", "build": 2}
        )
        self.assertBad("unknown_field", anchor={"x": 1, "y": 2, "z": 3})

    def test_text_fields(self):
        for text in ("two\nlines", "a\ttab", "bell\x07", "line separator", "\x85next"):
            for field in ("companion", "licence", "provenance"):
                with self.subTest(field=field, text=repr(text)):
                    self.assertBad(**{field: text})
        self.assertBad(generator={"name": "Open\nSculptBoy", "version": "1"})
        self.assertEqual(check_manifest(manifest(companion="Ünïcödé 名前")), [])
        self.assertTrue(check_manifest([]))


class TestFaces(unittest.TestCase):
    def test_actions(self):
        from anny.models.facial_actions import FACIAL_ACTION_LABELS

        names = set(FACIAL_ACTION_LABELS)
        self.assertEqual(len(names), 52)
        self.assertEqual(set(EXPRESSION_MIXES), set(EXPRESSIONS))
        faces = pack_faces()
        self.assertEqual(len(faces), 90)
        self.assertEqual([frame_name(*f[:3]) for f in faces], frame_names(EXPRESSIONS))
        for x, m, e, a in faces:
            with self.subTest(frame=frame_name(x, m, e)):
                self.assertLessEqual(set(a), names)
                self.assertTrue(all(0 < v <= 1 for v in a.values()), a)

    def test_mouths_and_eyes_differ(self):
        mouth = ("jawOpen", "mouthLowerDownLeft", "mouthFunnel")
        for x in EXPRESSIONS:
            for e in EYES:
                with self.subTest(expression=x, eyes=e):
                    jaw = [frame_actions(x, m, e).get("jawOpen", 0) for m in MOUTHS]
                    self.assertLess(jaw[0], jaw[1])
                    self.assertLess(jaw[1], jaw[2])
                    # the eyes leave the mouth as it is
                    a = frame_actions(x, "half", "open")
                    b = frame_actions(x, "half", e)
                    self.assertEqual(
                        [a.get(k) for k in mouth], [b.get(k) for k in mouth]
                    )
            for m in MOUTHS:
                blink = [frame_actions(x, m, e).get("eyeBlinkLeft", 0) for e in EYES]
                self.assertLess(blink[0], blink[1])
                self.assertLess(blink[1], blink[2])
                self.assertEqual(blink[2], 1.0)
                for e in EYES:
                    f = frame_actions(x, m, e)
                    self.assertEqual(f.get("eyeBlinkLeft"), f.get("eyeBlinkRight"))

    def test_presets(self):
        # the presets scaled by the expression's weights, the vowels and the blink on top
        self.assertEqual(frame_actions("neutral", "closed", "open"), {})
        self.assertEqual(
            frame_actions("neutral", "closed", "closed"),
            {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0},
        )
        smile = frame_actions("smile", "closed", "open")
        self.assertAlmostEqual(smile["mouthSmileLeft"], 0.8 * 0.6)
        half = frame_actions("neutral", "half", "half")
        self.assertAlmostEqual(half["jawOpen"], 0.42 * 0.4 + 0.06 * 0.2)
        self.assertAlmostEqual(half["eyeBlinkLeft"], 0.5)
        up = frame_actions("thinking", "closed", "open")
        self.assertAlmostEqual(up["eyeLookUpLeft"], 0.35)
        # the blink takes over the widened eyes
        self.assertNotIn("eyeWideLeft", frame_actions("surprised", "closed", "closed"))
        drowsy = frame_actions("drowsy", "closed", "open")["eyeBlinkLeft"]
        self.assertAlmostEqual(drowsy, 0.35 * 0.8 + 0.4)
        with self.assertRaises(ValueError):
            frame_actions("grin", "closed", "open")


class TestCommand(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_check_command(self):
        folder = write_pack(self.dir / "pack")
        code, out, _ = run_cli("portrait-pack", "--check", str(folder), "--json")
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out)["pass"])
        (folder / "neutral.open.open.png").unlink()
        code, out, _ = run_cli("portrait-pack", "--check", str(folder), "--json")
        self.assertEqual(code, 1)
        self.assertIn("missing_frame", codes(json.loads(out)))
        code, out, _ = run_cli("portrait-pack", "--check", str(folder))
        self.assertEqual(code, 1)
        self.assertIn("missing_frame: ", out)
        code, _, _ = run_cli("portrait-pack", "--check", str(self.dir / "none"))
        self.assertEqual(code, 2)

    def test_usage(self):
        card = self.dir / "card.json"
        card.write_text(json.dumps({"name": "Synthetic", "phenotype": {"muscle": 0.7}}))
        out = self.dir / "out"
        for args in (
            ["portrait-pack"],
            ["portrait-pack", "--out", str(out), "--size", "2000x10"],
            ["portrait-pack", "--out", str(out), "--expressions", "neutral,grin"],
            ["portrait-pack", "--out", str(out), "--samples", "0"],
            # a card holds no licence of its pictures
            ["portrait-pack", "--out", str(out), "--character", str(card)],
            ["portrait-pack", "--out", str(out), "--licence", "two\nlines"],
            ["portrait-pack", "--check", str(out), "--out", str(out)],
        ):
            with self.subTest(args=args[1:]):
                code, _, err = run_cli(*args)
                self.assertEqual(code, 2, err)
        self.assertFalse(out.exists())

    def test_parse(self):
        self.assertEqual(render.parse_size("768x1024"), (768, 1024))
        self.assertEqual(
            render.parse_expressions("joy, smile"), ["neutral", "smile", "joy"]
        )
        self.assertEqual(render.parse_expressions(None), list(EXPRESSIONS))

    def test_chromium_args(self):
        # the graphics card first, by platform; SwiftShader as the fallback
        self.assertIn("--use-angle=d3d11", render.gpu_args("win32"))
        self.assertIn("--use-angle=metal", render.gpu_args("darwin"))
        self.assertIn("--enable-gpu", render.gpu_args("linux"))
        for args in (render.gpu_args("win32"), render.gpu_args("linux")):
            self.assertNotIn("--enable-unsafe-swiftshader", args)
        self.assertIn("--use-angle=swiftshader", render.SOFTWARE_ARGS)
        self.assertTrue(
            render.is_software("ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device))")
        )
        self.assertFalse(
            render.is_software("ANGLE (NVIDIA, Direct3D11 vs_5_0 ps_5_0, D3D11)")
        )

    def test_folder(self):
        # an empty or new folder, or a previous pack (emptied first); never a folder of other files
        render.prepare_folder(self.dir / "new")
        old = write_pack(self.dir / "old", EXPRESSIONS)
        render.prepare_folder(old)
        self.assertEqual(list(old.iterdir()), [])
        other = self.dir / "other"
        other.mkdir()
        (other / "keep.txt").write_text("x")
        with self.assertRaises(ValueError):
            render.prepare_folder(other)
        self.assertTrue((other / "keep.txt").exists())

    def test_card_files(self):
        (self.dir / "maps").mkdir()
        (self.dir / "maps" / "skin.png").write_bytes(b"")
        (self.dir / "detail.safetensors").write_bytes(b"")
        (self.dir / "shirt.wearable.json").write_text(
            json.dumps({"garment": "shirt.bin"})
        )
        (self.dir / "shirt.bin").write_bytes(b"")
        card = self.dir / "card.json"
        card.write_text(
            json.dumps(
                {
                    "name": "x.json",
                    "detail": {
                        "levels": [{"subdiv": 0, "offsets": "detail.safetensors"}]
                    },
                    "material": {"textures": {"base": "maps/skin.png"}},
                    "wearables": [{"id": "shirt", "file": "shirt.wearable.json"}],
                    "look": {"hair": {"style": "bob.json"}},
                }
            )
        )
        names = sorted(
            p.relative_to(self.dir).as_posix() for p in render.card_files(card)
        )
        self.assertEqual(
            names,
            [
                "card.json",
                "detail.safetensors",
                "maps/skin.png",
                "shirt.bin",
                "shirt.wearable.json",
            ],
        )
        (self.dir / "shirt.bin").unlink()
        with self.assertRaises(FileNotFoundError):
            render.card_files(card)


def alpha_box(a):
    ys, xs = np.nonzero(a[..., 3])
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()])


def regions(manifest):
    """The eye and mouth boxes (x0, y0, x1, y1) around the anchor, in pixels: the frame is 2.5
    heads high (the chin to the crown, PORTRAIT_BUST in viewer/src/main.ts)."""
    h = manifest["frame"]["height"] / 2.5
    x, y = manifest["anchor"]["x"], manifest["anchor"]["y"]
    eyes = (x - 0.4 * h, y - 0.12 * h, x + 0.4 * h, y + 0.1 * h)
    mouth = (x - 0.3 * h, y + 0.2 * h, x + 0.3 * h, y + 0.6 * h)
    face = (x - 0.6 * h, y - 0.6 * h, x + 0.6 * h, y + 0.95 * h)
    return [tuple(int(round(v)) for v in r) for r in (eyes, mouth, face)]


def crop(a, box):
    x0, y0, x1, y1 = box
    return a[y0:y1, x0:x1].astype(np.float32)


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestDefaultPack(unittest.TestCase):
    """The default character's full pack, rendered by the page."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = pathlib.Path(tempfile.mkdtemp())
        given = os.environ.get("OPENSCULPTBOY_PORTRAIT_PACK")
        if given:
            cls.folder = pathlib.Path(given)
            return
        cls.folder = cls.tmp / "pack"
        env = {render.CHROMIUM_ENV: str(CHROMIUM)} if CHROMIUM.exists() else {}
        with mock.patch.dict(os.environ, env):
            code, out, err = run_cli(
                "portrait-pack",
                "--out",
                str(cls.folder),
                "--size",
                "288x384",
                "--samples",
                "1",
                "--page",
                str(PAGE),
                "--json",
            )
        cls.result = (code, out, err)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def frames(self):
        return {
            n: np.asarray(Image.open(self.folder / n)) for n in frame_names(EXPRESSIONS)
        }

    def test_pack(self):
        report = check_pack(self.folder)
        self.assertTrue(report["pass"], report["problems"])
        self.assertEqual(report["frames"], 90)
        m = json.loads((self.folder / MANIFEST).read_text())
        self.assertEqual(m["expressions"], list(EXPRESSIONS))
        self.assertEqual(m["baseline"], m["frame"]["height"])
        self.assertNotIn("companion", m)
        if hasattr(self, "result"):
            self.assertEqual(self.result[0], 0, self.result[2][-2000:])

    def test_frames(self):
        m = json.loads((self.folder / MANIFEST).read_text())
        frames = self.frames()
        eyes, mouth, face = regions(m)
        base = frames[frame_name("neutral", "closed", "open")]
        h, w = base.shape[:2]
        self.assertEqual((w, h), (m["frame"]["width"], m["frame"]["height"]))
        box = alpha_box(base)
        # a bust: transparent above the head, the figure cut by the bottom edge
        self.assertGreater(box[1], 0)
        self.assertEqual(box[3], h - 1)
        self.assertEqual(base[0, 0, 3], 0)
        outside = np.ones((h, w), bool)
        x0, y0, x1, y1 = face
        outside[max(0, y0) : y1, max(0, x0) : x1] = False
        for name, a in frames.items():
            with self.subTest(frame=name):
                self.assertEqual(a.shape, base.shape)
                self.assertLessEqual(np.abs(alpha_box(a) - box).max(), 1)
                # apart from the face, the outline stays
                moved = np.abs(a[..., 3].astype(int) - base[..., 3].astype(int)) > 32
                self.assertLess((moved & outside).mean(), 0.001)
        for x in EXPRESSIONS:
            for e in EYES:
                with self.subTest(expression=x, eyes=e, part="mouth"):
                    pics = [crop(frames[frame_name(x, mo, e)], mouth) for mo in MOUTHS]
                    for i in range(3):
                        for j in range(i + 1, 3):
                            self.assertGreater(np.abs(pics[i] - pics[j]).mean(), 1.0)
            for mo in MOUTHS:
                with self.subTest(expression=x, mouth=mo, part="eyes"):
                    pics = [crop(frames[frame_name(x, mo, e)], eyes) for e in EYES]
                    for i in range(3):
                        for j in range(i + 1, 3):
                            self.assertGreater(np.abs(pics[i] - pics[j]).mean(), 1.0)


@local_only("Playwright and Chromium")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestLaunch(unittest.TestCase):
    def test_software_fallback(self):
        # Chromium without WebGL 2 on the graphics card's arguments relaunches in software
        from playwright.sync_api import sync_playwright

        logs = []
        page = render.PortraitPage(log=logs.append)
        env = {render.CHROMIUM_ENV: str(CHROMIUM)} if CHROMIUM.exists() else {}
        no_webgl = ["--disable-gpu", "--disable-webgl"]
        with (
            mock.patch.dict(os.environ, env),
            mock.patch.object(render, "gpu_args", lambda: no_webgl),
        ):
            with sync_playwright() as p:
                page._launch(p).close()
        self.assertTrue(page.renderer)
        self.assertTrue(page.software)
        self.assertIn("no WebGL 2", logs[0])
        self.assertIn("Chromium with software", logs[1])


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestNeutralPack(unittest.TestCase):
    def test_neutral_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp) / "pack"
            env = {render.CHROMIUM_ENV: str(CHROMIUM)} if CHROMIUM.exists() else {}
            with mock.patch.dict(os.environ, env):
                code, out, err = run_cli(
                    "portrait-pack",
                    "--out",
                    str(folder),
                    "--expressions",
                    "neutral",
                    "--size",
                    "192x256",
                    "--samples",
                    "1",
                    "--companion",
                    "Test",
                    "--page",
                    str(PAGE),
                    "--json",
                )
            self.assertEqual(code, 0, err[-2000:])
            result = json.loads(out)
            self.assertTrue(result["check"]["pass"])
            self.assertEqual(
                sorted(p.name for p in folder.iterdir()),
                sorted([MANIFEST, *frame_names(["neutral"])]),
            )
            self.assertTrue(result["renderer"])
            m = json.loads((folder / MANIFEST).read_text())
            self.assertEqual(m["companion"], "Test")
            self.assertEqual(m["expressions"], ["neutral"])
            # a second run writes over the previous pack
            with mock.patch.dict(os.environ, env):
                code, _, err = run_cli(
                    "portrait-pack",
                    "--out",
                    str(folder),
                    "--expressions",
                    "neutral",
                    "--size",
                    "96x128",
                    "--samples",
                    "1",
                    "--page",
                    str(PAGE),
                )
            self.assertEqual(code, 0, err[-2000:])
            self.assertTrue(check_pack(folder)["pass"])
            self.assertEqual(
                Image.open(folder / "neutral.open.open.png").size, (96, 128)
            )


if __name__ == "__main__":
    unittest.main()

# OpenSculptBoy
# Apache License, Version 2.0
"""
The layered portrait pack (squire.portraits/2, opensculptboy.portrait_pack.layers, docs/portrait-pack.md).

In CI: the check with synthetic layered packs (a valid one, then each broken case with its reason
code), the WebP header reader, the framings' sizes and cameras from synthetic outlines, the front
layer's mask, the shared pose file (its format, every humanoid bone resolving to a rig bone, the
T-pose turns, the stills' skeleton landing on the VRM file's skeleton, the head kept), the shots'
cameras and the command's options. Local only (Playwright with the built page): the default
character's layered pack at a fifth of its scale with one pass per layer, its check, the head on
the same pixels in every pose of a framing and the face layers fitting the bodies.
"""

import contextlib
import io
import json
import os
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from opensculptboy import cli
from opensculptboy.portrait_pack import contract, layered_contract as LC, layers, render
from opensculptboy.portrait_pack import poses as stage_poses
from opensculptboy.portrait_pack import shared_poses, shots
from test.markers import local_only
from test.test_hair_page import CHROMIUM, PAGE, playwright_available


def layered_manifest(fmt="webp"):
    return {
        "schema": LC.SCHEMA_V2,
        "format": fmt,
        "expressions": ["neutral"],
        "framings": {
            "close": {
                "frame": {"width": 24, "height": 32},
                "head": {"x": 12, "y": 10, "top": 4, "chin": 16},
                "face": {"x": 8, "y": 8, "width": 8, "height": 8},
            },
            "full": {
                "frame": {"width": 16, "height": 40},
                "head": {"x": 8, "y": 6, "top": 3, "chin": 9},
                "ground": 38,
            },
        },
        "poses": {
            "at_ease": {"framings": ["close", "full"]},
            "chin": {"framings": ["close"], "front": ["close"]},
        },
        "leap": {"framing": "full", "frames": ["crouch", "peak"]},
        "extra": {"above": 0.3, "width": 0},
        "generator": {"name": "OpenSculptBoy", "version": "test"},
        "licence": "CC0-1.0",
        "provenance": "Synthetic layers for tests; no model.",
    }


def write_layered(folder, m=None):
    m = m or layered_manifest()
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / contract.MANIFEST).write_text(json.dumps(m), encoding="utf-8")
    for name, (w, h) in LC.wanted_layers(m).items():
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = (200, 150, 120, 255)
        layers.save_layer(folder / name, rgba, 80 if name.endswith(".webp") else None)
    return folder


def codes(report):
    return sorted({p["code"] for p in report["problems"]})


def run_cli(*args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(list(args))
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), err.getvalue()


class TestLayeredCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_valid_pack_passes_in_both_formats(self):
        for fmt in LC.FORMATS:
            with self.subTest(fmt=fmt):
                folder = write_layered(self.tmp / fmt, layered_manifest(fmt))
                report = contract.check_pack(folder)
                self.assertTrue(report["pass"], report["problems"])
                # bodies 3, the front 1, faces 9 at close, leap 2
                self.assertEqual(report["frames"], 3 + 1 + 9 + 2)

    def test_webp_header(self):
        folder = write_layered(self.tmp / "p")
        head = LC.webp_header((folder / "body.at_ease.close.webp").read_bytes()[:64])
        self.assertEqual(head, {"width": 24, "height": 32, "alpha": True})
        self.assertIsNone(LC.webp_header(b"RIFF\0\0\0\0WAVEfmt " + b"\0" * 20))

    def test_broken_packs(self):
        def broken(change):
            folder = write_layered(self.tmp / f"b{len(list(self.tmp.iterdir()))}")
            change(folder)
            return codes(contract.check_pack(folder))

        self.assertEqual(
            broken(lambda f: (f / "face.close.neutral.half.open.webp").unlink()),
            ["missing_frame"],
        )
        self.assertEqual(
            broken(lambda f: (f / "Thumbs.db").write_bytes(b"x")), ["extra_file"]
        )
        self.assertEqual(broken(lambda f: (f / "sub").mkdir()), ["subfolder"])

        def wrong_size(f):
            layers.save_layer(
                f / "body.chin.close.webp", np.zeros((10, 10, 4), np.uint8), 80
            )

        self.assertEqual(broken(wrong_size), ["wrong_size"])

        def opaque(f):
            Image.new("RGB", (24, 32)).save(
                f / "body.chin.close.webp", format="WEBP", quality=80
            )

        self.assertEqual(broken(opaque), ["not_rgba"])

        def manifest_change(fn):
            def change(f):
                m = json.loads((f / contract.MANIFEST).read_text())
                fn(m)
                (f / contract.MANIFEST).write_text(json.dumps(m))

            return change

        self.assertIn(
            "unknown_field", broken(manifest_change(lambda m: m.update(colour="red")))
        )
        self.assertIn(
            "bad_manifest", broken(manifest_change(lambda m: m["poses"].pop("at_ease")))
        )
        self.assertIn(
            "bad_manifest",
            broken(
                manifest_change(
                    lambda m: m["poses"]["at_ease"].update(framings=["close"])
                )
            ),
        )
        self.assertIn(
            "bad_manifest",
            broken(
                manifest_change(lambda m: m["framings"]["close"]["head"].update(chin=5))
            ),
        )
        self.assertIn(
            "bad_manifest",
            broken(
                manifest_change(
                    lambda m: m["framings"].update(bust=m["framings"]["close"])
                )
            ),
        )
        self.assertIn(
            "bad_manifest",
            broken(
                manifest_change(lambda m: m["leap"].update(frames=["peak", "crouch"]))
            ),
        )
        self.assertIn(
            "too_large",
            broken(
                manifest_change(
                    lambda m: m["framings"]["full"]["frame"].update(width=5000)
                )
            ),
        )

    def test_layer_names(self):
        self.assertEqual(
            LC.layer_kind("body.chin.close.webp"), ("body", "chin", "close", "webp")
        )
        self.assertEqual(
            LC.layer_kind("face.thigh.joy.open.half.png"),
            ("face", "thigh", "joy", "open", "half", "png"),
        )
        self.assertEqual(LC.layer_kind("leap.peak.webp"), ("leap", "peak", "webp"))
        # a pose's own face set, and the framings of a single shot
        self.assertEqual(
            LC.layer_kind("face.thought.chin.thinking.half.open.webp"),
            ("face", "thought", "chin", "thinking", "half", "open", "webp"),
        )
        self.assertEqual(
            LC.layer_kind("body.downcast.low.webp"), ("body", "downcast", "low", "webp")
        )
        for bad in (
            "face.thought.Chin.thinking.half.open.webp",
            "face.close.chin.grin.half.open.webp",
            "body.chin.bust.webp",
            "face.close.joy.open.webp",
            "leap.top.webp",
            "body.Chin.close.webp",
            "neutral.closed.open.png",
        ):
            self.assertIsNone(LC.layer_kind(bad), bad)

    def test_shot_framings_and_face_sets(self):
        """The framings of the shots: a framing through each shot's camera, recorded in the
        manifest, and a face set of its own for a pose that turns its head."""
        m = layered_manifest()
        m["expressions"] = ["neutral", "stern", "thinking"]
        cam = {
            "shot": "thought",
            "lens": 85,
            "height": "eyes",
            "above": 0,
            "tilt": 0,
            "turn": 25,
        }
        m["framings"]["close"]["head"]["size"] = 10.4
        m["framings"]["close"]["camera"] = {**cam, "shot": "close_up", "turn": 20}
        m["framings"]["thought"] = {
            "frame": {"width": 24, "height": 32},
            "head": {"x": 12, "y": 10, "top": 4, "chin": 16, "size": 10.4},
            "camera": cam,
            "sets": {
                "chin": {
                    "face": {"x": 8, "y": 8, "width": 8, "height": 8},
                    "expressions": ["neutral", "thinking"],
                }
            },
        }
        m["framings"]["stern"] = {
            "frame": {"width": 24, "height": 32},
            "head": {"x": 12, "y": 10, "top": 4, "chin": 16},
            "camera": {
                **cam,
                "shot": "close_up",
                "expression": "stern",
                "tilt": 3,
                "turn": 8,
            },
            "face": {"x": 8, "y": 8, "width": 8, "height": 8},
            "expressions": ["neutral", "stern"],
        }
        m["poses"] = {
            "at_ease": {"framings": ["close", "full", "stern"]},
            "chin": {"framings": ["thought"]},
        }
        folder = write_layered(self.tmp / "sets", m)
        report = LC.check_layered_pack(folder)
        self.assertTrue(report["pass"], report["problems"])
        names = {p.name for p in folder.iterdir()}
        self.assertIn("face.thought.chin.thinking.open.closed.webp", names)
        self.assertIn("face.stern.stern.half.half.webp", names)
        self.assertNotIn("face.stern.thinking.half.half.webp", names)
        self.assertEqual(sum(n.startswith("face.close.") for n in names), 27)
        for change, want in (
            (
                lambda x: x["framings"]["thought"]["sets"]["chin"].update(
                    expressions=["joy"]
                ),
                "sets",
            ),
            (
                lambda x: x["framings"]["stern"].update(expressions=["stern"]),
                "expressions",
            ),
            (
                lambda x: x["framings"]["close"]["camera"].update(height="knees"),
                "camera",
            ),
            (lambda x: x["framings"]["close"]["head"].update(size=-1), "head"),
            (
                lambda x: x["poses"]["chin"].update(framings=["thought", "close"]),
                "face set of its own",
            ),
            (
                lambda x: x["poses"]["at_ease"].update(framings=["stern", "full"]),
                "at_ease",
            ),
        ):
            x = json.loads(json.dumps(m))
            change(x)
            with self.subTest(want=want):
                problems = LC.check_layered_manifest(x)
                self.assertTrue(any(want in p["message"] for p in problems), problems)

    def test_command_checks_a_layered_pack(self):
        folder = write_layered(self.tmp / "p")
        code, out, _ = run_cli("portrait-pack", "--check", str(folder), "--json")
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out)["pass"])

    def test_command_options(self):
        with mock.patch.object(
            layers,
            "write_layered_pack",
            return_value={
                "check": {"pass": True, "problems": [], "bytes": 10},
                "files": 3,
                "seconds": 1,
            },
        ) as w:
            code, _, _ = run_cli(
                "portrait-pack",
                "--out",
                str(self.tmp / "o"),
                "--layers",
                "--poses",
                "at_ease,chin",
                "--framings",
                "close",
                "--no-leap",
                "--scale",
                "0.5",
                "--quality",
                "80",
                "--poses-file",
                "poses.json",
                "--shots-file",
                "shots.json",
            )
        self.assertEqual(code, 0)
        kw = w.call_args.kwargs
        self.assertEqual(
            (kw["poses_file"], kw["shots_file"]), ("poses.json", "shots.json")
        )
        self.assertEqual(kw["poses"], ["at_ease", "chin"])
        self.assertEqual(kw["framings"], ["close"])
        self.assertFalse(kw["leap"])
        self.assertEqual(kw["scale"], 0.5)
        self.assertEqual(kw["quality"], {"body": 80, "face": 82})
        code, _, err = run_cli(
            "portrait-pack", "--out", str(self.tmp / "o"), "--layers", "--scale", "2"
        )
        self.assertEqual(code, 2)
        self.assertIn("--scale", err)


def synthetic_figure(dx=0.0, crown=1.6):
    """A figure whose head stands at x = dx, its crown at `crown` metres and its chin 0.22 below."""
    return {
        "crown": [dx, crown, 0.0],
        "chin": [dx, crown - 0.22, 0.1],
        "eyes": [dx, crown - 0.11, 0.1],
        "ground": 0.0,
    }


def synthetic_probe(dx=0.0, crown=1.6, arm_up=0.0, half_width=0.25):
    ys = np.linspace(crown + 0.6, -0.05, 400)
    top = crown + arm_up
    covered = (ys <= top) & (ys >= 0.0)
    return {
        "y": ys,
        "x0": np.full(400, dx - half_width),
        "x1": np.full(400, dx + half_width),
        "covered": covered,
        "edge": False,
    }


class TestRenderCuts(unittest.TestCase):
    """The cuts of the render's time: the leap at its own pixels per head, the expressions each
    framing needs, fewer samples where the face is small, and the leap's own frame in the
    contract."""

    def test_leap_alias(self):
        f = TestFramer().framer()
        full = f.geo["full"]
        f.alias("leap", "full", layers.LEAP_SCALE)
        leap = f.geo["leap"]
        self.assertEqual(layers.FRAMINGS["full"]["scale"], 160)
        self.assertEqual(leap["scale"], layers.LEAP_SCALE)
        self.assertAlmostEqual(leap["height"] / full["height"], 230 / 160, delta=0.02)
        self.assertEqual((leap["top"], leap["below"]), (full["top"], full["below"]))
        h = f.head_rows("leap")
        self.assertAlmostEqual(h["y"] / f.head_rows("full")["y"], 230 / 160, delta=0.02)

    def test_tables(self):
        self.assertEqual(
            layers.FRAMING_EXPRESSIONS["full"], ("neutral", "joy", "smile")
        )
        self.assertEqual(layers.FRAMING_EXPRESSIONS["stern"], ("neutral", "stern"))
        self.assertEqual(layers.FRAMING_LEAVES["close"], {"stern": "stern"})
        self.assertEqual(layers.FRAMING_SAMPLES, {"thigh": 8, "full": 8})

    def test_framing_samples(self):
        self.assertIsNone(layers.parse_framing_samples(None))
        self.assertEqual(layers.parse_framing_samples("none"), {})
        self.assertEqual(
            layers.parse_framing_samples("thigh=4, low=8"), {"thigh": 4, "low": 8}
        )
        for bad in ("thigh=0", "knee=8", "full=x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                layers.parse_framing_samples(bad)
        code, _, err = run_cli(
            "portrait-pack", "--out", "x", "--framing-samples", "thigh=8"
        )
        self.assertEqual(code, 2)
        self.assertIn("take --layers", err)

    def test_leap_frame_in_the_contract(self):
        m = layered_manifest()
        m["leap"]["frame"] = {"width": 24, "height": 60}
        m["leap"]["head"] = {"x": 12, "y": 9, "top": 4, "chin": 14, "size": 9.5}
        self.assertEqual(LC.check_layered_manifest(m), [])
        self.assertEqual(LC.wanted_layers(m)["leap.peak.webp"], (24, 60))
        self.assertEqual(LC.wanted_layers(m)["body.at_ease.full.webp"], (16, 40))
        folder = write_layered(self.tmp / "leap", m)
        self.assertTrue(contract.check_pack(folder)["pass"])
        for change in (
            lambda lp: lp.pop("head"),
            lambda lp: lp["frame"].update(width=5000),
            lambda lp: lp["head"].update(top=20),
            lambda lp: lp.update(scale=2),
        ):
            m2 = json.loads(json.dumps(m))
            change(m2["leap"])
            with self.subTest(leap=m2["leap"]):
                self.assertIn(
                    "bad_manifest", codes({"problems": LC.check_layered_manifest(m2)})
                )

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestJoints(unittest.TestCase):
    """The joints of each body layer and leap frame in the manifest (``joints``)."""

    def manifest(self):
        m = layered_manifest()
        n = len(LC.JOINT_NAMES)
        m["joints"] = {
            "names": list(LC.JOINT_NAMES),
            "poses": {
                "at_ease": {"close": [1.5] * (2 * n), "full": [-20.0] * (2 * n)},
                "chin": {"close": [3] * (2 * n)},
            },
            "leap": {"peak": [4.0] * (2 * n)},
        }
        return m

    def test_valid(self):
        self.assertEqual(LC.check_layered_manifest(self.manifest()), [])

    def test_broken(self):
        n = len(LC.JOINT_NAMES)
        for change in (
            lambda j: j.update(extra=1),
            lambda j: j.update(names=["eyes", "eyes"]),
            lambda j: j.update(names=["toes"]),
            lambda j: j["poses"]["at_ease"].update(waist=[0] * (2 * n)),
            lambda j: j["poses"].update(wave={"close": [0] * (2 * n)}),
            lambda j: j["poses"]["chin"].update(close=[0] * 3),
            lambda j: j["poses"]["chin"].update(close=[1e9] * (2 * n)),
            lambda j: j["leap"].update(rise=[0] * (2 * n)),
        ):
            m = self.manifest()
            change(m["joints"])
            with self.subTest(joints=str(m["joints"])[:80]):
                self.assertIn(
                    "bad_manifest", codes({"problems": LC.check_layered_manifest(m)})
                )

    def test_manifest_text(self):
        """The manifest keeps an indent of 2 and writes each list of numbers on one line."""
        m = self.manifest()
        text = layers.manifest_text(m)
        self.assertEqual(json.loads(text), m)
        self.assertIn('"peak": [4.0, 4.0', text)
        self.assertIn('"names": [\n', text)


class TestResume(unittest.TestCase):
    """--resume: the layers an unfinished run left behind are kept when they decode whole."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_layer_ok(self):
        rgba = np.zeros((32, 24, 4), np.uint8)
        rgba[8:24, 6:18] = (200, 150, 120, 255)
        good = self.tmp / "body.at_ease.close.webp"
        layers.save_layer(good, rgba, 80)
        self.assertTrue(layers.layer_ok(good, (24, 32)))
        self.assertFalse(layers.layer_ok(good, (32, 24)), "the wrong size")
        cut = self.tmp / "body.wave.close.webp"
        cut.write_bytes(good.read_bytes()[: len(good.read_bytes()) // 2])
        self.assertFalse(
            layers.layer_ok(cut, (24, 32)), "a file a stopped run cut short"
        )
        self.assertFalse(layers.layer_ok(self.tmp / "missing.webp", (24, 32)))
        flat = self.tmp / "body.fist.close.webp"
        Image.fromarray(rgba[..., :3]).save(flat, format="WEBP")
        self.assertFalse(layers.layer_ok(flat, (24, 32)), "no alpha")

    def test_folder_kept(self):
        folder = write_layered(self.tmp / "p")
        n = len(list(folder.iterdir()))
        layers.prepare_layered_folder(folder, keep=True)
        self.assertEqual(len(list(folder.iterdir())), n)
        layers.prepare_layered_folder(folder)
        self.assertEqual(list(folder.iterdir()), [])

    def test_command(self):
        code, _, err = run_cli(
            "portrait-pack", "--out", str(self.tmp / "x"), "--resume"
        )
        self.assertEqual(code, 2)
        self.assertIn("take --layers", err)


class TestFramer(unittest.TestCase):
    def framer(self):
        figures = {
            "at_ease": synthetic_figure(),
            "wave": synthetic_figure(dx=0.05, crown=1.58),
        }
        probes = {
            "at_ease": synthetic_probe(),
            "wave": synthetic_probe(dx=0.05, crown=1.58, arm_up=0.3, half_width=0.4),
        }
        poses_of = {n: list(figures) for n in layers.FRAMINGS}
        poses_of["close"] = ["at_ease"]
        return layers.Framer(figures, probes, dict(layers.FRAMINGS), poses_of)

    def test_scales_and_sizes(self):
        f = self.framer()
        self.assertAlmostEqual(f.H, 0.22)
        for name, d in layers.FRAMINGS.items():
            g = f.geo[name]
            head = f.head_rows(name)
            # the frame's pixels per head are the framing's scale, to the pixel
            self.assertLessEqual(
                abs((head["chin"] - head["top"]) - d["scale"]), 1, name
            )
            self.assertEqual(g["width"] % layers.GRID, 0)
            self.assertEqual(g["height"] % layers.GRID, 0)
        # raised hands lift the top of the framings that keep them, and the close framing keeps its own top
        self.assertGreater(f.geo["thigh"]["top"], 0.3 / 0.22)
        self.assertLess(f.geo["close"]["top"], 1.0)
        # the full framing reaches the ground
        self.assertGreaterEqual(f.geo["full"]["below"], 1.6 / 0.22)

    def test_camera_follows_the_head(self):
        f = self.framer()
        for name in layers.FRAMINGS:
            a, b = f.view(name, "at_ease"), f.view(name, "wave")
            # the camera moves with the head: the same offset from the eyes and the crown
            self.assertAlmostEqual(b["cx"] - a["cx"], 0.05)
            self.assertAlmostEqual(b["cy"] - a["cy"], -0.02)
            self.assertAlmostEqual(b["cz"], 0.1)
            self.assertEqual((a["width"], a["height"]), (b["width"], b["height"]))

    def test_front_mask(self):
        base = np.zeros((40, 40, 4), np.uint8)
        base[..., :] = (180, 140, 120, 255)
        self.assertIsNone(layers.front_mask(base.copy(), base))
        hand = base.copy()
        hand[20:30, 10:25] = (90, 60, 50, 255)
        m = layers.front_mask(hand, base)
        self.assertIsNotNone(m)
        self.assertEqual(m[25, 15], 255)
        self.assertEqual(m[5, 5], 0)
        # a pose that moves only a little of the collar beside the face still gets a front
        wide = np.zeros((100, 100, 4), np.uint8)
        wide[..., :] = (180, 140, 120, 255)
        collar = wide.copy()
        collar[60:63, 70:73] = (60, 50, 40, 255)
        m = layers.front_mask(collar, wide)
        self.assertIsNotNone(m)
        self.assertEqual(m[61, 71], 255)

    def test_front_mask_takes_the_shade_beside_the_face(self):
        # a pose darkens the neck beside the face by less than the threshold: the front takes it,
        # with a soft ramp, and leaves the pixels the faces change to the faces
        base = np.zeros((80, 80, 4), np.uint8)
        base[..., :] = (200, 160, 140, 255)
        pose = base.copy()
        pose[40:80, 50:80, :3] -= 25
        m = layers.front_mask(pose, base)
        self.assertIsNotNone(m)
        self.assertEqual(m[60, 65], 255)
        self.assertEqual(m[60, 30], 0)
        core = np.zeros((80, 80), bool)
        core[40:80, 60:80] = True
        m = layers.front_mask(pose, base, core=core)
        self.assertEqual(m[60, 70], 0)
        self.assertEqual(m[60, 54], 255)
        # the renders' noise stays out of it
        rng = np.random.default_rng(1)
        noisy = base.copy()
        noisy[..., :3] = np.clip(
            base[..., :3].astype(np.int16) + rng.integers(-4, 5, (80, 80, 3)), 0, 255
        ).astype(np.uint8)
        self.assertIsNone(layers.front_mask(noisy, base))

    def test_soft_mask(self):
        changed = np.zeros((60, 80), bool)
        changed[20:22, 20:30] = changed[40:43, 30:50] = True
        w = layers.soft_mask(changed, 3, 2)
        self.assertEqual(w[21, 25], 1.0)
        self.assertEqual(w[5, 5], 0.0)
        self.assertTrue(((w > 0) & (w < 1)).any())
        self.assertTrue((w[changed] > 0.4).all())
        m = layers.front_mask(
            np.full((60, 80, 4), 90, np.uint8), np.full((60, 80, 4), 200, np.uint8), w
        )
        self.assertEqual(m[5, 5], 0)
        self.assertEqual(m[21, 25], 255)

    def test_snap_rect(self):
        self.assertEqual(
            layers.snap_rect(3, 9, 21, 30, 64, 64),
            {"x": 0, "y": 8, "width": 24, "height": 24},
        )
        self.assertEqual(
            layers.snap_rect(-5, -5, 70, 70, 64, 64),
            {"x": 0, "y": 0, "width": 64, "height": 64},
        )


class TestSharedPoses(unittest.TestCase):
    """The shared pose file (squire.poses/1) on the default character's rig."""

    @classmethod
    def setUpClass(cls):
        cls.shared = shared_poses.load()
        cls.rig = shared_poses.card_rig(None)
        cls.index = {n: j for j, n in enumerate(cls.rig.labels)}

    def page_fk(self, name):
        """The rig's joints (page frame) for a shared pose, as the page poses it."""
        rig = self.rig
        R = shared_poses.page_rotations(
            self.shared.local(name), rig.labels, rig.parents, rig.turns
        )
        W, P = np.empty_like(R), np.empty((len(R), 3))
        for j, p in enumerate(rig.parents):
            W[j] = R[j] if p < 0 else W[p] @ R[j]
            P[j] = (
                rig.heads[j] if p < 0 else P[p] + W[p] @ (rig.heads[j] - rig.heads[p])
            )
        return W, P

    def vrm_fk(self, name):
        """The joints of the VRM file of the same card for the same pose: the T pose's joints,
        the humanoid bones turned by the file's normalized rotations, the forearm's twist bone by
        its roll constraint, every other node carried by its parent; the wrist hangs from
        lowerarm01, as in the file."""
        from opensculptboy.export.tpose import roll_constraint

        rig = self.rig
        T = np.empty((len(rig.labels), 3))
        for j, p in enumerate(rig.parents):
            T[j] = (
                rig.heads[j]
                if p < 0
                else T[p] + rig.turns[p] @ (rig.heads[j] - rig.heads[p])
            )
        local_h = self.shared.local(name)
        to_human = {r: h for h, r in shared_poses.bone_map().items()}
        twists = shared_poses.twist_rules()
        parents = list(rig.parents)
        for t, (src, _, _) in twists.items():
            parents[self.index[src]] = parents[self.index[t]]
        W, P = np.empty((len(T), 3, 3)), np.empty((len(T), 3))
        for j, n in enumerate(rig.labels):
            if n in to_human:
                L = local_h.get(to_human[n], np.eye(3))
            elif n in twists:
                src, axis, w = twists[n]
                L = roll_constraint(local_h.get(to_human[src], np.eye(3)), axis, w)
            else:
                L = np.eye(3)
            p = parents[j]
            W[j] = L if p < 0 else W[p] @ L
            P[j] = T[j] if p < 0 else P[p] + W[p] @ (T[j] - T[p])
        return P

    def test_file(self):
        self.assertEqual(shared_poses.DEFAULT_FILE.name, "poses.json")
        for name in stage_poses.POSES:
            self.assertIn(name, self.shared.bones)
        b = self.shared.bones
        self.assertEqual(b["at_ease"]["leftUpperArm"], (8.0, -6.0, -79.0))
        # the stance of at_ease: the weight on the left leg, the hips and the shoulders tilted
        # against each other, and the head turned back toward the camera
        self.assertEqual(b["at_ease"]["hips"], (0.0, 5.0, 4.0))
        self.assertEqual(b["at_ease"]["chest"], (0.0, -2.0, -3.0))
        self.assertEqual(b["arms_crossed"]["spine"], (-2.0, -2.0, -4.0))
        self.assertEqual(b["arms_crossed"]["hips"], (0.0, 5.0, 4.0))
        # the poses that turn their own head, with their face sets, and the leap's key poses
        self.assertEqual(
            set(self.shared.own_head),
            {"chin", "hand_on_chest", "downcast", *stage_poses.LEAP_POSES.values()},
        )
        self.assertEqual(self.shared.faces["chin"], ["neutral", "thinking"])
        self.assertEqual(self.shared.faces["cheer_peak"], ["joy"])
        self.assertEqual(b["at_ease"]["rightUpperArm"], (8.0, 6.0, 79.0))
        self.assertEqual(b["fist"]["leftThumbMetacarpal"], (20.0, 45.0, -10.0))
        self.assertEqual(b["wave"]["leftIndexProximal"], (0.0, -7.5, -4.0))
        self.assertEqual(b["relaxed"]["rightUpperArm"], (10.0, 2.0, 80.0))
        self.assertEqual(self.shared.keep, ("neck", "head"))
        # angles about X, then Y, then Z in the parent's frame
        np.testing.assert_allclose(
            shared_poses.euler_matrix((30, 0, 40)),
            shared_poses.euler_matrix((0, 0, 40))
            @ shared_poses.euler_matrix((30, 0, 0)),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            shared_poses.euler_matrix((0, -90, 0)) @ [1, 0, 0], [0, 0, 1], atol=1e-12
        )

    def test_broken_files(self):
        data = json.loads(shared_poses.DEFAULT_FILE.read_text(encoding="utf-8"))
        cases = [
            ({"format": "x"}, "format"),
            (
                {"poses": {"p": {"bones": {"tail": [0, 0, 1]}}}},
                "not a VRM humanoid bone",
            ),
            ({"poses": {"p": {"left": "up"}}}, "no arm preset"),
            (
                {"poses": {"p": {"left": {"hand": {"curl": "claw"}}}}},
                "no finger preset",
            ),
            ({"keepHead": {"pose": "nope", "bones": ["head"]}}, "keepHead"),
            ({"poses": {"p": {"stance": "lounge"}}}, "no stance"),
            (
                {"poses": {"p": {"faces": ["neutral"]}}},
                "only a pose that turns its own head",
            ),
            (
                {"poses": {"p": {"keepHead": False, "faces": ["neutral", "neutral"]}}},
                "faces",
            ),
        ]
        for change, want in cases:
            with self.subTest(want=want):
                with self.assertRaisesRegex(shared_poses.PosesError, want):
                    shared_poses.parse({**data, **change})

    def test_every_bone_resolves(self):
        rig, table = self.rig, shared_poses.bone_map()
        for human, bone in table.items():
            self.assertIn(human, shared_poses.HUMANOID_PARENT, human)
            self.assertIn(bone, self.index, f"{human} -> {bone}")
        governing = shared_poses.governing(rig.labels, rig.parents)
        self.assertEqual(set(governing), set(rig.labels))
        self.assertEqual(governing["root"], "hips")
        self.assertEqual(governing["spine03"], "spine")
        self.assertEqual(governing["neck03"], "neck")
        self.assertEqual(governing["lowerarm02.L"], "leftLowerArm")
        self.assertEqual(governing["metacarpal2.R"], "rightHand")
        self.assertEqual(governing["toe2-1.L"], "leftFoot")
        for name, bones in self.shared.bones.items():
            for human in bones:
                with self.subTest(pose=name, bone=human):
                    self.assertIn(human, table)
        for twist, (src, _, _) in shared_poses.twist_rules().items():
            self.assertIn(twist, self.index)
            self.assertIn(src, self.index)

    def test_t_pose(self):
        """With no rotation the rig stands in the VRM file's T pose: the arms along X, the legs
        straight down."""
        rig = self.rig
        W = shared_poses.rig_world({}, rig.labels, rig.parents, rig.turns)
        _, P = self.page_fk_with(W)
        unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
        for s, sign in (("L", 1), ("R", -1)):
            arm = unit(
                P[self.index[f"lowerarm01.{s}"]] - P[self.index[f"upperarm01.{s}"]]
            )
            np.testing.assert_allclose(arm, [sign, 0, 0], atol=1e-6)
            leg = unit(P[self.index[f"foot.{s}"]] - P[self.index[f"upperleg01.{s}"]])
            np.testing.assert_allclose(leg, [0, -1, 0], atol=1e-6)

    def page_fk_with(self, Rw):
        rig = self.rig
        P = np.empty((len(Rw), 3))
        for j, p in enumerate(rig.parents):
            P[j] = (
                rig.heads[j] if p < 0 else P[p] + Rw[p] @ (rig.heads[j] - rig.heads[p])
            )
        return Rw, P

    def test_lands_on_the_vrm_skeleton(self):
        """Every pose puts the stills' joints where the VRM file of the same card puts them."""
        joints = [
            "head",
            "neck01",
            "upperarm01.L",
            "lowerarm01.L",
            "lowerarm01.R",
            "wrist.L",
            "wrist.R",
            "finger3-3.L",
            "finger3-3.R",
            "lowerleg01.L",
            "lowerleg01.R",
            "foot.L",
            "foot.R",
        ]
        for name in [*stage_poses.POSES, *stage_poses.LEAP_POSES.values()]:
            with self.subTest(pose=name):
                _, page = self.page_fk(name)
                vrm = self.vrm_fk(name)
                for j in joints:
                    i = self.index[j]
                    self.assertLess(np.linalg.norm(page[i] - vrm[i]), 0.002, j)

    def test_upper_arm_roll_shared(self):
        """The upper arm's first bone keeps half of the arm's own roll and its second bone takes
        the rest, and the joints stay within 2 mm of where the whole roll at the shoulder puts
        them (the roll turns about X, and upperarm02's head lies about a millimetre off that
        line), as close as the VRM file's joints."""
        from opensculptboy.export.tpose import roll_constraint

        rig = self.rig
        local = self.shared.local("hands_behind_head")
        W = shared_poses.rig_world(local, rig.labels, rig.parents, rig.turns)
        saved = dict(shared_poses.UPPER_ARM_ROLL)
        try:
            shared_poses.UPPER_ARM_ROLL.clear()
            whole = shared_poses.rig_world(local, rig.labels, rig.parents, rig.turns)
        finally:
            shared_poses.UPPER_ARM_ROLL.update(saved)
        _, P = self.page_fk_with(W)
        _, P_whole = self.page_fk_with(whole)
        np.testing.assert_allclose(P, P_whole, atol=0.002)
        for s, arm in (("L", "leftUpperArm"), ("R", "rightUpperArm")):
            with self.subTest(side=s):
                i, k = self.index[f"upperarm01.{s}"], self.index[f"upperarm02.{s}"]
                T = rig.turns[i]
                taken = (whole[i] @ T.T).T @ (W[i] @ T.T)
                np.testing.assert_allclose(
                    taken, roll_constraint(local[arm], [1, 0, 0], -0.5), atol=1e-9
                )
                angle = np.degrees(np.arccos(np.clip((np.trace(taken) - 1) / 2, -1, 1)))
                self.assertGreater(angle, 20.0)
                np.testing.assert_allclose(W[k], whole[k], atol=1e-12)

    def test_head_kept(self):
        ref, _ = self.page_fk("at_ease")
        for name in stage_poses.POSES:
            W, _ = self.page_fk(name)
            i = self.index["head"]
            if name in self.shared.own_head:
                # a pose with its own face set turns its head
                self.assertGreater(np.abs(W[i] - ref[i]).max(), 0.05, name)
                continue
            for bone in ("neck01", "neck02", "neck03", "head"):
                with self.subTest(pose=name, bone=bone):
                    i = self.index[bone]
                    np.testing.assert_allclose(W[i], ref[i], atol=1e-9)

    def test_page_pose(self):
        import anny.poses

        bones = anny.poses.library().bones
        rig = self.rig
        for name in stage_poses.POSES:
            with self.subTest(pose=name):
                p = shared_poses.page_pose(
                    self.shared, name, bones, rig.labels, rig.parents, rig.turns
                )
                q = np.asarray(p["q"]).reshape(-1, 4)
                self.assertEqual(len(q), len(bones))
                np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-9)
                self.assertEqual(p["root"], [0.0, 0.0, 0.0])
        # the leap's key poses come from the same file
        for key in stage_poses.LEAP:
            p = shared_poses.page_pose(
                self.shared,
                stage_poses.shared_name(key),
                bones,
                rig.labels,
                rig.parents,
                rig.turns,
            )
            self.assertEqual(len(p["q"]), 4 * len(bones))


def shots_data():
    """A shots file of two framings."""
    return {
        "format": shots.FORMAT,
        "sensor": 24,
        "heights": {"eyes": 0, "chest": 1.5, "hips": 3.5},
        "renders": {"close": "close_up", "full": "full"},
        "shots": {
            "close_up": {
                "span": 1.0,
                "share": [0.36, 0.4, 0.42],
                "top": 0.06,
                "stills": "close",
                "camera": {
                    "lens": 85,
                    "height": "eyes",
                    "above": 0,
                    "tilt": 0,
                    "turn": 20,
                },
            },
            "full": {
                "span": 9.0,
                "share": 1,
                "fit": "body",
                "top": 0.3,
                "bottom": 0.95,
                "stills": "full",
                "camera": {
                    "lens": 50,
                    "height": "hips",
                    "above": 0,
                    "tilt": -8,
                    "turn": 20,
                },
            },
            "thought": {
                "span": 2.4,
                "share": 0.95,
                "top": 0.24,
                "stills": "close",
                "camera": {
                    "lens": 85,
                    "height": "eyes",
                    "above": 0,
                    "tilt": 0,
                    "turn": 25,
                },
            },
        },
    }


class TestShotCameras(unittest.TestCase):
    """The shots' cameras of the framings (--shots-file)."""

    def test_parse(self):
        cams = shots.parse(shots_data())
        self.assertEqual(set(cams), {"close", "full"})
        c, f = cams["close"], cams["full"]
        self.assertEqual(
            (c.shot, c.lens, c.below_eyes, c.tilt, c.turn), ("close_up", 85, 0, 0, 20)
        )
        self.assertEqual((f.shot, f.below_eyes, f.tilt), ("full", 3.5, -8))
        self.assertAlmostEqual(c.window, 1.0 / 0.4)
        self.assertAlmostEqual(f.window, 9.0 / 0.65)
        self.assertAlmostEqual(c.fov, 16.0714, places=3)
        # the window's height at the eyes' distance spans the shot's head heights
        d = c.distance(0.2)
        self.assertAlmostEqual(2 * d * np.tan(np.radians(c.fov) / 2), c.window * 0.2)
        self.assertEqual(
            f.manifest(),
            {
                "shot": "full",
                "lens": 50.0,
                "height": "hips",
                "above": 0.0,
                "tilt": -8.0,
                "turn": 20.0,
            },
        )

    def test_render_with_an_expression(self):
        """The square-on stern close-up renders through the close-up's camera with its stern
        values, as a framing of its own."""
        data = shots_data()
        data["shots"]["close_up"]["byExpression"] = {
            "stern": {"stills": "stern", "camera": {"tilt": 3, "turn": 8}}
        }
        data["renders"]["stern"] = {"shot": "close_up", "expression": "stern"}
        s = shots.parse(data)["stern"]
        self.assertEqual(
            (s.shot, s.expression, s.lens, s.tilt, s.turn),
            ("close_up", "stern", 85, 3, 8),
        )
        self.assertEqual(s.manifest()["expression"], "stern")
        data["shots"]["close_up"]["byExpression"]["stern"]["stills"] = "close"
        with self.assertRaisesRegex(
            shots.ShotsError, r"close_up \(stern\), whose stills"
        ):
            shots.parse(data)

    def test_broken(self):
        for change, want in (
            (lambda d: d.update(format="x"), "format"),
            (lambda d: d["renders"].update(close="full"), "whose stills are 'full'"),
            (lambda d: d["renders"].update(waist="medium_close"), "unknown shot"),
            (
                lambda d: d["shots"]["close_up"]["camera"].update(height="knees"),
                "height",
            ),
            (lambda d: d["shots"]["full"]["camera"].update(tilt=45), "out of range"),
            (
                lambda d: d["renders"].update(
                    stern={"shot": "close_up", "expression": "joy"}
                ),
                "no values for",
            ),
        ):
            data = shots_data()
            change(data)
            with (
                self.subTest(want=want),
                self.assertRaisesRegex(shots.ShotsError, want),
            ):
                shots.parse(data)

    def framer(self):
        cams = shots.parse(shots_data())
        figure = {
            "crown": [0.0, 1.7, 0.03],
            "chin": [0.0, 1.48, 0.1],
            "eyes": [0.0, 1.6, 0.1],
            "ground": 0.0,
            "joints": {"root": [0.0, 0.95, 0.0]},
        }
        moved = {
            **figure,
            "crown": [0.05, 1.68, 0.03],
            "chin": [0.05, 1.46, 0.1],
            "eyes": [0.05, 1.58, 0.1],
        }
        rows = np.linspace(-1, 9, 500)
        cov = (rows >= -0.1) & (rows <= 7.8)
        outline = (rows, np.full(500, -1.2), np.full(500, 1.2), cov)
        figs = {n: {"at_ease": figure, "wave": moved} for n in ("close", "full")}
        outs = {n: {"at_ease": outline, "wave": outline} for n in ("close", "full")}
        defs = {n: dict(layers.FRAMINGS[n]) for n in ("close", "full")}
        poses_of = {n: ["at_ease", "wave"] for n in ("close", "full")}
        return layers.ShotFramer(figs, outs, defs, poses_of, cams), figs

    def test_camera_and_view(self):
        f, figs = self.framer()
        for name in ("close", "full"):
            g, head = f.geo[name], f.head_rows(name)
            # the scale is pixels per head at the eyes' depth; the rows are the crown's and the
            # chin's own, which a tilted camera foreshortens (the chin lies forward of the crown)
            self.assertLess(
                abs((head["chin"] - head["top"]) / g["scale"] - 1), 0.06, name
            )
            # the head's projected size: one stage head height at the eyes' depth
            hu = (1.6 - 0.95) / 3.5
            self.assertAlmostEqual(head["size"], g["scale"] * hu / 0.22, delta=0.01)
            for pose in ("at_ease", "wave"):
                v = f.view(name, pose)
                c = v["camera"]
                self.assertEqual(v["turn"], 20)
                cam = {"eye": np.array(c["eye"]), "pitch": c["pitch"]}
                # the eyes land on the framing's anchor in every pose: the camera follows the head
                u, w, _ = layers.project_points(
                    cam, figs[name][pose]["eyes"], c["focal"]
                )[0]
                self.assertAlmostEqual(u + c["px"], head["x"], places=6)
                self.assertAlmostEqual(w + c["py"], head["y"], places=6)
        # the full shot's camera sits level with the hips and looks 8 degrees up
        c = f.view("full", "at_ease")["camera"]
        hu = (1.6 - 0.95) / 3.5
        self.assertAlmostEqual(c["eye"][1], 1.6 - 3.5 * hu)
        self.assertEqual(c["pitch"], -8)


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestDefaultLayeredPack(unittest.TestCase):
    """The default character's layered pack at a fifth of its scale, one pass per layer."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = pathlib.Path(tempfile.mkdtemp())
        cls.folder = cls.tmp / "pack"
        env = {render.CHROMIUM_ENV: str(CHROMIUM)} if CHROMIUM.exists() else {}
        cls.args = (
            "portrait-pack",
            "--layers",
            "--out",
            str(cls.folder),
            "--scale",
            "0.2",
            "--samples",
            "1",
            "--expressions",
            "neutral,joy",
            "--poses",
            "at_ease,hand_on_chest,cheer",
            "--page",
            str(PAGE),
            "--json",
        )
        with mock.patch.dict(os.environ, env):
            cls.result = run_cli(*cls.args)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_pack(self):
        code, out, err = self.result
        self.assertEqual(code, 0, err[-2000:])
        result = json.loads(out)
        self.assertTrue(result["check"]["pass"], result["check"]["problems"])
        m = result["manifest"]
        self.assertEqual(set(m["framings"]), set(layers.BASE_FRAMINGS))
        self.assertEqual(m["poses"]["cheer"]["framings"], ["thigh", "full"])
        # the bow turns the head: its own face set at the framings it shows at
        self.assertEqual(m["poses"]["hand_on_chest"]["framings"], ["waist", "thigh"])
        for f in ("waist", "thigh"):
            self.assertEqual(
                m["framings"][f]["sets"]["hand_on_chest"]["expressions"], ["neutral"]
            )
        self.assertEqual(m["leap"]["frames"], list(stage_poses.LEAP))
        # every body layer and leap frame records its joints; the eyes lie on the head's anchor
        j = m["joints"]
        self.assertEqual(j["names"], list(LC.JOINT_NAMES))
        self.assertEqual(set(j["leap"]), set(m["leap"]["frames"]))
        e = j["names"].index("eyes")
        for pose, p in m["poses"].items():
            self.assertEqual(set(j["poses"][pose]), set(p["framings"]))
            for f in p["framings"]:
                if pose in m["framings"][f].get("sets", {}):
                    continue  # a pose that turns its own head moves its eyes
                with self.subTest(pose=pose, framing=f):
                    xy = j["poses"][pose][f][2 * e : 2 * e + 2]
                    head = m["framings"][f]["head"]
                    self.assertAlmostEqual(xy[0], head["x"], delta=1.5)
                    self.assertAlmostEqual(xy[1], head["y"], delta=1.5)

    def test_resume(self):
        """A second run with --resume keeps every layer (a pack without passes has no scales to
        set again), removes nothing, and gives the same manifest."""
        code, out, err = self.result
        self.assertEqual(code, 0, err[-2000:])
        first = json.loads(out)
        env = {render.CHROMIUM_ENV: str(CHROMIUM)} if CHROMIUM.exists() else {}
        with mock.patch.dict(os.environ, env):
            code, out, err = run_cli(*self.args, "--resume")
        self.assertEqual(code, 0, err[-2000:])
        again = json.loads(out)
        self.assertTrue(again["check"]["pass"], again["check"]["problems"])
        self.assertEqual(again["kept"], first["check"]["frames"])
        self.assertEqual(again["removed"], 0)
        self.assertIn("faces", again["timing"])
        self.assertEqual(again["manifest"]["framings"], first["manifest"]["framings"])

    def test_faces_fit_the_bodies(self):
        """The neutral face layer lies on every body of its framing to the pixel: where the face layer paints,
        inside the middle of its rectangle, the body matches it best with no shift."""
        m = json.loads((self.folder / contract.MANIFEST).read_text())
        for name, f in m["framings"].items():
            r = f["face"]
            rgba = np.asarray(
                Image.open(
                    self.folder / f"face.{name}.neutral.closed.open.webp"
                ).convert("RGBA"),
                np.float32,
            )
            face, opaque = rgba[..., :3], rgba[..., 3] > 200
            h, w = face.shape[:2]
            y0, y1, x0, x1 = h // 4, h - h // 4, w // 4, w - w // 4
            inner = opaque[y0:y1, x0:x1]
            self.assertGreater(
                inner.mean(), 0.3, "the face layer paints the middle of the face"
            )
            # the poses that share the main face set (hand_on_chest has its own)
            for pose in ("at_ease", "cheer"):
                if name not in m["poses"][pose]["framings"]:
                    continue
                with self.subTest(framing=name, pose=pose):
                    body = np.asarray(
                        Image.open(self.folder / f"body.{pose}.{name}.webp").convert(
                            "RGB"
                        ),
                        np.float32,
                    )

                    def diff(dx, dy):
                        crop = body[
                            r["y"] + y0 + dy : r["y"] + y1 + dy,
                            r["x"] + x0 + dx : r["x"] + x1 + dx,
                        ]
                        return float(np.abs(crop - face[y0:y1, x0:x1])[inner].mean())

                    shifts = {
                        (dx, dy): diff(dx, dy)
                        for dx in (-2, -1, 0, 1, 2)
                        for dy in (-2, -1, 0, 1, 2)
                    }
                    self.assertEqual(min(shifts, key=shifts.get), (0, 0), shifts)

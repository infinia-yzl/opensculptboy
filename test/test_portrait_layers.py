# OpenSculptBoy
# Apache License, Version 2.0
"""
The layered portrait pack (squire.portraits/2, opensculptboy.portrait_pack.layers, docs/portrait-pack.md).

In CI: the check with synthetic layered packs (a valid one, then each broken case with its reason
code), the WebP header reader, the framings' sizes and cameras from synthetic outlines, the front
layer's mask, the stage poses on the authoring rig (the head's turn kept, the hands where each pose
puts them) and the command's options. Local only (Playwright with the built page): the default
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
        for bad in (
            "body.chin.bust.webp",
            "face.close.joy.open.webp",
            "leap.top.webp",
            "body.Chin.close.webp",
            "neutral.closed.open.png",
        ):
            self.assertIsNone(LC.layer_kind(bad), bad)

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
            )
        self.assertEqual(code, 0)
        kw = w.call_args.kwargs
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

    def test_snap_rect(self):
        self.assertEqual(
            layers.snap_rect(3, 9, 21, 30, 64, 64),
            {"x": 0, "y": 8, "width": 24, "height": 24},
        )
        self.assertEqual(
            layers.snap_rect(-5, -5, 70, 70, 64, 64),
            {"x": 0, "y": 0, "width": 64, "height": 64},
        )


class TestStagePoses(unittest.TestCase):
    """The poses on the authoring rig: the head's turn of at_ease in every pose, and the hands
    where the poses put them."""

    @classmethod
    def setUpClass(cls):
        cls.P, cls.L, _ = stage_poses._tools()

    def world(self, name):
        return self.P.fk(stage_poses.built(name))

    def test_head_kept(self):
        ref = stage_poses.head_turn(stage_poses.HEAD_POSE)
        for name in stage_poses.POSES:
            with self.subTest(pose=name):
                np.testing.assert_allclose(stage_poses.head_turn(name), ref, atol=1e-6)

    def test_hands(self):
        P = self.P
        pos = lambda name, bone: self.world(name)[1][P.IDX[bone]]  # noqa: E731
        crown = lambda name: stage_poses._key_points(stage_poses.built(name))["crown"]  # noqa: E731
        # hands in the air above the crown, and the waving hand above the eyes
        for name, sides in (("cheer", "LR"), ("fist", "R")):
            for s in sides:
                with self.subTest(pose=name, side=s):
                    self.assertGreater(pos(name, f"wrist.{s}")[1], crown(name)[1])
        eyes = stage_poses._key_points(stage_poses.built("wave"))["eyes"]
        self.assertGreater(pos("wave", "wrist.R")[1], eyes[1])
        # behind the head: both wrists above the eyes and behind the head's joint
        for s in "LR":
            w, head = (
                pos("hands_behind_head", f"wrist.{s}"),
                pos("hands_behind_head", "head"),
            )
            self.assertGreater(w[1], head[1])
            self.assertLess(w[2], head[2])
        # the hand under the chin and the hand on the chest
        chin = stage_poses._key_points(stage_poses.built("chin"))["chin"]
        self.assertLess(np.linalg.norm(pos("chin", "wrist.R") - chin), 0.15)
        chest = pos("hand_on_chest", "spine01")
        self.assertLess(
            np.linalg.norm(
                pos("hand_on_chest", "wrist.R")[:2] - chest[:2] - [0.0, 0.0]
            ),
            0.12,
        )

    def test_page_pose(self):
        import anny.poses

        bones = anny.poses.library().bones
        for name in stage_poses.BUILDERS:
            with self.subTest(pose=name):
                p = stage_poses.page_pose(name, bones)
                q = np.asarray(p["q"]).reshape(-1, 4)
                self.assertEqual(len(q), len(bones))
                np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1, atol=1e-6)
                self.assertEqual(len(p["root"]), 3)


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
        with mock.patch.dict(os.environ, env):
            cls.result = run_cli(
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
                "at_ease,chin,cheer",
                "--page",
                str(PAGE),
                "--json",
            )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_pack(self):
        code, out, err = self.result
        self.assertEqual(code, 0, err[-2000:])
        result = json.loads(out)
        self.assertTrue(result["check"]["pass"], result["check"]["problems"])
        m = result["manifest"]
        self.assertEqual(set(m["framings"]), set(layers.FRAMINGS))
        self.assertEqual(m["poses"]["cheer"]["framings"], ["waist", "thigh", "full"])
        self.assertEqual(m["leap"]["frames"], list(stage_poses.LEAP))

    def test_faces_fit_the_bodies(self):
        m = json.loads((self.folder / contract.MANIFEST).read_text())
        for name, f in m["framings"].items():
            r = f["face"]
            face = np.asarray(
                Image.open(
                    self.folder / f"face.{name}.neutral.closed.open.webp"
                ).convert("RGBA"),
                np.float32,
            )
            for pose in ("at_ease", "chin", "cheer"):
                if name not in m["poses"][pose]["framings"]:
                    continue
                with self.subTest(framing=name, pose=pose):
                    body = np.asarray(
                        Image.open(self.folder / f"body.{pose}.{name}.webp").convert(
                            "RGBA"
                        ),
                        np.float32,
                    )
                    crop = body[
                        r["y"] : r["y"] + r["height"], r["x"] : r["x"] + r["width"]
                    ]
                    diff = np.abs(crop - face)[..., :3].mean()
                    # the neutral face lies on every body of its framing within the noise of one pass and lossy WebP,
                    # apart from the hand of chin, which its front layer covers
                    self.assertLess(diff, 25 if pose == "chin" else 8)

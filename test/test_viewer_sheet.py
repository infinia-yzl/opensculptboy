# OpenSculptBoy
# Apache License, Version 2.0
"""
The character sheet of the viewer (viewer/src/sheet/, docs/viewer.md "Sheet"): the settings
object and the layout maths in node (the framing of full figures at one scale on one ground
row, the head-and-shoulders frame, the page), and a sheet of the default character built in the
built page through window.sheet.build, in Chromium with SwiftShader through Playwright (local
only): the PNG's size, every panel's pictures with something in them and a transparent
background, one height for the turnaround's views and one cell size for the expressions.
"""

import base64
import io
import json
import pathlib
import subprocess
import unittest

from test.markers import local_only
from test.test_hair_page import CHROMIUM, PAGE, playwright_available
from test.test_viewer_parity import node_available

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "viewer" / "test" / "sheet.mjs"
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]


@unittest.skipUnless(
    node_available(), "node 22 or later is needed to run the page's TypeScript"
)
class TestSheetMaths(unittest.TestCase):
    def test_settings_and_layout(self):
        r = subprocess.run(
            ["node", str(SCRIPT)], capture_output=True, text=True, cwd=REPO
        )
        self.assertEqual(r.returncode, 0, r.stderr[-4000:])
        out = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertTrue(out["ok"])
        self.assertEqual(out["page"]["width"], 4096)
        self.assertEqual(out["page"]["pictures"], 22)


def png_size(data: bytes):
    """the width and height of a PNG from its header"""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestSheetPage(unittest.TestCase):
    def test_default_character(self):
        from playwright.sync_api import sync_playwright

        settings = {"render": {"scale": 0.25, "samples": 1}, "title": "Test sheet"}
        with sync_playwright() as p:
            kw = dict(executable_path=str(CHROMIUM)) if CHROMIUM.exists() else {}
            browser = p.chromium.launch(args=ARGS, **kw)
            tab = browser.new_context(
                viewport=dict(width=1200, height=800)
            ).new_page()
            errors = []
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.goto(PAGE.as_uri() + "?shot&acc=1")
            tab.wait_for_function("() => window.__ready === true", timeout=300000)
            pose = tab.evaluate("() => window.__MOTION.cur.name")
            png = base64.b64decode(
                tab.evaluate(
                    "(s) => window.sheet.build(s).then((b) => window.sheet.base64(b))",
                    settings,
                )
            )
            rep = tab.evaluate("() => window.sheet.report()")
            picture = tab.evaluate("() => window.sheet.picture('turnaround', 0)")
            after = tab.evaluate(
                "() => ({ pose: window.__MOTION.cur.name, turn: window.__RIG.holder.rotation.y, busy: window.sheet.busy() })"
            )
            browser.close()
        self.assertFalse(errors, errors)
        print(
            f"\nsheet {rep['width']} x {rep['height']}, {rep['png_bytes']} bytes, {rep['ms']} ms"
        )
        # the PNG of the page, at a quarter of the print size
        self.assertEqual(png_size(png), (rep["width"], rep["height"]))
        self.assertEqual(rep["width"], 1024)
        self.assertEqual(len(png), rep["png_bytes"])
        panels = rep["panels"]
        self.assertEqual(
            sorted(panels), sorted(["info", "turnaround", "expressions", "poses"])
        )
        self.assertEqual(len(panels["turnaround"]["pictures"]), 4)
        self.assertEqual(len(panels["expressions"]["pictures"]), 12)
        self.assertEqual(len(panels["poses"]["pictures"]), 6)
        for name in ("turnaround", "expressions", "poses"):
            for q in panels[name]["pictures"]:
                # something in each picture, on a transparent background
                self.assertGreater(q["coverage"], 0.03, (name, q))
                self.assertLess(q["coverage"], 0.9, (name, q))
        # one height for the turnaround's views and the poses, one cell for the expressions
        turn = panels["turnaround"]["pictures"]
        self.assertEqual({q["h"] for q in turn}, {512})
        self.assertEqual({q["h"] for q in panels["poses"]["pictures"]}, {384})
        self.assertEqual(
            {(q["w"], q["h"]) for q in panels["expressions"]["pictures"]}, {(216, 240)}
        )
        side, front = turn[2], turn[0]
        self.assertLess(side["w"], front["w"])
        self.assertLess(panels["turnaround"]["ground_row"], 512)
        self.assertGreater(panels["turnaround"]["ground_row"], 480)
        # a panel picture on its own keeps its transparent background
        from PIL import Image

        img = Image.open(
            io.BytesIO(base64.b64decode(picture.split(",", 1)[1]))
        ).convert("RGBA")
        self.assertEqual(img.size, (front["w"], front["h"]))
        self.assertEqual(img.getpixel((0, 0))[3], 0)
        # the composed page: the paper background in a corner, and the figure in the turnaround's place
        page = Image.open(io.BytesIO(png)).convert("RGB")
        self.assertEqual(page.getpixel((2, 2)), (0xF1, 0xEE, 0xE8))
        r = panels["turnaround"]["rect"]
        crop = page.crop(
            (int(r["x"]), int(r["y"]), int(r["x"] + r["w"]), int(r["y"] + r["h"]))
        )
        self.assertGreater(len(set(crop.getdata())), 50)
        # the page's own pose and turn come back
        self.assertEqual(after["pose"], pose)
        self.assertEqual(after["turn"], 0)
        self.assertFalse(after["busy"])

# OpenSculptBoy
# Apache License, Version 2.0
"""
The Wearables section of the built viewer page, in Chromium with SwiftShader through Playwright:
the synthetic outfit of test.test_viewer_wearables opened through the section's file input, the
pieces and the hides rules as opensculptboy.wearables resolves them, the dressed vertices
against opensculptboy.wearables.dress on the same body, the switches of the section (a hidden
piece, a piece of another outfit in its slot, the covered skin), a slider edit, and no request
of the page during the load. The test skips when Playwright or the built page is missing.
"""

import pathlib
import shutil
import tempfile
import unittest

import numpy as np

from test.markers import local_only
from test.test_hair_page import CHROMIUM, PAGE, playwright_available
from test.test_viewer_wearables import _packages

ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
ANNY_TO_PAGE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestWearablesPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        torch.set_num_threads(1)
        cls.dir = pathlib.Path(tempfile.mkdtemp())
        cls.card = _packages(cls.dir)
        cls.files = sorted(str(f) for f in cls.dir.iterdir())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def run_page(self, steps):
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            kw = dict(executable_path=str(CHROMIUM)) if CHROMIUM.exists() else {}
            browser = p.chromium.launch(args=ARGS, **kw)
            tab = browser.new_context(viewport=dict(width=900, height=640)).new_page()
            requests = []
            tab.on("request", lambda r: requests.append((r.method, r.url)))
            tab.goto(PAGE.as_uri() + "?shot&acc=1")
            tab.wait_for_function("() => window.__ready === true", timeout=300000)
            before = len(requests)
            tab.locator(
                "input[type=file][multiple]:not([webkitdirectory])"
            ).set_input_files(self.files)
            tab.wait_for_function(
                "() => window.wearStats() && window.wearStats().pieces.length > 0"
                " && window.wearStats().timing.full === true",
                timeout=120000,
            )
            out = steps(tab)
            out["requests"] = requests[before:]
            browser.close()
        return out

    def compare(self, pieces, expected, scale, median, largest):
        """The page's dressed vertices against Python's: the page's body comes from its
        compressed shape space and stands on the floor, so the pieces match after one shift,
        within the error of the shape space."""
        got = {p["ref"]: p for p in pieces}
        self.assertEqual(sorted(got), sorted(p.ref for p in expected))
        page = np.concatenate(
            [np.array(got[p.ref]["vertices"]).reshape(-1, 3) for p in expected]
        )
        want = np.concatenate([p.vertices for p in expected])
        anny = (page @ ANNY_TO_PAGE) / scale
        error = np.linalg.norm(anny - (anny - want).mean(0) - want, axis=1)
        print(
            f"\npage against Python: {error.max() * 1000:.2f} mm at most, "
            f"{np.median(error) * 1000:.3f} mm median"
        )
        self.assertLess(float(np.median(error)), median)
        self.assertLess(float(error.max()), largest)

    def test_page(self):
        from opensculptboy import Character
        from opensculptboy.wearables import card_body, dress, worn_pieces
        from test.wearable_fixtures import template_model

        def switch(tab, ref):
            tab.evaluate(
                "(ref) => document.querySelector(`.wear-piece[data-ref='${ref}'] button.switch`).click()",
                ref,
            )
            tab.wait_for_function("() => window.wearStats().timing.full === true")
            return tab.evaluate("() => window.wearPieces()")

        def steps(tab):
            out = {"stats": tab.evaluate("() => window.wearStats()")}
            out["scale"] = tab.evaluate(
                "() => window.__BODY.anny.meta.shape.frame.scale"
            )
            out["pieces"] = tab.evaluate("() => window.wearPieces()")
            out["no_cape"] = switch(tab, "travel/cape")
            out["smock"] = switch(tab, "work/smock")
            # the last switch of the section: the covered skin
            tab.evaluate(
                "() => [...document.querySelectorAll('.wear-body button.switch')].pop().click()"
            )
            out["kept"] = tab.evaluate("() => window.wearStats().hidden_triangles")
            out["slider"] = tab.evaluate("() => window.setSliders({ weight: 0.9 })")
            out["fast"] = tab.evaluate("() => window.wearStats().timing")
            tab.wait_for_function("() => window.wearStats().timing.full === true")
            out["settled"] = tab.evaluate("() => window.wearStats().timing")
            out["heavy"] = tab.evaluate("() => window.wearPieces()")
            return out

        out = self.run_page(steps)
        body = card_body(Character(), template_model())
        expected = dress(body, worn_pieces(self.card, self.dir))
        got = {p["ref"]: p for p in out["pieces"]}
        self.assertEqual(sorted(got), sorted(p.ref for p in expected))
        self.assertEqual(got["travel/tunic"]["hidden"], ["back"])
        self.compare(out["pieces"], expected, out["scale"], 0.0005, 0.004)
        self.assertGreater(out["stats"]["hidden_triangles"], 1000)
        # the switches: the cape off shows the tunic's back again; the smock takes the tunic's slot
        no_cape = {p["ref"]: p for p in out["no_cape"]}
        self.assertNotIn("travel/cape", no_cape)
        self.assertEqual(no_cape["travel/tunic"]["hidden"], [])
        smock = [p["ref"] for p in out["smock"]]
        self.assertIn("work/smock", smock)
        self.assertNotIn("travel/tunic", smock)
        self.assertEqual(out["kept"], 0)
        # a slider edit moves the pieces at once, and the push-out follows when it rests
        self.assertFalse(out["fast"]["full"])
        self.assertNotIn("push", out["fast"])
        self.assertTrue(out["settled"]["full"])
        print(f"slider: fast {out['fast']}, settled {out['settled']}")
        # the heavier body, with the cape hidden and the smock in the tunic's slot
        heavy = card_body(Character(phenotype={"weight": 0.9}), template_model())
        worn = worn_pieces(
            self.card, self.dir, show=["work/smock"], hide=["travel/cape"]
        )
        self.compare(out["heavy"], dress(heavy, worn), out["scale"], 0.001, 0.006)
        # the files stay in the browser
        self.assertFalse(
            [r for r in out["requests"] if r[0] != "GET" or "wearable" in r[1]]
        )


if __name__ == "__main__":
    unittest.main()

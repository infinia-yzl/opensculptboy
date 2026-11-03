# OpenSculptBoy
# Apache License, Version 2.0
"""
The studio's display states on every mesh of the built page (viewer/src/studio, DESIGN-0014):
X-ray, the cut-away and the parts hidden or isolated by bone reach the worn pieces, the props,
the eyes and the hair as well as the body, so a point by a sword hand can be reached. In
Chromium through Playwright (local only), with the synthetic outfit of test_viewer_wearables (a
tunic, a cape and a rigid circlet) on Anny's default body: every mesh's shader calls the cut;
X-ray turns every mesh see-through; with every bone hidden, and with the cut-away opened past
the figure, the picture above the floor matches the picture with every mesh switched off.
"""

import os
import pathlib
import shutil
import tempfile
import unittest

import numpy as np

from test.markers import local_only
from test.test_hair_page import CHROMIUM, PAGE, playwright_available

ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestStudioDisplayPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        from test.test_viewer_wearables import _packages
        from test.test_viewer_wearables_page import _card_files

        torch.set_num_threads(1)
        cls.dir = pathlib.Path(tempfile.mkdtemp())
        _packages(cls.dir)
        cls.files = sorted(str(f) for f in _card_files(cls.dir))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_display_states(self):
        from io import BytesIO

        from PIL import Image
        from playwright.sync_api import sync_playwright

        def picture(tab):
            tab.evaluate("() => window.setFrame('body')")
            return np.asarray(
                Image.open(BytesIO(tab.screenshot())).convert("RGB"), dtype=np.float32
            )

        with sync_playwright() as p:
            kw = dict(executable_path=str(CHROMIUM)) if CHROMIUM.exists() else {}
            browser = p.chromium.launch(args=ARGS, **kw)
            tab = browser.new_context(viewport=dict(width=700, height=800)).new_page()
            errors = []
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.on(
                "console",
                lambda m: errors.append(m.text)
                if m.type == "error" and "Shader Error" in m.text
                else None,
            )
            tab.goto(PAGE.as_uri() + "?shot&acc=1&base")
            tab.set_default_timeout(120000)
            tab.wait_for_function("() => window.__ready === true", timeout=300000)
            tab.locator(
                "input[type=file][multiple]:not([webkitdirectory])"
            ).set_input_files(self.files)
            tab.wait_for_function(
                "() => window.wearStats() && window.wearStats().pieces.length > 0"
                " && window.wearStats().timing.full === true",
                timeout=120000,
            )
            meshes = tab.evaluate("() => window.__studio.meshes()")
            # the body, its eyes, the hair and at least the outfit's three pieces, each with the cut in its shader
            self.assertGreaterEqual(len(meshes), 5, meshes)
            self.assertTrue(all(m["cut"] for m in meshes), meshes)
            # X-ray: every mesh drawn see-through, and solid again after
            tab.keyboard.press("Alt+KeyZ")
            tab.evaluate("() => window.setFrame('body')")
            xr = tab.evaluate("() => window.__studio.meshes()")
            self.assertTrue(all(all(a < 0.5 for a in m["xray"]) for m in xr), xr)
            tab.keyboard.press("Alt+KeyZ")
            tab.evaluate("() => window.setFrame('body')")
            self.assertTrue(
                all(
                    all(a == 1 for a in m["xray"])
                    for m in tab.evaluate("() => window.__studio.meshes()")
                )
            )
            full = picture(tab)
            # the picture without any mesh, above the floor (the floor keeps the shadows of the shadow maps)
            tab.evaluate("() => window.__studio.showMeshes(false)")
            empty = picture(tab)
            tab.evaluate("() => window.__studio.showMeshes(true)")
            top = slice(0, int(full.shape[0] * 0.78))
            self.assertGreater(np.abs(full[top] - empty[top]).mean(), 1.0)
            # every bone hidden: the body, the pieces on their bones, the eyes and the hair leave the picture
            tab.evaluate("() => window.__studio.parts(window.__studio.bones(), null)")
            hidden = picture(tab)
            out = os.environ.get("STUDIO_DISPLAY_OUT")
            if out:
                for n, a in (("full", full), ("empty", empty), ("hidden", hidden)):
                    Image.fromarray(a.astype(np.uint8)).save(
                        os.path.join(out, n + ".png")
                    )
            tab.evaluate("() => window.__studio.parts(null, null)")
            self.assertLess(np.abs(hidden[top] - empty[top]).max(), 12, "hidden")
            self.assertLess(np.abs(hidden[top] - empty[top]).mean(), 0.3, "hidden")
            # the cut-away opened past the figure: every surface leaves the picture
            tab.evaluate("() => window.__studio.cut(100)")
            cut = picture(tab)
            tab.evaluate("() => window.__studio.cut(0)")
            self.assertLess(np.abs(cut[top] - empty[top]).max(), 12, "cut")
            self.assertLess(np.abs(cut[top] - empty[top]).mean(), 0.3, "cut")
            # isolate the head's bones: the rest turns faint, the head and the rigid circlet keep their look
            tab.evaluate(
                "() => window.__studio.parts(null, window.__studio.bones().filter((n) => /^(head|neck)/.test(n)))"
            )
            iso = picture(tab)
            tab.evaluate("() => window.__studio.parts(null, null)")
            self.assertGreater(np.abs(iso[top] - full[top]).mean(), 0.5)
            browser.close()
        self.assertFalse(errors, errors)


if __name__ == "__main__":
    unittest.main()

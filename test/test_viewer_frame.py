# OpenSculptBoy
# Apache License, Version 2.0
"""
The frame of the viewer page (viewer/src/shell): the six modes, the panels, the View menu's theme,
the welcome card, the one Open path and the phone layout. In node: the actions of the interface's
state (viewer/test/frame.mjs). In the built page, in Chromium with SwiftShader through Playwright
(local only): the top bar's six modes, the welcome card of a first visit, the theme, the files that
a drop routes (sheet settings, a character of the page, a GLB without a card, and a card through
the Open path's file input), Ctrl O, the phone's bar of six modes with its bottom sheet, and the
interface kept out of the still pictures (?shot).
"""

import base64
import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

from test.markers import local_only

REPO = pathlib.Path(__file__).resolve().parents[1]
PAGE = REPO / "viewer" / "dist" / "anny_viewer.html"
CHROMIUM = pathlib.Path("/opt/pw-browsers/chromium")
SCRIPT = REPO / "viewer" / "test" / "frame.mjs"
ARGS = ["--use-angle=swiftshader", "--use-gl=angle", "--enable-unsafe-swiftshader"]
MODES = ["Create", "Dress", "Express", "Pose", "Sheet", "Export"]
# a drop of files by name and bytes, as a script does (a DataTransfer of files)
DROP = """(files) => {
  const dt = new DataTransfer();
  for (const [name, b64] of Object.entries(files)) dt.items.add(new File([Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))], name));
  document.dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true }));
}"""


def node_available():
    if shutil.which("node") is None:
        return False
    out = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout
    return int(out.strip().lstrip("v").split(".")[0]) >= 22


def playwright_available():
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return PAGE.exists()


def b64(obj) -> str:
    data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
    return base64.b64encode(data).decode()


@unittest.skipUnless(node_available(), "node 22 or later is needed to run the page's TypeScript")
class TestFrameState(unittest.TestCase):
    def test_state(self):
        r = subprocess.run(["node", str(SCRIPT)], capture_output=True, text=True, cwd=REPO)
        self.assertEqual(r.returncode, 0, r.stderr[-4000:])
        out = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertTrue(out["ok"])
        self.assertEqual(out["modes"], [m.lower() for m in MODES])


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(playwright_available(), "Playwright and the built viewer page are needed")
class TestFramePage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = pathlib.Path(tempfile.mkdtemp()) / "card"
        cls.dir.mkdir()
        (cls.dir / "character.json").write_text(
            json.dumps({"schema": 1, "name": "Frame test", "phenotype": {"muscle": 0.8}}), encoding="utf-8"
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir.parent, ignore_errors=True)

    def open_page(self, p, width=1440, height=900, touch=False, query="?acc=1&blink=off"):
        kw = dict(executable_path=str(CHROMIUM)) if CHROMIUM.exists() else {}
        browser = p.chromium.launch(args=ARGS, **kw)
        ctx = browser.new_context(viewport=dict(width=width, height=height), has_touch=touch, is_mobile=touch)
        tab = ctx.new_page()
        errors = []
        tab.on("pageerror", lambda e: errors.append(str(e)))
        tab.goto(PAGE.as_uri() + query)
        tab.set_default_timeout(120000)
        tab.wait_for_function("() => window.__ready === true", timeout=300000)
        return browser, tab, errors

    def test_frame(self):
        from playwright.sync_api import sync_playwright

        state = "() => { const s = __shell.ui.get(); return { mode: s.mode, open: s.open, sub: s.sub, dialog: s.openDialog, error: s.openError, welcome: s.welcome }; }"
        with sync_playwright() as p:
            browser, tab, errors = self.open_page(p)
            # the top bar: the six modes with their keys, the mode's panel and the welcome card of a first visit
            modes = tab.eval_on_selector_all("nav[aria-label='Modes'] .osb-mode", "(bs) => bs.map((b) => b.firstChild.textContent)")
            self.assertEqual(modes, MODES)
            self.assertEqual(tab.eval_on_selector_all("nav[aria-label='Modes'] .osb-hint", "(h) => h.map((x) => x.textContent).join('')"), "CDXPSE")
            self.assertTrue(tab.locator("#sec-body").is_visible())
            self.assertTrue(tab.locator(".welcome").is_visible())
            # the mode buttons
            tab.click("#tab-export")
            self.assertTrue(tab.locator("#sec-export").is_visible())
            tab.click("#tab-export")
            self.assertFalse(tab.evaluate(state)["open"])
            # the theme of the View menu
            for theme in ("dark", "light", "dark"):
                tab.evaluate(f"() => __shell.actions.setTheme('{theme}')")
                self.assertEqual(tab.evaluate("() => document.documentElement.dataset.theme"), theme)
            # the welcome card closes and stays closed on the next visit
            tab.locator(".welcome button", has_text="Got it").click()
            self.assertEqual(tab.locator(".welcome").count(), 0)
            # a drop of sheet settings opens them in the Sheet mode
            tab.evaluate(DROP, {"settings.json": b64({"format": "opensculptboy/sheet-settings@1", "name": "Dropped sheet"})})
            tab.wait_for_function("() => __shell.ui.get().mode === 'sheet'")
            self.assertEqual(tab.evaluate("() => window.sheet.settings().name"), "Dropped sheet")
            # a character of the page (Export's .json) opens in Create
            look = {"format": "anny-viewer/look@3", "name": "Dropped look", "phenotype": {"weight": 0.9}}
            tab.evaluate(DROP, {"look.json": b64(look)})
            tab.wait_for_function("() => window.__look().name === 'Dropped look'")
            self.assertEqual(tab.evaluate(state)["mode"], "create")
            # a GLB alone: the Open dialog says what happened and how to go on
            tab.evaluate(DROP, {"model.glb": b64(b"glTF\x02\x00\x00\x00")})
            tab.wait_for_function("() => __shell.ui.get().openDialog")
            alert = tab.locator("[role='alert']").inner_text()
            self.assertIn("GLB", alert)
            self.assertIn("How to fit a 3D model", alert)
            tab.keyboard.press("Escape")
            tab.wait_for_function("() => !__shell.ui.get().openDialog")
            # Ctrl O opens the dialog; its file input reads a card, which opens in Create
            tab.mouse.click(900, 450)
            tab.keyboard.press("Control+o")
            tab.wait_for_function("() => __shell.ui.get().openDialog")
            tab.locator("#open-files").set_input_files([str(f) for f in self.dir.iterdir()])
            tab.wait_for_function("() => window.wearStats() && window.wearStats().card === 'Frame test'")
            tab.wait_for_function("() => !__shell.ui.get().openDialog")
            self.assertEqual(tab.evaluate(state)["mode"], "create")
            self.assertIn("Frame test", tab.locator(".osb-identity").inner_text())
            self.assertEqual(tab.locator("input[type=file][multiple]:not([webkitdirectory])").count(), 1)
            # the next visit starts without the welcome card, in the last mode
            tab.reload()
            tab.wait_for_function("() => window.__ready === true", timeout=300000)
            self.assertEqual(tab.locator(".welcome").count(), 0)
            self.assertFalse(errors, errors)
            browser.close()

            # a phone: the six modes in the bottom bar, the panel as a bottom sheet, touch words in the welcome card
            browser, tab, errors = self.open_page(p, 390, 844, touch=True)
            self.assertEqual(tab.locator(".phone-bar .osb-phonetab").count(), 6)
            self.assertEqual(tab.locator("nav.osb-modes").count(), 0)
            self.assertIn("pinch", tab.locator(".welcome").inner_text())
            tab.locator(".phone-bar .osb-phonetab", has_text="Dress").click()
            self.assertEqual(tab.evaluate("() => document.getElementById('insp-left').dataset.mode"), "dress")
            box = tab.locator("#insp-left").bounding_box()
            self.assertGreater(box["y"], 300)   # a bottom sheet over the lower half
            self.assertLess(box["y"] + box["height"], 844 - 40)   # above the bar
            self.assertFalse(errors, errors)
            browser.close()

            # the still pictures of the tests: no interface
            browser, tab, errors = self.open_page(p, query="?shot&acc=1")
            self.assertEqual(tab.evaluate("() => getComputedStyle(document.getElementById('ui-root')).display"), "none")
            browser.close()


if __name__ == "__main__":
    unittest.main()

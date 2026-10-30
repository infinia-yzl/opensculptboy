# OpenSculptBoy
# Apache License, Version 2.0
"""
The frame of the viewer page (viewer/src/shell): the six modes, the panels, the View menu's theme,
the welcome card, the one Open path, the New screen, the Character panel, Save card and the phone
layout. In node: the actions of the interface's state (viewer/test/frame.mjs). In the built page,
in Chromium through Playwright (local only): the first visit and its welcome card with the statue's
download link, a card's pose,
the top bar's six modes, the theme, the figure framed between the bars at every framing, the files
that a drop routes (sheet settings, a character of the page, a GLB without a card, a GLB with a card
of sliders, a GLB with a fitted card that asks for its files, a card's .zip, and a card through the
Open path's file input), Ctrl O, Save card (Ctrl S) and the saved card opened again, the New
screen's statues and blank body, the stone library, the Character panel, View > Settings, the phone's bar of six
modes with its bottom sheet, and the interface kept out of the still pictures (?shot).
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
  for (const [name, b64] of Object.entries(files))
    dt.items.add(new File([Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))], name));
  document.dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true }));
}"""


def glb_with(card):
    """a GLB file whose scene extras hold a card, as OpenSculptBoy's exporters write it"""
    text = json.dumps(
        {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [], "extras": {"opensculptboy": {"character": card}}}],
        }
    )
    text += " " * (-len(text) % 4)
    body = text.encode()
    import struct

    return (
        struct.pack("<III", 0x46546C67, 2, 20 + len(body))
        + struct.pack("<II", len(body), 0x4E4F534A)
        + body
    )


def zip_of(files):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


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


@unittest.skipUnless(
    node_available(), "node 22 or later is needed to run the page's TypeScript"
)
class TestFrameState(unittest.TestCase):
    def test_state(self):
        r = subprocess.run(
            ["node", str(SCRIPT)], capture_output=True, text=True, cwd=REPO
        )
        self.assertEqual(r.returncode, 0, r.stderr[-4000:])
        out = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertTrue(out["ok"])
        self.assertEqual(out["modes"], [m.lower() for m in MODES])


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestFramePage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = pathlib.Path(tempfile.mkdtemp()) / "card"
        cls.dir.mkdir()
        (cls.dir / "character.json").write_text(
            json.dumps(
                {"schema": 1, "name": "Frame test", "phenotype": {"muscle": 0.8}}
            ),
            encoding="utf-8",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir.parent, ignore_errors=True)

    def open_page(
        self, p, width=1440, height=900, touch=False, query="?acc=1&blink=off"
    ):
        kw = dict(executable_path=str(CHROMIUM)) if CHROMIUM.exists() else {}
        browser = p.chromium.launch(args=ARGS, **kw)
        ctx = browser.new_context(
            viewport=dict(width=width, height=height), has_touch=touch, is_mobile=touch
        )
        tab = ctx.new_page()
        errors = []
        tab.on("pageerror", lambda e: errors.append(str(e)))
        tab.goto(PAGE.as_uri() + query)
        tab.set_default_timeout(120000)
        tab.wait_for_function("() => window.__ready === true", timeout=300000)
        return browser, tab, errors

    def test_frame(self):
        from playwright.sync_api import sync_playwright

        state = (
            "() => { const s = __shell.ui.get(); return { mode: s.mode, open: s.open, sub: s.sub,"
            " dialog: s.openDialog, error: s.openError, welcome: s.welcome }; }"
        )
        with sync_playwright() as p:
            browser, tab, errors = self.open_page(p)
            # a first visit of a page opened as a file stays on Anny's default body, and the welcome card gives the
            # default statue's download link (a page opened as a file reads nothing beside it)
            first = tab.evaluate(
                "() => ({ look: window.__look(), clip: window.viewState().clip })"
            )
            self.assertEqual(first["look"]["name"], "Baseline")
            self.assertEqual(first["clip"], "a_pose")
            tab.wait_for_function("() => __shell.ui.get().starter.note !== null")
            self.assertEqual(
                tab.evaluate("() => __shell.ui.get().starter.note.problem"), "file"
            )
            self.assertIn("Meet the studio", tab.locator(".welcome").inner_text())
            self.assertTrue(
                tab.locator(".welcome .welcome-starter a")
                .get_attribute("href")
                .endswith("/doryphoros.zip")
            )
            # the figure stands between the top bar and the view bar at every framing and angle
            bars = tab.evaluate(
                "() => ({ top: document.querySelector('.osb-top').getBoundingClientRect().bottom,"
                " low: document.querySelector('.viewbar').getBoundingClientRect().top })"
            )
            for frame in ("body", "upper", "face"):
                for yaw in (0, 24, 90, 180):
                    tab.evaluate(
                        f"() => {{ window.__view.VIEW.yaw = {yaw}; window.setFrame('{frame}'); }}"
                    )
                    fig = tab.evaluate("() => window.figureOnScreen()")
                    self.assertGreater(fig["top"], bars["top"], (frame, yaw, fig, bars))
                    if frame == "body":
                        self.assertLess(
                            fig["bottom"], bars["low"], (frame, yaw, fig, bars)
                        )
            # the top bar: the six modes with their keys, the mode's panel and the welcome card of a first visit
            modes = tab.eval_on_selector_all(
                "nav[aria-label='Modes'] .osb-mode",
                "(bs) => bs.map((b) => b.firstChild.textContent)",
            )
            self.assertEqual(modes, MODES)
            self.assertEqual(
                tab.eval_on_selector_all(
                    "nav[aria-label='Modes'] .osb-hint",
                    "(h) => h.map((x) => x.textContent).join('')",
                ),
                "CDXPSE",
            )
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
                self.assertEqual(
                    tab.evaluate("() => document.documentElement.dataset.theme"), theme
                )
            # the welcome card closes and stays closed on the next visit
            tab.locator(".welcome button", has_text="Got it").click()
            self.assertEqual(tab.locator(".welcome").count(), 0)
            # a card's pose (each bone's turn in anny's frame) stands the figure in it, and the card of the pose
            # shown gives it back
            turn = [0.0, 0.0, 0.258819, 0.965926]
            tab.evaluate(
                DROP,
                {
                    "character.json": base64.b64encode(
                        json.dumps(
                            {
                                "schema": 5,
                                "name": "Fitted",
                                "phenotype": {"muscle": 0.3},
                                "face_shapes": {"nose-hump": 0.5},
                                "pose": {"bones": {"upperarm01.L": turn}},
                                "look": {"skin": {"stone": "carrara"}},
                            }
                        ).encode()
                    ).decode()
                },
            )
            tab.wait_for_function("() => window.viewState().clip === 'card:Fitted'")
            fitted = tab.evaluate(
                "() => ({ look: window.__look(), card: window.poseCard() })"
            )
            self.assertEqual(fitted["look"]["skin"]["stone"], "carrara")
            self.assertAlmostEqual(fitted["look"]["phenotype"]["muscle"], 0.3)
            self.assertEqual(fitted["look"]["face"], {"nose-hump": 0.5})
            self.assertEqual(sorted(fitted["card"]["pose"]["bones"]), ["upperarm01.L"])
            for a, b in zip(fitted["card"]["pose"]["bones"]["upperarm01.L"], turn):
                self.assertAlmostEqual(a, b, delta=2e-4)
            # a drop of sheet settings opens them in the Sheet mode
            tab.evaluate(
                DROP,
                {
                    "settings.json": b64(
                        {
                            "format": "opensculptboy/sheet-settings@1",
                            "name": "Dropped sheet",
                        }
                    )
                },
            )
            tab.wait_for_function("() => __shell.ui.get().mode === 'sheet'")
            self.assertEqual(
                tab.evaluate("() => window.sheet.settings().name"), "Dropped sheet"
            )
            # a character of the page (Export's .json) opens in Create
            look = {
                "format": "anny-viewer/look@3",
                "name": "Dropped look",
                "phenotype": {"weight": 0.9},
            }
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
            # a GLB with a card of sliders in its scene extras opens whole
            slider_card = {
                "schema": 5,
                "name": "From a GLB",
                "phenotype": {"weight": 0.2},
                "look": {
                    "skin": {
                        "tone": 0.6,
                        "undertone": 0,
                        "stone": "basalt",
                        "polish": 0.3,
                    }
                },
            }
            tab.evaluate(
                DROP, {"from.glb": base64.b64encode(glb_with(slider_card)).decode()}
            )
            tab.wait_for_function("() => window.__look().name === 'From a GLB'")
            self.assertEqual(tab.evaluate("() => window.__look().skin.stone"), "basalt")
            # a GLB with a fitted card asks for its files, and opens its sliders alone on request
            fitted = {
                "schema": 5,
                "name": "Fitted",
                "phenotype": {"weight": 0.7},
                "detail": {
                    "levels": [{"subdiv": 0, "offsets": "fitted.detail0.safetensors"}]
                },
            }
            tab.evaluate(
                DROP, {"fitted.vrm": base64.b64encode(glb_with(fitted)).decode()}
            )
            tab.wait_for_function("() => __shell.ui.get().openDialog")
            alert = tab.locator("[role='alert']").inner_text()
            self.assertIn("without its files", alert)
            tab.locator(
                "[role='alert'] button", has_text="Open the sliders alone"
            ).click()
            tab.wait_for_function("() => window.__look().name === 'Fitted'")
            # a card's .zip opens as its files
            tab.evaluate(
                DROP,
                {
                    "zipped.zip": base64.b64encode(
                        zip_of(
                            {
                                "zipped/character.json": json.dumps(
                                    {
                                        "schema": 1,
                                        "name": "Zipped",
                                        "phenotype": {"height": 0.9},
                                    }
                                )
                            }
                        )
                    ).decode()
                },
            )
            tab.wait_for_function("() => window.__look().name === 'Zipped'")
            # Save card writes the character with its look block, and the saved card opens again with its stone
            tab.evaluate("() => __shell.actions.openMode('create', 'skin')")
            tab.locator("#sec-skin [data-stone='bronze']").click()
            tab.wait_for_function("() => window.__look().skin.stone === 'bronze'")
            tab.mouse.click(900, 450)
            with tab.expect_download() as dl:
                tab.keyboard.press("Control+s")
            saved = json.loads(
                pathlib.Path(dl.value.path()).read_text(encoding="utf-8")
            )
            self.assertEqual(saved["schema"], 5)
            self.assertEqual(saved["look"]["skin"]["stone"], "bronze")
            self.assertEqual(saved["name"], "Zipped")
            tab.evaluate("() => window.setLook({ name: 'Other' })")
            tab.evaluate(DROP, {"character.json": b64(saved)})
            tab.wait_for_function(
                "() => window.__look().name === 'Zipped' && window.__look().skin.stone === 'bronze'"
            )
            # the New screen: the two statues, Doryphoros first, and the blank body; a statue whose package the page
            # cannot read gives its note with the download link, and the blank body starts at once
            tab.mouse.click(900, 450)
            tab.keyboard.press("n")
            tab.wait_for_function(
                "() => document.querySelectorAll('[data-starter]').length === 3"
            )
            self.assertEqual(
                tab.eval_on_selector_all(
                    "[data-starter]", "(bs) => bs.map((b) => b.dataset.starter)"
                ),
                ["doryphoros", "david_goliath", "blank"],
            )
            tab.locator("[data-starter='david_goliath']").click()
            tab.locator(".new-start").click()
            tab.wait_for_selector(".new-note")
            self.assertTrue(
                tab.locator(".new-note a")
                .get_attribute("href")
                .endswith("/david_goliath.zip")
            )
            tab.locator("[data-starter='blank']").click()
            tab.locator(".new-start").click()
            tab.wait_for_function("() => !__shell.ui.get().newDialog")
            self.assertEqual(tab.evaluate("() => window.viewState().clip"), "a_pose")
            # the Character panel: what the character holds
            tab.locator(".osb-identity").click()
            panel = tab.locator("#char-panel").inner_text()
            for word in ("Body", "Face", "Skin", "Hair", "Outfit"):
                self.assertIn(word, panel)
            tab.locator(".osb-identity").click()
            # View > Settings: the AI providers
            tab.locator("[aria-label='View menu']").click()
            tab.locator("[data-slot='settings']").click()
            tab.wait_for_function("() => __shell.ui.get().tool === 'settings'")
            self.assertTrue(tab.locator("#sec-settings").is_visible())
            tab.evaluate("() => __shell.actions.closeTool()")
            # Ctrl O opens the dialog; its file input reads a card, which opens in Create
            tab.mouse.click(900, 450)
            tab.keyboard.press("Control+o")
            tab.wait_for_function("() => __shell.ui.get().openDialog")
            tab.locator("#open-files").set_input_files(
                [str(f) for f in self.dir.iterdir()]
            )
            tab.wait_for_function(
                "() => window.wearStats() && window.wearStats().card === 'Frame test'"
            )
            tab.wait_for_function("() => !__shell.ui.get().openDialog")
            self.assertEqual(tab.evaluate(state)["mode"], "create")
            self.assertIn("Frame test", tab.locator(".osb-identity").inner_text())
            self.assertEqual(
                tab.locator(
                    "input[type=file][multiple]:not([webkitdirectory])"
                ).count(),
                1,
            )
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
            self.assertEqual(
                tab.evaluate("() => document.getElementById('insp-left').dataset.mode"),
                "dress",
            )
            box = tab.locator("#insp-left").bounding_box()
            self.assertGreater(box["y"], 300)  # a bottom sheet over the lower half
            self.assertLess(box["y"] + box["height"], 844 - 40)  # above the bar
            self.assertFalse(errors, errors)
            browser.close()

            # the still pictures of the tests: no interface
            browser, tab, errors = self.open_page(p, query="?shot&acc=1")
            self.assertEqual(
                tab.evaluate(
                    "() => getComputedStyle(document.getElementById('ui-root')).display"
                ),
                "none",
            )
            browser.close()


if __name__ == "__main__":
    unittest.main()

# OpenSculptBoy
# Apache License, Version 2.0
"""
The keyboard shortcuts of the viewer page (viewer/src/ui/index.ts, the table and the help's
list in viewer/src/ui/shortcuts.ts). In node: the keys that a keydown event means on other
keyboards (Caps Lock, a keyboard in another script, an input method, AZERTY's digit row) and the
help's list against the keys of the modes, of Create's sections and of the View menu's tools. In
the built page, in Chromium with SwiftShader through Playwright (local only): the steps a person
takes (a click on the figure, D for the Dress mode, a card opened through Open folder, a slider, a
switch, a segmented button, a fold, the help closed by its button, a name typed into the Sheet
section and left with Esc), then every shortcut of the help with its effect, the shortcuts kept
off while a text field has the focus, and the panel keys of other keyboards as synthetic events.
"""

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

from test.markers import local_only

REPO = pathlib.Path(__file__).resolve().parents[1]
# as test.test_hair_page and test.test_viewer_parity give them, without their imports of the model
PAGE = REPO / "viewer" / "dist" / "anny_viewer.html"
CHROMIUM = pathlib.Path("/opt/pw-browsers/chromium")


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


SCRIPT = REPO / "viewer" / "test" / "shortcuts.mjs"
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
# what each panel key opens (viewer/src/ui/shortcuts.ts PANEL_KEYS): its panel (the mode's on the
# left, the View menu's tools on the right), its mode, and its section (None: the mode's last one)
PANELS = {
    "c": ("left", "create", None),
    "d": ("left", "dress", "wear"),
    "x": ("left", "express", "expression"),
    "p": ("left", "pose", "pose"),
    "s": ("left", "sheet", "sheet"),
    "e": ("left", "export", "export"),
    "b": ("left", "create", "body"),
    "f": ("left", "create", "face"),
    "h": ("left", "create", "hair"),
    "k": ("left", "create", "skin"),
    "r": ("right", None, "rig"),
    "l": ("right", None, "scene"),
}
# the id of each panel key in PANEL_KEYS: a mode, a section of Create or a tool
PANEL_IDS = {k: (sec if mode is None or k in "bfhk" else mode) for k, (_, mode, sec) in PANELS.items()}
# the sections that show, the dialogs, the focus and what the view shortcuts change
STATE = """() => {
  const vis = (side) => {
    const p = document.getElementById('insp-' + side);
    if (!p || p.hidden) return null;
    const s = p.querySelector('.sec:not([hidden])');
    return s ? s.id.replace('sec-', '') : null;
  };
  const a = document.activeElement;
  const focus = a ? a.tagName.toLowerCase() + (a.type ? '[' + a.type + ']' : '') : null;
  const lp = document.getElementById('insp-left');
  return Object.assign({ left: vis('left'), right: vis('right'), mode: lp ? lp.dataset.mode : null,
    help: !!document.querySelector('dialog.help[open]'),
    stats: !document.getElementById('stats').hidden, focus,
    look: JSON.stringify(window.__look().phenotype) }, window.viewState());
}"""


@unittest.skipUnless(
    node_available(), "node 22 or later is needed to run the page's TypeScript"
)
class TestShortcutKeys(unittest.TestCase):
    def test_keys(self):
        r = subprocess.run(
            ["node", str(SCRIPT)], capture_output=True, text=True, cwd=REPO
        )
        self.assertEqual(r.returncode, 0, r.stderr[-4000:])
        out = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertTrue(out["ok"])
        self.assertEqual(
            {k: v for k, v in out["panels"].items()},
            {sec: key for key, sec in PANEL_IDS.items()},
        )


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestShortcutsPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # a card folder as a person opens it: a card of its own, without packages
        cls.dir = pathlib.Path(tempfile.mkdtemp()) / "card"
        cls.dir.mkdir()
        (cls.dir / "character.json").write_text(
            json.dumps(
                {"schema": 1, "name": "Shortcut test", "phenotype": {"muscle": 0.7}}
            ),
            encoding="utf-8",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir.parent, ignore_errors=True)

    def test_shortcuts(self):
        from playwright.sync_api import sync_playwright

        log = []
        with sync_playwright() as p:
            kw = dict(executable_path=str(CHROMIUM)) if CHROMIUM.exists() else {}
            browser = p.chromium.launch(args=ARGS, **kw)
            tab = browser.new_context(viewport=dict(width=1000, height=700)).new_page()
            errors = []
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.goto(PAGE.as_uri() + "?acc=1&blink=off")
            tab.set_default_timeout(120000)
            tab.wait_for_function("() => window.__ready === true", timeout=300000)
            state = lambda: tab.evaluate(STATE)  # noqa: E731

            def settle():
                tab.wait_for_function(
                    "() => !window.viewState().tweening", timeout=60000
                )

            def press(key, label=""):
                tab.keyboard.press(key)
                tab.wait_for_timeout(60)
                return state()

            def shows(key, st):
                """the panel of the key shows: its section, or for a mode any of its sections"""
                side, mode, sec = PANELS[key]
                if sec is None:
                    return st["mode"] == mode and st[side] is not None
                return st[side] == sec and (mode is None or st["mode"] == mode)

            def panel_works(key, step):
                """the panel key opens its panel, or closes it when it shows"""
                side = PANELS[key][0]
                before = state()
                after = press(key)
                ok = after[side] is None if shows(key, before) else shows(key, after)
                log.append((step, key, before["focus"], ok))
                self.assertTrue(ok, (step, key, before, after))

            # ---------------------------------------------------------------- a person's steps
            tab.mouse.click(500, 350)
            panel_works("d", "a click on the figure")
            self.assertEqual(state()["left"], "wear")
            # Open folder: the button opens the folder chooser, and the focus stays on the button
            with tab.expect_file_chooser() as fc:
                tab.locator("#sec-wear button", has_text="Open folder").click()
            try:
                fc.value.set_files(str(self.dir))
            except Exception:
                fc.value.set_files([str(f) for f in self.dir.iterdir()])
            tab.wait_for_function(
                "() => window.wearStats() && window.wearStats().card === 'Shortcut test'",
                timeout=120000,
            )
            panel_works("x", "the card opened through Open folder")
            panel_works("s", "the card opened through Open folder")
            # a slider of the Body section: the Create mode, then its Body tab
            if state()["mode"] != "create" or state()["left"] is None:
                tab.click("#tab-create")
            tab.click("#tab-body")
            tab.locator("#sec-body input[type=range]").first.click()
            panel_works("x", "a click on a slider")
            # a switch and a segmented button of the Sheet section
            if state()["left"] != "sheet":
                tab.click("#tab-sheet")
            tab.locator("#sec-sheet .seg button", has_text="Paper").first.click()
            panel_works("d", "a click on a segmented button")
            if state()["left"] != "sheet":
                tab.click("#tab-sheet")
            tab.locator("#sec-sheet button.switch").first.click()
            panel_works("e", "a click on a switch")
            if state()["left"] != "sheet":
                tab.click("#tab-sheet")
            tab.locator("#sec-sheet summary").first.click()
            panel_works("c", "a click on a fold")
            # the help, closed by its button
            press("?")
            self.assertTrue(state()["help"])
            tab.locator("dialog.help[open] .help-head button").click()
            self.assertFalse(state()["help"])
            panel_works("x", "the help closed by its button")
            # a text field keeps every key while it has the focus, and Esc leaves it
            if state()["left"] != "sheet":
                tab.click("#tab-sheet")
            field = tab.locator("#sec-sheet input.cs-input").first
            field.click()
            before = state()
            tab.keyboard.type("Wx")
            after = state()
            self.assertEqual(field.input_value(), "Wx")
            self.assertEqual(
                (before["left"], before["right"]), (after["left"], after["right"])
            )
            press("Escape")
            self.assertNotEqual(state()["focus"], "input[text]")
            panel_works("x", "Esc in a text field")

            # ---------------------------------------------------------------- every shortcut of the help
            tab.mouse.click(500, 350)
            for key in PANELS:
                panel_works(key, "the panel keys")
            for key in PANELS:
                if shows(key, state()):
                    panel_works(key, "the panel keys close their panel")
            self.assertTrue(press("`")["stats"])
            self.assertFalse(press("`")["stats"])
            self.assertTrue(press("?")["help"])
            self.assertFalse(press("Escape")["help"])
            # Esc closes the panels
            press("x")
            press("r")
            s = press("Escape")
            s = press("Escape")
            self.assertEqual((s["left"], s["right"]), (None, None))
            # the camera
            tab.mouse.click(500, 350)
            press("0")
            settle()
            rest = state()
            press("3")
            settle()
            face = state()

            def dist(s):
                return (
                    sum((a - b) ** 2 for a, b in zip(s["camera"], s["target"])) ** 0.5
                )

            self.assertLess(dist(face), dist(rest) * 0.8)
            press("2")
            settle()
            self.assertGreater(dist(state()), dist(face))
            press("1")
            settle()
            for key in "4567":
                before = state()
                press(key)
                settle()
                self.assertNotEqual(state()["camera"], before["camera"], key)
            for key, what in (
                ("ArrowLeft", "camera"),
                ("ArrowRight", "camera"),
                ("ArrowUp", "target"),
                ("ArrowDown", "target"),
                ("Shift+ArrowUp", "camera"),
                ("Shift+ArrowDown", "camera"),
            ):
                before = state()
                press(key)
                settle()
                self.assertNotEqual(state()[what], before[what], key)
            for key, closer in (("+", True), ("-", False), ("=", True), ("_", False)):
                before = dist(state())
                press(key)
                settle()
                self.assertEqual(dist(state()) < before, closer, key)
            press("0")
            settle()
            self.assertLess(
                max(abs(a - b) for a, b in zip(state()["camera"], rest["camera"])), 1e-3
            )
            # the view
            self.assertTrue(press("t")["turntable"])
            self.assertFalse(press("t")["turntable"])
            hair = state()["hair"]
            self.assertEqual(press("Shift+H")["hair"], not hair)
            self.assertEqual(press("Shift+H")["hair"], hair)
            self.assertEqual(press("Shift+B")["rig"], "skeleton")
            self.assertEqual(press("Shift+B")["rig"], "off")
            self.assertEqual(press("Shift+W")["rig"], "weights")
            self.assertEqual(press("Shift+W")["rig"], "off")
            presets = state()["presets"]
            for i, name in enumerate(presets[:4]):
                self.assertEqual(press(f"Alt+Digit{i + 1}")["preset"], name)
            press("Alt+Digit1")
            loop = tab.evaluate(
                "() => (window.__MOTION.clips.find((c) => c.kind === 'loop') || {}).name"
            )
            if loop:
                tab.evaluate("(n) => window.setMotion(n, 0, false)", loop)
                self.assertTrue(state()["playing"])
                self.assertFalse(press("Space")["playing"])
                self.assertTrue(press("Space")["playing"])
                tab.evaluate("() => window.setMotion('a_pose', 0, true)")
            # undo and redo of a slider's edit
            press("b")
            slider = tab.locator("#sec-body input[type=range]").first
            slider.focus()
            look0 = state()["look"]
            tab.keyboard.press("ArrowRight")
            tab.wait_for_function(
                "(w) => JSON.stringify(window.__look().phenotype) !== w", arg=look0
            )
            look1 = state()["look"]
            tab.mouse.click(500, 350)
            for key, want in (
                ("Control+z", look0),
                ("Control+Shift+z", look1),
                ("Control+z", look0),
                ("Control+y", look1),
            ):
                tab.keyboard.press(key)
                tab.wait_for_function(
                    "(w) => JSON.stringify(window.__look().phenotype) === w",
                    arg=want,
                    timeout=30000,
                )

            # ---------------------------------------------------------------- other keyboards
            tab.mouse.click(500, 350)
            for ev, side, sec in (
                ({"key": "X", "code": "KeyX"}, "right", "expression"),
                ({"key": "ы", "code": "KeyS"}, "right", "sheet"),
                ({"key": "Process", "code": "KeyD", "keyCode": 229}, "left", "wear"),
            ):
                for s0 in ("left", "right"):
                    if state()[s0]:
                        press("Escape")
                before = state()
                tab.evaluate(
                    "(ev) => document.body.dispatchEvent(new KeyboardEvent('keydown',"
                    " { ...ev, bubbles: true, cancelable: true }))",
                    ev,
                )
                # the interface shows the panel once the page has drawn the key's change (at most a frame or two)
                try:
                    tab.wait_for_function(
                        "([side, sec]) => { const p = document.getElementById('insp-' + side);"
                        " const s = p && !p.hidden ? p.querySelector('.sec:not([hidden])') : null;"
                        " return !!s && s.id === 'sec-' + sec; }",
                        arg=[side, sec],
                        timeout=5000,
                    )
                except Exception:
                    pass
                self.assertEqual(state()[side], sec, (ev, before, state()))
            browser.close()
        print(
            "\n"
            + "\n".join(
                f"{s:40} {k:6} focus {f:20} {'ok' if ok else 'FAIL'}"
                for s, k, f, ok in log
            )
        )
        self.assertFalse(errors, errors)


if __name__ == "__main__":
    unittest.main()

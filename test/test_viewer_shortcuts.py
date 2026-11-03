# OpenSculptBoy
# Apache License, Version 2.0
"""
The keys of the viewer page: key map version 2 (viewer/src/studio/keymap.ts) in the action
registry (viewer/src/studio/actions.ts). In node (viewer/test/shortcuts.mjs): the chords that a
keydown event means on other keyboards, no two actions on one key in a scope, no key the browser
keeps, the help against the registry, and the handler's order of scopes. In the built page, in
Chromium through Playwright (local only): the modes and sections, a text field that keeps its
keys, every key of the registry with its action and effect, Ctrl+S, Alt+arrows, and the camera's
drags with recorded mouse input.
"""

import json
import pathlib
import shutil
import subprocess
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
        keys = out["keys"]
        # key map version 2: the modes on Shift and a digit, Export behind Ctrl+Shift, Create's sections on 1 to 4
        for i, mode in enumerate(["create", "dress", "express", "pose", "sheet"]):
            self.assertEqual(keys[f"global Shift+{i + 1}"], "mode." + mode)
        self.assertEqual(keys["global Ctrl+Shift+E"], "mode.export")
        for i, sec in enumerate(["body", "face", "skin", "hair"]):
            self.assertEqual(keys[f"create {i + 1}"], "create." + sec)
        for key, action in (
            ("F", "view.frame"),
            ("Home", "view.home"),
            ("`", "view.menu"),
            ("Z", "display.menu"),
            ("Alt+Z", "display.xray"),
            ("Backspace", "view.back"),
            ("F1", "help.keys"),
            ("F3", "help.commands"),
        ):
            self.assertEqual(keys["global " + key], action)


# a chord of the registry as Playwright presses it
CODES = {
    "`": "Backquote",
    "/": "Slash",
    "=": "Equal",
    "-": "Minus",
    "[": "BracketLeft",
    "]": "BracketRight",
    "Numpad.": "NumpadDecimal",
    "?": "Shift+Slash",
    "+": "Shift+Equal",
}


def pw_key(chord):
    *mods, key = chord.split("+") if chord != "+" else ["+"]
    if chord.endswith("++"):
        mods, key = chord[:-2].split("+"), "+"
    key = CODES.get(key, key)
    if len(key) == 1 and key.isalpha():
        key = "Key" + key.upper()
    elif len(key) == 1 and key.isdigit():
        key = "Digit" + key
    mods = ["Control" if m == "Ctrl" else m for m in mods]
    return "+".join(mods + [key])


# what each action of the key map changes, read from the page's state before and after the key
def cam_moved(a, b):
    return any(
        abs(x - y) > 1e-4
        for x, y in zip(
            a["cam"]["position"] + a["cam"]["pivot"],
            b["cam"]["position"] + b["cam"]["pivot"],
        )
    )


EFFECTS = {
    "view.frame": lambda a, b: True,
    "view.home": lambda a, b: True,
    "view.turn-left": cam_moved,
    "view.turn-right": cam_moved,
    "view.tilt-up": cam_moved,
    "view.tilt-down": cam_moved,
    "view.up": cam_moved,
    "view.down": cam_moved,
    "view.zoom-in": lambda a, b: b["cam"]["distance"] < a["cam"]["distance"],
    "view.zoom-out": lambda a, b: b["cam"]["distance"] > a["cam"]["distance"],
    "view.front": cam_moved,
    "view.back-side": cam_moved,
    "view.right": cam_moved,
    "view.left": cam_moved,
    "view.top": cam_moved,
    "view.bottom": cam_moved,
    "view.back": lambda a, b: b["cam"]["back"] < a["cam"]["back"] or cam_moved(a, b),
    "view.forward": lambda a, b: b["cam"]["forward"] < a["cam"]["forward"],
    "view.menu": lambda a, b: b["ring"],
    "display.menu": lambda a, b: b["ring"],
    "display.xray": lambda a, b: b["cam"]["xray"] != a["cam"]["xray"],
    "help.keys": lambda a, b: b["help"],
    "help.keys-q": lambda a, b: b["help"],
    "help.commands": lambda a, b: b["cmd"],
    "file.open": lambda a, b: b["open"],
    "panel.left": lambda a, b: b["left"] != a["left"],
    "panel.right": lambda a, b: (b["tool"], b["character"])
    != (a["tool"], a["character"]),
    "panel.full": lambda a, b: True,
    "mode.export": lambda a, b: b["mode"] == "export",
}
STATE = """() => { const s = window.__shell.ui.get(); return { mode: s.mode, open: s.open, sub: s.sub, tool: s.tool,
  character: s.character,
  left: !document.getElementById('insp-left').hidden, cam: window.__studio.cam(), status: window.__studio.status(),
  ring: !!document.querySelector('.ring-menu:not([hidden])'), help: !!document.querySelector('dialog.help-v2[open]'),
  cmd: !!document.querySelector('dialog.cmd[open]'), open: s.openDialog, ran: window.__studio.ran.slice(-3),
  url: location.href,
  look: JSON.stringify(window.__look().phenotype), tweening: window.viewState().tweening }; }"""


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestShortcutsPage(unittest.TestCase):
    """Key map version 2 in the built page: every key of the registry pressed in its scope, with its action run and
    its effect seen; a text field keeps the keys; and the camera's drags with recorded mouse input (DESIGN-0014,
    "How to test it"): no coast, the turn per pixel, the pan 1:1 under the pointer, the zoom steps from a framed
    hand, Alt+right-drag, Esc during a drag, the double-click's pivot, the view history and the Hand framing."""

    def test_keys_and_drags(self):
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            kw = dict(executable_path=str(CHROMIUM)) if CHROMIUM.exists() else {}
            browser = p.chromium.launch(args=ARGS, **kw)
            ctx = browser.new_context(
                viewport=dict(width=1600, height=900), accept_downloads=True
            )
            tab = ctx.new_page()
            errors = []
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.goto(PAGE.as_uri() + "?base&acc=1&blink=off")
            tab.set_default_timeout(120000)
            tab.wait_for_function("() => window.__ready === true", timeout=300000)
            tab.evaluate("() => window.__shell.actions.dismissWelcome()")
            state = lambda: tab.evaluate(STATE)  # noqa: E731

            def settle():
                tab.wait_for_function(
                    "() => !window.viewState().tweening", timeout=60000
                )
                tab.wait_for_timeout(50)

            def close_all():
                for _ in range(3):
                    st = state()
                    if st["ring"] or st["help"] or st["cmd"] or st["open"]:
                        tab.keyboard.press("Escape")
                        tab.wait_for_timeout(80)

            def view_focus():
                tab.evaluate(
                    "() => { document.activeElement && document.activeElement.blur && document.activeElement.blur(); }"
                )

            # ---------------------------------------------------------------- the modes and Create's sections
            view_focus()
            for i, mode in enumerate(["create", "dress", "express", "pose", "sheet"]):
                tab.keyboard.press(f"Shift+Digit{i + 1}")
                self.assertEqual(state()["mode"], mode)
            tab.keyboard.press("Shift+Digit1")
            for i, sec in enumerate(["body", "face", "skin", "hair"]):
                tab.keyboard.press(f"Digit{i + 1}")
                st = state()
                self.assertEqual((st["mode"], st["sub"]["create"]), ("create", sec))
            # version 1's letters do nothing now
            before = state()
            for k in ("c", "d", "x", "p", "s", "e", "b", "k", "r", "l"):
                tab.keyboard.press(k)
            after = state()
            self.assertEqual(
                (before["mode"], before["sub"], before["tool"]),
                (after["mode"], after["sub"], after["tool"]),
            )
            # a text field keeps every key, and Esc leaves it
            # (a slider's value box of the Body section)
            tab.evaluate("() => window.__shell.actions.openMode('create', 'body')")
            field = tab.locator("#sec-body .osb-sl-num").first
            field.click()
            c0 = tab.evaluate("() => window.__studio.cam()")
            tab.keyboard.press("Control+KeyA")
            tab.keyboard.type("F1z")
            self.assertEqual(field.input_value(), "F1z")
            self.assertEqual(state()["mode"], "create")
            self.assertEqual(
                tab.evaluate("() => window.__studio.cam()")["back"], c0["back"]
            )
            tab.keyboard.press("Escape")
            self.assertNotEqual(
                tab.evaluate("() => document.activeElement.className"), "osb-sl-num"
            )
            tab.keyboard.press("Shift+Digit1")
            tab.keyboard.press("Shift+Digit1")
            self.assertEqual(state()["mode"], "create")

            # ---------------------------------------------------------------- every key of the registry
            actions = tab.evaluate("() => window.__studio.actions()")
            # a selected part for H, Shift+H, Alt+H and /
            tab.evaluate(
                "() => window.__studio.setSelection({ label: 'Right hand', frame: () => null, bones: ['wrist.R'] })"
            )
            log = []
            for a in actions:
                if (
                    not a["run"]
                    or a["scope"] not in ("global", "create")
                    or a["id"] in ("help.escape", "file.save", "drag.cancel")
                ):
                    continue
                for chord in a["keys"]:
                    close_all()
                    view_focus()
                    if a["scope"] == "create":
                        tab.keyboard.press("Shift+Digit1")
                    if a["id"] == "view.forward":
                        tab.keyboard.press("Backspace")
                        settle()
                    if a["id"] == "panel.right":
                        tab.evaluate("() => window.__shell.ui.set({ tool: null })")
                    b0 = state()
                    tab.keyboard.press(pw_key(chord))
                    tab.wait_for_timeout(60)
                    settle()
                    b1 = state()
                    ok_ran = a["id"] in b1["ran"] or a["id"] in ("motion.play",)
                    eff = EFFECTS.get(a["id"])
                    ok_eff = eff(b0, b1) if eff else True
                    log.append((a["id"], chord, ok_ran, ok_eff))
                    self.assertTrue(ok_ran, (a["id"], chord, b1["ran"]))
                    self.assertTrue(ok_eff, (a["id"], chord, b0, b1))
                    if a["id"] == "panel.full":
                        tab.keyboard.press("Control+Space")
            close_all()
            tab.evaluate("() => window.__studio.setSelection(null)")
            # Ctrl+S saves the card as a download
            with tab.expect_download() as dl:
                tab.keyboard.press("Control+KeyS")
            self.assertTrue(dl.value.suggested_filename.endswith(".json"))
            # Alt+arrows never leave the page
            url = state()["url"]
            tab.keyboard.press("Alt+ArrowLeft")
            tab.keyboard.press("Alt+ArrowRight")
            tab.wait_for_timeout(300)
            self.assertEqual(state()["url"], url)

            # ---------------------------------------------------------------- drags with recorded mouse input
            tab.evaluate(
                "() => window.__shell.ui.set({ tool: null, character: false, open: false })"
            )
            view_focus()
            tab.keyboard.press("Home")
            settle()
            # a turn: 0.25 degrees a pixel, no coast after release
            c0 = tab.evaluate("() => window.__studio.cam()")
            tab.mouse.move(1450, 300)
            tab.mouse.down()
            for i in range(1, 21):
                tab.mouse.move(1450 - i * 10, 300)
                tab.wait_for_timeout(16)
            c1 = tab.evaluate("() => window.__studio.cam()")
            tab.mouse.up()
            tab.wait_for_timeout(500)
            c2 = tab.evaluate("() => window.__studio.cam()")
            turn = ((c1["yaw"] - c0["yaw"]) + 540) % 360 - 180
            self.assertAlmostEqual(abs(turn), 50, delta=2)
            self.assertLess(abs(c2["yaw"] - c1["yaw"]), 1)
            # Esc during a turn: back to where it began
            tab.mouse.move(1450, 300)
            tab.mouse.down()
            tab.mouse.move(1350, 300, steps=8)
            tab.keyboard.press("Escape")
            tab.mouse.up()
            settle()
            self.assertLess(
                abs(tab.evaluate("() => window.__studio.cam()")["yaw"] - c2["yaw"]), 0.5
            )
            # a middle-drag on the body: the grabbed spot stays under the pointer (1:1)
            chest = tab.evaluate(
                "() => window.__studio.project(window.__studio.joint('spine03'))"
            )
            x0, y0 = chest["x"], chest["y"]
            hit = tab.evaluate(f"() => window.__studio.bodyRay({x0}, {y0})")
            self.assertIsNotNone(hit)
            tab.mouse.move(x0, y0)
            tab.mouse.down(button="middle")
            tab.mouse.move(x0 + 120, y0 + 30, steps=15)
            tab.mouse.up(button="middle")
            s = tab.evaluate(
                f"() => window.__studio.project([{hit[0]}, {hit[1]}, {hit[2]}])"
            )
            self.assertLess(
                abs(s["x"] - (x0 + 120)) + abs(s["y"] - (y0 + 30)), 4, (s, x0, y0)
            )
            tab.keyboard.press("Backspace")
            settle()
            # the Hand framing puts every joint of the hand inside the free space
            tab.locator(".viewbar button[data-value=hand]").click()
            settle()
            pts = tab.evaluate(
                "() => window.__studio.hand('R').concat(window.__studio.hand('L'))"
            )
            inside = [
                all(0 <= x <= 1600 and 56 <= y <= 900 - 46 for x, y in pts[i : i + 20])
                for i in (0, 20)
            ]
            self.assertTrue(any(inside), pts)
            # from the framed hand to a fingertip's view (15 cm from the skin) in 10 notches or fewer; the camera stops
            # 4 cm from the skin
            side = 0 if inside[0] else 20
            hx = sum(x for x, _ in pts[side : side + 20]) / 20
            hy = sum(y for _, y in pts[side : side + 20]) / 20
            tab.mouse.move(hx, hy)
            n, near = 0, None
            while n < 40:
                tab.mouse.wheel(0, -100)
                tab.wait_for_timeout(40)
                n += 1
                h = tab.evaluate(f"() => window.__studio.bodyRay({hx}, {hy})")
                if h and h[3] < 0.15 and near is None:
                    near = n
                if h and h[3] < 0.045:
                    break
            self.assertIsNotNone(near)
            self.assertLessEqual(near, 10)
            self.assertGreater(h[3], 0.035)
            # Alt+right-drag zooms
            d1 = tab.evaluate("() => window.__studio.cam().distance")
            tab.keyboard.down("Alt")
            tab.mouse.down(button="right")
            tab.mouse.move(800, 550, steps=10)
            tab.mouse.up(button="right")
            tab.keyboard.up("Alt")
            self.assertGreater(
                tab.evaluate("() => window.__studio.cam().distance"), d1 * 1.2
            )
            # the view history goes back past those moves
            backs = tab.evaluate("() => window.__studio.cam().back")
            self.assertGreaterEqual(backs, 3)
            # a double-click on the body sets the pivot on its skin; on the background it frames the whole figure
            tab.keyboard.press("Home")
            settle()
            hip = tab.evaluate(
                "() => window.__studio.project(window.__studio.joint('pelvis.L'))"
            )
            hit = tab.evaluate(f"() => window.__studio.bodyRay({hip['x']}, {hip['y']})")
            self.assertIsNotNone(hit)
            tab.mouse.dblclick(hip["x"], hip["y"])
            settle()
            piv = tab.evaluate("() => window.__studio.cam().pivot")
            self.assertLess(
                sum((a - b) ** 2 for a, b in zip(piv, hit[:3])) ** 0.5, 0.01
            )
            tab.mouse.dblclick(1500, 120)
            settle()
            self.assertGreater(
                sum(
                    (a - b) ** 2
                    for a, b in zip(
                        tab.evaluate("() => window.__studio.cam().pivot"), piv
                    )
                )
                ** 0.5,
                0.05,
            )
            browser.close()
        print(
            "\n" + "\n".join(f"{i:22} {c:16} ran {r} effect {e}" for i, c, r, e in log)
        )
        self.assertFalse(errors, errors)


if __name__ == "__main__":
    unittest.main()

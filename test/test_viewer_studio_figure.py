# OpenSculptBoy
# Apache License, Version 2.0
"""
The figure and the scan in Pose's Edit (docs/viewer.md, "The figure and the scan"), in the built page
with David's starter, in Chromium through Playwright (local only; OPENSCULPTBOY_STARTERS names the
folder of the statue packages). As a person would:

- Place points, then Show the figure off: the scene pass draws none of the figure's triangles, the
  picture above the floor matches the picture with every mesh of the figure switched off, the status
  line says "Figure hidden", and the toggles leave the history and the card alone;
- with the figure hidden, a point placed on the scan is still picked by a click, and a joint and a
  prop under the pointer are not;
- Shift+Z swaps the figure and the scan, and both shown become the scan alone;
- the display menu (Z) stays open after a click on a switch, so two switches flip in one opening,
  with their ticks updated;
- Edit's footer stands below the panel's scrolling body, so the last row of the Props list scrolls
  into view above it.

The rules alone (the swap, never both hidden, the menu's switches and ticks) have node checks in
viewer/test/pose_edit.mjs and viewer/test/shortcuts.mjs.
"""

import base64
import io
import pathlib
import shutil
import tempfile
import unittest

import numpy as np

from test.markers import local_only
from test.test_viewer_starters import (
    DROP,
    open_tab,
    playwright_available,
    real_packages,
    served,
    stop,
)

REPORT = "() => window.poseEditReport()"
MENU = """() => { const m = document.querySelector('.ring-menu'); if (!m || m.hidden) return null;
  const at = (e) => { const r = e.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; };
  return Object.fromEntries([...m.querySelectorAll('.ring-item')].map((e) => [e.dataset.action,
    { checked: e.getAttribute('aria-checked'), disabled: e.getAttribute('aria-disabled'), box: at(e) }])); }"""
PANEL = """() => { const sc = document.querySelector('#insp-left .insp-scroll');
  const foot = document.querySelector('.pe-foot');
  const rows = [...document.querySelectorAll('.pe-prop-row')], last = rows[rows.length - 1];
  last.scrollIntoView({ block: 'nearest' });
  const s = sc.getBoundingClientRect(), f = foot.getBoundingClientRect(), r = last.getBoundingClientRect();
  const at = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
  return { scroll: [s.top, s.bottom], foot: [f.top, f.bottom], row: [r.top, r.bottom], hit: !!at && last.contains(at),
    inFoot: !!at && !!at.closest('.pe-foot'), rows: rows.length }; }"""


def report(tab):
    return tab.evaluate(REPORT)


def settle(tab, frames=4):
    for _ in range(frames):
        tab.evaluate("() => new Promise((r) => requestAnimationFrame(() => r()))")


def press_in_view(tab, key):
    tab.evaluate("() => document.getElementById('c').focus()")
    tab.keyboard.press(key)


@local_only("Playwright with the built viewer page and the statue packages")
@unittest.skipUnless(
    playwright_available() and real_packages(),
    "Playwright, the built page and OPENSCULPTBOY_STARTERS are needed",
)
class TestFigureAndScan(unittest.TestCase):
    def test_figure_hidden_swap_and_menu(self):
        from PIL import Image
        from playwright.sync_api import sync_playwright

        folder = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        servers, url = served(folder, {})
        self.addCleanup(stop, servers)
        david = (real_packages() / "david_goliath.zip").read_bytes()

        def picture(tab):
            # the canvas alone: the interface and the points' layer hidden for the capture
            tab.evaluate("() => document.body.classList.add('shot')")
            settle(tab, 30)
            tab.wait_for_timeout(500)
            png = tab.screenshot()
            tab.evaluate("() => document.body.classList.remove('shot')")
            return np.asarray(
                Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float64
            )

        with sync_playwright() as p:
            browser, tab = open_tab(
                p, url + "?base&noprompt&acc=1&querypool=off", 1600, 1000
            )
            errors = []
            tab.on("pageerror", lambda e: errors.append(str(e)))
            tab.evaluate("() => window.__shell.actions.dismissWelcome()")
            tab.evaluate(DROP, {"david_goliath.zip": base64.b64encode(david).decode()})
            tab.wait_for_function(
                "() => window.wearStats() && window.wearStats().card && window.wearStats().timing.full === true",
                timeout=300000,
            )
            tab.evaluate("() => __shell.actions.openMode('pose')")
            tab.wait_for_function("() => !!window.__poseEdit")
            press_in_view(tab, "Tab")
            tab.wait_for_function(
                "() => { const s = window.poseEditReport().state;"
                " return s.on && s.scan.available && !s.scan.loading; }",
                timeout=120000,
            )
            not_now = tab.get_by_role("button", name="Not now")
            if not_now.count() and not_now.first.is_visible():
                not_now.first.click()
            tab.evaluate("() => window.__poseEdit.placeAll()")
            tab.wait_for_function(
                "() => window.poseEditReport().state.counts.placed > 0", timeout=120000
            )
            settle(tab, 10)
            # with the figure shown, a click on the sword selects it and a click on the left elbow its joint
            sword_at = (
                "() => { const q = window.__poseEdit.host.props()"
                ".find((x) => x.name === 'statue/sword');"
                " const P = q.positions, n = P.length / 3, s = [0, 0, 0];"
                " for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) s[k] += P[i * 3 + k] / n;"
                " return window.__studio.project(s); }"
            )
            elbow_at = (
                "() => window.__studio.project(window.__studio.joint('lowerarm01.L'))"
            )
            sword = tab.evaluate(sword_at)
            tab.mouse.click(sword["x"], sword["y"])
            settle(tab)
            self.assertEqual(report(tab)["state"]["prop"], "statue/sword")
            tab.evaluate("() => window.__poseEdit.selectProp(null)")
            press_in_view(tab, "2")
            settle(tab)
            elbow = tab.evaluate(elbow_at)
            tab.mouse.click(elbow["x"], elbow["y"])
            settle(tab)
            self.assertIsNotNone(report(tab)["state"]["joint"])
            tab.evaluate("() => window.__poseEdit.selectJoint(null)")
            press_in_view(tab, "1")
            settle(tab, 10)
            undo0 = len(report(tab)["undo"])
            both = picture(tab)
            drawn = tab.evaluate("() => window.viewState().figure")
            self.assertFalse(drawn["hidden"])
            self.assertTrue(report(tab)["figureShown"])

            # ---------------------------------------------------------------- the figure hidden
            tab.get_by_role("switch", name="Show the figure").click()
            settle(tab, 10)
            r = report(tab)
            self.assertEqual(
                [r["state"]["figure"], r["state"]["scan"]["shown"]], [False, True]
            )
            self.assertFalse(r["figureShown"])
            self.assertTrue(
                tab.evaluate("() => window.__studio.status()").startswith(
                    "Figure hidden"
                )
            )
            alone = picture(tab)
            hidden = tab.evaluate("() => window.viewState().figure")
            self.assertTrue(hidden["hidden"])
            # the scene pass draws the floor and the background alone
            self.assertLess(
                hidden["triangles"], 0.01 * drawn["triangles"], (hidden, drawn)
            )
            # the same picture as every mesh of the figure switched off, above the floor (the floor's shadow
            # map keeps the figure's shadow there)
            tab.evaluate(
                "() => { window.__poseEdit.setFigure(true); window.__studio.showMeshes(false); }"
            )
            empty = picture(tab)
            tab.evaluate(
                "() => { window.__studio.showMeshes(true); window.__poseEdit.setFigure(false); }"
            )
            settle(tab, 10)
            floor = tab.evaluate("() => window.groundInfo().floor")
            top = (
                int(tab.evaluate("(y) => window.__studio.project([0, y, 0]).y", floor))
                - 40
            )
            self.assertGreater(top, 200)
            diff = np.abs(alone[:top] - empty[:top])
            self.assertLess(diff.mean(), 0.5, "no figure pixels")
            self.assertGreater(
                np.abs(both[:top] - alone[:top]).mean(), 1.0, "the figure showed before"
            )
            # a view setting: no step in the history, nothing in the card
            self.assertEqual(len(report(tab)["undo"]), undo0)
            card = tab.evaluate(
                "() => JSON.stringify(window.__poseEdit.host.card().card)"
            )
            self.assertNotIn('"figure"', card)

            # a point placed on the scan is picked; a joint and a prop under the pointer are not
            r = report(tab)
            name, pt = next(
                (k, v)
                for k, v in r["points"].items()
                if v["target"] is not None
                and v["visible"]
                and v["screen"]
                and 200 < v["screen"]["x"] < 1400
            )
            tab.mouse.click(pt["screen"]["x"], pt["screen"]["y"])
            settle(tab)
            self.assertEqual(report(tab)["state"]["selected"], name)
            press_in_view(tab, "Escape")
            sword = tab.evaluate(sword_at)
            tab.mouse.click(sword["x"], sword["y"])
            settle(tab)
            self.assertIsNone(report(tab)["state"]["prop"])
            press_in_view(tab, "2")
            settle(tab)
            elbow = tab.evaluate(elbow_at)
            tab.mouse.click(elbow["x"], elbow["y"])
            settle(tab)
            self.assertIsNone(report(tab)["state"]["joint"])
            press_in_view(tab, "1")

            # ---------------------------------------------------------------- Shift+Z swaps them
            press_in_view(tab, "Shift+KeyZ")
            settle(tab)
            r = report(tab)
            self.assertEqual(
                [r["state"]["figure"], r["state"]["scan"]["shown"]], [True, False]
            )
            self.assertTrue(
                tab.evaluate("() => window.__studio.status()").startswith("Scan hidden")
            )
            self.assertFalse(tab.evaluate("() => window.viewState().figure.hidden"))
            press_in_view(tab, "Shift+KeyZ")
            settle(tab)
            r = report(tab)
            self.assertEqual(
                [r["state"]["figure"], r["state"]["scan"]["shown"]], [False, True]
            )
            tab.get_by_role("switch", name="Show the figure").click()
            press_in_view(tab, "Shift+KeyZ")
            settle(tab)
            r = report(tab)
            self.assertEqual(
                [r["state"]["figure"], r["state"]["scan"]["shown"]],
                [False, True],
                "both shown become the scan alone",
            )
            tab.evaluate("() => window.__poseEdit.setFigure(true)")

            # ---------------------------------------------------------------- two switches in one opening of Z
            tab.mouse.move(800, 500)
            press_in_view(tab, "KeyZ")
            settle(tab)
            m = tab.evaluate(MENU)
            self.assertEqual(m["pose.figure"]["checked"], "true")
            self.assertEqual(m["display.xray"]["checked"], "false")
            tab.mouse.click(*m["pose.figure"]["box"])
            settle(tab)
            m = tab.evaluate(MENU)
            self.assertIsNotNone(m, "the menu stays open after a switch")
            self.assertEqual(m["pose.figure"]["checked"], "false")
            tab.mouse.click(*m["display.xray"]["box"])
            settle(tab)
            m = tab.evaluate(MENU)
            self.assertIsNotNone(m)
            self.assertEqual(
                [m["pose.figure"]["checked"], m["display.xray"]["checked"]],
                ["false", "true"],
            )
            self.assertTrue(tab.evaluate("() => window.__studio.cam().xray"))
            self.assertFalse(report(tab)["figureShown"])
            tab.keyboard.press("Escape")
            settle(tab)
            self.assertIsNone(tab.evaluate(MENU))
            tab.evaluate("() => { window.__poseEdit.setFigure(true); }")
            press_in_view(tab, "Alt+KeyZ")

            # ---------------------------------------------------------------- the footer below the Props list
            props = tab.locator(".pe-props details")
            if not props.evaluate("(d) => d.open"):
                tab.locator(".pe-props summary").first.click()
            settle(tab)
            panel = tab.evaluate(PANEL)
            self.assertGreaterEqual(panel["rows"], 2)
            self.assertTrue(panel["hit"], panel)
            self.assertFalse(panel["inFoot"], panel)
            self.assertGreaterEqual(panel["foot"][0], panel["scroll"][1] - 1, panel)
            self.assertLessEqual(panel["row"][1], panel["foot"][0] + 1, panel)
            browser.close()
        self.assertFalse(errors, errors)


if __name__ == "__main__":
    unittest.main()

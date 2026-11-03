# OpenSculptBoy
# Apache License, Version 2.0
"""
The pose editor in the built viewer page (DESIGN-0013, phase 1), in Chromium through Playwright
(local only). A synthetic statue (``pose_edit_fixtures.points_statue``: the default body's skin in
another pose as its scan) opens as a dropped card; then, as a person would:

- Tab in Pose turns Edit on; the scan loads and shows;
- Place points puts every point on the scan, and the arms turn towards the scan's pose;
- a click on a point selects it; with the Move tool a drag with the mouse slides it on the scan
  (recorded moves of the pointer), the hand follows, and the drag is one step of undo;
- the wheel during a drag and Page Down push a point into the stone; Esc during a drag puts it
  back; Ctrl+Z and Ctrl+Shift+Z undo and redo;
- Save points writes the points and the pose into the card.

The solve, the placement, Squeeze and the balance mark have node checks against Python
(test_pose_edit_problem, test_pose_edit_place, test_pose_edit_contact).
"""

import base64
import io
import json
import os
import pathlib
import shutil
import tempfile
import unittest
import zipfile

import numpy as np

from test.markers import local_only
from test.test_viewer_starters import DROP, open_tab, playwright_available, served, stop

REPORT = "() => window.poseEditReport()"


def report(tab):
    return tab.evaluate(REPORT)


def press_in_view(tab, key):
    tab.evaluate("() => document.getElementById('c').focus()")
    tab.keyboard.press(key)


def open_statue(tab, folder):
    from test.pose_edit_fixtures import points_statue

    files, rig, rest, truth = points_statue(folder)
    tab.evaluate(
        DROP, {f.name: base64.b64encode(f.read_bytes()).decode() for f in files}
    )
    tab.wait_for_function(
        "() => window.wearStats && window.wearStats() && window.wearStats().card === 'Points statue'",
        timeout=120000,
    )
    tab.evaluate("() => __shell.actions.openMode('pose')")
    tab.wait_for_function("() => !!window.__poseEdit")
    return rig, rest, truth


def edit_on(tab):
    press_in_view(tab, "Tab")
    tab.wait_for_function(
        "() => window.poseEditReport().state.on && window.poseEditReport().state.scan.available",
        timeout=60000,
    )


def settle(tab, frames=4):
    for _ in range(frames):
        tab.evaluate("() => new Promise((r) => requestAnimationFrame(() => r()))")


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestPoseEditPage(unittest.TestCase):
    def test_points_on_a_statue(self):
        from playwright.sync_api import sync_playwright

        folder = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        page_dir = folder / "page"
        page_dir.mkdir()
        servers, url = served(page_dir, {})
        self.addCleanup(stop, servers)
        shots = os.environ.get("OSB_POSE_SHOTS")
        with sync_playwright() as p:
            browser, tab = open_tab(p, url + "?base", 1920, 1080)

            def shot(name):
                if shots:
                    tab.screenshot(path=str(pathlib.Path(shots) / f"{name}.png"))

            rig, rest, truth = open_statue(tab, folder)
            edit_on(tab)
            # the synthetic scan's pose is a little away from Anny's rest pose; the page starts from its own clip
            tab.evaluate("() => window.__poseEdit.restPose()")
            settle(tab)
            r = report(tab)
            self.assertEqual(r["state"]["counts"]["placed"], 0)
            shot("1a_edit_on")

            # Place points: every point on the scan, the arms turned towards the scan's pose
            tab.get_by_role("button", name="Place points").click()
            tab.wait_for_function(
                "() => window.poseEditReport().state.counts.placed > 0"
            )
            settle(tab)
            r = report(tab)
            placed = [v for v in r["points"].values() if v["target"] is not None]
            self.assertGreaterEqual(len(placed), 50)
            for name, v in r["points"].items():
                if v["target"] is not None and v["depth"] == 0:
                    self.assertLess(v["onScan"], 1e-4, name)
            # the right hand's points: Anny's points meet their places within a few millimetres
            gaps = [
                np.linalg.norm(np.subtract(v["anny"], v["target"]))
                for k, v in r["points"].items()
                if k.endswith(".R") and v["on"]
            ]
            self.assertLess(float(np.median(gaps)), 0.006)
            shot("1b_placed")

            # the right hand selected from its cluster (the camera frames it), then point 22 on the hand map
            tab.evaluate("() => window.__poseEdit.selectHand('R')")
            tab.wait_for_timeout(600)
            tab.locator(".hand-map [data-name='index_pip.R']").click()
            settle(tab)
            r = report(tab)
            self.assertEqual(r["state"]["selected"], "index_pip.R")
            shot("1c_hand_selected")

            # a drag with the Move tool: the pointer's recorded moves slide the point on the scan
            press_in_view(tab, "w")
            self.assertEqual(report(tab)["state"]["tool"], "move")
            before = report(tab)["points"]["index_pip.R"]
            x, y = before["screen"]["x"], before["screen"]["y"]
            undo0 = len(report(tab)["undo"])
            tab.mouse.move(x, y)
            tab.mouse.down()
            for k in range(1, 13):
                tab.mouse.move(x + 3 * k, y + 1.5 * k)
                settle(tab, 1)
            shot("1d_dragging")
            tab.mouse.up()
            settle(tab)
            after = report(tab)["points"]["index_pip.R"]
            moved = np.linalg.norm(np.subtract(after["target"], before["target"]))
            self.assertGreater(moved, 0.003)
            self.assertLess(after["onScan"], 1e-4)
            self.assertEqual(after["placed"], "owner")
            self.assertEqual(len(report(tab)["undo"]), undo0 + 1)
            # the finger followed its point
            self.assertLess(
                np.linalg.norm(np.subtract(after["anny"], after["target"])), 0.01
            )

            # Ctrl+Z puts it back; Ctrl+Shift+Z brings the drag again
            press_in_view(tab, "Control+z")
            settle(tab)
            np.testing.assert_allclose(
                report(tab)["points"]["index_pip.R"]["target"],
                before["target"],
                atol=1e-9,
            )
            press_in_view(tab, "Control+Shift+z")
            settle(tab)
            np.testing.assert_allclose(
                report(tab)["points"]["index_pip.R"]["target"],
                after["target"],
                atol=1e-9,
            )

            # Esc during a drag puts the point back where it was
            s = report(tab)["points"]["index_pip.R"]["screen"]
            tab.mouse.move(s["x"], s["y"])
            tab.mouse.down()
            for k in range(1, 6):
                tab.mouse.move(s["x"] - 4 * k, s["y"])
                settle(tab, 1)
            tab.keyboard.press("Escape")
            tab.mouse.up()
            settle(tab)
            np.testing.assert_allclose(
                report(tab)["points"]["index_pip.R"]["target"],
                after["target"],
                atol=1e-9,
            )

            # the wheel during a drag pushes the point 1 mm a notch; Page Down 1 mm more; Page Up pulls it out
            tip = report(tab)["points"]["middle_tip.R"]["screen"]
            tab.locator(".hand-map [data-name='middle_tip.R']").click()
            tab.mouse.move(tip["x"], tip["y"])
            tab.mouse.down()
            tab.mouse.move(tip["x"] + 5, tip["y"])
            settle(tab, 1)
            for _ in range(3):
                tab.mouse.wheel(0, 100)
                settle(tab, 1)
            tab.mouse.up()
            settle(tab)
            self.assertAlmostEqual(
                report(tab)["points"]["middle_tip.R"]["depth"], 3.0, places=6
            )
            press_in_view(tab, "PageDown")
            settle(tab)
            self.assertAlmostEqual(
                report(tab)["points"]["middle_tip.R"]["depth"], 4.0, places=6
            )
            shot("1e_pushed")
            press_in_view(tab, "PageUp")
            self.assertAlmostEqual(
                report(tab)["points"]["middle_tip.R"]["depth"], 3.0, places=6
            )
            press_in_view(tab, "Alt+r")
            self.assertAlmostEqual(
                report(tab)["points"]["middle_tip.R"]["depth"], 0.0, places=6
            )

            # [ and ] walk along the finger; Esc clears the selection
            press_in_view(tab, "[")
            self.assertEqual(report(tab)["state"]["selected"], "middle_dip.R")
            press_in_view(tab, "]")
            self.assertEqual(report(tab)["state"]["selected"], "middle_tip.R")
            press_in_view(tab, "Escape")
            self.assertIsNone(report(tab)["state"]["selected"])

            # contact pins and the balance mark
            tab.evaluate(
                "() => { const e = window.__poseEdit; e.selectHand(null, false); "
                "e.addPin('sole.L', { kind: 'ground' }); e.setBalance(true); }"
            )
            settle(tab, 6)
            r = report(tab)
            self.assertEqual(len(r["state"]["pins"]), 1)
            self.assertIsNotNone(r["state"]["balance"]["inside"])
            shot("1f_pins_balance")

            # Save points: the card with its points, pins and pose
            with tab.expect_download() as dl:
                tab.get_by_role("button", name="Save points").click()
            data = pathlib.Path(dl.value.path()).read_bytes()
            if data[:2] == b"PK":
                card = json.loads(
                    zipfile.ZipFile(io.BytesIO(data)).read("character.json")
                )
            else:
                card = json.loads(data)
            ref = card["detail"]["fit"]["reference"]
            self.assertEqual(ref["points"]["format"], "opensculptboy.card-points/1")
            self.assertGreaterEqual(
                sum(len(v) for v in ref["points"]["chains"].values()), 50
            )
            self.assertEqual(len(ref["pins"]), 1)
            self.assertIn("ghost", ref)
            self.assertTrue(card["pose"]["bones"])

            # Tab turns Edit off
            press_in_view(tab, "Tab")
            self.assertFalse(report(tab)["state"]["on"])
            browser.close()


if __name__ == "__main__":
    unittest.main()

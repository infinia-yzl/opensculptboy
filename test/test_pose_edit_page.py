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


def picture(png):
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float64)


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
            tab.evaluate("() => window.__poseEdit.placeAll()")
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

            # a drag that leaves the scan: the point keeps its last spot on the surface, and the magnifier holds
            # on that spot (it showed the empty stage round the pointer before)
            self.assertTrue(report(tab)["unsaved"])
            s = report(tab)["points"]["index_pip.R"]["screen"]
            # a place on the view where the pointer's ray meets no scan
            ox, oy = tab.evaluate(OFF_SCAN)
            tab.mouse.move(s["x"], s["y"])
            tab.mouse.down()
            for k in range(1, 9):
                tab.mouse.move(
                    s["x"] + (ox - s["x"]) * k / 8, s["y"] + (oy - s["y"]) * k / 8
                )
                settle(tab, 1)
            r = report(tab)
            mag, spot = r["magnifier"], r["points"]["index_pip.R"]["screen"]
            self.assertTrue(mag["off"])
            self.assertLess(np.hypot(mag["x"] - spot["x"], mag["y"] - spot["y"]), 2.0)
            self.assertLess(r["points"]["index_pip.R"]["onScan"], 1e-4)
            # the point stayed where the pointer left the scan, near where the drag began
            self.assertLess(
                np.linalg.norm(
                    np.subtract(r["points"]["index_pip.R"]["target"], after["target"])
                ),
                0.03,
            )
            shot("1d_off_scan")
            tab.keyboard.press("Escape")
            tab.mouse.up()
            settle(tab)
            np.testing.assert_allclose(
                report(tab)["points"]["index_pip.R"]["target"],
                after["target"],
                atol=1e-9,
            )

            # the wheel during a drag pushes the point 1 mm a notch; Page Down 1 mm more; Page Up pulls it out
            # the selection turns the view round the point, so its place on the screen is read after it
            tab.locator(".hand-map [data-name='middle_tip.R']").click()
            tab.wait_for_timeout(400)
            settle(tab)
            tip = report(tab)["points"]["middle_tip.R"]["screen"]
            tab.mouse.move(tip["x"], tip["y"])
            tab.mouse.down()
            tab.mouse.move(tip["x"] + 5, tip["y"])
            settle(tab, 1)
            for _ in range(3):
                tab.mouse.wheel(0, 100)
                settle(tab, 1)
            dragged = report(tab)["state"]["selected"]
            tab.mouse.up()
            settle(tab)
            self.assertEqual(dragged, "middle_tip.R")
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
            # saved: nothing left to ask about
            self.assertFalse(report(tab)["unsaved"])

            # Tab turns Edit off
            press_in_view(tab, "Tab")
            self.assertFalse(report(tab)["state"]["on"])
            browser.close()

    def test_every_key(self):
        """Every key of the pose editor in its scope, pressed in the view: each runs its action
        (the registry's record of the actions that ran), and the studio's modal grab and turn
        (G and R with an axis key, a typed value, Enter and Esc) move and put back."""
        from playwright.sync_api import sync_playwright

        folder = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        page_dir = folder / "page"
        page_dir.mkdir()
        servers, url = served(page_dir, {})
        self.addCleanup(stop, servers)
        with sync_playwright() as p:
            browser, tab = open_tab(p, url + "?base", 1920, 1080)
            open_statue(tab, folder)
            edit_on(tab)
            tab.evaluate("() => window.__poseEdit.restPose()")
            tab.evaluate("() => window.__poseEdit.placeAll()")
            tab.wait_for_function(
                "() => window.poseEditReport().state.counts.placed > 0"
            )
            settle(tab)
            pressed = set()

            def key(chord, action, setup=None):
                if setup:
                    tab.evaluate(setup)
                    settle(tab, 2)
                tab.evaluate("() => { window.__studio.ran.length = 0; }")
                press_in_view(tab, chord)
                settle(tab, 2)
                ran = tab.evaluate("() => window.__studio.ran.slice()")
                self.assertIn(action, ran, f"{chord} in the view runs {action}")
                pressed.add(action)

            select = "() => window.__poseEdit.select('index_pip.R')"
            target = lambda: report(tab)["points"]["index_pip.R"]["target"]  # noqa: E731

            # the tools and the sections
            for chord, action in [
                ("q", "pose.tool.select"),
                ("w", "pose.tool.move"),
                ("e", "pose.tool.turn"),
                ("1", "pose.tool.1"),
            ]:
                key(chord, action)
            self.assertEqual(report(tab)["state"]["section"], "points")

            # G: the modal grab; Esc puts the point back
            t0 = target()
            key("g", "pose.grab", select)
            box = tab.locator("#c").bounding_box()
            cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            for k in range(1, 8):
                tab.mouse.move(cx + 4 * k, cy + 2 * k)
                settle(tab, 1)
            key("Escape", "modal.cancel")
            np.testing.assert_allclose(target(), t0, atol=1e-9)
            # G, X, 5, Enter: 5 mm along the page's X axis, one step of undo
            undo0 = len(report(tab)["undo"])
            key("g", "pose.grab")
            key("x", "modal.x")
            key("5", "modal.type")
            key("Enter", "modal.confirm")
            moved = float(np.linalg.norm(np.subtract(target(), t0)))
            self.assertAlmostEqual(moved, 0.005, delta=2e-4)
            self.assertEqual(len(report(tab)["undo"]), undo0 + 1)

            # the selection's keys
            key("[", "pose.parent", select)
            key("]", "pose.child")
            key("Shift+BracketLeft", "pose.finger-prev")
            key("Shift+BracketRight", "pose.finger-next")
            key("PageDown", "pose.push", select)
            key("PageDown", "pose.push")
            key("PageUp", "pose.pull")
            key("Alt+r", "pose.reset-turn")
            key("ArrowLeft", "pose.nudge")
            key("Alt+g", "pose.reset-point")
            key("Escape", "pose.clear")
            self.assertIsNone(report(tab)["state"]["selected"])
            key("a", "pose.select-all", "() => window.__poseEdit.selectHand('R')")
            key("Alt+a", "pose.select-none")
            # Shift+Z swaps the figure and the scan: both shown become the scan alone, then the figure alone
            key("Shift+z", "pose.swap")
            self.assertEqual(
                [report(tab)["state"]["figure"], report(tab)["state"]["scan"]["shown"]],
                [False, True],
            )
            key("Shift+z", "pose.swap")
            self.assertEqual(
                [report(tab)["state"]["figure"], report(tab)["state"]["scan"]["shown"]],
                [True, False],
            )
            tab.evaluate("() => window.__poseEdit.setScanShown(true)")
            press_in_view(tab, "Escape")

            # the pose's keys: mirror, copy and paste, undo and redo
            key("m", "pose.mirror")
            key("Control+c", "pose.copy", "() => window.__poseEdit.selectHand('R')")
            key("Control+v", "pose.paste")
            key("Control+Shift+v", "pose.paste-mirrored")
            key("Control+z", "pose.undo")
            key("Control+Shift+z", "pose.redo")
            key("Control+z", "pose.undo")
            key("Control+y", "pose.redo")

            # Joints: R, the modal turn; X, 30, Enter bends the elbow 30 degrees; Esc puts a turn back
            key("2", "pose.tool.2")
            self.assertEqual(report(tab)["state"]["section"], "joints")
            tab.evaluate("() => window.__poseEdit.selectJoint('lowerarm01.R')")
            settle(tab, 2)
            b0 = report(tab)["state"]["jv"]["value"][0]
            key("r", "pose.turn-keys")
            key("x", "modal.x")
            key("3", "modal.type")
            key("0", "modal.type")
            key("Enter", "modal.confirm")
            b1 = report(tab)["state"]["jv"]["value"][0]
            self.assertAlmostEqual(b1 - b0, 30, delta=0.5)
            key("r", "pose.turn-keys")
            for k in range(1, 8):
                tab.mouse.move(cx + 6 * k, cy - 5 * k)
                settle(tab, 1)
            key("Escape", "modal.cancel")
            self.assertAlmostEqual(
                report(tab)["state"]["jv"]["value"][0], b1, delta=1e-6
            )
            key("Alt+r", "pose.reset-joint")
            self.assertAlmostEqual(report(tab)["state"]["jv"]["value"][0], 0, delta=0.5)
            key("Escape", "pose.clear-joint")
            self.assertIsNone(report(tab)["state"]["joint"])

            for chord, action in [
                ("3", "pose.tool.3"),
                ("4", "pose.tool.4"),
                ("5", "pose.tool.5"),
                ("Tab", "pose.edit"),
            ]:
                key(chord, action)
            self.assertFalse(report(tab)["state"]["on"])

            # every registered key of the pose editor was pressed; the props' keys need a prop, and
            # test_viewer_props_page presses them
            actions = tab.evaluate("() => window.__studio.actions()")
            want = {
                a["id"]
                for a in actions
                if a["id"].startswith("pose.")
                and not a["id"].startswith("pose.prop.")
                and a["keys"]
            }
            self.assertEqual(want - pressed, set())
            browser.close()

    def test_saved_hand_on_another_character(self):
        """A hand pose saved in the library applies to another character: the statue's right hand
        made a fist and saved, then Anny's default body opened and the pose applied to its right
        hand alone (DESIGN-0013, P4)."""
        from playwright.sync_api import sync_playwright

        folder = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        page_dir = folder / "page"
        page_dir.mkdir()
        servers, url = served(page_dir, {})
        self.addCleanup(stop, servers)
        hand = lambda r, side: {  # noqa: E731
            name: rot
            for name, rot in zip(r["labels"], r["rotations"])
            if name.endswith("." + side) and name.startswith(("finger", "metacarpal"))
        }
        with sync_playwright() as p:
            # the page leaves with an unsaved edit: noprompt keeps the browser from asking
            browser, tab = open_tab(p, url + "?base&noprompt", 1920, 1080)
            open_statue(tab, folder)
            edit_on(tab)
            tab.evaluate(
                "() => { const e = window.__poseEdit; e.restPose(); e.setSection('hands'); "
                "e.setHandSide('R'); e.applyPreset('fist'); e.savePose('Fist hand'); }"
            )
            settle(tab)
            fist = hand(report(tab), "R")
            self.assertTrue(any(abs(v[0] - 1) > 0.05 for v in fist.values()))

            # another character: Anny's default body, the library kept in the browser
            tab.goto(url + "?base&noprompt")
            tab.wait_for_function("() => window.__ready && !!window.__shell")
            tab.evaluate("() => __shell.actions.openMode('pose')")
            tab.wait_for_function("() => !!window.__poseEdit")
            press_in_view(tab, "Tab")
            tab.wait_for_function("() => window.poseEditReport().state.on")
            tab.evaluate("() => window.__poseEdit.restPose()")
            settle(tab)
            left0 = hand(report(tab), "L")
            poses = report(tab)["state"]["poses"]
            saved = [q for q in poses if q["label"] == "Fist hand"]
            self.assertEqual(len(saved), 1)
            tab.evaluate(
                "(id) => window.__poseEdit.applyPose(id, 'hand.R')", saved[0]["id"]
            )
            settle(tab)
            r = report(tab)
            for name, rot in hand(r, "R").items():
                np.testing.assert_allclose(rot, fist[name], atol=1e-5, err_msg=name)
            for name, rot in hand(r, "L").items():
                np.testing.assert_allclose(rot, left0[name], atol=1e-9, err_msg=name)
            browser.close()

    def test_grip_library_and_unsaved_edits(self):
        """Grip from the hand as it stands on rods across the right hand (a rod on the root and one
        carried by the hand), Lift out, the scan's picture gone at once when it is hidden, the
        built-in poses in the Library tool applied to one part and blended, and the prompt before
        leaving with unsaved edits."""
        from playwright.sync_api import sync_playwright

        folder = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        page_dir = folder / "page"
        page_dir.mkdir()
        servers, url = served(page_dir, {})
        self.addCleanup(stop, servers)
        with sync_playwright() as p:
            browser, tab = open_tab(p, url + "?base", 1920, 1080)
            open_statue(tab, folder)
            edit_on(tab)
            tab.evaluate("() => window.__poseEdit.restPose()")
            settle(tab)

            # Grip: a rod on the palm's side of the fingers in the reference pose; each finger closes to 1 mm
            tab.evaluate(ROD, {"out": 0.035, "radius": 0.012, "bone": "root"})
            tab.evaluate(
                "() => { const e = window.__poseEdit; e.setSection('hands'); e.setHandSide('R'); "
                "e.setGrip({ hand: 'R', prop: 'test/rod' }); e.setSqueeze(1); }"
            )
            settle(tab)
            st = report(tab)["state"]
            # the straight thumb may lie in the rod beside the palm; the four fingers grip it
            self.assertFalse(st["grip_carried"])
            fingers = {f["finger"]: f for f in st["grip_report"]}
            for f in ("index", "middle", "ring"):
                self.assertGreaterEqual(fingers[f]["clearance"], 1.0, f)
                self.assertLess(fingers[f]["clearance"], 1.5, f)
            for f in ("index", "middle", "ring", "little"):
                self.assertFalse(fingers[f]["blocked"] or fingers[f]["inside"], f)
            # Grip again from that grip: the fingers stand at contact, so each opens a little and closes again
            tab.evaluate(GRIP)
            settle(tab)
            again = {f["finger"]: f for f in report(tab)["state"]["grip_report"]}
            for f in ("index", "middle", "ring"):
                self.assertAlmostEqual(
                    again[f]["clearance"], fingers[f]["clearance"], delta=0.5
                )

            # a rod in the plane of the straight fingers: inside at every curl, so the grip blocks; Lift out moves
            # the hand off the rod (the rod stands on the root) and the grip closes from there
            tab.evaluate("() => window.__poseEdit.restPose()")
            tab.evaluate(ROD, {"out": None, "radius": 0.012, "bone": "root"})
            tab.evaluate(GRIP)
            settle(tab)
            st = report(tab)["state"]
            self.assertTrue(st["grip_blocked"])
            self.assertTrue(any(f["blocked"] for f in st["grip_report"]))
            self.assertTrue(tab.evaluate("() => window.__poseEdit.liftHand()"))
            settle(tab)
            r = report(tab)
            lifted = {f["finger"]: f for f in r["state"]["grip_report"]}
            for f in ("index", "middle", "ring", "little"):
                self.assertFalse(lifted[f]["blocked"] or lifted[f]["inside"], f)
            self.assertLessEqual(r["lift"]["trace"][-1]["mm"], 60)
            self.assertEqual(r["undo"][-1], "Lift the hand out")
            # the same rod carried by the hand: lifting the hand takes it along, so Lift out leaves it
            tab.evaluate("() => window.__poseEdit.restPose()")
            tab.evaluate(ROD, {"out": None, "radius": 0.012, "bone": "metacarpal4.R"})
            tab.evaluate(GRIP)
            settle(tab)
            self.assertTrue(report(tab)["state"]["grip_carried"])
            self.assertFalse(tab.evaluate("() => window.__poseEdit.liftHand()"))
            tab.evaluate("() => window.__poseEdit.setGrip(null)")

            # Show the scan off after the solid scan: the picture leaves the scan at once
            def view():
                settle(tab, 30)
                tab.wait_for_timeout(800)
                return picture(
                    tab.screenshot(clip=dict(x=420, y=80, width=1400, height=900))
                )

            tab.evaluate("() => window.__poseEdit.setScanLook({ shown: false })")
            bare = view()
            tab.evaluate(
                "() => window.__poseEdit.setScanLook({ look: 'solid', shown: true })"
            )
            solid = view()
            tab.evaluate("() => window.__poseEdit.setScanLook({ shown: false })")
            hidden = view()
            shown_diff = np.abs(solid - bare).mean()
            self.assertGreater(shown_diff, 1.0)
            self.assertLess(np.abs(hidden - bare).mean(), 0.25 * shown_diff)
            tab.evaluate(
                "() => window.__poseEdit.setScanLook({ look: 'ghost', shown: true })"
            )

            # the Library tool lists the page's built-in poses in groups, with the search
            tab.evaluate(
                "() => { const e = window.__poseEdit; e.restPose(); e.setSection('library'); }"
            )
            settle(tab)
            builtin = tab.evaluate("() => window.__poseEdit.builtinPoses()")
            self.assertGreaterEqual(len(builtin), 50)
            self.assertEqual(
                tab.locator(".pe-builtin [data-pose]").count(), len(builtin)
            )
            tab.get_by_placeholder(f"Search {len(builtin)} poses").fill(
                builtin[3]["label"][:6]
            )
            self.assertGreaterEqual(tab.locator(".pe-builtin [data-pose]").count(), 1)
            self.assertLess(
                tab.locator(".pe-builtin [data-pose]").count(), len(builtin)
            )
            tab.get_by_placeholder(f"Search {len(builtin)} poses").fill("")
            # one part: a built-in pose on the right hand leaves the rest of the figure
            r0 = report(tab)
            rot0 = dict(zip(r0["labels"], r0["rotations"]))
            tab.evaluate("() => window.__poseEdit.setApplyTo('hand.R')")
            moving = None
            for c in builtin:
                # a chip in a closed fold: the fold opened and the chip clicked
                tab.evaluate(CHIP, c["name"])
                settle(tab)
                r = report(tab)
                rot = dict(zip(r["labels"], r["rotations"]))
                changed = [
                    n for n in rot if np.abs(np.subtract(rot[n], rot0[n])).max() > 1e-6
                ]
                if changed:
                    moving = c
                    break
            self.assertIsNotNone(moving)
            self.assertTrue(
                all(
                    n.endswith(".R") and n.startswith(("finger", "metacarpal"))
                    for n in changed
                ),
                changed,
            )
            self.assertTrue(r["undo"][-1].startswith("Apply "))
            # a blend of two built-in poses on the whole body, half way
            tab.evaluate("() => window.__poseEdit.setApplyTo('all')")
            ids = ["builtin:" + builtin[0]["name"], "builtin:" + builtin[1]["name"]]
            tab.evaluate(
                "(ids) => window.__poseEdit.blendPoses(ids[0], ids[1], 0.5)", ids
            )
            settle(tab)
            self.assertEqual(report(tab)["state"]["blend"]["t"], 0.5)

            # unsaved edits: the browser asks before the page goes, unless the setting is off
            self.assertTrue(report(tab)["unsaved"])
            asked = []

            def on_dialog(d):
                asked.append(d.type)
                d.dismiss()

            tab.on("dialog", on_dialog)
            press_in_view(tab, "Escape")
            try:
                tab.reload(timeout=4000)
            except Exception:
                pass  # the dismissed prompt keeps the page
            self.assertEqual(asked, ["beforeunload"])
            self.assertTrue(report(tab)["unsaved"])
            self.assertTrue(tab.evaluate(UNLOAD))
            # View › Controls turns the prompt off
            tab.evaluate("() => window.__studioSettings.set({ warnUnsaved: false })")
            self.assertFalse(tab.evaluate(UNLOAD))
            tab.evaluate("() => window.__studioSettings.set({ warnUnsaved: true })")
            # undone back to the state as opened: no edit left
            while report(tab)["undo"]:
                tab.evaluate("() => window.__poseEdit.undo()")
            self.assertFalse(report(tab)["unsaved"])
            self.assertFalse(tab.evaluate(UNLOAD))
            browser.close()

            # noprompt in the URL: the page tests leave with edits
            browser, tab = open_tab(p, url + "?base&noprompt", 1920, 1080)
            open_statue(tab, folder)
            edit_on(tab)
            tab.evaluate("() => window.__poseEdit.restPose()")
            settle(tab)
            tab.evaluate(
                "() => { const e = window.__poseEdit; e.setSection('hands'); "
                "e.setHandSide('R'); e.applyPreset('fist'); }"
            )
            self.assertTrue(report(tab)["unsaved"])
            self.assertFalse(report(tab)["prompt"])
            self.assertFalse(tab.evaluate(UNLOAD))
            browser.close()


# a rod across the right hand's fingers for Grip (card frame), 14 cm long, as a closed capsule; the editor's host
# shows it as the prop test/rod
ROD = """(o) => {
  const e = window.__poseEdit, F = e.frames, head = (l) => e.headOf(l);
  const w = head('wrist.R'), m = head('finger3-1.R');
  const i = head('finger2-1.R'), l = head('finger5-1.R');
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const unit = (v) => { const n = Math.hypot(...v); return v.map((x) => x / n); };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]];
  const along = unit(sub(m, w)), across = unit(sub(i, l));
  let n = unit(cross(along, across));
  // the palm's side: where a curled finger goes
  const S = e.session, R0 = S.rotations.map((r) => r.slice());
  const tip0 = head('finger3-3.R');
  e.hands().setCurl(S.rotations, 'R', 2, 1); S.setPose(S.rotations);
  const tip1 = head('finger3-3.R'); S.setPose(R0);
  const d = sub(tip1, tip0);
  if (d[0] * n[0] + d[1] * n[1] + d[2] * n[2] < 0) n = n.map((x) => -x);
  // across the middle joints (the reference hand's fingers curl a little); `out` metres past the joint centres of
  // the four fingers on the palm's side, or through the middle finger's middle joint when out is null
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const k0 = [0, 1, 2].map((k) => (i[k] + l[k]) / 2), pip = head('finger3-2.R');
  const base = [0, 1, 2].map((k) => k0[k] + along[k] * dot(sub(pip, k0), along));
  let off = dot(sub(pip, base), n);
  if (o.out !== null) {
    off = -Infinity;
    for (const f of [2, 3, 4, 5]) {
      for (const j of [1, 2, 3]) off = Math.max(off, dot(sub(head(`finger${f}-${j}.R`), base), n));
    }
    off += o.out;
  }
  const c = [0, 1, 2].map((k) => base[k] + n[k] * off);
  const b = cross(across, n), r = o.radius, half = 0.07, ring = 16, P = [], I = [];
  const at = (t, a, s) => [0, 1, 2].map((k) => c[k] + across[k] * t
    + (n[k] * Math.cos(a) + b[k] * Math.sin(a)) * s);
  const cap = [3, 2, 1].map((j) => [Math.cos(j * Math.PI / 8), Math.sin(j * Math.PI / 8)]);
  const rows = [[-half - r, 0], ...cap.reverse().map(([x, y]) => [-half - r * x, r * y]),
    [-half, r], [half, r], ...cap.reverse().map(([x, y]) => [half + r * x, r * y]),
    [half + r, 0]];
  for (const [t, s] of rows) {
    for (let a = 0; a < ring; a++) P.push(...F.toWorld(at(t, 2 * Math.PI * a / ring, s)));
  }
  for (let q = 0; q + 1 < rows.length; q++) {
    for (let a = 0; a < ring; a++) {
      const a1 = (a + 1) % ring, p0 = q * ring + a, p1 = q * ring + a1;
      const p2 = (q + 1) * ring + a, p3 = (q + 1) * ring + a1;
      I.push(p0, p2, p1, p1, p2, p3);
    }
  }
  const positions = Float32Array.from(P), indices = Uint32Array.from(I);
  e.host.props = () => [{ name: 'test/rod', positions, indices, bone: o.bone }];
}"""
GRIP = (
    "() => { const e = window.__poseEdit; "
    "e.setGrip({ hand: 'R', prop: 'test/rod' }); e.setSqueeze(1); }"
)
# the first spot of a grid over the view where the pointer's ray meets no scan (client pixels)
OFF_SCAN = """() => {
  const e = window.__poseEdit, r = e.host.canvas.getBoundingClientRect();
  for (let j = 1; j < 10; j++) for (let i = 1; i < 10; i++) {
    const x = r.left + r.width * i / 10, y = r.top + r.height * j / 10;
    if (x > 400 && !e.scanHit(x, y, null)) return [x, y];
  }
  return [r.left + 420, r.top + 20];
}"""
CHIP = (
    "(name) => { const c = document.querySelector(`.pe-builtin [data-pose='${name}']`); "
    "c.closest('details').open = true; c.click(); }"
)
UNLOAD = (
    "() => { const e = new Event('beforeunload', { cancelable: true }); "
    "dispatchEvent(e); return e.defaultPrevented; }"
)


if __name__ == "__main__":
    unittest.main()

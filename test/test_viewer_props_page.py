# OpenSculptBoy
# Apache License, Version 2.0
"""
Props moved by hand in the studio, in the built viewer page (local only: Playwright and Chromium).
A synthetic statue (``pose_edit_fixtures.points_statue``) opens as a dropped card with one prop, a
rod across the right palm bound rigidly to ``metacarpal4.R`` (a package written here); then, as a
person would:

- with Edit on, a click on the rod selects it, and the panel's list shows it held by the hand;
- the Move tool's drag with recorded moves of the mouse moves the rod across the view, and the
  drag is one step of undo; Ctrl+Z puts it back and Ctrl+Shift+Z moves it again;
- G, then X and a typed 5, then Enter moves it 5 mm along the view's X axis;
- Held by gives it to the ground (the root) where it stands, and a turn of the right wrist then
  leaves it in place, where before it followed the hand;
- Save points writes the card with the rod's place (card schema 6), and the saved card opens again
  with the rod where it was.
"""

import base64
import io
import json
import pathlib
import shutil
import tempfile
import unittest
import zipfile

import numpy as np

from test.markers import local_only
from test.test_pose_edit_page import edit_on, press_in_view, report, settle
from test.test_viewer_starters import DROP, open_tab, playwright_available, served, stop

REF = "statue/rod"
# the rod's posed vertices in the world, from the page's own props
ROD_WORLD = """() => { const p = window.__poseEdit.host.props().find((q) => q.name === 'statue/rod');
  return p ? Array.from(p.positions) : null; }"""
# a pixel of the view where the pointer's ray meets the rod first (no skin in front)
ROD_PIXEL = f"() => window.__poseEdit.propPixel('{REF}')"


def rod_package(folder: pathlib.Path):
    """A rod of 24 cm, 12 mm across, through the right palm of the default body, bound rigidly to
    the bone of the ring finger's palm (as a statue's sword is), as a package of the slot prop."""
    from opensculptboy.wearables import (
        Garment,
        Material,
        Wearable,
        bind_rigid,
        write_wearable,
    )
    from test.wearable_fixtures import LICENCE, template_body

    body = template_body()
    head = lambda b: body.bone_poses[body.bone_labels.index(b)][:3, 3]  # noqa: E731
    centre = 0.5 * (head("metacarpal4.R") + head("finger3-1.R"))
    across = head("finger2-1.R") - head("finger5-1.R")
    across /= np.linalg.norm(across)
    ref = np.array([0.0, 0.0, 1.0])
    u = np.cross(across, ref)
    u /= np.linalg.norm(u)
    w = np.cross(across, u)
    ring, rows, r = 16, np.linspace(-0.12, 0.12, 9), 0.006
    P = [
        centre + across * t + r * (np.cos(a) * u + np.sin(a) * w)
        for t in rows
        for a in np.linspace(0, 2 * np.pi, ring, endpoint=False)
    ]
    P += [centre - across * 0.12, centre + across * 0.12]
    T = []
    for q in range(len(rows) - 1):
        for a in range(ring):
            a1 = (a + 1) % ring
            p0, p1, p2, p3 = (
                q * ring + a,
                q * ring + a1,
                (q + 1) * ring + a,
                (q + 1) * ring + a1,
            )
            T += [(p0, p2, p1), (p1, p2, p3)]
    end0, end1 = len(P) - 2, len(P) - 1
    last = (len(rows) - 1) * ring
    for a in range(ring):
        a1 = (a + 1) % ring
        T += [(end0, a, a1), (end1, last + a1, last + a)]
    V = np.asarray(P)
    N = V - centre - np.outer((V - centre) @ across, across)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    garment = Garment(
        V,
        np.asarray(T, np.int64),
        normals=N,
        materials=[Material("stone", (0.8, 0.78, 0.74, 1.0), 0.0, 0.6)],
    )
    rod = Wearable(
        name="rod",
        label="Rod",
        slot="prop",
        layer=1,
        garment=garment,
        binding=bind_rigid(garment, body, "metacarpal4.R"),
        weights="rigid",
        licence=dict(LICENCE),
        covers=np.zeros(len(body.vertices), bool),
    )
    path = write_wearable(folder / "rod.wearable.json", rod)
    return sorted(p for p in folder.glob("rod.*") if p.is_file()), path


def open_prop_statue(tab, folder):
    from test.pose_edit_fixtures import points_statue

    files, *_ = points_statue(folder)
    rod_files, _ = rod_package(folder)
    card = json.loads((folder / "character.json").read_text(encoding="utf-8"))
    card["outfits"] = [
        {
            "id": "statue",
            "label": "Statue pieces",
            "pieces": [{"piece": "rod", "file": "rod.wearable.json", "enabled": True}],
        }
    ]
    card["active_outfit"] = "statue"
    (folder / "character.json").write_text(json.dumps(card), encoding="utf-8")
    drop(tab, files + rod_files)


def drop(tab, files, entries=None):
    payload = {f.name: base64.b64encode(f.read_bytes()).decode() for f in files}
    for name, data in (entries or {}).items():
        payload[name] = base64.b64encode(data).decode()
    tab.evaluate(DROP, payload)
    tab.wait_for_function(
        "() => window.wearStats && window.wearStats() && "
        "window.wearStats().card === 'Points statue' && window.wearStats().timing.full === true",
        timeout=120000,
    )
    tab.evaluate("() => __shell.actions.openMode('pose')")
    tab.wait_for_function("() => !!window.__poseEdit")


def world(tab):
    return np.asarray(tab.evaluate(ROD_WORLD), np.float64).reshape(-1, 3)


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestPropsPage(unittest.TestCase):
    def test_select_drag_undo_held_by_save_and_reload(self):
        from playwright.sync_api import sync_playwright

        folder = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        page_dir = folder / "page"
        page_dir.mkdir()
        servers, url = served(page_dir, {})
        self.addCleanup(stop, servers)
        with sync_playwright() as p:
            browser, tab = open_tab(p, url + "?base&noprompt", 1920, 1080)
            errors = []
            tab.on("pageerror", lambda e: errors.append(str(e)))
            open_prop_statue(tab, folder)
            edit_on(tab)
            tab.evaluate("() => window.__poseEdit.restPose()")
            settle(tab)
            st = report(tab)["state"]
            self.assertEqual([q["ref"] for q in st["propList"]], [REF])
            self.assertEqual(st["propList"][0]["carrier"], "metacarpal4.R")
            # frame the rod, then a click on it selects it (Select tool)
            tab.evaluate(f"() => window.__poseEdit.selectProp('{REF}', true)")
            tab.wait_for_timeout(700)
            tab.evaluate("() => window.__poseEdit.selectProp(null)")
            press_in_view(tab, "q")
            settle(tab)
            at = tab.evaluate(ROD_PIXEL)
            self.assertIsNotNone(at, "a pixel on the rod")
            tab.mouse.move(at[0], at[1])
            tab.mouse.down()
            tab.mouse.up()
            settle(tab)
            self.assertEqual(report(tab)["state"]["prop"], REF)
            self.assertTrue(
                tab.locator(".pe-prop-row[aria-pressed='true']").is_visible()
            )

            # the Move tool: a drag with recorded moves of the mouse, 120 px to the right in 12 steps. The page
            # grounds the figure again when a prop's place changes (the floor takes the props), and Edit's pose has not
            # been grounded yet: a place set to where the rod stands grounds it first
            tab.evaluate(f"() => window.__poseEdit.host.setPropPlace('{REF}', null)")
            settle(tab)
            before = world(tab)
            undo0 = len(report(tab)["undo"])
            press_in_view(tab, "w")
            settle(tab)
            at = tab.evaluate(ROD_PIXEL)
            tab.mouse.move(at[0], at[1])
            tab.mouse.down()
            for k in range(1, 13):
                tab.mouse.move(at[0] + 10 * k, at[1], steps=2)
                settle(tab, 1)
            tab.mouse.up()
            settle(tab)
            r = report(tab)
            moved = world(tab)
            shift = (moved - before).mean(axis=0)
            self.assertGreater(np.linalg.norm(shift), 0.01)
            # the rod moved as one: every vertex by the same shift, across the view
            np.testing.assert_allclose(
                moved - before, np.broadcast_to(shift, moved.shape), atol=1e-5
            )
            self.assertIsNotNone(r["props"]["place"])
            self.assertEqual(r["props"]["place"]["bone"], "metacarpal4.R")
            self.assertEqual(len(r["undo"]), undo0 + 1)
            self.assertEqual(r["undo"][-1], "Move the Rod")
            # undo and redo
            press_in_view(tab, "Control+z")
            settle(tab)
            np.testing.assert_allclose(world(tab), before, atol=1e-6)
            self.assertIsNone(report(tab)["props"]["place"])
            press_in_view(tab, "Control+Shift+z")
            settle(tab)
            np.testing.assert_allclose(world(tab), moved, atol=1e-6)

            # G, X, a typed 5, Enter: 5 mm along the view's X axis
            before = world(tab)
            tab.mouse.move(960, 540)
            press_in_view(tab, "g")
            for k in ("x", "5", "Enter"):
                tab.keyboard.press(k)
            settle(tab)
            unit = tab.evaluate("() => window.__poseEdit.host.frame().scale")
            np.testing.assert_allclose(
                (world(tab) - before).mean(axis=0) / unit, [0.005, 0, 0], atol=1e-6
            )
            self.assertEqual(report(tab)["undo"][-1], "Move the Rod")

            # Held by: the ground, where the rod stands; a turn of the right wrist then leaves it in place
            before = world(tab)
            tab.locator(".pe-held-by select").select_option("root")
            settle(tab)
            r = report(tab)
            self.assertEqual(r["props"]["carrier"], "root")
            np.testing.assert_allclose(world(tab), before, atol=2e-5)
            TURN_WRIST = """() => { const e = window.__poseEdit, J = e.joints(), b = J.bone('wrist.R');
              e.selectJoint('wrist.R'); e.setJointValue(0, J.values(b, e.session.rotations).value[0] + 25); }"""
            tab.evaluate(TURN_WRIST)
            settle(tab)
            np.testing.assert_allclose(world(tab), before, atol=2e-5)
            press_in_view(tab, "Control+z")
            press_in_view(tab, "Control+z")
            settle(tab)
            self.assertEqual(report(tab)["props"]["carrier"], "metacarpal4.R")
            tab.evaluate(TURN_WRIST)
            settle(tab)
            self.assertGreater(np.abs(world(tab) - before).max(), 0.005)
            press_in_view(tab, "Control+z")
            settle(tab)
            tab.evaluate(f"() => window.__poseEdit.selectProp('{REF}')")
            tab.locator(".pe-held-by select").select_option("root")
            settle(tab)
            held = world(tab)
            place = report(tab)["props"]["place"]

            # Save points: the card with the rod's place, schema 6
            with tab.expect_download() as dl:
                tab.get_by_role("button", name="Save points").click()
            data = pathlib.Path(dl.value.path()).read_bytes()
            saved = (
                json.loads(zipfile.ZipFile(io.BytesIO(data)).read("character.json"))
                if data[:2] == b"PK"
                else json.loads(data)
            )
            self.assertEqual(saved["schema"], 6)
            piece = saved["outfits"][0]["pieces"][0]
            self.assertEqual(piece["place"]["bone"], "root")
            np.testing.assert_allclose(
                piece["place"]["translation"], place["translation"], atol=1e-6
            )
            from opensculptboy.character import Character

            Character.from_dict(
                {k: v for k, v in saved.items() if k not in ("detail",)}
            )

            # the saved card opens again with the rod where it was
            tab.reload()
            tab.wait_for_function("() => window.__ready === true", timeout=300000)
            files = [
                f
                for f in folder.iterdir()
                if f.is_file() and f.name != "character.json"
            ]
            drop(tab, files, {"character.json": json.dumps(saved).encode()})
            edit_on(tab)
            tab.evaluate("() => window.__poseEdit.restPose()")
            settle(tab)
            r = report(tab)
            self.assertEqual(r["state"]["propList"][0]["carrier"], "root")
            np.testing.assert_allclose(world(tab), held, atol=2e-5)
            self.assertEqual(errors, [])
            browser.close()


if __name__ == "__main__":
    unittest.main()

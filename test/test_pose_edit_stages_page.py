# OpenSculptBoy
# Apache License, Version 2.0
"""
The stages of Pose › Edit › Points in the built viewer page, in Chromium through Playwright
(local only). The default character opens; Edit turns on, and the page gets a synthetic scan built
from the figure's own skin: both arms moved sideways, the whole figure turned by 8 degrees about
the vertical and moved by a few centimetres. Then, as a person would:

- stage 1 shows one point at the pelvis and nothing else; a drag of that point with the mouse in
  the front view and another in the side view line the pelvis up with the scan's pelvis, and the
  Turn slider turns the figure onto the scan; the drag is one step of undo;
- Next shows stage 2 with the head, the hands and the feet alone; a drag of the left wrist onto
  the scan's turns the left arm, the elbow follows it, and the figure's move of stage 1 and the
  other limbs stay; Place these points puts the other points of the stage on the scan;
- Next shows stage 3 with the twelve points in between; Place these points puts them on the scan,
  and the stage-2 points keep their places and stay met;
- the stepper goes back to stage 1, and a session reopened from the card's points opens at the
  stage where its work is left.

Set OSB_POSE_SHOTS to a folder to keep a picture of each stage.
"""

import os
import pathlib
import shutil
import tempfile
import unittest

import numpy as np

from test.markers import local_only
from test.test_viewer_starters import open_tab, playwright_available, served, stop

REPORT = "() => window.poseEditReport()"

# the synthetic scan: the figure's skin in another pose, turned and moved (card frame, Anny's axes, z up)
TURN_DEG = 8.0
SHIFT = [0.045, -0.03, 0.025]
SCAN = """(a) => {
  const ed = window.__poseEdit, S = ed.session, J = ed.joints(), body = ed.skinnedBody();
  const R = S.rotations.map((r) => r.slice());
  J.set(R, J.bone('upperarm01.R'), 1, 25, false);
  J.set(R, J.bone('upperarm01.L'), 1, 25, false);
  const V = body.vertices(R), F = body.faces, n = V.length / 3;
  const c = Math.cos(a.turn * Math.PI / 180), s = Math.sin(a.turn * Math.PI / 180), t = a.shift;
  const P = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const x = V[i * 3], y = V[i * 3 + 1], z = V[i * 3 + 2];
    P[i * 3] = c * x - s * y + t[0]; P[i * 3 + 1] = s * x + c * y + t[1]; P[i * 3 + 2] = z + t[2];
  }
  const N = new Float32Array(n * 3);
  for (let k = 0; k < F.length; k += 3) {
    const i = F[k] * 3, j = F[k + 1] * 3, l = F[k + 2] * 3;
    const e1 = [P[j] - P[i], P[j + 1] - P[i + 1], P[j + 2] - P[i + 2]];
    const e2 = [P[l] - P[i], P[l + 1] - P[i + 1], P[l + 2] - P[i + 2]];
    const m = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    for (const v of [i, j, l]) for (let q = 0; q < 3; q++) N[v + q] += m[q];
  }
  for (let i = 0; i < n; i++) {
    const l = Math.hypot(N[i * 3], N[i * 3 + 1], N[i * 3 + 2]) || 1;
    for (let q = 0; q < 3; q++) N[i * 3 + q] /= l;
  }
  // Anny's points on the scan's pose: the truth the stages work towards
  const snap = S.snapshot();
  const D = R[S.root], Z = [c, -s, 0, s, c, 0, 0, 0, 1], ZD = new Array(9).fill(0);
  for (let r = 0; r < 3; r++)
    for (let q = 0; q < 3; q++) for (let k = 0; k < 3; k++) ZD[r * 3 + q] += Z[r * 3 + k] * D[k * 3 + q];
  S.setPose(R);
  S.setRoot(ZD, t);
  const truth = Object.fromEntries([...S.positions()].map(([k, v]) => [k, v]));
  S.restore(snap);
  ed.setScan({ positions: P, indices: Uint32Array.from(F), normals: N, bytes: P.byteLength + F.length * 4 });
  window.__truth = truth;
  return truth;
}"""


def report(tab):
    return tab.evaluate(REPORT)


def settle(tab, frames=4):
    for _ in range(frames):
        tab.evaluate("() => new Promise((r) => requestAnimationFrame(() => r()))")


def screen_of(tab, card_point):
    """The client pixels of a point of the card frame."""
    return tab.evaluate(
        "(x) => { const w = window.__poseEdit.frames.toWorld(x), s = window.__studio.project(w);"
        " return [s.x, s.y]; }",
        card_point,
    )


def drag(tab, a, b, steps=24):
    tab.mouse.move(a[0], a[1])
    tab.mouse.down()
    for i in range(1, steps + 1):
        tab.mouse.move(
            a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps
        )
    tab.mouse.up()
    settle(tab)


def gap(r, truth, name):
    return float(np.linalg.norm(np.subtract(r["points"][name]["anny"], truth[name])))


@local_only("Playwright and Chromium with the built page")
@unittest.skipUnless(playwright_available(), "Playwright and the built page are needed")
class TestPointStagesPage(unittest.TestCase):
    def test_stages_one_to_three(self):
        from playwright.sync_api import sync_playwright

        folder = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        servers, url = served(folder, {})
        self.addCleanup(stop, servers)
        shots = os.environ.get("OSB_POSE_SHOTS")
        with sync_playwright() as p:
            browser, tab = open_tab(p, url + "?base", 1600, 1000)

            def shot(name):
                if shots:
                    # the first visit's offers stay out of the pictures
                    for label in ("Not now", "Got it"):
                        button = tab.get_by_role("button", name=label)
                        if button.count() and button.first.is_visible():
                            button.first.click()
                    tab.screenshot(
                        path=str(pathlib.Path(shots) / f"{name}.png"), timeout=600000
                    )

            tab.evaluate("() => __shell.actions.openMode('pose')")
            tab.wait_for_function("() => !!window.__poseEdit")
            tab.evaluate("() => document.getElementById('c').focus()")
            tab.keyboard.press("Tab")
            tab.wait_for_function("() => window.poseEditReport().state.on")
            tab.evaluate("() => window.__poseEdit.restPose()")
            truth = tab.evaluate(SCAN, {"turn": TURN_DEG, "shift": SHIFT})
            tab.wait_for_function("() => window.poseEditReport().state.scan.available")
            tab.keyboard.press("Numpad1")
            tab.wait_for_timeout(700)
            settle(tab)

            # ---------------------------------------------------------- stage 1
            r = report(tab)
            self.assertEqual(r["state"]["stage"], 1)
            self.assertEqual(r["shown"], [], "stage 1 shows no reference point")
            self.assertIsNotNone(r["figure"]["screen"])
            panel = tab.locator(".pe-points")
            self.assertIn("Stage 1 of 4", panel.inner_text())
            self.assertIn("pelvis", panel.locator(".pe-stage-line").inner_text())
            shot("stage1_start")
            # the scan's pelvis: the truth's pelvis is the shift (the root's head)
            start = r["figure"]["screen"]
            drag(tab, (start["x"], start["y"]), screen_of(tab, SHIFT))
            r = report(tab)
            pel = r["figure"]["pelvis"]
            # the front view moves the pelvis across and up (Anny's x and z); its depth stays
            self.assertLess(abs(pel[0] - SHIFT[0]), 0.004)
            self.assertLess(abs(pel[2] - SHIFT[2]), 0.004)
            self.assertLess(abs(pel[1]), 0.002)
            self.assertEqual(r["undo"][-1], "Move the figure")
            # the side view moves its depth
            tab.keyboard.press("Numpad3")
            tab.wait_for_timeout(700)
            settle(tab)
            s = report(tab)["figure"]["screen"]
            drag(tab, (s["x"], s["y"]), screen_of(tab, SHIFT))
            pel = report(tab)["figure"]["pelvis"]
            self.assertLess(float(np.linalg.norm(np.subtract(pel, SHIFT))), 0.006)
            tab.keyboard.press("Numpad1")
            tab.wait_for_timeout(700)
            # the turn: typed into the slider's number
            box = tab.get_by_role("spinbutton", name="Turn the figure")
            box.fill(str(int(TURN_DEG)))
            box.press("Enter")
            settle(tab)
            r = report(tab)
            self.assertAlmostEqual(r["figure"]["root"]["turn"], TURN_DEG, places=3)
            self.assertTrue(r["state"]["root"]["moved"])
            # the body's points stand within a few millimetres of the scan's where the pose agrees (the legs)
            for name in ("knee.L", "knee.R", "ankle.L", "head_top"):
                self.assertLess(gap(r, truth, name), 0.012, name)
            settle(tab)
            shot("stage1_placed")

            # ---------------------------------------------------------- stage 2
            tab.get_by_role("button", name="Next").click()
            settle(tab)
            r = report(tab)
            self.assertEqual(r["state"]["stage"], 2)
            self.assertEqual(r["state"]["tool"], "move")
            self.assertEqual(
                set(r["shown"]),
                {"head_top", "wrist.R", "wrist.L", "ankle.R", "ankle.L"},
            )
            self.assertEqual(r["state"]["counts"]["total"], 5)
            self.assertIn("Stage 2 of 4", panel.inner_text())
            self.assertEqual(tab.locator(".body-map .map-point").count(), 5)
            pelvis = r["figure"]["pelvis"]
            still = {
                n: r["points"][n]["anny"] for n in ("knee.L", "wrist.R", "head_top")
            }
            elbow = r["points"]["elbow.L"]["anny"]
            gap0 = gap(r, truth, "wrist.L")
            # a drag of the left wrist onto the scan's: the left arm turns and the elbow follows
            a = r["points"]["wrist.L"]["screen"]
            drag(tab, (a["x"], a["y"]), screen_of(tab, truth["wrist.L"]), steps=16)
            r = report(tab)
            w = r["points"]["wrist.L"]
            self.assertEqual(r["state"]["selected"], "wrist.L")
            self.assertEqual(
                r["state"]["hand"], None, "a wrist stays on the stage's map"
            )
            self.assertEqual(w["placed"], "owner")
            self.assertLess(w["onScan"], 1e-4)
            self.assertLess(
                float(np.linalg.norm(np.subtract(w["anny"], w["target"]))), 0.01
            )
            self.assertLess(gap(r, truth, "wrist.L"), 0.5 * gap0)
            self.assertGreater(
                float(
                    np.linalg.norm(np.subtract(r["points"]["elbow.L"]["anny"], elbow))
                ),
                0.005,
            )
            self.assertEqual(r["figure"]["pelvis"], pelvis, "the figure's move stays")
            for n, x in still.items():
                np.testing.assert_allclose(
                    r["points"][n]["anny"], x, atol=1e-9, err_msg=n
                )
            shot("stage2_wrist")
            tab.get_by_role("button", name="Place these points").click()
            settle(tab)
            r = report(tab)
            placed = [n for n in r["shown"] if r["points"][n]["target"] is not None]
            self.assertGreaterEqual(len(placed), 4)
            self.assertEqual(r["points"]["wrist.L"]["placed"], "owner")
            self.assertEqual(
                r["points"]["wrist.L"]["target"],
                w["target"],
                "a point placed by hand stays",
            )
            for n in ("shoulder.R", "index_tip.R", "knee.L"):
                self.assertIsNone(r["points"][n]["target"], n)
            for n in placed:
                if r["points"][n]["on"]:
                    met = np.linalg.norm(
                        np.subtract(r["points"][n]["anny"], r["points"][n]["target"])
                    )
                    self.assertLess(float(met), 0.015, n)
            stage2 = {n: r["points"][n]["target"] for n in placed}
            shot("stage2_placed")

            # ---------------------------------------------------------- stage 3
            tab.get_by_role("button", name="Next").click()
            settle(tab)
            r = report(tab)
            self.assertEqual(r["state"]["stage"], 3)
            self.assertEqual(r["state"]["counts"]["total"], 12)
            self.assertIn("Stage 3 of 4", panel.inner_text())
            self.assertEqual(tab.locator(".body-map .map-point").count(), 12)
            self.assertTrue(set(stage2) <= set(r["shown"]))
            self.assertFalse(any(n.startswith("index") for n in r["shown"]))
            tab.get_by_role("button", name="Place these points").click()
            settle(tab)
            r = report(tab)
            for n, x in stage2.items():
                self.assertEqual(r["points"][n]["target"], x, n)
            own = [n for n in r["shown"] if n not in stage2]
            self.assertEqual(len(own), 12)
            self.assertGreaterEqual(
                sum(r["points"][n]["target"] is not None for n in own), 8
            )
            # the stage-2 points stay met in the solves of stage 3
            w = r["points"]["wrist.L"]
            self.assertLess(
                float(np.linalg.norm(np.subtract(w["anny"], w["target"]))), 0.015
            )
            looks = r["state"]["counts"]["look"]
            self.assertEqual(looks, len(r["state"]["looks"]))
            shot("stage3_placed")

            # ---------------------------------------------------------- back, and a session reopened
            tab.locator(".pe-stage[data-stage='1']").click()
            settle(tab)
            r = report(tab)
            self.assertEqual(r["state"]["stage"], 1)
            self.assertIn("Stage 1 of 4", panel.inner_text())
            self.assertEqual(r["shown"], [])
            # the points read back from the card's entry open the session where work is left: the first of
            # stages 2 and 3 with a point not placed or one that needs a look
            stage = tab.evaluate(
                "() => { const ed = window.__poseEdit, S = ed.session;"
                " S.readCard(S.cardPoints(), []); return ed.openingStage(); }"
            )
            r2 = report(tab)

            def left(names):
                return any(
                    r2["points"][n]["target"] is None
                    or (
                        not r2["points"][n]["on"]
                        and r2["points"][n]["confidence"] < 0.5
                    )
                    for n in names
                )

            main = ["head_top", "wrist.R", "wrist.L", "ankle.R", "ankle.L"]
            between = [
                n
                for n in r2["points"]
                if n not in main and r2["points"][n]["number"] <= 15
            ]
            if left(main):
                self.assertEqual(stage, 2)
            elif left(between):
                self.assertEqual(stage, 3)
            else:
                self.assertIn(stage, (3, 4))
            for n in stage2:
                self.assertEqual(r2["points"][n]["target"], stage2[n], n)
            tab.locator(".pe-stage[data-stage='4']").click()
            settle(tab)
            self.assertIn("Open Hands", tab.locator(".pe-points").inner_text())
            shot("stage4")
            tab.get_by_role("button", name="Open Hands").click()
            settle(tab)
            r = report(tab)
            self.assertEqual(r["state"]["section"], "hands")
            self.assertTrue(r["state"]["fingers"])
            self.assertEqual(len(r["shown"]), 21)
            self.assertEqual(
                tab.locator(".pe-finger-points .hand-map .map-point").count(), 21
            )
            shot("stage4_hands")
            browser.close()


if __name__ == "__main__":
    unittest.main()

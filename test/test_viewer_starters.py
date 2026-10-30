# OpenSculptBoy
# Apache License, Version 2.0
"""
The statue starters of the viewer page and the floor under their props, in Chromium through
Playwright with the built page (local only).

- The packages beside a served page (viewer/serve_viewer.py on a free loopback port): a package
  of the pinned size with other bytes, which the page refuses by its integrity on the first visit
  and from the New screen, and a missing package, whose note gives its download link. The page
  asks only its own server.
- The floor under a synthetic plinth (a prop under the soles, held by the root bone) on a dropped
  card: the plinth's bottom stands on the floor and the body on the plinth; with the plinth
  switched off in Dress the soles go back to the floor, and on again the body rises.
- With OPENSCULPTBOY_STARTERS set to a folder of the real packages (the release's files), each
  statue opens from the New screen with its props on the floor, and again with its props off.

The page's loader and the floor's maths have node checks without a browser in
viewer/test/card_file.mjs (test.test_viewer_card_file).
"""

import base64
import json
import os
import pathlib
import shutil
import socket
import tempfile
import unittest

import numpy as np

from test.markers import local_only
from test.test_hair_page import CHROMIUM, PAGE, playwright_available

REPO = pathlib.Path(__file__).resolve().parents[1]
PINS = json.loads(
    (REPO / "viewer" / "src" / "starter_packages.json").read_text(encoding="utf-8")
)
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
DROP = """(files) => {
  const dt = new DataTransfer();
  for (const [name, b64] of Object.entries(files))
    dt.items.add(new File([Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))], name));
  document.dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true }));
}"""
DRESSED = "() => window.wearStats() && window.wearStats().timing.full === true"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def served(folder: pathlib.Path, packages: dict):
    """The built page in ``folder`` with ``packages`` (file name to bytes) in its starters
    folder, served by viewer/serve_viewer.py on a free port; returns the servers and the URL."""
    import importlib.util

    shutil.copy(PAGE, folder / PAGE.name)
    (folder / "starters").mkdir(exist_ok=True)
    for name, data in packages.items():
        (folder / "starters" / name).write_bytes(data)
    spec = importlib.util.spec_from_file_location(
        "serve_viewer", REPO / "viewer" / "serve_viewer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    port = free_port()
    servers = module.start(folder / PAGE.name, port)
    return servers, f"http://localhost:{port}/{PAGE.name}"


def open_tab(p, url: str, width=1280, height=800):
    kw = dict(executable_path=str(CHROMIUM)) if CHROMIUM.exists() else {}
    browser = p.chromium.launch(args=ARGS, **kw)
    tab = browser.new_context(viewport=dict(width=width, height=height)).new_page()
    tab.set_default_timeout(120000)
    tab.goto(url)
    tab.wait_for_function("() => window.__ready === true", timeout=300000)
    return browser, tab


def plinth_card(folder: pathlib.Path):
    """A card of Anny's default body with one prop: a box of 10 cm under the soles, held by the
    root bone. The Python side of this branch has no slot ``prop`` (feat/statue-starters adds
    it), so the package is written as an accessory and its JSON names the slot ``prop`` after."""
    from opensculptboy.wearables import (
        Garment,
        Material,
        Wearable,
        bind_rigid,
        write_wearable,
    )
    from test.wearable_fixtures import LICENCE, template_body

    body = template_body()
    v = body.vertices
    low = float(v[:, 2].min())
    feet = v[v[:, 2] < low + 0.05]
    (x0, y0), (x1, y1) = feet[:, :2].min(0) - 0.08, feet[:, :2].max(0) + 0.08
    corners = np.array(
        [[x, y, z] for z in (low - 0.1, low) for y in (y0, y1) for x in (x0, x1)]
    )
    quads = [
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    ]
    triangles = np.array(
        [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], np.int64
    )
    centre = corners.mean(0)
    normals = (corners - centre) / np.linalg.norm(corners - centre, axis=1)[:, None]
    garment = Garment(
        corners,
        triangles,
        normals=normals,
        materials=[Material("marble", (0.85, 0.84, 0.82, 1.0), 0.0, 0.6)],
    )
    plinth = Wearable(
        name="plinth",
        slot="accessory",
        layer=1,
        garment=garment,
        binding=bind_rigid(garment, body, "root"),
        weights="rigid",
        licence=dict(LICENCE),
        covers=np.zeros(len(v), bool),
    )
    path = folder / "plinth.wearable.json"
    write_wearable(path, plinth)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["slot"] = "prop"
    path.write_text(json.dumps(data), encoding="utf-8")
    card = {
        "schema": 5,
        "name": "Plinth",
        "outfits": [
            {
                "id": "statue",
                "label": "Statue pieces",
                "pieces": [{"piece": "plinth", "file": "plinth.wearable.json"}],
            }
        ],
        "active_outfit": "statue",
    }
    (folder / "character.json").write_text(json.dumps(card), encoding="utf-8")
    return sorted(f for f in folder.iterdir() if f.is_file())


def switch(tab, ref: str, on: bool):
    """a piece's switch in Dress"""
    tab.evaluate("() => __shell.actions.openMode('dress')")
    button = tab.locator(f"button[role=switch][data-ref='{ref}']")
    if (button.get_attribute("aria-checked") == "true") != on:
        button.click()
    tab.wait_for_function(DRESSED)


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestStarterPackages(unittest.TestCase):
    def test_served(self):
        from playwright.sync_api import sync_playwright

        folder = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        dory, david = PINS["packages"]["doryphoros"], PINS["packages"]["david_goliath"]
        # the pinned size with other bytes: the page fetches it and the browser refuses it
        servers, url = served(folder, {dory["file"]: b"\x00" * dory["bytes"]})
        self.addCleanup(lambda: [s.shutdown() for s in servers])
        with sync_playwright() as p:
            browser, tab = open_tab(p, url)
            self.addCleanup(browser.close)
            requests = []
            tab.on("request", lambda r: requests.append(r.url))
            # the first visit tries the default statue: the bytes differ, so the welcome card shows its note
            tab.wait_for_function("() => __shell.ui.get().starter.note !== null")
            note = tab.evaluate("() => __shell.ui.get().starter.note")
            self.assertEqual((note["id"], note["problem"]), ("doryphoros", "other"))
            self.assertEqual(tab.evaluate("() => window.__look().name"), "Baseline")
            self.assertIsNone(tab.evaluate("() => window.wearStats()?.card || null"))
            welcome = tab.locator(".welcome")
            self.assertIn("Meet the studio", welcome.inner_text())
            self.assertEqual(
                welcome.locator(".welcome-starter a").get_attribute("href"),
                PINS["download"] + "doryphoros.zip",
            )
            # the New screen: the two statues, Doryphoros first, then the blank body; the missing package is tagged
            tab.keyboard.press("n")
            tab.wait_for_function(
                "() => document.querySelector(\"[data-starter='david_goliath']\")?.dataset.package"
            )
            slots = tab.eval_on_selector_all(
                "[data-starter]",
                "(bs) => bs.map((b) => [b.dataset.starter, b.dataset.package || ''])",
            )
            self.assertEqual(
                slots,
                [["doryphoros", "here"], ["david_goliath", "missing"], ["blank", ""]],
            )
            tab.locator("[data-starter='david_goliath']").click()
            tab.locator(".new-start").click()
            tab.wait_for_selector(".new-note")
            text = tab.locator(".new-note").inner_text()
            self.assertIn("david_goliath.zip (%.1f MB)" % (david["bytes"] / 1e6), text)
            self.assertEqual(
                tab.locator(".new-note a").get_attribute("href"),
                PINS["download"] + "david_goliath.zip",
            )
            tab.locator("[data-starter='doryphoros']").click()
            self.assertEqual(tab.locator(".new-note").count(), 0)
            tab.locator(".new-start").click()
            tab.wait_for_selector(".new-note")
            self.assertIn("another version", tab.locator(".new-note").inner_text())
            self.assertTrue(tab.evaluate("() => __shell.ui.get().newDialog"))
            # the blank body
            tab.locator("[data-starter='blank']").click()
            tab.locator(".new-start").click()
            tab.wait_for_function("() => !__shell.ui.get().newDialog")
            self.assertEqual(tab.evaluate("() => window.viewState().clip"), "a_pose")
            origin = url.rsplit("/", 1)[0] + "/"
            self.assertEqual([r for r in requests if not r.startswith(origin)], [])


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestPropFloor(unittest.TestCase):
    def test_plinth(self):
        from playwright.sync_api import sync_playwright

        folder = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        files = plinth_card(folder)
        payload = {f.name: base64.b64encode(f.read_bytes()).decode() for f in files}
        with sync_playwright() as p:
            browser, tab = open_tab(p, PAGE.as_uri() + "?base")
            self.addCleanup(browser.close)
            before = tab.evaluate("() => window.groundInfo()")
            self.assertIsNone(before["props"])
            self.assertAlmostEqual(before["body"], before["floor"], delta=1e-4)
            tab.evaluate(DROP, payload)
            tab.wait_for_function(DRESSED)
            on = tab.evaluate("() => window.groundInfo()")
            # the plinth's bottom on the floor, the body on the plinth: 10 cm of anny's frame in the page's scale
            self.assertAlmostEqual(on["props"], on["floor"], delta=1e-4)
            self.assertAlmostEqual(on["body"] - on["floor"], on["lift"], delta=1e-4)
            self.assertGreater(on["lift"], 0.08)
            self.assertLess(on["lift"], 0.1)
            switch(tab, "statue/plinth", False)
            off = tab.evaluate("() => window.groundInfo()")
            self.assertIsNone(off["props"])
            self.assertAlmostEqual(off["lift"], 0, delta=1e-6)
            self.assertAlmostEqual(off["body"], off["floor"], delta=1e-4)
            switch(tab, "statue/plinth", True)
            again = tab.evaluate("() => window.groundInfo()")
            self.assertAlmostEqual(again["props"], again["floor"], delta=1e-4)
            self.assertAlmostEqual(again["lift"], on["lift"], delta=1e-5)
            # the drawn floor stays under the plinth
            self.assertLess(again["plane"], again["props"])


def real_packages():
    """the folder of the real packages when OPENSCULPTBOY_STARTERS names one that holds them"""
    folder = os.environ.get("OPENSCULPTBOY_STARTERS")
    if not folder:
        return None
    folder = pathlib.Path(folder)
    ok = all((folder / e["file"]).exists() for e in PINS["packages"].values())
    return folder if ok else None


@local_only("Playwright with the built viewer page and the statue packages")
@unittest.skipUnless(
    playwright_available() and real_packages(),
    "Playwright, the built page and OPENSCULPTBOY_STARTERS are needed",
)
class TestStatueStarters(unittest.TestCase):
    def test_statues(self):
        from playwright.sync_api import sync_playwright

        folder = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        packages = {
            e["file"]: (real_packages() / e["file"]).read_bytes()
            for e in PINS["packages"].values()
        }
        servers, url = served(folder, packages)
        self.addCleanup(lambda: [s.shutdown() for s in servers])
        with sync_playwright() as p:
            browser, tab = open_tab(p, url + "?querypool=off")
            self.addCleanup(browser.close)
            # the first visit opens the Doryphoros
            tab.wait_for_function(
                "() => window.wearStats()?.card === %s"
                % json.dumps(PINS["packages"]["doryphoros"]["card"]),
                timeout=300000,
            )
            for statue, entry in PINS["packages"].items():
                with self.subTest(statue=statue):
                    # the blank body first, so that the statue's card opens anew
                    tab.evaluate("() => __shell.actions.showNew()")
                    tab.locator("[data-starter='blank']").click()
                    tab.locator(".new-start").click()
                    tab.wait_for_function("() => !window.wearStats()?.card")
                    tab.evaluate("() => __shell.actions.showNew()")
                    tab.locator(f"[data-starter='{statue}']").click()
                    tab.locator(".new-start").click()
                    tab.wait_for_function(
                        "(n) => window.wearStats()?.card === n && window.wearStats().timing.full === true",
                        arg=entry["card"],
                        timeout=300000,
                    )
                    on = tab.evaluate("() => window.groundInfo()")
                    self.assertAlmostEqual(on["props"], on["floor"], delta=1e-4)
                    self.assertGreaterEqual(on["lift"], 0)
                    tab.evaluate("() => __shell.actions.openMode('dress')")
                    refs = tab.eval_on_selector_all(
                        "button[role=switch][data-ref^='statue/']",
                        "(bs) => bs.map((b) => b.dataset.ref)",
                    )
                    self.assertTrue(refs)
                    for ref in refs:
                        switch(tab, ref, False)
                    off = tab.evaluate("() => window.groundInfo()")
                    self.assertIsNone(off["props"])
                    self.assertAlmostEqual(off["body"], off["floor"], delta=1e-4)
                    for ref in refs:
                        switch(tab, ref, True)
                    again = tab.evaluate("() => window.groundInfo()")
                    self.assertAlmostEqual(again["lift"], on["lift"], delta=1e-4)


if __name__ == "__main__":
    unittest.main()

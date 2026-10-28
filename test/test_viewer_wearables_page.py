# OpenSculptBoy
# Apache License, Version 2.0
"""
The Wearables section of the built viewer page, in Chromium with SwiftShader through Playwright:
the synthetic outfit of test.test_viewer_wearables opened through the section's file input, the
pieces and the hides rules as opensculptboy.wearables resolves them, the dressed vertices
against opensculptboy.wearables.dress on the same body, the switches of the section (a hidden
piece, a piece of another outfit in its slot, the covered skin), a slider edit, and no request
of the page during the load. The test skips when Playwright or the built page is missing.
"""

import pathlib
import shutil
import tempfile
import unittest

import numpy as np

from test.markers import local_only
from test.test_hair_page import CHROMIUM, PAGE, playwright_available
from test.test_viewer_wearables import _packages

ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
ANNY_TO_PAGE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestWearablesPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        torch.set_num_threads(1)
        cls.dir = pathlib.Path(tempfile.mkdtemp())
        cls.card = _packages(cls.dir)[0]
        cls.files = sorted(str(f) for f in _card_files(cls.dir))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def run_page(self, steps):
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            kw = dict(executable_path=str(CHROMIUM)) if CHROMIUM.exists() else {}
            browser = p.chromium.launch(args=ARGS, **kw)
            tab = browser.new_context(viewport=dict(width=900, height=640)).new_page()
            requests = []
            tab.on("request", lambda r: requests.append((r.method, r.url)))
            tab.goto(PAGE.as_uri() + "?shot&acc=1")
            tab.wait_for_function("() => window.__ready === true", timeout=300000)
            before = len(requests)
            tab.locator(
                "input[type=file][multiple]:not([webkitdirectory])"
            ).set_input_files(self.files)
            tab.wait_for_function(
                "() => window.wearStats() && window.wearStats().pieces.length > 0"
                " && window.wearStats().timing.full === true",
                timeout=120000,
            )
            out = steps(tab)
            out["requests"] = requests[before:]
            browser.close()
        return out

    def compare(self, pieces, expected, scale, median, largest):
        """The page's dressed vertices against Python's: the page's body comes from its
        compressed shape space and stands on the floor, so the pieces match after one shift,
        within the error of the shape space."""
        got = {p["ref"]: p for p in pieces}
        self.assertEqual(sorted(got), sorted(p.ref for p in expected))
        page = np.concatenate(
            [np.array(got[p.ref]["vertices"]).reshape(-1, 3) for p in expected]
        )
        want = np.concatenate([p.vertices for p in expected])
        anny = (page @ ANNY_TO_PAGE) / scale
        error = np.linalg.norm(anny - (anny - want).mean(0) - want, axis=1)
        print(
            f"\npage against Python: {error.max() * 1000:.2f} mm at most, "
            f"{np.median(error) * 1000:.3f} mm median"
        )
        self.assertLess(float(np.median(error)), median)
        self.assertLess(float(error.max()), largest)

    def test_page(self):
        from opensculptboy import Character
        from opensculptboy.wearables import card_body, dress, worn_pieces
        from test.wearable_fixtures import template_model

        def switch(tab, ref):
            tab.evaluate(
                "(ref) => document.querySelector(`.wear-piece[data-ref='${ref}'] button.switch`).click()",
                ref,
            )
            tab.wait_for_function("() => window.wearStats().timing.full === true")
            return tab.evaluate("() => window.wearPieces()")

        def steps(tab):
            out = {"stats": tab.evaluate("() => window.wearStats()")}
            out["scale"] = tab.evaluate(
                "() => window.__BODY.anny.meta.shape.frame.scale"
            )
            out["pieces"] = tab.evaluate("() => window.wearPieces()")
            out["no_cape"] = switch(tab, "travel/cape")
            out["smock"] = switch(tab, "work/smock")
            # the last switch of the section: the covered skin
            tab.evaluate(
                "() => [...document.querySelectorAll('.wear-body button.switch')].pop().click()"
            )
            out["kept"] = tab.evaluate("() => window.wearStats().hidden_triangles")
            out["slider"] = tab.evaluate("() => window.setSliders({ weight: 0.9 })")
            out["fast"] = tab.evaluate("() => window.wearStats().timing")
            tab.wait_for_function("() => window.wearStats().timing.full === true")
            out["settled"] = tab.evaluate("() => window.wearStats().timing")
            out["heavy"] = tab.evaluate("() => window.wearPieces()")
            return out

        out = self.run_page(steps)
        body = card_body(Character(), template_model())
        expected = dress(body, worn_pieces(self.card, self.dir))
        got = {p["ref"]: p for p in out["pieces"]}
        self.assertEqual(sorted(got), sorted(p.ref for p in expected))
        self.assertEqual(got["travel/tunic"]["hidden"], ["back"])
        self.compare(out["pieces"], expected, out["scale"], 0.0005, 0.004)
        self.assertGreater(out["stats"]["hidden_triangles"], 1000)
        # the switches: the cape off shows the tunic's back again; the smock takes the tunic's slot
        no_cape = {p["ref"]: p for p in out["no_cape"]}
        self.assertNotIn("travel/cape", no_cape)
        self.assertEqual(no_cape["travel/tunic"]["hidden"], [])
        smock = [p["ref"] for p in out["smock"]]
        self.assertIn("work/smock", smock)
        self.assertNotIn("travel/tunic", smock)
        self.assertEqual(out["kept"], 0)
        # a slider edit moves the pieces at once, and the push-out follows when it rests
        self.assertFalse(out["fast"]["full"])
        self.assertNotIn("push", out["fast"])
        self.assertTrue(out["settled"]["full"])
        print(f"slider: fast {out['fast']}, settled {out['settled']}")
        # the heavier body, with the cape hidden and the smock in the tunic's slot
        heavy = card_body(Character(phenotype={"weight": 0.9}), template_model())
        worn = worn_pieces(
            self.card, self.dir, show=["work/smock"], hide=["travel/cape"]
        )
        self.compare(out["heavy"], dress(heavy, worn), out["scale"], 0.001, 0.006)
        # the files stay in the browser
        self.assertFalse(
            [r for r in out["requests"] if r[0] != "GET" or "wearable" in r[1]]
        )


def _card_files(folder: pathlib.Path):
    """The files of the first card of _packages: its second card (made together) and that
    card's packages stay out, so that the page opens one card."""
    other = ("together.", "inner.", "coat.")
    return [f for f in folder.iterdir() if not f.name.startswith(other)]


def _dropped_card(folder: pathlib.Path):
    """The synthetic packages of a card fitted on the anny-mouth rig, as the files of a drop:
    the outfit and loose pieces of _packages, a helm with its own weights on head and jaw (a
    bone that the page's rig moves with head), and a piece on a bone that no rig has. The card
    names the packages in a subfolder and its level-0 detail offsets in a parent folder, and
    the files lie in one folder, as a drop by file name alone gives them. Returns the files and
    the references of the pieces that the page dresses."""
    import json

    from safetensors.numpy import save_file

    from opensculptboy.wearables import bind_surface, write_wearable
    from test.wearable_fixtures import circlet, template_body, tunic

    card = _packages(folder)[0]
    body = template_body()
    helm = circlet()
    helm.name, helm.slot, helm.layer, helm.weights = "helm", "head", 60, "own"
    helm.binding = bind_surface(helm.garment, body)
    n = helm.garment.vertices.shape[0]
    helm.binding.skin_weights = np.tile([0.75, 0.25], (n, 1))
    helm.binding.skin_joints = np.tile([0, 1], (n, 1))
    helm.binding.joint_names = ["head", "jaw"]
    write_wearable(folder / "helm.wearable.json", helm)
    odd = tunic(offset=0.02)
    odd.name, odd.slot, odd.layer, odd.weights = "odd", "accessory", 75, "own"
    m = odd.garment.vertices.shape[0]
    odd.binding.skin_weights = np.ones((m, 1))
    odd.binding.skin_joints = np.zeros((m, 1), np.int64)
    odd.binding.joint_names = ["no_such_bone"]
    write_wearable(folder / "odd.wearable.json", odd)
    data = card.to_dict()
    data["rig"] = "anny-mouth"
    data["outfits"][0]["pieces"].append({"piece": "helm", "file": "helm.wearable.json"})
    data["wearables"].append({"id": "odd", "file": "odd.wearable.json"})
    for o in data["outfits"]:
        for p in o["pieces"]:
            p["file"] = "wearables/" + p["file"]
    for w in data["wearables"]:
        w["file"] = "wearables/" + w["file"]
    rows = len(body.vertices)
    save_file(
        {"offsets": np.zeros((rows, 3), np.float32)},
        str(folder / "card.detail0.safetensors"),
    )
    data["detail"] = {
        "levels": [{"subdiv": 0, "offsets": "../../fit/new/card.detail0.safetensors"}]
    }
    (folder / "card.json").write_text(json.dumps(data, indent=1))
    worn = ["travel/tunic", "travel/cape", "travel/circlet", "travel/helm"]
    return sorted(_card_files(folder)), worn + ["plate", "vest"]


@local_only("Playwright with the built viewer page")
@unittest.skipUnless(
    playwright_available(), "Playwright and the built viewer page are needed"
)
class TestWearablesDrop(unittest.TestCase):
    """A card and its packages dropped on the page by a script (a DataTransfer whose items
    have no entries): the worn pieces become meshes, the helm's jaw weights move with head,
    and the piece on an unknown bone shows a message in the section and in the console."""

    def test_drop(self):
        import base64

        from playwright.sync_api import sync_playwright

        folder = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        files, worn = _dropped_card(folder)
        payload = {f.name: base64.b64encode(f.read_bytes()).decode() for f in files}
        with sync_playwright() as p:
            kw = dict(executable_path=str(CHROMIUM)) if CHROMIUM.exists() else {}
            browser = p.chromium.launch(args=ARGS, **kw)
            tab = browser.new_context(viewport=dict(width=900, height=640)).new_page()
            errors = []
            tab.on(
                "console",
                lambda m: errors.append(m.text) if m.type == "error" else None,
            )
            tab.goto(PAGE.as_uri() + "?stats=1")
            tab.wait_for_function("() => window.__ready === true", timeout=300000)
            tab.evaluate(
                """(files) => {
                  const dt = new DataTransfer();
                  for (const [name, b64] of Object.entries(files)) {
                    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
                    dt.items.add(new File([bytes], name));
                  }
                  document.dispatchEvent(new DragEvent('drop', { dataTransfer: dt, bubbles: true, cancelable: true }));
                }""",
                payload,
            )
            tab.wait_for_function(
                "() => window.wearStats() && window.wearStats().timing.full === true",
                timeout=120000,
            )
            out = tab.evaluate(
                """() => {
                  const w = window.__WEAR(), head = window.__RIG.bones.findIndex((b) => b.name === 'head');
                  const meshes = [...w.meshes.entries()].map(([ref, m]) =>
                    ({ ref, visible: m.mesh.visible, parent: !!m.mesh.parent }));
                  const helm = w.meshes.get('travel/helm').geo.attributes;
                  const bones = new Set();
                  for (let i = 0; i < helm.skinWeight.count * 4; i++)
                    if (helm.skinWeight.array[i] > 0) bones.add(helm.skinIndex.array[i]);
                  return { pieces: window.wearStats().pieces.map((p) => p.ref), meshes, head, helmBones: [...bones],
                    notes: w.notes, msg: document.querySelector('.wear-msg').textContent,
                    kind: document.querySelector('.wear-msg').dataset.kind };
                }"""
            )
            browser.close()
        self.assertEqual(sorted(out["pieces"]), sorted(worn))
        self.assertEqual(sorted(m["ref"] for m in out["meshes"]), sorted(worn))
        self.assertTrue(all(m["visible"] and m["parent"] for m in out["meshes"]))
        self.assertEqual(out["helmBones"], [out["head"]])
        self.assertIn(
            "travel/helm: the page moves its bones jaw with head", out["notes"]
        )
        self.assertIn("odd: the page's rig lacks its bones no_such_bone", out["msg"])
        self.assertEqual(out["kind"], "error")
        self.assertTrue([e for e in errors if "no_such_bone" in e])


if __name__ == "__main__":
    unittest.main()

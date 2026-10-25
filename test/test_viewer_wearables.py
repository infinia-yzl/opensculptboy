# OpenSculptBoy
# Apache License, Version 2.0
"""
Parity of the page's dressing (viewer/src/wearables/) with opensculptboy.wearables: the skin
triangles that the page rebuilds from its quads, and the pieces of synthetic outfits dressed on
several bodies (surface and rigid bindings, the body's and the package's own skin weights, the
push-out over the body and a lower layer, hides rules, slot replacement and the covered skin),
in Anny's frame and in the page's frame. Node runs the TypeScript modules directly; the test
skips when node is missing.
"""

import dataclasses
import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from test.test_viewer_parity import node_available

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "viewer" / "test" / "wearables_parity.mjs"
# the page's frame: Y up, the figure facing +Z, scaled (as anny.poses.authoring.rig)
PAGE_FRAME = {
    "scale": 0.8916,
    "rotation": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
    "offset": [0.0, 0.62, 0.03],
}
BODIES = {
    "template": {},
    "thin": {"weight": 0.0, "muscle": 0.0},
    "random": {
        "gender": 0.81,
        "age": 0.37,
        "muscle": 0.66,
        "weight": 0.71,
        "height": 0.28,
        "proportions": 0.55,
    },
}
CASES = [
    {"name": "template", "body": "template"},
    {"name": "thin", "body": "thin"},
    {"name": "random", "body": "random"},
    {"name": "hide_cape", "body": "thin", "choices": {"hide": ["cape"]}},
    {"name": "swap", "body": "random", "choices": {"show": ["work/smock"]}},
    {"name": "no_push", "body": "thin", "push": False},
    {"name": "page_frame", "body": "thin", "frame": True},
    {"name": "estimate", "body": "random", "estimate": True},
    # a card that names no outfit wears its loose pieces alone, as the export does
    {"name": "no_outfit", "body": "thin", "card": "bare"},
    # an override that shows a piece of another outfit puts it in its slot
    {"name": "override_swap", "body": "random", "card": "override"},
]


def _packages(folder: pathlib.Path):
    """The synthetic packages and the card: an outfit (tunic, cape that hides the tunic's
    back, rigid circlet), a second outfit whose smock takes the tunic's slot, and two loose
    pieces: a rigid plate under the tunic and a vest with its own skin weights, an opaque front
    and a see-through back, and no cover mask (the ray test decides)."""
    from opensculptboy import Character
    from opensculptboy.wearables import (
        Material,
        bind_rigid,
        bind_surface,
        write_wearable,
    )
    from opensculptboy.wearables.bind import dense_body_weights, smooth_weights
    from test.wearable_fixtures import cape, circlet, template_body, tunic

    body = template_body()
    for w in (tunic(), cape(), circlet()):
        write_wearable(folder / f"{w.name}.wearable.json", w)
    plate = tunic(offset=0.008)
    plate = dataclasses.replace(
        plate,
        name="plate",
        slot="accessory",
        layer=40,
        weights="rigid",
        binding=bind_rigid(plate.garment, body, "spine03"),
    )
    write_wearable(folder / "plate.wearable.json", plate)
    vest = tunic(offset=0.014)
    vest.name, vest.slot, vest.layer = "vest", "accessory", 70
    # a shell off the skin along the vertex normals puts each garment vertex on the normal ray
    # of a body vertex, where the ray test turns on the last bit; a small jitter avoids it
    jitter = np.random.default_rng(5).normal(
        scale=0.0004, size=vest.garment.vertices.shape
    )
    vest.garment.vertices = vest.garment.vertices + jitter
    vest.binding = bind_surface(vest.garment, body)
    vest.covers, vest.group_covers = None, {}
    vest.garment.materials = [
        Material("cloth", (0.3, 0.5, 0.2, 1.0), 0.0, 0.8),
        Material("lace", (0.9, 0.9, 0.9, 1.0), 0.0, 0.6, alpha_mode="MASK"),
    ]
    vest.garment.triangle_material = vest.garment.triangle_group.copy()
    dense = dense_body_weights(vest.binding, body)
    smooth = smooth_weights(dense, vest.garment.vertices, vest.garment.triangles, 40)
    keep = np.flatnonzero(smooth.max(0) > 0)
    order = np.argsort(-smooth[:, keep], axis=1, kind="stable")[:, :6]
    vest.weights = "own"
    vest.binding.skin_weights = np.take_along_axis(smooth[:, keep], order, axis=1)
    vest.binding.skin_joints = order
    vest.binding.joint_names = [body.bone_labels[j] for j in keep]
    write_wearable(folder / "vest.wearable.json", vest)
    card = Character(
        name="dressed",
        outfits=[
            {
                "id": "travel",
                "label": "Travel",
                "pieces": [
                    {"piece": n, "file": f"{n}.wearable.json"}
                    for n in ("tunic", "cape", "circlet")
                ],
            },
            {
                "id": "work",
                "label": "Work",
                "pieces": [{"piece": "smock", "file": "tunic.wearable.json"}],
            },
        ],
        active_outfit="travel",
        wearables=[
            {"id": "plate", "file": "plate.wearable.json"},
            {"id": "vest", "file": "vest.wearable.json"},
        ],
    )
    (folder / "card.json").write_text(json.dumps(card.to_dict(), indent=1))
    return card


@unittest.skipUnless(
    node_available(), "node 22 or later is needed to run the page's TypeScript"
)
class TestViewerWearables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        from opensculptboy import Character
        from opensculptboy.wearables import card_body, dress, worn_pieces
        from opensculptboy.wearables.cover import outfit_cover
        from opensculptboy.wearables.viewer import page_quads
        from test.wearable_fixtures import template_model

        torch.set_num_threads(1)
        cls.dir = pathlib.Path(tempfile.mkdtemp())
        folder = cls.dir / "wear"
        folder.mkdir()
        card = _packages(folder)
        cards = {
            "card": card,
            "bare": dataclasses.replace(card, active_outfit=""),
            "override": dataclasses.replace(
                card, wearable_overrides={"work/smock": {"enabled": True}}
            ),
        }
        for name in ("bare", "override"):
            text = json.dumps(cards[name].to_dict(), indent=1)
            (folder / f"{name}.json").write_text(text)
        model = template_model()
        cls.quads = page_quads()
        cls.n = int(cls.quads.max()) + 1  # the page's body rows
        cls.quads.astype(np.uint32).tofile(cls.dir / "quads.u32")
        cls.bodies, inp_bodies = {}, []
        for name, phenotype in BODIES.items():
            body = card_body(Character(phenotype=phenotype), model)
            cls.bodies[name] = body
            body.vertices[: cls.n].astype(np.float64).tofile(cls.dir / f"{name}.f64")
            body.bone_poses.astype(np.float64).tofile(cls.dir / f"{name}_poses.f64")
            inp_bodies.append({"vertices": f"{name}.f64", "poses": f"{name}_poses.f64"})
        body = cls.bodies["template"]
        body.weights[: cls.n].astype(np.float64).tofile(cls.dir / "weights.f64")
        body.indices[: cls.n].astype(np.int32).tofile(cls.dir / "indices.i32")
        names = list(BODIES)
        cases = [
            dict(c, body=names.index(c["body"]), card=f"{c.get('card', 'card')}.json")
            for c in CASES
        ]
        inp = {
            "folder": str(folder),
            "cases": cases,
            "bodies": inp_bodies,
            "bone_labels": list(body.bone_labels),
            "K": int(body.weights.shape[1]),
            "frame": PAGE_FRAME,
        }
        (cls.dir / "input.json").write_text(json.dumps(inp))
        subprocess.run(["node", str(SCRIPT), str(cls.dir)], check=True, cwd=REPO)
        cls.out = {
            c["name"]: c
            for c in json.loads((cls.dir / "output.json").read_text())["cases"]
        }
        # opensculptboy's own dressing of every case
        cls.expected = {}
        for c in CASES:
            body = cls.bodies[c["body"]]
            choices = c.get("choices", {})
            worn = worn_pieces(
                cards[c.get("card", "card")],
                folder,
                show=choices.get("show", ()),
                hide=choices.get("hide", ()),
            )
            pieces = dress(body, worn, push=c.get("push", True))
            cls.expected[c["name"]] = (pieces, outfit_cover(body, pieces))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def page(self, case, k, kind, dtype):
        return np.fromfile(self.dir / f"{case}_{k}_{kind}", dtype=dtype)

    def test_skin_triangles(self):
        from opensculptboy.wearables.fit import skin_triangles
        from test.wearable_fixtures import template_model

        page = np.fromfile(self.dir / "skin_triangles.i32", dtype=np.int32).reshape(
            -1, 3
        )
        skin = skin_triangles(template_model())

        def canonical(t):
            t = np.asarray(t, dtype=np.int64)
            first = np.argmin(t, axis=1)
            rolled = np.stack([np.roll(r, -k) for r, k in zip(t, first)])
            return rolled[np.lexsort(rolled.T[::-1])]

        np.testing.assert_array_equal(canonical(page), canonical(skin))

    def test_skin_split_file_is_current(self):
        from opensculptboy.wearables.fit import skin_triangles
        from opensculptboy.wearables.viewer import (
            TARGET,
            skin_split_codes,
            skin_split_source,
        )
        from test.wearable_fixtures import template_model

        codes = skin_split_codes(self.quads, skin_triangles(template_model()))
        self.assertEqual((REPO / TARGET).read_text(), skin_split_source(codes))

    def compare(self, case, frame=False, vertex_tolerance=1e-7):
        pieces, cover = self.expected[case]
        out = self.out[case]
        self.assertEqual([p["ref"] for p in out["pieces"]], [p.ref for p in pieces])
        R = np.array(PAGE_FRAME["rotation"]).reshape(3, 3)
        s, o = PAGE_FRAME["scale"], np.array(PAGE_FRAME["offset"])
        for k, (got, want) in enumerate(zip(out["pieces"], pieces)):
            with self.subTest(case=case, piece=want.ref):
                self.assertEqual(got["shown"], want.shown)
                self.assertEqual(
                    sorted(got["hidden_groups"]), sorted(want.hidden_groups)
                )
                V = self.page(case, k, "vertices.f64", np.float64).reshape(-1, 3)
                N = self.page(case, k, "normals.f64", np.float64).reshape(-1, 3)
                if frame:
                    V = ((V - o) / s) @ R
                    N = N @ R
                self.assertLess(np.abs(V - want.vertices).max(), vertex_tolerance)
                self.assertLess(np.abs(N - want.normals).max(), 1e-6)
                self.assertEqual(got["pushed"], want.pushed)
                W = self.page(case, k, "weights.f32", np.float32).reshape(-1, 4)
                J = self.page(case, k, "joints.u16", np.uint16).reshape(-1, 4)
                self.assertLess(np.abs(W - want.weights).max(), 1e-6)
                strong = want.weights > 1e-6
                np.testing.assert_array_equal(J[strong], want.joints[strong])
        page_cover = np.fromfile(self.dir / f"{case}_cover.u8", dtype=np.uint8) > 0
        np.testing.assert_array_equal(page_cover, cover[: self.n])

    def test_cases(self):
        for case in (
            "template",
            "thin",
            "random",
            "hide_cape",
            "swap",
            "no_push",
            "no_outfit",
            "override_swap",
        ):
            self.compare(case)

    def test_sliced(self):
        """The page runs the push-out in slices between frames: slices of 7 vertices give
        every vertex, normal and count of the dress at once."""
        for name, case in self.out.items():
            with self.subTest(case=name):
                self.assertEqual(case["sliced_difference"], 0)

    def test_override_swap(self):
        """A piece of another outfit that the card's overrides show takes its slot."""
        refs = [p["ref"] for p in self.out["override_swap"]["pieces"]]
        self.assertIn("work/smock", refs)
        self.assertNotIn("travel/tunic", refs)

    def test_no_outfit(self):
        """A card with an empty active_outfit wears none of its outfits in the page."""
        refs = [p["ref"] for p in self.out["no_outfit"]["pieces"]]
        self.assertEqual(refs, ["plate", "vest"])

    def test_page_frame(self):
        self.compare("page_frame", frame=True, vertex_tolerance=1e-6)

    def test_cases_move_and_hide(self):
        thin = {p["ref"]: p for p in self.out["thin"]["pieces"]}
        self.assertGreater(sum(p["pushed"] for p in thin.values()), 50)
        self.assertEqual(thin["travel/tunic"]["hidden_groups"], ["back"])
        hidden = {p["ref"]: p for p in self.out["hide_cape"]["pieces"]}
        self.assertNotIn("travel/cape", hidden)
        self.assertEqual(hidden["travel/tunic"]["hidden_groups"], [])
        swap = [p["ref"] for p in self.out["swap"]["pieces"]]
        self.assertIn("work/smock", swap)
        self.assertNotIn("travel/tunic", swap)
        self.assertGreater(int(self.expected["thin"][1].sum()), 300)

    def test_estimated_rigid_pose(self):
        """The page has no rest rotations: rigid pieces turn with their bone's skin. On a
        body far from the template the circlet stays within 4 mm of its place."""
        pieces, _ = self.expected["estimate"]
        out = self.out["estimate"]["pieces"]
        k = [p.ref for p in pieces].index("travel/circlet")
        V = self.page("estimate", k, "vertices.f64", np.float64).reshape(-1, 3)
        error = float(np.linalg.norm(V - pieces[k].vertices, axis=1).max())
        print(f"\nestimated rigid pose: {error * 1000:.2f} mm at most")
        self.assertEqual(out[k]["ref"], "travel/circlet")
        self.assertLess(error, 0.004)

    def test_times(self):
        for name, case in self.out.items():
            print(
                f"\n{name}: dress {case['ms']:.0f} ms, cover {case['cover_ms']:.0f} ms, {case['timing']}"
            )


if __name__ == "__main__":
    unittest.main()

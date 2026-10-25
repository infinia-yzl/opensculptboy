# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the outfit tools with synthetic garments (``test/wearable_fixtures.py``,
``test/mhclo_fixtures.py``): ``wearables catalogue`` (``opensculptboy.wearables.catalogue``:
the JSON entries, the stable order, the filters, the colours, the quality summary, assets not
yet imported, thumbnails and their cache), ``wearables check``
(``opensculptboy.wearables.check``: the report, the gates on a clean outfit and on planted
defects of skin and layers, the exit status, the review sheet) and outfit files
(``opensculptboy.wearables.outfitfile``: export from a card, the schema, a card made from a
file in another folder, repeated runs, dry runs, replacement and the coded errors).
"""

import contextlib
import io
import json
import pathlib
import tempfile
import unittest

import numpy as np

from opensculptboy import Character
from opensculptboy.wearables import Wearable, bind_surface, write_wearable
from opensculptboy.wearables.format import Garment, Material
from test.mhclo_fixtures import write_asset
from test.wearable_fixtures import (
    LICENCE,
    _region_shell,
    template_body,
    torso_mask,
    write_outfit,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(args) -> tuple[int, dict, str]:
    """Run the command; returns the exit status, the JSON result and the error stream."""
    from opensculptboy.cli import main

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            status = main([str(a) for a in args])
        except SystemExit as exit:
            status = exit.code
    return status, json.loads(out.getvalue()), err.getvalue()


def validate(data: dict, schema_path: pathlib.Path) -> None:
    try:
        import jsonschema
    except ImportError:  # the dev extra holds it
        return
    jsonschema.validate(data, json.loads(schema_path.read_text(encoding="utf-8")))


def shell(
    name, offset, layer, colour, covers=True, slot="torso", inside_out=False
) -> Wearable:
    """A shell of anny's torso ``offset`` metres off the skin (negative: under it), with the
    skin under it as its cover mask when ``covers``; ``inside_out`` winds its triangles
    toward the body."""
    body = template_body()
    vertices, triangles, rows = _region_shell(body, torso_mask(body), offset)
    if inside_out:
        triangles = triangles[:, ::-1]
    garment = Garment(vertices, triangles, materials=[Material(name, colour, 0.0, 0.9)])
    mask = np.zeros(len(body.vertices), bool)
    if covers:
        mask[rows] = True
        for _ in range(2):  # the border rings are not under the shell
            border = np.zeros(len(body.vertices), bool)
            outside = ~mask[body.triangles]
            border[body.triangles[outside.any(1)].ravel()] = True
            mask &= ~border
    return Wearable(
        name=name,
        slot=slot,
        layer=layer,
        garment=garment,
        binding=bind_surface(garment, body),
        licence=dict(LICENCE),
        covers=mask,
        tags=["synthetic", "shell"],
        description=f"a synthetic shell {offset * 1000:g} mm off the torso",
    )


def card_file(
    folder: pathlib.Path, pieces, name="card.json", outfit="test"
) -> pathlib.Path:
    card = Character(
        name="checked",
        outfits=[
            {
                "id": outfit,
                "label": "Test",
                "pieces": [
                    {"piece": p, "file": f"wear/{p}.wearable.json", "enabled": True}
                    for p in pieces
                ],
            }
        ],
        active_outfit=outfit,
    )
    path = folder / name
    path.write_text(json.dumps(card.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


class TestCatalogue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        write_outfit(cls.folder / "wear")
        (cls.folder / "wear" / "broken.wearable.json").write_text("{", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_entries_order_and_filters(self):
        status, result, _ = run(["wearables", "catalogue", self.folder])
        self.assertEqual(status, 0)
        self.assertTrue(result["ok"])
        names = [e["name"] for e in result["pieces"]]
        self.assertEqual(names, ["cape", "circlet", "tunic"])  # sorted by path
        self.assertEqual(result["skipped"][0]["path"], "wear/broken.wearable.json")
        self.assertEqual(result["skipped"][0]["code"], "invalid_package")
        tunic = result["pieces"][2]
        for key in (
            "path",
            "label",
            "slot",
            "layer",
            "tags",
            "description",
            "colours",
            "groups",
            "triangles",
            "licence",
            "quality",
            "thumbnail",
        ):
            self.assertIn(key, tunic)
        self.assertEqual(tunic["groups"], ["front", "back"])
        # linear (0.2, 0.3, 0.6) in sRGB
        self.assertEqual(tunic["colours"], [{"material": "tunic", "hex": "#7c95cb"}])
        self.assertEqual(tunic["licence"]["spdx"], "CC0-1.0")
        self.assertIn("CC0-1.0", tunic["licence"]["credit"])
        self.assertEqual(tunic["quality"]["errors"], 0)
        self.assertEqual(set(tunic["quality"]["codes"]), {"error", "warning", "info"})
        again = run(["wearables", "catalogue", self.folder])[1]
        self.assertEqual(again, result)
        for args, expected in (
            (["--slot", "cloak"], ["cape"]),
            (["--slot", "head,torso"], ["circlet", "tunic"]),
            (["--query", "tunics"], ["tunic"]),
            (["--query", "gold"], []),
            (["--tag", "none"], []),
        ):
            with self.subTest(args=args):
                got = run(["wearables", "catalogue", self.folder, *args])[1]
                self.assertEqual([e["name"] for e in got["pieces"]], expected)

    def test_errors(self):
        status, result, _ = run(["wearables", "catalogue", self.folder / "missing"])
        self.assertEqual((status, result["error"]["code"]), (1, "folder_not_found"))
        self.assertTrue(result["error"]["hint"])
        status, result, _ = run(["wearables", "catalogue", self.folder, "--slot", "x"])
        self.assertEqual(result["error"]["code"], "unknown_slot")
        status, result, _ = run(["wearables", "catalogue", self.folder, "--bogus"])
        self.assertEqual((status, result["error"]["code"]), (2, "usage"))

    def test_assets_not_yet_imported(self):
        from opensculptboy.wearables.mhclo import import_mhclo
        from test.wearable_fixtures import template_model

        packs = self.folder / "packs"
        shirt = write_asset(packs / "shirt", "shirt")
        write_asset(packs / "tights", "tights", helper=True, tags=("tights",))
        got = run(["wearables", "catalogue", self.folder, "--mhclo", packs])[1]
        assets = [e for e in got["pieces"] if e["kind"] == "mhclo"]
        self.assertEqual([e["name"] for e in assets], ["shirt", "tights"])
        self.assertEqual(assets[0]["slot"], "torso")
        self.assertFalse(assets[0]["imported"])
        self.assertEqual(
            assets[0]["import"][:3], ["opensculptboy", "wearables", "import-mhclo"]
        )
        self.assertEqual(assets[0]["colours"][0]["hex"][0], "#")
        wearable, _ = import_mhclo(
            shirt, template_body(), Character(), template_model()
        )
        write_wearable(self.folder / "imported" / "shirt.wearable.json", wearable)
        try:
            got = run(["wearables", "catalogue", self.folder, "--mhclo", packs])[1]
            assets = [e["name"] for e in got["pieces"] if e["kind"] == "mhclo"]
            self.assertEqual(assets, ["tights"])  # the shirt's uuid has a package
        finally:
            for f in (self.folder / "imported").iterdir():
                f.unlink()
            (self.folder / "imported").rmdir()

    def test_thumbnails_and_their_cache(self):
        from PIL import Image

        args = [
            "wearables",
            "catalogue",
            self.folder,
            "--slot",
            "torso",
            "--thumbnails",
        ]
        args += ["--thumbnail-size", "96"]
        status, first, _ = run(args)
        self.assertEqual(status, 0)
        self.assertEqual(first["thumbnails"], {"rendered": 1, "cached": 0})
        path = self.folder / first["pieces"][0]["thumbnail"]
        self.assertEqual(path.name, "tunic.thumb.png")
        image = Image.open(path).convert("RGB")
        self.assertEqual(image.size, (96, 96))
        pixels = np.asarray(image).reshape(-1, 3).astype(int)
        blue = (pixels[:, 2] > pixels[:, 0] + 40).mean()
        self.assertGreater(blue, 0.05)  # the tunic's colour shows
        second = run(args)[1]
        self.assertEqual(second["thumbnails"], {"rendered": 0, "cached": 1})
        plain = run(["wearables", "catalogue", self.folder, "--slot", "torso"])[1]
        self.assertEqual(plain["pieces"][0]["thumbnail"], "wear/tunic.thumb.png")


class TestCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        write_outfit(cls.folder / "wear")
        wear = cls.folder / "wear"
        write_wearable(
            wear / "tight.wearable.json", shell("tight", -0.004, 50, (0.6, 0.2, 0.2))
        )
        write_wearable(
            wear / "vest.wearable.json",
            shell("vest", 0.002, 60, (0.1, 0.5, 0.1), covers=False, slot="accessory"),
        )
        write_wearable(
            wear / "lining.wearable.json",
            shell("lining", 0.004, 50, (0.7, 0.7, 0.2), inside_out=True),
        )
        cls.clean = card_file(cls.folder, ["tunic", "cape", "circlet"], "clean.json")
        cls.lining = card_file(cls.folder, ["lining", "cape"], "lining.json")
        cls.skin = card_file(cls.folder, ["tight"], "skin.json")
        cls.layers = card_file(cls.folder, ["tunic", "vest"], "layers.json")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, card, *extra):
        args = ["wearables", "check", card, "--bodies", "default", "--clips", "wave"]
        return run(args + ["--frames", "2", *extra])

    def test_clean_outfit_passes(self):
        sheet = self.folder / "clean.png"
        status, report, _ = self.check(self.clean, "--sheet", sheet)
        self.assertEqual(status, 0, report["failed"])
        self.assertTrue(report["pass"])
        names = [g["name"] for g in report["gates"]]
        self.assertEqual(
            names,
            [
                "quality",
                "skin_rest",
                "layers_rest",
                "skin_tpose",
                "layers_tpose",
                "skin_clips",
                "layers_clips",
            ],
        )
        states = [(s["body"], s["state"], s["frame"]) for s in report["states"]]
        self.assertEqual(states[:2], [("default", "rest", 0), ("default", "tpose", 0)])
        self.assertEqual(len(states), 4)
        self.assertEqual(
            [p["ref"] for p in report["pieces"]],
            ["test/tunic", "test/cape", "test/circlet"],
        )
        rest = report["states"][0]
        self.assertGreater(rest["skin"]["covered"], 100)
        self.assertEqual(rest["skin"]["showing"], 0)
        from PIL import Image

        image = Image.open(sheet)
        self.assertEqual(image.size[0], 6 * 200)
        self.assertEqual(report["sheet"], str(sheet))

    def test_inside_out_piece_measures_outward(self):
        """A piece wound toward the body measures from the side away from the skin."""
        status, report, _ = self.check(self.lining, "--sheet", "none")
        self.assertEqual(status, 0, report["failed"])
        self.assertGreater(report["states"][0]["skin"]["covered"], 100)

    def test_planted_skin_defect_fails(self):
        status, report, _ = self.check(self.skin, "--sheet", "none", "--no-tpose")
        self.assertEqual(status, 1)
        self.assertFalse(report["pass"])
        self.assertIn("skin_rest", report["failed"])
        gate = next(g for g in report["gates"] if g["name"] == "skin_rest")
        self.assertGreater(gate["value"], 50)
        self.assertEqual(gate["worst"]["body"], "default")
        self.assertEqual(gate["worst"]["piece"], "test/tight")
        self.assertIn(gate["worst"]["region"], ("torso", "hips"))
        self.assertGreater(gate["worst"]["largest_mm"], 3.0)
        self.assertTrue(
            next(g for g in report["gates"] if g["name"] == "skin_tpose")["skipped"]
        )

    def test_planted_layer_defect_fails(self):
        status, report, _ = self.check(
            self.layers, "--sheet", "none", "--clips", "none"
        )
        self.assertEqual(status, 1)
        self.assertIn("layers_rest", report["failed"])
        gate = next(g for g in report["gates"] if g["name"] == "layers_rest")
        self.assertEqual(gate["worst"]["piece"], "test/tunic through test/vest")
        self.assertGreater(gate["worst"]["largest_mm"], 2.5)

    def test_errors_exit_with_two(self):
        status, report, _ = self.check(self.folder / "missing.json")
        self.assertEqual((status, report["error"]["code"]), (2, "card_not_found"))
        status, report, _ = self.check(self.clean, "--bodies", "thin")
        self.assertEqual((status, report["error"]["code"]), (2, "bad_option"))
        status, report, _ = self.check(self.clean, "--outfit", "nothing")
        self.assertEqual((status, report["error"]["code"]), (2, "unknown_outfit"))
        status, report, _ = run(["wearables", "check", self.clean, "--frames", "x"])
        self.assertEqual((status, report["error"]["code"]), (2, "usage"))

    def test_bodies(self):
        from opensculptboy.wearables.check import BODIES, region_of, test_bodies

        card = Character(phenotype={"weight": 0.7})
        bodies = test_bodies(card, BODIES)
        self.assertEqual([b.name for b in bodies], list(BODIES))
        self.assertEqual(bodies[2].character.phenotype, {"weight": 0.0})
        self.assertEqual(bodies[5].character.phenotype, {"weight": 0.7, "muscle": 1.0})
        self.assertEqual(
            test_bodies(card, ["age=0.25"])[0].character.phenotype["age"], 0.25
        )
        self.assertEqual(region_of("lowerarm02.L"), "forearm.L")
        self.assertEqual(region_of("spine03"), "torso")
        self.assertEqual(region_of("toe1-1.R"), "foot.R")


class TestOutfitFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        write_outfit(cls.folder / "shared" / "wear")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def source_card(self) -> pathlib.Path:
        card = card_file(
            self.folder / "shared", ["tunic", "cape", "circlet"], "ada.json", "travel"
        )
        data = json.loads(card.read_text(encoding="utf-8"))
        pieces = data["outfits"][0]["pieces"]
        pieces[0]["overrides"] = {"layer": 52}
        pieces[2]["hides"] = [{"piece": "tunic", "groups": ["front"]}]
        pieces[1]["springs"] = [{"name": "loose", "kind": "auto", "preset": "cape"}]
        data["wearable_overrides"] = {"travel/circlet": {"enabled": False}}
        card.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return card

    def test_export_and_add(self):
        from opensculptboy.wearables.outfitfile import SCHEMA_PATH
        from opensculptboy.wearables.wardrobe import worn_pieces

        card = self.source_card()
        out = self.folder / "shared" / "outfits" / "travel.outfit.json"
        args = [
            "wearables",
            "outfit",
            "export",
            card,
            "--outfit",
            "travel",
            "--out",
            out,
        ]
        status, dry, _ = run(args + ["--dry-run", "--tag", "travel"])
        self.assertEqual((status, dry["written"]), (0, False))
        self.assertFalse(out.exists())
        status, first, _ = run(args + ["--tag", "travel"])
        self.assertEqual((status, first["written"]), (0, True))
        data = json.loads(out.read_text(encoding="utf-8"))
        validate(data, SCHEMA_PATH)
        self.assertEqual(data["format"], "opensculptboy.outfit/1")
        self.assertEqual(data["pieces"][0]["file"], "../wear/tunic.wearable.json")
        self.assertEqual(data["pieces"][0]["overrides"], {"layer": 52})
        self.assertFalse(data["pieces"][2]["enabled"])  # the card's override
        self.assertEqual(
            data["hides"],
            [{"piece": "circlet", "target": "tunic", "groups": ["front"]}],
        )
        self.assertEqual(data["springs"]["cape"][0]["name"], "loose")
        self.assertEqual(data["tags"], ["travel"])
        self.assertTrue(data["credits"])
        status, again, _ = run(args + ["--tag", "travel"])
        self.assertEqual((status, again["changed"]), (0, False))
        status, clash, _ = run(args + ["--label", "Other"])
        self.assertEqual(clash["error"]["code"], "file_exists")
        self.assertEqual(run(args + ["--label", "Other", "--replace"])[0], 0)
        run(args + ["--tag", "travel", "--replace"])

        other = self.folder / "elsewhere" / "bo.json"
        other.parent.mkdir(exist_ok=True)
        other.write_text(
            json.dumps(Character(name="bo").to_dict()) + "\n", encoding="utf-8"
        )
        add = ["wearables", "add", other, out, "--activate"]
        status, result, _ = run(add + ["--dry-run"])
        self.assertEqual(
            (status, result["written"], result["changed"]), (0, False, True)
        )
        status, result, _ = run(add)
        self.assertEqual((status, result["written"]), (0, True))
        self.assertEqual(
            result["added"], ["travel/tunic", "travel/cape", "travel/circlet"]
        )
        card_data = json.loads(other.read_text(encoding="utf-8"))
        entry = card_data["outfits"][0]
        self.assertEqual(card_data["active_outfit"], "travel")
        self.assertEqual(entry["source"], "../shared/outfits/travel.outfit.json")
        self.assertEqual(
            entry["pieces"][0]["file"], "../shared/wear/tunic.wearable.json"
        )
        self.assertEqual(
            entry["pieces"][2]["hides"], [{"piece": "tunic", "groups": ["front"]}]
        )
        before = other.read_text(encoding="utf-8")
        status, result, _ = run(add)
        self.assertEqual((status, result["changed"]), (0, False))
        self.assertEqual(other.read_text(encoding="utf-8"), before)
        # the settings apply: the tunic takes layer 52, and an enabled circlet hides its front
        character = Character.from_dict(card_data)
        character.wearable_overrides = {"travel/circlet": {"enabled": True}}
        pieces = {p.ref: p for p in worn_pieces(character, other.parent)}
        self.assertEqual(pieces["travel/tunic"].wearable.layer, 52)
        # the cape hides the back and the circlet the front: the tunic shows no group
        self.assertEqual(set(pieces["travel/tunic"].hidden_groups), {"front", "back"})
        self.assertFalse(pieces["travel/tunic"].shown)
        self.assertEqual(pieces["travel/cape"].wearable.springs[0]["name"], "loose")
        # another outfit file of the same id
        data["pieces"] = data["pieces"][:2]
        del data["hides"]  # the circlet's rule goes with it
        smaller = self.folder / "shared" / "outfits" / "small.outfit.json"
        smaller.write_text(json.dumps(data), encoding="utf-8")
        status, result, _ = run(["wearables", "add", other, smaller])
        self.assertEqual(result["error"]["code"], "outfit_exists")
        status, result, _ = run(
            ["wearables", "add", other, smaller, "--outfit", "small"]
        )
        self.assertEqual(result["added"], ["small/tunic", "small/cape"])
        status, result, _ = run(["wearables", "add", other, smaller, "--replace"])
        self.assertEqual(result["removed"], ["travel/circlet"])

    def test_errors(self):
        folder = self.folder / "shared"
        card = folder / "plain.json"
        card.write_text(json.dumps(Character().to_dict()) + "\n", encoding="utf-8")
        good = {
            "format": "opensculptboy.outfit/1",
            "id": "x",
            "pieces": [{"piece": "tunic", "file": "wear/tunic.wearable.json"}],
        }
        cases = (
            ({**good, "format": "other"}, "invalid_outfit"),
            ({**good, "id": "a/b"}, "invalid_outfit"),
            ({**good, "colour": "red"}, "invalid_outfit"),
            ({**good, "pieces": []}, "invalid_outfit"),
            (
                {
                    **good,
                    "hides": [{"piece": "cape", "target": "tunic", "groups": ["x"]}],
                },
                "unknown_piece",
            ),
            (
                {
                    **good,
                    "hides": [{"piece": "tunic", "target": "tunic", "groups": ["x"]}],
                },
                "unknown_group",
            ),
            ({**good, "springs": {"cape": []}}, "unknown_piece"),
            (
                {
                    **good,
                    "pieces": [{"piece": "tunic", "file": "wear/none.wearable.json"}],
                },
                "package_not_found",
            ),
            (
                {
                    **good,
                    "pieces": [{**good["pieces"][0], "overrides": {"layer": 300}}],
                },
                "invalid_outfit",
            ),
        )
        for k, (data, code) in enumerate(cases):
            with self.subTest(code=code, case=k):
                path = folder / f"bad{k}.outfit.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                status, result, _ = run(["wearables", "add", card, path])
                self.assertEqual(status, 1)
                self.assertEqual(result["error"]["code"], code)
                self.assertTrue(result["error"]["hint"])
        status, result, _ = run(["wearables", "add", card, folder / "none.outfit.json"])
        self.assertEqual(result["error"]["code"], "outfit_not_found")
        status, result, _ = run(
            [
                "wearables",
                "outfit",
                "export",
                card,
                "--outfit",
                "x",
                "--out",
                folder / "x.outfit.json",
            ]
        )
        self.assertEqual(result["error"]["code"], "unknown_outfit")
        status, result, _ = run(
            [
                "wearables",
                "outfit",
                "export",
                card,
                "--outfit",
                "x",
                "--out",
                folder / "x.json",
            ]
        )
        self.assertEqual(result["error"]["code"], "bad_option")

    def test_private_outfit_never_goes_into_the_repository(self):
        from opensculptboy.wearables import read_wearable
        from opensculptboy.wearables.format import PRIVATE

        folder = self.folder / "private"
        write_outfit(folder / "wear")
        tunic = read_wearable(folder / "wear" / "tunic.wearable.json")
        tunic.provenance = PRIVATE
        write_wearable(folder / "wear" / "tunic.wearable.json", tunic)
        card = card_file(folder, ["tunic"], "card.json", "secret")
        out = ROOT / "test" / "data" / "never.outfit.json"
        status, result, _ = run(
            ["wearables", "outfit", "export", card, "--outfit", "secret", "--out", out]
        )
        self.assertEqual(result["error"]["code"], "private_in_repository")
        self.assertFalse(out.exists())
        status, result, _ = run(
            [
                "wearables",
                "outfit",
                "export",
                card,
                "--outfit",
                "secret",
                "--out",
                folder / "s.outfit.json",
            ]
        )
        self.assertEqual(status, 0)
        self.assertEqual(result["outfit"]["provenance"], "private")

    def test_schema_file_is_valid(self):
        from opensculptboy.wearables.outfitfile import SCHEMA_PATH

        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["properties"]["format"]["const"], "opensculptboy.outfit/1"
        )
        try:
            import jsonschema
        except ImportError:
            return
        jsonschema.Draft202012Validator.check_schema(schema)


if __name__ == "__main__":
    unittest.main()

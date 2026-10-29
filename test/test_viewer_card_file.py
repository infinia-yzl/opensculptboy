# OpenSculptBoy
# Apache License, Version 2.0
"""
The character as one file: the card's ``look`` block (schema 5) in Python
(opensculptboy.character.check_look), the round trip through JSON and through the scene extras of
a GLB, and the cards that the viewer page writes (viewer/src/card/file.ts, run in node by
viewer/test/card_file.mjs: Save card of the page's character and of an open card, the starters,
the .zip, the card in a GLB's extras), read back with ``Character.from_dict``.
"""

import json
import pathlib
import shutil
import struct
import subprocess
import tempfile
import unittest

from opensculptboy.character import SCHEMA_VERSION, Character, check_look

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "viewer" / "test" / "card_file.mjs"
LOOK = {
    "skin": {"tone": 0.35, "undertone": 0, "stone": "carrara", "polish": 0.45},
    "hair": {"style": "lob", "color": "#271f16", "length": 0.8, "part": "left"},
    "eyes": {"color": "#875f3d"},
}


def node_available():
    if shutil.which("node") is None:
        return False
    out = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout
    try:
        return int(out.strip().lstrip("v").split(".")[0]) >= 22
    except ValueError:
        return False


class TestLookBlock(unittest.TestCase):
    def test_schema(self):
        self.assertEqual(SCHEMA_VERSION, 5)
        plain = Character(name="a")
        self.assertEqual(plain.to_dict()["schema"], 1)
        self.assertNotIn("look", plain.to_dict())
        card = Character(name="a", look=LOOK).to_dict()
        self.assertEqual(card["schema"], 5)
        self.assertEqual(card["look"], LOOK)

    def test_round_trip(self):
        card = Character(name="a", phenotype={"muscle": 0.7}, look=LOOK)
        back = Character.from_dict(json.loads(json.dumps(card.to_dict())))
        self.assertEqual(back, card)
        self.assertEqual(back.to_dict(), card.to_dict())

    def test_schema_4_has_no_look(self):
        with self.assertRaisesRegex(ValueError, "schema 5"):
            Character.from_dict({"schema": 4, "name": "a", "look": LOOK})

    def test_bad_looks(self):
        for bad, words in (
            ([], "JSON object"),
            ({"face": {}}, "Unknown look sections"),
            ({"skin": {"tone": 2}}, "look.skin.tone"),
            ({"skin": {"tone": True}}, "look.skin.tone"),
            ({"skin": {"stone": "marble"}}, "look.skin.stone"),
            ({"skin": {"glow": 1}}, "Unknown keys"),
            ({"hair": {"color": "brown"}}, "look.hair.color"),
            ({"hair": {"part": "middle"}}, "look.hair.part"),
            ({"hair": {"style": ""}}, "look.hair.style"),
            ({"eyes": "blue"}, "look.eyes"),
        ):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, words):
                    check_look(bad)
        check_look({})
        check_look({"skin": {"stone": "#AABBCC"}})

    def test_glb_extras(self):
        """an export keeps the look in its scene extras, where the page reads the card"""
        from opensculptboy.export.gltf import export_glb

        card = Character(name="statue", look=LOOK)
        folder = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        path = folder / "statue.glb"
        export_glb(path, card, morph_targets=())
        data = path.read_bytes()
        length = struct.unpack_from("<I", data, 12)[0]
        doc = json.loads(data[20 : 20 + length])
        kept = doc["scenes"][doc.get("scene", 0)]["extras"]["opensculptboy"][
            "character"
        ]
        self.assertEqual(kept["look"], LOOK)
        self.assertEqual(Character.from_dict(kept), card)


@unittest.skipUnless(
    node_available(), "node 22 or later is needed to run the page's TypeScript"
)
class TestPageCards(unittest.TestCase):
    """the cards that the page writes read as cards in Python"""

    @classmethod
    def setUpClass(cls):
        cls.out = pathlib.Path(tempfile.mkdtemp())
        r = subprocess.run(
            ["node", str(SCRIPT), str(cls.out)],
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        if r.returncode:
            raise AssertionError(r.stderr[-4000:])
        cls.summary = json.loads(r.stdout.strip().splitlines()[-1])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def test_node(self):
        self.assertTrue(self.summary["ok"])

    def test_cards(self):
        for name in ("card.json", "card_open.json"):
            with self.subTest(name=name):
                data = json.loads((self.out / name).read_text(encoding="utf-8"))
                card = Character.from_dict(data)
                self.assertEqual(card.to_dict()["schema"], 5)
                self.assertEqual(card.look["skin"]["stone"], "carrara")
                written = card.to_dict()
                for key, value in data.items():
                    self.assertEqual(written[key], value, key)

    def test_starters(self):
        starters = json.loads((self.out / "starters.json").read_text(encoding="utf-8"))
        self.assertEqual(len(starters), 6)
        for data in starters:
            with self.subTest(name=data["name"]):
                card = Character.from_dict(data)
                self.assertIn(
                    card.look["skin"]["stone"],
                    (
                        "carrara",
                        "travertine",
                        "bardiglio",
                        "basalt",
                        "bronze",
                        "verdigris",
                    ),
                )

    def test_zip(self):
        import zipfile

        with zipfile.ZipFile(self.out / "stored.zip") as z:
            self.assertEqual(
                z.namelist(),
                [
                    "character.json",
                    "wear/cape.wearable.json",
                    "card.detail0.safetensors",
                ],
            )
            self.assertIsNone(z.testzip())
            Character.from_dict(json.loads(z.read("character.json")))


if __name__ == "__main__":
    unittest.main()

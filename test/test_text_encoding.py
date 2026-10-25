# OpenSculptBoy
# Apache License, Version 2.0
"""
Every text read and write in ``src/opensculptboy`` names its encoding, so that cards, packages
and reports read and write the same bytes under any locale (Windows defaults to a code page
such as cp1252). ``TestTextEncoding`` scans the sources with ``ast``; ``TestNonAsciiCard``
writes and reads a card and a package with non-ASCII labels in a child process whose locale
encoding is ASCII.
"""

import ast
import os
import pathlib
import subprocess
import sys
import textwrap
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCES = ROOT / "src" / "opensculptboy"

# Modules whose ``open`` is binary or takes no encoding.
BINARY_OPENERS = {"Image", "os", "zipfile", "tarfile", "webbrowser", "wave"}
# Modules whose ``open`` is binary unless the mode holds "t".
COMPRESSED_OPENERS = {"gzip", "bz2", "lzma"}
NUMPY_TEXT = {"loadtxt", "savetxt", "genfromtxt", "fromregex"}
TEMPORARY_FILES = {"NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"}
SUBPROCESS = {"run", "check_output", "Popen", "call", "check_call"}


def _argument(call: ast.Call, name: str, index: int | None):
    """The node of the argument ``name`` (or at ``index``), or None."""
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    if index is not None and len(call.args) > index:
        return call.args[index]
    return None


def _is_text_mode(call: ast.Call, index: int, default: str) -> bool:
    """Whether the mode at ``index`` (or the ``mode`` keyword) opens a file as text; a mode
    that is not a literal counts as text."""
    node = _argument(call, "mode", index)
    if node is None:
        return "b" not in default
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return "b" not in node.value
    return True


def _receiver(func: ast.AST) -> str | None:
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id
    return None


def text_io_without_encoding(source: str) -> list[tuple[int, str]]:
    """The calls of ``source`` that read or write text without naming an encoding, as
    ``(line, call)``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if any(
            k.arg is None for k in node.keywords
        ):  # **kwargs: the encoding may be there
            continue
        func = node.func
        name = (
            func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        )
        receiver = _receiver(func)
        flagged = False
        if name == "read_text":
            flagged = _argument(node, "encoding", 0) is None
        elif name == "write_text":
            flagged = _argument(node, "encoding", 1) is None
        elif name == "open":
            if receiver in BINARY_OPENERS:
                continue
            if receiver in COMPRESSED_OPENERS:
                text = _is_text_mode(node, 1, "rb")
                encoding = _argument(node, "encoding", 3)
            elif isinstance(func, ast.Name) or receiver in ("io", "builtins"):
                text = _is_text_mode(node, 1, "r")
                encoding = _argument(node, "encoding", 3)
            else:  # pathlib.Path.open
                text = _is_text_mode(node, 0, "r")
                encoding = _argument(node, "encoding", 2)
            flagged = text and encoding is None
        elif name in TEMPORARY_FILES:
            flagged = (
                _is_text_mode(node, 0, "w+b") and _argument(node, "encoding", 3) is None
            )
        elif name == "TextIOWrapper":
            flagged = _argument(node, "encoding", 1) is None
        elif name in NUMPY_TEXT and receiver in ("np", "numpy"):
            flagged = _argument(node, "encoding", None) is None
        elif name in SUBPROCESS and receiver == "subprocess":
            text = any(
                k.arg in ("text", "universal_newlines")
                and not (isinstance(k.value, ast.Constant) and not k.value.value)
                for k in node.keywords
            )
            flagged = text and _argument(node, "encoding", None) is None
        if flagged:
            found.append((node.lineno, ast.unparse(node.func)))
    return found


class TestTextEncoding(unittest.TestCase):
    def test_scanner(self):
        """The scanner finds text I/O without an encoding and passes binary and encoded I/O."""
        bad = """
            p.read_text()
            p.write_text(s)
            open(p)
            open(p, "w")
            p.open()
            p.open(mode="a")
            gzip.open(p, "rt")
            tempfile.NamedTemporaryFile("w")
            io.TextIOWrapper(f)
            np.savetxt(p, a)
            subprocess.run(c, text=True)
            open(p, mode)
        """
        good = """
            p.read_text(encoding="utf-8")
            p.read_text("utf-8")
            p.write_text(s, encoding="utf-8")
            p.write_text(s, "utf-8")
            open(p, "rb")
            open(p, "w", encoding="utf-8")
            open(p, "r", -1, "utf-8")
            p.open("wb")
            p.open("r", encoding="utf-8")
            Image.open(p)
            gzip.open(p)
            tempfile.NamedTemporaryFile()
            subprocess.run(c, capture_output=True)
            subprocess.run(c, text=True, encoding="utf-8")
            safe_open(p, framework="numpy")
            p.read_bytes()
        """
        found = text_io_without_encoding(textwrap.dedent(bad))
        lines = [line for line in textwrap.dedent(bad).splitlines() if line.strip()]
        self.assertEqual(
            [line for line, _ in found], list(range(2, 2 + len(lines))), found
        )
        self.assertEqual(text_io_without_encoding(textwrap.dedent(good)), [])

    def test_sources_name_the_encoding(self):
        """Every text read and write of ``src/opensculptboy`` names its encoding."""
        offenders = []
        for path in sorted(SOURCES.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            for line, call in text_io_without_encoding(source):
                offenders.append(f"{path.relative_to(ROOT)}:{line}: {call}")
        self.assertEqual(
            offenders,
            [],
            "Name the encoding (encoding='utf-8') of these text reads and writes.",
        )


CHILD = r"""
import json, locale, pathlib, sys, tempfile
import numpy as np

encoding = locale.getpreferredencoding(False)
if "utf" in encoding.lower().replace("-", ""):
    print("SKIP", encoding)
    sys.exit(0)

from opensculptboy.cli import _load_character
from opensculptboy.character import Character
from opensculptboy.wearables.format import (
    Binding, Garment, Wearable, read_wearable, write_wearable,
)

LABEL = "Caf" + chr(0xE9) + " " + chr(0x2014) + " " + chr(0x65C5) + chr(0x306E) + chr(0x670D)
folder = pathlib.Path(tempfile.mkdtemp())

# A card that an editor saved as UTF-8 with the label as it reads.
card = Character(outfits=[{"id": "travel", "label": LABEL, "pieces": []}], active_outfit="travel")
path = folder / "card.json"
path.write_bytes(json.dumps(card.to_dict(), ensure_ascii=False).encode("utf-8"))
assert _load_character(str(path)).outfits[0]["label"] == LABEL

# A package written by the library, then edited by hand as UTF-8.
vertices = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
garment = Garment(vertices, np.array([[0, 1, 2]]))
binding = Binding(
    method="rigid", body="anny-template", body_vertices=np.zeros((1, 3)), bone="head",
    local=vertices, normal=garment.normals,
)
wearable = Wearable(
    name="ring", slot="accessory", layer=60, garment=garment, binding=binding,
    weights="rigid", label=LABEL,
    licence={"spdx": "CC0-1.0", "author": LABEL, "design_source": "test"},
)
described = write_wearable(folder / "ring", wearable)
assert read_wearable(described).label == LABEL
data = json.loads(described.read_bytes())
data["licence"]["source"] = LABEL
described.write_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"))
again = read_wearable(described)
assert again.label == LABEL and again.licence["source"] == LABEL, again.licence
print("OK", encoding)
"""


class TestNonAsciiCard(unittest.TestCase):
    def test_round_trip_under_an_ascii_locale(self):
        """A card and a package with the label "Café — 旅の服" round-trip in a process whose
        locale encoding is ASCII (C locale, no UTF-8 mode, no locale coercion)."""
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("LC_", "PYTHONIOENCODING", "PYTHONUTF8"))
        }
        env.update(
            LC_ALL="C",
            LANG="C",
            PYTHONCOERCECLOCALE="0",
            PYTHONUTF8="0",
            PYTHONPATH=os.pathsep.join(
                [str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")]
            ),
        )
        result = subprocess.run(
            [sys.executable, "-X", "utf8=0", "-c", CHILD],
            capture_output=True,
            env=env,
            cwd=ROOT,
            timeout=600,
        )
        out = result.stdout.decode("utf-8", "replace")
        err = result.stderr.decode("utf-8", "replace")
        self.assertEqual(result.returncode, 0, err[-3000:])
        if out.startswith("SKIP"):
            self.skipTest(f"the locale encoding stays UTF-8 here ({out.strip()})")
        self.assertTrue(out.startswith("OK"), out + err)


if __name__ == "__main__":
    unittest.main()

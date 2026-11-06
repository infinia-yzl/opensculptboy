# OpenSculptBoy
# Apache License, Version 2.0
"""
Render a portrait pack with the web viewer's page in headless Chromium (Playwright).

The frames show the model as the viewer shows it: its textures, skin, eyes, lashes, teeth, hair
and outfit. The page's ``window.portrait`` entry points (``viewer/src/main.ts``) hold the scene
still with the hair at rest, light it with the page's lights, frame a bust of the figure seen from
the front, and draw each face with a transparent background and straight alpha. The camera, the
lights and the pose stay the same for every frame, so only the face changes from one frame to the
next.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import pathlib
import re
import time

import numpy as np

from opensculptboy.portrait_pack.contract import (
    EXPRESSIONS,
    MANIFEST,
    MAX_SIDE,
    REQUIRED_EXPRESSION,
    SCHEMA,
    TEXT_LIMITS,
    check_pack,
    frame_name,
    is_frame_name,
    text_problem,
)
from opensculptboy.portrait_pack.faces import pack_faces

DEFAULT_SIZE = (768, 1024)
# The accumulated passes of each frame. A browser with a graphics card draws a pass in a few
# milliseconds; one that draws in software (SwiftShader) takes seconds for each pass.
DEFAULT_SAMPLES = 16
# The built page in a clone of the repository.
REPO_PAGE = (
    pathlib.Path(__file__).resolve().parents[3] / "viewer" / "dist" / "anny_viewer.html"
)
# Chromium falls back to drawing WebGL in software on a computer without a graphics card.
CHROMIUM_ARGS = ["--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
# The environment variable that names a Chromium executable for Playwright to launch.
CHROMIUM_ENV = "OPENSCULPTBOY_CHROMIUM"

# The licence of the default character's pictures: the licences of the model data.
DEFAULT_LICENCE = (
    "The OpenSculptBoy default character: Anny body model and OpenSculptBoy code and data under "
    "the Apache License 2.0; MakeHuman, MPFB2 and Face Units assets under CC0 1.0; ICT-FaceKit "
    "face shapes under the MIT License."
)
PAGE_TIMEOUT_MS = 15 * 60 * 1000
FILE_EXT = re.compile(
    r"\.(safetensors|png|jpe?g|webp|ktx2|json|glb|bin)$", re.IGNORECASE
)


class PortraitError(RuntimeError):
    """The page could not render the pack."""


def parse_size(text: str) -> tuple[int, int]:
    """``"WxH"`` as (width, height), each side 1 to 1024 pixels."""
    m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", text or "")
    if not m:
        raise ValueError(f"The size {text!r} is not WIDTHxHEIGHT, such as 768x1024.")
    w, h = int(m.group(1)), int(m.group(2))
    if not (1 <= w <= MAX_SIDE and 1 <= h <= MAX_SIDE):
        raise ValueError(
            f"The size {w}x{h} breaks the limit: each side takes 1 to {MAX_SIDE} pixels."
        )
    return w, h


def parse_expressions(text: str | None) -> list[str]:
    """A comma-separated list of expressions, in the order of the contract with ``neutral``
    first (``neutral`` is added when the list leaves it out); None gives every expression."""
    if text is None:
        return list(EXPRESSIONS)
    names = [n.strip() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in EXPRESSIONS]
    if unknown:
        raise ValueError(
            f"Unknown expressions {unknown}; the expressions are {list(EXPRESSIONS)}."
        )
    want = set(names) | {REQUIRED_EXPRESSION}
    return [x for x in EXPRESSIONS if x in want]


def card_files(card: pathlib.Path) -> list[pathlib.Path]:
    """
    The card and every file it names, as the page's Open path takes them: the files that the
    card names (its detail levels, its maps and parts, its wearable packages), relative to the
    card, and the files that each named JSON file names in turn, relative to that file.
    """
    card = pathlib.Path(card).resolve()
    seen: dict[pathlib.Path, None] = {card: None}
    todo = [card]
    missing = []
    while todo:
        path = todo.pop()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for name in _named_files(data):
            f = (path.parent / name).resolve()
            if f in seen:
                continue
            if not f.is_file():
                missing.append(name)
                continue
            seen[f] = None
            if f.suffix.lower() == ".json":
                todo.append(f)
    if missing:
        raise FileNotFoundError(
            f"The card names files that are missing beside it: {', '.join(sorted(set(missing))[:5])}."
        )
    return list(seen)


def _named_files(value, key: str = "") -> list[str]:
    out = []
    if isinstance(value, str):
        if (
            key not in ("name", "label")
            and FILE_EXT.search(value)
            and not re.match(r"^https?:", value)
        ):
            out.append(value)
    elif isinstance(value, list):
        for v in value:
            out.extend(_named_files(v, key))
    elif isinstance(value, dict):
        for k, v in value.items():
            if k != "look":
                out.extend(_named_files(v, k))
    return out


class PortraitPage:
    """
    The viewer's page in headless Chromium, holding one character for its portraits:

        with PortraitPage(card=None) as page:
            info = page.begin(768, 1024)          # the framing, the anchor and the actions
            rgba = page.frame({"jawOpen": 0.5})   # (1024, 768, 4) uint8, straight alpha
    """

    def __init__(self, card=None, page=None, log=None):
        self.card = pathlib.Path(card) if card else None
        self.page_path = pathlib.Path(page) if page else REPO_PAGE
        self.log = log or (lambda m: None)
        self._stack = contextlib.ExitStack()
        self.tab = None
        self.errors: list[str] = []

    def __enter__(self):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as error:
            raise PortraitError(
                "The portrait pack needs Playwright and its Chromium: install the 'portrait' extra "
                "(pip install 'opensculptboy[portrait]') and run 'playwright install chromium'."
            ) from error
        if not self.page_path.is_file():
            raise FileNotFoundError(
                f"The viewer's page {self.page_path} is missing: build it (opensculptboy viewer build) "
                "or name the built page with --page."
            )
        p = self._stack.enter_context(sync_playwright())
        kw = {}
        if os.environ.get(CHROMIUM_ENV):
            kw["executable_path"] = os.environ[CHROMIUM_ENV]
        browser = p.chromium.launch(args=CHROMIUM_ARGS, **kw)
        self._stack.callback(browser.close)
        self.tab = browser.new_context(viewport=dict(width=640, height=480)).new_page()
        self.tab.set_default_timeout(PAGE_TIMEOUT_MS)
        self.tab.on("pageerror", lambda e: self.errors.append(str(e)))
        t = time.time()
        self.tab.goto(self.page_path.resolve().as_uri() + "?shot&base&acc=1")
        self.tab.wait_for_function(
            "() => window.__ready === true", timeout=PAGE_TIMEOUT_MS
        )
        self.log(f"the page is ready ({time.time() - t:.0f} s)")
        if self.card:
            self._open_card()
        return self

    def __exit__(self, *exc):
        self._stack.close()
        return False

    def _open_card(self):
        files = card_files(self.card)
        name = (
            json.loads(self.card.read_text(encoding="utf-8")).get("name") or "character"
        )
        t = time.time()
        self.tab.locator("#open-files").set_input_files([str(f) for f in files])
        state = self.tab.wait_for_function(
            """(name) => {
              const u = window.__shell && window.__shell.ui.get();
              if (!u) return false;
              if (u.openError) return { error: u.openError };
              const w = window.wearStats && window.wearStats();
              return u.openBusy === '' && !u.progress && w && w.card === name ? { ok: true } : false;
            }""",
            arg=name,
            timeout=PAGE_TIMEOUT_MS,
        ).json_value()
        if "error" in state:
            e = state["error"]
            raise PortraitError(
                f"The page could not open the card: {e.get('title', '')}. {e.get('body', '')}".strip()
            )
        self.log(f"the page opened the card {name!r} ({time.time() - t:.0f} s)")

    def begin(self, width: int, height: int) -> dict:
        """Hold the scene for frames of width by height pixels; returns the page's framing
        (``view``), the ``anchor`` and the facial ``actions`` it knows."""
        try:
            info = self.tab.evaluate(
                "([w, h]) => window.portrait.begin(w, h)", [width, height]
            )
        except Exception as error:  # noqa: BLE001 - the page's own message
            raise PortraitError(
                f"The page could not hold the scene: {error}"
            ) from error
        self.size = (width, height)
        return info

    def frame(self, actions: dict, samples: int = DEFAULT_SAMPLES) -> np.ndarray:
        """One frame of these facial actions as an (H, W, 4) uint8 array with straight alpha;
        the colour of fully transparent pixels is 0."""
        b64 = self.tab.evaluate(
            "([a, s]) => window.portrait.frame(a, s)", [actions, samples]
        )
        width, height = self.size
        rgba = (
            np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
            .reshape(height, width, 4)
            .copy()
        )
        rgba[rgba[..., 3] == 0] = 0
        return rgba

    def end(self) -> None:
        """Give the page its own view again."""
        self.tab.evaluate("() => window.portrait.end()")


def _previous_pack(folder: pathlib.Path) -> bool:
    """Whether a folder holds a pack (or the frames of an unfinished one) and nothing else."""
    entries = list(folder.iterdir())
    for p in entries:
        if not p.is_file() or not (p.name == MANIFEST or is_frame_name(p.name)):
            return False
    if (folder / MANIFEST).is_file():
        try:
            return (
                json.loads((folder / MANIFEST).read_text(encoding="utf-8")).get(
                    "schema"
                )
                == SCHEMA
            )
        except (OSError, ValueError, AttributeError):
            return False
    return True


def prepare_folder(out) -> pathlib.Path:
    """The pack's folder, emptied of a previous pack; refuses a folder that holds anything else."""
    out = pathlib.Path(out)
    if out.exists() and not out.is_dir():
        raise ValueError(f"{out} is a file; name a folder for the pack.")
    if out.is_dir() and any(out.iterdir()):
        if not _previous_pack(out):
            raise ValueError(
                f"{out} holds files that are not a portrait pack; name an empty folder or a previous pack."
            )
        for p in out.iterdir():
            p.unlink()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("opensculptboy")
    except Exception:  # noqa: BLE001 - a source tree without the distribution's metadata
        return "unknown"


def _one_line(text: str, limit: int) -> str:
    return " ".join(str(text).split())[:limit]


def provenance(card_name: str | None) -> str:
    """The manifest's provenance: the model, its source, and how the frames were rendered."""
    who = (
        f"The OpenSculptBoy character {card_name!r}"
        if card_name
        else "The OpenSculptBoy default character"
    )
    return (
        f"{who}, built on the Anny body model, rendered as the model is by the OpenSculptBoy web "
        "viewer in headless Chromium: its own textures, skin, eyes, hair and outfit, with one "
        "fixed camera, light and pose, and the face set by Anny's ARKit facial actions."
    )


def save_frame(path: pathlib.Path, rgba: np.ndarray) -> None:
    """One frame as an 8-bit RGBA PNG (colour type 6) without interlacing, compressed losslessly
    at the highest level."""
    from PIL import Image

    Image.fromarray(rgba, "RGBA").save(
        path, format="PNG", optimize=True, compress_level=9
    )


def write_pack(
    out,
    card=None,
    size=DEFAULT_SIZE,
    expressions=None,
    licence: str | None = None,
    companion: str | None = None,
    samples: int = DEFAULT_SAMPLES,
    page=None,
    log=None,
) -> dict:
    """
    Render the portrait pack of a character card (a path; None for the default character) into
    the folder ``out`` and check it. ``licence`` is required for a card: cards hold no licence
    of the pictures. ``companion`` defaults to the card's name. Returns the manifest, the check's
    report and the time taken.
    """
    log = log or (lambda m: None)
    width, height = size
    expressions = list(expressions or EXPRESSIONS)
    if expressions[0] != REQUIRED_EXPRESSION:
        raise ValueError(
            f"The expression {REQUIRED_EXPRESSION!r} is required, and comes first."
        )
    card_name = None
    if card is not None:
        data = json.loads(pathlib.Path(card).read_text(encoding="utf-8"))
        card_name = (
            data.get("name")
            if data.get("name") not in (None, "", "character")
            else None
        )
    if licence is None:
        if card is not None:
            raise ValueError(
                "Cards hold no licence of their pictures: give the licence of this character's "
                "images with --licence."
            )
        licence = DEFAULT_LICENCE
    if companion is None and card_name:
        companion = _one_line(card_name, TEXT_LIMITS["companion"])
    for key, value in (("licence", licence), ("companion", companion)):
        if value is None:
            continue
        why = text_problem(value)
        if why or not 1 <= len(value) <= TEXT_LIMITS[key]:
            raise ValueError(
                f"The {key} {why or 'is too long'}: it takes 1 to {TEXT_LIMITS[key]} characters on one line."
            )
    folder = prepare_folder(out)
    faces = pack_faces(expressions)
    t0 = time.time()
    with PortraitPage(card=card, page=page, log=log) as pp:
        info = pp.begin(width, height)
        known = set(info["actions"])
        unknown = sorted({k for *_, a in faces for k in a} - known)
        if unknown:
            raise PortraitError(f"The page does not know the facial actions {unknown}.")
        for i, (x, m, e, actions) in enumerate(faces):
            t = time.time()
            save_frame(folder / frame_name(x, m, e), pp.frame(actions, samples))
            log(f"{i + 1}/{len(faces)} {frame_name(x, m, e)} ({time.time() - t:.1f} s)")
        pp.end()
        if pp.errors:
            raise PortraitError(f"The page reported errors: {pp.errors[:3]}")
    manifest = {"schema": SCHEMA}
    if companion:
        manifest["companion"] = companion
    manifest.update(
        frame={"width": width, "height": height},
        anchor={"x": round(info["anchor"]["x"], 2), "y": round(info["anchor"]["y"], 2)},
        baseline=height,
        expressions=expressions,
        generator={"name": "OpenSculptBoy", "version": _version()},
        licence=licence,
        provenance=provenance(card_name),
    )
    (folder / MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = check_pack(folder)
    return {
        "manifest": manifest,
        "check": report,
        "seconds": round(time.time() - t0, 1),
        "samples": samples,
    }

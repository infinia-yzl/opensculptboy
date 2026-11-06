# Portrait pack

A portrait pack is a folder of still frames of one humanoid 3D character, rendered by the web viewer as the model is, for an app that shows the character as a 2D portrait that talks and blinks, as visual novels do. The app swaps the frames in place: the mouth opens and closes while the text types, the eyes blink, and the expression changes with each message. Still frames keep the graphics card free for other work, and the frames come from the character's own model with its textures, skin, eyes, lashes, teeth, hair and outfit.

```bash
uv sync --extra portrait                                     # Playwright for the export
uv run playwright install chromium                           # its Chromium, once
opensculptboy portrait-pack --out pack/                      # the default character: 90 frames and manifest.json
opensculptboy portrait-pack --character ada.json --out pack/ --licence "CC-BY-4.0, by the card's author"
opensculptboy portrait-pack --character ada.json --out pack/ --licence CC0-1.0 --expressions smile,sad --size 600x800
opensculptboy portrait-pack --check pack/                    # check a pack against the contract (exit 0 pass, 1 fail, 2 error)
```

```python
from opensculptboy.portrait_pack import check_pack, write_pack

result = write_pack("pack", card="ada.json", licence="CC0-1.0")   # the manifest, the check's report and the time
report = check_pack("pack")                                      # {"pass", "problems": [{"code", "message", "file"}], ...}
```

The command options:

| Option | Default | What it sets |
|---|---|---|
| `--character CARD` | the default character | the character card; its files lie beside it, as the viewer's Open path takes them |
| `--out FOLDER` | | the pack's folder: new, empty or a previous pack, whose files the export replaces; a folder with other files is refused |
| `--size WxH` | `768x1024` | the frame size, each side up to 1024 pixels |
| `--expressions LIST` | all ten | the expressions to render, comma-separated; `neutral` is always in |
| `--licence TEXT` | the default character's own | the licence of the images: a card holds no licence of its pictures, so a card needs it |
| `--companion NAME` | the card's name | the character's name in the manifest |
| `--samples N` | 16 | the accumulated passes of each frame |
| `--page FILE` | `viewer/dist/anny_viewer.html` of the repository | the built viewer page |
| `--json` | | the result as JSON on stdout |
| `--check FOLDER` | | check a pack and write nothing |

## How the frames are made

The export opens the viewer's page in headless Chromium through Playwright, opens the card through the page's Open path, and drives the page's `window.portrait` entry points (`viewer/src/main.ts`). These render through the machinery of the character sheet:

- The scene holds still: the hair rests without its simulation, so its outline stays the same in every frame, the blink timer stops and the pose stays the card's own (Anny's rest pose for a card without one).
- The page's own lights and exposure light the figure, and one fixed camera, the sheet's long lens, looks at it from the front.
- The framing is a bust: the frame's top lies 0.3 heads above the crown and the frame is 2.5 heads high, a head being the height from the chin to the crown, so the bottom edge cuts the figure across the upper chest. The framing is the same for every frame of a pack.
- Each frame accumulates `--samples` passes with a transparent background. The display pass divides the colour by the coverage, so the PNG files hold straight alpha, and every pixel outside the figure has alpha 0.
- The frames are 8-bit RGBA PNG files (colour type 6) without interlacing, compressed losslessly at the highest level.

Chromium draws on the graphics card when there is one: the export launches it with the graphics card's arguments for the platform (ANGLE on Direct3D 11 on Windows, on Metal on macOS, the default GL elsewhere), reads the WebGL renderer, and launches it again in software (SwiftShader) when WebGL 2 is missing. The log names the renderer that the run used, and the JSON result holds it as `renderer`. `OPENSCULPTBOY_CHROMIUM_ARGS` sets Chromium's arguments (separated by spaces) in place of both, and `OPENSCULPTBOY_CHROMIUM` names a Chromium executable.

On a graphics card a frame takes about a second, and a full pack takes under two minutes. In software each pass takes about 40 seconds at 768 by 1024 pixels: a full pack takes about 16 hours at 16 samples and about 4 hours at 4. `--samples 4` keeps the hair smooth; one pass leaves the hair grainy.

## The faces

Each frame is a mix of Anny's 52 ARKit facial actions. The page plays them as its Express mode does, `jawOpen` as the turn of the jaw. An expression mixes the VRM 1.0 presets that the VRM export writes (`src/opensculptboy/data/vrm/expressions.json`; `lookUp` and `lookDown` as the eye-look actions), scaled by its weights, with facial actions of its own on top. Then the frame's mouth adds its vowels and the frame's eyes add `eyeBlinkLeft` and `eyeBlinkRight`. Since the export sets the actions itself, no preset blocks the blink or the vowels, and each of the nine frames of an expression shows its own mouth and eyes. Every action is clamped to the range from 0 to 1.

| Expression | Presets | Actions on top | Lids of the open eyes |
|---|---|---|---|
| `neutral` | none | | |
| `smile` | `happy` 0.6 | | |
| `joy` | `happy` 1.0 | | blink 0.25 (the eyes close a little in the smile) |
| `thinking` | `relaxed` 0.4, `lookUp` 0.35 | `mouthPressLeft` and `mouthPressRight` 0.2 | |
| `surprised` | `surprised` 1.0 | | |
| `worried` | `sad` 0.45 | `browInnerUp` 0.45, `eyeWideLeft` and `eyeWideRight` 0.15, `mouthPressLeft` and `mouthPressRight` 0.25 | |
| `sad` | `sad` 0.9 | | |
| `stern` | `angry` 0.6 | | |
| `embarrassed` | `happy` 0.35, `lookDown` 0.3 | `browInnerUp` 0.2 | |
| `drowsy` | `relaxed` 0.8 | | blink 0.15 (heavy lids) |

| Mouth | Vowels |
|---|---|
| `closed` | none |
| `half` | `aa` 0.4, `ih` 0.2 |
| `open` | `aa` 0.9, `oh` 0.2 |

The eyes start from the expression's own lids: the blink that its presets hold (`sad` and `relaxed` hold some) and the lids of the table. `open` keeps them, `half` closes them half way to a full blink, and `closed` closes them fully, so the three states differ for every expression. As the lids close, the blink takes over the widened eyes of `surprised` and `worried`. Anny has no blush, so `embarrassed` uses the presets and the brows alone. `opensculptboy.portrait_pack.faces` holds the mixes (`EXPRESSION_MIXES`, `MOUTH_MIXES`, `EYE_CLOSE`), and `frame_actions(expression, mouth, eyes)` gives the actions of one frame.

## The contract

The pack follows the contract `squire.portraits/1` of the consuming app, and `opensculptboy portrait-pack --check` (`opensculptboy.portrait_pack.check_pack`) applies every rule, so the export checks its own output.

- A pack is one folder that holds exactly `manifest.json` and the frames. It has no other files and no subfolders, and it is never a `.zip` file.
- Each frame is named `<expression>.<mouth>.<eyes>.png` in lower case, with the expressions `neutral`, `smile`, `joy`, `thinking`, `surprised`, `worried`, `sad`, `stern`, `embarrassed` and `drowsy`, the mouths `closed`, `half` and `open`, and the eyes `open`, `half` and `closed`.
- Each exported expression has all nine of its frames. `neutral` is required; the app falls back from a missing expression to a near one, and then to `neutral`. A full pack holds 90 frames.
- Each frame is an 8-bit RGBA PNG (colour type 6) without interlacing, at most 1024 by 1024 pixels, with a transparent background and straight alpha. One camera, light and pose serve every frame, and only the face changes.
- A pack holds at most 100 files, `manifest.json` included, and at most 128 MiB in all (`MAX_PACK_BYTES`, which matches the consuming app's contract).

`manifest.json` is one JSON object in UTF-8, at most 64 KiB. Fields that this table does not name are refused, inside `frame`, `anchor` and `generator` as well.

| Field | Value |
|---|---|
| `schema` | `"squire.portraits/1"` |
| `companion` | optional: the character's name, 1 to 64 characters |
| `frame` | `{"width", "height"}`, each 1 to 1024 |
| `anchor` | `{"x", "y"}` in whole pixels: the centre of the face from the top-left corner, the point between the eyes (the middle of the eye bones' heads) projected by the camera and rounded to the nearest pixel, inside the frame |
| `baseline` | the row where the figure is cut, in whole pixels: below `anchor.y` and at most `frame.height`; for a bust cut by the bottom edge, `frame.height` |
| `expressions` | the exported expressions, each once, with `neutral` first |
| `generator` | `{"name": "OpenSculptBoy", "version": "<version>"}` |
| `licence` | the licence of the images, 1 to 500 characters |
| `provenance` | the model, its source and "rendered as the model is", 1 to 2000 characters |

The consuming app reads `frame`, `anchor` and `baseline` as unsigned integers, so the check refuses a fraction there, also one such as `6.0`, and refuses `true` and `false`. The text fields hold no control characters and no line breaks. The default character's licence names the licences of the model data: the Anny body model and this project's code and data under the Apache License 2.0, the MakeHuman, MPFB2 and Face Units assets under CC0 1.0, and the ICT-FaceKit face shapes under the MIT License. A card's character needs `--licence`, and the export stops with a message without it.

The check gives one reason code for each problem:

| Code | Problem |
|---|---|
| `extra_file` | a file that is not the manifest or a frame of a listed expression |
| `missing_frame` | a frame of a listed expression is missing |
| `wrong_size` | a frame's size differs from `frame` |
| `not_rgba` | a frame is not an 8-bit RGBA PNG |
| `interlaced` | a frame is interlaced |
| `unknown_field` | the manifest has a field that the contract does not name |
| `subfolder` | the folder holds a subfolder |
| `too_many_files` | the folder holds more than 100 entries |
| `too_large` | the pack, the manifest or a frame is over its limit |
| `bad_manifest` | the manifest is missing, is not JSON in UTF-8, or breaks a rule of its fields |

## Size

A full pack of the default character at 768 by 1024 pixels and 4 samples holds 40,993,641 bytes (39.1 MiB) in 91 files: the frames hold from 452,568 to 458,192 bytes each (about 445 KiB). A fitted card with its own textures, rendered at the default size on a graphics card with 16 samples, gave 90 frames of about 940 KB each, 80.6 MiB in all. A textured card therefore needs the 128 MiB limit of the contract. Most of the bytes lie in the hair and the skin, and more samples give smoother pictures that compress a little better.

## Tests

`test/test_portrait_pack.py` checks synthetic packs in CI: a valid pack, each broken case with its reason code, the manifest's rules, the faces of every frame, the folder rules and the command's exit codes. Its local tests need Playwright with the built page: the default character's full pack (90 frames at 288 by 384 pixels with one pass each, about 30 minutes in software) passes the check, the outline of the figure stays within one pixel in every frame apart from the face, the mouth differs between `closed`, `half` and `open` and the eyes between `open`, `half` and `closed` for each expression, and `--expressions neutral` gives a valid pack of nine frames. `OPENSCULPTBOY_PORTRAIT_PACK=FOLDER` checks a pack of the default character that was rendered before, and `OPENSCULPTBOY_CHROMIUM` names a Chromium executable for Playwright.

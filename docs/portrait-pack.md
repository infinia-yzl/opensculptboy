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

On a graphics card a frame takes about a second, and a full pack takes under two minutes. In software each pass takes about 40 seconds at 768 by 1024 pixels: a full pack takes about 16 hours at 16 samples and about 4 hours at 4, and the measure of the half-open eyes adds about 20 minutes. `--samples 4` keeps the hair smooth; one pass leaves the hair grainy.

## The faces

Each frame is a mix of Anny's 52 ARKit facial actions. The page plays them as its Express mode does, `jawOpen` as the turn of the jaw. An expression mixes the VRM 1.0 presets that the VRM export writes (`src/opensculptboy/data/vrm/expressions.json`; `lookUp` and `lookDown` as the eye-look actions), scaled by its weights, with facial actions of its own on top. Then the frame's mouth adds its vowels and the frame's eyes add `eyeBlinkLeft` and `eyeBlinkRight`. Since the export sets the actions itself, no preset blocks the blink or the vowels, and each of the nine frames of an expression shows its own mouth and eyes. Every action is clamped to the range from 0 to 1.

| Expression | Presets | Actions on top | Lids of the open eyes |
|---|---|---|---|
| `neutral` | none | | |
| `smile` | `happy` 0.6 | | |
| `joy` | `happy` 1.0 | | blink 0.25 (the eyes close a little in the smile) |
| `thinking` | `relaxed` 0.2, `lookUp` 0.7 | the gaze aside (`eyeLookOutRight` and `eyeLookInLeft` 0.7), one brow up and the other down (`browOuterUpLeft` 1.0, `browDownRight` 0.6, `browInnerUp` 0.2), the mouth pushed to one side and pressed (`mouthRight` 1.0, `mouthPressLeft` and `mouthPressRight` 0.4) | |
| `surprised` | `surprised` 1.0 | | |
| `worried` | `sad` 0.2 | `browInnerUp` 0.85, `browDownLeft` and `browDownRight` 0.15, `eyeWideLeft` and `eyeWideRight` 0.45, `mouthFrownLeft` and `mouthFrownRight` 0.2, `mouthStretchLeft` and `mouthStretchRight` 0.65 | |
| `sad` | `sad` 0.9 | | |
| `stern` | `angry` 0.9 | `browDownLeft` and `browDownRight` 0.1, `mouthPressLeft` and `mouthPressRight` 0.4 | |
| `embarrassed` | `happy` 0.6, `lookDown` 0.75 | `browInnerUp` 0.5, `cheekSquintLeft` and `cheekSquintRight` 0.2, `mouthRollLower` 0.35, `mouthPressLeft` and `mouthPressRight` 0.25 | |
| `drowsy` | `relaxed` 0.8 | | blink 0.15 (heavy lids) |

| Mouth | Vowels |
|---|---|
| `closed` | none |
| `half` | `aa` 0.4, `ih` 0.2 |
| `open` | `aa` 0.9, `oh` 0.2 |

The eyes start from the expression's own lids: the blink that its presets hold (`sad` and `relaxed` hold some) and the lids of the table. `open` keeps them, and `closed` closes them with a full blink. As the lids close, the blink takes over the widened eyes of `surprised` and `worried`. `thinking` and `worried` keep apart from `sad` by the gaze, the brows and the mouth; `embarrassed` keeps apart from `smile` by its lowered gaze and raised inner brows. Anny has no blush, so `embarrassed` holds no colour change. `opensculptboy.portrait_pack.faces` holds the mixes (`EXPRESSION_MIXES`, `MOUTH_MIXES`, `EYE_CLOSE`), and `frame_actions(expression, mouth, eyes, half)` gives the actions of one frame.

### Half-open eyes

The eye looks closed well before a blink of 0.5: on the default character a blink of 0.5 leaves about a third of the visible eye, because the lashes and the lids' own thickness cover the eyeball sooner than the lid margins meet. So the export measures the half-open eyes on each character in the page, with the character's own lids, eye targets and lashes. For each expression it draws a small picture of the eyes from the front twice, once with the eyeballs drawn plain white and once without them; the pixels that differ are the eyeball that the camera sees. The opening of each eye is the height of those pixels in the columns through its middle. A bisection of `eyeBlinkLeft` and `eyeBlinkRight` between the open eyes' blink and a full blink (`window.portrait.halfBlink`, five steps) finds the blinks that leave half of the open eyes' opening, and the half frames of every mouth use them. On the default character the half-open blink lies near 0.4 for `neutral`. The JSON result lists the blinks and the openings in millimetres under `half_blinks`, and the log names them. Without the page, `frame_actions` closes the half-open eyes half way from the expression's own lids to a full blink.

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

## The layered pack

A layered pack (`squire.portraits/2`) shows the character posed and at several framings, from a close-up of the head and shoulders to the whole body, for an app that stages the character as the subject of a large window. It holds layers that the app composes: a body for each pose and framing, a face over it (from the framing's main face set, or from the pose's own set when the pose turns its head), a front where a hand passes before the face, and the key poses of a leap of joy.

```bash
opensculptboy portrait-pack --layers --out pack/                                     # the default character: 11 poses, 4 framings, 10 expressions, the leap
opensculptboy portrait-pack --layers --character ada.json --out pack/ --licence CC0-1.0
opensculptboy portrait-pack --layers --out pack/ --poses at_ease,chin --framings close,thigh --no-leap --scale 0.5 --samples 4   # a quick trial
opensculptboy portrait-pack --layers --out poses/ --preview --scale 0.5 --samples 4  # the body layers alone, to look at the poses
opensculptboy portrait-pack --layers --character card.json --out pack/ --poses-file poses.json --shots-file shots.json   # the stage's poses and shot cameras: 7 framings
opensculptboy portrait-pack --check pack/                                            # the check reads the schema and checks either kind of pack
```

| Option | Default | What it sets |
|---|---|---|
| `--layers` | | write a layered pack |
| `--poses LIST` | all eleven | the poses, comma-separated; `at_ease` is always in |
| `--framings LIST` | every framing the shots render, else the first four | the framings, among `close`, `waist`, `thigh`, `full`, and with `--shots-file` `thought`, `stern` and `low` |
| `--no-leap` | | leave out the leap's key poses |
| `--format` | `webp` | `webp` (lossy colour with lossless alpha) or `png` |
| `--quality N` | 88 | the WebP quality of the body layers; the faces take 2 more |
| `--scale F` | 1 | a factor on every framing's pixels per head, at most 1, for trials |
| `--preview` | | the body layers and the leap alone, with no faces |
| `--poses-file FILE` | `opensculptboy/data/portrait/poses.json` | the companion's shared pose file (`squire.poses/1`) |
| `--shots-file FILE` | | the stage's shots (`squire.shots/1`): each framing renders through the camera of the shot it serves |

### Framings

Each framing has its own scale in pixels per head (the height from the chin to the crown), so that an app with a window of 1440 by 900 pixels at device scale 2 never shows a layer larger than its pixels.

| Framing | For | Pixels per head | The frame |
|---|---|---|---|
| `close` | a close-up and a thought | 810 | from the top of the hair or headpiece to 2.2 heads below the crown |
| `waist` | a medium close-up | 660 | from the highest raised hand to 3.6 heads below the crown |
| `thigh` | a two-shot and a far, low shot | 330 | from the highest raised hand to 6 heads below the crown |
| `full` | the whole body and the leap | 160; the leap's frames 230 | from the highest raised hand (the leap's included) to the ground; the leap's frames have the same camera and a frame of their own (`leap.frame`, `leap.head`), since the stage pushes in at the top of the leap until the figure takes 88 % of the height, which asks for about 210 |
| `thought` | the thought (`chin`) | 810 | as `close` |
| `stern` | the square-on stern close-up (`at_ease`, `arms_crossed`; the main faces `neutral` and `stern`) | 810 | as `close` |
| `low` | the low, far shot of sadness (`downcast`) | 330 | from the top of the head to 6.8 heads below the crown |

The export first renders a small outline of every pose and sets each framing's width from the widest pose at its height. Each framing holds the poses its shot shows (`layers.FRAMING_POSES`): the close framing leaves out the poses with a hand above the head, `chin` shows at `thought` alone, `downcast` at `low` alone, and the bow of `hand_on_chest` at `waist` and `thigh`. `thought`, `stern` and `low` render only through their shots' cameras (`--shots-file`), and `at_ease` shows at each of the first four.

### Expressions by framing

A framing's main face set holds only the expressions its shots show (`layers.FRAMING_EXPRESSIONS` and `FRAMING_LEAVES`), as the stage's rules give them (each rule's expressions, through the shots it takes at each intensity and window size, to the framing of each shot's stills):

| Framing | The rules that show it | The expressions they show | The main set holds |
|---|---|---|---|
| `close` | `reply` at intensity 3 | every expression but `stern`, which the stern close-up takes | every expression but `stern` when the pack holds `stern`, else all |
| `waist` | `joy`, `moved`, `relaxed`, `reply`, `stern` | all | all |
| `thigh` | `held`, `joy`, `listen`, `reply`, `state`, `stern` | all | all |
| `full` | `joy` (the celebration) | `joy` | `neutral`, `joy` and `smile` (the moment before and after the leap; the leap's frames carry their own face of joy) |
| `stern` | `reply` and `stern` with the stern expression | `stern` | `neutral` and `stern` |
| `thought` | `thinking` | `chin`'s own set | `chin`'s own set: `neutral`, `thinking` |
| `low` | `sad` | `sad`, `worried` | `downcast`'s own set: `neutral`, `sad`, `worried` |

### Poses

The eleven poses come from the companion's shared pose file (`opensculptboy.portrait_pack.shared_poses`, format `squire.poses/1`): the file that the stage's 3D preview reads as well, so the stills show the body exactly as the VRM shows it. `data/portrait/poses.json` is the command's default, and `--poses-file` names another. The file gives angles on the normalized humanoid bones of VRM 1.0, which rest in a T pose. Each humanoid bone maps to its rig bone by the VRM export's table (`vrm_tables.HUMANOID_1`), and each rig bone first turns from its rest (an A pose) into the VRM file's T pose by its T-pose turn (`tpose.t_pose_turns`), computed for the card's own body: its rest-frame correction. A rig bone outside the humanoid (a twist bone, `spine03`, the neck's upper bones, the metacarpals, the toes) turns with the humanoid bone above it, and the forearm's twist bone takes half the hand's roll, as the VRM file's roll constraint gives it. The rig's joints then land where the VRM exported from the same card puts them (`test_lands_on_the_vrm_skeleton`, within 2 mm), and the page shows the rotations through `window.portrait.pose`.

| Pose | The body |
|---|---|
| `at_ease` | standing at rest, the weight on one leg, the arms down |
| `relaxed` | the weight on one leg, a hand on the hip |
| `hands_behind_head` | both hands behind the head, the elbows out |
| `chin` | one hand under the chin, the head turned toward the thought |
| `scratch_head` | one hand on the back of the head |
| `arms_crossed` | the arms folded across the chest |
| `hand_on_chest` | a hand on the chest with a slight bow |
| `wave` | one hand raised beside the head |
| `fist` | one fist raised above the head |
| `cheer` | both hands in the air |
| `downcast` | standing at rest with the head lowered and turned away |

The file's stances (`stances`, named by a pose's `stance`) give the hips, the spine and the legs: `rest` puts the weight on one leg with the hips and the shoulders tilted against each other, and `lean` is the same stance stronger. The leap's key poses `crouch`, `rise`, `peak` and `land` come from the same file, as `leap_crouch`, `leap_rise`, `cheer_peak` and `leap_land` (`poses.LEAP_POSES`), and carry the face of joy. Most poses keep the head and the neck of `at_ease` in the world (the file's `keepHead`) and share the framing's main face set; `chin`, `hand_on_chest` and `downcast` turn their own head (`keepHead: false`), and the pack renders a face set of their own at each framing they show at, with the expressions the file lists in their `faces`.

### How the layers are made

- The camera of a framing follows the head: for each pose it stands where the head lands on the same pixels of the frame, and it looks at the plane through the eyes (the `cz` of `window.portrait.view`). With the head's turn kept, a face rendered in `at_ease` fits every body of the framing.
- The face layers cover the rectangle of the face that changes between the faces: the export draws each expression's open mouth and closed eyes against the neutral face around the head at one pass and takes the box of the changed pixels with a margin of 0.05 heads, snapped to 8 pixels. It then renders the 90 faces in that rectangle alone (`region`), with 24 more pixels around it that it cuts off again, so the skin's screen-space pass sees the same neighbours as in the whole frame. Each face layer paints only the face's own pixels: those that any face changes, grown by 0.04 heads with an edge softened over 0.02 heads (pixels that differ between two pictures of one face count as noise and stay out); the rest of the rectangle (the hair, the ears, the shoulders) is transparent, so the body shows there.
- A front layer holds the pixels of the face's own pixels that a pose covers (a hand): where the pose's body differs from `at_ease`'s by more than 40 of 255, grown by 2 pixels with a soft edge; a pose that covers less than 0.3 % of the rectangle has none.
- A pose that turns its own head gets its own face set at each of its framings: its rectangle and its face's own pixels are found as for the main set, in the pose itself with the framing's camera, and its faces are rendered in the pose (`face.<framing>.<pose>.<expression>.<mouth>.<eyes>`), so a hand under the chin stays in place and the pose needs no front layer.
- The leap's frames use the full framing's camera of `at_ease`, so the crouch stands lower and the raised hands reach higher in the same frame.
- With `--shots-file`, each framing renders through the camera of the shot it serves (the file's `renders`: `close` for the close-up, `waist` for the medium close-up, `thigh` for the two-shot, `full` for the full shot), in place of the long lens from the front. The shot names a lens in millimetres, full-frame equivalent over the window's height; the body point the camera sits level with (the eyes, the chest or the hips, in the stage's head heights below the eyes: the hips-to-eyes distance over 3.5) and how far above it; a tilt; and the body's turn, which the page applies to the figure (`turn` of `window.portrait.view`). The camera stands where the lens's field of view spans the shot's window (its `span` over its `share`) at the eyes, and it follows the head as above. The outlines then come from probes through that camera, and `window.portrait.view` takes the camera as `{eye, pitch, focal, px, py}`: the frame is a rectangle of a larger picture centred on the principal point. A tilted camera foreshortens the head, so the crown's and the chin's rows of the manifest differ by up to 5 % from the framing's pixels per head, which holds at the eyes' depth; the manifest therefore records the head's projected `size` (one stage head height, upright at the eyes' depth, in pixels), and each framing's `camera`: the shot (with the expression for a render such as `{"shot": "close_up", "expression": "stern"}`, whose `byExpression` values apply), the lens, the height and the distance above it, the tilt and the body's turn.
- The body layers carry the neutral face (mouth closed, eyes open). The lights, the exposure, the long lens and the straight alpha are those of the bust pack.

### The manifest and the check

`manifest.json` holds `schema` (`squire.portraits/2`), `format`, `expressions`, `framings` (each with `frame`, `head` with the anchor `x` and `y`, the crown's and the chin's rows `top` and `chin` and the projected `size`, the shot's `camera`, the main face set's rectangle `face` and its `expressions` where they are fewer than the pack's, the own face sets `sets` of the poses that turn their head, and the ground's row `ground` for `full`), `poses` (each with its `framings` and the framings of its `front` layers), `leap` (`framing` and `frames`), `extra` (`above` and `width` in heads: what the hair and a headpiece add to the figure), `joints` (`names`, then for each pose and framing in `poses`, and for each leap frame in `leap`, the x and y of each name in the layer's own pixels: the point between the eyes, the crown, the chin and VRM humanoid bones by their heads, so an app can check where its frame's edges cut the figure; each list of numbers sits on one line), and `generator`, `licence`, `provenance` and the optional `companion` as in the bust pack. `opensculptboy.portrait_pack.layered_contract.check_layered_pack` checks a layered pack with the reason codes of the bust pack's check: the layers' names and sizes against the manifest, WebP with alpha or 8-bit RGBA PNG, at most 600 files, 256 MiB and 4096 pixels a side.

### Resuming a stopped run

`--resume` continues a layered pack that a stopped run left in `--out`, with the same settings (the cuts of the next section included: a layer of another size or an expression the framing no longer holds is rendered again or removed). The folder is kept, and each layer whose file and pass files decode whole at their size is kept; a file that a stopped run cut short fails to decode and is rendered again. The first body layer is always rendered again, since its passes set the pack's scales, and a kept body layer whose front layer is missing has its passes rendered again for the front's. The probes, the framings, the face rectangles and the joints are measured again, which takes a minute or two. The result's `kept` counts the layers kept.

### Size and time

**Where the time goes.** Each layer with light passes was four pictures until 2026-11-07: the layer, then the key, the fill and the rim, each at 16 samples, each pass read back from the page as half floats and written as WebP at the slowest effort with its colour difference measured on every pixel. A dressed card's close framing took about 110 seconds for each body layer and 1,842 seconds for its 90 faces (20 seconds a face), and its waist bodies about 160 seconds each; the graphics card stood mostly idle while Python encoded and measured. The render now cuts that time:

- **The layer is its passes' sum.** With passes, the page renders the three passes alone and the export writes the layer as their sum through the display pass (`Lit.compose`): three pictures for each layer, and the layer is the very sum an app gives at the strengths of 1, up to the files' compression.
- **Only the expressions each framing needs** (the table above): the full framing's faces fall from 90 to 27.
- **Fewer samples where the face is small:** `--framing-samples` (default `thigh=8,full=8`; `none` gives every framing `--samples`); every other framing and the leap take `--samples` (16), and a framing never takes more.
- **The leap at its own scale:** the full framing's bodies and faces at 160 pixels per head, the leap's four frames alone at 230.
- **Writing beside the page:** every layer is written on a pool of four threads while the page renders the next one, with at most six waiting; the pass files use WebP's effort 4 (about seven times faster than 6 at the same size), and the colour difference is measured on every fourth pixel of the figure.
- **The timing.** The log and the result's `timing` give the seconds of each phase: the probes and framings, the face rectangles, the passes' rendering, composing, the bodies and fronts, the faces, the leap, the writing on the pool (which overlaps the page) and the waits for it.

The cuts give, for the same card and the seven framings (estimates from the measured rates above; the next full render measures them): about 12 seconds for each face at 16 samples and 5 to 7 at 8, about 75 seconds for each large body layer, and about 1 hour 50 minutes for the whole pack, against about 4 hours before. With `--resume` a run keeps the earlier run's layers when the scales it reads back from the first body layer's pass files agree within 2 % with the scales it measures again, and uses those scales; otherwise it renders every layer again. The layers the new settings no longer name (an expression a framing no longer holds, a layer of another size) leave the folder at the end, and the result counts them (`removed`).

A textured, dressed card renders on a graphics card in about 20 minutes: the body layers take 10 to 25 seconds each at 16 samples (the waist frames are the largest, about 3600 by 3400 pixels), and each face 1 to 3 seconds. Its full pack of version 1's ten poses held 435 files and 37.4 MB (35.7 MiB), most of them in the face layers. With the seven framings of the shots and the face sets the full pack holds about 550 files and 45 MB: the main faces of `close`, `waist`, `thigh` and `full` (360) and of `stern` (18), the own sets of `chin`, `hand_on_chest` and `downcast` (99), 35 body layers, about 30 front layers and the leap. The default character's pack at a fifth of the scale with one pass renders in under a minute.

### Light passes

`--passes key,fill,rim` adds the light passes of every layer, for an app that relights the stills live. Each body, face, front and leap layer then comes with one file per pass beside it, at every framing the pack holds (the shots' own framings `thought`, `stern` and `low` among them) and for every face set (the framing's main set and each pose's own set), named after the layer with the pass before the format (`body.at_ease.close.key.webp`): the key light alone, the fill alone (the environment: the sky, the ground and the soft boxes, with a garment's own light) and the rim light alone. `--passes base,rim` keeps the key and the fill in one pass (`base`), for an app that must hold fewer textures.

- **The lights.** `--light-rig stage` (the default with passes) lights the figure with the stage's rig: the studio preset's lights and exposure turned so that the key comes from the open side of the frame (the frame's right for a figure on the left), 40 degrees off the camera and 20 degrees up, as loop lighting; the large fill box beside the camera on the side away from the key; and a neutral rim from behind on that side. `--light-rig studio` keeps the page's studio preset, the look of the bust pack. The layers themselves are rendered in the same rig, so a pack with passes still plays in an app that ignores them.
- **How a pass is made.** `window.portrait.light(rig)` lights the page with the rig, `window.portrait.pass(name)` keeps one part of its light (the shaders reach the light only through the key's colour and its soft box's reflection, the rim's colour, the environment's two strengths and the garments' emissive factor, so the parts add up to the whole), and `window.portrait.linear(actions, samples)` reads the accumulated picture before the display pass: the straight colour times the exposure, with the coverage, as half floats.
- **How a pass is stored.** WebP holds 8 bits a channel, so each pass stores its linear colour over the pack's scale for that pass (the pass's light at the 99.9th percentile of the first body layer's covered pixels, with 15 % headroom), through the sRGB transfer curve, with the layer's own alpha, lossless and the same in every pass. The curve's steps follow the eye's, so a pass keeps its light to about 1 % from a tenth of its scale up and 2.5 % from a fiftieth, and an app's graphics card decodes it for free when it reads the file as an sRGB texture, filtering in linear light. The colour reaches 4 pixels into the transparent surround (WebP's `exact`), so a resampled edge keeps its colour. The quality is the layer's and 4 more (`--pass-quality`).
- **The sum.** An app sums the passes, each times its scale and its strength, and applies the page's display pass to the sum: three.js's AgX tone curve, the saturation about the Rec. 709 luminance, the sRGB curve and the contrast (`opensculptboy.portrait_pack.light.display`). At the strengths of 1 the sum gives the layer. The command's result (`light.reproduction`) reports the CIEDE2000 colour difference between each layer as rendered and its stored passes' sum, over the figure's pixels (alpha at least 128), beside the difference of the layer's own WebP file.
- **The manifest.** `passes` lists each pass's `name` and `scale`, and `light` describes the light: the `rig`, the `key_side` of the frame, the `exposure`, the `tone` (`curve` agx, `saturation`, `contrast`), the `encoding` (srgb), the `lights` and each pass's mean light on the lit side of the face of `at_ease` (`face`, linear). The check wants every pass file at its layer's size, and its bound on files grows by the passes' files.

Measured on a desktop graphics card (2026-11-07), as the CIEDE2000 difference over the figure's pixels between each layer as rendered and its passes' sum:

| Pack | Rig | Mean | 95th percentile | The layer's own WebP, mean |
|---|---|---|---|---|
| The default character at full scale, the close framing (85 files) | Studio (the bust pack's look) | 0.81 | 2.3 | 0.89 |
| The default character at full scale, the close and full framings and the leap (257 files) | Stage | 0.81 | 2.25 | 0.91 |
| The default character at a fifth of the scale (185 files) | Studio | 1.19 | 3.3 | 1.39 |
| A dressed card at a fifth of the scale (1,041 files) | Stage | 1.92 | 5.2 | 1.84 |

The passes take about three times the bytes of the layers (the key's and the fill's each about the layers', the rim's about half), and the time of three more pictures for each layer: the default character's pack at full scale above took 32 minutes, 16 of them for the passes. A dressed card's full pack with the seven framings of the shots (about 550 files and 45 MB) gains about 135 MB of passes and 1,650 files, inside the bounds of a pack with three passes (2,400 files, 256 MiB).

## Tests

`test/test_portrait_light.py` checks the light passes in CI (the rigs against the page's preset, the encoding and its precision, the colour bled around the figure, the tone curve, CIEDE2000 on Sharma's test pairs, the check of a pack with passes and the command's options) and renders the default character's pack with passes at a fifth of its scale locally, with the sum of its passes against each layer and the key on the face's right. `test/test_portrait_layers.py` checks synthetic layered packs, the framings, the front layer's mask, the shared pose file (its format, every bone resolving, the T pose, the joints against the VRM file's, the head kept) and the shots' cameras in CI, and renders the default character's layered pack at a fifth of its scale locally. `test/test_portrait_pack.py` checks synthetic packs in CI: a valid pack, each broken case with its reason code, the manifest's rules, the faces of every frame, the folder rules and the command's exit codes. Its local tests need Playwright with the built page: the default character's full pack (90 frames at 288 by 384 pixels with one pass each, about 30 minutes in software) passes the check, the outline of the figure stays within one pixel in every frame apart from the face, the mouth differs between `closed`, `half` and `open` and the eyes between `open`, `half` and `closed` for each expression, the half-open eyes of every expression leave 45 to 55 % of the open eyes' opening, and `--expressions neutral` gives a valid pack of nine frames. `OPENSCULPTBOY_PORTRAIT_PACK=FOLDER` checks a pack of the default character that was rendered before, and `OPENSCULPTBOY_CHROMIUM` names a Chromium executable for Playwright.

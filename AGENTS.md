# AGENTS.md

This file provides guidance to AI Agents when working with code in this repository.

## Project Overview

**Anny** is a differentiable, parametric body mesh model in PyTorch with a common topology and parameter space. Based on MakeHuman assets, it provides full-body, hand, and face models.

This repository is **OpenSculptBoy**, an independent fork of Anny (NAVER LABS Europe) that adds the web viewer, face shapes, hair, poses and correctives. The README states that AI agents write and maintain the additions under one human author, and it carries the credits and the no-warranty disclaimer; keep those sections accurate when you change the project. The Python distribution is `opensculptboy` (`pyproject.toml`) and holds two packages: `anny` (`src/anny`, the body model, kept close to upstream so that merges stay clean: `git fetch upstream main`, remote `upstream` is `https://github.com/naver/anny`) and `opensculptboy` (`src/opensculptboy`, this project's additions on top of the model: `Character`, the glTF and VRM export and the `opensculptboy` command). Put new features that only use the model's public API in `opensculptboy`, and change `src/anny` only where the model itself must change. Use "OpenSculptBoy" for the project and "Anny" for the model and the `anny` package, and never present the fork as an official NAVER release. In the README, the viewer and other user-facing text, present the models as humanoid 3D characters for free, unlimited creativity: never as humans, and never by human age. The `age` parameter keeps its name in the API and appears as "Form" in the viewer, with the ends "Compact" and "Elongated".

### Conventions

- Write Markdown docs with one line per paragraph or list item. Never hard-wrap prose; editors and GitHub wrap it for display.
- Bring upstream updates in their own pull request: a branch from `main` with `git merge upstream/main`, merged with a merge commit (never squashed or rebased), so that git keeps tracking which upstream commits the fork has. Keep upstream commits out of feature pull requests.

## Commands

### Setup
```bash
uv sync --extra examples  # full install with demo dependencies
```

### Testing

The checks come in three tiers, and `scripts/check.sh` runs each of them. GitHub Actions (`.github/workflows/lint.yml`, `viewer.yml` and `tests.yml`) calls the same script, so a local run of a tier matches CI.

| Tier | Where | Command | Runs |
|---|---|---|---|
| quick | locally, while working | `scripts/check.sh quick test.test_gltf_export ...` | lint (`uv lock --check`, ruff at the version pinned in the `dev` extra, copyright headers), then the test modules you name, with the CI settings |
| ci | GitHub Actions on every pull request and push to `main`; locally before a push | `scripts/check.sh ci` | lint; the viewer type-check and stale-page check; the whole suite with `OPENSCULPTBOY_CI=1` and `OPENSCULPTBOY_SKIP_NONCOMMERCIAL=1`; the wheel installed into a fresh environment with CPU torch |
| full | locally only | `scripts/check.sh full` | the ci tier, with the local-only tests and the non-commercial data tests where their tools and data are present |

Rules for agents:
- While working, run `scripts/check.sh quick` with the test modules your change touches. Before a push, run `scripts/check.sh ci`, or at least the jobs your change touches (`scripts/check.sh lint|viewer|tests|package`).
- `OPENSCULPTBOY_CI=1` skips the tests marked `@local_only(reason)` (`test/markers.py`). Mark a new test `local_only` when it needs something CI lacks: Playwright with the built page (`test_hair_page`, `TestViewerFacePage`), the viewer data build (`test_viewer_body`, `test_hair_parity`, `TestPageSolver`, `TestViewerFaceData`), MediaPipe's models (`TestPoseFromImage`), the viewer's node packages three-vrm and gltf-validator with Playwright and Chromium (`test_vrm_three`, with the `faces` extra and after `cd viewer && npm ci`), or licensed files (`test_smpl`).
- `OPENSCULPTBOY_SKIP_NONCOMMERCIAL=1` makes the download of the non-commercial SMPL and SMPL-X data raise `unittest.SkipTest` (`test/__init__.py`), so CI never fetches it. Put each topology of a loop in its own `subTest`, so that the smpl and smplx cases skip alone.
- `lint.yml` runs on every pull request and on every push to `main`, without path filters. The other workflows run on the same events only when their files change (GitHub `paths:` filters): `tests.yml` (the suite and the package) for changes to `src/`, `test/`, `viewer/src/`, `viewer/test/`, `pyproject.toml`, `uv.lock`, `MANIFEST.in`, `scripts/check.sh` or the workflow; `viewer.yml` for changes to `viewer/`. Draft pull requests skip the tests and the package; marking a pull request ready for review runs them. A change to the docs alone runs lint only. The repository is public, so GitHub-hosted runners cost nothing; the filters save runner time. When you add a directory that the tests read, add it to the paths of `tests.yml`.
- `test_vrm_schema` validates the VRM extensions of the shared exports against the VRM schemas and the glTF base schemas. The vrm-c/vrm-specification repository has no licence, so never vendor its schemas: the test downloads them at the commits pinned in `VRM_SPEC_COMMIT` and `GLTF_COMMIT` into `ANNY_CACHE_DIR/vrm_schema/<commit>/` and skips offline. Its structural checks (`check_vrm1`, `check_vrm0`) need no download and are the CI guard for both versions.
- `test/vrm_fixtures.py` caches the shared VRM exports of one character (`vrm_export`, `vrm_json`, `vrm_build`, `vrm_glb`); every VRM test module uses them instead of exporting again. `test/gltf_reader.py` evaluates a GLB or VRM file as an engine does (`evaluate_nodes` skins every primitive of every skinned mesh node; `structure_errors` lists dangling indices and undeclared extensions).
- `test_viewer_wearables_page` (local only: Playwright with the built page) opens the synthetic outfit of `test_viewer_wearables` through the Wearables section's file input, and `TestWearablesDrop` drops a card on the `anny-mouth` rig with its packages as a script does (a `DataTransfer` of files by name alone). The page never holds a wearable package: tests hand their packages to the page through that input or a drop.
- `TestGltfReference` (`test/test_gltf_export.py`) guards the GLB output against `test/data/gltf_reference.json` (format `opensculptboy/gltf-reference@2`): the structure of seven exports (among them the float32 export of the command, a body with every phenotype and local changes, and a head part model with dense morph targets), with a SHA-256 of each integer accessor and order-sensitive sums of each float accessor, so that moved rows, flipped triangles and wrong indices fail it. The float32 case allows for the rounding of other CPUs, so the test also compares it value by value with a float64 export of the same character. The exports are byte-identical to those of the exporter before the VRM work, except for the `target` of dense morph-target buffer views, the eyeball triangles, whose tied diagonals Anny now splits alike on every machine (`DIAGONAL_TIE` in `anny.utils.mesh_utils`), and the mouth cavity in the targets that press the lips onto it (`mouthClose`, `mouthFunnel`, `mouthRollLower`), which `opensculptboy.export.mouth` pushes behind the skin. The reference holds on every CPU: to check an export on the code path of GitHub's runners, run the test with `MKL_CBWR=COMPATIBLE` and an empty `ANNY_CACHE_DIR`. After a deliberate change of the GLB output, record it again with `uv run python -c "from test.test_gltf_export import write_reference; write_reference()"`.
- A change to `viewer/src` or `viewer/shell.html` needs the rebuilt page committed (`cd viewer && node build.mjs --reuse-data`); the viewer job fails on a stale `viewer/dist/anny_viewer.html`.
- NAVER's cluster workflow (`.github/workflows/tests.yml`, `.github/scripts/ci-install.sh`) is deleted in this fork. When an upstream merge touches those files, keep them deleted.

```bash
uv run --extra dev --extra examples python -m unittest test.test_various  # a single test file, as in the quick tier
```

### Documentation
```bash
bash build_doc.bash  # build HTML docs from the jupytext py:percent tutorials in tutorials/*.py
```

### Export (OpenSculptBoy)
```bash
uv run opensculptboy character > card.json                                  # a character card with the default settings
uv run opensculptboy export out.glb --character card.json --animation walk  # glTF 2.0 binary; --morph-targets all, --influences 8
uv run python -m unittest test.test_gltf_export                             # evaluates the file as an engine does and compares it with Anny
uv run opensculptboy export out.vrm --character card.json --author me       # VRM 1.0; --vrm-version 0, --thumbnail auto, --meta meta.json, --bind forward, --budget strict
uv run python -m unittest test.test_vrm_export test.test_vrm_tpose          # the VRM files against the T-pose rebind, both versions
uv run opensculptboy export out.vrm --character card.json --author me --fit-budget   # reduce the pieces, bake their normals again, atlas their materials until the file fits (--fit-priority, --fit-triangles)
uv run --extra viewer python -m unittest test.test_vrm_fit_budget          # --fit-budget: reduction, atlas, report per piece, fitted files of both versions, re-bake, command
uv run python -m unittest test.test_vrm_skin_weights                        # the refined skin weights: T-pose folds and hand poses (about 30 s in CI; two more corners run locally)
uv run python -m opensculptboy.export.authoring.skin_weights --force        # optimise and write data/vrm/skin_weights.safetensors (its strict gates fail by design without the refinement), and print the gate report
uv run python -m opensculptboy.export.authoring.refine --edits FILE         # the gate report of the refinement for each body
uv run --extra dev --extra faces python -m unittest test.test_vrm_schema test.test_vrm_three  # VRM schemas, three-vrm and the Khronos validator (the last two need cd viewer && npm ci)
node viewer/test/vrm_page/validate.mjs out.vrm                              # the Khronos glTF validator on a GLB or VRM file (JSON report)
uv run --extra pose opensculptboy pose photo.jpg --card pose.json --svg pose.svg --png pose.png   # a character posed from a picture, drawn flat
uv run --extra viewer opensculptboy fit handoff/ --out card.json --detail 1 --textures  # fit a reference mesh (docs/reference-fit.md); writes card.detail0/1.safetensors and the baked textures
uv run opensculptboy export out.glb --character card.json --detail 1             # a level of the card's detail layer (VRM files keep within their budget)
uv run --extra viewer opensculptboy bake handoff/ --card card.json --material skin.json  # bake a fitted card again: atlas 4096, head tile, ear tile, roughness, thickness, curvature, and the card's skin material
blender --python src/opensculptboy/export/blender_skin.py -- out.glb            # import a GLB into Blender and set the skin's subsurface and the eyes' cornea from its extras
uv run opensculptboy export out.glb --character card.json --eyes eyes.json      # the eyes of opensculptboy.eyes (docs/eyes.md): eyeball, cornea, caruncle, iris texture
uv run python -m unittest test.test_eyes                                        # the eyes: card, geometry, texture, eyed mesh, caruncle, GLB, VRM
uv run python -m unittest test.test_lashes                                      # the lashes: card, generic texture, lid margins, blinks, reference lashes, GLB, VRM
uv run python -m unittest test.test_groom                                       # a reference's hair and brows: handoff, carry, skin and chains, files, card, GLB, VRM, page blinks
uv run python -m unittest test.test_blinks                                      # the blinks: rims that meet, eye clearance, in-between, lashes, files at 0 to 100 %, a reference's blink
uv run python -m unittest test.test_texture_levels                              # normal maps re-encoded for a coarser level than the bake's
uv run python -m unittest test.test_mouth_rig                                   # the anny-mouth rig: jaw and tongue bones, jawOpen as a jaw turn and a corrective, GLB and VRM
uv run python -m unittest test.test_mouth_lips                                  # the sealed lips (lip line, edges, seal) and the jaw's weights from a reference's mouth poses
uv run python -m unittest test.test_mouth_poses                                 # a reference's mouth poses as targets: carry, file, GLB turns, VRM visemes
uv run python -m opensculptboy.mouth.authoring.rig_cache --check neck01        # rebuild data/mouth/rig_cache.safetensors (the mouth bones' cached orientations)
uv run python -m unittest test.test_mouth_inner                                 # a reference's inner mouth: handoff pieces, joint map, card, hidden cavity, skin, GLB, VRM
uv run opensculptboy export out.glb --character card.json --teeth                # the teeth of opensculptboy.teeth (docs/teeth.md): the reference's from the bake, else generic
uv run python -m unittest test.test_teeth                                       # the teeth: card, generic set behind the lips, jaw follow, reference teeth, GLB, VRM, --teeth
uv run python -m unittest test.test_mouth                                       # the mouth cavity kept behind the skin in the rest mesh and each morph target
uv run python -m opensculptboy.topology.build                                    # compose the enabled patches into data/topology/sculpt.safetensors and sculpt.json (--with placeholder, --only NAME, --out FILE)
uv run python -m unittest test.test_sculpt_topology                             # the sculpt topology with the placeholder patch: format, exact rows, mesh, UVs, levels, GLB, converter
uv run python -m unittest test.test_sculpt_eye_corner                           # the eye-corner patch: ray test of the corner, crossings, flips, detail levels, the eyes on sculpt
uv run python -m unittest test.test_sculpt_ear_root                             # the ear-root patch (off in the shipped file): band, root line, validation, levels
uv run opensculptboy convert card.json --to sculpt --out card_sculpt.json        # a card and its detail offsets and ear UVs on the sculpt mesh (--to anny for the reverse)
uv run --extra viewer opensculptboy fit handoff/ --out card.json --topology sculpt  # fit the sculpt mesh
uv run python -m unittest test.test_sculpt_downstream                           # sculpt downstream: base-mesh lookups, fit masks, ear root line, GLB and VRM, synthetic fit, convert (about 2 min)
uv run --extra viewer opensculptboy bake-tiers card.json                        # the cheaper tiers' normal maps baked from the master, and the occlusion (docs/budget-tiers.md)
uv run opensculptboy export out.glb --character card.json --tier medium          # a rendering-budget tier: master, high, medium (.glb) or low (.vrm by default)
uv run opensculptboy portraits --character card.json --out avatars/              # a portrait and an expression sheet (neutral, smile, surprised, sad, angry) in flat toon shading
uv run --extra viewer python -m unittest test.test_budget_tiers                   # the tiers: meshes, budgets, baked normals against the master, files, portraits (about 20 s)
uv run opensculptboy export out.glb --character card.json --outfit travel --hide cape   # an outfit with single pieces hidden (docs/wearables.md)
uv run opensculptboy export out.glb --character card.json --wardrobe                   # every piece of every outfit as its own node
uv run opensculptboy wearables reference handoff/ --card card.json --out folder/       # reference wearables (private packages, outside the repository)
uv run opensculptboy wearables bind garment.glb --out wear/tunic --slot torso --licence CC0-1.0 --design-source TEXT   # a garment modelled on the default body (export base.glb) as a package; --rigid BONE, --weights own, --groups-from, --covers, --hides, --body card.json, --tag, --description, --accept-quality
uv run opensculptboy wearables import-mhclo shirt.mhclo --out wear/shirt               # a MakeHuman clothes asset (.mhclo, .obj, .mhmat) as a package; --slot, --layer, --tag, --pack-json, --bump-strength, --covers, --normal-green, --texture-limit, --accept-quality, --dry-run; one JSON result
uv run opensculptboy wearables add card.json wear/shirt.wearable.json --outfit casual --activate   # packages into an outfit or as loose pieces (--loose, --replace, --dry-run); remove and outfit --activate NAME
uv run opensculptboy wearables catalogue wear/ --thumbnails --slot torso --query linen   # the packages of a folder as one JSON result: slot, layer, tags, colours, licence, quality, thumbnail; --mhclo FOLDER, --tag, --refresh, --no-quality
uv run opensculptboy wearables check card.json --sheet card.check.png                  # the outfit's quality and fit on test bodies at rest, in the T-pose and in the clips, with gates and a review sheet; --outfit, --bodies, --clips, --frames, --pose-limit; exit 0 pass, 1 fail, 2 error
uv run opensculptboy wearables outfit export card.json --outfit casual --out casual.outfit.json   # an outfit file (opensculptboy.outfit/1); wearables add other.json casual.outfit.json puts it on a card (--outfit, --replace, --activate, --dry-run)
uv run python -m unittest test.test_outfit_tools                                       # the catalogue, the check with planted defects and the outfit files, with synthetic garments
uv run python -m unittest test.test_wearables_import                                   # import-mhclo, the quality checks and the card commands with synthetic MakeHuman assets, the package schema
uv run python -m unittest test.test_wearables                                          # the wearables with synthetic garments: format, provenance, binding, fit gate, outfits, files
uv run python -m unittest test.test_viewer_wearables                                   # the page's dressing (viewer/src/wearables/) against opensculptboy.wearables, in node
uv run python -m opensculptboy.wearables.viewer                                         # viewer/src/wearables/skin_split.ts: the skin triangles as halves of the page's body quads
uv run python -m unittest test.test_wearable_springs                                   # the spring chains of a synthetic cape and skirt: chains, weights, colliders, GLB, VRM, the spring update, a robe's sleeves on the arms
uv run python -m unittest test.test_wearables_bind                                     # wearables bind with synthetic garment files: fields, round trip, frame, dressed bodies, far vertices, quality, command
uv run python -m opensculptboy.statues run src/opensculptboy/data/statues/doryphoros.json --scans scans/ --out out/   # the statue pipeline from a config: convert, landmarks, preview, fit, measure (--steps, --detail, --fit NAME=VALUE, --variant; --from out/<name> runs the detail levels alone from an earlier run, and with --reconvert converts the scan again with new masks)
uv run python -m opensculptboy.statues starter src/opensculptboy/data/statues/doryphoros.json --run out/ --out starters/doryphoros.zip   # a fitted statue as a finished starter of the web viewer: card with the marble look and the scan's credit, detail files, thumbnail (--levels, --piece, --variant)
uv run python -m opensculptboy.statues convert statue.stl --out handoff/ --lost hand.l,nose   # a statue scan as a handoff alone: upright, plinth off, scaled, masks (--masks, --landmarks, --up, --front, --licence)
uv run python -m unittest test.test_statues                                            # the statue conversion with a synthetic statue: frame, plinth, masks, lost parts, borders, region distances
```

### Face calibration
```bash
uv sync --extra viewer --extra faces
uv run python -m anny.faces.authoring.sources       # download the data sources into ANNY_CACHE_DIR/faces
uv run python -m anny.faces.authoring.landmarks     # data/keypoints/craniofacial.json
uv run python -m anny.faces.authoring.fit_3d        # fit anny's face shapes to ICT-FaceKit identities
uv run python -m anny.faces.authoring.detail        # data/faces/detail_shapes.safetensors, from the fit residuals
uv run python -m anny.faces.authoring.mediapipe_map # data/keypoints/mediapipe.json
uv run python -m anny.faces.authoring.calibrate     # data/shape_calibration/face_prior.safetensors
uv run python -m anny.faces.authoring.benchmark     # benchmark.html in ANNY_CACHE_DIR/faces
uv run python -m anny.faces.authoring.review        # review.png in ANNY_CACHE_DIR/faces: random faces to judge by eye
```

### Expression shapes
```bash
uv run python -m opensculptboy.expressions.authoring.ict --card card.json --out out.glb --eyes --teeth   # ICT-FaceKit's expression shapes carried onto Anny's face (--only NAMES, --smooth, --strength, --report FILE)
uv run python -m opensculptboy.expressions.authoring.faces --out faces.json  # the review's faces: 52 actions at 1.0 and 0.5, VRM presets and visemes, 20 expressions, colliding mixes
blender -b --factory-startup -P src/opensculptboy/expressions/authoring/blender_faces.py -- --glb out.glb --faces faces.json --out renders/ --helper src/opensculptboy/export/blender_skin.py   # front and 3/4, textured and clay
uv run python -m unittest test.test_expression_transfer test.test_expression_faces  # the transfer with synthetic data (the ICT data test is local only) and the review's faces
```

### Hair
```bash
uv sync --extra viewer
uv run python -m anny.hair.authoring.layout                 # data/hair/scalp_layout.safetensors
uv run python -m anny.hair.authoring.presets                # grow every style: data/hair/styles/*.json and styles.safetensors
uv run python -m anny.hair.authoring.presets --only lob     # grow some styles
uv run python -m anny.hair.authoring.presets --specs-only   # rewrite the specs alone (render, physics, controls)
uv run python -m anny.hair.authoring.benchmark              # load, slider and render times of the built page
uv run python -m anny.hair.authoring.review                 # review.png in ANNY_CACHE_DIR/hair: every style at three lengths on three bodies
uv run python -m anny.hair.authoring.photos                 # photos.html in ANNY_CACHE_DIR/hair: the hairline of every style against FairFace photos (needs --extra faces)
```

### Web viewer
```bash
uv sync --extra viewer                # scipy, tetgen, embreex for the build
uv run python -m anny.viewer build    # data (cached under ANNY_CACHE_DIR/viewer) and the page viewer/dist/anny_viewer.html
cd viewer && node build.mjs --reuse-data   # the page alone, with the model data of the last build (for changes to src/ or shell.html)
cd viewer && npx tsc --noEmit         # type-check the page
uv run --with playwright python -m anny.viewer.benchmark   # uploads, draws and vertices of a moving frame
uv run python -m unittest test.test_viewer_expressions     # the page's faces against Anny: actions, the jaw's turn, the sparse subdivision (node)
uv run python -m unittest test.test_viewer_sheet   # the character sheet: settings and layout in node, a sheet of the default character in the built page (local only)
uv run python -m unittest test.test_viewer_shortcuts   # the keyboard shortcuts: keys of other keyboards in node, every shortcut of the help after a person's steps in the built page (local only)
```

## Architecture

### Entry Points

`src/anny/__init__.py` exports the public API:
- `Anny(...)` — the model class, for full-body and part models alike; calling `anny.Anny(...)` builds a model and `isinstance(model, Anny)` holds for any Anny model. Accepts `rig`, `topology`, `local_changes`, `facial_actions`, `face_shapes`, `phenotypes`, `extrapolate_phenotypes`, `pose_parameterization`, and `skinning_method`.
- `AnnyInverter`, `KeypointsRegressor`, `Anthropometry` — see Key Subsystems.
- `create_fullbody_model(...)` — deprecated legacy full-body factory. It preserves the old default rig preset (`rig="default"`) and old full-body defaults; prefer `Anny(...)`.
- `create_hand_model()` / `create_head_model()` — deprecated part-model factories; use `Anny(rig="anny-hand.R", topology="hand.R")` or `Anny(rig="makehuman-head", topology="head")` instead.

### Core Class Hierarchy

- **`RiggedModelWithLinearBlendShapes`** (`models/rigged_model.py`) — base class; holds template vertices/faces/blend shapes, implements forward kinematics and LBS. The `bone_orientation` parameter (`"blender"`, `"procrustes"`, or `"cached"`) selects how rest bone orientations are computed: `"blender"` from bone tails, `"procrustes"` by aligning each bone to the vertices it skins, and `"cached"` by the same alignment from precomputed cross-covariance data (`data/cached/{anny,soma}.pth`, generated by `scripts/precompute_rig_caches.py`).
- **`Anny`** (`models/phenotype.py`) — inherits directly from `RiggedModelWithLinearBlendShapes`; adds the 9 phenotype dimensions (gender, age, muscle, weight, height, proportions, race, cupsize, firmness) and computes blend shape coefficients from these semantic scalars.
- **`SMPL`** / **`SMPLX`** (`models/smpl.py`) — first-class model types that also inherit directly from `RiggedModelWithLinearBlendShapes`; wrap the `smplx` library and follow the same initialization pattern as `Anny`, but accept `betas` + pose parameters instead of phenotype dimensions. Require the optional `smplx` package (`uv sync --extra smpl`).

### Rigs & Topologies

**Rigs** (`anny`, `makehuman`, `cmu_mb`, `game_engine`, `mixamo`, `soma`): MakeHuman-derived bone hierarchies are defined as JSON in `src/anny/data/mpfb2/rigs/standard/`; the `soma` rig lives in `src/anny/data/soma/`. `anny` is the default: the MakeHuman source rig with the `notongue`, `nobreasts`, `nofacialexpression`, and `pruned` modifiers, and `"cached"` bone orientations. OpenSculptBoy's `anny-mouth` (`opensculptboy.mouth`) is `anny` with MakeHuman's jaw and tongue chain kept, which `opensculptboy.mouth.rig` builds on anny's public builder. Rig specs accept `-`-separated modifiers, including the subtree selectors `head`, `hand.L`, and `hand.R` (for `anny` and `makehuman` only). `default` is a legacy preset accepted only by `create_fullbody_model(...)` and preserves the old full MakeHuman rig defaults. `makehuman` is the full MakeHuman rig with tail/blender orientation and `root_identity_orientation=True`. Rig orientation is part of rig resolution; public constructors do not accept a separate `bone_orientation` argument.

**Topologies**: `anny` (the default MakeHuman mesh, with nudity edits), OpenSculptBoy's `sculpt` and `sculpt-quads` (anny's mesh with patches, built by `opensculptboy.topology`, not by `anny.Anny`), `makehuman` (the unedited MakeHuman mesh, quads and unattached vertices kept), the part meshes `head`, `hand.L`, and `hand.R`, and the alternative topologies `smplx`, `smpl`, `soma`, `anny_from_soma`, `notoes*`, and `legacy_default`. `default` is accepted only by `create_fullbody_model(...)`. Alternative topologies are built by retopology (`models/retopology.py`) onto target meshes stored in `src/anny/data/topology/` and `src/anny/data/soma/`. The SMPL and SMPL-X topologies are non-commercial only.

### Key Subsystems

| Subsystem | Location | Purpose |
|-----------|----------|---------|
| Forward kinematics | `utils/kinematics.py` | Tree traversal with parallel propagation fronts |
| Skinning | `skinning/skinning.py`, `skinning/warp_skinning.py` | LBS and dual-quaternion skinning (`skinning_method="lbs"`, `"dqs"`, `"warp_lbs"` on the GPU) |
| Collision | `utils/collision.py` | Self-intersection detection, warp-accelerated |
| Model data | `models/model_data.py`, `models/model_transforms.py` | `ModelData` bundles the template mesh, blend shapes and rig, with safetensors caching; the transforms retopologise, convert bone orientations and clean meshes |
| Fitting and measures | `anny_inverter.py`, `keypoints.py`, `anthropometry.py` | `AnnyInverter` fits pose and shape to a mesh; `KeypointsRegressor` gives keypoints as vertex combinations; `Anthropometry` measures height, volume and mass |
| Facial actions | `models/facial_actions.py` | The 52 ARKit expressions from MakeHuman's face units; the web viewer plays them, `jawOpen` as the `anny-mouth` jaw's turn ([docs/viewer.md](docs/viewer.md#faces)) |
| Face shapes | `models/face_shapes.py`, `faces/` | 103 face shapes and 10 detail shapes, and the calibrated face distribution (see Face Shapes below; [docs/body-model.md](docs/body-model.md)) |
| Subdivision | `utils/subdivision.py` | Catmull-Clark as sparse linear operators; `MixedSubdivision` refines a region |
| Poses and correctives | `poses/`, `correctives/` | 50 poses and 7 clips as `local-ref` parameters; soft-tissue correctives driven by joint angles |
| Hair | `hair/` | Scalp layout, 25 styles, strand binding and the page's solver repeated in NumPy; see Hairline below |
| glTF export | `opensculptboy/export/gltf.py`, `document.py`, `body.py`, `mouth.py` | `export_glb` writes the rig, skin, morph targets and animations in glTF's frame (C = (x, y, z) -> (x, z, -y)); `clear_mouth` keeps Anny's mouth cavity behind the skin; `TestGltfReference` guards the bytes ([docs/export.md](docs/export.md)) |
| VRM export | `opensculptboy/export/vrm.py`, `tpose.py`, `vrm_tables.py`, `mtoon.py`, `budget.py` | VRM 1.0 and 0.x files with an exact T-pose rebind, refined skin weights, ARKit expressions, MToon materials and budgets; `test_bind_error` holds `MEASURED_BIND_ERROR`, so measure again when a change moves the bind ([docs/vrm.md](docs/vrm.md)) |
| Pose from a picture | `opensculptboy/posing/`, `opensculptboy/render/` | MediaPipe landmarks to an Anny pose, with flat SVG and PNG drawings; `viewer/src/pose_from_image.ts` repeats it and `test_viewer_pose_parity` checks both ([docs/pose-from-picture.md](docs/pose-from-picture.md)) |
| Reference fit | `opensculptboy/fit/` | Fits Anny to an appearance handoff: placement, parameters, then per-level non-rigid offsets with the ears, lids and lips handled apart; docs/reference-fit.md quotes the synthetic fit's numbers, so measure again when a change moves them ([docs/reference-fit.md](docs/reference-fit.md)) |
| Detail layer and textures | `opensculptboy/detail/`, `export/detail.py`, `texture/` | Subdivision levels 1 and 2 with offsets, the texture bake, head and ear tiles, MikkTSpace tangents and level-matched normal maps ([docs/reference-fit.md](docs/reference-fit.md)) |
| Skin material | `opensculptboy/material.py`, `export/skin_ramp.py`, `export/blender_skin.py` | `Character.material`: PBR skin with subsurface settings in the extras, an MToon skin ramp for VRM, and the Blender helper ([docs/character-card.md](docs/character-card.md)) |
| VRM budget fit | `opensculptboy/export/pieces.py`, `fit_budget.py`, `decimate.py`, `atlas.py`, `rebake.py`, `budget_quality.py`, `render/ids.py` | The report per piece in the VRM summary and budget messages, and `--fit-budget`: vertex-subset quadric reduction over every piece (seams, borders and morph targets kept, importance by priority and visibility), normal maps baked again from the full pieces, MToon atlases, joint pruning, texture shrinking, and quality measures against the full file ([docs/vrm.md](docs/vrm.md#fitting-a-file-into-its-budget)) |
| Budget tiers | `opensculptboy/tiers.py`, `texture/transfer.py`, `render/sheet.py` | Master, high, medium and low tiers cut down from the master, with baked normal maps and portraits; `test_budget_tiers` holds `MEASURED_LOW_ERROR`, so measure again when a change moves it ([docs/budget-tiers.md](docs/budget-tiers.md)) |
| Eyes | `opensculptboy/eyes/` | Eyeball, cornea, caruncle, iris texture and lashes that replace Anny's eyeballs in the exports, and blinks from a reference's lid joints, its lashes included, or from Anny's own; a fitted card has them by default ([docs/eyes.md](docs/eyes.md)) |
| Teeth | `opensculptboy/teeth/` | The reference's teeth or a generic set, appended after the body mesh ([docs/teeth.md](docs/teeth.md)) |
| Grooms | `opensculptboy/groom/`, `eyes/page.py` | A reference's hair and brows on a fitted card: read with their skins, vertex colours and alpha, carried through the fit's correspondence, written as `opensculptboy/groom@1` files (`material.hair`, `material.brows`); the hair as a mesh of its own with spring chains (spring bones in VRM), the brows as parts of the eyed mesh that follow the skin's morph targets; and the exports' eye targets for the web viewer (`opensculptboy/page-blinks@1`) ([docs/reference-fit.md](docs/reference-fit.md#hair-and-brows)) |
| Mouth | `opensculptboy/mouth/`, `fit/lips.py` | The `anny-mouth` rig, `jawOpen` as a jaw turn and corrective, a reference's inner mouth, sealed lips, the jaw's weights and mouth poses as targets; `test_mouth_rig` holds `MEASURED_ANGLE` and `MEASURED_EXPLAINED` ([docs/mouth.md](docs/mouth.md)) |
| Sculpt topology | `opensculptboy/topology/` | Anny's mesh with append-only patches (the eye corner on; the ear root off), built into `data/topology/sculpt.safetensors`, and the card converter ([docs/sculpt-topology.md](docs/sculpt-topology.md)) |
| Wearables | `opensculptboy/wearables/` | Wearable packages (`opensculptboy.wearable/1`, JSON Schema in `data/wearables/`, with a `provenance` guard), MakeHuman clothes import (`mhclo.py`, `basemesh.py`: base-mesh rows that Anny lacks come from the MPFB2 targets), quality checks and safe fixes (`quality.py`), card changes (`cardedit.py`), surface and rigid binding to Anny's template, the fit with layers and push-out, cover masks, outfits with slot replacement and hides rules, dressed GLB, wardrobe and VRM files, and reference wearables from a handoff's `equipment`, the catalogue (`catalogue.py`), the fit check with its review sheet (`check.py`, `render/dressed.py`) and outfit files (`outfitfile.py`, `opensculptboy.outfit/1`), spring chains with body colliders for loose parts (`wearables/springs.py`, `VRMC_springBone` and VRM 0.x `secondaryAnimation`); `TestNoPrivatePackages` fails on any private package under `src/` or `test/data/`. Agents that dress characters follow the skill `skills/outfits/SKILL.md` ([docs/wearables.md](docs/wearables.md)) |
| Statue scans | `opensculptboy/statues/`, `data/statues/` | One pipeline for every statue from a config file (`run_statue`, `opensculptboy.statue/1`: convert, landmarks found on pictures, preview over a grid, a pruning first fit and the fit with closed eyes and lips, measure); a statue scan as a handoff (`scan_to_handoff`: up and front axes, plinth, stature, masks for supports, breaks, drapery and hair, the licence in `statue`), `fit_statue` with lost parts left without data (`FitOptions.no_data`), the scan's open borders dropped and a start pose from the library (`FitOptions.initial_pose`), the hands and feet fitted one limb at a time (`chains.refine_chains`), the skin that coarse masks took given back (`restore_near`), the rims along the masks' cuts pruned (`prune_rim`), the hat, the locks and the props rebuilt as closed meshes in Blender (`blender_pieces.py`, `export_props` with a config's `prop_shapes`), skin given no data in `no_data_regions`, and distances per region of the figure ([docs/reference-fit.md](docs/reference-fit.md#statue-scans)) |
| Character card | `opensculptboy/character.py` | `Character`, the JSON card that every export reads and stores in its scene extras ([docs/character-card.md](docs/character-card.md)) |

### Phenotype System

Phenotypes are blended linearly between discrete anchor states defined in `src/anny/data/mpfb2/targets/`. Default mode omits race, cupsize, and firmness; pass `phenotypes="all"` to enable them. Blend shape data is computed at model creation and cached in `~/.cache/anny/`. Set the `ANNY_CACHE_DIR` environment variable to use a different location.

### Face Shapes

`Anny(face_shapes="all")` adds 103 named, symmetric face-shape parameters (`data/faces/face_shapes.json`, rows `face_shape:{name}.pos` and `.neg`) and 10 detail shapes from the ICT-FaceKit fit residuals (`data/faces/detail_shapes.safetensors`). The rows of a group scale with the size of that part of the head, measured on the craniofacial landmarks. The model cache key carries a digest of both files, and `scripts/precompute_rig_caches.py --append` adds rig-cache rows for new blend shapes and leaves the others bit for bit. `FaceShapeDistribution` samples faces from `data/shape_calibration/face_prior.safetensors` (sources in `data/faces/SOURCES.md`); only the skull-size shapes (`MEAN_SHAPES`) move its mean, and the facial features stay at Anny's default face.

Rules for agents: check every change to the calibration by rendering random faces (`python -m anny.faces.authoring.review`) as well as by the measurement check, since fits that meet the numbers can still give implausible faces. Judge each random face by whether it passes as a plausible character of that form, and show the calibrated mean in the grid. `test.test_faces_calibration.TestPlausibleFaces` guards these choices and the approved spread (`APPROVED_FACIAL_SPREAD`): when a change breaks one of its tests, review the grid with the user before updating the test.

### Hairline

Every style shares Anny's default hairline (`HAIRLINE_PHI` and `HAIRLINE_EL` in `hair/chart.py`), placed on Anny's landmarks and outer ears (`hair/authoring/anatomy.py`), and the style fields count degrees above it. A change to the table needs `python -m anny.hair.authoring.layout`, then `presets` and the viewer build; check it with `test.test_hair_styles.TestHairlineAnatomy`, the photo benchmark (`python -m anny.hair.authoring.photos`) and the review grid.

### Pose Parameterization

Five built-in variants: `local-ref` (the `Anny()` default), `local-bone`, `local-bone-world`, `world`, `world-orient`, selected by `pose_parameterization`. The deprecated `create_fullbody_model(...)` keeps the old `local-bone` default.

### Authoring Rig and Viewer Frame

The pose, corrective and hair authoring code (`*/authoring/`) and the viewer work in the frame of `poses/authoring/rig.py`: Anny's default body in metres, Y up, X toward the figure's left, Z forward, scaled by 0.8916. `authoring_rig(phenotype)` builds it for other sliders, and `ANNY_AUTHORING_PHENOTYPE` (a JSON object) sets the body of the authoring tools.

### Web Viewer

`viewer/` is a node project (TypeScript, three.js, esbuild) that builds the single-file page `viewer/dist/anny_viewer.html` ([docs/viewer.md](docs/viewer.md)). `src/anny_shape.ts` and `src/subdivision.ts` repeat Anny's maths, checked by `test/test_viewer_parity.py`; `src/face/` plays Anny's facial actions, `jawOpen` as the `anny-mouth` rig's jaw turn and a card's own mouth poses and eye targets behind the interface of `src/face/api.ts` (on the GPU in `face/morph.ts`), checked by `test_viewer_expressions`; `src/main.ts` holds the renderer, `src/ui/` the interface and `src/hair/` the hair, checked by `test_hair_parity`, `test_hair_page` and `test_hair_dynamics`. The page keeps one `<canvas>` (the screenshot tools select it), so the interface draws its graphs in SVG, and `?shot` hides the interface. `test/vrm_page/` is the three-vrm test page; nothing in `src/` imports its devDependencies. `python -m anny.viewer.benchmark` and the Performance card measure a moving frame.

### Dependencies

`warp-lang`, `trimesh`, and `requests` are core dependencies. Optional extras:

- `smpl` (`smplx`, `chumpy`) — required for the `SMPL` and `SMPLX` model classes; install via `uv sync --extra smpl`
- `examples` (`gradio`, `jsonargparse`, `matplotlib`, `scipy`, `jupytext`, `notebook`, `py-soma-x`, ...) — needed for examples, tutorials, and SOMA comparison tests. `py-soma-x` is pinned to match `anny.models.soma.SOMA_ASSETS_REVISION`.
- `viewer` (`scipy`, `tetgen`, `embreex`) — needed for the viewer build and the authoring tools; install via `uv sync --extra viewer` (the page build also needs node 22 or later)
- `faces` (`scipy`, `mediapipe`, `pyarrow`, `matplotlib`, `playwright`) — needed for the face calibration and the photo benchmark (`python -m anny.faces.authoring.*`); install via `uv sync --extra faces` (MediaPipe needs the system libraries `libegl1` and `libgles2`)
- `pose` (`mediapipe`, `contourpy`) — needed for the pose from a picture (`opensculptboy pose`, `opensculptboy.pose_from_image`) and the flat drawings; MediaPipe needs the system libraries `libegl1` and `libgles2`
- `dev` (`ruff`, `build`, `twine`, `jsonschema`) — formatting checks (`test/test_ruff.py`), packaging and the VRM schema checks (`test/test_vrm_schema.py`)

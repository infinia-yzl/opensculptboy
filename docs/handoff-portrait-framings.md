# Handoff: the layered portrait pack (`feat/portrait-framings`)

This branch extends the portrait pack export (`feat/portrait-pack`) for a 2.5D chat stage. This page is for the next maintainer: what the branch holds, how to land it, and what is left.

## 1. Landing the branch
1. The base of `feat/portrait-framings` is `feat/portrait-pack` (merge that first if it is not yet in the main line).
2. Run the tests: `python -m unittest discover -s test` (CI tier), and the local-only tier on a machine with a GPU and Chromium (`OPENSCULPTBOY_LOCAL=1`, see `AGENTS.md`).
3. Open a pull request into `feat/portrait-pack` (or the main line once that has merged), with a semantic title such as `feat: layered portrait packs with framings, poses and light passes (squire.portraits/2)`.

## 2. What the branch adds
- **The layered pack (`squire.portraits/2`):** body, face and front layers per pose and framing; seven framings, each rendered through one shot's camera (`--shots-file`); faces per framing with their own sets for poses that turn the head; the leap as four keys with its own frame and head size; joints of every layer in the manifest; lossy WebP with lossless alpha. Contract and check: `portrait_pack/layered_contract.py`, `docs/portrait-pack.md`.
- **Shared poses:** `--poses-file` reads one pose file in VRM humanoid bones (`portrait_pack/shared_poses.py`, a default in `data/portrait/poses.json`), with stances (contrapposto), head turns, the leap keys, and the upper arm's roll shared with its twist bone (`UPPER_ARM_ROLL`).
- **Light passes:** `--passes key,fill,rim` and `--light-rig stage` render each layer as linear key, fill and rim passes so a stage can relight it live; each layer is written as the sum of its passes; per-pack scales in the manifest.
- **Speed and resume:** expressions only where each framing needs them, per-framing samples, a writer pool, per-phase timing, and `--resume` that keeps only layers that still match the pack.
- **The viewer page:** eyes with the iris behind the cornea, a lid contact shadow and a crisp catchlight; skin with more scattering and transmission from the card's thickness map; a cloak that hangs from shoulder and hip chains in raised-arm poses; the front layer's mask that keeps a pose's own shading next to the face.

## 3. Known issues
- `test_portrait_pack.TestCommand.test_card_files` fails on Windows when the temp folder has a short (8.3) path name; the test compares two spellings of the same path.
- The viewer page cannot be built with esbuild under a Windows account that cannot read its parent folders; build it elsewhere and copy `viewer/dist/anny_viewer.html`.

## 4. Left to do, in order
1. **Raised-arm skin (the main task).** With the upper arm raised past about 120°, linear blend skinning stretches the armpit skin about 2.4 times (up to 5.5) and the texture there smears. Sharing the arm's roll helped the shear; it did not fix the stretch. Moving more of the lift into the clavicle made it worse. Add a fix in the body for every character: smoother armpit weights and/or a corrective shape driven by upper-arm elevation (and, if it helps, a helper bone). Measure the stretch before and after on the default character at the poses `cheer`, `cheer_peak`, `hands_behind_head`, `scratch_head` and `wave`, with a UV checker, and keep the joints of the shared poses unchanged (the shared-pose tests check them).
2. **The VRM export keeps more of the materials:** the metallic-roughness maps, the skin's roughness map and the hair's vertex colours (VRM 1.0 keeps glTF's material fields beside MToon), so real-time viewers match the stills without reading the card.
3. **Capes and long garments:** weights that follow the spine and shoulders in raised-arm poses, generalising the page's cloak hang into the wearable package.
4. **The Windows temp-path test** in section 3.

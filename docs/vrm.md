# VRM avatars

`opensculptboy export` writes a VRM avatar when the output file ends in `.vrm`. VRM 1.0 is the default, for Warudo, VMagicMirror and three-vrm, and `--vrm-version 0` writes VRM 0.x for VSeeFace and 3tene.

```bash
opensculptboy export ada.vrm --character ada.json --author "Your name"                   # VRM 1.0
opensculptboy export ada0.vrm --character ada.json --author "Your name" --vrm-version 0  # VRM 0.x
opensculptboy export ada.vrm --author "Your name" --thumbnail auto --meta meta.json      # a rendered portrait, and licence fields from a file
```

```python
from opensculptboy import Character, export_vrm

export_vrm("ada.vrm", Character(name="ada", phenotype={"muscle": 0.7}), author="Your name")
```

## What the file holds

The avatar rests in the VRM T-pose: the arms, hands and fingers point straight out to the sides with the palms down, the thumbs point level at 45 degrees toward the front, the legs hang straight and the feet point forward. The file holds:
- the mesh with 4 skin weights per vertex, and the humanoid bone map of the `anny` rig (with `jaw` on the `anny-mouth` rig, [Mouth](mouth.md));
- expressions mixed from the 52 ARKit facial actions: the visemes aa, ih, ou, ee and oh, blink, blinkLeft and blinkRight, the emotions happy, angry, sad, relaxed and surprised, and neutral; in VRM 1.0 files an emotion damps blinking and lip sync in proportion to its weight;
- the 52 facial actions as perfect-sync expressions named in PascalCase (`EyeBlinkLeft`), and in VRM 0.x files the VSeeFace visemes SIL, CH, DD, FF, KK, NN, PP, RR, SS and TH;
- look-at on the eye bones, and MToon toon materials;
- the licence metadata with the credits, and an optional thumbnail;
- the [character card](character-card.md) in the scene extras, so that `read_character("ada.vrm")` reads it back.

The face shapes are baked into the mesh, because a VRM skeleton is fixed, and the facial actions become the expressions. The eyes of `--eyes`, the teeth of `--teeth` and a fitted card's inner mouth go into the file as they do for [glTF](export.md).

## Metadata and licence

A VRM file needs an author: pass `--author`, or `"authors"` in a `--meta` file of `VrmMeta` fields. The licence fields start restrictive: only the author may use the avatar, for personal non-profit use, without redistribution or modification. `--meta` changes them. The `allow_*` fields take a JSON `true` or `false`; `avatar_permission`, `commercial_usage`, `credit_notation` and `modification` take one of their VRM 1.0 values, such as `"commercial_usage": "personalProfit"` (an invalid value prints the choices). `--thumbnail auto` renders a square portrait, and a given thumbnail must be a square PNG or JPEG file. Every file credits OpenSculptBoy, Anny (NAVER Corp., Apache 2.0), the MakeHuman, MPFB2 and Face Units assets (CC0) and ICT-FaceKit (MIT).

## Rig, bind and skin weights

A VRM file needs the `anny` rig (or `anny-mouth`) and the MakeHuman body mesh with its eyes: the `anny`, `makehuman` or [sculpt](sculpt-topology.md) topology, with modifiers such as `-quads` or `-full`. The export refuses the SMPL and SMPL-X topologies, whose data is non-commercial, and the retopologised meshes.

VRM apps turn only the humanoid bones. A VRM 1.0 file therefore turns the forearm twist bones with roll constraints from the wrists (`--twist constraint`), and merges the other twist weights into their bones. A VRM 0.x file has no constraints, so it merges every twist weight (`--twist merge`).

- `--bind inverse` (the default) gives the mesh that the file's 4 skin weights pose back onto the character's own rest pose exactly. In arms-down poses, the usual idle of a VTuber avatar, 99 % of the vertices land within about 8 mm of the character's own posed mesh.
- `--bind forward` skins the character into the T-pose with its own weights, which gives a smoother T-pose and suits poses with raised arms.
- `--keep-leg-spread` keeps the legs at the rig's natural spread.

Straightening the arms, the fingers and the legs into the T-pose folds the skin where two neighbouring bones turn toward each other: the webs of the fingers and the thumb, and the groin. The export refines the file's skin weights for each character around those places, which removes about 80 % of those folds and adds a few seconds to an export (`--no-refine-weights` turns it off).

## Budgets

Each export prints its counts and checks the budget of its version: 70,000 triangles, 200 joints, 16 materials and 4096 px textures for VRM 1.0, and 32,000 triangles, 128 joints, 8 materials and 2048 px textures for VRM 0.x. A fitted card's finer detail levels step down to stay within it ([Fit to a reference](reference-fit.md)). `--budget strict` fails a file over its budget with exit status 1, and `--budget off` skips the check. Invalid options, metadata, thumbnails or cards end the command with one error line and exit status 2.

## Testing

The files pass the VRM 1.0 and 0.x schemas (`test/test_vrm_schema.py`). three-vrm loads both versions with their look-at and expressions, and the Khronos glTF validator reports no errors (`test/test_vrm_three.py`). VSeeFace, Warudo, VMagicMirror and UniVRM are untested so far.

## Not in the file yet

Hair, eyelashes, outfits and spring bones. Painted textures come only from a reference bake; other characters get a plain skin colour.

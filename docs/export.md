# glTF export

`opensculptboy.export_glb` writes a character as a binary glTF 2.0 file (`.glb`), the open format that Blender, Godot, Unity, Unreal Engine and three.js import.

```python
from opensculptboy import Character, export_glb, read_character

character = Character(name="ada", phenotype={"muscle": 0.7, "weight": 0.6}, facial_actions={"mouthSmileLeft": 0.4})
export_glb("ada.glb", character, animations=["walk", "wave", "seated"])
read_character("ada.glb")  # the same Character, from the extras of the file
```

The same export from the command line, with a [character card](character-card.md):

```bash
opensculptboy character > ada.json                # a card with the default settings; edit its values
opensculptboy export ada.glb --character ada.json --animation walk --animation wave
opensculptboy export ada.glb --morph-targets all  # the facial actions and all 113 face shapes
opensculptboy names poses                         # the 50 poses and 7 clips of the library
```

## What the file holds

- The mesh in the rest pose of the character, with normals and UVs, standing on the floor. Axes: metres, Y up, the figure faces +Z.
- The 104 bones of the `anny` rig as a skeleton, with the skin weights of the 4 strongest bones of each vertex (`--influences 8` adds a second set).
- Morph targets named after the 52 ARKit facial actions (`jawOpen`, `eyeBlinkLeft`, ...), and on request the face shapes (`<name>.pos` and `<name>.neg`).
- One animation for each pose or clip of `anny.poses` that you name, after one named `pose` when the character holds its own pose.
- The character card in the scene `extras`, so that `read_character` reads it back, and the Anny vertex of each glTF vertex in the `_ANNY_VERTEX` attribute.

The phenotype, the local changes and the face shapes that are not morph targets are baked into the mesh and the skeleton.

## Options

- `--detail N` exports a level of a fitted card's detail layer ([Fit to a reference](reference-fit.md)).
- `--eyes [FILE]` replaces anny's plain eyeballs with an iris texture, a refracting cornea and a caruncle ([Eyes](eyes.md)).
- `--teeth [FILE|off]` adds teeth: the reference's teeth for a fitted card, else a generic set ([Teeth](teeth.md)).
- A card fitted to a reference with an inner mouth carries that mouth, a jaw bone and the reference's mouth poses ([Mouth](mouth.md)).
- `--tier NAME` writes one of the rendering budgets of a fitted card ([Budget tiers](budget-tiers.md)).
- `blender --python src/opensculptboy/export/blender_skin.py -- ada.glb` imports a file into Blender and sets the skin's subsurface and the eyes' cornea from its extras.

## Accuracy and testing

With 4 bones per vertex, 99 % of the vertices lie within 4 mm of Anny's own output on the walk, run, arms-crossed and seated animations; with `--influences 8`, every vertex lies within 0.2 mm. The morph targets match Anny exactly. `test/test_gltf_export.py` evaluates the file as an engine does and checks it against Anny. The Khronos glTF validator reports no errors and no warnings, and three.js (`GLTFLoader`) draws the skeleton, the morph targets and the animations. Blender, Godot, Unity and Unreal Engine are untested so far.

## Not in the file yet

Hair and the soft-tissue correctives: the export uses plain skinning. Painted textures come only from a reference bake ([Fit to a reference](reference-fit.md)); other characters get a plain skin colour.

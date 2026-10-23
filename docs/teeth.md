# Teeth

Anny's mouth holds a cavity and a tongue but no teeth, so an open mouth shows an empty dark hole. A card whose `material` section holds `teeth` (`opensculptboy.material.TeethMaterial`, `opensculptboy.teeth`) adds teeth to the GLB and VRM exports:

- a fitted card takes the teeth of its reference: `opensculptboy fit --textures` and `opensculptboy bake` find them among the reference's mouth parts and write them next to the card (`<card>.teeth.safetensors`), in the card's rest frame, with the colour and roughness of their material;
- any other card takes generic teeth: two arches of seven teeth a side behind the lips, with gums along their roots.

Cards without the section export exactly as before (`TestGltfReference` in `test/test_gltf_export.py` holds the files to their recorded fingerprints).

```bash
opensculptboy export character.glb --character character.json --teeth               # the card's teeth, else generic teeth
opensculptboy export character.glb --character character.json --teeth teeth.json    # teeth.json laid over them
opensculptboy export character.glb --character character.json --teeth off           # no teeth, whatever the card says
opensculptboy export character.vrm --character character.json --teeth --author "Your name"
```

```python
from opensculptboy import Character, export_glb

card = Character(material={"teeth": {"color": [0.7, 0.65, 0.55]}})
export_glb("character.glb", card)
```

## The card

Every key of `material.teeth` is optional; colours are linear RGB. A `material` section that holds nothing but `teeth` (and `eyes`) keeps the exporters' own skin. The section belongs to character schema 3 and reads back from the files (`read_character`).

| Key | Default | Meaning |
|---|---|---|
| `source` | `"auto"` | `"auto"`: the reference's teeth when `mesh` names them, else the generic teeth; `"reference"` (an error without `mesh`); `"generic"` |
| `mesh` | none | the reference's teeth that the bake writes, as a path relative to the card |
| `color` | `[0.72, 0.66, 0.56]` | the enamel, a warm off-white; the bake writes the reference's base colour factor |
| `roughness` | 0.35 | the enamel's roughness; the bake writes the reference's |
| `gums` | true | generic teeth: the gums along the roots |
| `gum_color` | `[0.45, 0.13, 0.12]` | the gums |

## How the teeth move

The upper teeth move rigidly with the head, and the lower teeth move rigidly with the jaw. Anny's rig has no jaw bone: its jaw actions (`jawOpen`, `jawLeft`, `jawRight`, `jawForward`) are morph targets, and they turn the tongue with the jaw. So the exports carry the file's rest mesh, bind mesh, morph targets, normals and skin weights over to the teeth (`opensculptboy.teeth.mesh.ToothedMesh`) by these rules:

- every tooth follows the rigid motion (Kabsch) of the head, the vertices that the `head` bone skins alone, which keeps the teeth in place in the T-pose of a VRM file;
- in each morph target, the upper teeth take the rigid motion of the head's vertices that no target of the file moves (none for the facial actions, so they stay still), and the lower teeth take the rigid motion of the tongue when the target moves the tongue rigidly (its fit leaves under 10 % of its motion: `jawOpen` leaves 4 %), else the head's (`tongueOut` bends the tongue and leaves 23 %);
- the teeth take the mean skin weights of the head's vertices, which is the `head` bone.

The teeth share the body's vertices, morph targets and skin, as the eyes do, and draw with materials of their own: in glTF `teeth` (the enamel with a clear coat for the wet surface) and `gums` (pink, wet), whose extras hold `opensculptboy_teeth` (`opensculptboy/teeth@1`, with the role and the settings); in VRM files MToon materials of the same names. The scene extras record the teeth (`teeth`: the source, the colour, the counts).

A card with `material.mouth` (a reference's whole inner mouth, [mouth.md](mouth.md)) takes the reference's own teeth with its gums, palate, tongue and throat, and no teeth of this section.

## The teeth of a reference

An appearance handoff gives the inside of the mouth as parts with the role `mouth` (their material or part names name teeth, a tooth, the tongue, the gums or the mouth's lining; see [reference-fit.md](reference-fit.md)). The fit leaves them out, and the bake (`opensculptboy.teeth.reference.reference_teeth`) keeps every connected piece of those parts that is no larger than 15 mm across, as a game draws one tooth (a crown, or a crown and its root). The larger pieces (the gums, the tongue, the lining) stay out, since anny's own mouth holds a cavity and a tongue. The pieces fall into the upper and the lower arch by their heights (two clusters), and move from the reference's frame into the card's rest frame by the rigid motion that takes the fitted head, as the fit placed and posed it, back to its rest pose. The file `<card>.teeth.safetensors` (`opensculptboy/teeth@1`) holds their points, normals, UVs and triangles, the arch of each point and the part of each triangle.

## Generic teeth

`opensculptboy.teeth.generic.generic_teeth` draws the set from a handful of numbers and needs no third-party assets: per arch the central and lateral incisors, the canine, two premolars and two molars a side, each crown a rounded box (a superellipse in section) that thins toward the biting edge of the incisors and the canines, and a half-tube of gums along the roots. The arches follow a parabola whose front lies 2.5 mm behind the inner margin of anny's lips (the border of the base mesh's `mouth_cavity`), at the height where the lips meet; the set scales with the width of that margin. The upper front teeth overlap the lower ones by 2 mm, 3 mm in front of them, and the back teeth meet. Anny's lips keep a slit of about 2.5 mm between them at rest, so the edges of the upper front teeth can show through it, as they do on many faces.

## Checks

`test.test_teeth` holds the card and its validation, the teeth file, the generic set behind the closed lips (no tooth in front of the lips' inner margin, and a ray from every tooth toward the front meets the face unless it leaves through the slit), the toothed mesh (the upper teeth still in `jawOpen`, `jawLeft`, `tongueOut` and `mouthSmileLeft`, the lower teeth rigid in the jaw actions and still in the others, the head's skin weights), the reference's teeth brought back to the rest frame and split into arches, the GLB file (materials, extras, the teeth against the file's own `jawOpen` as an engine evaluates it), the VRM files of both versions and the command's `--teeth`.

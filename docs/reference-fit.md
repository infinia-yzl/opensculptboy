# Fit to a reference

`opensculptboy fit` rebuilds an existing humanoid 3D character on Anny's body. It fits Anny's parameters and pose to a target mesh of any topology, and it keeps what the parametric body cannot reach as a detail layer: per-vertex offsets on subdivisions of Anny's mesh. The target is any skinned or static triangle mesh in metres, given as a handoff folder.

```bash
uv sync --extra viewer                                          # SciPy for the fit, embreex for the texture bake
opensculptboy fit handoff/ --out character.json --detail 1      # the card, character.detail0.safetensors and character.detail1.safetensors
opensculptboy fit handoff/ --out character.json --textures      # also the baked textures, eyes, lashes, teeth and inner mouth
opensculptboy export character.glb --character character.json --detail 1
opensculptboy export character.vrm --character character.json --detail 1 --author "Your name"
```

```python
from opensculptboy.fit import fit_handoff

result = fit_handoff("handoff/", detail=1)
print(result.errors[1]["reverse"]["mean"])   # mean distance from the target to the fitted surface, in mm
result.write("character.json")
```

## The handoff folder

```
handoff/
    replica.glb        the skinned parts in their bind pose; metres, Y up, glTF axes (the figure faces +Z)
    textures/*.png     the textures of the parts
    appearance.json    {"landmarks": {name: [x, y, z]}, "parts": ..., "materials": ..., "masks": ..., "mouth": ...}
```

`opensculptboy.fit.handoff.load_handoff` reads every primitive of `replica.glb` as one part, in its bind pose, and turns it into Anny's frame (metres, Z up, the figure facing -Y). Each part keeps its names, UVs, file normals and its base colour, normal and roughness textures.

### Parts and roles

`parts` is a list of objects, or an object keyed by part name. Each entry names a mesh or node of the GLB file (`"name"`, `"mesh"`, `"node"` or `"node_name"`) and may give a `"role"`; an entry of an object without one takes its key as the role. Entries of `materials` may give a role per material. The role of a primitive is, in this order: its material's declared role, the role that its material's name names, its part's declared role, the role that its own name names, else skin. The roles are `skin` (also `body`, `head`, `face`, `hands`, `feet`), `lashes`, `brows`, `beard`, `hair`, `mouth` (teeth, tongue, gums), `eyes`, `nails` and `clothing`. Only the skin takes part in the fit. The eye parts that are spheres place Anny's eyeballs and give the eye texture. The optional `mouth` section describes the inner mouth ([Mouth](mouth.md)), and the optional `equipment` section the clothing that `opensculptboy wearables reference` turns into reference wearables ([Wearables](wearables.md)).

### Landmarks

The landmarks are optional. Names match loosely (`LeftEyeOuter`, `eye.outer.L` and `left_eye_outer` all mean `eye_outer_l`), and `l` and `r` are the figure's own left and right. Unknown names are ignored.

| Landmark | Counterpart on Anny | Weight |
|---|---|---|
| `eye_outer_l`, `eye_outer_r` | exocanthion (`ex`) | 1 |
| `eye_inner_l`, `eye_inner_r` | endocanthion (`en`) | 1 |
| `nose_tip` | pronasale (`prn`) | 1 |
| `mouth_corner_l`, `mouth_corner_r` | cheilion (`ch`) | 1 |
| `chin` | gnathion (`gn`) | 1 |
| `ear_l`, `ear_r` | tragion (`t`) | 0.5 |
| `ear_top_l`, `ear_top_r` | the highest point of the ear: the tip of a pointed ear | 0.2 |
| `ear_lobe_l`, `ear_lobe_r` | the lowest point of the lobe | 0.2 |
| `ear_back_l`, `ear_back_r` | the hindmost point of the ear | 0.2 |
| `ear_lateral_l`, `ear_lateral_r` | the point of the ear furthest from the head | 0.2 |
| `crown` | vertex (`v`), the top of the head | 0.5 |
| `neck_base` | the head of bone `neck01` | 0 |
| `shoulder_l`, `shoulder_r` | the head of `upperarm01` | 0.1 |
| `elbow_l`, `elbow_r` | the head of `lowerarm01` | 0.2 |
| `wrist_l`, `wrist_r` | the head of `wrist` | 0.2 |
| `hip_l`, `hip_r` | the head of `upperleg01` | 0 |
| `knee_l`, `knee_r` | the head of `lowerleg01` | 0.2 |
| `ankle_l`, `ankle_r` | the head of `foot` | 0.2 |

The face landmarks are points of the skin, and the body landmarks are joint centres, which sit differently in every rig; the weights reflect that. A source may give a landmark twice, as a joint position and as the nearest point of the skin (`mouth_corner_l_skin`): the face takes the point of the skin. Without landmarks, the fit places the target by its bounding geometry (the crown, the fingertips and the soles), which assumes an upright figure facing +Z.

### Ear outline and ear mask

A pointed ear far from Anny's round one needs the four ear outline landmarks above, which a source can take from its head skin as the extremes of the outer ear's vertices. An ear that stands off from the head also needs its mask, `masks.ear`: one entry per skin part with an ear, with the indices of the outer ear's vertices and, optionally, of the line where the ear meets the head:

```json
"masks": {"ear": [{"mesh": "head", "vertices": [812, 813, 860], "attachment": [812, 860]}]}
```

Without `attachment`, the fit takes the border of the mask. With the mask, the ears fit apart from the head (stage 4 below).

## The pipeline

1. **Placement.** A rigid alignment of Anny's landmarks onto the target's (Umeyama 1991), or of the bounding landmarks. The height slider starts at the target's stature.
2. **Parameters.** Adam over the phenotypes, the local changes, the 113 face shapes, the bone turns and a rigid placement, in 10 rounds that each find the closest points again. Where a face shape and a local change move the same MakeHuman targets, only the face shape moves (`FitOptions.shared_targets`). The fingers stay at rest until round 7 (`FitOptions.finger_rounds`), so that a finger cannot slide onto its neighbour.
3. **Non-rigid refinement.** The whole mesh moves onto the target by a non-rigid ICP (Amberg et al., CVPR 2007), with a Laplacian stiffness that falls over four stages and a reverse term that pulls the surface past its borders, such as the tip of a pointed ear. The eye sockets, the inside of the mouth, the eyeballs and the tongue take no data and follow the skin; the eyeballs move onto the target's eyeball spheres. The fit seals the closed lips where the target's lips meet ([Mouth](mouth.md)) and repairs the folds that the offsets add around the mouth. Last, the vertices within 2 mm of the target move onto it (`project_onto`), and `fair_skin` smooths the head's skin on the target, so that a fine level keeps none of the facets of a coarse target.
4. **Ears.** With `masks.ear`, Anny's outer ears match the target's ears alone and the rest of Anny matches the rest of the target, and the root of the ear takes the attachment line (`FitOptions.ear_split`). The ears and the skin around them are then faired on the target, and folds that the target lacks are repaired.
5. **Lids.** With the target's eyeballs, `seat_lids` places the lids, the sockets and the inner eye corners on the target's skin outside the eyeball and on the eyeball where the skin goes under it, so that the lids meet the exported eyeball flush (`FitOptions.seat_lids`).
6. **Detail levels.** Each level of the detail layer repeats the refinement on its subdivided mesh, starting from the level below.
7. **Residuals.** The registered mesh of each level goes back to the rest frame by the inverse of each vertex's blended bone transform; the offsets are the registered rest mesh minus the parametric rest mesh.

`--topology sculpt` fits the [sculpt topology](sculpt-topology.md) in place of Anny's mesh, and `opensculptboy convert` carries a fitted card between both meshes.

The card holds the fitted parameters, the target's bind pose (which the GLB export plays as the animation `pose`), and the detail layer with the placement and the measured errors ([Character card](character-card.md)). The code lives in `src/opensculptboy/fit/`.

## The detail layer

Level 0 is Anny's own mesh. Levels 1 and 2 are one and two levels of Catmull-Clark subdivision of its quad mesh (`anny.utils.subdivision`). Because the subdivision is linear, it carries the rest mesh, the morph targets, the skin weights (each vertex keeps its 12 strongest bones) and the UVs to the finer levels. `opensculptboy.detail.DetailModel` applies a level: it runs the Anny model, subdivides the rest mesh, adds the offsets after the blend shapes and skins the result. The offsets belong to the card's own slider values; when a slider changes later, they stay where they were in the rest frame.

| Mesh | Vertices | Triangles |
|---|---|---|
| level 0 (Anny's mesh) | 13,718 | 27,420 |
| level 1 on the head and the neck, level 0 elsewhere | 27,644 | 55,272 |
| level 1 | 54,848 | 109,680 |
| level 2 on the head and the neck, level 1 elsewhere | 110,504 | 220,992 |
| level 2 | 219,368 | 438,720 |

`export_glb`, `export_vrm` and `opensculptboy export` take `--detail` (default 0). A GLB file takes the full level. A VRM file keeps within the triangle budget of its version: when the full level is over it, the head and the neck alone take the level, and when that is still over it, the level steps down, with a `BudgetWarning` (`--budget off` keeps the full level). For several levels of detail at once, see [Budget tiers](budget-tiers.md).

## Textures

`--textures` bakes the target onto Anny's UV atlas, on the mesh of the finest level, and `opensculptboy bake handoff/ --card character.json` bakes a fitted card again without fitting. For each texel, a ray along the fitted surface's normal (Embree) finds the target's skin within 10 mm. Texels that no ray reaches take the nearest reached value. The target's vertex colours are never read, since a source may keep other data in them.

| Map | Content |
|---|---|
| `base_color` | the target's base colour texture |
| `normal` | the target's shading normal (its file normals, bent by its normal map) in the MikkTSpace frames of the fitted surface |
| `roughness` | the G channel of the target's metallic-roughness texture times its roughness factor |
| `thickness` | from the fitted mesh: the distance through it along the inward normal, for translucency |
| `curvature` | from the fitted mesh: the mean curvature, for a pre-integrated skin lookup (Penner 2011) |

The shading normal is the target's own, because artists smooth a character's file normals so that its surface reads differently from its bare geometry; a bake of the bare geometry gives a harder jaw and a plastic skin. The exporters write the same MikkTSpace frames as `TANGENT` (`opensculptboy.texture.tangents`, a NumPy port of Mikkelsen's algorithm that matches Blender's tangents), and they re-encode the normal maps for a coarser exported level (`opensculptboy.texture.levels`), so that every level shades alike.

Three tiles give parts of the body textures of their own, each drawn as a primitive of the body mesh with its own material (the same vertices, morph targets and skin):

| Tile | Size | Covers |
|---|---|---|
| eyes | 2048 by 1024 | both eyeballs, through a texture transform |
| head | 2048 by 4096 | the head's UV island, through a texture transform |
| ears | 2048 by 1024 | the outer ears on levels 1 and 2, with UVs of their own unwrapped from the fitted ears |

The texture transforms use `KHR_texture_transform` in glTF and VRM 1.0, and the ST vector of `_MainTex` in VRM 0.x. The bake also writes the target's eyes straight onto the layout of the exported eyes (`eye_layout`, 2048 by 1024 by default, `--eye-size`, with their normals), with each eye's gaze, and the fitted card's exports show them without `--eyes` ([Eyes](eyes.md)). It takes the target's lashes ([Eyes](eyes.md#lashes)), teeth and inner mouth ([Teeth](teeth.md), [Mouth](mouth.md)) as well, and the target's blinks when the handoff has them ([Eyes](eyes.md#a-references-blinks)).

## The skin material in the exports

A card's skin material ([Character card](character-card.md)) holds the baked maps and the values that a renderer needs to rebuild the skin; `--material skin.json` lays further values over the baked ones.

- **glTF.** The skin, head and ear materials are metallic-roughness materials with `KHR_materials_specular` and `KHR_materials_ior`. glTF has no ratified subsurface extension, so the subsurface settings and the thickness and curvature maps go in each skin material's extras under `opensculptboy_skin`. `blender --python src/opensculptboy/export/blender_skin.py -- character.glb` imports a file into Blender and sets the Principled BSDF's subsurface from them.
- **VRM.** MToon has no subsurface scattering, so the export fits MToon's ramp and shade colour to the falloff of the card's subsurface settings (`opensculptboy.export.skin_ramp`). VRM files hold the base colour and the normal map of each tile.

## Accuracy

`test/test_reference_fit.py` fits a synthetic target: Anny with random sliders, local changes, face shapes and a small pose, smooth random offsets, a random placement and a remeshed surface of another topology, with 1 mm of noise on its landmarks. The distances are in millimetres, from the fitted vertices to the target (forward) and from 20,000 target vertices to the fitted surface (reverse):

| Stage | Forward mean | Reverse mean | Reverse p95 |
|---|---|---|---|
| parameters alone | 0.99 | 1.05 | 2.72 |
| level 0 | 0.02 | 0.40 | 1.02 |
| level 1 | 0.01 | 0.13 | 0.33 |
| level 2 | 0.00 | 0.03 | 0.09 |

The last stage moves the vertices onto the target, so the reverse errors carry the measure. The fit to level 1 takes about 75 s; level 2 adds about 180 s and about 4 GB of memory. A synthetic target is the best case: a real character leaves larger residuals in the parameters and leans more on the detail layer. `TestPointedEars` fits a target with pointed ears that stand off from the head, with and without `masks.ear`.

`opensculptboy.fit.metrics.measure` measures a fitted mesh as it is, per region: signed distances both ways, the angle to the target's geometric and file normals, triangles that bridge a gap, folds that the target lacks, faces turned away, and with an ear mask the skin around the ears and the strip along their root. `fit_levels` records these measures in `FitResult.measures`, and `opensculptboy.fit.views` renders grey-shaded pictures and heat maps of them.

## Limits

- Hair, eyelashes, brows and clothing are left out of the fit; the bake takes the eyelashes for the exports.
- The inside of the mouth and the eye sockets follow the skin around them. A handoff that describes its inner mouth gives the exports the target's own gums, palate, tongue, throat and teeth ([Mouth](mouth.md)).
- An ear far from Anny's shape needs the ear outline landmarks, and an ear that stands off from the head needs `masks.ear`.
- The body landmarks are joint centres of another rig, so they guide the pose and the proportions loosely; the face landmarks carry most of the placement. A target in a pose far from Anny's (arms raised high, crouching) needs landmarks on the elbows, the wrists and the knees.
- The offsets belong to the card's slider values: changing a slider afterwards moves the parametric body under fixed offsets.
- The bake carries the target's base colour, roughness and shading normals; the subsurface values come from the card.
- The tip of a pointed ear packs its texels closer on the ear tile, because the tip is a cone that no flat map spreads evenly.

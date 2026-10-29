# Fit to a reference

`opensculptboy fit` rebuilds an existing humanoid 3D character on Anny's body. It fits Anny's parameters and pose to a target mesh of any topology, and it keeps what the parametric body cannot reach as a detail layer: per-vertex offsets on subdivisions of Anny's mesh. The target is any skinned or static triangle mesh in metres, given as a handoff folder.

```bash
uv sync --extra viewer                                          # SciPy for the fit, embreex for the texture bake
opensculptboy fit handoff/ --out character.json --detail 1      # the card, character.detail0.safetensors and character.detail1.safetensors
opensculptboy fit handoff/ --out character.json --textures      # also the baked textures, eyes, lashes, teeth, inner mouth, hair and brows
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

The texture transforms use `KHR_texture_transform` in glTF and VRM 1.0, and the ST vector of `_MainTex` in VRM 0.x. The bake also writes the target's eyes straight onto the layout of the exported eyes (`eye_layout`, 2048 by 1024 by default, `--eye-size`, with their normals), with each eye's gaze, and the fitted card's exports show them without `--eyes` ([Eyes](eyes.md)). It takes the target's lashes ([Eyes](eyes.md#lashes)), teeth and inner mouth ([Teeth](teeth.md), [Mouth](mouth.md)) as well, its hair and brows (see "Hair and brows" below), and the target's blinks when the handoff has them ([Eyes](eyes.md#a-references-blinks)). It also writes the eye targets of the card's GLB export at the level of its bake for the web viewer (`<card>.blinks.safetensors`, `opensculptboy/page-blinks@1`, named by `material.reference_eyes.blinks`; [Web viewer](viewer.md#a-cards-look)).

## Hair and brows

The parts of the roles `hair` and `brows` stay out of the fit, and the bake carries them onto the card's head (`opensculptboy.groom`):

- **Reading.** `opensculptboy.groom.handoff` reads every hair and brow primitive of `replica.glb` straight from the file, with its UVs, its base colour and normal textures, its vertex colours (`COLOR_0`, which holds a game hair's colour while its texture holds the strands' alpha), its material's alpha mode, cutoff and sides, and its skin: the weights and the bind position and parent of each joint. The primitives that `hidden` names stay out.
- **The carry.** Each point moves with the fitted head as the fit placed and posed it: the rigid motion (Kabsch) of the head's vertices back to their rest pose, plus two fields of the head's vertices blended at the point by Gaussian weights of its 16 nearest head vertices (at a scale of at least 4 mm): the head's residual from that rigid motion, and the offset from the target's skin to the fitted surface, taken from the head's vertices within 3 mm of the target's skin (the vertices inside the head, in the mouth's cavity and the eye sockets, have no skin of the target near them). A brow strip that lies 0.5 mm over the target's skin therefore lies 0.5 mm over the card's. The hair, whose skin reaches the neck and the shoulders, takes that offset and then the inverse of the fit's pose, the bone transforms of the fitted body blended with the hair's own weights on Anny's bones, so that the exports, which pose the hair with those weights, lay the strands over the nape where the target has them.
- **The hair's skin.** The target's joint with the most weight over the hair is the head. The joints below it that weight the hair are chain joints: their weight goes to Anny's `head`, and each chain becomes a spring chain from its first joint down to a tail one segment past its last joint. Every other joint becomes Anny's nearest bone among the upper spine, the clavicles, the neck and the head (a nape's weight on the neck stays on the neck).
- **Drawing order.** The triangles of a blended material run from the inside out, by the distance of their centres from the centre of the head, so that an engine that draws them in order draws the outer cards over the inner ones from any view outside the head.
- **Files.** `<card>.hair.safetensors` and `<card>.brows.safetensors` (`opensculptboy/groom@1`) hold the points, normals, UVs (glTF convention), triangles, the material of each triangle, the vertex colours, the materials and, for the hair, its skin on Anny's bones, its chains and each point's share of weight on chain joints. The textures go next to them as `<card>.<kind>.<material>.<slot>.png`. The card's `material.hair` and `material.brows` sections name them (`{"source": "reference", "mesh": ..., "textures": {material: {slot: path}}}`); `false` leaves a groom out.
- **The skin bake.** The bake rays meet the target's skin parts alone, so the baked skin holds what the target paints on its skin and no strand of the brow cards. A game character may paint brows on its skin under its brow cards; the exports then show both layers as the game does: the painted layer and the cards over it. The web viewer's own brows step aside for both.

The exports write the grooms as the target draws them ([VRM avatars](vrm.md#hair-and-brows) for the VRM files):

- **Hair.** A mesh `hair` of its own, skinned with the hair's own weights, with `COLOR_0`, a material per groom material and one spring chain per chain of the target (the spring preset `hair`, hanging from `head`); its node's extras hold `opensculptboy_groom`, and the scene extras record it under `opensculptboy.hair`. VRM files write its chains as spring bones.
- **Brows.** Parts of the body mesh, as the lashes are: each brow point takes the closest point of the skin under it at rest and keeps its offset from that point, so the brows follow the skin's morph targets (`browInnerUp`, the squints), its skin weights and a VRM file's T-pose. The brows come with the eyes of `opensculptboy.eyes`; their materials are named `brows:<material>` in GLB files, and the scene extras record them under `opensculptboy.brows`.
- **Checks.** `test/test_groom.py` checks a synthetic handoff with a hair cap, three coloured cards on a chain of two joints and two brow strips: the reading, the carry, the skin and the chains, the files and the card's sections, the GLB file (the hair's node, `COLOR_0`, its chain's weights and the brows through `browInnerUp`), the VRM files of both versions, and the page's blink targets against the GLB file's.

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

## Statue scans

`opensculptboy.statues` fits a scan of a statue (STL, OBJ, PLY or GLB, in any unit and with any axis up) with one pipeline for every statue. A statue is a config file (`opensculptboy.statue/1`, the shipped ones in `src/opensculptboy/data/statues/`): the scan's file, name, licence, source and credit, the stature, the up and front axes when the guesses fail, the plinth, the masks, the lost parts, the start pose and landmark hints. Positions are in the frame of the converted figure (`replica.glb`: metres, Y up, facing +Z, on y = 0), which the `preview` step draws over a 10 cm grid.

```bash
python -m opensculptboy.statues run doryphoros.json david_goliath.json --scans scans/ --out out/ --steps convert,preview   # a first look
python -m opensculptboy.statues run doryphoros.json david_goliath.json --scans scans/ --out out/                           # convert, preview, fit, measure
python -m opensculptboy.statues convert statue.stl --out handoff/ --lost hand.l,nose                                       # the conversion alone
```

Each statue writes `out/<name>/`: `handoff/`, `preview/` (front, side and back over the grid, and the fit's skin alone), `card/<name>.json` with its detail levels, `measures.json` (the distances per region of each stage) and `meshes/` (the scan's skin, the removed pieces and each fitted stage in glTF's frame).

```python
from opensculptboy.statues import fit_statue, scan_to_handoff

scan_to_handoff("statue.stl", "handoff/", stature=1.8, masks=masks, landmarks=marks, lost=["hand.l"])
fit_statue("handoff/", detail=2).write("statue.json")
```

- **Frame.** The up axis is the scan's longest principal axis, pointing away from the end with the wider footprint, and the scan's own axis when it lies within 12 degrees of one; the front is the direction in which the outline of the head stands out most in the band of the nose. `--up` and `--front` give them by hand, as a raised arm over the head needs.
- **Plinth.** Walking up from the bottom, the plinth ends at the first slab whose footprint falls under 45 % of the lowest slab's; its top is the height that many of the plinth's points share. The figure above it is scaled to `--stature`.
- **Masks.** Boxes, spheres, half spaces, capsules (a segment with a radius, for spears, swords and slanted struts), cylinders (the inside, or with `outside` the outside, between two planes: a round base, a hat's brim around the head) and vertex lists in the frame of `replica.glb` take pieces out: `support` and `break` are removed, `drapery` becomes a part of the role `clothing` and `hair` a part of the role `hair`, both out of the fit. Loose fragments under 2 % of the skin's area are removed too. `removed.glb` holds what was taken out, and `appearance.json` records the source, the licence and the transform from the scan under `statue`.
- **Lost parts.** `--lost` names parts the statue lost (`hand.l`, `forearm.r`, `arm.l`, `foot.r`, `lower_leg.l`, `leg.r`, `nose`). `fit_statue` gives their vertices no data (`FitOptions.no_data`), so that anny keeps its own shape there and the part follows the fitted skin; the plain fit pulls a lost hand onto the thigh beside it.
- **Borders.** Where a support or a hat came off, the scan has an open border. `fit_statue` marks the scan's open borders, two rings around them and every scan vertex within 12 mm of them as the reference's `border` region, and the fit drops the pairs whose closest point lies on it, so that the skin along the cut keeps anny's shape and shows no crease.
- **Landmarks from pictures.** The `landmarks` step draws the converted figure from the front and from its left side and finds MediaPipe's body landmarks on both (`opensculptboy.statues.landmarks`, the `pose` extra): the front gives x and height, the side depth and height, and each pair of the side picture takes the left-right order whose heights agree with the front's. The joints (shoulders, elbows, wrists, hips, knees, ankles) become landmarks under the config's hints; the face's points are left out, as the body model placed them 5 cm high on a bowed head under a hat. The crown is the skin's highest point only when nothing masked lies over it.
- **Pruning.** With `prune` (25 mm by default), a first fit to level 0 tells the figure from what is left of props and supports: the scan's triangles further than that from the fitted surface come out (`prune_far`), those by the scalp and the neck as a hair piece (`hair_cut`, role `hair`) and the others to `removed.glb`; the face, the hands and the trunk keep all of their skin. The fit then runs on the pruned scan.
- **The skull under a hat, the feet by a removed piece.** The same first fit marks the vertices whose nearest piece of the statue is a hat (the brow, the ears, the scalp and the neck, judged on the first fit's sliders and pose), fused locks (the scalp, the ears and the neck) or a piece the conversion took out (a support, the plinth, a prop: the limbs and the trunk, judged on the first fit's level 0) (`covered_skin`, `statue.covered` in `appearance.json`). The fit gives them no data and their offsets are the smooth extension of the fitted offsets around them (`FitOptions.fill_no_data`), so the skull keeps anny's shape under the hat and a foot set on a removed head stays whole. `measures.json` gives the share of the skull's fill that comes through the statue and how deep under it the fill lies (`under_cover`).
- **Seated pieces.** The measure step seats the open borders of the hat and the locks on the finest fitted skin, within 5 cm for a hat and 3 cm for hair, and writes them beside the card (`card/<name>.<piece>.glb`, `seat_piece`).
- **Solid statues.** A statue has carved eyes and closed lips. With `solid` (the default), each level carves anny's eyeballs onto the scan along rays from their centres within 60 degrees of the eye's forward direction and keeps the linings of the sockets and the mouth and the rest of the eyeballs 4 mm inside the scan, each moved in along the ray from the centre of its opening (`FitOptions.close_openings`, `close_inside`); the lips are sealed where the fitted lips meet (`FitOptions.lip_seal_self`), as a carved mouth has no crease for the seal's line finder. Every export and the viewer then show solid eyes and closed lips.
- **Pits.** A dense fit of a coarse scan dents single vertices where a few points of the scan pull them. `fit_statue` relaxes the fitted skin's pits that lie more than 0.5 mm deeper than the scan's own around them (`FitOptions.relax_dents`).
- **Start pose.** A statue stands far from anny's rest pose, and with few landmarks the parameter stage reached half of the scan. `fit_statue` starts the bone turns (`FitOptions.initial_pose`) from the pose of anny's library with the smallest trimmed chamfer distance to the scan (`library_start`), or from the config's `pose`. The joints found on the pictures, or hints in the config, guide a pose that no library pose matches; hints of the face snap to the scan.
- **Front.** The front guess takes the nose under the crown, then the chest's widest axis, so that a turned head leaves the body facing the camera. A hat's brim, a helmet or a head bowed under long hair defeat it, and the config gives `front`.
- **Measures.** `opensculptboy.statues.compare.region_distances` measures a fitted mesh against a surface per region of the figure (face, scalp, ears, neck, torso, arms, hands, legs, feet), both ways, with the share of fitted vertices more than 5 mm from it and the share of the scan's points more than 20 mm from the fit (pieces the fit does not cover, left out of the reverse distances).

On a synthetic statue (anny in contrapposto with a 4 mm sculpted surface and a 9 mm mass of curls, a plinth, a stump fused to the right leg, a lost left hand and a nose broken flat, remeshed at 4 mm with 0.2 mm of noise), level 2 lies within 0.13 mm (mean) of the statue as made on the torso and the arms, 0.17 mm on the curls and 0.31 mm on the face; measured from the statue as made to the fitted surface, the lost hand keeps anny's shape within 3.9 mm (mean) and the leg along the stump within 0.31 mm, where the plain fit leaves 15.6 mm and 3.9 mm. `test/test_statues.py` checks the conversion on a synthetic statue.

## Limits

- Hair, eyelashes, brows and clothing are left out of the fit; the bake takes the eyelashes, the hair and the brows for the exports.
- The hair follows the head and its spring chains alone: a slider that changes the head's shape after the fit moves the hair with the head without reshaping it, and the shape of a helm over the hair is the helm's own (the `equipment.head.hair` rule of a handoff is not read).
- The inside of the mouth and the eye sockets follow the skin around them. A handoff that describes its inner mouth gives the exports the target's own gums, palate, tongue, throat and teeth ([Mouth](mouth.md)).
- An ear far from Anny's shape needs the ear outline landmarks, and an ear that stands off from the head needs `masks.ear`.
- The body landmarks are joint centres of another rig, so they guide the pose and the proportions loosely; the face landmarks carry most of the placement. A target in a pose far from Anny's (arms raised high, crouching) needs landmarks on the elbows, the wrists and the knees.
- The offsets belong to the card's slider values: changing a slider afterwards moves the parametric body under fixed offsets.
- The bake carries the target's base colour, roughness and shading normals; the subsurface values come from the card.
- The tip of a pointed ear packs its texels closer on the ear tile, because the tip is a cone that no flat map spreads evenly.

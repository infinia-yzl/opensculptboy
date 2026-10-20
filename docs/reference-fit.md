# Fitting a reference mesh

`opensculptboy fit` reproduces a given humanoid figure with anny: it fits anny's parameters and pose to a target mesh of any topology, and it keeps what the parametric body cannot reach as a detail layer of per-vertex offsets on subdivisions of anny's mesh. The first target is a replica of a Dragon's Dogma 2 character, exported by the DD2DA tools as an appearance handoff folder, but any skinned or static triangle mesh in metres works.

```bash
uv sync --extra viewer                                       # SciPy for the fit, embreex for the texture bake
opensculptboy fit handoff/ --out character.json --detail 1      # the card, character.detail0.safetensors and character.detail1.safetensors
opensculptboy fit handoff/ --out character.json --textures      # also character.base_color.png, character.normal.png and character.eyes.png
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

The folder follows the draft contract `docs/contracts/APPEARANCE-HANDOFF.md` of DD2DA:

```
handoff/
    replica.glb        the skinned parts in their edited bind pose; metres, Y up, glTF axes (the figure faces +Z)
    textures/*.png     the textures of the parts
    appearance.json    {"landmarks": {name: [x, y, z]}, "parts": ..., "materials": ...}
```

`opensculptboy.fit.handoff.load_handoff` reads every primitive of `replica.glb` as one part, with its node transforms (the skinning is not applied, since the parts are in their bind pose), and turns it into Anny's frame: metres, Z up, the figure facing -Y, so that glTF's (x, y, z) becomes (x, -z, y). Each part keeps the name of its mesh, the name of its material, its UVs and its base colour and normal textures where the file has them.

`parts` may be a list of objects or an object keyed by part name, as the contract writes it (`{"body": {"node_name": ...}, "hair": {...}}`). Each entry names a mesh or node of the GLB file (`"name"`, `"mesh"`, `"node"` or `"node_name"`; a name also matches the nodes that start with it and an underscore, as the DD2DA tools split a part into `<node_name>_group<N>_...` nodes) and may give a `"role"` (or `"kind"`, `"type"`, `"category"`); an entry of an object without one takes its key as the role. Entries of `materials` may give a `"role"` per material. The role of a primitive is, in this order: its material's declared role; the role that its material's name names; its part's declared role; the role that its own name names; else skin. The names name the roles `lashes`, `brows`, `beard`, `hair`, `mouth` (teeth, the tongue, the gums), `eyes`, `nails` and `clothing`, so that the eyes, the lashes and the inside of the mouth of a head part leave the skin even when the part itself is skin. The roles `skin`, `body`, `head`, `face`, `hands` and `feet` are skin, and only the skin takes part in the fit. The parts of role `eyes` give the eyeballs: each one that is a sphere (`opensculptboy.fit.reference.eye_spheres`) places anny's eyeball, and the texture bake takes the iris from them.

### Landmarks

The landmarks are optional. Each name of the table matches loosely (`LeftEyeOuter`, `eye.outer.L` and `left_eye_outer` all mean `eye_outer_l`), and `l` and `r` are the figure's own left and right (+X and -X in both frames). Unknown names are ignored.

| Landmark | Counterpart on anny | Weight |
|---|---|---|
| `eye_outer_l`, `eye_outer_r` | exocanthion (craniofacial landmark `ex`) | 1 |
| `eye_inner_l`, `eye_inner_r` | endocanthion (`en`) | 1 |
| `nose_tip` | pronasale (`prn`) | 1 |
| `mouth_corner_l`, `mouth_corner_r` | cheilion (`ch`) | 1 |
| `chin` | gnathion (`gn`) | 1 |
| `ear_l`, `ear_r` | tragion (`t`) | 0.5 |
| `ear_top_l`, `ear_top_r` | superaurale (`sa`), the highest point of the ear: the tip of a pointed ear | 0.2 |
| `ear_lobe_l`, `ear_lobe_r` | subaurale (`sba`), the lowest point of the lobe | 0.2 |
| `ear_back_l`, `ear_back_r` | postaurale (`pa`), the hindmost point of the ear | 0.2 |
| `ear_lateral_l`, `ear_lateral_r` | the point of the ear furthest from the head (`ela`) | 0.2 |
| `crown` | vertex (`v`), the top of the head | 0.5 |
| `neck_base` | the head of bone `neck01` | 0 |
| `shoulder_l`, `shoulder_r` | the head of `upperarm01` | 0.1 |
| `elbow_l`, `elbow_r` | the head of `lowerarm01` | 0.2 |
| `wrist_l`, `wrist_r` | the head of `wrist` | 0.2 |
| `hip_l`, `hip_r` | the head of `upperleg01` | 0 |
| `knee_l`, `knee_r` | the head of `lowerleg01` | 0.2 |
| `ankle_l`, `ankle_r` | the head of `foot` | 0.2 |

The face landmarks are points of the skin, and the body landmarks are joint centres, which sit differently in every rig; the weights reflect that. The neck base and the hips take no weight: they are read and reported, but the joint centres of the first real source lay 20 to 27 mm from anny's, and they pulled the placement and the pose further than they helped. A source may give a landmark twice, as a joint position and as the nearest point of the skin, the latter named with a `skin` or `surface` word (`mouth_corner_l` and `mouth_corner_l_skin`): the face landmarks take the point of the skin, and the joint centres the other. Loose names of the ear outline include `ear_tip_l` and `helix_tip_l` (`ear_top_l`), `earlobe_l` (`ear_lobe_l`) and `postaurale_l` (`ear_back_l`). Without landmarks the fit places the target by its bounding geometry: the highest point (the crown), the points furthest to each side (the fingertips) and the soles, which the same rule finds on anny. This assumes the contract's axes, with the figure standing upright and facing +Z.

### Proposed additions to the handoff contract

The tragion alone leaves the shape of the ear to the surface terms, and a pointed ear far from anny's then fits with crumpled folds. The fit reads four more points per ear, which a handoff can take from the head skin as the extremes of the outer ear's vertices (in glTF's frame with the figure facing +Z):

| Name | Point of the outer ear |
|---|---|
| `ear_top_l`, `ear_top_r` | the highest vertex (+Y): the tip of a pointed ear |
| `ear_lobe_l`, `ear_lobe_r` | the lowest vertex (-Y): the bottom of the lobe |
| `ear_back_l`, `ear_back_r` | the hindmost vertex (-Z) |
| `ear_lateral_l`, `ear_lateral_r` | the vertex furthest from the midline (largest \|X\|) |

Each is a point of the skin, so no `_skin` variant is needed. The outer ear is the handoff's optional `masks.ear` field: one entry per side, naming the part's mesh or node, its primitive or material, the `side` (`l` or `r`) and the indices of the outer ear's `vertices` in that primitive, without the line where the ear attaches to the head; the entry may also say where the mask comes from (`source`) and give that attachment line (`attachment`). `load_handoff` reads it onto the part (`HandoffPart.ear_vertices`) and ignores entries that name no loaded part. The real fit below takes the outline from such a mask by exactly these rules.

## The pipeline

1. **Placement.** With landmarks, a rigid alignment of anny's landmarks onto the target's (Umeyama 1991); without them, the centres of the bounding landmarks. The height phenotype starts at the value whose stature matches the target.
2. **Parameters.** Adam over the 6 phenotypes, the local changes, the 113 face shapes, the turns of every bone but the root and the eyes, and a rigid placement, in 10 rounds of 30 steps. Each face shape sums the left and right MakeHuman targets of a pair of local changes (`chin-prominent` moves the vertices of `chin-prominent-incr` and `chin-prominent-decr`), so fitted together the two would split one movement between two controls and could run past the slider ranges. The fit therefore takes the face shapes on the 125 local changes of the face that share a target with one (`FitOptions.shared_targets`, `"face_shapes"`); those local changes stay at 0, the 131 local changes of the body stay free, and the asymmetry that the symmetric face shapes cannot hold is left to the detail layer. `"local_changes"` fits the local changes instead, and `"both"` both. Each round finds the closest points again. The loss adds point-to-plane distances from anny's vertices to the target surface, point-to-plane distances from a sample of 20,000 target vertices to anny's nearest vertices, the landmark distances and small ridge penalties on the local changes, the face shapes and the bone turns. Pairs farther apart than a limit (50 mm in the first round, 10 mm in the last) or whose normals disagree drop out. The first three rounds free the placement, the phenotypes and the large joints alone, and the fingers stay at rest until round 7 (`FitOptions.finger_rounds`), when the limit is under 20 mm: freed earlier, a finger of anny can slide onto its neighbour and stay there.
3. **Non-rigid refinement.** Anny's whole mesh moves onto the target by a non-rigid ICP in the spirit of optimal-step NICP (Amberg et al., CVPR 2007): closest points pull the vertices, and a Laplacian term keeps the displacement smooth, with a stiffness that falls over four stages. A reverse term pulls each vertex that is the nearest of some target vertices (within the distance limit, with agreeing normals) to their mean: closest points alone never pull a surface past its border, so without it the parts of a target that reach beyond anny's, such as the tip of a pointed ear or eyelids that cover more of the eye, stay uncovered. The eye sockets, the inside of the mouth and every closed part apart from the body (the eyeballs and the tongue) take no data, and neither does the lips' inner margin, the ring of vertices along the inside of the mouth (`FitOptions.lip_margin_weight`), which follows its neighbours: pulled onto a closed mouth, it folded into the corners as a notch. The sockets and the mouth take the displacement of the nearest fitted skin in space, and a vertex of them still in front of the skin goes back behind it: along the mesh they hang from the lips' and the lids' inner margins alone, and a refinement that pulls the lips back would leave the gums in front of the upper lip. Each eyeball moves and scales onto the target's eyeball sphere when the target has one, and otherwise moves with the skin around it, as the tongue does. The outer ears (`ear_weights`: the displacement of MakeHuman's ear translation targets, 1 on the ear and falling to 0 around it) are 3 times stiffer than the rest (`FitOptions.ear_stiffness`), and after the refinement their vertices relax along the target for 10 rounds (`relax_along`: each moves by half its umbrella vector within its tangent plane, then onto its closest point of the target), which spreads the stretch of a pointed tip, and of the texture on it, over the whole ear and keeps the folds from crumpling. Last, every vertex within 2 mm of the target whose normal agrees with it to 0.8 moves onto its closest point (`project_onto`, `FitOptions.project_distance`), but on the ears: the Laplacian term left the coarse vertices of level 0 off the surface by a few tenths of a millimetre, which showed as small lumps along the jaw and the cheeks. The forward errors are then close to 0 by construction, and the reverse errors measure the fit.
4. **Detail levels.** Each level of the detail layer repeats the refinement on its subdivided mesh, starting from the level below, with a lower stiffness.
5. **Residuals.** The registered mesh of each level goes back to the rest frame by the inverse of each vertex's blended bone transform, and the offsets are the registered rest mesh minus the parametric rest mesh. Added to the subdivided rest mesh and skinned again, they give back the registered mesh.

The card holds the fitted parameters, the fitted pose (the target's bind pose, which the GLB export plays as the animation named `pose`), and the detail layer with the placement and the measured errors.

The code lives in `src/opensculptboy/fit/`: `handoff.py` (the folder), `reference.py` (the target, its landmarks and the closest-point queries), `register.py` (stages 1 to 4, `fit_reference`, `fit_levels` for the levels of a fitted card, `FitResult.write`) and `residual.py` (the unposing, the offsets and `surface_error`). It reuses the Umeyama alignment and the uniform Laplacian of `anny.faces.authoring.nicp` and the closest-triangle search of `anny.hair`.

## The detail layer

A card with a detail layer uses character schema 2; a card without one is still written as schema 1, byte for byte as before, and both schemas load.

```json
{
  "schema": 2,
  "name": "character",
  "phenotype": {"gender": 0.62, "age": 0.48, "...": 0.5},
  "detail": {
    "levels": [
      {"subdiv": 0, "offsets": "character.detail0.safetensors", "error_mm": 0.26},
      {"subdiv": 1, "offsets": "character.detail1.safetensors", "error_mm": 0.04}
    ],
    "textures": {"base_color": "character.base_color.png", "normal": "character.normal.png", "eyes": "character.eyes.png", "subdiv": 1},
    "fit": {"placement": {"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "translation": [0, 0, 0]}, "errors_mm": {}, "seconds": {}}
  }
}
```

Level 0 is anny's own mesh of 13,718 vertices. Levels 1 and 2 are one and two levels of Catmull-Clark subdivision of the quad mesh of the same vertices (`topology="anny-quads"`), with 54,848 and 219,368 vertices. The subdivision is `anny.utils.subdivision`, this repository's own implementation of Catmull-Clark as sparse linear operators (Apache-2.0); no outside library is involved. Because the subdivision is linear, the same operator carries every per-vertex quantity of level 0 to the finer levels:

- the rest mesh and the 52 ARKit morph targets, exactly;
- the skin weights, after which each vertex keeps its 12 strongest bones, renormalised (a file then keeps 4 or 8);
- the UVs, face-varying and bilinear inside each face of the level above, so that anny's atlas keeps its seams and its layout on every level.

Each offsets file (`opensculptboy/detail-offsets@1`) holds one float32 vector per vertex of its full level, in Anny's rest frame (metres, Z up), with the level and the topology in its metadata. The paths are relative to the card. `opensculptboy.detail.DetailModel` applies a level: it runs the Anny model, subdivides the rest mesh, adds the offsets after the blend shapes and skins the result with the subdivided weights. The offsets belong to the card's own parameter values; when a slider changes later, they stay where they were in the rest frame.

## Exports

`export_glb`, `export_vrm` and `opensculptboy export` take `detail_level` (`--detail`, default 0). Level 0 applies the card's level-0 offsets when it has them; a card without detail writes the same files as before, and `test/test_detail_layer.py` and the reference fingerprints of `test/test_gltf_export.py` guard that. Finer levels need the offsets of that level in the card.

| Mesh | Vertices | Triangles |
|---|---|---|
| level 0 (anny's mesh) | 13,718 | 27,420 |
| level 1 on the head and the neck, level 0 elsewhere | 27,644 | 55,272 |
| level 1 | 54,848 | 109,680 |
| level 2 on the head and the neck, level 1 elsewhere | 110,504 | 220,992 |
| level 2 | 219,368 | 438,720 |

A GLB file takes the full level. A VRM file keeps within the triangle budget of its version (`opensculptboy.export.budget`): when the full level is over it, the head and the neck alone take the level, and when that is still over it, the level steps down to one the card holds, with a `BudgetWarning`. So `--detail 1` gives VRM 1.0 files (70,000 triangles) level 1 on the head and the neck, and VRM 0.x files (32,000 triangles) level 0. `--budget off` keeps the full level. The scene extras record the level and the region in use. The fine meshes carry no `_ANNY_VERTEX` attribute, since their vertices are not Anny's. A card with an eye texture adds the eyeballs' primitive and material (see Textures), which raises the material count of a VRM file to 2 of its budget's 16 (1.0) or 8 (0.x).

## Textures

`--textures` (`opensculptboy.texture.bake.bake_fit`) bakes the target onto anny's UV atlas, on the mesh of the finest level, and `opensculptboy bake handoff/ --card character.json` (`bake_card`) bakes a fitted card again without fitting, from its own offsets and placement. For each texel, a ray along the fitted surface's normal (Embree, as in the viewer's bakes) finds the target's skin within 10 mm, on a triangle that faces the same way (on a thin part such as an ear, the nearest surface along the ray can be the other side). The texels of anny's eyeballs take the target's eyes in the same way. Texels that no ray reaches take the nearest reached value, which dilates every island to the edge of its texture, so that filtering and mipmaps do not pull in the background at the seams. The vertex colours of the target are never read: a source may keep other data in them, such as a game's masks or the albedo of hair cards.

| Map | Slot | Content |
|---|---|---|
| base colour | `base_color` | the target part's base colour texture at the hit (bilinear) |
| normal | `normal` | the target's shading normal in the tangent space of the fitted surface |
| roughness | `roughness` | the G channel of the target's metallic-roughness texture times its roughness factor (grey) |
| thickness | `thickness` | from the fitted mesh: the mean distance through it along the inward normal and six rays 30 degrees around it, over `thickness_mm` (30 mm), the eyeballs passed through (grey) |
| curvature | `curvature` | from the fitted mesh: the mean curvature (1 / radius, positive where convex) as 0.5 + 0.5 k / `curvature_per_m` (100 per metre), smoothed over 4 rings (grey) |

The shading normal is the target's own: its file normals (glTF `NORMAL`), which artists shape and smooth so that a surface reads differently from its bare geometry, with its normal map applied in its MikkTSpace frames, as Blender and glTF viewers apply it. The first real target showed why both matter: its file normals differ from its geometric normals by a median of 5.5 degrees and a 95th percentile of 22 degrees along the jaw and the lower face, and 10 to 20 degrees at the seam of the head and the body, so a bake of its geometry alone gave the jaw a bulge and the skin a plastic smoothness that the target does not show. The tangent space of the fitted surface is MikkTSpace too (`opensculptboy.texture.tangents`, an independent NumPy port of Mikkelsen's algorithm) on the mesh split at its UV seams with the normals of the welded mesh: the frames that the exporters write as `TANGENT` and that Blender computes, so that the map reads back as it was baked. On a 16,232-triangle head of a game character, the port agrees with Blender's tangents to 0.03 degrees (float32 rounding) and with its signs exactly, except next to the triangles of zero UV area, where Blender's own port is not consistent within a triangle; anny's meshes have none.

### Tiles

A 4096-texel atlas gives anny's head about 4.6 texels per millimetre of skin, as much as the target's 2048-texel head texture, so its detail would be resampled at about the rate it has. Three tiles therefore give parts of the body textures of their own, each drawn as a primitive of the body mesh with a material of its own (the same vertices, morph targets and skin, so that the bones, the look-at and the expressions are unchanged):

| Tile | Slots | Size | Covers | Layout |
|---|---|---|---|---|
| eyes (`opensculptboy.texture.eyes`) | `eyes` | 1024 by 512 | both eyeballs | the box of the atlas around them, through a texture transform |
| head (`opensculptboy.texture.tiles`) | `head_base_color`, `head_normal`, `head_roughness` | 2048 by 4096 | the head's UV island: the face, the scalp, the ears and the upper neck, 3,116 of anny's quads | the island's box of the atlas, through a texture transform |
| ears (`opensculptboy.texture.ears`) | `ears_base_color`, `ears_normal`, `ears_roughness`, `ears_thickness`, `ears_curvature` | 2048 by 1024 | the outer ears, on levels 1 and 2 | UVs of their own, unwrapped from the fitted ears |

The eye and head tiles keep anny's UVs and carry their box as a texture transform: `KHR_texture_transform` in glTF and VRM 1.0 (on every texture of the material), and the ST vector of `_MainTex` in VRM 0.x, which MToon 0.x applies to every texture and three-vrm and UniVRM read back as the same transform; a reader without `KHR_texture_transform` shows the whole tile squeezed onto its part. The head tile holds about 6.3 texels per millimetre, 1.35 times the target's head texture. The head's island ends at UV seams, so the border between the skin and the head materials follows a seam of the atlas; the triangles of the island are found by their UV centroids (`on_head_tile`), which works on every level and on the head region's meshes.

Anny's eyeballs cover about 52 by 52 texels each of a 2048-texel atlas, too few for an iris, so the bake also writes an eye texture (`character.eyes.png`, slot `eyes`, 1024 by 512 texels; `opensculptboy.texture.eyes`): the box of the atlas around both eyeballs, padded and widened to 2 by 1, baked from the target's eyes at about nine times the atlas's resolution along each side. The exports then draw the eyeballs as a second primitive of the body mesh (the same vertices, morph targets and skin, so that the eye bones, the look-at and the expressions are unchanged) with a material of their own, whose texture carries the box as a texture transform: `KHR_texture_transform` in glTF and VRM 1.0 (on the base, shade and normal textures of the MToon material), and the ST vector of `_MainTex` in VRM 0.x, which MToon 0.x applies to every texture and three-vrm and UniVRM read back as the same transform. The files keep one UV set, and a reader without `KHR_texture_transform` shows the whole eye texture squeezed onto the eyeballs.

Anny's atlas lays out its round ears, and a fit that reaches a pointed ear stretches the texels of the tip: on the first real target, the tip's texels on the head tile came to 1.5 per millimetre (5th percentile) against 3.5 at the median, and the bake showed streaks there. For levels 1 and 2, the ear tile unwraps each outer ear (the faces whose ear weight, `opensculptboy.fit.register.ear_weights`, is over 0.5: a disk of 6,557 vertices per ear on level 2) from the fitted rest surface of the card's finest level by a least-squares conformal map (Levy et al. 2002) refined by 20 as-rigid-as-possible iterations (Liu et al. 2008), with one scale for both ears, the left ear (+X) in the left half and each turned with its long side along v. Catmull-Clark numbers the vertex points of a level first, so vertex `i` of level 1 is vertex `i` of level 2, and one table of UVs per vertex of the finest level (`<card>.ear_uvs.safetensors`, `material.ears.uvs`) serves both levels; `with_ear_uvs` appends the ears' UVs after the atlas's and points the ear triangles at them, and the exporters draw them with the ear tile's material. Level 0 keeps anny's UVs. On the first real target, the ear tile holds about 6.7 texels per millimetre at the median and 4.5 at the 5th percentile (8.2 by area), against 3.5 and 2.0 on the target's own ears. The ear islands are new UV seams; the normals stay welded across them.

The bake casts 2 by 2 rays per texel on the head tile and 3 by 3 on the ear tile, spread over the texel, and averages what they hit, so that the edges of colour, the rims of thin parts and the borders of the islands do not stair-step; the atlas takes one ray per texel. On two cores the bake of the first real target on level 2 with a 4096 atlas, the 2048 by 4096 head tile and the thickness and curvature maps took 136 s and 5.1 GB of memory with one ray per texel; the supersampled tiles add their rays to that.

## The skin material

A card may carry a skin material (`Character.material`, character schema 3; `opensculptboy.material.SkinMaterial`): the textures above as paths relative to the card, and the values that a renderer needs to rebuild the skin. `--textures` and `opensculptboy bake` write the textures' paths, and `--material skin.json` lays further values over them. Cards without the section keep their schema (1 or 2) and export exactly as before.

```json
"material": {
  "base_color": "character.base_color.png", "normal": "character.normal.png", "roughness": "character.roughness.png",
  "thickness": "character.thickness.png", "thickness_mm": 30.0, "curvature": "character.curvature.png", "curvature_per_m": 100.0,
  "roughness_factor": 1.0, "specular": 1.0, "specular_color": [1, 1, 1], "ior": 1.5, "normal_scale": 1.0,
  "subsurface": {"weight": 0.1, "radius_mm": [5.4, 2.1, 1.2], "ior": 1.4, "anisotropy": 0.0, "method": "random_walk_skin"},
  "head": {"base_color": "character.head_base_color.png", "normal": "character.head_normal.png", "roughness": "character.head_roughness.png"},
  "ears": {"uvs": "character.ear_uvs.safetensors", "base_color": "character.ears_base_color.png", "normal": "character.ears_normal.png", "...": "..."}
}
```

The subsurface radius is the mean free path per RGB channel in millimetres (Blender's Subsurface Radius times its Subsurface Scale), and `method` names Blender's methods (`random_walk_skin`, `random_walk`, `burley`).

**glTF.** The skin, head and ear materials are metallic-roughness materials (metallic 0, the roughness texture in `metallicRoughnessTexture`, read by its G channel) with `KHR_materials_specular` and `KHR_materials_ior`, and every primitive carries MikkTSpace `TANGENT`s when the file has a normal texture. glTF has no ratified subsurface extension: `KHR_materials_diffuse_transmission` is a release candidate for thin translucent surfaces such as leaves, `KHR_materials_volume` needs transmission, and the proposed `KHR_materials_scatter`, which would replace `KHR_materials_volume_scatter`, is still under review; an opaque skin under them would render as translucent in the viewers that support them. The export therefore keeps the subsurface settings in each skin material's extras, under `opensculptboy_skin` (format `opensculptboy/skin@1`): `subsurface` as in the card, and `thickness` and `curvature` as texture references (`index`, `texCoord`) with their scales (`max_mm`, `max_per_m`), for real-time engines that shade skin with a thickness map (translucency through the ears) and a pre-integrated lookup by curvature (Penner 2011). The head material reads the atlas's thickness and curvature maps, and the ear material its own. Blender's importer builds the Principled BSDF from the material and its extensions and keeps the extras as custom properties; `src/opensculptboy/export/blender_skin.py`, which needs nothing but Blender's own Python, then sets the subsurface:

```bash
blender --python src/opensculptboy/export/blender_skin.py -- character.glb   # import the file and set its skin
```

Inside Blender, `exec` the script and call `apply_skin_materials()` after an import (`apply_skin_materials("character.glb")` reads the settings from the file when the extras did not come along). It sets the subsurface method, Subsurface Weight, Subsurface Radius (normalised to its largest channel), Subsurface Scale (the largest channel in scene units), Subsurface IOR and Subsurface Anisotropy; Blender 4 and later take the subsurface colour from the base colour.

**VRM.** MToon has no subsurface scattering, so the VRM export bakes a skin ramp (`opensculptboy.export.skin_ramp`): for each channel, the diffuse falloff against N.L of a sphere of 20 mm radius (the scale of the nose, the lips and the ears) whose light spreads along the surface with the radial distribution of Burley's normalised diffusion profile at the channel's radius, mixed with a Lambert falloff by the subsurface weight; MToon's one ramp (`shadingShiftFactor`, `shadingToonyFactor`) and its shade colour per channel are fitted to the three falloffs by a grid search and least squares. The shade colour factor is the fitted ratio (times the base colour factor), over the base colour texture as the shade texture. It is an approximation: MToon has one ramp shape for all channels, no light through thin parts and no blur of the texture, and one curvature stands for the whole skin. Scattering of a few millimetres changes the falloff of a face little: the material above gets the shade ratio (0.011, 0.003, 0.001), shift -0.5 and toony 0.5, a Lambert ramp with a faint red terminator, and a subsurface weight of 1 gives (0.13, 0.02, 0.00). The ramp follows the material; it does not add the lighter, softer shade of a stylised toon skin, which the cards without a skin material keep. VRM files hold the base colour and the normal map of the skin, the head tile and the ear tile, and no roughness, thickness or curvature maps.

## Accuracy

`test/test_reference_fit.py` fits a synthetic target: anny with random phenotypes, 12 random local changes, 10 random face shapes and a small pose, plus smooth random offsets of 3 mm RMS on level 0 and 0.8 mm RMS on level 1, evaluated on level 2, moved by a random rigid placement and remeshed (the vertices slide within the surface, and vertex clustering on a 5 mm grid merges them into 53,192 vertices and 107,818 triangles, without the eyeballs, the tongue and the insides of the mouth and the eye sockets). The landmarks carry 1 mm of noise. The errors, in millimetres, are the distances from the fitted vertices to the target surface (forward) and from 20,000 target vertices to the fitted surface (reverse):

| Stage | Forward mean | Forward p95 | Reverse mean | Reverse p95 |
|---|---|---|---|---|
| parameters alone | 0.989 | 2.535 | 1.053 | 2.715 |
| level 0 | 0.019 | 0.000 | 0.397 | 1.019 |
| level 1 | 0.005 | 0.000 | 0.129 | 0.327 |
| level 2 | 0.003 | 0.000 | 0.028 | 0.088 |

The refinement ends by moving the vertices onto the target (see the pipeline), so the forward errors are close to 0 by construction and the reverse errors carry the measure; before that step, the reverse means were 0.488 mm on level 0, 0.133 mm on level 1 and 0.072 mm on level 2. Some of the random local changes act on targets that the face shapes share, which the fit leaves to the face shapes, so the parameters alone fit no better than before.

The test holds level 1 under 0.5 mm in both directions. Without landmarks, the same target fits to 0.33 mm (forward) and 0.46 mm (reverse) on level 0. On two CPU cores the fit to level 1 takes about 75 s (parameters 38 s, level 0 8 s, level 1 30 s), and level 2 adds about 180 s and needs about 4 GB of memory.

A synthetic target is the best case: anny's parameter space holds its shape, and the residuals are smooth. A real character leaves larger residuals in the parameters and leans more on the detail layer.

### A real target

The first real test fitted a game-character replica: a slim figure 1.60 m tall from the soles to the top of the scalp, whose skin is 24,656 vertices in three parts (body, hands, head) with pointed ears, in a bind pose with the arms lowered and the forearms turned forward. Its head part also holds the eyes, the lashes and the inside of the mouth as primitives of other materials, which the loader leaves out of the skin by their materials. 24 of its 40 landmarks map to the table; it gives each face landmark twice, as a joint position and as the nearest point of the skin. The fit below adds the 8 ear outline landmarks of the proposal, taken from the replica's ear mask. The errors are in millimetres: forward from anny's vertices that see out of the fitted surface, reverse from every reference vertex that sees out of the reference (94 % of them; the others lie inside the mouth, behind the eyes or in the folds of the ears), split by the region of the reference. The parametric row measures all of anny's fitted vertices.

| Stage | Face forward mean / p95 | Face reverse mean / p95 | Ears reverse mean / p95 | Body forward mean / p95 | Body reverse mean / p95 | Hands reverse mean / p95 |
|---|---|---|---|---|---|---|
| parameters alone | 1.86 / 5.18 | 2.01 / 5.60 | 2.39 / 6.40 | 2.71 / 7.13 | 2.70 / 7.64 | 1.60 / 6.51 |
| level 0 | 0.01 / 0.00 | 0.21 / 0.63 | 0.70 / 2.87 | 0.04 / 0.00 | 0.64 / 1.76 | 0.38 / 0.99 |
| level 1 | 0.00 / 0.00 | 0.05 / 0.17 | 0.27 / 0.94 | 0.00 / 0.00 | 0.13 / 0.42 | 0.07 / 0.24 |
| level 2 | 0.00 / 0.00 | 0.01 / 0.05 | 0.12 / 0.42 | 0.00 / 0.00 | 0.04 / 0.12 | 0.02 / 0.06 |

Before the changes of this round (no ear outline, no shared-target rule, the tongue fitted as skin, no relaxation or projection), the reverse means were 0.38 (face), 1.23 (ears), 0.85 (body) and 0.47 (hands) on level 0, 0.07, 0.31, 0.15 and 0.07 on level 1, and 0.02, 0.08, 0.05 and 0.02 on level 2. Without the ear outline landmarks (the handoff as the tools write it today) the ears reach 1.15, 0.35 and 0.15 mm on levels 0, 1 and 2, and the rest stays within 0.03 mm of the table. On the ears, the share of edges folded by more than 60 degrees fell from 7.0 % to 3.0 % on level 1 and from 3.2 % to 1.4 % on level 2, and the 95th percentile of the angle between the fitted and the reference normals from 91 to 38 degrees on level 1: the folds no longer crumple. The stretch of the texture on the ears (the square root of the area ratio to the parametric surface) reaches at most 4.5 on level 1, against 4.3 before, but the tip now holds the pointed outline that the old fit left uncovered. The level 2 ears lose some of their reverse error (0.12 against 0.08 mm) to the relaxation and the stiffness, which buy the folds.

The fitted landmarks lie within 0.6 to 4 mm of the face landmarks after the parameter stage (the inner eye corners 7 mm, in depth: the source's inner corner lies on the skin in front of anny's endocanthion), within 3 to 6 mm of the shoulders, elbows, wrists, knees and ankles, and 34 to 35 mm from the hips and the neck base, which no longer pull. The ear outline pulls the tragion 9 mm from the source's: anny's round ear cannot meet all five points of a pointed one, and with the outline weighed 0.2 against the tragion's 0.5 the parameters move the ear's shape more than its place; the refinement then lays the surface on the target. On two cores the fit took 40 s for the parameters, 7 s for level 0, 26 s for level 1 and 179 s for level 2; loading the model and baking the 2048-texel textures and the eye texture on level 2 added about 20 s.

By eye, at the same cameras and light, level 1 and level 2 with the baked textures pass for the replica in the face, the profile, the ears and the full-body silhouette, and level 0 now does too at a normal distance: the lumps along the jaw and the cheeks are gone, and the irises come from the eye texture. The test found and fixed, in two rounds: the eyes, lashes and mouth inside a head part counted as skin; a finger of anny slid onto its neighbour (the finger rounds); the gums showed through the upper lip (the mouth follows the nearest skin); the eye openings stayed anny's and the pointed ear tips stayed uncovered (the reverse term); the eyeballs were larger than the target's (the eyeball spheres); the eyes took the colour of the lids (the eye bake); the tongue, fitted as skin, broke through the lower lip; the lips' inner margin folded into the mouth corners; the coarse vertices of level 0 stayed off the surface; the ear folds crumpled; and local changes and face shapes split one movement between them. What remains:

- The ears keep a little roughness at the lobe and where the back of the ear meets the head, and the rim of the pointed tip shows a jagged edge of colour on levels 0 and 1, where the bake hits the front and the back of the thin rim in turn.
- Level 0 shows a thin dark line of the inside of the mouth between the closed lips (about 900 pixels over five close-up cameras; none on levels 1 and 2), and a small mark at the mouth corners.
- Anny's eyeballs have 72 vertices each, so on level 0 the edge of the iris shows the octagon of the eyeball's cornea in a close-up; levels 1 and 2 are round.
- The ear outline landmarks are a proposal; until the tools write them, the ears fit as in the paragraph above.

## Limits

- Hair, eyelashes, brows, teeth, the tongue and the eyeballs are left out of the fit; they attach to the fitted body in a later step. Clothing is left out too.
- The inside of the mouth and the eye sockets follow the skin around them and do not fit the target's.
- An ear far from anny's shape (a pointed ear with deep folds) needs the ear outline landmarks; with them it reaches its outline and its main folds on the detail levels, without them it reaches the outline alone.
- The body landmarks are joint centres of another rig, so they guide the pose and the proportions loosely. The face landmarks carry most of the placement.
- The fit assumes the contract's axes and an upright figure. A target in a very different pose from anny's (arms raised high, crouching) needs landmarks on the elbows, the wrists and the knees.
- The offsets are tied to the card's values: changing a slider afterwards moves the parametric body under fixed offsets.
- The bake carries the target's base colour, roughness and shading normals, not its other maps (specular, cavity or subsurface masks); the subsurface values come from the card. The tiles' resolution is fixed per tile (the eyes 1024 by 512, the head 2048 by 4096 at `--head-size 4096`, the ears 2048 by 1024).
- The ear tile's unwrap may fold a few triangles of a deep fold (18 of 25,968 on level 2 of the first real target).
- The fit needs SciPy (the `viewer` or `examples` extra), and the texture bake needs `embreex` (the `viewer` extra).

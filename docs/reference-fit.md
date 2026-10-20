# Fitting a reference mesh

`opensculptboy fit` reproduces a given humanoid figure with anny: it fits anny's parameters and pose to a target mesh of any topology, and it keeps what the parametric body cannot reach as a detail layer of per-vertex offsets on subdivisions of anny's mesh. The first target is a replica of a Dragon's Dogma 2 character, exported by the DD2DA tools as an appearance handoff folder, but any skinned or static triangle mesh in metres works.

```bash
uv sync --extra viewer                                       # SciPy for the fit, embreex for the texture bake
opensculptboy fit handoff/ --out character.json --detail 1      # the card, character.detail0.safetensors and character.detail1.safetensors
opensculptboy fit handoff/ --out character.json --textures      # also character.base_color.png and character.normal.png
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
| `crown` | vertex (`v`), the top of the head | 0.5 |
| `neck_base` | the head of bone `neck01` | 0.1 |
| `shoulder_l`, `shoulder_r` | the head of `upperarm01` | 0.1 |
| `elbow_l`, `elbow_r` | the head of `lowerarm01` | 0.2 |
| `wrist_l`, `wrist_r` | the head of `wrist` | 0.2 |
| `hip_l`, `hip_r` | the head of `upperleg01` | 0.1 |
| `knee_l`, `knee_r` | the head of `lowerleg01` | 0.2 |
| `ankle_l`, `ankle_r` | the head of `foot` | 0.2 |

The face landmarks are points of the skin, and the body landmarks are joint centres, which sit differently in every rig; the weights reflect that. A source may give a landmark twice, as a joint position and as the nearest point of the skin, the latter named with a `skin` or `surface` word (`mouth_corner_l` and `mouth_corner_l_skin`): the face landmarks take the point of the skin, and the joint centres the other. Without landmarks the fit places the target by its bounding geometry: the highest point (the crown), the points furthest to each side (the fingertips) and the soles, which the same rule finds on anny. This assumes the contract's axes, with the figure standing upright and facing +Z.

## The pipeline

1. **Placement.** With landmarks, a rigid alignment of anny's landmarks onto the target's (Umeyama 1991); without them, the centres of the bounding landmarks. The height phenotype starts at the value whose stature matches the target.
2. **Parameters.** Adam over the 6 phenotypes, the 256 local changes, the 113 face shapes, the turns of every bone but the root and the eyes, and a rigid placement, in 10 rounds of 30 steps. Each round finds the closest points again. The loss adds point-to-plane distances from anny's vertices to the target surface, point-to-plane distances from a sample of 20,000 target vertices to anny's nearest vertices, the landmark distances and small ridge penalties on the local changes, the face shapes and the bone turns. Pairs farther apart than a limit (50 mm in the first round, 10 mm in the last) or whose normals disagree drop out. The first three rounds free the placement, the phenotypes and the large joints alone, and the fingers stay at rest until round 7 (`FitOptions.finger_rounds`), when the limit is under 20 mm: freed earlier, a finger of anny can slide onto its neighbour and stay there.
3. **Non-rigid refinement.** Anny's whole mesh moves onto the target by a non-rigid ICP in the spirit of optimal-step NICP (Amberg et al., CVPR 2007): closest points pull the vertices, and a Laplacian term keeps the displacement smooth, with a stiffness that falls over four stages. A reverse term pulls each vertex that is the nearest of some target vertices (within the distance limit, with agreeing normals) to their mean: closest points alone never pull a surface past its border, so without it the parts of a target that reach beyond anny's, such as the tip of a pointed ear or eyelids that cover more of the eye, stay uncovered. The eyeballs, the eye sockets and the inside of the mouth take no data. The sockets and the mouth take the displacement of the nearest fitted skin in space, and a vertex of them still in front of the skin goes back behind it: along the mesh they hang from the lips' and the lids' inner margins alone, and a refinement that pulls the lips back would leave the gums in front of the upper lip. Each eyeball moves and scales onto the target's eyeball sphere when the target has one, and otherwise moves with the skin around it.
4. **Detail levels.** Each level of the detail layer repeats the refinement on its subdivided mesh, starting from the level below, with a lower stiffness.
5. **Residuals.** The registered mesh of each level goes back to the rest frame by the inverse of each vertex's blended bone transform, and the offsets are the registered rest mesh minus the parametric rest mesh. Added to the subdivided rest mesh and skinned again, they give back the registered mesh.

The card holds the fitted parameters, the fitted pose (the target's bind pose, which the GLB export plays as the animation named `pose`), and the detail layer with the placement and the measured errors.

The code lives in `src/opensculptboy/fit/`: `handoff.py` (the folder), `reference.py` (the target, its landmarks and the closest-point queries), `register.py` (stages 1 to 4, `fit_reference`, `FitResult.write`) and `residual.py` (the unposing, the offsets and `surface_error`). It reuses the Umeyama alignment and the uniform Laplacian of `anny.faces.authoring.nicp` and the closest-triangle search of `anny.hair`.

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
    "textures": {"base_color": "character.base_color.png", "normal": "character.normal.png", "subdiv": 1},
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

A GLB file takes the full level. A VRM file keeps within the triangle budget of its version (`opensculptboy.export.budget`): when the full level is over it, the head and the neck alone take the level, and when that is still over it, the level steps down to one the card holds, with a `BudgetWarning`. So `--detail 1` gives VRM 1.0 files (70,000 triangles) level 1 on the head and the neck, and VRM 0.x files (32,000 triangles) level 0. `--budget off` keeps the full level. The scene extras record the level and the region in use. The fine meshes carry no `_ANNY_VERTEX` attribute, since their vertices are not Anny's.

## Textures

`--textures` (`opensculptboy.texture.bake.bake_fit`) bakes the target onto anny's UV atlas, on the mesh of the finest level: for each texel, a ray along the fitted surface's normal (Embree, as in the viewer's bakes) finds the target's skin within 10 mm, on a triangle that faces the same way (on a thin part such as an ear, the nearest surface along the ray can be the other side). The texels of anny's eyeballs take the target's eyes in the same way. The base colour comes from the target part's base colour texture at the hit, and the normal map holds the target's shading normal in the tangent space of the fitted surface. Texels that no ray reaches take the nearest reached value. The GLB export puts the files in the material's `baseColorTexture` and `normalTexture`, and the VRM export in the MToon base, shade and normal slots.

## Accuracy

`test/test_reference_fit.py` fits a synthetic target: anny with random phenotypes, 12 random local changes, 10 random face shapes and a small pose, plus smooth random offsets of 3 mm RMS on level 0 and 0.8 mm RMS on level 1, evaluated on level 2, moved by a random rigid placement and remeshed (the vertices slide within the surface, and vertex clustering on a 5 mm grid merges them into 53,487 vertices and 108,410 triangles, without the eyeballs and the insides of the mouth and the eye sockets). The landmarks carry 1 mm of noise. The errors, in millimetres, are the distances from the fitted vertices to the target surface (forward) and from 20,000 target vertices to the fitted surface (reverse):

| Stage | Forward mean | Forward p95 | Reverse mean | Reverse p95 |
|---|---|---|---|---|
| parameters alone | 0.969 | 2.493 | 1.044 | 2.687 |
| level 0 | 0.249 | 0.658 | 0.488 | 1.232 |
| level 1 | 0.034 | 0.112 | 0.133 | 0.342 |
| level 2 | 0.015 | 0.051 | 0.072 | 0.178 |

The level 2 row was measured before the reverse term and the finger rounds; levels 0 and 1 are measured with them.

The test holds level 1 under 0.5 mm in both directions. Without landmarks, the same target fits to 0.61 mm (forward) and 0.59 mm (reverse) on level 0. On two CPU cores the fit to level 1 takes about 70 s (parameters 37 s, level 0 6 s, level 1 25 s), and level 2 adds about 170 s and needs about 4 GB of memory.

A synthetic target is the best case: anny's parameter space holds its shape, and the residuals are smooth. A real character leaves larger residuals in the parameters and leans more on the detail layer.

### A real target

The first real test fitted a game-character replica: a slim figure 1.60 m tall from the soles to the top of the scalp, whose skin is 24,656 vertices in three parts (body, hands, head) with pointed ears, in a bind pose with the arms lowered and the forearms turned forward. Its head part also holds the eyes, the lashes and the inside of the mouth as primitives of other materials, which the loader now leaves out of the skin by their materials. 24 of its 40 landmarks map to the table; it gives each face landmark twice, as a joint position and as the nearest point of the skin. The errors below are in millimetres: forward from anny's vertices that see out of the fitted surface, reverse from every reference vertex that sees out of the reference (94 % of them; the others lie inside the mouth, behind the eyes or in the folds of the ears), split by the region of the reference. The parametric row measures all of anny's fitted vertices.

| Stage | Face forward mean / p95 | Face reverse mean / p95 | Ears reverse mean / p95 | Body forward mean / p95 | Body reverse mean / p95 | Hands reverse mean / p95 |
|---|---|---|---|---|---|---|
| parameters alone | 2.61 / 9.99 | 2.16 / 6.32 | 3.77 / 8.96 | 2.80 / 7.22 | 2.86 / 7.89 | 1.68 / 6.90 |
| level 0 | 0.24 / 0.69 | 0.38 / 1.14 | 1.23 / 4.20 | 0.58 / 1.51 | 0.85 / 2.23 | 0.47 / 1.30 |
| level 1 | 0.02 / 0.07 | 0.07 / 0.25 | 0.31 / 1.31 | 0.08 / 0.26 | 0.15 / 0.45 | 0.07 / 0.25 |
| level 2 | 0.01 / 0.03 | 0.02 / 0.07 | 0.08 / 0.32 | 0.03 / 0.10 | 0.05 / 0.16 | 0.02 / 0.07 |

The fitted landmarks lie within 0.5 to 3 mm of the face landmarks after the parameter stage (the inner eye corners 6 mm, in depth: the source's inner corner lies on the skin in front of anny's endocanthion), within 3 to 6 mm of the shoulders, elbows, wrists, knees and ankles, and 20 to 27 mm from the hips and the neck base, whose joint centres sit elsewhere in the source's rig. The detail levels then slide the vertices along the surface, so a landmark regressed from anny's level-0 vertices can lie 5 to 10 mm from the source's point while the surfaces agree. On two cores the fit took 37 s for the parameters, 6 s for level 0, 25 s for level 1 and 168 s for level 2; loading the model and baking 2048-texel textures on level 2 added about 20 s.

By eye, at the same cameras and light, level 1 and level 2 with the baked textures pass for the replica in the face, the profile and the full-body silhouette; level 0 keeps small lumps along the jaw and the cheeks. The test found and fixed: the eyes, lashes and mouth inside a head part counted as skin; a finger of anny slid onto its neighbour (the finger rounds); the gums showed through the upper lip (the mouth now follows the nearest skin); the eye openings stayed anny's, larger than the target's, and the pointed ear tips stayed uncovered (the reverse term); the eyeballs were larger than the target's (the eyeball spheres); and the eyes took the colour of the lids (the eye bake). What remains:

- The ears reach the pointed outline, but anny's ear has neither the folds nor the depth of the target's, and its inner surfaces crumple where the refinement pulls them onto folds they do not match. The ears are the weakest region at every level.
- The corners of the mouth keep a small notch on levels 0 and 1, where the inside of the lips meets the skin.
- Anny's eyeballs have 72 vertices each and a small region of the atlas, so the baked iris is coarse, more so on level 0.
- The baked colour stretches where the fit stretches anny's surface far from its own shape (the ear tips).

## Limits

- Hair, eyelashes, brows, teeth, the tongue and the eyeballs are left out of the fit; they attach to the fitted body in a later step. Clothing is left out too.
- The inside of the mouth and the eye sockets follow the skin around them and do not fit the target's.
- An ear far from anny's shape (a pointed ear with deep folds) reaches its outline on the detail levels, but not its folds.
- The body landmarks are joint centres of another rig, so they guide the pose and the proportions loosely. The face landmarks carry most of the placement.
- The fit assumes the contract's axes and an upright figure. A target in a very different pose from anny's (arms raised high, crouching) needs landmarks on the elbows, the wrists and the knees.
- The offsets are tied to the card's values: changing a slider afterwards moves the parametric body under fixed offsets.
- The texture bake does not yet carry the target's own normal map, uses per-triangle tangents (MikkTSpace is not implemented), and bakes no roughness or subsurface maps.
- The fit needs SciPy (the `viewer` or `examples` extra), and the texture bake needs `embreex` (the `viewer` extra).

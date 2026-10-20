# Fitting a reference mesh

`opensculptboy fit` reproduces a given humanoid figure with anny: it fits anny's parameters and pose to a target mesh of any topology, and it keeps what the parametric body cannot reach as a detail layer of per-vertex offsets on subdivisions of anny's mesh. The first target is a replica of a Dragon's Dogma 2 character, exported by the DD2DA tools as an appearance handoff folder, but any skinned or static triangle mesh in metres works.

```bash
uv sync --extra viewer                                       # SciPy for the fit, embreex for the texture bake
opensculptboy fit handoff/ --out arisen.json --detail 1      # the card, arisen.detail0.safetensors and arisen.detail1.safetensors
opensculptboy fit handoff/ --out arisen.json --textures      # also arisen.base_color.png and arisen.normal.png
opensculptboy export arisen.glb --character arisen.json --detail 1
opensculptboy export arisen.vrm --character arisen.json --detail 1 --author "Your name"
```

```python
from opensculptboy.fit import fit_handoff

result = fit_handoff("handoff/", detail=1)
print(result.errors[1]["reverse"]["mean"])   # mean distance from the target to the fitted surface, in mm
result.write("arisen.json")
```

## The handoff folder

The folder follows the draft contract `docs/contracts/APPEARANCE-HANDOFF.md` of DD2DA:

```
handoff/
    replica.glb        the skinned parts in their edited bind pose; metres, Y up, glTF axes (the figure faces +Z)
    textures/*.png     the textures of the parts
    appearance.json    {"landmarks": {name: [x, y, z]}, "parts": ..., "materials": ...}
```

`opensculptboy.fit.handoff.load_handoff` reads every part of `replica.glb` with its node transforms (the skinning is not applied, since the parts are in their bind pose) and turns it into Anny's frame: metres, Z up, the figure facing -Y, so that glTF's (x, y, z) becomes (x, -z, y). It keeps each part's UVs and its base colour and normal textures where the file has them.

`parts` may be a list of objects or an object keyed by part name. Each entry names a mesh or node of the GLB file (`"name"`, `"mesh"` or `"node"`) and may give a `"role"` (or `"kind"`, `"type"`, `"category"`). The roles `skin`, `body`, `head`, `face`, `hands` and `feet` are skin, and only the skin takes part in the fit. A part without a role is skin unless its name names hair, lashes, brows, a beard, teeth, the tongue, the eyeballs or clothing.

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

The face landmarks are points of the skin, and the body landmarks are joint centres, which sit differently in every rig; the weights reflect that. Without landmarks the fit places the target by its bounding geometry: the highest point (the crown), the points furthest to each side (the fingertips) and the soles, which the same rule finds on anny. This assumes the contract's axes, with the figure standing upright and facing +Z.

## The pipeline

1. **Placement.** With landmarks, a rigid alignment of anny's landmarks onto the target's (Umeyama 1991); without them, the centres of the bounding landmarks. The height phenotype starts at the value whose stature matches the target.
2. **Parameters.** Adam over the 6 phenotypes, the 256 local changes, the 113 face shapes, the turns of every bone but the root and the eyes, and a rigid placement, in 10 rounds of 30 steps. Each round finds the closest points again. The loss adds point-to-plane distances from anny's vertices to the target surface, point-to-plane distances from a sample of 20,000 target vertices to anny's nearest vertices, the landmark distances and small ridge penalties on the local changes, the face shapes and the bone turns. Pairs farther apart than a limit (50 mm in the first round, 10 mm in the last) or whose normals disagree drop out. The first three rounds free the placement, the phenotypes and the large joints alone.
3. **Non-rigid refinement.** Anny's whole mesh moves onto the target by a non-rigid ICP in the spirit of optimal-step NICP (Amberg et al., CVPR 2007): closest points pull the vertices, and a Laplacian term keeps the displacement smooth, with a stiffness that falls over four stages. The eyeballs, the eye sockets and the inside of the mouth take no data: the sockets and the mouth follow the skin through the Laplacian term, and each eyeball moves with the skin around it.
4. **Detail levels.** Each level of the detail layer repeats the refinement on its subdivided mesh, starting from the level below, with a lower stiffness.
5. **Residuals.** The registered mesh of each level goes back to the rest frame by the inverse of each vertex's blended bone transform, and the offsets are the registered rest mesh minus the parametric rest mesh. Added to the subdivided rest mesh and skinned again, they give back the registered mesh.

The card holds the fitted parameters, the fitted pose (the target's bind pose, which the GLB export plays as the animation named `pose`), and the detail layer with the placement and the measured errors.

The code lives in `src/opensculptboy/fit/`: `handoff.py` (the folder), `reference.py` (the target, its landmarks and the closest-point queries), `register.py` (stages 1 to 4, `fit_reference`, `FitResult.write`) and `residual.py` (the unposing, the offsets and `surface_error`). It reuses the Umeyama alignment and the uniform Laplacian of `anny.faces.authoring.nicp` and the closest-triangle search of `anny.hair`.

## The detail layer

A card with a detail layer uses character schema 2; a card without one is still written as schema 1, byte for byte as before, and both schemas load.

```json
{
  "schema": 2,
  "name": "arisen",
  "phenotype": {"gender": 0.62, "age": 0.48, "...": 0.5},
  "detail": {
    "levels": [
      {"subdiv": 0, "offsets": "arisen.detail0.safetensors", "error_mm": 0.26},
      {"subdiv": 1, "offsets": "arisen.detail1.safetensors", "error_mm": 0.04}
    ],
    "textures": {"base_color": "arisen.base_color.png", "normal": "arisen.normal.png", "subdiv": 1},
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

`--textures` (`opensculptboy.texture.bake.bake_fit`) bakes the target onto anny's UV atlas, on the mesh of the finest level: for each texel, a ray along the fitted surface's normal (Embree, as in the viewer's bakes) finds the target's skin within 10 mm. The base colour comes from the target part's base colour texture at the hit, and the normal map holds the target's shading normal in the tangent space of the fitted surface. Texels that no ray reaches take the nearest reached value. The GLB export puts the files in the material's `baseColorTexture` and `normalTexture`, and the VRM export in the MToon base, shade and normal slots.

## Accuracy

`test/test_reference_fit.py` fits a synthetic target: anny with random phenotypes, 12 random local changes, 10 random face shapes and a small pose, plus smooth random offsets of 3 mm RMS on level 0 and 0.8 mm RMS on level 1, evaluated on level 2, moved by a random rigid placement and remeshed (the vertices slide within the surface, and vertex clustering on a 5 mm grid merges them into 53,487 vertices and 108,410 triangles, without the eyeballs and the insides of the mouth and the eye sockets). The landmarks carry 1 mm of noise. The errors, in millimetres, are the distances from the fitted vertices to the target surface (forward) and from 20,000 target vertices to the fitted surface (reverse):

| Stage | Forward mean | Forward p95 | Reverse mean | Reverse p95 |
|---|---|---|---|---|
| parameters alone | 0.968 | 2.500 | 1.043 | 2.687 |
| level 0 | 0.258 | 0.689 | 0.465 | 1.161 |
| level 1 | 0.035 | 0.111 | 0.153 | 0.379 |
| level 2 | 0.015 | 0.051 | 0.072 | 0.178 |

The test holds level 1 under 0.5 mm in both directions. Without landmarks, the same target fits to 0.70 mm (forward) and 0.59 mm (reverse) on level 0. On two CPU cores the fit to level 1 takes about 70 s (parameters 37 s, level 0 6 s, level 1 25 s), and level 2 adds about 170 s and needs about 4 GB of memory.

A synthetic target is the best case: anny's parameter space holds its shape, and the residuals are smooth. A real character will leave larger residuals in the parameters and lean more on the detail layer.

## Limits

- Hair, eyelashes, brows, teeth, the tongue and the eyeballs are left out of the fit; they attach to the fitted body in a later step. Clothing is left out too.
- The inside of the mouth and the eye sockets follow the skin around them and do not fit the target's.
- The body landmarks are joint centres of another rig, so they guide the pose and the proportions loosely. The face landmarks carry most of the placement.
- The fit assumes the contract's axes and an upright figure. A target in a very different pose from anny's (arms raised high, crouching) needs landmarks on the elbows, the wrists and the knees.
- The offsets are tied to the card's values: changing a slider afterwards moves the parametric body under fixed offsets.
- The texture bake does not yet carry the target's own normal map, uses per-triangle tangents (MikkTSpace is not implemented), and bakes no roughness or subsurface maps.
- The fit needs SciPy (the `viewer` or `examples` extra), and the texture bake needs `embreex` (the `viewer` extra).

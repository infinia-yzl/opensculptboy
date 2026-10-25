# Sculpt topology

`sculpt` is OpenSculptBoy's own version of Anny's mesh: Anny's mesh with patches where Anny's surface follows a detailed character poorly. The shipped file holds one patch, at the inner corner of each eye. A second patch, at the root of the outer ear, builds and is tested, and stays off until its detail levels are as smooth as Anny's around the ear.

Every part of Anny keeps working on `sculpt`: the phenotypes, local changes, face shapes, facial actions, the rig and its skin weights, the landmarks, the detail levels, the texture bake, the glTF and VRM exports and the reference fit. A card made on `anny` loads on `sculpt` with the same parameters, and `opensculptboy convert` carries a card between both meshes, its detail offsets, ear UVs, eye poses and mouth poses included.

```bash
opensculptboy fit handoff/ --out card.json --topology sculpt          # fit the sculpt mesh
opensculptboy convert card.json --to sculpt --out card_sculpt.json    # a card and its detail offsets on the sculpt mesh (--to anny for the reverse)
python -m opensculptboy.topology.build                                # compose the enabled patches into the data file
```

## Design

- **Append-only patches.** `sculpt` keeps every vertex of Anny at its own index and in its own order. A patch removes some of Anny's quads, appends new vertices after Anny's last vertex, and adds new faces. Code that indexes Anny's vertices keeps working unchanged. A vertex of Anny that no face uses any more stays in the list as an unreferenced vertex, which the exporters and the fit leave out.
- **New vertices are fixed blends of Anny's vertices.** Each new vertex is a convex combination of at most 8 Anny vertices, its anchors. `anny.models.model_transforms.apply_retopology` takes exactly such rows, so every blend shape, facial action, face shape, skin weight and landmark carries over to the new vertices exactly, under every parameter setting. A convex combination can lie off Anny's surface, so a patch can bridge a pocket.
- **Quads and triangles.** Patches are written on the quad mesh (`anny-quads`), since the detail levels subdivide it. `sculpt-quads` is the patched quad mesh and `sculpt` its triangulation. A triangle is allowed where a patch joins Anny's faces, and the detail layer subdivides the mixed mesh.
- **UVs.** A patch keeps Anny's texture atlas: each new face takes its UVs from the same region of the atlas, so the bake, the tiles and the eye texture keep their layout.
- **Data file.** `src/opensculptboy/data/topology/sculpt.safetensors` and its sidecar `sculpt.json` hold the composed patches: the removed faces, the anchors and weights of the new vertices, the new faces and UVs, a label per new vertex and face, and a digest of Anny's quad mesh, which the loader checks along with the file's SHA-256.
- **Base-mesh indices.** The new vertices take MakeHuman base-mesh indices from 19158 upward, so a lookup of a base-mesh table at a new vertex fails loudly. `opensculptboy.topology.lookup` gives such tables on the new vertices through their anchors.

`src/anny` stays unchanged: `sculpt_model` builds Anny's `anny-quads` model data, patches it and loads it with `Anny.from_model_data`.

## Writing a patch

A patch is a module of `opensculptboy.topology.patches` with `NAME`, `ORDER`, `ENABLED`, `OPTIONS` and `build(base, **options) -> Patch`, authored on Anny's quad mesh at the default body. `src/opensculptboy/topology/README.md` holds the contract. `validate_patch` checks convex rows, closed holes, UVs and left-right symmetry, and `compose` refuses a face that two patches claim. The builder writes the file deterministically; `--with`, `--without` and `--only` choose the patches.

## The eye-corner patch

Anny has a deep pocket of skin at the inner corner of each eye, which shows as a dark hole on a detailed character. `patches/eye_corner.py` replaces the 49 quads of that pocket in each eye with 49 new quads: a caruncle (the pink mound) and a closed fold of the conjunctiva, with the same connectivity, so the new faces meet Anny's along the border. Each new vertex is the convex blend of nearby skin vertices that best reaches its designed position. The faces are labelled `eye_corner/caruncle`, `eye_corner/conjunctiva` and `eye_corner`.

`test.test_sculpt_eye_corner` checks the corner with orthographic depth renders from the front and from up to 35 degrees to the side and 20 degrees up or down: no view reaches the skin of the socket behind the new surface, at rest, with the eyes turned, in a blink, on four phenotype corners and two random faces. On the default character, the visible pocket shrinks from 52 to 15 mm², summed over the 15 views.

In a fit, the patch is placed at the blends of its anchors after each level (`register.PatchAnchors`), and then on the reference's tissue and eyeball by `register.seat_lids`, so that the inner corner meets the eyeball flush ([Fit to a reference](reference-fit.md#the-pipeline)). The eyes of `--eyes` draw the patch's caruncle and conjunctiva with their own wet material, and keep the whole corner in the skin's material on a card with a baked skin ([Eyes](eyes.md)).

Limits: a full blink folds the corner under the closed lids, as it folds Anny's own pocket. Without the eyes of `--eyes`, Anny's 72-vertex eyeball lies inside its sphere, so a ray can pass between its facets and the fold.

## The ear-root patch

`patches/ear_root.py` replaces the strip where the outer ear meets the head with a denser, regular band of quads, so that the root can bend smoothly from the head into the ear and the fit's ear split has a clean line to part along. Its middle row is labelled `ear_root/line`, which the fit takes as the ear's attachment line. On a synthetic pointed-ear target it lowers the ears' distance to the reference by about two fifths, and its folds on level 2 stay below Anny's own around the root. It stays off (`ENABLED = False`; `build --with ear_root` adds it) because its level 2 shows gentle ridges on the head above the helix, behind the ear and below the lobe, where the band fans out to the head's coarse border loop. `test.test_sculpt_ear_root` guards it.

## Code

- `opensculptboy.topology.patch`: `Patch`, `blend_rows`, `validate_patch`, `check_symmetry`.
- `opensculptboy.topology.sculpt`: `compose`, `read_sculpt_data`, `sculpt_data` and `use_sculpt_data` (for tests), `sculpt_model_data`, `sculpt_model`, `build_model`.
- `opensculptboy.topology.convert`: `to_sculpt`, `to_anny`, `convert_offsets`, `convert_eye_poses`, `convert_mouth_poses` and `convert_card` (`opensculptboy convert`), which carry vertex fields, detail offsets, the ear UVs and a reference's eye and mouth poses between both meshes.
- `opensculptboy.topology.lookup`: base-mesh tables, patch labels and the ear root line on a sculpt model, which the fit, the measures and the VRM export use.

`test.test_sculpt_topology` and `test.test_sculpt_downstream` guard the format, the exact rows, the meshes, the UVs, the detail levels, the GLB and VRM files, a synthetic fit on `sculpt` and the converter. The web viewer keeps Anny's mesh.

# Wearables

Wearables are clothes, footwear, headgear, capes and accessories that fit any Anny body. A character card lists them as loose pieces or as outfits, and the GLB and VRM exports dress the character in them. Each piece is a **wearable package** (`opensculptboy.wearables`), bound once to Anny's template and evaluated on every body.

```bash
opensculptboy export ada.glb --character ada.json                          # the card's active outfit
opensculptboy export ada.glb --character ada.json --outfit travel --hide cape
opensculptboy export ada.glb --character ada.json --show work/apron       # a piece of another outfit takes its slot
opensculptboy export ada.glb --character ada.json --covered keep          # keep the skin under the clothes
opensculptboy export ada_wardrobe.glb --character ada.json --wardrobe     # every piece as its own node
opensculptboy export ada.vrm --character ada.json --author "Your name"    # the outfit in a VRM avatar
opensculptboy wearables list --character ada.json                         # the outfits, pieces and groups
opensculptboy wearables bind tunic.glb --out wear/tunic --slot torso --licence CC0-1.0 --design-source "own design"
opensculptboy wearables import-mhclo shirt/shirt.mhclo --out wear/shirt     # a MakeHuman clothes asset as a package
opensculptboy wearables add ada.json wear/shirt.wearable.json --outfit casual --activate
opensculptboy wearables catalogue wear/ --thumbnails --query shirt         # the packages of a folder as JSON
opensculptboy wearables check ada.json                                    # the fit on test bodies, with a review sheet
opensculptboy wearables outfit export ada.json --outfit casual --out casual.outfit.json
opensculptboy wearables add bo.json casual.outfit.json --activate         # an outfit file on another card
```

## The package

A package is three files with one name:

| File | Holds |
|---|---|
| `<name>.wearable.json` | the description (format `opensculptboy.wearable/1`) |
| `<name>.glb` | the garment on the binding body at rest (metres, glTF axes), with its materials and textures; one mesh whose primitives share one set of vertices, one primitive per group and material, the group in the primitive's `extras` |
| `<name>.binding.safetensors` | the binding of every vertex, the rest vertices of the binding body and the cover masks |

| Field | Meaning |
|---|---|
| `format` | `opensculptboy.wearable/1` |
| `name`, `label` | the package's name and the name to show |
| `description`, `tags` | one line about the piece, and free words for it ("shirt", "long sleeves", "linen") |
| `slot` | `head`, `face`, `neck`, `torso`, `arms`, `hands`, `legs`, `feet`, `cloak`, `under` or `accessory` |
| `layer` | 0 (skin) to 100 (outermost), as in MakeHuman's z-depth: underwear 10, legs 30, torso 50, headgear 60, cloaks 80 |
| `provenance` | `public` or `private` (see "Provenance") |
| `licence` | `{spdx, declared, author, source, design_source, attribution}`, as the tools found or were given it; `spdx` is null when no licence is known, and the exports carry each `attribution` line as a credit |
| `binding` | `{body, method, data, bone?}`: `body` is `anny-template` (the default card), `card` (the card of `wearables bind --body`) or `reference`, `method` is `surface` or `rigid` |
| `weights` | `body` (the body's skin weights at the bound points), `rigid` (the one bone) or `own` (weights in the binding file, by bone name in `binding.joints`) |
| `covers`, `group_covers`, `cover_ray` | whether the binding file holds a cover mask, the groups that hold their own masks, and whether a ray test confirms the masks |
| `groups` | the named groups of the garment |
| `hides` | rules toward other pieces: `{"piece": id, "groups": [name, ...]}` |
| `springs`, `constraints` | spring chains and constraints for the VRM export; packages keep them, and a later version writes them |

The JSON Schema of the description is `src/opensculptboy/data/wearables/wearable-1.schema.json` (`opensculptboy.wearables.format.SCHEMA_PATH`). A package's GLB file embeds its textures and holds MikkTSpace tangents when a material has a normal map.

The tools record the licence they find or are given; choosing assets and following their licences is the user's responsibility.

### Binding

- **Surface.** Each garment vertex stores three vertices of a triangle of Anny's template, the barycentric coordinates of its bound point and its offset from that point in the point's frame: the tangent (the triangle's first edge, made orthogonal to the normal), the bitangent and the normal blended from the corners' vertex normals. The offset is divided by the frame's scale, the blended length of the corners' edges. On another body the same point and frame give the vertex back, so the offset turns and stretches with the skin. On the binding body the garment returns to its own vertices exactly.
- **Rigid.** The vertices are stored in the rest frame of one bone and move with it, for pieces that keep their shape (a circlet, a buckle, a staff).

### Fitting

`opensculptboy.wearables.fit.dress` evaluates the worn pieces on a body from the lowest layer up. Each vertex takes the body's skin weights at its bound point (the strongest four, or eight with `--influences 8`, renormalised), the package's own weights (`weights: own`) or the one bone of a rigid piece. A **push-out** then moves every vertex of a surface-bound piece outside the body and outside the shown pieces of lower layers: at least by a 1 mm gap, or by as much as the vertex lay outside them on the binding body, whichever is less. A piece therefore keeps the relations it was made with: a tight piece never sinks into a heavier body, and a strap that sinks into a coat on the binding body may sink into it on every body. Rigid pieces take no push-out. A rigid piece of a lower layer that was bound on another body has no known place on the binding body, so the push keeps the full gap from it. A Laplacian fade spreads each push over two rings of neighbouring vertices, and a last pass makes every vertex meet its target.

### Covered skin

A package covers skin through a mask over the vertices of Anny's template: `covers` holds the skin under the groups that always show, and `group_covers` adds the skin under a group while that group shows. With `cover_ray` set, a vertex of the masks counts only where its normal ray also meets the dressed piece within 3 cm. A package without a `covers` mask covers the skin whose normal ray meets the dressed piece within 3 cm, and its group masks add to that. The ray tests see only triangles of opaque materials, since a masked or blended material (lace, a sheer stocking) may let the skin show. An export removes a body triangle when all three of its corners lie under a shown piece; `--covered keep` keeps every triangle (for engines that swap clothes at run time). A wardrobe file keeps the body whole.

## Outfits on the card

Character schema 4 adds four fields ([Character card](character-card.md)):

```json
"outfits": [
  {"id": "travel", "label": "Travel clothes", "pieces": [
    {"piece": "tunic", "file": "wear/tunic.wearable.json", "enabled": true},
    {"piece": "cape", "file": "wear/cape.wearable.json", "enabled": true}
  ]}
],
"active_outfit": "travel",
"wearables": [{"id": "ring", "file": "wear/ring.wearable.json", "enabled": true}],
"wearable_overrides": {"travel/cape": {"enabled": false}}
```

An outfit may also hold a `description`, `tags` and the `source` outfit file it came from, and each of its pieces the settings of "Outfit files": `overrides` of the package's `layer` and `label`, `hides` rules added to the package's own, and `springs` in place of the package's own.

A piece is named by its reference: `<outfit>/<piece>` for a piece of an outfit, the bare id for a loose piece. A bare id also names the piece of that id in the worn outfit, so `--hide cape` hides `travel/cape`. The export resolves what shows in this order: the worn outfit's pieces as the card enables them, the enabled loose pieces, the card's overrides, then `--hide` and `--show`.

- **Slot replacement.** A piece shown from another outfit, and a loose piece, replace the pieces of the same slot that come from elsewhere. Pieces of `accessory` never replace.
- **Hides rules.** While piece A shows, the groups that its `hides` rules name are hidden on piece B. Hiding A shows them again. A rule names a piece of its own outfit by its id and a piece of another outfit by `<outfit>/<piece>`. A piece whose every group the shown pieces hide does not show: the files leave it out, the summaries mark it `shown: false` with its hidden groups, and its own rules hide nothing (underwear that trousers hide whole).

## Files

- **GLB.** Each shown piece is a mesh node of its own, named by its reference and skinned by the character's skin, with glTF PBR materials and MikkTSpace tangents for its normal maps. The node's `extras.opensculptboy_wearable` and the scene's `extras.opensculptboy.wearables` record the pieces (each by its reference `ref`, with its `piece` id and its `outfit`), their hidden groups and the removed body triangles; the VRM file records them alike. The summary of a GLB or VRM export lists the shown pieces by reference. The ground lift takes footwear into account.
- **Wardrobe GLB** (`--wardrobe`). Every piece of every outfit is a node, and every group that a rule can hide is a node of its own (`<piece>.<group>`), all at the root, with the body whole. The default scene holds the body and what the worn outfit shows, so Blender and other importers open the character as it is dressed; a second scene holds every piece, for tools that switch scenes or let the user show nodes one by one. `--node-visibility` writes one scene with every piece instead and marks the pieces and groups that the worn outfit does not show with `KHR_node_visibility` (`visible: false`). The Khronos validator reports that extension as unknown (an info). The glTF importer of Blender 5.2 reads every node of every scene and ignores the extension; `blender --python src/opensculptboy/export/blender_skin.py -- wardrobe.glb` imports a wardrobe file and hides the pieces and groups that its extras mark as not shown (`opensculptboy_wearable.shown`, `opensculptboy_wearable_group.shown`).
- **VRM.** Each shown piece is a mesh bound into the T-pose with the file's skin weights at its bound points (the inverse bind, so that the avatar posed at Anny's rest pose shows the piece's rest vertices exactly) and an MToon material per garment material, its normal map without vertex tangents (see [VRM avatars](vrm.md#what-the-file-holds)). Pieces count against the budgets of [VRM avatars](vrm.md). Spring bones and constraints come later.

## Make a wearable

`opensculptboy wearables bind` turns a garment modelled in Blender (or any tool that writes GLB files) on the body of a card into a package.

1. Export the body to model on: `opensculptboy export base.glb` gives the default card. For another card, export it with `--character card.json --bare` and bind with `--body card.json`.
2. Import `base.glb` into Blender (File > Import > glTF 2.0) and model the garment on the body at rest. Keep the scene in metres and leave the body where the import puts it.
3. Name the groups. By default each material is a group, named by the material. `--groups-from nodes` names a group by its mesh object, and `--groups-from extras` reads a custom property `group` of the mesh data or the object (export with Include > Custom Properties). Hides rules of other pieces and the cover masks work per group.
4. Unwrap UVs for textured materials. A textured material on a mesh without UVs is an error; a garment without any UVs binds with a warning.
5. Export the garment objects alone: File > Export > glTF 2.0, format glTF Binary, Include > Selected Objects, Transform > +Y Up. For `--weights own`, parent the meshes to the imported armature, paint their weights and select the armature too.
6. Bind the file and list the package on a card (see "Outfits on the card").

```bash
opensculptboy export base.glb
opensculptboy wearables bind tunic.glb --out wear/tunic --slot torso --layer 50 --label "Linen tunic" \
  --licence CC-BY-4.0 --author "Your name" --source https://example.org/tunic --design-source "own design" --hides cape:hood
opensculptboy wearables bind circlet.glb --out wear/circlet --slot head --layer 60 --rigid head --licence CC0-1.0 --design-source "own design"
```

The frame of the file is the frame of the export: metres, glTF axes (Y up, the figure facing +Z). The export stands the body on the floor at its default pose (its lowest vertex at y = 0, a lift of 0.819 m for the default card), and its bind pose keeps the body centred as Anny's template. `--frame auto` (the default) takes the frame in which the garment lies nearer the skin, so a garment modelled on either pose binds; `--frame floor` and `--frame centred` choose one. The reader applies the node transforms and the skins of the file at its default pose, as an importer shows them, and skips a mesh that lies on the skin (the body itself, exported by mistake).

| Option | Effect |
|---|---|
| `--out STEM` | the package's three files `STEM.wearable.json`, `STEM.glb` and `STEM.binding.safetensors`; `--name` defaults to the stem, `--label` to the name |
| `--slot`, `--layer` | the slot (one of the slots above) and the layer (default 50) |
| `--rigid BONE` | bind every vertex to one bone, for pieces that keep their shape; surface binding by default |
| `--weights body\|own` | the body's skin weights at the bound points (default), or the file's `JOINTS_0` and `WEIGHTS_0` by joint name, which must name bones of the body's rig |
| `--groups-from materials\|nodes\|extras` | how the triangles form groups (default `materials`) |
| `--covers auto\|none` | `auto` (default): each group's mask holds the skin vertices whose normal ray meets the group's opaque triangles within 3 cm on the binding body, and the package sets `cover_ray`; `none`: the piece covers no skin |
| `--hides PIECE:GROUP,...` | hides rules toward other pieces; repeatable |
| `--licence SPDX`, `--author`, `--source URL`, `--design-source TEXT` | the licence block, recorded as given (each optional) |
| `--tag WORDS`, `--description TEXT` | the package's tags (repeatable, or comma-separated) and its one-line description |
| `--accept-quality`, `--texture-limit PIXELS`, `--normal-green auto\|keep\|flip` | the quality checks (see "Quality") |
| `--provenance public\|private` | `public` by default; a private package never goes into a checkout of this repository (see "Provenance") |
| `--body CARD` | bind to the body of a card at rest, its detail level 0 included; the default card by default |
| `--frame auto\|floor\|centred` | the frame of the garment file (default `auto`) |
| `--max-distance MM` | the limit of the refusal below (default 100) |

The command prints a JSON report: the quality items (`quality`, see "Quality"), the vertices, triangles, materials and groups, the largest and the 95th-percentile distance of the garment's vertices from the skin (`bind_distance_mm_max`, `bind_distance_mm_p95`), the vertices further than 50 mm from the skin (`far_vertices`), the covered skin vertices, the frame and its lift, the round trip (`round_trip_mm_max`: the written package read back and evaluated on the binding body, against the garment's vertices) and the warnings, which also go to the error stream.

- A surface garment whose median distance from the skin exceeds `--max-distance` (for `--rigid`, its nearest vertex) is refused: it was modelled on another body, at another scale or in another frame. The message gives the distance in each frame it tried.
- Vertices further than 50 mm from the skin bind with a warning: they follow the skin at their bound points and may stretch on other bodies.
- A mesh without normals takes normals computed from its triangles, with a warning.
- An unknown slot, an unknown bone for `--rigid`, own weights without a skin, with joints that the rig lacks or with vertices that carry no weight, and a malformed `--hides` item are errors.
- A textured material on a mesh without UVs is a quality error (`missing_uvs`); with `--accept-quality` the package keeps the material without its textures.

## Free outfits

MakeHuman's community shares clothes as asset packs: <https://static.makehumancommunity.org/assets/assetpacks/index.html> lists them (shirts, pants, skirts, suits, dresses, shoes, gloves, underwear), each pack under CC0 or CC-BY. A clothes asset is a folder with a `.mhclo` file, its `.obj` mesh, its `.mhmat` material and the textures. A CC-BY asset needs a credit to its author ([MakeHuman's FAQ](https://static.makehumancommunity.org/oldsite/faq/what_do_i_need_to_do_when_i_use_a_ccby_asset.html) reads it as CC BY 4.0). MakeHuman's own source and some assets are under AGPL. The licences are the user's responsibility: the import records the licence, the author and the source that the header and the pack metadata name (the licence as `text`, its name as `declared` and its SPDX id as `spdx`), and writes an `attribution` line for every asset whose header names any of them, whatever the licence; the GLB export lists those lines in `extras.opensculptboy.wearables.credits` and the VRM export adds them to the file's credits.

Two commands put a free outfit on a character:

```bash
opensculptboy wearables import-mhclo packs/shirts01/toigo_fisherman_sweater/toigo_fisherman_sweater.mhclo --out wear/sweater
opensculptboy wearables import-mhclo packs/pants01/some_jeans/some_jeans.mhclo --out wear/jeans --tag denim
opensculptboy wearables add ada.json wear/sweater.wearable.json wear/jeans.wearable.json --outfit casual --label "Casual" --activate
opensculptboy export ada.glb --character ada.json
```

### Import a MakeHuman asset

`opensculptboy wearables import-mhclo FILE.mhclo [--out STEM]` reads the asset and writes a package bound to the default body (`binding.body` is `anny-template`).

- **Header.** `name`, `uuid`, `description`, `tag` lines, `z_depth`, `obj_file`, `material`, `basemesh` (only `hm08`, MakeHuman's default base mesh) and the licence keys `author`, `license`, `homepage` and `copyright`, as keys or in comment lines (`# license CC0`); MPFB2 writes the description in a comment line too (`# description: ...`), which a `description` key replaces. The package takes the name, the label, the description and the tags from them (`--name`, `--label`, `--description` and `--tag` add or replace them), the layer from `z_depth` (50 without one, as MakeHuman uses), and the slot from the words of the name and the tags (`--slot` sets it): a garment word or its plural (`underpants` and `briefs` are underwear, `stockings` and `tights` legs, `hats` head, `tank`, `halter`, `turtleneck`, `camisole`, `cami`, `bodice`, `corset`, `polo`, `tee`, `crop` and `top` torso), else the longest garment word inside a compound name (`hoodedjacket` is torso), where a word of three letters counts only at the end (`tophat`, `sportsbra`), else `short` (legs, as in `short01`), which counts only without another garment word (`short sleeves`). A tube top is torso by `top`; `tube` alone names no slot (`tube socks`). The names of MakeHuman's clothes and of the community packs Shirts 01 and Hats 01 check the words (`test_slot_words`). When the name and the tags name no slot, the import tries the folder name (`toigo_keyhole_tank_top`), the file stem and the pack's category, in this order; `slot_from` gives the word and where it was found, and without a slot the import fails with `unknown_slot`.
- **Licence.** The header's licence text may carry words after the name: a part in brackets, a URL, `see also ...` or a clause after `;` (`# license AGPL3 (see also http://www.makehuman.org/doc/node/external_tools_license.html)`). The licence block keeps the text as written (`text`), the name before those words (`declared`, `AGPL3`) and the SPDX id of the name (`spdx`): `CC0` is `CC0-1.0`; `CC-BY`, `CC BY 4.0` and `Creative Commons Attribution-ShareAlike 4.0` are `CC-BY-4.0` and `CC-BY-SA-4.0`, where a CC licence without a version is 4.0, as MakeHuman's FAQ reads its community's CC-BY; `AGPL3`, `AGPLv3` and `AGPL-3` are `AGPL-3.0-only`, with `+` or `or later` `AGPL-3.0-or-later`, and GPL and LGPL with a version alike; `MIT`, `Apache 2.0`, `BSD 2-Clause`, `BSD 3-Clause`, `MPL 2.0`, `Unlicense`, `ISC`, `Zlib` and `OFL` have their ids. Another name keeps `spdx` null. The attribution line names the licence by `declared`.
- **Pack metadata.** A MakeHuman asset pack holds its items under `clothes/<item>/` and its metadata in `packs/<pack>.json`, an object keyed by item folder name with `author`, `category` (`Shirt/T-shirt/Sweater`), `description`, `license`, `original_author`, `original_source`, `source` (the item's page) and other fields. The import walks up from the `.mhclo` file, at most four folders, to the folder that holds `clothes/` and `packs/`, and reads the entry of the item's folder name or file stem; `--pack-json FILE` names the file instead. The entry gives the package's description (in place of the header's; `--description` replaces both), tags from the category (split at `/`), the author (credited after the header's author when the two differ; `--author` replaces both), the source (in place of the header's homepage; `--source` replaces it), `original_author` and `original_source` (credited as `original by ...`) and the pack's licence, recorded apart from the file's in `licence.pack` (`file`, `item`, `spdx`, `declared`, `text`, `author`, `source`, `original_author`, `original_source`). The file's licence comes first; a file without one takes the pack's. When both name a licence and the two differ, the info item `licence_mismatch` notes it. A `--pack-json` file that lists the item under neither name gives the warning item `pack_item_missing`. The result holds the entry under `pack` (null without one).
- **Binding.** After `verts`, one line per `.obj` vertex: three base-mesh vertex indices, three weights and an offset, or one index alone. On a body `h` in MakeHuman's frame (decimetres, Y up) a vertex lies at `w1 h[v1] + w2 h[v2] + w3 h[v3] + S d`. `S` is diagonal: for each axis n of `x_scale`, `y_scale` and `z_scale` (`v1 v2 length`), `|h[v1][n] - h[v2][n]| / length`, so that the offset stretches with the body; without scale lines it is the identity, and `shear_x`, `shear_y` and `shear_z` (`v1 v2 x1 x2`) give `(h[v2][n] - h[v1][n]) / (x2 - x1)`. The import evaluates the binding on the default body, then binds the garment to the nearest triangles of the skin with the code of `wearables bind`. The `.obj` vertices may lie away from the place the binding gives: MakeClothes 2 exports the human standing on the ground, about 8 dm above the feet of `base.obj`. The report gives that mean translation as `source_offset_mm` (Anny's frame, with the info item `source_offset` above 5 mm), and with it taken out compares the `.obj` vertices with the binding evaluated on `base.obj` (`source_fit_mm_median`, `source_fit_mm_max`) and on the default body (`source_fit_body_mm_median`, `source_fit_body_mm_max`): a large value means a file that does not follow the format, or a garment modelled on another body. The binding alone places the garment.
- **Base mesh.** Anny's template is MakeHuman's `base.obj` (19,158 vertices) in Anny's frame. Anny's default topology keeps the 13,718 vertices that its faces use (body, eyes and tongue); it leaves out the helper geometry (`helper-tights`, `helper-skirt`, `helper-hair` and the others) and the vertices of its nudity edits. Many assets bind to the helpers. The MPFB2 targets that Anny bundles move the helper vertices as they move the skin around them, so the import takes the vertices that Anny lacks from those targets with the card's blend-shape coefficients. On other bodies the package follows the skin triangles under it: on three random bodies a patch bound to `helper-tights` stays within 6 mm (median) and 12 mm (largest) of the helper that the targets give (`test_helper_geometry_follows_the_targets`).
- **Covered skin.** `delete_verts` (numbers and ranges `a - b`) becomes the package's `covers` mask on Anny's rows; deleted vertices outside Anny's rows are counted in the report. Without `delete_verts` the import uses the ray test of `wearables bind`. `--covers ray` always uses the ray test and `--covers none` covers no skin.
- **Material.** The `.mhmat` file becomes one glTF PBR material and one group named by it: `diffuseColor` (sRGB, made linear) and `diffuseTexture` give the base colour and `opacity` its alpha; `roughnessFactor` is `clamp(1 - shininess, 0.04, 1)`, 1 when `specularColor` is 0, and a `roughness` key wins; MakeHuman's material reader ignores `diffuseIntensity` and `specularIntensity` as deprecated keys, so the import leaves them out too and reports them (`deprecated_key`); `metallic` gives the metallic factor; `transparencymapTexture` becomes the alpha of the base colour texture; `transparent` gives BLEND, and `alphaToCoverage`, or an alpha map on an opaque material, gives MASK with cutoff 0.5; `backfaceCull False` gives a double-sided material; `normalmapTexture` with `normalmapIntensity`, `aomapTexture` and `emissiveColor` give the normal map, the occlusion and the emission. A bump map (`bumpmapTexture`, as MakeHuman reads it, or `bumpTexture`, as the community packs write it) is a height map: when the material has no normal map, the import converts it to a tangent-space normal map in the OpenGL convention of glTF (green up), the normal of the surface `z = s h(x, y)` with the height `h` in [0, 1] and central differences in texels, wrapped at the borders. The strength `s` is `--bump-strength` (2 by default: a step from black to white rises two texels) times the material's `bumpmapIntensity`, and the fix item `bump_converted` records it (`value`), as does `extras.mhclo.bump_strength` of the package; `--normal-green flip` leaves the converted map as it is. A bump map beside a normal map is dropped, as MakeHuman draws the normal map alone, and so is a bump map that `shaderConfig bump false` turns off. Every other texture key has no glTF PBR slot (`specularmapTexture`, `displacementmapTexture`, `litsphereTexture` and any other key that ends in `Texture`): every texture key is either mapped or reported, with one `map_dropped` item per dropped map (`slot` names the key). Textures in other formats than PNG and JPEG are converted to PNG.

`--dry-run` checks and binds the asset and prints the result without writing the package (`"written": false`). The command prints one JSON result on stdout (`"ok": true`, the package, `dry_run` and `written`, the report of `wearables bind`, `licence`, `slot_from`, `pack`, `mhclo` and `quality`); warnings go to the error stream and stay in the result. A failure prints `{"ok": false, "error": {"code", "message", "hint"}}` and exits with status 1; the codes include `file_not_found`, `malformed_header`, `malformed_pack`, `malformed_binding`, `malformed_obj`, `binding_mismatch`, `unknown_basemesh`, `unknown_slot`, `too_far_from_skin`, `missing_uvs`, `private_in_repository`, `file_error`, `invalid_input` and `usage` (status 2). An unexpected error prints the same result with the code `internal_error`, the exception's message and status 2; the commands `add`, `remove` and `outfit` report their failures alike.

Stdout holds the JSON result alone, so that a program can read it whole: every `wearables` command, `export` and the other commands that print a JSON result send what the libraries print while the command runs (Warp's start-up lines and module loads, and the output of native code) to the error stream, and print the result when the command ends. `test.test_wearables_import.TestStdout` runs the commands in new processes and reads their stdout as one JSON value.

Sources of the format: MakeHuman 1.x `shared/proxy.py` (the binding, the scale and shear references, `delete_verts`) and `makehuman.py` (`LicenseInfo`, the licence keys) at <https://github.com/makehumancommunity/makehuman>, and MPFB2 `src/mpfb/entities/clothes/mhclo.py` (the header keys, the single-index bindings) at <https://github.com/makehumancommunity/mpfb2>.

### Put pieces on a card

```bash
opensculptboy wearables add ada.json wear/sweater.wearable.json wear/jeans.wearable.json --outfit casual --label "Casual" --activate
opensculptboy wearables add ada.json wear/ring.wearable.json --loose
opensculptboy wearables add ada.json wear/coat.wearable.json --outfit casual --replace --dry-run
opensculptboy wearables remove ada.json casual/jeans ring
opensculptboy wearables remove ada.json --outfit casual
opensculptboy wearables outfit ada.json --activate none
```

- `add` puts packages into an outfit (`--outfit NAME`, created when missing, with `--label`) or as loose pieces (no `--outfit`, or `--loose`). A piece takes the package's name as its id, and its file is written relative to the card. `--activate` makes the outfit the worn one. A piece of the same id with another file is an error (`piece_exists`) unless `--replace`, which also takes out the pieces of the same slot in the same place (an `accessory` piece only by id) with their overrides. Each package is read and checked first.
- `remove` takes pieces off by reference (`<outfit>/<piece>`, a loose piece's id, or a bare id of a piece of the worn outfit) with their overrides, or a whole outfit with `--outfit`. A reference that the card lacks is reported in `missing` and changes nothing.
- `outfit --activate NAME` sets `active_outfit` (`none` for no outfit).
- Each command keeps the card's other fields as they are, raises an older card to schema 4 when it adds wardrobe fields, and checks the result as a card. `--dry-run` prints the change without writing it. A second run with the same arguments changes nothing (`"changed": false`).
- A private package never goes on a card stored inside a checkout of this repository (`private_in_repository`).
- Each command prints one JSON result (`ok`, `changed`, `written`, `schema` before and after, `added`, `updated`, `removed`, `unchanged` or `missing`, `removed_overrides` (the overrides of the pieces taken off), `warnings` and the new `wardrobe` fields); errors are JSON as for `import-mhclo` (`card_not_found`, `invalid_card`, `package_not_found`, `invalid_package`, `piece_exists`, `ambiguous_piece`, `unknown_outfit`, `bad_option`, `bad_outfit`).

## Quality

`wearables bind` and `wearables import-mhclo` check each garment before they bind it (`opensculptboy.wearables.quality`) and report under `quality`: `errors`, `warnings`, `infos`, the codes of the `fixes`, and the `items`, each with a stable `code`, a `severity` (`error`, `warning` or `info`), a `message` and a `hint`. An error-severity item fails the command with its code unless `--accept-quality` is passed.

| Area | Codes |
|---|---|
| Geometry | `degenerate_triangles_removed`, `unused_vertices_removed`, `duplicate_vertices_merged` and `duplicate_triangles_removed` (fixes; a duplicate triangle has the same three vertices in the same turn), `non_manifold_edges`, `parts`, `loose_fragments`, `counts`, `triangle_budget` (with the default body's 27,420 triangles against the VRM 1.0 budget of 70,000, a warning, and the VRM 0.x budget of 32,000, an info), `no_triangles` (error) |
| Normals | `winding_fixed` (triangles wound against their neighbours turned over), `normals_computed` (angle-weighted, welded by position, so that UV seams and material borders stay smooth), `hard_edges` (split vertices whose file normals differ, kept), `normals_against_winding`, `part_faces_inward` (a part that faces into the body; an inner lining may do so on purpose) |
| UVs | `missing_uvs` (error), `uv_outside_unit_square`, `uv_degenerate`, `uv_flipped` (triangles mirrored against the rest of their UV island; mirrored halves that meet at a seam count as none), `uv_overlap` (a UV area covered twice in the same direction), `uv_mirrored_overlap` (an info: symmetric halves that share a texture), `texel_density` or `texel_density_spread` (the 95th over the 5th percentile above 4) |
| Materials | `roughness_mapped` (with the formula), `alpha_mode`, `double_sided`, `alpha_map_merged`, `alpha_map_on_opaque`, `bump_converted` (`import-mhclo`: a bump map converted to the normal map; a fix with the strength), `map_dropped` (a texture key without a glTF slot, or a bump map beside a normal map), `material_missing`, `deprecated_key` |
| Textures | `texture_missing`, `texture_unreadable`, `texture_converted`, `texture_not_power_of_two`, `texture_too_large` (above `--texture-limit`, 4096 by default), `linear_texture_tagged_srgb` (a normal, roughness or occlusion map tagged as sRGB; glTF reads it as linear), `base_colour_tagged_linear`, `alpha_unused` (a base colour texture with alpha on an opaque material), `alpha_missing` (a transparent material without alpha), `normal_map_not_tangent_space`, `normal_map_convention`, `normal_map_directx` or `normal_green_flipped`, `textures_embedded`, `tangents_mikktspace`, `tangents_degenerate` |
| Licence | `licence_mismatch` (`import-mhclo`: the file's licence and the pack's differ; an info), `pack_item_missing` (`--pack-json` names a file that does not list the item; a warning) |
| Placement | `source_offset` (`import-mhclo`: the `.obj` lies more than 5 mm from the place its binding gives, as in an export on the ground; an info) |

The green channel of a normal map follows the OpenGL convention in MakeHuman and in glTF (green points up the image). The check measures the correlation between the change of red up the image and the change of green to the right, which a relief gives alike: positive for OpenGL, negative for DirectX. `--normal-green auto` (the default) flips a map that reads DirectX with a correlation below -0.25, `keep` leaves it with a warning, and `flip` always flips.

## Catalogue

`opensculptboy wearables catalogue FOLDER` lists the packages of a folder and its sub-folders, so that a person or an agent can choose pieces by their words, their colours and their quality. The command prints one JSON result (`format` `opensculptboy.catalogue/1`) whose `pieces` list holds one entry per package, sorted by path:

```bash
opensculptboy wearables catalogue wear/                                    # every package
opensculptboy wearables catalogue wear/ --slot torso --query "linen shirt" # filters
opensculptboy wearables catalogue wear/ --thumbnails                       # wear/NAME.thumb.png
opensculptboy wearables catalogue wear/ --mhclo packs/                     # and the assets not imported yet
```

| Field | Meaning |
|---|---|
| `kind`, `path` | `package` (or `mhclo` for an asset not imported yet) and the file's path relative to the folder |
| `name`, `label`, `slot`, `layer`, `tags`, `description`, `provenance` | the package's fields |
| `colours` | `[{"material", "hex"}]`: the mean base colour of each material, the base colour factor times the mean of the base colour texture (weighted by its alpha), as sRGB hex |
| `groups`, `vertices`, `triangles`, `materials` | the garment's groups and counts |
| `binding`, `weights`, `hides` | the binding's body and method, the weights and the hides rules |
| `licence` | `spdx`, `declared`, `author`, `source` and `credit`, the line that the exports write (null without one) |
| `quality` | the checks of "Quality" run on the package as it stands: `errors`, `warnings`, `infos`, the `codes` of each severity and the error and warning `items`; the package holds the fixes of its import, and its normals count as the file's, so normals that the import computed are also tested against the winding (`normals_against_winding`); `--no-quality` leaves them out |
| `thumbnail` | the path of `NAME.thumb.png` next to the package, or null |

- **Filters.** `--slot` keeps the slots named (repeatable or comma-separated), `--tag` keeps the pieces with every tag named, and `--query` keeps the pieces whose name, label, tags or description hold every word; a word of more than three letters that ends in `s` also matches without it (`shirts` finds `shirt`). `count` gives the entries kept and `total` the entries found.
- **Thumbnails.** `--thumbnails` dresses each package on the default body and renders it from the front, turned 30 degrees, framed on the piece, with the skin under it left out (`--thumbnail-size`, 256 by default). A thumbnail newer than the package's three files is reused (`thumbnails.cached`); `--refresh` renders every one again. Without `--thumbnails`, an existing thumbnail is listed.
- **Assets not imported.** `--mhclo FOLDER` (repeatable) adds the `.mhclo` assets of a folder that no listed package was imported from, by the asset's uuid, else its name. An entry holds what the header, the pack metadata and the material give (the slot with `slot_from`, the layer from `z_depth`, the tags, the description, the colour and the licence), `imported: false` and the `import` command.
- A package that cannot be read goes into `skipped` with its `code` and `message`. The errors are `folder_not_found`, `unknown_slot` and `usage`.

## Check

`opensculptboy wearables check CARD` checks the worn outfit of a card (or `--outfit NAME`) before an export: the quality of every worn piece, and the fit on test bodies at rest, in the T-pose of the VRM export and through the built-in clips. It prints one JSON report (`format` `opensculptboy.check/1`) and writes a review sheet. The exit status is 0 when every gate passes, 1 when a gate fails, and 2 on an error.

```bash
opensculptboy wearables check ada.json                                         # every body, clip and gate; ada.check.png
opensculptboy wearables check ada.json --outfit travel --sheet travel.png
opensculptboy wearables check ada.json --bodies card,default --clips walk --frames 2   # a quick round
```

- **Test bodies.** `--bodies` takes a comma-separated list; the default is every body: `card` (the card's own body with its detail offsets), `default` (the default card's body), and `weight-0`, `weight-1`, `muscle-0`, `muscle-1`, `height-0` and `height-1` (the card with that slider at 0 or 1). `SLIDER=VALUE` sets any phenotype slider of the card's model. Each body is dressed as an export dresses it (`opensculptboy.wearables.fit.dress`).
- **States.** Rest; the VRM T-pose (`--no-tpose` leaves it out); the card's own pose when it has one; and `--frames` frames (4 by default) of each clip of `--clips` (comma-separated, every clip of the pose library by default, `none` for none). The body and the pieces move by linear blend skinning with the same bone transforms: the body with its own weights, the pieces with the weights of the dressing. In the T-pose both take the weights of the VRM file without the refinement, blended at the pieces' bound points.
- **Skin through.** A body vertex under a piece's cover mask (taken on that body at rest, see "Covered skin") counts when it lies more than `--tolerance` (2 mm) outside every piece that covers it.
- **Layers through.** A vertex of a lower piece counts when an upper piece covers it (the line along its normal meets the upper piece's visible triangles within 3 cm on either side, on that body at rest) and it lies more than the tolerance outside the upper piece.
- **Partners.** The distance outside a piece is measured from the vertex's partner: the nearest point of the piece at rest on the line along the vertex's normal, within 3 cm on either side, else its closest point on the piece within 5 cm (a vertex further away takes no part), carried into each pose by the piece's triangle and barycentric coordinates, along the piece's normal there, turned away from the body where it faces the skin at rest (a piece wound inside out, the far side of a double-sided panel). A partner keeps each vertex to the part of the piece above it, so that an arm lowered against the torso is measured against its sleeve and never against the side of the shirt.
- **Regions.** Each vertex belongs to the region of its strongest bone: `head`, `torso`, `hips`, `upper_arm.L`, `forearm.L`, `hand.L`, `thigh.L`, `shin.L`, `foot.L` and their right sides.

| Gate | Passes when |
|---|---|
| `quality` | no worn piece has a quality item of error severity |
| `skin_rest`, `layers_rest` | no vertex counts at rest on any body |
| `skin_tpose`, `layers_tpose` | at most `--pose-limit` per cent (1 by default) of the covered vertices count in the T-pose of any body |
| `skin_clips`, `layers_clips` | at most `--pose-limit` per cent count on any sampled frame, and in the card's own pose |

The report holds `pass`, `failed` (the names of the failed gates), `gates` (each with `pass`, `limit`, `unit`, `value` and `worst`: the `body`, `state`, `frame`, `count`, `share_percent`, `largest_mm`, `region` and `piece`, where a layer fault names `LOWER through UPPER`; a gate without states is `skipped`), `pieces` (each worn piece with its quality summary as in "Catalogue"), `states` (per body and state: `skin` with `showing`, `covered`, `share_percent`, `largest_mm`, `pieces` and `regions`, and `layers` with `cutting`, `covered`, `share_percent`, `largest_mm`, `pairs` and `regions`), `bodies`, `clips`, `notes` and `sheet`.

- **Review sheet.** `--sheet PATH` (by default `CARD.check.png` next to the card; `none` for no sheet) holds one row per test body: the body at rest from the front, the side and the back, then its worst state from the same three views. The skin is drawn in skin colour and each piece in its mean colours, with red dots on skin through and orange dots on layers through; each picture is labelled with the body, the state, the view and the counts, and the title gives the gates. A sheet of a card with a private piece never goes into a checkout of this repository.
- The errors are `card_not_found`, `invalid_card`, `package_not_found`, `invalid_package`, `unknown_outfit`, `no_pieces`, `bad_option` (an unknown body, clip or slider), `private_in_repository` and `usage`.

## Outfit files

An outfit file `NAME.outfit.json` (format `opensculptboy.outfit/1`) holds an outfit apart from any card, so that it can be shared with its packages. Its JSON Schema is `src/opensculptboy/data/wearables/outfit-1.schema.json` (`opensculptboy.wearables.outfitfile.SCHEMA_PATH`).

```json
{
  "format": "opensculptboy.outfit/1",
  "id": "travel",
  "label": "Travel clothes",
  "description": "a linen shirt under a hooded wool cape",
  "tags": ["travel"],
  "provenance": "public",
  "pieces": [
    {"piece": "shirt", "file": "../wear/shirt.wearable.json", "enabled": true, "overrides": {"layer": 52}},
    {"piece": "cape", "file": "../wear/cape.wearable.json", "enabled": true}
  ],
  "hides": [{"piece": "cape", "target": "shirt", "groups": ["hood"]}],
  "springs": {"cape": [{"name": "loose", "kind": "auto", "preset": "cape"}]},
  "credits": ["\"Linen shirt\" by Ada, CC0-1.0"]
}
```

| Field | Meaning |
|---|---|
| `id`, `label`, `description`, `tags` | the outfit's id (no `/`, not `none`), its name to show, one line about it and free words |
| `provenance` | `private` when a piece is private; such a file never goes into a checkout of this repository |
| `pieces` | each piece's id, its package file relative to the outfit file, `enabled`, and `overrides` of the package's `layer` and `label` |
| `hides` | rules added to the packages' own: while `piece` shows, the named `groups` of `target` are hidden; `target` is a piece of the outfit by id, else a piece of the card as `OUTFIT/PIECE` or a loose id |
| `springs` | the spring chains of a piece by its id, in the form of a package's `springs` field, in place of the package's own |
| `credits` | the credit line of each piece, as the exports write them |

```bash
opensculptboy wearables outfit export ada.json --outfit travel --out outfits/travel.outfit.json
opensculptboy wearables add bo.json outfits/travel.outfit.json --activate
opensculptboy wearables add bo.json outfits/travel.outfit.json --outfit travel2 --dry-run
```

- `wearables outfit export CARD --outfit ID --out FILE` writes one outfit of a card: the paths relative to the file, `enabled` as the card's overrides leave it, the pieces' settings (below) and the credits; `--label`, `--description` and `--tag` set those fields. An existing file with other contents stays unless `--replace`; the same contents again report `changed: false`; `--dry-run` prints the file in `outfit` and writes nothing.
- `wearables add CARD FILE.outfit.json` puts the file on a card as the outfit of its id (`--outfit NAME` gives another id, `--label` another label), with the paths relative to the card and the file as the outfit's `source`. Every package is read and checked first, and a rule must name groups that its target has. The same outfit again changes nothing; an outfit of that id with other contents is an error unless `--replace`, which also takes out the card's overrides of the old pieces. `--activate` makes it the worn outfit, and `--dry-run` prints the change.
- On the card, each piece of an outfit keeps the file's settings: `overrides` (`layer`, `label`), `hides` (the file's rules of that piece, in the form of a package's `hides`) and `springs`. The export applies them to the package when it dresses the card, so a rule of the card works as a rule of the package.
- The errors are `outfit_not_found`, `invalid_outfit` (the message names the field), `unknown_piece` (a rule or a spring names a piece the outfit lacks), `unknown_group`, `package_not_found`, `invalid_package`, `outfit_exists`, `file_exists`, `unknown_outfit`, `no_pieces`, `bad_option`, `private_in_repository` and `usage`.

## Agents

The agent skill [`skills/outfits/SKILL.md`](../skills/outfits/SKILL.md) gives the workflow from a request in words to a dressed export (catalogue, import or bind, add, check, read the sheet, tune, export), the JSON fields and error codes to act on, the gates, the credits and the provenance rule.

## Reference wearables

An appearance handoff ([Fit to a reference](reference-fit.md)) may describe what the reference character wears in an `equipment` section. `opensculptboy wearables reference` builds one package per piece for a card fitted to that handoff:

```bash
opensculptboy wearables reference handoff/ --card fitted.json --out dressed/
opensculptboy export dressed.glb --character dressed/fitted.json
```

The section lists one set of `objects` (the clothing objects of one look, gathered into pieces by the first word of their names), or several `outfits`, each with `pieces` of one or more objects. An object names a node of `replica.glb` (skinned in the same bind pose as the skin) and its primitives, one per mesh group with the group in the primitive's `extras`. A piece may carry `hides` rules toward the groups of other pieces and `covered` masks: triangles of the reference's skin, for the piece's fixed groups and for each hideable group.

1. Each clothing vertex finds its closest point on the reference's skin (every skin part of the handoff, whether or not a `covered` entry names it), and that point its closest point on the fitted card's skin, posed and placed as the fit left it (the fit's correspondence).
2. The card's skin weights at those points, diffused over the piece (40 steps, each halfway to the mean of the neighbours, with vertices at one position joined), become the piece's own weights, the four strongest. Neighbouring vertices then blend the same bones, so that a cloak whose edge lies near a hand moves as one sheet. The vertex goes back to the card's rest pose by the inverse of the skinning with these weights, and its normal by the inverse of the same matrix, and binds to the same point of the card's rest skin.
3. A card vertex is covered where its closest triangle of the reference's skin is covered. The packages set `cover_ray`, so an export keeps a vertex of the mask covered only where its normal ray also meets the dressed piece within 3 cm: the skin between the laces of open pieces such as laced boots stays.

The packages go into `<out>/wearables/` with `provenance` private, next to a copy of the card that lists the outfits and names the default outfit as `active_outfit`. `<out>/wearables/report.json` gives per piece the vertex count, the distances of the correspondence and the round trip (the piece bound, evaluated at rest, posed and placed back onto the reference).

## Provenance

A package records where it comes from. Public packages may live in this repository. Private packages are made from a reference character's own files, and they, the cards that list them and every export of those cards stay with their owner: `write_wearable` and the reference builder refuse to write a private package anywhere inside a checkout of this repository, `wearables add` refuses to put one on a card inside a checkout, and `test.test_wearables.TestNoPrivatePackages` fails when a file under `src/` or `test/data/` declares the private provenance.

Every piece whose licence records an attribution, an author, an SPDX id or a source gets a credit line in the files that hold it, whatever the licence: the `attribution` as given, else the label with the author, the SPDX id, the source and the design source. The GLB export lists the lines in `extras.opensculptboy.wearables.credits`, and the VRM export adds them after the base model's credits (`Wearable: <line>`). The licences are the user's responsibility; the tools record what they find and write the credits.

## Checks

`test.test_wearables_import` writes synthetic MakeHuman assets in the test (`test/mhclo_fixtures.py`: a tube around the torso bound to base-mesh triangles with scale references and `delete_verts`, and a patch of `helper-tights` bound vertex by vertex) and checks the header and binding formats, the binding on `base.obj` and on the default body, the helper geometry against the targets on random bodies, the recorded licences (with words after the name) and the credits in a dressed export, an asset in a pack folder with `packs/<pack>.json` and `--pack-json`, bump maps (ramps against the OpenGL convention, a bump-only material with both key names, the strength and `bumpmapIntensity`, a material with a normal map and a bump map, and every texture key mapped or reported), one asset per quality defect (a flipped triangle, missing UVs on a textured material, a DirectX normal map, a missing texture, an alpha map on an opaque material, an oversized and an odd-sized texture), the JSON results and errors, the JSON alone on stdout of commands run in new processes, the card commands with dry runs, repeated runs and the provenance guard, and the packages against the JSON Schema. `test.test_wearables` uses synthetic garments only (`test/wearable_fixtures.py`: a tunic, a rigid circlet and a cape that hides the tunic's back): the package round trip and the provenance guard, garments that return to their own vertices on the template, ten random bodies on which no covered skin lies more than 2 mm outside the tunic, the push-out between a rigid plate and a coat, the card's outfits with slot replacement, overrides and hides rules, the GLB, wardrobe and VRM files of both versions, the command, and reference wearables built from a synthetic reference character in both forms of the `equipment` section. `test.test_wearables_bind` writes garment files in the test (a tube around the torso in two materials and a ring around the head) and checks `wearables bind`: the package's fields, the round trip on the template, the frame of the default export, a dressed export on two random bodies, the far-from-body warning and refusal, both frames, own weights and the command. `test.test_outfit_tools` checks the catalogue (the entries, the order, the filters, the colours, the quality summary, assets not yet imported, thumbnails and their cache), the check (the report, a clean outfit that passes, a shell under the skin that fails `skin_rest`, a lower piece outside an upper one that fails `layers_rest`, the exit status, the sheet) and the outfit files (export, schema, a card in another folder, repeated and dry runs, replacement, the settings in the dressing, the coded errors and the provenance guard). `TestGltfReference` holds a dressed case.

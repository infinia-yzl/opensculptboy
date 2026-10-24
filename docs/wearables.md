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

`opensculptboy.wearables.fit.dress` evaluates the worn pieces on a body from the lowest layer up. Each vertex takes the body's skin weights at its bound point (the strongest four, renormalised). A **push-out** then moves every vertex of a piece outside the body and outside the shown pieces of lower layers: at least by a 1 mm gap, or by as much as the vertex lay outside them on the binding body, whichever is less. A piece therefore keeps the relations it was made with: a tight piece never sinks into a heavier body, and a strap that sinks into a coat on the binding body may sink into it on every body. A Laplacian fade spreads each push over two rings of neighbouring vertices, and a last pass makes every vertex meet its target.

### Covered skin

A package covers skin through a mask over the vertices of Anny's template: `covers` holds the skin under the groups that always show, and `group_covers` adds the skin under a group while that group shows. With `cover_ray` set, a vertex of the masks counts only where its normal ray also meets the dressed piece within 3 cm. A package without masks covers the skin whose normal ray meets the dressed piece within 3 cm. The ray tests see only triangles of opaque materials, since a masked or blended material (lace, a sheer stocking) may let the skin show. An export removes a body triangle when all three of its corners lie under a shown piece; `--covered keep` keeps every triangle (for engines that swap clothes at run time). A wardrobe file keeps the body whole.

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

A piece is named by its reference: `<outfit>/<piece>` for a piece of an outfit, the bare id for a loose piece. A bare id also names the piece of that id in the worn outfit, so `--hide cape` hides `travel/cape`. The export resolves what shows in this order: the worn outfit's pieces as the card enables them, the enabled loose pieces, the card's overrides, then `--hide` and `--show`.

- **Slot replacement.** A piece shown from another outfit, and a loose piece, replace the pieces of the same slot that come from elsewhere. Pieces of `accessory` never replace.
- **Hides rules.** While piece A shows, the groups that its `hides` rules name are hidden on piece B. Hiding A shows them again. A rule names a piece of its own outfit by its id and a piece of another outfit by `<outfit>/<piece>`.

## Files

- **GLB.** Each shown piece is a mesh node of its own, named by its reference and skinned by the character's skin, with glTF PBR materials and MikkTSpace tangents for its normal maps. The node's `extras.opensculptboy_wearable` and the scene's `extras.opensculptboy.wearables` record the pieces, their hidden groups and the removed body triangles. The ground lift takes footwear into account.
- **Wardrobe GLB** (`--wardrobe`). Every piece of every outfit is a node, and every group that a rule can hide is a node of its own (`<piece>.<group>`), all at the root, with the body whole. The default scene holds the body and what the worn outfit shows, so Blender and other importers open the character as it is dressed; a second scene holds every piece, for tools that switch scenes or let the user show nodes one by one. `--node-visibility` writes one scene with every piece instead and marks the pieces and groups that the worn outfit does not show with `KHR_node_visibility` (`visible: false`). The Khronos validator reports that extension as unknown (an info). The glTF importer of Blender 5.2 reads every node of every scene and ignores the extension; `blender --python src/opensculptboy/export/blender_skin.py -- wardrobe.glb` imports a wardrobe file and hides the pieces and groups that its extras mark as not shown (`opensculptboy_wearable.shown`, `opensculptboy_wearable_group.shown`).
- **VRM.** Each shown piece is a mesh bound into the T-pose with the file's skin weights at its bound points (the inverse bind, so that the avatar posed at Anny's rest pose shows the piece's rest vertices exactly) and an MToon material per garment material. Pieces count against the budgets of [VRM avatars](vrm.md). Spring bones and constraints come later.

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
- An unknown slot, an unknown bone for `--rigid`, own weights without a skin or with joints that the rig lacks, and a malformed `--hides` item are errors.
- A textured material on a mesh without UVs is a quality error (`missing_uvs`); with `--accept-quality` the package keeps the material without its textures.

## Free outfits

MakeHuman's community shares clothes as asset packs: <https://static.makehumancommunity.org/assets/assetpacks/index.html> lists them (shirts, pants, skirts, suits, dresses, shoes, gloves, underwear), each pack under CC0 or CC-BY. A clothes asset is a folder with a `.mhclo` file, its `.obj` mesh, its `.mhmat` material and the textures. CC0 assets need no credit. A CC-BY asset needs a credit to its author ([MakeHuman's FAQ](https://static.makehumancommunity.org/oldsite/faq/what_do_i_need_to_do_when_i_use_a_ccby_asset.html) reads it as CC BY 4.0): the import writes an `attribution` line for it, the GLB export lists those lines in `extras.opensculptboy.wearables.credits` and the VRM export adds them to the file's credits. MakeHuman's own source and some assets are under AGPL; the tools record whatever licence the header names, and the choice is the user's.

Two commands put a free outfit on a character:

```bash
opensculptboy wearables import-mhclo packs/shirts01/toigo_fisherman_sweater/toigo_fisherman_sweater.mhclo --out wear/sweater
opensculptboy wearables import-mhclo packs/pants01/some_jeans/some_jeans.mhclo --out wear/jeans --tag denim
opensculptboy wearables add ada.json wear/sweater.wearable.json wear/jeans.wearable.json --outfit casual --label "Casual" --activate
opensculptboy export ada.glb --character ada.json
```

### Import a MakeHuman asset

`opensculptboy wearables import-mhclo FILE.mhclo [--out STEM]` reads the asset and writes a package bound to the default body (`binding.body` is `anny-template`).

- **Header.** `name`, `uuid`, `description`, `tag` lines, `z_depth`, `obj_file`, `material`, `basemesh` (only `hm08`, MakeHuman's default base mesh) and the licence keys `author`, `license`, `homepage` and `copyright`, as keys or in comment lines (`# license CC0`); MPFB2 writes the description in a comment line too (`# description: ...`), which a `description` key replaces. The package takes the name, the label, the description and the tags from them (`--name`, `--label`, `--description` and `--tag` add or replace them), the layer from `z_depth` (50 without one, as MakeHuman uses), and the slot from the words of the name and the tags (`--slot` sets it).
- **Binding.** After `verts`, one line per `.obj` vertex: three base-mesh vertex indices, three weights and an offset, or one index alone. On a body `h` in MakeHuman's frame (decimetres, Y up) a vertex lies at `w1 h[v1] + w2 h[v2] + w3 h[v3] + S d`. `S` is diagonal: for each axis n of `x_scale`, `y_scale` and `z_scale` (`v1 v2 length`), `|h[v1][n] - h[v2][n]| / length`, so that the offset stretches with the body; without scale lines it is the identity, and `shear_x`, `shear_y` and `shear_z` (`v1 v2 x1 x2`) give `(h[v2][n] - h[v1][n]) / (x2 - x1)`. The import evaluates the binding on the default body, then binds the garment to the nearest triangles of the skin with the code of `wearables bind`. The report's `source_fit_mm_max` compares the binding evaluated on `base.obj` with the `.obj` vertices: a large value means a file that does not follow the format.
- **Base mesh.** Anny's template is MakeHuman's `base.obj` (19,158 vertices) in Anny's frame. Anny's default topology keeps the 13,718 vertices that its faces use (body, eyes and tongue); it leaves out the helper geometry (`helper-tights`, `helper-skirt`, `helper-hair` and the others) and the vertices of its nudity edits. Many assets bind to the helpers. The MPFB2 targets that Anny bundles move the helper vertices as they move the skin around them, so the import takes the vertices that Anny lacks from those targets with the card's blend-shape coefficients. On other bodies the package follows the skin triangles under it: on three random bodies a patch bound to `helper-tights` stays within 6 mm (median) and 12 mm (largest) of the helper that the targets give (`test_helper_geometry_follows_the_targets`).
- **Covered skin.** `delete_verts` (numbers and ranges `a - b`) becomes the package's `covers` mask on Anny's rows; deleted vertices outside Anny's rows are counted in the report. Without `delete_verts` the import uses the ray test of `wearables bind`. `--covers ray` always uses the ray test and `--covers none` covers no skin.
- **Material.** The `.mhmat` file becomes one glTF PBR material and one group named by it: `diffuseColor` (sRGB, made linear) times `diffuseIntensity` and `diffuseTexture` give the base colour and `opacity` its alpha; `roughnessFactor` is `clamp(1 - shininess, 0.04, 1)`, 1 when `specularIntensity` or `specularColor` is 0, and a `roughness` key wins; `metallic` gives the metallic factor; `transparencymapTexture` becomes the alpha of the base colour texture; `transparent` gives BLEND, and `alphaToCoverage`, or an alpha map on an opaque material, gives MASK with cutoff 0.5; `backfaceCull False` gives a double-sided material; `normalmapTexture` with `normalmapIntensity`, `aomapTexture` and `emissiveColor` give the normal map, the occlusion and the emission. Specular, bump and displacement maps have no glTF PBR slot: the report lists each one it drops. Textures in other formats than PNG and JPEG are converted to PNG.

`--dry-run` checks and binds the asset and prints the result without writing the package (`"written": false`). The command prints one JSON result on stdout (`"ok": true`, the package, `dry_run` and `written`, the report of `wearables bind`, `licence`, `slot_from`, `mhclo` and `quality`); warnings go to the error stream and stay in the result. A failure prints `{"ok": false, "error": {"code", "message", "hint"}}` and exits with status 1; the codes include `file_not_found`, `malformed_header`, `malformed_binding`, `malformed_obj`, `binding_mismatch`, `unknown_basemesh`, `unknown_slot`, `too_far_from_skin`, `missing_uvs`, `private_in_repository` and `usage` (status 2).

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
| UVs | `missing_uvs` (error), `uv_outside_unit_square`, `uv_degenerate`, `uv_flipped` (triangles mirrored against the rest of their UV island), `uv_overlap` (a UV area covered twice in the same direction), `uv_mirrored_overlap` (an info: symmetric halves that share a texture), `texel_density` or `texel_density_spread` (the 95th over the 5th percentile above 4) |
| Materials | `roughness_mapped` (with the formula), `alpha_mode`, `double_sided`, `alpha_map_merged`, `alpha_map_on_opaque`, `map_dropped`, `material_missing` |
| Textures | `texture_missing`, `texture_unreadable`, `texture_converted`, `texture_not_power_of_two`, `texture_too_large` (above `--texture-limit`, 4096 by default), `linear_texture_tagged_srgb` (a normal, roughness or occlusion map tagged as sRGB; glTF reads it as linear), `base_colour_tagged_linear`, `alpha_unused` (a base colour texture with alpha on an opaque material), `alpha_missing` (a transparent material without alpha), `normal_map_not_tangent_space`, `normal_map_convention`, `normal_map_directx` or `normal_green_flipped`, `textures_embedded`, `tangents_mikktspace`, `tangents_degenerate` |

The green channel of a normal map follows the OpenGL convention in MakeHuman and in glTF (green points up the image). The check measures the correlation between the change of red up the image and the change of green to the right, which a relief gives alike: positive for OpenGL, negative for DirectX. `--normal-green auto` (the default) flips a map that reads DirectX with a correlation below -0.25, `keep` leaves it with a warning, and `flip` always flips.

## Reference wearables

An appearance handoff ([Fit to a reference](reference-fit.md)) may describe what the reference character wears in an `equipment` section. `opensculptboy wearables reference` builds one package per piece for a card fitted to that handoff:

```bash
opensculptboy wearables reference handoff/ --card fitted.json --out dressed/
opensculptboy export dressed.glb --character dressed/fitted.json
```

The section lists one set of `objects` (the clothing objects of one look, gathered into pieces by the first word of their names), or several `outfits`, each with `pieces` of one or more objects. An object names a node of `replica.glb` (skinned in the same bind pose as the skin) and its primitives, one per mesh group with the group in the primitive's `extras`. A piece may carry `hides` rules toward the groups of other pieces and `covered` masks: triangles of the reference's skin, for the piece's fixed groups and for each hideable group.

1. Each clothing vertex finds its closest point on the reference's skin, and that point its closest point on the fitted card's skin, posed and placed as the fit left it (the fit's correspondence).
2. The card's skin weights at those points, diffused over the piece (40 steps, each halfway to the mean of the neighbours, with vertices at one position joined), become the piece's own weights, the four strongest. Neighbouring vertices then blend the same bones, so that a cloak whose edge lies near a hand moves as one sheet. The vertex goes back to the card's rest pose by the inverse of the skinning with these weights, and its normal by the inverse of the same matrix, and binds to the same point of the card's rest skin.
3. A card vertex is covered where its closest triangle of the reference's skin is covered. The packages set `cover_ray`, so an export keeps a vertex of the mask covered only where its normal ray also meets the dressed piece within 3 cm: the skin between the laces of open pieces such as laced boots stays.

The packages go into `<out>/wearables/` with `provenance` private, next to a copy of the card that lists the outfits and names the default outfit as `active_outfit`. `<out>/wearables/report.json` gives per piece the vertex count, the distances of the correspondence and the round trip (the piece bound, evaluated at rest, posed and placed back onto the reference).

## Provenance

A package records where it comes from. Public packages may live in this repository. Private packages are made from a reference character's own files, and they, the cards that list them and every export of those cards stay with their owner: `write_wearable` and the reference builder refuse to write a private package anywhere inside a checkout of this repository, `wearables add` refuses to put one on a card inside a checkout, and `test.test_wearables.TestNoPrivatePackages` fails when a file under `src/` or `test/data/` declares the private provenance.

## Checks

`test.test_wearables_import` writes synthetic MakeHuman assets in the test (`test/mhclo_fixtures.py`: a tube around the torso bound to base-mesh triangles with scale references and `delete_verts`, and a patch of `helper-tights` bound vertex by vertex) and checks the header and binding formats, the binding on `base.obj` and on the default body, the helper geometry against the targets on random bodies, the recorded licences and the credits in a dressed export, one asset per quality defect (a flipped triangle, missing UVs on a textured material, a DirectX normal map, a missing texture, an alpha map on an opaque material, an oversized and an odd-sized texture), the JSON results and errors, the card commands with dry runs, repeated runs and the provenance guard, and the packages against the JSON Schema. `test.test_wearables` uses synthetic garments only (`test/wearable_fixtures.py`: a tunic, a rigid circlet and a cape that hides the tunic's back): the package round trip and the provenance guard, garments that return to their own vertices on the template, ten random bodies on which no covered skin lies more than 2 mm outside the tunic, the push-out between a rigid plate and a coat, the card's outfits with slot replacement, overrides and hides rules, the GLB, wardrobe and VRM files of both versions, the command, and reference wearables built from a synthetic reference character in both forms of the `equipment` section. `test.test_wearables_bind` writes garment files in the test (a tube around the torso in two materials and a ring around the head) and checks `wearables bind`: the package's fields, the round trip on the template, the frame of the default export, a dressed export on two random bodies, the far-from-body warning and refusal, both frames, own weights and the command. `TestGltfReference` holds a dressed case.

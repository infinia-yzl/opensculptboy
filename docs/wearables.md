# Wearables

Wearables are clothes, footwear, headgear, capes and accessories that fit any Anny body. A character card lists them as loose pieces or as outfits, and the GLB and VRM exports dress the character in them. Each piece is a **wearable package** (`opensculptboy.wearables`), bound once to Anny's template and evaluated on every body.

```bash
opensculptboy export ada.glb --character ada.json                          # the card's active outfit
opensculptboy export ada.glb --character ada.json --outfit travel --hide cape
opensculptboy export ada.glb --character ada.json --show work/apron       # a piece of another outfit takes its slot
opensculptboy export ada.glb --character ada.json --covered keep          # keep the skin under the clothes
opensculptboy export ada_wardrobe.glb --character ada.json --wardrobe     # every piece as its own node
opensculptboy export ada.vrm --character ada.json --author "Your name"    # the outfit in a VRM avatar
opensculptboy export ada.vrm --character ada.json --author "Your name" --springs off   # no spring chains
opensculptboy wearables list --character ada.json                         # the outfits, pieces and groups
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
| `slot` | `head`, `face`, `neck`, `torso`, `arms`, `hands`, `legs`, `feet`, `cloak`, `under` or `accessory` |
| `layer` | 0 (skin) to 100 (outermost), as in MakeHuman's z-depth: underwear 10, legs 30, torso 50, headgear 60, cloaks 80 |
| `provenance` | `public` or `private` (see "Provenance") |
| `licence` | `{spdx, author, source, design_source, attribution}`; `spdx` and `design_source` are required for a public package |
| `binding` | `{body, method, data, bone?}`: `body` is `anny-template` or `reference`, `method` is `surface` or `rigid` |
| `weights` | `body` (the body's skin weights at the bound points), `rigid` (the one bone) or `own` (weights in the binding file, by bone name in `binding.joints`) |
| `covers`, `group_covers`, `cover_ray` | whether the binding file holds a cover mask, the groups that hold their own masks, and whether a ray test confirms the masks |
| `groups` | the named groups of the garment |
| `hides` | rules toward other pieces: `{"piece": id, "groups": [name, ...]}` |
| `springs` | the spring chains of the loose parts (see "Springs") |
| `constraints` | constraints for the VRM export; packages keep them, and a later version writes them |

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

A piece is named by its reference: `<outfit>/<piece>` for a piece of an outfit, the bare id for a loose piece. A bare id also names the piece of that id in the worn outfit, so `--hide cape` hides `travel/cape`. The export resolves what shows in this order: the worn outfit's pieces as the card enables them, the enabled loose pieces, the card's overrides, then `--hide` and `--show`.

- **Slot replacement.** A piece shown from another outfit, and a loose piece, replace the pieces of the same slot that come from elsewhere. Pieces of `accessory` never replace.
- **Hides rules.** While piece A shows, the groups that its `hides` rules name are hidden on piece B. Hiding A shows them again. A rule names a piece of its own outfit by its id and a piece of another outfit by `<outfit>/<piece>`.

## Files

- **GLB.** Each shown piece is a mesh node of its own, named by its reference and skinned by the character's skin, with glTF PBR materials and MikkTSpace tangents for its normal maps. The node's `extras.opensculptboy_wearable` and the scene's `extras.opensculptboy.wearables` record the pieces (each by its reference `ref`, with its `piece` id and its `outfit`), their hidden groups and the removed body triangles; the VRM file records them alike. The summary of a GLB or VRM export lists the shown pieces by reference. The ground lift takes footwear into account. The joints of the spring chains follow the bones in the skin, and `extras.opensculptboy.wearables.springs` records each chain's nodes and settings and each collider in its node's frame.
- **Wardrobe GLB** (`--wardrobe`). Every piece of every outfit is a node, and every group that a rule can hide is a node of its own (`<piece>.<group>`), all at the root, with the body whole. The default scene holds the body and what the worn outfit shows, so Blender and other importers open the character as it is dressed; a second scene holds every piece, for tools that switch scenes or let the user show nodes one by one. `--node-visibility` writes one scene with every piece instead and marks the pieces and groups that the worn outfit does not show with `KHR_node_visibility` (`visible: false`). The Khronos validator reports that extension as unknown (an info). The glTF importer of Blender 5.2 reads every node of every scene and ignores the extension; `blender --python src/opensculptboy/export/blender_skin.py -- wardrobe.glb` imports a wardrobe file and hides the pieces and groups that its extras mark as not shown (`opensculptboy_wearable.shown`, `opensculptboy_wearable_group.shown`).
- **VRM.** Each shown piece is a mesh bound into the T-pose with the file's skin weights at its bound points (the inverse bind, so that the avatar posed at Anny's rest pose shows the piece's rest vertices exactly) and an MToon material per garment material. Pieces count against the budgets of [VRM avatars](vrm.md). The spring chains become spring bones ([VRM avatars](vrm.md#spring-bones)); constraints come later.

## Springs

Loose parts of a piece (a cape, a skirt, a long hem) swing on chains of extra joints. A package lists them in `springs`, and `opensculptboy.wearables.springs` builds them on the body of each export:

```json
"springs": [
  {"name": "back"},
  {"name": "hem", "preset": "skirt", "columns": 8, "joints": 4, "settings": {"stiffness": 1.6}},
  {"name": "tail", "points": [[0, 0.4, -0.1], [0, 0.0, -0.13], [0, -0.4, -0.15]], "parent": "spine03"}
]
```

| Key | Meaning |
|---|---|
| `name` | the spring's name, distinct within the package |
| `kind` | `auto` (the default) or `chain` (the default when `points` is given) |
| `preset` | `cape`, `skirt` or `hem`; by default `cape` for the `cloak` slot, `skirt` for `legs` and `under`, `hem` for the others |
| `settings` | `stiffness`, `gravity_power`, `gravity_dir` (glTF axes), `drag_force` and `hit_radius` (metres), in place of the preset's |
| `groups` | the garment groups of the spring (every group by default) |
| `columns`, `joints` | the chains of an `auto` spring and the joints of each, the tail included (cape 5, skirt 8, hem 3; 4 joints) |
| `points` | a `chain` spring's joints, in metres and glTF axes, on the binding body at rest |
| `parent` | the bone that the chains hang from; `root` for the skirt preset, else the strongest torso bone at each chain's first joint |
| `colliders` | the collider groups that the chains hit: `head`, `torso`, `legs`, `arms` (all by default) |
| `gap`, `hang`, `reach` | the loose test of an `auto` spring (2.5 cm from the skin, 3 cm below the closest skin point) and an authored chain's reach over the garment (15 cm) |

| Preset | Stiffness | Gravity | Drag | Hit radius |
|---|---|---|---|---|
| `cape` | 0.8 | 0.3 | 0.45 | 4 cm |
| `skirt` | 1.4 | 0.1 | 0.5 | 3 cm |
| `hem` | 1.0 | 0.2 | 0.5 | 2 cm |

- **Auto chains.** A vertex is loose when it lies more than `gap` from the skin or more than `hang` below its closest skin point. The columns share the loose vertices by their azimuth around the body's vertical axis: all around for a skirt, across the back for a cape. Each column runs from the top of its loose vertices (the median of three neighbouring columns' tops) to the lowest vertex of its band, its last joint the tail.
- **Authored chains.** On another body each point moves with the six garment vertices nearest to it on the binding body.
- **Hierarchy.** Each chain hangs from one bone, a node belongs to one chain only, and a branch is a chain of its own. The tail joints carry no weight.
- **Weights.** A vertex takes the joints of its two nearest chains by its place along each, and its body weights fade into the chain weights over the first segment, so the top of the region stays on the body. An authored chain's share fades out over its reach.
- **Colliders.** A sphere inside the head and capsules on the neck, the chest, the shoulders, the spine, the hips, the upper and lower legs and the upper arms, fitted inside the skin of each character. The neck, the chest and the shoulders take the ends, quantiles and spans of the viewer's hair colliders. Each joint's hit radius stays within its clearance from the colliders at rest, so a chain rests where the garment was made.
- **Off.** `--springs off` (or `Dressing(springs=False)`) leaves every piece on the body's bones.

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

A cloak, and a piece whose objects carry physics in the handoff, gets one `auto` spring named `loose` (see "Springs"); a piece without a loose region keeps to the body's bones, with a `SpringWarning`. The packages go into `<out>/wearables/` with `provenance` private, next to a copy of the card that lists the outfits and names the default outfit as `active_outfit`. `<out>/wearables/report.json` gives per piece the vertex count, the distances of the correspondence and the round trip (the piece bound, evaluated at rest, posed and placed back onto the reference).

## Provenance

A package records where it comes from. Public packages carry a licence and may live in this repository. Private packages are made from a reference character's own files, and they, the cards that list them and every export of those cards stay with their owner: `write_wearable` and the reference builder refuse to write a private package anywhere inside a checkout of this repository, and `test.test_wearables.TestNoPrivatePackages` fails when a file under `src/` or `test/data/` declares the private provenance.

Every piece whose licence records an attribution, an author, an SPDX id or a source gets a credit line in the files that hold it, whatever the licence: the `attribution` as given, else the label with the author, the SPDX id, the source and the design source. The GLB export lists the lines in `extras.opensculptboy.wearables.credits`, and the VRM export adds them after the base model's credits (`Wearable: <line>`). The licences are the user's responsibility; the tools record what they find and write the credits.

## Checks

`test.test_wearables` uses synthetic garments only (`test/wearable_fixtures.py`: a tunic, a rigid circlet and a cape that hides the tunic's back): the package round trip and the provenance guard, garments that return to their own vertices on the template, ten random bodies on which no covered skin lies more than 2 mm outside the tunic, the push-out between a rigid plate and a coat, the card's outfits with slot replacement, overrides and hides rules, the GLB, wardrobe and VRM files of both versions, the command, and reference wearables built from a synthetic reference character in both forms of the `equipment` section. `TestGltfReference` holds a dressed case. `test.test_wearable_springs` checks the springs of a synthetic long cape and skirt: the entries, the chain structure, the weights and the attachment at the top, the colliders, the GLB and VRM files of both versions within their budgets and schemas, and four seconds of the VRM spring update (`SpringSim`) in which the pieces swing and stay outside the body's capsules.

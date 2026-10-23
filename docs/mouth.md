# The mouth

`opensculptboy.mouth` gives exported characters a jaw and, for a fitted character, the inside of its reference's mouth, its sealed lips and its own mouth poses. Anny's default rig (`anny`) opens the mouth with the facial action `jawOpen`, a fixed offset per vertex, and has no jaw bone. The `anny-mouth` rig keeps MakeHuman's jaw bone and tongue chain, so that a file can open the mouth by turning the jaw.

## The `anny-mouth` rig

Anny builds its default rig from MakeHuman's rig with the modifiers `notongue`, `nobreasts`, `nofacialexpression` and `pruned`, and moves the skin weights of every removed bone to its nearest kept ancestor. The jaw and the tongue hang from the head, so their weights belong to `head` in `anny`.

The `anny-mouth` rig keeps 12 more bones: `jaw` (child of `head`) and the tongue chain `tongue00` to `tongue07.L/R` (children of `jaw`), with MakeHuman's own weights. Every other bone keeps its rest frame bit for bit, and the mesh is Anny's own. Only the head gives weight to the new bones.

A card takes the rig with `"rig": "anny-mouth"`:

```bash
uv run opensculptboy character > card.json        # then set "rig": "anny-mouth" in card.json
uv run opensculptboy export out.glb --character card.json
uv run opensculptboy export out.vrm --character card.json --author me
```

Cards with `"rig": "anny"` build and export exactly as before.

`opensculptboy.mouth.rig` builds the rig with Anny's public builder and leaves `src/anny` unchanged. Anny's bones take Anny's cached rest orientations (`data/cached/anny.pth`), and the mouth bones take the rows of `data/mouth/rig_cache.safetensors`, computed as Anny computes its own cache. The reader checks that the file holds the same blend-shape rows as Anny's cache, so an update of Anny's cache (`--append`) needs the file built again:

```bash
uv run python -m opensculptboy.mouth.authoring.rig_cache --check neck01   # rebuild the file; --check compares a bone of anny's rig with anny's cache
```

## Jaw open as a turn of the jaw

`opensculptboy.mouth.jaw.fit_jaw_open` splits Anny's `jawOpen` into a turn of the jaw about its head and a corrective shape, a rest-frame offset like any morph target:

- The turn is the least-squares fit of the target by linear blend skinning with the weights of the jaw and the tongue.
- The corrective completes it exactly: the skinned turn of the corrected rest mesh is the rest mesh plus `jawOpen`.

On the default body the turn is about 22 degrees and carries about 89 % of the target. The files keep a pure turn, which a VRM humanoid jaw can play.

## What the files hold

- **GLB:** the 116 joints of the rig, the facial action `jawOpen` as before, one more morph target `jawOpen.corrective`, and the animation `jawOpen.jaw`, which turns the jaw and raises the corrective from 0 to 1 over one second. At its last keyframe the file shows Anny's `jawOpen`, because the corrective is computed for the file's own skin weights. The scene extras record the turn under `opensculptboy.mouth` (`opensculptboy/mouth@1`).
- **VRM 1.0 and 0.x:** the humanoid `jaw` maps to the jaw bone, and the tongue bones stay outside the humanoid map. VRM expressions bind morph targets only, so the expressions keep Anny's `jawOpen` target and the files carry no corrective; an application can turn the humanoid jaw on its own. The skin weights are those of the `anny` rig for the same body, with each vertex's `head` weight shared among `head` and the mouth bones in the proportions of MakeHuman's weights (`split_head_weights`).

`test.test_mouth_rig` checks the rig, the exact split of `jawOpen`, the GLB animation against the target, and the VRM jaw and weights.

## A reference's inner mouth

Anny's mouth is a cavity of skin with a tongue inside and no teeth. A fitted character takes the inside of its reference's mouth when the appearance handoff describes it (`opensculptboy.mouth.handoff`, `opensculptboy.mouth.inner`).

### In the handoff

`appearance.json` may hold a `mouth` section:

- `pieces`: one entry per piece of the inner mouth, with its `role` (`gums_upper`, `gums_lower`, `palate`, `tongue`, `mouth_floor`, `throat`, or `tooth_<arch>_<side><n>` for a separate tooth) and its primitive of `replica.glb` (`mesh` and `primitive`); a piece that is part of a primitive lists its `vertices`.
- `materials`: the mouth's material, whose `converted` maps (`base_color`, `normal`, `roughness`: paths in the handoff folder) draw it.
- `poses`: the reference's own mouth poses (see [Mouth expressions](#mouth-expressions-from-the-reference)).

`load_handoff` reads each piece straight from the GLB file with the skin weights of its own rig, as a part of role `mouth` in place of the mouth's primitives. The fit leaves the pieces out, as it leaves out every part but the skin.

### In the card

`opensculptboy fit --textures` and `opensculptboy bake` move the pieces into the card's rest frame by the rigid motion that takes the fitted head back to its rest pose, and write them next to the card:

- `<card>.mouth.safetensors` (`opensculptboy/inner-mouth@1`): the points, normals, UVs and triangles, the arch of each point (the head's or the jaw's), the part of each triangle (`teeth`, `gums`, `palate`, `tongue`, `floor`, `throat`) and the skin of each point on the `anny-mouth` rig;
- `<card>.mouth_<slot>.png`: the mouth's maps;
- `material.mouth`: `{"mesh": ..., "base_color": ..., "normal": ..., "roughness": ...}`, with the optional `color`, `roughness_factor` and `clearcoat` (the wet surface, 0.6).

The card also takes the `anny-mouth` rig when its rig is `anny`. The skin of the reference's rig maps onto it by name: the jaw joint and the joints of the chin and the lower lip become `jaw`, the tongue chain becomes Anny's chain `tongue00` to `tongue04` by its place from the root to the tip, and every other joint becomes `head`. Each point keeps its 4 strongest bones.

### In the files

The GLB and VRM exports append the mouth to the body as they append teeth ([Teeth](teeth.md)): it shares the body's vertices, morph targets and skin, and draws with one material, `mouth` (in VRM files an MToon material). A card with `material.mouth` takes no other teeth.

- **Hidden:** Anny's tongue and the part of Anny's cavity that lies behind the front of the inner mouth stay in the mesh, so that the morph targets and the indices of the other parts keep their rows, but no primitive draws them. On the `anny-mouth` rig a row of the cavity hides only behind the pieces of its own arch, so that a turn of the jaw moves a hidden row and the pieces in front of it alike. The lining in front of the inner mouth stays, as it closes the space between the lips and the front of the gums.
- **Skin:** on a rig with the mouth bones, each point takes its own weights, so the lower gums, the lower teeth, the floor of the mouth and the tongue turn with the jaw. A bone the rig lacks gives its weight to its nearest kept ancestor.
- **Morph targets:** the upper pieces stay with the head, and the lower pieces take the rigid motion of Anny's tongue in a target that turns it (`jawOpen`, `jawLeft`). In a target that bends Anny's tongue (`tongueOut`), the reference's tongue follows Anny's 4 nearest tongue vertices.

`test.test_mouth_inner` checks the handoff's pieces, the joint map, the card's section, the hidden triangles, the skin of each piece, the pieces behind the drawn skin at rest and with the jaw open, and the GLB and VRM files of both versions.

## Sealed lips

A reference's closed lips meet along a line. A game-ready head mesh often draws its lips as separate rows whose borders touch, which leaves an opening in the skin; a sculpted head welds them, and they meet at the bottom of a crease. The fit alone misses that line, and Anny's own mouth is modelled a little open, so the fitted lips would part by a slit through which the inside of the mouth shows.

`opensculptboy.fit.lips` closes them as the reference closes its own:

- `lip_line` finds where the reference's lips end, seen from the front: the borders of its mouth opening, split into each lip's edge where they stand apart, or the bottom of the crease between its lips.
- `lip_masks` tells Anny's upper lip from its lower lip by how far Anny's `jawOpen` moves them, `lip_edges` gives the rows where each lip ends, and `lip_corners` the mouth's corners.
- `seal_lips` moves each edge row onto the reference's edge of the same lip and each corner onto the line's nearer end. The rows just past the line's ends (`FitOptions.lip_seal_beyond`) move onto the line carried on, which closes the corners. The mouth's skin around the rows follows smoothly (`lip_seal_rings`), and its skin that already lies on the reference beyond `lip_seal_reach` stays.

The seal closes only lips that the reference closes: when the reference's lips stand further apart than `FitOptions.lip_seal_max_gap` (1 mm), or the rows would move further than `lip_seal_max_shift` (4 mm), the level stays unsealed. `fit_levels` runs the seal on each level before the mouth's repair (`FitOptions.lip_seal`, on by default), then fairs the skin around the visible corners (`FitOptions.lip_corner_fairness`). `lip_slit` and `lip_gap` measure what the front view shows between the lips and the gap of the edge rows, recorded under `FitResult.measures[level]["lips"]`. On levels 1 and 2 the lips meet along the whole line; level 0 is too coarse for its rows to follow it.

## The jaw's weights from the reference

A reference's rig need not give its lips to the jaw: a game-ready face rig often moves the lips and the chin with joints of their own. What carries over is how far each point of the skin follows the jaw when the reference opens its mouth. `mouth.weights.reference_jaw_share` poses the reference with each of its mouth poses that turns the jaw by 2 degrees or more and measures, for each skin vertex, the share of the jaw's turn that it follows, from 0 to 1. `jaw_share` carries the share to level 0 of the fitted card, with the inside of the mouth taking the harmonic extension of the skin's values.

`opensculptboy fit` writes the share as `<card>.jaw_share.safetensors` (`opensculptboy/mouth-weights@1`) under `detail.mouth`. `card_model` gives a model of the `anny-mouth` rig the jaw weights that follow it (`with_jaw_share`). The exports keep MakeHuman's jaw weights for now: a turn of the jaw alone with the share's weights leaves the lower lip behind, since the reference opens its lips with joints of their own.

## Mouth expressions from the reference

A handoff may carry the reference's mouth poses under `mouth.poses` of appearance.json: per pose, a `joints` object that gives joints of the GLB file's skins a `translation` and a `rotation` (a quaternion x, y, z, w), and `known: false` for a pose that is a guess. `opensculptboy.fit.poses.GlbRig` applies a pose as a glTF viewer does: each joint's local transform takes the delta's translation and rotation, and the skins move the vertices by linear blend skinning.

A fitted character with a reference's inner mouth takes these poses as morph targets (`opensculptboy.mouth.poses`):

- `reference_pose_fields` poses the reference and gives the motion of every vertex of its skin and of every piece of its inner mouth. A pose that moves nothing is left out.
- `carry_field` carries the skin's motion to the fitted card: a vertex near the reference takes the motion at its closest point, and the inside of the mouth takes a smooth extension, mixed by the jaw's share so that each lip's lining follows its own lip. The lips' rows and the lining just inside them move by the reference's own blended skin transform at their closest point on the same lip, so that they roll with their lip as the reference's lips do.
- `smooth_lip_motion` smooths rough spots of the lips' carried motion, and keeps a real ramp of the motion in its shape.
- When the card has a level 1, each pose is also carried there, and level 0 takes the offsets whose subdivision follows that motion best, which keeps the rounded mouth's corners together as the reference's.
- Each detail level up to level 2 also carries each pose on its own vertices, so that the lips' rest detail turns with the lips. The difference to level 0's pose is stored per level (`MouthPoses.levels`) and blended out near the mouth's corners; the exports add it to that level's pose targets and jaw correctives.

`opensculptboy fit --textures` and `opensculptboy bake` write the poses as `<card>.mouth_poses.safetensors` (`opensculptboy/mouth-poses@1`) and name the file under `material.mouth.poses`. The exporters add one morph target `mouth.<pose>` per pose:

- **GLB:** on the `anny-mouth` rig each pose also splits into a turn of the jaw and a corrective, as `jawOpen` does: the target `mouth.<pose>.corrective` and the animation `mouth.<pose>.jaw`, whose last keyframe shows the pose. The scene extras list the poses under `opensculptboy.mouth.poses`.
- **VRM:** the files take the targets of the vowels (`viseme_a`, `viseme_i`, `viseme_u`, `viseme_e` and `viseme_o`), and the presets `aa`, `ih`, `ou`, `ee` and `oh` (`A`, `I`, `U`, `E`, `O` in VRM 0.x) bind them at weight 1. The other expressions keep their mixes of Anny's facial actions.

`test.test_mouth_lips` checks the lip line, the seal and the jaw share; `test.test_mouth_poses` checks the carry, the file, the GLB targets, turns and animations, and the visemes of both VRM versions.

## Limits

- The mouth's corners still differ from the reference's. Each closed corner ends in a small tuck, the rounded poses (O and U) open a little wider than the reference toward the corners, and in the open poses a milder crease along the lips remains near the corners. One fix for the whole corner region, where both lips, their lining and the cheek meet, is still to come.
- The expressions with no reference pose keep the ARKit mixes of Anny's facial actions.
- `anny` stays the default rig: `anny-mouth` would change every file's joints and the VRM skin weights near the mouth, and the visemes would need a corrective each.

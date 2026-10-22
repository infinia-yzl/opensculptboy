# The mouth

`opensculptboy.mouth` gives exported characters a jaw and, for a fitted character, the inside of its reference's mouth. Anny's default rig (`anny`) opens the mouth with the facial action `jawOpen`, a fixed offset per vertex, and has no jaw bone. The `anny-mouth` rig keeps MakeHuman's jaw bone and tongue chain, so that a file can open the mouth by turning the jaw.

## The `anny-mouth` rig

Anny builds its default rig from MakeHuman's rig with the modifiers `notongue`, `nobreasts`, `nofacialexpression` and `pruned`, and moves the skin weights of every removed bone to its nearest kept ancestor. The jaw and the tongue hang from the head, so their weights belong to `head` in `anny`.

The `anny-mouth` rig keeps 12 more bones: `jaw` (child of `head`) and the tongue chain `tongue00` to `tongue07.L/R` (children of `jaw`), with MakeHuman's own weights. Every other bone keeps its rest frame bit for bit, and the mesh is anny's own. Only the head gives weight to the new bones: the jaw and the tongue move 1,236 vertices of the mesh, 519 of them by more than half their weight.

A card takes the rig with `"rig": "anny-mouth"`:

```bash
uv run opensculptboy character > card.json        # then set "rig": "anny-mouth" in card.json
uv run opensculptboy export out.glb --character card.json
uv run opensculptboy export out.vrm --character card.json --author me
```

Cards with `"rig": "anny"` build and export exactly as before.

### Rest frames without changes to `src/anny`

Anny gives the bones of its default rig cached rest orientations (`data/cached/anny.pth`, built by `scripts/precompute_rig_caches.py`), which hold no rows for the jaw or the tongue. `opensculptboy.mouth.rig` builds the rig's model data with anny's public builder and then applies the cached orientations itself: the 104 bones of `anny` take anny's own rows, `head` among them, and the 12 mouth bones take the rows of `data/mouth/rig_cache.safetensors`. That file is computed exactly as anny computes its cache (same reference body, squared skin weights, tail aim of weight 0.5):

```bash
uv run python -m opensculptboy.mouth.authoring.rig_cache --check neck01   # rebuild the file; --check compares a bone of anny's rig with anny's cache
```

Computed this way, the rows of anny's own bones match anny's cache to 1e-15, which shows that the computation is anny's. The reader checks that the file holds the same blend-shape rows as anny's cache, so an update of anny's cache (`--append`) needs the file built again.

As for every bone of anny, the `local-ref` pose parameters are relative to the bone's orientation on the reference body. At the identity pose, the jaw therefore sits 0.45 degrees from its rest turn relative to the head on the default body (`neck01` sits 1 degree from its own).

## Jaw open as a turn of the jaw

`opensculptboy.mouth.jaw.fit_jaw_open` splits anny's `jawOpen` into a turn `R` of the jaw about its head and a corrective shape `c`, a rest-frame offset like any morph target:

- The turn is the least-squares fit of the target by linear blend skinning with the weights of the jaw and the tongue.
- The corrective completes it exactly: the skinned turn of the corrected rest mesh `v + c` is `v + jawOpen(v)`.

Measured on the default body, the turn is 21.8 degrees and carries 88.8 % of the target's squared length. The target moves 2,475 vertices by 8.0 mm on average (38.7 mm at most), and the turn leaves 3.1 mm on average (9.0 mm at the 95th percentile, 14.8 mm at most) for the corrective, most of it on vertices with small jaw weights below the chin. Over other bodies the turn is 22 to 24 degrees and carries 87 to 89 %. Letting the jaw's head shift as well (`translate=True`) gives 29.6 degrees, a 11.5 mm shift and 90.4 %; the files keep the pure turn, which a VRM humanoid jaw can play.

## What the files hold

- **GLB:** the 116 joints of the rig, the facial action `jawOpen` as before, one more morph target `jawOpen.corrective`, and the animation `jawOpen.jaw`, which turns the jaw and raises `jawOpen.corrective` from 0 to 1 over one second. At its last keyframe the file shows anny's `jawOpen` to float32 precision, because the corrective is computed for the file's own skin weights (the strongest 4 or 8 bones). The scene extras record the turn under `opensculptboy.mouth` (`opensculptboy/mouth@1`: the jaw's local rotation and the measures above).
- **VRM 1.0 and 0.x:** the humanoid `jaw` maps to the jaw bone, and the tongue bones stay outside the humanoid map. VRM expressions bind morph targets only, so the expressions keep anny's `jawOpen` target and the files carry no corrective; an application can turn the humanoid jaw on its own. The file's skin weights are those of the `anny` rig for the same body (with the eyelid and twist moves and the refinement of the VRM export), with each vertex's `head` weight shared among `head` and the mouth bones in the proportions of MakeHuman's weights (`mouth.rig.split_head_weights`); every vertex that the mouth bones do not move keeps its weights exactly.

`test.test_mouth_rig` checks the rig, the exact split of `jawOpen`, the GLB animation against the target and the VRM jaw and weights.

## A reference's inner mouth

Anny's mouth is a cavity of skin with a tongue inside and no teeth. A fitted character takes the inside of its reference's mouth when the appearance handoff describes it (`opensculptboy.mouth.handoff`, `opensculptboy.mouth.inner`).

### In the handoff

`appearance.json` may hold a `mouth` section:

- `pieces`: one entry per piece of the inner mouth, with its `role` (`gums_upper`, `gums_lower`, `palate`, `tongue`, `mouth_floor`, `throat`, or `tooth_<arch>_<side><n>` for a separate tooth) and its primitive of `replica.glb` (`mesh` and `primitive`); a piece that is part of a primitive lists its `vertices`.
- `materials`: the mouth's material, whose `converted` maps (`base_color`, `normal`, `roughness`: paths in the handoff folder) draw it.

`load_handoff` reads each piece straight from the GLB file with the skin weights of its own rig (`JOINTS_0` and `WEIGHTS_0`, which the trimesh loader leaves out), as a part of role `mouth` whose `mouth_role` names the piece, in place of the mouth's primitives. The fit leaves the pieces out, as it leaves out every part but the skin.

### In the card

`opensculptboy fit --textures` and `opensculptboy bake` take the pieces into the card's rest frame by the rigid motion (Kabsch) that takes the fitted head, as the fit placed and posed it, back to its rest pose, and write them next to the card:

- `<card>.mouth.safetensors` (`MouthData`, `opensculptboy/inner-mouth@1`): the points, normals, UVs and triangles, the arch of each point (the head's or the jaw's), the part of each triangle (`teeth`, `gums`, `palate`, `tongue`, `floor`, `throat`) and the skin of each point on the `anny-mouth` rig;
- `<card>.mouth_<slot>.png`: the mouth's maps;
- `material.mouth`: `{"mesh": ..., "base_color": ..., "normal": ..., "roughness": ...}`, with the optional `color` (the base colour without a map), `roughness_factor` and `clearcoat` (the wet surface, 0.6).

The card also takes the `anny-mouth` rig when its rig is `anny`. The skin of the reference's rig maps onto it by name: the jaw joint and the joints of the chin and the lower lip become `jaw`, the tongue chain becomes anny's chain `tongue00` to `tongue04` by its place from the root to the tip, and every other joint becomes `head`. Each point keeps its 4 strongest bones.

### In the files

The GLB and VRM exports append the mouth to the body as they append teeth (`opensculptboy.teeth.mesh.toothed_mesh`): it shares the body's vertices, morph targets and skin, and draws with one material, `mouth` (the reference's maps; in VRM files an MToon material with its base colour as the lit and the shade texture). A card with `material.mouth` takes no other teeth.

- **Hidden:** anny's tongue and the part of anny's cavity that lies behind the front of the inner mouth (`behind_mouth`: a vertex of the cavity, apart from the lips' inner margin, hides when a point of the inner mouth within 4 mm across the depth stands in front of it) stay in the mesh, so that the morph targets and the indices of the other parts keep their rows, but no primitive draws them. On the `anny-mouth` rig a row of the cavity hides only behind the pieces of its own arch: rows that follow the jaw (its weight over 0.5) behind the lower pieces, the others behind the upper ones, so that a jaw's turn moves a hidden row and the pieces in front of it alike. Lining that stays with the head, hidden behind the lower gums at rest, opened into holes in every open pose. The lining in front of the inner mouth stays, as it closes the space between the lips and the front of the gums.
- **Skin:** on a rig with the mouth bones, each point takes its own weights, so the lower gums, the lower teeth, the floor of the mouth and the tongue turn with the jaw. A bone the rig lacks gives its weight to its nearest kept ancestor (a tongue bone to `jaw`, `jaw` to `head`).
- **Morph targets:** as for teeth, the upper pieces stay with the head, and the lower pieces take the rigid motion of anny's tongue in a target that turns it (`jawOpen`, `jawLeft`). In a target that bends anny's tongue (`tongueOut`), the reference's tongue takes the displacement of anny's 4 nearest tongue vertices, blended by inverse distance.

### Checks

`test.test_mouth_inner` holds the handoff's pieces, the joint map, the placement and the file, the card's section, the hidden triangles, the skin of each piece, the pieces behind the drawn skin at rest and with the jaw open, the tongue following anny's in `tongueOut`, the GLB file with the jaw's animation and the VRM files of both versions. Its synthetic mouth is the generic teeth with anny's own tongue.

## Sealed lips

A reference's closed lips meet along a line. A game's head mesh draws its lips as separate rows whose borders touch, which leaves an opening in the skin; a sculpted head welds them, and they meet at the bottom of a crease. The fit misses that line: the reference's points pull anny's visible lips onto its surface, and the points of a crease project onto its walls, so each of anny's lips stops short. anny's own mouth is also modelled a little open. The fitted lips then part by a slit through which the inside of the mouth shows, widest at the corners.

`opensculptboy.fit.lips` closes them as the reference closes its own:

- `lip_line` finds where the reference's lips end, from the front (anny's -y), within a box around the mouth from anny's craniofacial landmarks `ch.L`, `ch.R`, `ls` and `li`. A reference with an opening there gives the opening's borders (`_opening_line`, its vertices welded where a source splits them at UV seams): in each column of 0.2 mm the border splits into the upper lip's and the lower lip's at its largest step in height, when that step exceeds 0.3 mm, and both lips share one line where the borders touch. Otherwise each column's most recessed point between the lips' fullest points is the bottom of the crease, kept where it lies at least 0.2 mm behind both lips. The result is a `LipLine` with an edge for each lip, one line where the lips meet; a mouth that stands open keeps them apart.
- `lip_masks` tells anny's lips apart by the displacement of anny's `jawOpen` face unit (`register.jaw_share`): the upper lip is the skin of the mouth that the jaw's opening leaves (under 0.3 of its largest move), the lower lip the skin it carries (over 0.6). Around the corners the share runs between the two, and the middle of the mouth splits it: the line midway between the lips' edges, run out to the corners.
- `lip_edges` gives the rows where each lip ends: the vertices of the upper lip that no vertex of the upper lip within half an edge's length of their x lies below, and the vertices of the lower lip that none of the lower lip lies above. The rolls behind them stay out, since at the same x they lie higher (or lower). `lip_corners` gives the vertex of the lips nearest each corner landmark, where both edges meet.
- `seal_lips` moves each edge row onto the reference's edge of the same lip at its own x, and each corner onto the line's nearer end. The rows of both lips within `FitOptions.lip_seal_beyond` (2 mm) past an end of the line move onto the line where the lips meet, carried on past that end along its direction, so that both lips meet up to the corner: left where they were, those rows opened into a dark notch at each corner. The mouth's skin within `FitOptions.lip_seal_rings` faces of those rows (3 on level 0, twice as many on each finer level) follows by the harmonic extension of the move in the rest frame, but its skin that lies on the reference (within 0.2 mm, the normals agreeing to 0.5) farther than `FitOptions.lip_seal_reach` (1.5 mm) from the rows stays.

The seal closes only lips that the reference closes. When the reference's lips stand further apart than `FitOptions.lip_seal_max_gap` (1 mm, the median of its line), the fit leaves them unsealed: a reference cut open at its mouth, as the synthetic target of the tests is, gives the borders of the cut as its line, and moving anny's lips onto them tore the mouth (the seal moved the rows 7.8 mm on level 0 and raised the torn edges from 0.4 % to 5.7 %). A level whose rows would move further than `FitOptions.lip_seal_max_shift` (4 mm, their mean) to reach the line also stays unsealed.

`fit_levels` runs the seal on each level after `smooth_mouth` and before `repair_mouth`, which then holds the sealed rows (`FitOptions.lip_seal`, on by default). `lip_slit` measures what the front view shows between the lips: the area where the fit lies more than 0.5 mm behind the reference within the lips' band, and the part of it within 4 mm of either corner; `lip_gap` the distance from each row of the upper lip to the lower lip's edge at its x. `fit_levels` records both under `FitResult.measures[level]["lips"]` and prints them.

Measured on the first real replica (its mouth an opening whose borders touch along 385 columns), from the same parameters, without the seal and with it:

| Level | Slit seen from the front (mm²) | Of it at the corners | Gap of the edge rows (mean, mm) | Visible lips to the replica (mean / p95, mm) | Torn edges, closed / open |
|---|---|---|---|---|---|
| 0 | 65.5 → 48.7 | 27.7 → 16.1 | 3.61 → 0.13 | 0.133 / 0.836 → 0.060 / 0.414 | 12 / 9 → 65 / 53 (of 2,133) |
| 1 | 48.6 → 3.8 | 24.0 → 0.9 | 3.01 → 0.17 | 0.089 / 0.676 → 0.036 / 0.228 | 33 / 58 → 104 / 101 (of 8,576) |
| 2 | 30.2 → 2.8 | 7.9 → 0.8 | 2.82 → 0.02 | 0.026 / 0.217 → 0.019 / 0.145 | 78 / 105 → 230 / 233 (of 34,892) |

The gap without the seal is measured to the nearest row of the lower lip, which can only make it smaller. On levels 1 and 2 the lips now meet along the whole line, and small marks remain at the corners of level 2; level 0 is too coarse for its rows to follow the line. The torn edges (`metrics.mouth_tears`: folds over 60 degrees and 45 more than on the card without offsets) grow, most of them at the corners, where the fans of anny's mouth corners take the corner's move within a few faces. Without the corners' own targets (`lip_seal_corners=False`) level 1 kept 9.2 mm² of slit, 6.3 of it at the corners.

On level 2 the repair and the seal reach further around the corners: `FitOptions.mouth_repair_reach` is 5, 5 and 8 faces on levels 0, 1 and 2, and `lip_seal_reach` 1.5, 1.5 and 3 mm (one value, or one per level). Anny's lips fold into each other at the corners more deeply than a replica's, and on level 2 the fit crumpled that fold into a fan of creases below each corner and a small pocket at the corner itself. On the first real replica at rest, level 2, the visible edges within 4 mm of anny's corner landmarks that bend more than 20 degrees beyond the card without offsets fall from 15.5 % to 10.4 % (259 of 1,669 against 132 of 1,269), those beyond 45 degrees from 71 to 45, and the slit at the corners from 0.40 to 0.08 mm² (along the whole mouth from 2.10 to 1.52 mm²). Over the 11 moving poses of the reference, the folds near the mouth fall from 1,710 to 1,456 edges and the torn edges from 477 to 416, the mouth opening seen from the front keeps its overlap with the replica's (0.829 against 0.826 on average) and the lips stay as close to the posed replica (0.084 against 0.091 mm, median). By eye the corners' pocket narrows, and a twisted crease remains just above it. On level 1 the wider repair deepened the corners' pockets (the slit at the corners from 0.52 to 4.3 mm²), so level 1 keeps the former reach. A fairing of the skin around the corners on the reference, as `fair_skin` fairs the face, raised the folds and tears of the open poses and moved the lips off, and stays out.

## Reference poses

A handoff may carry the reference's own mouth poses under `mouth.poses` of appearance.json: per pose, a `joints` object that gives joints of the GLB file's skins a `translation` and a `rotation` (a quaternion x, y, z, w), and `known: false` for a pose that is a guess. `opensculptboy.fit.poses.GlbRig` reads the file's nodes, skins and skinned primitives, and `pose` applies a pose as the contract defines it: each joint's posed local transform is its own translation plus the delta's and its own rotation times the delta's, and the skins move the vertices by linear blend skinning with their inverse bind matrices, as a glTF viewer does. `posed_parts` gives the posed vertices of every part of a loaded handoff in Anny's frame, matched to the file's primitives by their rest positions.

## The jaw's weights from the reference

A reference's rig need not give its lips to the jaw. A game's face rig moves the lips and the chin with joints of their own, and its mouth poses turn them together with the jaw: on the first real replica no skin vertex takes any weight of the jaw itself. What carries over is how far each point of the skin follows the jaw when the reference opens its mouth. `mouth.weights.reference_jaw_share` poses the reference with each of its known mouth poses that turns the jaw by 2 degrees or more (`MIN_TURN_DEG`) and measures, for each skin vertex with motion d and the motion j that the jaw's turn alone would give it, the share sum(d . j) / sum(j . j) over those poses, clipped to 0..1; skin that the jaw's turn moves by less than 0.5 mm over the poses (`MIN_MOTION`) takes 0. `jaw_share` carries the share to a fitted level: anny's skin within 3 mm of the reference takes the value at its closest point, and the inside of the mouth and the lips' hidden roll take the harmonic extension of those, so that each lip's lining follows its lip. The tongue keeps its own weights.

`opensculptboy fit` (and `attach_jaw_share`) records the share of level 0 as `<card>.jaw_share.safetensors` (`opensculptboy/mouth-weights@1`) under `detail.mouth`, with the poses it comes from. `card_model` gives a model of the `anny-mouth` rig the jaw weights that follow it: on each vertex that the head, the jaw or the share moves, the head's and the jaw's weights together are split as (1 - share, share), and every other bone keeps its weight (`with_jaw_share`).

The exports keep MakeHuman's jaw weights for now. On the first real replica a turn of the jaw alone, with the weights of the share, leaves the lower lip behind and stretches it over the opening: the game opens the lips with joints of their own, so that only 255 of level 0's 13,784 vertices follow the jaw by more than half. With MakeHuman's weights, the sealed lips part cleanly when the jaw turns by the 7.9 degrees of the game's widest pose. The share is the data for the mouth expressions to come, which add the lips' own motion as correctives on the jaw.

On the first real replica, 7 of its 12 mouth poses turn the jaw by 2 degrees or more (the visemes `a`, `e` and `o`, the emotional `a` and `o`, the widest frame of the talking loop and a cutscene's speech); the closed mouth, the laugh, the teeth view and the visemes `i` and `u` count for nothing.

`test.test_mouth_lips` checks the lip line on surfaces with a touching opening, an open mouth, a crease and none, the seal of anny's own lips onto a line between them, the jaw share of a small skinned GLB file under a pose, and the split of the weights.

## Mouth expressions from the reference

A fitted character with a reference's inner mouth takes its reference's own mouth poses (`mouth.poses` of appearance.json) as morph targets (`opensculptboy.mouth.poses`):

- `reference_pose_fields` poses the reference with each known pose (`fit.poses.posed_parts`) and gives the motion of every vertex of its skin and of every piece of its inner mouth. A pose that moves nothing (the closed mouth) is left out. The pieces of the inner mouth are rows of one primitive of the file, so `posed_parts` matches a part that no primitive matches as a whole to the rows of a primitive at its vertices' rest positions (`_subset_rows`): matched by vertex count alone, the gums, the tongue and the mouth's floor kept still in every pose, and the tongue stood behind the upper teeth with the scalloped edge of the lower gums while the separate lower teeth dropped.
- `carry_field` carries the skin's motion to level 0 of the fitted card: a vertex within 3 mm of the reference takes the motion at its closest point, unless that point lies within 1.5 mm of a border of the reference (a mouth opening, where the closest point can fall on either lip); the inside of the mouth, the lips' hidden roll and those rows take the harmonic extension of the others (`register.smooth_mouth`), so that each lip's lining follows its own lip. With the jaw's share of each vertex (`register.jaw_share`) the lining takes two extensions, one from the skin that follows the jaw (share 0.5 or more) and one from the rest, mixed by its own share: one extension from both lips made the lower lip's lining lag behind the lower lip, which pinched the lower lip into a fold in the open poses.
- The lips themselves follow the reference's rig (`reference_pose_transforms`): each lip's rows of `lip_sides`, and the lining off the skin within 3 faces of them (`lining_sides`, `LINING_RINGS`, each row by the lip that reaches it first), move by the reference's blended skin transform at their closest point on the reference's same lip, applied to their own positions. The transform is the linear blend of the skin's joint matrices with the reference's weights at that point, so the rows that lie off the reference's surface (the inner roll of the lips and the lining just inside them) roll with their lip as the reference's own lips do. Carried as the surface's motion alone, those rows took the harmonic extension, which pulled the upper lip's inner edge up into an arch above the teeth and thinned both lips in the wide poses. On the first real replica, level 2, the share of the mouth opening seen from the front that matches the replica's went from 0.76 to 0.87 in the widest pose (0.68 to 0.78 for `u`, the lowest), the mean offset of the opening's lower edge from the replica's from 2.7 to 1.4 mm, the median distance of the lips to the posed replica from 0.15 to 0.13 mm and its 90th percentile from 1.5 to 1.0 mm. The rough spots of this motion are smoothed more lightly (`LIP_SMOOTH_WEIGHT_RIG`, 1 in place of 4), which keeps the new folds near the mouth at the level of the carry before (50 against 58 edges in the widest pose).
- Before the fill, `smooth_lip_motion` smooths the motion that the lips (the rows of `lip_sides`) and the skin within 2 faces of them took from the reference directly, among those vertices only: a vertex whose carried motion stands out of its neighbours' mean by more than 0.8 mm (`LIP_ROUGH`), and one ring around it, take the motion closest to the carried one that is smooth over their neighbours (30 Jacobi steps of (I + 4 L) o = carried), and the rest keep theirs, so that a real ramp of the motion keeps its shape (smoothing every lip vertex moved the lips' median distance to the reference from 0.1 to 0.3 mm in the open poses), without the edges between the upper and the lower lip but within one face of the mouth's corners: the carried motion keeps the roughness of the reference's own triangles, which the detail levels folded into creases and torn spots.
- `mouth_poses` turns the motion into the card's rest frame by the rigid motion of the fitted head (the Kabsch fit that also places the inner mouth) and gives the inner mouth's points the motion of their own pieces. The result is a `MouthPoses`: rest-frame offsets per pose for anny's level 0 and for the inner mouth.

`opensculptboy fit --textures` and `opensculptboy bake` write it as `<card>.mouth_poses.safetensors` (`opensculptboy/mouth-poses@1`) and name it under `material.mouth.poses` (`write_card_poses`). The exporters add one morph target `mouth.<pose>` per pose: the offsets of level 0, carried to a detail level by its subdivision like every other target, and the inner mouth's own offsets (`ToothedMesh.targets(overrides=...)`).

- **GLB:** on the `anny-mouth` rig each pose also splits into a turn of the jaw and a corrective, as anny's jawOpen does (`pose_turns`): the target `mouth.<pose>.corrective` and the animation `mouth.<pose>.jaw`, whose last keyframe shows the pose's target. The scene extras list the poses under `opensculptboy.mouth.poses`, with the jaw's turn of each.
- **VRM:** VRM expressions bind morph targets only. The files take the targets of the vowels (`VRM_VISEMES`: `viseme_a`, `viseme_i`, `viseme_u`, `viseme_e` and `viseme_o`), and the presets `aa`, `ih`, `ou`, `ee` and `oh` (`A`, `I`, `U`, `E`, `O` in VRM 0.x) bind them at weight 1 in place of their ARKit mixes. The other expressions keep their mixes of anny's facial actions.

`test.test_mouth_poses` checks the carry of a posed reference's motion onto a surface, the file, the GLB targets, turns and animations (the animation's last keyframe gives the pose's target, the inner mouth included) and the visemes of both VRM versions.

## Next steps

The mouth expressions come from the reference's poses as whole targets; the ARKit mixes of anny's facial actions stay for the expressions that have no pose. Whether `anny-mouth` should become the default rig is open: it changes every file's joints and the VRM skin weights near the mouth, and the visemes would need a corrective each.

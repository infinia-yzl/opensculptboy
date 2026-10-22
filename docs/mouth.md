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

- **Hidden:** anny's tongue and the part of anny's cavity that lies behind the front of the inner mouth (`behind_mouth`: a vertex of the cavity, apart from the lips' inner margin, hides when a point of the inner mouth within 4 mm across the depth stands in front of it) stay in the mesh, so that the morph targets and the indices of the other parts keep their rows, but no primitive draws them. The lining in front of the inner mouth stays, as it closes the space between the lips and the front of the gums.
- **Skin:** on a rig with the mouth bones, each point takes its own weights, so the lower gums, the lower teeth, the floor of the mouth and the tongue turn with the jaw. A bone the rig lacks gives its weight to its nearest kept ancestor (a tongue bone to `jaw`, `jaw` to `head`).
- **Morph targets:** as for teeth, the upper pieces stay with the head, and the lower pieces take the rigid motion of anny's tongue in a target that turns it (`jawOpen`, `jawLeft`). In a target that bends anny's tongue (`tongueOut`), the reference's tongue takes the displacement of anny's 4 nearest tongue vertices, blended by inverse distance.

### Checks

`test.test_mouth_inner` holds the handoff's pieces, the joint map, the placement and the file, the card's section, the hidden triangles, the skin of each piece, the pieces behind the drawn skin at rest and with the jaw open, the tongue following anny's in `tongueOut`, the GLB file with the jaw's animation and the VRM files of both versions. Its synthetic mouth is the generic teeth with anny's own tongue.

## Next steps

Planned next, in phases: the lips sealed where the reference's lips meet, and the mouth expressions rebuilt on the jaw from the reference's mouth poses. Whether `anny-mouth` should become the default rig is open: it changes every file's joints and the VRM skin weights near the mouth, and the visemes would need a corrective each.

# The mouth

`opensculptboy.mouth` gives exported characters a jaw. Anny's default rig (`anny`) opens the mouth with the facial action `jawOpen`, a fixed offset per vertex, and has no jaw bone. The `anny-mouth` rig keeps MakeHuman's jaw bone and tongue chain, so that a file can open the mouth by turning the jaw.

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

## Next steps

Planned next, in phases: a reference's inner mouth (teeth, gums, tongue, lining) on the jaw and the tongue chain, the lips sealed where the reference's lips meet, and the mouth expressions rebuilt on the jaw from the reference's mouth poses. Whether `anny-mouth` should become the default rig is open: it changes every file's joints and the VRM skin weights near the mouth, and the visemes would need a corrective each.

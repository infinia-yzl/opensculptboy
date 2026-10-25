# VRM avatars

`opensculptboy export` writes a VRM avatar when the output file ends in `.vrm`. VRM 1.0 is the default, for Warudo, VMagicMirror and three-vrm, and `--vrm-version 0` writes VRM 0.x for VSeeFace and 3tene.

```bash
opensculptboy export ada.vrm --character ada.json --author "Your name"                   # VRM 1.0
opensculptboy export ada0.vrm --character ada.json --author "Your name" --vrm-version 0  # VRM 0.x
opensculptboy export ada.vrm --author "Your name" --thumbnail auto --meta meta.json      # a rendered portrait, and licence fields from a file
```

```python
from opensculptboy import Character, export_vrm

export_vrm("ada.vrm", Character(name="ada", phenotype={"muscle": 0.7}), author="Your name")
```

## What the file holds

The avatar rests in the VRM T-pose: the arms, hands and fingers point straight out to the sides with the palms down, the thumbs point level at 45 degrees toward the front, the legs hang straight and the feet point forward. The file holds:
- the mesh with 4 skin weights per vertex, and the humanoid bone map of the `anny` rig (with `jaw` on the `anny-mouth` rig, [Mouth](mouth.md));
- expressions mixed from the 52 ARKit facial actions: the visemes aa, ih, ou, ee and oh, blink, blinkLeft and blinkRight, the emotions happy, angry, sad, relaxed and surprised, and neutral; in VRM 1.0 files an emotion damps blinking and lip sync in proportion to its weight;
- the 52 facial actions as perfect-sync expressions named in PascalCase (`EyeBlinkLeft`), and in VRM 0.x files the VSeeFace visemes SIL, CH, DD, FF, KK, NN, PP, RR, SS and TH;
- look-at on the eye bones, and MToon toon materials; the primitives carry no `TANGENT`, so a normal map takes its tangent frame from the UVs as glTF asks (UniVRM computes tangents on import, three-vrm a frame per pixel), since three-vrm's MToon shader fails to link a primitive with vertex tangents and a normal map;
- the licence metadata with the credits, and an optional thumbnail;
- the [character card](character-card.md) in the scene extras, so that `read_character("ada.vrm")` reads it back.

The face shapes are baked into the mesh, because a VRM skeleton is fixed, and the facial actions become the expressions. The eyes of `--eyes` with their lashes, the teeth of `--teeth` and a fitted card's inner mouth go into the file as they do for [glTF](export.md).

## Metadata and licence

A VRM file needs an author: pass `--author`, or `"authors"` in a `--meta` file of `VrmMeta` fields. The licence fields start restrictive: only the author may use the avatar, for personal non-profit use, without redistribution or modification. `--meta` changes them. The `allow_*` fields take a JSON `true` or `false`; `avatar_permission`, `commercial_usage`, `credit_notation` and `modification` take one of their VRM 1.0 values, such as `"commercial_usage": "personalProfit"` (an invalid value prints the choices). `--thumbnail auto` renders a square portrait, and a given thumbnail must be a square PNG or JPEG file. Every file credits OpenSculptBoy, Anny (NAVER Corp., Apache 2.0), the MakeHuman, MPFB2 and Face Units assets (CC0) and ICT-FaceKit (MIT).

## Rig, bind and skin weights

A VRM file needs the `anny` rig (or `anny-mouth`) and the MakeHuman body mesh with its eyes: the `anny`, `makehuman` or [sculpt](sculpt-topology.md) topology, with modifiers such as `-quads` or `-full`. The export refuses the SMPL and SMPL-X topologies, whose data is non-commercial, and the retopologised meshes.

VRM apps turn only the humanoid bones. A VRM 1.0 file therefore turns the forearm twist bones with roll constraints from the wrists (`--twist constraint`), and merges the other twist weights into their bones. A VRM 0.x file has no constraints, so it merges every twist weight (`--twist merge`).

- `--bind inverse` (the default) gives the mesh that the file's 4 skin weights pose back onto the character's own rest pose exactly. In arms-down poses, the usual idle of a VTuber avatar, 99 % of the vertices land within about 8 mm of the character's own posed mesh.
- `--bind forward` skins the character into the T-pose with its own weights, which gives a smoother T-pose and suits poses with raised arms.
- `--keep-leg-spread` keeps the legs at the rig's natural spread.

Straightening the arms, the fingers and the legs into the T-pose folds the skin where two neighbouring bones turn toward each other: the webs of the fingers and the thumb, and the groin. The export refines the file's skin weights for each character around those places, which removes about 80 % of those folds and adds a few seconds to an export (`--no-refine-weights` turns it off).

## Budgets

Each export prints its counts and checks the budget of its version: 70,000 triangles, 200 joints, 16 materials and 4096 px textures for VRM 1.0, and 32,000 triangles, 128 joints, 8 materials and 2048 px textures for VRM 0.x. A fitted card's finer detail levels step down to stay within it ([Fit to a reference](reference-fit.md)). `--budget strict` fails a file over its budget with exit status 1, and `--budget off` skips the check. Invalid options, metadata, thumbnails or cards end the command with one error line and exit status 2.

The summary lists the pieces of the file under `pieces` (`opensculptboy.export.pieces.piece_report`): the body, the eyes, the lashes, the teeth or the inner mouth, and each wearable piece by its reference, each with its triangles, its vertices, its materials, its textures with their sizes and the joints that weight it. A file over its budget names the same pieces in its warning, and in the error of `--budget strict`, so that a user or an agent sees what to cut.

### Fitting a file into its budget

`--fit-budget` (`export_vrm(..., fit_budget=True)`, or a `FitOptions` of `opensculptboy.export.fit_budget`) changes the file until it keeps within the budget of its version, prints one line per step on stderr (`fit: ...`) and gives the full report under `fit` in the summary. A file within its budget stays as it is, byte for byte, and without the option every export stays as it was. The steps run only for the counts over the budget:

1. **Triangles.** One reduction over every piece removes the vertices whose removal costs least (`opensculptboy.export.decimate`). A collapse moves a vertex onto a neighbour, so each kept vertex keeps its position, normal, UVs, skin weights, morph target offsets and its wearable binding (the bound point and offset of a surface binding, the own weights or the bone), and the inverse bind of every kept vertex stays exact. Open edges (hems), UV seams, hard edges and material borders keep their lines; a collapse that would flip a triangle, flip a UV triangle, join two sheets or make a sliver is refused, and a connected part of two triangles or fewer (a fur card, a gem) stays. The cost of a collapse is its distance from the surface in the rest shape and in the six morph targets that move the most, times the vertex's importance: its piece's priority (the face of the body nine times and its hands four times, the wearables at the head four times; the inner mouth and the teeth half), times the share of its surface that ten views around the figure see (the front the most), so inner layers and hidden faces go first; a small part of a wearable (a pendant, a buckle, a gem: under 20 cm²) weighs more as it gets smaller, up to eight times. The eyes and the lashes stay whole, and each piece keeps a floor of its triangles: the body at least 30 % (and 95 % while the file keeps half of its triangles), the inner mouth and the teeth 15 % and 25 %, each wearable 8 %. When the budget asks for less than 35 % of the triangles, the small parts of the wearables (64 triangles or fewer) that no view sees go first. When the collapses cannot reach the budget, the small parts (16 triangles or fewer) of the wearables that the views see least (fur cards, gems) go whole, then the wearables' floors give way, and the body's last.
2. **Normal maps.** For each reduced wearable material, rays from its texels find the full piece, and where the mean angle between the full and the reduced shading normals exceeds 1 degree, the normal map is baked again from the full piece in the reduced piece's MikkTSpace frames (`opensculptboy.export.rebake`; it needs Embree, from the `viewer` extra). Texels of mirrored or repeated islands take the mean of their sides where these agree, and a material without a normal map gets one where the bake brings its shading closer. A bake that does not lower the error is dropped.
3. **Materials.** The wearables' MToon materials that draw alike (alpha mode and cutoff, double-sided, shading, rim and outline) merge into atlases (`opensculptboy.export.atlas`); when the groups are still too many, the cutoffs of `MASK` materials join, then `OPAQUE` joins `MASK`, then single-sided joins double-sided. Each tile copies the texels of its UV box exactly at its native size, with the base colour factor and the normal scale baked in (colour in linear light with premultiplied alpha, normals as vectors), an 8-texel margin from the same map, and an origin on a multiple of 8 texels for the mip levels. A page larger than the texture size shrinks the densest tiles first (texels per metre), and the base colour and the normal map share one layout. The skin, its tiles, the eyes, the lashes and the mouth keep their materials.
4. **Joints.** Over the joint budget, the joints that weight no vertex leave the skin; their nodes stay in the humanoid map.
5. **Textures.** Every image larger than the texture size of the version shrinks to it (VRM 0.x: 2048), colour in linear light, normal maps as vectors.

`--fit-priority PIECE=FACTOR` scales the importance of a piece (`body`, `eyes`, `lashes`, `teeth`, `mouth`, `wearable` or a piece's reference such as `travel/cape`; a priority makes the eyes and the lashes reducible), and `--fit-triangles PIECE=COUNT` fixes a piece's triangles; both repeat. The report holds, per step, what it did, and under `quality` the measures of `opensculptboy.export.budget_quality` against the full file, drawn from five views: the silhouette difference (share of pixels and mean offset in millimetres), per piece the distance between the surfaces, the angle between the geometric normals and between the shading normals (with the normal maps), the change of the drawn colour, the UV triangles of zero area, and the texel density of the base colour before and after. The scene extras of a fitted file record the steps under `opensculptboy.budget_fit`.

A dressed character of 146,806 triangles, 34 materials and 4096 px textures (4 wearable pieces) fits VRM 1.0 at 70,000 triangles and 7 materials with every tile at its native size, and VRM 0.x at 32,000 triangles, 8 materials and 2048 px textures. The skin of that character holds 116 joints, so spring joints for the wearables (a later version) have 12 joints of room under the VRM 0.x budget of 128, plus the joints that weight no vertex, which the fit takes out of the skin; the fit does not merge or remove weighted joints.

Limits of the fit: the reduction places no new vertices, so a very tight budget flattens curved pieces into fewer, larger triangles before the floors give way; the parts that no view sees are judged in the T-pose, so a part that a pose reveals (under a cloak that swings) may be gone; only the wearables' materials join atlases; the normal maps need Embree to be baked again, and without it the reduced pieces keep their maps as they are; the fit of that character takes about 80 s with 16 threads.

## Spring bones

The spring chains of the [wearables](wearables.md#springs) become `VRMC_springBone` springs in VRM 1.0 files: one spring per chain, its tail last, with the body's sphere and capsule colliders in the groups `head`, `torso`, `legs` and `arms`. VRM 0.x files carry them as `secondaryAnimation`: a bone group per spring of a piece, with the first joint of each chain in `bones`, and a collider group per collider, whose capsules become spheres along their axes. The joints ride on their parent bones into the T-pose and keep the identity rotation, as every node does.

The chains' skin joints count against the joint budget. Unless `--budget off` is given, a spring takes fewer joints (3 at least), then fewer chains, then leaves the file until the file fits, and the export reports it with a `SpringWarning` and in `spring_messages` of the summary: the 104 bones of the `anny` rig leave a VRM 0.x file 24 joints, enough for a cape of 5 chains and a skirt of 7, each of 3 joints.

## Testing

The files pass the VRM 1.0 and 0.x schemas (`test/test_vrm_schema.py`), and the fitted files of `--fit-budget` too (`test/test_vrm_fit_budget.py`, with synthetic meshes and garments). three-vrm loads both versions with their look-at, expressions and spring bones, and the Khronos glTF validator reports no errors (`test/test_vrm_three.py`). VSeeFace, Warudo, VMagicMirror and UniVRM are untested so far.

## Not in the file yet

Hair and its spring bones; outfits with their spring bones and lashes come with [Wearables](wearables.md) and [Eyes](eyes.md#lashes). Painted textures come only from a reference bake; other characters get a plain skin colour.

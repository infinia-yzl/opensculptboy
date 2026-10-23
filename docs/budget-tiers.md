# Rendering-budget tiers

A character is authored once, as its master: the finest level of the card's detail layer with the card's own maps, the textures that `opensculptboy fit --textures` or `opensculptboy bake` baked from the reference (see [reference-fit.md](reference-fit.md)). Cheaper versions of the same humanoid 3D character are derived from the master and only ever cut down from it: a coarser mesh, smaller textures, fewer maps and a cheaper skin. Nothing in a cheaper tier is authored by hand, so a change to the master carries to every tier by baking and exporting again.

```bash
uv sync --extra viewer                                                  # SciPy and embreex for the bake
opensculptboy bake-tiers character.json                                 # the normal maps of the high, medium and low tiers, and the occlusion
opensculptboy export character_master.glb --character character.json --tier master
opensculptboy export character_high.glb   --character character.json --tier high
opensculptboy export character_medium.glb --character character.json --tier medium
opensculptboy export character_low.vrm    --character character.json --tier low --author "Your name"
opensculptboy export character_low        --character character.json --tier low --author "Your name"   # no suffix: the tier's format (.vrm)
opensculptboy portraits --character character.json --out avatars/       # portrait.png, five expressions and expressions.png
```

```python
from opensculptboy import Character
from opensculptboy.tiers import TIERS, bake_tiers, export_tier, tier_budget, tier_textures

character, report = bake_tiers("character.json")                # writes the maps next to the card and updates it
summary = export_tier("character_medium.glb", character, "medium", detail_dir=".")
print(summary["tier_budget"])                                   # the mesh, the textures and their GPU memory
```

## The tiers

| Tier | Mesh | Triangles | Vertices | Skin | Atlas | Head tile | Ear tile | Eyes | Morph targets | Bones per vertex | Format |
|---|---|---|---|---|---|---|---|---|---|---|---|
| master | level 2 | 438,720 | 219,368 | full subsurface scattering (Cycles random walk) | 4096 | 2048 by 4096 | 2048 by 1024 | 1024 by 512 | 52 facial actions | 8 | glTF |
| high | level 2 on the head and the neck, level 1 elsewhere | 220,992 | 110,504 | engine subsurface skin (Unreal Subsurface Profile, Unity HDRP Diffusion Profile) | 2048 | 1024 by 2048 | 1024 by 512 | 1024 by 512 | 52 facial actions | 8 | glTF |
| medium | level 1 on the head and the neck, level 0 elsewhere | 55,272 | 27,644 | normal and roughness maps, baked thickness and curvature for a cheap skin glow | 1024 | 512 by 1024 | none | 512 by 256 | 52 facial actions | 4 | glTF |
| low | level 0 (anny's own mesh) | 27,420 | 13,718 | toon (MToon) with the baked skin ramp and ambient occlusion | 1024 | none | none | 512 by 256 | the 31 facial actions of the expression presets (glTF); 52 in VRM | 4 | VRM 1.0 |

`opensculptboy.tiers.TIERS` holds the presets (`Tier`). The mesh of a tier is the tier's own mesh or the master's, whichever is coarser (`tier_mesh`): a card whose finest level is 1 has a master on level 1, a high tier on level 1 and a medium tier on level 1 on the head; a card without a detail layer gives level 0 to every tier. The textures of a tier are the card's textures that the tier keeps, shrunk to the tier's sizes (never enlarged; a normal map is averaged as vectors and normalised again), with the normal maps that `bake-tiers` baked for the tier's own mesh in place of the master's (`tier_textures`). Each file records the tier in its scene extras (`opensculptboy.tier`: the name, the mesh, the shading, the sizes and the maps), and the glTF skin materials name the shading in their extras (`opensculptboy_skin.shading`: `path_traced`, `subsurface_profile`, `preintegrated` or `toon`).

The maps of each tier, on the atlas and on the tiles that the tier keeps:

| Tier | Base colour | Normal | Roughness | Thickness | Curvature | Occlusion | Subsurface settings |
|---|---|---|---|---|---|---|---|
| master | yes | the master's own | yes | yes | yes | no | in the extras, for Blender |
| high | yes | baked for the body; the head and the ears keep the master's, whose mesh they share | yes | yes (transmission) | no | no | in the extras, for the engine's profile |
| medium | yes | baked for the body and the head | yes | yes | yes | no | in the extras, for the pre-integrated lookup |
| low | yes | baked | no | no | no | yes (glTF `occlusionTexture`; in VRM, the shade texture) | the skin ramp of MToon |

`--morph-targets`, `--influences`, `--animation` and the VRM options still apply to a tier and override its choices; `--detail` does not, since the tier sets the level.

The eyes, the teeth and a reference's inner mouth come with every tier: `--eyes` and `--teeth` apply to a tier as they do to any export, the eye texture and its normal map shrink to the tier's eye size, and a card with a reference's inner mouth keeps its pieces, their maps (at their own size) and its mouth poses (`mouth.<pose>` targets in glTF, the vowels in VRM) in every tier.

## Budgets

| Tier | Triangle budget | Largest texture | Texels (every map present) | GPU memory, RGBA8 with mipmaps | GPU memory, BC7 with mipmaps |
|---|---|---|---|---|---|
| master | 440,000 | 4096 | 120.1 million | 611 MB | 153 MB |
| high | 225,000 | 2048 | 25.7 million | 131 MB | 33 MB |
| medium | 60,000 | 1024 | 7.0 million | 35 MB | 9 MB |
| low | 32,000 | 1024 | 3.3 million | 17 MB | 4 MB |

The grey maps (roughness, thickness, curvature and occlusion) are separate PNG files; an engine that packs them into the channels of one texture (Unity's mask map, Unreal's ORM textures) needs a quarter of their memory. `export_tier` counts the written file (`opensculptboy.export.budget.counts`) and warns, and records in its summary (`tier_messages`), when the file is over its tier's budget; `tier_budget` gives the mesh, the size of every texture, the texels and their memory before a file is written.

Against the VRM budgets of `opensculptboy.export.budget.BUDGETS` (VRM 1.0: 70,000 triangles and 4096 texels; VRM 0.x: 32,000 triangles and 2048 texels): the low tier fits both versions, with every texture at 1024 or less; the medium tier fits VRM 1.0 (55,272 triangles) and not VRM 0.x; the high and master tiers fit neither, so `export_tier` refuses to write them as VRM files (`--budget off` writes them all the same). A VRM file of a tier takes the tier's mesh as it is, instead of stepping down as `--detail` does.

## Baking the tiers from the master

Every level of the detail layer lays out anny's atlas alike, so the maps of the surface (the base colour, the roughness, the thickness, the curvature and the occlusion) serve every tier as they are, resized. A normal map does not: it holds the shading normal in the tangent frames of the mesh it was baked on, and a coarser mesh has other normals and lacks the master's pores and folds. `opensculptboy bake-tiers` (`opensculptboy.tiers.bake_tiers`) therefore bakes the master onto the mesh of each cheaper tier (`opensculptboy.texture.transfer`):

1. `master_reference` turns the master's rest mesh and its normal maps into the reference of a texture bake: one part per map (the atlas, the head tile and the ear tile, each on its own UVs), whose shading normals are the master's welded vertex normals bent by its normal maps in its MikkTSpace frames, as Blender and glTF viewers read them.
2. `bake_normals` casts, for each texel of the tier's mesh, a ray along that mesh's normal (6 mm each way) onto the master, and writes the master's shading normal there in the MikkTSpace frame of the tier's mesh, which the exporters write as its `TANGENT` (the bake of `opensculptboy.texture.bake`, with the master as the reference).
3. A tier bakes only the maps whose mesh differs from the master's (`needs_bake`): the atlas whenever its mesh is coarser, and a tile when the tier keeps the tile and its head is coarser than the master's. The high tier's head and ears are the master's own level 2, so it keeps the master's head and ear tiles and bakes the atlas alone.
4. A tier that keeps the occlusion, the thickness or the curvature and finds the card without it gets the map from the master's geometry, at the largest size a tier asks for: the occlusion of `vertex_occlusion` (32 cosine-weighted rays per vertex over the hemisphere, each occluding by `1 - t / 50 mm` when it meets the mesh at `t`), the thickness and the curvature of the texture bake.

The bake writes `<card>.<tier>.normal.png` (and `<card>.<tier>.head_normal.png`, `<card>.<tier>.ears_normal.png`) and `<card>.occlusion.png` next to the card, and records them in the skin material:

```json
"material": {
  "normal": "character.normal.png", "occlusion": "character.occlusion.png", "...": "...",
  "tiers": {
    "high": {"subdiv": 2, "region": "head", "normal": "character.high.normal.png"},
    "medium": {"subdiv": 1, "region": "head", "normal": "character.medium.normal.png", "head_normal": "character.medium.head_normal.png"},
    "low": {"subdiv": 0, "region": "full", "normal": "character.low.normal.png"}
  }
}
```

Each entry names the mesh its maps belong to; a tier whose mesh has changed since (a card that gained a level) uses them with a warning until `bake-tiers` runs again, and a tier without an entry takes the master's normal map with a warning. The master's normal maps are read at scale 1, and the material's `normal_scale` applies to every tier as to the master. `bake-tiers` on a level-2 master with 1,024 to 2,048 texel atlases took 55 s and 2.2 GB of memory: 25 s for the high tier, 10 s for the medium tier, 6 s for the low tier and 13 s for the occlusion, thickness and curvature maps.

### Measured improvement

`test/tier_fixtures.py` builds a synthetic master: anny with smooth offsets of 2 mm on every level, bumps of about 0.25 mm along the normal on the finest level (pores and folds a few millimetres across) and a normal map of bumps of 7 degrees on the atlas. The error is the angle between the master's shading normal and the tier's, at 20,000 random points of the tier's mesh (uniform over its area, the eyeballs left out), with the tier's normal map read as an engine reads it from the file; the table gives the mean, the median and the 95th percentile in degrees, at the tiers' own atlas sizes.

| Master | Tier (mesh, atlas) | No normal map | The master's normal map | The baked normal map |
|---|---|---|---|---|
| level 2 | high (level 2 head, 2048) | 6.67 / 6.74 / 10.37 | 1.74 / 1.31 / 4.81 | 0.28 / 0.14 / 0.67 |
| level 2 | medium (level 1 head, 1024 and its head tile) | 7.11 / 6.97 / 11.96 | 2.92 / 2.26 / 7.28 | 0.61 / 0.21 / 2.03 |
| level 2 | low (level 0, 1024) | 7.10 / 6.94 / 11.82 | 2.82 / 2.10 / 7.45 | 0.64 / 0.21 / 2.21 |
| level 2, without its normal map | low (level 0, 1024) | 2.79 / 2.06 / 7.40 | none | 0.62 / 0.19 / 2.16 |
| level 1 | low (level 0, 512, the test) | 6.70 | 1.80 | 0.48 |

The master's own normal map on a coarser mesh restores the bumps of the map but not the master's geometry, and reads in the wrong tangent frames; the baked map holds both. Without a normal map on the master, the bake carries the geometry of level 2 alone onto level 0 and cuts the error from 2.79 to 0.62 degrees. `test.test_budget_tiers` measures the last row in CI and holds the baked error under 0.4 times that of the master's map.

## Engine notes

The files are glTF 2.0: metres, Y up, the figure facing +Z, metallic-roughness materials with `KHR_materials_specular` and `KHR_materials_ior`, MikkTSpace tangents, and the skin's subsurface settings in each skin material's extras under `opensculptboy_skin` (format `opensculptboy/skin@1`, see [reference-fit.md](reference-fit.md#the-skin-material)): `subsurface.weight`, `subsurface.radius_mm` (the mean free path per RGB channel), `subsurface.ior`, `subsurface.anisotropy`, `thickness` and `curvature` as texture references with their scales (`max_mm`, `max_per_m`), and `shading`. The importers of the engines keep the extras as metadata at best, so the skin is set up by hand or by an import script, as follows. These notes map the fields; they are not engine code, and the property names follow Unreal Engine 5 and Unity HDRP 14 to 17, which may differ in other versions.

### Unreal Engine (high: Subsurface Profile; medium: Preintegrated Skin)

- **Textures.** glTF normal maps are +Y (OpenGL); Unreal reads -Y (DirectX). Set Flip Green Channel on the normal textures unless the importer already flipped them. The roughness texture reads its G channel (glTF's metallic-roughness layout); the thickness, curvature and occlusion textures are grey, linear (sRGB off).
- **Material.** Base Color from the base colour texture times `baseColorFactor`, Normal from the normal texture, Roughness from the roughness texture's G times `roughnessFactor`, Metallic 0, and Specular `specularFactor * F0 / 0.08`, where `F0 = ((ior - 1) / (ior + 1))^2` (0.5 for the default IOR of 1.5).
- **High tier.** Shading Model Subsurface Profile, with a Subsurface Profile asset per character: its Mean Free Path Color is `radius_mm` divided by its largest channel, and its Mean Free Path Distance the largest channel, converted to the profile's units (its World Unit Scale relates them to the scene's centimetres). The material's Opacity input masks the scattering: `subsurface.weight` scaled to taste (1 is the full profile). The thickness map (`texel * max_mm`, millimetres) can drive the profile's transmission tint through the ears and the nostrils, or the Opacity mask.
- **Medium tier.** Shading Model Preintegrated Skin, Unreal's cheap skin: Subsurface Color from the base colour (or `subsurface.color`), and Opacity from `1 - thickness` (thin parts glow more). For a custom shader, the curvature map gives the row of a pre-integrated diffusion lookup (Penner 2011): `k = (2 * texel - 1) * max_per_m`, the mean curvature in 1 / metres.
- **Head tile and eyes.** The head, ear and eye materials carry `KHR_texture_transform` on their textures; set the same offset and scale on the texture coordinates in the material (or keep the importer's transform, if it reads the extension).

### Unity HDRP (high: Diffusion Profile; medium: thickness and curvature in a Shader Graph)

- **Textures.** glTF normal maps are +Y, as Unity expects: import them as Normal maps, no flip. Unity's Lit mask map packs metallic (R), ambient occlusion (G), a detail mask (B) and smoothness (A): build it from 0, the occlusion (or 1), 0 and `1 - roughness` from the roughness texture's G channel.
- **High tier.** HDRP Lit with Material Type Subsurface Scattering and Transmission on, and a Diffusion Profile asset per character: its Scattering Distance is `radius_mm` divided by its largest channel, with the multiplier the largest channel (the profile shows its Max Radius in millimetres; World Scale 1 for scenes in metres), and its Index of Refraction `subsurface.ior`. Subsurface Mask is `subsurface.weight`, and the Thickness Map is the thickness texture with its remapping from 0 to `max_mm` millimetres, which drives the transmission through thin parts. Add the profile to the HDRP global settings' list of diffusion profiles.
- **Medium tier.** HDRP's screen-space scattering costs a pass; for the medium tier a Shader Graph with the pre-integrated lookup by curvature and a wrap of the thickness for transmission keeps the warm terminator and the glow of the ears at the cost of a texture lookup (in URP too).
- **Head tile and eyes.** Apply the `KHR_texture_transform` of the head, ear and eye materials as the tiling and offset of their textures.

### VRM apps (low)

The low tier is a VRM 1.0 file (or VRM 0.x with `--vrm-version 0`) with MToon materials: the shade colour and the shading shift and toony factors of the skin ramp imitate the warmth of the master's subsurface scattering (`opensculptboy.export.skin_ramp`), and the shade texture is the base colour darkened by the master's ambient occlusion (by 80 % of it), so that the creases and the inside of the mouth fall into shade with the light. VRM 1.0 apps (Warudo, VMagicMirror, three-vrm) and 0.x apps (VSeeFace, 3tene) read both without further setup. A glTF file of the low tier carries the occlusion as its `occlusionTexture` instead.

## Pre-rendered pictures

`opensculptboy portraits` (`opensculptboy.render.sheet`) renders the character's head and shoulders for chat avatars and profile pictures: `portrait.png` (the neutral face at twice the size), one picture per expression (`neutral`, `smile`, `surprised`, `sad`, `angry`: the mixes of anny's ARKit facial actions that the VRM presets `neutral`, `happy`, `surprised`, `sad` and `angry` use, from `data/vrm/expressions.json`) and `expressions.png`, the five side by side with their names. The pictures use the flat toon shading of the VRM thumbnail (`opensculptboy.render.flat.shaded_png`) on the card's level-0 mesh with its level-0 offsets, in the median colour of the card's base colour texture; they take about a second each, without a GPU or a path tracer.

For offline renders, the master tier's GLB file imports into Blender with its subsurface skin set from the extras by the helper of the skin material, and Cycles renders it with random-walk subsurface scattering:

```bash
blender --python src/opensculptboy/export/blender_skin.py -- character_master.glb   # import the file and set its skin
```

## Limits

- The bake needs a master with its detail layer; a card without one (level 0 only) has nothing coarser to bake for, and every tier is the master.
- The rays of the bake reach 6 mm each way from the tier's mesh. A tier whose mesh lies further from the master (a level 0 far from a very detailed master, between fingers or at the tip of a pointed ear) leaves those texels to the nearest reached value.
- The occlusion is the master's own, of the rest pose with the arms down; it does not follow a pose, and the arms darken the sides of the torso a little.
- The tier presets are fixed; a custom tier is a `Tier` of its own (`dataclasses.replace(TIERS["medium"], atlas_size=2048)`), passed to `bake_tiers` and `export_tier` alike.
- The engine notes describe the mapping of the fields; no engine-side importer is provided.
- The portraits are flat toon pictures without the textures.

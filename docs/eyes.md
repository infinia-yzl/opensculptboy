# Eyes

Anny's eyeballs are spheres of 72 vertices, and its skin goes on behind them into the eye socket. In an export the eyeballs read as faceted balls without an iris (or with a baked iris that is grey and has a wide pupil, as a game's eye textures often are, coloured and sized by the game's shader), the cornea gives no highlight, and the skin of the socket that shows between the lids and the eyeball at the inner corner of the eye reads as a dark hole. A card whose `material` section holds `eyes` (`opensculptboy.material.EyeMaterial`, `opensculptboy.eyes`) replaces them in the GLB and VRM exports with:

- a smooth eyeball fitted to anny's, with the iris recessed a little behind the limbus;
- a separate cornea shell over it, clear and refracting, with the corneal bulge over the iris (GLB files);
- an eye texture of its own: a procedural iris, pupil, limbal ring and sclera, or the baked eyes of a fitted card brought to the card's iris colour and sizes;
- the caruncle, a low pink mound over the pocket in the inner corner of the eye, and the skin of the socket drawn pink and wet.

Cards without the section export exactly as before (`TestGltfReference` in `test/test_gltf_export.py` holds the files to their recorded fingerprints).

```bash
opensculptboy export character.glb --character character.json --eyes                 # the card's eyes, else the defaults
opensculptboy export character.glb --character character.json --eyes eyes.json       # eyes.json laid over them
opensculptboy export character.vrm --character character.json --eyes --author "Your name"
blender --python src/opensculptboy/export/blender_skin.py -- character.glb           # the cornea's refraction in Cycles
```

```python
from opensculptboy import Character, export_glb

card = Character(material={"eyes": {"iris_color": [0.05, 0.12, 0.2], "pupil_mm": 3.5}})
export_glb("character.glb", card)
```

![Close-ups of the default character's left eye, before and after, rendered from the exported GLB files by a small ray tracer with refraction through the cornea](figures/eyes_render_check.jpg)

## The card

Every key of `material.eyes` is optional; colours are linear RGB and sizes millimetres. A `material` section that holds nothing but `eyes` keeps the exporters' own skin, so a card can change its eyes alone. The section belongs to character schema 3 and reads back from the files (`read_character`).

| Key | Default | Meaning |
|---|---|---|
| `source` | `"auto"` | `"auto"`: the baked eye texture when the file has one (the detail layer's `eyes`), else the procedural eye; `"procedural"`; `"baked"` (an error without a baked texture) |
| `base_color` | none | an eye texture of its own, in the layout below, as a path relative to the card |
| `iris_color` | `[0.20, 0.10, 0.04]` | the outer iris, a medium brown |
| `iris_color_inner` | a lighter amber of it | the zone inside the collarette |
| `iris_tint` | 0 | 0 to 1: how far a baked iris turns to `iris_color`, keeping its pattern |
| `iris_mm` | 11.7 | the iris (limbus) diameter |
| `pupil_mm` | 3.5 | the pupil diameter, 3 to 4 mm in neutral light |
| `limbal_ring`, `limbal_ring_mm` | 0.6, 0.6 | the darkening of the ring around the iris and its width |
| `sclera_color` | `[0.80, 0.74, 0.68]` | a slightly warm white |
| `veins` | 0.15 | 0 to 1: faint red veins at the periphery of the sclera |
| `seed` | 0 | the procedural pattern |
| `sclera_roughness` | 0.45 | the eyeball under the cornea |
| `cornea` | true | GLB files: the refracting cornea shell over a recessed iris; false: one glossy surface with the iris on its bulge, as the VRM files have |
| `cornea_ior`, `cornea_roughness` | 1.376, 0.02 | the cornea |
| `bulge` | 1 | the corneal bulge, a share of its anatomical height (the export lowers it further to clear the lids) |
| `caruncle` | true | the caruncle and the pink skin of the socket |
| `caruncle_color` | `[0.42, 0.15, 0.13]` | both of them |
| `highlight` | 1 | VRM files: the strength of the matcap highlight |

The sizes are millimetres, not shares of the eyeball: the default character's eyeball is 32 mm across (anny's template 30 mm), larger than a human's 24 mm, while its lids open about as wide as a human's, so an iris scaled with the eyeball would fill the whole opening.

## Geometry

`opensculptboy.eyes.geometry` builds the parts in each eye's frame: the centre and radius of the least-squares sphere through the eyeball's vertices, the gaze along -Y of Anny's rest frame (where the figure looks), up along +Z. The parts are surfaces of revolution about the gaze, 48 segments around in GLB files and 32 in VRM files:

- the eyeball: the sclera sphere to the back pole, and inside the limbus a short wall down to the iris, a shallow dome (0.3 mm) whose rim lies 0.45 mm behind the plane of the limbus;
- the cornea: a shell 0.15 mm over the sclera, out to 100 degrees from the gaze under the lids, with the corneal bulge, a sphere of 7.8 mm radius through the limbus (its apex 1.7 mm over the sclera on the default character);
- the single-surface eyeball (VRM files, and `cornea: false`): the sclera with the same bulge, the iris drawn on it.

Anny's lids rest on the sphere of its eyeball. Before it builds the parts, the export checks the outward-facing skin within 80 degrees of the gaze (`bulge_clearance`): the eyeball shrinks by up to 3 % and the bulge drops until the lids stay 0.08 mm outside the eye, with 1 % of those vertices (at least 2) allowed closer, since a fitted or hand-made mesh may touch the eyeball here and there; skin inside the fitted sphere already met anny's eyeball and is left aside. Both eyes take the smaller of their two values. The default character keeps the full bulge.

The caruncle (`geometry.caruncle`) comes from a front view of the corner: a depth raster along the gaze (0.01 eyeball radii a texel) of the skin and the eyeball's sphere. The opening is what that view sees of the sphere, or of skin within 1.12 eyeball radii of its centre, connected to the gaze axis; the pocket is the skin of the opening on the side of the nose that lies at least 0.03 eyeball radii below the level of the lids around it (the grey closing of the view over 0.25 eyeball radii). The mound covers an ellipse of the pocket's spread, rises 0.025 eyeball radii over the plane through its rim, stays under the level of the lids and well under the skin outside the pocket, and sinks at its rim under what lies there, so that its edge does not show. On the default character it hides about 80 % of the pocket's texels of the front view (`test_caruncle_covers_the_pocket`). The skin of the socket (`geometry.socket`: within 1.35 eyeball radii, its normal facing the eyeball's centre) takes the caruncle's pink material.

| Default character, level 0 | Vertices | Triangles |
|---|---|---|
| anny's mesh (GLB file without eyes) | 14,898 | 27,420 |
| GLB file with eyes (eyeballs, corneas, caruncles; anny's 280 eyeball triangles gone) | 19,114 | 35,356 |
| VRM file with eyes (single-surface eyeballs, caruncles) | | 30,268, within the 32,000 of VRM 0.x |

## On the mesh

`opensculptboy.eyes.mesh.eyed_mesh` takes the mesh of a file (anny's own, or a level of the detail layer with the card's offsets) and returns it without anny's eyeball triangles, the parts appended as new vertices. Each part is a linear map of some of the mesh's own vertices: the eyeball and the cornea follow the least-squares affine map of their eyeball's vertices, which is exact for the turns, moves and scales that anny's eye bones and eye targets give an eyeball, and the caruncle the affine map of the skin around the inner corner. So the same maps carry the rest mesh, every morph target, and the T-pose bind mesh and its targets of a VRM file over to the parts. The eyeball and the cornea take the mean skin weights of their eyeball's vertices and turn rigidly with the eye bone; the caruncle blends the weights of its four nearest skin vertices. The look-at of a VRM file, its eye bones and its expressions are unchanged (the lid-only `eyeLook*` targets move no eye part), and the parts' vertices name the nearest Anny vertex of what they follow in `_ANNY_VERTEX`. Anny's own `eyeLook*` targets turn its eyeball with 0.13 mm of other movement; the eye parts follow the best affine map of it.

## On the sculpt topology

A model of the `sculpt` topology has a caruncle and a fold of the conjunctiva of its own (the eye-corner patch, docs/sculpt-topology.md). There `eyed_mesh` adds no caruncle part, and the patch's faces labelled `eye_corner/caruncle` and `eye_corner/conjunctiva` take the conjunctiva's material with the skin of the socket (`corner_triangles` finds them by their UVs, which keep the region of the removed quads in the atlas, so the detail levels find them too). On `anny` the eyes are as before.

## The eye texture

The eye texture (`opensculptboy.eyes.texture`) is 1024 by 512 texels, one square per eye, the left eye (+X) in the left half. Within its square, a point lies at its angle about the gaze (from +X toward up, so that the square shows the eye from the front) and at a distance from the centre that grows with its distance from the gaze axis: the front half of the eyeball fills 90 % of the square's half-width, the back half the thin ring outside it. The iris disc and the corneal bulge take the texels of their distance from the axis, so the iris looks the same size from the front on every kind of eyeball. The layout has no seam.

The procedural eye draws the pupil (with a soft edge of 0.08 mm), an iris of radial fibres with a collarette, its lighter inner zone and a darker outer band, the limbal ring, and a warm sclera, greyer toward the back, with faint veins at its periphery.

A fitted card's baked eye texture (`detail.textures.eyes`, on anny's atlas: `opensculptboy.texture.eyes`) is carried over through the eyeball of the exported mesh: each eyeball triangle of the mesh is drawn into the layout by the directions of its corners from the eye's centre, with its UVs on the atlas, and the baked texture is sampled there. Then, per eye (`fit_baked_iris`), the darkest 1 % of the front gives the pupil's centre; the median luminance of rings about it gives the pupil's edge (half way from the pupil to the iris) and the limbus (half way from the iris to the sclera); the iris moves onto the gaze and stretches radially, piecewise linearly, to `pupil_mm` and `iris_mm`, blending back into the sclera beyond it; the iris turns toward `iris_color` by `iris_tint`, keeping its luminance pattern; and the pupil is drawn dark at its size. An eye whose texture shows no dark pupil in a darker iris takes the other eye's mirror image, and when neither does the procedural eye replaces both. The scene extras record what each side took (`eyes.sides`).

## glTF

| Material | What | Extensions |
|---|---|---|
| `eyes` | the eye texture; under a cornea a diffuse surface with no reflection of its own, since it lies in the eye's own medium (a rough reflection there reads as a milky veil over the iris) | under a cornea, `KHR_materials_specular` (0); without a cornea, `KHR_materials_clearcoat` (1, the cornea's roughness) and `KHR_materials_ior` (1.376) for the wet highlight |
| `cornea` | white, roughness 0.02, OPAQUE | `KHR_materials_transmission` (1), `KHR_materials_volume` (3 mm, the depth of the anterior chamber), `KHR_materials_ior` (1.376) |
| `conjunctiva` | the caruncle and the skin of the socket, `caruncle_color`, roughness 0.5 | `KHR_materials_clearcoat` (0.8, roughness 0.08) |

All of them are ratified Khronos extensions, and Blender's importer reads them. The transmission extension asks for alphaMode OPAQUE on a transmissive material that does not use alpha as coverage, and says that a renderer should at least show the opaque objects through it: three.js, Babylon.js, Filament and Blender refract the iris behind the cornea. A reader without the extension draws the cornea white over the iris, so for such readers export with `"cornea": false`, which gives one glossy surface with a clear coat instead. Each eye material's extras hold `opensculptboy_eye` (format `opensculptboy/eye@1`): its `role` (`eyeball`, `cornea` or `conjunctiva`), the iris and pupil sizes, the cornea's IOR and roughness, the sclera's roughness, whether a cornea covers the eyeball (`cornea`) and the bulge kept. The scene extras record the eyes under `opensculptboy.eyes` (the radius of each eyeball, the bulge, what the texture came from).

**Blender.** `src/opensculptboy/export/blender_skin.py`, which also sets the skin, reads those extras (`apply_eye_materials()`): the cornea becomes a clear refracting surface (Transmission Weight 1 at its IOR and roughness) whose shadow rays pass through a Transparent BSDF (Light Path, Is Shadow Ray), since a refracting surface blocks a path tracer's direct light and would leave the iris in its shadow; EEVEE's refraction is turned on for the material where the Blender version has the setting. The cornea takes no coat, sheen or subsurface, and the importer's Volume Absorption node of `KHR_materials_volume` (density 0) leaves its material output, since the open shell encloses no volume. The eyeball takes the sclera's roughness, no subsurface scattering (a millimetre of it blurs the iris and greys the pupil) and, under a cornea, Specular IOR Level 0: its own rough reflection at the importer's IOR of 1.5 laid a milky veil over the whole eye. So the cornea gives the one sharp highlight. The conjunctiva takes subsurface scattering (2 mm). Everything inside the cornea is one medium of its IOR, from the cornea to the iris, as a single interface renders it.

## VRM

MToon has no refraction, so the VRM files take the single-surface eyeball with the iris on its bulge: an MToon material whose lit and shade textures are the eye texture, with a soft ramp, and a matcap (`texture.matcap_pixels`: black, with a soft window light above and to the left of the view and a faint fill below) for the wet highlight, which stays toward the light of the view as the eye turns (`matcapTexture` in VRM 1.0, `_SphereAdd` in VRM 0.x; `highlight` scales it). The caruncle and the socket take a pink MToon material with half the matcap. The eye texture and the matcap are the files' images `eye_base_color` and `eye_matcap`; the eyes add 2 materials (3 with the skin, within the 8 of VRM 0.x and the 16 of VRM 1.0) and, on level 0, 3,128 triangles in place of anny's 280 eyeball triangles.

## Checks and limits

`test/test_eyes.py` checks the card and its validation, the surfaces (outward normals, the recessed iris, the bulge's height, the lids lowering it), the layout's round trip, the procedural iris's sizes, the eyed mesh (exact rest positions, an eye turn carried rigidly to the parts, one set of weights per eyeball on its eye bone), the caruncle over the pocket, the synthetic baked eyes brought onto the gaze at the card's sizes and colour, the GLB file (materials, extensions, the eyes against the file's own morph targets), level 1 of a card with offsets and a baked texture, the VRM files of both versions within strict budgets and the structural VRM checks, the Blender helper's reading and the command's `--eyes`. The render check above comes from a small Embree ray tracer that reads the exported files (not part of the repository): Lambert and Blinn-Phong shading with shadows and ambient occlusion, a studio environment with a soft window light, and Snell refraction with Schlick's Fresnel reflection at the cornea.

What the synthetic checks cannot tell, and a real fitted character should show:

- whether the baked iris of the real eye texture is found (its pupil darker than its iris, its iris darker than its sclera), and how its pattern looks once stretched to the card's sizes and tinted;
- whether the caruncle sits well in the real lids' inner corner, at rest and in a blink;
- whether the real lids leave room for the full corneal bulge (the scene extras record the share kept) and whether the iris size suits the real eye opening;
- the look in Blender's Cycles and EEVEE with the helper, and in a VRM app with the matcap highlight.

# Eyes

Anny's eyeballs are spheres of 72 vertices, and its skin goes on behind them into the eye socket. In an export they read as faceted balls without an iris, the cornea gives no highlight, and the socket's skin at the inner corner reads as a dark hole. A card whose `material` section holds `eyes` (`opensculptboy.eyes`) replaces them in the GLB and VRM exports with:

- a smooth eyeball, with the iris recessed a little behind the limbus;
- a separate, clear, refracting cornea over it (GLB files with procedural eyes); baked eyes take one surface with the corneal bulge under a clear coat, as the reference draws them;
- an eye texture of its own: a procedural iris, pupil, limbal ring and sclera, or the baked eyes of a fitted card;
- the caruncle, a low pink mound over the pocket in the inner corner, and the socket's skin drawn pink and wet.

Cards without the section export exactly as before.

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

![Close-ups of the default character's left eye, before and after, rendered from the exported GLB files with refraction through the cornea](figures/eyes_render_check.jpg)

## The card

Every key of `material.eyes` is optional; colours are linear RGB and sizes millimetres. A `material` section that holds nothing but `eyes` keeps the exporters' own skin, so a card can change its eyes alone.

| Key | Default | Meaning |
|---|---|---|
| `source` | `"auto"` | `"auto"`: the baked eye texture when the file has one (the detail layer's `eyes`), else the procedural eye; `"procedural"`; `"baked"` (an error without a baked texture) |
| `base_color` | none | an eye texture of its own, in the layout below, as a path relative to the card |
| `iris_color` | `[0.20, 0.10, 0.04]` | the outer iris, a medium brown |
| `iris_color_inner` | a lighter amber of it | the zone inside the collarette |
| `iris_tint` | 0 | 0 to 1: how far a baked iris turns to `iris_color`, keeping its pattern |
| `iris_mm` | `"auto"` | the iris (limbus) diameter: millimetres, `"baked"` (the size found in the file's baked eye texture; an error without one) or `"auto"` (that size when the eyes come from a bake and an iris is found there, else 11.7) |
| `pupil_mm` | `"auto"` | the pupil diameter, likewise (3.5 when not baked: 3 to 4 mm in neutral light) |
| `limbal_ring`, `limbal_ring_mm` | 0.6, 0.6 | the darkening of the ring around the iris and its width |
| `sclera_color` | `[0.80, 0.74, 0.68]` | a slightly warm white |
| `veins` | 0.15 | 0 to 1: faint red veins at the periphery of the sclera |
| `seed` | 0 | the procedural pattern |
| `sclera_roughness` | 0.45 | the eyeball's own surface (under the clear coat of a single surface) |
| `cornea` | `"auto"` | GLB files: true, the refracting cornea shell over a recessed iris; false, one surface with a clear coat and the iris on its bulge, as the VRM files have; `"auto"`, false when the eye texture comes from a bake and true otherwise |
| `cornea_ior`, `cornea_roughness` | `"auto"` | the cornea (on a single surface, the IOR of the eyeball's own surface and the roughness of its clear coat): numbers, or `"auto"`, the values of the card's `reference_eyes` (below), else 1.376 and 0.02 |
| `cornea_radius_mm` | `"auto"` | the radius of curvature of the cornea's front, which sets the bulge's height and the size of the highlights: millimetres, or `"auto"`, the value of the card's `reference_eyes`, else 7.8 |
| `bulge` | 1 | the corneal bulge, a share of its anatomical height (the export lowers it further to clear the lids) |
| `caruncle` | true | the caruncle and the pink skin of the socket |
| `caruncle_color` | `[0.42, 0.15, 0.13]` | both of them |
| `highlight` | 1 | VRM files: the strength of the matcap highlight |

A fitted card also records its reference's cornea in `material.reference_eyes`, which `opensculptboy fit --textures` and `opensculptboy bake` write when the reference has eye parts: the radius of curvature of the outer eye surface's front (`cornea_radius_mm`), its roughness and its IOR. The eyes' `"auto"` values take them, and a number in `material.eyes` or in the file of `--eyes` overrides each one. The section turns no eyes on by itself.

The sizes are millimetres, because Anny's eyeball is larger than its lid opening suggests (about 32 mm across on the default character): an iris scaled with the eyeball would fill the whole opening.

## Geometry

`opensculptboy.eyes.geometry` builds the parts in each eye's frame (the least-squares sphere of the eyeball, the gaze along -Y) as surfaces of revolution: the eyeball with the iris as a shallow dome behind the limbus; the cornea, a shell over it with the corneal bulge, a sphere of `cornea_radius_mm` through the limbus; and the single-surface eyeball of VRM files and `cornea: false`. Before it builds them, the export checks the lids (`bulge_clearance`): the eyeball shrinks by up to 3 % and the bulge drops until the lids stay clear of the eye, alike in both eyes. The caruncle (`geometry.caruncle`) covers the pocket of the inner corner, found on a front view of the corner, and sinks at its rim under the lids, so that its edge does not show.

`opensculptboy.eyes.mesh.eyed_mesh` takes the mesh of a file (Anny's own, or a detail level) and returns it without Anny's eyeball triangles, the parts appended as new vertices. Each part follows a linear map of some of the mesh's own vertices, so the same maps carry the rest mesh, every morph target and the VRM T-pose over to the parts. The eyeball and the cornea turn rigidly with the eye bone, and the look-at, eye bones and expressions of a VRM file are unchanged. On level 0, a GLB file with eyes has 35,356 triangles and a VRM file 30,268, within the 32,000 of VRM 0.x.

On the [sculpt topology](sculpt-topology.md), the eye-corner patch already holds a caruncle and a fold of the conjunctiva, so `eyed_mesh` adds no caruncle and gives the patch's faces inside the lid opening the conjunctiva's material. A file with a baked skin (a fitted card) keeps the whole corner in the skin's material, so that the corner shows the reference's own caruncle and lids.

## The eye texture

The eye texture (`opensculptboy.eyes.texture`) holds one square per eye, the left eye (+X) in the left half; within its square, a point lies at its angle about the gaze and at a distance from the centre that grows with its distance from the gaze axis, so the layout has no seam. The procedural eye draws the pupil, an iris of radial fibres with a collarette, the limbal ring and a warm sclera with faint veins.

A fitted card's baked eyes are carried over from Anny's atlas through the eyeball of the exported mesh. Per eye (`fit_baked_iris`), the darkest part of the front gives the pupil's centre, and rings about it give the pupil's edge and the limbus; a pale outer band of the iris, which game textures often draw, counts as iris. Sizes of `"auto"` or `"baked"` take the diameters found there. The iris then moves onto the gaze and stretches to the card's sizes, and turns toward `iris_color` by `iris_tint`; the pupil keeps the bake's own texels. An eye with no pupil to find takes the other eye's mirror image. The bake also writes the eyeballs' normals (`eyes_normal`), which carry the shading of a game's eye shader, such as a concave iris; the glTF export gives them to the eyeball as its normal texture. The bake writes the eye texture at 2048 by 1024 texels (`--eye-size`).

## glTF

| Material | What | Extensions |
|---|---|---|
| `eyes` | the eye texture, and for baked eyes the baked normals (`eye_normal`); under a cornea a diffuse surface with no reflection of its own, since it lies in the eye's own medium (a rough reflection there reads as a milky veil over the iris) | under a cornea, `KHR_materials_specular` (0); without a cornea, `KHR_materials_clearcoat` (1, the cornea's roughness, at the clear coat's fixed IOR of 1.5) and `KHR_materials_ior` (`cornea_ior`) for the wet highlight |
| `cornea` | white, roughness 0.02, OPAQUE | `KHR_materials_transmission` (1), `KHR_materials_volume` (3 mm, the depth of the anterior chamber), `KHR_materials_ior` (1.376) |
| `conjunctiva` | the caruncle and the skin of the socket, `caruncle_color`, roughness 0.5 | `KHR_materials_clearcoat` (0.8, roughness 0.08) |

All of them are ratified Khronos extensions, and Blender's importer reads them. A reader without `KHR_materials_transmission` draws the cornea white over the iris; for such readers, export with `"cornea": false`. Each eye material's extras hold `opensculptboy_eye` (its role, the sizes, the cornea's values), and the scene extras record the eyes under `opensculptboy.eyes`.

In Blender, `blender_skin.py` (`apply_eye_materials()`) makes the cornea a clear refracting surface whose shadow rays pass through, so that the iris stays lit, and leaves the eyeball under it without reflection or subsurface scattering, so that the cornea gives the one sharp highlight.

## VRM

MToon has no refraction, so VRM files take the single-surface eyeball with the iris on its bulge: an MToon material with the eye texture and a matcap for the wet highlight (`matcapTexture` in VRM 1.0, `_SphereAdd` in VRM 0.x; `highlight` scales it). The caruncle and the socket take a pink MToon material. The eyes add 2 materials.

## Checks and limits

`test/test_eyes.py` checks the card, the surfaces, the layout, the procedural iris, the eyed mesh, the caruncle, the baked eyes and their measured sizes, the GLB file, a detail level, the VRM files of both versions within strict budgets, the Blender helper and `--eyes`. The render check above comes from a small ray tracer with refraction at the cornea, outside the repository.

- On level 0, Anny's 72-vertex eyeballs show an octagonal limbus in a close-up without `--eyes`.
- The look in a VRM app depends on the app's matcap support; VSeeFace, Warudo and VMagicMirror are untested.

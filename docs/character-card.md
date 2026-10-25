# Character card

A character card is a JSON file that describes one character: its sliders, its pose and, for a fitted character, its detail layer and materials. `opensculptboy character` writes a card with the default settings, the exporters read cards, and every `.glb` and `.vrm` file carries its card in the scene extras, so that `read_character` reads it back. In Python, the card is `opensculptboy.Character`.

```json
{
  "schema": 1,
  "name": "ada",
  "rig": "anny",
  "topology": "anny",
  "phenotypes": "default",
  "phenotype": {"muscle": 0.7, "weight": 0.6},
  "local_changes": {},
  "face_shapes": {"nose-scale-horiz": 0.3},
  "facial_actions": {"mouthSmileLeft": 0.4},
  "pose": {}
}
```

Every field is optional, and a missing name keeps its default. Paths in a card are relative to the card.

| Field | Holds |
|---|---|
| `schema` | 1 for the fields above, 2 with `detail`, 3 with `material`, 4 with the wardrobe fields. A card is written with the lowest schema that holds its fields, so that older cards stay byte for byte the same. |
| `name` | the character's name |
| `rig` | `anny` (104 bones), or `anny-mouth` with a jaw bone and a tongue chain ([Mouth](mouth.md)) |
| `topology` | `anny`, or `sculpt` and `sculpt-quads` ([Sculpt topology](sculpt-topology.md)) |
| `phenotypes` | `default` (six sliders) or `all` (with race, cup size and firmness) |
| `phenotype` | the body sliders, 0 to 1 (default 0.5); the `age` slider appears as *Form* in the viewer |
| `local_changes` | MakeHuman's local changes, -1 to 1 |
| `face_shapes` | the 103 face shapes and 10 detail shapes ([Body model](body-model.md)) |
| `facial_actions` | the 52 ARKit facial actions, 0 to 1 |
| `pose` | a still pose: `"bones"` maps bone names to unit quaternions (x, y, z, w) relative to the parent, and `"root"` moves the root bone (metres); an empty pose is the rest pose |
| `detail` | the detail layer of a fitted character |
| `material` | the skin, eyes, teeth, inner mouth and tiers |
| `outfits`, `active_outfit` | the outfits (`{"id", "label", "pieces": [{"piece", "file", "enabled"}]}`, with optional `description`, `tags`, `source` and per-piece `overrides`, `hides` and `springs`) and the one worn ([Wearables](wearables.md)) |
| `wearables` | loose pieces: `{"id", "file", "enabled"}`, each file a wearable package |
| `wearable_overrides` | single pieces switched on or off: `{"<outfit>/<piece>": {"enabled": false}}` |

## `detail`

```json
"detail": {
  "levels": [
    {"subdiv": 0, "offsets": "character.detail0.safetensors", "error_mm": 0.26},
    {"subdiv": 1, "offsets": "character.detail1.safetensors", "error_mm": 0.04}
  ],
  "textures": {"base_color": "character.base_color.png", "normal": "character.normal.png", "subdiv": 1},
  "fit": {"placement": {"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "translation": [0, 0, 0]}, "errors_mm": {}, "seconds": {}}
}
```

`levels` lists one offsets file per subdivision level: one float32 vector per vertex of the full level, in Anny's rest frame. `textures` names the baked maps and the level they were baked on, and `fit` records the placement and the measured errors. `mouth` names the jaw's share of each skin vertex, taken from the reference. [Fit to a reference](reference-fit.md) explains the levels.

## `material`

```json
"material": {
  "base_color": "character.base_color.png", "normal": "character.normal.png", "roughness": "character.roughness.png",
  "thickness": "character.thickness.png", "thickness_mm": 30.0, "curvature": "character.curvature.png", "curvature_per_m": 100.0,
  "occlusion": "character.occlusion.png",
  "roughness_factor": 1.0, "specular": 1.0, "specular_color": [1, 1, 1], "ior": 1.5, "normal_scale": 1.0,
  "subsurface": {"weight": 0.1, "radius_mm": [5.4, 2.1, 1.2], "ior": 1.4, "anisotropy": 0.0, "method": "random_walk_skin"},
  "head": {"base_color": "character.head_base_color.png", "normal": "character.head_normal.png", "roughness": "character.head_roughness.png"},
  "ears": {"uvs": "character.ear_uvs.safetensors", "base_color": "character.ears_base_color.png", "normal": "character.ears_normal.png"},
  "eyes": {"iris_color": [0.2, 0.1, 0.04], "pupil_mm": "auto"},
  "reference_eyes": {"cornea_radius_mm": 9.25, "cornea_roughness": 0.05, "cornea_ior": 1.5, "gaze": {"L": [0.05, -0.998, 0.0], "R": [-0.05, -0.998, 0.0]}},
  "lashes": {"source": "reference", "mesh": "character.lashes.safetensors", "base_color": "character.lashes.png"},
  "teeth": {"source": "auto"},
  "mouth": {"mesh": "character.mouth.safetensors", "base_color": "character.mouth_base_color.png", "poses": "character.mouth_poses.safetensors"},
  "tiers": {"low": {"subdiv": 0, "region": "full", "normal": "character.low.normal.png"}}
}
```

| Key | Holds | Described in |
|---|---|---|
| `base_color`, `normal`, `roughness` and their factors | the skin's maps on Anny's UV atlas (sRGB colour; tangent-space normals in MikkTSpace frames, +Y up) | [Fit to a reference](reference-fit.md#textures) |
| `thickness`, `curvature`, `occlusion` | grey maps for real-time skin shading and the toon tier | [Fit to a reference](reference-fit.md#textures), [Budget tiers](budget-tiers.md) |
| `specular`, `specular_color`, `ior` | the surface's reflection | |
| `subsurface` | the scattering: weight 0 to 1, mean free path per RGB channel in mm, IOR, anisotropy, and one of Blender's methods (`random_walk_skin`, `random_walk`, `burley`) | [Fit to a reference](reference-fit.md#the-skin-material-in-the-exports) |
| `head`, `ears` | the head tile's and the ear tile's maps; `ears.uvs` holds the ears' own UVs | [Fit to a reference](reference-fit.md#textures) |
| `eyes` | the eyes of `--eyes`: iris, pupil, sclera, cornea and caruncle; `false` keeps anny's eyeballs | [Eyes](eyes.md) |
| `reference_eyes` | the cornea's values and the gaze measured on the reference, used where `eyes` says `"auto"`, and the file of the reference's blinks (`poses`); a fitted card's eyes are on without `eyes` | [Eyes](eyes.md) |
| `lashes` | the lashes that come with the eyes: the reference's or generic ones; `false` leaves them out | [Eyes](eyes.md#lashes) |
| `teeth` | the teeth of `--teeth`: the reference's or a generic set | [Teeth](teeth.md) |
| `mouth` | the reference's inner mouth and its mouth poses | [Mouth](mouth.md) |
| `tiers` | per rendering-budget tier, the normal maps baked for that tier's mesh | [Budget tiers](budget-tiers.md) |

A card without `material` gets the exporters' own plain skin. A section that holds only `eyes`, `reference_eyes`, `lashes`, `teeth` or `mouth` keeps that plain skin and changes those parts alone. `fit --textures` and `opensculptboy bake` write the section of a fitted character, and `--material skin.json` lays further values over it.

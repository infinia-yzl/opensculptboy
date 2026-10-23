# Body model

OpenSculptBoy's characters are built on Anny's body model: a differentiable, parametric body mesh in PyTorch. This page covers the Python API of the `anny` package.

```python
import torch, anny, trimesh

model = anny.Anny(local_changes="default", facial_actions="all").to(dtype=torch.float32)
pose_parameters = torch.eye(4)[None, None].repeat(1, model.bone_count, 1, 1)  # one 4x4 per bone (model.bone_labels)
phenotype_kwargs = {key: 0.5 for key in model.phenotype_labels}              # shape sliders within [0, 1]
local_changes = {"stomach-pregnant-incr": 1.0}                                # local changes within [-1, 1] (model.local_change_labels)
facial_actions = {"jawOpen": 0.8, "mouthSmileLeft": 0.4}                      # within [0, 1] (model.facial_action_labels)
output = model(pose_parameters=pose_parameters, phenotype_kwargs=phenotype_kwargs,
               local_changes_kwargs=local_changes, facial_actions=facial_actions)
trimesh.Trimesh(vertices=output["vertices"].squeeze(dim=0).numpy(), faces=model.faces).export("anny_output.ply")
```

The model accepts dictionaries and stacked tensors. The phenotype sliders are gender, age, muscle, weight, height and proportions, plus race, cup size and firmness with `phenotypes="all"`.

## Poses, soft tissue and hair

All of these follow the phenotype sliders:
- `anny.poses` holds 50 poses and 7 animation clips for the `anny` rig, standing on the floor, with a stool for the seated poses (`anny.poses.names()` lists them).
- `anny.correctives.SoftTissueCorrectives(model)` adds corrective shapes at the shoulders, the elbows, the hips and the knees.
- `anny.hair.styles` holds 25 procedural hairstyles, from buzz cuts to long hair and buns, with length, curl, volume, density and fade parameters. `anny.hair.StrandBinding` ties the strands to the skin, so a groom follows every body, and `anny.hair.dynamics.HairSim` simulates the hair.
- `anny.utils.subdivision` holds Catmull-Clark subdivision as sparse linear operators.

```python
import anny, anny.poses
from anny.correctives import SoftTissueCorrectives

model = anny.Anny()
phenotype = {"weight": 0.3, "muscle": 0.7}
posed = anny.poses.pose_parameters(model, "seated", phenotype_kwargs=phenotype)
output = model(pose_parameters=posed["pose_parameters"], phenotype_kwargs=phenotype)
output = SoftTissueCorrectives(model)(output)  # corrected output["vertices"]
```

## Face and head shapes

`anny.Anny(face_shapes="all")` adds 103 named, symmetric shapes of the head and the face from the MakeHuman targets, and 10 detail shapes from the 3D faces of ICT-FaceKit. The shapes scale with the size of the head. `FaceShapeDistribution` draws random faces around anny's own face, from fits to the ICT-FaceKit identity space, and `CraniofacialMeasurements` measures the head mesh:

```python
import anny, torch
from anny.faces.distribution import FaceShapeDistribution
from anny.faces.measurements import CraniofacialMeasurements

model = anny.Anny(face_shapes="all")
phenotype = {"weight": torch.full((4,), 0.6), "gender": torch.tensor([0.0, 0.0, 1.0, 1.0])}
faces = FaceShapeDistribution(model).sample(phenotype)
output = model(phenotype_kwargs=phenotype, face_shape_kwargs=faces)
lengths = CraniofacialMeasurements(model)(output)["headlength"]  # mm
```

`src/anny/data/faces/SOURCES.md` lists the data behind the distribution and their licences.

## Rigs

`anny.Anny()` uses the compact `anny` rig with 104 bones: the main body, hand and head articulation of the MakeHuman rig, without its facial-expression, eye, tongue and zero-weight bones. `rig="makehuman"` gives the full 163-bone MakeHuman rig with the old blender orientation, for compatibility with old models or for the face, tongue and eye bones. OpenSculptBoy adds `anny-mouth`, the `anny` rig with MakeHuman's jaw and tongue bones ([Mouth](mouth.md)). The facial actions stay available as blend shapes with `facial_actions="all"` on every rig.

## Topologies

`anny.Anny()` uses the `anny` topology: a MakeHuman-derived full-body mesh with minor mesh edits, unattached vertices removed, and triangular faces, so every vertex belongs to a face and the mesh works directly with most mesh tools. `topology="anny-quads"` keeps the quad faces, `topology="anny-full"` keeps the unattached vertices and the unedited mesh, and `topology="makehuman"` gives the unedited quad MakeHuman mesh. The retopologies `smplx`, `smpl` and `soma` serve interoperability with those ecosystems (the SMPL and SMPL-X data is for non-commercial use only). OpenSculptBoy adds `sculpt`, anny's mesh with patches ([Sculpt topology](sculpt-topology.md)).

## Migration from legacy defaults

`anny.Anny()` uses the defaults above (`rig="anny"`, `topology="anny"`, `pose_parameterization="local-ref"`), and the deprecated `anny.create_fullbody_model(...)` factory preserves the legacy full-body defaults. If you just want the new defaults, call `anny.Anny()` directly. If you need the exact legacy behavior, either keep using `create_fullbody_model(...)` while migrating, or translate its arguments to `Anny(...)` as follows:

| Legacy `create_fullbody_model(...)` argument | Equivalent `anny.Anny(...)` argument |
|---|---|
| `rig="default"` (full 163-bone MakeHuman rig) | `rig="makehuman"` |
| `topology="default"` (with the default `triangulate_faces=False`) | `topology="anny-quads"` |
| `topology="default", triangulate_faces=True` | `topology="anny"` |
| `remove_unattached_vertices=False` | append `-full` to the topology string (e.g. `topology="anny-quads-full"`) |
| `pose_parameterization` (default `"local-bone"`) | `pose_parameterization="local-bone"`, passed explicitly (`Anny()` defaults to `"local-ref"`) |
| `all_phenotypes=True` | `phenotypes="all"` |
| `bone_orientation=...` | folded into the rig string via modifiers such as `-blender` or `-rootidentity` |

Pose parameters belong to their pose parameterization: pose data saved with the legacy `"local-bone"` default gives a different pose under the `"local-ref"` default. Either pass `pose_parameterization="local-bone"` explicitly for compatibility, or convert existing pose data with `anny.utils.pose.transfer_pose_parameters` (see the pose portability tutorial below).

## Caching

Anny parses the MakeHuman assets and caches the blend-shape data, so the first instantiation of a model can take a few minutes. The cache lives in `~/.cache/anny/`; set `ANNY_CACHE_DIR` to use another folder.

## Tutorials

The Anny tutorials (jupytext files in `tutorials/`) cover:
- [Shape parameterization](https://naver.github.io/anny/build/shape_parameterization.html)
- [Pose parameterization](https://naver.github.io/anny/build/pose_parameterization.html)
- [Portability of pose parameterizations](https://naver.github.io/anny/build/pose_transfer.html)
- [Texture coordinates](https://naver.github.io/anny/build/texture.html)
- [Alternative models](https://naver.github.io/anny/build/alternative_models.html)

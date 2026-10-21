# The sculpt topology (plan)

Status: workstream 1 (Core) landed on `sculpt/core`, 2026-10-21: the format, the loader, the ModelData transform, `Character` integration, the detail levels on mixed faces, the converter and the tests, with a test-only placeholder patch. The shipped file holds no patch yet, so `sculpt` equals `anny` until the eye-corner and ear-root patches land. Branch `feat/sculpt-topology`.

## Goal

`sculpt` is OpenSculptBoy's own version of anny's mesh. It is anny's mesh with patches where anny's surface cannot follow a detailed character well. The first two patches are the inner corner of the eye and the root of the outer ear. On a real fitted character (a game replica), the fit leaves a dark hole in the inner corner, because anny has a deep pocket of skin there that the replica lacks. It also leaves a crack where the ear meets the cheek, because the game's ear is a separate piece stitched to the head, while anny's ear and head share one coarse skin. Fixes in the fit have reduced both artefacts. A better base mesh removes them for every character.

Every part of anny keeps working on `sculpt`: the phenotypes, local changes, face shapes, facial actions, the rig and its skin weights, the craniofacial landmarks, the detail levels, the texture bake, the glTF and VRM exports and the reference fit. A card made on `anny` loads on `sculpt` with the same parameters, and a converter carries the detail offsets of a card between both meshes.

## Design

### Append-only patches

`sculpt` keeps every vertex of anny at its own index and in its own order. A patch removes some of anny's faces, appends new vertices after anny's last vertex, and adds new faces over the old and new vertices. Code that indexes anny's vertices (the eyeball vertices of the VRM tables, the base-mesh labels, the ear weights, the hair's anatomy, `masks.ear` work in the fit) therefore keeps working unchanged. A vertex of anny that no face uses any more stays in the vertex list as an unreferenced vertex: the exporters write only referenced vertices (the glTF body already writes the vertices of its faces), and the fit and the measures leave them out (`Topology.referenced`).

### New vertices are fixed blends of anny's vertices

Each new vertex is a convex combination of at most 8 anny vertices (its anchors), with non-negative weights that sum to 1: `p_new = sum_k w_k p_anchor_k`. `anny.models.model_transforms.apply_retopology` takes exactly such rows (with `check_weights=True`), so every blend shape, every facial action, every face shape, the skin weights and the landmarks carry over to the new vertices in one step, and they stay exact under every parameter setting. Kept vertices take the identity row. A convex combination can lie off anny's surface, inside the hull of its anchors, so a patch can bridge a pocket (a caruncle spanning the rim of the eye-corner pocket) or add a denser ring of vertices around the ear root.

### Quads and triangles

Patches are written on the quad mesh (`anny-quads`), since the detail levels subdivide it with Catmull-Clark. `sculpt-quads` is the patched quad mesh, and `sculpt` its triangulation, as `anny` is the triangulation of `anny-quads`. New faces are quads where possible; a triangle is allowed where a patch joins anny's faces (Catmull-Clark accepts mixed meshes, and `opensculptboy.detail.subdivision` must keep working on them).

### UVs

A patch keeps anny's texture atlas. Each new face takes its UVs from the same convex combination of its anchors' UVs, within one UV chart: a patch only replaces faces inside one chart and covers the same region of the atlas, so the bake, the head tile, the ear tile and the eye texture box keep their layout. A test checks that no new UV triangle flips and that the new faces cover the removed faces' region without overlap.

### Data file

One file, `src/opensculptboy/data/topology/sculpt.safetensors` with a JSON sidecar `sculpt.json`, holds the composed patches:

- `base`: `"anny-quads"` and a digest of anny's quad faces and template vertices (the build refuses another base);
- `removed_faces`: indices into anny's quad faces;
- `anchors` (M, 8) int64 and `weights` (M, 8) float64: the new vertices (unused slots: anchor 0, weight 0);
- `faces`: the new faces as quads (a triangle repeats its last index), over vertex indices where index N + m is new vertex m;
- `uv_faces` and `uvs`: the new faces' UV indices and the new UVs, appended to anny's UVs;
- `labels`: per new vertex, its patch (`"eye_corner"`, `"ear_root"`), and per new face the same;
- the version and, per patch, the script and options that built it.

Authoring scripts build the file deterministically and left-right symmetrically: one module per patch under `src/opensculptboy/topology/patches/` and a builder `python -m opensculptboy.topology.build` that composes them in a fixed order and writes the file. Patches must not overlap: each claims its faces, and the builder refuses a face claimed twice.

### Code

- `opensculptboy.topology.sculpt`: loads the file, checks the digest, and turns anny's `ModelData` into the patched `ModelData` (removed faces, appended rows through `apply_retopology`, UVs); `sculpt_model(rig, phenotypes, ...)` gives an Anny model on the new mesh.
- `Character.topology` accepts `"sculpt"` (and `"sculpt-quads"` for the detail layer), and `Character.build_model` builds it. A card on `anny` stays byte-identical to today's output.
- `opensculptboy.topology.convert`: `to_anny` and `to_sculpt` for vertex fields of a level (detail offsets, per-vertex maps): identity on anny's own vertices, and on the patched regions a barycentric transfer from the closest points of the other mesh. The change of `src/anny` stays at zero if the integration can live in `opensculptboy`; a hook in `anny` is allowed only where the model itself must change, and kept small for upstream merges.

## API (workstream 1)

The contract of a patch module is in `src/opensculptboy/topology/README.md`: a module of `opensculptboy.topology.patches` with `NAME`, `ORDER` (eye_corner 100, ear_root 200, placeholder 900), `ENABLED`, `OPTIONS` and `build(base: BaseQuadMesh, **options) -> Patch`, authored against `opensculptboy.topology.base_quad_mesh()` (anny's quad mesh at the default body).

- `opensculptboy.topology.patch`: `Patch(name, removed_faces, anchors, weights, faces, uv_faces, uv_anchors, uv_weights, vertex_labels=None, face_labels=None, symmetric=True, options={})` in its own numbering (`N + m` a new vertex, `U + j` a new UV), with `mirrored`, `union` and `with_mirror`; `blend_rows([{anchor: weight}])`; `validate_patch(patch, base)` and `check_symmetry`, raising `PatchError`. New UVs are convex rows over anny's UVs (`uv_anchors`, `uv_weights`), so that the mirror image of a patch carries its UVs over.
- `opensculptboy.topology.sculpt`: `compose(names=None, include=(), exclude=(), base=None, options=None) -> SculptData`; `SculptData.write(path)` and `read_sculpt_data(path)` (checks the sidecar's SHA-256 and the digest of anny's quad mesh); `sculpt_data()` (the innermost `use_sculpt_data(data)` block, else `OPENSCULPTBOY_SCULPT_TOPOLOGY`, else the shipped file); `sculpt_model_data(base_data, data=None, triangulate=False)`; `sculpt_model(rig, topology="sculpt" | "sculpt-quads", local_changes, facial_actions, phenotypes, ..., data=None)`; `build_model(topology, **kwargs)` for any topology.
- `python -m opensculptboy.topology.build [--with P] [--without P] [--only P] [--out FILE]` writes `src/opensculptboy/data/topology/sculpt.safetensors` and `sculpt.json`, deterministically.
- `Character(topology="sculpt")` and `"sculpt-quads"` build through `sculpt_model`; `opensculptboy.detail.subdivision.level("sculpt", n)` subdivides `sculpt-quads`, with `catmull_clark_mixed` on level 0 when it holds triangles (the general rules; quad meshes keep anny's `catmull_clark`), and its cache follows the current sculpt data.
- `opensculptboy.topology.convert`: `to_sculpt(field, subdiv=0, vertices=None, data=None)`, `to_anny(...)`, `LevelCorrespondence(subdiv)` and `convert_offsets(source, target, topology)`.

Choices of the core:

- `src/anny` is unchanged: `sculpt_model` builds anny's `anny-quads` ModelData (cached by anny), patches it and calls `Anny.from_model_data`. Every row goes through `apply_retopology(check_weights=True)` except the blend shapes, whose rows follow its formula outside it (identity rows copied bit for bit, new rows summed as it sums them), which saves several GB with every blend shape. Anny's rows keep their skin weights bit for bit (apply_retopology renormalises them).
- Faces of `sculpt-quads`: anny's quads without the removed ones, in anny's order, then the new faces. `sculpt` triangulates the kept quads as anny does and keeps the triangles.
- The new vertices take `base_mesh_vertex_indices` from 19158 (the vertex count of MakeHuman's `base.obj`) upward, so a lookup of a base-mesh table at a new vertex fails loudly instead of reading another vertex.
- The bone data (heads, orientations, landmarks) does not depend on the vertices and is anny's own; procrustes buffers index anny's vertices, which keep their indices.

Known places that assume anny's mesh, for workstream 4: `export.vrm.check_topology` and `vrm_tables.VRM_TOPOLOGIES` refuse `sculpt` (`TopologyConfig.from_string`); `vrm_tables.skin_weight_digest` changes with the new vertices, so `skin_weight_data` warns and the refinement edits need a sculpt entry; the tables indexed by base-mesh indices (`fit.register.excluded_vertices` and `ear_weights`, `fit.metrics` regions, `posing.head` MediaPipe lookup, `anny.keypoints.KeypointsRegressor`, `anny.viewer.landmarks`, `anny.poses.authoring.rig`) index past their arrays at the new vertices; `fit.register.fit_model` builds `anny.Anny()` and not the card's topology; `export.body.triangulated_faces` now drops the empty half of a `[a, b, c, c]` face, but other code that splits `(F, 4)` faces (`anny.utils.subdivision.quads_to_triangles`, `MixedSubdivision`) assumes quads; `texture.ears` and the ear UV files, `eyes.mesh.eyed_mesh` and its caruncle, the bake and the head and ear tiles take the topology name but were only checked on anny; the eyes' caruncle should drop on `sculpt` once the eye-corner patch lands; the budgets count anny's vertex numbers.

## Workstreams

1. **Core** (`opensculptboy/topology/`): the data format, the loader, the ModelData transform, `Character` integration, `sculpt` and `sculpt-quads`, the detail levels on `sculpt-quads`, a placeholder patch for the tests (a small quad split on the back), and the tests: identity on anny's vertices for every blend shape, facial action and face shape; the new rows exactly reproduced; manifold and watertight where anny is; symmetry; UV checks; the glTF export of a `sculpt` card; byte-identical `anny` exports.
2. **Eye-corner patch** (`patches/eye_corner.py`): replace the faces of the inner-corner pocket of each eye with a caruncle and a closed fold of the conjunctiva, so that the corner reads as closed pink tissue from the front and from up to 35 degrees to either side and 20 degrees up or down, at rest, with the eyes looking left and right and in a blink, on levels 0, 1 and 2. The eyes of `opensculptboy.eyes` drop their own caruncle part on `sculpt` (their socket material keeps its pink, wet look on the patch's faces).
3. **Ear-root patch** (`patches/ear_root.py`): replace the faces along the root of the outer ear (the strip where the ear meets the head, from the top of the helix root past the tragus and the lobe to the back) with a denser, regular band of quads, so that the root can bend smoothly from the head into the ear, and the fit's ear split has a clean line to part along.
4. **Downstream and converter**: the fit (`excluded_vertices`, regions, `ear_weights`, the ear split's `pinna` and attachment on the new vertices), the measures, the bake, the eyes, the VRM export (`check_topology`), the budgets, the converter of detail offsets, and the acceptance run on the real character.

Workstream 1 lands the format and the loader first. Workstreams 2 and 3 author their patches against this file format from the start and test them with a local loader until the core lands. Workstream 4 starts with an audit of every place that assumes anny's mesh, then adapts each one once the core lands.

## Acceptance

- Synthetic targets in CI: the inner corner shows no socket skin through the lids from the views above (a ray test), the ear root has no crack or fold that the target lacks (`metrics`), and every other measure of `test.test_reference_fit` holds.
- The real character (private data, outside CI): refit on `sculpt` with `masks.ear`, rendered with `render_compare.py` in Blender against the replica, with no hole in the inner corner and no crack at the ear root, and the concha and tip gains of the ear split kept.
- No regression on `anny`: the full CI tier passes, and `TestGltfReference` holds unchanged.

## Out of scope for the first version

- The web viewer keeps anny's mesh; showing `sculpt` there is a later step.
- SMPL, SMPL-X and SOMA topologies stay as they are.

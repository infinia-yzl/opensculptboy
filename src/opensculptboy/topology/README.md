# The sculpt topology: patch modules

`sculpt` is anny's mesh with patches (docs/sculpt-topology.md). This page is the contract of a patch module. It is stable: the core composes every module that follows it.

## A patch module

A patch module is one file `src/opensculptboy/topology/patches/<name>.py` with:

- `NAME: str`: the patch's label, also the label of its new vertices and faces (`"eye_corner"`, `"ear_root"`).
- `ORDER: int`: its place in the composition (`eye_corner` 100, `ear_root` 200, the test-only `placeholder` 900). The builder composes the modules by `ORDER`, then by name, so a new module needs no edit elsewhere.
- `ENABLED: bool`: whether the shipped file holds the patch (`placeholder` is `False`).
- `OPTIONS: dict`: the default keyword arguments of `build`, JSON values only; the file records the options each patch was built with.
- `build(base: BaseQuadMesh, **options) -> Patch`: the patch, deterministic, from nothing but `base` and the options.

The builder imports every module of `opensculptboy.topology.patches` whose name does not start with `_` and that has a callable `build`.

## The base mesh

`base` is `opensculptboy.topology.patch.base_quad_mesh()`: anny's quad mesh (`topology="anny-quads"`) at the default body, in anny's frame (metres, Z up, the face toward -Y, X toward the figure's left). Author against it:

- `base.vertices` (N, 3) float64, `base.quads` (F, 4) counter-clockwise seen from outside, `base.uv_quads` (F, 4) into `base.uvs` (U, 2) (OBJ convention, v = 0 at the bottom), `base.base_mesh_vertex_indices` (N,) the MakeHuman `base.obj` index of each vertex. Vertex i of `base` is vertex i of every model of the `anny`, `anny-quads` and `sculpt` topologies.
- `base.face_of(vertices)`, `base.vertex_faces[v]`, `base.face_uvs(faces)` (the UV of each vertex of the faces; raises `PatchError` across a UV seam), `base.face_labels()` (the MakeHuman body segmentation per quad), `base.mirror_vertices`, `base.mirror_faces`, `base.mirror_uvs`.
- To build anny's ModelData or a model for the patch's own checks, use `anny.Anny(topology="anny-quads")`, whose vertices and faces are those of `base`.

## The Patch

`opensculptboy.topology.patch.Patch` holds one patch in its own local numbering (N = `base.vertex_count`, U = `base.uv_count`):

- `name`: `NAME`.
- `removed_faces` (R,): indices into `base.quads` of the faces the patch replaces, all inside one UV chart.
- `anchors` (M, 8) int64 and `weights` (M, 8) float64: new vertex m is `sum_k weights[m, k] * base.vertices[anchors[m, k]]`, with non-negative weights summing to 1, distinct anchors and unused slots anchor 0 with weight 0. `blend_rows([{anchor: weight}, ...])` gives them in canonical form. A convex combination may lie off anny's surface, inside the hull of its anchors.
- `faces` (F, 4) int64: the new faces, counter-clockwise seen from outside, over `i < N` (anny's vertex i) and `N + m` (the patch's new vertex m). A triangle repeats its last index: `[a, b, c, c]`. Use quads where possible and triangles where the patch joins anny's faces.
- `uv_faces` (F, 4) int64: the UV of each corner, over `j < U` (anny's UV j) and `U + n` (the patch's new UV n), with the same repeated last index.
- `uv_anchors` (J, 8) and `uv_weights` (J, 8): new UV n is `sum_k uv_weights[n, k] * base.uvs[uv_anchors[n, k]]`, by the same rules. Take the UV anchors from the UVs of the removed faces (`base.face_uvs(removed_faces)`), usually with the weights of the matching vertex.
- `vertex_labels` (M,) and `face_labels` (F,): optional finer labels (default `name` everywhere), such as `"eye_corner/caruncle"`.
- `symmetric`: `True` (the default) makes the validator check that the patch is its own mirror image across X = 0.
- `options`: the options `build` used.

`Patch.mirrored(base)` gives the mirror image, `Patch.union(other, base)` joins two patches of one name, and `Patch.with_mirror(base)` joins one side and its mirror image: author the left side (x > 0) and return `left.with_mirror(base)`.

## The rules

`opensculptboy.topology.patch.validate_patch(patch, base)` raises `PatchError` unless:

- the removed faces are unique and in range; the vertex and UV rows are convex as above, with anchors in range;
- the faces use valid indices, quads have four distinct vertices and triangles three; every new vertex and new UV is used; the UV faces have the shape and the triangles of the faces;
- no new face repeats a kept face of anny or another new face, and no directed edge is used twice;
- the new faces close the hole exactly: their boundary is the boundary of the removed faces, edge for edge and in the same direction, in vertices and in UVs;
- the new UV triangles keep the orientation of the removed faces (none flipped or degenerate) and cover the same area of the atlas;
- with `symmetric`, the patch is its own mirror image.

The builder also refuses a face that two patches remove.

## Building and testing a patch

```bash
python -m opensculptboy.topology.build                     # compose the ENABLED patches and write data/topology/sculpt.safetensors and sculpt.json
python -m opensculptboy.topology.build --with placeholder  # add patches that are off by default (never commit such a file)
python -m opensculptboy.topology.build --only eye_corner --out /tmp/sculpt   # one patch, elsewhere
```

In a test, compose in memory and use the result for the duration of a block:

```python
from opensculptboy.topology import compose, use_sculpt_data
data = compose(["eye_corner"])
with use_sculpt_data(data):
    model = Character(topology="sculpt").build_model()
```

`opensculptboy.topology.sculpt_model_data(data, base_data)` turns anny's quad ModelData into the patched one, and `sculpt_model(...)` builds the Anny model.

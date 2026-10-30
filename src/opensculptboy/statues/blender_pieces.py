# OpenSculptBoy
# Apache License, Version 2.0
"""
Blender helper: rebuild a statue's hat and locks as clean closed meshes.

The hat and the locks of a statue are pieces cut from the scan (``seated_*.ply`` of the
``measure`` step, or ``card/<name>.<piece>.glb``): their edges are ragged where the masks cut
them, and they keep the scan's holes where the scanner did not see into the curls. For each
piece this script

1. welds the scan's split vertices and fills its holes of up to ``sides`` edges (the long cut
   along the skin stays open);
2. smooths the open border along itself (the sawtooth of the cut);
3. grows a thin solid of ``thick`` metres behind the carved surface (plain offsets clamped to
   the neighbouring edges: an even offset spiked at the scan's folded vertices);
4. voxel-remeshes the solid at ``voxel`` metres into one closed surface;
5. drops loose bits under 1 % of the largest and reduces the piece to its triangle budget.

Usage::

    blender -b --factory-startup -P blender_pieces.py -- MESH_DIR OUT_DIR \\
        [voxel=0.0008] [thick=0.002] [sides=80] [hat=150000] [locks=200000]

``MESH_DIR`` holds ``seated_clothing_drapery.ply`` (the hat) and ``seated_hair_hair.ply`` with
``seated_hair_hair_cut.ply`` (the locks); ``OUT_DIR`` receives ``hat.ply``, ``locks.ply`` (in the
same frame; with ``union=1`` one piece ``hat_locks.ply``, which closes the slit along their shared
cut under the brim) and ``report.json`` (triangles before and after, the non-manifold edges, which a
closed piece has none of, and the islands).
"""

import json
import os
import sys

PIECES = {
    "hat": ["seated_clothing_drapery.ply"],
    "locks": ["seated_hair_hair.ply", "seated_hair_hair_cut.ply"],
}
DEFAULTS = {
    "voxel": 0.0008,
    "thick": 0.002,
    "sides": 80,
    "clamp": 1.0,
    "border_rings": 3,
    "border_repeat": 10,
    "hat": 150000,
    "locks": 200000,
    # 1: the hat and the locks as one piece (they share the cut under the brim, and apart
    # they left a dark slit there)
    "union": 0,
}


def parse(argv) -> tuple[str, str, dict]:
    """The mesh folder, the output folder and the settings of the command line."""
    if len(argv) < 2:
        raise SystemExit("usage: blender_pieces.py -- MESH_DIR OUT_DIR [name=value...]")
    settings = dict(DEFAULTS)
    for item in argv[2:]:
        key, _, value = item.partition("=")
        if key not in DEFAULTS:
            raise SystemExit(f"unknown setting {key!r}; use {sorted(DEFAULTS)}")
        settings[key] = type(DEFAULTS[key])(float(value))
    return argv[0], argv[1], settings


def _islands(bm) -> list:
    seen, islands = set(), []
    for face in bm.faces:
        if face.index in seen:
            continue
        stack, island = [face], []
        seen.add(face.index)
        while stack:
            f = stack.pop()
            island.append(f)
            for edge in f.edges:
                for g in edge.link_faces:
                    if g.index not in seen:
                        seen.add(g.index)
                        stack.append(g)
        islands.append(island)
    return sorted(islands, key=len, reverse=True)


def rebuild(mesh_dir: str, out_dir: str, settings: dict) -> dict:
    """Rebuild every piece found in ``mesh_dir`` (see the module documentation)."""
    import bmesh
    import bpy

    def triangles(ob):
        return sum(len(p.vertices) - 2 for p in ob.data.polygons)

    os.makedirs(out_dir, exist_ok=True)
    report = {}
    pieces = dict(PIECES)
    budgets = {k: int(settings[k]) for k in PIECES}
    if settings["union"]:
        pieces = {"hat_locks": PIECES["hat"] + PIECES["locks"]}
        budgets = {"hat_locks": int(settings["hat"]) + int(settings["locks"])}
    for piece, files in pieces.items():
        paths = [os.path.join(mesh_dir, f) for f in files]
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            continue
        bpy.ops.wm.read_factory_settings(use_empty=True)
        parts = []
        for path in paths:
            bpy.ops.wm.ply_import(filepath=path)
            parts.append(bpy.context.selected_objects[0])
        raw = sum(triangles(p) for p in parts)
        bpy.ops.object.select_all(action="DESELECT")
        for p in parts:
            p.select_set(True)
        bpy.context.view_layer.objects.active = parts[0]
        if len(parts) > 1:
            bpy.ops.object.join()
        ob = bpy.context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.mesh.fill_holes(sides=int(settings["sides"]))
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.mesh.select_non_manifold(
            extend=False,
            use_wire=False,
            use_boundary=True,
            use_multi_face=False,
            use_non_contiguous=False,
            use_verts=False,
        )
        for _ in range(int(settings["border_rings"])):
            bpy.ops.mesh.select_more()
        bpy.ops.mesh.vertices_smooth(factor=0.5, repeat=int(settings["border_repeat"]))
        bpy.ops.object.mode_set(mode="OBJECT")
        solid = ob.modifiers.new("solid", "SOLIDIFY")
        solid.thickness = float(settings["thick"])
        solid.offset = -1.0
        solid.use_even_offset = False
        solid.thickness_clamp = float(settings["clamp"])
        solid.use_rim = True
        bpy.ops.object.modifier_apply(modifier=solid.name)
        remesh = ob.modifiers.new("remesh", "REMESH")
        remesh.mode = "VOXEL"
        remesh.voxel_size = float(settings["voxel"])
        remesh.adaptivity = 0.0
        bpy.ops.object.modifier_apply(modifier=remesh.name)
        remeshed = triangles(ob)
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        islands = _islands(bm)
        small = [
            f for isl in islands[1:] if len(isl) < 0.01 * len(islands[0]) for f in isl
        ]
        if small:
            bmesh.ops.delete(bm, geom=small, context="FACES")
        bm.to_mesh(ob.data)
        bm.free()
        budget = budgets[piece]
        if triangles(ob) > budget:
            decimate = ob.modifiers.new("decimate", "DECIMATE")
            decimate.ratio = budget / triangles(ob)
            bpy.ops.object.modifier_apply(modifier=decimate.name)
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        report[piece] = {
            "raw_triangles": raw,
            "remeshed_triangles": remeshed,
            "triangles": triangles(ob),
            "non_manifold_edges": sum(1 for e in bm.edges if len(e.link_faces) != 2),
            "islands": len(_islands(bm)),
        }
        bm.free()
        bpy.ops.object.select_all(action="DESELECT")
        ob.select_set(True)
        bpy.ops.wm.ply_export(
            filepath=os.path.join(out_dir, f"{piece}.ply"),
            export_selected_objects=True,
            export_normals=False,
            export_uv=False,
            export_colors="NONE",
            export_triangulated_mesh=True,
            ascii_format=False,
        )
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


if __name__ == "__main__" and "--" in sys.argv:
    print(rebuild(*parse(sys.argv[sys.argv.index("--") + 1 :])), flush=True)

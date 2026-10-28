# OpenSculptBoy
# Apache License, Version 2.0
"""
Render head-and-shoulders pictures of facial-action mixes of an OpenSculptBoy GLB file in Blender, for
the review of expression shapes (front and three-quarter views, a textured look and a clay look).

Run it with Blender in the background::

    blender -b --factory-startup -P blender_faces.py -- --glb FILE --faces faces.json --out DIR \\
        [--views front,tq] [--looks textured,clay] [--res 512] [--samples 32] \\
        [--helper src/opensculptboy/export/blender_skin.py] [--only PREFIX,...] [--engine eevee|cycles]

``faces.json`` is ``{"faces": [{"name": "a1/jawOpen", "mix": {"jawOpen": 1.0}}, ...]}``
(:mod:`opensculptboy.expressions.authoring.faces` writes it). The faces play as a GLB file plays
them (:func:`face_weights`): a mix key ``K`` with a shape key ``K.corrective`` and an action
``K.jaw`` turns the jaw by the weight's share of the action's last key while the corrective takes
the weight (``jawOpen`` and a reference's mouth poses on the ``anny-mouth`` rig), and the blinks
raise their in-between key (1 at a weight of 0.5). The clay look gives every material one grey clay
and makes the cornea, the lashes, the hair and the brows clear. EEVEE draws the refracting cornea
opaque, so the textured look gives it a clear coat of gloss there.

Outputs: ``DIR/<look>/<view>/<face name with / as __>.png`` and ``DIR/render_log.json``.
"""

import argparse
import json
import math
import os
import sys
import time


def face_weights(mix: dict, rig_keys, key_names) -> tuple[dict, dict]:
    """
    The shape-key values of a mix of facial actions, and the weight of each jaw turn.

    ``rig_keys`` are the mix keys that play as a turn of the jaw with a corrective (their
    shape key stays at 0, ``<key>.corrective`` takes the weight), and ``key_names`` the shape
    keys of the file: an ``eyeBlink*`` key raises ``<key>.inbetween`` to ``1 - |2 w - 1|``.
    """
    values, turns = {}, {}
    for k, w in mix.items():
        if k in rig_keys:
            values[k + ".corrective"] = values.get(k + ".corrective", 0.0) + w
            turns[k] = turns.get(k, 0.0) + w
            continue
        values[k] = values.get(k, 0.0) + w
        if k.startswith("eyeBlink") and k + ".inbetween" in key_names:
            values[k + ".inbetween"] = max(0.0, 1.0 - abs(2 * w - 1.0))
    return values, turns


def main():
    import bpy
    from mathutils import Quaternion, Vector

    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--glb", required=True)
    p.add_argument("--faces", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--views", default="front,tq")
    p.add_argument("--looks", default="textured,clay")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--helper", default=None)
    p.add_argument("--only", default=None)
    p.add_argument("--engine", default="eevee")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    LOG = {"glb": a.glb, "faces": 0, "times": {}, "notes": []}

    def note(*x):
        s = " ".join(str(v) for v in x)
        print("[blender_faces]", s, flush=True)
        LOG["notes"].append(s)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    t0 = time.time()
    bpy.ops.import_scene.gltf(filepath=a.glb)
    note("import", round(time.time() - t0, 1), "s")
    sc = bpy.context.scene

    helper = None
    if a.helper:
        import importlib.util

        spec = importlib.util.spec_from_file_location("blender_skin", a.helper)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        note("skin set on", helper.apply_skin_materials(a.glb))
        note("eyes set on", helper.apply_eye_materials(a.glb))

    meshes = [o for o in sc.objects if o.type == "MESH"]

    def eevee_cornea(m):
        """EEVEE draws the refracting cornea opaque: a clear coat of gloss over the iris instead."""
        nt = m.node_tree
        out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        gl = nt.nodes.new("ShaderNodeBsdfGlossy")
        gl.inputs["Roughness"].default_value = 0.03
        fr = nt.nodes.new("ShaderNodeFresnel")
        fr.inputs["IOR"].default_value = 1.376
        mx = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(fr.outputs[0], mx.inputs[0])
        nt.links.new(tr.outputs[0], mx.inputs[1])
        nt.links.new(gl.outputs[0], mx.inputs[2])
        nt.links.new(mx.outputs[0], out.inputs["Surface"])
        for link in list(out.inputs["Volume"].links):
            nt.links.remove(link)
        for attr, val in (
            ("blend_method", "HASHED"),
            ("surface_render_method", "DITHERED"),
            ("use_transparent_shadow", True),
        ):
            if hasattr(m, attr):
                try:
                    setattr(m, attr, val)
                except Exception:  # noqa: BLE001
                    pass

    if a.engine != "cycles":
        for m in bpy.data.materials:
            if "cornea" in m.name.lower() and m.use_nodes:
                eevee_cornea(m)
                note("cornea drawn clear for EEVEE:", m.name)

    arm = next((o for o in sc.objects if o.type == "ARMATURE"), None)

    # --- the jaw turns of the file's ".jaw" actions, read in Blender's pose space
    jaw_turns = {}
    if arm is not None:
        if arm.animation_data is None:
            arm.animation_data_create()
        ad = arm.animation_data
        for track in list(ad.nla_tracks):
            track.mute = True
        for act in bpy.data.actions:
            base = act.name
            for suffix in ("_Armature", "_" + arm.name):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
            if not base.endswith(".jaw"):
                continue
            key = base[: -len(".jaw")]
            try:
                ad.action = act
                if hasattr(ad, "action_slot") and len(getattr(act, "slots", [])):
                    slot = next(
                        (s for s in act.slots if s.target_id_type == "OBJECT"),
                        act.slots[0],
                    )
                    ad.action_slot = slot
                sc.frame_set(int(round(act.frame_range[1])))
                q = arm.pose.bones["jaw"].rotation_quaternion.copy()
                jaw_turns[key] = q
            except Exception as e:  # noqa: BLE001
                note("jaw action", act.name, "failed:", e)
            ad.action = None
        note("jaw turns", {k: [round(c, 4) for c in q] for k, q in jaw_turns.items()})
        for pb in arm.pose.bones:
            pb.location = (0, 0, 0)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.rotation_euler = (0, 0, 0)
            pb.scale = (1, 1, 1)
        sc.frame_set(0)
    for o in meshes:
        if o.data.shape_keys and o.data.shape_keys.animation_data:
            o.data.shape_keys.animation_data.action = None
            for t in o.data.shape_keys.animation_data.nla_tracks:
                t.mute = True

    key_names = set()
    for o in meshes:
        if o.data.shape_keys:
            key_names.update(k.name for k in o.data.shape_keys.key_blocks[1:])
    rig_keys = {k for k in jaw_turns if k + ".corrective" in key_names}
    note("rig keys (jaw turn + corrective)", sorted(rig_keys))

    def set_face(mix):
        values, turns = face_weights(mix, rig_keys, key_names)
        rotvec = Vector((0, 0, 0))
        for k, w in turns.items():
            q = jaw_turns[k]
            ang = 2 * math.acos(max(-1, min(1, q.w)))
            axis = Vector((q.x, q.y, q.z))
            if axis.length > 1e-9:
                rotvec += axis.normalized() * ang * w
        missing = [k for k in values if k not in key_names]
        for o in meshes:
            sk = o.data.shape_keys
            if not sk:
                continue
            for kb in sk.key_blocks[1:]:
                value = values.get(kb.name, 0.0)
                kb.slider_max = max(1.0, value)
                kb.value = value
        if arm is not None and "jaw" in arm.pose.bones:
            pb = arm.pose.bones["jaw"]
            pb.rotation_mode = "QUATERNION"
            if rotvec.length > 1e-9:
                pb.rotation_quaternion = Quaternion(rotvec.normalized(), rotvec.length)
            else:
                pb.rotation_quaternion = (1, 0, 0, 0)
        bpy.context.view_layer.update()
        return missing

    # --- framing: the head bone and the top of the body mesh
    def world_bone_head(name):
        b = arm.pose.bones[name]
        return arm.matrix_world @ b.head

    body = max(meshes, key=lambda o: len(o.data.vertices))
    if arm is not None and "head" in arm.pose.bones:
        hb = world_bone_head("head")
    else:
        zs = [body.matrix_world @ v.co for v in body.data.vertices]
        top = max(v.z for v in zs)
        hb = Vector((0, 0, top - 0.2))
    near = [body.matrix_world @ v.co for v in body.data.vertices]
    near = [v for v in near if v.z > hb.z - 0.05 and abs(v.x - hb.x) < 0.15]
    top = max(v.z for v in near)
    front_y = min(v.y for v in near)
    bottom = hb.z - 0.12
    span = (top - bottom) * 1.08
    cz = (top + bottom) / 2
    target = Vector((hb.x, (hb.y + front_y) / 2 + 0.01, cz))
    note(
        "frame",
        "head",
        tuple(round(c, 3) for c in hb),
        "top",
        round(top, 3),
        "span",
        round(span, 3),
    )

    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.lens = 85
    cam.data.sensor_fit = "VERTICAL"
    cam.data.sensor_height = 24
    dist = (span / 2) / math.tan(math.atan(12 / 85))
    VIEWS = {"front": 0.0, "tq": 35.0}

    def place(view):
        az = math.radians(VIEWS[view])
        cam.location = target + Vector((math.sin(az) * dist, -math.cos(az) * dist, 0.0))
        cam.rotation_euler = (math.radians(90), 0, az)

    # --- lights and world
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.42, 0.44, 0.47, 1)
    bg.inputs[1].default_value = 0.18

    def area(name, loc, energy, size, color=(1, 1, 1)):
        L = bpy.data.lights.new(name, "AREA")
        L.energy = energy
        L.shape = "DISK"
        L.size = size
        L.color = color
        o = bpy.data.objects.new(name, L)
        sc.collection.objects.link(o)
        o.location = target + Vector(loc)
        d = target - o.location
        o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        return o

    area("key", (-0.6, -0.75, 0.85), 36, 0.45, (1.0, 0.97, 0.93))
    area("fill", (0.9, -0.6, 0.0), 5, 0.9, (0.93, 0.96, 1.0))
    area("rim", (0.5, 0.8, 0.6), 18, 0.4)

    engines = {
        e.identifier
        for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    }
    if a.engine == "cycles":
        sc.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for dev in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = dev
                prefs.get_devices()
                for d in prefs.devices:
                    d.use = True
                sc.cycles.device = "GPU"
                break
            except Exception:  # noqa: BLE001
                continue
        sc.cycles.samples = a.samples
        sc.cycles.use_denoising = True
    else:
        sc.render.engine = (
            "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
        )
        ee = sc.eevee
        ee.taa_render_samples = a.samples
        for attr, val in (
            ("use_raytracing", True),
            ("use_shadows", True),
            ("shadow_ray_count", 2),
            ("use_gtao", True),
        ):
            if hasattr(ee, attr):
                try:
                    setattr(ee, attr, val)
                except Exception:  # noqa: BLE001
                    pass
    sc.render.resolution_x = sc.render.resolution_y = a.res
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.view_settings.view_transform = (
        "AgX"
        if "AgX"
        in [
            x.identifier
            for x in sc.view_settings.bl_rna.properties["view_transform"].enum_items
        ]
        else "Filmic"
    )
    note("engine", sc.render.engine)

    # --- looks
    originals = {o.name: [s.material for s in o.material_slots] for o in meshes}
    clay = bpy.data.materials.new("clay")
    clay.use_nodes = True
    cb = clay.node_tree.nodes.get("Principled BSDF")
    cb.inputs["Base Color"].default_value = (0.55, 0.55, 0.56, 1)
    cb.inputs["Roughness"].default_value = 0.55
    clear = bpy.data.materials.new("clay_clear")
    clear.use_nodes = True
    nt = clear.node_tree
    for n in list(nt.nodes):
        if n.bl_idname != "ShaderNodeOutputMaterial":
            nt.nodes.remove(n)
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    nt.links.new(tr.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    for attr, val in (
        ("blend_method", "HASHED"),
        ("surface_render_method", "DITHERED"),
    ):
        if hasattr(clear, attr):
            try:
                setattr(clear, attr, val)
            except Exception:  # noqa: BLE001
                pass
    CLEAR_WORDS = ("cornea", "lash", "hair", "brow", "band")

    def set_look(look):
        for o in meshes:
            for s, m in zip(o.material_slots, originals[o.name]):
                if look == "textured":
                    s.material = m
                else:
                    name = (m.name if m else "").lower()
                    s.material = (
                        clear if any(wd in name for wd in CLEAR_WORDS) else clay
                    )

    faces = json.load(open(a.faces))["faces"]
    if a.only:
        pre = tuple(a.only.split(","))
        faces = [f for f in faces if f["name"].startswith(pre)]
    views = a.views.split(",")
    missing_all = {}
    for look in a.looks.split(","):
        set_look(look)
        for f in faces:
            miss = set_face(f["mix"])
            if miss:
                missing_all[f["name"]] = miss
            for view in views:
                place(view)
                d = os.path.join(a.out, look, view)
                os.makedirs(d, exist_ok=True)
                sc.render.filepath = os.path.join(
                    d, f["name"].replace("/", "__") + ".png"
                )
                t = time.time()
                bpy.ops.render.render(write_still=True)
                LOG["times"].setdefault(look, []).append(round(time.time() - t, 2))
            LOG["faces"] += 1
        note(look, "done", len(faces), "faces", round(time.time() - t0, 1), "s")
    LOG["missing_keys"] = missing_all
    LOG["total_s"] = round(time.time() - t0, 1)
    json.dump(LOG, open(os.path.join(a.out, "render_log.json"), "w"), indent=1)
    note("total", LOG["total_s"], "s")


if __name__ == "__main__" and "bpy" in sys.modules:
    main()

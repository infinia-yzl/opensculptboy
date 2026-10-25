# OpenSculptBoy
# Apache License, Version 2.0
"""
Fit a VRM file into the budget of its version (``opensculptboy export --fit-budget``).

:func:`fit_spec` changes a :class:`~opensculptboy.export.vrm.VrmSpec` until the file it writes
keeps within :data:`opensculptboy.export.budget.BUDGETS`, and reports what it did:

1. **Triangles.** The pieces (:mod:`opensculptboy.export.pieces`) share the budget by the cost
   of their reduction: one run of :func:`opensculptboy.export.decimate.decimate` over every
   piece removes the vertices whose removal costs least, where the cost of a vertex is its
   geometric error times its importance. The importance is the piece's priority
   (:data:`PRIORITIES`, the face of the body six times and the hands three times its rest),
   times how much of the vertex's surface the views of :mod:`opensculptboy.render.ids` see
   (the front most), so hidden and inner surfaces go first. Each piece keeps a floor of its
   triangles (:data:`FLOORS`, more when the budget asks for little: :data:`FLOOR_RATIOS`); the
   eyes and the lashes stay whole. A piece can be given a
   fixed number of triangles or another priority. The reduction keeps a subset of each
   piece's vertices, each with its position, normal, UVs, skin weights and morph target
   offsets, so the bind and the expressions of every kept vertex stay exact.
2. **Normal maps.** A reduced wearable piece whose shading drifts from the full piece gets its
   normal maps baked again from the full piece (:mod:`opensculptboy.export.rebake`, with the
   viewer extra's Embree), and a piece without a normal map gets one where the bake brings
   its shading closer.
3. **Materials.** :func:`opensculptboy.export.atlas.atlas_spec` merges the wearables'
   materials into atlases.
4. **Joints.** Over the joint budget, the joints that weight no vertex leave the skin (their
   nodes stay in the humanoid map).
5. **Textures.** Every image larger than the version's texture size shrinks to it.

Without ``--fit-budget`` nothing here runs, and the files stay as they were.
"""

from __future__ import annotations

import copy
import dataclasses
import time

import numpy as np

# the priority of each kind of piece, and the factors of its face and hands
PRIORITIES = {"body": 1.0, "teeth": 0.5, "mouth": 0.5, "wearable": 1.0}
REGION_FACTORS = {
    "body": {"face": 6.0, "hands": 3.0},
    "wearable": {"face": 2.0, "hands": 1.5},
}
# the share of its triangles that each kind of piece keeps at least: the larger of a fixed
# share and a multiple of the file's own share (the budget over the triangles), so the body
# keeps at least 95 % of its own while the file keeps half of its triangles
FLOORS = {"body": 0.3, "teeth": 0.25, "mouth": 0.15, "wearable": 0.08}
FLOOR_RATIOS = {"body": 2.0, "teeth": 1.0, "mouth": 1.0, "wearable": 0.25}
MIN_TRIANGLES = 32
# pieces that stay whole unless given a number of triangles
WHOLE = ("eyes", "lashes")
# the visibility of a vertex counts from this floor, so hidden surfaces keep some weight
VISIBILITY_FLOOR = 0.1
VIEW_SIZE = 384
# morph targets whose shapes join the cost of the body's reduction (the largest movers)
SHAPES = 6


@dataclasses.dataclass
class FitOptions:
    """
    Options of :func:`fit_spec`.

    Args:
        priorities: a factor on the importance of a piece, by name (``body``, ``eyes``,
            ``lashes``, ``teeth``, ``mouth`` or a wearable's reference) or by kind.
        triangles: a fixed number of triangles for a piece, by name.
        rebake: bake normal maps again for reduced wearables (needs Embree).
        quality: compare the fitted file with the full one (:mod:`opensculptboy.export.budget_quality`).
    """

    priorities: dict = dataclasses.field(default_factory=dict)
    triangles: dict = dataclasses.field(default_factory=dict)
    rebake: bool = True
    quality: bool = True

    @classmethod
    def parse(cls, priorities=(), triangles=(), **kwargs) -> "FitOptions":
        """Options from ``NAME=VALUE`` strings, as the command takes them."""

        def pairs(items, kind):
            out = {}
            for item in items or ():
                name, sep, value = str(item).rpartition("=")
                if not sep or not name:
                    raise ValueError(f"{item!r}: give NAME=VALUE")
                try:
                    out[name] = kind(value)
                except ValueError:
                    raise ValueError(f"{item!r}: {value!r} is not a number") from None
                if out[name] < 0:
                    raise ValueError(f"{item!r}: the value must not be negative")
            return out

        return cls(
            priorities=pairs(priorities, float),
            triangles=pairs(triangles, int),
            **kwargs,
        )


def _target(spec) -> str:
    return "vrm1" if spec.version == "1.0" else "vrm0"


def _descendants(spec, roots) -> set:
    out, stack = set(), [r for r in roots if r is not None]
    while stack:
        n = stack.pop()
        if n in out:
            continue
        out.add(n)
        stack.extend(spec.nodes[n].children)
    return out


def region_weights(spec, mesh) -> tuple[np.ndarray, np.ndarray]:
    """(N,) the skin weight of each vertex on the head and its children (the face), and on
    the hands and the fingers."""
    head = _descendants(spec, [spec.humanoid.get("head")])
    hands = _descendants(
        spec, [spec.humanoid.get("leftHand"), spec.humanoid.get("rightHand")]
    )
    joints = np.asarray(mesh.body.joints)
    weights = np.asarray(mesh.body.weights, np.float64)
    face = (np.isin(joints, sorted(head)) * weights).sum(1)
    hand = (np.isin(joints, sorted(hands)) * weights).sum(1)
    return face, hand


def _take(body, rows: np.ndarray) -> None:
    from opensculptboy.export.atlas import _take_vertices

    _take_vertices(body, rows)


def reduce_spec(spec, limit: int, options: FitOptions | None = None) -> dict:
    """
    Reduce the triangles of a spec's pieces to ``limit`` in all (see the module); changes the
    spec in place and returns, per piece (``pieces``), the triangles before and after, the
    floor and the largest error (mm), with the passes of the reduction and why it stopped.
    """
    from opensculptboy.export.decimate import decimate
    from opensculptboy.export.pieces import primitive_triangles, spec_pieces
    from opensculptboy.render.ids import views, visible_pixels

    options = options or FitOptions()
    pieces = spec_pieces(spec)
    offsets = np.cumsum([0] + [len(m.body.positions) for m in spec.meshes])
    positions = np.concatenate([np.asarray(m.body.positions) for m in spec.meshes])
    normals = np.concatenate([np.asarray(m.body.normals) for m in spec.meshes])
    uvs = np.concatenate(
        [
            np.zeros((len(m.body.positions), 2))
            if m.body.uvs is None
            else np.asarray(m.body.uvs)
            for m in spec.meshes
        ]
    )
    # the triangles of every piece, labelled by (mesh, primitive) and grouped by piece
    tri_parts, label_parts, group_parts, prim_keys, face_parts = [], [], [], [], []
    face_of = {}
    for g, piece in enumerate(pieces):
        mesh = spec.meshes[piece.mesh]
        if piece.mesh not in face_of:
            rows = np.asarray(mesh.body.triangles, np.int64).reshape(-1, 3)
            faces = np.asarray(mesh.body.triangle_faces, np.int64)
            face_of[piece.mesh] = _row_lookup(rows, faces)
        for p in piece.primitives:
            t = primitive_triangles(mesh, mesh.primitives[p])
            face_parts.append(face_of[piece.mesh](t))
            t = t + offsets[piece.mesh]
            tri_parts.append(t)
            label_parts.append(np.full(len(t), len(prim_keys)))
            group_parts.append(np.full(len(t), g))
            prim_keys.append((piece.mesh, p))
    T = np.concatenate(tri_parts)
    face_rows = np.concatenate(face_parts)
    labels = np.concatenate(label_parts)
    groups = np.concatenate(group_parts)
    counts = np.bincount(groups, minlength=len(pieces))
    ratio = min(1.0, float(limit) / max(len(T), 1))

    # the importance of each vertex: priority, region and visibility
    forward = np.array([0.0, 0.0, 1.0 if spec.version == "1.0" else -1.0])
    cameras = views(positions, forward, VIEW_SIZE)
    seen = visible_pixels(positions, T, cameras)
    p = positions[T]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    pixel_area = cameras[0].scale ** 2
    total_weight = sum(c.weight for c in cameras)
    seen_v = np.zeros(len(positions))
    area_v = np.zeros(len(positions))
    for k in range(3):
        np.add.at(seen_v, T[:, k], seen)
        np.add.at(area_v, T[:, k], area / pixel_area * total_weight)
    visibility = seen_v / np.maximum(area_v, 1e-12)
    top = np.quantile(visibility[area_v > 0], 0.95) if (area_v > 0).any() else 1.0
    visibility = VISIBILITY_FLOOR + np.clip(visibility / max(top, 1e-9), 0.0, 1.0)
    importance = np.zeros(len(positions))
    locked = np.zeros(len(positions), bool)
    floors, fixed = {}, {}
    for g, piece in enumerate(pieces):
        mesh = spec.meshes[piece.mesh]
        rows = np.unique(T[groups == g])
        kind = piece.kind
        factor = options.priorities.get(piece.name, options.priorities.get(kind, 1.0))
        priority = max(PRIORITIES.get(kind, 1.0) * float(factor), 1e-3)
        region = REGION_FACTORS.get("body" if kind == "body" else "wearable")
        face, hand = region_weights(spec, mesh)
        local = rows - offsets[piece.mesh]
        weight = (
            1.0
            + (region["face"] - 1.0) * face[local]
            + (region["hands"] - 1.0) * hand[local]
        )
        importance[rows] = np.maximum(
            importance[rows], priority * weight * visibility[rows]
        )
        if piece.name in options.triangles:
            fixed[g] = min(int(options.triangles[piece.name]), int(counts[g]))
        elif kind in WHOLE and not (
            piece.name in options.priorities or kind in options.priorities
        ):
            fixed[g] = int(counts[g])
        else:
            share = max(
                FLOORS.get(kind, FLOORS["wearable"]),
                FLOOR_RATIOS.get(kind, FLOOR_RATIOS["wearable"]) * ratio,
            )
            floors[g] = min(
                int(counts[g]), max(MIN_TRIANGLES, int(np.ceil(share * counts[g])))
            )
        if kind == "body":  # the open edges of the face (lids, lips) stay
            t = T[groups == g]
            e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), 1)
            key, number = np.unique(e, axis=0, return_counts=True)
            ends = np.unique(key[number == 1])
            locked[ends[face[ends - offsets[piece.mesh]] > 0.5]] = True
    shapes = []
    body_mesh = spec.meshes[0]
    if body_mesh.body.targets:
        motion = [
            float(np.linalg.norm(t, axis=1).sum()) for t in body_mesh.body.targets
        ]
        for k in np.argsort(motion)[::-1][:SHAPES]:
            s = np.zeros_like(positions)
            s[: offsets[1]] = body_mesh.body.targets[k]
            shapes.append(s)

    # the fixed pieces first, each to its own number; then every other piece at once
    report = {}
    started = time.time()
    keep_T, keep_labels, keep_groups, keep_origin = [], [], [], []
    fixed_total = 0
    for g, number in fixed.items():
        pick = groups == g
        if number >= counts[g]:
            out_t, out_l, out_o = T[pick], labels[pick], np.flatnonzero(pick)
            err = 0.0
        else:
            r = decimate(
                positions,
                T[pick],
                number,
                labels=labels[pick],
                uvs=uvs,
                normals=normals,
                shapes=shapes,
                importance=importance,
                locked=locked,
            )
            out_t, out_l, out_o = r.triangles, r.labels, np.flatnonzero(pick)[r.origin]
            err = max(r.error.values(), default=0.0)
        keep_T.append(out_t)
        keep_labels.append(out_l)
        keep_groups.append(np.full(len(out_t), g))
        keep_origin.append(out_o)
        fixed_total += len(out_t)
        report[pieces[g].name] = {
            "before": int(counts[g]),
            "after": int(len(out_t)),
            "fixed": True,
            "error_mm": 1000 * err,
        }
    free = ~np.isin(groups, list(fixed))
    target = max(int(limit) - fixed_total, 0)
    floor_total = sum(floors.values())
    run = {"passes": 0, "stopped": "target"}
    if fixed_total > limit:
        raise ValueError(
            f"the pieces that keep their triangles hold {fixed_total:,}, over the budget of "
            f"{limit:,}: give some pieces fewer triangles (--fit-triangles)"
        )
    if floor_total > 0.9 * target:  # the floors give way, alike, to the budget
        scale = 0.9 * target / floor_total
        floors = {g: max(1, int(f * scale)) for g, f in floors.items()}
    if free.any():
        if free.sum() > target:
            out_t, out_l, out_g = T[free], labels[free], groups[free]
            out_o = np.flatnonzero(free)
            errors, passes = {}, 0
            # a run that stops above the target (the floors, or no valid collapse left) runs
            # again with the floors halved
            for attempt in range(4):
                r = decimate(
                    positions,
                    out_t,
                    target,
                    labels=out_l,
                    groups=out_g,
                    floors=floors,
                    uvs=uvs,
                    normals=normals,
                    shapes=shapes,
                    importance=importance,
                    locked=locked,
                )
                out_t, out_l, out_g = r.triangles, r.labels, r.groups
                out_o = out_o[r.origin]
                for g, e in r.error.items():
                    errors[g] = max(errors.get(g, 0.0), e)
                passes += r.passes
                if len(out_t) <= target:
                    break
                floors = {g: f // 2 for g, f in floors.items()}
            run = {"passes": passes, "stopped": r.stopped, "runs": attempt + 1}
        else:
            out_t, out_l, out_g, out_o = (
                T[free],
                labels[free],
                groups[free],
                np.flatnonzero(free),
            )
            errors = {}
            run = {"passes": 0, "stopped": "target"}
        keep_T.append(out_t)
        keep_labels.append(out_l)
        keep_groups.append(out_g)
        keep_origin.append(out_o)
        after = np.bincount(out_g, minlength=len(pieces))
        for g in np.unique(groups[free]):
            report[pieces[g].name] = {
                "before": int(counts[g]),
                "after": int(after[g]),
                "floor": int(floors.get(int(g), 0)),
                "error_mm": 1000 * float(errors.get(int(g), 0.0)),
            }
    T2 = np.concatenate(keep_T)
    L2 = np.concatenate(keep_labels)
    F2 = face_rows[np.concatenate(keep_origin)]

    # back into the meshes: each mesh keeps the vertices its triangles use
    from opensculptboy.export.vrm import SpecPrimitive

    for m, mesh in enumerate(spec.meshes):
        prims = [k for k, key in enumerate(prim_keys) if key[0] == m]
        if not prims:
            continue
        lo, hi = offsets[m], offsets[m + 1]
        parts = [(prim_keys[k][1], T2[L2 == k] - lo, F2[L2 == k]) for k in prims]
        used = np.unique(np.concatenate([t.ravel() for _, t, _ in parts]))
        remap = np.full(hi - lo, -1)
        remap[used] = np.arange(len(used))
        _take(mesh.body, used)
        new_prims, faces = [], []
        for p_index, t, f in parts:
            if len(t):
                new_prims.append(
                    SpecPrimitive(mesh.primitives[p_index].material, remap[t])
                )
                faces.append(f)
        mesh.primitives = new_prims
        mesh.body.triangles = np.concatenate([p.triangles for p in new_prims])
        mesh.body.triangle_faces = np.concatenate(faces)
    return {"pieces": report, "seconds": round(time.time() - started, 2), **run}


def _row_lookup(rows: np.ndarray, faces: np.ndarray):
    """A function that gives the ``faces`` entry of each triangle among ``rows`` (0 for a
    triangle that is not a row)."""
    if len(faces) != len(rows):
        return lambda t: np.zeros(len(t), np.int64)
    n = int(rows.max(initial=0)) + 1
    key = (rows[:, 0] * n + rows[:, 1]) * n + rows[:, 2]
    order = np.argsort(key)
    sorted_key = key[order]

    def lookup(t):
        t = np.asarray(t, np.int64)
        k = (t[:, 0] * n + t[:, 1]) * n + t[:, 2]
        pos = np.minimum(np.searchsorted(sorted_key, k), len(sorted_key) - 1)
        found = sorted_key[pos] == k
        return np.where(found, faces[order[pos]], 0)

    return lookup


def image_roles(spec) -> dict:
    """The role of each image of a spec: ``normal`` for normal maps, ``colour`` otherwise."""
    roles = {}
    for m in spec.materials:
        if m.normal_texture is not None:
            roles[m.normal_texture] = "normal"
        for f in (
            "base_texture",
            "shade_texture",
            "matcap_texture",
            "emissive_texture",
            "rim_multiply_texture",
        ):
            k = getattr(m, f)
            if k is not None:
                roles.setdefault(k, "colour")
        if m.outline_width_texture is not None:
            roles.setdefault(m.outline_width_texture, "data")
    return roles


def shrink_textures(spec, size: int) -> list:
    """Shrink every image of a spec (and its thumbnail) whose larger side exceeds ``size``,
    keeping its aspect; returns the names and sizes."""
    from opensculptboy.export.atlas import resize_image
    from opensculptboy.export.budget import image_size

    roles = image_roles(spec)
    out = []
    images = list(spec.images)
    for k, (name, data) in enumerate(images):
        found = image_size(data)
        if found is None or max(found) <= size:
            continue
        factor = size / max(found)
        new = (
            max(1, int(round(found[0] * factor))),
            max(1, int(round(found[1] * factor))),
        )
        images[k] = (name, resize_image(data, new, roles.get(k, "colour")))
        out.append({"name": name, "from": list(found), "to": list(new)})
    spec.images = images
    if spec.thumbnail is not None:
        found = image_size(spec.thumbnail)
        if found is not None and max(found) > size:
            spec.thumbnail = resize_image(spec.thumbnail, (size, size), "colour")
            out.append({"name": "thumbnail", "from": list(found), "to": [size, size]})
    return out


def prune_joints(spec) -> dict:
    """Take the joints that weight no vertex out of the spec's skins (their nodes stay, with
    the humanoid map and the constraints), and renumber the vertices' joints; returns the
    joints before and after."""
    used = set()
    for mesh in spec.meshes:
        joints = np.asarray(mesh.body.joints)
        weights = np.asarray(mesh.body.weights)
        used |= {int(j) for j in np.unique(joints[weights > 0])}
    before = sum(len(skin.joints) for skin in spec.skins)
    for skin in spec.skins:
        keep = [k for k in range(len(skin.joints)) if k in used]
        index = np.zeros(len(skin.joints), np.int64)
        index[keep] = np.arange(len(keep))
        skin.joints = [skin.joints[k] for k in keep]
        skin.inverse_bind = np.asarray(skin.inverse_bind)[keep]
    for mesh in spec.meshes:
        mesh.body.joints = index[np.asarray(mesh.body.joints)]
    return {"before": int(before), "after": int(sum(len(s.joints) for s in spec.skins))}


def fit_spec(spec, options: FitOptions | None = None) -> dict:
    """
    Fit a spec into the budget of its version (see the module): changes it in place and
    returns the report: the counts and pieces before and after, each step and its results,
    and the budget messages left (when a count cannot come down, such as the joints).
    """
    from opensculptboy.export import budget as budgets
    from opensculptboy.export.atlas import atlas_spec
    from opensculptboy.export.pieces import piece_report, spec_counts

    options = options or FitOptions()
    target = _target(spec)
    limits = budgets.BUDGETS[target]
    before_counts = spec_counts(spec)
    report = {
        "target": target,
        "before": {"counts": before_counts, "pieces": piece_report(spec)},
        "steps": [],
    }
    over = budgets.check(before_counts, target, "warn")
    if not over:  # nothing to do: the spec stays as it is
        report["after"] = report["before"]
        report["messages"] = []
        return report
    full = copy.deepcopy(spec) if (options.rebake or options.quality) else None
    if before_counts["triangles"] > limits["triangles"]:
        started = time.time()
        reduced = reduce_spec(spec, limits["triangles"], options)
        report["steps"].append({"step": "reduce", **reduced})
        if options.rebake and full is not None:
            from opensculptboy.export.rebake import rebake_spec

            started = time.time()
            baked = rebake_spec(full, spec)
            report["steps"].append(
                {
                    "step": "rebake",
                    "pieces": baked,
                    "seconds": round(time.time() - started, 2),
                }
            )
    if len(spec.materials) > limits["materials"]:
        started = time.time()
        atlas = atlas_spec(spec, limits["materials"], limits["texture_size"])
        report["steps"].append(
            {
                "step": "atlas",
                **atlas.to_dict(),
                "seconds": round(time.time() - started, 2),
            }
        )
    if len(spec.skins) == 1 and spec_counts(spec)["joints"] > limits["joints"]:
        report["steps"].append({"step": "joints", **prune_joints(spec)})
    shrunk = shrink_textures(spec, limits["texture_size"])
    if shrunk:
        report["steps"].append({"step": "textures", "images": shrunk})
    after_counts = spec_counts(spec)
    report["after"] = {"counts": after_counts, "pieces": piece_report(spec)}
    report["messages"] = budgets.check(after_counts, target, "warn")
    if options.quality and full is not None and report["steps"]:
        from opensculptboy.export.budget_quality import compare

        started = time.time()
        report["quality"] = compare(full, spec)
        report["quality"]["seconds"] = round(time.time() - started, 2)
    if report["steps"]:
        spec.extras["budget_fit"] = {
            "target": target,
            "steps": [s["step"] for s in report["steps"]],
            "before": {
                k: before_counts[k]
                for k in ("triangles", "materials", "max_image_size")
            },
        }
    return report


def summary_lines(report: dict) -> list[str]:
    """What :func:`fit_spec` did, one line per step, for the command's output."""
    lines = []
    b, a = report["before"]["counts"], report["after"]["counts"]
    lines.append(
        f"fit to the {report['target']} budget: {b['triangles']:,} -> {a['triangles']:,} "
        f"triangles, {b['materials']} -> {a['materials']} materials, largest texture "
        f"{b['max_image_size']} -> {a['max_image_size']}"
    )
    for step in report["steps"]:
        if step["step"] == "reduce":
            parts = [
                f"{name} {v['before']:,}->{v['after']:,}"
                for name, v in step["pieces"].items()
                if v["before"] != v["after"]
            ]
            lines.append("reduced: " + ", ".join(parts))
        elif step["step"] == "rebake":
            done = [n for n, v in step["pieces"].items() if v.get("baked")]
            lines.append("normal maps baked again: " + (", ".join(done) or "none"))
        elif step["step"] == "atlas":
            lines.append(
                f"atlas: {step['merged']} materials into {len(step['pages'])} pages "
                + ", ".join(f"{p['size'][0]}x{p['size'][1]}" for p in step["pages"])
            )
        elif step["step"] == "joints":
            lines.append(f"joints: {step['before']} -> {step['after']} in the skin")
        elif step["step"] == "textures":
            lines.append(
                "textures shrunk: "
                + ", ".join(
                    f"{i['name']} {i['from'][0]}->{i['to'][0]}" for i in step["images"]
                )
            )
    for message in report.get("messages", []):
        lines.append("still over: " + message)
    return lines

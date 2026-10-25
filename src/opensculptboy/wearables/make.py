# OpenSculptBoy
# Apache License, Version 2.0
"""
A wearable package from a garment modelled on a body (``opensculptboy wearables bind``,
docs/wearables.md, "Make a wearable").

The garment comes as a GLB file whose meshes sit on the body of a card at rest, in the frame
of an ``opensculptboy export`` of that card: metres, glTF axes (Y up, the figure facing +Z),
with the body standing on the floor (``floor``: y = 0 at its lowest vertex, as an engine
shows the export at its default pose) or centred as Anny's template (``centred``: the
export's bind pose, or an export with ``--no-ground``); ``auto`` takes the frame in which
the garment lies nearer the skin. :func:`read_garment` reads
the meshes in the file's default scene at its default pose (node transforms and skins
applied, as an importer shows them) and gathers their triangles into groups by material, by
mesh node or by the ``group`` of the primitive's extras. :func:`make_wearable` binds the
garment to the body (surface binding, or rigid binding to one bone), adds cover masks from a
ray test and returns the package with a report.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from opensculptboy.wearables.bind import BodySurface, bind_rigid, bind_surface, evaluate
from opensculptboy.wearables.cover import RAY_REACH
from opensculptboy.wearables.format import Garment, Wearable, WearableError
from opensculptboy.wearables.geometry import ray_hits, unit_rows, vertex_normals

GROUPS_FROM = ("materials", "nodes", "extras")
COVERS = ("auto", "none")
# the garment's frame: the body on the floor (an export's default pose), the body centred
# (Anny's template; an export's bind pose, or --no-ground), or whichever fits
FRAMES = ("auto", "floor", "centred")
WARN_DISTANCE = 0.05  # metres: a surface vertex further from the skin gets a warning
MAX_DISTANCE = 0.10  # metres: the median distance above which the binding is refused
BODY_DISTANCE = 0.0005  # metres: a mesh this close to the skin is the body itself
BODY_VERTICES = 2000


class GarmentError(WearableError):
    """A garment file that cannot become a package as it stands."""

    code = "invalid_garment"


@dataclasses.dataclass
class GarmentFile:
    """The garment of a GLB file in the file's frame (glTF axes), with notes on what the
    reader changed, and the skin weights of the file by joint name (or None)."""

    garment: Garment
    skin_weights: np.ndarray | None
    skin_joints: np.ndarray | None
    joint_names: list | None
    node_names: list
    triangle_node: np.ndarray
    notes: list
    normals_missing: bool = False  # no mesh of the file has normals


def _normal_matrix(m: np.ndarray) -> np.ndarray:
    return np.linalg.inv(m[:3, :3]).T


def _scene_nodes(gltf: dict) -> list[int]:
    """The nodes of the default scene and their descendants (every node without scenes)."""
    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes") or []
    if not scenes:
        return list(range(len(nodes)))
    roots = scenes[gltf.get("scene", 0)].get("nodes", [])
    out, stack = [], list(roots)
    while stack:
        i = stack.pop()
        out.append(i)
        stack.extend(nodes[i].get("children", []))
    return sorted(set(out))


def read_garment(path, groups_from: str = "materials") -> GarmentFile:
    """
    The garment meshes of a GLB file at its default pose, in the file's frame.

    Args:
        path: the GLB file.
        groups_from: ``materials`` (one group per material, named by the material),
            ``nodes`` (one group per mesh node, named by the node) or ``extras`` (the
            ``group`` of the primitive's extras, else of its mesh's, else of its node's,
            else ``main``).
    """
    from opensculptboy.mouth.handoff import _world_matrices, accessor, read_glb
    from opensculptboy.wearables.gltf import read_materials

    if groups_from not in GROUPS_FROM:
        raise GarmentError(f"groups_from must be one of {GROUPS_FROM}.")
    gltf, binary = read_glb(path)
    world = _world_matrices(gltf)
    nodes = gltf.get("nodes", [])
    entries = gltf.get("materials", [])
    notes: list[str] = []
    parts = []
    with_normals = 0
    for n in _scene_nodes(gltf):
        node = nodes[n]
        if "mesh" not in node:
            continue
        mesh = gltf["meshes"][node["mesh"]]
        node_name = node.get("name") or mesh.get("name") or f"node{n}"
        skin = gltf["skins"][node["skin"]] if "skin" in node else None
        joint_matrices = None
        if skin is not None:
            count = len(skin["joints"])
            inverse = np.tile(np.eye(4), (count, 1, 1))
            if "inverseBindMatrices" in skin:
                raw = accessor(gltf, binary, skin["inverseBindMatrices"])
                inverse = raw.astype(np.float64).reshape(count, 4, 4).transpose(0, 2, 1)
            joint_matrices = np.stack(
                [world[j] @ inverse[k] for k, j in enumerate(skin["joints"])]
            )
        for p, primitive in enumerate(mesh["primitives"]):
            label = f"{node_name} primitive {p}"
            if primitive.get("mode", 4) != 4:
                notes.append(
                    f"{label}: not triangles (mode {primitive['mode']}); left out"
                )
                continue
            attributes = primitive["attributes"]
            local = accessor(gltf, binary, attributes["POSITION"]).astype(np.float64)
            if "indices" in primitive:
                triangles = accessor(gltf, binary, primitive["indices"]).astype(
                    np.int64
                )
            else:
                triangles = np.arange(len(local), dtype=np.int64)
            triangles = triangles.reshape(-1, 3)
            normals = None
            if "NORMAL" in attributes:
                normals = accessor(gltf, binary, attributes["NORMAL"]).astype(
                    np.float64
                )
            uvs = None
            if "TEXCOORD_0" in attributes:
                uvs = accessor(gltf, binary, attributes["TEXCOORD_0"]).astype(
                    np.float64
                )
            weights = joints = None
            if skin is not None and "JOINTS_0" in attributes:
                joints = accessor(gltf, binary, attributes["JOINTS_0"]).astype(np.int64)
                weights = accessor(gltf, binary, attributes["WEIGHTS_0"]).astype(float)
                if "JOINTS_1" in attributes:
                    joints = np.concatenate(
                        [joints, accessor(gltf, binary, attributes["JOINTS_1"])], 1
                    ).astype(np.int64)
                    weights = np.concatenate(
                        [weights, accessor(gltf, binary, attributes["WEIGHTS_1"])], 1
                    ).astype(float)
            if weights is not None and joint_matrices is not None:
                total = weights.sum(axis=1, keepdims=True)
                w = weights / np.where(total > 0, total, 1.0)
                M = np.einsum("nk,nkij->nij", w, joint_matrices[joints])
                M[total[:, 0] <= 0] = world[n]
            else:
                M = np.broadcast_to(world[n], (len(local), 4, 4))
            # the vertices that this primitive's triangles use (primitives may share one
            # set of vertices, as in a package or an export)
            rows = np.unique(triangles)
            remap = np.full(len(local), -1, np.int64)
            remap[rows] = np.arange(len(rows))
            triangles = remap[triangles]
            local, M = local[rows], np.asarray(M)[rows]
            if normals is not None:
                normals = normals[rows]
            if uvs is not None:
                uvs = uvs[rows]
            if weights is not None:
                weights, joints = weights[rows], joints[rows]
            positions = np.einsum("nij,nj->ni", M[:, :3, :3], local) + M[:, :3, 3]
            flip = np.linalg.det(M[:, :3, :3]) < 0
            if len(triangles) and flip[triangles].all(axis=1).any():
                rows = flip[triangles].all(axis=1)
                triangles[rows] = triangles[rows][:, ::-1]
            with_normals += normals is not None
            if normals is None:
                normals = vertex_normals(positions, triangles)
                notes.append(f"{label}: no normals; computed from the triangles")
            else:
                N = np.linalg.inv(M[:, :3, :3]).transpose(0, 2, 1)
                normals = unit_rows(np.einsum("nij,nj->ni", N, normals))
            material = primitive.get("material")
            if groups_from == "materials":
                group = "main" if material is None else entries[material].get("name")
                group = group or f"material{material}"
            elif groups_from == "nodes":
                group = node_name
            else:
                group = "main"
                for source in (primitive, mesh, node):
                    value = (source.get("extras") or {}).get("group")
                    if isinstance(value, str) and value:
                        group = value
                        break
            parts.append(
                dict(
                    label=label,
                    node=node_name,
                    positions=positions,
                    normals=normals,
                    uvs=uvs,
                    triangles=triangles,
                    material=material,
                    group=group,
                    weights=weights,
                    joints=None if joints is None else joints,
                    joint_names=None
                    if skin is None
                    else [nodes[j].get("name", f"node{j}") for j in skin["joints"]],
                )
            )
    if not parts:
        raise GarmentError(
            f"{path}: no triangle meshes in the default scene.",
            "no_meshes",
            "Export the garment objects as triangle meshes into the default scene.",
        )
    used = sorted({p["material"] for p in parts if p["material"] is not None})
    materials, index = read_materials(gltf, binary, which=used)
    if any(p["material"] is None for p in parts):
        from opensculptboy.wearables.format import Material

        index[None] = len(materials)
        materials.append(Material("garment"))
    with_uvs = [p["uvs"] is not None for p in parts]
    if any(with_uvs) and not all(with_uvs):
        missing = [p["label"] for p in parts if p["uvs"] is None]
        notes.append(f"no UVs on {missing}; their UVs are 0")
    elif not any(with_uvs):
        notes.append("no UVs: the package holds no texture coordinates")
    groups: list[str] = []
    node_names: list[str] = []
    vertices, normals, uvs, tris, tmat, tgroup, tnode = [], [], [], [], [], [], []
    count = 0
    for p in parts:
        if p["group"] not in groups:
            groups.append(p["group"])
        if p["node"] not in node_names:
            node_names.append(p["node"])
        vertices.append(p["positions"])
        normals.append(p["normals"])
        uvs.append(
            p["uvs"] if p["uvs"] is not None else np.zeros((len(p["positions"]), 2))
        )
        tris.append(p["triangles"] + count)
        k = len(p["triangles"])
        tmat.append(np.full(k, index[p["material"]]))
        tgroup.append(np.full(k, groups.index(p["group"])))
        tnode.append(np.full(k, node_names.index(p["node"])))
        count += len(p["positions"])
    garment = Garment(
        np.concatenate(vertices),
        np.concatenate(tris),
        normals=np.concatenate(normals),
        uvs=np.concatenate(uvs) if any(with_uvs) else None,
        triangle_material=np.concatenate(tmat),
        triangle_group=np.concatenate(tgroup),
        materials=materials,
        groups=groups,
    )
    skin_weights = skin_joints = joint_names = None
    if all(p["weights"] is not None for p in parts):
        joint_names = []
        rows_w, rows_j = [], []
        width = max(p["weights"].shape[1] for p in parts)
        for p in parts:
            names = [p["joint_names"][j] for j in range(len(p["joint_names"]))]
            for name in names:
                if name not in joint_names:
                    joint_names.append(name)
            remap = np.array([joint_names.index(name) for name in names])
            pad = width - p["weights"].shape[1]
            rows_w.append(np.pad(p["weights"], ((0, 0), (0, pad))))
            rows_j.append(np.pad(remap[p["joints"]], ((0, 0), (0, pad))))
        skin_weights = np.concatenate(rows_w)
        skin_joints = np.concatenate(rows_j)
        total = skin_weights.sum(axis=1, keepdims=True)
        skin_weights = skin_weights / np.where(total > 0, total, 1.0)
    return GarmentFile(
        garment,
        skin_weights,
        skin_joints,
        joint_names,
        node_names,
        np.concatenate(tnode),
        notes,
        normals_missing=with_normals == 0,
    )


def ground_lift(body: BodySurface) -> float:
    """The height (metres) by which an export stands the body on the floor: the lowest
    vertex as a file stores it (float32), as ``export_glb`` does."""
    return -float(np.float32(body.vertices[:, 2].min()))


def to_anny(points: np.ndarray, lift: float) -> np.ndarray:
    """Points of an export's frame (glTF axes, lifted by ``lift``) in Anny's frame."""
    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    shifted = np.asarray(points, dtype=np.float64) - np.array([0.0, lift, 0.0])
    return shifted @ GLTF_TO_ANNY.T


def _without(garment: Garment, keep: np.ndarray) -> tuple[Garment, np.ndarray]:
    """The garment with only the triangles ``keep`` and the vertices they use; returns the
    kept vertex rows too."""
    tris = garment.triangles[keep]
    rows = np.unique(tris)
    remap = np.full(len(garment.vertices), -1)
    remap[rows] = np.arange(len(rows))
    used = sorted(set(garment.triangle_group[keep].tolist()))
    group_map = {g: k for k, g in enumerate(used)}
    out = Garment(
        garment.vertices[rows],
        remap[tris],
        normals=garment.normals[rows],
        uvs=None if garment.uvs is None else garment.uvs[rows],
        triangle_material=garment.triangle_material[keep],
        triangle_group=np.array(
            [group_map[g] for g in garment.triangle_group[keep]], np.int64
        ),
        materials=garment.materials,
        groups=[garment.groups[g] for g in used],
    )
    return out, rows


def parse_hides(values) -> list[dict]:
    """``hides`` rules from ``PIECE:GROUP`` items (comma-separated or repeated)."""
    rules: dict[str, list[str]] = {}
    for value in values or ():
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            piece, sep, group = item.partition(":")
            if not sep or not piece or not group:
                raise GarmentError(
                    f"--hides takes PIECE:GROUP items (for example tunic:sleeves); "
                    f"got {item!r}.",
                    "bad_hides",
                    "Write each --hides item as PIECE:GROUP.",
                )
            rules.setdefault(piece, [])
            if group not in rules[piece]:
                rules[piece].append(group)
    return [{"piece": p, "groups": g} for p, g in rules.items()]


def make_wearable(
    path,
    body: BodySurface,
    name: str,
    slot: str,
    layer: int,
    *,
    label: str | None = None,
    rigid: str | None = None,
    weights: str = "body",
    groups_from: str = "materials",
    covers: str = "auto",
    hides=(),
    licence: dict | None = None,
    provenance: str = "public",
    body_kind: str = "anny-template",
    frame: str = "auto",
    max_distance: float = MAX_DISTANCE,
    extras: dict | None = None,
    tags=(),
    description: str = "",
    texture_limit: int | None = None,
    normal_green: str = "auto",
    accept_quality: bool = False,
) -> tuple[Wearable, dict]:
    """
    Bind the garment of a GLB file to ``body`` (see the module documentation).

    Returns the checked package, for :func:`~opensculptboy.wearables.format.write_wearable`
    to write, and a report: the vertices bound, the largest and the 95th-percentile distance
    of the bound vertices from the skin, the vertices further than :data:`WARN_DISTANCE`,
    the covered skin vertices, the round-trip error on the binding body and the warnings.
    """
    if frame not in FRAMES:
        raise GarmentError(f"frame must be one of {FRAMES}.")
    if covers not in COVERS:
        raise GarmentError(f"covers must be one of {COVERS}.")
    if weights not in ("body", "own"):
        raise GarmentError("weights must be 'body' or 'own'.")
    if rigid is not None and weights == "own":
        raise GarmentError("A rigid piece follows its bone; leave out --weights own.")
    if rigid is not None and rigid not in body.bone_labels:
        raise GarmentError(
            f"Unknown bone {rigid!r} for --rigid; the rig's bones include "
            f"{body.bone_labels[:8]} (opensculptboy export writes them as the joint nodes).",
            "unknown_bone",
            "Name a bone of the body's rig (the joint nodes of an export).",
        )
    read = read_garment(path, groups_from)
    warnings = list(read.notes)
    garment = read.garment
    file_vertices = garment.vertices
    lifts = {"floor": ground_lift(body), "centred": 0.0}
    tried = {}
    for option in FRAMES[1:] if frame == "auto" else (frame,):
        d = body.surface.closest(to_anny(file_vertices, lifts[option])).distance
        tried[option] = float(d.min() if rigid else np.median(d))
    frame = min(tried, key=tried.get)
    lift = lifts[frame]
    garment.vertices = to_anny(file_vertices, lift)
    garment.normals = unit_rows(to_anny(garment.normals, 0.0))
    found = body.surface.closest(garment.vertices)
    # a mesh that lies on the skin is the body itself: leave it out
    body_nodes = []
    for k, node in enumerate(read.node_names):
        rows = np.unique(garment.triangles[read.triangle_node == k])
        if (
            len(rows) >= BODY_VERTICES
            and np.percentile(found.distance[rows], 95) < BODY_DISTANCE
        ):
            body_nodes.append(k)
    keep_rows = np.arange(len(garment.vertices))
    if body_nodes:
        names = [read.node_names[k] for k in body_nodes]
        warnings.append(f"the meshes {names} lie on the skin (the body); left out")
        keep = ~np.isin(read.triangle_node, body_nodes)
        if not keep.any():
            raise GarmentError(
                f"{path}: every mesh lies on the skin; export the garment objects alone."
            )
        garment, keep_rows = _without(garment, keep)
    from opensculptboy.wearables.quality import (
        TEXTURE_LIMIT,
        check_garment,
        require,
        strip_textures_without_uvs,
    )

    garment, rows, quality = check_garment(
        garment,
        body=body,
        normals_from_file=not read.normals_missing,
        texture_limit=TEXTURE_LIMIT if texture_limit is None else texture_limit,
        normal_green=normal_green,
    )
    require(quality, accept_quality)
    strip_textures_without_uvs(garment, quality)
    keep_rows = keep_rows[rows]
    found = body.surface.closest(garment.vertices)
    own = None
    if weights == "own":
        if read.skin_weights is None:
            raise GarmentError(
                "--weights own needs every garment mesh skinned to the body's bones "
                "(JOINTS_0 and WEIGHTS_0); parent the meshes to the armature of the "
                "exported body and paint their weights.",
                "missing_weights",
                "Parent the meshes to the exported armature and paint their weights.",
            )
        own = (
            read.skin_weights[keep_rows],
            read.skin_joints[keep_rows],
            list(read.joint_names),
        )
    wearable, report = bind_garment(
        garment,
        body,
        name,
        slot,
        layer,
        label=label,
        rigid=rigid,
        own_weights=own,
        covers=covers,
        hides=hides,
        licence=licence,
        provenance=provenance,
        body_kind=body_kind,
        max_distance=max_distance,
        extras=extras,
        tags=tags,
        description=description,
        found=found,
        warnings=warnings,
        source=str(path),
        tried=tried,
        file_size=float(np.ptp(file_vertices, axis=0).max()),
    )
    report["frame"] = frame
    report["ground_lift_m"] = round(lift, 6)
    report["quality"] = quality.to_dict()
    return wearable, report


def bind_garment(
    garment: Garment,
    body: BodySurface,
    name: str,
    slot: str,
    layer: int,
    *,
    label: str | None = None,
    rigid: str | None = None,
    own_weights=None,
    covers: str = "auto",
    fixed_covers: np.ndarray | None = None,
    hides=(),
    licence: dict | None = None,
    provenance: str = "public",
    body_kind: str = "anny-template",
    max_distance: float = MAX_DISTANCE,
    extras: dict | None = None,
    tags=(),
    description: str = "",
    found=None,
    warnings: list | None = None,
    source: str = "the garment",
    tried: dict | None = None,
    file_size: float | None = None,
) -> tuple[Wearable, dict]:
    """
    Bind a garment that lies on ``body`` at rest (Anny's frame) and return the checked
    package with a report (see :func:`make_wearable`). ``own_weights`` is (weights, joints,
    joint names) of the garment's own skin, ``fixed_covers`` a mask over the body's vertices
    that the piece always covers (for example MakeHuman's ``delete_verts``; ``covers="auto"``
    then adds no ray masks), ``tried`` the median distances of the frames tried, for the
    message of a refusal.
    """
    warnings = [] if warnings is None else warnings
    if found is None:
        found = body.surface.closest(garment.vertices)
    distance = found.distance
    measure = float(distance.min() if rigid else np.median(distance))
    if measure > max_distance:
        what = "nearest" if rigid else "median"
        tried = tried or {"": measure}
        seen = ", ".join(f"{k} {v * 1000:.0f} mm".strip() for k, v in tried.items())
        hint = ""
        if file_size is not None and file_size > 10.0:
            hint = f" The garment spans {file_size:.0f} units: export it in metres."
        raise GarmentError(
            f"{source}: the garment lies too far from the skin ({what} distance: {seen}; "
            f"the limit is {max_distance * 1000:.0f} mm). Model it on the body of "
            "'opensculptboy export base.glb' (the default card, or the card of --body), "
            f"in metres with Y up, and keep the body where the file puts it.{hint}",
            "too_far_from_skin",
            "Model the garment on the binding body in metres, or raise --max-distance.",
        )
    far = distance > WARN_DISTANCE
    if rigid is None and far.any():
        warnings.append(
            f"{int(far.sum())} vertices lie more than {WARN_DISTANCE * 1000:.0f} mm from "
            f"the skin (the largest {distance.max() * 1000:.0f} mm); they follow the skin "
            "at their bound points and may stretch on other bodies"
        )
    if rigid is None:
        binding = bind_surface(
            garment,
            body,
            body_kind,
            body.triangles[found.triangle],
            found.barycentric,
        )
    else:
        binding = bind_rigid(garment, body, rigid, body_kind)
    weights_kind = "rigid" if rigid else ("own" if own_weights is not None else "body")
    if own_weights is not None:
        skin_weights, skin_joints, joint_names = own_weights
        used = np.unique(skin_joints[skin_weights > 0])
        missing = sorted({joint_names[j] for j in used} - set(body.bone_labels))
        if missing:
            raise GarmentError(
                f"The garment's skin names joints the body's rig lacks: {missing}.",
                "unknown_joints",
                "Skin the garment to the bones of the exported armature.",
            )
        empty = skin_weights.sum(axis=1) <= 0
        if empty.any():
            raise GarmentError(
                f"{int(empty.sum())} garment vertices carry no skin weight; paint their "
                "weights, or leave out --weights own.",
                "unweighted_vertices",
                "Paint a weight on every garment vertex, or bind with the body's weights.",
            )
        # the joints that weigh on the garment, so that the file's other joints (a
        # control bone, a joint of the body's mesh) need no bone of the rig
        remap = np.zeros(len(joint_names), np.int64)
        remap[used] = np.arange(len(used))
        binding.skin_weights = skin_weights
        binding.skin_joints = remap[skin_joints]
        binding.joint_names = [joint_names[j] for j in used]
    group_covers: dict[str, np.ndarray] = {}
    if covers == "auto" and fixed_covers is None:
        for g, group in enumerate(garment.groups):
            opaque = garment.triangle_group == g
            for m, material in enumerate(garment.materials):
                if material.alpha_mode != "OPAQUE":
                    opaque &= garment.triangle_material != m
            if not opaque.any():
                continue
            hit = ray_hits(
                body.vertices,
                body.normals,
                garment.vertices,
                garment.triangles[opaque],
                RAY_REACH,
            )
            if hit.any():
                group_covers[group] = hit
    always = np.zeros(len(body.vertices), bool)
    if fixed_covers is not None:
        always = np.asarray(fixed_covers, bool).copy()
    covered = always.copy()
    for mask in group_covers.values():
        covered |= mask
    wearable = Wearable(
        name=name,
        label=label,
        slot=slot,
        layer=int(layer),
        garment=garment,
        binding=binding,
        provenance=provenance,
        licence={k: v for k, v in (licence or {}).items() if v or k == "spdx"},
        weights=weights_kind,
        covers=always,
        group_covers=group_covers,
        cover_ray=covers == "auto" and fixed_covers is None,
        hides=parse_hides(hides),
        tags=clean_tags(tags),
        description=" ".join(str(description or "").split()),
        extras=dict(extras or {}),
    )
    wearable.check()
    back, _ = evaluate(binding, body)
    report = {
        "name": name,
        "slot": slot,
        "layer": int(layer),
        "method": binding.method,
        "bone": rigid,
        "weights": weights_kind,
        "groups": list(garment.groups),
        "vertices": int(len(garment.vertices)),
        "triangles": int(len(garment.triangles)),
        "materials": len(garment.materials),
        "tags": list(wearable.tags),
        "description": wearable.description,
        "bind_distance_mm_max": round(float(distance.max()) * 1000, 2),
        "bind_distance_mm_p95": round(float(np.percentile(distance, 95)) * 1000, 2),
        "far_vertices": int(far.sum()),
        "far_distance_mm": round(WARN_DISTANCE * 1000),
        "covered_skin_vertices": int(covered.sum()),
        "round_trip_mm_max": float(
            np.linalg.norm(back - garment.vertices, axis=1).max() * 1000
        ),
        "warnings": warnings,
    }
    return wearable, report


def clean_tags(values) -> list[str]:
    """Tags from ``--tag`` values and header words: comma-separated items, lower case, single
    spaces, each once, in their first order."""
    out: list[str] = []
    for value in values or ():
        for item in str(value).split(","):
            tag = " ".join(item.split()).lower()
            if tag and tag not in out:
                out.append(tag)
    return out


def round_trip(path, body: BodySurface, vertices: np.ndarray) -> float:
    """The largest distance (mm) between the vertices of a written package, read back and
    evaluated on ``body``, and ``vertices`` (Anny's frame)."""
    from opensculptboy.wearables.format import read_wearable

    back, _ = evaluate(read_wearable(path).binding, body)
    return float(np.linalg.norm(back - vertices, axis=1).max() * 1000)

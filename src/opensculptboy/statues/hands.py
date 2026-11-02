# OpenSculptBoy
# Apache License, Version 2.0
"""
A statue's hands as anny's own hands, posed to the scan.

A scan's hands are often too poor to fit their surface: a fist round a grip or on a hip is
carved as one lump, and the fit's detail levels copied the lump onto anny's fingers. With a
config's ``posed_hands``, a hand keeps anny's own shape (its fingers, nails and palm) on every
detail level, posed by its chain (:mod:`opensculptboy.statues.chains`: the wrist, from the
scan) and by the curl of its fingers, which :func:`pose_hands` solves against the objects the
hand holds: each finger takes its library pose's curl (:data:`FIST`) scaled as far as it goes
before it passes into an object, so that its tip touches a sword's grip or a stone without
entering it. A hand that holds a stone gets the stone as a prop of its own
(:func:`stone_mesh`). A hand whose grip the curl cannot find (a sword's grip with a finger
over the guard) takes a keyed pose instead (``key``: :func:`key_hand`), its bones' turns set
by hand as an animator keys them and checked by picture.
:func:`hand_masks` gives the hand's vertices, which keep anny's shape, and a band of rings at
the wrist, which blends the fitted forearm into the hand.

The objects are in the masks' frame (the scan's: y up), as the config's masks are:
``{"kind": "capsule", "start", "end", "radius"}``, ``{"kind": "sphere", "centre", "radius"}``
or the stone's ``{"kind": "ellipsoid", "centre", "radii", "axis"}``.
"""

from __future__ import annotations

import numpy as np

# the bones of a hand, from the wrist down, by their words (with the side's suffix)
HAND_BONES = ("wrist", "metacarpal", "finger")
# the fingers: finger1 is the thumb
FINGERS = ("finger1", "finger2", "finger3", "finger4", "finger5")
# the library pose whose fingers' curl a posed hand starts from
FIST = "mh_boxing"
# the curl scales tried for each finger (1: the library's curl)
CURLS = tuple(np.round(np.arange(0.0, 1.31, 0.05), 3))
# the clearance (metres) a finger keeps from an object it holds
MARGIN = 0.001
# the rings of the band at the wrist that blends the fitted forearm into anny's hand
BLEND_RINGS = 4


def _side(chain: str) -> str:
    part, _, side = chain.lower().partition(".")
    if part != "hand" or side not in ("l", "r"):
        raise ValueError(f"A posed hand is 'hand.l' or 'hand.r'; got {chain!r}.")
    return side.upper()


def hand_bones(model, chain: str) -> list:
    """The bones (indices) of a hand from the wrist down."""
    suffix = "." + _side(chain)
    return [
        i
        for i, b in enumerate(model.bone_labels)
        if b.endswith(suffix) and b.startswith(HAND_BONES)
    ]


def _hand_weight(model, bones) -> np.ndarray:
    weights = model.vertex_bone_weights.detach().double().cpu().numpy()
    indices = model.vertex_bone_indices.cpu().numpy()
    return (np.isin(indices, bones) * weights).sum(1)


def hand_masks(model, chain: str, rings: int = BLEND_RINGS):
    """Masks (V,) of the level-0 vertices of a hand (more than half their weight on its
    bones), which keep anny's shape, and of the band of ``rings`` rings round them, which
    blends the forearm's fit into the hand."""
    from opensculptboy.fit.lips import rings_from

    hand = _hand_weight(model, hand_bones(model, chain)) > 0.5
    triangles = model.faces.cpu().numpy()
    grown = rings_from(triangles, np.nonzero(hand)[0], int(rings), len(hand))
    return hand, grown & ~hand


def _to_anny(points) -> np.ndarray:
    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    return np.asarray(points, np.float64) @ GLTF_TO_ANNY.T


def signed_distance(points: np.ndarray, objects: list) -> np.ndarray:
    """The signed distance (N,) of points in anny's frame to the nearest of the objects
    (in the masks' frame): negative inside."""
    P = np.asarray(points, np.float64)
    out = np.full(len(P), np.inf)
    for o in objects:
        kind = o["kind"]
        if kind == "sphere":
            c = _to_anny(o["centre"])
            d = np.linalg.norm(P - c, axis=1) - float(o["radius"])
        elif kind == "capsule":
            a, b = _to_anny(o["start"]), _to_anny(o["end"])
            ab = b - a
            t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
            d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1) - float(o["radius"])
        elif kind == "ellipsoid":
            c = _to_anny(o["centre"])
            frame = _ellipsoid_frame(o)
            radii = np.asarray(o["radii"], np.float64)
            local = (P - c) @ frame.T
            # the distance scaled by the smallest radius (exact on a sphere)
            d = (np.linalg.norm(local / radii, axis=1) - 1.0) * radii.min()
        else:
            raise ValueError(
                f"A held object is a sphere, capsule or ellipsoid; got {kind!r}."
            )
        out = np.minimum(out, d)
    return out


def _ellipsoid_frame(o) -> np.ndarray:
    """The rows: the ellipsoid's axes in anny's frame, the first along its ``axis``."""
    axis = _to_anny(o.get("axis", [1.0, 0.0, 0.0]))
    axis = axis / np.linalg.norm(axis)
    other = (
        np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    )
    second = np.cross(axis, other)
    second /= np.linalg.norm(second)
    return np.stack([axis, second, np.cross(axis, second)])


def stone_mesh(stone: dict, subdivisions: int = 4):
    """A closed egg (trimesh) of a stone entry, in the masks' frame."""
    import trimesh

    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    V = np.asarray(mesh.vertices) * np.asarray(stone["radii"], np.float64)
    # an egg: the half along +axis a little narrower than the other
    V[:, 1:] *= 1.0 - 0.12 * np.clip(V[:, :1] / stone["radii"][0], 0.0, 1.0)
    frame = _ellipsoid_frame(stone)
    anny = V @ frame + _to_anny(stone["centre"])
    return trimesh.Trimesh(anny @ GLTF_TO_ANNY, mesh.faces, process=False)


def place_stone(joints: dict, radii, gap: float = 0.02) -> dict:
    """A stone in the palm of a hand posed with its fingers open: centred between the
    wrist and the middle knuckles, ``gap`` plus its smallest radius off the palm's bones
    towards the palm, its long axis across the hand. ``joints`` maps "wrist", "index",
    "middle", "little" and "tip" (the middle finger's last joint) to their places in anny's
    frame. Returns the stone entry in the masks' frame."""
    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    wrist, index, little = (
        np.asarray(joints[k], np.float64) for k in ("wrist", "index", "little")
    )
    middle, tip = np.asarray(joints["middle"]), np.asarray(joints["tip"])
    across = index - little
    along = middle - wrist
    normal = np.cross(across, along)
    normal /= np.linalg.norm(normal)
    # the palm faces the side the curled fingers go to
    if (tip - middle) @ normal < 0:
        normal = -normal
    radii = np.asarray(radii, np.float64)
    centre = 0.45 * wrist + 0.55 * middle + normal * (gap + radii[1:].min())
    return {
        "kind": "ellipsoid",
        "centre": (centre @ GLTF_TO_ANNY).round(5).tolist(),
        "radii": radii.tolist(),
        "axis": ((across / np.linalg.norm(across)) @ GLTF_TO_ANNY).round(5).tolist(),
    }


# the library pose whose joints' turning axes a keyed finger's curl turns about
KEY_FIST = "mh_fight"


def key_hand(model, pose, chain: str, key: dict) -> None:
    """Set a hand's pose as an animator keys it, in place on ``pose`` (1, B, 4, 4).

    ``key["bones"]`` maps bones (without the side's suffix: ``"lowerarm02"``, ``"wrist"``,
    ``"finger1-1"``) to their local turns, rotation vectors in degrees, which replace the
    pose's. ``key["fingers"]`` maps a finger (``"finger2"``) to the curl of its three joints
    in degrees, each about that joint's own bending axis (the axis of its turn in the library
    pose :data:`KEY_FIST`); a joint's entry may instead be a rotation vector in degrees (the
    thumb's opposition)."""
    import roma
    import torch

    from opensculptboy.statues.chains import _library_rotations

    side = "." + _side(chain)
    labels = list(model.bone_labels)

    def turn(v):
        v = torch.as_tensor(np.radians(np.asarray(v, np.float64)), dtype=pose.dtype)
        return roma.rotvec_to_rotmat(v)

    for name, value in key.get("bones", {}).items():
        if name + side not in labels:
            raise ValueError(f"A keyed hand's bone {name!r} is not in the rig.")
        pose[0, labels.index(name + side), :3, :3] = turn(value)
    if key.get("fingers"):
        library = _library_rotations(model, key.get("library", KEY_FIST))
        for finger, joints in key["fingers"].items():
            if finger not in FINGERS or len(joints) != 3:
                raise ValueError(
                    f"A keyed finger is one of {FINGERS} with three joints; got {finger!r}."
                )
            bones = [labels.index(f"{finger}-{k}{side}") for k in (1, 2, 3)]
            axes = roma.rotmat_to_rotvec(torch.as_tensor(library[bones])).numpy()
            norms = np.linalg.norm(axes, axis=1)
            for k, (b, value) in enumerate(zip(bones, joints)):
                if np.ndim(value) == 0:
                    axis = axes[k] if norms[k] > 1e-3 else axes[int(np.argmax(norms))]
                    value = float(value) * axis / np.linalg.norm(axis)
                pose[0, b, :3, :3] = turn(value)


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """The turn (3, 3) that best takes the points P (k, 3) to Q, both about the origin."""
    U, _, Vt = np.linalg.svd(P.T @ Q)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def carry_with(o: dict, before: np.ndarray, after: np.ndarray) -> None:
    """Move an object entry (masks' frame) in place with the rigid motion that best takes the
    points ``before`` (k, 3, anny's frame) to ``after``: a stone with the hand that holds it."""
    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    a, b = before.mean(0), after.mean(0)
    R = _kabsch(before - a, after - b)
    for k in ("centre", "start", "end"):
        if k in o:
            p = (_to_anny(o[k]) - a) @ R.T + b
            o[k] = (p @ GLTF_TO_ANNY).round(6).tolist()
    if "axis" in o:
        o["axis"] = ((_to_anny(o["axis"]) @ R.T) @ GLTF_TO_ANNY).round(6).tolist()


def pose_hands(model, character, rotation, translation, hands: dict, log=None):
    """
    The card with the fingers of each hand in ``hands`` (``{"hand.r": {"objects": [...],
    "fingers": "mh_boxing", "stone": {"radii": [a, b, c]}}}``) curled round the objects it
    holds: each finger (the thumb included) takes its library pose's curl, its root and its
    two outer joints each scaled by one of :data:`CURLS`, the pair that brings the finger
    closest to the objects with all its vertices :data:`MARGIN` outside them (the most
    curled pair of those within a millimetre of it); a hand that holds nothing takes the
    largest curl.
    A hand with a ``stone`` first gets the stone, placed in its palm with the fingers open
    (:func:`place_stone`), among its objects. A hand with ``key`` takes that pose instead of
    the curl's (:func:`key_hand`). A hand with ``clear_body`` (metres) then moves out of the
    body (:func:`_clear_body`), and its stone moves with it. Returns the card and a report
    per hand (the curl of each finger (root, outer joints) or ``"key"``, its clearance in
    mm, and the stone).
    """
    import roma
    import torch

    from opensculptboy.character import Character
    from opensculptboy.statues.chains import (
        _identity_pose,
        _library_rotations,
        _Skinner,
    )

    character = Character.from_dict(character.to_dict())
    labels = list(model.bone_labels)
    pose = character.pose_parameters(model)
    pose = (_identity_pose(model) if pose is None else pose).detach().clone()
    skin = _Skinner(model, character, rotation, translation)
    weights = model.vertex_bone_weights.detach().cpu().numpy()
    indices = model.vertex_bone_indices.cpu().numpy()
    strongest = indices[np.arange(len(indices)), weights.argmax(1)]
    report = {}
    for chain, spec in hands.items():
        side = "." + _side(chain)
        library = _library_rotations(model, spec.get("fingers", FIST))
        fingers = {
            f: [labels.index(f"{f}-{k}{side}") for k in (1, 2, 3)] for f in FINGERS
        }
        objects = list(spec.get("objects", []))
        stone = None
        if spec.get("stone"):
            opened = pose.clone()
            curled = pose.clone()
            for bones in fingers.values():
                opened[0, bones, :3, :3] = torch.eye(3, dtype=pose.dtype)
                curled[0, bones, :3, :3] = torch.as_tensor(
                    library[bones], dtype=pose.dtype
                )
            with torch.no_grad():
                skin(curled)
                # the palm's side: where the library's curled middle finger goes
                tip = skin.joints.numpy()[labels.index("finger3-3" + side)]
                skin(opened)
            J = skin.joints.numpy()
            stone = place_stone(
                {
                    "wrist": J[labels.index("wrist" + side)],
                    "index": J[labels.index("finger2-1" + side)],
                    "middle": J[labels.index("finger3-1" + side)],
                    "little": J[labels.index("finger5-1" + side)],
                    "tip": tip,
                },
                spec["stone"]["radii"],
                float(spec["stone"].get("gap", 0.02)),
            )
            stone["name"] = spec["stone"].get("name", "Stone")
            objects.append(stone)
        margin = float(spec.get("margin", MARGIN))
        if spec.get("key"):
            key_hand(model, pose, chain, spec["key"])
            with torch.no_grad():
                X = skin(pose).numpy()
            report[chain] = {
                "wrist_turn_deg": None,
                "curl": "key",
                "clearance_mm": {
                    f: round(
                        1000
                        * float(
                            signed_distance(X[np.isin(strongest, bones)], objects).min()
                        ),
                        2,
                    )
                    if objects
                    else None
                    for f, bones in fingers.items()
                },
                "stone": stone,
            }
            if log:
                log(
                    f"hand {chain}: keyed, clearance (mm) {report[chain]['clearance_mm']}"
                )
            continue
        turn = None
        if spec.get("objects") and spec.get("place", True):
            turn = _place_wrist(
                skin,
                pose,
                labels,
                strongest,
                side,
                fingers,
                library,
                objects,
                margin,
                spec,
            )
            wrist = labels.index("wrist" + side)
            pose[0, wrist, :3, :3] = pose[0, wrist, :3, :3] @ torch.as_tensor(
                turn, dtype=pose.dtype
            )
        rotvecs = {
            f: roma.rotmat_to_rotvec(torch.as_tensor(library[bones], dtype=pose.dtype))
            for f, bones in fingers.items()
        }
        # each finger's root (s1) and its two outer joints (s2) curl apart: a fist closes
        # round a grip with the knuckle less bent than the finger's joints
        curls = tuple(spec.get("curls", CURLS))
        grid = [(a, b) for a in curls for b in curls]
        clear = {f: [] for f in FINGERS}
        for s1, s2 in grid:
            trial = pose.clone()
            for f, bones in fingers.items():
                rv = rotvecs[f]
                trial[0, bones[0], :3, :3] = roma.rotvec_to_rotmat(s1 * rv[0])
                trial[0, bones[1:], :3, :3] = roma.rotvec_to_rotmat(s2 * rv[1:])
            with torch.no_grad():
                X = skin(trial).numpy()
            for f, bones in fingers.items():
                rows = np.isin(strongest, bones)
                clear[f].append(float(signed_distance(X[rows], objects).min()))
        chosen = {}
        for f, bones in fingers.items():
            c = np.asarray(clear[f])
            ok = np.nonzero(c >= margin)[0]
            if not len(objects) or not len(ok):
                k = len(grid) - 1 if not len(objects) else 0
            else:
                # the closest to touching; within a millimetre of it, the most curled
                best = c[ok].min()
                near = ok[c[ok] <= best + 0.001]
                k = near[np.argmax([sum(grid[n]) for n in near])]
            s1, s2 = grid[k]
            chosen[f] = ((float(s1), float(s2)), round(1000 * float(c[k]), 2))
            rv = rotvecs[f]
            pose[0, bones[0], :3, :3] = roma.rotvec_to_rotmat(float(s1) * rv[0])
            pose[0, bones[1:], :3, :3] = roma.rotvec_to_rotmat(float(s2) * rv[1:])
        report[chain] = {
            "wrist_turn_deg": None
            if turn is None
            else round(
                float(np.degrees(np.arccos(np.clip((np.trace(turn) - 1) / 2, -1, 1)))),
                1,
            ),
            "curl": {f: c[0] for f, c in chosen.items()},
            "clearance_mm": {f: c[1] for f, c in chosen.items()},
            "stone": stone,
        }
        if spec.get("clear_body"):
            hand = [
                labels.index(n)
                for n in labels
                if n.endswith(side) and n.startswith(HAND_BONES)
            ]
            with torch.no_grad():
                skin(pose)
            before = skin.joints.numpy()[hand]
            report[chain]["body"] = _clear_body(
                skin, pose, labels, strongest, side, float(spec["clear_body"])
            )
            with torch.no_grad():
                skin(pose)
            if stone is not None:
                # the stone in the palm goes where the hand goes
                carry_with(stone, before, skin.joints.numpy()[hand])
        if log:
            log(
                f"hand {chain}: curls {report[chain]['curl']}, clearance (mm) "
                f"{report[chain]['clearance_mm']}, body {report[chain].get('body')}"
            )
    character.pose = Character.pose_from_parameters(model, pose)
    return character, report


# the turns (degrees) of the wrist that the placement tries about each axis, coarse then fine
PLACE_REACH = 40.0
PLACE_STEP = 10.0
PLACE_FINE = (9.0, 3.0)


def _place_wrist(
    skin, pose, labels, strongest, side, fingers, library, objects, margin, spec
):
    """The turn (3, 3) of the wrist that puts the held objects in the library fist: the
    four fingers at their library curl closest to touching the objects without entering
    them, and the palm outside them. A search over turns about each axis, coarse then
    fine, within ``spec["reach"]`` degrees (default :data:`PLACE_REACH`) and then
    ``spec["fine"]`` (reach, step; default :data:`PLACE_FINE`)."""
    import roma
    import torch

    wrist = labels.index("wrist" + side)
    fist = pose.clone()
    for bones in fingers.values():
        fist[0, bones, :3, :3] = torch.as_tensor(library[bones], dtype=pose.dtype)
    palm = np.isin(
        strongest,
        [
            i
            for i, b in enumerate(labels)
            if b.endswith(side) and b.startswith(("wrist", "metacarpal"))
        ],
    )
    rows = {f: np.isin(strongest, fingers[f]) for f in FINGERS[1:]}
    base = fist[0, wrist, :3, :3].clone()

    def cost(turn):
        trial = fist.clone()
        trial[0, wrist, :3, :3] = base @ torch.as_tensor(turn, dtype=pose.dtype)
        with torch.no_grad():
            X = skin(trial).numpy()
        total = 0.0
        for f, r in rows.items():
            c = float(signed_distance(X[r], objects).min())
            total += abs(c - margin) + 10.0 * max(0.0, margin - c)
        c = float(signed_distance(X[palm], objects).min())
        return total + 10.0 * max(0.0, margin - c)

    def grid(reach, step):
        a = np.radians(np.arange(-reach, reach + 1e-9, step))
        v = np.stack(np.meshgrid(a, a, a), -1).reshape(-1, 3)
        return roma.rotvec_to_rotmat(torch.as_tensor(v)).numpy()

    reach = float(spec.get("reach", PLACE_REACH))
    best, best_cost = np.eye(3), cost(np.eye(3))
    for turn in grid(reach, PLACE_STEP):
        c = cost(turn)
        if c < best_cost:
            best, best_cost = turn, c
    coarse = best
    for turn in grid(*spec.get("fine", PLACE_FINE)):
        c = cost(coarse @ turn)
        if c < best_cost:
            best, best_cost = coarse @ turn, c
    return best


# the arm's bones that may move a hand off the body, and the steps of that solve
CLEAR_BONES = ("shoulder01", "upperarm01", "lowerarm01")
CLEAR_STEPS = 300


def _clear_body(skin, pose, labels, strongest, side, margin):
    """Lift a hand that rests on the body (a fist on the hip) out of it: the shoulder's, the
    upper arm's and the elbow's turns (the wrist keeps the chain's angle) move the hand until
    its vertices lie ``margin`` outside the body's surface around it, as a contact solve
    would, paying for each turn away from the chain's pose. Returns the hand's vertices
    inside the body and the deepest of them (mm), before and after."""
    import roma
    import torch
    from scipy.spatial import cKDTree

    from opensculptboy.fit.reference import vertex_normals

    arm = (
        "shoulder",
        "clavicle",
        "upperarm",
        "lowerarm",
        "wrist",
        "metacarpal",
        "finger",
    )
    hand_bones_ = [
        i
        for i, n in enumerate(labels)
        if n.endswith(side) and n.startswith(("wrist", "metacarpal", "finger"))
    ]
    arm_bones = [
        i for i, n in enumerate(labels) if n.endswith(side) and n.startswith(arm)
    ]
    hand = np.isin(strongest, hand_bones_)
    faces = skin.model.faces.cpu().numpy()
    with torch.no_grad():
        X = skin(pose).numpy()
    centre = X[hand].mean(0)
    body = ~np.isin(strongest, arm_bones) & (np.linalg.norm(X - centre, axis=1) < 0.15)
    body_rows = np.nonzero(body)[0]
    tree = cKDTree(X[body_rows])
    normals = vertex_normals(X, faces)[body_rows]
    points = X[body_rows]

    def inside(V):
        _, k = tree.query(V[hand])
        depth = ((V[hand] - points[k]) * normals[k]).sum(1)
        return int((depth < 0).sum()), round(1000 * float(max(0.0, -depth.min())), 1)

    before = inside(X)
    bone = {k: labels.index(k + side) for k in CLEAR_BONES if k + side in labels}
    rest = {k: pose[0, i, :3, :3].clone() for k, i in bone.items()}
    x = torch.zeros(3 * len(bone), dtype=pose.dtype, requires_grad=True)

    def posed(x):
        p = pose.clone()
        for n, (k, i) in enumerate(bone.items()):
            p[0, i, :3, :3] = rest[k] @ roma.rotvec_to_rotmat(x[3 * n : 3 * n + 3])
        return p

    optimizer = torch.optim.Adam([x], lr=0.01)
    rows = np.nonzero(hand)[0]
    for step in range(CLEAR_STEPS):
        if step % 50 == 0:
            with torch.no_grad():
                V = skin(posed(x)).numpy()
            _, k = tree.query(V[rows])
            q = torch.as_tensor(points[k], dtype=pose.dtype)
            n = torch.as_tensor(normals[k], dtype=pose.dtype)
        optimizer.zero_grad()
        V = skin(posed(x))[rows]
        depth = ((V - q) * n).sum(1)
        loss = (torch.relu(margin - depth) ** 2).sum() / 1e-6 + 0.3 * (x**2).sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        moved = posed(x)
        X = skin(moved).numpy()
    for k, i in bone.items():
        pose[0, i, :3, :3] = moved[0, i, :3, :3]
    return {
        "inside_before": before[0],
        "deepest_before_mm": before[1],
        "inside_after": inside(X)[0],
        "deepest_after_mm": inside(X)[1],
        "arm_turn_deg": round(float(np.degrees(x.detach().norm())), 1),
    }


def add_stone_prop(folder, stone: dict, role: str = "support") -> None:
    """Add a hand's stone (:func:`stone_mesh`) to a handoff's props (``props.glb`` and
    ``props.json``) under its name, replacing a prop of that name."""
    import json
    import pathlib

    import trimesh

    folder = pathlib.Path(folder)
    name = stone.get("name", "Stone")
    path = folder / "props.glb"
    scene = trimesh.Scene()
    if path.exists():
        old = trimesh.load(str(path), force="scene")
        for n, g in old.geometry.items():
            if n != name:
                scene.add_geometry(g, node_name=n, geom_name=n)
    scene.add_geometry(stone_mesh(stone), node_name=name, geom_name=name)
    scene.export(str(path))
    info_path = folder / "props.json"
    info = (
        json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    )
    props = [p for p in info.get("props", []) if p.get("name") != name]
    props.append({"name": name, "role": role, "stone": stone})
    info["props"] = props
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

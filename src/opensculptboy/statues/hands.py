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
(:func:`stone_mesh`). A hand that holds a prop's grip (``hold``) closes the other way round:
its fist keeps a library pose that holds a rod, the prop moves so that its grip lies along
the fist's channel (:func:`fist_channel`), and each finger opens until it clears the grip.
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
    (:func:`place_stone`), among its objects. A hand with ``hold`` (``{"prop": "Sword",
    "tip": [x, y, z]}``, and ``"fingers"`` a fist that holds a rod) holds the prop whose grip
    is the capsule marked ``"grip": true`` among its objects, by :func:`_hold`. Returns the
    card and a report per hand (the curl of each finger (root, outer joints), its clearance
    in mm, and the stone).
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
        if spec.get("hold"):
            report[chain] = _hold(
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
            if log:
                r = report[chain]
                log(
                    f"hand {chain}: holds {r['carry']['prop']}, wrist {r['wrist_turn_deg']} "
                    f"deg, arm {r['arm_turns_deg']}, grasp off {r['grasp_off_mm']} mm, "
                    f"prop turn {r['prop_turn_deg']} deg, tip moved {r['tip_moved_mm']} mm, "
                    f"curls {r['curl']}, "
                    f"clearance (mm) {r['clearance_mm']}, wrap {r['wrap_deg']}"
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
            "carry": None,
        }
        if spec.get("clear_body"):
            report[chain]["body"] = _clear_body(
                skin, pose, labels, strongest, side, float(spec["clear_body"])
            )
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


# a finger's loop in the library fist: the largest radius (metres) that counts as one, and
# the place (in the hand's frame from the knuckle: towards the palm, towards the fingers,
# across) taken for a finger without one
LOOP_LARGEST = 0.04
LOOP_DEFAULT = np.array([0.028, -0.016, 0.0])
# the room (metres) between the hand and the guard or the collar
GRASP_ROOM = 0.02
# the wrist's largest turn from straight (degrees) before the arm's solve pays for it, and the
# solve's steps
WRIST_LIMIT = 30.0
IK_STEPS = 3000
# the arm's solve: what each turn costs (per radian: the shoulder, the upper arm, the elbow,
# each about three axes, the forearm's twist and the wrist's swing about two; the wrist and
# the forearm cost more, so that the arm, not the wrist, carries the hand to the grip)
REG_WEIGHTS = np.array([1.0] * 3 + [0.5] * 3 + [0.5] * 3 + [1.0] + [1.5] * 2)
# how hard the solve pulls the little finger's end down onto the guard's top (``low``; the
# fist low on the grip); 0: anywhere on the grip's free length
LOW_WEIGHT = 0.0
# the forearm's twists (degrees) the arm's solve starts from, near its chain's pose
IK_TWISTS = (0.0, 45.0, -45.0)
# the largest turn (degrees) of a held prop about its tip and its largest shift (metres)
HOLD_TURN = 3.0
HOLD_SHIFT = 0.03
# a holding hand's fingers close in steps of this share of the library fist's turns, up to
# the largest share
HOLD_STEP = 0.05
HOLD_CURL_MAX = 1.6


def _tip_rows(X, J, labels, strongest, side) -> dict:
    """The vertex of each finger's tip: the farthest from its last joint among the vertices
    of its last bone."""
    out = {}
    for f in FINGERS:
        last = labels.index(f"{f}-3{side}")
        rows = np.nonzero(strongest == last)[0]
        out[f] = int(rows[np.argmax(np.linalg.norm(X[rows] - J[last], axis=1))])
    return out


def _rotation_between(u, v) -> np.ndarray:
    """The smallest turn (3, 3) that takes the unit vector u to v."""
    w = np.cross(u, v)
    s, c = np.linalg.norm(w), float(u @ v)
    if s < 1e-12:
        return np.eye(3)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / s
    angle = np.arctan2(s, c)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * K @ K


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """The turn (3, 3) that best takes the points P (k, 3) to Q, both about the origin."""
    U, _, Vt = np.linalg.svd(P.T @ Q)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def _turn_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """The turn (3, 3) by ``angle`` (radians) about ``axis``."""
    k = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * K @ K


def _local_axes(skin, pose, bone: int, followers) -> np.ndarray:
    """The world directions (columns, 3 x 3) of a bone's own turning axes: three probe turns
    about them show where each points, by the turn of the joints that follow the bone."""
    import roma
    import torch

    with torch.no_grad():
        skin(pose)
    J0 = skin.joints.numpy()[followers]
    head = skin.joints.numpy()[bone]
    axes = []
    for k in range(3):
        probe = pose.clone()
        e = torch.zeros(3, dtype=pose.dtype)
        e[k] = 0.5
        probe[0, bone, :3, :3] = pose[0, bone, :3, :3] @ roma.rotvec_to_rotmat(e)
        with torch.no_grad():
            skin(probe)
        turn = _kabsch(J0 - head, skin.joints.numpy()[followers] - head)
        axes.append(roma.rotmat_to_rotvec(torch.as_tensor(turn)).numpy())
    U, _, Vt = np.linalg.svd(np.stack(axes, 1))
    return U @ Vt


def _circle(points: np.ndarray):
    """The centre (3,) and radius of the circle that best fits points (k, 3), in the plane
    that best fits them."""
    c = points.mean(0)
    _, _, vt = np.linalg.svd(points - c)
    Q = np.c_[(points - c) @ vt[0], (points - c) @ vt[1]]
    A = np.c_[2.0 * Q, np.ones(len(Q))]
    x = np.linalg.lstsq(A, (Q**2).sum(1), rcond=None)[0]
    radius = float(np.sqrt(max(x[2] + x[0] ** 2 + x[1] ** 2, 0.0)))
    return c + x[0] * vt[0] + x[1] * vt[1], radius


def _hand_frame(J, ids, sign):
    """The hand's frame at its knuckles (rows: towards the palm, from the wrist to the
    knuckles, from the index finger to the little finger), from its joints J."""
    index, little, wrist = J[ids["index"]], J[ids["little"]], J[ids["wrist"]]
    knuckles = (index + J[ids["middle"]] + J[ids["ring"]] + little) / 4.0
    along = knuckles - wrist
    across = little - index
    normal = sign * np.cross(-across, along)
    normal /= np.linalg.norm(normal)
    along = along - (along @ normal) * normal
    along /= np.linalg.norm(along)
    return np.stack([normal, along, np.cross(normal, along)])


def _grasp_points(J, ids, sign, local):
    """The ends of a hand's power grasp: the points ``local`` ((2, 3), in the hand's frame,
    :func:`_hand_frame`) from the index finger's and the little finger's knuckles."""
    F = _hand_frame(J, ids, sign)
    return J[ids["index"]] + local[0] @ F, J[ids["little"]] + local[1] @ F


def _fist_loops(skin, fist, labels, strongest, side, ids, sign, radius, margin):
    """Where a hand's power grasp holds a grip, from the library fist ``fist`` (a pose):
    the centres of the loops that the index finger and the little finger close round (each
    the circle through the finger's three joints and its tip), in the hand's frame from
    their knuckles, moved out from the palm as far as a grip of ``radius`` needs to clear
    the open hand (palm and straight fingers but the thumb) by ``margin``, so that the
    fingers close round it. Returns the two points (2, 3) in the hand's frame."""
    import torch

    with torch.no_grad():
        X = skin(fist).numpy()
    J = skin.joints.numpy()
    tips = _tip_rows(X, J, labels, strongest, side)
    F = _hand_frame(J, ids, sign)
    centres = []
    for f, knuckle in (("finger2", "index"), ("finger5", "little")):
        points = [J[labels.index(f"{f}-{k}{side}")] for k in (1, 2, 3)] + [X[tips[f]]]
        centre, loop = _circle(np.array(points))
        if not loop < LOOP_LARGEST:
            # a finger the fist leaves nearly straight closes round no loop: the default
            centre = J[ids[knuckle]] + LOOP_DEFAULT @ F
        centres.append(centre)
    # the open hand (its fingers straight, as the closing starts) must clear the grip
    opened = fist.clone()
    for i, n in enumerate(labels):
        if n.endswith(side) and n.startswith("finger"):
            opened[0, i, :3, :3] = torch.eye(3, dtype=fist.dtype)
    with torch.no_grad():
        Y = skin(opened).numpy()
    hand = Y[
        np.isin(
            strongest,
            [
                i
                for i, n in enumerate(labels)
                if n.endswith(side)
                and n.startswith(
                    ("wrist", "metacarpal", "finger2", "finger3", "finger4", "finger5")
                )
            ],
        )
    ]
    a, b = centres
    for _ in range(8):
        ab = b - a
        # the grip runs on past the hand: its whole line must clear the hand
        t = ((hand - a) @ ab) / (ab @ ab)
        nearest = float(np.linalg.norm(hand - (a + t[:, None] * ab), axis=1).min())
        push = radius + margin - nearest
        if push <= 1e-4:
            break
        a, b = a + push * F[0], b + push * F[0]
    out = np.array([(a - J[ids["index"]]) @ F.T, (b - J[ids["little"]]) @ F.T])
    return out


def _hold(skin, pose, labels, strongest, side, fingers, library, objects, margin, spec):
    """A hand that holds a prop's grip in a power grasp.

    The grasp: the grip runs through the loops that the library fist's (``spec["fingers"]``)
    index finger and little finger close round (:func:`_fist_loops`), as a power grip
    holds a handle across the palm, moved out from the palm as far as the grip needs; the
    index finger's end and the grasp's middle lie on the grip's free length
    (:data:`GRASP_ROOM` under the collar), the index finger up the grip towards the
    pommel and the little finger just above the guard. The arm
    reaches it: a damped least-squares solve (Levenberg-Marquardt, as inverse kinematics
    solves a limb) of the shoulder's, the upper arm's and the elbow's turns, the forearm's
    twist (in its twist bone, lowerarm02, shared along the forearm, so that the wrist does
    not kink) and the wrist's swing, which brings the grasp's ends onto the grip's axis and
    pays for moving away from the chain's pose and for a wrist turned more than
    :data:`WRIST_LIMIT` degrees. The prop then turns about ``spec["hold"]["tip"]`` (up to
    :data:`HOLD_TURN` degrees) and shifts (up to :data:`HOLD_SHIFT`) for the rest. The
    fingers then close as GraspIt's AutoGrasp closes a hand, and the thumb opposes first.
    ``pose`` takes the arm and the fingers. Returns the hand's report, whose ``carry`` moves
    the prop (:func:`move_prop`)."""
    import roma
    import torch

    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    M = GLTF_TO_ANNY
    hold = spec["hold"]
    grips = [o for o in objects if o.get("grip")]
    if not grips:
        raise ValueError("A hand that holds a prop needs a capsule marked 'grip'.")
    grip = grips[0]
    radius = float(grip["radius"])
    a0, b0 = _to_anny(grip["start"]), _to_anny(grip["end"])
    up = (b0 - a0) / np.linalg.norm(b0 - a0)
    # the grasp's top under the lowest other object along the grip above its start (the
    # collar), or at the grip's end
    ceiling = np.linalg.norm(b0 - a0)
    for o in objects:
        if o is grip or o.get("kind") != "capsule":
            continue
        for key in ("start", "end"):
            t = (_to_anny(o[key]) - a0) @ up
            if (
                0.0 < t < ceiling + 0.2
                and np.linalg.norm((_to_anny(o[key]) - a0) - t * up) < 0.01
            ):
                ceiling = min(ceiling, t - float(o["radius"]))
    room = float(hold.get("room", GRASP_ROOM))
    # and above the guard: the highest other object across the grip near its start
    floor = 0.0
    for o in objects:
        if o is grip or o.get("kind") != "capsule":
            continue
        s0, s1 = _to_anny(o["start"]), _to_anny(o["end"])
        mid = 0.5 * (s0 + s1)
        t = (mid - a0) @ up
        across = np.linalg.norm((mid - a0) - t * up)
        if abs(t) < 0.03 and across < 0.05:
            floor = max(floor, t + float(o["radius"]) + room)
    ids = {
        "wrist": labels.index("wrist" + side),
        "index": labels.index("finger2-1" + side),
        "middle": labels.index("finger3-1" + side),
        "ring": labels.index("finger4-1" + side),
        "little": labels.index("finger5-1" + side),
    }
    # the palm's side: where the library fist's middle finger curls to
    fist = pose.clone()
    for bones in fingers.values():
        fist[0, bones, :3, :3] = torch.as_tensor(library[bones], dtype=pose.dtype)
    with torch.no_grad():
        skin(fist)
    Jf = skin.joints.numpy()
    with torch.no_grad():
        skin(pose)
    J0 = skin.joints.numpy()
    tip_curl = Jf[labels.index("finger3-3" + side)] - Jf[ids["middle"]]
    normal0 = np.cross(
        J0[ids["index"]] - J0[ids["little"]], J0[ids["middle"]] - J0[ids["wrist"]]
    )
    sign = 1.0 if tip_curl @ normal0 > 0 else -1.0
    local = _fist_loops(skin, fist, labels, strongest, side, ids, sign, radius, margin)
    # 2. the arm's solve, in world turns about each joint of the chain's pose (outer joints
    # first, so that each inner turn carries the outer joints with it); a bone's own turn is
    # the world turn seen in its own axes (_local_axes), which an inner turn does not change
    names = ("shoulder01", "upperarm01", "lowerarm01", "lowerarm02", "wrist")
    bone = {k: labels.index(k + side) for k in names if k + side in labels}
    hand_joints = [
        i
        for i, n in enumerate(labels)
        if n.endswith(side) and n.startswith(("metacarpal", "finger"))
    ]
    axes = {}
    for k, i in bone.items():
        followers = [j for j in [bone["wrist"]] + hand_joints if j != i]
        axes[k] = _local_axes(skin, pose, i, followers)
    with torch.no_grad():
        skin(pose)
    J0 = skin.joints.numpy()
    head = {k: J0[i] for k, i in bone.items()}
    forearm = head["wrist"] - head["lowerarm02"]
    forearm /= np.linalg.norm(forearm)
    hand_dir = J0[ids["middle"]] - head["wrist"]
    hand_dir /= np.linalg.norm(hand_dir)
    swing = np.linalg.svd(hand_dir[None])[2][1:]  # two world axes across the hand
    rows = [ids["wrist"], ids["index"], ids["middle"], ids["ring"], ids["little"]]
    points0 = J0[rows]
    limit_rad = np.radians(float(hold.get("wrist_limit", WRIST_LIMIT)))
    wrist_rest = pose[0, bone["wrist"], :3, :3].numpy()
    free = [k for k in ("shoulder01", "upperarm01", "lowerarm01") if k in bone]

    def rot(v):
        angle = np.linalg.norm(v)
        return np.eye(3) if angle < 1e-12 else _turn_about(v, angle)

    def turns(values):
        out = {k: rot(values[3 * n : 3 * n + 3]) for n, k in enumerate(free)}
        out["lowerarm02"] = rot(values[9] * forearm)
        out["wrist"] = rot(values[10:12] @ swing)
        return out

    def forward(values):
        W = turns(values)
        P = points0.copy()
        P[1:] = (P[1:] - head["wrist"]) @ W["wrist"].T + head["wrist"]
        for k in ("lowerarm02", "lowerarm01", "upperarm01", "shoulder01"):
            if k in W:
                P = (P - head[k]) @ W[k].T + head[k]
        J = J0.copy()
        J[rows] = P
        return J, W

    def wrist_angle(W):
        local = wrist_rest @ (axes["wrist"].T @ W["wrist"] @ axes["wrist"])
        return float(np.arccos(np.clip((np.trace(local) - 1.0) / 2.0, -1.0, 1.0)))

    def residuals(values):
        J, W = forward(values)
        top, bottom = _grasp_points(J, ids, sign, local)
        out = []
        for point in (top, bottom):
            q = point - a0
            out.append((q - (q @ up) * up) / 0.01)  # off the grip's axis
        # the index finger's end up the grip (towards the pommel), the little finger's down
        # (towards the guard), as a hand holds a sword point down; both on the grip's free
        # length, the little finger's end just above the guard (low on the grip)
        t_top, t_bottom = (top - a0) @ up, (bottom - a0) @ up
        out.append([max(0.0, t_bottom - t_top) / 0.005])
        out.append([max(0.0, t_top - (ceiling - room)) / 0.01])
        out.append([max(0.0, floor - t_bottom) / 0.01])
        out.append([(t_bottom - floor) / 0.01 * float(hold.get("low", LOW_WEIGHT))])
        out.append(REG_WEIGHTS * np.asarray(values))
        out.append([np.sqrt(30.0) * max(0.0, wrist_angle(W) - limit_rad)])
        return np.concatenate(out)

    # a damped least-squares solve (Levenberg-Marquardt, as inverse kinematics solves a limb)
    # from the chain's pose with the forearm twisted three ways near it; the best is kept
    from scipy.optimize import least_squares

    best = None
    for start in IK_TWISTS:
        x0 = np.zeros(12)
        x0[9] = np.radians(start)
        fit = least_squares(
            residuals, x0, method="lm", max_nfev=int(hold.get("steps", IK_STEPS))
        )
        value = float((fit.fun**2).sum())
        if best is None or value < best[0]:
            best = (value, fit.x)
    x = best[1]
    W = turns(x)
    for k, i in bone.items():
        own = axes[k].T @ W[k] @ axes[k]
        pose[0, i, :3, :3] = pose[0, i, :3, :3] @ torch.as_tensor(own, dtype=pose.dtype)
    with torch.no_grad():
        skin(pose)
    J = skin.joints.numpy()
    top, bottom = _grasp_points(J, ids, sign, local)
    solved = forward(x)[0]
    fk_error = float(np.abs(solved[rows] - J[rows]).max())
    # 3. the prop turns about its tip, then shifts, for what the arm left
    tip = _to_anny(hold["tip"]) if hold.get("tip") is not None else a0.copy()
    middle = 0.5 * (top + bottom)
    near = a0 + ((middle - a0) @ up) * up
    turn = _rotation_between(
        (near - tip) / np.linalg.norm(near - tip),
        (middle - tip) / np.linalg.norm(middle - tip),
    )
    size = np.arccos(np.clip((np.trace(turn) - 1.0) / 2.0, -1.0, 1.0))
    largest = np.radians(float(hold.get("turn", HOLD_TURN)))
    if size > largest:
        axis = roma.rotmat_to_rotvec(torch.as_tensor(turn)).numpy()
        turn = roma.rotvec_to_rotmat(torch.as_tensor(axis / size * largest)).numpy()
    a1 = tip + turn @ (a0 - tip)
    u1 = turn @ up
    near = a1 + ((middle - a1) @ u1) * u1
    shift = middle - near
    reach = float(hold.get("shift", HOLD_SHIFT))
    if np.linalg.norm(shift) > reach:
        shift *= reach / np.linalg.norm(shift)
    carry = {
        "prop": hold.get("prop"),
        "pivot": (tip @ M).tolist(),
        "rotation": (M.T @ turn @ M).tolist(),
        "shift": (M.T @ shift).tolist(),
    }
    moved = [move_object(o, carry) for o in objects]
    # 4. the fingers close as GraspIt's AutoGrasp closes a hand (Miller and Allen 2004): the
    # joints of each finger turn together at the library fist's rates; when a phalanx comes
    # within the margin of the prop, the step is taken back and its joint and those before
    # it stop, and the joints after it go on. The thumb first opposes (its first joint
    # alone, as far as the library fist's, or less until it clears the prop), then curls the
    # same way.
    rotvecs = {
        f: roma.rotmat_to_rotvec(torch.as_tensor(library[bones], dtype=pose.dtype))
        for f, bones in fingers.items()
    }
    step = float(hold.get("curl_step", HOLD_STEP))
    curl_cap = float(hold.get("curl_max", HOLD_CURL_MAX))
    scale = {f: [0.0, 0.0, 0.0] for f in FINGERS}
    frozen = {f: [False, False, False] for f in FINGERS}
    links = {f: [strongest == b for b in bones] for f, bones in fingers.items()}
    rows = {f: np.isin(strongest, bones) for f, bones in fingers.items()}

    def set_fingers(p):
        for f, bones in fingers.items():
            for k in range(3):
                p[0, bones[k], :3, :3] = roma.rotvec_to_rotmat(
                    scale[f][k] * rotvecs[f][k]
                )
        return p

    def cap(f, k):
        # the thumb opposes as far as the library fist does, no farther
        return 1.0 if (f, k) == ("finger1", 0) else curl_cap

    def close(active):
        """Close the joints of ``active`` ({finger: [joint, ...]}) until each stops."""
        while True:
            moving = {
                f: [k for k in ks if not frozen[f][k] and scale[f][k] < cap(f, k)]
                for f, ks in active.items()
            }
            if not any(moving.values()):
                return
            before = {f: list(v) for f, v in scale.items()}
            for f, ks in moving.items():
                for k in ks:
                    scale[f][k] = min(cap(f, k), scale[f][k] + step)
            with torch.no_grad():
                Y = skin(set_fingers(pose.clone())).numpy()
            for f, ks in moving.items():
                if not ks:
                    continue
                for k in range(3):
                    if not links[f][k].any():
                        continue
                    if float(signed_distance(Y[links[f][k]], moved).min()) < margin:
                        # this phalanx touches: the joints that move it and moved in this
                        # step stop where they were, or, when none did (the skin by the
                        # next joint bends with it), the nearest joint that moved; every
                        # phalanx is checked
                        stop = [j for j in ks if j <= k] or [min(ks)]
                        for j in stop:
                            scale[f][j] = before[f][j]
                            frozen[f][j] = True

    # the thumb's opposition: as far as the library fist's, or back from it (away from the
    # palm, below zero) until the thumb clears the prop
    thumb = links["finger1"]
    thumb_rows = thumb[0] | thumb[1] | thumb[2]
    for s in np.arange(1.0, -1.0001, -step):
        scale["finger1"][0] = float(s)
        with torch.no_grad():
            Y = skin(set_fingers(pose.clone())).numpy()
        if float(signed_distance(Y[thumb_rows], moved).min()) >= margin:
            break
    for f in FINGERS:
        frozen[f] = [False, False, False] if f != "finger1" else [True, False, False]
    close({f: ([1, 2] if f == "finger1" else [0, 1, 2]) for f in FINGERS})
    set_fingers(pose)
    with torch.no_grad():
        Y = skin(pose).numpy()
    clearance = {
        f: round(1000 * float(signed_distance(Y[rows[f]], moved).min()), 2)
        for f in FINGERS
    }
    J = skin.joints.numpy()
    top, bottom = _grasp_points(J, ids, sign, local)
    g = moved[objects.index(grip)]
    ga, gb = _to_anny(g["start"]), _to_anny(g["end"])
    gu = (gb - ga) / np.linalg.norm(gb - ga)

    def off(p):
        q = p - ga
        return float(np.linalg.norm(q - (q @ gu) * gu))

    wrist_turn = roma.rotmat_to_rotvec(pose[0, bone["wrist"], :3, :3]).norm()
    x = torch.as_tensor(x)
    moved_tip = move_object({"centre": (tip @ M).tolist()}, carry)["centre"]
    return {
        "wrist_turn_deg": round(float(np.degrees(float(wrist_turn))), 1),
        "arm_turns_deg": {
            k: round(float(np.degrees(np.linalg.norm(v))), 1)
            for k, v in (
                ("shoulder", x.detach()[0:3].numpy()),
                ("upper arm", x.detach()[3:6].numpy()),
                ("elbow", x.detach()[6:9].numpy()),
                ("forearm twist", x.detach()[9:10].numpy()),
                ("wrist swing", x.detach()[10:12].numpy()),
            )
        },
        "grasp_off_mm": [round(1000 * off(top), 1), round(1000 * off(bottom), 1)],
        "kinematics_mm": round(1000 * fk_error, 2),
        "grasp_local_mm": np.round(1000 * local, 1).tolist(),
        "curl": {f: tuple(round(v, 2) for v in scale[f]) for f in FINGERS},
        "clearance_mm": clearance,
        "stone": None,
        "carry": carry,
        "prop_turn_deg": round(float(np.degrees(min(size, largest))), 2),
        "tip_moved_mm": round(
            1000 * float(np.linalg.norm(_to_anny(moved_tip) - tip)), 1
        ),
        "wrap_deg": _wrap(skin, pose, labels, strongest, side, moved),
    }


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


def _wrap(skin, pose, labels, strongest, side, objects):
    """How far each of the four fingers wraps round a grip (degrees about its axis, from
    the finger's knuckle to its tip; past 180 the tip is beyond the grip's far side)."""
    import torch

    grip = next(o for o in objects if o.get("grip"))
    a, b = _to_anny(grip["start"]), _to_anny(grip["end"])
    axis = (b - a) / np.linalg.norm(b - a)
    with torch.no_grad():
        X = skin(pose).numpy()
    J = skin.joints.numpy()
    tips = _tip_rows(X, J, labels, strongest, side)
    out = {}
    for f in FINGERS[1:]:
        points = [J[labels.index(f"{f}-{k}{side}")] for k in (1, 2, 3)] + [X[tips[f]]]
        flat = [(p - a) - ((p - a) @ axis) * axis for p in points]
        total = 0.0
        for u, v in zip(flat, flat[1:]):
            total += np.degrees(np.arctan2(np.cross(u, v) @ axis, u @ v))
        out[f] = round(abs(float(total)), 1)
    return out


def move_object(o: dict, carry: dict) -> dict:
    """A held object (masks' frame) moved with its prop: ``carry`` turns it by ``rotation``
    round ``pivot`` and shifts it by ``shift``."""
    R = np.asarray(carry["rotation"], np.float64)
    pivot = np.asarray(carry["pivot"], np.float64)
    shift = np.asarray(carry["shift"], np.float64)
    moved = dict(o)
    for key in ("start", "end", "centre"):
        if key in o:
            p = np.asarray(o[key], np.float64)
            moved[key] = ((p - pivot) @ R.T + pivot + shift).tolist()
    if "axis" in o:
        moved["axis"] = (np.asarray(o["axis"], np.float64) @ R.T).tolist()
    return moved


def move_prop(folder, carry: dict) -> None:
    """Move a carried prop (``carry["prop"]``, a name in a handoff's ``props.glb``) as
    :func:`_hold` found it; the other props stay. ``props.json`` keeps the move under
    ``carried``, and a later move first takes the earlier one back, so that a run on the
    same handoff moves the prop from where the scan has it."""
    import json
    import pathlib

    import trimesh

    folder = pathlib.Path(folder)
    path = folder / "props.glb"
    info_path = folder / "props.json"
    info = (
        json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    )
    scene = trimesh.load(str(path), force="scene")
    name = carry["prop"]
    if name not in scene.geometry:
        raise ValueError(f"The carried prop {name!r} is not in {path}.")
    earlier = info.get("carried", {}).get(name)
    out = trimesh.Scene()
    for n, g in scene.geometry.items():
        if n == name:
            g = g.copy()
            V = np.asarray(g.vertices, np.float64)
            if earlier:
                R = np.asarray(earlier["rotation"], np.float64)
                p = np.asarray(earlier["pivot"], np.float64)
                V = (V - np.asarray(earlier["shift"], np.float64) - p) @ R + p
            R = np.asarray(carry["rotation"], np.float64)
            p = np.asarray(carry["pivot"], np.float64)
            g.vertices = (V - p) @ R.T + p + np.asarray(carry["shift"], np.float64)
        out.add_geometry(g, node_name=n, geom_name=n)
    out.export(str(path))
    info.setdefault("carried", {})[name] = {
        k: carry[k] for k in ("prop", "pivot", "rotation", "shift")
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


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

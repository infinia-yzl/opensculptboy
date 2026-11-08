# OpenSculptBoy
# Apache License, Version 2.0
"""
The companion's shared poses (``squire.poses/1``): one file of key poses that the
stage's 3D preview (a VRM in three-vrm) and the layered portrait pack both read, so the
stills show the body exactly as the VRM shows it.

The file names the normalized humanoid bones of VRM 1.0 and gives each bone angles in degrees
about X, then Y, then Z, each in the parent's frame (three.js's Euler order ``ZYX``), with the
figure facing +Z, its left at +X and up at +Y: the axes of the viewer page and of the authoring
rig. Arms and hands are written as for the left side and mirrored onto the right
(``[x, -y, -z]``); presets carry the arms down, the fingers' curls, the thumb and the stances
(the hips, the spine and the legs of a pose, such as the weight on one leg). A pose that turns its
own head (``keepHead: false``) lists in ``faces`` the expressions of its own face set.
``data/portrait/poses.json`` is the command's default file.

The rig of a card has more bones than the humanoid, and it rests in an A pose. A pose turns it as
the VRM export of the same card turns (:mod:`opensculptboy.export.tpose`):

- each humanoid bone maps to its rig bone by the export's table
  (:data:`opensculptboy.export.vrm_tables.HUMANOID_1`);
- each rig bone first turns from its rest into the VRM T pose by its T-pose turn ``U_j``, its
  rest-frame correction, computed for the card's own body (:func:`rest_turns`);
- the humanoid bone ``h`` then turns it by its rotation in the world ``R_h``: the rig bone's
  rotation in the world is ``R_h U_j``. A rig bone outside the humanoid takes the humanoid bone
  above it in the VRM file (:func:`governing`), and the forearm's twist bone takes the roll of
  the VRM file's constraint (half the hand's roll about the forearm). The upper arm's own roll
  is shared the same way (:data:`UPPER_ARM_ROLL`): its first bone keeps half of it and its second
  bone, at mid-arm, takes the rest, so a raised arm's roll no longer twists the armpit's skin at
  the shoulder joint. A roll about the bone's own axis moves no joint, so the skeleton stays the
  VRM's.

The rig bones' places then follow by forward kinematics from their own rest heads, as the
VRM's joints follow from the T pose, so the stills' skeleton lands on the VRM's skeleton.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

FORMAT = "squire.poses/1"
DEFAULT_FILE = (
    pathlib.Path(__file__).resolve().parents[1] / "data" / "portrait" / "poses.json"
)

# VRM 1.0's humanoid bones and their parents (with every optional bone present).
_CENTRE = {
    "hips": None,
    "spine": "hips",
    "chest": "spine",
    "upperChest": "chest",
    "neck": "upperChest",
    "head": "neck",
    "leftEye": "head",
    "rightEye": "head",
    "jaw": "head",
}


def _humanoid_parents() -> dict:
    p = dict(_CENTRE)
    for s in ("left", "right"):
        p.update(
            {
                f"{s}UpperLeg": "hips",
                f"{s}LowerLeg": f"{s}UpperLeg",
                f"{s}Foot": f"{s}LowerLeg",
                f"{s}Toes": f"{s}Foot",
                f"{s}Shoulder": "upperChest",
                f"{s}UpperArm": f"{s}Shoulder",
                f"{s}LowerArm": f"{s}UpperArm",
                f"{s}Hand": f"{s}LowerArm",
                f"{s}ThumbMetacarpal": f"{s}Hand",
                f"{s}ThumbProximal": f"{s}ThumbMetacarpal",
                f"{s}ThumbDistal": f"{s}ThumbProximal",
            }
        )
        for f in ("Index", "Middle", "Ring", "Little"):
            p[f"{s}{f}Proximal"] = f"{s}Hand"
            p[f"{s}{f}Intermediate"] = f"{s}{f}Proximal"
            p[f"{s}{f}Distal"] = f"{s}{f}Intermediate"
    return p


HUMANOID_PARENT = _humanoid_parents()
HUMAN_BONES = tuple(HUMANOID_PARENT)  # parents come first
LIMBS = ("Shoulder", "UpperArm", "LowerArm", "Hand")
THUMB = ("ThumbMetacarpal", "ThumbProximal", "ThumbDistal")
FINGERS = ("Index", "Middle", "Ring", "Little")
# Anny's frame (metres, Z up, the figure facing -Y) to the page's and the VRM file's frame (Y up,
# the figure facing +Z): (x, y, z) -> (x, z, -y), as opensculptboy.export.body.ANNY_TO_GLTF.
ANNY_TO_PAGE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
# the arms' axis in the T pose, about which an arm rolls
_X = np.array([1.0, 0.0, 0.0])


class PosesError(ValueError):
    """A shared pose file that breaks the format."""


def mirror(e) -> tuple:
    """The right side's angles for a rotation written for the left side."""
    return (float(e[0]), -float(e[1]) + 0.0, -float(e[2]) + 0.0)


def euler_matrix(e) -> np.ndarray:
    """The rotation (3, 3) of angles in degrees: about X, then Y, then Z, in the parent's frame."""
    x, y, z = np.radians(np.asarray(e, dtype=np.float64))
    cx, sx, cy, sy, cz, sz = (
        np.cos(x),
        np.sin(x),
        np.cos(y),
        np.sin(y),
        np.cos(z),
        np.sin(z),
    )
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _side(s: str, e) -> tuple:
    return tuple(float(v) for v in e) if s == "left" else mirror(e)


@dataclasses.dataclass
class SharedPoses:
    """The poses of a file, expanded to angles on the humanoid bones."""

    bones: dict  # pose -> {humanoid bone -> (x, y, z) degrees}
    keep: tuple  # the bones that keep the reference pose's turn in the world
    keep_pose: str | None
    own_head: frozenset  # the poses that turn the head themselves
    source: str = ""
    faces: dict = dataclasses.field(
        default_factory=dict
    )  # pose -> its own face set's expressions

    @property
    def names(self) -> list:
        return list(self.bones)

    def local(self, name: str) -> dict:
        """The local rotations (3, 3) of a pose's humanoid bones, with the kept head."""
        if name not in self.bones:
            raise KeyError(
                f"The pose file holds no pose {name!r}; it holds {self.names}."
            )
        rot = {b: euler_matrix(e) for b, e in self.bones[name].items()}
        if not self.keep or self.keep_pose is None or name in self.own_head:
            return rot
        want = world(
            {b: euler_matrix(e) for b, e in self.bones[self.keep_pose].items()}
        )
        for b in self.keep:
            parent = HUMANOID_PARENT[b]
            pw = world(rot)[parent] if parent else np.eye(3)
            rot[b] = pw.T @ want[b]
        return rot


def world(local: dict) -> dict:
    """Each humanoid bone's rotation in the world, from local rotations (the others at rest)."""
    out = {}
    for b in HUMAN_BONES:
        own = local.get(b, np.eye(3))
        p = HUMANOID_PARENT[b]
        out[b] = own if p is None else out[p] @ own
    return out


def _hand(data: dict, s: str, h: dict, where: str) -> dict:
    out = {}
    curl = h.get("curl")
    if isinstance(curl, str):
        if curl not in data.get("fingers", {}):
            raise PosesError(f"{where}: no finger preset {curl!r}.")
        curl = data["fingers"][curl]
    if not (isinstance(curl, list) and len(curl) == 3):
        raise PosesError(f"{where}: a hand's curl is a preset's name or three angles.")
    spread = float(h.get("spread", 0))
    for i, f in enumerate(FINGERS):
        fan = spread * (1.5 - i)
        out[f"{s}{f}Proximal"] = _side(s, (0, -fan, -curl[0]))
        out[f"{s}{f}Intermediate"] = _side(s, (0, 0, -curl[1]))
        out[f"{s}{f}Distal"] = _side(s, (0, 0, -curl[2]))
    thumb = float(h.get("thumb", 0))
    k, open_ = max(0.0, thumb), max(0.0, -thumb)
    for t in THUMB:
        rest, fist = data["thumb"]["rest"][t], data["thumb"]["fist"][t]
        out[f"{s}{t}"] = _side(
            s, [(rest[i] + (fist[i] - rest[i]) * k) * (1 - open_) for i in range(3)]
        )
    return out


def _arm(data: dict, s: str, a, where: str) -> dict:
    spec = {"from": a} if isinstance(a, str) else dict(a)
    base = {}
    if "from" in spec:
        if spec["from"] not in data.get("arms", {}):
            raise PosesError(f"{where}: no arm preset {spec['from']!r}.")
        base = data["arms"][spec["from"]]
    out = {}
    for part in (base, spec):
        for limb in LIMBS:
            if limb in part:
                out[f"{s}{limb}"] = _side(s, part[limb])
    hand = spec.get("hand", base.get("hand"))
    if hand is not None:
        out.update(_hand(data, s, hand, where))
    return out


def parse(data: dict, source: str = "") -> SharedPoses:
    """The poses of a file's JSON, expanded; raises PosesError for a broken file."""
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise PosesError(f"The pose file's format is not {FORMAT}.")
    poses = data.get("poses")
    if not isinstance(poses, dict) or not poses:
        raise PosesError("The pose file holds no poses.")
    stances = data.get("stances") or {}
    if not isinstance(stances, dict):
        raise PosesError("The pose file's stances must be an object of stances.")
    bones, own, faces = {}, set(), {}
    for name, p in poses.items():
        where = f"pose {name}"
        b = {}
        stance = {}
        if "stance" in p:
            stance = stances.get(p["stance"])
            if not isinstance(stance, dict):
                raise PosesError(f"{where}: no stance {p['stance']!r}.")
        for bone, e in [*stance.items(), *(p.get("bones") or {}).items()]:
            if bone not in HUMANOID_PARENT:
                raise PosesError(f"{where}: {bone} is not a VRM humanoid bone.")
            if not (isinstance(e, list) and len(e) == 3):
                raise PosesError(f"{where}: {bone} takes three angles.")
            b[bone] = tuple(float(v) for v in e)
        for s in ("left", "right"):
            if s in p:
                b.update(_arm(data, s, p[s], where))
        bones[name] = b
        if p.get("keepHead") is False:
            own.add(name)
            xs = p.get("faces", ["neutral"])
            if not (
                isinstance(xs, list)
                and xs
                and all(isinstance(x, str) for x in xs)
                and len(set(xs)) == len(xs)
            ):
                raise PosesError(f"{where}: faces lists expressions, each once.")
            faces[name] = list(xs)
        elif "faces" in p:
            raise PosesError(
                f"{where}: only a pose that turns its own head (keepHead false) has faces."
            )
    keep = data.get("keepHead")
    keep_bones, keep_pose = (), None
    if keep:
        keep_pose = keep.get("pose")
        keep_bones = tuple(keep.get("bones", ()))
        if keep_pose not in bones:
            raise PosesError(
                f"keepHead names the pose {keep_pose!r}, which the file lacks."
            )
        bad = [b for b in keep_bones if b not in HUMANOID_PARENT]
        if bad:
            raise PosesError(f"keepHead names {bad}, which are not VRM humanoid bones.")
    return SharedPoses(bones, keep_bones, keep_pose, frozenset(own), source, faces)


def load(path=None) -> SharedPoses:
    """The poses of a file (the copy in data/portrait/ by default)."""
    path = pathlib.Path(path or DEFAULT_FILE)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PosesError(f"{path} is not JSON: {error}") from error
    return parse(data, str(path))


# ------------------------------------------------------------------ the rig
def bone_map() -> dict:
    """VRM humanoid bone -> rig bone (the VRM export's table)."""
    from opensculptboy.export.vrm_tables import HUMANOID_1

    return dict(HUMANOID_1)


# The share of the upper arm's own roll (its turn about the arm, X in the T pose) that the
# upper arm's first rig bone keeps; the second (upperarm02, at mid-arm) takes the rest, as
# the forearm's twist bone takes half the hand's roll. With the whole roll at the shoulder
# joint, the armpit's skin twisted about the arm in the raised-arm poses (hands behind the
# head, scratching the head), which the texture showed as stretched streaks.
UPPER_ARM_ROLL = {
    "upperarm01.L": ("leftUpperArm", 0.5),
    "upperarm01.R": ("rightUpperArm", 0.5),
}


def twist_rules() -> dict:
    """Rig twist bone -> (rig bone whose roll it follows, its axis, the weight), as the VRM file's
    roll constraints."""
    from opensculptboy.export.vrm_tables import TWIST_BONES

    axes = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
    return {b: (src, np.array(axes[a]), w) for b, (src, a, w) in TWIST_BONES.items()}


def governing(labels, parents) -> dict:
    """For each rig bone, the humanoid bone whose rotation it takes in the VRM file: its own, or
    that of the nearest rig bone above it that maps to one (the root maps to the hips). Twist
    bones keep their mapped parent here; :func:`page_rotations` adds their roll."""
    to_human = {rig: human for human, rig in bone_map().items()}
    labels = list(labels)
    out = {}
    for j, name in enumerate(labels):
        k = j
        while k >= 0 and labels[k] not in to_human:
            k = int(parents[k])
        if k < 0:
            raise PosesError(f"The rig bone {name} has no humanoid bone above it.")
        out[name] = to_human[labels[k]]
    return out


def rest_turns(model, rest_bone_poses) -> np.ndarray:
    """The T-pose turn of each rig bone in the page's frame (J, 3, 3): the rest-frame correction
    from the rig's rest (an A pose) to the VRM file's T pose, for this body."""
    from opensculptboy.export.tpose import t_pose_turns

    B = np.asarray(rest_bone_poses, dtype=np.float64).reshape(-1, 4, 4)
    U = t_pose_turns(list(model.bone_labels), [int(p) for p in model.bone_parents], B)
    return ANNY_TO_PAGE @ U @ ANNY_TO_PAGE.T


@dataclasses.dataclass
class CardRig:
    """A card's rig in the page's frame: the bones, their parents, their T-pose turns (J, 3, 3)
    and their rest heads (J, 3), in Anny's metres turned into the page's axes."""

    labels: list
    parents: list
    turns: np.ndarray
    heads: np.ndarray


def card_rig(card=None) -> CardRig:
    """The rig of a card (a path, or None for the default character), with its T-pose turns."""
    import torch

    from opensculptboy.character import Character

    character = (
        Character()
        if card is None
        else Character.from_dict(
            json.loads(pathlib.Path(card).read_text(encoding="utf-8"))
        )
    )
    model = character.build_model(face_shapes=bool(character.face_shapes))
    with torch.no_grad():
        rest = model(pose_parameterization="local-ref", **character.model_kwargs())
    B = rest["rest_bone_poses"][0].detach().double().cpu().numpy()
    return CardRig(
        list(model.bone_labels),
        [int(p) for p in model.bone_parents],
        rest_turns(model, B),
        B[:, :3, 3] @ ANNY_TO_PAGE.T,
    )


def rig_world(local: dict, labels, parents, turns) -> np.ndarray:
    """Each rig bone's rotation in the world (J, 3, 3) for the humanoid bones' local rotations."""
    from opensculptboy.export.tpose import roll_constraint

    labels = list(labels)
    W = world(local)
    gov = governing(labels, parents)
    twists = twist_rules()
    index = {n: j for j, n in enumerate(labels)}
    to_human = {rig: human for human, rig in bone_map().items()}
    out = np.empty((len(labels), 3, 3))
    for j, name in enumerate(labels):
        R = W[gov[name]]
        if name in twists:
            src, axis, weight = twists[name]
            hand = to_human.get(src)
            if hand is not None and src in index:
                R = R @ roll_constraint(local.get(hand, np.eye(3)), axis, weight)
        if name in UPPER_ARM_ROLL:
            arm, keep = UPPER_ARM_ROLL[name]
            R = R @ roll_constraint(local.get(arm, np.eye(3)), _X, keep - 1.0)
        out[j] = R @ turns[j]
    return out


def page_rotations(local: dict, labels, parents, turns) -> np.ndarray:
    """Each rig bone's local rotation (J, 3, 3) relative to its parent, in the page's frame."""
    Rw = rig_world(local, labels, parents, turns)
    out = np.empty_like(Rw)
    for j, p in enumerate(parents):
        out[j] = Rw[j] if p < 0 else Rw[int(p)].T @ Rw[j]
    return out


def quaternion(R: np.ndarray) -> np.ndarray:
    """A rotation (3, 3) as a unit quaternion (x, y, z, w) with w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
            0.25 * s,
        ]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[2, 1] - R[1, 2]) / s,
        ]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
            (R[0, 2] - R[2, 0]) / s,
        ]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
            (R[1, 0] - R[0, 1]) / s,
        ]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def page_pose(shared: SharedPoses, name: str, bones, labels, parents, turns) -> dict:
    """A shared pose for the viewer's ``window.portrait.pose``: one rotation (x, y, z, w) per
    bone in the order of ``bones`` (the page's rig; bones the card's rig lacks rest), and no
    root offset (the page stands the figure on the ground)."""
    R = page_rotations(shared.local(name), labels, parents, turns)
    index = {n: j for j, n in enumerate(labels)}
    q = []
    for b in bones:
        q.extend(
            float(x) for x in (quaternion(R[index[b]]) if b in index else (0, 0, 0, 1))
        )
    return {"q": q, "root": [0.0, 0.0, 0.0], "label": name}

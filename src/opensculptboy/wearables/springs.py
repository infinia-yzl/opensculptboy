# OpenSculptBoy
# Apache License, Version 2.0
"""
Spring chains of loose wearables, and the body colliders that they hit (docs/wearables.md,
"Springs").

A package lists its springs in its ``springs`` field. Each entry is one of two kinds:

- ``auto``: columns of joints down the loose region of the piece. A vertex is loose when it
  lies more than ``gap`` (2.5 cm) from the skin or more than ``hang`` (3 cm) below its closest
  skin point. The columns spread over the azimuth of the loose vertices around the body's
  vertical axis (all around for a skirt, across the back for a cape), and each column runs
  from the top of its loose vertices to the lowest vertex of its band, its last joint the tail.
  Outside the ``cloak`` slot, the vertices weighted mostly to the arms (sleeves) take no
  part: they follow the arms.
- ``chain``: a chain of joints that the author places (``points``, metres, glTF axes, on the
  binding body at rest). On another body each point moves with the garment vertices around it.

Each chain hangs from one body bone: ``parent``, or the strongest bone of the garment at the
chain's first joint, taken up the rig to the torso. A node belongs to one chain only, and a
branch is a chain of its own.

:func:`rig_springs` builds the chains of the dressed pieces of an export and blends the
pieces' skin weights into them: a vertex takes the joints of the two nearest chains, by its
place along each chain, and its body weights fade into the chain weights over the first
segment, so that the top of the region stays attached to the body. With chains that hang
from the torso, each vertex of a piece outside the ``cloak`` slot keeps its weight on the
arms, so that a sleeve follows the arm that it covers. :func:`body_colliders`
fits capsules and spheres inside the skin of a body (the head, the neck, the chest and the
shoulders as the viewer's hair colliders, ``viewer/src/hair/colliders.ts``, and the spine, the
hips, the legs and the upper arms), and each joint's hit radius stays within its clearance from
them at rest, so that a chain rests where the garment was made. :class:`SpringSim` repeats the
VRM spring-bone update on joint positions, for tests.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np

from opensculptboy.wearables.geometry import unit_rows

# glTF axes (Y up, the figure facing +Z) to Anny's (Z up, the figure facing -Y), and back
GLTF_TO_ANNY = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
ANNY_TO_GLTF = GLTF_TO_ANNY.T


class SpringWarning(UserWarning):
    """A spring that the joint budget cut down or left out, or that found no loose region."""


KINDS = ("auto", "chain")
SETTING_KEYS = (
    "stiffness",
    "stiffness_top",
    "gravity_power",
    "gravity_dir",
    "drag_force",
    "hit_radius",
)
SPEC_KEYS = {
    "name",
    "kind",
    "preset",
    "settings",
    "groups",
    "columns",
    "joints",
    "points",
    "parent",
    "colliders",
    "gap",
    "hang",
    "reach",
}
MIN_JOINTS = 3  # the fewest joints that a budget leaves a chain: two swinging segments
LOOSE_GAP = 0.025  # metres from the skin
LOOSE_HANG = 0.03  # metres below the closest skin point
REACH = 0.15  # metres: the reach of an authored chain over the garment
CLEARANCE = 0.002  # metres kept between a joint's hit sphere and a collider at rest
# The bones a chain may hang from: the torso, the girdles, the neck and the head.
TORSO = (
    "root",
    "pelvis.L",
    "pelvis.R",
    "spine05",
    "spine04",
    "spine03",
    "spine02",
    "spine01",
    "clavicle.L",
    "clavicle.R",
    "shoulder01.L",
    "shoulder01.R",
    "neck01",
    "neck02",
    "neck03",
    "head",
)
# The bones that the chains of a cloak may hang from: the torso less the neck, the head and
# the shoulders (which turn with the arms).
CLOAK_ROOTS = tuple(b for b in TORSO if not b.startswith(("neck", "head", "shoulder")))
# The upper arms: they and every bone below them are the arms. A sleeve follows them, so the
# chains of a spring that hang from the torso leave the vertices weighted to the arms on their
# bones: a vertex with ARM_SHARE or more of its weight on the arms takes no part in the auto
# columns and no chain weight, and below that the chains' share falls evenly to zero. A
# piece of the cloak slot keeps its chains over the arms: it hangs from the upper body.
ARM_ROOTS = ("upperarm01.L", "upperarm01.R")
ARM_SHARE = 0.5


@dataclasses.dataclass
class SpringSettings:
    """
    The spring of a chain in VRM terms: ``stiffness`` (the pull toward the rest direction),
    ``stiffness_top`` (the factor of the stiffness at the first joint, which falls evenly to
    1 at the last joint that turns, so that the top follows the body and the hem swings),
    ``gravity_power``, ``gravity_dir`` (a unit vector in Anny's frame), ``drag_force`` (0 to
    1) and ``hit_radius`` (metres, the joints' collision spheres).
    """

    stiffness: float = 1.0
    gravity_power: float = 0.4
    gravity_dir: tuple = (0.0, 0.0, -1.0)
    drag_force: float = 0.5
    hit_radius: float = 0.02
    stiffness_top: float = 1.0

    def joint_stiffness(self, count: int) -> np.ndarray:
        """(count,) the stiffness of each joint of a chain of ``count`` joints (the tail
        takes the last value; it drives no segment)."""
        turning = np.linspace(self.stiffness_top, 1.0, max(count - 1, 1))
        return self.stiffness * np.r_[turning, turning[-1]][:count]

    def json(self) -> dict:
        """The settings as a package writes them (``gravity_dir`` in glTF axes)."""
        return {
            "stiffness": float(self.stiffness),
            "stiffness_top": float(self.stiffness_top),
            "gravity_power": float(self.gravity_power),
            "gravity_dir": [
                round(float(x), 6) for x in ANNY_TO_GLTF @ np.array(self.gravity_dir)
            ],
            "drag_force": float(self.drag_force),
            "hit_radius": float(self.hit_radius),
        }


# Defaults for the kinds of loose garment, measured with the VRM spring update on
# test_wearable_springs' long cape and skirt (TestSpringResponse): at rest a chain hangs
# within a few centimetres of the shape it was made in; in a walk at 1.25 m/s its tail lags
# 0.1 to 0.25 m; and after a stop it settles within 0.5 s. A stiffness of 1, the
# VRMC_springBone default, lets a chain stream back by most of its length in a walk, since
# the pull toward the rest direction of each joint adds up down the chain.
PRESETS = {
    "cape": SpringSettings(4.0, 0.6, (0.0, 0.0, -1.0), 0.5, 0.04, stiffness_top=2.0),
    "skirt": SpringSettings(3.0, 0.3, (0.0, 0.0, -1.0), 0.5, 0.03, stiffness_top=2.0),
    "hem": SpringSettings(3.0, 0.4, (0.0, 0.0, -1.0), 0.5, 0.02, stiffness_top=2.0),
}
SLOT_PRESETS = {"cloak": "cape", "legs": "skirt", "under": "skirt"}
# The bone that the chains of a preset hang from when the spring names none (else the
# strongest torso bone at each chain's first joint: the upper spine or a clavicle for a cape).
PRESET_PARENTS = {"skirt": "root"}
COLUMNS = {"cape": 5, "skirt": 8, "hem": 3}
JOINTS = 4


@dataclasses.dataclass
class SpringSpec:
    """One entry of a package's ``springs``, read (:meth:`parse`)."""

    name: str
    kind: str
    preset: str
    settings: SpringSettings
    groups: list | None = None
    columns: int = 1
    joints: int = JOINTS
    points: np.ndarray | None = None  # (n, 3) Anny's frame, on the binding body at rest
    parent: str | None = None
    colliders: list | None = None
    gap: float = LOOSE_GAP
    hang: float = LOOSE_HANG
    reach: float = REACH

    @classmethod
    def parse(cls, entry, slot: str, groups) -> "SpringSpec":
        """Read an entry; raises ValueError for an entry that a package cannot carry."""
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError(f"A spring needs a 'name': {entry!r}.")
        name = entry["name"]
        unknown = sorted(set(entry) - SPEC_KEYS)
        if unknown:
            raise ValueError(
                f"Spring {name!r}: unknown keys {unknown}; the keys are {sorted(SPEC_KEYS)}."
            )
        kind = entry.get("kind", "chain" if "points" in entry else "auto")
        if kind not in KINDS:
            raise ValueError(f"Spring {name!r}: kind must be one of {KINDS}.")
        preset = entry.get("preset", SLOT_PRESETS.get(slot, "hem"))
        if preset not in PRESETS:
            raise ValueError(
                f"Spring {name!r}: unknown preset {preset!r}; use one of {list(PRESETS)}."
            )
        values = dataclasses.asdict(PRESETS[preset])
        given = entry.get("settings") or {}
        if not isinstance(given, dict) or set(given) - set(SETTING_KEYS):
            raise ValueError(
                f"Spring {name!r}: settings take the keys {list(SETTING_KEYS)}."
            )
        for key, value in given.items():
            if key == "gravity_dir":
                d = np.asarray(value, dtype=np.float64).reshape(-1)
                if d.shape != (3,) or not np.linalg.norm(d) > 0:
                    raise ValueError(f"Spring {name!r}: gravity_dir is a 3-vector.")
                values[key] = tuple(GLTF_TO_ANNY @ (d / np.linalg.norm(d)))
            else:
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(
                        f"Spring {name!r}: {key} is a number of 0 or more."
                    )
                values[key] = float(value)
        if values["drag_force"] > 1:
            raise ValueError(f"Spring {name!r}: drag_force lies within 0 to 1.")
        spec = cls(
            name=name,
            kind=kind,
            preset=preset,
            settings=SpringSettings(**values),
            groups=entry.get("groups"),
            columns=int(entry.get("columns", COLUMNS[preset] if kind == "auto" else 1)),
            joints=int(entry.get("joints", JOINTS)),
            parent=entry.get("parent", PRESET_PARENTS.get(preset)),
            colliders=entry.get("colliders"),
            gap=float(entry.get("gap", LOOSE_GAP)),
            hang=float(entry.get("hang", LOOSE_HANG)),
            reach=float(entry.get("reach", REACH)),
        )
        if spec.groups is not None:
            missing = sorted(set(spec.groups) - set(groups))
            if missing:
                raise ValueError(f"Spring {name!r} names groups it lacks: {missing}.")
        if spec.colliders is not None:
            missing = sorted(set(spec.colliders) - set(COLLIDER_GROUPS))
            if missing:
                raise ValueError(
                    f"Spring {name!r}: unknown collider groups {missing}; the groups are "
                    f"{list(COLLIDER_GROUPS)}."
                )
        if kind == "chain":
            points = np.asarray(entry.get("points", []), dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
                raise ValueError(f"Spring {name!r}: points are two or more 3-vectors.")
            spec.points = points @ GLTF_TO_ANNY.T
            spec.joints = int(entry.get("joints", len(points)))
        if spec.joints < 2 or spec.columns < 1:
            raise ValueError(f"Spring {name!r}: a chain has 2 joints or more.")
        return spec

    @property
    def cost(self) -> int:
        """The skin joints of the spring: every joint but the tail of each chain."""
        return self.columns * (self.joints - 1)


def parse_springs(wearable) -> list[SpringSpec]:
    """The springs of a package, read; raises ValueError for a bad entry or a name twice."""
    specs = [
        SpringSpec.parse(e, wearable.slot, wearable.garment.groups)
        for e in wearable.springs
    ]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"The springs of a package have distinct names: {names}.")
    return specs


# --- colliders -----------------------------------------------------------------------------


@dataclasses.dataclass
class Collider:
    """A capsule (``head`` to ``tail``) or a sphere (``tail`` is ``head``) carried by
    ``bone``, at rest in Anny's frame."""

    name: str
    group: str
    bone: str
    head: np.ndarray
    tail: np.ndarray
    radius: float

    @property
    def sphere(self) -> bool:
        return bool(np.allclose(self.head, self.tail))

    def array(self) -> np.ndarray:
        """(7,) the ends and the radius, as :class:`SpringSim` and the hair solver take."""
        return np.concatenate([self.head, self.tail, [self.radius]])


# (name, group, ends, carrying bone, quantile, span): the radius is the quantile of the
# distances from the axis of the skin vertices whose projection falls within the span, as
# fitColliders of viewer/src/hair/colliders.ts fits them, and at most the same quantile of
# the distances from the ends of the skin vertices beyond them, so that the round ends stay
# inside the skin too. "neck", "chest" and "shoulder.*"
# take the ends, quantiles and spans of the viewer's table (test_wearable_springs checks it).
COLLIDER_TABLE = (
    ("neck", "head", ("neck01", "head"), "neck01", 0.08, (0.1, 0.8)),
    ("chest", "torso", ("spine02", "neck01"), "spine01", 0.04, (0.2, 0.9)),
    (
        "shoulder.L",
        "arms",
        ("clavicle.L", "upperarm01.L"),
        "clavicle.L",
        0.08,
        (0.3, 0.95),
    ),
    (
        "shoulder.R",
        "arms",
        ("clavicle.R", "upperarm01.R"),
        "clavicle.R",
        0.08,
        (0.3, 0.95),
    ),
    ("spine", "torso", ("spine04", "spine02"), "spine04", 0.04, (0.1, 0.9)),
    ("hips", "torso", ("upperleg01.L", "upperleg01.R"), "root", 0.05, (0.0, 1.0)),
    (
        "upperleg.L",
        "legs",
        ("upperleg01.L", "lowerleg01.L"),
        "upperleg01.L",
        0.05,
        (0.1, 0.9),
    ),
    (
        "upperleg.R",
        "legs",
        ("upperleg01.R", "lowerleg01.R"),
        "upperleg01.R",
        0.05,
        (0.1, 0.9),
    ),
    (
        "lowerleg.L",
        "legs",
        ("lowerleg01.L", "foot.L"),
        "lowerleg01.L",
        0.05,
        (0.1, 0.9),
    ),
    (
        "lowerleg.R",
        "legs",
        ("lowerleg01.R", "foot.R"),
        "lowerleg01.R",
        0.05,
        (0.1, 0.9),
    ),
    (
        "upperarm.L",
        "arms",
        ("upperarm01.L", "lowerarm01.L"),
        "upperarm01.L",
        0.05,
        (0.1, 0.9),
    ),
    (
        "upperarm.R",
        "arms",
        ("upperarm01.R", "lowerarm01.R"),
        "upperarm01.R",
        0.05,
        (0.1, 0.9),
    ),
)
COLLIDER_GROUPS = ("head", "torso", "legs", "arms")
HEAD_QUANTILE = (
    0.1  # the head sphere: this quantile of the head's distances from its centre
)
REFERENCE_HEIGHT = (
    1.75  # metres: the height at which the viewer's 0.25 m cut-off applies
)


def _segment_offsets(head, tail, points):
    """(N, 3) the offsets of points from their nearest points on a segment, and (N,) the
    unclamped parameter along it."""
    e = tail - head
    ee = float(e @ e)
    t = ((points - head) @ e) / ee if ee > 0 else np.zeros(len(points))
    return points - (head + np.clip(t, 0.0, 1.0)[:, None] * e), t


def body_colliders(body) -> list[Collider]:
    """
    The colliders of a body at rest (:class:`~opensculptboy.wearables.bind.BodySurface`): a
    sphere inside the head (at the middle of the box of the skin that the head bone carries)
    and the capsules of :data:`COLLIDER_TABLE`, each fitted inside
    the skin of this body, so that they scale with the character.
    """
    labels = list(body.bone_labels)
    heads = body.bone_poses[:, :3, 3]
    skin = np.unique(body.triangles)
    V = body.vertices[skin]
    height = float(np.ptp(body.vertices[:, 2])) or REFERENCE_HEIGHT
    cut = 0.25 * height / REFERENCE_HEIGHT
    out = []
    if "head" in labels:
        h = labels.index("head")
        dense = np.zeros(len(body.vertices))
        for k in range(body.weights.shape[1]):
            dense += np.where(body.indices[:, k] == h, body.weights[:, k], 0.0)
        rows = skin[dense[skin] > 0.9]
        if len(rows):
            head_vertices = body.vertices[rows]
            centre = 0.5 * (head_vertices.min(axis=0) + head_vertices.max(axis=0))
            dist = np.linalg.norm(head_vertices - centre, axis=1)
            radius = float(np.quantile(dist, HEAD_QUANTILE))
            out.append(Collider("head", "head", "head", centre, centre.copy(), radius))
    for name, group, ends, bone, quantile, span in COLLIDER_TABLE:
        if not all(b in labels for b in (*ends, bone)):
            continue
        A, B = heads[labels.index(ends[0])], heads[labels.index(ends[1])]
        offsets, t = _segment_offsets(A, B, V)
        dist = np.linalg.norm(offsets, axis=1)
        keep = (t >= span[0]) & (t <= span[1]) & (dist < cut)
        if not keep.any():
            continue
        d = np.sort(dist[keep])
        radius = float(d[int(np.floor(quantile * (len(d) - 1)))])
        # the round ends too stay inside: the skin beyond the ends sets a radius the same way
        beyond = ((t < 0) | (t > 1)) & (dist < cut)
        if beyond.any():
            d = np.sort(dist[beyond])
            radius = min(radius, float(d[int(np.floor(quantile * (len(d) - 1)))]))
        out.append(Collider(name, group, bone, A.copy(), B.copy(), radius))
    return out


def collider_distance(collider_arrays: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(N, C) the distance of each point from the surface of each collider (7,) row (below
    zero inside)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = np.zeros((len(points), len(collider_arrays)))
    for c, cap in enumerate(np.asarray(collider_arrays).reshape(-1, 7)):
        offsets, _ = _segment_offsets(cap[:3], cap[3:6], points)
        out[:, c] = np.linalg.norm(offsets, axis=1) - cap[6]
    return out


# --- chains --------------------------------------------------------------------------------


@dataclasses.dataclass
class Chain:
    """
    A chain of joints at rest in Anny's frame, on the body of an export: the first joint
    hangs from the body bone ``parent``, the last is the tail. ``hit_radii`` (n,) are the
    joints' hit radii, within their clearance from the colliders at rest.
    """

    name: str
    piece: str
    spring: str
    parent: str
    positions: np.ndarray
    settings: SpringSettings
    colliders: list
    hit_radii: np.ndarray
    preset: str = "hem"

    @property
    def count(self) -> int:
        return len(self.positions)


@dataclasses.dataclass
class SpringNode:
    """A node of the chains: its name (the chain's name and the joint's place, with ``_``
    for the ``/`` of a piece reference, since engines read ``/`` in node names as a path),
    its parent (a body bone index below the bone count,
    or the index of another node of the rig), its rest position (Anny's frame), its chain and
    its place there. Tails come after every other node, and they carry no skin weight."""

    name: str
    parent: int
    position: np.ndarray
    chain: int
    joint: int
    tail: bool


@dataclasses.dataclass
class SpringRig:
    """
    The chains of an export and the body's colliders. The nodes of the chains follow the
    body's bones: index ``bone_count + k`` is the k-th entry of :meth:`nodes`, and the skin
    joints are the bones and then the nodes that are not tails (:attr:`skin_joint_count`).
    """

    bone_count: int
    chains: list
    colliders: list
    messages: list = dataclasses.field(default_factory=list)

    @property
    def skin_joint_count(self) -> int:
        return sum(c.count - 1 for c in self.chains)

    def first_joint(self, chain: int) -> int:
        """The index of a chain's first joint."""
        return self.bone_count + sum(c.count - 1 for c in self.chains[:chain])

    def joint_index(self, chain: int, joint: int) -> int:
        """The index of a joint (a skin joint, or a tail after every skin joint)."""
        if joint == self.chains[chain].count - 1:
            return self.bone_count + self.skin_joint_count + chain
        return self.first_joint(chain) + joint

    def nodes(self, bone_labels) -> list[SpringNode]:
        out = []
        labels = list(bone_labels)
        for tails in (False, True):
            for c, chain in enumerate(self.chains):
                for k in range(chain.count):
                    if (k == chain.count - 1) != tails:
                        continue
                    parent = (
                        labels.index(chain.parent)
                        if k == 0
                        else self.joint_index(c, k - 1)
                    )
                    out.append(
                        SpringNode(
                            f"{chain.name.replace('/', '_')}.{k}",
                            parent,
                            chain.positions[k].copy(),
                            c,
                            k,
                            tails,
                        )
                    )
        return out

    def parent_bones(self, bone_labels) -> np.ndarray:
        """(skin_joint_count,) the body bone that each skin joint of the chains rides on at
        rest (the chain's parent)."""
        labels = list(bone_labels)
        return np.array(
            [labels.index(c.parent) for c in self.chains for _ in range(c.count - 1)],
            dtype=np.int64,
        )

    def extras(self) -> dict:
        """The ``springs`` entry of a file's scene extras."""
        return {
            "chains": [
                {
                    "name": c.name,
                    "piece": c.piece,
                    "spring": c.spring,
                    "preset": c.preset,
                    "parent": c.parent,
                    "joints": c.count,
                    "settings": c.settings.json(),
                    "hit_radii": [round(float(r), 5) for r in c.hit_radii],
                    "colliders": list(c.colliders),
                }
                for c in self.chains
            ],
            "colliders": [
                {
                    "name": k.name,
                    "group": k.group,
                    "bone": k.bone,
                    "shape": "sphere" if k.sphere else "capsule",
                    "radius": round(float(k.radius), 5),
                }
                for k in self.colliders
            ],
            "messages": list(self.messages),
        }


def _dense(weights, joints, count) -> np.ndarray:
    dense = np.zeros((len(weights), count))
    rows = np.repeat(np.arange(len(weights)), weights.shape[1])
    np.add.at(dense, (rows, np.asarray(joints).reshape(-1)), weights.reshape(-1))
    return dense


def _region(piece, spec) -> np.ndarray:
    """(V,) the vertices of the spring's groups (all groups when it names none)."""
    garment = piece.garment
    if spec.groups is None:
        return np.ones(len(piece.vertices), bool)
    keep = np.zeros(len(piece.vertices), bool)
    for g, group in enumerate(garment.groups):
        if group in spec.groups:
            keep[garment.triangles[garment.triangle_group == g].reshape(-1)] = True
    return keep


def _parent_bone(piece, point, parents, labels, given) -> str:
    """The bone a chain hangs from: ``given``, else the bone that the garment's vertices
    nearest to ``point`` are weighted to most, each bone's weight taken up the rig to a bone
    of :data:`TORSO`. A piece of the ``cloak`` slot hangs from the upper body
    (:data:`CLOAK_ROOTS`): the weights of the neck, the head and the shoulders go up to the
    spine or the clavicles, so that a head turn does not swing the cloak."""
    if given is not None:
        if given not in labels:
            raise ValueError(f"A spring names the unknown parent bone {given!r}.")
        return given
    allowed = CLOAK_ROOTS if piece.wearable.slot == "cloak" else TORSO
    near = np.argsort(np.linalg.norm(piece.vertices - point, axis=1))[:8]
    dense = _dense(piece.weights[near], piece.joints[near], len(labels)).sum(axis=0)
    totals = np.zeros(len(labels))
    for bone in np.flatnonzero(dense > 0):
        root = int(bone)
        while labels[root] not in allowed and parents[root] >= 0:
            root = int(parents[root])
        totals[root] += dense[bone]
    return labels[int(np.argmax(totals))]


def arm_bones(labels, parents) -> np.ndarray:
    """(J,) the bones of the arms: the upper arms (:data:`ARM_ROOTS`) and every bone below
    them."""
    labels = list(labels)
    out = np.zeros(len(labels), bool)
    for j in range(len(labels)):
        b = j
        while b >= 0 and labels[b] not in ARM_ROOTS:
            b = int(parents[b])
        out[j] = b >= 0
    return out


def arm_share(piece, arms: np.ndarray) -> np.ndarray:
    """(V,) the share of each vertex's skin weights on the bones of the arms (``arms``, from
    :func:`arm_bones`; the joints of chains count as other bones)."""
    joints = np.asarray(piece.joints)
    on_arm = np.zeros(joints.shape, bool)
    inside = joints < len(arms)
    on_arm[inside] = arms[joints[inside]]
    weights = np.asarray(piece.weights, dtype=np.float64)
    total = np.maximum(weights.sum(axis=1), 1e-12)
    return (weights * on_arm).sum(axis=1) / total


def _auto_columns(piece, body, spec, arm=None) -> list[np.ndarray]:
    """The joint positions (n, 3) of each column of an ``auto`` spring on this body;
    ``arm`` (V,) is the vertices' share of weight on the arms (:func:`arm_share`), and the
    vertices weighted mostly to the arms (sleeves) take no part."""
    region = _region(piece, spec)
    if arm is not None:
        region &= arm < ARM_SHARE
    rows = np.flatnonzero(region)
    if len(rows) == 0:
        return []
    V = piece.vertices[rows]
    found = body.surface.closest(V)
    loose = (found.distance > spec.gap) | (found.point[:, 2] - V[:, 2] > spec.hang)
    if not loose.any():
        return []
    labels = list(body.bone_labels)
    centre = body.bone_poses[labels.index("root"), :2, 3] if "root" in labels else 0
    az = np.arctan2(V[:, 1] - centre[1], V[:, 0] - centre[0])
    a = np.sort(az[loose])
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
    widest = int(np.argmax(gaps))
    C = spec.columns
    # the azimuth from the far side of the widest gap, so that a span never wraps; the
    # bands hold equal shares of the loose vertices (equal widths on an even mesh)
    start = a[(widest + 1) % len(a)]
    around = np.mod(az - start, 2 * np.pi)
    if gaps[widest] < 2 * np.pi / C:  # all around: a skirt
        edges = np.linspace(0.0, 2 * np.pi, C + 1)
    else:  # across a span: a cape
        edges = np.quantile(around[loose], np.linspace(0.0, 1.0, C + 1))
        edges[-1] += 1e-9
    columns = []  # (band, top of its loose vertices, its lowest vertex)
    for c in range(C):
        band = (around >= edges[c]) & (around <= edges[c + 1])
        if (band & loose).any():
            columns.append(
                [band, float(V[band & loose, 2].max()), float(V[band, 2].min())]
            )
    # each top is the median of the tops of three neighbouring columns (all around:
    # wrapped; at the ends of a span: the three at that end), so that a column whose top
    # vertices miss the loose test by a hair starts with the others
    tops = np.array([t for _, t, _ in columns])
    ring = gaps[widest] < 2 * np.pi / C
    for c, column in enumerate(columns):
        if len(columns) < 3:
            break
        if ring:
            near = [(c + k) % len(columns) for k in (-1, 0, 1)]
        else:
            first = min(max(c - 1, 0), len(columns) - 3)
            near = [first, first + 1, first + 2]
        column[1] = min(float(np.median(tops[near])), float(V[column[0], 2].max()))
    out = []
    for band, top, bottom in columns:
        if top - bottom < 0.03:
            continue
        n = spec.joints
        heights = np.linspace(top, bottom, n)
        step = (top - bottom) / (n - 1)
        points = []
        for z in heights:
            near = band & (np.abs(V[:, 2] - z) <= 0.5 * step)
            if not near.any():
                gap = np.abs(V[band, 2] - z).min()
                near = band & (np.abs(V[:, 2] - z) <= gap + 1e-9)
            xy = V[near, :2].mean(axis=0)
            points.append([xy[0], xy[1], z])
        out.append(np.array(points))
    return out


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    """``count`` points evenly along a polyline, its ends kept."""
    if count == len(points):
        return points.copy()
    lengths = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )
    at = np.linspace(0.0, lengths[-1], count)
    return np.stack([np.interp(at, lengths, points[:, k]) for k in range(3)], axis=1)


def _authored_chain(piece, spec) -> np.ndarray:
    """An authored chain on this body: each point moves as the garment vertices around it
    (the six nearest on the binding body) move from the binding body to this one."""
    rest = piece.garment.vertices
    moved = piece.vertices - rest
    points = []
    for p in spec.points:
        near = np.argsort(np.linalg.norm(rest - p, axis=1))[:6]
        points.append(p + moved[near].mean(axis=0))
    return _resample(np.array(points), spec.joints)


def _polyline(points: np.ndarray, X: np.ndarray):
    """For each point of X: (distance to the polyline, place along it in segments; below 0
    above the first joint)."""
    best_d = np.full(len(X), np.inf)
    best_s = np.zeros(len(X))
    for k in range(len(points) - 1):
        offsets, t = _segment_offsets(points[k], points[k + 1], X)
        if k > 0:
            t = np.maximum(t, 0.0)
        if k < len(points) - 2:
            t = np.minimum(t, 1.0)
        d = np.linalg.norm(offsets, axis=1)
        better = d < best_d - 1e-12
        best_d = np.where(better, d, best_d)
        best_s = np.where(better, k + t, best_s)
    return best_d, best_s


def _chain_weights(piece, chains, first_joints, region, reach, count):
    """(V, count) the chain part of a piece's skin weights and (V,) its share."""
    V = piece.vertices
    D = np.full((len(V), len(chains)), np.inf)
    S = np.zeros((len(V), len(chains)))
    for c, chain in enumerate(chains):
        D[:, c], S[:, c] = _polyline(chain.positions, V)
    nearest = np.argsort(D, axis=1)[:, :2]
    rows = np.arange(len(V))
    out = np.zeros((len(V), count))
    d = D[rows[:, None], nearest]
    inverse = 1.0 / np.maximum(d, 1e-4) ** 2
    if len(chains) == 1:
        inverse[:, 1:] = 0.0
    mix = inverse / inverse.sum(axis=1, keepdims=True)
    for slot in range(nearest.shape[1]):
        for c, chain in enumerate(chains):
            pick = (nearest[:, slot] == c) & region
            if not pick.any():
                continue
            n = chain.count
            s = np.clip(S[pick, c], 0.5, n - 1.5)
            for b in range(n - 1):
                tent = np.maximum(0.0, 1.0 - np.abs(s - (b + 0.5)))
                out[pick, first_joints[c] + b] += mix[pick, slot] * tent
    s0 = S[rows, nearest[:, 0]]
    share = np.clip(s0, 0.0, 1.0) * region
    if np.isfinite(reach):
        fade = np.clip(1.0 - d[:, 0] / reach, 0.0, 1.0)
        share = share * (fade * fade * (3 - 2 * fade))
    total = out.sum(axis=1, keepdims=True)
    out = np.where(total > 0, out / np.maximum(total, 1e-12), 0.0)
    share = np.where(total[:, 0] > 0, share, 0.0)
    return out, share


def _fit_budget(planned, max_joints, messages):
    """Fewer joints, then fewer columns, then fewer springs, until the skin joints of the
    chains fit within ``max_joints``."""
    import copy

    planned = [(piece, [copy.copy(s) for s in specs]) for piece, specs in planned]
    every = [s for _, specs in planned for s in specs]

    def cost():
        return sum(s.cost for s in every if s.columns > 0)

    while cost() > max_joints:
        live = [s for s in every if s.columns > 0]
        spec = max(live, key=lambda s: (s.cost, s.joints))
        if spec.joints > MIN_JOINTS:
            spec.joints -= 1
        elif spec.columns > 2 or (spec.kind == "chain" and spec.columns > 1):
            spec.columns -= 1
        else:
            spec.columns = 0
            messages.append(f"spring {spec.name!r} left out: the joint budget is full")
    return [(piece, [s for s in specs if s.columns > 0]) for piece, specs in planned]


def rig_springs(body, pieces, parents, max_skin_joints=None) -> SpringRig | None:
    """
    The chains of dressed pieces (:class:`~opensculptboy.wearables.fit.DressedPiece`) on
    the body of an export, with the pieces' skin weights blended into them in place; None
    when no piece has springs.

    Args:
        body: the body of the export (:func:`~opensculptboy.wearables.fit.card_body`), which
            places the colliders and the loose regions.
        pieces: the dressed pieces that the file writes.
        parents: (J,) the parent of each bone of the rig, for the bones the chains hang from.
        max_skin_joints: the skin joints that the chains may add (a VRM file's joint budget
            less its bones): springs then take fewer joints, then fewer columns, then leave
            the file, and :attr:`SpringRig.messages` says so. None for no limit.
    """
    labels = list(body.bone_labels)
    J = len(labels)
    planned = [(p, parse_springs(p.wearable)) for p in pieces if p.wearable.springs]
    planned = [(p, specs) for p, specs in planned if specs]
    if not planned:
        return None
    messages: list[str] = []
    if max_skin_joints is not None:
        before = [(s.columns, s.joints) for _, specs in planned for s in specs]
        fitted = _fit_budget(planned, max(int(max_skin_joints), 0), messages)
        _note_reductions(planned, before, fitted, messages)
        planned = fitted
    colliders = body_colliders(body)
    arms = arm_bones(labels, parents)
    on_arms = {  # a cloak hangs from the upper body whole (CLOAK_ROOTS)
        id(piece): np.zeros(len(piece.vertices))
        if piece.wearable.slot == "cloak"
        else arm_share(piece, arms)
        for piece, _ in planned
    }
    chains: list[Chain] = []
    owners = []  # (piece, spec, chain indices)
    for piece, specs in planned:
        for spec in specs:
            if spec.kind == "auto":
                columns = _auto_columns(piece, body, spec, on_arms[id(piece)])
            else:
                columns = [_authored_chain(piece, spec)]
            made = []
            for c, points in enumerate(columns):
                parent = _parent_bone(piece, points[0], parents, labels, spec.parent)
                groups = list(spec.colliders or COLLIDER_GROUPS)
                hits = _hit_radii(points, spec.settings.hit_radius, colliders, groups)
                made.append(len(chains))
                chains.append(
                    Chain(
                        name=f"{piece.ref}.{spec.name}.{c}",
                        piece=piece.ref,
                        spring=spec.name,
                        parent=parent,
                        positions=points,
                        settings=spec.settings,
                        colliders=groups,
                        hit_radii=hits,
                        preset=spec.preset,
                    )
                )
            if not made:
                messages.append(
                    f"{piece.ref}: spring {spec.name!r} found no loose region"
                )
            owners.append((piece, spec, made))
    for message in messages:
        warnings.warn(message, SpringWarning, stacklevel=2)
    if not chains:
        return None
    rig = SpringRig(J, chains, colliders, messages)
    count = J + rig.skin_joint_count
    for piece, spec, made in owners:
        if not made:
            continue
        region = _region(piece, spec)
        mine = [chains[c] for c in made]
        first = [rig.first_joint(c) for c in made]
        reach = np.inf if spec.kind == "auto" else spec.reach
        chain_part, share = _chain_weights(piece, mine, first, region, reach, count)
        if not any(arms[labels.index(c.parent)] for c in mine):
            # chains of the torso: a sleeve keeps its weights, and the chains' share falls
            # to zero as the vertex's share on the arms grows to ARM_SHARE
            share = share * np.clip(1.0 - on_arms[id(piece)] / ARM_SHARE, 0.0, 1.0)
        k = piece.weights.shape[1]
        dense = _dense(piece.weights, piece.joints, count)
        dense = (1.0 - share)[:, None] * dense + share[:, None] * chain_part
        order = np.argsort(-np.round(dense, 9), axis=1, kind="stable")[:, :k]
        weights = np.take_along_axis(dense, order, axis=1)
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        piece.weights = weights.astype(piece.weights.dtype)
        piece.joints = np.where(weights > 0, order, 0).astype(piece.joints.dtype)
    return rig


def _note_reductions(planned, before, fitted, messages):
    """A message for each spring that the joint budget cut down."""
    kept = {(p.ref, s.name): s for p, specs in fitted for s in specs}
    every = [(p, s) for p, specs in planned for s in specs]
    for (piece, spec), (columns, joints) in zip(every, before):
        now = kept.get((piece.ref, spec.name))
        if now is not None and (now.columns, now.joints) != (columns, joints):
            messages.append(
                f"{piece.ref}: spring {spec.name!r} takes {now.columns} chains of "
                f"{now.joints} joints (from {columns} of {joints}) to keep within the "
                "joint budget"
            )


def _hit_radii(points, hit_radius, colliders, groups) -> np.ndarray:
    """The hit radius of each joint: at most its clearance from the colliders at rest."""
    arrays = np.array([c.array() for c in colliders if c.group in groups]).reshape(
        -1, 7
    )
    if not len(arrays):
        return np.full(len(points), float(hit_radius))
    clearance = collider_distance(arrays, points).min(axis=1) - CLEARANCE
    return np.clip(clearance, 0.0, float(hit_radius))


# --- the spring update ---------------------------------------------------------------------


def _turn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation (3, 3) that turns the unit vector a onto the unit vector b."""
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -1 + 1e-9:  # opposite: half a turn about any axis normal to a
        axis = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


class SpringSim:
    """
    The VRM spring-bone update on the joint positions of chains, after the VRM 1.0
    specification and three-vrm: each step moves the tail of every segment by its inertia
    (less the drag), the stiffness toward its rest direction in its parent's frame and the
    gravity, keeps its length, and pushes it out of the colliders by the joint's hit radius.

    Args:
        chains: per chain, a dict of ``positions`` (n, 3) at rest, ``stiffness``,
            ``gravity_power``, ``gravity_dir`` (3,), ``drag_force`` and ``hit_radii`` (n,);
            the stiffness, the gravity power and the drag force are numbers or (n,) per
            joint, as a VRM 1.0 file writes them (joint k drives the segment from it to
            joint k + 1).
        colliders: (m, 7) the colliders at rest (two ends and a radius).
    """

    def __init__(self, chains, colliders=()):
        self.chains = [dict(c) for c in chains]
        self.colliders = np.asarray(colliders, dtype=np.float64).reshape(-1, 7)
        for c in self.chains:
            p = np.asarray(c["positions"], dtype=np.float64)
            c["positions"] = p
            for key in ("stiffness", "gravity_power", "drag_force"):
                c[key] = np.broadcast_to(
                    np.asarray(c[key], dtype=np.float64), (len(p),)
                ).copy()
            c["lengths"] = np.linalg.norm(np.diff(p, axis=0), axis=1)
            c["rest_dirs"] = unit_rows(np.diff(p, axis=0))
            c["tails"] = p[1:].copy()
            c["prev"] = p[1:].copy()

    def joints(self) -> list[np.ndarray]:
        """The current joint positions (n, 3) of each chain (the first joint at rest)."""
        return [
            np.vstack([c.get("head", c["positions"][0]), c["tails"]])
            for c in self.chains
        ]

    def step(self, dt: float, motion: np.ndarray | None = None) -> None:
        """One step of ``dt`` seconds with the body at the rigid ``motion`` (4, 4) from its
        rest (the first joints and the colliders move with it)."""
        M = np.eye(4) if motion is None else np.asarray(motion, dtype=np.float64)
        R, t = M[:3, :3], M[:3, 3]
        caps = self.colliders.copy()
        caps[:, :3] = caps[:, :3] @ R.T + t
        caps[:, 3:6] = caps[:, 3:6] @ R.T + t
        for c in self.chains:
            head = R @ c["positions"][0] + t
            c["head"] = head
            Q = R
            g = np.asarray(c["gravity_dir"], dtype=np.float64)
            for k in range(len(c["lengths"])):
                cur, prev = c["tails"][k], c["prev"][k]
                target = Q @ c["rest_dirs"][k]
                nxt = (
                    cur
                    + (cur - prev) * (1.0 - c["drag_force"][k])
                    + dt * c["stiffness"][k] * target
                    + dt * c["gravity_power"][k] * g
                )
                nxt = head + unit_rows((nxt - head)[None])[0] * c["lengths"][k]
                radius = float(c["hit_radii"][k])
                for cap in caps:
                    offset, _ = _segment_offsets(cap[:3], cap[3:6], nxt[None])
                    dist = float(np.linalg.norm(offset))
                    reach = cap[6] + radius
                    if dist < reach and dist > 0:
                        nxt = nxt + offset[0] / dist * (reach - dist)
                        nxt = head + unit_rows((nxt - head)[None])[0] * c["lengths"][k]
                c["prev"][k] = cur
                c["tails"][k] = nxt
                direction = unit_rows((nxt - head)[None])[0]
                Q = _turn(target, direction) @ Q
                head = nxt

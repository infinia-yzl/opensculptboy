# OpenSculptBoy
# Apache License, Version 2.0
"""
A statue scan as an appearance handoff folder for ``opensculptboy fit`` (docs/statues.md).

:func:`scan_to_handoff` reads a scan, turns it upright and to face +Z in glTF's frame, cuts
the plinth off, scales the figure to a stature inside anny's range, takes the masked pieces
out and writes::

    <out>/
        replica.glb       the figure as the skin part "statue"; the masked drapery and hair as
                          parts of the roles "clothing" and "hair", which stay out of the fit
        removed.glb       what the conversion took out (plinth, supports, breaks, loose
                          fragments), one node per role, for review
        appearance.json   the landmarks, the parts and a "statue" section: the source, the
                          licence and credit, the transform from the scan and what was removed

Masks and landmarks are given in the frame of ``replica.glb`` (metres, Y up, the figure facing
+Z) or, with ``"frame": "scan"``, in the scan's own coordinates. A first run without masks
writes the frame, so that a person can read the positions of supports and breaks off
``replica.glb`` in any glTF viewer.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

from opensculptboy.statues.scan import (
    Mask,
    components,
    frame_rotation,
    guess_front,
    guess_up,
    load_scan,
    plinth_top,
    select_triangles,
)

FORMAT = "opensculptboy/statue-handoff@1"
# the stature (metres) a figure takes by default: inside anny's height range
DEFAULT_STATURE = 1.8
# loose pieces of the skin below this share of its area are fragments, taken out
DEBRIS_SHARE = 0.02
# triangles of the plinth: all corners within this share of the height over the plinth's top
PLINTH_MARGIN = 0.002
REMOVED_ROLES = ("plinth", "support", "break", "debris")


@dataclasses.dataclass
class StatueHandoff:
    """What :func:`scan_to_handoff` wrote and how it placed the scan."""

    folder: pathlib.Path
    # Anny's frame from the scan's coordinates: anny = scale * (rotation @ scan) + translation
    rotation: np.ndarray
    scale: float
    translation: np.ndarray
    up: np.ndarray  # the scan's up direction (scan axes)
    front: np.ndarray  # the scan's front direction (scan axes)
    up_confidence: float
    front_score: float
    plinth: bool
    triangles: dict  # role -> triangle count
    landmarks: dict  # name -> point in Anny's frame

    def to_anny(self, points) -> np.ndarray:
        """Scan coordinates (N, 3) in Anny's frame."""
        return self.scale * (np.asarray(points, dtype=np.float64) @ self.rotation.T) + (
            self.translation
        )

    def to_scan(self, points) -> np.ndarray:
        """Points of Anny's frame (N, 3) in the scan's coordinates."""
        return (
            (np.asarray(points, dtype=np.float64) - self.translation) / self.scale
        ) @ self.rotation


def _gltf_to_anny(points) -> np.ndarray:
    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    return np.asarray(points, dtype=np.float64) @ GLTF_TO_ANNY.T


def _anny_to_gltf(points) -> np.ndarray:
    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    return np.asarray(points, dtype=np.float64) @ GLTF_TO_ANNY


def _mask_in_anny(entry, to_anny, scale: float) -> Mask:
    """A mask entry (a dict or a :class:`Mask`) with its positions in Anny's frame."""
    entry = dict(dataclasses.asdict(entry) if isinstance(entry, Mask) else entry)
    frame = entry.pop("frame", "handoff")
    if frame not in ("handoff", "scan"):
        raise ValueError(f"A mask's frame is 'handoff' or 'scan'; got {frame!r}.")
    entry = {k: v for k, v in entry.items() if v is not None}
    mask = Mask.from_dict(entry)
    if mask.kind == "vertices":
        return mask

    def point(p):
        p = np.asarray(p, dtype=np.float64)[None]
        return tuple(to_anny(p)[0] if frame == "scan" else _gltf_to_anny(p)[0])

    def direction(d):
        from opensculptboy.statues.scan import axis_vector

        d = axis_vector(d)[None]
        if frame == "scan":
            moved = to_anny(d) - to_anny(np.zeros((1, 3)))
        else:
            moved = _gltf_to_anny(d)
        return tuple(moved[0] / np.linalg.norm(moved[0]))

    if mask.centre is not None:
        mask.centre = point(mask.centre)
    if mask.point is not None:
        mask.point = point(mask.point)
    if mask.normal is not None:
        mask.normal = direction(mask.normal)
    if mask.kind == "box" and frame == "handoff" and mask.size is not None:
        # glTF (x, y, z) sizes are Anny's (x, z, y)
        x, y, z = mask.size
        mask.size = (x, z, y)
    elif frame == "scan":
        if mask.size is not None:
            if mask.kind == "box":
                raise ValueError(
                    "Give a box mask in the handoff's frame, which is upright."
                )
        if mask.radius is not None:
            mask.radius = float(mask.radius) * scale
    return mask


def _areas(vertices, triangles) -> np.ndarray:
    p = vertices[triangles]
    return 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)


def _compact(vertices, triangles):
    rows = np.unique(triangles)
    remap = np.full(len(vertices), -1, np.int64)
    remap[rows] = np.arange(len(rows))
    return vertices[rows], remap[triangles]


def scan_to_handoff(
    scan,
    out,
    stature: float = DEFAULT_STATURE,
    up=None,
    front=None,
    masks=(),
    landmarks: dict | None = None,
    landmark_frame: str = "handoff",
    plinth: bool = True,
    debris_share: float = DEBRIS_SHARE,
    source: str | None = None,
    licence: str | None = None,
    credit: str | None = None,
    title: str | None = None,
) -> StatueHandoff:
    """
    Write an appearance handoff folder from a statue scan (see the module documentation).

    Args:
        scan: a scan file (STL, OBJ, PLY, GLB), or ``(vertices, triangles)``.
        out: the folder to write.
        stature: the height (metres) of the figure above the plinth after scaling.
        up, front: the scan's up and front axes (``"+z"``, ``"-y"`` or three numbers);
            guessed by :func:`~opensculptboy.statues.scan.guess_up` and
            :func:`~opensculptboy.statues.scan.guess_front` when None.
        masks: :class:`~opensculptboy.statues.scan.Mask` objects or dicts, applied in order
            (a later mask wins), each with an optional ``"frame"``.
        landmarks: names of docs/reference-fit.md to points in ``landmark_frame``
            (``"handoff"`` or ``"scan"``); the highest point of the figure is the ``crown``
            when they do not give one.
        plinth: cut the plinth off (:func:`~opensculptboy.statues.scan.plinth_top`).
        debris_share: loose pieces of the skin under this share of its area are taken out.
        source, licence, credit, title: the scan's page, licence, credit line and name,
            kept in appearance.json and later in the card.
    """
    from opensculptboy.fit.handoff import write_handoff

    if isinstance(scan, (str, pathlib.Path)):
        scan_path = pathlib.Path(scan)
        V, F = load_scan(scan_path)
    else:
        scan_path = None
        V, F = (np.asarray(a) for a in scan)
        V = V.astype(np.float64)
        F = F.astype(np.int64)
    if up is None:
        up_vector, up_confidence = guess_up(V)
    else:
        from opensculptboy.statues.scan import axis_vector

        up_vector, up_confidence = axis_vector(up), 1.0
    if front is None:
        front_vector, front_score = guess_front(V, up_vector)
    else:
        from opensculptboy.statues.scan import axis_vector

        front_vector, front_score = axis_vector(front), 1.0
    rotation = frame_rotation(up_vector, front_vector)
    X = V @ rotation.T

    role = np.zeros(len(F), np.int64)  # 0 skin; see ROLE_NAMES
    names = ["skin", "plinth", "support", "break", "drapery", "hair", "debris"]
    top = plinth_top(X, F) if plinth else None
    floor = X[:, 2].min()
    if top is not None:
        margin = PLINTH_MARGIN * np.ptp(X[:, 2])
        role[(X[F, 2] < top + margin).all(1)] = names.index("plinth")
        floor = top
    height = X[:, 2].max() - floor
    scale = float(stature) / height
    used = np.unique(F[role == 0])
    shift = np.array([-X[used, 0].mean(), -X[used, 1].mean(), -floor], dtype=np.float64)
    translation = scale * shift
    A = scale * (X + shift)

    def to_anny(points):
        return scale * (np.asarray(points, dtype=np.float64) @ rotation.T + shift)

    for entry in masks or ():
        mask = _mask_in_anny(entry, to_anny, scale)
        chosen = select_triangles(A, F, mask)
        role[chosen] = names.index(mask.role)

    skin = role == 0
    if skin.any() and debris_share > 0:
        pieces = components(F[skin], len(A))
        area = np.bincount(pieces, _areas(A, F[skin]))
        small = area < debris_share * area.sum()
        rows = np.nonzero(skin)[0]
        role[rows[small[pieces]]] = names.index("debris")
        skin = role == 0
    if not skin.any():
        raise ValueError("Nothing of the scan is left for the fit after the masks.")

    skin_vertices, skin_triangles = _compact(A, F[skin])
    marks = {}
    if landmarks:
        for name, value in landmarks.items():
            p = np.asarray(value, dtype=np.float64)[None]
            marks[name] = (
                to_anny(p) if landmark_frame == "scan" else _gltf_to_anny(p)
            )[0]
    if not any(k.lower() in ("crown", "head_top", "vertex") for k in marks):
        marks["crown"] = skin_vertices[np.argmax(skin_vertices[:, 2])]

    parts = []
    for kind, part_role in (("drapery", "clothing"), ("hair", "hair")):
        chosen = role == names.index(kind)
        if chosen.any():
            v, t = _compact(A, F[chosen])
            parts.append((kind, v, t, part_role))
    out = pathlib.Path(out)
    write_handoff(out, skin_vertices, skin_triangles, marks, name="statue", parts=parts)

    _write_removed(out / "removed.glb", A, F, role, names)
    counts = {
        n: int((role == k).sum()) for k, n in enumerate(names) if (role == k).any()
    }
    appearance_path = out / "appearance.json"
    appearance = json.loads(appearance_path.read_text(encoding="utf-8"))
    appearance["statue"] = {
        "format": FORMAT,
        "title": title,
        "source": source,
        "licence": licence,
        "credit": credit,
        "scan_file": scan_path.name if scan_path is not None else None,
        "stature": float(stature),
        # glTF's frame of replica.glb from the scan's coordinates: g = scale * R s + t
        "transform": {
            "rotation": (np.asarray(_anny_to_gltf(rotation.T).T)).tolist(),
            "scale": scale,
            "translation": _anny_to_gltf(translation[None])[0].tolist(),
        },
        "up": [float(x) for x in up_vector],
        "front": [float(x) for x in front_vector],
        "up_confidence": round(up_confidence, 4),
        "front_score": round(front_score, 4),
        "plinth": top is not None,
        "triangles": counts,
        "removed": "removed.glb",
    }
    appearance_path.write_text(
        json.dumps(appearance, indent=2) + "\n", encoding="utf-8"
    )
    return StatueHandoff(
        folder=out,
        rotation=rotation,
        scale=scale,
        translation=translation,
        up=up_vector,
        front=front_vector,
        up_confidence=float(up_confidence),
        front_score=float(front_score),
        plinth=top is not None,
        triangles=counts,
        landmarks=marks,
    )


def _write_removed(path, vertices, triangles, role, names) -> None:
    """The removed pieces as one node per role in glTF's frame (an empty scene when none)."""
    import trimesh

    scene = trimesh.Scene()
    for k, name in enumerate(names):
        if name not in REMOVED_ROLES or not (role == k).any():
            continue
        v, t = _compact(vertices, triangles[role == k])
        scene.add_geometry(
            trimesh.Trimesh(_anny_to_gltf(v), t, process=False),
            node_name=name,
            geom_name=name,
        )
    if scene.geometry:
        scene.export(str(path))

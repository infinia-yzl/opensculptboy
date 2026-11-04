# OpenSculptBoy
# Apache License, Version 2.0
"""
The place of a rigid piece on a card: a prop moved by hand in the web viewer's studio, and the
bone that carries it.

A piece entry of a card (an outfit's piece or a loose wearable) may hold ``place`` (card
schema 6)::

    "place": {"bone": "metacarpal4.R", "rotation": [x, y, z, w], "translation": [x, y, z]}

``bone`` is the bone that carries the piece; ``rotation`` (a unit quaternion) and
``translation`` (metres) move the piece in the card's rest frame (Anny's frame), about the rest
head ``h`` of that bone on the card's body: a rest vertex ``x`` of the piece, as its rigid
binding gives it, goes to ``h + Q (x - h) + t``, and the piece then follows ``bone`` alone. The
axes are the card's own, so the page and the exporters agree without the bones' rest turns,
and the place follows the bone's head when the body's sliders move it. Without ``place`` a
piece stays where its package binds it. A piece with a place must have a rigid binding.

The web viewer's twin is ``viewer/src/wearables/place.ts``; ``test_wearable_place`` checks
both.
"""

from __future__ import annotations

import math

import numpy as np

PLACE_KEYS = {"bone", "rotation", "translation"}
# the schema of a card whose pieces hold a place
PLACE_SCHEMA = 6


def _numbers(value, count: int) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == count
        and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
            for x in value
        )
    )


def check_place(place, ref: str) -> None:
    """Raise ValueError for a ``place`` of the piece ``ref`` that does not hold."""
    if not isinstance(place, dict) or set(place) - PLACE_KEYS:
        raise ValueError(
            f"place of piece {ref!r} holds {sorted(PLACE_KEYS)}: {place!r}."
        )
    if not isinstance(place.get("bone"), str) or not place["bone"]:
        raise ValueError(f"place of piece {ref!r} names the bone that carries it.")
    rotation = place.get("rotation", [0.0, 0.0, 0.0, 1.0])
    if not _numbers(rotation, 4) or abs(math.hypot(*rotation) - 1.0) > 1e-3:
        raise ValueError(
            f"place.rotation of piece {ref!r} is a unit quaternion (x, y, z, w): "
            f"{rotation!r}."
        )
    if not _numbers(place.get("translation", [0.0, 0.0, 0.0]), 3):
        raise ValueError(
            f"place.translation of piece {ref!r} is three numbers in metres."
        )


def card_has_places(data: dict) -> bool:
    """Whether a card (as JSON) holds a ``place`` on any piece."""
    for outfit in data.get("outfits") or []:
        for piece in (outfit or {}).get("pieces") or []:
            if isinstance(piece, dict) and "place" in piece:
                return True
    return any(
        isinstance(e, dict) and "place" in e for e in data.get("wearables") or []
    )


def rotation_matrix(q) -> np.ndarray:
    """The (3, 3) rotation of a quaternion (x, y, z, w), normalised."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quaternion(R) -> list:
    """The unit quaternion (x, y, z, w, w >= 0) of a rotation (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
            0.25 * s,
        ]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-300))
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    if q[3] < 0:
        q = [-v for v in q]
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


def place_matrix(place: dict, head) -> np.ndarray:
    """The (4, 4) move of a place in the card's rest frame: x -> h + Q (x - h) + t."""
    Q = rotation_matrix(place.get("rotation", [0.0, 0.0, 0.0, 1.0]))
    t = np.asarray(place.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    h = np.asarray(head, dtype=np.float64)
    M = np.eye(4)
    M[:3, :3] = Q
    M[:3, 3] = h - Q @ h + t
    return M


def place_of(M, head, bone: str, digits: int = 7) -> dict:
    """The place (``bone``, ``rotation``, ``translation``) of a rigid move ``M`` (4, 4) of the
    rest frame about the rest head ``head`` of ``bone``."""
    M = np.asarray(M, dtype=np.float64)
    h = np.asarray(head, dtype=np.float64)
    t = M[:3, :3] @ h + M[:3, 3] - h
    return {
        "bone": bone,
        "rotation": [round(float(v), digits) for v in quaternion(M[:3, :3])],
        "translation": [round(float(v), digits) for v in t],
    }


def apply_place(place: dict, vertices, normals, head):
    """The rest vertices and normals (V, 3) of a piece moved by its place about ``head``."""
    M = place_matrix(place, head)
    V = np.asarray(vertices, dtype=np.float64) @ M[:3, :3].T + M[:3, 3]
    N = np.asarray(normals, dtype=np.float64) @ M[:3, :3].T
    return V, N


def carrier_weights(bone: int, count: int, max_influences: int):
    """Skin weights (count, k) and bone indices of a piece on one bone."""
    weights = np.zeros((count, max_influences))
    weights[:, 0] = 1.0
    indices = np.zeros((count, max_influences), np.int64)
    indices[:, 0] = bone
    return weights, indices


def rehold(
    place: dict | None,
    binding_bone: str,
    skin_old,
    skin_new,
    head_old,
    head_new,
    bone: str,
) -> dict:
    """The place that keeps a piece where it stands in a pose while ``bone`` takes it over:
    ``skin_old`` and ``skin_new`` (4, 4) are the skinning matrices (rest to posed) of the bone
    that carries it now and of ``bone``, and ``head_old``, ``head_new`` their rest heads. A
    piece without a place is carried by ``binding_bone`` with no move."""
    current = place or {"bone": binding_bone}
    A = place_matrix(current, head_old)
    M = (
        np.linalg.solve(
            np.asarray(skin_new, np.float64), np.asarray(skin_old, np.float64)
        )
        @ A
    )
    return place_of(M, head_new, bone)

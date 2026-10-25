# OpenSculptBoy
# Apache License, Version 2.0
"""
Dressing a body: the body of a card, the evaluation of each worn piece, its skin weights,
the layers and the push-out.

:func:`card_body` gives the skin of a character card at rest on the rows of Anny's template
(level 0 of the detail layer included), as a
:class:`~opensculptboy.wearables.bind.BodySurface`. :func:`dress` evaluates the worn pieces
on it from the lowest layer up. The push-out moves each vertex of a piece outside the body
and outside the pieces of lower layers: at least by ``gap`` (1 mm), or by as much as the
vertex lay outside them on the body that the piece was bound on, whichever is less. So a
piece keeps the relations it was made with (a strap that sinks into a coat on the binding
body sinks into it on every body), and a tight piece never sinks into a heavier body. A
relation to a lower piece counts only where the two pieces were made together
(:func:`made_together`); a lower piece made apart takes the full gap. The push also
moves the triangles of a piece over the vertices of a lower piece, by the same targets
along each lower vertex's normal, so that a lower piece denser than the piece over it
shows through none of its triangles. A Laplacian fade spreads each push over two rings of
the neighbouring vertices, and a final pass makes every vertex meet its target again.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from opensculptboy.wearables.bind import BodySurface, body_weights, evaluate
from opensculptboy.wearables.format import Garment, Wearable
from opensculptboy.wearables.geometry import (
    Surface,
    adjacency,
    line_partners,
    unit_rows,
    vertex_normals,
)

GAP = 0.001  # metres between a piece and what lies under it
DEPTH = 0.03  # metres: a vertex deeper inside a surface than this is left alone
PUSH_ROUNDS = 3
FADE_RINGS = 2
REACH = 0.03  # metres: the line from a lower piece's point to the piece over it


def anny_count(model) -> int:
    """The number of Anny's own rows in a model (the sculpt topology appends its own)."""
    from opensculptboy.topology.lookup import model_sculpt_data

    data = model_sculpt_data(model)
    return len(model.template_vertices) if data is None else int(data.vertex_count)


def skin_triangles(model) -> np.ndarray:
    """The triangles of the outer skin on Anny's rows, which garments bind to: anny's
    triangles without the eyeballs, the mouth cavity and the tongue."""
    from opensculptboy.export.body import triangulated_faces
    from opensculptboy.export.mouth import mouth_interior

    n = anny_count(model)
    interior = mouth_interior(model)
    triangles = (
        triangulated_faces(model)[0] if interior is None else interior.skin_triangles
    )
    triangles = np.asarray(triangles, dtype=np.int64)
    keep = (triangles < n).all(1)
    try:
        from opensculptboy.export.vrm_tables import eyeball_vertices

        eyes = np.zeros(len(model.template_vertices), bool)
        for rows in eyeball_vertices(model).values():
            eyes[rows] = True
        keep &= ~eyes[triangles].any(1)
    except ValueError:
        pass
    return triangles[keep]


def card_body(
    character, model=None, detail_dir=None, rest_vertices=None
) -> BodySurface:
    """
    The skin of a card at rest: the model's rest mesh of the card's sliders on Anny's rows,
    with the offsets of level 0 of the card's detail layer, the bones' rest poses and the
    model's skin weights. ``rest_vertices`` (the model's rows) replaces the evaluation.
    """
    import torch

    from opensculptboy.detail.layer import character_offsets

    if model is None:
        model = character.build_model(dtype=torch.float64)
    with torch.no_grad():
        out = model(
            phenotype_kwargs=dict(character.phenotype) or None,
            local_changes_kwargs=dict(character.local_changes) or None,
            face_shape_kwargs=dict(character.face_shapes) or None,
        )
    if rest_vertices is None:
        rest_vertices = out["rest_vertices"][0].detach().cpu().double().numpy()
        offsets = character_offsets(character, 0, detail_dir)
        if offsets is not None:
            rest_vertices = rest_vertices + offsets[: len(rest_vertices)]
    n = anny_count(model)
    return BodySurface(
        np.asarray(rest_vertices, dtype=np.float64)[:n],
        skin_triangles(model),
        model.vertex_bone_weights.detach().cpu().double().numpy()[:n],
        model.vertex_bone_indices.detach().cpu().numpy()[:n],
        out["rest_bone_poses"][0].detach().cpu().double().numpy(),
        model.bone_labels,
    )


@dataclasses.dataclass
class DressedPiece:
    """
    One piece on a body: the package, its rest vertices and normals (Anny's frame), its skin
    weights and bones, whether it shows, which of its groups other pieces hide now and which
    they can hide, and how far the push-out moved it.
    """

    ref: str
    piece: str
    outfit: str | None
    wearable: Wearable
    vertices: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    joints: np.ndarray
    shown: bool = True
    hidden_groups: list = dataclasses.field(default_factory=list)
    hideable_groups: list = dataclasses.field(default_factory=list)
    pushed: int = 0
    push_mm: float = 0.0
    source: str | None = None

    @property
    def garment(self) -> Garment:
        return self.wearable.garment

    def visible_triangles(self) -> np.ndarray:
        garment = self.garment
        keep = np.ones(len(garment.triangles), bool)
        for g, group in enumerate(garment.groups):
            if group in self.hidden_groups:
                keep &= garment.triangle_group != g
        return garment.triangles[keep]

    def extras(self) -> dict:
        w = self.wearable
        out = {
            "ref": self.ref,
            "piece": self.piece,
            "outfit": self.outfit,
            "name": w.name,
            "label": w.label or w.name,
            "slot": w.slot,
            "layer": int(w.layer),
            "provenance": w.provenance,
            "shown": bool(self.shown),
            "hidden_groups": list(self.hidden_groups),
            "pushed_vertices": int(self.pushed),
            "largest_push_mm": round(float(self.push_mm), 3),
        }
        if w.licence:
            out["licence"] = dict(w.licence)
        return out


def _fade(displacement: np.ndarray, moved: np.ndarray, triangles: np.ndarray):
    """Spread the displacement of the moved vertices over their neighbours (FADE_RINGS
    rings), fading out; the moved vertices keep theirs."""
    if not moved.any():
        return displacement
    A = adjacency(len(displacement), triangles)
    degree = np.maximum(np.asarray(A.sum(axis=1)).reshape(-1), 1.0)
    region = moved.copy()
    for _ in range(FADE_RINGS):
        region = region | (A @ region.astype(float) > 0)
    free = region & ~moved
    out = displacement.copy()
    for _ in range(FADE_RINGS * 4):
        mean = (A @ out) / degree[:, None]
        out[free] = 0.5 * mean[free]
    return out


def _outward(
    body: BodySurface, vertices: np.ndarray, normals: np.ndarray
) -> np.ndarray:
    """The unit ``normals`` of a piece's ``vertices`` turned away from the body where they
    face it (the body's normal at the closest point decides)."""
    found = body.surface.closest(vertices)
    skin = body.surface.normal(found.triangle, found.barycentric)
    normals = unit_rows(np.asarray(normals, dtype=np.float64))
    return (
        normals * np.where(np.einsum("ij,ij->i", normals, skin) < 0, -1.0, 1.0)[:, None]
    )


def made_together(upper: Wearable, lower: Wearable) -> bool:
    """Whether two packages were made as one set, so that the relation between them on
    their binding body is part of their design: both are bound on the same body, and that
    body is a card's or a reference's own body. Anny's template is the common body of
    packages made apart (a MakeHuman asset, a garment bound with ``wearables bind``), so
    the relation of two packages on it says nothing about how they layer."""
    a, b = upper.binding, lower.binding
    return (
        a.body != "anny-template"
        and a.body == b.body
        and np.array_equal(a.body_vertices, b.body_vertices)
    )


def relation_home(upper, lower, body: BodySurface):
    """The rest vertices of two dressed pieces (or packages) on the binding body of the
    upper one, where they were made together (:func:`made_together`): (upper vertices,
    lower vertices), or None where the relation as made is unknown."""
    upper_w = getattr(upper, "wearable", upper)
    lower_w = getattr(lower, "wearable", lower)
    if not made_together(upper_w, lower_w):
        return None
    home = body
    if not np.array_equal(upper_w.binding.body_vertices, body.vertices):
        home = body.with_vertices(upper_w.binding.body_vertices)
    upper_home = _rest_on(upper_w, home, body)
    lower_home = _rest_on(lower_w, home, body)
    if upper_home is None or lower_home is None:
        return None
    return upper_home[0], lower_home[0]


def _rest_on(wearable: Wearable, home: BodySurface, body: BodySurface):
    """
    The rest vertices and normals of a lower piece on ``home``, the binding body of an upper
    piece, or None where they are unknown. ``home`` carries the bones of ``body`` (the body
    being dressed), so a rigid piece takes its own vertices on its own binding body, and has
    no known place on another binding body.
    """
    if wearable.binding.method == "surface" or home is body:
        return evaluate(wearable.binding, home)
    if np.array_equal(wearable.binding.body_vertices, home.vertices):
        return wearable.garment.vertices, wearable.garment.normals
    return None


def _cover_points(vertices, triangles, points, directions, targets):
    """Move the triangles of a piece so that each point lies under its partner (the point
    of the triangles on the line along the point's direction, ahead first; found once)
    by at least its target (metres along the direction): the corners of each partner's
    triangle move along the direction by their barycentric share, the largest move of
    each corner wins. Returns the moves (V, 3)."""
    hit, triangle, barycentric, _ = line_partners(
        points, directions, vertices, triangles, REACH
    )
    rows = np.flatnonzero(hit)
    corners = triangles[triangle[rows]]
    barycentric = barycentric[rows]
    directions = directions[rows]
    points = points[rows]
    targets = targets[rows]

    def moves(current):
        foot = np.einsum("nk,nkj->nj", barycentric, current[corners])
        along = np.einsum("ij,ij->i", foot - points, directions)
        need = targets - along
        bad = (need > 1e-7) & (along > -DEPTH)
        out = np.zeros_like(current)
        if not bad.any():
            return out
        amount = need[bad] / np.maximum((barycentric[bad] ** 2).sum(axis=1), 1e-12)
        size = np.zeros(len(current))
        way = np.zeros_like(current)
        for i in range(3):
            share = barycentric[bad, i] * amount
            np.maximum.at(size, corners[bad, i], share)
            np.add.at(way, corners[bad, i], share[:, None] * directions[bad])
        moving = size > 0
        out[moving] = size[moving, None] * unit_rows(way[moving])
        return out

    return moves


def _push(
    vertices: np.ndarray,
    normals: np.ndarray,
    triangles: np.ndarray,
    colliders: list[Surface],
    targets: list[np.ndarray],
    covered: list = (),
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Move vertices outside the colliders by at least their targets (signed metres along
    each collider's normal at the closest point), and move triangles over the points of
    ``covered`` ((points, outward directions, targets) of each lower piece), so that a
    lower piece denser than this one shows through none of its triangles."""
    start = vertices.copy()
    moved = np.zeros(len(vertices), bool)
    movers = [
        _cover_points(vertices, triangles, points, directions, target)
        for points, directions, target in covered
    ]
    for final in [False] * PUSH_ROUNDS + [True]:
        # the triangles first: the vertices' targets, met last, hold after the pass
        for mover in movers:
            step = mover(vertices)
            some = np.abs(step).sum(axis=1) > 0
            if some.any():
                vertices = vertices + step
                moved |= some
        for surface, target in zip(colliders, targets):
            signed, found, n = surface.signed_distance(vertices)
            bad = (signed < target - 1e-7) & (found.distance < DEPTH)
            if bad.any():
                vertices[bad] += (target - signed)[bad, None] * n[bad]
                moved |= bad
        if not final:
            vertices = start + _fade(vertices - start, moved, triangles)
    shift = np.linalg.norm(vertices - start, axis=1)
    if moved.any() and len(triangles):
        # new normals where the push moved the surface; the package's own elsewhere (hard
        # edges and split normals stay)
        fresh = vertex_normals(vertices, triangles)
        changed = shift > 1e-9
        normals = normals.copy()
        normals[changed] = fresh[changed]
    return vertices, normals, int((shift > 1e-7).sum()), float(shift.max(initial=0.0))


def dress(
    body: BodySurface,
    pieces: list,
    max_influences: int = 4,
    push: bool = True,
    gap: float = GAP,
    skin_weights: tuple | None = None,
) -> list[DressedPiece]:
    """
    Evaluate worn pieces on a body (see the module documentation).

    Args:
        body: the body (:func:`card_body`).
        pieces: :class:`~opensculptboy.wearables.wardrobe.WornPiece` items; the pieces that do
            not show are evaluated too (without a push), so that a wardrobe file can hold them.
        max_influences: bones per vertex of the pieces' skin weights.
        push: run the push-out.
        gap: the gap of the push-out in metres.
        skin_weights: (weights, indices) of Anny's rows to blend in place of the body's own
            (for example the weights of a VRM file).

    Returns:
        One :class:`DressedPiece` per piece, in the order given.
    """
    weight_body = body
    if skin_weights is not None:
        weight_body = BodySurface(
            body.vertices,
            body.triangles,
            np.asarray(skin_weights[0])[: len(body.vertices)],
            np.asarray(skin_weights[1])[: len(body.vertices)],
            body.bone_poses,
            body.bone_labels,
        )
    bind_bodies: dict[int, BodySurface] = {}

    def bind_body(wearable: Wearable) -> BodySurface:
        key = id(wearable.binding.body_vertices)
        if key not in bind_bodies:
            same = np.array_equal(wearable.binding.body_vertices, body.vertices)
            bind_bodies[key] = (
                body if same else body.with_vertices(wearable.binding.body_vertices)
            )
        return bind_bodies[key]

    order = sorted(range(len(pieces)), key=lambda k: int(pieces[k].wearable.layer))
    dressed: list[DressedPiece | None] = [None] * len(pieces)
    placed: list[tuple[int, DressedPiece]] = []  # (layer, piece) of the shown pieces
    for k in order:
        worn = pieces[k]
        wearable = worn.wearable
        vertices, normals = evaluate(wearable.binding, body)
        pushed, push_mm = 0, 0.0
        triangles = wearable.garment.triangles
        if push and worn.shown and wearable.binding.method == "surface":
            home = bind_body(wearable)
            colliders = [body.surface]
            home_vertices = evaluate(wearable.binding, home)[0]
            targets = [np.minimum(home.surface.signed_distance(home_vertices)[0], gap)]
            covered = []
            for layer, lower in placed:
                if layer >= int(wearable.layer):
                    continue
                tris = lower.visible_triangles()
                if len(tris) == 0:
                    continue
                colliders.append(Surface(lower.vertices, tris, lower.normals))
                rows = np.unique(tris)
                points = lower.vertices[rows]
                directions = _outward(
                    body, lower.vertices, vertex_normals(lower.vertices, tris)
                )
                lower_home = None
                if made_together(wearable, lower.wearable):
                    lower_home = _rest_on(lower.wearable, home, body)
                if lower_home is None:  # the relation as made is unknown
                    targets.append(np.full(len(home_vertices), gap))
                    covered.append((points, directions[rows], np.full(len(rows), gap)))
                    continue
                then = Surface(lower_home[0], tris, lower_home[1])
                targets.append(np.minimum(then.signed_distance(home_vertices)[0], gap))
                # the relation as made of the lower piece's points under this piece
                home_points = lower_home[0][rows]
                home_directions = _outward(
                    home, lower_home[0], vertex_normals(lower_home[0], tris)
                )[rows]
                hit, _, _, along = line_partners(
                    home_points, home_directions, home_vertices, triangles, REACH
                )
                covered.append(
                    (
                        points,
                        directions[rows],
                        np.where(hit, np.minimum(along, gap), -np.inf),
                    )
                )
            vertices, normals, pushed, push_mm = _push(
                vertices, normals, triangles, colliders, targets, covered
            )
            push_mm *= 1000.0
        weights, joints = body_weights(wearable.binding, weight_body, max_influences)
        piece = DressedPiece(
            ref=worn.ref,
            piece=worn.piece,
            outfit=worn.outfit,
            wearable=wearable,
            vertices=vertices,
            normals=normals,
            weights=weights,
            joints=joints,
            shown=worn.shown,
            hidden_groups=list(worn.hidden_groups),
            hideable_groups=list(getattr(worn, "hideable_groups", [])),
            pushed=pushed,
            push_mm=push_mm,
            source=getattr(worn, "path", None),
        )
        dressed[k] = piece
        if worn.shown:
            placed.append((int(wearable.layer), piece))
    return dressed

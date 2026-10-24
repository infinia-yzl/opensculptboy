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
Laplacian fade spreads each push over two rings of the neighbouring vertices, and a final
pass makes every vertex meet its target again.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from opensculptboy.wearables.bind import BodySurface, body_weights, evaluate
from opensculptboy.wearables.format import Garment, Wearable
from opensculptboy.wearables.geometry import Surface, adjacency, vertex_normals

GAP = 0.001  # metres between a piece and what lies under it
DEPTH = 0.03  # metres: a vertex deeper inside a surface than this is left alone
PUSH_ROUNDS = 3
FADE_RINGS = 2


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


def _push(
    vertices: np.ndarray,
    normals: np.ndarray,
    triangles: np.ndarray,
    colliders: list[Surface],
    targets: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Move vertices outside the colliders by at least their targets (signed metres along
    each collider's normal at the closest point)."""
    start = vertices.copy()
    moved = np.zeros(len(vertices), bool)
    for final in [False] * PUSH_ROUNDS + [True]:
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
            for layer, lower in placed:
                if layer >= int(wearable.layer):
                    continue
                tris = lower.visible_triangles()
                if len(tris) == 0:
                    continue
                colliders.append(Surface(lower.vertices, tris, lower.normals))
                lower_home = _rest_on(lower.wearable, home, body)
                if lower_home is None:  # the relation on the binding body is unknown
                    targets.append(np.full(len(home_vertices), gap))
                    continue
                then = Surface(lower_home[0], tris, lower_home[1])
                targets.append(np.minimum(then.signed_distance(home_vertices)[0], gap))
            vertices, normals, pushed, push_mm = _push(
                vertices, normals, triangles, colliders, targets
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

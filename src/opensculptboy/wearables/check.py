# OpenSculptBoy
# Apache License, Version 2.0
"""
The fit check of a card's outfit (``opensculptboy wearables check``, docs/wearables.md,
"Check").

:func:`check_card` runs the quality checks of every worn piece and dresses the outfit on a set
of test bodies (:data:`BODIES`: the card's own body, the default body, and the card with the
lowest and the highest weight, muscle and height). On each body it measures two faults at
rest, in the T-pose of the VRM export and on sampled frames of the built-in clips:

- **skin through**: a body vertex under a piece's cover mask (taken on that body at rest,
  :func:`opensculptboy.wearables.cover.piece_cover`) that lies more than the tolerance (2 mm)
  outside every piece that covers it;
- **layers through**: a vertex of a lower piece that an upper piece covers (the line along
  its normal meets the upper piece within 3 cm on either side, on that body at rest) and that
  lies more than the tolerance outside the upper piece, beyond how far it lay outside on the
  binding body of two pieces made together (the relation as made, which the push-out keeps:
  :func:`opensculptboy.wearables.fit.made_together`); the vertices within the relation as
  made count apart (``as_made``).

"Outside" is measured from the vertex's partner on the piece (:class:`Partners`): the
nearest point of the piece at rest ahead on the line along the vertex's normal turned away
from the body (within 3 cm), else the nearest point behind it (within 3 cm), else its
closest point (within 5 cm; a vertex further away takes no part),
carried into each pose by the piece's own triangle and barycentric coordinates, along the
piece's normal there, turned away from the body. A partner keeps each vertex to the part of
the piece above it, so that an arm against the torso is measured against its sleeve and
never against the side of the shirt.

The body and the pieces move by linear blend skinning with one set of bone transforms: the
body with its own skin weights, the pieces with the weights of the dressing; in the T-pose
both take the weights of the VRM file (:func:`opensculptboy.export.tpose.file_skin_weights`,
blended at the pieces' bound points). Each fault is counted per piece and per body region
(the region of the vertex's strongest bone).

The gates: no quality item of error severity; zero vertices of either fault at rest; in the
T-pose and in the clips, at most ``pose_limit`` per cent of the covered vertices. Each gate
gives its worst body, state, frame, region and piece. :func:`review_sheet` draws each body
at rest and in its worst state from the front, the side and the back, with the faulty
vertices marked.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

from opensculptboy.wearables.format import PRIVATE, WearableError, check_destination

FORMAT = "opensculptboy.check/1"
TOLERANCE = 0.002  # metres outside a piece that count as a fault
NEAR = 0.05  # metres: a vertex further from the piece than this does not count
RAY = 0.03  # metres: the reach of the cover test between pieces
POSE_LIMIT = 1.0  # per cent of the covered vertices allowed in the T-pose and the clips
FRAMES = 4  # frames per clip
BODIES = (
    "card",
    "default",
    "weight-0",
    "weight-1",
    "muscle-0",
    "muscle-1",
    "height-0",
    "height-1",
)
SLIDERS = ("weight", "muscle", "height")
SKIN_MARK = (225, 25, 25)
LAYER_MARK = (255, 150, 0)
AS_MADE_MARK = (130, 60, 200)


class CheckError(WearableError):
    """A check that cannot run as asked."""

    code = "bad_option"


def region_of(label: str) -> str:
    """The body region of a bone: head, torso, hips, upper_arm, forearm, hand, thigh, shin
    or foot, with the side (``.L``, ``.R``) of a paired bone."""
    base, _, side = label.partition(".")
    side = f".{side}" if side else ""
    table = (
        (("neck", "head", "eye", "jaw", "tongue", "teeth"), "head"),
        (("upperarm",), "upper_arm"),
        (("lowerarm",), "forearm"),
        (("wrist", "finger", "metacarpal", "thumb", "hand"), "hand"),
        (("upperleg",), "thigh"),
        (("lowerleg",), "shin"),
        (("foot", "toe"), "foot"),
        (("root", "pelvis"), "hips"),
        (("spine", "clavicle", "shoulder", "breast"), "torso"),
    )
    for prefixes, name in table:
        if base.startswith(prefixes):
            return name if name in ("head", "torso", "hips") else name + side
    return "other"


def _regions(weights, indices, labels) -> np.ndarray:
    names = np.array([region_of(label) for label in labels])
    strongest = np.take_along_axis(
        np.asarray(indices), np.argmax(weights, axis=1)[:, None], axis=1
    )[:, 0]
    return names[strongest]


@dataclasses.dataclass
class TestBody:
    name: str
    character: object
    detail: bool  # the card's detail offsets (the card's own body only)


def test_bodies(character, names) -> list[TestBody]:
    """The test bodies of ``names``: ``card``, ``default``, ``SLIDER-0`` and ``SLIDER-1``
    (the card with one of :data:`SLIDERS` at its end), or ``SLIDER=VALUE`` for any phenotype
    slider of the card's model at a value from 0 to 1."""
    from opensculptboy.character import Character

    out, seen = [], set()
    for name in names:
        name = name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        if name == "card":
            out.append(TestBody(name, character, True))
            continue
        if name == "default":
            out.append(
                TestBody(
                    name,
                    Character(
                        rig=character.rig,
                        topology=character.topology,
                        phenotypes=character.phenotypes,
                    ),
                    False,
                )
            )
            continue
        slider, value = None, None
        if "=" in name:
            slider, _, text = name.partition("=")
            try:
                value = float(text)
            except ValueError:
                value = None
        elif name.rsplit("-", 1)[0] in SLIDERS and name.rsplit("-", 1)[-1] in "01":
            slider, value = name.rsplit("-", 1)[0], float(name.rsplit("-", 1)[1])
        if slider is None or value is None or not 0.0 <= value <= 1.0:
            raise CheckError(
                f"Unknown test body {name!r}.",
                "bad_option",
                f"Use {', '.join(BODIES)}, or SLIDER=VALUE with a value from 0 to 1.",
            )
        variant = dataclasses.replace(
            character, phenotype={**character.phenotype, slider: value}, detail={}
        )
        out.append(TestBody(name, variant, False))
    if not out:
        raise CheckError("No test bodies.", "bad_option", "Name one or more bodies.")
    return out


def _forward(model, character, pose=None):
    import torch

    kwargs = dict(
        phenotype_kwargs=dict(character.phenotype) or None,
        local_changes_kwargs=dict(character.local_changes) or None,
        face_shape_kwargs=dict(character.face_shapes) or None,
    )
    with torch.no_grad():
        if pose is None:
            return model(**kwargs)
        return model(pose_parameters=pose, pose_parameterization="local-ref", **kwargs)


def _clip_frames(count: int, frames: int) -> list[int]:
    if count <= frames:
        return list(range(count))
    return sorted({int(k) for k in np.linspace(0, count, frames, endpoint=False)})


def pose_states(model, character, clips, frames):
    """``(state, frame, transforms)`` of the clips: the bone transforms (J, 4, 4) of each
    sampled frame (posed world matrix times the inverse rest matrix), and of the card's own
    pose when it has one (state ``pose``)."""
    import anny.poses

    from opensculptboy.export.tpose import rigid_inverse

    out = []
    pose = character.pose_parameters(model)
    if pose is not None:
        res = _forward(model, character, pose)
        rest = res["rest_bone_poses"][0].double().cpu().numpy()
        posed = res["bone_poses"][0].double().cpu().numpy()
        out.append(("pose", 0, posed @ rigid_inverse(rest)))
    for clip in clips:
        params = anny.poses.pose_parameters(
            model,
            clip,
            phenotype_kwargs=dict(character.phenotype) or None,
            local_changes_kwargs=dict(character.local_changes) or None,
            grounded=False,
        )["pose_parameters"]
        picks = _clip_frames(int(params.shape[0]), frames)
        res = _forward(
            model, character, params[picks].to(model.template_vertices.dtype)
        )
        rest = res["rest_bone_poses"][0].double().cpu().numpy()
        posed = res["bone_poses"].double().cpu().numpy()
        inverse = rigid_inverse(rest)
        for k, frame in enumerate(picks):
            out.append((clip, int(frame), posed[k] @ inverse))
    return out


def _visible_rows(piece) -> np.ndarray:
    return np.unique(piece.visible_triangles())


def ray_partners(origins, directions, vertices, triangles, reach: float = RAY):
    """The partner of each origin on the line along its direction, within ``reach`` on
    either side: the nearest point of the triangles ahead of the origin, else the nearest
    point behind it: (hit (N,), triangle (N,), barycentric (N, 3)), Moller-Trumbore on the
    triangles near each origin. With the direction turned away from the body, a point
    between two walls of a piece (the sole and the upper of a boot, the two sides of a
    glove) takes the outer wall, so that it counts as under the piece."""
    from scipy.spatial import cKDTree

    from opensculptboy.wearables.geometry import unit_rows

    origins = np.asarray(origins, dtype=np.float64)
    directions = unit_rows(np.asarray(directions, dtype=np.float64))
    count = len(origins)
    best = np.full(count, np.inf)
    triangle = np.zeros(count, np.int64)
    barycentric = np.zeros((count, 3))
    if not len(triangles) or not count:
        return np.isfinite(best), triangle, barycentric
    corners = vertices[triangles]
    centroids = corners.mean(axis=1)
    radius = float(np.linalg.norm(corners - centroids[:, None], axis=2).max())
    near = cKDTree(centroids).query_ball_point(origins, reach + radius)
    rows = np.repeat(np.arange(count), [len(n) for n in near])
    if not len(rows):
        return np.isfinite(best), triangle, barycentric
    cols = np.concatenate([np.asarray(n, dtype=np.int64) for n in near])
    for start in range(0, len(rows), 400_000):
        r, c = rows[start : start + 400_000], cols[start : start + 400_000]
        a, b, cc = (corners[c, i] for i in range(3))
        d = directions[r]
        e1, e2 = b - a, cc - a
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = origins[r] - a
        u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, e1)
        v = np.einsum("ij,ij->i", d, q) * inv
        along = np.einsum("ij,ij->i", e2, q) * inv
        # the rank of a hit: the distance ahead of the origin, or the reach plus the
        # distance behind it, so that every hit ahead ranks before every hit behind
        t = np.where(along >= 0.0, along, reach - along)
        good = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (np.abs(along) <= reach)
        k = np.flatnonzero(good)
        k = k[np.lexsort((c[k], t[k], r[k]))]  # per origin, the best hit first
        first = np.ones(len(k), bool)
        first[1:] = r[k][1:] != r[k][:-1]
        k = k[first]
        k = k[t[k] < best[r[k]]]  # nearer than a hit of an earlier chunk
        best[r[k]] = t[k]
        triangle[r[k]] = c[k]
        barycentric[r[k]] = np.stack([1.0 - u[k] - v[k], u[k], v[k]], axis=1)
    return np.isfinite(best), triangle, barycentric


@dataclasses.dataclass
class Partners:
    """Points of a set of ``rows`` (body or lower-piece vertices) and their partners on a
    piece at rest (``triangle``, an index into the piece's visible triangles, and
    ``barycentric``), which each pose carries along: the nearest point of the piece ahead
    on the line along the point's normal (turned away from the body) within 3 cm, else the
    nearest point behind it within 3 cm, else the closest point of the piece within 5 cm.
    ``side`` (+1 or -1) turns the piece's normal at the partner away from the body:
    the normal of the triangles' winding, turned where it faces the skin at rest, so that a
    piece wound inside out, or the far side of a double-sided panel, still measures
    outward."""

    rows: np.ndarray
    triangle: np.ndarray
    barycentric: np.ndarray
    side: np.ndarray

    @classmethod
    def at_rest(cls, points, rows, piece, body, directions=None):
        from opensculptboy.wearables.geometry import (
            Surface,
            interpolate,
            unit_rows,
            vertex_normals,
        )

        tris = piece.visible_triangles()
        if not len(rows) or not len(tris):
            empty = np.zeros(0, np.int64)
            return cls(empty, empty, np.zeros((0, 3)), np.zeros(0))
        found = Surface(piece.vertices, tris).closest(points[rows])
        near = found.distance < NEAR
        triangle, barycentric = found.triangle.copy(), found.barycentric.copy()
        if directions is not None:
            hit, ray_triangle, ray_barycentric = ray_partners(
                points[rows], directions, piece.vertices, tris
            )
            triangle[hit] = ray_triangle[hit]
            barycentric[hit] = ray_barycentric[hit]
            near |= hit
        triangle, barycentric = triangle[near], barycentric[near]
        corners = tris[triangle]
        n = unit_rows(
            interpolate(vertex_normals(piece.vertices, tris), corners, barycentric)
        )
        foot = interpolate(piece.vertices, corners, barycentric)
        skin = body.surface.closest(foot)
        outward = body.surface.normal(skin.triangle, skin.barycentric)
        side = np.where(np.einsum("ij,ij->i", n, outward) < 0, -1.0, 1.0)
        return cls(rows[near], triangle, barycentric, side)

    def signed(self, points, piece, piece_vertices) -> np.ndarray:
        """The signed distance of each point (``points[rows]``) from its partner, along the
        piece's outward normal there, with the piece at ``piece_vertices``."""
        from opensculptboy.wearables.geometry import (
            interpolate,
            unit_rows,
            vertex_normals,
        )

        tris = piece.visible_triangles()
        corners = tris[self.triangle]
        foot = interpolate(piece_vertices, corners, self.barycentric)
        normals = vertex_normals(piece_vertices, tris)
        n = unit_rows(interpolate(normals, corners, self.barycentric))
        return self.side * np.einsum("ij,ij->i", points[self.rows] - foot, n)


@dataclasses.dataclass
class Layout:
    """What one body's measures need from the rest: the shown pieces, the covered body
    vertices of each piece with their partners, the covered vertices of each pair of
    layers with theirs, and the regions."""

    body: object
    pieces: list
    covers: list  # Partners of the body vertices under each piece
    pairs: list  # (upper index, lower index, Partners of the lower piece's vertices)
    body_regions: np.ndarray
    piece_regions: list

    @property
    def covered(self) -> np.ndarray:
        mask = np.zeros(len(self.body.vertices), bool)
        for partners in self.covers:
            mask[partners.rows] = True
        return mask


def layout(body, dressed) -> Layout:
    """The covers and pairs of a dressed body at rest (see the module documentation)."""
    from opensculptboy.wearables.cover import piece_cover
    from opensculptboy.wearables.geometry import vertex_normals

    from opensculptboy.wearables.fit import relation_home

    pieces = [p for p in dressed if p.shown]
    covers = []
    for p in pieces:
        rows = np.flatnonzero(piece_cover(body, p))
        covers.append(
            Partners.at_rest(body.vertices, rows, p, body, body.normals[rows])
        )
    pairs = []
    for u, upper in enumerate(pieces):
        tris = upper.visible_triangles()
        if not len(tris):
            continue
        for k, lower in enumerate(pieces):
            if int(lower.wearable.layer) >= int(upper.wearable.layer):
                continue
            rows = _visible_rows(lower)
            if not len(rows):
                continue
            normals = vertex_normals(lower.vertices, lower.visible_triangles())[rows]
            # turned away from the body, so that the partner ahead lies outward
            skin = body.surface.closest(lower.vertices[rows])
            outward = body.surface.normal(skin.triangle, skin.barycentric)
            normals *= np.where(np.einsum("ij,ij->i", normals, outward) < 0, -1, 1)[
                :, None
            ]
            hit = ray_partners(lower.vertices[rows], normals, upper.vertices, tris)[0]
            if not hit.any():
                continue
            partners = Partners.at_rest(
                lower.vertices, rows[hit], upper, body, normals[hit]
            )
            # the relation as made: how far each vertex lay outside the upper piece on
            # the binding body of two pieces made together, which the push-out keeps
            allowance = np.zeros(len(partners.rows))
            home = relation_home(upper, lower, body)
            if home is not None:
                allowance = np.maximum(partners.signed(home[1], upper, home[0]), 0.0)
            pairs.append((u, k, partners, allowance))
    return Layout(
        body,
        pieces,
        covers,
        pairs,
        _regions(body.weights, body.indices, body.bone_labels),
        [_regions(p.weights, p.joints, body.bone_labels) for p in pieces],
    )


def _counts(regions: np.ndarray) -> dict:
    names, counts = np.unique(regions, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return {str(names[i]): int(counts[i]) for i in order}


def measure(lay: Layout, body_vertices, piece_vertices, tolerance=TOLERANCE):
    """The faults of one state from the posed body vertices (N, 3) and the posed vertices of
    each shown piece of the layout: (the counts, the faulty points to mark)."""
    covered = lay.covered
    showing = np.zeros(len(body_vertices), bool)
    holds = np.zeros(len(body_vertices), bool)
    largest = np.zeros(len(body_vertices))
    per_piece = {}
    for piece, vertices, partners in zip(lay.pieces, piece_vertices, lay.covers):
        rows = partners.rows
        if not len(rows):
            continue
        signed = partners.signed(body_vertices, piece, vertices)
        outside = signed > tolerance
        holds[rows[~outside]] = True
        showing[rows[outside]] = True
        largest[rows[outside]] = np.maximum(largest[rows[outside]], signed[outside])
        if outside.any():
            per_piece[piece.ref] = int(outside.sum())
    showing &= ~holds
    skin_rows = np.flatnonzero(showing)
    skin = {
        "showing": int(len(skin_rows)),
        "covered": int(covered.sum()),
        "share_percent": round(100.0 * len(skin_rows) / max(int(covered.sum()), 1), 3),
        "largest_mm": round(float(largest[skin_rows].max(initial=0.0)) * 1000.0, 2),
        "pieces": per_piece,
        "regions": _counts(lay.body_regions[skin_rows]),
    }
    cut_points, cut_regions, per_pair = [], [], {}
    made_points, per_made = [], {}
    cutting = covered_pairs = as_made = 0
    worst_mm = 0.0
    for u, k, partners, allowance in lay.pairs:
        upper, lower = lay.pieces[u], lay.pieces[k]
        rows = partners.rows
        signed = partners.signed(piece_vertices[k], upper, piece_vertices[u])
        bad = signed > tolerance + allowance
        made = (signed > tolerance) & ~bad
        if made.any():
            as_made += int(made.sum())
            per_made[f"{lower.ref} through {upper.ref}"] = int(made.sum())
            made_points.append(piece_vertices[k][rows[made]])
        covered_pairs += len(rows)
        if bad.any():
            cutting += int(bad.sum())
            per_pair[f"{lower.ref} through {upper.ref}"] = int(bad.sum())
            cut_points.append(piece_vertices[k][rows[bad]])
            cut_regions.append(lay.piece_regions[k][rows[bad]])
            worst_mm = max(worst_mm, float(signed[bad].max()))
    layers = {
        "cutting": cutting,
        "covered": int(covered_pairs),
        "share_percent": round(100.0 * cutting / max(covered_pairs, 1), 3),
        "largest_mm": round(worst_mm * 1000.0, 2),
        "pairs": per_pair,
        "regions": _counts(np.concatenate(cut_regions)) if cut_regions else {},
        "as_made": as_made,
        "as_made_pairs": per_made,
    }
    marks = {
        "skin": body_vertices[skin_rows],
        "layers": np.concatenate(cut_points) if cut_points else np.zeros((0, 3)),
        "as_made": np.concatenate(made_points) if made_points else np.zeros((0, 3)),
    }
    return {"skin": skin, "layers": layers}, marks


def _tpose(model, body, pieces):
    """The T-pose transforms of a body, and the VRM file's weights of the body and of each
    piece (blended at the pieces' bound points), or the dressing's weights with a note when
    the file's weights cannot be had."""
    import torch

    from opensculptboy.export.tpose import file_skin_weights, rigid_inverse, vrm_t_pose
    from opensculptboy.wearables.bind import BodySurface, body_weights

    B = body.bone_poses
    T = vrm_t_pose(model, torch.as_tensor(B, dtype=torch.float64)).double().numpy()
    transforms = T @ rigid_inverse(B)
    note = None
    try:
        weights, indices = file_skin_weights(model, "1.0")
        n = len(body.vertices)
        weights, indices = np.asarray(weights)[:n], np.asarray(indices)[:n]
        weight_body = BodySurface(
            body.vertices, body.triangles, weights, indices, B, body.bone_labels
        )
        piece_weights = [
            body_weights(p.wearable.binding, weight_body, 4) for p in pieces
        ]
    except (ValueError, KeyError, IndexError, OSError) as error:
        note = f"the T-pose uses the dressing's weights ({error})"
        weights, indices = body.weights, body.indices
        piece_weights = [(p.weights, p.joints) for p in pieces]
    return transforms, (weights, indices), piece_weights, note


def _gate(name, states, key, limit, unit) -> dict:
    """A gate over the states of one kind: pass when the worst value keeps to the limit."""
    if not states:
        return {
            "name": name,
            "pass": True,
            "skipped": True,
            "limit": limit,
            "unit": unit,
        }
    field = "showing" if key == "skin" else "cutting"

    def value(s):
        part = s[key]
        return part[field] if unit == "vertices" else part["share_percent"]

    worst = max(states, key=lambda s: (value(s), s[key][field]))
    part = worst[key]
    detail = part["pieces"] if key == "skin" else part["pairs"]
    return {
        "name": name,
        "pass": bool(value(worst) <= limit),
        "limit": limit,
        "unit": unit,
        "value": value(worst),
        "worst": {
            "body": worst["body"],
            "state": worst["state"],
            "frame": worst["frame"],
            "count": part[field],
            "share_percent": part["share_percent"],
            "largest_mm": part["largest_mm"],
            "region": next(iter(part["regions"]), None),
            "piece": max(detail, key=detail.get) if detail else None,
        },
    }


def check_card(
    card,
    *,
    outfit: str | None = None,
    bodies=BODIES,
    clips=None,
    frames: int = FRAMES,
    tolerance: float = TOLERANCE,
    pose_limit: float = POSE_LIMIT,
    sheet=None,
    tpose: bool = True,
    log=None,
) -> dict:
    """
    The check of a card's worn outfit (``outfit`` overrides ``active_outfit``) on ``bodies``
    (:func:`test_bodies`), at rest, in the T-pose (``tpose``) and on ``frames`` frames of each
    clip of ``clips`` (every clip of the pose library when None). Writes the review sheet to
    ``sheet`` when given. Returns the report (see the module documentation).
    """
    import anny.poses
    import torch

    from opensculptboy.export.tpose import skin
    from opensculptboy.wearables.cardedit import CardError, _read_card
    from opensculptboy.wearables.fit import card_body, dress
    from opensculptboy.wearables.wardrobe import worn_pieces

    from opensculptboy.character import Character

    log = log or (lambda message: None)
    path, raw = _read_card(card)
    character = Character.from_dict(raw)
    detail_dir = path.parent
    library = anny.poses.names("loop")
    clips = list(library if clips is None else clips)
    unknown = [c for c in clips if c not in library]
    if unknown:
        raise CheckError(
            f"Unknown clips {unknown}.",
            "bad_option",
            f"Use clips of the pose library: {', '.join(library)}, or 'none'.",
        )
    if frames < 1:
        raise CheckError(
            "--frames must be 1 or more.", "bad_option", "Pass --frames 4."
        )
    try:
        worn = worn_pieces(character, detail_dir, outfit)
    except FileNotFoundError as error:
        raise CardError(
            f"{error}", "package_not_found", "Fix the piece's file path on the card."
        ) from error
    except ValueError as error:
        code = "unknown_outfit" if "Unknown outfit" in str(error) else "invalid_package"
        raise CardError(
            str(error),
            code,
            "Name one of the card's outfits, or 'none'."
            if code == "unknown_outfit"
            else "Check the pieces of the card ('opensculptboy wearables list').",
        ) from error
    shown = [p for p in worn if p.shown]
    if not shown:
        raise CardError(
            "The card shows no pieces in this outfit.",
            "no_pieces",
            "Choose an outfit with --outfit, or add pieces with 'wearables add'.",
        )
    private = any(p.wearable.provenance == PRIVATE for p in worn)
    if sheet is not None and private:
        check_destination(sheet, PRIVATE)
    tests = test_bodies(character, bodies)
    model = character.build_model(dtype=torch.float64)
    for test in tests:  # unknown sliders fail before the work starts
        unknown = sorted(set(test.character.phenotype) - set(model.phenotype_labels))
        if unknown:
            raise CheckError(
                f"Unknown phenotype sliders {unknown}.",
                "bad_option",
                f"Use sliders of {list(model.phenotype_labels)}.",
            )
    from opensculptboy.wearables.catalogue import quality_summary

    pieces_report = []
    for p in worn:
        w = p.wearable
        pieces_report.append(
            {
                "ref": p.ref,
                "label": w.label or w.name,
                "slot": w.slot,
                "layer": int(w.layer),
                "shown": bool(p.shown),
                "hidden_groups": list(p.hidden_groups),
                "provenance": w.provenance,
                "vertices": int(len(w.garment.vertices)),
                "triangles": int(len(w.garment.triangles)),
                "quality": quality_summary(w),
            }
        )
    notes, states, views = [], [], []
    for test in tests:
        log(f"body {test.name}")
        body = card_body(test.character, model, detail_dir if test.detail else None)
        dressed = dress(body, worn)
        lay = layout(body, dressed)
        rest_vertices = [p.vertices for p in lay.pieces]
        body_states = []

        def record(state, frame, body_v, piece_v, extra=None):
            result, marks = measure(lay, body_v, piece_v, tolerance)
            entry = {"body": test.name, "state": state, "frame": frame, **result}
            entry.update(extra or {})
            states.append(entry)
            body_states.append((entry, body_v, piece_v, marks))

        record("rest", 0, body.vertices, rest_vertices)
        if tpose:
            transforms, (bw, bi), piece_w, note = _tpose(model, body, lay.pieces)
            if note and note not in notes:
                notes.append(note)
            record(
                "tpose",
                0,
                skin(body.vertices, bw, bi, transforms),
                [
                    skin(p.vertices, w, j, transforms)
                    for p, (w, j) in zip(lay.pieces, piece_w)
                ],
            )
        for state, frame, transforms in pose_states(
            model, test.character, clips, frames
        ):
            record(
                state,
                frame,
                skin(body.vertices, body.weights, body.indices, transforms),
                [skin(p.vertices, p.weights, p.joints, transforms) for p in lay.pieces],
            )
        rest = body_states[0]
        others = body_states[1:] or body_states
        worst = max(
            others,
            key=lambda s: (s[0]["skin"]["showing"] + s[0]["layers"]["cutting"]),
        )
        views.append((test.name, lay, body, rest, worst))
    rest_states = [s for s in states if s["state"] == "rest"]
    tpose_states = [s for s in states if s["state"] == "tpose"]
    clip_states = [s for s in states if s["state"] not in ("rest", "tpose")]
    errors = sum(p["quality"]["errors"] for p in pieces_report)
    gates = [
        {
            "name": "quality",
            "pass": errors == 0,
            "limit": 0,
            "unit": "error items",
            "value": errors,
            "worst": {
                "piece": next(
                    (p["ref"] for p in pieces_report if p["quality"]["errors"]), None
                ),
                "codes": sorted(
                    {c for p in pieces_report for c in p["quality"]["codes"]["error"]}
                ),
            },
        },
        _gate("skin_rest", rest_states, "skin", 0, "vertices"),
        _gate("layers_rest", rest_states, "layers", 0, "vertices"),
        _gate("skin_tpose", tpose_states, "skin", pose_limit, "percent"),
        _gate("layers_tpose", tpose_states, "layers", pose_limit, "percent"),
        _gate("skin_clips", clip_states, "skin", pose_limit, "percent"),
        _gate("layers_clips", clip_states, "layers", pose_limit, "percent"),
    ]
    passed = all(g["pass"] for g in gates)
    report = {
        "format": FORMAT,
        "card": str(path),
        "outfit": outfit if outfit is not None else (character.active_outfit or None),
        "pass": passed,
        "failed": [g["name"] for g in gates if not g["pass"]],
        "gates": gates,
        "tolerance_mm": round(tolerance * 1000.0, 3),
        "pose_limit_percent": pose_limit,
        "bodies": [
            {
                "name": t.name,
                "phenotype": dict(t.character.phenotype),
                "detail": t.detail,
            }
            for t in tests
        ],
        "clips": clips,
        "frames": frames,
        "pieces": pieces_report,
        "states": states,
        "notes": notes,
        "sheet": None,
    }
    if sheet is not None:
        log("review sheet")
        sheet = pathlib.Path(sheet)
        sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.write_bytes(review_sheet(report, views, character))
        report["sheet"] = str(sheet)
    return report


def _meshes(lay, body, body_vertices, piece_vertices):
    from opensculptboy.render.dressed import SKIN, Mesh
    from opensculptboy.wearables.catalogue import triangle_colours

    meshes = [Mesh(body_vertices, body.triangles, SKIN)]
    for piece, vertices in zip(lay.pieces, piece_vertices):
        garment = piece.garment
        keep = np.ones(len(garment.triangles), bool)
        for g, group in enumerate(garment.groups):
            if group in piece.hidden_groups:
                keep &= garment.triangle_group != g
        meshes.append(
            Mesh(vertices, garment.triangles[keep], triangle_colours(garment)[keep])
        )
    return meshes


def _summary(entry) -> str:
    skin, layers = entry["skin"], entry["layers"]
    return (
        f"skin {skin['showing']} ({skin['share_percent']:.1f}%), "
        f"layers {layers['cutting']} ({layers['share_percent']:.1f}%)"
    )


def _state_name(entry) -> str:
    if entry["state"] in ("rest", "tpose", "pose"):
        return entry["state"]
    return f"{entry['state']} frame {entry['frame']}"


def _sample(points: np.ndarray, most: int = 300) -> np.ndarray:
    """At most ``most`` of the points, evenly spread over their order, so that the marks of
    a large fault leave the cloth under them in sight."""
    if len(points) <= most:
        return points
    return points[np.linspace(0, len(points) - 1, most).astype(int)]


def review_sheet(report, views, character, tile=(200, 320)) -> bytes:
    """The review sheet (PNG): one row per test body with its rest and its worst state from
    the front, the side and the back, each tile labelled with the body, the state, the view
    and its counts; red dots mark skin through, orange dots layers through."""
    from opensculptboy.render.dressed import VIEWS, Marker, draw, grid, labelled, png

    rows = []
    for name, lay, body, rest, worst in views:
        row = []
        for entry, body_v, piece_v, marks in (rest, worst):
            meshes = _meshes(lay, body, body_v, piece_v)
            markers = [
                Marker(_sample(marks["skin"]), SKIN_MARK, 2.0),
                Marker(_sample(marks["layers"]), LAYER_MARK, 2.0),
                Marker(_sample(marks["as_made"]), AS_MADE_MARK, 1.5),
            ]
            for view in ("front", "side", "back"):
                image = draw(meshes, VIEWS[view], size=tile, markers=markers)
                tag = "" if entry is rest[0] else "worst: "
                row.append(
                    labelled(
                        image,
                        [
                            f"{name} - {tag}{_state_name(entry)} - {view}",
                            _summary(entry),
                        ],
                        colour=(160, 20, 20)
                        if entry["skin"]["showing"] or entry["layers"]["cutting"]
                        else (30, 110, 40),
                    )
                )
        rows.append(row)
    gates = ", ".join(
        f"{g['name']} {'pass' if g['pass'] else 'FAIL'}" for g in report["gates"]
    )
    title = [
        f"wearables check: {character.name}, outfit {report['outfit'] or 'none'} - "
        f"{'PASS' if report['pass'] else 'FAIL'}",
        gates,
        f"red: skin more than {report['tolerance_mm']:g} mm outside its piece; orange: a "
        f"lower piece more than {report['tolerance_mm']:g} mm outside an upper piece; "
        "violet: a lower piece outside an upper piece as the two were made (at most 300 "
        "dots of each)",
        "each row: rest (front, side, back), then the body's worst state (front, side, "
        "back)",
    ]
    return png(grid(rows, title))

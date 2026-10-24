# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the blinks of the exported eyes (``opensculptboy.eyes.blink``, docs/eyes.md
"Blinks"): anny's own blink crosses its lids and cuts into the eye; the synthetic lid rig
turns the lids about the eye until the rims meet, keeps them outside the eye along the path
with the in-between (GLB) and along the straight line of the morph target (VRM), and turns the
lashes with their lid by a share of its turn. The files are measured as an engine plays them,
at blink 0, 25, 50, 75 and 100 %, for the left eye, the right eye and both, in GLB and in VRM
1.0 and 0.x. A reference's blink, carried from a synthetic rig, is the blink of a card that
has one, its lashes included.
"""

import functools
import json
import pathlib
import tempfile
import unittest

import numpy as np
import torch

from opensculptboy import Character, export_glb
from opensculptboy.eyes import blink as B
from opensculptboy.eyes import geometry
from opensculptboy.eyes.lashes import LOWER, UPPER
from opensculptboy.eyes.mesh import eyed_mesh
from opensculptboy.eyes.poses import (
    EyePoses,
    card_blink_poses,
    card_eye_poses,
    eye_poses,
)
from opensculptboy.material import EyeMaterial, LashMaterial, check_material
from test.gltf_reader import GLB, evaluate_nodes, structure_errors

STAGES = B.STAGES
# the closed upper lid passes the lower rim by at most this much (mm), and at most this many
# pairs of triangles of the two lids cut through each other under it: a game character's
# upper lid closes over its lower lid (2.0 to 2.5 mm at the reference's joints, with 13 to 41
# such pairs), and the turn that brings the rims together fades out toward the corners
CROSSING_MM = 2.5
CLIPPING_PAIRS = 20
# the files' tolerances (mm): float32 positions and the welds of the files' split vertices
EYE_MM = 0.02
# a closed eye's margins meet: the widest opening left between them (mm), toward the
# corners where the turn that brings the rims together fades out
CLOSED_OPENING_MM = 0.5
# lash points may lie this deep (mm) inside the lids in a blink, as the roots of the lashes
# that a closing lid covers (a reference's own lashes reach 0.95 mm)
LASH_DEPTH_MM = 1.5
# a lash root that slides over the eye on the way may dip this far (mm) into the cornea's
# shell, out of sight under the rim of the lid
LASH_EYE_MM = 0.2
CASES = {"left": ("L",), "right": ("R",), "both": ("L", "R")}

_TMP = tempfile.TemporaryDirectory()
CARD = Character(name="blinks", material={"eyes": {}})


# ------------------------------------------------------------------- measuring a file
# the file's frame to Anny's: GLB and VRM 1.0 face +Z with Y up; VRM 0.x faces -Z
TO_ANNY = {
    "gltf": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "vrm0": np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
}


class FileEyes:
    """The eyes of an exported file as an engine plays it: per eye, the lids on the file's
    welded skin, the outer surface of the eye and the lashes, from the vertices that
    ``evaluate_nodes`` gives for a set of morph weights."""

    def __init__(self, path, frame: str = "gltf"):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        self.glb = GLB(path)
        g = self.glb.json
        self.C = TO_ANNY[frame]
        self.node = next(
            i
            for i, n in enumerate(g["nodes"])
            if "mesh" in n
            and "skin" in n
            and any(
                g["materials"][p["material"]]["name"] == "lashes"
                for p in g["meshes"][n["mesh"]]["primitives"]
            )
        )
        mesh = g["meshes"][g["nodes"][self.node]["mesh"]]
        self.names = list(mesh.get("extras", {}).get("targetNames", []))
        prims = mesh["primitives"]
        material = [g["materials"][p["material"]]["name"] for p in prims]
        tris = {
            k: self.glb.accessor(p["indices"]).reshape(-1, 3)
            for k, p in enumerate(prims)
        }
        rest = self.positions({})
        eye_names = {"eyes", "cornea", "lashes"}
        skin = np.concatenate(
            [tris[k] for k, m in enumerate(material) if m not in eye_names]
        )
        # weld the skin's split vertices; leave out the parts apart from the body (caruncles)
        welded, wt, inverse = B.weld(rest, skin, 1e-6)
        n = len(welded)
        e = np.concatenate([wt[:, [0, 1]], wt[:, [1, 2]]])
        graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        _, label = connected_components(graph, directed=False)
        body = np.bincount(label[wt[:, 0]]).argmax()
        wt = wt[label[wt[:, 0]] == body]
        self.inverse, self.rest, self.triangles = inverse, welded, wt
        self.first = np.zeros(n, np.int64)
        self.first[inverse[::-1]] = np.arange(len(inverse))[::-1]
        used = np.zeros(n, bool)
        used[wt] = True
        self.eyes = {}
        lash_k = [k for k, m in enumerate(material) if m == "lashes"]
        lash_t = np.concatenate([tris[k] for k in lash_k])
        n_lash = int(lash_t.max()) + 1
        e = np.concatenate([lash_t[:, [0, 1]], lash_t[:, [1, 2]]])
        graph = coo_matrix(
            (np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_lash, n_lash)
        )
        _, lash_label = connected_components(graph, directed=False)

        def of_side(t, side):  # the triangles of one eye, by their centres
            return t[(rest[t].mean(1)[:, 0] > 0) == (side == "L")]

        for side in ("L", "R"):
            eye_t = of_side(
                np.concatenate(
                    [tris[k] for k, m in enumerate(material) if m == "eyes"]
                ),
                side,
            )
            frame = geometry.eye_frame(side, rest[np.unique(eye_t)])
            outer = [tris[k] for k, m in enumerate(material) if m == "cornea"]
            side_t = of_side(np.concatenate(outer), side) if outer else eye_t
            sv, st, _ = B.weld(rest, side_t, 1e-6)
            surface = B.EyeSurface(sv, st)
            if surface.depth(frame.centre[None])[0] < 0:
                surface = B.EyeSurface(sv, st[:, ::-1])
            lids = B.eye_lids(frame, welded, wt, used)
            lash = np.unique(of_side(lash_t, side))
            # each strip or card of the lashes belongs to the lid whose margin it nears most
            upper = np.zeros(len(lash), bool)
            for label in np.unique(lash_label[lash]):
                mine = lash_label[lash] == label
                pts = rest[lash[mine]]
                du = np.linalg.norm(
                    pts[:, None] - welded[lids.upper][None], axis=2
                ).min()
                dl = np.linalg.norm(
                    pts[:, None] - welded[lids.lower][None], axis=2
                ).min()
                upper[mine] = du < dl
            self.eyes[side] = (lids, surface, lash, upper)

    def crack_mm(self, weights: dict) -> float:
        """The widest split (mm) between vertices that share a place at rest: a crack in
        the skin."""
        x = self.positions(weights)
        mean = np.zeros((len(self.rest), 3))
        np.add.at(mean, self.inverse, x)
        mean /= np.maximum(np.bincount(self.inverse, minlength=len(self.rest)), 1)[
            :, None
        ]
        return float(np.linalg.norm(x - mean[self.inverse], axis=1).max() * 1000.0)

    def positions(self, weights: dict) -> np.ndarray:
        """The vertices (V, 3) in Anny's frame for morph weights by target name."""
        w = [float(weights.get(n, 0.0)) for n in self.names]
        ev = evaluate_nodes(self.glb, weights={self.node: w})[self.node]
        return np.asarray(ev[0].positions, np.float64) @ self.C.T

    def measure(self, weights: dict, side: str, closed: bool = False) -> dict:
        x = self.positions(weights)
        rest_x = self.positions({})
        lids, surface, lash, upper = self.eyes[side]
        skin = x[self.first]
        cross, opening = B.lid_crossing(lids, skin)
        depth, _ = B.eye_penetration(lids, skin, self.rest, surface)
        lash_depth, _ = B.lash_penetration(lids, skin, self.rest, x[lash], rest_x[lash])
        out = {
            "crossing_mm": cross,
            "opening_mm": opening,
            "clipping_pairs": B.lid_clipping(lids, skin),
            "folded_triangles": B.folded_triangles(lids, skin, self.rest, surface),
            "eye_mm": depth,
            "lash_mm": lash_depth,
        }
        if closed:
            out["lash_order_mm"] = B.lash_order(
                lids.frame, x[lash][upper], x[lash][~upper]
            )[0]
        return out


@functools.lru_cache(maxsize=None)
def card_export(kind: str):
    """An export of :data:`CARD`: ``"glb"``, ``"vrm1"`` or ``"vrm0"``."""
    folder = pathlib.Path(_TMP.name)
    if kind == "glb":
        path = folder / "blinks.glb"
        export_glb(path, CARD)
        return path, "gltf"
    from opensculptboy.export.vrm import export_vrm

    path = folder / f"blinks_{kind}.vrm"
    export_vrm(
        path,
        CARD,
        author="OpenSculptBoy tests",
        version="0.x" if kind == "vrm0" else "1.0",
    )
    return path, "vrm0" if kind == "vrm0" else "gltf"


@functools.lru_cache(maxsize=None)
def file_eyes(kind: str) -> FileEyes:
    path, frame = card_export(kind)
    return FileEyes(path, frame)


def stage_weights(sides, t: float, inbetween: bool) -> dict:
    out = {}
    for side in sides:
        name = B.BLINK_TARGETS[side]
        out[name] = t
        if inbetween:
            out[name + B.INBETWEEN_SUFFIX] = 1.0 - abs(2.0 * t - 1.0)
    return out


# ------------------------------------------------------------------- the eyed mesh
@functools.lru_cache(maxsize=None)
def anny_blinks():
    """Anny's default character with the eyes and generic lashes: the rest mesh, the eyed
    mesh, and anny's own blink target of each eye."""
    model = Character().build_model(dtype=torch.float64)
    labels = list(model.facial_action_labels)
    actions = torch.zeros(3, len(labels), dtype=torch.float64)
    for k, side in enumerate(("L", "R")):
        actions[k + 1, labels.index(B.BLINK_TARGETS[side])] = 1.0
    with torch.no_grad():
        out = model(facial_actions=actions)
    rest = out["rest_vertices"][0].numpy()
    targets = {
        side: out["rest_vertices"][k + 1].numpy() - rest
        for k, side in enumerate(("L", "R"))
    }
    eyed = eyed_mesh(
        model, None, rest, EyeMaterial(), "gltf", lashes=(LashMaterial(), None)
    )
    return rest, eyed, targets


class TestAnnyBlink(unittest.TestCase):
    def test_anny_crosses_and_cuts_the_eye(self):
        # the problem: anny's own blink closes the upper lid past the lower one and cuts
        # into the eye on the way
        rest, eyed, targets = anny_blinks()
        lids, surfaces = B.eyed_lids(eyed, rest), B.eye_surfaces(eyed)
        for side in ("L", "R"):
            with self.subTest(side=side):
                m = B.measure_blink(
                    lids[side], rest, rest + targets[side], surfaces[side]
                )
                self.assertGreater(m["1"]["crossing_mm"], 2.0)
                self.assertGreater(max(m[t]["eye_mm"] for t in m), 0.2)

    def test_rig_blink(self):
        rest, eyed, targets = anny_blinks()
        offsets = np.stack([targets["L"], targets["R"]])
        names = [B.BLINK_TARGETS["L"], B.BLINK_TARGETS["R"]]
        for inbetween in (True, False):
            shaped, blinks = B.blink_offsets(
                eyed, rest, offsets, names, None, inbetween
            )
            self.assertEqual(sorted(blinks.rows), ["L", "R"])
            self.assertEqual(shaped.shape[0], 4 if inbetween else 2)
            for side in ("L", "R"):
                shape = blinks.shapes[side]
                self.assertEqual(blinks.extras[B.BLINK_TARGETS[side]]["source"], "rig")
                lids, surface = blinks.lids[side], blinks.surfaces[side]
                # GLB files play the path with the in-between, VRM files the straight line
                half = shape.inbetween if inbetween else None
                m = B.measure_blink(lids, rest, rest + shape.closed, surface, half=half)
                with self.subTest(side=side, inbetween=inbetween):
                    for t, row in m.items():
                        self.assertLessEqual(row["crossing_mm"], CROSSING_MM, t)
                        self.assertLessEqual(row["clipping_pairs"], CLIPPING_PAIRS, t)
                        self.assertEqual(row["folded_triangles"], 0, t)
                        self.assertLessEqual(row["eye_mm"], 0.0, t)
                    self.assertLessEqual(m["1"]["opening_mm"], CLOSED_OPENING_MM)
                    # the upper lid travels most of the way
                    _, phi0, _ = B.cylinder(lids.frame, rest[lids.upper])
                    _, phi1, _ = B.cylinder(
                        lids.frame, rest[lids.upper] + shape.closed[lids.upper]
                    )
                    _, low0, _ = B.cylinder(lids.frame, rest[lids.lower])
                    _, low1, _ = B.cylinder(
                        lids.frame, rest[lids.lower] + shape.closed[lids.lower]
                    )
                    self.assertGreater(
                        np.median(phi0 - phi1), 3.0 * np.median(np.abs(low1 - low0))
                    )
                # the rows away from the lids keep anny's target
                far = np.ones(len(rest), bool)
                far[blinks.lids[side].region] = False
                np.testing.assert_array_equal(shape.closed[far], targets[side][far])
                if not inbetween:
                    continue
                # the in-between moves the half-way point out, around the eye
                mid = rest + 0.5 * shape.closed
                centre = lids.frame.centre
                moved = np.linalg.norm(shape.inbetween, axis=1) > 1e-5
                self.assertGreater(moved.sum(), 50)
                r0 = np.linalg.norm(mid[moved] - centre, axis=1)
                r1 = np.linalg.norm(
                    mid[moved] + shape.inbetween[moved] - centre, axis=1
                )
                self.assertGreater(np.mean(r1 - r0), 0.0)

    def test_steady_folds(self):
        # a sliver folded over at rest, squeezed by a blink, turns its corner's normal over
        x = np.array(
            [[0, 0, 0], [2, 0, 0], [1, 2, 0], [1, -2, 0], [1, 0.05, -0.02]], float
        )
        tris = np.array([[0, 1, 2], [0, 3, 1], [0, 4, 1]])
        self.assertEqual(B.folded_at_rest(x, tris).tolist(), [False, False, True])
        move = np.zeros_like(x)
        move[4] = [0.0, -0.04, 0.0]
        move[0] = [0.0, 0.0, 0.3]
        steady = B.steady_folds(x, tris, move, np.arange(5))
        # the sliver's corners move as one piece: its shape stays as at rest
        for k in (0, 1, 4):
            np.testing.assert_allclose(steady[k], move[[0, 1, 4]].mean(0))
        np.testing.assert_array_equal(steady[[2, 3]], move[[2, 3]])

    def test_lashes(self):
        rest, eyed, targets = anny_blinks()
        offsets = np.stack([targets["L"], targets["R"]])
        names = [B.BLINK_TARGETS["L"], B.BLINK_TARGETS["R"]]
        shaped, blinks = B.blink_offsets(eyed, rest, offsets, names)
        full = B.blink_lash_offsets(eyed, blinks, rest, eyed.targets(shaped))
        self.assertEqual(full.shape, (4, eyed.vertex_count, 3))
        for side in ("L", "R"):
            parts = B.lash_rows(eyed, side)
            self.assertEqual({p.lid for p, _ in parts}, {UPPER, LOWER})
            m = blinks.extras[B.BLINK_TARGETS[side]]["measures"]
            for t, row in m.items():
                with self.subTest(side=side, stage=t):
                    self.assertLessEqual(row["lash_mm"], LASH_DEPTH_MM)
            # each lash turns about the eye's horizontal axis by a share of its lid's turn,
            # and its root goes with the lid: the closed upper lashes keep pointing out of
            # the eye, as the reference's do
            shape = blinks.shapes[side]
            frame = blinks.lids[side].frame
            surface = blinks.surfaces[side]
            for p, rows in parts:
                closed = p.rest + full[blinks.rows[side], rows]
                tips = p.uvs[:, 1] > 0.9
                roots = p.uvs[:, 1] < 0.1
                v0 = p.rest[tips].mean(0) - p.rest[roots].mean(0)
                v1 = closed[tips].mean(0) - closed[roots].mean(0)
                q0, q1 = v0 @ frame.axes, v1 @ frame.axes
                turned = np.arctan2(q1[1], q1[2]) - np.arctan2(q0[1], q0[2])
                lid_turn = np.mean(p.follow.root @ shape.turn[p.follow.rows])
                with self.subTest(side=side, lid=p.lid):
                    self.assertAlmostEqual(
                        np.degrees(turned),
                        np.degrees(B.LASH_TURN * lid_turn),
                        delta=2.0,
                    )
                    # no lash point goes into the eye deeper than at rest (a root on the
                    # lid margin may touch it) at any stage of the path
                    floor = np.maximum(surface.depth(p.rest), 0.0) + 2e-5
                    for t in STAGES:
                        tent = 1.0 - abs(2.0 * t - 1.0)
                        x = p.rest + t * full[blinks.rows[side], rows]
                        x = x + tent * full[blinks.inbetween_rows[side], rows]
                        over = float((surface.depth(x) - floor).max()) * 1000.0
                        self.assertLessEqual(over, LASH_EYE_MM, (t, over))
            upper = [p for p, _ in parts if p.lid == UPPER][0]
            self.assertLess(
                np.degrees(np.mean(upper.follow.root @ shape.turn[upper.follow.rows])),
                -20.0,
            )


# ------------------------------------------------------------------------- the files
class TestBlinkFiles(unittest.TestCase):
    def _check(self, kind, inbetween):
        eyes = file_eyes(kind)
        for case, sides in CASES.items():
            for t in STAGES:
                weights = stage_weights(sides, t, inbetween)
                for side in sides:
                    m = eyes.measure(weights, side, closed=t == 1.0)
                    with self.subTest(
                        kind=kind, inbetween=inbetween, case=case, side=side, t=t
                    ):
                        self.assertLessEqual(m["crossing_mm"], CROSSING_MM)
                        self.assertLessEqual(m["clipping_pairs"], CLIPPING_PAIRS)
                        self.assertEqual(m["folded_triangles"], 0)
                        if inbetween or kind != "glb":
                            # a GLB target alone cuts the corner of the path around the eye
                            self.assertLessEqual(m["eye_mm"], EYE_MM)
                        self.assertLessEqual(m["lash_mm"], LASH_DEPTH_MM)
                        if t == 1.0:
                            self.assertLessEqual(m["opening_mm"], CLOSED_OPENING_MM)
            # the other eye stays open in a blink of one eye
            if len(sides) == 1:
                other = "R" if sides[0] == "L" else "L"
                m = eyes.measure(stage_weights(sides, 1.0, inbetween), other)
                self.assertGreater(m["opening_mm"], 5.0)

    def test_glb_straight_line(self):
        self._check("glb", False)

    def test_glb_with_inbetween(self):
        self._check("glb", True)

    def test_vrm1(self):
        self._check("vrm1", False)

    def test_vrm0(self):
        self._check("vrm0", False)

    def test_glb_targets_and_animations(self):
        path, _ = card_export("glb")
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        g = glb.json
        extras = g["scenes"][0]["extras"]["opensculptboy"]
        names = extras["morph_targets"]
        for side in ("L", "R"):
            self.assertIn(B.BLINK_TARGETS[side] + B.INBETWEEN_SUFFIX, names)
            blink = extras["eyes"]["blinks"][B.BLINK_TARGETS[side]]
            self.assertEqual(blink["source"], "rig")
            self.assertEqual(
                sorted(blink["measures"]), sorted(f"{t:g}" for t in STAGES)
            )
        animations = {a["name"]: a for a in g["animations"]}
        for name in ("eyeBlinkLeft.close", "eyeBlinkRight.close", "eyeBlink.close"):
            self.assertIn(name, animations)
        # half way through, the in-between is fully on and the blink target half
        a = animations["eyeBlink.close"]
        times = glb.accessor(a["samplers"][0]["input"]).ravel()
        weights = glb.accessor(a["samplers"][0]["output"]).reshape(len(times), -1)
        k = int(np.argmin(np.abs(times - 0.5)))
        self.assertAlmostEqual(weights[k, names.index("eyeBlinkLeft")], 0.5)
        self.assertAlmostEqual(weights[k, names.index("eyeBlinkLeft.inbetween")], 1.0)
        self.assertAlmostEqual(weights[-1, names.index("eyeBlinkRight.inbetween")], 0.0)

    def test_vrm_has_no_inbetween(self):
        for kind in ("vrm1", "vrm0"):
            with self.subTest(kind=kind):
                eyes = file_eyes(kind)
                self.assertFalse(
                    [n for n in eyes.names if n.endswith(B.INBETWEEN_SUFFIX)]
                )

    def test_cards_without_eyes_keep_anny_blink(self):
        # the blink shaping comes with the eyes; other cards export as before
        path = pathlib.Path(_TMP.name) / "plain.glb"
        export_glb(path, Character(name="plain"))
        g = GLB(path).json
        names = g["scenes"][0]["extras"]["opensculptboy"]["morph_targets"]
        self.assertFalse([n for n in names if n.endswith(B.INBETWEEN_SUFFIX)])
        self.assertFalse(
            [a for a in g.get("animations", []) if a["name"].endswith(".close")]
        )


class TestBlinkDetailLevels(unittest.TestCase):
    """Level 1 of a card's detail layer on anny's mesh and on the sculpt mesh: the finer
    lids hold more than one row of margin vertices and more triangles to fold."""

    def test_level1(self):
        from opensculptboy.detail import level, save_offsets

        for topology in ("anny", "sculpt"):
            with self.subTest(topology=topology), tempfile.TemporaryDirectory() as tmp:
                folder = pathlib.Path(tmp)
                levels = []
                for subdiv in (0, 1):
                    name = f"card.detail{subdiv}.safetensors"
                    count = level(topology, subdiv).vertex_count
                    save_offsets(folder / name, np.zeros((count, 3)), subdiv, topology)
                    levels.append({"subdiv": subdiv, "offsets": name})
                card = Character(
                    name="level1",
                    topology=topology,
                    material={"eyes": {}},
                    detail={"levels": levels},
                )
                path = folder / "level1.glb"
                export_glb(path, card, detail_level=1, detail_dir=folder)
                eyes = FileEyes(path)
                for inbetween in (False, True):
                    for t in STAGES:
                        weights = stage_weights(("L", "R"), t, inbetween)
                        for side in ("L", "R"):
                            m = eyes.measure(weights, side, closed=t == 1.0)
                            label = f"{side} {t} {inbetween}"
                            self.assertLessEqual(m["crossing_mm"], CROSSING_MM, label)
                            self.assertLessEqual(
                                m["clipping_pairs"], CLIPPING_PAIRS, label
                            )
                            self.assertEqual(m["folded_triangles"], 0, label)
                            if inbetween:
                                self.assertLessEqual(m["eye_mm"], EYE_MM, label)
                            self.assertLessEqual(m["lash_mm"], LASH_DEPTH_MM, label)
                            if t == 1.0:
                                self.assertLessEqual(
                                    m["opening_mm"], CLOSED_OPENING_MM, label
                                )


# ------------------------------------------------------------------------ a reference
def _reference_rig(folder: pathlib.Path, rest, triangles, lid_weight, pivot):
    """A synthetic reference: anny's own skin as a triangle soup in glTF's frame, skinned to
    a root joint and a lid joint at the left eye's centre with ``lid_weight``."""
    from opensculptboy.fit.handoff import GLTF_TO_ANNY
    from test.test_mouth_lips import _write_rig_glb

    C = GLTF_TO_ANNY
    soup = (rest @ C)[triangles.ravel()]  # Anny's frame to glTF's: v C (C orthogonal)
    J = np.zeros((len(soup), 4), int)
    J[:, 1] = 1
    W = np.zeros((len(soup), 4))
    W[:, 1] = lid_weight[triangles.ravel()]
    W[:, 0] = 1.0 - W[:, 1]
    _write_rig_glb(folder / "replica.glb", soup, J, W, (pivot @ C).tolist())


class TestReferenceClosedEye(unittest.TestCase):
    def test_carry(self):
        from opensculptboy.fit.handoff import load_handoff
        from opensculptboy.fit.reference import Reference

        rest, eyed, targets = anny_blinks()
        triangles = eyed.triangles[: int(eyed.kept.sum())]
        frame = eyed.frames["L"]
        lids = B.eyed_lids(eyed, rest)["L"]
        # the upper lid of the left eye turns down about the eye's horizontal axis
        # a smooth share of the turn: in full on the upper lid, none below the eye's centre
        local = frame.to_local(rest)
        weight = np.clip((local[:, 1] + 0.1) / 0.3, 0.0, 1.0)
        weight *= np.clip((2.2 - np.linalg.norm(local, axis=1)) / 0.6, 0.0, 1.0)
        angle = np.radians(25.0)
        from opensculptboy.fit.handoff import GLTF_TO_ANNY

        axis = frame.axes[:, 0]
        q = [*(np.sin(angle / 2) * (axis @ GLTF_TO_ANNY)), np.cos(angle / 2)]
        q_half = [*(np.sin(angle / 4) * (axis @ GLTF_TO_ANNY)), np.cos(angle / 4)]
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            _reference_rig(folder, rest, triangles, weight, frame.centre)
            appearance = {
                "parts": [{"name": "head", "role": "skin"}],
                "eyes": {
                    "format": "eyes/1",
                    "poses": {
                        "open": {"known": True, "joints": {}},
                        "close_l": {
                            "known": True,
                            "joints": {"lower_jaw": {"rotation": q}},
                        },
                        "close_r": {
                            "known": False,
                            "joints": {"lower_jaw": {"rotation": q}},
                        },
                        "half_l": {
                            "known": True,
                            "joints": {"lower_jaw": {"rotation": q_half}},
                        },
                    },
                },
            }
            (folder / "appearance.json").write_text(json.dumps(appearance))
            handoff = load_handoff(folder)
            reference = Reference.from_handoff(handoff)
            from opensculptboy.eyes.mesh import eyeball_rows

            model = Character().build_model(dtype=torch.float64)
            rows = eyeball_rows(model)
            head = np.flatnonzero(np.linalg.norm(rest - frame.centre, axis=1) > 0.05)
            poses = eye_poses(handoff, reference, rest, rest, head, rows, triangles)
        # close_r is a guess: only close_l carries, with its half-closed pose
        self.assertEqual(poses.names, ("close_l", "half_l"))
        self.assertIsNone(poses.lashes)  # the synthetic reference has no lashes
        closed = poses.closed()
        self.assertEqual(list(closed), ["L"])
        # the upper lid's rows move as the reference's do: turned about the eye's axis
        u = lids.upper_rows
        local = rest[u] - frame.centre
        near = np.linalg.norm(local, axis=1) < 1.6 * frame.radius
        for k, a in ((0, angle), (1, 0.5 * angle)):
            R = _rotation(axis, a)
            expected = weight[u, None] * (local @ R.T - local)
            err = np.linalg.norm(poses.offsets[k][u][near] - expected[near], axis=1)
            # within 15 % of the motion: the carry smooths it a little
            size = np.linalg.norm(expected[near], axis=1)
            self.assertLess(np.median(err), 0.15 * np.median(size))
        # nothing moves away from the eye, nor on the eyeballs
        far = np.linalg.norm(rest - frame.centre, axis=1) > 2.7 * frame.radius
        self.assertLess(np.abs(closed["L"][far]).max(), 1e-12)
        self.assertLess(np.abs(closed["L"][rows["L"]]).max(), 1e-12)
        # the file of the poses, and a card that names it
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "c.eye_poses.safetensors"
            poses.lashes = np.zeros((2, 5, 3))
            poses.lashes[0, 2] = [0.001, 0.0, 0.0]
            poses.save(path)
            back = EyePoses.load(path)
            self.assertEqual(back.names, poses.names)
            np.testing.assert_allclose(back.offsets, poses.offsets, atol=1e-6)
            np.testing.assert_allclose(back.lashes, poses.lashes, atol=1e-9)
            card = Character(
                material={"eyes": {}, "reference_eyes": {"poses": path.name}}
            )
            self.assertIsNotNone(card_eye_poses(card, tmp))
            got = card_blink_poses(card, None, tmp)
            self.assertEqual(list(got), ["L"])
            np.testing.assert_allclose(got["L"].closed, poses.offsets[0], atol=1e-6)
            np.testing.assert_allclose(got["L"].half, poses.offsets[1], atol=1e-6)
            np.testing.assert_allclose(got["L"].lash_closed, poses.lashes[0], atol=1e-9)
            poses.lashes = None
        # the blink of a card with a reference is the reference's blink, its upper lid
        # turned further down where it leaves the rims apart: the synthetic reference turns
        # its lid by 25 degrees only
        offsets = np.stack([targets["L"], targets["R"]])
        names = [B.BLINK_TARGETS["L"], B.BLINK_TARGETS["R"]]
        pose = got["L"]
        shaped, blinks = B.blink_offsets(eyed, rest, offsets, names, {"L": pose})
        self.assertEqual(blinks.extras["eyeBlinkLeft"]["source"], "reference")
        self.assertEqual(blinks.extras["eyeBlinkRight"]["source"], "rig")
        shape = blinks.shapes["L"]
        turned = shape.extra != 0.0
        self.assertTrue(turned.any())
        self.assertLess(shape.extra.max(), 1e-12)  # down only
        np.testing.assert_allclose(
            shape.closed[~turned], pose.closed[~turned], atol=2e-4
        )
        np.testing.assert_allclose(
            (shape.inbetween + 0.5 * shape.closed)[~turned],
            pose.half[~turned],
            atol=2e-4,
        )
        m = B.measure_blink(
            lids, rest, rest + shape.closed, blinks.surfaces["L"], half=shape.inbetween
        )
        for t, row in m.items():
            self.assertLessEqual(row["crossing_mm"], CROSSING_MM, t)
            self.assertLessEqual(row["eye_mm"], 0.0, t)
        # the turn closes most of the opening the reference leaves (its turn stays within
        # MAX_RIM_TURN)
        own = B.measure_blink(lids, rest, rest + pose.closed, blinks.surfaces["L"])
        self.assertLess(m["1"]["opening_mm"], 0.5 * own["1"]["opening_mm"])

    def test_reference_lashes(self):
        # the lashes of a reference take the reference's own lash poses, rows of the card's
        # lashes (here the generic lashes, stood in as a reference's)
        from opensculptboy.eyes.poses import BlinkPose
        from opensculptboy.material import LashMaterial

        rest, eyed, targets = anny_blinks()
        data = eyed.lashes
        model = Character().build_model(dtype=torch.float64)
        ref = eyed_mesh(
            model, None, rest, EyeMaterial(), "gltf", lashes=(LashMaterial(), data)
        )
        rng = np.random.default_rng(3)
        lash_closed = rng.normal(0.0, 0.001, (len(data.points), 3))
        lash_half = rng.normal(0.0, 0.001, (len(data.points), 3))
        pose = BlinkPose(targets["L"], 0.5 * targets["L"], lash_closed, lash_half)
        offsets = np.stack([targets["L"], targets["R"]])
        names = [B.BLINK_TARGETS["L"], B.BLINK_TARGETS["R"]]
        shaped, blinks = B.blink_offsets(ref, rest, offsets, names, {"L": pose})
        full = B.blink_lash_offsets(ref, blinks, rest, ref.targets(shaped))
        parts = B.lash_rows(ref, "L")
        self.assertTrue(parts)
        shape = blinks.shapes["L"]
        now = 0.5 * shape.closed + shape.inbetween
        for p, rows in parts:
            # the reference's own lash poses, their roots moved with the lid where the
            # rims were brought together
            r = p.follow.rows
            closed = (
                lash_closed[p.points] + p.follow.root @ (shape.closed - pose.closed)[r]
            )
            half = lash_half[p.points] + p.follow.root @ (now - pose.half)[r]
            np.testing.assert_allclose(full[blinks.rows["L"], rows], closed, atol=1e-12)
            np.testing.assert_allclose(
                full[blinks.inbetween_rows["L"], rows], half - 0.5 * closed, atol=1e-12
            )
        # the other eye has no reference: its lashes follow the rig
        for p, rows in B.lash_rows(ref, "R"):
            self.assertFalse(
                np.allclose(full[blinks.rows["R"], rows], lash_closed[p.points])
            )

    def test_card_section(self):
        check_material(
            {"eyes": {}, "reference_eyes": {"poses": "c.eye_poses.safetensors"}}
        )
        for bad in ({"poses": 3}, {"poses": ""}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_material({"reference_eyes": bad})
        self.assertIsNone(card_eye_poses(Character(material={"eyes": {}})))


def _rotation(axis, angle):
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


if __name__ == "__main__":
    unittest.main()

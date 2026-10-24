# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the blinks of the exported eyes (``opensculptboy.eyes.blink``, docs/eyes.md
"Blinks"): anny's own blink crosses its lids and cuts into the eye; the shaped blink seals the
lids along one line, keeps them outside the eye along the straight line of the morph target
and along the path with the in-between, and keeps the lashes outside the lids, the upper
lashes over the lower ones. The files are measured as an engine plays them, at blink 0, 25,
50, 75 and 100 %, for the left eye, the right eye and both, in GLB (with and without the
in-between) and in VRM 1.0 and 0.x. A reference's closed eye, carried from a synthetic rig,
starts the blink of a card that has one.
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
    card_closed_offsets,
    card_eye_poses,
    eye_poses,
)
from opensculptboy.material import EyeMaterial, LashMaterial, check_material
from test.gltf_reader import GLB, evaluate_nodes, structure_errors

STAGES = B.STAGES
# the files' tolerances (mm): float32 positions and the welds of the files' split vertices
CROSSING_MM = 0.05
EYE_MM = 0.02
LASH_MM = 0.02
ORDER_MM = 0.02
# a closed eye's margins meet: the widest opening left between them (mm)
CLOSED_OPENING_MM = 0.4
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
            "folded_triangles": B.folded_triangles(lids, skin, self.rest),
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

    def test_shaped_blink(self):
        rest, eyed, targets = anny_blinks()
        offsets = np.stack([targets["L"], targets["R"]])
        names = [B.BLINK_TARGETS["L"], B.BLINK_TARGETS["R"]]
        shaped, blinks = B.blink_offsets(eyed, rest, offsets, names)
        self.assertEqual(sorted(blinks.rows), ["L", "R"])
        self.assertEqual(shaped.shape[0], 4)  # one in-between each
        for side in ("L", "R"):
            shape = blinks.shapes[side]
            lids, surface = blinks.lids[side], blinks.surfaces[side]
            for half in (None, shape.inbetween):
                m = B.measure_blink(lids, rest, rest + shape.closed, surface, half=half)
                with self.subTest(side=side, inbetween=half is not None):
                    for t, row in m.items():
                        self.assertLessEqual(row["crossing_mm"], CROSSING_MM, t)
                        self.assertEqual(row["clipping_pairs"], 0, t)
                        self.assertEqual(row["folded_triangles"], 0, t)
                        self.assertLessEqual(row["eye_mm"], 0.0, t)
                    self.assertLessEqual(m["1"]["opening_mm"], CLOSED_OPENING_MM)
                    # the upper lid travels most of the way
                    s, phi0, _ = B.cylinder(lids.frame, rest[lids.upper])
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
            # the in-between moves the half-way point out, around the eye
            half = rest + 0.5 * shape.closed
            centre = blinks.lids[side].frame.centre
            moved = np.linalg.norm(shape.inbetween, axis=1) > 1e-5
            self.assertGreater(moved.sum(), 50)
            r0 = np.linalg.norm(half[moved] - centre, axis=1)
            r1 = np.linalg.norm(half[moved] + shape.inbetween[moved] - centre, axis=1)
            self.assertGreater(np.mean(r1 - r0), 0.0)

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
                    self.assertLessEqual(row["lash_mm"], 0.0)
            self.assertLessEqual(m["1"]["lash_order_mm"], ORDER_MM)
            # the closed upper lashes point down, out of the closed lid
            frame = blinks.lids[side].frame
            for p, rows in parts:
                if p.lid != UPPER:
                    continue
                closed = p.rest + full[blinks.rows[side], rows]
                tips = closed[p.uvs[:, 1] > 0.9]
                roots = closed[p.uvs[:, 1] < 0.1]
                self.assertLess(
                    frame.to_local(tips)[:, 1].mean(),
                    frame.to_local(roots)[:, 1].mean(),
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
                        self.assertEqual(m["clipping_pairs"], 0)
                        self.assertEqual(m["folded_triangles"], 0)
                        self.assertLessEqual(m["eye_mm"], EYE_MM)
                        self.assertLessEqual(m["lash_mm"], LASH_MM)
                        if t == 1.0:
                            self.assertLessEqual(m["opening_mm"], CLOSED_OPENING_MM)
                            self.assertLessEqual(m["lash_order_mm"], ORDER_MM)
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
            self.assertEqual(blink["source"], "target")
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
                            self.assertEqual(m["clipping_pairs"], 0, label)
                            self.assertEqual(m["folded_triangles"], 0, label)
                            self.assertLessEqual(m["eye_mm"], EYE_MM, label)
                            self.assertLessEqual(m["lash_mm"], LASH_MM, label)
                            if t == 1.0:
                                self.assertLessEqual(
                                    m["opening_mm"], CLOSED_OPENING_MM, label
                                )
                                self.assertLessEqual(
                                    m["lash_order_mm"], ORDER_MM, label
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
        # close_r is a guess: only close_l carries
        self.assertEqual(poses.names, ("close_l",))
        closed = poses.closed()
        self.assertEqual(list(closed), ["L"])
        # the upper lid's rows move as the reference's do: turned about the eye's axis
        u = lids.upper_rows
        local = rest[u] - frame.centre
        R = _rotation(axis, angle)
        expected = weight[u, None] * (local @ R.T - local)
        moved = closed["L"][u]
        near = np.linalg.norm(local, axis=1) < 1.6 * frame.radius
        err = np.linalg.norm(moved[near] - expected[near], axis=1)
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
            poses.save(path)
            back = EyePoses.load(path)
            self.assertEqual(back.names, poses.names)
            np.testing.assert_allclose(back.offsets, poses.offsets, atol=1e-6)
            card = Character(
                material={"eyes": {}, "reference_eyes": {"poses": path.name}}
            )
            self.assertIsNotNone(card_eye_poses(card, tmp))
            got = card_closed_offsets(card, None, tmp)
            np.testing.assert_allclose(got["L"], poses.offsets[0], atol=1e-6)
        # the blink of a card with a reference starts from the reference's closed eye
        offsets = np.stack([targets["L"], targets["R"]])
        names = [B.BLINK_TARGETS["L"], B.BLINK_TARGETS["R"]]
        shaped, blinks = B.blink_offsets(eyed, rest, offsets, names, closed)
        self.assertEqual(blinks.extras["eyeBlinkLeft"]["source"], "reference")
        # the rows away from the lids keep the reference's closed pose: no step at the edge
        # of the lids' region
        far = np.ones(len(rest), bool)
        far[blinks.lids["L"].region] = False
        np.testing.assert_array_equal(blinks.shapes["L"].closed[far], closed["L"][far])
        self.assertEqual(blinks.extras["eyeBlinkRight"]["source"], "target")
        m = B.measure_blink(
            lids, rest, rest + blinks.shapes["L"].closed, blinks.surfaces["L"]
        )
        for t, row in m.items():
            self.assertLessEqual(row["crossing_mm"], CROSSING_MM, t)
            self.assertLessEqual(row["eye_mm"], 0.0, t)
        self.assertLessEqual(m["1"]["opening_mm"], CLOSED_OPENING_MM)

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

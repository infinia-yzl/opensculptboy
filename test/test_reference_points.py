# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the reference points (``opensculptboy.posing.reference_points``), their solve
(``point_solve``) and its twin in the page (``viewer/src/pose_edit/solve.ts``), their placement
on a scan (``point_place``), Squeeze and the balance mark (``contact``) and a statue posed by its
points (``opensculptboy.statues.points``), with the default character and synthetic data.
"""

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from test.statue_fixtures import HEROIC, _model, synthetic_statue

REPO = pathlib.Path(__file__).resolve().parents[1]
NODE_SCRIPT = REPO / "viewer" / "test" / "pose_edit_parity.mjs"


def _node_available():
    if shutil.which("node") is None:
        return False
    out = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout
    return int(out.strip().lstrip("v").split(".")[0]) >= 22


def _rig(character=None, rotation=None, translation=None, pose=None):
    from opensculptboy.character import Character
    from opensculptboy.posing.pose_rig import PoseRig

    return PoseRig.from_card(
        _model(), character or Character(), rotation, translation, pose=pose
    )


def _library(name):
    import anny.poses

    return (
        anny.poses.pose_parameters(_model(), name)["pose_parameters"][0, :, :3, :3]
        .double()
        .numpy()
    )


def _random_pose(rig, R0, chains, seed=0, share=(0.1, 0.5)):
    """A pose of the chains' bones: each free joint coordinate a random share of its range's
    upper end."""
    from opensculptboy.posing import point_solve as ps

    frames = ps.joint_frames(rig)
    rng = np.random.default_rng(seed)
    R = R0.copy()
    for b in ps.chain_bones(rig, chains):
        f = frames[b]
        c = np.zeros(3)
        for k in range(3):
            if f["dofs"][k]:
                c[k] = rng.uniform(*share) * f["hi"][k]
        R[b] = ps._exp(f["axes"] @ (f["sign"] * c))
    return R


def _targets(rig, R, chains, off=()):
    from opensculptboy.posing.reference_points import (
        influences_of,
        points_of,
        positions,
    )

    use = points_of(chains)
    influences = influences_of(rig, use)
    places = positions(rig, R, use)
    return use, [
        {
            "influence": f,
            "target": t,
            "finger": p["finger"],
            "on": p["name"] not in off,
        }
        for f, t, p in zip(influences, places, use)
    ]


class TestForwardModel(unittest.TestCase):
    def test_numpy_skin_matches_anny(self):
        import torch

        from opensculptboy.character import Character
        from opensculptboy.statues.chains import _Skinner

        model = _model()
        import anny.poses

        pose = anny.poses.pose_parameters(model, "mh_fight")["pose_parameters"]
        card = Character(
            phenotype={"muscle": 0.8, "height": 0.3},
            local_changes={"r-hand-scale-incr": 0.4},
            pose=Character.pose_from_parameters(model, pose),
        )
        a = np.radians(30.0)
        R = 1.1 * np.array(
            [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]
        )
        t = np.array([0.1, -0.2, 0.9])
        rig, rotations = _rig(card, R, t)
        skin = _Skinner(model, card, R, t)
        with torch.no_grad():
            expected = skin(card.pose_parameters(model)).numpy()
        # the model is exact: the same posed and placed skin to the rounding of float64
        self.assertLess(np.abs(rig.vertices(rotations) - expected).max(), 1e-9)
        joints = rig.to_output(rig.joints(rig.transforms(rotations)))
        self.assertLess(np.abs(joints - skin.joints.numpy()).max(), 1e-9)


class TestPointSet(unittest.TestCase):
    def test_stored_set_matches_a_build(self):
        from opensculptboy.posing.reference_points import build_point_set, load_points

        rig, _ = _rig()
        built = {p["name"]: p for p in build_point_set(rig)}
        stored = load_points()
        self.assertEqual(sorted(built), sorted(p["name"] for p in stored))
        for p in stored:
            self.assertEqual(p["vertices"], built[p["name"]]["vertices"], p["name"])
            np.testing.assert_allclose(
                p["barycentric"], built[p["name"]]["barycentric"], atol=1e-5
            )

    def test_names_follow_mediapipe_and_the_body(self):
        from opensculptboy.posing.reference_points import HAND_NAMES, load_points

        points = load_points()
        for side in (".R", ".L"):
            hand = [p for p in points if p["chain"] == "hand." + side[1].lower()]
            self.assertEqual([p["name"] for p in hand], [n + side for n in HAND_NAMES])
            self.assertEqual([p["mediapipe"] for p in hand], list(range(21)))
            body = {
                p["name"] for p in points if p["chain"] != "hand." + side[1].lower()
            }
            for n in ("shoulder", "elbow", "hip", "knee", "ankle", "heel", "toe"):
                self.assertIn(n + side, body)
        self.assertIn("head_top", {p["name"] for p in points})

    def test_points_lie_on_the_skin_near_their_joints_at_rest_and_in_a_fist(self):
        import torch

        from opensculptboy.character import Character
        from opensculptboy.posing.point_place import closest_on_triangles
        from opensculptboy.posing.point_solve import _hand_axes
        from opensculptboy.posing.reference_points import load_points, positions

        model = _model()
        points = load_points()
        rig, rest = _rig()
        fist = rest.copy()
        library = _library("mh_fight")
        for b, label in enumerate(rig.labels):
            if label.endswith(".R") and label.startswith(("finger", "metacarpal")):
                fist[b] = library[b]
        palm, _ = _hand_axes(rig, ".R")
        rest_distance = {}
        for name, R in (("rest", rest), ("fist", fist)):
            X = positions(rig, R, points)
            K = rig.transforms(R)
            J = rig.to_output(rig.joints(K))
            # Anny's own posed skin
            card = Character(
                pose=Character.pose_from_parameters(model, _as_pose(R, rig))
            )
            with torch.no_grad():
                V = model(pose_parameters=card.pose_parameters(model))["vertices"][0]
            V = V.double().numpy()
            for p, x in zip(points, X):
                with self.subTest(pose=name, point=p["name"]):
                    rows = np.unique(
                        np.nonzero(np.linalg.norm(V - x, axis=1) < 0.03)[0]
                    )
                    tri = model.faces.cpu().numpy()
                    near = tri[np.isin(tri, rows).any(1)]
                    _, _, d = closest_on_triangles(x[None], V, near)
                    self.assertLess(d[0], 0.002)
                    joint = J[rig.index[p["joint"]]]
                    dist = float(np.linalg.norm(x - joint))
                    limit = 0.032 if p["chain"].startswith("hand") else 0.16
                    self.assertLess(dist, limit)
                    if name == "rest":
                        rest_distance[p["name"]] = dist
                        if (
                            p["chain"] == "hand.r"
                            and not p["name"].startswith(("thumb", "wrist"))
                            and not p["end"]
                        ):
                            # on the back of the hand
                            self.assertLess((x - joint) @ palm, 0.0)
                    else:
                        # the point rides with its joint
                        self.assertLess(dist, 1.3 * rest_distance[p["name"]] + 0.003)

    def test_left_points_mirror_the_right(self):
        from opensculptboy.posing.reference_points import load_points, positions

        rig, rest = _rig()
        points = load_points()
        X = dict(zip((p["name"] for p in points), positions(rig, rest, points)))
        for name, x in X.items():
            if name.endswith(".R"):
                mirrored = X[name[:-2] + ".L"] * [-1.0, 1.0, 1.0]
                self.assertLess(np.linalg.norm(mirrored - x), 0.001, name)


def _as_pose(R, rig):
    import torch

    P = torch.eye(4, dtype=torch.float64).repeat(1, len(rig.labels), 1, 1)
    P[0, :, :3, :3] = torch.as_tensor(R)
    return P


class TestSolve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig, cls.rest = _rig()

    def test_a_pose_is_found_again_from_its_points(self):
        from opensculptboy.posing.point_solve import solve

        rig = self.rig
        for chains, seed in ((["arm.r"], 0), (["arm.l"], 1), (["leg.r"], 2)):
            with self.subTest(chains=chains):
                truth = _random_pose(rig, self.rest, chains, seed)
                _, points = _targets(rig, truth, chains)
                result = solve(
                    rig, self.rest, points, chains, prior=1e-6, iterations=100
                )
                self.assertLess(max(result.residual_mm), 0.1)
                moved = np.abs(rig.vertices(result.rotations) - rig.vertices(truth))
                self.assertLess(moved.max(), 0.001)
                # with the default pull to the start, each point stays within a millimetre
                result = solve(rig, self.rest, points, chains)
                self.assertLess(max(result.residual_mm), 1.0)

    def test_an_off_finger_curls_with_its_neighbours(self):
        from opensculptboy.posing import point_solve as ps

        rig = self.rig
        truth = _random_pose(rig, self.rest, ["arm.r"], 3, share=(0.4, 0.6))
        ring = {f"ring_{k}.R" for k in ("mcp", "pip", "dip", "tip")}
        _, points = _targets(rig, truth, ["arm.r"], off=ring)
        result = ps.solve(rig, self.rest, points, ["arm.r"])
        c = ps.coordinates(rig, result.rotations)
        for j in (1, 2, 3):
            neighbours = np.mean([c[f"finger3-{j}.R"][0], c[f"finger5-{j}.R"][0]])
            self.assertAlmostEqual(c[f"finger4-{j}.R"][0], neighbours, delta=4.0)
        self.assertEqual(sum(d is None for d in result.residual_mm), 4)

    def test_the_ranges_hold(self):
        from opensculptboy.posing import point_solve as ps

        rig = self.rig
        frames = ps.joint_frames(rig)
        elbow = rig.index["lowerarm01.R"]
        # the forearm bent backwards past its range: the solve stops at the range's end
        truth = self.rest.copy()
        f = frames[elbow]
        truth[elbow] = ps._exp(f["axes"] @ (f["sign"] * np.radians([-40.0, 0.0, 0.0])))
        _, points = _targets(rig, truth, ["arm.r"])
        held = ps.solve(rig, self.rest, points, ["arm.r"])
        free = ps.solve(rig, self.rest, points, ["arm.r"], limits=False)
        lo = np.degrees(f["lo"][0])
        self.assertGreaterEqual(
            ps.coordinates(rig, held.rotations)["lowerarm01.R"][0], lo - 1e-6
        )
        self.assertLess(
            ps.coordinates(rig, free.rotations)["lowerarm01.R"][0], lo - 10.0
        )

    def test_a_twist_spreads_along_the_forearm(self):
        from opensculptboy.posing import point_solve as ps

        rig = self.rig
        frames = ps.joint_frames(rig)
        table = ps.load_ranges()["twist"]["forearm"]["shares"]
        tau = np.radians(80.0)

        def twisted(shares):
            R = self.rest.copy()
            for name, share in shares.items():
                b = rig.index[name + ".R"]
                f = frames[b]
                R[b] = ps._exp(
                    f["axes"] @ (f["sign"] * np.array([0.0, 0.0, share * tau]))
                )
            return R

        truth = twisted(table)
        _, points = _targets(rig, truth, ["arm.r"])
        result = ps.solve(rig, self.rest, points, ["arm.r"], prior=1e-6, iterations=100)
        self.assertLess(max(result.residual_mm), 0.2)
        shared = twist_profile(rig, self.rest, result.rotations)
        kinked = twist_profile(rig, self.rest, twisted({"lowerarm02": 1.0}))
        # no kink: the skin's turn steps a little from one ring of the forearm to the next
        self.assertLess(np.abs(np.diff(shared)).max(), 0.16 * np.degrees(tau))
        # the measure sees a twist kept in one bone
        self.assertGreater(np.abs(np.diff(kinked)).max(), 0.18 * np.degrees(tau))

    def test_contact_pins_hold_their_points(self):
        from opensculptboy.posing import point_solve as ps
        from opensculptboy.posing.reference_points import influences_of, load_points

        rig = self.rig
        truth = _random_pose(rig, self.rest, ["arm.r"], 5)
        use, points = _targets(rig, truth, ["arm.r"])
        names = [p["name"] for p in use]
        # the wrist's point held on a plane 5 mm above where the points put it, and the
        # middle knuckle held on a point
        wrist = points[names.index("wrist.R")]
        knuckle = points[names.index("middle_mcp.R")]
        stored = {p["name"]: p for p in load_points()}
        pin_infl = influences_of(rig, [stored["wrist.R"], stored["middle_mcp.R"]])
        pins = [
            {
                "influence": pin_infl[0],
                "kind": "plane",
                "point": np.asarray(wrist["target"]) + [0.0, 0.0, 0.005],
                "normal": [0.0, 0.0, 1.0],
            },
            {"influence": pin_infl[1], "kind": "point", "target": knuckle["target"]},
        ]
        result = ps.solve(rig, self.rest, points, ["arm.r"], pins=pins)
        self.assertLess(max(result.pins_mm), 1.0)
        # a pin on a surface given as a function: a sphere round the knuckle
        centre = np.asarray(knuckle["target"]) + [0.0, -0.03, 0.0]

        def sphere(x):
            n = (x - centre) / np.linalg.norm(x - centre)
            return centre + 0.02 * n, n

        pins = [
            {
                "influence": pin_infl[1],
                "kind": "plane",
                "point": centre,
                "normal": [0, 1, 0],
            }
        ]
        result = ps.solve(
            rig, self.rest, points, ["arm.r"], pins=pins, surfaces={0: sphere}
        )
        x = result.points[names.index("middle_mcp.R")]
        self.assertLess(abs(np.linalg.norm(x - centre) - 0.02), 0.001)


def twist_profile(rig, rest, R, side=".R", bins=14) -> np.ndarray:
    """The skin's mean turn (degrees) about the forearm's axis in rings from the upper arm's
    end to the hand, for a pose R against ``rest``."""
    ix = rig.index
    J = rig.joints(rig.transforms(rest))
    elbow, wrist = J[ix["lowerarm01" + side]], J[ix["wrist" + side]]
    axis = (wrist - elbow) / np.linalg.norm(wrist - elbow)
    V0 = rig.from_output(rig.vertices(rest))
    V1 = rig.from_output(rig.vertices(R))
    strongest = rig.indices[np.arange(len(rig.indices)), rig.weights.argmax(1)]
    bones = [ix[b + side] for b in ("upperarm02", "lowerarm01", "lowerarm02", "wrist")]
    m = np.isin(strongest, bones)
    t = ((V0[m] - elbow) @ axis) / np.linalg.norm(wrist - elbow)
    r0 = V0[m] - elbow
    r0 -= (r0 @ axis)[:, None] * axis
    r1 = V1[m] - elbow
    r1 -= (r1 @ axis)[:, None] * axis
    angle = np.degrees(np.arctan2(np.cross(r0, r1) @ axis, (r0 * r1).sum(1)))
    edges = np.linspace(-0.25, 1.15, bins + 1)
    return np.array(
        [angle[(t >= a) & (t < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
    )


@unittest.skipUnless(_node_available(), "node 22 or later runs the page's TypeScript")
class TestPageParity(unittest.TestCase):
    def test_the_page_finds_the_same_pose(self):
        from opensculptboy.posing import point_solve as ps

        rig, rest = _rig()
        truth = _random_pose(rig, rest, ["arm.r"], 4)
        ring = {f"ring_{k}.R" for k in ("mcp", "pip", "dip", "tip")}
        use, points = _targets(rig, truth, ["arm.r"], off=ring)
        stored = {p["name"]: p for p in use}
        from opensculptboy.posing.reference_points import influences_of

        pin = influences_of(rig, [stored["wrist.R"]])[0]
        problems = [
            ps.build_problem(rig, rest, points, ["arm.r"]),
            ps.build_problem(
                rig,
                rest,
                points,
                ["arm.r"],
                limits=False,
                pins=[
                    {
                        "influence": pin,
                        "kind": "plane",
                        "point": np.asarray(points[0]["target"]) + [0.0, 0.0, 0.01],
                        "normal": [0.0, 0.0, 1.0],
                    }
                ],
            ),
            ps.build_problem(
                rig,
                rest,
                _targets(rig, _random_pose(rig, rest, ["leg.l"], 6), ["leg.l"])[1],
                ["leg.l"],
            ),
        ]
        python = [ps.solve_problem(json.loads(json.dumps(p))) for p in problems]
        # a drag: the index fingertip's place moves 1 mm a frame, each frame from the last pose
        index = [p["name"] for p in use].index("index_tip.R")
        settled = ps.solve(rig, rest, points, ["arm.r"])
        drag = ps.build_problem(rig, settled.rotations, points, ["arm.r"])
        start = np.asarray(points[index]["target"])
        frames = [(start + [0.001 * k, 0.0005 * k, 0.0]).tolist() for k in range(1, 61)]
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, "problems.json").write_text(json.dumps(problems))
            pathlib.Path(tmp, "drag.json").write_text(
                json.dumps(
                    {"problem": drag, "point": index, "frames": frames, "iterations": 8}
                )
            )
            subprocess.run(["node", str(NODE_SCRIPT), tmp], check=True)
            out = json.loads(pathlib.Path(tmp, "output.json").read_text())
        influences = [p["influence"] for p in points]
        for (D, _, _), page in zip(python, out["results"]):
            Dp = np.asarray(page["rotations"]).reshape(-1, 3, 3)
            self.assertLess(np.abs(rig.vertices(D) - rig.vertices(Dp)).max(), 1e-4)
        for (D, _, _), page in zip(python[:2], out["results"][:2]):
            X = rig.to_output(rig.points(rig.transforms(D), influences))
            self.assertLess(np.abs(X - np.asarray(page["points"])).max(), 1e-4)
        times = np.asarray(out["drag"]["times"])
        print(
            f"page solve of an arm and hand, a drag of {len(times)} frames: mean "
            f"{times.mean():.2f} ms, median {np.median(times):.2f} ms, max {times.max():.2f} ms"
        )
        # the drag follows: the fingertip ends at its last place
        tip = np.asarray(out["drag"]["points"][index])
        self.assertLess(np.linalg.norm(tip - frames[-1]), 0.002)
        self.assertLess(float(np.median(times)), 10.0)


class TestPlacement(unittest.TestCase):
    def test_points_land_on_a_scan_with_their_confidence(self):
        from opensculptboy.posing.point_place import _vertex_normals, place_points
        from opensculptboy.posing.reference_points import load_points

        rig, rest = _rig()
        points = load_points()
        V = rig.vertices(rest)
        # a scan 2 mm outside the skin, without the left hand
        scan = V + 0.002 * _vertex_normals(V, rig.faces)
        strongest = rig.indices[np.arange(len(rig.indices)), rig.weights.argmax(1)]
        left_hand = np.array(
            [
                rig.labels[b].endswith(".L")
                and rig.labels[b].startswith(("wrist", "metacarpal", "finger"))
                for b in strongest
            ]
        )
        T = rig.faces[~left_hand[rig.faces].any(1)]
        placed = {e["name"]: e for e in place_points(rig, rest, points, scan, T)}
        for name, e in placed.items():
            with self.subTest(point=name):
                word = name[:-2]
                if name.endswith(".L") and word in _hand_words() and word != "wrist":
                    # the lost hand's points find no scan
                    self.assertFalse(e["on"])
                elif name.endswith(".R") or name == "head_top":
                    self.assertTrue(e["on"])
                    self.assertEqual(e["method"], "normal")
                    self.assertAlmostEqual(e["distance_mm"], 2.0, delta=0.3)

    def test_a_fist_marks_its_fused_points(self):
        from opensculptboy.posing.point_place import _vertex_normals, place_points
        from opensculptboy.posing.reference_points import points_of

        rig, rest = _rig()
        fist = rest.copy()
        library = _library("mh_fight")
        for b, label in enumerate(rig.labels):
            if label.endswith(".R") and label.startswith(("finger", "metacarpal")):
                fist[b] = library[b]
        V = rig.vertices(fist)
        scan = V + 0.002 * _vertex_normals(V, rig.faces)
        placed = place_points(rig, fist, points_of(["hand.r"]), scan, rig.faces)
        by = {e["name"]: e for e in placed}
        # the knuckles stand clear; somewhere in the closed fingers parts press together
        for n in ("index_mcp.R", "middle_mcp.R", "ring_mcp.R", "pinky_mcp.R"):
            self.assertTrue(by[n]["on"], n)
        self.assertTrue(any(e["marks"]["fused"] < 1.0 for e in placed))

    def test_the_placement_is_a_config_entry(self):
        from opensculptboy.posing.point_place import placement_json

        data = placement_json(
            [
                {
                    "name": "wrist.R",
                    "position": [0.1, 0.2, 0.3],
                    "confidence": 0.9,
                    "on": True,
                }
            ]
        )
        entry = data["chains"]["hand.r"][0]
        # the masks' frame: (x, z, -y) of Anny's
        np.testing.assert_allclose(entry["position"], [0.1, 0.3, -0.2])


def _hand_words():
    from opensculptboy.posing.reference_points import HAND_NAMES

    return set(HAND_NAMES)


class TestContact(unittest.TestCase):
    def test_squeeze_closes_onto_a_ball(self):
        from opensculptboy.posing import contact
        from opensculptboy.posing.point_solve import _hand_axes

        rig, rest = _rig()
        J = rig.to_output(rig.joints(rig.transforms(rest)))
        ix = rig.index
        palm, _ = _hand_axes(rig, ".R")
        centre = 0.5 * (J[ix["wrist.R"]] + J[ix["finger3-1.R"]]) + palm * 0.042
        ball = {"kind": "sphere", "centre": centre.tolist(), "radius": 0.03}
        fingers = ("finger2", "finger3", "finger4")
        closed, report = contact.squeeze(
            rig, rest, "hand.r", [ball], 1.0, fingers=fingers
        )
        for f in fingers:
            self.assertIsNotNone(report[f]["contact"])
            self.assertAlmostEqual(report[f]["clearance_mm"], 1.0, delta=0.15)
        half, report_half = contact.squeeze(
            rig, rest, "hand.r", [ball], 0.5, fingers=fingers
        )
        for f in fingers:
            self.assertAlmostEqual(
                report_half[f]["scale"], 0.5 * report[f]["contact"], places=3
            )
            self.assertGreater(report_half[f]["clearance_mm"], 2.0)

    def test_centre_of_mass_of_a_ball(self):
        import trimesh

        from opensculptboy.posing.contact import centre_of_mass

        ball = trimesh.creation.icosphere(subdivisions=4, radius=0.1)
        centre, volume = centre_of_mass(
            np.asarray(ball.vertices) + [1.0, 2.0, 3.0], ball.faces
        )
        np.testing.assert_allclose(centre, [1.0, 2.0, 3.0], atol=1e-9)
        self.assertAlmostEqual(
            volume, 4 / 3 * np.pi * 0.1**3, delta=0.002 * 4 / 3 * np.pi * 0.1**3
        )

    def test_the_balance_mark_follows_the_body(self):
        from opensculptboy.posing import contact
        from opensculptboy.posing import point_solve as ps

        rig, rest = _rig()
        still = contact.balance(rig, rest)
        self.assertTrue(still["inside"])
        self.assertGreater(still["margin_mm"], 20.0)
        self.assertLess(abs(still["mark"][0]), 0.01)
        self.assertGreater(len(still["polygon"]), 3)
        # the body bent forwards at the waist: the mark moves forwards (-y) and leaves the feet
        frames = ps.joint_frames(rig)
        bent = rest.copy()
        for name in ("spine04", "spine03", "spine02"):
            f = frames[rig.index[name]]
            bent[rig.index[name]] = ps._exp(
                f["axes"] @ (f["sign"] * np.radians([30.0, 0.0, 0.0]))
            )
        leaning = contact.balance(rig, bent)
        self.assertLess(leaning["mark"][1], still["mark"][1] - 0.03)


class TestStatuePoints(unittest.TestCase):
    def test_a_config_takes_reference_points(self):
        from opensculptboy.statues.pipeline import StatueConfig

        base = {"format": "opensculptboy.statue/1", "name": "kouros", "scan": "k.stl"}
        config = StatueConfig.from_dict(
            {
                **base,
                "reference_points": {
                    "arm.r": [{"name": "wrist.R", "position": [0, 1, 0]}]
                },
            }
        )
        self.assertIn("arm.r", config.reference_points)
        with self.assertRaises(ValueError):
            StatueConfig.from_dict(
                {
                    **base,
                    "reference_points": {
                        "arm.r": [{"name": "elbow.X", "position": [0, 1, 0]}]
                    },
                }
            )
        with self.assertRaises(ValueError):
            StatueConfig.from_dict(
                {
                    **base,
                    "reference_points": {
                        "wing.r": [{"name": "wrist.R", "position": [0, 1, 0]}]
                    },
                }
            )

    def test_a_synthetic_statue_takes_its_reference_points(self):
        import types
        from unittest import mock

        from opensculptboy.character import Character
        from opensculptboy.fit.handoff import GLTF_TO_ANNY
        from opensculptboy.posing.point_place import place_on_handoff
        from opensculptboy.posing.reference_points import (
            load_points,
            points_of,
            positions,
        )
        from opensculptboy.statues import fit_statue, scan_to_handoff
        from opensculptboy.statues.fit import stage1_dict
        from test.statue_fixtures import to_scan

        model = _model()
        statue = synthetic_statue(0)
        true = Character(
            phenotype=dict(HEROIC),
            pose=Character.pose_from_parameters(model, statue["card"]["pose"]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp) / "kouros" / "handoff"
            conversion = scan_to_handoff(
                (statue["vertices"], statue["triangles"]), folder
            )

            def place(x):
                return conversion.to_anny(to_scan(statue, x))

            # the placement of Anny's frame on the handoff's: a similarity
            c = place(np.zeros((1, 3)))[0]
            M = (place(np.eye(3)) - c).T
            # the first placement on the scan from the true card: the right hand lands on the
            # scan, and the lost left hand is unsure
            placed = place_on_handoff(folder, true, M, c)
            by = {e["name"]: e for e in placed["points"]}
            for n in ("index_tip.R", "middle_mcp.R", "elbow.R"):
                self.assertTrue(by[n]["on"], n)
                self.assertLess(by[n]["distance_mm"], 3.0)
            self.assertFalse(by["middle_mcp.L"]["on"])
            # the owner's points: where the true body has them (the masks' frame)
            rig, R_true = _rig(true, M, c)
            chosen = {}
            for chain in ("arm.r", "arm.l"):
                use = points_of([chain])
                X = positions(rig, R_true, use)
                chosen[chain] = [
                    {"name": p["name"], "position": (x @ GLTF_TO_ANNY).tolist()}
                    for p, x in zip(use, X)
                ]
            from opensculptboy.statues.points import in_anny

            # the card to fit: the true one with both arms at rest
            moved = R_true.copy()
            from opensculptboy.posing.point_solve import chain_bones

            for b in chain_bones(rig, ["arm.r", "arm.l"]):
                moved[b] = np.eye(3)
            start = Character(
                phenotype=dict(HEROIC),
                pose=Character.pose_from_parameters(
                    model, _as_pose_from(model, statue, moved)
                ),
            )
            data = json.loads(
                json.dumps(
                    stage1_dict(
                        types.SimpleNamespace(
                            character=start, rotation=M, translation=c
                        )
                    )
                )
            )

            def levels(reference, character, rotation, translation, *args, **kwargs):
                return types.SimpleNamespace(
                    character=character, rotation=rotation, translation=translation
                )

            def repose(model_, result, character):
                return types.SimpleNamespace(
                    character=character,
                    rotation=result.rotation,
                    translation=result.translation,
                )

            with (
                mock.patch("opensculptboy.fit.register.fit_levels", side_effect=levels),
                mock.patch(
                    "opensculptboy.fit.register.repose_fit", side_effect=repose
                ) as rp,
            ):
                result = fit_statue(
                    folder,
                    detail=0,
                    stage1=data,
                    posed_hands={
                        "hand.r": {"after_fit": False},
                        "hand.l": {"after_fit": True},
                    },
                    reference_points=in_anny(chosen),
                )
        # the right hand posed before the fit, the left after it; both follow their points
        self.assertTrue(rp.called)
        for chain in ("hand.r", "hand.l"):
            report = result.hands[chain]["points"]
            self.assertEqual(result.hands[chain]["curl"], "points")
            self.assertLess(report["max_mm"], 1.5, chain)
        rig_after, R_after = _rig(result.character, M, c)
        strongest = rig.indices[np.arange(len(rig.indices)), rig.weights.argmax(1)]
        hand = np.isin(
            strongest,
            [
                rig.index[b]
                for b in rig.labels
                if b.endswith(".R") and b.startswith("finger")
            ],
        )
        gap = np.linalg.norm(
            rig_after.vertices(R_after)[hand] - rig.vertices(R_true)[hand], axis=1
        )
        self.assertLess(float(np.median(gap)), 0.002)
        self.assertGreater(len(load_points()), 50)


def _as_pose_from(model, statue, R):
    import torch

    P = statue["card"]["pose"].detach().clone().double()
    P[0, :, :3, :3] = torch.as_tensor(R)
    return P


if __name__ == "__main__":
    unittest.main()

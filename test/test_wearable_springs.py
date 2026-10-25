# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the spring chains of loose wearables (``opensculptboy.wearables.springs``,
docs/wearables.md, "Springs") with synthetic garments only: a long cape and a skirt on
anny's template (``test.wearable_fixtures``). The package entries, the chain structure, the
weights and the attachment at the top, the body colliders, the GLB and VRM files of both
versions within their budgets, and a few seconds of the VRM spring update in which the cape
and the skirt swing and stay outside the body's capsules.
"""

import contextlib
import io
import json
import pathlib
import re
import tempfile
import unittest

import numpy as np

from opensculptboy import Character, export_glb
from opensculptboy.export.vrm import export_vrm, vrm_spec
from opensculptboy.wearables import WornPiece, card_body, dress, read_wearable
from opensculptboy.wearables import write_wearable
from opensculptboy.wearables.export import Dressing
from opensculptboy.wearables.springs import (
    CLEARANCE,
    CLOAK_ROOTS,
    COLLIDER_TABLE,
    TORSO,
    SpringSim,
    SpringWarning,
    body_colliders,
    collider_distance,
    parse_springs,
    rig_springs,
)
from test.gltf_reader import GLB, evaluate_nodes, structure_errors
from test.test_vrm_schema import check_vrm0, check_vrm1
from test.wearable_fixtures import (
    long_cape,
    skirt,
    template_body,
    template_model,
    write_loose,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
C = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # Anny's frame to glTF's


def _worn(*wearables):
    return [
        WornPiece(f"loose/{w.name}", w.name, "loose", "", w, True) for w in wearables
    ]


def _parents():
    return [int(p) for p in template_model().bone_parents]


def _rigged(*wearables, max_skin_joints=None):
    body = template_body()
    pieces = dress(body, _worn(*wearables))
    return pieces, rig_springs(body, pieces, _parents(), max_skin_joints)


class TestSpringEntries(unittest.TestCase):
    def test_round_trip_and_defaults(self):
        springs = [
            {"name": "back", "settings": {"stiffness": 2.0, "gravity_dir": [0, -2, 0]}},
            {
                "name": "tail",
                "points": [[0, 1.0, -0.1], [0, 0.6, -0.15], [0, 0.3, -0.2]],
            },
        ]
        wearable = long_cape(springs=springs)
        with tempfile.TemporaryDirectory() as folder:
            path = write_wearable(pathlib.Path(folder) / "cape.wearable.json", wearable)
            back = read_wearable(path)
        self.assertEqual(back.springs, springs)
        auto, chain = parse_springs(back)
        self.assertEqual(
            (auto.kind, auto.preset, auto.columns, auto.joints), ("auto", "cape", 5, 4)
        )
        self.assertEqual(auto.settings.stiffness, 2.0)
        self.assertEqual(auto.settings.json()["gravity_dir"], [0.0, -1.0, 0.0])
        np.testing.assert_allclose(auto.settings.gravity_dir, [0, 0, -1])
        self.assertEqual((chain.kind, chain.joints), ("chain", 3))
        np.testing.assert_allclose(
            chain.points[0], [0.0, 0.1, 1.0]
        )  # glTF axes to Anny's
        self.assertEqual(parse_springs(skirt())[0].parent, "root")
        self.assertEqual(parse_springs(skirt())[0].preset, "skirt")

    def test_bad_entries(self):
        for entry in (
            {"name": "a", "stiff": 1},
            {"name": "a", "preset": "scarf"},
            {"name": "a", "groups": ["sleeve"]},
            {"name": "a", "colliders": ["tail"]},
            {"name": "a", "settings": {"drag_force": 2.0}},
            {"name": "a", "settings": {"stiffness": -1}},
            {"name": "a", "points": [[0, 0, 0]]},
            {"name": "a", "joints": 1},
            {"kind": "auto"},
        ):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                long_cape(springs=[entry]).check()
        with self.assertRaises(ValueError):
            long_cape(springs=[{"name": "a"}, {"name": "a"}]).check()


class TestChains(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = template_body()
        cls.labels = list(cls.body.bone_labels)
        cls.plain = dress(cls.body, _worn(long_cape(), skirt()))
        cls.pieces, cls.rig = _rigged(long_cape(), skirt())

    def chains(self, piece):
        return [c for c in self.rig.chains if c.piece == f"loose/{piece}"]

    def test_chain_structure(self):
        cape, hem = self.chains("long_cape"), self.chains("skirt")
        self.assertEqual([c.count for c in cape], [4] * 5)
        self.assertEqual([c.count for c in hem], [4] * 8)
        self.assertEqual({c.parent for c in hem}, {"root"})
        self.assertTrue(all(c.parent in TORSO for c in cape))
        for chain in self.rig.chains:
            self.assertTrue((np.diff(chain.positions[:, 2]) < 0).all(), chain.name)
        # the cape's columns spread across the back, the skirt's all around the hips
        xs = sorted(c.positions[0, 0] for c in cape)
        self.assertLess(xs[0], -0.15)
        self.assertGreater(xs[-1], 0.15)
        root = self.body.bone_poses[self.labels.index("root"), :2, 3]
        az = sorted(np.arctan2(*(c.positions[-1, :2] - root)[::-1]) for c in hem)
        self.assertLess(np.diff(az + [az[0] + 2 * np.pi]).max(), 2 * np.pi / 8 * 1.6)
        # one parent line per chain, each node in one chain, the tails last
        J = len(self.labels)
        nodes = self.rig.nodes(self.labels)
        self.assertEqual(len(nodes), sum(c.count for c in self.rig.chains))
        self.assertEqual(len({n.name for n in nodes}), len(nodes))
        skin = self.rig.skin_joint_count
        for k, node in enumerate(nodes):
            self.assertEqual(node.tail, k >= skin)
            if node.joint == 0:
                self.assertLess(node.parent, J)
            else:
                parent = nodes[node.parent - J]
                self.assertEqual(
                    (parent.chain, parent.joint), (node.chain, node.joint - 1)
                )
        self.assertEqual(sum(n.tail for n in nodes), len(self.rig.chains))

    def test_weights_attach_at_the_top(self):
        J = len(self.labels)
        for piece, plain in zip(self.pieces, self.plain):
            with self.subTest(piece.ref):
                np.testing.assert_allclose(piece.weights.sum(axis=1), 1.0, atol=1e-9)
                z = piece.vertices[:, 2]
                chain = (piece.joints >= J) * piece.weights
                top = z >= z.max() - 1e-6
                bottom = z <= z.min() + 1e-6
                # the top row keeps the body's weights; the hem moves with the chains alone
                np.testing.assert_array_equal(chain[top].sum(axis=1), 0.0)
                np.testing.assert_allclose(piece.weights[top], plain.weights[top])
                np.testing.assert_array_equal(piece.joints[top], plain.joints[top])
                self.assertGreater(chain[bottom].sum(axis=1).min(), 0.99)
                # the share of the chains grows down the piece
                rows = [
                    chain[np.abs(z - h) < 0.02].sum(axis=1).mean()
                    for h in np.linspace(z.max(), z.min(), 6)
                ]
                self.assertTrue((np.diff(rows) >= -1e-9).all(), rows)

    def test_cloak_hangs_from_the_upper_body(self):
        """A cloak whose top rows are weighted to the head and the neck (a hood, a mantle
        over the shoulders) hangs its chains from the spine or the clavicles."""
        body = template_body()
        (piece,) = dress(body, _worn(long_cape()))
        labels = list(body.bone_labels)
        z = piece.vertices[:, 2]
        top = z > z.max() - 0.25
        piece.joints[top] = 0
        piece.weights[top] = 0.0
        piece.joints[top, 0] = labels.index("head")
        piece.joints[top, 1] = labels.index("neck02")
        piece.weights[top, 0] = 0.7
        piece.weights[top, 1] = 0.3
        rig = rig_springs(body, [piece], _parents())
        parents = {c.parent for c in rig.chains}
        self.assertTrue(parents <= set(CLOAK_ROOTS), parents)
        self.assertFalse(any(p.startswith(("neck", "head")) for p in parents))

    def test_hit_radii_keep_the_rest_shape(self):
        arrays = np.array([c.array() for c in self.rig.colliders])
        for chain in self.rig.chains:
            d = collider_distance(arrays, chain.positions).min(axis=1)
            self.assertTrue(
                (chain.hit_radii <= chain.settings.hit_radius + 1e-12).all()
            )
            hit = chain.hit_radii > 0
            self.assertTrue((d[hit] - chain.hit_radii[hit] >= CLEARANCE - 1e-9).all())

    def test_authored_chain_and_reach(self):
        # glTF axes on the binding body at rest (the package's GLB): down the cape's middle
        points = [
            [0.0, 0.45, -0.10],
            [0.0, 0.1, -0.12],
            [0.0, -0.25, -0.14],
            [0.0, -0.6, -0.15],
        ]
        spring = {"name": "middle", "points": points, "preset": "hem", "reach": 0.08}
        pieces, rig = _rigged(long_cape(springs=[spring]))
        (chain,) = rig.chains
        self.assertEqual(chain.count, 4)
        self.assertEqual(chain.parent, "spine01")
        piece = pieces[0]
        J = len(self.labels)
        share = ((piece.joints >= J) * piece.weights).sum(axis=1)
        far = np.abs(piece.vertices[:, 0]) > 0.08 + 0.01
        self.assertEqual(float(share[far].max()), 0.0)
        self.assertGreater(float(share[~far].max()), 0.9)

    def test_joint_budget(self):
        with self.assertWarns(SpringWarning):
            _, rig = _rigged(long_cape(), skirt(), max_skin_joints=24)
        self.assertLessEqual(rig.skin_joint_count, 24)
        self.assertTrue(rig.messages)
        self.assertTrue(all(c.count >= 3 for c in rig.chains))
        with self.assertWarnsRegex(SpringWarning, "left out"):
            self.assertIsNone(_rigged(long_cape(), max_skin_joints=1)[1])


class TestColliders(unittest.TestCase):
    def test_inside_the_skin_and_outside_the_garments(self):
        body = template_body()
        colliders = body_colliders(body)
        names = [c.name for c in colliders]
        self.assertEqual(names, ["head"] + [row[0] for row in COLLIDER_TABLE])
        arrays = np.array([c.array() for c in colliders])
        # every collider lies inside the skin: no skin vertex lies inside a collider by more
        # than a few millimetres (the quantile leaves a few of the nearest ones inside)
        skin = body.vertices[np.unique(body.triangles)]
        inside = collider_distance(arrays, skin) < -0.005
        self.assertLess(inside.mean(axis=0).max(), 0.02)
        for wearable in (long_cape(), skirt()):
            d = collider_distance(arrays, dress(body, _worn(wearable))[0].vertices)
            self.assertGreater(float(d.min()), 0.0, wearable.name)

    def test_scale_with_the_character(self):
        def radius(card, name):
            colliders = body_colliders(card_body(card))
            return next(c for c in colliders if c.name == name)

        base = Character()
        heavy = Character(phenotype={"weight": 0.95})
        tall = Character(phenotype={"height": 0.95})
        self.assertGreater(
            radius(heavy, "hips").radius, 1.05 * radius(base, "hips").radius
        )
        leg = radius(tall, "upperleg.L")
        base_leg = radius(base, "upperleg.L")
        self.assertGreater(
            np.linalg.norm(leg.tail - leg.head),
            np.linalg.norm(base_leg.tail - base_leg.head),
        )

    def test_shared_with_the_viewer(self):
        """The colliders that the hair of the viewer also has take its ends, bones,
        quantiles and spans (viewer/src/hair/colliders.ts)."""
        text = (ROOT / "viewer" / "src" / "hair" / "colliders.ts").read_text()
        rows = {}
        for match in re.finditer(
            r"\{ name: '([^']+)', ends: \['([^']+)', '([^']+)'\], bones: \['([^']+)', '([^']+)'\],([^}]*)\}",
            text,
        ):
            name, a, b, bone, _, rest = match.groups()
            q = re.search(r"quantile: ([0-9.]+)", rest)
            span = re.search(r"span: \[([0-9.]+), ([0-9.]+)\]", rest)
            rows[name] = (
                (a, b),
                bone,
                q and float(q.group(1)),
                span and (float(span.group(1)), float(span.group(2))),
            )
        shared = [r for r in COLLIDER_TABLE if r[0] in rows]
        self.assertEqual(
            [r[0] for r in shared], ["neck", "chest", "shoulder.L", "shoulder.R"]
        )
        for name, _, ends, _, quantile, span in shared:
            with self.subTest(name):
                self.assertEqual(rows[name][0], ends)
                self.assertEqual(rows[name][2], quantile)
                self.assertEqual(rows[name][3], span)


def _world(nodes):
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}

    def at(i):
        p = np.zeros(3)
        while i is not None:
            p = p + np.array(nodes[i].get("translation", [0.0, 0.0, 0.0]))
            i = parent.get(i)
        return p

    return at


def _vrm1_sim(doc):
    """A :class:`SpringSim` of a VRM 1.0 file's springs and colliders (the nodes keep the
    identity rotation, so the offsets add to the nodes' positions)."""
    at = _world(doc["nodes"])
    ext = doc["extensions"]["VRMC_springBone"]
    caps = []
    for c in ext["colliders"]:
        w, shape = at(c["node"]), c["shape"]
        if "sphere" in shape:
            a = w + shape["sphere"]["offset"]
            caps.append([*a, *a, shape["sphere"]["radius"]])
        else:
            s = shape["capsule"]
            caps.append([*(w + s["offset"]), *(w + s["tail"]), s["radius"]])
    chains = []
    for spring in ext["springs"]:
        joints = spring["joints"]
        chains.append(
            dict(
                positions=np.array([at(j["node"]) for j in joints]),
                stiffness=np.array([j["stiffness"] for j in joints]),
                gravity_power=np.array([j["gravityPower"] for j in joints]),
                gravity_dir=np.array(joints[0]["gravityDir"]),
                drag_force=np.array([j["dragForce"] for j in joints]),
                hit_radii=np.array([j["hitRadius"] for j in joints]),
                name=spring["name"],
            )
        )
    return SpringSim(chains, np.array(caps)), chains, np.array(caps)


class TestSpringFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        cls.card = write_loose(cls.folder)
        cls.model = template_model()
        cls.glb_path = cls.folder / "loose.glb"
        cls.glb = export_glb(cls.glb_path, cls.card, cls.model, detail_dir=cls.folder)
        cls.vrm = {}
        for version in ("1.0", "0.x"):
            path = cls.folder / f"loose_{version}.vrm"
            summary = export_vrm(
                path,
                cls.card,
                author="test",
                version=version,
                detail_dir=cls.folder,
                refine_weights=False,
            )
            cls.vrm[version] = (path, summary, GLB(path))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_glb(self):
        glb = GLB(self.glb_path)
        self.assertEqual(structure_errors(glb), [])
        J = self.model.bone_count
        nodes = glb.json["nodes"]
        springs = glb.json["scenes"][0]["extras"]["opensculptboy"]["wearables"][
            "springs"
        ]
        self.assertEqual(len(springs["chains"]), 13)
        skin_joints = sum(c["joints"] - 1 for c in springs["chains"])
        self.assertEqual(len(glb.json["skins"][0]["joints"]), J + skin_joints)
        self.assertEqual(self.glb["joints"], J + skin_joints)
        for chain in springs["chains"]:
            first = nodes[chain["nodes"][0]]["name"]
            self.assertEqual(first, chain["name"].replace("/", "_") + ".0")
            self.assertIn("opensculptboy_spring", nodes[chain["nodes"][0]]["extras"])
        self.assertTrue(
            all("node" in c and "offset" in c for c in springs["colliders"])
        )
        # at rest the pieces sit where the fit put them: the chains' inverse binds hold
        dressed = dress(template_body(), _worn(long_cape(), skirt()))
        evaluated = evaluate_nodes(glb)
        names = [n.get("name") for n in nodes]
        for piece in dressed:
            vertices = evaluated[names.index(piece.ref)][0].positions
            expected = piece.vertices @ C.T
            lift = np.median(vertices - expected, axis=0)
            np.testing.assert_allclose(vertices, expected + lift, atol=2e-5)

    def test_vrm_files(self):
        from opensculptboy.export.budget import BUDGETS

        for version, check, target, extension in (
            ("1.0", check_vrm1, "vrm1", "VRMC_springBone"),
            ("0.x", check_vrm0, "vrm0", "VRM"),
        ):
            with self.subTest(version):
                path, summary, glb = self.vrm[version]
                self.assertEqual(structure_errors(glb), [])
                self.assertEqual(check(glb.json), [])
                self.assertIn(extension, glb.json["extensionsUsed"])
                self.assertLessEqual(summary["joints"], BUDGETS[target]["joints"])
                self.assertEqual(summary["budget"], [])
        v1 = self.vrm["1.0"][2].json["extensions"]["VRMC_springBone"]
        self.assertEqual(len(v1["springs"]), 13)
        self.assertEqual(
            {tuple(j["gravityDir"]) for s in v1["springs"] for j in s["joints"]},
            {(0.0, -1.0, 0.0)},
        )
        shapes = [next(iter(c["shape"])) for c in v1["colliders"]]
        self.assertEqual(shapes.count("sphere"), 1)
        v0 = self.vrm["0.x"][2].json["extensions"]["VRM"]["secondaryAnimation"]
        self.assertEqual(
            [g["comment"] for g in v0["boneGroups"]],
            ["loose/long_cape.back", "loose/skirt.hem"],
        )
        self.assertTrue(
            self.vrm["0.x"][1]["spring_messages"]
        )  # 0.x has 128 joints: fewer joints
        self.assertGreater(
            min(len(g["colliders"]) for g in v0["colliderGroups"][1:]), 1
        )

    def test_vrm_schemas(self):
        from test.test_vrm_schema import schema_errors

        try:
            errors = schema_errors(
                "VRMC_springBone.schema.json",
                self.vrm["1.0"][2].json["extensions"]["VRMC_springBone"],
            )
        except unittest.SkipTest as skip:
            self.skipTest(str(skip))
        self.assertEqual(errors, [])
        self.assertEqual(
            schema_errors(
                "vrm.schema.json", self.vrm["0.x"][2].json["extensions"]["VRM"]
            ),
            [],
        )

    def test_vrm_posed_back_to_rest(self):
        """The file's pieces, posed back to Anny's rest pose with the file's weights (the
        chain joints riding on their parent bones), show the pieces' rest vertices."""
        from opensculptboy.export.tpose import blend, rigid_inverse

        spec = vrm_spec(
            self.card,
            version="1.0",
            author="test",
            detail_dir=self.folder,
            refine_weights=False,
        )
        labels = list(self.model.bone_labels)
        X = spec.rebind.transforms
        X = np.concatenate([X, X[spec.springs.rig.parent_bones(labels)]])
        dressed = {
            p.ref: p for p in dress(template_body(), _worn(long_cape(), skirt()))
        }
        for mesh in spec.meshes[1:]:
            body = mesh.body
            bind = (body.positions + mesh.translation - spec.offset) @ spec.frame
            N = blend(body.weights.astype(np.float64), body.joints, rigid_inverse(X))
            rest = np.einsum("nij,nj->ni", N[:, :3, :3], bind) + N[:, :3, 3]
            np.testing.assert_allclose(rest, dressed[mesh.name].vertices, atol=2e-5)

    def test_the_springs_swing_and_stay_outside_the_body(self):
        """Four seconds of the VRM spring update on the VRM 1.0 file: the chains rest within
        4 cm of their shape, swing when the body moves on and back and turns, and keep out
        of the body's capsules (within 5 mm, the slack of the length step after a push)."""
        sim, chains, caps = _vrm1_sim(self.vrm["1.0"][2].json)
        rest = [c["positions"] for c in chains]
        cape = [k for k, c in enumerate(chains) if "long_cape" in c["name"]]
        hem = [k for k, c in enumerate(chains) if "skirt" in c["name"]]

        def turn(a):
            c, s = np.cos(a), np.sin(a)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        M, dt = np.eye(4), 1.0 / 60.0
        swing, worst = {"rest": 0.0, "move": 0.0, "turn": 0.0}, np.inf
        for step in range(240):
            phase = "rest" if step < 60 else "move" if step < 140 else "turn"
            v = {"rest": 0.0, "move": 2.0 if step < 100 else -2.0, "turn": 0.0}[phase]
            w = 4.0 if 140 <= step < 200 else -4.0 if step >= 200 else 0.0
            M[2, 3] += v * dt
            M[:3, :3] = turn(w * dt) @ M[:3, :3]
            sim.step(dt, M)
            joints = sim.joints()
            moved = [
                np.linalg.norm(
                    joints[k] - (rest[k] @ M[:3, :3].T + M[:3, 3]), axis=1
                ).max()
                for k in range(len(joints))
            ]
            swing[phase] = max(swing[phase], max(moved))
            posed = caps.copy()
            posed[:, :3] = caps[:, :3] @ M[:3, :3].T + M[:3, 3]
            posed[:, 3:6] = caps[:, 3:6] @ M[:3, :3].T + M[:3, 3]
            tails = np.vstack([j[1:] for j in joints])
            worst = min(worst, float(collider_distance(posed, tails).min()))
            if phase == "move" and step == 99:
                cape_swing = max(moved[k] for k in cape)
                hem_swing = max(moved[k] for k in hem)
        self.assertLess(swing["rest"], 0.04)
        self.assertGreater(cape_swing, 0.15)
        self.assertGreater(hem_swing, 0.05)
        self.assertGreater(swing["turn"], 0.1)
        self.assertGreater(worst, -0.005)

    def test_walk_lag_and_settle(self):
        """The VRM spring update on the VRM 1.0 file in a walk: the chains hang at rest
        within 3 cm of their shape, in a walk at 1.25 m/s the cape's tails lag 0.1 to
        0.25 m behind their place at rest (the skirt's at most 0.25 m), and within 0.5 s of
        the stop every tail is back within 3 cm of where it hung before the walk."""
        sim, chains, _ = _vrm1_sim(self.vrm["1.0"][2].json)
        rest = [c["positions"][-1] for c in chains]
        kinds = {
            kind: [k for k, c in enumerate(chains) if kind in c["name"]]
            for kind in ("long_cape", "skirt")
        }
        M, dt = np.eye(4), 1.0 / 60.0
        hang, lag, settle, before = 0.0, {"long_cape": 0.0, "skirt": 0.0}, 0.0, None
        for step in range(60 + 120 + 60):
            walking = 60 <= step < 180
            M[2, 3] += (1.25 if walking else 0.0) * dt  # the figure faces +Z
            sim.step(dt, M)
            tails = np.array([j[-1] for j in sim.joints()]) - M[:3, 3]
            if step < 60:
                hang = max(hang, float(np.linalg.norm(tails - rest, axis=1).max()))
                before = tails.copy()
            elif walking and step >= 150:  # the last half second of the walk
                for kind, rows in kinds.items():
                    d = np.linalg.norm(tails[rows] - np.array(rest)[rows], axis=1)
                    lag[kind] = max(lag[kind], float(d.max()))
            elif not walking:
                if np.linalg.norm(tails - before, axis=1).max() > 0.03:
                    settle = (step - 179) * dt
        self.assertLess(hang, 0.03)
        self.assertGreater(lag["long_cape"], 0.1)
        self.assertLess(lag["long_cape"], 0.25)
        self.assertLess(lag["skirt"], 0.25)
        self.assertLessEqual(settle, 0.5)

    def test_springs_off(self):
        path = self.folder / "still.glb"
        export_glb(
            path,
            self.card,
            self.model,
            detail_dir=self.folder,
            dressing=Dressing(springs=False),
        )
        glb = GLB(path)
        self.assertEqual(len(glb.json["skins"][0]["joints"]), self.model.bone_count)
        self.assertNotIn(
            "springs", glb.json["scenes"][0]["extras"]["opensculptboy"]["wearables"]
        )
        path = self.folder / "still.vrm"
        export_vrm(
            path,
            self.card,
            author="test",
            detail_dir=self.folder,
            refine_weights=False,
            dressing=Dressing(springs=False),
        )
        self.assertNotIn("VRMC_springBone", GLB(path).json["extensions"])
        from opensculptboy.cli import main

        card = self.folder / "card.json"
        card.write_text(json.dumps(self.card.to_dict()))
        out = self.folder / "cli.glb"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(
                    ["export", str(out), "--character", str(card), "--springs", "off"]
                ),
                0,
            )
        self.assertEqual(
            len(GLB(out).json["skins"][0]["joints"]), self.model.bone_count
        )


if __name__ == "__main__":
    unittest.main()

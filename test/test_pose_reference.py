# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the conversion between Anny's local-ref pose and the web page's pose
(``opensculptboy.posing.frames``, ``viewer/src/pose_reference.ts`` and the rest orientations of
``viewer/src/anny_shape.ts``): a pose skinned as the page skins it, after the conversion, gives
Anny's own vertices for the default body, a statue-like card (sliders, local changes, face shapes)
and random bodies; the page's turns, computed in node from the viewer data's tables, equal
Python's; and a card gains its turns as ``pose.reference``.
"""

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "viewer" / "test" / "pose_reference.mjs"
PHENOTYPES = ("gender", "age", "muscle", "weight", "height", "proportions")


def _node_available():
    if shutil.which("node") is None:
        return False
    out = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout
    return int(out.strip().lstrip("v").split(".")[0]) >= 22


def _rigid_residual(P, Q):
    """The largest distance (m) between P and Q after the best rigid map, and its turn
    (degrees)."""
    a, b = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - a).T @ (Q - b))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    res = np.linalg.norm((P - a) @ R.T + b - Q, axis=1)
    turn = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return float(res.max()), float(turn)


def _angles(A, B) -> np.ndarray:
    """The angles (degrees) of the turns between two sets of rotations (B, 3, 3), from the
    skew and the trace of A^T B, so that matrices a little off orthonormal (Anny's reference
    orientations are stored to about 3e-7) do not read as turns."""
    T = np.einsum("bji,bjk->bik", A, B)
    w = np.stack(
        [T[:, 2, 1] - T[:, 1, 2], T[:, 0, 2] - T[:, 2, 0], T[:, 1, 0] - T[:, 0, 1]], 1
    )
    return np.degrees(
        np.arctan2(np.linalg.norm(w, axis=1), np.einsum("bii->b", T) - 1.0)
    )


def _turn(rng, scale):
    import roma
    import torch

    v = rng.normal(size=3)
    v *= np.radians(scale) * rng.random() / np.linalg.norm(v)
    return roma.rotvec_to_rotmat(torch.as_tensor(v)).numpy()


def _cases():
    from opensculptboy.character import Character

    rng = np.random.default_rng(7)
    cases = [("default body", Character(), "mh_boxing")]
    cases.append(
        (
            "statue card",
            Character(
                phenotype={"gender": 0.9, "age": 0.55, "muscle": 0.8, "height": 0.7},
                local_changes={
                    "r-hand-scale-incr": 0.59,
                    "l-hand-scale-incr": 0.59,
                    "r-hand-fingers-length-incr": 0.06,
                    "r-hand-trans-out": -0.28,
                    "l-foot-trans-forward": 0.5,
                },
                face_shapes={"nose-scale-vert": 0.4, "chin-width": -0.3},
            ),
            "random",
        )
    )
    for k in range(3):
        cases.append(
            (
                f"random body {k}",
                Character(phenotype={p: float(rng.random()) for p in PHENOTYPES}),
                "random",
            )
        )
    return cases


def _pose(model, name, rng):
    if name != "random":
        import anny.poses

        return (
            anny.poses.pose_parameters(model, name)["pose_parameters"][0, :, :3, :3]
            .double()
            .numpy()
        )
    D = np.repeat(np.eye(3)[None], len(model.bone_labels), 0)
    for b in range(1, len(D)):
        D[b] = _turn(rng, 50.0)
    return D


def _anny_vertices(model, character, D):
    import torch

    P = torch.eye(4, dtype=model.template_vertices.dtype).repeat(
        1, len(model.bone_labels), 1, 1
    )
    P[0, :, :3, :3] = torch.as_tensor(D, dtype=P.dtype)
    with torch.no_grad():
        out = model(
            pose_parameters=P,
            phenotype_kwargs=dict(character.phenotype) or None,
            local_changes_kwargs=dict(character.local_changes) or None,
            face_shape_kwargs=dict(character.face_shapes) or None,
        )
    return out["vertices"][0].double().numpy()


def _coefficients(model, character):
    """Anny's coefficients of every blend-shape row of ``model`` for a card's body."""
    import torch

    def known(values, labels):
        return {k: v for k, v in (values or {}).items() if k in set(labels)} or None

    ph = model._parse_parameter_kwargs(
        known(character.phenotype, model.phenotype_labels),
        model.phenotype_labels,
        0.5,
        "p",
    )
    lc = model._parse_parameter_kwargs(
        known(character.local_changes, model.local_change_labels),
        model.local_change_labels,
        0.0,
        "l",
    )
    fa = model._parse_parameter_kwargs(None, model.facial_action_labels, 0.0, "f")
    fs = model._parse_parameter_kwargs(
        known(character.face_shapes, model.face_shape_labels),
        model.face_shape_labels,
        0.0,
        "s",
    )
    with torch.no_grad():
        return model._get_phenotype_blendshape_coefficients(ph, lc, fa, fs)[0].numpy()


class TestPoseReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from opensculptboy.fit.register import fit_model

        cls.model = fit_model()
        cls.parents = [int(p) for p in cls.model.bone_parents]
        rng = np.random.default_rng(11)
        cls.cases = [
            (name, ch, _pose(cls.model, pose, rng)) for name, ch, pose in _cases()
        ]

    def test_the_converted_pose_skins_as_anny_does(self):
        from opensculptboy.posing.frames import (
            page_skin,
            reference_turns,
            to_page,
            to_reference,
        )

        for name, character, D in self.cases:
            with self.subTest(name):
                C = reference_turns(self.model, character)
                R = to_page(D, C, self.parents)
                # C is orthonormal to about 3e-7, so the round trip holds to that
                np.testing.assert_allclose(
                    to_reference(R, C, self.parents), D, atol=1e-5
                )
                page = page_skin(self.model, character, R)
                anny = _anny_vertices(self.model, character, D)
                residual, turn = _rigid_residual(page, anny)
                self.assertLess(residual, 1e-6)
                self.assertLess(turn, 1e-6)
                # without the conversion the page's pose is off by millimetres
                wrong, _ = _rigid_residual(page_skin(self.model, character, D), anny)
                self.assertGreater(wrong, 0.003)

    @unittest.skipUnless(
        _node_available(), "node 22 or later runs the page's TypeScript"
    )
    def test_the_page_finds_the_same_turns(self):
        import anny
        import torch
        from anny.viewer import export

        from opensculptboy.posing.frames import page_skin, reference_turns, to_page

        page = anny.Anny(topology="anny-quads", face_shapes="all", phenotypes="all").to(
            dtype=torch.float64
        )
        local = anny.Anny(topology="anny-quads", local_changes="all").to(
            dtype=torch.float64
        )
        ids = np.arange(10)
        ot = export.orientation_tables(page)
        face = export.face_offsets(page, ids)["turns"]
        loc = export.local_offsets(local, ids)["turns"]
        n = int(page.stacked_phenotype_blend_shapes_mask.shape[0])
        first_face = export.face_tables(page)["first_row"]
        first_local = int(local.stacked_phenotype_blend_shapes_mask.shape[0]) + len(
            local.facial_action_labels
        )
        n_local = 2 * len(local.local_change_labels)

        def table(t):
            return {"index": t["index"].tolist(), "values": t["values"].tolist()}

        tables = {
            "template": ot["template"].reshape(-1, 9).tolist(),
            "reference": ot["reference"].reshape(-1, 9).tolist(),
            "blend": table(ot["blend"]),
            "face": table(face),
            "local": table(loc),
            "root_identity": ot["root_identity"],
            "parents": self.parents,
        }
        cases = []
        for _, character, D in self.cases:
            cp = _coefficients(page, character)
            cl = _coefficients(local, character)
            cases.append(
                {
                    "phenotype": cp[:n].tolist(),
                    "face": cp[first_face:].tolist(),
                    "local": cl[first_local : first_local + n_local].tolist(),
                    "D": D.reshape(-1).tolist(),
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            (tmp / "tables.json").write_text(json.dumps(tables))
            (tmp / "cases.json").write_text(json.dumps(cases))
            run = subprocess.run(
                ["node", str(SCRIPT), str(tmp)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            got = json.loads((tmp / "output.json").read_text())
        for (name, character, D), g in zip(self.cases, got):
            with self.subTest(name):
                C = reference_turns(self.model, character)
                Cn = np.asarray(g["C"]).reshape(-1, 3, 3)
                angle = _angles(C, Cn)
                self.assertLess(angle.max(), 1e-4)
                Rn = np.asarray(g["R"]).reshape(-1, 3, 3)
                np.testing.assert_allclose(Rn, to_page(D, C, self.parents), atol=1e-5)
                np.testing.assert_allclose(
                    np.asarray(g["D"]).reshape(-1, 3, 3), D, atol=1e-5
                )
                residual, _ = _rigid_residual(
                    page_skin(self.model, character, Rn),
                    _anny_vertices(self.model, character, D),
                )
                self.assertLess(residual, 1e-4)

    def test_a_card_gains_its_turns(self):
        import roma
        import torch

        from opensculptboy.character import Character
        from opensculptboy.posing.frames import reference_turns, with_reference

        name, character, D = self.cases[1]
        card = character.to_dict()
        card["pose"] = Character.pose_from_parameters(
            self.model,
            torch.cat(
                [
                    torch.cat([torch.as_tensor(D), torch.zeros(len(D), 3, 1)], 2),
                    torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
                    .double()
                    .repeat(len(D), 1, 1),
                ],
                1,
            ),
        )
        out = with_reference(card, self.model)
        reference = out["pose"]["reference"]
        self.assertEqual(len(reference), len(self.model.bone_labels))
        C = reference_turns(self.model, character)
        q = torch.as_tensor([reference[b] for b in self.model.bone_labels])
        np.testing.assert_allclose(roma.unitquat_to_rotmat(q).numpy(), C, atol=1e-6)
        # the card still reads, and its pose is unchanged
        fields = {k: v for k, v in out.items() if k != "look"}
        again = Character.from_dict(fields)
        self.assertEqual(again.pose["bones"], card["pose"]["bones"])
        self.assertNotIn("reference", with_reference({"pose": {}}, self.model)["pose"])


if __name__ == "__main__":
    unittest.main()

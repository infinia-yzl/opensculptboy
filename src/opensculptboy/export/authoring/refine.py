# OpenSculptBoy
# Apache License, Version 2.0
"""
The refinement of the VRM skin weights for each body, which the VRM export runs by default.

    python -m opensculptboy.export.authoring.refine --edits FILE   # the gate report

The edits of :mod:`opensculptboy.export.authoring.skin_weights` (the data file
``data/vrm/skin_weights.safetensors``) give each of 1124 vertices in and around the folds of
the T-pose logits quadratic in the six default phenotypes, so one data file serves every
body; alone they fail the hand gates of 36 of 82 bodies. :func:`body_skin_weights`, which
``opensculptboy.export.vrm.vrm_spec`` calls for the inverse bind, refines them for the body
of the file with a :class:`Refiner`, which specialises the loss of that module to one body:

- the start is the data file's weights for the body's phenotype; each row may take the 4
  bones of the file and, with ``candidates`` above 4, its own and its neighbours' bones of the
  model's weights (they start 6 below the weakest logit of the row, and a last phase keeps the
  strongest 4). When the start already passes the check below, the file takes it as it is;
- the terms are those of ``skin_weights.Loss`` for this body alone: the folds and collapsed
  triangles of its T-posed bind, its hand poses against its own posed Anny mesh, the mean
  distances of both binds over the library frames, and the caps of every library entry;
- the caps of the hand poses and of the library may stand a margin above those of the model's
  weights (the gates allow 10 % on the hand distances and the worst library distance, and 5 %
  on ``bind_error``);
- the check: the T-pose of the weights cut to 4 bones has no new fold and no collapsed
  triangle, and the hand poses pass their gates on the body itself (the 5 hand poses and the
  2 finger curls: no more folds and collapsed triangles than with the model's weights, and the
  largest and the 99th-percentile distance of the hand within 10 %). It runs every 50 steps
  and ends the refinement early when it passes;
- after the steps, a hand pose that fails its gate weighs ``repair_factor`` times more in the
  fold term of the hand poses, its caps tighten by 5 %, and ``repair_steps`` more steps
  follow, at most ``repairs`` times;
- :meth:`Refiner.safe_weights` then takes the refined weights or, if the hand gates still
  fail, the weights refined in the same way from the model's own weights (in the same rows
  and bones), or the first blend toward the data file's weights and then toward the model's
  weights that passes them.

The steps run in float64 on :data:`THREADS` threads from inputs that the body fixes, so the
same body gives the same weights, bit for bit, on the same machine with the same build of
torch. A CPU with other vector instructions (AVX2 against AVX-512, for example) rounds the sums
of the steps differently and can give other weights for the same body: up to 0.2 of weight on
a few dozen rows of an extreme body.

Measured in the gate report of :func:`main` with the defaults below, for VRM 1.0 and 0.x
(with the model's weights: 1293 new folds and 274 collapsed triangles over the default body,
the fixture body and the 64 corners of the six default phenotypes; with the data file alone:
266 and 1):

- the T-pose: 232 and 266 new folds over the 66 bodies and 8 collapsed triangles; none on the
  default and fixture bodies; 37 and 35 over 16 random bodies (233 with the model's weights).
  Of the 232 folds of VRM 1.0, 134 lie at the webs of the fingers and 90 at the groin. The
  plan's gate asked for 65 at most; the user chose to ship this trade-off;
- every other gate holds: the 35 hand cases (the largest distance of the hand within 1.098
  times the model's, the 99th percentile within 1.085, and 201 fewer folds and 31 fewer
  collapsed triangles in all), ``bind_error`` within 1.009, ``humanoid_error`` within 1.000,
  and the worst distance of every library pose and clip within 1.007; the hand gates hold on
  all 82 bodies;
- the time on 4 CPUs: 8.3 and 7.9 s per body at the median; of the 84 bodies of each
  version, 11 take the data file's weights as they are (0.4 s), 7 and 13 need the second
  start (about 20 s), and 3 keep the model's weights. A VRM export of the default character takes 1.5 to 2.2 s with the
  refinement (0.4 to 0.6 s without), the fixture body of the tests 4 to 6 s, corner 010100
  11 s, and the slowest bodies 19 to 23 s.

Without the posing terms, the refinement opens all but 11 of the 192 folds of the six hardest
bodies, and then every hand pose fails its gate on them: unfolding a finger web moves its
vertices in the bind, and the same vertices then land further from Anny in a fist. Raising
the weight of the groin edges, of the peace sign or of the finger curls, and moving the floor
of the bends from 90 to 105 degrees, did not help. Two departures from the VRM T-pose, measured
and left out, do not help either (VRM 1.0, the 66 bodies): fingers that keep 1, 2 or 3 degrees
of their rest spread fold 1203, 1175 and 1117 edges with the model's weights and 288, 266 and
258 with the refinement; legs that turn in only until the thighs meet, on the 21 bodies whose
groin folds, fold 1252 and 234.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time

import numpy as np
import torch

from opensculptboy.export import mesh_checks, tpose, vrm_tables
from opensculptboy.export.body import triangulated_faces
from opensculptboy.export.authoring import skin_weights as sw


@dataclasses.dataclass
class RefineSettings:
    """
    The settings of :meth:`Refiner.refine`; the defaults are those of the best trade-off
    measured (see the module docstring). The weights of the terms are those of
    ``skin_weights.Settings``.
    """

    steps: int = 150
    lr: float = 0.05
    fold_weight: float = 1.0
    area_weight: float = 1e5
    hand_fold_weight: float = 1.0
    worst_weight: float = 1.0
    cap_weight: float = 3000.0
    mean_weight: float = 1.0
    forward_weight: float = 0.5
    library_weight: float = 3000.0
    change_weight: float = 10.0
    floor: float = 90.0
    hand_floor: float = 118.0
    area_floor: float = 0.4
    # The library pairs (frame, row) whose distances reach ``active_ratio`` of their caps,
    # picked again every ``refresh`` steps. A vertex may pass its cap between two picks, so
    # one body needs them more often than the static optimisation (every 50 steps at 0.8
    # let a vertex of the heavy body move from 14 to 21 mm in mh_hero).
    active_ratio: float = 0.6
    refresh: int = 10
    # The caps of the hand poses (the largest and the 99th-percentile distance of the hand)
    # and of the library entries (the worst distance, and the 99th percentile of relaxed and
    # walk) as shares of those of the model's weights.
    hand_max_margin: float = 1.05
    hand_p99_margin: float = 1.05
    library_margin: float = 1.05
    library_p99_margin: float = 1.02
    # The candidate slots of each row (4: the bones of the data file), and the rings of new
    # rows around the folds of the body with the warm start (0: none).
    candidates: int = 8
    rings: int = 0
    # With more than 4 candidates: the share of the steps after which the weight beyond the
    # strongest 4 of a row costs, its weight, and the steps of the last phase on those 4.
    sparse_from: float = 0.5
    sparse_weight: float = 100.0
    final_steps: int = 50
    # Factors on the fold term of the T-pose per part of the body ("spine" holds the groin,
    # "fingers", "thumb", "neck", ...) and on the fold term of each hand pose.
    part_weights: dict = dataclasses.field(default_factory=dict)
    pose_weights: dict = dataclasses.field(default_factory=dict)
    # The repairs after the check of the hand gates (see the module docstring).
    repairs: int = 2
    repair_steps: int = 50
    repair_factor: float = 30.0
    # An early stop: every ``check_every`` steps of the first phase (0: never), the weights
    # cut to 4 bones whose T-pose has no new fold and no collapsed triangle and whose hand
    # poses pass their gates end the refinement. ``check_start``: the data file's weights for
    # the body are taken as they are when they pass the same check.
    check_every: int = 50
    check_start: bool = True
    # The steps of the first phase and the repairs of the second start, from the model's
    # own weights (see :meth:`Refiner.safe_weights`).
    second_steps: int = 150
    second_repairs: int = 2

    def loss_settings(self) -> sw.Settings:
        names = {f.name for f in dataclasses.fields(sw.Settings)}
        return sw.Settings(
            **{k: v for k, v in dataclasses.asdict(self).items() if k in names}
        )


def region_of(mesh: sw.Mesh, rows, candidates, valid, start) -> sw.Region:
    """The :class:`~opensculptboy.export.authoring.skin_weights.Region` of given rows."""
    V = mesh.vertex_count
    in_region = np.zeros(V, dtype=bool)
    in_region[rows] = True
    touched = in_region[mesh.triangles].any(axis=1)
    pair_ids = np.flatnonzero(touched[mesh.pairs].any(axis=1))
    triangle_ids = np.unique(mesh.pairs[pair_ids])
    local = np.unique(mesh.triangles[triangle_ids])
    number = np.full(V, -1)
    number[local] = np.arange(len(local))
    triangle_number = np.full(len(mesh.triangles), -1)
    triangle_number[triangle_ids] = np.arange(len(triangle_ids))
    return sw.Region(
        rows=rows,
        candidates=candidates,
        valid=valid,
        start=start,
        local=local,
        place=number[rows],
        triangles=number[mesh.triangles[triangle_ids]],
        triangle_ids=triangle_ids,
        pairs=triangle_number[mesh.pairs[pair_ids]],
        pair_ids=pair_ids,
    )


def unique_slots(bones: np.ndarray, logits: np.ndarray):
    """
    Bones (E, S) and logits (E, S) with the slots that repeat a bone folded into its first
    slot (the log of the summed exponentials); -1 marks an empty slot.
    """
    bones, logits = bones.copy(), logits.copy()
    for r in np.flatnonzero([len(set(b[b >= 0])) < (b >= 0).sum() for b in bones]):
        seen = {}
        for k in range(bones.shape[1]):
            b = bones[r, k]
            if b < 0:
                continue
            if b in seen:
                logits[r, seen[b]] = np.logaddexp(logits[r, seen[b]], logits[r, k])
                bones[r, k] = -1
            else:
                seen[b] = k
    return bones, logits


def dense_skin(
    dense: torch.Tensor, vertices: torch.Tensor, transforms: np.ndarray, chunk: int = 32
) -> np.ndarray:
    """
    Linear blend skinning (F, V, 3) of vertices (V, 3) with dense weights (V, J) and bone
    transforms (F, J, 4, 4): the blended matrices are one matrix product per chunk of frames.
    """
    M = torch.from_numpy(np.ascontiguousarray(transforms[:, :, :3, :]))
    out = []
    for s in range(0, len(M), chunk):
        m = M[s : s + chunk]
        f = m.shape[0]
        A = dense @ m.permute(1, 0, 2, 3).reshape(m.shape[1], -1)  # (V, f * 12)
        A = A.reshape(-1, f, 3, 4).permute(1, 0, 2, 3)  # (f, V, 3, 4)
        out.append((A[..., :3] @ vertices[..., None])[..., 0] + A[..., 3])
    return torch.cat(out).numpy()


class Refiner:
    """
    The refinement of the skin-weight edits ``edits`` (``{twist: (vertices, bones, logits)}``,
    as ``skin_weights.load_edits`` reads them) for the bodies of one model and twist mode.
    """

    def __init__(
        self,
        model,
        twist: str,
        edits: dict,
        mesh: sw.Mesh | None = None,
        start: str = "file",
    ):
        import anny.poses

        if start not in ("file", "today"):
            raise ValueError(f"Unknown start {start!r}; use 'file' or 'today'.")
        self.model = model
        self.twist = twist
        self.mesh = mesh or sw.Mesh(model)
        self.edits = edits[twist]
        # The start of :meth:`safe_weights`: the data file's weights for the body, or the
        # model's own weights in the rows and candidate bones of the data file.
        self.start = start
        J = len(model.bone_labels)
        raw = vrm_tables.vrm_skin_weights(model, twist, edits=False)
        self.today = tpose._strongest_four(*raw)
        self.hand = sw.hand_mask(model, sw.dense_weights(*raw, J))
        self.edge_part, _ = sw.region_names(model, self.mesh.triangles, self.mesh.pairs)
        self.today_dense = sw.dense_weights(*self.today, J)
        self.anny_dense = torch.from_numpy(
            sw.dense_weights(*sw._anny_weights(model), J)
        )
        self.file_dense = torch.from_numpy(self.today_dense)
        # The library frames of skin_weights.FRAMES in one batch, and their entries.
        lib = anny.poses.library()
        rotations, roots, entries = [], [], []
        for name, step in sw.FRAMES:
            for entry in anny.poses.names("pose") if name is None else [name]:
                r, o = lib.frames(entry)
                rotations.append(r[::step])
                roots.append(o[::step])
                entries += [entry] * len(r[::step])
        self.rotations, self.roots = torch.cat(rotations), torch.cat(roots)
        self.entries = np.array(entries)
        self.hand_rotations = torch.cat([lib.frames(n)[0] for n in sw.HAND_POSES])
        self.hand_roots = torch.cat([lib.frames(n)[1] for n in sw.HAND_POSES])
        self.library_bones = lib.bones
        self._hand_key = None

    def bone_poses(self, body: sw.Body, rotations, roots) -> np.ndarray:
        """Anny's world bone poses (F, J, 4, 4) of library rotations for a body."""
        import anny.poses

        dtype = self.model.template_vertices.dtype
        B = torch.from_numpy(body.rest_bone_poses).to(dtype)
        rest = dict(
            rest_bone_poses=B[None],
            rest_bone_heads=B[None, :, :3, 3],
            rest_vertices=torch.from_numpy(body.rest).to(dtype)[None],
        )
        with torch.no_grad():
            params = anny.poses.to_local_ref(
                self.model, rotations, roots, self.library_bones, rest_output=rest
            )
            _, W = self.model.get_bone_transforms(
                params, B[None], pose_parameterization="local-ref"
            )
        return W.double().cpu().numpy()

    def hand_poses(self, body: sw.Body) -> np.ndarray:
        """Anny's world bone poses (H, J, 4, 4) of the hand poses and the finger curls."""
        library = self.bone_poses(body, self.hand_rotations, self.hand_roots)
        curls = [
            sw.curl_bone_poses(self.model, body, a, t) for a, t in sw.CURLS.values()
        ]
        return np.concatenate([library, *curls])

    def terms(self, body: sw.Body, region: sw.Region, settings: RefineSettings):
        """The terms of the loss of one body (those of ``skin_weights.body_terms``)."""
        st = settings
        rows, local, K = region.rows, region.local, region.candidates.shape[1]
        bind = sw.inverse_bind(body, *self.today)
        rest_t, bind_t = torch.from_numpy(body.rest), torch.from_numpy(bind)
        unpose = tpose.rigid_inverse(body.rest_bone_poses)
        untee = tpose.rigid_inverse(body.t_pose)
        triangles, pairs = self.mesh.triangles, self.mesh.pairs
        terms = dict(
            rest=body.rest[rows],
            unbind=body.unbind,
            bind=bind[local],
            rest_bend=mesh_checks.bends(body.rest, triangles, pairs)[region.pair_ids],
            rest_area=mesh_checks.areas(body.rest, triangles[region.triangle_ids]),
        )
        # The hand poses.
        W = self.hand_poses(body)
        anny = dense_skin(self.anny_dense, rest_t, W @ unpose)
        file = dense_skin(self.file_dense, bind_t, W @ untee)
        error = np.linalg.norm(file - anny, axis=-1) * 1000.0
        hand = self.hand
        cap_hand = np.minimum(
            st.hand_max_margin * error[:, hand].max(axis=1, keepdims=True),
            np.maximum(
                st.hand_p99_margin
                * np.percentile(error[:, hand], 99.0, axis=1, keepdims=True),
                error[:, rows],
            ),
        )
        cap_all = st.hand_max_margin * error.max(axis=1, keepdims=True)
        terms.update(
            hand_transforms=W @ untee,
            hand_file=file[:, local],
            hand_anny=anny[:, rows],
            hand_bend=mesh_checks.bends(anny[:, local], region.triangles, region.pairs),
            hand_area=mesh_checks.areas(anny[:, local], region.triangles),
            hand_cap=np.where(hand[rows][None], cap_hand, cap_all),
        )
        # The library frames: the caps over the whole mesh of each entry, and the mean
        # distances of both binds as quadratic forms (see skin_weights.body_terms).
        W = self.bone_poses(body, self.rotations, self.roots)
        anny = dense_skin(self.anny_dense, rest_t, W @ unpose)
        Y = W @ untee
        forward = sw.forward_bind(self.model, body)
        file = dense_skin(self.file_dense, bind_t, Y)
        error = np.linalg.norm(file - anny, axis=-1) * 1000.0  # (F, V)
        cap = np.full((len(W), len(rows)), np.inf)
        forward_cap = np.full((len(W), len(rows)), np.inf)
        for entry in np.unique(self.entries):
            f = np.flatnonzero(self.entries == entry)
            cap[f] = st.library_margin * error[f].max()
            if entry in sw.P99_ENTRIES:
                cap[f] = np.minimum(
                    cap[f],
                    np.maximum(
                        st.library_p99_margin * np.percentile(error[f], 99.0),
                        error[f][:, rows],
                    ),
                )
                moved = dense_skin(self.file_dense, torch.from_numpy(forward), Y[f])
                e = np.linalg.norm(moved - anny[f], axis=-1) * 1000.0
                forward_cap[f] = np.minimum(
                    st.library_margin * e.max(),
                    np.maximum(
                        st.library_p99_margin * np.percentile(e, 99.0), e[:, rows]
                    ),
                )
        a = torch.from_numpy(anny[:, rows])
        Yt = torch.from_numpy(Y[:, :, :3, :])
        cand = torch.from_numpy(region.candidates)
        fh = torch.from_numpy(np.append(forward[rows], np.ones((len(rows), 1)), axis=1))
        R = len(rows)
        H = torch.zeros(R, 4 * K, 4 * K, dtype=torch.float64)
        g = torch.zeros(R, 4 * K, dtype=torch.float64)
        s = torch.zeros(R, dtype=torch.float64)
        Hf = torch.zeros(R, K, K, dtype=torch.float64)
        gf = torch.zeros(R, K, dtype=torch.float64)
        for start in range(0, len(Yt), 32):
            G = Yt[start : start + 32][:, cand]  # (f, R, K, 3, 4)
            Gf = torch.einsum("frkab,rb->frka", G, fh).permute(1, 0, 3, 2)
            Gf = Gf.reshape(R, -1, K)
            G = G.permute(0, 1, 3, 2, 4).reshape(G.shape[0], R, 3, 4 * K)
            G = G.permute(1, 0, 2, 3).reshape(R, -1, 4 * K)
            A = a[start : start + 32].permute(1, 0, 2).reshape(R, -1)
            H += G.transpose(1, 2) @ G
            g += (G.transpose(1, 2) @ A[..., None])[..., 0]
            s += (A**2).sum(dim=1)
            Hf += Gf.transpose(1, 2) @ Gf
            gf += (Gf.transpose(1, 2) @ A[..., None])[..., 0]
        terms.update(
            library_Y=Y[:, :, :3, :],
            library_anny=anny[:, rows],
            library_cap=cap,
            library_fwd_cap=forward_cap,
            forward=forward[rows],
        )
        result = {
            k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in terms.items()
        }
        count = len(Yt)
        result.update(quad_H=H / count, quad_g=g / count, quad_s=s / count)
        result.update(fwd_H=Hf / count, fwd_g=gf / count, fwd_s=s / count)
        return result

    def widen(self, rows, bones, warm, count, extra_rows=()):
        """
        Each row widened to ``count`` candidate slots: its bones of the data file first, then
        its own and its neighbours' bones of the model's weights (scored as in
        ``skin_weights.fold_region``), at a logit 6 below the weakest of the row; and the
        ``extra_rows`` with the model's weights.
        """
        today = self.today_dense
        held_rows = set(rows.tolist())
        new_rows = [r for r in extra_rows if r not in held_rows]
        all_rows = np.concatenate([rows, np.array(new_rows, dtype=rows.dtype)])
        cand = np.full((len(all_rows), count), -1, dtype=np.int64)
        logit = np.zeros((len(all_rows), count))
        for i, v in enumerate(all_rows):
            if i < len(rows):
                held = [b for b in bones[i] if b >= 0]
                base = [warm[i, k] for k in range(bones.shape[1]) if bones[i, k] >= 0]
            else:
                held = [int(b) for b in np.flatnonzero(today[v] > 0)]
                base = [float(np.log(today[v, b])) for b in held]
            score = 10.0 * today[v] + today[self.mesh.neighbours[v]].sum(axis=0)
            order = [int(j) for j in np.argsort(-score, kind="stable") if score[j] > 0]
            slots = (held + [j for j in order if j not in held])[
                : max(count, len(held))
            ]
            cand[i, : len(slots)] = slots
            low = min(base) - 6.0
            logit[i, : len(slots)] = base + [low] * (len(slots) - len(base))
        return all_rows, cand, logit

    def warm_weights(self, body: sw.Body):
        """The file's 4 weights (V, 4) of a body with the data file's edits alone."""
        raw = vrm_tables.vrm_skin_weights(
            self.model,
            self.twist,
            edits={self.twist: self.edits},
            phenotype=body.phenotype,
        )
        return tpose._strongest_four(*raw)

    def weights_of(self, edits) -> tuple[np.ndarray, np.ndarray]:
        """The file's 4 weights (V, 4) of refined rows (vertices, bones, weights)."""
        raw = vrm_tables.vrm_skin_weights(
            self.model, self.twist, edits={self.twist: edits}
        )
        return tpose._strongest_four(*raw)

    def refine(
        self,
        body: sw.Body,
        settings: RefineSettings | None = None,
        log=None,
        from_today: bool = False,
    ):
        """
        The refined rows (vertices (E,), bones (E, 4), weights (E, 4)) of a body, from the
        data file's weights, or from the model's own weights with ``from_today``.
        """
        settings = settings or RefineSettings()
        started = time.perf_counter()
        rows, bones, logits = self.edits
        features = vrm_tables.edit_features(body.phenotype, logits.shape[-1])
        warm = logits.astype(np.float64) @ features  # (E, 4)
        bones, warm = unique_slots(bones.astype(np.int64), warm)
        rows = rows.astype(np.int64)
        extra = ()
        if settings.rings:
            new, small = sw.tpose_problems(
                self.mesh, body, sw.inverse_bind(body, *self.warm_weights(body))
            )
            seeds = np.zeros(self.mesh.vertex_count, dtype=bool)
            seeds[self.mesh.triangles[self.mesh.pairs[new]].ravel()] = True
            seeds[self.mesh.triangles[small].ravel()] = True
            extra = np.flatnonzero(self.mesh.dilate(seeds, settings.rings))
        if settings.candidates > 4 or len(extra):
            rows, bones, warm = self.widen(
                rows, bones, warm, settings.candidates, extra
            )
        valid = bones >= 0
        cand = np.maximum(bones, 0)
        start = np.where(valid, self.today_dense[rows[:, None], cand], 0.0)
        if from_today:
            # The model's own weights as the start, their slots at 1e-4 at least.
            warm = np.log(np.clip(start, 1e-4, None))
        region = region_of(self.mesh, rows, cand, valid, start)
        terms = self.terms(body, region, settings)
        if log:
            log(f"terms in {time.perf_counter() - started:.2f} s")
        started = time.perf_counter()

        loss = sw.Loss(region, settings.loss_settings())
        batch = sw.stack_terms(region, {body.name: terms})
        names = list(sw.HAND_POSES) + list(sw.CURLS)
        if settings.part_weights:
            parts = self.edge_part[region.pair_ids]
            fold_w = np.ones(len(parts))
            for part, factor in settings.part_weights.items():
                fold_w[parts == part] = factor
            batch["fold_w"] = torch.from_numpy(fold_w)[None]
        if settings.pose_weights or settings.repairs:
            batch["hand_pose_w"] = torch.tensor(
                [settings.pose_weights.get(n, 1.0) for n in names], dtype=torch.float64
            )
        mask = torch.from_numpy(valid)
        negative = torch.tensor(-1e9, dtype=torch.float64)
        logits = torch.from_numpy(np.where(valid, warm, 0.0)).requires_grad_(True)

        def weights():
            return torch.softmax(torch.where(mask, logits, negative), dim=-1)

        wide = bool(valid.sum(axis=1).max() > 4)

        def rows_now():
            with torch.no_grad():
                w = weights().numpy()
            if wide:
                w = sw._keep_four(np.where(mask.numpy(), w, 0.0))
            order = np.argsort(-w, axis=1, kind="stable")[:, :4]
            values = np.take_along_axis(w, order, axis=1)
            held = np.take_along_axis(region.candidates, order, axis=1)
            bones = np.where(values > 0, held, -1)
            return region.rows, bones, np.where(bones >= 0, values, 0.0)

        phases = [settings.steps] + ([settings.final_steps] if wide else [])
        phases += [None] * settings.repairs
        for phase, steps in enumerate(phases):
            if phase == 1 and wide:
                with torch.no_grad():
                    mask = mask & torch.from_numpy(sw._top_four(weights().numpy()))
            if steps is None:
                failing = ~self.hand_passes(body, *self.weights_of(rows_now()))
                if not failing.any():
                    break
                if log:
                    log(f"  repair {[n for n, x in zip(names, failing) if x]}")
                factor = np.where(failing, settings.repair_factor, 1.0)
                batch["hand_pose_w"] = batch["hand_pose_w"] * torch.from_numpy(factor)
                tight = torch.from_numpy(np.where(failing, 0.95, 1.0))
                batch["hand_cap"] = batch["hand_cap"] * tight[None, :, None]
                steps = settings.repair_steps
            optimiser = torch.optim.Adam([logits], lr=settings.lr)
            schedule = torch.optim.lr_scheduler.LambdaLR(
                optimiser,
                lambda k, n=steps: 0.1 + 0.45 * (1.0 + np.cos(np.pi * k / max(n, 1))),
            )
            for step in range(steps):
                if (
                    phase == 0
                    and settings.check_every
                    and step
                    and step % settings.check_every == 0
                ):
                    current = rows_now()
                    if self.passes(body, *self.weights_of(current)):
                        if log:
                            log(f"  early stop at step {step}")
                        return current
                if step % settings.refresh == 0:
                    with torch.no_grad():
                        sw.active_pairs(
                            loss, weights()[None], batch, settings.active_ratio
                        )
                optimiser.zero_grad()
                w = weights()
                parts = loss(w[None], batch)
                if wide and phase == 0 and step >= settings.sparse_from * steps:
                    with torch.no_grad():
                        kept = torch.zeros_like(mask)
                        kept.scatter_(1, w.topk(4, dim=1).indices, True)
                    beyond = torch.where(kept, 0.0, w).sum(dim=-1)
                    parts["sparse"] = settings.sparse_weight * (beyond**2).sum()
                sum(parts.values()).backward()
                optimiser.step()
                schedule.step()
        if log:
            log(f"steps in {time.perf_counter() - started:.2f} s")
        return rows_now()

    # The check of the hand gates.

    def _hand_context(self, body: sw.Body) -> dict:
        # Keyed on the content of the body, so that another body can never take the context of
        # an earlier one (a new Body object may reuse the address of a freed one).
        key = _body_digest(body)
        if self._hand_key != key:
            W = self.hand_poses(body)
            anny = dense_skin(
                self.anny_dense,
                torch.from_numpy(body.rest),
                W @ tpose.rigid_inverse(body.rest_bone_poses),
            )
            triangles, pairs = self.mesh.triangles, self.mesh.pairs
            self._hand = dict(
                Y=W @ tpose.rigid_inverse(body.t_pose),
                anny=anny,
                bend=mesh_checks.bends(anny, triangles, pairs),
                area=mesh_checks.areas(anny, triangles),
            )
            self._hand_key = key
            self._hand["today"] = self.hand_measures(body, *self.today)
        return self._hand

    def hand_measures(self, body: sw.Body, weights, indices) -> dict:
        """
        Per hand pose (H,): the largest and the 99th-percentile distance (mm) of the hand
        from Anny's posed mesh, and the new folds and collapsed triangles of the whole mesh,
        as ``skin_weights.hand_report`` measures them.
        """
        context = self._hand_context(body)
        bind = sw.inverse_bind(body, weights, indices)
        dense = sw.dense_weights(weights, indices, len(self.model.bone_labels))
        file = dense_skin(torch.from_numpy(dense), torch.from_numpy(bind), context["Y"])
        triangles, pairs = self.mesh.triangles, self.mesh.pairs
        d = np.linalg.norm(file - context["anny"], axis=-1)[:, self.hand] * 1000.0
        new = mesh_checks.new_folds(
            mesh_checks.bends(file, triangles, pairs), context["bend"]
        )
        small = mesh_checks.collapsed(
            mesh_checks.areas(file, triangles), context["area"]
        )
        return dict(
            max=d.max(axis=1),
            p99=np.percentile(d, 99.0, axis=1),
            folds=new.sum(axis=1),
            collapsed=small.sum(axis=1),
        )

    def passes(self, body: sw.Body, weights, indices) -> bool:
        """
        Whether the file's 4 weights of a body give a T-pose without new folds and collapsed
        triangles and hand poses that pass their gates.
        """
        new, small = sw.tpose_problems(
            self.mesh, body, sw.inverse_bind(body, weights, indices)
        )
        if new.any() or small.any():
            return False
        return bool(self.hand_passes(body, weights, indices).all())

    def hand_passes(self, body: sw.Body, weights, indices) -> np.ndarray:
        """Whether each hand pose (H,) passes its hand gate against the model's weights."""
        before = self._hand_context(body)["today"]
        after = self.hand_measures(body, weights, indices)
        return (
            (after["folds"] <= before["folds"])
            & (after["collapsed"] <= before["collapsed"])
            & (after["max"] <= 1.1 * before["max"])
            & (after["p99"] <= 1.1 * before["p99"])
        )

    def safe_weights(
        self,
        body: sw.Body,
        settings: RefineSettings | None = None,
        log=None,
        shares=(1.0, 0.75, 0.5, 0.25),
        second_start: bool = True,
    ):
        """
        The file's 4 weights (V, 4) of a body and how they were chosen: the refined weights,
        or with ``second_start`` the weights refined from the model's own weights, or the
        first blend toward the warm start, then toward the model's weights, whose hand poses
        pass the hand gates (``(first, second, share)``).
        """
        settings = settings or RefineSettings()
        J = len(self.model.bone_labels)
        from_today = self.start == "today"
        if settings.check_start and not from_today:
            warm = self.warm_weights(body)
            if self.passes(body, *warm):
                return warm, ("warm", "warm", 1.0)
        dense = {
            "refined": sw.dense_weights(
                *self.weights_of(self.refine(body, settings, log, from_today)), J
            ),
            "warm": self.today_dense
            if from_today
            else sw.dense_weights(*self.warm_weights(body), J),
            "today": self.today_dense,
        }
        chain = [("refined", "warm", a) for a in shares]
        if second_start:
            chain.insert(1, ("from today", "from today", 1.0))
        chain += [("warm", "today", a) for a in shares] + [("today", "today", 1.0)]
        for first, second, a in chain:
            if first == "from today" and first not in dense:
                again = dataclasses.replace(
                    settings,
                    steps=settings.second_steps,
                    repairs=settings.second_repairs,
                )
                rows = self.refine(body, again, log, from_today=True)
                dense[first] = sw.dense_weights(*self.weights_of(rows), J)
            d = a * dense[first] + (1.0 - a) * dense[second]
            order = np.argsort(-d, axis=1, kind="stable")[:, :4]
            w = np.take_along_axis(d, order, axis=1)
            w = w / w.sum(axis=1, keepdims=True)
            i = np.where(w > 0, order, 0)
            if self.hand_passes(body, w, i).all():
                return (w, i), (first, second, a)
        raise AssertionError("The model's weights pass their own hand gates.")


# The number of threads of the refinement, set for its duration and restored afterwards, so
# that its sums do not depend on the caller's setting.
THREADS = 4
# The names of the sources of safe_weights in the outcome of body_skin_weights.
_SOURCES = {
    "refined": "refined",
    "from today": "refined from the model's weights",
    "warm": "data file",
    "today": "model's weights",
}
_REFINERS: dict = {}
_WEIGHTS: dict = {}


def _digest(*arrays) -> str:
    """The SHA-256 of arrays: their types, shapes and bytes."""
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(f"{array.dtype}{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _body_digest(body: sw.Body) -> str:
    """The digest of what a body fixes: its phenotype, rest mesh, rest bone poses and T-pose."""
    phenotype = json.dumps(sorted((k, float(v)) for k, v in body.phenotype.items()))
    return phenotype + _digest(body.rest, body.rest_bone_poses, body.t_pose)


def _refiner(model, twist: str, edits: dict) -> Refiner:
    """
    The :class:`Refiner` of a model, twist mode and data file, kept for the next body of the
    same model. The refiners of the most recent model alone are kept (one per twist mode and
    data file), since a Refiner holds its model and about 400 MB of tables, and an export
    that builds its own model never meets that model again.
    """
    key = (id(model), twist, _digest(*edits[twist]))
    held = _REFINERS.get(key)
    if held is None or held[0] is not model:
        for other in [k for k, (m, _) in _REFINERS.items() if m is not model]:
            del _REFINERS[other]
        held = _REFINERS[key] = (model, Refiner(model, twist, edits))
    return held[1]


def outcome_name(choice) -> str:
    """The outcome of :meth:`Refiner.safe_weights` as text, e.g. "refined"."""
    first, second, share = choice
    if share == 1.0 or first == second:
        return _SOURCES[first]
    return f"{share:g} {_SOURCES[first]} + {1.0 - share:g} {_SOURCES[second]}"


def body_skin_weights(
    model,
    twist: str,
    rest_vertices: np.ndarray,
    rest_bone_poses: np.ndarray,
    phenotype=None,
    keep_leg_spread: bool = False,
    refine: bool = True,
    settings: RefineSettings | None = None,
):
    """
    The skin weights and bone indices of the VRM file of a body for the inverse bind, before
    the file's truncation to 4 bones, and the outcome of their refinement as text.

    With ``refine`` (the default of the VRM export), the rows of the data file
    (``vrm_tables.skin_weight_data``) start :meth:`Refiner.safe_weights` for the body: its
    rest vertices (V, 3) and rest bone poses (J, 4, 4), its phenotype values (the features of
    the data file's logits) and its T-pose (``keep_leg_spread``). The outcome names the
    weights that the file takes: "refined", "refined from the model's weights", "data file"
    (the data file's weights pass as they are), a blend such as "0.75 refined + 0.25 data
    file", or "model's weights"; these weights have 4 bones per vertex (V, 4). Without
    ``refine``, or when the data file is missing or was made for another model (with a
    warning), the weights are those of ``vrm_tables.vrm_skin_weights`` as they are (V, K):
    the model's weights with the moves of the eyelids and the twist bones, with the outcome
    "off" or "no data". Truncated as the VRM writer truncates them, they give the weights of
    a file without the refinement exactly.

    The refinement runs in float64 on :data:`THREADS` threads, from inputs that the body
    fixes, so the same body gives the same weights on the same machine (another CPU can round
    the steps differently, see the module docstring); they are kept for the rest of the
    process (the shared exports of the tests reuse them).
    """
    today = vrm_tables.vrm_skin_weights(model, twist)
    if not refine:
        return (*today, "off")
    data = vrm_tables.skin_weight_data(model, warn_missing=True)
    if data is None:
        return (*today, "no data")
    rest = np.ascontiguousarray(rest_vertices, dtype=np.float64)
    B = np.ascontiguousarray(rest_bone_poses, dtype=np.float64)
    features = vrm_tables.edit_features(phenotype, data[twist][2].shape[-1])
    # The key holds everything that the result depends on: the model's weights and topology
    # (anny and anny-quads share the digest of the data file but split their quads into
    # other triangles), the data file, the body and the settings.
    key = (
        vrm_tables.skin_weight_digest(model),
        _digest(triangulated_faces(model)[0]),
        _digest(*data[twist]),
        twist,
        _digest(rest, B, features),
        bool(keep_leg_spread),
        repr(settings),
    )
    if key not in _WEIGHTS:
        threads = torch.get_num_threads()
        torch.set_num_threads(THREADS)
        try:
            refiner = _refiner(model, twist, data)
            T = tpose.vrm_t_pose(model, torch.from_numpy(B), keep_leg_spread)
            phenotype = {
                k: float(np.asarray(torch.as_tensor(v).detach().cpu()).reshape(-1)[0])
                for k, v in dict(phenotype or {}).items()
            }
            body = sw.Body("export", phenotype, rest, B, T.double().cpu().numpy())
            (weights, indices), choice = refiner.safe_weights(body, settings)
            # the new vertices of a sculpt model follow their refined anchors
            weights, indices = vrm_tables.blend_new_rows(
                model, weights, indices, refiner.today
            )
        finally:
            torch.set_num_threads(threads)
        if len(_WEIGHTS) >= 16:
            _WEIGHTS.pop(next(iter(_WEIGHTS)))
        _WEIGHTS[key] = (weights, indices, outcome_name(choice))
    weights, indices, outcome = _WEIGHTS[key]
    return weights.copy(), indices.copy(), outcome


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--edits", required=True, help="a data file of skin_weights")
    parser.add_argument(
        "--settings",
        default="{}",
        help="other RefineSettings as JSON, e.g. '{\"steps\": 300}'",
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    settings = RefineSettings(**json.loads(args.settings))
    model = sw.authoring_model()
    mesh = sw.Mesh(model)
    edits = sw.load_edits(args.edits)
    refiners = {t: Refiner(model, t, edits, mesh) for t in vrm_tables.TWIST_MODES}
    memo, times = {}, []

    def weights(version, twist, body):
        key = (twist, body.name)
        if key not in memo:
            started = time.perf_counter()
            memo[key], choice = refiners[twist].safe_weights(body, settings)
            times.append(time.perf_counter() - started)
            print(f"  {twist}, {body.name}: {times[-1]:.1f} s, {choice}", flush=True)
        return memo[key]

    _, passed = sw.gate_report(model, weights)
    print(f"gates: {'all pass' if passed else 'some fail'}")
    print(f"refinement: {np.mean(times):.1f} s per body on average, {max(times):.1f} s")


if __name__ == "__main__":
    main()

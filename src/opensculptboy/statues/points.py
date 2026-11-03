# OpenSculptBoy
# Apache License, Version 2.0
"""
A statue's pose from reference points placed on its scan (a config's ``reference_points``).

The config maps a solve's chain (``"arm.r"``: the arm and its hand; ``"hand.l"``, ``"leg.r"``;
see :func:`opensculptboy.posing.point_solve.chain_bones`) to the points placed for it: each
point's ``name`` in the stored set (:mod:`opensculptboy.posing.reference_points`), its
``position`` in the masks' frame (the scan's, y up), and optionally ``on`` and ``weight``.
A chain's list may also hold contact pins (entries with a ``kind``): the stored point
``name`` held on a ``"point"`` (``target``) or on a ``"plane"`` (``point`` and ``normal``), in
the same frame, with an optional ``weight`` (the solve's pins,
:func:`opensculptboy.posing.point_solve.solve`).
:func:`solve_on_pose` turns the chain's bones of a fitted card so that Anny's points meet
them (:func:`opensculptboy.posing.point_solve.solve`); a posed hand's points pose that hand
in :func:`opensculptboy.statues.hands.pose_hands`, before the fit or after it as the hand
chooses, and other chains are solved after the chains' fit.
"""

from __future__ import annotations

import numpy as np


def check(reference_points: dict) -> None:
    """Raise ValueError for an unknown chain or point name."""
    from opensculptboy.posing.point_solve import load_ranges
    from opensculptboy.posing.reference_points import load_points

    names = {p["name"] for p in load_points()}
    parts = set(load_ranges()["chains"])
    for chain, entries in reference_points.items():
        part, _, side = chain.lower().partition(".")
        if part not in parts or side not in ("l", "r", ""):
            raise ValueError(
                f"Reference points need a chain such as 'arm.r'; got {chain!r}."
            )
        for e in entries:
            if e.get("name") not in names:
                raise ValueError(f"Unknown reference point {e.get('name')!r}.")
            if "kind" in e:
                needs = {"point": ("target",), "plane": ("point", "normal")}.get(
                    e["kind"]
                )
                if needs is None:
                    raise ValueError(f"A pin is a point or a plane; got {e['kind']!r}.")
                for k in needs:
                    if len(e.get(k, ())) != 3:
                        raise ValueError(f"The pin on {e['name']!r} needs its {k!r}.")
                continue
            if len(e.get("position", ())) != 3:
                raise ValueError(f"The reference point {e['name']!r} needs a position.")


def in_anny(reference_points: dict) -> dict:
    """The points with their positions in Anny's frame (from the masks' frame)."""
    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    def turned(e):
        return {
            **e,
            **{
                k: (np.asarray(e[k], np.float64) @ GLTF_TO_ANNY.T).tolist()
                for k in ("position", "target", "point", "normal")
                if k in e
            },
        }

    return {
        chain: [turned(e) for e in entries]
        for chain, entries in (reference_points or {}).items()
    }


def solve_on_pose(model, character, rotation, translation, pose, chain, entries):
    """Solve a chain's bones on Anny's pose parameters ``pose`` (1, B, 4, 4) of a card placed
    by ``rotation`` and ``translation`` so that its reference points meet ``entries`` (Anny's
    frame), and the entries that are pins to theirs. Returns the rotations (B, 3, 3) and a
    report: each point's distance (mm), each pin's (mm), the steps and the points that were
    off."""
    from opensculptboy.posing.point_solve import solve
    from opensculptboy.posing.pose_rig import PoseRig
    from opensculptboy.posing.reference_points import influences_of, load_points

    rig, R = PoseRig.from_card(model, character, rotation, translation, pose=pose)
    by_name = {p["name"]: p for p in load_points()}
    pin_entries = [e for e in entries if "kind" in e]
    entries = [e for e in entries if "kind" not in e]
    pins = []
    if pin_entries:
        pin_influences = influences_of(rig, [by_name[e["name"]] for e in pin_entries])
        for f, e in zip(pin_influences, pin_entries):
            pin = {"influence": f, "kind": e["kind"]}
            for k in ("target", "point", "normal", "weight"):
                if k in e:
                    pin[k] = e[k]
            pins.append(pin)
    stored = [by_name[e["name"]] for e in entries]
    influences = influences_of(rig, stored) if stored else []
    points = [
        {
            "influence": f,
            "target": e["position"],
            "on": bool(e.get("on", True)),
            "weight": float(e.get("weight", 1.0)),
            "finger": p["finger"],
        }
        for f, e, p in zip(influences, entries, stored)
    ]
    result = solve(rig, R, points, [chain], pins=pins)
    report = {
        "chain": chain,
        "distance_mm": {e["name"]: d for e, d in zip(entries, result.residual_mm)},
        "pins_mm": {e["name"]: d for e, d in zip(pin_entries, result.pins_mm)},
        "iterations": result.iterations,
        "off": [e["name"] for e in entries if not e.get("on", True)],
    }
    on = [d for d in result.residual_mm if d is not None]
    report["mean_mm"] = round(float(np.mean(on)), 3) if on else None
    report["max_mm"] = round(float(np.max(on)), 3) if on else None
    return result.rotations, report


def solve_card(model, character, rotation, translation, solves: dict, log=None):
    """The card with each chain of ``solves`` (chain -> entries, Anny's frame) solved to its
    points in turn, and a report per chain."""
    import torch

    from opensculptboy.character import Character
    from opensculptboy.statues.chains import _identity_pose

    character = Character.from_dict(character.to_dict())
    pose = character.pose_parameters(model)
    pose = (_identity_pose(model) if pose is None else pose).detach().clone()
    report = {}
    for chain, entries in solves.items():
        R, report[chain] = solve_on_pose(
            model, character, rotation, translation, pose, chain, entries
        )
        pose[0, :, :3, :3] = torch.as_tensor(R, dtype=pose.dtype)
        if log:
            log(
                f"points {chain}: mean {report[chain]['mean_mm']} mm, max "
                f"{report[chain]['max_mm']} mm"
            )
    character.pose = Character.pose_from_parameters(model, pose)
    return character, report

# OpenSculptBoy
# Apache License, Version 2.0
"""
A character card saved in the studio, applied to its statue's config: the points placed on the
scan become the config's ``reference_points``, and the hands posed on the card become
``posed_hands.<hand>.key``, so that the pipeline rebuilds the starter in the card's pose
(DESIGN-0013, "History, library and files").

The card is a statue starter's card (its ``detail.fit.placement`` places it on the scan) as the
page saves it:

- ``detail.fit.reference.points`` (``opensculptboy.card-points/1``): by point chain (``hand.r``,
  ``arm.r``, ``leg.l``, ``neck``), each point's ``name`` (of ``data/posing/reference_points.json``),
  ``position``, ``on`` and ``weight``, in the card's frame (``"frame": "card"``: Anny's axes of
  the card's posed body) or in the masks' frame (``"frame": "scan"``: the frame of
  ``replica.glb``, y up);
- ``detail.fit.reference.pins``: contact pins (``name``, ``kind`` ``"point"`` with ``target`` or
  ``"plane"`` with ``point`` and ``normal``, ``weight``) in the same frame;
- ``pose.bones``: each bone's ``local-ref`` turn as a unit quaternion (x, y, z, w);
- ``local_changes``: the hands' shape (``r-hand-*``, ``l-hand-*``).

:func:`apply_card` writes, in the masks' frame:

- ``reference_points``: a hand's points and its arm's (the shoulder and the elbow) pose the
  chain ``arm.<side>`` (the arm and the hand), a leg's ``leg.<side>`` and the head's ``neck``;
  a pin joins the chain of the point it names (:func:`opensculptboy.statues.points.check`
  takes both);
- ``posed_hands.hand.<side>.key``: ``bones``, the turns of every bone of the side from
  ``shoulder01`` down as rotation vectors in degrees, ``local_changes`` (the hand's), and the
  hand's ``after_fit`` (:data:`AFTER_FIT`: the right hand before the fit, the left after it, as
  David's runs chose). The rest of the hand's entry (its stone, ``clear_body``, ``pin``) stays.

The forearm's twist is shared as the posing solve shares it (:func:`share_twist`): the twist
coordinates of ``lowerarm01``, ``lowerarm02`` and the wrist (each bone's turn about its own
twist axis, in the joint frames of :func:`opensculptboy.posing.point_solve.joint_frames`) are
summed and given back by the table's shares (0.4, 0.5, 0.1), so that a turn keyed in one bone
does not kink the forearm halfway. The bones' twist axes differ a little, so the shared
twist alone moves the hand; the posing solve then turns those bones again in its own
parameters (each bone's bend and side, and the group's twist by its shares), without the
ranges, until the hand's 21 points lie where the card has them. A pose that the page's solve
made has those shares already and stays as it is.

The turns are local: the hand lands where the card shows it when the run that the config
rebuilds starts from the same fit as the card (``run --from`` the starter's run).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

FORMAT = "opensculptboy.card-points/1"
# when each hand is posed (fit_statue's per-hand after_fit)
AFTER_FIT = {"hand.r": False, "hand.l": True}
# the bones of a side that a hand's key sets, from the shoulder down
ARM_WORDS = ("shoulder01", "upperarm", "lowerarm", "wrist", "metacarpal", "finger")
# the twist groups that apply_card shares (data/posing/joint_ranges.json's "twist")
SHARED = ("forearm",)


def _solve_chain(point_chain: str) -> str:
    part, _, side = point_chain.lower().partition(".")
    if part in ("hand", "arm"):
        return f"arm.{side}"
    return point_chain.lower()


def _to_masks(placement: dict, frame: str):
    """Functions taking a point and a direction of ``frame`` to the masks' frame."""
    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    if frame == "scan":
        return (lambda p: np.asarray(p, np.float64)), (
            lambda n: np.asarray(n, np.float64)
        )
    if frame != "card":
        raise ValueError(
            f"Card points are in the frame 'card' or 'scan'; got {frame!r}."
        )
    R = np.asarray(placement["rotation"], np.float64)
    t = np.asarray(placement["translation"], np.float64)
    return (
        lambda p: (R @ np.asarray(p, np.float64) + t) @ GLTF_TO_ANNY,
        lambda n: (R @ np.asarray(n, np.float64)) @ GLTF_TO_ANNY,
    )


def card_points(card: dict) -> dict:
    """The card's points and pins as a config's ``reference_points`` (the masks' frame), by
    solve chain."""
    from opensculptboy.posing.reference_points import load_points

    reference = ((card.get("detail") or {}).get("fit") or {}).get("reference") or {}
    block = reference.get("points") or {}
    pins = reference.get("pins") or []
    if not block and not pins:
        return {}
    if block and block.get("format", FORMAT) != FORMAT:
        raise ValueError(
            f"Card points have the format {FORMAT!r}; got {block.get('format')!r}."
        )
    frame = block.get("frame", "card")
    placement = card["detail"]["fit"]["placement"]
    point, direction = _to_masks(placement, frame)
    out: dict = {}
    for chain, entries in (block.get("chains") or {}).items():
        target = _solve_chain(chain)
        for e in entries:
            entry = {
                "name": e["name"],
                "position": [round(float(x), 6) for x in point(e["position"])],
                "on": bool(e.get("on", True)),
                "weight": float(e.get("weight", 1.0)),
            }
            out.setdefault(target, []).append(entry)
    chain_of = {p["name"]: p["chain"] for p in load_points()}
    for e in pins:
        if e.get("name") not in chain_of:
            raise ValueError(
                f"A pin names an unknown reference point {e.get('name')!r}."
            )
        pin = {"name": e["name"], "kind": e["kind"]}
        if e["kind"] == "point":
            pin["target"] = [round(float(x), 6) for x in point(e["target"])]
        elif e["kind"] == "plane":
            pin["point"] = [round(float(x), 6) for x in point(e["point"])]
            n = direction(e["normal"])
            pin["normal"] = [round(float(x), 6) for x in n / np.linalg.norm(n)]
        else:
            raise ValueError(f"A pin is a point or a plane; got {e['kind']!r}.")
        if "weight" in e:
            pin["weight"] = float(e["weight"])
        out.setdefault(_solve_chain(chain_of[e["name"]]), []).append(pin)
    return out


def share_twist(rig, rotations: np.ndarray, groups=SHARED) -> np.ndarray:
    """The pose (B, 3, 3) with each named twist group's twist spread over its bones by the
    table's shares, on each side: the sum of the members' twist coordinates is kept, and each
    member's bend and side stay."""
    from opensculptboy.posing.point_solve import (
        _coordinates,
        _exp,
        joint_frames,
        load_ranges,
    )

    table = load_ranges()
    frames = joint_frames(rig, table)
    out = np.array(rotations, np.float64)
    for name in groups:
        g = table["twist"][name]
        for side in (".L", ".R"):
            members = [rig.index.get(m + side) for m in g["shares"]]
            if any(m is None or m not in frames for m in members):
                continue
            coords = [_coordinates(frames[m], out[m]) for m in members]
            total = sum(c[2] for c in coords)
            for m, c, share in zip(members, coords, g["shares"].values()):
                c = c.copy()
                c[2] = float(share) * total
                f = frames[m]
                out[m] = _exp(f["axes"] @ (f["sign"] * c))
    return out


def _keep_hand(rig, rotations, influences, places, groups, suffix):
    """The pose with the twist groups' bones of one side turned again, as the posing solve
    turns them (each bone's bend and side, and the group's twist by its shares), so that
    the hand's points come back to ``places``: the shared twist leaves the hand where the
    card has it."""
    from opensculptboy.posing.point_solve import load_ranges, solve

    table = load_ranges()
    bones = [
        rig.index[m + suffix]
        for name in groups
        for m in table["twist"][name]["shares"]
        if m + suffix in rig.index
    ]
    if not bones:
        return rotations
    targets = [{"influence": f, "target": x} for f, x in zip(influences, places)]
    result = solve(
        rig,
        rotations,
        targets,
        [],
        bones=bones,
        limits=False,
        prior=1e-6,
        iterations=100,
    )
    return result.rotations


def _side_bones(labels, side: str) -> list:
    suffix = "." + side.upper()
    return [
        i
        for i, b in enumerate(labels)
        if b.endswith(suffix) and b[: -len(suffix)].startswith(ARM_WORDS)
    ]


def hand_keys(
    model, character, hands=tuple(AFTER_FIT), groups=SHARED
) -> tuple[dict, dict]:
    """The keys of the card's hands (``{"hand.r": {"bones", "local_changes"}}``) and a report:
    the twist moved by the sharing (degrees per bone) and how far the hand's points lie from
    the card's after it (mm)."""
    import roma
    import torch

    from opensculptboy.posing.pose_rig import PoseRig
    from opensculptboy.posing.reference_points import influences_of, load_points
    from opensculptboy.statues.chains import _identity_pose

    pose = character.pose_parameters(model)
    pose = (_identity_pose(model) if pose is None else pose).detach()
    rig, R = PoseRig.from_card(model, character, np.eye(3), np.zeros(3), pose=pose)
    shared = share_twist(rig, R, groups)
    labels = list(model.bone_labels)
    points = load_points()
    keys, report = {}, {}
    for hand in hands:
        side = hand.split(".")[1]
        suffix = "." + side.upper()
        own = [p for p in points if p["chain"] == f"hand.{side}"]
        f = influences_of(rig, own)
        before = rig.to_output(rig.points(rig.transforms(R), f))
        shared = _keep_hand(rig, shared, f, before, groups, suffix)
        bones = _side_bones(labels, side)
        vectors = np.degrees(
            roma.rotmat_to_rotvec(torch.as_tensor(shared[bones])).numpy()
        )
        key = {
            "bones": {
                labels[b][: -len(suffix)]: [round(float(x), 4) for x in v]
                for b, v in zip(bones, vectors)
            },
            "local_changes": {
                k: float(v)
                for k, v in (character.local_changes or {}).items()
                if k.startswith(f"{side}-hand-")
            },
        }
        keys[hand] = key
        after = rig.to_output(rig.points(rig.transforms(shared), f))
        moved = np.linalg.norm(after - before, axis=1)
        twist = {}
        for name in groups:
            for m in load_twist_members(name):
                b = rig.index.get(m + suffix)
                if b is not None:
                    d = np.degrees(
                        roma.rotmat_to_rotvec(torch.as_tensor(R[b].T @ shared[b]))
                        .norm()
                        .item()
                    )
                    twist[m] = round(float(d), 3)
        report[hand] = {
            "bones": len(bones),
            "twist_change_deg": twist,
            "points_moved_mm": round(1000 * float(moved.max()), 3),
        }
    return keys, report


def load_twist_members(name: str) -> list:
    from opensculptboy.posing.point_solve import load_ranges

    return list(load_ranges()["twist"][name]["shares"])


def apply_card(
    card: dict,
    config: dict,
    model=None,
    hands=tuple(AFTER_FIT),
    after_fit: dict | None = None,
    groups=SHARED,
) -> tuple[dict, dict]:
    """The statue ``config`` (a dict) with the card's points, pins and hand keys (see the
    module); returns the new config and a report."""
    from opensculptboy.character import Character
    from opensculptboy.statues.points import check

    out = json.loads(json.dumps(config))
    report: dict = {}
    points = card_points(card)
    if points:
        check(points)
        merged = dict(out.get("reference_points") or {})
        merged.update(points)
        out["reference_points"] = merged
        report["reference_points"] = {
            c: {
                "points": sum("kind" not in e for e in v),
                "on": sum("kind" not in e and e["on"] for e in v),
                "pins": sum("kind" in e for e in v),
            }
            for c, v in points.items()
        }
    if hands:
        fields = {k: v for k, v in card.items() if k != "look"}
        # the card without its look block, in the schema that holds a detail layer and a
        # wardrobe
        fields["schema"] = 4
        character = Character.from_dict(fields)
        if model is None:
            from opensculptboy.fit.register import fit_model

            model = fit_model(character.topology)
        keys, report["hands"] = hand_keys(model, character, hands, groups)
        posed = dict(out.get("posed_hands") or {})
        chains = [c.lower() for c in out.get("chains") or []]
        timing = {**AFTER_FIT, **(after_fit or {})}
        for hand, key in keys.items():
            spec = dict(posed.get(hand) or {})
            spec["key"] = key
            spec["after_fit"] = bool(timing.get(hand, False))
            # the key carries the hand's shape; a shape of the spec's own would override it
            spec.pop("local_changes", None)
            posed[hand] = spec
            if hand not in chains:
                chains.append(hand)
        out["posed_hands"] = posed
        out["chains"] = chains
    return out, report


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m opensculptboy.statues apply-card",
        description="Apply a character card saved in the studio to its statue's config: "
        "the placed points become reference_points and the hand poses posed_hands keys.",
    )
    parser.add_argument(
        "card", help="the saved card (character.json, or a starter .zip)"
    )
    parser.add_argument("config", help="the statue config (JSON)")
    parser.add_argument("--out", required=True, help="the config to write")
    parser.add_argument(
        "--hands",
        default="hand.r,hand.l",
        help="the hands to key, by commas (default hand.r,hand.l; empty for none)",
    )
    parser.add_argument(
        "--after-fit",
        default="",
        metavar="HAND=BOOL,...",
        help="when each hand is posed (default hand.r=false,hand.l=true)",
    )
    parser.add_argument("--report", help="also write the report (JSON) to this file")
    args = parser.parse_args(argv)
    card = _read_card(args.card)
    config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    timing = {}
    for item in filter(None, args.after_fit.split(",")):
        hand, _, value = item.partition("=")
        timing[hand.strip()] = value.strip().lower() in ("1", "true", "yes")
    hands = tuple(h.strip() for h in args.hands.split(",") if h.strip())
    out, report = apply_card(card, config, hands=hands, after_fit=timing)
    from opensculptboy.statues.pipeline import StatueConfig

    StatueConfig.from_dict(out, args.config)
    pathlib.Path(args.out).write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    if args.report:
        pathlib.Path(args.report).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"config": args.out, **report}, indent=2))
    return 0


def _read_card(path) -> dict:
    path = pathlib.Path(path)
    if path.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(path) as z:
            return json.loads(z.read("character.json"))
    return json.loads(path.read_text(encoding="utf-8"))

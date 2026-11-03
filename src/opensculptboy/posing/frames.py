# OpenSculptBoy
# Apache License, Version 2.0
"""
Anny's local-ref pose and the web page's pose.

A card's pose (``Character.pose``) holds Anny's ``local-ref`` turns ``D``: each bone's turn
relative to its parent, measured from Anny's reference pose, in which every bone's frame takes
its reference orientation (``model.reference_bone_orientations``). That pose stands a little
apart from the rest pose: on the default body the heads move by up to 14 mm and the skin by up
to 11 mm. The web page poses from the rest pose instead, each bone turned about its rest head
along the world axes (the pose library's rotations ``R``). With each bone's turn
``C = reference rest^T`` (the rest orientation of the body's own sliders, local changes and
face shapes), the two convert as :func:`anny.poses.pose_parameters` converts the library::

    D_b = C_parent R_b C_b^T        R_b = C_parent^T D_b C_b        (C_parent = I for the root)

:func:`reference_turns` gives ``C`` for a card; :func:`to_page` and :func:`to_reference`
convert; :func:`page_skin` skins a pose as the page does, the check that the conversion gives
Anny's own vertices. A card may carry its body's turns as ``pose.reference`` (quaternions
x, y, z, w by bone, :func:`card_reference`), which the page compares with the turns it finds
for the same body (``viewer/src/pose_reference.ts``). ``python -m opensculptboy.posing.frames
card IN [--out OUT]`` adds them to a card or a starter zip.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np


def _rest_output(model, character):
    import torch

    pose = torch.eye(4, dtype=model.template_vertices.dtype).repeat(
        1, len(model.bone_labels), 1, 1
    )
    with torch.no_grad():
        return model(
            pose_parameters=pose,
            phenotype_kwargs=dict(character.phenotype) or None,
            local_changes_kwargs=dict(character.local_changes) or None,
            face_shape_kwargs=dict(character.face_shapes) or None,
        )


def reference_turns(model, character) -> np.ndarray:
    """Each bone's turn ``C = reference rest^T`` (B, 3, 3) for a card's body."""
    out = _rest_output(model, character)
    rest = out["rest_bone_poses"][0, :, :3, :3].detach().double().cpu().numpy()
    if model.reference_bone_orientations is None:
        return np.repeat(np.eye(3)[None], len(rest), 0)
    reference = model.reference_bone_orientations.detach().double().cpu().numpy()
    return reference @ np.swapaxes(rest, -1, -2)


def to_page(D: np.ndarray, C: np.ndarray, parents) -> np.ndarray:
    """The page's rotations R (B, 3, 3) of local-ref turns D."""
    out = np.empty_like(np.asarray(D, np.float64))
    for b, p in enumerate(parents):
        Cp = C[p] if p >= 0 else np.eye(3)
        out[b] = Cp.T @ D[b] @ C[b]
    return out


def to_reference(R: np.ndarray, C: np.ndarray, parents) -> np.ndarray:
    """Local-ref turns D (B, 3, 3) of the page's rotations R."""
    out = np.empty_like(np.asarray(R, np.float64))
    for b, p in enumerate(parents):
        Cp = C[p] if p >= 0 else np.eye(3)
        out[b] = Cp @ R[b] @ C[b].T
    return out


def page_skin(model, character, R: np.ndarray) -> np.ndarray:
    """The rest mesh (V, 3) skinned as the page skins it: each bone turned by R about its rest
    head along the world axes, the root at its rest head (Anny's frame, before the root's
    own placement)."""
    out = _rest_output(model, character)
    heads = out["rest_bone_poses"][0, :, :3, 3].detach().double().cpu().numpy()
    rest = out["rest_vertices"][0].detach().double().cpu().numpy()
    parents = [int(p) for p in model.bone_parents]
    B = len(parents)
    G = np.zeros((B, 4, 4))
    done = [False] * B
    while not all(done):
        for b, p in enumerate(parents):
            if done[b] or (p >= 0 and not done[p]):
                continue
            L = np.eye(4)
            L[:3, :3] = R[b]
            L[:3, 3] = heads[b] - (heads[p] if p >= 0 else 0.0)
            G[b] = (G[p] if p >= 0 else np.eye(4)) @ L
            done[b] = True
    S = G.copy()
    S[:, :3, 3] -= np.einsum("bij,bj->bi", G[:, :3, :3], heads)
    W = model.vertex_bone_weights.detach().double().cpu().numpy()
    index = model.vertex_bone_indices.cpu().numpy()
    M = np.einsum("vk,vkij->vij", W, S[index])
    return np.einsum("vij,vj->vi", M[:, :3, :3], rest) + M[:, :3, 3]


def card_reference(card: dict, model=None) -> dict:
    """The card's body's turns as ``pose.reference``: unit quaternions (x, y, z, w) by bone."""
    import roma
    import torch

    from opensculptboy.character import Character

    fields = {
        k: v
        for k, v in card.items()
        if k not in ("look", "wearables", "outfits", "active_outfit", "detail")
    }
    fields["schema"] = 4
    character = Character.from_dict(fields)
    if model is None:
        from opensculptboy.fit.register import fit_model

        model = fit_model(character.topology)
    C = reference_turns(model, character)
    q = roma.rotmat_to_unitquat(torch.as_tensor(C)).numpy()
    return {
        label: [round(float(x), 7) for x in v] for label, v in zip(model.bone_labels, q)
    }


def with_reference(card: dict, model=None) -> dict:
    """The card with ``pose.reference`` set for its body (a card without a pose gains none)."""
    out = json.loads(json.dumps(card))
    if not (out.get("pose") or {}).get("bones"):
        return out
    out["pose"]["reference"] = card_reference(out, model)
    return out


def main(argv=None) -> int:
    import argparse
    import io
    import zipfile

    parser = argparse.ArgumentParser(
        prog="python -m opensculptboy.posing.frames",
        description="Add a card's bone turns (pose.reference) for the web page's check.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    card = sub.add_parser("card", help="a card (JSON) or a starter zip")
    card.add_argument("path")
    card.add_argument("--out", help="the file to write (default: the input)")
    args = parser.parse_args(argv)
    path = pathlib.Path(args.path)
    out = pathlib.Path(args.out) if args.out else path
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            entries = {n: z.read(n) for n in z.namelist()}
        data = with_reference(json.loads(entries["character.json"]))
        entries["character.json"] = (json.dumps(data, indent=2) + "\n").encode("utf-8")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for name, blob in entries.items():
                z.writestr(name, blob)
        out.write_bytes(buffer.getvalue())
    else:
        data = with_reference(json.loads(path.read_text(encoding="utf-8")))
        out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"out": str(out), "bones": len(data.get("pose", {}).get("reference", {}))}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# OpenSculptBoy
# Apache License, Version 2.0
"""
Reference wearables: the clothing of a reference character, carried onto a fitted card.

An appearance handoff may describe what the reference character wears in an ``equipment``
section (docs/reference-fit.md, docs/wearables.md): the clothing objects, each a node of
``replica.glb`` skinned in the same bind pose as the skin, with one primitive per mesh group;
the skin each piece covers; and rules by which one piece hides groups of another. The section
lists either one set of ``objects`` or several ``outfits`` with ``pieces``. A piece gathers the
objects of one slot of the reference (its headgear, its top, its trousers, its cloak, its
underwear).

:func:`build_reference_wearables` turns every piece into a wearable package:

1. each clothing vertex finds its closest point on the reference's skin, and that point its
   closest point on the fitted card's skin, posed and placed as the fit left it on the
   reference (the fit's correspondence);
2. the vertex goes back to the card's rest pose by the inverse of the skinning at that point
   (the card's skin weights there, the four strongest), and binds to the same point of the
   card's rest skin (a surface binding on the reference body);
3. the covered skin becomes a mask over the card's vertices: a card vertex is covered when its
   closest triangle of the reference's skin is covered.

The packages carry ``provenance`` private: they are made from a reference character's own
files, and :func:`~opensculptboy.wearables.format.check_destination` keeps them out of the
repository tree. The builder also writes a copy of the card that lists the outfits.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re

import numpy as np

from opensculptboy.wearables.format import PRIVATE, Garment, Wearable, write_wearable
from opensculptboy.wearables.geometry import Surface, unit_rows

TOP_WEIGHTS = 4  # the skin weights that undo the pose, as the GLB files keep them
COVER_REACH = 0.04  # metres: the ray test that splits one cover mask among the pieces


@dataclasses.dataclass
class ObjectSpec:
    name: str
    node: str | None
    primitives: list  # [(primitive index, group number or None)]
    deformer: dict | None = None
    physics: list | None = None


@dataclasses.dataclass
class PieceSpec:
    id: str
    label: str
    slot: str
    layer: int
    objects: list
    hides: list  # [{"piece": id, "groups": ["<object>:group<n>", ...]}]
    covered: dict | None = None  # {region: [entries]} for the piece's fixed groups
    covered_groups: list = dataclasses.field(
        default_factory=list
    )  # [(group, {region})]
    head: dict | None = None
    split_cover: bool = False  # the outfit's one mask, split by a ray test


@dataclasses.dataclass
class OutfitSpec:
    id: str
    label: str
    default: bool
    pieces: list
    covered: dict | None = None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(text).lower()).strip("_") or "outfit"


def _group_name(obj: str, group) -> str:
    if isinstance(group, str):
        found = re.search(r"(\d+)$", group)
        group = int(found.group(1)) if found else group
    return f"{obj}:group{group}" if isinstance(group, int) else f"{obj}:{group}"


def _mask_bits(mask) -> list[int]:
    value = int(mask, 16) if isinstance(mask, str) else int(mask)
    return [b for b in range(value.bit_length()) if value >> b & 1]


def _piece_of_object(name: str) -> str:
    """The piece of a clothing object of a one-set section: the first word of its name."""
    return re.split(r"[ _]", name.strip())[0]


def _objects_specs(objects) -> list[ObjectSpec]:
    out = []
    for o in objects:
        if o.get("written") is False or not o.get("node"):
            continue
        prims = []
        for p in o.get("primitives", []):
            if isinstance(p, dict):
                prims.append((int(p["index"]), p.get("group")))
            else:
                prims.append((int(p), None))
        out.append(
            ObjectSpec(o["name"], o["node"], prims, o.get("deformer"), o.get("physics"))
        )
    return out


def outfit_specs(equipment: dict, outfit_id: str | None = None, label=None):
    """The outfits of an ``equipment`` section: its ``outfits``, or one outfit made of its
    ``objects`` (named ``outfit_id``, else after the section's ``source.look``)."""
    if equipment.get("outfits"):
        default = equipment.get("default_outfit")
        outfits = []
        for o in equipment["outfits"]:
            pieces = []
            for p in o.get("pieces", []):
                pid = str(p["piece"])
                if pid.startswith(f"{o['id']}."):
                    pid = pid[len(o["id"]) + 1 :]
                hides = []
                for rule in p.get("hides", []):
                    target = str(rule.get("piece", ""))
                    outfit_of_target = None
                    if "." in target:
                        outfit_of_target, target = target.split(".", 1)
                    if outfit_of_target and outfit_of_target != o["id"]:
                        target = f"{outfit_of_target}/{target}"
                    groups = [
                        _group_name(rule.get("object", target), g)
                        for g in rule.get("groups", [])
                    ]
                    hides.append({"piece": target, "groups": groups})
                extra = []
                for entry in p.get("covered_groups", []):
                    covered = entry.get("covered")
                    if isinstance(covered, dict):
                        extra.append(
                            (_group_name(entry["object"], entry["group"]), covered)
                        )
                pieces.append(
                    PieceSpec(
                        id=pid,
                        label=p.get("label") or pid,
                        slot=p["slot"],
                        layer=int(p.get("layer", 50)),
                        objects=_objects_specs(p.get("objects", [])),
                        hides=hides,
                        covered=p.get("covered"),
                        covered_groups=extra,
                        head=p.get("head"),
                    )
                )
            outfits.append(
                OutfitSpec(
                    id=o["id"],
                    label=o.get("label") or o["id"],
                    default=bool(o.get("default")) or o["id"] == default,
                    pieces=pieces,
                    covered=o.get("covered"),
                )
            )
        return outfits
    # one set of objects: a piece per slot of the reference (the first word of the names)
    objects = equipment.get("objects", [])
    names = [o["name"] for o in objects]
    by_piece: dict[str, list] = {}
    for o in objects:
        by_piece.setdefault(_piece_of_object(o["name"]), []).append(o)
    hides: dict[str, list] = {}
    for rule in equipment.get("hides", []):
        by = _piece_of_object(str(rule.get("by", "")))
        target_object = str(rule.get("object", "")).replace(" ", "_")
        if target_object not in names:  # "pants lg" -> pants_lg
            close = [n for n in names if n.startswith(target_object)]
            target_object = close[0] if close else target_object
        groups = [_group_name(target_object, b) for b in _mask_bits(rule["mask"])]
        hides.setdefault(by, []).append(
            {"piece": _piece_of_object(target_object), "groups": groups}
        )
    source = equipment.get("source") or {}
    oid = outfit_id or _slug(source.get("look") or "outfit")
    pieces = []
    for pid, objs in by_piece.items():
        specs = _objects_specs(objs)
        if not specs:
            continue
        pieces.append(
            PieceSpec(
                id=pid,
                label=pid,
                slot=objs[0].get("slot", "torso"),
                layer=max(int(o.get("layer", 50)) for o in objs),
                objects=specs,
                hides=hides.get(pid, []),
                head=equipment.get("head") if objs[0].get("slot") == "head" else None,
                split_cover=True,
            )
        )
    return [
        OutfitSpec(oid, label or oid, True, pieces, covered=equipment.get("covered"))
    ]


class _Replica:
    """The nodes and primitives of the reference's GLB file, in Anny's frame."""

    def __init__(self, path: pathlib.Path):
        from opensculptboy.fit.poses import GlbRig

        self.path = path
        self.rig = GlbRig(path)
        self.world = self.rig.world()

    def node_index(self, name: str) -> int:
        if name not in self.rig.names:
            raise ValueError(f"{self.path.name} has no node {name!r}.")
        return self.rig.names[name]

    def primitive(self, node: str, index: int) -> dict:
        from opensculptboy.fit.handoff import GLTF_TO_ANNY

        n = self.node_index(node)
        entry = self.rig.nodes[n]
        primitive = self.rig.document["meshes"][entry["mesh"]]["primitives"][index]
        a = primitive["attributes"]
        M = self.world[n]
        to_anny = GLTF_TO_ANNY @ M[:3, :3]
        v = self.rig.accessor(a["POSITION"]).astype(np.float64)
        v = (v @ M[:3, :3].T + M[:3, 3]) @ GLTF_TO_ANNY.T
        normals = None
        if "NORMAL" in a:
            normals = unit_rows(
                self.rig.accessor(a["NORMAL"]).astype(np.float64) @ to_anny.T
            )
        uvs = None
        if "TEXCOORD_0" in a:
            uvs = self.rig.accessor(a["TEXCOORD_0"]).astype(np.float64)
        if "indices" in primitive:
            t = self.rig.accessor(primitive["indices"]).reshape(-1, 3).astype(np.int64)
        else:
            t = np.arange(len(v)).reshape(-1, 3)
        return {
            "vertices": v,
            "normals": normals,
            "uvs": uvs,
            "triangles": t,
            "material": primitive.get("material"),
            "group": (primitive.get("extras") or {}).get("group"),
        }


class _Skin:
    """The reference's skin primitives that the cover entries name, as one surface."""

    def __init__(self, replica: _Replica, keys):
        self.keys = list(dict.fromkeys(keys))
        vertices, triangles, self.offset = [], [], {}
        count = tri = 0
        for key in self.keys:
            p = replica.primitive(*key)
            vertices.append(p["vertices"])
            triangles.append(p["triangles"] + count)
            self.offset[key] = tri
            count += len(p["vertices"])
            tri += len(p["triangles"])
        self.triangle_count = tri
        self.surface = Surface(np.concatenate(vertices), np.concatenate(triangles))

    def flags(self, covered: dict | None) -> np.ndarray:
        """(T,) the skin triangles that cover entries name."""
        out = np.zeros(self.triangle_count, bool)
        for entries in (covered or {}).values():
            for e in entries:
                key = (e["node"], int(e["primitive"]))
                if key in self.offset:
                    out[self.offset[key] + np.asarray(e["triangles"], np.int64)] = True
        return out


def _cover_keys(specs) -> list:
    keys = []

    def add(covered):
        for entries in (covered or {}).values():
            for e in entries:
                keys.append((e["node"], int(e["primitive"])))

    for outfit in specs:
        add(outfit.covered)
        for piece in outfit.pieces:
            add(piece.covered)
            for _, covered in piece.covered_groups:
                add(covered)
    return keys


def _relative_card(card: dict, old: pathlib.Path, new: pathlib.Path) -> dict:
    """The card with every path that names a file next to the old card made relative to the
    new card's folder."""

    def visit(value):
        if isinstance(value, dict):
            return {k: visit(v) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v) for v in value]
        if isinstance(value, str) and not os.path.isabs(value) and "." in value:
            path = old / value
            if path.is_file():
                return pathlib.Path(os.path.relpath(path, new)).as_posix()
        return value

    return visit(card)


def build_reference_wearables(
    handoff: str | pathlib.Path,
    card: str | pathlib.Path,
    out: str | pathlib.Path,
    outfit_id: str | None = None,
    outfit_label: str | None = None,
    log=print,
) -> dict:
    """
    Build the reference wearables of a handoff for a fitted card (see the module
    documentation) into ``out/wearables/``, and write ``out/<card name>`` with the outfits.
    Returns a report: per piece the vertex count, the distances of the correspondence and the
    round-trip error (the garment bound, evaluated at rest and posed back onto the
    reference).
    """
    import torch

    from opensculptboy.character import Character
    from opensculptboy.detail import DetailModel
    from opensculptboy.detail.layer import bone_transforms, character_offsets
    from opensculptboy.export.body import top_skin_weights
    from opensculptboy.fit.residual import blended_transforms
    from opensculptboy.wearables.bind import BodySurface, bind_surface, evaluate
    from opensculptboy.wearables.cover import ray_hits
    from opensculptboy.wearables.fit import anny_count, card_body, skin_triangles
    from opensculptboy.wearables.format import Binding, check_destination
    from opensculptboy.wearables.gltf import read_materials

    handoff, card, out = pathlib.Path(handoff), pathlib.Path(card), pathlib.Path(out)
    if handoff.is_file():
        handoff = handoff.parent
    check_destination(out / "wearables" / "x.wearable.json", PRIVATE)
    appearance = json.loads((handoff / "appearance.json").read_text())
    equipment = appearance.get("equipment")
    if not equipment:
        raise ValueError(f"{handoff}/appearance.json has no equipment section.")
    specs = outfit_specs(equipment, outfit_id, outfit_label)
    card_data = json.loads(card.read_text())
    character = Character.from_dict(card_data)
    placement = (character.detail.get("fit") or {}).get("placement")
    if not placement:
        raise ValueError(f"{card} holds no fit placement (detail.fit.placement).")
    R = np.asarray(placement["rotation"], dtype=np.float64)
    t = np.asarray(placement["translation"], dtype=np.float64)

    model = character.build_model(dtype=torch.float64)
    offsets = character_offsets(character, 0, card.parent)
    detailed = DetailModel(model, 0, offsets, "full", character.topology)
    with torch.no_grad():
        posed_out = detailed(
            pose_parameters=character.pose_parameters(model), **character.model_kwargs()
        )
    n = anny_count(model)
    posed = posed_out["vertices"][0].numpy()[:n] @ R.T + t
    transforms = bone_transforms(posed_out)[0].numpy()
    rest_body = card_body(character, model, card.parent)
    triangles = skin_triangles(model)
    posed_body = BodySurface(
        posed,
        triangles,
        rest_body.weights,
        rest_body.indices,
        rest_body.bone_poses,
        rest_body.bone_labels,
    )
    replica = _Replica(handoff / "replica.glb")
    skin = _Skin(replica, _cover_keys(specs))
    log(f"skin: {len(skin.keys)} primitives, {skin.triangle_count} triangles")
    # the skin triangle of the reference under each card vertex (for the cover masks)
    under = skin.surface.closest(posed[: len(rest_body.vertices)]).triangle

    folder = out / "wearables"
    folder.mkdir(parents=True, exist_ok=True)
    report = {"handoff": handoff.name, "card": card.name, "outfits": []}
    outfits_field = []
    for outfit in specs:
        outfit_cover = skin.flags(outfit.covered)[under] if outfit.covered else None
        entries = []
        outfit_report = {"id": outfit.id, "pieces": []}
        for piece in outfit.pieces:
            parts = []
            for obj in piece.objects:
                for index, group in obj.primitives:
                    p = replica.primitive(obj.node, index)
                    p["group_name"] = _group_name(
                        obj.name, group if group is not None else p["group"] or 0
                    )
                    parts.append(p)
            if not parts:
                continue
            groups = list(dict.fromkeys(p["group_name"] for p in parts))
            used = list(dict.fromkeys(p["material"] for p in parts))
            materials, index = read_materials(
                replica.rig.document, replica.rig.binary, handoff, which=used
            )
            vertices, normals, uvs, tris, tmat, tgroup = [], [], [], [], [], []
            count = 0
            for p in parts:
                vertices.append(p["vertices"])
                normals.append(
                    p["normals"]
                    if p["normals"] is not None
                    else np.zeros_like(p["vertices"])
                )
                uvs.append(
                    p["uvs"]
                    if p["uvs"] is not None
                    else np.zeros((len(p["vertices"]), 2))
                )
                tris.append(p["triangles"] + count)
                tmat.append(np.full(len(p["triangles"]), index[p["material"]]))
                tgroup.append(
                    np.full(len(p["triangles"]), groups.index(p["group_name"]))
                )
                count += len(p["vertices"])
            gear = np.concatenate(vertices)
            gear_normals = np.concatenate(normals)
            # the fit's correspondence: gear -> the reference's skin -> the card's skin
            on_skin = skin.surface.closest(gear)
            on_card = posed_body.surface.closest(on_skin.point)
            corners = posed_body.triangles[on_card.triangle]
            bary = on_card.barycentric
            probe = Binding("surface", "reference", rest_body.vertices, corners, bary)
            from opensculptboy.wearables.bind import body_weights

            w, j = body_weights(probe, rest_body, TOP_WEIGHTS)
            w, j, _ = top_skin_weights(w, j, TOP_WEIGHTS)
            M = blended_transforms(w.astype(np.float64), j, transforms)
            local = (gear - t) @ R  # the placement undone
            L = M[:, :3, :3]
            rest = np.einsum("nij,nj->ni", np.linalg.inv(L), local - M[:, :3, 3])
            placed_normals = gear_normals @ R
            rest_normals = unit_rows(np.einsum("nji,nj->ni", L, placed_normals))
            garment = Garment(
                rest,
                np.concatenate(tris),
                normals=rest_normals,
                uvs=np.concatenate(uvs),
                triangle_material=np.concatenate(tmat),
                triangle_group=np.concatenate(tgroup),
                materials=materials,
                groups=groups,
            )
            binding = bind_surface(garment, rest_body, "reference", corners, bary)
            # round trip: evaluate at rest, pose with the same weights, place
            back, _ = evaluate(binding, rest_body)
            again = np.einsum("nij,nj->ni", L, back) + M[:, :3, 3]
            error = np.linalg.norm(again @ R.T + t - gear, axis=1)
            # cover masks over the card's vertices
            covers = None
            if piece.covered:
                covers = skin.flags(piece.covered)[under]
            elif outfit_cover is not None and piece.split_cover:
                hit = ray_hits(
                    rest_body.vertices,
                    rest_body.normals,
                    rest,
                    garment.triangles,
                    COVER_REACH,
                )
                covers = outfit_cover & hit
            group_covers = {
                g: skin.flags(c)[under] for g, c in piece.covered_groups if g in groups
            }
            extras = {
                "source": "reference handoff",
                "outfit": outfit.id,
                "objects": [o.name for o in piece.objects],
                "deformers": {o.name: o.deformer for o in piece.objects if o.deformer},
                "physics": sorted(
                    {
                        ph.get("kind")
                        for o in piece.objects
                        for ph in (o.physics or [])
                        if isinstance(ph, dict) and ph.get("kind")
                    }
                ),
            }
            if piece.head:
                extras["head"] = {
                    k: piece.head[k] for k in ("hair", "ears") if k in piece.head
                }
            wearable = Wearable(
                name=f"{outfit.id}.{piece.id}",
                label=piece.label,
                slot=piece.slot,
                layer=piece.layer,
                garment=garment,
                binding=binding,
                provenance=PRIVATE,
                covers=covers,
                group_covers=group_covers,
                hides=piece.hides,
                extras=extras,
            )
            path = write_wearable(
                folder / f"{outfit.id}.{piece.id}.wearable.json", wearable
            )
            entries.append(
                {
                    "piece": piece.id,
                    "file": pathlib.Path(os.path.relpath(path, out)).as_posix(),
                    "enabled": True,
                }
            )
            stats = {
                "piece": piece.id,
                "objects": [o.name for o in piece.objects],
                "vertices": int(len(gear)),
                "triangles": int(len(garment.triangles)),
                "materials": len(materials),
                "groups": groups,
                "gear_to_skin_mm_p95": round(
                    float(np.percentile(on_skin.distance, 95)) * 1000, 2
                ),
                "skin_to_card_mm_p95": round(
                    float(np.percentile(on_card.distance, 95)) * 1000, 3
                ),
                "round_trip_mm_max": round(float(error.max()) * 1000, 4),
                "covered_vertices": None if covers is None else int(covers.sum()),
                "hides": piece.hides,
            }
            outfit_report["pieces"].append(stats)
            log(json.dumps(stats))
        outfits_field.append(
            {"id": outfit.id, "label": outfit.label, "pieces": entries}
        )
        report["outfits"].append(outfit_report)
    default = next((o.id for o in specs if o.default), specs[0].id if specs else "")
    new_card = _relative_card(card_data, card.parent.resolve(), out.resolve())
    new_card.update(schema=4, outfits=outfits_field, active_outfit=default)
    Character.from_dict(new_card)  # validate
    target = out / card.name
    check_destination(target, PRIVATE)
    target.write_text(json.dumps(new_card, indent=2) + "\n")
    report["card"] = str(target)
    (out / "wearables" / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

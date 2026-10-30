# OpenSculptBoy
# Apache License, Version 2.0
"""
A fitted statue as a finished starter of the web viewer's New screen: one ``.zip`` beside the
page (``starters/<id>.zip``) that the page opens through its Open path.

The zip holds the statue's card as ``character.json`` with the detail files it names, the
statue's props as wearable packages of the slot ``prop`` (:func:`prop_packages`: the base, a
tree stump, a sword, a hat, locks, cut from the scan by the config's masks and bound rigidly
to the nearest bone, so that the viewer's wardrobe shows and hides each), and
``thumbnail.png``. The card is the fitted card of
:func:`opensculptboy.statues.pipeline.run_statue` with:

- ``name``: the statue's title;
- ``look`` (card schema 5, the web viewer's block): marble skin (``stone``), no page hair, as
  the carved hair is in the sculpted surface, and stone-coloured eyes;
- ``detail.fit.reference``: the scan's title, source, licence and credit, so that the credit
  travels with every copy of the card;
- an outfit ``statue`` that wears the props.

:func:`light_card` writes the light version beside it: the fit's sliders, face shapes and
pose without the detail files, a card of a few kilobytes.

:func:`build_starter` writes the zip and returns what it holds; ``python -m
opensculptboy.statues starter`` is its command.
"""

from __future__ import annotations

import io
import json
import pathlib
import zipfile

import numpy as np

# the web viewer's stone for marble and its colour for the hair and the eyes
# (viewer/src/starters.ts, STONE_HAIR)
STONE = "carrara"
STONE_COLOUR = "#dcd7cf"
# the card schema that carries the look block
LOOK_SCHEMA = 5
THUMBNAIL_SIZE = (256, 384)


def statue_look(stone: str = STONE, colour: str = STONE_COLOUR) -> dict:
    """The look block of a marble statue: the stone, no page hair, stone-coloured eyes."""
    return {
        "skin": {"tone": 0.35, "undertone": 0.0, "stone": stone, "polish": 0.5},
        "hair": {"style": "bald", "color": colour},
        "eyes": {"color": colour},
    }


def starter_card(card: dict, config, levels=(0, 1, 2)) -> dict:
    """The card of the starter: the fitted ``card`` (a dict) with the statue's name, the
    marble look, the scan's credit and only the detail ``levels`` asked for."""
    out = json.loads(json.dumps(card))
    out["schema"] = LOOK_SCHEMA
    out["name"] = config.title or config.name
    detail = out.setdefault("detail", {})
    detail["levels"] = [
        e for e in detail.get("levels", []) if int(e.get("subdiv", -1)) in set(levels)
    ]
    fit = detail.setdefault("fit", {})
    fit["reference"] = {
        "title": config.title or config.name,
        "source": config.source,
        "licence": config.licence,
        "credit": config.credit,
        "scan": config.scan,
    }
    out["look"] = statue_look()
    # the schema first, as the web viewer writes its cards
    return {"schema": out.pop("schema"), **out}


def thumbnail(vertices: np.ndarray, triangles: np.ndarray, size=THUMBNAIL_SIZE):
    """A picture (PIL, RGB) of the fitted statue from the front three-quarter, in marble."""
    from opensculptboy.render.dressed import Mesh, View, draw

    return draw(
        [Mesh(np.asarray(vertices), np.asarray(triangles), (214, 210, 202))],
        View(-25.0),
        size,
        12.0,
        background=(46, 47, 50),
    )


PROP_SLOT = "prop"
# a prop keeps at most this many triangles for the page
PROP_TRIANGLES = 20000
# and a hat or locks, whose carving the reduction to PROP_TRIANGLES smoothed away, by role
PIECE_TRIANGLES = {"drapery": 80000, "hair": 100000}
# the rebuilt pieces (opensculptboy/statues/blender_pieces.py) beside a run, by role: they
# take the place of the seated cuts
REBUILT = {"drapery": "hat.ply", "hair": "locks.ply"}
# a prop that reaches this close (metres) to the ground stands on it: bound to the root
GROUND_REACH = 0.03
PROP_COLOUR = (0.84, 0.82, 0.79, 1.0)
# the props bound to the root: they stand on the ground
ROOT_PROPS = ("Base",)
OUTFIT = "statue"


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def statue_props(root, props=None) -> list:
    """
    The props of a statue run, as (name, vertices (N, 3) in the handoff's frame, triangles):
    the named pieces of ``props.glb`` beside the handoff (``props``, default
    ``<root>/handoff``), with a hat and locks taken from the rebuilt pieces in
    ``<root>/pieces/`` (:data:`REBUILT`, the closed meshes of ``blender_pieces.py``) when
    they exist, else from the measure step's seated pieces (``meshes/seated_*``), which meet
    the fitted skin. Each entry carries the piece's role as a fourth item.
    """
    import trimesh

    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    folder = pathlib.Path(props) if props else pathlib.Path(root) / "handoff"
    path = folder / "props.glb"
    if not path.exists():
        return []
    roles = {}
    if (folder / "props.json").exists():
        info = json.loads((folder / "props.json").read_text(encoding="utf-8"))
        roles = {p["name"]: p["role"] for p in info.get("props", [])}
    scene = trimesh.load(str(path), force="scene")
    out = []
    seated_role = {"drapery": "clothing", "hair": "hair"}
    for name, geometry in scene.geometry.items():
        role = roles.get(name, "support")
        rebuilt = pathlib.Path(root) / "pieces" / REBUILT.get(role, "-")
        if rebuilt.exists():
            geometry = trimesh.load(str(rebuilt), process=False)
        elif role in seated_role:
            files = sorted(
                (pathlib.Path(root) / "meshes").glob(
                    f"seated_{seated_role[role]}_*.ply"
                )
            )
            if files:
                meshes = [trimesh.load(str(f), process=False) for f in files]
                geometry = trimesh.util.concatenate(meshes)
        out.append(
            (
                name,
                np.asarray(geometry.vertices, np.float64) @ GLTF_TO_ANNY.T,
                np.asarray(geometry.faces),
                role,
            )
        )
    return out


def _write_prop_glb(path, vertices, triangles, name):
    """A prop as a GLB of one marble primitive (glTF's frame, Anny's rest)."""
    from opensculptboy.export.body import ANNY_TO_GLTF
    from opensculptboy.export.document import ARRAY_BUFFER, GltfDocument
    from opensculptboy.wearables.gltf import index_accessor

    positions = (np.asarray(vertices) @ ANNY_TO_GLTF.T).astype(np.float32)
    doc = GltfDocument(name)
    material = doc.add_material(
        {
            "name": "marble",
            "pbrMetallicRoughness": {
                "baseColorFactor": list(PROP_COLOUR),
                "metallicFactor": 0.0,
                "roughnessFactor": 0.45,
            },
        }
    )
    primitive = {
        "attributes": {"POSITION": doc.accessor(positions, ARRAY_BUFFER, True)},
        "indices": index_accessor(doc, np.asarray(triangles), len(positions)),
        "material": material,
    }
    mesh = doc.add_mesh({"name": name, "primitives": [primitive]})
    doc.scene_nodes = [doc.add_node({"name": name, "mesh": mesh})]
    doc.write(path)


def prop_packages(root, card: dict, config, out_dir, props=None, log=print) -> list:
    """
    Write the statue's props (:func:`statue_props`) as wearable packages of the slot
    ``prop`` into ``out_dir`` and return their entries (name, label, files, bone). Each prop
    goes from the placed, posed statue to the card's rest by the inverse of its bone's
    turn, is
    reduced to :data:`PROP_TRIANGLES` (a hat and locks to :data:`PIECE_TRIANGLES`) and bound
    rigidly to that bone, so that the viewer
    poses it with the figure. A prop that stands on the ground (within :data:`GROUND_REACH`
    of the base's top) is bound to the root as the base is; the others to the bone of the
    skin nearest their closest points.
    """
    import torch
    from scipy.spatial import cKDTree

    from opensculptboy.character import Character
    from opensculptboy.export.decimate import decimate
    from opensculptboy.fit import residual as R_
    from opensculptboy.fit.register import _card_rest, fit_model
    from opensculptboy.wearables.fit import card_body
    from opensculptboy.wearables.format import write_wearable
    from opensculptboy.wearables.make import make_wearable

    pieces = statue_props(root, props)
    if not pieces:
        return []
    root = pathlib.Path(root)
    fitted = {k: v for k, v in card.items() if k != "look"}
    fitted["schema"] = min(int(fitted.get("schema", 4)), 4)
    character = Character.from_dict(fitted)
    model = fit_model(character.topology)
    rest0, transforms = _card_rest(model, character)
    placement = card["detail"]["fit"]["placement"]
    R = np.asarray(placement["rotation"], np.float64)
    t = np.asarray(placement["translation"], np.float64)
    weights = model.vertex_bone_weights.detach().double().numpy()
    indices = model.vertex_bone_indices.numpy()
    posed = R_.pose_vertices(rest0, weights, indices, transforms, R, t)
    dominant = indices[np.arange(len(indices)), weights.argmax(1)]
    labels = list(model.bone_labels)
    tree = cKDTree(posed)
    # the ground: the base's top where there is a base, else the figure's lowest point
    floor = posed[:, 2].min()
    for name, V, _, _ in pieces:
        if name in ROOT_PROPS:
            floor = float(np.quantile(V[:, 2], 0.98))
    body = card_body(
        character, character.build_model(dtype=torch.float64), root / "card"
    )
    licence = {
        "spdx": config.licence or "NOASSERTION",
        "author": config.credit,
        "source": config.source,
        "design_source": f"cut from the scan of {config.title or config.name}",
    }
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for layer, (name, V, T, role) in enumerate(pieces):
        sample = V[:: max(1, len(V) // 20000)]
        d, near = tree.query(sample)
        grounded = sample[:, 2].min() < floor + GROUND_REACH
        if name in ROOT_PROPS or grounded:
            bone = labels.index("root")
        else:
            # the bone at the contact: the skin nearest the prop's closest points (a sword's
            # hilt in the hand, a hat on the head)
            contact = d <= np.quantile(d, 0.01)
            bone = int(np.bincount(dominant[near[contact]]).argmax())
        one = np.zeros((len(V), weights.shape[1]))
        one[:, 0] = 1.0
        idx = np.zeros((len(V), indices.shape[1]), indices.dtype)
        idx[:, 0] = bone
        rest = R_.unpose(V, one, idx, transforms, R, t)
        budget = PIECE_TRIANGLES.get(role, PROP_TRIANGLES)
        if len(T) > budget:
            reduced = decimate(rest, T, budget)
            T = reduced.triangles
            used = np.unique(T)
            remap = np.full(len(rest), -1)
            remap[used] = np.arange(len(used))
            rest, T = rest[used], remap[T]
        slug = _slug(name)
        glb = out_dir / f"{slug}.source.glb"
        _write_prop_glb(glb, rest, T, name)
        wearable, report = make_wearable(
            glb,
            body,
            slug,
            PROP_SLOT,
            layer,
            label=name,
            rigid=labels[bone],
            covers="none",
            licence=licence,
            frame="centred",
            max_distance=2.0,
            extras={"source": "opensculptboy statue starter", "statue": config.name},
            description=f"{name} of {config.title or config.name}, cut from its scan",
            accept_quality=True,
        )
        written = write_wearable(out_dir / slug, wearable)
        glb.unlink()
        files = sorted(p for p in out_dir.glob(f"{slug}.*") if p.is_file())
        log(f"prop {name}: {len(T)} triangles on {labels[bone]}")
        entries.append(
            {
                "name": slug,
                "label": name,
                "bone": labels[bone],
                "file": written.name,
                "files": [str(p) for p in files],
            }
        )
    return entries


def light_card(card: dict, config) -> dict:
    """The light version of a statue's card: its sliders, face shapes, local changes and
    pose with the marble look and the scan's credit, without the detail files."""
    out = starter_card(card, config, levels=())
    detail = out.pop("detail", {})
    reference = detail.get("fit", {}).get("reference")
    out.pop("wearables", None)
    out.pop("outfits", None)
    out.pop("active_outfit", None)
    if reference:
        # the credit has no field of its own outside the detail layer: a detail block
        # without levels carries it
        out["detail"] = {"fit": {"reference": reference}}
    return out


def build_starter(
    root,
    config,
    out,
    levels=(0, 1, 2),
    pieces=(),
    picture=None,
    props=None,
    with_props: bool = True,
    log=print,
) -> dict:
    """
    Write the starter zip ``out`` of a statue whose pipeline wrote ``root`` (the folder
    ``<out>/<name>/`` of :func:`~opensculptboy.statues.pipeline.run_statue`, with
    ``card/<name>.json`` and its detail files): the card (:func:`starter_card`) with the
    detail ``levels``, the statue's props (:func:`prop_packages`, from ``props`` or the
    run's handoff) worn in the outfit ``statue``, other package files ``pieces`` and
    ``thumbnail.png`` (``picture``, a PNG file, or a drawing of the finest fitted level),
    and the light card beside it (``<out stem>.light.json``). Returns the entries and their
    sizes in bytes.
    """
    import tempfile

    root = pathlib.Path(root)
    card_path = root / "card" / f"{root.name}.json"
    fitted = json.loads(card_path.read_text(encoding="utf-8"))
    card = starter_card(fitted, config, levels)
    files = {}
    for entry in card["detail"]["levels"]:
        files[entry["offsets"]] = (card_path.parent / entry["offsets"]).read_bytes()
    for piece in pieces:
        piece = pathlib.Path(piece)
        files[piece.name] = piece.read_bytes()
    made = []
    if with_props:
        with tempfile.TemporaryDirectory() as tmp:
            made = prop_packages(root, card, config, tmp, props, log)
            for entry in made:
                for f in entry["files"]:
                    files[pathlib.Path(f).name] = pathlib.Path(f).read_bytes()
        if made:
            card["outfits"] = [
                {
                    "id": OUTFIT,
                    "label": "Statue pieces",
                    "pieces": [
                        {"piece": e["name"], "file": e["file"], "enabled": True}
                        for e in made
                    ],
                }
            ]
            card["active_outfit"] = OUTFIT
    if picture is not None:
        png = pathlib.Path(picture).read_bytes()
    else:
        import trimesh

        from opensculptboy.fit.handoff import GLTF_TO_ANNY

        finest = (
            max(int(e["subdiv"]) for e in card["detail"]["levels"])
            if card["detail"]["levels"]
            else 0
        )
        mesh = trimesh.load(
            str(root / "meshes" / f"fitted_level{finest}.ply"), process=False
        )
        buffer = io.BytesIO()
        thumbnail(np.asarray(mesh.vertices) @ GLTF_TO_ANNY.T, mesh.faces).save(
            buffer, "PNG"
        )
        png = buffer.getvalue()
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = {"character.json": (json.dumps(card, indent=2) + "\n").encode("utf-8")}
    entries.update(files)
    entries["thumbnail.png"] = png
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in entries.items():
            z.writestr(name, data)
    out.with_suffix(".png").write_bytes(png)
    light = out.with_suffix(".light.json")
    light.write_text(
        json.dumps(light_card(fitted, config), indent=2) + "\n", encoding="utf-8"
    )
    return {
        "zip": str(out),
        "bytes": out.stat().st_size,
        "entries": {name: len(data) for name, data in entries.items()},
        "levels": [int(e["subdiv"]) for e in card["detail"]["levels"]],
        "props": [{k: e[k] for k in ("label", "bone", "file")} for e in made],
        "light": str(light),
        "light_bytes": light.stat().st_size,
    }

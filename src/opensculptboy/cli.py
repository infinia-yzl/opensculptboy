# OpenSculptBoy
# Apache License, Version 2.0
"""
The ``opensculptboy`` command.

    opensculptboy character > me.json                # a character card with the default settings
    opensculptboy export me.glb --character me.json --animation walk --animation wave
    opensculptboy export me.glb --morph-targets all  # every facial action and face shape
    opensculptboy export me.vrm --character me.json --author "Me"   # a VRM 1.0 avatar
    opensculptboy export me0.vrm --author "Me" --vrm-version 0 --thumbnail auto   # VRM 0.x
    opensculptboy names poses                        # the poses and clips of the library
    opensculptboy pose photo.jpg --card pose.json --glb pose.glb --svg pose.svg --png pose.png
    opensculptboy fit handoff/ --out me.json --detail 1    # fit anny to a reference mesh
    opensculptboy export me.glb --character me.json --detail 1   # with the detail layer
    opensculptboy viewer build                       # the web viewer (python -m anny.viewer build)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

from opensculptboy.character import Character

# The options of ``export`` that only one of the two formats takes.
_GLB_ONLY = ("animation", "morph_targets", "influences", "no_ground")
_VRM_ONLY = (
    "author",
    "name",
    "meta",
    "vrm_version",
    "thumbnail",
    "twist",
    "bind",
    "keep_leg_spread",
    "budget",
    "anny_vertex",
    "refine_weights",
)


def _flag(dest: str) -> str:
    return "--" + dest.replace("_", "-")


def _load_character(path: str | None, parser=None) -> Character:
    """
    The character card at ``path``, or the default character; with ``parser``, a card that
    cannot be read is a usage error.
    """
    if path is None:
        return Character()
    try:
        return Character.from_dict(json.loads(pathlib.Path(path).read_text()))
    except (OSError, ValueError, TypeError) as error:
        if parser is None:
            raise
        parser.error(f"--character {path}: {_one_line(error)}")


def _one_line(error: BaseException) -> str:
    """The message of an error on one line."""
    return " ".join(str(error).split()) or type(error).__name__


def _detail_dir(path: str | None) -> pathlib.Path | None:
    """The folder of a card, which its detail files are relative to."""
    return None if path is None else pathlib.Path(path).resolve().parent


def _with_eyes(character, value, parser):
    """The character with the eyes of ``--eyes``: None keeps the card as it is; "" turns on
    the eyes of opensculptboy.eyes (the card's, else the defaults); a file's values are laid
    over the card's eyes."""
    if value is None:
        return character
    import dataclasses

    from opensculptboy.material import check_eyes

    eyes = dict(character.material.get("eyes", {}))
    if value:
        try:
            eyes.update(json.loads(pathlib.Path(value).read_text()))
            check_eyes(eyes)
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"--eyes {value}: {_one_line(error)}")
    return dataclasses.replace(character, material={**character.material, "eyes": eyes})


def _with_teeth(character, value, parser):
    """The character with the teeth of ``--teeth``: None keeps the card as it is; "" turns
    on the teeth of opensculptboy.teeth (the card's, else generic teeth); "off" leaves them
    out; a file's values are laid over the card's teeth."""
    if value is None:
        return character
    import dataclasses

    from opensculptboy.material import check_teeth

    material = dict(character.material)
    if value == "off":
        material.pop("teeth", None)
        return dataclasses.replace(character, material=material)
    teeth = dict(material.get("teeth", {}))
    if value:
        try:
            teeth.update(json.loads(pathlib.Path(value).read_text()))
            check_teeth(teeth)
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"--teeth {value}: {_one_line(error)}")
    return dataclasses.replace(character, material={**material, "teeth": teeth})


def _export(args, parser) -> int:
    is_vrm = pathlib.Path(args.output).suffix.lower() == ".vrm"
    wrong = _GLB_ONLY if is_vrm else _VRM_ONLY
    given = [_flag(d) for d in wrong if getattr(args, d) not in (None, False)]
    if given:
        kind, other = (".vrm", ".glb") if is_vrm else (".glb", ".vrm")
        parser.error(
            f"{', '.join(given)}: only for {other} files, not for the {kind} file "
            f"{args.output}."
        )
    if is_vrm:
        return _export_vrm(args, parser)
    from opensculptboy.export.gltf import export_glb

    character = _with_eyes(_load_character(args.character, parser), args.eyes, parser)
    character = _with_teeth(character, args.teeth, parser)
    model, morph_targets = None, None  # export_glb builds the model it needs
    if args.morph_targets == "none":
        morph_targets = []
    elif args.morph_targets == "all":
        model = character.build_model(face_shapes=True)
        morph_targets = list(model.facial_action_labels) + list(model.face_shape_labels)
    elif args.morph_targets not in (None, "facial"):
        morph_targets = [n.strip() for n in args.morph_targets.split(",") if n.strip()]
    if args.detail and args.detail not in character.detail_levels():
        parser.error(
            f"--detail {args.detail}: the character has no detail level {args.detail} "
            f"(its levels: {character.detail_levels() or 'none'})."
        )
    summary = export_glb(
        args.output,
        character,
        model,
        animations=args.animation or (),
        morph_targets=morph_targets,
        max_influences=args.influences or 4,
        ground=not args.no_ground,
        detail_level=args.detail or 0,
        detail_dir=_detail_dir(args.character),
    )
    print(json.dumps(summary, indent=2))
    return 0


def _export_vrm(args, parser) -> int:
    from opensculptboy.export.budget import BudgetError
    from opensculptboy.export.vrm import DEFAULT_BIND, VrmMeta, export_vrm

    meta = None
    if args.meta:
        try:
            meta = VrmMeta.from_dict(json.loads(pathlib.Path(args.meta).read_text()))
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"--meta {args.meta}: {_one_line(error)}")
    if not args.author and not (meta and meta.authors):
        parser.error(
            "a VRM file needs an author: pass --author NAME, or --meta FILE with "
            '"authors": ["NAME"].'
        )
    character = _with_eyes(_load_character(args.character, parser), args.eyes, parser)
    character = _with_teeth(character, args.teeth, parser)
    # A file over a strict budget fails (exit 1); a character, an option or a thumbnail
    # that a VRM file cannot carry is a usage error (exit 2). Each prints one line, after
    # the warnings of the export.
    failure = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            summary = export_vrm(
                args.output,
                character,
                version={"1": "1.0", "0": "0.x"}[args.vrm_version or "1"],
                meta=meta,
                author=args.author,
                name=args.name,
                thumbnail=args.thumbnail,
                bare=args.bare,
                twist=args.twist,
                bind=args.bind or DEFAULT_BIND,
                keep_leg_spread=args.keep_leg_spread,
                budget=args.budget or "warn",
                keep_anny_vertex=args.anny_vertex,
                refine_weights=args.refine_weights is not False,
                detail_level=args.detail or 0,
                detail_dir=_detail_dir(args.character),
            )
        except BudgetError as error:
            failure = (1, _one_line(error))
        except (OSError, ValueError, TypeError) as error:
            failure = (2, _one_line(error))
        finally:
            for warning in caught:
                print(f"warning: {warning.message}", file=sys.stderr)
    if failure is not None:
        code, message = failure
        if code == 2:
            parser.error(message)
        print(f"opensculptboy export: {message}", file=sys.stderr)
        return code
    print(json.dumps(summary, indent=2))
    return 0


def _pose(args) -> int:
    from opensculptboy.posing.picture import flat_view, pose_from_image, posed_mesh
    from opensculptboy.render.flat import outline_svg, shaded_png, silhouette_svg

    if not any((args.card, args.glb, args.svg, args.outline, args.png)):
        args.card = "-"
    character = _load_character(args.character)
    model = character.build_model()
    character = pose_from_image(
        args.image,
        character,
        model,
        refine=not args.no_refine,
        hands=not args.no_hands,
        face=not args.no_face,
        head=dict(turn=args.head_turn, up=args.head_up, tilt=args.head_tilt),
    )
    written = []
    if args.card:
        text = json.dumps(character.to_dict(), indent=2)
        if args.card == "-":
            print(text)
        else:
            pathlib.Path(args.card).write_text(text + "\n")
            written.append(args.card)
    if args.glb:
        from opensculptboy.export.gltf import export_glb

        export_glb(args.glb, character, model)
        written.append(args.glb)
    if args.svg or args.outline or args.png:
        vertices, faces = posed_mesh(character, model)
        view = flat_view(character, args.view)
        size = (args.size, args.size)
        desc = f"{character.name}, posed from {pathlib.Path(args.image).name}"
        if args.svg:
            text = silhouette_svg(vertices, faces, view, size=size, desc=desc)
            pathlib.Path(args.svg).write_text(text)
            written.append(args.svg)
        if args.outline:
            text = outline_svg(vertices, faces, view, size=size, desc=desc)
            pathlib.Path(args.outline).write_text(text)
            written.append(args.outline)
        if args.png:
            shaded_png(vertices, faces, view, size=size).save(args.png)
            written.append(args.png)
    for path in written:
        print(f"wrote {path}", file=sys.stderr)
    return 0


def _fit(args, parser) -> int:
    from opensculptboy.fit.handoff import load_handoff
    from opensculptboy.fit.reference import Reference
    from opensculptboy.fit.register import FitOptions, fit_reference

    try:
        handoff = load_handoff(args.handoff)
    except (OSError, ValueError) as error:
        parser.error(f"{args.handoff}: {_one_line(error)}")
    reference = Reference.from_handoff(handoff)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    name = args.name or pathlib.Path(handoff.path).resolve().name
    print(
        f"fitting {name}: {len(reference.vertices)} vertices in "
        f"{len(handoff.skin_parts())} skin parts, "
        + (
            "landmarks from the bounding geometry"
            if reference.auto
            else f"{len(reference.landmarks)} landmarks"
        ),
        file=sys.stderr,
    )
    result = fit_reference(
        reference,
        detail=args.detail,
        name=name,
        options=FitOptions(verbose=not args.quiet, topology=args.topology),
    )
    from opensculptboy.mouth.weights import attach_jaw_share

    if attach_jaw_share(result, handoff):
        print(
            "jaw share from the reference's mouth poses: "
            + ", ".join(result.jaw_share_poses),
            file=sys.stderr,
        )
    textures = None
    if args.textures:
        from opensculptboy.texture.bake import bake_fit

        baked = bake_fit(
            result,
            size=args.texture_size,
            head_size=args.head_size,
            skin_maps=True,
            ear_tile=args.head_size > 0,
        )
        textures, material = _write_textures(
            baked, out, int(max(result.offsets)), result.character.topology
        )
        result.character.material = _merge_material(material, args.material, parser)
        _mouth_rig(result.character)
    result.write(out, textures=textures)
    summary = {
        "card": str(out),
        "levels": sorted(int(k) for k in result.offsets),
        "error_mm": {
            str(k): {d: round(v[d]["mean"], 4) for d in ("forward", "reverse")}
            for k, v in result.errors.items()
        },
        "seconds": {k: round(v, 1) for k, v in result.seconds.items()},
    }
    if textures:
        summary["textures"] = textures
    print(json.dumps(summary, indent=2))
    return 0


def _write_textures(baked, card: pathlib.Path, subdiv: int, topology: str = "anny"):
    """Write the baked textures next to a card as ``<card>.<slot>.png``: the detail layer's
    ``textures`` (the eye texture) and the skin's ``material`` section."""
    stem = card.name[: -len(card.suffix)] if card.suffix else card.name
    names = {}
    for slot, data in baked.png_files().items():
        path = card.parent / f"{stem}.{slot}.png"
        path.write_bytes(data)
        names[slot] = path.name
    if baked.ear_uvs is not None:
        path = card.parent / f"{stem}.ear_uvs.safetensors"
        baked.ear_uvs.save(path, topology)
        names["ear_uvs"] = path.name
    if getattr(baked, "teeth", None) is not None:
        path = card.parent / f"{stem}.teeth.safetensors"
        baked.teeth.save(path)
        names["teeth"] = path.name
    if getattr(baked, "mouth", None) is not None:  # the reference's inner mouth
        path = card.parent / f"{stem}.mouth.safetensors"
        baked.mouth.save(path)
        names["mouth"] = path.name
        for slot, source in baked.mouth_textures.items():
            path = card.parent / f"{stem}.mouth_{slot}.png"
            path.write_bytes(pathlib.Path(source).read_bytes())
            names[f"mouth_{slot}"] = path.name
    textures = {"subdiv": subdiv}
    for slot in ("eyes", "eyes_normal"):
        if slot in names:
            textures[slot] = names[slot]
    return textures, baked.material(names)


def _mouth_rig(character) -> None:
    """Give a card with a reference's inner mouth (``material.mouth``) the ``anny-mouth``
    rig, whose jaw and tongue bones move it."""
    from opensculptboy.mouth.rig import MOUTH_RIG

    if "mouth" in (character.material or {}) and character.rig == "anny":
        character.rig = MOUTH_RIG


def _merge_material(material: dict, path: str | None, parser) -> dict:
    """The baked ``material`` section with the values of a JSON file (subsurface, specular,
    roughness factor...) laid over it."""
    from opensculptboy.material import check_material

    if path:
        try:
            extra = json.loads(pathlib.Path(path).read_text())
        except (OSError, ValueError) as error:
            parser.error(f"{path}: {_one_line(error)}")
        if not isinstance(extra, dict):
            parser.error(f"{path} must hold a JSON object.")
        material = {**material, **extra}
    try:
        check_material(material)
    except ValueError as error:
        parser.error(_one_line(error))
    return material


def _bake(args, parser) -> int:
    from opensculptboy.texture.bake import bake_card

    card = pathlib.Path(args.card)
    out = pathlib.Path(args.out) if args.out else card
    try:
        character = _load_character(str(card), parser)
        baked = bake_card(
            card,
            args.handoff,
            args.level,
            args.texture_size,
            head_size=args.head_size,
            eye_size=args.eye_size,
            skin_maps=True,
            ear_tile=not args.no_ear_tile,
        )
    except (OSError, ValueError) as error:
        parser.error(_one_line(error))
    out.parent.mkdir(parents=True, exist_ok=True)
    level = args.level if args.level is not None else max(character.detail_levels())
    textures, material = _write_textures(baked, out, int(level), character.topology)
    character.material = _merge_material(material, args.material, parser)
    _mouth_rig(character)
    detail = dict(character.detail)
    detail["textures"] = textures
    if out.parent.resolve() != card.parent.resolve():
        # the offsets stay where they are: point at them from the new card's folder
        import os

        detail["levels"] = [
            dict(e, offsets=os.path.relpath(card.parent / e["offsets"], out.parent))
            for e in detail.get("levels", [])
        ]
    character.detail = detail
    out.write_text(json.dumps(character.to_dict(), indent=2) + "\n")
    print(
        json.dumps(
            {"card": str(out), "covered": round(baked.covered_share, 4), **textures},
            indent=2,
        )
    )
    return 0


def _convert(args, parser) -> int:
    from opensculptboy.topology.convert import convert_card

    out = pathlib.Path(args.out)
    if out.resolve() == pathlib.Path(args.card).resolve():
        parser.error("--out must name another card than the one converted.")
    try:
        summary = convert_card(args.card, out, args.to, body=not args.default_body)
    except (OSError, ValueError, KeyError) as error:
        parser.error(f"{args.card}: {_one_line(error)}")
    print(json.dumps(summary, indent=2))
    return 0


def _names(args) -> int:
    import anny

    if args.kind == "poses":
        import anny.poses

        names = anny.poses.names()
    elif args.kind == "local-changes":
        names = anny.Anny(local_changes="all").local_change_labels
    else:
        model = Character().build_model(face_shapes=args.kind == "face-shapes")
        attribute = {
            "phenotypes": "phenotype_labels",
            "facial-actions": "facial_action_labels",
            "face-shapes": "face_shape_labels",
        }[args.kind]
        names = getattr(model, attribute)
    print("\n".join(names))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="opensculptboy", description="Characters of the Anny body model."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("character", help="print a character card with the default settings")

    e = sub.add_parser(
        "export",
        help="export a character as a glTF 2.0 binary file (.glb) or a VRM avatar (.vrm)",
        description="Export a character. The suffix of the output picks the format: .vrm "
        "writes a VRM avatar for VTuber apps (VRM 1.0, or VRM 0.x with --vrm-version 0), "
        "any other suffix a glTF 2.0 binary file.",
    )
    e.add_argument("output", help="the .glb or .vrm file to write")
    e.add_argument(
        "--character",
        help="a character card (JSON); the default character when omitted",
    )
    e.add_argument(
        "--bare",
        action="store_true",
        help="the body alone: no outfit, hair cards, teeth or lashes",
    )
    e.add_argument(
        "--detail",
        type=int,
        choices=(0, 1, 2),
        help="the level of the card's detail layer (default 0: anny's own mesh with the "
        "card's level-0 offsets, if any); a VRM file keeps within its triangle budget",
    )
    e.add_argument(
        "--eyes",
        nargs="?",
        const="",
        metavar="FILE",
        help="smooth eyes with an iris, a cornea and a caruncle in place of anny's eyeballs "
        "(the card's material.eyes, else the defaults); FILE, a JSON object of eye values "
        "(docs/eyes.md), is laid over them",
    )
    e.add_argument(
        "--teeth",
        nargs="?",
        const="",
        metavar="FILE",
        help="teeth in anny's mouth (the card's material.teeth: the reference's teeth of the "
        "bake, else generic teeth); FILE, a JSON object of teeth values (docs/teeth.md), is "
        "laid over them, and 'off' leaves the teeth out",
    )
    g = e.add_argument_group("glTF (.glb) options")
    g.add_argument(
        "--animation",
        action="append",
        help="a pose or clip of the library; repeat for more",
    )
    g.add_argument(
        "--morph-targets",
        help="'facial' (the default: the 52 facial actions), 'all' (with the face shapes), "
        "'none', or a comma-separated list of names",
    )
    g.add_argument(
        "--influences",
        type=int,
        choices=(4, 8),
        help="bones per vertex (default 4)",
    )
    g.add_argument(
        "--no-ground", action="store_true", help="keep the rest pose at Anny's origin"
    )
    r = e.add_argument_group("VRM (.vrm) options")
    r.add_argument(
        "--author",
        help="the avatar's author (required, unless --meta names the authors)",
    )
    r.add_argument("--name", help="the avatar's name (default: the character's name)")
    r.add_argument(
        "--meta",
        metavar="FILE",
        help="licence metadata as JSON: the fields of opensculptboy.export.vrm.VrmMeta "
        '(for example {"authors": ["Me"], "commercial_usage": "personalProfit"})',
    )
    r.add_argument(
        "--vrm-version",
        choices=("1", "0"),
        help="1 for VRM 1.0 (the default: Warudo, VMagicMirror, three-vrm), 0 for VRM 0.x "
        "(VSeeFace, 3tene)",
    )
    r.add_argument(
        "--thumbnail",
        metavar="PATH|auto",
        help="the thumbnail: a square PNG or JPEG file, or 'auto' for a rendered portrait",
    )
    r.add_argument(
        "--twist",
        choices=("constraint", "merge"),
        help="the forearm twist bones: roll constraints from the wrists (VRM 1.0 only, and "
        "its default) or their weights merged into the forearm bones (the VRM 0.x default); "
        "the upper arm, thigh and shin twist weights always merge",
    )
    r.add_argument(
        "--bind",
        choices=("forward", "inverse"),
        help="how the mesh moves into the T-pose (default: inverse, closest to the "
        "character's own skinning with the arms down)",
    )
    r.add_argument(
        "--keep-leg-spread",
        action="store_true",
        help="keep the rig's leg spread in the T-pose (default: legs vertical)",
    )
    r.add_argument(
        "--budget",
        choices=("strict", "warn", "off"),
        help="a file over the budget of its VRM version fails, warns (the default) or "
        "passes",
    )
    r.add_argument(
        "--anny-vertex",
        action="store_true",
        help="keep the _ANNY_VERTEX attribute (the Anny vertex of each file vertex)",
    )
    r.add_argument(
        "--refine-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="refine the skin weights for the character's body, which reduces the skin "
        "folds of the T-pose at the fingers, the groin and the neck and adds about 3 s to "
        "the export (up to 30 s on some bodies); on by default, with the inverse bind only",
    )

    p = sub.add_parser(
        "pose",
        help="pose a character as the figure in a picture, and draw it flat "
        "(needs the pose extra)",
    )
    p.add_argument("image", help="the picture (JPEG, PNG, WebP)")
    p.add_argument(
        "--character",
        help="a character card (JSON) to pose; the default character when omitted",
    )
    p.add_argument(
        "--card", help="write the posed character card (JSON); '-' prints it"
    )
    p.add_argument("--glb", help="write a glTF 2.0 binary file with the pose")
    p.add_argument("--svg", help="write the silhouette (SVG)")
    p.add_argument("--outline", help="write the line drawing (SVG)")
    p.add_argument("--png", help="write the flat-shaded picture (PNG)")
    p.add_argument(
        "--view",
        choices=("image", "front", "three-quarter"),
        default="image",
        help="the camera of the flat drawings (default: the picture's)",
    )
    p.add_argument(
        "--size",
        type=int,
        default=512,
        help="the drawings' size in pixels (default 512)",
    )
    p.add_argument(
        "--no-refine",
        action="store_true",
        help="skip the refinement against the picture",
    )
    p.add_argument("--no-hands", action="store_true", help="leave the fingers at rest")
    p.add_argument("--no-face", action="store_true", help="leave the face neutral")
    for name, what in (
        ("turn", "toward the figure's right (negative: its left)"),
        ("up", "to raise the face (negative: lower it)"),
        ("tilt", "toward the figure's right shoulder (negative: the left)"),
    ):
        p.add_argument(
            f"--head-{name}",
            type=float,
            default=0.0,
            metavar="DEG",
            help=f"turn the head further {what}, when the picture leaves it unclear",
        )

    f = sub.add_parser(
        "fit",
        help="fit a character to a reference mesh (an appearance handoff folder)",
        description="Fit anny to the skin of an appearance handoff folder (replica.glb, "
        "appearance.json with optional landmarks): the card takes the parameters and the "
        "pose, and the detail layer keeps the residuals in <card>.detail<N>.safetensors "
        "next to it. Needs SciPy (the viewer extra); see docs/reference-fit.md.",
    )
    f.add_argument("handoff", help="the handoff folder")
    f.add_argument("--out", required=True, help="the character card (JSON) to write")
    f.add_argument(
        "--detail",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="the finest level of the detail layer (default 1)",
    )
    f.add_argument("--name", help="the character's name (default: the folder's name)")
    f.add_argument(
        "--textures",
        action="store_true",
        help="bake the reference's skin (base colour, normals, roughness, thickness, "
        "curvature) onto anny's UVs and write the card's skin material (needs embreex)",
    )
    f.add_argument(
        "--texture-size",
        type=int,
        default=1024,
        help="the side of the baked textures in pixels (default 1024)",
    )
    f.add_argument(
        "--head-size",
        type=int,
        default=0,
        help="also bake the head tile, this many pixels high, and the ear tile (0: neither; "
        "4096 for the finest)",
    )
    f.add_argument(
        "--material",
        help="a JSON file of skin material values (subsurface, specular...) for the card",
    )
    f.add_argument(
        "--topology",
        choices=("anny", "sculpt"),
        default="anny",
        help="the mesh of the card: anny's, or sculpt (anny's with the patches of "
        "opensculptboy.topology; default anny)",
    )
    f.add_argument("--quiet", action="store_true", help="no progress lines")

    b = sub.add_parser(
        "bake",
        help="bake a handoff's skin onto a fitted card again, without fitting",
        description="Bake the reference of an appearance handoff folder onto a card that "
        "opensculptboy fit wrote (its offsets and placement), and write the textures and the "
        "card's skin material. Needs SciPy and embreex (the viewer extra).",
    )
    b.add_argument("handoff", help="the handoff folder")
    b.add_argument("--card", required=True, help="the fitted character card (JSON)")
    b.add_argument("--out", help="the card to write (default: the card itself)")
    b.add_argument(
        "--level",
        type=int,
        choices=(0, 1, 2),
        help="the level to bake on (default: finest)",
    )
    b.add_argument(
        "--texture-size", type=int, default=4096, help="atlas side (default 4096)"
    )
    b.add_argument(
        "--head-size",
        type=int,
        default=4096,
        help="head tile height (default 4096; 0: none)",
    )
    b.add_argument(
        "--eye-size",
        type=int,
        default=2048,
        help="eye texture width (default 2048; its height is half; 0: none)",
    )
    b.add_argument(
        "--material", help="a JSON file of skin material values for the card"
    )
    b.add_argument(
        "--no-ear-tile",
        action="store_true",
        help="keep the ears on the atlas's UVs (no ear tile for levels 1 and 2)",
    )

    c = sub.add_parser(
        "convert",
        help="convert a character card between the anny and sculpt meshes",
        description="Write a character card on another mesh (anny or sculpt): the same "
        "parameters, its detail offsets and ear tile UVs converted level by level and "
        "written next to the new card; the textures stay on anny's atlas. See "
        "docs/sculpt-topology.md.",
    )
    c.add_argument("card", help="the character card (JSON) to convert")
    c.add_argument(
        "--to", required=True, choices=("anny", "sculpt"), help="the target mesh"
    )
    c.add_argument("--out", required=True, help="the card (JSON) to write")
    c.add_argument(
        "--default-body",
        action="store_true",
        help="transfer the detail on anny's default body (default: the card's own body)",
    )

    n = sub.add_parser("names", help="list parameter, pose and clip names")
    n.add_argument(
        "kind",
        choices=(
            "poses",
            "phenotypes",
            "local-changes",
            "facial-actions",
            "face-shapes",
        ),
    )

    v = sub.add_parser(
        "viewer", help="build the web viewer (see python -m anny.viewer)"
    )
    v.add_argument("rest", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)
    if args.command == "character":
        print(json.dumps(Character().to_dict(), indent=2))
        return 0
    if args.command == "export":
        return _export(args, e)
    if args.command == "names":
        return _names(args)
    if args.command == "pose":
        return _pose(args)
    if args.command == "fit":
        return _fit(args, f)
    if args.command == "bake":
        return _bake(args, b)
    if args.command == "convert":
        return _convert(args, c)
    if args.command == "viewer":
        from anny.viewer.__main__ import main as viewer_main

        return viewer_main(args.rest)
    return 1


if __name__ == "__main__":
    sys.exit(main())

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
    opensculptboy bake-tiers me.json                 # the normal maps of the cheaper tiers
    opensculptboy export me_low.vrm --character me.json --tier low --author "Me"
    opensculptboy portraits --character me.json --out avatars/   # a portrait, five expressions
    opensculptboy viewer build                       # the web viewer (python -m anny.viewer build)
    opensculptboy export me.glb --character me.json --outfit travel --hide cape   # an outfit
    opensculptboy export me_wardrobe.glb --character me.json --wardrobe   # every piece a node
    opensculptboy wearables list --character me.json # the card's outfits and pieces
    opensculptboy wearables reference handoff/ --card me.json --out dressed/   # reference wear
    opensculptboy wearables bind tunic.glb --out wear/tunic --slot torso --licence CC0-1.0 \
        --design-source "my own design"              # a garment modelled on the default body
    opensculptboy wearables import-mhclo shirt.mhclo --out wear/shirt   # MakeHuman clothes
    opensculptboy wearables add me.json wear/shirt.wearable.json --outfit casual --activate
    opensculptboy wearables remove me.json casual/shirt --dry-run
    opensculptboy wearables outfit me.json --activate casual
    opensculptboy wearables outfit export me.json --outfit casual --out casual.outfit.json
    opensculptboy wearables add you.json casual.outfit.json --activate   # an outfit file
    opensculptboy wearables catalogue wear/ --thumbnails --slot torso   # the pieces as JSON
    opensculptboy wearables check me.json --sheet me.check.png   # the fit on test bodies
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
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
    "fit_budget",
    "fit_priority",
    "fit_triangles",
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
        return Character.from_dict(
            json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        )
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
    the eyes of opensculptboy.eyes (the card's, else the defaults); "off" keeps anny's own
    eyeballs; a file's values are laid over the card's eyes."""
    if value is None:
        return character
    import dataclasses

    from opensculptboy.material import check_eyes

    if value == "off":
        material = {**character.material, "eyes": False}
        return dataclasses.replace(character, material=material)
    eyes = character.material.get("eyes")
    eyes = dict(eyes) if isinstance(eyes, dict) else {}
    if value:
        try:
            eyes.update(json.loads(pathlib.Path(value).read_text(encoding="utf-8")))
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
            teeth.update(json.loads(pathlib.Path(value).read_text(encoding="utf-8")))
            check_teeth(teeth)
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"--teeth {value}: {_one_line(error)}")
    return dataclasses.replace(character, material={**material, "teeth": teeth})


def _dressing(args, character, parser):
    """The wearables of an export (opensculptboy.wearables.export.Dressing), or None for a
    card without wardrobe fields and no wearable options."""
    wants = (
        args.outfit is not None
        or args.show
        or args.hide
        or args.wardrobe
        or args.covered is not None
        or getattr(args, "springs", None) is not None
    )
    if not character.has_wardrobe():
        if wants:
            parser.error(
                "--outfit, --show, --hide, --covered, --springs and --wardrobe need a "
                "card with wearables or outfits (docs/wearables.md)."
            )
        return None
    from opensculptboy.wearables.export import Dressing

    outfit = args.outfit
    if getattr(args, "bare", False):
        outfit = "none"
    return Dressing(
        outfit=outfit,
        show=tuple(args.show or ()),
        hide=tuple(args.hide or ()),
        covered=args.covered or "remove",
        wardrobe=bool(args.wardrobe),
        visibility=bool(args.node_visibility),
        springs=getattr(args, "springs", None) != "off",
    )


def _export(args, parser) -> int:
    if args.tier:
        return _export_tier(args, parser)
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
    dressing = _dressing(args, character, parser)
    try:
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
            dressing=dressing,
        )
    except (OSError, ValueError) as error:  # a missing package or a wrong piece name
        if dressing is None:
            raise
        parser.error(_one_line(error))
    _result(summary)
    return 0


def _fit_options(args, parser):
    """The options of --fit-budget, or None without it."""
    if not args.fit_budget:
        if args.fit_priority or args.fit_triangles:
            parser.error("--fit-priority and --fit-triangles need --fit-budget.")
        return None
    from opensculptboy.export.fit_budget import FitOptions

    try:
        return FitOptions.parse(args.fit_priority, args.fit_triangles)
    except ValueError as error:
        parser.error(f"--fit-priority/--fit-triangles: {_one_line(error)}")


def _print_fit(summary) -> None:
    """What --fit-budget did, one line per step on stderr."""
    if summary.get("fit"):
        from opensculptboy.export.fit_budget import summary_lines

        for line in summary_lines(summary["fit"]):
            print(f"fit: {line}", file=sys.stderr)


def _export_vrm(args, parser) -> int:
    from opensculptboy.export.budget import BudgetError
    from opensculptboy.export.vrm import DEFAULT_BIND, VrmMeta, export_vrm

    meta = None
    if args.meta:
        try:
            meta = VrmMeta.from_dict(
                json.loads(pathlib.Path(args.meta).read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"--meta {args.meta}: {_one_line(error)}")
    if not args.author and not (meta and meta.authors):
        parser.error(
            "a VRM file needs an author: pass --author NAME, or --meta FILE with "
            '"authors": ["NAME"].'
        )
    character = _with_eyes(_load_character(args.character, parser), args.eyes, parser)
    character = _with_teeth(character, args.teeth, parser)
    dressing = _dressing(args, character, parser)
    if dressing is not None and dressing.wardrobe:
        parser.error("--wardrobe writes .glb files; a VRM file wears one outfit.")
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
                dressing=dressing,
                fit_budget=_fit_options(args, parser),
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
    _print_fit(summary)
    _result(summary)
    return 0


def _export_tier(args, parser) -> int:
    """``export --tier``: the tier's mesh, textures and defaults (opensculptboy.tiers)."""
    from opensculptboy.export.budget import BudgetError
    from opensculptboy.tiers import export_tier, get_tier

    tier = get_tier(args.tier)
    output = pathlib.Path(args.output)
    is_vrm = output.suffix.lower() == ".vrm" or (
        not output.suffix and tier.format == "vrm"
    )
    if args.detail is not None:
        parser.error(
            "--detail: the tier sets the detail level; leave out one of the two."
        )
    wrong = _GLB_ONLY if is_vrm else _VRM_ONLY
    given = [_flag(d) for d in wrong if getattr(args, d) not in (None, False)]
    if given:
        kind = ".vrm" if is_vrm else ".glb"
        parser.error(
            f"{', '.join(given)}: not for the {kind} file of the {tier.name} tier."
        )
    character = _with_eyes(_load_character(args.character, parser), args.eyes, parser)
    character = _with_teeth(character, args.teeth, parser)
    kwargs = {}
    if is_vrm:
        from opensculptboy.export.vrm import VrmMeta

        meta = None
        if args.meta:
            try:
                meta = VrmMeta.from_dict(
                    json.loads(pathlib.Path(args.meta).read_text(encoding="utf-8"))
                )
            except (OSError, ValueError, TypeError) as error:
                parser.error(f"--meta {args.meta}: {_one_line(error)}")
        if not args.author and not (meta and meta.authors):
            parser.error(
                "a VRM file needs an author: pass --author NAME, or --meta FILE with "
                '"authors": ["NAME"].'
            )
        kwargs = dict(
            meta=meta,
            author=args.author,
            name=args.name,
            thumbnail=args.thumbnail,
            bare=args.bare,
            twist=args.twist,
            keep_leg_spread=args.keep_leg_spread,
            budget=args.budget or "warn",
            keep_anny_vertex=args.anny_vertex,
            refine_weights=args.refine_weights is not False,
            fit_budget=_fit_options(args, parser),
        )
        if args.bind:
            kwargs["bind"] = args.bind
        if args.vrm_version:
            kwargs["version"] = {"1": "1.0", "0": "0.x"}[args.vrm_version]
    else:
        kwargs = dict(animations=args.animation or (), ground=not args.no_ground)
        if args.influences:
            kwargs["max_influences"] = args.influences
        if args.morph_targets == "none":
            kwargs["morph_targets"] = []
        elif args.morph_targets == "all":
            model = character.build_model(face_shapes=True)
            kwargs["model"] = model
            kwargs["morph_targets"] = list(model.facial_action_labels) + list(
                model.face_shape_labels
            )
        elif args.morph_targets not in (None, "facial"):
            kwargs["morph_targets"] = [
                n.strip() for n in args.morph_targets.split(",") if n.strip()
            ]
    dressing = _dressing(args, character, parser)
    if dressing is not None:
        if is_vrm and dressing.wardrobe:
            parser.error("--wardrobe writes .glb files; a VRM file wears one outfit.")
        kwargs["dressing"] = dressing
    failure = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            summary = export_tier(
                output,
                character,
                tier,
                detail_dir=_detail_dir(args.character),
                format="vrm" if is_vrm else "glb",
                **kwargs,
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
    _print_fit(summary)
    _result(summary)
    return 0


def _bake_tiers(args, parser) -> int:
    from opensculptboy.tiers import TIERS, bake_tiers

    names = [n.strip() for n in (args.tiers or "high,medium,low").split(",") if n]
    unknown = [n for n in names if n not in TIERS]
    if unknown:
        parser.error(f"--tiers: unknown tiers {unknown}; the tiers are {list(TIERS)}.")
    try:
        _, report = bake_tiers(
            args.card,
            names,
            out=args.out,
            rays=args.rays,
            progress=None if args.quiet else lambda m: print(m, file=sys.stderr),
        )
    except (OSError, ValueError) as error:
        parser.error(_one_line(error))
    _result(report)
    return 0


def _portraits(args, parser) -> int:
    from opensculptboy.render.sheet import write_portraits

    character = _load_character(args.character, parser)
    written = write_portraits(
        args.out, character, args.size, detail_dir=_detail_dir(args.character)
    )
    _result(written)
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
            _result(text)
        else:
            pathlib.Path(args.card).write_text(text + "\n", encoding="utf-8")
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
            pathlib.Path(args.svg).write_text(text, encoding="utf-8")
            written.append(args.svg)
        if args.outline:
            text = outline_svg(vertices, faces, view, size=size, desc=desc)
            pathlib.Path(args.outline).write_text(text, encoding="utf-8")
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
    _write_mouth_poses(out, handoff, reference)
    _write_eye_poses(out, handoff, reference)
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
    _result(summary)
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
    if getattr(baked, "lashes", None) is not None:  # the reference's lashes
        from opensculptboy.export.document import png_bytes

        path = card.parent / f"{stem}.lashes.safetensors"
        baked.lashes.save(path)
        names["lashes"] = path.name
        if baked.lash_texture is not None:
            path = card.parent / f"{stem}.lashes.png"
            path.write_bytes(png_bytes(baked.lash_texture))
            names["lashes_base_color"] = path.name
    if getattr(baked, "mouth", None) is not None:  # the reference's inner mouth
        path = card.parent / f"{stem}.mouth.safetensors"
        baked.mouth.save(path)
        names["mouth"] = path.name
        for slot, source in baked.mouth_textures.items():
            path = card.parent / f"{stem}.mouth_{slot}.png"
            path.write_bytes(pathlib.Path(source).read_bytes())
            names[f"mouth_{slot}"] = path.name
    textures = {"subdiv": subdiv}
    for slot in ("eyes", "eyes_normal", "eye_layout", "eye_layout_normal"):
        if slot in names:
            textures[slot] = names[slot]
    return textures, baked.material(names)


def _write_mouth_poses(card: pathlib.Path, handoff, reference=None) -> None:
    """Write a fitted card's mouth poses from its handoff (``opensculptboy.mouth.poses``)
    when the card has a reference's inner mouth and the handoff moving mouth poses."""
    from opensculptboy.mouth.poses import write_card_poses

    path = write_card_poses(card, handoff, reference)
    if path is not None:
        print(f"mouth poses: {path.name}", file=sys.stderr)


def _write_eye_poses(card: pathlib.Path, handoff, reference=None) -> None:
    """Write a fitted card's blinks from its handoff (``opensculptboy.eyes.poses``) when
    the handoff has its reference's closed eyes."""
    from opensculptboy.eyes.poses import write_card_eye_poses

    path = write_card_eye_poses(card, handoff, reference)
    if path is not None:
        print(f"eye poses: {path.name}", file=sys.stderr)


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
            extra = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
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
    out.write_text(json.dumps(character.to_dict(), indent=2) + "\n", encoding="utf-8")
    from opensculptboy.fit.handoff import load_handoff

    handoff = load_handoff(args.handoff)
    if "mouth" in (character.material or {}):
        _write_mouth_poses(out, handoff)
    _write_eye_poses(out, handoff)
    _result({"card": str(out), "covered": round(baked.covered_share, 4), **textures})
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
    _result(summary)
    return 0


def _wearables(args, parser) -> int:
    if args.wearables_command == "list":
        character = _load_character(args.character, parser)
        from opensculptboy.wearables.wardrobe import worn_pieces

        try:
            pieces = worn_pieces(
                character, _detail_dir(args.character), every_outfit=True
            )
        except (OSError, ValueError) as error:
            parser.error(_one_line(error))
        listing = {
            "active_outfit": character.active_outfit or None,
            "outfits": [
                {"id": o["id"], "label": o.get("label", o["id"])}
                for o in character.outfits
            ],
            "pieces": [
                {
                    "ref": p.ref,
                    "label": p.wearable.label or p.wearable.name,
                    "slot": p.wearable.slot,
                    "layer": p.wearable.layer,
                    "provenance": p.wearable.provenance,
                    "shown": p.shown,
                    "groups": list(p.wearable.garment.groups),
                    "hidden_groups": p.hidden_groups,
                    "hides": p.wearable.hides,
                }
                for p in pieces
            ],
        }
        _result(listing)
        return 0
    if args.wearables_command == "bind":
        return _wearables_bind(args, parser)
    if args.wearables_command == "import-mhclo":
        return _wearables_import(args)
    if args.wearables_command in ("add", "remove", "outfit", "outfit-export"):
        return _wearables_card(args)
    if args.wearables_command == "catalogue":
        return _wearables_catalogue(args)
    if args.wearables_command == "check":
        return _wearables_check(args)
    from opensculptboy.wearables.format import ProvenanceError
    from opensculptboy.wearables.reference import build_reference_wearables

    try:
        report = build_reference_wearables(
            args.handoff,
            args.card,
            args.out,
            args.outfit_id,
            args.outfit_label,
            log=lambda m: print(m, file=sys.stderr),
        )
    except (OSError, ValueError, ProvenanceError) as error:
        parser.error(_one_line(error))
    _result(report)
    return 0


def _wearables_bind(args, parser) -> int:
    import torch

    from opensculptboy.wearables.fit import card_body
    from opensculptboy.wearables.format import (
        ProvenanceError,
        check_destination,
        package_paths,
        write_wearable,
    )
    from opensculptboy.wearables.make import make_wearable, round_trip

    paths = package_paths(args.out)
    name = args.name or paths["stem"].name
    licence = {
        "spdx": args.licence,
        "author": args.author,
        "source": args.source,
        "design_source": args.design_source,
    }
    try:
        check_destination(paths["json"], args.provenance)
    except ProvenanceError as error:
        parser.error(_one_line(error))
    character = _load_character(args.body, parser)
    detail_dir = _detail_dir(args.body)
    print("binding to the body ...", file=sys.stderr)
    body = card_body(character, character.build_model(dtype=torch.float64), detail_dir)
    extras = {"source": "opensculptboy wearables bind"}
    if args.body:
        extras["body_card"] = pathlib.Path(args.body).name
    try:
        wearable, report = make_wearable(
            args.garment,
            body,
            name,
            args.slot,
            args.layer,
            label=args.label,
            rigid=args.rigid,
            weights=args.weights,
            groups_from=args.groups_from,
            covers=args.covers,
            hides=args.hides,
            licence=licence,
            provenance=args.provenance,
            body_kind="card" if args.body else "anny-template",
            frame=args.frame,
            max_distance=args.max_distance / 1000.0,
            extras=extras,
            tags=args.tag or (),
            description=args.description or "",
            texture_limit=args.texture_limit,
            normal_green=args.normal_green,
            accept_quality=args.accept_quality,
        )
        vertices = wearable.garment.vertices.copy()
        path = write_wearable(paths["json"], wearable)
        report["round_trip_mm_max"] = round_trip(path, body, vertices)
    except (OSError, ValueError, ProvenanceError) as error:
        parser.error(_one_line(error))
    report["package"] = str(path)
    for warning in report["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    _print_quality(report.get("quality"))
    _result(report)
    return 0


def _print_quality(quality) -> None:
    """The warning- and error-severity quality items on the error stream."""
    for item in (quality or {}).get("items", []):
        if item["severity"] != "info":
            print(
                f"{item['severity']}: [{item['code']}] {item['message']}",
                file=sys.stderr,
            )


# The wearables commands whose results and errors are JSON on stdout.
JSON_COMMANDS = (
    "import-mhclo",
    "add",
    "remove",
    "outfit",
    "outfit-export",
    "catalogue",
    "check",
)

# The results of the running command, printed on stdout when it ends (_results_alone).
_RESULTS: list[str] | None = None


def _result(value) -> None:
    """Print a command's result on stdout: a JSON value (indented), or text. While a command
    runs inside :func:`_results_alone`, the result waits for the command's end."""
    text = value if isinstance(value, str) else json.dumps(value, indent=2)
    if _RESULTS is None:
        print(text)
    else:
        _RESULTS.append(text)


def _flush_native() -> None:
    """Flush the C library's stdout buffer (native code that prints with printf)."""
    try:
        import ctypes

        ctypes.CDLL(None).fflush(None)
    except Exception:  # noqa: BLE001 - a platform without it has nothing to flush
        pass


@contextlib.contextmanager
def _results_alone():
    """
    Keep stdout for the command's results (:func:`_result`): while the command runs,
    everything else written to stdout goes to stderr, from Python (``print``, as Warp's
    start-up lines and module loads) and from native code (file descriptor 1); the results
    follow on stdout when the command ends, also when it fails.
    """
    global _RESULTS
    target = sys.stdout
    saved = None
    for stream in (target, sys.__stdout__):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass
    _flush_native()
    try:
        saved = os.dup(1)
        os.dup2(2, 1)
    except OSError:
        saved = None
    _RESULTS = []
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        results, _RESULTS = _RESULTS, None
        for stream in (sys.__stdout__, sys.stderr):
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                pass
        _flush_native()
        if saved is not None:
            os.dup2(saved, 1)
            os.close(saved)
        for text in results:
            print(text, file=target)
        try:
            target.flush()
        except (AttributeError, OSError, ValueError):
            pass


# The hints of the errors that carry none of their own.
_ERROR_HINTS = {
    "file_error": "Check that the path exists and can be read and written.",
    "invalid_input": "Check the files and the options; --help lists the options.",
    "internal_error": (
        "An unexpected error: report it with the command line and this message."
    ),
}


def _json_error(error: BaseException, command: str, internal: bool = False) -> int:
    """Print a failed command's JSON result (``code``, ``message``, ``hint``); returns 1, or
    2 for an unexpected error (``internal``, code ``internal_error``)."""
    from opensculptboy.wearables.format import WearableError

    if internal:
        code, hint = "internal_error", ""
    else:
        code = (
            getattr(error, "code", None) if isinstance(error, WearableError) else None
        )
        code = code or ("file_error" if isinstance(error, OSError) else "invalid_input")
        hint = getattr(error, "hint", "") if isinstance(error, WearableError) else ""
    hint = hint or _ERROR_HINTS.get(code, _ERROR_HINTS["invalid_input"])
    message = _one_line(error)
    if internal:
        message = f"{type(error).__name__}: {message}"
    _result(
        {
            "ok": False,
            "command": command,
            "error": {"code": code, "message": message, "hint": hint},
        }
    )
    print(f"error: [{code}] {message}", file=sys.stderr)
    return 2 if internal else 1


def _json_usage(parser, command: str):
    """Make a subcommand's usage errors JSON results (code ``usage``) on stdout."""

    def error(message):
        _result(
            {
                "ok": False,
                "command": command,
                "error": {
                    "code": "usage",
                    "message": message,
                    "hint": f"See 'opensculptboy {command} --help'.",
                },
            }
        )
        print(f"{parser.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(2)

    parser.error = error


def _wearables_import(args) -> int:
    command = "wearables import-mhclo"
    try:
        report = _import_report(args)
    except (OSError, ValueError, KeyError) as error:
        return _json_error(error, command)
    except Exception as error:  # one JSON error for every failure, never a traceback
        return _json_error(error, command, internal=True)
    report = {"ok": True, "command": command, **report}
    for warning in report["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    _print_quality(report.get("quality"))
    _result(report)
    return 0


def _import_report(args) -> dict:
    import torch

    from opensculptboy.wearables.fit import card_body
    from opensculptboy.wearables.format import package_paths, write_wearable
    from opensculptboy.wearables.make import round_trip
    from opensculptboy.wearables.mhclo import import_mhclo

    source = pathlib.Path(args.mhclo)
    paths = package_paths(args.out or str(source.with_suffix("")))
    character = Character()
    model = character.build_model(dtype=torch.float64)
    print("binding to the default body ...", file=sys.stderr)
    body = card_body(character, model)
    wearable, report = import_mhclo(
        source,
        body,
        character,
        model,
        slot=args.slot,
        layer=args.layer,
        name=args.name,
        label=args.label,
        tags=args.tag or (),
        description=args.description,
        author=args.author,
        source=args.source,
        covers=args.covers,
        hides=args.hides,
        max_distance=None if args.max_distance is None else args.max_distance / 1000,
        texture_limit=args.texture_limit,
        normal_green=args.normal_green,
        accept_quality=args.accept_quality,
        pack_json=args.pack_json,
        bump_strength=args.bump_strength,
    )
    if args.dry_run:  # the checks and the binding, without the files
        return {
            "package": str(paths["json"]),
            "dry_run": True,
            "written": False,
            **report,
        }
    vertices = wearable.garment.vertices.copy()
    path = write_wearable(paths["json"], wearable)
    report["round_trip_mm_max"] = round_trip(path, body, vertices)
    return {"package": str(path), "dry_run": False, "written": True, **report}


def _wearables_card(args) -> int:
    from opensculptboy.wearables import cardedit

    command = f"wearables {args.wearables_command}"
    try:
        result = _card_change(args, cardedit)
    except (OSError, ValueError, KeyError) as error:
        return _json_error(error, command)
    except Exception as error:  # one JSON error for every failure, never a traceback
        return _json_error(error, command, internal=True)
    for warning in result.get("warnings", []):
        print(f"warning: {warning}", file=sys.stderr)
    _result(result)
    return 0


def _card_change(args, cardedit) -> dict:
    from opensculptboy.wearables import outfitfile

    if args.wearables_command == "outfit-export":
        return outfitfile.export_outfit(
            args.card,
            args.outfit,
            args.out,
            label=args.label,
            description=args.description,
            tags=_tag_words(args.tag),
            replace=args.replace,
            dry_run=args.dry_run,
        )
    if args.wearables_command == "add" and any(
        outfitfile.is_outfit_file(p) for p in args.packages
    ):
        if len(args.packages) != 1 or args.loose:
            raise cardedit.CardError(
                "An outfit file goes on a card alone.",
                "bad_option",
                "Pass one .outfit.json file, without packages and without --loose.",
            )
        return outfitfile.add_outfit(
            args.card,
            args.packages[0],
            outfit=args.outfit,
            label=args.label,
            activate=args.activate,
            replace=args.replace,
            dry_run=args.dry_run,
        )
    if args.wearables_command == "add":
        return cardedit.add_pieces(
            args.card,
            args.packages,
            outfit=args.outfit,
            label=args.label,
            activate=args.activate,
            loose=args.loose,
            replace=args.replace,
            dry_run=args.dry_run,
        )
    if args.wearables_command == "remove":
        if not args.refs and args.outfit is None:
            raise cardedit.CardError(
                "Nothing to remove.",
                "bad_option",
                "Name pieces as OUTFIT/PIECE or loose ids, or pass --outfit NAME.",
            )
        return cardedit.remove_pieces(
            args.card, args.refs, outfit=args.outfit, dry_run=args.dry_run
        )
    return cardedit.activate_outfit(args.card, args.activate, dry_run=args.dry_run)


def _tag_words(values) -> list[str]:
    """The words of repeatable, comma-separated options (``--tag a,b --tag c``)."""
    return [w.strip() for v in values or () for w in v.split(",") if w.strip()]


def _wearables_catalogue(args) -> int:
    from opensculptboy.wearables.catalogue import catalogue

    command = "wearables catalogue"
    try:
        result = catalogue(
            args.folder,
            mhclo_folders=args.mhclo or (),
            slots=_tag_words(args.slot),
            tags=_tag_words(args.tag),
            query=[w for q in args.query or () for w in q.split()],
            quality=not args.no_quality,
            thumbnails=args.thumbnails,
            thumbnail_size=args.thumbnail_size,
            refresh=args.refresh,
            log=lambda m: print(m, file=sys.stderr),
        )
    except (OSError, ValueError, KeyError) as error:
        return _json_error(error, command)
    except Exception as error:  # one JSON error for every failure, never a traceback
        return _json_error(error, command, internal=True)
    _result({"ok": True, "command": command, **result})
    return 0


def _wearables_check(args) -> int:
    """0 when every gate passes, 1 when a gate fails, 2 on an error."""
    from opensculptboy.wearables.check import BODIES, check_card

    command = "wearables check"
    card = pathlib.Path(args.card)
    sheet = args.sheet
    if sheet is None:
        sheet = card.with_name(f"{card.stem}.check.png")
    elif sheet == "none":
        sheet = None
    clips = None
    if args.clips is not None:
        clips = [] if args.clips == "none" else _tag_words([args.clips])
    try:
        report = check_card(
            card,
            outfit=args.outfit,
            bodies=_tag_words([args.bodies]) if args.bodies else BODIES,
            clips=clips,
            frames=args.frames,
            tolerance=args.tolerance / 1000.0,
            pose_limit=args.pose_limit,
            sheet=sheet,
            tpose=not args.no_tpose,
            log=lambda m: print(m, file=sys.stderr),
        )
    except (OSError, ValueError, KeyError) as error:
        _json_error(error, command)
        return 2
    except Exception as error:  # one JSON error for every failure, never a traceback
        return _json_error(error, command, internal=True)
    for gate in report["gates"]:
        if not gate["pass"]:
            print(f"gate failed: {gate['name']} {gate.get('worst')}", file=sys.stderr)
    _result({"ok": True, "command": command, **report})
    return 0 if report["pass"] else 1


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


def _package_options(p) -> None:
    p.add_argument(
        "--tag",
        action="append",
        metavar="WORDS",
        help="free words for the package (repeatable, or comma-separated)",
    )
    p.add_argument("--description", help="one line about the package")


def _quality_options(p) -> None:
    from opensculptboy.wearables.quality import NORMAL_GREEN, TEXTURE_LIMIT

    p.add_argument(
        "--accept-quality",
        action="store_true",
        help="write the package even with error-severity quality items",
    )
    p.add_argument(
        "--texture-limit",
        type=int,
        default=TEXTURE_LIMIT,
        metavar="PIXELS",
        help=f"the largest texture side without a warning (default {TEXTURE_LIMIT})",
    )
    p.add_argument(
        "--normal-green",
        choices=NORMAL_GREEN,
        default="auto",
        help="the normal map's green channel: auto flips a DirectX map to OpenGL "
        "(default), keep leaves it, flip always flips",
    )


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
        "(the card's material.eyes, else the defaults; a fitted card has them without the "
        "option); FILE, a JSON object of eye values (docs/eyes.md), is laid over them, and "
        "'off' keeps anny's eyeballs",
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
    e.add_argument(
        "--tier",
        choices=("master", "high", "medium", "low"),
        help="a rendering-budget tier (docs/budget-tiers.md): it sets the detail level, the "
        "texture sizes and maps, the morph targets and the skin weights per vertex, and an "
        "output without a suffix takes the tier's format (.glb, or .vrm for low)",
    )
    w = e.add_argument_group("wearables (docs/wearables.md)")
    w.add_argument(
        "--outfit",
        metavar="ID",
        help="the outfit to wear (default: the card's active_outfit; 'none' for none)",
    )
    w.add_argument(
        "--show",
        action="append",
        metavar="PIECE",
        help="show a piece: a piece of the outfit, a loose piece, or OUTFIT/PIECE of another "
        "outfit, which replaces the pieces of its slot; repeat for more",
    )
    w.add_argument(
        "--hide",
        action="append",
        metavar="PIECE",
        help="hide a piece; the groups it hides on other pieces come back; repeat for more",
    )
    w.add_argument(
        "--covered",
        choices=("remove", "keep"),
        help="the body triangles that the shown pieces cover: 'remove' (the default) or "
        "'keep'",
    )
    w.add_argument(
        "--springs",
        choices=("on", "off"),
        help="the spring chains of loose pieces (capes, skirts, long hems): 'on' (the "
        "default) or 'off', which leaves every piece on the body's bones",
    )
    w.add_argument(
        "--wardrobe",
        action="store_true",
        help="a .glb file with every piece of every outfit as its own node and the body "
        "whole: the default scene shows the worn outfit, a second scene holds every piece",
    )
    w.add_argument(
        "--node-visibility",
        action="store_true",
        help="with --wardrobe: one scene with every piece, the pieces that do not show "
        "marked hidden with KHR_node_visibility",
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
        "--fit-budget",
        action="store_true",
        help="fit the file into the budget of its VRM version: reduce the pieces (the "
        "body's face and hands last), bake their normal maps again, merge the wearables' "
        "materials into atlases and shrink the textures, and print what was done "
        "(docs/vrm.md)",
    )
    r.add_argument(
        "--fit-priority",
        action="append",
        metavar="PIECE=FACTOR",
        help="with --fit-budget: a factor on a piece's importance (body, eyes, lashes, "
        "teeth, mouth, wearable or a piece's reference such as travel/cape); repeatable",
    )
    r.add_argument(
        "--fit-triangles",
        action="append",
        metavar="PIECE=COUNT",
        help="with --fit-budget: a fixed number of triangles for a piece; repeatable",
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
        "parameters, its detail offsets, ear tile UVs, eye poses and mouth poses converted "
        "level by level and written next to the new card; the textures stay on anny's "
        "atlas. See docs/sculpt-topology.md.",
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

    t = sub.add_parser(
        "bake-tiers",
        help="bake the normal maps of the cheaper tiers from a card's master",
        description="Bake the master's surface (the finest level of the card's detail layer "
        "and its normal maps) into normal maps for the meshes of the cheaper tiers, and the "
        "occlusion, thickness and curvature maps that the tiers keep and the card lacks; "
        "write them next to the card and record them in its skin material. Needs SciPy and "
        "embreex (the viewer extra).",
    )
    t.add_argument("card", help="the character card (JSON) with its detail layer")
    t.add_argument(
        "--tiers", help="comma-separated tiers to bake (default: high,medium,low)"
    )
    t.add_argument(
        "--out", help="the card to write, in the same folder (default: the card itself)"
    )
    t.add_argument(
        "--rays",
        type=int,
        default=32,
        help="occlusion rays per vertex (default 32)",
    )
    t.add_argument("--quiet", action="store_true", help="no progress lines")

    q = sub.add_parser(
        "portraits",
        help="render a portrait and an expression sheet for chat avatars",
        description="Render the character's head and shoulders in flat toon shading: "
        "portrait.png, one picture per expression (neutral, smile, surprised, sad, angry) "
        "and expressions.png, the sheet of all five.",
    )
    q.add_argument("--character", help="a character card (JSON); the default character")
    q.add_argument("--out", required=True, help="the folder to write the pictures into")
    q.add_argument(
        "--size", type=int, default=512, help="side of each expression (default 512)"
    )

    wear = sub.add_parser(
        "wearables",
        help="list and change a card's outfits, bind a garment, import MakeHuman clothes, "
        "or build reference wearables",
    )
    wear_sub = wear.add_subparsers(dest="wearables_command", required=True)
    wl = wear_sub.add_parser("list", help="print a card's outfits and pieces")
    wl.add_argument("--character", required=True, help="a character card (JSON)")
    wr = wear_sub.add_parser(
        "reference",
        help="build wearable packages from a handoff's equipment for a fitted card",
        description="Build the reference wearables of an appearance handoff's equipment "
        "section for a card fitted to that handoff: one package per piece in "
        "OUT/wearables/, marked private, and a copy of the card in OUT that lists the "
        "outfits. OUT must lie outside the repository.",
    )
    wr.add_argument("handoff", help="the handoff folder (replica.glb, appearance.json)")
    wr.add_argument("--card", required=True, help="the card fitted to that handoff")
    wr.add_argument("--out", required=True, help="the folder to write into")
    wr.add_argument(
        "--outfit-id", help="the id of the outfit of a section without outfits"
    )
    wr.add_argument("--outfit-label", help="its label")
    from opensculptboy.wearables.format import PROVENANCES, SLOTS
    from opensculptboy.wearables.make import COVERS, FRAMES, GROUPS_FROM, MAX_DISTANCE

    wb = wear_sub.add_parser(
        "bind",
        help="make a wearable package from a garment modelled on the default body",
        description="Bind the garment meshes of a GLB file to a body and write the "
        "package OUT.wearable.json, OUT.glb and OUT.binding.safetensors. The garment "
        "sits on the body of 'opensculptboy export base.glb' at rest (metres, glTF axes, "
        "the body on the floor or centred); see docs/wearables.md, 'Make a wearable'.",
    )
    wb.add_argument("garment", help="the garment as a GLB file")
    wb.add_argument("--out", required=True, help="the package's stem, e.g. wear/tunic")
    wb.add_argument("--slot", required=True, choices=SLOTS, help="the piece's slot")
    wb.add_argument(
        "--layer",
        type=int,
        default=50,
        help="0 (skin) to 100 (outermost); torso 50, cloaks 80 (default 50)",
    )
    wb.add_argument("--name", help="the package's name (default: the stem of --out)")
    wb.add_argument("--label", help="the name to show (default: the name)")
    wb.add_argument(
        "--rigid",
        metavar="BONE",
        help="bind every vertex to one bone (a circlet, a buckle); else surface binding",
    )
    wb.add_argument(
        "--weights",
        choices=("body", "own"),
        default="body",
        help="body: the body's skin weights at the bound points (default); own: the "
        "file's JOINTS_0 and WEIGHTS_0, by joint name",
    )
    wb.add_argument(
        "--groups-from",
        choices=GROUPS_FROM,
        default="materials",
        help="name the groups by material (default), by mesh node, or by the 'group' "
        "of the extras of the primitive, mesh or node",
    )
    wb.add_argument(
        "--covers",
        choices=COVERS,
        default="auto",
        help="auto: the skin under each group by a ray test, kept where the dressed "
        "piece meets the ray (default); none: the piece covers no skin",
    )
    wb.add_argument(
        "--hides",
        action="append",
        metavar="PIECE:GROUP,...",
        help="groups of other pieces to hide while this one shows (repeatable)",
    )
    wb.add_argument("--licence", metavar="SPDX", help="the SPDX licence id, recorded")
    wb.add_argument("--author", help="the author of the garment")
    wb.add_argument("--source", help="where the garment's files come from (a URL)")
    wb.add_argument("--design-source", help="where the design comes from")
    wb.add_argument(
        "--provenance",
        choices=PROVENANCES,
        default="public",
        help="public, or private (made from someone's own files; never inside the "
        "repository)",
    )
    _package_options(wb)
    _quality_options(wb)
    wb.add_argument(
        "--body",
        metavar="CARD",
        help="bind to the body of this card (default: the default card)",
    )
    wb.add_argument(
        "--frame",
        choices=FRAMES,
        default="auto",
        help="floor: the body stands on the floor, as an export shows at its default "
        "pose; centred: the body at Anny's origin (the export's bind pose, or "
        "--no-ground); auto: whichever puts the garment nearer the skin (default)",
    )
    wb.add_argument(
        "--max-distance",
        type=float,
        default=MAX_DISTANCE * 1000,
        metavar="MM",
        help="refuse a garment whose median distance from the skin (the nearest vertex "
        f"for --rigid) exceeds this (default {MAX_DISTANCE * 1000:.0f})",
    )

    wi = wear_sub.add_parser(
        "import-mhclo",
        help="make a wearable package from a MakeHuman clothes asset (.mhclo)",
        description="Read a MakeHuman clothes asset (the .mhclo file, its .obj mesh and "
        "its .mhmat material with the textures), bind it to the default body and write "
        "the package OUT.wearable.json, OUT.glb and OUT.binding.safetensors. Prints one "
        "JSON result; see docs/wearables.md, 'Free outfits'.",
    )
    wi.add_argument("mhclo", help="the .mhclo file")
    wi.add_argument(
        "--out",
        help="the package's stem, e.g. wear/shirt (default: next to the .mhclo)",
    )
    wi.add_argument(
        "--slot", help=f"the piece's slot ({', '.join(SLOTS)}); default: from the name"
    )
    wi.add_argument(
        "--layer", type=int, help="0 (skin) to 100 (outermost); default: the z_depth"
    )
    wi.add_argument("--name", help="the package's name (default: the asset's name)")
    wi.add_argument("--label", help="the name to show (default: the asset's name)")
    wi.add_argument("--author", help="the author, in place of the header's")
    wi.add_argument("--source", help="where the asset comes from (a URL)")
    wi.add_argument(
        "--pack-json",
        metavar="FILE",
        help="the asset pack's JSON file (packs/<pack>.json) with the item's metadata "
        "(default: found above the .mhclo, in the folder that holds clothes/ and packs/)",
    )
    from opensculptboy.wearables.mhclo import DEFAULT_BUMP_STRENGTH, IMPORT_COVERS

    wi.add_argument(
        "--bump-strength",
        type=float,
        metavar="TEXELS",
        help="the relief of a bump map converted to a normal map: a step from black to "
        f"white rises this many texels (default {DEFAULT_BUMP_STRENGTH:g}, times the "
        "material's bumpmapIntensity)",
    )

    wi.add_argument(
        "--covers",
        choices=IMPORT_COVERS,
        default="auto",
        help="auto: the skin under delete_verts, else a ray test (default); ray: the ray "
        "test; none: no covered skin",
    )
    wi.add_argument(
        "--hides",
        action="append",
        metavar="PIECE:GROUP,...",
        help="groups of other pieces to hide while this one shows (repeatable)",
    )
    wi.add_argument(
        "--max-distance",
        type=float,
        metavar="MM",
        help=f"refuse a garment whose median distance from the skin exceeds this "
        f"(default {MAX_DISTANCE * 1000:.0f})",
    )
    wi.add_argument(
        "--dry-run",
        action="store_true",
        help="check and bind the asset, print the result and write no files",
    )
    _package_options(wi)
    _quality_options(wi)
    _json_usage(wi, "wearables import-mhclo")

    wa = wear_sub.add_parser(
        "add",
        help="put wearable packages on a card, in an outfit or as loose pieces",
        description="Add packages to an outfit of a card (created when missing) or as "
        "loose pieces, or put an outfit file (.outfit.json) on the card as an outfit, "
        "with paths relative to the card. An older card is raised to schema 4; the "
        "card's other fields stay as they are. Prints one JSON result.",
    )
    wa.add_argument("card", help="the character card (JSON)")
    wa.add_argument(
        "packages",
        nargs="+",
        help="the packages' .wearable.json files, or one .outfit.json file",
    )
    wa.add_argument(
        "--outfit",
        help="the outfit to add to (created when missing); for an outfit file, the id "
        "on the card (default: the file's id)",
    )
    wa.add_argument("--label", help="the label of a new outfit")
    wa.add_argument(
        "--activate", action="store_true", help="make --outfit the worn one"
    )
    wa.add_argument("--loose", action="store_true", help="add as loose pieces")
    wa.add_argument(
        "--replace",
        action="store_true",
        help="replace a piece of the same id, and the pieces of the same slot",
    )
    wa.add_argument("--dry-run", action="store_true", help="print the change only")
    _json_usage(wa, "wearables add")
    wd = wear_sub.add_parser(
        "remove",
        help="take pieces or an outfit off a card",
        description="Take pieces off a card by reference (OUTFIT/PIECE, a loose id, or a "
        "piece of the worn outfit), or a whole outfit with --outfit. Prints one JSON "
        "result.",
    )
    wd.add_argument("card", help="the character card (JSON)")
    wd.add_argument("refs", nargs="*", help="the pieces to take off")
    wd.add_argument("--outfit", help="take this whole outfit off")
    wd.add_argument("--dry-run", action="store_true", help="print the change only")
    _json_usage(wd, "wearables remove")
    wo = wear_sub.add_parser(
        "outfit",
        help="choose the worn outfit of a card",
        description="Set the card's active_outfit. Prints one JSON result.",
    )
    wo.add_argument("card", help="the character card (JSON)")
    wo.add_argument(
        "--activate", required=True, metavar="NAME", help="the outfit, or 'none'"
    )
    wo.add_argument("--dry-run", action="store_true", help="print the change only")
    _json_usage(wo, "wearables outfit")
    wx = wear_sub.add_parser(
        "outfit-export",
        prog="opensculptboy wearables outfit export",
        help="write an outfit of a card as an outfit file (also: wearables outfit export)",
        description="Write one outfit of a card as an outfit file (opensculptboy.outfit/1): "
        "its pieces with paths relative to the file, their overrides, hides rules and "
        "springs, and the credits. An existing file with other contents stays unless "
        "--replace. Prints one JSON result; see docs/wearables.md, 'Outfit files'.",
    )
    wx.add_argument("card", help="the character card (JSON)")
    wx.add_argument("--outfit", required=True, help="the outfit to write")
    wx.add_argument("--out", required=True, help="the outfit file, NAME.outfit.json")
    wx.add_argument("--label", help="the outfit's label (default: the card's)")
    wx.add_argument("--description", help="one line about the outfit")
    wx.add_argument(
        "--tag", action="append", metavar="WORDS", help="words for the outfit"
    )
    wx.add_argument(
        "--replace", action="store_true", help="write over a file with other contents"
    )
    wx.add_argument("--dry-run", action="store_true", help="print the file only")
    _json_usage(wx, "wearables outfit export")

    wc = wear_sub.add_parser(
        "catalogue",
        help="list the wearable packages of a folder as JSON, with thumbnails",
        description="Walk a folder for wearable packages (and with --mhclo for MakeHuman "
        "clothes assets not imported yet) and print one JSON result whose 'pieces' list "
        "gives each piece's name, label, slot, layer, tags, description, colours, groups, "
        "counts, licence, quality summary and thumbnail. See docs/wearables.md, "
        "'Catalogue'.",
    )
    wc.add_argument("folder", help="the folder of packages (searched with sub-folders)")
    wc.add_argument(
        "--mhclo",
        action="append",
        metavar="FOLDER",
        help="also list the .mhclo assets of FOLDER that no package was imported from "
        "(repeatable)",
    )
    wc.add_argument(
        "--slot", action="append", metavar="SLOT", help="keep these slots (repeatable)"
    )
    wc.add_argument(
        "--tag",
        action="append",
        metavar="WORDS",
        help="keep pieces with every one of these tags (repeatable, or comma-separated)",
    )
    wc.add_argument(
        "--query",
        action="append",
        metavar="WORDS",
        help="keep pieces whose name, label, tags or description hold every word",
    )
    wc.add_argument(
        "--thumbnails",
        action="store_true",
        help="render missing or stale thumbnails next to the packages (NAME.thumb.png)",
    )
    from opensculptboy.wearables.catalogue import THUMBNAIL_SIZE

    wc.add_argument(
        "--thumbnail-size",
        type=int,
        default=THUMBNAIL_SIZE,
        metavar="PIXELS",
        help=f"the side of a thumbnail (default {THUMBNAIL_SIZE})",
    )
    wc.add_argument(
        "--refresh", action="store_true", help="render every thumbnail again"
    )
    wc.add_argument(
        "--no-quality",
        action="store_true",
        help="leave out the quality checks (faster for large folders)",
    )
    _json_usage(wc, "wearables catalogue")

    from opensculptboy.wearables.check import BODIES, FRAMES, POSE_LIMIT, TOLERANCE

    wk = wear_sub.add_parser(
        "check",
        help="check the fit of a card's outfit on test bodies, at rest, in the T-pose "
        "and in the clips, and write a review sheet",
        description="Run the quality checks of every worn piece, dress the outfit on the "
        "test bodies, measure skin through the pieces and lower pieces through upper "
        "ones, print one JSON report with gates, and write a review sheet. Exit status 0 "
        "when every gate passes, 1 when one fails, 2 on an error. See docs/wearables.md, "
        "'Check'.",
    )
    wk.add_argument("card", help="the character card (JSON)")
    wk.add_argument("--outfit", help="the outfit to check (default: the worn one)")
    wk.add_argument(
        "--bodies",
        metavar="LIST",
        help=f"comma-separated test bodies: {', '.join(BODIES)} (the default), or "
        "SLIDER=VALUE",
    )
    wk.add_argument(
        "--clips",
        metavar="LIST",
        help="comma-separated clips of the pose library (default: all), or none",
    )
    wk.add_argument(
        "--frames",
        type=int,
        default=FRAMES,
        help=f"frames sampled from each clip (default {FRAMES})",
    )
    wk.add_argument("--no-tpose", action="store_true", help="leave out the VRM T-pose")
    wk.add_argument(
        "--tolerance",
        type=float,
        default=TOLERANCE * 1000,
        metavar="MM",
        help=f"the distance outside a piece that counts (default {TOLERANCE * 1000:g})",
    )
    wk.add_argument(
        "--pose-limit",
        type=float,
        default=POSE_LIMIT,
        metavar="PERCENT",
        help="the share of covered vertices that may fail in the T-pose and the clips "
        f"(default {POSE_LIMIT:g})",
    )
    wk.add_argument(
        "--sheet",
        metavar="PATH",
        help="the review sheet PNG (default: CARD.check.png next to the card; none for "
        "no sheet)",
    )
    _json_usage(wk, "wearables check")

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

    words = sys.argv[1:] if argv is None else list(argv)
    if words[:3] == ["wearables", "outfit", "export"]:
        words = ["wearables", "outfit-export", *words[3:]]
        argv = words
    if words[:1] == ["wearables"] and words[1:2] and words[1] in JSON_COMMANDS:
        _json_usage(parser, f"wearables {words[1]}")  # unknown options at the top level
    args = parser.parse_args(argv)
    if args.command in ("names", "viewer"):  # text for people, printed as it comes
        return _run(args, e, wear, f, b, c, t, q)
    with _results_alone():
        return _run(args, e, wear, f, b, c, t, q)


def _run(args, e, wear, f, b, c, t, q) -> int:
    if args.command == "character":
        _result(Character().to_dict())
        return 0
    if args.command == "export":
        return _export(args, e)
    if args.command == "names":
        return _names(args)
    if args.command == "wearables":
        return _wearables(args, wear)
    if args.command == "pose":
        return _pose(args)
    if args.command == "fit":
        return _fit(args, f)
    if args.command == "bake":
        return _bake(args, b)
    if args.command == "convert":
        return _convert(args, c)
    if args.command == "bake-tiers":
        return _bake_tiers(args, t)
    if args.command == "portraits":
        return _portraits(args, q)
    if args.command == "viewer":
        from anny.viewer.__main__ import main as viewer_main

        return viewer_main(args.rest)
    return 1


if __name__ == "__main__":
    sys.exit(main())

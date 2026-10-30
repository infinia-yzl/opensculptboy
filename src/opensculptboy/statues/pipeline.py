# OpenSculptBoy
# Apache License, Version 2.0
"""
One pipeline for every statue: a statue is a config file, and the steps are the same.

A config (``opensculptboy.statue/1``, JSON) names the scan and says what the scan cannot::

    {
      "format": "opensculptboy.statue/1",
      "name": "doryphoros",
      "title": "Doryphoros (the spear bearer)",
      "scan": "doryphoros.stl",
      "source": "https://open.smk.dk/artwork/image/KAS11",
      "licence": "PDM-1.0",
      "credit": "SMK, National Gallery of Denmark, cast KAS11",
      "stature": 1.8,
      "up": null, "front": null,
      "plinth": true,
      "masks": [{"kind": "box", "role": "support", "centre": [-0.2, 0.4, -0.1], "size": [0.2, 0.8, 0.2]}],
      "lost": ["hand.l"],
      "pose": "auto",
      "landmarks": {"nose_tip": [0.0, 1.62, 0.12], "knee_l": [0.1, 0.5, 0.0]}
    }

Positions of masks and landmarks are in the frame of the converted figure (``replica.glb``:
metres, Y up, the figure facing +Z, standing on y = 0), which depends on the config's
``stature``, ``up``, ``front`` and ``plinth`` alone; the ``preview`` step draws that frame
with a grid, so that a person reads the positions off its pictures. Landmark hints of the
face snap to the nearest point of the scan; hints of joints stay where they are given. ``pose``
names the pose of anny's library that the fit starts from, ``"auto"`` (the default) for the
closest one, or null for the rest pose.

:func:`run_statue` runs the steps in order, each writing into ``<out>/<name>/``:

1. ``convert``: :func:`~opensculptboy.statues.scan_to_handoff` into ``handoff/``.
2. ``landmarks``: joint and face landmarks found on pictures of the scan
   (:mod:`opensculptboy.statues.landmarks`, the ``pose`` extra), with the config's hints over
   them (``"auto_landmarks": false`` keeps the hints alone).
3. ``preview``: front and side pictures of the converted figure, the removed pieces, the
   masked parts and the landmarks over a 10 cm grid (``preview/``).
4. ``fit``: :func:`~opensculptboy.statues.fit_statue` to the config's detail level, the card
   in ``card/<name>.json``.
5. ``measure``: the distances per region of the figure between each level and the scan
   (``measures.json``), and the meshes for review in glTF's frame (``meshes/*.ply``).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time

import numpy as np

FORMAT = "opensculptboy.statue/1"
STEPS = ("convert", "landmarks", "preview", "fit", "measure")


@dataclasses.dataclass
class StatueConfig:
    """A statue's config (see the module documentation)."""

    name: str
    scan: str
    title: str | None = None
    source: str | None = None
    licence: str | None = None
    credit: str | None = None
    stature: float = 1.8
    up: object = None
    front: object = None
    plinth: bool = True
    masks: list = dataclasses.field(default_factory=list)
    lost: list = dataclasses.field(default_factory=list)
    landmarks: dict = dataclasses.field(default_factory=dict)
    auto_landmarks: bool = True
    # the face's landmarks from MediaPipe's face landmarker on a picture of the head (eyes'
    # and mouth's corners, nose tip, chin), so that anny's eyes and mouth sit on the statue's
    face_landmarks: bool = True
    # False (the default): anny's own working face, its lids and lips held open as anny's
    # (hold: the rings around the eyes' and the mouth's openings that take no data); True:
    # carved eyes and closed lips on the scan
    solid: bool = False
    hold: list = dataclasses.field(default_factory=lambda: [5, 5])
    # the band along the scan's cut by a hat or locks whose skin takes no data (rings, metres)
    cut: list = dataclasses.field(default_factory=lambda: [1, 0.003])
    prune: float | bool = 0.025
    pose: str | None = "auto"
    detail: int = 2
    # FitOptions fields that this statue's fit takes over the defaults (name -> value)
    fit: dict = dataclasses.field(default_factory=dict)
    notes: str | None = None
    path: pathlib.Path | None = None  # the config file, for relative scan paths

    @classmethod
    def from_dict(cls, data: dict, path=None) -> "StatueConfig":
        data = dict(data)
        found = data.pop("format", FORMAT)
        if found != FORMAT:
            raise ValueError(
                f"A statue config has the format {FORMAT!r}; got {found!r}."
            )
        fields = {f.name for f in dataclasses.fields(cls)} - {"path"}
        unknown = set(data) - fields
        if unknown:
            raise ValueError(f"Unknown statue config fields {sorted(unknown)}.")
        for key in ("name", "scan"):
            if not isinstance(data.get(key), str) or not data[key]:
                raise ValueError(f"A statue config needs {key!r}.")
        config = cls(**data, path=None if path is None else pathlib.Path(path))
        if config.detail not in (0, 1, 2):
            raise ValueError(f"detail is 0, 1 or 2; got {config.detail!r}.")
        from opensculptboy.fit.register import FitOptions

        options = {f.name for f in dataclasses.fields(FitOptions)}
        unknown = set(config.fit) - options
        if unknown:
            raise ValueError(f"Unknown fit options {sorted(unknown)}.")
        return config

    @classmethod
    def load(cls, path) -> "StatueConfig":
        path = pathlib.Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), path)

    def scan_path(self, scans=None) -> pathlib.Path:
        """The scan file: ``scan`` as given when it exists, else in ``scans``, else next to
        the config."""
        candidates = [pathlib.Path(self.scan)]
        if scans is not None:
            candidates.append(pathlib.Path(scans) / self.scan)
        if self.path is not None:
            candidates.append(self.path.parent / self.scan)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"The scan {self.scan!r} of {self.name!r} is missing.")


def snap_landmarks(landmarks: dict, vertices, triangles) -> dict:
    """The landmarks (Anny's frame) with the points of the face and the crown moved onto
    the nearest point of the surface; joint landmarks stay."""
    from anny.hair import closest_triangles
    from opensculptboy.fit.reference import LANDMARKS, canonical_landmark

    names = list(landmarks)
    surface = []
    for name in names:
        key = canonical_landmark(name)
        if key is not None and key in LANDMARKS and LANDMARKS[key][0] != "bone":
            surface.append(name)
    out = {k: np.asarray(v, dtype=np.float64) for k, v in landmarks.items()}
    if surface:
        points = np.stack([out[k] for k in surface])
        tri, bary, _ = closest_triangles(points, vertices, triangles)
        snapped = np.einsum("nk, nkd -> nd", bary, vertices[triangles[tri]])
        for k, p in zip(surface, snapped):
            out[k] = p
    return out


def _gltf(points) -> np.ndarray:
    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    return np.asarray(points, dtype=np.float64) @ GLTF_TO_ANNY


def _anny(points) -> np.ndarray:
    from opensculptboy.fit.handoff import GLTF_TO_ANNY

    return np.asarray(points, dtype=np.float64) @ GLTF_TO_ANNY.T


def convert(config: StatueConfig, folder, scans=None):
    """Step 1: the handoff folder ``folder`` from the config's scan."""
    from opensculptboy.statues.handoff import scan_to_handoff

    scan = config.scan_path(scans)
    result = scan_to_handoff(
        scan,
        folder,
        stature=config.stature,
        up=config.up,
        front=config.front,
        masks=config.masks,
        plinth=config.plinth,
        source=config.source,
        licence=config.licence,
        credit=config.credit,
        title=config.title or config.name,
        lost=config.lost,
    )
    return result


def landmarks(config: StatueConfig, folder, out=None) -> dict:
    """
    Step 2: the landmarks of the converted figure, written into the handoff's appearance.json:
    those that :func:`~opensculptboy.statues.landmarks.detect_landmarks` finds on pictures of
    the scan when ``auto_landmarks`` is on, with the config's hints over them; the points of
    the face snap onto the scan. ``out`` receives the pictures with the found points. Returns
    the landmarks by name with where each came from.
    """
    from opensculptboy.fit.handoff import load_handoff

    folder = pathlib.Path(folder)
    skin = load_handoff(folder).skin_parts()[0]
    found, sources = {}, {}
    if config.auto_landmarks:
        from opensculptboy.statues.landmarks import detect_landmarks

        pictures = {}
        try:
            found = detect_landmarks(skin.vertices, skin.triangles, pictures)
        except (ValueError, ImportError) as error:
            sources["error"] = str(error)
        if out is not None:
            out = pathlib.Path(out)
            out.mkdir(parents=True, exist_ok=True)
            for label, image in pictures.items():
                image.save(out / f"landmarks_{label}.png")
        sources.update({k: "detected" for k in found})
        if config.face_landmarks:
            from opensculptboy.statues.landmarks import detect_face

            try:
                face = detect_face(skin.vertices, skin.triangles, pictures)
            except (ValueError, ImportError) as error:
                face = {}
                sources["face_error"] = str(error)
            found.update(face)
            sources.update({k: "face" for k in face})
            if out is not None and "face" in pictures:
                pictures["face"].save(pathlib.Path(out) / "landmarks_face.png")
    hints = {k: _anny([v])[0] for k, v in (config.landmarks or {}).items()}
    sources.update({k: "hint" for k in hints})
    marks = snap_landmarks({**found, **hints}, skin.vertices, skin.triangles)
    appearance_path = folder / "appearance.json"
    appearance = json.loads(appearance_path.read_text(encoding="utf-8"))
    appearance["landmarks"].update(
        {k: [float(x) for x in _gltf([v])[0]] for k, v in marks.items()}
    )
    appearance_path.write_text(
        json.dumps(appearance, indent=2) + "\n", encoding="utf-8"
    )
    return {
        k: {"point": [float(x) for x in _gltf([marks[k]])[0]], "source": sources[k]}
        for k in marks
    } | ({"error": sources["error"]} if "error" in sources else {})


def _grid_lines(image, view, points, size, margin, step=0.1):
    """A metric grid over a picture of :func:`opensculptboy.render.dressed.draw`, labelled
    in the converted figure's glTF coordinates."""
    from PIL import ImageDraw

    from opensculptboy.render.dressed import _font, _frame

    s, ox, oy = _frame(points, view, size, margin)
    right, up, _ = view.basis()
    across = _gltf([right])[0]
    axis = int(np.argmax(np.abs(across)))
    sign = np.sign(across[axis])
    name = "xyz"[axis]
    paint = ImageDraw.Draw(image, "RGBA")
    font = _font(max(10, size[0] // 40))
    xy, _ = view.project(points)
    lo, hi = xy.min(0), xy.max(0)
    digits = 1 if step >= 0.1 else 2
    label_every = 1 if step >= 0.1 else 2
    for v in np.arange(np.floor(lo[1] / step) * step, hi[1] + step, step):
        y = -v * s + oy
        strong = abs(v / step - round(v / step * 0.5) * 2) < 1e-6
        paint.line([(0, y), (size[0], y)], fill=(40, 90, 160, 110 if strong else 50))
        paint.text((2, y - 12), f"y {v:.{digits}f}", (40, 90, 160), font)
    for u in np.arange(np.floor(lo[0] / step) * step, hi[0] + step, step):
        x = u * s + ox
        paint.line([(x, 0), (x, size[1])], fill=(160, 70, 40, 60))
        if round(u / step) % label_every == 0:
            paint.text(
                (x + 2, size[1] - 14),
                f"{name} {sign * u:+.{digits}f}",
                (160, 70, 40),
                font,
            )
    return image


def preview(config: StatueConfig, folder, out, size=(720, 1080)):
    """Step 2: front and side pictures of the converted figure over a 10 cm grid, with the
    removed pieces in terracotta, drapery in blue, hair in green and the landmarks in red."""
    import trimesh

    from opensculptboy.fit.handoff import load_handoff
    from opensculptboy.render.dressed import Marker, Mesh, View, draw, labelled

    folder = pathlib.Path(folder)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    handoff = load_handoff(folder)
    colours = {"clothing": (120, 150, 210), "hair": (120, 190, 120)}
    meshes, everything = [], []
    for part in handoff.parts:
        colour = (
            (236, 234, 228) if part.is_skin else colours.get(part.role, (200, 200, 200))
        )
        meshes.append(Mesh(part.vertices, part.triangles, colour))
        everything.append(part.vertices)
    removed = folder / "removed.glb"
    if removed.exists():
        scene = trimesh.load(str(removed), force="scene")
        for geometry in scene.geometry.values():
            v = _anny(np.asarray(geometry.vertices))
            meshes.append(Mesh(v, np.asarray(geometry.faces), (196, 120, 90)))
            everything.append(v)
    points = np.concatenate(everything)
    marks = np.stack(list(handoff.landmarks.values())) if handoff.landmarks else []
    pictures = {}
    for label, view in (
        ("front", View(0.0)),
        ("side", View(-90.0)),
        ("back", View(180.0)),
    ):
        image = draw(meshes, view, size, 30.0, points, [Marker(marks, radius=4)])
        image = _grid_lines(image, view, points, size, 30.0)
        image = labelled(
            image,
            [
                f"{config.title or config.name}: {label}",
                "grid 10 cm, glTF frame of replica.glb",
            ],
        )
        path = out / f"{label}.png"
        image.save(path)
        pictures[label] = str(path)
    # the figure that the fit takes, alone
    skin = [m for m, p in zip(meshes, handoff.parts) if p.is_skin]
    for label, view in (("figure_front", View(0.0)), ("figure_side", View(-90.0))):
        image = draw(skin, view, size, 30.0, points)
        image = _grid_lines(image, view, points, size, 30.0)
        image = labelled(
            image, [f"{config.title or config.name}: {label}", "the skin of the fit"]
        )
        path = out / f"{label}.png"
        image.save(path)
        pictures[label] = str(path)
    return pictures


def _fit_options(fit: dict) -> dict:
    """FitOptions fields from a config's ``fit`` (JSON lists become tuples)."""
    return {k: tuple(v) if isinstance(v, list) else v for k, v in fit.items()}


def stage1_from_card(run) -> dict:
    """The stage-1 data (:func:`~opensculptboy.statues.fit.stage1_dict`) of an earlier run's
    folder from its card: the sliders, the pose and the fit's placement."""
    import types

    from opensculptboy.character import Character
    from opensculptboy.statues.fit import stage1_dict

    cards = sorted((pathlib.Path(run) / "card").glob("*.json"))
    if not cards:
        raise FileNotFoundError(f"{run}: no card, and no stage1/main.json.")
    data = json.loads(cards[0].read_text(encoding="utf-8"))
    placement = data["detail"]["fit"]["placement"]
    return stage1_dict(
        types.SimpleNamespace(
            character=Character.from_dict(data),
            rotation=placement["rotation"],
            translation=placement["translation"],
        )
    )


def fit(config: StatueConfig, folder, card, verbose: bool = False, stage1=None):
    """Step 4: the fit of the handoff folder, written to the card ``card``. With ``prune``
    (a distance in metres), a first fit to level 0 takes out the scan's pieces further than
    that from it (:func:`~opensculptboy.statues.fit.prune_far`: props and leftovers of
    supports, and locks of hair as a hair piece) before the fit itself. The sliders, pose
    and placement of both fits go to ``stage1/`` beside the card's folder; with ``stage1``
    (such a folder of an earlier run, whose handoff ``folder`` already holds), the covered
    skin is judged again on its first fit and only the detail levels run."""
    import types

    from opensculptboy.fit.register import FitOptions
    from opensculptboy.statues.fit import (
        covered_skin,
        fit_statue,
        stage1_dict,
        stage1_from,
    )

    def run(detail, start=None):
        return fit_statue(
            folder,
            detail=detail,
            name=config.name,
            options=FitOptions(verbose=verbose, **_fit_options(config.fit)),
            start_pose=config.pose,
            solid=config.solid,
            hold=tuple(config.hold),
            stage1=start,
        )

    saved = pathlib.Path(card).parent.parent / "stage1"
    pruned = None
    if stage1 is not None:
        stage1 = pathlib.Path(stage1)
        main = json.loads((stage1 / "main.json").read_text(encoding="utf-8"))
        if config.prune and (stage1 / "first.json").exists():
            data = json.loads((stage1 / "first.json").read_text(encoding="utf-8"))
            character, rotation, translation = stage1_from(data)
            level0 = np.load(stage1 / "first_level0.npy")
            first = types.SimpleNamespace(
                character=character,
                rotation=rotation,
                translation=translation,
                meshes={0: (level0, None)},
            )
            pruned = {"covered": covered_skin(folder, first, cut=tuple(config.cut))}
        if stage1.resolve() != saved.resolve():
            saved.mkdir(parents=True, exist_ok=True)
            for item in stage1.iterdir():
                (saved / item.name).write_bytes(item.read_bytes())
        result = run(config.detail, main)
    else:
        saved.mkdir(parents=True, exist_ok=True)
        if config.prune:
            # a first fit to level 0 tells the figure from what is left of props and supports
            from opensculptboy.statues.fit import prune_far

            first = run(0)
            (saved / "first.json").write_text(
                json.dumps(stage1_dict(first)) + "\n", encoding="utf-8"
            )
            np.save(saved / "first_level0.npy", np.asarray(first.meshes[0][0]))
            pruned = prune_far(folder, first, distance=float(config.prune))
            # the skull under a hat and fused locks keeps anny's own shape
            pruned["covered"] = covered_skin(folder, first, cut=tuple(config.cut))
        result = run(config.detail)
        (saved / "main.json").write_text(
            json.dumps(stage1_dict(result)) + "\n", encoding="utf-8"
        )
    if verbose and pruned:
        print(f"pruned {pruned}", flush=True)
    result.pruned = pruned
    card = pathlib.Path(card)
    card.parent.mkdir(parents=True, exist_ok=True)
    result.write(card)
    return result


def measure(config: StatueConfig, folder, result, out) -> dict:
    """Step 4: the distances per region between the scan and the parametric body and each
    level of the fit (mm), and the meshes for review in glTF's frame."""
    import torch
    import trimesh

    from opensculptboy.detail import detail_topology
    from opensculptboy.fit.handoff import load_handoff
    from opensculptboy.fit.register import fit_model
    from opensculptboy.statues.compare import (
        carry_regions,
        region_distances,
        statue_regions,
        under_cover,
    )

    out = pathlib.Path(out)
    meshes = out / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    model = fit_model()
    regions0 = statue_regions(model)
    handoff = load_handoff(folder)
    skin = handoff.skin_parts()[0]
    c = result.character
    with torch.no_grad():
        posed = model(
            pose_parameters=c.pose_parameters(model),
            phenotype_kwargs=dict(c.phenotype),
            local_changes_kwargs=dict(c.local_changes) or None,
            face_shape_kwargs=dict(c.face_shapes) or None,
        )["vertices"][0]
    parametric = posed.double().numpy() @ result.rotation.T + result.translation
    stages = {"parametric": (parametric, model.faces.cpu().numpy())}
    stages.update({f"level{k}": v for k, v in result.meshes.items()})

    def save(name, V, T):
        trimesh.Trimesh(_gltf(V), T, process=False).export(str(meshes / f"{name}.ply"))

    save("scan_skin", skin.vertices, skin.triangles)
    for part in handoff.parts:
        if not part.is_skin:
            save(f"part_{part.role}_{part.name}", part.vertices, part.triangles)
    # the hat and the locks seated on the finest fitted skin, for review and for the card
    from opensculptboy.statues.fit import SEAT_BY_ROLE, seat_piece, trim_piece

    finest = result.meshes[max(result.meshes)]
    seated = {}
    (out / "card").mkdir(parents=True, exist_ok=True)
    for part in handoff.parts:
        if part.is_skin or part.role not in ("hair", "clothing"):
            continue
        reach, blend = SEAT_BY_ROLE[part.role]
        V, T, dropped = trim_piece(part.vertices, part.triangles, *finest)
        V, moved = seat_piece(V, T, *finest, reach=reach, blend=blend)
        save(f"seated_{part.role}_{part.name}", V, T)
        seated[part.name] = {
            "role": part.role,
            "border_moved": moved,
            "triangles_dropped": dropped,
        }
        trimesh.Trimesh(_gltf(V), T, process=False).export(
            str(out / "card" / f"{config.name}.{part.name}.glb")
        )
    removed = pathlib.Path(folder) / "removed.glb"
    if removed.exists():
        trimesh.load(str(removed), force="mesh").export(str(meshes / "removed.ply"))
    report = {
        "name": config.name,
        "title": config.title,
        "licence": config.licence,
        "source": config.source,
        "seconds": result.seconds,
        "errors": result.errors,
        "card": {
            "phenotype": c.phenotype,
            "face_shapes": len(c.face_shapes),
            "local_changes": len(c.local_changes),
        },
        "stages": {},
        "seated": seated,
    }
    # the whole statue over the skull: the skin with its hair and drapery pieces
    covers = [
        p for p in handoff.parts if not p.is_skin and p.role in ("hair", "clothing")
    ]
    cover = None
    if covers:
        pieces = [skin] + covers
        base = np.cumsum([0] + [len(p.vertices) for p in pieces[:-1]])
        cover = (
            np.concatenate([p.vertices for p in pieces]),
            np.concatenate([p.triangles + b for p, b in zip(pieces, base)]),
        )
    for name, (V, T) in stages.items():
        level = 0 if name == "parametric" else int(name[len("level") :])
        regions = (
            regions0
            if level == 0
            else carry_regions(detail_topology(model, level), regions0)
        )
        report["stages"][name] = region_distances(
            V, T, regions, skin.vertices, skin.triangles
        )
        if cover is not None:
            report["stages"][name]["under_cover"] = under_cover(
                V, regions, ("scalp",), *cover
            )
        save(f"fitted_{name}", V, T)
    (out / "measures.json").write_text(
        json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8"
    )
    return report


def run_statue(
    config: StatueConfig,
    out,
    scans=None,
    steps=STEPS,
    verbose: bool = False,
    stage1=None,
) -> dict:
    """Run ``steps`` (of :data:`STEPS`, in their order) for one statue into
    ``<out>/<name>/``; returns a summary with the time of each step. With ``stage1`` (the
    folder of an earlier run of the statue), its handoff and its sliders, pose and placement
    are taken over and only the fit's detail levels and the measures run."""
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"Unknown steps {sorted(unknown)}; use {STEPS}.")
    root = pathlib.Path(out) / config.name
    root.mkdir(parents=True, exist_ok=True)
    folder = root / "handoff"
    summary = {"name": config.name, "steps": {}}
    result = None
    source = None
    if stage1 is not None:
        import shutil

        earlier = pathlib.Path(stage1)
        if earlier.resolve() != root.resolve():
            shutil.copytree(earlier / "handoff", folder, dirs_exist_ok=True)
        source = earlier / "stage1"
        if not (source / "main.json").exists():
            # a run from before the stage-1 files: its card's sliders, pose and placement
            source = root / "stage1"
            source.mkdir(parents=True, exist_ok=True)
            (source / "main.json").write_text(
                json.dumps(stage1_from_card(earlier)) + "\n", encoding="utf-8"
            )
        steps = [s for s in steps if s in ("fit", "measure")]
        summary["stage1"] = str(earlier)
    for step in STEPS:
        if step not in steps:
            continue
        clock = time.perf_counter()
        if step == "convert":
            converted = convert(config, folder, scans)
            summary["frame"] = {
                "up": converted.up.tolist(),
                "front": converted.front.tolist(),
                "up_confidence": converted.up_confidence,
                "front_score": converted.front_score,
                "plinth": converted.plinth,
                "triangles": converted.triangles,
                "scale": converted.scale,
            }
        elif step == "landmarks":
            summary["landmarks"] = landmarks(config, folder, root / "preview")
        elif step == "preview":
            summary["preview"] = preview(config, folder, root / "preview")
        elif step == "fit":
            result = fit(
                config, folder, root / "card" / f"{config.name}.json", verbose, source
            )
            summary["pruned"] = getattr(result, "pruned", None)
        elif step == "measure":
            if result is None:
                raise ValueError("The measure step needs the fit step in the same run.")
            report = measure(config, folder, result, root)
            summary["distances"] = {k: v["all"] for k, v in report["stages"].items()}
        summary["steps"][step] = round(time.perf_counter() - clock, 2)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8"
    )
    return summary

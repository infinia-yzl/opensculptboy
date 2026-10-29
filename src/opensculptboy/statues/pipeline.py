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
2. ``preview``: front and side pictures of the converted figure, the removed pieces, the
   masked parts and the landmarks over a 10 cm grid (``preview/``).
3. ``fit``: :func:`~opensculptboy.statues.fit_statue` to the config's detail level, the card
   in ``card/<name>.json``.
4. ``measure``: the distances per region of the figure between each level and the scan
   (``measures.json``), and the meshes for review in glTF's frame (``meshes/*.ply``).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time

import numpy as np

FORMAT = "opensculptboy.statue/1"
STEPS = ("convert", "preview", "fit", "measure")


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
    pose: str | None = "auto"
    detail: int = 2
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
    if config.landmarks:
        from opensculptboy.fit.handoff import load_handoff

        skin = load_handoff(folder).skin_parts()[0]
        hints = {k: _anny([v])[0] for k, v in config.landmarks.items()}
        marks = snap_landmarks(hints, skin.vertices, skin.triangles)
        appearance_path = pathlib.Path(folder) / "appearance.json"
        appearance = json.loads(appearance_path.read_text(encoding="utf-8"))
        appearance["landmarks"].update(
            {k: [float(x) for x in _gltf([v])[0]] for k, v in marks.items()}
        )
        appearance_path.write_text(
            json.dumps(appearance, indent=2) + "\n", encoding="utf-8"
        )
    return result


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
    for v in np.arange(np.floor(lo[1] / step) * step, hi[1] + step, step):
        y = -v * s + oy
        strong = abs(v / step - round(v / step * 0.5) * 2) < 1e-6
        paint.line([(0, y), (size[0], y)], fill=(40, 90, 160, 110 if strong else 50))
        paint.text((2, y - 12), f"y {v:.1f}", (40, 90, 160), font)
    for u in np.arange(np.floor(lo[0] / step) * step, hi[0] + step, step):
        x = u * s + ox
        paint.line([(x, 0), (x, size[1])], fill=(160, 70, 40, 60))
        paint.text(
            (x + 2, size[1] - 14), f"{name} {sign * u:+.1f}", (160, 70, 40), font
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


def fit(config: StatueConfig, folder, card, verbose: bool = False):
    """Step 3: the fit of the handoff folder, written to the card ``card``."""
    from opensculptboy.fit.register import FitOptions
    from opensculptboy.statues.fit import fit_statue

    result = fit_statue(
        folder,
        detail=config.detail,
        name=config.name,
        options=FitOptions(verbose=verbose),
        start_pose=config.pose,
    )
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
    }
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
) -> dict:
    """Run ``steps`` (of :data:`STEPS`, in their order) for one statue into
    ``<out>/<name>/``; returns a summary with the time of each step."""
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"Unknown steps {sorted(unknown)}; use {STEPS}.")
    root = pathlib.Path(out) / config.name
    root.mkdir(parents=True, exist_ok=True)
    folder = root / "handoff"
    summary = {"name": config.name, "steps": {}}
    result = None
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
        elif step == "preview":
            summary["preview"] = preview(config, folder, root / "preview")
        elif step == "fit":
            result = fit(config, folder, root / "card" / f"{config.name}.json", verbose)
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

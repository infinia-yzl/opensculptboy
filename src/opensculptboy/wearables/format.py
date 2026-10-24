# OpenSculptBoy
# Apache License, Version 2.0
"""
The wearable package ``opensculptboy.wearable/1`` (docs/wearables.md).

A package is three files that share a name: ``<name>.wearable.json`` (the description),
``<name>.glb`` (the garment's geometry and materials in the binding body's rest pose; metres,
glTF axes) and ``<name>.binding.safetensors`` (the binding of every garment vertex, the body
it was bound on and the cover mask). The GLB file holds one mesh whose primitives share one
set of vertices, so that the order of the vertices in the file is the order of the binding.

:func:`write_wearable` and :func:`read_wearable` write and read packages. A package carries
a ``provenance`` field, ``public`` or ``private``; :func:`check_destination` refuses to write
a private package anywhere inside a checkout of this repository, and the public test suite
fails when a file under ``src/`` or ``test/data/`` declares the private provenance.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np

FORMAT = "opensculptboy.wearable/1"
PUBLIC = "public"
PRIVATE = "private"
PROVENANCES = (PUBLIC, PRIVATE)
SLOTS = ("head", "face", "neck", "torso", "arms", "hands", "legs", "feet", "cloak")
SLOTS += ("under", "accessory")
METHODS = ("surface", "rigid")
BODIES = ("anny-template", "card", "reference")
LICENCE_KEYS = ("spdx", "author", "source", "design_source", "attribution")


class ProvenanceError(ValueError):
    """A private package aimed at a place where private packages never go."""


@dataclasses.dataclass
class Material:
    """
    A garment material in glTF metallic-roughness terms. ``textures`` maps the slots
    ``base_color``, ``metallic_roughness``, ``normal``, ``occlusion`` and ``emissive`` to PNG or
    JPEG bytes on the garment's UVs.
    """

    name: str
    base_color: tuple = (0.8, 0.8, 0.8, 1.0)  # linear RGBA
    metallic: float = 0.0
    roughness: float = 0.8
    emissive: tuple = (0.0, 0.0, 0.0)
    alpha_mode: str = "OPAQUE"  # "OPAQUE", "MASK" or "BLEND"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    normal_scale: float = 1.0
    textures: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Garment:
    """
    The geometry of a wearable: welded vertices in Anny's frame (metres, Z up, the figure
    facing -Y) on the binding body at rest, unit normals, UVs (glTF convention, v = 0 at the
    top), triangles, and for each triangle its material and its named group.
    """

    vertices: np.ndarray  # (V, 3)
    triangles: np.ndarray  # (T, 3)
    normals: np.ndarray | None = None  # (V, 3)
    uvs: np.ndarray | None = None  # (V, 2)
    triangle_material: np.ndarray | None = None  # (T,) index into materials
    triangle_group: np.ndarray | None = None  # (T,) index into groups
    materials: list = dataclasses.field(default_factory=list)
    groups: list = dataclasses.field(default_factory=lambda: ["main"])

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.triangles = np.asarray(self.triangles, dtype=np.int64).reshape(-1, 3)
        if self.normals is None:
            from opensculptboy.wearables.geometry import vertex_normals

            self.normals = vertex_normals(self.vertices, self.triangles)
        self.normals = np.asarray(self.normals, dtype=np.float64)
        if self.triangle_material is None:
            self.triangle_material = np.zeros(len(self.triangles), np.int64)
        if self.triangle_group is None:
            self.triangle_group = np.zeros(len(self.triangles), np.int64)
        self.triangle_material = np.asarray(self.triangle_material, dtype=np.int64)
        self.triangle_group = np.asarray(self.triangle_group, dtype=np.int64)
        if not self.materials:
            self.materials = [Material("garment")]

    def check(self) -> None:
        if len(self.triangles) and int(self.triangles.max()) >= len(self.vertices):
            raise ValueError("A garment triangle names a vertex that does not exist.")
        if self.normals.shape != self.vertices.shape:
            raise ValueError("A garment needs one normal per vertex.")
        if self.uvs is not None and len(self.uvs) != len(self.vertices):
            raise ValueError("A garment needs one UV per vertex.")
        if len(self.triangle_material) != len(self.triangles):
            raise ValueError("A garment needs one material per triangle.")
        if len(self.triangle_group) != len(self.triangles):
            raise ValueError("A garment needs one group per triangle.")
        if len(self.triangles) and int(self.triangle_material.max()) >= len(
            self.materials
        ):
            raise ValueError("A garment triangle names a material that does not exist.")
        if len(self.triangles) and int(self.triangle_group.max()) >= len(self.groups):
            raise ValueError("A garment triangle names a group that does not exist.")


@dataclasses.dataclass
class Binding:
    """
    How each garment vertex follows the body.

    ``surface``: ``corners`` (V, 3) are the vertices of the body triangle (rows of Anny's
    template mesh), ``barycentric`` (V, 3) the bound point, ``offset`` (V, 3) the vertex's
    offset from the bound point in the point's frame (tangent, bitangent, normal) divided by
    the frame's scale, and ``normal`` (V, 3) the vertex normal in that frame.
    ``rigid``: ``bone`` names one bone, and ``local`` (V, 3) and ``normal`` (V, 3) hold the
    vertices and normals in that bone's rest frame.
    ``body_vertices`` (N, 3) holds the rest vertices of the body that the garment was bound
    on (Anny's rows), so that the fitting can compare a body with the binding body.
    ``skin_weights`` and ``skin_joints`` (V, K), with the bone names ``joint_names``, are the
    garment's own skin weights (``Wearable.weights == "own"``), for example the body's weights
    smoothed over the garment so that neighbouring vertices move alike.
    """

    method: str
    body: str
    body_vertices: np.ndarray
    corners: np.ndarray | None = None
    barycentric: np.ndarray | None = None
    offset: np.ndarray | None = None
    normal: np.ndarray | None = None
    bone: str | None = None
    local: np.ndarray | None = None
    skin_weights: np.ndarray | None = None
    skin_joints: np.ndarray | None = None
    joint_names: list | None = None

    def arrays(self) -> dict[str, np.ndarray]:
        out = {"body_vertices": np.asarray(self.body_vertices, dtype=np.float64)}
        for key in (
            "corners",
            "barycentric",
            "offset",
            "normal",
            "local",
            "skin_weights",
            "skin_joints",
        ):
            value = getattr(self, key)
            if value is not None:
                dtype = np.int32 if key in ("corners", "skin_joints") else np.float64
                out[key] = np.ascontiguousarray(np.asarray(value, dtype=dtype))
        return out

    @property
    def vertex_count(self) -> int:
        source = self.corners if self.method == "surface" else self.local
        return 0 if source is None else len(source)


@dataclasses.dataclass
class Wearable:
    """
    A wearable package: its description, its garment and its binding.

    ``hides`` lists rules toward other pieces: ``{"piece": id, "groups": [name, ...]}`` hides
    those named groups of that piece while this one is shown. ``covers`` is a mask over the
    vertices of Anny's template (``True`` where the garment covers the skin), or None; the
    masks of ``group_covers`` (by group name) add the skin under a group while that group
    shows. With ``cover_ray`` a vertex of a mask counts as covered only where its normal ray
    also meets the dressed piece (for masks made on another body, and for open pieces such as
    laced boots whose mask spans the gaps).
    ``springs`` and ``constraints`` are kept for the VRM export (none are written yet).
    """

    name: str
    slot: str
    layer: int
    garment: Garment
    binding: Binding
    provenance: str = PUBLIC
    licence: dict = dataclasses.field(default_factory=dict)
    weights: str = "body"
    covers: np.ndarray | None = None
    group_covers: dict = dataclasses.field(default_factory=dict)
    cover_ray: bool = False
    hides: list = dataclasses.field(default_factory=list)
    springs: list = dataclasses.field(default_factory=list)
    constraints: list = dataclasses.field(default_factory=list)
    label: str | None = None
    extras: dict = dataclasses.field(default_factory=dict)

    def check(self) -> None:
        if self.provenance not in PROVENANCES:
            raise ValueError(
                f"provenance must be one of {PROVENANCES}; got {self.provenance!r}."
            )
        if self.slot not in SLOTS:
            raise ValueError(f"Unknown slot {self.slot!r}; the slots are {SLOTS}.")
        if not 0 <= int(self.layer) <= 100:
            raise ValueError(f"layer must lie in 0..100; got {self.layer!r}.")
        if self.provenance == PUBLIC:
            missing = [k for k in ("spdx", "design_source") if not self.licence.get(k)]
            if missing:
                raise ValueError(
                    f"A public wearable needs a licence with {missing} (the keys are "
                    f"{LICENCE_KEYS})."
                )
        if self.binding.method not in METHODS:
            raise ValueError(f"Unknown binding method {self.binding.method!r}.")
        if self.binding.body not in BODIES:
            raise ValueError(f"Unknown binding body {self.binding.body!r}.")
        if self.binding.vertex_count != len(self.garment.vertices):
            raise ValueError(
                f"The binding holds {self.binding.vertex_count} vertices and the garment "
                f"{len(self.garment.vertices)}."
            )
        if self.weights not in ("body", "rigid", "own"):
            raise ValueError(
                f"Unknown weights {self.weights!r}; use 'body', 'rigid' or 'own'."
            )
        if self.weights == "own" and (
            self.binding.skin_weights is None
            or self.binding.skin_joints is None
            or not self.binding.joint_names
        ):
            raise ValueError("Own weights need skin_weights, skin_joints and joints.")
        unknown = sorted(set(self.group_covers) - set(self.garment.groups))
        if unknown:
            raise ValueError(f"group_covers names groups the garment lacks: {unknown}.")
        for rule in self.hides:
            if not isinstance(rule, dict) or not isinstance(rule.get("piece"), str):
                raise ValueError(f"A hides rule needs 'piece' and 'groups': {rule!r}.")
            if not all(isinstance(g, str) for g in rule.get("groups", [])):
                raise ValueError(f"The groups of a hides rule are names: {rule!r}.")
        self.garment.check()

    def description(self, stem: str) -> dict[str, Any]:
        binding = {
            "body": self.binding.body,
            "method": self.binding.method,
            "data": f"{stem}.binding.safetensors",
        }
        if self.binding.bone is not None:
            binding["bone"] = self.binding.bone
        if self.binding.joint_names is not None:
            binding["joints"] = list(self.binding.joint_names)
        out = {
            "format": FORMAT,
            "name": self.name,
            "label": self.label or self.name,
            "slot": self.slot,
            "layer": int(self.layer),
            "provenance": self.provenance,
            "licence": dict(self.licence),
            "geometry": f"{stem}.glb",
            "binding": binding,
            "weights": self.weights,
            "covers": "mask" if self.covers is not None else None,
            "group_covers": sorted(self.group_covers),
            "cover_ray": bool(self.cover_ray),
            "groups": list(self.garment.groups),
            "hides": [dict(r) for r in self.hides],
            "springs": list(self.springs),
            "constraints": list(self.constraints),
        }
        if self.extras:
            out["extras"] = dict(self.extras)
        return out


def repository_root(path: str | pathlib.Path) -> pathlib.Path | None:
    """The checkout of this repository that holds ``path`` (a folder with ``pyproject.toml``
    and ``src/opensculptboy``), or None."""
    path = pathlib.Path(path).resolve()
    for folder in [path, *path.parents]:
        if (folder / "pyproject.toml").is_file() and (
            folder / "src" / "opensculptboy" / "__init__.py"
        ).is_file():
            return folder
    return None


def check_destination(path: str | pathlib.Path, provenance: str) -> None:
    """Raise :class:`ProvenanceError` when a private package would land in a checkout of the
    repository (its source, its tests or any other folder of it)."""
    if provenance != PUBLIC:
        root = repository_root(path)
        if root is not None:
            raise ProvenanceError(
                f"A package with the provenance {provenance!r} never goes into the "
                f"repository tree ({root}); write it into a folder outside the checkout."
            )


def package_paths(path: str | pathlib.Path) -> dict[str, pathlib.Path]:
    """The three files of a package from the path of any of them or of its stem."""
    path = pathlib.Path(path)
    name = path.name
    for suffix in (".wearable.json", ".binding.safetensors", ".glb"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    folder = path.parent
    return {
        "json": folder / f"{name}.wearable.json",
        "glb": folder / f"{name}.glb",
        "binding": folder / f"{name}.binding.safetensors",
        "stem": folder / name,
    }


def write_wearable(path: str | pathlib.Path, wearable: Wearable) -> pathlib.Path:
    """Write a package; ``path`` is its ``.wearable.json`` file or its stem. Returns the
    path of the description."""
    from safetensors.numpy import save_file

    wearable.check()
    paths = package_paths(path)
    check_destination(paths["json"], wearable.provenance)
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    stem = paths["stem"].name
    arrays = wearable.binding.arrays()
    if wearable.covers is not None:
        arrays["covers"] = np.asarray(wearable.covers, dtype=np.uint8)
    for group, mask in wearable.group_covers.items():
        arrays[f"covers:{group}"] = np.asarray(mask, dtype=np.uint8)
    save_file(arrays, str(paths["binding"]), metadata={"format": FORMAT})
    _write_geometry(paths["glb"], wearable)
    paths["json"].write_text(json.dumps(wearable.description(stem), indent=2) + "\n")
    return paths["json"]


def read_wearable(path: str | pathlib.Path) -> Wearable:
    """Read a package from its ``.wearable.json`` file (or any file of it, or its stem)."""
    from safetensors.numpy import load_file

    paths = package_paths(path)
    data = json.loads(paths["json"].read_text())
    if data.get("format") != FORMAT:
        raise ValueError(
            f"{paths['json']}: format {data.get('format')!r}; this version reads {FORMAT}."
        )
    folder = paths["json"].parent
    binding_info = data.get("binding") or {}
    arrays = load_file(str(folder / binding_info.get("data", paths["binding"].name)))
    binding = Binding(
        method=binding_info.get("method", "surface"),
        body=binding_info.get("body", "anny-template"),
        body_vertices=arrays["body_vertices"],
        corners=_get(arrays, "corners", np.int64),
        barycentric=_get(arrays, "barycentric"),
        offset=_get(arrays, "offset"),
        normal=_get(arrays, "normal"),
        bone=binding_info.get("bone"),
        local=_get(arrays, "local"),
        skin_weights=_get(arrays, "skin_weights"),
        skin_joints=_get(arrays, "skin_joints", np.int64),
        joint_names=binding_info.get("joints"),
    )
    garment = _read_geometry(folder / data.get("geometry", paths["glb"].name))
    groups = data.get("groups")
    if groups and len(groups) >= len(garment.groups):
        index = {g: k for k, g in enumerate(groups)}
        garment.triangle_group = np.array(
            [index[garment.groups[g]] for g in garment.triangle_group], np.int64
        )
        garment.groups = list(groups)
    covers = arrays.get("covers")
    wearable = Wearable(
        name=data["name"],
        label=data.get("label"),
        slot=data["slot"],
        layer=int(data["layer"]),
        garment=garment,
        binding=binding,
        provenance=data.get("provenance", PUBLIC),
        licence=data.get("licence") or {},
        weights=data.get("weights", "body"),
        covers=None if covers is None else covers.astype(bool),
        group_covers={
            k.split(":", 1)[1]: v.astype(bool)
            for k, v in arrays.items()
            if k.startswith("covers:")
        },
        cover_ray=bool(data.get("cover_ray", False)),
        hides=list(data.get("hides") or []),
        springs=list(data.get("springs") or []),
        constraints=list(data.get("constraints") or []),
        extras=dict(data.get("extras") or {}),
    )
    wearable.check()
    return wearable


def _get(arrays, key, dtype=np.float64):
    value = arrays.get(key)
    return None if value is None else np.asarray(value, dtype=dtype)


def _write_geometry(path: pathlib.Path, wearable: Wearable) -> None:
    from opensculptboy.export.body import ANNY_TO_GLTF
    from opensculptboy.export.document import ARRAY_BUFFER, GltfDocument
    from opensculptboy.wearables.gltf import add_materials, index_accessor

    garment = wearable.garment
    doc = GltfDocument("OpenSculptBoy wearable package")
    attributes = {
        "POSITION": doc.accessor(
            (garment.vertices @ ANNY_TO_GLTF.T).astype(np.float32),
            ARRAY_BUFFER,
            minmax=True,
        ),
        "NORMAL": doc.accessor(
            (garment.normals @ ANNY_TO_GLTF.T).astype(np.float32), ARRAY_BUFFER
        ),
    }
    if garment.uvs is not None:
        attributes["TEXCOORD_0"] = doc.accessor(
            np.asarray(garment.uvs, dtype=np.float32), ARRAY_BUFFER
        )
    materials = add_materials(doc, garment.materials)
    primitives = []
    for g, group in enumerate(garment.groups):
        for m in range(len(garment.materials)):
            rows = (garment.triangle_group == g) & (garment.triangle_material == m)
            if not rows.any():
                continue
            primitives.append(
                {
                    "attributes": dict(attributes),
                    "indices": index_accessor(
                        doc, garment.triangles[rows], len(garment.vertices)
                    ),
                    "material": materials[m],
                    "mode": 4,
                    "extras": {"group": group},
                }
            )
    mesh = doc.add_mesh({"name": wearable.name, "primitives": primitives})
    node = doc.add_node({"name": wearable.name, "mesh": mesh})
    doc.scene_name = wearable.name
    doc.scene_nodes = [node]
    doc.scene_extras = {
        "opensculptboy_wearable": {
            "format": FORMAT,
            "provenance": wearable.provenance,
            "groups": list(garment.groups),
        }
    }
    doc.write(path)


def _read_geometry(path: pathlib.Path) -> Garment:
    from opensculptboy.fit.handoff import GLTF_TO_ANNY
    from opensculptboy.mouth.handoff import accessor, read_glb

    gltf, binary = read_glb(path)
    mesh = gltf["meshes"][0]
    first = mesh["primitives"][0]["attributes"]
    vertices = accessor(gltf, binary, first["POSITION"]).astype(np.float64)
    normals = accessor(gltf, binary, first["NORMAL"]).astype(np.float64)
    uvs = None
    if "TEXCOORD_0" in first:
        uvs = accessor(gltf, binary, first["TEXCOORD_0"]).astype(np.float64)
    from opensculptboy.wearables.gltf import read_materials

    materials, material_index = read_materials(gltf, binary)
    groups: list[str] = []
    triangles, tri_material, tri_group = [], [], []
    for primitive in mesh["primitives"]:
        group = (primitive.get("extras") or {}).get("group", "main")
        if group not in groups:
            groups.append(group)
        t = accessor(gltf, binary, primitive["indices"]).reshape(-1, 3).astype(np.int64)
        triangles.append(t)
        tri_material.append(
            np.full(len(t), material_index.get(primitive.get("material"), 0))
        )
        tri_group.append(np.full(len(t), groups.index(group)))
    return Garment(
        vertices=vertices @ GLTF_TO_ANNY.T,
        normals=normals @ GLTF_TO_ANNY.T,
        uvs=uvs,
        triangles=np.concatenate(triangles) if triangles else np.zeros((0, 3)),
        triangle_material=np.concatenate(tri_material) if tri_material else None,
        triangle_group=np.concatenate(tri_group) if tri_group else None,
        materials=materials,
        groups=groups or ["main"],
    )

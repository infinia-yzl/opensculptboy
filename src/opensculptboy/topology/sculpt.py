# OpenSculptBoy
# Apache License, Version 2.0
"""
The composed patches of the ``sculpt`` topology: the data file, its loader, and the transform
from anny's quad :class:`~anny.models.model_data.ModelData` to the patched one.

The file ``src/opensculptboy/data/topology/sculpt.safetensors`` holds the tensors and its
sidecar ``sculpt.json`` the rest (format ``opensculptboy/sculpt-topology@1``):

- ``base``: ``"anny-quads"``, and ``base_digest``, the :attr:`BaseQuadMesh.digest` of anny's
  quad mesh; the loader refuses another base;
- ``removed_faces`` (R,): indices into anny's quad faces, sorted;
- ``anchors`` (M, 8) int64 and ``weights`` (M, 8) float64: the new vertices;
- ``faces`` (F, 4): the new faces over vertex indices, N + m being new vertex m (a triangle
  repeats its last index);
- ``uv_faces`` (F, 4), ``uvs`` (J, 2), ``uv_anchors`` and ``uv_weights`` (J, 8): the new
  faces' UV indices (U + j being new UV j) and the new UVs, appended to anny's UVs;
- ``vertex_labels`` (M,) and ``face_labels`` (F,): indices into the sidecar's ``labels``;
- in the sidecar, per patch: its name, module, options, and its ranges of vertices, faces and
  UVs; and the SHA-256 of the tensor file, which the loader checks.

Faces of ``sculpt-quads`` are anny's quads without the removed ones, in anny's order, then the
new faces. ``sculpt`` is its triangulation, as ``anny`` is that of ``anny-quads``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import json
import os
import pathlib
from typing import Iterable, Sequence

import numpy as np
import torch

from opensculptboy.topology.patch import (
    MAX_ANCHORS,
    BaseQuadMesh,
    Patch,
    PatchError,
    _corners,
    _directed_edges,
    base_from_model_data,
    base_quad_mesh,
    validate_patch,
)

FORMAT = "opensculptboy/sculpt-topology@1"
BASE = "anny-quads"
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "topology"
DEFAULT_PATH = DATA_DIR / "sculpt.safetensors"
# Set to the path of another sculpt.safetensors to use it in place of the shipped file.
PATH_VARIABLE = "OPENSCULPTBOY_SCULPT_TOPOLOGY"
# Vertices of MakeHuman's base.obj; the new vertices take base-mesh indices from here on, so a
# lookup of a base-mesh table at a new vertex fails loudly instead of reading another vertex.
MAKEHUMAN_VERTEX_COUNT = 19158
TOPOLOGIES = ("sculpt", "sculpt-quads", "sculpt-tris")


@dataclasses.dataclass
class SculptData:
    """The composed patches (see the module docstring for the fields)."""

    base_digest: str
    vertex_count: int  # N, anny's vertices
    uv_count: int  # U, anny's UVs
    quad_count: int  # anny's quad faces
    removed_faces: np.ndarray
    anchors: np.ndarray
    weights: np.ndarray
    faces: np.ndarray
    uv_faces: np.ndarray
    uvs: np.ndarray
    uv_anchors: np.ndarray
    uv_weights: np.ndarray
    vertex_labels: np.ndarray
    face_labels: np.ndarray
    labels: list[str]
    patches: list[dict]

    @property
    def new_vertex_count(self) -> int:
        return len(self.anchors)

    @property
    def total_vertex_count(self) -> int:
        return self.vertex_count + self.new_vertex_count

    def _tensors(self) -> dict[str, np.ndarray]:
        return {
            "removed_faces": self.removed_faces.astype(np.int64),
            "anchors": self.anchors.astype(np.int64),
            "weights": self.weights.astype(np.float64),
            "faces": self.faces.astype(np.int64),
            "uv_faces": self.uv_faces.astype(np.int64),
            "uvs": self.uvs.astype(np.float64),
            "uv_anchors": self.uv_anchors.astype(np.int64),
            "uv_weights": self.uv_weights.astype(np.float64),
            "vertex_labels": self.vertex_labels.astype(np.int64),
            "face_labels": self.face_labels.astype(np.int64),
        }

    def _header(self) -> dict:
        return {
            "format": FORMAT,
            "base": BASE,
            "base_digest": self.base_digest,
            "counts": {
                "base_vertices": self.vertex_count,
                "base_uvs": self.uv_count,
                "base_quads": self.quad_count,
                "removed_faces": int(len(self.removed_faces)),
                "new_vertices": int(self.new_vertex_count),
                "new_faces": int(len(self.faces)),
                "new_uvs": int(len(self.uvs)),
            },
            "labels": list(self.labels),
            "patches": self.patches,
        }

    @functools.cached_property
    def digest(self) -> str:
        """SHA-256 of the content (tensors and header), for cache keys."""
        h = hashlib.sha256(json.dumps(self._header(), sort_keys=True).encode())
        for name, array in sorted(self._tensors().items()):
            h.update(name.encode())
            h.update(np.ascontiguousarray(array).tobytes())
        return h.hexdigest()

    def kept_faces(self) -> np.ndarray:
        """(F,) mask of anny's quads that ``sculpt-quads`` keeps."""
        keep = np.ones(self.quad_count, dtype=bool)
        keep[self.removed_faces] = False
        return keep

    def positions(self, vertices) -> np.ndarray:
        """(N + M, 3) the vertices of ``sculpt`` from anny's (N, 3) vertices (NumPy)."""
        v = np.asarray(vertices)
        new = np.einsum("mk,mk...->m...", self.weights, v[self.anchors])
        return np.concatenate([v, new])

    def write(self, path: str | os.PathLike = DEFAULT_PATH) -> pathlib.Path:
        """Write ``path`` (``.safetensors``) and its ``.json`` sidecar."""
        import safetensors.torch

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {k: torch.from_numpy(v.copy()) for k, v in self._tensors().items()}
        safetensors.torch.save_file(tensors, str(path))
        header = self._header()
        header["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".json").write_text(
            json.dumps(header, indent=2) + "\n", encoding="utf-8"
        )
        return path


def _sidecar(path: pathlib.Path) -> pathlib.Path:
    return path.with_suffix(".json")


def read_sculpt_data(
    path: str | os.PathLike = DEFAULT_PATH, base: BaseQuadMesh | None = None
) -> SculptData:
    """
    Read a sculpt file and its sidecar. Raise ValueError when the tensor file does not match
    the sidecar's SHA-256, or when the file was built on another base than anny's current quad
    mesh (``base``, by default :func:`~opensculptboy.topology.patch.base_quad_mesh`).
    """
    import safetensors.torch

    path = pathlib.Path(path)
    header = json.loads(_sidecar(path).read_text(encoding="utf-8"))
    if header.get("format") != FORMAT or header.get("base") != BASE:
        raise ValueError(
            f"{path} is not a sculpt topology file of format {FORMAT} on {BASE}."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != header.get("sha256"):
        raise ValueError(
            f"{path} does not match its sidecar {_sidecar(path).name}: rebuild it with "
            "python -m opensculptboy.topology.build."
        )
    base = base_quad_mesh() if base is None else base
    if header["base_digest"] != base.digest:
        raise ValueError(
            f"{path} was built on another version of anny's quad mesh; rebuild it with "
            "python -m opensculptboy.topology.build."
        )
    t = {k: v.numpy() for k, v in safetensors.torch.load_file(str(path)).items()}
    counts = header["counts"]
    return SculptData(
        base_digest=header["base_digest"],
        vertex_count=counts["base_vertices"],
        uv_count=counts["base_uvs"],
        quad_count=counts["base_quads"],
        labels=list(header["labels"]),
        patches=list(header["patches"]),
        **t,
    )


_OVERRIDES: list[SculptData] = []


@contextlib.contextmanager
def use_sculpt_data(data: SculptData):
    """Use ``data`` as the ``sculpt`` topology inside the block (for tests and authoring)."""
    _OVERRIDES.append(data)
    try:
        yield data
    finally:
        _OVERRIDES.pop()


@functools.lru_cache(maxsize=4)
def _cached_file(path: str, mtime: float) -> SculptData:
    del mtime
    return read_sculpt_data(path)


def sculpt_data() -> SculptData:
    """The current ``sculpt`` topology: the innermost :func:`use_sculpt_data`, else the file
    named by ``OPENSCULPTBOY_SCULPT_TOPOLOGY``, else the shipped file."""
    if _OVERRIDES:
        return _OVERRIDES[-1]
    path = pathlib.Path(os.environ.get(PATH_VARIABLE) or DEFAULT_PATH)
    return _cached_file(str(path), path.stat().st_mtime)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def compose(
    names: Sequence[str] | None = None,
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
    base: BaseQuadMesh | None = None,
    options: dict[str, dict] | None = None,
) -> SculptData:
    """
    Compose the patches of :mod:`opensculptboy.topology.patches` in their order.

    ``names`` lists the patches to use (default: the modules with ``ENABLED``, plus
    ``include``, minus ``exclude``); ``options`` maps a patch name to keyword arguments of
    its ``build``. Each patch is validated (:func:`validate_patch`), and a face that two
    patches remove is refused.
    """
    from opensculptboy.topology.patches import patch_modules

    base = base_quad_mesh() if base is None else base
    modules = patch_modules()
    known = {m.NAME for m in modules}
    wanted = (
        set(names)
        if names is not None
        else {m.NAME for m in modules if getattr(m, "ENABLED", True)} | set(include)
    ) - set(exclude)
    unknown = sorted(wanted - known)
    if unknown:
        raise PatchError(f"Unknown patches {unknown}; the patches are {sorted(known)}.")
    built: list[tuple[object, Patch]] = []
    for module in modules:
        if module.NAME not in wanted:
            continue
        kwargs = dict(getattr(module, "OPTIONS", {}))
        kwargs.update((options or {}).get(module.NAME, {}))
        patch = module.build(base, **kwargs)
        if patch.name != module.NAME:
            raise PatchError(
                f"Module {module.__name__} returned patch {patch.name!r}, not {module.NAME!r}."
            )
        if not patch.options:
            patch.options = kwargs
        validate_patch(patch, base)
        built.append((module, patch))
    return compose_patches([p for _, p in built], base, [m for m, _ in built])


def compose_patches(
    patches: Sequence[Patch], base: BaseQuadMesh, modules: Sequence | None = None
) -> SculptData:
    """Compose validated patches in the given order into one :class:`SculptData`."""
    n, u = base.vertex_count, base.uv_count
    claimed: dict[int, str] = {}
    labels: list[str] = []
    anchors, weights, faces, uv_faces = [], [], [], []
    uv_anchors, uv_weights, vertex_labels, face_labels = [], [], [], []
    records = []
    m_total = j_total = f_total = 0
    for i, patch in enumerate(patches):
        for f in patch.removed_faces.tolist():
            if f in claimed:
                raise PatchError(
                    f"Face {f} is removed by both {claimed[f]!r} and {patch.name!r}."
                )
            claimed[f] = patch.name
        m, j = patch.vertex_count, len(patch.uv_anchors)
        faces.append(np.where(patch.faces >= n, patch.faces + m_total, patch.faces))
        uv_faces.append(
            np.where(patch.uv_faces >= u, patch.uv_faces + j_total, patch.uv_faces)
        )
        anchors.append(patch.anchors)
        weights.append(patch.weights)
        uv_anchors.append(patch.uv_anchors)
        uv_weights.append(patch.uv_weights)
        for kind, target in (
            (patch.labels_of_vertices(), vertex_labels),
            (patch.labels_of_faces(), face_labels),
        ):
            for label in kind:
                if label not in labels:
                    labels.append(label)
                target.append(labels.index(label))
        module = modules[i] if modules is not None else None
        records.append(
            {
                "name": patch.name,
                "module": None if module is None else module.__name__,
                "options": patch.options,
                "removed_faces": int(len(patch.removed_faces)),
                "vertices": [n + m_total, n + m_total + m],
                "faces": [f_total, f_total + len(patch.faces)],
                "uvs": [u + j_total, u + j_total + j],
                "symmetric": bool(patch.symmetric),
            }
        )
        m_total += m
        j_total += j
        f_total += len(patch.faces)

    def cat(parts, shape, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(shape, dtype)

    data = SculptData(
        base_digest=base.digest,
        vertex_count=n,
        uv_count=u,
        quad_count=len(base.quads),
        removed_faces=np.array(sorted(claimed), dtype=np.int64),
        anchors=cat(anchors, (0, MAX_ANCHORS), np.int64),
        weights=cat(weights, (0, MAX_ANCHORS), np.float64),
        faces=cat(faces, (0, 4), np.int64),
        uv_faces=cat(uv_faces, (0, 4), np.int64),
        uvs=np.zeros((0, 2)),
        uv_anchors=cat(uv_anchors, (0, MAX_ANCHORS), np.int64),
        uv_weights=cat(uv_weights, (0, MAX_ANCHORS), np.float64),
        vertex_labels=np.array(vertex_labels, dtype=np.int64),
        face_labels=np.array(face_labels, dtype=np.int64),
        labels=labels,
        patches=records,
    )
    data.uvs = np.einsum("jk,jkd->jd", data.uv_weights, base.uvs[data.uv_anchors])
    check_composed(data, base)
    return data


def check_composed(data: SculptData, base: BaseQuadMesh) -> None:
    """Raise :class:`PatchError` unless the patched quad mesh keeps every directed edge at
    most once (no doubled or flipped face across patches)."""
    quads = np.concatenate([base.quads[data.kept_faces()], data.faces])
    edges = _directed_edges(_corners(quads))
    doubled = [e for e, c in edges.items() if c > 1]
    if doubled:
        raise PatchError(f"The patched mesh uses directed edges twice: {doubled[:5]}.")


# ---------------------------------------------------------------------------
# ModelData and models
# ---------------------------------------------------------------------------


def is_sculpt(topology) -> bool:
    """Whether ``topology`` names the sculpt topology (``sculpt``, ``sculpt-quads``)."""
    return isinstance(topology, str) and topology.split("-")[0] == "sculpt"


def _triangulated(topology: str) -> bool:
    if topology not in TOPOLOGIES:
        raise ValueError(
            f"Unknown sculpt topology {topology!r}; use one of {', '.join(TOPOLOGIES)}."
        )
    return topology != "sculpt-quads"


def sculpt_model_data(
    base_data, data: SculptData | None = None, triangulate: bool = False
):
    """
    The ModelData of ``sculpt-quads`` (or ``sculpt`` with ``triangulate``) from anny's
    ``anny-quads`` ModelData of any rig and blend shape selection.

    Every row goes through :func:`anny.models.model_transforms.apply_retopology` with
    ``check_weights=True``: an identity row for each of anny's vertices, which keeps its index,
    position, blend shapes and skin weights exactly, and the convex rows of the new vertices,
    which carry every blend shape, the skin weights and the landmarks. The bone data does not
    depend on the vertices and carries over. Anny's vertices that no face uses any more stay
    in place, unreferenced.
    """
    from anny.models.model_transforms import apply_retopology

    data = sculpt_data() if data is None else data
    base = base_from_model_data(base_data)
    if base.digest != data.base_digest:
        raise ValueError(
            "The sculpt topology was built on another version of anny's quad mesh; rebuild "
            "it with python -m opensculptboy.topology.build."
        )
    n, m = data.vertex_count, data.new_vertex_count
    template = base_data.template_vertices
    dtype, device = template.dtype, template.device
    width = int((data.weights > 0).sum(1).max(initial=1))
    width = max(width, 1)
    indices = torch.zeros((n + m, width), dtype=torch.int64, device=device)
    weights = torch.zeros((n + m, width), dtype=dtype, device=device)
    indices[:n, 0] = torch.arange(n, device=device)
    weights[:n, 0] = 1
    indices[n:] = torch.as_tensor(data.anchors[:, :width], device=device)
    weights[n:] = torch.as_tensor(data.weights[:, :width], dtype=dtype, device=device)
    new_vertices = sum(
        template[indices[n:, k]] * weights[n:, k, None] for k in range(width)
    )
    vertices = torch.cat([template, new_vertices]) if m else template.clone()
    keep = torch.as_tensor(data.kept_faces(), device=device)
    faces = torch.cat(
        [base_data.faces[keep], torch.as_tensor(data.faces, device=device)]
    )
    base_index = torch.cat(
        [
            base_data.base_mesh_vertex_indices,
            MAKEHUMAN_VERTEX_COUNT + torch.arange(m, dtype=torch.int64, device=device),
        ]
    )
    # The blend shapes pass through apply_retopology without their rows, and their rows follow
    # its formula below: an identity row gives anny's rows bit for bit, so they are copied, and
    # the new rows are summed as apply_retopology sums them. This saves K gathers of the whole
    # (blend shapes, N, 3) tensor (several GB with every local change and face shape).
    blendshapes = base_data.blendshapes
    out = apply_retopology(
        dataclasses.replace(base_data, blendshapes=blendshapes[:0]),
        vertices=vertices,
        faces=faces,
        reference_vertex_indices=indices,
        barycentric_coordinates=weights,
        base_mesh_vertex_indices=base_index,
        check_weights=True,
    )
    # Anny's own rows keep their skin weights bit for bit (apply_retopology renormalises).
    vbw, vbi = out.vertex_bone_weights, out.vertex_bone_indices
    k = max(vbw.shape[1], base_data.vertex_bone_weights.shape[1])
    vbw = _pad(vbw.to(base_data.vertex_bone_weights.dtype), k)
    vbi = _pad(vbi.to(base_data.vertex_bone_indices.dtype), k)
    vbw[:n] = _pad(base_data.vertex_bone_weights, k)
    vbi[:n] = _pad(base_data.vertex_bone_indices, k)
    uvs = base_data.texture_coordinates
    new_rows = sum(
        blendshapes[:, indices[n:, k]] * weights[None, n:, k, None]
        for k in range(width)
    )
    out = dataclasses.replace(
        out,
        blendshapes=torch.cat([blendshapes, new_rows], 1) if m else blendshapes,
        vertex_bone_weights=vbw,
        vertex_bone_indices=vbi,
        texture_coordinates=torch.cat(
            [uvs, torch.as_tensor(data.uvs, dtype=uvs.dtype, device=uvs.device)]
        ),
        face_texture_coordinate_indices=torch.cat(
            [
                base_data.face_texture_coordinate_indices[keep],
                torch.as_tensor(data.uv_faces, device=device),
            ]
        ),
    )
    return triangulate_mixed(out) if triangulate else out


def _pad(t: torch.Tensor, width: int) -> torch.Tensor:
    if t.shape[1] == width:
        return t.clone()
    pad = torch.zeros((t.shape[0], width - t.shape[1]), dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], 1)


def triangulate_mixed(model_data):
    """Triangulate a ModelData whose (F, 4) faces mix quads and triangles ``[a, b, c, c]``:
    the quads as anny triangulates them, the triangles kept, in face order."""
    from anny.utils.mesh_utils import triangulate_faces_with_texture_coordinates

    faces = _corners(model_data.faces.cpu().numpy())
    uv_faces = _corners(model_data.face_texture_coordinate_indices.cpu().numpy())
    tri, tri_uv = triangulate_faces_with_texture_coordinates(
        vertices=model_data.template_vertices,
        faces=faces,
        face_texture_coordinate_indices=uv_faces,
    )
    return dataclasses.replace(
        model_data,
        faces=torch.as_tensor(tri, dtype=torch.int64),
        face_texture_coordinate_indices=torch.as_tensor(tri_uv, dtype=torch.int64),
    )


def sculpt_model(
    rig="anny",
    topology: str = "sculpt",
    local_changes="none",
    facial_actions="none",
    phenotypes="default",
    extrapolate_phenotypes: bool = False,
    pose_parameterization="local-ref",
    skinning_method=None,
    face_shapes="none",
    scale_face_shapes: bool = True,
    data: SculptData | None = None,
):
    """
    An Anny model on the ``sculpt`` (or ``sculpt-quads``) topology, with the arguments of
    :class:`anny.Anny`. ``data`` defaults to :func:`sculpt_data`.
    """
    import anny
    from anny.models import build_model_data
    from anny.models.model_data import AnnyModelConfig, RigConfig, TopologyConfig

    from opensculptboy.mouth.rig import is_mouth_rig, mouth_model_data, mouth_rig_config

    triangulate = _triangulated(topology)
    if is_mouth_rig(rig):
        # the anny-mouth rig of opensculptboy.mouth, which anny's RigConfig does not name
        rig_config = mouth_rig_config()
        base_data = mouth_model_data(
            topology=TopologyConfig.from_string(BASE),
            local_changes=local_changes,
            facial_actions=facial_actions,
            face_shapes=face_shapes,
        )
    else:
        rig_config = RigConfig.from_string(rig) if isinstance(rig, str) else rig
        base_data = build_model_data(
            rig=rig_config,
            topology=TopologyConfig.from_string(BASE),
            local_changes=local_changes,
            facial_actions=facial_actions,
            face_shapes=face_shapes,
        )
    data = sculpt_data() if data is None else data
    model_data = sculpt_model_data(base_data, data, triangulate=triangulate)
    model = anny.Anny.from_model_data(
        model_data,
        skinning_method=skinning_method,
        pose_parameterization=pose_parameterization,
        bone_orientation=rig_config.bone_orientation,
        root_identity_orientation=rig_config.root_identity_orientation,
        phenotypes=phenotypes,
        extrapolate_phenotypes=extrapolate_phenotypes,
        scale_face_shapes=scale_face_shapes,
    )
    model.config = AnnyModelConfig(
        rig=rig,
        topology=topology,
        local_changes=local_changes,
        facial_actions=facial_actions,
        extrapolate_phenotypes=extrapolate_phenotypes,
        phenotypes=phenotypes,
        pose_parameterization=pose_parameterization,
        skinning_method=skinning_method,
        face_shapes=face_shapes,
        scale_face_shapes=scale_face_shapes,
    )
    # the lookups of opensculptboy.topology.lookup (anchors, labels) and the VRM skin-weight
    # digest of anny's rows (vrm_tables.skin_weight_digest) read these
    model.sculpt_data = data
    model.sculpt_bone_slots = int(base_data.vertex_bone_weights.shape[1])
    return model


def build_model(topology: str = "anny", **kwargs):
    """An Anny model of any topology: :func:`sculpt_model` for the sculpt topologies,
    :class:`anny.Anny` otherwise (same keyword arguments)."""
    if is_sculpt(topology):
        return sculpt_model(topology=topology, **kwargs)
    from opensculptboy.mouth.rig import is_mouth_rig, mouth_model

    if is_mouth_rig(kwargs.get("rig")):
        return mouth_model(topology=topology, **kwargs)
    import anny

    return anny.Anny(topology=topology, **kwargs)

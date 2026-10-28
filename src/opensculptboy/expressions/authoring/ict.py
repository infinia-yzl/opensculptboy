# OpenSculptBoy
# Apache License, Version 2.0
"""
ICT-FaceKit's expression shapes on Anny's face.

The ICT Face Model Light (ICT-FaceKit, MIT licence; Li et al., CVPR 2020) holds 53 expression
shapes on its generic neutral face. They follow ARKit's names with ``_L`` and ``_R`` for the
sides, and split ``browInnerUp`` and ``cheekPuff`` into a left and a right shape; it has no
``tongueOut``. Two more shapes (``cheekRaiser`` and ``PupilDilate``, each per side) have no
ARKit name. :data:`ARKIT_FROM_ICT` gives each of Anny's facial actions the ICT shapes whose sum
stands for it.

Anny's face shapes are already fitted to the ICT face (:mod:`anny.faces.authoring.fit_3d`), and
the same registration carries the expressions:

1. **Registration.** :func:`anny.faces.authoring.fit_3d.register` lays Anny's head onto ICT's
   neutral face (landmarks, a fit of Anny's face shapes, then a non-rigid refinement), and gives
   each vertex of Anny's head a point of ICT's surface. The registration leaves out the ears,
   the eye and mouth cavities and the eyes.
2. **Sides.** Near the mouth and the eyes, the closest point may lie on the other lip or lid.
   :func:`side_correspondences` therefore searches each vertex of a lip or a lid only on the
   same lip or lid of ICT's face: the share of each vertex in ``jawOpen`` tells the lower lip
   from the upper lip, and its share in ``eyeBlinkLeft`` or ``eyeBlinkRight`` the upper lid from
   the lower lid, on both faces.
3. **Delta.** Each corresponding vertex takes ICT's offset at its point (:func:`sample`), turned
   into Anny's frame (:func:`to_anny_frame`) and scaled by the size of the character's head
   against the registered head.
4. **Smoothing and extension.** :func:`extend` solves for the difference to Anny's own facial
   action over the whole mesh: it keeps the transferred offsets on the corresponding vertices
   up to a smoothing term, and carries the difference smoothly into what ICT's face lacks (the
   cavities, the ears, the neck below the chin), so that the inside of the mouth and the tongue
   keep Anny's own motion with the lips' correction. The eyeballs, which no edge joins to the
   skin, keep Anny's own motion.

The ICT files stay in ``ANNY_CACHE_DIR/faces/ICT-FaceKit`` (``anny.faces.authoring.sources``);
the repository keeps no ICT data.

Usage (exports a GLB file whose facial actions take the transferred shapes)::

    python -m opensculptboy.expressions.authoring.ict --card card.json --out out.glb \\
        [--only mouthSmileLeft,jawOpen] [--eyes] [--teeth] [--smooth 0.3]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

import numpy as np

# Anny's facial action -> the ICT shapes whose sum stands for it. "_L" is the character's own
# left, as Anny's "Left" (checked by check_sides on the data).
ARKIT_FROM_ICT: dict[str, tuple[str, ...]] = {}
for _name in (
    "browDown browOuterUp cheekSquint eyeBlink eyeLookDown eyeLookIn eyeLookOut eyeLookUp "
    "eyeSquint eyeWide mouthDimple mouthFrown mouthLowerDown mouthPress mouthSmile "
    "mouthStretch mouthUpperUp noseSneer"
).split():
    ARKIT_FROM_ICT[_name + "Left"] = (_name + "_L",)
    ARKIT_FROM_ICT[_name + "Right"] = (_name + "_R",)
for _name in (
    "jawForward jawLeft jawOpen jawRight mouthClose mouthFunnel mouthLeft mouthPucker "
    "mouthRight mouthRollLower mouthRollUpper mouthShrugLower mouthShrugUpper"
).split():
    ARKIT_FROM_ICT[_name] = (_name,)
ARKIT_FROM_ICT["browInnerUp"] = ("browInnerUp_L", "browInnerUp_R")
ARKIT_FROM_ICT["cheekPuff"] = ("cheekPuff_L", "cheekPuff_R")
# ICT's shapes without an ARKit name, and the ARKit action without an ICT shape
ICT_EXTRA = ("cheekRaiser_L", "cheekRaiser_R", "PupilDilate_L", "PupilDilate_R")
ICT_MISSING = ("tongueOut",)
# The vertices of ICT's face and of its head and neck (README of ICT-FaceKit)
SKIN_VERTICES = 11248
# The share of a vertex in jawOpen or eyeBlink at or above which it belongs to the lower lip or
# the upper lid, and the difference of shares that marks a closest point on the wrong side
SIDE_THRESHOLD = 0.35
SIDE_GAP = 0.3


@dataclasses.dataclass
class IctExpressions:
    """ICT's skin (face, head and neck) in Anny's frame (metres) with its expression offsets."""

    neutral: np.ndarray  # (N, 3)
    triangles: np.ndarray  # (T, 3)
    names: list[str]  # the ICT shape names
    deltas: np.ndarray  # (E, N, 3) the offsets of each shape at weight 1

    def shape(self, name: str) -> np.ndarray:
        return self.deltas[self.names.index(name)]

    def arkit(self, action: str) -> np.ndarray:
        """(N, 3) the offsets that stand for one of Anny's facial actions."""
        return sum(self.shape(n) for n in ARKIT_FROM_ICT[action])


def read_obj_vertices(path, count: int | None = None) -> np.ndarray:
    """The first ``count`` vertex positions of an OBJ file (all when None)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                rows.append(line.split()[1:4])
                if count is not None and len(rows) == count:
                    break
    return np.asarray(rows, dtype=np.float64)


def load_expressions(root: pathlib.Path | None = None) -> IctExpressions:
    """ICT's expression shapes, read from ``FaceXModel`` (``root``: the ICT-FaceKit folder)
    and cached as ``ict_expressions.npz`` next to it."""
    from anny.faces.authoring import ict

    if root is None:
        from anny.faces.authoring.sources import fetch_ict_facekit

        root = fetch_ict_facekit()
    root = pathlib.Path(root)
    cached = root.parent / "ict_expressions.npz"
    if cached.exists():
        with np.load(cached) as z:
            return IctExpressions(
                neutral=z["neutral"],
                triangles=z["triangles"],
                names=[str(n) for n in z["names"]],
                deltas=z["deltas"],
            )
    model = root / "FaceXModel"
    names = json.loads((model / "vertex_indices.json").read_text(encoding="utf-8"))[
        "expressions"
    ]
    names = list(names) + [n for n in ICT_EXTRA if (model / f"{n}.obj").exists()]
    neutral = read_obj_vertices(model / "generic_neutral_mesh.obj", SKIN_VERTICES)
    deltas = np.stack(
        [
            (read_obj_vertices(model / f"{n}.obj", SKIN_VERTICES) - neutral)
            @ ict.TO_ANNY.T
            for n in names
        ]
    )
    out = IctExpressions(
        neutral=neutral @ ict.TO_ANNY.T,
        triangles=ict.load().triangles,
        names=names,
        deltas=deltas,
    )
    np.savez(
        cached,
        neutral=out.neutral,
        triangles=out.triangles,
        names=np.array(names),
        deltas=out.deltas,
    )
    return out


# ------------------------------------------------------------------ geometry
def share(delta: np.ndarray, quantile: float = 95.0) -> np.ndarray:
    """(N,) each vertex's displacement in a shape against the shape's largest displacements
    (the ``quantile`` of the moved vertices), at most 1."""
    size = np.linalg.norm(delta, axis=1)
    moved = size[size > 1e-3 * size.max()] if size.max() > 0 else size
    top = np.percentile(moved, quantile) if len(moved) else 0.0
    return np.minimum(size / top, 1.0) if top > 0 else size


def side_correspondences(
    points: np.ndarray,
    point_shares: list[np.ndarray],
    target_V: np.ndarray,
    target_T: np.ndarray,
    target_shares: list[np.ndarray],
    threshold: float = SIDE_THRESHOLD,
    gap: float = SIDE_GAP,
):
    """
    The closest triangle of ``(target_V, target_T)`` for each point, with barycentric
    coordinates and the distance, kept on the same side of each moving part.

    ``point_shares[k]`` and ``target_shares[k]`` give each point and each target vertex its
    share in the k-th moving part (``jawOpen``, ``eyeBlinkLeft``, ...). A point whose closest
    point lies on the other side of ``threshold`` in some part, with shares that differ by more
    than ``gap`` (an upper-lip vertex that lands on the lower lip, for one), searches again
    among the target triangles of its own side only: all three corners at or above
    ``threshold`` when the point's share is at or above it, all three below it otherwise.
    """
    from anny.hair import closest_triangles

    tri, bary, dist = closest_triangles(points, target_V, target_T)
    for p, t in zip(point_shares, target_shares):
        hit = np.einsum("hk, hk -> h", bary, t[target_T[tri]])
        wrong = ((p >= threshold) != (hit >= threshold)) & (np.abs(p - hit) > gap)
        for upper in (True, False):
            rows = np.nonzero(wrong & ((p >= threshold) == upper))[0]
            if not len(rows):
                continue
            corner = t[target_T]
            keep = (
                (corner >= threshold).all(1) if upper else (corner < threshold).all(1)
            )
            sub = np.nonzero(keep)[0]
            tt, bb, dd = closest_triangles(points[rows], target_V, target_T[sub])
            tri[rows], bary[rows], dist[rows] = sub[tt], bb, dd
    return tri, bary, dist


def sample(delta: np.ndarray, triangles: np.ndarray, tri, bary) -> np.ndarray:
    """(H, 3) the offsets ``delta`` (N, 3) at points given by triangle and barycentric weights."""
    return np.einsum("hk, hkd -> hd", bary, delta[triangles[tri]])


def to_anny_frame(
    delta: np.ndarray, align: np.ndarray, fit: np.ndarray, scale: float = 1.0
) -> np.ndarray:
    """
    Offsets of ICT's face (in Anny's axes, as :class:`IctExpressions` holds them) as offsets of
    Anny's head: ``align`` is the rotation that laid ICT's face onto Anny's landmarks
    (``x -> align x + t``) and ``fit`` the rotation of the fitted Anny head onto ICT's face
    (``x -> fit x + t``), so an offset ``d`` of ICT's face is ``fit^T align d`` on Anny's head.
    """
    return scale * (delta @ align.T) @ fit


def edges_of(faces: np.ndarray) -> np.ndarray:
    """(E, 2) the unique edges of polygons (F, k), each as (low, high)."""
    faces = np.asarray(faces)
    k = faces.shape[1]
    e = np.concatenate([faces[:, [i, (i + 1) % k]] for i in range(k)])
    e = e[e[:, 0] != e[:, 1]]
    return np.unique(np.sort(e, axis=1), axis=0)


def normalised_laplacian(n: int, edges: np.ndarray):
    """L = I - D^-1 A of the edge graph (vertices without an edge get a zero row)."""
    import scipy.sparse

    i = np.concatenate([edges[:, 0], edges[:, 1]])
    j = np.concatenate([edges[:, 1], edges[:, 0]])
    A = scipy.sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    degree = np.asarray(A.sum(1)).ravel()
    inv = np.where(degree > 0, 1.0 / np.maximum(degree, 1.0), 0.0)
    eye = scipy.sparse.diags((degree > 0).astype(np.float64))
    return (eye - scipy.sparse.diags(inv) @ A).tocsr()


def extend(
    n: int,
    edges: np.ndarray,
    known: np.ndarray,
    values: np.ndarray,
    prior: np.ndarray | None = None,
    fixed: np.ndarray | None = None,
    smooth: float = 0.3,
    ridge: float = 1e-6,
) -> np.ndarray:
    """
    A smooth offset field (n, 3) that follows ``values`` (n, 3) on the ``known`` vertices.

    The field is ``prior`` (zero when None) plus a correction ``C`` that minimises::

        sum_known |C_i - (values_i - prior_i)|^2 + smooth |L C|^2 + ridge |C|^2

    with ``L`` the normalised graph Laplacian of ``edges``, and ``C_i = -prior_i`` on the
    ``fixed`` vertices (the field is zero there). The squared Laplacian carries the correction
    smoothly into the vertices that are neither known nor fixed; a part that no edge joins to a
    known vertex keeps the prior.
    """
    import scipy.sparse
    import scipy.sparse.linalg

    prior = np.zeros((n, 3)) if prior is None else np.asarray(prior, np.float64)
    known = np.asarray(known, bool)
    w = known.astype(np.float64)
    target = np.zeros((n, 3))
    target[known] = np.asarray(values, np.float64)[known] - prior[known]
    if fixed is not None:
        fixed = np.asarray(fixed, bool) & ~known
        w[fixed] = 1e3
        target[fixed] = -prior[fixed]
    L = normalised_laplacian(n, edges)
    A = (scipy.sparse.diags(w + ridge) + smooth * (L.T @ L)).tocsc()
    solve = scipy.sparse.linalg.factorized(A)
    rhs = w[:, None] * target
    C = np.stack([solve(rhs[:, k]) for k in range(3)], 1)
    return prior + C


def check_sides(
    names_to_deltas: dict[str, np.ndarray], V: np.ndarray
) -> dict[str, float]:
    """For each shape, the displacement-weighted mean x of the moved vertices (positive: the
    character's left side in Anny's frame)."""
    out = {}
    for name, d in names_to_deltas.items():
        w = np.linalg.norm(d, axis=1)
        out[name] = float((w * V[:, 0]).sum() / max(w.sum(), 1e-12))
    return out


# ------------------------------------------------------------------ registration
def registration(cache: bool = True) -> dict:
    """The registration of Anny's head on ICT's neutral face (fit_3d.register), cached as
    ``ict_expression_registration.npz`` in ``ANNY_CACHE_DIR/faces``."""
    from anny.faces.authoring import fit_3d, ict
    from anny.faces.authoring.sources import cache_dir

    path = cache_dir() / "ict_expression_registration.npz"
    if cache and path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    model = fit_3d.fitting_model()
    ict_model = ict.load()
    reg = fit_3d.register(
        model, ict_model.V, ict_model.triangles, ict_model.landmarks()
    )
    out = dict(
        vertex_ids=reg["vertex_ids"],
        refined=reg["refined"],
        target_V=reg["target_V"],
        align=reg["rotation"],
        align_translation=reg["translation"],
        fit=reg["neutral_fit"].rotation[0].numpy(),
        fit_translation=reg["neutral_fit"].translation[0].numpy(),
    )
    np.savez(path, **out)
    return out


@dataclasses.dataclass
class Transfer:
    """The correspondence of Anny's head vertices with ICT's face, ready to carry shapes."""

    vertex_ids: np.ndarray  # (H,) Anny vertices with a point on ICT's face
    triangles: np.ndarray  # (H,) ICT triangle of each
    bary: np.ndarray  # (H, 3)
    distance: np.ndarray  # (H,) metres
    valid: np.ndarray  # (H,) the point lies within max_distance
    align: np.ndarray  # (3, 3)
    fit: np.ndarray  # (3, 3)


def build_transfer(
    ict_shapes: IctExpressions,
    anny_actions: dict[str, np.ndarray],
    reg: dict | None = None,
    max_distance: float = 0.006,
) -> Transfer:
    """The side-aware correspondences of the registered head (``anny_actions``: Anny's own
    ``jawOpen``, ``eyeBlinkLeft`` and ``eyeBlinkRight`` offsets on the whole mesh)."""
    reg = registration() if reg is None else reg
    ids = reg["vertex_ids"]
    parts = ("jawOpen", "eyeBlinkLeft", "eyeBlinkRight")
    point_shares = [share(anny_actions[p])[ids] for p in parts]
    target_shares = [share(ict_shapes.arkit(p)) for p in parts]
    tri, bary, d = side_correspondences(
        reg["refined"],
        point_shares,
        reg["target_V"],
        ict_shapes.triangles,
        target_shares,
    )
    return Transfer(
        vertex_ids=ids,
        triangles=tri,
        bary=bary,
        distance=d,
        valid=d < max_distance,
        align=reg["align"],
        fit=reg["fit"],
    )


def head_scale(rest: np.ndarray, reg: dict) -> float:
    """The size of a character's head (rest vertices, Anny's frame) against the registered
    head: the ratio of their RMS distances from their centroids over the head vertices."""
    ids = reg["vertex_ids"]
    a = rest[ids] - rest[ids].mean(0)
    b = reg["refined"] - reg["refined"].mean(0)
    return float(np.sqrt((a**2).sum(1).mean() / (b**2).sum(1).mean()))


def transfer_action(
    action: str,
    ict_shapes: IctExpressions,
    transfer: Transfer,
    n_vertices: int,
    edges: np.ndarray,
    prior: np.ndarray,
    fixed: np.ndarray,
    scale: float = 1.0,
    smooth: float = 0.3,
) -> np.ndarray:
    """(V, 3) the offsets of one of Anny's facial actions from ICT's shapes."""
    delta = sample(
        ict_shapes.arkit(action),
        ict_shapes.triangles,
        transfer.triangles,
        transfer.bary,
    )
    delta = to_anny_frame(delta, transfer.align, transfer.fit, scale)
    known = np.zeros(n_vertices, bool)
    known[transfer.vertex_ids[transfer.valid]] = True
    values = np.zeros((n_vertices, 3))
    values[transfer.vertex_ids] = delta
    return extend(n_vertices, edges, known, values, prior, fixed, smooth)


def facial_action_rows(model) -> tuple[int, list[str]]:
    """The first row of the facial actions in ``model.blendshapes`` and their labels."""
    return int(model._n_phenotype_blendshapes), list(model.facial_action_labels)


def far_vertices(model, reg: dict, below: float = 0.08) -> np.ndarray:
    """(V,) the vertices more than ``below`` metres under the registered head's lowest
    vertex (in the template): the field is held at zero there."""
    T = model.template_vertices.double().cpu().numpy()
    lowest = T[reg["vertex_ids"], 2].min()
    return T[:, 2] < lowest - below


def transfer_shapes(
    model,
    character=None,
    names=None,
    smooth: float = 0.3,
    ict_shapes: IctExpressions | None = None,
    reg: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """
    ICT's shapes for Anny's facial actions ``names`` (every action ICT has when None) on
    ``model`` (a model with the facial actions), for ``character``'s head size. Returns the
    offsets (V, 3) by name and a report.
    """
    import torch

    from opensculptboy.character import Character

    character = character or Character()
    ict_shapes = load_expressions() if ict_shapes is None else ict_shapes
    reg = registration() if reg is None else reg
    start, labels = facial_action_rows(model)
    B = model.blendshapes.double().cpu().numpy()
    own = {name: B[start + i] for i, name in enumerate(labels)}
    transfer = build_transfer(ict_shapes, own, reg)
    with torch.no_grad():
        rest = (
            model(**character.model_kwargs())["rest_vertices"][0].double().cpu().numpy()
        )
    scale = head_scale(rest, reg)
    n = len(B[0])
    edges = edges_of(model.faces.cpu().numpy())
    fixed = far_vertices(model, reg)
    names = [a for a in labels if a in ARKIT_FROM_ICT] if names is None else names
    sides = check_sides(
        {"browDownLeft": ict_shapes.arkit("browDownLeft")}, ict_shapes.neutral
    )
    if sides["browDownLeft"] <= 0:
        raise ValueError("ICT's _L shapes move the character's right side.")
    shapes, report = (
        {},
        dict(
            scale=scale,
            valid=int(transfer.valid.sum()),
            head_vertices=len(transfer.valid),
            median_distance_mm=float(1000 * np.median(transfer.distance)),
            actions={},
        ),
    )
    for name in names:
        d = transfer_action(
            name, ict_shapes, transfer, n, edges, own[name], fixed, scale, smooth
        )
        shapes[name] = d
        report["actions"][name] = dict(
            ict_max_mm=float(1000 * np.linalg.norm(d, axis=1).max()),
            anny_max_mm=float(1000 * np.linalg.norm(own[name], axis=1).max()),
        )
    return shapes, report


def apply_shapes(model, shapes: dict[str, np.ndarray], strength: float = 1.0) -> None:
    """Replace the facial-action rows of ``model.blendshapes`` with ``shapes`` (in place);
    ``strength`` mixes the new shape with Anny's own (1: the new shape alone)."""
    import torch

    start, labels = facial_action_rows(model)
    with torch.no_grad():
        for name, d in shapes.items():
            row = start + labels.index(name)
            new = torch.as_tensor(d, dtype=model.blendshapes.dtype)
            model.blendshapes[row] = (
                strength * new + (1 - strength) * model.blendshapes[row]
            )


def main(argv=None) -> None:
    import dataclasses as dc

    from opensculptboy.character import Character
    from opensculptboy.export.gltf import export_glb

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--card", default=None, help="a character card (default character)"
    )
    parser.add_argument("--out", required=True, help="the GLB file to write")
    parser.add_argument(
        "--only", default=None, help="facial actions to replace (comma list)"
    )
    parser.add_argument(
        "--eyes", action="store_true", help="the eyes of opensculptboy.eyes"
    )
    parser.add_argument("--teeth", action="store_true", help="generic teeth")
    parser.add_argument("--smooth", type=float, default=0.3)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--report", default=None, help="a JSON file for the report")
    args = parser.parse_args(argv)
    character = (
        Character.from_dict(json.loads(pathlib.Path(args.card).read_text("utf-8")))
        if args.card
        else Character()
    )
    material = dict(character.material)
    if args.eyes:
        material["eyes"] = dict(material.get("eyes") or {})
    if args.teeth:
        material["teeth"] = dict(material.get("teeth") or {})
    character = dc.replace(character, material=material)
    model = character.build_model()
    names = args.only.split(",") if args.only else None
    shapes, report = transfer_shapes(model, character, names, smooth=args.smooth)
    apply_shapes(model, shapes, args.strength)
    summary = export_glb(args.out, character, model)
    report["export"] = {
        k: summary[k] for k in ("vertices", "morph_targets") if k in summary
    }
    text = json.dumps(report, indent=1)
    print(text)
    if args.report:
        pathlib.Path(args.report).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

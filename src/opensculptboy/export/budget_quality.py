# OpenSculptBoy
# Apache License, Version 2.0
"""
Quality measures of a fitted VRM file against the full one (``--fit-budget``).

:func:`compare` takes the :class:`~opensculptboy.export.vrm.VrmSpec` before and after
:func:`opensculptboy.export.fit_budget.fit_spec` and draws both from the views of
:mod:`opensculptboy.render.ids` (the fragments of masked materials whose alpha falls below the
cutoff are discarded, as the apps draw them). It reports:

- **silhouette**: per view, the pixels that one picture covers and the other does not, as a
  share of the full picture's pixels, and as a mean offset of the outline in millimetres
  (those pixels' area over the outline's length);
- **surface**: where both pictures show the same piece, the distance between the two surfaces
  along the view (mm), the angle between their interpolated normals and the angle between
  their shading normals (the normals bent by the normal maps in the MikkTSpace frames that
  the apps derive from the UVs), in degrees, per piece;
- **uv**: per piece, the UV triangles of zero area or flipped against the triangle's own
  winding, before and after;
- **texel density**: per piece, the base colour texels per metre of surface (the square root of
  the texels its UV triangles cover over their area), before and after, and their ratio;
- **colour**: where both pictures show the same piece, the mean difference of the drawn colour
  (base colour, alpha test and a toon light), in 8-bit sRGB steps.

:func:`picture` draws a spec with its textures in that light, for review pages.
"""

from __future__ import annotations

import numpy as np

SIZE = 512
VIEWS = ("front", "front-left", "left", "back", "front-above")


def scene(spec) -> dict:
    """Every drawn triangle of a spec in one mesh: positions (with each mesh's translation),
    normals, UVs, triangles, and each triangle's material and piece name."""
    from opensculptboy.export.pieces import primitive_triangles, spec_pieces

    piece_of = {}
    for piece in spec_pieces(spec):
        for p in piece.primitives:
            piece_of[(piece.mesh, p)] = piece.name
    names = sorted(set(piece_of.values()), key=list(piece_of.values()).index)
    drawn = sorted({n.mesh for n in spec.nodes if n.mesh is not None})
    from opensculptboy.texture.tangents import mikktspace

    P, N, UV, T, M, G, TG, HAS = [], [], [], [], [], [], [], []
    offset = 0
    for m in drawn:
        mesh = spec.meshes[m]
        body = mesh.body
        P.append(np.asarray(body.positions) + np.asarray(mesh.translation))
        N.append(np.asarray(body.normals))
        uv = (
            np.zeros((len(body.positions), 2))
            if body.uvs is None
            else np.asarray(body.uvs)
        )
        UV.append(uv)
        mesh_t = []
        for p, primitive in enumerate(mesh.primitives):
            t = primitive_triangles(mesh, primitive)
            mesh_t.append(t)
            T.append(t + offset)
            M.append(np.full(len(t), primitive.material))
            G.append(np.full(len(t), names.index(piece_of[(m, p)])))
            HAS.append(np.full(len(t), body.uvs is not None))
        tangents = np.tile([1.0, 0.0, 0.0, 1.0], (len(body.positions), 1))
        mapped = any(
            spec.materials[pr.material].normal_texture is not None
            for pr in mesh.primitives
        )
        if (
            mapped and body.uvs is not None
        ):  # the frames that the apps derive (MikkTSpace)
            uv_obj = np.stack([uv[:, 0], 1.0 - uv[:, 1]], 1)
            tangents = mikktspace(
                body.positions, body.normals, uv_obj, np.concatenate(mesh_t)
            )
        TG.append(tangents)
        offset += len(body.positions)
    return {
        "positions": np.concatenate(P),
        "normals": np.concatenate(N),
        "uvs": np.concatenate(UV),
        "triangles": np.concatenate(T),
        "material": np.concatenate(M),
        "piece": np.concatenate(G),
        "has_uvs": np.concatenate(HAS),
        "tangents": np.concatenate(TG),
        "names": names,
    }


class _Textures:
    """The decoded images of a spec, by index."""

    def __init__(self, spec):
        self.spec = spec
        self.cache = {}

    def __call__(self, index):
        if index is None:
            return None
        if index not in self.cache:
            from opensculptboy.export.atlas import decode

            self.cache[index] = decode(self.spec.images[index][1])
        return self.cache[index]


def _sample(image, uv):
    from opensculptboy.texture.bake import _sample

    return _sample(image, uv)


def _draw(spec, sc, camera, textures, light=True):
    """(piece ids (H, W), colour (H, W, 3) uint8 or None, depth, normals) of a scene."""
    from opensculptboy.render.ids import fragments, nearest

    pixel, depth, tri, bary = fragments(sc["positions"], sc["triangles"], camera)
    uv = np.einsum("nk,nkd->nd", bary, sc["uvs"][sc["triangles"][tri]])
    keep = np.ones(len(tri), bool)
    colour = np.zeros((len(tri), 4))
    mats = sc["material"][tri]
    for k in np.unique(mats):
        pick = mats == k
        material = spec.materials[k]
        factor = np.asarray(material.base_color, np.float64)
        image = textures(material.base_texture)
        transform = material.texture_transform
        coords = uv[pick]
        if transform is not None:
            coords = coords * np.asarray(transform.get("scale", [1, 1])) + np.asarray(
                transform.get("offset", [0, 0])
            )
        if image is None:
            c = np.tile(np.r_[_srgb(factor[:3]), factor[3]], (pick.sum(), 1))
        else:
            c = _sample(image, coords)[:, :4] / 255.0
            lin = _linear(c[:, :3]) * factor[:3]
            c = np.concatenate([_srgb(lin), (c[:, 3] * factor[3])[:, None]], 1)
        colour[pick] = c
        if material.alpha_mode == "MASK":
            keep[np.flatnonzero(pick)[c[:, 3] < material.alpha_cutoff]] = False
        elif material.alpha_mode == "BLEND":
            keep[np.flatnonzero(pick)[c[:, 3] < 0.5]] = False
    w, h = camera.size
    pixel, depth, tri, bary, colour = (
        pixel[keep],
        depth[keep],
        tri[keep],
        bary[keep],
        colour[keep],
    )
    first = np.r_[True, pixel[1:] != pixel[:-1]] if len(pixel) else np.zeros(0, bool)
    ids, dimg, bimg = nearest(pixel, depth, tri, bary, camera.size)
    piece = np.where(ids >= 0, sc["piece"][np.maximum(ids, 0)], -1)
    normals = np.zeros((h, w, 3))
    shading = np.zeros((h, w, 3))
    seen = ids >= 0
    if seen.any():
        corners = sc["triangles"][ids[seen]]
        b = bimg[seen]
        n = np.einsum("nk,nkd->nd", b, sc["normals"][corners])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        normals[seen] = n
        shaded = n.copy()
        frame = sc["tangents"][corners]
        t = np.einsum("nk,nkd->nd", b, frame[..., :3])
        t -= (t * n).sum(1, keepdims=True) * n
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        sign = np.where(np.einsum("nk,nk->n", b, frame[..., 3]) < 0, -1.0, 1.0)
        bt = sign[:, None] * np.cross(n, t)
        uv = np.einsum("nk,nkd->nd", b, sc["uvs"][corners])
        mats = sc["material"][ids[seen]]
        for k in np.unique(mats):
            material = spec.materials[k]
            image = textures(material.normal_texture)
            if image is None:
                continue
            pick = mats == k
            coords = uv[pick]
            transform = material.texture_transform
            if transform is not None:
                coords = coords * np.asarray(
                    transform.get("scale", [1, 1])
                ) + np.asarray(transform.get("offset", [0, 0]))
            c = _sample(image, coords)[:, :3] / 127.5 - 1.0
            c[:, :2] *= float(material.normal_scale)
            c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
            v = c[:, :1] * t[pick] + c[:, 1:2] * bt[pick] + c[:, 2:3] * n[pick]
            shaded[pick] = v / np.maximum(
                np.linalg.norm(v, axis=1, keepdims=True), 1e-12
            )
        shading[seen] = shaded
    rgb = None
    if light:
        img = np.ones((h * w, 3))
        img[pixel[first]] = colour[first, :3]
        img = img.reshape(h, w, 3)
        to_light = 0.6 * camera.toward + 0.6 * camera.up + 0.35 * camera.right
        to_light /= np.linalg.norm(to_light)
        facing = np.where(
            (normals * camera.toward).sum(-1, keepdims=True) < 0, -1.0, 1.0
        )
        nl = (normals * facing * to_light).sum(-1)
        lit = 0.62 + 0.38 * np.clip((nl + 0.1) * 4.0, 0.0, 1.0)
        img = np.where(seen[..., None], img * lit[..., None], 1.0)
        rgb = np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
    return piece, rgb, dimg, normals, shading


def _linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(c):
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _cameras(spec, sc, size, names=VIEWS):
    from opensculptboy.render.ids import views

    forward = np.array([0.0, 0.0, 1.0 if spec.version == "1.0" else -1.0])
    return views(sc["positions"], forward, size, names=names)


def picture(
    spec, view: str = "front", size: int = SIZE, focus=None, span: float | None = None
) -> np.ndarray:
    """(H, W, 3) uint8: the spec drawn with its base colours and normal maps in a toon light
    from ``view``; ``focus`` (a point of the file's frame) and ``span`` (metres across) draw a
    part of it, such as the face."""
    sc = scene(spec)
    (camera,) = _cameras(spec, sc, size, names=(view,))
    if focus is not None:
        camera.centre = np.asarray(focus, np.float64)
        camera.scale = float(span or 0.3) / size
    return _draw(spec, sc, camera, _Textures(spec))[1]


def _outline_length(mask: np.ndarray) -> float:
    edges = (mask[:, 1:] != mask[:, :-1]).sum() + (mask[1:, :] != mask[:-1, :]).sum()
    return float(edges)


def _uv_problems(sc, piece):
    t = sc["triangles"][sc["piece"] == piece]
    uv = sc["uvs"][t]
    p = sc["positions"][t]
    signed = (uv[:, 1, 0] - uv[:, 0, 0]) * (uv[:, 2, 1] - uv[:, 0, 1]) - (
        uv[:, 2, 0] - uv[:, 0, 0]
    ) * (uv[:, 1, 1] - uv[:, 0, 1])
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    return {
        "zero_area": int((np.abs(signed) < 1e-12).sum()),
        "triangles": int(len(t)),
        "orientation": signed,
        "area": area,
    }


def _texel_density(spec, sc, piece) -> float:
    """Base colour texels per metre of a piece's surface."""
    from opensculptboy.export.budget import image_size

    pick = sc["piece"] == piece
    t = sc["triangles"][pick]
    mats = sc["material"][pick]
    uv = sc["uvs"][t]
    p = sc["positions"][t]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    texels = 0.0
    for k in np.unique(mats):
        material = spec.materials[k]
        if material.base_texture is None:
            continue
        w, h = image_size(spec.images[material.base_texture][1])
        sel = mats == k
        u = uv[sel]
        signed = (u[:, 1, 0] - u[:, 0, 0]) * (u[:, 2, 1] - u[:, 0, 1]) - (
            u[:, 2, 0] - u[:, 0, 0]
        ) * (u[:, 1, 1] - u[:, 0, 1])
        scale = 1.0
        if material.texture_transform is not None:
            s = material.texture_transform.get("scale", [1, 1])
            scale = abs(s[0] * s[1])
        texels += float(np.abs(signed).sum() * 0.5 * w * h * scale)
    total = float(area.sum())
    return float(np.sqrt(texels / total)) if total > 0 and texels > 0 else 0.0


def compare(full, fitted, size: int = SIZE, names=VIEWS) -> dict:
    """The measures of the module, comparing the spec ``fitted`` with ``full``."""
    a, b = scene(full), scene(fitted)
    cams = _cameras(full, a, size, names)
    ta, tb = _Textures(full), _Textures(fitted)
    views = {}
    per_piece: dict[str, dict] = {}
    for camera in cams:
        pa, ca, da, na, sa = _draw(full, a, camera, ta)
        pb, cb, db, nb, sb = _draw(fitted, b, camera, tb)
        ma, mb = pa >= 0, pb >= 0
        xor = int((ma ^ mb).sum())
        outline = max(_outline_length(ma), 1.0)
        views[camera.name] = {
            "pixels": int(ma.sum()),
            "silhouette_share": float(xor / max(ma.sum(), 1)),
            "silhouette_mm": float(xor / outline * camera.scale * 1000.0),
            "colour_steps": float(
                np.abs(ca.astype(float) - cb.astype(float))[ma & mb].mean()
            )
            if (ma & mb).any()
            else 0.0,
        }
        for g, name in enumerate(a["names"]):
            if name not in b["names"]:
                continue
            gb = b["names"].index(name)
            same = (pa == g) & (pb == gb)
            if not same.any():
                continue
            entry = per_piece.setdefault(
                name,
                {"pixels": 0, "depth": [], "angle": [], "shading": [], "colour": []},
            )
            entry["pixels"] += int(same.sum())
            entry["depth"].append(np.abs(da[same] - db[same]))
            cos = np.clip((na[same] * nb[same]).sum(1), -1, 1)
            entry["angle"].append(np.degrees(np.arccos(np.abs(cos))))
            cos = np.clip((sa[same] * sb[same]).sum(1), -1, 1)
            entry["shading"].append(np.degrees(np.arccos(np.abs(cos))))
            entry["colour"].append(
                np.abs(ca[same].astype(float) - cb[same].astype(float)).mean(1)
            )
    pieces = {}
    for name, entry in per_piece.items():
        depth = np.concatenate(entry["depth"]) * 1000.0
        angle = np.concatenate(entry["angle"])
        shade = np.concatenate(entry["shading"])
        colour = np.concatenate(entry["colour"])
        pieces[name] = {
            "pixels": entry["pixels"],
            "surface_mm_mean": float(depth.mean()),
            "surface_mm_p95": float(np.quantile(depth, 0.95)),
            "normal_deg_mean": float(angle.mean()),
            "normal_deg_p95": float(np.quantile(angle, 0.95)),
            "shading_deg_mean": float(shade.mean()),
            "shading_deg_p95": float(np.quantile(shade, 0.95)),
            "colour_steps_mean": float(colour.mean()),
        }
    for g, name in enumerate(a["names"]):
        if name not in b["names"]:
            continue
        gb = b["names"].index(name)
        before, after = _texel_density(full, a, g), _texel_density(fitted, b, gb)
        entry = pieces.setdefault(name, {})
        if a["has_uvs"][a["piece"] == g].all():
            ua, ub = _uv_problems(a, g), _uv_problems(b, gb)
            entry["uv_zero_area"] = [ua["zero_area"], ub["zero_area"]]
        entry["texels_per_m"] = [round(before, 1), round(after, 1)]
        entry["texel_ratio"] = float(after / before) if before > 0 else 1.0
    return {
        "views": views,
        "pieces": pieces,
        "silhouette_mm_mean": float(
            np.mean([v["silhouette_mm"] for v in views.values()])
        ),
        "silhouette_share_max": float(
            max(v["silhouette_share"] for v in views.values())
        ),
    }

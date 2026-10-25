# OpenSculptBoy
# Apache License, Version 2.0
"""
MakeHuman clothes assets as wearable packages (``opensculptboy wearables import-mhclo``,
docs/wearables.md, "Free outfits").

A MakeHuman clothes asset is three kinds of file: the ``.mhclo`` file (a header and the
binding of each garment vertex to MakeHuman's base mesh), its ``.obj`` mesh and its
``.mhmat`` material with the textures. This module reads them as MakeHuman 1.x and MPFB2
write them:

- **Header.** ``name``, ``uuid``, ``description``, ``tag`` (one per line), ``z_depth``,
  ``obj_file``, ``material``, ``basemesh``, and the licence keys ``author``, ``license``,
  ``homepage`` and ``copyright``, either as keys or in comment lines (``# license CC0``);
  MPFB2 writes the description in a comment line too (``# description: ...``), which a
  ``description`` key replaces.
- **Binding.** After ``verts``, one line per garment vertex in the order of the ``.obj``
  vertices: three base-mesh vertex indices, three weights and an offset (``v1 v2 v3 w1 w2 w3
  dx dy dz``), or one index alone (the vertex sits on that base-mesh vertex). On a body ``h``
  (MakeHuman's frame: decimetres, Y up) the vertex lies at ``w1 h[v1] + w2 h[v2] + w3 h[v3] +
  S d``, where ``S`` is diagonal: for each axis n of ``x_scale``, ``y_scale`` and ``z_scale``
  (``v1 v2 length``), ``|h[v1][n] - h[v2][n]| / length``, so that the offset stretches with
  the body. Without scale lines ``S`` is the identity; ``shear_x``, ``shear_y`` and
  ``shear_z`` (``v1 v2 x1 x2``) give ``(h[v2][n] - h[v1][n]) / (x2 - x1)`` per axis.
- **Covered skin.** ``delete_verts`` lists base-mesh vertices that the garment hides, as
  numbers and ranges ``a - b``.

The binding is evaluated on the binding body (the default card), with the base-mesh
vertices that Anny's template lacks taken from MakeHuman's targets
(:mod:`opensculptboy.wearables.basemesh`), and the evaluated garment then binds to the
nearest triangles of the skin with :func:`opensculptboy.wearables.make.bind_garment`.

:func:`licence_block` records the licence, the author and the source of the header as
found; choosing assets and following their licences is the user's responsibility.

Sources: MakeHuman 1.x ``shared/proxy.py`` and ``makehuman.py`` (``LicenseInfo``), MPFB2
``entities/clothes/mhclo.py``, and the MakeHuman asset FAQ on CC-BY attribution; see
docs/wearables.md for the links.
"""

from __future__ import annotations

import dataclasses
import io
import pathlib
import re

import numpy as np

from opensculptboy.wearables.format import (
    PUBLIC,
    SLOTS,
    Garment,
    Material,
    WearableError,
)

IMPORT_COVERS = ("auto", "ray", "none")
DEFAULT_LAYER = 50  # MakeHuman's z_depth when a file gives none
# The distance of the .obj from its binding's place above which the report notes it.
SOURCE_OFFSET_MM = 5.0

# Words of an asset's name and tags that name a slot, in the order they are tried.
SLOT_WORDS = {
    "head": "hat cap helmet hood beanie crown tiara bonnet headband beret turban headscarf "
    "headwear",
    "face": "glasses spectacles sunglasses mask goggles eyepatch",
    "neck": "scarf necklace collar choker bowtie necktie",
    "hands": "glove gloves gauntlet gauntlets mitten mittens",
    "feet": "shoe shoes boot boots sandal sandals sneaker sneakers heels slipper slippers "
    "sock socks loafers flats footwear",
    "under": "underwear underpants undies knickers bra panties briefs boxer boxers "
    "lingerie thong undershirt bikini",
    "cloak": "cape cloak poncho mantle",
    "legs": "pants trousers jeans shorts skirt leggings tights stockings pantyhose bottom "
    "bottoms kilt",
    "torso": "shirt tshirt top blouse sweater jumper hoodie jacket coat vest dress suit "
    "tunic robe gown jumpsuit overall overalls fullbody uniform cardigan",
    "arms": "sleeve sleeves bracer bracers armlet",
    "accessory": "belt bag ring bracelet earring earrings watch jewelry jewellery",
}
_LOOSE_WORDS = {"clothes", "clothing", "female", "male", "makehuman", "mh", "proxy"}


class MhcloError(WearableError):
    """A MakeHuman clothes asset that cannot become a package as it stands."""

    code = "invalid_mhclo"


@dataclasses.dataclass
class MhcloAsset:
    """The contents of a ``.mhclo`` file. ``refs``, ``weights`` and ``offsets`` (V, 3) are the
    binding of each garment vertex (offsets in MakeHuman's frame, decimetres, Y up), ``scale``
    the three ``(v1, v2, length)`` scale references or None, ``shear`` the three ``(v1, v2,
    x1, x2)`` shear references or None, and ``delete_verts`` the base-mesh vertices that the
    garment hides."""

    path: pathlib.Path
    name: str | None = None
    uuid: str | None = None
    description: str | None = None
    tags: list = dataclasses.field(default_factory=list)
    author: str | None = None
    licence: str | None = None
    homepage: str | None = None
    copyright: str | None = None
    z_depth: int | None = None
    obj_file: pathlib.Path | None = None
    material: pathlib.Path | None = None
    basemesh: str | None = None
    refs: np.ndarray | None = None
    weights: np.ndarray | None = None
    offsets: np.ndarray | None = None
    single: np.ndarray | None = None  # (V,) True for a binding of one index
    scale: list | None = None
    shear: list | None = None
    delete_verts: np.ndarray | None = None
    notes: list = dataclasses.field(default_factory=list)


def _file(folder: pathlib.Path, value: str, suffix: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if not path.suffix:
        path = path.with_suffix(suffix)
    return path if path.is_absolute() else folder / path


def _number(word: str) -> bool:
    try:
        float(word)
    except ValueError:
        return False
    return True


LICENCE_KEYS = {"author", "license", "licence", "homepage", "copyright"}


def read_mhclo(path) -> MhcloAsset:
    """Read a ``.mhclo`` file (see the module documentation)."""
    path = pathlib.Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise MhcloError(
            f"{path}: {error.strerror or error}",
            "file_not_found",
            "Name an existing .mhclo file.",
        ) from error
    folder = path.parent
    asset = MhcloAsset(path=path)
    refs, weights, offsets, single = [], [], [], []
    deleted: set[int] = set()
    scale: list = [None, None, None]
    shear: list = [None, None, None]
    section = None
    ignored = set()
    number, line = 0, ""
    try:
        for number, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("//"):
                words = line.lstrip("#/").split()
                if words:
                    key = words[0].rstrip(":").lower()
                    if key in LICENCE_KEYS and len(words) > 1:
                        _set_licence(asset, key, " ".join(words[1:]))
                    elif key == "description" and asset.description is None:
                        asset.description = " ".join(words[1:]) or None  # MPFB2
                continue
            words = line.split()
            key = words[0].rstrip(":")
            if section == "verts" and _number(key):
                if len(words) == 1:
                    v = int(words[0])
                    refs.append((v, v, v))
                    weights.append((1.0, 0.0, 0.0))
                    offsets.append((0.0, 0.0, 0.0))
                    single.append(True)
                elif len(words) in (6, 9):
                    refs.append(tuple(int(w) for w in words[:3]))
                    weights.append(tuple(float(w) for w in words[3:6]))
                    d = words[6:9] if len(words) == 9 else (0, 0, 0)
                    offsets.append(tuple(float(w) for w in d))
                    single.append(False)
                else:
                    raise MhcloError(
                        f"{path}, line {number}: a vertex binding holds one index, or three "
                        f"indices, three weights and an optional offset; got {len(words)} "
                        "numbers.",
                        "malformed_binding",
                        "Export the asset again with MakeClothes or MPFB2.",
                    )
                continue
            if section == "delete" and (_number(key) or key == "-" or "-" in key):
                _delete_words(words, deleted, path, number)
                continue
            if section == "weights" and _number(key):
                continue
            section = None
            lower = key.lower()
            value = " ".join(words[1:])
            if lower in LICENCE_KEYS:
                _set_licence(asset, lower, value)
            elif lower == "name":
                asset.name = value
            elif lower == "uuid":
                asset.uuid = value
            elif lower == "description":
                asset.description = value
            elif lower == "tag":
                if value:
                    asset.tags.append(value.lower())
            elif lower == "z_depth":
                asset.z_depth = int(float(words[1]))
            elif lower == "obj_file":
                asset.obj_file = _file(folder, value, ".obj")
            elif lower == "material":
                asset.material = _file(folder, value, ".mhmat")
            elif lower == "basemesh":
                asset.basemesh = value
            elif lower in ("x_scale", "y_scale", "z_scale"):
                scale["xyz".index(lower[0])] = (
                    int(words[1]),
                    int(words[2]),
                    float(words[3]),
                )
            elif lower in ("shear_x", "shear_y", "shear_z"):
                shear["xyz".index(lower[-1])] = (
                    int(words[1]),
                    int(words[2]),
                    float(words[3]),
                    float(words[4]),
                )
            elif lower == "verts":
                section = "verts"
            elif lower == "delete_verts":
                section = "delete"
                if len(words) > 1:
                    _delete_words(words[1:], deleted, path, number)
            elif lower == "weights":
                section = "weights"
                ignored.add("weights")
            else:
                ignored.add(lower)
    except (ValueError, IndexError) as error:
        if isinstance(error, MhcloError):
            raise
        what = "a vertex binding" if section == "verts" else f"the line {line!r}"
        raise MhcloError(
            f"{path}, line {number}: {what} has a missing or malformed value "
            f"({error}).",
            "malformed_binding" if section == "verts" else "malformed_header",
            "Correct the line as MakeHuman writes it, or export the asset again with "
            "MakeClothes or MPFB2.",
        ) from error
    if not refs:
        raise MhcloError(
            f"{path}: no vertex bindings (a 'verts' section).",
            "malformed_binding",
            "Name a MakeHuman clothes file (.mhclo) with a 'verts' section.",
        )
    asset.refs = np.array(refs, np.int64)
    asset.weights = np.array(weights, np.float64)
    asset.offsets = np.array(offsets, np.float64)
    asset.single = np.array(single, bool)
    if all(s is not None for s in scale):
        asset.scale = scale
    elif any(s is not None for s in scale):
        asset.notes.append(
            "the file gives scale references for some axes only; unscaled"
        )
    if asset.scale is None and all(s is not None for s in shear):
        asset.shear = shear
    asset.delete_verts = np.array(sorted(deleted), np.int64)
    ignored -= {"version", "max_pole", "special_pose"}
    if ignored:
        asset.notes.append(f"keys left out: {sorted(ignored)}")
    return asset


def _set_licence(asset: MhcloAsset, key: str, value: str) -> None:
    value = value.strip()
    if not value:
        return
    if key in ("license", "licence"):
        asset.licence = value
    else:
        setattr(asset, key, value)


def _delete_words(words, deleted: set, path, number) -> None:
    items: list[str] = []
    for word in words:  # "a - b", "a -b", "a- b" and "a-b"
        items.extend(p for p in re.split(r"(-)", word) if p)
    previous = None
    pending = False
    for item in items:
        if item == "-":
            pending = True
            continue
        try:
            v = int(item)
        except ValueError as error:
            raise MhcloError(
                f"{path}, line {number}: delete_verts holds numbers and ranges a - b; "
                f"got {item!r}.",
                "malformed_binding",
                "Export the asset again with MakeClothes or MPFB2.",
            ) from error
        if pending and previous is not None:
            deleted.update(range(min(previous, v), max(previous, v) + 1))
        else:
            deleted.add(v)
        previous, pending = v, False


# -- the licence gate -------------------------------------------------------------------


def spdx_of(text: str | None) -> str | None:
    """The SPDX id of a licence as a MakeHuman header names it, when the text names one
    plainly (``CC0`` is CC0-1.0, ``CC-BY 4.0`` is CC-BY-4.0, ``AGPL3`` is AGPL-3.0-only),
    else None. A licence without a version keeps None, and the text stays in ``declared``."""
    if not text or not text.strip():
        return None
    t = " ".join(text.lower().replace("_", " ").split())
    t = t.replace("creative commons", "cc").replace("attribution", "by")
    if re.fullmatch(r"cc[- ]?(0|zero)([- ]?(v?1(\.0)?|universal))?", t):
        return "CC0-1.0"
    m = re.fullmatch(
        r"cc[- ]?by((?:[- ](?:sa|nc|nd))*)[- ]?v?(\d(?:\.\d)?)(?: international)?", t
    )
    if m:
        parts = [x for x in re.split(r"[- ]", m.group(1)) if x]
        version = m.group(2) if "." in m.group(2) else m.group(2) + ".0"
        return "-".join(["CC-BY", *(x.upper() for x in parts), version])
    m = re.fullmatch(r"agpl[- ]?v?(3)(\.0)?", t)
    if m:
        return "AGPL-3.0-only"
    return None


def licence_block(
    asset: MhcloAsset,
    label: str,
    *,
    author: str | None = None,
    source: str | None = None,
) -> dict:
    """
    The licence block of a package from an asset's header, as found: ``spdx`` (the SPDX id
    when the header names one plainly, else None), ``declared`` (the header's text),
    ``author``, ``source`` (``source`` or the header's homepage), ``design_source`` and an
    ``attribution`` line that the exports carry, whatever the licence, when the header names
    a licence, an author or a source. ``author`` replaces the header's author. The tools
    record the licence and write the credit; choosing assets and following their licences
    is the user's responsibility.
    """
    declared = asset.licence
    author = author or asset.author
    source = source or asset.homepage
    name = asset.name or asset.path.stem
    design = f"MakeHuman clothes asset '{name}'" + (
        f" ({asset.uuid})" if asset.uuid else ""
    )
    block = {
        "spdx": spdx_of(declared),
        "declared": declared,
        "author": author,
        "source": source,
        "design_source": design,
    }
    if declared or author or source:
        credit = f"'{label}'" + (f" by {author}" if author else "")
        credit += f", {declared}" if declared else ""
        block["attribution"] = credit + (f", from {source}" if source else "")
    return {k: v for k, v in block.items() if v or k == "spdx"}


# -- the mesh and the material ----------------------------------------------------------


@dataclasses.dataclass
class ObjMesh:
    """An ``.obj`` mesh: vertices (MakeHuman's frame), UVs (OBJ convention, v = 0 at the
    bottom), and triangles as rows of (vertex, uv) index pairs (T, 3, 2); uv -1 for none."""

    vertices: np.ndarray
    uvs: np.ndarray
    corners: np.ndarray
    notes: list


def read_obj(path) -> ObjMesh:
    """Read the vertices, UVs and faces of an ``.obj`` file; polygons become triangle fans."""
    path = pathlib.Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise MhcloError(
            f"{path}: {error.strerror or error}",
            "file_not_found",
            "Keep the .obj file of the asset next to its .mhclo file.",
        ) from error
    vertices, uvs, corners = [], [], []
    notes: list[str] = []
    number = 0
    try:
        for number, line in enumerate(text.splitlines(), 1):
            words = line.split()
            if not words:
                continue
            if words[0] == "v":
                vertices.append([float(w) for w in words[1:4]])
                if len(vertices[-1]) != 3:
                    raise ValueError("a vertex needs three coordinates")
            elif words[0] == "vt":
                uvs.append([float(w) for w in words[1:3]])
                if len(uvs[-1]) != 2:
                    raise ValueError("a UV needs two coordinates")
            elif words[0] == "f":
                face = []
                for token in words[1:]:
                    parts = token.split("/")
                    v = int(parts[0])
                    v = v - 1 if v > 0 else len(vertices) + v
                    t = -1
                    if len(parts) > 1 and parts[1]:
                        t = int(parts[1])
                        t = t - 1 if t > 0 else len(uvs) + t
                    face.append((v, t))
                for k in range(1, len(face) - 1):
                    corners.append([face[0], face[k], face[k + 1]])
    except (ValueError, IndexError) as error:
        raise MhcloError(
            f"{path}, line {number}: a malformed value ({error}).",
            "malformed_obj",
            "Export the mesh again as OBJ with vertices, UVs and faces.",
        ) from error
    if not corners:
        raise MhcloError(
            f"{path}: no faces.", "no_meshes", "Name the asset's mesh in obj_file."
        )
    return ObjMesh(
        np.array(vertices, np.float64).reshape(-1, 3),
        np.array(uvs, np.float64).reshape(-1, 2),
        np.array(corners, np.int64),
        notes,
    )


def _srgb_to_linear(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _image(path: pathlib.Path, quality, material: str, slot: str) -> bytes | None:
    """PNG or JPEG bytes of a texture file (other formats become PNG), or None."""
    label = dict(material=material, slot=slot)
    try:
        data = path.read_bytes()
    except OSError:
        quality.add(
            "texture_missing",
            "warning",
            f"{material}: the {slot} texture {path.name} is missing; left out",
            "Keep the textures next to the .mhmat file (or in its textures folder).",
            **label,
        )
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff":
        return data
    try:
        from PIL import Image

        out = io.BytesIO()
        Image.open(io.BytesIO(data)).save(out, format="PNG")
    except Exception:  # noqa: BLE001 - any unreadable image is left out with an item
        quality.add(
            "texture_unreadable",
            "warning",
            f"{material}: the {slot} texture {path.name} is not a readable image; left out",
            "Save the texture as PNG or JPEG.",
            **label,
        )
        return None
    quality.fix(
        "texture_converted",
        f"{material}: the {slot} texture {path.name} converted to PNG",
        **label,
    )
    return out.getvalue()


def _find_texture(folder: pathlib.Path, value: str) -> pathlib.Path:
    path = pathlib.Path(value.replace("\\", "/"))
    if path.is_absolute():
        return path
    for candidate in (
        folder / path,
        folder / path.name,
        folder / "textures" / path.name,
    ):
        if candidate.is_file():
            return candidate
    return folder / path


ROUGHNESS_FORMULA = (
    "roughnessFactor = clamp(1 - shininess, 0.04, 1), and 1 when specularColor is 0; a "
    "'roughness' key of the .mhmat file wins"
)


def read_mhmat(path, fallback_name: str, quality) -> Material:
    """
    A glTF PBR material from a ``.mhmat`` file, with the mapping in ``quality``:
    ``diffuseColor`` (sRGB, made linear) and ``diffuseTexture`` give the base colour,
    ``opacity`` its alpha; ``shininess`` and ``specularColor`` give the roughness
    (:data:`ROUGHNESS_FORMULA`), ``metallic`` the metallic factor;
    ``transparencymapTexture`` becomes the alpha of the base colour texture; ``transparent``
    gives BLEND, or MASK (cutoff 0.5) with ``alphaToCoverage`` or for an alpha map on an
    opaque material; ``backfaceCull false`` gives a double-sided material;
    ``normalmapTexture`` with ``normalmapIntensity`` the normal map, ``aomapTexture`` the
    occlusion and ``emissiveColor`` the emission. Specular, bump and displacement maps have
    no glTF PBR slot and are dropped with an item each. MakeHuman's material reader
    (``shared/material.py``) ignores ``diffuseIntensity`` and ``specularIntensity`` as
    deprecated keys and draws the colours as they are; the import does the same and reports
    the keys with an item.
    """
    path = pathlib.Path(path) if path else None
    if path is None or not path.is_file():
        quality.add(
            "material_missing",
            "warning",
            "the asset has no material file"
            + (f" ({path.name} is missing)" if path is not None else "")
            + "; a plain grey material",
            "Keep the .mhmat file next to the .mhclo file.",
        )
        return Material(fallback_name)
    folder = path.parent
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        words = line.split()
        if not words or words[0].startswith("#") or words[0].startswith("//"):
            continue
        values[words[0]] = " ".join(words[1:])

    def flag(key, default=False):
        value = values.get(key)
        return (
            default if value is None else value.strip().lower() in ("true", "1", "yes")
        )

    def number(key, default):
        try:
            return float(values[key].split()[0])
        except (KeyError, ValueError, IndexError):
            return default

    def colour(key, default):
        try:
            rgb = [float(w) for w in values[key].split()[:3]]
        except (KeyError, ValueError):
            return default
        return rgb if len(rgb) == 3 else default

    name = values.get("name") or fallback_name
    material = Material(name)
    rgb = _srgb_to_linear(colour("diffuseColor", [1.0, 1.0, 1.0]))
    alpha = float(np.clip(number("opacity", 1.0), 0.0, 1.0))
    material.base_color = (float(rgb[0]), float(rgb[1]), float(rgb[2]), alpha)
    specular = np.asarray(colour("specularColor", [1.0, 1.0, 1.0]))
    deprecated = [k for k in ("diffuseIntensity", "specularIntensity") if k in values]
    if deprecated:
        quality.add(
            "deprecated_key",
            "info",
            f"{name}: {' and '.join(deprecated)} left out: MakeHuman ignores "
            f"{'these deprecated keys' if len(deprecated) > 1 else 'this deprecated key'} "
            "and draws the colours as they are",
            "Set the colours with diffuseColor and specularColor.",
            material=name,
            count=len(deprecated),
        )
    if "roughness" in values:
        material.roughness = float(np.clip(number("roughness", 0.8), 0.0, 1.0))
    elif float(np.max(specular)) <= 0.0:
        material.roughness = 1.0
    else:
        material.roughness = float(np.clip(1.0 - number("shininess", 0.2), 0.04, 1.0))
    material.metallic = float(np.clip(number("metallic", 0.0), 0.0, 1.0))
    quality.add(
        "roughness_mapped",
        "info",
        f"{name}: roughnessFactor {material.roughness:.3f} ({ROUGHNESS_FORMULA})",
        material=name,
        value=round(material.roughness, 4),
    )
    material.emissive = tuple(
        float(c) for c in _srgb_to_linear(colour("emissiveColor", [0, 0, 0]))
    )
    material.double_sided = not flag("backfaceCull", True)
    if material.double_sided:
        quality.add(
            "double_sided",
            "info",
            f"{name}: backfaceCull false; the material is double-sided",
            material=name,
        )
    transparent = flag("transparent") or alpha < 1.0
    if transparent:
        material.alpha_mode = "MASK" if flag("alphaToCoverage") else "BLEND"
    textures = {}
    for key, slot in (
        ("diffuseTexture", "base_color"),
        ("normalmapTexture", "normal"),
        ("aomapTexture", "occlusion"),
    ):
        if values.get(key):
            data = _image(_find_texture(folder, values[key]), quality, name, slot)
            if data is not None:
                textures[slot] = data
    if values.get("transparencymapTexture"):
        mask = _image(
            _find_texture(folder, values["transparencymapTexture"]),
            quality,
            name,
            "alpha",
        )
        if mask is not None:
            textures["base_color"] = _with_alpha(textures.get("base_color"), mask)
            quality.fix(
                "alpha_map_merged",
                f"{name}: the transparency map became the alpha of the base colour texture",
                material=name,
            )
            if material.alpha_mode == "OPAQUE":
                material.alpha_mode = "MASK"
                quality.add(
                    "alpha_map_on_opaque",
                    "warning",
                    f"{name}: the material is opaque and has a transparency map; written "
                    f"as MASK with cutoff {material.alpha_cutoff}",
                    "Set 'transparent True' in the .mhmat file for sheer cloth (BLEND).",
                    material=name,
                )
    if material.alpha_mode != "OPAQUE":
        quality.add(
            "alpha_mode",
            "info",
            f"{name}: alphaMode {material.alpha_mode}"
            + (
                f", cutoff {material.alpha_cutoff}"
                if material.alpha_mode == "MASK"
                else ""
            ),
            material=name,
        )
    if "normal" in textures:
        material.normal_scale = number("normalmapIntensity", 1.0)
    for key, what in (
        ("specularmapTexture", "specular map"),
        ("bumpmapTexture", "bump map"),
        ("displacementmapTexture", "displacement map"),
    ):
        if values.get(key):
            quality.add(
                "map_dropped",
                "info",
                f"{name}: the {what} ({pathlib.Path(values[key]).name}) has no glTF PBR "
                "slot; dropped",
                "Bake it into the normal map or a metallic-roughness map.",
                material=name,
                slot=key,
            )
    material.textures = textures
    return material


def _with_alpha(base: bytes | None, mask: bytes) -> bytes:
    """A PNG of the base colour texture (white without one) with the mask's grey as alpha."""
    from PIL import Image

    alpha = Image.open(io.BytesIO(mask)).convert("L")
    if base is None:
        rgb = Image.new("RGB", alpha.size, (255, 255, 255))
    else:
        rgb = Image.open(io.BytesIO(base)).convert("RGB")
        if alpha.size != rgb.size:
            alpha = alpha.resize(rgb.size)
    rgb.putalpha(alpha)
    out = io.BytesIO()
    rgb.save(out, format="PNG")
    return out.getvalue()


# -- slot and binding -------------------------------------------------------------------


def _singular(word: str) -> str | None:
    """The singular of a plain English plural (hats, scarves), or None."""
    if word.endswith("ves") and len(word) > 4:
        return word[:-3] + "f"
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return None


def infer_slot(name: str, tags) -> tuple[str | None, str | None]:
    """(slot, reason) from the words of an asset's name and tags, or (None, None). A word of
    :data:`SLOT_WORDS` (or its plural) names its slot; else the longest slot word inside a
    compound word (``underpants``, ``hoodedjacket``), where a slot word of three letters
    (``hat``, ``bra``) counts only at the end (``tophat``, ``sportsbra``)."""
    words = []
    for text in [name, *tags]:
        for word in re.split(r"[\s_\-.,;/]+", str(text).lower()):
            word = re.sub(r"\d+$", "", word)
            if word and word not in _LOOSE_WORDS:
                words.append(word)
    table = {w: slot for slot, text in SLOT_WORDS.items() for w in text.split()}
    for word in words:
        for form in (word, _singular(word)):
            if form in table:
                return table[form], f"the word {word!r}"
    for word in words:  # compound names such as casualsuit or tshirtlong
        found = [
            key
            for key in table
            if (len(key) >= 4 and key in word)
            or (len(key) == 3 and word.endswith(key) and len(word) > 3)
        ]
        if found:
            key = max(found, key=len)  # the first of the longest, in the table's order
            return table[key], f"{key!r} in the word {word!r}"
    return None, None


def evaluate_binding(asset: MhcloAsset, h: np.ndarray) -> np.ndarray:
    """(V, 3) the garment vertices on base-mesh vertices ``h`` (19158, 3) in MakeHuman's
    frame (decimetres, Y up)."""
    S = np.eye(3)
    if asset.scale is not None:
        for n, (v1, v2, length) in enumerate(asset.scale):
            S[n, n] = abs(h[v1, n] - h[v2, n]) / length
    elif asset.shear is not None:
        for n, (v1, v2, x1, x2) in enumerate(asset.shear):
            S[n, n] = (h[v2, n] - h[v1, n]) / (x2 - x1)
    points = np.einsum("vk,vkd->vd", asset.weights, h[asset.refs])
    return points + asset.offsets @ S.T


def referenced(asset: MhcloAsset) -> np.ndarray:
    """The base-mesh vertices that the binding and its scale references use."""
    rows = [asset.refs.reshape(-1)]
    for entry in (asset.scale or []) + (asset.shear or []):
        rows.append(np.array(entry[:2], np.int64))
    return np.unique(np.concatenate(rows))


def garment_from(
    asset: MhcloAsset, mesh: ObjMesh, vertices: np.ndarray, material: Material
) -> tuple[Garment, np.ndarray]:
    """The garment with one vertex per (obj vertex, UV) pair, the normals of the welded
    mesh (so that UV seams stay smooth) and one group named by the material; returns the
    obj vertex of each garment vertex too."""
    from opensculptboy.wearables.geometry import vertex_normals

    welded = mesh.corners[..., 0]
    normals = vertex_normals(vertices, welded)
    pairs = mesh.corners.reshape(-1, 2)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    triangles = np.asarray(inverse).reshape(-1, 3)
    source = unique[:, 0]
    uvs = None
    if (unique[:, 1] >= 0).any():
        uv = np.zeros((len(unique), 2))
        ok = unique[:, 1] >= 0
        uv[ok] = mesh.uvs[unique[ok, 1]]
        uvs = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]])  # glTF: v = 0 at the top
    garment = Garment(
        vertices[source],
        triangles,
        normals=normals[source],
        uvs=uvs,
        materials=[material],
        groups=[material.name],
    )
    return garment, source


def import_mhclo(
    path,
    body,
    character,
    model,
    *,
    slot: str | None = None,
    layer: int | None = None,
    name: str | None = None,
    label: str | None = None,
    tags=(),
    description: str | None = None,
    author: str | None = None,
    source: str | None = None,
    covers: str = "auto",
    hides=(),
    max_distance: float | None = None,
    texture_limit: int | None = None,
    normal_green: str = "auto",
    accept_quality: bool = False,
) -> tuple[object, dict]:
    """
    A wearable package from a MakeHuman clothes asset, bound to ``body`` (the
    :class:`~opensculptboy.wearables.bind.BodySurface` of ``character`` and ``model``).

    ``covers``: ``auto`` takes the skin under ``delete_verts`` when the file lists any, else
    the ray test of ``wearables bind``; ``ray`` always the ray test; ``none`` no covered skin.
    The garment goes through :func:`opensculptboy.wearables.quality.check_garment`
    (``texture_limit``, ``normal_green``); an error-severity item raises
    :class:`~opensculptboy.wearables.quality.QualityError` unless ``accept_quality``.
    Returns the checked package and a report.
    """
    from opensculptboy.wearables.basemesh import (
        MH_TO_ANNY,
        base_groups,
        base_mesh_vertices,
        rows_of_base,
    )
    from opensculptboy.wearables.make import MAX_DISTANCE, bind_garment, clean_tags
    from opensculptboy.wearables.quality import (
        TEXTURE_LIMIT,
        Quality,
        check_garment,
        require,
        strip_textures_without_uvs,
    )

    if covers not in IMPORT_COVERS:
        raise MhcloError(
            f"covers must be one of {IMPORT_COVERS}.",
            "bad_option",
            f"Use --covers {' or '.join(IMPORT_COVERS)}.",
        )
    asset = read_mhclo(path)
    warnings = list(asset.notes)
    quality = Quality()
    stem = pathlib.Path(path).stem
    name = (
        name or re.sub(r"[^A-Za-z0-9_.-]+", "_", asset.name or stem).strip("_") or stem
    )
    label = label or (asset.name or name).replace("_", " ")
    licence = licence_block(asset, label, author=author, source=source)
    provenance = PUBLIC
    if asset.basemesh and asset.basemesh.lower() not in ("hm08",):
        raise MhcloError(
            f"{path}: the asset binds to the base mesh {asset.basemesh!r}; Anny's template "
            "is MakeHuman's hm08.",
            "unknown_basemesh",
            "Choose an asset made for MakeHuman's default base mesh (hm08).",
        )
    count = 19158
    bad = (asset.refs < 0) | (asset.refs >= count)
    references = [e[:2] for e in (asset.scale or []) + (asset.shear or [])]
    if bad.any() or any(not 0 <= v < count for e in references for v in e):
        raise MhcloError(
            f"{path}: the binding names base-mesh vertices outside 0..{count - 1}.",
            "unknown_basemesh",
            "Choose an asset made for MakeHuman's default base mesh (hm08).",
        )
    spans = [e[2] for e in asset.scale or []] + [e[3] - e[2] for e in asset.shear or []]
    if any(not np.isfinite(s) or s == 0 for s in spans):
        raise MhcloError(
            f"{path}: a scale or shear reference has a length of 0.",
            "malformed_header",
            "Export the asset again with MakeClothes or MPFB2.",
        )
    if slot is None:
        slot, reason = infer_slot(asset.name or stem, asset.tags)
        if slot is None:
            raise MhcloError(
                f"{path}: no slot from the name {asset.name or stem!r} and the tags "
                f"{asset.tags}.",
                "unknown_slot",
                f"Pass --slot with one of {', '.join(SLOTS)}.",
            )
    else:
        reason = "--slot"
        if slot not in SLOTS:
            raise MhcloError(
                f"Unknown slot {slot!r}.",
                "unknown_slot",
                f"Use one of {', '.join(SLOTS)}.",
            )
    if layer is None:
        if asset.z_depth is None:
            warnings.append(f"no z_depth; layer {DEFAULT_LAYER}, as MakeHuman uses")
            layer = DEFAULT_LAYER
        else:
            layer = int(np.clip(asset.z_depth, 0, 100))
    if asset.obj_file is None:
        asset.obj_file = pathlib.Path(path).with_suffix(".obj")
    mesh = read_obj(asset.obj_file)
    if len(mesh.vertices) != len(asset.refs):
        raise MhcloError(
            f"{path}: the binding holds {len(asset.refs)} vertices and "
            f"{asset.obj_file.name} {len(mesh.vertices)}.",
            "binding_mismatch",
            "Keep the .obj file that the .mhclo file was made with.",
        )
    v, t = mesh.corners[..., 0], mesh.corners[..., 1]
    if (
        (v < 0).any()
        or (v >= len(mesh.vertices)).any()
        or (t < -1).any()
        or (t >= len(mesh.uvs)).any()
    ):
        raise MhcloError(
            f"{asset.obj_file}: a face names a vertex or a UV that does not exist.",
            "binding_mismatch",
            "Export the mesh again.",
        )
    material = read_mhmat(asset.material, name, quality)
    # the binding on the binding body, in MakeHuman's frame
    needed = referenced(asset)
    to_mh = np.linalg.inv(MH_TO_ANNY)
    h_anny, from_targets = base_mesh_vertices(character, model, body, needed)
    h = h_anny @ to_mh.T
    vertices = evaluate_binding(asset, h) @ MH_TO_ANNY.T
    # the binding on the base mesh itself, against the .obj vertices (a check of the file)
    rows = rows_of_base(model, len(body.vertices))
    template = np.full((count, 3), np.nan)
    present = rows >= 0
    template[present] = (
        model.template_vertices.detach().cpu().double().numpy()[rows[present]]
    )
    missing = np.intersect1d(np.flatnonzero(~present), needed)
    if len(missing):
        template[missing] = _template_rows(missing)
    on_base = evaluate_binding(asset, template @ to_mh.T)
    # The .obj may lie away from the place its binding gives: MakeClothes 2 exports the
    # human standing on the ground, about 8 dm above base.obj's feet. The offset is the mean
    # translation, and the fits compare the shapes with it taken out, on base.obj and on
    # the binding body (the human that MakeClothes 2 models on is the default one).
    shift = (mesh.vertices - on_base).mean(axis=0)
    source_fit = np.linalg.norm(on_base + shift - mesh.vertices, axis=1) * 100.0  # mm
    on_body = vertices @ to_mh.T
    body_shift = (mesh.vertices - on_body).mean(axis=0)
    body_fit = np.linalg.norm(on_body + body_shift - mesh.vertices, axis=1) * 100.0
    source_offset = shift @ MH_TO_ANNY.T * 1000.0  # Anny's frame, mm
    if float(np.linalg.norm(source_offset)) > SOURCE_OFFSET_MM:
        quality.add(
            "source_offset",
            "info",
            f"the .obj lies {np.linalg.norm(source_offset):.0f} mm from the place its "
            f"binding gives on base.obj (x, y, z: {np.round(source_offset).tolist()} mm, "
            "as MakeClothes 2 exports a human on the ground); the binding places the garment",
            value=round(float(np.linalg.norm(source_offset)), 1),
        )
    garment, _ = garment_from(asset, mesh, vertices, material)
    garment, _, quality = check_garment(
        garment,
        body=body,
        normals_from_file=False,
        texture_limit=TEXTURE_LIMIT if texture_limit is None else texture_limit,
        normal_green=normal_green,
        quality=quality,
    )
    require(quality, accept_quality)
    strip_textures_without_uvs(garment, quality)
    # covered skin
    fixed = None
    deleted_outside = 0
    if covers == "auto" and len(asset.delete_verts):
        valid = asset.delete_verts[
            (asset.delete_verts >= 0) & (asset.delete_verts < count)
        ]
        anny_rows = rows[valid]
        deleted_outside = int((anny_rows < 0).sum())
        fixed = np.zeros(len(body.vertices), bool)
        fixed[anny_rows[anny_rows >= 0]] = True
    groups = base_groups()
    used = np.unique(asset.refs[asset.weights != 0])
    helpers = {
        group: int(np.isin(asset.refs, members).any(axis=1).sum())
        for group, members in sorted(groups.items())
        if group.startswith("helper-") and np.isin(used, members).any()
    }
    target_vertices = int(from_targets[asset.refs].any(axis=1).sum())
    all_tags = clean_tags([*asset.tags, *tags])
    text = description if description is not None else (asset.description or "")
    if not text:
        text = f"{label}, a MakeHuman clothes asset" + (
            f" by {licence['author']}" if licence.get("author") else ""
        )
    wearable, report = bind_garment(
        garment,
        body,
        name,
        slot,
        layer,
        label=label,
        covers="none" if covers == "none" else "auto",
        fixed_covers=fixed,
        hides=hides,
        licence=licence,
        provenance=provenance,
        body_kind="anny-template",
        max_distance=MAX_DISTANCE if max_distance is None else max_distance,
        extras={
            "source": "opensculptboy wearables import-mhclo",
            "mhclo": {
                "file": pathlib.Path(path).name,
                "name": asset.name,
                "uuid": asset.uuid,
                "declared_licence": asset.licence,
                "z_depth": asset.z_depth,
            },
        },
        tags=all_tags,
        description=text,
        warnings=warnings,
        source=str(path),
    )
    report.update(
        {
            "licence": dict(wearable.licence),
            "provenance": provenance,
            "slot_from": reason,
            "mhclo": {
                "name": asset.name,
                "uuid": asset.uuid,
                "declared_licence": asset.licence,
                "z_depth": asset.z_depth,
                "obj_vertices": int(len(mesh.vertices)),
                "single_bindings": int(asset.single.sum()),
                "scale": "scale"
                if asset.scale
                else ("shear" if asset.shear else "none"),
                "delete_verts": int(len(asset.delete_verts)),
                "delete_verts_outside_anny": deleted_outside,
                "helper_bound_vertices": helpers,
                "target_evaluated_vertices": target_vertices,
                "source_offset_mm": [round(float(c), 1) for c in source_offset],
                "source_fit_mm_median": round(float(np.median(source_fit)), 3),
                "source_fit_mm_max": round(float(source_fit.max()), 3),
                "source_fit_body_mm_median": round(float(np.median(body_fit)), 3),
                "source_fit_body_mm_max": round(float(body_fit.max()), 3),
            },
            "covers_from": "none"
            if covers == "none"
            else ("delete_verts" if fixed is not None else "ray"),
            "quality": quality.to_dict(),
        }
    )
    return wearable, report


def _template_rows(rows) -> np.ndarray:
    """The base-mesh vertices ``rows`` of ``base.obj`` in Anny's frame."""
    from opensculptboy.wearables.basemesh import _full_targets

    return _full_targets(tuple(int(r) for r in rows))[0]

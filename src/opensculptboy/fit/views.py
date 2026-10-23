# OpenSculptBoy
# Apache License, Version 2.0
"""
Grey-shaded and heat-map pictures of a fitted mesh and its reference, for judging the geometry
of a fit without its textures.

:func:`render` casts one ray per pixel (Embree, as in the texture bake) from a perspective
camera (:class:`Camera`, :func:`head_cameras`) and shades the hit with the normals interpolated
at it: the mesh's own vertex normals by default, or given ones (the normals that the reference's
file shades with). With per-vertex ``values`` it colours the surface by them (a heat map),
shaded lightly so that its shape stays readable. :func:`sheet` lays pictures out in a grid
with captions. It needs ``embreex`` (the ``viewer`` extra) and Pillow.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class Camera:
    """A perspective camera in Anny's frame (metres, Z up)."""

    eye: np.ndarray
    target: np.ndarray
    fov: float = 20.0  # vertical field of view, degrees
    up: tuple = (0.0, 0.0, 1.0)

    def rays(self, width: int, height: int):
        eye, target = np.asarray(self.eye, float), np.asarray(self.target, float)
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, self.up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        half = np.tan(np.radians(self.fov) / 2)
        ys, xs = np.mgrid[0:height, 0:width]
        u = (2 * (xs + 0.5) / width - 1) * half * width / height
        v = (1 - 2 * (ys + 0.5) / height) * half
        d = forward + u[..., None] * right + v[..., None] * up
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        return np.broadcast_to(eye, d.shape).reshape(-1, 3), d.reshape(-1, 3)


def head_cameras(landmarks: dict) -> dict[str, Camera]:
    """
        Cameras on the head of a figure in Anny's frame (facing -Y), from its landmarks (the
        names of :data:`opensculptboy.fit.reference.LANDMARKS`; the eyes, the mouth corners and
        the chin): ``front``, ``three_quarter`` (35 degrees toward the figure's left),
        ``side`` (its left), ``under_chin`` (from below the chin, in front) and ``jaw``
        (a close-up of the left jaw line and its angle), ``ear`` and ``ear_back`` (close-ups of the
    left ear from its front and from behind).
    """
    eyes = [landmarks[k] for k in ("eye_outer_l", "eye_outer_r") if k in landmarks]
    mouth = [
        landmarks[k] for k in ("mouth_corner_l", "mouth_corner_r") if k in landmarks
    ]
    centre = np.mean(eyes + mouth, axis=0)
    chin = np.asarray(landmarks.get("chin", centre - [0, 0, 0.06]))

    def around(angle, elevation, distance, target, fov):
        a, e = np.radians(angle), np.radians(elevation)
        direction = np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
        return Camera(target + distance * direction, target, fov)

    jaw_target = chin + [0.035, 0.03, 0.01]
    ear = np.asarray(landmarks.get("ear_l", centre + [0.07, 0.07, 0.0]), float)
    if "ear_top_l" in landmarks and "ear_lobe_l" in landmarks:
        ear = (
            np.asarray(landmarks["ear_top_l"]) + np.asarray(landmarks["ear_lobe_l"])
        ) / 2 + [0.01, 0.0, 0.0]
    return {
        "front": around(0, 0, 0.9, centre, 17),
        "three_quarter": around(35, 0, 0.9, centre, 17),
        "side": around(90, 0, 0.9, centre + [0, 0.02, 0], 17),
        "under_chin": around(15, -50, 0.8, chin + [0, 0.02, -0.01], 14),
        "jaw": around(50, -10, 0.6, jaw_target, 9),
        "ear": around(70, 5, 0.5, ear, 10),
        "ear_back": around(150, 10, 0.5, ear, 10),
    }


def render(
    vertices: np.ndarray,
    triangles: np.ndarray,
    camera: Camera,
    size: tuple = (640, 640),
    normals: np.ndarray | None = None,
    values: np.ndarray | None = None,
    colormap: str = "RdBu_r",
    limits: tuple = (-0.5, 0.5),
    background=(46, 46, 52),
    base=(0.78, 0.78, 0.78),
) -> np.ndarray:
    """An RGB picture (H, W, 3) of a mesh: grey, or coloured by per-vertex ``values`` within
    ``limits`` (values NaN draw grey)."""
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    from opensculptboy.fit.reference import vertex_normals

    V = np.asarray(vertices, dtype=np.float64)
    T = np.asarray(triangles)
    if normals is None:
        normals = vertex_normals(V, T)
    width, height = size
    origins, directions = camera.rays(width, height)
    mesh = trimesh.Trimesh(V, T, process=False)
    tri, ray, hits = RayMeshIntersector(mesh).intersects_id(
        origins, directions, multiple_hits=False, return_locations=True
    )
    image = np.empty((width * height, 3))
    image[:] = np.asarray(background) / 255.0
    if len(tri):
        bary = trimesh.triangles.points_to_barycentric(V[T[tri]], hits)
        n = np.einsum("nk, nkd -> nd", bary, np.asarray(normals)[T[tri]])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        view = -directions[ray]
        flip = (n * view).sum(1) < 0
        n[flip] *= -1
        eye = np.asarray(camera.eye, float)
        forward = np.asarray(camera.target, float) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, camera.up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        key = -forward + 0.6 * right + 0.8 * up
        key /= np.linalg.norm(key)
        fill = -forward - 0.7 * right + 0.1 * up
        fill /= np.linalg.norm(fill)
        light = (
            0.15
            + 0.65 * np.clip((n * key).sum(1), 0, 1)
            + 0.25 * np.clip((n * fill).sum(1), 0, 1)
        )
        half = key + view
        half /= np.linalg.norm(half, axis=1, keepdims=True)
        specular = 0.12 * np.clip((n * half).sum(1), 0, 1) ** 40
        colour = np.tile(np.asarray(base, float), (len(tri), 1))
        if values is not None:
            from matplotlib import colormaps

            value = np.einsum("nk, nk -> n", bary, np.asarray(values, float)[T[tri]])
            known = np.isfinite(value)
            lo, hi = limits
            scaled = np.clip((value[known] - lo) / (hi - lo), 0, 1)
            colour[known] = colormaps[colormap](scaled)[:, :3]
            light = 0.45 + 0.55 * light
            specular = 0.0 * specular
        image[ray] = np.clip(colour * light[:, None] + specular[:, None], 0, 1)
    return (image.reshape(height, width, 3) * 255).astype(np.uint8)


def sheet(rows: list, captions: list | None = None, label_height: int = 18):
    """A PIL image of pictures in a grid: ``rows`` of (H, W, 3) arrays, and a caption for each
    picture in the same layout."""
    from PIL import Image, ImageDraw

    h, w = rows[0][0].shape[:2]
    columns = max(len(r) for r in rows)
    out = Image.new("RGB", (columns * w, len(rows) * (h + label_height)), (20, 20, 24))
    draw = ImageDraw.Draw(out)
    for i, row in enumerate(rows):
        for j, picture in enumerate(row):
            y = i * (h + label_height)
            out.paste(Image.fromarray(picture), (j * w, y + label_height))
            if captions:
                draw.text((j * w + 4, y + 3), captions[i][j], fill=(230, 230, 230))
    return out


def colour_bar(
    colormap: str, limits: tuple, unit: str, width: int = 640, height: int = 40
):
    """A horizontal colour bar (H, W, 3) with its limits written at its ends."""
    from matplotlib import colormaps
    from PIL import Image, ImageDraw

    ramp = colormaps[colormap](np.linspace(0, 1, width))[:, :3]
    bar = np.tile(ramp[None], (height // 2, 1, 1))
    image = Image.new("RGB", (width, height), (20, 20, 24))
    image.paste(Image.fromarray((bar * 255).astype(np.uint8)), (0, 0))
    draw = ImageDraw.Draw(image)
    lo, hi = limits
    draw.text((4, height // 2 + 3), f"{lo:g} {unit}", fill=(230, 230, 230))
    draw.text(
        (width // 2 - 10, height // 2 + 3), f"{(lo + hi) / 2:g}", fill=(230, 230, 230)
    )
    draw.text((width - 70, height // 2 + 3), f"{hi:g} {unit}", fill=(230, 230, 230))
    return np.asarray(image)

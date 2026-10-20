# OpenSculptBoy
# Apache License, Version 2.0
"""
The geometry of the exported eyes, in the frame of each eye.

Each eye has a frame (:class:`EyeFrame`): the centre and the radius of the sphere through
anny's eyeball vertices, and axes whose third column is the gaze (-Y of Anny's rest frame,
where the figure looks), whose second is up (+Z) and whose first is +X (the figure's left).
The parts are surfaces of revolution about the gaze (:func:`revolve`), in units of the eyeball
radius, so that one set of proportions fits every eyeball:

- the eyeball (:func:`eyeball_profile`): the sclera sphere and, inside the limbus, a short
  wall down to the iris, a shallow dome a little behind the plane of the limbus. A refracting
  cornea in front of it makes the iris look as deep as it is (Cycles, three.js);
- the cornea (:func:`cornea_profile`): a shell just over the sclera with the corneal bulge, a
  sphere of radius :data:`CORNEA_RADIUS` that meets the shell at the limbus;
- the single-surface eyeball (:func:`single_profile`): the sclera with the corneal bulge and
  the iris drawn on it, for MToon and for renderers without refraction;
- the caruncle (:func:`caruncle`): a low mound over the pocket in the inner corner of the eye,
  where the front view between the lids shows the skin of the socket, which reads as a dark
  hole without it; :func:`socket` finds that skin, which the exports draw pink and wet.

The sizes of the iris, the pupil and the cornea are millimetres (:class:`EyeShape` turns them
into eyeball radii): anny's eyeball is 29.9 mm across, larger than a human's 24 mm, while its
lids open about as wide as a human's, so an iris scaled with the eyeball would fill the whole
opening. The cornea's bulge is lowered where the lids lie closer to the eyeball than its full
height (:func:`bulge_clearance`): anny's lids rest on the sphere of its eyeball.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# a human eyeball's diameter, for reference
REFERENCE_EYEBALL_MM = 24.0
# the radius of curvature of the cornea's front
CORNEA_RADIUS_MM = 7.8
# the cornea shell over the sclera
CORNEA_GAP_MM = 0.15
# the cornea shell reaches this far from the gaze, under the lids (degrees)
CORNEA_EXTENT = 100.0
# the iris: its rim lies this far behind the plane of the limbus, and its centre rises by the
# dome toward the cornea
IRIS_RECESS_MM = 0.45
IRIS_DOME_MM = 0.3
# the depth of the anterior chamber, from the cornea's apex to the iris: the thickness of the
# glTF cornea's volume
CHAMBER_DEPTH_MM = 3.0
# the limbus may reach at most this far from the gaze axis (eyeball radii)
MAX_IRIS = 0.8
# segments around the gaze, by kind of file
SEGMENTS = {"gltf": 48, "vrm": 32}
# the step of the sclera's rings in front, and the angle beyond which it turns coarse (degrees)
SCLERA_STEP = 6.0
SCLERA_FINE = 96.0
# how close the outward-facing skin of the lids may come to the eyeball's outer surface
LID_CLEARANCE_MM = 0.08


@dataclasses.dataclass
class EyeShape:
    """The proportions of an eye in eyeball radii: the limbus radius (``iris``), the cornea's
    radius of curvature, the cornea shell's gap, the iris's recess and dome, and the lids'
    clearance."""

    iris: float
    cornea: float
    gap: float
    recess: float
    dome: float
    clearance: float

    @classmethod
    def of(cls, iris_mm: float, radius_m: float) -> "EyeShape":
        """The shape of an eye with an iris of ``iris_mm`` on an eyeball of ``radius_m``."""
        r = radius_m * 1000.0
        iris = min(iris_mm / 2.0 / r, MAX_IRIS)
        return cls(
            iris=iris,
            cornea=max(CORNEA_RADIUS_MM / r, iris * 1.05),
            gap=CORNEA_GAP_MM / r,
            recess=IRIS_RECESS_MM / r,
            dome=IRIS_DOME_MM / r,
            clearance=LID_CLEARANCE_MM / r,
        )


@dataclasses.dataclass
class EyeFrame:
    """An eye's centre (Anny's frame, metres), radius and axes (columns: +X, up, gaze)."""

    side: str
    centre: np.ndarray
    radius: float
    axes: np.ndarray

    def to_world(self, local: np.ndarray) -> np.ndarray:
        """Points (..., 3) in eyeball radii of the eye's frame, in Anny's frame."""
        return self.centre + (np.asarray(local) * self.radius) @ self.axes.T

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """Points (..., 3) of Anny's frame in eyeball radii of the eye's frame."""
        return ((np.asarray(points) - self.centre) @ self.axes) / self.radius

    def directions(self, vectors: np.ndarray) -> np.ndarray:
        """Directions (..., 3) of the eye's frame in Anny's frame."""
        return np.asarray(vectors) @ self.axes.T


# Anny's rest frame: the figure faces -Y with Z up; +X is the figure's left
GAZE = np.array([0.0, -1.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])


def fit_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """The centre and radius of the least-squares sphere through points (N, 3)."""
    points = np.asarray(points, dtype=np.float64)
    a = np.c_[2.0 * points, np.ones(len(points))]
    b = (points**2).sum(1)
    x = np.linalg.lstsq(a, b, rcond=None)[0]
    centre = x[:3]
    return centre, float(np.sqrt(x[3] + centre @ centre))


def eye_frame(side: str, points: np.ndarray) -> EyeFrame:
    """The frame of an eye whose eyeball has the vertices ``points`` (N, 3)."""
    centre, radius = fit_sphere(points)
    axes = np.stack([np.cross(UP, GAZE), UP, GAZE], axis=1)
    return EyeFrame(side, centre, radius, axes)


@dataclasses.dataclass
class Surface:
    """
    A surface in an eye's frame: points and unit normals (M, 3) in eyeball radii, the polar
    coordinates (M, 2) of the eye texture (``s``, the distance from the gaze axis in front and
    2 minus it behind, and the angle about the gaze from +X toward up), and triangles (T, 3).
    """

    points: np.ndarray
    normals: np.ndarray
    polar: np.ndarray
    triangles: np.ndarray


def _polar(points: np.ndarray) -> np.ndarray:
    rho = np.hypot(points[:, 0], points[:, 1])
    s = np.where(points[:, 2] >= 0.0, rho, 2.0 - rho)
    return np.stack([s, np.arctan2(points[:, 1], points[:, 0])], axis=1)


def revolve(strips: list[np.ndarray], segments: int) -> Surface:
    """
    The surface of revolution about the gaze of profile strips, each a (K, 4) array of points
    (rho, z) and normals (n_rho, n_z) running outward from the front. A point on the axis
    becomes one vertex (a pole). Strips are not joined: a crease between two strips repeats
    its points with the normals of each side.
    """
    phi = 2.0 * np.pi * np.arange(segments) / segments
    cos, sin = np.cos(phi), np.sin(phi)
    points, normals, triangles = [], [], []
    count = 0
    for strip in strips:
        rings = []
        for rho, z, nr, nz in np.asarray(strip, dtype=np.float64):
            n = np.array([nr, nz]) / np.hypot(nr, nz)
            if rho <= 1e-12:
                rings.append(np.array([count]))
                points.append([[0.0, 0.0, z]])
                normals.append([[0.0, 0.0, np.sign(n[1]) or 1.0]])
                count += 1
            else:
                rings.append(count + np.arange(segments))
                points.append(np.stack([rho * cos, rho * sin, np.full(segments, z)], 1))
                normals.append(
                    np.stack([n[0] * cos, n[0] * sin, np.full(segments, n[1])], 1)
                )
                count += segments
        for a, b in zip(rings[:-1], rings[1:]):
            j = np.arange(segments)
            k = (j + 1) % segments
            if len(a) == 1:
                triangles.append(np.stack([np.full(segments, a[0]), b[j], b[k]], 1))
            elif len(b) == 1:
                triangles.append(np.stack([a[j], np.full(segments, b[0]), a[k]], 1))
            else:
                triangles.append(np.stack([a[j], b[j], b[k]], 1))
                triangles.append(np.stack([a[j], b[k], a[k]], 1))
    points = np.concatenate(points)
    normals = np.concatenate(normals)
    triangles = np.concatenate(triangles).astype(np.int64)
    # wind every triangle so that its face normal agrees with its vertices' normals
    a, b, c = (points[triangles[:, i]] for i in range(3))
    face = np.cross(b - a, c - a)
    flip = (face * normals[triangles].sum(1)).sum(1) < 0
    triangles[flip] = triangles[flip][:, [0, 2, 1]]
    return Surface(points, normals, _polar(points), triangles)


def limbus(iris_radius: float, radius: float = 1.0) -> tuple[float, float]:
    """The limbus circle on a sphere of ``radius``: its distance from the gaze axis and its
    height along the gaze (eyeball radii)."""
    return iris_radius, float(np.sqrt(radius**2 - iris_radius**2))


def _sclera(theta_from: float, theta_to: float, radius: float = 1.0) -> np.ndarray:
    """Sclera profile points (rho, z, n_rho, n_z) from angle ``theta_from`` to ``theta_to``
    (radians from the gaze): fine in front, coarse behind."""
    fine = np.radians(min(SCLERA_FINE, np.degrees(theta_to)))
    steps = max(1, int(np.ceil((fine - theta_from) / np.radians(SCLERA_STEP))))
    angles = list(np.linspace(theta_from, fine, steps + 1))
    if theta_to > fine + 1e-9:
        back = max(1, int(np.ceil((theta_to - fine) / np.radians(30.0))))
        angles += list(np.linspace(fine, theta_to, back + 1)[1:])
    angles = np.array(angles)
    rho, z = radius * np.sin(angles), radius * np.cos(angles)
    rho[np.abs(angles - np.pi) < 1e-9] = 0.0
    return np.stack([rho, z, np.sin(angles), np.cos(angles)], 1)


def _bulge(shape: EyeShape, radius: float, bulge: float, count: int = 7):
    """Profile points of the corneal bulge over a sphere of ``radius``: a sphere of the
    cornea's radius of curvature through the limbus, its height above the limbus scaled by
    ``bulge`` (1 is the full height)."""
    rho_l, z_j = limbus(shape.iris, radius)
    rc = max(shape.cornea, rho_l * 1.0001)
    z_cc = z_j - np.sqrt(rc**2 - rho_l**2)
    rho = rho_l * np.linspace(0.0, 1.0, count)
    root = np.sqrt(rc**2 - rho**2)
    z = z_j + bulge * (z_cc + root - z_j)
    slope = -bulge * rho / root  # dz / drho
    return np.stack([rho, z, -slope, np.ones_like(rho)], 1)


def bulge_height(shape: EyeShape, radius: float = 1.0) -> float:
    """The full height of the corneal bulge above the limbus plane (eyeball radii)."""
    profile = _bulge(shape, radius, 1.0, 2)
    return float(profile[0, 1] - profile[-1, 1])


def outer_radius(theta: np.ndarray, shape: EyeShape, bulge: float, shell: float):
    """The distance from the eye's centre of its outer surface (the cornea shell, or the
    single-surface eyeball) at angles ``theta`` from the gaze (eyeball radii)."""
    theta = np.asarray(theta, dtype=np.float64)
    rho_l, z_j = limbus(shape.iris, shell)
    rc = max(shape.cornea, rho_l * 1.0001)
    z_cc = z_j - np.sqrt(rc**2 - rho_l**2)
    out = np.full(theta.shape, float(shell))
    inside = np.sin(theta) * shell < rho_l
    inside &= np.cos(theta) > 0
    if inside.any():
        # the bulge along the ray at angle theta: z = z_j + k (z_cc + sqrt(rc^2 - rho^2) - z_j),
        # solved for the distance by bisection (the profile is monotonic in rho)
        t = theta[inside]
        lo, hi = np.full(t.shape, shell * 0.9), np.full(t.shape, shell * 1.5)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            rho, z = mid * np.sin(t), mid * np.cos(t)
            surface = z_j + bulge * (
                z_cc + np.sqrt(np.maximum(rc**2 - rho**2, 0.0)) - z_j
            )
            beyond = z > surface
            hi = np.where(beyond, mid, hi)
            lo = np.where(beyond, lo, mid)
        out[inside] = 0.5 * (lo + hi)
    return out


def eyeball_profile(shape: EyeShape) -> list[np.ndarray]:
    """The strips of the eyeball under a cornea: the iris dome, the wall at the limbus and the
    sclera to the back pole."""
    rho_l, z_l = limbus(shape.iris)
    z_rim = z_l - shape.recess
    rho = rho_l * np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.97])
    z = z_rim + shape.dome * (1.0 - (rho / rho_l) ** 2)
    slope = -2.0 * shape.dome * rho / rho_l**2
    iris = np.stack([rho, z, -slope, np.ones_like(rho)], 1)
    rim = z_rim + shape.dome * (1.0 - 0.97**2)
    d_rho, d_z = rho_l - rho[-1], z_l - rim
    wall = np.array([[rho[-1], rim, -d_z, d_rho], [rho_l, z_l, -d_z, d_rho]])
    return [iris, wall, _sclera(np.arcsin(rho_l), np.pi)]


def cornea_profile(shape: EyeShape, bulge: float) -> list[np.ndarray]:
    """The strip of the cornea shell: the bulge, then the shell over the sclera to
    :data:`CORNEA_EXTENT` degrees from the gaze. Its normal at the limbus is the mean of
    both sides."""
    shell = 1.0 + shape.gap
    cap = _bulge(shape, shell, bulge)
    rest = _sclera(np.arcsin(shape.iris / shell), np.radians(CORNEA_EXTENT), shell)
    joint = cap[-1, 2:] / np.hypot(*cap[-1, 2:]) + rest[0, 2:]
    cap[-1, 2:] = joint
    return [np.concatenate([cap, rest[1:]])]


def single_profile(shape: EyeShape, bulge: float) -> list[np.ndarray]:
    """The strip of the single-surface eyeball: the bulge, then the sclera to the back."""
    cap = _bulge(shape, 1.0, bulge)
    rest = _sclera(np.arcsin(shape.iris), np.pi)
    joint = cap[-1, 2:] / np.hypot(*cap[-1, 2:]) + rest[0, 2:]
    cap[-1, 2:] = joint
    return [np.concatenate([cap, rest[1:]])]


# the eyeball shrinks at most to this share of its fitted radius to clear the lids, and this
# share of the lids' vertices (at least BULGE_TOLERANCE_COUNT) may come closer than the
# clearance: a fitted or hand-made mesh may have lids that touch the eyeball here and there
MIN_SPHERE = 0.97
BULGE_TOLERANCE = 0.01
BULGE_TOLERANCE_COUNT = 2


def bulge_clearance(
    frame: EyeFrame,
    skin: np.ndarray,
    skin_normals: np.ndarray,
    shape: EyeShape,
    shell: float,
) -> tuple[float, float]:
    """
    The largest share of the corneal bulge (0 to 1) and the radius scale of the eyeball (from
    :data:`MIN_SPHERE` to 1) for which the outward-facing skin near the eye (the lids, within
    80 degrees of the gaze, outside the fitted sphere) stays the shape's clearance beyond the
    eye's outer surface, whose sclera lies at ``shell`` eyeball radii. Skin inside the fitted
    sphere already met anny's eyeball and is left aside, and a few vertices
    (:data:`BULGE_TOLERANCE`) may come closer.
    """
    clearance = shape.clearance
    local = frame.to_local(skin)
    dist = np.linalg.norm(local, axis=1)
    normals = skin_normals @ frame.axes
    facing = (normals * local).sum(1) / np.maximum(dist, 1e-12)
    theta = np.arccos(np.clip(local[:, 2] / np.maximum(dist, 1e-12), -1.0, 1.0))
    keep = (dist >= 1.0) & (dist < 1.35) & (facing > 0.0) & (theta < np.radians(80.0))
    if not keep.any():
        return 1.0, 1.0
    dist, theta = dist[keep], theta[keep]
    allowed = max(BULGE_TOLERANCE_COUNT, int(BULGE_TOLERANCE * len(dist)))
    if allowed >= len(dist):
        return 1.0, 1.0
    sphere = float(
        np.clip(np.sort(dist / (shell + clearance))[allowed], MIN_SPHERE, 1.0)
    )

    def fits(bulge):
        surface = sphere * outer_radius(theta, shape, bulge, shell) + clearance
        return int((dist < surface).sum()) <= allowed

    if fits(1.0):
        return 1.0, sphere
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo, sphere


def socket(
    frame: EyeFrame, vertices: np.ndarray, normals: np.ndarray, skin: np.ndarray
):
    """
    The mask (N,) of the skin of an eye's socket on a mesh: the vertices of ``skin`` (a mask
    of the vertices that are not on an eyeball) within 1.35 eyeball radii of the eye's centre
    whose unit ``normals`` face it (the inside of the lids, the conjunctiva).
    """
    local = frame.to_local(vertices)
    dist = np.linalg.norm(local, axis=1)
    facing = ((normals @ frame.axes) * local).sum(1) / np.maximum(dist, 1e-12)
    return skin & (dist < 1.35) & (facing < 0.0)


# the caruncle: the depth raster of the corner has this step (eyeball radii); skin within this
# distance of the eyeball's centre counts as the socket; the pocket lies at least this far
# from the gaze toward the nose; the mound rises by CARUNCLE_HEIGHT over the plane through its
# rim, which sinks by CARUNCLE_SINK under what lies there, and it reaches CARUNCLE_REACH times
# the pocket's spread (two standard deviations) along each axis
CARUNCLE_STEP = 0.01
CARUNCLE_NEAR = 1.12
CARUNCLE_NASAL = 0.4
CARUNCLE_HEIGHT = 0.025
CARUNCLE_SINK = 0.02
CARUNCLE_REACH = 0.75
# the pocket lies at least this far (eyeball radii) below the level of the lids around it, the
# grey closing of the front view's depth over a window of CARUNCLE_WINDOW
CARUNCLE_RECESS = 0.03
CARUNCLE_WINDOW = 0.25


def _depth_raster(
    points: np.ndarray, triangles: np.ndarray, xs: np.ndarray, ys: np.ndarray
):
    """The largest z (H, W) of triangles over a grid of x and y (an orthographic view along
    -z from the front), -inf where none lies."""
    depth = np.full((len(ys), len(xs)), -np.inf)
    x0, dx, y0, dy = xs[0], xs[1] - xs[0], ys[0], ys[1] - ys[0]
    for tri in triangles:
        a, b, c = points[tri]
        lo = np.floor(
            (np.minimum(np.minimum(a, b), c)[:2] - [x0, y0]) / [dx, dy]
        ).astype(int)
        hi = np.ceil(
            (np.maximum(np.maximum(a, b), c)[:2] - [x0, y0]) / [dx, dy]
        ).astype(int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, [len(xs) - 1, len(ys) - 1])
        if (hi < lo).any():
            continue
        gy, gx = np.mgrid[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
        px, py = xs[gx], ys[gy]
        m = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
        det = np.linalg.det(m)
        if abs(det) < 1e-14:
            continue
        inv = np.linalg.inv(m)
        u = inv[0, 0] * (px - a[0]) + inv[0, 1] * (py - a[1])
        v = inv[1, 0] * (px - a[0]) + inv[1, 1] * (py - a[1])
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        z = a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2])
        cur = depth[gy, gx]
        depth[gy, gx] = np.where(inside & (z > cur), z, cur)
    return depth


def caruncle(
    frame: EyeFrame, skin: np.ndarray, skin_triangles: np.ndarray, shape: EyeShape
) -> Surface | None:
    """
    The caruncle of an eye: a low mound over the pocket in the inner corner of the eye, where
    the front view between the lids shows the skin of the socket instead of the eyeball (anny's
    mesh has no caruncle there, and the pocket reads as a dark hole). ``skin`` (N, 3) are mesh
    vertices in Anny's frame and ``skin_triangles`` (T, 3) the skin's triangles over them.

    The front view is a depth raster along the gaze (:func:`_depth_raster`) of the skin and
    the eyeball's sphere. The opening is what that view sees of the sphere, or of skin within
    :data:`CARUNCLE_NEAR` eyeball radii of its centre, connected to the gaze axis; the pocket,
    the skin of the opening on the side of the nose. The mound covers an ellipse of the
    pocket's spread, spans the plane through what lies at its rim, and sinks there under the
    skin or the sclera, so that its edge does not show. Returns None when the view shows no
    pocket.
    """
    local = frame.to_local(skin)
    nasal = -1.0 if frame.centre[0] >= 0 else 1.0  # the nose: -X of the left eye (+X)
    near = (np.linalg.norm(local, axis=1) < 1.6) & (local[:, 2] > -0.2)
    tris = skin_triangles[near[skin_triangles].all(1)]
    step = CARUNCLE_STEP
    xs = np.arange(-1.3, 1.3 + step / 2, step)
    ys = np.arange(-0.8, 0.8 + step / 2, step)
    depth = _depth_raster(local, tris, xs, ys)
    gx, gy = np.meshgrid(xs, ys)
    rr = gx**2 + gy**2
    sphere = np.where(rr < 1.0, np.sqrt(np.maximum(1.0 - rr, 0.0)), -np.inf)
    skin_first = depth > sphere
    radius = np.sqrt(rr + np.where(np.isfinite(depth), depth, 0.0) ** 2)
    opening = (~skin_first & np.isfinite(sphere)) | (
        skin_first & (radius < CARUNCLE_NEAR)
    )
    from scipy import ndimage

    labels, _ = ndimage.label(opening)
    centre_label = labels[len(ys) // 2, len(xs) // 2]
    if centre_label == 0:
        return None
    # the surface that the front view sees: the front of the skin or of the sphere, and the
    # level of the lids around each point (the valleys of the view filled)
    under = np.maximum(depth, sphere)
    finite = np.isfinite(under)
    if not finite.all():
        _, (iy, ix) = ndimage.distance_transform_edt(~finite, return_indices=True)
        under = under[iy, ix]
    window = max(3, int(round(CARUNCLE_WINDOW / step)))
    lids = ndimage.grey_closing(under, size=(window, window))
    recessed = lids - under > CARUNCLE_RECESS
    pocket = (
        (labels == centre_label) & skin_first & recessed & (nasal * gx > CARUNCLE_NASAL)
    )
    if pocket.sum() < 6:
        return None
    # the pocket and the eyeball's sclera beside it: where the mound may stand
    inner = ndimage.binary_dilation(pocket, iterations=2) | ~skin_first
    pts = np.stack([gx[pocket], gy[pocket]], 1)
    mean = pts.mean(0)
    values, vectors = np.linalg.eigh(np.cov(pts.T))
    axes = vectors * (CARUNCLE_REACH * 2.0 * np.sqrt(np.maximum(values, step**2)))

    def sample(grid, x, y):
        ix = np.clip(np.rint((x - xs[0]) / step).astype(int), 0, len(xs) - 1)
        iy = np.clip(np.rint((y - ys[0]) / step).astype(int), 0, len(ys) - 1)
        return grid[iy, ix]

    def below(x, y):
        return sample(under, x, y)

    rings = np.array([0.0, 0.2, 0.4, 0.6, 0.75, 0.88, 1.0])
    segments = 20
    psi = 2.0 * np.pi * np.arange(segments) / segments
    circle = np.stack([np.cos(psi), np.sin(psi)], 1)
    rim = mean + circle @ axes.T
    rim_z = below(rim[:, 0], rim[:, 1])
    plane = np.linalg.lstsq(np.c_[rim, np.ones(len(rim))], rim_z, rcond=None)[0]
    points = []
    for u in rings:
        xy = mean[None] + (u * circle) @ axes.T if u > 0 else mean[None]
        flat = np.c_[xy, np.ones(len(xy))] @ plane
        z = flat + CARUNCLE_HEIGHT * (1.0 - u**2) ** 0.7
        # under the level of the lids, and well under the skin outside the pocket (the
        # lids, which an oblique view sees past their edges); the rim sinks under what lies
        # there
        z = np.minimum(z, sample(lids, xy[:, 0], xy[:, 1]) - CARUNCLE_SINK)
        outside = ~sample(inner, xy[:, 0], xy[:, 1])
        tucked = below(xy[:, 0], xy[:, 1]) - 2.0 * CARUNCLE_SINK
        z = np.where(outside, np.minimum(z, tucked), z)
        z = np.where(u >= 1.0, below(xy[:, 0], xy[:, 1]) - CARUNCLE_SINK, z)
        points.append(np.c_[xy, z])
    index = [np.array([0])]
    for _ in rings[1:]:
        index.append(sum(len(i) for i in index) + np.arange(segments))
    points = np.concatenate(points)
    j = np.arange(segments)
    k = (j + 1) % segments
    triangles = [np.stack([np.zeros(segments, int), index[1][j], index[1][k]], 1)]
    for a, b in zip(index[1:-1], index[2:]):
        triangles.append(np.stack([a[j], b[j], b[k]], 1))
        triangles.append(np.stack([a[j], b[k], a[k]], 1))
    triangles = np.concatenate(triangles).astype(np.int64)
    from opensculptboy.export.body import vertex_normals

    a, b, c = (points[triangles[:, i]] for i in range(3))
    back = np.cross(b - a, c - a)[:, 2] < 0  # every face looks toward the front (+z)
    triangles[back] = triangles[back][:, [0, 2, 1]]
    normals = vertex_normals(points, triangles)
    return Surface(points, normals, _polar(points), triangles)

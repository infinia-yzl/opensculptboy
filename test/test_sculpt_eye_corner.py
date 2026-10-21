# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of the eye-corner patch of the ``sculpt`` topology
(``opensculptboy.topology.patches.eye_corner``, docs/sculpt-topology.md): the patch format, the
ray test of the inner corner from the front and from 35 degrees to either side and 20 degrees
up or down (no ray between the lids reaches the skin of the socket behind the new surface, and
the dark pocket that anny shows there closes), the caruncle clear of the eyeball and the lids,
no crossing or flipped face, at rest, with the eyes turned, in a blink, on phenotype corners and
face-shape samples, and the corner on the detail levels 1 and 2.

The views are orthographic depth rasters along each view direction in the eye's frame
(:func:`view_raster`), so the test needs no ray tracer.
"""

import functools
import unittest

import numpy as np
import torch

from opensculptboy.eyes import geometry
from opensculptboy.topology import base_quad_mesh, validate_patch
from opensculptboy.topology.patches import eye_corner

# degrees toward the temple (yaw) and up (pitch) off the gaze
VIEWS = tuple(
    (yaw, pitch) for yaw in (-35, -17.5, 0, 17.5, 35) for pitch in (-20, 0, 20)
)
# the raster's step and the window of the lids' level (eyeball radii); a pixel of the pocket
# lies RECESS below that level, within NEAR of the eye's centre and NASAL toward the nose
STEP = 0.01
WINDOW = 0.25
RECESS = 0.06
NEAR = 1.12
NASAL = 0.4


def split(quads):
    """Triangles of quads (a triangle repeats its last index) and the quad of each."""
    quads = np.asarray(quads)
    tri = quads[:, 2] == quads[:, 3]
    first = quads[:, [0, 1, 3]]
    second = quads[~tri][:, [1, 2, 3]]
    source = np.r_[np.arange(len(quads)), np.flatnonzero(~tri)]
    return np.concatenate([first, second]), source


def raster(points, triangles, xs, ys):
    """The largest z (H, W) of the triangles over a grid (a view along -z) and the triangle
    there (-1 where none)."""
    depth = np.full((len(ys), len(xs)), -np.inf)
    ident = np.full((len(ys), len(xs)), -1)
    x0, dx, y0, dy = xs[0], xs[1] - xs[0], ys[0], ys[1] - ys[0]
    p = points[triangles]
    lo = np.floor((p.min(1)[:, :2] - [x0, y0]) / [dx, dy]).astype(int)
    hi = np.ceil((p.max(1)[:, :2] - [x0, y0]) / [dx, dy]).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [len(xs) - 1, len(ys) - 1])
    for t in np.flatnonzero((hi >= lo).all(1)):
        a, b, c = p[t]
        det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(det) < 1e-14:
            continue
        gy, gx = np.mgrid[lo[t, 1] : hi[t, 1] + 1, lo[t, 0] : hi[t, 0] + 1]
        px, py = xs[gx] - a[0], ys[gy] - a[1]
        u = ((c[1] - a[1]) * px - (c[0] - a[0]) * py) / det
        v = (-(b[1] - a[1]) * px + (b[0] - a[0]) * py) / det
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        z = a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2])
        better = inside & (z > depth[gy, gx])
        depth[gy, gx] = np.where(better, z, depth[gy, gx])
        ident[gy, gx] = np.where(better, t, ident[gy, gx])
    return depth, ident


def corner_views(frame, points, groups, views=VIEWS):
    """
    What the views of an eye's inner corner see: ``groups`` maps a name to triangles over
    ``points`` (Anny's frame), and the eyeball is its fitted sphere (as the eyes of
    :mod:`opensculptboy.eyes` draw it; anny's own eyeball of 72 vertices lies inside it, and
    rays pass between its facets and the lids). Returns, by view, the area (mm^2) where the
    first surface is ``"socket"`` skin inside the eyeball's sphere on the side of the nose
    (``deep``), and the area of the pocket (``hole``): the first surface not the eyeball and
    RECESS below the level of what lies around it (the grey closing over WINDOW).
    """
    from scipy import ndimage

    local = frame.to_local(points)
    names = list(groups)
    tris = np.concatenate([groups[n] for n in names]).astype(np.int64)
    kind = np.concatenate([np.full(len(groups[n]), i) for i, n in enumerate(names)])
    near = np.linalg.norm(local[tris].mean(1), axis=1) < 4.5
    tris, kind = tris[near], kind[near]
    nasal = -1.0 if frame.centre[0] >= 0 else 1.0
    xs = np.arange(-1.4, 1.4 + STEP / 2, STEP)
    ys = np.arange(-0.9, 0.9 + STEP / 2, STEP)
    gx, gy = np.meshgrid(xs, ys)
    window = max(3, int(round(WINDOW / STEP)))
    pixel = (STEP * frame.radius * 1000.0) ** 2
    out = {"deep": {}, "hole": {}}
    for yaw, pitch in views:
        a, b = np.radians(yaw), np.radians(pitch)
        view = np.array(
            [-nasal * np.sin(a) * np.cos(b), np.sin(b), np.cos(a) * np.cos(b)]
        )
        right = np.cross([0.0, 1.0, 0.0], view)
        right /= np.linalg.norm(right)
        turn = np.stack([right, np.cross(view, right), view])
        depth, ident = raster(local @ turn.T, tris, xs, ys)
        sphere = np.sqrt(np.maximum(1.0 - gx**2 - gy**2, 0.0))
        sphere[gx**2 + gy**2 > 1.0] = -np.inf
        on_sphere = sphere > depth
        depth = np.where(on_sphere, sphere, depth)
        ident = np.where(on_sphere, -2, ident)
        finite = np.isfinite(depth)
        _, (iy, ix) = ndimage.distance_transform_edt(~finite, return_indices=True)
        filled = depth[iy, ix]
        level = ndimage.grey_closing(filled, size=(window, window))
        hit = np.stack([gx, gy, filled], -1) @ turn
        first = np.where(ident >= 0, kind[np.maximum(ident, 0)], -1)
        first[ident == -2] = names.index("eyeball")
        corner = (nasal * hit[..., 0] > NASAL) & (hit[..., 2] > 0.2)
        radius = np.linalg.norm(hit, axis=-1)
        deep = (first == names.index("socket")) & (radius < 1.0) & corner
        hole = (first >= 0) & (first != names.index("eyeball")) & corner
        hole &= (level - filled > RECESS) & (radius < NEAR)
        out["deep"][(yaw, pitch)] = float(deep.sum()) * pixel
        out["hole"][(yaw, pitch)] = float(hole.sum()) * pixel
    return out


def _segments_cross(p0, p1, a, b, c):
    d = p1 - p0
    e1, e2 = b - a, c - a
    h = np.cross(d, e2)
    det = (e1 * h).sum(1)
    ok = np.abs(det) > 1e-18
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = p0 - a
    u = (s * h).sum(1) * inv
    q = np.cross(s, e1)
    v = (d * q).sum(1) * inv
    t = (e2 * q).sum(1) * inv
    return (
        ok & (u > 1e-9) & (v > 1e-9) & (u + v < 1 - 1e-9) & (t > 1e-9) & (t < 1 - 1e-9)
    )


def crossings(points, first, second) -> int:
    """The pairs of triangles of ``first`` and ``second`` (no shared vertex) that cross."""
    p = points
    ca, cb = p[first].mean(1), p[second].mean(1)
    ra = np.linalg.norm(p[first] - ca[:, None], axis=2).max(1)
    rb = np.linalg.norm(p[second] - cb[:, None], axis=2).max(1)
    count = 0
    for i in range(len(first)):
        near = np.flatnonzero(np.linalg.norm(cb - ca[i], axis=1) < ra[i] + rb)
        near = near[~np.isin(second[near], first[i]).any(1)]
        if not len(near):
            continue
        a = p[first[i]]
        b = p[second[near]]
        hit = np.zeros(len(near), bool)
        for k in range(3):
            s0 = np.repeat(a[k][None], len(near), 0)
            s1 = np.repeat(a[(k + 1) % 3][None], len(near), 0)
            hit |= _segments_cross(s0, s1, b[:, 0], b[:, 1], b[:, 2])
            corners = [np.repeat(a[m][None], len(near), 0) for m in range(3)]
            hit |= _segments_cross(b[:, k], b[:, (k + 1) % 3], *corners)
        count += int(hit.sum())
    return count


def normals(points, triangles):
    n = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    return n / np.linalg.norm(n, axis=1, keepdims=True)


@functools.lru_cache(maxsize=1)
def anny_model():
    import anny

    return anny.Anny(face_shapes="all", facial_actions="all").to(dtype=torch.float64)


def eyes_turned(model, degrees):
    """``local-ref`` pose parameters that turn both eye bones about the vertical."""
    import roma

    labels = list(model.bone_labels)
    params = torch.eye(4, dtype=torch.float64).repeat(1, len(labels), 1, 1)
    turn = roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, np.radians(degrees)]))
    for bone in ("eye.L", "eye.R"):
        params[0, labels.index(bone), :3, :3] = turn
    return params


@functools.lru_cache(maxsize=1)
def states() -> dict:
    """Anny's posed vertices (N, 3) in the states the patch must hold in."""
    from anny.faces.distribution import FaceShapeDistribution

    model = anny_model()
    blink = {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0}

    def run(pose=None, **kwargs):
        with torch.no_grad():
            return model(pose_parameters=pose, **kwargs)["vertices"][0].numpy()

    faces = FaceShapeDistribution(model).sample(
        batch_size=2, generator=torch.Generator().manual_seed(7), spread=1.0
    )
    out = {
        "rest": run(),
        "look out": run(eyes_turned(model, 25.0)),
        "look in": run(eyes_turned(model, -25.0)),
        "half blink": run(facial_actions={k: 0.5 for k in blink}),
        "blink": run(facial_actions=blink),
    }
    corners = {
        "compact": dict(gender=0.0, age=0.0, muscle=0.0, weight=0.0, height=0.0),
        "elongated": dict(gender=1.0, age=1.0, muscle=1.0, weight=1.0, height=1.0),
        "light": dict(gender=1.0, age=0.0, weight=0.0, muscle=1.0),
        "heavy": dict(gender=0.0, age=1.0, weight=1.0, muscle=0.0),
    }
    for name, phenotype in corners.items():
        out[name] = run(phenotype_kwargs=phenotype)
    for i, values in enumerate(faces.double()):
        out[f"face {i}"] = run(face_shape_kwargs=values[None])
        out[f"face {i} blink"] = run(
            face_shape_kwargs=values[None], facial_actions=blink
        )
    return out


BLINKS = ("blink", "face 0 blink", "face 1 blink")
# in a half blink the opening is a slit, and everything between the lids lies below their
# level: the pocket measure does not apply there, the check of the socket does
SLITS = BLINKS + ("half blink",)


class TestEyeCorner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = base_quad_mesh()
        cls.patch = eye_corner.build(cls.base)
        eyes = eye_corner.eyeball_rows(cls.base)
        cls.eyes = eyes
        quads = cls.base.quads
        n = cls.base.vertex_count
        on_eye = np.zeros(n, bool)
        on_eye[np.r_[eyes["L"], eyes["R"]]] = True
        kept = np.ones(len(quads), bool)
        kept[cls.patch.removed_faces] = False
        eyeball = on_eye[quads].all(1)
        cls.eyeball = split(quads[eyeball])[0]
        cls.skin = split(quads[kept & ~eyeball])[0]
        cls.removed = split(quads[cls.patch.removed_faces])[0]
        cls.anny_skin = split(quads[~eyeball])[0]
        cls.new, source = split(cls.patch.faces)
        labels = np.array(cls.patch.labels_of_faces())[source]
        cls.tags = np.array(
            [lab.split("/")[-1] if "/" in lab else "skin" for lab in labels]
        )

    def points(self, vertices):
        return np.concatenate([vertices, self.patch.positions(self.base, vertices)])

    def left(self, points, triangles):
        return triangles[points[triangles].mean(1)[:, 0] > 0]

    def test_patch_follows_the_format(self):
        validate_patch(self.patch, self.base)
        per_eye = len(self.patch.faces) // 2
        self.assertEqual(len(self.patch.faces), 2 * per_eye)
        self.assertTrue(20 <= per_eye <= 60, per_eye)
        self.assertEqual(len(self.patch.removed_faces), 2 * per_eye)
        labels = set(self.patch.labels_of_faces())
        self.assertLessEqual(
            {"eye_corner/caruncle", "eye_corner/conjunctiva"}, labels, labels
        )
        self.assertTrue(
            (self.patch.weights[:, 1:] > 0).any()
        )  # real blends, not copies

    def test_new_vertices_stay_near_their_anchors(self):
        rest = states()["rest"]
        new = self.patch.positions(self.base, rest)
        used = self.patch.weights > 0
        spread = np.linalg.norm(rest[self.patch.anchors] - new[:, None], axis=2)
        self.assertLess(float(spread[used].max()), 0.012)  # metres

    def views(self, name, patched=True):
        vertices = states()[name]
        frame = geometry.eye_frame("L", vertices[self.eyes["L"]])
        if not patched:
            groups = {"socket": self.anny_skin, "eyeball": self.eyeball}
            return corner_views(frame, vertices, groups)
        groups = {"socket": self.skin, "eyeball": self.eyeball, "patch": self.new}
        return corner_views(frame, self.points(vertices), groups)

    def test_no_ray_reaches_the_socket_behind_the_corner(self):
        # From the front and from 35 degrees to either side and 20 up or down, between the
        # lids: no view sees anny's skin of the socket behind the new surface, and the dark
        # pocket (what lies well below the level around it) closes
        for name in states():
            with self.subTest(state=name):
                before = self.views(name, patched=False)
                after = self.views(name)
                self.assertEqual(len(after["deep"]), len(VIEWS))
                self.assertLess(max(after["deep"].values()), DEEP_LIMIT, after["deep"])
                if name in SLITS:
                    continue
                self.assertLess(
                    sum(after["hole"].values()),
                    HOLE_SHARE * sum(before["hole"].values()) + 0.5,
                    (before["hole"], after["hole"]),
                )

    def test_caruncle_stays_clear(self):
        # The caruncle never crosses the eyeball or the lids, the patch never crosses itself
        # or anny's skin, at rest, with the eyes turned, in a blink and over the bodies
        caruncle = self.new[self.tags == "caruncle"]
        for name, vertices in states().items():
            with self.subTest(state=name):
                p = self.points(vertices)
                mine = self.left(p, caruncle)
                centre = p[self.eyes["L"]].mean(0)
                near = self.skin[
                    np.linalg.norm(p[self.skin].mean(1) - centre, axis=1) < 0.04
                ]
                self.assertEqual(crossings(p, mine, self.eyeball), 0)
                self.assertEqual(crossings(p, mine, near), 0)
                new = self.left(p, self.new)
                if name in BLINKS:
                    # a full blink folds the corner under the closed lids: no more crossings
                    # of the lids than anny's own pocket makes
                    pocket = self.left(vertices, self.removed)
                    self.assertLessEqual(
                        crossings(p, new, near), crossings(vertices, pocket, near)
                    )
                    continue
                self.assertEqual(crossings(p, new, new), 0)
                self.assertEqual(crossings(p, new, near), 0)

    def test_no_flipped_faces(self):
        # Each new face keeps its orientation of the default character over the bodies, as
        # far as anny's own faces of the pocket keep theirs (the most compact bodies turn one
        # of anny's faces at the inner ring almost edge-on)
        rest = states()["rest"]
        reference = normals(self.points(rest), self.new)
        anny_reference = normals(rest, self.removed)
        for name, vertices in states().items():
            if name in BLINKS:
                continue
            with self.subTest(state=name):
                n = normals(self.points(vertices), self.new)
                anny = normals(vertices, self.removed)
                floor = (
                    min(0.0, float((anny * anny_reference).sum(1).min())) - FLIP_MARGIN
                )
                self.assertGreater(float((n * reference).sum(1).min()), floor)

    def test_detail_levels_stay_smooth(self):
        # Catmull-Clark of the patched quads: no crease of the patch's visible surface
        # sharper than anny's own corner (or than SMOOTH_LIMIT)
        from anny.utils.subdivision import catmull_clark

        rest = states()["rest"]
        quads = self.base.quads
        kept = np.ones(len(quads), bool)
        kept[self.patch.removed_faces] = False
        centre = rest[self.eyes["L"]].mean(0)

        def sharpest(points, faces, is_patch, levels, tags=None):
            local = np.linalg.norm(points[faces].mean(1) - centre, axis=1) < 0.03
            faces, is_patch = faces[local], is_patch[local]
            used = np.unique(faces)
            remap = np.full(len(points), -1)
            remap[used] = np.arange(len(used))
            q, p, mark = remap[faces], points[used], is_patch
            fold = (
                np.zeros(len(q), bool) if tags is None else tags[local] == "conjunctiva"
            )
            for _ in range(levels):
                level = catmull_clark(len(p), q)
                p = np.asarray(level.operator.apply(p))
                mark = mark[level.parent_faces]
                fold = fold[level.parent_faces]
                q = level.quads
            n = normals(p, q[:, [0, 1, 2]])
            # the angle across each edge between two faces of the patch region
            edges = {}
            worst = 0.0
            for f, face in enumerate(q):
                if not mark[f]:
                    continue
                for k in range(4):
                    key = tuple(sorted((face[k], face[(k + 1) % 4])))
                    if key in edges and not (fold[f] and fold[edges[key]]):
                        worst = max(
                            worst,
                            float(
                                np.degrees(
                                    np.arccos(np.clip(n[f] @ n[edges[key]], -1, 1))
                                )
                            ),
                        )
                    else:
                        edges[key] = f
            return worst

        sculpt = self.points(rest)
        faces = np.concatenate([quads[kept], self.patch.faces])
        is_patch = np.r_[
            np.zeros(kept.sum(), bool), np.ones(len(self.patch.faces), bool)
        ]
        labels = self.patch.labels_of_faces()
        tags = np.r_[["anny"] * int(kept.sum()), [lab.split("/")[-1] for lab in labels]]
        anny_faces = quads
        anny_mark = np.zeros(len(quads), bool)
        anny_mark[self.patch.removed_faces] = True
        for levels in (1, 2):
            with self.subTest(level=levels):
                # the visible surface: the fold's crest, where the conjunctiva turns back
                # under the caruncle, is a fold by design and hidden under it
                mine = sharpest(sculpt, faces, is_patch, levels, tags)
                theirs = sharpest(rest, anny_faces, anny_mark, levels)
                self.assertLess(mine, max(theirs, SMOOTH_LIMIT[levels]), (mine, theirs))


class TestSculptEyes(unittest.TestCase):
    """The eyes of :mod:`opensculptboy.eyes` on the ``sculpt`` topology: no caruncle part of
    their own, the patch's caruncle and conjunctiva inside the lid opening drawn with the
    conjunctiva's pink, wet material, in the GLB and the VRM files, and none of it on a file
    whose skin is baked."""

    @classmethod
    def setUpClass(cls):
        import tempfile

        from opensculptboy import Character
        from opensculptboy.topology import compose, use_sculpt_data

        cls.data = compose(["eye_corner"])
        cls.scope = use_sculpt_data(cls.data)
        cls.scope.__enter__()
        cls.card = Character(topology="sculpt", material={"eyes": {}})
        cls.model = cls.card.build_model(dtype=torch.float64)
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.scope.__exit__(None, None, None)
        cls.tmp.cleanup()

    def eyed(self, kind="gltf", skin_baked=False):
        from opensculptboy.eyes.mesh import eyed_mesh

        with torch.no_grad():
            rest = self.model()["rest_vertices"][0].numpy()
        eye = self.card.eye_material()
        eyed = eyed_mesh(self.model, None, rest, eye, kind, skin_baked=skin_baked)
        return eyed, rest

    def test_the_patch_is_the_caruncle(self):
        from opensculptboy.export.body import triangulated_faces

        eyed, rest = self.eyed()
        self.assertNotIn("caruncle", {p.name for p in eyed.parts})
        triangles, _ = triangulated_faces(self.model)
        # the triangles of the patch's caruncle and conjunctiva faces
        n = self.data.vertex_count
        pink = [
            i
            for i, label in enumerate(self.data.labels)
            if label in ("eye_corner/caruncle", "eye_corner/conjunctiva")
        ]
        faces = self.data.faces[np.isin(self.data.face_labels, pink)]
        corners = {frozenset(f.tolist()) for f in faces}
        is_pink = np.array(
            [any(set(t.tolist()) <= c for c in corners) for t in triangles]
        )
        self.assertGreater(is_pink.sum(), 40)
        start = int(eyed.kept.sum())
        socket = eyed.part_mask("socket")[:start]
        kept = eyed.triangles[:start]
        # every pink triangle of the patch takes the conjunctiva (the eyeball's triangles are
        # gone from the kept list, so compare by vertices)
        kept_pink = np.array([any(set(t.tolist()) <= c for c in corners) for t in kept])
        # every pink triangle of the patch inside the lid opening takes the conjunctiva, and
        # those whose corners reach out onto the lids keep the skin's material
        from opensculptboy.eyes.mesh import CORNER_REACH

        reach = np.full(len(kept), np.inf)
        for frame in eyed.frames.values():
            d = np.linalg.norm(rest[kept] - frame.centre, axis=2).max(1) / frame.radius
            reach = np.minimum(reach, d)
        inside = kept_pink & (reach < CORNER_REACH)
        self.assertGreater(inside.sum(), 40)
        self.assertTrue(socket[inside].all())
        self.assertFalse(socket[kept_pink & ~inside].any())
        self.assertGreater(int((triangles >= n).any(1).sum()), 0)

    def test_pink_only_inside_the_opening(self):
        """Seen from the front of each eye, the conjunctiva's material lies inside the lid
        opening: where the view sees the eyeball, or skin within NEAR eyeball radii of the
        eye's centre (the tissue between the lids, behind their margins), connected to the
        gaze axis. Before the reach of the pink triangles, the patch's caruncle reached out
        over the lower lid's margin on the default character."""
        from scipy import ndimage

        eyed, rest = self.eyed()
        start = int(eyed.kept.sum())
        skin = eyed.triangles[:start]
        pink = eyed.part_mask("socket")[:start]
        self.assertTrue(pink.any())
        xs = np.arange(-1.6, 1.4 + STEP / 2, STEP)
        ys = np.arange(-0.9, 0.9 + STEP / 2, STEP)
        gx, gy = np.meshgrid(xs, ys)
        rr = gx**2 + gy**2
        for side, frame in eyed.frames.items():
            local = frame.to_local(rest)
            near = (np.linalg.norm(local[skin], axis=2) < 2.4).all(1)
            tris = skin[near]
            depth, ident = raster(local, tris, xs, ys)
            sphere = np.where(rr < 1.0, np.sqrt(np.maximum(1.0 - rr, 0.0)), -np.inf)
            on_skin = depth > sphere
            radius = np.sqrt(rr + np.where(np.isfinite(depth), depth, 0.0) ** 2)
            opening = (~on_skin & np.isfinite(sphere)) | (on_skin & (radius < NEAR))
            labels, _ = ndimage.label(opening)
            opening = labels == labels[np.argmin(np.abs(ys)), np.argmin(np.abs(xs))]
            seen = on_skin & (ident >= 0) & pink[near][np.maximum(ident, 0)]
            with self.subTest(side=side):
                self.assertGreater(int(seen.sum()), 100)
                self.assertEqual(int((seen & ~opening).sum()), 0)

    def test_a_baked_skin_keeps_the_corner(self):
        """A file whose skin is baked shows the reference's own corner: no triangle of the
        mesh takes the conjunctiva's material, and the GLB file has none."""
        import io
        import pathlib

        from PIL import Image

        from opensculptboy import export_glb
        from test.gltf_reader import GLB, structure_errors

        eyed, _ = self.eyed(skin_baked=True)
        self.assertFalse(eyed.part_mask("caruncle", "socket").any())
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), (200, 150, 130)).save(buffer, "PNG")
        path = pathlib.Path(self.tmp.name) / "sculpt_eyes_baked.glb"
        export_glb(
            path, self.card, self.model, textures={"base_color": buffer.getvalue()}
        )
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        names = {
            glb.json["materials"][p["material"]]["name"]
            for p in glb.json["meshes"][0]["primitives"]
        }
        self.assertNotIn("conjunctiva", names)
        self.assertIn("eyes", names)

    def test_fit_leaves_the_corner_out(self):
        from opensculptboy.fit import register

        for label in ("eye_corner/caruncle", "eye_corner/conjunctiva", "eye_corner"):
            self.assertEqual(register.PATCH_ROLES.get(label), "anchors")
        n = self.data.vertex_count
        excluded = register.excluded_vertices(self.model)
        # every new vertex of the patch is left out and placed at the blend of its anchors
        self.assertEqual(len(excluded), n + self.data.new_vertex_count)
        self.assertTrue(excluded[n:].all())
        # anny's own vertices keep their roles but those the patch leaves without a face
        from opensculptboy import Character
        from opensculptboy.topology.lookup import referenced_vertices

        anny = register.excluded_vertices(Character().build_model(dtype=torch.float64))
        used = referenced_vertices(self.model)[:n]
        np.testing.assert_array_equal(excluded[:n][used], anny[used])
        self.assertTrue(excluded[:n][~used].all())

    def test_glb_and_vrm(self):
        import pathlib

        from opensculptboy import export_glb
        from opensculptboy.export.vrm import export_vrm
        from test.gltf_reader import GLB, structure_errors

        folder = pathlib.Path(self.tmp.name)
        path = folder / "sculpt_eyes.glb"
        export_glb(path, self.card, self.model)
        glb = GLB(path)
        self.assertEqual(structure_errors(glb), [])
        names = [
            glb.json["materials"][p["material"]]["name"]
            for p in glb.json["meshes"][0]["primitives"]
        ]
        self.assertEqual(sorted(names), ["conjunctiva", "cornea", "eyes", "skin"])
        vrm = folder / "sculpt_eyes.vrm"
        export_vrm(vrm, self.card, self.model, author="test")
        self.assertGreater(vrm.stat().st_size, 0)


# the limits of the checks: the area of a view that may see the socket (two pixels' worth),
# the share of anny's pocket that may stay, and the sharpest angle between faces of the
# patch on each detail level (degrees)
DEEP_LIMIT = 0.2
HOLE_SHARE = 0.5
FLIP_MARGIN = 0.1
SMOOTH_LIMIT = {1: 60.0, 2: 35.0}


# the lids' edge seen from the front and 15 degrees off it: the largest median, over sectors
# of LID_SECTOR degrees about the gaze, of the distance from the eyeball's edge to the skin
# (eyeball radii) on each level (level 0's lid margin has vertices about 0.1 radii apart,
# and the synthetic target's eyes squint), and the sectors of the inner corner (about 180
# degrees, toward the nose). Without the seating the corner of the real character stood
# 0.12 radii off, and the synthetic target's lids up to 0.07 on level 1
LID_GAP_LIMIT = {0: 0.08, 1: 0.03}
LID_SECTOR = 15
LID_VIEWS = ((0, 0), (15, 0), (-15, 0), (0, 15), (0, -15))
CORNER_SECTORS = (150, 165, 180, 195)


def lid_gaps(points, triangles, centre, radius, axes, nasal, views=LID_VIEWS):
    """
    How far the skin stays from an eyeball along the edge of the opening: the eyeball is the
    sphere (``centre``, ``radius``; ``axes`` (3, 3) has the columns +X, up and the gaze) and
    ``triangles`` the skin. For each view (degrees toward the temple and up), an
    orthographic depth raster along it finds the eyeball's pixels next to the skin's (the
    edge of the eyeball that the view sees), and each of their points on the sphere takes its
    distance to the nearest skin, in radii: about 0 where the skin meets the eyeball, however
    thick the lid, and the width of the gap where a trench or a lip stands off it. They are
    gathered by the angle about the gaze (0 toward the temple, 90 up, 180 toward the nose,
    ``nasal`` the sign of +X toward the nose). Returns {sector start: median}, over all views.
    """
    from scipy import ndimage

    from anny.hair import closest_triangles

    local = ((np.asarray(points) - centre) @ axes) / radius
    near = np.linalg.norm(local[triangles].mean(1), axis=1) < 2.4
    tris = triangles[near]
    step = 0.005
    xs = np.arange(-1.4, 1.4 + step / 2, step)
    ys = np.arange(-1.1, 1.1 + step / 2, step)
    gx, gy = np.meshgrid(xs, ys)
    gaps = {a: [] for a in range(0, 360, LID_SECTOR)}
    for yaw, pitch in views:
        a, b = np.radians(yaw), np.radians(pitch)
        view = np.array(
            [-nasal * np.sin(a) * np.cos(b), np.sin(b), np.cos(a) * np.cos(b)]
        )
        right = np.cross([0.0, 1.0, 0.0], view)
        right /= np.linalg.norm(right)
        turn = np.stack([right, np.cross(view, right), view])
        depth, ident = raster(local @ turn.T, tris, xs, ys)
        inside = gx**2 + gy**2 <= 1.0
        sphere = np.where(
            inside, np.sqrt(np.maximum(1.0 - gx**2 - gy**2, 0.0)), -np.inf
        )
        ball = sphere > depth
        skin = (ident >= 0) & ~ball
        edge = ball & ndimage.binary_dilation(skin) & (gx**2 + gy**2 < 0.98)
        on = np.stack([gx[edge], gy[edge], sphere[edge]], -1) @ turn
        _, _, distance = closest_triangles(on, local, tris)
        angle = np.degrees(np.arctan2(on[:, 1], -nasal * on[:, 0])) % 360
        for start in gaps:
            sel = (angle >= start) & (angle < start + LID_SECTOR)
            gaps[start].extend(distance[sel].tolist())
    return {k: float(np.median(v)) for k, v in gaps.items() if len(v) >= 3}


class TestCornerFit(unittest.TestCase):
    """The synthetic reference fit (test/fit_fixtures.py, with its eyeballs) on ``sculpt`` and
    on ``anny``, levels 0 and 1: the lids and the inner corner meet the eyeball all around the
    opening (``register.seat_lids``: no gap or trench between the skin and the eyeball), the
    written offsets reproduce the fitted mesh, and the corner turns no more faces over than
    anny's own corner does."""

    @classmethod
    def setUpClass(cls):
        import pathlib
        import tempfile

        from opensculptboy.fit.handoff import load_handoff, write_handoff
        from opensculptboy.fit.reference import Reference
        from opensculptboy.fit.register import FitOptions, fit_model, fit_reference
        from opensculptboy.topology import compose, use_sculpt_data
        from test.fit_fixtures import synthetic_target

        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name)
        target = synthetic_target(1)
        write_handoff(
            cls.folder / "handoff",
            target["vertices"],
            target["triangles"],
            dict(target["landmarks"]),
            parts=[
                (f"eye_{side}", v, t, "eyes") for side, (v, t) in target["eyes"].items()
            ],
        )
        cls.reference = Reference.from_handoff(load_handoff(cls.folder / "handoff"))
        cls.data = compose(["eye_corner"])
        cls.scope = use_sculpt_data(cls.data)
        cls.scope.__enter__()
        cls.models, cls.results, cls.cards = {}, {}, {}
        for topology in ("anny", "sculpt"):
            cls.models[topology] = fit_model(topology)
            cls.results[topology] = fit_reference(
                cls.reference,
                detail=1,
                name=topology,
                options=FitOptions(topology=topology, measure=False),
                model=cls.models[topology],
            )
            cls.cards[topology] = cls.results[topology].write(
                cls.folder / f"{topology}.json"
            )

    @classmethod
    def tearDownClass(cls):
        cls.scope.__exit__(None, None, None)
        cls.tmp.cleanup()

    def level(self, topology, subdiv):
        """The fitted level's posed vertices, its triangles without the eyeballs, and the
        parametric posed vertices."""
        from opensculptboy.detail import detail_topology
        from opensculptboy.export.vrm_tables import eyeball_vertices
        from opensculptboy.fit import register
        from opensculptboy.fit import residual as R_

        model = self.models[topology]
        result = self.results[topology]
        rest0, transforms = register._card_rest(model, result.character)
        level = detail_topology(model, subdiv, "full", topology)
        w, i = level.skin_weights(
            model.vertex_bone_weights.detach().double().numpy(),
            model.vertex_bone_indices.cpu().numpy(),
        )
        parametric = np.asarray(level.apply(rest0))
        posed = R_.pose_vertices(
            parametric + result.offsets[subdiv],
            w,
            i,
            transforms,
            result.rotation,
            result.translation,
        )
        eye = np.zeros(len(rest0))
        for rows in eyeball_vertices(model).values():
            eye[rows] = 1.0
        on_eye = np.asarray(level.apply(eye)) > 0.5
        triangles = level.triangles[~on_eye[level.triangles].all(1)]
        flat = R_.pose_vertices(
            parametric, w, i, transforms, result.rotation, result.translation
        )
        return posed, triangles, flat, (model, transforms, result)

    def test_written_offsets_reproduce_the_fit(self):
        import json

        from opensculptboy import Character
        from opensculptboy.detail import character_offsets

        for topology in ("anny", "sculpt"):
            card = Character.from_dict(json.loads(self.cards[topology].read_text()))
            for subdiv in (0, 1):
                with self.subTest(topology=topology, subdiv=subdiv):
                    written = character_offsets(card, subdiv, self.folder)
                    np.testing.assert_allclose(
                        written, self.results[topology].offsets[subdiv], atol=1e-6
                    )
                    posed, _, _, _ = self.level(topology, subdiv)
                    fitted, _ = self.results[topology].meshes[subdiv]
                    self.assertLess(float(np.abs(posed - fitted).max()), 1e-6)

    def test_lids_meet_the_eyeball(self):
        from opensculptboy.fit import register

        self.assertEqual(len(self.reference.eyes), 2)
        for topology in ("anny", "sculpt"):
            for subdiv in (0, 1):
                posed, triangles, _, (model, transforms, result) = self.level(
                    topology, subdiv
                )
                eyes = self.reference.eyes
                gazes = register.eye_gazes(
                    model, eyes, transforms, result.rotation, result.translation
                )
                up0 = result.rotation @ np.array([0.0, 0.0, 1.0])
                for (centre, radius), gaze in zip(eyes, gazes):
                    up = up0 - (up0 @ gaze) * gaze
                    up /= np.linalg.norm(up)
                    axes = np.stack([np.cross(up, gaze), up, gaze], axis=1)
                    side = (result.rotation.T @ (centre - result.translation))[0]
                    nasal = -1.0 if side > 0 else 1.0
                    gaps = lid_gaps(posed, triangles, centre, radius, axes, nasal)
                    with self.subTest(topology=topology, subdiv=subdiv, side=side > 0):
                        worst = max(gaps, key=gaps.get)
                        corner = max(gaps.get(a, 0.0) for a in CORNER_SECTORS)
                        print(
                            f"\n{topology} level {subdiv} {'L' if side > 0 else 'R'}: "
                            f"largest gap {gaps[worst]:.3f} radii at {worst} degrees, "
                            f"corner {corner:.3f}, sectors {len(gaps)}"
                        )
                        self.assertGreaterEqual(len(gaps), 20)
                        self.assertLess(gaps[worst], LID_GAP_LIMIT[subdiv])

    def corner_flips(self, topology, subdiv):
        """Faces of the eye corner (the patch's faces on sculpt, the pocket's on anny) on
        level ``subdiv`` outside the eyeball whose fitted normal turns away from the
        parametric mesh's (the seating tucks the corner's fold under the eyeball, where no
        view reaches it)."""
        from opensculptboy.detail import detail_topology

        model = self.models[topology]
        fitted, triangles = self.results[topology].meshes[subdiv]
        _, _, parametric, _ = self.level(topology, subdiv)
        level = detail_topology(model, subdiv, "full", topology)
        corner = np.zeros(len(self.results[topology].parametric_rest))
        if topology == "sculpt":
            corner[np.unique(self.data.faces)] = 1.0
        else:
            quads = base_quad_mesh().quads
            corner[np.unique(quads[self.data.removed_faces])] = 1.0
        inside = np.asarray(level.apply(corner))[triangles].min(1) > 0.5
        under = np.zeros(len(fitted), bool)
        for centre, radius in self.reference.eyes:
            under |= np.linalg.norm(fitted - centre, axis=1) < radius
        inside &= ~under[triangles].all(1)
        a, b = (
            normals(fitted, triangles[inside]),
            normals(parametric, triangles[inside]),
        )
        return int(inside.sum()), int(((a * b).sum(1) < 0).sum())

    def test_no_flipped_faces_beyond_annys(self):
        for subdiv in (0, 1):
            with self.subTest(subdiv=subdiv):
                count, sculpt = self.corner_flips("sculpt", subdiv)
                _, anny = self.corner_flips("anny", subdiv)
                print(
                    f"\nlevel {subdiv}: {sculpt} of {count} corner faces outside the "
                    "eyeball turned over "
                    f"on sculpt, {anny} on anny"
                )
                self.assertGreater(count, 0)
                self.assertLessEqual(sculpt, anny)


if __name__ == "__main__":
    unittest.main()

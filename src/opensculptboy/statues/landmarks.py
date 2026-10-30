# OpenSculptBoy
# Apache License, Version 2.0
"""
Joint and face landmarks of a statue, found on pictures of its scan.

:func:`detect_landmarks` draws the converted figure (Anny's frame: Z up, facing -Y) from the
front and from its left side with an orthographic camera, finds MediaPipe's body landmarks on
both pictures (:func:`opensculptboy.posing.landmarks.detect`, the ``pose`` extra) and carries
them back to 3D: the front picture gives x and z, the side picture y and z. MediaPipe may swap
the left and the right of a figure seen from its side, so each pair of the side picture takes
the order whose heights agree best with the front's. The result uses the landmark names of
docs/reference-fit.md (``elbow_l``, ``knee_r``, ``nose_tip``...), so that the fit takes it
as a person's hints; points of the face snap to the scan later
(:func:`opensculptboy.statues.pipeline.snap_landmarks`).
"""

from __future__ import annotations

import numpy as np

# MediaPipe's names (the person's own left and right) and the contract's
NAMES = {
    "nose": "nose_tip",
    "left_eye_outer": "eye_outer_l",
    "right_eye_outer": "eye_outer_r",
    "left_eye_inner": "eye_inner_l",
    "right_eye_inner": "eye_inner_r",
    "mouth_left": "mouth_corner_l",
    "mouth_right": "mouth_corner_r",
    "left_shoulder": "shoulder_l",
    "right_shoulder": "shoulder_r",
    "left_elbow": "elbow_l",
    "right_elbow": "elbow_r",
    "left_wrist": "wrist_l",
    "right_wrist": "wrist_r",
    "left_hip": "hip_l",
    "right_hip": "hip_r",
    "left_knee": "knee_l",
    "right_knee": "knee_r",
    "left_ankle": "ankle_l",
    "right_ankle": "ankle_r",
}
# the face's points come from the front picture alone (the side picture sees one eye)
FACE = (
    "nose",
    "left_eye_outer",
    "right_eye_outer",
    "left_eye_inner",
    "right_eye_inner",
)
FACE += ("mouth_left", "mouth_right")
PAIRS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
# landmarks that MediaPipe sees less clearly than this are left out
VISIBILITY = 0.5
SIZE = (768, 1152)
MARGIN = 40.0


def _picture(vertices, triangles, view):
    """A grey picture of the figure and the projection that made it."""
    from opensculptboy.render.dressed import Mesh, _frame, draw

    image = draw(
        [Mesh(vertices, triangles, (205, 200, 192))],
        view,
        SIZE,
        MARGIN,
        background=(70, 72, 78),
    )
    return image, _frame(vertices, view, SIZE, MARGIN)


def _unproject(pixel, frame):
    """Picture coordinates (right, up) of a pixel of :func:`_picture`."""
    s, ox, oy = frame
    return (pixel[0] - ox) / s, (oy - pixel[1]) / s


def detect_landmarks(
    vertices, triangles, pictures: dict | None = None, face: bool = False
) -> dict:
    """
    The landmarks (name -> point in Anny's frame) of a figure standing upright and facing -Y,
    found on its front and side pictures: the joints, and with ``face`` the points of the
    face too. MediaPipe's body model places the face's points loosely, and on a bowed head
    under a cut hat it put them 5 cm too high, which turned the fitted head back; the fit
    finds the face better from the scan alone. ``pictures``, when given, receives the two PIL
    pictures with the found points drawn on them, by view.
    """
    from PIL import ImageDraw

    from opensculptboy.posing.landmarks import BODY, detect
    from opensculptboy.render.dressed import View

    V = np.asarray(vertices, dtype=np.float64)
    T = np.asarray(triangles)
    found = {}
    for label, view in (("front", View(0.0)), ("side", View(-90.0))):
        image, frame = _picture(V, T, view)
        marks = detect(np.asarray(image), hands=False, face=False)
        points = {}
        for k, name in enumerate(BODY):
            if name not in NAMES or marks.image is None:
                continue
            seen = 1.0 if marks.visibility is None else float(marks.visibility[k])
            if seen < VISIBILITY:
                continue
            points[name] = _unproject(marks.image[k], frame)
        found[label] = points
        if pictures is not None:
            paint = ImageDraw.Draw(image)
            for k, name in enumerate(BODY):
                if name in points:
                    x, y = marks.image[k][:2]
                    paint.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(230, 40, 40))
            pictures[label] = image
    front, side = found["front"], found["side"]
    # the side picture's left and right: the order whose heights agree with the front's
    for part in PAIRS:
        left, right = f"left_{part}", f"right_{part}"
        if not all(k in front and k in side for k in (left, right)):
            continue
        keep = abs(front[left][1] - side[left][1]) + abs(
            front[right][1] - side[right][1]
        )
        swap = abs(front[left][1] - side[right][1]) + abs(
            front[right][1] - side[left][1]
        )
        if swap < keep:
            side[left], side[right] = side[right], side[left]
    out = {}
    for name, key in NAMES.items():
        if name not in front or (name in FACE and not face):
            continue
        x, z = front[name]
        if name in FACE or name not in side:
            # depth from the surface under the pixel: the front-most point of the face, the
            # middle of a limb; a joint that the side picture hides (the far leg) or that
            # MediaPipe put beside the limb takes the nearest vertices within 5 cm
            gap = np.linalg.norm(V[:, [0, 2]] - [x, z], axis=1)
            near = gap < max(0.01, float(np.partition(gap, 29)[29]))
            if gap.min() > 0.05:
                continue
            y = V[near, 1].min() if name in FACE else V[near, 1].mean()
        else:
            y, z_side = side[name]
            z = 0.5 * (z + z_side)
        out[key] = np.array([x, y, z])
    return out


# the points of MediaPipe's face mesh (its 468 points; the person's own left and right) that
# give the contract's landmarks of the face
FACE_MESH = {
    "eye_outer_l": 263,
    "eye_inner_l": 362,
    "eye_outer_r": 33,
    "eye_inner_r": 133,
    "nose_tip": 1,
    "mouth_corner_l": 291,
    "mouth_corner_r": 61,
    "chin": 152,
}
# the head's pictures, tried in turn: the scan's triangles within this distance (metres) of
# its top, and within this distance across of the top's centre (0: all; a head cut low under a
# hat showed its shoulders, and its face was too small to find)
HEAD_CROPS = ((0.32, 0.0), (0.24, 0.13), (0.2, 0.1))
# the views tried in turn (yaw, pitch: from below when positive, degrees): a bowed head under
# a hat shows its face from below
FACE_VIEWS = ((0.0, 0.0), (0.0, 20.0), (0.0, 35.0), (20.0, 15.0), (-20.0, 15.0))


def detect_face(vertices, triangles, pictures: dict | None = None) -> dict:
    """
    The landmarks of the face (name -> point in Anny's frame) of a figure standing upright and
    facing -Y, found by MediaPipe's face landmarker on a picture of its head alone (the crops
    of :data:`HEAD_CROPS` around the top), where the face fills the picture: the
    eyes' corners, the nose's tip, the mouth's corners and the chin (:data:`FACE_MESH`). The
    views of :data:`FACE_VIEWS` are tried in turn. Each point takes the surface nearest the
    camera under its pixel. An empty dict when no view shows a face. ``pictures``, when
    given, receives the picture (with the points when found) as ``"face"``.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    from PIL import ImageDraw

    from opensculptboy.posing.landmarks import model_path
    from opensculptboy.render.dressed import View

    V = np.asarray(vertices, dtype=np.float64)
    T = np.asarray(triangles)
    top = V[:, 2].max()
    centre = V[V[:, 2] > top - 0.06, :2].mean(0)
    crops = []
    for reach, width in HEAD_CROPS:
        keep = (V[T][:, :, 2] > top - reach).all(1)
        if width:
            keep &= (np.linalg.norm(V[T][:, :, :2] - centre, axis=2) < width).all(1)
        used = np.unique(T[keep])
        if len(used) < 3:
            continue
        remap = np.full(len(V), -1)
        remap[used] = np.arange(len(used))
        crops.append((V[used], remap[T[keep]]))
    if not crops:
        return {}
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path("face"))),
        num_faces=1,
    )
    tries = [(c, v) for c in crops for v in FACE_VIEWS]
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for (head, head_triangles), (yaw, pitch) in tries:
            view = View(yaw, pitch)
            image, frame = _picture(head, head_triangles, view)
            if pictures is not None and "face" not in pictures:
                pictures["face"] = image
            pixels = np.ascontiguousarray(np.asarray(image.convert("RGB")))
            found = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=pixels)
            )
            if found.face_landmarks:
                break
        else:
            return {}
    h, w = pixels.shape[:2]
    mesh = found.face_landmarks[0]
    picture, depth = view.project(head)
    out = {}
    paint = ImageDraw.Draw(image) if pictures is not None else None
    for name, k in FACE_MESH.items():
        px = (mesh[k].x * w, mesh[k].y * h)
        at = np.array(_unproject(px, frame))
        gap = np.linalg.norm(picture - at, axis=1)
        if gap.min() > 0.01:
            continue
        near = np.nonzero(gap < max(0.002, float(np.partition(gap, 9)[9])))[0]
        out[name] = head[near[np.argmin(depth[near])]]
        if paint is not None:
            paint.ellipse(
                [px[0] - 3, px[1] - 3, px[0] + 3, px[1] + 3], fill=(40, 200, 90)
            )
    if pictures is not None:
        pictures["face"] = image
    return out

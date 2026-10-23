# Pose from a picture

`opensculptboy.pose_from_image` poses a character as the figure in a picture: the body, the head, the fingers and the facial expression. It reads the picture with Google's [MediaPipe](https://ai.google.dev/edge/mediapipe) pose, hand and face landmarkers, which download on first use into `ANNY_CACHE_DIR/opensculptboy/models`.

```bash
uv sync --extra pose
opensculptboy pose photo.jpg --card pose.json --glb pose.glb --svg pose.svg --outline pose_line.svg --png pose.png
opensculptboy pose photo.jpg --png pose.png --view three-quarter        # the picture's camera (the default), front or three-quarter
opensculptboy pose photo.jpg --png pose.png --head-turn 20 --head-up 7  # turn the head further (degrees; the figure's right and up)
```

```python
from opensculptboy import Character, export_glb, pose_from_image

character = pose_from_image("photo.jpg", Character(name="ada", phenotype={"height": 0.7}))
export_glb("ada.glb", character)  # the pose becomes an animation named "pose"
```

The [character card](character-card.md) stores the pose in `pose`: a quaternion for each bone that moves (Anny's `local-ref` parameters). The face's scores become `facial_actions`, because MediaPipe and Anny use the same 52 ARKit names. `opensculptboy.render.flat` draws any posed mesh as a silhouette SVG, an outline SVG (the contour and the lines where a limb passes in front of the body) or a flat-shaded PNG with a transparent background.

## How it works

Each bone turns to follow its landmarks, with hinges at the elbows, knees and finger joints. A short optimisation then turns the torso, the collarbones and the limbs until the model's landmarks meet the picture's. A single picture fixes each limb's direction across the picture well and its depth less well: on drawn test poses, every limb comes back within 8° in the picture plane.

Pictures can be noisy, and pictures made by AI models can draw anatomy that no body has. The head takes its turn from the face mesh when MediaPipe finds a face, and otherwise from the head points that agree with each other. A hand with impossible proportions keeps its rest pose, and the optimisation limits how far any single bad landmark can pull the pose. When a drawn face leaves the head's turn unclear, three angles correct it by hand: `--head-turn` (toward the figure's right), `--head-up` and `--head-tilt` (toward the right shoulder), or `head=dict(turn=..., up=..., tilt=...)` in Python.

The [web viewer](viewer.md) offers the same step under **From a picture**, for the body, the head and the fingers.

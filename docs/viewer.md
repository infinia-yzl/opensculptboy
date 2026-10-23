# Web viewer

`viewer/dist/anny_viewer.html` shapes and poses a character in a browser. It is one HTML file that draws with WebGL2, with skin, eye and hair shading. Open it in a browser; it needs no server.

## Using it

- The **Character** segment on the left holds presets, the body sliders, the 103 face shapes with a **Random face** button, 25 hairstyles, and the skin and eye colours. The `age` parameter appears as **Form**, with the ends *Compact* and *Elongated*.
- The **Stage** segment on the right holds the poses and clips, the rig and skin-weight views, and the lighting and quality settings.
- The Pose section offers **From a picture**: it poses the body, the head and the fingers as the figure in a photo or a drawing ([Pose from a picture](pose-from-picture.md) describes the method). The **Head turn**, **Head up** and **Head tilt** sliders correct the head afterwards. The picture stays in the browser, and MediaPipe's code and models load from their CDNs the first time.
- The camera orbits with a drag and zooms with the wheel. The `?` key lists every shortcut.
- The **Performance** card (the `` ` `` key or `?stats=1`) shows the frame times and runs a 10-second benchmark.
- `?shot` hides the whole interface, for screenshots.

The Pose section saves the pose shown as a character card ([Character card](character-card.md)) for `opensculptboy export`, or as a silhouette SVG.

## Building the page

The build needs node 22 or later:

```bash
uv sync --extra viewer
uv run python -m anny.viewer build      # the model data and the page
node viewer/build.mjs --reuse-data      # the page alone, after a change to viewer/src
node viewer/build.mjs --parts <dir>     # the model data in separate files, for hosts that limit the size of a file
```

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

## Wearables

The **Wearables** section (the `W` key) dresses the character in the outfits of a character card ([Wearables](wearables.md)). **Open files** or **Open folder** picks the card and its wearable packages, and a drop of the files or their folder anywhere on the page does the same. The page reads the files in the browser: it sends them nowhere and keeps them only until it closes. The published page holds no wearables.

- The card's sliders, face shapes and level-0 detail offsets shape the body. The page leaves out the higher levels of the detail layer and the card's local changes.
- The section lists the outfits and every piece with a switch. A piece of another outfit takes the slot of the worn outfit's piece, as `--show` does in an export. The hides rules hide groups of other pieces while their piece shows.
- **Hide covered skin** leaves the body triangles under the shown pieces out of the picture, as an export does by default.
- The pieces are meshes skinned by the page's rig, so they follow the poses and clips. A slider edit moves them at once (the bindings and the skin weights); the push-out and the covered skin follow when the sliders rest for a quarter of a second.
- A rigid piece turns with its bone's skin: the page has the bone heads of every body and none of Anny's rest rotations, so the rotation comes from the vertices that the bone skins (within 3 mm of Anny's on the test bodies).
- The dressing (`viewer/src/wearables/dress.ts`, `geometry.ts`) repeats `opensculptboy.wearables` without the DOM and three.js, so a worker can run it. `test.test_viewer_wearables` checks it against Python with synthetic outfits in node, and `test.test_viewer_wearables_page` checks the section in the built page (local only).
- The skin triangles that garments bind to come from the page's body quads with the codes of `viewer/src/wearables/skin_split.ts`, written by `python -m opensculptboy.wearables.viewer`.

## Building the page

The build needs node 22 or later:

```bash
uv sync --extra viewer
uv run python -m anny.viewer build      # the model data and the page
node viewer/build.mjs --reuse-data      # the page alone, after a change to viewer/src
node viewer/build.mjs --parts <dir>     # the model data in separate files, for hosts that limit the size of a file
```

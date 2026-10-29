// OpenSculptBoy
// Apache License, Version 2.0
//
// MediaPipe's pose and hand landmarkers in the page, for the pose from a picture. MediaPipe's code is bundled into the
// page, and its WebAssembly and its two models sit in the page's data block (build.mjs, mediapipe_assets.ts), so the
// page loads nothing from another host and works offline.

import { HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import { mediapipeAssets, runLoader } from './mediapipe_assets.ts';
import { fromMediapipe, type Landmarks } from './pose_from_image.ts';

interface Point { x: number; y: number; z: number; visibility?: number }
interface Found { landmarks: Point[][]; worldLandmarks: Point[][] }
interface Landmarker { detect(image: TexImageSource): Found }

let loading: Promise<{ pose: Landmarker; hand: Landmarker }> | null = null;
function landmarkers() {
  if (!loading) {
    loading = (async () => {
      const assets = await mediapipeAssets();
      // MediaPipe runs its loader script before each landmarker and clears it after; the page runs the loader itself
      // (an empty wasmLoaderPath), one landmarker at a time
      const files = { wasmLoaderPath: '', wasmBinaryPath: assets.wasm };
      const make = async (cls: any, model: Uint8Array, extra: object) => {
        await runLoader(assets.loader);
        return cls.createFromOptions(files, { baseOptions: { modelAssetBuffer: model, delegate: 'CPU' }, runningMode: 'IMAGE', ...extra });
      };
      const pose = await make(PoseLandmarker, assets.models.pose, { numPoses: 1 });
      const hand = await make(HandLandmarker, assets.models.hand, { numHands: 2 });
      return { pose, hand };
    })();
    loading.catch(() => { loading = null; });
  }
  return loading;
}

// the landmarks of the most prominent figure in a picture (anny's frame), and which hands MediaPipe found
export async function detect(image: TexImageSource & { width: number; height: number }): Promise<{ landmarks: Landmarks; hands: string[] }> {
  const { pose, hand } = await landmarkers();
  const found = pose.detect(image);
  if (!found.worldLandmarks.length) throw new Error('No figure found in the picture.');
  const img = found.landmarks[0];
  const landmarks: Landmarks = {
    body: fromMediapipe(found.worldLandmarks[0]), hands: {},
    image: img.map((q) => [q.x * image.width, q.y * image.height]), visibility: img.map((q) => q.visibility ?? 1),
  };
  // MediaPipe names a hand's side as seen in a mirror; the nearest wrist in the picture is surer, and a hand belongs
  // to a wrist only within most of a forearm's length of it
  const wrists: Record<string, Point> = { '.L': img[15], '.R': img[16] }, elbows: Record<string, Point> = { '.L': img[13], '.R': img[14] };
  const px = (a: Point, b: Point) => Math.hypot((a.x - b.x) * image.width, (a.y - b.y) * image.height);
  const hands = hand.detect(image), taken = new Set<string>();
  hands.landmarks.forEach((points, i) => {
    const at = points[0];
    let side = '', best = Infinity;
    for (const s of ['.L', '.R']) {
      if (taken.has(s)) continue;
      const d = Math.hypot((wrists[s].x - at.x) * image.width, (wrists[s].y - at.y) * image.height);
      if (d < best) { best = d; side = s; }
    }
    if (!side || best > 0.6 * px(wrists[side], elbows[side])) return;
    taken.add(side);
    landmarks.hands[side] = fromMediapipe(hands.worldLandmarks[i]);
  });
  return { landmarks, hands: [...taken] };
}

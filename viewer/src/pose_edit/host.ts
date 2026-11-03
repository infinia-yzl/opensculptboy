// OpenSculptBoy
// Apache License, Version 2.0
//
// What the pose editor needs from the renderer (main.ts): the rig and the coarse body in the page's frame, the pose
// as the page's quaternions, a way to show a new pose at once, the open card with its files, and a hook to draw over
// the picture. main.ts implements it; the editor (editor.ts) holds no reference to main.ts's globals.

import type * as THREE from 'three';

export interface PoseHost {
  THREE: typeof THREE;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  canvas: HTMLCanvasElement;
  ready(): boolean;
  // the rig: names, parents, each bone's rest head (page frame, the current body), the root bone
  bones(): string[];
  parents(): number[];
  restHeads(): number[][];
  root(): number;
  // the page's frame of Anny's card frame: page = scale * rotation * x + offset (+ the body's lift on y)
  frame(): { scale: number; rotation: number[]; offset: number[] };
  lift(): number;
  // the root's offset (page frame) and the figure's turn (the turntable's holder), applied after the pose
  rootOffset(): number[];
  holder(): THREE.Matrix4;
  // the pose: a quaternion (x, y, z, w) per bone in the page's axes, each relative to its parent
  pose(): Float32Array;
  // show a pose at once (an edit): the clip stops, the root stays where it stands, and the picture refines again
  setPose(q: Float32Array, label?: string): void;
  // the coarse body (Anny's level-0 vertices of the current sliders, page frame), its skin (8 bone indices then 8
  // weights out of 255 per vertex), its quads, and the map from its rows to Anny's rows (card_coarse)
  body(): { coarse: Float32Array; skin: Uint8Array; nBody: number; quads: Uint32Array };
  annyRows(): Uint32Array | null;
  // the open card (the wardrobe's), its file as opened, and its files
  card(): { card: any; source: any; bytes(rel: string): Promise<Uint8Array> } | null;
  // a drawer over the picture (after the display pass, with the overlay's render target), and a redraw of the display
  // and the overlays without starting the refinement again
  setOverlay(draw: ((target: THREE.WebGLRenderTarget) => void) | null): void;
  overlayTarget(): THREE.WebGLRenderTarget;
  sceneDepth(): { texture: THREE.Texture | null; uvScale: THREE.Vector2 };
  redraw(): void;
  // a callback on every frame of the page (the points follow the camera)
  onFrame(fn: (() => void) | null): void;
  // the props worn (the wardrobe's shown pieces of the slot prop), as triangle meshes in the page's frame
  props(): { name: string; positions: Float32Array; indices: Uint32Array }[];
  // the pose shown as a card's pose (opensculptboy.Character.pose: each bone's turn relative to its parent in Anny's frame)
  cardPose(): { bones: Record<string, number[]>; root: number[] };
  // a toast
  toast(text: string): void;
}

let current: PoseHost | null = null;
export function registerPoseHost(h: PoseHost) { current = h; }
export function poseHost(): PoseHost | null { return current; }

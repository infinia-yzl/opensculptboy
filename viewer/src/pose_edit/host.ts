// OpenSculptBoy
// Apache License, Version 2.0
//
// What the pose editor needs from the renderer (main.ts): the rig and the coarse body in the page's frame, the pose
// as the page's quaternions, a way to show a new pose at once, the open card with its files, and a hook to draw over
// the picture. main.ts implements it; the editor (editor.ts) holds no reference to main.ts's globals.

import type * as THREE from 'three';
import type { Place } from '../wearables/place.ts';

// a prop worn as a rigid piece, as the studio moves it: its reference and label, the bone its package binds it to, its
// place on the card (or null), the bone that carries it now, and its rest vertices (the page's frame, with its place)
// and triangles
export interface PropRig { ref: string; label: string; made: string | null; place: Place | null; carrier: string; vertices: Float64Array; triangles: Uint32Array }

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
  // each bone's turn C = reference rest^T (bones * 9, Anny's axes) for the body shown (pose_reference.ts), or null
  restTurns(): ArrayLike<number> | null;
  root(): number;
  // the page's frame of Anny's card frame: page = scale * rotation * x + offset (+ the body's lift on y)
  frame(): { scale: number; rotation: number[]; offset: number[] };
  lift(): number;
  // the root's offset (page frame) and the figure's turn (the turntable's holder), applied after the pose
  rootOffset(): number[];
  holder(): THREE.Matrix4;
  // the pose: a quaternion (x, y, z, w) per bone in the page's axes, each relative to its parent
  pose(): Float32Array;
  // show a pose at once (an edit): the clip stops, and the picture refines again. The figure stands on the floor that
  // this pose finds, as a pose of the library stands (the same pose, the same height); with hold, while a drag or a
  // modal grab or turn lasts, the floor stays where it was
  setPose(q: Float32Array, label?: string, hold?: boolean): void;
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
  // the props worn (the wardrobe's shown pieces of the slot prop), as triangle meshes in the page's frame, with the bone
  // that carries a rigid prop
  props(): { name: string; positions: Float32Array; indices: Uint32Array; bone?: string }[];
  // the shown props that are rigid pieces, which the studio moves (PropRig)
  propRigs(): PropRig[];
  // the skinning matrix of a bone in the pose shown (rest to posed, the page's frame before the holder's turn, row-major
  // 4 x 4), or null for a bone the rig lacks
  skinMatrix(bone: string): Float64Array | null;
  // a prop's place on the card (null: the place its package binds it to), shown at once
  setPropPlace(ref: string, place: Place | null): boolean;
  // the page's library of poses (the 50 built-in poses: name, label, group and credit), and one of them as the page's
  // quaternions (its first frame), or null
  builtinPoses(): { name: string; label: string; group?: string; credit?: string }[];
  builtinPose(name: string): Float32Array | null;
  // the pose shown as a card's pose (opensculptboy.Character.pose: each bone's turn relative to its parent in Anny's frame)
  cardPose(): { bones: Record<string, number[]>; root: number[] };
  // the first point of the posed body under a ray (page frame), or null
  bodyRay(origin: THREE.Vector3, dir: THREE.Vector3): { point: THREE.Vector3; distance: number } | null;
  // a toast
  toast(text: string): void;
}

let current: PoseHost | null = null;
export function registerPoseHost(h: PoseHost) { current = h; }
export function poseHost(): PoseHost | null { return current; }

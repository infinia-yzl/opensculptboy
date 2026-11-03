// OpenSculptBoy
// Apache License, Version 2.0
//
// The studio's camera, as the rest of the page sees it (DESIGN-0014, "The camera model"): framing a part, the pivot,
// the view history, isolate and X-ray, and a callback on every change. The navigation (ui/camera_nav.ts) implements
// it and registers it with `registerCamera`; until then every call is a no-op, so a part of the page can use the
// interface before the camera exists (the same pattern as face/api.ts).

import type * as THREE from 'three';

export type Vec3Like = { x: number; y: number; z: number };
export type BoxLike = { min: Vec3Like; max: Vec3Like };

export interface FrameOpts {
  keepAngle?: boolean;      // keep the current angle even when it does not show the part (default: keep it when it shows the part, else turn to the side that faces the camera most)
  pivot?: Vec3Like;         // the pivot after the framing (default: the box's centre)
  margin?: number;          // the share of the free space round the part (default 0.12)
  ms?: number;              // the flight (default 300 ms; 0 jumps)
  history?: boolean;        // add the view before it to the history (default true)
  label?: string;           // a name for the framing ('Right hand'), shown in the status line and the toast
}

export interface CameraState {
  yaw: number;              // degrees round the vertical axis from the figure's front (+z), towards the figure's left
  tilt: number;             // degrees above the horizon
  distance: number;         // metres from the pivot
  pivot: [number, number, number];
  position: [number, number, number];
  back: number;             // views in the history before this one
  forward: number;          // views after it
  xray: boolean;
  isolated: boolean;
  cutAway: boolean;         // the body is clipped round the camera, to show the pivot
  flying: boolean;
}

export interface StudioCamera {
  // fit a box (or points) into the free space of the view, turning round its centre
  frameBounds(box: BoxLike | Vec3Like[], opts?: FrameOpts): void;
  // move the pivot to a point without moving the camera's angle (with `centre`, the camera flies so the pivot is centred)
  setPivot(point: Vec3Like, opts?: { centre?: boolean; ms?: number; history?: boolean }): void;
  pivot(): THREE.Vector3;
  // the view history (50 views): every flight, framing and reset adds the view before it
  pushView(): void;
  previousView(): boolean;
  nextView(): boolean;
  // the rest of the scene faint and not pickable; null shows everything again
  isolate(objects: THREE.Object3D[] | null): void;
  isolated(): THREE.Object3D[] | null;
  // the body and the scan see-through
  xray(on: boolean): void;
  xrayOn(): boolean;
  // a ray from the camera through a point of the canvas (client pixels)
  rayAt(clientX: number, clientY: number): THREE.Ray;
  // a world point on the canvas (client pixels), and its depth from the camera
  toScreen(p: Vec3Like): { x: number; y: number; depth: number };
  state(): CameraState;
  // called after every change of the view (a turn, a move, a zoom, a flight's frame) and of the display (X-ray, isolate)
  onChange(fn: () => void): () => void;
}

const noop = () => {};
const stub: StudioCamera = {
  frameBounds: noop, setPivot: noop,
  pivot: () => ({ x: 0, y: 0, z: 0 } as any),
  pushView: noop, previousView: () => false, nextView: () => false,
  isolate: noop, isolated: () => null, xray: noop, xrayOn: () => false,
  rayAt: () => ({ origin: { x: 0, y: 0, z: 0 }, direction: { x: 0, y: 0, z: -1 } } as any),
  toScreen: () => ({ x: 0, y: 0, depth: 0 }),
  state: () => ({ yaw: 0, tilt: 0, distance: 1, pivot: [0, 0, 0], position: [0, 0, 1], back: 0, forward: 0, xray: false, isolated: false, cutAway: false, flying: false }),
  onChange: () => noop,
};

let current: StudioCamera = stub;
export function registerCamera(c: StudioCamera): void { current = c; }
export function cam(): StudioCamera { return current; }
export function cameraReady(): boolean { return current !== stub; }

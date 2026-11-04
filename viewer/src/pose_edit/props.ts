// OpenSculptBoy
// Apache License, Version 2.0
//
// The studio's props, moved by hand (DESIGN-0013, "Props"): the parts of the prop tool that need no DOM, so node runs
// them (viewer/test/props.mjs). A prop is a rigid piece of the slot `prop` (a statue's sword, its base): the editor
// (editor.ts) selects it in the view or in the panel's list, moves it with the Move and Turn tools, G and R, the arrows
// and typed values, and changes the bone that holds it; the place itself is wearables/place.ts. This file gives the
// choices of "Held by", the place as the panel's numbers (millimetres and degrees in the page's axes: X across, Y up,
// Z forward) and back, the moves of a drag (along the view's plane, along an axis, round an axis), and the hit tests
// of the move handle and the turn rings on screen.

import { axisAngle, eulerOf, fromEuler, IDENTITY_Q, m4, mul4, quaternionOf, rotationMatrix, type M4, type Place, type Similarity } from '../wearables/place.ts';

export type V3 = [number, number, number];
export interface HeldBy { bone: string; label: string }

// the bone of the other side ('metacarpal4.R' -> 'metacarpal4.L'), or null
export function mirrorBone(bone: string): string | null {
  const m = /^(.*)\.(L|R)$/.exec(bone);
  return m ? `${m[1]}.${m[2] === 'L' ? 'R' : 'L'}` : null;
}

// the choices of "Held by": the bone the prop was made on, the same bone of the other hand, the wrists, the head, the
// hips and the root (the prop stays where it stands in the world and moves with the figure only)
export function heldByChoices(made: string | null, bones: string[], name: (bone: string) => string): HeldBy[] {
  const list: string[] = [];
  const add = (b: string | null) => { if (b && bones.includes(b) && !list.includes(b)) list.push(b); };
  add(made); add(made ? mirrorBone(made) : null);
  for (const b of ['wrist.R', 'wrist.L', 'head', 'pelvis', 'spine01', 'root']) add(b);
  return list.map((b) => ({ bone: b, label: (b === 'root' ? 'The ground (stays in place)' : name(b)) + (b === made ? ' (as made)' : '') }));
}

// a place as the panel's numbers: its translation in millimetres along the page's axes (unscaled), and its turn as angles
// about the page's X, Y and Z (degrees), in the figure's rest frame
export function pageNumbers(p: Place | null, F: Similarity): { mm: V3; deg: V3 } {
  if (!p) return { mm: [0, 0, 0], deg: [0, 0, 0] };
  const R = F.rotation, t = p.translation, Q = rotationMatrix(p.rotation || IDENTITY_Q);
  const mm = [0, 1, 2].map((r) => 1000 * (R[r * 3] * t[0] + R[r * 3 + 1] * t[1] + R[r * 3 + 2] * t[2])) as V3;
  // the turn in the page's axes: R Q R^T
  const RQ = [0, 1, 2].flatMap((i) => [0, 1, 2].map((j) => R[i * 3] * Q[j] + R[i * 3 + 1] * Q[3 + j] + R[i * 3 + 2] * Q[6 + j]));
  const QP = [0, 1, 2].flatMap((i) => [0, 1, 2].map((j) => RQ[i * 3] * R[j * 3] + RQ[i * 3 + 1] * R[j * 3 + 1] + RQ[i * 3 + 2] * R[j * 3 + 2]));
  return { mm, deg: eulerOf(quaternionOf(QP)) as V3 };
}
// the place of the panel's numbers
export function placeFromNumbers(bone: string, mm: ArrayLike<number>, deg: ArrayLike<number>, F: Similarity): Place {
  const R = F.rotation, QP = rotationMatrix(fromEuler(deg));
  // Q = R^T QP R and t = R^T mm / 1000
  const RtQ = [0, 1, 2].flatMap((i) => [0, 1, 2].map((j) => R[i] * QP[j] + R[3 + i] * QP[3 + j] + R[6 + i] * QP[6 + j]));
  const Q = [0, 1, 2].flatMap((i) => [0, 1, 2].map((j) => RtQ[i * 3] * R[j] + RtQ[i * 3 + 1] * R[3 + j] + RtQ[i * 3 + 2] * R[6 + j]));
  const r7 = (v: number) => Math.round(v * 1e7) / 1e7;
  return { bone, rotation: quaternionOf(Q).map(r7), translation: [0, 1, 2].map((c) => r7((R[c] * mm[0] + R[3 + c] * mm[1] + R[6 + c] * mm[2]) / 1000)) };
}
// whether a place moves nothing on the bone it was made on: no place is needed
export function isRest(p: Place | null, made: string | null): boolean {
  if (!p) return true;
  const t = p.translation, q = p.rotation;
  return p.bone === made && Math.hypot(t[0], t[1], t[2]) < 1e-7 && Math.abs(Math.abs(q[3]) - 1) < 1e-10;
}

// ------------------------------------------------------------------ the moves of a drag (row-major, the world)
export function translation(v: ArrayLike<number>): M4 { return m4(undefined, v); }
// a turn of `rad` about `axis` through `pivot`
export function turnAbout(pivot: ArrayLike<number>, axis: ArrayLike<number>, rad: number): M4 {
  return mul4(mul4(translation(pivot), m4(axisAngle(axis, rad))), translation([-pivot[0], -pivot[1], -pivot[2]]));
}
// where a ray meets the plane through `pivot` with normal `n`, or null
export function onPlane(o: ArrayLike<number>, d: ArrayLike<number>, pivot: ArrayLike<number>, n: ArrayLike<number>): V3 | null {
  const dn = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
  if (Math.abs(dn) < 1e-9) return null;
  const t = ((pivot[0] - o[0]) * n[0] + (pivot[1] - o[1]) * n[1] + (pivot[2] - o[2]) * n[2]) / dn;
  return [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
}
// the distance (metres of the world) along an axis that a pointer offset (dx, dy pixels) gives, from the axis's picture
// on screen: `sx`, `sy` the screen offset of 1 world unit along it
export function alongAxis(dx: number, dy: number, sx: number, sy: number): number {
  const l2 = sx * sx + sy * sy;
  return l2 < 1e-9 ? 0 : (dx * sx + dy * sy) / l2;
}
// the angle swept round a screen centre from one pointer place to the next, unwrapped
export function sweep(cx: number, cy: number, x0: number, y0: number, x1: number, y1: number): number {
  let d = Math.atan2(y1 - cy, x1 - cx) - Math.atan2(y0 - cy, x0 - cx);
  if (d > Math.PI) d -= 2 * Math.PI;
  if (d < -Math.PI) d += 2 * Math.PI;
  return d;
}

// ------------------------------------------------------------------ the handles on screen
export interface Screen { x: number; y: number }
// the distance from a point to a polyline on screen
export function toPolyline(p: Screen, pts: [number, number][]): number {
  let best = Infinity;
  for (let i = 1; i < pts.length; i++) {
    const [ax, ay] = pts[i - 1], [bx, by] = pts[i], dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy || 1;
    const t = Math.min(1, Math.max(0, ((p.x - ax) * dx + (p.y - ay) * dy) / l2));
    best = Math.min(best, Math.hypot(p.x - ax - t * dx, p.y - ay - t * dy));
  }
  return best;
}
// the part of the prop's handles under the pointer: an arrow of the move handle (0, 1, 2), its centre ('free'), a ring of
// the turn handle (0, 1, 2) or its outer ring ('view'); null for none
export type HandlePart = { kind: 'axis'; k: number } | { kind: 'free' } | { kind: 'ring'; k: number } | { kind: 'view' };
export function hitHandles(p: Screen, h: { centre: Screen; arrows?: [number, number][][]; rings?: [number, number][][]; outer?: number }, reach: number): HandlePart | null {
  if (h.arrows) {
    if (Math.hypot(p.x - h.centre.x, p.y - h.centre.y) <= reach + 4) return { kind: 'free' };
    let best: HandlePart | null = null, bd = reach;
    h.arrows.forEach((a, k) => { const d = toPolyline(p, a); if (d < bd) { bd = d; best = { kind: 'axis', k }; } });
    return best;
  }
  if (h.rings) {
    let best: HandlePart | null = null, bd = reach;
    h.rings.forEach((r, k) => { const d = toPolyline(p, r); if (d < bd) { bd = d; best = { kind: 'ring', k }; } });
    if (!best && h.outer !== undefined && Math.abs(Math.hypot(p.x - h.centre.x, p.y - h.centre.y) - h.outer) < reach) return { kind: 'view' };
    return best;
  }
  return null;
}

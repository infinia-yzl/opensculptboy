// OpenSculptBoy
// Apache License, Version 2.0
//
// The hand controls of the pose editor (DESIGN-0013, "Hands"; artboard 2b), on the joint coordinates of joints.ts:
// - a finger's Curl bends its three joints together in the ratio of the library fist (mh_fight, as
//   opensculptboy.posing.contact.FIST_BENDS has it), 0 % at the reference pose and 100 % at the fist;
// - a finger's Spread turns its knuckle sideways, away from the middle finger (each finger's side axis is told by
//   the rig, so the left hand mirrors the right);
// - the whole hand's Fist, Spread and Thumb set the fingers together, and Relax brings the hand back to a rest curl;
// - presets (open, relaxed, fist, grip round a handle, pointing, pinch, holding a ball), and the user's own;
// - the gesture pad blends four presets by a point in a square (bilinear weights);
// - mirroring takes a pose to the other side with the reflection x -> -x (Anny's x is the figure's left), which is
//   S D S for each bone's turn; copying a hand to the other hand does the same for the hand's bones alone.

import { FIST_BENDS } from './contact.ts';
import { Joints } from './joints.ts';
import { splitSide, transforms, type RigSpace } from './problem.ts';
import { capitalised, fingersPhrase } from '../ui/words.ts';

export const FINGERS = ['thumb', 'index', 'middle', 'ring', 'little'] as const;
export const FINGER_LABELS = ['Thumb', 'Index finger', 'Middle finger', 'Ring finger', 'Little finger'];
// Grip's warning for the fingers that stay inside a prop: "The index and middle fingers stay inside the sword, ..."
export function gripWarning(fingers: string[], prop: string): string {
  const f = fingersPhrase(fingers);
  return `${capitalised(f.text)} ${f.plural ? 'stay' : 'stays'} inside the ${prop}, at ${f.plural ? 'their' : 'its'} present curl and straight alike.`;
}
const SPREAD_DEG = [0, 14, 4, 9, 16];   // a finger's knuckle at 100 % spread
const FAN = [0, 1, 0.3, -0.5, -1];       // which way each finger leaves the middle of the hand (+: towards the thumb)

export interface HandShape { curl: number[]; spread: number }   // curl per finger (thumb first), 0 to 1; spread -1 to 1
export const PRESETS: { id: string; label: string; shape: HandShape }[] = [
  { id: 'open', label: 'Open', shape: { curl: [0, -0.08, -0.08, -0.08, -0.08], spread: 0.6 } },
  { id: 'relaxed', label: 'Relaxed', shape: { curl: [0.15, 0.18, 0.24, 0.3, 0.36], spread: 0.1 } },
  { id: 'fist', label: 'Fist', shape: { curl: [0.85, 1, 1, 1, 1], spread: 0 } },
  { id: 'grip', label: 'Handle grip', shape: { curl: [0.6, 0.82, 0.9, 0.95, 1], spread: 0 } },
  { id: 'pointing', label: 'Pointing', shape: { curl: [0.7, 0, 1, 1, 1], spread: 0 } },
  { id: 'pinch', label: 'Pinch', shape: { curl: [0.55, 0.6, 0.35, 0.4, 0.45], spread: 0.1 } },
  { id: 'ball', label: 'Ball', shape: { curl: [0.35, 0.42, 0.42, 0.42, 0.42], spread: 0.55 } },
];
export const RELAXED = PRESETS[1].shape;

const DEG = Math.PI / 180;

export class HandControls {
  readonly joints: Joints;
  private sideSign = new Map<string, number>();
  constructor(joints: Joints) { this.joints = joints; }

  bones(side: 'L' | 'R', k: number): number[] { return [1, 2, 3].map((j) => this.joints.bone(`finger${k + 1}-${j}.${side}`)!).filter((b) => b !== undefined); }
  handBones(side: 'L' | 'R'): number[] {
    return this.joints.rig.labels.map((l, i) => [l, i] as [string, number]).filter(([l]) => l.endsWith('.' + side) && /^(finger|metacarpal)/.test(l)).map(([, i]) => i);
  }

  // a finger's curl (0 at the reference pose, 1 at the fist), from its joints' bends
  curl(R: number[][], side: 'L' | 'R', k: number): number {
    const fist = FIST_BENDS[`finger${k + 1}`];
    const v = this.bones(side, k).map((b, j) => this.joints.values(b, R)!.value[0] / fist[j]);
    return v.reduce((s, x) => s + x, 0) / v.length;
  }
  setCurl(R: number[][], side: 'L' | 'R', k: number, c: number, limits = true) {
    const fist = FIST_BENDS[`finger${k + 1}`];
    this.bones(side, k).forEach((b, j) => this.joints.set(R, b, 0, c * fist[j], limits));
  }

  // which sign of a knuckle's side axis moves the finger towards the thumb, from the rig's heads
  private towardsThumb(side: 'L' | 'R', k: number): number {
    const key = side + k;
    let s = this.sideSign.get(key);
    if (s !== undefined) return s;
    const rig = this.joints.rig, b = this.bones(side, k)[0], tip = this.bones(side, k)[2];
    const thumb = this.joints.bone(`finger1-1.${side}`)!;
    const id = rig.labels.map(() => [1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const at = (R: number[][]) => { const K = transforms(rig, R); return [0, 1, 2].map((r) => K.R[tip * 9 + r * 3] * rig.heads[tip][0] + K.R[tip * 9 + r * 3 + 1] * rig.heads[tip][1] + K.R[tip * 9 + r * 3 + 2] * rig.heads[tip][2] + K.t[tip * 3 + r]); };
    const t0 = at(id), R1 = id.map((r) => r.slice());
    this.joints.set(R1, b, 1, 10, false);
    const t1 = at(R1), d = [t1[0] - t0[0], t1[1] - t0[1], t1[2] - t0[2]], h = rig.heads[thumb], toward = [h[0] - t0[0], h[1] - t0[1], h[2] - t0[2]];
    s = d[0] * toward[0] + d[1] * toward[1] + d[2] * toward[2] > 0 ? 1 : -1;
    this.sideSign.set(key, s);
    return s;
  }
  // a finger's spread (-1 together .. 1 apart), and the whole hand's
  spread(R: number[][], side: 'L' | 'R', k: number): number {
    if (k === 0 || !SPREAD_DEG[k]) return 0;
    const v = this.joints.values(this.bones(side, k)[0], R)!.value[1];
    return v * this.towardsThumb(side, k) / (FAN[k] * SPREAD_DEG[k]);
  }
  setSpread(R: number[][], side: 'L' | 'R', k: number, s: number, limits = true) {
    if (k === 0) return;
    this.joints.set(R, this.bones(side, k)[0], 1, s * FAN[k] * SPREAD_DEG[k] * this.towardsThumb(side, k), limits);
  }
  handSpread(R: number[][], side: 'L' | 'R'): number { return this.spread(R, side, 4); }
  setHandSpread(R: number[][], side: 'L' | 'R', s: number) { for (let k = 1; k < 5; k++) this.setSpread(R, side, k, s); }

  fist(R: number[][], side: 'L' | 'R'): number { return [1, 2, 3, 4].reduce((s, k) => s + this.curl(R, side, k), 0) / 4; }
  setFist(R: number[][], side: 'L' | 'R', f: number) { for (let k = 1; k < 5; k++) this.setCurl(R, side, k, f); }
  thumb(R: number[][], side: 'L' | 'R'): number { return this.curl(R, side, 0); }
  setThumb(R: number[][], side: 'L' | 'R', t: number) { this.setCurl(R, side, 0, t); }

  shape(R: number[][], side: 'L' | 'R'): HandShape { return { curl: FINGERS.map((_, k) => this.curl(R, side, k)), spread: this.handSpread(R, side) }; }
  apply(R: number[][], side: 'L' | 'R', s: HandShape) {
    FINGERS.forEach((_, k) => this.setCurl(R, side, k, s.curl[k]));
    this.setHandSpread(R, side, s.spread);
  }

  // the gesture pad: the shape at (u, v) in [0, 1]^2 between four corner shapes (top left, top right, bottom left,
  // bottom right)
  static blend(corners: HandShape[], u: number, v: number): HandShape {
    const w = [(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v];
    return { curl: FINGERS.map((_, k) => corners.reduce((s, c, i) => s + w[i] * c.curl[k], 0)), spread: corners.reduce((s, c, i) => s + w[i] * c.spread, 0) };
  }
}

// ------------------------------------------------------------------ mirror and copy
const reflect = (D: number[]) => [D[0], -D[1], -D[2], -D[3], D[4], D[5], -D[6], D[7], D[8]];   // S D S, S = diag(-1, 1, 1)

// the whole pose mirrored: each side takes the other side's turns reflected, and the middle bones are reflected
export function mirrorPose(rig: RigSpace, R: number[][], root: number): number[][] {
  const index = new Map(rig.labels.map((l, i) => [l, i]));
  return R.map((D, i) => {
    if (i === root) return reflect(D);
    const [base, side] = splitSide(rig.labels[i]);
    if (!side) return reflect(D);
    const other = index.get(base + (side === '.L' ? '.R' : '.L'));
    return other === undefined ? D.slice() : reflect(R[other]);
  });
}

// some bones' turns taken onto their mirror bones on the other side (a hand onto the other hand)
export function copyMirrored(rig: RigSpace, R: number[][], bones: number[]): number[] {
  const index = new Map(rig.labels.map((l, i) => [l, i])), changed: number[] = [];
  for (const b of bones) {
    const [base, side] = splitSide(rig.labels[b]);
    if (!side) continue;
    const other = index.get(base + (side === '.L' ? '.R' : '.L'));
    if (other === undefined) continue;
    R[other] = reflect(R[b]);
    changed.push(other);
  }
  return changed;
}

void DEG;

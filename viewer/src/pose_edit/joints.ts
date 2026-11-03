// OpenSculptBoy
// Apache License, Version 2.0
//
// The joints of the pose editor (DESIGN-0013, "Turning a joint"): each bone of the range table with its plain name,
// the joint it turns at, and its Bend, Side-Side and Twist in its own joint frame (opensculptboy.posing.point_solve:
// joint_frames, the coordinates c with D = exp(A (sign c))), in degrees, within the joint's range. A bone of a twist
// group (the upper arm, the forearm and the wrist, the thigh, the shin) turns its group together: its Twist is the
// group's twist, shared over the group's bones by their shares, so a far twist spreads along the limb and no single
// bone pinches the skin (the candy-wrapper pitfall).

import { coordinatesOf, jointFrames, rotationOf, splitSide, TABLE, type JointFrame, type RigSpace } from './problem.ts';

export const AXES = [{ key: 'bend', label: 'Bend', colour: '#2f6fd6' }, { key: 'side', label: 'Side-Side', colour: '#6a4fb0' }, { key: 'twist', label: 'Twist', colour: '#d4691c' }] as const;
const DEG = 180 / Math.PI;

const NAMES: Record<string, [string, string]> = {
  spine05: ['Lower back', 'Waist'], spine04: ['Back', 'Waist'], spine03: ['Chest', 'Back'], spine02: ['Chest', 'Chest'], spine01: ['Upper chest', 'Chest'],
  neck01: ['Neck', 'Neck'], neck02: ['Neck', 'Neck'], neck03: ['Neck', 'Neck'], head: ['Head', 'Neck'],
  clavicle: ['collarbone', 'Collarbone'], shoulder01: ['shoulder', 'Shoulder blade'], upperarm01: ['upper arm', 'Shoulder'], upperarm02: ['upper arm, lower half', 'Upper arm'],
  lowerarm01: ['forearm', 'Elbow'], lowerarm02: ['forearm, lower half', 'Forearm'], wrist: ['hand', 'Wrist'],
  metacarpal1: ['palm by the index finger', 'Palm'], metacarpal2: ['palm by the middle finger', 'Palm'], metacarpal3: ['palm by the ring finger', 'Palm'],
  metacarpal4: ['palm by the little finger', 'Palm'],
  pelvis: ['hip bone', 'Pelvis'], upperleg01: ['thigh', 'Hip'], upperleg02: ['thigh, lower half', 'Thigh'], lowerleg01: ['shin', 'Knee'], lowerleg02: ['shin, lower half', 'Shin'],
  foot: ['foot', 'Ankle'],
};
const FINGER: Record<string, string> = { '1': 'thumb', '2': 'index finger', '3': 'middle finger', '4': 'ring finger', '5': 'little finger' };
const FINGER_JOINT = { thumb: ['base', 'knuckle', 'end joint'], other: ['knuckle', 'middle joint', 'end joint'] };
const TOE: Record<string, string> = { '1': 'big toe', '2': 'second toe', '3': 'third toe', '4': 'fourth toe', '5': 'little toe' };

// a bone's plain name and its joint: ('Left upper arm', 'Shoulder'), ('Right index finger, middle joint', 'Finger')
export function jointName(label: string): { name: string; joint: string } {
  const [base, sfx] = splitSide(label), side = sfx === '.L' ? 'Left ' : sfx === '.R' ? 'Right ' : '';
  const f = /^finger(\d)-(\d)$/.exec(base);
  if (f) {
    const words = FINGER_JOINT[f[1] === '1' ? 'thumb' : 'other'][+f[2] - 1];
    return { name: `${side}${FINGER[f[1]]}, ${words}`, joint: 'Finger' };
  }
  const t = /^toe(\d)-(\d)$/.exec(base);
  if (t) return { name: `${side}${TOE[t[1]]}, joint ${t[2]}`, joint: 'Toe' };
  const n = NAMES[base];
  if (!n) return { name: label, joint: '' };
  return { name: side ? side + n[0] : n[0], joint: n[1] };
}

export interface JointValues { bone: number; label: string; name: string; joint: string; dofs: boolean[]; lo: number[]; hi: number[]; value: number[]; group: string | null }

interface Group { name: string; members: number[]; shares: number[]; read: number; readShare: number; lo: number; hi: number }

// the joints of one rig: frames, twist groups, and the values of a pose
export class Joints {
  readonly rig: RigSpace;
  readonly frames: Map<number, JointFrame>;
  private groups = new Map<number, Group>();
  private index: Map<string, number>;

  constructor(rig: RigSpace) {
    this.rig = rig;
    this.frames = jointFrames(rig);
    this.index = new Map(rig.labels.map((l, i) => [l, i]));
    for (const [name, g] of Object.entries<any>(TABLE.twist)) for (const side of ['.L', '.R']) {
      const members = Object.keys(g.shares).map((m) => this.index.get(m + side));
      if (members.some((m) => m === undefined)) continue;
      const grp: Group = { name: name + side, members: members as number[], shares: Object.values<number>(g.shares), read: this.index.get(g.read + side)!,
        readShare: g.shares[g.read], lo: g.range[0], hi: g.range[1] };
      for (const m of grp.members) this.groups.set(m, grp);
    }
  }

  bone(label: string): number | undefined { return this.index.get(label); }
  has(b: number): boolean { return this.frames.has(b); }

  // a bone's values (degrees) on a pose, with its ranges and its free axes (a twist group's member twists with it)
  values(b: number, rotations: number[][]): JointValues | null {
    const f = this.frames.get(b);
    if (!f) return null;
    const c = coordinatesOf(f, rotations[b]).map((x) => x * DEG);
    const lo = f.lo.map((x) => x * DEG), hi = f.hi.map((x) => x * DEG), dofs = f.dofs.slice();
    const g = this.groups.get(b);
    if (g) {
      const fr = this.frames.get(g.read)!;
      c[2] = coordinatesOf(fr, rotations[g.read])[2] * DEG / g.readShare;
      lo[2] = g.lo; hi[2] = g.hi; dofs[2] = true;
    }
    const label = this.rig.labels[b], n = jointName(label);
    return { bone: b, label, name: n.name, joint: n.joint, dofs, lo, hi, value: c, group: g ? g.name : null };
  }

  // a pose with one value of a bone set (degrees); `limits` holds it within the range. Returns the bones changed.
  set(rotations: number[][], b: number, axis: number, deg: number, limits = true): number[] {
    const f = this.frames.get(b);
    if (!f) return [];
    const g = this.groups.get(b);
    if (axis === 2 && g) {
      const v = limits ? Math.min(Math.max(deg, g.lo), g.hi) : deg;
      g.members.forEach((m, i) => {
        const fm = this.frames.get(m)!, c = coordinatesOf(fm, rotations[m]);
        c[2] = g.shares[i] * v / DEG;
        rotations[m] = rotationOf(fm, c);
      });
      return g.members.slice();
    }
    const c = coordinatesOf(f, rotations[b]);
    let v = deg;
    if (limits) v = Math.min(Math.max(v, f.lo[axis] * DEG), f.hi[axis] * DEG);
    c[axis] = v / DEG;
    rotations[b] = rotationOf(f, c);
    return [b];
  }

  // a bone (and its twist group) back to the reference pose
  reset(rotations: number[][], b: number): number[] {
    const g = this.groups.get(b), list = g ? g.members : [b];
    for (const m of list) rotations[m] = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    return list;
  }

  // the bones of a limb from a bone: its chain of the table (arm, leg, hand, neck, spine)
  limbOf(b: number): number[] {
    const [base, side] = splitSide(this.rig.labels[b]);
    const part = Object.entries<string[]>(TABLE.chains).find(([k, words]) => k !== 'hand' && k !== 'foot' && words.some((w) => base.startsWith(w)))?.[0];
    if (!part) return [b];
    return this.rig.labels.map((l, i) => [l, i] as [string, number]).filter(([l]) => {
      const [n, s] = splitSide(l);
      return s === side && TABLE.chains[part].some((w: string) => n.startsWith(w));
    }).map(([, i]) => i);
  }
}

// the joints of the body map (artboard 2a): the middle from the head down, then each side from the collarbone to the ankle
export const BODY_JOINTS: { label: string; at: [number, number] }[] = [
  { label: 'head', at: [110, 30] }, { label: 'neck02', at: [110, 62] }, { label: 'spine02', at: [110, 96] }, { label: 'spine04', at: [110, 130] },
  { label: 'spine05', at: [110, 158] },
  ...(['R', 'L'] as const).flatMap((s) => {
    const x = (dx: number) => (s === 'R' ? 110 - dx : 110 + dx);
    return [
      { label: 'clavicle.' + s, at: [x(22), 74] as [number, number] }, { label: 'upperarm01.' + s, at: [x(40), 82] as [number, number] },
      { label: 'lowerarm01.' + s, at: [x(52), 132] as [number, number] }, { label: 'wrist.' + s, at: [x(62), 178] as [number, number] },
      { label: 'upperleg01.' + s, at: [x(20), 174] as [number, number] }, { label: 'lowerleg01.' + s, at: [x(22), 236] as [number, number] },
      { label: 'foot.' + s, at: [x(24), 296] as [number, number] },
    ];
  }),
];

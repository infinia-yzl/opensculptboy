// OpenSculptBoy
// Apache License, Version 2.0
//
// The page's builder of a reference-point solve: the twin of opensculptboy.posing.point_solve.build_problem (with
// joint_frames, chain_bones and the joint coordinates) and of PoseRig.influences. It writes the same problem
// (format opensculptboy.point-solve/1) from the same parts, so that solve.ts runs it as the Python solve does;
// test_pose_edit_problem checks the two builders against each other.
//
// A rig here is the space of the turns of opensculptboy.posing.pose_rig: each bone's head in the reference pose
// (`heads`, Anny's axes, z up) and the fixed map `frame` (4 x 4, row-major) from that space to the output frame. A
// pose is a rotation per bone (3 x 3, row-major), each a turn about the bone's head after its parent's.

import RANGES from '../../../src/opensculptboy/data/posing/joint_ranges.json' with { type: 'json' };
import { logRotation, type SolveProblem } from './solve.ts';
import { listOf } from '../ui/words.ts';

export type V3 = [number, number, number];
export type M3 = number[];          // 3 x 3, row-major
export const TABLE: any = RANGES;

// the weights of the Python solve (point_solve.PRIOR_WEIGHT, NATURAL_WEIGHT, PIN_WEIGHT)
export const PRIOR_WEIGHT = 0.003;
export const NATURAL_WEIGHT = 0.03;
export const PIN_WEIGHT = 10.0;
const MIRROR: V3 = [1, -1, -1];
const DIRECTIONS: Record<string, V3> = { front: [0, -1, 0], back: [0, 1, 0], up: [0, 0, 1], down: [0, 0, -1] };

export interface RigSpace {
  labels: string[];
  parents: number[];
  heads: number[][];     // (B, 3), the space of the turns
  frame: number[][];     // (4, 4), the space of the turns to the output frame
}

export interface Influence { bones: number[]; s: number[]; u: number[][] }
export interface PointInput { influence: Influence; target: number[]; weight?: number; on?: boolean; finger?: string | null }
export interface PinInput { influence: Influence; kind: 'point' | 'plane'; target?: number[]; point?: number[]; normal?: number[]; weight?: number }
export interface JointFrame { axes: number[][]; sign: V3; dofs: boolean[]; lo: V3; hi: V3 }

// ------------------------------------------------------------------ small vector helpers
const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: ArrayLike<number>, b: ArrayLike<number>): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const sub = (a: ArrayLike<number>, b: ArrayLike<number>): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a: ArrayLike<number>, k: number): V3 => [a[0] * k, a[1] * k, a[2] * k];
const add = (a: ArrayLike<number>, b: ArrayLike<number>): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
// a unit vector, or the vector itself when it is (nearly) zero, as point_solve._unit
export function unit(v: ArrayLike<number>): V3 {
  const n = Math.sqrt(dot(v, v));
  return n > 1e-12 ? [v[0] / n, v[1] / n, v[2] / n] : [v[0], v[1], v[2]];
}
const radians = (d: number) => d * Math.PI / 180;

export function splitSide(label: string): [string, string] {
  for (const s of ['.L', '.R']) if (label.endsWith(s)) return [label.slice(0, -s.length), s];
  return [label, ''];
}

// the bones with every parent before its children (pose_rig.topological_order)
export function topologicalOrder(parents: ArrayLike<number>): number[] {
  const n = parents.length, order: number[] = [], seen = new Uint8Array(n);
  while (order.length < n) {
    for (let b = 0; b < n; b++) if (!seen[b] && (parents[b] < 0 || seen[parents[b]])) { order.push(b); seen[b] = 1; }
  }
  return order;
}

function indexOf(rig: RigSpace): Map<string, number> {
  const m = new Map<string, number>();
  rig.labels.forEach((l, i) => m.set(l, i));
  return m;
}

// the palm's normal and the direction towards the little finger of a hand in the reference pose (point_solve._hand_axes)
export function handAxes(rig: RigSpace, side: string, ix = indexOf(rig)): [V3, V3] {
  const H = rig.heads;
  const wrist = H[ix.get('wrist' + side)!];
  const along = sub(H[ix.get('finger3-1' + side)!], wrist);
  let across = sub(H[ix.get('finger2-1' + side)!], H[ix.get('finger5-1' + side)!]);
  const ua = unit(along);
  across = sub(across, scale(ua, dot(across, ua)));
  const sign = side === '.R' ? 1 : -1;
  const palm = scale(unit(cross(along, across)), -sign);
  return [palm, scale(unit(across), -1)];
}

// each bone's joint frame in the table (point_solve.joint_frames): the columns of `axes` are the bend, side and
// twist axes in the space of the turns
export function jointFrames(rig: RigSpace, table: any = TABLE): Map<number, JointFrame> {
  const H = rig.heads, ix = indexOf(rig), out = new Map<number, JointFrame>();
  rig.labels.forEach((label, b) => {
    const [name, side] = splitSide(label);
    const spec = table.bones[name];
    if (!spec) return;
    const nxt: string = spec.next;
    let t: V3;
    if (nxt === '+z') t = [0, 0, 1];
    else if (nxt === 'tip') t = unit(sub(H[b], H[rig.parents[b]]));
    else {
      const j = ix.get(nxt + side);
      if (j === undefined) return;
      t = unit(sub(H[j], H[b]));
    }
    let d: V3;
    if (spec.toward === 'palm' || spec.toward === 'thumb') {
      const [palm, little] = handAxes(rig, side, ix);
      d = spec.toward === 'palm' ? palm : unit(add(scale(palm, 0.5), scale(little, 0.85)));
    } else d = [...DIRECTIONS[spec.toward]] as V3;
    let a: V3;
    if (spec.hinge) {
      const h = spec.hinge as number[];
      a = unit(side === '.L' ? [h[0] * MIRROR[0], h[1] * MIRROR[1], h[2] * MIRROR[2]] : h);
      t = unit(sub(t, scale(a, dot(t, a))));
    } else {
      d = sub(d, scale(t, dot(d, t)));
      a = unit(cross(t, d));
    }
    const s = cross(t, a);
    out.set(b, {
      axes: [[a[0], s[0], t[0]], [a[1], s[1], t[1]], [a[2], s[2], t[2]]],
      sign: side === '.L' ? [1, -1, -1] : [1, 1, 1],
      dofs: ['b', 's', 't'].map((k) => spec.dofs.includes(k)),
      lo: [radians(spec.bend[0]), radians(spec.side[0]), radians(spec.twist[0])],
      hi: [radians(spec.bend[1]), radians(spec.side[1]), radians(spec.twist[1])],
    });
  });
  return out;
}

// the bones (indices) of chains such as "arm.r", "hand.l" or "leg.r", parents first (point_solve.chain_bones)
export function chainBones(rig: RigSpace, chains: string[], table: any = TABLE, order = topologicalOrder(rig.parents)): number[] {
  const picked = new Set<number>();
  for (const chain of chains) {
    const lower = chain.toLowerCase(), dot0 = lower.indexOf('.');
    const part = dot0 < 0 ? lower : lower.slice(0, dot0), side = dot0 < 0 ? '' : lower.slice(dot0 + 1);
    const words: string[] | undefined = table.chains[part];
    if (!words) throw new Error(`Unknown chain ${chain}; use one of ${listOf(Object.keys(table.chains).sort(), 'or')}.`);
    const suffix = side ? '.' + side.toUpperCase() : '';
    rig.labels.forEach((label, b) => {
      const [name, s] = splitSide(label);
      if (s !== suffix) return;
      if (words.some((w) => name.startsWith(w))) picked.add(b);
    });
  }
  return order.filter((b) => picked.has(b));
}

// a rotation's joint coordinates (bend, side, twist; radians) in a joint frame (point_solve._coordinates)
export function coordinatesOf(f: JointFrame, R: ArrayLike<number>, o = 0): V3 {
  const w = logRotation(R, o), A = f.axes;
  return [0, 1, 2].map((k) => f.sign[k] * (A[0][k] * w[0] + A[1][k] * w[1] + A[2][k] * w[2])) as V3;
}

// the rotation of joint coordinates in a joint frame: exp(A (sign * c)), row-major
export function rotationOf(f: JointFrame, c: ArrayLike<number>): M3 {
  const v = [0, 1, 2].map((r) => f.axes[r][0] * f.sign[0] * c[0] + f.axes[r][1] * f.sign[1] * c[1] + f.axes[r][2] * f.sign[2] * c[2]);
  return expRotation(v);
}

export function expRotation(v: ArrayLike<number>): M3 {
  const angle = Math.sqrt(dot(v, v));
  const K = [0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0];
  const KK = mul3(K, K);
  const a = angle < 1e-9 ? 1 : Math.sin(angle) / angle, b = angle < 1e-9 ? 0 : (1 - Math.cos(angle)) / (angle * angle);
  return [0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => (i % 4 === 0 ? 1 : 0) + a * K[i] + b * KK[i]);
}

export function mul3(A: ArrayLike<number>, B: ArrayLike<number>): M3 {
  const out = new Array(9).fill(0);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c];
  return out;
}

// x -> frame^-1 (x - t): output frame to the space of the turns (PoseRig.from_output)
export function fromOutput(frame: number[][], x: ArrayLike<number>): V3 {
  const A = [frame[0][0], frame[0][1], frame[0][2], frame[1][0], frame[1][1], frame[1][2], frame[2][0], frame[2][1], frame[2][2]];
  const M = invert3(A), d = [x[0] - frame[0][3], x[1] - frame[1][3], x[2] - frame[2][3]];
  return [M[0] * d[0] + M[1] * d[1] + M[2] * d[2], M[3] * d[0] + M[4] * d[1] + M[5] * d[2], M[6] * d[0] + M[7] * d[1] + M[8] * d[2]];
}

export function toOutput(frame: number[][], x: ArrayLike<number>): V3 {
  return [0, 1, 2].map((r) => frame[r][0] * x[0] + frame[r][1] * x[1] + frame[r][2] * x[2] + frame[r][3]) as V3;
}

export function invert3(m: ArrayLike<number>): M3 {
  const [a, b, c, d, e, f, g, h, i] = m as number[];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [A / det, -(b * i - c * h) / det, (b * f - c * e) / det, B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
    C / det, -(a * h - b * g) / det, (a * e - b * d) / det];
}

export interface ProblemOptions {
  table?: any; limits?: boolean; natural?: number[][] | null; pins?: PinInput[]; prior?: number; naturalWeight?: number;
  iterations?: number; bones?: number[] | null;
}

// the solve as plain JSON, as build_problem writes it. rotations: (B, 9) or (B, 3, 3), row-major
export function buildProblem(rig: RigSpace, rotations: ArrayLike<number>[] | number[][][], points: PointInput[], chains: string[],
  opts: ProblemOptions = {}): SolveProblem {
  const table = opts.table || TABLE, limits = opts.limits ?? true;
  const R = rotations.map((r: any) => (Array.isArray(r[0]) ? (r as number[][]).flat() : Array.from(r as ArrayLike<number>)));
  const frames = jointFrames(rig, table), ix = indexOf(rig), order = topologicalOrder(rig.parents);
  let active = chains.length ? chainBones(rig, chains, table, order) : [];
  if (opts.bones) {
    const keep = new Set([...active, ...opts.bones]);
    active = order.filter((b) => keep.has(b));
  }
  const activeSet = new Set(active);
  // the twist groups whose bones all turn
  const groups: any[] = [], grouped = new Set<number>();
  for (const [name, g] of Object.entries<any>(table.twist)) {
    for (const side of ['.L', '.R']) {
      const members = Object.keys(g.shares).map((m) => ix.get(m + side));
      if (members.every((m) => m !== undefined && activeSet.has(m))) {
        groups.push({
          name: name + side, axis: [ix.get(g.axis[0] + side)!, ix.get(g.axis[1] + side)!], members: members as number[],
          shares: Object.values<number>(g.shares).map(Number), read: ix.get(g.read + side)!, read_share: Number(g.shares[g.read]),
          lo: radians(g.range[0]), hi: radians(g.range[1]),
        });
        for (const m of members) grouped.add(m!);
      }
    }
  }
  const columns: any[] = [], bonesOut: any[] = [];
  for (const b of active) {
    const fr = frames.get(b);
    if (!fr) continue;
    const dofs = fr.dofs.slice(), lo = fr.lo.slice(), hi = fr.hi.slice();
    const [name] = splitSide(rig.labels[b]);
    for (const g of Object.values<any>(table.twist)) {
      if (name in g.shares && !grouped.has(b)) {
        dofs[2] = true;
        const share = Number(g.shares[name]);
        lo[2] = radians(g.range[0]) * share; hi[2] = radians(g.range[1]) * share;
      }
    }
    bonesOut.push({ bone: b, axes: fr.axes.map((r) => r.slice()), sign: fr.sign.slice(), dofs, lo, hi, start: coordinatesOf(fr, R[b]) });
    for (let k = 0; k < 3; k++) if (dofs[k]) columns.push({ bone: b, axis: k });
  }
  groups.forEach((_, gi) => columns.push({ group: gi }));
  const F = rig.frame;
  const det = F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1]) - F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0]) +
    F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]);
  const S = Math.cbrt(Math.abs(det));
  const pts = points.map((p) => {
    const w = p.weight ?? 1, on = (p.on ?? true) && w > 0;
    return {
      bones: p.influence.bones.map(Number), s: p.influence.s.map(Number), u: p.influence.u.map((r) => r.map(Number)),
      target: fromOutput(F, p.target), weight: on ? w : 0, finger: p.finger ?? null,
    };
  });
  const pinOut = (opts.pins || []).map((q) => {
    const e: any = { bones: q.influence.bones.map(Number), s: q.influence.s.map(Number), u: q.influence.u.map((r) => r.map(Number)), kind: q.kind,
      weight: q.weight ?? PIN_WEIGHT };
    if (q.kind === 'point') e.target = fromOutput(F, q.target!);
    else if (q.kind === 'plane') {
      e.point = fromOutput(F, q.point!);
      const n = q.normal!;
      e.normal = unit([0, 1, 2].map((c) => n[0] * F[0][c] + n[1] * F[1][c] + n[2] * F[2][c]));
    } else throw new Error(`A pin is a point or a plane; got ${(q as any).kind}.`);
    return e;
  });
  // the natural curl of fingers whose points are all off: by their neighbours, else the pose
  const fingers = new Map<string, boolean[]>();
  for (const p of pts) if (p.finger) { if (!fingers.has(p.finger)) fingers.set(p.finger, []); fingers.get(p.finger)!.push(p.weight > 0); }
  const off = [...fingers].filter(([, f]) => !f.some(Boolean)).map(([k]) => k).sort();
  const onSet = new Set([...fingers].filter(([, f]) => f.some(Boolean)).map(([k]) => k));
  const natural = opts.natural ? opts.natural.map((r: any) => (Array.isArray(r[0]) ? r.flat() : r)) : R;
  const curls: any[] = [];
  for (const f of off) {
    const [name, side] = splitSide(f);
    const k = parseInt(name[name.length - 1], 10);
    const joints = [1, 2, 3].map((j) => ix.get(`${name}-${j}${side}`));
    if (joints.some((j) => j === undefined || !activeSet.has(j))) continue;
    const neighbours: number[] = [];
    if (k > 1) {
      for (const dk of [1, -1, 2, -2, 3, -3]) {
        const n = k + dk;
        if (n >= 2 && n <= 5 && onSet.has(`finger${n}${side}`)) neighbours.push(n);
        if (neighbours.length === 2) break;
      }
    }
    joints.forEach((b, j) => {
      curls.push({ bone: b!, neighbours: neighbours.map((n) => ix.get(`finger${n}-${j + 1}${side}`)!), natural: coordinatesOf(frames.get(b!)!, natural[b!]) });
    });
  }
  return {
    format: 'opensculptboy.point-solve/1', parents: rig.parents.slice(), heads: rig.heads.map((h) => h.slice()), order, scale: S,
    frame: F.map((r) => r.slice()), rotations: R, bones: bonesOut, groups, columns, points: pts, pins: pinOut, curls, limits,
    weights: { prior: opts.prior ?? PRIOR_WEIGHT, natural: opts.naturalWeight ?? NATURAL_WEIGHT }, iterations: opts.iterations ?? 40,
  };
}

// ------------------------------------------------------------------ skinning influences
// A point given by a triangle's three vertex rows and barycentric weights skins as a sum of influences (bone, s, u):
// x = sum R_b u + s t_b (PoseRig.influences). rest(row): the rest vertex; skin(row): its bones and weights;
// bind(bone): the bone's map from the rest bone frame to the reference one (4 x 4 row-major), or null for none.
export interface SkinSource {
  rest(row: number): ArrayLike<number>;
  skin(row: number): { bones: ArrayLike<number>; weights: ArrayLike<number> };
  bind?(bone: number): ArrayLike<number> | null;
}

export function influenceOf(vertices: number[], barycentric: number[], src: SkinSource): Influence {
  const acc = new Map<number, number[]>();
  vertices.forEach((vi, k) => {
    const a = barycentric[k], v = src.rest(vi), { bones, weights } = src.skin(vi);
    for (let j = 0; j < bones.length; j++) {
      const w = weights[j];
      if (w === 0) continue;
      const b = bones[j], M = src.bind ? src.bind(b) : null;
      let h: number[];
      if (M) h = [0, 1, 2, 3].map((r) => a * w * (M[r * 4] * v[0] + M[r * 4 + 1] * v[1] + M[r * 4 + 2] * v[2] + M[r * 4 + 3]));
      else h = [a * w * v[0], a * w * v[1], a * w * v[2], a * w];
      const e = acc.get(b);
      if (e) for (let r = 0; r < 4; r++) e[r] += h[r]; else acc.set(b, h);
    }
  });
  const bones = [...acc.keys()].sort((x, y) => x - y);
  return { bones, u: bones.map((b) => acc.get(b)!.slice(0, 3)), s: bones.map((b) => acc.get(b)![3]) };
}

// the chains of turns K (rotations B x 9 and translations B x 3) of a pose in the space of the turns (PoseRig.transforms)
export function transforms(rig: RigSpace, rotations: ArrayLike<number>[], order = topologicalOrder(rig.parents)): { R: Float64Array; t: Float64Array } {
  const B = rig.labels.length, KR = new Float64Array(B * 9), Kt = new Float64Array(B * 3), H = rig.heads;
  for (const b of order) {
    const p = rig.parents[b];
    if (p < 0) { KR[b * 9] = KR[b * 9 + 4] = KR[b * 9 + 8] = 1; continue; }
    const D = rotations[b], h = H[b];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
      KR[b * 9 + r * 3 + c] = KR[p * 9 + r * 3] * D[c] + KR[p * 9 + r * 3 + 1] * D[3 + c] + KR[p * 9 + r * 3 + 2] * D[6 + c];
    }
    const l = [h[0] - (D[0] * h[0] + D[1] * h[1] + D[2] * h[2]), h[1] - (D[3] * h[0] + D[4] * h[1] + D[5] * h[2]), h[2] - (D[6] * h[0] + D[7] * h[1] + D[8] * h[2])];
    for (let r = 0; r < 3; r++) Kt[b * 3 + r] = KR[p * 9 + r * 3] * l[0] + KR[p * 9 + r * 3 + 1] * l[1] + KR[p * 9 + r * 3 + 2] * l[2] + Kt[p * 3 + r];
  }
  return { R: KR, t: Kt };
}

// a point of influences on a pose's chains, in the output frame
export function pointAt(rig: RigSpace, K: { R: Float64Array; t: Float64Array }, f: Influence): V3 {
  let x = 0, y = 0, z = 0;
  f.bones.forEach((b, k) => {
    const o = b * 9, u = f.u[k], s = f.s[k];
    x += K.R[o] * u[0] + K.R[o + 1] * u[1] + K.R[o + 2] * u[2] + s * K.t[b * 3];
    y += K.R[o + 3] * u[0] + K.R[o + 4] * u[1] + K.R[o + 5] * u[2] + s * K.t[b * 3 + 1];
    z += K.R[o + 6] * u[0] + K.R[o + 7] * u[1] + K.R[o + 8] * u[2] + s * K.t[b * 3 + 2];
  });
  return toOutput(rig.frame, [x, y, z]);
}

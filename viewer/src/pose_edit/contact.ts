// OpenSculptBoy
// Apache License, Version 2.0
//
// The pose editor's contact helpers in the page, the twins of opensculptboy.posing.contact: Squeeze closes a hand's
// fingers together onto a held prop (each finger along its own curl until its skin comes within the clearance of the
// prop, then all of them scaled by one value, as GraspIt's AutoGrasp closes a hand: A. T. Miller and P. K. Allen,
// "GraspIt!", IEEE Robotics and Automation Magazine, 2004), and the balance mark, the posed skin's centre of mass
// (a solid of uniform density, by the divergence theorem: C. Zhang and T. Chen, "Efficient feature extraction for
// 2D/3D objects in mesh representation", 2001) over the convex hull of the foot skin on the ground (Andrew's
// monotone chain). test_pose_edit_contact checks both against Python.

import { coordinatesOf, jointFrames, rotationOf, transforms, type RigSpace } from './problem.ts';

export const CLEARANCE = 0.001;
export const MAX_SCALE = 1.6;
export const SCAN_STEP = 0.04;
export const BISECTION = 12;
export const FIST_BENDS: Record<string, [number, number, number]> = {
  finger1: [26.0, 21.0, 54.3], finger2: [71.5, 64.3, 78.2], finger3: [59.1, 77.3, 82.1], finger4: [59.6, 78.8, 76.6], finger5: [46.5, 90.9, 56.0],
};
export const CONTACT = 0.01;

// Anny's level-0 skin on a rig (PoseRig.vertices): rest vertices (n * 3), k influences per vertex, faces, and each
// bone's bind map (4 x 4 row-major) or none
export class SkinnedBody {
  readonly n: number;
  readonly strongest: Int32Array;
  readonly rig: RigSpace; readonly rest: Float64Array; readonly indices: Int32Array; readonly weights: Float64Array; readonly k: number;
  readonly faces: Uint32Array; readonly bind: ((b: number) => ArrayLike<number> | null) | null;
  constructor(rig: RigSpace, rest: Float64Array, indices: Int32Array, weights: Float64Array, k: number, faces: Uint32Array,
    bind: ((b: number) => ArrayLike<number> | null) | null = null) {
    this.rig = rig; this.rest = rest; this.indices = indices; this.weights = weights; this.k = k; this.faces = faces; this.bind = bind;
    this.n = rest.length / 3;
    this.strongest = new Int32Array(this.n);
    for (let v = 0; v < this.n; v++) {
      let best = 0;
      for (let j = 1; j < k; j++) if (weights[v * k + j] > weights[v * k + best]) best = j;
      this.strongest[v] = indices[v * k + best];
    }
  }

  // the posed vertices (output frame) of rows, or of every vertex
  vertices(rotations: ArrayLike<number>[], rows?: ArrayLike<number>): Float64Array {
    const K = transforms(this.rig, rotations), F = this.rig.frame, k = this.k;
    const list = rows || null, m = list ? list.length : this.n, out = new Float64Array(m * 3);
    const M = new Float64Array(12);
    for (let q = 0; q < m; q++) {
      const v = list ? list[q] : q, px = this.rest[v * 3], py = this.rest[v * 3 + 1], pz = this.rest[v * 3 + 2];
      let x = 0, y = 0, z = 0;
      for (let j = 0; j < k; j++) {
        const w = this.weights[v * k + j];
        if (w === 0) continue;
        const b = this.indices[v * k + j], B = this.bind ? this.bind(b) : null;
        let a0 = px, a1 = py, a2 = pz;
        if (B) {
          a0 = B[0] * px + B[1] * py + B[2] * pz + B[3]; a1 = B[4] * px + B[5] * py + B[6] * pz + B[7]; a2 = B[8] * px + B[9] * py + B[10] * pz + B[11];
        }
        const o = b * 9;
        M[0] = K.R[o] * a0 + K.R[o + 1] * a1 + K.R[o + 2] * a2 + K.t[b * 3];
        M[1] = K.R[o + 3] * a0 + K.R[o + 4] * a1 + K.R[o + 5] * a2 + K.t[b * 3 + 1];
        M[2] = K.R[o + 6] * a0 + K.R[o + 7] * a1 + K.R[o + 8] * a2 + K.t[b * 3 + 2];
        x += w * M[0]; y += w * M[1]; z += w * M[2];
      }
      out[q * 3] = F[0][0] * x + F[0][1] * y + F[0][2] * z + F[0][3];
      out[q * 3 + 1] = F[1][0] * x + F[1][1] * y + F[1][2] * z + F[1][3];
      out[q * 3 + 2] = F[2][0] * x + F[2][1] * y + F[2][2] * z + F[2][3];
    }
    return out;
  }

  // the rows whose strongest bone is one of bones
  rowsOf(bones: number[]): Int32Array {
    const set = new Set(bones), out: number[] = [];
    for (let v = 0; v < this.n; v++) if (set.has(this.strongest[v])) out.push(v);
    return Int32Array.from(out);
  }
}

// ------------------------------------------------------------------ props as analytic objects
export type Prop = { kind: 'sphere'; centre: number[]; radius: number } | { kind: 'capsule'; start: number[]; end: number[]; radius: number }
  | { kind: 'ellipsoid'; centre: number[]; radii: number[]; axis?: number[] } | ((p: Float64Array) => Float64Array);

// the signed distance (negative inside) of points (n * 3) to the nearest object (contact.signed_distance)
export function signedDistance(P: Float64Array, objects: Prop[]): Float64Array {
  const n = P.length / 3, out = new Float64Array(n).fill(Infinity);
  for (const o of objects) {
    let d: Float64Array;
    if (typeof o === 'function') d = o(P);
    else {
      d = new Float64Array(n);
      if (o.kind === 'sphere') {
        for (let i = 0; i < n; i++) d[i] = Math.hypot(P[i * 3] - o.centre[0], P[i * 3 + 1] - o.centre[1], P[i * 3 + 2] - o.centre[2]) - o.radius;
      } else if (o.kind === 'capsule') {
        const a = o.start, ab = [o.end[0] - a[0], o.end[1] - a[1], o.end[2] - a[2]], l2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
        for (let i = 0; i < n; i++) {
          const px = P[i * 3] - a[0], py = P[i * 3 + 1] - a[1], pz = P[i * 3 + 2] - a[2];
          const t = Math.min(Math.max((px * ab[0] + py * ab[1] + pz * ab[2]) / l2, 0), 1);
          d[i] = Math.hypot(px - t * ab[0], py - t * ab[1], pz - t * ab[2]) - o.radius;
        }
      } else {
        let ax = o.axis || [1, 0, 0];
        const l = Math.hypot(ax[0], ax[1], ax[2]); ax = [ax[0] / l, ax[1] / l, ax[2] / l];
        const other = Math.abs(ax[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
        let s = [ax[1] * other[2] - ax[2] * other[1], ax[2] * other[0] - ax[0] * other[2], ax[0] * other[1] - ax[1] * other[0]];
        const ls = Math.hypot(s[0], s[1], s[2]); s = [s[0] / ls, s[1] / ls, s[2] / ls];
        const third = [ax[1] * s[2] - ax[2] * s[1], ax[2] * s[0] - ax[0] * s[2], ax[0] * s[1] - ax[1] * s[0]];
        const F = [ax, s, third], r = o.radii, rmin = Math.min(...r);
        for (let i = 0; i < n; i++) {
          const q = [P[i * 3] - o.centre[0], P[i * 3 + 1] - o.centre[1], P[i * 3 + 2] - o.centre[2]];
          const loc = F.map((f, k) => (f[0] * q[0] + f[1] * q[1] + f[2] * q[2]) / r[k]);
          d[i] = (Math.hypot(loc[0], loc[1], loc[2]) - 1) * rmin;
        }
      }
    }
    for (let i = 0; i < n; i++) if (d[i] < out[i]) out[i] = d[i];
  }
  return out;
}

export interface SqueezeFinger { contact: number | null; scale: number; clearance_mm: number; inside: boolean }

// close a hand's fingers ("hand.r") onto objects by one value from 0 to 1 (contact.squeeze); returns the rotations
// (B x 9) and a report per finger
export function squeeze(body: SkinnedBody, rotations: ArrayLike<number>[], hand: string, objects: Prop[], value = 1,
  clearance = CLEARANCE, fingers = ['finger1', 'finger2', 'finger3', 'finger4', 'finger5']): { rotations: number[][]; report: Record<string, SqueezeFinger> } {
  const rig = body.rig, side = '.' + hand.split('.')[1].toUpperCase();
  value = Math.min(Math.max(value, 0), 1);
  const D = rotations.map((r) => Array.from(r));
  const frames = jointFrames(rig), ix = new Map(rig.labels.map((l, i) => [l, i] as [string, number]));
  const report: Record<string, SqueezeFinger> = {};
  for (const finger of fingers) {
    const bones = [1, 2, 3].map((j) => ix.get(`${finger}-${j}${side}`)!);
    const coords = bones.map((b) => coordinatesOf(frames.get(b)!, D[b]));
    let bends = coords.map((c) => c[0]);
    if (bends.reduce((s, b) => s + Math.abs(b) * 180 / Math.PI, 0) < 15) bends = FIST_BENDS[finger].map((d) => d * Math.PI / 180);
    const rows = body.rowsOf(bones);
    const posed = (scale: number) => {
      const trial = D.slice();
      bones.forEach((b, j) => { const c = coords[j].slice(); c[0] = scale * bends[j]; trial[b] = rotationOf(frames.get(b)!, c); });
      return trial;
    };
    const gap = (scale: number) => { const d = signedDistance(body.vertices(posed(scale), rows), objects); let m = Infinity; for (const x of d) if (x < m) m = x; return m; };
    let contact: number | null = null, previous = 0;
    if (gap(0) <= clearance) contact = 0;
    else {
      const steps = Math.ceil((MAX_SCALE + 1e-9 - SCAN_STEP) / SCAN_STEP);
      for (let i = 0; i < steps; i++) {
        const scale = SCAN_STEP + i * SCAN_STEP;
        if (gap(scale) <= clearance) {
          let lo = previous, hi = scale;
          for (let it = 0; it < BISECTION; it++) { const mid = 0.5 * (lo + hi); if (gap(mid) > clearance) lo = mid; else hi = mid; }
          contact = lo;
          break;
        }
        previous = scale;
      }
    }
    const chosen = value * (contact === null ? MAX_SCALE : contact);
    const now = posed(chosen);
    for (const b of bones) D[b] = now[b];
    const r4 = (x: number) => Math.round(x * 1e4) / 1e4;
    report[finger] = { contact: contact === null ? null : r4(contact), scale: r4(chosen), clearance_mm: Math.round(1000 * gap(chosen) * 100) / 100,
      inside: contact === 0 && gap(0) < 0 };
  }
  return { rotations: D, report };
}

export interface GripFinger { start: number; contact: number | null; scale: number; clearance_mm: number; opened: boolean; wedged: boolean; blocked: boolean }
export const WEDGE_STEP = 0.01;

// Grip from the hand as it stands (the editor's Grip): each finger of hand ("hand.r") starts from its present curl.
// A finger clear of the objects keeps that start; a finger that touches or enters them opens along its own curl in
// steps of SCAN_STEP until it clears them by the clearance (the finger is "opened"). A finger outside the objects that
// clears them by the clearance at no curl down to straight lies between two of their parts (a finger between a
// sword's grip and its guard): it is "wedged" and takes the curl, among those it reaches from the present one without
// entering the objects, at which it lies farthest from them; first, though, it curls on past the part it touches when
// a little more curl clears it, and closes from there. A finger inside them at every curl down to straight is
// "blocked" and keeps its present curl. From its start each finger closes until it comes
// within the clearance (bisection, as squeeze), and value from 0 to 1 sets each finger between its start and its
// contact, so Squeeze loosens the grip back to the start and never through the prop. GraspIt's AutoGrasp closes the
// fingers from a pre-shape clear of the object (A. T. Miller and P. K. Allen, 2004); the present pose serves as that
// pre-shape, opened only where it must be. Returns the rotations (B x 9) and a report per finger.
export function gripFrom(body: SkinnedBody, rotations: ArrayLike<number>[], hand: string, objects: Prop[], value = 1,
  clearance = CLEARANCE, fingers = ['finger1', 'finger2', 'finger3', 'finger4', 'finger5']): { rotations: number[][]; report: Record<string, GripFinger> } {
  const rig = body.rig, side = '.' + hand.split('.')[1].toUpperCase();
  value = Math.min(Math.max(value, 0), 1);
  const D = rotations.map((r) => Array.from(r));
  const frames = jointFrames(rig), ix = new Map(rig.labels.map((l, i) => [l, i] as [string, number]));
  const report: Record<string, GripFinger> = {};
  const r4 = (x: number) => Math.round(x * 1e4) / 1e4;
  for (const finger of fingers) {
    const bones = [1, 2, 3].map((j) => ix.get(`${finger}-${j}${side}`)!);
    const coords = bones.map((b) => coordinatesOf(frames.get(b)!, D[b]));
    let bends = coords.map((c) => c[0]), present = 1;
    // a finger near straight curls along the fist's bends, from straight
    if (bends.reduce((s, b) => s + Math.abs(b) * 180 / Math.PI, 0) < 15) { bends = FIST_BENDS[finger].map((d) => d * Math.PI / 180); present = 0; }
    const rows = body.rowsOf(bones);
    const posed = (scale: number) => {
      const trial = D.slice();
      bones.forEach((b, j) => { const c = coords[j].slice(); c[0] = scale * bends[j]; trial[b] = rotationOf(frames.get(b)!, c); });
      return trial;
    };
    const gap = (scale: number) => { const d = signedDistance(body.vertices(posed(scale), rows), objects); let m = Infinity; for (const x of d) if (x < m) m = x; return m; };
    // the boundary between a clear scale lo and a touching scale hi
    const boundary = (lo: number, hi: number) => { for (let it = 0; it < BISECTION; it++) { const mid = 0.5 * (lo + hi); if (gap(mid) > clearance) lo = mid; else hi = mid; } return lo; };
    let start = present, contact: number | null = null, opened = false, blocked = false, wedged = false;
    const g0 = gap(present);
    if (g0 > clearance) {
      // clear: close from here
      for (let s = present + SCAN_STEP; s <= MAX_SCALE + 1e-9; s += SCAN_STEP) if (gap(s) <= clearance) { contact = boundary(s - SCAN_STEP, s); break; }
    } else {
      // touching or inside: open along the curl until the finger clears the prop
      let clear: number | null = null;
      // (a finger outside the prop never opens through another part of it)
      for (let s = present - SCAN_STEP; s > -1e-9; s -= SCAN_STEP) {
        const g = gap(Math.max(s, 0));
        if (g > clearance) { clear = Math.max(s, 0); break; }
        if (g0 >= 0 && g < 0) break;
      }
      // a finger outside the prop that touches it: a little more curl may take it off this part of the prop (the
      // index finger off a sword's guard), and it then closes to the next contact (the grip)
      let further: number | null = null;
      if (clear === null && g0 >= 0) {
        for (let s = present + SCAN_STEP; s <= MAX_SCALE + 1e-9; s += SCAN_STEP) { const g = gap(s); if (g < 0) break; if (g > clearance) { further = s; break; } }
      }
      if (clear !== null) { opened = true; contact = boundary(clear, Math.min(clear + SCAN_STEP, present)); start = clear; }
      else if (further !== null) {
        start = further;
        for (let s = further + SCAN_STEP; s <= MAX_SCALE + 1e-9; s += SCAN_STEP) if (gap(s) <= clearance) { contact = boundary(s - SCAN_STEP, s); break; }
      } else if (g0 >= 0) {
        // wedged: the curl farthest from the objects within the stretch round the present one that stays outside
        let best = present, bestGap = g0;
        for (const dir of [-1, 1]) {
          for (let s = present + dir * WEDGE_STEP; s >= 0 && s <= MAX_SCALE + 1e-9; s += dir * WEDGE_STEP) {
            const g = gap(s);
            if (g < 0) break;
            if (g > bestGap) { bestGap = g; best = s; }
          }
        }
        wedged = true; start = best; contact = best;
      } else blocked = true;
    }
    const end = contact === null ? MAX_SCALE : contact;
    const chosen = blocked ? present : wedged ? start : start + value * (end - start);
    const now = posed(chosen);
    for (const b of bones) D[b] = now[b];
    report[finger] = { start: r4(start), contact: contact === null ? null : r4(contact), scale: r4(chosen), clearance_mm: Math.round(1000 * gap(chosen) * 100) / 100, opened, wedged, blocked };
  }
  return { rotations: D, report };
}

// ------------------------------------------------------------------ balance
export function centreOfMass(V: ArrayLike<number>, T: ArrayLike<number>): { centre: number[]; volume: number } {
  const n = V.length / 3, o = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) o[k] += V[i * 3 + k] / n;
  let total = 0;
  const acc = [0, 0, 0];
  for (let t = 0; t < T.length; t += 3) {
    const a = [0, 1, 2].map((k) => V[T[t] * 3 + k] - o[k]), b = [0, 1, 2].map((k) => V[T[t + 1] * 3 + k] - o[k]), c = [0, 1, 2].map((k) => V[T[t + 2] * 3 + k] - o[k]);
    const vol = (a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6;
    total += vol;
    for (let k = 0; k < 3; k++) acc[k] += (a[k] + b[k] + c[k]) / 4 * vol;
  }
  return { centre: [0, 1, 2].map((k) => o[k] + acc[k] / total), volume: Math.abs(total) };
}

// the convex hull of 2D points, counter-clockwise (Andrew's monotone chain), with points rounded to 1e-9 and deduplicated
export function hull2(points: number[][]): number[][] {
  const seen = new Set<string>(), P: number[][] = [];
  for (const p of points) {
    const q = [Math.round(p[0] * 1e9) / 1e9, Math.round(p[1] * 1e9) / 1e9], key = q.join(',');
    if (!seen.has(key)) { seen.add(key); P.push(q); }
  }
  if (P.length < 3) return P;
  P.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: number[], a: number[], b: number[]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: number[][] = [], upper: number[][] = [];
  for (const p of P) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop(); lower.push(p); }
  for (let i = P.length - 1; i >= 0; i--) { const p = P[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop(); upper.push(p); }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

export interface Balance { centre: number[]; mark: number[]; polygon: number[][]; inside: boolean; margin_mm: number | null; volume_m3: number }

// the balance mark of a posed body (contact.balance)
export function balance(body: SkinnedBody, rotations: ArrayLike<number>[], up = [0, 0, 1], contact = CONTACT): Balance {
  const lu = Math.hypot(up[0], up[1], up[2]); up = up.map((x) => x / lu);
  const V = body.vertices(rotations), n = body.n;
  const { centre, volume } = centreOfMass(V, body.faces);
  const height = new Float64Array(n);
  let ground = Infinity;
  for (let i = 0; i < n; i++) { height[i] = V[i * 3] * up[0] + V[i * 3 + 1] * up[1] + V[i * 3 + 2] * up[2]; if (height[i] < ground) ground = height[i]; }
  const labels = body.rig.labels;
  let support = [...Array(n).keys()].filter((i) => /^(foot|toe)/.test(labels[body.strongest[i]]) && height[i] <= ground + contact);
  if (!support.length) support = [...Array(n).keys()].filter((i) => height[i] <= ground + contact);
  const crossv = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  let e1 = crossv(up, [1, 0, 0]);
  if (Math.hypot(e1[0], e1[1], e1[2]) < 1e-6) e1 = crossv(up, [0, 1, 0]);
  const l1 = Math.hypot(e1[0], e1[1], e1[2]); e1 = e1.map((x) => x / l1);
  const e2 = crossv(up, e1);
  const dot = (a: ArrayLike<number>, o: number, b: number[]) => a[o] * b[0] + a[o + 1] * b[1] + a[o + 2] * b[2];
  const hull = hull2(support.map((i) => [dot(V, i * 3, e1), dot(V, i * 3, e2)]));
  const mark2 = [dot(centre, 0, e1), dot(centre, 0, e2)];
  let margin = Infinity;
  for (let i = 0; i < hull.length; i++) {
    const p = hull[i], q = hull[(i + 1) % hull.length], edge = [q[0] - p[0], q[1] - p[1]], l = Math.max(Math.hypot(edge[0], edge[1]), 1e-12);
    const normal = [edge[1] / l, -edge[0] / l];
    margin = Math.min(margin, -((mark2[0] - p[0]) * normal[0] + (mark2[1] - p[1]) * normal[1]));
  }
  const h = dot(centre, 0, up) - ground;
  return {
    centre, mark: [0, 1, 2].map((k) => centre[k] - h * up[k]),
    polygon: hull.map(([x, y]) => [0, 1, 2].map((k) => e1[k] * x + e2[k] * y + ground * up[k])),
    inside: hull.length >= 3 && margin >= 0, margin_mm: hull.length >= 3 ? Math.round(1000 * margin * 100) / 100 : null,
    volume_m3: Math.round(volume * 1e6) / 1e6,
  };
}

// OpenSculptBoy
// Apache License, Version 2.0
//
// The frames of the pose editor. The page shows Anny's rest frame (z up, metres) as page = s A x + o, lifted by the
// body's floor shift, with A = ANNY_TO_PAGE ((x, y, z) -> (x, z, -y)); the posed body then moves by the root's
// offset and the turntable's holder. The card frame of the scan and of the saved points is the frame of Anny's
// forward pass (opensculptboy.posing.pose_rig with an identity placement): the root's head of the rest body at the
// origin, turned by the root's own turn, so x_card = Rr (x_rest - h_root). The solve works in the space of the turns
// of Anny's local-ref pose (PoseRig: each bone's reference head in Anny's axes, the bones bound by their map from the
// rest to the reference pose) with that map as its frame. The page keeps its rotations R from the rest pose; the card
// and the solve keep local-ref turns D, which pose_reference.ts converts with each bone's turn C of the body shown:
// D_b = C_parent R_b C_b^T. A page quaternion q of a bone is the rotation A R A^T.

import type { PoseHost } from './host.ts';
import type { M3, RigSpace, SkinSource, V3 } from './problem.ts';
import { pageToRef, refToPage, referencePose } from '../pose_reference.ts';

export const A = [1, 0, 0, 0, 0, 1, 0, -1, 0];

export function quatToMat(x: number, y: number, z: number, w: number): M3 {
  return [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)];
}

export function matToQuat(m: ArrayLike<number>): [number, number, number, number] {
  const tr = m[0] + m[4] + m[8];
  let x: number, y: number, z: number, w: number;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2; w = 0.25 * s; x = (m[7] - m[5]) / s; y = (m[2] - m[6]) / s; z = (m[3] - m[1]) / s;
  } else if (m[0] > m[4] && m[0] > m[8]) {
    const s = Math.sqrt(1 + m[0] - m[4] - m[8]) * 2; w = (m[7] - m[5]) / s; x = 0.25 * s; y = (m[1] + m[3]) / s; z = (m[2] + m[6]) / s;
  } else if (m[4] > m[8]) {
    const s = Math.sqrt(1 + m[4] - m[0] - m[8]) * 2; w = (m[2] - m[6]) / s; x = (m[1] + m[3]) / s; y = 0.25 * s; z = (m[5] + m[7]) / s;
  } else {
    const s = Math.sqrt(1 + m[8] - m[0] - m[4]) * 2; w = (m[3] - m[1]) / s; x = (m[2] + m[6]) / s; y = (m[5] + m[7]) / s; z = 0.25 * s;
  }
  const l = Math.hypot(x, y, z, w) || 1;
  return [x / l, y / l, z / l, w / l];
}

const mul = (P: ArrayLike<number>, Q: ArrayLike<number>): M3 => {
  const o = new Array(9).fill(0);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[r * 3 + c] = P[r * 3] * Q[c] + P[r * 3 + 1] * Q[3 + c] + P[r * 3 + 2] * Q[6 + c];
  return o;
};
// 4 x 4 row-major product
const mul4 = (P: ArrayLike<number>, Q: ArrayLike<number>): number[] => {
  const o = new Array(16).fill(0);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) o[r * 4 + c] = P[r * 4] * Q[c] + P[r * 4 + 1] * Q[4 + c] + P[r * 4 + 2] * Q[8 + c] + P[r * 4 + 3] * Q[12 + c];
  return o;
};
const T = (P: ArrayLike<number>): M3 => [P[0], P[3], P[6], P[1], P[4], P[7], P[2], P[5], P[8]];

// page rotation -> card rotation, and back
export const toCardRot = (R: ArrayLike<number>) => mul(T(A), mul(R, A));
export const toPageRot = (D: ArrayLike<number>) => mul(A, mul(D, T(A)));

export class Frames {
  host: PoseHost;
  constructor(host: PoseHost) { this.host = host; }

  // page (rest, before the root's offset and the holder) -> card frame
  toCard(p: ArrayLike<number>): V3 {
    const f = this.host.frame(), s = f.scale, o = f.offset, l = this.host.lift(), R = f.rotation;
    const d = [(p[0] - o[0]) / s, (p[1] - o[1] - l) / s, (p[2] - o[2]) / s];
    return [R[0] * d[0] + R[3] * d[1] + R[6] * d[2], R[1] * d[0] + R[4] * d[1] + R[7] * d[2], R[2] * d[0] + R[5] * d[1] + R[8] * d[2]];
  }
  // card frame -> page (rest)
  toPage(x: ArrayLike<number>): V3 {
    const f = this.host.frame(), s = f.scale, o = f.offset, l = this.host.lift(), R = f.rotation;
    return [s * (R[0] * x[0] + R[1] * x[1] + R[2] * x[2]) + o[0], s * (R[3] * x[0] + R[4] * x[1] + R[5] * x[2]) + o[1] + l,
      s * (R[6] * x[0] + R[7] * x[1] + R[8] * x[2]) + o[2]];
  }
  // the root's shift: card frame (Anny's axes, metres, the card's pose.root) to the page's axes (s A t) and back. The
  // page draws the figure moved by it, and host.rootOffset() leaves it out, so the scan and the card frame stay put
  shiftToPage(t: ArrayLike<number>): number[] {
    const f = this.host.frame(), s = f.scale, R = f.rotation;
    return [0, 1, 2].map((r) => s * (R[r * 3] * t[0] + R[r * 3 + 1] * t[1] + R[r * 3 + 2] * t[2]));
  }
  pageToShift(p: ArrayLike<number>): V3 {
    const f = this.host.frame(), s = f.scale, R = f.rotation;
    return [0, 1, 2].map((c) => (R[c] * p[0] + R[3 + c] * p[1] + R[6 + c] * p[2]) / s) as V3;
  }
  // the root's head of the rest body (rest frame): the card frame's origin
  rootHead(): V3 { return this.toCard(this.host.restHeads()[this.host.root()]); }
  // the root's own turn of the pose shown (card axes)
  rootTurn(): M3 { const q = this.host.pose(), r = this.host.root(); return toCardRot(quatToMat(q[r * 4], q[r * 4 + 1], q[r * 4 + 2], q[r * 4 + 3])); }
  // card frame -> the rest frame of the posed figure (before the root's offset): x_rest = Rr^T x_card + h_root; the
  // root's turn is part of the pose, so the posed body in the rest frame is Rr (x - h) + h, and x_card its point less h
  private cardToRest(x: ArrayLike<number>): V3 { const h = this.rootHead(); return [x[0] + h[0], x[1] + h[1], x[2] + h[2]]; }
  private restToCard(x: ArrayLike<number>): V3 { const h = this.rootHead(); return [x[0] - h[0], x[1] - h[1], x[2] - h[2]]; }
  // card frame of the posed figure -> the world (root offset and holder)
  toWorld(x: ArrayLike<number>, out: number[] = [0, 0, 0]): number[] {
    const p = this.toPage(this.cardToRest(x)), r = this.host.rootOffset(), M = this.host.holder().elements;
    const a = p[0] + r[0], b = p[1] + r[1], c = p[2] + r[2];
    out[0] = M[0] * a + M[4] * b + M[8] * c + M[12]; out[1] = M[1] * a + M[5] * b + M[9] * c + M[13]; out[2] = M[2] * a + M[6] * b + M[10] * c + M[14];
    return out;
  }
  // the world -> card frame of the posed figure
  fromWorld(w: ArrayLike<number>): V3 {
    const M = this.host.holder().clone().invert().elements, r = this.host.rootOffset();
    const a = M[0] * w[0] + M[4] * w[1] + M[8] * w[2] + M[12] - r[0], b = M[1] * w[0] + M[5] * w[1] + M[9] * w[2] + M[13] - r[1],
      c = M[2] * w[0] + M[6] * w[1] + M[10] * w[2] + M[14] - r[2];
    return this.restToCard(this.toCard([a, b, c]));
  }
  // a direction in the world -> card frame
  dirToCard(d: ArrayLike<number>): V3 {
    const M = this.host.holder().elements, R = this.host.frame().rotation;
    // the holder turns only: its transpose takes the world to the page's axes
    const p = [M[0] * d[0] + M[1] * d[1] + M[2] * d[2], M[4] * d[0] + M[5] * d[1] + M[6] * d[2], M[8] * d[0] + M[9] * d[1] + M[10] * d[2]];
    const v: V3 = [R[0] * p[0] + R[3] * p[1] + R[6] * p[2], R[1] * p[0] + R[4] * p[1] + R[7] * p[2], R[2] * p[0] + R[5] * p[1] + R[8] * p[2]];
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
  }
  // a direction of the card frame -> the world
  dirToWorld(v: ArrayLike<number>): number[] {
    const R = this.host.frame().rotation, M = this.host.holder().elements;
    const p = [R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];
    const w = [M[0] * p[0] + M[4] * p[1] + M[8] * p[2], M[1] * p[0] + M[5] * p[1] + M[9] * p[2], M[2] * p[0] + M[6] * p[1] + M[10] * p[2]];
    const l = Math.hypot(w[0], w[1], w[2]) || 1;
    return [w[0] / l, w[1] / l, w[2] / l];
  }
  // the 4 x 4 matrix (three.js, column-major elements) that takes the card frame to the world
  cardToWorld(THREE: any): any {
    const f = this.host.frame(), s = f.scale, R = f.rotation, l = this.host.lift(), r = this.host.rootOffset(), h = this.rootHead();
    // the page of x + h
    const o = [0, 1, 2].map((k) => f.offset[k] + s * (R[k * 3] * h[0] + R[k * 3 + 1] * h[1] + R[k * 3 + 2] * h[2]));
    const m = new THREE.Matrix4().set(s * R[0], s * R[1], s * R[2], o[0] + r[0], s * R[3], s * R[4], s * R[5], o[1] + l + r[1], s * R[6], s * R[7], s * R[8], o[2] + r[2], 0, 0, 0, 1);
    return new THREE.Matrix4().multiplyMatrices(this.host.holder(), m);
  }

  // each bone's turn C = reference rest^T for the body shown (identity without the page's orientation tables)
  turns(): ArrayLike<number> {
    const C = this.host.restTurns(), n = this.host.bones().length;
    if (C) return C;
    const I = new Float64Array(n * 9);
    for (let b = 0; b < n; b++) { I[b * 9] = 1; I[b * 9 + 4] = 1; I[b * 9 + 8] = 1; }
    return I;
  }

  // the card's local-ref turns D (B x 9, Anny's axes) of the page's pose: D_b = C_parent R_b C_b^T (pose_reference.ts)
  cardRotations(q: Float32Array = this.host.pose()): number[][] {
    const n = q.length / 4, R: number[][] = [];
    for (let i = 0; i < n; i++) R.push(toCardRot(quatToMat(q[i * 4], q[i * 4 + 1], q[i * 4 + 2], q[i * 4 + 3])));
    return pageToRef(R, this.turns(), this.host.parents());
  }
  // the page's quaternions of local-ref turns D: R_b = C_parent^T D_b C_b
  pageQuaternions(D: number[][]): Float32Array {
    const R = refToPage(D, this.turns(), this.host.parents()), q = new Float32Array(R.length * 4);
    R.forEach((Rb, i) => q.set(matToQuat(toPageRot(Rb)), i * 4));
    return q;
  }

  // the reference pose of the body shown (Anny's rest axes): each bone's map from its rest frame to its reference one,
  // [C_b, h_ref - C_b h_b] (PoseRig.bind = ref rest^-1), and the reference heads h_ref (the page's rest rig posed by
  // R_b = C_parent^T C_b, pose_reference.referencePose)
  reference(): { bind: number[][]; heads: V3[] } {
    const H = this.host.restHeads().map((h) => this.toCard(h)), parents = this.host.parents(), C = this.turns();
    const R = referencePose(C, parents), G: number[][] = new Array(H.length), heads: V3[] = new Array(H.length);
    const order: number[] = [], seen = new Set<number>();
    const visit = (b: number) => { if (seen.has(b)) return; if (parents[b] >= 0) visit(parents[b]); seen.add(b); order.push(b); };
    H.forEach((_, b) => visit(b));
    for (const b of order) {
      // the local turn about the rest head, then the parent's map: G_b = G_p [R_b, h_b - R_b h_b]
      const r = R[b], h = H[b], t = [0, 1, 2].map((k) => h[k] - (r[k * 3] * h[0] + r[k * 3 + 1] * h[1] + r[k * 3 + 2] * h[2]));
      const L = [r[0], r[1], r[2], t[0], r[3], r[4], r[5], t[1], r[6], r[7], r[8], t[2], 0, 0, 0, 1];
      const p = parents[b];
      G[b] = p >= 0 ? mul4(G[p], L) : L;
      const g = G[b];
      heads[b] = [0, 1, 2].map((k) => g[k * 4] * h[0] + g[k * 4 + 1] * h[1] + g[k * 4 + 2] * h[2] + g[k * 4 + 3]) as V3;
    }
    return { bind: G, heads };
  }

  // the rig in the space of the turns (Python's PoseRig with an identity placement): the reference heads, and the
  // frame of the root's turn, x_card = D_root (x - h_root), moved by the root's shift (card frame, stage 1 of Points)
  rig(rotations?: number[][], shift: ArrayLike<number> = [0, 0, 0]): RigSpace {
    const ref = this.reference(), root = this.host.root();
    const Dr = (rotations || this.cardRotations())[root], h = ref.heads[root];
    const t = [0, 1, 2].map((r) => shift[r] - (Dr[r * 3] * h[0] + Dr[r * 3 + 1] * h[1] + Dr[r * 3 + 2] * h[2]));
    return {
      labels: this.host.bones(), parents: this.host.parents(), heads: ref.heads,
      frame: [[Dr[0], Dr[1], Dr[2], t[0]], [Dr[3], Dr[4], Dr[5], t[1]], [Dr[6], Dr[7], Dr[8], t[2]], [0, 0, 0, 1]],
    };
  }

  // the skin of Anny's rows from the page's coarse body (rest positions in the card frame, the coarse skin's 8 weights)
  skin(): SkinSource {
    const b = this.host.body(), rows = this.host.annyRows(), index = new Map<number, number>();
    if (rows) for (let c = 0; c < rows.length; c++) index.set(rows[c], c);
    const at = (row: number) => { const c = rows ? index.get(row) : row; if (c === undefined) throw new Error(`The page has no row ${row} of Anny's mesh.`); return c; };
    const cache = new Map<number, V3>();
    const bind = this.reference().bind;
    return {
      bind: (b) => bind[b] || null,
      rest: (row) => { let v = cache.get(row); if (!v) { const c = at(row); v = this.toCard([b.coarse[c * 3], b.coarse[c * 3 + 1], b.coarse[c * 3 + 2]]); cache.set(row, v); } return v; },
      skin: (row) => { const c = at(row), K = b.skin; const bones: number[] = [], weights: number[] = [];
        for (let k = 0; k < 8; k++) { const w = K[c * 16 + 8 + k]; if (w) { bones.push(K[c * 16 + k]); weights.push(w / 255); } }
        return { bones, weights }; },
    };
  }
}

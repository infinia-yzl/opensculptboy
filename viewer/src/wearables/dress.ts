// OpenSculptBoy
// Apache License, Version 2.0
//
// Dressing a body in the page, as opensculptboy.wearables (bind.evaluate, bind.body_weights, fit.dress and cover)
// computes it: surface bindings give back each garment vertex from a point of the skin, its frame and its offset
// scaled by the local edge length; rigid bindings follow one bone; the skin weights come from the body at the bound
// points; the push-out moves each piece outside the body and the lower layers; the cover masks and ray tests give the
// skin under the pieces. The body is any frame that is a similarity of Anny's (the page's frame: Y up, scaled by the
// authoring rig's scale), and `unit` gives its length of one metre. No DOM and no three.js: a worker can run it.

import { adjacency, frames, norm3, QUERY_SLICE, rayHitsSteps, runSteps, Surface, unitRows, vertexNormals, vertexScales, type F64, type I32, type Steps } from './geometry.ts';
import type { Wearable } from './package.ts';

export const GAP = 0.001;        // metres between a piece and what lies under it
export const DEPTH = 0.03;       // metres: a vertex deeper inside a surface than this is left alone
export const PUSH_ROUNDS = 3;
export const FADE_RINGS = 2;
export const RAY_REACH = 0.03;   // metres
export const MAX_INFLUENCES = 4;

// a similarity of Anny's frame: p = scale * R p_anny + offset (R row-major)
export interface Frame { scale: number; rotation: number[]; offset: number[] }
export const IDENTITY_FRAME: Frame = { scale: 1, rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], offset: [0, 0, 0] };
export function fromAnny(frame: Frame, v: F64, rows = v.length / 3): F64 {
  const R = frame.rotation, s = frame.scale, o = frame.offset, out = new Float64Array(rows * 3);
  for (let i = 0; i < rows; i++) {
    const x = v[i * 3], y = v[i * 3 + 1], z = v[i * 3 + 2];
    for (let r = 0; r < 3; r++) out[i * 3 + r] = s * (R[r * 3] * x + R[r * 3 + 1] * y + R[r * 3 + 2] * z) + o[r];
  }
  return out;
}

export interface BonePose { R: F64; t: F64 }   // the bone's rest rotation (row-major) and head in the body's frame

// the skin of a body on Anny's rows: rest vertices, the skin triangles, the skin weights (K per row) and the bones
export class BodySurface {
  vertices: F64; n: number; triangles: I32; weights: F64; indices: I32; K: number; bones: string[]; frame: Frame;
  pose: (bone: number, body: BodySurface, wearable: Wearable) => BonePose;
  private _surface: Surface | null = null;
  constructor(o: { vertices: F64; triangles: I32; weights: F64; indices: I32; K: number; bones: string[]; frame: Frame; pose: (bone: number, body: BodySurface, wearable: Wearable) => BonePose }) {
    this.vertices = o.vertices; this.n = o.vertices.length / 3; this.triangles = o.triangles; this.weights = o.weights; this.indices = o.indices;
    this.K = o.K; this.bones = o.bones; this.frame = o.frame; this.pose = o.pose;
  }
  get surface(): Surface { return this._surface || (this._surface = new Surface(this.vertices, this.triangles)); }
  get normals(): F64 { return this.surface.N; }
  get scales(): F64 { return this.surface.scales; }
  get unit(): number { return this.frame.scale; }
  // the same skin with other rest vertices (the bones stay)
  withVertices(vertices: F64): BodySurface {
    return new BodySurface({ vertices, triangles: this.triangles, weights: this.weights, indices: this.indices, K: this.K, bones: this.bones, frame: this.frame, pose: this.pose });
  }
}

// ------------------------------------------------------------------ evaluation and skin weights
export function evaluate(w: Wearable, body: BodySurface): { vertices: F64; normals: F64 } {
  const b = w.binding;
  if (b.method === 'rigid') {
    const bone = body.bones.indexOf(b.bone!);
    if (bone < 0) throw new Error(`The body's rig lacks the bone ${b.bone}.`);
    const { R, t } = body.pose(bone, body, w), L = b.local!, Nn = b.normal!, n = L.length / 3, s = body.unit;
    const vertices = new Float64Array(n * 3), normals = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) for (let r = 0; r < 3; r++) {
      const o = i * 3;
      vertices[o + r] = s * (R[r * 3] * L[o] + R[r * 3 + 1] * L[o + 1] + R[r * 3 + 2] * L[o + 2]) + t[r];
      normals[o + r] = R[r * 3] * Nn[o] + R[r * 3 + 1] * Nn[o + 1] + R[r * 3 + 2] * Nn[o + 2];
    }
    return { vertices, normals: unitRows(normals) };
  }
  const { point, axes, scale } = frames(body.vertices, body.normals, body.scales, b.corners!, b.barycentric!);
  const O = b.offset!, Nn = b.normal!, n = O.length / 3, vertices = new Float64Array(n * 3), normals = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    const o = i * 3, q = i * 9, s = scale[i];
    for (let k = 0; k < 3; k++) {
      vertices[o + k] = point[o + k] + s * (axes[q + k] * O[o] + axes[q + 3 + k] * O[o + 1] + axes[q + 6 + k] * O[o + 2]);
      normals[o + k] = axes[q + k] * Nn[o] + axes[q + 3 + k] * Nn[o + 1] + axes[q + 6 + k] * Nn[o + 2];
    }
  }
  return { vertices, normals: unitRows(normals) };
}

const round9 = (x: number) => { const y = x * 1e9, r = Math.round(y); return (Math.abs(y - Math.trunc(y)) === 0.5 && r % 2 !== 0 ? r - 1 : r) / 1e9; };

// the skin weights of the garment's vertices (MAX_INFLUENCES per vertex, renormalised; bone 0 for empty slots), as
// opensculptboy.wearables.bind.body_weights
export function bodyWeights(w: Wearable, body: BodySurface, k = MAX_INFLUENCES): { weights: Float32Array; joints: Uint16Array } {
  const b = w.binding, n = w.garment.count, weights = new Float32Array(n * k), joints = new Uint16Array(n * k);
  const finish = (i: number, ws: number[], js: number[]) => {
    let sum = 0;
    for (let j = 0; j < k; j++) sum += ws[j] || 0;
    sum = Math.max(sum, 1e-12);
    for (let j = 0; j < k; j++) { const x = (ws[j] || 0) / sum; weights[i * k + j] = x; joints[i * k + j] = x > 0 ? js[j] : 0; }
  };
  if (b.skinWeights) {
    const names = b.jointNames || [], index = names.map((nm) => body.bones.indexOf(nm));
    const missing = names.filter((_, i) => index[i] < 0);
    if (missing.length) throw new Error(`The body's rig lacks the garment's bones ${missing.join(', ')}.`);
    const K = b.skinK!;
    for (let i = 0; i < n; i++) {
      const cols = Array.from({ length: K }, (_, c) => c).sort((x, y) => b.skinWeights![i * K + y] - b.skinWeights![i * K + x] || x - y).slice(0, k);
      finish(i, cols.map((c) => b.skinWeights![i * K + c]), cols.map((c) => index[b.skinJoints![i * K + c]]));
    }
    return { weights, joints };
  }
  if (b.method === 'rigid') {
    const bone = body.bones.indexOf(b.bone!);
    for (let i = 0; i < n; i++) { weights[i * k] = 1; joints[i * k] = bone; }
    return { weights, joints };
  }
  const C = b.corners!, B = b.barycentric!, K = body.K, W = body.weights, I = body.indices;
  const dense = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    dense.clear();
    for (let c = 0; c < 3; c++) {
      const row = C[i * 3 + c], u = B[i * 3 + c];
      for (let q = 0; q < K; q++) { const bone = I[row * K + q], x = W[row * K + q] * u; dense.set(bone, (dense.get(bone) || 0) + x); }
    }
    // the largest values, compared at 1e-9, ties by bone index (opensculptboy.wearables.bind.strongest)
    const order = [...dense.entries()].map(([bone, x]) => [bone, x, round9(x)]).sort((p, q) => q[2] - p[2] || p[0] - q[0]).slice(0, k);
    finish(i, order.map((p) => p[1]), order.map((p) => p[0]));
  }
  return { weights, joints };
}

// ------------------------------------------------------------------ rigid poses in the page
// The page has the bone heads of every body but no rest rotations. A rigid piece then turns with its bone's skin: the
// rotation that carries the binding body's vertices under that bone onto the current body's (weighted by the bone's
// skin weights) times the bone's rotation on the binding body, which the package gives back: its garment vertices are
// the bone's rest frame applied to the binding's local vertices.
export function rotationBetween(X: F64, Y: F64, w?: F64): F64 {
  // Horn's quaternion method: the rotation R that minimises sum w |R (x - cx) - (y - cy)|^2
  const n = X.length / 3;
  let W = 0;
  const cx = [0, 0, 0], cy = [0, 0, 0];
  for (let i = 0; i < n; i++) { const wi = w ? w[i] : 1; W += wi; for (let k = 0; k < 3; k++) { cx[k] += wi * X[i * 3 + k]; cy[k] += wi * Y[i * 3 + k]; } }
  if (!(W > 0)) return Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  for (let k = 0; k < 3; k++) { cx[k] /= W; cy[k] /= W; }
  const S = new Float64Array(9);
  for (let i = 0; i < n; i++) {
    const wi = w ? w[i] : 1;
    if (!wi) continue;
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) S[a * 3 + b] += wi * (X[i * 3 + a] - cx[a]) * (Y[i * 3 + b] - cy[b]);
  }
  const [xx, xy, xz, yx, yy, yz, zx, zy, zz] = S;
  const N = [[xx + yy + zz, yz - zy, zx - xz, xy - yx], [yz - zy, xx - yy - zz, xy + yx, zx + xz],
    [zx - xz, xy + yx, -xx + yy - zz, yz + zy], [xy - yx, zx + xz, yz + zy, -xx - yy + zz]];
  const q = largestEigenvector(N);
  const [w0, x, y, z] = q;
  return Float64Array.from([1 - 2 * (y * y + z * z), 2 * (x * y - z * w0), 2 * (x * z + y * w0), 2 * (x * y + z * w0), 1 - 2 * (x * x + z * z),
    2 * (y * z - x * w0), 2 * (x * z - y * w0), 2 * (y * z + x * w0), 1 - 2 * (x * x + y * y)]);
}
function largestEigenvector(A: number[][]): number[] {
  // Jacobi rotations of a symmetric 4 x 4 matrix
  const a = A.map((r) => r.slice()), v: number[][] = [0, 1, 2, 3].map((i) => [0, 1, 2, 3].map((j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) off += a[p][q] * a[p][q];
    if (off < 1e-30) break;
    for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 4; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
      for (let k = 0; k < 4; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < 4; k++) { const vkp = v[k][p], vkq = v[k][q]; v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq; }
    }
  }
  let best = 0;
  for (let i = 1; i < 4; i++) if (a[i][i] > a[best][best]) best = i;
  const q = v.map((r) => r[best]), l = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) || 1;
  return q.map((x) => x / l);
}
const mul3 = (A: F64, B: F64) => {
  const out = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c];
  return out;
};
// the rest pose of a rigid piece's bone on a body: `head` is the bone's head on that body
export function estimatedPose(w: Wearable, body: BodySurface, head: ArrayLike<number>): BonePose {
  const b = w.binding, bone = body.bones.indexOf(b.bone!), s = body.unit;
  const local = b.local!.map((x) => x * s), garment = fromAnny(body.frame, w.garment.positions);
  const Rb = rotationBetween(local, garment);
  const bind = fromAnny(body.frame, b.bodyVertices, body.n), wt = new Float64Array(body.n);
  for (let i = 0; i < body.n; i++) for (let q = 0; q < body.K; q++) if (body.indices[i * body.K + q] === bone) wt[i] += body.weights[i * body.K + q];
  const Q = rotationBetween(bind, body.vertices, wt);
  return { R: mul3(Q, Rb), t: Float64Array.from(head) };
}

// ------------------------------------------------------------------ the push-out
// the adjacency of each garment, built once
const ADJACENCY = new WeakMap<I32, ReturnType<typeof adjacency>>();
function fade(D: F64, moved: Uint8Array, T: I32): F64 {
  if (!moved.some((x) => x)) return D;
  const n = moved.length;
  let A = ADJACENCY.get(T);
  if (!A || A.degree.length !== n) { A = adjacency(n, T); ADJACENCY.set(T, A); }
  let region = moved.slice();
  for (let r = 0; r < FADE_RINGS; r++) {
    const next = region.slice();
    for (let i = 0; i < n; i++) if (!region[i]) for (let k = A.start[i]; k < A.start[i + 1]; k++) if (region[A.ids[k]]) { next[i] = 1; break; }
    region = next;
  }
  let out = D.slice();
  for (let it = 0; it < FADE_RINGS * 4; it++) {
    const next = out.slice();
    for (let i = 0; i < n; i++) {
      if (!region[i] || moved[i]) continue;
      let x = 0, y = 0, z = 0;
      for (let k = A.start[i]; k < A.start[i + 1]; k++) { const j = A.ids[k] * 3; x += out[j]; y += out[j + 1]; z += out[j + 2]; }
      const d = Math.max(A.degree[i], 1);
      next[i * 3] = 0.5 * (x / d); next[i * 3 + 1] = 0.5 * (y / d); next[i * 3 + 2] = 0.5 * (z / d);
    }
    out = next;
  }
  return out;
}

function* pushOut(V: F64, N: F64, T: I32, colliders: Surface[], targets: F64[], unit: number, slice: number) {
  const start = V.slice(), n = V.length / 3, moved = new Uint8Array(n);
  for (let round = 0; round <= PUSH_ROUNDS; round++) {
    for (let c = 0; c < colliders.length; c++) {
      // a vertex farther than DEPTH from every triangle never moves, so its query may stop early
      const { signed, found, normal } = yield* colliders[c].signedDistanceSteps(V, DEPTH * unit, slice), target = targets[c];
      for (let i = 0; i < n; i++) {
        if (signed[i] < target[i] - 1e-7 * unit && found.distance[i] < DEPTH * unit) {
          const d = target[i] - signed[i];
          V[i * 3] += d * normal[i * 3]; V[i * 3 + 1] += d * normal[i * 3 + 1]; V[i * 3 + 2] += d * normal[i * 3 + 2];
          moved[i] = 1;
        }
      }
      yield;
    }
    if (round < PUSH_ROUNDS) {
      const D = new Float64Array(V.length);
      for (let i = 0; i < V.length; i++) D[i] = V[i] - start[i];
      const F = fade(D, moved, T);
      for (let i = 0; i < V.length; i++) V[i] = start[i] + F[i];
      yield;
    }
  }
  const shift = new Float64Array(n);
  let pushed = 0, largest = 0;
  for (let i = 0; i < n; i++) {
    shift[i] = norm3(V[i * 3] - start[i * 3], V[i * 3 + 1] - start[i * 3 + 1], V[i * 3 + 2] - start[i * 3 + 2]);
    if (shift[i] > 1e-7 * unit) pushed++;
    largest = Math.max(largest, shift[i]);
  }
  let normals = N;
  if (moved.some((x) => x) && T.length) {
    const fresh = vertexNormals(V, T, n);
    normals = N.slice();
    for (let i = 0; i < n; i++) if (shift[i] > 1e-9 * unit) { normals[i * 3] = fresh[i * 3]; normals[i * 3 + 1] = fresh[i * 3 + 1]; normals[i * 3 + 2] = fresh[i * 3 + 2]; }
  }
  return { vertices: V, normals, pushed, pushMm: largest / unit * 1000 };
}

// ------------------------------------------------------------------ dressing
export interface WornPiece { ref: string; piece: string; outfit: string | null; wearable: Wearable; shown: boolean; hiddenGroups: string[]; hideableGroups: string[] }
export interface DressedPiece extends WornPiece { vertices: F64; normals: F64; weights: Float32Array; joints: Uint16Array; pushed: number; pushMm: number }

export function visibleTriangles(p: { wearable: Wearable; hiddenGroups: string[] }, opaque = false): I32 {
  const g = p.wearable.garment, hidden = new Set(p.hiddenGroups.map((h) => g.groups.indexOf(h)));
  const keep: number[] = [];
  for (let t = 0; t < g.triangles.length / 3; t++) {
    if (hidden.has(g.triangleGroup[t])) continue;
    if (opaque && (g.materials[g.triangleMaterial[t]]?.alphaMode || 'OPAQUE') !== 'OPAQUE') continue;
    keep.push(g.triangles[t * 3], g.triangles[t * 3 + 1], g.triangles[t * 3 + 2]);
  }
  return Int32Array.from(keep);
}

function sameRows(a: F64, b: F64): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// the rest vertices and normals of a lower piece on `home`, the binding body of an upper piece, or null where they are
// unknown (opensculptboy.wearables.fit._rest_on). `home` carries the bones of `body`, the body being dressed, so a
// rigid piece takes its own garment vertices on its own binding body, and has no known place on another binding body.
function restOn(w: Wearable, home: BodySurface, body: BodySurface): { vertices: F64; normals: F64 } | null {
  if (w.binding.method === 'surface' || sameRows(home.vertices, body.vertices)) return evaluate(w, home);
  if (sameRows(fromAnny(home.frame, w.binding.bodyVertices, home.n), home.vertices)) {
    const R = home.frame.rotation, Nn = w.garment.normals, normals = new Float64Array(Nn.length);
    for (let i = 0; i < Nn.length; i += 3) for (let r = 0; r < 3; r++) normals[i + r] = R[r * 3] * Nn[i] + R[r * 3 + 1] * Nn[i + 1] + R[r * 3 + 2] * Nn[i + 2];
    return { vertices: fromAnny(home.frame, w.garment.positions), normals: unitRows(normals) };
  }
  return null;
}

// what a dress keeps between calls: the binding bodies and the push targets that depend on them alone
export class DressCache {
  homes = new Map<Wearable, { home: BodySurface; vertices: F64; target: F64; lower: Map<string, F64> }>();
  // the skin weights of each piece: they depend on the binding and the body's skin weights alone, which a slider
  // edit leaves as they are
  weights = new Map<Wearable, { source: F64; weights: Float32Array; joints: Uint16Array }>();
}

// `slice`: the garment vertices per slice of a closest-point query in dressSteps
export interface DressOptions { push?: boolean; gap?: number; cache?: DressCache; timing?: Record<string, number>; slice?: number }

// evaluate the worn pieces on a body from the lowest layer up (opensculptboy.wearables.fit.dress); pieces that do not
// show are evaluated without a push
export function dress(body: BodySurface, pieces: WornPiece[], opts: DressOptions = {}): DressedPiece[] {
  return runSteps(dressSteps(body, pieces, opts));
}
// the same dress in steps that yield between slices of the work (the page runs them between frames); the slices change
// no number of the result. With steps, the times of `timing` include the waits between the slices.
export function* dressSteps(body: BodySurface, pieces: WornPiece[], opts: DressOptions = {}): Steps<DressedPiece[]> {
  const push = opts.push !== false, unit = body.unit, gap = (opts.gap ?? GAP) * unit, cache = opts.cache || new DressCache();
  const slice = opts.slice ?? QUERY_SLICE;
  const timing = opts.timing || {};
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const order = pieces.map((_, i) => i).sort((a, b) => pieces[a].wearable.layer - pieces[b].wearable.layer || a - b);
  const out: DressedPiece[] = new Array(pieces.length), placed: [number, DressedPiece][] = [];
  for (const k of order) {
    const worn = pieces[k], w = worn.wearable;
    let t0 = now();
    let { vertices, normals } = evaluate(w, body);
    timing.evaluate = (timing.evaluate || 0) + now() - t0;
    yield;
    let pushed = 0, pushMm = 0;
    if (push && worn.shown && w.binding.method === 'surface') {
      t0 = now();
      let h = cache.homes.get(w);
      if (!h) {
        const home = body.withVertices(fromAnny(body.frame, w.binding.bodyVertices, body.n)), hv = evaluate(w, home).vertices;
        const sd = (yield* home.surface.signedDistanceSteps(hv, Infinity, slice)).signed;
        h = { home, vertices: hv, target: sd.map((x) => Math.min(x, gap)), lower: new Map() };
        cache.homes.set(w, h);
      }
      const colliders = [body.surface], targets = [h.target];
      for (const [layer, lower] of placed) {
        if (layer >= w.layer) continue;
        const tris = visibleTriangles(lower);
        if (!tris.length) continue;
        colliders.push(new Surface(lower.vertices, tris, lower.normals));
        const key = `${lower.ref}#${lower.hiddenGroups.slice().sort().join(',')}`;
        let target = h.lower.get(key);
        if (!target) {
          const lh = restOn(lower.wearable, h.home, body);
          target = lh
            ? (yield* new Surface(lh.vertices, tris, lh.normals).signedDistanceSteps(h.vertices, Infinity, slice)).signed.map((x) => Math.min(x, gap))
            : new Float64Array(h.vertices.length / 3).fill(gap);   // the relation on the binding body is unknown
          if (lower.wearable.binding.method === 'surface' || !sameRows(h.home.vertices, body.vertices)) h.lower.set(key, target);
        }
        targets.push(target);
      }
      const r = yield* pushOut(vertices, normals, w.garment.triangles, colliders, targets, unit, slice);
      vertices = r.vertices; normals = r.normals; pushed = r.pushed; pushMm = r.pushMm;
      timing.push = (timing.push || 0) + now() - t0;
    }
    t0 = now();
    let sw = cache.weights.get(w);
    if (!sw || sw.source !== body.weights) { sw = { source: body.weights, ...bodyWeights(w, body) }; cache.weights.set(w, sw); }
    const { weights, joints } = sw;
    timing.weights = (timing.weights || 0) + now() - t0;
    const piece: DressedPiece = { ...worn, hiddenGroups: [...worn.hiddenGroups], hideableGroups: [...worn.hideableGroups], vertices, normals, weights, joints, pushed, pushMm };
    out[k] = piece;
    if (worn.shown) placed.push([w.layer, piece]);
  }
  return out;
}

// ------------------------------------------------------------------ covered skin
export function rayCover(body: BodySurface, piece: DressedPiece, reach = RAY_REACH): Uint8Array {
  return runSteps(rayCoverSteps(body, piece, reach));
}
function rayCoverSteps(body: BodySurface, piece: DressedPiece, reach = RAY_REACH): Steps<Uint8Array> {
  return rayHitsSteps(body.vertices, body.normals, piece.vertices, visibleTriangles(piece, true), reach * body.unit);
}
// the cover mask of a dressed piece on the body's rows (opensculptboy.wearables.cover.piece_cover)
export function pieceCover(body: BodySurface, piece: DressedPiece): Uint8Array {
  return runSteps(pieceCoverSteps(body, piece));
}
function* pieceCoverSteps(body: BodySurface, piece: DressedPiece): Steps<Uint8Array> {
  const w = piece.wearable, n = body.n;
  if (!w.covers) return yield* rayCoverSteps(body, piece);
  const mask = w.covers.slice(0, n);
  for (const [group, extra] of Object.entries(w.groupCovers)) if (!piece.hiddenGroups.includes(group)) for (let i = 0; i < n; i++) mask[i] |= extra[i];
  const hidden = piece.hiddenGroups.filter((g) => !(g in w.groupCovers));
  if (w.coverRay || hidden.length) {
    const ray = yield* rayCoverSteps(body, piece, w.coverRay ? RAY_REACH : 2 * RAY_REACH);
    for (let i = 0; i < n; i++) mask[i] &= ray[i];
  }
  return mask;
}
export function outfitCover(body: BodySurface, pieces: DressedPiece[]): Uint8Array {
  return runSteps(outfitCoverSteps(body, pieces));
}
// the same mask in steps, one per piece
export function* outfitCoverSteps(body: BodySurface, pieces: DressedPiece[]): Steps<Uint8Array> {
  const mask = new Uint8Array(body.n);
  for (const p of pieces) if (p.shown) { const m = yield* pieceCoverSteps(body, p); for (let i = 0; i < body.n; i++) mask[i] |= m[i]; yield; }
  return mask;
}

export { runSteps, vertexScales, type Steps };

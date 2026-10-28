// OpenSculptBoy
// Apache License, Version 2.0
//
// The faces of the page without the DOM and three.js (docs/viewer.md, "Faces"), so that node runs them in the parity
// test (test/test_viewer_expressions.py):
//   - Anny's 52 ARKit facial actions as the exports write them (opensculptboy.viewer.face_tables): sparse offsets of
//     the coarse body, carried to the fine body by the page's own subdivision, sparse from level to level;
//   - jawOpen as the turn of the anny-mouth rig's jaw about its head with the corrective that completes it
//     (opensculptboy.mouth.jaw): at weight w the jaw turns by w times the turn and the corrective takes the weight w,
//     as the GLB animation "jawOpen.jaw" plays them, so that weight 1 gives Anny's jawOpen exactly;
//   - the face layer of the fine body: the shapes of a face over a copy of the body at rest, the normals of the moved
//     surface from the change of its triangles, and the range of vertices that an upload sends.

import type { FineSubdivision } from '../subdivision.ts';

export interface FaceMeta {
  format: string;
  actions: string[];
  starts: number[];
  counts: number[];
  step: number;
  cleared: string[];
  jaw: { action: string; rotation: number[]; angle_degrees: number; pivot: number[]; vertices: number };
  eye_turns: Record<string, Record<string, number[]>>;   // per action: per eye ('l', 'r') a rotation vector (page frame)
  tongue: { rest: number[]; motion: Record<string, { kind: 'rigid'; rotation: number[]; translation: number[] } | { kind: 'follow'; offsets: number[] }> };
  region: number;
  presets: Record<string, Record<string, number>>;
  visemes: Record<string, Record<string, number>>;
}

export interface FaceData {
  meta: FaceMeta;
  ids: Uint32Array;          // the coarse vertex of each sparse offset
  offsets: Int16Array;       // (total, 3) in steps of meta.step
  jawIds: Uint32Array;       // the coarse vertices that the jaw and the tongue chain move
  jawWeights: Float32Array;  // and their weights
}

// the buffers of the model data (face_action_ids, face_action_offsets: records of x, y, z and a pad; face_jaw: pairs of
// a vertex and a float32 weight)
export function decodeFaceData(meta: FaceMeta, ids: Uint32Array, offsets4: Int16Array, jaw: Uint32Array): FaceData {
  const n = offsets4.length / 4, offsets = new Int16Array(n * 3);
  for (let i = 0; i < n; i++) { offsets[i * 3] = offsets4[i * 4]; offsets[i * 3 + 1] = offsets4[i * 4 + 1]; offsets[i * 3 + 2] = offsets4[i * 4 + 2]; }
  const m = jaw.length / 2, jawIds = new Uint32Array(m), jawWeights = new Float32Array(m), f = new Float32Array(jaw.buffer, jaw.byteOffset, jaw.length);
  for (let i = 0; i < m; i++) { jawIds[i] = jaw[i * 2]; jawWeights[i] = f[i * 2 + 1]; }
  return { meta, ids, offsets, jawIds, jawWeights };
}

// ------------------------------------------------------------------ rotations
// the matrix (row-major 3x3) of a rotation vector
export function rotvecMatrix(x: number, y: number, z: number, out: Float64Array = new Float64Array(9)): Float64Array {
  const t = Math.hypot(x, y, z);
  if (t < 1e-12) { out.set([1, 0, 0, 0, 1, 0, 0, 0, 1]); return out; }
  const kx = x / t, ky = y / t, kz = z / t, c = Math.cos(t), s = Math.sin(t), v = 1 - c;
  out[0] = c + kx * kx * v; out[1] = kx * ky * v - kz * s; out[2] = kx * kz * v + ky * s;
  out[3] = ky * kx * v + kz * s; out[4] = c + ky * ky * v; out[5] = ky * kz * v - kx * s;
  out[6] = kz * kx * v - ky * s; out[7] = kz * ky * v + kx * s; out[8] = c + kz * kz * v;
  return out;
}

// the corrective c of a vertex for the full turn R1: (I + w (R1 - I)) c = target - w (R1 - I)(v - p)
// (opensculptboy.mouth.jaw.corrective_for); A = (1 - w) I + w R1 is solved by its adjugate
function corrective(R: Float64Array, w: number, tx: number, ty: number, tz: number, dx: number, dy: number, dz: number, out: Float32Array, o: number) {
  const ex = (R[0] - 1) * dx + R[1] * dy + R[2] * dz, ey = R[3] * dx + (R[4] - 1) * dy + R[5] * dz, ez = R[6] * dx + R[7] * dy + (R[8] - 1) * dz;
  const bx = tx - w * ex, by = ty - w * ey, bz = tz - w * ez;
  const a = 1 - w + w * R[0], b = w * R[1], c = w * R[2], d = w * R[3], e = 1 - w + w * R[4], f = w * R[5], g = w * R[6], h = w * R[7], k = 1 - w + w * R[8];
  const A = e * k - f * h, B = f * g - d * k, C = d * h - e * g, det = a * A + b * B + c * C || 1e-30;
  out[o] = (A * bx + (c * h - b * k) * by + (b * f - c * e) * bz) / det;
  out[o + 1] = (B * bx + (a * k - c * g) * by + (c * d - a * f) * bz) / det;
  out[o + 2] = (C * bx + (b * g - a * h) * by + (a * e - b * d) * bz) / det;
}

// ------------------------------------------------------------------ sparse subdivision
// The page's subdivision (subdivision.ts) on sparse fields: a field that is zero outside a few coarse vertices stays
// zero outside the rows whose stencils reach them, so each level computes only those rows. The transposes of the
// operators cover the rows that the coarse vertices of `mask` reach.
export class SparseSubdivision {
  sub: FineSubdivision;
  masks: Uint8Array[] = [];          // per level (0: the coarse body): the rows that the mask reaches
  tStart: Uint32Array[] = []; tRows: Uint32Array[] = [];
  private dense: Float32Array[] = []; private ch = 0; private seen: Uint8Array[] = [];
  constructor(sub: FineSubdivision, mask: Uint8Array) {
    this.sub = sub;
    let m = mask;
    this.masks.push(m);
    for (const op of sub.ops) {
      const out = new Uint8Array(op.rows), count = new Uint32Array(op.cols + 1);
      for (let r = 0; r < op.rows; r++) for (let e = op.start[r]; e < op.start[r + 1]; e++) if (m[op.index[e]]) { out[r] = 1; count[op.index[e] + 1]++; }
      for (let c = 0; c < op.cols; c++) count[c + 1] += count[c];
      const rows = new Uint32Array(count[op.cols]), fill = count.slice(0, op.cols);
      for (let r = 0; r < op.rows; r++) for (let e = op.start[r]; e < op.start[r + 1]; e++) { const c = op.index[e]; if (m[c]) rows[fill[c]++] = r; }
      this.tStart.push(count); this.tRows.push(rows); this.masks.push(out);
      m = out;
    }
    this.seen = this.masks.map((x) => new Uint8Array(x.length));
  }
  // the rows of the fine body that the mask reaches
  region(): Uint32Array {
    const m = this.masks[this.masks.length - 1], out: number[] = [];
    for (let i = 0; i < m.length; i++) if (m[i]) out.push(i);
    return Uint32Array.from(out);
  }
  // the field (ids: rows of the coarse body, values: ch per id) on the fine body, sorted by row; adds[i] adds a sparse
  // field on the rows of level i + 1 (a card's offsets of that level). Values outside the mask are left out.
  apply(ids: ArrayLike<number>, values: ArrayLike<number>, ch: number, adds?: ({ ids: ArrayLike<number>; values: ArrayLike<number> } | null)[]): { ids: Uint32Array; values: Float32Array } {
    const ops = this.sub.ops;
    if (this.ch !== ch) { this.dense = this.masks.map((x) => new Float32Array(x.length * ch)); this.ch = ch; }
    let X = this.dense[0], touched: number[] = [];
    const s0 = this.seen[0];
    for (let i = 0; i < ids.length; i++) {
      const v = ids[i];
      if (!this.masks[0][v]) continue;
      if (!s0[v]) { s0[v] = 1; touched.push(v); }
      for (let k = 0; k < ch; k++) X[v * ch + k] += values[i * ch + k];
    }
    for (let l = 0; l < ops.length; l++) {
      const op = ops[l], Y = this.dense[l + 1], seen = this.seen[l + 1], T = this.tStart[l], TR = this.tRows[l], rows: number[] = [];
      for (const c of touched) for (let q = T[c]; q < T[c + 1]; q++) { const r = TR[q]; if (!seen[r]) { seen[r] = 1; rows.push(r); } }
      for (const r of rows) {
        for (let k = 0; k < ch; k++) {
          let a = 0;
          for (let e = op.start[r]; e < op.start[r + 1]; e++) a += op.value[e] * X[op.index[e] * ch + k];
          Y[r * ch + k] = a;
        }
      }
      const a = adds && l < ops.length - 1 ? adds[l] : null;
      if (a) for (let i = 0; i < a.ids.length; i++) {
        const r = a.ids[i];
        if (r >= op.rows || !this.masks[l + 1][r]) continue;
        if (!seen[r]) { seen[r] = 1; rows.push(r); for (let k = 0; k < ch; k++) Y[r * ch + k] = 0; }
        for (let k = 0; k < ch; k++) Y[r * ch + k] += a.values[i * ch + k];
      }
      const sPrev = this.seen[l];
      for (const c of touched) { sPrev[c] = 0; for (let k = 0; k < ch; k++) X[c * ch + k] = 0; }
      touched = rows; X = Y;
    }
    const last = this.seen[this.seen.length - 1];
    const out = Uint32Array.from(touched).sort(), vals = new Float32Array(out.length * ch);
    for (let i = 0; i < out.length; i++) { const r = out[i]; last[r] = 0; for (let k = 0; k < ch; k++) { vals[i * ch + k] = X[r * ch + k]; X[r * ch + k] = 0; } }
    return { ids: out, values: vals };
  }
}

// ------------------------------------------------------------------ the rig of the face
// what the face needs of the page's body (body.ts AnnyBody)
export interface FaceBody {
  sub: FineSubdivision; nBody: number; nc: number; nFine: number; nWeld: number;
  coarse: Float32Array; quads: Uint32Array;
  // the head bone's rest head on this body and on anny's default body, and the head's scale (the page's body)
  joints?: Float32Array; jointsDefault?: Float32Array; headScale?: () => number; head?: number;
}

// a sparse shape of the fine body: offsets of its vertices, and (once computed, face/rig.ts FaceLayer.normalShape) the
// change of the normals at weight 1 on the vertices it turns
export interface Shape { idx: Uint32Array; d: Float32Array; nIdx?: Uint32Array; dn?: Float32Array }

// The facial actions and the jaw's turn on a body. The fine shapes of the actions follow from the coarse offsets by
// the sparse subdivision, once per action when it first shows; the jaw's pivot and corrective follow each body.
export class FaceRig {
  data: FaceData;
  body: FaceBody;
  names: string[];
  index = new Map<string, number>();
  jaw: number;                         // the action that the jaw's turn plays (jawOpen), or -1
  sparse: SparseSubdivision;
  private fine: (Shape | null)[];
  R1: Float64Array;                    // the jaw's full turn
  rotation: number[];                  // its rotation vector
  pivot = [0, 0, 0];
  coarseJaw: Float32Array;             // per coarse vertex the jaw's weight
  fineJaw: { idx: Uint32Array; w: Float32Array } | null = null;
  coarseCorrective: Float32Array | null = null;   // per coarse vertex (nc * 3), for the coarse body's own face
  constructor(body: FaceBody, data: FaceData) {
    this.body = body; this.data = data;
    const m = data.meta;
    this.names = m.actions.slice();
    this.names.forEach((n, i) => this.index.set(n, i));
    this.jaw = m.jaw ? this.names.indexOf(m.jaw.action) : -1;
    this.rotation = m.jaw ? m.jaw.rotation.slice() : [0, 0, 0];
    this.R1 = rotvecMatrix(this.rotation[0], this.rotation[1], this.rotation[2]);
    this.coarseJaw = new Float32Array(body.nc);
    for (let i = 0; i < data.jawIds.length; i++) this.coarseJaw[data.jawIds[i]] = data.jawWeights[i];
    // the coarse vertices of the face: every action's and the jaw's, and two rings of quads around them (a card's own
    // mouth poses lie there)
    const mask = new Uint8Array(body.nBody);
    for (let i = 0; i < data.ids.length; i++) if (data.ids[i] < body.nBody) mask[data.ids[i]] = 1;
    for (let i = 0; i < data.jawIds.length; i++) if (data.jawIds[i] < body.nBody) mask[data.jawIds[i]] = 1;
    const Q = body.quads, nq = Q.length / 4;
    for (let ring = 0; ring < 2; ring++) {
      const grow = mask.slice();
      for (let f = 0; f < nq; f++) {
        let any = false;
        for (let k = 0; k < 4; k++) if (mask[Q[f * 4 + k]]) { any = true; break; }
        if (any) for (let k = 0; k < 4; k++) grow[Q[f * 4 + k]] = 1;
      }
      mask.set(grow);
    }
    this.sparse = new SparseSubdivision(body.sub, mask);
    this.fine = this.names.map(() => null);
  }
  get coarseMask(): Uint8Array { return this.sparse.masks[0]; }

  // the offsets of action a on the coarse body (sparse: ids and values, page frame)
  coarseShape(a: number): Shape {
    const m = this.data.meta, s = m.starts[a], n = m.counts[a], st = m.step;
    const idx = this.data.ids.subarray(s, s + n), d = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) d[i] = this.data.offsets[s * 3 + i] * st;
    return { idx, d };
  }
  // the offsets of action a on the fine body
  fineShape(a: number): Shape {
    let f = this.fine[a];
    if (!f) {
      const c = this.coarseShape(a), r = this.sparse.apply(c.idx, c.d, 3);
      f = this.fine[a] = { idx: r.ids, d: r.values };
    }
    return f;
  }
  // every fine shape, for a page that plays faces often (the work of a few hundred milliseconds)
  prepare() { for (let a = 0; a < this.names.length; a++) this.fineShape(a); }
  prepared(a: number) { return !!this.fine[a]; }

  // the jaw on the current body: its pivot from the coarse body, and the correctives of the coarse and the fine body
  // (fine: the fine positions at rest, nFine * 3)
  rebase(fine: Float32Array) {
    const m = this.data.meta;
    if (this.jaw < 0 || !m.jaw) return;
    // the pivot of anny's default body, carried with the head to the current body (a move and a uniform scale, as the
    // page carries the eyes and the hair); the corrective takes up the rest, so that weight 1 gives jawOpen exactly
    const C = this.body.coarse, p0 = m.jaw.pivot, B = this.body;
    if (B.joints && B.jointsDefault && B.headScale && B.head !== undefined && B.head >= 0) {
      const h = B.head * 3, k = B.headScale();
      this.pivot = [0, 1, 2].map((c) => B.joints![h + c] + k * (p0[c] - B.jointsDefault![h + c]));
    } else this.pivot = p0.slice();
    const p = this.pivot, R = this.R1;
    // the coarse corrective
    const J = this.coarseShape(this.jaw), nc = this.body.nc, cc = this.coarseCorrective || (this.coarseCorrective = new Float32Array(nc * 3));
    cc.fill(0);
    const target = new Float32Array(nc * 3);
    for (let i = 0; i < J.idx.length; i++) for (let k = 0; k < 3; k++) target[J.idx[i] * 3 + k] = J.d[i * 3 + k];
    for (let v = 0; v < nc; v++) {
      const wv = this.coarseJaw[v], o = v * 3;
      if (wv === 0 && target[o] === 0 && target[o + 1] === 0 && target[o + 2] === 0) continue;
      corrective(R, wv, target[o], target[o + 1], target[o + 2], C[o] - p[0], C[o + 1] - p[1], C[o + 2] - p[2], cc, o);
    }
    // the fine weights (the subdivision of the coarse weights) and the fine corrective over the jaw's and jawOpen's rows
    if (!this.fineJaw) {
      const ids2: number[] = [], vals: number[] = [];
      for (let i = 0; i < this.data.jawIds.length; i++) if (this.data.jawIds[i] < this.body.nBody) { ids2.push(this.data.jawIds[i]); vals.push(this.data.jawWeights[i]); }
      const r = this.sparse.apply(ids2, vals, 1);
      this.fineJaw = { idx: r.ids, w: r.values };
    }
    const F = this.fineShape(this.jaw), fj = this.fineJaw, rows = mergeIds(F.idx, fj.idx);
    const wAt = new Map<number, number>(), tAt = new Map<number, number>();
    for (let i = 0; i < fj.idx.length; i++) wAt.set(fj.idx[i], fj.w[i]);
    for (let i = 0; i < F.idx.length; i++) tAt.set(F.idx[i], i);
    const d = new Float32Array(rows.length * 3);
    rows.forEach((v, i) => {
      const wv = wAt.get(v) ?? 0, t = tAt.get(v), o = v * 3;
      const tx = t === undefined ? 0 : F.d[t * 3], ty = t === undefined ? 0 : F.d[t * 3 + 1], tz = t === undefined ? 0 : F.d[t * 3 + 2];
      corrective(R, wv, tx, ty, tz, fine[o] - p[0], fine[o + 1] - p[1], fine[o + 2] - p[2], d, i * 3);
    });
    this.jawCorrective = { idx: rows, d };
  }
  jawCorrective: Shape | null = null;

  // the coarse body with a face (weights by action; the jaw's action as the turn), in place: for the parity test and
  // for the parts that follow the coarse body
  coarseFace(w: ArrayLike<number>, V: Float32Array) {
    for (let a = 0; a < this.names.length; a++) {
      const k = w[a];
      if (!k || a === this.jaw) continue;
      const s = this.coarseShape(a);
      for (let i = 0; i < s.idx.length; i++) for (let c = 0; c < 3; c++) V[s.idx[i] * 3 + c] += k * s.d[i * 3 + c];
    }
    const wj = this.jaw >= 0 ? w[this.jaw] : 0;
    if (!wj || !this.coarseCorrective) return;
    const cc = this.coarseCorrective;
    for (let i = 0; i < V.length; i++) V[i] += wj * cc[i];
    const R = rotvecMatrix(wj * this.rotation[0], wj * this.rotation[1], wj * this.rotation[2]), p = this.pivot;
    for (let v = 0; v < this.body.nc; v++) {
      const k = this.coarseJaw[v];
      if (!k) continue;
      turnPoint(V, v * 3, R, p, k);
    }
  }
  // the jaw's turn at weight wj (row-major 3x3)
  turn(wj: number, out: Float64Array = new Float64Array(9)) { return rotvecMatrix(wj * this.rotation[0], wj * this.rotation[1], wj * this.rotation[2], out); }
}

// p += k ((R - I)(p - pivot)) for the point at o
export function turnPoint(P: Float32Array | Float64Array, o: number, R: ArrayLike<number>, p: number[], k: number) {
  const dx = P[o] - p[0], dy = P[o + 1] - p[1], dz = P[o + 2] - p[2];
  P[o] += k * ((R[0] - 1) * dx + R[1] * dy + R[2] * dz);
  P[o + 1] += k * (R[3] * dx + (R[4] - 1) * dy + R[5] * dz);
  P[o + 2] += k * (R[6] * dx + R[7] * dy + (R[8] - 1) * dz);
}

// the sorted union of two sorted id lists
function mergeIds(a: Uint32Array, b: Uint32Array): Uint32Array {
  const out = new Uint32Array(a.length + b.length);
  let i = 0, j = 0, n = 0;
  while (i < a.length || j < b.length) {
    const x = i < a.length ? a[i] : Infinity, y = j < b.length ? b[j] : Infinity;
    if (x === y) { out[n++] = x; i++; j++; } else if (x < y) { out[n++] = x; i++; } else { out[n++] = y; j++; }
  }
  return out.slice(0, n);
}

// ------------------------------------------------------------------ the face layer of the fine body
// what the layer moves: the body's arrays (body.ts) and its welded triangles
export interface LayerBody extends FaceBody {
  pos: Float32Array; nrm: Float32Array; ns: Float32Array;
  weld: Uint32Array; weldShown: Uint32Array | null; hide: boolean; copyOf: Uint32Array;
}

// extra shapes over the actions: a card's eye targets and mouth poses (fine shapes of the page frame)
export interface ShapeGroup { shapes: Shape[]; weights: Float32Array; after: boolean }

export class FaceLayer {
  rig: FaceRig;
  body: LayerBody;
  // the vertices that a face can move (the region), their positions at rest, and the normals at rest of the region and
  // of its ring of neighbours
  region = new Uint32Array(0); at = new Int32Array(0);
  bP = new Float32Array(0);
  nIds = new Uint32Array(0); nAt = new Int32Array(0); bN = new Float32Array(0); bS = new Float32Array(0); gBase = new Float32Array(0);
  tris = new Uint32Array(0); fBase = new Float32Array(0);   // the triangles with a vertex in the region and their normals at rest
  vtStart = new Uint32Array(0); vtTris = new Uint32Array(0);   // per region vertex its triangles (indices into tris)
  copies = new Uint32Array(0);                                // pairs (copy, source) of the seam copies among the normal vertices
  groups = new Map<string, ShapeGroup>();
  stageA = new Float32Array(0);                               // per region vertex the offset of the face before the extra shapes marked `after`
  private moved = new Uint32Array(0); private normals = new Uint32Array(0); private nMoved = 0; private nNormals = 0;
  private mark = new Uint8Array(0); private triMark = new Uint8Array(0); private triList = new Uint32Array(0); private acc = new Float32Array(0);
  ready = false;
  rebaseMs = 0;
  private scratch = new Float32Array(0);   // the body at rest, for the normals of single shapes
  outside = 0;   // vertices of extra shapes outside the region (left out)
  constructor(rig: FaceRig, body: LayerBody) { this.rig = rig; this.body = body; }

  // after a new body: the copy of the rest, the jaw, and the region's triangles; returns the ms taken
  rebase(): number {
    const t0 = now(), b = this.body, n = b.nFine;
    if (this.at.length !== n) {
      const reach = this.rig.sparse.region();
      this.at = new Int32Array(n).fill(-1);
      const extra: number[] = [];
      for (const g of this.groups.values()) for (const s of g.shapes) for (const v of s.idx) extra.push(v);
      const mark = new Uint8Array(n);
      for (const v of reach) mark[v] = 1;
      for (const v of extra) if (v < n) mark[v] = 1;
      const ids: number[] = [];
      for (let v = 0; v < n; v++) if (mark[v]) ids.push(v);
      this.region = Uint32Array.from(ids);
      this.region.forEach((v, i) => { this.at[v] = i; });
      this.mark = new Uint8Array(n); this.acc = new Float32Array(n * 3);
      this.moved = new Uint32Array(n); this.normals = new Uint32Array(n);
      this.buildTriangles();
    }
    const R = this.region, P = b.pos;
    this.bP = new Float32Array(R.length * 3);
    for (let i = 0; i < R.length; i++) for (let k = 0; k < 3; k++) this.bP[i * 3 + k] = P[R[i] * 3 + k];
    const N = b.nrm, S = b.ns, NI = this.nIds;
    this.bN = new Float32Array(NI.length * 3); this.bS = new Float32Array(NI.length * 3);
    for (let i = 0; i < NI.length; i++) for (let k = 0; k < 3; k++) { this.bN[i * 3 + k] = N[NI[i] * 3 + k]; this.bS[i * 3 + k] = S[NI[i] * 3 + k]; }
    // the normals of the region's triangles at rest, and the summed normals of the normal vertices (every triangle of
    // the body counts for them)
    const T = this.tris, F = this.fBase = new Float32Array(T.length);
    for (let t = 0; t < T.length; t += 3) faceNormal(P, T[t], T[t + 1], T[t + 2], F, t);
    const G = this.gBase = new Float32Array(NI.length * 3), all = this.allTris(), nAt = this.nAt;
    for (let t = 0; t < all.length; t += 3) {
      const a = all[t], c1 = all[t + 1], c2 = all[t + 2];
      if (nAt[a] < 0 && nAt[c1] < 0 && nAt[c2] < 0) continue;
      const ax = P[a * 3], ay = P[a * 3 + 1], az = P[a * 3 + 2];
      const e1x = P[c1 * 3] - ax, e1y = P[c1 * 3 + 1] - ay, e1z = P[c1 * 3 + 2] - az, e2x = P[c2 * 3] - ax, e2y = P[c2 * 3 + 1] - ay, e2z = P[c2 * 3 + 2] - az;
      const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
      for (const v of [a, c1, c2]) { const q = nAt[v]; if (q >= 0) { G[q * 3] += nx; G[q * 3 + 1] += ny; G[q * 3 + 2] += nz; } }
    }
    this.rig.rebase(P);
    if (this.scratch.length !== P.length) this.scratch = new Float32Array(P.length);
    this.scratch.set(P);
    this.stageA = new Float32Array(R.length * 3);
    this.nMoved = 0; this.nNormals = 0;
    this.ready = true;
    this.rebaseMs = now() - t0;
    return this.rebaseMs;
  }
  private allTris() { const b = this.body; return b.hide && b.weldShown ? b.weldShown : b.weld; }
  // the triangles with a vertex in the region, per region vertex its triangles, the normal vertices (the corners of those
  // triangles) and the seam copies among them
  private buildTriangles() {
    const all = this.allTris(), at = this.at, keep: number[] = [];
    for (let t = 0; t < all.length; t += 3) if (at[all[t]] >= 0 || at[all[t + 1]] >= 0 || at[all[t + 2]] >= 0) keep.push(all[t], all[t + 1], all[t + 2]);
    // copies of seam vertices take their source's place in the welded triangles
    const b = this.body, srcOf = new Map<number, number>();
    for (let i = 0; i < b.copyOf.length; i += 2) srcOf.set(b.copyOf[i], b.copyOf[i + 1]);
    for (let i = 0; i < this.region.length; i++) {
      const v = this.region[i], s = srcOf.get(v);
      if (s !== undefined && at[s] < 0) { /* a copy whose source lies outside: the source's triangles reach it */ }
    }
    this.tris = Uint32Array.from(keep);
    const T = this.tris, nR = this.region.length, cnt = new Uint32Array(nR + 1);
    for (let t = 0; t < T.length; t++) { const q = at[T[t]]; if (q >= 0) cnt[q + 1]++; }
    for (let i = 0; i < nR; i++) cnt[i + 1] += cnt[i];
    const fill = cnt.slice(0, nR), list = new Uint32Array(cnt[nR]);
    for (let t = 0; t < T.length; t++) { const q = at[T[t]]; if (q >= 0) list[fill[q]++] = (t / 3) | 0; }
    this.vtStart = cnt; this.vtTris = list;
    const n = b.nFine, nmark = new Uint8Array(n);
    for (let t = 0; t < T.length; t++) nmark[T[t]] = 1;
    // the copies follow their sources: a copy of a normal vertex is a normal vertex
    const copies: number[] = [];
    for (let i = 0; i < b.copyOf.length; i += 2) { const c = b.copyOf[i], s = b.copyOf[i + 1]; if (nmark[s]) { nmark[c] = 1; copies.push(c, s); } }
    const ids: number[] = [];
    for (let v = 0; v < n; v++) if (nmark[v]) ids.push(v);
    this.nIds = Uint32Array.from(ids);
    this.nAt = new Int32Array(n).fill(-1);
    this.nIds.forEach((v, i) => { this.nAt[v] = i; });
    this.copies = Uint32Array.from(copies);
    this.triMark = new Uint8Array(T.length / 3); this.triList = new Uint32Array(T.length / 3);
  }

  // the extra shapes of a group (null takes it off); `after`: added after the stage the parts follow (a card's eye
  // targets, whose parts take their own offsets)
  setGroup(name: string, shapes: Shape[] | null, after = false) {
    if (!shapes || !shapes.length) { this.groups.delete(name); }
    else this.groups.set(name, { shapes, weights: new Float32Array(shapes.length), after });
    // the region holds every vertex of the groups: shapes outside it make the layer take a new copy of the rest
    let outside = false;
    if (shapes) for (const s of shapes) for (const v of s.idx) if (this.at[v] === undefined || this.at[v] < 0) { outside = true; break; }
    if (outside) { this.at = new Int32Array(0); this.ready = false; }
  }

  // a face: action weights (by the rig's order), and the weights of each group; writes the body's arrays and returns the
  // range of vertices to upload ([lo, hi], hi < lo when nothing changed)
  // exact: the normals of the moved triangles; otherwise the sum of each shape's change of the normals at weight 1, which
  // costs less (the frames of a fade)
  apply(w: ArrayLike<number>, groups: Record<string, ArrayLike<number>> = {}, exact = true): [number, number] {
    if (!this.ready) this.rebase();
    const b = this.body, P = b.pos, N = b.nrm, S = b.ns, at = this.at, bP = this.bP, nAt = this.nAt, bN = this.bN, bS = this.bS, rig = this.rig, A = this.stageA;
    let lo = Infinity, hi = -1;
    // back to the rest for the vertices that the last face moved
    const pm = this.moved, pn = this.normals;
    for (let i = 0; i < this.nMoved; i++) {
      const v = pm[i], q = at[v] * 3, o = v * 3;
      P[o] = bP[q]; P[o + 1] = bP[q + 1]; P[o + 2] = bP[q + 2]; A[q] = 0; A[q + 1] = 0; A[q + 2] = 0;
      if (v < lo) lo = v; if (v > hi) hi = v;
    }
    for (let i = 0; i < this.nNormals; i++) {
      const v = pn[i], q = nAt[v] * 3, o = v * 3;
      N[o] = bN[q]; N[o + 1] = bN[q + 1]; N[o + 2] = bN[q + 2]; S[o] = bS[q]; S[o + 1] = bS[q + 1]; S[o + 2] = bS[q + 2];
      if (v < lo) lo = v; if (v > hi) hi = v;
    }
    const mark = this.mark, moved = this.moved;
    let nm = 0, outside = 0;
    const add = (s: Shape, k: number) => {
      const idx = s.idx, d = s.d, m = idx.length;
      for (let i = 0; i < m; i++) {
        const v = idx[i];
        if (at[v] < 0) { outside++; continue; }
        if (mark[v] === 0) { mark[v] = 1; moved[nm++] = v; }
        const o = v * 3, q = i * 3;
        P[o] += k * d[q]; P[o + 1] += k * d[q + 1]; P[o + 2] += k * d[q + 2];
      }
    };
    for (let a = 0; a < rig.names.length; a++) { const k = w[a]; if (k && a !== rig.jaw) add(rig.fineShape(a), k); }
    for (const [name, g] of this.groups) {
      const gw = groups[name];
      if (!gw || g.after) continue;
      for (let i = 0; i < g.shapes.length; i++) if (gw[i]) add(g.shapes[i], gw[i]);
    }
    const wj = rig.jaw >= 0 ? w[rig.jaw] || 0 : 0;
    if (wj && rig.jawCorrective && rig.fineJaw) {
      add(rig.jawCorrective, wj);
      const R = rig.turn(wj), p = rig.pivot, J = rig.fineJaw, px = p[0], py = p[1], pz = p[2];
      const r0 = R[0] - 1, r1 = R[1], r2 = R[2], r3 = R[3], r4 = R[4] - 1, r5 = R[5], r6 = R[6], r7 = R[7], r8 = R[8] - 1;
      for (let i = 0; i < J.idx.length; i++) {
        const v = J.idx[i];
        if (at[v] < 0) continue;
        if (mark[v] === 0) { mark[v] = 1; moved[nm++] = v; }
        const o = v * 3, k = J.w[i], dx = P[o] - px, dy = P[o + 1] - py, dz = P[o + 2] - pz;
        P[o] += k * (r0 * dx + r1 * dy + r2 * dz); P[o + 1] += k * (r3 * dx + r4 * dy + r5 * dz); P[o + 2] += k * (r6 * dx + r7 * dy + r8 * dz);
      }
    }
    // the stage the parts follow
    for (let i = 0; i < nm; i++) { const v = moved[i], q = at[v] * 3, o = v * 3; A[q] = P[o] - bP[q]; A[q + 1] = P[o + 1] - bP[q + 1]; A[q + 2] = P[o + 2] - bP[q + 2]; }
    for (const [name, g] of this.groups) {
      const gw = groups[name];
      if (!gw || !g.after) continue;
      for (let i = 0; i < g.shapes.length; i++) if (gw[i]) add(g.shapes[i], gw[i]);
    }
    for (let i = 0; i < nm; i++) { const v = moved[i]; mark[v] = 0; if (v < lo) lo = v; if (v > hi) hi = v; }
    this.nMoved = nm;
    this.outside = outside;
    const [l2, h2] = exact ? this.updateNormals() : this.blendNormals(w, groups, wj);
    return [Math.min(lo, l2), Math.max(hi, h2)];
  }

  // the change of the normals of a shape at weight 1 on the body at rest (as updateNormals finds it), kept with the shape
  normalShape(s: Shape) {
    if (s.dn) return;
    const P = this.scratch, bP = this.bP, at = this.at, nAt = this.nAt, T = this.tris, F = this.fBase, G = this.gBase, bN = this.bN;
    const vtS = this.vtStart, vtT = this.vtTris, idx = s.idx, d = s.d;
    const tm = new Uint8Array(T.length / 3), tl: number[] = [], seen = new Map<number, number>(), acc: number[] = [];
    for (let i = 0; i < idx.length; i++) {
      const v = idx[i], q = at[v];
      if (q < 0) continue;
      const o = v * 3;
      P[o] += d[i * 3]; P[o + 1] += d[i * 3 + 1]; P[o + 2] += d[i * 3 + 2];
      for (let e = vtS[q]; e < vtS[q + 1]; e++) { const t = vtT[e]; if (!tm[t]) { tm[t] = 1; tl.push(t); } }
    }
    for (const t of tl) {
      const o = t * 3, a = T[o] * 3, c1 = T[o + 1] * 3, c2 = T[o + 2] * 3;
      const ax = P[a], ay = P[a + 1], az = P[a + 2];
      const e1x = P[c1] - ax, e1y = P[c1 + 1] - ay, e1z = P[c1 + 2] - az, e2x = P[c2] - ax, e2y = P[c2 + 1] - ay, e2z = P[c2 + 2] - az;
      const dx = e1y * e2z - e1z * e2y - F[o], dy = e1z * e2x - e1x * e2z - F[o + 1], dz = e1x * e2y - e1y * e2x - F[o + 2];
      for (let k = 0; k < 3; k++) {
        const v = T[o + k];
        let j = seen.get(v);
        if (j === undefined) { j = acc.length; seen.set(v, j); acc.push(0, 0, 0); }
        acc[j] += dx; acc[j + 1] += dy; acc[j + 2] += dz;
      }
    }
    for (let i = 0; i < idx.length; i++) { const v = idx[i], q = at[v]; if (q >= 0) { P[v * 3] = bP[q * 3]; P[v * 3 + 1] = bP[q * 3 + 1]; P[v * 3 + 2] = bP[q * 3 + 2]; } }
    const ids: number[] = [], dn: number[] = [], C = this.copies, byV = new Map<number, number>();
    for (const [v, j] of seen) {
      const q = nAt[v] * 3;
      const gx = G[q], gy = G[q + 1], gz = G[q + 2], gl = 1 / (Math.sqrt(gx * gx + gy * gy + gz * gz) || 1);
      let hx = gx + acc[j], hy = gy + acc[j + 1], hz = gz + acc[j + 2];
      const hl = 1 / (Math.sqrt(hx * hx + hy * hy + hz * hz) || 1);
      hx *= hl; hy *= hl; hz *= hl;
      let x = bN[q] + hx - gx * gl, y = bN[q + 1] + hy - gy * gl, z = bN[q + 2] + hz - gz * gl;
      const l = 1 / (Math.sqrt(x * x + y * y + z * z) || 1);
      x *= l; y *= l; z *= l;
      byV.set(v, ids.length);
      ids.push(v); dn.push(x - bN[q], y - bN[q + 1], z - bN[q + 2]);
    }
    for (let i = 0; i < C.length; i += 2) {
      const j = byV.get(C[i + 1]);
      if (j === undefined) continue;
      ids.push(C[i]); dn.push(dn[j * 3], dn[j * 3 + 1], dn[j * 3 + 2]);
    }
    s.nIdx = Uint32Array.from(ids); s.dn = Float32Array.from(dn);
  }

  // the normals as the sum of each shown shape's change at its weight, turned by the jaw
  private blendNormals(w: ArrayLike<number>, groups: Record<string, ArrayLike<number>>, wj: number): [number, number] {
    const b = this.body, N = b.nrm, S = b.ns, nAt = this.nAt, bN = this.bN, bS = this.bS, rig = this.rig, acc = this.acc, mark = this.mark, out = this.normals;
    let no = 0, lo = Infinity, hi = -1;
    const add = (s: Shape, k: number) => {
      this.normalShape(s);
      const idx = s.nIdx!, dn = s.dn!;
      for (let i = 0; i < idx.length; i++) {
        const v = idx[i], o = v * 3;
        if (nAt[v] < 0) continue;
        if (mark[v] === 0) { mark[v] = 1; out[no++] = v; acc[o] = 0; acc[o + 1] = 0; acc[o + 2] = 0; }
        acc[o] += k * dn[i * 3]; acc[o + 1] += k * dn[i * 3 + 1]; acc[o + 2] += k * dn[i * 3 + 2];
      }
    };
    for (let a = 0; a < rig.names.length; a++) { const k = w[a]; if (k && a !== rig.jaw) add(rig.fineShape(a), k); }
    for (const [name, g] of this.groups) { const gw = groups[name]; if (gw) for (let i = 0; i < g.shapes.length; i++) if (gw[i]) add(g.shapes[i], gw[i]); }
    const J = rig.fineJaw;
    if (wj && rig.jawCorrective && J) {
      add(rig.jawCorrective, wj);
      for (let i = 0; i < J.idx.length; i++) { const v = J.idx[i], o = v * 3; if (nAt[v] >= 0 && mark[v] === 0) { mark[v] = 1; out[no++] = v; acc[o] = 0; acc[o + 1] = 0; acc[o + 2] = 0; } }
    }
    for (let i = 0; i < no; i++) {
      const v = out[i], q = nAt[v] * 3, o = v * 3;
      const ax = bN[q], ay = bN[q + 1], az = bN[q + 2];
      let x = ax + acc[o], y = ay + acc[o + 1], z = az + acc[o + 2];
      const l = 1 / (Math.sqrt(x * x + y * y + z * z) || 1);
      N[o] = x * l; N[o + 1] = y * l; N[o + 2] = z * l;
    }
    // the jaw's turn of the normals, by each vertex's weight
    if (wj && J) {
      const R = rig.turn(wj);
      for (let i = 0; i < J.idx.length; i++) {
        const v = J.idx[i], o = v * 3, k = J.w[i];
        if (nAt[v] < 0) continue;
        const x = N[o], y = N[o + 1], z = N[o + 2];
        let nx = x + k * ((R[0] - 1) * x + R[1] * y + R[2] * z), ny = y + k * (R[3] * x + (R[4] - 1) * y + R[5] * z), nz = z + k * (R[6] * x + R[7] * y + (R[8] - 1) * z);
        const l = 1 / (Math.sqrt(nx * nx + ny * ny + nz * nz) || 1);
        nx *= l; ny *= l; nz *= l;
        N[o] = nx; N[o + 1] = ny; N[o + 2] = nz;
      }
    }
    for (let i = 0; i < no; i++) {
      const v = out[i], q = nAt[v] * 3, o = v * 3;
      mark[v] = 0;
      const ax = bN[q], ay = bN[q + 1], az = bN[q + 2], x = N[o], y = N[o + 1], z = N[o + 2];
      const kx = ay * z - az * y, ky = az * x - ax * z, kz = ax * y - ay * x, c = ax * x + ay * y + az * z, s2 = kx * kx + ky * ky + kz * kz;
      let vx = bS[q], vy = bS[q + 1], vz = bS[q + 2];
      if (s2 > 1e-12 && c > -0.9) {
        const ff = (kx * vx + ky * vy + kz * vz) * (1 - c) / s2;
        const rx = vx * c + (ky * vz - kz * vy) + kx * ff, ry = vy * c + (kz * vx - kx * vz) + ky * ff, rz = vz * c + (kx * vy - ky * vx) + kz * ff;
        vx = rx; vy = ry; vz = rz;
      }
      const mm = 1 / (Math.sqrt(vx * vx + vy * vy + vz * vz) || 1);
      S[o] = vx * mm; S[o + 1] = vy * mm; S[o + 2] = vz * mm;
      if (v < lo) lo = v; if (v > hi) hi = v;
    }
    this.nNormals = no;
    return [lo, hi];
  }

  // the normals of the moved surface: the summed normal at rest plus the change of the moved triangles, turned onto the
  // body's own normal at rest; the smooth shading normal turns with the normal (as the correctives turn it in main.ts)
  private updateNormals(): [number, number] {
    const b = this.body, P = b.pos, N = b.nrm, S = b.ns, T = this.tris, F = this.fBase, at = this.at, nAt = this.nAt;
    const tm = this.triMark, acc = this.acc, mark = this.mark, moved = this.moved, nm = this.nMoved, vtS = this.vtStart, vtT = this.vtTris;
    const tl = this.triList, out = this.normals;
    let nt = 0, no = 0, lo = Infinity, hi = -1;
    for (let i = 0; i < nm; i++) {
      const q = at[moved[i]];
      for (let e = vtS[q]; e < vtS[q + 1]; e++) { const t = vtT[e]; if (tm[t] === 0) { tm[t] = 1; tl[nt++] = t; } }
    }
    for (let i = 0; i < nt; i++) {
      const t = tl[i];
      tm[t] = 0;
      const o = t * 3, a = T[o] * 3, c1 = T[o + 1] * 3, c2 = T[o + 2] * 3;
      const ax = P[a], ay = P[a + 1], az = P[a + 2];
      const e1x = P[c1] - ax, e1y = P[c1 + 1] - ay, e1z = P[c1 + 2] - az, e2x = P[c2] - ax, e2y = P[c2 + 1] - ay, e2z = P[c2 + 2] - az;
      const dx = e1y * e2z - e1z * e2y - F[o], dy = e1z * e2x - e1x * e2z - F[o + 1], dz = e1x * e2y - e1y * e2x - F[o + 2];
      for (let k = 0; k < 3; k++) {
        const v = T[o + k], m = v * 3;
        if (mark[v] === 0) { mark[v] = 1; out[no++] = v; acc[m] = 0; acc[m + 1] = 0; acc[m + 2] = 0; }
        acc[m] += dx; acc[m + 1] += dy; acc[m + 2] += dz;
      }
    }
    const G = this.gBase, bN = this.bN, bS = this.bS;
    for (let i = 0; i < no; i++) {
      const v = out[i], q = nAt[v] * 3, o = v * 3;
      const gx = G[q], gy = G[q + 1], gz = G[q + 2], gl = 1 / (Math.sqrt(gx * gx + gy * gy + gz * gz) || 1);
      let hx = gx + acc[o], hy = gy + acc[o + 1], hz = gz + acc[o + 2];
      const hl = 1 / (Math.sqrt(hx * hx + hy * hy + hz * hz) || 1);
      hx *= hl; hy *= hl; hz *= hl;
      const ax = bN[q], ay = bN[q + 1], az = bN[q + 2];
      let x = ax + hx - gx * gl, y = ay + hy - gy * gl, z = az + hz - gz * gl;
      const l = 1 / (Math.sqrt(x * x + y * y + z * z) || 1);
      x *= l; y *= l; z *= l;
      N[o] = x; N[o + 1] = y; N[o + 2] = z;
      // the smooth normal: the shortest turn from the old normal to the new one
      const kx = ay * z - az * y, ky = az * x - ax * z, kz = ax * y - ay * x, c = ax * x + ay * y + az * z, s2 = kx * kx + ky * ky + kz * kz;
      let vx = bS[q], vy = bS[q + 1], vz = bS[q + 2];
      if (s2 > 1e-12 && c > -0.9) {
        const ff = (kx * vx + ky * vy + kz * vz) * (1 - c) / s2;
        const rx = vx * c + (ky * vz - kz * vy) + kx * ff, ry = vy * c + (kz * vx - kx * vz) + ky * ff, rz = vz * c + (kx * vy - ky * vx) + kz * ff;
        vx = rx; vy = ry; vz = rz;
      }
      const mm = 1 / (Math.sqrt(vx * vx + vy * vy + vz * vz) || 1);
      S[o] = vx * mm; S[o + 1] = vy * mm; S[o + 2] = vz * mm;
      if (v < lo) lo = v; if (v > hi) hi = v;
    }
    // the seam copies take their source's normals
    const C = this.copies;
    for (let i = 0; i < C.length; i += 2) {
      const c = C[i], s = C[i + 1];
      if (mark[s] === 0) continue;
      const o = c * 3, os = s * 3;
      N[o] = N[os]; N[o + 1] = N[os + 1]; N[o + 2] = N[os + 2]; S[o] = S[os]; S[o + 1] = S[os + 1]; S[o + 2] = S[os + 2];
      out[no++] = c;
      if (c < lo) lo = c; if (c > hi) hi = c;
    }
    for (let i = 0; i < no; i++) mark[out[i]] = 0;
    this.nNormals = no;
    return [lo, hi];
  }

  // the offset of a fine vertex in the stage that the parts follow (zero outside the region)
  offsetOf(v: number, out: number[] | Float32Array, o = 0) {
    const q = this.at[v];
    if (q === undefined || q < 0) { out[o] = 0; out[o + 1] = 0; out[o + 2] = 0; return; }
    out[o] = this.stageA[q * 3]; out[o + 1] = this.stageA[q * 3 + 1]; out[o + 2] = this.stageA[q * 3 + 2];
  }
}

function faceNormal(P: Float32Array, a: number, b: number, c: number, out: Float32Array, o: number) {
  const ax = P[a * 3], ay = P[a * 3 + 1], az = P[a * 3 + 2];
  const e1x = P[b * 3] - ax, e1y = P[b * 3 + 1] - ay, e1z = P[b * 3 + 2] - az, e2x = P[c * 3] - ax, e2y = P[c * 3 + 1] - ay, e2z = P[c * 3 + 2] - az;
  out[o] = e1y * e2z - e1z * e2y; out[o + 1] = e1z * e2x - e1x * e2z; out[o + 2] = e1x * e2y - e1y * e2x;
}

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

// ------------------------------------------------------------------ binding points to the skin
// the nearest of the given fine vertices to each point (a uniform grid of the cell size h)
export function nearestVertices(points: ArrayLike<number>, P: Float32Array, ids: ArrayLike<number>, h = 0.01): Uint32Array {
  const grid = new Map<string, number[]>(), key = (x: number, y: number, z: number) => `${Math.floor(x / h)},${Math.floor(y / h)},${Math.floor(z / h)}`;
  for (let i = 0; i < ids.length; i++) { const v = ids[i], k = key(P[v * 3], P[v * 3 + 1], P[v * 3 + 2]); let c = grid.get(k); if (!c) grid.set(k, c = []); c.push(v); }
  const n = points.length / 3, out = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const x = points[i * 3], y = points[i * 3 + 1], z = points[i * 3 + 2];
    const gx = Math.floor(x / h), gy = Math.floor(y / h), gz = Math.floor(z / h);
    let best = -1, bd = Infinity;
    for (let r = 1; r <= 8 && best < 0; r++) {
      for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) for (let dz = -r; dz <= r; dz++) {
        const c = grid.get(`${gx + dx},${gy + dy},${gz + dz}`);
        if (!c) continue;
        for (const v of c) { const d = (P[v * 3] - x) ** 2 + (P[v * 3 + 1] - y) ** 2 + (P[v * 3 + 2] - z) ** 2; if (d < bd) { bd = d; best = v; } }
      }
    }
    out[i] = best < 0 ? (ids.length ? ids[0] : 0) : best;
  }
  return out;
}

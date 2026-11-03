// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// anny's phenotype sliders in the page: the same blend-shape coefficients as
// Anny._get_phenotype_blendshape_coefficients (src/anny/models/phenotype.py), the coarse body from the
// principal components that anny.viewer.export keeps, and the bone heads from the full joint blend shapes.
// This module has no dependencies, so node runs it directly in the parity test (test/test_viewer_parity.py).

export interface ShapeTables {
  labels: string[];                       // the sliders of the model (anny's default: gender, age, ...)
  variations: [string, string[]][];       // PHENOTYPE_VARIATIONS, in order (race first)
  anchors: Record<string, number[]>;      // anchor values of each phenotype except race
  extrapolate: boolean;
  mask: string[];                         // one string of 0 and 1 per blend shape (length 26)
  default: number;                        // the value of a slider that is not set (0.5)
}

const FEATURES = ['age', 'gender', 'muscle', 'weight', 'height', 'proportions', 'cupsize', 'firmness'];

// anny.utils.interpolation.linear_interpolation_coefficients for one value
export function interpolationCoefficients(value: number, anchors: number[], extrapolate: boolean): Float64Array {
  const n = anchors.length;
  const out = new Float64Array(n);
  let idx = 0;
  while (idx < n && anchors[idx] < value) idx++;          // torch.searchsorted(side="left")
  idx = Math.min(Math.max(idx, 1), n - 1);
  const lo = anchors[idx - 1], hi = anchors[idx];
  let alpha = (value - lo) / (hi - lo);
  if (!extrapolate) alpha = Math.min(Math.max(alpha, 0), 1);
  out[idx - 1] = 1 - alpha;
  out[idx] = alpha;
  return out;
}

export interface CoefficientRule {
  tables: ShapeTables;
  masks: Uint8Array[];      // per blend shape, the indices of the variations it multiplies
  order: string[];          // the 26 variation keys in order
}

export function makeRule(tables: ShapeTables): CoefficientRule {
  const order: string[] = [];
  for (const [, keys] of tables.variations) order.push(...keys);
  const masks = tables.mask.map((m) => {
    const ids: number[] = [];
    for (let j = 0; j < m.length; j++) if (m[j] === '1') ids.push(j);
    return Uint8Array.from(ids);
  });
  return { tables, masks, order };
}

// the coefficients of the phenotype blend shapes for slider values (missing sliders take the default)
export function coefficients(rule: CoefficientRule, values: Record<string, number>): Float64Array {
  const t = rule.tables;
  const value = (label: string) => (t.labels.includes(label) && typeof values[label] === 'number') ? values[label] : t.default;
  const phen: Record<string, number> = {};
  for (const f of FEATURES) {
    const keys = t.variations.find((v) => v[0] === f)[1];
    const w = interpolationCoefficients(value(f), t.anchors[f], t.extrapolate);
    keys.forEach((k, i) => { phen[k] = w[i]; });
  }
  const race = ['african', 'asian', 'caucasian'].map(value);
  const sum = race[0] + race[1] + race[2];
  ['african', 'asian', 'caucasian'].forEach((k, i) => { phen[k] = sum > 0 ? race[i] / sum : 1 / 3; });
  const vec = rule.order.map((k) => phen[k]);
  const out = new Float64Array(rule.masks.length);
  for (let n = 0; n < rule.masks.length; n++) {
    let p = 1;
    const m = rule.masks[n];
    for (let j = 0; j < m.length; j++) p *= vec[m[j]];
    out[n] = p;
  }
  return out;
}

export interface ShapeSpaceData {
  template: Float32Array;       // (coarse vertices, 3) anny's template in the page frame
  components: Int16Array;       // (K, coarse vertices, 3) in steps of componentScale[k]
  componentScale: number[];
  projection: Float32Array;     // (K, blend shapes)
  jointTemplate: Float32Array;  // (bones, 3)
  jointBlend: Float32Array;     // (blend shapes, bones, 3)
}

export class ShapeSpace {
  data: ShapeSpaceData;
  K: number;
  N: number;
  nc: number;
  nb: number;
  weights: Float64Array;
  constructor(data: ShapeSpaceData) {
    this.data = data;
    this.K = data.componentScale.length;
    this.N = data.projection.length / this.K;
    this.nc = data.template.length / 3;
    this.nb = data.jointTemplate.length / 3;
    this.weights = new Float64Array(this.K);
  }
  // coarse vertices (nc * 3) for the coefficients c
  coarse(c: Float64Array, out: Float32Array = new Float32Array(this.nc * 3)): Float32Array {
    const { projection, components, componentScale, template } = this.data;
    const K = this.K, N = this.N, n3 = this.nc * 3;
    for (let k = 0; k < K; k++) {
      let w = 0;
      const row = k * N;
      for (let i = 0; i < N; i++) w += projection[row + i] * c[i];
      this.weights[k] = w * componentScale[k];
    }
    const acc = new Float64Array(n3);
    for (let i = 0; i < n3; i++) acc[i] = template[i];
    for (let k = 0; k < K; k++) {
      const w = this.weights[k];
      if (w === 0) continue;
      const o = k * n3;
      for (let i = 0; i < n3; i++) acc[i] += w * components[o + i];
    }
    for (let i = 0; i < n3; i++) out[i] = acc[i];
    return out;
  }
  // rest bone heads (nb * 3) for the coefficients c
  joints(c: Float64Array, out: Float32Array = new Float32Array(this.nb * 3)): Float32Array {
    const { jointTemplate, jointBlend } = this.data;
    const nb3 = this.nb * 3;
    const acc = new Float64Array(nb3);
    for (let i = 0; i < nb3; i++) acc[i] = jointTemplate[i];
    for (let n = 0; n < this.N; n++) {
      const w = c[n];
      if (w === 0) continue;
      const o = n * nb3;
      for (let i = 0; i < nb3; i++) acc[i] += w * jointBlend[o + i];
    }
    for (let i = 0; i < nb3; i++) out[i] = acc[i];
    return out;
  }
}

// ------------------------------------------------------------------ face shapes (anny.models.face_shapes)
// Each face-shape parameter owns one or two blend-shape rows (its positive and negative directions). A row's
// weight is the rectified value of its parameter times the scale of its group: the size of that part of the head
// (landmark distances on the phenotype body) over its size on anny's default body.

export interface FacePrior {
  age_anchors: number[];
  gender_anchors: number[];
  means: number[][][];                    // (A, 2, F)
  races?: string[];                       // anny's race phenotypes, in the order of race_offsets
  race_offsets?: number[][][];            // (2, races, F): offsets of the mean, weighted by the shares of the races
  rank: number;
  spread?: number;                        // the scale of the spread of random faces (anny.faces.distribution.DEFAULT_SPREAD)
}

export interface FaceTables {
  names: string[];
  groups: string[];
  group_order: string[];
  ranges: [number, number][];
  ends?: [string, string][];
  row_param: number[];
  row_sign: number[];
  row_scale: number[];
  scale_groups: string[];
  landmarks: string[];
  scale_pairs: [number, number][][];
  reference_sizes: number[];
  starts: number[];
  counts: number[];
  bones: number[];
  step: number;
  prior?: FacePrior;
}

export interface FaceData {
  tables: FaceTables;
  lmTemplate: Float32Array;   // (K, 3) landmarks of the template
  lmBlend: Float32Array;      // (N, K, 3) landmarks of the phenotype blend shapes
  ids: Uint32Array;           // coarse vertex of each sparse offset
  offsets: Int16Array;        // (total, 3) offsets in steps of tables.step
  boneDeltas: Float32Array;   // (rows, bones, 3) bone-head deltas of each row on tables.bones
  prior?: Float32Array;       // (A, 2, F, rank) low-rank factors of the face-shape distribution
  turns?: TurnRows;           // the rows' blend of the bones' rest-orientation matrices
}

// scale of each group (anny's Anny._face_shape_scales) for the phenotype coefficients c
export function faceScales(face: FaceData, c: Float64Array): Float64Array {
  const t = face.tables, K = t.landmarks.length, K3 = K * 3, N = face.lmBlend.length / K3;
  const L = new Float64Array(K3);
  for (let i = 0; i < K3; i++) L[i] = face.lmTemplate[i];
  for (let n = 0; n < N; n++) {
    const w = c[n];
    if (w === 0) continue;
    const o = n * K3;
    for (let i = 0; i < K3; i++) L[i] += w * face.lmBlend[o + i];
  }
  const out = new Float64Array(t.scale_groups.length);
  t.scale_pairs.forEach((pairs, g) => {
    let s = 0;
    for (const [a, b] of pairs) {
      const dx = L[a * 3] - L[b * 3], dy = L[a * 3 + 1] - L[b * 3 + 1], dz = L[a * 3 + 2] - L[b * 3 + 2];
      s += Math.log(Math.sqrt(dx * dx + dy * dy + dz * dz));
    }
    out[g] = Math.exp(s / pairs.length) / t.reference_sizes[g];
  });
  return out;
}

// the weight of each face-shape row for parameter values (in the order of tables.names)
export function faceRowWeights(t: FaceTables, values: ArrayLike<number>, scales: Float64Array): Float64Array {
  const out = new Float64Array(t.row_param.length);
  for (let r = 0; r < out.length; r++) {
    const v = values[t.row_param[r]] * t.row_sign[r];
    out[r] = v > 0 ? v * scales[t.row_scale[r]] : 0;
  }
  return out;
}

// sparse rows of offsets on the coarse body with bone-head deltas: the face shapes (FaceData) and the local changes
// (LocalData) share this form
export interface SparseRows {
  tables: { starts: number[]; counts: number[]; bones: number[]; step: number };
  ids: Uint32Array;           // coarse vertex of each sparse offset
  offsets: Int16Array;        // (total, 3) offsets in steps of tables.step
  boneDeltas: Float32Array;   // (rows, bones, 3) bone-head deltas of each row on tables.bones
  turns?: TurnRows;           // the rows' blend of the bones' rest-orientation matrices
}

// ------------------------------------------------------------------ local changes (MakeHuman's modifiers)
// Each local change owns two rows: its positive direction, weighted by max(v, 0), and its negative direction,
// weighted by max(-v, 0), as Anny._get_phenotype_blendshape_coefficients weighs them; the rows are not scaled.
export interface LocalData extends SparseRows {
  tables: { labels: string[]; starts: number[]; counts: number[]; bones: number[]; step: number };
}

// the weight of each local-change row for values by label (missing labels take 0)
export function localRowWeights(local: LocalData, values: Record<string, number> | null | undefined): Float64Array {
  const labels = local.tables.labels, out = new Float64Array(labels.length * 2);
  if (!values) return out;
  labels.forEach((name, i) => {
    const v = values[name];
    if (typeof v !== 'number' || !isFinite(v)) return;
    out[i * 2] = v > 0 ? v : 0;
    out[i * 2 + 1] = v < 0 ? -v : 0;
  });
  return out;
}

// add weighted sparse rows to coarse vertices (nc * 3) and to bone heads (nb * 3)
export function addRows(data: SparseRows, weights: Float64Array, coarse: Float32Array, joints?: Float32Array) {
  const t = data.tables, step = t.step, nb = t.bones.length;
  for (let r = 0; r < weights.length; r++) {
    const w = weights[r];
    if (w === 0) continue;
    const ws = w * step, end = t.starts[r] + t.counts[r];
    for (let k = t.starts[r]; k < end; k++) {
      const o = data.ids[k] * 3, q = k * 3;
      coarse[o] += ws * data.offsets[q];
      coarse[o + 1] += ws * data.offsets[q + 1];
      coarse[o + 2] += ws * data.offsets[q + 2];
    }
    if (joints) {
      const o = r * nb * 3;
      for (let j = 0; j < nb; j++) {
        const b = t.bones[j] * 3, q = o + j * 3;
        joints[b] += w * data.boneDeltas[q];
        joints[b + 1] += w * data.boneDeltas[q + 1];
        joints[b + 2] += w * data.boneDeltas[q + 2];
      }
    }
  }
}

// add the face-shape rows to coarse vertices (nc * 3) and to bone heads (nb * 3)
export function addFace(face: FaceData, weights: Float64Array, coarse: Float32Array, joints?: Float32Array) {
  addRows(face, weights, coarse, joints);
}
// the face-shape distribution (anny.faces.distribution) for anny's slider values: mean (F) and a low-rank factor (F, rank)
export function facePriorParameters(face: FaceData, values: Record<string, number>): { mean: Float64Array; factor: Float64Array } {
  const p = face.tables.prior, F = face.tables.names.length, R = p.rank;
  const val = (k: string) => (typeof values[k] === 'number' ? values[k] : 0.5);
  const wa = interpolationCoefficients(val('age'), p.age_anchors, false);
  const [g0, g1] = p.gender_anchors;
  const t = Math.min(1, Math.max(0, (val('gender') - g0) / (g1 - g0)));
  const mean = new Float64Array(F), factor = new Float64Array(F * R);
  for (let a = 0; a < wa.length; a++) {
    if (wa[a] === 0) continue;
    for (let g = 0; g < 2; g++) {
      const w = wa[a] * (g === 0 ? 1 - t : t);
      if (w === 0) continue;
      const m = p.means[a][g], o = (a * 2 + g) * F * R;
      for (let f = 0; f < F; f++) mean[f] += w * m[f];
      for (let i = 0; i < F * R; i++) factor[i] += w * face.prior[o + i];
    }
  }
  // the race offsets, weighted by the shares of the race sliders (equal shares when they are all 0)
  if (p.race_offsets && p.races) {
    const r = p.races.map(val), sum = r.reduce((a, b) => a + b, 0);
    p.races.forEach((_, k) => {
      const share = sum > 0 ? r[k] / sum : 1 / p.races.length;
      for (let g = 0; g < 2; g++) {
        const w = share * (g === 0 ? 1 - t : t), off = p.race_offsets[g][k];
        if (w !== 0) for (let f = 0; f < F; f++) mean[f] += w * off[f];
      }
    });
  }
  return { mean, factor };
}

// a face drawn from the distribution with its spread scaled by prior.spread, clipped to the slider ranges
// (random: uniform numbers in [0, 1))
export function sampleFace(face: FaceData, values: Record<string, number>, random: () => number = Math.random): Float64Array {
  const { mean, factor } = facePriorParameters(face, values);
  const F = mean.length, R = face.tables.prior.rank, spread = face.tables.prior.spread ?? 1;
  const z = new Float64Array(R);
  for (let k = 0; k < R; k += 2) {
    const u = Math.max(random(), 1e-12), v = random();
    const r = Math.sqrt(-2 * Math.log(u));
    z[k] = r * Math.cos(2 * Math.PI * v);
    if (k + 1 < R) z[k + 1] = r * Math.sin(2 * Math.PI * v);
  }
  const out = new Float64Array(F);
  for (let f = 0; f < F; f++) {
    let s = mean[f];
    for (let k = 0; k < R; k++) s += spread * factor[f * R + k] * z[k];
    const [lo, hi] = face.tables.ranges[f];
    out[f] = Math.min(hi, Math.max(lo, s));
  }
  return out;
}

// ------------------------------------------------------------------ rest orientations
// Anny's "cached" bone orientation (anny.models.rigged_model._get_cached_rest_model): each bone's cross-covariance
// matrix is a template plus the blend rows weighted by the same coefficients as the vertices (the phenotype's, the
// face shapes' and the local changes'), and the bone's rest orientation is its special Procrustes solution (the
// rotation nearest to it, roma.special_procrustes). The reference orientations are the zero of Anny's local-ref
// pose; a bone's turn C = reference rest^T converts a local-ref pose into the page's (pose_reference.ts). All in
// Anny's axes, 3 x 3 matrices row-major.

export interface TurnRows { index: Uint32Array; values: Float32Array }   // (row, bone) pairs, and 9 values each
export interface TurnData { template: Float32Array; reference: Float32Array; blend: TurnRows; rootIdentity: boolean }

// add weighted blend rows to the matrices M (bones * 9)
export function addTurnRows(rows: TurnRows, weights: ArrayLike<number>, M: Float64Array) {
  const I = rows.index, V = rows.values, n = I.length / 2;
  for (let e = 0; e < n; e++) {
    const w = weights[I[e * 2]];
    if (!w) continue;
    const o = I[e * 2 + 1] * 9, v = e * 9;
    for (let k = 0; k < 9; k++) M[o + k] += w * V[v + k];
  }
}

// the eigenvectors (columns of V, row-major) and eigenvalues of a symmetric 3 x 3 matrix, by Jacobi rotations
function symmetricEigen(A: Float64Array): { values: number[]; V: Float64Array } {
  const a = Float64Array.from(A), V = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  for (let sweep = 0; sweep < 50; sweep++) {
    const off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
    if (off < 1e-30) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      const apq = a[p * 3 + q];
      if (Math.abs(apq) < 1e-300) continue;
      const theta = (a[q * 3 + q] - a[p * 3 + p]) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) {   // A <- J^T A J
        const akp = a[k * 3 + p], akq = a[k * 3 + q];
        a[k * 3 + p] = c * akp - s * akq; a[k * 3 + q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p * 3 + k], aqk = a[q * 3 + k];
        a[p * 3 + k] = c * apk - s * aqk; a[q * 3 + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = V[k * 3 + p], vkq = V[k * 3 + q];
        V[k * 3 + p] = c * vkp - s * vkq; V[k * 3 + q] = s * vkp + c * vkq;
      }
    }
  }
  return { values: [a[0], a[4], a[8]], V };
}

// the rotation nearest to M (row-major 9 at offset o) into out at offset oo: U diag(1, 1, det(U V^T)) V^T from the SVD
// M = U S V^T, as roma.special_procrustes
export function specialProcrustes(M: ArrayLike<number>, o: number, out: Float64Array, oo: number) {
  const m = (r: number, c: number) => M[o + r * 3 + c];
  const A = new Float64Array(9);   // M^T M
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let s = 0;
    for (let k = 0; k < 3; k++) s += m(k, i) * m(k, j);
    A[i * 3 + j] = s;
  }
  const { values, V } = symmetricEigen(A);
  const order = [0, 1, 2].sort((x, y) => values[y] - values[x]);
  const col = (j: number) => [V[order[j]], V[3 + order[j]], V[6 + order[j]]];
  const v1 = col(0), v2c = col(1);
  const Mv = (v: number[]) => [0, 1, 2].map((r) => m(r, 0) * v[0] + m(r, 1) * v[1] + m(r, 2) * v[2]);
  const unit = (v: number[]) => { const n = Math.hypot(v[0], v[1], v[2]) || 1; return v.map((x) => x / n); };
  const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  // V with det 1, U = M V / S on the two largest singular values, the third from the cross product: R = U V^T is then
  // the special Procrustes solution whatever the sign of the smallest singular value
  let v2 = unit(v2c.map((x, k) => x - dot(v2c, v1) * v1[k]));
  const v3 = cross(v1, v2);
  const u1 = unit(Mv(v1));
  let u2 = Mv(v2);
  u2 = unit(u2.map((x, k) => x - dot(u2, u1) * u1[k]));
  const u3 = cross(u1, u2);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[oo + r * 3 + c] = u1[r] * v1[c] + u2[r] * v2[c] + u3[r] * v3[c];
}

// each bone's rest orientation (bones * 9) for the phenotype coefficients c and further weighted rows (the face shapes'
// and the local changes')
export function restOrientations(data: TurnData, c: ArrayLike<number>, extra: [TurnRows | undefined, ArrayLike<number> | null][] = []): Float64Array {
  const nb = data.template.length / 9, M = Float64Array.from(data.template), out = new Float64Array(nb * 9);
  addTurnRows(data.blend, c, M);
  for (const [rows, w] of extra) if (rows && w) addTurnRows(rows, w, M);
  for (let b = 0; b < nb; b++) specialProcrustes(M, b * 9, out, b * 9);
  if (data.rootIdentity) out.set([1, 0, 0, 0, 1, 0, 0, 0, 1], 0);
  return out;
}

// each bone's turn C = reference rest^T (bones * 9): local-ref D = C_parent R C^T for the page's rotation R
export function referenceTurns(data: TurnData, rest: Float64Array): Float64Array {
  const nb = rest.length / 9, F = data.reference, out = new Float64Array(nb * 9);
  for (let b = 0; b < nb; b++) {
    const o = b * 9;
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += F[o + r * 3 + k] * rest[o + c * 3 + k];
      out[o + r * 3 + c] = s;
    }
  }
  return out;
}

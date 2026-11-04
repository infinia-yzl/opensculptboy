// OpenSculptBoy
// Apache License, Version 2.0
//
// The place of a rigid piece on a card, as opensculptboy.wearables.place defines it: a prop moved by hand in the
// studio, and the bone that carries it. A piece entry of a card (an outfit's piece or a loose wearable) may hold
//   "place": { "bone": "metacarpal4.R", "rotation": [x, y, z, w], "translation": [x, y, z] }    (card schema 6)
// The rotation and the translation (metres) move the piece in the card's rest frame (Anny's axes) about the rest head
// h of `bone`: a rest vertex x of the piece, as its rigid binding gives it, goes to h + Q (x - h) + t, and the piece
// then follows `bone` alone. The page works in its own frame (a similarity of Anny's: p = s R x + o), so the move
// there is h' + R Q R^T (x' - h') + s R t, with h' the head in the page's frame. A change of the carrying bone keeps
// the piece where it stands in a pose: the new move is S_new^-1 S_old A_old, with S the bones' skinning matrices
// (rest to posed) and A the move as a matrix of the rest frame. No DOM and no three.js: node runs it
// (viewer/test/props.mjs, and test_wearable_place against the Python twin).

export interface Place { bone: string; rotation: number[]; translation: number[] }
export interface Similarity { scale: number; rotation: number[]; offset: number[] }   // p = scale * R x + offset, R row-major
export type M4 = Float64Array;   // a 4 x 4 matrix, row-major
export const PLACE_SCHEMA = 6;
export const IDENTITY_Q = [0, 0, 0, 1];

const finite = (v: unknown, n: number) => Array.isArray(v) && v.length === n && v.every((x) => typeof x === 'number' && isFinite(x));

// why a place does not hold, or null
export function placeProblem(p: any): string | null {
  if (!p || typeof p !== 'object' || Array.isArray(p)) return 'a place is an object';
  if (Object.keys(p).some((k) => !['bone', 'rotation', 'translation'].includes(k))) return 'a place holds bone, rotation and translation';
  if (typeof p.bone !== 'string' || !p.bone) return 'a place names the bone that carries the piece';
  const q = p.rotation ?? IDENTITY_Q;
  if (!finite(q, 4) || Math.abs(Math.hypot(q[0], q[1], q[2], q[3]) - 1) > 1e-3) return 'place.rotation is a unit quaternion (x, y, z, w)';
  if (!finite(p.translation ?? [0, 0, 0], 3)) return 'place.translation is three numbers in metres';
  return null;
}

// ------------------------------------------------------------------ rotations
export function rotationMatrix(q: ArrayLike<number>): number[] {
  let [x, y, z, w] = [q[0], q[1], q[2], q[3]];
  const n = Math.hypot(x, y, z, w) || 1; x /= n; y /= n; z /= n; w /= n;
  return [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)];
}
// the unit quaternion (x, y, z, w with w >= 0) of a rotation, row-major (Shepperd's method)
export function quaternionOf(R: ArrayLike<number>): number[] {
  const t = R[0] + R[4] + R[8];
  let q: number[];
  if (t > 0) {
    const s = 2 * Math.sqrt(t + 1);
    q = [(R[7] - R[5]) / s, (R[2] - R[6]) / s, (R[3] - R[1]) / s, 0.25 * s];
  } else {
    const d = [R[0], R[4], R[8]], i = d[0] >= d[1] && d[0] >= d[2] ? 0 : d[1] >= d[2] ? 1 : 2, j = (i + 1) % 3, k = (i + 2) % 3;
    const r = (a: number, b: number) => R[a * 3 + b];
    const s = 2 * Math.sqrt(Math.max(1 + r(i, i) - r(j, j) - r(k, k), 1e-300));
    q = [0, 0, 0, (r(k, j) - r(j, k)) / s];
    q[i] = 0.25 * s; q[j] = (r(j, i) + r(i, j)) / s; q[k] = (r(k, i) + r(i, k)) / s;
  }
  if (q[3] < 0) q = q.map((v) => -v);
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return q.map((v) => v / n);
}
// a turn of `rad` radians about a unit axis, row-major
export function axisAngle(axis: ArrayLike<number>, rad: number): number[] {
  const l = Math.hypot(axis[0], axis[1], axis[2]) || 1, x = axis[0] / l, y = axis[1] / l, z = axis[2] / l, s = Math.sin(rad / 2);
  return rotationMatrix([x * s, y * s, z * s, Math.cos(rad / 2)]);
}

// ------------------------------------------------------------------ 4 x 4 matrices, row-major
export function m4(R: ArrayLike<number> = [1, 0, 0, 0, 1, 0, 0, 0, 1], t: ArrayLike<number> = [0, 0, 0]): M4 {
  return Float64Array.from([R[0], R[1], R[2], t[0], R[3], R[4], R[5], t[1], R[6], R[7], R[8], t[2], 0, 0, 0, 1]);
}
export function mul4(A: ArrayLike<number>, B: ArrayLike<number>): M4 {
  const o = new Float64Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) { let s = 0; for (let k = 0; k < 4; k++) s += A[r * 4 + k] * B[k * 4 + c]; o[r * 4 + c] = s; }
  return o;
}
// the inverse of a rigid matrix (a rotation and a translation)
export function rigidInverse(M: ArrayLike<number>): M4 {
  const R = [M[0], M[4], M[8], M[1], M[5], M[9], M[2], M[6], M[10]], t = [M[3], M[7], M[11]];
  return m4(R, [-(R[0] * t[0] + R[1] * t[1] + R[2] * t[2]), -(R[3] * t[0] + R[4] * t[1] + R[5] * t[2]), -(R[6] * t[0] + R[7] * t[1] + R[8] * t[2])]);
}
export function rotationOf(M: ArrayLike<number>): number[] { return [M[0], M[1], M[2], M[4], M[5], M[6], M[8], M[9], M[10]]; }
export function apply4(M: ArrayLike<number>, p: ArrayLike<number>): number[] {
  return [0, 1, 2].map((r) => M[r * 4] * p[0] + M[r * 4 + 1] * p[1] + M[r * 4 + 2] * p[2] + M[r * 4 + 3]);
}
// a column-major 4 x 4 (three.js, main.ts poseMatrices) at offset o, as row-major
export function fromColumnMajor(E: ArrayLike<number>, o = 0): M4 {
  const out = new Float64Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out[r * 4 + c] = E[o + c * 4 + r];
  return out;
}
// the nearest rotation of a matrix's linear part (two steps of the polar iteration), for skinning matrices that carry
// a little rounding
export function orthonormal(M: ArrayLike<number>): M4 {
  let R = rotationOf(M);
  for (let it = 0; it < 3; it++) {
    const det = R[0] * (R[4] * R[8] - R[5] * R[7]) - R[1] * (R[3] * R[8] - R[5] * R[6]) + R[2] * (R[3] * R[7] - R[4] * R[6]);
    const inv = [(R[4] * R[8] - R[5] * R[7]) / det, (R[2] * R[7] - R[1] * R[8]) / det, (R[1] * R[5] - R[2] * R[4]) / det,
      (R[5] * R[6] - R[3] * R[8]) / det, (R[0] * R[8] - R[2] * R[6]) / det, (R[2] * R[3] - R[0] * R[5]) / det,
      (R[3] * R[7] - R[4] * R[6]) / det, (R[1] * R[6] - R[0] * R[7]) / det, (R[0] * R[4] - R[1] * R[3]) / det];
    const invT = [inv[0], inv[3], inv[6], inv[1], inv[4], inv[7], inv[2], inv[5], inv[8]];
    R = R.map((v, i) => 0.5 * (v + invT[i]));
  }
  return m4(R, [M[3], M[7], M[11]]);
}

// ------------------------------------------------------------------ a place as a move of the rest frame
// the move of a place in Anny's frame about the head h (Anny's frame): x -> h + Q (x - h) + t
export function placeMatrix(p: Place, head: ArrayLike<number>): M4 {
  const Q = rotationMatrix(p.rotation || IDENTITY_Q), t = p.translation || [0, 0, 0];
  const Qh = [0, 1, 2].map((r) => Q[r * 3] * head[0] + Q[r * 3 + 1] * head[1] + Q[r * 3 + 2] * head[2]);
  return m4(Q, [head[0] - Qh[0] + t[0], head[1] - Qh[1] + t[1], head[2] - Qh[2] + t[2]]);
}
// the place of a move M of Anny's rest frame about the head of `bone` (Anny's frame)
export function placeOf(M: ArrayLike<number>, head: ArrayLike<number>, bone: string, digits = 7): Place {
  const f = 10 ** digits, r = (v: number) => Math.round(v * f) / f;
  const h = apply4(M, head);
  return { bone, rotation: quaternionOf(rotationOf(M)).map(r), translation: [h[0] - head[0], h[1] - head[1], h[2] - head[2]].map(r) };
}
// the same move in a similarity's frame: F M F^-1 (the page's frame from Anny's)
export function toFrame(M: ArrayLike<number>, F: Similarity): M4 {
  const R = F.rotation, s = F.scale, Fm = m4(R.map((v) => v * s), F.offset);
  const RT = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]].map((v) => v / s);
  const o = F.offset, Fi = m4(RT, [0, 1, 2].map((r) => -(RT[r * 3] * o[0] + RT[r * 3 + 1] * o[1] + RT[r * 3 + 2] * o[2])));
  return mul4(mul4(Fm, M), Fi);
}
export function fromFrame(M: ArrayLike<number>, F: Similarity): M4 {
  const R = F.rotation, s = F.scale;
  const RT = [R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8]].map((v) => v / s);
  const o = F.offset, Fi = m4(RT, [0, 1, 2].map((r) => -(RT[r * 3] * o[0] + RT[r * 3 + 1] * o[1] + RT[r * 3 + 2] * o[2])));
  return mul4(mul4(Fi, M), m4(R.map((v) => v * s), F.offset));
}
// a point of a similarity's frame in Anny's, and back
export function pointFrom(F: Similarity, p: ArrayLike<number>): number[] {
  const R = F.rotation, s = F.scale, d = [p[0] - F.offset[0], p[1] - F.offset[1], p[2] - F.offset[2]];
  return [0, 1, 2].map((c) => (R[c] * d[0] + R[3 + c] * d[1] + R[6 + c] * d[2]) / s);
}
export function pointTo(F: Similarity, x: ArrayLike<number>): number[] {
  const R = F.rotation, s = F.scale;
  return [0, 1, 2].map((r) => s * (R[r * 3] * x[0] + R[r * 3 + 1] * x[1] + R[r * 3 + 2] * x[2]) + F.offset[r]);
}

// the rest vertices and normals of a piece (a similarity's frame, `headPage` the carrying bone's rest head there) moved
// by its place
export function applyPlace(p: Place, vertices: ArrayLike<number>, normals: ArrayLike<number>, headPage: ArrayLike<number>, F: Similarity): { vertices: Float64Array; normals: Float64Array } {
  const M = toFrame(placeMatrix(p, pointFrom(F, headPage)), F), n = vertices.length / 3;
  const V = new Float64Array(n * 3), N = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) for (let r = 0; r < 3; r++) {
    const o = i * 3;
    V[o + r] = M[r * 4] * vertices[o] + M[r * 4 + 1] * vertices[o + 1] + M[r * 4 + 2] * vertices[o + 2] + M[r * 4 + 3];
    N[o + r] = M[r * 4] * normals[o] + M[r * 4 + 1] * normals[o + 1] + M[r * 4 + 2] * normals[o + 2];
  }
  return { vertices: V, normals: N };
}

// the page-frame move of a place (identity for none), and the place of a page-frame move about a bone's page head
export function pageMove(p: Place | null, headPage: ArrayLike<number>, F: Similarity): M4 {
  return p ? toFrame(placeMatrix(p, pointFrom(F, headPage)), F) : m4();
}
export function placeOfPage(M: ArrayLike<number>, headPage: ArrayLike<number>, bone: string, F: Similarity): Place {
  return placeOf(fromFrame(M, F), pointFrom(F, headPage), bone);
}
// the place that keeps a piece where it stands while `bone` takes it over: S the skinning matrices (rest to posed,
// row-major, the page's frame) of the carrying bone now and of the new one, heads their rest heads (page frame)
export function rehold(p: Place | null, sOld: ArrayLike<number>, sNew: ArrayLike<number>, headOld: ArrayLike<number>, headNew: ArrayLike<number>, bone: string, F: Similarity): Place {
  const A = pageMove(p, headOld, F);
  return placeOfPage(mul4(mul4(rigidInverse(orthonormal(sNew)), orthonormal(sOld)), A), headNew, bone, F);
}
// the place after a move W of the posed piece (W row-major in the posed page frame, about any point), carried by the
// bone whose skinning matrix is S
export function movedInPose(p: Place | null, carrier: string, S: ArrayLike<number>, W: ArrayLike<number>, head: ArrayLike<number>, F: Similarity): Place {
  const So = orthonormal(S), A = pageMove(p, head, F);
  return placeOfPage(mul4(mul4(mul4(rigidInverse(So), W), So), A), head, carrier, F);
}
// how far a place moves the piece from its package's place: the translation (mm) and the turn (degrees)
export function placeAmount(p: Place | null): { mm: number; deg: number } {
  if (!p) return { mm: 0, deg: 0 };
  const t = p.translation || [0, 0, 0], q = p.rotation || IDENTITY_Q;
  return { mm: 1000 * Math.hypot(t[0], t[1], t[2]), deg: 2 * Math.acos(Math.min(1, Math.abs(q[3]))) * 180 / Math.PI };
}
// a place's turn as angles about X, then Y, then Z of the card's frame (degrees, R = Rz Ry Rx), and back
export function eulerOf(q: ArrayLike<number>): number[] {
  const R = rotationMatrix(q), d = 180 / Math.PI, sy = -R[6];
  if (Math.abs(sy) > 0.999999) return [Math.atan2(-R[5], R[4]) * d, Math.asin(Math.max(-1, Math.min(1, sy))) * d, 0];
  return [Math.atan2(R[7], R[8]) * d, Math.asin(sy) * d, Math.atan2(R[3], R[0]) * d];
}
export function fromEuler(deg: ArrayLike<number>): number[] {
  const r = Math.PI / 180, X = axisAngle([1, 0, 0], deg[0] * r), Y = axisAngle([0, 1, 0], deg[1] * r), Z = axisAngle([0, 0, 1], deg[2] * r);
  const mul = (A: number[], B: number[]) => [0, 1, 2].flatMap((i) => [0, 1, 2].map((j) => A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j]));
  return quaternionOf(mul(Z, mul(Y, X)));
}

// ------------------------------------------------------------------ the card
// the entry of a piece on a card: an outfit's piece (<outfit>/<piece>) or a loose wearable (its id)
export function pieceEntry(card: any, ref: string): any | null {
  for (const o of card?.outfits || []) for (const p of o.pieces || []) if (`${o.id}/${p.piece}` === ref) return p;
  for (const e of card?.wearables || []) if ((e.id || String(e.file).split('/').pop()!.split('.')[0]) === ref) return e;
  return null;
}
export function cardPlace(card: any, ref: string): Place | null {
  const e = pieceEntry(card, ref);
  return e && e.place && !placeProblem(e.place) ? e.place : null;
}
// set or clear a piece's place on a card; whether the card holds that piece
export function setCardPlace(card: any, ref: string, p: Place | null): boolean {
  const e = pieceEntry(card, ref);
  if (!e) return false;
  if (p) e.place = { bone: p.bone, rotation: p.rotation.slice(), translation: p.translation.slice() }; else delete e.place;
  return true;
}
export function cardHasPlaces(card: any): boolean {
  return (card?.outfits || []).some((o: any) => (o.pieces || []).some((p: any) => p && p.place)) || (card?.wearables || []).some((e: any) => e && e.place);
}

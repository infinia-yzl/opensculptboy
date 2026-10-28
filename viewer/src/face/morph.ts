// OpenSculptBoy
// Apache License, Version 2.0
//
// The faces on the GPU (docs/viewer.md, "Faces"): the fine shapes of the face shown live in the layers of two array
// textures, one texel per vertex of the face's block (the vertices that any facial action or the jaw moves, which the
// build puts together in the vertex order), positions in one and the change of the normals in the other. The skin's
// vertex shaders add the weighted layers and turn the lower face with the jaw (FACE_GLSL), so a change of the face
// costs a few uniforms; a shape seen for the first time costs the upload of one layer. Layer 0 holds the jaw's
// corrective and, in its fourth channel, the jaw's weight. The page reads the same layers on the CPU for the few
// vertices it needs (the roots of the brows and the lashes, the skin under a card's parts).

import type * as THREE_NS from 'three';
import type { Shape } from './rig.ts';

export const FACE_WIDTH = 2048;
export const FACE_SLOTS = 16;    // the shapes that a face can show at once (the jaw's corrective aside)

// the vertex shader's part (with FACE_MORPH): fP, fN and fS take the place of position, normal and nsmooth
export const FACE_GLSL = /* glsl */`
#ifdef FACE_MORPH
uniform highp sampler2DArray uFaceP; uniform highp sampler2DArray uFaceN;
uniform int uFaceStart; uniform int uFaceLen; uniform int uFaceCount; uniform float uFaceOn;
uniform int uFaceSlot[${FACE_SLOTS}]; uniform float uFaceW[${FACE_SLOTS}];
uniform float uJawW; uniform mat3 uJawR; uniform vec3 uJawP;
vec3 fP; vec3 fN; vec3 fS;
void faceMorph(vec3 p, vec3 n, vec3 s) {
  fP = p; fN = n; fS = s;
  int r = gl_VertexID - uFaceStart;
  if (uFaceOn < 0.5 || r < 0 || r >= uFaceLen || (uFaceCount == 0 && uJawW == 0.0)) return;
  ivec2 c = ivec2(r % ${FACE_WIDTH}, r / ${FACE_WIDTH});
  vec3 dp = vec3(0.0), dn = vec3(0.0);
  for (int k = 0; k < ${FACE_SLOTS}; k++) {
    if (k >= uFaceCount) break;
    ivec3 q = ivec3(c, uFaceSlot[k]);
    dp += uFaceW[k] * texelFetch(uFaceP, q, 0).xyz;
    dn += uFaceW[k] * texelFetch(uFaceN, q, 0).xyz;
  }
  vec4 jaw = texelFetch(uFaceP, ivec3(c, 0), 0);
  if (uJawW > 0.0) { dp += uJawW * jaw.xyz; dn += uJawW * texelFetch(uFaceN, ivec3(c, 0), 0).xyz; }
  fP = p + dp;
  vec3 m = normalize(n + dn);
  if (uJawW > 0.0 && jaw.w > 0.0) {
    fP += jaw.w * ((uJawR - mat3(1.0)) * (fP - uJawP));
    m = normalize(m + jaw.w * ((uJawR - mat3(1.0)) * m));
  }
  fN = m;
  // the smooth shading normal turns with the normal, by the shortest turn
  vec3 k = cross(n, m); float co = dot(n, m), s2 = dot(k, k);
  if (s2 > 1e-12 && co > -0.9) fS = normalize(s * co + cross(k, s) + k * (dot(k, s) * (1.0 - co) / s2));
}
#define FACE_P fP
#define FACE_N fN
#define FACE_S fS
#else
void faceMorph(vec3 p, vec3 n, vec3 s) {}
#define FACE_P position
#define FACE_N normal
#define FACE_S nsmooth
#endif
`;

// the uniforms of FACE_GLSL (shared by every material of the body)
export function faceUniforms(T: typeof THREE_NS) {
  return {
    uFaceP: { value: null as any }, uFaceN: { value: null as any }, uFaceStart: { value: 0 }, uFaceLen: { value: 0 }, uFaceCount: { value: 0 }, uFaceOn: { value: 1 },
    uFaceSlot: { value: new Array(FACE_SLOTS).fill(0) }, uFaceW: { value: new Array(FACE_SLOTS).fill(0) },
    uJawW: { value: 0 }, uJawR: { value: new T.Matrix3() }, uJawP: { value: new T.Vector3() },
  };
}

// float32 to float16 (round to nearest), for the texels
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
export function toHalf(x: number): number {
  f32[0] = x;
  const b = u32[0], s = (b >>> 16) & 0x8000;
  let e = ((b >>> 23) & 0xff) - 112, m = b & 0x7fffff;
  if (e <= 0) { if (e < -10) return s; m |= 0x800000; const t = 14 - e; return s | ((m + (1 << (t - 1))) >> t); }
  if (e >= 31) return s | 0x7c00;
  const h = s | (e << 10) | (m >> 13);
  return (m & 0x1000) ? h + 1 : h;
}

// float16 to float32
export function fromHalf(h: number): number {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * m * 5.960464477539063e-8;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * Math.pow(2, e - 15);
}

export class FaceMorph {
  T: typeof THREE_NS;
  start: number; count: number; rows: number;
  texP: THREE_NS.DataArrayTexture; texN: THREE_NS.DataArrayTexture;
  u: ReturnType<typeof faceUniforms>;
  jawW: Float32Array;                           // per vertex of the block the jaw's weight
  private slotOf = new Map<Shape, number>(); private shapeIn: (Shape | null)[]; private used: number[];
  uploads = 0;
  constructor(T: typeof THREE_NS, u: ReturnType<typeof faceUniforms>, start: number, count: number) {
    this.T = T; this.u = u; this.start = start; this.count = count;
    this.rows = Math.ceil(count / FACE_WIDTH);
    const layers = FACE_SLOTS + 1, n = FACE_WIDTH * this.rows * 4 * layers;
    const make = () => {
      const t = new T.DataArrayTexture(new Uint16Array(n), FACE_WIDTH, this.rows, layers);
      t.format = T.RGBAFormat; t.type = T.HalfFloatType; t.minFilter = t.magFilter = T.NearestFilter;
      t.generateMipmaps = false; t.needsUpdate = true;
      return t;
    };
    this.texP = make(); this.texN = make();
    this.jawW = new Float32Array(count);
    this.shapeIn = new Array(layers).fill(null);
    this.used = new Array(layers).fill(0);
    u.uFaceP.value = this.texP; u.uFaceN.value = this.texN; u.uFaceStart.value = start; u.uFaceLen.value = count;
  }
  get bytes() { return 2 * FACE_WIDTH * this.rows * 4 * (FACE_SLOTS + 1) * 2; }

  // write a shape into a layer: its offsets, the change of its normals, and for layer 0 the jaw's weights
  private write(layer: number, s: Shape | null, jaw?: { idx: Uint32Array; w: Float32Array }) {
    const per = FACE_WIDTH * this.rows * 4, o0 = layer * per, P = this.texP.image.data as Uint16Array, N = this.texN.image.data as Uint16Array;
    P.fill(0, o0, o0 + per); N.fill(0, o0, o0 + per);
    const st = this.start, n = this.count;
    if (s) {
      for (let i = 0; i < s.idx.length; i++) {
        const r = s.idx[i] - st;
        if (r < 0 || r >= n) continue;
        const o = o0 + r * 4;
        for (let k = 0; k < 3; k++) P[o + k] = toHalf(s.d[i * 3 + k]);
      }
      if (s.nIdx && s.dn) for (let i = 0; i < s.nIdx.length; i++) {
        const r = s.nIdx[i] - st;
        if (r < 0 || r >= n) continue;
        const o = o0 + r * 4;
        for (let k = 0; k < 3; k++) N[o + k] = toHalf(s.dn[i * 3 + k]);
      }
    }
    if (jaw) {
      this.jawW.fill(0);
      for (let i = 0; i < jaw.idx.length; i++) { const r = jaw.idx[i] - st; if (r >= 0 && r < n) { P[o0 + r * 4 + 3] = toHalf(jaw.w[i]); this.jawW[r] = jaw.w[i]; } }
    }
    this.texP.addLayerUpdate(layer); this.texN.addLayerUpdate(layer);
    this.texP.needsUpdate = true; this.texN.needsUpdate = true;
    this.uploads++;
  }
  // the jaw's corrective and weights of the current body (layer 0)
  setJaw(corrective: Shape | null, weights: { idx: Uint32Array; w: Float32Array } | null) {
    this.write(0, corrective, weights || { idx: new Uint32Array(0), w: new Float32Array(0) });
  }
  // the layer of a shape, written when it has none (the layer used longest ago gives way)
  private slot(s: Shape, tick: number): number {
    let k = this.slotOf.get(s);
    if (k === undefined) {
      k = -1;
      for (let i = 1; i <= FACE_SLOTS; i++) if (!this.shapeIn[i]) { k = i; break; }
      if (k < 0) { k = 1; for (let i = 2; i <= FACE_SLOTS; i++) if (this.used[i] < this.used[k]) k = i; }
      const old = this.shapeIn[k];
      if (old) this.slotOf.delete(old);
      this.shapeIn[k] = s; this.slotOf.set(s, k);
      this.write(k, s);
    }
    this.used[k] = tick;
    return k;
  }
  private tick = 0;
  // the face: the shapes with their weights (the largest FACE_SLOTS), and the jaw's weight, turn (row-major) and pivot
  show(shapes: { s: Shape; w: number }[], jaw: { w: number; R: ArrayLike<number>; p: number[] } | null) {
    const list = shapes.filter((x) => x.w).sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, FACE_SLOTS);
    this.tick++;
    const u = this.u;
    list.forEach((x, i) => { u.uFaceSlot.value[i] = this.slot(x.s, this.tick); u.uFaceW.value[i] = x.w; });
    u.uFaceCount.value = list.length;
    this.shown = list.map((x) => ({ layer: this.slotOf.get(x.s)!, w: x.w }));
    if (jaw && jaw.w) {
      u.uJawW.value = jaw.w;
      const R = jaw.R;
      u.uJawR.value.set(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8]);
      u.uJawP.value.set(jaw.p[0], jaw.p[1], jaw.p[2]);
      this.jaw = { w: jaw.w, R: Array.from(R), p: jaw.p.slice() };
    } else { u.uJawW.value = 0; this.jaw = null; }
  }
  shown: { layer: number; w: number }[] = [];
  jaw: { w: number; R: number[]; p: number[] } | null = null;

  // the face's position of a vertex of the body (rest position x, y, z), as the shader gives it; `skip` leaves out the
  // layers of some shapes (a card's eye targets, whose parts take their own offsets)
  at(v: number, x: number, y: number, z: number, out: number[], skip?: Set<Shape>) {
    out[0] = x; out[1] = y; out[2] = z;
    const r = v - this.start;
    if (r < 0 || r >= this.count) return;
    const P = this.texP.image.data as Uint16Array, per = FACE_WIDTH * this.rows * 4;
    for (const { layer, w } of this.shown) {
      if (skip && skip.has(this.shapeIn[layer]!)) continue;
      const o = layer * per + r * 4;
      out[0] += w * fromHalf(P[o]); out[1] += w * fromHalf(P[o + 1]); out[2] += w * fromHalf(P[o + 2]);
    }
    const j = this.jaw;
    if (j) {
      const o = r * 4;
      out[0] += j.w * fromHalf(P[o]); out[1] += j.w * fromHalf(P[o + 1]); out[2] += j.w * fromHalf(P[o + 2]);
      const k = this.jawW[r];
      if (k) {
        const R = j.R, p = j.p, dx = out[0] - p[0], dy = out[1] - p[1], dz = out[2] - p[2];
        out[0] += k * ((R[0] - 1) * dx + R[1] * dy + R[2] * dz);
        out[1] += k * (R[3] * dx + (R[4] - 1) * dy + R[5] * dz);
        out[2] += k * (R[6] * dx + R[7] * dy + (R[8] - 1) * dz);
      }
    }
  }
  // forget the layers of shapes that changed (a new body's correctives, a card's shapes)
  forget(s: Shape) { const k = this.slotOf.get(s); if (k !== undefined) { this.slotOf.delete(s); this.shapeIn[k] = null; } }
  dispose() { this.texP.dispose(); this.texN.dispose(); }
}

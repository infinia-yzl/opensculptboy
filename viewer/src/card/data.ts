// OpenSculptBoy
// Apache License, Version 2.0
//
// What the page takes from a character card to show it as its exports show it (docs/viewer.md, "A card's look"),
// without the DOM and three.js, so that node runs it in the tests:
//   - the card's detail level that the page shows, and its offsets on the page's rows (opensculptboy.viewer.detail_rows
//     maps every row of the page's levels 1 and 2 to the row of the card topology's full level);
//   - the skin's maps and values (opensculptboy.material, the glTF export's skin material);
//   - the eyes of a fitted card (opensculptboy.eyes): the layout of the eye texture and the single-surface eyeball;
//   - the parts that the exports append: the reference's lashes, teeth and inner mouth, in the card's rest frame.

export interface Frame { scale: number; rotation: number[]; offset: number[] }   // Anny's frame to the page's: p = s R x + o

// the textures of the skin that the page reads, by slot of opensculptboy.export.detail.TEXTURE_SLOTS
export const SKIN_SLOTS = ['base_color', 'normal', 'roughness', 'head_base_color', 'head_normal', 'head_roughness'] as const;

export interface CardPlan {
  topology: string;
  levels: Record<number, string>;      // offsets file of each level
  shown: number;                       // the level the page shows (0 for the coarse level alone)
  textures: Record<string, string>;    // skin slots and eye_layout to files
  bakeLevel: number | null;
  skin: { roughness: number; specular: number; ior: number; normalScale: number; subsurface: number; radiusMm: number[] };
  eyes: null | { corneaRadiusMm: number; corneaRoughness: number; corneaIor: number; gaze: Record<string, number[]> | null };
  lashes: null | { mesh: string; baseColor: string | null; roughness: number };
  mouth: null | { mesh: string; baseColor: string | null; normal: string | null; roughness: string | null };
  teeth: null | { mesh: string; color: number[]; roughness: number };
  localChanges: Record<string, number>;
  notes: string[];
}

const num = (v: any, d: number) => (typeof v === 'number' && isFinite(v) ? v : d);

// the plan of a card: which files the page reads and which values it takes
export function cardPlan(card: any): CardPlan {
  const notes: string[] = [];
  const detail = card?.detail || {}, material = card?.material && typeof card.material === 'object' ? card.material : {};
  const levels: Record<number, string> = {};
  for (const l of detail.levels || []) if (l && typeof l.offsets === 'string') levels[Number(l.subdiv)] = l.offsets;
  const dt = detail.textures || {};
  const textures: Record<string, string> = {};
  // the detail layer's textures, and the skin material's, which take precedence (texture_images)
  for (const slot of [...SKIN_SLOTS, 'eye_layout']) if (typeof dt[slot] === 'string') textures[slot] = dt[slot];
  for (const slot of SKIN_SLOTS) {
    if (slot.startsWith('head_')) {
      const k = slot.slice(5);
      if (typeof material.head?.[k] === 'string') textures[slot] = material.head[k];
    } else if (typeof material[slot] === 'string') textures[slot] = material[slot];
  }
  const bakeLevel = typeof dt.subdiv === 'number' ? dt.subdiv : null;
  // the page shows the level of the bake, so that the normal maps match the surface they were baked on; without a
  // bake, the finest level up to 2
  const held = Object.keys(levels).map(Number).filter((l) => l >= 0 && l <= 2);
  let shown = 0;
  if (bakeLevel !== null && levels[bakeLevel] !== undefined) shown = bakeLevel;
  else if (held.length) shown = Math.max(...held);
  if (bakeLevel !== null && bakeLevel !== shown && textures.normal) notes.push(`the page shows level ${shown}; the normal maps were baked on level ${bakeLevel}`);
  const ss = material.subsurface || {};
  const skin = { roughness: num(material.roughness_factor, 1), specular: num(material.specular, 1), ior: num(material.ior, 1.5),
    normalScale: num(material.normal_scale, 1), subsurface: num(ss.weight, 0.1),
    radiusMm: Array.isArray(ss.radius_mm) && ss.radius_mm.length === 3 ? ss.radius_mm.map(Number) : [5.4, 2.1, 1.2] };
  // the eyes of a fitted card: on when its bake measured the reference's eyes or baked their layout, unless switched off
  let eyes: CardPlan['eyes'] = null;
  const re = material.reference_eyes, me = material.eyes;
  if (me !== false && (re || textures.eye_layout || me)) {
    if (textures.eye_layout) {
      const auto = (k: string, d: number) => (typeof me?.[k] === 'number' ? me[k] : num(re?.[k], d));
      eyes = { corneaRadiusMm: auto('cornea_radius_mm', 7.8), corneaRoughness: auto('cornea_roughness', 0.02), corneaIor: auto('cornea_ior', 1.376),
        gaze: re?.gaze && me?.gaze !== 'straight' ? re.gaze : null };
    } else notes.push('the card has no baked eye layout; the page keeps its own eyes');
  }
  let lashes: CardPlan['lashes'] = null;
  const la = material.lashes;
  if (eyes && la !== false && la?.mesh && la.source !== 'generic') lashes = { mesh: la.mesh, baseColor: la.base_color || null, roughness: num(la.roughness, 1) };
  let mouth: CardPlan['mouth'] = null;
  const mo = material.mouth;
  if (mo?.mesh) mouth = { mesh: mo.mesh, baseColor: mo.base_color || null, normal: mo.normal || null, roughness: mo.roughness || null };
  let teeth: CardPlan['teeth'] = null;
  const te = material.teeth;
  if (!mouth && te && te !== false) {
    if (te.mesh && te.source !== 'generic') teeth = { mesh: te.mesh, color: Array.isArray(te.color) ? te.color : [0.72, 0.66, 0.56], roughness: num(te.roughness, 0.35) };
    else notes.push('the page leaves out the generic teeth');
  }
  if (card?.facial_actions && Object.values(card.facial_actions).some((v: any) => v)) notes.push('the page leaves out the card\'s facial actions');
  return { topology: String(card?.topology || 'anny').replace(/-quads$/, ''), levels, shown, textures, bakeLevel, skin, eyes, lashes, mouth, teeth,
    localChanges: card?.local_changes && typeof card.local_changes === 'object' ? card.local_changes : {}, notes };
}

// vectors of Anny's frame in the page's frame, one per page row: s R values[rows[i]] (rows: identity when null)
export function pageVectors(values: ArrayLike<number>, rows: ArrayLike<number> | null, n: number, frame: Frame): Float32Array {
  const R = frame.rotation, s = frame.scale, out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const j = (rows ? rows[i] : i) * 3, x = values[j], y = values[j + 1], z = values[j + 2];
    for (let r = 0; r < 3; r++) out[i * 3 + r] = s * (R[r * 3] * x + R[r * 3 + 1] * y + R[r * 3 + 2] * z);
  }
  return out;
}

// points of Anny's frame (a card's rest frame) in the page's frame, lifted by the body's floor shift
export function pagePoints(values: ArrayLike<number>, frame: Frame, lift: number): Float32Array {
  const n = values.length / 3, out = pageVectors(values, null, n, frame), o = frame.offset;
  for (let i = 0; i < n; i++) { out[i * 3] += o[0]; out[i * 3 + 1] += o[1] + lift; out[i * 3 + 2] += o[2]; }
  return out;
}

// ------------------------------------------------------------------ eyes (opensculptboy.eyes)
// the layout of the eye texture (opensculptboy.eyes.texture.polar_to_uv), glTF convention (v = 0 at the top): s is the
// distance from the gaze axis in eyeball radii in front and 2 minus it behind, phi the angle about the gaze from +X up
export const FRONT_SHARE = 0.9;
export function polarToUv(s: number, phi: number, left: boolean): [number, number] {
  let t = s <= 1 ? 0.5 * FRONT_SHARE * s : 0.5 * FRONT_SHARE + 0.5 * (1 - FRONT_SHARE) * (s - 1);
  t = Math.min(t, 0.5);
  const cu = left ? 0.25 : 0.75;
  return [cu + 0.5 * t * Math.cos(phi), 1 - (0.5 + t * Math.sin(phi))];
}

// the axes of an eye (columns +X, up, gaze in Anny's frame, row-major 3x3) turned by the smallest turn from -Y to gaze
// (opensculptboy.texture.eyes.turned_axes)
export function eyeAxes(gaze: number[] | null): number[] {
  const A = [1, 0, 0, 0, 0, -1, 0, 1, 0];   // rows of the matrix whose columns are (1, 0, 0), (0, 0, 1) and (0, -1, 0)
  if (!gaze) return A;
  const l = Math.hypot(gaze[0], gaze[1], gaze[2]) || 1, g = [gaze[0] / l, gaze[1] / l, gaze[2] / l], f = [0, -1, 0];
  const c = f[0] * g[0] + f[1] * g[1] + f[2] * g[2];
  const k = [f[1] * g[2] - f[2] * g[1], f[2] * g[0] - f[0] * g[2], f[0] * g[1] - f[1] * g[0]];
  const s2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
  if (s2 < 1e-16) return A;
  // Rodrigues: R = I + [k]x + [k]x^2 (1 - c) / s^2
  const K = [0, -k[2], k[1], k[2], 0, -k[0], -k[1], k[0], 0], R = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const K2 = new Array(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let m = 0; m < 3; m++) K2[i * 3 + j] += K[i * 3 + m] * K[m * 3 + j];
  for (let i = 0; i < 9; i++) R[i] += K[i] + K2[i] * (1 - c) / s2;
  const out = new Array(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let m = 0; m < 3; m++) out[i * 3 + j] += R[i * 3 + m] * A[m * 3 + j];
  return out;
}

// the distance from the eye's centre of the single-surface eyeball (opensculptboy.eyes.geometry.outer_radius with a shell
// of 1) at the angle theta from the gaze, in eyeball radii, for a limbus of radius iris and a cornea of radius cornea
export function outerRadius(theta: number, iris: number, cornea: number, bulge: number): number {
  const rhoL = iris, zJ = Math.sqrt(Math.max(1 - iris * iris, 0)), rc = Math.max(cornea, rhoL * 1.0001);
  const zcc = zJ - Math.sqrt(rc * rc - rhoL * rhoL);
  if (!(Math.sin(theta) < rhoL && Math.cos(theta) > 0)) return 1;
  let lo = 0.9, hi = 1.5;
  for (let i = 0; i < 50; i++) {
    const mid = 0.5 * (lo + hi), rho = mid * Math.sin(theta), z = mid * Math.cos(theta);
    const surface = zJ + bulge * (zcc + Math.sqrt(Math.max(rc * rc - rho * rho, 0)) - zJ);
    if (z > surface) hi = mid; else lo = mid;
  }
  return 0.5 * (lo + hi);
}

// the limbus of a baked eye in the layout (the pupil and the iris darker than the sclera): the radius in eyeball radii
// where the mean brightness of the rings about the centre of the square first rises past the middle between the iris
// and the sclera; lum(u, v) gives the brightness at a point of the texture (glTF convention)
export function irisRadius(lum: (u: number, v: number) => number, left: boolean): number {
  const ring = (s: number) => {
    let a = 0;
    for (let k = 0; k < 64; k++) { const [u, v] = polarToUv(s, (k / 64) * 2 * Math.PI, left); a += lum(u, v); }
    return a / 64;
  };
  const steps: number[] = [];
  for (let s = 0.05; s <= 0.95; s += 0.01) steps.push(ring(s));
  const iris = steps.slice(5, 25).reduce((x, y) => x + y, 0) / 20, sclera = steps.slice(-15).reduce((x, y) => x + y, 0) / 15;
  const mid = 0.5 * (iris + sclera);
  for (let i = 10; i < steps.length; i++) if (steps[i] > mid) return 0.05 + i * 0.01;
  return 0.4;
}

// ------------------------------------------------------------------ the parts in the card's rest frame
export interface Part { points: Float64Array; normals: Float64Array | null; uvs: Float32Array | null; triangles: Uint32Array }

// a part file of the bake (opensculptboy/lashes@1, opensculptboy/teeth@1, opensculptboy/inner-mouth@1): points,
// normals, UVs (OBJ convention, turned to glTF's) and triangles
export function partOf(tensors: Record<string, { values: ArrayLike<number> }>): Part {
  const P = tensors.points, N = tensors.normals, U = tensors.uvs, T = tensors.triangles;
  if (!P || !T) throw new Error('the file holds no points and triangles');
  let uvs: Float32Array | null = null;
  if (U) { uvs = Float32Array.from(U.values); for (let i = 1; i < uvs.length; i += 2) uvs[i] = 1 - uvs[i]; }
  return { points: Float64Array.from(P.values), normals: N ? Float64Array.from(N.values) : null, uvs, triangles: Uint32Array.from(T.values) };
}

// per-vertex tangents (x, y, z, handedness) from glTF UVs: the UV gradient summed over the triangles and made
// orthogonal to the normal, as the garments take them
export function uvTangents(P: ArrayLike<number>, N: ArrayLike<number>, UV: ArrayLike<number>, T: ArrayLike<number>, out?: Float32Array, uvScale = 1): Float32Array {
  const n = P.length / 3, t1 = new Float64Array(n * 3), t2 = new Float64Array(n * 3), res = out || new Float32Array(n * 4);
  for (let t = 0; t < T.length; t += 3) {
    const a = T[t], b = T[t + 1], c = T[t + 2];
    const x1 = P[b * 3] - P[a * 3], y1 = P[b * 3 + 1] - P[a * 3 + 1], z1 = P[b * 3 + 2] - P[a * 3 + 2];
    const x2 = P[c * 3] - P[a * 3], y2 = P[c * 3 + 1] - P[a * 3 + 1], z2 = P[c * 3 + 2] - P[a * 3 + 2];
    const s1 = (UV[b * 2] - UV[a * 2]) * uvScale, v1 = (UV[b * 2 + 1] - UV[a * 2 + 1]) * uvScale;
    const s2 = (UV[c * 2] - UV[a * 2]) * uvScale, v2 = (UV[c * 2 + 1] - UV[a * 2 + 1]) * uvScale;
    const r = 1 / ((s1 * v2 - s2 * v1) || 1e-12);
    const sx = (v2 * x1 - v1 * x2) * r, sy = (v2 * y1 - v1 * y2) * r, sz = (v2 * z1 - v1 * z2) * r;
    const tx = (s1 * x2 - s2 * x1) * r, ty = (s1 * y2 - s2 * y1) * r, tz = (s1 * z2 - s2 * z1) * r;
    t1[a * 3] += sx; t1[a * 3 + 1] += sy; t1[a * 3 + 2] += sz; t2[a * 3] += tx; t2[a * 3 + 1] += ty; t2[a * 3 + 2] += tz;
    t1[b * 3] += sx; t1[b * 3 + 1] += sy; t1[b * 3 + 2] += sz; t2[b * 3] += tx; t2[b * 3 + 1] += ty; t2[b * 3 + 2] += tz;
    t1[c * 3] += sx; t1[c * 3 + 1] += sy; t1[c * 3 + 2] += sz; t2[c * 3] += tx; t2[c * 3 + 1] += ty; t2[c * 3 + 2] += tz;
  }
  for (let i = 0; i < n; i++) {
    const o = i * 3, nx = N[o], ny = N[o + 1], nz = N[o + 2], d = t1[o] * nx + t1[o + 1] * ny + t1[o + 2] * nz;
    let x = t1[o] - d * nx, y = t1[o + 1] - d * ny, z = t1[o + 2] - d * nz;
    const l = Math.sqrt(x * x + y * y + z * z) || 1; x /= l; y /= l; z /= l;
    // glTF's v runs down the texture, so the bitangent is n x t times w
    const bx = ny * z - nz * y, by = nz * x - nx * z, bz = nx * y - ny * x;
    const w = bx * t2[o] + by * t2[o + 1] + bz * t2[o + 2] < 0 ? -1 : 1;
    res[i * 4] = x; res[i * 4 + 1] = y; res[i * 4 + 2] = z; res[i * 4 + 3] = w;
  }
  return res;
}

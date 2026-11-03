// OpenSculptBoy
// Apache License, Version 2.0
//
// A bounding volume hierarchy over a triangle mesh, for the pose editor's work on a scan of a few hundred thousand
// triangles: rays from the camera through the pointer (every crossing, so that a dragged point keeps the surface
// nearest to where it was), rays along the skin's normal for the first placement, and the closest point of the
// surface. The tree is built by splitting each node at the median of its longest axis (the "object median" split of
// Wald, Boulos and Shirley, "Ray tracing deformable scenes using dynamic bounding volume hierarchies", 2007), with
// the triangles of a leaf stored together; the ray-triangle test is Moller and Trumbore's (1997), as in
// opensculptboy.posing.reference_points.ray_hits, and the closest point on a triangle Ericson's ("Real-Time
// Collision Detection", 5.1.5), as in point_place.closest_on_triangles.

export interface RayHit { t: number; tri: number; u: number; v: number }   // barycentric (1 - u - v, u, v)
export interface Closest { point: [number, number, number]; tri: number; distance: number }

const LEAF = 8;

export class TriangleBVH {
  readonly P: Float32Array | Float64Array;
  readonly I: Uint32Array;
  readonly count: number;
  // nodes: bounds (6 per node), then first child or first triangle, and the count (0 for an inner node)
  private box: Float32Array;
  private node: Uint32Array;
  private order: Uint32Array;
  private nodes = 0;
  readonly maxEdge: number;

  constructor(positions: Float32Array | Float64Array, indices: Uint32Array) {
    this.P = positions; this.I = indices;
    const n = (this.count = indices.length / 3);
    const cent = new Float32Array(n * 3), lo = new Float32Array(n * 3), hi = new Float32Array(n * 3);
    let edge = 0;
    for (let t = 0; t < n; t++) {
      const a = indices[t * 3] * 3, b = indices[t * 3 + 1] * 3, c = indices[t * 3 + 2] * 3;
      for (let k = 0; k < 3; k++) {
        const x = positions[a + k], y = positions[b + k], z = positions[c + k];
        lo[t * 3 + k] = Math.min(x, y, z); hi[t * 3 + k] = Math.max(x, y, z); cent[t * 3 + k] = (x + y + z) / 3;
      }
      const e = Math.hypot(positions[a] - positions[b], positions[a + 1] - positions[b + 1], positions[a + 2] - positions[b + 2]);
      if (e > edge) edge = e;
    }
    this.maxEdge = edge;
    this.order = new Uint32Array(n);
    for (let i = 0; i < n; i++) this.order[i] = i;
    const cap = Math.max(1, 2 * Math.ceil(n / LEAF) + 1);
    this.box = new Float32Array(cap * 6);
    this.node = new Uint32Array(cap * 2);
    // iterative build: a stack of (node, start, end)
    const stack: number[] = [this.alloc(), 0, n];
    while (stack.length) {
      const end = stack.pop()!, start = stack.pop()!, id = stack.pop()!;
      const B = this.box, o = id * 6;
      B[o] = B[o + 1] = B[o + 2] = Infinity; B[o + 3] = B[o + 4] = B[o + 5] = -Infinity;
      let c0 = Infinity, c1 = Infinity, c2 = Infinity, d0 = -Infinity, d1 = -Infinity, d2 = -Infinity;
      for (let i = start; i < end; i++) {
        const t = this.order[i] * 3;
        for (let k = 0; k < 3; k++) { if (lo[t + k] < B[o + k]) B[o + k] = lo[t + k]; if (hi[t + k] > B[o + 3 + k]) B[o + 3 + k] = hi[t + k]; }
        const x = cent[t], y = cent[t + 1], z = cent[t + 2];
        if (x < c0) c0 = x; if (x > d0) d0 = x; if (y < c1) c1 = y; if (y > d1) d1 = y; if (z < c2) c2 = z; if (z > d2) d2 = z;
      }
      if (end - start <= LEAF) { this.node[id * 2] = start; this.node[id * 2 + 1] = end - start; continue; }
      const ex = [d0 - c0, d1 - c1, d2 - c2], axis = ex[0] >= ex[1] && ex[0] >= ex[2] ? 0 : ex[1] >= ex[2] ? 1 : 2;
      const mid = (start + end) >> 1;
      this.select(cent, axis, start, end, mid);
      const left = this.alloc(), right = this.alloc();
      this.node[id * 2] = left; this.node[id * 2 + 1] = 0;
      if (right !== left + 1) throw new Error('BVH: children out of order');
      stack.push(left, start, mid, right, mid, end);
    }
  }

  private alloc(): number {
    const id = this.nodes++;
    if (id * 2 + 2 > this.node.length) {
      const nb = new Float32Array(this.box.length * 2); nb.set(this.box); this.box = nb;
      const nn = new Uint32Array(this.node.length * 2); nn.set(this.node); this.node = nn;
    }
    return id;
  }

  // the k-th smallest centre along an axis at index mid (Hoare's selection), so the halves split at the median
  private select(cent: Float32Array, axis: number, lo: number, hi: number, mid: number) {
    const O = this.order;
    let l = lo, r = hi - 1;
    while (l < r) {
      const pivot = cent[O[(l + r) >> 1] * 3 + axis];
      let i = l, j = r;
      while (i <= j) {
        while (cent[O[i] * 3 + axis] < pivot) i++;
        while (cent[O[j] * 3 + axis] > pivot) j--;
        if (i <= j) { const s = O[i]; O[i] = O[j]; O[j] = s; i++; j--; }
      }
      if (mid <= j) r = j; else if (mid >= i) l = i; else break;
    }
  }

  // the ray's entry into a node's box, or Infinity
  private enter(o: number, ox: number, oy: number, oz: number, ix: number, iy: number, iz: number, tmax: number): number {
    const B = this.box;
    let t0 = 0, t1 = tmax;
    let a = (B[o] - ox) * ix, b = (B[o + 3] - ox) * ix;
    if (a > b) { const s = a; a = b; b = s; }
    if (a > t0) t0 = a; if (b < t1) t1 = b;
    if (t0 > t1) return Infinity;
    a = (B[o + 1] - oy) * iy; b = (B[o + 4] - oy) * iy;
    if (a > b) { const s = a; a = b; b = s; }
    if (a > t0) t0 = a; if (b < t1) t1 = b;
    if (t0 > t1) return Infinity;
    a = (B[o + 2] - oz) * iz; b = (B[o + 5] - oz) * iz;
    if (a > b) { const s = a; a = b; b = s; }
    if (a > t0) t0 = a; if (b < t1) t1 = b;
    return t0 > t1 ? Infinity : t0;
  }

  // Moller and Trumbore's test of one triangle; the hit's distance or -1
  private hitTri(t: number, ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, out: RayHit): boolean {
    const P = this.P, I = this.I, a = I[t * 3] * 3, b = I[t * 3 + 1] * 3, c = I[t * 3 + 2] * 3;
    const e1x = P[b] - P[a], e1y = P[b + 1] - P[a + 1], e1z = P[b + 2] - P[a + 2];
    const e2x = P[c] - P[a], e2y = P[c + 1] - P[a + 1], e2z = P[c + 2] - P[a + 2];
    const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
    const det = e1x * px + e1y * py + e1z * pz;
    if (Math.abs(det) <= 1e-14) return false;
    const inv = 1 / det, sx = ox - P[a], sy = oy - P[a + 1], sz = oz - P[a + 2];
    const u = (sx * px + sy * py + sz * pz) * inv;
    if (u < 0 || u > 1) return false;
    const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
    const v = (dx * qx + dy * qy + dz * qz) * inv;
    if (v < 0 || u + v > 1) return false;
    const d = (e2x * qx + e2y * qy + e2z * qz) * inv;
    if (!(d > 1e-9)) return false;
    out.t = d; out.tri = t; out.u = u; out.v = v;
    return true;
  }

  // every crossing of a ray within tmax, nearest first
  rayAll(o: ArrayLike<number>, d: ArrayLike<number>, tmax = Infinity): RayHit[] {
    const out: RayHit[] = [];
    this.walk(o, d, tmax, (h) => { out.push({ ...h }); return tmax; });
    return out.sort((a, b) => a.t - b.t);
  }

  // the nearest crossing of a ray within tmax, or null
  ray(o: ArrayLike<number>, d: ArrayLike<number>, tmax = Infinity): RayHit | null {
    let best: RayHit | null = null;
    this.walk(o, d, tmax, (h) => { if (!best || h.t < best.t) best = { ...h }; return best.t; });
    return best;
  }

  private walk(o: ArrayLike<number>, d: ArrayLike<number>, tmax: number, hit: (h: RayHit) => number) {
    if (!this.count) return;
    const ox = o[0], oy = o[1], oz = o[2], dx = d[0], dy = d[1], dz = d[2];
    const ix = 1 / dx, iy = 1 / dy, iz = 1 / dz, tmp: RayHit = { t: 0, tri: 0, u: 0, v: 0 };
    const stack = [0];
    let limit = tmax;
    while (stack.length) {
      const id = stack.pop()!;
      if (this.enter(id * 6, ox, oy, oz, ix, iy, iz, limit) === Infinity) continue;
      const first = this.node[id * 2], n = this.node[id * 2 + 1];
      if (n) {
        for (let i = first; i < first + n; i++) {
          if (this.hitTri(this.order[i], ox, oy, oz, dx, dy, dz, tmp) && tmp.t <= limit) limit = hit(tmp);
        }
      } else stack.push(first, first + 1);
    }
  }

  // the closest point of the surface to p within maxDist, or null
  closest(p: ArrayLike<number>, maxDist = Infinity): Closest | null {
    if (!this.count) return null;
    let best = maxDist * maxDist, bestTri = -1;
    const q: [number, number, number] = [0, 0, 0], tmp: [number, number, number] = [0, 0, 0];
    const B = this.box, px = p[0], py = p[1], pz = p[2];
    const boxDist = (id: number) => {
      const o = id * 6;
      const x = Math.max(B[o] - px, 0, px - B[o + 3]), y = Math.max(B[o + 1] - py, 0, py - B[o + 4]), z = Math.max(B[o + 2] - pz, 0, pz - B[o + 5]);
      return x * x + y * y + z * z;
    };
    const stack = [0];
    while (stack.length) {
      const id = stack.pop()!;
      if (boxDist(id) > best) continue;
      const first = this.node[id * 2], n = this.node[id * 2 + 1];
      if (n) {
        for (let i = first; i < first + n; i++) {
          const t = this.order[i];
          const d2 = closestOnTriangle(this.P, this.I, t, px, py, pz, tmp);
          if (d2 < best) { best = d2; bestTri = t; q[0] = tmp[0]; q[1] = tmp[1]; q[2] = tmp[2]; }
        }
      } else {
        // the nearer child last, so it comes off the stack first
        const a = boxDist(first), b = boxDist(first + 1);
        if (a < b) stack.push(first + 1, first); else stack.push(first, first + 1);
      }
    }
    return bestTri < 0 ? null : { point: q, tri: bestTri, distance: Math.sqrt(best) };
  }

  // a triangle's unit normal (right-handed order of its corners)
  normal(t: number): [number, number, number] {
    const P = this.P, I = this.I, a = I[t * 3] * 3, b = I[t * 3 + 1] * 3, c = I[t * 3 + 2] * 3;
    const e1 = [P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]], e2 = [P[c] - P[a], P[c + 1] - P[a + 1], P[c + 2] - P[a + 2]];
    const n: [number, number, number] = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const l = Math.max(Math.hypot(n[0], n[1], n[2]), 1e-30);
    return [n[0] / l, n[1] / l, n[2] / l];
  }
}

// Ericson's closest point of triangle t to p (5.1.5): the point in out, the squared distance returned
export function closestOnTriangle(P: ArrayLike<number>, I: ArrayLike<number>, t: number, px: number, py: number, pz: number, out: number[]): number {
  const a = I[t * 3] * 3, b = I[t * 3 + 1] * 3, c = I[t * 3 + 2] * 3;
  const ax = P[a], ay = P[a + 1], az = P[a + 2], bx = P[b], by = P[b + 1], bz = P[b + 2], cx = P[c], cy = P[c + 1], cz = P[c + 2];
  const abx = bx - ax, aby = by - ay, abz = bz - az, acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
  const set = (x: number, y: number, z: number) => { out[0] = x; out[1] = y; out[2] = z; const dx = px - x, dy = py - y, dz = pz - z; return dx * dx + dy * dy + dz * dz; };
  if (d1 <= 0 && d2 <= 0) return set(ax, ay, az);
  const bpx = px - bx, bpy = py - by, bpz = pz - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) return set(bx, by, bz);
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / (d1 - d3); return set(ax + v * abx, ay + v * aby, az + v * abz); }
  const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) return set(cx, cy, cz);
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / (d2 - d6); return set(ax + w * acx, ay + w * acy, az + w * acz); }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return set(bx + w * (cx - bx), by + w * (cy - by), bz + w * (cz - bz));
  }
  const denom = 1 / (va + vb + vc), v = vb * denom, w = vc * denom;
  return set(ax + abx * v + acx * w, ay + aby * v + acy * w, az + abz * v + acz * w);
}

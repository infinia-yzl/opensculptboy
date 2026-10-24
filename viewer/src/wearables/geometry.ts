// OpenSculptBoy
// Apache License, Version 2.0
//
// The surface queries of the wearables in the page, as opensculptboy.wearables.geometry computes them: vertex normals
// and scales, the frames of bound points, the closest point of a triangle mesh (the 24 triangles of the nearest
// centroids, then Ericson's exact point to triangle solve), signed distances, ray hits and the vertex adjacency.
// Every array is a flat Float64Array or Int32Array of rows of 3; the module has no dependencies, so node runs it in
// the parity test (test/test_viewer_wearables.py) and a worker can run it.

export type F64 = Float64Array;
export type I32 = Int32Array;

// candidate triangles per query point (opensculptboy.wearables.geometry.CANDIDATES)
export const CANDIDATES = 24;

// the length of a vector as NumPy's norm sums it
export const norm3 = (x: number, y: number, z: number) => Math.sqrt(x * x + y * y + z * z);

// area-weighted unit vertex normals of a triangle mesh; a vertex of no triangle keeps a zero normal
export function vertexNormals(V: F64, T: I32, n: number): F64 {
  const N = new Float64Array(n * 3);
  for (let t = 0; t < T.length; t += 3) {
    const a = T[t] * 3, b = T[t + 1] * 3, c = T[t + 2] * 3;
    const e1x = V[b] - V[a], e1y = V[b + 1] - V[a + 1], e1z = V[b + 2] - V[a + 2];
    const e2x = V[c] - V[a], e2y = V[c + 1] - V[a + 1], e2z = V[c + 2] - V[a + 2];
    const x = e1y * e2z - e1z * e2y, y = e1z * e2x - e1x * e2z, z = e1x * e2y - e1y * e2x;
    N[a] += x; N[a + 1] += y; N[a + 2] += z; N[b] += x; N[b + 1] += y; N[b + 2] += z; N[c] += x; N[c + 1] += y; N[c + 2] += z;
  }
  return unitRows(N);
}

// the mean length of the edges at each vertex (0 for a vertex of no triangle)
export function vertexScales(V: F64, T: I32, n: number): F64 {
  const total = new Float64Array(n), count = new Float64Array(n);
  for (let t = 0; t < T.length; t += 3) {
    for (const [i, j] of [[0, 1], [1, 2], [2, 0]]) {
      const p = T[t + i], q = T[t + j];
      const l = norm3(V[p * 3] - V[q * 3], V[p * 3 + 1] - V[q * 3 + 1], V[p * 3 + 2] - V[q * 3 + 2]);
      total[p] += l; count[p] += 1; total[q] += l; count[q] += 1;
    }
  }
  for (let i = 0; i < n; i++) total[i] /= Math.max(count[i], 1);
  return total;
}

export function unitRows(v: F64): F64 {
  for (let o = 0; o < v.length; o += 3) {
    const l = norm3(v[o], v[o + 1], v[o + 2]);
    if (l > 0) { v[o] /= l; v[o + 1] /= l; v[o + 2] /= l; }
  }
  return v;
}

// barycentric blends of per-vertex values of `width` over triangle corners (n, 3)
export function interpolate(values: F64, corners: I32, bary: F64, width = 3): F64 {
  const n = corners.length / 3, out = new Float64Array(n * width);
  for (let i = 0; i < n; i++) {
    const a = corners[i * 3] * width, b = corners[i * 3 + 1] * width, c = corners[i * 3 + 2] * width;
    const u = bary[i * 3], v = bary[i * 3 + 1], w = bary[i * 3 + 2];
    for (let k = 0; k < width; k++) out[i * width + k] = (u * values[a + k] + v * values[b + k]) + w * values[c + k];
  }
  return out;
}

// the frames of bound points: positions (n, 3), unit axes as rows (tangent, bitangent, normal) (n, 9) and scales (n)
// (opensculptboy.wearables.geometry.frames)
export function frames(V: F64, N: F64, S: F64, corners: I32, bary: F64): { point: F64; axes: F64; scale: F64 } {
  const n = corners.length / 3;
  const point = interpolate(V, corners, bary), nrm = unitRows(interpolate(N, corners, bary)), scale = interpolate(S, corners, bary, 1);
  const axes = new Float64Array(n * 9);
  for (let i = 0; i < n; i++) {
    const a = corners[i * 3] * 3, b = corners[i * 3 + 1] * 3, o = i * 3;
    const nx = nrm[o], ny = nrm[o + 1], nz = nrm[o + 2];
    let ex = V[b] - V[a], ey = V[b + 1] - V[a + 1], ez = V[b + 2] - V[a + 2];
    const d = ex * nx + ey * ny + ez * nz;
    ex -= d * nx; ey -= d * ny; ez -= d * nz;
    const l = norm3(ex, ey, ez);
    if (l > 0) { ex /= l; ey /= l; ez /= l; }
    const q = i * 9;
    axes[q] = ex; axes[q + 1] = ey; axes[q + 2] = ez;
    axes[q + 3] = ny * ez - nz * ey; axes[q + 4] = nz * ex - nx * ez; axes[q + 5] = nx * ey - ny * ex;
    axes[q + 6] = nx; axes[q + 7] = ny; axes[q + 8] = nz;
  }
  return { point, axes, scale };
}

const fin = (x: number) => (Number.isNaN(x) ? 0 : x === Infinity ? Number.MAX_VALUE : x === -Infinity ? -Number.MAX_VALUE : x);

// the barycentric coordinates of the closest point of triangle (a, b, c) to p, in the regions and the order of
// opensculptboy.wearables.geometry._closest_on_triangles (later regions win)
export function closestBary(P: F64, p: number, V: F64, a: number, b: number, c: number, out: F64) {
  const ax = V[a], ay = V[a + 1], az = V[a + 2], bx = V[b], by = V[b + 1], bz = V[b + 2], cx = V[c], cy = V[c + 1], cz = V[c + 2];
  const px = P[p], py = P[p + 1], pz = P[p + 2];
  const abx = bx - ax, aby = by - ay, abz = bz - az, acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
  const bpx = px - bx, bpy = py - by, bpz = pz - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
  const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
  const va = d3 * d6 - d5 * d4, vb = d5 * d2 - d1 * d6, vc = d1 * d4 - d3 * d2;
  const denom = va + vb + vc;
  const v = denom !== 0 ? vb / denom : 0, w = denom !== 0 ? vc / denom : 0;
  let u0 = 1 - v - w, u1 = v, u2 = w;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) { const e = (d4 - d3) / ((d4 - d3) + (d5 - d6)); u0 = 0; u1 = 1 - e; u2 = e; }
  if (vb <= 0 && d2 >= 0 && d6 <= 0) { const e = d2 / (d2 - d6); u0 = 1 - e; u1 = 0; u2 = e; }
  if (vc <= 0 && d1 >= 0 && d3 <= 0) { const e = d1 / (d1 - d3); u0 = 1 - e; u1 = e; u2 = 0; }
  if (d6 >= 0 && d5 <= d6) { u0 = 0; u1 = 0; u2 = 1; }
  if (d3 >= 0 && d4 <= d3) { u0 = 0; u1 = 1; u2 = 0; }
  if (d1 <= 0 && d2 <= 0) { u0 = 1; u1 = 0; u2 = 0; }
  out[0] = fin(u0); out[1] = fin(u1); out[2] = fin(u2);
}

// NumPy's round(x, 12): half to even
function round12(x: number): number {
  const y = x * 1e12, r = Math.round(y);
  return (Math.abs(y - Math.trunc(y)) === 0.5 && r % 2 !== 0 ? r - 1 : r) / 1e12;
}

// a uniform grid of points (the centroids of triangles), for nearest-neighbour and ball queries
export class PointGrid {
  lo: number[]; h: number; dims: number[]; start: I32; items: I32; P: F64; n: number;
  constructor(P: F64, cell: number) {
    const n = P.length / 3;
    this.P = P; this.n = n;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) { const x = P[i * 3 + k]; if (x < lo[k]) lo[k] = x; if (x > hi[k]) hi[k] = x; }
    if (!n) { lo.fill(0); hi.fill(0); }
    let h = Math.max(cell, 1e-9);
    const cells = (s: number) => [0, 1, 2].map((k) => Math.max(1, Math.floor((hi[k] - lo[k]) / s) + 1));
    while (cells(h).reduce((a, b) => a * b, 1) > 4e6) h *= 1.5;
    this.lo = lo; this.h = h; this.dims = cells(h);
    const [nx, ny, nz] = this.dims, count = new Int32Array(nx * ny * nz + 1), id = new Int32Array(n);
    for (let i = 0; i < n; i++) { id[i] = this.cellOf(P[i * 3], P[i * 3 + 1], P[i * 3 + 2]); count[id[i] + 1]++; }
    for (let c = 0; c < nx * ny * nz; c++) count[c + 1] += count[c];
    const fill = count.slice(0, nx * ny * nz), items = new Int32Array(n);
    for (let i = 0; i < n; i++) items[fill[id[i]]++] = i;
    this.start = count; this.items = items;
  }
  axis(x: number, k: number) { return Math.min(this.dims[k] - 1, Math.max(0, Math.floor((x - this.lo[k]) / this.h))); }
  cellOf(x: number, y: number, z: number) { return (this.axis(x, 0) * this.dims[1] + this.axis(y, 1)) * this.dims[2] + this.axis(z, 2); }

  // the k nearest points of q (sorted by distance, ties by index): their count; ids and squared distances in the arrays.
  // With `limit`, a point that has no point within that distance gets none.
  knn(qx: number, qy: number, qz: number, k: number, ids: I32, d2: F64, limit = Infinity): number {
    const P = this.P, nx = this.dims[0], ny = this.dims[1], nz = this.dims[2], h = this.h, start = this.start, items = this.items;
    const cx = this.axis(qx, 0), cy = this.axis(qy, 1), cz = this.axis(qz, 2), lx = this.lo[0], ly = this.lo[1], lz = this.lo[2];
    // how far q lies outside the grid along each axis
    const ox = Math.max(0, lx - qx, qx - (lx + nx * h)), oy = Math.max(0, ly - qy, qy - (ly + ny * h)), oz = Math.max(0, lz - qz, qz - (lz + nz * h));
    const lim2 = limit * limit;
    let m = 0;
    for (let r = 0; ; r++) {
      const x0 = cx - r, x1 = cx + r, y0 = cy - r, y1 = cy + r, z0 = cz - r, z1 = cz + r;
      const xa = Math.max(x0, 0), xb = Math.min(x1, nx - 1), ya = Math.max(y0, 0), yb = Math.min(y1, ny - 1), za = Math.max(z0, 0), zb = Math.min(z1, nz - 1);
      for (let x = xa; x <= xb; x++) for (let y = ya; y <= yb; y++) {
        const shell = x === x0 || x === x1 || y === y0 || y === y1, base = (x * ny + y) * nz;
        for (let z = za; z <= zb; z++) {
          if (!shell && z !== z0 && z !== z1) { z = z1 - 1; continue; }
          for (let s = start[base + z], e = start[base + z + 1]; s < e; s++) {
            const i = items[s], dx = P[i * 3] - qx, dy = P[i * 3 + 1] - qy, dz = P[i * 3 + 2] - qz, d = dx * dx + dy * dy + dz * dz;
            if (m === k && (d > d2[m - 1] || (d === d2[m - 1] && i > ids[m - 1]))) continue;
            let j = m < k ? m++ : m - 1;
            while (j > 0 && (d2[j - 1] > d || (d2[j - 1] === d && ids[j - 1] > i))) { d2[j] = d2[j - 1]; ids[j] = ids[j - 1]; j--; }
            d2[j] = d; ids[j] = i;
          }
        }
      }
      // the nearest point that the cells beyond ring r can hold: the distance to the part of the grid past each face
      let bound2 = Infinity, more = false;
      if (x0 > 0) { more = true; const d = Math.max(0, qx - (lx + x0 * h)); bound2 = Math.min(bound2, d * d + oy * oy + oz * oz); }
      if (x1 < nx - 1) { more = true; const d = Math.max(0, lx + (x1 + 1) * h - qx); bound2 = Math.min(bound2, d * d + oy * oy + oz * oz); }
      if (y0 > 0) { more = true; const d = Math.max(0, qy - (ly + y0 * h)); bound2 = Math.min(bound2, d * d + ox * ox + oz * oz); }
      if (y1 < ny - 1) { more = true; const d = Math.max(0, ly + (y1 + 1) * h - qy); bound2 = Math.min(bound2, d * d + ox * ox + oz * oz); }
      if (z0 > 0) { more = true; const d = Math.max(0, qz - (lz + z0 * h)); bound2 = Math.min(bound2, d * d + ox * ox + oy * oy); }
      if (z1 < nz - 1) { more = true; const d = Math.max(0, lz + (z1 + 1) * h - qz); bound2 = Math.min(bound2, d * d + ox * ox + oy * oy); }
      if (!more) return m;
      if (m === k && d2[m - 1] < bound2) return m;
      if (bound2 > lim2 && (m === 0 || d2[0] > lim2)) return 0;
    }
  }

  // the points within radius of q
  ball(qx: number, qy: number, qz: number, radius: number, out: number[]) {
    out.length = 0;
    const P = this.P, r2 = radius * radius, [, ny, nz] = this.dims;
    const a0 = this.axis(qx - radius, 0), a1 = this.axis(qx + radius, 0), b0 = this.axis(qy - radius, 1), b1 = this.axis(qy + radius, 1);
    const c0 = this.axis(qz - radius, 2), c1 = this.axis(qz + radius, 2);
    for (let x = a0; x <= a1; x++) for (let y = b0; y <= b1; y++) for (let z = c0; z <= c1; z++) {
      const cell = (x * ny + y) * nz + z;
      for (let s = this.start[cell]; s < this.start[cell + 1]; s++) {
        const i = this.items[s], dx = P[i * 3] - qx, dy = P[i * 3 + 1] - qy, dz = P[i * 3 + 2] - qz;
        if (dx * dx + dy * dy + dz * dz <= r2) out.push(i);
      }
    }
  }
}

export interface Closest { triangle: I32; bary: F64; point: F64; distance: F64 }

// a triangle mesh with its vertex normals and scales, for closest-point queries (opensculptboy.wearables.geometry.Surface)
export class Surface {
  V: F64; T: I32; N: F64; nv: number;
  radius = 0;   // the largest distance of a corner from its triangle's centroid
  private _scales: F64 | null = null; private _grid: PointGrid | null = null;
  constructor(V: F64, T: I32, normals?: F64) {
    this.V = V; this.T = T; this.nv = V.length / 3;
    this.N = normals || vertexNormals(V, T, this.nv);
  }
  get scales(): F64 { return this._scales || (this._scales = vertexScales(this.V, this.T, this.nv)); }
  get grid(): PointGrid {
    if (this._grid) return this._grid;
    const T = this.T, V = this.V, m = T.length / 3, C = new Float64Array(m * 3);
    let edge = 0;
    for (let t = 0; t < m; t++) {
      const a = T[t * 3] * 3, b = T[t * 3 + 1] * 3, c = T[t * 3 + 2] * 3;
      for (let k = 0; k < 3; k++) C[t * 3 + k] = (V[a + k] + V[b + k] + V[c + k]) / 3;
      edge += norm3(V[a] - V[b], V[a + 1] - V[b + 1], V[a + 2] - V[b + 2]);
      for (const p of [a, b, c]) this.radius = Math.max(this.radius, norm3(V[p] - C[t * 3], V[p + 1] - C[t * 3 + 1], V[p + 2] - C[t * 3 + 2]));
    }
    return (this._grid = new PointGrid(C, 2.5 * (m ? edge / m : 1)));
  }
  // the closest point of the surface to each point; with `limit`, a point farther than `limit` from every triangle may
  // get no triangle (distance Infinity)
  closest(points: F64, candidates = CANDIDATES, limit = Infinity): Closest {
    const n = points.length / 3, m = this.T.length / 3, k = Math.min(candidates, m), grid = this.grid, T = this.T, V = this.V;
    const triangle = new Int32Array(n), bary = new Float64Array(n * 3), point = new Float64Array(n * 3), distance = new Float64Array(n);
    const ids = new Int32Array(k), d2 = new Float64Array(k), b = new Float64Array(3), reach = limit + this.radius;
    for (let i = 0; i < n; i++) {
      const o = i * 3, got = grid.knn(points[o], points[o + 1], points[o + 2], k, ids, d2, reach);
      if (!got) { distance[i] = Infinity; continue; }
      let best = -1, bestR = Infinity, bd = Infinity, b0 = 0, b1 = 0, b2 = 0;
      for (let j = 0; j < got; j++) {
        const t = ids[j], a = T[t * 3] * 3, bb = T[t * 3 + 1] * 3, c = T[t * 3 + 2] * 3;
        closestBary(points, o, V, a, bb, c, b);
        const fx = b[0] * V[a] + b[1] * V[bb] + b[2] * V[c], fy = b[0] * V[a + 1] + b[1] * V[bb + 1] + b[2] * V[c + 1], fz = b[0] * V[a + 2] + b[1] * V[bb + 2] + b[2] * V[c + 2];
        const d = norm3(points[o] - fx, points[o + 1] - fy, points[o + 2] - fz), r = round12(d);
        if (r < bestR || (r === bestR && t < best)) { best = t; bestR = r; bd = d; b0 = b[0]; b1 = b[1]; b2 = b[2]; }
      }
      triangle[i] = Math.max(best, 0); bary[o] = b0; bary[o + 1] = b1; bary[o + 2] = b2; distance[i] = bd;
      const a = T[triangle[i] * 3] * 3, bb = T[triangle[i] * 3 + 1] * 3, c = T[triangle[i] * 3 + 2] * 3;
      for (let q = 0; q < 3; q++) point[o + q] = (b0 * V[a + q] + b1 * V[bb + q]) + b2 * V[c + q];
    }
    return { triangle, bary, point, distance };
  }
  // the distance along the surface normal at the closest point, the closest points and those normals
  signedDistance(points: F64, limit = Infinity): { signed: F64; found: Closest; normal: F64 } {
    const found = this.closest(points, CANDIDATES, limit), n = points.length / 3, T = this.T, N = this.N;
    const corners = new Int32Array(n * 3);
    for (let i = 0; i < n; i++) for (let q = 0; q < 3; q++) corners[i * 3 + q] = T[found.triangle[i] * 3 + q];
    const normal = unitRows(interpolate(N, corners, found.bary)), signed = new Float64Array(n);
    for (let i = 0, o = 0; i < n; i++, o += 3)
      if (found.distance[i] === Infinity) signed[i] = Infinity;
      else signed[i] = (points[o] - found.point[o]) * normal[o] + (points[o + 1] - found.point[o + 1]) * normal[o + 1] + (points[o + 2] - found.point[o + 2]) * normal[o + 2];
    return { signed, found, normal };
  }
}

// whether the ray from each origin along its direction meets a triangle within reach (Moller-Trumbore, as
// opensculptboy.wearables.geometry.ray_hits)
export function rayHits(origins: F64, directions: F64, V: F64, T: I32, reach: number): Uint8Array {
  const n = origins.length / 3, hit = new Uint8Array(n), m = T.length / 3;
  if (!m || !n) return hit;
  const D = unitRows(directions.slice()), C = new Float64Array(m * 3);
  let radius = 0, edge = 0;
  for (let t = 0; t < m; t++) {
    const a = T[t * 3] * 3, b = T[t * 3 + 1] * 3, c = T[t * 3 + 2] * 3;
    for (let k = 0; k < 3; k++) C[t * 3 + k] = (V[a + k] + V[b + k] + V[c + k]) / 3;
    for (const p of [a, b, c]) radius = Math.max(radius, norm3(V[p] - C[t * 3], V[p + 1] - C[t * 3 + 1], V[p + 2] - C[t * 3 + 2]));
    edge += norm3(V[a] - V[b], V[a + 1] - V[b + 1], V[a + 2] - V[b + 2]);
  }
  const grid = new PointGrid(C, Math.max(reach, 1.5 * edge / m)), near: number[] = [];
  for (let i = 0; i < n; i++) {
    const o = i * 3, ox = origins[o], oy = origins[o + 1], oz = origins[o + 2], dx = D[o], dy = D[o + 1], dz = D[o + 2];
    grid.ball(ox, oy, oz, reach + radius, near);
    for (const t of near) {
      const a = T[t * 3] * 3, b = T[t * 3 + 1] * 3, c = T[t * 3 + 2] * 3;
      const e1x = V[b] - V[a], e1y = V[b + 1] - V[a + 1], e1z = V[b + 2] - V[a + 2], e2x = V[c] - V[a], e2y = V[c + 1] - V[a + 1], e2z = V[c + 2] - V[a + 2];
      const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
      const det = e1x * px + e1y * py + e1z * pz;
      if (!(Math.abs(det) > 1e-14)) continue;
      const inv = 1 / det, sx = ox - V[a], sy = oy - V[a + 1], sz = oz - V[a + 2];
      const u = (sx * px + sy * py + sz * pz) * inv;
      const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
      const v = (dx * qx + dy * qy + dz * qz) * inv, s = (e2x * qx + e2y * qy + e2z * qz) * inv;
      if (u >= 0 && v >= 0 && u + v <= 1 && s > 0 && s <= reach) { hit[i] = 1; break; }
    }
  }
  return hit;
}

// the vertex adjacency of a triangle mesh in compressed rows (each neighbour once)
export interface Adjacency { start: I32; ids: I32; degree: F64 }
export function adjacency(count: number, T: I32): Adjacency {
  const lists: Set<number>[] = Array.from({ length: count }, () => new Set<number>());
  for (let t = 0; t < T.length; t += 3) {
    const a = T[t], b = T[t + 1], c = T[t + 2];
    lists[a].add(b); lists[b].add(c); lists[c].add(a); lists[b].add(a); lists[c].add(b); lists[a].add(c);
  }
  const start = new Int32Array(count + 1);
  for (let i = 0; i < count; i++) start[i + 1] = start[i] + lists[i].size;
  const ids = new Int32Array(start[count]), degree = new Float64Array(count);
  for (let i = 0; i < count; i++) { let k = start[i]; for (const j of [...lists[i]].sort((x, y) => x - y)) ids[k++] = j; degree[i] = lists[i].size; }
  return { start, ids, degree };
}

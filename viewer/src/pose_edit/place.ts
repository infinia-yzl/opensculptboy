// OpenSculptBoy
// Apache License, Version 2.0
//
// The first placement of the reference points on a scan: the twin of opensculptboy.posing.point_place.place_points.
// Each of Anny's points on the card's posed body goes to the scan along the skin's normal there (the nearest crossing
// in front or behind, within REACH); where the line meets no surface, the closest point of the scan takes its place.
// Each placed point gets a confidence, the product of three marks: the distance it moved, the scan's normal against
// the skin's, and the fused carving (another part of the figure about as near to the spot as the point's own part).
// A point is on when its confidence reaches ON; the page shows the others in amber. test_pose_edit_place checks it
// against the Python placement.

import { TriangleBVH } from './bvh.ts';

export const REACH = 0.03;
export const SEARCH = 0.05;
export const DISTANCE_SCALE = 0.012;
export const NORMAL_RANGE: [number, number] = [0.0, 0.7];
export const FUSED_MARGIN = 0.004;
export const ON = 0.5;

// the parts of the figure, by the bones' words (without the side's suffix)
export const PARTS: [string, string[]][] = [
  ['palm', ['wrist', 'metacarpal']], ['thumb', ['finger1-']], ['index', ['finger2-']], ['middle', ['finger3-']], ['ring', ['finger4-']],
  ['pinky', ['finger5-']], ['forearm', ['lowerarm']], ['upperarm', ['upperarm', 'shoulder01']], ['thigh', ['upperleg']], ['shin', ['lowerleg']],
  ['foot', ['foot', 'toe']], ['head', ['neck', 'head', 'eye']], ['torso', ['root', 'spine', 'pelvis', 'clavicle']],
];
// a point's own parts: those of its joint, and the part each joint meets
export const MEETS: Record<string, string[]> = {
  palm: ['forearm'], thumb: ['palm'], index: ['palm'], middle: ['palm'], ring: ['palm'], pinky: ['palm'],
  forearm: ['upperarm', 'palm'], upperarm: ['torso', 'forearm'], thigh: ['torso', 'shin'], shin: ['thigh', 'foot'], foot: ['shin'], head: ['torso'],
};

// the part of a bone, with its side's suffix ("index.R")
export function partOf(label: string): string {
  let name = label, side = '';
  for (const s of ['.L', '.R']) if (label.endsWith(s)) { name = label.slice(0, -s.length); side = s; }
  for (const [part, words] of PARTS) if (words.some((w) => name.startsWith(w))) return part + (part !== 'head' && part !== 'torso' ? side : '');
  return 'torso';
}

export interface PlaceInput { name: string; joint: string; position: number[]; normal: number[] }
// the posed body: its vertices (n * 3) and each vertex's part (partOf of its strongest bone)
export interface PlaceBody { vertices: ArrayLike<number>; parts: string[] }
export interface Placed {
  name: string; position: number[]; confidence: number; on: boolean; method: 'normal' | 'closest' | 'none'; distance_mm: number | null;
  marks: { distance: number; normals: number; fused: number };
  tri: number;     // the scan's triangle under the point, or -1
}

const round = (x: number, d: number) => Math.round(x * 10 ** d) / 10 ** d;

// the nearest vertex of each part to a point, by part (brute force: a body has some 14 000 vertices)
export class PartIndex {
  private byPart = new Map<string, Float64Array>();
  constructor(body: PlaceBody) {
    const lists = new Map<string, number[]>(), V = body.vertices;
    body.parts.forEach((p, i) => { if (!lists.has(p)) lists.set(p, []); lists.get(p)!.push(V[i * 3], V[i * 3 + 1], V[i * 3 + 2]); });
    for (const [p, l] of lists) this.byPart.set(p, Float64Array.from(l));
  }
  parts(): string[] { return [...this.byPart.keys()]; }
  nearest(part: string, x: ArrayLike<number>): number {
    const P = this.byPart.get(part);
    if (!P) return Infinity;
    let best = Infinity;
    for (let i = 0; i < P.length; i += 3) {
      const dx = P[i] - x[0], dy = P[i + 1] - x[1], dz = P[i + 2] - x[2], d = dx * dx + dy * dy + dz * dz;
      if (d < best) best = d;
    }
    return Math.sqrt(best);
  }
}

// the confidence marks of a point placed at `position` with the scan's normal there (point_place.place_points)
export function marks(p: PlaceInput, position: number[], scanNormal: number[], method: 'normal' | 'closest', parts: PartIndex) {
  const n = p.normal, d = Math.hypot(position[0] - p.position[0], position[1] - p.position[1], position[2] - p.position[2]);
  const markD = Math.exp(-d / DISTANCE_SCALE);
  const c = scanNormal[0] * n[0] + scanNormal[1] * n[1] + scanNormal[2] * n[2];
  const cos = method === 'normal' ? Math.abs(c) : c;
  const [lo, hi] = NORMAL_RANGE;
  const markN = Math.min(Math.max((cos - lo) / (hi - lo), 0), 1);
  const base = partOf(p.joint), side = base.includes('.') ? base.slice(base.indexOf('.')) : '', word = base.split('.')[0];
  const own = new Set([base, ...(MEETS[word] || []).map((m) => m + (m !== 'head' && m !== 'torso' ? side : ''))]);
  let dOwn = Infinity, dOther = Infinity;
  for (const name of parts.parts()) {
    const dd = parts.nearest(name, position);
    if (own.has(name)) dOwn = Math.min(dOwn, dd); else dOther = Math.min(dOther, dd);
  }
  const markF = Math.min(Math.max((dOther - dOwn) / FUSED_MARGIN, 0), 1);
  return { d, markD, markN, markF };
}

// place each point on the scan (vertices and triangles in the same frame as the points)
export function placePoints(points: PlaceInput[], body: PlaceBody, scan: TriangleBVH, opts: { reach?: number; on?: number } = {}): Placed[] {
  const reach = opts.reach ?? REACH, onAt = opts.on ?? ON, parts = new PartIndex(body);
  const radius = SEARCH + scan.maxEdge;
  return points.map((p) => {
    const x = p.position, n = p.normal;
    let position: number[] | null = null, method: 'normal' | 'closest' = 'normal', scanNormal: number[] = [0, 0, 1], tri = -1;
    const hits: [number, number, number][] = [];
    for (const sign of [1, -1]) {
      const h = scan.ray(x, [sign * n[0], sign * n[1], sign * n[2]], reach);
      if (h) hits.push([h.t, sign, h.tri]);
    }
    if (hits.length) {
      hits.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      const [t, sign, row] = hits[0];
      position = [x[0] + sign * t * n[0], x[1] + sign * t * n[1], x[2] + sign * t * n[2]];
      scanNormal = scan.normal(row); tri = row;
    } else {
      const c = scan.closest(x, radius);
      if (c) { position = c.point; scanNormal = scan.normal(c.tri); method = 'closest'; tri = c.tri; }
    }
    if (!position) {
      return { name: p.name, position: x.slice(), confidence: 0, on: false, method: 'none', distance_mm: null, marks: { distance: 0, normals: 0, fused: 0 }, tri: -1 };
    }
    const m = marks(p, position, scanNormal, method, parts);
    const confidence = m.markD * m.markN * m.markF;
    return {
      name: p.name, position, confidence: round(confidence, 4), on: confidence >= onAt, method, distance_mm: round(1000 * m.d, 2),
      marks: { distance: round(m.markD, 4), normals: round(m.markN, 4), fused: round(m.markF, 4) }, tri,
    };
  });
}

// why a point needs a look, in a few words for the panel ("14 mm from the scan", "On fused carving")
export function reason(p: Placed): string {
  if (p.method === 'none') return 'No scan near it';
  const m = p.marks;
  const worst = Math.min(m.distance, m.normals, m.fused);
  if (worst === m.fused) return 'On fused carving';
  if (worst === m.normals) return 'On a face turned away';
  return `${Math.round(p.distance_mm ?? 0)} mm from the scan`;
}

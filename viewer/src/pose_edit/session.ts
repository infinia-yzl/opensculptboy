// OpenSculptBoy
// Apache License, Version 2.0
//
// The pose editor's state without a renderer: the character's pose as rotations in the space of the turns (Anny's
// axes, opensculptboy.posing.pose_rig), the reference points placed on the scan (Anny's axes of the card frame, the
// frame of `detail.fit.reference.points`), the contact pins, and the solves that turn a chain so that Anny's points
// meet their places. A drag solves a few warm steps a frame from the last pose, as the engine's drag test does, and a
// full solve when it ends. Every change is one step of the editor's history (stored rotations and places).

import { buildProblem, chainBones, pointAt, topologicalOrder, transforms, type Influence, type PinInput, type PointInput, type RigSpace, type SkinSource,
  type V3, influenceOf } from './problem.ts';
import { PointSolver } from './solve.ts';
import { POINTS, POINT_BY_NAME, pointsOfChain, type RefPoint } from './points.ts';

export interface Target {
  position: V3;              // card frame
  on: boolean;
  weight: number;
  confidence: number;
  placed: 'auto' | 'owner';
  depth_mm: number;          // how far the point lies inside the stone along `ray`
  surface?: V3;              // the spot on the surface it was pushed in from
  ray?: V3;                  // the direction it was pushed along (the camera's ray, card frame)
  auto?: V3;                 // where the automatic placement had it
  reason?: string;           // why it needs a look, when its confidence is low
}

export interface Pin {
  name: string;              // "Left palm on the left hip"
  skin: { vertices: number[]; barycentric: number[] };
  kind: 'plane' | 'point';
  point?: V3; normal?: V3; target?: V3;
  weight: number;
}

export interface Snapshot { rotations: number[][]; targets: [string, Target][]; pins: Pin[] }

export const DRAG_STEPS = 8;
export const FULL_STEPS = 40;
export const PIN_WEIGHT = 10;

export class EditSession {
  rig: RigSpace;
  rotations: number[][];
  targets = new Map<string, Target>();
  pins: Pin[] = [];
  limits = true;
  private skin: SkinSource;
  private influences = new Map<string, Influence>();
  private order: number[];
  private drag: { chain: string; solver: PointSolver; index: number; bones: number[] } | null = null;
  readonly undoStack: { label: string; snap: Snapshot }[] = [];
  readonly redoStack: { label: string; snap: Snapshot }[] = [];
  onChange: () => void = () => {};

  constructor(rig: RigSpace, skin: SkinSource, rotations: number[][]) {
    this.rig = rig; this.skin = skin; this.rotations = rotations.map((r) => r.slice());
    this.order = topologicalOrder(rig.parents);
  }

  // a new body (its rig and skin) keeps the places and the pose
  setRig(rig: RigSpace, skin: SkinSource) { this.rig = rig; this.skin = skin; this.influences.clear(); this.drag = null; }
  setPose(rotations: number[][]) { this.rotations = rotations.map((r) => r.slice()); }

  influence(name: string): Influence {
    let f = this.influences.get(name);
    if (!f) {
      const p = POINT_BY_NAME.get(name)!;
      f = influenceOf(p.vertices, p.barycentric, this.skin);
      this.influences.set(name, f);
    }
    return f;
  }
  skinInfluence(vertices: number[], barycentric: number[]): Influence { return influenceOf(vertices, barycentric, this.skin); }

  // Anny's points on the present pose, card frame
  positions(points: RefPoint[] = POINTS): Map<string, V3> {
    const K = transforms(this.rig, this.rotations, this.order), out = new Map<string, V3>();
    for (const p of points) out.set(p.name, pointAt(this.rig, K, this.influence(p.name)));
    return out;
  }
  pinPositions(): V3[] {
    const K = transforms(this.rig, this.rotations, this.order);
    return this.pins.map((q) => pointAt(this.rig, K, this.skinInfluence(q.skin.vertices, q.skin.barycentric)));
  }

  // the problem of a chain: its points with places (the others off), and the pins whose skin the chain moves
  problem(chain: string) {
    const points: PointInput[] = [], names: string[] = [];
    for (const p of pointsOfChain(chain)) {
      const t = this.targets.get(p.name);
      points.push({ influence: this.influence(p.name), target: t ? t.position : [0, 0, 0], weight: t ? t.weight : 0, on: !!t && t.on, finger: p.finger });
      names.push(p.name);
    }
    const bones = new Set(chainBones(this.rig, [chain]));
    const pins: PinInput[] = [];
    for (const q of this.pins) {
      const f = this.skinInfluence(q.skin.vertices, q.skin.barycentric);
      if (!f.bones.some((b) => this.moves(b, bones))) continue;
      pins.push({ influence: f, kind: q.kind, point: q.point, normal: q.normal, target: q.target, weight: q.weight });
    }
    return { problem: buildProblem(this.rig, this.rotations, points, [chain], { limits: this.limits, pins }), names, bones: [...bones] };
  }

  // whether a bone lies in or below one of the turning bones
  private moves(b: number, turning: Set<number>): boolean {
    for (let p = b; p >= 0; p = this.rig.parents[p]) if (turning.has(p)) return true;
    return false;
  }

  private take(rotations: Float64Array, bones: number[]) {
    for (const b of bones) this.rotations[b] = Array.from(rotations.subarray(b * 9, b * 9 + 9));
  }

  // a full solve of a chain from the present pose; whether any of its points has a place
  solve(chain: string, iterations = FULL_STEPS): boolean {
    const { problem, bones } = this.problem(chain);
    if (!problem.points.some((p) => p.weight > 0) && !problem.pins.length) return false;
    const r = new PointSolver(problem).solve(iterations);
    this.take(r.rotations, bones);
    this.onChange();
    return true;
  }

  // the chains whose points have places
  placedChains(): string[] {
    const out = new Set<string>();
    for (const [name, t] of this.targets) if (t.on) out.add(POINT_BY_NAME.get(name)!.solve);
    return [...out];
  }

  solveAll(iterations = FULL_STEPS) { for (const c of this.placedChains()) this.solve(c, iterations); }

  // ---------------------------------------------------------------- a drag of one point
  beginDrag(name: string) {
    const p = POINT_BY_NAME.get(name)!;
    if (!this.targets.has(name)) {
      const at = this.positions([p]).get(name)!;
      this.targets.set(name, { position: at, on: true, weight: 1, confidence: 1, placed: 'owner', depth_mm: 0 });
    }
    const { problem, names, bones } = this.problem(p.solve);
    const solver = new PointSolver(problem);
    this.drag = { chain: p.solve, solver, index: names.indexOf(name), bones };
  }

  // the point's new place (card frame): a few warm steps of the solve
  dragTo(name: string, position: V3, extra: Partial<Target> = {}) {
    const t = this.targets.get(name)!;
    // a point the owner moves is on
    Object.assign(t, extra, { position, placed: 'owner' as const, on: true });
    if (!this.drag) return;
    const d = this.drag;
    d.solver.setTarget(d.index, position);
    d.solver.setWeight(d.index, t.weight);
    const r = d.solver.solve(DRAG_STEPS);
    this.take(r.rotations, d.bones);
    this.onChange();
  }

  endDrag(full = true) {
    const d = this.drag;
    this.drag = null;
    if (d && full) this.solve(d.chain);
  }
  dragging(): boolean { return !!this.drag; }

  // ---------------------------------------------------------------- places
  setTarget(name: string, t: Target, solve = true) {
    this.targets.set(name, t);
    if (solve) this.solve(POINT_BY_NAME.get(name)!.solve);
  }
  setOn(name: string, on: boolean) {
    const t = this.targets.get(name);
    if (!t) return;
    t.on = on;
    this.solve(POINT_BY_NAME.get(name)!.solve);
  }
  setWeight(name: string, weight: number) {
    const t = this.targets.get(name);
    if (!t) return;
    t.weight = weight;
    this.solve(POINT_BY_NAME.get(name)!.solve);
  }

  // ---------------------------------------------------------------- history
  snapshot(): Snapshot {
    return { rotations: this.rotations.map((r) => r.slice()), targets: [...this.targets].map(([k, v]) => [k, JSON.parse(JSON.stringify(v))]),
      pins: JSON.parse(JSON.stringify(this.pins)) };
  }
  restore(s: Snapshot) {
    this.rotations = s.rotations.map((r) => r.slice());
    this.targets = new Map(s.targets.map(([k, v]) => [k, JSON.parse(JSON.stringify(v))]));
    this.pins = JSON.parse(JSON.stringify(s.pins));
    this.drag = null;
    this.onChange();
  }
  // one step of the history: call before the change
  record(label: string) {
    this.undoStack.push({ label, snap: this.snapshot() });
    if (this.undoStack.length > 200) this.undoStack.shift();
    this.redoStack.length = 0;
  }
  undo(): string | null {
    const e = this.undoStack.pop();
    if (!e) return null;
    this.redoStack.push({ label: e.label, snap: this.snapshot() });
    this.restore(e.snap);
    return e.label;
  }
  redo(): string | null {
    const e = this.redoStack.pop();
    if (!e) return null;
    this.undoStack.push({ label: e.label, snap: this.snapshot() });
    this.restore(e.snap);
    return e.label;
  }

  // ---------------------------------------------------------------- the card's entry
  // detail.fit.reference.points (opensculptboy.card-points/1)
  cardPoints(): any {
    const chains: Record<string, any[]> = {};
    for (const p of POINTS) {
      const t = this.targets.get(p.name);
      if (!t) continue;
      const r = (x: number) => Math.round(x * 1e6) / 1e6;
      (chains[p.chain] ||= []).push({ name: p.name, position: t.position.map(r), on: t.on, weight: t.weight, confidence: Math.round(t.confidence * 1e4) / 1e4,
        placed: t.placed, depth_mm: Math.round(t.depth_mm * 100) / 100 });
    }
    return { format: 'opensculptboy.card-points/1', frame: 'card', chains };
  }
  cardPins(): any[] {
    return this.pins.map((q) => ({ name: q.name, vertices: q.skin.vertices, barycentric: q.skin.barycentric, kind: q.kind, point: q.point, normal: q.normal,
      target: q.target, weight: q.weight }));
  }
  readCard(points: any, pins: any) {
    this.targets.clear();
    for (const list of Object.values<any>(points?.chains || {})) for (const e of list) {
      if (!POINT_BY_NAME.has(e.name) || !Array.isArray(e.position)) continue;
      this.targets.set(e.name, { position: e.position as V3, on: e.on !== false, weight: Number(e.weight ?? 1), confidence: Number(e.confidence ?? 1),
        placed: e.placed === 'auto' ? 'auto' : 'owner', depth_mm: Number(e.depth_mm || 0) });
    }
    this.pins = (Array.isArray(pins) ? pins : []).filter((q) => q && Array.isArray(q.vertices)).map((q) => ({
      name: String(q.name || 'Pin'), skin: { vertices: q.vertices, barycentric: q.barycentric }, kind: q.kind === 'point' ? 'point' : 'plane',
      point: q.point, normal: q.normal, target: q.target, weight: Number(q.weight ?? PIN_WEIGHT),
    }));
  }
}

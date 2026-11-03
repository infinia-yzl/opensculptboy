// OpenSculptBoy
// Apache License, Version 2.0
//
// The pose editor of the studio (DESIGN-0013, phase 1: reference points). It ties the session (session.ts: places,
// solves, history) to the page: the scan in the view (view3d.ts), the points over it (layer.ts), the pointer
// (studio/pointer.ts: picking, dragging on the scan, the wheel's depth, Esc), the keys (studio/actions.ts, key map
// version 2: Tab, Q W E, 1 to 5, G, [ ], A, Alt+G, Alt+R, Page Up and Down, the arrows, Shift+Z, Ctrl+Z), the status
// line, the selection that F frames, and the store that the panel (shell/sections/PoseEditSection.tsx) reads.
//
// A drag casts a ray from the camera through the pointer every frame; of the surfaces it meets the point takes the hit
// nearest to where it was, so it does not jump to a finger in front. Shift moves the pointer at a quarter of its
// speed, Ctrl snaps to the automatic place, the wheel pushes the point into the stone or pulls it out along the ray,
// and Esc or a right-click puts it back. Each drag, push, nudge or placement is one step of the editor's history.

import { createStore, ui } from '../shell/state.ts';
import { defineScope, registerActions, type ActionDef } from '../studio/actions.ts';
import { cam } from '../studio/camera.ts';
import { viewFocused } from '../studio/keymap.ts';
import { registerPicker, registerSurface, type PickHit, type PointerInfo, type DragInfo } from '../studio/pointer.ts';
import { setSelection } from '../studio/selection.ts';
import { settings } from '../studio/settings.ts';
import { PRIORITY, clearStatus, setStatus, type StatusPart } from '../studio/status.ts';
import { TriangleBVH } from './bvh.ts';
import { SkinnedBody, balance, squeeze, type Balance, type Prop } from './contact.ts';
import { Frames } from './frames.ts';
import type { PoseHost } from './host.ts';
import { PointLayer, type DrawPoint, type Frame } from './layer.ts';
import { placePoints, partOf, reason, type PlaceInput } from './place.ts';
import { POINTS, POINT_BY_NAME, handPoints, pointsOfChain, type RefPoint } from './points.ts';
import { handAxes, type V3 } from './problem.ts';
import { ghostOf, readScan, type ScanMesh } from './scan.ts';
import { EditSession, PIN_WEIGHT, type Pin, type Target } from './session.ts';
import { ScanView, type ScanLook } from './view3d.ts';

export type Tool = 'select' | 'move' | 'turn';
export type Section = 'points' | 'joints' | 'hands' | 'limbs' | 'library';
export const SECTIONS: { id: Section; label: string }[] = [
  { id: 'points', label: 'Points' }, { id: 'joints', label: 'Joints' }, { id: 'hands', label: 'Hands' }, { id: 'limbs', label: 'Limbs' }, { id: 'library', label: 'Library' },
];

export interface PointInfo { name: string; number: number; label: string; state: 'placed' | 'look' | 'off' | 'unplaced'; reason?: string; weight: number; depth: number;
  distance: number | null }
export interface EditState {
  on: boolean;
  tool: Tool;
  section: Section;
  scan: { available: boolean; loading: boolean; shown: boolean; look: ScanLook; opacity: number; error: string; credit: string };
  selected: string | null;          // a point's name
  hand: 'L' | 'R' | null;           // the hand whose map shows
  filter: 'all' | 'hands' | 'body';
  counts: { placed: number; look: number; off: number; total: number };
  looks: PointInfo[];               // the points that need a look
  point: PointInfo | null;          // the selected point
  hands: Record<'L' | 'R', PointInfo[]>;
  body: PointInfo[];
  dragging: boolean;
  magnifier: boolean;
  behind: boolean;                   // show the points behind the stone while dragging
  pins: { name: string }[];
  pinDraft: string | null;           // a pin waiting for its surface
  grip: { hand: 'L' | 'R'; prop: string } | null;
  squeeze: number;
  props: string[];
  balance: { shown: boolean; inside: boolean | null; margin: number | null };
  undo: string; redo: string;
  version: number;
}

const init = (): EditState => ({
  on: false, tool: 'select', section: 'points',
  scan: { available: false, loading: false, shown: true, look: 'ghost', opacity: 0.45, error: '', credit: '' },
  selected: null, hand: null, filter: 'all', counts: { placed: 0, look: 0, off: 0, total: POINTS.length }, looks: [], point: null,
  hands: { L: [], R: [] }, body: [], dragging: false, magnifier: true, behind: true, pins: [], pinDraft: null, grip: null, squeeze: 1, props: [],
  balance: { shown: false, inside: null, margin: null }, undo: '', redo: '', version: 0,
});

export const editStore = createStore<EditState>(init());

const MAG_SIZE = 220, MAG_ZOOM = 4;
const SNAP_PX = 10;

// the parts a pin can hold: a palm or a sole
export const PIN_PARTS = [
  { id: 'palm.L', label: 'Left palm' }, { id: 'palm.R', label: 'Right palm' }, { id: 'sole.L', label: 'Left sole' }, { id: 'sole.R', label: 'Right sole' },
];

export class PoseEditor {
  host: PoseHost;
  frames: Frames;
  session: EditSession | null = null;
  view: ScanView;
  layer: PointLayer | null = null;
  scan: ScanMesh | null = null;
  bvh: TriangleBVH | null = null;
  private scanCard: any = null;
  private offs: (() => void)[] = [];
  private dirty = true;
  private hovered: string | null = null;
  private drag: { name: string; vx: number; vy: number; before: any; last: V3; depth: number; ray: V3 } | null = null;
  private mag: { x: number; y: number } | null = null;
  private visible = new Map<string, boolean>();
  private screen = new Map<string, { x: number; y: number; depth: number }>();
  private body: SkinnedBody | null = null;
  private bodyTris: { bvh: TriangleBVH; rows: Uint32Array } | null = null;
  private bal: Balance | null = null;
  private grab: { name: string; before: any; x0: number; y0: number; off: (() => void) } | null = null;
  private propBvh: TriangleBVH | null = null;
  private surfaceOff: (() => void) | null = null;

  constructor(host: PoseHost) {
    this.host = host;
    this.frames = new Frames(host);
    this.view = new ScanView(host);
    this.registerKeys();
  }

  // ---------------------------------------------------------------- the store
  private publish(patch: Partial<EditState> = {}) {
    const s = this.session;
    const info = (p: RefPoint): PointInfo => {
      const t = s?.targets.get(p.name);
      const state: PointInfo['state'] = !t ? 'unplaced' : !t.on ? 'off' : t.confidence < 0.5 && t.placed === 'auto' ? 'look' : 'placed';
      return { name: p.name, number: p.number, label: p.label, state, reason: t?.reason, weight: t ? t.weight : 1, depth: t ? t.depth_mm : 0,
        distance: null };
    };
    const all = POINTS.map(info);
    const counts = { placed: all.filter((p) => p.state === 'placed').length, look: all.filter((p) => p.state === 'look').length,
      off: all.filter((p) => p.state === 'off').length, total: all.length };
    const st = editStore.get();
    const sel = patch.selected !== undefined ? patch.selected : st.selected;
    let point = sel ? all.find((p) => p.name === sel) || null : null;
    if (point && s) {
      const t = s.targets.get(point.name);
      if (t) { const a = s.positions([POINT_BY_NAME.get(point.name)!]).get(point.name)!; point = { ...point, distance: 1000 * Math.hypot(a[0] - t.position[0], a[1] - t.position[1], a[2] - t.position[2]) }; }
    }
    editStore.set({
      counts, looks: all.filter((p) => p.state === 'look'), point,
      hands: { R: handPoints('R').map(info), L: handPoints('L').map(info) }, body: POINTS.filter((p) => p.group === 'body').map(info),
      pins: (s?.pins || []).map((q) => ({ name: q.name })), undo: s?.undoStack.at(-1)?.label || '', redo: s?.redoStack.at(-1)?.label || '',
      version: st.version + 1, ...patch,
    });
    this.dirty = true;
  }

  // ---------------------------------------------------------------- Edit on and off
  async setOn(on: boolean) {
    if (on === editStore.get().on) return;
    if (!on) { this.leave(); return; }
    if (!this.host.ready()) { this.host.toast('The character is still loading.'); return; }
    const R = this.frames.cardRotations();
    const rig = this.frames.rig(R);
    if (!this.session) {
      this.session = new EditSession(rig, this.frames.skin(), R);
      this.session.onChange = () => this.poseChanged();
    } else { this.session.setRig(rig, this.frames.skin()); this.session.setPose(R); }
    this.body = null; this.bodyTris = null;
    const card = this.host.card();
    if (card && card.card !== this.scanCard) {
      // a new card: its saved points and pins
      const ref = card.card?.detail?.fit?.reference;
      this.session.readCard(ref?.points, ref?.pins);
      this.session.undoStack.length = 0; this.session.redoStack.length = 0;
    }
    if (!this.layer) this.layer = new PointLayer((this.host.canvas.parentElement as HTMLElement) || document.body);
    this.host.setOverlay((t) => this.drawOverlay(t));
    this.host.onFrame(() => this.frame());
    this.offs.push(cam().onChange(() => { this.dirty = true; }));
    document.body.classList.add('pose-edit-on');
    this.publish({ on: true, props: this.host.props().map((p) => p.name) });
    this.statusIdle();
    this.loadScan();
  }

  private leave() {
    this.cancelGrab();
    for (const off of this.offs.splice(0)) off();
    this.host.setOverlay(null); this.host.onFrame(null);
    this.layer?.hide();
    document.body.classList.remove('pose-edit-on');
    clearStatus('pose-edit');
    setSelection(null);
    this.publish({ on: false, selected: null, dragging: false });
    this.host.redraw();
  }

  // the card's light scan (detail.fit.reference.ghost), read once per card
  private async loadScan() {
    const card = this.host.card();
    if (!card) { this.publish({ scan: { ...editStore.get().scan, available: false, loading: false, error: '' } }); return; }
    if (card.card === this.scanCard) return;
    this.scanCard = card.card;
    const g = ghostOf(card.card);
    if (!g) { this.setScan(null); this.publish({ scan: { ...editStore.get().scan, available: false, loading: false, error: '', credit: '' } }); return; }
    this.publish({ scan: { ...editStore.get().scan, loading: true, error: '' } });
    try {
      const bytes = await card.bytes(g.file);
      const scan = readScan(bytes);
      this.setScan(scan);
      this.publish({ scan: { ...editStore.get().scan, available: true, loading: false, error: '', credit: g.credit || '' } });
    } catch (e: any) {
      this.setScan(null);
      this.publish({ scan: { ...editStore.get().scan, available: false, loading: false, error: e?.message || String(e) } });
    }
    this.host.redraw();
  }

  // a scan for the view and the placement (also from a test: window.__poseEdit.setScan)
  setScan(scan: ScanMesh | null) {
    this.scan = scan;
    this.bvh = scan ? new TriangleBVH(scan.positions, scan.indices) : null;
    this.view.setScan(scan);
    if (scan && !this.surfaceOff) {
      this.surfaceOff = registerSurface('scan', (ray) => {
        if (!editStore.get().on || !this.bvh || !this.view.visible()) return null;
        const o = this.frames.fromWorld([ray.origin.x, ray.origin.y, ray.origin.z]), d = this.frames.dirToCard([ray.direction.x, ray.direction.y, ray.direction.z]);
        const h = this.bvh.ray(o, d);
        if (!h) return null;
        const w = this.frames.toWorld([o[0] + d[0] * h.t, o[1] + d[1] * h.t, o[2] + d[2] * h.t]);
        const T = this.host.THREE, p = new T.Vector3(w[0], w[1], w[2]);
        return { point: p, distance: p.distanceTo(ray.origin) };
      });
    }
    this.publish({ scan: { ...editStore.get().scan, available: !!scan } });
  }

  setScanLook(patch: Partial<{ shown: boolean; look: ScanLook; opacity: number }>) {
    if (patch.shown !== undefined) this.view.shown = patch.shown;
    if (patch.look) this.view.look = patch.look;
    if (patch.opacity !== undefined) this.view.opacity = patch.opacity;
    this.publish({ scan: { ...editStore.get().scan, ...patch } });
    this.host.redraw();
  }

  setTool(tool: Tool) { this.publish({ tool }); this.statusIdle(); }
  setSection(section: Section) { this.publish({ section }); }

  // ---------------------------------------------------------------- the pose
  private poseChanged() {
    if (!this.session) return;
    this.host.setPose(this.frames.pageQuaternions(this.session.rotations));
    this.bal = null;
    this.dirty = true;
  }

  private record(label: string) { this.session?.record(label); }

  undo(): boolean {
    const l = this.session?.undo();
    if (!l) return false;
    this.host.toast('Undo: ' + l);
    this.publish();
    return true;
  }
  redo(): boolean {
    const l = this.session?.redo();
    if (!l) return false;
    this.host.toast('Redo: ' + l);
    this.publish();
    return true;
  }

  // ---------------------------------------------------------------- the automatic placement
  // the posed coarse body in the card frame: its skin, faces and the parts of its vertices
  private skinnedBody(): SkinnedBody {
    if (this.body) return this.body;
    const b = this.host.body(), rows = this.host.annyRows(), n = b.nBody, S = this.session!;
    const rest = new Float64Array(n * 3), idx = new Int32Array(n * 8), w = new Float64Array(n * 8);
    for (let c = 0; c < n; c++) {
      const v = this.frames.toCard([b.coarse[c * 3], b.coarse[c * 3 + 1], b.coarse[c * 3 + 2]]);
      rest.set(v, c * 3);
      for (let k = 0; k < 8; k++) { idx[c * 8 + k] = b.skin[c * 16 + k]; w[c * 8 + k] = b.skin[c * 16 + 8 + k] / 255; }
    }
    const tris: number[] = [];
    for (let i = 0; i + 3 < b.quads.length; i += 4) {
      const q = b.quads;
      if (q[i] >= n || q[i + 1] >= n || q[i + 2] >= n || q[i + 3] >= n) continue;
      tris.push(q[i], q[i + 1], q[i + 2], q[i], q[i + 2], q[i + 3]);
    }
    void rows;
    this.body = new SkinnedBody(S.rig, rest, idx, w, 8, Uint32Array.from(tris));
    return this.body;
  }

  // the points of Anny on the present pose with the skin's normals there, and the posed body with its parts
  private placeInputs(points: RefPoint[]): { inputs: PlaceInput[]; body: { vertices: Float64Array; parts: string[] } } {
    const S = this.session!, body = this.skinnedBody(), V = body.vertices(S.rotations), F = body.faces, N = new Float64Array(V.length);
    for (let t = 0; t < F.length; t += 3) {
      const a = F[t] * 3, b = F[t + 1] * 3, c = F[t + 2] * 3;
      const e1 = [V[b] - V[a], V[b + 1] - V[a + 1], V[b + 2] - V[a + 2]], e2 = [V[c] - V[a], V[c + 1] - V[a + 1], V[c + 2] - V[a + 2]];
      const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      for (const v of [a, b, c]) for (let k = 0; k < 3; k++) N[v + k] += n[k];
    }
    const rows = this.host.annyRows(), index = new Map<number, number>();
    if (rows) for (let c = 0; c < rows.length; c++) index.set(rows[c], c);
    const pos = S.positions(points), labels = S.rig.labels;
    const inputs = points.map((p) => {
      const n = [0, 0, 0];
      p.vertices.forEach((row, k) => { const c = rows ? index.get(row)! : row; for (let j = 0; j < 3; j++) n[j] += p.barycentric[k] * N[c * 3 + j]; });
      const l = Math.hypot(n[0], n[1], n[2]) || 1;
      return { name: p.name, joint: p.joint, position: pos.get(p.name)!, normal: n.map((x) => x / l) };
    });
    return { inputs, body: { vertices: V, parts: Array.from(body.strongest, (b) => partOf(labels[b])) } };
  }

  // Place points: every point (or the given ones) on the scan, then the solve of each chain
  placeAll(names?: string[]): { placed: number; look: number } {
    const S = this.session;
    if (!S || !this.bvh) { this.host.toast('Show a scan first: Place points needs a statue starter with its scan.'); return { placed: 0, look: 0 }; }
    this.record(names ? 'Place again' : 'Place points');
    const pts = names ? names.map((n) => POINT_BY_NAME.get(n)!) : POINTS;
    const { inputs, body } = this.placeInputs(pts);
    const placed = placePoints(inputs, body, this.bvh);
    let look = 0;
    for (const e of placed) {
      if (e.method === 'none') continue;
      const t: Target = { position: e.position as V3, on: e.on || e.confidence > 0.15, weight: 1, confidence: e.confidence, placed: 'auto', depth_mm: 0,
        auto: e.position as V3, reason: e.on ? undefined : reason(e) };
      if (!e.on) look++;
      S.targets.set(e.name, t);
    }
    S.solveAll();
    this.publish();
    const n = placed.filter((e) => e.method !== 'none').length;
    this.host.toast(look ? `Placed ${n} points. ${look} need a look.` : `Placed ${n} points.`);
    return { placed: n, look };
  }

  // the rest pose: every bone below the root at its reference turn
  restPose() {
    const S = this.session;
    if (!S) return;
    this.record('Rest pose');
    const root = this.host.root();
    S.setPose(S.rotations.map((r, i) => (i === root ? r : [1, 0, 0, 0, 1, 0, 0, 0, 1])));
    this.poseChanged();
    this.publish();
  }

  // the whole pose solved again from the places
  solveAll() { this.session?.solveAll(); this.publish(); }

  // ---------------------------------------------------------------- selection
  select(name: string | null, opts: { frame?: boolean } = {}) {
    const p = name ? POINT_BY_NAME.get(name) : null;
    const hand = p && p.group !== 'body' ? (p.side as 'L' | 'R') : editStore.get().hand;
    this.publish({ selected: name, hand: name ? hand : editStore.get().hand });
    if (!p) { setSelection(null); this.statusIdle(); return; }
    setSelection({
      label: `Point ${p.number}`,
      bones: [p.joint],
      frame: (step) => this.framing(p, step),
    });
    this.statusSelected();
    if (opts.frame) { const f = this.framing(p, 1); if (f) cam().frameBounds(f.box, { label: f.label, pivot: f.pivot }); }
    else {
      // turn round the selection (DESIGN-0014): the pivot moves to the point, the camera stays
      const t = this.target(p.name), at = t ? t.position : this.session?.positions([p]).get(p.name);
      if (at) { const w = this.frames.toWorld(at); cam().setPivot({ x: w[0], y: w[1], z: w[2] }, { history: false }); }
    }
  }

  selectHand(side: 'L' | 'R' | null, frame = true) {
    this.publish({ hand: side, selected: null });
    if (!side) { setSelection(null); this.statusIdle(); return; }
    const wrist = POINT_BY_NAME.get('wrist.' + side)!;
    setSelection({ label: side === 'R' ? 'Right hand' : 'Left hand', frame: (step) => this.framing(wrist, step + 1) });
    if (frame) { const f = this.framing(wrist, 1); if (f) cam().frameBounds(f.box, { label: f.label }); }
    this.statusIdle();
  }

  // F's steps from a point: the point, its hand, its arm (or leg), then the whole figure (null)
  private framing(p: RefPoint, step: number) {
    const S = this.session;
    if (!S) return null;
    const anny = S.positions();
    const world = (names: string[]) => names.map((n) => { const t = S.targets.get(n); const w = this.frames.toWorld(t ? t.position : anny.get(n)!); return { x: w[0], y: w[1], z: w[2] }; });
    if (step === 0) {
      const c = world([p.name])[0], r = 0.03;
      return { box: { min: { x: c.x - r, y: c.y - r, z: c.z - r }, max: { x: c.x + r, y: c.y + r, z: c.z + r } }, label: `Point ${p.number}`, pivot: c };
    }
    if (step === 1 && p.group !== 'body') return { box: world(handPoints(p.side as 'L' | 'R').map((q) => q.name)), label: p.side === 'R' ? 'Right hand' : 'Left hand' };
    const limb = step <= 2 ? pointsOfChain(p.solve) : null;
    if (limb && limb.length) return { box: world(limb.map((q) => q.name)), label: p.solve.startsWith('arm') ? (p.side === 'R' ? 'Right arm' : 'Left arm') : p.solve.startsWith('leg') ? (p.side === 'R' ? 'Right leg' : 'Left leg') : 'Head' };
    return null;
  }

  // a point lit from the maps
  hover(name: string | null) { this.hovered = name; this.dirty = true; }

  // [ and ]: the point of the parent or the child joint; Shift: the previous or next finger
  step(dir: -1 | 1, finger = false): boolean {
    const name = editStore.get().selected;
    if (!name) return false;
    const p = POINT_BY_NAME.get(name)!;
    let next: RefPoint | undefined;
    if (p.group !== 'body') {
      const hand = handPoints(p.side as 'L' | 'R'), m = p.mediapipe ?? 0;
      if (finger) {
        if (m === 0) return false;
        const f = Math.floor((m - 1) / 4), k = (m - 1) % 4, nf = Math.min(4, Math.max(0, f + dir));
        next = hand[1 + nf * 4 + k];
      } else if (dir > 0) next = m === 0 ? hand[1] : (m - 1) % 4 < 3 ? hand[m + 1] : undefined;
      else next = m === 0 ? POINT_BY_NAME.get('elbow.' + p.side) : (m - 1) % 4 > 0 ? hand[m - 1] : hand[0];
    } else {
      const chain: Record<string, string[]> = { arm: ['shoulder', 'elbow', 'wrist'], leg: ['hip', 'knee', 'ankle', 'heel', 'toe'] };
      const base = name.replace(/\.[LR]$/, ''), list = Object.values(chain).find((c) => c.includes(base));
      if (!list) return false;
      const i = list.indexOf(base) + dir;
      if (i >= 0 && i < list.length) next = POINT_BY_NAME.get(list[i] + '.' + p.side);
    }
    if (!next) return false;
    this.select(next.name);
    return true;
  }

  // ---------------------------------------------------------------- changes of one point
  private target(name: string): Target | undefined { return this.session?.targets.get(name); }

  usePoint(name: string, on: boolean) { if (!this.session?.targets.has(name)) return; this.record(on ? 'Use the point' : 'Switch the point off'); this.session.setOn(name, on); this.publish(); }
  setWeight(name: string, w: number, record = true) { if (!this.session?.targets.has(name)) return; if (record) this.record('Weight'); this.session.setWeight(name, w); this.publish(); }

  // push the point into the stone (mm > 0) or pull it out, along the ray it was pushed on
  push(name: string, mm: number, record = true): boolean {
    const S = this.session, t = this.target(name);
    if (!S || !t) return false;
    if (record) this.record(mm > 0 ? 'Push in' : 'Pull out');
    if (!t.surface || !t.ray) {
      const ray = this.viewRayTo(t.position);
      t.surface = t.position.slice() as V3; t.ray = ray;
    }
    t.depth_mm = Math.max(0, t.depth_mm + mm);
    const d = t.depth_mm / 1000;
    t.position = [t.surface[0] + t.ray[0] * d, t.surface[1] + t.ray[1] * d, t.surface[2] + t.ray[2] * d];
    t.placed = 'owner';
    if (t.depth_mm === 0) { delete t.surface; delete t.ray; }
    S.setTarget(name, t);
    this.publish();
    if (editStore.get().selected === name) this.statusSelected();
    return true;
  }
  toSurface(name: string): boolean { const t = this.target(name); if (!t || !t.depth_mm) return false; return this.push(name, -t.depth_mm); }

  // Alt+G: back where the automatic placement had it (Shift: the whole hand or limb)
  resetPoint(name: string, whole = false): boolean {
    const S = this.session;
    if (!S) return false;
    const p = POINT_BY_NAME.get(name)!;
    const names = whole ? (p.group === 'body' ? pointsOfChain(p.solve).map((q) => q.name) : handPoints(p.side as 'L' | 'R').map((q) => q.name)) : [name];
    const list = names.filter((n) => S.targets.get(n)?.auto);
    if (!list.length) return false;
    this.record(whole ? 'Put the points back' : 'Put the point back');
    for (const n of list) {
      const t = S.targets.get(n)!;
      t.position = t.auto!.slice() as V3; t.depth_mm = 0; delete t.surface; delete t.ray; t.placed = 'auto';
    }
    S.solve(p.solve);
    this.publish();
    return true;
  }

  // arrows: move the point 1 mm along the surface in the view's directions (Shift: a fifth)
  nudge(name: string, dx: number, dy: number, mm: number): boolean {
    const S = this.session, t = this.target(name);
    if (!S || !t || !this.bvh) return false;
    const c = this.host.camera, T = this.host.THREE;
    const right = new T.Vector3().setFromMatrixColumn(c.matrixWorld, 0), up = new T.Vector3().setFromMatrixColumn(c.matrixWorld, 1);
    const r = this.frames.dirToCard([right.x, right.y, right.z]), u = this.frames.dirToCard([up.x, up.y, up.z]);
    this.record('Nudge');
    const base = t.surface || t.position, k = mm / 1000;
    const moved: V3 = [base[0] + k * (dx * r[0] + dy * u[0]), base[1] + k * (dx * r[1] + dy * u[1]), base[2] + k * (dx * r[2] + dy * u[2])];
    const q = this.bvh.closest(moved);
    const spot = (q ? q.point : moved) as V3;
    if (t.depth_mm > 0 && t.ray) { t.surface = spot; const d = t.depth_mm / 1000; t.position = [spot[0] + t.ray[0] * d, spot[1] + t.ray[1] * d, spot[2] + t.ray[2] * d]; }
    else t.position = spot;
    t.placed = 'owner';
    S.setTarget(name, t);
    this.publish();
    return true;
  }

  // the camera's ray through a card point, in the card frame
  private viewRayTo(x: V3): V3 {
    const w = this.frames.toWorld(x), c = this.host.camera.position;
    return this.frames.dirToCard([w[0] - c.x, w[1] - c.y, w[2] - c.z]);
  }

  // place the selected point by hand: on the scan under a click
  placeAt(name: string, clientX: number, clientY: number): boolean {
    const S = this.session;
    if (!S || !this.bvh) return false;
    const hit = this.scanHit(clientX, clientY, null);
    if (!hit) return false;
    this.record('Place by hand');
    const old = S.targets.get(name);
    S.setTarget(name, { position: hit.point, on: true, weight: old?.weight ?? 1, confidence: 1, placed: 'owner', depth_mm: 0, auto: old?.auto });
    this.publish();
    return true;
  }

  // the scan under a point of the canvas: of the crossings, the one nearest `near` (else the first)
  private scanHit(cx: number, cy: number, near: V3 | null): { point: V3; ray: V3 } | null {
    if (!this.bvh) return null;
    const ray = cam().rayAt(cx, cy);
    const o = this.frames.fromWorld([ray.origin.x, ray.origin.y, ray.origin.z]), d = this.frames.dirToCard([ray.direction.x, ray.direction.y, ray.direction.z]);
    const hits = this.bvh.rayAll(o, d);
    let best: V3 | null = null, bd = Infinity;
    for (const h of hits) {
      const p: V3 = [o[0] + d[0] * h.t, o[1] + d[1] * h.t, o[2] + d[2] * h.t];
      const dist = near ? Math.hypot(p[0] - near[0], p[1] - near[1], p[2] - near[2]) : h.t;
      if (dist < bd) { bd = dist; best = p; }
      if (!near) break;
    }
    if (!best) {
      // the ray meets nothing: the surface spot closest to the ray, near the point's last place
      if (!near) return null;
      const s = (near[0] - o[0]) * d[0] + (near[1] - o[1]) * d[1] + (near[2] - o[2]) * d[2];
      const q = this.bvh.closest([o[0] + d[0] * s, o[1] + d[1] * s, o[2] + d[2] * s]);
      if (!q) return null;
      best = q.point;
    }
    return { point: best, ray: d };
  }

  // ---------------------------------------------------------------- the drag of a point
  beginDrag(name: string, x: number, y: number) {
    const S = this.session!;
    const before = S.snapshot();
    S.beginDrag(name);
    const t = S.targets.get(name)!;
    this.drag = { name, vx: x, vy: y, before, last: t.position.slice() as V3, depth: t.depth_mm, ray: (t.ray || this.viewRayTo(t.position)) as V3 };
    this.mag = { x, y };
    this.publish({ dragging: true, selected: name });
    this.statusDrag();
  }

  dragMove(dx: number, dy: number, snap: boolean) {
    const D = this.drag, S = this.session;
    if (!D || !S) return;
    D.vx += dx; D.vy += dy;
    this.mag = { x: D.vx, y: D.vy };
    const hit = this.scanHit(D.vx, D.vy, D.last);
    if (!hit) return;
    let spot = hit.point;
    const t = S.targets.get(D.name)!;
    if (snap && t.auto) {
      const a = cam().toScreen(this.v3(this.frames.toWorld(t.auto)));
      if (Math.hypot(a.x - D.vx, a.y - D.vy) <= SNAP_PX) spot = t.auto.slice() as V3;
    }
    D.last = spot;
    const dd = D.depth / 1000;
    const position: V3 = D.depth > 0 ? [spot[0] + D.ray[0] * dd, spot[1] + D.ray[1] * dd, spot[2] + D.ray[2] * dd] : spot;
    S.dragTo(D.name, position, D.depth > 0 ? { surface: spot, ray: D.ray, depth_mm: D.depth } : { depth_mm: 0 });
    this.statusDrag();
    this.publish();
  }

  dragDepth(mm: number) {
    const D = this.drag;
    if (!D) return;
    D.depth = Math.max(0, D.depth + mm);
    if (D.depth > 0 && !this.target(D.name)?.ray) D.ray = this.viewRayTo(D.last);
    this.dragMove(0, 0, false);
  }

  endDrag() {
    const D = this.drag, S = this.session;
    if (!D || !S) return;
    this.drag = null; this.mag = null;
    S.endDrag(true);
    // one step of the history: the state before the drag
    S.undoStack.push({ label: `Point ${POINT_BY_NAME.get(D.name)!.number}`, snap: D.before });
    S.redoStack.length = 0;
    clearStatus('pose-edit-drag');
    this.publish({ dragging: false });
    this.statusSelected();
  }

  cancelDrag() {
    const D = this.drag, S = this.session;
    if (!D || !S) return;
    this.drag = null; this.mag = null;
    S.endDrag(false);
    S.restore(D.before);
    clearStatus('pose-edit-drag');
    this.publish({ dragging: false });
    this.statusSelected();
  }

  // G: the selected point follows the mouse until a click or Enter; Esc or a right-click puts it back
  startGrab(): boolean {
    const name = editStore.get().selected;
    if (!name || !this.bvh || this.drag || this.grab) return false;
    const t = this.target(name);
    if (!t) return false;
    const s = cam().toScreen(this.v3(this.frames.toWorld(t.position)));
    this.beginDrag(name, s.x, s.y);
    let lx = s.x, ly = s.y, first = true;
    const move = (e: PointerEvent) => {
      if (first) { lx = e.clientX; ly = e.clientY; first = false; return; }
      const k = settings.get().dragSpeed * (e.shiftKey ? 0.25 : 1);
      this.dragMove((e.clientX - lx) * k, (e.clientY - ly) * k, e.ctrlKey || e.metaKey);
      lx = e.clientX; ly = e.clientY;
    };
    const down = (e: PointerEvent) => {
      e.preventDefault(); e.stopImmediatePropagation();
      if (e.button === 2) this.cancelGrab(); else this.confirmGrab();
    };
    window.addEventListener('pointermove', move, true);
    window.addEventListener('pointerdown', down, true);
    this.grab = { name, before: null, x0: s.x, y0: s.y, off: () => { window.removeEventListener('pointermove', move, true); window.removeEventListener('pointerdown', down, true); } };
    return true;
  }
  confirmGrab(): boolean { if (!this.grab) return false; this.grab.off(); this.grab = null; this.endDrag(); return true; }
  cancelGrab(): boolean { if (!this.grab) return false; this.grab.off(); this.grab = null; this.cancelDrag(); return true; }
  grabbing(): boolean { return !!this.grab; }

  // ---------------------------------------------------------------- contact pins, Squeeze and balance
  // the skin spot of a palm or a sole on the rest body: a ray from the hand's middle along the palm's normal, or from
  // the foot's middle downwards, onto Anny's level-0 skin
  private partSpot(part: string): { vertices: number[]; barycentric: number[] } | null {
    const S = this.session!;
    if (!this.bodyTris) {
      const body = this.skinnedBody(), rows = this.host.annyRows();
      if (!rows) return null;
      this.bodyTris = { bvh: new TriangleBVH(body.rest, body.faces), rows };
    }
    const [kind, side] = part.split('.'), ix = new Map(S.rig.labels.map((l, i) => [l, i] as [string, number])), H = S.rig.heads;
    let o: number[], d: number[];
    if (kind === 'palm') {
      const w = H[ix.get('wrist.' + side)!], m = H[ix.get('finger3-1.' + side)!];
      o = [0, 1, 2].map((k) => 0.5 * (w[k] + m[k]));
      d = handAxes(S.rig, '.' + side)[0];
    } else {
      const f = H[ix.get('foot.' + side)!], t = H[ix.get('toe1-1.' + side)!];
      o = [0, 1, 2].map((k) => 0.5 * (f[k] + t[k]));
      d = [0, 0, -1];
    }
    const h = this.bodyTris.bvh.ray(o, d);
    if (!h) return null;
    const F = this.skinnedBody().faces, rows = this.bodyTris.rows;
    return { vertices: [0, 1, 2].map((k) => rows[F[h.tri * 3 + k]]), barycentric: [1 - h.u - h.v, h.u, h.v] };
  }

  // start a pin of a part; the next click on the scan (or the ground) gives its surface
  beginPin(part: string) { this.publish({ pinDraft: part }); setStatus('pose-edit', [{ text: `Click the surface the ${PIN_PARTS.find((p) => p.id === part)?.label.toLowerCase()} rests on` }, { keys: 'Esc', text: 'cancel' }], { priority: PRIORITY.tool }); }
  cancelPin() { this.publish({ pinDraft: null }); this.statusIdle(); }

  addPin(part: string, where: { kind: 'ground' } | { kind: 'scan'; x: number; y: number }): boolean {
    const S = this.session;
    if (!S) return false;
    const skin = this.partSpot(part);
    if (!skin) { this.host.toast('That part has no skin to pin.'); return false; }
    const label = PIN_PARTS.find((p) => p.id === part)!.label;
    let point: V3, normal: V3, name: string;
    if (where.kind === 'ground') {
      const at = S.positions([]);
      void at;
      const V = this.skinnedBody().vertices(S.rotations);
      let lo = Infinity; for (let i = 2; i < V.length; i += 3) lo = Math.min(lo, V[i]);
      point = [0, 0, lo]; normal = [0, 0, 1]; name = `${label} on the ground`;
    } else {
      const hit = this.scanHit(where.x, where.y, null);
      if (!hit || !this.bvh) return false;
      const q = this.bvh.closest(hit.point);
      const n = q ? this.bvh.normal(q.tri) : [0, 0, 1];
      const toCam = [-hit.ray[0], -hit.ray[1], -hit.ray[2]];
      const s = n[0] * toCam[0] + n[1] * toCam[1] + n[2] * toCam[2] < 0 ? -1 : 1;
      point = hit.point; normal = [n[0] * s, n[1] * s, n[2] * s]; name = `${label} on the scan`;
    }
    this.record('Add a pin');
    S.pins.push({ name, skin, kind: 'plane', point, normal, weight: PIN_WEIGHT });
    for (const c of new Set([...S.placedChains(), part.startsWith('palm') ? 'arm.' + part.slice(-1).toLowerCase() : 'leg.' + part.slice(-1).toLowerCase()])) S.solve(c);
    this.publish({ pinDraft: null });
    this.statusIdle();
    return true;
  }
  removePin(i: number) {
    const S = this.session;
    if (!S || !S.pins[i]) return;
    this.record('Remove a pin');
    S.pins.splice(i, 1);
    S.solveAll();
    this.publish();
  }

  // Grip: the hand and the prop; Squeeze closes its fingers onto the prop
  setGrip(grip: { hand: 'L' | 'R'; prop: string } | null) { this.propBvh = null; this.publish({ grip }); }
  setSqueeze(value: number, record = true): boolean {
    const S = this.session, g = editStore.get().grip;
    if (!S || !g) return false;
    const prop = this.host.props().find((p) => p.name === g.prop);
    if (!prop) { this.host.toast('That prop is not shown.'); return false; }
    if (!this.propBvh) {
      // the prop in the card frame (it stands where it is shown now)
      const P = new Float32Array(prop.positions.length);
      for (let i = 0; i < P.length; i += 3) P.set(this.frames.fromWorld([prop.positions[i], prop.positions[i + 1], prop.positions[i + 2]]), i);
      this.propBvh = new TriangleBVH(P, prop.indices);
    }
    const bvh = this.propBvh;
    const sdf: Prop = (X: Float64Array) => {
      const out = new Float64Array(X.length / 3);
      for (let i = 0; i < out.length; i++) {
        const p = [X[i * 3], X[i * 3 + 1], X[i * 3 + 2]], q = bvh.closest(p);
        if (!q) { out[i] = Infinity; continue; }
        const n = bvh.normal(q.tri), s = (p[0] - q.point[0]) * n[0] + (p[1] - q.point[1]) * n[1] + (p[2] - q.point[2]) * n[2];
        out[i] = s < 0 ? -q.distance : q.distance;
      }
      return out;
    };
    if (record) this.record('Squeeze');
    const r = squeeze(this.skinnedBody(), S.rotations, g.hand === 'R' ? 'hand.r' : 'hand.l', [sdf], value);
    S.setPose(r.rotations);
    this.poseChanged();
    this.publish({ squeeze: value });
    return true;
  }

  setBalance(shown: boolean) { this.bal = null; this.publish({ balance: { ...editStore.get().balance, shown } }); }

  // ---------------------------------------------------------------- the card
  // the points, the pins and the pose into the open card, then Save card
  async savePoints(save: () => Promise<void>) {
    const S = this.session, card = this.host.card();
    if (!S) return;
    const target = card?.source || card?.card;
    if (target) {
      const ref = (((target.detail ||= {}).fit ||= {}).reference ||= {});
      ref.points = S.cardPoints();
      ref.pins = S.cardPins();
    }
    this.host.setPose(this.frames.pageQuaternions(S.rotations), 'Edited pose');
    if (target) target.pose = this.host.cardPose();
    await save();
  }

  // ---------------------------------------------------------------- drawing
  private v3(a: ArrayLike<number>) { return { x: a[0], y: a[1], z: a[2] }; }

  private drawOverlay(target: any) {
    if (!editStore.get().on) return;
    const T = this.host.THREE, M = this.frames.cardToWorld(T);
    this.view.draw(target, M);
    const st = editStore.get();
    if (this.mag && st.magnifier && st.dragging && this.scan) {
      const r = this.host.canvas.getBoundingClientRect();
      // the magnifier sits above and to the right of the pointer, inside the view
      const cx = Math.min(r.right - MAG_SIZE / 2 - 12, this.mag.x + 150), cy = Math.max(r.top + MAG_SIZE / 2 + 70, this.mag.y - 150);
      this.magCam = this.view.magnify(target, M, this.mag.x, this.mag.y, cx, cy, MAG_SIZE, MAG_ZOOM);
      this.magAt = { cx, cy };
    } else this.magCam = null;
  }
  private magCam: any = null;
  private magAt: { cx: number; cy: number } | null = null;

  private frame() {
    if (!editStore.get().on || !this.layer || !this.session) return;
    if (!this.dirty && !this.drag) return;
    this.dirty = false;
    this.drawLayer();
    if (this.drag || this.mag) this.host.redraw();
  }

  // whether a card point lies behind the scan from the camera
  private behindScan(x: V3): boolean {
    if (!this.bvh || !this.view.visible()) return false;
    const c = this.host.camera.position, o = this.frames.fromWorld([c.x, c.y, c.z]);
    const d = [x[0] - o[0], x[1] - o[1], x[2] - o[2]], l = Math.hypot(d[0], d[1], d[2]);
    const h = this.bvh.ray(o, [d[0] / l, d[1] / l, d[2] / l], l - 0.002);
    return !!h;
  }

  drawLayer() {
    const S = this.session!, st = editStore.get(), scale = settings.targetScale();
    const anny = S.positions();
    const f: Frame = { points: [], anny: [], labels: [], clusters: [], pins: [], balance: null, magnifier: null, scale };
    this.screen.clear(); this.visible.clear();
    const toS = (x: V3) => cam().toScreen(this.v3(this.frames.toWorld(x)));
    // the hands drawn small when they are small on screen
    const clustered = new Set<'L' | 'R'>();
    for (const side of ['R', 'L'] as const) {
      if (st.hand === side) continue;
      const xs = handPoints(side).map((p) => toS(S.targets.get(p.name)?.position || anny.get(p.name)!));
      const w = Math.max(...xs.map((a) => a.x)) - Math.min(...xs.map((a) => a.x)), h = Math.max(...xs.map((a) => a.y)) - Math.min(...xs.map((a) => a.y));
      if (Math.max(w, h) < 90 * scale) clustered.add(side);
    }
    for (const p of POINTS) {
      if (st.filter === 'hands' && p.group === 'body') continue;
      if (st.filter === 'body' && p.group !== 'body') continue;
      const t = S.targets.get(p.name);
      const at = t ? t.position : anny.get(p.name)!;
      const s = toS(at);
      if (s.depth <= 0) continue;
      const hidden = this.behindScan(at);
      this.screen.set(p.name, s); this.visible.set(p.name, !hidden);
      const small = p.group !== 'body' && clustered.has(p.side as 'L' | 'R');
      const state: DrawPoint['state'] = !t ? 'unplaced' : !t.on ? 'off' : t.confidence < 0.5 && t.placed === 'auto' ? 'look' : 'placed';
      if (hidden && st.dragging && !st.behind && p.name !== st.selected) continue;
      f.points.push({ name: p.name, number: p.number, x: s.x, y: s.y, state, hidden, selected: p.name === st.selected, hovered: p.name === this.hovered,
        pushed: t ? t.depth_mm : 0, small });
      // Anny's point, while the point is dragged or its hand is shown
      const showAnny = t && (st.dragging ? p.solve === POINT_BY_NAME.get(st.selected || '')?.solve : (st.selected === p.name));
      if (showAnny) { const a = toS(anny.get(p.name)!); f.anny.push({ name: p.name, x: a.x, y: a.y, tx: s.x, ty: s.y, line: true }); }
    }
    for (const side of clustered) {
      const xs = handPoints(side).map((p) => this.screen.get(p.name)).filter(Boolean) as { x: number; y: number }[];
      if (!xs.length) continue;
      const look = st.hands[side].filter((p) => p.state === 'look').length, placed = st.hands[side].filter((p) => p.state !== 'unplaced').length;
      f.clusters.push({ x: Math.max(...xs.map((a) => a.x)), y: xs.reduce((s, a) => s + a.y, 0) / xs.length, title: `${side === 'R' ? 'Right' : 'Left'} hand: 21 points`,
        line: !placed ? 'Not placed yet' : look ? `${look} need${look === 1 ? 's' : ''} a look` : 'All placed' });
    }
    // the label of the selected or hovered point
    const lab = this.hovered || st.selected;
    if (lab && this.screen.has(lab)) {
      const p = POINT_BY_NAME.get(lab)!, s = this.screen.get(lab)!, t = S.targets.get(lab);
      const line = st.dragging ? 'Slides on the scan. The hand follows.' : !t ? 'Not placed yet' : !t.on ? 'Switched off' : t.depth_mm > 0 ? `${t.depth_mm.toFixed(1)} mm into the stone`
        : t.confidence < 0.5 && t.placed === 'auto' ? `Needs a look: ${t.reason || 'unsure'}` : !this.visible.get(lab) ? 'Behind the stone' : 'Placed on the scan';
      f.labels.push({ x: s.x, y: s.y - 34, title: `${p.number} ${p.label}`, line, side: 'left' });
    }
    // the pins and the balance mark
    S.pinPositions().forEach((x, i) => { const s = toS(x); f.pins.push({ x: s.x, y: s.y, label: S.pins[i].name }); });
    if (st.balance.shown) {
      if (!this.bal) {
        this.bal = balance(this.skinnedBody(), S.rotations);
        const b = this.bal;
        queueMicrotask(() => this.publish({ balance: { shown: true, inside: b.inside, margin: b.margin_mm } }));
      }
      const b = this.bal, c = toS(b.centre as V3), m = toS(b.mark as V3);
      f.balance = { centre: [c.x, c.y], mark: [m.x, m.y], polygon: b.polygon.map((q) => { const s = toS(q as V3); return [s.x, s.y] as [number, number]; }), inside: b.inside };
    }
    if (this.magCam && this.magAt && this.drag) {
      const t = S.targets.get(this.drag.name)!, w = this.frames.toWorld(t.position);
      const T = this.host.THREE, v = new T.Vector3(w[0], w[1], w[2]).project(this.magCam);
      const half = MAG_SIZE / 2;
      f.magnifier = { cx: this.magAt.cx, cy: this.magAt.cy, r: half, zoom: MAG_ZOOM,
        point: { x: this.magAt.cx + v.x * half, y: this.magAt.cy - v.y * half, number: POINT_BY_NAME.get(this.drag.name)!.number, state: 'placed' } };
    }
    this.layer!.draw(f);
  }

  // ---------------------------------------------------------------- picking
  private hitPoints(p: PointerInfo): PickHit[] {
    const st = editStore.get();
    if (!st.on || st.pinDraft) return [];
    const out: PickHit[] = [];
    for (const [name, s] of this.screen) {
      const d = Math.hypot(s.x - p.x, s.y - p.y);
      if (d > p.radius + 11 * settings.targetScale()) continue;
      const pt = POINT_BY_NAME.get(name)!;
      out.push({ item: name, distPx: Math.max(0, d - 11 * settings.targetScale()), depth: s.depth, hidden: !this.visible.get(name), label: `${pt.number} ${pt.label}` });
    }
    return out;
  }

  private registerPickers() {
    const pick = (h: PickHit, p: PointerInfo) => {
      const name = h.item as string, pt = POINT_BY_NAME.get(name)!, st = editStore.get();
      // a point of a hand drawn small selects the hand and frames it
      if (pt.group !== 'body' && st.hand !== pt.side && !st.selected?.endsWith('.' + pt.side) && this.isClustered(pt.side as 'L' | 'R')) { this.selectHand(pt.side as 'L' | 'R'); return; }
      this.select(name);
      void p;
    };
    const common = {
      hit: (p: PointerInfo) => this.hitPoints(p),
      onClick: pick,
      onHover: (h: PickHit | null) => { this.hovered = h ? (h.item as string) : null; this.dirty = true; },
      onBackground: (p: PointerInfo) => {
        const st = editStore.get();
        if (st.pinDraft) { this.addPin(st.pinDraft, { kind: 'scan', x: p.x, y: p.y }); return; }
        if (st.selected && st.tool === 'move' && !this.target(st.selected) && this.bvh) { this.placeAt(st.selected, p.x, p.y); return; }
        this.select(null);
      },
      cursor: 'pointer',
    };
    this.offs.push(registerPicker({ id: 'pose-points-select', priority: 10, enabled: () => editStore.get().on && editStore.get().tool !== 'move', ...common }));
    this.offs.push(registerPicker({
      id: 'pose-points-move', priority: 10, enabled: () => editStore.get().on && editStore.get().tool === 'move', ...common,
      onDown: (h: PickHit) => { if (!this.bvh) return false; this.select(h.item as string); return true; },
      onDrag: (h: PickHit, p: PointerInfo, d: DragInfo) => {
        if (!this.drag) this.beginDrag(h.item as string, d.x0, d.y0);
        this.dragMove(d.ddx * d.scale, d.ddy * d.scale, d.snap);
      },
      onUp: () => this.endDrag(),
      onCancel: () => this.cancelDrag(),
      onWheel: (_h: PickHit, p: PointerInfo, notches: number) => { this.dragDepth(notches * (p.shift ? 0.2 : 1)); return true; },
      cursor: 'grab',
    }));
  }
  private isClustered(side: 'L' | 'R'): boolean {
    const xs = handPoints(side).map((p) => this.screen.get(p.name)).filter(Boolean) as { x: number; y: number }[];
    if (!xs.length) return false;
    const w = Math.max(...xs.map((a) => a.x)) - Math.min(...xs.map((a) => a.x)), h = Math.max(...xs.map((a) => a.y)) - Math.min(...xs.map((a) => a.y));
    return Math.max(w, h) < 90 * settings.targetScale();
  }

  // ---------------------------------------------------------------- the status line
  statusIdle() {
    if (!editStore.get().on) return;
    const st = editStore.get();
    const parts: StatusPart[] = [{ text: st.tool === 'move' ? 'Drag a point: slide it on the scan' : 'Click a point to select it' },
      { keys: 'Left-drag', text: 'turn', shift: { keys: 'Left-drag', text: 'turn finely' } }, { keys: 'Middle-drag', text: 'move the view' }, { keys: 'Wheel', text: 'zoom' }, { keys: 'F', text: 'frame' }];
    setStatus('pose-edit', parts, { priority: PRIORITY.tool });
  }
  statusSelected() {
    const st = editStore.get(), name = st.selected;
    if (!name) { this.statusIdle(); return; }
    const p = POINT_BY_NAME.get(name)!, t = this.target(name);
    if (t && t.depth_mm > 0) {
      setStatus('pose-edit', [{ text: `Point ${p.number} · ${t.depth_mm.toFixed(1)} mm into the stone` }, { keys: 'Wheel', text: 'deeper or shallower' },
        { keys: 'PgUp', text: 'pull out' }, { keys: 'Alt+R', text: 'back to the surface' }], { priority: PRIORITY.tool });
      return;
    }
    setStatus('pose-edit', [{ text: `Point ${p.number} selected` }, { keys: 'Drag', text: st.tool === 'move' ? 'slide it on the scan' : 'turn the view (W: move points)' },
      { keys: 'Arrows', text: 'nudge 1 mm', shift: { keys: 'Arrows', text: 'nudge 0.2 mm' } }, { keys: 'PgDn', text: 'push in' }, { keys: 'Alt+G', text: 'put it back' },
      { keys: 'F', text: 'frame' }], { priority: PRIORITY.tool });
  }
  private statusDrag() {
    const D = this.drag, S = this.session;
    if (!D || !S) return;
    const p = POINT_BY_NAME.get(D.name)!, t = S.targets.get(D.name)!, a = S.positions([p]).get(D.name)!;
    const mm = 1000 * Math.hypot(a[0] - t.position[0], a[1] - t.position[1], a[2] - t.position[2]);
    const lead = D.depth > 0 ? `Point ${p.number} · ${D.depth.toFixed(1)} mm into the stone` : `Point ${p.number} · ${mm.toFixed(0)} mm from Anny's point`;
    setStatus('pose-edit-drag', [{ text: lead }, { keys: 'Shift', text: 'fine' }, { keys: 'Wheel', text: 'push in or pull out' }, { keys: 'Ctrl', text: 'snap' },
      { keys: 'Esc', text: 'put it back where it was' }], { priority: PRIORITY.drag });
  }

  // ---------------------------------------------------------------- keys
  private registerKeys() {
    defineScope({ id: 'pose-edit', parent: 'pose', active: () => editStore.get().on });
    defineScope({ id: 'pose-select', parent: 'pose-edit', active: () => !!editStore.get().selected });
    defineScope({ id: 'pose-grab', parent: 'pose-edit', active: () => !!this.grab, rank: 90 });
    const sel = () => editStore.get().selected;
    const tool = (t: Tool, label: string, key: string, desc: string, live = true): ActionDef => ({
      id: 'pose.tool.' + t, label, desc, keys: [key], group: 'Posing tools', tier: 'frequent', scope: 'pose-edit', place: 'Pose › Edit, the view\'s tools',
      checked: () => editStore.get().tool === t, run: () => { if (!live) { this.host.toast('The Turn tool arrives with the joint dial.'); return; } this.setTool(t); },
    });
    const list: ActionDef[] = [
      { id: 'pose.edit', label: 'Edit the pose', desc: 'Posing Edit on or off: points on the scan, joints and hands', keys: ['Tab'], group: 'Posing tools', tier: 'frequent',
        scope: 'pose', when: () => viewFocused() && ui.get().width !== 'phone', place: 'Pose › Edit', checked: () => editStore.get().on, run: () => { this.setOn(!editStore.get().on); } },
      tool('select', 'Select tool', 'Q', 'Clicks select; a drag on the background turns the camera'),
      tool('move', 'Move tool', 'W', 'Drags move points on the scan'),
      tool('turn', 'Turn tool', 'E', 'Drags turn a joint\'s rings', false),
      ...SECTIONS.map((s, i): ActionDef => ({ id: `pose.tool.${i + 1}`, label: s.label, desc: `Pose Edit: ${s.label.toLowerCase()}`, keys: [String(i + 1)], group: 'Posing tools',
        tier: 'occasional', scope: 'pose-edit', checked: () => editStore.get().section === s.id, run: () => this.setSection(s.id) })),
      { id: 'pose.grab', label: 'Grab the point', desc: 'The selected point follows the mouse until a click or Enter', keys: ['G'], group: 'Points', tier: 'frequent',
        scope: 'pose-select', hint: 'Select a point first', run: () => this.startGrab() },
      { id: 'pose.grab-confirm', label: 'Confirm the grab', keys: ['Enter'], group: 'Points', tier: 'frequent', scope: 'pose-grab', help: false, run: () => this.confirmGrab() },
      { id: 'pose.grab-cancel', label: 'Cancel the grab', keys: ['Escape'], group: 'Points', tier: 'frequent', scope: 'pose-grab', help: false, run: () => this.cancelGrab() },
      { id: 'pose.parent', label: 'Parent joint', keys: ['['], group: 'Points', tier: 'frequent', scope: 'pose-select', row: 'pose.hier', rowLabel: 'Parent joint, child joint',
        run: () => this.step(-1) },
      { id: 'pose.child', label: 'Child joint', keys: [']'], group: 'Points', tier: 'frequent', scope: 'pose-select', row: 'pose.hier', run: () => this.step(1) },
      { id: 'pose.finger-prev', label: 'Previous finger', keys: ['Shift+['], group: 'Points', tier: 'frequent', scope: 'pose-select', row: 'pose.finger',
        rowLabel: 'Previous finger, next finger', run: () => this.step(-1, true) },
      { id: 'pose.finger-next', label: 'Next finger', keys: ['Shift+]'], group: 'Points', tier: 'frequent', scope: 'pose-select', row: 'pose.finger', run: () => this.step(1, true) },
      { id: 'pose.select-all', label: 'Select the hand', desc: 'All the points of the hand in view', keys: ['A'], group: 'Points', tier: 'frequent', scope: 'pose-edit',
        run: () => { const s = sel(); const side = s ? POINT_BY_NAME.get(s)!.side : editStore.get().hand; if (!side) return false; this.selectHand(side as 'L' | 'R', false); } },
      { id: 'pose.select-none', label: 'Select nothing', keys: ['Alt+A'], group: 'Points', tier: 'frequent', scope: 'pose-edit', run: () => { this.select(null); this.selectHand(null, false); } },
      { id: 'pose.reset-point', label: 'Put the point back', desc: 'Back where the automatic placement had it (Shift: the whole hand or limb)', keys: ['Alt+G', 'Shift+Alt+G'],
        group: 'Points', tier: 'frequent', scope: 'pose-select', run: (c) => this.resetPoint(sel()!, c.shift) },
      { id: 'pose.reset-turn', label: 'Back to the surface', desc: 'A pushed point back to the surface (with the joints: reset the joint\'s turn)', keys: ['Alt+R', 'Shift+Alt+R'],
        group: 'Points', tier: 'frequent', scope: 'pose-select', run: () => this.toSurface(sel()!) },
      { id: 'pose.push', label: 'Push in', desc: 'Push the point 1 mm into the stone (Shift: 0.2 mm)', keys: ['PageDown', 'Shift+PageDown', 'Alt+PageDown'], group: 'Points',
        tier: 'frequent', scope: 'pose-select', repeat: true, run: (c) => this.push(sel()!, c.shift ? 0.2 : 1) },
      { id: 'pose.pull', label: 'Pull out', desc: 'Pull the point 1 mm out of the stone (Shift: 0.2 mm)', keys: ['PageUp', 'Shift+PageUp', 'Alt+PageUp'], group: 'Points',
        tier: 'frequent', scope: 'pose-select', repeat: true, run: (c) => this.push(sel()!, c.shift ? -0.2 : -1) },
      { id: 'pose.nudge', label: 'Nudge the point', desc: '1 mm along the surface (Shift: 0.2 mm)', keys: ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Shift+ArrowLeft',
        'Shift+ArrowRight', 'Shift+ArrowUp', 'Shift+ArrowDown'], group: 'Points', tier: 'frequent', scope: 'pose-select', repeat: true,
        run: (c) => { const k = c.event?.key || c.chord.replace('Shift+', ''); const dx = k === 'ArrowLeft' ? -1 : k === 'ArrowRight' ? 1 : 0, dy = k === 'ArrowUp' ? 1 : k === 'ArrowDown' ? -1 : 0;
          return this.nudge(sel()!, dx, dy, c.shift ? 0.2 : 1); } },
      { id: 'pose.clear', label: 'Clear the selection', keys: ['Escape'], group: 'Points', tier: 'constant', scope: 'pose-select', run: () => { this.select(null); } },
      { id: 'pose.scan', label: 'Show the scan', desc: 'Show or hide the statue\'s scan', keys: ['Shift+Z'], group: 'Display', tier: 'occasional', scope: 'pose-edit',
        checked: () => editStore.get().scan.shown, hint: 'This character has no scan', menu: 'display', menuLabel: 'Scan shown',
        run: () => { if (!this.scan) return false; this.setScanLook({ shown: !editStore.get().scan.shown }); } },
      { id: 'pose.scan-ghost', label: 'Scan as a ghost', desc: 'The scan see-through over the figure', group: 'Display', tier: 'occasional', scope: 'pose-edit', menu: 'display',
        menuLabel: 'Scan ghost', checked: () => editStore.get().scan.look === 'ghost' && editStore.get().scan.shown,
        run: () => { if (!this.scan) return false; this.setScanLook({ look: 'ghost', shown: true }); } },
      { id: 'pose.scan-solid', label: 'Scan solid', desc: 'The scan lit and solid beside the figure', group: 'Display', tier: 'occasional', scope: 'pose-edit', menu: 'display',
        menuLabel: 'Scan solid', checked: () => editStore.get().scan.look === 'solid' && editStore.get().scan.shown,
        run: () => { if (!this.scan) return false; this.setScanLook({ look: 'solid', shown: true }); } },
      { id: 'pose.undo', label: 'Undo the pose edit', keys: ['Ctrl+Z'], group: 'History', tier: 'constant', scope: 'pose-edit', help: false, run: () => this.undo() },
      { id: 'pose.redo', label: 'Redo the pose edit', keys: ['Ctrl+Shift+Z', 'Ctrl+Y'], group: 'History', tier: 'constant', scope: 'pose-edit', help: false, run: () => this.redo() },
      { id: 'pose.place', label: 'Place points', desc: 'Every point on the scan, from the fitted character', group: 'Points', tier: 'occasional', scope: 'pose-edit', place: 'Pose › Edit',
        run: () => { this.placeAll(); } },
    ];
    registerActions(list);
    this.registerPickers();
    // the pickers stay registered; enabled() keeps them out while Edit is off
    this.offs = [];
  }
}

// for the page tests: the editor's state in plain numbers (card frame, client pixels)
export function editorReport(ed: PoseEditor) {
  const S = ed.session;
  const anny = S ? S.positions() : new Map();
  const pts: Record<string, any> = {};
  for (const p of POINTS) {
    const t = S?.targets.get(p.name), a = anny.get(p.name), s = (ed as any).screen.get(p.name);
    let onScan: number | null = null;
    if (t && ed.bvh) { const q = ed.bvh.closest(t.surface || t.position); onScan = q ? q.distance : null; }
    pts[p.name] = { number: p.number, target: t ? t.position : null, on: t ? t.on : null, confidence: t ? t.confidence : null, depth: t ? t.depth_mm : 0,
      placed: t ? t.placed : null, anny: a || null, screen: s || null, visible: (ed as any).visible.get(p.name) ?? null, onScan };
  }
  return { state: { ...editStore.get(), hands: undefined, body: undefined }, points: pts, rotations: S ? S.rotations : null, undo: S ? S.undoStack.map((e) => e.label) : [],
    labels: S ? S.rig.labels : [] };
}

let EDITOR: PoseEditor | null = null;
export function installPoseEditor(host: PoseHost): PoseEditor {
  if (!EDITOR) EDITOR = new PoseEditor(host);
  (window as any).__poseEdit = EDITOR;
  (window as any).poseEditReport = () => editorReport(EDITOR!);
  return EDITOR;
}
export function poseEditor(): PoseEditor | null { return EDITOR; }

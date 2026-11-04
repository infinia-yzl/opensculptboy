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
import { cancelModal, confirmModal, startModal } from '../studio/modal.ts';
import { TriangleBVH } from './bvh.ts';
import { SkinnedBody, balance, gripFrom, signedDistance, type Balance, type Prop } from './contact.ts';
import { Frames } from './frames.ts';
import type { PoseHost } from './host.ts';
import { PointLayer, type DrawPoint, type Frame } from './layer.ts';
import { placePoints, partOf, reason, type PlaceInput } from './place.ts';
import { POINTS, POINT_BY_NAME, handPoints, pointsOfChain, type RefPoint } from './points.ts';
import { handAxes, type V3 } from './problem.ts';
import { ghostOf, readScan, type ScanMesh } from './scan.ts';
import { EditSession, PIN_WEIGHT, type Pin, type Target } from './session.ts';
import { ScanView, type ScanLook } from './view3d.ts';
import { AXES, BODY_JOINTS, Joints, jointName, type JointValues } from './joints.ts';
import { FINGERS, FINGER_LABELS, HandControls, PRESETS, RELAXED, copyMirrored, mirrorPose, type HandShape } from './hands.ts';
import { transforms, coordinatesOf, rotationOf, expRotation, mul3, buildProblem, type PointInput, type PinInput } from './problem.ts';
import { PointSolver, logRotation as logOf } from './solve.ts';
import { cardTurns } from '../pose_reference.ts';
import { apply4, fromColumnMajor, movedInPose, mul4, pageMove, placeAmount, rehold, rotationOf as rot4, type M4, type Place } from '../wearables/place.ts';
import { alongAxis, heldByChoices, hitHandles, isRest, onPlane, pageNumbers, placeFromNumbers, sweep, translation, turnAbout, type HandlePart, type HeldBy } from './props.ts';
import type { PropRig } from './host.ts';

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
  // the joints and the hands (phase 2)
  joint: string | null;              // the selected bone's label
  jv: JointValues | null;
  limits: boolean;
  handSide: 'L' | 'R';
  handShape: HandShape & { fist: number; thumb: number };
  preset: string | null;
  userPresets: { id: string; label: string; shape: HandShape }[];
  grip_report: { finger: string; clearance: number; inside: boolean; opened?: boolean; wedged?: boolean; blocked?: boolean }[];
  grip_blocked: boolean;            // a finger is inside the prop at every curl from its present one to straight
  grip_carried: boolean;            // the prop moves with the gripping hand (it is bound to a bone of that arm)
  kept: string[];                   // the chains the last placement left as they stood
  clip: string;                      // what Ctrl+C took
  // the limbs (phase 3)
  limb: End | null;                  // the end whose pole handle shows
  pinned: End[];
  follow: boolean;                   // the body follows a dragged hand
  // the library (phase 4)
  poses: SavedPose[];
  blend: { a: string; b: string; t: number } | null;
  applyTo: string;
  // the props (moved by hand): the selected one, every rigid prop shown, the point turns go round, the skin inside the
  // selected prop
  prop: string | null;
  propList: PropInfo[];
  propPivot: 'carrier' | 'centre';
  propInside: { mm: number; part: string } | null;
  version: number;
}
// a prop of the panel's list: its reference and label, the bone that carries it and its plain name, the bone it was made
// on, how far its place moves it, its place as the panel's numbers, the choices of Held by, and the hand that holds it
export interface PropInfo { ref: string; label: string; carrier: string; carrierName: string; made: string | null; moved: { mm: number; deg: number };
  numbers: { mm: number[]; deg: number[] }; choices: HeldBy[]; side: 'L' | 'R' | null }

const init = (): EditState => ({
  on: false, tool: 'select', section: 'points',
  scan: { available: false, loading: false, shown: true, look: 'ghost', opacity: 0.45, error: '', credit: '' },
  selected: null, hand: null, filter: 'all', counts: { placed: 0, look: 0, off: 0, total: POINTS.length }, looks: [], point: null,
  hands: { L: [], R: [] }, body: [], dragging: false, magnifier: true, behind: true, pins: [], pinDraft: null, grip: null, squeeze: 1, props: [],
  balance: { shown: false, inside: null, margin: null }, undo: '', redo: '', version: 0,
  joint: null, jv: null, limits: true, handSide: 'R', handShape: { curl: [0, 0, 0, 0, 0], spread: 0, fist: 0, thumb: 0 }, preset: null, userPresets: loadPresets(),
  grip_report: [], grip_blocked: false, grip_carried: false, kept: [], clip: '', limb: null, pinned: [], follow: false, poses: loadPoses(), blend: null, applyTo: 'all',
  prop: null, propList: [], propPivot: 'carrier', propInside: null,
});

// a chain's plain name ("arm.l" -> "Left arm")
const CHAIN_NAMES: Record<string, string> = { arm: 'arm', leg: 'leg', hand: 'hand', spine: 'Spine', neck: 'Head and neck' };
function chainName(c: string): string {
  const [base, side] = c.split('.');
  const word = CHAIN_NAMES[base] || base;
  return side ? `${side === 'l' ? 'Left' : 'Right'} ${word}` : word;
}
export type End = 'hand.L' | 'hand.R' | 'foot.L' | 'foot.R' | 'hips' | 'head';
export const ENDS: { id: End; label: string }[] = [
  { id: 'hand.R', label: 'Right hand' }, { id: 'hand.L', label: 'Left hand' }, { id: 'foot.R', label: 'Right foot' }, { id: 'foot.L', label: 'Left foot' },
  { id: 'hips', label: 'Hips' }, { id: 'head', label: 'Head' },
];
// the bone at an end, the chain that moves it, the bones that keep its turn, and its pole's bone
const END_BONES: Record<string, { bone: string; chain: string; keep: string[]; pole: string | null }> = {
  'hand.L': { bone: 'wrist.L', chain: 'arm.l', keep: ['finger3-1.L', 'finger2-1.L'], pole: 'lowerarm01.L' },
  'hand.R': { bone: 'wrist.R', chain: 'arm.r', keep: ['finger3-1.R', 'finger2-1.R'], pole: 'lowerarm01.R' },
  'foot.L': { bone: 'foot.L', chain: 'leg.l', keep: ['toe1-1.L', 'toe5-1.L'], pole: 'lowerleg01.L' },
  'foot.R': { bone: 'foot.R', chain: 'leg.r', keep: ['toe1-1.R', 'toe5-1.R'], pole: 'lowerleg01.R' },
  hips: { bone: 'upperleg01.L', chain: 'spine', keep: [], pole: null },
  head: { bone: 'head', chain: 'neck', keep: [], pole: null },
};
// the library: poses saved in this browser, as the card's pose (each bone's turn in Anny's frame, quaternions)
export interface SavedPose { id: string; label: string; bones: Record<string, number[]> }
const POSE_KEY = 'osb.poseLibrary';
// the id of a built-in pose of the page's library in the Library tool
export const BUILTIN = 'builtin:';
function loadPoses(): SavedPose[] {
  try { const v = JSON.parse(localStorage.getItem(POSE_KEY) || '[]'); return Array.isArray(v) ? v.filter((x) => x && x.bones && x.label) : []; } catch { return []; }
}
function savePoses(list: SavedPose[]) { try { localStorage.setItem(POSE_KEY, JSON.stringify(list)); } catch { /* kept for this visit */ } }
// the parts a saved pose applies to
export const PARTS: { id: string; label: string; test: (label: string) => boolean }[] = [
  { id: 'all', label: 'Whole body', test: () => true },
  { id: 'hands', label: 'Both hands', test: (l) => /^(finger|metacarpal)/.test(l) },
  { id: 'hand.R', label: 'Right hand', test: (l) => /^(finger|metacarpal)/.test(l) && l.endsWith('.R') },
  { id: 'hand.L', label: 'Left hand', test: (l) => /^(finger|metacarpal)/.test(l) && l.endsWith('.L') },
  { id: 'arms', label: 'Arms', test: (l) => /^(clavicle|shoulder01|upperarm|lowerarm|wrist)/.test(l) },
  { id: 'legs', label: 'Legs', test: (l) => /^(pelvis|upperleg|lowerleg|foot|toe)/.test(l) },
  { id: 'upper', label: 'Spine and head', test: (l) => /^(spine|neck|head)/.test(l) },
];

// the user's own hand presets, in this browser
const PRESET_KEY = 'osb.handPresets';
function loadPresets(): { id: string; label: string; shape: HandShape }[] {
  try { const v = JSON.parse(localStorage.getItem(PRESET_KEY) || '[]'); return Array.isArray(v) ? v.filter((x) => x && x.shape && Array.isArray(x.shape.curl)) : []; } catch { return []; }
}
function savePresets(list: { id: string; label: string; shape: HandShape }[]) { try { localStorage.setItem(PRESET_KEY, JSON.stringify(list)); } catch { /* kept for this visit */ } }

export const editStore = createStore<EditState>(init());

const MAG_SIZE = 220, MAG_ZOOM = 4;
// three rays out of a point, along no axis, for the parity test of a prop's inside
const PARITY_RAYS = [[0.267, 0.535, 0.802], [-0.802, 0.267, 0.535], [0.535, -0.802, 0.267]];
const SNAP_PX = 10;
// metres: the farthest that skin can lie inside a prop for the props' contact shading and Move it out
const INSIDE_REACH = 0.12;
// Move it out: the cost of a millimetre moved against a millimetre of the palm inside the prop, and the longest move (mm)
const MOVE_COST = 0.25, MOVE_LIMIT = 40;

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
  // the magnifier's centre (client pixels): the pointer over the scan; off the scan, the point's last spot on the
  // surface (off), so the magnifier keeps showing the surface the point holds to
  private mag: { x: number; y: number; off?: boolean } | null = null;
  private visible = new Map<string, boolean>();
  private screen = new Map<string, { x: number; y: number; depth: number }>();
  private body: SkinnedBody | null = null;
  private bodyTris: { bvh: TriangleBVH; rows: Uint32Array } | null = null;
  private bal: Balance | null = null;
  private grab: { name: string; before: any; x0: number; y0: number; off: (() => void) } | null = null;
  private pointer: { x: number; y: number } | null = null;   // the pointer's last place over the page (client pixels)
  private propBvh: TriangleBVH | null = null;
  private surfaceOff: (() => void) | null = null;

  constructor(host: PoseHost) {
    this.host = host;
    this.frames = new Frames(host);
    this.view = new ScanView(host);
    this.registerKeys();
    if (typeof addEventListener === 'function') addEventListener('pointermove', (e: PointerEvent) => { this.pointer = { x: e.clientX, y: e.clientY }; }, { passive: true });
    // unsaved pose or point edits: the browser asks before the page goes (a reload, a closed tab, Alt+Left)
    if (typeof addEventListener === 'function') addEventListener('beforeunload', (e: BeforeUnloadEvent) => {
      if (!this.promptOn() || !this.unsavedEdits()) return;
      e.preventDefault();
      e.returnValue = '';
    });
  }

  // ---------------------------------------------------------------- the store
  private publish(patch: Partial<EditState> = {}) {
    const s = this.session;
    const info = (p: RefPoint): PointInfo => {
      const t = s?.targets.get(p.name);
      const state: PointInfo['state'] = !t ? 'unplaced' : !t.on && t.placed === 'auto' && t.confidence < 0.5 ? 'look' : !t.on ? 'off' : 'placed';
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
    const jl = patch.joint !== undefined ? patch.joint : st.joint;
    const J = s ? this.joints() : null, jb = jl && J ? J.bone(jl) : undefined;
    const jv = J && jb !== undefined && s ? J.values(jb, s.rotations) : null;
    const side = patch.handSide || st.handSide, H = s ? this.hands() : null;
    const hs = H && s ? { ...H.shape(s.rotations, side), fist: H.fist(s.rotations, side), thumb: H.thumb(s.rotations, side) } : st.handShape;
    editStore.set({
      jv, handShape: hs,
      counts, looks: all.filter((p) => p.state === 'look'), point,
      hands: { R: handPoints('R').map(info), L: handPoints('L').map(info) }, body: POINTS.filter((p) => p.group === 'body').map(info),
      pins: (s?.pins || []).map((q) => ({ name: q.name })), undo: s?.undoStack.at(-1)?.label || '', redo: s?.redoStack.at(-1)?.label || '',
      propList: this.propInfos(),
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
    let fresh = false;
    if (!this.session) {
      this.session = new EditSession(rig, this.frames.skin(), R);
      this.session.onChange = () => this.poseChanged();
      fresh = true;
    } else { this.session.setRig(rig, this.frames.skin()); this.session.setPose(R); }
    this.body = null; this.bodyTris = null;
    const card = this.host.card();
    if (card && card.card !== this.scanCard) {
      // a new card: its saved points and pins
      const ref = card.card?.detail?.fit?.reference;
      this.session.readCard(ref?.points, ref?.pins);
      this.session.undoStack.length = 0; this.session.redoStack.length = 0;
      fresh = true;
    }
    // the props' places as the card holds them (the history keeps them with the pose)
    this.session.places = Object.fromEntries(this.rigs().map((r) => [r.ref, r.place ? JSON.parse(JSON.stringify(r.place)) : null]));
    // the state as opened: edits after it are unsaved until Save points or Save the pose
    if (fresh) this.savedSig = this.signature();
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
    if (this.propDrag) this.cancelPropDrag();
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
  setSection(section: Section) {
    const st = editStore.get();
    const tool: Tool = section === 'joints' ? 'turn' : section === 'points' && st.tool === 'turn' ? 'select' : st.tool;
    if (section === 'hands' && st.selected) { const p = POINT_BY_NAME.get(st.selected); if (p && p.group !== 'body') this.publish({ handSide: p.side as 'L' | 'R' }); }
    this.publish({ section, tool });
    this.statusIdle();
  }

  // ---------------------------------------------------------------- the pose
  private poseChanged() {
    if (!this.session) return;
    this.host.setPose(this.frames.pageQuaternions(this.session.rotations));
    this.syncPlaces();
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
    // the bones bound by their map from the rest pose to the reference one (the zero of the local-ref turns)
    const bind = this.frames.reference().bind;
    this.body = new SkinnedBody(S.rig, rest, idx, w, 8, Uint32Array.from(tris), (k) => bind[k] || null);
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
      // as the Python placement: a point is on when its confidence reaches ON; the others need a look, and the solve
      // leaves them out until the owner moves them or switches them on
      const t: Target = { position: e.position as V3, on: e.on, weight: 1, confidence: e.confidence, placed: 'auto', depth_mm: 0,
        auto: e.position as V3, reason: e.on ? undefined : reason(e) };
      if (!e.on) look++;
      S.targets.set(e.name, t);
    }
    // only the chains with enough confident points turn; the others keep their pose (a lone point never pulls a limb)
    const solved = S.confidentChains(), kept = S.placedChains().filter((c) => !solved.includes(c));
    for (const c of solved) S.solve(c);
    this.publish({ kept });
    const n = placed.filter((e) => e.method !== 'none').length;
    this.host.toast((look ? `Placed ${n} points. ${look} need a look.` : `Placed ${n} points.`)
      + (kept.length ? ` ${kept.map(chainName).join(', ')} kept ${kept.length > 1 ? 'their' : 'its'} pose: too few sure points.` : ''));
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
    this.publish({ selected: name, hand: name ? hand : editStore.get().hand, ...(name ? { prop: null } : {}) });
    if (!p) { if (!editStore.get().prop) { setSelection(null); this.statusIdle(); } return; }
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
  private scanHit(cx: number, cy: number, near: V3 | null): { point: V3; ray: V3; off?: boolean } | null {
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
      return { point: q.point, ray: d, off: true };
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
    // off the scan (the pointer's ray meets no scan) the point keeps its last spot on the surface, and the magnifier
    // holds on that spot; a push with the wheel still takes the point in
    const raw = this.scanHit(D.vx, D.vy, D.last), hit = raw && !raw.off ? raw : null;
    if (!hit) {
      const a = cam().toScreen(this.v3(this.frames.toWorld(D.last)));
      this.mag = { x: a.x, y: a.y, off: true };
      this.dirty = true;
      this.statusDrag();
      if (dx || dy) return;
    }
    let spot = hit ? hit.point : D.last;
    const t = S.targets.get(D.name)!;
    if (snap && t.auto) {
      const a = cam().toScreen(this.v3(this.frames.toWorld(t.auto)));
      if (Math.hypot(a.x - D.vx, a.y - D.vy) <= SNAP_PX) spot = t.auto.slice() as V3;
    }
    D.last = spot;
    if (hit) this.mag = { x: D.vx, y: D.vy };
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

  // G: the selected point follows the mouse until a click or Enter; Esc or a right-click puts it back. The input,
  // the axis keys, the typed value and the status line are the studio's modal grab (studio/modal.ts); X, Y or Z
  // keeps the move along the screen's picture of that axis of the view, and a typed number moves the place that many
  // millimetres along it, off the surface
  startGrab(): boolean {
    const name = editStore.get().selected;
    if (!name || !this.bvh || this.drag || this.grab) return false;
    const t = this.target(name);
    if (!t) return false;
    const start = t.position.slice() as V3, w0 = this.frames.toWorld(start);
    const s = cam().toScreen(this.v3(w0));
    this.beginDrag(name, s.x, s.y);
    let lx = 0, ly = 0;
    const axisOf = (a: 'x' | 'y' | 'z'): V3 => (a === 'x' ? [1, 0, 0] : a === 'y' ? [0, 1, 0] : [0, 0, 1]);
    this.grab = { name, before: null, x0: s.x, y0: s.y, off: () => {} };
    startModal({
      kind: 'grab', label: `point ${POINT_BY_NAME.get(name)!.number}`,
      update: (v) => {
        const D = this.drag, S = this.session;
        if (!D || !S) return;
        if (v.axis === 'free') {
          this.dragMove(v.dx - lx, v.dy - ly, v.snap); lx = v.dx; ly = v.dy;
          return this.drag ? POINT_BY_NAME.get(name)!.label : undefined;
        }
        // the axis of the page (Y up) as a direction on the screen and in the card frame
        const ax = axisOf(v.axis), q = cam().toScreen(this.v3([w0[0] + ax[0] * 0.1, w0[1] + ax[1] * 0.1, w0[2] + ax[2] * 0.1]));
        const sx = q.x - s.x, sy = q.y - s.y, px = Math.hypot(sx, sy) || 1;
        const mm = v.typed !== null ? v.typed : (v.dx * sx + v.dy * sy) / px / (px / 100);
        const d = this.frames.dirToCard(ax), r = (v.snap ? Math.round(mm) : mm) / 1000;
        const position: V3 = [start[0] + d[0] * r, start[1] + d[1] * r, start[2] + d[2] * r];
        D.last = position;
        S.dragTo(name, position, { depth_mm: 0 });
        this.publish();
        return `${(r * 1000).toFixed(1)} mm`;
      },
      confirm: () => { this.grab = null; this.endDrag(); },
      cancel: () => { this.grab = null; this.cancelDrag(); },
    }, { x: s.x, y: s.y });
    return true;
  }
  confirmGrab(): boolean { if (!this.grab) return false; confirmModal(); return true; }
  cancelGrab(): boolean { if (!this.grab) return false; cancelModal(); return true; }
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
  // Grip (DESIGN-0013, "Grip a prop"; GraspIt's closing of the fingers) starts from the hand as it stands: each finger
  // clear of the prop closes from its present curl, and a finger that touches or enters the prop first opens along its
  // curl only as far as it must (contact.gripFrom). A statue's hand already wraps its prop, and a hand opened straight
  // drives a finger through the prop's guard (David's index finger crossed the sword's guard bar), so the open start
  // blocked the grip there. Each finger then closes to contact with 1 mm to spare, and Squeeze sets each finger between
  // its start and its contact. A finger inside the prop at every curl blocks the grip with a note; Lift out moves the
  // hand off the prop (the arm follows) when the prop stands apart from the hand.
  private gripStart: number[][] | null = null;
  private propOutside = 1;
  private propBone: number | null = null;
  lastLift: { normal: number[]; trace: { mm: number; miss_mm: number; worst?: number }[] } | null = null;   // the last Lift out's steps, for the tests
  setGrip(grip: { hand: 'L' | 'R'; prop: string } | null) {
    this.propBvh = null; this.gripStart = null;
    this.publish({ grip, grip_blocked: false, grip_report: [], grip_carried: grip ? this.carries(grip.hand, grip.prop) : false });
  }

  // whether the prop moves with the hand's arm: its bone is a bone of that arm (the shoulder down to the fingers)
  private carries(side: 'L' | 'R', name: string): boolean {
    const S = this.session, prop = this.host.props().find((p) => p.name === name);
    if (!S || !prop?.bone) return false;
    let b = S.rig.labels.indexOf(prop.bone);
    const arm = new RegExp(`^(clavicle|shoulder|upperarm|lowerarm|wrist|metacarpal|finger)[0-9-]*\\.${side}$`);
    while (b >= 0) { if (arm.test(S.rig.labels[b])) return true; b = S.rig.parents[b]; }
    return false;
  }

  // the prop as a signed distance (card frame), as it stands now
  private propSdf(): Prop | null {
    const g = editStore.get().grip;
    if (!g) return null;
    const prop = this.host.props().find((p) => p.name === g.prop);
    if (!prop) { this.host.toast('That prop is not shown.'); return null; }
    if (!this.propBvh) {
      // the prop in the card frame (it stands where it is shown now)
      const P = new Float32Array(prop.positions.length);
      for (let i = 0; i < P.length; i += 3) P.set(this.frames.fromWorld([prop.positions[i], prop.positions[i + 1], prop.positions[i + 2]]), i);
      this.propBvh = new TriangleBVH(P, prop.indices);
      this.propBone = prop.bone ? this.session!.rig.labels.indexOf(prop.bone) : null;
      // the side the faces' winding calls outside: a mesh wound clockwise has a negative volume, and its normals point in
      let vol = 0;
      const I = prop.indices;
      for (let t = 0; t + 2 < I.length; t += 3) {
        const a = I[t] * 3, b = I[t + 1] * 3, c = I[t + 2] * 3;
        vol += P[a] * (P[b + 1] * P[c + 2] - P[b + 2] * P[c + 1]) - P[a + 1] * (P[b] * P[c + 2] - P[b + 2] * P[c]) + P[a + 2] * (P[b] * P[c + 1] - P[b + 1] * P[c]);
      }
      this.propOutside = vol < 0 ? -1 : 1;
    }
    const bvh = this.propBvh, outside = this.propOutside;
    const sdf: Prop = (X: Float64Array) => {
      const out = new Float64Array(X.length / 3);
      for (let i = 0; i < out.length; i++) {
        const p = [X[i * 3], X[i * 3 + 1], X[i * 3 + 2]], q = bvh.closest(p);
        if (!q) { out[i] = Infinity; continue; }
        // inside or out: the closest face's side, and where a face's winding is not to be trusted (within 3 cm of the
        // prop, where the sign matters), the parity of crossings along three rays, by majority (the pieces are closed;
        // on David's sword this agrees with the generalised winding number at every hand vertex within 3 cm)
        const n = bvh.normal(q.tri), s = (p[0] - q.point[0]) * n[0] + (p[1] - q.point[1]) * n[1] + (p[2] - q.point[2]) * n[2];
        let inside = s * outside < 0;
        if (q.distance < 0.03) {
          let odd = 0;
          for (const d of PARITY_RAYS) if (bvh.rayAll(p, d).length % 2 === 1) odd++;
          inside = odd >= 2;
        }
        out[i] = inside ? -q.distance : q.distance;
      }
      return out;
    };
    return sdf;
  }
  private gripOf(R: number[][], side: 'L' | 'R', sdf: Prop, value: number) {
    return gripFrom(this.skinnedBody(), R, side === 'R' ? 'hand.r' : 'hand.l', [sdf], value);
  }
  setSqueeze(value: number, record = true): boolean {
    const S = this.session, g = editStore.get().grip;
    if (!S || !g) return false;
    const sdf = this.propSdf();
    if (!sdf) return false;
    if (record) this.record('Squeeze');
    if (!this.gripStart) this.gripStart = S.rotations.map((r) => r.slice());
    const r = this.gripOf(this.gripStart, g.hand, sdf, value);
    S.setPose(r.rotations);
    this.poseChanged();
    // each finger's gap to the prop at the chosen squeeze (negative: inside it), for the panel
    const report = FINGERS.map((f, k) => {
      const q = r.report[`finger${k + 1}`];
      return { finger: f, clearance: q ? q.clearance_mm : NaN, inside: !!q && q.clearance_mm < 0, opened: !!q?.opened, wedged: !!q?.wedged, blocked: !!q?.blocked };
    });
    this.publish({ squeeze: value, grip_report: report, grip_blocked: report.some((x) => x.blocked) });
    return true;
  }
  // Lift out: the hand moved off the prop by the arm's IK (its turn kept), in steps of 2 mm up to 60 mm, until every
  // finger can clear the prop; then the grip closes from there. The way out is the hand's depenetration direction (as a
  // physics engine separates two bodies in contact): the sum over the hand's vertices inside the prop of their depth
  // times the direction to their closest point on the prop's surface, with the palm's normal both ways after it. A prop
  // that moves with the hand's arm moves with the lift: `keepProp` then puts it back where it stood, as a new place relative
  // to the bone that carries it (the hand leaves the prop); without it Lift out leaves such a prop alone, and Move it out
  // (moveOut) moves the prop off the hand instead.
  liftHand(keepProp = false): boolean {
    const S = this.session, g = editStore.get().grip;
    if (!S || !g) return false;
    const sdf = this.propSdf();
    if (!sdf) return false;
    const label = g.prop.split('/').pop() || 'prop';
    const carried = this.carries(g.hand, g.prop);
    if (carried && !keepProp) { this.host.toast(`The ${label} moves with this hand, so lifting the hand takes it along.`); return false; }
    const rig = carried ? this.rigOf(g.prop) : null, held = rig ? { place: rig.place, carrier: rig.carrier, skin: this.host.skinMatrix(rig.carrier), head: this.headPage(rig.carrier) } : null;
    if (!this.gripStart) this.gripStart = S.rotations.map((r) => r.slice());
    const side = g.hand, end = ('hand.' + side) as End, before = S.snapshot(), start = this.gripStart;
    S.setPose(start);
    const body = this.skinnedBody();
    const handBones = S.rig.labels.map((l, i) => [l, i] as [string, number]).filter(([l]) => new RegExp(`^(wrist|metacarpal|finger)[0-9-]*\\.${side}$`).test(l)).map(([, i]) => i);
    const rows = body.rowsOf(handBones), V = body.vertices(start, rows), d = signedDistance(V, [sdf]), bvh = this.propBvh!;
    const dir = [0, 0, 0];
    for (let i = 0; i < d.length; i++) {
      if (d[i] >= 0) continue;
      const p = [V[i * 3], V[i * 3 + 1], V[i * 3 + 2]], q = bvh.closest(p)!, l = q.distance || 1;
      for (let k = 0; k < 3; k++) dir[k] += -d[i] * (q.point[k] - p[k]) / l;
    }
    const w = this.headOf('wrist.' + side)!, a = this.headOf('finger3-1.' + side)!, ix = this.headOf('finger2-1.' + side)!, l5 = this.headOf('finger5-1.' + side)!;
    const along = [a[0] - w[0], a[1] - w[1], a[2] - w[2]], across = [ix[0] - l5[0], ix[1] - l5[1], ix[2] - l5[2]];
    const c = [along[1] * across[2] - along[2] * across[1], along[2] * across[0] - along[0] * across[2], along[0] * across[1] - along[1] * across[0]];
    const unit = (v: number[]) => { const n = Math.hypot(v[0], v[1], v[2]); return n > 1e-9 ? v.map((x) => x / n) : null; };
    const palm = unit(c) || [0, 0, 1], out = unit(dir);
    const ways = [...(out ? [out] : []), palm, palm.map((x) => -x)];
    const trace: { mm: number; miss_mm: number; worst?: number }[] = [];
    const tryMove = (n: number[], dd: number): number[][] | null => {
      S.setPose(start);
      const { problem, bones } = this.limbProblem(end), sv = new PointSolver(problem), e = END_BONES[end];
      const target = (p: V3): V3 => [p[0] + n[0] * dd, p[1] + n[1] * dd, p[2] + n[2] * dd];
      sv.setTarget(0, target(this.headOf(e.bone)!));
      e.keep.forEach((k, j) => sv.setTarget(1 + j, target(this.headOf(k)!)));
      const r = sv.solve(25), R = start.map((x) => x.slice());
      for (const b of bones) R[b] = Array.from(r.rotations.subarray(b * 9, b * 9 + 9));
      S.setPose(R);
      const reached = this.headOf(e.bone)!, t = target(w), miss = Math.hypot(reached[0] - t[0], reached[1] - t[1], reached[2] - t[2]);
      trace.push({ mm: Math.round(dd * 1000), miss_mm: Math.round(miss * 10000) / 10 });
      return miss < 0.002 ? R : null;
    };
    this.lastLift = { normal: ways[0], trace };
    let found: number[][] | null = null;
    for (const n of ways) {
      for (let mm = 2; mm <= 60 && !found; mm += 2) {
        const R = tryMove(n, mm / 1000);
        if (!R) continue;
        const r = this.gripOf(R, side, sdf, 0), worst = Math.min(...Object.values(r.report).map((f) => f.clearance_mm));
        trace[trace.length - 1].worst = Math.round(worst * 10) / 10;
        if (!Object.values(r.report).some((f) => f.blocked)) { found = R; this.lastLift.normal = n; }
      }
      if (found) break;
    }
    if (!found) { S.restore(before); this.host.toast(`The arm cannot take the hand off the ${label}.`); this.publish(); return false; }
    S.restore(before);
    this.record('Lift the hand out');
    this.gripStart = found;
    S.setPose(found);
    if (rig && held?.skin && held.head) {
      // the hand moved off the prop that it carries: the prop stays where it stood, on the same bone
      const now = this.host.skinMatrix(held.carrier);
      if (now) { const p = rehold(held.place, held.skin, now, held.head, held.head, held.carrier, this.host.frame()); this.putPlace(rig.ref, isRest(p, rig.made) ? null : p); }
    }
    return this.setSqueeze(editStore.get().squeeze, false);
  }

  setBalance(shown: boolean) { this.bal = null; this.publish({ balance: { ...editStore.get().balance, shown } }); }

  // ---------------------------------------------------------------- props, moved by hand (DESIGN-0013, "Props")
  // A prop worn as a rigid piece (a statue's sword, its base) moves by hand. A click selects it in the view (a ray onto
  // its triangles when no skin lies in front of them) or in the panel's list. The Move tool's drag takes it along the
  // view's plane through its pivot and the move handle's arrows along one axis of the view; the Turn tool's rings turn it
  // round one axis through the pivot (the head of the bone that carries it, or its centre) and its outer ring round the
  // line of sight; G and R grab and turn it with the studio's modal keys (X, Y, Z, typed millimetres and degrees); the
  // arrows nudge it 1 mm across the view and Page Up and Down 1 mm towards and away from the camera; the panel takes
  // typed values. Every move is one step of the history and goes into the card at once as the piece's place
  // (wearables/place.ts), relative to the bone that carries it, so the prop follows the hand. Held by gives it to another
  // bone, or to the root, where it stands in the pose shown.
  private propBvhs = new WeakMap<Float64Array, { bvh: TriangleBVH; outside: number; centre: number[]; lo: number[]; hi: number[] }>();
  private propDrag: { ref: string; part: HandlePart; start: Place | null; before: any; pivot: number[]; x0: number; y0: number; ex: number; ey: number;
    sweep: number; axes: number[][]; moved: boolean; label: string } | null = null;
  private propHandles: { centre: { x: number; y: number }; arrows?: [number, number][][]; rings?: [number, number][][]; outer?: number; axes: number[][] } | null = null;
  private hoverProp: string | null = null;
  private insideAt: number[] | null = null;
  private insideTime = 0;

  private rigs(): PropRig[] { return this.host.ready() && typeof this.host.propRigs === 'function' ? this.host.propRigs() : []; }
  private rigOf(ref: string | null | undefined): PropRig | null { return ref ? this.rigs().find((r) => r.ref === ref) || null : null; }
  private holderM(): M4 { return fromColumnMajor(this.host.holder().elements); }
  private inv4(M: ArrayLike<number>): M4 { const m = new this.host.THREE.Matrix4().set(...(Array.from(M) as [number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number])).invert(); return fromColumnMajor(m.elements); }
  private headPage(bone: string): number[] | null { const i = this.host.bones().indexOf(bone); return i < 0 ? null : this.host.restHeads()[i]; }
  boneName(b: string): string { return b === 'root' ? 'The ground' : jointName(b).name; }
  // the matrix that takes a prop's rest vertices (the page's frame) to the world in the pose shown
  private propWorld(r: PropRig): M4 | null { const S = this.host.skinMatrix(r.carrier); return S ? mul4(this.holderM(), S) : null; }
  // a prop's triangles as a BVH of its rest vertices, the side its faces' winding calls outside, and its box
  private restOf(r: PropRig) {
    let c = this.propBvhs.get(r.vertices);
    if (!c) {
      const P = r.vertices, I = r.triangles, lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < P.length; i += 3) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], P[i + k]); hi[k] = Math.max(hi[k], P[i + k]); }
      let vol = 0;
      for (let t = 0; t + 2 < I.length; t += 3) {
        const a = I[t] * 3, b = I[t + 1] * 3, cc = I[t + 2] * 3;
        vol += P[a] * (P[b + 1] * P[cc + 2] - P[b + 2] * P[cc + 1]) - P[a + 1] * (P[b] * P[cc + 2] - P[b + 2] * P[cc]) + P[a + 2] * (P[b] * P[cc + 1] - P[b + 1] * P[cc]);
      }
      c = { bvh: new TriangleBVH(P, I), outside: vol < 0 ? -1 : 1, centre: lo.map((x, k) => (x + hi[k]) / 2), lo, hi };
      this.propBvhs.set(r.vertices, c);
    }
    return c;
  }
  private sideOf(bone: string): 'L' | 'R' | null { const m = /^(wrist|metacarpal|finger)[0-9-]*\.(L|R)$/.exec(bone); return m ? (m[2] as 'L' | 'R') : null; }
  private propInfos(): PropInfo[] {
    if (!this.host.ready() || typeof this.host.propRigs !== 'function') return [];
    const bones = this.host.bones(), F = this.host.frame();
    return this.rigs().map((r) => ({ ref: r.ref, label: r.label, carrier: r.carrier, carrierName: this.boneName(r.carrier), made: r.made, moved: placeAmount(r.place),
      numbers: pageNumbers(r.place, F), choices: heldByChoices(r.made, bones, (b) => this.boneName(b)), side: this.sideOf(r.carrier) }));
  }
  // the view's hits on the props: the first triangle along the ray in each prop's rest frame, unless skin lies in front
  private hitProps(p: PointerInfo): PickHit[] {
    const out: PickHit[] = [], rigs = this.rigs();
    if (!rigs.length) return out;
    const o = [p.ray.origin.x, p.ray.origin.y, p.ray.origin.z], d = [p.ray.direction.x, p.ray.direction.y, p.ray.direction.z];
    const skin = this.host.bodyRay(p.ray.origin, p.ray.direction);
    for (const r of rigs) {
      const T = this.propWorld(r);
      if (!T) continue;
      const Ti = this.inv4(T), ro = apply4(Ti, o), R = rot4(Ti), rd = [0, 1, 2].map((k) => R[k * 3] * d[0] + R[k * 3 + 1] * d[1] + R[k * 3 + 2] * d[2]);
      const h = this.restOf(r).bvh.ray(ro, rd);
      if (!h || (skin && skin.distance < h.t - 1e-4)) continue;
      out.push({ item: r.ref, distPx: 0, depth: h.t, label: r.label });
    }
    return out;
  }
  // where turns of the prop go round (the world): the carrying bone's head, else the prop's centre
  private propPivotWorld(r: PropRig): number[] | null {
    const T = this.propWorld(r);
    if (!T) return null;
    if (editStore.get().propPivot === 'carrier' && r.carrier !== 'root') { const h = this.headPage(r.carrier); if (h) return apply4(T, h); }
    return apply4(T, this.restOf(r).centre);
  }
  private propCorners(r: PropRig): { x: number; y: number; z: number }[] {
    const T = this.propWorld(r), c = this.restOf(r);
    if (!T) return [];
    const out: { x: number; y: number; z: number }[] = [];
    for (let i = 0; i < 8; i++) { const w = apply4(T, [i & 1 ? c.hi[0] : c.lo[0], i & 2 ? c.hi[1] : c.lo[1], i & 4 ? c.hi[2] : c.lo[2]]); out.push({ x: w[0], y: w[1], z: w[2] }); }
    return out;
  }
  // F's steps from a prop: the prop, the prop with the bone that holds it, then the whole figure
  private propFraming(ref: string, step: number) {
    const r = this.rigOf(ref);
    if (!r) return null;
    if (step === 0) return { box: this.propCorners(r), label: r.label };
    if (step === 1 && r.carrier !== 'root') {
      const T = this.propWorld(r), h = this.headPage(r.carrier);
      if (T && h) { const w = apply4(T, h); return { box: [...this.propCorners(r), { x: w[0], y: w[1], z: w[2] }], label: `${r.label} and ${this.boneName(r.carrier).toLowerCase()}` }; }
    }
    return null;
  }

  selectProp(ref: string | null, frame = false) {
    const r = this.rigOf(ref);
    if (!r) {
      if (editStore.get().prop) { this.insideAt = null; this.publish({ prop: null, propInside: null }); setSelection(null); this.statusIdle(); }
      return;
    }
    this.publish({ prop: r.ref, selected: null, joint: null });
    setSelection({ label: r.label, frame: (step) => this.propFraming(r.ref, step) });
    const c = this.propPivotWorld(r);
    if (frame) { const f = this.propFraming(r.ref, 0); if (f) cam().frameBounds(f.box, { label: f.label }); }
    else if (c) cam().setPivot({ x: c[0], y: c[1], z: c[2] }, { history: false });
    this.updateInside();
    this.statusProp();
  }
  setPropPivot(pivot: 'carrier' | 'centre') { this.publish({ propPivot: pivot }); }

  // a prop's place, into the history's state and the card (the page shows it at once)
  private putPlace(ref: string, place: Place | null) {
    if (this.session) this.session.places[ref] = place ? JSON.parse(JSON.stringify(place)) : null;
    this.host.setPropPlace(ref, place);
    this.propBvh = null;
    this.dirty = true;
  }
  // the card's places shown again after an undo or a redo
  private syncPlaces() {
    const S = this.session;
    if (!S) return;
    for (const r of this.rigs()) {
      if (!(r.ref in S.places)) continue;
      const want = S.places[r.ref] ?? null;
      if (JSON.stringify(want) !== JSON.stringify(r.place ?? null)) { this.host.setPropPlace(r.ref, want); this.propBvh = null; }
    }
  }
  // the place after a rigid move W of the world (row-major) of the prop as it stood with the place `from`
  private placeAfter(r: PropRig, from: Place | null, W: M4): Place | null {
    const S = this.host.skinMatrix(r.carrier), head = this.headPage(r.carrier);
    if (!S || !head) return from;
    const H = this.holderM(), p = movedInPose(from, r.carrier, S, mul4(mul4(this.inv4(H), W), H), head, this.host.frame());
    return isRest(p, r.made) ? null : p;
  }
  private propEdit(label: string, fn: (r: PropRig) => Place | null | undefined): boolean {
    const r = this.rigOf(editStore.get().prop), S = this.session;
    if (!r || !S) return false;
    const next = fn(r);
    if (next === undefined) return false;
    this.record(label);
    this.putPlace(r.ref, next);
    this.publish();
    this.updateInside();
    return true;
  }
  // the world's length of a millimetre of the card (the page draws the figure scaled)
  private unit(): number { return this.host.frame().scale; }
  // Held by: the prop goes to another bone (or to the root), keeping its place in the world in the pose shown
  setHeldBy(bone: string): boolean {
    const F = this.host.frame();
    return this.propEdit(`Held by: ${this.boneName(bone).toLowerCase()}`, (r) => {
      const sOld = this.host.skinMatrix(r.carrier), sNew = this.host.skinMatrix(bone), hOld = this.headPage(r.carrier), hNew = this.headPage(bone);
      if (!sOld || !sNew || !hOld || !hNew || bone === r.carrier) return undefined;
      const p = rehold(r.place, sOld, sNew, hOld, hNew, bone, F);
      return isRest(p, r.made) ? null : p;
    });
  }
  // back to the starter's placement, on the bone it was made on
  resetProp(): boolean { return this.propEdit('Put the prop back', (r) => (r.place ? null : undefined)); }
  clearPropMove(): boolean {
    return this.propEdit('Clear the prop\'s move', (r) => { if (!r.place) return undefined; const p = { ...r.place, translation: [0, 0, 0] }; return isRest(p, r.made) ? null : p; });
  }
  clearPropTurn(): boolean {
    return this.propEdit('Clear the prop\'s turn', (r) => { if (!r.place) return undefined; const p = { ...r.place, rotation: [0, 0, 0, 1] }; return isRest(p, r.made) ? null : p; });
  }
  // the panel's typed values: the move in millimetres and the turn in degrees about the page's axes
  setPropNumbers(mm: number[], deg: number[]): boolean {
    const F = this.host.frame();
    return this.propEdit('Type the prop\'s place', (r) => { const p = placeFromNumbers(r.carrier, mm, deg, F); return isRest(p, r.made) ? null : p; });
  }
  // the arrows (across the view) and Page Up and Down (along the line of sight), 1 mm or 0.2 mm with Shift
  nudgeProp(dx: number, dy: number, dz: number, mm: number): boolean {
    const r = this.rigOf(editStore.get().prop);
    if (!r) return false;
    const e = this.host.camera.matrixWorld.elements, k = mm / 1000 * this.unit();
    const v = [0, 1, 2].map((i) => (e[i] * dx + e[4 + i] * dy - e[8 + i] * dz) * k);
    return this.propEdit('Nudge the prop', (q) => this.placeAfter(q, q.place, translation(v)));
  }

  // ---- drags
  private propAxes(r: PropRig): number[][] {
    const T = this.propWorld(r), head = this.headPage(r.carrier);
    if (!T || !head) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const A = mul4(T, this.pageMoveOf(r, head)), R = rot4(A);
    return [0, 1, 2].map((k) => { const a = [R[k], R[3 + k], R[6 + k]], l = Math.hypot(a[0], a[1], a[2]) || 1; return a.map((x) => x / l); });
  }
  // the place's move in the page's frame (its turn gives the prop's own axes)
  private pageMoveOf(r: PropRig, head: number[]): M4 { return pageMove(r.place, head, this.host.frame()); }
  private beginPropDrag(ref: string, part: HandlePart, x: number, y: number): boolean {
    const r = this.rigOf(ref), S = this.session;
    if (!r || !S || this.propDrag) return false;
    const pivot = this.propPivotWorld(r);
    if (!pivot) return false;
    this.propDrag = { ref, part, start: r.place ? JSON.parse(JSON.stringify(r.place)) : null, before: S.snapshot(), pivot, x0: x, y0: y, ex: 0, ey: 0, sweep: 0,
      axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], moved: false, label: r.label };
    return true;
  }
  // the move of the drag so far as a rigid move of the world, and its text for the status line
  private propMove(D: NonNullable<PoseEditor['propDrag']>, snap: boolean, typed: number | null = null, axis: number[] | null = null): { W: M4; text: string } {
    const c = this.v3(D.pivot), sc = cam().toScreen(c), u = this.unit();
    const part = D.part;
    if (part.kind === 'free' && !axis) {
      const n = this.host.camera.getWorldDirection(new this.host.THREE.Vector3()), N = [n.x, n.y, n.z];
      const ray = (x: number, y: number) => { const q = cam().rayAt(x, y); return onPlane([q.origin.x, q.origin.y, q.origin.z], [q.direction.x, q.direction.y, q.direction.z], D.pivot, N); };
      const a = ray(D.x0, D.y0), b = ray(D.x0 + D.ex, D.y0 + D.ey);
      let v = a && b ? [b[0] - a[0], b[1] - a[1], b[2] - a[2]] : [0, 0, 0];
      if (snap) v = v.map((x) => Math.round(x / u * 1000) / 1000 * u);
      return { W: translation(v), text: `${(Math.hypot(v[0], v[1], v[2]) / u * 1000).toFixed(1)} mm` };
    }
    if (part.kind === 'axis' || axis) {
      const a = axis || D.axes[(part as any).k];
      const q = cam().toScreen({ x: D.pivot[0] + a[0], y: D.pivot[1] + a[1], z: D.pivot[2] + a[2] });
      let m = typed !== null ? typed / 1000 * u : alongAxis(D.ex, D.ey, q.x - sc.x, q.y - sc.y);
      if (snap && typed === null) m = Math.round(m / u * 1000) / 1000 * u;
      return { W: translation(a.map((x) => x * m)), text: `${(m / u * 1000).toFixed(1)} mm` };
    }
    // a turn: round one axis (a ring) or round the line of sight (the outer ring, a drag on the prop with the Turn tool)
    const cp = this.host.camera.position, toCam = [cp.x - D.pivot[0], cp.y - D.pivot[1], cp.z - D.pivot[2]], l = Math.hypot(toCam[0], toCam[1], toCam[2]) || 1;
    const ax = part.kind === 'ring' ? D.axes[part.k] : toCam.map((x) => x / l);
    const facing = ax[0] * toCam[0] + ax[1] * toCam[1] + ax[2] * toCam[2] > 0 ? -1 : 1;
    let deg = typed !== null ? typed : facing * D.sweep * 180 / Math.PI;
    if (snap && typed === null) deg = Math.round(deg / 5) * 5;
    return { W: turnAbout(D.pivot, ax, deg * Math.PI / 180), text: `${Math.round(deg * 10) / 10}°` };
  }
  private propDragTo(dx: number, dy: number, snap: boolean, typed: number | null = null, axis: number[] | null = null): string {
    const D = this.propDrag;
    if (!D) return '';
    const sc = cam().toScreen(this.v3(D.pivot));
    D.sweep += sweep(sc.x, sc.y, D.x0 + D.ex, D.y0 + D.ey, D.x0 + D.ex + dx, D.y0 + D.ey + dy);
    D.ex += dx; D.ey += dy;
    const r = this.rigOf(D.ref);
    if (!r) return '';
    const { W, text } = this.propMove(D, snap, typed, axis);
    this.putPlace(D.ref, this.placeAfter(r, D.start, W));
    D.moved = true;
    const now = performance.now();
    if (now - this.insideTime > 120) this.updateInside(true);
    const inside = editStore.get().propInside;
    setStatus('pose-edit-drag', [{ text: `${D.label} · ${text}${inside && inside.mm > 0.5 ? ` · skin ${inside.mm.toFixed(1)} mm inside` : ''}` }, { keys: 'Shift', text: 'fine' },
      { keys: 'Ctrl', text: D.part.kind === 'ring' || D.part.kind === 'view' ? 'snap to 5°' : 'snap to 1 mm' }, { keys: 'Esc', text: 'put it back' }], { priority: PRIORITY.drag });
    return text;
  }
  private endPropDrag() {
    const D = this.propDrag, S = this.session;
    this.propDrag = null;
    clearStatus('pose-edit-drag');
    if (!D || !S) return;
    if (D.moved) {
      S.undoStack.push({ label: D.part.kind === 'ring' || D.part.kind === 'view' ? `Turn the ${D.label}` : `Move the ${D.label}`, snap: D.before });
      S.redoStack.length = 0;
    }
    this.publish();
    this.updateInside();
    this.statusProp();
  }
  private cancelPropDrag() {
    const D = this.propDrag, S = this.session;
    this.propDrag = null;
    clearStatus('pose-edit-drag');
    if (!D || !S) return;
    if (D.moved) S.restore(D.before);
    this.publish();
    this.updateInside();
  }
  // G: the prop follows the mouse until a click or Enter (X, Y or Z: along that axis of the view, again: of the prop
  // itself; a typed number: that many millimetres along it, along X without an axis); R: it turns round the line of
  // sight through its pivot (X, Y or Z: round that axis; a typed number: degrees). Esc or a right-click puts it back.
  private propModal(kind: 'grab' | 'turn'): boolean {
    const r = this.rigOf(editStore.get().prop);
    if (!r || this.propDrag) return false;
    const pivot = this.propPivotWorld(r);
    if (!pivot) return false;
    const sc = cam().toScreen(this.v3(pivot));
    const at = this.pointer && Math.hypot(this.pointer.x - sc.x, this.pointer.y - sc.y) > 12 ? { ...this.pointer } : { x: sc.x + 80, y: sc.y };
    if (!this.beginPropDrag(r.ref, kind === 'grab' ? { kind: 'free' } : { kind: 'view' }, at.x, at.y)) return false;
    const own = this.propAxes(r), world = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    let lx = 0, ly = 0;
    startModal({
      kind, label: r.label,
      update: (v) => {
        const D = this.propDrag;
        if (!D) return;
        const k = v.axis === 'free' ? -1 : { x: 0, y: 1, z: 2 }[v.axis];
        const ax = k >= 0 ? (v.local ? own : world)[k] : kind === 'grab' && v.typed !== null ? world[0] : null;
        if (kind === 'turn') D.part = ax ? { kind: 'ring', k: 0 } : { kind: 'view' };
        if (ax && kind === 'turn') D.axes = [ax, ax, ax];
        const text = this.propDragTo(v.dx - lx, v.dy - ly, v.snap, v.typed, kind === 'grab' ? ax : null);
        lx = v.dx; ly = v.dy;
        return text;
      },
      confirm: () => this.endPropDrag(),
      cancel: () => this.cancelPropDrag(),
    }, at);
    return true;
  }
  startPropGrab(): boolean { return this.propModal('grab'); }
  startPropTurn(): boolean { return this.propModal('turn'); }

  // ---- the skin inside the selected prop (contact shading): its deepest point and the part of the body there
  private cardSdf(r: PropRig): ((X: Float64Array) => Float64Array) | null {
    const T = this.propWorld(r);
    if (!T) return null;
    const Ti = this.inv4(T), c = this.restOf(r), u = this.unit(), w = [0, 0, 0];
    return (X: Float64Array) => {
      const out = new Float64Array(X.length / 3);
      for (let i = 0; i < out.length; i++) {
        this.frames.toWorld([X[i * 3], X[i * 3 + 1], X[i * 3 + 2]], w);
        const p = apply4(Ti, w), q = c.bvh.closest(p);
        if (!q) { out[i] = Infinity; continue; }
        // inside or out by the parity of crossings along three rays, by majority, within REACH of the prop: the side of
        // the closest face misleads further away (on David's sword it called the left shin 31 cm inside the blade), and
        // no skin lies that deep inside a prop
        let inside = false;
        if (q.distance < INSIDE_REACH * u) { let odd = 0; for (const d of PARITY_RAYS) if (c.bvh.rayAll(p, d).length % 2 === 1) odd++; inside = odd >= 2; }
        out[i] = (inside ? -q.distance : q.distance) / u;
      }
      return out;
    };
  }
  private updateInside(quiet = false) {
    this.insideTime = typeof performance !== 'undefined' ? performance.now() : 0;
    const st = editStore.get(), r = this.rigOf(st.prop), S = this.session;
    const none = () => { this.insideAt = null; if (st.propInside) { if (quiet) editStore.set({ propInside: null }); else this.publish({ propInside: null }); } };
    if (!r || !S) { none(); return; }
    const sdf = this.cardSdf(r), T = this.propWorld(r);
    if (!sdf || !T) { none(); return; }
    // the skin within the prop's box (the card frame), then its signed distance
    const corners = this.propCorners(r).map((q) => this.frames.fromWorld([q.x, q.y, q.z]));
    const lo = [0, 1, 2].map((k) => Math.min(...corners.map((q) => q[k])) - 0.005), hi = [0, 1, 2].map((k) => Math.max(...corners.map((q) => q[k])) + 0.005);
    const body = this.skinnedBody(), V = body.vertices(S.rotations), rows: number[] = [];
    for (let v = 0; v < body.n; v++) { const x = V[v * 3], y = V[v * 3 + 1], z = V[v * 3 + 2]; if (x >= lo[0] && x <= hi[0] && y >= lo[1] && y <= hi[1] && z >= lo[2] && z <= hi[2]) rows.push(v); }
    if (!rows.length) { none(); return; }
    const X = new Float64Array(rows.length * 3);
    rows.forEach((v, i) => { X[i * 3] = V[v * 3]; X[i * 3 + 1] = V[v * 3 + 1]; X[i * 3 + 2] = V[v * 3 + 2]; });
    const d = sdf(X);
    let best = -1, bd = 0;
    for (let i = 0; i < d.length; i++) if (d[i] < bd) { bd = d[i]; best = i; }
    if (best < 0) { none(); return; }
    this.insideAt = this.frames.toWorld([X[best * 3], X[best * 3 + 1], X[best * 3 + 2]]);
    const res = { mm: Math.round(-bd * 10000) / 10, part: jointName(S.rig.labels[body.strongest[rows[best]]]).name };
    if (quiet) editStore.set({ propInside: res }); else this.publish({ propInside: res });
    this.dirty = true;
  }
  // a pixel of the view where the pointer's ray meets the prop first, inside the free space between the panels (for the
  // page tests), or null
  propPixel(ref: string): [number, number] | null {
    const r = this.rigOf(ref), T = r ? this.propWorld(r) : null;
    if (!r || !T) return null;
    const rect = this.host.canvas.getBoundingClientRect(), V = r.vertices, I = r.triangles;
    for (let t = 0; t + 2 < I.length; t += 3 * 5) {
      const c = [0, 1, 2].map((k) => (V[I[t] * 3 + k] + V[I[t + 1] * 3 + k] + V[I[t + 2] * 3 + k]) / 3), s = cam().toScreen(this.v3(apply4(T, c)));
      if (s.depth <= 0 || s.x < rect.left + 440 || s.x > rect.right - 440 || s.y < rect.top + 90 || s.y > rect.bottom - 90) continue;
      const hits = this.hitProps({ x: s.x, y: s.y, ray: cam().rayAt(s.x, s.y), radius: 16 } as PointerInfo);
      if (hits.length && hits[0].item === ref) return [s.x, s.y];
    }
    return null;
  }
  propReport() {
    const st = editStore.get(), r = this.rigOf(st.prop);
    return { prop: st.prop, place: r?.place ?? null, carrier: r?.carrier ?? null, inside: st.propInside, pivot: r ? this.propPivotWorld(r) : null,
      handles: this.propHandles ? { centre: this.propHandles.centre, arrows: this.propHandles.arrows, rings: this.propHandles.rings?.map((q) => q.filter((_, i) => i % 6 === 0)), outer: this.propHandles.outer } : null };
  }
  // Move it out: the prop moves to the nearest seat in the hand that holds it. A hand wraps the grip it holds, so a
  // push that clears the whole palm takes the prop out of the hand (on David's sword 58 mm, the fingers then 18 to 40 mm
  // off it). The search instead lowers the palm's deepest skin inside the prop plus MOVE_COST per millimetre moved, by
  // a pattern search (R. Hooke and T. A. Jeeves, "Direct search solution of numerical and statistical problems",
  // J. ACM, 1961) over moves along the prop's long axis and two ways across it (the second the way out of the skin
  // inside it, the depth times the direction to the prop's surface summed over that skin, as a physics engine
  // separates two bodies), in steps of 8, 4, 2 and 1 mm up to 40 mm in all; the fingers' skin is left to Grip, which
  // opens and closes them round the prop where it then stands.
  moveOut(): boolean {
    const st = editStore.get(), S = this.session, ref = st.grip?.prop || st.prop, r = this.rigOf(ref);
    const side = st.grip && st.grip.prop === ref ? st.grip.hand : r ? this.sideOf(r.carrier) : null;
    if (!r || !S || !side) { this.host.toast('Select a prop that a hand holds first.'); return false; }
    const sdf = this.cardSdf(r);
    if (!sdf) return false;
    const body = this.skinnedBody(), bones = (re: RegExp) => S.rig.labels.map((l, i) => [l, i] as [string, number]).filter(([l]) => re.test(l)).map(([, i]) => i);
    const rows = body.rowsOf(bones(new RegExp(`^(wrist|metacarpal|finger)[0-9-]*\\.${side}$`))), palm = new Set(Array.from(body.rowsOf(bones(new RegExp(`^(wrist|metacarpal)[0-9-]*\\.${side}$`)))));
    const V = body.vertices(S.rotations, rows), d = sdf(V);
    const c = this.restOf(r), T = this.propWorld(r)!, Ti = this.inv4(T), dir = [0, 0, 0], w = [0, 0, 0];
    for (let i = 0; i < d.length; i++) {
      if (d[i] >= 0) continue;
      const x = [V[i * 3], V[i * 3 + 1], V[i * 3 + 2]], q = c.bvh.closest(apply4(Ti, this.frames.toWorld(x, w)))!, s = this.frames.fromWorld(apply4(T, q.point));
      const l = Math.hypot(s[0] - x[0], s[1] - x[1], s[2] - x[2]) || 1;
      for (let k = 0; k < 3; k++) dir[k] += -d[i] * (s[k] - x[k]) / l;
    }
    const n = Math.hypot(dir[0], dir[1], dir[2]);
    if (!(n > 0)) { this.lastMoveOut = null; this.host.toast(`No skin of the hand is inside the ${r.label}.`); return false; }
    const unit = (v: number[]) => { const l = Math.hypot(v[0], v[1], v[2]); return l > 1e-9 ? v.map((x) => x / l) : null; };
    const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    // the prop's long axis: the first principal direction of its rest vertices (power iteration), in the card frame
    const P = r.vertices, m = c.centre, C = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let i = 0; i < P.length; i += 3) { const q = [P[i] - m[0], P[i + 1] - m[1], P[i + 2] - m[2]]; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) C[a * 3 + b] += q[a] * q[b]; }
    let ax = [1, 0.3, 0.1];
    for (let it = 0; it < 50; it++) ax = unit([0, 1, 2].map((a) => C[a * 3] * ax[0] + C[a * 3 + 1] * ax[1] + C[a * 3 + 2] * ax[2])) || ax;
    const R = rot4(T), axw = [0, 1, 2].map((k) => R[k * 3] * ax[0] + R[k * 3 + 1] * ax[1] + R[k * 3 + 2] * ax[2]), axis = this.frames.dirToCard(axw) as number[];
    // the hand must go along `out`; the prop goes the other way
    const out = dir.map((x) => x / n), across = unit(out.map((x, k) => x - axis[k] * (out[0] * axis[0] + out[1] * axis[1] + out[2] * axis[2]))) || unit(cross(axis, [0, 0, 1]))!, third = cross(axis, across);
    const pi: number[] = [];
    for (let i = 0; i < rows.length; i++) if (palm.has(rows[i])) pi.push(i);
    const VP = new Float64Array(pi.length * 3);
    pi.forEach((i, j) => { VP[j * 3] = V[i * 3]; VP[j * 3 + 1] = V[i * 3 + 1]; VP[j * 3 + 2] = V[i * 3 + 2]; });
    // the palm's deepest skin inside the prop (mm) with the prop moved by -x (metres, the card frame)
    const palmDepth = (x: number[]) => {
      const Y = VP.slice();
      for (let i = 0; i < Y.length; i += 3) { Y[i] += x[0]; Y[i + 1] += x[1]; Y[i + 2] += x[2]; }
      let worst = 0;
      for (const v of sdf(Y)) worst = Math.min(worst, v);
      return -worst * 1000;
    };
    const basis = [axis, across, third], trace: { way?: string; mm: number; palm_mm: number; blocked?: boolean }[] = [];
    const at = (c: number[]) => [0, 1, 2].map((k) => (c[0] * basis[0][k] + c[1] * basis[1][k] + c[2] * basis[2][k]) / 1000);
    const score = (c: number[]) => { const d = palmDepth(at(c)), mm = Math.hypot(c[0], c[1], c[2]); trace.push({ way: c.map((x) => x.toFixed(0)).join(' '), mm: Math.round(mm * 10) / 10, palm_mm: Math.round(d * 10) / 10 }); return { s: d + MOVE_COST * mm, d }; };
    const start = palmDepth([0, 0, 0]);
    let cur = [0, 0, 0], best = { s: start, d: start };
    for (const step of [8, 4, 2, 1]) {
      for (let improved = true, n = 0; improved && n < 20; n++) {
        improved = false;
        for (let a = 0; a < 3; a++) for (const sg of [1, -1]) {
          const c = cur.slice(); c[a] += sg * step;
          if (Math.hypot(c[0], c[1], c[2]) > MOVE_LIMIT) continue;
          const r2 = score(c);
          if (r2.s < best.s - 1e-6) { best = r2; cur = c; improved = true; }
        }
      }
    }
    const moved = Math.hypot(cur[0], cur[1], cur[2]), shift = at(cur);
    const shifted = (X: Float64Array) => { const Y = X.slice(); for (let i = 0; i < Y.length; i += 3) { Y[i] += shift[0]; Y[i + 1] += shift[1]; Y[i + 2] += shift[2]; } return sdf(Y); };
    const blocked = moved > 0 && Object.values(this.gripOf(S.rotations, side, shifted, 0).report).some((g) => g.blocked);
    this.lastMoveOut = { mm: Math.round(moved * 10) / 10, direction: shift.map((x) => (moved ? x / moved * 1000 : 0)), way: `axis ${cur[0]} mm, across ${cur[1]} mm and ${cur[2]} mm`,
      palm_mm: Math.round(best.d * 10) / 10, start_mm: Math.round(start * 10) / 10, blocked, trace };
    if (!moved || best.d >= start - 0.5) { this.host.toast(`No move of the ${r.label} within ${MOVE_LIMIT} mm seats it better in the hand.`); return false; }
    const pick = { mm: moved, u: shift.map((x) => x / (moved / 1000)), palm: best.d };
    // the prop moves the other way: -mm along u (the card frame) as a move of the world
    const wd = this.frames.dirToWorld(pick.u), dist = -pick.mm / 1000 * this.unit();
    this.record(`Move the ${r.label} out of the hand`);
    this.putPlace(r.ref, this.placeAfter(r, r.place, translation(wd.map((x) => x * dist))));
    this.host.toast(pick.palm <= 0.05 ? `Moved the ${r.label} ${pick.mm.toFixed(0)} mm: the palm is clear of it.` : `Moved the ${r.label} ${pick.mm.toFixed(0)} mm: the palm stays ${pick.palm.toFixed(1)} mm inside it, against ${start.toFixed(1)} mm before.`);
    this.publish();
    this.updateInside();
    if (st.grip && st.grip.prop === r.ref) { this.gripStart = null; this.setSqueeze(st.squeeze, false); }
    return true;
  }
  lastMoveOut: { mm: number; direction: number[]; way?: string; palm_mm?: number; start_mm?: number; blocked?: boolean; trace?: { way?: string; mm: number; palm_mm: number; blocked?: boolean }[] } | null = null;

  // the prop's handles over the view: the move handle (three arrows and a centre) with the Move tool, the turn rings (and
  // an outer ring) with the Turn tool, the prop's name and the bone that holds it, and the deepest skin inside it
  private drawPropHandles(f: Frame, st: EditState) {
    this.propHandles = null;
    const r = this.rigOf(st.prop);
    if (!r) return;
    const pivot = this.propPivotWorld(r);
    if (!pivot) return;
    const sc = cam().toScreen(this.v3(pivot));
    if (sc.depth <= 0) return;
    const H = this.host.canvas.clientHeight || 800, R0 = 70 * f.scale, lw = R0 * sc.depth * 2 * Math.tan((this.host.camera.fov * Math.PI / 180) / 2) / H;
    const COL = ['#d0553a', '#3fa66b', '#2f6fd6'], NAMES = ['X', 'Y', 'Z'], axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const strokes = f.strokes || (f.strokes = []), tags = f.tags || (f.tags = []);
    const toS = (p: number[]) => cam().toScreen({ x: p[0], y: p[1], z: p[2] });
    const path = (pts: [number, number][]) => 'M' + pts.map((p) => p.map((x) => x.toFixed(1)).join(' ')).join(' L');
    const hot = (key: string) => this.hoverProp === key || (this.propDrag && JSON.stringify(this.propDrag.part) === key);
    if (st.tool === 'move') {
      const arrows: [number, number][][] = [];
      axes.forEach((a, k) => {
        const e = toS(pivot.map((x, i) => x + a[i] * lw)), pts: [number, number][] = [[sc.x, sc.y], [e.x, e.y]];
        arrows.push(pts);
        const key = JSON.stringify({ kind: 'axis', k }), wd = hot(key) ? 3.4 : 2.2;
        strokes.push({ d: path(pts), colour: COL[k], width: wd, opacity: 0.95 });
        const dx = e.x - sc.x, dy = e.y - sc.y, l = Math.hypot(dx, dy) || 1, ux = dx / l, uy = dy / l, s = 9 * f.scale;
        strokes.push({ d: `M${(e.x - ux * s - uy * s * 0.5).toFixed(1)} ${(e.y - uy * s + ux * s * 0.5).toFixed(1)} L${e.x.toFixed(1)} ${e.y.toFixed(1)} L${(e.x - ux * s + uy * s * 0.5).toFixed(1)} ${(e.y - uy * s - ux * s * 0.5).toFixed(1)}`, colour: COL[k], width: wd, opacity: 0.95 });
        tags.push({ x: e.x + ux * 14, y: e.y + uy * 14, text: NAMES[k], colour: COL[k] });
      });
      let cd = '';
      for (let i = 0; i <= 32; i++) { const t = i / 32 * 2 * Math.PI, rr = 9 * f.scale; cd += (i ? ' L' : 'M') + (sc.x + rr * Math.cos(t)).toFixed(1) + ' ' + (sc.y + rr * Math.sin(t)).toFixed(1); }
      strokes.push({ d: cd, colour: '#d4ae52', width: hot(JSON.stringify({ kind: 'free' })) ? 3.4 : 2.2, opacity: 0.95 });
      this.propHandles = { centre: { x: sc.x, y: sc.y }, arrows, axes };
    } else if (st.tool === 'turn') {
      const rings: [number, number][][] = [];
      axes.forEach((a, k) => {
        const ref = Math.abs(a[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
        let u = [a[1] * ref[2] - a[2] * ref[1], a[2] * ref[0] - a[0] * ref[2], a[0] * ref[1] - a[1] * ref[0]];
        const lu = Math.hypot(u[0], u[1], u[2]); u = u.map((x) => x / lu);
        const v = [a[1] * u[2] - a[2] * u[1], a[2] * u[0] - a[0] * u[2], a[0] * u[1] - a[1] * u[0]], rr = lw * (1 - 0.1 * k);
        const pts: [number, number][] = [];
        for (let i = 0; i <= 72; i++) { const t = i / 72 * 2 * Math.PI, q = toS([0, 1, 2].map((j) => pivot[j] + rr * (Math.cos(t) * u[j] + Math.sin(t) * v[j]))); pts.push([q.x, q.y]); }
        rings.push(pts);
        strokes.push({ d: path(pts), colour: COL[k], width: hot(JSON.stringify({ kind: 'ring', k })) ? 3.2 : 1.8, opacity: 0.95 });
      });
      const outer = R0 + 22 * f.scale;
      let od = '';
      for (let i = 0; i <= 96; i++) { const t = i / 96 * 2 * Math.PI; od += (i ? ' L' : 'M') + (sc.x + outer * Math.cos(t)).toFixed(1) + ' ' + (sc.y + outer * Math.sin(t)).toFixed(1); }
      strokes.push({ d: od, colour: '#c9c2b7', width: hot(JSON.stringify({ kind: 'view' })) ? 2.6 : 1.4, dash: '4 4', opacity: 0.9 });
      this.propHandles = { centre: { x: sc.x, y: sc.y }, rings, outer, axes };
    } else {
      let cd = '';
      for (let i = 0; i <= 24; i++) { const t = i / 24 * 2 * Math.PI, rr = 5 * f.scale; cd += (i ? ' L' : 'M') + (sc.x + rr * Math.cos(t)).toFixed(1) + ' ' + (sc.y + rr * Math.sin(t)).toFixed(1); }
      strokes.push({ d: cd, colour: '#d4ae52', width: 2, opacity: 0.95 });
    }
    f.labels.push({ x: sc.x - 12, y: sc.y - R0 - 30 * f.scale, title: r.label, line: `Held by: ${this.boneName(r.carrier).toLowerCase()}`, side: 'left' });
    const inside = st.propInside;
    if (inside && inside.mm > 0.5 && this.insideAt) {
      const s = toS(this.insideAt);
      if (s.depth > 0) (f.alerts || (f.alerts = [])).push({ x: s.x, y: s.y, title: `Skin ${inside.mm.toFixed(1)} mm inside the ${r.label}`, line: inside.part });
    }
  }

  private registerPropPickers() {
    const on = () => { const st = editStore.get(); return st.on && !st.pinDraft && !(st.section === 'points' && st.tool === 'move' && !!st.selected); };
    const clear = () => { if (editStore.get().prop) this.selectProp(null); };
    const drag = {
      onDrag: (_h: PickHit, _p: PointerInfo, d: DragInfo) => { if (this.propDrag) this.propDragTo(d.ddx * d.scale, d.ddy * d.scale, d.snap); },
      onUp: () => this.endPropDrag(),
      onCancel: () => this.cancelPropDrag(),
    };
    this.offs.push(registerPicker({
      id: 'pose-prop-handles', priority: 25, enabled: () => on() && !!editStore.get().prop && editStore.get().tool !== 'select' && !!this.propHandles,
      hit: (p) => { const part = this.propHandles ? hitHandles(p, this.propHandles, 9 * settings.targetScale()) : null;
        return part ? { item: part, distPx: 0, label: part.kind === 'free' ? 'Move along the view' : part.kind === 'view' ? 'Turn round the line of sight' : `${part.kind === 'axis' ? 'Move along' : 'Turn round'} ${'XYZ'[part.k]}` } : null; },
      onDown: (h, p) => { const ref = editStore.get().prop; if (!ref || !this.beginPropDrag(ref, h.item as HandlePart, p.x, p.y)) return false; if (this.propHandles) this.propDrag!.axes = this.propHandles.axes; return true; },
      ...drag,
      onClick: () => this.endPropDrag(),
      onHover: (h) => { this.hoverProp = h ? JSON.stringify(h.item) : null; this.dirty = true; },
      cursor: 'grab',
    }));
    this.offs.push(registerPicker({
      id: 'pose-props', priority: 3, enabled: () => on() && editStore.get().tool === 'select', hit: (p) => this.hitProps(p),
      onClick: (h) => this.selectProp(h.item as string), onBackground: clear, cursor: 'pointer',
    }));
    this.offs.push(registerPicker({
      id: 'pose-props-drag', priority: 3, enabled: () => on() && editStore.get().tool !== 'select', hit: (p) => this.hitProps(p),
      onDown: (h, p) => {
        const st = editStore.get(), ref = h.item as string;
        // the Move tool drags any prop; the Turn tool turns the selected prop only (a click selects another)
        if (st.tool === 'turn' && st.prop !== ref) return true;
        if (st.prop !== ref) this.selectProp(ref);
        this.beginPropDrag(ref, st.tool === 'turn' ? { kind: 'view' } : { kind: 'free' }, p.x, p.y);
        return true;
      },
      ...drag,
      onClick: (h) => { this.endPropDrag(); this.selectProp(h.item as string); },
      onBackground: clear, cursor: 'grab',
    }));
  }
  statusProp() {
    const st = editStore.get(), r = this.rigOf(st.prop);
    if (!r) { this.statusIdle(); return; }
    setStatus('pose-edit', [{ text: `${r.label} selected` }, { keys: 'Drag', text: st.tool === 'move' ? 'move it' : st.tool === 'turn' ? 'turn it' : 'turn the view (W: move it)' },
      { keys: 'G', text: 'grab' }, { keys: 'R', text: 'turn' }, { keys: 'Arrows', text: 'nudge 1 mm', shift: { keys: 'Arrows', text: 'nudge 0.2 mm' } },
      { keys: 'Alt+G', text: 'clear the move' }, { keys: 'Alt+R', text: 'clear the turn' }, { keys: 'F', text: 'frame' }], { priority: PRIORITY.tool });
  }

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
    this.savedSig = this.signature();
  }

  // ---------------------------------------------------------------- unsaved edits (DESIGN-0014, "Browser keys")
  // the points, the pins and the pose as a string, to tell an unsaved edit from the state last opened or saved (an
  // edit undone back to that state is no edit)
  private savedSig: string | null = null;
  private signature(): string {
    const S = this.session;
    if (!S) return '';
    return JSON.stringify([S.cardPoints(), S.cardPins(), S.rotations.map((r) => r.map((x) => Math.round(x * 1e6))), S.places]);
  }
  unsavedEdits(): boolean { return !!this.session && this.savedSig !== null && this.signature() !== this.savedSig; }
  // the prompt before leaving the page: on unless View › Controls turns it off or the URL carries noprompt (the page
  // tests leave pages with edits)
  promptOn(): boolean {
    let flag = false;
    try { flag = new URLSearchParams(location.search).has('noprompt'); } catch { /* no location: on */ }
    return settings.get().warnUnsaved && !flag;
  }

  // ---------------------------------------------------------------- joints (phase 2)
  private jointsCache: { rig: any; joints: Joints; hands: HandControls } | null = null;
  joints(): Joints {
    const S = this.session!;
    if (!this.jointsCache || this.jointsCache.rig !== S.rig) { const j = new Joints(S.rig); this.jointsCache = { rig: S.rig, joints: j, hands: new HandControls(j) }; }
    return this.jointsCache.joints;
  }
  hands(): HandControls { this.joints(); return this.jointsCache!.hands; }

  selectJoint(label: string | null) {
    this.publish({ joint: label, selected: null, ...(label ? { prop: null } : {}) });
    if (!label) { setSelection(null); this.statusIdle(); return; }
    const J = this.joints(), b = J.bone(label);
    if (b === undefined) return;
    const n = jointName(label);
    setSelection({ label: n.name, bones: [label], frame: (step) => this.jointFraming(b, step) });
    const c = this.jointCentre(b);
    if (c) { const w = this.frames.toWorld(c); cam().setPivot({ x: w[0], y: w[1], z: w[2] }, { history: false }); }
    setStatus('pose-edit', [{ text: n.name }, { keys: 'Drag a ring', text: 'turn the joint', shift: { keys: 'Drag a ring', text: 'turn finely' } }, { keys: 'Ctrl', text: 'snap to 5°' },
      { keys: 'R', text: 'turn by keys' }, { keys: 'Alt+R', text: 'reset' }], { priority: PRIORITY.tool });
  }
  // F's steps from a joint: the bone, its limb, the whole figure
  private jointFraming(b: number, step: number) {
    const S = this.session!, J = this.joints();
    const list = step === 0 ? [b, ...S.rig.labels.map((_, i) => i).filter((i) => S.rig.parents[i] === b)] : step === 1 ? J.limbOf(b) : null;
    if (!list) return null;
    const pts = list.map((i) => this.jointCentre(i)).filter(Boolean).map((c) => { const w = this.frames.toWorld(c!); return { x: w[0], y: w[1], z: w[2] }; });
    if (pts.length === 1) { const p = pts[0], r = 0.06; return { box: { min: { x: p.x - r, y: p.y - r, z: p.z - r }, max: { x: p.x + r, y: p.y + r, z: p.z + r } }, label: jointName(S.rig.labels[b]).name }; }
    return { box: pts, label: jointName(S.rig.labels[b]).name };
  }
  // a bone's posed head (card frame)
  jointCentre(b: number): V3 | null {
    const S = this.session;
    if (!S) return null;
    const K = transforms(S.rig, S.rotations), h = S.rig.heads[b], F = S.rig.frame;
    const x = [0, 1, 2].map((r) => K.R[b * 9 + r * 3] * h[0] + K.R[b * 9 + r * 3 + 1] * h[1] + K.R[b * 9 + r * 3 + 2] * h[2] + K.t[b * 3 + r]);
    return [0, 1, 2].map((r) => F[r][0] * x[0] + F[r][1] * x[1] + F[r][2] * x[2] + F[r][3]) as V3;
  }
  // a joint's axes in the card frame (the columns of the parent's posed turn times the joint frame), bend, side, twist
  private jointAxes(b: number): V3[] {
    const S = this.session!, J = this.joints(), f = J.frames.get(b)!, K = transforms(S.rig, S.rotations), p = S.rig.parents[b], F = S.rig.frame;
    return [0, 1, 2].map((k) => {
      const a = [f.axes[0][k] * f.sign[k], f.axes[1][k] * f.sign[k], f.axes[2][k] * f.sign[k]];
      const w = [0, 1, 2].map((r) => K.R[p * 9 + r * 3] * a[0] + K.R[p * 9 + r * 3 + 1] * a[1] + K.R[p * 9 + r * 3 + 2] * a[2]);
      return [0, 1, 2].map((r) => F[r][0] * w[0] + F[r][1] * w[1] + F[r][2] * w[2]) as V3;
    });
  }

  private change(label: string, fn: (R: number[][]) => void, record = true) {
    const S = this.session;
    if (!S) return;
    if (record) this.record(label);
    fn(S.rotations);
    this.poseChanged();
    this.publish();
  }

  // a value of the selected joint (degrees): Bend 0, Side-Side 1, Twist 2
  setJointValue(axis: number, deg: number, record = true) {
    const st = editStore.get(), J = this.session ? this.joints() : null, b = st.joint && J ? J.bone(st.joint) : undefined;
    if (b === undefined || !J) return;
    this.change(AXES[axis].label, (R) => J.set(R, b, axis, deg, st.limits), record);
  }
  // Alt+R: the joint back to its reference turn (Shift: the whole limb)
  resetJoint(whole = false): boolean {
    const st = editStore.get(), J = this.session ? this.joints() : null, b = st.joint && J ? J.bone(st.joint) : undefined;
    if (b === undefined || !J) return false;
    this.change(whole ? 'Reset the limb' : 'Reset the joint', (R) => { for (const x of whole ? J.limbOf(b) : [b]) J.reset(R, x); });
    return true;
  }
  setLimits(on: boolean) { this.publish({ limits: on }); }

  // a free turn of the selected joint about a card-frame axis by an angle (radians), kept within the ranges
  private turnJoint(axisCard: V3, angle: number) {
    const st = editStore.get(), S = this.session!, J = this.joints(), b = J.bone(st.joint!)!, f = J.frames.get(b)!;
    const K = transforms(S.rig, S.rotations), p = S.rig.parents[b], F = S.rig.frame;
    // the axis in the parent's frame of the turns: Kp^T Rr^T a
    const ar = [0, 1, 2].map((c) => F[0][c] * axisCard[0] + F[1][c] * axisCard[1] + F[2][c] * axisCard[2]);
    const ap = [0, 1, 2].map((c) => K.R[p * 9 + c] * ar[0] + K.R[p * 9 + 3 + c] * ar[1] + K.R[p * 9 + 6 + c] * ar[2]);
    const Q = expRotation(ap.map((x) => x * angle));
    let D = mul3(Q, S.rotations[b]);
    if (st.limits) {
      const c = coordinatesOf(f, D);
      for (let k = 0; k < 3; k++) c[k] = f.dofs[k] || k === 2 ? Math.min(Math.max(c[k], f.lo[k]), f.hi[k]) : c[k];
      D = rotationOf(f, c);
    }
    S.rotations[b] = D;
  }

  // ---------------------------------------------------------------- hands (phase 2)
  setHandSide(side: 'L' | 'R') { this.publish({ handSide: side }); }
  setHand(kind: 'fist' | 'spread' | 'thumb' | 'curl' | 'fspread', k: number, v: number, record = true) {
    const H = this.hands(), side = editStore.get().handSide;
    this.change(kind === 'curl' ? `Curl of the ${FINGERS[k]}` : kind === 'fspread' ? `Spread of the ${FINGERS[k]}` : kind[0].toUpperCase() + kind.slice(1), (R) => {
      if (kind === 'fist') H.setFist(R, side, v); else if (kind === 'spread') H.setHandSpread(R, side, v); else if (kind === 'thumb') H.setThumb(R, side, v);
      else if (kind === 'curl') H.setCurl(R, side, k, v); else H.setSpread(R, side, k, v);
    }, record);
    this.publish({ preset: null });
  }
  applyShape(shape: HandShape, label: string, id: string | null = null, record = true) {
    const H = this.hands(), side = editStore.get().handSide;
    this.change(label, (R) => H.apply(R, side, shape), record);
    this.publish({ preset: id });
  }
  applyPreset(id: string) {
    const p = [...PRESETS, ...editStore.get().userPresets].find((x) => x.id === id);
    if (p) this.applyShape(p.shape, `Hand: ${p.label}`, id);
  }
  relax() { this.applyShape(RELAXED, 'Relax', 'relaxed'); }
  // the gesture pad: the hand blended between four corner shapes at (u, v); a drag is one step of the history
  blendPad(corners: HandShape[], u: number, v: number, first: boolean) {
    this.applyShape(HandControls.blend(corners, u, v), 'Gesture pad', null, first);
  }
  fingerSpread(k: number): number { const S = this.session; return S ? this.hands().spread(S.rotations, editStore.get().handSide, k) : 0; }
  saveHandPreset(label: string) {
    const st = editStore.get(), shape = { curl: st.handShape.curl.slice(), spread: st.handShape.spread };
    const list = [...st.userPresets, { id: 'user-' + Date.now().toString(36), label, shape }];
    savePresets(list);
    this.publish({ userPresets: list });
  }

  // M: the whole pose mirrored
  mirror(): boolean {
    const S = this.session;
    if (!S) return false;
    this.change('Mirror the pose', (R) => { const M = mirrorPose(S.rig, R, this.host.root()); for (let i = 0; i < R.length; i++) R[i] = M[i]; });
    return true;
  }
  // the selected hand's pose onto the other hand
  copyHand(): boolean {
    const S = this.session;
    if (!S) return false;
    const side = editStore.get().handSide;
    this.change(`Copy to the ${side === 'R' ? 'left' : 'right'} hand`, (R) => copyMirrored(S.rig, R, this.hands().handBones(side)));
    return true;
  }
  // Ctrl+C, Ctrl+V and Ctrl+Shift+V: the selected limb's or hand's pose
  private clipboard: { bones: number[]; rotations: number[][]; label: string } | null = null;
  private selectedBones(): { bones: number[]; label: string } | null {
    const st = editStore.get(), S = this.session;
    if (!S) return null;
    if (st.section === 'hands') return { bones: this.hands().handBones(st.handSide), label: st.handSide === 'R' ? 'the right hand' : 'the left hand' };
    if (st.joint) { const J = this.joints(), b = J.bone(st.joint)!; return { bones: J.limbOf(b), label: jointName(st.joint).name }; }
    const p = st.selected ? POINT_BY_NAME.get(st.selected) : null;
    if (p && p.group !== 'body') return { bones: this.hands().handBones(p.side as 'L' | 'R'), label: p.side === 'R' ? 'the right hand' : 'the left hand' };
    // a hand selected from its cluster or the body map
    if (st.hand) return { bones: this.hands().handBones(st.hand), label: st.hand === 'R' ? 'the right hand' : 'the left hand' };
    return null;
  }
  copyPose(): boolean {
    const sel = this.selectedBones(), S = this.session;
    if (!sel || !S) return false;
    this.clipboard = { bones: sel.bones, rotations: sel.bones.map((b) => S.rotations[b].slice()), label: sel.label };
    this.publish({ clip: sel.label });
    this.host.toast(`Copied the pose of ${sel.label}`);
    return true;
  }
  pastePose(mirrored = false): boolean {
    const c = this.clipboard, S = this.session;
    if (!c || !S) return false;
    this.change(mirrored ? 'Paste mirrored' : 'Paste', (R) => {
      if (!mirrored) { c.bones.forEach((b, i) => { R[b] = c.rotations[i].slice(); }); return; }
      // onto the other side: the copied turns reflected onto the mirror bones, the copied bones left as they are
      const tmp = R.map((r) => r.slice());
      c.bones.forEach((b, i) => { tmp[b] = c.rotations[i].slice(); });
      for (const o of copyMirrored(S.rig, tmp, c.bones)) R[o] = tmp[o];
    });
    return true;
  }

  // R: the selected joint turns with the mouse about the view's axis until a click or Enter (Blender's R: the angle
  // the pointer sweeps round the joint's centre on the screen); X, Y or Z takes the bend, the side or the twist ring
  // instead, a typed number sets the angle in degrees and Ctrl snaps to 5 degrees. The input is the studio's modal
  // turn (studio/modal.ts); Esc or a right-click puts the joint back
  private turning: { before: any } | null = null;
  startTurn(): boolean {
    const st = editStore.get(), S = this.session;
    if (!S || !st.joint || this.turning) return false;
    const J = this.joints(), b = J.bone(st.joint)!, c = this.jointCentre(b);
    if (!c) return false;
    const sc = cam().toScreen(this.v3(this.frames.toWorld(c)));
    const before = S.snapshot(), R0 = S.rotations.map((r) => r.slice());
    // the pointer's start: where it last was over the canvas, else a little right of the joint
    const at = this.pointer && Math.hypot(this.pointer.x - sc.x, this.pointer.y - sc.y) > 12 ? { ...this.pointer } : { x: sc.x + 80, y: sc.y };
    const a0 = Math.atan2(at.y - sc.y, at.x - sc.x);
    let sweep = 0, lastA = a0;
    this.turning = { before };
    const RING = { x: 0, y: 1, z: 2 } as const;
    startModal({
      kind: 'turn', label: jointName(st.joint).name,
      update: (v) => {
        // the angle swept round the centre, unwrapped, from the pointer's place (start + the modal's offset)
        const a = Math.atan2(at.y + v.dy - sc.y, at.x + v.dx - sc.x);
        let d = a - lastA; if (d > Math.PI) d -= 2 * Math.PI; if (d < -Math.PI) d += 2 * Math.PI;
        sweep += d; lastA = a;
        let deg = v.typed !== null ? v.typed : sweep * 180 / Math.PI;
        if (v.snap) deg = Math.round(deg / 5) * 5;
        S.rotations = R0.map((r) => r.slice());
        const w = this.frames.toWorld(c), cp = this.host.camera.position;
        if (v.axis === 'free') {
          const view = this.frames.dirToCard([w[0] - cp.x, w[1] - cp.y, w[2] - cp.z]);
          this.turnJoint(view, deg * Math.PI / 180);
          this.poseChanged();
          return `${Math.round(deg)}°`;
        }
        const k = RING[v.axis], v0 = J.values(b, R0)!, ax = this.jointAxes(b)[k];
        const toCam = this.frames.dirToCard([cp.x - w[0], cp.y - w[1], cp.z - w[2]]), facing = ax[0] * toCam[0] + ax[1] * toCam[1] + ax[2] * toCam[2] > 0 ? -1 : 1;
        J.set(S.rotations, b, k, v0.value[k] + (v.typed !== null ? deg : facing * deg), editStore.get().limits);
        this.poseChanged();
        return `${AXES[k].label} ${Math.round(J.values(b, S.rotations)!.value[k])}°`;
      },
      confirm: () => {
        this.turning = null;
        S.undoStack.push({ label: 'Turn the joint', snap: before }); S.redoStack.length = 0;
        this.publish();
      },
      cancel: () => { this.turning = null; S.restore(before); this.publish(); },
    }, at);
    return true;
  }
  confirmTurn(): boolean { if (!this.turning) return false; confirmModal(); return true; }
  cancelTurn(): boolean { if (!this.turning) return false; cancelModal(); return true; }

  // the bone under a point of the canvas: the strongest bone of the nearest skin vertex to the body's first hit
  private boneAt(x: number, y: number): string | null {
    const S = this.session;
    if (!S) return null;
    const ray = cam().rayAt(x, y), hit = this.host.bodyRay(ray.origin, ray.direction);
    if (!hit) return null;
    const p = this.frames.fromWorld([hit.point.x, hit.point.y, hit.point.z]), body = this.skinnedBody(), V = body.vertices(S.rotations);
    let best = -1, bd = Infinity;
    for (let i = 0; i < body.n; i++) { const d = (V[i * 3] - p[0]) ** 2 + (V[i * 3 + 1] - p[1]) ** 2 + (V[i * 3 + 2] - p[2]) ** 2; if (d < bd) { bd = d; best = i; } }
    if (best < 0) return null;
    // a bone outside the range table selects its nearest ancestor in it
    const J = this.joints();
    for (let b = body.strongest[best]; b >= 0; b = S.rig.parents[b]) if (J.has(b)) return S.rig.labels[b];
    return null;
  }

  // ---------------------------------------------------------------- limbs: handles with IK (phase 3)
  // A drag on a hand's or foot's handle moves the end in the plane facing the camera; the limb's chain is solved
  // to it with the point solve (damped least squares within the joint ranges, so a limb never snaps at full length):
  // the end's joint goes to the handle, two points of the hand or foot keep its turn in the world, the elbow or knee
  // leans to its pole handle, and pinned ends hold where they are. With Body follows on, the spine joins the chain
  // and bends a little towards a far reach (the HumanIK feel).
  private poles = new Map<string, V3>();
  private handleScreens: { id: string; kind: 'end' | 'pole'; x: number; y: number }[] = [];
  private limbDrag: { id: string; kind: 'end' | 'pole'; before: any; solver: PointSolver | null; names: string[]; bones: number[]; start: V3; keep: V3[]; plane: V3 } | null = null;

  private headOf(label: string): V3 | null { const J = this.joints(), b = J.bone(label); return b === undefined ? null : this.jointCentre(b); }
  endAt(id: End): V3 | null { return this.headOf(END_BONES[id].bone); }
  poleAt(id: End): V3 | null {
    const e = END_BONES[id];
    if (!e.pole) return null;
    const saved = this.poles.get(id);
    if (saved) return saved;
    // in front of the elbow or the knee: away from the line between the limb's root and its end
    const S = this.session!, J = this.joints(), mid = this.headOf(e.pole)!, end = this.headOf(e.bone)!;
    const rootLabel = id.startsWith('hand') ? 'upperarm01.' + id.slice(-1) : 'upperleg01.' + id.slice(-1), root = this.headOf(rootLabel)!;
    const t = [end[0] - root[0], end[1] - root[1], end[2] - root[2]], tl = Math.hypot(t[0], t[1], t[2]) || 1;
    const m = [mid[0] - root[0], mid[1] - root[1], mid[2] - root[2]], k = (m[0] * t[0] + m[1] * t[1] + m[2] * t[2]) / (tl * tl);
    let off = [m[0] - k * t[0], m[1] - k * t[1], m[2] - k * t[2]];
    const ol = Math.hypot(off[0], off[1], off[2]);
    off = ol < 1e-4 ? (id.startsWith('hand') ? [0, 1, 0] : [0, -1, 0]) : off.map((x) => x / ol);
    void S; void J;
    return [mid[0] + 0.25 * off[0], mid[1] + 0.25 * off[1], mid[2] + 0.25 * off[2]];
  }
  setPinned(id: End, on: boolean) {
    const list = editStore.get().pinned.filter((x) => x !== id);
    if (on) list.push(id);
    this.publish({ pinned: list });
  }
  setFollow(on: boolean) { this.publish({ follow: on }); }
  selectLimb(id: End | null) { this.publish({ limb: id }); }

  private limbProblem(id: End) {
    const S = this.session!, st = editStore.get(), e = END_BONES[id], J = this.joints();
    const chains = [e.chain, ...(st.follow && e.chain.startsWith('arm') ? ['spine'] : [])];
    const pin = (label: string) => ({ bones: [J.bone(label)!], s: [1], u: [S.rig.heads[J.bone(label)!].slice()] });
    const points: PointInput[] = [], names: string[] = [];
    const add = (label: string, target: V3, weight: number) => { points.push({ influence: pin(label), target, weight, on: true }); names.push(label); };
    add(e.bone, this.headOf(e.bone)!, 1);
    for (const k of e.keep) add(k, this.headOf(k)!, 0.5);
    if (e.pole) add(e.pole, this.poleAt(id)!, 0.08);
    const pins: PinInput[] = [];
    for (const other of st.pinned) {
      if (other === id) continue;
      const o = END_BONES[other];
      pins.push({ influence: pin(o.bone), kind: 'point', target: this.headOf(o.bone)!, weight: 10 });
      if (st.follow && o.chain.startsWith('arm') && !chains.includes(o.chain)) chains.push(o.chain);
    }
    for (const q of S.pins) pins.push({ influence: S.skinInfluence(q.skin.vertices, q.skin.barycentric), kind: q.kind, point: q.point, normal: q.normal, target: q.target, weight: q.weight });
    const problem = buildProblem(S.rig, S.rotations, points, chains, { limits: st.limits, pins });
    const active = new Set(problem.bones.map((b: any) => b.bone));
    return { problem, names, bones: [...active] as number[] };
  }

  beginLimbDrag(id: string, kind: 'end' | 'pole') {
    const S = this.session!, end = id as End;
    const before = S.snapshot();
    const start = kind === 'end' ? this.endAt(end)! : this.poleAt(end)!;
    const keep = END_BONES[end].keep.map((k) => this.headOf(k)!);
    // the drag's plane faces the camera through the handle
    const c = this.host.camera.position, w = this.frames.toWorld(start), plane = this.frames.dirToCard([w[0] - c.x, w[1] - c.y, w[2] - c.z]);
    const prob = kind === 'end' ? this.limbProblem(end) : null;
    this.limbDrag = { id, kind, before, solver: prob ? new PointSolver(prob.problem) : null, names: prob ? prob.names : [], bones: prob ? prob.bones : [], start, keep, plane };
  }
  limbDragTo(x: number, y: number) {
    const D = this.limbDrag, S = this.session;
    if (!D || !S) return;
    const ray = cam().rayAt(x, y), o = this.frames.fromWorld([ray.origin.x, ray.origin.y, ray.origin.z]), d = this.frames.dirToCard([ray.direction.x, ray.direction.y, ray.direction.z]);
    const n = D.plane, den = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
    if (Math.abs(den) < 1e-9) return;
    const t = ((D.start[0] - o[0]) * n[0] + (D.start[1] - o[1]) * n[1] + (D.start[2] - o[2]) * n[2]) / den;
    const p: V3 = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t], move = [p[0] - D.start[0], p[1] - D.start[1], p[2] - D.start[2]];
    if (D.kind === 'pole') { this.poles.set(D.id, p); this.solveLimb(D.id as End); this.dirty = true; return; }
    const sv = D.solver!;
    sv.setTarget(0, p);
    D.keep.forEach((k, i) => sv.setTarget(1 + i, [k[0] + move[0], k[1] + move[1], k[2] + move[2]]));
    const r = sv.solve(10);
    for (const b of D.bones) S.rotations[b] = Array.from(r.rotations.subarray(b * 9, b * 9 + 9));
    this.poseChanged();
  }
  endLimbDrag(cancel = false) {
    const D = this.limbDrag, S = this.session;
    this.limbDrag = null;
    if (!D || !S) return;
    if (cancel) { S.restore(D.before); this.publish(); return; }
    S.undoStack.push({ label: D.kind === 'pole' ? 'Pole' : ENDS.find((e) => e.id === D.id)!.label, snap: D.before }); S.redoStack.length = 0;
    this.publish();
  }
  // a full solve of a limb to its present end, keeps and pole (after the pole moves)
  private solveLimb(id: End) {
    const S = this.session!, { problem, bones } = this.limbProblem(id);
    const r = new PointSolver(problem).solve(20);
    for (const b of bones) S.rotations[b] = Array.from(r.rotations.subarray(b * 9, b * 9 + 9));
    this.poseChanged();
  }

  private drawHandles(f: Frame, st: EditState) {
    const toS = (x: V3) => cam().toScreen(this.v3(this.frames.toWorld(x)));
    this.handleScreens = [];
    f.handles = [];
    for (const e of ENDS.slice(0, 4)) {
      const at = this.endAt(e.id);
      if (!at) continue;
      const s = toS(at);
      if (s.depth <= 0) continue;
      this.handleScreens.push({ id: e.id, kind: 'end', x: s.x, y: s.y });
      f.handles.push({ x: s.x, y: s.y, kind: 'end', pinned: st.pinned.includes(e.id), selected: st.limb === e.id, hovered: this.hoverHandle === e.id + ':end' });
    }
    if (st.limb) {
      const pole = this.poleAt(st.limb), mid = this.headOf(END_BONES[st.limb].pole || '');
      if (pole && mid) {
        const s = toS(pole), m = toS(mid);
        this.handleScreens.push({ id: st.limb, kind: 'pole', x: s.x, y: s.y });
        f.handles.push({ x: s.x, y: s.y, kind: 'pole', pinned: false, selected: false, hovered: this.hoverHandle === st.limb + ':pole', from: [m.x, m.y] });
      }
    }
  }
  private hoverHandle: string | null = null;

  private registerLimbPickers() {
    const on = () => { const st = editStore.get(); return st.on && st.section === 'limbs'; };
    this.offs.push(registerPicker({
      id: 'pose-handles', priority: 20, enabled: on,
      hit: (p) => this.handleScreens.map((h) => ({ item: h.id + ':' + h.kind, distPx: Math.max(0, Math.hypot(h.x - p.x, h.y - p.y) - (h.kind === 'end' ? 13 : 8)),
        label: h.kind === 'pole' ? 'Pole' : ENDS.find((e) => e.id === h.id)!.label })).filter((h) => h.distPx <= p.radius),
      onDown: (h) => { const [id, kind] = (h.item as string).split(':'); this.selectLimb(id as End); this.beginLimbDrag(id, kind as any); return true; },
      onDrag: (_h, p) => this.limbDragTo(p.x, p.y),
      onUp: () => this.endLimbDrag(false),
      onCancel: () => this.endLimbDrag(true),
      onClick: (h) => { this.endLimbDrag(true); const [id] = (h.item as string).split(':'); this.selectLimb(id as End); },
      onContext: (h) => { const [id, kind] = (h.item as string).split(':'); if (kind !== 'end') return false; this.setPinned(id as End, !editStore.get().pinned.includes(id as End)); },
      onHover: (h) => { this.hoverHandle = h ? (h.item as string) : null; this.dirty = true; },
      cursor: 'grab',
    }));
  }

  // ---------------------------------------------------------------- the library (phase 4)
  savePose(label: string) {
    const S = this.session;
    if (!S) return;
    const list = [...editStore.get().poses, { id: 'pose-' + Date.now().toString(36), label, bones: this.host.cardPose().bones }];
    savePoses(list);
    this.publish({ poses: list });
    this.host.toast(`Saved the pose ${label}`);
  }
  deletePose(id: string) { const list = editStore.get().poses.filter((p) => p.id !== id); savePoses(list); this.publish({ poses: list }); }
  // a library pose's local-ref turns by bone index: a saved pose's pose.bones, as a card keeps them (a bone it lacks
  // stands at the zero, Anny's reference pose), or a built-in pose (an id "builtin:NAME") converted from the page's
  // quaternions for the body shown
  private poseRotations(id: string): number[][] | null {
    if (id.startsWith(BUILTIN)) {
      const q = this.host.builtinPose(id.slice(BUILTIN.length));
      return q ? this.frames.cardRotations(q) : null;
    }
    const p = editStore.get().poses.find((x) => x.id === id);
    return p ? cardTurns(p.bones, this.session!.rig.labels) : null;
  }
  poseLabel(id: string): string {
    if (id.startsWith(BUILTIN)) return this.host.builtinPoses().find((c) => c.name === id.slice(BUILTIN.length))?.label || id.slice(BUILTIN.length);
    return editStore.get().poses.find((x) => x.id === id)?.label || id;
  }
  // the built-in poses of the page's library, for the Library tool
  builtinPoses() { return this.host.builtinPoses(); }
  // a library pose (saved or built-in) onto the part chosen (the whole body, a hand, the arms, the legs)
  applyPose(id: string, part = editStore.get().applyTo) {
    const S = this.session, R = S ? this.poseRotations(id) : null;
    if (!S || !R) return;
    const test = (PARTS.find((x) => x.id === part) || PARTS[0]).test, root = this.host.root();
    this.change(`Apply ${this.poseLabel(id)}`, (cur) => { S.rig.labels.forEach((l, i) => { if (i !== root && test(l)) cur[i] = R[i]; }); });
  }
  setApplyTo(part: string) { if (PARTS.some((p) => p.id === part)) this.publish({ applyTo: part }); }
  // two library poses (saved or built-in) blended by t (a turn per bone, along the shortest arc)
  blendPoses(a: string, b: string, t: number, record = true) {
    const S = this.session, RA = S ? this.poseRotations(a) : null, RB = S ? this.poseRotations(b) : null;
    if (!S || !RA || !RB) return;
    const test = (PARTS.find((x) => x.id === editStore.get().applyTo) || PARTS[0]).test, root = this.host.root();
    this.change('Blend poses', (cur) => {
      S.rig.labels.forEach((l, i) => {
        if (i === root || !test(l)) return;
        // R = RA exp(t log(RA^T RB))
        const RAt = [RA[i][0], RA[i][3], RA[i][6], RA[i][1], RA[i][4], RA[i][7], RA[i][2], RA[i][5], RA[i][8]];
        const rel = mul3(RAt, RB[i]), w = logOf(rel);
        cur[i] = mul3(RA[i], expRotation(w.map((x) => x * t)));
      });
    }, record);
    this.publish({ blend: { a, b, t } });
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
      const state: DrawPoint['state'] = !t ? 'unplaced' : !t.on && t.placed === 'auto' && t.confidence < 0.5 ? 'look' : !t.on ? 'off' : 'placed';
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
    // the joints (the Joints tool and the Turn tool) and the dial of the selected joint
    if (st.section === 'joints' || st.tool === 'turn') { f.points = f.points.filter((q) => q.selected); f.anny = []; f.clusters = []; this.drawJoints(f, st); }
    else { this.jointScreens = []; this.dial = null; }
    if (st.section === 'limbs') { f.points = []; f.anny = []; f.clusters = []; this.drawHandles(f, st); } else this.handleScreens = [];
    // the label of the selected or hovered point
    const lab = st.section === 'points' ? this.hovered || st.selected : null;
    if (lab && this.screen.has(lab)) {
      const p = POINT_BY_NAME.get(lab)!, s = this.screen.get(lab)!, t = S.targets.get(lab);
      const line = st.dragging ? 'Slides on the scan. The hand follows.' : !t ? 'Not placed yet'
        : !t.on && t.placed === 'auto' && t.confidence < 0.5 ? `Needs a look: ${t.reason || 'unsure'}` : !t.on ? 'Switched off'
          : t.depth_mm > 0 ? `${t.depth_mm.toFixed(1)} mm into the stone` : !this.visible.get(lab) ? 'Behind the stone' : 'Placed on the scan';
      f.labels.push({ x: s.x, y: s.y - 34, title: `${p.number} ${p.label}`, line, side: 'left' });
    }
    // Grip: the fingers inside the prop, marked red at their tips
    if (st.section === 'hands' && st.grip && st.grip_report.length) {
      const side = st.grip.hand, tips = ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip'], at = S.positions(tips.map((t) => POINT_BY_NAME.get(`${t}.${side}`)!));
      f.alerts = st.grip_report.flatMap((r, k) => {
        if (!r.inside) return [];
        const s = toS(at.get(`${tips[k]}.${side}`)!);
        return [{ x: s.x, y: s.y, title: `${FINGER_LABELS[k]}: ${Math.abs(r.clearance).toFixed(0)} mm inside the ${(st.grip!.prop.split('/').pop() || 'prop')}`, line: 'Red marks skin inside a prop' }];
      });
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
    if (st.prop) this.drawPropHandles(f, st); else this.propHandles = null;
    if (this.magCam && this.magAt && this.drag && this.mag) {
      // the magnifier shows the spot under the pointer MAG_ZOOM times larger, so a point lies at its offset from the
      // pointer times the zoom
      const t = S.targets.get(this.drag.name)!, q = toS(t.position), half = MAG_SIZE / 2;
      const px = this.magAt.cx + (q.x - this.mag.x) * MAG_ZOOM, py = this.magAt.cy + (q.y - this.mag.y) * MAG_ZOOM;
      f.magnifier = { cx: this.magAt.cx, cy: this.magAt.cy, r: half, zoom: MAG_ZOOM, note: this.mag.off ? 'Off the scan: the point stays here' : undefined,
        point: Math.hypot(px - this.magAt.cx, py - this.magAt.cy) < half - 8 ? { x: px, y: py, number: POINT_BY_NAME.get(this.drag.name)!.number, state: 'placed' } : undefined };
    }
    this.layer!.draw(f);
  }

  private jointScreens: { label: string; x: number; y: number }[] = [];
  private dial: { b: number; cx: number; cy: number; outer: number; rings: { k: number; pts: [number, number][] }[] } | null = null;
  private hoverJoint: string | null = null;
  private hoverRing: number | null = null;

  private drawJoints(f: Frame, st: EditState) {
    const S = this.session!, J = this.joints(), toS = (x: V3) => cam().toScreen(this.v3(this.frames.toWorld(x)));
    const labels = new Set(BODY_JOINTS.map((j) => j.label));
    // the finger joints of a hand when the camera is close to it
    for (const side of ['L', 'R'] as const) for (let k = 1; k <= 5; k++) for (let j = 1; j <= 3; j++) labels.add(`finger${k}-${j}.${side}`);
    this.jointScreens = [];
    f.joints = [];
    for (const l of labels) {
      const b = J.bone(l);
      if (b === undefined) continue;
      const c = this.jointCentre(b)!, s = toS(c);
      if (s.depth <= 0) continue;
      const finger = l.startsWith('finger');
      if (finger) {
        // only where the fingers stand apart on screen
        const p = J.bone(l.replace(/-\d/, '-1'))!, tip = this.jointCentre(J.bone(l.replace(/-\d/, '-3'))!)!, ps = toS(this.jointCentre(p)!), ts = toS(tip);
        if (Math.hypot(ps.x - ts.x, ps.y - ts.y) < 40) continue;
      }
      this.jointScreens.push({ label: l, x: s.x, y: s.y });
      f.joints.push({ x: s.x, y: s.y, selected: l === st.joint, hovered: l === this.hoverJoint, side: l.endsWith('.L') ? 'L' : l.endsWith('.R') ? 'R' : '' });
    }
    this.dial = null;
    if (!st.joint) return;
    const b = J.bone(st.joint);
    if (b === undefined || !J.has(b)) return;
    const c = this.jointCentre(b)!, sc = toS(c), cam3 = this.host.camera, H = this.host.canvas.clientHeight || 800;
    const R0 = 74 * f.scale, rw = R0 * sc.depth * 2 * Math.tan((cam3.fov * Math.PI / 180) / 2) / H;
    const axes = this.jointAxes(b), jv = J.values(b, S.rotations)!;
    const strokes: any[] = [], tags: any[] = [], rings: { k: number; pts: [number, number][] }[] = [];
    axes.forEach((a, k) => {
      if (!jv.dofs[k]) return;
      // two directions square to the axis; the first along the bone where it can be
      const ref = Math.abs(a[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
      let u = [a[1] * ref[2] - a[2] * ref[1], a[2] * ref[0] - a[0] * ref[2], a[0] * ref[1] - a[1] * ref[0]];
      const lu = Math.hypot(u[0], u[1], u[2]); u = u.map((x) => x / lu);
      const v = [a[1] * u[2] - a[2] * u[1], a[2] * u[0] - a[0] * u[2], a[0] * u[1] - a[1] * u[0]];
      const r = rw * (1 - 0.12 * k);
      const at = (t: number): [number, number] => { const p: V3 = [0, 1, 2].map((i) => c[i] + r * (Math.cos(t) * u[i] + Math.sin(t) * v[i])) as V3; const q = toS(p); return [q.x, q.y]; };
      const pts: [number, number][] = [];
      for (let i = 0; i <= 72; i++) pts.push(at(i / 72 * 2 * Math.PI));
      rings.push({ k, pts });
      const d = 'M' + pts.map((p) => p.map((x) => x.toFixed(1)).join(' ')).join(' L');
      const hot = this.hoverRing === k || this.ringDrag?.k === k;
      strokes.push({ d, colour: AXES[k].colour, width: hot ? 3 : 1.8, opacity: 0.95 });
      // the range: a thick arc from the low end to the high end, the present value at angle 0
      const DEGR = Math.PI / 180, lo = (jv.lo[k] - jv.value[k]) * DEGR, hi = (jv.hi[k] - jv.value[k]) * DEGR;
      if (st.limits && hi - lo < 2 * Math.PI - 1e-3) {
        const arc: [number, number][] = [];
        for (let i = 0; i <= 48; i++) arc.push(at(lo + (hi - lo) * i / 48));
        strokes.push({ d: 'M' + arc.map((p) => p.map((x) => x.toFixed(1)).join(' ')).join(' L'), colour: AXES[k].colour, width: 6, opacity: 0.85 });
      }
      // the tags stand outside the dial, one above another
      tags.push({ x: sc.x + R0 + 70 * f.scale, y: sc.y - 28 + 28 * k, text: `${AXES[k].label} ${Math.round(jv.value[k])}°`, colour: AXES[k].colour });
    });
    // the outer ring: a free turn
    const outer = R0 + 22 * f.scale;
    let od = '';
    for (let i = 0; i <= 96; i++) { const t = i / 96 * 2 * Math.PI; od += (i ? ' L' : 'M') + (sc.x + outer * Math.cos(t)).toFixed(1) + ' ' + (sc.y + outer * Math.sin(t)).toFixed(1); }
    strokes.unshift({ d: od, colour: '#c9c2b7', width: this.hoverRing === 3 ? 2.6 : 1.4, dash: '4 4', opacity: 0.9 });
    tags.push({ x: sc.x - outer - 60, y: sc.y + outer * 0.55, text: 'Outer ring: free turn', colour: '#9e968a' });
    f.strokes = strokes; f.tags = tags;
    this.dial = { b, cx: sc.x, cy: sc.y, outer, rings };
  }

  private ringDrag: { k: number; before: any; last: number; value: number; snapped: number } | null = null;
  private hitRing(p: PointerInfo): number | null {
    const D = this.dial;
    if (!D) return null;
    let best: number | null = null, bd = 9 * settings.targetScale();
    for (const r of D.rings) {
      for (let i = 1; i < r.pts.length; i++) {
        const [ax, ay] = r.pts[i - 1], [bx, by] = r.pts[i], dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy || 1;
        const t = Math.min(1, Math.max(0, ((p.x - ax) * dx + (p.y - ay) * dy) / l2)), d = Math.hypot(p.x - ax - t * dx, p.y - ay - t * dy);
        if (d < bd) { bd = d; best = r.k; }
      }
    }
    if (best === null && Math.abs(Math.hypot(p.x - D.cx, p.y - D.cy) - D.outer) < 9 * settings.targetScale()) best = 3;
    return best;
  }
  private ringDragMove(p: PointerInfo, d: DragInfo) {
    const D = this.dial, R = this.ringDrag, S = this.session;
    if (!D || !R || !S) return;
    const J = this.joints(), b = D.b;
    if (R.k === 3) {
      // the outer ring: an arcball turn about the axis square to the drag and the view
      const cp = this.host.camera.position, c = this.jointCentre(b)!, w = this.frames.toWorld(c);
      const view = this.frames.dirToCard([w[0] - cp.x, w[1] - cp.y, w[2] - cp.z]);
      const cm = this.host.camera.matrixWorld.elements, right = this.frames.dirToCard([cm[0], cm[1], cm[2]]), up = this.frames.dirToCard([cm[4], cm[5], cm[6]]);
      const mv = [right[0] * d.ddx - up[0] * d.ddy, right[1] * d.ddx - up[1] * d.ddy, right[2] * d.ddx - up[2] * d.ddy];
      const ax = [mv[1] * view[2] - mv[2] * view[1], mv[2] * view[0] - mv[0] * view[2], mv[0] * view[1] - mv[1] * view[0]], l = Math.hypot(ax[0], ax[1], ax[2]);
      if (l < 1e-9) return;
      this.turnJoint([ax[0] / l, ax[1] / l, ax[2] / l] as V3, -Math.hypot(d.ddx, d.ddy) * d.scale / D.outer);
    } else {
      const a = Math.atan2(p.y - D.cy, p.x - D.cx);
      let da = a - R.last; if (da > Math.PI) da -= 2 * Math.PI; if (da < -Math.PI) da += 2 * Math.PI;
      R.last = a;
      const ax = this.jointAxes(b)[R.k], cp = this.host.camera.position, w = this.frames.toWorld(this.jointCentre(b)!);
      const toCam = this.frames.dirToCard([cp.x - w[0], cp.y - w[1], cp.z - w[2]]), facing = ax[0] * toCam[0] + ax[1] * toCam[1] + ax[2] * toCam[2] > 0 ? -1 : 1;
      R.value += facing * da * 180 / Math.PI * d.scale;
      const v = d.snap ? Math.round(R.value / 5) * 5 : R.value;
      J.set(S.rotations, b, R.k, v, editStore.get().limits);
    }
    this.poseChanged();
    this.dirty = true;
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
    const pointsOn = () => { const st = editStore.get(); return st.on && st.section === 'points' && st.tool !== 'turn'; };
    this.offs.push(registerPicker({ id: 'pose-points-select', priority: 10, enabled: () => pointsOn() && editStore.get().tool !== 'move', ...common }));
    this.offs.push(registerPicker({
      id: 'pose-points-move', priority: 10, enabled: () => pointsOn() && editStore.get().tool === 'move', ...common,
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
  private registerJointPickers() {
    const jointsOn = () => { const st = editStore.get(); return st.on && (st.section === 'joints' || st.tool === 'turn'); };
    this.offs.push(registerPicker({
      id: 'pose-rings', priority: 30, enabled: () => jointsOn() && !!editStore.get().joint,
      hit: (p) => { const k = this.hitRing(p); return k === null ? null : { item: k, distPx: 0, label: k === 3 ? 'Free turn' : AXES[k].label }; },
      onDown: (h) => { const st = editStore.get(), J = this.joints(), b = J.bone(st.joint!)!, v = J.values(b, this.session!.rotations)!;
        this.ringDrag = { k: h.item as number, before: this.session!.snapshot(), last: 0, value: (h.item as number) < 3 ? v.value[h.item as number] : 0, snapped: 0 };
        return true; },
      onDrag: (_h, p, d) => {
        const R = this.ringDrag; if (!R) return;
        if (R.last === 0 && this.dial) R.last = Math.atan2(d.y0 - this.dial.cy, d.x0 - this.dial.cx);
        this.ringDragMove(p, d);
      },
      onUp: () => { const R = this.ringDrag, S = this.session; this.ringDrag = null; if (R && S) { S.undoStack.push({ label: R.k === 3 ? 'Free turn' : AXES[R.k].label, snap: R.before }); S.redoStack.length = 0; } this.publish(); },
      onCancel: () => { const R = this.ringDrag, S = this.session; this.ringDrag = null; if (R && S) S.restore(R.before); this.publish(); },
      onHover: (h) => { this.hoverRing = h ? (h.item as number) : null; this.dirty = true; },
      cursor: 'grab',
    }));
    this.offs.push(registerPicker({
      id: 'pose-joints', priority: 5, enabled: jointsOn,
      hit: (p) => this.jointScreens.map((j) => ({ item: j.label, distPx: Math.max(0, Math.hypot(j.x - p.x, j.y - p.y) - 8), label: jointName(j.label).name }))
        .filter((h) => h.distPx <= p.radius),
      onClick: (h) => this.selectJoint(h.item as string),
      onHover: (h) => { this.hoverJoint = h ? (h.item as string) : null; this.dirty = true; },
      onBackground: (p) => { const l = this.boneAt(p.x, p.y); this.selectJoint(l); },
      cursor: 'pointer',
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
    if (st.prop) { this.statusProp(); return; }
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
    defineScope({ id: 'pose-joint', parent: 'pose-edit', active: () => !!editStore.get().joint && (editStore.get().section === 'joints' || editStore.get().tool === 'turn') });
    defineScope({ id: 'pose-prop', parent: 'pose-edit', active: () => !!editStore.get().prop });
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
      tool('turn', 'Turn tool', 'E', 'Drags turn a joint\'s rings'),
      ...SECTIONS.map((s, i): ActionDef => ({ id: `pose.tool.${i + 1}`, label: s.label, desc: `Pose Edit: ${s.label.toLowerCase()}`, keys: [String(i + 1)], group: 'Posing tools',
        tier: 'occasional', scope: 'pose-edit', checked: () => editStore.get().section === s.id, run: () => this.setSection(s.id) })),
      { id: 'pose.grab', label: 'Grab the point', desc: 'The selected point follows the mouse until a click or Enter', keys: ['G'], group: 'Points', tier: 'frequent',
        scope: 'pose-select', hint: 'Select a point first', run: () => this.startGrab() },
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
      { id: 'pose.turn-keys', label: 'Turn the joint', desc: 'The joint turns with the mouse until a click or Enter; X, Y or Z keep to one ring', keys: ['R'], group: 'Joints',
        tier: 'frequent', scope: 'pose-joint', run: () => this.startTurn() },
      { id: 'pose.reset-joint', label: 'Reset the joint', desc: 'The joint back to its rest turn (Shift: the whole limb)', keys: ['Alt+R', 'Shift+Alt+R'], group: 'Joints',
        tier: 'frequent', scope: 'pose-joint', run: (c) => this.resetJoint(c.shift) },
      { id: 'pose.clear-joint', label: 'Clear the joint', keys: ['Escape'], group: 'Joints', tier: 'constant', scope: 'pose-joint', help: false, run: () => { this.selectJoint(null); } },
      { id: 'pose.mirror', label: 'Mirror the pose', desc: 'The whole pose to the other side', keys: ['M'], group: 'Joints', tier: 'occasional', scope: 'pose-edit', run: () => this.mirror() },
      { id: 'pose.copy', label: 'Copy the pose', desc: 'The selected hand\'s or limb\'s pose', keys: ['Ctrl+C'], group: 'Joints', tier: 'occasional', scope: 'pose-edit',
        hint: 'Select a hand or a joint first', run: () => this.copyPose() },
      { id: 'pose.paste', label: 'Paste the pose', keys: ['Ctrl+V'], group: 'Joints', tier: 'occasional', scope: 'pose-edit', hint: 'Copy a pose first (Ctrl+C)', run: () => this.pastePose(false) },
      { id: 'pose.paste-mirrored', label: 'Paste mirrored', desc: 'The copied pose onto the other side', keys: ['Ctrl+Shift+V'], group: 'Joints', tier: 'occasional',
        scope: 'pose-edit', hint: 'Copy a pose first (Ctrl+C)', run: () => this.pastePose(true) },
      { id: 'pose.undo', label: 'Undo the pose edit', keys: ['Ctrl+Z'], group: 'History', tier: 'constant', scope: 'pose-edit', help: false, run: () => this.undo() },
      { id: 'pose.redo', label: 'Redo the pose edit', keys: ['Ctrl+Shift+Z', 'Ctrl+Y'], group: 'History', tier: 'constant', scope: 'pose-edit', help: false, run: () => this.redo() },
      { id: 'pose.place', label: 'Place points', desc: 'Every point on the scan, from the fitted character', group: 'Points', tier: 'occasional', scope: 'pose-edit', place: 'Pose › Edit',
        run: () => { this.placeAll(); } },

      // the props (DESIGN-0013, "Props"): live while a prop is selected, in place of the point's and the joint's keys
      { id: 'pose.prop.grab', label: 'Grab the prop', desc: 'The selected prop follows the mouse until a click or Enter; X, Y or Z keep to one axis, a number moves it that many millimetres',
        keys: ['G'], group: 'Props', tier: 'frequent', scope: 'pose-prop', run: () => this.startPropGrab() },
      { id: 'pose.prop.turn', label: 'Turn the prop', desc: 'The selected prop turns round its pivot with the mouse until a click or Enter; X, Y or Z keep to one axis, a number turns it that many degrees',
        keys: ['R'], group: 'Props', tier: 'frequent', scope: 'pose-prop', run: () => this.startPropTurn() },
      { id: 'pose.prop.clear-move', label: 'Clear the prop\'s move', desc: 'Its turn stays', keys: ['Alt+G'], group: 'Props', tier: 'frequent', scope: 'pose-prop',
        hint: 'The prop is where it was made', run: () => this.clearPropMove() },
      { id: 'pose.prop.clear-turn', label: 'Clear the prop\'s turn', desc: 'Its move stays', keys: ['Alt+R'], group: 'Props', tier: 'frequent', scope: 'pose-prop',
        hint: 'The prop is where it was made', run: () => this.clearPropTurn() },
      { id: 'pose.prop.nudge', label: 'Nudge the prop', desc: '1 mm across the view (Shift: 0.2 mm)', keys: ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Shift+ArrowLeft',
        'Shift+ArrowRight', 'Shift+ArrowUp', 'Shift+ArrowDown'], group: 'Props', tier: 'frequent', scope: 'pose-prop', repeat: true,
        run: (c) => { const k = c.event?.key || c.chord.replace('Shift+', ''); return this.nudgeProp(k === 'ArrowLeft' ? -1 : k === 'ArrowRight' ? 1 : 0, k === 'ArrowUp' ? 1 : k === 'ArrowDown' ? -1 : 0, 0, c.shift ? 0.2 : 1); } },
      { id: 'pose.prop.push', label: 'Move the prop away', desc: '1 mm away from the camera (Shift: 0.2 mm)', keys: ['PageDown', 'Shift+PageDown'], group: 'Props', tier: 'frequent',
        scope: 'pose-prop', repeat: true, run: (c) => this.nudgeProp(0, 0, 1, c.shift ? 0.2 : 1) },
      { id: 'pose.prop.pull', label: 'Move the prop nearer', desc: '1 mm towards the camera (Shift: 0.2 mm)', keys: ['PageUp', 'Shift+PageUp'], group: 'Props', tier: 'frequent',
        scope: 'pose-prop', repeat: true, run: (c) => this.nudgeProp(0, 0, -1, c.shift ? 0.2 : 1) },
      { id: 'pose.prop.reset', label: 'Put the prop back', desc: 'Back to the starter\'s placement, on the bone it was made on', group: 'Props', tier: 'occasional', scope: 'pose-prop',
        place: 'Pose › Edit, Props', run: () => this.resetProp() },
      { id: 'pose.prop.clear', label: 'Clear the prop selection', keys: ['Escape'], group: 'Props', tier: 'constant', scope: 'pose-prop', help: false, run: () => { this.selectProp(null); } },
    ];
    registerActions(list);
    this.registerPickers();
    this.registerJointPickers();
    this.registerLimbPickers();
    this.registerPropPickers();
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
    // how far Anny's own point lies from the scan (the ghost's offset there), within 20 cm
    let annyGap: number | null = null;
    if (a && ed.bvh) { const q = ed.bvh.closest(a, 0.2); annyGap = q ? q.distance : null; }
    pts[p.name] = { number: p.number, target: t ? t.position : null, on: t ? t.on : null, confidence: t ? t.confidence : null, depth: t ? t.depth_mm : 0,
      placed: t ? t.placed : null, anny: a || null, screen: s || null, visible: (ed as any).visible.get(p.name) ?? null, onScan, annyGap };
  }
  return { state: { ...editStore.get(), hands: undefined, body: undefined }, lift: ed.lastLift, points: pts, magnifier: (ed as any).mag, props: ed.propReport(), moveOut: ed.lastMoveOut,
    unsaved: ed.unsavedEdits(), prompt: ed.promptOn(), rotations: S ? S.rotations : null, undo: S ? S.undoStack.map((e) => e.label) : [],
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

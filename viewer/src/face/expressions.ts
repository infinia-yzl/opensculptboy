// OpenSculptBoy
// Apache License, Version 2.0
//
// The face of the page (docs/viewer.md, "Faces"): the FaceDriver of face/api.ts on the page's body. A face is a mix of
// Anny's facial actions (face/rig.ts plays them on the fine body, jawOpen as the jaw's turn), a gaze and a turn of the
// head; a card opened in the Wearables section adds its own mouth poses ('mouth.<pose>') and plays its own eye targets
// for the blinks and the squints, as its exports do. The presets are the VRM presets and visemes of
// opensculptboy/data/vrm/expressions.json, which the VRM export writes.

import { registerFace, type FaceDriver, type FaceState } from './api.ts';
import { FaceLayer, FaceRig, rotvecMatrix, turnPoint, type FaceData, type LayerBody, type Shape } from './rig.ts';
import type { FaceMorph } from './morph.ts';

// the actions that a card's eye targets play in place of Anny's (opensculptboy.eyes.page.ACTIONS)
export const CARD_EYE_ACTIONS = ['eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight'];
// the reference's poses that the VRM presets of the vowels play (opensculptboy.mouth.poses.VRM_VISEMES)
export const VRM_VISEMES: Record<string, string> = { aa: 'viseme_a', ih: 'viseme_i', ou: 'viseme_u', ee: 'viseme_e', oh: 'viseme_o' };
export const POSE_PREFIX = 'mouth.';
const DEG = Math.PI / 180;

// a card's mouth poses on the page (card/look.ts): per pose its offsets of the coarse body and of the rows of the level
// the page shows, and of the points of the inner mouth (page frame)
export interface CardPoses {
  names: string[];
  coarse: Shape[];
  level: number;                 // the page's level of `rows` (1 or 2), 0 for none
  rows: Shape[];
  inner: (Float32Array | null)[];
}

// the motion of the face for the parts of a card (card/look.ts followFace)
export interface FaceMotion {
  skin(v: number, out: number[]): void;                       // the offset of a fine vertex
  region: Uint32Array;                                         // the fine vertices that the face moves
  lower(x: number, y: number, z: number, out: number[]): void;   // the offset of a point of the lower jaw
  tongue(x: number, y: number, z: number, out: number[], key?: number): void;  // the offset of a point of the tongue (key: a number of the point, which keeps its nearest vertices of Anny's tongue)
  jawTurn: Float64Array;                                      // the jaw's turn (row-major), for the normals of the lower jaw
  coarse(): Float32Array | null;                              // the face's offsets of the coarse body (made when asked)
  poses: Float32Array;                                        // the weight of each of the card's poses
}

// what the face needs from the renderer (main.ts)
export interface FaceHost {
  body(): LayerBody;
  data(): FaceData | null;
  upload(lo: number, hi: number): void;            // the body's arrays changed on these vertices: upload and draw again
  redraw(): void;                                  // the face on the GPU changed: draw again
  morph(): FaceMorph | null;                       // the face on the GPU (face/morph.ts), when the page has it
  // the page's brows and lashes follow the skin; `at` gives the face's position of a vertex when the body's arrays hold
  // the rest (the face on the GPU)
  strands(at?: (v: number, out: number[]) => void): void;
  eyes(turns: Record<string, Float64Array>): void;   // the turn of each eye (row-major, page frame), about its centre
  head(turn: number, tilt: number, nod: number): void;   // degrees: turn to the figure's left, tilt to its left shoulder, nod down
  toDefault(p: number[]): number[];                // a point of the current head carried to the head of anny's default body
  headScale(): number;                             // how much larger the head is than on anny's default body
  card(): FaceCard | null;                         // the card's look, when a card shows
  drawn(): Promise<void>;                          // resolves after the next drawn frame
}

// what the face needs from a card's look (card/look.ts)
export interface FaceCard {
  hasBlinks: boolean;
  setFaceEyes(close: { L: number; R: number }, squint: { L: number; R: number }): void;
  poses(): CardPoses | null;
  warmFace?(): void;          // the eye targets' shapes on the current body, made ahead of the first blink
  followFace(m: FaceMotion | null): void;
}

export class PageFace implements FaceDriver {
  host: FaceHost;
  rig: FaceRig | null = null;
  layer: FaceLayer | null = null;
  state: FaceState = { actions: {} };
  timing = { apply: 0, rebase: 0, follow: 0, count: 0 };
  private blinkW: Float32Array | null = null;     // the card's eye targets' weights (card/look.ts)
  private poses: CardPoses | null = null; private poseKey = '';
  private fade: { from: FaceState; to: FaceState; t0: number; ms: number; resolve: () => void } | null = null;
  private applying = false;
  constructor(host: FaceHost) { this.host = host; }

  // the face's data on the page's body (at load)
  init(): boolean {
    const d = this.host.data();
    if (!d) return false;
    this.rig = new FaceRig(this.host.body(), d);
    this.layer = new FaceLayer(this.rig, this.host.body());
    registerFace(this);
    return true;
  }
  get ready() { return !!this.layer; }

  // ---------------------------------------------------------------- the interface (face/api.ts)
  actionNames(): string[] {
    if (!this.rig) return [];
    const card = this.host.card()?.poses();
    return [...this.rig.names, ...(card ? card.names.map((n) => POSE_PREFIX + n) : [])];
  }
  presetNames(): string[] {
    const m = this.host.data()?.meta;
    return m ? [...Object.keys(m.presets), ...Object.keys(m.visemes)] : [];
  }
  preset(name: string, weight = 1): FaceState {
    const m = this.host.data()?.meta, w = Math.min(1, Math.max(0, weight));
    if (!m) return { actions: {} };
    const mix = m.presets[name] || m.visemes[name] || {};
    // a card with the reference's vowel plays it, as its VRM export does
    const pose = VRM_VISEMES[name], card = this.host.card()?.poses();
    if (pose && card && card.names.includes(pose)) return { actions: { [POSE_PREFIX + pose]: w } };
    const actions: Record<string, number> = {};
    for (const [k, v] of Object.entries(mix)) if (v * w) actions[k] = v * w;
    return { actions };
  }
  async set(state: FaceState): Promise<void> {
    this.fade = null;
    this.show(clean(state));
    await this.host.drawn();
  }
  get(): FaceState { return JSON.parse(JSON.stringify(this.state)); }

  // a face reached over ms milliseconds (the Expression section's buttons); resolves once it shows
  fadeTo(state: FaceState, ms = 180): Promise<void> {
    if (this.fade) this.fade.resolve();
    return new Promise((resolve) => { this.fade = { from: this.get(), to: clean(state), t0: performance.now(), ms, resolve }; });
  }
  // a frame of the page: the fade moves on; once the face rests, its normals follow its triangles exactly
  tick(now: number) {
    const f = this.fade;
    if (!f) {
      if (this.waiting) this.apply(false);
      else if (this.rough && now - this.lastApply > 60) this.apply(true);
      return;
    }
    const t = Math.min(1, (now - f.t0) / f.ms), e = t * t * (3 - 2 * t);
    if (t >= 1) { this.fade = null; this.show(f.to); this.host.drawn().then(f.resolve); }
    else this.show(mix(f.from, f.to, e), false);
  }
  private rough = false; private lastApply = 0; private waiting = false;
  get fading() { return !!this.fade; }

  // ---------------------------------------------------------------- the page's side
  // a step of the warm-up while the page rests: the copy of the rest, then one action's fine shape and the change of its
  // normals; returns false once every action is ready
  private warmAt = 0; private warmCard = false;
  warm(): boolean {
    const rig = this.rig, layer = this.layer;
    if (!rig || !layer) return false;
    if (!layer.ready) { layer.rebase(); const m = this.host.morph(); if (m) this.jawLayer(m); return true; }
    while (this.warmAt < rig.names.length && this.warmAt === rig.jaw) this.warmAt++;
    if (this.warmAt < rig.names.length) { layer.normalShape(rig.fineShape(this.warmAt++)); return true; }
    // a card's faces: its poses' shapes, and its eye targets on the body
    const card = this.host.card();
    if (!card || this.warmCard) return false;
    this.warmCard = true;
    this.cardPoses(card);
    for (const g of layer.groups.values()) for (const sh of g.shapes) layer.normalShape(sh);
    card.warmFace?.();
    return true;
  }

  // the body is about to change: the next face takes a new copy of the rest
  invalidate() { if (this.layer) this.layer.ready = false; this.tongueNear.clear(); this.warmCard = false; }
  private tongueNear = new Map<number, [number, number][]>();   // per point of a card's tongue its nearest vertices of Anny's tongue
  // a new body: the copy of the rest and the jaw follow it, and the face shows again (a face at rest waits for the next
  // face, so that a slider edit costs nothing more)
  bodyChanged() {
    if (!this.layer) return;
    if (this.atRest()) { this.host.card()?.followFace(null); return; }
    this.apply();
  }
  private atRest() {
    const s = this.state;
    return !Object.keys(s.actions).length && !(s.gaze && (s.gaze.yaw || s.gaze.pitch)) && !(this.blinkW && this.blinkW.some(Boolean));
  }
  // the card's eye targets (card/look.ts): their shapes on the fine body (with the change of their normals), and their
  // weights
  blinkShapes(shapes: { idx: Uint32Array; d: Float32Array; n?: Float32Array }[] | null) {
    if (!this.layer) return;
    const list: Shape[] | null = shapes ? shapes.map((x) => ({ idx: x.idx, d: x.d, nIdx: x.n ? x.idx : undefined, dn: x.n })) : null;
    this.layer.setGroup('blink', list, true);
    this.blinkW = list ? new Float32Array(list.length) : null;
  }
  blinkWeights(w: Float32Array) {
    if (!this.blinkW || w.length !== this.blinkW.length) return;
    this.blinkW.set(w);
    // the frames of a blink take the quicker normals, and the eyes at rest the exact ones
    if (!this.applying) this.apply(false);
  }
  // a card came or went: its poses join the faces, its eyes play the blinks and squints
  cardChanged() {
    this.poses = null; this.poseKey = ''; this.warmCard = false;
    if (!this.layer) return;
    this.layer.setGroup('poses', null);
    // the card's pocket changes the triangles of the normals: a new region
    this.layer.at = new Int32Array(0); this.layer.ready = false;
    this.apply();
  }

  // the state shown
  private show(state: FaceState, exact = true) {
    const prevHead = this.state.head, prevGaze = this.state.gaze;
    this.state = state;
    this.apply(exact);
    const h = state.head || { turn: 0, tilt: 0, nod: 0 }, ph = prevHead || { turn: 0, tilt: 0, nod: 0 };
    if (h.turn !== ph.turn || h.tilt !== ph.tilt || h.nod !== ph.nod) this.host.head(h.turn, h.tilt, h.nod);
    void prevGaze;
  }

  // the weights of the rig's actions, of the card's poses and of the eye targets, on the body
  apply(exact = true) {
    const rig = this.rig, layer = this.layer;
    if (!rig || !layer) return;
    this.rough = !exact; this.lastApply = performance.now();
    const t0 = performance.now(), s = this.state, card = this.host.card();
    this.applying = true;
    try {
      const w = new Float32Array(rig.names.length);
      for (const [k, v] of Object.entries(s.actions)) { const i = rig.index.get(k); if (i !== undefined) w[i] = v; }
      // the gaze turns the eyes through Anny's eye-look actions (their lids follow), as far as they reach
      const turns = this.gazeWeights(w, s.gaze);
      // a card's own eye targets for the blinks and the squints
      if (card?.hasBlinks) {
        const val = (n: string) => { const i = rig.index.get(n)!; const x = w[i]; w[i] = 0; return x; };
        card.setFaceEyes({ L: val('eyeBlinkLeft'), R: val('eyeBlinkRight') }, { L: val('eyeSquintLeft'), R: val('eyeSquintRight') });
      }
      // a card's own mouth poses
      const pw = this.cardPoses(card);
      const groups: Record<string, Float32Array> = {};
      if (pw) groups.poses = pw;
      if (this.blinkW) groups.blink = this.blinkW;
      const wasReady = layer.ready, morph = this.host.morph();
      let skin: (v: number, out: number[]) => void;
      if (morph) {
        // the face on the GPU: the shapes' layers and weights, and the jaw's turn
        if (!wasReady) { layer.rebase(); this.jawLayer(morph); }
        const shown: { s: Shape; w: number }[] = [];
        for (let a = 0; a < rig.names.length; a++) if (w[a] && a !== rig.jaw) { const sh = rig.fineShape(a); layer.normalShape(sh); shown.push({ s: sh, w: w[a] }); }
        const after = new Set<Shape>();
        for (const [name, g] of layer.groups) {
          const gw = groups[name];
          if (!gw) continue;
          g.shapes.forEach((sh, i) => { if (gw[i]) { layer.normalShape(sh); shown.push({ s: sh, w: gw[i] }); if (g.after) after.add(sh); } });
        }
        const wj = rig.jaw >= 0 ? w[rig.jaw] : 0;
        // the frames of a fade write at most two new layers each; a still face writes all of them
        morph.show(shown, wj ? { w: wj, R: rig.turn(wj), p: rig.pivot } : null, exact ? Infinity : 2);
        this.waiting = morph.pending > 0;
        this.host.redraw();
        const P = this.host.body().pos, o = [0, 0, 0];
        const at = (v: number, out: number[]) => morph.at(v, P[v * 3], P[v * 3 + 1], P[v * 3 + 2], out);
        this.host.strands(at);
        skin = (v, out) => { morph.at(v, P[v * 3], P[v * 3 + 1], P[v * 3 + 2], o, after); out[0] = o[0] - P[v * 3]; out[1] = o[1] - P[v * 3 + 1]; out[2] = o[2] - P[v * 3 + 2]; };
      } else {
        const [lo, hi] = layer.apply(w, groups, exact);
        if (hi >= lo) this.host.upload(lo, hi);
        this.host.strands();
        skin = (v, out) => layer.offsetOf(v, out);
      }
      if (!wasReady) this.timing.rebase = layer.rebaseMs;
      this.host.eyes(turns);
      const tf = performance.now();
      if (card) card.followFace(this.motion(w, pw, skin));
      this.timing.follow = performance.now() - tf;
      this.timing.apply = performance.now() - t0; this.timing.count++;
    } finally { this.applying = false; }
  }

  // the eye-look actions that a gaze plays (added to w), and the turn of each eye: the actions' turns, and the rest of the
  // gaze past their reach
  private gazeWeights(w: Float32Array, gaze?: { yaw: number; pitch: number }): Record<string, Float64Array> {
    const rig = this.rig!, m = rig.data.meta, out: Record<string, Float64Array> = {};
    const want = { yaw: gaze?.yaw || 0, pitch: gaze?.pitch || 0 };
    for (const eye of ['l', 'r']) {
      const rv = [0, 0, 0];
      let yaw = want.yaw * DEG, pitch = want.pitch * DEG;
      // the actions of this eye by the direction they turn its gaze (+Z): the share of the gaze that each one carries
      for (const [name, turnsOf] of Object.entries(m.eye_turns)) {
        const r = turnsOf[eye], i = rig.index.get(name);
        if (!r || i === undefined) continue;
        const R = rotvecMatrix(r[0], r[1], r[2]);
        const gx = R[2], gy = R[5];   // the turned gaze's x (to the figure's left) and y (up)
        const reachYaw = Math.atan2(gx, R[8]), reachPitch = Math.atan2(gy, R[8]);
        if (Math.abs(reachYaw) > Math.abs(reachPitch)) {
          if (yaw * reachYaw > 0) { const k = Math.min(1, yaw / reachYaw); w[i] = Math.min(1, w[i] + k); yaw -= k * reachYaw; }
        } else if (pitch * reachPitch > 0) { const k = Math.min(1, pitch / reachPitch); w[i] = Math.min(1, w[i] + k); pitch -= k * reachPitch; }
      }
      for (const [name, turnsOf] of Object.entries(m.eye_turns)) {
        const r = turnsOf[eye], i = rig.index.get(name);
        if (!r || i === undefined || !w[i]) continue;
        for (let k = 0; k < 3; k++) rv[k] += w[i] * r[k];
      }
      // the rest of the gaze: a yaw about the head's up and a pitch about its side axis
      const A = rotvecMatrix(rv[0], rv[1], rv[2]), Y = rotvecMatrix(0, yaw, 0), X = rotvecMatrix(-pitch, 0, 0);
      out[eye] = mul3(mul3(Y, X), A);
    }
    return out;
  }

  // the card's poses as fine shapes of the layer, and their weights from the state
  private cardPoses(card: FaceCard | null): Float32Array | null {
    const rig = this.rig!, layer = this.layer!;
    const P = card?.poses() || null;
    const key = P ? P.names.join('|') + ':' + P.level : '';
    if (key !== this.poseKey) {
      this.poseKey = key; this.poses = P;
      if (P) {
        const shapes = P.coarse.map((c, k) => {
          const adds: ({ ids: ArrayLike<number>; values: ArrayLike<number> } | null)[] = [null, null];
          if (P.level > 0 && P.rows[k]) adds[P.level - 1] = { ids: P.rows[k].idx, values: P.rows[k].d };
          const ids: number[] = [], vals: number[] = [];
          for (let i = 0; i < c.idx.length; i++) if (c.idx[i] < rig.body.nBody) { ids.push(c.idx[i]); vals.push(c.d[i * 3], c.d[i * 3 + 1], c.d[i * 3 + 2]); }
          const r = rig.sparse.apply(ids, vals, 3, adds);
          return { idx: r.ids, d: r.values };
        });
        layer.setGroup('poses', shapes);
      } else layer.setGroup('poses', null);
    }
    if (!P) return null;
    const w = new Float32Array(P.names.length);
    P.names.forEach((n, k) => { w[k] = this.state.actions[POSE_PREFIX + n] || 0; });
    return w;
  }

  // the face's motion for a card's parts
  // the jaw's corrective of the current body and its weights, in the GPU's layer 0
  private jawLayer(morph: FaceMorph) {
    const rig = this.rig!, layer = this.layer!;
    if (rig.jawCorrective) layer.normalShape(rig.jawCorrective);
    morph.setJaw(rig.jawCorrective, rig.fineJaw);
  }

  private motion(w: Float32Array, pw: Float32Array | null, skin: (v: number, out: number[]) => void): FaceMotion {
    const rig = this.rig!, layer = this.layer!, m = rig.data.meta, host = this.host;
    const wj = rig.jaw >= 0 ? w[rig.jaw] : 0, R = rig.turn(wj), k = host.headScale(), p = rig.pivot;
    const tongue = m.tongue, rigid = Object.entries(tongue.motion).filter(([n, t]) => t.kind === 'rigid' && n !== m.jaw.action && w[rig.index.get(n) ?? -1]);
    const open = tongue.motion[m.jaw.action];
    const R1 = rig.R1;
    const rigidOffset = (t: any, p0: number[], out: number[], f: number) => {
      const r = t.rotation, tr = t.translation;
      for (let c = 0; c < 3; c++) out[c] += f * k * (r[c * 3] * p0[0] + r[c * 3 + 1] * p0[1] + r[c * 3 + 2] * p0[2] + tr[c] - p0[c]);
    };
    const lowerAt = (x: number, y: number, z: number, out: number[]) => {
      out[0] = out[1] = out[2] = 0;
      const p0 = host.toDefault([x, y, z]);
      for (const [n, t] of rigid) rigidOffset(t, p0, out, w[rig.index.get(n)!]);
      if (wj) {
        // the jaw's corrective at weight 1: what completes its full turn to the tongue's own motion in jawOpen
        const J = [0, 0, 0];
        if (open && open.kind === 'rigid') rigidOffset(open, p0, J, 1);
        const dx = x - p[0], dy = y - p[1], dz = z - p[2];
        const e = [J[0] - ((R1[0] - 1) * dx + R1[1] * dy + R1[2] * dz), J[1] - (R1[3] * dx + (R1[4] - 1) * dy + R1[5] * dz), J[2] - (R1[6] * dx + R1[7] * dy + (R1[8] - 1) * dz)];
        // c = R1^T e (the lower jaw follows the jaw fully)
        const q = new Float64Array([x + out[0], y + out[1], z + out[2]]);
        for (let c = 0; c < 3; c++) q[c] += wj * (R1[c] * e[0] + R1[3 + c] * e[1] + R1[6 + c] * e[2]);
        turnPoint(q, 0, R, p, 1);
        out[0] = q[0] - x; out[1] = q[1] - y; out[2] = q[2] - z;
      }
    };
    // the motion of the lower jaw is affine in the point: its matrix from four points, then three dot products a point
    const L = new Float64Array(12), o0 = [0, 0, 0], oi = [0, 0, 0];
    lowerAt(0, 0, 0, o0);
    for (let c = 0; c < 3; c++) {
      lowerAt(c === 0 ? 1 : 0, c === 1 ? 1 : 0, c === 2 ? 1 : 0, oi);
      for (let r = 0; r < 3; r++) L[r * 4 + c] = oi[r] - o0[r];
    }
    for (let r = 0; r < 3; r++) L[r * 4 + 3] = o0[r];
    const lower = (x: number, y: number, z: number, out: number[]) => {
      out[0] = L[0] * x + L[1] * y + L[2] * z + L[3]; out[1] = L[4] * x + L[5] * y + L[6] * z + L[7]; out[2] = L[8] * x + L[9] * y + L[10] * z + L[11];
    };
    // the tongue: the lower jaw's motion, and in an action that bends Anny's tongue the blend of its 4 nearest vertices
    const follow = Object.entries(tongue.motion).filter(([n, t]) => t.kind === 'follow' && w[rig.index.get(n) ?? -1]);
    const rest = tongue.rest, nt = rest.length / 3;
    const near = this.tongueNear;
    const tongueAt = (x: number, y: number, z: number, out: number[], key?: number) => {
      lower(x, y, z, out);
      if (!follow.length || !nt) return;
      let best = key === undefined ? undefined : near.get(key);
      if (!best) {
        const p0 = host.toDefault([x, y, z]);
        best = [];
        for (let i = 0; i < nt; i++) {
          const d = Math.hypot(rest[i * 3] - p0[0], rest[i * 3 + 1] - p0[1], rest[i * 3 + 2] - p0[2]);
          if (best.length < 4 || d < best[3][1]) { best.push([i, d]); best.sort((a, b) => a[1] - b[1]); if (best.length > 4) best.pop(); }
        }
        if (key !== undefined) near.set(key, best);
      }
      let sw = 0; for (const [, d] of best) sw += 1 / Math.max(d, 1e-5);
      for (const [n, t] of follow) {
        const f = w[rig.index.get(n)!] * k, off = (t as any).offsets as number[];
        for (const [i, d] of best) { const g = f / Math.max(d, 1e-5) / sw; for (let c = 0; c < 3; c++) out[c] += g * off[i * 3 + c]; }
      }
    };
    // the face's offsets of the coarse body (for a sculpt card's patch), made when asked
    let made: Float32Array | null = null;
    const coarse = () => {
      if (made) return made;
      const C = rig.body.coarse, V = C.slice();
      rig.coarseFace(w, V);
      made = new Float32Array(V.length);
      for (let i = 0; i < V.length; i++) made[i] = V[i] - C[i];
      const P = this.poses;
      if (P && pw) P.coarse.forEach((c, j) => { const f = pw[j]; if (f) for (let i = 0; i < c.idx.length; i++) for (let q = 0; q < 3; q++) made![c.idx[i] * 3 + q] += f * c.d[i * 3 + q]; });
      return made;
    };
    return { skin, region: layer.region, lower, tongue: tongueAt, jawTurn: R, coarse, poses: pw || new Float32Array(0) };
  }

  stats() {
    const l = this.layer;
    return { ready: !!l, actions: this.rig?.names.length || 0, region: l?.region.length || 0, outside: l?.outside || 0,
      jaw: this.rig ? { degrees: this.rig.data.meta.jaw.angle_degrees, pivot: this.rig.pivot } : null, timing: { ...this.timing }, state: this.get() };
  }
}

function clean(s: FaceState): FaceState {
  const actions: Record<string, number> = {};
  for (const [k, v] of Object.entries(s?.actions || {})) if (typeof v === 'number' && isFinite(v) && v !== 0) actions[k] = Math.min(1, Math.max(0, v));
  const out: FaceState = { actions };
  if (s?.gaze) out.gaze = { yaw: +s.gaze.yaw || 0, pitch: +s.gaze.pitch || 0 };
  if (s?.head) out.head = { turn: +s.head.turn || 0, tilt: +s.head.tilt || 0, nod: +s.head.nod || 0 };
  return out;
}

function mix(a: FaceState, b: FaceState, t: number): FaceState {
  const actions: Record<string, number> = {};
  for (const k of new Set([...Object.keys(a.actions), ...Object.keys(b.actions)])) {
    const v = (a.actions[k] || 0) + ((b.actions[k] || 0) - (a.actions[k] || 0)) * t;
    if (v) actions[k] = v;
  }
  const out: FaceState = { actions };
  const ga = a.gaze || { yaw: 0, pitch: 0 }, gb = b.gaze || { yaw: 0, pitch: 0 };
  if (a.gaze || b.gaze) out.gaze = { yaw: ga.yaw + (gb.yaw - ga.yaw) * t, pitch: ga.pitch + (gb.pitch - ga.pitch) * t };
  const ha = a.head || { turn: 0, tilt: 0, nod: 0 }, hb = b.head || { turn: 0, tilt: 0, nod: 0 };
  if (a.head || b.head) out.head = { turn: ha.turn + (hb.turn - ha.turn) * t, tilt: ha.tilt + (hb.tilt - ha.tilt) * t, nod: ha.nod + (hb.nod - ha.nod) * t };
  return out;
}

function mul3(A: Float64Array, B: Float64Array): Float64Array {
  const out = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { let s = 0; for (let k = 0; k < 3; k++) s += A[i * 3 + k] * B[k * 3 + j]; out[i * 3 + j] = s; }
  return out;
}

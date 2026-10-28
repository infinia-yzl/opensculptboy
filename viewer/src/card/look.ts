// OpenSculptBoy
// Apache License, Version 2.0
//
// A character card's look in the page (docs/viewer.md, "A card's look"): what the GLB export of the card shows, drawn
// with the page's lights. The card's local changes and the offsets of its detail level shape the body (body.ts); its
// baked maps colour the skin through the UVs of anny's atlas, the head tile's maps on the head's island; its baked
// eyes take the single-surface eyeball of opensculptboy.eyes on the spheres of the fitted eyeballs, looking where the
// reference looks; the reference's lashes, teeth, inner mouth, hair and brows ride on the head, the grooms with the
// materials of the exports; and the eyes blink as the GLB export's targets close them, on a timer and from the panel.
// The plan of what the page takes lives in card/data.ts. Nothing of the user's files is stored or sent: the page keeps them in memory until it closes.

import type * as THREE_NS from 'three';
import type { AnnyBody } from '../body.ts';
import { sphereFit, vertexNormals } from '../body.ts';
import type { FileSet } from '../wearables/files.ts';
import { readSafetensors } from '../wearables/package.ts';
import { BLEND_SHADOW_CUTOFF, GARMENT_DEPTH_FS, GARMENT_DEPTH_VS, GARMENT_FS, GARMENT_VS } from '../wearables/shading.ts';
import { BLINK_SECONDS, blinkCurve, blinksOf, blinkWeights, cardPlan, eyeAxes, groomOf, irisRadius, mouthPosesOf, outerRadius, pagePoints, pageVectors, partOf, tonguePoints,
  uvTangents, type Blinks, type CardPlan, type Frame, type Groom, type MouthPoses, type Part } from './data.ts';
import type { CardPoses, FaceMotion } from '../face/expressions.ts';
import { nearestVertices } from '../face/rig.ts';

// what the card's look needs from the renderer (main.ts)
export interface CardHost {
  THREE: typeof THREE_NS;
  scene: THREE_NS.Scene;
  opaque: any[];
  skeleton(): THREE_NS.Skeleton;
  uniforms: Record<string, any>;
  depth: THREE_NS.ShaderMaterial;
  envDefines(): Record<string, string>;
  toMRT(fs: string): string;
  body(): AnnyBody;
  geometry(): THREE_NS.BufferGeometry;
  manifest(): any;
  buffer(name: string): ArrayBuffer | null;          // a copy of a buffer of the model data (card_coarse, card_rows_*, card_patch_*)
  frame(): Frame;
  headBone(): number;
  skinMaterial(defines: Record<string, string>, uniforms: Record<string, any>): THREE_NS.ShaderMaterial;
  setSkinMaterial(m: THREE_NS.ShaderMaterial | null): void;   // null: the page's own skin
  eyeMaterial(uniforms: Record<string, any>): THREE_NS.ShaderMaterial;
  eyes(): any[];                                     // the eyes of the rig: { mesh, u, side, shape, card }
  setPageParts(on: { hair: boolean; brows: boolean; lashes: boolean } | null): void;   // null: as the look has them
  setBodyCard(c: { local: Record<string, number> | null; coarse: Float32Array | null; levels: (Float32Array | null)[]; pageDetail: boolean; hide: boolean } | null): void;
  changed(): void;
  // the eye targets on the body: sparse shapes of the fine body (vertices, position offsets and normal offsets, page
  // frame) and their weights; null takes them off
  blinkShapes(shapes: BodyShape[] | null): void;
  blinkWeights(w: Float32Array): void;
  moved(): void;                                      // a frame of a blink: the picture and the shadows again
  faceChanged(): void;                                // the card's faces came or went (face/expressions.ts)
}

export interface BodyShape { idx: Uint32Array; d: Float32Array; n: Float32Array }
type PartKind = 'lashes' | 'mouth' | 'teeth' | 'hair' | 'brows';

// face: the offsets of the face shown (face/expressions.ts); bind: the fine vertex under each point (the lashes and the
// brows follow the skin); arch and tongue: the points of the lower jaw and of the tongue (the teeth and the inner mouth)
interface PartMesh { name: string; kind: PartKind; mesh: any; geo: THREE_NS.BufferGeometry; rest: Float32Array; normals: Float32Array; materials: any[]; depth: any[]; blink: Float32Array | null;
  face: Float32Array | null; bind: Uint32Array | null; arch: Uint8Array | null; tongue: Uint8Array | null; inner: boolean }

// the blinks of a card: its eye targets, their offsets on the page (the parts' and the patch's in the page frame), the
// state of the panel and of the timer
interface BlinkState {
  data: Blinks; ready: boolean; auto: boolean; close: number; squint: number; start: number; next: number; active: boolean; vertices: number;
  // the face's closure and squint of each eye (face/expressions.ts), which the blinks start from as well
  face: { close: { L: number; R: number }; squint: { L: number; R: number } };
}

const ANISOTROPY = 8;
// the share of the key light that the skin around a point reflects onto it (shading.ts, CARD_SKIN), set against the
// GLB's renders in Blender (Cycles) on the page's cameras and lights
export const CARD_BOUNCE = 0.06;

export class CardLook {
  host: CardHost;
  plan: CardPlan | null = null;
  notes: string[] = [];
  failures = 0;
  private textures = new Map<string, THREE_NS.Texture>();
  private images = new Map<string, ImageBitmap>();
  private skin: THREE_NS.ShaderMaterial | null = null;
  private parts: PartMesh[] = [];
  private partData: { name: string; part: Part; kind: PartKind }[] = [];
  private blink: BlinkState | null = null;
  private mouthPoses: MouthPoses | null = null; private cardPoses: CardPoses | null = null;
  private face: FaceMotion | null = null; private patchFace: Float32Array | null = null;
  private eyeState: { saved: any[]; meshes: any[] } | null = null;
  private iris: Record<string, number> = {};
  private placedHead: { joint: number[]; scale: number } | null = null;
  private placed = false;
  private coarse0: Float32Array | null = null;   // the level-0 offsets of the coarse body (page frame)
  private patchBorder = 0;
  private patch: null | { rows: Uint32Array; start: Uint32Array; cols: Uint32Array; values: Float32Array; uvs: Float32Array; near: Uint32Array;
    triangles: Uint32Array; offsets: Float32Array; mesh: any; geo: THREE_NS.BufferGeometry | null } = null;   // the card's sliders shape the body: the eyes and the parts may take their places
  timing: Record<string, number> = {};

  constructor(host: CardHost) { this.host = host; }
  get active() { return !!this.plan; }

  // ---------------------------------------------------------------- loading
  // the card's files: its body (local changes and detail offsets, set on the body at once), its maps and its parts;
  // place() then puts them on the card's body
  async load(card: any, files: FileSet, folder: string): Promise<{ notes: string[]; failures: number }> {
    this.unload(false);
    const t0 = performance.now();
    const plan = cardPlan(card), notes = [...plan.notes];
    let failures = 0;
    const fail = (note: string, err?: unknown) => { notes.push(note); failures++; console.error(`Card: ${note}`, ...(err instanceof Error ? [err] : [])); };
    const frame = this.host.frame(), body = this.host.body(), man = this.host.manifest();
    const topology = ['anny', 'sculpt'].includes(plan.topology) ? plan.topology : 'anny';
    if (topology !== plan.topology) notes.push(`the page has no rows of the ${plan.topology} topology; it reads the card as anny's`);
    // the detail layer: level 0 on the coarse body (the eyes and the wearables read it), and the level shown on the
    // fine body
    let coarse: Float32Array | null = null;
    const levels: (Float32Array | null)[] = [null, null];
    const u32 = (name: string) => { const b = this.host.buffer(name); return b ? new Uint32Array(b) : null; };
    const coarseRows = u32('card_coarse');
    const read = async (rel: string) => readSafetensors(await files.bytes(rel, folder)).tensors.offsets?.values;
    if (plan.levels[0] !== undefined) {
      try {
        const v = await read(plan.levels[0]);
        if (!v || !coarseRows) throw new Error('the offsets file holds no offsets');
        let max = 0; for (let i = 0; i < coarseRows.length; i++) max = Math.max(max, coarseRows[i]);
        if (v.length < (max + 1) * 3) throw new Error(`the offsets hold ${v.length / 3} rows; the body has ${max + 1}`);
        coarse = pageVectors(v, coarseRows, coarseRows.length, frame);
      } catch (err: any) { fail(`detail level 0: ${err.message || err}`, err); }
    }
    if (plan.shown > 0) {
      const rows = u32(`card_rows_${topology}_${plan.shown}`);
      try {
        const v = await read(plan.levels[plan.shown]);
        if (!v || !rows) throw new Error('the page or the file holds no rows of this level');
        let max = 0; for (let i = 0; i < rows.length; i++) max = Math.max(max, rows[i]);
        if (v.length < (max + 1) * 3) throw new Error(`the offsets hold ${v.length / 3} rows; level ${plan.shown} needs ${max + 1}`);
        const lv = levels[plan.shown - 1] = pageVectors(v, rows, rows.length, frame);
        // the rows whose stencil a patch of the topology changed: the move to the closest point of the card's surface
        // and the blend of its three rows' offsets (records of the page row, three rows, three weights and the move)
        const fix = this.host.buffer(`card_fix_${topology}_${plan.shown}`);
        if (fix) {
          const f = new Float32Array(fix), ids = new Uint32Array(fix);
          for (let q = 0; q < f.length; q += 10) {
            const blend = pageVectors(v, ids.subarray(q + 1, q + 4), 3, frame), o = ids[q] * 3;
            for (let c = 0; c < 3; c++) lv[o + c] = f[q + 7 + c] + f[q + 4] * blend[c] + f[q + 5] * blend[3 + c] + f[q + 6] * blend[6 + c];
          }
        }
        // a sculpt card: the faces of the patches, drawn in place of the faces of anny that they replace
        if (topology === 'sculpt') this.loadPatch(v, plan.shown, frame);
        const fb = man?.card?.levels?.[topology]?.fallback?.[String(plan.shown)];
        if (fb) notes.push(`${fb} rows of level ${plan.shown} take the nearest row of the ${topology} topology`);
      } catch (err: any) { fail(`detail level ${plan.shown}: ${err.message || err}`, err); plan.shown = 0; }
    }
    const hasDetail = !!coarse || levels.some(Boolean);
    this.coarse0 = coarse;
    const unknown = Object.keys(plan.localChanges).filter((k) => !(body.local?.tables.labels || []).includes(k));
    if (unknown.length) notes.push(`the page has no local changes ${unknown.slice(0, 4).join(', ')}${unknown.length > 4 ? ' and others' : ''}`);
    if (Object.keys(plan.localChanges).length && !body.local) notes.push('the page has no local changes');
    this.timing.offsets = performance.now() - t0;
    // the maps: the images decode while the rest loads
    const t1 = performance.now();
    await Promise.all(Object.entries(plan.textures).map(async ([slot, rel]) => {
      try {
        const bytes = await files.bytes(rel, folder);
        const bmp = await createImageBitmap(new Blob([bytes as any], { type: bytes[0] === 0x89 ? 'image/png' : 'image/jpeg' }), { imageOrientation: 'none' as any, colorSpaceConversion: 'none' as any, premultiplyAlpha: 'none' as any });
        this.images.set(slot, bmp);
      } catch (err: any) { fail(`${slot}: ${err.message || err}`, err); }
    }));
    if (plan.textures.base_color && !this.images.has('base_color')) notes.push('the page keeps its own skin');
    this.timing.textures = performance.now() - t1;
    // the parts of the bake
    const parts: [string, string | undefined, 'lashes' | 'mouth' | 'teeth'][] = [['lashes', plan.lashes?.mesh, 'lashes'], ['mouth', plan.mouth?.mesh, 'mouth'], ['teeth', plan.teeth?.mesh, 'teeth']];
    for (const [name, rel, kind] of parts) {
      if (!rel) continue;
      try { this.partData.push({ name, part: partOf(readSafetensors(await files.bytes(rel, folder)).tensors), kind }); }
      catch (err: any) { fail(`${name}: ${err.message || err}`, err); }
    }
    for (const [slot, rel] of [['lashes', plan.lashes?.baseColor], ['mouth_base_color', plan.mouth?.baseColor], ['mouth_normal', plan.mouth?.normal], ['mouth_roughness', plan.mouth?.roughness]] as [string, string | null | undefined][]) {
      if (!rel) continue;
      try {
        const bytes = await files.bytes(rel, folder);
        this.images.set(slot, await createImageBitmap(new Blob([bytes as any]), { imageOrientation: 'none' as any, colorSpaceConversion: 'none' as any, premultiplyAlpha: 'none' as any }));
      } catch (err: any) { fail(`${slot}: ${err.message || err}`, err); }
    }
    if (plan.mouth) notes.push('the page draws the reference\'s inner mouth over anny\'s own');
    // the reference's hair and brows, and their textures by material and slot
    for (const kind of ['hair', 'brows'] as const) {
      const g = plan[kind];
      if (!g) continue;
      try {
        const st = readSafetensors(await files.bytes(g.mesh, folder));
        this.partData.push({ name: kind, part: groomOf(st.tensors, st.metadata), kind });
      } catch (err: any) { fail(`${kind}: ${err.message || err}`, err); continue; }
      for (const [material, slots] of Object.entries(g.textures)) for (const [slot, rel] of Object.entries(slots)) {
        try {
          const bytes = await files.bytes(rel, folder);
          this.images.set(`${kind}:${material}:${slot}`, await createImageBitmap(new Blob([bytes as any]), { imageOrientation: 'none' as any, colorSpaceConversion: 'none' as any, premultiplyAlpha: 'none' as any }));
        } catch (err: any) { fail(`${kind} ${material} ${slot}: ${err.message || err}`, err); }
      }
    }
    // the exports' eye targets, which the page plays on a timer
    if (plan.blinks) {
      try {
        const st = readSafetensors(await files.bytes(plan.blinks, folder));
        const data = blinksOf(st.tensors, st.metadata);
        if (data.level !== plan.shown) notes.push(`the blinks hold level ${data.level}; the page shows level ${plan.shown} and plays them on the parts alone`);
        const q = typeof location !== 'undefined' ? new URLSearchParams(location.search) : null;
        const auto = !(q && (q.has('shot') || q.get('blink') === 'off'));
        this.blink = { data, ready: false, auto, close: 0, squint: 0, start: -1, next: performance.now() + 1500, active: false, vertices: 0,
          face: { close: { L: 0, R: 0 }, squint: { L: 0, R: 0 } } };
      } catch (err: any) { fail(`blinks: ${err.message || err}`, err); }
    }
    // the reference's mouth poses, which the page plays as faces
    if (plan.mouthPoses) {
      try {
        const st = readSafetensors(await files.bytes(plan.mouthPoses, folder));
        this.mouthPoses = mouthPosesOf(st.tensors, st.metadata);
      } catch (err: any) { fail(`mouth poses: ${err.message || err}`, err); }
    }
    // the body: the card's local changes and its detail offsets, with the page's own detail layers left out
    this.plan = plan; this.notes = notes; this.failures = failures;
    this.host.setBodyCard({ local: Object.keys(plan.localChanges).length ? plan.localChanges : null, coarse, levels, pageDetail: !hasDetail, hide: !!this.patch });
    this.timing.total = performance.now() - t0;
    return { notes, failures };
  }

  // ---------------------------------------------------------------- the look on the card's body
  // after the card's sliders: the skin's material and tangents, the eyes and the parts on the body
  place() {
    if (!this.plan) return;
    const t0 = performance.now(), plan = this.plan, T = this.host.THREE;
    this.placed = true;
    // the reference's hair takes the place of the page's strand hair, which stays for a card without it; the brows of a
    // baked skin are in its maps (or absent), and the reference's brows and lashes replace the page's
    const has = (k: PartKind) => this.partData.some((p) => p.kind === k);
    this.host.setPageParts({ hair: !has('hair'), brows: !this.images.has('base_color') && !has('brows'), lashes: !has('lashes') });
    if (this.images.has('base_color') && !this.skin) {
      const defines: Record<string, string> = { CARD_SKIN: '' };
      const tex = (slot: string, srgb: boolean) => this.texture(slot, srgb);
      const xf = (this.host.manifest()?.card?.head_tile || {}), box = xf[plan.topology] || xf.anny;
      const uniforms: Record<string, any> = {
        uCardBase: { value: tex('base_color', true) }, uCardNormal: { value: null }, uCardRough: { value: null },
        uHeadBase: { value: null }, uHeadNormal: { value: null }, uHeadRough: { value: null },
        uHeadXf: { value: new T.Vector4(box ? box.scale[0] : 1, box ? box.scale[1] : 1, box ? box.offset[0] : 0, box ? box.offset[1] : 0) },
        uCardNormalScale: { value: plan.skin.normalScale }, uCardRoughK: { value: plan.skin.roughness },
        uCardF0: { value: Math.min(1, ((plan.skin.ior - 1) / (plan.skin.ior + 1)) ** 2) * plan.skin.specular },
        uCardSSS: { value: Math.min(1, Math.max(0, plan.skin.subsurface)) },
      };
      if (this.images.has('normal')) { defines.CARD_NORMAL = ''; uniforms.uCardNormal.value = tex('normal', false); }
      if (this.images.has('roughness')) { defines.CARD_ROUGH = ''; uniforms.uCardRough.value = tex('roughness', false); }
      if (box && this.images.has('head_base_color')) {
        defines.CARD_HEAD = '';
        uniforms.uHeadBase.value = tex('head_base_color', true);
        uniforms.uHeadNormal.value = this.images.has('head_normal') ? tex('head_normal', false) : uniforms.uCardNormal.value || tex('base_color', false);
        uniforms.uHeadRough.value = this.images.has('head_roughness') ? tex('head_roughness', false) : uniforms.uCardRough.value || tex('base_color', false);
      }
      if (!uniforms.uCardNormal.value) uniforms.uCardNormal.value = uniforms.uCardBase.value;
      if (!uniforms.uCardRough.value) uniforms.uCardRough.value = uniforms.uCardBase.value;
      for (const k of ['uHeadBase', 'uHeadNormal', 'uHeadRough']) if (!uniforms[k].value) uniforms[k].value = uniforms.uCardBase.value;
      uniforms.uCardBounce = { value: CARD_BOUNCE };
      this.skin = this.host.skinMaterial(defines, uniforms);
      this.host.setSkinMaterial(this.skin);
    }
    // the pocket leaves the picture and the shadow maps while the patch shows
    this.host.uniforms.uCardHide.value = this.patch && this.skin ? 1 : 0;
    this.bodyChanged();
    this.timing.place = performance.now() - t0;
    this.host.faceChanged();
  }

  // the body changed (a slider, the card's own body): the tangents of the normal maps, the eyes and the parts follow
  bodyChanged() {
    if (!this.plan || !this.placed) return;
    const t0 = performance.now();
    if (this.blink) this.blink.ready = false;   // the body's shapes follow the new body when the eyes move next
    for (const p of this.parts) p.bind = null;   // the parts bind to the new body's skin when the face moves next
    this.placeEyes();
    this.placeParts();
    this.placePatch();
    if (this.blink && (this.blink.close || this.blink.squint)) this.showEyes(this.blink.close);
    this.host.changed();
    this.timing.follow = performance.now() - t0;
  }

  // ---------------------------------------------------------------- eyes
  private placeEyes() {
    const plan = this.plan!, layout = this.images.get('eye_layout');
    if (!plan.eyes || !layout) return;
    const T = this.host.THREE, body = this.host.body(), frame = this.host.frame(), eyes = this.host.eyes();
    if (!this.eyeState) this.eyeState = { saved: eyes.map((e) => ({ geo: e.mesh.geometry, mat: e.mesh.material, u: e.u, rot0: e.mesh.userData.rot0 })), meshes: [] };
    if (!this.iris.L) this.measureIris(layout);
    const tex = this.texture('eye_layout', true);
    eyes.forEach((e, i) => {
      const side = e.side === 'l' ? 'L' : 'R', [a, b] = body.eyeRange[e.side];
      const [c, r] = sphereFit(body.coarse, a, b);
      // the eye's axes (columns +X, up, gaze) in the page's frame
      const A = eyeAxes(plan.eyes!.gaze?.[side] || null), R = frame.rotation;
      const P = new Array(9).fill(0);
      for (let i2 = 0; i2 < 3; i2++) for (let j = 0; j < 3; j++) for (let m = 0; m < 3; m++) P[i2 * 3 + j] += R[i2 * 3 + m] * A[m * 3 + j];
      const rAnny = r / frame.scale;
      const iris = Math.min(this.iris[side] || 0.39, 0.8);
      const cornea = Math.max(plan.eyes!.corneaRadiusMm / 1000 / rAnny, iris * 1.05);
      const { bulge, sphere } = this.clearance(c, r, P, iris, cornea);
      const geo = eyeGeometry(T, c, r * sphere, P, iris, cornea, bulge);
      let mat = this.eyeState!.meshes[i]?.material;
      if (!mat) {
        const u = { uEyeRot: { value: new T.Matrix3() }, uEyeLayout: { value: tex }, uEyeLeft: { value: side === 'L' ? 1 : 0 },
          uScleraRough: { value: 0.45 }, uCoatRough: { value: plan.eyes!.corneaRoughness }, uCoatF0: { value: ((plan.eyes!.corneaIor - 1) / (plan.eyes!.corneaIor + 1)) ** 2 } };
        mat = this.host.eyeMaterial(u);
        this.eyeState!.meshes[i] = { material: mat };
      }
      const old = e.mesh.geometry;
      e.mesh.geometry = geo; e.mesh.material = mat; e.u = mat.uniforms; e.card = true; e.shape.identity();
      e.mesh.userData.rot0 = new T.Matrix3().set(P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7], P[8]);
      if (old !== this.eyeState!.saved[i].geo) old.dispose();
      e.card = { iris, cornea, bulge, sphere, radius: r, center: c };
    });
  }

  // the iris of each baked eye: the limbus where the rings of the layout turn bright (data.ts irisRadius)
  private measureIris(img: ImageBitmap) {
    const w = 512, h = 256;
    try {
      const cv = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : Object.assign(document.createElement('canvas'), { width: w, height: h });
      const ctx = (cv as any).getContext('2d');
      ctx.drawImage(img, 0, 0, w, h);
      const px = ctx.getImageData(0, 0, w, h).data;
      const lum = (u: number, v: number) => {
        const x = Math.min(w - 1, Math.max(0, Math.floor(u * w))), y = Math.min(h - 1, Math.max(0, Math.floor(v * h))), o = (y * w + x) * 4;
        return 0.2126 * px[o] + 0.7152 * px[o + 1] + 0.0722 * px[o + 2];
      };
      this.iris = { L: irisRadius(lum, true), R: irisRadius(lum, false) };
    } catch (e) { this.iris = { L: 0.39, R: 0.39 }; }
  }

  // the bulge and the eyeball's scale that keep the lids clear of the eye (opensculptboy.eyes.geometry.bulge_clearance):
  // the skin near the eye facing out, within 80 degrees of the gaze and outside the fitted sphere, stays 0.08 mm out
  private clearance(c: number[], r: number, A: number[], iris: number, cornea: number) {
    const body = this.host.body(), P = body.pos, N = body.nrm, n = body.nWeld, frame = this.host.frame();
    const clear = 0.08 / 1000 * frame.scale / r, dist: number[] = [], theta: number[] = [];
    for (let v = 0; v < n; v++) {
      const dx = P[v * 3] - c[0], dy = P[v * 3 + 1] - c[1], dz = P[v * 3 + 2] - c[2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > (1.35 * r) ** 2 || d2 < r * r) continue;
      const d = Math.sqrt(d2);
      if ((N[v * 3] * dx + N[v * 3 + 1] * dy + N[v * 3 + 2] * dz) <= 0) continue;
      const z = (dx * A[2] + dy * A[5] + dz * A[8]) / d;
      const th = Math.acos(Math.min(1, Math.max(-1, z)));
      if (th >= 80 * Math.PI / 180) continue;
      dist.push(d / r); theta.push(th);
    }
    if (!dist.length) return { bulge: 1, sphere: 1 };
    const allowed = Math.max(2, Math.floor(0.01 * dist.length));
    if (allowed >= dist.length) return { bulge: 1, sphere: 1 };
    const sorted = dist.map((d) => d / (1 + clear)).sort((x, y) => x - y);
    const sphere = Math.min(1, Math.max(0.97, sorted[allowed]));
    const fits = (b: number) => { let k = 0; for (let i = 0; i < dist.length; i++) if (dist[i] < sphere * outerRadius(theta[i], iris, cornea, b) + clear) k++; return k <= allowed; };
    if (fits(1)) return { bulge: 1, sphere };
    let lo = 0, hi = 1;
    for (let i = 0; i < 30; i++) { const mid = 0.5 * (lo + hi); if (fits(mid)) lo = mid; else hi = mid; }
    return { bulge: lo, sphere };
  }

  // ---------------------------------------------------------------- the sculpt patch
  private loadPatch(offsets: ArrayLike<number>, level: number, frame: Frame) {
    const b = (name: string) => this.host.buffer(`card_patch_${level}_${name}`);
    const rows = b('rows'), start = b('start'), cols = b('cols'), values = b('values'), uvs = b('uvs'), near = b('near'), tris = b('triangles');
    if (!rows || !start || !cols || !values || !uvs || !near || !tris) return;
    const R = new Uint32Array(rows);
    let max = 0; for (let i = 0; i < R.length; i++) max = Math.max(max, R[i]);
    if (offsets.length < (max + 1) * 3) return;
    this.patch = { rows: R, start: new Uint32Array(start), cols: new Uint32Array(cols), values: new Float32Array(values), uvs: new Float32Array(uvs),
      near: new Uint32Array(near), triangles: new Uint32Array(tris), offsets: pageVectors(offsets, R, R.length, frame), mesh: null, geo: null };
  }

  // the patch's vertices on the current body: its blend of the coarse body without the level-0 offsets, plus the
  // card's offsets of the level; the other fields of each vertex come from the nearest vertex of the page's body
  private placePatch() {
    const p = this.patch;
    if (!p || !this.skin) return;
    const T = this.host.THREE, body = this.host.body(), C = body.coarse, O = this.coarse0, n = p.rows.length;
    if (!p.geo) {
      const page = this.host.geometry().attributes as any, geo = new T.BufferGeometry();
      const copy = (name: string, size: number) => {
        const a = page[name];
        if (!a) return;
        const src = a.array, out = new (src.constructor as any)(n * size);
        for (let i = 0; i < n; i++) for (let k = 0; k < size; k++) out[i * size + k] = src[p.near[i] * size + k];
        geo.setAttribute(name, new T.BufferAttribute(out, size, a.normalized));
      };
      for (const [name, size] of [['attrA', 4], ['attrB', 4], ['attrC', 4], ['attrW', 4], ['attrV', 4], ['rest', 3], ['skinIndex', 4], ['skinWeight', 4], ['skinIndex2', 4], ['skinWeight2', 4]] as [string, number][]) copy(name, size);
      const uv = new Uint16Array(n * 4), tile = page.cardUV.array as Uint16Array;
      for (let i = 0; i < n; i++) { uv[i * 4] = Math.round(p.uvs[i * 2] * 65535); uv[i * 4 + 1] = Math.round(p.uvs[i * 2 + 1] * 65535); uv[i * 4 + 2] = tile[p.near[i] * 4 + 2]; }
      geo.setAttribute('cardUV', new T.BufferAttribute(uv, 4, true));
      geo.setAttribute('position', new T.BufferAttribute(new Float32Array(n * 3), 3));
      geo.setAttribute('normal', new T.BufferAttribute(new Float32Array(n * 3), 3));
      geo.setAttribute('nsmooth', new T.BufferAttribute(new Float32Array(n * 3), 3));
      geo.setIndex(new T.BufferAttribute(p.triangles, 1));
      const mesh = new T.SkinnedMesh(geo, this.skin);
      mesh.bind(this.host.skeleton(), new T.Matrix4());
      mesh.frustumCulled = false; mesh.name = 'card patch'; mesh.userData.depthMat = this.host.depth;
      this.host.scene.add(mesh); this.host.opaque.push(mesh);
      p.geo = geo; p.mesh = mesh;
    }
    const P = p.geo.attributes.position.array as Float32Array;
    for (let i = 0; i < n; i++) {
      let x = 0, y = 0, z = 0;
      for (let e = p.start[i]; e < p.start[i + 1]; e++) {
        const c = p.cols[e] * 3, w = p.values[e];
        x += w * (C[c] - (O ? O[c] : 0)); y += w * (C[c + 1] - (O ? O[c + 1] : 0)); z += w * (C[c + 2] - (O ? O[c + 2] : 0));
      }
      P[i * 3] = x + p.offsets[i * 3]; P[i * 3 + 1] = y + p.offsets[i * 3 + 1]; P[i * 3 + 2] = z + p.offsets[i * 3 + 2];
    }
    // the patch's border (a vertex within half a millimetre of its nearest page vertex, where the page's skin shows)
    // takes the place and the normal of the page's skin, so that the two meet and the shading runs on across them
    const BP = body.pos, near2 = (0.0005 * this.host.frame().scale) ** 2, border: number[] = [];
    const hidden = (this.host.geometry().attributes as any).cardUV.array as Uint16Array;
    for (let i = 0; i < n; i++) {
      const o = i * 3, q = p.near[i] * 3, dx = P[o] - BP[q], dy = P[o + 1] - BP[q + 1], dz = P[o + 2] - BP[q + 2];
      const shown = hidden[p.near[i] * 4 + 3] === 0;
      if (shown && dx * dx + dy * dy + dz * dz < near2) { P[o] = BP[q]; P[o + 1] = BP[q + 1]; P[o + 2] = BP[q + 2]; border.push(i); }
    }
    (p as any).base = P.slice(); (p as any).border = border;
    this.patchFace = null;
    this.patchNormals();
    this.patchBorder = border.length;
  }

  // the patch's normals, its border taking the page skin's
  private patchNormals() {
    const p = this.patch!, P = p.geo!.attributes.position.array as Float32Array, BN = this.host.body().nrm, border: number[] = (p as any).border || [];
    const N = p.geo!.attributes.normal.array as Float32Array;
    vertexNormals(P, p.triangles, N);
    for (const i of border) { const o = i * 3, q = p.near[i] * 3; N[o] = BN[q]; N[o + 1] = BN[q + 1]; N[o + 2] = BN[q + 2]; }
    (p.geo!.attributes.nsmooth.array as Float32Array).set(N);
    p.geo!.attributes.position.needsUpdate = true; p.geo!.attributes.normal.needsUpdate = true; p.geo!.attributes.nsmooth.needsUpdate = true;
    p.geo!.computeBoundingSphere();
  }

  // ---------------------------------------------------------------- the parts
  private placeParts() {
    if (!this.partData.length) return;
    const body = this.host.body(), frame = this.host.frame(), head = this.host.headBone();
    const J = body.joints, jh = [J[head * 3], J[head * 3 + 1], J[head * 3 + 2]], k = body.headScale();
    if (!this.parts.length) {
      if (this.blink) this.blink.ready = false;
      for (const d of this.partData) {
        const rest = pagePoints(d.part.points, frame, body.floorShift), n = rest.length / 3;
        const normals = d.part.normals ? pageVectors(d.part.normals, null, n, frame) : new Float32Array(n * 3);
        for (let i = 0; i < n; i++) { const l = Math.hypot(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]) || 1; normals[i * 3] /= l; normals[i * 3 + 1] /= l; normals[i * 3 + 2] /= l; }
        if (!d.part.normals) vertexNormals(rest, d.part.triangles, normals);
        if (d.kind === 'hair' || d.kind === 'brows') this.parts.push(...this.groomMeshes(d.kind, d.part as Groom, rest, normals, head));
        else this.parts.push(this.partMesh(d.name, d.kind, d.part, rest, normals, head));
      }
      this.placedHead = { joint: jh, scale: k };
    }
    this.moveParts();
  }

  // the parts keep their place on the head of the card's body: they move and scale with the head, and the lashes and
  // the brows take the eye targets' offsets of the blink shown
  private moveParts() {
    const body = this.host.body(), head = this.host.headBone(), J = body.joints, jh = [J[head * 3], J[head * 3 + 1], J[head * 3 + 2]];
    const ph = this.placedHead!, s = body.headScale() / ph.scale;
    for (const p of this.parts) {
      const at = p.geo.attributes.position as any, A = at.array as Float32Array, R = p.rest, B = p.blink, F = p.face;
      for (let i = 0; i < A.length; i += 3) for (let c = 0; c < 3; c++) A[i + c] = jh[c] + (R[i + c] - ph.joint[c] + (B ? B[i + c] : 0)) * s + (F ? F[i + c] : 0);
      at.needsUpdate = true;
      p.geo.computeBoundingSphere();
    }
  }

  // ---------------------------------------------------------------- faces (face/expressions.ts)
  // The card's own face (its facial actions), its mouth poses as faces, and its eye targets for the face's blinks and
  // squints. The face's motion moves the parts: the lashes and the brows with the skin under them, the lower teeth and
  // gums, the tongue and the floor of the mouth with the jaw, the inner mouth with the poses' own offsets, and a sculpt
  // card's patch with the coarse body.
  get facialActions(): Record<string, number> { return this.plan?.facialActions || {}; }
  setFaceEyes(close: { L: number; R: number }, squint: { L: number; R: number }) {
    const b = this.blink;
    if (!b) return;
    b.face = { close: { ...close }, squint: { ...squint } };
    if (!b.active && this.placed) this.showEyes(b.close);
  }
  // the poses on the page: offsets of the coarse body, of the rows of the level shown and of the inner mouth's points
  poses(): CardPoses | null {
    const m = this.mouthPoses, plan = this.plan;
    if (!m || !plan || !m.names.length) return null;
    if (this.cardPoses) return this.cardPoses;
    const frame = this.host.frame(), u32 = (name: string) => { const x = this.host.buffer(name); return x ? new Uint32Array(x) : null; };
    const coarseRows = u32('card_coarse');
    if (!coarseRows) return null;
    const topology = ['anny', 'sculpt'].includes(plan.topology) ? plan.topology : 'anny';
    const level = plan.shown > 0 && m.levels[plan.shown] ? plan.shown : 0;
    const pageRows = level ? u32(`card_rows_${topology}_${level}`) : null;
    const at = new Map<number, number>();
    if (level) m.levels[level].rows.forEach((r, k) => at.set(r, k));
    const P = m.names.length, sparse = (full: Float32Array) => {
      const idx: number[] = [], d: number[] = [];
      for (let i = 0; i < full.length / 3; i++) if (full[i * 3] || full[i * 3 + 1] || full[i * 3 + 2]) { idx.push(i); d.push(full[i * 3], full[i * 3 + 1], full[i * 3 + 2]); }
      return { idx: Uint32Array.from(idx), d: Float32Array.from(d) };
    };
    const coarse = [], rows = [], inner: (Float32Array | null)[] = [], mouth = this.partData.find((x) => x.kind === 'mouth');
    for (let k = 0; k < P; k++) {
      coarse.push(sparse(pageVectors(m.skin.subarray(k * m.rows * 3, (k + 1) * m.rows * 3), coarseRows, coarseRows.length, frame)));
      if (level && pageRows) {
        const L = m.levels[level], n = L.rows.length, full = new Float32Array(pageRows.length * 3), src = L.offsets.subarray(k * n * 3, (k + 1) * n * 3);
        for (let i = 0; i < pageRows.length; i++) {
          const q = at.get(pageRows[i]);
          if (q === undefined) continue;
          const v = pageVectors(src.subarray(q * 3, q * 3 + 3), null, 1, frame);
          full[i * 3] = v[0]; full[i * 3 + 1] = v[1]; full[i * 3 + 2] = v[2];
        }
        rows.push(sparse(full));
      } else rows.push({ idx: new Uint32Array(0), d: new Float32Array(0) });
      inner.push(mouth && m.points === mouth.part.points.length / 3 ? pageVectors(m.inner.subarray(k * m.points * 3, (k + 1) * m.points * 3), null, m.points, frame) : null);
    }
    this.cardPoses = { names: m.names.slice(), coarse, level: level && pageRows ? level : 0, rows, inner };
    return this.cardPoses;
  }
  // the face's motion on the parts and on the patch (null: the face at rest)
  followFace(m: FaceMotion | null) {
    this.face = m;
    if (!this.plan || !this.placed) return;
    const body = this.host.body(), o = [0, 0, 0], P = this.cardPoses;
    for (const p of this.parts) {
      const n = p.rest.length / 3;
      if (!m) { p.face = null; continue; }
      const F = p.face || (p.face = new Float32Array(n * 3));
      const A = p.geo.attributes.position.array as Float32Array;
      if (p.kind === 'lashes' || p.kind === 'brows') {
        if (!p.bind) p.bind = nearestVertices(A, body.pos, m.region, 0.004);
        for (let i = 0; i < n; i++) { m.skin(p.bind[i], o); F[i * 3] = o[0]; F[i * 3 + 1] = o[1]; F[i * 3 + 2] = o[2]; }
      } else if ((p.kind === 'mouth' || p.kind === 'teeth') && p.arch) {
        // the points at the place the head gives them (the face's own offsets are taken off first)
        for (let i = 0; i < n; i++) {
          const x = A[i * 3] - F[i * 3], y = A[i * 3 + 1] - F[i * 3 + 1], z = A[i * 3 + 2] - F[i * 3 + 2];
          if (p.tongue && p.tongue[i]) m.tongue(x, y, z, o); else if (p.arch[i]) m.lower(x, y, z, o); else { o[0] = o[1] = o[2] = 0; }
          F[i * 3] = o[0]; F[i * 3 + 1] = o[1]; F[i * 3 + 2] = o[2];
        }
        // the lower jaw's normals turn with the jaw
        const N = p.geo.attributes.normal.array as Float32Array, R = m.jawTurn, N0 = p.normals;
        for (let i = 0; i < n; i++) {
          const q = i * 3, x = N0[q], y = N0[q + 1], z = N0[q + 2];
          if (p.arch[i]) { N[q] = R[0] * x + R[1] * y + R[2] * z; N[q + 1] = R[3] * x + R[4] * y + R[5] * z; N[q + 2] = R[6] * x + R[7] * y + R[8] * z; }
          else { N[q] = x; N[q + 1] = y; N[q + 2] = z; }
        }
        p.geo.attributes.normal.needsUpdate = true;
      } else { p.face = null; continue; }
      // the inner mouth's own offsets in the card's poses
      if (p.inner && P) P.inner.forEach((D, k) => { const f = m.poses[k]; if (f && D && D.length === F.length) for (let i = 0; i < F.length; i++) F[i] += f * D[i]; });
    }
    if (this.parts.length) this.moveParts();
    // the patch of a sculpt card: its blend of the coarse body's face
    const q = this.patch;
    if (q?.geo && (q as any).base) {
      if (m?.coarse) {
        const D = this.patchFace || (this.patchFace = new Float32Array(q.rows.length * 3)), C = m.coarse;
        for (let i = 0; i < q.rows.length; i++) {
          let x = 0, y = 0, z = 0;
          for (let e = q.start[i]; e < q.start[i + 1]; e++) { const c = q.cols[e] * 3, w = q.values[e]; x += w * C[c]; y += w * C[c + 1]; z += w * C[c + 2]; }
          D[i * 3] = x; D[i * 3 + 1] = y; D[i * 3 + 2] = z;
        }
      } else this.patchFace = null;
      this.showPatch();
    }
    this.host.moved();
  }
  // the patch at rest, with the blink's and the face's offsets
  private showPatch() {
    const q = this.patch;
    if (!q?.geo || !(q as any).base) return;
    const P = q.geo.attributes.position.array as Float32Array, B = (q as any).base as Float32Array, D = this.patchBlink, F = this.patchFace;
    for (let i = 0; i < P.length; i++) P[i] = B[i] + (D ? D[i] : 0) + (F ? F[i] : 0);
    this.patchNormals();
  }
  private patchBlink: Float32Array | null = null;

  // ---------------------------------------------------------------- blinks
  // The page plays the GLB export's eye targets (opensculptboy.eyes.page): on the fine body as sparse shapes of its
  // vertices (the card's level carried through the page's remaining subdivision), on the sculpt patch by its card rows,
  // and on the lashes and the brows by their points. A blink of the timer closes both eyes over BLINK_SECONDS, every
  // 2.5 to 6 seconds; the panel sets the closure and the squint that the blinks start from.
  get hasBlinks() { return !!this.blink; }
  blinkPanel() { const b = this.blink; return b ? { auto: b.auto, close: b.close, squint: b.squint } : null; }
  setBlinkAuto(on: boolean) { const b = this.blink; if (!b) return; b.auto = on; b.next = performance.now() + 600; if (!on && b.active) { b.active = false; this.showEyes(b.close); } }
  blinkNow() { const b = this.blink; if (!b) return; b.active = true; b.start = performance.now(); }
  setEyes(close: number | null, squint: number | null) {
    const b = this.blink;
    if (!b) return;
    if (close !== null) b.close = Math.min(1, Math.max(0, close));
    if (squint !== null) b.squint = Math.min(1, Math.max(0, squint));
    if (!b.active) this.showEyes(b.close);
  }
  // the timer: true while a blink moves the eyes this frame
  tick(now: number): boolean {
    const b = this.blink;
    if (!b || !this.placed) return false;
    if (b.auto && !b.active && now >= b.next) { b.active = true; b.start = now; }
    if (!b.active) return false;
    const t = (now - b.start) / (BLINK_SECONDS * 1000);
    if (t >= 1) { b.active = false; b.next = now + 2500 + Math.random() * 3500; this.showEyes(b.close); return true; }
    this.showEyes(Math.max(b.close, blinkCurve(t)));
    return true;
  }

  private showEyes(close: number) {
    const b = this.blink!;
    if (!b.ready) this.prepareBlinks();
    const f = b.face;
    const w = blinkWeights(b.data.names, { L: Math.max(close, f.close.L), R: Math.max(close, f.close.R) }, { L: Math.max(b.squint, f.squint.L), R: Math.max(b.squint, f.squint.R) });
    this.host.blinkWeights(w);
    const sum = (shapes: Float32Array[] | undefined, out: Float32Array) => {
      out.fill(0);
      if (!shapes) return;
      shapes.forEach((D, t) => { const k = w[t]; if (k) for (let i = 0; i < out.length; i++) out[i] += k * D[i]; });
    };
    for (const p of this.parts) if (p.blink) sum(this.partBlinks.get(p), p.blink);
    if (this.parts.length) this.moveParts();
    const q = this.patch;
    if (q?.geo && (q as any).base && this.patchBlinks) {
      const D = this.patchBlink || (this.patchBlink = new Float32Array(q.geo.attributes.position.array.length));
      sum(this.patchBlinks, D);
      this.showPatch();
    }
    this.host.moved();
  }

  private partBlinks = new Map<PartMesh, Float32Array[]>();
  private patchBlinks: Float32Array[] | null = null;

  // the shapes of the current body: the fine body's (when the file holds the level the page shows), the patch's and
  // the parts'
  private prepareBlinks() {
    const b = this.blink!, plan = this.plan!, d = b.data, body = this.host.body(), frame = this.host.frame();
    const T = d.names.length, K = d.rows.length, R = frame.rotation, sc = frame.scale;
    b.ready = true;
    const at = new Map<number, number>();
    for (let k = 0; k < K; k++) at.set(d.rows[k], k);
    // s R offset of target t at moved row k, added to out at o with the weight f
    const add = (src: Float32Array, rows: number, t: number, k: number, out: Float32Array, o: number, f = 1) => {
      const q = (t * rows + k) * 3, x = src[q], y = src[q + 1], z = src[q + 2];
      for (let r = 0; r < 3; r++) out[o + r] += f * sc * (R[r * 3] * x + R[r * 3 + 1] * y + R[r * 3 + 2] * z);
    };
    const topology = ['anny', 'sculpt'].includes(plan.topology) ? plan.topology : 'anny';
    const u32 = (name: string) => { const x = this.host.buffer(name); return x ? new Uint32Array(x) : null; };
    const rows = d.level === plan.shown ? (d.level > 0 ? u32(`card_rows_${topology}_${d.level}`) : u32('card_coarse')) : null;
    const shapes: BodyShape[] = [];
    if (rows) {
      const fixBuf = d.level > 0 ? this.host.buffer(`card_fix_${topology}_${d.level}`) : null;
      const tris = body.hide && body.weldShown ? body.weldShown : body.weld, n = body.nFine;
      // the normals' change against the normals of the same positions without the blink (the body's own normals hold
      // the correctives' and the pocket's corrections)
      const N0 = body.copyRows(vertexNormals(body.pos, tris), 3);
      let touched = 0;
      for (let t = 0; t < T; t++) {
        const lv = new Float32Array(rows.length * 3);
        for (let i = 0; i < rows.length; i++) { const k = at.get(rows[i]); if (k !== undefined) add(d.offsets, K, t, k, lv, i * 3); }
        if (fixBuf) {   // the rows whose stencil a patch changed: the blend of their three card rows
          const f = new Float32Array(fixBuf), ids = new Uint32Array(fixBuf);
          for (let q = 0; q < f.length; q += 10) {
            const o = ids[q] * 3;
            lv[o] = lv[o + 1] = lv[o + 2] = 0;
            for (let j = 0; j < 3; j++) { const k = at.get(ids[q + 1 + j]); if (k !== undefined) add(d.offsets, K, t, k, lv, o, f[q + 4 + j]); }
          }
        }
        const fine = d.level > 0
          ? body.sub.apply(new Float32Array(body.nBody * 3), 3, d.level === 1 ? [lv, null] : [null, lv]).slice()
          : body.sub.apply(lv.subarray(0, body.nBody * 3), 3).slice();
        const P1 = body.pos.slice();
        for (let i = 0; i < fine.length; i++) P1[i] += fine[i];
        const N1 = body.copyRows(vertexNormals(P1, tris), 3), idx: number[] = [];
        for (let v = 0; v < n; v++) {
          const o = v * 3, dd = fine[o] ** 2 + fine[o + 1] ** 2 + fine[o + 2] ** 2;
          const dn = (N1[o] - N0[o]) ** 2 + (N1[o + 1] - N0[o + 1]) ** 2 + (N1[o + 2] - N0[o + 2]) ** 2;
          if (dd > 1e-14 || dn > 1e-8) idx.push(v);
        }
        const D = new Float32Array(idx.length * 3), Nd = new Float32Array(idx.length * 3);
        idx.forEach((v, i) => { for (let c = 0; c < 3; c++) { D[i * 3 + c] = fine[v * 3 + c]; Nd[i * 3 + c] = N1[v * 3 + c] - N0[v * 3 + c]; } });
        shapes.push({ idx: Uint32Array.from(idx), d: D, n: Nd });
        touched = Math.max(touched, idx.length);
      }
      b.vertices = touched;
      // the patch's vertices: their card rows of the level
      const q = this.patch;
      this.patchBlinks = null;
      if (q && d.level > 0) {
        this.patchBlinks = [];
        for (let t = 0; t < T; t++) {
          const D = new Float32Array(q.rows.length * 3);
          for (let i = 0; i < q.rows.length; i++) { const k = at.get(q.rows[i]); if (k !== undefined) add(d.offsets, K, t, k, D, i * 3); }
          this.patchBlinks.push(D);
        }
      }
    }
    this.host.blinkShapes(rows ? shapes : null);
    // the lashes and the brows: the offsets of their points
    this.partBlinks.clear();
    for (const p of this.parts) {
      const src = p.kind === 'lashes' ? d.lashes : p.kind === 'brows' ? d.brows : null;
      const m = p.rest.length / 3;
      if (!src || src.length !== T * m * 3) { p.blink = null; continue; }
      const list: Float32Array[] = [];
      for (let t = 0; t < T; t++) { const D = new Float32Array(m * 3); for (let i = 0; i < m; i++) add(src, m, t, i, D, i * 3); list.push(D); }
      this.partBlinks.set(p, list);
      p.blink = new Float32Array(m * 3);
    }
  }

  // a groom of the reference (opensculptboy.groom): one mesh per material, with its texture, alpha and sides as the
  // GLB export writes them, and the hair's vertex colours
  private groomMeshes(kind: 'hair' | 'brows', g: Groom, rest: Float32Array, normals: Float32Array, head: number): PartMesh[] {
    const out: PartMesh[] = [];
    g.materials.forEach((m, k) => {
      const tris: number[] = [];
      for (let t = 0; t < g.material.length; t++) if (g.material[t] === k) tris.push(g.triangles[t * 3], g.triangles[t * 3 + 1], g.triangles[t * 3 + 2]);
      if (!tris.length) return;
      const part: Part = { points: g.points, normals: g.normals, uvs: g.uvs, triangles: Uint32Array.from(tris) };
      const look = { base: m.base, rough: m.roughness, alpha: m.alpha, cutoff: m.cutoff, double: m.double, normalScale: m.normalScale,
        map: this.images.has(`${kind}:${m.name}:base_color`) ? this.texture(`${kind}:${m.name}:base_color`, true) : null,
        normal: this.images.has(`${kind}:${m.name}:normal`) && g.uvs ? this.texture(`${kind}:${m.name}:normal`, false) : null,
        colors: g.colors };
      out.push(this.partMesh(`${kind}:${m.name}`, kind, part, rest, normals, head, look));
    });
    return out;
  }

  private partMesh(name: string, kind: PartKind, part: Part, rest: Float32Array, normals: Float32Array, head: number, look: any = null): PartMesh {
    const T = this.host.THREE, plan = this.plan!, n = rest.length / 3, geo = new T.BufferGeometry();
    geo.setAttribute('position', new T.BufferAttribute(rest.slice(), 3));
    geo.setAttribute('normal', new T.BufferAttribute(normals, 3));
    geo.setAttribute('uv', new T.BufferAttribute(part.uvs || new Float32Array(n * 2), 2));
    const si = new Uint16Array(n * 4), sw = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) { si[i * 4] = head; sw[i * 4] = 1; }
    geo.setAttribute('skinIndex', new T.BufferAttribute(si, 4));
    geo.setAttribute('skinWeight', new T.BufferAttribute(sw, 4));
    geo.setIndex(new T.BufferAttribute(part.triangles, 1));
    const mt: any = { base: [1, 1, 1, 1], rough: 0.5, map: null, normal: null, mr: null, blend: false, mask: false, double: false, cutoff: 0.5, normalScale: 1 };
    if (look) {
      Object.assign(mt, { base: look.base.slice(0, 4), rough: look.rough, map: look.map, normal: look.normal, blend: look.alpha === 'BLEND', mask: look.alpha === 'MASK',
        double: look.double, cutoff: look.cutoff, normalScale: look.normalScale });
      if (look.colors) geo.setAttribute('gColor', new T.BufferAttribute(look.colors, 4));
    } else if (kind === 'lashes') { mt.map = this.images.has('lashes') ? this.texture('lashes', true) : null; mt.rough = plan.lashes!.roughness; mt.blend = true; mt.double = true; }
    else if (kind === 'mouth') {
      mt.map = this.images.has('mouth_base_color') ? this.texture('mouth_base_color', true) : null;
      mt.normal = this.images.has('mouth_normal') && part.uvs ? this.texture('mouth_normal', false) : null;
      mt.mr = this.images.has('mouth_roughness') ? this.texture('mouth_roughness', false) : null; mt.rough = 1; mt.double = true;
    } else { mt.base = [...plan.teeth!.color, 1]; mt.rough = plan.teeth!.roughness; }
    if (mt.normal) geo.setAttribute('gTangent', new T.BufferAttribute(uvTangents(rest, normals, part.uvs!, part.triangles), 4));
    const defines: Record<string, string> = Object.assign({}, this.host.envDefines());
    if (mt.map) defines.HAS_MAP = '';
    if (mt.normal) defines.HAS_NORMAL_MAP = '';
    if (mt.mr) defines.HAS_MR = '';
    if (mt.blend) defines.ALPHA_BLEND = '';
    if (mt.mask) defines.ALPHA_MASK = '';
    if (look?.colors) defines.HAS_VCOLOR = '';
    const own = { uBase: { value: new T.Vector4(...(mt.base as [number, number, number, number])) }, uMetal: { value: 0 }, uRough: { value: mt.rough },
      uEmissive: { value: new T.Vector3(0, 0, 0) }, uCutoff: { value: mt.cutoff }, uNormalScale: { value: mt.normalScale },
      uMap: { value: mt.map }, uMR: { value: mt.mr }, uNormalMap: { value: mt.normal }, uAO: { value: null }, uEmissiveMap: { value: null } };
    const mat = new T.ShaderMaterial({ uniforms: Object.assign({}, this.host.uniforms, own), vertexShader: GARMENT_VS, fragmentShader: this.host.toMRT(GARMENT_FS),
      defines, glslVersion: T.GLSL3, blending: T.NoBlending, side: mt.double ? T.DoubleSide : T.FrontSide });
    const depth = mt.blend || mt.mask ? new T.ShaderMaterial({ uniforms: { uBase: own.uBase, uMap: own.uMap, uCutoff: { value: mt.blend ? BLEND_SHADOW_CUTOFF : mt.cutoff } },
      vertexShader: GARMENT_DEPTH_VS, fragmentShader: GARMENT_DEPTH_FS, defines: Object.assign({}, this.host.depth.defines, mt.map ? { HAS_MAP: '' } : {}), side: T.DoubleSide }) : this.host.depth;
    const mesh = new T.SkinnedMesh(geo, mat);
    mesh.bind(this.host.skeleton(), new T.Matrix4());
    mesh.frustumCulled = false; mesh.name = `card ${name}`; mesh.userData.depthMat = depth;
    this.host.scene.add(mesh); this.host.opaque.push(mesh);
    return { name, kind, mesh, geo, rest, normals: normals.slice(), materials: [mat], depth: [depth], blink: null, face: null, bind: null,
      arch: part.arch && part.arch.length === n ? part.arch : null, tongue: tonguePoints(part), inner: kind === 'mouth' };
  }

  private texture(slot: string, srgb: boolean): THREE_NS.Texture {
    const key = `${slot}:${srgb}`;
    let tex = this.textures.get(key);
    if (tex) return tex;
    const T = this.host.THREE;
    tex = new T.Texture(this.images.get(slot) as any);
    tex.flipY = false; tex.colorSpace = srgb ? T.SRGBColorSpace : T.NoColorSpace;
    tex.wrapS = tex.wrapT = T.ClampToEdgeWrapping; tex.anisotropy = ANISOTROPY;
    tex.needsUpdate = true;
    this.textures.set(key, tex);
    return tex;
  }

  // ---------------------------------------------------------------- unloading
  unload(restore = true) {
    for (const p of this.parts) {
      this.host.scene.remove(p.mesh);
      const i = this.host.opaque.indexOf(p.mesh);
      if (i >= 0) this.host.opaque.splice(i, 1);
      p.geo.dispose(); p.materials.forEach((m) => m.dispose()); p.depth.forEach((m) => { if (m !== this.host.depth) m.dispose(); });
    }
    this.parts = []; this.partData = []; this.placedHead = null; this.placed = false;
    this.mouthPoses = null; this.cardPoses = null; this.face = null; this.patchFace = null; this.patchBlink = null;
    if (this.blink) { this.host.blinkShapes(null); this.blink = null; }
    this.partBlinks.clear(); this.patchBlinks = null;
    if (this.patch?.mesh) {
      this.host.scene.remove(this.patch.mesh);
      const i = this.host.opaque.indexOf(this.patch.mesh);
      if (i >= 0) this.host.opaque.splice(i, 1);
      this.patch.geo!.dispose();
    }
    this.patch = null; this.coarse0 = null; this.host.uniforms.uCardHide.value = 0;
    if (this.eyeState) {
      const eyes = this.host.eyes();
      eyes.forEach((e, i) => {
        const s = this.eyeState!.saved[i];
        if (e.mesh.geometry !== s.geo) e.mesh.geometry.dispose();
        e.mesh.geometry = s.geo; e.mesh.material = s.mat; e.u = s.u; e.mesh.userData.rot0 = s.rot0; e.card = null;
      });
      this.eyeState.meshes.forEach((m) => m.material?.dispose());
      this.eyeState = null;
    }
    if (this.skin) { this.host.setSkinMaterial(null); this.skin.dispose(); this.skin = null; }
    for (const t of this.textures.values()) t.dispose();
    this.textures.clear();
    for (const b of this.images.values()) b.close?.();
    this.images.clear();
    this.iris = {};
    const had = !!this.plan;
    this.plan = null; this.notes = []; this.failures = 0;
    if (restore && had) { this.host.setBodyCard(null); this.host.setPageParts(null); this.host.changed(); }
    if (had) this.host.faceChanged();
  }

  stats() {
    const eyes = this.host.eyes().filter((e) => e.card).map((e) => ({ side: e.side, ...e.card }));
    return { active: !!this.plan, level: this.plan?.shown ?? null, topology: this.plan?.topology ?? null, textures: [...this.images.keys()].sort(),
      patch: this.patch ? { vertices: this.patch.rows.length, triangles: this.patch.triangles.length / 3, border: this.patchBorder } : null,
      skin: !!this.skin, parts: this.parts.map((p) => ({ name: p.name, vertices: p.rest.length / 3 })), eyes, notes: this.notes, timing: this.timing,
      blinks: this.blink ? { targets: this.blink.data.names, level: this.blink.data.level, vertices: this.blink.vertices, auto: this.blink.auto,
        close: this.blink.close, squint: this.blink.squint } : null };
  }
}

// the single-surface eyeball of opensculptboy.eyes.geometry.single_profile on a latitude-longitude grid: in the eye's
// frame (columns of A: +X, up, gaze, page frame) about the centre c, of radius r; the attributes hold each vertex's
// point and normal in the eye's frame (eyeball radii) and a falloff of the light toward the back of the eye
export function eyeGeometry(T: typeof THREE_NS, c: number[], r: number, A: number[], iris: number, cornea: number, bulge: number) {
  const nlat = 72, nlon = 96, nv = (nlat + 1) * (nlon + 1);
  const pos = new Float32Array(nv * 3), loc = new Float32Array(nv * 3), ao = new Float32Array(nv);
  let v = 0;
  for (let i = 0; i <= nlat; i++) {
    const th = Math.PI * Math.pow(i / nlat, 1.6), t = outerRadius(th, iris, cornea, bulge);
    for (let j = 0; j <= nlon; j++, v++) {
      const ph = 2 * Math.PI * j / nlon;
      const x = Math.sin(th) * Math.cos(ph) * t, y = Math.sin(th) * Math.sin(ph) * t, z = Math.cos(th) * t;
      loc[v * 3] = x; loc[v * 3 + 1] = y; loc[v * 3 + 2] = z;
      for (let k = 0; k < 3; k++) pos[v * 3 + k] = c[k] + r * (A[k * 3] * x + A[k * 3 + 1] * y + A[k * 3 + 2] * z);
      ao[v] = 0.3 + 0.7 * Math.min(1, Math.max(0, (z + 0.3) / 0.9));
    }
  }
  const idx = new Uint32Array(nlat * nlon * 6);
  let q = 0;
  const id = (i: number, j: number) => i * (nlon + 1) + j;
  for (let i = 0; i < nlat; i++) for (let j = 0; j < nlon; j++) {
    const a = id(i, j), b = id(i + 1, j), cc = id(i + 1, j + 1), d = id(i, j + 1);
    idx[q++] = a; idx[q++] = b; idx[q++] = cc; idx[q++] = a; idx[q++] = cc; idx[q++] = d;
  }
  // the normals of the surface in the eye's frame, the same on both edges of the longitude seam and outward at the poles
  const nrm = vertexNormals(loc, idx);
  for (let i = 0; i <= nlat; i++) {
    const a = id(i, 0) * 3, b = id(i, nlon) * 3;
    for (let k = 0; k < 3; k++) { const m = 0.5 * (nrm[a + k] + nrm[b + k]); nrm[a + k] = m; nrm[b + k] = m; }
  }
  for (const i of [0, nlat]) for (let j = 0; j <= nlon; j++) { const o = id(i, j) * 3; nrm[o] = 0; nrm[o + 1] = 0; nrm[o + 2] = i === 0 ? 1 : -1; }
  for (let o = 0; o < nrm.length; o += 3) {
    const l = Math.hypot(nrm[o], nrm[o + 1], nrm[o + 2]) || 1, out = nrm[o] * loc[o] + nrm[o + 1] * loc[o + 1] + nrm[o + 2] * loc[o + 2] < 0 ? -1 : 1;
    nrm[o] *= out / l; nrm[o + 1] *= out / l; nrm[o + 2] *= out / l;
  }
  const g = new T.BufferGeometry();
  g.setAttribute('position', new T.BufferAttribute(pos, 3));
  g.setAttribute('eyelocal', new T.BufferAttribute(loc, 3));
  g.setAttribute('eyenormal', new T.BufferAttribute(nrm, 3));
  g.setAttribute('eyeao', new T.BufferAttribute(ao, 1));
  const wn = new Float32Array(nv * 3);
  for (let i = 0; i < nv; i++) for (let k = 0; k < 3; k++) wn[i * 3 + k] = A[k * 3] * nrm[i * 3] + A[k * 3 + 1] * nrm[i * 3 + 1] + A[k * 3 + 2] * nrm[i * 3 + 2];
  g.setAttribute('normal', new T.BufferAttribute(wn, 3));
  g.setIndex(new T.BufferAttribute(idx, 1));
  g.computeBoundingSphere();
  return g;
}

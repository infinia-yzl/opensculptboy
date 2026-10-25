// OpenSculptBoy
// Apache License, Version 2.0
//
// The wardrobe of the page: a character card with its outfits and wearable packages from the user's own files,
// dressed on the current body (dress.ts) and drawn as meshes skinned by the page's rig, so that the pieces follow
// the poses and clips. A slider edit moves the pieces at once (the bindings and the skin weights); the push-out and
// the covered skin follow when the sliders rest, in slices of at most SLICE_MS between frames, so that the page keeps
// drawing. The body's triangles under the shown pieces leave its index buffer.
// Nothing of the user's files is stored or sent: the packages live in this page's memory until it closes.

import type * as THREE_NS from 'three';
import { BodySurface, DressCache, dress, dressSteps, estimatedPose, MOUTH_ANCESTOR, packageBones, outfitCoverSteps, runSteps, visibleTriangles, type DressedPiece, type Frame, type Steps, type WornPiece } from './dress.ts';
import { dirname, FileSet, type PickedFile } from './files.ts';
import { readSafetensors, readWearable, type Material, type Wearable } from './package.ts';
import { BLEND_SHADOW_CUTOFF, GARMENT_DEPTH_FS, GARMENT_DEPTH_VS, GARMENT_FS, GARMENT_VS } from './shading.ts';
import { skinTriangles } from './skin.ts';
import { cardOutfit, NO_OUTFIT, resolveRef, wornPieces, type WornRef } from './wardrobe.ts';

// what the wardrobe needs from the renderer (main.ts)
export interface WardrobeHost {
  THREE: typeof THREE_NS;
  scene: THREE_NS.Scene;
  opaque: any[];
  skeleton(): THREE_NS.Skeleton;
  uniforms: Record<string, any>;
  depth: THREE_NS.ShaderMaterial;                        // the page's material of the shadow maps
  envDefines(): Record<string, string>;
  toMRT(fs: string): string;
  body(): { coarse: Float32Array; joints: Float32Array; nBody: number; quads: Uint32Array; coarseSkin: Uint8Array; bones: string[]; frame: Frame;
    subdivision: { ops: SparseRows[] }; geometry: THREE_NS.BufferGeometry };   // the body's subdivision, coarse to fine
  setBodyOffsets(offsets: Float32Array | null): void;   // the card's level-0 detail offsets in the page's frame (nBody * 3)
  applyCard(card: any): void;                            // the card's sliders and face into the look
  changed(): void;                                       // a new picture: shadows and accumulation
}

export interface SparseRows { rows: number; start: Uint32Array; index: Uint32Array; value: Float32Array }
export interface PieceRow { ref: string; piece: string; outfit: string | null; label: string; slot: string; layer: number; shown: boolean;
  hiddenGroups: string[]; groups: string[]; provenance: string; pushed: number; pushMm: number; vertices: number }

const SETTLE_MS = 250;   // the push-out and the covered skin follow a slider edit after this rest
const SLICE_MS = 8;      // the longest run of the settled dress on the main thread before it yields to the browser
const INDEX_SLICE = 32768;   // body triangles per slice of the covered skin's index, and fine rows per slice of its mask

// a task soon after the current one, without the clamp of nested timeouts
function soon(f: () => void) {
  if (typeof MessageChannel === 'undefined') { setTimeout(f, 0); return; }
  const c = new MessageChannel();
  c.port1.onmessage = () => { c.port1.close(); f(); };
  c.port2.postMessage(0);
}

export class Wardrobe {
  host: WardrobeHost;
  card: any = null; cardName = ''; files: FileSet | null = null;
  packages = new Map<string, Wearable>();    // by reference
  outfit: string | null = null; show: string[] = []; hide: string[] = []; keepSkin = false;
  pieces: DressedPiece[] = []; worn: WornRef[] = [];
  notes: string[] = [];
  failures = 0;   // the notes that report a piece or a file that failed
  timing: Record<string, number> = {};
  private meshes = new Map<string, { mesh: any; geo: THREE_NS.BufferGeometry; wearable: Wearable; materials: any[]; depth: any[]; hidden: string; weights: Float32Array | null }>();
  private cache = new DressCache();
  private skin: Int32Array | null = null; private weights: Float64Array | null = null; private indices: Int32Array | null = null;
  private bodyIndex: Uint32Array | null = null; private indexScratch: Uint32Array | null = null;
  private textures = new Map<string, THREE_NS.Texture>();
  private settle: any = null;
  private job = 0; private jobs = 0;   // the settled dress in flight (its number, 0 for none) and the count of them
  listeners: (() => void)[] = [];

  constructor(host: WardrobeHost) { this.host = host; }
  get loaded() { return !!this.card; }
  private notify() { for (const f of this.listeners) f(); }
  // a failure: a note for the section and an error in the console
  private fail(note: string, err?: unknown) {
    this.notes = [...this.notes, note]; this.failures++;
    console.error(`Wearables: ${note}`, ...(err instanceof Error ? [err] : []));
  }

  // ---------------------------------------------------------------- loading
  // a card and its packages from the picked files; packages without a card become loose pieces
  async load(picked: PickedFile[]): Promise<{ pieces: number; notes: string[] }> {
    const files = new FileSet(picked), notes: string[] = [];
    let failures = 0;
    const fail = (note: string, err?: unknown) => { notes.push(note); failures++; console.error(`Wearables: ${note}`, ...(err instanceof Error ? [err] : [])); };
    let card: any = null, cardPath = '';
    const jsons = [...files.byPath.keys()].filter((p) => p.toLowerCase().endsWith('.json') && !p.endsWith('.wearable.json'));
    for (const p of jsons) {
      try {
        const j = JSON.parse(await files.byPath.get(p)!.text());
        if (j && (Array.isArray(j.outfits) || Array.isArray(j.wearables)) && (!card || (j.outfits?.length && !card.outfits?.length))) { card = j; cardPath = p; }
      } catch (e) { /* not a card */ }
    }
    if (!card) {
      const packs = [...files.byPath.keys()].filter((p) => p.endsWith('.wearable.json'));
      if (!packs.length) throw new Error('The files hold no character card with outfits and no wearable package.');
      card = { name: 'wearables', wearables: packs.map((p) => ({ id: p.split('/').pop()!.replace('.wearable.json', ''), file: p, enabled: true })) };
      cardPath = '';
    }
    const folder = dirname(cardPath), packages = new Map<string, Wearable>(), byFile = new Map<File, Wearable>();
    const entries = [...(card.outfits || []).flatMap((o: any) => (o.pieces || []).map((p: any) => ({ ref: `${o.id}/${p.piece}`, file: p.file }))),
      ...(card.wearables || []).map((w: any) => ({ ref: w.id || String(w.file).split('/').pop()!.split('.')[0], file: w.file }))];
    for (const e of entries) {
      const f = files.find(e.file, folder);
      if (!f) { fail(`${e.ref}: ${e.file} is missing`); continue; }
      try {
        let w = byFile.get(f);
        if (!w) {
          const where = dirname(files.pathOf(f));
          w = await readWearable(JSON.parse(await f.text()), (rel) => files.bytes(rel, where));
          byFile.set(f, w);
        }
        packages.set(e.ref, w);
      } catch (err: any) { fail(`${e.ref}: ${err.message || err}`, err); }
    }
    // the bones of each piece in the page's rig: the mouth bones of a card on the anny-mouth rig move with their kept
    // ancestor; a piece on a bone that the rig lacks stays out
    const bones = this.host.body().bones;
    for (const [ref, w] of packages) {
      const { folded, missing } = packageBones(w, bones);
      if (missing.length) { packages.delete(ref); fail(`${ref}: the page's rig lacks its bones ${missing.join(', ')}`); }
      else if (folded.length) notes.push(`${ref}: the page moves its bones ${folded.join(', ')} with ${MOUTH_ANCESTOR}`);
    }
    // only the pieces that loaded take part
    const ok = (file: string, ref: string) => packages.has(ref) && !!file;
    card = JSON.parse(JSON.stringify(card));
    for (const o of card.outfits || []) o.pieces = (o.pieces || []).filter((p: any) => ok(p.file, `${o.id}/${p.piece}`));
    card.wearables = (card.wearables || []).filter((w: any) => ok(w.file, w.id || String(w.file).split('/').pop()!.split('.')[0]));
    const refs = this.refsOf(card);
    for (const k of Object.keys(card.wearable_overrides || {})) {
      try { resolveRef(k, refs, cardOutfit(card)); } catch (e) { delete card.wearable_overrides[k]; notes.push(`override ${k}: no such piece`); }
    }
    if ((card.local_changes && Object.keys(card.local_changes).length)) notes.push('the card\'s local changes are not shown in the page');
    // the level-0 offsets of the card's detail layer
    let offsets: Float32Array | null = null;
    const level0 = (card.detail?.levels || []).find((l: any) => Number(l.subdiv) === 0);
    if (level0) {
      try { offsets = this.pageOffsets(readSafetensors(await files.bytes(level0.offsets, folder)).tensors.offsets?.values); }
      catch (err: any) { fail(`detail level 0: ${err.message || err}`, err); }
    }
    if ((card.detail?.levels || []).some((l: any) => Number(l.subdiv) > 0)) notes.push('the page shows level 0 of the card\'s detail layer');
    this.cancel();
    this.clearMeshes();
    this.card = card; this.cardName = card.name || 'character'; this.files = files; this.packages = packages; this.notes = notes; this.failures = failures;
    this.outfit = cardOutfit(card);   // a card that names no outfit wears none, as in an export
    this.show = []; this.hide = []; this.cache = new DressCache(); this.worn = []; this.pieces = [];
    this.host.setBodyOffsets(offsets);
    if (card.phenotype || card.face_shapes) this.host.applyCard(card);
    this.redress();
    return { pieces: packages.size, notes };
  }

  private refsOf(card: any): string[] {
    return [...(card.outfits || []).flatMap((o: any) => (o.pieces || []).map((p: any) => `${o.id}/${p.piece}`)),
      ...(card.wearables || []).map((w: any) => w.id || String(w.file).split('/').pop()!.split('.')[0])];
  }
  // Anny's frame (metres, Z up) to the page's, for offsets (vectors): scale and turn, no move
  private pageOffsets(values: ArrayLike<number> | undefined): Float32Array {
    const b = this.host.body(), R = b.frame.rotation, s = b.frame.scale, n = b.nBody, out = new Float32Array(n * 3);
    if (!values || values.length < n * 3) throw new Error(`the offsets hold ${values ? values.length / 3 : 0} rows; the body has ${n}`);
    for (let i = 0; i < n; i++) {
      const x = values[i * 3], y = values[i * 3 + 1], z = values[i * 3 + 2];
      for (let r = 0; r < 3; r++) out[i * 3 + r] = s * (R[r * 3] * x + R[r * 3 + 1] * y + R[r * 3 + 2] * z);
    }
    return out;
  }

  unload() {
    this.cancel();
    this.clearMeshes();
    // the dress cache holds the packages (their garments and texture bytes) by reference
    this.card = null; this.packages.clear(); this.pieces = []; this.worn = []; this.files = null; this.notes = []; this.failures = 0; this.cache = new DressCache();
    this.host.setBodyOffsets(null);
    this.restoreBody();
    this.host.changed(); this.notify();
  }

  // ---------------------------------------------------------------- the choices
  outfits(): { id: string; label: string }[] { return (this.card?.outfits || []).map((o: any) => ({ id: o.id, label: o.label || o.id })); }
  setOutfit(id: string | null) { this.outfit = id; this.show = []; this.hide = []; this.redress(); }
  // show or hide one piece; a piece of another outfit takes its slot
  setShown(ref: string, on: boolean) {
    this.show = this.show.filter((r) => r !== ref); this.hide = this.hide.filter((r) => r !== ref);
    (on ? this.show : this.hide).push(ref);
    this.redress();
  }
  setKeepSkin(on: boolean) { this.keepSkin = on; this.applyCover(); this.host.changed(); this.notify(); }
  rows(): PieceRow[] {
    if (!this.card) return [];
    let every: WornRef[] = [];
    try { every = wornPieces(this.card, (r) => this.info(r), this.choices(), true); } catch (e) { every = []; }
    const dressed = new Map(this.pieces.map((p) => [p.ref, p]));
    return every.map((p) => {
      const w = this.packages.get(p.ref)!, d = dressed.get(p.ref);
      return { ref: p.ref, piece: p.piece, outfit: p.outfit, label: w.label || w.name, slot: w.slot, layer: w.layer, shown: p.shown,
        hiddenGroups: p.hiddenGroups, groups: w.garment.groups, provenance: w.provenance, pushed: d?.pushed || 0, pushMm: d?.pushMm || 0, vertices: w.garment.count };
    });
  }
  private choices() { return { outfit: this.outfit ?? NO_OUTFIT, show: this.show, hide: this.hide }; }
  private info(ref: string) { const w = this.packages.get(ref)!; return { slot: w.slot, hides: w.hides }; }

  // ---------------------------------------------------------------- dressing
  private bodySurface(): BodySurface {
    const b = this.host.body(), n = b.nBody;
    if (!this.skin) {
      this.skin = skinTriangles(b.quads);
      const K = 8, W = new Float64Array(n * K), I = new Int32Array(n * K), S = b.coarseSkin;
      for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) { W[i * K + k] = S[i * 16 + 8 + k] / 255; I[i * K + k] = S[i * 16 + k]; }
      this.weights = W; this.indices = I;
    }
    const V = Float64Array.from(b.coarse.subarray(0, n * 3)), J = b.joints;
    const surface: BodySurface = new BodySurface({ vertices: V, triangles: this.skin, weights: this.weights!, indices: this.indices!, K: 8, bones: b.bones, frame: b.frame,
      // rigid pieces turn with their bone's skin on the current body (also when the push-out evaluates them on a binding body)
      pose: (bone, _body, w) => estimatedPose(w, surface, J.subarray(bone * 3, bone * 3 + 3)) });
    return surface;
  }
  // the pieces again: a new choice or a new card; the push-out and the covered skin follow in slices
  redress() {
    if (!this.card) return;
    this.cancel();
    try { this.worn = wornPieces(this.card, (r) => this.info(r), this.choices()); }
    catch (e: any) { this.fail(e.message || String(e), e); this.worn = []; }
    try { this.dressNow(); }
    catch (e: any) {
      // a piece that cannot be dressed on this body leaves the wardrobe with a note; the others dress
      if (!this.dropFailing()) { this.giveUp(e); return; }
      try { this.worn = wornPieces(this.card, (r) => this.info(r), this.choices()); this.dressNow(); }
      catch (e2: any) { this.giveUp(e2); return; }
    }
    this.settleNow();
  }
  private giveUp(e: any) {
    this.fail(`the pieces could not be dressed: ${e?.message || e}`, e);
    this.pieces = [];
    try { this.upload(); } catch (e2) { /* the meshes stay hidden */ }
    this.host.changed(); this.notify();
  }
  // the worn pieces that fail alone, out of the card and the packages; whether there were any
  private dropFailing(): boolean {
    const body = this.bodySurface(), bad = new Set<string>();
    for (const p of this.wornWearables()) {
      try { dress(body, [p], { push: false }); }
      catch (e: any) { bad.add(p.ref); this.fail(`${p.ref}: ${e.message || e}`, e); }
    }
    if (!bad.size) return false;
    for (const ref of bad) this.packages.delete(ref);
    for (const o of this.card.outfits || []) o.pieces = (o.pieces || []).filter((p: any) => !bad.has(`${o.id}/${p.piece}`));
    this.card.wearables = (this.card.wearables || []).filter((w: any) => !bad.has(w.id || String(w.file).split('/').pop()!.split('.')[0]));
    this.show = this.show.filter((r) => !bad.has(r)); this.hide = this.hide.filter((r) => !bad.has(r));
    const refs = this.refsOf(this.card);
    for (const k of Object.keys(this.card.wearable_overrides || {})) {
      try { resolveRef(k, refs, cardOutfit(this.card)); } catch (e) { delete this.card.wearable_overrides[k]; }
    }
    return true;
  }
  // the body changed: the pieces follow at once, the push-out and the covered skin when the sliders rest
  bodyChanged() {
    if (!this.card || !this.worn.length) return;
    this.cancel();
    try { this.dressNow(); }
    catch (e: any) { this.redress(); return; }
    this.settle = setTimeout(() => { this.settle = null; this.settleNow(); }, SETTLE_MS);
  }
  private cancel() { clearTimeout(this.settle); this.settle = null; this.job = 0; }
  private wornWearables(): WornPiece[] { return this.worn.map((p) => ({ ...p, wearable: this.packages.get(p.ref)! })); }
  // the bindings and the skin weights alone, at once
  private dressNow() {
    const t0 = performance.now(), timing: Record<string, number> = {};
    this.pieces = dress(this.bodySurface(), this.wornWearables(), { push: false, cache: this.cache, timing });
    const t1 = performance.now();
    this.upload();
    timing.upload = performance.now() - t1;
    timing.total = performance.now() - t0;
    this.setTiming(timing, false);
    this.host.changed(); this.notify();
  }
  // the push-out and the covered skin: steps run in slices of SLICE_MS, each its own task, so that the browser draws
  // frames between them; a new body or choice drops the steps in flight. The pieces, their push-out and the covered
  // skin show together when the last step ends.
  private settleNow() {
    const id = this.job = ++this.jobs, steps = this.settleSteps(), started = performance.now();
    const stat = { slices: 0, longest: 0, work: 0 };
    const run = () => {
      if (this.job !== id) return;
      const t0 = performance.now();
      let r: IteratorResult<void, Record<string, number>>;
      try { do r = steps.next(); while (!r.done && performance.now() - t0 < SLICE_MS); }
      catch (e: any) { this.job = 0; this.fail(`the push-out and the covered skin failed: ${e.message || e}`, e); this.notify(); return; }
      const ms = performance.now() - t0;
      stat.slices++; stat.work += ms; stat.longest = Math.max(stat.longest, ms);
      if (!r.done) { soon(run); return; }
      this.job = 0;
      const timing = r.value;
      timing.total = performance.now() - started;   // from the first slice to the last, with the frames between them
      timing.work = stat.work; timing.slices = stat.slices; timing['longest slice'] = stat.longest;
      this.setTiming(timing, true);
      this.host.changed(); this.notify();
    };
    soon(run);
  }
  private *settleSteps(): Steps<Record<string, number>> {
    const timing: Record<string, number> = {}, body = this.bodySurface();
    // the body's surface, its scales and its grid, each in a step of its own
    const surface = body.surface;
    yield;
    void body.scales;
    yield;
    void surface.grid;
    yield;
    const pieces = yield* dressSteps(body, this.wornWearables(), { push: true, cache: this.cache, timing });
    let t = performance.now();
    const cover = yield* outfitCoverSteps(body, pieces);
    timing.cover = performance.now() - t;
    t = performance.now();
    const index = yield* this.coverIndex(cover);
    timing['body index'] = performance.now() - t;
    t = performance.now();
    this.pieces = pieces; this.cover = cover;
    this.upload(true);
    this.setBodyIndex(index);
    timing.upload = performance.now() - t;
    return timing;
  }
  private setTiming(timing: Record<string, number>, full: boolean) {
    this.timing = Object.fromEntries(Object.entries(timing).map(([k, v]) => [k, Math.round(v * 10) / 10]));
    (this.timing as any).full = full;
  }
  private cover: Uint8Array | null = null;

  // ---------------------------------------------------------------- meshes
  // the dressed pieces into their meshes; `settled` also renews the tangents of the normal maps for the new body
  private upload(settled = false) {
    const keep = new Set(this.pieces.map((p) => p.ref));
    for (const [ref, m] of this.meshes) {
      if (m.wearable !== this.packages.get(ref)) this.dropMesh(ref);
      else if (!keep.has(ref)) m.mesh.visible = false;   // a hidden piece keeps its mesh for the next time it shows
    }
    for (const p of this.pieces) {
      let m = this.meshes.get(p.ref);
      if (!m) m = this.makeMesh(p);
      const at = m.geo.attributes as any;
      at.position.array.set(p.vertices); at.position.needsUpdate = true;
      at.normal.array.set(p.normals); at.normal.needsUpdate = true;
      if (settled && at.gTangent) {
        const g = p.wearable.garment;
        at.gTangent.array.set(tangents(p.vertices, p.normals, g.uvs!, g.triangles)); at.gTangent.needsUpdate = true;
      }
      if (m.weights !== p.weights) {   // the same weights after a slider edit (DressCache)
        at.skinIndex.array.set(p.joints); at.skinIndex.needsUpdate = true;
        at.skinWeight.array.set(p.weights); at.skinWeight.needsUpdate = true;
        m.weights = p.weights;
      }
      const hidden = p.hiddenGroups.slice().sort().join(',');
      if (hidden !== m.hidden) { this.setIndex(m, p); m.hidden = hidden; }
      m.geo.computeBoundingSphere();
      m.mesh.visible = true;
    }
  }
  private makeMesh(p: DressedPiece) {
    const T = this.host.THREE, w = p.wearable, g = w.garment, n = g.count, geo = new T.BufferGeometry();
    geo.setAttribute('position', new T.BufferAttribute(new Float32Array(n * 3), 3));
    geo.setAttribute('normal', new T.BufferAttribute(new Float32Array(n * 3), 3));
    geo.setAttribute('uv', new T.BufferAttribute(g.uvs || new Float32Array(n * 2), 2));
    geo.setAttribute('skinIndex', new T.BufferAttribute(new Uint16Array(n * 4), 4));
    geo.setAttribute('skinWeight', new T.BufferAttribute(new Float32Array(n * 4), 4));
    if (g.uvs && g.materials.some((mt) => mt.textures.normal)) geo.setAttribute('gTangent', new T.BufferAttribute(tangents(p.vertices, p.normals, g.uvs, g.triangles), 4));
    geo.setIndex(new T.BufferAttribute(n < 65536 ? new Uint16Array(g.triangles.length) : new Uint32Array(g.triangles.length), 1));
    const materials = g.materials.map((mt) => this.material(mt, geo));
    // the shadow maps: the page's depth material for opaque materials, a cut at the alpha cutoff for the others
    const depth = g.materials.map((mt, i) => (mt.alphaMode === 'MASK' || mt.alphaMode === 'BLEND' ? this.depthMaterial(mt, materials[i]) : this.host.depth));
    const mesh = new T.SkinnedMesh(geo, materials);
    mesh.bind(this.host.skeleton(), new T.Matrix4());
    mesh.frustumCulled = false; mesh.name = p.ref; mesh.userData.depthMat = depth;
    this.host.scene.add(mesh); this.host.opaque.push(mesh);
    const m = { mesh, geo, wearable: w, materials, depth, hidden: '\u0000', weights: null as Float32Array | null };
    this.meshes.set(p.ref, m);
    return m;
  }
  // the visible triangles, one draw group per material
  private setIndex(m: { geo: THREE_NS.BufferGeometry }, p: DressedPiece) {
    const g = p.wearable.garment, hidden = new Set(p.hiddenGroups.map((h) => g.groups.indexOf(h)));
    const idx = m.geo.index!, A = idx.array as any;
    let k = 0;
    m.geo.clearGroups();
    for (let mat = 0; mat < g.materials.length; mat++) {
      const start = k;
      for (let t = 0; t < g.triangles.length / 3; t++) {
        if (g.triangleMaterial[t] !== mat || hidden.has(g.triangleGroup[t])) continue;
        A[k++] = g.triangles[t * 3]; A[k++] = g.triangles[t * 3 + 1]; A[k++] = g.triangles[t * 3 + 2];
      }
      if (k > start) m.geo.addGroup(start, k - start, mat);
    }
    m.geo.setDrawRange(0, k);
    idx.needsUpdate = true;
  }
  private material(mt: Material, geo: THREE_NS.BufferGeometry) {
    const T = this.host.THREE, defines: Record<string, string> = Object.assign({}, this.host.envDefines());
    const own: Record<string, any> = {
      uBase: { value: new T.Vector4(...(mt.baseColor as [number, number, number, number])) }, uMetal: { value: mt.metallic }, uRough: { value: mt.roughness },
      uEmissive: { value: new T.Vector3(...(mt.emissive as [number, number, number])) }, uCutoff: { value: mt.alphaCutoff }, uNormalScale: { value: mt.normalScale },
      uMap: { value: null }, uMR: { value: null }, uNormalMap: { value: null }, uAO: { value: null }, uEmissiveMap: { value: null },
    };
    const slots: [string, string, string, boolean][] = [['base_color', 'uMap', 'HAS_MAP', true], ['metallic_roughness', 'uMR', 'HAS_MR', false],
      ['normal', 'uNormalMap', 'HAS_NORMAL_MAP', false], ['occlusion', 'uAO', 'HAS_AO', false], ['emissive', 'uEmissiveMap', 'HAS_EMISSIVE_MAP', true]];
    for (const [slot, uniform, define, srgb] of slots) {
      const tex = mt.textures[slot];
      if (!tex || (slot === 'normal' && !geo.attributes.gTangent)) continue;
      own[uniform].value = this.texture(tex, srgb);
      defines[define] = '';
    }
    if (mt.alphaMode === 'MASK') defines.ALPHA_MASK = '';
    if (mt.alphaMode === 'BLEND') defines.ALPHA_BLEND = '';
    return new T.ShaderMaterial({
      uniforms: Object.assign({}, this.host.uniforms, own), vertexShader: GARMENT_VS, fragmentShader: this.host.toMRT(GARMENT_FS), defines,
      glslVersion: T.GLSL3, blending: T.NoBlending, side: mt.doubleSided ? T.DoubleSide : T.FrontSide,
    });
  }
  // the depth of a masked or blended material in the shadow maps: its fragments at or above the alpha cutoff (a masked
  // material's own, BLEND_SHADOW_CUTOFF for a blended one), with the base colour's alpha and texture of the material
  private depthMaterial(mt: Material, shown: any) {
    const T = this.host.THREE, defines: Record<string, any> = Object.assign({}, this.host.depth.defines);
    if (shown.defines.HAS_MAP !== undefined) defines.HAS_MAP = '';
    return new T.ShaderMaterial({
      uniforms: { uBase: shown.uniforms.uBase, uMap: shown.uniforms.uMap, uCutoff: { value: mt.alphaMode === 'MASK' ? mt.alphaCutoff : BLEND_SHADOW_CUTOFF } },
      vertexShader: GARMENT_DEPTH_VS, fragmentShader: GARMENT_DEPTH_FS, defines, side: T.DoubleSide,
    });
  }
  private texture(t: { bytes: Uint8Array; mime: string }, srgb: boolean): THREE_NS.Texture {
    // one texture per image content: a hash of every byte (a few sampled bytes match for images of one size)
    const T = this.host.THREE, key = `${srgb}:${t.bytes.byteLength}:${contentHash(t.bytes)}`;
    let tex = this.textures.get(key);
    if (tex) return tex;
    tex = new T.Texture();
    tex.flipY = false; tex.colorSpace = srgb ? T.SRGBColorSpace : T.NoColorSpace;
    tex.wrapS = tex.wrapT = T.RepeatWrapping; tex.anisotropy = 4;
    createImageBitmap(new Blob([t.bytes as any], { type: t.mime }), { imageOrientation: 'none' as any }).then((bmp) => {
      tex!.image = bmp; tex!.needsUpdate = true; this.host.changed();
    }).catch((e) => console.warn('A garment texture could not be read.', e));
    this.textures.set(key, tex);
    return tex;
  }
  private dropMesh(ref: string) {
    const m = this.meshes.get(ref);
    if (!m) return;
    this.host.scene.remove(m.mesh);
    const i = this.host.opaque.indexOf(m.mesh);
    if (i >= 0) this.host.opaque.splice(i, 1);
    m.geo.dispose(); m.materials.forEach((x) => x.dispose());
    m.depth.forEach((x) => { if (x !== this.host.depth) x.dispose(); });
    this.meshes.delete(ref);
  }
  private clearMeshes() {
    for (const ref of [...this.meshes.keys()]) this.dropMesh(ref);
    for (const t of this.textures.values()) t.dispose();
    this.textures.clear();
  }

  // ---------------------------------------------------------------- the covered skin
  // the body's triangles whose three corners lie under a shown piece move to the end of the index buffer, and the
  // draw range leaves them out (a fine vertex is covered where all the coarse vertices under it are)
  private applyCover() { this.setBodyIndex(runSteps(this.coverIndex(this.cover))); }
  private get restIndex(): Uint32Array {
    return this.bodyIndex || (this.bodyIndex = (this.host.body().geometry.index!.array as Uint32Array).slice());
  }
  // the body's index with the covered triangles at its end and the count of the others, or null for the whole body
  private *coverIndex(cover: Uint8Array | null): Steps<{ index: Uint32Array; count: number } | null> {
    if (this.keepSkin || !cover || !cover.some((x) => x)) return null;
    const I = this.restIndex, n = this.host.body().nBody, mask = new Float32Array(n);
    for (let i = 0; i < n; i++) mask[i] = cover[i];
    // the mask on the fine body, level by level: y = A x for each sparse operator, as the subdivision computes it
    let fine = mask;
    for (const op of this.host.body().subdivision.ops) {
      const y = new Float32Array(op.rows), { start, index, value } = op;
      for (let r0 = 0; r0 < op.rows; r0 += INDEX_SLICE) {
        yield;
        for (let r = r0, r1 = Math.min(op.rows, r0 + INDEX_SLICE); r < r1; r++) {
          let a = 0;
          for (let e = start[r]; e < start[r + 1]; e++) a += value[e] * fine[index[e]];
          y[r] = a;
        }
      }
      fine = y;
    }
    const out = this.indexScratch || (this.indexScratch = new Uint32Array(I.length));   // read only by setBodyIndex
    let k = 0, e = I.length;
    for (let t0 = 0; t0 < I.length; t0 += INDEX_SLICE * 3) {
      yield;
      for (let t = t0, t1 = Math.min(I.length, t0 + INDEX_SLICE * 3); t < t1; t += 3) {
        const a = I[t], b = I[t + 1], c = I[t + 2];
        if (fine[a] > 0.9999 && fine[b] > 0.9999 && fine[c] > 0.9999) { e -= 3; out[e] = a; out[e + 1] = b; out[e + 2] = c; }
        else { out[k++] = a; out[k++] = b; out[k++] = c; }
      }
    }
    return { index: out, count: k };
  }
  private setBodyIndex(r: { index: Uint32Array; count: number } | null) {
    const geo = this.host.body().geometry, idx = geo.index!, A = idx.array as Uint32Array;
    if (r) { A.set(r.index); geo.setDrawRange(0, r.count); }
    else { A.set(this.restIndex); geo.setDrawRange(0, Infinity); }
    idx.needsUpdate = true;
  }
  private restoreBody() {
    if (!this.bodyIndex) return;
    const geo = this.host.body().geometry, idx = geo.index!;
    (idx.array as Uint32Array).set(this.bodyIndex); geo.setDrawRange(0, Infinity); idx.needsUpdate = true;
    this.cover = null;
  }
  // the counts for the interface and the tests
  stats() {
    const geo = this.host.body().geometry, all = this.bodyIndex ? this.bodyIndex.length / 3 : (geo.index?.count || 0) / 3;
    const drawn = Math.min(geo.drawRange.count, geo.index?.count || 0) / 3;
    return { card: this.cardName, pieces: this.pieces.map((p) => ({ ref: p.ref, vertices: p.wearable.garment.count, triangles: visibleTriangles(p).length / 3, pushed: p.pushed, push_mm: +p.pushMm.toFixed(2) })),
      body_triangles: all, hidden_triangles: Math.round(all - drawn), timing: this.timing, notes: this.notes };
  }
}

// FNV-1a of the bytes, as 8 hexadecimal digits
export function contentHash(b: Uint8Array): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < b.length; i++) { h ^= b[i]; h = Math.imul(h, 0x01000193); }
  return (h >>> 0).toString(16).padStart(8, '0');
}

// per-vertex tangents from the UVs (summed over the triangles, made orthogonal to the normal), for normal maps
function tangents(P: Float64Array, N: Float64Array, UV: Float32Array, T: Int32Array): Float32Array {
  const n = P.length / 3, t1 = new Float64Array(n * 3), t2 = new Float64Array(n * 3), out = new Float32Array(n * 4);
  for (let t = 0; t < T.length; t += 3) {
    const a = T[t], b = T[t + 1], c = T[t + 2];
    const x1 = P[b * 3] - P[a * 3], y1 = P[b * 3 + 1] - P[a * 3 + 1], z1 = P[b * 3 + 2] - P[a * 3 + 2];
    const x2 = P[c * 3] - P[a * 3], y2 = P[c * 3 + 1] - P[a * 3 + 1], z2 = P[c * 3 + 2] - P[a * 3 + 2];
    const s1 = UV[b * 2] - UV[a * 2], v1 = UV[b * 2 + 1] - UV[a * 2 + 1], s2 = UV[c * 2] - UV[a * 2], v2 = UV[c * 2 + 1] - UV[a * 2 + 1];
    const r = 1 / ((s1 * v2 - s2 * v1) || 1e-12);
    const sx = (v2 * x1 - v1 * x2) * r, sy = (v2 * y1 - v1 * y2) * r, sz = (v2 * z1 - v1 * z2) * r;
    const tx = (s1 * x2 - s2 * x1) * r, ty = (s1 * y2 - s2 * y1) * r, tz = (s1 * z2 - s2 * z1) * r;
    for (const v of [a, b, c]) { t1[v * 3] += sx; t1[v * 3 + 1] += sy; t1[v * 3 + 2] += sz; t2[v * 3] += tx; t2[v * 3 + 1] += ty; t2[v * 3 + 2] += tz; }
  }
  for (let i = 0; i < n; i++) {
    const o = i * 3, nx = N[o], ny = N[o + 1], nz = N[o + 2], d = t1[o] * nx + t1[o + 1] * ny + t1[o + 2] * nz;
    let x = t1[o] - d * nx, y = t1[o + 1] - d * ny, z = t1[o + 2] - d * nz;
    const l = Math.sqrt(x * x + y * y + z * z) || 1; x /= l; y /= l; z /= l;
    // the handedness: glTF's v runs down the texture, so the bitangent is n x t times w
    const bx = ny * z - nz * y, by = nz * x - nx * z, bz = nx * y - ny * x;
    const w = bx * t2[o] + by * t2[o + 1] + bz * t2[o + 2] < 0 ? -1 : 1;
    out[i * 4] = x; out[i * 4 + 1] = y; out[i * 4 + 2] = z; out[i * 4 + 3] = w;
  }
  return out;
}

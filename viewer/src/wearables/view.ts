// OpenSculptBoy
// Apache License, Version 2.0
//
// The wardrobe of the page: a character card with its outfits and wearable packages from the user's own files,
// dressed on the current body (dress.ts) and drawn as meshes skinned by the page's rig, so that the pieces follow
// the poses and clips. A slider edit moves the pieces at once (the bindings and the skin weights); the push-out and
// the covered skin follow when the sliders rest. The body's triangles under the shown pieces leave its index buffer.
// Nothing of the user's files is stored or sent: the packages live in this page's memory until it closes.

import type * as THREE_NS from 'three';
import { BodySurface, DressCache, dress, estimatedPose, outfitCover, visibleTriangles, type DressedPiece, type Frame, type WornPiece } from './dress.ts';
import { dirname, FileSet, type PickedFile } from './files.ts';
import { readSafetensors, readWearable, type Material, type Wearable } from './package.ts';
import { GARMENT_FS, GARMENT_VS } from './shading.ts';
import { skinTriangles } from './skin.ts';
import { NO_OUTFIT, resolveRef, wornPieces, type WornRef } from './wardrobe.ts';

// what the wardrobe needs from the renderer (main.ts)
export interface WardrobeHost {
  THREE: typeof THREE_NS;
  scene: THREE_NS.Scene;
  opaque: any[];
  skeleton(): THREE_NS.Skeleton;
  uniforms: Record<string, any>;
  envDefines(): Record<string, string>;
  toMRT(fs: string): string;
  body(): { coarse: Float32Array; joints: Float32Array; nBody: number; quads: Uint32Array; coarseSkin: Uint8Array; bones: string[]; frame: Frame;
    fine(mask: Float32Array): Float32Array; geometry: THREE_NS.BufferGeometry };
  setBodyOffsets(offsets: Float32Array | null): void;   // the card's level-0 detail offsets in the page's frame (nBody * 3)
  applyCard(card: any): void;                            // the card's sliders and face into the look
  changed(): void;                                       // a new picture: shadows and accumulation
}

export interface PieceRow { ref: string; piece: string; outfit: string | null; label: string; slot: string; layer: number; shown: boolean;
  hiddenGroups: string[]; groups: string[]; provenance: string; pushed: number; pushMm: number; vertices: number }

const SETTLE_MS = 250;   // the push-out and the covered skin follow a slider edit after this rest

export class Wardrobe {
  host: WardrobeHost;
  card: any = null; cardName = ''; files: FileSet | null = null;
  packages = new Map<string, Wearable>();    // by reference
  outfit: string | null = null; show: string[] = []; hide: string[] = []; keepSkin = false;
  pieces: DressedPiece[] = []; worn: WornRef[] = [];
  notes: string[] = [];
  timing: Record<string, number> = {};
  private meshes = new Map<string, { mesh: any; geo: THREE_NS.BufferGeometry; wearable: Wearable; materials: any[]; hidden: string }>();
  private cache = new DressCache();
  private skin: Int32Array | null = null; private weights: Float64Array | null = null; private indices: Int32Array | null = null;
  private bodyIndex: Uint32Array | null = null;
  private textures = new Map<string, THREE_NS.Texture>();
  private settle: any = null;
  listeners: (() => void)[] = [];

  constructor(host: WardrobeHost) { this.host = host; }
  get loaded() { return !!this.card; }
  private notify() { for (const f of this.listeners) f(); }

  // ---------------------------------------------------------------- loading
  // a card and its packages from the picked files; packages without a card become loose pieces
  async load(picked: PickedFile[]): Promise<{ pieces: number; notes: string[] }> {
    const files = new FileSet(picked), notes: string[] = [];
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
      if (!f) { notes.push(`${e.ref}: ${e.file} is missing`); continue; }
      try {
        let w = byFile.get(f);
        if (!w) {
          const where = dirname(files.pathOf(f));
          w = await readWearable(JSON.parse(await f.text()), (rel) => files.bytes(rel, where));
          byFile.set(f, w);
        }
        packages.set(e.ref, w);
      } catch (err: any) { notes.push(`${e.ref}: ${err.message || err}`); }
    }
    // only the pieces that loaded take part
    const ok = (file: string, ref: string) => packages.has(ref) && !!file;
    card = JSON.parse(JSON.stringify(card));
    for (const o of card.outfits || []) o.pieces = (o.pieces || []).filter((p: any) => ok(p.file, `${o.id}/${p.piece}`));
    card.wearables = (card.wearables || []).filter((w: any) => ok(w.file, w.id || String(w.file).split('/').pop()!.split('.')[0]));
    const refs = this.refsOf(card);
    for (const k of Object.keys(card.wearable_overrides || {})) {
      try { resolveRef(k, refs, card.active_outfit || null); } catch (e) { delete card.wearable_overrides[k]; notes.push(`override ${k}: no such piece`); }
    }
    if ((card.local_changes && Object.keys(card.local_changes).length)) notes.push('the card\'s local changes are not shown in the page');
    // the level-0 offsets of the card's detail layer
    let offsets: Float32Array | null = null;
    const level0 = (card.detail?.levels || []).find((l: any) => Number(l.subdiv) === 0);
    if (level0) {
      try { offsets = this.pageOffsets(readSafetensors(await files.bytes(level0.offsets, folder)).tensors.offsets.values); }
      catch (err: any) { notes.push(`detail level 0: ${err.message || err}`); }
    }
    if ((card.detail?.levels || []).some((l: any) => Number(l.subdiv) > 0)) notes.push('the page shows level 0 of the card\'s detail layer');
    clearTimeout(this.settle); this.settle = null;
    this.clearMeshes();
    this.card = card; this.cardName = card.name || 'character'; this.files = files; this.packages = packages; this.notes = notes;
    this.outfit = card.active_outfit && card.active_outfit !== NO_OUTFIT ? card.active_outfit : (card.outfits?.[0]?.id ?? null);
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
  private pageOffsets(values: ArrayLike<number>): Float32Array {
    const b = this.host.body(), R = b.frame.rotation, s = b.frame.scale, n = b.nBody, out = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const x = values[i * 3], y = values[i * 3 + 1], z = values[i * 3 + 2];
      for (let r = 0; r < 3; r++) out[i * 3 + r] = s * (R[r * 3] * x + R[r * 3 + 1] * y + R[r * 3 + 2] * z);
    }
    return out;
  }

  unload() {
    clearTimeout(this.settle); this.settle = null;
    this.clearMeshes();
    // the dress cache holds the packages (their garments and texture bytes) by reference
    this.card = null; this.packages.clear(); this.pieces = []; this.worn = []; this.files = null; this.notes = []; this.cache = new DressCache();
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
  // the pieces again: a new choice or a new card (with the push-out and the covered skin)
  redress() {
    if (!this.card) return;
    try { this.worn = wornPieces(this.card, (r) => this.info(r), this.choices()); }
    catch (e: any) { this.notes = [...this.notes, e.message || String(e)]; this.worn = []; }
    this.dressNow(true);
  }
  // the body changed: the pieces follow at once, the push-out and the covered skin when the sliders rest
  bodyChanged() {
    if (!this.card || !this.worn.length) return;
    this.dressNow(false);
    clearTimeout(this.settle);
    this.settle = setTimeout(() => { this.settle = null; this.dressNow(true); }, SETTLE_MS);
  }
  private dressNow(full: boolean) {
    const t0 = performance.now(), timing: Record<string, number> = {};
    const body = this.bodySurface();
    const worn: WornPiece[] = this.worn.map((p) => ({ ...p, wearable: this.packages.get(p.ref)! }));
    this.pieces = dress(body, worn, { push: full, cache: this.cache, timing });
    const t1 = performance.now();
    this.upload();
    timing.upload = performance.now() - t1;
    if (full) {
      const t2 = performance.now();
      this.cover = this.keepSkin ? null : outfitCover(body, this.pieces);
      timing.cover = performance.now() - t2;
      const t3 = performance.now();
      this.applyCover();
      timing['body index'] = performance.now() - t3;
    }
    timing.total = performance.now() - t0;
    this.timing = Object.fromEntries(Object.entries(timing).map(([k, v]) => [k, Math.round(v * 10) / 10]));
    (this.timing as any).full = full;
    this.host.changed(); this.notify();
  }
  private cover: Uint8Array | null = null;

  // ---------------------------------------------------------------- meshes
  private upload() {
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
      at.skinIndex.array.set(p.joints); at.skinIndex.needsUpdate = true;
      at.skinWeight.array.set(p.weights); at.skinWeight.needsUpdate = true;
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
    const mesh = new T.SkinnedMesh(geo, materials);
    mesh.bind(this.host.skeleton(), new T.Matrix4());
    mesh.frustumCulled = false; mesh.name = p.ref;
    this.host.scene.add(mesh); this.host.opaque.push(mesh);
    const m = { mesh, geo, wearable: w, materials, hidden: '\u0000' };
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
  private applyCover() {
    const geo = this.host.body().geometry, idx = geo.index!, A = idx.array as Uint32Array;
    if (!this.bodyIndex) this.bodyIndex = A.slice();
    const I = this.bodyIndex, n = this.host.body().nBody;
    if (this.keepSkin || !this.cover || !this.cover.some((x) => x)) { A.set(I); geo.setDrawRange(0, Infinity); idx.needsUpdate = true; return; }
    const mask = new Float32Array(n);
    for (let i = 0; i < n; i++) mask[i] = this.cover[i];
    const fine = this.host.body().fine(mask);
    let k = 0, e = I.length;
    for (let t = 0; t < I.length; t += 3) {
      const a = I[t], b = I[t + 1], c = I[t + 2];
      if (fine[a] > 0.9999 && fine[b] > 0.9999 && fine[c] > 0.9999) { e -= 3; A[e] = a; A[e + 1] = b; A[e + 2] = c; }
      else { A[k++] = a; A[k++] = b; A[k++] = c; }
    }
    geo.setDrawRange(0, k);
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

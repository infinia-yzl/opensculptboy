// OpenSculptBoy
// Apache License, Version 2.0
//
// A character as one file (docs/viewer.md, "Opening files" and "Saving the character"): the card's `look` block
// (schema 5, opensculptboy.character.check_look) that carries what the page shows beyond the model's parameters, the
// card that Save card writes, a .zip of a card with its files (written stored, read stored or deflated), and the card
// in the scene extras of OpenSculptBoy's own GLB and VRM files. These functions hold no DOM and no renderer, so the
// node tests (viewer/test/card_file.mjs) run them as the page does.

import type { Look } from '../ui/app.ts';

export const CARD_SCHEMA = 5;

// ------------------------------------------------------------------ the look block
const r3 = (v: number) => Math.round(v * 1000) / 1000;
const HAIR_KEYS = ['length', 'curl', 'volume', 'density', 'fade'] as const;

// the card's look block of a look of the page
export function lookBlock(L: Look): any {
  const skin: any = { tone: r3(L.skin.tone), undertone: r3(L.skin.undertone) };
  if (L.skin.stone) { skin.stone = L.skin.stone; skin.polish = r3(L.skin.polish ?? 0.5); }
  const hair: any = { style: L.hair.style, color: L.hair.color };
  for (const k of HAIR_KEYS) if (typeof (L.hair as any)[k] === 'number') hair[k] = r3((L.hair as any)[k]);
  if (L.hair.part) hair.part = L.hair.part;
  return { skin, hair, eyes: { color: L.eyes.color } };
}

// a look with the card's look block over it: the block's sections replace the look's, and a card without a block
// keeps the look's colours and hair but shows skin (a stone is the page's choice, which a card carries in its block)
export function withLookBlock(L: Look, block: any): Look {
  const out: Look = JSON.parse(JSON.stringify(L));
  if (!block || typeof block !== 'object') { delete out.skin.stone; delete out.skin.polish; return out; }
  if (block.skin && typeof block.skin === 'object') {
    out.skin = { tone: L.skin.tone, undertone: L.skin.undertone };
    for (const k of ['tone', 'undertone', 'polish'] as const) if (typeof block.skin[k] === 'number') (out.skin as any)[k] = block.skin[k];
    if (typeof block.skin.stone === 'string') out.skin.stone = block.skin.stone;
  } else { delete out.skin.stone; delete out.skin.polish; }
  if (block.hair && typeof block.hair === 'object') out.hair = { ...block.hair, color: block.hair.color || L.hair.color, style: block.hair.style || L.hair.style };
  if (block.eyes && typeof block.eyes.color === 'string') out.eyes = { color: block.eyes.color };
  return out;
}

// ------------------------------------------------------------------ the card of Save card
// the card of the character shown: the open card's own fields (its detail, material and wardrobe) with the page's
// edits (the name, the sliders, the face shapes and the look), or a card of the page's character
export function cardOf(L: Look, base: any = null, extra: { facial_actions?: Record<string, number> } = {}): any {
  const card: any = base ? JSON.parse(JSON.stringify(base)) : { rig: 'anny', topology: 'anny' };
  delete card.schema;
  card.name = L.name || card.name || 'character';
  card.phenotypes = 'all';
  card.phenotype = Object.fromEntries(Object.entries(L.phenotype).map(([k, v]) => [k, r3(v)]));
  card.face_shapes = Object.fromEntries(Object.entries(L.face || {}).filter(([, v]) => v !== 0).map(([k, v]) => [k, r3(v)]));
  if (extra.facial_actions) card.facial_actions = extra.facial_actions;
  card.look = lookBlock(L);
  // the order of the Python card's fields, with the schema first
  const order = ['name', 'rig', 'topology', 'phenotypes', 'phenotype', 'local_changes', 'face_shapes', 'facial_actions', 'pose', 'detail', 'material',
    'wearables', 'outfits', 'active_outfit', 'wearable_overrides', 'look'];
  const out: any = { schema: CARD_SCHEMA };
  for (const k of order) if (k in card) out[k] = card[k];
  for (const k of Object.keys(card)) if (!(k in out)) out[k] = card[k];
  return out;
}

// the files that a card names (its detail levels, its material's maps and parts, its wearable packages), as written
const FILE_EXT = /\.(safetensors|png|jpe?g|webp|ktx2|json|glb|bin)$/i;
export function namedFiles(card: any): string[] {
  const out = new Set<string>();
  const walk = (v: any, key: string) => {
    if (typeof v === 'string') { if (key !== 'name' && key !== 'label' && FILE_EXT.test(v) && !/^https?:/.test(v)) out.add(v); return; }
    if (Array.isArray(v)) { for (const x of v) walk(x, key); return; }
    if (v && typeof v === 'object') for (const [k, x] of Object.entries(v)) if (k !== 'look') walk(x, k);
  };
  walk(card, '');
  return [...out];
}
// a card of sliders alone opens whole; a card that names files needs them
export const needsFiles = (card: any) => namedFiles(card).length > 0;

// ------------------------------------------------------------------ the card in a GLB or VRM file
// the card that OpenSculptBoy's exporters keep in the extras of the scene (opensculptboy.export: extras.opensculptboy
// .character), from a GLB or VRM file's bytes or a .gltf file's text; null when the file holds none
export function cardInGltf(bytes: Uint8Array): any | null {
  let json: any = null;
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.length >= 20 && dv.getUint32(0, true) === 0x46546c67) {
    const len = dv.getUint32(12, true), type = dv.getUint32(16, true);
    if (type !== 0x4e4f534a || 20 + len > bytes.length) return null;
    try { json = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + len))); } catch (e) { return null; }
  } else {
    try { json = JSON.parse(new TextDecoder().decode(bytes)); } catch (e) { return null; }
  }
  if (!json || !Array.isArray(json.scenes)) return null;
  const scenes = [json.scenes[json.scene ?? 0], ...json.scenes];
  for (const s of scenes) {
    const c = s?.extras?.opensculptboy?.character;
    if (c && typeof c === 'object') return c;
  }
  return null;
}

// ------------------------------------------------------------------ zip
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
export function crc32(b: Uint8Array): number { let c = 0xffffffff; for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }

// a .zip of the entries, stored (the maps and the offsets are compressed already), with UTF-8 names
export function writeZip(entries: { path: string; data: Uint8Array }[], date = new Date(2026, 0, 1)): Uint8Array {
  const enc = new TextEncoder();
  const dosTime = (date.getHours() << 11) | (date.getMinutes() << 5) | (date.getSeconds() >> 1);
  const dosDate = ((date.getFullYear() - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();
  const parts: Uint8Array[] = [], central: Uint8Array[] = [];
  let offset = 0;
  for (const e of entries) {
    const name = enc.encode(e.path.replace(/\\/g, '/')), crc = crc32(e.data), n = e.data.length;
    const lh = new Uint8Array(30 + name.length), l = new DataView(lh.buffer);
    l.setUint32(0, 0x04034b50, true); l.setUint16(4, 20, true); l.setUint16(6, 0x0800, true); l.setUint16(8, 0, true);
    l.setUint16(10, dosTime, true); l.setUint16(12, dosDate, true); l.setUint32(14, crc, true); l.setUint32(18, n, true); l.setUint32(22, n, true);
    l.setUint16(26, name.length, true); l.setUint16(28, 0, true); lh.set(name, 30);
    const ch = new Uint8Array(46 + name.length), c = new DataView(ch.buffer);
    c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true); c.setUint16(10, 0, true);
    c.setUint16(12, dosTime, true); c.setUint16(14, dosDate, true); c.setUint32(16, crc, true); c.setUint32(20, n, true); c.setUint32(24, n, true);
    c.setUint16(28, name.length, true); c.setUint32(42, offset, true); ch.set(name, 46);
    parts.push(lh, e.data); central.push(ch);
    offset += lh.length + n;
  }
  const cdSize = central.reduce((a, b) => a + b.length, 0);
  const end = new Uint8Array(22), d = new DataView(end.buffer);
  d.setUint32(0, 0x06054b50, true); d.setUint16(8, entries.length, true); d.setUint16(10, entries.length, true);
  d.setUint32(12, cdSize, true); d.setUint32(16, offset, true);
  const out = new Uint8Array(offset + cdSize + 22);
  let p = 0;
  for (const b of [...parts, ...central, end]) { out.set(b, p); p += b.length; }
  return out;
}

async function inflateRaw(data: Uint8Array): Promise<Uint8Array> {
  const DS = (globalThis as any).DecompressionStream;
  if (!DS) throw new Error('This browser cannot unpack a compressed .zip. Unpack it and open its folder.');
  const stream = new Blob([data as BlobPart]).stream().pipeThrough(new DS('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// the files of a .zip (stored or deflated entries), by their paths; folders and the macOS resource forks left out
export async function readZip(bytes: Uint8Array): Promise<{ path: string; data: Uint8Array }[]> {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let e = -1;
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 65557); i--) if (dv.getUint32(i, true) === 0x06054b50) { e = i; break; }
  if (e < 0) throw new Error('This .zip file is damaged: its directory is missing.');
  const count = dv.getUint16(e + 10, true);
  let p = dv.getUint32(e + 16, true);
  const dec = new TextDecoder(), out: { path: string; data: Uint8Array }[] = [];
  for (let k = 0; k < count; k++) {
    if (dv.getUint32(p, true) !== 0x02014b50) throw new Error('This .zip file is damaged: an entry of its directory is wrong.');
    const method = dv.getUint16(p + 10, true), size = dv.getUint32(p + 20, true);
    const nl = dv.getUint16(p + 28, true), xl = dv.getUint16(p + 30, true), cl = dv.getUint16(p + 32, true), at = dv.getUint32(p + 42, true);
    const path = dec.decode(bytes.subarray(p + 46, p + 46 + nl));
    p += 46 + nl + xl + cl;
    if (path.endsWith('/') || path.startsWith('__MACOSX/') || /(^|\/)\._/.test(path)) continue;
    const start = at + 30 + dv.getUint16(at + 26, true) + dv.getUint16(at + 28, true);
    const raw = bytes.subarray(start, start + size);
    if (method === 0) out.push({ path, data: raw.slice() });
    else if (method === 8) out.push({ path, data: await inflateRaw(raw) });
    else throw new Error(`The .zip file packs ${path} in a way the page cannot read (method ${method}). Unpack it and open its folder.`);
  }
  return out;
}

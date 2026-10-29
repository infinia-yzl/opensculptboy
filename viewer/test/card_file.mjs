// OpenSculptBoy
// Apache License, Version 2.0
//
// A character as one file without a browser (run by test/test_viewer_card_file.py):
//   node viewer/test/card_file.mjs OUT
// It checks the card's look block both ways (src/card/file.ts), Save card's card of the page's character and of an
// open card with its files, the files that a card names, a .zip written and read again (stored, and deflated by
// node's zlib as other tools write it), the card in the scene extras of a GLB and of a .gltf, the stone library
// (src/stones.ts), the starters (src/starters.ts) with the files of a statue's fit, and a statue's props in their
// slot. It writes OUT/card.json and OUT/card_open.json for the Python side to read with
// opensculptboy.character.Character, and prints a JSON summary. A failed check throws.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { CARD_SCHEMA, cardInGltf, cardOf, crc32, lookBlock, namedFiles, needsFiles, readZip, withLookBlock, writeZip } from '../src/card/file.ts';
import { STONES, stoneOf, stoneRoughness } from '../src/stones.ts';
import { STARTERS, STONE_HAIR, finishedPicture, finishedUrl, lightCardUrl, lightLook } from '../src/starters.ts';
import { wornPieces } from '../src/wearables/wardrobe.ts';

const out = process.argv[2] || '.';
fs.mkdirSync(out, { recursive: true });
const look = {
  format: 'anny-viewer/look@4', name: 'Spear Bearer',
  skin: { tone: 0.35, undertone: 0, stone: 'carrara', polish: 0.45 },
  hair: { color: '#dcd7cf', style: 'curly_top_taper', length: 0.8, curl: 0.6, volume: 1, density: 1 },
  eyes: { color: '#dcd7cf' },
  phenotype: { gender: 0, age: 0.8, muscle: 0.7234, weight: 0.5, height: 0.6, proportions: 0.66, african: 0, asian: 1, caucasian: 1 },
  face: { 'nose-hump': 0.2, 'chin-width': 0 },
};

// the look block both ways
const block = lookBlock(look);
assert.deepEqual(block.skin, { tone: 0.35, undertone: 0, stone: 'carrara', polish: 0.45 });
assert.equal(block.hair.style, 'curly_top_taper');
assert.equal(block.eyes.color, '#dcd7cf');
const back = withLookBlock({ ...look, skin: { tone: 0.1, undertone: 0.3 }, hair: { color: '#000000', style: 'lob' } }, block);
assert.deepEqual(back.skin, block.skin);
assert.equal(back.hair.style, 'curly_top_taper');
// a card without a block keeps the page's colours and shows skin
const plain = withLookBlock(look, undefined);
assert.equal(plain.skin.stone, undefined);
assert.equal(plain.hair.color, look.hair.color);

// the card of the page's character
const card = cardOf(look);
assert.equal(card.schema, CARD_SCHEMA);
assert.deepEqual(Object.keys(card), ['schema', 'name', 'rig', 'topology', 'phenotypes', 'phenotype', 'face_shapes', 'look']);
assert.equal(card.phenotype.muscle, 0.723);
assert.deepEqual(card.face_shapes, { 'nose-hump': 0.2 });
assert.ok(!needsFiles(card));
fs.writeFileSync(path.join(out, 'card.json'), JSON.stringify(card, null, 2));

// the card of an open card: its own fields stay, the page's edits replace the sliders and the look
const opened = { schema: 4, name: 'old', rig: 'anny-mouth', topology: 'anny', phenotypes: 'all', phenotype: { muscle: 0.1 }, local_changes: { 'nose-scale': 0.2 },
  detail: { levels: [{ subdiv: 0, offsets: 'card.detail0.safetensors' }] },
  outfits: [{ id: 'travel', label: 'Travel', pieces: [{ piece: 'cape', file: 'wear/cape.wearable.json', enabled: true }] }], active_outfit: 'travel' };
const saved = cardOf(look, opened);
assert.equal(saved.schema, CARD_SCHEMA);
assert.equal(saved.rig, 'anny-mouth');
assert.deepEqual(saved.local_changes, { 'nose-scale': 0.2 });
assert.equal(saved.phenotype.muscle, 0.723);
assert.equal(saved.active_outfit, 'travel');
assert.deepEqual(namedFiles(saved).sort(), ['card.detail0.safetensors', 'wear/cape.wearable.json']);
assert.ok(needsFiles(saved));
fs.writeFileSync(path.join(out, 'card_open.json'), JSON.stringify(saved, null, 2));

// a .zip: stored as Save card writes it, and deflated as other tools write it
assert.equal(crc32(new TextEncoder().encode('123456789')), 0xcbf43926);
const files = [{ path: 'character.json', data: new TextEncoder().encode(JSON.stringify(saved)) }, { path: 'wear/cape.wearable.json', data: new TextEncoder().encode('{"format":"x"}') },
  { path: 'card.detail0.safetensors', data: Uint8Array.from({ length: 4096 }, (_, i) => (i * 7) & 255) }];
const zip = writeZip(files);
fs.writeFileSync(path.join(out, 'stored.zip'), zip);
const read = await readZip(zip);
assert.deepEqual(read.map((e) => e.path), files.map((e) => e.path));
read.forEach((e, i) => assert.deepEqual(Buffer.from(e.data), Buffer.from(files[i].data)));
// a deflated entry, with a folder entry and a macOS resource fork that the reader leaves out
const deflated = zlib.deflateRawSync(Buffer.from(files[2].data));
const mixed = writeZip([{ path: 'folder/', data: new Uint8Array() }, { path: '__MACOSX/._x', data: new Uint8Array(3) }, { path: 'x.safetensors', data: deflated }]);
// mark the third entry deflated with its true size, in the local and the central header
const dv = new DataView(mixed.buffer);
let p = 0;
for (let k = 0; k < 3; k++) { const nl = dv.getUint16(p + 26, true); if (k === 2) { dv.setUint16(p + 8, 8, true); dv.setUint32(p + 22, files[2].data.length, true); } p += 30 + nl + dv.getUint32(p + 18, true); }
for (let k = 0; k < 3; k++) { const nl = dv.getUint16(p + 28, true); if (k === 2) { dv.setUint16(p + 10, 8, true); dv.setUint32(p + 24, files[2].data.length, true); } p += 46 + nl; }
const inflated = await readZip(mixed);
assert.deepEqual(inflated.map((e) => e.path), ['x.safetensors']);
assert.deepEqual(Buffer.from(inflated[0].data), Buffer.from(files[2].data));

// the card in the scene extras of a GLB and of a .gltf
const gltf = { asset: { version: '2.0' }, scene: 0, scenes: [{ nodes: [], extras: { opensculptboy: { generator: 'opensculptboy', character: card } } }] };
let text = JSON.stringify(gltf);
while (text.length % 4) text += ' ';
const json = new TextEncoder().encode(text), glb = new Uint8Array(20 + json.length), g = new DataView(glb.buffer);
g.setUint32(0, 0x46546c67, true); g.setUint32(4, 2, true); g.setUint32(8, glb.length, true); g.setUint32(12, json.length, true); g.setUint32(16, 0x4e4f534a, true);
glb.set(json, 20);
assert.deepEqual(cardInGltf(glb), card);
assert.deepEqual(cardInGltf(new TextEncoder().encode(JSON.stringify(gltf))), card);
assert.equal(cardInGltf(new TextEncoder().encode('{"asset":{},"scenes":[{}]}')), null);
assert.equal(cardInGltf(new Uint8Array([1, 2, 3])), null);

// the stones and the starters
assert.deepEqual(STONES.map((s) => s.id), ['carrara', 'travertine', 'bardiglio', 'basalt', 'bronze', 'verdigris']);
assert.equal(stoneOf({ stone: 'bronze' }).metal, 1);
assert.equal(stoneOf({ stone: '#AABBCC' }).base, '#aabbcc');
assert.equal(stoneOf({ stone: 'marble' }), null);
assert.equal(stoneOf({}), null);
for (const s of STONES) { assert.ok(stoneRoughness(s, 0) > stoneRoughness(s, 1)); assert.ok(STONE_HAIR[s.id]); }
assert.equal(STARTERS.length, 6);
for (const s of STARTERS) {
  assert.ok(STONES.some((x) => x.id === s.light.skin.stone), s.id);
  assert.ok(!/\b(woman|man|girl|boy|youth|old|young|child)\b/i.test(s.name + ' ' + s.description), s.id);
}
// the files of a statue's fit beside the page, and its light card over the table: sliders and face, the stone kept
const spear = STARTERS.find((s) => s.id === 'spear_bearer');
assert.deepEqual([finishedUrl(spear), finishedPicture(spear), lightCardUrl(spear)], ['starters/doryphoros.zip', 'starters/doryphoros.png', 'starters/doryphoros.light.json']);
assert.equal(finishedUrl(STARTERS.find((s) => s.id === 'muse')), 'starters/muse.zip');
const fitted = lightLook(spear, { name: 'Doryphoros', phenotype: { muscle: 0.9 }, face_shapes: { 'nose-hump': 0.1 } });
assert.equal(fitted.name, 'Doryphoros');
assert.equal(fitted.phenotype.muscle, 0.9);
assert.equal(fitted.phenotype.height, spear.light.phenotype.height);
assert.deepEqual(fitted.face, { 'nose-hump': 0.1 });
assert.equal(fitted.skin.stone, 'carrara');
assert.deepEqual(lightLook(spear, null), spear.light);
assert.notEqual(lightLook(spear, null), spear.light);
// a statue's props sit beside each other in their slot, as accessories do; another slot's loose piece replaces
const slots = { base: 'prop', stump: 'prop', sword: 'prop', tunic: 'torso', vest: 'torso' };
const statue = { name: 'statue', outfits: [{ id: 'statue', pieces: [{ piece: 'base', file: 'b' }, { piece: 'stump', file: 's' }, { piece: 'tunic', file: 't' }] }],
  active_outfit: 'statue', wearables: [{ id: 'sword', file: 'w', enabled: true }, { id: 'vest', file: 'v', enabled: true }] };
const worn = wornPieces(statue, (ref) => ({ slot: slots[ref.split('/').pop()], hides: [] })).map((p) => p.ref);
assert.deepEqual(worn.sort(), ['statue/base', 'statue/stump', 'sword', 'vest']);
fs.writeFileSync(path.join(out, 'starters.json'), JSON.stringify(STARTERS.map((s) => cardOf(s.light))));
console.log(JSON.stringify({ ok: true, zip: zip.length, starters: STARTERS.length }));

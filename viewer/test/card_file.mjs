// OpenSculptBoy
// Apache License, Version 2.0
//
// A character as one file without a browser (run by test/test_viewer_card_file.py):
//   node viewer/test/card_file.mjs OUT
// It checks the card's look block both ways (src/card/file.ts), Save card's card of the page's character and of an
// open card with its files, the files that a card names, a .zip written and read again (stored, and deflated by
// node's zlib as other tools write it), the card in the scene extras of a GLB and of a .gltf, the stone library
// (src/stones.ts), the statue starters (src/starters.ts): their pins, and the loader with a synthetic package that is
// missing, of another size, of other bytes or right, a statue's props in their slot, and the floor under the props
// (src/ground.ts) with a synthetic plinth. It writes OUT/card.json and OUT/card_open.json for the Python side to read
// with opensculptboy.character.Character, OUT/starter_pins.json, and prints a JSON summary. A failed check throws.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import crypto from 'node:crypto';
import { CARD_SCHEMA, cardInGltf, cardOf, crc32, lookBlock, namedFiles, needsFiles, readZip, withLookBlock, writeZip } from '../src/card/file.ts';
import { STONES, stoneOf, stoneRoughness } from '../src/stones.ts';
import { DEFAULT_STARTER, STARTERS, STARTER_PINS, STONE_HAIR, downloadUrl, fetchStarter, integrityOf, packageUrl, starterOfName, startersOf } from '../src/starters.ts';
import { groundShift, lowestPieces, lowestSkinnedY } from '../src/ground.ts';
import { STARTER_THUMBS } from '../src/starter_thumbs.ts';
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
// the statues of the New screen: Doryphoros first, each with its pinned package and its picture in the page
assert.deepEqual(STARTERS.map((s) => s.id), ['doryphoros', 'david_goliath']);
assert.equal(DEFAULT_STARTER, 'doryphoros');
for (const s of STARTERS) {
  assert.ok(STONES.some((x) => x.id === s.stone), s.id);
  assert.ok(!/\b(woman|man|girl|boy|youth|old|young|child)\b/i.test(s.name + ' ' + s.description), s.id);
  assert.match(s.package.sha256, /^[0-9a-f]{64}$/);
  assert.ok(s.package.bytes > 1e5, s.id);
  assert.equal(s.package.file, s.id + '.zip');
  assert.equal(packageUrl(s), 'starters/' + s.id + '.zip');
  assert.equal(downloadUrl(s), STARTER_PINS.download + s.id + '.zip');
  assert.match(STARTER_THUMBS[s.id], /^data:image\/(webp|png);base64,/);
  assert.equal(starterOfName(s.package.card), s);
}
assert.match(STARTER_PINS.download, /^https:\/\/github\.com\/infinia-yzl\/opensculptboy\/releases\/download\/[\w.-]+\/$/);
assert.equal(starterOfName('Baseline'), null);
// the integrity string: the SHA-256 in base64, as the browser compares it
const sha = (b) => crypto.createHash('sha256').update(b).digest();
const synthetic = new Uint8Array(writeZip([{ path: 'character.json', data: new TextEncoder().encode('{"schema": 5, "name": "Synthetic statue"}') }]));
assert.equal(integrityOf(sha(synthetic).toString('hex')), 'sha256-' + sha(synthetic).toString('base64'));
// the loader against a synthetic package: a fetch that answers as the page's server would, and checks the integrity as
// the browser does (it refuses the response when the bytes differ from the integrity's hash)
const pins = { format: 'opensculptboy/starter-packages@1', tag: 'test', release: 'https://example.invalid/r', download: 'https://example.invalid/d/',
  packages: { doryphoros: { file: 'doryphoros.zip', bytes: synthetic.length, sha256: sha(synthetic).toString('hex'), card: 'Synthetic statue' } } };
const [dory] = startersOf(pins);
assert.equal(startersOf(pins).length, 1);
const server = (served) => async (url, init = {}) => {
  assert.equal(url, 'starters/doryphoros.zip');
  if (!served) return new Response('Not found.', { status: 404 });
  if (init.method === 'HEAD') return new Response(null, { status: 200, headers: { 'Content-Length': String(served.length) } });
  if (init.integrity && init.integrity !== 'sha256-' + sha(served).toString('base64')) throw new TypeError('Failed to fetch: integrity');
  return new Response(served, { status: 200 });
};
const asFile = await fetchStarter(dory, { pins, protocol: 'file:', fetch: () => { throw new Error('a page opened as a file fetches nothing'); } });
assert.equal(asFile.note.problem, 'file');
assert.equal(asFile.note.link, 'https://example.invalid/d/doryphoros.zip');
const missing = await fetchStarter(dory, { pins, protocol: 'http:', fetch: server(null) });
assert.equal(missing.ok, false);
assert.equal(missing.note.problem, 'missing');
assert.match(missing.note.message, /doryphoros\.zip \(0\.0 MB\) is not in the starters folder beside this page/);
assert.equal(missing.note.label, 'Download doryphoros.zip');
const longer = new Uint8Array(synthetic.length + 7); longer.set(synthetic);
assert.equal((await fetchStarter(dory, { pins, protocol: 'http:', fetch: server(longer) })).note.problem, 'other');
const changed = synthetic.slice(); changed[changed.length - 1] ^= 0xff;
const other = await fetchStarter(dory, { pins, protocol: 'http:', fetch: server(changed) });
assert.equal(other.note.problem, 'other');
assert.match(other.note.message, /another version than the one this page pins/);
const right = await fetchStarter(dory, { pins, protocol: 'http:', fetch: server(synthetic) });
assert.equal(right.ok, true);
assert.equal(right.file.name, 'doryphoros.zip');
assert.deepEqual(new Uint8Array(await right.file.arrayBuffer()), synthetic);
assert.equal((await readZip(new Uint8Array(await right.file.arrayBuffer())))[0].path, 'character.json');
fs.writeFileSync(path.join(out, 'starter_pins.json'), JSON.stringify(STARTER_PINS));
// the floor under the props (src/ground.ts): a synthetic plinth of two triangles' corners, 0.1 below the soles, held
// by the root bone, and a hat on the head; bone matrices that lift the root by 0.3 and turn the head
const ident = () => { const m = new Float32Array(16); m[0] = m[5] = m[10] = m[15] = 1; return m; };
const mats = new Float32Array(32);
mats.set(ident(), 0); mats[13] = 0.3;
mats.set(ident(), 16); mats[16 + 13] = 0.3 + 0.05;
const plinth = { slot: 'prop', vertices: [-0.3, -0.95, -0.3, 0.3, -0.95, -0.3, 0.3, -0.95, 0.3, -0.3, -0.85, 0.3], joints: new Uint16Array(16), weights: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] };
const hat = { slot: 'prop', vertices: [0, 0.8, 0, 0.1, 0.85, 0], joints: [1, 0, 0, 0, 1, 0, 0, 0], weights: [1, 0, 0, 0, 1, 0, 0, 0] };
const boots = { slot: 'feet', vertices: [0, -1.2, 0], joints: [0, 0, 0, 0], weights: [1, 0, 0, 0] };
assert.ok(Math.abs(lowestSkinnedY(plinth, mats) - (-0.65)) < 1e-6);
assert.ok(Math.abs(lowestSkinnedY(hat, mats) - 1.15) < 1e-6);
// a half and half weight between the two bones
assert.ok(Math.abs(lowestSkinnedY({ vertices: [0, 0, 0], joints: [0, 1, 0, 0], weights: [0.5, 0.5, 0, 0] }, mats) - 0.325) < 1e-6);
assert.equal(lowestSkinnedY({ vertices: [], joints: [], weights: [] }, mats), Infinity);
// only the props count: a piece of another slot leaves the floor alone
assert.ok(Math.abs(lowestPieces([plinth, hat, boots], mats) - (-0.65)) < 1e-6);
assert.equal(lowestPieces([boots], mats), Infinity);
// the soles at -0.5 after the pose and the floor at -0.8415: the plinth's bottom goes to the floor, the body stands on it
const floor = -0.8415;
const lifted = groundShift(floor, -0.5, lowestPieces([plinth, hat], mats));
assert.ok(Math.abs(lifted.ground - (floor + 0.65)) < 1e-6);
assert.ok(Math.abs(lifted.lift - 0.15) < 1e-6);
// the props switched off: the body's soles go back to the floor
const bare = groundShift(floor, -0.5, lowestPieces([], mats));
assert.ok(Math.abs(bare.ground - (floor + 0.5)) < 1e-9);
assert.equal(bare.lift, 0);
// a prop above the soles (a hat) leaves the body on its own ground
assert.equal(groundShift(floor, -0.5, lowestPieces([hat], mats)).lift, 0);
// a statue's props sit beside each other in their slot, as accessories do; another slot's loose piece replaces
const slots = { base: 'prop', stump: 'prop', sword: 'prop', tunic: 'torso', vest: 'torso' };
const statue = { name: 'statue', outfits: [{ id: 'statue', pieces: [{ piece: 'base', file: 'b' }, { piece: 'stump', file: 's' }, { piece: 'tunic', file: 't' }] }],
  active_outfit: 'statue', wearables: [{ id: 'sword', file: 'w', enabled: true }, { id: 'vest', file: 'v', enabled: true }] };
const worn = wornPieces(statue, (ref) => ({ slot: slots[ref.split('/').pop()], hides: [] })).map((p) => p.ref);
assert.deepEqual(worn.sort(), ['statue/base', 'statue/stump', 'sword', 'vest']);
console.log(JSON.stringify({ ok: true, zip: zip.length, starters: STARTERS.length }));

// OpenSculptBoy
// Apache License, Version 2.0
//
// Node checks of the pose editor's parts that need no browser, run by test/test_pose_edit_node.py:
//   node viewer/test/pose_edit.mjs
// - the light scan of a starter: a mesh compressed with EXT_meshopt_compression (16-bit positions, the node's scale and
//   translation) reads back into Anny's axes of the card frame (src/pose_edit/scan.ts);
// - the BVH's rays (every crossing and the nearest) and closest points against a brute force (src/pose_edit/bvh.ts);
// - the numbered point set of the screens and its plain names (src/pose_edit/points.ts);
// - the editor's keys in the action registry: the ids and keys of key map version 2, no two actions on a key in one
//   scope, and no key the browser keeps (src/pose_edit/editor.ts, src/studio/actions.ts).
// - names joined into the messages' sentences, one, two and three of them (src/ui/words.ts, Grip's warning, Place
//   points' kept chains);
// - the floor under an edited pose: each edit stands on its own floor, and a modal turn holds it until its end.
// - the figure, the scan or both (src/pose_edit/display.ts): the swap of Shift+Z, the figure and the scan never both
//   hidden, a character without a scan always showing its figure, the status line's note, and the display menu's
//   switches with their ticks and their greyed state.
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { MeshoptEncoder } from 'meshoptimizer';

globalThis.window = globalThis;
await import('../src/meshopt_ref.js');
const { readScan } = await import('../src/pose_edit/scan.ts');
const { TriangleBVH, closestOnTriangle } = await import('../src/pose_edit/bvh.ts');
const { POINTS, POINT_BY_NUMBER, plainName, handPoints } = await import('../src/pose_edit/points.ts');

// ------------------------------------------------------------------ a sphere of triangles
const P = [], I = [];
const nu = 120, nv = 60;
for (let i = 0; i <= nv; i++) for (let j = 0; j <= nu; j++) {
  const th = Math.PI * i / nv, ph = 2 * Math.PI * j / nu;
  P.push(0.3 * Math.sin(th) * Math.cos(ph) + 0.1, 0.3 * Math.sin(th) * Math.sin(ph) - 0.2, 0.3 * Math.cos(th) + 1.0);
}
for (let i = 0; i < nv; i++) for (let j = 0; j < nu; j++) { const a = i * (nu + 1) + j, b = a + 1, c = a + nu + 1, d = c + 1; I.push(a, c, b, b, c, d); }

// ------------------------------------------------------------------ the scan: encode as a starter carries it, read it back
await MeshoptEncoder.ready;
const n = P.length / 3;
// glTF axes: (x, y, z)_gltf = (x, z, -y)_anny
const G = new Float64Array(n * 3);
for (let i = 0; i < n; i++) { G[i * 3] = P[i * 3]; G[i * 3 + 1] = P[i * 3 + 2]; G[i * 3 + 2] = -P[i * 3 + 1]; }
const lo = [0, 1, 2].map((k) => Math.min(...Array.from({ length: n }, (_, i) => G[i * 3 + k])));
const hi = [0, 1, 2].map((k) => Math.max(...Array.from({ length: n }, (_, i) => G[i * 3 + k])));
const s = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 65535;
const q = new Uint16Array(n * 4);
for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) q[i * 4 + k] = Math.round((G[i * 3 + k] - lo[k]) / s);
const pos = MeshoptEncoder.encodeGltfBuffer(new Uint8Array(q.buffer), n, 8, 'ATTRIBUTES');
const idx = MeshoptEncoder.encodeGltfBuffer(new Uint8Array(Uint32Array.from(I).buffer), I.length, 4, 'TRIANGLES');
const pad = (b) => { const out = new Uint8Array(Math.ceil(b.length / 4) * 4); out.set(b); return out; };
const bin = new Uint8Array([...pad(pos), ...pad(idx)]);
const gltf = {
  asset: { version: '2.0' }, extensionsUsed: ['EXT_meshopt_compression', 'KHR_mesh_quantization'], extensionsRequired: ['EXT_meshopt_compression', 'KHR_mesh_quantization'],
  scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0, scale: [s, s, s], translation: lo }],
  meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1, mode: 4 }] }],
  buffers: [{ byteLength: bin.length }, { byteLength: n * 8 + I.length * 4, extensions: { EXT_meshopt_compression: { fallback: true } } }],
  bufferViews: [
    { buffer: 1, byteOffset: 0, byteLength: n * 8, byteStride: 8, extensions: { EXT_meshopt_compression: { buffer: 0, byteOffset: 0, byteLength: pos.length, byteStride: 8, mode: 'ATTRIBUTES', count: n } } },
    { buffer: 1, byteOffset: n * 8, byteLength: I.length * 4, extensions: { EXT_meshopt_compression: { buffer: 0, byteOffset: pad(pos).length, byteLength: idx.length, byteStride: 4, mode: 'TRIANGLES', count: I.length } } },
  ],
  accessors: [{ bufferView: 0, componentType: 5123, count: n, type: 'VEC3' }, { bufferView: 1, componentType: 5125, count: I.length, type: 'SCALAR' }],
};
const js = pad(new TextEncoder().encode(JSON.stringify(gltf)).map((c) => c));
for (let i = new TextEncoder().encode(JSON.stringify(gltf)).length; i < js.length; i++) js[i] = 0x20;
const glb = new Uint8Array(12 + 8 + js.length + 8 + bin.length), dv = new DataView(glb.buffer);
dv.setUint32(0, 0x46546c67, true); dv.setUint32(4, 2, true); dv.setUint32(8, glb.length, true);
dv.setUint32(12, js.length, true); dv.setUint32(16, 0x4e4f534a, true); glb.set(js, 20);
dv.setUint32(20 + js.length, bin.length, true); dv.setUint32(24 + js.length, 0x004e4942, true); glb.set(bin, 28 + js.length);
const scan = readScan(glb);
assert.equal(scan.positions.length, P.length);
assert.deepEqual(Array.from(scan.indices), I);
let err = 0;
for (let i = 0; i < P.length; i++) err = Math.max(err, Math.abs(scan.positions[i] - P[i]));
assert.ok(err <= s, `the scan reads back within its quantisation step (${err} > ${s})`);
// smooth normals lie along the sphere's radius (away from the poles, where the strip's triangles meet in a point)
for (let i = nu + 1; i < n - nu - 1; i += 97) {
  const d = [scan.positions[i * 3] - 0.1, scan.positions[i * 3 + 1] + 0.2, scan.positions[i * 3 + 2] - 1.0], l = Math.hypot(...d);
  assert.ok(Math.abs(d[0] * scan.normals[i * 3] + d[1] * scan.normals[i * 3 + 1] + d[2] * scan.normals[i * 3 + 2]) / l > 0.99);
}
console.log(`scan: ${n} vertices, ${I.length / 3} triangles, ${glb.length} bytes, read back within ${err.toExponential(2)} m`);

// ------------------------------------------------------------------ the BVH against a brute force
const Pa = Float64Array.from(P), Ia = Uint32Array.from(I), bvh = new TriangleBVH(Pa, Ia);
let seed = 7;
const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
for (let k = 0; k < 100; k++) {
  const o = [rnd() * 2 - 0.9, rnd() * 2 - 1.2, rnd() * 2], d = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5];
  const all = bvh.rayAll(o, d), near = bvh.ray(o, d), hit = { t: 0, tri: 0, u: 0, v: 0 };
  let count = 0;
  for (let t = 0; t < Ia.length / 3; t++) if (bvh.hitTri(t, o[0], o[1], o[2], d[0], d[1], d[2], hit)) count++;
  assert.equal(all.length, count);
  if (near) assert.equal(near.t, all[0].t);
  const c = bvh.closest(o), tmp = [0, 0, 0];
  let best = Infinity;
  for (let t = 0; t < Ia.length / 3; t++) best = Math.min(best, closestOnTriangle(Pa, Ia, t, o[0], o[1], o[2], tmp));
  assert.ok(Math.abs(c.distance - Math.sqrt(best)) < 1e-12);
}
console.log('bvh: rays and closest points match a brute force');

// ------------------------------------------------------------------ the points as the screens number them
assert.equal(POINTS.length, 57);
assert.equal(POINT_BY_NUMBER.get(1).name, 'head_top');
assert.equal(POINT_BY_NUMBER.get(9).label, 'Left knee');
assert.equal(POINT_BY_NUMBER.get(16).name, 'wrist.R');
assert.equal(POINT_BY_NUMBER.get(22).label, 'Right index finger, middle joint');
assert.equal(POINT_BY_NUMBER.get(28).label, 'Right middle finger tip');
assert.equal(POINT_BY_NUMBER.get(31).label, 'Right ring finger, end joint');
assert.equal(POINT_BY_NUMBER.get(36).label, 'Right little finger tip');
assert.equal(POINT_BY_NUMBER.get(37).name, 'wrist.L');
assert.equal(plainName('thumb_cmc.L'), 'Left thumb, base');
assert.deepEqual(handPoints('R').map((p) => p.mediapipe), [...Array(21).keys()]);
for (const p of POINTS) assert.ok(!/[._]/.test(p.label), p.label);
console.log('points: 57, numbered as the screens');

// ------------------------------------------------------------------ the chains a placement turns
const { solvableChains } = await import('../src/pose_edit/session.ts');
assert.deepEqual(solvableChains([
  // the right arm: four points over three bones; the left arm: one point; the left leg: four points on one bone
  { chain: 'arm.r', bone: 1 }, { chain: 'arm.r', bone: 2 }, { chain: 'arm.r', bone: 3 }, { chain: 'arm.r', bone: 3 },
  { chain: 'arm.l', bone: 7 },
  { chain: 'leg.l', bone: 9 }, { chain: 'leg.l', bone: 9 }, { chain: 'leg.l', bone: 9 }, { chain: 'leg.l', bone: 9 },
]), ['arm.r']);
console.log('placement: only the chains with four points on two bones or more turn');

// ------------------------------------------------------------------ the editor's keys
const actions = await import('../src/studio/actions.ts');
const { installPoseEditor } = await import('../src/pose_edit/editor.ts');
const host = { THREE, ready: () => false, canvas: {}, camera: new THREE.PerspectiveCamera() };
installPoseEditor(host);
const want = { 'pose.edit': ['Tab'], 'pose.tool.select': ['Q'], 'pose.tool.move': ['W'], 'pose.tool.turn': ['E'], 'pose.grab': ['G'], 'pose.parent': ['['],
  'pose.child': [']'], 'pose.select-all': ['A'], 'pose.reset-point': ['Alt+G'], 'pose.reset-turn': ['Alt+R'], 'pose.push': ['PageDown'], 'pose.pull': ['PageUp'],
  'pose.nudge': ['ArrowLeft'], 'pose.swap': ['Shift+Z'], 'pose.tool.1': ['1'] };
for (const [id, keys] of Object.entries(want)) {
  const a = actions.action(id);
  assert.ok(a, `${id} is registered`);
  for (const k of keys) assert.ok(a.keys.includes(actions.normChord(k)), `${id} has ${k}`);
}
assert.deepEqual(actions.conflicts().filter((c) => c.scope.startsWith('pose')), []);
assert.deepEqual(actions.reservedUses().filter((u) => u.id.startsWith('pose.')), []);
console.log('keys: the pose editor\'s actions are registered without conflicts');

// ------------------------------------------------------------------ names joined into a sentence
const { listOf, fingersPhrase } = await import('../src/ui/words.ts');
const { gripWarning } = await import('../src/pose_edit/hands.ts');
const { keptMessage } = await import('../src/pose_edit/editor.ts');
assert.equal(listOf([]), '');
assert.equal(listOf(['wrist.R']), 'wrist.R');
assert.equal(listOf(['wrist.R', 'jaw']), 'wrist.R and jaw');
assert.equal(listOf(['arm', 'leg', 'spine']), 'arm, leg and spine');
assert.equal(listOf(['arm', 'leg', 'spine'], 'or'), 'arm, leg or spine');
assert.equal(gripWarning(['index'], 'sword'), 'The index finger stays inside the sword, at its present curl and straight alike.');
assert.equal(gripWarning(['middle', 'index'], 'sword'), 'The index and middle fingers stay inside the sword, at their present curl and straight alike.');
assert.equal(gripWarning(['index', 'middle', 'ring'], 'sword'), 'The index, middle and ring fingers stay inside the sword, at their present curl and straight alike.');
assert.equal(gripWarning(['thumb'], 'rod'), 'The thumb stays inside the rod, at its present curl and straight alike.');
assert.equal(fingersPhrase(['thumb', 'index']).text, 'the thumb and index finger');
assert.equal(fingersPhrase(['little', 'thumb', 'ring']).text, 'the thumb, ring and little fingers');
assert.equal(gripWarning([], 'rod'), 'A finger stays inside the rod, at its present curl and straight alike.');
assert.equal(keptMessage(['arm.l']), 'The left arm kept its pose: too few sure points.');
assert.equal(keptMessage(['arm.l', 'spine']), 'The left arm and the spine kept their pose: too few sure points.');
assert.equal(keptMessage(['arm.l', 'leg.r', 'neck']), 'The left arm, the right leg and the head and neck kept their pose: too few sure points.');
console.log('words: one, two and three names joined without a comma before "and"');

// ------------------------------------------------------------------ the floor under an edited pose
// The editor stands each edit on its own floor (host.setPose with hold false), holds the floor while a modal grab or
// turn lasts (hold true), and settles it on the first frame after the gesture, once; Edit off settles a held floor.
{
  const { startModal, confirmModal } = await import('../src/studio/modal.ts');
  globalThis.addEventListener ??= () => {}; globalThis.removeEventListener ??= () => {};
  const ed = installPoseEditor(host), calls = [];
  const I = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  Object.assign(host, { restTurns: () => null, parents: () => [-1, 0], bones: () => ['root', 'b'], root: () => 0, frame: () => ({ scale: 1, rotation: I, offset: [0, 0, 0] }),
    setPose: (q, label, hold) => calls.push(!!hold) });
  ed.session = { rotations: [I.slice(), I.slice()], places: {}, shift: [0, 0, 0], syncFrame: () => {} };
  ed.poseChanged();
  assert.deepEqual(calls, [false], 'an edit outside a gesture stands on its floor');
  ed.frame();
  assert.deepEqual(calls, [false], 'a settled floor needs no new pose');
  startModal({ kind: 'turn', label: 'b', update: () => { ed.poseChanged(); }, confirm: () => {}, cancel: () => {} }, { x: 0, y: 0 });
  assert.deepEqual(calls, [false, true], 'a modal turn holds the floor');
  ed.frame();
  assert.deepEqual(calls, [false, true], 'the floor holds while the turn lasts');
  confirmModal();
  ed.frame(); ed.frame();
  assert.deepEqual(calls, [false, true, false], 'the frame after the turn settles the floor once');
  startModal({ kind: 'turn', label: 'b', update: () => { ed.poseChanged(); }, confirm: () => {}, cancel: () => {} }, { x: 0, y: 0 });
  ed.settleFloor(true);
  assert.deepEqual(calls, [false, true, false, true, false], 'Edit off settles a held floor');
  confirmModal();
  ed.session = null;
  console.log('floor: each edit stands on its own floor; a gesture holds it until its end');
}

// ------------------------------------------------------------------ the figure, the scan or both
{
  const { showFigure, showScan, swapShown, effective, shownNote } = await import('../src/pose_edit/display.ts');
  const B = { figure: true, scan: true }, F = { figure: true, scan: false }, S = { figure: false, scan: true }, N = { figure: false, scan: false };
  // Shift+Z: the figure alone and the scan alone in turn; both shown become the scan alone
  assert.deepEqual(swapShown(F, true), S);
  assert.deepEqual(swapShown(S, true), F);
  assert.deepEqual(swapShown(B, true), S);
  assert.deepEqual(swapShown(swapShown(B, true), true), F);
  // hiding one while the other is hidden shows the other
  assert.deepEqual(showFigure(F, false, true), S);
  assert.deepEqual(showScan(S, false, true), F);
  assert.deepEqual(showFigure(B, false, true), S);
  assert.deepEqual(showScan(B, false, true), F);
  assert.deepEqual(showFigure(S, true, true), B);
  assert.deepEqual(showScan(F, true, true), B);
  // never both hidden, from any state and by any step; without a scan the figure always shows
  for (const st of [B, F, S, N]) for (const has of [true, false]) {
    for (const r of [showFigure(st, false, has), showFigure(st, true, has), showScan(st, false, has), showScan(st, true, has), swapShown(st, has)]) {
      if (has) assert.ok(r.figure || r.scan, JSON.stringify([st, has, r]));
      else assert.ok(r.figure, JSON.stringify([st, has, r]));
    }
    const e = effective(st, has);
    assert.ok(e.figure || e.scan);
  }
  assert.deepEqual(effective(N, true), F);
  assert.equal(shownNote(S, true), 'Figure hidden');
  assert.equal(shownNote(F, true), 'Scan hidden');
  assert.equal(shownNote(B, true), '');
  assert.equal(shownNote(S, false), '');

  // the editor: the display menu's switches, their ticks and the status line, on a character with a scan
  const { ui } = await import('../src/shell/state.ts');
  const { statusText } = await import('../src/studio/status.ts');
  const { editStore } = await import('../src/pose_edit/editor.ts');
  if (!actions.scopeOf('pose')) actions.defineScope({ id: 'pose', label: 'Pose', active: () => ui.get().mode === 'pose' });
  const ed = installPoseEditor(host), figure = [];
  Object.assign(host, { redraw: () => {}, setFigureShown: (on) => figure.push(on) });
  ui.set({ mode: 'pose' });
  editStore.set({ on: true });
  const ids = () => actions.menuItems('display').map((a) => a.id);
  const ticks = () => Object.fromEntries(actions.menuItems('display').map((a) => [a.id, a.checked ? a.checked() : null]));
  const greyed = () => Object.fromEntries(actions.menuItems('display').map((a) => [a.id, a.unavailable ? a.unavailable() : '']));
  assert.deepEqual(ids(), ['pose.figure', 'pose.scan', 'pose.scan-see']);
  // no scan: the switches stay in their places, greyed out with the reason; Shift+Z has nothing to swap
  assert.deepEqual(greyed(), { 'pose.figure': 'This character has no scan', 'pose.scan': 'This character has no scan', 'pose.scan-see': 'This character has no scan' });
  assert.equal(actions.runAction('pose.swap'), false);
  assert.equal(actions.runAction('pose.figure'), false);
  assert.equal(ed.figureShown(), true);
  // a scan
  ed.scan = { positions: new Float32Array(9), indices: new Uint32Array(3), normals: new Float32Array(9) };
  editStore.set({ scan: { ...editStore.get().scan, available: true, loading: false } });
  assert.deepEqual(greyed(), { 'pose.figure': '', 'pose.scan': '', 'pose.scan-see': '' });
  assert.deepEqual(ticks(), { 'pose.figure': true, 'pose.scan': true, 'pose.scan-see': true });
  assert.equal(actions.runAction('pose.swap'), true);
  assert.deepEqual(ticks(), { 'pose.figure': false, 'pose.scan': true, 'pose.scan-see': true });
  assert.equal(ed.figureShown(), false);
  assert.equal(figure.at(-1), false);
  assert.ok(statusText().startsWith('Figure hidden'), statusText());
  actions.runAction('pose.swap');
  assert.deepEqual(ticks(), { 'pose.figure': true, 'pose.scan': false, 'pose.scan-see': true });
  assert.equal(figure.at(-1), true);
  assert.ok(statusText().startsWith('Scan hidden'), statusText());
  assert.equal(greyed()['pose.scan-see'], 'The scan is hidden');
  // the switches: the scan on, the figure off, then the scan off shows the figure again
  actions.runAction('pose.scan');
  assert.deepEqual(ticks(), { 'pose.figure': true, 'pose.scan': true, 'pose.scan-see': true });
  assert.ok(!statusText().includes('hidden'), statusText());
  actions.runAction('pose.figure');
  assert.deepEqual(ticks(), { 'pose.figure': false, 'pose.scan': true, 'pose.scan-see': true });
  actions.runAction('pose.scan');
  assert.deepEqual(ticks(), { 'pose.figure': true, 'pose.scan': false, 'pose.scan-see': true });
  // see-through: on is the ghost, off is solid
  actions.runAction('pose.scan');
  actions.runAction('pose.scan-see');
  assert.equal(editStore.get().scan.look, 'solid');
  assert.equal(ticks()['pose.scan-see'], false);
  actions.runAction('pose.scan-see');
  assert.equal(editStore.get().scan.look, 'ghost');
  // the figure hides only in Pose: another mode shows it, and Pose hides it again
  actions.runAction('pose.figure');
  assert.equal(figure.at(-1), false);
  ui.set({ mode: 'create' });
  assert.equal(figure.at(-1), true);
  assert.ok(!statusText().includes('Figure hidden'), statusText());
  ui.set({ mode: 'pose' });
  assert.equal(figure.at(-1), false);
  // the card's scan gone: the figure shows
  ed.scan = null;
  editStore.set({ scan: { ...editStore.get().scan, available: false } });
  ed.syncFigure();
  assert.equal(figure.at(-1), true);
  assert.equal(ed.figureShown(), true);
  editStore.set({ on: false, figure: true, scan: { ...editStore.get().scan, shown: true, look: 'ghost' } });
  ed.syncFigure();
  console.log('display: Shift+Z swaps the figure and the scan, never both hidden, ticks and greyed switches as the state is');
}

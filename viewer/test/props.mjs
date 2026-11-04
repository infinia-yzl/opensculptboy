// OpenSculptBoy
// Apache License, Version 2.0
//
// Node checks of the studio's props, moved by hand (run by test/test_wearable_place.py, which hands in a case of
// opensculptboy.wearables.place's numbers):
//   node viewer/test/props.mjs ['{"place": ..., "head": ..., "points": ..., "vertices": ..., "normals": ...}']
// - a place's move in Anny's frame and in the page's frame (src/wearables/place.ts) against the Python twin;
// - a place and its matrix, a change of the bone that holds it (the piece stays where it stands in a pose), a move of
//   the posed piece, and the panel's numbers back and forth (src/pose_edit/props.ts);
// - the card: a piece's place written and read, schema 6 for a card with a place and 5 without (src/card/file.ts),
//   and the place carried to the worn pieces (src/wearables/wardrobe.ts);
// - a rigid piece dressed with a place: moved about its new bone's head, on that bone alone (src/wearables/dress.ts);
// - the handles' hit tests, the choices of Held by, and the prop keys in the action registry without a conflict.
import assert from 'node:assert/strict';
import * as THREE from 'three';

globalThis.window = globalThis;
const P = await import('../src/wearables/place.ts');
const T = await import('../src/pose_edit/props.ts');
const { cardOf, CARD_SCHEMA } = await import('../src/card/file.ts');
const { wornPieces } = await import('../src/wearables/wardrobe.ts');
const { BodySurface, dress, placedPiece } = await import('../src/wearables/dress.ts');

const close = (a, b, tol, what) => {
  assert.equal(a.length, b.length, what);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) <= tol, `${what}: ${a[i]} against ${b[i]} at ${i}`);
};
const PAGE = { scale: 0.8916, rotation: [1, 0, 0, 0, 0, 1, 0, -1, 0], offset: [0.01, 0.62, 0.03] };
const IDENT = { scale: 1, rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], offset: [0, 0, 0] };
const turn = (axis, deg) => { const a = axis.map((x) => x / Math.hypot(...axis)), h = deg * Math.PI / 360; return [...a.map((x) => x * Math.sin(h)), Math.cos(h)]; };
const rigid = (seed) => {
  let s = seed;
  const rnd = () => { s = (s * 16807) % 2147483647; return s / 2147483647 - 0.5; };
  return P.m4(P.rotationMatrix([rnd(), rnd(), rnd(), rnd()]), [rnd() * 0.4, rnd() * 0.4, rnd() * 0.4]);
};

// ------------------------------------------------------------------ the Python twin's numbers
const arg = process.argv[2];
const c = arg ? JSON.parse(arg) : { place: { bone: 'wrist.R', rotation: turn([0, 0, 1], 30), translation: [0.01, 0.02, -0.03] }, head: [0.02, -0.04, 1.3],
  points: [[0.1, 0, 1.2], [0, 0.05, 1.25]], vertices: null, normals: null };
const X = c.points.flat(), head = c.head;
const anny = P.applyPlace(c.place, X, X, head, IDENT);
if (c.vertices) { close(anny.vertices, c.vertices.flat(), 1e-12, 'vertices against Python'); close(anny.normals, c.normals.flat(), 1e-12, 'normals against Python'); }
// the same move in the page's frame
const toPage = (v) => { const out = []; for (let i = 0; i < v.length; i += 3) out.push(...P.pointTo(PAGE, [v[i], v[i + 1], v[i + 2]])); return out; };
const paged = P.applyPlace(c.place, toPage(X), X.map((v, i) => v), P.pointTo(PAGE, head), PAGE);
close(paged.vertices, toPage(anny.vertices), 1e-12, 'the page frame');
console.log(`place: ${c.vertices ? 'Python\'s numbers' : 'its own case'}, Anny's frame and the page's agree within 1e-12`);

// ------------------------------------------------------------------ matrices, Held by, moves in a pose, the panel's numbers
const M = P.placeMatrix(c.place, head), back = P.placeOf(M, head, 'wrist.R', 12);
close(back.rotation, c.place.rotation, 1e-9, 'rotation round trip'); close(back.translation, c.place.translation, 1e-12, 'translation round trip');
for (let k = 0; k < 20; k++) {
  const sOld = rigid(11 + k), sNew = rigid(101 + k), hOld = [0.1 * k, 0.2, 0.9], hNew = [0, -0.1 * k, 1.1];
  const p0 = { bone: 'wrist.R', rotation: turn([1, k, 2], 10 * k), translation: [0.001 * k, 0, 0.01] };
  const before = P.mul4(sOld, P.pageMove(p0, hOld, PAGE)), p1 = P.rehold(p0, sOld, sNew, hOld, hNew, 'root', PAGE), after = P.mul4(sNew, P.pageMove(p1, hNew, PAGE));
  assert.equal(p1.bone, 'root');
  close(after, before, 2e-6, 'rehold keeps the posed piece');
  // a move W of the posed piece
  const W = T.turnAbout([0.1, 0.2, 0.3], [0, 1, 1], 0.3 + k * 0.1), p2 = P.movedInPose(p0, 'wrist.R', sOld, W, hOld, PAGE);
  close(P.mul4(sOld, P.pageMove(p2, hOld, PAGE)), P.mul4(W, before), 2e-6, 'a move in the pose');
  // the panel's numbers and back
  const n = T.pageNumbers(p0, PAGE), p3 = T.placeFromNumbers('wrist.R', n.mm, n.deg, PAGE);
  close(P.placeMatrix(p3, hOld), P.placeMatrix(p0, hOld), 1e-6, 'the panel\'s numbers');
}
assert.ok(T.isRest(null, 'head') && T.isRest({ bone: 'head', rotation: [0, 0, 0, 1], translation: [0, 0, 0] }, 'head'));
assert.ok(!T.isRest({ bone: 'root', rotation: [0, 0, 0, 1], translation: [0, 0, 0] }, 'head'));
assert.deepEqual(T.pageNumbers({ bone: 'b', rotation: [0, 0, 0, 1], translation: [0, 0, 0.01] }, PAGE).mm.map((x) => Math.round(x * 1e6) / 1e6), [0, 10, 0]);
assert.equal(P.placeProblem(c.place), null);
assert.ok(P.placeProblem({ bone: 'b', rotation: [0, 0, 0, 2] }));
assert.ok(P.placeProblem({ rotation: [0, 0, 0, 1] }));
console.log('place: Held by keeps the piece where it stands, moves in a pose and the panel\'s numbers hold in 20 random cases');

// ------------------------------------------------------------------ the card
const card = { name: 'synthetic', outfits: [{ id: 'statue', pieces: [{ piece: 'sword', file: 'sword.wearable.json' }, { piece: 'base', file: 'base.wearable.json' }] }],
  active_outfit: 'statue', wearables: [{ id: 'stone', file: 'stone.wearable.json' }] };
const look = { name: 'synthetic', phenotype: {}, face: {}, skin: { tone: 0.3, undertone: 0 }, hair: { style: 'bald', color: '#000000' }, eyes: { color: '#000000' } };
assert.equal(cardOf(look, card).schema, CARD_SCHEMA);
assert.ok(P.setCardPlace(card, 'statue/sword', c.place));
assert.ok(P.setCardPlace(card, 'stone', { bone: 'root', rotation: [0, 0, 0, 1], translation: [0, 0, 0.1] }));
assert.ok(!P.setCardPlace(card, 'statue/missing', c.place));
const saved = JSON.parse(JSON.stringify(cardOf(look, card)));
assert.equal(saved.schema, 6);
assert.deepEqual(P.cardPlace(saved, 'statue/sword'), c.place);
assert.deepEqual(P.cardPlace(saved, 'stone').bone, 'root');
assert.equal(P.cardPlace(saved, 'statue/base'), null);
const worn = wornPieces(saved, () => ({ slot: 'prop', hides: [] }));
assert.deepEqual(worn.find((w) => w.ref === 'statue/sword').place, c.place);
assert.equal(worn.find((w) => w.ref === 'statue/base').place, null);
P.setCardPlace(saved, 'statue/sword', null); P.setCardPlace(saved, 'stone', null);
assert.equal(cardOf(look, saved).schema, CARD_SCHEMA);
console.log('card: a place written and read, schema 6 with a place and 5 without, carried to the worn pieces');

// ------------------------------------------------------------------ a placed rigid piece dressed on a body
{
  // a body of three bones: root, head and wrist.R, each resting along the axes; a box of a prop bound rigidly to head
  const bones = ['root', 'head', 'wrist.R'], heads = Float64Array.from([0, 0, 0, 0, 0, 1.6, -0.3, 0.1, 1.0]);
  const V = Float64Array.from([0, 0, 0, 0, 0, 1.6, -0.3, 0.1, 1.0]), weights = Float64Array.from([1, 0, 1, 0, 1, 0]), indices = Int32Array.from([0, 0, 1, 0, 2, 0]);
  const pose = (b) => ({ R: Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]), t: heads.slice(b * 3, b * 3 + 3) });
  const body = new BodySurface({ vertices: V, triangles: Int32Array.from([0, 1, 2]), weights, indices, K: 2, bones, frame: IDENT, pose, heads });
  const local = Float64Array.from([0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1]), normal = Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const wearable = { name: 'box', label: 'Box', slot: 'prop', layer: 0, hides: [], binding: { method: 'rigid', bone: 'head', local, normal, bodyVertices: V, body: 'anny-template' },
    garment: { count: 3, positions: local, normals: normal, triangles: Int32Array.from([0, 1, 2]), groups: [], triangleGroup: new Int32Array(1), triangleMaterial: new Int32Array(1), materials: [] } };
  const place = { bone: 'wrist.R', rotation: turn([0, 0, 1], 90), translation: [0, 0, 0.05] };
  const pieces = dress(body, [{ ref: 'box', piece: 'box', outfit: null, wearable, shown: true, hiddenGroups: [], hideableGroups: [], place }], { push: false });
  const rest = [0.1, 0, 1.6, 0, 0.1, 1.6, 0, 0, 1.7], want = P.applyPlace(place, rest, normal, [-0.3, 0.1, 1.0], IDENT);
  close(pieces[0].vertices, want.vertices, 1e-12, 'the dressed piece');
  close(pieces[0].base.vertices, rest, 1e-12, 'its rest before the place');
  for (let i = 0; i < 3; i++) { assert.equal(pieces[0].joints[i * 4], 2); assert.equal(pieces[0].weights[i * 4], 1); }
  assert.throws(() => placedPiece({ ref: 'box', piece: 'box', outfit: null, wearable: { ...wearable, binding: { ...wearable.binding, method: 'surface' } }, shown: true,
    hiddenGroups: [], hideableGroups: [], place }, body, Float64Array.from(rest), normal), /rigidly bound/);
}
console.log('dress: a placed rigid piece moves about its new bone\'s head and follows that bone alone');

// ------------------------------------------------------------------ handles, Held by and keys
const hit = T.hitHandles({ x: 150, y: 100 }, { centre: { x: 100, y: 100 }, arrows: [[[100, 100], [170, 100]], [[100, 100], [100, 30]], [[100, 100], [60, 140]]] }, 9);
assert.deepEqual(hit, { kind: 'axis', k: 0 });
assert.deepEqual(T.hitHandles({ x: 102, y: 101 }, { centre: { x: 100, y: 100 }, arrows: [[[100, 100], [170, 100]]] }, 9), { kind: 'free' });
const ring = []; for (let i = 0; i <= 72; i++) ring.push([100 + 50 * Math.cos(i / 72 * 2 * Math.PI), 100 + 50 * Math.sin(i / 72 * 2 * Math.PI)]);
assert.deepEqual(T.hitHandles({ x: 151, y: 100 }, { centre: { x: 100, y: 100 }, rings: [ring], outer: 90 }, 9), { kind: 'ring', k: 0 });
assert.deepEqual(T.hitHandles({ x: 100, y: 191 }, { centre: { x: 100, y: 100 }, rings: [ring], outer: 90 }, 9), { kind: 'view' });
assert.ok(Math.abs(T.alongAxis(30, 0, 300, 0) - 0.1) < 1e-12);
assert.ok(Math.abs(T.sweep(0, 0, 1, 0, 0, 1) - Math.PI / 2) < 1e-12);
const choices = T.heldByChoices('metacarpal4.R', ['root', 'pelvis', 'head', 'wrist.L', 'wrist.R', 'metacarpal4.R', 'metacarpal4.L'], (b) => b);
assert.deepEqual(choices.map((h) => h.bone), ['metacarpal4.R', 'metacarpal4.L', 'wrist.R', 'wrist.L', 'head', 'pelvis', 'root']);
assert.ok(choices[0].label.endsWith('(as made)'));
const actions = await import('../src/studio/actions.ts');
const { installPoseEditor } = await import('../src/pose_edit/editor.ts');
installPoseEditor({ THREE, ready: () => false, canvas: {}, camera: new THREE.PerspectiveCamera() });
const want = { 'pose.prop.grab': ['G'], 'pose.prop.turn': ['R'], 'pose.prop.clear-move': ['Alt+G'], 'pose.prop.clear-turn': ['Alt+R'], 'pose.prop.nudge': ['ArrowLeft'],
  'pose.prop.push': ['PageDown'], 'pose.prop.pull': ['PageUp'], 'pose.prop.clear': ['Escape'] };
for (const [id, keys] of Object.entries(want)) {
  const a = actions.action(id);
  assert.ok(a, `${id} is registered`);
  assert.equal(a.scope, 'pose-prop');
  for (const k of keys) assert.ok(a.keys.includes(actions.normChord(k)), `${id} has ${k}`);
}
assert.ok(actions.action('pose.prop.reset'));
assert.deepEqual(actions.conflicts(), []);
assert.deepEqual(actions.reservedUses().filter((u) => u.id.startsWith('pose.')), []);
console.log('props: the handles\' hit tests, Held by\'s choices and the prop keys without a conflict');

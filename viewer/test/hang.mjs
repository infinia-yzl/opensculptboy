// OpenSculptBoy
// Apache License, Version 2.0
//
// A cloak that hangs (viewer/src/wearables/dress.ts, hangWeights), run by test/test_viewer_wearables.py:
//   node viewer/test/hang.mjs
// A vertex on the upper arm keeps its weight on the arm; a vertex of the cape behind the back takes it onto the
// shoulder, and a vertex of the hem beside the leg takes the leg's onto the pelvis.
import assert from 'node:assert/strict';
import { hangs, hangWeights, HANG_FAR } from '../src/wearables/dress.ts';

const bones = ['root', 'clavicle.L', 'shoulder01.L', 'upperarm01.L', 'lowerarm01.L', 'wrist.L', 'pelvis.L', 'upperleg01.L', 'lowerleg01.L', 'foot.L'];
const heads = [
  0, 1.0, 0, 0.02, 1.4, 0, 0.12, 1.4, 0, 0.16, 1.4, 0, 0.42, 1.4, 0, 0.65, 1.4, 0,
  0.05, 0.95, 0, 0.1, 0.9, 0, 0.1, 0.5, 0, 0.1, 0.08, 0,
];
const body = { heads, bones, unit: 1 };
const k = 4;
const V = new Float64Array([0.3, 1.42, 0.0, 0.2, 1.2, -0.25, 0.1, 0.6, -0.2]);
const weights = new Float32Array(3 * k), joints = new Uint16Array(3 * k);
weights.set([1, 0, 0, 0], 0); joints.set([3, 0, 0, 0], 0);
weights.set([0.6, 0.4, 0, 0], 4); joints.set([3, 0, 0, 0], 4);
weights.set([0.7, 0.3, 0, 0], 8); joints.set([7, 0, 0, 0], 8);
const r = hangWeights(V, weights, joints, body);
assert.equal(r.joints[0], 3);
assert.equal(r.weights[0], 1);
const far = new Map(); for (let c = 0; c < k; c++) far.set(r.joints[4 + c], (far.get(r.joints[4 + c]) || 0) + r.weights[4 + c]);
assert.ok(Math.abs((far.get(2) || 0) - 0.6) < 1e-6, 'the cape behind the back takes the arm weight onto the shoulder');
assert.ok(!far.get(3), 'and keeps none on the upper arm');
const hem = new Map(); for (let c = 0; c < k; c++) hem.set(r.joints[8 + c], (hem.get(r.joints[8 + c]) || 0) + r.weights[8 + c]);
assert.ok((hem.get(6) || 0) > 0.69, 'the hem beside the leg hangs from the pelvis');
assert.ok(HANG_FAR > 0.1);
assert.equal(hangs({ slot: 'cloak', springs: 1 }), true);
assert.equal(hangs({ slot: 'cloak', springs: 0 }), false);
assert.equal(hangs({ slot: 'torso', springs: 2 }), false);
console.log('ok');

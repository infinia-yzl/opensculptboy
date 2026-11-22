// OpenSculptBoy
// Apache License, Version 2.0
//
// Node checks of the stages of the Points tab, run by test/test_pose_edit_node.py:
//   node viewer/test/pose_edit_stages.mjs
// - the four stages split the 57 reference points without a gap or an overlap: one handle (the figure) in stage 1, the
//   head and the ends of the limbs in stage 2, the other body points in stage 3 and the finger points in stage 4;
// - the counts of a stage, its "need a look" among them, take its own points alone;
// - a saved session opens at the stage where its work is left, and an empty one at stage 1 (stage 2 when the card's
//   pose moves the root already);
// - the view shows a stage's points and the placed points of the earlier stages;
// - the figure's move of stage 1 (EditSession.setRoot): every point moves with the shift and turns about the pelvis,
//   the history keeps it, and no place of a point changes.
import assert from 'node:assert/strict';

const { POINTS } = await import('../src/pose_edit/points.ts');
const { FIGURE, MAIN_POINTS, STAGES, shownPoints, stageCounts, stageOf, stagePoints, startStage } = await import('../src/pose_edit/stages.ts');
const { EditSession } = await import('../src/pose_edit/session.ts');

// ------------------------------------------------------------------ the split
assert.equal(STAGES.length, 4);
assert.deepEqual(stagePoints(1), []);
assert.equal(stageOf(FIGURE), 1);
assert.deepEqual(stagePoints(2), ['head_top', 'wrist.R', 'wrist.L', 'ankle.R', 'ankle.L']);
assert.deepEqual(MAIN_POINTS, stagePoints(2));
const three = stagePoints(3), four = stagePoints(4);
assert.equal(three.length, 12);
for (const n of ['shoulder.R', 'shoulder.L', 'elbow.R', 'elbow.L', 'hip.R', 'hip.L', 'knee.R', 'knee.L', 'heel.R', 'toe.L']) assert.ok(three.includes(n), n);
assert.equal(four.length, 40);
assert.ok(four.every((n) => !n.startsWith('wrist')));
const all = [...stagePoints(2), ...three, ...four];
assert.equal(new Set(all).size, all.length, 'no point in two stages');
assert.deepEqual(new Set(all), new Set(POINTS.map((p) => p.name)), 'every point in a stage');
for (const s of STAGES) assert.ok(s.line.endsWith('.') && !/ not /.test(s.line), s.line);

// ------------------------------------------------------------------ counts and the opening stage
const states = (m) => (n) => m[n] || 'unplaced';
// an old session placed by the automatic placement: every point, some fingers and an elbow needing a look
const old = Object.fromEntries(POINTS.map((p) => [p.name, 'placed']));
old['index_pip.R'] = 'look'; old['ring_dip.L'] = 'look'; old['elbow.L'] = 'look';
assert.deepEqual(stageCounts(2, states(old)), { placed: 5, look: 0, off: 0, total: 5 });
assert.deepEqual(stageCounts(3, states(old)), { placed: 11, look: 1, off: 0, total: 12 });
assert.deepEqual(stageCounts(4, states(old)), { placed: 38, look: 2, off: 0, total: 40 });
assert.deepEqual(stageCounts(1, states({}), false), { placed: 0, look: 0, off: 0, total: 1 });
assert.deepEqual(stageCounts(1, states({}), true), { placed: 1, look: 0, off: 0, total: 1 });
assert.equal(startStage(states(old)), 3, 'the elbow that needs a look');
delete old['elbow.L'];
old['elbow.L'] = 'placed';
assert.equal(startStage(states(old)), 4, 'the fingers that need a look');
old['index_pip.R'] = 'placed'; old['ring_dip.L'] = 'off';
assert.equal(startStage(states(old)), 4, 'everything checked: the highest stage with a placed point');
assert.equal(startStage(states({})), 1);
assert.equal(startStage(states({}), true), 2);
assert.equal(startStage(states({ 'wrist.R': 'placed', head_top: 'placed' })), 2);
assert.equal(startStage(states(Object.fromEntries(MAIN_POINTS.map((n) => [n, 'placed'])))), 3);
assert.equal(startStage(states({ ...Object.fromEntries([...MAIN_POINTS, ...three].map((n) => [n, 'placed'])) })), 3);

// the view: the stage's points and the earlier stages' placed ones, never a later stage's
const placed = new Set(['wrist.R', 'elbow.R', 'index_tip.R']);
assert.deepEqual(new Set(shownPoints(2, (n) => placed.has(n))), new Set(MAIN_POINTS));
const shown3 = shownPoints(3, (n) => placed.has(n));
assert.ok(shown3.includes('wrist.R') && !shown3.includes('wrist.L') && shown3.includes('knee.L') && !shown3.includes('index_tip.R'));
assert.equal(shownPoints(4, () => false).length, 40);

// ------------------------------------------------------------------ the figure's move (stage 1)
// a rig of two bones: the root at the pelvis and one bone above it that carries every vertex; the rows' rest places
// spread round the root
const rig = { labels: ['root', 'spine01'], parents: [-1, 0], heads: [[0.1, 0.2, 0.9], [0.1, 0.2, 1.0]], frame: null };
const skin = { rest: (row) => [0.1 + 0.001 * (row % 50), 0.2 + 0.002 * (row % 7), 0.9 + 0.0005 * (row % 900)], skin: () => ({ bones: [1], weights: [1] }) };
const I = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const S = new EditSession(rig, skin, [I.slice(), I.slice()]);
assert.deepEqual(S.pelvis(), [0, 0, 0], 'the pelvis is the card frame origin');
const before = S.positions();
S.targets.set('wrist.R', { position: [0.3, 0.1, 0.2], on: true, weight: 1, confidence: 1, placed: 'owner', depth_mm: 0 });
S.record('Move the figure');
S.setRoot(null, [0.05, -0.02, 0.01]);
assert.deepEqual(S.pelvis(), [0.05, -0.02, 0.01]);
for (const [n, x] of S.positions()) {
  const y = before.get(n);
  assert.ok(Math.abs(x[0] - y[0] - 0.05) < 1e-12 && Math.abs(x[1] - y[1] + 0.02) < 1e-12 && Math.abs(x[2] - y[2] - 0.01) < 1e-12, n);
}
// a quarter turn about the vertical through the pelvis: every point keeps its height and its distance from the pelvis
const Z = [0, -1, 0, 1, 0, 0, 0, 0, 1];
S.setRoot(Z, null);
const p = S.pelvis(), moved = S.positions();
for (const [n, x] of moved) {
  const y = before.get(n), dy = [y[0], y[1]], dx = [x[0] - p[0], x[1] - p[1]];
  assert.ok(Math.abs(x[2] - p[2] - y[2]) < 1e-12, n);
  assert.ok(Math.abs(Math.hypot(...dx) - Math.hypot(...dy)) < 1e-12, n);
  assert.ok(Math.abs(dx[0] + dy[1]) < 1e-12 && Math.abs(dx[1] - dy[0]) < 1e-12, n);
}
// the places of the points stay; undo gives the figure back
assert.deepEqual(S.targets.get('wrist.R').position, [0.3, 0.1, 0.2]);
assert.equal(S.undo(), 'Move the figure');
assert.deepEqual(S.pelvis(), [0, 0, 0]);
assert.deepEqual(S.rotations[0], I);
for (const [n, x] of S.positions()) assert.ok(x.every((v, k) => Math.abs(v - before.get(n)[k]) < 1e-12), n);
assert.equal(S.redo(), 'Move the figure');
assert.deepEqual(S.pelvis(), [0.05, -0.02, 0.01]);
// the card's points never name the figure's handle
assert.ok(!JSON.stringify(S.cardPoints()).includes(FIGURE));

console.log(`stages: ${stagePoints(2).length} + ${three.length} + ${four.length} points, the opening stage and the figure's move hold`);

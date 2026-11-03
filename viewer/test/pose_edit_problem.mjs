// OpenSculptBoy
// Apache License, Version 2.0
//
// Parity of the page's problem builder (src/pose_edit/problem.ts) with opensculptboy.posing.point_solve.build_problem,
// run by test/test_pose_edit_problem.py:
//   node viewer/test/pose_edit_problem.mjs <folder>
// reads <folder>/input.json (a rig in the space of the turns, a pose, the skin rows of the points' triangles, the
// points, the pins and the cases to build) and writes <folder>/output.json: each case's problem, the chains' bones,
// the joint coordinates of the pose and the points' influences, as the page builds them.
import fs from 'fs';
import path from 'path';
import { buildProblem, chainBones, coordinatesOf, influenceOf, jointFrames } from '../src/pose_edit/problem.ts';
import { solveProblem } from '../src/pose_edit/solve.ts';

const dir = process.argv[2];
const inp = JSON.parse(fs.readFileSync(path.join(dir, 'input.json')));
const rig = inp.rig;
const skin = {
  rest: (row) => inp.skin.rest[row],
  skin: (row) => ({ bones: inp.skin.indices[row], weights: inp.skin.weights[row] }),
  bind: (b) => inp.skin.bind[b],
};
const influence = (p) => influenceOf(p.vertices, p.barycentric, skin);
const points = inp.points.map((p) => ({ influence: influence(p), target: p.target, weight: p.weight, on: p.on, finger: p.finger }));
const pins = inp.pins.map((q) => ({ ...q, influence: influence(q) }));
const cases = inp.cases.map((c) => {
  const problem = buildProblem(rig, inp.rotations, points, c.chains, { limits: c.limits, pins: c.pins ? pins : [] });
  const out = { problem };
  if (c.solve) {
    const r = solveProblem(JSON.parse(JSON.stringify(problem)));
    out.solve = { rotations: Array.from(r.rotations), points: r.points, iterations: r.iterations, cost: r.cost };
  }
  return out;
});
const frames = jointFrames(rig);
const coordinates = {};
for (const [b, f] of frames) coordinates[rig.labels[b]] = coordinatesOf(f, inp.rotations[b]);
const chains = Object.fromEntries(inp.chains.map((c) => [c, chainBones(rig, [c])]));
fs.writeFileSync(path.join(dir, 'output.json'), JSON.stringify({ cases, coordinates, chains, influences: points.map((p) => p.influence) }));

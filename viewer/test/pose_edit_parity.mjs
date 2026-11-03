// OpenSculptBoy
// Apache License, Version 2.0
//
// Parity of the page's reference-point solve (src/pose_edit/solve.ts) with opensculptboy.posing.point_solve, run by
// test/test_reference_points.py:
//   node viewer/test/pose_edit_parity.mjs <folder>
// reads <folder>/problems.json (a list of problems) and writes <folder>/output.json: each problem's rotations, points,
// steps and cost; with <folder>/drag.json, it also times a drag (a point's place moved a little each frame, each frame
// solved from the last one's pose) and writes the times per frame.
import fs from 'fs';
import path from 'path';
import { PointSolver, solveProblem } from '../src/pose_edit/solve.ts';

const dir = process.argv[2];
const problems = JSON.parse(fs.readFileSync(path.join(dir, 'problems.json')));
const results = problems.map((p) => {
  const r = solveProblem(p);
  return { rotations: Array.from(r.rotations), points: r.points, iterations: r.iterations, cost: r.cost };
});
const out = { results };
const dragPath = path.join(dir, 'drag.json');
if (fs.existsSync(dragPath)) {
  const drag = JSON.parse(fs.readFileSync(dragPath));
  const solver = new PointSolver(drag.problem);
  solver.solve();
  // warm up the engine, then time each frame
  const frames = drag.frames;
  const times = [];
  let last = null;
  for (let pass = 0; pass < 2; pass++) {
    const fresh = new PointSolver(JSON.parse(JSON.stringify(drag.problem)));
    fresh.solve();
    for (const target of frames) {
      fresh.setTarget(drag.point, target);
      const t0 = performance.now();
      last = fresh.solve(drag.iterations);
      const t1 = performance.now();
      if (pass === 1) times.push(t1 - t0);
    }
  }
  out.drag = { times, points: last.points, rotations: Array.from(last.rotations) };
}
fs.writeFileSync(path.join(dir, 'output.json'), JSON.stringify(out));

// OpenSculptBoy
// Apache License, Version 2.0
//
// Parity of the page's placement of reference points on a scan (src/pose_edit/place.ts, with its BVH) with
// opensculptboy.posing.point_place.place_points, run by test/test_pose_edit_place.py:
//   node viewer/test/pose_edit_place.mjs <folder>
// reads <folder>/input.json (cases of points with their skin normals, the posed body with each vertex's part, and a
// scan) and writes <folder>/output.json: each case's placed points, and the time of the BVH's build and placement.
import fs from 'fs';
import path from 'path';
import { TriangleBVH } from '../src/pose_edit/bvh.ts';
import { placePoints } from '../src/pose_edit/place.ts';

const dir = process.argv[2];
const inp = JSON.parse(fs.readFileSync(path.join(dir, 'input.json')));
const out = inp.cases.map((c) => {
  const t0 = performance.now();
  const bvh = new TriangleBVH(Float64Array.from(c.scan.vertices.flat()), Uint32Array.from(c.scan.triangles.flat()));
  const t1 = performance.now();
  const placed = placePoints(c.points, { vertices: Float64Array.from(c.body.vertices.flat()), parts: c.body.parts }, bvh);
  const t2 = performance.now();
  return { placed, build_ms: t1 - t0, place_ms: t2 - t1 };
});
fs.writeFileSync(path.join(dir, 'output.json'), JSON.stringify(out));

// OpenSculptBoy
// Apache License, Version 2.0
//
// Parity of the page's bone turns (src/anny_shape.ts: restOrientations, referenceTurns) and pose conversion
// (src/pose_reference.ts) with opensculptboy.posing.frames, run by test/test_pose_reference.py:
//   node viewer/test/pose_reference.mjs <folder>
// reads <folder>/tables.json (the orientation tables of anny.viewer.export, and the face and local-change rows) and
// <folder>/cases.json (per case: the weights of the phenotype, face-shape and local-change rows, and the card's
// local-ref turns D, bones x 9), and writes <folder>/output.json: per case, the turns C and the page's rotations R
// (bones x 9 each), and the turns D again from R.
import fs from 'fs';
import path from 'path';
import { referenceTurns, restOrientations } from '../src/anny_shape.ts';
import { pageToRef, refToPage } from '../src/pose_reference.ts';

const dir = process.argv[2];
const T = JSON.parse(fs.readFileSync(path.join(dir, 'tables.json')));
const cases = JSON.parse(fs.readFileSync(path.join(dir, 'cases.json')));
const rows = (r) => (r ? { index: Uint32Array.from(r.index.flat()), values: Float32Array.from(r.values.flat()) } : undefined);
const data = { template: Float32Array.from(T.template.flat()), reference: Float32Array.from(T.reference.flat()), blend: rows(T.blend), rootIdentity: T.root_identity };
const face = rows(T.face), local = rows(T.local);
const mats = (flat) => Array.from({ length: flat.length / 9 }, (_, b) => flat.slice(b * 9, b * 9 + 9));
const out = cases.map((c) => {
  const C = referenceTurns(data, restOrientations(data, c.phenotype, [[face, c.face], [local, c.local]]));
  const R = refToPage(mats(c.D), C, T.parents);
  const D = pageToRef(R, C, T.parents);
  return { C: Array.from(C), R: R.flat(), D: D.flat() };
});
fs.writeFileSync(path.join(dir, 'output.json'), JSON.stringify(out));

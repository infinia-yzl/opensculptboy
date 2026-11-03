// OpenSculptBoy
// Apache License, Version 2.0
//
// Parity of the page's Squeeze and balance mark (src/pose_edit/contact.ts) with opensculptboy.posing.contact, run by
// test/test_pose_edit_contact.py:
//   node viewer/test/pose_edit_contact.mjs <folder>
// reads <folder>/input.json (a rig, its skin and cases) and writes <folder>/output.json.
import fs from 'fs';
import path from 'path';
import { SkinnedBody, balance, squeeze } from '../src/pose_edit/contact.ts';

const dir = process.argv[2];
const inp = JSON.parse(fs.readFileSync(path.join(dir, 'input.json')));
const k = inp.k, bind = inp.bind;
const body = new SkinnedBody(inp.rig, Float64Array.from(inp.rest.flat()), Int32Array.from(inp.indices.flat()), Float64Array.from(inp.weights.flat()), k,
  Uint32Array.from(inp.faces.flat()), (b) => bind[b]);
const out = inp.cases.map((c) => {
  const t0 = performance.now();
  if (c.kind === 'squeeze') {
    const r = squeeze(body, c.rotations, c.hand, c.objects, c.value, undefined, c.fingers);
    return { rotations: r.rotations, report: r.report, ms: performance.now() - t0 };
  }
  return { balance: balance(body, c.rotations), ms: performance.now() - t0 };
});
fs.writeFileSync(path.join(dir, 'output.json'), JSON.stringify(out));

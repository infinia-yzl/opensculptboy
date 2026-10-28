// OpenSculptBoy
// Apache License, Version 2.0
//
// Parity of the page's view of a character card (viewer/src/card/data.ts, the level offsets of
// viewer/src/subdivision.ts, the local changes of viewer/src/anny_shape.ts) with opensculptboy and anny
// (run by test/test_viewer_card.py):
//   node viewer/test/card_parity.mjs <folder>
// reads <folder>/input.json and binary inputs, writes binary outputs next to them.
import fs from 'fs';
import path from 'path';
import { addRows, localRowWeights } from '../src/anny_shape.ts';
import { eyeAxes, outerRadius, polarToUv } from '../src/card/data.ts';
import { FineSubdivision } from '../src/subdivision.ts';

const dir = process.argv[2];
const inp = JSON.parse(fs.readFileSync(path.join(dir, 'input.json')));
const bytes = (name) => { const b = fs.readFileSync(path.join(dir, name)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const f32 = (name) => new Float32Array(bytes(name));
const i16 = (name) => new Int16Array(bytes(name));
const u32 = (name) => new Uint32Array(bytes(name));
const write = (name, arr) => fs.writeFileSync(path.join(dir, name), Buffer.from(arr.buffer));

// the eye texture's layout, the turned axes of an eye and its outer surface
const e = inp.eyes;
write('eye_uv.bin', new Float64Array(e.polar.flatMap(([s, phi, left]) => polarToUv(s, phi, left))));
write('eye_axes.bin', new Float64Array(e.gaze.flatMap((g) => eyeAxes(g))));
write('eye_radius.bin', new Float64Array(e.theta.map((t) => outerRadius(t, e.iris, e.cornea, e.bulge))));

// a level-1 offset added after the page's first level, then the page's second level
const s = inp.subdivision;
const sub = new FineSubdivision(s.n, u32('quads.bin'), 1, u32('rows.bin'));
write('fine.bin', sub.apply(f32('coarse.bin'), 3, [f32('level1.bin')]).slice());

// MakeHuman's local changes on coarse vertices and bone heads
const l = inp.local;
const local = { tables: l.tables, ids: u32('local_ids.bin'), offsets: i16('local_offsets.bin'), boneDeltas: f32('local_bones.bin') };
const V = new Float32Array(l.values.length * l.vertices * 3), J = new Float32Array(l.values.length * l.bones * 3);
l.values.forEach((values, i) => {
  const v = new Float32Array(l.vertices * 3), j = new Float32Array(l.bones * 3);
  addRows(local, localRowWeights(local, values), v, j);
  V.set(v, i * l.vertices * 3); J.set(j, i * l.bones * 3);
});
write('local_coarse.bin', V);
write('local_joints.bin', J);

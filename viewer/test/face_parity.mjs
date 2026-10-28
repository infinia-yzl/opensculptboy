// OpenSculptBoy
// Apache License, Version 2.0
//
// The page's faces (src/face/rig.ts) on the inputs of test/test_viewer_expressions.py:
//   node viewer/test/face_parity.mjs <folder>
// The folder holds input.json (the face tables, the subdivision's sizes and the mixes) and raw buffers; the script writes
// per mix the coarse body (coarse_<i>.bin), the fine body (fine_<i>.bin) and its normals (normals_<i>.bin), float32.
import fs from 'fs';
import path from 'path';
import { FineSubdivision } from '../src/subdivision.ts';
import { vertexNormals } from '../src/body.ts';
import { FaceLayer, FaceRig } from '../src/face/rig.ts';

const dir = process.argv[2];
const inp = JSON.parse(fs.readFileSync(path.join(dir, 'input.json')));
const read = (name, T) => { const b = fs.readFileSync(path.join(dir, name)); return new T(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const coarse = read('coarse.bin', Float32Array), quads = read('quads.bin', Uint32Array), rows = read('rows.bin', Uint32Array);
const tris = read('triangles.bin', Uint32Array);
const n = inp.n_body;
const sub = new FineSubdivision(n, quads, inp.levels, rows);
const pos = sub.apply(coarse, 3).slice();
const nrm = vertexNormals(pos, tris), nrm0 = nrm.slice();
const body = { sub, nBody: n, nc: n, nFine: rows.length, nWeld: rows.length, coarse, quads, pos, nrm, ns: nrm.slice(), weld: tris, weldShown: null, hide: false,
  copyOf: new Uint32Array(0) };
const offsets4 = read('offsets.bin', Int16Array), m = offsets4.length / 4, offsets = new Int16Array(m * 3);
for (let i = 0; i < m; i++) for (let k = 0; k < 3; k++) offsets[i * 3 + k] = offsets4[i * 4 + k];
const data = { meta: inp.meta, ids: read('ids.bin', Uint32Array), offsets, jawIds: read('jaw_ids.bin', Uint32Array), jawWeights: read('jaw_weights.bin', Float32Array) };
const rig = new FaceRig(body, data);
const layer = new FaceLayer(rig, body);
const t0 = performance.now();
layer.rebase();
const out = { rebase_ms: performance.now() - t0, apply_ms: [], region: layer.region.length, pivot: rig.pivot };
// the sparse subdivision against the dense one on a random field of the face's coarse vertices
{
  const mask = rig.coarseMask, ids = [], vals = [];
  let seed = 7;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };
  for (let v = 0; v < n; v++) if (mask[v] && rnd() > 0.3) { ids.push(v); vals.push(rnd(), rnd(), rnd()); }
  const dense = new Float32Array(n * 3);
  ids.forEach((v, i) => { for (let k = 0; k < 3; k++) dense[v * 3 + k] = vals[i * 3 + k]; });
  const full = sub.apply(dense, 3).slice(), r = rig.sparse.apply(ids, vals, 3);
  let err = 0, outside = 0;
  const at = new Map(); r.ids.forEach((v, i) => at.set(v, i));
  for (let v = 0; v < rows.length; v++) {
    const i = at.get(v);
    for (let k = 0; k < 3; k++) { const s = i === undefined ? 0 : r.values[i * 3 + k]; err = Math.max(err, Math.abs(s - full[v * 3 + k])); }
    if (i === undefined && (full[v * 3] || full[v * 3 + 1] || full[v * 3 + 2])) outside++;
  }
  out.sparse_error = err; out.sparse_outside = outside;
}
inp.mixes.forEach((mix, i) => {
  const w = new Float32Array(rig.names.length);
  for (const [k, v] of Object.entries(mix)) w[rig.index.get(k)] = v;
  const V = coarse.slice();
  rig.coarseFace(w, V);
  fs.writeFileSync(path.join(dir, `coarse_${i}.bin`), Buffer.from(V.buffer));
  rig.prepare();
  const ta = performance.now();
  layer.apply(w);
  out.apply_ms.push(performance.now() - ta);
  fs.writeFileSync(path.join(dir, `fine_${i}.bin`), Buffer.from(body.pos.slice().buffer));
  fs.writeFileSync(path.join(dir, `normals_${i}.bin`), Buffer.from(body.nrm.slice().buffer));
});
// back to the rest: the layer leaves the body as it found it
layer.apply(new Float32Array(rig.names.length));
const rest = sub.apply(coarse, 3).slice();
let back = 0;
for (let i = 0; i < pos.length; i++) back = Math.max(back, Math.abs(body.pos[i] - rest[i]));
out.rest_error = back;
let nback = 0;
for (let i = 0; i < nrm0.length; i++) nback = Math.max(nback, Math.abs(body.nrm[i] - nrm0[i]));
out.rest_normal_error = nback;
fs.writeFileSync(path.join(dir, 'output.json'), JSON.stringify(out));

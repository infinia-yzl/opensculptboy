// OpenSculptBoy
// Apache License, Version 2.0
//
// The faces of the page on the data of anny.viewer (run by test/test_viewer_expressions.py, TestViewerFaceData):
//   node viewer/test/face_body.mjs <build folder> <output json>
// It builds AnnyBody and the face's rig and layer from the raw buffers of viewer/build, as the page does after decoding,
// plays the presets and some actions, and reports the times of each step, the vertices moved, whether every value is
// finite, and whether the rest comes back.
import fs from 'fs';
import path from 'path';
import { AnnyBody, vertexNormals } from '../src/body.ts';
import { decodeFaceData, FaceLayer, FaceRig } from '../src/face/rig.ts';

const [dir, out] = process.argv.slice(2);
const man = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json')));
const info = Object.fromEntries(man.buffers.map((b) => [b.name, b]));
const bytes = (name) => { const b = fs.readFileSync(path.join(dir, name + '.raw')); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const drop4th = (a) => { const n = a.length / 4, o = new a.constructor(n * 3); for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) o[i * 3 + k] = a[i * 4 + k]; return o; };
const hv = info.head_v, n = hv.count, rec = new DataView(bytes('head_v'));
const rest = new Float32Array(n * 3);
for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) rest[i * 3 + k] = hv.lo[k] + rec.getUint16(i * hv.stride + k * 2, true) / 65535 * (hv.hi[k] - hv.lo[k]);
const index = new Uint32Array(bytes('head_i'));
const seams = info.head_seam ? new Uint32Array(bytes('head_seam')).slice(0, info.head_seam.count * 2) : undefined;
const body = new AnnyBody({ ...man, detail_step: info.head_detail.step }, {
  template: new Float32Array(bytes('coarse_template')), components: drop4th(new Int16Array(bytes('shape_components'))),
  projection: new Float32Array(bytes('shape_projection')), jointTemplate: new Float32Array(bytes('joint_template')),
  jointBlend: new Float32Array(bytes('joint_blend')), quads: new Uint32Array(bytes('coarse_quads')), rows: new Uint32Array(bytes('head_row')),
  detail: drop4th(new Int16Array(bytes('head_detail'))), relief: new Int16Array(bytes('head_detail')).filter((_, i) => i % 4 === 3), index, rest,
  nsmooth: vertexNormals(rest, index), coarseSkin: new Uint8Array(bytes('coarse_skin')), seams,
});
const t = (f) => { const t0 = performance.now(); const r = f(); return [performance.now() - t0, r]; };
const result = { timing: {} };
result.timing.body = t(() => body.update({}))[0];
const data = decodeFaceData(man.expressions, new Uint32Array(bytes('face_action_ids')), new Int16Array(bytes('face_action_offsets')), new Uint32Array(bytes('face_jaw')));
const [tr, rig] = t(() => new FaceRig(body, data));
result.timing.rig = tr;
const layer = new FaceLayer(rig, body);
result.timing.rebase = layer.rebase();
result.timing.prepare = t(() => rig.prepare())[0];
result.region = layer.region.length;
result.triangles = layer.tris.length / 3;
result.pivot = rig.pivot;
const pos0 = body.pos.slice(), nrm0 = body.nrm.slice();
const weights = (mix) => { const w = new Float32Array(rig.names.length); for (const [k, v] of Object.entries(mix)) w[rig.index.get(k)] = v; return w; };
const faces = Object.entries(man.expressions.presets).concat([['jawOpen', { jawOpen: 1 }], ['jawHalf', { jawOpen: 0.5 }]]);
result.faces = {};
for (const [name, mix] of faces) {
  const w = weights(mix);
  const times = [];
  let range = null;
  for (let r = 0; r < 5; r++) { const [ms, rg] = t(() => layer.apply(w)); times.push(ms); range = rg; layer.apply(new Float32Array(rig.names.length)); }
  layer.apply(w);
  let moved = 0, finite = true;
  for (let i = 0; i < body.pos.length; i += 3) {
    if (!Number.isFinite(body.pos[i]) || !Number.isFinite(body.nrm[i])) finite = false;
    if (Math.hypot(body.pos[i] - pos0[i], body.pos[i + 1] - pos0[i + 1], body.pos[i + 2] - pos0[i + 2]) > 1e-6) moved++;
  }
  times.sort((a, b) => a - b);
  const fast = [];
  for (let r = 0; r < 5; r++) { fast.push(t(() => layer.apply(w, {}, false))[0]); layer.apply(new Float32Array(rig.names.length), {}, false); }
  fast.sort((a, b) => a - b);
  // the quicker normals of a fade against the exact ones
  layer.apply(w, {}, false);
  const nf = body.nrm.slice();
  layer.apply(w);
  let worst = 0;
  for (let i = 0; i < nf.length; i += 3) worst = Math.max(worst, Math.acos(Math.min(1, nf[i] * body.nrm[i] + nf[i + 1] * body.nrm[i + 1] + nf[i + 2] * body.nrm[i + 2])) * 180 / Math.PI);
  result.faces[name] = { ms: times[2], first: times[0], fast_ms: fast[2], fast_first: fast[0], fast_worst_degrees: worst, range: [range[0], range[1]], moved, finite };
}
layer.apply(new Float32Array(rig.names.length));
let back = 0, nback = 0;
for (let i = 0; i < pos0.length; i++) { back = Math.max(back, Math.abs(body.pos[i] - pos0[i])); nback = Math.max(nback, Math.abs(body.nrm[i] - nrm0[i])); }
result.rest_error = back; result.rest_normal_error = nback;
// a face on another body: the jaw's pivot and corrective follow it
body.update({ age: 0.1, weight: 0.9, gender: 0.2 });
layer.ready = false;
const [ms2] = t(() => layer.apply(weights({ jawOpen: 1 })));
result.other_body = { ms: ms2, pivot: rig.pivot, finite: body.pos.every(Number.isFinite) && body.nrm.every(Number.isFinite) };
fs.writeFileSync(out, JSON.stringify(result, null, 1));

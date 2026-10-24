// OpenSculptBoy
// Apache License, Version 2.0
//
// Parity of the page's dressing (viewer/src/wearables/) with opensculptboy.wearables (run by
// test/test_viewer_wearables.py):
//   node viewer/test/wearables_parity.mjs <folder>
// reads <folder>/input.json, the bodies and the packages, and writes the dressed pieces of every case next to them.
import fs from 'fs';
import path from 'path';
import { BodySurface, dress, estimatedPose, IDENTITY_FRAME, outfitCover } from '../src/wearables/dress.ts';
import { readWearable } from '../src/wearables/package.ts';
import { skinTriangles } from '../src/wearables/skin.ts';
import { wornPieces } from '../src/wearables/wardrobe.ts';

const dir = process.argv[2];
const inp = JSON.parse(fs.readFileSync(path.join(dir, 'input.json')));
const bytes = (file) => { const b = fs.readFileSync(file); return new Uint8Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
const f64 = (name) => new Float64Array(bytes(path.join(dir, name)).buffer);
const i32 = (name) => new Int32Array(bytes(path.join(dir, name)).buffer);
const write = (name, arr) => fs.writeFileSync(path.join(dir, name), Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength));

const triangles = skinTriangles(new Uint32Array(bytes(path.join(dir, 'quads.u32')).buffer));
write('skin_triangles.i32', triangles);
const weights = f64('weights.f64'), indices = i32('indices.i32');

// the packages of a folder, read once
const packages = new Map();
async function load(folder, file) {
  const p = path.resolve(folder, file);
  if (!packages.has(p)) {
    const json = JSON.parse(fs.readFileSync(p, 'utf8'));
    packages.set(p, await readWearable(json, async (rel) => bytes(path.resolve(path.dirname(p), rel))));
  }
  return packages.get(p);
}
const rot = (A, R) => { const o = new Float64Array(9); for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[r * 3 + c] = A[r * 3] * R[c] + A[r * 3 + 1] * R[3 + c] + A[r * 3 + 2] * R[6 + c]; return o; };

const report = { cases: [] };
for (const c of inp.cases) {
  const body = inp.bodies[c.body], frame = c.frame ? inp.frame : IDENTITY_FRAME;
  const P = f64(body.poses), A = frame.rotation, s = frame.scale, o = frame.offset;
  const toFrame = (x, y, z) => [0, 1, 2].map((r) => s * (A[r * 3] * x + A[r * 3 + 1] * y + A[r * 3 + 2] * z) + o[r]);
  const anny = f64(body.vertices), V = new Float64Array(anny.length);
  for (let i = 0; i < anny.length; i += 3) V.set(toFrame(anny[i], anny[i + 1], anny[i + 2]), i);
  const exact = (bone) => {
    const m = P.subarray(bone * 16, bone * 16 + 16), R = Float64Array.from([m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]]);
    return { R: rot(A, R), t: Float64Array.from(toFrame(m[3], m[7], m[11])) };
  };
  const bodyArgs = { vertices: V, triangles, weights, indices, K: inp.K, bones: inp.bone_labels, frame, pose: exact };
  const surface = new BodySurface(bodyArgs);
  const card = JSON.parse(fs.readFileSync(path.join(inp.folder, c.card), 'utf8'));
  const info = {};
  for (const e of [...(card.outfits || []).flatMap((x) => (x.pieces || []).map((p) => ({ ref: `${x.id}/${p.piece}`, file: p.file }))),
    ...(card.wearables || []).map((w) => ({ ref: w.id, file: w.file }))]) info[e.ref] = await load(inp.folder, e.file);
  const worn = wornPieces(card, (ref) => info[ref], c.choices || {}).map((p) => ({ ...p, wearable: info[p.ref] }));
  if (c.estimate) surface.pose = (bone, b, wearable) => estimatedPose(wearable, b, exact(bone).t);
  const t0 = performance.now(), timing = {};
  const dressed = dress(surface, worn, { push: c.push !== false, timing });
  const ms = performance.now() - t0, t1 = performance.now();
  const cover = outfitCover(surface, dressed);
  const coverMs = performance.now() - t1;
  const pieces = dressed.map((p, k) => {
    const name = `${c.name}_${k}`;
    write(`${name}_vertices.f64`, p.vertices); write(`${name}_normals.f64`, p.normals);
    write(`${name}_weights.f32`, p.weights); write(`${name}_joints.u16`, p.joints);
    return { ref: p.ref, shown: p.shown, hidden_groups: p.hiddenGroups, pushed: p.pushed, push_mm: p.pushMm, file: name };
  });
  write(`${c.name}_cover.u8`, cover);
  report.cases.push({ name: c.name, pieces, ms, cover_ms: coverMs, timing });
}
fs.writeFileSync(path.join(dir, 'output.json'), JSON.stringify(report, null, 1));

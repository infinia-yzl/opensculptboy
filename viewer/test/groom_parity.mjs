// OpenSculptBoy
// Apache License, Version 2.0
//
// Parity of the page's grooms and blinks (viewer/src/card/data.ts) with opensculptboy.groom and opensculptboy.eyes.page
// (run by test/test_viewer_card.py):
//   node viewer/test/groom_parity.mjs <folder>
// reads <folder>/input.json (a card, the paths of a groom file and a blinks file, closures to weigh) and writes
// <folder>/output.json.
import fs from 'fs';
import path from 'path';
import { blinkCurve, blinksOf, blinkWeights, cardPlan, groomOf } from '../src/card/data.ts';
import { readSafetensors } from '../src/wearables/package.ts';

const dir = process.argv[2];
const inp = JSON.parse(fs.readFileSync(path.join(dir, 'input.json')));
const read = (name) => readSafetensors(new Uint8Array(fs.readFileSync(path.join(dir, name))));
const plan = cardPlan(inp.card);
const g = read(inp.groom), groom = groomOf(g.tensors, g.metadata);
const b = read(inp.blinks), blinks = blinksOf(b.tensors, b.metadata);
const out = {
  plan: { hair: plan.hair, brows: plan.brows, blinks: plan.blinks, shown: plan.shown },
  groom: { kind: groom.kind, points: Array.from(groom.points.slice(0, 9)), uvs: Array.from(groom.uvs.slice(0, 6)), triangles: groom.triangles.length / 3,
    material: Array.from(groom.material), colors: groom.colors ? Array.from(groom.colors.slice(0, 8)) : null, materials: groom.materials },
  blinks: { names: blinks.names, level: blinks.level, rows: Array.from(blinks.rows), offsets: Array.from(blinks.offsets.slice(0, 12)),
    lashes: blinks.lashes ? blinks.lashes.length : null },
  weights: inp.closures.map(([l, r, sl, sr]) => Array.from(blinkWeights(blinks.names, { L: l, R: r }, { L: sl, R: sr }))),
  curve: inp.curve.map((t) => blinkCurve(t)),
};
fs.writeFileSync(path.join(dir, 'output.json'), JSON.stringify(out));

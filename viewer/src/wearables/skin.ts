// OpenSculptBoy
// Apache License, Version 2.0
//
// The skin triangles that garments bind to (opensculptboy.wearables.fit.skin_triangles), rebuilt from the page's body
// quads with the codes of skin_split.ts.

import { QUAD_COUNT, SPLIT_CODES } from './skin_split.ts';

const SPLITS = [[[0, 1, 2], [0, 2, 3]], [[0, 1, 3], [1, 2, 3]]];

export function skinTriangles(quads: ArrayLike<number>): Int32Array {
  const nq = quads.length / 4;
  if (nq !== QUAD_COUNT) throw new Error(`The page has ${nq} body quads; the skin codes are for ${QUAD_COUNT}.`);
  const bytes = Uint8Array.from(atob(SPLIT_CODES), (c) => c.charCodeAt(0)), out: number[] = [];
  for (let q = 0; q < nq; q++) {
    const code = (bytes[q >> 1] >> ((q & 1) * 4)) & 15, halves = SPLITS[code & 1];
    if (code & 2) for (const i of halves[0]) out.push(quads[q * 4 + i]);
    if (code & 4) for (const i of halves[1]) out.push(quads[q * 4 + i]);
  }
  return Int32Array.from(out);
}

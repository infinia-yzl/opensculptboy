// OpenSculptBoy
// Apache License, Version 2.0
//
// The floor under the figure: the page stands the figure on the lowest point of what it shows, the body and the props
// (wearables of the slot `prop`: a statue's base, a tree stump, a head at its feet), so that a prop under the soles
// stands on the floor and the body stands on the prop. main.ts skins the body's coarse vertices (body.ts
// lowestSkinned) and the shown props' vertices (this file) by the pose's bone matrices, and lifts the root by the
// difference between the floor and the lowest of them.

// the slots whose pieces the floor takes into account
export const GROUNDED_SLOTS = new Set(['prop']);

// a dressed piece as the wardrobe holds it: its rest vertices (x, y, z each), and four joints and weights each
export interface SkinnedPoints { vertices: ArrayLike<number>; joints: ArrayLike<number>; weights: ArrayLike<number> }

// the lowest height of points skinned by bone matrices (column-major 4x4 per bone, rest heads removed, as
// main.ts poseMatrices writes them); Infinity for no points
export function lowestSkinnedY(p: SkinnedPoints, M: ArrayLike<number>): number {
  const V = p.vertices, J = p.joints, W = p.weights, n = Math.floor(V.length / 3);
  let low = Infinity;
  for (let i = 0; i < n; i++) {
    const x = V[i * 3], y = V[i * 3 + 1], z = V[i * 3 + 2];
    let py = 0, sum = 0;
    for (let k = 0; k < 4; k++) {
      const w = W[i * 4 + k];
      if (!w) continue;
      const m = J[i * 4 + k] * 16;
      py += w * (M[m + 1] * x + M[m + 5] * y + M[m + 9] * z + M[m + 13]);
      sum += w;
    }
    if (sum > 0 && py / sum < low) low = py / sum;
  }
  return low;
}

// the lowest height of the shown pieces of the grounded slots
export function lowestPieces(pieces: (SkinnedPoints & { slot: string })[], M: ArrayLike<number>): number {
  let low = Infinity;
  for (const p of pieces) if (GROUNDED_SLOTS.has(p.slot)) low = Math.min(low, lowestSkinnedY(p, M));
  return low;
}

// the root's lift that stands the lowest point on the floor, and how far the props lift the body above its own ground
export function groundShift(floor: number, bodyLow: number, piecesLow: number): { ground: number; lift: number } {
  const low = Math.min(bodyLow, piecesLow);
  return { ground: floor - low, lift: bodyLow - low };
}

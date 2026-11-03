// OpenSculptBoy
// Apache License, Version 2.0
//
// A character card's pose and the page's pose. The card holds Anny's local-ref turns D (opensculptboy.Character.pose):
// each bone's turn relative to its parent, measured from Anny's reference pose, in which every bone's frame takes its
// reference orientation (model.reference_bone_orientations), so the skeleton stands a little apart from its rest
// pose (the heads up to about 14 mm on the default body). The page poses from the rest pose instead: each bone turns
// about its rest head along the world axes (the pose library's rotations R). With each bone's turn
// C = reference rest^T (anny_shape.ts, referenceTurns, for the body shown), the two convert as
// anny.poses.pose_parameters converts the library:
//
//   D_b = C_parent R_b C_b^T        R_b = C_parent^T D_b C_b        (C_parent = identity for the root)
//
// All matrices here are in Anny's axes, 3 x 3 row-major; quaternions are (x, y, z, w). A card written by Python may
// carry its body's turns as pose.reference (opensculptboy.posing.frames), which the page compares with its own.

import { I3, mul, quaternion, rotation, transpose, type M3 } from './pose_from_image.ts';

// how far (in matrix entries, about 0.01 degrees) a turn may leave the zero and still count as the zero
export const ZERO = 2e-4;

// bone b's turn C_b from a table (bones * 9)
export function turnOf(C: ArrayLike<number>, b: number): M3 {
  return Array.from({ length: 9 }, (_, k) => C[b * 9 + k]);
}

// the page's rotations R (Anny's axes) of local-ref turns D, by bone index
export function refToPage(D: M3[], C: ArrayLike<number>, parents: number[]): M3[] {
  return D.map((Db, b) => {
    const p = parents[b];
    const Cp = p >= 0 ? turnOf(C, p) : I3;
    return mul(mul(transpose(Cp), Db), turnOf(C, b));
  });
}

// local-ref turns D of the page's rotations R (Anny's axes), by bone index
export function pageToRef(R: M3[], C: ArrayLike<number>, parents: number[]): M3[] {
  return R.map((Rb, b) => {
    const p = parents[b];
    const Cp = p >= 0 ? turnOf(C, p) : I3;
    return mul(mul(Cp, Rb), transpose(turnOf(C, b)));
  });
}

// the page's rotations of Anny's reference pose (the zero of local-ref): the reference heads are the rest rig posed by
// them, which is where Python's PoseRig turns the bones
export function referencePose(C: ArrayLike<number>, parents: number[]): M3[] {
  return refToPage(parents.map(() => I3.slice()), C, parents);
}

// a card's pose.bones (quaternions by bone name) as local-ref turns by bone index; missing bones keep the zero
export function cardTurns(bones: Record<string, number[]> | null | undefined, names: string[]): M3[] {
  return names.map((name) => {
    const v = bones?.[name];
    if (!Array.isArray(v) || v.length !== 4 || !v.every(Number.isFinite)) return I3.slice();
    const n = Math.hypot(v[0], v[1], v[2], v[3]);
    return n ? rotation(v.map((x) => x / n)) : I3.slice();
  });
}

// the "bones" of a card's pose for the page's rotations q (bones x 4, quaternions in a frame that M turns Anny's frame
// into): the local-ref turns that leave the zero by more than ZERO (the page keeps its rotations in 16 bits, so a bone at
// the zero comes back a few 1e-5 off it), rounded as Python writes them
export function cardBonesRef(q: ArrayLike<number>, names: string[], parents: number[], C: ArrayLike<number> | null, M: M3 = I3): Record<string, number[]> {
  const Mt = transpose(M);
  const R = names.map((_, i) => mul(mul(Mt, rotation(q, i * 4)), M));
  const D = C ? pageToRef(R, C, parents) : R;
  const out: Record<string, number[]> = {};
  names.forEach((name, i) => {
    if (D[i].every((v, k) => Math.abs(v - I3[k]) <= ZERO)) return;
    out[name] = quaternion(D[i]).map((v) => Math.round(v * 1e7) / 1e7);
  });
  return out;
}

// the largest angle (degrees) between a card's turns (pose.reference: quaternions by bone name) and the page's
export function referenceDifference(reference: Record<string, number[]> | null | undefined, C: ArrayLike<number>, names: string[]): number {
  let worst = 0;
  names.forEach((name, b) => {
    const v = reference?.[name];
    if (!Array.isArray(v) || v.length !== 4) return;
    const A = rotation(v.map((x) => x / Math.hypot(v[0], v[1], v[2], v[3]))), B = turnOf(C, b);
    const T = mul(transpose(A), B), c = Math.min(1, Math.max(-1, (T[0] + T[4] + T[8] - 1) / 2));
    worst = Math.max(worst, Math.acos(c) * 180 / Math.PI);
  });
  return worst;
}

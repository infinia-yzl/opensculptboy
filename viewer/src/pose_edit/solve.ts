// OpenSculptBoy
// Apache License, Version 2.0
//
// The page's solve of Anny's pose to reference points: the twin of opensculptboy.posing.point_solve (its _Solver and
// solve_problem). It reads the same problem (format opensculptboy.point-solve/1, built in Python by build_problem or in the
// page from the same parts) and repeats each step's arithmetic, so that both give the same pose; test_reference_points
// checks them against each other.
//
// The unknowns are each turning bone's joint coordinates c (bend, side, twist; radians), with D_b = exp(A_b c_b), where
// A_b holds the joint frame's axes times their side's signs. A bone turns about its head in the reference pose, after its
// parent: K_b = K_parent [D_b, h_b - D_b h_b]. A point is a sum of influences R_b u + s t_b. Damped least squares
// (Levenberg-Marquardt with Nielsen's damping) turns the bones so that the points meet their places, within the ranges,
// the ranges held as boxes, near the start, with fingers whose points are all off curled as their neighbours. Matrices are 3x3, row-major.

export interface SolveBone {
  bone: number; axes: number[][]; sign: number[]; dofs: boolean[]; lo: number[]; hi: number[]; start: number[];
}
export interface SolveGroup {
  name: string; axis: number[]; members: number[]; shares: number[]; read: number; read_share: number; lo: number; hi: number;
}
export interface SolvePoint { bones: number[]; s: number[]; u: number[][]; target: number[]; weight: number; finger?: string | null }
export interface SolvePin {
  bones: number[]; s: number[]; u: number[][]; kind: 'point' | 'plane'; weight: number;
  target?: number[]; point?: number[]; normal?: number[];
}
export interface SolveCurl { bone: number; neighbours: number[]; natural: number[] }
export interface SolveProblem {
  format: string; parents: number[]; heads: number[][]; order: number[]; scale: number; frame: number[][];
  rotations: number[][]; bones: SolveBone[]; groups: SolveGroup[]; columns: ({ bone: number; axis: number } | { group: number })[];
  points: SolvePoint[]; pins: SolvePin[]; curls: SolveCurl[]; limits: boolean;
  weights: { prior: number; natural: number }; iterations: number;
}

// ---------------------------------------------------------------- rotations
function expInto(v0: number, v1: number, v2: number, out: Float64Array, o: number) {
  const angle = Math.sqrt(v0 * v0 + v1 * v1 + v2 * v2);
  let a: number, b: number;
  if (angle < 1e-9) { a = 1; b = 0; } else { a = Math.sin(angle) / angle; b = (1 - Math.cos(angle)) / (angle * angle); }
  // I + a K + b K^2, K = [v]x
  const xx = v0 * v0, yy = v1 * v1, zz = v2 * v2, xy = v0 * v1, xz = v0 * v2, yz = v1 * v2;
  out[o] = 1 + b * (-yy - zz); out[o + 1] = -a * v2 + b * xy; out[o + 2] = a * v1 + b * xz;
  out[o + 3] = a * v2 + b * xy; out[o + 4] = 1 + b * (-xx - zz); out[o + 5] = -a * v0 + b * yz;
  out[o + 6] = -a * v1 + b * xz; out[o + 7] = a * v0 + b * yz; out[o + 8] = 1 + b * (-xx - yy);
}

// SO(3)'s left Jacobian at v times the vector w: exp(v + d) = exp(Jl(v) d) exp(v)
function jlTimes(v0: number, v1: number, v2: number, w0: number, w1: number, w2: number, out: number[]) {
  const angle = Math.sqrt(v0 * v0 + v1 * v1 + v2 * v2);
  let a: number, b: number;
  if (angle < 1e-6) { a = 0.5; b = 1 / 6; } else {
    a = (1 - Math.cos(angle)) / (angle * angle); b = (angle - Math.sin(angle)) / (angle * angle * angle);
  }
  // K w and K (K w)
  const k0 = v1 * w2 - v2 * w1, k1 = v2 * w0 - v0 * w2, k2 = v0 * w1 - v1 * w0;
  const kk0 = v1 * k2 - v2 * k1, kk1 = v2 * k0 - v0 * k2, kk2 = v0 * k1 - v1 * k0;
  out[0] = w0 + a * k0 + b * kk0; out[1] = w1 + a * k1 + b * kk1; out[2] = w2 + a * k2 + b * kk2;
}

// the rotation vector of a rotation matrix (opensculptboy.posing.point_solve._log)
export function logRotation(R: ArrayLike<number>, o = 0): [number, number, number] {
  const tr = R[o] + R[o + 4] + R[o + 8];
  const c = Math.min(1, Math.max(-1, (tr - 1) / 2));
  const angle = Math.acos(c);
  const w0 = R[o + 7] - R[o + 5], w1 = R[o + 2] - R[o + 6], w2 = R[o + 3] - R[o + 1];
  if (angle < 1e-6) return [0.5 * w0, 0.5 * w1, 0.5 * w2];
  if (angle > Math.PI - 1e-6) {
    const M = [R[o] - c, (R[o + 1] + R[o + 3]) / 2, (R[o + 2] + R[o + 6]) / 2, (R[o + 1] + R[o + 3]) / 2, R[o + 4] - c,
      (R[o + 5] + R[o + 7]) / 2, (R[o + 2] + R[o + 6]) / 2, (R[o + 5] + R[o + 7]) / 2, R[o + 8] - c];
    let k = 0;
    if (M[4] > M[k * 4]) k = 1;
    if (M[8] > M[k * 4]) k = 2;
    const n = Math.sqrt(Math.max(M[k * 4], 1e-15));
    let ax = [M[k] / n, M[3 + k] / n, M[6 + k] / n];
    if (ax[0] * w0 + ax[1] * w1 + ax[2] * w2 < 0) ax = ax.map((x) => -x);
    return [angle * ax[0], angle * ax[1], angle * ax[2]];
  }
  const f = angle / (2 * Math.sin(angle));
  return [f * w0, f * w1, f * w2];
}

// x = A^-1 y for a symmetric positive definite A (n x n, row-major), by Cholesky's factors
function choleskySolve(A: Float64Array, b: Float64Array, n: number, L: Float64Array, out: Float64Array) {
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) L[i * n + i] = Math.sqrt(Math.max(s, 1e-300));
      else L[i * n + j] = s / L[j * n + j];
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k];
    y[i] = s / L[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * out[k];
    out[i] = s / L[i * n + i];
  }
}

interface Member { bone: number; k: number; share: number }

export interface SolveOutcome { rotations: Float64Array; points: number[][]; iterations: number; cost: number }

export class PointSolver {
  readonly P: SolveProblem;
  readonly B: number;
  readonly n: number;
  private parents: Int32Array;
  private heads: Float64Array;
  private order: Int32Array;
  private fixed: Float64Array;
  private A: Map<number, Float64Array> = new Map();
  private bones: Map<number, SolveBone> = new Map();
  private members: Member[][];
  private ancestor: Uint8Array;
  // the coordinates of each turning bone (3 per bone), and the bones in the problem's order
  private coordBones: number[];
  private C: Map<number, Float64Array> = new Map();
  // scratch
  private D: Float64Array;
  private KR: Float64Array;
  private Kt: Float64Array;
  private J: Float64Array;
  private H: Float64Array;
  private g: Float64Array;
  private Lm: Float64Array;
  private axesW: Float64Array; // per column member: its world axis (3)
  private memberOffset: number[];
  private coordRow: Map<string, Float64Array> = new Map();
  private frameA: number[];
  private frameAinv: number[];
  private frameT: number[];

  constructor(problem: SolveProblem) {
    this.P = problem;
    const B = (this.B = problem.parents.length);
    this.parents = Int32Array.from(problem.parents);
    this.heads = Float64Array.from(problem.heads.flat());
    this.order = Int32Array.from(problem.order);
    this.fixed = Float64Array.from(problem.rotations.flat());
    this.coordBones = [];
    for (const e of problem.bones) {
      this.bones.set(e.bone, e);
      const A = new Float64Array(9);
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) A[i * 3 + j] = e.axes[i][j] * e.sign[j];
      this.A.set(e.bone, A);
      this.C.set(e.bone, Float64Array.from(e.start));
      this.coordBones.push(e.bone);
    }
    this.members = problem.columns.map((c) => {
      if ('group' in c) {
        const g = problem.groups[c.group];
        return g.members.map((m, i) => ({ bone: m, k: 2, share: g.shares[i] }));
      }
      return [{ bone: c.bone, k: c.axis, share: 1 }];
    });
    this.n = this.members.length;
    this.ancestor = new Uint8Array(B * B);
    for (let b = 0; b < B; b++) for (let p = b; p >= 0; p = this.parents[p]) this.ancestor[p * B + b] = 1;
    this.memberOffset = [];
    let m = 0;
    for (const ms of this.members) { this.memberOffset.push(m); m += ms.length; }
    this.axesW = new Float64Array(m * 3);
    const n = this.n;
    this.D = new Float64Array(B * 9);
    this.KR = new Float64Array(B * 9);
    this.Kt = new Float64Array(B * 3);
    this.J = new Float64Array(B * 3);
    this.H = new Float64Array(n * n);
    this.g = new Float64Array(n);
    this.Lm = new Float64Array(n * n);
    // d c_bk / d column, for the rows linear in the coordinates
    for (const b of this.coordBones) for (let k = 0; k < 3; k++) {
      const row = new Float64Array(n);
      this.members.forEach((ms, ci) => { for (const mm of ms) if (mm.bone === b && mm.k === k) row[ci] += mm.share; });
      this.coordRow.set(b * 3 + k + '', row);
    }
    const F = problem.frame;
    this.frameA = [F[0][0], F[0][1], F[0][2], F[1][0], F[1][1], F[1][2], F[2][0], F[2][1], F[2][2]];
    this.frameT = [F[0][3], F[1][3], F[2][3]];
    this.frameAinv = invert3(this.frameA);
  }

  // a point's place in the output frame (the card's placement)
  setTarget(index: number, xyz: number[]) {
    const d = [xyz[0] - this.frameT[0], xyz[1] - this.frameT[1], xyz[2] - this.frameT[2]];
    const M = this.frameAinv;
    this.P.points[index].target = [M[0] * d[0] + M[1] * d[1] + M[2] * d[2], M[3] * d[0] + M[4] * d[1] + M[5] * d[2],
      M[6] * d[0] + M[7] * d[1] + M[8] * d[2]];
  }

  setWeight(index: number, weight: number) { this.P.points[index].weight = weight; }

  private rotations(C: Map<number, Float64Array>, out: Float64Array) {
    out.set(this.fixed);
    for (const b of this.coordBones) {
      const A = this.A.get(b)!, c = C.get(b)!;
      expInto(A[0] * c[0] + A[1] * c[1] + A[2] * c[2], A[3] * c[0] + A[4] * c[1] + A[5] * c[2],
        A[6] * c[0] + A[7] * c[1] + A[8] * c[2], out, b * 9);
    }
  }

  private transforms(D: Float64Array) {
    const { KR, Kt, heads, parents, J } = this;
    for (let i = 0; i < this.order.length; i++) {
      const b = this.order[i], p = parents[b];
      const o = b * 9, h0 = heads[b * 3], h1 = heads[b * 3 + 1], h2 = heads[b * 3 + 2];
      if (p < 0) {
        KR.fill(0, o, o + 9); KR[o] = KR[o + 4] = KR[o + 8] = 1; Kt[b * 3] = Kt[b * 3 + 1] = Kt[b * 3 + 2] = 0;
      } else {
        const q = p * 9;
        for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
          KR[o + r * 3 + c] = KR[q + r * 3] * D[o + c] + KR[q + r * 3 + 1] * D[o + 3 + c] + KR[q + r * 3 + 2] * D[o + 6 + c];
        }
        // h - D h, then the parent's turn and shift
        const l0 = h0 - (D[o] * h0 + D[o + 1] * h1 + D[o + 2] * h2);
        const l1 = h1 - (D[o + 3] * h0 + D[o + 4] * h1 + D[o + 5] * h2);
        const l2 = h2 - (D[o + 6] * h0 + D[o + 7] * h1 + D[o + 8] * h2);
        for (let r = 0; r < 3; r++) Kt[b * 3 + r] = KR[q + r * 3] * l0 + KR[q + r * 3 + 1] * l1 + KR[q + r * 3 + 2] * l2 + Kt[p * 3 + r];
      }
      for (let r = 0; r < 3; r++) J[b * 3 + r] = KR[o + r * 3] * h0 + KR[o + r * 3 + 1] * h1 + KR[o + r * 3 + 2] * h2 + Kt[b * 3 + r];
    }
  }

  // each column member's world axis: d D_b / d c_bk turns the bone's subtree about its head by Rp Jl(A c) A_k
  private computeAxes(C: Map<number, Float64Array>) {
    const tmp = [0, 0, 0];
    this.members.forEach((ms, ci) => {
      let m = this.memberOffset[ci];
      for (const mm of ms) {
        const A = this.A.get(mm.bone)!, c = C.get(mm.bone)!;
        const v0 = A[0] * c[0] + A[1] * c[1] + A[2] * c[2], v1 = A[3] * c[0] + A[4] * c[1] + A[5] * c[2];
        const v2 = A[6] * c[0] + A[7] * c[1] + A[8] * c[2];
        jlTimes(v0, v1, v2, A[mm.k], A[3 + mm.k], A[6 + mm.k], tmp);
        const q = this.parents[mm.bone] * 9, KR = this.KR;
        for (let r = 0; r < 3; r++) {
          this.axesW[m * 3 + r] = mm.share * (KR[q + r * 3] * tmp[0] + KR[q + r * 3 + 1] * tmp[1] + KR[q + r * 3 + 2] * tmp[2]);
        }
        m++;
      }
    });
  }

  // the cost, and with `build` the normal equations H = J^T J and g = J^T r
  private evaluate(C: Map<number, Float64Array>, build: boolean): number {
    this.rotations(C, this.D);
    this.transforms(this.D);
    const n = this.n, S = this.P.scale, H = this.H, g = this.g, KR = this.KR, Kt = this.Kt, J = this.J, B = this.B;
    if (build) { this.computeAxes(C); H.fill(0); g.fill(0); }
    let cost = 0;
    const row = new Float64Array(n);
    const addRow = (r: number, jac: Float64Array | null, w: number) => {
      cost += 0.5 * r * r;
      if (!build || jac === null) return;
      for (let i = 0; i < n; i++) {
        const ji = w * jac[i];
        if (ji === 0) continue;
        g[i] += r * ji;
        for (let j = 0; j <= i; j++) H[i * n + j] += ji * w * jac[j];
      }
    };
    // a point's position and its rows' derivatives (3 x n)
    const Jx = new Float64Array(3 * n);
    const pointRows = (e: SolvePoint | SolvePin, weight: number, kind: 'point' | 'plane', target: number[], normal?: number[]) => {
      const nInf = e.bones.length;
      const ys = new Float64Array(nInf * 3);
      let x0 = 0, x1 = 0, x2 = 0;
      for (let k = 0; k < nInf; k++) {
        const b = e.bones[k], o = b * 9, u = e.u[k], s = e.s[k];
        const y0 = KR[o] * u[0] + KR[o + 1] * u[1] + KR[o + 2] * u[2] + s * Kt[b * 3];
        const y1 = KR[o + 3] * u[0] + KR[o + 4] * u[1] + KR[o + 5] * u[2] + s * Kt[b * 3 + 1];
        const y2 = KR[o + 6] * u[0] + KR[o + 7] * u[1] + KR[o + 8] * u[2] + s * Kt[b * 3 + 2];
        ys[k * 3] = y0; ys[k * 3 + 1] = y1; ys[k * 3 + 2] = y2;
        x0 += y0; x1 += y1; x2 += y2;
      }
      if (build) {
        Jx.fill(0);
        this.members.forEach((ms, ci) => {
          let m = this.memberOffset[ci];
          for (const mm of ms) {
            const b = mm.bone;
            let Y0 = 0, Y1 = 0, Y2 = 0, Ss = 0, any = false;
            for (let k = 0; k < nInf; k++) {
              if (this.ancestor[b * B + e.bones[k]]) { Y0 += ys[k * 3]; Y1 += ys[k * 3 + 1]; Y2 += ys[k * 3 + 2]; Ss += e.s[k]; any = true; }
            }
            if (any) {
              const d0 = Y0 - Ss * J[b * 3], d1 = Y1 - Ss * J[b * 3 + 1], d2 = Y2 - Ss * J[b * 3 + 2];
              const a0 = this.axesW[m * 3], a1 = this.axesW[m * 3 + 1], a2 = this.axesW[m * 3 + 2];
              Jx[ci] += a1 * d2 - a2 * d1; Jx[n + ci] += a2 * d0 - a0 * d2; Jx[2 * n + ci] += a0 * d1 - a1 * d0;
            }
            m++;
          }
        });
      }
      const w = weight * S;
      if (kind === 'point') {
        const x = [x0, x1, x2];
        for (let r = 0; r < 3; r++) addRow(w * (x[r] - target[r]), build ? Jx.subarray(r * n, (r + 1) * n) : null, w);
      } else {
        const nr = normal!;
        const d = nr[0] * (x0 - target[0]) + nr[1] * (x1 - target[1]) + nr[2] * (x2 - target[2]);
        if (build) for (let i = 0; i < n; i++) row[i] = nr[0] * Jx[i] + nr[1] * Jx[n + i] + nr[2] * Jx[2 * n + i];
        addRow(w * d, build ? row : null, w);
      }
    };
    for (const e of this.P.points) if (e.weight > 0) pointRows(e, e.weight, 'point', e.target);
    for (const e of this.P.pins) {
      if (e.kind === 'point') pointRows(e, e.weight, 'point', e.target!);
      else pointRows(e, e.weight, 'plane', e.point!, e.normal!);
    }
    const wp = this.P.weights.prior;
    for (const b of this.coordBones) {
      const e = this.bones.get(b)!, c = C.get(b)!;
      for (let k = 0; k < 3; k++) {
        const jr = this.coordRow.get(b * 3 + k + '')!;
        let used = false;
        for (let i = 0; i < n; i++) if (jr[i] !== 0) { used = true; break; }
        if (!used) continue;
        addRow(wp * (c[k] - e.start[k]), jr, wp);
      }
    }
    const wn = this.P.weights.natural;
    for (const cd of this.P.curls) {
      if (!this.bones.has(cd.bone)) continue;
      const nb = cd.neighbours.filter((x) => this.bones.has(x));
      let target: number;
      if (nb.length) { target = 0; for (const x of nb) target += C.get(x)![0]; target /= nb.length; } else target = cd.natural[0];
      addRow(wn * (C.get(cd.bone)![0] - target), this.coordRow.get(cd.bone * 3 + '')!, wn);
    }
    if (build) for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) H[j * n + i] = H[i * n + j];
    return cost;
  }

  // the coordinates after a step, held within the ranges (a coordinate outside its range may stay or move back)
  private step(C: Map<number, Float64Array>, delta: Float64Array): Map<number, Float64Array> {
    const out = new Map<number, Float64Array>();
    for (const [b, c] of C) out.set(b, Float64Array.from(c));
    this.members.forEach((ms, ci) => { for (const mm of ms) out.get(mm.bone)![mm.k] += mm.share * delta[ci]; });
    if (!this.P.limits) return out;
    for (const b of this.coordBones) {
      const e = this.bones.get(b)!, was = C.get(b)!, c = out.get(b)!;
      for (let k = 0; k < 3; k++) {
        if (!e.dofs[k]) continue;
        const lo = Math.min(e.lo[k], was[k]), hi = Math.max(e.hi[k], was[k]);
        c[k] = Math.min(Math.max(c[k], lo), hi);
      }
    }
    for (const gr of this.P.groups) {
      const v = out.get(gr.read)![2] / gr.read_share, was = C.get(gr.read)![2] / gr.read_share;
      const clamped = Math.min(Math.max(v, Math.min(gr.lo, was)), Math.max(gr.hi, was));
      if (clamped !== v) gr.members.forEach((m, i) => { out.get(m)![2] += gr.shares[i] * (clamped - v); });
    }
    return out;
  }

  // run the solve from the present coordinates (a warm start after the first)
  solve(iterations = this.P.iterations): SolveOutcome {
    const n = this.n;
    let C = this.C;
    let taken = 0;
    let cost = this.evaluate(C, n > 0);
    if (n > 0) {
      let mu = -1, nu = 2;
      const A = new Float64Array(n * n), delta = new Float64Array(n), neg = new Float64Array(n);
      for (let it = 0; it < iterations; it++) {
        const H = this.H, g = this.g;
        if (mu < 0) { let mx = 0; for (let i = 0; i < n; i++) mx = Math.max(mx, H[i * n + i]); mu = 1e-3 * Math.max(mx, 1e-12); }
        A.set(H);
        for (let i = 0; i < n; i++) { A[i * n + i] += mu; neg[i] = -g[i]; }
        choleskySolve(A, neg, n, this.Lm, delta);
        let predicted = 0;
        for (let i = 0; i < n; i++) predicted += delta[i] * (mu * delta[i] - g[i]);
        predicted *= 0.5;
        const trial = this.step(C, delta);
        const costNew = this.evaluate(trial, false);
        taken = it + 1;
        if (costNew < cost) {
          const rho = (cost - costNew) / Math.max(predicted, 1e-300);
          mu *= Math.max(1 / 3, 1 - Math.pow(2 * rho - 1, 3));
          nu = 2;
          const improved = cost - costNew;
          C = trial; cost = costNew;
          this.evaluate(C, true);
          let mx = 0;
          for (let i = 0; i < n; i++) mx = Math.max(mx, Math.abs(delta[i]));
          if (improved <= 1e-12 * Math.max(cost, 1e-30) || mx < 1e-9) break;
        } else {
          // the normal equations of C stay in H and g
          mu *= nu; nu *= 2;
          if (mu > 1e10) break;
        }
      }
    }
    this.C = C;
    this.rotations(C, this.D);
    this.transforms(this.D);
    return { rotations: Float64Array.from(this.D), points: this.positions(), iterations: taken, cost };
  }

  // the points (output frame) on the present pose
  positions(): number[][] {
    const { KR, Kt } = this, A = this.frameA, T = this.frameT;
    return this.P.points.map((e) => {
      let x0 = 0, x1 = 0, x2 = 0;
      e.bones.forEach((b, k) => {
        const o = b * 9, u = e.u[k], s = e.s[k];
        x0 += KR[o] * u[0] + KR[o + 1] * u[1] + KR[o + 2] * u[2] + s * Kt[b * 3];
        x1 += KR[o + 3] * u[0] + KR[o + 4] * u[1] + KR[o + 5] * u[2] + s * Kt[b * 3 + 1];
        x2 += KR[o + 6] * u[0] + KR[o + 7] * u[1] + KR[o + 8] * u[2] + s * Kt[b * 3 + 2];
      });
      return [A[0] * x0 + A[1] * x1 + A[2] * x2 + T[0], A[3] * x0 + A[4] * x1 + A[5] * x2 + T[1],
        A[6] * x0 + A[7] * x1 + A[8] * x2 + T[2]];
    });
  }

  // the joint coordinates (radians) of the turning bones, by bone index
  coordinates(): Map<number, number[]> {
    const out = new Map<number, number[]>();
    for (const [b, c] of this.C) out.set(b, Array.from(c));
    return out;
  }
}

function invert3(m: number[]): number[] {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [A / det, -(b * i - c * h) / det, (b * f - c * e) / det, B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
    C / det, -(a * h - b * g) / det, (a * e - b * d) / det];
}

// a whole problem, as solve_problem runs it in Python
export function solveProblem(problem: SolveProblem): SolveOutcome {
  return new PointSolver(problem).solve();
}

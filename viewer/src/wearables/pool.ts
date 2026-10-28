// OpenSculptBoy
// Apache License, Version 2.0
//
// The closest-point queries of the settled dress on worker threads: a large query (the vertices of a dense piece
// against the body or a lower piece) splits into one contiguous run of points per worker, each worker answers its
// run with Surface.closest (query_worker.ts), and the answers join in order. Each point's answer depends on that
// point alone, so the result is the same as on the page's own thread. The page sends each surface once to each worker.
// While the wardrobe runs its settled dress (view.ts), Surface.closestSteps yields the pool's promise and the
// wardrobe resumes the steps with the answer.

import type { Closest, F64, Surface } from './geometry.ts';

export const POOL_MIN_POINTS = 4096;   // smaller queries stay on the page's thread

export class QueryPool {
  private workers: { w: Worker; has: Set<number> }[] = [];
  private jobs = new Map<number, (r: any) => void>();
  private job = 0;
  active = false;   // set by the wardrobe while it runs the settled dress
  failed = false;
  constructor(source: string, count: number) {
    const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
    for (let i = 0; i < count; i++) {
      const w = new Worker(url);
      w.onmessage = (e: MessageEvent) => { const f = this.jobs.get(e.data.job); if (f) { this.jobs.delete(e.data.job); f(e.data); } };
      w.onerror = (e: any) => { this.failed = true; console.warn('A query worker failed; the page answers the queries itself.', e.message); };
      this.workers.push({ w, has: new Set() });
    }
  }
  get size() { return this.workers.length; }
  usable(n: number) { return this.active && !this.failed && this.workers.length > 1 && n >= POOL_MIN_POINTS; }
  closest(surface: Surface, points: F64, candidates: number, limit: number): Promise<Closest> {
    const n = points.length / 3, W = this.workers.length, per = Math.ceil(n / W), id = (surface as any).id;
    const parts: Promise<any>[] = [];
    for (let k = 0; k < W; k++) {
      const i0 = k * per, i1 = Math.min(n, i0 + per);
      if (i0 >= i1) break;
      const wk = this.workers[k];
      if (!wk.has.has(id)) {
        wk.w.postMessage({ type: 'surface', id, V: surface.V, T: surface.T, N: surface.N });
        wk.has.add(id);
        if (wk.has.size > 6) wk.has.delete(wk.has.values().next().value!);
      }
      const job = ++this.job, pts = points.slice(i0 * 3, i1 * 3);
      parts.push(new Promise((res) => { this.jobs.set(job, res); wk.w.postMessage({ type: 'closest', job, id, points: pts, candidates, limit }, [pts.buffer]); }));
    }
    return Promise.all(parts).then((rs) => {
      const out: Closest = { triangle: new Int32Array(n), bary: new Float64Array(n * 3), point: new Float64Array(n * 3), distance: new Float64Array(n) };
      rs.forEach((r, k) => {
        const i0 = k * per;
        out.triangle.set(r.triangle, i0); out.bary.set(r.bary, i0 * 3); out.point.set(r.point, i0 * 3); out.distance.set(r.distance, i0);
      });
      return out;
    });
  }
}

let POOL: QueryPool | null = null;
export function setQueryPool(p: QueryPool | null) { POOL = p; }
export function queryPool(): QueryPool | null { return POOL; }

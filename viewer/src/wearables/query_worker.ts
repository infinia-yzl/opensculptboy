// OpenSculptBoy
// Apache License, Version 2.0
//
// A worker of the query pool (pool.ts): it keeps the last surfaces the page sent and answers closest-point queries
// on a run of points with Surface.closest.
//   { type: 'surface', id, V, T, N }
//   { type: 'closest', job, id, points, candidates, limit } -> { job, triangle, bary, point, distance }

import { Surface } from './geometry.ts';

const surfaces = new Map<number, Surface>();
self.onmessage = (e: MessageEvent) => {
  const m = e.data;
  if (m.type === 'surface') {
    surfaces.set(m.id, new Surface(m.V, m.T, m.N));
    while (surfaces.size > 6) surfaces.delete(surfaces.keys().next().value!);
  } else if (m.type === 'closest') {
    const r = surfaces.get(m.id)!.closest(m.points, m.candidates, m.limit);
    (self as any).postMessage({ job: m.job, ...r }, [r.triangle.buffer, r.bary.buffer, r.point.buffer, r.distance.buffer]);
  }
};

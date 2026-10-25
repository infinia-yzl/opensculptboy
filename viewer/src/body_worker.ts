// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The body update of a slider edit in a worker. The page keeps drawing frames while the worker builds the body
// (AnnyBody.update: the shape, the subdivision, the frames and the detail), and the page then copies the result in.
// The worker holds its own AnnyBody, built from copies of the same buffers, so it computes the same numbers as the
// page's own update. The output buffers travel back and forth by transfer, so an update allocates nothing.
//
// Messages to the worker:
//   { type: 'init', meta, bufs, detailStep }
//   { type: 'update', id, gen, t, values, face, out: { pos, nrm, ns, coarse, joints } }
// Messages from the worker:
//   { type: 'ready', ms } or { type: 'error', message }
//   { type: 'done', id, gen, t, ms, timing, eyes, values, face, out } (id, gen and t come back as they went)

import { AnnyBody } from './body.ts';

let body: AnnyBody | null = null;
const post = (m: any, transfer: Transferable[] = []) => (self as any).postMessage(m, transfer);

self.onmessage = (e: MessageEvent) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      const t0 = performance.now();
      body = new AnnyBody(m.meta, m.bufs);
      body.detailStep = m.detailStep;
      post({ type: 'ready', ms: performance.now() - t0 });
    } else if (m.type === 'update' && body) {
      const r = body.update(m.values, m.face);
      const o = m.out;
      o.pos.set(body.pos); o.nrm.set(body.nrm); o.ns.set(body.ns); o.coarse.set(body.coarse); o.joints.set(body.joints);
      post({ type: 'done', id: m.id, gen: m.gen, t: m.t, ms: r.ms, timing: body.timing, eyes: body.eyes, values: body.values, face: body.faceValues, out: o },
        [o.pos.buffer, o.nrm.buffer, o.ns.buffer, o.coarse.buffer, o.joints.buffer]);
    }
  } catch (err) {
    post({ type: 'error', message: String((err as any)?.stack || err) });
  }
};

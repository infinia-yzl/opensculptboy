// OpenSculptBoy
// Apache License, Version 2.0
//
// The geometry of a character sheet, without the DOM or three.js (node tests run it):
// - maskBox: the box of what a probe picture covers, in metres on the plane through the camera's target;
// - figureFrames: one camera frame per view or pose of a full figure, framed from the boxes, with one scale (pixels per
//   metre) and one ground line for all of them;
// - faceFrame and headFrame: the face and the head-and-shoulders frames of the expression grid;
// - pageLayout: the places of the panels' pictures, labels, ground lines and texts on the page.
// The sheet's cameras look along the horizontal, so a box's x runs to the picture's right and its y up the world.

export interface Box { x0: number; x1: number; y0: number; y1: number }
// a camera frame: the target plane's centre (cx, cy), half its height in metres, and the picture's size in pixels
export interface Frame { cx: number; cy: number; halfH: number; width: number; height: number }

// the box of the pixels of a probe picture whose alpha passes the threshold (alpha: one byte per pixel, rows from the
// top), in metres on the target plane of a frame centred on (cx, cy) with halfH; edge: the box touches the border, so
// the figure may reach beyond the probe
export function maskBox(alpha: ArrayLike<number>, w: number, h: number, probe: { cx: number; cy: number; halfH: number }, threshold = 8): { box: Box; edge: boolean } | null {
  let i0 = w, i1 = -1, j0 = h, j1 = -1;
  for (let j = 0; j < h; j++) {
    const row = j * w;
    for (let i = 0; i < w; i++) {
      if (alpha[row + i] < threshold) continue;
      if (i < i0) i0 = i;
      if (i > i1) i1 = i;
      if (j < j0) j0 = j;
      if (j > j1) j1 = j;
    }
  }
  if (i1 < 0) return null;
  const m = 2 * probe.halfH / h;
  const box = { x0: probe.cx + (i0 - w / 2) * m, x1: probe.cx + (i1 + 1 - w / 2) * m, y1: probe.cy + (h / 2 - j0) * m, y0: probe.cy + (h / 2 - j1 - 1) * m };
  return { box, edge: i0 === 0 || j0 === 0 || i1 === w - 1 || j1 === h - 1 };
}

// the alpha channel of RGBA pixels
export function alphaOf(rgba: ArrayLike<number>): Uint8Array {
  const n = rgba.length / 4, out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = rgba[i * 4 + 3];
  return out;
}

// the share of the pixels that a picture covers (alpha of 8 or more): a picture of an empty view covers none
export function coverage(rgba: ArrayLike<number>): number {
  const n = rgba.length / 4;
  let k = 0;
  for (let i = 0; i < n; i++) if (rgba[i * 4 + 3] >= 8) k++;
  return n ? k / n : 0;
}

export interface FigureFrames { pxPerM: number; y0: number; y1: number; groundRow: number; frames: Frame[] }

// frames of full figures at one scale: every picture is `height` pixels tall and spans the same heights, from below
// the ground to above the highest top, so the ground lies on the same row of every picture; each picture is as wide
// as its figure. pad: the margin around the figures, as a share of the span.
export function figureFrames(boxes: Box[], ground: number, height: number, pad = 0.035): FigureFrames {
  const top = Math.max(...boxes.map((b) => b.y1));
  const bottom = Math.min(ground, ...boxes.map((b) => b.y0));
  const span = Math.max(1e-3, top - bottom), p = span * pad;
  const y1 = top + p, y0 = bottom - p;
  const pxPerM = height / (y1 - y0);
  const frames = boxes.map((b) => {
    const width = Math.max(2, 2 * Math.ceil((b.x1 - b.x0 + 2 * p) * pxPerM / 2));
    return { cx: (b.x0 + b.x1) / 2, cy: (y0 + y1) / 2, halfH: (y1 - y0) / 2, width, height };
  });
  return { pxPerM, y0, y1, groundRow: (y1 - ground) * pxPerM, frames };
}

// the face frame: from just below the chin to just above the top of the head or its hair, so the face (the chin to the
// hairline, about three quarters of the head) fills about 60 % of the height; tall headgear is cut a third of the head's
// height above its top
export function faceFrame(chinY: number, crownY: number, hairTop: number, headX: number, height: number, aspect = 0.8): Frame {
  const H = Math.max(0.05, crownY - chinY);
  const y0 = chinY - H * 0.09, y1 = Math.min(Math.max(hairTop, crownY), crownY + H * 0.3) + H * 0.05;
  const width = Math.max(2, 2 * Math.round(height * aspect / 2));
  return { cx: headX, cy: (y0 + y1) / 2, halfH: (y1 - y0) / 2, width, height };
}

// the head-and-shoulders frame: from above the top of the head (and its hair) to below the shoulders, centred on the
// head, as wide as aspect times its height
export function headFrame(top: number, shoulderY: number, headX: number, height: number, aspect = 0.9): Frame {
  const band = Math.max(0.05, top - shoulderY);
  const y1 = top + band * 0.12, y0 = shoulderY - band * 0.42;
  const width = Math.max(2, 2 * Math.round(height * aspect / 2));
  return { cx: headX, cy: (y0 + y1) / 2, halfH: (y1 - y0) / 2, width, height };
}

// ------------------------------------------------------------------ the page
export type Role = 'ink' | 'muted' | 'faint' | 'line';
export type Item =
  | { type: 'text'; x: number; y: number; text: string; size: number; font: 'serif' | 'sans'; weight?: number; italic?: boolean; caps?: boolean; align: 'left' | 'center' | 'right'; role: Role; maxW?: number }
  | { type: 'picture'; panel: string; index: number; x: number; y: number; w: number; h: number }
  | { type: 'line'; x0: number; y0: number; x1: number; y1: number; width: number; role: Role; dash?: number[] };

export interface PanelInput {
  id: 'info' | 'turnaround' | 'expressions' | 'poses';
  title: string;
  // the pictures' sizes in pixels, and the label under each one
  sizes: { w: number; h: number }[];
  labels: string[];
  // the ground row of the pictures (pixels from their top), for a ground line across the panel
  groundRow?: number;
}
export interface PageInput { k: number; name: string; title: string; subtitle: string; panels: PanelInput[] }
export interface PageLayout { width: number; height: number; items: Item[]; rects: Record<string, { x: number; y: number; w: number; h: number }> }

export const PAGE_WIDTH = 4096;
const MAX_ROW = 6;

// rows of at most six pictures, as even as their count allows (12 gives two rows of six, 14 three of five)
export function rows(n: number): number[] {
  if (n <= 0) return [];
  const r = Math.ceil(n / MAX_ROW), per = Math.ceil(n / r), out: number[] = [];
  for (let left = n; left > 0; left -= per) out.push(Math.min(per, left));
  return out;
}

// the page: the info block, then each panel under a heading, at k times the print size (k: the render scale)
export function pageLayout(input: PageInput): PageLayout {
  const k = input.k, W = Math.round(PAGE_WIDTH * k), M = 150 * k, A = W - 2 * M, G = 56 * k;
  const items: Item[] = [], rects: PageLayout['rects'] = {};
  const LABEL = 40 * k, HEAD = 42 * k;
  let y = M;
  const heading = (title: string) => {
    items.push({ type: 'text', x: M, y: y + HEAD, text: title, size: HEAD, font: 'sans', weight: 600, caps: true, align: 'left', role: 'muted' });
    y += HEAD + 26 * k;
    items.push({ type: 'line', x0: M, y0: y, x1: W - M, y1: y, width: 2 * k, role: 'line' });
    y += 48 * k;
  };
  const labelUnder = (x: number, w: number, top: number, text: string) => {
    if (text) items.push({ type: 'text', x: x + w / 2, y: top + LABEL * 1.45, text, size: LABEL, font: 'sans', weight: 500, align: 'center', role: 'muted', maxW: w + G * 0.8 });
  };
  let first = true;
  for (const P of input.panels) {
    if (!first) y += 110 * k;
    first = false;
    const top = y;
    if (P.id === 'info') {
      const NAME = 140 * k, TITLE = 58 * k, SUB = 36 * k;
      items.push({ type: 'text', x: M, y: y + NAME * 0.82, text: input.name || 'Character', size: NAME, font: 'serif', italic: true, align: 'left', role: 'ink', maxW: A * 0.7 });
      y += NAME;
      if (input.title) { items.push({ type: 'text', x: M, y: y + TITLE * 1.1, text: input.title, size: TITLE, font: 'sans', weight: 400, align: 'left', role: 'muted', maxW: A * 0.7 }); y += TITLE * 1.4; }
      items.push({ type: 'text', x: W - M, y: top + NAME * 0.4, text: 'Character sheet', size: SUB, font: 'sans', weight: 600, caps: true, align: 'right', role: 'muted' });
      if (input.subtitle) items.push({ type: 'text', x: W - M, y: top + NAME * 0.4 + SUB * 1.6, text: input.subtitle, size: SUB, font: 'sans', weight: 400, align: 'right', role: 'faint', maxW: A * 0.28 });
      y += 30 * k;
      items.push({ type: 'line', x0: M, y0: y, x1: W - M, y1: y, width: 3 * k, role: 'ink' });
      y += 6 * k;
      rects[P.id] = { x: M, y: top, w: A, h: y - top };
      continue;
    }
    heading(P.title);
    const n = P.sizes.length;
    if (P.id === 'expressions') {
      // a grid of equal cells, as wide as the page allows
      const R = rows(n), cols = R[0] || 1;
      const cw = (A - G * (cols - 1)) / cols, s0 = P.sizes[0] || { w: 9, h: 10 }, ch = cw * s0.h / s0.w;
      let i = 0;
      for (const count of R) {
        const x0 = M + (A - (count * cw + (count - 1) * G)) / 2;
        for (let c = 0; c < count; c++, i++) {
          const x = x0 + c * (cw + G);
          items.push({ type: 'picture', panel: P.id, index: i, x, y, w: cw, h: ch });
          labelUnder(x, cw, y + ch, P.labels[i] || '');
        }
        y += ch + LABEL * 1.6 + 28 * k;
      }
    } else {
      // full figures at one scale in rows; a ground line runs under each row
      const R = rows(n);
      let i = 0, fit = Infinity;
      for (const count of R) {
        const sum = P.sizes.slice(i, i + count).reduce((a, s) => a + s.w, 0);
        fit = Math.min(fit, 1, (A - G * (count - 1)) / Math.max(1, sum));
        i += count;
      }
      i = 0;
      for (const count of R) {
        const sizes = P.sizes.slice(i, i + count);
        const rowW = sizes.reduce((a, s) => a + s.w * fit, 0) + G * (count - 1);
        const rowH = Math.max(...sizes.map((s) => s.h * fit));
        let x = M + (A - rowW) / 2;
        if (P.groundRow !== undefined) {
          const gy = y + P.groundRow * fit;
          items.push({ type: 'line', x0: M, y0: gy, x1: W - M, y1: gy, width: 2 * k, role: 'faint' });
        }
        for (const s of sizes) {
          const w = s.w * fit, h = s.h * fit;
          items.push({ type: 'picture', panel: P.id, index: i, x, y, w, h });
          labelUnder(x, w, y + rowH, P.labels[i] || '');
          x += w + G; i++;
        }
        y += rowH + LABEL * 1.6 + 28 * k;
      }
    }
    rects[P.id] = { x: M, y: top, w: A, h: y - top };
  }
  return { width: W, height: Math.round(y + M - 40 * k), items, rects };
}

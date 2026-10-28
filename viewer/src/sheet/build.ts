// OpenSculptBoy
// Apache License, Version 2.0
//
// Build a character sheet from its settings (settings.ts): for each panel, frame its pictures from probes of the
// figure, render them offscreen in the panel's style (styles.ts) with transparent backgrounds, and lay out the page
// (layout.ts). The work runs in steps between frames, so the page keeps drawing: each picture's passes stop for the
// next frame once a frame has drawn about two million pixels. The faces of the expression grid come from the face
// interface (face/api.ts); until a face driver registers, every cell shows the neutral face.

import { face } from '../face/api.ts';
import type { FaceState } from '../face/api.ts';
import type { MeterReading, SheetHost, SheetView } from './host.ts';
import { alphaOf, coverage, faceFrame, figureFrames, headFrame, maskBox, pageLayout, type Box, type Frame, type PageLayout, type PanelInput } from './layout.ts';
import { expressionLabel, PANEL_LABELS, panelStyle, shownPanels, TURNAROUND_POSE, VIEWS, type PanelId, type PoseSlot, type SheetSettings } from './settings.ts';
import { styleById } from './styles.ts';

// the pictures' heights at print size (render scale 1)
export const HEIGHTS = { turnaround: 2048, expressions: 960, poses: 1536 };
// the expression cells: portrait for the face framing, wider for the head and shoulders
export const CELL_ASPECT = { face: 0.8, shoulders: 0.9 };
const PROBE = 192;
// the height of the meter's picture of the whole figure
const METER_H = 320;
const PIXELS_PER_FRAME = 2.2e6;

export interface Picture { data: ImageData; bitmap: ImageBitmap | null; w: number; h: number; coverage: number; label: string }
// boxes: what the probes saw of each picture's figure, in metres on the target plane (layout.ts Box)
export interface PanelResult { id: PanelId; title: string; labels: string[]; frames: Frame[]; boxes: Box[]; groundRow?: number; pictures: (Picture | null)[]; ms: number }
// meter: the reading that set the exposure of the studio lights (null with the page's lights)
export interface SheetResult { settings: SheetSettings; name: string; panels: PanelResult[]; layout: PageLayout; ms: number; notes: string[]; meter: MeterReading | null }

export interface BuildEvents {
  // the share of the work done, and what runs now
  progress?(share: number, label: string): void;
  // a picture is ready, or a panel changed its frames: the page can draw again
  changed?(partial: SheetResult): void;
}
export class SheetCancelled extends Error { constructor() { super('The sheet was stopped.'); } }

const nextFrame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));

export function estimatePanel(id: PanelId, n: number, k: number, aspect = CELL_ASPECT.face): { w: number; h: number }[] {
  if (id === 'expressions') { const h = Math.round(HEIGHTS.expressions * k); return Array.from({ length: n }, () => ({ w: 2 * Math.round(h * aspect / 2), h })); }
  const h = Math.round((id === 'turnaround' ? HEIGHTS.turnaround : HEIGHTS.poses) * k);
  return Array.from({ length: n }, () => ({ w: Math.round(h * (id === 'turnaround' ? 0.4 : 0.5)), h }));
}

// the page of a (partial) result: built pictures take their real size, the others an estimate
export function layoutOf(settings: SheetSettings, name: string, panels: PanelResult[]): PageLayout {
  const k = settings.render.scale;
  const inputs: PanelInput[] = panels.map((P) => ({
    id: P.id, title: P.title, labels: P.labels, groundRow: P.groundRow,
    sizes: P.frames.length ? P.frames.map((f) => ({ w: f.width, h: f.height })) : estimatePanel(P.id, P.labels.length, k, CELL_ASPECT[settings.panels.expressions.framing]),
  }));
  return pageLayout({ k, name, title: settings.title, subtitle: 'Humanoid 3D character, made with OpenSculptBoy', panels: inputs });
}

export function planPanels(settings: SheetSettings, host: SheetHost): PanelResult[] {
  const P = settings.panels, poses = host.poses();
  const poseLabel = (s: PoseSlot) => s.label || poses.find((p) => p.name === s.pose)?.label || s.pose;
  return shownPanels(settings).map((id) => {
    const labels = id === 'turnaround' ? P.turnaround.views.map((v) => VIEWS[v].label)
      : id === 'expressions' ? P.expressions.items.map(expressionLabel)
        : id === 'poses' ? P.poses.slots.map(poseLabel) : [];
    return { id, title: PANEL_LABELS[id], labels, frames: [], boxes: [], pictures: labels.map(() => null), ms: 0 };
  });
}

export async function buildSheet(host: SheetHost, settings: SheetSettings, ev: BuildEvents = {}, ctl: { cancelled: boolean } = { cancelled: false }): Promise<SheetResult> {
  const t0 = performance.now(), k = settings.render.scale, S = settings.render.samples;
  const name = settings.name || host.characterName();
  const panels = planPanels(settings, host);
  const notes: string[] = [];
  let meter: MeterReading | null = null;
  const result = (): SheetResult => ({ settings, name, panels, layout: layoutOf(settings, name, panels), ms: Math.round(performance.now() - t0), notes, meter });
  // the work in pixels drawn: the probes and every pass of every picture, with the estimated sizes until the frames
  let total = 0, done = 0, budget = 0;
  const aspect = CELL_ASPECT[settings.panels.expressions.framing];
  for (const P of panels) for (const s of estimatePanel(P.id, P.labels.length, k, aspect)) total += (s.w * s.h * S) + PROBE * PROBE;
  if (settings.render.lights === 'studio') total += PROBE * PROBE + METER_H * METER_H;
  let label = '';
  const say = (l: string) => { label = l; ev.progress?.(Math.min(0.99, done / Math.max(1, total)), label); };
  const tick = async (px: number) => {
    done += px; budget += px;
    ev.progress?.(Math.min(0.99, done / Math.max(1, total)), label);
    if (budget >= PIXELS_PER_FRAME) { budget = 0; await nextFrame(); }
    if (ctl.cancelled) throw new SheetCancelled();
  };
  const show = (slot: PoseSlot) => {
    if (host.showPose(slot)) return;
    notes.push(`The pose ${slot.label || slot.pose} is not in the library, so the sheet shows ${TURNAROUND_POSE}`);
    host.showPose({ pose: TURNAROUND_POSE });
  };
  // the box of the whole figure as the camera sees it: a probe wide enough for the joints, the hair and loose clothes,
  // twice wider when the figure reaches its border
  const probeFigure = async (turn: number, style: string): Promise<Box> => {
    const F = host.figure();
    const cx = (F.lo[0] + F.hi[0]) / 2, cy = (F.ground + F.hi[1] + 0.3) / 2;
    let halfH = 1.15 * Math.max((F.hi[1] + 0.3 - F.ground) / 2, (F.hi[0] - F.lo[0]) / 2 + 0.35, (F.hi[2] - F.lo[2]) / 2 + 0.35);
    for (let tries = 0; ; tries++) {
      const view: SheetView = { turn, cx, cy, halfH, width: PROBE, height: PROBE };
      const img = await styleById(style).render(host, view, 1, tick);
      const m = maskBox(alphaOf(img.data), PROBE, PROBE, view);
      if (!m) return { x0: cx - 0.2, x1: cx + 0.2, y0: F.ground, y1: F.hi[1] };
      if (!m.edge || tries >= 2) return m.box;
      halfH *= 1.6;
    }
  };
  const picture = async (P: PanelResult, i: number, view: SheetView, style: string) => {
    const data = await styleById(style).render(host, view, S, tick);
    const bitmap = typeof createImageBitmap === 'function' ? await createImageBitmap(data) : null;
    P.pictures[i] = { data, bitmap, w: data.width, h: data.height, coverage: +coverage(data.data).toFixed(4), label: P.labels[i] };
    ev.changed?.(result());
    await nextFrame();
  };
  // a panel's work from the estimate to its real frames
  const reframe = (P: PanelResult, frames: Frame[]) => {
    for (const s of estimatePanel(P.id, P.labels.length, k, aspect)) total -= s.w * s.h * S;
    for (const f of frames) total += f.width * f.height * S;
    P.frames = frames;
    ev.changed?.(result());
  };
  const savedFace: FaceState = face().get();
  host.hold(true);
  try {
    host.lights(settings.render.lights);
    if (settings.render.lights === 'studio') {
      // one exposure for the whole sheet, metered on the whole figure from the front in the turnaround's pose
      say('Metering the light');
      show(settings.panels.turnaround.pose); host.turn(0);
      const style = panelStyle(settings, 'turnaround');
      const ff = figureFrames([await probeFigure(0, style)], host.figure().ground, METER_H);
      meter = await host.meter({ turn: 0, ...ff.frames[0] }, settings.render.exposure, tick);
    }
    for (const P of panels) {
      const tp = performance.now(), style = panelStyle(settings, P.id);
      if (P.id === 'turnaround') {
        const T = settings.panels.turnaround, turns = T.views.map((v) => -VIEWS[v].yaw);
        say('Turnaround: framing');
        show(T.pose);
        const boxes: Box[] = [];
        for (const turn of turns) { host.turn(turn); boxes.push(await probeFigure(turn, style)); }
        const ff = figureFrames(boxes, host.figure().ground, Math.round(HEIGHTS.turnaround * k));
        P.groundRow = ff.groundRow; P.boxes = boxes;
        reframe(P, ff.frames);
        for (let i = 0; i < turns.length; i++) {
          say(`Turnaround: ${P.labels[i]}`);
          host.turn(turns[i]);
          await picture(P, i, { turn: turns[i], ...ff.frames[i] }, style);
        }
      } else if (P.id === 'expressions') {
        const E = settings.panels.expressions, h = Math.round(HEIGHTS.expressions * k);
        let turn = -VIEWS[E.view].yaw;
        say('Expressions: preparing the faces');
        await host.warmFaces();
        say('Expressions: framing');
        show(E.pose);
        await face().set({ actions: {} });
        if (E.framing === 'face') {
          // the face looks straight at the camera: the figure turns by the head's own turn in the pose (up to 45°)
          host.turn(0);
          const d = host.figure().headDir, psi = Math.atan2(d[0], d[2]) * 180 / Math.PI;
          turn -= Math.max(-45, Math.min(45, psi));
        }
        host.turn(turn);
        const F = host.figure(), J = F.joints;
        const neck = J.neck01 ? J.neck01[1] : F.top - 0.3;
        const size = Math.max(0.1, F.top - neck);
        const head = Math.max(0.05, F.crown[1] - F.chin[1]);
        const probe: SheetView = E.framing === 'face'
          ? { turn, cx: F.head[0], cy: (F.chin[1] + F.crown[1]) / 2 + head * 0.1, halfH: head * 1.0, width: PROBE, height: PROBE }
          : { turn, cx: F.head[0], cy: F.head[1] + size * 0.2, halfH: size * 1.9, width: PROBE, height: PROBE };
        const img = await styleById(style).render(host, probe, 1, tick);
        const m = maskBox(alphaOf(img.data), PROBE, PROBE, probe);
        let fr: Frame;
        if (E.framing === 'face') {
          fr = faceFrame(F.chin[1], F.crown[1], m ? m.box.y1 : F.crown[1], F.head[0], h, CELL_ASPECT.face);
        } else {
          const top = m ? m.box.y1 : F.top + 0.03;
          const sh = ['upperarm01.L', 'upperarm01.R'].filter((n) => J[n]).map((n) => J[n][1]);
          const shoulder = sh.length ? sh.reduce((a, b) => a + b, 0) / sh.length : neck - size * 0.35;
          fr = headFrame(top, shoulder, F.head[0], h, CELL_ASPECT.shoulders);
        }
        P.boxes = m ? [m.box] : [];
        reframe(P, E.items.map(() => fr));
        for (let i = 0; i < E.items.length; i++) {
          say(`Expressions: ${P.labels[i]}`);
          await face().set(face().preset(E.items[i]));
          await picture(P, i, { turn, ...fr }, style);
        }
        await face().set(savedFace);
      } else if (P.id === 'poses') {
        const Q = settings.panels.poses, turn = -VIEWS[Q.view].yaw;
        const boxes: Box[] = [];
        for (let i = 0; i < Q.slots.length; i++) {
          say(`Poses: framing ${P.labels[i]}`);
          show(Q.slots[i]); host.turn(turn);
          boxes.push(await probeFigure(turn, style));
        }
        const ff = figureFrames(boxes, host.figure().ground, Math.round(HEIGHTS.poses * k));
        P.groundRow = ff.groundRow; P.boxes = boxes;
        reframe(P, ff.frames);
        for (let i = 0; i < Q.slots.length; i++) {
          say(`Poses: ${P.labels[i]}`);
          show(Q.slots[i]); host.turn(turn);
          await picture(P, i, { turn, ...ff.frames[i] }, style);
        }
      }
      P.ms = Math.round(performance.now() - tp);
    }
  } finally {
    await face().set(savedFace).catch(() => {});
    host.lights('page');
    host.hold(false);
  }
  ev.progress?.(1, 'Done');
  return result();
}

// what a test or a command reads of a sheet: the page's size, the time, and each picture's size and coverage
export function report(r: SheetResult, pngBytes = 0) {
  return {
    width: r.layout.width, height: r.layout.height, ms: r.ms, png_bytes: pngBytes, name: r.name, notes: r.notes.slice(),
    lights: r.settings.render.lights, meter: r.meter,
    panels: Object.fromEntries(r.panels.map((P) => [P.id, { ms: P.ms, ground_row: P.groundRow ?? null, frames: P.frames, boxes: P.boxes,
      pictures: P.pictures.map((p, i) => (p ? { label: p.label, w: p.w, h: p.h, coverage: p.coverage } : { label: P.labels[i], w: 0, h: 0, coverage: 0 })),
      rect: r.layout.rects[P.id] || null }])),
  };
}

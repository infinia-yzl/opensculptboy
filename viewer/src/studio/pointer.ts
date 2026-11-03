// OpenSculptBoy
// Apache License, Version 2.0
//
// Pointer arbitration in the 3D view (DESIGN-0014, "Selection, picking and occlusion" and the mouse map of key map
// version 2). Parts of the page that put things to pick in the view (points, rings, handles, joints) register a
// picker; on a press of the left button the pickers are asked in priority order, the nearest hit within the picking
// radius wins (visible items before hidden ones, then the nearest to the pointer), and its picker owns the click or
// the drag. A press that hits nothing, Alt with the left button, the middle button and the wheel go to the camera.
// A second click on the same spot selects the next item behind. A right-click without a drag opens the item's menu,
// and a right-click or Esc during a drag cancels it. Hover asks the same pickers, for the cursor and the highlight.
//
// The camera's controller (ui/camera_nav.ts) registers itself as the driver of the gestures that no picker takes;
// surfaces (the body, a statue's scan) register a ray test, which the double-click uses to set the pivot.

import type * as THREE from 'three';
import { cam } from './camera.ts';
import { defineScope, registerAction } from './actions.ts';
import { settings } from './settings.ts';

export interface PointerInfo {
  x: number; y: number;               // client pixels
  ndc: { x: number; y: number };      // the canvas from -1 to 1, y up
  ray: THREE.Ray;                     // from the camera through the pointer
  radius: number;                     // the picking radius in CSS pixels (16, or 24 with Larger targets)
  shift: boolean; ctrl: boolean; alt: boolean;
  event: PointerEvent | MouseEvent | WheelEvent;
}
export interface PickHit {
  item: unknown;                      // the picker's own item (a point's index, a joint's name)
  distPx: number;                     // from the pointer on screen, in CSS pixels
  depth?: number;                     // from the camera, metres (for the cycling order)
  hidden?: boolean;                   // behind a surface: visible items win
  label?: string;                     // its plain name ('Right index finger, middle joint')
  cursor?: string;
  picker?: string;                    // filled in by the arbitration
}
export interface DragInfo {
  x0: number; y0: number;             // the press, client pixels
  dx: number; dy: number;             // the pointer since the press
  ddx: number; ddy: number;           // the pointer since the last move
  fine: boolean;                      // Shift: a quarter of the movement
  snap: boolean;                      // Ctrl
  scale: number;                      // the drag speed of the settings, times a quarter with Shift
}
export interface Picker {
  id: string;
  priority: number;                   // higher first (a ring 30, a handle 20, a point 10, a joint 0)
  enabled?: () => boolean;
  hit(p: PointerInfo): PickHit | PickHit[] | null;
  onDown?(hit: PickHit, p: PointerInfo): boolean | void;          // false declines: the camera takes the press
  onClick?(hit: PickHit, p: PointerInfo): void;                   // a press and release without a drag
  onDrag?(hit: PickHit, p: PointerInfo, d: DragInfo): void;       // without it, a drag from the item turns the camera
  onUp?(hit: PickHit, p: PointerInfo, d: DragInfo): void;
  onCancel?(hit: PickHit): void;                                  // Esc or a right-click: put the item back
  onWheel?(hit: PickHit, p: PointerInfo, notches: number): boolean | void;   // the wheel during its drag (push in, pull out)
  onHover?(hit: PickHit | null, p: PointerInfo | null): void;
  onDoubleClick?(hit: PickHit, p: PointerInfo): boolean | void;   // false lets the camera take it
  onContext?(hit: PickHit, p: PointerInfo): boolean | void;       // a right-click without a drag
  onBackground?(p: PointerInfo): void;                            // a click on nothing (clear the selection)
  cursor?: string | ((hit: PickHit) => string);                   // over a hit (default 'pointer')
}

// what the camera does with the gestures that no picker takes
export type CameraGesture = 'orbit' | 'pan' | 'zoom' | 'right';
export interface CameraDriver {
  begin(kind: CameraGesture, e: PointerEvent, from: { x: number; y: number }): void;
  move(e: PointerEvent): void;
  end(e: PointerEvent): void;
  cancel(): void;
  wheel(e: WheelEvent): void;
  doubleClick(e: MouseEvent, surface: SurfaceHit | null): void;
  // the cursor while the camera's gesture lasts
  cursor(): string;
}
export interface SurfaceHit { point: THREE.Vector3; distance: number; id: string }
export type Surface = (ray: THREE.Ray) => { point: THREE.Vector3; distance: number } | null;

const PICKERS = new Map<string, Picker>();
const SURFACES = new Map<string, Surface>();
let driver: CameraDriver | null = null;

export function registerPicker(p: Picker): () => void {
  PICKERS.set(p.id, p);
  return () => { if (PICKERS.get(p.id) === p) PICKERS.delete(p.id); };
}
export function registerSurface(id: string, s: Surface): () => void {
  SURFACES.set(id, s);
  return () => { if (SURFACES.get(id) === s) SURFACES.delete(id); };
}
export function registerCameraDriver(d: CameraDriver) { driver = d; }
// the first surface a ray meets
export function surfaceHit(ray: THREE.Ray): SurfaceHit | null {
  let best: SurfaceHit | null = null;
  for (const [id, s] of SURFACES) {
    const h = s(ray);
    if (h && (!best || h.distance < best.distance)) best = { ...h, id };
  }
  return best;
}

// ------------------------------------------------------------------ the state of a gesture
type Gesture =
  | { kind: 'none' }
  | { kind: 'press'; picker: Picker; hit: PickHit; x0: number; y0: number; id: number }
  | { kind: 'drag'; picker: Picker; hit: PickHit; x0: number; y0: number; lx: number; ly: number; id: number }
  | { kind: 'camera'; x0: number; y0: number; id: number; moved: boolean; button: number }
  | { kind: 'right'; x0: number; y0: number; id: number; hits: PickHit[] };
let g: Gesture = { kind: 'none' };
let hovered: { picker: Picker; hit: PickHit } | null = null;
let lastClick: { x: number; y: number; t: number; idx: number; n: number } | null = null;
let canvasEl: HTMLCanvasElement | null = null;
const DRAG_PX = 4;

export function dragging(): boolean { return g.kind === 'drag'; }
export function gestureKind(): string { return g.kind; }
export function hoveredHit(): PickHit | null { return hovered ? hovered.hit : null; }

function info(e: PointerEvent | MouseEvent | WheelEvent): PointerInfo {
  const r = canvasEl!.getBoundingClientRect();
  const ndc = { x: (e.clientX - r.left) / r.width * 2 - 1, y: -((e.clientY - r.top) / r.height) * 2 + 1 };
  return { x: e.clientX, y: e.clientY, ndc, ray: cam().rayAt(e.clientX, e.clientY), radius: settings.pickRadius(),
    shift: e.shiftKey, ctrl: e.ctrlKey || e.metaKey, alt: e.altKey, event: e };
}
// every hit under the pointer, in the order of the arbitration
export function hitsAt(p: PointerInfo): { picker: Picker; hit: PickHit }[] {
  const out: { picker: Picker; hit: PickHit }[] = [];
  for (const pk of PICKERS.values()) {
    if (pk.enabled && !pk.enabled()) continue;
    const r = pk.hit(p);
    const list = !r ? [] : Array.isArray(r) ? r : [r];
    for (const h of list) if (h && h.distPx <= p.radius) out.push({ picker: pk, hit: { ...h, picker: pk.id } });
  }
  out.sort((a, b) => b.picker.priority - a.picker.priority || Number(!!a.hit.hidden) - Number(!!b.hit.hidden)
    || a.hit.distPx - b.hit.distPx || (a.hit.depth ?? 0) - (b.hit.depth ?? 0));
  return out;
}
function dragInfo(e: PointerEvent, x0: number, y0: number, lx: number, ly: number): DragInfo {
  const fine = e.shiftKey;
  return { x0, y0, dx: e.clientX - x0, dy: e.clientY - y0, ddx: e.clientX - lx, ddy: e.clientY - ly, fine, snap: e.ctrlKey || e.metaKey,
    scale: settings.get().dragSpeed * (fine ? 0.25 : 1) };
}
function setCursor(c: string) { if (canvasEl && canvasEl.style.cursor !== c) canvasEl.style.cursor = c; }
function cursorOf(pk: Picker, h: PickHit) { return h.cursor || (typeof pk.cursor === 'function' ? pk.cursor(h) : pk.cursor) || 'pointer'; }

// cancel the gesture in progress: a picker's drag puts its item back; the camera stops where it is
export function cancelGesture(): boolean {
  if (g.kind === 'drag' || g.kind === 'press') {
    const { picker, hit } = g;
    g = { kind: 'none' };
    picker.onCancel?.(hit);
    setCursor('default');
    return true;
  }
  if (g.kind === 'camera') { g = { kind: 'none' }; driver?.cancel(); setCursor('default'); return true; }
  return false;
}

// ------------------------------------------------------------------ the listeners
export function installPointer(canvas: HTMLCanvasElement) {
  if (canvasEl) return;
  canvasEl = canvas;
  canvas.style.touchAction = 'none';
  // Esc during a drag puts the item back (the deepest scope: it comes before clearing the selection)
  defineScope({ id: 'drag', active: () => g.kind === 'drag' || g.kind === 'camera', rank: 100 });
  registerAction({ id: 'drag.cancel', label: 'Cancel the drag', keys: ['Escape'], group: 'Mouse', tier: 'constant', scope: 'drag',
    inDialog: true, help: false, command: false, run: () => cancelGesture() });

  // capture on the window, so the arbitration comes before any listener of the canvas
  const onCanvas = (e: Event) => e.target === canvas;
  window.addEventListener('pointerdown', (e) => {
    if (!onCanvas(e) || e.pointerType === 'touch') return;
    // a second button during a drag: a right-click cancels it
    if (g.kind === 'drag' || g.kind === 'press') { if (e.button === 2) { cancelGesture(); e.preventDefault(); e.stopImmediatePropagation(); } return; }
    if (g.kind !== 'none') return;
    canvas.focus({ preventScroll: true });
    const p = info(e);
    e.stopImmediatePropagation();
    if (e.button === 0 && !e.altKey) {
      const hits = hitsAt(p);
      // a second click on the same spot: the next item behind
      let idx = 0;
      const now = performance.now();
      if (lastClick && hits.length > 1 && Math.hypot(e.clientX - lastClick.x, e.clientY - lastClick.y) < 4 && now - lastClick.t < 1500 && lastClick.n === hits.length) idx = (lastClick.idx + 1) % hits.length;
      for (let k = 0; k < hits.length; k++) {
        const { picker, hit } = hits[(idx + k) % hits.length];
        if (picker.onDown?.(hit, p) === false) continue;
        g = { kind: 'press', picker, hit, x0: e.clientX, y0: e.clientY, id: e.pointerId };
        lastClick = { x: e.clientX, y: e.clientY, t: now, idx: (idx + k) % hits.length, n: hits.length };
        canvas.setPointerCapture(e.pointerId);
        return;
      }
      lastClick = null;
    }
    if (!driver) return;
    const kind: CameraGesture = e.button === 0 ? 'orbit' : e.button === 1 ? 'pan' : e.altKey ? 'zoom' : 'right';
    if (e.button === 2 && !e.altKey) {
      g = { kind: 'right', x0: e.clientX, y0: e.clientY, id: e.pointerId, hits: hitsAt(p).map((x) => x.hit) };
    } else {
      g = { kind: 'camera', x0: e.clientX, y0: e.clientY, id: e.pointerId, moved: false, button: e.button };
      driver.begin(kind, e, { x: e.clientX, y: e.clientY });
      setCursor(driver.cursor());
    }
    canvas.setPointerCapture(e.pointerId);
    if (e.button === 1) e.preventDefault();
  }, true);

  window.addEventListener('pointermove', (e) => {
    if (!onCanvas(e) && g.kind === 'none') { if (hovered) { hovered.picker.onHover?.(null, null); hovered = null; } return; }
    if (g.kind === 'none') { hover(e); return; }
    if (e.pointerId !== g.id) return;
    e.stopImmediatePropagation();
    if (g.kind === 'press') {
      if (Math.hypot(e.clientX - g.x0, e.clientY - g.y0) < DRAG_PX) return;
      const { picker, hit, x0, y0, id } = g;
      if (picker.onDrag) { g = { kind: 'drag', picker, hit, x0, y0, lx: x0, ly: y0, id }; }
      else {
        // a drag from an item that does not move turns the camera from the press
        g = { kind: 'camera', x0, y0, id, moved: true, button: 0 };
        driver?.begin('orbit', e, { x: x0, y: y0 });
        setCursor(driver?.cursor() || 'grabbing');
      }
    }
    if (g.kind === 'drag') {
      const d = dragInfo(e, g.x0, g.y0, g.lx, g.ly);
      g.lx = e.clientX; g.ly = e.clientY;
      g.picker.onDrag!(g.hit, info(e), d);
      setCursor('grabbing');
    } else if (g.kind === 'right') {
      if (Math.hypot(e.clientX - g.x0, e.clientY - g.y0) >= DRAG_PX && driver) {
        const { x0, y0, id } = g;
        g = { kind: 'camera', x0, y0, id, moved: true, button: 2 };
        driver.begin('right', e, { x: x0, y: y0 });
        setCursor(driver.cursor());
      }
    } else if (g.kind === 'camera') { g.moved = true; driver?.move(e); }
  }, true);

  const up = (e: PointerEvent) => {
    if (g.kind === 'none' || e.pointerId !== g.id) return;
    e.stopImmediatePropagation();
    const p = info(e), was = g;
    g = { kind: 'none' };
    try { canvas.releasePointerCapture(e.pointerId); } catch { /* released already */ }
    if (was.kind === 'press') was.picker.onClick?.(was.hit, p);
    else if (was.kind === 'drag') was.picker.onUp?.(was.hit, p, dragInfo(e, was.x0, was.y0, was.lx, was.ly));
    else if (was.kind === 'camera') {
      driver?.end(e);
      // a click on the background: the pickers clear their selection
      if (!was.moved && was.button === 0 && Math.hypot(e.clientX - was.x0, e.clientY - was.y0) < DRAG_PX) {
        for (const pk of PICKERS.values()) if (!pk.enabled || pk.enabled()) pk.onBackground?.(p);
      }
    } else if (was.kind === 'right') {
      for (const { picker, hit } of hitsAt(p)) if (picker.onContext && picker.onContext(hit, p) !== false) break;
    }
    setCursor('default');
    hover(e);
  };
  window.addEventListener('pointerup', up, true);
  window.addEventListener('pointercancel', (e) => { if (g.kind !== 'none' && e.pointerId === g.id) { cancelGesture(); e.stopImmediatePropagation(); } }, true);

  // the menu of the browser never shows over the view: a right-click is the studio's
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  // the wheel: a picker's drag takes it (push in, pull out), else the camera zooms; Ctrl+wheel never zooms the page
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault(); e.stopImmediatePropagation();
    if (g.kind === 'drag' && g.picker.onWheel) {
      const notches = Math.sign(e.deltaY) * Math.max(1, Math.round(Math.abs(e.deltaY) / 100));
      if (g.picker.onWheel(g.hit, info(e), notches) !== false) return;
    }
    driver?.wheel(e);
  }, { passive: false, capture: true });
  // a double-click: a picker's item first, else the camera sets the pivot on the surface (or frames the figure)
  canvas.addEventListener('dblclick', (e) => {
    e.stopImmediatePropagation();
    const p = info(e);
    if (!e.altKey) for (const { picker, hit } of hitsAt(p)) if (picker.onDoubleClick && picker.onDoubleClick(hit, p) !== false) return;
    driver?.doubleClick(e, surfaceHit(p.ray));
  }, true);
  canvas.addEventListener('pointerleave', () => { if (g.kind === 'none' && hovered) { hovered.picker.onHover?.(null, null); hovered = null; setCursor('default'); } });
}

// hover: the item under the pointer grows and names itself (its picker draws that); the cursor says what a press does
let hoverQueued: PointerEvent | null = null;
function hover(e: PointerEvent) {
  if (!canvasEl || e.target !== canvasEl) return;
  if (!hoverQueued) requestAnimationFrame(() => {
    const ev = hoverQueued!; hoverQueued = null;
    if (g.kind !== 'none') return;
    const p = info(ev);
    const top = ev.altKey ? null : hitsAt(p)[0] || null;
    if (hovered && (!top || top.picker !== hovered.picker || top.hit.item !== hovered.hit.item)) hovered.picker.onHover?.(null, null);
    if (top && (!hovered || top.picker !== hovered.picker || top.hit.item !== hovered.hit.item)) top.picker.onHover?.(top.hit, p);
    hovered = top;
    setCursor(top ? cursorOf(top.picker, top.hit) : 'default');
  });
  hoverQueued = e;
}

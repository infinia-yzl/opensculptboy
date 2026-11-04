// OpenSculptBoy
// Apache License, Version 2.0
//
// The studio's control settings (DESIGN-0014, "Settings", View menu › Controls): four speeds, each from half to twice
// its default, Larger targets, the Blender middle mouse, and the prompt before leaving with unsaved edits. The camera reads the turn, move and zoom speeds; the
// posing editor reads the drag speed of points and rings and the picking radius. The settings stay in this browser
// for this person; a browser that keeps nothing (a private window, blocked storage) gets the defaults.

export interface ControlSettings {
  turnSpeed: number;        // the camera's turn by the mouse and the arrow keys, 0.5 to 2
  moveSpeed: number;        // the view's move by the keys (the mouse's pan stays 1:1), 0.5 to 2
  zoomStep: number;         // the wheel's zoom step, 0.5 to 2 (1: 12 % a notch)
  dragSpeed: number;        // the drag speed of points and rings, 0.5 to 2
  largerTargets: boolean;   // points, handles and rings draw larger, and the picking radius grows from 16 to 24 px
  blenderMiddle: boolean;   // the middle mouse turns the camera and Shift+middle moves the view, as in Blender
  warnUnsaved: boolean;     // the browser asks before the page goes with unsaved pose or point edits
}

export const DEFAULTS: ControlSettings = { turnSpeed: 1, moveSpeed: 1, zoomStep: 1, dragSpeed: 1, largerTargets: false, blenderMiddle: false, warnUnsaved: true };
export const SPEED_RANGE: [number, number] = [0.5, 2];
export const SPEEDS: { key: keyof ControlSettings; label: string; note: string }[] = [
  { key: 'turnSpeed', label: 'Turn speed', note: 'How fast a drag or an arrow key turns the camera' },
  { key: 'moveSpeed', label: 'Move speed', note: 'How far the keys move the view; the mouse moves it under the pointer' },
  { key: 'zoomStep', label: 'Zoom step', note: 'How far a notch of the wheel zooms' },
  { key: 'dragSpeed', label: 'Drag speed of points and rings', note: 'How far a point or a ring moves for the pointer’s movement' },
];

const KEY = 'osb.controls';
const clampSpeed = (v: unknown, d: number) => (typeof v === 'number' && isFinite(v) ? Math.min(SPEED_RANGE[1], Math.max(SPEED_RANGE[0], v)) : d);

function load(): ControlSettings {
  let o: any = {};
  try { o = JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch { o = {}; }
  return {
    turnSpeed: clampSpeed(o.turnSpeed, 1), moveSpeed: clampSpeed(o.moveSpeed, 1), zoomStep: clampSpeed(o.zoomStep, 1),
    dragSpeed: clampSpeed(o.dragSpeed, 1), largerTargets: !!o.largerTargets, blenderMiddle: !!o.blenderMiddle, warnUnsaved: o.warnUnsaved !== false,
  };
}

let state: ControlSettings = typeof localStorage === 'undefined' ? { ...DEFAULTS } : load();
const listeners = new Set<() => void>();

export const settings = {
  get(): ControlSettings { return state; },
  set(patch: Partial<ControlSettings>) {
    const next = { ...state };
    for (const [k, v] of Object.entries(patch) as [keyof ControlSettings, any][]) {
      (next as any)[k] = typeof DEFAULTS[k] === 'boolean' ? !!v : clampSpeed(v, DEFAULTS[k] as number);
    }
    if (Object.keys(next).every((k) => (next as any)[k] === (state as any)[k])) return;
    state = next;
    try { localStorage.setItem(KEY, JSON.stringify(state)); } catch { /* the browser keeps nothing: the settings last this visit */ }
    for (const l of listeners) l();
  },
  reset(key?: keyof ControlSettings) { settings.set(key ? { [key]: DEFAULTS[key] } : { ...DEFAULTS }); },
  subscribe(fn: () => void): () => void { listeners.add(fn); return () => { listeners.delete(fn); }; },
  // the picking radius in CSS pixels: the nearest pickable item within it lights up before the click
  pickRadius(): number { return state.largerTargets ? 24 : 16; },
  // the scale of points, handles and rings as drawn
  targetScale(): number { return state.largerTargets ? 1.5 : 1; },
};

// the page tests read and set the settings
if (typeof window !== 'undefined') (window as any).__studioSettings = settings;

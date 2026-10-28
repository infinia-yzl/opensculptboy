// OpenSculptBoy
// Apache License, Version 2.0
//
// The settings of a character sheet (docs/viewer.md, "Sheet"): one plain object with the style, the background, the
// name and title, the panels with their poses and expressions, and the render quality. The Sheet section edits it,
// window.sheet.build takes it, and a later card stores it in its `sheet` block, so the same settings build the same
// sheet. normaliseSettings fills what an object leaves out and drops what it cannot use; it needs no DOM, so node
// tests run it.

export const SETTINGS_FORMAT = 'opensculptboy/sheet-settings@1';

// a pose of the page's library by name, or a pose of its own (a pose read from a picture): the rotation of every bone
// relative to its parent as quaternions (x, y, z, w per bone, in the order of the page's rig)
export interface PoseSlot { pose: string; label?: string; q?: number[] }

export type ViewName = 'front' | 'three' | 'side' | 'back';
export type Background = 'paper' | 'dark' | 'transparent';

export interface TurnaroundPanel { on: boolean; style: string | null; pose: PoseSlot; views: ViewName[] }
// framing: 'face' frames each cell from below the chin to above the head, 'shoulders' takes the head and shoulders
export type Framing = 'face' | 'shoulders';
export interface ExpressionsPanel { on: boolean; style: string | null; pose: PoseSlot; view: ViewName; framing: Framing; items: string[] }
export interface PosesPanel { on: boolean; style: string | null; view: ViewName; slots: PoseSlot[] }
export interface InfoPanel { on: boolean }

export interface SheetSettings {
  format: string;
  style: string;
  background: Background;
  // the name on the sheet; empty: the name of the card or the look shown
  name: string;
  title: string;
  panels: { turnaround: TurnaroundPanel; expressions: ExpressionsPanel; poses: PosesPanel; info: InfoPanel };
  // scale: the size of every picture against the print size (1); samples: the accumulated samples of each picture
  render: { scale: number; samples: number };
}

export const PANEL_ORDER = ['info', 'turnaround', 'expressions', 'poses'] as const;
export type PanelId = typeof PANEL_ORDER[number];
export const PANEL_LABELS: Record<PanelId, string> = { info: 'Info block', turnaround: 'Turnaround', expressions: 'Expressions', poses: 'Poses' };

// the figure's view: the turn of the camera about the figure, in degrees toward the figure's left
export const VIEWS: Record<ViewName, { label: string; yaw: number }> = {
  front: { label: 'Front', yaw: 0 },
  three: { label: 'Three-quarter', yaw: 45 },
  side: { label: 'Side', yaw: 90 },
  back: { label: 'Back', yaw: 180 },
};
export const VIEW_ORDER: ViewName[] = ['front', 'three', 'side', 'back'];

// the expressions of the grid: the presets of the VRM export (data/vrm/expressions.json) that show on a face
export const EXPRESSION_ORDER = ['neutral', 'happy', 'angry', 'sad', 'relaxed', 'surprised', 'blink', 'blinkLeft', 'blinkRight', 'aa', 'ih', 'ou', 'ee', 'oh'];
export const DEFAULT_EXPRESSIONS = ['neutral', 'happy', 'angry', 'sad', 'relaxed', 'surprised', 'blink', 'aa', 'ih', 'ou', 'ee', 'oh'];
const EXPRESSION_LABELS: Record<string, string> = {
  neutral: 'Neutral', happy: 'Happy', angry: 'Angry', sad: 'Sad', relaxed: 'Relaxed', surprised: 'Surprised',
  blink: 'Blink', blinkLeft: 'Wink left', blinkRight: 'Wink right', aa: 'Vowel A', ih: 'Vowel I', ou: 'Vowel U', ee: 'Vowel E', oh: 'Vowel O',
};
export function expressionLabel(name: string) {
  return EXPRESSION_LABELS[name] || name.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/^./, (c) => c.toUpperCase());
}

// the turnaround's relaxed stance, and six poses of the library that show a standing character's manner
export const TURNAROUND_POSE = 'relaxed';
export const DEFAULT_POSES = ['weight_shift', 'arms_crossed', 'hands_on_hips', 'mh_thinking', 'mh_goodbye', 'mh_hero'];
export const MAX_POSES = 8;
export const MAX_EXPRESSIONS = 16;
export const RENDER_PRESETS: Record<string, { scale: number; samples: number }> = { print: { scale: 1, samples: 24 }, draft: { scale: 0.5, samples: 8 } };

export function defaultSettings(): SheetSettings {
  return {
    format: SETTINGS_FORMAT, style: 'textured', background: 'paper', name: '', title: '',
    panels: {
      turnaround: { on: true, style: null, pose: { pose: TURNAROUND_POSE }, views: [...VIEW_ORDER] },
      expressions: { on: true, style: null, pose: { pose: TURNAROUND_POSE }, view: 'front', framing: 'face', items: [...DEFAULT_EXPRESSIONS] },
      poses: { on: true, style: null, view: 'three', slots: DEFAULT_POSES.map((pose) => ({ pose })) },
      info: { on: true },
    },
    render: { ...RENDER_PRESETS.print },
  };
}

const isObj = (v: any) => !!v && typeof v === 'object' && !Array.isArray(v);
const str = (v: any, def: string, max = 200) => (typeof v === 'string' ? v.slice(0, max) : def);
const bool = (v: any, def: boolean) => (typeof v === 'boolean' ? v : def);
const num = (v: any, def: number, lo: number, hi: number) => (typeof v === 'number' && isFinite(v) ? Math.min(hi, Math.max(lo, v)) : def);
const view = (v: any, def: ViewName): ViewName => (typeof v === 'string' && v in VIEWS ? v as ViewName : def);

export function normalisePose(v: any, def: PoseSlot): PoseSlot {
  if (typeof v === 'string' && v) return { pose: v };
  if (!isObj(v) || typeof v.pose !== 'string' || !v.pose) return { ...def };
  const out: PoseSlot = { pose: v.pose.slice(0, 80) };
  if (typeof v.label === 'string' && v.label) out.label = v.label.slice(0, 80);
  if (Array.isArray(v.q) && v.q.length && v.q.length % 4 === 0 && v.q.every((x: any) => typeof x === 'number' && isFinite(x))) {
    out.q = v.q.map((x: number) => Math.round(x * 1e5) / 1e5);
  }
  return out;
}

// a full settings object from any input: missing fields take the defaults, unknown fields and bad values drop out
export function normaliseSettings(input: any, styles: string[] = ['textured']): SheetSettings {
  const d = defaultSettings();
  const s = isObj(input) ? input : {};
  const p = isObj(s.panels) ? s.panels : {};
  const style = (v: any) => (typeof v === 'string' && styles.includes(v) ? v : null);
  const T = isObj(p.turnaround) ? p.turnaround : {}, E = isObj(p.expressions) ? p.expressions : {}, P = isObj(p.poses) ? p.poses : {}, I = isObj(p.info) ? p.info : {};
  const views = Array.isArray(T.views) ? VIEW_ORDER.filter((v) => T.views.includes(v)) : d.panels.turnaround.views;
  const items = Array.isArray(E.items)
    ? Array.from(new Set(E.items.filter((x: any) => typeof x === 'string' && x).map((x: string) => x.slice(0, 60)))).slice(0, MAX_EXPRESSIONS) as string[]
    : d.panels.expressions.items;
  const slots = Array.isArray(P.slots) ? P.slots.map((x: any) => normalisePose(x, { pose: '' })).filter((x: PoseSlot) => x.pose).slice(0, MAX_POSES) : d.panels.poses.slots;
  const R = isObj(s.render) ? s.render : {};
  return {
    format: SETTINGS_FORMAT,
    style: style(s.style) || d.style,
    background: (['paper', 'dark', 'transparent'] as const).includes(s.background) ? s.background : d.background,
    name: str(s.name, ''),
    title: str(s.title, ''),
    panels: {
      turnaround: { on: bool(T.on, true), style: style(T.style), pose: normalisePose(T.pose, d.panels.turnaround.pose), views },
      expressions: { on: bool(E.on, true), style: style(E.style), pose: normalisePose(E.pose, d.panels.expressions.pose), view: view(E.view, 'front'),
        framing: E.framing === 'shoulders' ? 'shoulders' : 'face', items },
      poses: { on: bool(P.on, true), style: style(P.style), view: view(P.view, 'three'), slots },
      info: { on: bool(I.on, true) },
    },
    render: { scale: num(R.scale, d.render.scale, 0.1, 2), samples: Math.round(num(R.samples, d.render.samples, 1, 128)) },
  };
}

// the panels that the settings show: a panel without data stays out
export function shownPanels(s: SheetSettings): PanelId[] {
  const P = s.panels;
  return PANEL_ORDER.filter((id) => {
    if (!P[id].on) return false;
    if (id === 'turnaround') return P.turnaround.views.length > 0;
    if (id === 'expressions') return P.expressions.items.length > 0;
    if (id === 'poses') return P.poses.slots.length > 0;
    return true;
  });
}

// the style of a panel: its own, else the sheet's
export function panelStyle(s: SheetSettings, id: PanelId): string {
  const p: any = s.panels[id];
  return (p && p.style) || s.style;
}

// a stable text of the settings: equal settings give equal keys (the Sheet section knows when its sheet is stale)
export function settingsKey(s: SheetSettings): string { return JSON.stringify(s); }

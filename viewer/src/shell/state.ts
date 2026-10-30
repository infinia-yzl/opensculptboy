// OpenSculptBoy
// Apache License, Version 2.0
//
// The state of the interface and the bus between the interface and the renderer. The store holds what the person
// chose (the mode, its section, the open panels, the theme); it changes on clicks and keys only, never on a frame.
// The bus carries the renderer's news (the look changed, the motion changed, a tenth of a second passed): components
// that follow the look while a slider moves paint themselves from the bus through refs, so React renders nothing on
// a frame of the renderer.

import { MODES, SECTIONS, TOOLS, type ModeId } from '../ui/shortcuts.ts';
import { saveUi, uiState } from '../ui/dom.ts';
import type { StarterNote } from '../starters.ts';

export type Theme = 'light' | 'dark' | 'system';
export type ToolId = 'rig' | 'scene' | 'settings';

export interface OpenError { title: string; body: string; fit?: boolean; action?: { label: string; run: () => void } }
// the progress card of a long step (a card's open): its title and its steps, each done, now or to do
export interface Progress { title: string; steps: [string, 'done' | 'now' | 'todo'][] }

export interface UIState {
  mode: ModeId;
  open: boolean;                       // the mode's panel
  sub: Record<ModeId, string>;         // the section shown in each mode
  tool: ToolId | null;                 // the tool panel on the right (View menu)
  theme: Theme;
  dark: boolean;                       // the theme in use
  width: 'phone' | 'medium' | 'wide';
  snap: 'half' | 'full';               // the phone's bottom sheet
  welcome: boolean;                    // the welcome card of the first visit
  openDialog: boolean;
  openBusy: string;
  openError: OpenError | null;
  newDialog: boolean;                  // the New character screen
  // the statue whose package the page fetches, and the note of the last statue that did not open (starters.ts)
  starter: { busy: string; note: StarterNote | null };
  character: boolean;                  // the Character panel: what the character holds
  progress: Progress | null;
  stats: boolean;
  history: { undo: string | null; redo: string | null };
  view: { frame: string; angle: string; turntable: boolean };
  identity: { name: string; kind: string };
  dragging: boolean;                   // files over the page
  hasClips: boolean;
}

type Listener = () => void;
export interface Store<T> { get(): T; set(patch: Partial<T> | ((s: T) => Partial<T>)): void; subscribe(l: Listener): () => void }

function same(a: any, b: any): boolean {
  if (a === b) return true;
  if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => same(a[k], b[k]));
}

export function createStore<T extends object>(init: T): Store<T> {
  let s = init;
  const ls = new Set<Listener>();
  return {
    get: () => s,
    set(patch) {
      const p = typeof patch === 'function' ? patch(s) : patch;
      let changed = false;
      const next: any = { ...s };
      for (const [k, v] of Object.entries(p)) if (!same((s as any)[k], v)) { next[k] = v; changed = true; }
      if (!changed) return;
      s = next;
      for (const l of ls) l();
    },
    subscribe(l) { ls.add(l); return () => { ls.delete(l); }; },
  };
}

// ------------------------------------------------------------------ the bus
export type BusEvent = 'look' | 'motion' | 'hair' | 'face' | 'tick' | 'wardrobe';
const handlers: Record<string, Set<() => void>> = {};
export const bus = {
  on(ev: BusEvent, fn: () => void) { (handlers[ev] ||= new Set()).add(fn); return () => { handlers[ev].delete(fn); }; },
  emit(ev: BusEvent) { const s = handlers[ev]; if (s) for (const fn of s) fn(); },
};

// ------------------------------------------------------------------ the store of the page
export const widthOf = (w: number): UIState['width'] => (w < 720 ? 'phone' : w < 1100 ? 'medium' : 'wide');
const saved = uiState().shell || {};
const subs = Object.fromEntries(MODES.map((m) => {
  const list = SECTIONS[m.id], want = saved.sub?.[m.id];
  return [m.id, list.some((s) => s.id === want) ? want : (m.id === 'create' ? 'body' : list[0].id)];
})) as Record<ModeId, string>;

export const ui = createStore<UIState>({
  mode: MODES.some((m) => m.id === saved.mode) ? saved.mode : 'create',
  open: saved.open ?? (typeof innerWidth === 'number' ? innerWidth >= 720 : true),
  sub: subs,
  tool: null,
  theme: ['light', 'dark', 'system'].includes(saved.theme) ? saved.theme : 'system',
  dark: false,
  width: typeof innerWidth === 'number' ? widthOf(innerWidth) : 'wide',
  snap: 'half',
  welcome: !uiState().welcomed,
  openDialog: false, openBusy: '', openError: null,
  newDialog: false, starter: { busy: '', note: null }, character: !!saved.character, progress: null,
  stats: false,
  history: { undo: null, redo: null },
  view: { frame: '', angle: '', turntable: false },
  identity: { name: '', kind: '' },
  dragging: false,
  hasClips: true,
});
// the choices that a later visit starts from
ui.subscribe(() => {
  const s = ui.get();
  const want = { mode: s.mode, open: s.open, sub: s.sub, theme: s.theme, character: s.character };
  if (!same(uiState().shell, want)) saveUi({ shell: want });
});

// ------------------------------------------------------------------ the actions
export const actions = {
  // a mode's key or button: the mode opens, or its panel closes when it shows
  toggleMode(mode: ModeId) {
    const s = ui.get();
    if (s.mode === mode && s.open) ui.set({ open: false });
    else ui.set({ mode, open: true });
  },
  openMode(mode: ModeId, sub?: string) {
    const s = ui.get();
    ui.set({ mode, open: true, sub: sub ? { ...s.sub, [mode]: sub } : s.sub });
  },
  // a section's key: it opens in its mode, or the panel closes when the section shows
  toggleSection(mode: ModeId, sub: string) {
    const s = ui.get();
    if (s.mode === mode && s.open && s.sub[mode] === sub) ui.set({ open: false });
    else actions.openMode(mode, sub);
  },
  setSection(mode: ModeId, sub: string) { const s = ui.get(); ui.set({ sub: { ...s.sub, [mode]: sub } }); },
  closePanel() { ui.set({ open: false }); },
  toggleTool(tool: ToolId) {
    const s = ui.get();
    // on a phone or a medium screen the tool takes the place of the mode's panel
    if (s.tool === tool) ui.set({ tool: null });
    else ui.set({ tool, ...(s.width !== 'wide' ? { open: false } : {}) });
  },
  closeTool() { ui.set({ tool: null }); },
  setTheme(theme: Theme) { ui.set({ theme }); },
  dismissWelcome() { ui.set({ welcome: false }); saveUi({ welcomed: 1 }); },
  showOpen(error: OpenError | null = null) { ui.set({ openDialog: true, openError: error, newDialog: false }); },
  hideOpen() { ui.set({ openDialog: false, openError: null, openBusy: '' }); },
  showNew() { ui.set({ newDialog: true, openDialog: false }); },
  hideNew() { ui.set({ newDialog: false }); },
  setStarter(starter: UIState['starter']) { ui.set({ starter }); },
  toggleCharacter() { ui.set({ character: !ui.get().character }); },
};

export const toolOf = (id: string) => TOOLS.find((t) => t.id === id);

// OpenSculptBoy
// Apache License, Version 2.0
//
// The studio's actions: one registry for every key, the help (F1), the command list (F3), the tooltips, the key caps
// and the Ctrl overlay (DESIGN-0014, key map version 2). A part of the page registers each of its actions once, with
// its keys, its group in the help, its tier (how often it is used) and its scope (where its keys are live); the page's
// single key handler (`dispatchKey`) finds the action of a key press here, so a key that the help lists works, and a
// key that works is listed.
//
// Scopes nest: `global` holds the keys of every mode, a mode's scope (`create`, `pose`, ...) holds that mode's keys,
// and deeper scopes (`pose-edit`, `pose-select`, `drag`) hold the keys of a state inside it. A key press tries the
// live actions of that key from the deepest live scope outwards; an action whose `run` returns false passes the key
// on (Esc cancels a drag, else clears the selection, else closes the panel). Two actions may share a key only in
// different scopes; `conflicts()` lists any pair that shares a key in one scope, and the node test fails on it.

export type Tier = 'constant' | 'frequent' | 'occasional' | 'rare';
export const TIERS: { id: Tier; label: string; note: string }[] = [
  { id: 'constant', label: 'Constant', note: 'Every few seconds' },
  { id: 'frequent', label: 'Frequent', note: 'Every minute or so' },
  { id: 'occasional', label: 'Occasional', note: 'A few times a session' },
  { id: 'rare', label: 'Rare', note: 'Once a session or less' },
];

// what a key press hands to an action
export interface RunCtx {
  // the key press, or null when the action runs from a button, the command list or a menu
  event: KeyboardEvent | null;
  // the chord that matched ('Shift+ArrowLeft'), or '' without a key
  chord: string;
  shift: boolean;
  ctrl: boolean;
  alt: boolean;
}

export interface ActionDef {
  id: string;                       // 'view.frame', 'pose.tool.move'
  label: string;                    // 'Frame the selection': the help, the command list and the tooltip's title
  desc?: string;                    // the tooltip's second line: what the control does
  keys?: string[];                  // chords: 'F', 'Shift+1', 'Ctrl+Shift+E', 'Alt+Z', '`', 'Numpad1', 'PageDown'
  mouse?: string[];                 // the mouse inputs for the help ('Left-drag', 'Alt+Left-drag'); never dispatched
  group: string;                    // the group in the help ('The view', 'Points', 'Modes')
  tier: Tier;
  scope?: string;                   // where the keys are live: 'global' (the default), a mode, or a deeper scope
  mode?: string;                    // the mode tab of the help that lists it (default: the scope's mode, else every mode)
  when?: () => boolean;             // live now (default: always, within its scope)
  run?: (ctx: RunCtx) => boolean | void;   // false: nothing to act on, the key passes on
  place?: string;                   // where the control lives, for the command list ('The view', 'Pose › Edit')
  icon?: string;                    // the icon of the command list
  hint?: string;                    // the toast when the key has nothing to act on ('Select a hand first')
  repeat?: boolean;                 // a held key repeats the action (arrows, nudges)
  row?: string;                     // actions that share a row of the help ('pose.hier': [ and ] in one row)
  rowLabel?: string;                // that row's text ('Parent joint, child joint')
  help?: boolean;                   // listed in the help (default true)
  command?: boolean;                // listed in the command list (default: whenever it has a run)
  inText?: boolean;                 // live while a text field has the focus (Esc, F1, Ctrl+S)
  inDialog?: boolean;               // live while a dialog is open
  menu?: string;                    // the ring menu that offers it ('view' for the backtick, 'display' for Z)
  menuOrder?: number;               // its place in that menu, clockwise from the top
  menuLabel?: string;               // its shorter name there ('Full figure')
  checked?: () => boolean;          // a switch or a choice that is on now (a tick in the menu, aria-checked)
}

export interface ScopeDef {
  id: string;
  parent?: string;                  // default 'global'
  active: () => boolean;
  rank?: number;                    // order of the search; default: 10 per level of nesting
  label?: string;                   // the help's tab for a mode scope
}

// ------------------------------------------------------------------ the registry
const ACTIONS = new Map<string, ActionDef>();
const SCOPES = new Map<string, ScopeDef>();
const listeners = new Set<() => void>();
let version = 0;
const changed = () => { version++; for (const l of listeners) l(); };

SCOPES.set('global', { id: 'global', active: () => true, rank: 0 });

export function defineScope(s: ScopeDef): () => void {
  SCOPES.set(s.id, { parent: 'global', ...s });
  changed();
  return () => { if (SCOPES.get(s.id)?.active === s.active) { SCOPES.delete(s.id); changed(); } };
}
export function scopeOf(id: string): ScopeDef | undefined { return SCOPES.get(id); }

// register an action (a later registration with the same id replaces it); returns its removal
export function registerAction(a: ActionDef): () => void {
  const def: ActionDef = { scope: 'global', ...a, keys: (a.keys || []).map(normChord) };
  ACTIONS.set(a.id, def);
  changed();
  return () => { if (ACTIONS.get(a.id) === def) { ACTIONS.delete(a.id); changed(); } };
}
export function registerActions(list: ActionDef[]): () => void {
  const offs = list.map(registerAction);
  return () => offs.forEach((f) => f());
}
export function action(id: string): ActionDef | undefined { return ACTIONS.get(id); }
export function allActions(): ActionDef[] { return [...ACTIONS.values()]; }
export function subscribeActions(fn: () => void): () => void { listeners.add(fn); return () => { listeners.delete(fn); }; }
export function actionsVersion(): number { return version; }

// the depth of a scope: its rank, else ten per level below global
function rankOf(id: string): number {
  const s = SCOPES.get(id);
  if (!s) return -1;
  if (s.rank !== undefined) return s.rank;
  let d = 0, p: ScopeDef | undefined = s;
  while (p && p.id !== 'global' && d < 20) { d++; p = SCOPES.get(p.parent || 'global'); }
  return d * 10;
}
// a scope is live when it and every scope above it are active
export function scopeLive(id: string): boolean {
  let s = SCOPES.get(id), n = 0;
  while (s && n++ < 20) {
    if (!s.active()) return false;
    if (s.id === 'global') return true;
    s = SCOPES.get(s.parent || 'global');
  }
  return false;
}
// the mode a scope belongs to: the scope under global on its chain
export function modeOfScope(id: string): string | null {
  let s = SCOPES.get(id), prev: ScopeDef | undefined;
  while (s && s.id !== 'global') { prev = s; s = SCOPES.get(s.parent || 'global'); }
  return prev && prev.label ? prev.id : null;
}
export function liveAction(a: ActionDef): boolean {
  return scopeLive(a.scope || 'global') && (!a.when || !!a.when());
}

// listeners of every action that runs (the tour ticks its steps from them)
const runListeners = new Set<(id: string) => void>();
export function onActionRun(fn: (id: string) => void): () => void { runListeners.add(fn); return () => { runListeners.delete(fn); }; }
const ran = (id: string) => { for (const f of runListeners) f(id); };

// run an action by its id (a button, the command list, a ring menu); false when it is not live or did nothing
export function runAction(id: string, ctx: Partial<RunCtx> = {}): boolean {
  const a = ACTIONS.get(id);
  if (!a || !a.run) return false;
  const r = a.run({ event: null, chord: '', shift: false, ctrl: false, alt: false, ...ctx });
  if (r !== false) ran(id);
  return r !== false;
}

// ------------------------------------------------------------------ chords
const MODS = ['Ctrl', 'Alt', 'Shift'] as const;
// a chord in one spelling: the modifiers in the order Ctrl, Alt, Shift, then the key ('Ctrl+Shift+E')
export function normChord(c: string): string {
  const parts = c.split('+');
  // a chord of the plus key itself ('Ctrl++', '+')
  let key = parts.pop()!;
  if (key === '' && parts.length && parts[parts.length - 1] === '') { parts.pop(); key = '+'; }
  const name: Record<string, string> = { cmd: 'Ctrl', meta: 'Ctrl', control: 'Ctrl', ctrl: 'Ctrl', alt: 'Alt', option: 'Alt', shift: 'Shift' };
  const mods = new Set(parts.map((p) => name[p.toLowerCase()] || p));
  if (key.length === 1 && /[a-z]/.test(key)) key = key.toUpperCase();
  return [...MODS.filter((m) => mods.has(m)), key].join('+');
}
export function chordMods(c: string): { ctrl: boolean; alt: boolean; shift: boolean; key: string } {
  const p = normChord(c).split('+');
  const key = p.pop() || '+';
  return { ctrl: p.includes('Ctrl'), alt: p.includes('Alt'), shift: p.includes('Shift'), key: key || '+' };
}

const CODE_TOKEN: Record<string, string> = {
  Backquote: '`', Minus: '-', Equal: '=', BracketLeft: '[', BracketRight: ']', Backslash: '\\', Semicolon: ';',
  Quote: "'", Comma: ',', Period: '.', Slash: '/', Space: 'Space', IntlBackslash: '\\',
  NumpadDecimal: 'Numpad.', NumpadAdd: '+', NumpadSubtract: '-', NumpadMultiply: '*', NumpadDivide: 'Numpad/', NumpadEnter: 'Enter',
};
const NAMED = new Set(['Escape', 'Enter', 'Tab', 'Backspace', 'Delete', 'Insert', 'Home', 'End', 'PageUp', 'PageDown',
  'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'ContextMenu']);
export const isMac = () => typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || '');

export interface KeyLike { key: string; code?: string; ctrlKey?: boolean; metaKey?: boolean; altKey?: boolean; shiftKey?: boolean }
// the chords that a key press can mean, in order: the key at its place (a digit, the backquote, a bracket and the
// numpad by their place on the keyboard, so that Shift+1 is Shift+1 on every layout; a Latin letter as the layout types
// it, so that AZERTY's Z keeps its meaning, else by its place for a keyboard in another script), then the character it
// types with Shift taken into it ('?' for Shift+/ on a US keyboard, '+' for Shift+=)
export function chordsOf(e: KeyLike, mac = isMac()): string[] {
  const ctrl = !!(e.ctrlKey || (mac && e.metaKey)), alt = !!e.altKey, shift = !!e.shiftKey;
  const k = e.key || '', code = e.code || '';
  const join = (key: string, sh = shift) => [ctrl && 'Ctrl', alt && 'Alt', sh && 'Shift', key].filter(Boolean).join('+');
  const out: string[] = [];
  const ascii = k.length === 1 && k.charCodeAt(0) > 0x20 && k.charCodeAt(0) < 0x7f;
  let m: RegExpExecArray | null;
  if (ascii && /[a-z]/i.test(k)) out.push(join(k.toUpperCase()));
  else if ((m = /^Key([A-Z])$/.exec(code))) out.push(join(m[1]));
  else if ((m = /^Digit([0-9])$/.exec(code))) out.push(join(m[1]));
  else if ((m = /^Numpad([0-9])$/.exec(code))) out.push(join('Numpad' + m[1]));
  else if ((m = /^F([0-9]{1,2})$/.exec(code)) || (m = /^F([0-9]{1,2})$/.exec(k))) out.push(join('F' + m[1]));
  else if (CODE_TOKEN[code]) out.push(join(CODE_TOKEN[code]));
  else if (NAMED.has(k)) out.push(join(k));
  else if (k === ' ') out.push(join('Space'));
  // the character typed, with Shift in it
  if (ascii && !/[a-z0-9]/i.test(k)) { const c = join(k, false); if (!out.includes(c)) out.push(c); }
  if (NAMED.has(k) && !out.includes(join(k))) out.push(join(k));
  return out;
}

// a chord as key caps: ['Ctrl', 'Shift', 'E'], with Cmd on a Mac and short names for the long keys
const CAP: Record<string, string> = {
  Escape: 'Esc', PageUp: 'PgUp', PageDown: 'PgDn', ArrowLeft: '←', ArrowRight: '→', ArrowUp: '↑', ArrowDown: '↓',
  Delete: 'Del', 'Numpad.': 'Num .', 'Numpad/': 'Num /',
};
export function capsOf(chord: string, mac = isMac()): string[] {
  const { ctrl, alt, shift, key } = chordMods(chord);
  const k = CAP[key] ?? (/^Numpad([0-9])$/.test(key) ? 'Num ' + key.slice(6) : key);
  return [ctrl && (mac ? 'Cmd' : 'Ctrl'), alt && (mac ? 'Option' : 'Alt'), shift && 'Shift', k].filter(Boolean) as string[];
}
// the first key of an action as caps, for a tooltip or a button's hint ([] without a key)
export function capsFor(id: string): string[] {
  const a = ACTIONS.get(id);
  return a?.keys?.length ? capsOf(a.keys[0]) : [];
}
// the keys of an action as one line of text ('Shift 4', 'F3 or Ctrl K')
export function keyLine(id: string): string {
  const a = ACTIONS.get(id);
  return (a?.keys || []).map((k) => capsOf(k).join(' ')).join(' or ');
}

// ------------------------------------------------------------------ the browser's keys
// keys a page cannot take in Chrome, and keys the studio leaves to the browser so that nothing surprising happens
// (DESIGN-0014, "Keys the browser keeps"); no action may use them
export const BROWSER_KEPT = ['Ctrl+N', 'Ctrl+T', 'Ctrl+W', 'Ctrl+Tab', 'Ctrl+Shift+Tab', 'Ctrl+Shift+N', 'Ctrl+Shift+T', 'Ctrl+Shift+W',
  ...'123456789'.split('').map((d) => 'Ctrl+' + d), 'Alt+F4'];
export const BROWSER_LEFT = ['F5', 'Ctrl+F5', 'Shift+F5', 'F11', 'F12', 'Ctrl+L', 'Ctrl+D', 'Ctrl+F', 'Ctrl+P', 'Ctrl+R', 'Ctrl+Shift+R',
  'Ctrl+Shift+I', 'Ctrl+Shift+J', 'Ctrl+U', 'Alt+D', 'Alt+E', 'Alt+F', 'Alt+ArrowLeft', 'Alt+ArrowRight', 'Ctrl+=', 'Ctrl+-', 'Ctrl+0',
  'Ctrl++'].map(normChord);
export const RESERVED = new Set([...BROWSER_KEPT.map(normChord), ...BROWSER_LEFT]);

// ------------------------------------------------------------------ the checks of the node test
export interface Conflict { chord: string; scope: string; ids: string[] }
// pairs of actions that share a key in one scope, and actions on a key the browser keeps
export function conflicts(): Conflict[] {
  const by = new Map<string, string[]>();
  for (const a of ACTIONS.values()) for (const k of a.keys || []) {
    const key = (a.scope || 'global') + '\u0000' + k;
    (by.get(key) || by.set(key, []).get(key)!).push(a.id);
  }
  const out: Conflict[] = [];
  for (const [key, ids] of by) if (ids.length > 1) { const [scope, chord] = key.split('\u0000'); out.push({ chord, scope, ids }); }
  return out;
}
export function reservedUses(): { id: string; chord: string }[] {
  const out: { id: string; chord: string }[] = [];
  for (const a of ACTIONS.values()) for (const k of a.keys || []) if (RESERVED.has(k)) out.push({ id: a.id, chord: k });
  return out;
}

// ------------------------------------------------------------------ the key handler
export interface DispatchOpts {
  typing?: boolean;                 // a text field has the focus
  dialog?: boolean;                 // a dialog or menu is open
  hint?: (text: string) => void;    // shows an action's hint when it had nothing to act on
}
// the action a key press runs: the live actions of its chords, deepest scope first; returns the action's id, or null
// when no action took the key (the page then leaves the key to the browser)
export function dispatchKey(e: KeyboardEvent, o: DispatchOpts = {}): string | null {
  const chords = chordsOf(e);
  if (!chords.length) return null;
  const cands: { a: ActionDef; chord: string; rank: number; order: number }[] = [];
  chords.forEach((chord, order) => {
    for (const a of ACTIONS.values()) {
      if (!a.run || !a.keys?.includes(chord)) continue;
      if (o.typing && !a.inText) continue;
      if (o.dialog && !a.inDialog) continue;
      if (e.repeat && !a.repeat) continue;
      if (!scopeLive(a.scope || 'global')) continue;
      cands.push({ a, chord, rank: rankOf(a.scope || 'global'), order });
    }
  });
  cands.sort((x, y) => y.rank - x.rank || x.order - y.order);
  let hinted: ActionDef | null = null;
  for (const c of cands) {
    if (c.a.when && !c.a.when()) { if (c.a.hint && !hinted) hinted = c.a; continue; }
    const ctx: RunCtx = { event: e, chord: c.chord, shift: e.shiftKey, ctrl: e.ctrlKey || (isMac() && e.metaKey), alt: e.altKey };
    if (c.a.run!(ctx) === false) { if (c.a.hint && !hinted) hinted = c.a; continue; }
    ran(c.a.id);
    return c.a.id;
  }
  if (hinted && o.hint) { o.hint(hinted.hint!); return hinted.id; }
  return null;
}

// ------------------------------------------------------------------ the help, the command list, the Ctrl overlay
export interface HelpRow { id: string; keys: string[]; mouse: string[]; label: string; group: string; tier: Tier; mode: string | null; ids: string[] }
// the text of an action's row in the help: the row's own text, given on any of its actions
export function rowLabelOf(a: ActionDef): string {
  if (!a.row) return a.label;
  for (const b of ACTIONS.values()) if (b.row === a.row && b.rowLabel) return b.rowLabel;
  return a.label;
}
// the help's rows: one per action, or one per shared row
export function helpRows(): HelpRow[] {
  const rows = new Map<string, HelpRow>();
  for (const a of ACTIONS.values()) {
    if (a.help === false) continue;
    const id = a.row || a.id;
    const mode = a.mode ?? modeOfScope(a.scope || 'global');
    const r = rows.get(id);
    if (r) { r.keys.push(...(a.keys || [])); r.mouse.push(...(a.mouse || [])); r.ids.push(a.id); continue; }
    rows.set(id, { id, keys: [...(a.keys || [])], mouse: [...(a.mouse || [])], label: rowLabelOf(a), group: a.group, tier: a.tier, mode, ids: [a.id] });
  }
  return [...rows.values()];
}
// the command list: every action with a run, by name, with its key and its place; live ones first
export function commands(query = ''): { a: ActionDef; live: boolean; score: number }[] {
  const q = query.trim().toLowerCase();
  const out: { a: ActionDef; live: boolean; score: number }[] = [];
  for (const a of ACTIONS.values()) {
    if (!a.run || a.command === false) continue;
    const text = (a.label + ' ' + (a.place || '') + ' ' + (a.desc || '') + ' ' + (a.keys || []).join(' ')).toLowerCase();
    let score = 0;
    if (q) {
      const words = q.split(/\s+/);
      if (!words.every((w) => text.includes(w))) continue;
      const l = a.label.toLowerCase();
      score = (l.startsWith(q) ? 4 : 0) + (l.includes(q) ? 2 : 0) + (a.keys?.length ? 1 : 0);
    }
    out.push({ a, live: liveAction(a), score });
  }
  out.sort((x, y) => Number(y.live) - Number(x.live) || y.score - x.score || x.a.label.localeCompare(y.a.label));
  return out;
}
// the items of a ring menu, in their order
export function menuItems(menu: string): ActionDef[] {
  return [...ACTIONS.values()].filter((a) => a.menu === menu && a.run && liveAction(a))
    .sort((x, y) => (x.menuOrder ?? 99) - (y.menuOrder ?? 99));
}
// the live actions that a held modifier starts ('Ctrl': Ctrl+Z, Ctrl+S, ...), for the overlay of a held key
export function modifierActions(mod: 'Ctrl' | 'Shift' | 'Alt'): { a: ActionDef; chord: string }[] {
  const out: { a: ActionDef; chord: string }[] = [];
  for (const a of ACTIONS.values()) {
    if (!liveAction(a)) continue;
    const k = (a.keys || []).find((c) => c.split('+').slice(0, -1).includes(mod));
    if (k) out.push({ a, chord: k });
  }
  return out;
}

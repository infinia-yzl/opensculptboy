// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The keyboard shortcuts of the page, from one table: the modes and their sections, the tools of the View menu, and
// every other key. The key handler (index.ts), the tooltips and key caps of the interface (src/shell) and the help
// sheet all read this table, so a key that the help lists works, and a key that works is listed.

import { h, iconButton } from './dom.ts';

export type ModeId = 'create' | 'dress' | 'express' | 'pose' | 'sheet' | 'export';
export interface ModeDef { id: ModeId; label: string; key: string; icon: string; tip: string }
export interface SectionDef { id: string; label: string; short?: string; key?: string; icon: string }

// the six modes of the top bar, in order
export const MODES: ModeDef[] = [
  { id: 'create', label: 'Create', key: 'c', icon: 'body', tip: 'Make the character: body, face, hair, skin and eyes' },
  { id: 'dress', label: 'Dress', key: 'd', icon: 'wear', tip: 'Dress the character' },
  { id: 'express', label: 'Express', key: 'x', icon: 'expression', tip: 'Give the character expressions' },
  { id: 'pose', label: 'Pose', key: 'p', icon: 'pose', tip: 'Pose the character' },
  { id: 'sheet', label: 'Sheet', key: 's', icon: 'sheet', tip: 'Make a character sheet' },
  { id: 'export', label: 'Export', key: 'e', icon: 'export', tip: 'Save the character and its pictures' },
];
// the sections of each mode; Create's are its sub-tabs, each with its key (Skin and eyes takes K, since E is Export)
export const SECTIONS: Record<ModeId, SectionDef[]> = {
  create: [
    { id: 'characters', label: 'Characters', short: 'Library', icon: 'characters' },
    { id: 'body', label: 'Body', key: 'b', icon: 'body' },
    { id: 'face', label: 'Face', key: 'f', icon: 'face' },
    { id: 'hair', label: 'Hair', key: 'h', icon: 'hair' },
    { id: 'skin', label: 'Skin and eyes', short: 'Skin', key: 'k', icon: 'skin' },
  ],
  dress: [{ id: 'wear', label: 'Wearables', icon: 'wear' }],
  express: [{ id: 'expression', label: 'Expression', icon: 'expression' }],
  pose: [{ id: 'pose', label: 'Pose', icon: 'pose' }],
  sheet: [{ id: 'sheet', label: 'Sheet', icon: 'sheet' }],
  export: [{ id: 'export', label: 'Export', icon: 'export' }],
};
// the tools of the View menu that open a panel on the right
export const TOOLS: SectionDef[] = [
  { id: 'rig', label: 'Rig', key: 'r', icon: 'bones' },
  { id: 'scene', label: 'Scene', key: 'l', icon: 'scene' },
];

// every key that opens a panel: a mode, a section of Create, or a tool
export const PANEL_KEYS: Record<string, string> = Object.fromEntries([
  ...MODES.map((m) => [m.id, m.key]),
  ...Object.values(SECTIONS).flat().filter((s) => s.key).map((s) => [s.id, s.key]),
  ...TOOLS.map((t) => [t.id, t.key]),
]);

// the help's list, by group: [keys as shown, what they do]; combinations are written with a space ("Ctrl Z")
export const SHORTCUTS: [string, [string, string][]][] = [
  ['Modes', [
    [MODES.map((m) => m.key.toUpperCase()).join(' '), MODES.map((m) => m.label).join(', ')],
    [SECTIONS.create.filter((s) => s.key).map((s) => s.key!.toUpperCase()).join(' '), 'Create: ' + SECTIONS.create.filter((s) => s.key).map((s) => s.label).join(', ')],
    [TOOLS.map((t) => t.key!.toUpperCase()).join(' '), TOOLS.map((t) => t.label).join(', ') + ' (View menu)'],
    ['`', 'Performance'],
    ['Esc', 'Close the panel or the dialog'],
  ]],
  ['Files', [
    ['Ctrl O', 'Open a character card, its folder, wearables or sheet settings'],
    ['Drop', 'Drop files anywhere on the page to open them'],
  ]],
  ['Camera', [
    ['Drag', 'Turn around the figure'],
    ['Right-drag, Shift drag, two fingers', 'Move the view'],
    ['Scroll, pinch', 'Zoom toward the pointer'],
    ['Double-click', 'Focus on the nearest joint'],
    ['← →', 'Turn'],
    ['↑ ↓', 'Move up or down the body'],
    ['Shift ↑ ↓', 'Tilt'],
    ['+ −', 'Zoom'],
    ['0', 'Reset the view'],
  ]],
  ['View', [
    ['1 2 3', 'Full body, upper body, face'],
    ['4 5 6 7', 'Front, three-quarter, side, back'],
    ['T', 'Turn the figure'],
    ['Shift H', 'Show or hide the hair'],
    ['Shift B', 'Show or hide the skeleton'],
    ['Shift W', 'Show or hide the skin weights'],
    ['Alt 1 to 4', 'Lighting presets'],
    ['Space', 'Play or pause the animation'],
  ]],
  ['Editing', [
    ['Ctrl Z', 'Undo'],
    ['Ctrl Shift Z, Ctrl Y', 'Redo'],
    ['Double-click a slider', 'Return it to its default'],
    ['Enter in a value', 'Apply a typed value'],
    ['?', 'This sheet'],
  ]],
];

// a key as the interface shows it: a letter in upper case, Ctrl as Cmd on a Mac
export const isMac = () => typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || '');
export function keyText(k: string): string {
  const t = k.length === 1 ? k.toUpperCase() : k;
  return isMac() ? t.replace(/\bCtrl\b/g, 'Cmd') : t;
}

// the keys that a keydown event can mean as a shortcut, in order: first its character (a letter in lower case without
// Shift and in upper case with it, so that Caps Lock leaves the panel keys alone), then the key's place on a US
// keyboard (e.code), for a keyboard in another script, an input method that reports 'Process', or a digit row that
// types symbols (AZERTY). The handler takes the first one that has an action.
const CODE_KEYS: Record<string, [string, string]> = {
  Backquote: ['`', '~'], Minus: ['-', '_'], Equal: ['=', '+'], Slash: ['/', '?'], Space: [' ', ' '],
  NumpadAdd: ['+', '+'], NumpadSubtract: ['-', '-'],
};
export function shortcutKeys(e: { key: string; code?: string; shiftKey?: boolean }): string[] {
  const k = e.key || '', code = e.code || '', shift = !!e.shiftKey;
  const out: string[] = [];
  const ascii = k.length === 1 && k.charCodeAt(0) >= 0x20 && k.charCodeAt(0) < 0x7f;
  if (ascii) out.push(/[a-z]/i.test(k) ? (shift ? k.toUpperCase() : k.toLowerCase()) : k);
  else if (k.length > 1 && k !== 'Process' && k !== 'Unidentified' && k !== 'Dead') out.push(k);
  // a Latin letter keeps its own meaning (on an AZERTY keyboard Z sits where a US keyboard has W)
  let m = /^Key([A-Z])$/.exec(code);
  if (m) { if (!(ascii && /[a-z]/i.test(k))) out.push(shift ? m[1] : m[1].toLowerCase()); }
  else if ((m = /^(?:Digit|Numpad)([0-9])$/.exec(code)) && !shift) out.push(m[1]);
  else if (CODE_KEYS[code]) out.push(CODE_KEYS[code][shift ? 1 : 0]);
  return out.filter((x, i) => out.indexOf(x) === i);
}

// the help sheet: the keys by group, with a search field
export function helpSheet() {
  const close = iconButton('close', 'Close', { key: 'Esc', tip: 'left' });
  const empty = h('p.osb-note.help-empty', { text: 'No key matches.', hidden: true });
  const groups = SHORTCUTS.map(([title, rows]) => {
    const items = rows.map(([k, v]) => {
      const dt = h('dt', {}, ...k.split(', ').map((x, i) => [i ? ', ' : '', ...x.split(' ').map((p, j) => [j ? ' ' : '', h('kbd.osb-key', { text: keyText(p) })]).flat()]).flat());
      const dd = h('dd', { text: v });
      return { dt, dd, text: (k + ' ' + v).toLowerCase() };
    });
    const sec = h('section', {}, h('h3.osb-eyebrow', { text: title }), h('dl', {}, ...items.flatMap((r) => [r.dt, r.dd])));
    return { sec, items };
  });
  const search = h('input.help-search', { type: 'search', placeholder: 'Search the keys', 'aria-label': 'Search the keys', autocomplete: 'off', spellcheck: 'false' }) as HTMLInputElement;
  search.addEventListener('input', () => {
    const q = search.value.trim().toLowerCase();
    let any = false;
    for (const g of groups) {
      let n = 0;
      for (const r of g.items) { const hit = !q || r.text.includes(q); r.dt.hidden = r.dd.hidden = !hit; if (hit) n++; }
      g.sec.hidden = n === 0; any = any || n > 0;
    }
    empty.hidden = any;
  });
  const dlg = h('dialog.help', { 'aria-labelledby': 'help-title' },
    h('header.help-head', {}, h('h2.osb-title', { id: 'help-title', text: 'Keys and help' }), close),
    h('p.help-lead', { text: 'OpenSculptBoy makes humanoid 3D characters on the Anny body model. Each mode holds one task; the keys below work whenever no text field has the focus.' }),
    h('div.help-tools', {}, search),
    h('div.help-grid', {}, ...groups.map((g) => g.sec)), empty,
  ) as HTMLDialogElement;
  close.addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });
  dlg.addEventListener('close', () => { search.value = ''; search.dispatchEvent(new Event('input')); });
  return dlg;
}

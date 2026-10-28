// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The keyboard shortcuts of the page and the help sheet that lists them.

import { h, iconButton } from './dom.ts';

// the key of each section of the two segments (index.ts): the help's Panels rows list the same letters
export const PANEL_KEYS: Record<string, string> = {
  characters: 'c', body: 'b', face: 'f', hair: 'h', skin: 'e', wear: 'w',
  pose: 'p', expression: 'x', rig: 'r', scene: 'l', sheet: 's',
};

export const SHORTCUTS: [string, [string, string][]][] = [
  ['Camera', [
    ['Drag', 'Orbit around the figure'],
    ['Right-drag, Shift+drag, two fingers', 'Move the view'],
    ['Scroll, pinch', 'Zoom toward the pointer'],
    ['Double-click', 'Focus on the nearest joint (the background reframes)'],
    ['← →', 'Orbit'],
    ['↑ ↓', 'Move up or down the body'],
    ['Shift+↑ ↓', 'Tilt'],
    ['+ −', 'Zoom'],
    ['0', 'Reset the view'],
  ]],
  ['View', [
    ['1 2 3', 'Full body, upper body, face'],
    ['4 5 6 7', 'Front, three-quarter, side, back'],
    ['T', 'Turntable'],
    ['Shift+H', 'Show or hide the hair'],
    ['Shift+B', 'Show or hide the skeleton'],
    ['Shift+W', 'Show or hide the skin weights'],
    ['Alt+1 to 4', 'Lighting presets'],
    ['Space', 'Play or pause the animation'],
  ]],
  ['Panels', [
    ['C B F H E W', 'Characters, Body, Face, Hair, Skin & eyes, Wearables'],
    ['P X R L S', 'Pose, Expression, Rig, Scene, Sheet'],
    ['`', 'Performance'],
    ['?', 'This sheet'],
    ['Esc', 'Close the panel or the sheet'],
  ]],
  ['Editing', [
    ['Ctrl+Z, ⌘Z', 'Undo'],
    ['Ctrl+Shift+Z, Ctrl+Y', 'Redo'],
    ['Double-click a slider', 'Return it to its default'],
    ['Enter in a value', 'Apply a typed value'],
  ]],
];

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

export function helpSheet() {
  const close = iconButton('close', 'Close', { key: 'Esc', tip: 'left' });
  const dlg = h('dialog.help', { 'aria-labelledby': 'help-title' },
    h('header.help-head', {}, h('h2', { id: 'help-title', text: 'Anny Viewer' }), close),
    h('p.help-lead', { text: 'OpenSculptBoy characters are humanoid 3D models built on the Anny body model. This page renders them live, with skin, eye and hair shading.' }),
    h('div.help-grid', {}, ...SHORTCUTS.map(([title, rows]) => h('section', {}, h('h3', { text: title }),
      h('dl', {}, ...rows.flatMap(([k, v]) => [h('dt', {}, ...k.split(', ').map((x, i) => [i ? ', ' : '', h('kbd', { text: x })]).flat()), h('dd', { text: v })]))))),
  ) as HTMLDialogElement;
  close.addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });
  return dlg;
}

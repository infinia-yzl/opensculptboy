// OpenSculptBoy
// Apache License, Version 2.0
//
// The key map version 2 without a browser (run by test/test_viewer_shortcuts.py):
//   node viewer/test/shortcuts.mjs
// It registers the studio's key map (src/studio/keymap.ts) and the posing editor's keys of DESIGN-0014's table, then
// checks: the chords that a keydown event means on other keyboards (Caps Lock, Shift, a keyboard in another script, an
// input method, AZERTY's digit row, the numpad, the backquote as a dead key); that no two actions share a key in one
// scope; that no action takes a key the browser keeps; that every key in the help has an action and every key of an
// action shows in the help or says why not; that a key reaches the deepest live scope first and falls through when its
// action has nothing to do. It prints a JSON summary; a failed check throws, so node exits with an error.
import assert from 'node:assert/strict';
import {
  allActions, chordsOf, conflicts, defineScope, dispatchKey, helpRows, normChord, registerAction, reservedUses, RESERVED, capsOf, commands,
  menuItems, modifierActions,
} from '../src/studio/actions.ts';
import { registerKeymap } from '../src/studio/keymap.ts';
import { ui } from '../src/shell/state.ts';
import { setSelection } from '../src/studio/selection.ts';
import { startModal, modalActive } from '../src/studio/modal.ts';

// ------------------------------------------------------------------ the chords of a key press
const cases = [
  [{ key: 'w', code: 'KeyW' }, ['W']],
  [{ key: 'W', code: 'KeyW' }, ['W']],                         // Caps Lock: the letter alone
  [{ key: 'W', code: 'KeyW', shiftKey: true }, ['Shift+W']],
  [{ key: 'ц', code: 'KeyW' }, ['W']],                         // a Cyrillic keyboard
  [{ key: 'Process', code: 'KeyX' }, ['X']],                   // an input method
  [{ key: 'z', code: 'KeyW' }, ['Z']],                         // AZERTY: Z keeps its own meaning
  [{ key: '&', code: 'Digit1' }, ['1', '&']],                  // AZERTY's digit row: the place first
  [{ key: '!', code: 'Digit1', shiftKey: true }, ['Shift+1', '!']],   // Shift+1 on a US keyboard
  [{ key: '1', code: 'Digit1', shiftKey: true }, ['Shift+1']],        // Shift+1 on AZERTY types the digit
  [{ key: '?', code: 'Slash', shiftKey: true }, ['Shift+/', '?']],
  [{ key: '+', code: 'Equal', shiftKey: true }, ['Shift+=', '+']],
  [{ key: 'Dead', code: 'Backquote' }, ['`']],                 // a dead key where US keyboards have the backquote
  [{ key: '`', code: 'Backquote' }, ['`']],
  [{ key: ' ', code: 'Space' }, ['Space']],
  [{ key: ' ', code: 'Space', ctrlKey: true }, ['Ctrl+Space']],
  [{ key: 'ArrowLeft', code: 'ArrowLeft', shiftKey: true }, ['Shift+ArrowLeft']],
  [{ key: 'Escape', code: 'Escape' }, ['Escape']],
  [{ key: 'End', code: 'Numpad1' }, ['Numpad1', 'End']],        // the numpad without Num Lock
  [{ key: '1', code: 'Numpad1', ctrlKey: true }, ['Ctrl+Numpad1']],
  [{ key: '.', code: 'NumpadDecimal' }, ['Numpad.', '.']],
  [{ key: 'E', code: 'KeyE', ctrlKey: true, shiftKey: true }, ['Ctrl+Shift+E']],
  [{ key: 'F3', code: 'F3' }, ['F3']],
  [{ key: '[', code: 'BracketLeft' }, ['[']],
  [{ key: '{', code: 'BracketLeft', shiftKey: true }, ['Shift+[', '{']],
  [{ key: 'PageDown', code: 'PageDown', altKey: true }, ['Alt+PageDown']],
  [{ key: 'Backspace', code: 'Backspace', shiftKey: true }, ['Shift+Backspace']],
];
for (const [e, want] of cases) assert.deepEqual(chordsOf(e, false), want, JSON.stringify(e));
assert.deepEqual(chordsOf({ key: 'z', code: 'KeyZ', metaKey: true }, true), ['Ctrl+Z']);   // Cmd on a Mac
assert.equal(normChord('shift+ctrl+e'), 'Ctrl+Shift+E');
assert.deepEqual(capsOf('Ctrl+Shift+E', false), ['Ctrl', 'Shift', 'E']);
assert.deepEqual(capsOf('PageDown', false), ['PgDn']);
assert.deepEqual(capsOf('Ctrl+Z', true), ['Cmd', 'Z']);

// ------------------------------------------------------------------ the key map, with the posing editor's keys
let ran = [];
const stubApp = {
  presets: { studio: { label: 'Studio' }, window: { label: 'Window' } },
  motion: () => ({ cur: null }), togglePlay() {}, turntable: () => false, setTurntable() {}, rigView: () => 'off', setRigView() {},
  hairVisible: () => true, setHairVisible() {}, applyPreset() {}, boneNames: () => ['root', 'wrist.R'], setParts: (h, f) => { ran.push('parts:' + JSON.stringify([h, f])); },
};
const stubNav = new Proxy({}, { get: (_, k) => (...a) => { ran.push('nav.' + String(k) + (a.length ? ':' + a.join(',') : '')); } });
registerKeymap({
  app: stubApp, nav: stubNav, undo: () => ran.push('undo'), redo: () => ran.push('redo'), saveCard: () => ran.push('save'), hosted: false,
  help: () => ran.push('help'), commands: () => ran.push('commands'), ringMenu: (m) => ran.push('ring:' + m), escape: () => false, toast: () => {},
});
// the posing editor's scopes and keys as DESIGN-0014's table gives them (stream P registers them in the page)
let editOn = false, selected = false;
defineScope({ id: 'pose-edit', parent: 'pose', active: () => editOn });
defineScope({ id: 'pose-select', parent: 'pose-edit', active: () => selected });
const P = [
  ['pose.edit', ['Tab'], 'pose'], ['pose.tool.select', ['Q']], ['pose.tool.move', ['W']], ['pose.tool.turn', ['E']], ['pose.grab', ['G']],
  ['pose.turn-keys', ['R']], ['pose.parent', ['[']], ['pose.child', [']']], ['pose.finger-prev', ['Shift+[']], ['pose.finger-next', ['Shift+]']],
  ['pose.select-all', ['A']], ['pose.select-none', ['Alt+A']], ['pose.reset-point', ['Alt+G', 'Shift+Alt+G']], ['pose.reset-turn', ['Alt+R', 'Shift+Alt+R']],
  ['pose.push', ['PageDown', 'Shift+PageDown', 'Alt+PageDown']], ['pose.pull', ['PageUp', 'Shift+PageUp', 'Alt+PageUp']], ['pose.mirror', ['M']],
  ['pose.copy', ['Ctrl+C']], ['pose.paste', ['Ctrl+V']], ['pose.paste-mirrored', ['Ctrl+Shift+V']], ['pose.scan', ['Shift+Z']],
  ['pose.tool.1', ['1']], ['pose.tool.2', ['2']], ['pose.tool.3', ['3']], ['pose.tool.4', ['4']], ['pose.tool.5', ['5']],
];
for (const [id, keys, scope] of P) registerAction({ id, label: id, keys, group: 'Points', tier: 'frequent', scope: scope || 'pose-edit', run: () => { ran.push(id); } });
registerAction({ id: 'pose.nudge', label: 'Nudge', keys: ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Shift+ArrowLeft', 'Shift+ArrowRight', 'Shift+ArrowUp', 'Shift+ArrowDown'],
  group: 'Points', tier: 'frequent', scope: 'pose-select', repeat: true, run: (c) => { ran.push('pose.nudge:' + c.chord); } });
registerAction({ id: 'pose.clear', label: 'Clear the selection', keys: ['Escape'], group: 'Points', tier: 'constant', scope: 'pose-select', run: () => { ran.push('pose.clear'); } });

// no two actions share a key in one scope, and none takes a key of the browser
assert.deepEqual(conflicts(), [], JSON.stringify(conflicts()));
assert.deepEqual(reservedUses(), []);
for (const k of ['Ctrl+N', 'Ctrl+T', 'Ctrl+W', 'Ctrl+Tab', 'Ctrl+1', 'Alt+F4', 'F5', 'F11', 'F12', 'Ctrl+L', 'Ctrl+D', 'Ctrl+F', 'Ctrl+P', 'Alt+D', 'Alt+E', 'Alt+F', 'Alt+ArrowLeft', 'Alt+ArrowRight']) {
  assert.ok(RESERVED.has(normChord(k)), k);
}
// a planted clash is found
const off = registerAction({ id: 'test.clash', label: 'clash', keys: ['F'], group: 'x', tier: 'rare', run: () => {} });
assert.equal(conflicts().length, 1);
off();
const off2 = registerAction({ id: 'test.reserved', label: 'reserved', keys: ['Ctrl+W'], group: 'x', tier: 'rare', run: () => {} });
assert.equal(reservedUses().length, 1);
off2();

// version 1's letters are gone: the modes, Create's sections, the framing digits, Rig and Scene
const keyed = new Map();
for (const a of allActions()) for (const k of a.keys || []) keyed.set((a.scope || 'global') + ' ' + k, a.id);
for (const k of ['C', 'D', 'X', 'P', 'S', 'B', 'K', 'L', '0', '6', '7']) assert.ok(!keyed.has('global ' + k), 'version 1 key ' + k);
assert.equal(keyed.get('global Shift+4'), 'mode.pose');
assert.equal(keyed.get('global Ctrl+Shift+E'), 'mode.export');
assert.equal(keyed.get('create 3'), 'create.skin');
assert.equal(keyed.get('global F'), 'view.frame');
assert.equal(keyed.get('global `'), 'view.menu');

// the help: every key of an action with a run shows in a row, and every row's key has an action
const rows = helpRows();
const inHelp = new Set(rows.flatMap((r) => r.ids));
for (const a of allActions()) {
  if (a.help === false) continue;
  assert.ok(inHelp.has(a.id), 'not in the help: ' + a.id);
}
for (const r of rows) for (const id of r.ids) assert.ok(allActions().some((a) => a.id === id));
const tiers = new Set(rows.map((r) => r.tier));
for (const t of ['constant', 'frequent', 'occasional', 'rare']) assert.ok(tiers.has(t), t);

// ------------------------------------------------------------------ the handler: the deepest live scope first
const press = (key, code, mods = {}) => {
  ran = [];
  const e = { key, code, ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, repeat: false, ...mods };
  const id = dispatchKey(e, {});
  return { id, ran };
};
ui.set({ mode: 'create' });
assert.equal(press('2', 'Digit2').id, 'create.face');
assert.equal(ui.get().sub.create, 'face');
assert.equal(press('!', 'Digit1', { shiftKey: true }).id, 'mode.create');
ui.set({ mode: 'pose' });
// outside Edit, the arrows turn the view; in Edit with a point selected, they nudge it
assert.deepEqual(press('ArrowLeft', 'ArrowLeft').ran, ['nav.orbit:-15']);
assert.deepEqual(press('ArrowLeft', 'ArrowLeft', { shiftKey: true }).ran, ['nav.orbit:-3']);
editOn = true;
assert.deepEqual(press('ArrowLeft', 'ArrowLeft').ran, ['nav.orbit:-15']);
assert.equal(press('w', 'KeyW').id, 'pose.tool.move');
assert.equal(press('1', 'Digit1').id, 'pose.tool.1');
selected = true;
assert.deepEqual(press('ArrowLeft', 'ArrowLeft').ran, ['pose.nudge:ArrowLeft']);
assert.equal(press('Escape', 'Escape').id, 'pose.clear');
selected = false;
// Esc with nothing to clear passes on to the panel's Esc, which has nothing to close: no action
assert.equal(press('Escape', 'Escape').id, null);
editOn = false;
assert.equal(press('w', 'KeyW').id, null);
assert.equal(press('f', 'KeyF').id, 'view.frame');
assert.equal(press('`', 'Backquote').id, 'view.menu');
assert.deepEqual(ran, ['ring:view']);
assert.equal(press('z', 'KeyZ', { altKey: true }).id, 'display.xray');
assert.equal(press('?', 'Slash', { shiftKey: true }).id, 'help.keys-q');
assert.equal(press('F1', 'F1').id, 'help.keys');
assert.equal(press('k', 'KeyK', { ctrlKey: true }).id, 'help.commands');
assert.equal(press('z', 'KeyZ', { ctrlKey: true }).id, 'edit.undo');
assert.equal(press('Z', 'KeyZ', { ctrlKey: true, shiftKey: true }).id, 'edit.redo');
// a held key's repeat runs only the actions that repeat
assert.equal(press('f', 'KeyF', { repeat: true }).id, null);
assert.equal(press('ArrowRight', 'ArrowRight', { repeat: true }).id, 'view.turn-right');
// typing in a field: only the actions marked for it
assert.equal(dispatchKey({ key: 'f', code: 'KeyF' }, { typing: true }), null);
assert.equal(dispatchKey({ key: 's', code: 'KeyS', ctrlKey: true }, { typing: true }), 'file.save');
let hint = '';
// H with nothing selected says so; with a selected part it hides that part's bones
assert.equal(dispatchKey({ key: 'h', code: 'KeyH' }, { hint: (t) => { hint = t; } }), 'part.hide');
assert.equal(hint, 'Select a part first');
setSelection({ label: 'Right hand', frame: () => null, bones: ['wrist.R'] });
assert.deepEqual(press('h', 'KeyH').ran, ['parts:[["wrist.R"],null]']);
assert.deepEqual(press('h', 'KeyH', { altKey: true }).ran, ['parts:[null,null]']);
assert.deepEqual(press('/', 'Slash').ran, ['parts:[null,["wrist.R"]]']);
setSelection(null);
// Space with no animation playing says so
assert.equal(dispatchKey({ key: ' ', code: 'Space' }, { hint: (t) => { hint = t; } }), 'motion.play');
assert.equal(hint, 'No animation is playing');

// ------------------------------------------------------------------ a modal grab takes the keys: an axis, a typed value, Enter
globalThis.addEventListener ??= () => {}; globalThis.removeEventListener ??= () => {};
const seen = [];
startModal({ kind: 'grab', label: 'Point 22', update: (v) => { seen.push([v.axis, v.local, v.typed]); return ''; }, confirm: () => seen.push('ok'), cancel: () => seen.push('back') }, { x: 0, y: 0 });
assert.ok(modalActive());
for (const [key, code] of [['x', 'KeyX'], ['x', 'KeyX'], ['3', 'Digit3'], ['.', 'Period'], ['5', 'Digit5']]) assert.ok(dispatchKey({ key, code }, { only: 'modal' }), key);
assert.equal(dispatchKey({ key: 'f', code: 'KeyF' }, { only: 'modal' }), null);   // F means nothing during a grab
assert.equal(dispatchKey({ key: 'Enter', code: 'Enter' }, { only: 'modal' }), 'modal.confirm');
assert.ok(!modalActive());
assert.deepEqual(seen.at(-1), 'ok');
assert.deepEqual(seen.at(-2), ['x', true, 3.5]);
assert.deepEqual(conflicts(), []);

// ------------------------------------------------------------------ menus, the command list and the Ctrl overlay
assert.deepEqual(menuItems('view').map((a) => a.menuLabel || a.label), ['Top', 'Face', 'Right', 'Right hand', 'Three-quarter', 'Front', 'Full figure', 'Left hand', 'Left', 'Back']);
assert.deepEqual(menuItems('display').map((a) => a.id), ['display.solid', 'display.xray', 'display.skeleton', 'display.weights', 'display.hair']);
const frame = commands('frame').map((c) => c.a.id);
assert.ok(frame.includes('view.frame') && frame.includes('view.home'), frame.join());
assert.ok(modifierActions('Ctrl').some((x) => x.a.id === 'edit.undo'));

console.log(JSON.stringify({ ok: true, cases: cases.length, actions: allActions().length, rows: rows.length,
  keys: Object.fromEntries([...keyed.entries()].filter(([k]) => !k.startsWith('pose-')).map(([k, v]) => [k, v])) }));

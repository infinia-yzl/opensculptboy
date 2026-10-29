// OpenSculptBoy
// Apache License, Version 2.0
//
// The interface's state without a browser (run by test/test_viewer_frame.py):
//   node viewer/test/frame.mjs
// It checks the actions of src/shell/state.ts that the keys, the mode buttons and the View menu call: a mode's key
// opens its mode or closes its panel, a section's key opens it in its mode or closes it, a tool opens on the right and
// takes the place of the mode's panel below the wide layout, the store tells its listeners only of real changes, and
// the widths of the layouts. A failed check throws, so node exits with an error.
import assert from 'node:assert/strict';
import { actions, ui, widthOf } from '../src/shell/state.ts';
import { MODES, SECTIONS } from '../src/ui/shortcuts.ts';

let calls = 0;
ui.subscribe(() => calls++);
ui.set({ width: 'wide', mode: 'create', open: true, tool: null });
const s = () => ui.get();
assert.equal(MODES.length, 6);
assert.deepEqual(MODES.map((m) => m.label), ['Create', 'Dress', 'Express', 'Pose', 'Sheet', 'Export']);
assert.deepEqual(SECTIONS.create.map((x) => x.id), ['characters', 'body', 'face', 'hair', 'skin']);
// a mode's key: the mode opens, or its panel closes when it shows
actions.toggleMode('dress');
assert.deepEqual([s().mode, s().open], ['dress', true]);
actions.toggleMode('dress');
assert.deepEqual([s().mode, s().open], ['dress', false]);
actions.toggleMode('dress');
assert.equal(s().open, true);
// a section's key opens it in its mode, and closes the panel when it shows
actions.toggleSection('create', 'face');
assert.deepEqual([s().mode, s().sub.create, s().open], ['create', 'face', true]);
actions.toggleSection('create', 'face');
assert.equal(s().open, false);
actions.toggleSection('create', 'hair');
assert.deepEqual([s().sub.create, s().open], ['hair', true]);
actions.toggleMode('create');
assert.equal(s().open, false);
actions.toggleMode('create');
assert.deepEqual([s().mode, s().sub.create], ['create', 'hair']);   // the mode keeps its last section
// the tools: beside the mode's panel on a wide screen, in its place on a narrower one
actions.toggleTool('rig');
assert.deepEqual([s().tool, s().open], ['rig', true]);
actions.toggleTool('rig');
assert.equal(s().tool, null);
ui.set({ width: 'medium' });
actions.toggleTool('scene');
assert.deepEqual([s().tool, s().open], ['scene', false]);
actions.closeTool();
// the store tells its listeners only of a change
const before = calls;
ui.set({ width: 'medium', tool: null });
assert.equal(calls, before);
ui.set({ view: { frame: 'body', angle: '', turntable: false } });
ui.set({ view: { frame: 'body', angle: '', turntable: false } });
assert.equal(calls, before + 1);
// the Open dialog and its message
actions.showOpen({ title: 'T', body: 'B' });
assert.deepEqual([s().openDialog, s().openError.title], [true, 'T']);
actions.hideOpen();
assert.deepEqual([s().openDialog, s().openError], [false, null]);
// the theme and the widths of the layouts
actions.setTheme('dark');
assert.equal(s().theme, 'dark');
assert.deepEqual([widthOf(390), widthOf(719), widthOf(720), widthOf(1099), widthOf(1100), widthOf(1440)], ['phone', 'phone', 'medium', 'medium', 'wide', 'wide']);
console.log(JSON.stringify({ ok: true, modes: MODES.map((m) => m.id) }));

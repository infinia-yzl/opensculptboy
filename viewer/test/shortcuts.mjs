// OpenSculptBoy
// Apache License, Version 2.0
//
// The keyboard shortcuts without a browser (run by test/test_viewer_shortcuts.py):
//   node viewer/test/shortcuts.mjs
// It checks the keys that a keydown event means on other keyboards (Caps Lock, Shift, a keyboard in another script,
// an input method, the digit row of an AZERTY keyboard) and that the help lists every section's key, and prints a JSON
// summary. A failed check throws, so node exits with an error.
import assert from 'node:assert/strict';
import { PANEL_KEYS, SHORTCUTS, shortcutKeys } from '../src/ui/shortcuts.ts';

const cases = [
  [{ key: 'w', code: 'KeyW' }, ['w']],
  [{ key: 'W', code: 'KeyW' }, ['w']],                       // Caps Lock: the letter alone
  [{ key: 'W', code: 'KeyW', shiftKey: true }, ['W']],        // Shift+W: the skin weights
  [{ key: 'w', code: 'KeyW', shiftKey: true }, ['W']],        // Shift with Caps Lock
  [{ key: 'ц', code: 'KeyW' }, ['w']],                       // a Cyrillic keyboard
  [{ key: 'Process', code: 'KeyX' }, ['x']],                 // an input method
  [{ key: 'Unidentified', code: 'KeyS' }, ['s']],
  [{ key: 'z', code: 'KeyW' }, ['z']],                       // AZERTY: Z keeps its own meaning
  [{ key: '&', code: 'Digit1' }, ['&', '1']],                // AZERTY's digit row
  [{ key: '-', code: 'Digit6' }, ['-', '6']],                // the minus first, then the digit
  [{ key: '?', code: 'Slash', shiftKey: true }, ['?']],
  [{ key: 'Dead', code: 'Backquote' }, ['`']],               // a dead key where US keyboards have the backquote
  [{ key: ' ', code: 'Space' }, [' ']],
  [{ key: 'ArrowLeft', code: 'ArrowLeft' }, ['ArrowLeft']],
  [{ key: 'Escape', code: 'Escape' }, ['Escape']],
];
for (const [e, want] of cases) assert.deepEqual(shortcutKeys(e), want, JSON.stringify(e));

// the help's Panels rows name every section's key, upper case, in the order of the segments
const panels = SHORTCUTS.find(([t]) => t === 'Panels')[1];
const listed = panels.slice(0, 2).flatMap(([k]) => k.split(' ')).map((x) => x.toLowerCase());
assert.deepEqual(listed, Object.values(PANEL_KEYS));
assert.equal(new Set(Object.values(PANEL_KEYS)).size, Object.keys(PANEL_KEYS).length);

console.log(JSON.stringify({ ok: true, cases: cases.length, panels: PANEL_KEYS }));

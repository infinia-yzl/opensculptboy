// OpenSculptBoy
// Apache License, Version 2.0
//
// Blender's modal grab and turn (G and R of key map version 2; DESIGN-0014): once started, the selection follows the
// mouse without a button held until a click or Enter confirms it; Esc or a right-click puts it back; X, Y or Z during
// the operation locks an axis (again: the axis in the item's own frame, a third time: free); a typed number sets the
// distance in millimetres or the angle in degrees; Shift moves at a quarter of the speed and Ctrl snaps. The posing
// editor starts one from its pose.grab and pose.turn-keys actions and applies the values it reports; this file owns
// the input, the keys and the status line.

import { defineScope, registerActions } from './actions.ts';
import { clearStatus, PRIORITY, setStatus } from './status.ts';
import { settings } from './settings.ts';

export type Axis = 'free' | 'x' | 'y' | 'z';
export interface ModalValue {
  dx: number; dy: number;            // the pointer since the start, CSS pixels, times the drag speed (a quarter with Shift)
  axis: Axis;
  local: boolean;                    // the axis in the item's own frame (the second press of the axis key)
  typed: number | null;              // a typed number (millimetres for a grab, degrees for a turn), or null
  snap: boolean;                     // Ctrl held
}
export interface ModalOp {
  kind: 'grab' | 'turn';
  label: string;                     // 'Point 22', 'Right wrist'
  update(v: ModalValue): string | void;   // applies the value; returns its text for the status line ('3.2 mm', '12°')
  confirm(): void;                   // one step of undo
  cancel(): void;                    // puts the item back
}

let op: ModalOp | null = null;
let state: { x0: number; y0: number; x: number; y: number; axis: Axis; local: boolean; text: string; snap: boolean; fine: boolean; ax: number; ay: number; lx: number; ly: number } | null = null;

export function modalActive(): boolean { return !!op; }

function value(): ModalValue {
  const s = state!;
  const typed = s.text && s.text !== '-' && s.text !== '.' ? parseFloat(s.text) : null;
  return { dx: s.ax, dy: s.ay, axis: s.axis, local: s.local, typed: typed !== null && isFinite(typed) ? typed : null, snap: s.snap };
}
function paint(readout: string | void) {
  if (!op || !state) return;
  const axisText = state.axis === 'free' ? '' : ` · ${state.local ? 'own ' : ''}${state.axis.toUpperCase()} axis`;
  setStatus('modal', [
    { text: (op.kind === 'grab' ? 'Grab ' : 'Turn ') + op.label + (readout ? ' · ' + readout : '') + axisText + (state.text ? ' · typed ' + state.text + (op.kind === 'grab' ? ' mm' : '°') : '') },
    { keys: 'X Y Z', text: 'lock an axis' }, { keys: '0-9', text: 'type a value' }, { keys: 'Shift', text: 'fine' }, { keys: 'Ctrl', text: 'snap' },
    { keys: 'Enter', text: 'confirm' }, { keys: 'Esc', text: 'put it back' },
  ], { priority: PRIORITY.drag + 10 });
}
function update() { if (op && state) paint(op.update(value())); }

const move = (e: PointerEvent) => {
  if (!state) return;
  const k = settings.get().dragSpeed * (e.shiftKey ? 0.25 : 1);
  state.ax += (e.clientX - state.lx) * k; state.ay += (e.clientY - state.ly) * k;
  state.lx = e.clientX; state.ly = e.clientY;
  state.snap = e.ctrlKey || e.metaKey;
  update();
};
const down = (e: PointerEvent) => {
  if (!op) return;
  e.preventDefault(); e.stopImmediatePropagation();
  if (e.button === 2) cancelModal(); else confirmModal();
};
const block = (e: Event) => { if (op) { e.preventDefault(); e.stopImmediatePropagation(); } };

function stop() {
  op = null; state = null;
  clearStatus('modal');
  removeEventListener('pointermove', move, true);
  removeEventListener('pointerdown', down, true);
  removeEventListener('contextmenu', block, true);
}
export function confirmModal() { const o = op; stop(); o?.confirm(); }
export function cancelModal() { const o = op; stop(); o?.cancel(); }

// start a grab or a turn at the pointer's place
export function startModal(o: ModalOp, at: { x: number; y: number }) {
  if (op) cancelModal();
  op = o;
  state = { x0: at.x, y0: at.y, x: at.x, y: at.y, lx: at.x, ly: at.y, ax: 0, ay: 0, axis: 'free', local: false, text: '', snap: false, fine: false };
  addEventListener('pointermove', move, true);
  addEventListener('pointerdown', down, true);
  addEventListener('contextmenu', block, true);
  update();
}

// the keys while a grab or a turn lasts: the deepest scope, so they mean nothing else meanwhile
defineScope({ id: 'modal', active: () => !!op, rank: 200 });
const axisKey = (a: Axis) => () => {
  if (!state) return;
  if (state.axis !== a) { state.axis = a; state.local = false; } else if (!state.local) state.local = true; else { state.axis = 'free'; state.local = false; }
  update();
};
const typeKey = (c: string) => () => {
  if (!state) return;
  if (c === '-') state.text = state.text.startsWith('-') ? state.text.slice(1) : '-' + state.text;
  else if (c === 'Backspace') state.text = state.text.slice(0, -1);
  else if (c !== '.' || !state.text.includes('.')) state.text += c;
  update();
};
registerActions([
  { id: 'modal.confirm', label: 'Confirm the grab or the turn', keys: ['Enter', 'Space'], group: 'Points', tier: 'frequent', scope: 'modal', help: false, command: false,
    inDialog: true, run: () => confirmModal() },
  { id: 'modal.cancel', label: 'Put it back', keys: ['Escape'], group: 'Points', tier: 'frequent', scope: 'modal', help: false, command: false, inDialog: true, run: () => cancelModal() },
  { id: 'modal.x', label: 'Lock the X axis', keys: ['X'], group: 'Points', tier: 'frequent', scope: 'modal', help: false, command: false, run: axisKey('x') },
  { id: 'modal.y', label: 'Lock the Y axis', keys: ['Y'], group: 'Points', tier: 'frequent', scope: 'modal', help: false, command: false, run: axisKey('y') },
  { id: 'modal.z', label: 'Lock the Z axis', keys: ['Z'], group: 'Points', tier: 'frequent', scope: 'modal', help: false, command: false, run: axisKey('z') },
  { id: 'modal.type', label: 'Type a value', keys: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Numpad0', 'Numpad1', 'Numpad2', 'Numpad3', 'Numpad4', 'Numpad5',
    'Numpad6', 'Numpad7', 'Numpad8', 'Numpad9', '.', 'Numpad.', '-', 'Backspace'], group: 'Points', tier: 'frequent', scope: 'modal', help: false, command: false, repeat: true,
    run: (c) => { const k = c.chord.replace('Numpad', ''); typeKey(k === 'Backspace' ? 'Backspace' : k)(); } },
]);

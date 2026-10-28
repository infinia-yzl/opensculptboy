// OpenSculptBoy
// Apache License, Version 2.0
//
// The Expression section of the Stage segment: the VRM presets of the exports as buttons, the visemes in a group that
// folds, and an intensity. The face itself lives in face/expressions.ts behind the interface of face/api.ts; a later
// editor of the facial actions takes the place of this section.

import { face, faceReady, type FaceState } from '../face/api.ts';
import { chip, foldGroup, sectionHead, slider } from './controls.ts';
import { h } from './dom.ts';

// the presets in the order of the buttons, with their labels (VRM 1.0 names)
const PRESETS: [string, string][] = [
  ['neutral', 'Neutral'], ['happy', 'Happy'], ['angry', 'Angry'], ['sad', 'Sad'], ['relaxed', 'Relaxed'], ['surprised', 'Surprised'],
  ['blink', 'Blink'], ['blinkLeft', 'Blink left'], ['blinkRight', 'Blink right'],
  ['aa', 'A'], ['ih', 'I'], ['ou', 'U'], ['ee', 'E'], ['oh', 'O'],
];

export function expressionSection() {
  const el = h('section');
  const head = sectionHead('Expression');
  const presets = h('div.chips', { role: 'group', 'aria-label': 'Expressions' });
  const fold = foldGroup('Visemes');
  fold.body.classList.add('chips');
  const note = h('p.note', { text: 'The expressions of the VRM export, as mixes of the 52 facial actions. On a card with the reference\'s mouth poses, the vowels play its own poses. Left and right are the figure\'s own.' });
  const empty = h('p.note', { text: 'This page has no facial actions.', hidden: true });
  const chips = new Map<string, HTMLButtonElement>();
  let current = 'neutral', weight = 1;
  const show = (fadeMs: number) => {
    if (!faceReady()) return;
    const d: any = face(), state: FaceState = d.preset(current, weight);
    // the head and the gaze stay as they are
    const was = d.get();
    if (was.gaze) state.gaze = was.gaze;
    if (was.head) state.head = was.head;
    // the section shows the preset chosen once the face shows it
    const shown = fadeMs && d.fadeTo ? d.fadeTo(state, fadeMs) : d.set(state);
    shown.then(sync);
  };
  const choose = (name: string) => { current = name; show(180); };
  for (const [name, label] of PRESETS) { const b = chip(label, () => choose(name)); b.dataset.expression = name; presets.append(b); chips.set(name, b); }
  const intensity = slider({ id: 'expr-weight', label: 'Intensity', min: 0, max: 1, step: 0.01, def: 1, ends: ['None', 'Full'],
    readout: (v) => `${Math.round(v * 100)} %`, onInput: (v) => { weight = v; show(0); } });
  el.append(head.el, presets, h('div.sliders', {}, intensity.el), fold.el, note, empty);
  let built = false;
  const build = () => {
    if (built || !faceReady()) return;
    built = true;
    const names = face().presetNames().filter((n) => !PRESETS.some(([p]) => p === n));
    for (const name of names) { const b = chip(name, () => choose(name)); b.dataset.expression = name; fold.body.append(b); chips.set(name, b); }
    fold.count.textContent = String(names.length);
  };
  const sync = () => {
    build();
    const ready = faceReady();
    empty.hidden = ready; presets.hidden = !ready; intensity.el.hidden = !ready; fold.el.hidden = !ready || !fold.body.children.length;
    // a face set elsewhere (a card's own face, a script) shows no preset as chosen
    const shown = ready ? face().get() : null, want = ready ? face().preset(current, weight) : null;
    const same = !!shown && !!want && sameActions(shown.actions, want.actions);
    for (const [n, b] of chips) b.setAttribute('aria-pressed', String(same && n === current));
    head.note.textContent = same ? (PRESETS.find(([p]) => p === current)?.[1] || current) : ready && Object.keys(shown!.actions).length ? 'Custom' : '';
  };
  return { el, sync };
}

function sameActions(a: Record<string, number>, b: Record<string, number>) {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const k of keys) if (Math.abs((a[k] || 0) - (b[k] || 0)) > 1e-4) return false;
  return true;
}

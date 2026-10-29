// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The editor of the look that the Create mode's sections use (src/shell/sections), and the random draws of the
// body, the face, the hair and the colours.

import type { App, Look } from './app.ts';

export interface Editor {
  app: App;
  begin(label: string): void;
  end(): void;
  // one edit with its history entry
  edit(label: string, mut: (n: Look) => void): void;
  // a whole look with its history entry (a character, a preset, a random draw)
  commit(label: string, look: any): void;
  // slider edits apply at most once a frame
  queue(mut: (n: Look) => void): void;
  // the name of the last character or look picked, for the "edited" state
  origin: string;
  faceSpread: number;
}

const pick = <T,>(a: T[]) => a[Math.floor(Math.random() * a.length)];
const rnd = (lo: number, hi: number) => lo + Math.random() * (hi - lo);
const r3 = (v: number) => Math.round(v * 1000) / 1000;

// ------------------------------------------------------------------ random draws
export function randomPhenotype(app: App) {
  const ph: Record<string, number> = { ...app.baselinePhenotype() };
  const set = (k: string, v: number) => { if (k in ph) ph[k] = r3(v); };
  set('gender', Math.random() < 0.5 ? rnd(0, 0.25) : rnd(0.75, 1));
  set('age', rnd(0.45, 0.96));
  set('muscle', rnd(0.2, 0.8)); set('weight', rnd(0.2, 0.8));
  set('height', rnd(0.2, 0.8)); set('proportions', rnd(0.3, 0.7));
  // one main source shape, sometimes mixed with a second one
  const races = ['african', 'asian', 'caucasian'].filter((k) => k in ph), main = pick(races);
  for (const k of races) set(k, k === main ? 1 : Math.random() < 0.3 ? rnd(0.2, 0.8) : 0);
  return ph;
}
export function randomColours(app: App, n: Look) {
  n.skin = { tone: r3(rnd(0.05, 0.95)), undertone: r3(rnd(-0.6, 0.6)) };
  n.hair.color = pick(app.palettes.hair)[1];
  n.eyes.color = pick(app.palettes.eyes)[1];
}
export function randomHair(app: App, n: Look) {
  const spec = pick(app.hairSpecs().filter((s) => s.name !== 'bald'));
  n.hair = { color: pick(app.palettes.hair)[1], style: spec.name, part: Math.random() < 0.5 ? 'left' : 'right' };
}

export function randomBodyInto(app: App, ed: Editor, L: Look) {
  const next = JSON.parse(JSON.stringify(L));
  next.phenotype = randomPhenotype(app);
  if (app.facePrior()) next.face = app.sampleFace(next.phenotype, ed.faceSpread);
  next.name = 'Custom';
  ed.commit('Random body', next);
}
export function randomFaceInto(app: App, ed: Editor) {
  if (!app.facePrior()) return;
  const L = app.look();
  ed.edit('Random face', (n) => { n.face = app.sampleFace(L.phenotype, ed.faceSpread); });
}

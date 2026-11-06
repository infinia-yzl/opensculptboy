// OpenSculptBoy
// Apache License, Version 2.0
//
// The stages of the Points tab, from coarse to fine. Each stage shows its own points, one instruction line and the
// counts of its points alone:
// 1. the figure: one handle at the pelvis (the root bone's head), which moves and turns the whole figure;
// 2. the head, the hands and the feet: each point pulls its own chain (the neck, an arm, a leg);
// 3. the joints in between: the shoulders, the elbows, the hips, the knees, the heels and the toes;
// 4. the fingers, which the Hands tab shows below Grip and Fist.
// A saved session starts at the first stage with work left (startStage); no point is dropped or moved by a stage.

import { POINTS } from './points.ts';

export type Stage = 1 | 2 | 3 | 4;
export type PointState = 'placed' | 'look' | 'off' | 'unplaced';

// the handle of stage 1: not a reference point of the card, so it never enters the card's points
export const FIGURE = 'figure';

export const MAIN_POINTS = ['head_top', 'wrist.R', 'wrist.L', 'ankle.R', 'ankle.L'];
const MAIN = new Set(MAIN_POINTS);

export interface StageInfo { stage: Stage; title: string; short: string; line: string }
export const STAGES: StageInfo[] = [
  { stage: 1, title: 'Place the figure', short: 'Figure', line: "Drag the point onto the statue's pelvis. The whole figure moves with it." },
  { stage: 2, title: 'The head, the hands and the feet', short: 'Head, hands, feet', line: 'Drag each point onto the same place on the statue. Each one pulls its own arm, leg or head.' },
  { stage: 3, title: 'The joints in between', short: 'Joints', line: 'Drag the shoulders, elbows, hips, knees and feet onto the statue for small corrections.' },
  { stage: 4, title: 'The fingers', short: 'Fingers', line: 'Shape each hand in Hands first, with Grip and Fist. The finger points are for fine-tuning.' },
];

// the stage of a reference point
export function stageOf(name: string): Stage {
  if (name === FIGURE) return 1;
  if (MAIN.has(name)) return 2;
  const p = POINTS.find((q) => q.name === name);
  return p && p.group !== 'body' ? 4 : 3;
}

// the reference points of a stage, in the screens' order (stage 1 has none: its handle is the figure itself)
export function stagePoints(stage: Stage): string[] {
  if (stage === 1) return [];
  if (stage === 2) return MAIN_POINTS.slice();
  return POINTS.filter((p) => stageOf(p.name) === stage).map((p) => p.name);
}

// the points the view shows in a stage: its own, and the placed points of the earlier stages, which its solves keep
export function shownPoints(stage: Stage, placed: (name: string) => boolean): string[] {
  const own = new Set(stagePoints(stage));
  return POINTS.filter((p) => own.has(p.name) || (stageOf(p.name) < stage && placed(p.name))).map((p) => p.name);
}

export interface StageCounts { placed: number; look: number; off: number; total: number }
export function stageCounts(stage: Stage, state: (name: string) => PointState, figureMoved = false): StageCounts {
  if (stage === 1) return { placed: figureMoved ? 1 : 0, look: 0, off: 0, total: 1 };
  const names = stagePoints(stage), states = names.map(state);
  return { placed: states.filter((s) => s === 'placed').length, look: states.filter((s) => s === 'look').length,
    off: states.filter((s) => s === 'off').length, total: names.length };
}

// where a session opens: with no point placed, stage 1 (stage 2 when the figure was moved already); otherwise the first
// of stages 2 and 3 with a point not yet placed or one that needs a look, then stage 4 when a finger point needs a look;
// with every point placed and checked, the highest stage that holds a placed point
export function startStage(state: (name: string) => PointState, figureMoved = false): Stage {
  const any = POINTS.some((p) => state(p.name) !== 'unplaced');
  if (!any) return figureMoved ? 2 : 1;
  for (const s of [2, 3] as Stage[]) if (stagePoints(s).some((n) => state(n) === 'unplaced' || state(n) === 'look')) return s;
  if (stagePoints(4).some((n) => state(n) === 'look')) return 4;
  return stagePoints(4).some((n) => state(n) !== 'unplaced') ? 4 : 3;
}

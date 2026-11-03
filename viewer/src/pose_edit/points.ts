// OpenSculptBoy
// Apache License, Version 2.0
//
// The reference points as the editor shows them: the set of opensculptboy.posing.reference_points
// (data/posing/reference_points.json), numbered as the screens number them (the body 1 to 15 from the head down,
// the right side first; the right hand 16 to 36 and the left hand 37 to 57 in MediaPipe's order), with plain names
// ("Right index finger, middle joint") and the chain each point turns when it moves.

import SET from '../../../src/opensculptboy/data/posing/reference_points.json' with { type: 'json' };

export interface RefPoint {
  name: string; chain: string; joint: string; end: boolean; finger: string | null; triangle: number; vertices: number[]; barycentric: number[];
  mediapipe?: number;
  number: number;
  label: string;             // plain words, the character's own side
  side: 'L' | 'R' | '';
  group: 'hand.r' | 'hand.l' | 'body';
  solve: string;             // the chain whose bones turn when the point moves
}

const BODY_ORDER = ['head_top', 'shoulder', 'elbow', 'hip', 'knee', 'ankle', 'heel', 'toe'];
const FINGERS: Record<string, string> = { thumb: 'thumb', index: 'index finger', middle: 'middle finger', ring: 'ring finger', pinky: 'little finger' };
const JOINTS: Record<string, string> = { cmc: 'base', mcp: 'knuckle', pip: 'middle joint', dip: 'end joint', ip: 'end joint' };
const BODY_WORDS: Record<string, string> = { shoulder: 'shoulder', elbow: 'elbow', hip: 'hip', knee: 'knee', ankle: 'ankle', heel: 'heel', toe: 'big toe' };

export function plainName(name: string): string {
  if (name === 'head_top') return 'Top of the head';
  const side = name.endsWith('.L') ? 'Left' : name.endsWith('.R') ? 'Right' : '';
  const base = name.replace(/\.[LR]$/, '');
  if (base === 'wrist') return `${side} wrist`;
  const [finger, joint] = base.split('_');
  if (FINGERS[finger]) return joint === 'tip' ? `${side} ${FINGERS[finger]} tip` : `${side} ${FINGERS[finger]}, ${JOINTS[joint]}`;
  return `${side} ${BODY_WORDS[base] || base}`;
}

function solveChain(chain: string): string {
  if (chain.startsWith('hand.')) return 'arm.' + chain.slice(5);
  return chain;
}

function build(): RefPoint[] {
  const raw: any[] = (SET as any).points;
  const byName = new Map(raw.map((p) => [p.name, p]));
  const order: string[] = [];
  for (const w of BODY_ORDER) {
    if (w === 'head_top') order.push('head_top');
    else order.push(w + '.R', w + '.L');
  }
  for (const side of ['.R', '.L']) for (const p of raw.filter((x) => x.chain === 'hand.' + side[1].toLowerCase()).sort((a, b) => a.mediapipe - b.mediapipe)) order.push(p.name);
  for (const p of raw) if (!order.includes(p.name)) order.push(p.name);
  return order.filter((n) => byName.has(n)).map((n, i) => {
    const p = byName.get(n);
    const side = n.endsWith('.L') ? 'L' : n.endsWith('.R') ? 'R' : '';
    const group = p.chain === 'hand.r' ? 'hand.r' : p.chain === 'hand.l' ? 'hand.l' : 'body';
    return { ...p, number: i + 1, label: plainName(n), side, group, solve: solveChain(p.chain) } as RefPoint;
  });
}

export const POINTS: RefPoint[] = build();
export const POINT_BY_NAME = new Map(POINTS.map((p) => [p.name, p]));
export const POINT_BY_NUMBER = new Map(POINTS.map((p) => [p.number, p]));

// the points a solve of a chain uses (reference_points.points_of): an arm's are its hand's too
export function pointsOfChain(chain: string): RefPoint[] {
  const lower = chain.toLowerCase(), [part, side] = lower.split('.');
  const wanted = new Set<string>();
  if (part === 'arm') { wanted.add(`arm.${side}`); wanted.add(`hand.${side}`); }
  else if (part === 'leg' || part === 'foot') wanted.add(`leg.${side}`);
  else wanted.add(lower);
  return POINTS.filter((p) => wanted.has(p.chain));
}

// the hand's points in MediaPipe's order (0 the wrist), for the hand map
export function handPoints(side: 'L' | 'R'): RefPoint[] {
  return POINTS.filter((p) => p.group === (side === 'R' ? 'hand.r' : 'hand.l')).sort((a, b) => (a.mediapipe ?? 0) - (b.mediapipe ?? 0));
}

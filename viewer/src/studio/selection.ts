// OpenSculptBoy
// Apache License, Version 2.0
//
// The selection as the navigation sees it: what F frames (with its steps outward: a point, its hand, its arm, the
// whole figure), what H hides and / isolates, and its name for the status line. The posing editor sets it when the
// person selects a point, a joint or a hand; nothing selected, F frames the whole figure.

import type { BoxLike, Vec3Like } from './camera.ts';

export interface Framing { box: BoxLike | Vec3Like[]; label: string; pivot?: Vec3Like }
export interface StudioSelection {
  label: string;                                  // 'Point 22', 'Right hand'
  // the framing of each step of F: 0 the selection itself, then wider parts; null past the last step
  frame(step: number): Framing | null;
  // the bones whose skin H hides and / isolates (the rig's names), when the selection is a part of the body
  bones?: string[];
}

let current: StudioSelection | null = null;
const listeners = new Set<() => void>();
export function setSelection(sel: StudioSelection | null) { current = sel; for (const l of listeners) l(); }
export function selection(): StudioSelection | null { return current; }
export function subscribeSelection(fn: () => void): () => void { listeners.add(fn); return () => { listeners.delete(fn); }; }

// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The modes of the page and their sections, and the tools of the View menu. Their keys, with every other key of the
// page, live in the action registry of studio/actions.ts (key map version 2, studio/keymap.ts), which the key handler,
// the tooltips, the key caps, the help and the command list read.

export type ModeId = 'create' | 'dress' | 'express' | 'pose' | 'sheet' | 'export';
export interface ModeDef { id: ModeId; label: string; icon: string; tip: string }
export interface SectionDef { id: string; label: string; short?: string; icon: string }

// the six modes of the top bar, in order
export const MODES: ModeDef[] = [
  { id: 'create', label: 'Create', icon: 'body', tip: 'Make the character: body, face, hair, skin and eyes' },
  { id: 'dress', label: 'Dress', icon: 'wear', tip: 'Dress the character' },
  { id: 'express', label: 'Express', icon: 'expression', tip: 'Give the character expressions' },
  { id: 'pose', label: 'Pose', icon: 'pose', tip: 'Pose the character' },
  { id: 'sheet', label: 'Sheet', icon: 'sheet', tip: 'Make a character sheet' },
  { id: 'export', label: 'Export', icon: 'export', tip: 'Save the character and its pictures' },
];
// the sections of each mode; Create's are its sub-tabs, on 1 to 4
export const SECTIONS: Record<ModeId, SectionDef[]> = {
  create: [
    { id: 'body', label: 'Body', icon: 'body' },
    { id: 'face', label: 'Face', icon: 'face' },
    { id: 'skin', label: 'Skin and eyes', short: 'Skin', icon: 'skin' },
    { id: 'hair', label: 'Hair', icon: 'hair' },
  ],
  dress: [{ id: 'wear', label: 'Wearables', icon: 'wear' }],
  express: [{ id: 'expression', label: 'Expression', icon: 'expression' }],
  pose: [{ id: 'pose', label: 'Pose', icon: 'pose' }],
  sheet: [{ id: 'sheet', label: 'Sheet', icon: 'sheet' }],
  export: [{ id: 'export', label: 'Export', icon: 'export' }],
};
// the tools of the View menu that open a panel on the right
export const TOOLS: SectionDef[] = [
  { id: 'rig', label: 'Rig', icon: 'bones' },
  { id: 'scene', label: 'Scene', icon: 'scene' },
];

// a key as the interface shows it: a letter in upper case, Ctrl as Cmd on a Mac
export const isMac = () => typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || '');
export function keyText(k: string): string {
  const t = k.length === 1 ? k.toUpperCase() : k;
  return isMac() ? t.replace(/\bCtrl\b/g, 'Cmd') : t;
}

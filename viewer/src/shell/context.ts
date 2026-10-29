// OpenSculptBoy
// Apache License, Version 2.0
//
// What the interface gets from the page (ui/index.ts): the renderer's App, the editor with its history, the sections
// that are not yet rebuilt in React (their own DOM, hosted in the new panels), the page's actions and the Open path.

import { createContext, useContext } from 'react';
import type { App } from '../ui/app.ts';
import type { Editor } from '../ui/character_ui.ts';
import type { Opener } from './open.ts';

export interface Legacy {
  // the existing sections by id: characters, face, hair, wear, expression, pose, sheet, rig, scene
  sections: Record<string, HTMLElement>;
  // the Sheet mode's layout view of the page, and the refresh of its section when it shows
  sheetView: HTMLElement;
  sheetRefresh(): void;
  undo(): void;
  redo(): void;
  random: { character(): void; body(): void; face(): void; hair(): void; colours(): void };
  reset: { all(): void; body(): void; face(): void };
  saveImage(): void;
  poseCard(): void;
  silhouette(): void;
  downloadLook(): void;
  toggleStats(): void;
  showHelp(): void;
  fullscreen(): void;
  // the view: framings, angles, the turntable and the reset (camera_nav.ts)
  frame(name: string): void;
  angle(name: string): void;
  turntable(): void;
  resetView(): void;
  // the refinement ring of the view bar: paints itself from the frame hook
  ring: HTMLElement;
  // the height rail of the view (camera_nav.ts)
  heightRail: HTMLElement;
  // the character shown: its name, its kind (a card, a fitted card) and the colours of its dot (skin, hair)
  identity(): { name: string; kind: string; a: string; b: string };
  opener: Opener;
  hosted: boolean;
}

export interface ShellDeps { app: App; ed: Editor; legacy: Legacy }

export const ShellCtx = createContext<ShellDeps | null>(null);
export function useShell(): ShellDeps {
  const c = useContext(ShellCtx);
  if (!c) throw new Error('The interface needs its ShellCtx.');
  return c;
}

// the slot of the View menu's Settings item: a later settings panel registers here (open() shows it)
export const settingsSlot: { open: (() => void) | null; label: string } = { open: null, label: 'Settings' };

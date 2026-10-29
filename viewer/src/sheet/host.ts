// OpenSculptBoy
// Apache License, Version 2.0
//
// What the sheet needs from the renderer (main.ts): it holds the scene still, shows a pose, turns the figure, renders
// one picture offscreen with a fixed camera and reports where the figure stands. The sheet's camera stays in front of
// the figure (on +Z, looking along -Z), and the figure turns on its vertical axis, so the page's lights fall on every
// view of a turnaround alike.

import type { PoseSlot } from './settings.ts';

// a meter reading: the exposure chosen, the limit that set it, and the display values of the figure's covered pixels at
// that exposure (the share with a channel at 250 of 255 or more, and percentiles of the luminance)
export interface MeterReading { exposure: number; limit: 'highlights' | 'midtones'; pixels: number; clipped: number; p5: number; median: number; p95: number; p99: number }

// a camera frame of the sheet (layout.ts Frame) with the figure's turn about the vertical axis, in degrees: a positive
// turn swings its front toward its own left, so a turn of -45 shows its left side as a three-quarter view does
export interface SheetView { turn: number; cx: number; cy: number; halfH: number; width: number; height: number }

export interface FigureInfo {
  ground: number;
  top: number;                                   // the top of the head (without the hair)
  head: [number, number, number];                // the face
  chin: [number, number, number];                // the bottom of the chin
  crown: [number, number, number];               // the top of the skull
  headDir: [number, number, number];             // where the face looks
  lo: [number, number, number]; hi: [number, number, number];   // the box of the joints
  joints: Record<string, [number, number, number]>;
}

export interface SheetHost {
  // hold the scene for the sheet: no hair physics, no blink timer, a paused clip, and the page's own view kept on its
  // last picture, dimmed; off puts back the pose, the turn and the settings of before
  hold(on: boolean): void;
  // show a pose of the library or a pose of its own at once; false for an unknown pose
  showPose(slot: PoseSlot): boolean;
  turn(deg: number): void;
  // one picture with a transparent background: `samples` accumulated passes; tick(pixels) runs after each pass and
  // waits for the next frame when the frame has drawn enough
  render(view: SheetView, samples: number, tick: (pixels: number) => Promise<void>): Promise<ImageData>;
  // the lights of the pictures that follow: the sheet's studio lights or the page's preset (settings.ts Lights)
  lights(mode: 'studio' | 'page'): void;
  // meter a full-figure view for the studio lights: the exposure of the pictures that follow, times 2^stops, and what
  // the meter read
  meter(view: SheetView, stops: number, tick: (pixels: number) => Promise<void>): Promise<MeterReading>;
  figure(): FigureInfo;
  // the rotations of a pose of the library, or of the pose shown (name ''), as quaternions per bone
  poseQuaternions(name: string): number[] | null;
  // prepare every facial action's shapes (and a card's own faces) in short steps between frames
  warmFaces(): Promise<void>;
  // the poses of the library: name, label, group
  poses(): { name: string; label: string; group: string }[];
  // the name of the card opened in the Wearables section, else of the look shown
  characterName(): string;
}

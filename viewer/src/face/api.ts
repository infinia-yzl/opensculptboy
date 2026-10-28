// OpenSculptBoy
// Apache License, Version 2.0
//
// The face's interface for the rest of the page (DESIGN of the character sheet, docs/viewer.md): the sheet, the
// expression editor and scripts set the face through these functions. The expressions track implements them in
// face/expressions.ts and registers it with `registerFace`; until then every call is a no-op that resolves at once,
// so the sheet can use the interface before the face moves.

// A face: weights from 0 to 1 by facial action name (Anny's ARKit names, model.facial_action_labels, such as
// "jawOpen" or "eyeBlinkLeft"; Left and Right are the character's own sides), an optional gaze in degrees (yaw to
// the character's left, pitch up) and an optional head turn, tilt and nod in degrees.
export interface FaceState {
  actions: Record<string, number>;
  gaze?: { yaw: number; pitch: number };
  head?: { turn: number; tilt: number; nod: number };
}

export interface FaceDriver {
  actionNames(): string[];
  presetNames(): string[];
  preset(name: string, weight?: number): FaceState;
  // shows the face; resolves once the page draws it
  set(state: FaceState): Promise<void>;
  get(): FaceState;
}

const NEUTRAL: FaceState = { actions: {} };

const stub: FaceDriver = {
  actionNames: () => [],
  presetNames: () => [],
  preset: () => ({ actions: {} }),
  set: async () => {},
  get: () => NEUTRAL,
};

let driver: FaceDriver = stub;

export function registerFace(d: FaceDriver): void { driver = d; }
export function face(): FaceDriver { return driver; }
export function faceReady(): boolean { return driver !== stub; }

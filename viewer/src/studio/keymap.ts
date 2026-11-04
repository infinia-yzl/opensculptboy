// OpenSculptBoy
// Apache License, Version 2.0
//
// Key map version 2 (DESIGN-0014): the studio's own actions in the registry of actions.ts, with their keys, their
// groups in the help and their tiers. The posing editor registers its own actions (pose.*) in the scopes `pose-edit`
// and below; this file holds the modes, the files, the history, the view, the display and the help.
//
// Hands: the right hand on the mouse, the left hand on W A S D; frequent work takes single keys round WASD, rare or
// destructive work takes a modifier. A key has one meaning in each scope, and the mode letters of key map version 1
// (C D X P S E), Create's letters (B F K H) and the framing digits (0 to 7) are gone.

import type { App } from '../ui/app.ts';
import type { CameraNav } from '../ui/camera_nav.ts';
import { MODES, SECTIONS, type ModeId } from '../ui/shortcuts.ts';
import { actions as uiActions, ui } from '../shell/state.ts';
import { defineScope, registerActions, type ActionDef } from './actions.ts';
import { selection } from './selection.ts';
import { cam } from './camera.ts';

export interface KeymapDeps {
  app: App;
  nav: CameraNav;
  undo(): void;
  redo(): void;
  saveCard(): void;
  hosted: boolean;
  help(): void;
  commands(): void;
  ringMenu(menu: 'view' | 'display'): void;
  escape(): boolean;
  toast(text: string): void;
}

// whether the 3D view has the focus: the canvas, or nothing in particular (the page's body)
export function viewFocused(): boolean {
  const a = document.activeElement;
  return !a || a === document.body || a.tagName === 'CANVAS';
}

export function registerKeymap(d: KeymapDeps): () => void {
  const { app, nav } = d;
  // ---------------------------------------------------------------- the scopes of the modes
  const offs: (() => void)[] = [];
  for (const m of MODES) offs.push(defineScope({ id: m.id, label: m.label, active: () => ui.get().mode === m.id }));

  // ---------------------------------------------------------------- F: the selection, then wider parts
  let fStep = 0, fAt = 0, fSel: unknown = null;
  const frameSelection = () => {
    const sel = selection();
    const now = performance.now();
    fStep = sel && sel === fSel && now - fAt < 4000 ? fStep + 1 : 0;
    fAt = now; fSel = sel;
    const f = sel ? sel.frame(fStep) : null;
    if (f) { cam().frameBounds(f.box, { pivot: f.pivot, label: f.label }); return; }
    // past the last step, or nothing selected: the whole figure
    fStep = -1;
    nav.frameFigure();
  };
  // Ctrl+Space: the 3D view fills the window; again, the panels come back as they were
  let before: { open: boolean; tool: any; character: boolean } | null = null;
  const fullView = () => {
    const s = ui.get();
    if (before) { ui.set(before); before = null; return; }
    before = { open: s.open, tool: s.tool, character: s.character };
    ui.set({ open: false, tool: null, character: false });
  };
  // H, Shift+H, Alt+H and /: the selected part's skin hidden, everything else hidden, all shown, the part isolated
  const parts = { hidden: new Set<string>(), isolated: null as string[] | null };
  const applyParts = () => app.setParts(parts.hidden.size ? [...parts.hidden] : null, parts.isolated);
  const selBones = () => selection()?.bones || [];
  const hasPart = () => selBones().length > 0;
  const motionLoop = () => { const m = app.motion(); return !!m.cur && m.cur.kind === 'loop'; };
  const mode = (id: ModeId) => () => { uiActions.toggleMode(id); };

  const A: ActionDef[] = [
    // ---------------------------------------------------------------- the mouse and the held keys (the help alone)
    { id: 'mouse.turn', label: 'Turn round the pivot', mouse: ['Left-drag'], group: 'Mouse', tier: 'constant' },
    { id: 'mouse.alt', label: 'Turn, even over a point', mouse: ['Alt+Left-drag'], group: 'Mouse', tier: 'constant', help: true },
    { id: 'mouse.move', label: 'Move the view', mouse: ['Middle-drag'], group: 'Mouse', tier: 'constant' },
    { id: 'mouse.zoom', label: 'Zoom to the pointer', mouse: ['Wheel'], group: 'Mouse', tier: 'constant' },
    { id: 'mouse.zoomdrag', label: 'Zoom by dragging', mouse: ['Alt+Right-drag'], group: 'Mouse', tier: 'frequent' },
    { id: 'mouse.fly', label: 'Fly: W and S forward and back, A and D sideways, E and Q up and down, the wheel for the speed', mouse: ['Right-hold'], group: 'Mouse', tier: 'occasional' },
    { id: 'mouse.click', label: 'Select', mouse: ['Click'], group: 'Mouse', tier: 'constant' },
    { id: 'mouse.dbl', label: 'Turn round this spot', mouse: ['Double-click'], group: 'Mouse', tier: 'frequent' },
    { id: 'mouse.right', label: 'Cancel a drag, or open the menu', mouse: ['Right-click'], group: 'Held keys', tier: 'frequent' },
    { id: 'held.shift', label: 'Fine movement while held', keys: ['Shift'], group: 'Held keys', tier: 'constant' },
    { id: 'held.ctrl', label: 'Snap while held', keys: ['Ctrl'], group: 'Held keys', tier: 'constant' },
    { id: 'held.alt', label: 'Turn over points while held', keys: ['Alt'], group: 'Held keys', tier: 'constant' },

    // ---------------------------------------------------------------- the modes and their sections
    ...MODES.filter((m) => m.id !== 'export').map((m, i): ActionDef => ({
      id: 'mode.' + m.id, label: m.label, desc: m.tip, keys: ['Shift+' + (i + 1)], group: 'Modes', tier: 'occasional',
      row: 'mode.all', rowLabel: 'Create, Dress, Express, Pose, Sheet', run: mode(m.id), place: 'The top bar', icon: m.icon,
    })),
    { id: 'mode.export', label: 'Export', desc: 'Save the character and its pictures: GLB, VRM and pictures', keys: ['Ctrl+Shift+E'], group: 'Files', tier: 'rare',
      run: mode('export'), place: 'The top bar', icon: 'export' },
    ...SECTIONS.create.map((s, i): ActionDef => ({
      id: 'create.' + s.id, label: s.label, desc: 'Create: ' + s.label.toLowerCase(), keys: [String(i + 1)], group: 'Modes', tier: 'occasional', scope: 'create',
      row: 'create.sections', rowLabel: 'Body, face, skin and eyes, hair', run: () => uiActions.toggleSection('create', s.id), place: 'Create', icon: s.icon,
    })),

    // ---------------------------------------------------------------- files and history
    { id: 'file.save', label: 'Save the character card', keys: ['Ctrl+S'], group: 'Files', tier: 'rare', inText: true, place: 'The Character panel', icon: 'save',
      run: () => { if (!d.hosted) d.saveCard(); } },
    { id: 'file.open', label: 'Open', desc: 'Open a character, wearables or sheet settings', keys: ['Ctrl+O'], group: 'Files', tier: 'rare', inText: true, place: 'The top bar', icon: 'open',
      run: () => uiActions.showOpen() },
    { id: 'file.new', label: 'New character', desc: 'A starter, the base or your own 3D model', group: 'Files', tier: 'rare', place: 'The top bar', icon: 'new',
      run: () => uiActions.showNew() },
    { id: 'edit.undo', label: 'Undo', keys: ['Ctrl+Z'], group: 'Editing', tier: 'constant', place: 'The top bar', icon: 'undo', repeat: true, run: () => d.undo() },
    { id: 'edit.redo', label: 'Redo', keys: ['Ctrl+Shift+Z', 'Ctrl+Y'], group: 'Editing', tier: 'constant', place: 'The top bar', icon: 'redo', repeat: true, run: () => d.redo() },

    // ---------------------------------------------------------------- the view
    { id: 'view.frame', label: 'Frame the selection', desc: 'Again to widen: point, hand, arm, whole figure', keys: ['F', 'Numpad.'], group: 'The view', tier: 'constant',
      place: 'The view', icon: 'focus', run: frameSelection },
    { id: 'view.home', label: 'Frame the whole figure', keys: ['Home'], group: 'The view', tier: 'frequent', place: 'The view', icon: 'home', run: () => nav.frameFigure(),
      menu: 'view', menuOrder: 6, menuLabel: 'Full figure' },
    { id: 'view.menu', label: 'View menu', desc: 'Front, back, sides, top, face, hands and the whole figure round the pointer', keys: ['`'], group: 'The view', tier: 'frequent',
      place: 'The view', icon: 'compass', run: () => d.ringMenu('view') },
    { id: 'view.back', label: 'Previous view', keys: ['Backspace'], group: 'The view', tier: 'frequent', place: 'The view', icon: 'undo',
      row: 'view.history', rowLabel: 'Previous view, next view', hint: 'No earlier view', run: () => { if (!cam().previousView()) return false; d.toast('Back to the earlier view'); } },
    { id: 'view.forward', label: 'Next view', keys: ['Shift+Backspace'], group: 'The view', tier: 'occasional', place: 'The view', icon: 'redo',
      row: 'view.history', hint: 'No later view', run: () => { if (!cam().nextView()) return false; d.toast('On to the later view'); } },
    { id: 'view.turn-left', label: 'Turn left', keys: ['ArrowLeft', 'Shift+ArrowLeft'], group: 'The view', tier: 'frequent', repeat: true,
      row: 'view.arrows', rowLabel: 'Turn and tilt the view, 15° a press (Shift: 3°)', run: (c) => nav.orbit(c.shift ? -3 : -15) },
    { id: 'view.turn-right', label: 'Turn right', keys: ['ArrowRight', 'Shift+ArrowRight'], group: 'The view', tier: 'frequent', repeat: true, row: 'view.arrows',
      run: (c) => nav.orbit(c.shift ? 3 : 15) },
    { id: 'view.tilt-up', label: 'Look from higher', keys: ['ArrowUp', 'Shift+ArrowUp'], group: 'The view', tier: 'frequent', repeat: true, row: 'view.arrows',
      run: (c) => nav.tilt(c.shift ? 3 : 15) },
    { id: 'view.tilt-down', label: 'Look from lower', keys: ['ArrowDown', 'Shift+ArrowDown'], group: 'The view', tier: 'frequent', repeat: true, row: 'view.arrows',
      run: (c) => nav.tilt(c.shift ? -3 : -15) },
    { id: 'view.up', label: 'Move up the body', keys: ['Ctrl+ArrowUp'], group: 'The view', tier: 'occasional', repeat: true,
      row: 'view.updown', rowLabel: 'Move up or down the body', run: () => nav.move(0.05) },
    { id: 'view.down', label: 'Move down the body', keys: ['Ctrl+ArrowDown'], group: 'The view', tier: 'occasional', repeat: true, row: 'view.updown', run: () => nav.move(-0.05) },
    { id: 'view.zoom-in', label: 'Zoom in', keys: ['=', '+'], group: 'The view', tier: 'occasional', repeat: true, row: 'view.zoom', rowLabel: 'Zoom in, zoom out', run: () => nav.zoom(0.88) },
    { id: 'view.zoom-out', label: 'Zoom out', keys: ['-'], group: 'The view', tier: 'occasional', repeat: true, row: 'view.zoom', run: () => nav.zoom(1 / 0.88) },
    // the views of the view menu, with Blender's numpad keys; Ctrl for the opposite side
    { id: 'view.front', label: 'Front', keys: ['Numpad1'], group: 'The view', tier: 'occasional', menu: 'view', menuOrder: 5, run: () => nav.view('front'),
      row: 'view.numpad', rowLabel: 'Front, right side, top (Ctrl: the opposite side)', place: 'The view menu' },
    { id: 'view.back-side', label: 'Back', keys: ['Ctrl+Numpad1'], group: 'The view', tier: 'occasional', menu: 'view', menuOrder: 9, run: () => nav.view('back'), row: 'view.numpad', place: 'The view menu' },
    { id: 'view.right', label: 'Right side', keys: ['Numpad3'], group: 'The view', tier: 'occasional', menu: 'view', menuOrder: 2, menuLabel: 'Right', run: () => nav.view('right'), row: 'view.numpad', place: 'The view menu' },
    { id: 'view.left', label: 'Left side', keys: ['Ctrl+Numpad3'], group: 'The view', tier: 'occasional', menu: 'view', menuOrder: 8, menuLabel: 'Left', run: () => nav.view('left'), row: 'view.numpad', place: 'The view menu' },
    { id: 'view.top', label: 'Top', keys: ['Numpad7'], group: 'The view', tier: 'occasional', menu: 'view', menuOrder: 0, run: () => nav.view('top'), row: 'view.numpad', place: 'The view menu' },
    { id: 'view.bottom', label: 'Bottom', keys: ['Ctrl+Numpad7'], group: 'The view', tier: 'occasional', run: () => nav.view('bottom'), row: 'view.numpad', place: 'The view menu' },
    { id: 'view.three', label: 'Three-quarter', group: 'The view', tier: 'occasional', menu: 'view', menuOrder: 4, run: () => nav.angle('three'), place: 'The view menu', help: false },
    { id: 'view.face', label: 'Face', group: 'The view', tier: 'occasional', menu: 'view', menuOrder: 1, run: () => nav.frame('face'), place: 'The view menu', help: false },
    { id: 'view.hand-r', label: 'Right hand', group: 'The view', tier: 'occasional', menu: 'view', menuOrder: 3, run: () => nav.frameHand('R'), place: 'The view menu', help: false },
    { id: 'view.hand-l', label: 'Left hand', group: 'The view', tier: 'occasional', menu: 'view', menuOrder: 7, run: () => nav.frameHand('L'), place: 'The view menu', help: false },
    { id: 'view.upper', label: 'Upper body', group: 'The view', tier: 'occasional', run: () => nav.frame('upper'), place: 'The view bar', help: false },
    { id: 'view.turntable', label: 'Turn the figure', desc: 'The figure turns on its own until you move the view', group: 'The view', tier: 'rare',
      run: () => app.setTurntable(!app.turntable()), checked: () => app.turntable(), place: 'The view bar', icon: 'turntable', help: false },

    // ---------------------------------------------------------------- the display
    // the display menu's places, clockwise from the top (studio/menus.ts): the figure, the scan and its see-through in
    // the top right (Pose › Edit adds them), the skin weights, the hair and the skeleton below, X-ray on the left
    { id: 'display.menu', label: 'Display menu', desc: 'X-ray, the figure, the scan, the skeleton, the skin weights and the hair round the pointer, as switches', keys: ['Z'],
      group: 'The view', tier: 'frequent', place: 'The view', icon: 'display', run: () => d.ringMenu('display') },
    { id: 'display.xray', label: 'X-ray', icon: 'xray', desc: 'The figure see-through, so points behind it show', keys: ['Alt+Z'], group: 'The view', tier: 'frequent',
      menu: 'display', menuOrder: 6, place: 'The display menu', checked: () => cam().xrayOn(), run: () => cam().xray(!cam().xrayOn()) },
    { id: 'display.skeleton', label: 'Skeleton', desc: 'The bones over the figure; it shares the rig view with the skin weights', icon: 'bones', group: 'Display', tier: 'occasional',
      menu: 'display', menuOrder: 5, help: false, place: 'The display menu',
      checked: () => app.rigView() === 'skeleton', run: () => app.setRigView(app.rigView() === 'skeleton' ? 'off' : 'skeleton') },
    { id: 'display.weights', label: 'Skin weights', desc: 'The skin coloured by a bone\'s weights; it shares the rig view with the skeleton', icon: 'display', group: 'Display', tier: 'occasional',
      menu: 'display', menuOrder: 3, help: false, place: 'The display menu',
      checked: () => app.rigView() === 'weights', run: () => app.setRigView(app.rigView() === 'weights' ? 'off' : 'weights') },
    { id: 'display.hair', label: 'Hair', icon: 'hair', group: 'Display', tier: 'occasional', menu: 'display', menuOrder: 4, help: false, place: 'The display menu',
      checked: () => app.hairVisible(), run: () => app.setHairVisible(!app.hairVisible()) },

    { id: 'part.hide', label: 'Hide the selected part', keys: ['H'], group: 'The view', tier: 'frequent', row: 'part.hide', rowLabel: 'Hide the selected part, hide the rest, show all',
      place: 'The view', icon: 'eyeOff', when: hasPart, hint: 'Select a part first',
      run: () => { for (const b of selBones()) parts.hidden.add(b); applyParts(); d.toast('Hid ' + (selection()?.label || 'the part')); } },
    { id: 'part.hide-rest', label: 'Hide everything else', keys: ['Shift+H'], group: 'The view', tier: 'frequent', row: 'part.hide', place: 'The view', icon: 'eyeOff',
      when: hasPart, hint: 'Select a part first',
      run: () => { const keep = new Set(selBones()); for (const n of app.boneNames()) if (!keep.has(n)) parts.hidden.add(n); applyParts(); } },
    { id: 'part.show', label: 'Show everything', keys: ['Alt+H'], group: 'The view', tier: 'frequent', row: 'part.hide', place: 'The view', icon: 'eye',
      when: () => parts.hidden.size > 0 || !!parts.isolated, hint: 'Nothing is hidden',
      run: () => { parts.hidden.clear(); parts.isolated = null; applyParts(); } },
    { id: 'part.isolate', label: 'Isolate the selected part', desc: 'The rest turns faint; again to leave', keys: ['/'], group: 'The view', tier: 'frequent', place: 'The view',
      icon: 'focus', when: () => hasPart() || !!parts.isolated, hint: 'Select a part first', checked: () => !!parts.isolated,
      run: () => { parts.isolated = parts.isolated ? null : selBones(); applyParts(); } },
    { id: 'panel.controls', label: 'Controls', desc: 'Turn, move and zoom speeds, the drag speed of points, Larger targets', group: 'Panels', tier: 'rare', place: 'More › Controls',
      icon: 'mouse', help: false, run: () => uiActions.toggleTool('controls') },

    // ---------------------------------------------------------------- the panels and the animation
    { id: 'panel.right', label: 'Right panel', desc: 'The Character panel or the tool panel on or off', keys: ['N'], group: 'Panels', tier: 'occasional',
      row: 'panel.both', rowLabel: 'Right panel, left panel', place: 'The view',
      run: () => { const s = ui.get(); if (s.tool) uiActions.closeTool(); else uiActions.toggleCharacter(); } },
    { id: 'panel.left', label: 'Left panel', desc: 'The mode’s panel on or off', keys: ['T'], group: 'Panels', tier: 'occasional', row: 'panel.both', place: 'The view',
      run: () => { const s = ui.get(); if (s.open) uiActions.closePanel(); else uiActions.openMode(s.mode); } },
    { id: 'panel.full', label: 'The view fills the window', desc: 'Again to bring the panels back', keys: ['Ctrl+Space'], group: 'Panels', tier: 'occasional', place: 'The view', run: fullView },
    { id: 'motion.play', label: 'Play or pause', keys: ['Space'], group: 'Panels', tier: 'occasional', place: 'Pose', when: motionLoop, hint: 'No animation is playing',
      run: () => app.togglePlay() },
    ...Object.keys(app.presets).slice(0, 4).map((n, i): ActionDef => ({
      id: 'light.' + n, label: app.presets[n].label + ' lighting', keys: ['Alt+' + (i + 1)], group: 'Display', tier: 'rare', row: 'light.presets', rowLabel: 'Lighting presets',
      run: () => app.applyPreset(n), place: 'The Scene panel',
    })),

    // ---------------------------------------------------------------- help
    { id: 'help.commands', label: 'Command list', desc: 'Every action by name, with its key', keys: ['F3', 'Ctrl+K'], group: 'Help', tier: 'rare', inText: true, place: 'The top bar',
      icon: 'search', run: () => d.commands() },
    // F1 works in a text field too; ? types itself there
    { id: 'help.keys', label: 'This help', desc: 'The keys and the mouse by use and by mode', keys: ['F1'], group: 'Help', tier: 'rare', place: 'The top bar', icon: 'help',
      inText: true, row: 'help.keys', rowLabel: 'This help', run: () => d.help() },
    { id: 'help.keys-q', label: 'This help', keys: ['?'], group: 'Help', tier: 'rare', row: 'help.keys', command: false, run: () => d.help() },
    { id: 'help.escape', label: 'Close the menu or the panel', keys: ['Escape'], group: 'Help', tier: 'constant', help: false, command: false, run: () => d.escape() },
  ];
  offs.push(registerActions(A));
  return () => offs.forEach((f) => f());
}

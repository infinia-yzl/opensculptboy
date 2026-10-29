// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The interface of the viewer. The frame (the top bar, the six modes, the panels, the view bar, the welcome card and
// the Open dialog) is React (src/shell); this file builds what the frame hosts and drives: the editor and its
// history, the sections that are still the page's own DOM (character_ui.ts, stage_ui.ts, rig_ui.ts, sheet_ui.ts,
// expression_ui.ts, wearables_ui.ts), the camera's navigation and height rail, the performance card, the help sheet,
// the name of the bone under the pointer, the refinement ring, and the keyboard shortcuts of shortcuts.ts.
// main.ts calls initUI once the model is ready, and calls the returned hooks when the look, the motion or a frame
// changes; the hooks reach the React interface through the bus of src/shell/state.ts, never by a React render.

import type { App, Hooks } from './app.ts';
import { cameraNav, type CameraNav } from './camera_nav.ts';
import { characterSections, randomBodyInto, randomColours, randomFaceInto, randomHair, randomPhenotype, type Editor } from './character_ui.ts';
import { download, h, iconButton, slug, typing } from './dom.ts';
import { lookHistory } from './history.ts';
import { helpSheet, MODES, SECTIONS, shortcutKeys, TOOLS } from './shortcuts.ts';
import { sheetSection } from './sheet_ui.ts';
import { stageSections } from './stage_ui.ts';
import { expressionSection } from './expression_ui.ts';
import { wearablesSection, type WearablesUI } from './wearables_ui.ts';
import { makeStats, type Stats } from './stats.ts';
import { toast } from './toast.ts';
import { actions, bus, ui, widthOf } from '../shell/state.ts';
import { makeOpener } from '../shell/open.ts';
import { mountShell } from '../shell/Shell.tsx';
import type { Legacy } from '../shell/context.ts';

export interface UI { hooks: Hooks; nav: CameraNav; stats: Stats }

const rgb = (c: number[]) => `rgb(${c.map((v) => Math.round(v * 255)).join(',')})`;

export function initUI(app: App): UI {
  document.body.classList.toggle('shot', app.shot);
  const nav = cameraNav(app);
  const stats = makeStats(app);

  // ---------------------------------------------------------------- the edits and their history
  let pending: ((n: any) => void)[] = [];
  const hist = lookHistory(() => app.look(), (l) => app.applyLook(l));
  const ed: Editor = {
    app, origin: app.look().name, faceSpread: 1,
    begin: (label) => hist.begin(label),
    end: () => hist.end(),
    edit(label, mut) { hist.begin(label); app.editLook(mut); hist.end(); },
    commit(label, look) { hist.begin(label); app.applyLook(look); hist.end(); },
    queue(mut) {
      if (!pending.length) requestAnimationFrame(() => { const ms = pending; pending = []; app.editLook((n) => { for (const m of ms) m(n); }); });
      pending.push(mut);
    },
  };
  const undo = () => { const l = hist.undo(); if (l) toast('Undo: ' + l); };
  const redo = () => { const l = hist.redo(); if (l) toast('Redo: ' + l); };
  const randomAll = () => {
    const next = JSON.parse(JSON.stringify(app.look()));
    next.phenotype = randomPhenotype(app);
    if (app.facePrior()) next.face = app.sampleFace(next.phenotype, ed.faceSpread);
    randomHair(app, next); randomColours(app, next); next.name = 'Custom';
    ed.commit('Random character', next); ed.origin = 'Random character';
  };
  hist.onChange = () => ui.set({ history: { undo: hist.canUndo() ? hist.nextUndo() : null, redo: hist.canRedo() ? hist.nextRedo() : null } });
  hist.onChange();

  // ---------------------------------------------------------------- the sections that the frame hosts
  const layoutSoon = () => requestAnimationFrame(layout);
  const cs = characterSections(ed);
  const ss = stageSections(app);
  const sheet = sheetSection(app, layoutSoon);
  const xs = expressionSection();
  let wear: WearablesUI;
  const opener = makeOpener({
    loadCard: (picked) => wear.load(picked),
    applyLook: (json) => { const look = app.normaliseLook(json); ed.commit('Open ' + look.name, look); ed.origin = look.name; return look.name; },
    setSheet: (json) => { (window as any).sheet.set(json); },
    posePicture: async (file) => { await app.poseFromPicture(file); },
  });
  wear = wearablesSection(app, opener);
  ui.set({ hasClips: app.clips().length > 0 });

  // ---------------------------------------------------------------- the page's actions
  // a hosted page (the artifact build, with its data in parts) cannot start a download, so it shows the picture to
  // save by hand; the single-file page downloads it
  const hosted = !!(window as any).MODEL_PARTS;
  const shotDlg = h('dialog.help.shot-dlg', { 'aria-label': 'Saved picture' }) as HTMLDialogElement;
  shotDlg.addEventListener('click', (e) => { if (e.target === shotDlg) shotDlg.close(); });
  const saveImage = () => {
    app.renderPass();
    app.canvas.toBlob((b) => {
      if (!b) { toast('The picture could not be made', 'error'); return; }
      const name = `opensculptboy-${slug(app.look().name)}.png`;
      if (!hosted) { download(name, b); toast('Saved the picture'); return; }
      const url = URL.createObjectURL(b);
      const close = iconButton('close', 'Close', { key: 'Esc', tip: 'left', on: () => shotDlg.close() });
      shotDlg.replaceChildren(h('header.help-head', {}, h('h2.osb-title', { text: 'Picture' }), close),
        h('p.help-lead', { text: 'Right-click the picture and choose Save image, or press and hold it on a phone.' }),
        h('img.shot-img', { src: url, alt: 'The current view of ' + app.look().name }));
      shotDlg.addEventListener('close', () => URL.revokeObjectURL(url), { once: true });
      shotDlg.showModal();
    }, 'image/png');
  };
  const fileName = () => slug(app.look().name || 'character');
  const help = helpSheet();
  stats.onToggle = (on) => ui.set({ stats: on });
  const fullscreen = () => {
    if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen().catch(() => toast('Full screen is not allowed here', 'error'));
  };

  // ---------------------------------------------------------------- the refinement ring of the view bar
  const ns = 'http://www.w3.org/2000/svg';
  const ringSvg = document.createElementNS(ns, 'svg'); ringSvg.setAttribute('viewBox', '0 0 24 24'); ringSvg.setAttribute('aria-hidden', 'true');
  ringSvg.innerHTML = '<circle cx="12" cy="12" r="8" class="bg"/><circle cx="12" cy="12" r="8" class="fg" pathLength="100" stroke-dasharray="0 100"/>';
  const ringLabel = h('span.ring-label');
  const ring = h('button.osb-ring.ringb', { type: 'button', 'data-tip': 'Refinement: open Performance', 'data-tip-pos': 'up', 'aria-live': 'polite' }, ringSvg, ringLabel);
  ring.addEventListener('click', () => stats.toggle());
  let lastRing = '';
  const paintRing = () => {
    const s = app.samples(), m = app.motion();
    const moving = !!m.cur && m.cur.kind === 'loop' && m.playing;
    const done = s.n >= s.max;
    const txt = moving ? 'Playing' : app.turntable() ? 'Turning' : done ? 'Refined' : `${Math.round(s.n / s.max * 100)} %`;
    const key = txt + (done ? 1 : 0);
    if (key === lastRing) return;
    lastRing = key;
    ringLabel.textContent = txt;
    (ringSvg.querySelector('.fg') as SVGElement).setAttribute('stroke-dasharray', `${moving || app.turntable() ? 100 : Math.min(100, s.n / s.max * 100)} 100`);
    ring.dataset.state = moving || app.turntable() ? 'live' : done ? 'done' : 'busy';
    ring.setAttribute('aria-label', `Picture ${txt.toLowerCase()}. Open the performance card.`);
  };
  // the view bar's state: changes only on a new framing, angle or turntable (the store drops equal values)
  const paintView = () => {
    const a = Object.entries({ front: 0, three: 24, side: 90, back: 180 }).find(([, y]) => y === app.view.yaw);
    ui.set({ view: { frame: app.view.userMoved ? '' : app.currentFrame(), angle: a ? a[0] : '', turntable: app.turntable() } });
  };
  nav.onView = paintView;

  // ---------------------------------------------------------------- the bone under the pointer
  // While the rig shows (the Rig panel), the name of the bone under the pointer follows it. In the weight view,
  // pointing at a bone shows its weights; a click keeps a bone, and a click beside the bones clears it.
  const boneTip = h('div.bone-tip', { 'aria-hidden': 'true', hidden: true });
  let tipAt: { x: number; y: number } | null = null, tipQueued = false;
  const paintBoneTip = () => {
    tipQueued = false;
    const name = tipAt && app.rigView() !== 'off' ? app.pickBone(tipAt.x, tipAt.y) : null;
    boneTip.hidden = !name;
    if (name && app.rigView() === 'weights' && !app.rigBoneKept() && name !== app.rigBone()) app.setRigBone(name);
    if (!name || !tipAt) return;
    boneTip.textContent = name;
    boneTip.style.left = tipAt.x + 'px'; boneTip.style.top = tipAt.y + 'px';
  };
  app.canvas.addEventListener('pointermove', (e) => {
    // a drag turns the camera: the name waits until the pointer rests on a bone again
    if (e.buttons || e.pointerType === 'touch') { tipAt = null; boneTip.hidden = true; return; }
    tipAt = { x: e.clientX, y: e.clientY };
    if (!tipQueued && app.rigView() !== 'off') { tipQueued = true; requestAnimationFrame(paintBoneTip); }
  });
  app.canvas.addEventListener('pointerleave', () => { tipAt = null; boneTip.hidden = true; });
  // a click is a press without a drag
  let press: { x: number; y: number; t: number } | null = null;
  app.canvas.addEventListener('pointerdown', (e) => { press = { x: e.clientX, y: e.clientY, t: performance.now() }; });
  app.canvas.addEventListener('pointerup', (e) => {
    const p = press; press = null;
    if (!p || app.rigView() === 'off' || Math.hypot(e.clientX - p.x, e.clientY - p.y) > 5 || performance.now() - p.t > 400) return;
    const name = app.pickBone(e.clientX, e.clientY);
    app.setRigBone(name, !!name);
  });

  // ---------------------------------------------------------------- the character shown
  const identity = () => {
    const L = app.look(), w = app.wardrobe();
    let name = L.name;
    const named = app.characters.concat(app.looks).some((c) => c.name === ed.origin);
    if (L.name === 'Custom' && named && ed.origin !== 'Baseline') name = ed.origin + ' · edited';
    let kind = '';
    if (w?.loaded) { name = w.cardName; kind = (w as any).card?.detail || (w as any).card?.material ? 'Fitted card' : 'Your card'; }
    return { name, kind, a: rgb(app.skinBase(L.skin.tone, L.skin.undertone)), b: L.hair.color };
  };
  app.wardrobe()?.listeners.push(() => bus.emit('wardrobe'));

  // ---------------------------------------------------------------- the frame (React)
  const legacy: Legacy = {
    sections: { characters: cs.chars, face: cs.face, hair: cs.hair, wear: wear.el, expression: xs.el, pose: ss.pose, sheet: sheet.el, rig: ss.rig, scene: ss.scene },
    sheetView: sheet.view, sheetRefresh: () => sheet.refresh(),
    undo, redo,
    random: {
      character: randomAll,
      body: () => { randomBodyInto(app, ed, app.look()); ed.origin = 'Random body'; },
      face: () => randomFaceInto(app, ed),
      hair: () => ed.edit('Random hair', (n) => randomHair(app, n)),
      colours: () => ed.edit('Random colours', (n) => randomColours(app, n)),
    },
    reset: {
      all: () => { ed.commit('Reset all', app.baseline()); ed.origin = 'Baseline'; },
      body: () => ed.edit('Reset body', (n) => { n.phenotype = app.baselinePhenotype(); }),
      face: () => ed.edit('Reset face', (n) => { n.face = {}; }),
    },
    saveImage,
    poseCard: () => download(`${fileName()}-pose.json`, new Blob([JSON.stringify(app.poseCard(), null, 2) + '\n'], { type: 'application/json' })),
    silhouette: () => download(`${fileName()}-silhouette.svg`, new Blob([app.poseSilhouette()], { type: 'image/svg+xml' })),
    downloadLook: () => { const L = app.look(); download(`opensculptboy-${slug(L.name)}.json`, new Blob([JSON.stringify(L, null, 2)], { type: 'application/json' })); },
    toggleStats: () => stats.toggle(), showHelp: () => help.showModal(), fullscreen,
    frame: (v) => nav.frame(v), angle: (v) => nav.angle(v), turntable: () => app.setTurntable(!app.turntable()), resetView: () => nav.reset(),
    ring, heightRail: nav.rail, identity, opener, hosted,
  };
  document.body.append(sheet.view, nav.rail, stats.el, help, shotDlg, boneTip);
  mountShell({ app, ed, legacy });

  // ---------------------------------------------------------------- the layout
  // the free space of the 3D view lies between the mode's panel (or the rail) on the left and the tool panel on the
  // right; the camera centres the figure in it, and the view bar sits under its middle
  function layout() {
    const s = ui.get(), B = document.body;
    B.dataset.width = s.width; B.dataset.mode = s.mode;
    B.classList.toggle('panel-open', s.open); B.classList.toggle('tool-open', !!s.tool);
    sheet.setOpen(s.mode === 'sheet' && s.open);
    const phone = s.width === 'phone';
    B.classList.toggle('sheet-open', phone && (s.open || !!s.tool));
    if (phone || app.shot) { nav.setInsets(0, 0); B.classList.remove('vb-compact'); return; }
    const r = (sel: string) => { const el = document.querySelector(sel) as HTMLElement | null; return el && !el.hidden ? el.getBoundingClientRect() : null; };
    const lr = [r('#insp-left'), r('.mode-rail')].filter(Boolean) as DOMRect[], rr = [r('#insp-right')].filter(Boolean) as DOMRect[];
    const l = lr.length ? Math.max(...lr.map((x) => x.right)) : 0, rt = rr.length ? innerWidth - Math.min(...rr.map((x) => x.left)) : 0;
    nav.setInsets(l, rt);
    B.classList.toggle('vb-compact', innerWidth - l - rt < 640);
  }
  ui.subscribe(layoutSoon);
  addEventListener('resize', () => { ui.set({ width: widthOf(innerWidth) }); layout(); });
  // a drag of files over the page outlines the canvas
  ui.subscribe(() => document.body.classList.toggle('osb-dragging', ui.get().dragging));

  // ---------------------------------------------------------------- the keyboard
  const dialogOpen = () => !!document.querySelector('dialog[open], [role="dialog"][data-state="open"], [role="menu"][data-state="open"]');
  document.addEventListener('keydown', (e) => {
    if (e.defaultPrevented) return;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && !e.altKey && (e.key === 'z' || e.key === 'Z' || e.key === 'y')) {
      if (typing(e)) return;
      e.preventDefault();
      if (e.key === 'y' || e.shiftKey) redo(); else undo();
      return;
    }
    if (mod && !e.altKey && !e.shiftKey && (e.key === 'o' || e.key === 'O' || e.code === 'KeyO')) { e.preventDefault(); actions.showOpen(); return; }
    if (typing(e)) {
      // Esc leaves a text field, so the shortcuts work again (a slider's value box and a search field put back their
      // text first)
      if (e.key === 'Escape' && !mod) (e.target as HTMLElement).blur();
      return;
    }
    if (mod) return;
    if (dialogOpen()) return;
    if (e.altKey) {
      const m = /^Digit([1-4])$/.exec(e.code);
      if (m) { const names = Object.keys(app.presets); const n = names[+m[1] - 1]; if (n) { e.preventDefault(); app.applyPreset(n); } }
      return;
    }
    const act: Record<string, () => void> = {
      ArrowLeft: () => nav.orbit(-8), ArrowRight: () => nav.orbit(8),
      ArrowUp: () => e.shiftKey ? nav.tilt(5) : nav.move(0.05), ArrowDown: () => e.shiftKey ? nav.tilt(-5) : nav.move(-0.05),
      '+': () => nav.zoom(0.88), '=': () => nav.zoom(0.88), '-': () => nav.zoom(1.14), '_': () => nav.zoom(1.14),
      '0': () => nav.reset(), '1': () => nav.frame('body'), '2': () => nav.frame('upper'), '3': () => nav.frame('face'),
      '4': () => nav.angle('front'), '5': () => nav.angle('three'), '6': () => nav.angle('side'), '7': () => nav.angle('back'),
      H: () => app.setHairVisible(!app.hairVisible()), t: () => app.setTurntable(!app.turntable()),
      B: () => app.setRigView(app.rigView() === 'skeleton' ? 'off' : 'skeleton'),
      W: () => app.setRigView(app.rigView() === 'weights' ? 'off' : 'weights'),
      '`': () => stats.toggle(), '?': () => help.showModal(),
      ' ': () => { const m = app.motion(); if (m.cur && m.cur.kind === 'loop') app.togglePlay(); },
      Escape: () => {
        const a = document.activeElement, s = ui.get();
        const left = document.getElementById('insp-left'), right = document.getElementById('insp-right');
        if (s.tool && right?.contains(a)) { actions.closeTool(); (document.querySelector('[aria-label="View menu"]') as HTMLElement)?.focus(); }
        else if (s.open && left?.contains(a)) { actions.closePanel(); document.getElementById('tab-' + s.mode)?.focus(); }
        else if (s.tool) actions.closeTool();
        else if (s.open) actions.closePanel();
        else if (s.welcome) actions.dismissWelcome();
        else if (stats.isOpen()) stats.close();
      },
    };
    // the modes, Create's sections and the tools of the View menu (shortcuts.ts)
    for (const m of MODES) act[m.key] = () => actions.toggleMode(m.id);
    for (const s of SECTIONS.create) if (s.key) act[s.key] = () => actions.toggleSection('create', s.id);
    for (const t of TOOLS) act[t.key!] = () => actions.toggleTool(t.id as any);
    const t = e.target as HTMLElement;
    const keys = shortcutKeys(e);
    // a focused button keeps Space and Enter; the arrows stay with the camera only outside the controls that use them
    if (keys[0] === ' ' && t.closest('button, summary')) return;
    if (keys[0]?.startsWith('Arrow') && t.closest('input, .seg, .osb-seg, [role="tablist"], [role="radiogroup"], [role="menu"]')) return;
    // Shift with a panel's letter opens it as the letter alone does, where Shift gives the letter no meaning of its own
    const k = keys.find((x) => act[x]) ?? keys.map((x) => x.length === 1 ? x.toLowerCase() : x).find((x) => act[x]);
    if (!k) return;
    e.preventDefault();
    act[k]();
  });

  // ---------------------------------------------------------------- the start state
  if (!app.shot && app.qs.get('stats') === '1') stats.open();
  layout();
  const syncLook = () => {
    const L = app.look();
    cs.sync(L);
    if (L.name !== 'Custom') ed.origin = L.name;
    bus.emit('look');
  };
  syncLook(); ss.sync(); xs.sync(); paintView(); paintRing();
  let tFrame = 0;
  const hooks: Hooks = {
    look: syncLook,
    motion: () => { ss.sync(); if (app.rigView() === 'off') boneTip.hidden = true; bus.emit('motion'); },
    hair: () => { cs.sync(app.look()); ss.sync(); bus.emit('hair'); },
    face: () => { xs.sync(); bus.emit('face'); },
    frame(now, dt, drew, jsMs) {
      stats.frame(now, dt, drew, jsMs);
      if (now - tFrame > 100) { tFrame = now; ss.frame(); paintRing(); paintView(); }
    },
    passBegin: () => stats.passBegin(),
    passEnd: () => stats.passEnd(),
    interacted: () => { /* the welcome card stays until it is closed */ },
  };
  return { hooks, nav, stats };
}

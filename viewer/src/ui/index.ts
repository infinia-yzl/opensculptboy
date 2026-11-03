// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The interface of the viewer. The frame (the top bar, the six modes, the panels, the view bar, the welcome card and
// the Open dialog) is React (src/shell); this file builds what the frame hosts and drives: the editor and its
// history, the sections that are still the page's own DOM (character_ui.ts, stage_ui.ts, rig_ui.ts, sheet_ui.ts,
// expression_ui.ts, wearables_ui.ts), the camera's navigation and height rail, the performance card, the help sheet,
// the name of the bone under the pointer, the refinement ring, and the keys of key map version 2 (studio/keymap.ts in
// the registry of studio/actions.ts), with the pointer's arbitration (studio/pointer.ts) and the status line.
// main.ts calls initUI once the model is ready, and calls the returned hooks when the look, the motion or a frame
// changes; the hooks reach the React interface through the bus of src/shell/state.ts, never by a React render.

import type { App, Hooks } from './app.ts';
import { cameraNav, type CameraNav } from './camera_nav.ts';
import { randomBodyInto, randomColours, randomFaceInto, randomHair, randomPhenotype, type Editor } from './character_ui.ts';
import { download, h, iconButton, slug, typing } from './dom.ts';
import { lookHistory } from './history.ts';
import { sheetController } from '../sheet/controller.ts';
import { stageSections } from './stage_ui.ts';
import { wearLoader } from '../shell/sections/DressSection.tsx';
import { cardOf, writeZip } from '../card/file.ts';
import { STARTERS, fetchStarter, starterById } from '../starters.ts';
import { stoneOf } from '../stones.ts';
import { makeStats, type Stats } from './stats.ts';
import { toast } from './toast.ts';
import { actions, bus, ui, widthOf } from '../shell/state.ts';
import { makeOpener } from '../shell/open.ts';
import { mountShell } from '../shell/Shell.tsx';
import type { Legacy } from '../shell/context.ts';
import { flushSync } from 'react-dom';
import { allActions, dispatchKey, onActionRun } from '../studio/actions.ts';
import { cam } from '../studio/camera.ts';
import { setSelection } from '../studio/selection.ts';
import { statusText } from '../studio/status.ts';
import { modalActive } from '../studio/modal.ts';
import { offerTour, startTour } from '../studio/Onboarding.tsx';
import { settings } from '../studio/settings.ts';
import { helpSheet } from '../studio/help.ts';
import { registerKeymap } from '../studio/keymap.ts';
import { commandList, ringMenu } from '../studio/menus.ts';
import { installPointer, registerPicker } from '../studio/pointer.ts';
import { PRIORITY, setStatus, trackModifiers } from '../studio/status.ts';

export interface UI { hooks: Hooks; nav: CameraNav; stats: Stats; firstStarter: (id: string) => Promise<void> }

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
  const ss = stageSections(app);
  const sheet = sheetController(app, layoutSoon);
  const opener = makeOpener({
    loadCard: wearLoader(app),
    applyLook: (json) => { const look = app.normaliseLook(json); ed.commit('Open ' + look.name, look); ed.origin = look.name; return look.name; },
    setSheet: (json) => { (window as any).sheet.set(json); },
    posePicture: async (file) => { await app.poseFromPicture(file); },
  });
  ui.set({ hasClips: app.clips().length > 0 });

  // ---------------------------------------------------------------- the character as one file
  // Save card: the open card with the page's edits (its look block, schema 5), as .json, or with its files as a .zip
  const saveCard = async () => {
    const w = app.wardrobe(), L = app.look();
    const base = w?.loaded ? w.source : null;
    const card = cardOf(L, base);
    const name = slug(L.name || card.name || 'character');
    const files = w?.loaded && w.source ? w.files : null;
    // a card opened without files of its own saves as .json
    const others = files ? [...files.byPath.keys()].filter((p) => p !== w!.cardPath) : [];
    if (!files || !others.length) {
      download(`${name}.json`, new Blob([JSON.stringify(card, null, 2) + '\n'], { type: 'application/json' }));
      toast('Saved the card');
      return;
    }
    // the card's own files beside it, at their paths relative to the card
    const folder = w!.cardPath.includes('/') ? w!.cardPath.slice(0, w!.cardPath.lastIndexOf('/')) : '';
    const entries: { path: string; data: Uint8Array }[] = [{ path: 'character.json', data: new TextEncoder().encode(JSON.stringify(card, null, 2) + '\n') }];
    for (const [path, file] of files.byPath as Map<string, File>) {
      if (path === w!.cardPath) continue;
      const rel = folder && path.startsWith(folder + '/') ? path.slice(folder.length + 1) : path;
      entries.push({ path: rel, data: new Uint8Array(await file.arrayBuffer()) });
    }
    download(`${name}.zip`, new Blob([writeZip(entries) as BlobPart], { type: 'application/zip' }));
    toast(`Saved the card with ${entries.length - 1} files`);
  };
  // a statue of the New screen: its package from beside the page (starters.ts checks it against the page's pins),
  // opened through the Open path as a card's .zip; a package that is missing or different leaves a note with its
  // download link for the New screen and the welcome card
  const startStarter = async (id: string): Promise<boolean> => {
    const s = starterById(id);
    if (!s) return false;
    actions.setStarter({ busy: id, note: null });
    const r = await fetchStarter(s);
    if ('note' in r) { actions.setStarter({ busy: '', note: r.note }); return false; }
    actions.setStarter({ busy: '', note: null });
    const done = await opener.open([{ path: r.file.name, file: r.file }]);
    if (done && ui.get().welcome) actions.dismissWelcome();
    return done;
  };
  // a first visit opens the default statue when its package lies beside the page; otherwise the welcome card shows
  // the note with its download link
  const firstStarter = async (id: string) => {
    const s = starterById(id);
    if (!s) return;
    const r = await fetchStarter(s);
    if ('note' in r) { actions.setStarter({ busy: '', note: r.note }); return; }
    await opener.open([{ path: r.file.name, file: r.file }], { quiet: true });
  };
  const startBase = () => {
    app.wardrobe()?.unload();
    ed.commit('Start from the base', app.baseline()); ed.origin = 'Baseline';
    if (app.clips().some((c) => c.name === 'a_pose')) app.setMotion('a_pose');
    actions.hideNew();
  };

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
  const cmds = commandList();
  const rings = ringMenu();
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
  // the angle lights only while the camera is at it, so a free turn leaves none lit
  const paintView = () => {
    const st = nav.api.state(), yaw = st.yaw;
    const a = st.flying ? null : Object.entries({ front: 0, three: 24, side: 90, back: 180, right: -90 })
      .find(([, y]) => Math.abs(((yaw - y) % 360 + 540) % 360 - 180) < 1 && Math.abs(st.tilt) < 20);
    const frame = app.view.lit || (app.view.userMoved ? '' : app.currentFrame());
    ui.set({ view: { frame, angle: a ? (a[0] === 'right' ? 'side' : a[0]) : '', turntable: app.turntable() } });
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
  // a click keeps a bone, and a click beside the bones clears it (a picker of the pointer's arbitration)
  registerPicker({
    id: 'rig.bone', priority: 0,
    enabled: () => app.rigView() !== 'off',
    hit: (p) => { const name = app.pickBone(p.x, p.y); return name ? { item: name, distPx: 0, label: name } : null; },
    onClick: (hit) => app.setRigBone(hit.item as string, true),
    onBackground: () => app.setRigBone(null, false),
  });

  // ---------------------------------------------------------------- the character shown
  const identity = () => {
    const L = app.look(), w = app.wardrobe();
    let name = L.name;
    const named = app.looks.some((c) => c.name === ed.origin);
    if (L.name === 'Custom' && named && ed.origin !== 'Baseline') name = ed.origin + ' · edited';
    let kind = '';
    if (w?.loaded) { name = w.cardName; kind = (w as any).card?.detail || (w as any).card?.material ? 'Fitted card' : 'Your card'; }
    const st = stoneOf(L.skin);
    return { name, kind, a: st ? st.base : rgb(app.skinBase(L.skin.tone, L.skin.undertone)), b: L.hair.color };
  };
  app.wardrobe()?.listeners.push(() => bus.emit('wardrobe'));

  // ---------------------------------------------------------------- the frame (React)
  const legacy: Legacy = {
    sections: { rig: ss.rig, scene: ss.scene },
    sheet,
    saveCard, startStarter, startBase, starters: STARTERS,
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
    toggleStats: () => stats.toggle(), showHelp: () => help.open(), fullscreen,
    frame: (v) => nav.frame(v), angle: (v) => nav.angle(v), turntable: () => app.setTurntable(!app.turntable()), resetView: () => nav.reset(),
    ring, heightRail: nav.rail, identity, opener, hosted,
  };
  document.body.append(sheet.view, nav.pivotDot, stats.el, help.el, cmds.el, rings.el, shotDlg, boneTip);
  // the 3D view's backdrop follows the theme: the dark studio, or the light theme's marble stage
  const stage = () => app.setStage(ui.get().dark ? 'dark' : 'light');
  ui.subscribe(stage); stage();

  // ---------------------------------------------------------------- the layout
  // the free space of the 3D view lies between the mode's panel (or the rail) on the left and the tool panel on the
  // right; the camera centres the figure in it, and the view bar sits under its middle
  function layout() {
    const s = ui.get(), B = document.body;
    B.dataset.width = s.width; B.dataset.mode = s.mode;
    B.classList.toggle('panel-open', s.open); B.classList.toggle('tool-open', !!s.tool);
    B.classList.toggle('char-open', s.character);
    sheet.setOpen(s.mode === 'sheet' && s.open);
    const phone = s.width === 'phone';
    B.classList.toggle('sheet-open', phone && (s.open || !!s.tool));
    if (app.shot) { nav.setInsets(0, 0); B.classList.remove('vb-compact'); return; }
    // the bars over the canvas: the framings fill the height between the top bar and the view bar (or the phone bar)
    const bar = (sel: string) => { const el = document.querySelector(sel) as HTMLElement | null; const r = el ? el.getBoundingClientRect() : null; return r && r.height ? r : null; };
    const topR = bar('.osb-top'), lows = ['.viewbar', '.phone-bar'].map(bar).filter(Boolean) as DOMRect[];
    nav.setVInsets(topR ? topR.bottom + 8 : 0, lows.length ? innerHeight - Math.min(...lows.map((r) => r.top)) + 8 : 0);
    if (phone) { nav.setInsets(0, 0); B.classList.remove('vb-compact'); return; }
    const r = (sel: string) => { const el = document.querySelector(sel) as HTMLElement | null; return el && !el.hidden ? el.getBoundingClientRect() : null; };
    const lr = [r('#insp-left'), r('.mode-rail')].filter(Boolean) as DOMRect[], rr = [r('#insp-right'), r('.char-panel')].filter(Boolean) as DOMRect[];
    const l = lr.length ? Math.max(...lr.map((x) => x.right)) : 0, rt = rr.length ? innerWidth - Math.min(...rr.map((x) => x.left)) : 0;
    nav.setInsets(l, rt);
    B.classList.toggle('vb-compact', innerWidth - l - rt < 640);
  }
  ui.subscribe(layoutSoon);
  addEventListener('resize', () => { ui.set({ width: widthOf(innerWidth) }); layout(); });
  // a drag of files over the page outlines the canvas
  ui.subscribe(() => document.body.classList.toggle('osb-dragging', ui.get().dragging));

  // ---------------------------------------------------------------- the keyboard and the mouse
  // the key map version 2 (studio/keymap.ts) in the registry of studio/actions.ts; one handler for every key
  const dialogOpen = () => !!document.querySelector('dialog[open], [role="dialog"][data-state="open"], [role="menu"][data-state="open"]');
  let pointerAt = { x: innerWidth / 2, y: innerHeight / 2 };
  addEventListener('pointermove', (e) => { pointerAt = { x: e.clientX, y: e.clientY }; }, { passive: true });
  const escape = () => {
    const a = document.activeElement, s = ui.get();
    const left = document.getElementById('insp-left'), right = document.getElementById('insp-right');
    if (s.tool && right?.contains(a)) { actions.closeTool(); (document.querySelector('[data-menu="more"]') as HTMLElement)?.focus(); }
    else if (s.open && left?.contains(a)) { actions.closePanel(); document.getElementById('tab-' + s.mode)?.focus(); }
    else if (s.tool) actions.closeTool();
    else if (s.open) actions.closePanel();
    else if (s.welcome) actions.dismissWelcome();
    else if (stats.isOpen()) stats.close();
    else return false;
  };
  registerKeymap({
    app, nav, undo, redo, saveCard, hosted, toast: (t) => toast(t),
    help: () => help.open(), commands: () => cmds.open(),
    ringMenu: (m) => rings.open(m, m === 'view' ? 'Views' : 'Display', m === 'view' ? '`' : 'Z', pointerAt, m === 'view' ? 'Backquote' : 'KeyZ'),
    escape,
  });
  installPointer(app.canvas);
  trackModifiers();
  // the interface mounts once the keys are in the registry, so its tooltips and key caps show them from the start
  mountShell({ app, ed, legacy });
  // for the page tests: the registry, the camera, the status line, the settings and the actions that ran
  const ran: string[] = [];
  onActionRun((id) => { ran.push(id); if (ran.length > 200) ran.shift(); });
  (window as any).__studio = {
    actions: () => allActions().map((a) => ({ id: a.id, keys: a.keys || [], scope: a.scope || 'global', run: !!a.run, label: a.label })),
    cam: () => cam().state(), status: () => statusText(), settings, ran, setSelection,
    project: (p: number[]) => cam().toScreen({ x: p[0], y: p[1], z: p[2] }),
    joint: (name: string) => { const j = app.figure().joints.find((x) => x.name === name); return j ? [j.p.x, j.p.y, j.p.z] : null; },
    hand: (side: string) => app.figure().joints.filter((j) => j.name.endsWith('.' + side) && /^(finger|wrist|metacarpal)/.test(j.name)).map((j) => { const s = cam().toScreen(j.p); return [s.x, s.y]; }),
    bodyRay: (x: number, y: number) => { const r = cam().rayAt(x, y); const h = app.bodyRay(r.origin, r.direction); return h ? [h.point.x, h.point.y, h.point.z, h.distance] : null; },
    free: () => ({ freeW: app.view.freeW, freeH: app.view.freeH }),
    tour: () => startTour(), offerTour: () => offerTour(),
  };
  // the height rail after the interface, so the focus reaches it after the top bar, the panel and the view bar
  document.body.append(nav.rail);
  app.canvas.tabIndex = -1;
  // the line under the view, with nothing held, Shift held and Alt held (artboard 3b)
  setStatus('idle', [{ keys: 'Left-drag', text: 'turn' }, { keys: 'Middle-drag', text: 'move the view' }, { keys: 'Wheel', text: 'zoom' }, { keys: 'F', text: 'frame' },
    { keys: 'Double-click', text: 'turn round this spot' }], {
    priority: PRIORITY.idle,
    held: {
      shift: [{ text: 'Fine movement' }, { keys: 'Left-drag', text: 'turn slowly' }, { keys: 'Wheel', text: 'zoom 3 %' }, { keys: 'Drag a point', text: 'a quarter speed' }],
      alt: [{ text: 'Turn even over a point' }, { keys: 'Alt+left-drag', text: 'turn' }, { keys: 'Alt+right-drag', text: 'zoom' }],
      ctrl: [{ text: 'Snap' }, { keys: 'Ctrl+left-drag', text: 'turn in steps of 15°' }, { keys: 'Ctrl+Z', text: 'undo' }, { keys: 'Ctrl+S', text: 'save' }],
    },
  });
  document.addEventListener('keydown', (e) => {
    if (e.defaultPrevented) return;
    const isTyping = typing(e);
    // Esc leaves a text field, so the shortcuts work again (a slider's value box and a search field put back their
    // text first)
    if (isTyping && e.key === 'Escape' && !(e.ctrlKey || e.metaKey)) { (e.target as HTMLElement).blur(); return; }
    const t = e.target as HTMLElement;
    // a focused button keeps Space and Enter; the arrows stay with the controls that use them; Tab moves the focus
    // outside the 3D view
    if ((e.key === ' ' || e.key === 'Enter') && !e.ctrlKey && t.closest?.('button, summary, a')) return;
    if (e.key.startsWith('Arrow') && t.closest?.('input, .seg, .osb-seg, [role="tablist"], [role="radiogroup"], [role="menu"], [role="slider"]')) return;
    if (rings.isOpen()) return;
    // the interface shows a key's panel before the next frame
    let id: string | null = null;
    // a grab or a turn (studio/modal.ts) takes every key while it lasts
    const modal = modalActive();
    flushSync(() => { id = dispatchKey(e, { typing: isTyping, dialog: dialogOpen(), hint: (text) => toast(text), only: modal ? 'modal' : undefined }); });
    if (id || modal) e.preventDefault();
  });
  // Alt+← and Alt+→ would leave the page with unsaved edits; Ctrl+wheel would zoom the page over the view
  addEventListener('keydown', (e) => { if (e.altKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight') && !typing(e)) e.preventDefault(); }, true);

  // ---------------------------------------------------------------- the start state
  if (!app.shot && app.qs.get('stats') === '1') stats.open();
  layout();
  const syncLook = () => {
    const L = app.look();
    if (L.name !== 'Custom') ed.origin = L.name;
    bus.emit('look');
  };
  syncLook(); ss.sync(); paintView(); paintRing();
  let tFrame = 0;
  const hooks: Hooks = {
    look: syncLook,
    motion: () => { ss.sync(); if (app.rigView() === 'off') boneTip.hidden = true; bus.emit('motion'); },
    hair: () => { ss.sync(); bus.emit('hair'); },
    face: () => { bus.emit('face'); },
    frame(now, dt, drew, jsMs) {
      stats.frame(now, dt, drew, jsMs);
      if (now - tFrame > 100) { tFrame = now; paintRing(); paintView(); bus.emit('tick'); }
    },
    passBegin: () => stats.passBegin(),
    passEnd: () => stats.passEnd(),
    interacted: () => { /* the welcome card stays until it is closed */ },
  };
  return { hooks, nav, stats, firstStarter };
}

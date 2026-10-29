// OpenSculptBoy
// Apache License, Version 2.0
//
// The Sheet mode without its controls: the settings of the character sheet (remembered in the page's settings), the
// build of the sheet with its progress, the layout view beside the 3D view (the page as it will print, filled in panel
// by panel while the sheet builds), Download PNG, and window.sheet for scripts: window.sheet.build(settings) resolves
// to the PNG bytes of the page. The Sheet section of the interface (src/shell/sections/SheetSection.tsx) draws the
// controls and subscribes here; nothing here runs on a frame of the renderer.

import type { App } from '../ui/app.ts';
import { download, h, iconButton, saveUi, slug, uiState } from '../ui/dom.ts';
import { toast } from '../ui/toast.ts';
import { face, faceReady } from '../face/api.ts';
import { buildSheet, layoutOf, planPanels, report, SheetCancelled, type SheetResult } from './build.ts';
import { canvasPng, compose, loadFonts } from './compose.ts';
import { defaultSettings, EXPRESSION_ORDER, normaliseSettings, settingsKey, type PoseSlot, type SheetSettings } from './settings.ts';
import { styleIds } from './styles.ts';

const PREVIEW_W = 960;

export interface SheetState { building: boolean; built: boolean; progress: number; label: string; viewOn: boolean; shown: boolean }
export interface SheetController {
  settings(): SheetSettings;
  // change the settings (the build's pictures follow; the name, the title and the background redraw the page alone)
  change(mut: (S: SheetSettings) => void, opts?: { quiet?: boolean }): void;
  state(): SheetState;
  subscribe(fn: () => void): () => void;
  build(): void;
  stop(): void;
  downloadPng(): void;
  showPage(on: boolean): void;
  // the layout view of the page, and the section's opening and closing (the view shows with the section)
  view: HTMLElement;
  setOpen(open: boolean): void;
  shown(): boolean;
  poses(): { name: string; label: string; group: string }[];
  poseLabel(slot: PoseSlot): string;
  poseFromPicture(file: File): Promise<PoseSlot>;
  characterName(): string;
  expressionNames(): string[];
  faces(): boolean;
}

// the settings that change the pictures: the name, the title, the background and the info block draw on the page alone
function pictureKey(s: SheetSettings) {
  const c = JSON.parse(settingsKey(s));
  delete c.name; delete c.title; delete c.background; delete c.panels.info;
  return JSON.stringify(c);
}

function base64(bytes: Uint8Array) {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + 0x8000)));
  return btoa(s);
}

export function sheetController(app: App, relayout: () => void): SheetController {
  const host = app.sheetHost();
  const hosted = !!(window as any).MODEL_PARTS;
  const normalise = (s: any) => normaliseSettings(s, styleIds());
  let S: SheetSettings = normalise(uiState().sheet || defaultSettings());
  // a background that the user never picked follows the default (remembered settings from before the grey default
  // carry the paper)
  if (!uiState().sheetBackgroundPicked) S.background = defaultSettings().background;
  let result: SheetResult | null = null, resultKey = '';
  let ctl: { cancelled: boolean } | null = null;
  let lastPng: Uint8Array | null = null, lastReport: any = null;
  let progress = 0, label = '';
  const listeners = new Set<() => void>();
  const emit = () => { for (const f of listeners) f(); };
  const keep = () => saveUi({ sheet: S });
  const changed = () => { keep(); emit(); preview(); };

  // ---------------------------------------------------------------- the layout view
  const img = h('img.cs-page', { alt: 'The character sheet as it will print' }) as HTMLImageElement;
  const vStatus = h('span.cs-vstatus');
  const vBar = h('div.cs-bar', { 'aria-hidden': 'true' }, h('span'));
  const vBuild = iconButton('sheet', 'Build sheet', { tip: 'down', on: () => { run(S).catch(() => {}); } });
  const vDown = iconButton('download', 'Download PNG', { tip: 'down', on: () => { downloadPng(); } });
  const vHide = iconButton('close', 'Hide the page', { tip: 'left', on: () => setShown(false) });
  const view = h('aside.cs-view.glass', { 'aria-label': 'Character sheet', hidden: true },
    h('header.cs-vhead', {}, h('h2', { text: 'Sheet' }), vStatus, h('div.sec-actions', {}, vBuild, vDown, vHide)), vBar,
    h('div.cs-vbody', {}, img));
  let viewOn = uiState().sheetView !== false, open = false, shownNow = false;
  const showView = () => {
    shownNow = open && viewOn;
    view.hidden = !shownNow;
    if (shownNow) preview();
    relayout();
    emit();
  };
  const setShown = (on: boolean) => { viewOn = on; saveUi({ sheetView: on }); showView(); };
  const setProgress = (f: number, text: string) => {
    progress = f; label = text;
    (vBar.firstElementChild as HTMLElement).style.transform = `scaleX(${f})`; vBar.toggleAttribute('data-on', f > 0 && f < 1);
    vStatus.textContent = text;
    emit();
  };

  // the page as it stands: the built pictures while they match the settings, else frames where they will go
  let previewBusy = false, previewAgain = false, lastUrl = '';
  const picturesFor = (r: SheetResult | null) => (panel: string, i: number) => {
    const P = r?.panels.find((p) => p.id === panel);
    const p = P?.pictures[i];
    return p?.bitmap || null;
  };
  const currentLayout = (r: SheetResult | null) => {
    const name = S.name || host.characterName();
    const panels = r ? planPanels(S, host).map((P) => r.panels.find((q) => q.id === P.id) || P) : planPanels(S, host);
    return layoutOf(S, name, panels);
  };
  let live: SheetResult | null = null;   // the result of the build that runs
  const preview = () => {
    if (!shownNow && !ctl) return;
    if (previewBusy) { previewAgain = true; return; }
    previewBusy = true;
    const r = live || (result && resultKey === pictureKey(S) ? result : null);
    const L = live ? live.layout : currentLayout(r);
    const c = compose(L, live ? live.settings.background : S.background, picturesFor(r), Math.min(1, PREVIEW_W / L.width));
    c.toBlob((b) => {
      previewBusy = false;
      if (b) { const url = URL.createObjectURL(b); img.src = url; if (lastUrl) URL.revokeObjectURL(lastUrl); lastUrl = url; }
      if (previewAgain) { previewAgain = false; preview(); }
    }, 'image/png');
  };

  // ---------------------------------------------------------------- building and saving
  async function run(settings: SheetSettings): Promise<SheetResult> {
    if (ctl) throw new Error('A sheet is building already.');
    if (!app.bodyReady()) throw new Error('The figure is still loading.');
    ctl = { cancelled: false };
    emit();
    const key = pictureKey(settings);
    setProgress(0.001, 'Starting');
    try {
      await loadFonts();
      const r = await buildSheet(host, settings, {
        progress: (f, text) => setProgress(f, text),
        changed: (partial) => { live = partial; preview(); },
      }, ctl);
      result = r; resultKey = key;
      setProgress(1, `Built in ${(r.ms / 1000).toFixed(1)} s`);
      return r;
    } catch (e: any) {
      setProgress(0, e instanceof SheetCancelled ? 'Stopped' : 'The sheet could not be built');
      if (!(e instanceof SheetCancelled)) { console.error('Sheet:', e); toast('The sheet could not be built', 'error'); }
      throw e;
    } finally {
      ctl = null; live = null;
      emit(); preview();
    }
  }
  // the page at print size as PNG, with the current name, title and background
  async function pagePng(r: SheetResult, settings: SheetSettings) {
    await loadFonts();
    const name = settings.name || r.name;
    const L = layoutOf({ ...r.settings, name: settings.name, title: settings.title, background: settings.background, panels: { ...r.settings.panels, info: settings.panels.info } }, name,
      planPanels({ ...r.settings, panels: { ...r.settings.panels, info: settings.panels.info } }, host).map((P) => r.panels.find((q) => q.id === P.id) || P));
    const png = await canvasPng(compose(L, settings.background, picturesFor(r), 1));
    // faces: whether a face driver drew the expressions (face/api.ts); without one every cell shows the neutral face
    lastPng = png; lastReport = { ...report(r, png.length), width: L.width, height: L.height, faces: faceReady() };
    return png;
  }
  async function downloadPng() {
    try {
      const r = result && resultKey === pictureKey(S) ? result : await run(S);
      const png = await pagePng(r, S);
      const blob = new Blob([png as unknown as BlobPart], { type: 'image/png' }), name = `${slug(S.name || r.name)}-sheet.png`;
      if (!hosted) { download(name, blob); toast('Sheet saved'); return; }
      const url = URL.createObjectURL(blob);
      const shot = h('dialog.help.shot-dlg', { 'aria-label': 'Character sheet' }) as HTMLDialogElement;
      const close = iconButton('close', 'Close', { key: 'Esc', tip: 'left', on: () => shot.close() });
      shot.append(h('header.help-head', {}, h('h2', { text: 'Sheet' }), close),
        h('p.help-lead', { text: 'Right-click the picture and choose Save image, or press and hold it on a phone.' }),
        h('img.shot-img', { src: url, alt: 'The character sheet' }));
      shot.addEventListener('close', () => { URL.revokeObjectURL(url); shot.remove(); });
      document.body.append(shot); shot.showModal();
    } catch (e) { /* run reported it */ }
  }
  const expressionNames = () => {
    const names = faceReady() ? face().presetNames() : [];
    const all = names.length ? names : EXPRESSION_ORDER;
    return [...EXPRESSION_ORDER.filter((n) => all.includes(n)), ...all.filter((n) => !EXPRESSION_ORDER.includes(n))];
  };

  // ---------------------------------------------------------------- the script entry
  (window as any).sheet = {
    defaults: () => defaultSettings(),
    normalise: (s: any) => normalise(s),
    settings: () => JSON.parse(JSON.stringify(S)),
    set: (s: any) => { S = normalise(s); changed(); return JSON.parse(JSON.stringify(S)); },
    // build a sheet of the settings (else of the section's) and resolve to the PNG bytes of the page
    build: async (s?: any) => {
      const settings = s ? normalise(s) : S;
      const r = await run(settings);
      return pagePng(r, settings);
    },
    base64,
    report: () => lastReport,
    png: () => lastPng,
    // one panel picture of the last build as a PNG data URL, with its transparent background
    picture: (panel: string, i: number) => {
      const P = result?.panels.find((p) => p.id === panel), p = P?.pictures[i];
      if (!p) return null;
      const c = document.createElement('canvas'); c.width = p.w; c.height = p.h;
      c.getContext('2d')!.putImageData(p.data, 0, 0);
      return c.toDataURL('image/png');
    },
    busy: () => !!ctl,
    // the renderer's side (sheet/host.ts), for scripts that render single views
    host,
  };

  return {
    settings: () => S,
    change(mut, opts = {}) { mut(S); if (opts.quiet) { keep(); preview(); emit(); } else changed(); },
    state: () => ({ building: !!ctl, built: !!result && resultKey === pictureKey(S), progress, label, viewOn, shown: shownNow }),
    subscribe(fn) { listeners.add(fn); return () => { listeners.delete(fn); }; },
    build: () => { run(S).catch(() => {}); },
    stop: () => { if (ctl) ctl.cancelled = true; },
    downloadPng: () => { downloadPng(); },
    showPage: (on) => setShown(on),
    view,
    setOpen: (on) => { if (on === open) return; open = on; showView(); },
    shown: () => shownNow,
    poses: () => host.poses(),
    poseLabel: (s) => s.label || host.poses().find((p) => p.name === s.pose)?.label || s.pose,
    async poseFromPicture(file) {
      await app.poseFromPicture(file);
      const q = host.poseQuaternions('picture');
      if (!q) throw new Error('No pose');
      return { pose: 'picture', label: 'From a picture', q };
    },
    characterName: () => host.characterName(),
    expressionNames,
    faces: () => faceReady(),
  };
}

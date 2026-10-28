// OpenSculptBoy
// Apache License, Version 2.0
//
// The Sheet section of the Stage segment and its layout view: the character sheet of the figure shown (sheet/). The
// section holds the style, the background, the quality and the panels, each with a switch and its settings: the
// turnaround's pose and views, the expression grid's pose, view and expressions, the six pose slots (each opens the
// pose library or From a picture), and the info block's name and title. The layout view beside the 3D view shows the
// page as it will print and fills in panel by panel while the sheet builds; Download PNG writes the page.
// window.sheet gives scripts the same build: window.sheet.build(settings) resolves to the PNG bytes of the page.

import type { App } from './app.ts';
import { chip, foldGroup, searchField, sectionHead, segmented, textButton, toggle } from './controls.ts';
import { download, h, iconButton, saveUi, slug, uiState } from './dom.ts';
import { icon } from './icons.ts';
import { toast } from './toast.ts';
import { face, faceReady } from '../face/api.ts';
import { buildSheet, layoutOf, planPanels, report, SheetCancelled, type SheetResult } from '../sheet/build.ts';
import { canvasPng, compose, loadFonts } from '../sheet/compose.ts';
import {
  defaultSettings, EXPRESSION_ORDER, expressionLabel, MAX_POSES, normaliseSettings, PANEL_LABELS, RENDER_PRESETS, settingsKey, VIEWS,
  type PanelId, type PoseSlot, type SheetSettings, type ViewName,
} from '../sheet/settings.ts';
import { STYLES, styleIds } from '../sheet/styles.ts';

const PREVIEW_W = 960;

// el: the section; view: the layout view; setOpen: the section opened or closed (index.ts), so the view shows with it
export interface SheetUI { el: HTMLElement; view: HTMLElement; setOpen(open: boolean): void; shown(): boolean }

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

export function sheetSection(app: App, relayout: () => void): SheetUI {
  const host = app.sheetHost();
  const hosted = !!(window as any).MODEL_PARTS;
  const normalise = (s: any) => normaliseSettings(s, styleIds());
  let S: SheetSettings = normalise(uiState().sheet || defaultSettings());
  let result: SheetResult | null = null, resultKey = '';
  let ctl: { cancelled: boolean } | null = null;
  let lastPng: Uint8Array | null = null, lastReport: any = null;
  const keep = () => saveUi({ sheet: S });
  const changed = () => { keep(); paint(); preview(); };

  // ---------------------------------------------------------------- the pose picker: the library and From a picture
  const picker = h('dialog.help.cs-picker', { 'aria-labelledby': 'cs-picker-title' }) as HTMLDialogElement;
  let pick: ((slot: PoseSlot) => void) | null = null;
  {
    const close = iconButton('close', 'Close', { key: 'Esc', tip: 'left', on: () => picker.close() });
    const folds = new Map<string, ReturnType<typeof foldGroup>>();
    const wrap = h('div.folds');
    for (const p of host.poses()) {
      let f = folds.get(p.group);
      if (!f) { f = foldGroup(p.group || 'Poses'); f.body.classList.add('chips'); folds.set(p.group, f); wrap.append(f.el); }
      const b = chip(p.label, () => { picker.close(); pick?.({ pose: p.name }); });
      b.dataset.pose = p.name;
      f.body.append(b);
      f.count.textContent = String(f.body.children.length);
    }
    const search = searchField('Search the poses', (q) => {
      for (const [g, f] of folds) {
        let n = 0;
        for (const b of Array.from(f.body.children) as HTMLElement[]) { const hit = !q || b.textContent!.toLowerCase().includes(q) || g.toLowerCase().includes(q); b.hidden = !hit; if (hit) n++; }
        f.el.hidden = n === 0; if (q) f.el.open = n > 0;
      }
    });
    const file = h('input', { type: 'file', accept: 'image/*', hidden: true }) as HTMLInputElement;
    const msg = h('p.inline-msg', { role: 'status' });
    const choose = textButton('Choose a picture', () => file.click(), 'upload');
    file.addEventListener('change', async () => {
      const f = file.files && file.files[0]; file.value = '';
      if (!f) return;
      choose.disabled = true; msg.textContent = 'Reading the pose. The first picture loads MediaPipe, about 20 MB.'; msg.dataset.kind = '';
      try {
        await app.poseFromPicture(f);
        const q = host.poseQuaternions('picture');
        if (!q) throw new Error('No pose');
        picker.close();
        pick?.({ pose: 'picture', label: 'From a picture', q });
      } catch (e) {
        msg.textContent = 'The picture gave no pose. Try one where the whole figure shows.'; msg.dataset.kind = 'error';
      } finally { choose.disabled = false; }
    });
    picker.append(h('header.help-head', {}, h('h2', { id: 'cs-picker-title', text: 'Choose a pose' }), close),
      h('div.cs-picker-body', {}, search.el, wrap,
        h('h3.sub', { text: 'From a picture' }),
        h('p.note', { text: 'Pose the figure as the figure in a picture. The picture stays in this browser.' }),
        h('div.btn-row', {}, choose), msg, file));
    picker.addEventListener('click', (e) => { if (e.target === picker) picker.close(); });
    picker.addEventListener('close', () => { pick = null; });
  }
  const openPicker = (title: string, slot: PoseSlot, done: (s: PoseSlot) => void) => {
    (picker.querySelector('#cs-picker-title') as HTMLElement).textContent = title;
    for (const b of Array.from(picker.querySelectorAll('.chip')) as HTMLElement[]) b.setAttribute('aria-pressed', String(!slot.q && b.dataset.pose === slot.pose));
    pick = done;
    picker.showModal();
  };
  const poseLabel = (s: PoseSlot) => s.label || host.poses().find((p) => p.name === s.pose)?.label || s.pose;
  const poseButton = (title: string, slot: PoseSlot, done: (s: PoseSlot) => void) => {
    const b = h('button.cs-pose', { type: 'button', 'data-tip': 'Change the pose', 'data-tip-pos': 'up' }, icon('pose', 16), h('span', { text: poseLabel(slot) }), icon('chevron', 14));
    b.addEventListener('click', () => openPicker(title, slot, done));
    return b;
  };
  const viewChoice = (label: string, views: ViewName[], cur: ViewName, set: (v: ViewName) => void) => {
    const seg = segmented(label, views.map((v) => ({ value: v, label: VIEWS[v].label, short: v === 'three' ? '¾' : undefined })), (v) => set(v as ViewName), 'seg-fill');
    seg.set(cur);
    return seg.el;
  };

  // ---------------------------------------------------------------- the section
  const el = h('section');
  const stopB = textButton('Stop', () => { if (ctl) ctl.cancelled = true; }, 'stop', 'tb-quiet');
  stopB.hidden = true;
  const head = sectionHead('Sheet', stopB);
  const styleSeg = segmented('Style', STYLES.map((s) => ({ value: s.id, label: s.label, tip: s.tip })), (v) => { S.style = v; changed(); }, 'seg-fill');
  const bgSeg = segmented('Background', [{ value: 'paper', label: 'Paper' }, { value: 'dark', label: 'Dark' }, { value: 'transparent', label: 'Transparent' }], (v) => { S.background = v as any; changed(); }, 'seg-fill');
  const qualSeg = segmented('Quality', [{ value: 'print', label: 'Print', tip: 'Pictures at print size, 24 samples each' }, { value: 'draft', label: 'Draft', tip: 'Half the size and 8 samples: a quick look' }],
    (v) => { S.render = { ...RENDER_PRESETS[v] }; changed(); }, 'seg-fill');
  const panelsBox = h('div.folds.cs-panels');
  const buildB = textButton('Build sheet', () => { run(S).catch(() => {}); }, 'sheet', 'cs-primary');
  const downB = textButton('Download PNG', () => { downloadPng(); }, 'download');
  const bar = h('div.cs-bar', { 'aria-hidden': 'true' }, h('span'));
  const status = h('p.inline-msg.cs-status', { role: 'status' });
  const showB = textButton('Show the page', () => setShown(true), 'sheet', 'tb-quiet');
  el.append(head.el,
    h('p.sec-lead', { text: 'A character sheet of the figure shown: its turnaround, its expressions and its poses on one page, to share or to print.' }),
    h('div.btn-row', {}, buildB, downB), bar, status,
    h('h3.sub', { text: 'Style' }), styleSeg.el,
    h('h3.sub', { text: 'Background' }), bgSeg.el,
    h('h3.sub', { text: 'Quality' }), qualSeg.el,
    h('h3.sub', { text: 'Panels' }), panelsBox, showB);
  document.body.append(picker);

  const folds: Partial<Record<PanelId, HTMLDetailsElement>> = {};
  const panelFold = (id: PanelId, body: HTMLElement[]) => {
    const p: any = S.panels[id];
    const sw = toggle('', (on) => { p.on = on; changed(); }, 'Show this panel on the sheet');
    sw.set(p.on);
    sw.el.classList.add('cs-switch');
    sw.el.setAttribute('aria-label', 'Show the ' + PANEL_LABELS[id].toLowerCase());
    const f = foldGroup(PANEL_LABELS[id], [sw.el]);
    f.el.dataset.panel = id;
    f.el.open = folds[id]?.open ?? id === 'turnaround';
    f.el.toggleAttribute('data-off', !p.on);
    f.body.append(...body);
    folds[id] = f.el;
    f.el.addEventListener('toggle', () => { /* remembered in folds */ });
    return f.el;
  };
  const field = (label: string, value: string, placeholder: string, set: (v: string) => void) => {
    const input = h('input.cs-input', { type: 'text', value, placeholder, autocomplete: 'off', spellcheck: 'false', 'aria-label': label }) as HTMLInputElement;
    input.addEventListener('input', () => { set(input.value); keep(); preview(); });
    return h('label.cs-field', {}, h('span', { text: label }), input);
  };
  const expressionNames = () => {
    const names = faceReady() ? face().presetNames() : [];
    const all = names.length ? names : EXPRESSION_ORDER;
    return [...EXPRESSION_ORDER.filter((n) => all.includes(n)), ...all.filter((n) => !EXPRESSION_ORDER.includes(n))];
  };
  const paint = () => {
    const P = S.panels;
    styleSeg.set(S.style); bgSeg.set(S.background);
    qualSeg.set(S.render.scale === RENDER_PRESETS.draft.scale && S.render.samples === RENDER_PRESETS.draft.samples ? 'draft' : S.render.scale === 1 && S.render.samples === 24 ? 'print' : '');
    // the turnaround: its pose and its views
    const views = h('div.chips', { role: 'group', 'aria-label': 'Views' });
    for (const v of Object.keys(VIEWS) as ViewName[]) {
      const c = chip(VIEWS[v].label, () => {
        const on = P.turnaround.views.includes(v);
        P.turnaround.views = (Object.keys(VIEWS) as ViewName[]).filter((x) => (x === v ? !on : P.turnaround.views.includes(x)));
        changed();
      });
      c.setAttribute('aria-pressed', String(P.turnaround.views.includes(v)));
      views.append(c);
    }
    const turnaround = panelFold('turnaround', [
      h('p.note', { text: 'The full figure from four sides, at one scale on one ground line.' }),
      h('div.row', {}, h('span.row-label', { text: 'Pose' }), poseButton('The turnaround\'s pose', P.turnaround.pose, (s) => { P.turnaround.pose = s; changed(); })),
      views]);
    // the expression grid
    const exprs = h('div.chips', { role: 'group', 'aria-label': 'Expressions' });
    for (const n of expressionNames()) {
      const c = chip(expressionLabel(n), () => {
        const on = P.expressions.items.includes(n), order = expressionNames();
        P.expressions.items = on ? P.expressions.items.filter((x) => x !== n) : [...P.expressions.items, n].sort((a, b) => order.indexOf(a) - order.indexOf(b));
        changed();
      });
      c.setAttribute('aria-pressed', String(P.expressions.items.includes(n)));
      c.dataset.expression = n;
      exprs.append(c);
    }
    const expressions = panelFold('expressions', [
      h('p.note', { text: 'Head and shoulders, one picture per expression.' }),
      h('div.row', {}, h('span.row-label', { text: 'Pose' }), poseButton('The expressions\' pose', P.expressions.pose, (s) => { P.expressions.pose = s; changed(); })),
      viewChoice('View', ['front', 'three', 'side'], P.expressions.view, (v) => { P.expressions.view = v; changed(); }),
      exprs,
      ...(faceReady() ? [] : [h('p.note', { text: 'This page shows every expression with the neutral face for now.' })])]);
    // the poses
    const slots = h('div.cs-slots');
    P.poses.slots.forEach((slot, i) => {
      const rm = iconButton('close', 'Remove this pose', { tip: 'left', on: () => { P.poses.slots.splice(i, 1); changed(); } });
      slots.append(h('div.cs-slot', { dataset: { slot: String(i) } }, h('span.cs-n', { text: String(i + 1) }),
        poseButton(`Pose ${i + 1}`, slot, (s) => { P.poses.slots[i] = s; changed(); }), rm));
    });
    const add = textButton('Add a pose', () => openPicker(`Pose ${P.poses.slots.length + 1}`, { pose: '' }, (s) => { P.poses.slots.push(s); changed(); }), 'plus', 'tb-quiet');
    add.disabled = P.poses.slots.length >= MAX_POSES;
    const reset = textButton('Default poses', () => { P.poses.slots = defaultSettings().panels.poses.slots; changed(); }, 'undoSmall', 'tb-quiet');
    const poses = panelFold('poses', [
      h('p.note', { text: 'Full-figure poses that show the character\'s manner, at one scale.' }),
      viewChoice('View', ['front', 'three', 'side'], P.poses.view, (v) => { P.poses.view = v; changed(); }),
      slots, h('div.btn-row', {}, add, reset)]);
    const info = panelFold('info', [
      field('Name', S.name, host.characterName(), (v) => { S.name = v; }),
      field('Title', S.title, 'A line under the name', (v) => { S.title = v; })]);
    panelsBox.replaceChildren(info, turnaround, expressions, poses);
    head.note.textContent = ctl ? 'Building' : result && resultKey === pictureKey(S) ? 'Built' : '';
    buildB.disabled = !!ctl; stopB.hidden = !ctl;
  };

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
    view.hidden = !shownNow; showB.hidden = shownNow || !open;
    if (shownNow) preview();
    relayout();
  };
  const setShown = (on: boolean) => { viewOn = on; saveUi({ sheetView: on }); showView(); };
  const setProgress = (f: number, label: string) => {
    for (const b of [bar, vBar]) { (b.firstElementChild as HTMLElement).style.transform = `scaleX(${f})`; b.toggleAttribute('data-on', f > 0 && f < 1); }
    status.textContent = label; vStatus.textContent = label;
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
    paint();
    const key = pictureKey(settings);
    setProgress(0.001, 'Starting');
    try {
      await loadFonts();
      const r = await buildSheet(host, settings, {
        progress: (f, label) => setProgress(f, label),
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
      paint(); preview();
    }
  }
  // the page at print size as PNG, with the current name, title and background
  async function pagePng(r: SheetResult, settings: SheetSettings) {
    await loadFonts();
    const name = settings.name || r.name;
    const L = layoutOf({ ...r.settings, name: settings.name, title: settings.title, background: settings.background, panels: { ...r.settings.panels, info: settings.panels.info } }, name,
      planPanels({ ...r.settings, panels: { ...r.settings.panels, info: settings.panels.info } }, host).map((P) => r.panels.find((q) => q.id === P.id) || P));
    const png = await canvasPng(compose(L, settings.background, picturesFor(r), 1));
    lastPng = png; lastReport = { ...report(r, png.length), width: L.width, height: L.height };
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

  paint();
  return {
    el, view,
    setOpen: (on: boolean) => { if (on === open) return; open = on; showView(); },
    shown: () => shownNow,
  };
}

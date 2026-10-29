// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The tool panels of the View menu that are still the page's own DOM: Scene (the lights, their rotation, the
// exposure, the quality of the picture, the frame rate limit and the display toggles) and Rig (rig_ui.ts).

import type { App } from './app.ts';
import { chip, foldGroup, searchField, sectionHead, segmented, slider, textButton, toggle } from './controls.ts';
import { download, h, saveUi, slug, uiState } from './dom.ts';
import { icon } from './icons.ts';
import { rigSection } from './rig_ui.ts';

const QUALITY: Record<string, string> = {
  auto: 'Draws at a lower resolution while the figure moves, then refines the still picture at full resolution.',
  sharp: 'Always draws at full resolution with every hair strand. Moving frames can be slower.',
  fast: 'Draws fewer pixels, fewer samples and fewer strands, for slower devices and longer battery life.',
};

export function stageSections(app: App) {
  const syncs: (() => void)[] = [];

  // ---------------------------------------------------------------- Scene
  const scene = h('section');
  {
    const head = sectionHead('Scene');
    const cards = h('div.light-cards', { role: 'group', 'aria-label': 'Lighting' });
    const lightEls = Object.entries(app.presets).map(([name, p], i) => {
      // a preview of the backdrop and of the direction of the key light as seen from the front
      const bg = (c: number[]) => `rgb(${c.map((v) => Math.round(255 * Math.min(1, Math.pow(v * 5, 1 / 2.2)))).join(',')})`;
      const a = p.key.az * Math.PI / 180, behind = Math.abs(p.key.az) > 90;
      const x = 50 + Math.sin(a) * 36, y = 50 - p.key.el / 90 * 40;
      const prev = h('span.light-prev', { style: { background: `radial-gradient(circle at 50% 58%, ${bg(p.bg[0])}, ${bg(p.bg[1])})` }, 'aria-hidden': 'true' },
        h('span.light-fig'), h('span.light-dot' + (behind ? '.behind' : ''), { style: { left: x + '%', top: y + '%' } }));
      const b = h('button.light-card', { type: 'button', 'aria-pressed': 'false', dataset: { preset: name }, 'data-tip': `${p.label}  ·  Alt+${i + 1}`, 'data-tip-pos': 'up' }, prev, h('span', { text: p.label }));
      b.addEventListener('click', () => app.applyPreset(name));
      cards.append(b);
      return b;
    });
    const rot = slider({ id: 'sc-rot', label: 'Light rotation (°)', min: -180, max: 180, step: 1, def: 0, digits: 0, ends: ['Left', 'Right'],
      onInput: (v) => app.setLightRotation(v, false), onEnd: () => app.setLightRotation(rot.value(), true) });
    const exp = slider({ id: 'sc-exp', label: 'Exposure (stops)', min: -1.5, max: 1.5, step: 0.05, def: 0, ends: ['Darker', 'Brighter'],
      onInput: (v) => app.setExposureEV(v) });
    const qNote = h('p.note');
    const qual = segmented('Quality', [{ value: 'auto', label: 'Auto' }, { value: 'sharp', label: 'Sharp' }, { value: 'fast', label: 'Fast' }], (v) => { app.setQuality(v); saveUi({ quality: v }); sync(); }, 'seg-fill');
    const fps = segmented('Frame rate limit', app.fpsCaps.map((c) => ({ value: String(c), label: c ? `${c} fps` : 'Display', tip: c ? `At most ${c} frames per second` : 'As fast as the display refreshes' })),
      (v) => { app.setFpsCap(+v); saveUi({ fpsCap: +v }); sync(); }, 'seg-fill');
    const hairT = toggle('Hair', (on) => app.setHairVisible(on));
    const physT = toggle('Hair physics', (on) => app.setHairPhysics(on));
    const turnT = toggle('Turntable', (on) => app.setTurntable(on));
    scene.append(head.el, h('h3.sub', { text: 'Lighting' }), cards, h('div.sliders', {}, rot.el, exp.el),
      h('h3.sub', { text: 'Quality' }), qual.el, qNote,
      h('h3.sub', { text: 'Frame rate limit' }), fps.el,
      h('p.note', { text: 'A lower limit saves energy while the figure, the hair or the camera moves. A still picture refines and then stops drawing at any limit.' }),
      h('h3.sub', { text: 'Display' }), h('div.switches', {}, hairT.el, physT.el, turnT.el));
    const sync = () => {
      lightEls.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.preset === app.preset())));
      head.note.textContent = app.presets[app.preset()]?.label || '';
      rot.set(app.lightRotation()); exp.set(app.exposureEV());
      qual.set(app.quality()); qNote.textContent = QUALITY[app.quality()] || '';
      fps.set(String(app.fpsCap()));
      hairT.set(app.hairVisible()); physT.set(app.hairPhysics()); turnT.set(app.turntable());
    };
    syncs.push(sync);
  }

  // the saved quality and frame rate limit apply once at start
  const q = uiState().quality;
  if (q && q !== app.quality() && !app.shot) app.setQuality(q);
  const cap = uiState().fpsCap;
  if (typeof cap === 'number' && cap !== app.fpsCap() && !app.shot) app.setFpsCap(cap);
  // ---------------------------------------------------------------- Rig
  const rig = rigSection(app);
  syncs.push(rig.sync);

  return { rig: rig.el, scene, sync: () => { for (const s of syncs) s(); } };
}

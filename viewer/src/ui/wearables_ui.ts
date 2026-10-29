// OpenSculptBoy
// Apache License, Version 2.0
//
// The Wearables section of the Dress mode: a character card with its outfits and wearable packages opened from the
// user's own files (through the page's Open path, shell/open.ts), the outfits, every piece with a switch (a piece of
// another outfit takes its slot), the groups that hides rules hide, and the covered-skin switch. The files stay in
// the browser.

import type { App } from './app.ts';
import { chip, sectionHead, slider, textButton, toggle } from './controls.ts';
import { h } from './dom.ts';
import type { PickedFile } from '../wearables/files.ts';
import type { CardResult } from '../shell/open.ts';
import type { PieceRow, Wardrobe } from '../wearables/view.ts';

export interface WearablesUI { el: HTMLElement; load(picked: PickedFile[]): Promise<CardResult> }

export function wearablesSection(app: App, opener: { pickFiles(): void; pickFolder(): void }): WearablesUI {
  const el = h('section');
  const W = (): Wardrobe | null => app.wardrobe();
  const remove = textButton('Remove', () => { W()?.unload(); paint(); }, 'close', 'tb-quiet');
  const head = sectionHead('Wearables', remove);
  const msg = h('p.inline-msg.wear-msg', { role: 'status' });
  // the error of the last load that read no card, else the wardrobe's notes (with its failures) show in the message
  let loadError = '';
  const showNotes = () => {
    const w = W();
    if (loadError || !w?.loaded) { msg.textContent = loadError; msg.dataset.kind = loadError ? 'error' : ''; return; }
    msg.textContent = w.notes.length ? w.notes.join('. ') + '.' : '';
    msg.dataset.kind = w.failures ? 'error' : '';
  };
  // the files of a card or of wearable packages, from the page's Open path (shell/open.ts)
  const load = async (picked: PickedFile[]): Promise<CardResult> => {
    const w = W();
    if (!w) throw new Error('The page has no wardrobe.');
    loadError = ''; msg.textContent = 'Reading the files…'; msg.dataset.kind = '';
    try {
      const r = await w.load(picked);
      paint();
      return { name: w.cardName, pieces: r.pieces, worn: w.pieces.length };
    } catch (e: any) {
      loadError = e.message || String(e);
      console.error(`Wearables: ${loadError}`, e);
      paint();
      throw e;
    }
  };
  const drop = h('div.wear-drop.osb-drop.osb-drop--compact', {}, h('p.osb-note', { text: 'Drop a card and its packages, or their folder, anywhere on the page.' }));
  const tools = h('div.btn-row', {}, textButton('Open files', () => opener.pickFiles(), 'upload'), textButton('Open folder', () => opener.pickFolder(), 'folder'));
  const body = h('div.wear-body');
  const skinT = toggle('Hide covered skin', (on) => { W()?.setKeepSkin(!on); paint(); }, 'Leave out the body triangles under the shown pieces');
  const stats = h('p.note.wear-stats');
  // a card's eyes (card/look.ts): the blinks of the timer, one blink, and the closure and the squint that they start from
  const C = () => app.cardLook();
  const blinkT = toggle('Blink', (on) => C()?.setBlinkAuto(on), 'Close the eyes now and then, as the exports\' blink targets close them');
  const closeS = slider({ id: 'card-eyes-closed', label: 'Eyes closed', min: 0, max: 1, def: 0, onInput: (v) => C()?.setEyes(v, null) });
  const squintS = slider({ id: 'card-eyes-squint', label: 'Squint', min: 0, max: 1, def: 0, onInput: (v) => C()?.setEyes(null, v) });
  const eyes = h('div.wear-eyes', {}, blinkT.el, h('div.btn-row', {}, textButton('Blink once', () => C()?.blinkNow())), closeS.el, squintS.el);
  el.append(head.el,
    h('p.sec-lead', { text: 'A character card with its outfits and wearable packages, from your own files. The page reads them here and sends nothing.' }),
    tools, drop, msg, body);

  const row = (r: PieceRow, worn: string | null) => {
    const other = r.outfit !== null && r.outfit !== worn;
    const t = toggle(r.label, (on) => { W()?.setShown(r.ref, on); paint(); }, other ? 'Wear in place of the piece of its slot' : '');
    t.set(r.shown);
    t.el.dataset.ref = r.ref;
    const meta = [r.slot, `layer ${r.layer}`];
    if (r.hiddenGroups.length) meta.push(`${r.hiddenGroups.join(', ')} hidden`);
    if (r.shown && r.pushed) meta.push(`${r.pushed} pushed out, ${r.pushMm.toFixed(1)} mm at most`);
    return h('div.wear-piece', { dataset: { ref: r.ref } }, t.el, h('span.wear-meta', { text: meta.join(' · ') }));
  };
  const paint = () => {
    const w = W();
    showNotes();
    remove.hidden = !w?.loaded;
    drop.hidden = !!w?.loaded;
    head.note.textContent = w?.loaded ? w.cardName : '';
    if (!w || !w.loaded) { body.replaceChildren(); return; }
    const rows = w.rows(), worn = w.outfit;
    const outfits = h('div.chips', { role: 'group', 'aria-label': 'Outfit' });
    for (const o of [...w.outfits(), { id: null as any, label: 'No outfit' }]) {
      const c = chip(o.label, () => { w.setOutfit(o.id); paint(); });
      c.setAttribute('aria-pressed', String(o.id === worn));
      c.dataset.outfit = o.id ?? 'none';
      outfits.append(c);
    }
    const groups: [string, PieceRow[]][] = [];
    const add = (title: string, list: PieceRow[]) => { if (list.length) groups.push([title, list]); };
    add('Worn outfit', rows.filter((r) => r.outfit !== null && r.outfit === worn));
    add('Loose pieces', rows.filter((r) => r.outfit === null));
    for (const o of w.outfits()) if (o.id !== worn) add(o.label, rows.filter((r) => r.outfit === o.id));
    skinT.set(!w.keepSkin);
    const s = w.stats(), t = s.timing as any;
    stats.textContent = `${s.pieces.length} pieces shown, ${s.hidden_triangles} of ${s.body_triangles} body triangles under them.` +
      (t.total !== undefined ? ` Dressing ${t.total} ms${t.push ? `, push-out ${t.push} ms` : ''}${t.cover ? `, covered skin ${t.cover} ms` : ''}.` : '');
    body.replaceChildren(
      h('h3.sub', { text: 'Outfit' }), outfits,
      ...groups.flatMap(([title, list]) => [h('h3.sub', { text: title }), h('div.wear-list', {}, ...list.map((r) => row(r, worn)))]),
      h('h3.sub', { text: 'Skin' }), skinT.el, stats);
    const panel = C()?.blinkPanel();
    if (panel) {
      blinkT.set(panel.auto); closeS.set(panel.close); squintS.set(panel.squint);
      body.append(h('h3.sub', { text: 'Eyes' }), eyes);
    }
  };
  // a full dress repaints the section (the counts and times); the quick dress of a slider edit leaves it
  W()?.listeners.push(() => { if (W()?.loaded && (W()!.timing as any).full) paint(); else showNotes(); });
  paint();
  return { el, load };
}

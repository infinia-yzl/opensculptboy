// OpenSculptBoy
// Apache License, Version 2.0
//
// The Wearables section of the Character segment: a character card with its outfits and wearable packages opened
// from the user's own files (picked or dropped on the page), the outfits, every piece with a switch (a piece of
// another outfit takes its slot), the groups that hides rules hide, and the covered-skin switch. The files stay in
// the browser.

import type { App } from './app.ts';
import { chip, sectionHead, textButton, toggle } from './controls.ts';
import { h } from './dom.ts';
import { toast } from './toast.ts';
import { fromDrop, fromInput, type PickedFile } from '../wearables/files.ts';
import type { PieceRow, Wardrobe } from '../wearables/view.ts';

export function wearablesSection(app: App): HTMLElement {
  const el = h('section');
  const W = (): Wardrobe | null => app.wardrobe();
  const remove = textButton('Remove', () => { W()?.unload(); paint(); }, 'close', 'tb-quiet');
  const head = sectionHead('Wearables', remove);
  const msg = h('p.inline-msg', { role: 'status' });
  const files = h('input', { type: 'file', multiple: true, hidden: true, accept: '.json,.glb,.safetensors' }) as HTMLInputElement;
  const folder = h('input', { type: 'file', multiple: true, hidden: true }) as HTMLInputElement;
  folder.setAttribute('webkitdirectory', '');
  const open = async (picked: PickedFile[]) => {
    const w = W();
    if (!w || !picked.length) return;
    msg.textContent = 'Reading the files…'; msg.dataset.kind = '';
    try {
      const r = await w.load(picked);
      msg.textContent = r.notes.length ? r.notes.join('. ') + '.' : '';
      toast(`${w.cardName}: ${r.pieces} pieces`);
    } catch (e: any) {
      msg.textContent = e.message || String(e); msg.dataset.kind = 'error';
    }
    paint();
  };
  files.addEventListener('change', () => { open(fromInput(files.files || [])); files.value = ''; });
  folder.addEventListener('change', () => { open(fromInput(folder.files || [])); folder.value = ''; });
  const drop = h('div.wear-drop', { text: 'Drop a card and its packages, or their folder, anywhere on the page.' });
  // a drop anywhere on the page: files of a card or of wearable packages
  const holds = (e: DragEvent) => Array.from(e.dataTransfer?.types || []).includes('Files');
  document.addEventListener('dragover', (e) => { if (holds(e)) { e.preventDefault(); document.body.classList.add('wear-dragging'); } });
  document.addEventListener('dragleave', (e) => { if (e.target === document.documentElement || !e.relatedTarget) document.body.classList.remove('wear-dragging'); });
  document.addEventListener('drop', async (e) => {
    if (!holds(e)) return;
    e.preventDefault(); document.body.classList.remove('wear-dragging');
    open(await fromDrop(e.dataTransfer!));
  });
  const tools = h('div.btn-row', {}, textButton('Open files', () => files.click(), 'upload'), textButton('Open folder', () => folder.click(), 'upload'));
  const body = h('div.wear-body');
  const skinT = toggle('Hide covered skin', (on) => { W()?.setKeepSkin(!on); paint(); }, 'Leave out the body triangles under the shown pieces');
  const stats = h('p.note.wear-stats');
  el.append(head.el,
    h('p.sec-lead', { text: 'A character card with its outfits and wearable packages, from your own files. The page reads them here and sends nothing.' }),
    tools, drop, msg, files, folder, body);

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
  };
  // a full dress repaints the section (the counts and times); the quick dress of a slider edit leaves it
  W()?.listeners.push(() => { if (W()?.loaded && (W()!.timing as any).full) paint(); });
  paint();
  return el;
}

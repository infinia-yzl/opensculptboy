// OpenSculptBoy
// Apache License, Version 2.0
//
// The Dress mode's section in React: the open card's outfits and wearable packages (the page reads them from the
// user's own files through the Open path, shell/open.ts, and sends nothing), every piece with a switch (a piece of
// another outfit takes its slot) and a statue's props as switches of their own, the covered-skin switch, a line in
// plain words of what the dress did, the numbers behind a Details fold, and a card's eyes (its blinks). The section repaints when the wardrobe finishes a full dress;
// the quick dress of a slider edit leaves it as it is.

import { useState, useSyncExternalStore } from 'react';
import type { App } from '../../ui/app.ts';
import type { PickedFile } from '../../wearables/files.ts';
import type { PieceRow, Wardrobe } from '../../wearables/view.ts';
import type { CardResult } from '../open.ts';
import { useShell } from '../context.ts';
import { useBus } from '../hooks.ts';
import { createStore } from '../state.ts';
import { Slider } from '../components/Slider.tsx';
import { Chip, Fold, Note, Sub, Switch, TextButton } from '../components/controls.tsx';

// the error of the last open that read no card, kept beside the wardrobe's own notes
const dress = createStore({ loadError: '', rev: 0 });

// the loader that the Open path calls with a card's files
export function wearLoader(app: App) {
  return async (picked: PickedFile[], step: (id: string) => void): Promise<CardResult> => {
    const w = app.wardrobe();
    if (!w) throw new Error('The page has no wardrobe.');
    dress.set({ loadError: '' });
    try {
      const r = await w.load(picked, step);
      dress.set((s) => ({ rev: s.rev + 1 }));
      return { name: w.cardName, pieces: r.pieces, worn: w.pieces.length };
    } catch (e: any) {
      const msg = e.message || String(e);
      console.error(`Wearables: ${msg}`, e);
      dress.set((s) => ({ loadError: msg, rev: s.rev + 1 }));
      throw e;
    }
  };
}

// the dress in plain words: how many pieces show, and how far the fit moved them out of the skin
function plainWords(w: Wardrobe, rows: PieceRow[]): string {
  const shown = rows.filter((r) => r.shown), pushed = shown.filter((r) => r.pushed);
  if (!shown.length) return 'No piece is worn.';
  const most = pushed.reduce((a, r) => Math.max(a, r.pushMm), 0);
  return `${shown.length} piece${shown.length > 1 ? 's are' : ' is'} worn.` + (pushed.length ? ` The fit moved ${pushed.length === 1 ? 'one piece' : pushed.length + ' pieces'} out of the skin, by ${most < 1 ? 'less than 1 mm' : 'at most ' + Math.round(most) + ' mm'}.` : '') +
    (w.keepSkin ? '' : ' The skin under them leaves the picture.');
}

export function DressSection({ shown }: { shown: boolean }) {
  const { app, legacy } = useShell();
  const W = app.wardrobe();
  const [, setRev] = useState(0);
  const [eyes, setEyes] = useState(app.cardLook()?.blinkPanel() || null);
  // a full dress, a new card or its removal repaints the section; the quick dress of a slider edit does not
  useBus('wardrobe', () => { if (!W?.loaded || (W.timing as any).full !== false) setRev((r) => r + 1); setEyes(app.cardLook()?.blinkPanel() || null); });
  const err = useSyncExternalStore(dress.subscribe, () => dress.get().loadError);
  useSyncExternalStore(dress.subscribe, () => dress.get().rev);
  const loaded = !!W?.loaded;
  const msg = err || !loaded ? err : (W!.notes.length ? W!.notes.join('. ') + '.' : '');
  const kind = err ? 'error' : loaded && W!.failures ? 'error' : '';
  const rows = loaded ? W!.rows() : [], worn = loaded ? W!.outfit : null;
  const groups: [string, PieceRow[]][] = [];
  if (loaded) {
    const add = (title: string, list: PieceRow[]) => { if (list.length) groups.push([title, list]); };
    // a statue's props (pieces of the slot prop: its base, a stump, what it holds) first, as switches of their own
    const props = rows.filter((r) => r.slot === 'prop' && (r.outfit === null || r.outfit === worn)), other = rows.filter((r) => !props.includes(r));
    add('Props', props);
    add('Worn outfit', other.filter((r) => r.outfit !== null && r.outfit === worn));
    add('Loose pieces', other.filter((r) => r.outfit === null));
    for (const o of W!.outfits()) if (o.id !== worn) add(o.label, other.filter((r) => r.outfit === o.id));
  }
  const st = loaded ? W!.stats() : null, t = (st?.timing || {}) as any;
  const repaint = () => setRev((r) => r + 1);
  return (
    <section id="sec-wear" className="sec osb-section-stack" role="tabpanel" aria-labelledby="tab-wear" hidden={!shown}>
      <div className="osb-section">
        <h3 className="osb-eyebrow">Wearables <span className="osb-eyebrow-val">{loaded ? W!.cardName : ''}</span></h3>
        <Note className="sec-lead">A character card with its outfits and wearable packages, from your own files. The page reads them here and sends nothing.</Note>
        <div className="btn-row">
          <TextButton label="Open files" icon="upload" onClick={() => legacy.opener.pickFiles()} />
          <TextButton label="Open folder" icon="folder" onClick={() => legacy.opener.pickFolder()} />
          {loaded ? <TextButton label="Remove" icon="close" quiet onClick={() => { W!.unload(); repaint(); }} /> : null}
        </div>
        {loaded ? null : <div className="wear-drop osb-drop osb-drop--compact"><p className="osb-note">Drop a card and its packages, their folder or their .zip anywhere on the page.</p></div>}
        <p className="inline-msg wear-msg" role="status" data-kind={kind}>{msg}</p>
      </div>
      <div className="wear-body">
        {loaded ? (<>
          <Sub>Outfit</Sub>
          <div className="chips osb-chips" role="group" aria-label="Outfit">
            {[...W!.outfits(), { id: null as any, label: 'No outfit' }].map((o) => (
              <Chip key={o.id ?? 'none'} pressed={o.id === worn} data-outfit={o.id ?? 'none'} onClick={() => { W!.setOutfit(o.id); repaint(); }}>{o.label}</Chip>
            ))}
          </div>
          {groups.map(([title, list]) => (
            <div key={title}>
              <Sub>{title}</Sub>
              <div className="wear-list">
                {list.map((r) => (
                  <div key={r.ref} className="wear-piece" data-ref={r.ref}>
                    <Switch label={r.label} on={r.shown} data-ref={r.ref} tip={r.outfit !== null && r.outfit !== worn ? 'Wear in place of the piece of its slot' : ''}
                      onChange={(on) => { W!.setShown(r.ref, on); repaint(); }} />
                    <span className="wear-meta">{r.slot}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
          <Sub>Skin</Sub>
          <Switch label="Hide covered skin" on={!W!.keepSkin} onChange={(on) => { W!.setKeepSkin(!on); repaint(); }} tip="Leave out the body triangles under the shown pieces" />
          <Note>{plainWords(W!, rows)}</Note>
          <div className="osb-folds">
            <Fold title="Details" count={`${st!.pieces.length} pieces`}>
              <div className="wear-details">
                {rows.filter((r) => r.shown).map((r) => (
                  <p key={r.ref} className="osb-note wear-meta" data-ref={r.ref}>
                    {[r.label, r.slot, `layer ${r.layer}`, ...(r.hiddenGroups.length ? [`${r.hiddenGroups.join(', ')} hidden`] : []), ...(r.pushed ? [`${r.pushed} vertices pushed out, ${r.pushMm.toFixed(1)} mm at most`] : [])].join(' · ')}
                  </p>
                ))}
                <p className="note wear-stats">{`${st!.pieces.length} pieces shown, ${st!.hidden_triangles} of ${st!.body_triangles} body triangles under them.` +
                  (t.total !== undefined ? ` Dressing ${t.total} ms${t.push ? `, push-out ${t.push} ms` : ''}${t.cover ? `, covered skin ${t.cover} ms` : ''}.` : '')}</p>
              </div>
            </Fold>
          </div>
        </>) : null}
      </div>
      {loaded && eyes ? (
        <div className="osb-section wear-eyes">
          <Sub>Eyes</Sub>
          <Switch label="Blink" on={eyes.auto} onChange={(on) => { app.cardLook()?.setBlinkAuto(on); setEyes(app.cardLook()?.blinkPanel() || null); }} tip="Close the eyes now and then, as the exports' blink targets close them" />
          <div className="btn-row"><TextButton label="Blink once" onClick={() => app.cardLook()?.blinkNow()} /></div>
          <Slider id="card-eyes-closed" label="Eyes closed" min={0} max={1} def={0} events={['wardrobe']} get={() => app.cardLook()?.blinkPanel()?.close ?? 0} onInput={(v) => app.cardLook()?.setEyes(v, null)} />
          <Slider id="card-eyes-squint" label="Squint" min={0} max={1} def={0} events={['wardrobe']} get={() => app.cardLook()?.blinkPanel()?.squint ?? 0} onInput={(v) => app.cardLook()?.setEyes(null, v)} />
        </div>
      ) : null}
    </section>
  );
}

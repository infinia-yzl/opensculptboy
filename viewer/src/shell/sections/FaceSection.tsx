// OpenSculptBoy
// Apache License, Version 2.0
//
// Create's Face section in React: Anny's 103 face shapes in folds by part of the head (each with its count of changed
// shapes and a reset), a search over their names, and the spread of Random face. The sliders follow the look through
// the bus, so a drag renders nothing in React; the search and the folds' counts are the only React state.

import { useMemo, useRef, useState } from 'react';
import { useShell } from '../context.ts';
import { useBus } from '../hooks.ts';
import { Slider } from '../components/Slider.tsx';
import { Fold, Note, Search } from '../components/controls.tsx';
import { Icon } from '../components/ui/icon.tsx';

export function FaceSection({ shown }: { shown: boolean }) {
  const { app, ed } = useShell();
  const groups = useMemo(() => app.faceGroups().map((g) => ({ g, title: app.faceGroupTitle(g), sliders: app.faceSliders().filter((s) => s.group === g) })), [app]);
  const [q, setQ] = useState('');
  const [counts, setCounts] = useState<Record<string, number>>({});
  const note = useRef<HTMLSpanElement>(null);
  useBus('look', () => {
    const L = app.look(), next: Record<string, number> = {};
    let total = 0;
    for (const G of groups) { const n = G.sliders.filter((s) => (L.face?.[s.name] ?? 0) !== 0).length; next[G.g] = n; total += n; }
    setCounts((c) => (groups.every((G) => c[G.g] === next[G.g]) ? c : next));
    if (note.current) note.current.textContent = total ? `${total} changed` : '';
  });
  const query = q.trim().toLowerCase();
  const hit = (G: (typeof groups)[number], s: (typeof groups)[number]['sliders'][number]) => !query || s.label.toLowerCase().includes(query) || s.name.includes(query) || G.g.includes(query);
  const any = groups.some((G) => G.sliders.some((s) => hit(G, s)));
  return (
    <section id="sec-face" className="sec osb-section-stack" role="tabpanel" aria-labelledby="tab-face" hidden={!shown}>
      <div className="osb-section">
        <h3 className="osb-eyebrow">Face shapes <span ref={note} className="osb-eyebrow-val" /></h3>
        <Note>Named shapes of the head and the face, which scale with the size of the head. Random face draws from Anny's distribution for the current body.</Note>
        {app.facePrior() ? (
          <div className="sl-compact">
            <Slider id="ed-face-spread" label="Random variation" min={0.25} max={1.5} step={0.05} def={1} readout={(v) => `× ${v.toFixed(2)}`}
              get={() => ed.faceSpread} onInput={(v) => { ed.faceSpread = v; }} />
          </div>
        ) : null}
        <Search placeholder={`Search ${groups.reduce((a, G) => a + G.sliders.length, 0)} face shapes`} value={q} onChange={setQ} />
      </div>
      <div className="folds osb-folds">
        {groups.map((G) => {
          const rows = G.sliders.filter((s) => hit(G, s));
          if (!rows.length) return null;
          const n = counts[G.g] || 0;
          return (
            <Fold key={G.g + (query ? ':q' : '')} title={G.title} count={n ? `${n} changed` : ''} changed={n > 0} open={query ? true : undefined}
              actions={<button type="button" className="fold-reset" aria-label={'Reset the ' + G.title.toLowerCase()} title="Reset group"
                onClick={() => ed.edit('Reset ' + G.title.toLowerCase(), (L) => { for (const s of G.sliders) delete L.face[s.name]; })}><Icon name="undoSmall" size={14} /></button>}>
              {G.sliders.map((s) => (
                <div key={s.name} hidden={!rows.includes(s)}>
                  <Slider id={'ed-f-' + s.name} label={s.label} title={s.name} min={s.range[0]} max={s.range[1]} def={0} ends={s.ends}
                    get={() => app.look().face?.[s.name] ?? 0}
                    onStart={() => ed.begin('Face: ' + s.label)} onEnd={() => ed.end()}
                    onInput={(v) => ed.queue((L) => { L.face = { ...(L.face || {}), [s.name]: v }; if (v === 0) delete L.face[s.name]; })} />
                </div>
              ))}
            </Fold>
          );
        })}
      </div>
      {any ? null : <Note className="empty">No face shape matches.</Note>}
    </section>
  );
}

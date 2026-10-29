// OpenSculptBoy
// Apache License, Version 2.0
//
// Create's Body section in React: Anny's phenotype sliders, as the design's Create screen lays them out: Form,
// Muscle, Weight, Height and Proportions first, and the other sliders behind the More shape fold, with the bar of
// the shares of the three source shapes that mix. Each slider edits the look at most once a frame (Editor.queue) and
// Undo takes a whole drag as one step. The stature in the heading and the share bar paint themselves from the bus.

import { useRef } from 'react';
import { useShell } from '../context.ts';
import { useBus } from '../hooks.ts';
import { Slider } from '../components/Slider.tsx';
import { Icon } from '../components/ui/icon.tsx';

const RACE_COLOURS: Record<string, string> = { african: '#b8845a', asian: '#d9b77e', caucasian: '#e6c9b4' };

export function BodySection({ shown }: { shown: boolean }) {
  const { app, ed } = useShell();
  const all = app.bodySliders();
  const main = all.filter((s) => !s.race && s.name !== 'gender'), more = all.filter((s) => !s.race && s.name === 'gender'), races = all.filter((s) => s.race);
  // the ends that say more than the slider's name (Compact and Elongated for Form); none that describe people
  const ends = (s: (typeof all)[number]) => (s.name === 'gender' || (s.ends || []).some((e) => /^(Min|Max) /.test(e)) ? undefined : s.ends);
  const stature = useRef<HTMLSpanElement>(null), bar = useRef<HTMLDivElement>(null), count = useRef<HTMLSpanElement>(null), fold = useRef<HTMLDetailsElement>(null);
  useBus('look', () => {
    const L = app.look();
    const st = app.stature();
    if (stature.current) stature.current.textContent = st > 0 ? `${st.toFixed(2)} m tall` : '';
    if (bar.current) races.forEach((s, i) => {
      const seg = bar.current!.children[i] as HTMLElement, p = app.raceShare(L.phenotype, s.name);
      seg.style.flexGrow = String(Math.max(p, 0.0001)); seg.title = `${s.label} ${Math.round(p * 100)} %`;
    });
    const folded = [...more, ...races], changed = folded.filter((s) => Math.abs((L.phenotype[s.name] ?? 0.5) - 0.5) > 1e-6).length;
    if (count.current) count.current.textContent = changed ? `${changed} of ${folded.length} changed` : String(folded.length);
    fold.current?.toggleAttribute('data-changed', changed > 0);
  });
  const slider = (s: (typeof all)[number]) => (
    <Slider key={s.name} id={'ed-b-' + s.name} label={s.label} min={0} max={1} def={0.5} ends={ends(s)}
      readout={s.race ? (v) => `${Math.round(100 * app.raceShare({ ...app.look().phenotype, [s.name]: v }, s.name))} %` : undefined}
      get={() => app.look().phenotype[s.name] ?? 0.5}
      onStart={() => ed.begin(s.label)} onEnd={() => ed.end()}
      onInput={(v) => ed.queue((n) => { n.phenotype[s.name] = v; })} />
  );
  return (
    <section id="sec-body" className="sec osb-section-stack" role="tabpanel" aria-labelledby="tab-body" hidden={!shown}>
      <div className="osb-section">
        <h3 className="osb-eyebrow">Shape <span ref={stature} className="osb-eyebrow-val" /></h3>
        <div className="osb-sliders">{main.map(slider)}</div>
      </div>
      {races.length || more.length ? (
        <div className="osb-folds">
          <details ref={fold} className="osb-fold">
            <summary>
              <span className="osb-i"><Icon name="chevron" size={14} /></span>
              <span className="osb-fold-title">More shape</span>
              <span ref={count} className="osb-fold-n" />
            </summary>
            <div className="osb-fold-body">
              {more.length ? <div className="osb-sliders">{more.map(slider)}</div> : null}
              {races.length ? (<>
                <h4 className="osb-eyebrow osb-frieze">Source shapes</h4>
                <p className="osb-note">Three source shapes of the MakeHuman data, mixed by their shares in the bar. Each brings its own proportions of the head and the body.</p>
                <div ref={bar} className="share-bar" aria-hidden="true">
                  {races.map((s) => <span key={s.name} style={{ background: RACE_COLOURS[s.name] || '#999' }}><b>{s.label}</b></span>)}
                </div>
                <div className="osb-sliders">{races.map(slider)}</div>
              </>) : null}
            </div>
          </details>
        </div>
      ) : null}
    </section>
  );
}

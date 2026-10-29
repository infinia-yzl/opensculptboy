// OpenSculptBoy
// Apache License, Version 2.0
//
// The Express mode's section in React: the expressions of the VRM export (the presets, as mixes of the 52 facial
// actions), the visemes and the card's own faces in a fold, and an intensity. The face itself lives in
// face/expressions.ts behind the interface of face/api.ts; the chip shown as chosen follows the face through the bus,
// so a face set elsewhere (a card's own face, a script) shows no chip as chosen.

import { useRef, useState } from 'react';
import { face, faceReady, type FaceState } from '../../face/api.ts';
import { useBus } from '../hooks.ts';
import { Slider } from '../components/Slider.tsx';
import { Chip, Fold, Note } from '../components/controls.tsx';

// the presets in the order of the buttons, with their labels (VRM 1.0 names)
const PRESETS: [string, string][] = [
  ['neutral', 'Neutral'], ['happy', 'Happy'], ['angry', 'Angry'], ['sad', 'Sad'], ['relaxed', 'Relaxed'], ['surprised', 'Surprised'],
  ['blink', 'Blink'], ['blinkLeft', 'Blink left'], ['blinkRight', 'Blink right'],
  ['aa', 'A'], ['ih', 'I'], ['ou', 'U'], ['ee', 'E'], ['oh', 'O'],
];

function sameActions(a: Record<string, number>, b: Record<string, number>) {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const k of keys) if (Math.abs((a[k] || 0) - (b[k] || 0)) > 1e-4) return false;
  return true;
}

export function ExpressSection({ shown }: { shown: boolean }) {
  const [ready, setReady] = useState(faceReady());
  const [others, setOthers] = useState<string[]>([]);
  const [chosen, setChosen] = useState<string | null>(null);
  const [note, setNote] = useState('');
  const sel = useRef({ current: 'neutral', weight: 1 });
  const sync = () => {
    const r = faceReady();
    setReady(r);
    if (!r) return;
    const names = face().presetNames().filter((n) => !PRESETS.some(([p]) => p === n));
    setOthers((o) => (o.join() === names.join() ? o : names));
    const S = sel.current, now = face().get(), want = face().preset(S.current, S.weight);
    const same = !!now && !!want && sameActions(now.actions, want.actions);
    setChosen(same ? S.current : null);
    setNote(same ? (PRESETS.find(([p]) => p === S.current)?.[1] || S.current) : Object.keys(now?.actions || {}).length ? 'Custom' : '');
  };
  useBus('face', sync);
  const show = (fadeMs: number) => {
    if (!faceReady()) return;
    const d: any = face(), S = sel.current, state: FaceState = d.preset(S.current, S.weight);
    // the head and the gaze stay as they are
    const was = d.get();
    if (was.gaze) state.gaze = was.gaze;
    if (was.head) state.head = was.head;
    const done = fadeMs && d.fadeTo ? d.fadeTo(state, fadeMs) : d.set(state);
    done.then(sync);
  };
  const choose = (name: string) => { sel.current.current = name; show(180); };
  const chip = (name: string, label: string) => <Chip key={name} pressed={chosen === name} data-expression={name} onClick={() => choose(name)}>{label}</Chip>;
  return (
    <section id="sec-expression" className="sec osb-section-stack" role="tabpanel" aria-labelledby="tab-expression" hidden={!shown}>
      {ready ? (<>
        <div className="osb-section">
          <h3 className="osb-eyebrow">Expressions <span className="osb-eyebrow-val">{note}</span></h3>
          <div className="chips osb-chips" role="group" aria-label="Expressions">{PRESETS.map(([n, l]) => chip(n, l))}</div>
          <div className="sliders osb-sliders">
            <Slider id="expr-weight" label="Intensity" min={0} max={1} def={1} ends={['None', 'Full']} readout={(v) => `${Math.round(v * 100)} %`}
              events={['face']} get={() => sel.current.weight} onInput={(v) => { sel.current.weight = v; show(0); }} />
          </div>
        </div>
        {others.length ? (
          <div className="osb-folds"><Fold title="Visemes and the card's faces" count={others.length}><div className="chips osb-chips">{others.map((n) => chip(n, n))}</div></Fold></div>
        ) : null}
        <Note>The expressions of the VRM export, as mixes of the 52 facial actions. On a card with its own mouth poses, the vowels play its poses. Left and right are the figure's own.</Note>
      </>) : <Note>This page has no facial actions.</Note>}
    </section>
  );
}

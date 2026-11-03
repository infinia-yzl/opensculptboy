// OpenSculptBoy
// Apache License, Version 2.0
//
// Learning the controls in the studio itself (DESIGN-0014, "Onboarding and discoverability"; artboards 3g, 3h and
// 3i): the first-run card of a desktop with the six most used inputs, the tour prompt the first time posing's Edit
// opens (Show me, Not now, Don't ask again), and the tour's six steps, each of which ticks itself when the camera or
// the registry shows it done. The person's answers stay in this browser.

import { useEffect, useRef, useState } from 'react';
import { saveUi, uiState } from '../ui/dom.ts';
import { cam } from './camera.ts';
import { onActionRun, registerAction, scopeLive } from './actions.ts';
import { setStatusLit } from './status.ts';
import { Icon } from '../shell/components/ui/icon.tsx';

// a mouse with one part lit: the left button, the middle button (its wheel) or the wheel
export function MousePicture({ lit, size = 48 }: { lit: 'left' | 'middle' | 'wheel' | 'right'; size?: number }) {
  return (
    <svg className="mouse-pic" viewBox="0 0 48 64" width={size * 0.75} height={size} aria-hidden="true">
      <path d="M24 3 C12 3 5 11 5 24 V42 C5 54 13 61 24 61 C35 61 43 54 43 42 V24 C43 11 36 3 24 3 Z" className="mouse-body" />
      {lit === 'left' ? <path d="M24 3 C12 3 5 11 5 24 V27 H24 Z" className="mouse-lit" /> : null}
      {lit === 'right' ? <path d="M24 3 C36 3 43 11 43 24 V27 H24 Z" className="mouse-lit" /> : null}
      <path d="M5 27 H43 M24 3 V27" className="mouse-line" />
      <rect x="21" y="9" width="6" height="12" rx="3" className={lit === 'middle' || lit === 'wheel' ? 'mouse-lit mouse-wheel' : 'mouse-wheel'} />
    </svg>
  );
}

const TILES: { pic?: 'left' | 'middle' | 'wheel'; keys?: string[]; title: string; text: string }[] = [
  { pic: 'left', title: 'Left-drag', text: 'Turn' },
  { pic: 'middle', title: 'Middle-drag', text: 'Move the view' },
  { pic: 'wheel', title: 'Wheel', text: 'Zoom' },
  { keys: ['F'], title: 'F', text: 'Frame the selection' },
  { keys: ['`'], title: 'Backtick', text: 'View menu' },
  { keys: ['Ctrl', 'Z'], title: 'Ctrl+Z', text: 'Undo' },
];

// the first-run card of a desktop (artboard 3i)
export function FirstRun({ onClose, note }: { onClose: () => void; note?: React.ReactNode }) {
  const got = useRef<HTMLButtonElement>(null);
  return (
    <section className="osb-glass osb-mended welcome first-run" aria-labelledby="welcome-title">
      <header className="first-run-head">
        <h2 className="osb-title welcome-title" id="welcome-title">Getting round the character</h2>
        <button type="button" className="osb-ib welcome-x" aria-label="Close the welcome" onClick={onClose}><Icon name="close" size={16} /></button>
      </header>
      <ul className="first-run-tiles">
        {TILES.map((t) => (
          <li key={t.title} className="first-run-tile">
            <span className="first-run-pic">{t.pic ? <MousePicture lit={t.pic} /> : t.keys!.map((k) => <kbd key={k} className="osb-key">{k}</kbd>)}</span>
            <b>{t.title}</b><span>{t.text}</span>
          </li>
        ))}
      </ul>
      {note}
      <footer className="first-run-foot">
        <p className="osb-note">F1 lists every key.</p>
        <button ref={got} type="button" className="osb-btn" onClick={onClose}>Got it</button>
      </footer>
    </section>
  );
}

// ------------------------------------------------------------------ the tour
interface Step { n: string; keys: string; status: string; say: string; pic?: 'left' | 'middle' | 'wheel'; done: (s: TourState) => boolean }
interface TourState { yaw0: number; pivot0: number[]; dist0: number; ran: Set<string>; yaw: number; pivot: number[]; dist: number }
const STEPS: Step[] = [
  { n: 'Turn', keys: 'Left-drag', status: 'Left-drag', pic: 'left', say: 'Left-drag on the background turns the camera. Try it.',
    done: (s) => Math.abs(((s.yaw - s.yaw0) % 360 + 540) % 360 - 180) > 20 },
  { n: 'Move the view', keys: 'Middle-drag', status: 'Middle-drag', pic: 'middle', say: 'Middle-drag moves the view, and the spot you grab stays under the pointer.',
    done: (s) => Math.hypot(s.pivot[0] - s.pivot0[0], s.pivot[1] - s.pivot0[1], s.pivot[2] - s.pivot0[2]) > 0.02 },
  { n: 'Zoom', keys: 'Wheel', status: 'Wheel', pic: 'wheel', say: 'The wheel zooms toward the pointer. Hold Shift for small steps.',
    done: (s) => Math.abs(s.dist / s.dist0 - 1) > 0.1 },
  { n: 'Frame', keys: 'F', status: 'F', say: 'F frames the selection, or the whole figure; press it again to widen.', done: (s) => s.ran.has('view.frame') || s.ran.has('view.home') },
  { n: 'View menu', keys: '`', status: '', say: 'The backtick, left of 1, opens the views round the pointer. Pick one.', done: (s) => s.ran.has('view.menu') },
  { n: 'Undo', keys: 'Ctrl Z', status: '', say: 'Ctrl+Z undoes an edit, and Backspace goes back to the earlier view. Try Backspace.', done: (s) => s.ran.has('view.back') || s.ran.has('edit.undo') },
];

type TourMode = 'off' | 'prompt' | 'tour';
let setMode: ((m: TourMode) => void) | null = null;
export function startTour() { setMode?.('tour'); }

export function Tour() {
  const [mode, set] = useState<TourMode>('off');
  const [step, setStep] = useState(0);
  const [, tick] = useState(0);
  const st = useRef<TourState | null>(null);
  setMode = (m) => { set(m); if (m === 'tour') setStep(0); };
  // the prompt the first time posing's Edit opens, unless the person said not to ask again
  useEffect(() => {
    const t = setInterval(() => {
      if (mode !== 'off' || uiState().tourAsked) return;
      if (scopeLive('pose-edit')) { saveUi({ tourAsked: 1 }); set('prompt'); }
    }, 500);
    return () => clearInterval(t);
  }, [mode]);
  // the tour is a command too, and Help lists it
  useEffect(() => registerAction({ id: 'help.tour', label: 'Take the tour of the controls', desc: 'Six short steps: turn, move the view, zoom, frame, the view menu and undo',
    group: 'Help', tier: 'rare', place: 'Help', icon: 'mouse', run: () => startTour() }), []);
  // each step starts from the camera as it is, and ticks itself when the camera or a key shows it done
  useEffect(() => {
    if (mode !== 'tour') { setStatusLit(''); return; }
    const s0 = cam().state();
    st.current = { yaw0: s0.yaw, pivot0: s0.pivot, dist0: s0.distance, ran: new Set(), yaw: s0.yaw, pivot: s0.pivot, dist: s0.distance };
    setStatusLit(STEPS[step]?.status || '');
    const check = () => {
      const s = cam().state(), t = st.current!;
      t.yaw = s.yaw; t.pivot = s.pivot; t.dist = s.distance;
      if (!s.flying && STEPS[step] && STEPS[step].done(t)) { if (step + 1 >= STEPS.length) { set('off'); setStatusLit(''); } else setStep(step + 1); }
    };
    const off1 = cam().onChange(check);
    const off2 = onActionRun((id) => { st.current!.ran.add(id); setTimeout(check, 350); tick((n) => n + 1); });
    return () => { off1(); off2(); };
  }, [mode, step]);
  if (mode === 'prompt') {
    return (
      <section className="osb-glass tour-prompt" aria-labelledby="tour-q">
        <span className="tour-prompt-pic"><Icon name="mouse" size={24} /></span>
        <div>
          <h2 className="osb-heading" id="tour-q">Want a one-minute tour of the controls?</h2>
          <p className="osb-note">Six short steps: turn, move the view, zoom, frame, the view menu and undo.</p>
          <div className="osb-row">
            <button type="button" className="osb-btn osb-btn--primary" onClick={() => set('tour')}><Icon name="play" size={16} />Show me</button>
            <button type="button" className="osb-btn" onClick={() => set('off')}>Not now</button>
            <button type="button" className="osb-link tour-never" onClick={() => { saveUi({ tourAsked: 1, tourNever: 1 }); set('off'); }}>Don’t ask again</button>
          </div>
        </div>
      </section>
    );
  }
  if (mode !== 'tour') return null;
  const S = STEPS[step];
  return (
    <section className="osb-glass tour-card" aria-labelledby="tour-title" aria-live="polite">
      <header className="tour-head"><h2 className="osb-eyebrow" id="tour-title">Tour</h2><span className="osb-tag">Step {step + 1} of {STEPS.length}</span></header>
      <div className="tour-now">
        <span className="tour-pic">{S.pic ? <MousePicture lit={S.pic} size={64} /> : S.keys.split(' ').map((k) => <kbd key={k} className="osb-key">{k}</kbd>)}</span>
        <p>{S.say}</p>
      </div>
      <ol className="tour-steps">
        {STEPS.map((x, i) => (
          <li key={x.n} data-state={i < step ? 'done' : i === step ? 'now' : 'todo'}>
            <span className="tour-dot" aria-hidden="true">{i < step ? <Icon name="check" size={12} /> : null}</span>
            <span>{i + 1}. {x.n}</span><code>{x.keys}</code>
          </li>
        ))}
      </ol>
      <footer className="tour-foot">
        <button type="button" className="osb-btn osb-btn--ghost" onClick={() => set('off')}>End the tour</button>
        <span className="osb-note">The step ticks itself when done</span>
      </footer>
      {S.pic === 'left' ? (
        <svg className="tour-path" aria-hidden="true" viewBox="0 0 280 80" width="280" height="80">
          <path d="M10 60 Q 140 -10 270 40" /><path d="M262 30 L272 42 L258 46 Z" className="tour-arrow" />
        </svg>
      ) : null}
    </section>
  );
}

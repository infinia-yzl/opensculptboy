// OpenSculptBoy
// Apache License, Version 2.0
//
// The New character screen (N, or New in the top bar), as the design's New screen lays it out: the statue starters
// in two kinds (Finished: a fitted card with its sculpted detail, from beside the page; Light: the page's sliders and
// library parts, a few kilobytes), the libraries (skin, eyes, hair and wear) with Start from the base, and your own
// 3D model, which the Open path reads when OpenSculptBoy wrote it and else explains the fit. Under them: Open a card,
// Continue with the character shown, and Random.

import { useEffect, useState } from 'react';
import { useShell } from '../context.ts';
import { useUI } from '../hooks.ts';
import { actions } from '../state.ts';
import { FIT_DOCS } from '../open.ts';
import { finishedUrl } from '../../starters.ts';
import { STONES, stoneSwatch } from '../../stones.ts';
import { Button } from './ui/button.tsx';
import { Icon } from './ui/icon.tsx';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog.tsx';
import { Tip } from './ui/tooltip.tsx';

// the picture of each starter's pose in its slot
const POSE_ICON: Record<string, string> = { weight_shift: 'statue', mh_pointing: 'pose', mh_thinking: 'silhouette', ready: 'throw', mh_step: 'walk', mh_at_ease: 'taichi' };

export function NewScreen() {
  const { legacy } = useShell();
  const open = useUI((s) => s.newDialog), dragging = useUI((s) => s.dragging);
  const [pick, setPick] = useState(legacy.starters[0]?.id || '');
  const [kind, setKind] = useState<'finished' | 'light'>('light');
  const [finished, setFinished] = useState<Record<string, boolean>>({});
  const [name, setName] = useState('');
  // which finished starters lie beside the page (a page opened as a file reads none)
  useEffect(() => {
    if (!open) return;
    setName(legacy.identity().name);
    if (location.protocol === 'file:') return;
    let live = true;
    Promise.all(legacy.starters.map(async (s) => {
      try { const r = await fetch(finishedUrl(s), { method: 'HEAD' }); return [s.id, r.ok] as const; } catch (e) { return [s.id, false] as const; }
    })).then((rs) => { if (live) setFinished(Object.fromEntries(rs)); });
    return () => { live = false; };
  }, [open, legacy]);
  const S = legacy.starters.find((s) => s.id === pick) || legacy.starters[0];
  const anyFinished = Object.values(finished).some(Boolean);
  const library = (icon: string, label: string, go: () => void) => (
    <button type="button" className="osb-stage new-tile" onClick={() => { actions.hideNew(); go(); }}>
      <Icon name={icon} size={40} /><span className="new-tile-label">{label}</span>
    </button>
  );
  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) actions.hideNew(); }}>
      <DialogContent className="new-screen osb-stage osb-veins" aria-describedby="new-lead">
        <div className="new-head">
          <DialogTitle className="osb-display" id="new-title">New character</DialogTitle>
          <DialogDescription id="new-lead" className="osb-note">Start from a statue, from the base with the libraries, or from a 3D model of your own.</DialogDescription>
        </div>
        <div className="new-grid">
          <article className="osb-card new-card new-statues" aria-labelledby="new-s">
            <div className="osb-row new-card-head">
              <h2 className="osb-title" id="new-s">Statue</h2>
              <div className="osb-seg" role="radiogroup" aria-label="Kind">
                <Tip label="Finished: the sculpted statue with its own hair and drapery" side="top">
                  <button type="button" role="radio" aria-checked={kind === 'finished'} aria-pressed={kind === 'finished'} onClick={() => setKind('finished')}>Finished</button>
                </Tip>
                <Tip label="Light: small file, every part editable" side="top">
                  <button type="button" role="radio" aria-checked={kind === 'light'} aria-pressed={kind === 'light'} onClick={() => setKind('light')}>Light</button>
                </Tip>
              </div>
            </div>
            <div className="new-slots" role="radiogroup" aria-label="Statues">
              {legacy.starters.map((s) => {
                const st = STONES.find((x) => x.id === s.light.skin.stone) || STONES[0];
                return (
                  <button key={s.id} type="button" role="radio" className="osb-slot new-slot" aria-checked={s.id === pick} aria-pressed={s.id === pick} data-starter={s.id}
                    onClick={() => setPick(s.id)} onDoubleClick={() => legacy.startStarter(s.id, kind)}>
                    <span className="osb-slot-pic new-slot-pic" style={{ background: stoneSwatch(st) }}><Icon name={POSE_ICON[s.pose] || 'statue'} size={72} /></span>
                    <span className="new-slot-name">{s.name}</span>
                    <span className="osb-note new-slot-stone">{st.label}</span>
                  </button>
                );
              })}
            </div>
            <p className="osb-note new-desc">{S?.description}.{kind === 'finished' && !finished[S?.id || ''] ? ' The finished statues come with the statue cards; until then this starts the light version.' : ''}</p>
            <Button size="lg" className="new-start" onClick={() => S && legacy.startStarter(S.id, kind === 'finished' && finished[S.id] ? 'finished' : 'light')}>Start with the {S?.name}</Button>
            {anyFinished ? null : <span className="osb-tag new-planned">Finished statues: planned</span>}
          </article>
          <article className="osb-card new-card" aria-labelledby="new-l">
            <h2 className="osb-title" id="new-l">Libraries</h2>
            <div className="new-tiles">
              {library('skin', 'Skin', () => actions.openMode('create', 'skin'))}
              {library('eye', 'Eyes', () => actions.openMode('create', 'skin'))}
              {library('hair', 'Hair', () => actions.openMode('create', 'hair'))}
              {library('wear', 'Wear', () => actions.openMode('dress'))}
            </div>
            <Button variant="outline" size="lg" onClick={legacy.startBase}>Start from the base</Button>
          </article>
          <article className="osb-card new-card" aria-labelledby="new-m">
            <h2 className="osb-title" id="new-m">Your 3D model</h2>
            <div className="osb-drop new-drop" data-over={dragging ? '' : undefined}>
              <Icon name="model" size={64} />
              <p className="osb-note">Drop a GLB or VRM file that OpenSculptBoy wrote, or a card's .zip, anywhere on the page. Another 3D model becomes a character through a fit.</p>
              <a className="osb-link" href={FIT_DOCS} target="_blank" rel="noopener noreferrer">How to fit a 3D model</a>
              <span className="osb-tag osb-tag--mono new-exts">.glb .vrm .zip .json</span>
            </div>
            <Button variant="outline" size="lg" onClick={() => legacy.opener.pickFiles()}><Icon name="upload" size={16} />Choose a model</Button>
          </article>
        </div>
        <div className="osb-row new-foot">
          <Button variant="ghost" onClick={() => actions.showOpen()}><Icon name="open" size={16} />Open a card</Button>
          <span className="osb-sep" />
          <Button variant="ghost" onClick={() => actions.hideNew()}><Icon name="continue" size={16} />Continue {name}</Button>
          <span className="osb-sep" />
          <Button variant="ghost" onClick={() => { legacy.random.character(); actions.hideNew(); }}><Icon name="dice" size={16} />Random</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

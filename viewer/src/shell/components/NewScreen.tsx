// OpenSculptBoy
// Apache License, Version 2.0
//
// The New character screen (N, or New in the top bar), as the design's New screen lays it out: the statue starters
// (fitted statues from their packages beside the page, src/starters.ts) and a blank body, the libraries (skin, eyes,
// hair and wear) with Start from the base, and your own 3D model, which the Open path reads when OpenSculptBoy wrote
// it and else explains the fit. Under them: Open a card, Continue with the character shown, and Random. A statue
// whose package is not beside the page carries a Download tag, and picking it shows a note with its download link.

import { useEffect, useRef, useState } from 'react';
import { useShell } from '../context.ts';
import { useUI } from '../hooks.ts';
import { actions } from '../state.ts';
import { FIT_DOCS } from '../open.ts';
import { packageUrl, sizeLabel } from '../../starters.ts';
import { STARTER_THUMBS } from '../../starter_thumbs.ts';
import { STONES, stoneSwatch } from '../../stones.ts';
import { Button } from './ui/button.tsx';
import { Icon } from './ui/icon.tsx';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog.tsx';

// the blank body: anny's default body, the slot after the statues
const BLANK = 'blank';

export function NewScreen() {
  const { legacy } = useShell();
  const open = useUI((s) => s.newDialog), dragging = useUI((s) => s.dragging), starter = useUI((s) => s.starter);
  const [pick, setPick] = useState(legacy.starters[0]?.id || BLANK);
  // which packages lie beside the page (true, false, or undefined until the page has asked; a page opened as a file
  // reads none)
  const [there, setThere] = useState<Record<string, boolean>>({});
  const [name, setName] = useState('');
  const start = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    setName(legacy.identity().name);
    if (location.protocol === 'file:') { setThere(Object.fromEntries(legacy.starters.map((s) => [s.id, false]))); return; }
    let live = true;
    Promise.all(legacy.starters.map(async (s) => {
      try { const r = await fetch(packageUrl(s), { method: 'HEAD', cache: 'no-store' }); return [s.id, r.ok] as const; } catch (e) { return [s.id, false] as const; }
    })).then((rs) => { if (live) setThere(Object.fromEntries(rs)); });
    return () => { live = false; };
  }, [open, legacy]);
  const S = legacy.starters.find((s) => s.id === pick) || null;
  const note = starter.note && starter.note.id === pick ? starter.note : null;
  const go = () => { if (S) legacy.startStarter(S.id); else legacy.startBase(); };
  const library = (icon: string, label: string, fn: () => void) => (
    <button type="button" className="osb-stage new-tile" onClick={() => { actions.hideNew(); fn(); }}>
      <Icon name={icon} size={40} /><span className="new-tile-label">{label}</span>
    </button>
  );
  const choose = (id: string) => { setPick(id); if (starter.note) actions.setStarter({ busy: '', note: null }); };
  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) actions.hideNew(); }}>
      <DialogContent className="new-screen osb-stage osb-veins" aria-describedby="new-lead" onOpenAutoFocus={(e) => { e.preventDefault(); start.current?.focus(); }}>
        <div className="new-head">
          <DialogTitle className="osb-display" id="new-title">New character</DialogTitle>
          <DialogDescription id="new-lead" className="osb-note">Start from a statue or a blank body, from the base with the libraries, or from a 3D model of your own.</DialogDescription>
        </div>
        <div className="new-grid">
          <article className="osb-card new-card new-statues" aria-labelledby="new-s">
            <div className="osb-row new-card-head">
              <h2 className="osb-title" id="new-s">Statue</h2>
            </div>
            <div className="new-slots" role="radiogroup" aria-label="Starters">
              {legacy.starters.map((s) => {
                const st = STONES.find((x) => x.id === s.stone) || STONES[0], pic = STARTER_THUMBS[s.id];
                return (
                  <button key={s.id} type="button" role="radio" className="osb-slot new-slot" aria-checked={s.id === pick} aria-pressed={s.id === pick} data-starter={s.id}
                    data-package={there[s.id] === undefined ? undefined : there[s.id] ? 'here' : 'missing'}
                    onClick={() => choose(s.id)} onDoubleClick={() => legacy.startStarter(s.id)}>
                    <span className={'osb-slot-pic new-slot-pic' + (pic ? ' new-slot-statue' : '')} style={pic ? undefined : { background: stoneSwatch(st) }}>
                      {pic ? <img src={pic} alt="" /> : <Icon name="statue" size={72} />}
                      {there[s.id] === false ? <span className="osb-tag new-slot-tag">Download</span> : null}
                    </span>
                    <span className="new-slot-name">{s.name}</span>
                    <span className="osb-note new-slot-stone">{st.label} · {sizeLabel(s.package.bytes)}</span>
                  </button>
                );
              })}
              <button type="button" role="radio" className="osb-slot new-slot" aria-checked={pick === BLANK} aria-pressed={pick === BLANK} data-starter={BLANK}
                onClick={() => choose(BLANK)} onDoubleClick={() => legacy.startBase()}>
                <span className="osb-slot-pic new-slot-pic new-slot-blank"><Icon name="silhouette" size={72} /></span>
                <span className="new-slot-name">Blank body</span>
                <span className="osb-note new-slot-stone">Anny's default</span>
              </button>
            </div>
            {note ? (
              <div className="osb-error new-note" role="alert">
                <span className="osb-i"><Icon name="alert" /></span>
                <p>{note.message}</p>
                <a className="osb-btn" href={note.link} target="_blank" rel="noopener noreferrer"><Icon name="download" size={16} />{note.label}</a>
              </div>
            ) : (
              <p className="osb-note new-desc">{S ? S.description + '.' : 'Anny\'s default body in the A pose, with nothing on it: shape it from the start.'}</p>
            )}
            <Button ref={start} size="lg" className="new-start" disabled={!!starter.busy} onClick={go}>
              {starter.busy && S && starter.busy === S.id ? `Fetching ${S.name}` : S ? `Start with ${S.name}` : 'Start from a blank body'}
            </Button>
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

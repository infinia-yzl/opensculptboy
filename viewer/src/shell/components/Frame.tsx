// OpenSculptBoy
// Apache License, Version 2.0
//
// The parts of the frame around the canvas: the view bar (framings, angles, the turntable, the reset and the
// refinement ring), the welcome card of the first visit, and the Open dialog with its drop zone.

import { useLayoutEffect, useRef, useState } from 'react';
import { useShell } from '../context.ts';
import { useBus, useUI } from '../hooks.ts';
import { actions } from '../state.ts';
import { FIT_DOCS } from '../open.ts';
import { starterOfName } from '../../starters.ts';
import { Button } from './ui/button.tsx';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog.tsx';
import { Icon } from './ui/icon.tsx';
import { Tip } from './ui/tooltip.tsx';
import { runAction } from '../../studio/actions.ts';
import { useActionsVersion } from '../../studio/react.ts';
import { FirstRun } from '../../studio/Onboarding.tsx';

// the framings and the angles of the view bar; an angle pressed again turns to the opposite side
const FRAMES: [string, string, string, string][] = [['body', 'Full', 'Full body', 'The whole figure in the free space'], ['upper', 'Upper', 'Upper body', 'The head, the arms and the chest'],
  ['face', 'Face', 'Face', 'The head and the hair'], ['hand', 'Hand', 'Hand', 'A hand; again for the other hand']];
const ANGLES: [string, string, string, string][] = [['front', 'Front', 'Front', 'Again for the back'], ['three', '¾', 'Three-quarter', 'The front, turned a quarter'],
  ['side', 'Side', 'Side', 'Again for the other side'], ['back', 'Back', 'Back', 'Again for the front']];

export function ViewBar() {
  const { legacy } = useShell();
  useActionsVersion();
  const view = useUI((s) => s.view), width = useUI((s) => s.width);
  const [unfolded, setUnfolded] = useState(false);
  const ring = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => { ring.current?.append(legacy.ring); }, [legacy]);
  const phone = width === 'phone';
  const items = (
    <>
      <div className="osb-seg" role="group" aria-label="Framing">
        {FRAMES.map(([v, short, label, desc]) => (
          <Tip key={v} label={label} desc={desc} side="top">
            <button type="button" aria-pressed={view.frame === v} data-value={v} aria-label={label} onClick={() => legacy.frame(v)}>{short}</button>
          </Tip>
        ))}
      </div>
      <span className="osb-sep" />
      <div className="osb-seg" role="group" aria-label="Angle">
        {ANGLES.map(([v, short, label, desc]) => (
          <Tip key={v} label={label} desc={desc} side="top">
            <button type="button" aria-pressed={view.angle === v} data-value={v} aria-label={label} onClick={() => legacy.angle(v)}>{short}</button>
          </Tip>
        ))}
      </div>
      <span className="osb-sep" />
      {phone ? null : (<>
        <Tip action="view.frame" side="top">
          <button type="button" className="osb-ib" aria-label="Frame the selection (F)" onClick={() => runAction('view.frame')}><Icon name="focus" /></button>
        </Tip>
        <Tip action="view.menu" side="top">
          <button type="button" className="osb-ib" aria-label="View menu (backtick)" data-ring="view" onClick={() => runAction('view.menu')}><Icon name="compass" /></button>
        </Tip>
        <Tip action="display.menu" side="top">
          <button type="button" className="osb-ib" aria-label="Display menu (Z)" data-ring="display" onClick={() => runAction('display.menu')}><Icon name="display" /></button>
        </Tip>
      </>)}
      {phone ? (<>
        <Tip label="Turn the figure" side="top">
          <button type="button" className="osb-ib" aria-label="Turn the figure" aria-pressed={view.turntable} onClick={legacy.turntable}><Icon name="turntable" /></button>
        </Tip>
        <span className="osb-sep" />
        <div ref={ring} className="osb-ring-host" />
      </>) : null}
    </>
  );
  if (phone) {
    return (
      <div className={'viewbar viewbar--phone' + (unfolded ? ' open' : '')} role="toolbar" aria-label="View">
        <div className="osb-glass osb-toolbar vb-items" hidden={!unfolded}>{items}</div>
        <button type="button" className="osb-ib osb-glass vb-toggle" aria-label="View" aria-expanded={unfolded} onClick={() => setUnfolded(!unfolded)}><Icon name="camera" /></button>
      </div>
    );
  }
  return <div className="osb-glass osb-toolbar viewbar" role="toolbar" aria-label="View">{items}</div>;
}

export function Welcome() {
  const shown = useUI((s) => s.welcome), width = useUI((s) => s.width), note = useUI((s) => s.starter.note);
  const { legacy } = useShell();
  // the name of the character shown: a statue's package opens after the first frame
  const [name, setName] = useState(() => legacy.identity().name);
  useBus(['look', 'wardrobe'], () => setName(legacy.identity().name));
  if (!shown) return null;
  const starter = starterOfName(name);
  const touch = typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches;
  // a desktop: the six most used inputs (artboard 3i), with the starter's note when its package is missing
  if (!touch && width !== 'phone') {
    return <FirstRun onClose={() => actions.dismissWelcome()} note={!starter && note ? (
      <p className="osb-note welcome-starter">The statue starters come as packages of their own. <a className="osb-link" href={note.link} target="_blank" rel="noopener noreferrer">{note.label}</a> and drop it on the page to start from it.</p>
    ) : undefined} />;
  }
  return (
    <section className="osb-glass osb-mended welcome" aria-labelledby="welcome-title" data-width={width}>
      <Tip label="Close" keys="Esc" side="left">
        <button type="button" className="osb-ib welcome-x" aria-label="Close the welcome" onClick={() => actions.dismissWelcome()}><Icon name="close" size={16} /></button>
      </Tip>
      <h2 className="osb-title welcome-title" id="welcome-title">{starter ? `Meet ${starter.name}` : 'Meet the studio'}</h2>
      <p className="osb-note">{starter ? 'A statue to start from. Shape it, dress it, pose it and export it. Your files stay in this browser.' : 'Shape a character here, then dress, pose and export it. Your files stay in this browser.'}</p>
      {!starter && note ? (
        <p className="osb-note welcome-starter">The statue starters come as packages of their own. <a className="osb-link" href={note.link} target="_blank" rel="noopener noreferrer">{note.label}</a> and drop it on the page to start from it.</p>
      ) : null}
      <p className="osb-note welcome-keys">
        {touch
          ? <><b>Drag</b> to turn · <b>pinch</b> to zoom · <b>two fingers</b> to move</>
          : <><b>Drag</b> to turn · <b>scroll</b> to zoom · <b>?</b> for the keys</>}
      </p>
      <div className="osb-row">
        <Button onClick={() => { actions.dismissWelcome(); actions.showNew(); }}><Icon name="new" size={16} />New character</Button>
        <Button variant="ghost" onClick={() => actions.dismissWelcome()}>Got it</Button>
      </div>
    </section>
  );
}

export function OpenDialog() {
  const { legacy } = useShell();
  const open = useUI((s) => s.openDialog), busy = useUI((s) => s.openBusy), error = useUI((s) => s.openError), dragging = useUI((s) => s.dragging);
  const O = legacy.opener;
  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) actions.hideOpen(); }}>
      <DialogContent className="open-dialog" aria-describedby="open-lead">
        <DialogTitle className="osb-display osb-open-title">Open a character</DialogTitle>
        <DialogDescription id="open-lead" className="osb-lead">Your own files open here and stay in this browser. Drop them anywhere on the page, or choose them.</DialogDescription>
        {error ? (
          <div className="osb-error" role="alert">
            <span className="osb-i"><Icon name="alert" /></span>
            <h3 className="osb-heading">{error.title}</h3>
            <p>{error.body}</p>
            <div className="osb-row">
              {error.action ? <Button onClick={error.action.run}>{error.action.label}</Button> : null}
              {error.fit ? <a className="osb-btn" href={FIT_DOCS} target="_blank" rel="noopener noreferrer"><Icon name="link" size={16} />How to fit a 3D model</a> : null}
              <Button variant="ghost" onClick={() => O.pickFiles()}>Choose other files</Button>
            </div>
          </div>
        ) : null}
        <div className="osb-drop" data-over={dragging ? '' : undefined}>
          <p className="osb-drop-say"><Icon name="open" size={16} />Drop to open the files</p>
          <dl className="osb-formats">
            <dt>Character card</dt><dd><code>.json</code> with its files, their folder, or their <code>.zip</code></dd>
            <dt>OpenSculptBoy model</dt><dd><code>.glb</code> or <code>.vrm</code>: the card it holds</dd>
            <dt>Wearables</dt><dd><code>.wearable.json</code> packages with their files</dd>
            <dt>Sheet settings</dt><dd>a <code>.json</code> file of the Sheet mode</dd>
            <dt>Picture</dt><dd><code>.png</code> or <code>.jpg</code>, to pose the figure as it stands</dd>
          </dl>
          <div className="osb-row">
            <Button onClick={() => O.pickFiles()}><Icon name="file" size={16} />Choose files</Button>
            <Button variant="outline" onClick={() => O.pickFolder()}><Icon name="folder" size={16} />Choose a folder</Button>
          </div>
          <p className="osb-note">Another 3D model? <a className="osb-link" href={FIT_DOCS} target="_blank" rel="noopener noreferrer">How to fit it</a></p>
        </div>
        <p className="osb-note open-status" role="status">{busy ? busy + '…' : ''}</p>
      </DialogContent>
    </Dialog>
  );
}

// the progress card of a long step (a card's open): its steps, done, now or to do, and a bar
export function ProgressCard() {
  const p = useUI((s) => s.progress);
  if (!p) return null;
  const done = p.steps.filter(([, st]) => st === 'done').length;
  return (
    <section className="osb-glass osb-progress progress-card" role="status" aria-live="polite" aria-labelledby="progress-title">
      <h2 className="osb-heading" id="progress-title">{p.title}</h2>
      <div className="osb-bar" style={{ ['--p' as any]: Math.max(0.04, (done + 0.5) / p.steps.length) }}><span /></div>
      <ol className="osb-steps">
        {p.steps.map(([label, st], i) => (
          <li key={label} data-state={st}><span className="osb-num">{String(i + 1).padStart(2, '0')}</span><span>{label}</span><span className="osb-mark">{st === 'done' ? 'Done' : st === 'now' ? 'Now' : ''}</span></li>
        ))}
      </ol>
    </section>
  );
}

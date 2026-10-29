// OpenSculptBoy
// Apache License, Version 2.0
//
// The parts of the frame around the canvas: the view bar (framings, angles, the turntable, the reset and the
// refinement ring), the welcome card of the first visit, and the Open dialog with its drop zone.

import { useLayoutEffect, useRef, useState } from 'react';
import { useShell } from '../context.ts';
import { useUI } from '../hooks.ts';
import { actions } from '../state.ts';
import { FIT_DOCS } from '../open.ts';
import { Button } from './ui/button.tsx';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog.tsx';
import { Icon } from './ui/icon.tsx';
import { Tip } from './ui/tooltip.tsx';

const FRAMES: [string, string, string, string][] = [['body', 'Full', 'Full body', '1'], ['upper', 'Upper', 'Upper body', '2'], ['face', 'Face', 'Face', '3']];
const ANGLES: [string, string, string, string][] = [['front', 'Front', 'Front', '4'], ['three', '¾', 'Three-quarter', '5'], ['side', 'Side', 'Side', '6'], ['back', 'Back', 'Back', '7']];

export function ViewBar() {
  const { legacy } = useShell();
  const view = useUI((s) => s.view), width = useUI((s) => s.width);
  const [unfolded, setUnfolded] = useState(false);
  const ring = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => { ring.current?.append(legacy.ring); }, [legacy]);
  const phone = width === 'phone';
  const items = (
    <>
      <div className="osb-seg" role="group" aria-label="Framing">
        {FRAMES.map(([v, short, label, key]) => (
          <Tip key={v} label={label} keys={key} side="top">
            <button type="button" aria-pressed={view.frame === v} data-value={v} aria-label={label} onClick={() => legacy.frame(v)}>{short}</button>
          </Tip>
        ))}
      </div>
      <span className="osb-sep" />
      <div className="osb-seg" role="group" aria-label="Angle">
        {ANGLES.map(([v, short, label, key]) => (
          <Tip key={v} label={label} keys={key} side="top">
            <button type="button" aria-pressed={view.angle === v} data-value={v} aria-label={label} onClick={() => legacy.angle(v)}>{short}</button>
          </Tip>
        ))}
      </div>
      <span className="osb-sep" />
      <Tip label="Turn the figure" keys="T" side="top">
        <button type="button" className="osb-ib" aria-label="Turn the figure (T)" aria-pressed={view.turntable} onClick={legacy.turntable}><Icon name="turntable" /></button>
      </Tip>
      <Tip label="Reset the view" keys="0" side="top">
        <button type="button" className="osb-ib" aria-label="Reset the view (0)" onClick={legacy.resetView}><Icon name="reset" /></button>
      </Tip>
      <span className="osb-sep" />
      <div ref={ring} className="osb-ring-host" />
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
  const shown = useUI((s) => s.welcome), width = useUI((s) => s.width);
  if (!shown) return null;
  const touch = typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches;
  return (
    <section className="osb-glass osb-mended welcome" aria-labelledby="welcome-title" data-width={width}>
      <Tip label="Close" keys="Esc" side="left">
        <button type="button" className="osb-ib welcome-x" aria-label="Close the welcome" onClick={() => actions.dismissWelcome()}><Icon name="close" size={16} /></button>
      </Tip>
      <h2 className="osb-title welcome-title" id="welcome-title">Welcome to OpenSculptBoy</h2>
      <p className="osb-note">Make a humanoid 3D character, then dress it, give it expressions, pose it and export it. Everything stays in this browser.</p>
      <p className="osb-note welcome-keys">
        {touch
          ? <><b>Drag</b> to turn · <b>pinch</b> to zoom · <b>two fingers</b> to move</>
          : <><b>Drag</b> to turn · <b>right-drag</b> to move · <b>scroll</b> to zoom · <b>?</b> for the keys</>}
      </p>
      <div className="osb-row">
        <Button onClick={() => actions.showOpen()}><Icon name="open" size={16} />Open a character</Button>
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
              {error.fit ? <a className="osb-btn" href={FIT_DOCS} target="_blank" rel="noopener noreferrer"><Icon name="link" size={16} />How to fit a 3D model</a> : null}
              <Button variant="ghost" onClick={() => O.pickFiles()}>Choose other files</Button>
            </div>
          </div>
        ) : null}
        <div className="osb-drop" data-over={dragging ? '' : undefined}>
          <p className="osb-drop-say"><Icon name="open" size={16} />Drop to open the files</p>
          <dl className="osb-formats">
            <dt>Character card</dt><dd><code>.json</code> with its files, or their folder</dd>
            <dt>Wearables</dt><dd><code>.wearable.json</code> packages with their files</dd>
            <dt>Sheet settings</dt><dd>a <code>.json</code> file of the Sheet mode</dd>
            <dt>Page character</dt><dd>a <code>.json</code> file from Export</dd>
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

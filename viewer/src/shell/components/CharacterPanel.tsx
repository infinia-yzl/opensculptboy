// OpenSculptBoy
// Apache License, Version 2.0
//
// The Character panel (the character's button in the top bar): what the character holds, part by part, each with
// where it comes from (the card, or the page's library) and a link to the mode that changes it: the body, the face,
// the skin (a stone, the page's skin or the card's baked skin), the eyes, the hair, the outfit and the faces. For a
// light starter it offers the finished version, and for a character of the page it links to the fit, with one line on
// what a fit adds. It is a landmark with a heading, so a screen reader hears what the character holds.

import { useEffect, useState } from 'react';
import { face, faceReady } from '../../face/api.ts';
import { useShell } from '../context.ts';
import { useBus, useUI } from '../hooks.ts';
import { actions, ui } from '../state.ts';
import { FIT_DOCS } from '../open.ts';
import { stoneOf } from '../../stones.ts';
import { Icon } from './ui/icon.tsx';
import { Tip } from './ui/tooltip.tsx';

interface Part { id: string; label: string; value: string; card: boolean; go: () => void }

export function CharacterPanel() {
  const { app, legacy } = useShell();
  const shown = useUI((s) => s.character), width = useUI((s) => s.width), tool = useUI((s) => s.tool);
  const [rows, setRows] = useState<{ parts: Part[]; kind: string; starter: string | null }>({ parts: [], kind: '', starter: null });
  const refresh = () => {
    if (!ui.get().character) return;
    const L = app.look(), w = app.wardrobe(), c = app.cardLook()?.summary() || null, id = legacy.identity();
    const st = stoneOf(L.skin), spec = app.hairSpecs().find((s) => s.name === L.hair.style);
    const faceN = Object.keys(L.face || {}).length;
    const pieces = w?.loaded ? w.pieces.length : 0, outfit = w?.loaded ? w.outfits().find((o) => o.id === w.outfit)?.label : null;
    const faces = faceReady() ? face().presetNames().length : 0;
    const parts: Part[] = [
      { id: 'body', label: 'Body', value: c?.detail ? `Sliders and detail level ${c.detail}` : 'Sliders', card: !!w?.loaded, go: () => actions.openMode('create', 'body') },
      { id: 'face', label: 'Face', value: faceN ? `${faceN} shapes` : 'Anny\'s default', card: !!w?.loaded, go: () => actions.openMode('create', 'face') },
      { id: 'skin', label: 'Skin', value: st ? st.label : c?.skin ? 'Baked skin' : 'Page skin', card: !st && !!c?.skin, go: () => actions.openMode('create', 'skin') },
      { id: 'eyes', label: 'Eyes', value: st ? 'Stone' : c?.eyes ? 'Own' : 'Page eyes', card: !st && !!c?.eyes, go: () => actions.openMode('create', 'skin') },
      { id: 'hair', label: 'Hair', value: c?.hair ? 'Own' : app.hairVisible() ? spec?.label || L.hair.style : 'None', card: !!c?.hair, go: () => actions.openMode('create', 'hair') },
      { id: 'outfit', label: 'Outfit', value: pieces ? `${outfit || 'Loose pieces'}, ${pieces} pieces` : 'None', card: pieces > 0, go: () => actions.openMode('dress') },
      { id: 'faces', label: 'Faces', value: String(faces || 0), card: !!c?.faces, go: () => actions.openMode('express') },
    ];
    const starter = legacy.starters.find((s) => s.name === id.name || id.name.startsWith(s.name + ' ·'))?.id || null;
    setRows({ parts, kind: id.kind || (w?.loaded ? 'Card' : 'Page character'), starter });
  };
  useBus(['look', 'wardrobe', 'face', 'motion'], refresh);
  useEffect(refresh, [shown]);
  if (!shown || (tool && width !== 'wide')) return null;
  return (
    <aside id="char-panel" className="osb-char osb-glass char-panel" aria-labelledby="char-title">
      <div className="osb-row char-head">
        <h2 className="osb-title" id="char-title">Character</h2>
        <span className="osb-tag">{rows.kind}</span>
        <Tip label="Close" side="left"><button type="button" className="osb-ib" aria-label="Close the character panel" onClick={() => actions.toggleCharacter()}><Icon name="close" size={16} /></button></Tip>
      </div>
      <dl className="osb-char-list">
        {rows.parts.map((p) => [
          <dt key={p.id + 't'}><span className={'osb-src ' + (p.card ? 'osb-src--card' : 'osb-src--page')} role="img" aria-label={p.card ? 'From the card' : 'From the library'} />{p.label}</dt>,
          <dd key={p.id + 'd'}>{p.value}</dd>,
          <button key={p.id + 'b'} type="button" className="osb-ib" aria-label={'Open ' + p.label} onClick={p.go}><Icon name="chevron" size={16} /></button>,
        ])}
      </dl>
      <p className="osb-note char-legend"><span className="osb-src osb-src--card" aria-hidden="true" />card <span className="osb-src osb-src--page" aria-hidden="true" />library</p>
      {rows.starter ? (
        <button type="button" className="osb-link char-link" onClick={() => legacy.startStarter(rows.starter!, 'finished')}>Load the finished version</button>
      ) : !app.wardrobe()?.loaded ? (
        <p className="osb-note">A fit to a 3D model adds its detail, its baked skin, its own hair, eyes and mouth. <a className="osb-link" href={FIT_DOCS} target="_blank" rel="noopener noreferrer">Fit to a reference</a></p>
      ) : null}
    </aside>
  );
}

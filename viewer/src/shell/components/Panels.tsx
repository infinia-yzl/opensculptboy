// OpenSculptBoy
// Apache License, Version 2.0
//
// The panels: the mode's panel on the left (the design system's Panel, a slab with its heading, Create's sub-tabs, a
// scrolling body and a foot of actions), the tool panel on the right (Rig, Scene and Settings from the View menu), the
// rail of the modes at medium widths and the phone's bar of six modes, where the mode's panel becomes a bottom sheet.
// Every section of every mode stays mounted, so a section keeps its state while another shows.

import { useRef } from 'react';
import { MODES, SECTIONS, TOOLS, type ModeId } from '../../ui/shortcuts.ts';
import { useShell } from '../context.ts';
import { useUI } from '../hooks.ts';
import { actions, ui } from '../state.ts';
import { BodySection } from '../sections/BodySection.tsx';
import { DressSection } from '../sections/DressSection.tsx';
import { ExportSection } from '../sections/ExportSection.tsx';
import { ExpressSection } from '../sections/ExpressSection.tsx';
import { FaceSection } from '../sections/FaceSection.tsx';
import { HairSection } from '../sections/HairSection.tsx';
import { PoseSection } from '../sections/PoseSection.tsx';
import { SettingsSection } from '../sections/SettingsSection.tsx';
import { SheetSection } from '../sections/SheetSection.tsx';
import { SkinSection, setStone } from '../sections/SkinSection.tsx';
import { stoneOf } from '../../stones.ts';
import { DomSlot } from './DomSlot.tsx';
import { Button } from './ui/button.tsx';
import { Icon } from './ui/icon.tsx';
import { Tabs, TabsList, TabsTrigger } from './ui/tabs.tsx';
import { Tip } from './ui/tooltip.tsx';


function Foot({ mode, sub }: { mode: ModeId; sub: string }) {
  const { app, ed, legacy } = useShell();
  if (mode !== 'create') return null;
  const R = legacy.random, X = legacy.reset;
  // Use own: a statue becomes its own skin again (a card's baked skin, or the page's skin)
  const own = () => { if (stoneOf(app.look().skin)) ed.edit('Use own skin', (n) => setStone(n, null)); };
  const pair: Record<string, [string, () => void, string, () => void] | [null, null, string, () => void]> = {
    body: ['Reset', X.body, 'Random', R.body],
    face: ['Reset', X.face, 'Random', R.face],
    hair: [null, null, 'Random hair', R.hair],
    skin: ['Use own', own, 'Random colours', R.colours],
  };
  const p = pair[sub];
  if (!p) return null;
  const randomOff = sub === 'face' && !app.facePrior();
  return (
    <footer className="osb-panel-foot">
      {p[0] ? <Button variant="ghost" onClick={p[1]!}><Icon name="undoSmall" size={16} />{p[0]}</Button> : null}
      {randomOff ? null : <Button variant="outline" className="ml-auto" onClick={p[3]}><Icon name="dice" size={16} />{p[2]}</Button>}
    </footer>
  );
}

function SheetHandle() {
  const drag = useRef({ y: -1, h: 0, moved: false });
  const panel = () => document.getElementById('insp-left') as HTMLElement;
  return (
    <button type="button" className="osb-sheet-handle sheet-handle" aria-label="Resize the panel"
      onClick={() => { if (drag.current.moved) { drag.current.moved = false; return; } ui.set({ snap: ui.get().snap === 'full' ? 'half' : 'full' }); }}
      onPointerDown={(e) => { drag.current = { y: e.clientY, h: panel().getBoundingClientRect().height, moved: false }; (e.target as HTMLElement).setPointerCapture(e.pointerId); }}
      onPointerMove={(e) => { const d = drag.current; if (d.y < 0) return; panel().style.height = Math.max(120, d.h + d.y - e.clientY) + 'px'; }}
      onPointerUp={(e) => {
        const d = drag.current; if (d.y < 0) return;
        const hgt = d.h + d.y - e.clientY, moved = Math.abs(e.clientY - d.y) > 6;
        panel().style.height = ''; d.y = -1;
        if (!moved) return;
        d.moved = true;
        if (hgt < innerHeight * 0.22) actions.closePanel();
        else ui.set({ snap: hgt > innerHeight * 0.62 ? 'full' : 'half' });
      }}><span /></button>
  );
}

export function ModePanel() {
  const { legacy } = useShell();
  const mode = useUI((s) => s.mode), open = useUI((s) => s.open), sub = useUI((s) => s.sub), width = useUI((s) => s.width), snap = useUI((s) => s.snap);
  const hasClips = useUI((s) => s.hasClips);
  const M = MODES.find((m) => m.id === mode)!;
  const shown = (m: ModeId, id: string) => open && mode === m && sub[m] === id;
  return (
    <aside id="insp-left" className="osb-panel osb-glass insp mode-panel" data-mode={mode} data-snap={snap} aria-labelledby="mode-title" hidden={!open}>
      {width === 'phone' ? <SheetHandle /> : null}
      <header className="osb-panel-head">
        <h2 className="osb-title" id="mode-title">{M.label}</h2>
        <span className="osb-panel-note" />
        <Tip label="Close the panel" keys="Esc" side="right">
          <button type="button" className="osb-ib insp-x" aria-label="Close the panel (Esc)" onClick={() => actions.closePanel()}><Icon name="close" size={16} /></button>
        </Tip>
      </header>
      {mode === 'create' ? (
        <div className="osb-panel-tabs">
          <Tabs value={sub.create} onValueChange={(v) => actions.setSection('create', v)} activationMode="automatic">
            <TabsList aria-label="Create">
              {SECTIONS.create.map((s) => (
                <Tip key={s.id} label={s.label} keys={s.key}>
                  <TabsTrigger value={s.id} id={'tab-' + s.id} aria-controls={'sec-' + s.id}>
                    <Icon name={s.icon} />{s.short || s.label}{s.key ? <span className="osb-hint">{s.key.toUpperCase()}</span> : null}
                  </TabsTrigger>
                </Tip>
              ))}
            </TabsList>
          </Tabs>
        </div>
      ) : null}
      <div className="osb-panel-body insp-scroll">
        <BodySection shown={shown('create', 'body')} />
        <FaceSection shown={shown('create', 'face')} />
        <SkinSection shown={shown('create', 'skin')} />
        <HairSection shown={shown('create', 'hair')} />
        <DressSection shown={shown('dress', 'wear')} />
        <ExpressSection shown={shown('express', 'expression')} />
        {hasClips ? <PoseSection shown={shown('pose', 'pose')} />
          : (shown('pose', 'pose') ? <p className="osb-note sec" id="sec-pose">This page holds no poses.</p> : null)}
        <SheetSection shown={shown('sheet', 'sheet')} sheet={legacy.sheet} />
        <ExportSection shown={shown('export', 'export')} />
      </div>
      <Foot mode={mode} sub={sub[mode]} />
    </aside>
  );
}

export function ToolPanel() {
  const { legacy } = useShell();
  const tool = useUI((s) => s.tool);
  const T = tool === 'settings' ? { label: 'Settings' } : TOOLS.find((t) => t.id === tool);
  return (
    <aside id="insp-right" className="osb-panel osb-glass insp tool-panel" data-tool={tool || ''} aria-labelledby="tool-title" hidden={!tool}>
      <header className="osb-panel-head">
        <h2 className="osb-title" id="tool-title">{T?.label || ''}</h2>
        <span className="osb-panel-note">{tool === 'rig' ? 'The skeleton and the skin weights' : tool === 'scene' ? 'Lights, quality and frame rate' : tool === 'settings' ? 'AI providers' : ''}</span>
        <Tip label="Close the panel" keys="Esc" side="left">
          <button type="button" className="osb-ib insp-x" aria-label="Close the tool panel (Esc)" onClick={() => actions.closeTool()}><Icon name="close" size={16} /></button>
        </Tip>
      </header>
      <div className="osb-panel-body insp-scroll">
        <DomSlot el={legacy.sections.rig} id="rig" shown={tool === 'rig'} />
        <DomSlot el={legacy.sections.scene} id="scene" shown={tool === 'scene'} />
        <SettingsSection shown={tool === 'settings'} />
      </div>
    </aside>
  );
}

// the modes at medium widths: a rail at the left edge
export function Rail() {
  const mode = useUI((s) => s.mode), open = useUI((s) => s.open);
  return (
    <nav className="osb-rail osb-glass mode-rail" aria-label="Modes">
      {MODES.map((m) => (
        <Tip key={m.id} label={m.tip} keys={m.key} side="right">
          <button id={'tab-' + m.id} type="button" className="osb-rail-tab" aria-current={mode === m.id ? 'page' : undefined} aria-selected={mode === m.id}
            aria-expanded={mode === m.id ? open : undefined} aria-controls="insp-left" onClick={() => actions.toggleMode(m.id)}>
            <Icon name={m.icon} size={24} /><span className="osb-rail-label">{m.label}</span>
          </button>
        </Tip>
      ))}
    </nav>
  );
}

// the modes on a phone: six tabs in the bottom bar
export function PhoneBar() {
  const mode = useUI((s) => s.mode), open = useUI((s) => s.open);
  return (
    <nav className="osb-phonebar osb-glass phone-bar" aria-label="Modes">
      {MODES.map((m) => (
        <button key={m.id} id={'tab-' + m.id} type="button" className="osb-phonetab" aria-current={mode === m.id ? 'page' : undefined} aria-selected={mode === m.id}
          aria-expanded={mode === m.id ? open : undefined} aria-controls="insp-left" onClick={() => actions.toggleMode(m.id)}>
          <Icon name={m.icon} size={24} /><span>{m.label}</span>
        </button>
      ))}
    </nav>
  );
}

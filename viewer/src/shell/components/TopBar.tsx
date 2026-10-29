// OpenSculptBoy
// Apache License, Version 2.0
//
// The top bar (the design system's TopBar): the brand, the character, the six modes as a lintel over the canvas, and
// the actions: Open, Undo, Redo, the View menu and the help. On a phone the bar keeps the brand's mark, the character,
// Undo, Open and the View menu, and the modes move into the phone bar; at medium widths they move into the rail.

import { useRef } from 'react';
import { MODES, TOOLS } from '../../ui/shortcuts.ts';
import { useShell, settingsSlot } from '../context.ts';
import { useBus, useUI } from '../hooks.ts';
import { actions, type Theme } from '../state.ts';
import { Button } from './ui/button.tsx';
import { Icon } from './ui/icon.tsx';
import { Tip } from './ui/tooltip.tsx';
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuRadioGroup, DropdownMenuRadioItem,
  DropdownMenuSeparator, DropdownMenuTrigger,
} from './ui/dropdown-menu.tsx';

// the character's button: its stone dot (skin and hair), its name and its kind
function Identity({ phone }: { phone: boolean }) {
  const { legacy } = useShell();
  const dot = useRef<HTMLSpanElement>(null), name = useRef<HTMLSpanElement>(null), kind = useRef<HTMLElement>(null);
  const btn = useRef<HTMLButtonElement>(null);
  useBus(['look', 'wardrobe'], () => {
    const id = legacy.identity();
    if (dot.current) { dot.current.style.setProperty('--a', id.a); dot.current.style.setProperty('--b', id.b); }
    if (name.current) name.current.textContent = id.name;
    if (kind.current) { kind.current.textContent = id.kind; kind.current.hidden = !id.kind || phone; }
    btn.current?.setAttribute('aria-label', `Character: ${id.name}${id.kind ? ', ' + id.kind : ''}. Show the characters`);
  });
  return (
    <Tip label="Characters">
      <button ref={btn} type="button" className="osb-identity identity" onClick={() => actions.openMode('create', 'characters')}>
        <span ref={dot} className="osb-dot" /><span ref={name} className="osb-identity-name" /><small ref={kind} />
      </button>
    </Tip>
  );
}

export function ViewMenu() {
  const { legacy } = useShell();
  const tool = useUI((s) => s.tool), stats = useUI((s) => s.stats), theme = useUI((s) => s.theme);
  const full = typeof document !== 'undefined' && document.fullscreenEnabled;
  return (
    <DropdownMenu>
      <Tip label="View: rig, scene, performance, theme">
        <DropdownMenuTrigger asChild>
          <button type="button" className="osb-ib" aria-label="View menu"><Icon name="view" /></button>
        </DropdownMenuTrigger>
      </Tip>
      <DropdownMenuContent align="end" className="osb-view-menu">
        <DropdownMenuLabel>View</DropdownMenuLabel>
        {TOOLS.map((t) => (
          <DropdownMenuItem key={t.id} icon={t.icon} shortcut={t.key!.toUpperCase()} data-tool={t.id} aria-checked={tool === t.id}
            onSelect={() => actions.toggleTool(t.id as any)}>{t.label}</DropdownMenuItem>
        ))}
        <DropdownMenuItem icon="stats" shortcut="`" aria-checked={stats} onSelect={() => legacy.toggleStats()}>Performance</DropdownMenuItem>
        {full ? <DropdownMenuItem icon="fullscreen" onSelect={() => legacy.fullscreen()}>Full screen</DropdownMenuItem> : null}
        <DropdownMenuSeparator />
        <DropdownMenuLabel>Theme</DropdownMenuLabel>
        <DropdownMenuRadioGroup value={theme} onValueChange={(v) => actions.setTheme(v as Theme)}>
          <DropdownMenuRadioItem value="light" icon="sun">Light marble</DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="dark" icon="moon">Dark marble</DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="system" icon="system">Follow the system</DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
        {settingsSlot.open ? (<>
          <DropdownMenuSeparator />
          <DropdownMenuItem icon="settings" data-slot="settings" onSelect={() => settingsSlot.open?.()}>{settingsSlot.label}</DropdownMenuItem>
        </>) : null}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function TopBar() {
  const { legacy } = useShell();
  const mode = useUI((s) => s.mode), open = useUI((s) => s.open), width = useUI((s) => s.width), hist = useUI((s) => s.history);
  const phone = width === 'phone';
  return (
    <header className="osb-top topbar" aria-label="OpenSculptBoy">
      <span className="osb-brand" aria-label="OpenSculptBoy">{phone || width === 'medium' ? null : 'OpenSculptBoy'}</span>
      <Identity phone={phone} />
      {width === 'wide' ? (
        <nav className="osb-modes osb-glass" aria-label="Modes">
          {MODES.map((m) => (
            <Tip key={m.id} label={m.tip} keys={m.key}>
              <button id={'tab-' + m.id} type="button" className="osb-mode" aria-current={mode === m.id ? 'page' : undefined}
                data-state={mode === m.id ? 'active' : 'inactive'} aria-expanded={mode === m.id ? open : undefined} aria-controls="insp-left"
                onClick={() => actions.toggleMode(m.id)}>
                {m.label}<span className="osb-hint">{m.key.toUpperCase()}</span>
              </button>
            </Tip>
          ))}
        </nav>
      ) : <span className="osb-top-gap" />}
      <div className="osb-actions osb-glass">
        {phone ? (
          <>
            <Tip label="Undo" keys="Ctrl Z"><button type="button" className="osb-ib" aria-label="Undo (Ctrl Z)" disabled={!hist.undo} onClick={legacy.undo}><Icon name="undo" /></button></Tip>
            <Tip label="Open a character" keys="Ctrl O"><button type="button" className="osb-ib" aria-label="Open (Ctrl O)" onClick={() => actions.showOpen()}><Icon name="open" /></button></Tip>
          </>
        ) : (
          <>
            <Tip label="Open a character, wearables or sheet settings" keys="Ctrl O">
              <Button variant="outline" aria-label="Open (Ctrl O)" onClick={() => actions.showOpen()}><Icon name="open" size={16} />Open</Button>
            </Tip>
            <span className="osb-sep" />
            <Tip label={hist.undo ? 'Undo ' + hist.undo : 'Nothing to undo'} keys="Ctrl Z">
              <button type="button" className="osb-ib" aria-label="Undo (Ctrl Z)" disabled={!hist.undo} onClick={legacy.undo}><Icon name="undo" /></button>
            </Tip>
            <Tip label={hist.redo ? 'Redo ' + hist.redo : 'Nothing to redo'} keys="Ctrl Shift Z">
              <button type="button" className="osb-ib" aria-label="Redo (Ctrl Shift Z)" disabled={!hist.redo} onClick={legacy.redo}><Icon name="redo" /></button>
            </Tip>
          </>
        )}
        <ViewMenu />
        {phone ? null : (
          <Tip label="Help and keys" keys="?">
            <button type="button" className="osb-ib" aria-label="Help and keys (?)" onClick={legacy.showHelp}><Icon name="help" /></button>
          </Tip>
        )}
      </div>
    </header>
  );
}

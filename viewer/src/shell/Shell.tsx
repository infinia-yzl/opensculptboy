// OpenSculptBoy
// Apache License, Version 2.0
//
// The interface of the page in React (shadcn/ui on Radix, styled by the OpenSculptBoy Design System): the top bar,
// the rail or the phone bar, the mode's panel, the tool panel, the view bar, the welcome card and the Open dialog.
// It mounts beside the canvas in #ui-root; the 3D view stays the renderer of main.ts with its own frame loop, and
// the interface reaches it through the App of ui/app.ts and the bus of state.ts.

import { createRoot } from 'react-dom/client';
import { flushSync } from 'react-dom';
import { ShellCtx, type ShellDeps } from './context.ts';
import { useUI } from './hooks.ts';
import { startTheme } from './theme.ts';
import { actions, ui } from './state.ts';
import { TopBar } from './components/TopBar.tsx';
import { ModePanel, PhoneBar, Rail, ToolPanel } from './components/Panels.tsx';
import { OpenDialog, ViewBar, Welcome } from './components/Frame.tsx';
import { TooltipProvider } from './components/ui/tooltip.tsx';

function Shell() {
  const width = useUI((s) => s.width);
  return (
    <TooltipProvider>
      <TopBar />
      {width === 'medium' ? <Rail /> : null}
      <ModePanel />
      <ToolPanel />
      <ViewBar />
      <Welcome />
      <OpenDialog />
      {width === 'phone' ? <PhoneBar /> : null}
    </TooltipProvider>
  );
}

// mount the interface; it renders at once, so the page's ids (#insp-left, #sec-*, #tab-*) exist when this returns
export function mountShell(deps: ShellDeps) {
  startTheme();
  document.body.classList.add('osb-root');
  const host = document.createElement('div');
  host.id = 'ui-root';
  const parked = document.createElement('div');
  parked.id = 'osb-parked'; parked.hidden = true;
  document.body.append(host, parked);
  const root = createRoot(host);
  flushSync(() => root.render(<ShellCtx.Provider value={deps}><Shell /></ShellCtx.Provider>));
  // for scripts and the page tests: the interface's state and its actions (state.ts)
  (window as any).__shell = { ui, actions };
  return root;
}

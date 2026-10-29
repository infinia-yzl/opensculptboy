// OpenSculptBoy
// Apache License, Version 2.0
//
// The theme: white marble (light) or nero marquina (dark), chosen in the View menu or following the system setting.
// The root element carries data-theme, which the design system's tokens and components read (shell.html sets it
// before the page draws, from the same saved choice).

import { ui } from './state.ts';

export function startTheme() {
  const media = typeof matchMedia === 'function' ? matchMedia('(prefers-color-scheme: dark)') : null;
  const apply = () => {
    const s = ui.get();
    const dark = s.theme === 'dark' || (s.theme === 'system' && !!media?.matches);
    const root = document.documentElement;
    root.dataset.theme = dark ? 'dark' : 'light';
    root.classList.toggle('dark', dark);
    if (s.dark !== dark) ui.set({ dark });
  };
  media?.addEventListener('change', apply);
  ui.subscribe(apply);
  apply();
}

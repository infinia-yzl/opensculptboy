// OpenSculptBoy
// Apache License, Version 2.0
//
// React's view of the studio's registries: a component that shows keys, the status line or the settings renders
// again when they change.

import { useSyncExternalStore } from 'react';
import { actionsVersion, subscribeActions } from './actions.ts';
import { settings } from './settings.ts';
import { statusNow, subscribeStatus, type StatusLine } from './status.ts';

export function useActionsVersion(): number { return useSyncExternalStore(subscribeActions, actionsVersion, actionsVersion); }

let lastLine: StatusLine | null = null, lastKey = '';
function snapshot(): StatusLine {
  const s = statusNow();
  const key = JSON.stringify(s);
  if (key !== lastKey) { lastKey = key; lastLine = s; }
  return lastLine!;
}
export function useStatusLine(): StatusLine { return useSyncExternalStore(subscribeStatus, snapshot, snapshot); }

export function useSettings() { return useSyncExternalStore(settings.subscribe, settings.get, settings.get); }

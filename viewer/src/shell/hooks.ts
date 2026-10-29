// OpenSculptBoy
// Apache License, Version 2.0
//
// React's view of the interface state (state.ts): a selector over the store, and a bus listener that paints through
// refs without a render.

import { useEffect, useLayoutEffect, useRef, useSyncExternalStore } from 'react';
import { bus, ui, type BusEvent, type UIState } from './state.ts';

// a value of the store; the selector returns a value that stays the same object until it changes (a field of the
// store, or a primitive)
export function useUI<T>(select: (s: UIState) => T): T {
  return useSyncExternalStore(ui.subscribe, () => select(ui.get()));
}

// run fn on each event of the bus (and once at mount), with the latest fn; fn paints the DOM and sets no state
export function useBus(events: BusEvent | BusEvent[], fn: () => void) {
  const ref = useRef(fn);
  ref.current = fn;
  useLayoutEffect(() => {
    const offs = ([] as BusEvent[]).concat(events).map((e) => bus.on(e, () => ref.current()));
    ref.current();
    return () => offs.forEach((o) => o());
    // the events are fixed for a component
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}

export function useMounted(fn: () => void | (() => void)) {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(fn, []);
}

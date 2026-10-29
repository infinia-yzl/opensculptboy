// OpenSculptBoy
// Apache License, Version 2.0
//
// A section that is not yet rebuilt in React: its own DOM (ui/*_ui.ts), hosted in a panel of the new interface. The
// element keeps its id (sec-<id>), its role and its state; the slot shows it or hides it with the panel's choice.

import { useLayoutEffect, useRef } from 'react';

export function DomSlot({ el, id, shown, onShow }: { el: HTMLElement; id: string; shown: boolean; onShow?: () => void }) {
  const box = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    el.id = 'sec-' + id;
    el.classList.add('sec');
    el.setAttribute('role', 'tabpanel');
    el.setAttribute('aria-labelledby', 'tab-' + id);
    box.current!.append(el);
    // the element stays in the page while the panel is away, so its state and its listeners stay
    return () => { document.getElementById('osb-parked')?.append(el); };
  }, [el, id]);
  useLayoutEffect(() => {
    el.hidden = !shown;
    if (shown) onShow?.();
  }, [el, shown, onShow]);
  return <div ref={box} className="osb-slot-host" />;
}

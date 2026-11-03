// OpenSculptBoy
// Apache License, Version 2.0
//
// The status line under the view (artboard 3b): what the mouse does now, from studio/status.ts. A held Shift, Ctrl
// or Alt shows as a cap at the start of the line, and after half a second the overlay of that key's combinations
// opens at the left edge of the view (artboard 3c).

import type * as React from 'react';
import { capsOf, modifierActions } from './actions.ts';
import { useActionsVersion, useStatusLine } from './react.ts';
import type { StatusPart } from './status.ts';

const caps = (keys: string) => {
  // 'Ctrl+left-drag' and 'Drag a point': caps for the keys, words for the mouse
  const out: (string | React.ReactElement)[] = [];
  keys.split('+').forEach((k, i) => {
    if (i) out.push('+');
    out.push(/^(Shift|Ctrl|Alt|Esc|Tab|Enter|F\d+|[A-Z0-9`\[\]\/]|Home|Backspace|PgUp|PgDn|Space)$/.test(k) ? <kbd key={i} className="osb-key">{k}</kbd> : <b key={i}>{k}</b>);
  });
  return out;
};

function Part({ p }: { p: StatusPart }) {
  return <span className="status-part">{p.keys ? caps(p.keys) : null}{p.keys && p.text ? ': ' : null}{p.text}</span>;
}

export function StatusLine() {
  const s = useStatusLine();
  const heldName = s.held ? s.held[0].toUpperCase() + s.held.slice(1) : '';
  return (
    <>
      <div className="osb-glass status-line" role="status" aria-live="polite" aria-label="What the mouse does now" data-held={s.held || undefined}>
        {s.held ? <span className="status-held"><kbd className="osb-key">{heldName}</kbd> held</span> : null}
        {s.lead ? <span className="status-part status-lead">{s.lead}</span> : null}
        {s.parts.map((p, i) => <Part key={i} p={p} />)}
      </div>
      {s.heldLong ? <HeldOverlay mod={s.heldLong} /> : null}
    </>
  );
}

// the combinations of a held modifier, faint, near the left edge of the view
function HeldOverlay({ mod }: { mod: 'shift' | 'ctrl' | 'alt' }) {
  useActionsVersion();
  const name = mod === 'ctrl' ? 'Ctrl' : mod === 'alt' ? 'Alt' : 'Shift';
  const list = modifierActions(name);
  // one row per shared row of the help
  const seen = new Set<string>();
  const rows = list.filter(({ a }) => { const k = a.row || a.id; if (seen.has(k)) return false; seen.add(k); return true; });
  if (!rows.length) return null;
  return (
    <aside className="held-overlay" aria-label={name + ' combinations'}>
      <h2 className="osb-eyebrow">{name} held</h2>
      <dl>
        {rows.map(({ a, chord }) => (
          <div key={a.id} className="held-row">
            <dt>{capsOf(chord).map((c, i) => <kbd key={i} className="osb-key">{c}</kbd>)}</dt>
            <dd>{a.row ? (a.rowLabel || a.label) : a.label}</dd>
          </div>
        ))}
      </dl>
    </aside>
  );
}

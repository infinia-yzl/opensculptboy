// OpenSculptBoy
// Apache License, Version 2.0
//
// The status line under the view (artboard 3b): what the mouse does now, from studio/status.ts. A held Shift, Ctrl
// or Alt shows as a cap at the start of the line, and after half a second the overlay of that key's combinations
// opens at the left edge of the view (artboard 3c).

import { capsOf, modifierActions, rowLabelOf } from './actions.ts';
import { useActionsVersion, useStatusLine } from './react.ts';
import type { StatusPart } from './status.ts';

// the keys and the mouse of a part in bold, as the artboards write them ('Left-drag', 'F', 'Ctrl+Z')
const caps = (keys: string) => <b>{keys}</b>;

function Part({ p, lit }: { p: StatusPart; lit: boolean }) {
  return <span className={'status-part' + (lit ? ' status-lit' : '')}>{p.keys ? caps(p.keys) : null}{p.keys && p.text ? ': ' : null}{p.text}</span>;
}

export function StatusLine() {
  const s = useStatusLine();
  const heldName = s.held ? s.held[0].toUpperCase() + s.held.slice(1) : '';
  return (
    <>
      <div className="osb-glass status-line" aria-live="off" aria-label="What the mouse does now" data-held={s.held || undefined}>
        {s.held ? <span className="status-held"><kbd className="osb-key">{heldName}</kbd> held</span> : null}
        {s.lead ? <span className="status-part status-lead">{s.lead}</span> : null}
        {s.parts.map((p, i) => <Part key={i} p={p} lit={!!s.lit && p.keys === s.lit} />)}
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
            <dd>{rowLabelOf(a)}</dd>
          </div>
        ))}
      </dl>
    </aside>
  );
}

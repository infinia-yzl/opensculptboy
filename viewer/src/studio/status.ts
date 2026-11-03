// OpenSculptBoy
// Apache License, Version 2.0
//
// The status line under the view (DESIGN-0014, "Feedback"; artboard 3b): it says what the mouse does now. Each part of
// the page sets its own line under a source name (`setStatus('pose-edit', parts)`); the line shows the source of the
// highest priority, and the newest among equals. While Shift, Ctrl or Alt is held, the line shows that key's variant:
// a whole line given for it, else each part's own variant, and the held key as a cap at the start. Holding a modifier
// for half a second also opens the overlay of that key's combinations (artboard 3c), from the action registry.

export type Held = 'shift' | 'ctrl' | 'alt';

// a part of the line: key caps ('Left-drag', 'Ctrl Z', 'F') and words (': turn'); a part can carry its own variant
// for a held key, or null to leave it out while that key is held
export interface StatusPart {
  keys?: string;
  text?: string;
  shift?: StatusPart | null;
  ctrl?: StatusPart | null;
  alt?: StatusPart | null;
}
export interface StatusOpts {
  priority?: number;          // idle 0, hover 10, a tool 20, a drag 30 (the default 20)
  held?: Partial<Record<Held, StatusPart[]>>;   // whole lines for a held key
  lead?: string;              // a word before the parts ('Move tool')
}
export const PRIORITY = { idle: 0, hover: 10, tool: 20, drag: 30 } as const;

interface Entry { source: string; parts: StatusPart[]; opts: StatusOpts; t: number }
const entries = new Map<string, Entry>();
const listeners = new Set<() => void>();
let held: Held | null = null, heldLong: Held | null = null, heldTimer: any = 0, clock = 0;
const emit = () => { for (const l of listeners) l(); };

export function setStatus(source: string, parts: StatusPart[] | null, opts: StatusOpts = {}) {
  if (!parts) { clearStatus(source); return; }
  entries.set(source, { source, parts, opts, t: ++clock });
  emit();
}
export function clearStatus(source: string) { if (entries.delete(source)) emit(); }
export function subscribeStatus(fn: () => void): () => void { listeners.add(fn); return () => { listeners.delete(fn); }; }

export interface StatusLine { source: string | null; lead: string; parts: StatusPart[]; held: Held | null; heldLong: Held | null }
// the line as it shows now
export function statusNow(): StatusLine {
  let top: Entry | null = null;
  for (const e of entries.values()) {
    const p = e.opts.priority ?? 20, q = top ? (top.opts.priority ?? 20) : -1;
    if (!top || p > q || (p === q && e.t > top.t)) top = e;
  }
  if (!top) return { source: null, lead: '', parts: [], held, heldLong };
  let parts = top.parts;
  if (held && top.opts.held?.[held]) parts = top.opts.held[held]!;
  else if (held) parts = top.parts.map((p) => (held! in p ? p[held!] : p)).filter((p): p is StatusPart => !!p);
  return { source: top.source, lead: top.opts.lead || '', parts, held, heldLong };
}
export function heldKey(): Held | null { return held; }

// the held modifier: the line changes the moment the key goes down and goes back when it is released; after half a
// second the overlay of that key opens
function setHeld(h: Held | null) {
  if (h === held) return;
  held = h; clearTimeout(heldTimer);
  if (heldLong) heldLong = null;
  if (h) heldTimer = setTimeout(() => { if (held === h) { heldLong = h; emit(); } }, 500);
  emit();
}
let tracking = false;
export function trackModifiers(target: Window = window) {
  if (tracking) return;
  tracking = true;
  const of = (e: KeyboardEvent): Held | null => {
    const n = (e.ctrlKey || e.metaKey ? 1 : 0) + (e.altKey ? 1 : 0) + (e.shiftKey ? 1 : 0);
    if (n !== 1) return null;
    return e.ctrlKey || e.metaKey ? 'ctrl' : e.altKey ? 'alt' : 'shift';
  };
  target.addEventListener('keydown', (e) => {
    // a modifier alone; another key with it is a chord and ends the overlay
    if (['Shift', 'Control', 'Alt', 'Meta'].includes(e.key)) setHeld(of(e));
    else if (heldLong) { heldLong = null; clearTimeout(heldTimer); emit(); }
  }, true);
  target.addEventListener('keyup', (e) => { if (['Shift', 'Control', 'Alt', 'Meta'].includes(e.key)) setHeld(of(e)); }, true);
  target.addEventListener('blur', () => setHeld(null));
  // a pointer event knows the modifiers too: a key released outside the page
  target.addEventListener('pointermove', (e) => {
    const n = (e.ctrlKey || e.metaKey ? 1 : 0) + (e.altKey ? 1 : 0) + (e.shiftKey ? 1 : 0);
    const h: Held | null = n !== 1 ? null : e.ctrlKey || e.metaKey ? 'ctrl' : e.altKey ? 'alt' : 'shift';
    if (h !== held) setHeld(h);
  }, { passive: true });
}

// for a test: the line as plain text
export function statusText(): string {
  const s = statusNow();
  const lead = s.held ? s.held[0].toUpperCase() + s.held.slice(1) + ' held · ' : '';
  return lead + (s.lead ? s.lead + ' · ' : '') + s.parts.map((p) => (p.keys ? p.keys : '') + (p.text ? (p.keys ? ': ' : '') + p.text : '')).join(' · ');
}

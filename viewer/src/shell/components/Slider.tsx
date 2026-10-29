// OpenSculptBoy
// Apache License, Version 2.0
//
// The design system's Slider in React: a label, the value field, a reset that shows once the value moved, a fine
// groove with the gold fill from the default to the value, a mark at the default and the ends' meanings. The track
// is a native range input, as the design system's own preview draws it, so the arrow keys, Home and End work as in any
// range input and the tests' selectors stay. React renders the slider once: the value follows the look through the
// bus (useBus) and the slider paints itself through refs, so a drag renders nothing in React. Dragging changes the
// figure on every frame; Undo takes a whole drag as one step (onStart and onEnd).

import { useLayoutEffect, useRef } from 'react';
import { clamp } from '../../ui/dom.ts';
import { fmt, THUMB } from '../../ui/controls.ts';
import { useBus } from '../hooks.ts';
import type { BusEvent } from '../state.ts';
import { Icon } from './ui/icon.tsx';

export interface SliderProps {
  id: string;
  label: string;
  min: number;
  max: number;
  step?: number;
  def: number;
  digits?: number;
  ends?: string[];
  title?: string;
  // a readable form of the value (per cent, degrees), shown next to the number
  readout?: (v: number) => string;
  // a colour track: the track shows the colours it picks (paintTrack sets them), with no fill and no default mark
  colour?: boolean;
  paintTrack?: (input: HTMLInputElement) => void;
  // the value shown, read again on each event of `events`
  get: () => number;
  events?: BusEvent[];
  onInput: (v: number) => void;
  onStart?: () => void;
  onEnd?: () => void;
}

export function Slider(p: SliderProps) {
  const digits = p.digits ?? 2;
  const root = useRef<HTMLDivElement>(null), input = useRef<HTMLInputElement>(null), num = useRef<HTMLInputElement>(null);
  const read = useRef<HTMLSpanElement>(null), mark = useRef<HTMLSpanElement>(null), bubble = useRef<HTMLSpanElement>(null);
  const st = useRef({ editing: false, cur: p.def });
  const P = useRef(p);
  P.current = p;

  const pct = (v: number) => { const o = P.current; return o.max > o.min ? (v - o.min) / (o.max - o.min) * 100 : 0; };
  const paint = (v: number) => {
    const o = P.current, el = root.current, inp = input.current;
    if (!el || !inp) return;
    const q = pct(v), d = pct(clamp(o.def, o.min, o.max));
    inp.style.setProperty('--lo', Math.min(q, d) + '%');
    inp.style.setProperty('--hi', Math.max(q, d) + '%');
    if (mark.current) mark.current.style.left = `calc(${d}% + ${(0.5 - d / 100) * THUMB}px)`;
    if (bubble.current) bubble.current.style.left = `calc(${q}% + ${(0.5 - q / 100) * THUMB}px)`;
    const text = fmt(v, digits), rd = o.readout ? o.readout(v) : '';
    if (num.current && document.activeElement !== num.current) num.current.value = text;
    if (bubble.current) bubble.current.textContent = rd || text;
    if (read.current) read.current.textContent = rd;
    inp.setAttribute('aria-valuetext', rd ? `${text}, ${rd}` : text);
    el.toggleAttribute('data-changed', Math.abs(v - o.def) > 1e-6);
    o.paintTrack?.(inp);
  };
  const set = (v: number) => {
    st.current.cur = v;
    if (input.current && document.activeElement !== input.current) input.current.value = String(v);
    paint(v);
  };
  useBus(p.events || ['look'], () => set(P.current.get()));

  useLayoutEffect(() => {
    const inp = input.current!, n = num.current!, el = root.current!;
    const start = () => { if (!st.current.editing) { st.current.editing = true; P.current.onStart?.(); } };
    const end = () => { if (st.current.editing) { st.current.editing = false; P.current.onEnd?.(); } };
    const apply = (v: number) => { st.current.cur = v; paint(v); P.current.onInput(v); };
    const onInput = () => { start(); apply(parseFloat(inp.value)); };
    const down = () => el.classList.add('dragging');
    const up = () => el.classList.remove('dragging');
    const reset = () => { start(); inp.value = String(P.current.def); apply(P.current.def); end(); };
    const commitNum = () => {
      const v = parseFloat(n.value.replace(',', '.').replace(/[^\d.+-]/g, ''));
      if (!isFinite(v)) { n.value = fmt(st.current.cur, digits); return; }
      const c = clamp(v, P.current.min, P.current.max);
      start(); inp.value = String(c); apply(c); end();
    };
    const numKey = (e: KeyboardEvent) => {
      if (e.key === 'Enter') { e.preventDefault(); commitNum(); n.select(); }
      else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); n.value = fmt(st.current.cur, digits); n.blur(); }
      else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
        e.preventDefault();
        const s = (P.current.step ?? 0.01) * (e.shiftKey ? 10 : 1) * (e.key === 'ArrowUp' ? 1 : -1);
        n.value = fmt(clamp(st.current.cur + s, P.current.min, P.current.max), digits); commitNum();
      }
    };
    const resetB = el.querySelector('.osb-sl-reset') as HTMLButtonElement;
    const onReset = () => { reset(); inp.focus({ preventScroll: true }); };
    const sel = () => n.select();
    inp.addEventListener('input', onInput); inp.addEventListener('change', end);
    inp.addEventListener('pointerdown', down); inp.addEventListener('pointerup', up); inp.addEventListener('pointercancel', up); inp.addEventListener('blur', up);
    inp.addEventListener('dblclick', reset); resetB.addEventListener('click', onReset);
    n.addEventListener('keydown', numKey); n.addEventListener('blur', commitNum); n.addEventListener('focus', sel);
    return () => {
      inp.removeEventListener('input', onInput); inp.removeEventListener('change', end);
      inp.removeEventListener('pointerdown', down); inp.removeEventListener('pointerup', up); inp.removeEventListener('pointercancel', up); inp.removeEventListener('blur', up);
      inp.removeEventListener('dblclick', reset); resetB.removeEventListener('click', onReset);
      n.removeEventListener('keydown', numKey); n.removeEventListener('blur', commitNum); n.removeEventListener('focus', sel);
    };
    // the listeners read the latest props through P
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div ref={root} className={'sl osb-sl' + (p.colour ? ' sl-custom osb-sl--colour' : '')}>
      <div className="osb-sl-head">
        <label htmlFor={p.id} title={p.title || ''}>{p.label}</label>
        <span ref={read} className="osb-sl-read" />
        <button type="button" className="osb-sl-reset" aria-label={'Reset ' + p.label} title={'Reset to ' + fmt(p.def, digits)}><Icon name="undoSmall" size={14} /></button>
        <input ref={num} className="osb-sl-num" type="text" inputMode="decimal" autoComplete="off" spellCheck={false} aria-label={p.label + ' value'} defaultValue={fmt(p.def, digits)} />
      </div>
      <div className="osb-sl-track">
        <input ref={input} id={p.id} type="range" min={p.min} max={p.max} step={p.step ?? 0.01} defaultValue={p.def} />
        {p.colour ? null : <span ref={mark} className="osb-sl-def" aria-hidden="true" />}
        <span ref={bubble} className="osb-sl-bubble" aria-hidden="true" />
      </div>
      {p.ends && p.ends.some(Boolean) ? <div className="osb-sl-ends" aria-hidden="true">{p.ends.map((t, i) => <span key={i}>{t}</span>)}</div> : null}
    </div>
  );
}

// OpenSculptBoy
// Apache License, Version 2.0
//
// The small controls of the sections in React, with the design system's classes and the classes that the page tests
// and the legacy rules know (chip, seg, switch, fold): a chip that is pressed or not, a segmented control whose
// arrow keys move between its buttons, an on/off switch, a fold with a title, a count and actions, a search field, a
// row of a label and a control, and a section head with a live note.

import { useRef, type ReactNode } from 'react';
import { cn } from '../lib/utils.ts';
import { Icon } from './ui/icon.tsx';

export function Chip({ pressed, onClick, children, dot, className, ...rest }: { pressed: boolean; onClick: () => void; children: ReactNode; dot?: string; className?: string; [k: string]: any }) {
  return (
    <button type="button" className={cn('chip osb-chip', className)} aria-pressed={pressed} onClick={onClick} {...rest}>
      {dot ? <i style={{ background: dot }} /> : null}{children}
    </button>
  );
}

export interface SegItem { value: string; label: string; short?: string; tip?: string; icon?: string }
export function Seg({ label, items, value, onPick, fill = false, className }: { label: string; items: SegItem[]; value: string; onPick: (v: string) => void; fill?: boolean; className?: string }) {
  const box = useRef<HTMLDivElement>(null);
  return (
    <div ref={box} className={cn('seg osb-seg', fill && 'seg-fill osb-seg--fill', className)} role="group" aria-label={label}
      onKeyDown={(e) => {
        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
        const bs = Array.from(box.current!.querySelectorAll('button')), i = bs.indexOf(document.activeElement as any);
        if (i < 0) return;
        e.preventDefault(); e.stopPropagation();
        const j = (i + (e.key === 'ArrowRight' ? 1 : bs.length - 1)) % bs.length;
        bs[j].focus(); bs[j].click();
      }}>
      {items.map((it) => (
        <button key={it.value} type="button" aria-pressed={it.value === value} data-value={it.value} title={it.tip} aria-label={it.short ? it.label : undefined} onClick={() => onPick(it.value)}>
          {it.icon ? <Icon name={it.icon} size={18} /> : null}<span aria-hidden={it.short ? true : undefined}>{it.short || it.label}</span>
        </button>
      ))}
    </div>
  );
}

export function Switch({ label, on, onChange, tip, className, ...rest }: { label: string; on: boolean; onChange: (on: boolean) => void; tip?: string; className?: string; [k: string]: any }) {
  return (
    <button type="button" role="switch" aria-checked={on} className={cn('switch osb-switch', className)} title={tip} onClick={() => onChange(!on)} {...rest}>
      <span className="switch-track osb-switch-track" aria-hidden="true"><span className="switch-knob osb-switch-knob" /></span>
      <span className="switch-label">{label}</span>
    </button>
  );
}

export function Fold({ title, count, actions, children, open, onToggle, className, changed, ...rest }: {
  title: string; count?: ReactNode; actions?: ReactNode; children: ReactNode; open?: boolean; onToggle?: (open: boolean) => void; className?: string; changed?: boolean; [k: string]: any;
}) {
  return (
    <details className={cn('fold osb-fold', className)} open={open} data-changed={changed ? '' : undefined}
      onToggle={onToggle ? (e) => onToggle((e.target as HTMLDetailsElement).open) : undefined} {...rest}>
      <summary>
        <span className="osb-i"><Icon name="chevron" size={14} /></span>
        <span className="fold-title osb-fold-title">{title}</span>
        <span className="fold-n osb-fold-n">{count}</span>
        {actions ? <span className="osb-fold-actions" onClick={(e) => { e.preventDefault(); e.stopPropagation(); }}>{actions}</span> : null}
      </summary>
      <div className="fold-body osb-fold-body">{children}</div>
    </details>
  );
}

export function Search({ placeholder, value, onChange }: { placeholder: string; value: string; onChange: (q: string) => void }) {
  return (
    <label className="search osb-search">
      <Icon name="search" size={16} />
      <input className="search-input" type="search" placeholder={placeholder} aria-label={placeholder} autoComplete="off" spellCheck={false} value={value}
        onChange={(e) => onChange(e.target.value)} onKeyDown={(e) => { if (e.key === 'Escape' && value) { e.stopPropagation(); onChange(''); } }} />
    </label>
  );
}

export function Row({ label, children }: { label: string; children: ReactNode }) {
  return <div className="row osb-row-labelled"><span className="row-label">{label}</span>{children}</div>;
}

// the head of a section: its eyebrow title, a live note and actions
export function SectionHead({ title, note, children }: { title: string; note?: ReactNode; children?: ReactNode }) {
  return (
    <header className="sec-head">
      <h2 className="osb-eyebrow">{title}</h2>
      <span className="sec-note osb-panel-note">{note}</span>
      <div className="sec-actions">{children}</div>
    </header>
  );
}

export const Sub = ({ children }: { children: ReactNode }) => <h3 className="sub osb-eyebrow">{children}</h3>;
export const Note = ({ children, className, ...rest }: { children: ReactNode; className?: string; [k: string]: any }) => <p className={cn('note osb-note', className)} {...rest}>{children}</p>;

// a button of text with an icon, as the legacy sections drew them (class tb)
export function TextButton({ label, icon, onClick, quiet, primary, disabled, className, ...rest }: { label: string; icon?: string; onClick: () => void; quiet?: boolean; primary?: boolean; disabled?: boolean; className?: string; [k: string]: any }) {
  return (
    <button type="button" className={cn('tb osb-btn', quiet && 'tb-quiet osb-btn--quiet', primary && 'cs-primary osb-btn--primary', className)} disabled={disabled} onClick={onClick} {...rest}>
      {icon ? <Icon name={icon} size={16} /> : null}<span>{label}</span>
    </button>
  );
}

// OpenSculptBoy
// Apache License, Version 2.0
//
// The ring menus round the pointer (the view menu on the backtick, the display menu on Z; artboards 3d and 3e, after
// Blender's pies) and the command list (F3 or Ctrl+K; artboard 3f). Both read the action registry: a ring menu shows
// the live actions of its `menu`, clockwise from the top, and the command list every action by name, with its key and
// the place it lives.
//
// A ring menu opens round the pointer; a move toward a choice lights it, and a click goes there. Holding the key,
// moving toward a choice and letting go picks it as well. Esc or a click in the middle closes it.

import { h } from '../ui/dom.ts';
import { icon as iconSvg } from '../ui/icons.ts';
import { capsOf, commands, menuItems, runAction, type ActionDef } from './actions.ts';
import { clearStatus, PRIORITY, setStatus } from './status.ts';

const caps = (chord: string) => capsOf(chord).map((c) => h('kbd.osb-key', { text: c }));

// ------------------------------------------------------------------ the ring menu
export interface RingMenu { open(menu: string, title: string, key: string, at?: { x: number; y: number }, heldKey?: string): void; close(): void; isOpen(): boolean; el: HTMLElement }

export function ringMenu(): RingMenu {
  const root = h('div.ring-menu', { role: 'menu', hidden: true, 'aria-label': 'Menu' });
  const svgNS = 'http://www.w3.org/2000/svg';
  const line = document.createElementNS(svgNS, 'svg');
  line.setAttribute('class', 'ring-line'); line.setAttribute('aria-hidden', 'true');
  const seg = document.createElementNS(svgNS, 'line'); line.append(seg);
  const hub = h('button.ring-hub', { type: 'button', 'aria-label': 'Close the menu' });
  root.append(line, hub);
  let items: { a: ActionDef; el: HTMLElement; ang: number }[] = [];
  let cx = 0, cy = 0, lit = -1, openedAt = 0, held = '', moved = false;
  const R = 230;
  const light = (i: number) => {
    if (i === lit) return;
    lit = i;
    items.forEach((it, k) => it.el.classList.toggle('lit', k === i));
  };
  function pick(i: number) {
    const it = items[i];
    close();
    if (it) runAction(it.a.id);
  }
  function close() {
    if (root.hidden) return;
    root.hidden = true; items = []; lit = -1;
    clearStatus('ring');
    document.removeEventListener('pointermove', onMove, true);
    document.removeEventListener('pointerdown', onDown, true);
    document.removeEventListener('keydown', onKey, true);
    document.removeEventListener('keyup', onKeyUp, true);
  }
  function onMove(e: PointerEvent) {
    const dx = e.clientX - cx, dy = e.clientY - cy, r = Math.hypot(dx, dy);
    seg.setAttribute('x2', String(e.clientX)); seg.setAttribute('y2', String(e.clientY));
    if (r < 36) { light(-1); return; }
    moved = true;
    // the choice whose direction is nearest the pointer's
    const ang = Math.atan2(dx, -dy);
    let best = -1, bd = Infinity;
    items.forEach((it, k) => { const d = Math.abs(Math.atan2(Math.sin(ang - it.ang), Math.cos(ang - it.ang))); if (d < bd) { bd = d; best = k; } });
    light(best);
  }
  function onDown(e: PointerEvent) {
    e.preventDefault(); e.stopPropagation();
    const t = (e.target as HTMLElement).closest('.ring-item') as HTMLElement | null;
    if (t) { pick(Number(t.dataset.i)); return; }
    if (e.target === hub || Math.hypot(e.clientX - cx, e.clientY - cy) < 36) { close(); return; }
    if (lit >= 0) pick(lit); else close();
  }
  function onKey(e: KeyboardEvent) {
    e.stopPropagation();
    if (e.key === 'Escape') { e.preventDefault(); close(); return; }
    if (e.key === 'Enter') { e.preventDefault(); if (lit >= 0) pick(lit); return; }
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === 'Tab') { e.preventDefault(); light((lit + 1 + items.length) % items.length); return; }
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); light((lit - 1 + items.length) % items.length); return; }
    if (e.repeat) { e.preventDefault(); return; }
    // the menu's own key again closes it
    if (e.code === held) { e.preventDefault(); close(); }
  }
  // a held key released after a move: the lit choice is picked, as in Blender's pies
  function onKeyUp(e: KeyboardEvent) {
    if (e.code !== held) return;
    if (moved && lit >= 0 && performance.now() - openedAt > 250) pick(lit);
  }
  return {
    el: root,
    isOpen: () => !root.hidden,
    close,
    open(menu, title, key, at, heldKey = '') {
      close();
      const list = menuItems(menu);
      if (!list.length) return;
      const vw = innerWidth, vh = innerHeight;
      cx = Math.min(Math.max(at?.x ?? vw / 2, R + 100), vw - R - 100);
      cy = Math.min(Math.max(at?.y ?? vh / 2, R + 40), vh - R - 40);
      hub.replaceChildren(...caps(key), h('span', { text: title }));
      hub.style.left = cx + 'px'; hub.style.top = cy + 'px';
      seg.setAttribute('x1', String(cx)); seg.setAttribute('y1', String(cy)); seg.setAttribute('x2', String(cx)); seg.setAttribute('y2', String(cy));
      root.querySelectorAll('.ring-item').forEach((n) => n.remove());
      const n = list.length;
      items = list.map((a, i) => {
        const ang = (i / n) * Math.PI * 2;
        const on = a.checked ? a.checked() : null;
        const el = h('button.ring-item', { type: 'button', role: on === null ? 'menuitem' : 'menuitemcheckbox', 'data-i': String(i), 'data-action': a.id,
          ...(on === null ? {} : { 'aria-checked': String(on) }) },
          on ? h('span.ring-tick', { 'aria-hidden': 'true' }, iconSvg('check', 14)) : '',
          a.icon ? iconSvg(a.icon, 16) : '',
          h('span', { text: a.menuLabel || a.label }), ...(a.keys?.length ? caps(a.keys[0]) : []));
        // an ellipse a little wider than tall, as the artboards draw it
        el.style.left = (cx + Math.sin(ang) * R * 1.0) + 'px';
        el.style.top = (cy - Math.cos(ang) * R * 1.0) + 'px';
        root.append(el);
        return { a, el, ang };
      });
      lit = -1; moved = false; held = heldKey; openedAt = performance.now();
      root.hidden = false;
      setStatus('ring', [{ text: title + ' menu' }, { keys: 'Move', text: 'towards a choice: pick it' }, { keys: 'Click', text: 'go there' }, { keys: 'Esc', text: 'close' }],
        { priority: PRIORITY.drag + 5 });
      document.addEventListener('pointermove', onMove, true);
      document.addEventListener('pointerdown', onDown, true);
      document.addEventListener('keydown', onKey, true);
      document.addEventListener('keyup', onKeyUp, true);
    },
  };
}

// ------------------------------------------------------------------ the command list
export interface CommandList { el: HTMLDialogElement; open(): void }

export function commandList(): CommandList {
  const input = h('input.cmd-input', { type: 'search', placeholder: 'Type an action', 'aria-label': 'Search the actions', autocomplete: 'off', spellcheck: 'false',
    role: 'combobox', 'aria-expanded': 'true', 'aria-controls': 'cmd-list' }) as HTMLInputElement;
  const list = h('ul.cmd-list', { id: 'cmd-list', role: 'listbox', 'aria-label': 'Actions' });
  const count = h('span.cmd-count');
  const dlg = h('dialog.cmd', { 'aria-label': 'Command list' },
    h('div.cmd-search', {}, iconSvg('search', 18), input, h('kbd.osb-key', { text: 'F3' })),
    list,
    h('footer.cmd-foot', {}, h('kbd.osb-key', { text: '↑' }), h('kbd.osb-key', { text: '↓' }), ' choose ', h('kbd.osb-key', { text: 'Enter' }), ' run ',
      h('kbd.osb-key', { text: 'Esc' }), ' close ', count),
  ) as HTMLDialogElement;
  let rows: ActionDef[] = [], sel = 0;
  const mark = (label: string, q: string) => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!words.length) return [label];
    const re = new RegExp('(' + words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')', 'ig');
    return label.split(re).map((p, i) => (i % 2 ? h('b', { text: p }) : p));
  };
  function render() {
    const q = input.value;
    const found = commands(q);
    rows = found.map((f) => f.a);
    sel = Math.min(sel, Math.max(0, rows.length - 1));
    list.replaceChildren(...found.slice(0, 60).map(({ a, live }, i) => {
      const li = h('li.cmd-row', { role: 'option', id: 'cmd-' + i, 'aria-selected': String(i === sel), 'data-action': a.id, 'aria-disabled': String(!live) },
        h('span.cmd-icon', {}, iconSvg(a.icon || 'focus', 18)),
        h('span.cmd-text', {}, h('span.cmd-label', {}, ...mark(a.label, q)), h('span.cmd-place', { text: a.place || '' })),
        h('span.cmd-keys', {}, ...(a.keys?.length ? caps(a.keys[0]) : [h('span.cmd-nokey', { text: 'No key' })])));
      li.addEventListener('pointerdown', (e) => { e.preventDefault(); run(i); });
      return li;
    }));
    input.setAttribute('aria-activedescendant', rows.length ? 'cmd-' + sel : '');
    count.textContent = rows.length === 1 ? '1 match' : `${rows.length} matches`;
  }
  function run(i: number) {
    const a = rows[i];
    if (!a) return;
    dlg.close();
    // after the dialog has gone, so the action sees the page as the person does
    requestAnimationFrame(() => runAction(a.id));
  }
  input.addEventListener('input', () => { sel = 0; render(); });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); sel = Math.min(rows.length - 1, sel + 1); render(); list.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' }); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); sel = Math.max(0, sel - 1); render(); list.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' }); }
    else if (e.key === 'Enter') { e.preventDefault(); run(sel); }
  });
  dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });
  return {
    el: dlg,
    open() {
      input.value = ''; sel = 0; render();
      if (!dlg.open) dlg.showModal();
      input.focus();
    },
  };
}

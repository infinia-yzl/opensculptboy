// OpenSculptBoy
// Apache License, Version 2.0
//
// The ring menus round the pointer (the view menu on the backtick, the display menu on Z; artboards 3d and 3e, after
// Blender's pies) and the command list (F3 or Ctrl+K; artboard 3f). Both read the action registry: a ring menu shows
// the live actions of its `menu`, clockwise from the top, and the command list every action by name, with its key and
// the place it lives.
//
// A ring menu opens round the pointer; a move toward a choice lights it, and a click picks it (a switch flips and the
// menu stays open). Holding the key, moving toward a choice and letting go picks it as well. The menu's key again,
// Esc or a click in the middle closes it.

import { h } from '../ui/dom.ts';
import { icon as iconSvg } from '../ui/icons.ts';
import { capsOf, chordsOf, commands, menuItems, runAction, type ActionDef } from './actions.ts';
import { clearStatus, PRIORITY, setStatus, type StatusPart } from './status.ts';

const caps = (chord: string) => capsOf(chord).map((c) => h('kbd.osb-key', { text: c }));

// ------------------------------------------------------------------ the ring menu
// As in Blender's pie menus, each choice keeps a fixed direction: a menu has eight places clockwise from the top (more
// when its choices number more), and a choice's `menuOrder` is its place, so a choice that is missing in one state
// leaves its place empty and the others stay where the hand expects them. The lit choice is the one whose direction
// is nearest the pointer's, once the pointer is 12 pixels from the hub (Blender's pie threshold). A switch (an
// action with `checked`) flips on a click, on Enter or Space, or on its own key, and the menu stays open with the
// ticks updated, so several switches can flip in one opening; any other choice runs and closes the menu. Holding the
// menu's key, moving towards a choice and letting go picks it and closes the menu. A choice that does not apply now
// shows greyed out with its reason. The menu keeps inside the window: near an edge it moves inwards, and a small
// window draws it smaller.
export interface RingMenu { open(menu: string, title: string, key: string, at?: { x: number; y: number }, heldKey?: string): void; close(): void; isOpen(): boolean; el: HTMLElement }

// the places of a menu: eight, or one for each order when the menu has more
export function ringSlots(orders: number[]): number {
  return Math.max(8, ...orders.map((o) => o + 1));
}
// the direction of a place, in radians clockwise from the top
export function slotAngle(order: number, slots: number): number { return (order / slots) * Math.PI * 2; }
// the item nearest a direction (radians clockwise from the top), of items at the given angles
export function nearestItem(ang: number, angles: number[]): number {
  let best = -1, bd = Infinity;
  angles.forEach((a, k) => { const d = Math.abs(Math.atan2(Math.sin(ang - a), Math.cos(ang - a))); if (d < bd) { bd = d; best = k; } });
  return best;
}
// the shift that keeps a box inside the window with a margin (each axis alone; a box wider than the window is centred)
export function keepInside(box: { left: number; top: number; right: number; bottom: number }, vw: number, vh: number, margin = 8): { dx: number; dy: number } {
  const fit = (lo: number, hi: number, size: number) => (hi - lo > size - 2 * margin ? (size - lo - hi) / 2 : lo < margin ? margin - lo : hi > size - margin ? size - margin - hi : 0);
  return { dx: fit(box.left, box.right, vw), dy: fit(box.top, box.bottom, vh) };
}

const THRESHOLD = 12;

export function ringMenu(): RingMenu {
  const root = h('div.ring-menu', { role: 'menu', hidden: true, 'aria-label': 'Menu' });
  const svgNS = 'http://www.w3.org/2000/svg';
  const line = document.createElementNS(svgNS, 'svg');
  line.setAttribute('class', 'ring-line'); line.setAttribute('aria-hidden', 'true');
  const seg = document.createElementNS(svgNS, 'line'); line.append(seg);
  const hub = h('button.ring-hub', { type: 'button', 'aria-label': 'Close the menu' });
  root.append(line, hub);
  interface Item { a: ActionDef; el: HTMLElement; ang: number; dx: number; dy: number; check: HTMLElement | null; why: HTMLElement | null; label: HTMLElement; was?: string }
  let items: Item[] = [];
  let cx = 0, cy = 0, lit = -1, openedAt = 0, held = '', moved = false, title = '', keyCap = '', raf = 0;
  let before: Element | null = null;
  const reasonOf = (it: Item) => it.a.unavailable?.() || '';
  const place = () => {
    hub.style.left = cx + 'px'; hub.style.top = cy + 'px';
    for (const it of items) { it.el.style.left = (cx + it.dx) + 'px'; it.el.style.top = (cy + it.dy) + 'px'; }
    seg.setAttribute('x1', String(cx)); seg.setAttribute('y1', String(cy));
  };
  // the ticks and the greyed choices as they are now
  const refresh = () => {
    for (const it of items) {
      const on = it.a.checked ? it.a.checked() : null, why = reasonOf(it), key = on + '|' + why;
      if (key === it.was) continue;
      it.was = key;
      if (on !== null) it.el.setAttribute('aria-checked', String(on));
      it.el.classList.toggle('on', !!on);
      it.el.setAttribute('aria-disabled', String(!!why));
      it.el.classList.toggle('off', !!why);
      if (it.why) { it.why.textContent = why; it.why.hidden = !why; }
      it.el.title = why;
    }
  };
  const status = () => {
    const it = items[lit], why = it ? reasonOf(it) : '';
    const parts: StatusPart[] = [{ text: title + ' menu' }];
    if (why) parts.push({ text: (it.a.menuLabel || it.a.label) + ': ' + why });
    const switches = items.some((x) => !!x.a.checked);
    parts.push({ keys: 'Move', text: 'light a choice' }, { keys: 'Click', text: switches ? 'pick it; a switch flips and the menu stays' : 'go there' },
      { keys: keyCap + ' or Esc', text: 'close' });
    setStatus('ring', parts, { priority: PRIORITY.drag + 5 });
  };
  const light = (i: number, focus = false) => {
    if (i !== lit) {
      lit = i;
      items.forEach((it, k) => it.el.classList.toggle('lit', k === i));
      status();
    }
    if (focus && items[i]) items[i].el.focus({ preventScroll: true });
  };
  // a choice: a switch flips and the menu stays open; any other choice closes the menu and runs
  function activate(i: number, closeAfter = false) {
    const it = items[i];
    if (!it) { close(); return; }
    if (reasonOf(it)) { if (closeAfter) close(); else { light(i); status(); } return; }
    if (it.a.checked && !closeAfter) { runAction(it.a.id); refresh(); return; }
    close();
    runAction(it.a.id);
  }
  function close() {
    if (root.hidden) return;
    root.hidden = true; items = []; lit = -1;
    cancelAnimationFrame(raf);
    clearStatus('ring');
    document.removeEventListener('pointermove', onMove, true);
    document.removeEventListener('pointerdown', onDown, true);
    document.removeEventListener('keydown', onKey, true);
    document.removeEventListener('keyup', onKeyUp, true);
    // the focus back where it was (the canvas), when the keys had moved it onto a choice
    if (root.contains(document.activeElement) && before instanceof HTMLElement) before.focus({ preventScroll: true });
  }
  function onMove(e: PointerEvent) {
    const dx = e.clientX - cx, dy = e.clientY - cy, r = Math.hypot(dx, dy);
    seg.setAttribute('x2', String(e.clientX)); seg.setAttribute('y2', String(e.clientY));
    if (r < THRESHOLD) { light(-1); return; }
    moved = true;
    light(nearestItem(Math.atan2(dx, -dy), items.map((it) => it.ang)));
  }
  function onDown(e: PointerEvent) {
    e.preventDefault(); e.stopPropagation();
    if (e.button !== 0) { close(); return; }
    const t = (e.target as HTMLElement).closest('.ring-item') as HTMLElement | null;
    if (t) { activate(Number(t.dataset.i)); return; }
    if (e.target === hub || hub.contains(e.target as Node) || Math.hypot(e.clientX - cx, e.clientY - cy) < 44) { close(); return; }
    if (lit >= 0) activate(lit); else close();
  }
  function onKey(e: KeyboardEvent) {
    e.stopPropagation();
    const n = items.length;
    if (e.key === 'Escape') { e.preventDefault(); close(); return; }
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); if (lit >= 0) activate(lit); return; }
    const fwd = e.key === 'ArrowRight' || e.key === 'ArrowDown' || (e.key === 'Tab' && !e.shiftKey);
    const back = e.key === 'ArrowLeft' || e.key === 'ArrowUp' || (e.key === 'Tab' && e.shiftKey);
    if (fwd || back) { e.preventDefault(); light(lit < 0 ? (fwd ? 0 : n - 1) : (lit + (fwd ? 1 : -1) + n) % n, true); return; }
    if (e.repeat) { e.preventDefault(); return; }
    // the menu's own key again closes it
    if (e.code === held && !e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey) { e.preventDefault(); close(); return; }
    // a choice's own key (Alt+Z for X-ray) flips it with the menu open
    const chords = chordsOf(e);
    const k = items.findIndex((it) => (it.a.keys || []).some((c) => chords.includes(c)));
    if (k >= 0) { e.preventDefault(); activate(k); }
  }
  // a held key released after a move: the lit choice is picked and the menu closes, as in Blender's pies
  function onKeyUp(e: KeyboardEvent) {
    if (e.code !== held) return;
    if (moved && lit >= 0 && performance.now() - openedAt > 250) activate(lit, true);
  }
  const tick = () => { refresh(); raf = requestAnimationFrame(tick); };
  return {
    el: root,
    isOpen: () => !root.hidden,
    close,
    open(menu, label, key, at, heldKey = '') {
      close();
      const list = menuItems(menu);
      if (!list.length) return;
      before = document.activeElement;
      title = label; keyCap = capsOf(key).join(' ');
      const vw = innerWidth, vh = innerHeight;
      // a small window draws the menu smaller
      const R = Math.max(140, Math.min(230, Math.min(vw, vh) / 2 - 90));
      cx = at?.x ?? vw / 2; cy = at?.y ?? vh / 2;
      hub.replaceChildren(...caps(key), h('span', { text: label }));
      root.querySelectorAll('.ring-item').forEach((n) => n.remove());
      const slots = ringSlots(list.map((a, i) => a.menuOrder ?? i));
      items = list.map((a, i): Item => {
        const ang = slotAngle(a.menuOrder ?? i, slots);
        const toggle = !!a.checked;
        const check = toggle ? h('span.ring-check', { 'aria-hidden': 'true' }, iconSvg('check', 12)) : null;
        const why = h('small.ring-why', { hidden: true });
        const labelEl = h('span.ring-label', {}, h('span', { text: a.menuLabel || a.label }), why);
        const el = h('button.ring-item', { type: 'button', role: toggle ? 'menuitemcheckbox' : 'menuitem', 'data-i': String(i), 'data-action': a.id, tabindex: '-1' },
          check || '', a.icon ? iconSvg(a.icon, 16) : '', labelEl, ...(a.keys?.length ? caps(a.keys[0]) : []));
        root.append(el);
        return { a, el, ang, dx: Math.sin(ang) * R, dy: -Math.cos(ang) * R, check, why, label: labelEl };
      });
      refresh();
      lit = -1; moved = false; held = heldKey; openedAt = performance.now();
      // laid out round the pointer, then moved inwards as far as the window's edges need
      root.hidden = false;
      place();
      const boxes = [hub, ...items.map((it) => it.el)].map((e) => e.getBoundingClientRect());
      const shift = keepInside({ left: Math.min(...boxes.map((b) => b.left)), top: Math.min(...boxes.map((b) => b.top)),
        right: Math.max(...boxes.map((b) => b.right)), bottom: Math.max(...boxes.map((b) => b.bottom)) }, vw, vh);
      cx += shift.dx; cy += shift.dy;
      place();
      seg.setAttribute('x2', String(at?.x ?? cx)); seg.setAttribute('y2', String(at?.y ?? cy));
      status();
      document.addEventListener('pointermove', onMove, true);
      document.addEventListener('pointerdown', onDown, true);
      document.addEventListener('keydown', onKey, true);
      document.addEventListener('keyup', onKeyUp, true);
      raf = requestAnimationFrame(tick);
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
    // Esc closes the list at once (a search field would clear itself first)
    else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); dlg.close(); }
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

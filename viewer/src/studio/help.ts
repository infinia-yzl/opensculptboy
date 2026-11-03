// OpenSculptBoy
// Apache License, Version 2.0
//
// The help (F1 or ?; artboards 2f and 3j): the keys and the mouse from the action registry, in three tabs. By use
// groups them by how often each is used (constant, frequent, occasional, rare); the current mode's tab and Every mode
// group them by task. The search field has the focus when the help opens, and a key typed there filters the rows.

import { h, iconButton } from '../ui/dom.ts';
import { MODES } from '../ui/shortcuts.ts';
import { ui } from '../shell/state.ts';
import { capsOf, helpRows, scopeOf, TIERS, type HelpRow } from './actions.ts';

// a row's keys as caps: 'G or R', 'Shift [ ]'
function keysCell(r: HelpRow): HTMLElement {
  const dt = h('dt');
  const add = (chord: string) => {
    const caps = capsOf(chord);
    caps.forEach((c, i) => { if (i) dt.append(' '); dt.append(h('kbd.osb-key', { text: c })); });
  };
  const mouse = (m: string) => {
    const parts = m.split('+'), last = parts.pop()!;
    parts.forEach((p) => { dt.append(h('kbd.osb-key', { text: p })); dt.append(' + '); });
    dt.append(h('span.help-mouse', { text: last }));
  };
  // keys that differ from another key of the row only by a modifier the row explains (Shift for fine steps, Ctrl for
  // the opposite side) show once; a run of digits with one modifier shows as a range (Shift 1-5)
  let keys = r.keys.filter((k) => {
    const m = /^(Shift|Ctrl)\+(.+)$/.exec(k);
    return !(m && r.keys.includes(m[2]));
  });
  const run = /^((?:[A-Za-z]+\+)*)(\d)$/;
  const ms = keys.map((k) => run.exec(k));
  if (keys.length > 2 && ms.every((m) => m && m[1] === ms[0]![1])) {
    const ds = ms.map((m) => +m![2]), prefix = ms[0]![1];
    if (ds.every((d, i) => !i || d === ds[i - 1] + 1)) {
      if (prefix) {
        capsOf(prefix + '0').slice(0, -1).forEach((c) => { dt.append(h('kbd.osb-key', { text: c }), ' '); });
        dt.append(h('kbd.osb-key', { text: `${ds[0]}–${ds[ds.length - 1]}` }));
      } else ds.forEach((d, i) => { if (i) dt.append(' '); dt.append(h('kbd.osb-key', { text: String(d) })); });
      keys = [];
    }
  }
  keys.forEach((k, i) => { if (i) dt.append(keys.length > 2 ? ' ' : h('span.help-or', { text: ' or ' })); add(k); });
  r.mouse.forEach((m, i) => { if (i || keys.length) dt.append(h('span.help-or', { text: ' or ' })); mouse(m); });
  return dt;
}

export interface Help { el: HTMLDialogElement; open(tab?: string): void }

export function helpSheet(): Help {
  const close = iconButton('close', 'Close', { key: 'Esc', tip: 'left' });
  const search = h('input.help-search', { type: 'search', placeholder: 'Search keys and actions', 'aria-label': 'Search keys and actions', autocomplete: 'off', spellcheck: 'false' }) as HTMLInputElement;
  const tabs = h('div.help-tabs', { role: 'tablist', 'aria-label': 'Help' });
  const grid = h('div.help-grid');
  const empty = h('p.osb-note.help-empty', { text: 'No key matches.', hidden: true });
  const dlg = h('dialog.help.help-v2', { 'aria-labelledby': 'help-title' },
    h('header.help-head', {}, h('h2.osb-title', { id: 'help-title' }, 'Help and keys ', h('kbd.osb-key', { text: 'F1' })), close),
    h('div.help-tools', {}, tabs, search),
    grid, empty,
    h('p.help-foot.osb-note', { text: 'Each control names its key in its tooltip, and the line under the view says what the mouse does now. Left and right are the character’s own.' }),
  ) as HTMLDialogElement;
  let tab = 'use';

  const modeOfRow = (r: HelpRow) => r.mode;
  function render() {
    const mode = ui.get().mode;
    const label = MODES.find((m) => m.id === mode)?.label || 'Mode';
    tabs.replaceChildren(...[['use', 'By use'], ['mode', label], ['all', 'Every mode']].map(([id, text]) => {
      const b = h('button.help-tab', { type: 'button', role: 'tab', 'aria-selected': String(tab === id), 'data-tab': id, text });
      b.addEventListener('click', () => { tab = id; render(); search.focus(); });
      return b;
    }));
    const rows = helpRows().filter((r) => r.keys.length || r.mouse.length);
    const q = search.value.trim().toLowerCase();
    const hit = (r: HelpRow) => !q || (r.label + ' ' + r.keys.join(' ') + ' ' + r.mouse.join(' ') + ' ' + r.group).toLowerCase().includes(q);
    let groups: { title: string; note?: string; rows: HelpRow[] }[];
    if (tab === 'use') {
      groups = TIERS.map((t) => ({ title: t.label, note: t.note, rows: rows.filter((r) => r.tier === t.id && (!modeOfRow(r) || modeOfRow(r) === mode)) }));
    } else {
      const inTab = rows.filter((r) => tab === 'all' || !modeOfRow(r) || modeOfRow(r) === mode);
      const order: string[] = [];
      for (const r of inTab) if (!order.includes(r.group)) order.push(r.group);
      // the mode's own groups first, then the view, the mouse and the held keys
      const own = order.filter((g) => inTab.some((r) => r.group === g && modeOfRow(r) === mode));
      const rest = order.filter((g) => !own.includes(g));
      groups = [...own, ...rest].map((g) => ({ title: g, rows: inTab.filter((r) => r.group === g) }));
      if (tab === 'all') for (const g of groups) g.rows.sort((a, b) => Number(!!modeOfRow(a)) - Number(!!modeOfRow(b)));
    }
    let any = false;
    grid.replaceChildren(...groups.map((g) => {
      const shown = g.rows.filter(hit);
      if (!shown.length) return null;
      any = true;
      const items = shown.flatMap((r) => {
        const mode = r.mode && tab === 'all' ? ' (' + (scopeOf(r.mode)?.label || r.mode) + ')' : '';
        return [keysCell(r), h('dd', { text: r.label + mode })];
      });
      return h('section', {}, h('h3.osb-eyebrow', {}, g.title, g.note ? h('span.help-note', { text: g.note }) : ''), h('dl', {}, ...items));
    }).filter(Boolean) as HTMLElement[]);
    grid.dataset.tab = tab;
    empty.hidden = any;
  }
  search.addEventListener('input', render);
  // Esc closes the help at once, also with a query in the search (which a search field would clear first)
  search.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    e.preventDefault(); e.stopPropagation();
    dlg.close();
  });
  close.addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });
  dlg.addEventListener('close', () => { search.value = ''; });
  return {
    el: dlg,
    open(t = 'use') {
      tab = t; search.value = '';
      render();
      if (!dlg.open) dlg.showModal();
      // the search field has the focus, so typing filters at once
      search.focus();
    },
  };
}

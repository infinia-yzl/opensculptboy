// OpenSculptBoy
// Apache License, Version 2.0
//
// The points over the view, drawn in SVG (the page keeps one canvas): each placed point as a numbered blue circle,
// a point that needs a look as an amber diamond, a switched-off point as a dashed ring, a point behind the stone faint
// and dashed (an X-ray look), the selected point with a gold ring, Anny's own points as small white rings with a line
// to their places, a pushed point as a hollow ring with its depth, the magnifier's ring, the contact pins and the
// balance mark. Every mark has a dark outline and a light halo, so it holds 3 to 1 against white marble and the grey
// stage (DESIGN-0014, "Outlines for contrast"). The layer takes no pointer events; studio/pointer.ts asks the editor.

const NS = 'http://www.w3.org/2000/svg';
export const COLOURS = { placed: '#2f6fd6', look: '#f0a030', off: '#9e968a', behind: '#5f86b3', selected: '#d4ae52', outline: '#1d1b19', halo: '#ffffff',
  anny: '#ffffff', good: '#3fa66b', bad: '#d0553a', pin: '#24201c' };

export interface DrawPoint {
  name: string; number: number; x: number; y: number;
  state: 'placed' | 'look' | 'off' | 'unplaced';
  hidden: boolean; selected: boolean; hovered: boolean; pushed: number;   // pushed: mm into the stone
  small?: boolean;          // a dot of a hand cluster, without its number
}
export interface DrawAnny { name: string; x: number; y: number; tx: number; ty: number; line: boolean }
export interface DrawLabel { x: number; y: number; title: string; line?: string; side?: 'right' | 'left' }
export interface DrawCluster { x: number; y: number; title: string; line: string }
export interface DrawPin { x: number; y: number; label: string }
export interface DrawBalance { centre: [number, number]; mark: [number, number]; polygon: [number, number][]; inside: boolean }
export interface DrawMagnifier { cx: number; cy: number; r: number; zoom: number; point?: { x: number; y: number; number: number; state: DrawPoint['state'] } }
export interface DrawStroke { d: string; colour: string; width: number; dash?: string; opacity?: number }
export interface DrawTag { x: number; y: number; text: string; colour: string }
export interface DrawJoint { x: number; y: number; selected: boolean; hovered: boolean; side: 'L' | 'R' | '' }
export interface Frame {
  strokes?: DrawStroke[]; tags?: DrawTag[]; joints?: DrawJoint[];
  points: DrawPoint[]; anny: DrawAnny[]; labels: DrawLabel[]; clusters: DrawCluster[]; pins: DrawPin[]; balance: DrawBalance | null;
  magnifier: DrawMagnifier | null; scale: number;
}

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}, parent?: Element): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (parent) parent.appendChild(e);
  return e;
}

export class PointLayer {
  svg: SVGSVGElement;
  private root: SVGGElement;
  private html: HTMLDivElement;

  constructor(parent: HTMLElement) {
    this.svg = el('svg', { class: 'pose-points', 'aria-hidden': 'true' });
    Object.assign(this.svg.style, { position: 'absolute', inset: '0', width: '100%', height: '100%', pointerEvents: 'none', overflow: 'visible' });
    this.root = el('g', {}, this.svg);
    parent.appendChild(this.svg);
    // the labels are HTML, so their text wraps and takes the design system's type
    this.html = document.createElement('div');
    this.html.className = 'pose-labels';
    Object.assign(this.html.style, { position: 'absolute', inset: '0', pointerEvents: 'none' });
    parent.appendChild(this.html);
  }

  dispose() { this.svg.remove(); this.html.remove(); }
  hide() { this.root.replaceChildren(); this.html.replaceChildren(); }

  draw(f: Frame) {
    const g = el('g');
    const s = f.scale;
    // the balance mark under everything
    if (f.balance) this.balance(g, f.balance);
    // the joints' rings (the Joints tool) under the points
    for (const j of f.joints || []) {
      const r = (j.selected ? 9 : 7) * s * (j.hovered ? 4 / 3 : 1), col = j.side === 'L' ? '#2a9d8f' : j.side === 'R' ? '#e07a2f' : '#f2efe9';
      el('circle', { cx: j.x, cy: j.y, r: r + 1.5, fill: 'none', stroke: COLOURS.outline, 'stroke-width': 4.5 }, g);
      el('circle', { cx: j.x, cy: j.y, r, fill: j.selected ? '#ffffff' : 'none', 'fill-opacity': 0.25, stroke: j.selected ? COLOURS.selected : col, 'stroke-width': 2.4 }, g);
    }
    // the dial of the selected joint: its rings and their ranges
    for (const k of f.strokes || []) {
      el('path', { d: k.d, fill: 'none', stroke: COLOURS.outline, 'stroke-width': k.width + 2.5, 'stroke-opacity': 0.55 * (k.opacity ?? 1), 'stroke-linecap': 'round',
        ...(k.dash ? { 'stroke-dasharray': k.dash } : {}) }, g);
      el('path', { d: k.d, fill: 'none', stroke: k.colour, 'stroke-width': k.width, 'stroke-opacity': k.opacity ?? 1, 'stroke-linecap': 'round',
        ...(k.dash ? { 'stroke-dasharray': k.dash } : {}) }, g);
    }
    // Anny's points and their lines to their places
    for (const a of f.anny) {
      if (a.line) {
        el('line', { x1: a.x, y1: a.y, x2: a.tx, y2: a.ty, stroke: COLOURS.halo, 'stroke-width': 3, 'stroke-opacity': 0.7 }, g);
        el('line', { x1: a.x, y1: a.y, x2: a.tx, y2: a.ty, stroke: COLOURS.outline, 'stroke-width': 1.2, 'stroke-dasharray': '3 3' }, g);
      }
      el('circle', { cx: a.x, cy: a.y, r: 5 * s, fill: 'none', stroke: COLOURS.outline, 'stroke-width': 3.5 }, g);
      el('circle', { cx: a.x, cy: a.y, r: 5 * s, fill: 'none', stroke: COLOURS.anny, 'stroke-width': 2 }, g);
    }
    // hidden points first, then the visible ones, the selected and hovered last
    const order = f.points.slice().sort((a, b) => Number(!a.hidden) - Number(!b.hidden) || Number(a.selected || a.hovered) - Number(b.selected || b.hovered));
    for (const p of order) this.point(g, p, s);
    for (const q of f.pins) this.pin(g, q, s);
    if (f.magnifier) this.magnifier(g, f.magnifier, s);
    this.root.replaceWith(g);
    this.root = g;
    // the labels
    const box = document.createDocumentFragment();
    for (const c of f.clusters) box.appendChild(this.tag(c.x, c.y, c.title, c.line, 'right'));
    for (const l of f.labels) box.appendChild(this.tag(l.x, l.y, l.title, l.line, l.side || 'right'));
    for (const t of f.tags || []) {
      const d = document.createElement('div');
      d.className = 'osb-glass pose-ring-tag';
      Object.assign(d.style, { position: 'absolute', left: `${t.x}px`, top: `${t.y}px`, transform: 'translate(-50%, -50%)' });
      const i = document.createElement('i'); i.style.background = t.colour; d.appendChild(i);
      d.appendChild(document.createTextNode(t.text));
      box.appendChild(d);
    }
    this.html.replaceChildren(box);
  }

  private point(g: SVGGElement, p: DrawPoint, s: number) {
    const r = (p.small ? 3.5 : 11) * s * (p.hovered ? 4 / 3 : 1);
    const grp = el('g', { 'data-point': p.number, opacity: p.hidden ? 0.55 : 1 }, g);
    const dash = p.hidden ? { 'stroke-dasharray': '3 2' } : {};
    if (p.selected) {
      el('circle', { cx: p.x, cy: p.y, r: r + 5, fill: 'none', stroke: COLOURS.outline, 'stroke-width': 5.5 }, grp);
      el('circle', { cx: p.x, cy: p.y, r: r + 5, fill: 'none', stroke: COLOURS.selected, 'stroke-width': 3 }, grp);
    }
    if (p.pushed > 0) {
      // a pushed point: a hollow ring, and its depth beside it
      el('circle', { cx: p.x, cy: p.y, r, fill: 'none', stroke: COLOURS.halo, 'stroke-width': 5, 'stroke-opacity': 0.8 }, grp);
      el('circle', { cx: p.x, cy: p.y, r, fill: 'none', stroke: COLOURS.placed, 'stroke-width': 3, ...dash }, grp);
    } else if (p.state === 'look') {
      const d = r * 1.18, pts = `${p.x},${p.y - d} ${p.x + d},${p.y} ${p.x},${p.y + d} ${p.x - d},${p.y}`;
      el('polygon', { points: pts, fill: COLOURS.look, stroke: COLOURS.halo, 'stroke-width': 4, 'stroke-opacity': 0.75, 'stroke-linejoin': 'round' }, grp);
      el('polygon', { points: pts, fill: COLOURS.look, stroke: COLOURS.outline, 'stroke-width': 1.5, 'stroke-linejoin': 'round', ...dash }, grp);
    } else if (p.state === 'off' || p.state === 'unplaced') {
      el('circle', { cx: p.x, cy: p.y, r, fill: 'rgba(255,255,255,0.55)', stroke: COLOURS.halo, 'stroke-width': 4, 'stroke-opacity': 0.6 }, grp);
      el('circle', { cx: p.x, cy: p.y, r, fill: 'none', stroke: p.state === 'off' ? '#5b554d' : COLOURS.off, 'stroke-width': 1.6, 'stroke-dasharray': '3 2.5' }, grp);
    } else {
      const fill = p.hidden ? COLOURS.behind : COLOURS.placed;
      el('circle', { cx: p.x, cy: p.y, r, fill, stroke: COLOURS.halo, 'stroke-width': 4, 'stroke-opacity': 0.75 }, grp);
      el('circle', { cx: p.x, cy: p.y, r, fill, stroke: COLOURS.outline, 'stroke-width': 1.5, ...dash }, grp);
    }
    if (!p.small) {
      const dark = p.state === 'look' || p.state === 'off' || p.state === 'unplaced' || p.pushed > 0;
      const t = el('text', { x: p.x, y: p.y + 0.5, 'text-anchor': 'middle', 'dominant-baseline': 'central', 'font-size': 11 * s, 'font-weight': 600,
        'font-family': 'var(--font-sans)', fill: dark ? '#1d1b19' : '#ffffff', 'paint-order': 'stroke', stroke: dark ? 'rgba(255,255,255,0.8)' : 'none', 'stroke-width': 2.5 }, grp);
      t.textContent = String(p.number);
      if (p.pushed > 0) {
        const lx = p.x + r + 6;
        el('rect', { x: lx, y: p.y - 9, width: 58, height: 18, rx: 2, fill: '#141414', 'fill-opacity': 0.92 }, grp);
        const d = el('text', { x: lx + 6, y: p.y + 0.5, 'dominant-baseline': 'central', 'font-size': 11, 'font-weight': 600, 'font-family': 'var(--font-sans)', fill: '#ffffff' }, grp);
        d.textContent = `${Math.round(p.pushed)} mm in`;
      }
    }
  }

  private pin(g: SVGGElement, q: DrawPin, s: number) {
    const r = 13 * s;
    el('circle', { cx: q.x, cy: q.y, r, fill: COLOURS.pin, stroke: COLOURS.halo, 'stroke-width': 2.5 }, g);
    // a pin's head and needle
    el('path', { d: `M${q.x - 4} ${q.y - 5} l8 0 l-1.5 5 l3 2 l-11 0 l3 -2 z M${q.x} ${q.y + 2} l0 6`, fill: '#ffffff', stroke: '#ffffff', 'stroke-width': 1.2,
      'stroke-linejoin': 'round' }, g);
  }

  private balance(g: SVGGElement, b: DrawBalance) {
    const col = b.inside ? COLOURS.good : COLOURS.bad;
    if (b.polygon.length >= 3) {
      el('polygon', { points: b.polygon.map((p) => p.join(',')).join(' '), fill: col, 'fill-opacity': 0.18, stroke: col, 'stroke-width': 1.5 }, g);
    }
    el('line', { x1: b.centre[0], y1: b.centre[1], x2: b.mark[0], y2: b.mark[1], stroke: COLOURS.halo, 'stroke-width': 3.5, 'stroke-opacity': 0.7 }, g);
    el('line', { x1: b.centre[0], y1: b.centre[1], x2: b.mark[0], y2: b.mark[1], stroke: col, 'stroke-width': 1.6, 'stroke-dasharray': '5 4' }, g);
    // the centre of mass as the usual quartered disc
    const [x, y] = b.centre, r = 11;
    el('circle', { cx: x, cy: y, r, fill: '#ffffff', stroke: COLOURS.outline, 'stroke-width': 1.5 }, g);
    el('path', { d: `M${x} ${y} L${x + r} ${y} A${r} ${r} 0 0 0 ${x} ${y - r} Z M${x} ${y} L${x - r} ${y} A${r} ${r} 0 0 0 ${x} ${y + r} Z`, fill: col }, g);
    el('circle', { cx: b.mark[0], cy: b.mark[1], r: 7, fill: col, stroke: COLOURS.halo, 'stroke-width': 2 }, g);
  }

  private magnifier(g: SVGGElement, m: DrawMagnifier, s: number) {
    el('circle', { cx: m.cx, cy: m.cy, r: m.r + 2, fill: 'none', stroke: COLOURS.outline, 'stroke-width': 6 }, g);
    el('circle', { cx: m.cx, cy: m.cy, r: m.r, fill: 'none', stroke: COLOURS.selected, 'stroke-width': 2.5 }, g);
    el('line', { x1: m.cx - m.r + 8, y1: m.cy, x2: m.cx + m.r - 8, y2: m.cy, stroke: COLOURS.halo, 'stroke-opacity': 0.55, 'stroke-width': 1 }, g);
    el('line', { x1: m.cx, y1: m.cy - m.r + 8, x2: m.cx, y2: m.cy + m.r - 8, stroke: COLOURS.halo, 'stroke-opacity': 0.55, 'stroke-width': 1 }, g);
    if (m.point) this.point(g, { name: '', number: m.point.number, x: m.point.x, y: m.point.y, state: m.point.state, hidden: false, selected: true, hovered: false, pushed: 0 }, s);
    el('rect', { x: m.cx - 16, y: m.cy + m.r - 10, width: 32, height: 18, rx: 2, fill: '#141414' }, g);
    const t = el('text', { x: m.cx, y: m.cy + m.r - 0.5, 'text-anchor': 'middle', 'dominant-baseline': 'central', 'font-size': 11, 'font-weight': 600,
      'font-family': 'var(--font-sans)', fill: '#ffffff' }, g);
    t.textContent = `${m.zoom} ×`;
  }

  private tag(x: number, y: number, title: string, line: string | undefined, side: 'right' | 'left'): HTMLElement {
    const d = document.createElement('div');
    d.className = 'osb-tip pose-tag';
    Object.assign(d.style, { position: 'absolute', top: `${y}px`, transform: side === 'right' ? 'translate(0, -50%)' : 'translate(-100%, -50%)',
      left: `${side === 'right' ? x + 18 : x - 18}px`, whiteSpace: 'nowrap' });
    const t = document.createElement('div'); t.className = 'pose-tag-title'; t.textContent = title; d.appendChild(t);
    if (line) { const l = document.createElement('div'); l.className = 'pose-tag-line'; l.textContent = line; d.appendChild(l); }
    return d;
  }
}

// OpenSculptBoy
// Apache License, Version 2.0
//
// The body compass (DESIGN-0014, "Orientation and depth aids"; artboard 3k): a small figure in the lower right of the
// view that turns with the camera, its left side teal and its right side orange with L and R (the character's own
// sides), a nose that marks the front, and a ring on the floor. A click on it flies to the side of the figure under
// the pointer (front, back, the sides, top or bottom, as a ViewCube's faces do), a drag on it turns the view, and its
// home button goes back to the start view. Its hover text names the angle. It paints from the camera's changes through
// refs, so React renders nothing while the camera moves.

import { useEffect, useRef } from 'react';
import { cam } from './camera.ts';
import { runAction } from './actions.ts';
import { Icon } from '../shell/components/ui/icon.tsx';
import { Tip } from '../shell/components/ui/tooltip.tsx';

// the figure in its own frame (x to the figure's left, y up, z forward), in units of its height
type P3 = [number, number, number];
const J: Record<string, P3> = {
  head: [0, 0.86, 0], neck: [0, 0.76, 0], shL: [0.11, 0.74, 0], shR: [-0.11, 0.74, 0], elL: [0.17, 0.55, 0], elR: [-0.17, 0.55, 0],
  haL: [0.2, 0.38, 0.02], haR: [-0.2, 0.38, 0.02], hip: [0, 0.45, 0], hiL: [0.06, 0.45, 0], hiR: [-0.06, 0.45, 0],
  knL: [0.07, 0.22, 0.01], knR: [-0.07, 0.22, 0.01], ftL: [0.08, 0, 0.04], ftR: [-0.08, 0, 0.04], nose: [0, 0.86, 0.075],
};
const BONES: [string, string, 'L' | 'R' | 'C'][] = [
  ['neck', 'hip', 'C'], ['shL', 'shR', 'C'], ['hiL', 'hiR', 'C'],
  ['shL', 'elL', 'L'], ['elL', 'haL', 'L'], ['hiL', 'knL', 'L'], ['knL', 'ftL', 'L'],
  ['shR', 'elR', 'R'], ['elR', 'haR', 'R'], ['hiR', 'knR', 'R'], ['knR', 'ftR', 'R'],
];
const SIZE = 96, SCALE = 74, CX = SIZE / 2, CY = SIZE / 2 + 30;
const DEG = Math.PI / 180;

// the view's rotation: a point of the figure's frame to the screen (x right, y down) and its depth toward the viewer
function projector(yaw: number, tilt: number) {
  const cy = Math.cos(yaw * DEG), sy = Math.sin(yaw * DEG), ct = Math.cos(tilt * DEG), st = Math.sin(tilt * DEG);
  // the camera at (sin yaw, ., cos yaw) looks at the figure: its right is (cos yaw, 0, -sin yaw)
  return (p: P3) => {
    const x = p[0] * cy - p[2] * sy;
    const zf = p[0] * sy + p[2] * cy;   // toward the camera, before the tilt
    const y = p[1] - 0.45;
    const yv = y * ct - zf * st, zv = y * st + zf * ct;
    return { x: CX + x * SCALE, y: CY - 30 - yv * SCALE, z: zv };
  };
}

export function angleText(yaw: number, tilt: number): string {
  const y = Math.round(((yaw % 360) + 540) % 360 - 180);
  const named: [number, string][] = [[0, 'Front view'], [180, 'Back view'], [-180, 'Back view'], [90, 'Left side view'], [-90, 'Right side view'], [24, 'Three-quarter view']];
  const n = named.find(([a]) => Math.abs(a - y) < 1);
  const t = Math.round(tilt);
  const tiltText = Math.abs(t) >= 5 ? (t > 0 ? `, ${t}° from above` : `, ${-t}° from below`) : '';
  if (n) return n[1] + tiltText;
  return `Free view, ${Math.abs(y)}° round from the front toward the ${y > 0 ? 'left' : 'right'}${tiltText}`;
}

export function Compass() {
  const svg = useRef<SVGSVGElement>(null), box = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; moved: boolean; id: number } | null>(null);
  useEffect(() => {
    let queued = false;
    const paint = () => {
      queued = false;
      const s = cam().state();
      const P = projector(s.yaw, s.tilt), el = svg.current;
      if (!el) return;
      const pts = Object.fromEntries(Object.entries(J).map(([k, v]) => [k, P(v)]));
      // far bones first, so the near side draws over the far side
      const bones = BONES.map(([a, b, side]) => ({ a: pts[a], b: pts[b], side, z: (pts[a].z + pts[b].z) / 2 })).sort((u, v) => u.z - v.z);
      const g = el.querySelector('.cmp-body')!;
      const lines = bones.map((b) => `<line x1="${b.a.x.toFixed(1)}" y1="${b.a.y.toFixed(1)}" x2="${b.b.x.toFixed(1)}" y2="${b.b.y.toFixed(1)}" class="cmp-${b.side}"/>`).join('');
      const h = pts.head, n = pts.nose;
      g.innerHTML = lines + `<circle cx="${h.x.toFixed(1)}" cy="${h.y.toFixed(1)}" r="7" class="cmp-head"/>`
        + (n.z > h.z - 0.02 ? `<line x1="${h.x.toFixed(1)}" y1="${h.y.toFixed(1)}" x2="${n.x.toFixed(1)}" y2="${n.y.toFixed(1)}" class="cmp-nose"/>` : '');
      // the floor's ring, flattened by the tilt, with a tick at the figure's front
      const ring = el.querySelector('.cmp-floor') as SVGEllipseElement, tick = el.querySelector('.cmp-tick') as SVGLineElement;
      const f0 = P([0, 0, 0]), ry = Math.max(2, Math.abs(Math.sin(s.tilt * DEG)) * 26);
      ring.setAttribute('cx', f0.x.toFixed(1)); ring.setAttribute('cy', f0.y.toFixed(1)); ring.setAttribute('ry', ry.toFixed(1));
      const t0 = P([0, 0, 0.26]), t1 = P([0, 0, 0.34]);
      tick.setAttribute('x1', t0.x.toFixed(1)); tick.setAttribute('y1', t0.y.toFixed(1)); tick.setAttribute('x2', t1.x.toFixed(1)); tick.setAttribute('y2', t1.y.toFixed(1));
      const lL = el.querySelector('.cmp-lab-L') as SVGTextElement, lR = el.querySelector('.cmp-lab-R') as SVGTextElement;
      const aL = P([0.3, 0.3, 0]), aR = P([-0.3, 0.3, 0]);
      lL.setAttribute('x', aL.x.toFixed(1)); lL.setAttribute('y', aL.y.toFixed(1));
      lR.setAttribute('x', aR.x.toFixed(1)); lR.setAttribute('y', aR.y.toFixed(1));
      const text = angleText(s.yaw, s.tilt);
      box.current?.setAttribute('data-angle', text);
      el.setAttribute('aria-label', 'Body compass: ' + text + '. Click a side to fly there, drag to turn the view');
    };
    const queue = () => { if (!queued) { queued = true; requestAnimationFrame(paint); } };
    const off = cam().onChange(queue);
    const t = setInterval(queue, 1000);   // the camera may register after the compass mounts
    queue();
    return () => { off(); clearInterval(t); };
  }, []);

  // a click: the side of the figure under the pointer, as a point on a sphere round it
  const pick = (e: React.PointerEvent) => {
    const r = svg.current!.getBoundingClientRect();
    const dx = (e.clientX - r.left - CX) / (SIZE / 2), dy = (e.clientY - r.top - (CY - 30 - 0.0)) / (SIZE / 2);
    const len = Math.hypot(dx, dy);
    const sx = len > 1 ? dx / len : dx, sy = len > 1 ? dy / len : dy;
    const sz = Math.sqrt(Math.max(0, 1 - sx * sx - sy * sy));
    // the screen's (right, up, toward the viewer) back into the figure's frame
    const s = cam().state(), cy = Math.cos(s.yaw * DEG), syw = Math.sin(s.yaw * DEG), ct = Math.cos(s.tilt * DEG), st = Math.sin(s.tilt * DEG);
    const yv = -sy, zv = sz;
    const y = yv * ct + zv * st, zf = -yv * st + zv * ct;
    const x = sx * cy + zf * syw, z = -sx * syw + zf * cy;
    const ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
    const id = ay > ax && ay > az ? (y > 0 ? 'view.top' : 'view.bottom') : ax > az ? (x > 0 ? 'view.left' : 'view.right') : (z > 0 ? 'view.front' : 'view.back-side');
    runAction(id);
  };
  return (
    <div ref={box} className="compass osb-glass" data-angle="">
      <Tip label="Home view" keys="Home" desc="Back to the start view" side="left">
        <button type="button" className="osb-ib compass-home" aria-label="Home view (Home)" onClick={() => runAction('view.home')}><Icon name="home" size={16} /></button>
      </Tip>
      <svg ref={svg} className="compass-fig" viewBox={`0 0 ${SIZE} ${SIZE}`} width={SIZE} height={SIZE} role="img" aria-label="Body compass"
        onPointerDown={(e) => { drag.current = { x: e.clientX, moved: false, id: e.pointerId }; (e.target as Element).setPointerCapture?.(e.pointerId); }}
        onPointerMove={(e) => {
          const d = drag.current;
          if (!d || d.id !== e.pointerId) return;
          const dx = e.clientX - d.x;
          if (!d.moved && Math.abs(dx) < 3) return;
          d.moved = true; d.x = e.clientX;
          // a drag turns the view: the figure follows the pointer
          cam().turn(-dx * 1.5);
        }}
        onPointerUp={(e) => { const d = drag.current; drag.current = null; if (d && !d.moved) pick(e); }}>
        <ellipse className="cmp-floor" cx={CX} cy={CY} rx="26" ry="6" />
        <line className="cmp-tick" />
        <g className="cmp-body" />
        <text className="cmp-lab cmp-lab-L" textAnchor="middle">L</text>
        <text className="cmp-lab cmp-lab-R" textAnchor="middle">R</text>
      </svg>
    </div>
  );
}

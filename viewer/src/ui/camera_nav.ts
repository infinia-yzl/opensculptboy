// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// Navigation around the figure (DESIGN-0014, "The camera model"): a turntable orbit round an explicit pivot with no
// coast after release, a pan that keeps the grabbed spot under the pointer, a zoom to the pointer of 12 % a notch that
// slows near the skin and stops 4 cm from it, Alt+right-drag to zoom by dragging, a double-click that sets the pivot on
// the body (or frames the whole figure on the background), framings that fit a part into the free space between the
// panels and the bars, the angles (again for the opposite side), a view history of 50 views, the pivot dot while the
// camera turns, the height rail at the side of the view, keyboard steps, and the film offset that keeps the figure
// centred between the two inspectors. The camera of main.ts keeps its OrbitControls object for its state (the pivot,
// the turntable); the gestures come from the pointer arbitration of studio/pointer.ts, which gives the camera every
// press that no point, ring or handle takes. The interface for the rest of the page is studio/camera.ts.

import type * as THREE from 'three';
import type { App } from './app.ts';
import { clamp, h } from './dom.ts';
import { registerCamera, type BoxLike, type CameraState, type FrameOpts, type StudioCamera, type Vec3Like } from '../studio/camera.ts';
import { promoteRight, registerCameraDriver, registerSurface, surfaceHit, type CameraGesture } from '../studio/pointer.ts';
import { settings } from '../studio/settings.ts';
import { clearStatus, PRIORITY, setStatus } from '../studio/status.ts';

const DEG = Math.PI / 180;
// the yaw of each angle: degrees round the vertical axis from the figure's front toward the figure's left (+x)
export const ANGLES: Record<string, number> = { front: 0, three: 24, side: 90, back: 180, right: -90, left: 90 };
const OPPOSITE: Record<string, string> = { front: 'back', back: 'front', three: 'three', side: 'right', right: 'left', left: 'right' };
// the ticks of the height rail: a label and the joints it averages
const TICKS: [string, string[]][] = [['Head', ['@head']], ['Chest', ['spine01']], ['Hips', ['root']], ['Knees', ['lowerleg01.L', 'lowerleg01.R']], ['Feet', ['foot.L', 'foot.R']]];
// the turn of a drag at the default speed, degrees per pixel; the tilt from 80 degrees below to 85 above the horizon
const TURN_PER_PX = 0.25;
const TILT_LO = -80, TILT_HI = 85;
const ZOOM_NOTCH = 0.12, ZOOM_FINE = 0.03, NEAR_SURFACE = 0.04;
const HISTORY = 50;
// the cut-away works within this distance of the pivot (metres)
const CUT_RANGE = 0.9;
// the distance a double-click on the body flies in to, when the camera is farther (metres)
const POI_DISTANCE = 0.8;

export interface CameraNav {
  before(): void;
  after(dt: number): void;
  orbit(deg: number): void;
  tilt(deg: number): void;
  move(frac: number): void;
  zoom(k: number): void;
  frame(name: string): void;
  angle(name: string): void;
  // a view of the numpad or the view menu: front, back, left, right, top, bottom
  view(name: string): void;
  frameHand(side: 'L' | 'R' | null): void;
  frameFigure(): void;
  reset(): void;
  setInsets(left: number, right: number): void;
  // the top bar and the view bar over the canvas: the framings fill the height between them
  setVInsets(top: number, bottom: number): void;
  rail: HTMLElement;
  pivotDot: HTMLElement;
  api: StudioCamera;
  onView: () => void;
}

interface View { t: THREE.Vector3; p: THREE.Vector3 }

export function cameraNav(app: App): CameraNav {
  const { THREE, camera, controls, canvas } = app;
  const V = app.view;
  // the mouse's gestures come from studio/pointer.ts, which keeps them from OrbitControls; OrbitControls keeps the
  // pivot, the turntable and the touch gestures of a phone, without a coast after release
  controls.enableDamping = false;
  controls.enablePan = true;
  controls.screenSpacePanning = true;
  controls.minPolarAngle = (90 - TILT_HI) * DEG;
  controls.maxPolarAngle = (90 - TILT_LO) * DEG;
  controls.minDistance = 0.01;
  const lastT = new THREE.Vector3().copy(controls.target);
  let pivotLabel = 'the figure';
  // the camera flew away from the figure: the pivot's box round the figure waits for the next framing
  let flown = false;
  const off = new THREE.Vector3(), sph = new THREE.Spherical();
  let filmPx = 0, filmGoal = 0, lastAspect = 0;
  let railT = 0;
  const changeFns = new Set<() => void>();
  const changed = () => { for (const f of changeFns) f(); cutCheck(); };
  const userMove = () => { V.lit = ''; if (!V.userMoved) { V.userMoved = true; N.onView(); } };
  const emitStart = () => { controls.dispatchEvent({ type: 'start' }); document.body.classList.add('interacted'); };
  const emitChange = () => { controls.dispatchEvent({ type: 'change' }); app.resetAccum(); changed(); };
  // the automatic cut-away: when the camera works close to a pivot on a part (a double-click, a framing) and the body
  // stands between them more than 5 cm before the pivot, the skin nearer than that opens (DESIGN-0014, "The camera
  // model")
  let cutAt = 0, cutR = 0, cutLater: any = 0;
  function cutCheck(force = false) {
    const now = performance.now();
    if (!force && now - cutAt < 80) { clearTimeout(cutLater); cutLater = setTimeout(() => cutCheck(true), 90); return; }
    cutAt = now;
    let r = 0;
    const dp0 = camera.position.distanceTo(controls.target);
    // close work only (a framed hand, a spot zoomed in on): from farther away the cut would open the whole figure
    if (pivotLabel !== 'the figure' && pivotLabel !== 'the face' && dp0 < CUT_RANGE) {
      const toP = controls.target.clone().sub(camera.position), dp = toP.length();
      const hit = app.bodyRay(camera.position, toP.normalize());
      if (hit && hit.distance < dp - 0.05) r = dp - 0.05;
    }
    if (Math.abs(r - cutR) < 1e-4) return;
    cutR = r; display.cutAway = r > 0;
    app.setCut(r);
    if (r > 0) setStatus('cut', [{ text: 'Cut away to show the pivot' }, { keys: 'Home', text: 'the whole figure' }], { priority: PRIORITY.hover - 5 });
    else clearStatus('cut');
  }

  // ---------------------------------------------------------------- the view history
  const back: View[] = [], fwd: View[] = [];
  const here = (): View => ({ t: controls.target.clone(), p: camera.position.clone() });
  const sameView = (a: View, b: View) => a.t.distanceTo(b.t) < 1e-4 && a.p.distanceTo(b.p) < 1e-4;
  function pushView() {
    const v = here();
    if (back.length && sameView(back[back.length - 1], v)) return;
    back.push(v); if (back.length > HISTORY) back.shift();
    fwd.length = 0;
    changed();
  }
  function goTo(v: View, ms = 300) { V.userMoved = true; app.tweenTo(v.t, v.p, ms); N.onView(); changed(); }

  // ---------------------------------------------------------------- the height rail
  const thumb = h('button.hr-thumb', { type: 'button', 'aria-label': 'Camera height. Drag, or use the arrow keys.', role: 'slider', 'aria-orientation': 'vertical' });
  const ticks = h('div.hr-ticks');
  const track = h('div.hr-track', {}, h('div.hr-line'), ticks, thumb);
  const rail = h('div.hrail', { role: 'group', 'aria-label': 'Camera height' }, track);
  let range = { lo: 0, hi: 1 };
  const yOf = (clientY: number) => {
    const r = track.getBoundingClientRect();
    return range.lo + (1 - clamp((clientY - r.top) / r.height, 0, 1)) * (range.hi - range.lo);
  };
  const setHeight = (y: number) => {
    const dy = clamp(y, range.lo, range.hi) - controls.target.y;
    controls.target.y += dy; camera.position.y += dy;
    V.userMoved = true;
    controls.update(); lastT.copy(controls.target); emitChange();
  };
  let railDrag = false;
  thumb.addEventListener('pointerdown', (e) => { railDrag = true; pushView(); rail.classList.add('dragging'); thumb.setPointerCapture(e.pointerId); e.preventDefault(); });
  thumb.addEventListener('pointermove', (e) => { if (railDrag) setHeight(yOf(e.clientY)); });
  const stop = () => { railDrag = false; rail.classList.remove('dragging'); };
  thumb.addEventListener('pointerup', stop); thumb.addEventListener('pointercancel', stop);
  track.addEventListener('pointerdown', (e) => {
    if (e.target !== track && !(e.target as HTMLElement).classList.contains('hr-line')) return;
    flyHeight(yOf(e.clientY));
  });
  thumb.addEventListener('keydown', (e) => {
    const s = (range.hi - range.lo) * (e.shiftKey ? 0.1 : 0.03);
    if (e.key === 'ArrowUp') setHeight(controls.target.y + s);
    else if (e.key === 'ArrowDown') setHeight(controls.target.y - s);
    else return;
    e.preventDefault(); e.stopPropagation();
  });
  function flyHeight(y: number) {
    const t1 = controls.target.clone(); t1.y = clamp(y, range.lo, range.hi);
    const p1 = camera.position.clone().add(new THREE.Vector3(0, t1.y - controls.target.y, 0));
    pushView();
    V.userMoved = true;
    app.tweenTo(t1, p1);
  }
  const tickEls = TICKS.map(([label]) => {
    const b = h('button.hr-tick', { type: 'button', 'data-tip': 'Look at the ' + label.toLowerCase(), 'data-tip-pos': 'left', tabindex: '-1' }, h('span', { text: label }));
    ticks.append(b);
    return b;
  });
  const tickY = new Array(TICKS.length).fill(0);
  tickEls.forEach((b, i) => b.addEventListener('click', () => flyHeight(tickY[i])));
  function updateRail() {
    const f = app.figure();
    range = { lo: app.groundY(), hi: f.top + 0.12 };
    const byName = new Map(f.joints.map((j) => [j.name, j.p]));
    TICKS.forEach(([, names], i) => {
      let y = 0, n = 0;
      for (const nm of names) { const p = nm === '@head' ? f.head : byName.get(nm); if (p) { y += p.y; n++; } }
      tickY[i] = n ? y / n : range.lo;
      tickEls[i].style.top = (100 * (1 - (tickY[i] - range.lo) / (range.hi - range.lo))).toFixed(2) + '%';
    });
  }
  function paintThumb() {
    const t = clamp((controls.target.y - range.lo) / (range.hi - range.lo), 0, 1);
    thumb.style.top = (100 * (1 - t)).toFixed(2) + '%';
    thumb.setAttribute('aria-valuenow', controls.target.y.toFixed(2));
    thumb.setAttribute('aria-valuetext', `${Math.round((controls.target.y - range.lo) * 100 / 0.8916)} cm above the floor`);
  }

  // ---------------------------------------------------------------- rays and the screen
  const raycaster = new THREE.Raycaster(), ndc = new THREE.Vector2(), tmp = new THREE.Vector3(), tmp2 = new THREE.Vector3();
  function rayAt(x: number, y: number): THREE.Ray {
    const r = canvas.getBoundingClientRect();
    ndc.set((x - r.left) / r.width * 2 - 1, -((y - r.top) / r.height) * 2 + 1);
    camera.updateMatrixWorld();
    raycaster.setFromCamera(ndc, camera);
    return raycaster.ray.clone();
  }
  function toScreen(p: Vec3Like) {
    const r = canvas.getBoundingClientRect();
    tmp.set(p.x, p.y, p.z);
    const depth = tmp.clone().sub(camera.position).dot(camera.getWorldDirection(tmp2));
    tmp.project(camera);
    return { x: r.left + (tmp.x + 1) / 2 * r.width, y: r.top + (1 - tmp.y) / 2 * r.height, depth };
  }
  // the posed body as a surface for the pointer's ray tests (the double-click, the pan's depth, the zoom's slow-down)
  registerSurface('body', (ray) => app.bodyRay(ray.origin, ray.direction));
  // metres per CSS pixel at a depth from the camera
  const perPx = (depth: number) => depth * 2 * Math.tan(camera.fov * DEG / 2) / Math.max(1, canvas.getBoundingClientRect().height);

  // ---------------------------------------------------------------- moves
  function setSpherical(theta: number, phi: number) {
    off.copy(camera.position).sub(controls.target);
    sph.setFromVector3(off);
    sph.theta = theta; sph.phi = clamp(phi, controls.minPolarAngle, controls.maxPolarAngle);
    off.setFromSpherical(sph);
    camera.position.copy(controls.target).add(off);
    camera.lookAt(controls.target);
    controls.update();
  }
  function around(dTheta: number, dPhi: number) {
    off.copy(camera.position).sub(controls.target);
    sph.setFromVector3(off);
    setSpherical(sph.theta + dTheta, sph.phi + dPhi);
    emitChange();
  }
  // move the camera and the pivot by a vector
  function shift(v: THREE.Vector3) { camera.position.add(v); controls.target.add(v); controls.update(); }
  // a homothety about a point: the camera and the pivot move toward it by the factor f
  function scaleAbout(c: THREE.Vector3, f: number) {
    camera.position.sub(c).multiplyScalar(f).add(c);
    controls.target.sub(c).multiplyScalar(f).add(c);
    controls.update();
  }

  // ---------------------------------------------------------------- the gestures of the pointer
  let gest: { kind: CameraGesture | 'none'; x: number; y: number; theta0: number; phi0: number; ax: number; ay: number; depth: number; moved: boolean } =
    { kind: 'none', x: 0, y: 0, theta0: 0, phi0: 0, ax: 0, ay: 0, depth: 1, moved: false };
  let lastWheel = 0, startView: View | null = null;
  const pivotDot = h('div.pivot-dot', { 'aria-hidden': 'true', hidden: true });
  // the floor mark while the camera turns (DESIGN-0014, P2): a faint ring on the floor round the feet, with a tick at
  // the figure's front
  const NS = 'http://www.w3.org/2000/svg';
  const floorMark = document.createElementNS(NS, 'svg');
  floorMark.setAttribute('class', 'floor-mark'); floorMark.setAttribute('aria-hidden', 'true');
  const ringPath = document.createElementNS(NS, 'path'), tickPath = document.createElementNS(NS, 'path');
  tickPath.setAttribute('class', 'floor-tick');
  floorMark.append(ringPath, tickPath);
  pivotDot.append(floorMark);
  function paintPivot() {
    const s = toScreen(controls.target);
    pivotDot.style.left = s.x + 'px'; pivotDot.style.top = s.y + 'px';
    const f = app.figure(), y = app.groundY() + 0.002;
    const cx = (f.lo.x + f.hi.x) / 2, cz = (f.lo.z + f.hi.z) / 2, r = Math.max(0.3, Math.max(f.hi.x - f.lo.x, f.hi.z - f.lo.z) * 0.6);
    let d = '';
    for (let i = 0; i <= 48; i++) {
      const a = i / 48 * Math.PI * 2, q = toScreen({ x: cx + Math.sin(a) * r, y, z: cz + Math.cos(a) * r });
      if (q.depth <= 0) { d = ''; break; }
      d += (i ? 'L' : 'M') + (q.x - s.x).toFixed(1) + ' ' + (q.y - s.y).toFixed(1);
    }
    ringPath.setAttribute('d', d);
    const t0 = toScreen({ x: cx, y, z: cz + r }), t1 = toScreen({ x: cx, y, z: cz + r + 0.08 });
    tickPath.setAttribute('d', d ? `M${(t0.x - s.x).toFixed(1)} ${(t0.y - s.y).toFixed(1)}L${(t1.x - s.x).toFixed(1)} ${(t1.y - s.y).toFixed(1)}` : '');
  }
  function turnStatus() {
    setStatus('camera', [{ text: 'Turning round ' + pivotLabel, shift: { text: 'Turning slowly round ' + pivotLabel }, ctrl: { text: 'Turning in steps of 15°' } },
      { keys: 'Shift', text: 'fine', shift: null }, { keys: 'Ctrl', text: 'snap to 15°', ctrl: null }, { keys: 'Esc', text: 'back to where the turn began' }], { priority: PRIORITY.drag });
  }
  // ---------------------------------------------------------------- fly (Unreal's viewport): right-hold with W A S D
  // While the right button is held, the mouse looks round from the camera, W and S move forward and back, A and D
  // sideways, E and Q up and down, Shift moves three times as fast, and the wheel sets the speed. The pivot travels
  // with the camera at its distance, so a turn after the flight turns round what the camera looks at.
  const fly = (() => {
    const held = new Set<string>();
    let on = false, raf = 0, last = 0, speed = 0.6;
    const keys: Record<string, [number, number, number]> = { KeyW: [0, 0, 1], KeyS: [0, 0, -1], KeyA: [-1, 0, 0], KeyD: [1, 0, 0], KeyE: [0, 1, 0], KeyQ: [0, -1, 0] };
    const down = (e: KeyboardEvent) => {
      if (!keys[e.code] || e.ctrlKey || e.altKey || e.metaKey) return;
      if (!on && !promoteRight()) return;
      held.add(e.code); e.preventDefault(); e.stopImmediatePropagation();
    };
    const up = (e: KeyboardEvent) => { if (keys[e.code]) { held.delete(e.code); if (on) { e.preventDefault(); e.stopImmediatePropagation(); } } };
    addEventListener('keydown', down, true); addEventListener('keyup', up, true);
    let shift = false;
    addEventListener('keydown', (e) => { shift = e.shiftKey; }, true); addEventListener('keyup', (e) => { shift = e.shiftKey; }, true);
    const step = (t: number) => {
      if (!on) return;
      const dt = Math.min(0.05, (t - last) / 1000); last = t;
      if (held.size) {
        camera.updateMatrixWorld();
        const v = new THREE.Vector3();
        for (const c of held) { const k = keys[c]; v.x += k[0]; v.y += k[1]; v.z += k[2]; }
        const fwd = camera.getWorldDirection(new THREE.Vector3()), right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
        const m = new THREE.Vector3().addScaledVector(right, v.x).addScaledVector(fwd, v.z).addScaledVector(new THREE.Vector3(0, 1, 0), v.y);
        if (m.lengthSq() > 0) {
          m.normalize().multiplyScalar(speed * settings.get().moveSpeed * (shift ? 3 : 1) * dt);
          shift_(m); flown = true;
          userMove(); emitChange();
        }
      }
      raf = requestAnimationFrame(step);
    };
    const shift_ = (m: THREE.Vector3) => { camera.position.add(m); controls.target.add(m); controls.update(); };
    return {
      start() {
        on = true; last = performance.now(); raf = requestAnimationFrame(step);
        setStatus('camera', [{ text: 'Flying' }, { keys: 'W A S D', text: 'move' }, { keys: 'E Q', text: 'up and down' }, { keys: 'Wheel', text: 'speed' },
          { keys: 'Shift', text: 'faster' }], { priority: PRIORITY.drag });
      },
      stop() { on = false; held.clear(); cancelAnimationFrame(raf); },
      active: () => on,
      // a look round from the camera: the pivot turns about the camera at its distance
      look(dYaw: number, dPitch: number) {
        const d = camera.position.distanceTo(controls.target);
        off.copy(controls.target).sub(camera.position);
        sph.setFromVector3(off);
        sph.theta += dYaw; sph.phi = clamp(sph.phi - dPitch, 0.05, Math.PI - 0.05);
        off.setFromSpherical(sph).setLength(d);
        controls.target.copy(camera.position).add(off);
        camera.lookAt(controls.target);
        controls.update();
      },
      speed(k: number) { speed = clamp(speed * k, 0.05, 8); setStatus('camera', [{ text: `Flying at ${speed.toFixed(2)} m/s` }, { keys: 'W A S D', text: 'move' }, { keys: 'Wheel', text: 'speed' }], { priority: PRIORITY.drag }); },
    };
  })();

  // the kind of gesture of a button, after the Blender setting
  const kindOf = (k: CameraGesture, e: PointerEvent): CameraGesture => {
    if (k !== 'pan' || !settings.get().blenderMiddle) return k;
    return e.shiftKey ? 'pan' : 'orbit';
  };
  registerCameraDriver({
    begin(kind0, e, from) {
      const kind = kindOf(kind0, e);
      if (app.turntable()) app.setTurntable(false);
      pushView();
      startView = here();
      emitStart();
      off.copy(camera.position).sub(controls.target); sph.setFromVector3(off);
      let depth = camera.position.distanceTo(controls.target);
      if (kind === 'pan') {
        // the depth of the grabbed spot, so it stays under the pointer
        const hit = surfaceHit(rayAt(from.x, from.y));
        const fwdDir = camera.getWorldDirection(tmp2);
        if (hit) depth = hit.point.clone().sub(camera.position).dot(fwdDir);
        else depth = controls.target.clone().sub(camera.position).dot(fwdDir);
      }
      gest = { kind, x: from.x, y: from.y, theta0: sph.theta, phi0: sph.phi, ax: 0, ay: 0, depth: Math.max(0.02, depth), moved: false };
      if (kind === 'right') fly.start();
      else if (kind === 'orbit') { turnStatus(); pivotDot.hidden = false; paintPivot(); }
      else if (kind === 'zoom') setStatus('camera', [{ text: 'Zooming' }, { keys: 'Shift', text: 'fine' }, { keys: 'Esc', text: 'back to where it began' }], { priority: PRIORITY.drag });
      else setStatus('camera', [{ text: 'Moving the view' }, { keys: 'Esc', text: 'back to where it began' }], { priority: PRIORITY.drag });
      if (e.clientX !== from.x || e.clientY !== from.y) this.move(e);
    },
    move(e) {
      const dx = e.clientX - gest.x, dy = e.clientY - gest.y;
      gest.x = e.clientX; gest.y = e.clientY;
      if (!dx && !dy) return;
      gest.moved = true;
      if (gest.kind === 'orbit') {
        const k = TURN_PER_PX * DEG * settings.get().turnSpeed * (e.shiftKey ? 0.25 : 1);
        gest.ax -= dx * k; gest.ay -= dy * k;
        let theta = gest.theta0 + gest.ax, phi = gest.phi0 + gest.ay;
        // Ctrl: steps of 15 degrees
        if (e.ctrlKey || e.metaKey) { const s = 15 * DEG; theta = Math.round(theta / s) * s; phi = Math.round(phi / s) * s; }
        setSpherical(theta, phi);
        paintPivot();
      } else if (gest.kind === 'right') {
        // fly: the mouse looks round from where the camera stands
        const k = 0.15 * DEG * settings.get().turnSpeed * (e.shiftKey ? 0.25 : 1);
        fly.look(-dx * k, -dy * k);
      } else if (gest.kind === 'pan') {
        const m = perPx(gest.depth);
        camera.updateMatrixWorld();
        const right = tmp.setFromMatrixColumn(camera.matrixWorld, 0).multiplyScalar(-dx * m);
        const up = tmp2.setFromMatrixColumn(camera.matrixWorld, 1).multiplyScalar(dy * m);
        shift(right.add(up));
      } else if (gest.kind === 'zoom') {
        const f = Math.exp(dy * 0.005 * settings.get().zoomStep * (e.shiftKey ? 0.25 : 1));
        off.copy(camera.position).sub(controls.target);
        const d = clamp(off.length() * f, 0.03, controls.maxDistance);
        camera.position.copy(controls.target).addScaledVector(off.normalize(), d);
        controls.update();
      }
      userMove();
      emitChange();
    },
    end() {
      // no coast: the camera stops where the button is released
      if (gest.kind === 'right') fly.stop();
      gest.kind = 'none';
      pivotDot.hidden = true;
      clearStatus('camera');
      emitChange();
    },
    // Esc: the camera goes back to where the gesture began
    cancel() {
      if (gest.kind === 'right') fly.stop();
      gest.kind = 'none'; pivotDot.hidden = true; clearStatus('camera');
      if (startView) { back.pop(); app.tweenTo(startView.t, startView.p, 150); startView = null; changed(); }
    },
    wheel(e) {
      // the wheel while flying sets the speed
      if (gest.kind === 'right') { fly.speed(e.deltaY < 0 ? 1.25 : 0.8); return; }
      if (app.turntable()) app.setTurntable(false);
      const now = performance.now();
      if (now - lastWheel > 700) pushView();
      lastWheel = now;
      emitStart();
      // a notch is 100 pixels of a mouse's wheel; a touchpad gives smaller steps
      const unit = e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 600 : 1;
      const notches = clamp(-e.deltaY * unit / 100, -5, 5);
      if (!notches) return;
      const step = (e.shiftKey ? ZOOM_FINE : ZOOM_NOTCH) * settings.get().zoomStep;
      let f = Math.pow(1 - step, notches);
      // zoom toward the spot under the pointer: the skin there, else the plane of the pivot
      const ray = rayAt(e.clientX, e.clientY);
      const hit = surfaceHit(ray);
      let c: THREE.Vector3;
      if (hit) {
        c = hit.point;
        // near the skin the steps shrink with the distance, and the camera stops 4 cm from it
        if (f < 1) f = Math.max(f, NEAR_SURFACE / Math.max(NEAR_SURFACE, hit.distance));
      } else {
        const n = camera.getWorldDirection(tmp2).clone();
        const t = controls.target.clone().sub(ray.origin).dot(n) / Math.max(1e-6, ray.direction.dot(n));
        c = ray.origin.clone().addScaledVector(ray.direction, t);
      }
      const d0 = camera.position.distanceTo(controls.target);
      if (f > 1 && d0 * f > controls.maxDistance) f = controls.maxDistance / d0;
      if (Math.abs(f - 1) < 1e-4) return;
      scaleAbout(c, f);
      userMove();
      emitChange();
    },
    doubleClick(e, surface) {
      if (surface) { pivotLabel = 'this spot'; setPivotAt(surface.point, true, 300, true, true); }
      else N.frameFigure();
    },
    cursor: () => (gest.kind === 'zoom' ? 'ns-resize' : gest.kind === 'right' ? 'crosshair' : 'grabbing'),
  });

  // the pivot on a point; with `centre`, the camera moves so the point stands in the middle of the view, at the same angle
  function setPivotAt(p: Vec3Like, centre: boolean, ms = 300, history = true, approach = false) {
    const t1 = new THREE.Vector3(p.x, p.y, p.z);
    if (history) pushView();
    if (!centre) { controls.target.copy(t1); camera.lookAt(controls.target); controls.update(); emitChange(); return; }
    const dir = camera.position.clone().sub(controls.target).normalize();
    // a double-click also comes in to working distance, as Sketchfab's does (the point of interest of Mackinlay, Card
    // and Robertson): from the whole figure to a hand in one flight; closer views keep their distance
    const d0 = camera.position.distanceTo(t1);
    const d = Math.max(0.05, approach ? Math.min(d0, POI_DISTANCE) : d0);
    V.userMoved = true;
    app.tweenTo(t1, t1.clone().addScaledVector(dir, d), ms);
    N.onView(); changed();
  }

  // ---------------------------------------------------------------- framing a part
  // fit a box into the free space between the panels and the bars: at the current angle when it shows the part, else
  // turned to the side of the figure that the part faces
  function frameBounds(box: BoxLike | Vec3Like[], o: FrameOpts = {}) {
    const pts: THREE.Vector3[] = Array.isArray(box) ? box.map((p) => new THREE.Vector3(p.x, p.y, p.z))
      : [0, 1, 2, 3, 4, 5, 6, 7].map((i) => new THREE.Vector3(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z));
    if (!pts.length) return;
    const lo = pts[0].clone(), hi = pts[0].clone();
    for (const p of pts) { lo.min(p); hi.max(p); }
    const c = o.pivot ? new THREE.Vector3(o.pivot.x, o.pivot.y, o.pivot.z) : lo.clone().add(hi).multiplyScalar(0.5);
    let dir = camera.position.clone().sub(controls.target).normalize();
    if (!o.keepAngle) {
      // the part hidden behind the body from here: turn to the side it faces (horizontally, from the figure's axis)
      const toC = c.clone().sub(camera.position), dist = toC.length();
      const hit = app.bodyRay(camera.position, toC.clone().normalize());
      const rad = hi.distanceTo(lo) / 2;
      if (hit && hit.distance < dist - rad - 0.01) {
        const f = app.figure();
        const axis = new THREE.Vector3((f.lo.x + f.hi.x) / 2, c.y, (f.lo.z + f.hi.z) / 2);
        const out = c.clone().sub(axis); out.y = 0;
        if (out.lengthSq() > 1e-6) {
          out.normalize();
          const tiltY = clamp(dir.y, -0.5, 0.6);
          dir = out.multiplyScalar(Math.sqrt(1 - tiltY * tiltY)).setY(tiltY).normalize();
        }
      }
    }
    // the half extents of the part across the view, and its depth toward the camera
    const up0 = Math.abs(dir.y) > 0.99 ? new THREE.Vector3(0, 0, -Math.sign(dir.y)) : new THREE.Vector3(0, 1, 0);
    const right = new THREE.Vector3().crossVectors(up0, dir).normalize(), up = new THREE.Vector3().crossVectors(dir, right).normalize();
    let hx = 0, hy = 0, hz = 0;
    for (const p of pts) { const q = p.clone().sub(c); hx = Math.max(hx, Math.abs(q.dot(right))); hy = Math.max(hy, Math.abs(q.dot(up))); hz = Math.max(hz, q.dot(dir)); }
    const m = 1 + (o.margin ?? 0.12) * 2;
    const t = Math.tan(camera.fov * DEG / 2);
    const fh = (V.freeH || innerHeight) / innerHeight, fw = (V.freeW || innerWidth) / innerHeight;
    let d = Math.max(hy * m / (t * fh), hx * m / (t * fw)) + hz;
    d = clamp(Math.max(d, hz + NEAR_SURFACE + 0.02), 0.05, controls.maxDistance);
    // the free band between the top bar and the view bar is not the canvas's middle: lift the view by its shift
    const lift = (V.shiftY / (innerHeight / 2)) * t * d;
    const t1 = c.clone().addScaledVector(up, lift);
    const p1 = t1.clone().addScaledVector(dir, d);
    if (o.history !== false) pushView();
    V.userMoved = true;
    pivotLabel = o.label ? o.label.replace(/^[A-Z]/, (c) => 'the ' + c.toLowerCase()) : 'the selection';
    if (o.ms === 0) { controls.target.copy(t1); camera.position.copy(p1); camera.lookAt(t1); controls.update(); emitChange(); }
    else app.tweenTo(t1, p1, o.ms ?? 300);
    N.onView(); changed();
  }
  // the joints of a hand, with each fingertip one segment past the last joint
  function handPoints(side: 'L' | 'R'): THREE.Vector3[] {
    const f = app.figure();
    const by = new Map(f.joints.map((j) => [j.name, j.p]));
    const pts: THREE.Vector3[] = [];
    const wrist = by.get('wrist.' + side);
    if (wrist) pts.push(wrist.clone());
    for (let i = 1; i <= 5; i++) {
      for (let k = 1; k <= 3; k++) { const p = by.get(`finger${i}-${k}.${side}`); if (p) pts.push(p.clone()); }
      const a = by.get(`finger${i}-2.${side}`), b = by.get(`finger${i}-3.${side}`);
      if (a && b) pts.push(b.clone().multiplyScalar(2).sub(a).add(b.clone().sub(a).multiplyScalar(0.1)));
      const mc = by.get(`metacarpal${i}.${side}`); if (mc) pts.push(mc.clone());
    }
    // the skin round the bones: a centimetre and a half on every side
    const s = 0.015, out: THREE.Vector3[] = [];
    for (const p of pts) for (const d of [[s, 0, 0], [-s, 0, 0], [0, s, 0], [0, -s, 0], [0, 0, s], [0, 0, -s]]) out.push(p.clone().add(new THREE.Vector3(d[0], d[1], d[2])));
    return out;
  }
  let lastHand: 'L' | 'R' = 'L';

  // ---------------------------------------------------------------- the interface for the page
  const api: StudioCamera = {
    frameBounds,
    setPivot: (p, o = {}) => setPivotAt(p, !!o.centre, o.ms ?? 300, o.history !== false),
    pivot: () => controls.target.clone(),
    pushView,
    previousView() {
      if (!back.length) return false;
      fwd.push(here()); if (fwd.length > HISTORY) fwd.shift();
      goTo(back.pop()!);
      return true;
    },
    nextView() {
      if (!fwd.length) return false;
      back.push(here());
      goTo(fwd.pop()!);
      return true;
    },
    isolate(objs) { display.isolated = objs && objs.length ? objs : null; app.setXray(display.xray, display.isolated); changed(); },
    isolated: () => display.isolated,
    xray(on) { if (display.xray === on) return; display.xray = on; app.setXray(on, display.isolated); changed(); },
    xrayOn: () => display.xray,
    rayAt,
    toScreen,
    turn(dYaw, dTilt = 0) { if (app.turntable()) app.setTurntable(false); around(dYaw * DEG, -dTilt * DEG); userMove(); },
    state(): CameraState {
      off.copy(camera.position).sub(controls.target); sph.setFromVector3(off);
      return {
        yaw: ((sph.theta / DEG) % 360 + 540) % 360 - 180, tilt: 90 - sph.phi / DEG, distance: sph.radius,
        pivot: controls.target.toArray() as [number, number, number], position: camera.position.toArray() as [number, number, number],
        back: back.length, forward: fwd.length, xray: display.xray, isolated: !!display.isolated, cutAway: display.cutAway, flying: app.tweening(),
      };
    },
    onChange(fn) { changeFns.add(fn); return () => { changeFns.delete(fn); }; },
  };
  const display = { xray: false, isolated: null as THREE.Object3D[] | null, cutAway: false };
  registerCamera(api);

  // whether the pivot lies on the figure (within its posed bounds and a little more)
  const pivotOnFigure = () => {
    const f = app.figure(), t = controls.target, m = 0.06;
    return t.x > f.lo.x - m && t.x < f.hi.x + m && t.y > app.groundY() - m && t.y < f.top + m && t.z > f.lo.z - m && t.z < f.hi.z + m;
  };

  const N: CameraNav = {
    rail, pivotDot, api,
    onView: () => {},
    before() {
      // the camera stays 3 cm above the floor: the lowest polar angle for this target and distance
      const d = Math.max(1e-3, camera.position.distanceTo(controls.target));
      const c = clamp((app.groundY() + 0.03 - controls.target.y) / d, -1, 1);
      controls.maxPolarAngle = Math.max(controls.minPolarAngle + 0.01, Math.min((90 - TILT_LO) * DEG, Math.acos(c)));
      lastT.copy(controls.target);
    },
    after(dt: number) {
      if (app.tweening()) flown = false;
      if (!app.tweening() && !flown) {
        // the pivot stays in a box around the posed figure
        const f = app.figure(), t = controls.target;
        const x = clamp(t.x, f.lo.x - 0.35, f.hi.x + 0.35), y = clamp(t.y, app.groundY(), f.top + 0.15), z = clamp(t.z, f.lo.z - 0.35, f.hi.z + 0.35);
        if (x !== t.x || y !== t.y || z !== t.z) {
          const dx = x - t.x, dy = y - t.y, dz = z - t.z;
          t.set(x, y, z); camera.position.x += dx; camera.position.y += dy; camera.position.z += dz;
          controls.update();
        }
      } else if (app.tweening()) changed();
      // a hard limit as well: a flight can end below the floor's polar limit of this frame
      const lim = app.groundY() + 0.03;
      if (camera.position.y < lim) {
        off.copy(camera.position).sub(controls.target);
        sph.setFromVector3(off);
        sph.phi = Math.min(sph.phi, Math.acos(clamp((lim - controls.target.y) / Math.max(1e-3, sph.radius), -1, 1)));
        off.setFromSpherical(sph);
        camera.position.copy(controls.target).add(off);
        camera.lookAt(controls.target);
      }
      // the film offset eases toward the centre of the free space
      const aspect = camera.aspect;
      if (Math.abs(filmGoal - filmPx) > 0.25 || aspect !== lastAspect) {
        filmPx += (filmGoal - filmPx) * Math.min(1, dt * 10);
        if (Math.abs(filmGoal - filmPx) <= 0.25) filmPx = filmGoal;
        lastAspect = aspect;
        const fw = camera.getFilmWidth();
        camera.filmOffset = -(filmPx / innerWidth) * 2 * Math.tan(camera.fov * DEG / 2) * aspect * fw;
        camera.updateProjectionMatrix();
        app.resetAccum();
      }
      if (!pivotDot.hidden) paintPivot();
      railT += dt;
      if (railT > 0.1) { railT = 0; updateRail(); }
      paintThumb();
    },
    orbit(deg) { if (app.turntable()) app.setTurntable(false); around(deg * DEG * settings.get().turnSpeed, 0); userMove(); },
    tilt(deg) { around(0, -deg * DEG * settings.get().turnSpeed); userMove(); },
    move(frac) {
      const f = app.figure();
      const dy = frac * (f.top - app.groundY()) * settings.get().moveSpeed;
      const y = clamp(controls.target.y + dy, app.groundY(), f.top + 0.15);
      camera.position.y += y - controls.target.y; controls.target.y = y;
      V.userMoved = true; controls.update(); lastT.copy(controls.target); emitChange();
      N.onView();
    },
    zoom(k) {
      off.copy(camera.position).sub(controls.target);
      const d = clamp(off.length() * k, controls.minDistance, controls.maxDistance);
      camera.position.copy(controls.target).addScaledVector(off.normalize(), d);
      controls.update(); userMove(); emitChange();
    },
    frame(name) {
      if (name === 'hand') { N.frameHand(null); return; }
      pushView(); V.userMoved = false; V.lit = ''; pivotLabel = name === 'face' ? 'the face' : 'the figure'; app.flyTo(name); N.onView(); changed();
    },
    // an angle of the view bar: pressed again, the opposite side; after a free pan the figure comes back to the middle
    angle(name) {
      // the camera already at this angle: the opposite side
      const yawNow = api.state().yaw, want = ANGLES[name];
      const at = want !== undefined && Math.abs(((yawNow - want) % 360 + 540) % 360 - 180) < 1 && !app.tweening();
      if (at && OPPOSITE[name] && OPPOSITE[name] !== name) name = OPPOSITE[name];
      pushView();
      V.yaw = ANGLES[name] ?? 24;
      if (!V.userMoved || !pivotOnFigure()) { V.userMoved = false; app.flyTo(app.currentFrame()); N.onView(); changed(); return; }
      // a view of the user's: turn about the pivot at the same distance and height
      off.copy(camera.position).sub(controls.target);
      sph.setFromVector3(off);
      sph.theta = V.yaw * DEG;
      off.setFromSpherical(sph);
      app.tweenTo(controls.target.clone(), controls.target.clone().add(off));
      N.onView(); changed();
    },
    view(name) {
      if (name === 'top' || name === 'bottom') {
        pushView();
        off.copy(camera.position).sub(controls.target); sph.setFromVector3(off);
        sph.phi = (90 - (name === 'top' ? TILT_HI : TILT_LO)) * DEG;
        if (name === 'bottom') sph.phi = Math.min(sph.phi, controls.maxPolarAngle);
        off.setFromSpherical(sph);
        V.userMoved = true;
        app.tweenTo(controls.target.clone(), controls.target.clone().add(off));
        N.onView(); changed();
        return;
      }
      const yaw = ANGLES[name];
      if (yaw === undefined) return;
      pushView();
      V.yaw = yaw;
      if (!V.userMoved || !pivotOnFigure()) { V.userMoved = false; app.flyTo(app.currentFrame()); N.onView(); changed(); return; }
      off.copy(camera.position).sub(controls.target); sph.setFromVector3(off);
      sph.theta = yaw * DEG; sph.phi = Math.PI / 2 - 4 * DEG;
      off.setFromSpherical(sph);
      app.tweenTo(controls.target.clone(), controls.target.clone().add(off));
      N.onView(); changed();
    },
    // the Hand framing: the given hand, else the other hand than the last one framed (the right hand first)
    frameHand(side) {
      const s = side || (lastHand === 'L' ? 'R' : 'L');
      lastHand = s;
      frameBounds(handPoints(s), { label: s === 'R' ? 'Right hand' : 'Left hand', margin: 0.3 });
      V.lit = 'hand';
      N.onView();
    },
    // the whole figure at the current angle
    frameFigure() {
      const f = app.figure();
      pushView();
      off.copy(camera.position).sub(controls.target); sph.setFromVector3(off);
      V.yaw = Math.round(sph.theta / DEG);
      V.userMoved = false; V.lit = ''; pivotLabel = 'the figure';
      app.flyTo('body');
      N.onView(); changed();
      void f;
    },
    reset() { pushView(); V.yaw = ANGLES.three; V.userMoved = false; app.flyTo('body'); N.onView(); changed(); },
    setVInsets(top, bottom) {
      const freeH = Math.max(200, innerHeight - top - bottom), shiftY = app.shot ? 0 : (top - bottom) / 2;
      // the bars change their height by a few pixels with the layout (a wrapped view bar): the framing keeps its place
      if (V.freeH && Math.abs(freeH - V.freeH) < 16 && Math.abs(shiftY - V.shiftY) < 16) return;
      V.freeH = freeH; V.shiftY = shiftY;
      if (!V.userMoved && !app.tweening()) app.flyTo(app.currentFrame());
    },
    setInsets(left, right) {
      V.freeW = Math.max(200, innerWidth - left - right);
      filmGoal = app.shot ? 0 : (left - right) / 2;
      document.documentElement.style.setProperty('--inset-l', left + 'px');
      document.documentElement.style.setProperty('--inset-r', right + 'px');
    },
  };
  updateRail();
  return N;
}

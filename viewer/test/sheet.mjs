// OpenSculptBoy
// Apache License, Version 2.0
//
// The character sheet's settings and layout maths without a browser (run by test/test_viewer_sheet.py):
//   node viewer/test/sheet.mjs
// It checks the defaults and the normalisation of the settings object, the probe boxes, the framing of full figures
// (one scale, one ground row, every figure inside its frame), the head-and-shoulders frame and the page layout, and
// prints a JSON summary. A failed check throws, so node exits with an error.
import assert from 'node:assert/strict';
import {
  defaultSettings, DEFAULT_EXPRESSIONS, DEFAULT_POSES, normaliseSettings, shownPanels, settingsKey, TURNAROUND_POSE, VIEW_ORDER,
} from '../src/sheet/settings.ts';
import { coverage, faceFrame, figureFrames, headFrame, maskBox, pageLayout, rows } from '../src/sheet/layout.ts';

const out = {};

// ---------------------------------------------------------------- settings
{
  const d = defaultSettings();
  assert.equal(d.panels.turnaround.pose.pose, TURNAROUND_POSE);
  assert.deepEqual(d.panels.turnaround.views, ['front', 'three', 'side', 'back']);
  assert.deepEqual(d.panels.expressions.items, ['neutral', 'happy', 'angry', 'sad', 'relaxed', 'surprised', 'blink', 'aa', 'ih', 'ou', 'ee', 'oh']);
  assert.equal(d.panels.poses.slots.length, 6);
  assert.equal(d.panels.expressions.framing, 'face');
  assert.equal(normaliseSettings({ panels: { expressions: { framing: 'shoulders' } } }).panels.expressions.framing, 'shoulders');
  assert.equal(normaliseSettings({ panels: { expressions: { framing: 'feet' } } }).panels.expressions.framing, 'face');
  assert.deepEqual(d.panels.poses.slots.map((s) => s.pose), DEFAULT_POSES);
  assert.deepEqual(shownPanels(d), ['info', 'turnaround', 'expressions', 'poses']);
  // an empty object, and no object, give the defaults; the settings survive JSON and a second normalisation
  assert.deepEqual(normaliseSettings({}), d);
  assert.deepEqual(normaliseSettings(null), d);
  assert.deepEqual(normaliseSettings(JSON.parse(JSON.stringify(d))), d);
  // bad values drop out and missing ones take the defaults
  const q = Array.from({ length: 8 }, (_, i) => (i % 4 === 3 ? 1 : 0.123456789));
  const odd = normaliseSettings({
    style: 'pencil', background: 'neon', name: 7, title: 'Keeper of the lantern',
    panels: {
      turnaround: { views: ['back', 'left', 'front', 'front'], pose: 'mh_at_ease', style: 'textured' },
      expressions: { on: false, view: 'up', items: ['happy', 'happy', 3, '', 'sad'] },
      poses: { slots: ['ready', { pose: 'x', q: [1, 2, 3] }, { pose: 'own', label: 'From a picture', q }, {}, ...Array(12).fill('crouch')] },
      info: { on: 'yes' },
    },
    render: { scale: 9, samples: 0.2 },
    extra: true,
  });
  assert.equal(odd.style, 'textured');
  assert.equal(odd.background, 'paper');
  assert.equal(odd.name, '');
  assert.equal(odd.title, 'Keeper of the lantern');
  assert.deepEqual(odd.panels.turnaround.views, ['front', 'back']);
  assert.deepEqual(odd.panels.turnaround.pose, { pose: 'mh_at_ease' });
  assert.equal(odd.panels.turnaround.style, 'textured');
  assert.equal(odd.panels.expressions.on, false);
  assert.equal(odd.panels.expressions.view, 'front');
  assert.deepEqual(odd.panels.expressions.items, ['happy', 'sad']);
  assert.equal(odd.panels.poses.slots.length, 8);
  assert.deepEqual(odd.panels.poses.slots[0], { pose: 'ready' });
  assert.deepEqual(odd.panels.poses.slots[1], { pose: 'x' });   // a q of the wrong length drops out
  assert.equal(odd.panels.poses.slots[2].q.length, 8);
  assert.equal(odd.panels.poses.slots[2].q[0], 0.12346);
  assert.equal(odd.panels.info.on, true);
  assert.deepEqual(odd.render, { scale: 2, samples: 1 });
  assert.equal('extra' in odd, false);
  assert.equal(settingsKey(normaliseSettings(odd)), settingsKey(odd));
  // a panel without data stays out
  assert.deepEqual(shownPanels(normaliseSettings({ panels: { turnaround: { views: [] }, poses: { slots: [] } } })), ['info', 'expressions']);
  out.settings = { defaults: d, odd };
}

// ---------------------------------------------------------------- probe boxes and coverage
{
  const w = 100, h = 200, a = new Uint8Array(w * h);
  for (let j = 50; j < 150; j++) for (let i = 20; i < 40; i++) a[j * w + i] = 255;
  const probe = { cx: 0, cy: 1, halfH: 1 };   // 1 cm per pixel
  const m = maskBox(a, w, h, probe);
  const near = (x, y) => assert.ok(Math.abs(x - y) < 1e-9, `${x} != ${y}`);
  near(m.box.x0, -0.3); near(m.box.x1, -0.1); near(m.box.y1, 1.5); near(m.box.y0, 0.5);
  assert.equal(m.edge, false);
  a[0] = 255;
  assert.equal(maskBox(a, w, h, probe).edge, true);
  assert.equal(maskBox(new Uint8Array(16), 4, 4, probe), null);
  const rgba = new Uint8Array(16); rgba[3] = 255; rgba[7] = 4;
  assert.equal(coverage(rgba), 0.25);
}

// ---------------------------------------------------------------- framing of full figures
{
  // four views of a figure from the ground (-0.84) to its hair, with the side view narrower and a raised hand in one
  const ground = -0.84;
  const boxes = [
    { x0: -0.3, x1: 0.3, y0: -0.84, y1: 0.66 },
    { x0: -0.22, x1: 0.26, y0: -0.84, y1: 0.66 },
    { x0: -0.14, x1: 0.16, y0: -0.84, y1: 0.66 },
    { x0: -0.3, x1: 0.31, y0: -0.845, y1: 0.9 },
  ];
  const ff = figureFrames(boxes, ground, 2048);
  const top = Math.max(...boxes.map((b) => b.y1)), bottom = Math.min(ground, ...boxes.map((b) => b.y0));
  assert.ok(ff.y1 > top && ff.y0 < bottom, 'the frames reach above the tops and below the ground');
  for (const [i, f] of ff.frames.entries()) {
    const b = boxes[i];
    assert.equal(f.height, 2048);
    assert.equal(f.width % 2, 0);
    // one scale: the frame's height in metres is the same for all, and the width follows the figure
    assert.ok(Math.abs(2 * f.halfH * ff.pxPerM - 2048) < 1e-6);
    assert.ok(Math.abs(f.cy - (ff.y0 + ff.y1) / 2) < 1e-12, 'one centre height, so the ground lies on one row');
    // the figure inside its frame, with a margin
    const half = f.width / ff.pxPerM / 2;
    assert.ok(f.cx - half < b.x0 && f.cx + half > b.x1, `view ${i} inside its frame`);
    // the ground's row from the frame
    const row = (f.cy + f.halfH - ground) * ff.pxPerM;
    assert.ok(Math.abs(row - ff.groundRow) < 1e-6);
  }
  assert.ok(ff.frames[2].width < ff.frames[0].width, 'the side view is narrower');
  out.turnaround = ff;
  // the head-and-shoulders frame
  const hf = headFrame(0.66, 0.36, 0.01, 960, 0.9);
  assert.equal(hf.height, 960); assert.equal(hf.width, 864);
  assert.ok(hf.cy + hf.halfH > 0.66 && hf.cy - hf.halfH < 0.36, 'the frame holds the top of the head and the shoulders');
  assert.ok(Math.abs(hf.cx - 0.01) < 1e-12);
  out.head = hf;
  // the face frame: the chin and the top of the head inside, the head about three quarters of the height (the face,
  // from the chin to the hairline, about 60 %), tall headgear cut a third of the head above its top
  const chin = 0.414, crown = 0.61, H = crown - chin;
  const ff2 = faceFrame(chin, crown, 0.65, 0.0, 960, 0.8);
  assert.equal(ff2.width, 768);
  const lo = ff2.cy - ff2.halfH, hi = ff2.cy + ff2.halfH;
  assert.ok(lo < chin && hi > 0.65, 'the chin and the hair inside');
  const share = H / (hi - lo);
  assert.ok(share > 0.7 && share < 0.82, `the head fills ${share}`);
  assert.ok(Math.abs(0.75 * share - 0.58) < 0.05, 'the face fills about 60 %');
  const tall = faceFrame(chin, crown, crown + 0.4, 0.0, 960, 0.8);
  assert.ok(tall.cy + tall.halfH < crown + 0.4 * H, 'tall headgear is cut');
  out.face = { frame: ff2, share };
}

// ---------------------------------------------------------------- the page
{
  assert.deepEqual(rows(12), [6, 6]); assert.deepEqual(rows(14), [5, 5, 4]); assert.deepEqual(rows(6), [6]); assert.deepEqual(rows(4), [4]); assert.deepEqual(rows(7), [4, 3]);
  const turn = [{ w: 820, h: 2048 }, { w: 700, h: 2048 }, { w: 520, h: 2048 }, { w: 830, h: 2048 }];
  const L = pageLayout({ k: 1, name: 'Milo', title: 'A wanderer', subtitle: 'Humanoid 3D character', panels: [
    { id: 'info', title: 'Info block', sizes: [], labels: [] },
    { id: 'turnaround', title: 'Turnaround', sizes: turn, labels: ['Front', 'Three-quarter', 'Side', 'Back'], groundRow: 1990 },
    { id: 'expressions', title: 'Expressions', sizes: Array(12).fill({ w: 864, h: 960 }), labels: Array(12).fill('Happy') },
    { id: 'poses', title: 'Poses', sizes: [900, 600, 700, 650, 1100, 720].map((w) => ({ w, h: 1536 })), labels: Array(6).fill('Pose'), groundRow: 1500 },
  ] });
  assert.equal(L.width, 4096);
  const pics = L.items.filter((i) => i.type === 'picture');
  assert.equal(pics.length, 4 + 12 + 6);
  for (const p of pics) assert.ok(p.x >= 0 && p.x + p.w <= L.width + 1e-6 && p.y >= 0 && p.y + p.h <= L.height, 'pictures on the page');
  for (let a = 0; a < pics.length; a++) for (let b = a + 1; b < pics.length; b++) {
    const P = pics[a], Q = pics[b];
    const apart = P.x + P.w <= Q.x + 1e-6 || Q.x + Q.w <= P.x + 1e-6 || P.y + P.h <= Q.y + 1e-6 || Q.y + Q.h <= P.y + 1e-6;
    assert.ok(apart, `pictures ${a} and ${b} overlap`);
  }
  // the turnaround: one scale (each picture keeps its aspect and one height), one ground line through all four
  const T = pics.filter((p) => p.panel === 'turnaround');
  const fit = T[0].h / 2048;
  for (const [i, p] of T.entries()) { assert.ok(Math.abs(p.h - T[0].h) < 1e-6); assert.ok(Math.abs(p.w / p.h - turn[i].w / 2048) < 1e-9); assert.ok(Math.abs(p.y - T[0].y) < 1e-6); }
  const lines = L.items.filter((i) => i.type === 'line' && i.role === 'faint');
  const gy = T[0].y + 1990 * fit;
  assert.ok(lines.some((l) => Math.abs(l.y0 - gy) < 1e-6 && l.x0 <= T[0].x && l.x1 >= T[3].x + T[3].w), 'the ground line runs under the four views');
  // the expression grid: equal cells in two rows of six
  const E = pics.filter((p) => p.panel === 'expressions');
  assert.equal(new Set(E.map((p) => Math.round(p.y))).size, 2);
  for (const p of E) assert.ok(Math.abs(p.w - E[0].w) < 1e-6 && Math.abs(p.h - E[0].h) < 1e-6);
  // the texts: the name, the title and a label under every picture
  const texts = L.items.filter((i) => i.type === 'text').map((t) => t.text);
  assert.ok(texts.includes('Milo') && texts.includes('A wanderer') && texts.includes('Side'));
  assert.equal(texts.filter((t) => t === 'Happy').length, 12);
  // a draft at half size halves the page
  const half = pageLayout({ k: 0.5, name: 'Milo', title: '', subtitle: '', panels: [{ id: 'turnaround', title: 'Turnaround', sizes: turn.map((s) => ({ w: s.w / 2, h: s.h / 2 })), labels: [] }] });
  assert.equal(half.width, 2048);
  out.page = { width: L.width, height: L.height, pictures: pics.length };
}

console.log(JSON.stringify({ ok: true, page: out.page, turnaround: { pxPerM: out.turnaround.pxPerM, groundRow: out.turnaround.groundRow, widths: out.turnaround.frames.map((f) => f.width) } }));

// OpenSculptBoy
// Apache License, Version 2.0
//
// Draw a page layout (layout.ts) on a 2D canvas: the background, the texts, the ground lines and the pictures. A
// picture that is not built yet draws as a dashed frame, so the Sheet view shows the page as it fills.

import type { Background } from './settings.ts';
import type { Item, PageLayout } from './layout.ts';

// the colours of each background: the text and the lines follow the ground, so they stay readable on it (a
// transparent page keeps the paper's text colours, for a light place)
export const COLOURS: Record<Background, { bg: string | null; ink: string; muted: string; faint: string; line: string }> = {
  grey: { bg: '#3b3936', ink: '#f0ebe4', muted: '#c4bcb2', faint: '#a1998f', line: '#5c5751' },
  paper: { bg: '#f1eee8', ink: '#1d1b19', muted: '#6d665e', faint: '#857d73', line: '#d6d0c7' },
  dark: { bg: '#141518', ink: '#ece7e1', muted: '#a39b93', faint: '#6f6962', line: '#34332f' },
  transparent: { bg: null, ink: '#1d1b19', muted: '#6d665e', faint: '#857d73', line: '#d6d0c7' },
};
const FONTS = { serif: '"Instrument Serif", "Iowan Old Style", Georgia, "Times New Roman", serif', sans: '"Instrument Sans", "Helvetica Neue", Arial, system-ui, sans-serif' };

// the page's fonts, once: they come from the page's font links when the browser is online, else the fallbacks draw
let fontsReady: Promise<void> | null = null;
export function loadFonts(): Promise<void> {
  if (!fontsReady) {
    const fonts = (document as any).fonts;
    const all = fonts ? Promise.all(['italic 100px "Instrument Serif"', '400 40px "Instrument Sans"', '600 40px "Instrument Sans"'].map((f) => fonts.load(f).catch(() => null))) : Promise.resolve();
    fontsReady = Promise.race([all.then(() => undefined), new Promise<void>((r) => setTimeout(r, 1500))]);
  }
  return fontsReady;
}

export type PictureSource = (panel: string, index: number) => CanvasImageSource | null;

// draw the layout at `scale` (1: the print size) into a new canvas
export function compose(layout: PageLayout, background: Background, picture: PictureSource, scale = 1): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(layout.width * scale)); c.height = Math.max(1, Math.round(layout.height * scale));
  const g = c.getContext('2d')!;
  const col = COLOURS[background] || COLOURS.grey;
  if (col.bg) { g.fillStyle = col.bg; g.fillRect(0, 0, c.width, c.height); }
  g.scale(scale, scale);
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
  const role = (r: string) => (col as any)[r] || col.ink;
  for (const it of layout.items) drawItem(g, it, role, picture);
  return c;
}

function drawItem(g: CanvasRenderingContext2D, it: Item, role: (r: string) => string, picture: PictureSource) {
  if (it.type === 'line') {
    g.save(); g.strokeStyle = role(it.role); g.lineWidth = it.width; g.setLineDash(it.dash || []);
    g.beginPath(); g.moveTo(it.x0, it.y0); g.lineTo(it.x1, it.y1); g.stroke(); g.restore();
    return;
  }
  if (it.type === 'picture') {
    const src = picture(it.panel, it.index);
    if (src) { g.drawImage(src, it.x, it.y, it.w, it.h); return; }
    g.save(); g.strokeStyle = role('line'); g.lineWidth = Math.max(2, it.w / 160); g.setLineDash([it.w / 30, it.w / 40]);
    const r = Math.min(it.w, it.h) * 0.04;
    g.beginPath(); (g as any).roundRect ? (g as any).roundRect(it.x, it.y, it.w, it.h, r) : g.rect(it.x, it.y, it.w, it.h); g.stroke(); g.restore();
    return;
  }
  g.save();
  let size = it.size;
  const font = () => `${it.italic ? 'italic ' : ''}${it.weight || 400} ${size}px ${FONTS[it.font]}`;
  const text = it.caps ? it.text.toUpperCase() : it.text;
  g.font = font();
  if (it.caps && 'letterSpacing' in g) (g as any).letterSpacing = `${(size * 0.14).toFixed(1)}px`;
  if (it.maxW) { const w = g.measureText(text).width; if (w > it.maxW) { size *= it.maxW / w; g.font = font(); } }
  g.fillStyle = role(it.role); g.textAlign = it.align; g.textBaseline = 'alphabetic';
  g.fillText(text, it.x, it.y);
  g.restore();
}

export function canvasPng(c: HTMLCanvasElement): Promise<Uint8Array> {
  return new Promise((resolve, reject) => c.toBlob((b) => {
    if (!b) { reject(new Error('The sheet could not be encoded as PNG.')); return; }
    b.arrayBuffer().then((a) => resolve(new Uint8Array(a)), reject);
  }, 'image/png'));
}

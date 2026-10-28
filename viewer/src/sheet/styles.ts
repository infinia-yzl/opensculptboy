// OpenSculptBoy
// Apache License, Version 2.0
//
// The styles of a character sheet: each one renders a picture of the figure for a camera frame. The textured style
// draws the page's own look (the current renderer, with a card's look when a card is open). The list is the seam for
// the other styles of the design (toon, ink, flat colour, clay, silhouette): each adds an entry with its own render.

import type { SheetHost, SheetView } from './host.ts';

export interface SheetStyle {
  id: string;
  label: string;
  tip: string;
  render(host: SheetHost, view: SheetView, samples: number, tick: (pixels: number) => Promise<void>): Promise<ImageData>;
}

export const STYLES: SheetStyle[] = [
  {
    id: 'textured', label: 'Textured', tip: 'The page\'s look: the skin, the eyes, the hair, the outfits and the page\'s lights',
    render: (host, view, samples, tick) => host.render(view, samples, tick),
  },
];

export function styleById(id: string): SheetStyle { return STYLES.find((s) => s.id === id) || STYLES[0]; }
export function styleIds(): string[] { return STYLES.map((s) => s.id); }

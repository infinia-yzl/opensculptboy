// OpenSculptBoy
// Apache License, Version 2.0
//
// The stone library of the page: the materials that make a character a statue. A look's `skin.stone` names one of
// them (or holds a colour of its own, "#rrggbb"), and `skin.polish` sets how glossy it is (0 rough, 1 polished).
// The renderer (main.ts) turns a stone into the uniforms of the skin and eye shaders (shading.ts, stoneAlbedo and
// stoneLight): a base colour with soft clouds, veins of the vein colour, a roughness from the polish, and for the
// metals a tinted reflection. The interface (src/shell) draws the swatches from the same table.

export interface Stone {
  id: string;
  label: string;
  base: string;       // sRGB
  vein: string;       // sRGB
  veins: number;      // 0 to 1: how much the veins show
  metal: number;      // 0 stone, 1 metal
  wrap: number;       // the light that the stone lets through around its shadow edge (marble's glow)
  rough: [number, number];   // the roughness at polish 0 and at polish 1
  polish: number;     // the polish of the stone as it comes
}

export const STONES: Stone[] = [
  { id: 'carrara', label: 'Carrara', base: '#e9e5de', vein: '#a3a6aa', veins: 0.35, metal: 0, wrap: 0.45, rough: [0.62, 0.2], polish: 0.45 },
  { id: 'travertine', label: 'Travertine', base: '#dcc6a3', vein: '#bda27e', veins: 0.3, metal: 0, wrap: 0.25, rough: [0.78, 0.32], polish: 0.3 },
  { id: 'bardiglio', label: 'Bardiglio', base: '#a9adb0', vein: '#6c7175', veins: 0.45, metal: 0, wrap: 0.3, rough: [0.6, 0.2], polish: 0.5 },
  { id: 'basalt', label: 'Basalt', base: '#45433f', vein: '#2b2a28', veins: 0.2, metal: 0, wrap: 0, rough: [0.75, 0.3], polish: 0.35 },
  { id: 'bronze', label: 'Bronze', base: '#a76d38', vein: '#5e3a1a', veins: 0.12, metal: 1, wrap: 0, rough: [0.55, 0.22], polish: 0.55 },
  { id: 'verdigris', label: 'Verdigris', base: '#77ab95', vein: '#3f6d5c', veins: 0.45, metal: 0, wrap: 0.1, rough: [0.82, 0.45], polish: 0.2 },
];

export const isHexColour = (h: any): h is string => typeof h === 'string' && /^#[0-9a-fA-F]{6}$/.test(h);

// the stone of a look's skin: a stone of the library, a stone of a colour of its own, or null for skin
export function stoneOf(skin: { stone?: string | null } | null | undefined): Stone | null {
  const s = skin?.stone;
  if (!s) return null;
  const known = STONES.find((x) => x.id === s);
  if (known) return known;
  if (isHexColour(s)) return { id: s.toLowerCase(), label: 'Custom stone', base: s.toLowerCase(), vein: s.toLowerCase(), veins: 0.3, metal: 0, wrap: 0.3, rough: [0.65, 0.22], polish: 0.4 };
  return null;
}

// a swatch of the stone: its colour lit from the upper left, as the design's swatches draw it
export function stoneSwatch(st: Stone): string {
  return `radial-gradient(circle at 35% 30%, color-mix(in srgb, ${st.base} 55%, #fff), ${st.base} 60%, color-mix(in srgb, ${st.vein} 70%, #000))`;
}

// the roughness of a stone at a polish
export function stoneRoughness(st: Stone, polish: number): number {
  const p = Math.min(1, Math.max(0, polish));
  return st.rough[0] + (st.rough[1] - st.rough[0]) * p;
}

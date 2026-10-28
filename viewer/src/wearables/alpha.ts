// OpenSculptBoy
// Apache License, Version 2.0
//
// The alpha of a garment's base colour texture, for the cover ray test of masked materials
// (opensculptboy.wearables.cover.masked_solid): a PNG decoder for the alpha channel alone. A PNG with an alpha channel
// (grey and alpha, RGBA) or a palette (with its tRNS chunk, else opaque) gives the alpha of each pixel as 8 bits, as
// Pillow's convert("RGBA") gives it; any other image (JPEG, grey, RGB) gives null, and its material's alpha is its
// factor alone. No DOM: node runs it in the parity test, and a worker can run it.

import type { Garment, Material } from './package.ts';

export interface AlphaMap { width: number; height: number; alpha: Uint8Array }   // rows from the top

const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const CHANNELS: Record<number, number> = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };
// the Adam7 passes: start and step of the columns, then of the rows
const ADAM7 = [[0, 8, 0, 8], [4, 8, 0, 8], [0, 4, 4, 8], [2, 4, 0, 4], [0, 2, 2, 4], [1, 2, 0, 2], [0, 1, 1, 2]];

async function inflate(data: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([data as BlobPart]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function paeth(a: number, b: number, c: number): number {
  const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

// undo the filters of `rows` scanlines of `stride` bytes from `raw` at `offset`; returns the bytes and the next offset
function unfilter(raw: Uint8Array, offset: number, rows: number, stride: number, bpp: number): [Uint8Array, number] {
  const out = new Uint8Array(rows * stride);
  for (let y = 0; y < rows; y++) {
    const type = raw[offset++], row = y * stride, up = row - stride;
    if (type > 4) throw new Error(`The PNG file has an unknown filter ${type}.`);
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? out[row + x - bpp] : 0, b = y > 0 ? out[up + x] : 0, c = x >= bpp && y > 0 ? out[up + x - bpp] : 0;
      const v = raw[offset + x];
      out[row + x] = (type === 0 ? v : type === 1 ? v + a : type === 2 ? v + b : type === 3 ? v + ((a + b) >> 1) : v + paeth(a, b, c)) & 255;
    }
    offset += stride;
  }
  return [out, offset];
}

// the alpha of a PNG file's pixels, or null for an image without an alpha channel or a palette
export async function readAlpha(bytes: Uint8Array): Promise<AlphaMap | null> {
  if (bytes.length < 8 || SIGNATURE.some((x, i) => bytes[i] !== x)) return null;
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let o = 8, width = 0, height = 0, depth = 0, colour = 0, interlace = 0, trns: Uint8Array | null = null;
  const idat: Uint8Array[] = [];
  while (o + 8 <= bytes.length) {
    const len = dv.getUint32(o), type = String.fromCharCode(...bytes.subarray(o + 4, o + 8)), data = bytes.subarray(o + 8, o + 8 + len);
    if (type === 'IHDR') { width = dv.getUint32(o + 8); height = dv.getUint32(o + 12); depth = data[8]; colour = data[9]; interlace = data[12]; }
    else if (type === 'tRNS') trns = data;
    else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    o += 12 + len;
  }
  if (colour !== 3 && colour !== 4 && colour !== 6) return null;
  const all = new Uint8Array(idat.reduce((s, d) => s + d.length, 0));
  idat.reduce((s, d) => (all.set(d, s), s + d.length), 0);
  const raw = await inflate(all), channels = CHANNELS[colour], bits = channels * depth, bpp = Math.max(1, bits >> 3);
  const alpha = new Uint8Array(width * height);
  // the alpha of pixel x of an unfiltered scanline
  const sample = (line: Uint8Array, start: number, x: number): number => {
    if (colour === 3) {
      const bit = x * depth, index = (line[start + (bit >> 3)] >> (8 - depth - (bit & 7))) & ((1 << depth) - 1);
      return trns && index < trns.length ? trns[index] : 255;
    }
    // the last channel; of 16 bits, its high byte
    return line[start + ((x * bits) >> 3) + (channels - 1) * (depth >> 3)];
  };
  const passes = interlace ? ADAM7 : [[0, 1, 0, 1]];
  let offset = 0;
  for (const [x0, dx, y0, dy] of passes) {
    const w = Math.ceil((width - x0) / dx), h = Math.ceil((height - y0) / dy);
    if (w <= 0 || h <= 0) continue;
    const stride = Math.ceil((w * bits) / 8);
    const [lines, next] = unfilter(raw, offset, h, stride, bpp);
    offset = next;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) alpha[(y0 + y * dy) * width + x0 + x * dx] = sample(lines, y * stride, x);
  }
  return { width, height, alpha };
}

// the alpha maps of a garment's masked materials with a base colour texture (Material.alpha)
export async function readAlphaMaps(garment: Garment): Promise<void> {
  for (const m of garment.materials) {
    const texture = m.textures.base_color;
    if (m.alphaMode === 'MASK' && texture) m.alpha = await readAlpha(texture.bytes);
  }
}

// whether the alpha of a masked material reaches its cutoff at the corners, the edge midpoints and the centre of
// each triangle of the garment that has this material (opensculptboy.wearables.cover.masked_solid), so that no cut-out
// of the texture (lace, a frayed hem) lies on it; one flag per garment triangle (0 for the other materials)
export function maskedSolid(garment: Garment, material: number): Uint8Array {
  const mt: Material = garment.materials[material], T = garment.triangles, TM = garment.triangleMaterial, m = T.length / 3;
  const out = new Uint8Array(m), factor = mt.baseColor.length > 3 ? mt.baseColor[3] : 1, map = mt.textures.base_color ? mt.alpha : null;
  if (!map) {
    const solid = factor >= mt.alphaCutoff ? 1 : 0;
    for (let t = 0; t < m; t++) if (TM[t] === material) out[t] = solid;
    return out;
  }
  const uv = garment.uvs;
  if (!uv) return out;
  const { width: w, height: h, alpha } = map;
  const mod1 = (x: number) => { const r = x % 1; return r !== 0 && r < 0 ? r + 1 : r; };
  const solidAt = (u: number, v: number) => {
    const x = Math.min(Math.max(Math.trunc(mod1(u) * w), 0), w - 1), y = Math.min(Math.max(Math.trunc(mod1(v) * h), 0), h - 1);
    return (alpha[y * w + x] / 255) * factor >= mt.alphaCutoff;
  };
  for (let t = 0; t < m; t++) {
    if (TM[t] !== material) continue;
    const a = T[t * 3] * 2, b = T[t * 3 + 1] * 2, c = T[t * 3 + 2] * 2;
    const us = [uv[a], uv[b], uv[c]], vs = [uv[a + 1], uv[b + 1], uv[c + 1]];
    let solid = true;
    for (let k = 0; k < 3 && solid; k++) solid = solidAt(us[k], vs[k]) && solidAt(0.5 * (us[k] + us[(k + 1) % 3]), 0.5 * (vs[k] + vs[(k + 1) % 3]));
    out[t] = solid && solidAt((us[0] + us[1] + us[2]) / 3, (vs[0] + vs[1] + vs[2]) / 3) ? 1 : 0;
  }
  return out;
}

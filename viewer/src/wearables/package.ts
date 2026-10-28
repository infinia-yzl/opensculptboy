// OpenSculptBoy
// Apache License, Version 2.0
//
// Wearable packages (opensculptboy.wearable/1, docs/wearables.md) read in the page from the user's own files: the
// description (<name>.wearable.json), the garment (<name>.glb: one mesh whose primitives share one set of vertices,
// one primitive per group and material) and the binding (<name>.binding.safetensors). The reader repeats
// opensculptboy.wearables.format.read_wearable. Nothing here touches the DOM or three.js, so node and a worker run it.

import { readAlphaMaps, type AlphaMap } from './alpha.ts';

export const FORMAT = 'opensculptboy.wearable/1';

export interface Material {
  name: string;
  baseColor: number[];     // linear RGBA
  metallic: number; roughness: number; emissive: number[];
  alphaMode: string; alphaCutoff: number; doubleSided: boolean; normalScale: number;
  textures: Record<string, { bytes: Uint8Array; mime: string }>;   // base_color, metallic_roughness, normal, occlusion, emissive
  alpha?: AlphaMap | null;   // the alpha of the base colour texture of a masked material (wearables/alpha.ts)
}

export interface Garment {
  count: number;
  positions: Float64Array;   // the binding body's garment in Anny's frame (metres, Z up)
  normals: Float64Array;
  uvs: Float32Array | null;  // glTF convention, v = 0 at the top
  triangles: Int32Array;
  triangleGroup: Int32Array;
  triangleMaterial: Int32Array;
  groups: string[];
  materials: Material[];
}

export interface Binding {
  method: 'surface' | 'rigid';
  body: string;
  bodyVertices: Float64Array;    // the rest vertices of the binding body (Anny's rows, Anny's frame)
  corners?: Int32Array; barycentric?: Float64Array; offset?: Float64Array; normal?: Float64Array;
  bone?: string; local?: Float64Array;
  skinWeights?: Float64Array; skinJoints?: Int32Array; skinK?: number; jointNames?: string[];
}

export interface Wearable {
  name: string; label: string; slot: string; layer: number; provenance: string; licence: Record<string, any>;
  weights: 'body' | 'rigid' | 'own';
  covers: Uint8Array | null; groupCovers: Record<string, Uint8Array>; coverRay: boolean;
  hides: { piece: string; groups: string[] }[];
  garment: Garment; binding: Binding;
}

// ------------------------------------------------------------------ safetensors
const DTYPES: Record<string, [number, (b: ArrayBuffer) => ArrayLike<number> | bigint[]]> = {
  F64: [8, (b) => new Float64Array(b)], F32: [4, (b) => new Float32Array(b)], F16: [2, (b) => halfs(new Uint16Array(b))],
  I64: [8, (b) => Array.from(new BigInt64Array(b))], I32: [4, (b) => new Int32Array(b)], I16: [2, (b) => new Int16Array(b)],
  I8: [1, (b) => new Int8Array(b)], U8: [1, (b) => new Uint8Array(b)], BOOL: [1, (b) => new Uint8Array(b)],
  U16: [2, (b) => new Uint16Array(b)], U32: [4, (b) => new Uint32Array(b)],
};
function halfs(h: Uint16Array): Float32Array {
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) {
    const x = h[i], s = x & 0x8000 ? -1 : 1, e = (x >> 10) & 31, f = x & 1023;
    out[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}
export interface Tensor { dtype: string; shape: number[]; values: ArrayLike<number> }
export function readSafetensors(bytes: Uint8Array): { tensors: Record<string, Tensor>; metadata: Record<string, string> } {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const n = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + n)));
  const base = 8 + n, tensors: Record<string, Tensor> = {};
  for (const [key, info] of Object.entries<any>(header)) {
    if (key === '__metadata__') continue;
    const kind = DTYPES[info.dtype];
    if (!kind) throw new Error(`Unsupported tensor type ${info.dtype} (${key}).`);
    const [a, b] = info.data_offsets;
    const raw = bytes.slice(base + a, base + b).buffer;   // a copy: the offsets need not be aligned
    const v = kind[1](raw);
    tensors[key] = { dtype: info.dtype, shape: info.shape, values: Array.isArray(v) ? (v as bigint[]).map(Number) : v as ArrayLike<number> };
  }
  return { tensors, metadata: header.__metadata__ || {} };
}
const f64 = (t?: Tensor) => (t ? Float64Array.from(t.values) : undefined);
const i32 = (t?: Tensor) => (t ? Int32Array.from(t.values) : undefined);

// ------------------------------------------------------------------ GLB
export function readGlb(bytes: Uint8Array): { json: any; bin: Uint8Array } {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('Not a GLB file.');
  let o = 12, json: any = null, bin: Uint8Array = new Uint8Array(0);
  while (o + 8 <= bytes.byteLength) {
    const len = dv.getUint32(o, true), type = dv.getUint32(o + 4, true), data = bytes.subarray(o + 8, o + 8 + len);
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(data));
    else if (type === 0x004e4942) bin = data;
    o += 8 + len;
  }
  if (!json) throw new Error('The GLB file holds no JSON chunk.');
  return { json, bin };
}
const COMPONENTS: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
export function accessor(json: any, bin: Uint8Array, index: number): Float64Array {
  const acc = json.accessors[index], view = json.bufferViews[acc.bufferView], width = COMPONENTS[acc.type];
  const size = ({ 5126: 4, 5125: 4, 5123: 2, 5122: 2, 5121: 1, 5120: 1 } as any)[acc.componentType];
  const stride = view.byteStride || size * width, start = (view.byteOffset || 0) + (acc.byteOffset || 0);
  const dv = new DataView(bin.buffer, bin.byteOffset, bin.byteLength), out = new Float64Array(acc.count * width);
  const get = ({ 5126: (p: number) => dv.getFloat32(p, true), 5125: (p: number) => dv.getUint32(p, true), 5123: (p: number) => dv.getUint16(p, true),
    5122: (p: number) => dv.getInt16(p, true), 5121: (p: number) => dv.getUint8(p), 5120: (p: number) => dv.getInt8(p) } as any)[acc.componentType];
  for (let i = 0; i < acc.count; i++) for (let k = 0; k < width; k++) out[i * width + k] = get(start + i * stride + k * size);
  return out;
}
// glTF axes (Y up) to Anny's (Z up): (x, y, z) -> (x, -z, y)
function gltfToAnny(v: Float64Array): Float64Array {
  const out = new Float64Array(v.length);
  for (let o = 0; o < v.length; o += 3) { out[o] = v[o]; out[o + 1] = -v[o + 2]; out[o + 2] = v[o + 1]; }
  return out;
}
function imageBytes(json: any, bin: Uint8Array, index: number): { bytes: Uint8Array; mime: string } | null {
  const im = json.images?.[index];
  if (!im || im.bufferView === undefined) return null;
  const view = json.bufferViews[im.bufferView], s = view.byteOffset || 0;
  const bytes = bin.slice(s, s + view.byteLength);
  return { bytes, mime: im.mimeType || (bytes[0] === 0xff ? 'image/jpeg' : 'image/png') };
}
export function readMaterials(json: any, bin: Uint8Array): Material[] {
  const textures = json.textures || [];
  const image = (info: any) => {
    if (!info) return null;
    const src = textures[info.index]?.source;
    return src === undefined ? null : imageBytes(json, bin, src);
  };
  return (json.materials || []).map((m: any, k: number) => {
    const pbr = m.pbrMetallicRoughness || {};
    const found: Record<string, any> = { base_color: image(pbr.baseColorTexture), metallic_roughness: image(pbr.metallicRoughnessTexture),
      normal: image(m.normalTexture), occlusion: image(m.occlusionTexture), emissive: image(m.emissiveTexture) };
    return {
      name: m.name || `material${k}`, baseColor: pbr.baseColorFactor || [1, 1, 1, 1], metallic: pbr.metallicFactor ?? 1, roughness: pbr.roughnessFactor ?? 1,
      emissive: m.emissiveFactor || [0, 0, 0], alphaMode: m.alphaMode || 'OPAQUE', alphaCutoff: m.alphaCutoff ?? 0.5, doubleSided: !!m.doubleSided,
      normalScale: m.normalTexture?.scale ?? 1, textures: Object.fromEntries(Object.entries(found).filter(([, v]) => v)),
    };
  });
}
export function readGarment(bytes: Uint8Array): Garment {
  const { json, bin } = readGlb(bytes);
  const mesh = json.meshes[0], first = mesh.primitives[0].attributes;
  const positions = gltfToAnny(accessor(json, bin, first.POSITION)), normals = gltfToAnny(accessor(json, bin, first.NORMAL));
  const uvs = first.TEXCOORD_0 !== undefined ? Float32Array.from(accessor(json, bin, first.TEXCOORD_0)) : null;
  const materials = readMaterials(json, bin);
  const groups: string[] = [], tris: number[][] = [], tg: number[] = [], tm: number[] = [];
  for (const p of mesh.primitives) {
    const group = p.extras?.group ?? 'main';
    if (!groups.includes(group)) groups.push(group);
    const t = accessor(json, bin, p.indices);
    tris.push(Array.from(t));
    for (let i = 0; i < t.length / 3; i++) { tg.push(groups.indexOf(group)); tm.push(p.material ?? 0); }
  }
  return { count: positions.length / 3, positions, normals, uvs, triangles: Int32Array.from(tris.flat()), triangleGroup: Int32Array.from(tg),
    triangleMaterial: Int32Array.from(tm), groups: groups.length ? groups : ['main'], materials: materials.length ? materials : [defaultMaterial()] };
}
function defaultMaterial(): Material {
  return { name: 'garment', baseColor: [0.8, 0.8, 0.8, 1], metallic: 0, roughness: 0.8, emissive: [0, 0, 0], alphaMode: 'OPAQUE', alphaCutoff: 0.5,
    doubleSided: false, normalScale: 1, textures: {} };
}

// ------------------------------------------------------------------ the package
// a file of the package by its name, relative to the folder of the description
export type FileSource = (relative: string) => Promise<Uint8Array>;

export async function readWearable(json: any, file: FileSource): Promise<Wearable> {
  if (json.format !== FORMAT) throw new Error(`The format is ${JSON.stringify(json.format)}; this page reads ${FORMAT}.`);
  const info = json.binding || {}, stem = json.name;
  const { tensors } = readSafetensors(await file(info.data || `${stem}.binding.safetensors`));
  const sw = tensors.skin_weights;
  const binding: Binding = {
    method: info.method || 'surface', body: info.body || 'anny-template', bodyVertices: f64(tensors.body_vertices)!,
    corners: i32(tensors.corners), barycentric: f64(tensors.barycentric), offset: f64(tensors.offset), normal: f64(tensors.normal),
    bone: info.bone, local: f64(tensors.local), skinWeights: f64(sw), skinJoints: i32(tensors.skin_joints), skinK: sw ? sw.shape[1] : 0,
    jointNames: info.joints,
  };
  const garment = readGarment(await file(json.geometry || `${stem}.glb`));
  await readAlphaMaps(garment);
  const groups: string[] | undefined = json.groups;
  if (groups && groups.length >= garment.groups.length) {
    const index = garment.groups.map((g) => groups.indexOf(g));
    garment.triangleGroup = garment.triangleGroup.map((g) => index[g]);
    garment.groups = [...groups];
  }
  const mask = (t?: Tensor) => (t ? Uint8Array.from(t.values, (x) => (x ? 1 : 0)) : null);
  const groupCovers: Record<string, Uint8Array> = {};
  for (const [k, t] of Object.entries(tensors)) if (k.startsWith('covers:')) groupCovers[k.slice(7)] = mask(t)!;
  const count = binding.method === 'surface' ? (binding.corners?.length || 0) / 3 : (binding.local?.length || 0) / 3;
  if (count !== garment.count) throw new Error(`The binding holds ${count} vertices and the garment ${garment.count}.`);
  return {
    name: json.name, label: json.label || json.name, slot: json.slot, layer: Number(json.layer), provenance: json.provenance || 'public',
    licence: json.licence || {}, weights: json.weights || 'body', covers: mask(tensors.covers), groupCovers, coverRay: !!json.cover_ray,
    hides: json.hides || [], garment, binding,
  };
}

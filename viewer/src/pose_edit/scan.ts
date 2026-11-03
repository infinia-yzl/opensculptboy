// OpenSculptBoy
// Apache License, Version 2.0
//
// The light copy of a statue's scan that a starter carries beside its card (`scan.glb`, the card's entry
// `detail.fit.reference.ghost`): one mesh of one primitive, its positions quantised to 16 bits and the node's scale
// and translation turning them back into metres, compressed with EXT_meshopt_compression, in glTF's axes (y up) of the
// card's posed body. The page reads it into Anny's axes of the card frame (z up), the frame of the saved reference
// points, and computes smooth normals itself.

// the page's meshopt decoder (src/meshopt_ref.js, a script of the page that sets window.MeshoptDecoderRef)
const decoder = () => (globalThis as any).MeshoptDecoderRef;

export interface ScanMesh { positions: Float32Array; indices: Uint32Array; normals: Float32Array; bytes: number }

const SIZES: Record<number, number> = { 5126: 4, 5125: 4, 5123: 2, 5122: 2, 5121: 1, 5120: 1 };
const WIDTH: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

function glbChunks(bytes: Uint8Array): { json: any; bin: Uint8Array } {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength < 20 || dv.getUint32(0, true) !== 0x46546c67) throw new Error('The scan is not a GLB file.');
  let o = 12, json: any = null, bin: Uint8Array = new Uint8Array(0);
  while (o + 8 <= bytes.byteLength) {
    const len = dv.getUint32(o, true), type = dv.getUint32(o + 4, true), data = bytes.subarray(o + 8, o + 8 + len);
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(data));
    else if (type === 0x004e4942) bin = data;
    o += 8 + len;
  }
  if (!json) throw new Error('The scan holds no JSON chunk.');
  return { json, bin };
}

// a buffer view's bytes, decoded when EXT_meshopt_compression holds them
function viewBytes(json: any, bin: Uint8Array, index: number, cache: Map<number, Uint8Array>): Uint8Array {
  const hit = cache.get(index);
  if (hit) return hit;
  const view = json.bufferViews[index], ext = view.extensions?.EXT_meshopt_compression;
  let out: Uint8Array;
  if (ext) {
    const src = bin.subarray(ext.byteOffset || 0, (ext.byteOffset || 0) + ext.byteLength);
    out = new Uint8Array(ext.count * ext.byteStride);
    const d = decoder();
    if (!d) throw new Error('The page has no meshopt decoder.');
    d.decodeGltfBuffer(out, ext.count, ext.byteStride, src, ext.mode, ext.filter || 'NONE');
  } else out = bin.subarray(view.byteOffset || 0, (view.byteOffset || 0) + view.byteLength);
  cache.set(index, out);
  return out;
}

function readAccessor(json: any, bin: Uint8Array, index: number, cache: Map<number, Uint8Array>): Float64Array {
  const acc = json.accessors[index], view = json.bufferViews[acc.bufferView], width = WIDTH[acc.type], size = SIZES[acc.componentType];
  const ext = view.extensions?.EXT_meshopt_compression;
  const data = viewBytes(json, bin, acc.bufferView, cache);
  const stride = (ext ? ext.byteStride : view.byteStride) || size * width, start = acc.byteOffset || 0;
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength), out = new Float64Array(acc.count * width);
  const get = ({ 5126: (p: number) => dv.getFloat32(p, true), 5125: (p: number) => dv.getUint32(p, true), 5123: (p: number) => dv.getUint16(p, true),
    5122: (p: number) => dv.getInt16(p, true), 5121: (p: number) => dv.getUint8(p), 5120: (p: number) => dv.getInt8(p) } as any)[acc.componentType];
  for (let i = 0; i < acc.count; i++) for (let k = 0; k < width; k++) out[i * width + k] = get(start + i * stride + k * size);
  if (acc.normalized) {
    const div = ({ 5123: 65535, 5122: 32767, 5121: 255, 5120: 127 } as any)[acc.componentType] || 1;
    for (let i = 0; i < out.length; i++) out[i] = acc.componentType === 5122 || acc.componentType === 5120 ? Math.max(out[i] / div, -1) : out[i] / div;
  }
  return out;
}

// the node of the mesh: x -> T + R (S x) (rotation as a quaternion x, y, z, w)
function nodeMap(node: any): (x: number, y: number, z: number, out: number[]) => void {
  const s = node?.scale || [1, 1, 1], t = node?.translation || [0, 0, 0], q = node?.rotation || [0, 0, 0, 1];
  const [qx, qy, qz, qw] = q;
  const R = [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
    2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
    2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)];
  return (x, y, z, out) => {
    const a = x * s[0], b = y * s[1], c = z * s[2];
    out[0] = R[0] * a + R[1] * b + R[2] * c + t[0]; out[1] = R[3] * a + R[4] * b + R[5] * c + t[1]; out[2] = R[6] * a + R[7] * b + R[8] * c + t[2];
  };
}

// the scan in Anny's axes of the card frame: glTF (x, y, z) -> Anny (x, -z, y)
export function readScan(bytes: Uint8Array): ScanMesh {
  const { json, bin } = glbChunks(bytes);
  const used = json.extensionsRequired || [];
  for (const e of used) if (e !== 'EXT_meshopt_compression' && e !== 'KHR_mesh_quantization') throw new Error(`The scan needs ${e}, which the page does not read.`);
  const nodeIndex = (json.nodes || []).findIndex((n: any) => n.mesh !== undefined);
  if (nodeIndex < 0) throw new Error('The scan holds no mesh.');
  const node = json.nodes[nodeIndex], prim = json.meshes[node.mesh].primitives[0];
  if (prim.mode !== undefined && prim.mode !== 4) throw new Error('The scan is not made of triangles.');
  const cache = new Map<number, Uint8Array>();
  const q = readAccessor(json, bin, prim.attributes.POSITION, cache), n = q.length / 3;
  const map = nodeMap(node), positions = new Float32Array(n * 3), p = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    map(q[i * 3], q[i * 3 + 1], q[i * 3 + 2], p);
    positions[i * 3] = p[0]; positions[i * 3 + 1] = -p[2]; positions[i * 3 + 2] = p[1];
  }
  let indices: Uint32Array;
  if (prim.indices !== undefined) indices = Uint32Array.from(readAccessor(json, bin, prim.indices, cache));
  else { indices = new Uint32Array(n); for (let i = 0; i < n; i++) indices[i] = i; }
  return { positions, indices, normals: smoothNormals(positions, indices), bytes: bytes.byteLength };
}

// area-weighted vertex normals
export function smoothNormals(P: Float32Array, I: Uint32Array): Float32Array {
  const N = new Float32Array(P.length);
  for (let t = 0; t < I.length; t += 3) {
    const a = I[t] * 3, b = I[t + 1] * 3, c = I[t + 2] * 3;
    const e1x = P[b] - P[a], e1y = P[b + 1] - P[a + 1], e1z = P[b + 2] - P[a + 2];
    const e2x = P[c] - P[a], e2y = P[c + 1] - P[a + 1], e2z = P[c + 2] - P[a + 2];
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    for (const v of [a, b, c]) { N[v] += nx; N[v + 1] += ny; N[v + 2] += nz; }
  }
  for (let o = 0; o < N.length; o += 3) {
    const l = Math.hypot(N[o], N[o + 1], N[o + 2]) || 1;
    N[o] /= l; N[o + 1] /= l; N[o + 2] /= l;
  }
  return N;
}

// the card's entry of the ghost scan (detail.fit.reference.ghost), or null
export interface GhostEntry { file: string; frame: string; triangles?: number; vertices?: number; bytes?: number; licence?: string; credit?: string; source?: string }
export function ghostOf(card: any): GhostEntry | null {
  const g = card?.detail?.fit?.reference?.ghost;
  return g && typeof g.file === 'string' ? g : null;
}

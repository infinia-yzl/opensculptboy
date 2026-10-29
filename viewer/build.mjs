// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// Build the viewer page: encode the data of anny.viewer (build/*.raw, build/manifest.json) with meshopt and
// gzip, bundle the TypeScript with three.js, and write one HTML file (dist/anny_viewer.html).
// Ported from the legacy web/encode.mjs and web/build_page.py.
//
//     python -m anny.viewer build --data && npm run build
//     node build.mjs --reuse-data     # the page alone, with the model data of the last build

import { MeshoptEncoder } from 'meshoptimizer';
import * as esbuild from 'esbuild';
import fs from 'fs';
import path from 'path';
import zlib from 'zlib';
import crypto from 'crypto';
import { fileURLToPath } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dir = path.join(here, 'build');
const out = path.join(here, 'dist', 'anny_viewer.html');
// the model data: gzip of the container of every buffer
async function encodeData() {
  await MeshoptEncoder.ready;
  const man = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json')));
  const read = (name) => new Uint8Array(fs.readFileSync(path.join(dir, name + '.raw')));
  const META_KEYS = ['eyes', 'eye_mesh', 'lut', 'skin', 'rig', 'motion', 'correctives', 'shape', 'stool', 'authoring', 'hair', 'expressions'];
  const header = { buffers: [], meta: Object.fromEntries(META_KEYS.map((k) => [k, man[k]])) };
  const parts = [];
  let offset = 0;
  function push(entry, bytes) {
    const pad = (4 - (bytes.length % 4)) % 4;
    entry.byteOffset = offset; entry.byteLength = bytes.length;
    parts.push(Buffer.from(bytes)); if (pad) parts.push(Buffer.alloc(pad));
    offset += bytes.length + pad;
    header.buffers.push(entry);
  }
  const bufs = Object.fromEntries(man.buffers.map((b) => [b.name, b]));
  const COMPANIONS = ['_row', '_detail', '_skin', '_uv'];
  const REMAP = {};
  // the vertices that the page moves on every frame of a clip: the records of these sparse buffers (the corrective
  // shapes), then the vertices of the faces. They go first in the vertex order, each set after the one before, so the
  // upload of a frame or of a face is one small range.
  const HOT = { head: ['corr', 'face_region'] };

  // meshes: '<name>_v' with '<name>_i' are reordered for the vertex cache; the companion streams follow
  function encodeMesh(prefix) {
    const hv = bufs[prefix + '_v'];
    const vdata = read(prefix + '_v');
    const idx = new Uint32Array(read(prefix + '_i').buffer.slice(0));
    const [remap, unique] = MeshoptEncoder.reorderMesh(idx, true, false);
    // a stable partition of the optimised order: the hot vertices of each set in turn, then the others
    const sets = HOT[prefix] || [], hot = new Uint8Array(unique).fill(255);
    sets.forEach((name, k) => {
      const b = bufs[name];
      if (!b) return;
      const dv = new DataView(read(name).buffer.slice(0));
      for (let i = 0; i < b.count; i++) { const r = remap[dv.getUint32(i * b.stride, true)]; if (hot[r] === 255) hot[r] = k; }
    });
    const perm = new Uint32Array(unique);
    let nHot = 0;
    for (let k = 0; k < sets.length; k++) for (let v = 0; v < unique; v++) if (hot[v] === k) perm[v] = nHot++;
    for (let v = 0, k = nHot; v < unique; v++) if (hot[v] === 255) perm[v] = k++;
    for (let i = 0; i < remap.length; i++) if (remap[i] !== 0xffffffff) remap[i] = perm[remap[i]];
    for (let i = 0; i < idx.length; i++) idx[i] = perm[idx[i]];
    const nv = new Uint8Array(unique * hv.stride);
    for (let i = 0; i < hv.count; i++) { const r = remap[i]; if (r !== 0xffffffff) nv.set(vdata.subarray(i * hv.stride, (i + 1) * hv.stride), r * hv.stride); }
    push({ name: prefix + '_v', enc: 'mv', count: unique, stride: hv.stride, lo: hv.lo, hi: hv.hi }, MeshoptEncoder.encodeVertexBufferLevel(nv, unique, hv.stride, 3, 1));
    push({ name: prefix + '_i', enc: 'mi', count: idx.length, stride: 4 }, MeshoptEncoder.encodeIndexBuffer(new Uint8Array(idx.buffer), idx.length, 4));
    header.meta[prefix + '_count'] = unique;
    REMAP[prefix] = remap;
    for (const suffix of COMPANIONS) {
      const mb = bufs[prefix + suffix];
      if (!mb) continue;
      const md = read(prefix + suffix);
      const nm = new Uint8Array(unique * mb.stride);
      for (let i = 0; i < mb.count; i++) { const r = remap[i]; if (r !== 0xffffffff) nm.set(md.subarray(i * mb.stride, (i + 1) * mb.stride), r * mb.stride); }
      const entry = Object.assign({}, mb, { name: prefix + suffix, enc: 'mv', count: unique });
      delete entry.kind;
      push(entry, MeshoptEncoder.encodeVertexBufferLevel(nm, unique, mb.stride, 3, 1));
    }
    console.log(prefix, unique, 'vertices,', idx.length / 3, 'triangles,', nHot, 'moved on every frame');
  }
  const meshes = man.buffers.filter((b) => b.name.endsWith('_v') && bufs[b.name.slice(0, -2) + '_i']).map((b) => b.name.slice(0, -2));
  meshes.forEach(encodeMesh);

  // records that name vertices of a mesh: the vertex fields follow the new vertex order; the records of each
  // corrective shape are sorted by vertex again (sort: 'shapes'), the strand bindings keep their order
  function encodeSparse(b) {
    const br = REMAP[b.body];
    const data = read(b.name);
    const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
    for (let i = 0; i < b.count; i++) for (let f = 0; f < (b.fields || 1); f++) {
      const r = br[dv.getUint32(i * b.stride + f * 4, true)];
      if (r === 0xffffffff) throw new Error('record on an unused vertex: ' + b.name);
      dv.setUint32(i * b.stride + f * 4, r, true);
    }
    let outb = data.subarray(0, b.count * b.stride);
    if (b.sort === 'shapes') {
      outb = new Uint8Array(b.count * b.stride);
      for (const s of man.correctives.shapes) {
        const order = Array.from({ length: s.count }, (_, k) => s.start + k).sort((x, y) => dv.getUint32(x * b.stride, true) - dv.getUint32(y * b.stride, true));
        order.forEach((src, k) => outb.set(data.subarray(src * b.stride, (src + 1) * b.stride), (s.start + k) * b.stride));
      }
    }
    const entry = Object.assign({ enc: 'mv' }, b); delete entry.kind;
    push(entry, MeshoptEncoder.encodeVertexBufferLevel(outb, b.count, b.stride, 3, 1));
  }
  for (const b of man.buffers) {
    if (b.kind === 'sparse') { encodeSparse(b); continue; }
    if (meshes.some((p) => b.name === p + '_v' || b.name === p + '_i' || COMPANIONS.some((s) => b.name === p + s))) continue;
    const data = read(b.name);
    if (b.kind === 'vertex') push(Object.assign({ enc: 'mv' }, b), MeshoptEncoder.encodeVertexBufferLevel(data, b.count, b.stride, 3, 1));
    else push(Object.assign({ enc: 'raw' }, b), data);
  }
  const hjson = Buffer.from(JSON.stringify(header));
  const hl = Buffer.alloc(8); hl.write('HDB1', 0); hl.writeUInt32LE(hjson.length, 4);
  const blob = Buffer.concat([hl, hjson, Buffer.alloc((4 - (hjson.length % 4)) % 4), ...parts]);
  const gz = zlib.gzipSync(blob, { level: 9 });
  console.log('data', (blob.length / 1e6).toFixed(1), 'MB, gzip', (gz.length / 1e6).toFixed(1), 'MB');
  return gz;
}

// --reuse-data: the model data of the page built before (dist/anny_viewer.html), for changes to the page alone when
// the data stage has not run here
function reusedData() {
  const old = fs.readFileSync(out, 'utf8');
  const m = old.match(/window\.MODEL_B64 = "([A-Za-z0-9+/=]+)"/);
  if (!m) throw new Error('no model data in ' + out);
  console.log('data reused from', out);
  return Buffer.from(m[1], 'base64');
}
const gz = process.argv.includes('--reuse-data') ? reusedData() : await encodeData();

// MediaPipe for "From a picture": the WebAssembly loader and binary of the npm package @mediapipe/tasks-vision 1.0.1
// and the pose and hand models, as files beside the page (dist/mediapipe/, in git: they change only with the pinned
// version). Each file is pinned here by SHA-256; the page fetches each one from its own origin with that hash as its
// integrity, so the browser refuses any other bytes, and the policy allows the loader script by the same hash. The
// models come from dist/mediapipe/ or build/mediapipe/ when they are there, else from Google's model store.
const MEDIAPIPE = {
  'vision_wasm_internal.js': { sha256: 'e170ee67dd4e16c1a6fcd8840a206687e5a59b22c20e4a902bc445b095454d73', npm: true },
  'vision_wasm_internal.wasm': { sha256: '8da277a733926eacd0474b8704b36742d6ec3231c57a860c5b889dff8f1df886', npm: true },
  'pose_landmarker_full.task': { sha256: '4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad',
    url: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task' },
  'hand_landmarker.task': { sha256: 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1',
    url: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' },
};
const sha256 = (b) => crypto.createHash('sha256').update(b).digest('hex');
const mpDir = path.join(path.dirname(out), 'mediapipe');
async function mediapipeFiles() {
  fs.mkdirSync(mpDir, { recursive: true });
  const sri = {};
  for (const [name, pin] of Object.entries(MEDIAPIPE)) {
    const target = path.join(mpDir, name);
    const candidates = [target, path.join(dir, 'mediapipe', name)];
    if (pin.npm) candidates.unshift(path.join(here, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm', name));
    let bytes = null;
    for (const c of candidates) if (fs.existsSync(c)) { const b = fs.readFileSync(c); if (sha256(b) === pin.sha256) { bytes = b; break; } }
    if (!bytes && pin.url) {
      const r = await fetch(pin.url);
      if (!r.ok) throw new Error(`MediaPipe model ${name}: ${r.status} from ${pin.url}`);
      bytes = Buffer.from(await r.arrayBuffer());
    }
    if (!bytes || sha256(bytes) !== pin.sha256) throw new Error(`MediaPipe file ${name}: no copy with the pinned SHA-256 ${pin.sha256}`);
    if (!fs.existsSync(target) || !fs.readFileSync(target).equals(bytes)) fs.writeFileSync(target, bytes);
    sri[name] = 'sha256-' + Buffer.from(pin.sha256, 'hex').toString('base64');
  }
  console.log('mediapipe', Object.keys(MEDIAPIPE).length, 'files in', mpDir);
  return sri;
}
const mpSri = await mediapipeFiles();


// the body worker (src/body_worker.ts): its own bundle, which the page starts from a string
const worker = await esbuild.build({
  entryPoints: [path.join(here, 'src', 'body_worker.ts')],
  bundle: true, format: 'iife', minify: true, write: false, target: 'es2022',
  legalComments: 'none',
});
// the worker of the wearables' query pool (src/wearables/query_worker.ts), started the same way
const queryWorker = await esbuild.build({
  entryPoints: [path.join(here, 'src', 'wearables', 'query_worker.ts')],
  bundle: true, format: 'iife', minify: true, write: false, target: 'es2022',
  legalComments: 'none',
});
// MediaPipe's task code sends usage records (task, platform, timings) to Google every minute
// (odml.pa.googleapis.com/v1/log). The page keeps everything on the computer, so the build cuts that one call out of
// the bundle, and the policy would block it anyway. The build stops when the pattern is gone (a new MediaPipe).
const TELEMETRY = 'this.m.send({url:"https://odml.pa.googleapis.com/v1/log"';
const noTelemetry = {
  name: 'no-mediapipe-telemetry',
  setup(b) {
    b.onLoad({ filter: /tasks-vision[\\/]vision_bundle\.mjs$/ }, (args) => {
      const code = fs.readFileSync(args.path, 'utf8');
      if (code.split(TELEMETRY).length !== 2) throw new Error('MediaPipe\'s usage logging call was not found in ' + args.path);
      return { contents: code.replace(TELEMETRY, '((o, done) => done())({url:"about:blank"'), loader: 'js' };
    });
  },
};
// ------------------------------------------------------------------ the interface's styles
// The page carries its styles inline and loads nothing from the network: the fonts as data URLs, the design system's
// tokens (src/shell/design/tokens.json) as CSS variables for both themes, Tailwind's output for the classes that the
// interface uses (src/shell/design/tailwind.css, compiled with tailwindcss's own compile(), which has no native parts),
// the design system's components (osb.css) and the page's own rules (page.css).
async function interfaceCss() {
  const D = path.join(here, 'src', 'shell', 'design');
  const nm = (...p) => path.join(here, 'node_modules', ...p);
  const b64 = (f) => fs.readFileSync(f).toString('base64');
  // Cinzel, Instrument Sans and Instrument Serif (SIL Open Font License 1.1), the Latin subset
  const face = (family, file, weight, style = 'normal') => `@font-face { font-family: "${family}"; font-style: ${style}; font-weight: ${weight}; font-display: block;`
    + ` src: url(data:font/woff2;base64,${b64(file)}) format("woff2"); unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD; }`;
  const fonts = [
    '/* Cinzel (c) Natanael Gama; Instrument Sans and Instrument Serif (c) The Instrument Sans and Instrument Serif Project Authors. SIL Open Font License 1.1 (https://openfontlicense.org). */',
    face('Cinzel', nm('@fontsource-variable', 'cinzel', 'files', 'cinzel-latin-wght-normal.woff2'), '400 900'),
    face('Instrument Sans', nm('@fontsource-variable', 'instrument-sans', 'files', 'instrument-sans-latin-wght-normal.woff2'), '400 700'),
    face('Instrument Serif', nm('@fontsource', 'instrument-serif', 'files', 'instrument-serif-latin-400-italic.woff2'), '400', 'italic'),
    face('Instrument Serif', nm('@fontsource', 'instrument-serif', 'files', 'instrument-serif-latin-400-normal.woff2'), '400'),
  ].join('\n');

  // the tokens: every family but type is a list of {name, value}; a value is a string or one value per theme
  const T = JSON.parse(fs.readFileSync(path.join(D, 'tokens.json'), 'utf8'));
  const themes = T.color.themes.map((t) => t.id);
  const val = (v) => typeof v === 'string' ? v.replace(/^\{([\w.-]+)\}$/, 'var(--$1)') : String(v);
  const per = Object.fromEntries(themes.map((t) => [t, []]));
  for (const [fam, f] of Object.entries(T)) {
    if (!f || !Array.isArray(f.tokens)) continue;
    for (const t of f.tokens) {
      if (t.value && typeof t.value === 'object') themes.forEach((th, i) => per[th].push(`--${t.name}: ${val(t.value[th] ?? t.value[themes[0]])};`));
      else per[themes[0]].push(`--${t.name}: ${val(t.value)};`);
    }
  }
  for (const [k, v] of Object.entries(T.type.families)) per[themes[0]].push(`--font-${k}: ${v};`);
  const tokens = `:root { color-scheme: light; ${per[themes[0]].join(' ')} }\n`
    + themes.slice(1).map((th) => `:root[data-theme="${th}"] { color-scheme: ${th}; ${per[th].join(' ')} }`).join('\n');

  // Tailwind: the candidates are the words of the interface's sources, sorted, so the output is the same on every machine
  const { compile } = await import('tailwindcss');
  const tw = nm('tailwindcss');
  const loadStylesheet = async (id, base) => {
    const p = id === 'tailwindcss' ? path.join(tw, 'index.css') : id.startsWith('tailwindcss/') ? path.join(tw, id.slice(12).replace(/(\.css)?$/, '.css')) : path.resolve(base, id);
    return { path: p, base: path.dirname(p), content: fs.readFileSync(p, 'utf8') };
  };
  const compiler = await compile(fs.readFileSync(path.join(D, 'tailwind.css'), 'utf8'), {
    base: D, loadStylesheet, loadModule: async (id) => { throw new Error('no Tailwind plugins in the page: ' + id); },
  });
  const words = new Set();
  const walk = (dir) => {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else if (/\.(tsx?|html)$/.test(e.name)) for (const w of fs.readFileSync(p, 'utf8').split(/[\s"'`{}<>;]+/)) if (w && w.length < 80) words.add(w);
    }
  };
  walk(path.join(here, 'src', 'shell'));
  const twCss = compiler.build([...words].sort());

  const osb = fs.readFileSync(path.join(D, 'osb.css'), 'utf8');
  const own = fs.readFileSync(path.join(D, 'page.css'), 'utf8');
  return [fonts, twCss, tokens, osb, own].join('\n');
}

// the script: the TypeScript with three.js and the interface (React, src/shell/) in one bundle
const bundle = await esbuild.build({
  entryPoints: [path.join(here, 'src', 'main.ts')],
  bundle: true, format: 'esm', minify: true, write: false, target: 'es2022',
  alias: { 'three/addons': 'three/examples/jsm' },
  plugins: [noTelemetry],
  jsx: 'automatic',
  define: {
    __BODY_WORKER__: JSON.stringify(worker.outputFiles[0].text), __QUERY_WORKER__: JSON.stringify(queryWorker.outputFiles[0].text),
    __MEDIAPIPE_SRI__: JSON.stringify(mpSri), 'process.env.NODE_ENV': '"production"',
  },
  legalComments: 'none',
});
const app = bundle.outputFiles[0].text;
const ref = fs.readFileSync(path.join(here, 'src', 'meshopt_ref.js'), 'utf8');
const css = await interfaceCss();
// the shell with the interface's styles (in place of its DESIGN slot) and with the hashes of the page's scripts in its
// Content Security Policy: the scripts that the build adds, and the shell's own inline scripts (the early theme)
const hash = (text) => `'sha256-${crypto.createHash('sha256').update(text, 'utf8').digest('base64')}'`;
function shellWith(scripts) {
  const shell = fs.readFileSync(path.join(here, 'shell.html'), 'utf8');
  if (!shell.includes('script-src SCRIPT_HASHES') || !shell.includes('<!-- DESIGN:')) throw new Error('shell.html lacks SCRIPT_HASHES or the DESIGN slot');
  const own = [...shell.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
  const loader = `'${mpSri['vision_wasm_internal.js']}'`;
  return shell.replace('script-src SCRIPT_HASHES', 'script-src ' + [...own, ...scripts].map(hash).join(' ') + ' ' + loader)
    .replace(/<!-- DESIGN:[^>]*-->/, () => '<style id="osb-design">\n' + css + '\n</style>');
}
const appText = '\n' + app.replace(/<\/script/g, '<\\/script') + '\n';
const refText = '\n' + ref + '\n';
const dataText = 'window.MODEL_B64 = "' + gz.toString('base64') + '";';
console.log('script', (app.length / 1e6).toFixed(2), 'MB, interface styles', (css.length / 1e6).toFixed(2), 'MB');
const page = '<!doctype html><html lang="en-GB"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
  + shellWith([refText, dataText, appText]) + '\n<script>' + refText + '</script>\n<script>' + dataText + '</script>\n'
  + '<script type="module">' + appText + '</script>\n</head><body></body></html>\n';
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, page);
// the page's server (serve_viewer.py, docs/viewer.md "AI providers") and its port travel with the page
for (const [from, to] of [['serve_viewer.py', 'serve_viewer.py'], [path.join('src', 'ai', 'serve.json'), 'serve.json']]) fs.copyFileSync(path.join(here, from), path.join(path.dirname(out), to));
console.log('page', out, (page.length / 1e6).toFixed(1), 'MB');

// --parts <dir>: the page and its data as separate files, for hosts that limit the size of a file (such as a Claude
// artifact). The page carries no document tags, since such hosts wrap it in their own; the data goes to text files of
// base64 under 15 MB each, split at multiples of 4 characters, which the page fetches next to itself.
const partsAt = process.argv.indexOf('--parts');
if (partsAt >= 0) {
  const dirOut = path.resolve(here, process.argv[partsAt + 1] || path.join('build', 'artifact'));
  const b64 = gz.toString('base64'), limit = 15e6 - (15e6 % 4);
  const count = Math.ceil(b64.length / limit), size = Math.ceil(b64.length / count / 4) * 4;
  const names = [];
  fs.mkdirSync(path.join(dirOut, 'data'), { recursive: true });
  for (let k = 0; k < count; k++) {
    names.push(`data/model-${k + 1}.txt`);
    fs.writeFileSync(path.join(dirOut, names[k]), b64.slice(k * size, (k + 1) * size));
  }
  fs.cpSync(mpDir, path.join(dirOut, 'mediapipe'), { recursive: true });
  const partsText = 'window.MODEL_PARTS = ' + JSON.stringify(names) + ';';
  const small = shellWith([refText, partsText, appText]) + '\n<script>' + refText + '</script>\n<script>' + partsText + '</script>\n'
    + '<script type="module">' + appText + '</script>\n';
  fs.writeFileSync(path.join(dirOut, 'index.html'), small);
  console.log('parts', dirOut, (small.length / 1e6).toFixed(1), 'MB page,', count, 'data files of', (size / 1e6).toFixed(1), 'MB');
}

// OpenSculptBoy
// Apache License, Version 2.0
//
// MediaPipe's WebAssembly and models inside the page (build.mjs writes them): one gzip container in base64, in a
// script element of type application/octet-stream that the browser never runs, or in data files next to a page in
// parts (window.MEDIAPIPE_PARTS). The page reads them the first time a picture comes in and gives MediaPipe blob URLs,
// so no MediaPipe code or model ever comes from another host.

import { gunzipSync } from 'fflate';

export interface MediapipeAssets { loader: string; wasm: string; models: { pose: Uint8Array; hand: Uint8Array } }

async function base64(): Promise<string> {
  const el = document.getElementById('mediapipe-data');
  if (el && el.textContent) return el.textContent;
  const parts: string[] | undefined = (window as any).MEDIAPIPE_PARTS;
  if (!parts) throw new Error('This page has no MediaPipe data, so From a picture is not available.');
  const texts = await Promise.all(parts.map(async (url) => {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`the data file ${url} did not load (${r.status})`);
    return r.text();
  }));
  return texts.join('');
}

async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream === 'undefined') return gunzipSync(bytes);
  const s = new Blob([bytes as BlobPart]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(s).arrayBuffer());
}

let assets: Promise<MediapipeAssets> | null = null;
export function mediapipeAssets(): Promise<MediapipeAssets> {
  if (!assets) {
    assets = (async () => {
      const bin = atob(await base64());
      const raw = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) raw[i] = bin.charCodeAt(i);
      const blob = await gunzip(raw);
      const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
      if (String.fromCharCode(...blob.subarray(0, 4)) !== 'MPA1') throw new Error('The MediaPipe data of this page is damaged.');
      const hlen = dv.getUint32(4, true);
      const header = JSON.parse(new TextDecoder().decode(blob.subarray(8, 8 + hlen)));
      const base = 8 + hlen + ((4 - (hlen % 4)) % 4);
      const file = (name: string) => {
        const f = header.files.find((x: any) => x.name === name);
        if (!f) throw new Error('The MediaPipe data of this page lacks ' + name + '.');
        return blob.slice(base + f.offset, base + f.offset + f.length);
      };
      const url = (name: string, type: string) => URL.createObjectURL(new Blob([file(name) as BlobPart], { type }));
      return {
        loader: url('vision_wasm_internal.js', 'text/javascript'),
        wasm: url('vision_wasm_internal.wasm', 'application/wasm'),
        models: { pose: file('pose_landmarker_full.task'), hand: file('hand_landmarker.task') },
      };
    })();
    assets.catch(() => { assets = null; });
  }
  return assets;
}

// runs MediaPipe's loader script (a blob URL of the page's own data), which sets self.ModuleFactory
export function runLoader(src: string): Promise<void> {
  return new Promise((ok, bad) => {
    const s = document.createElement('script');
    s.src = src;
    s.onload = () => { s.remove(); ok(); };
    s.onerror = () => { s.remove(); bad(new Error('MediaPipe\'s loader did not run.')); };
    document.head.append(s);
  });
}

// OpenSculptBoy
// Apache License, Version 2.0
//
// MediaPipe's WebAssembly and models as files beside the page (mediapipe/, written by build.mjs, which pins each
// file by SHA-256 and gives the page the hashes as __MEDIAPIPE_SRI__). The page fetches each file from its own origin
// with its hash as the request's integrity, so the browser refuses any other bytes; the loader script runs from a
// script element with the same integrity, which the page's Content Security Policy allows by that hash, or else from
// a blob URL of the checked bytes. A page opened as a file cannot fetch them, so From a picture needs the page served.

declare const __MEDIAPIPE_SRI__: Record<string, string>;

export interface MediapipeAssets { loader: string; wasm: string; models: { pose: Uint8Array; hand: Uint8Array } }

export const NEEDS_SERVER = 'From a picture needs the page served on this computer: run python serve_viewer.py --open in the page\'s folder.';

const sri = (name: string) => {
  const s = typeof __MEDIAPIPE_SRI__ !== 'undefined' ? __MEDIAPIPE_SRI__[name] : undefined;
  if (!s) throw new Error('This page has no MediaPipe files, so From a picture is not available.');
  return s;
};
const url = (name: string) => new URL('mediapipe/' + name, location.href).href;

async function checked(name: string): Promise<Uint8Array> {
  let r: Response;
  try { r = await fetch(url(name), { integrity: sri(name), cache: 'no-store', credentials: 'omit' }); } catch (e) {
    throw new Error('MediaPipe\'s file ' + name + ' did not load, or its bytes do not match the page\'s pinned hash.');
  }
  if (!r.ok) throw new Error('MediaPipe\'s file ' + name + ' did not load (' + r.status + ').');
  return new Uint8Array(await r.arrayBuffer());
}

let assets: Promise<MediapipeAssets> | null = null;
export function mediapipeAssets(): Promise<MediapipeAssets> {
  if (!assets) {
    assets = (async () => {
      if (location.protocol === 'file:') throw new Error(NEEDS_SERVER);
      const [wasm, pose, hand] = await Promise.all(['vision_wasm_internal.wasm', 'pose_landmarker_full.task', 'hand_landmarker.task'].map(checked));
      return {
        loader: url('vision_wasm_internal.js'),
        wasm: URL.createObjectURL(new Blob([wasm as BlobPart], { type: 'application/wasm' })),
        models: { pose, hand },
      };
    })();
    assets.catch(() => { assets = null; });
  }
  return assets;
}

function scriptFrom(src: string, integrity?: string): Promise<boolean> {
  return new Promise((ok) => {
    const s = document.createElement('script');
    if (integrity) s.integrity = integrity;
    s.src = src;
    s.onload = () => { s.remove(); ok(true); };
    s.onerror = () => { s.remove(); ok(false); };
    document.head.append(s);
  });
}

// runs MediaPipe's loader script, which sets self.ModuleFactory: from its file with its integrity, or, in a browser
// whose policy checks only inline scripts by hash, from a blob URL of the bytes that the browser checked on fetch
let fallback: string | null = null;
export async function runLoader(src: string): Promise<void> {
  const name = 'vision_wasm_internal.js';
  if (!fallback && await scriptFrom(src, sri(name))) return;
  if (!fallback) fallback = URL.createObjectURL(new Blob([await checked(name) as BlobPart], { type: 'text/javascript' }));
  if (!await scriptFrom(fallback)) throw new Error('MediaPipe\'s loader did not run.');
}

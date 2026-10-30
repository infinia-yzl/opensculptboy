// OpenSculptBoy
// Apache License, Version 2.0
//
// The starter characters (DESIGN-0012, "Statue starters"): classical statues fitted to public-domain scans, each a
// character card with its sculpted detail and its props (wearables of the slot `prop`: a base, a tree stump, what the
// statue holds) in one package, `<statue>.zip`, made by `python -m opensculptboy.statues starter`. The packages are
// too large for the repository and go out as files of a release; the person puts them in the folder `starters/`
// beside the page, which fetches one when the person picks its statue. The page pins each package by its size and
// SHA-256 (starter_packages.json, written by viewer/starter_packages.py) and fetches it with that integrity, so the
// browser refuses any other bytes; a package that is missing or different gives a message with its download link.
// A page opened as a file reads nothing beside it, so there the message offers the download alone: a downloaded
// package dropped on the page opens through the Open path as any card's .zip does.

import PINS from './starter_packages.json' with { type: 'json' };

// the hair and the eyes of a statue take its stone's colour
export const STONE_HAIR: Record<string, string> = { carrara: '#dcd7cf', travertine: '#cdb48d', bardiglio: '#9a9ea2', basalt: '#3d3b38', bronze: '#8d5a2b', verdigris: '#6a9a86' };

export interface StarterPackage { file: string; bytes: number; sha256: string; card: string }
export interface StarterPins { format: string; tag: string; release: string; download: string; packages: Record<string, StarterPackage> }
// id: the statue's name (opensculptboy/data/statues/<id>.json); name: the New screen's label, which starts the name of
// the package's card
export interface Starter { id: string; name: string; description: string; stone: string; package: StarterPackage }

// the New screen's statues in their order, the first the default of a first visit
const STATUES: Omit<Starter, 'package'>[] = [
  { id: 'doryphoros', name: 'Doryphoros', stone: 'carrara',
    description: 'The spear bearer of Polykleitos at rest in contrapposto, with its tree stump and its base' },
  { id: 'david_goliath', name: 'David with the head of Goliath', stone: 'carrara',
    description: 'A figure with a sword and a brimmed hat, the head of Goliath at its feet, on its base' },
];

export const STARTER_PINS = PINS as StarterPins;
// the statues whose package the page pins
export function startersOf(pins: StarterPins): Starter[] {
  return STATUES.filter((s) => pins.packages[s.id]).map((s) => ({ ...s, package: pins.packages[s.id] }));
}
export const STARTERS: Starter[] = startersOf(STARTER_PINS);
export const DEFAULT_STARTER = STATUES[0].id;
export const starterById = (id: string) => STARTERS.find((s) => s.id === id) || null;
// the starter of a character by its name (the package's card names it)
export const starterOfName = (name: string) => STARTERS.find((s) => name === s.name || name === s.package.card || name.startsWith(s.name + ' ')) || null;

// the folder of the packages, beside the page
export const STARTER_FOLDER = 'starters/';
export const packageUrl = (s: Starter) => STARTER_FOLDER + s.package.file;
export const downloadUrl = (s: Starter, pins: StarterPins = STARTER_PINS) => pins.download + s.package.file;
// the package's subresource integrity: its SHA-256 in base64
export function integrityOf(sha256hex: string): string {
  const bytes = sha256hex.match(/../g)!.map((h) => parseInt(h, 16));
  return 'sha256-' + btoa(String.fromCharCode(...bytes));
}
export const sizeLabel = (bytes: number) => (bytes / 1e6).toFixed(1) + ' MB';

// why a package did not open: the page is a file, the package is not beside the page, it is another version (its size
// or its hash differ from the pins), or the request failed
export type StarterProblem = 'file' | 'missing' | 'other' | 'network';
export interface StarterNote { id: string; problem: StarterProblem; message: string; link: string; label: string }
export type StarterResult = { ok: true; file: File } | { ok: false; note: StarterNote };

export function starterNote(s: Starter, problem: StarterProblem, pins: StarterPins = STARTER_PINS): StarterNote {
  const f = s.package.file, size = sizeLabel(s.package.bytes);
  const message = {
    file: `This page was opened as a file, so it cannot read the statue packages beside it. Download ${f} (${size}) and drop it on the page, or serve the page with serve_viewer.py and put the package in its starters folder.`,
    missing: `${s.name} comes as a package of its own: ${f} (${size}) is not in the starters folder beside this page. Download it, put it in that folder and pick the statue again, or drop the downloaded file on the page.`,
    other: `The ${f} beside this page is another version than the one this page pins, so the page did not open it. Download the package again and put it in the starters folder beside the page.`,
    network: `The page could not read ${f} from the starters folder beside it. Pick the statue again, or download the package and drop it on the page.`,
  }[problem];
  return { id: s.id, problem, message, link: downloadUrl(s, pins), label: `Download ${f}` };
}

// the package of a starter from beside the page, checked against its pins; the fetch and the page's protocol can be
// given for the tests
export async function fetchStarter(s: Starter, opts: { fetch?: typeof fetch; protocol?: string; pins?: StarterPins } = {}): Promise<StarterResult> {
  const get = opts.fetch || fetch, protocol = opts.protocol ?? location.protocol, pins = opts.pins || STARTER_PINS;
  const fail = (problem: StarterProblem): StarterResult => ({ ok: false, note: starterNote(s, problem, pins) });
  if (protocol === 'file:') return fail('file');
  const url = packageUrl(s);
  // the size first, so that a missing package or another version costs no download
  try {
    const head = await get(url, { method: 'HEAD', cache: 'no-store' });
    if (head.status === 404) return fail('missing');
    if (!head.ok) return fail('network');
    const n = Number(head.headers.get('Content-Length'));
    if (Number.isFinite(n) && n > 0 && n !== s.package.bytes) return fail('other');
  } catch (e) { return fail('network'); }
  // the browser checks the bytes against the pinned SHA-256 and refuses the response when they differ
  let bytes: ArrayBuffer;
  try {
    const r = await get(url, { cache: 'no-store', integrity: integrityOf(s.package.sha256) });
    if (r.status === 404) return fail('missing');
    if (!r.ok) return fail('network');
    bytes = await r.arrayBuffer();
  } catch (e) { return fail('other'); }
  if (bytes.byteLength !== s.package.bytes) return fail('other');
  return { ok: true, file: new File([bytes], s.package.file, { type: 'application/zip' }) };
}

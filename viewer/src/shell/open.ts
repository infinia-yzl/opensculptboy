// OpenSculptBoy
// Apache License, Version 2.0
//
// The one Open path of the page: Open in the top bar (Ctrl O), its drop zone and file pickers, the New screen, the
// Dress mode's buttons and a drop anywhere on the page all hand their files here. Each file goes to the loader of what
// it holds: a character card with its files (or its folder, or a .zip of both), the card inside an OpenSculptBoy GLB
// or VRM file, wearable packages and outfits, sheet settings (opensculptboy/sheet-settings@1), a character of the page
// (anny-viewer/look), or a picture to pose from. A card that opens shows its steps (reading, the body, the maps, the
// parts, the dress) in the progress card. A file that the page cannot open gives a message in plain words with the
// next step.

import { fromDrop, fromInput, type PickedFile } from '../wearables/files.ts';
import { SETTINGS_FORMAT } from '../sheet/settings.ts';
import { cardInGltf, needsFiles, namedFiles, readZip } from '../card/file.ts';
import { actions, ui, type OpenError } from './state.ts';
import { toast } from '../ui/toast.ts';

export const FIT_DOCS = 'https://github.com/infinia-yzl/opensculptboy/blob/main/docs/reference-fit.md';

export interface CardResult { name: string; pieces: number; worn: number }
export interface OpenDeps {
  loadCard(picked: PickedFile[], step: (id: string) => void): Promise<CardResult>;
  applyLook(json: any): string;              // the name of the character
  setSheet(json: any): void;
  posePicture(file: File): Promise<void>;
}
export interface Opener { pickFiles(): void; pickFolder(): void; open(picked: PickedFile[]): Promise<boolean> }

// the steps of a card's open, in order, with their words
export const CARD_STEPS: [string, string][] = [
  ['read', 'Reading the files'], ['body', 'Building the body'], ['maps', 'Reading the skin\'s maps'],
  ['parts', 'The eyes, the mouth and the hair'], ['dress', 'Dressing'],
];

const ext = (p: string) => (p.toLowerCase().match(/\.([a-z0-9]+)$/) || [, ''])[1];
const MODELS = ['glb', 'gltf', 'vrm', 'fbx', 'obj', 'blend', 'dae', 'usdz', 'stl'];
const PICTURES = ['png', 'jpg', 'jpeg', 'webp'];
const fileOf = (path: string, data: Uint8Array | string) => ({ path, file: new File([data as BlobPart], path.split('/').pop() || path) });

export function makeOpener(deps: OpenDeps): Opener {
  // the page's two file inputs: files, and a folder
  const files = document.createElement('input');
  files.type = 'file'; files.multiple = true; files.hidden = true; files.id = 'open-files';
  files.accept = '.json,.glb,.vrm,.gltf,.zip,.safetensors,.png,.jpg,.jpeg,.webp,image/*';
  const folder = document.createElement('input');
  folder.type = 'file'; folder.multiple = true; folder.hidden = true; folder.id = 'open-folder';
  folder.setAttribute('webkitdirectory', '');
  document.body.append(files, folder);

  const fail = (e: OpenError) => { ui.set({ openBusy: '', progress: null }); actions.showOpen(e); };

  // a card's open with its steps in the progress card
  async function openCard(name: string, picked: PickedFile[]): Promise<CardResult> {
    let at = 0;
    const show = () => ui.set({ progress: { title: name ? `Opening ${name}` : 'Opening the character', steps: CARD_STEPS.map(([, label], i) => [label, i < at ? 'done' : i === at ? 'now' : 'todo']) } });
    show();
    const r = await deps.loadCard(picked, (id) => { const i = CARD_STEPS.findIndex(([k]) => k === id); if (i > at) { at = i; show(); } });
    ui.set({ progress: null });
    return r;
  }

  async function open(picked: PickedFile[]): Promise<boolean> {
    if (!picked.length) { fail({ title: 'The drop held no files', body: 'Drop the files themselves, or choose them with Choose files.' }); return false; }
    ui.set({ openBusy: picked.length > 1 ? `Reading ${picked.length} files` : 'Reading the file', openError: null });
    // a drop with the dialog closed says that the files are on their way
    if (!ui.get().openDialog) toast(ui.get().openBusy + '…');
    // a .zip opens as the files it holds
    const zips = picked.filter((p) => ext(p.path) === 'zip');
    if (zips.length) {
      const inner: PickedFile[] = picked.filter((p) => ext(p.path) !== 'zip');
      try {
        for (const z of zips) for (const e of await readZip(new Uint8Array(await z.file.arrayBuffer()))) inner.push(fileOf(e.path, e.data));
      } catch (err: any) { fail({ title: 'The .zip file could not be read', body: String(err?.message || err) }); return false; }
      if (!inner.length) { fail({ title: 'The .zip file is empty', body: 'Choose a .zip that holds a character card and its files.' }); return false; }
      picked = inner;
    }
    let sheet: any = null, look: any = null;
    const rest: PickedFile[] = [], pictures: PickedFile[] = [], models: PickedFile[] = [];
    for (const p of picked) {
      const e = ext(p.path);
      if (e === 'json' && !/\.(wearable|outfit)\.json$/i.test(p.path)) {
        let j: any = null;
        try { j = JSON.parse(await p.file.text()); } catch (err) { /* not JSON: the card loader says so */ }
        if (j && j.format === SETTINGS_FORMAT) { sheet = j; continue; }
        if (j && typeof j.format === 'string' && j.format.startsWith('anny-viewer/look')) { look = j; continue; }
      }
      if (PICTURES.includes(e)) { pictures.push(p); continue; }
      if (MODELS.includes(e)) { models.push(p); continue; }
      rest.push(p);
    }
    // an OpenSculptBoy GLB or VRM file: the card in its scene extras; a card of sliders opens whole, a card that names
    // files opens whole with them (its folder or its .zip), and else asks for them
    let inGltf: { card: any; from: string } | null = null;
    for (const m of models) {
      if (!['glb', 'gltf', 'vrm'].includes(ext(m.path))) continue;
      const card = cardInGltf(new Uint8Array(await m.file.arrayBuffer()));
      if (card) { inGltf = { card, from: m.path }; break; }
    }
    const done: string[] = [];
    let mode: [any, string?] | null = null;
    try {
      if (inGltf && !rest.some((p) => ext(p.path) === 'json')) {
        const { card, from } = inGltf, what = ext(from) === 'vrm' ? 'VRM file' : 'GLB file';
        const have = new Set(rest.map((p) => p.path.split('/').pop()));
        const missing = namedFiles(card).filter((f) => !have.has(f.split('/').pop()));
        if (needsFiles(card) && missing.length) {
          fail({ title: `This ${what} holds the card of ${card.name || 'a character'}, without its files`, fit: false,
            body: `The card names ${missing.length} file${missing.length > 1 ? 's' : ''} of its own (${missing.slice(0, 3).join(', ')}${missing.length > 3 ? ', …' : ''}). Open the card's .zip or its folder to see it whole, or open the sliders alone.`,
            action: { label: 'Open the sliders alone', run: () => open([fileOf('character.json', JSON.stringify(slidersOnly(card)))]) } });
          return false;
        }
        rest.unshift(fileOf('character.json', JSON.stringify(card)));
        done.push(`Read the card from the ${what}`);
      } else if (models.length && !rest.length && !look && !sheet && !pictures.length) {
        const e = ext(models[0].path);
        const what = e === 'vrm' ? 'VRM file' : e === 'glb' || e === 'gltf' ? 'GLB file' : '3D model';
        fail({ title: `This ${what} holds no character card`, fit: true,
          body: 'The page opens a character card (a .json file with its files, their folder or their .zip) and the GLB and VRM files that OpenSculptBoy writes. Another 3D model becomes a character through a fit.' });
        return false;
      }
      if (rest.length) {
        const r = await openCard(cardName(rest), rest);
        done.push(`Opened ${r.name}` + (r.pieces ? `: ${r.worn} of ${r.pieces} pieces worn` : ''));
        mode = r.pieces ? ['dress'] : ['create', 'body'];
      }
      if (look) { done.push(`Opened ${deps.applyLook(look)}`); mode ||= ['create', 'body']; }
      if (sheet) { deps.setSheet(sheet); done.push('Opened the sheet settings'); mode = ['sheet']; }
      if (pictures.length) {
        ui.set({ openBusy: 'Reading the pose. The first picture starts MediaPipe, which takes a few seconds' });
        await deps.posePicture(pictures[0].file);
        done.push('Posed the figure from the picture'); mode ||= ['pose'];
      }
    } catch (err: any) {
      const msg = String(err?.message || err);
      console.error('Open:', msg, err);
      if (/no character card and no wearable package/.test(msg)) {
        fail({ title: 'These files hold no character card', fit: true,
          body: 'A character card is a .json file such as character.json, opened with the files it names, with its whole folder or with its .zip. Choose the card and its files, the folder or the .zip.' });
      } else if (/gave no pose|No pose/i.test(msg)) {
        fail({ title: 'The picture gave no pose', body: 'Choose a picture where the whole figure shows, standing clear of the edges.' });
      } else {
        fail({ title: 'The files could not be opened', body: msg.replace(/\.?$/, '.') + ' Choose the card with every file it names, its folder or its .zip.' });
      }
      return false;
    }
    ui.set({ openBusy: '', progress: null });
    actions.hideOpen();
    actions.hideNew();
    if (ui.get().welcome) actions.dismissWelcome();
    if (mode) actions.openMode(mode[0], mode[1]);
    toast(done.join('. ') + '.');
    return true;
  }

  files.addEventListener('change', () => { const p = fromInput(files.files || []); files.value = ''; open(p); });
  folder.addEventListener('change', () => { const p = fromInput(folder.files || []); folder.value = ''; open(p); });

  // a drop anywhere on the page
  const holds = (e: DragEvent) => Array.from(e.dataTransfer?.types || []).includes('Files');
  let depth = 0;
  document.addEventListener('dragenter', (e) => { if (holds(e)) { depth++; ui.set({ dragging: true }); } });
  document.addEventListener('dragover', (e) => { if (holds(e)) { e.preventDefault(); ui.set({ dragging: true }); } });
  document.addEventListener('dragleave', (e) => { if (holds(e) && --depth <= 0) { depth = 0; ui.set({ dragging: false }); } });
  document.addEventListener('drop', async (e) => {
    if (!holds(e)) return;
    e.preventDefault(); depth = 0; ui.set({ dragging: false });
    let picked: PickedFile[];
    try { picked = await fromDrop(e.dataTransfer!); }
    catch (err: any) { fail({ title: 'The dropped files could not be read', body: `${err?.message || err}. Choose the files with Choose files.` }); return; }
    open(picked);
  });
  return { pickFiles: () => files.click(), pickFolder: () => folder.click(), open };
}

// the name of the card among the files, for the progress card
function cardName(picked: PickedFile[]): string {
  const j = picked.find((p) => /(^|\/)character\.json$/i.test(p.path)) || picked.find((p) => ext(p.path) === 'json' && !/\.(wearable|outfit)\.json$/i.test(p.path));
  return j ? j.path.split('/').pop()!.replace(/\.json$/i, '') : '';
}

// a card's sliders, face shapes and look, without the fields that name files
export function slidersOnly(card: any): any {
  const out: any = {};
  for (const k of ['schema', 'name', 'rig', 'topology', 'phenotypes', 'phenotype', 'local_changes', 'face_shapes', 'facial_actions', 'pose', 'look']) if (k in card) out[k] = card[k];
  return out;
}

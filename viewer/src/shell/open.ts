// OpenSculptBoy
// Apache License, Version 2.0
//
// The one Open path of the page: Open in the top bar (Ctrl O), its drop zone and file pickers, the Dress mode's
// buttons and a drop anywhere on the page all hand their files here. Each file goes to the loader of what it holds: a
// character card with its files (or its folder), wearable packages and outfits, sheet settings
// (opensculptboy/sheet-settings@1), a character of the page (anny-viewer/look), or a picture to pose from. A file
// that the page cannot open gives a message in plain words with the next step.

import { fromDrop, fromInput, type PickedFile } from '../wearables/files.ts';
import { SETTINGS_FORMAT } from '../sheet/settings.ts';
import { actions, ui, type OpenError } from './state.ts';
import { toast } from '../ui/toast.ts';

export const FIT_DOCS = 'https://github.com/infinia-yzl/opensculptboy/blob/main/docs/reference-fit.md';

export interface CardResult { name: string; pieces: number; worn: number }
export interface OpenDeps {
  loadCard(picked: PickedFile[]): Promise<CardResult>;
  applyLook(json: any): string;              // the name of the character
  setSheet(json: any): void;
  posePicture(file: File): Promise<void>;
}
export interface Opener { pickFiles(): void; pickFolder(): void; open(picked: PickedFile[]): Promise<boolean> }

const ext = (p: string) => (p.toLowerCase().match(/\.([a-z0-9]+)$/) || [, ''])[1];
const MODELS = ['glb', 'gltf', 'vrm', 'fbx', 'obj', 'blend', 'dae', 'usdz', 'stl'];
const PICTURES = ['png', 'jpg', 'jpeg', 'webp'];

export function makeOpener(deps: OpenDeps): Opener {
  // the page's two file inputs: files, and a folder
  const files = document.createElement('input');
  files.type = 'file'; files.multiple = true; files.hidden = true; files.id = 'open-files';
  files.accept = '.json,.glb,.vrm,.safetensors,.png,.jpg,.jpeg,.webp,image/*';
  const folder = document.createElement('input');
  folder.type = 'file'; folder.multiple = true; folder.hidden = true; folder.id = 'open-folder';
  folder.setAttribute('webkitdirectory', '');
  document.body.append(files, folder);

  const fail = (e: OpenError) => { ui.set({ openBusy: '' }); actions.showOpen(e); };

  async function open(picked: PickedFile[]): Promise<boolean> {
    if (!picked.length) { fail({ title: 'The drop held no files', body: 'Drop the files themselves, or choose them with Choose files.' }); return false; }
    ui.set({ openBusy: picked.length > 1 ? `Reading ${picked.length} files` : 'Reading the file', openError: null });
    // a drop with the dialog closed says that the files are on their way
    if (!ui.get().openDialog) toast(ui.get().openBusy + '…');
    let sheet: any = null, look: any = null;
    const rest: PickedFile[] = [], pictures: PickedFile[] = [];
    for (const p of picked) {
      const e = ext(p.path);
      if (e === 'json' && !/\.(wearable|outfit)\.json$/i.test(p.path)) {
        let j: any = null;
        try { j = JSON.parse(await p.file.text()); } catch (err) { /* not JSON: the card loader says so */ }
        if (j && j.format === SETTINGS_FORMAT) { sheet = j; continue; }
        if (j && typeof j.format === 'string' && j.format.startsWith('anny-viewer/look')) { look = j; continue; }
      }
      if (PICTURES.includes(e)) { pictures.push(p); continue; }
      rest.push(p);
    }
    const done: string[] = [];
    let mode: [any, string?] | null = null;
    try {
      if (rest.length) {
        const names = rest.map((p) => ext(p.path));
        if (names.every((e) => MODELS.includes(e))) {
          const what = names[0] === 'vrm' ? 'VRM file' : names[0] === 'glb' || names[0] === 'gltf' ? 'GLB file' : '3D model';
          fail({ title: `This ${what} holds no character card that the page reads`, fit: true,
            body: 'The page opens a character card (a .json file with its files, or their folder). A card inside an OpenSculptBoy GLB or VRM file opens in a later version. Another 3D model becomes a character through a fit.' });
          return false;
        }
        const r = await deps.loadCard(rest);
        done.push(`Opened ${r.name}` + (r.pieces ? `: ${r.worn} of ${r.pieces} pieces worn` : ''));
        mode = r.pieces ? ['dress'] : ['create', 'body'];
      }
      if (look) { done.push(`Opened ${deps.applyLook(look)}`); mode ||= ['create', 'body']; }
      if (sheet) { deps.setSheet(sheet); done.push('Opened the sheet settings'); mode = ['sheet']; }
      if (pictures.length) {
        ui.set({ openBusy: 'Reading the pose. The first picture loads MediaPipe, about 20 MB' });
        await deps.posePicture(pictures[0].file);
        done.push('Posed the figure from the picture'); mode ||= ['pose'];
      }
    } catch (err: any) {
      const msg = String(err?.message || err);
      console.error('Open:', msg, err);
      if (/no character card and no wearable package/.test(msg)) {
        fail({ title: 'These files hold no character card', fit: true,
          body: 'A character card is a .json file such as character.json, opened with the files it names, or with its whole folder. Choose the card and its files, or the folder.' });
      } else if (/gave no pose|No pose/i.test(msg)) {
        fail({ title: 'The picture gave no pose', body: 'Choose a picture where the whole figure shows, standing clear of the edges.' });
      } else {
        fail({ title: 'The files could not be opened', body: msg.replace(/\.?$/, '.') + ' Choose the card with every file it names, or its folder.' });
      }
      return false;
    }
    ui.set({ openBusy: '' });
    actions.hideOpen();
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

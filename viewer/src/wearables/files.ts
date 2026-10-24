// OpenSculptBoy
// Apache License, Version 2.0
//
// The user's own files for the wardrobe: a character card, its wearable packages and the level-0 offsets of its
// detail layer, picked with a file input (files or a folder) or dropped on the page. The page reads them in the
// browser and sends nothing anywhere.

export interface PickedFile { path: string; file: File }

// the files of an <input type=file> (webkitRelativePath holds the folder's path when a folder was picked)
export function fromInput(list: FileList | File[]): PickedFile[] {
  return Array.from(list, (file) => ({ path: (file as any).webkitRelativePath || file.name, file }));
}

// the files of a drop, walking dropped folders
export async function fromDrop(dt: DataTransfer): Promise<PickedFile[]> {
  const entries = Array.from(dt.items || []).map((it) => (it as any).webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return fromInput(dt.files);
  const out: PickedFile[] = [];
  const walk = async (entry: any, prefix: string): Promise<void> => {
    if (entry.isFile) {
      const file: File = await new Promise((res, rej) => entry.file(res, rej));
      out.push({ path: prefix + entry.name, file });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      for (;;) {
        const batch: any[] = await new Promise((res, rej) => reader.readEntries(res, rej));
        if (!batch.length) break;
        for (const e of batch) await walk(e, prefix + entry.name + '/');
      }
    }
  };
  for (const e of entries) await walk(e, '');
  return out;
}

export function normalise(path: string): string {
  const parts: string[] = [];
  for (const p of path.replace(/\\/g, '/').split('/')) {
    if (p === '' || p === '.') continue;
    if (p === '..') parts.pop(); else parts.push(p);
  }
  return parts.join('/');
}
export const dirname = (path: string) => { const i = path.lastIndexOf('/'); return i < 0 ? '' : path.slice(0, i); };
export const basename = (path: string) => path.replace(/\\/g, '/').split('/').pop() || '';

// the picked files by path, with a lookup of a path relative to a folder (or by its name alone when it is the only
// file of that name: a card may hold paths of another computer)
export class FileSet {
  byPath = new Map<string, File>();
  constructor(files: PickedFile[]) { for (const f of files) this.byPath.set(normalise(f.path), f.file); }
  find(relative: string, folder = ''): File | null {
    const direct = this.byPath.get(normalise(relative.startsWith('/') ? relative : (folder ? folder + '/' : '') + relative));
    if (direct) return direct;
    const name = basename(relative), same = [...this.byPath.entries()].filter(([p]) => basename(p) === name);
    return same.length === 1 ? same[0][1] : null;
  }
  pathOf(file: File): string { for (const [p, f] of this.byPath) if (f === file) return p; return file.name; }
  async bytes(relative: string, folder = ''): Promise<Uint8Array> {
    const f = this.find(relative, folder);
    if (!f) throw new Error(`The file ${relative} is not among the files picked.`);
    return new Uint8Array(await f.arrayBuffer());
  }
}

// OpenSculptBoy
// Apache License, Version 2.0
//
// Write src/ui/icon_paths.ts: the icons of the page, from Phosphor Icons in the Light weight (@phosphor-icons/core,
// MIT licence), as the design system names them (its README, "Iconography"). Run it after a change to the table:
//     node icons.mjs
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const here = path.dirname(fileURLToPath(import.meta.url));
// the page's name of each icon, and the Phosphor icon that draws it
const TABLE = {
  characters: 'users-three', body: 'person-simple', face: 'smiley-blank', hair: 'waves', skin: 'drop', wear: 't-shirt',
  pose: 'person-arms-spread', expression: 'mask-happy', sheet: 'layout', export: 'download-simple', open: 'folder-open',
  scene: 'sun', stats: 'pulse', undo: 'arrow-counter-clockwise', redo: 'arrow-clockwise', dice: 'dice-five', save: 'image-square',
  fullscreen: 'corners-out', help: 'question', more: 'dots-three', turntable: 'arrows-clockwise', bones: 'bone', reset: 'crosshair-simple',
  camera: 'camera', close: 'x', collapse: 'caret-up', expand: 'caret-down', chevron: 'caret-right', play: 'play', pause: 'pause',
  search: 'magnifying-glass', copy: 'copy', paste: 'clipboard-text', download: 'download-simple', upload: 'upload-simple', plus: 'plus',
  stop: 'stop', undoSmall: 'arrow-u-up-left', drag: 'dots-six-vertical', hud: 'monitor', head: 'user', focus: 'crosshair',
  view: 'eye', sun: 'sun', moon: 'moon', system: 'desktop', settings: 'gear-six', picture: 'image', glb: 'cube', vrm: 'user-square',
  zip: 'file-zip', json: 'file-code', file: 'file', folder: 'folder', alert: 'warning-circle', check: 'check', link: 'arrow-square-out',
  eyedropper: 'eyedropper', silhouette: 'person', planned: 'hourglass-simple', keyboard: 'keyboard', info: 'info',
};
const dir = path.join(here, 'node_modules', '@phosphor-icons', 'core', 'assets', 'light');
const lines = Object.entries(TABLE).map(([name, ph]) => {
  const svg = fs.readFileSync(path.join(dir, `${ph}-light.svg`), 'utf8');
  const body = svg.replace(/^<svg[^>]*>/, '').replace(/<\/svg>\s*$/, '');
  return `  ${name}: ${JSON.stringify(body)},`;
});
const out = `// OpenSculptBoy
// Apache License, Version 2.0
//
// The icons of the page: Phosphor Icons, Light weight (@phosphor-icons/core 2.1.1, MIT licence, phosphoricons.com),
// filled paths on a 256-unit grid in currentColor. Written by icons.mjs; edit its table, not this file.

export const ICON_PATHS: Record<string, string> = {
${lines.join('\n')}
};
`;
fs.writeFileSync(path.join(here, 'src', 'ui', 'icon_paths.ts'), out);
console.log(Object.keys(TABLE).length, 'icons');

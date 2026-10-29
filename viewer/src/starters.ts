// OpenSculptBoy
// Apache License, Version 2.0
//
// The starter characters (DESIGN-0012, "Statue starters"): classical statues in the stones of the library. Each has
// two kinds. The light starter is this table: Anny's sliders for the statue's proportions, a face of the calibrated
// distribution, a page hair style in the stone's colour, a stone and a classical pose of the pose library, a few
// kilobytes in the page. The finished starter is a fitted card with its sculpted detail, its own hair and drapery,
// as a .zip beside the page (`starters/<statue>.zip`, made by `python -m opensculptboy.statues starter`), which the New
// screen and a first visit open through the Open path when it is there; its props are wearables of the slot `prop`,
// which Dress shows as switches. A statue's fit gives the light kind too: a card beside the page
// (`starters/<statue>.light.json`, the fitted sliders, face shapes and pose) takes the place of this table's
// phenotype, face and pose, and the table keeps the stone and the hair. Until those files land, both kinds show this
// table.
// Four of the faces were drawn from the calibrated distribution for the page's earlier presets and chosen on review
// renders; the Athlete and the Guardian keep Anny's default face.

// the hair and the eyes of a statue take its stone's colour
export const STONE_HAIR: Record<string, string> = { carrara: '#dcd7cf', travertine: '#cdb48d', bardiglio: '#9a9ea2', basalt: '#3d3b38', bronze: '#8d5a2b', verdigris: '#6a9a86' };

// pose: a pose of the library (base) with some bones turned about the world's axes (degrees about x, the figure's
// left, y, up, and z, forward), which the page builds into a pose clip (main.ts starterPose)
export interface StarterPose { base: string; turns: [string, number[]][] }
// statue: the name of the statue's fit (opensculptboy/data/statues/<statue>.json), whose files lie beside the page
export interface Starter { id: string; name: string; description: string; statue?: string; pose: StarterPose; light: any }

// the folder of the finished starters, beside the page
export const STARTER_FOLDER = 'starters/';
const fileOf = (s: Starter) => STARTER_FOLDER + (s.statue || s.id);
export const finishedUrl = (s: Starter) => fileOf(s) + '.zip';
// the finished starter's picture, which the starter's command writes beside its .zip
export const finishedPicture = (s: Starter) => fileOf(s) + '.png';
export const lightCardUrl = (s: Starter) => fileOf(s) + '.light.json';
// the light look of a starter with the light card of its fit (or null): the card's sliders and face shapes over the
// table's, and the table's stone, hair and eyes
export function lightLook(s: Starter, card: any): any {
  const look = JSON.parse(JSON.stringify(s.light));
  if (!card) return look;
  look.phenotype = Object.assign({}, look.phenotype, card.phenotype || {});
  if (card.face_shapes) look.face = Object.assign({}, card.face_shapes);
  if (card.name) look.name = card.name;
  return look;
}

export const STARTERS: Starter[] = [
  { id: 'spear_bearer', name: 'Spear Bearer', description: 'A figure at rest in contrapposto, the weight on one leg', statue: 'doryphoros', pose: { base: 'relaxed', turns: [
      // contrapposto: the weight on the right leg, the hips down on the left and the shoulders down on the right
      ['root', [0, -6, -4]], ['upperleg01.R', [0, 0, 4]], ['spine05', [0, 4, 6]], ['spine03', [0, 0, 2]],
      // the left knee bent and the left foot back
      ['upperleg01.L', [10, 0, 2]], ['lowerleg01.L', [26, 0, 0]], ['foot.L', [-12, 0, 0]],
      // the right arm down, the left arm bent as if it held a spear
      ['upperarm01.R', [0, 0, 4]], ['upperarm01.L', [-12, 0, 8]], ['lowerarm01.L', [-78, 10, 0]], ['wrist.L', [0, 0, -10]],
      // the head turned to its right and a little down
      ['neck01', [6, -14, 0]], ['head', [4, -16, 0]]] },
    light: { format: 'anny-viewer/look@4', name: 'Spear Bearer', skin: { tone: 0.35, undertone: 0, stone: 'carrara' }, hair: { style: 'curly_top_taper', color: STONE_HAIR['carrara'] }, eyes: { color: STONE_HAIR['carrara'] },
      phenotype: {gender: 0, age: 0.8, muscle: 0.72, weight: 0.5, height: 0.6, proportions: 0.66, african: 0, asian: 1, caucasian: 1},
      face: { 'head-scale-vert': 0.41, 'head-fat': -0.04, 'head-rectangular': 0.02, 'head-back-scale-depth': -1.21, 'head-scale-depth': 1.42, 'head-scale-horiz': 0.21, 'head-triangular': 0.06, 'forehead-temple': 0.01, 'forehead-trans': -0.2, 'forehead-scale-vert': 0.02, 'eyebrows-trans-down-up': -0.05, 'eyebrows-trans-backward-forward': 0.01, 'eye-corner1': 0.05, 'eye-height1': -0.05, 'eye-scale': 0.08, 'eye-push1': -0.09, 'eye-eyefold-down-up': -0.09, 'eye-height3': -0.16, 'eye-height2': -0.06, 'eye-bag-decr-incr': -0.05, 'eye-push2': -0.11, 'eye-trans-down-up': 0.19, 'eye-bag-height': -0.14, 'eye-epicanthus': 0.07, 'eye-eyefold-angle': 0.07, 'eye-bag-in-out': 0.04, 'eye-corner2': -0.05, 'eye-trans-in-out': 0.11, 'eye-eyefold-concave-convex': 0.27, 'nose-nostrils-angle': -0.09, 'nose-flaring': 0.03, 'nose-curve': -0.07, 'nose-scale-depth': -0.07, 'nose-scale-vert': -0.03, 'nose-point-width': -0.09, 'nose-scale-horiz': 0.07, 'nose-width1': 0.01, 'nose-trans-backward-forward': -0.25, 'nose-trans-down-up': 0.06, 'nose-hump': -0.03, 'nose-nostrils-width': 0.01, 'nose-greek': -0.13, 'nose-point': 0.06, 'nose-width2': 0.07, 'nose-width3': -0.1, 'nose-septumangle': 0.1, 'nose-base': 0.12, 'nose-compression': -0.01, 'nose-volume': -0.09, 'cheek-trans': -0.01, 'cheek-volume': 0.01, 'cheek-inner': 0.12, 'cheek-bones': 0.21, 'mouth-lowerlip-height': 0.22, 'mouth-scale-horiz': -0.07, 'mouth-scale-depth': -0.04, 'mouth-lowerlip-volume': 0.04, 'mouth-angles': -0.08, 'mouth-upperlip-ext': 0.02, 'mouth-lowerlip-width': 0.01, 'mouth-trans-backward-forward': 0.33, 'mouth-upperlip-width': 0.01, 'mouth-scale-vert': 0.2, 'mouth-dimples': 0.02, 'mouth-upperlip-height': 0.09, 'mouth-upperlip-volume': 0.03, 'mouth-cupidsbow-width': 0.02, 'mouth-lowerlip-ext': 0.1, 'mouth-upperlip-middle': 0.27, 'mouth-trans-down-up': -0.01, 'mouth-lowerlip-middle': 0.07, 'mouth-philtrum-volume': -0.03, 'chin-jaw-drop': 0.01, 'chin-prominent': -0.05, 'chin-bones': 0.14, 'chin-width': -0.06, 'chin-cleft': 0.01, 'chin-triangle': 0.04, 'chin-prognathism': -0.02, 'ear-lobe': 0.21, 'ear-shape-square-round': -0.02, 'ear-trans-backward-forward': -0.12, 'ear-rot': -0.01, 'ear-trans-down-up': -0.16, 'ear-scale-vert': -0.13, 'ear-flap': 0.21, 'ear-shape-pointed-triangle': 0.17, 'ear-scale': -0.14, 'ear-scale-depth': 0.11, 'ear-wing': 0.19, 'detail-1': 0.12, 'detail-2': 0.04, 'detail-3': -0.14, 'detail-4': -0.14, 'detail-5': -0.16, 'detail-6': -0.03, 'detail-7': 0.52, 'detail-8': 0.05, 'detail-9': 0.15, 'detail-10': 0.18 } } },
  { id: 'orator', name: 'Orator', description: 'A speaker with a raised arm', pose: { base: 'relaxed', turns: [
      ['root', [0, 4, 3]], ['upperleg01.L', [0, 0, -3]], ['upperleg01.R', [8, 0, -2]], ['lowerleg01.R', [16, 0, 0]], ['foot.R', [-6, 0, 0]],
      // the right arm raised forward and out, the hand open to the crowd
      ['upperarm01.R', [-78, 0, -18]], ['lowerarm01.R', [-18, 0, 0]], ['wrist.R', [-20, 0, 0]],
      // the left arm bent at the waist, as if it held the folds of a mantle
      ['upperarm01.L', [-8, 0, 6]], ['lowerarm01.L', [-62, 0, 0]],
      ['neck01', [-4, -8, 0]], ['head', [-4, -10, 0]]] },
    light: { format: 'anny-viewer/look@4', name: 'Orator', skin: { tone: 0.35, undertone: 0, stone: 'bardiglio' }, hair: { style: 'textured_crop', color: STONE_HAIR['bardiglio'] }, eyes: { color: STONE_HAIR['bardiglio'] },
      phenotype: {gender: 0, age: 0.95, muscle: 0.5, weight: 0.58, height: 0.55, proportions: 0.55, african: 0, asian: 1, caucasian: 0},
      face: { 'head-scale-vert': 0.11, 'head-fat': -0.09, 'head-invertedtriangular': 0.07, 'head-back-scale-depth': -1.43, 'head-scale-depth': 0.81, 'head-scale-horiz': 0.11, 'forehead-temple': 0.08, 'forehead-trans': 0.07, 'forehead-scale-vert': -0.01, 'forehead-nubian': 0.01, 'eyebrows-angle': -0.03, 'eyebrows-trans-down-up': 0.02, 'eyebrows-trans-backward-forward': -0.03, 'eye-corner1': 0.02, 'eye-height1': 0.02, 'eye-scale': 0.14, 'eye-push1': -0.12, 'eye-eyefold-down-up': 0.02, 'eye-height3': -0.06, 'eye-height2': -0.06, 'eye-bag-decr-incr': 0.05, 'eye-push2': -0.26, 'eye-bag-height': 0.05, 'eye-epicanthus': 0.05, 'eye-eyefold-angle': 0.03, 'eye-bag-in-out': 0.04, 'eye-corner2': -0.04, 'eye-trans-in-out': 0.01, 'eye-eyefold-concave-convex': -0.13, 'nose-nostrils-angle': 0.01, 'nose-flaring': 0.04, 'nose-curve': -0.01, 'nose-scale-depth': -0.05, 'nose-scale-vert': 0.02, 'nose-point-width': -0.02, 'nose-scale-horiz': -0.02, 'nose-width1': -0.04, 'nose-trans-backward-forward': -0.08, 'nose-trans-down-up': -0.04, 'nose-hump': 0.02, 'nose-nostrils-width': -0.01, 'nose-greek': 0.04, 'nose-point': -0.04, 'nose-width2': 0.01, 'nose-width3': 0.06, 'nose-septumangle': -0.01, 'nose-base': 0.01, 'nose-compression': 0.01, 'cheek-trans': 0.02, 'cheek-volume': 0.01, 'cheek-inner': 0.07, 'cheek-bones': -0.04, 'mouth-lowerlip-height': 0.04, 'mouth-scale-horiz': -0.21, 'mouth-cupidsbow': 0.01, 'mouth-laugh-lines': 0.01, 'mouth-scale-depth': -0.13, 'mouth-lowerlip-volume': -0.07, 'mouth-angles': -0.04, 'mouth-upperlip-ext': -0.01, 'mouth-trans-backward-forward': 0.2, 'mouth-upperlip-width': -0.04, 'mouth-scale-vert': 0.06, 'mouth-dimples': 0.02, 'mouth-upperlip-height': -0.02, 'mouth-upperlip-volume': 0.02, 'mouth-cupidsbow-width': 0.03, 'mouth-upperlip-middle': 0.08, 'mouth-trans-down-up': 0.02, 'mouth-lowerlip-middle': 0.02, 'mouth-philtrum-volume': 0.03, 'chin-jaw-drop': 0.01, 'chin-prominent': 0.05, 'chin-bones': 0.18, 'chin-width': 0.04, 'chin-triangle': 0.04, 'chin-prognathism': 0.07, 'ear-lobe': 0.12, 'ear-shape-square-round': -0.13, 'ear-trans-backward-forward': 0.1, 'ear-rot': 0.03, 'ear-trans-down-up': 0.1, 'ear-scale-vert': 0.07, 'ear-flap': 0.05, 'ear-shape-pointed-triangle': -0.16, 'ear-scale': -0.09, 'ear-scale-depth': -0.18, 'ear-wing': 0.14, 'detail-1': 0.24, 'detail-2': -0.19, 'detail-3': -0.05, 'detail-4': -0.04, 'detail-5': -0.01, 'detail-6': 0.09, 'detail-7': -0.11, 'detail-8': -0.02, 'detail-9': -0.28, 'detail-10': 0.09 } } },
  { id: 'muse', name: 'Muse', description: 'A quiet figure lost in thought', pose: { base: 'relaxed', turns: [
      ['root', [0, 8, 4]], ['upperleg01.L', [0, 0, -4]], ['upperleg01.R', [6, 0, -2]], ['lowerleg01.R', [18, 0, 0]], ['foot.R', [-8, 0, 0]],
      ['spine05', [0, -4, -5]],
      // the right hand raised to the chin, the left arm across the waist
      ['upperarm01.R', [-40, 0, -10]], ['lowerarm01.R', [-120, 0, 0]], ['wrist.R', [-10, 0, 0]],
      ['upperarm01.L', [-14, 0, 6]], ['lowerarm01.L', [-70, -30, 0]],
      ['neck01', [10, 6, 0]], ['head', [12, 10, -6]]] },
    light: { format: 'anny-viewer/look@4', name: 'Muse', skin: { tone: 0.35, undertone: 0, stone: 'carrara' }, hair: { style: 'low_bun', color: STONE_HAIR['carrara'] }, eyes: { color: STONE_HAIR['carrara'] },
      phenotype: {gender: 1, age: 0.8, muscle: 0.42, weight: 0.45, height: 0.52, proportions: 0.6, african: 0, asian: 1, caucasian: 0},
      face: { 'head-scale-vert': 0.51, 'head-fat': -0.02, 'head-invertedtriangular': 0.03, 'head-back-scale-depth': -0.96, 'head-round': 0.02, 'head-scale-depth': 1.76, 'head-scale-horiz': 0.43, 'forehead-temple': -0.01, 'forehead-trans': 0.25, 'forehead-scale-vert': 0.06, 'forehead-nubian': -0.01, 'eyebrows-angle': -0.01, 'eyebrows-trans-backward-forward': 0.04, 'eye-corner1': 0.34, 'eye-height1': 0.1, 'eye-scale': 0.04, 'eye-push1': -0.01, 'eye-eyefold-down-up': 0.03, 'eye-height3': 0.16, 'eye-height2': 0.13, 'eye-bag-decr-incr': -0.01, 'eye-push2': -0.06, 'eye-trans-down-up': 0.18, 'eye-bag-height': 0.03, 'eye-epicanthus': 0.02, 'eye-eyefold-angle': -0.1, 'eye-bag-in-out': -0.03, 'eye-corner2': -0.26, 'eye-trans-in-out': -0.22, 'eye-eyefold-concave-convex': 0.02, 'nose-nostrils-angle': 0.04, 'nose-flaring': -0.11, 'nose-curve': -0.02, 'nose-scale-depth': 0.01, 'nose-scale-vert': 0.04, 'nose-point-width': -0.11, 'nose-scale-horiz': 0.07, 'nose-width1': -0.01, 'nose-trans-backward-forward': 0.08, 'nose-trans-down-up': -0.06, 'nose-hump': -0.01, 'nose-nostrils-width': 0.05, 'nose-greek': 0.03, 'nose-point': -0.04, 'nose-width2': -0.05, 'nose-width3': 0.03, 'nose-septumangle': 0.06, 'nose-base': -0.04, 'nose-compression': -0.03, 'nose-volume': 0.01, 'cheek-trans': 0.08, 'cheek-inner': -0.04, 'cheek-bones': -0.17, 'mouth-lowerlip-height': -0.01, 'mouth-scale-horiz': -0.08, 'mouth-cupidsbow': -0.01, 'mouth-laugh-lines': 0.01, 'mouth-scale-depth': -0.17, 'mouth-lowerlip-volume': -0.18, 'mouth-angles': 0.1, 'mouth-upperlip-ext': 0.01, 'mouth-lowerlip-width': -0.01, 'mouth-trans-backward-forward': 0.06, 'mouth-scale-vert': 0.04, 'mouth-dimples': 0.05, 'mouth-upperlip-height': 0.03, 'mouth-upperlip-volume': 0.03, 'mouth-cupidsbow-width': 0.03, 'mouth-lowerlip-ext': -0.06, 'mouth-upperlip-middle': -0.1, 'mouth-trans-down-up': -0.04, 'mouth-lowerlip-middle': -0.06, 'mouth-philtrum-volume': 0.04, 'chin-jaw-drop': 0.01, 'chin-height': 0.09, 'chin-prominent': 0.01, 'chin-bones': -0.01, 'chin-width': 0.02, 'chin-triangle': 0.02, 'chin-prognathism': 0.09, 'ear-lobe': 0.11, 'ear-shape-square-round': -0.04, 'ear-trans-backward-forward': -0.18, 'ear-rot': 0.01, 'ear-trans-down-up': 0.03, 'ear-scale-vert': 0.04, 'ear-flap': 0.02, 'ear-shape-pointed-triangle': 0.11, 'ear-scale': 0.14, 'ear-scale-depth': -0.06, 'ear-wing': 0.03, 'detail-1': 0.08, 'detail-2': -0.17, 'detail-3': -0.11, 'detail-4': 0.12, 'detail-5': 0.03, 'detail-6': -0.2, 'detail-7': 0.01, 'detail-8': -0.21, 'detail-9': -0.33, 'detail-10': 0.09 } } },
  { id: 'athlete', name: 'Athlete', description: 'A bronze figure at rest, the weight on one leg', statue: 'david_goliath', pose: { base: 'relaxed', turns: [
      // a victor tying the band of his head: the weight on the right leg, both arms raised to the head
      ['root', [0, -4, -4]], ['upperleg01.R', [0, 0, 4]], ['upperleg01.L', [10, 0, 2]], ['lowerleg01.L', [24, 0, 0]], ['foot.L', [-10, 0, 0]],
      ['spine05', [0, 4, 5]],
      ['upperarm01.L', [0, 0, 95]], ['lowerarm01.L', [0, 0, 70]], ['upperarm01.R', [0, 0, -95]], ['lowerarm01.R', [0, 0, -70]],
      ['neck01', [6, -10, 0]], ['head', [6, -12, 0]]] },
    light: { format: 'anny-viewer/look@4', name: 'Athlete', skin: { tone: 0.35, undertone: 0, stone: 'bronze' }, hair: { style: 'curly_top_taper', color: STONE_HAIR['bronze'] }, eyes: { color: STONE_HAIR['bronze'] },
      phenotype: {gender: 0, age: 0.82, muscle: 0.88, weight: 0.45, height: 0.58, proportions: 0.68, african: 1, asian: 0, caucasian: 0.4},
      face: {} } },
  { id: 'huntress', name: 'Huntress', description: 'A striding figure with the hair tied up', pose: { base: 'relaxed', turns: [
      // a stride forward on the right foot, the right hand reaching over the shoulder, the left arm forward and down
      ['upperleg01.R', [-20, 0, 0]], ['lowerleg01.R', [10, 0, 0]], ['upperleg01.L', [14, 0, 0]], ['lowerleg01.L', [22, 0, 0]], ['foot.L', [-14, 0, 0]],
      ['spine05', [0, -10, 0]],
      ['upperarm01.R', [-150, 0, -10]], ['lowerarm01.R', [-100, 0, 0]],
      ['upperarm01.L', [-30, 0, 10]], ['lowerarm01.L', [-20, 0, 0]],
      ['neck01', [0, 16, 0]], ['head', [0, 20, 0]]] },
    light: { format: 'anny-viewer/look@4', name: 'Huntress', skin: { tone: 0.35, undertone: 0, stone: 'travertine' }, hair: { style: 'top_knot', color: STONE_HAIR['travertine'] }, eyes: { color: STONE_HAIR['travertine'] },
      phenotype: {gender: 1, age: 0.82, muscle: 0.6, weight: 0.42, height: 0.62, proportions: 0.66, african: 0, asian: 1, caucasian: 1},
      face: { 'head-scale-vert': 0.34, 'head-fat': 0.07, 'head-invertedtriangular': 0.12, 'head-back-scale-depth': -0.82, 'head-round': 0.03, 'head-scale-depth': 1.48, 'head-scale-horiz': 0.21, 'forehead-temple': -0.23, 'forehead-trans': 0.05, 'forehead-scale-vert': 0.05, 'forehead-nubian': 0.01, 'eyebrows-angle': 0.01, 'eyebrows-trans-down-up': -0.02, 'eyebrows-trans-backward-forward': 0.07, 'eye-corner1': 0.12, 'eye-height1': -0.1, 'eye-scale': 0.24, 'eye-push1': 0.03, 'eye-eyefold-down-up': -0.04, 'eye-height3': -0.18, 'eye-height2': -0.14, 'eye-bag-decr-incr': -0.02, 'eye-push2': -0.04, 'eye-trans-down-up': 0.15, 'eye-bag-height': -0.04, 'eye-epicanthus': -0.01, 'eye-eyefold-angle': -0.06, 'eye-bag-in-out': -0.06, 'eye-corner2': -0.22, 'eye-trans-in-out': 0.01, 'eye-eyefold-concave-convex': 0.26, 'nose-nostrils-angle': -0.04, 'nose-flaring': -0.01, 'nose-curve': -0.07, 'nose-scale-depth': -0.03, 'nose-scale-vert': -0.1, 'nose-point-width': -0.06, 'nose-scale-horiz': -0.02, 'nose-width1': -0.04, 'nose-trans-backward-forward': 0.06, 'nose-trans-down-up': 0.02, 'nose-hump': -0.03, 'nose-nostrils-width': 0.09, 'nose-greek': -0.13, 'nose-point': -0.07, 'nose-width2': -0.03, 'nose-width3': -0.03, 'nose-septumangle': -0.11, 'nose-base': 0.16, 'nose-compression': 0.01, 'nose-volume': -0.14, 'cheek-trans': 0.04, 'cheek-volume': -0.08, 'cheek-inner': 0.07, 'cheek-bones': -0.12, 'mouth-lowerlip-height': -0.02, 'mouth-scale-horiz': -0.04, 'mouth-scale-depth': -0.01, 'mouth-lowerlip-volume': 0.12, 'mouth-angles': 0.1, 'mouth-lowerlip-width': 0.02, 'mouth-trans-backward-forward': -0.14, 'mouth-upperlip-width': 0.06, 'mouth-scale-vert': 0.01, 'mouth-dimples': -0.05, 'mouth-upperlip-height': 0.04, 'mouth-upperlip-volume': -0.08, 'mouth-lowerlip-ext': 0.02, 'mouth-upperlip-middle': 0.03, 'mouth-trans-down-up': 0.06, 'mouth-philtrum-volume': 0.03, 'chin-jaw-drop': -0.05, 'chin-height': -0.05, 'chin-prominent': -0.1, 'chin-width': -0.03, 'chin-prognathism': -0.17, 'ear-lobe': 0.02, 'ear-shape-square-round': 0.01, 'ear-trans-backward-forward': 0.12, 'ear-rot': -0.25, 'ear-trans-down-up': -0.11, 'ear-scale-vert': -0.01, 'ear-flap': 0.03, 'ear-shape-pointed-triangle': -0.1, 'ear-scale': 0.03, 'ear-scale-depth': -0.09, 'ear-wing': 0.16, 'detail-1': 0.06, 'detail-2': -0.16, 'detail-3': 0.12, 'detail-4': -0.04, 'detail-5': -0.22, 'detail-6': -0.38, 'detail-7': -0.03, 'detail-8': -0.21, 'detail-9': 0.14, 'detail-10': -0.2 } } },
  { id: 'guardian', name: 'Guardian', description: 'A broad figure standing at ease', pose: { base: 'relaxed', turns: [
      // a frontal stand, the feet apart, the arms crossed on the chest
      ['upperleg01.L', [0, 0, 5]], ['upperleg01.R', [0, 0, -5]],
      ['upperarm01.L', [-30, 0, 18]], ['lowerarm01.L', [-10, -100, 0]], ['upperarm01.R', [-30, 0, -18]], ['lowerarm01.R', [-10, 100, 0]],
      ['head', [-4, 0, 0]]] },
    light: { format: 'anny-viewer/look@4', name: 'Guardian', skin: { tone: 0.35, undertone: 0, stone: 'basalt' }, hair: { style: 'crew_cut', color: STONE_HAIR['basalt'] }, eyes: { color: STONE_HAIR['basalt'] },
      phenotype: {gender: 0.15, age: 0.88, muscle: 0.82, weight: 0.66, height: 0.7, proportions: 0.6, african: 0.6, asian: 0, caucasian: 0.6},
      face: {} } },
];

export const DEFAULT_STARTER = 'spear_bearer';
export const starterById = (id: string) => STARTERS.find((s) => s.id === id) || null;

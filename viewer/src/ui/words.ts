// OpenSculptBoy
// Apache License, Version 2.0
//
// Names joined into a sentence, as the interface's messages write them (Cambridge English: commas between the names
// and "and" before the last, with no comma before it). No DOM: node runs it (viewer/test/pose_edit.mjs).

// "a", "a and b", "a, b and c"; an empty list gives ''
export function listOf(names: string[], last = 'and'): string {
  if (names.length <= 1) return names[0] ?? '';
  return `${names.slice(0, -1).join(', ')} ${last} ${names[names.length - 1]}`;
}

// a sentence's first letter in capitals
export function capitalised(text: string): string { return text ? text[0].toUpperCase() + text.slice(1) : text; }

// the fingers of a hand ('thumb', 'index', 'middle', 'ring', 'little') as a sentence's subject, with the noun once:
// "the index finger", "the index and middle fingers", "the thumb and index finger", "the thumb, index and middle
// fingers"; plural tells the verb's form ("stays" or "stay"). No fingers give "a finger".
export const FINGER_WORDS: Record<string, string> = { thumb: 'thumb', index: 'index', middle: 'middle', ring: 'ring', little: 'little' };
export function fingersPhrase(fingers: string[]): { text: string; plural: boolean } {
  const order = Object.keys(FINGER_WORDS), list = order.filter((f) => fingers.includes(f));
  if (!list.length) return { text: 'a finger', plural: false };
  const others = list.filter((f) => f !== 'thumb').map((f) => FINGER_WORDS[f]);
  const words = list.includes('thumb') ? ['thumb', ...others] : others;
  const noun = others.length ? (others.length > 1 ? ' fingers' : ' finger') : '';
  return { text: `the ${listOf(words)}${noun}`, plural: list.length > 1 };
}

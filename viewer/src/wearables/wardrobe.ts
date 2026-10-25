// OpenSculptBoy
// Apache License, Version 2.0
//
// What a character wears, as opensculptboy.wearables.wardrobe.worn_pieces resolves it: the worn outfit's pieces as
// the card enables them, the enabled loose pieces, the card's overrides, then the page's hide and show lists. A piece
// shown from another outfit, and a loose piece, replace the pieces of the same slot that come from elsewhere (slot
// replacement; accessory pieces never replace). Last, the hides rules of the shown pieces hide the named groups of
// other shown pieces.

export const NO_OUTFIT = 'none';

export interface CardEntry { ref: string; piece: string; outfit: string | null; file: string; enabled: boolean }
export interface WornRef { ref: string; piece: string; outfit: string | null; file: string; shown: boolean; hiddenGroups: string[]; hideableGroups: string[] }
export interface Choices { outfit?: string | null; show?: string[]; hide?: string[] }
// the slot and the hides rules of each piece's package
export interface PieceInfo { slot: string; hides: { piece: string; groups: string[] }[] }

// the outfit that a card wears: its active_outfit, or null (no outfit) where that is empty or 'none', as the export
// reads it
export function cardOutfit(card: any): string | null {
  const active = card?.active_outfit;
  return typeof active === 'string' && active !== '' && active !== NO_OUTFIT ? active : null;
}

export function cardEntries(card: any): CardEntry[] {
  const out: CardEntry[] = [];
  for (const outfit of card.outfits || [])
    for (const p of outfit.pieces || []) out.push({ ref: `${outfit.id}/${p.piece}`, piece: p.piece, outfit: outfit.id, file: p.file, enabled: p.enabled !== false });
  for (const e of card.wearables || []) {
    const ref = e.id || String(e.file).split('/').pop()!.split('.')[0];
    out.push({ ref, piece: ref, outfit: null, file: e.file, enabled: e.enabled !== false });
  }
  return out;
}

// the reference that a name gives: itself, or <outfit>/<name> for a bare id of a piece of the worn outfit
export function resolveRef(name: string, refs: string[], outfit: string | null): string {
  if (refs.includes(name)) return name;
  if (outfit && refs.includes(`${outfit}/${name}`)) return `${outfit}/${name}`;
  const matches = refs.filter((r) => r.split('/').pop() === name);
  if (matches.length === 1) return matches[0];
  throw new Error(matches.length > 1 ? `Unknown piece ${name}: it names a piece of several outfits (${matches.join(', ')}).` : `Unknown piece ${name}.`);
}

// the pieces of a card and what shows; `every` also returns the pieces that do not show
export function wornPieces(card: any, info: (ref: string) => PieceInfo, choices: Choices = {}, every = false): WornRef[] {
  const entries = cardEntries(card), refs = entries.map((e) => e.ref);
  const outfitIds = (card.outfits || []).map((o: any) => o.id);
  let active: string | null = choices.outfit === undefined ? (card.active_outfit || null) : choices.outfit;
  if (active === NO_OUTFIT || active === '') active = null;
  if (active !== null && !outfitIds.includes(active)) throw new Error(`Unknown outfit ${active}; the outfits are ${outfitIds.join(', ')}.`);
  const shown: Record<string, boolean> = {};
  for (const e of entries) shown[e.ref] = e.outfit === null || e.outfit === active ? e.enabled : false;
  for (const [key, value] of Object.entries<any>(card.wearable_overrides || {})) shown[resolveRef(key, refs, active)] = !!value.enabled;
  for (const name of choices.hide || []) shown[resolveRef(name, refs, active)] = false;
  for (const name of choices.show || []) shown[resolveRef(name, refs, active)] = true;
  // slot replacement: a piece shown from another outfit (by the show list or by an override) and a loose piece replace
  // the pieces of their slot that come from elsewhere
  const replacers = entries.filter((e) => shown[e.ref] && (e.outfit === null || e.outfit !== active)).map((e) => e.ref);
  for (const ref of replacers) {
    const slot = info(ref).slot;
    if (slot === 'accessory') continue;
    const source = entries[refs.indexOf(ref)].outfit;
    for (const o of entries)
      if (o.ref !== ref && shown[o.ref] && o.outfit !== source && info(o.ref).slot === slot && !replacers.includes(o.ref)) shown[o.ref] = false;
  }
  const pieces: WornRef[] = entries.filter((e) => shown[e.ref] || every)
    .map((e) => ({ ref: e.ref, piece: e.piece, outfit: e.outfit, file: e.file, shown: !!shown[e.ref], hiddenGroups: [], hideableGroups: [] }));
  const byRef = Object.fromEntries(pieces.map((p) => [p.ref, p]));
  for (const p of pieces) {
    for (const rule of info(p.ref).hides) {
      let target = rule.piece;
      if (!target.includes('/') && p.outfit !== null) target = `${p.outfit}/${target}`;
      const other = byRef[target];
      if (!other) continue;
      for (const g of rule.groups || []) {
        if (!other.hideableGroups.includes(g)) other.hideableGroups.push(g);
        if (p.shown && other.shown && !other.hiddenGroups.includes(g)) other.hiddenGroups.push(g);
      }
    }
  }
  return pieces;
}

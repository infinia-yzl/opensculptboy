// OpenSculptBoy
// Apache License, Version 2.0
//
// Keys for one visit (docs/viewer.md, "AI providers"): where the authenticator has no PRF, the vault cannot encrypt
// keys, so the page keeps a pasted key in memory for the visit alone. Nothing here touches a store: a reload or the
// page's closing drops every key, and Forget drops one. The interface is the vault's own (`use`, `forget`), so the
// callers of a provider need not know which of the two holds the key.

import { cleanKey } from './vault.ts';

export interface SessionInfo { provider: string; added: number; checked?: number; ok?: boolean; note?: string }

export class SessionKeys {
  private keys = new Map<string, { key: string; info: SessionInfo }>();
  private listeners = new Set<(why: string) => void>();

  onChange(fn: (why: string) => void): () => void { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  private emit(why: string) { for (const fn of this.listeners) { try { fn(why); } catch (e) { /* a listener's fault stays its own */ } } }

  list(): SessionInfo[] { return [...this.keys.values()].map((k) => ({ ...k.info })).sort((a, b) => a.provider.localeCompare(b.provider)); }

  has(provider: string): boolean { return this.keys.has(provider); }

  set(provider: string, key: string): void {
    this.keys.set(provider, { key: cleanKey(key), info: { provider, added: Date.now() } });
    this.emit('save');
  }

  async use<T>(provider: string, fn: (key: string) => Promise<T>): Promise<T> {
    const k = this.keys.get(provider);
    if (!k) throw new Error('No key for ' + provider + ' in this visit. Paste it again.');
    return fn(k.key);
  }

  mark(provider: string, ok: boolean, note: string): void {
    const k = this.keys.get(provider);
    if (!k) return;
    Object.assign(k.info, { checked: Date.now(), ok, note: note.slice(0, 200) });
    this.emit('check');
  }

  forget(provider: string): void { this.keys.delete(provider); this.emit('forget'); }

  clear(): void { this.keys.clear(); this.emit('forget-all'); }
}

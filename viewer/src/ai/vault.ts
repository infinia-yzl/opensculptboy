// OpenSculptBoy
// Apache License, Version 2.0
//
// The key vault (docs/viewer.md, "AI providers"): API keys of AI providers, each encrypted with AES-GCM under one data
// key, which a passkey's PRF output unlocks. The vault has no interface of its own and no framework, so any interface
// can drive it: today's temporary panel (ui/ai_ui.ts) and the redesigned one.
//
// While the vault is locked, the page holds no key at all: the store has ciphertexts only. Unlock asks the passkey for
// its PRF output, derives the key-wrapping key, unwraps the data key as a non-extractable CryptoKey and forgets the
// PRF output. Lock, and an idle time without use, drop the data key. An API key is decrypted only for one call
// (`use`), and the vault never returns it in any other way.

import { aad, b64url, deriveKek, FORMAT, newDek, open, randomBytes, rewrapDek, seal, unwrapDek, type Bytes, type Sealed } from './crypto.ts';
import { PasskeyError, type PasskeyDriver } from './passkey.ts';
import type { VaultStore } from './store.ts';

export interface PasskeyRecord {
  format: string;
  credentialId: Bytes;
  salt: Bytes;
  wrapped: Sealed;
  created: number;
}

export interface KeyRecord {
  format: string;
  provider: string;
  sealed: Sealed;
  added: number;
  // the last check of the key: when, and its result in plain words (no secret)
  checked?: number;
  ok?: boolean;
  note?: string;
}

// what an interface may show of a stored key
export interface KeyInfo { provider: string; added: number; checked?: number; ok?: boolean; note?: string }

export interface VaultState { passkeys: number; unlocked: boolean; keys: KeyInfo[] }

export type VaultErrorCode = 'locked' | 'no-passkey' | 'has-passkey' | 'wrong-passkey' | 'no-key' | 'bad-key';

export class VaultError extends Error {
  code: VaultErrorCode;
  constructor(code: VaultErrorCode, message: string) { super(message); this.name = 'VaultError'; this.code = code; }
}

export interface Timers { now(): number; set(fn: () => void, ms: number): any; clear(handle: any): void }
const realTimers: Timers = { now: () => Date.now(), set: (fn, ms) => setTimeout(fn, ms), clear: (h) => clearTimeout(h) };

export interface VaultOptions {
  store: VaultStore;
  passkey: PasskeyDriver;
  // the vault locks itself after this time without use (default 10 minutes)
  idleMs?: number;
  timers?: Timers;
}

export const IDLE_MS = 10 * 60 * 1000;

// a key as people paste it: one line of printable characters without spaces
export function cleanKey(key: string): string {
  const k = (key || '').trim();
  if (!k) throw new VaultError('bad-key', 'The key is empty.');
  if (/\s/.test(k) || /[^\x21-\x7e]/.test(k)) throw new VaultError('bad-key', 'The key has spaces or other characters that keys never have. Copy it again.');
  if (k.length < 20 || k.length > 512) throw new VaultError('bad-key', 'The key is too ' + (k.length < 20 ? 'short' : 'long') + '. Copy it again.');
  return k;
}

export class KeyVault {
  private dek: CryptoKey | null = null;
  private timer: any = null;
  private listeners = new Set<(why: string) => void>();
  private store: VaultStore;
  private passkey: PasskeyDriver;
  private timers: Timers;
  readonly idleMs: number;

  constructor(o: VaultOptions) {
    this.store = o.store; this.passkey = o.passkey;
    this.timers = o.timers || realTimers;
    this.idleMs = o.idleMs ?? IDLE_MS;
  }

  // a listener for every change: setup, unlock, lock (with "idle" for an idle lock), save, check, forget
  onChange(fn: (why: string) => void): () => void { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  private emit(why: string) { for (const fn of this.listeners) { try { fn(why); } catch (e) { /* a listener's fault stays its own */ } } }

  unlocked(): boolean { return this.dek !== null; }

  async state(): Promise<VaultState> {
    const [passkeys, keys] = await Promise.all([this.store.all<PasskeyRecord>('passkeys'), this.store.all<KeyRecord>('keys')]);
    return {
      passkeys: passkeys.length, unlocked: this.unlocked(),
      keys: keys.map((k) => ({ provider: k.provider, added: k.added, checked: k.checked, ok: k.ok, note: k.note })).sort((a, b) => a.provider.localeCompare(b.provider)),
    };
  }

  // the first passkey: a new data key wrapped under it; the vault is open afterwards
  async setup(): Promise<void> {
    if ((await this.store.all('passkeys')).length) throw new VaultError('has-passkey', 'The vault has a passkey already. Unlock it, or forget everything to start again.');
    const salt = randomBytes(32);
    const { credentialId, prf } = await this.passkey.create(salt);
    const kek = await deriveKek(prf, salt);
    prf.fill(0);
    const { wrapped, dek } = await newDek(kek, aad.dek(credentialId));
    await this.store.put('passkeys', b64url(credentialId), { format: FORMAT, credentialId, salt, wrapped, created: this.timers.now() } satisfies PasskeyRecord);
    this.open(dek, 'setup');
  }

  // the data key from one of the vault's passkeys
  async unlock(): Promise<void> {
    const list = await this.store.all<PasskeyRecord>('passkeys');
    if (!list.length) throw new VaultError('no-passkey', 'The vault has no passkey yet. Set one up first.');
    const kek = await this.kekOf(list);
    this.open(kek.dek, 'unlock');
  }

  private async kekOf(list: PasskeyRecord[]): Promise<{ record: PasskeyRecord; kek: CryptoKey; dek: CryptoKey }> {
    const { credentialId, prf } = await this.passkey.evaluate(list.map((p) => ({ credentialId: p.credentialId, salt: p.salt })));
    const record = list.find((p) => b64url(p.credentialId) === b64url(credentialId));
    if (!record) { prf.fill(0); throw new VaultError('wrong-passkey', 'That passkey does not belong to this vault.'); }
    const kek = await deriveKek(prf, record.salt);
    prf.fill(0);
    try {
      return { record, kek, dek: await unwrapDek(record.wrapped, kek, aad.dek(record.credentialId)) };
    } catch (e) {
      throw new VaultError('wrong-passkey', 'The passkey did not open the vault: its answer does not match the stored record.');
    }
  }

  // another passkey for the same vault (a second device or a security key): asks a passkey of the vault, then creates
  // the new one
  async addPasskey(): Promise<void> {
    this.need();
    const list = await this.store.all<PasskeyRecord>('passkeys');
    const old = await this.kekOf(list);
    const salt = randomBytes(32);
    const { credentialId, prf } = await this.passkey.create(salt);
    const kek = await deriveKek(prf, salt);
    prf.fill(0);
    const wrapped = await rewrapDek(old.record.wrapped, old.kek, aad.dek(old.record.credentialId), kek, aad.dek(credentialId));
    await this.store.put('passkeys', b64url(credentialId), { format: FORMAT, credentialId, salt, wrapped, created: this.timers.now() } satisfies PasskeyRecord);
    this.touch(); this.emit('passkey');
  }

  lock(why = 'lock'): void {
    const was = this.dek !== null;
    this.dek = null;
    if (this.timer !== null) { this.timers.clear(this.timer); this.timer = null; }
    if (was) this.emit(why);
  }

  // use counts as activity: the idle time starts again
  touch(): void {
    if (!this.dek) return;
    if (this.timer !== null) this.timers.clear(this.timer);
    this.timer = this.timers.set(() => { this.timer = null; this.lock('idle'); }, this.idleMs);
  }

  private open(dek: CryptoKey, why: string) { this.dek = dek; this.touch(); this.emit(why); }

  private need(): CryptoKey {
    if (!this.dek) throw new VaultError('locked', 'The vault is locked. Unlock it with the passkey.');
    return this.dek;
  }

  // stores a provider's key, encrypted; replaces the key stored before
  async save(provider: string, key: string): Promise<void> {
    const dek = this.need();
    const sealed = await seal(dek, cleanKey(key), aad.key(provider));
    await this.store.put('keys', provider, { format: FORMAT, provider, sealed, added: this.timers.now() } satisfies KeyRecord);
    this.touch(); this.emit('save');
  }

  has(provider: string): Promise<boolean> { return this.store.get('keys', provider).then((r) => r !== undefined); }

  // the provider's key, decrypted for one call of fn; the key lives only in fn's arguments
  async use<T>(provider: string, fn: (key: string) => Promise<T>): Promise<T> {
    const dek = this.need();
    const rec = await this.store.get<KeyRecord>('keys', provider);
    if (!rec) throw new VaultError('no-key', 'No key is stored for ' + provider + '.');
    let key: string;
    try { key = await open(dek, rec.sealed, aad.key(provider)); } catch (e) {
      throw new VaultError('wrong-passkey', 'The stored key for ' + provider + ' does not decrypt. Forget it and add it again.');
    }
    this.touch();
    try { return await fn(key); } finally { key = ''; this.touch(); }
  }

  // the result of a key check, kept with the record (when, whether it worked, the words shown)
  async mark(provider: string, ok: boolean, note: string): Promise<void> {
    const rec = await this.store.get<KeyRecord>('keys', provider);
    if (!rec) return;
    await this.store.put('keys', provider, { ...rec, checked: this.timers.now(), ok, note: note.slice(0, 200) });
    this.emit('check');
  }

  // forgetting needs no passkey: the record goes, and nobody can decrypt what is gone
  async forget(provider: string): Promise<void> {
    await this.store.delete('keys', provider);
    this.emit('forget');
  }

  // every key and every passkey record; the vault locks, and the authenticator is asked to drop the passkeys
  async forgetAll(): Promise<void> {
    const list = await this.store.all<PasskeyRecord>('passkeys');
    await this.store.clear();
    this.lock('forget-all');
    for (const p of list) { try { await this.passkey.forget?.(p.credentialId); } catch (e) { /* best effort */ } }
    this.emit('forget-all');
  }
}

export { PasskeyError };

// OpenSculptBoy
// Apache License, Version 2.0
//
// Where the key vault keeps its records: IndexedDB in the page (database "opensculptboy-ai"), or memory in tests.
// The records hold ciphertexts, nonces, PRF salts and credential IDs; no record holds a key in the clear.
//
// Two tables: "passkeys" (one record per passkey: its credential ID, its PRF salt and the data key wrapped under it)
// and "keys" (one record per provider: the API key sealed under the data key).

export type Table = 'passkeys' | 'keys';

export interface VaultStore {
  get<T>(table: Table, id: string): Promise<T | undefined>;
  all<T>(table: Table): Promise<T[]>;
  put(table: Table, id: string, value: unknown): Promise<void>;
  delete(table: Table, id: string): Promise<void>;
  // every record of both tables
  clear(): Promise<void>;
}

const TABLES: Table[] = ['passkeys', 'keys'];

// a store in memory; values go through structuredClone as IndexedDB copies them
export function memoryStore(): VaultStore & { tables: Record<Table, Map<string, unknown>> } {
  const tables = { passkeys: new Map<string, unknown>(), keys: new Map<string, unknown>() };
  return {
    tables,
    async get(t, id) { const v = tables[t].get(id); return v === undefined ? undefined : structuredClone(v) as any; },
    async all(t) { return [...tables[t].values()].map((v) => structuredClone(v)) as any; },
    async put(t, id, value) { tables[t].set(id, structuredClone(value)); },
    async delete(t, id) { tables[t].delete(id); },
    async clear() { for (const t of TABLES) tables[t].clear(); },
  };
}

// a store in IndexedDB; each call opens its own transaction
export function idbStore(name = 'opensculptboy-ai'): VaultStore {
  let db: Promise<IDBDatabase> | null = null;
  const open = () => db || (db = new Promise((ok, bad) => {
    if (typeof indexedDB === 'undefined') { bad(new Error('IndexedDB is not available in this page')); return; }
    const r = indexedDB.open(name, 1);
    r.onupgradeneeded = () => { for (const t of TABLES) if (!r.result.objectStoreNames.contains(t)) r.result.createObjectStore(t); };
    r.onsuccess = () => {
      const d = r.result;
      // another tab deletes or upgrades the database: close this connection and open again on the next call
      d.onversionchange = () => { d.close(); db = null; };
      ok(d);
    };
    r.onerror = () => { db = null; bad(r.error); };
    r.onblocked = () => { db = null; bad(new Error('the key vault is open in an older tab; close that tab')); };
  }));
  const req = <T>(r: IDBRequest<T>) => new Promise<T>((ok, bad) => { r.onsuccess = () => ok(r.result); r.onerror = () => bad(r.error); });
  const done = (tx: IDBTransaction) => new Promise<void>((ok, bad) => { tx.oncomplete = () => ok(); tx.onerror = () => bad(tx.error); tx.onabort = () => bad(tx.error); });
  return {
    async get(t, id) { return req((await open()).transaction(t).objectStore(t).get(id)) as any; },
    async all(t) { return req((await open()).transaction(t).objectStore(t).getAll()) as any; },
    async put(t, id, value) { const tx = (await open()).transaction(t, 'readwrite'); tx.objectStore(t).put(value, id); await done(tx); },
    async delete(t, id) { const tx = (await open()).transaction(t, 'readwrite'); tx.objectStore(t).delete(id); await done(tx); },
    async clear() { const tx = (await open()).transaction(TABLES, 'readwrite'); for (const t of TABLES) tx.objectStore(t).clear(); await done(tx); },
  };
}

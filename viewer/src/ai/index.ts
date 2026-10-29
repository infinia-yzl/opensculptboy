// OpenSculptBoy
// Apache License, Version 2.0
//
// Connect an AI provider (docs/viewer.md, "AI providers"): the key vault, the passkey and the provider adapters, for
// any interface. pageVault() gives the page's one vault: records in IndexedDB, the passkey through WebAuthn, and a
// lock when the page goes away.

import { idbStore } from './store.ts';
import { webauthnPasskey } from './passkey.ts';
import { KeyVault } from './vault.ts';

export { KeyVault, VaultError, cleanKey, IDLE_MS, type KeyInfo, type VaultState } from './vault.ts';
export { pageSupport, PasskeyError, LOCAL_PORT, type Support } from './passkey.ts';
export { PROVIDERS, ProviderError, keyHint, type ChatMessage, type Provider, type ProviderId } from './providers.ts';
export { memoryStore, idbStore, type VaultStore } from './store.ts';

let shared: KeyVault | null = null;

export function pageVault(): KeyVault {
  if (shared) return shared;
  const v = new KeyVault({ store: idbStore(), passkey: webauthnPasskey() });
  // the data key never outlives the page, nor a stay in the back-forward cache
  addEventListener('pagehide', () => v.lock());
  shared = v;
  return v;
}

// OpenSculptBoy
// Apache License, Version 2.0
//
// The cryptography of the key vault (docs/viewer.md, "AI providers"), on WebCrypto alone, so that the page and node
// run the same code.
//
// A passkey's PRF output (32 bytes from the authenticator) goes through HKDF-SHA-256 to a key-wrapping key (the KEK).
// The KEK wraps one random AES-GCM data key (the DEK), and the DEK encrypts each API key. Both keys stay inside
// WebCrypto as non-extractable CryptoKey objects: the page can use them while the vault is open, and no script can
// read their bytes. Every ciphertext carries additional data that names what it holds, so a ciphertext copied to
// another record fails to decrypt.

// bytes on an ArrayBuffer of their own, as WebCrypto and WebAuthn take them
export type Bytes = Uint8Array<ArrayBuffer>;

export const FORMAT = 'opensculptboy/ai-vault@1';
const KEK_INFO = 'opensculptboy/ai-vault@1 key-wrapping key';

const enc = new TextEncoder();
const dec = new TextDecoder();
const subtle = () => {
  const s = globalThis.crypto && globalThis.crypto.subtle;
  if (!s) throw new Error('WebCrypto is not available in this page');
  return s;
};

export function randomBytes(n: number): Bytes {
  return globalThis.crypto.getRandomValues(new Uint8Array(n));
}

// a sealed value: the 12-byte nonce and the ciphertext with its 16-byte tag
export interface Sealed { iv: Bytes; data: Bytes }

// the additional data of each kind of record
export const aad = {
  key: (provider: string) => enc.encode(FORMAT + '|key|' + provider),
  dek: (credentialId: Bytes) => enc.encode(FORMAT + '|dek|' + b64url(credentialId)),
};

// the KEK from a PRF output; the PRF salt of the passkey is the HKDF salt as well
export async function deriveKek(prf: Bytes, salt: Bytes): Promise<CryptoKey> {
  if (prf.byteLength < 32) throw new Error('the passkey gave a PRF output of ' + prf.byteLength + ' bytes; 32 are needed');
  const s = subtle();
  const ikm = await s.importKey('raw', prf, 'HKDF', false, ['deriveKey']);
  return s.deriveKey({ name: 'HKDF', hash: 'SHA-256', salt, info: enc.encode(KEK_INFO) }, ikm,
    { name: 'AES-GCM', length: 256 }, false, ['wrapKey', 'unwrapKey']);
}

// a new DEK wrapped under a KEK, and a non-extractable copy of it to use
export async function newDek(kek: CryptoKey, extra: Bytes): Promise<{ wrapped: Sealed; dek: CryptoKey }> {
  const s = subtle();
  const fresh = await s.generateKey({ name: 'AES-GCM', length: 256 }, true, ['encrypt', 'decrypt']);
  const wrapped = await wrapDek(fresh, kek, extra);
  return { wrapped, dek: await unwrapDek(wrapped, kek, extra) };
}

// wrap a DEK under a KEK (the DEK must be extractable for this; only newDek holds such a copy, for one moment)
async function wrapDek(dek: CryptoKey, kek: CryptoKey, extra: Bytes): Promise<Sealed> {
  const iv = randomBytes(12);
  const data = new Uint8Array(await subtle().wrapKey('raw', dek, kek, { name: 'AES-GCM', iv, additionalData: extra }));
  return { iv, data };
}

// the DEK, non-extractable; throws when the KEK is another one (a wrong passkey) or the record was changed
export async function unwrapDek(wrapped: Sealed, kek: CryptoKey, extra: Bytes): Promise<CryptoKey> {
  return subtle().unwrapKey('raw', wrapped.data, kek, { name: 'AES-GCM', iv: wrapped.iv, additionalData: extra },
    { name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt']);
}

// the same DEK wrapped under another KEK (a second passkey): the DEK is unwrapped as an extractable key for this one
// step and wrapped again at once; only the wrapped copies leave this function
export async function rewrapDek(wrapped: Sealed, kek: CryptoKey, extra: Bytes, nextKek: CryptoKey, nextExtra: Bytes): Promise<Sealed> {
  const once = await subtle().unwrapKey('raw', wrapped.data, kek, { name: 'AES-GCM', iv: wrapped.iv, additionalData: extra },
    { name: 'AES-GCM', length: 256 }, true, ['encrypt', 'decrypt']);
  return wrapDek(once, nextKek, nextExtra);
}

export async function seal(dek: CryptoKey, text: string, extra: Bytes): Promise<Sealed> {
  const iv = randomBytes(12);
  const data = new Uint8Array(await subtle().encrypt({ name: 'AES-GCM', iv, additionalData: extra }, dek, enc.encode(text)));
  return { iv, data };
}
export async function open(dek: CryptoKey, sealed: Sealed, extra: Bytes): Promise<string> {
  const plain = new Uint8Array(await subtle().decrypt({ name: 'AES-GCM', iv: sealed.iv, additionalData: extra }, dek, sealed.data));
  const text = dec.decode(plain);
  plain.fill(0);
  return text;
}

export function b64url(bytes: Bytes): string {
  let s = '';
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

export function fromB64url(text: string): Bytes {
  const s = atob(text.replace(/-/g, '+').replace(/_/g, '/'));
  return Uint8Array.from(s, (c) => c.charCodeAt(0));
}

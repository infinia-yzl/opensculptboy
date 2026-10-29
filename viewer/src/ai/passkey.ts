// OpenSculptBoy
// Apache License, Version 2.0
//
// The passkey of the key vault: what the page needs for passkeys, and a driver that creates a passkey and gets its
// PRF output through WebAuthn's "prf" extension (docs/viewer.md, "AI providers").
//
// The PRF output is a secret of 32 bytes that the authenticator computes from the passkey's own secret and a salt.
// The authenticator gives it only after the person confirms with a fingerprint, a face, a PIN or a security key, and
// the same passkey with the same salt always gives the same output. No server checks the passkey's signature here:
// the vault's protection comes from the PRF output alone.

import { b64url, type Bytes } from './crypto.ts';
import serve from './serve.json' with { type: 'json' };

export type SupportCode = 'ok' | 'file-page' | 'ip-address' | 'insecure' | 'no-webauthn';

export interface Support {
  ok: boolean;
  code: SupportCode;
  message: string;
  // a command that serves the page at a proper origin, when the page can name one
  command?: string;
  url?: string;
}

// the port of the served page, from serve.json, which serve_viewer.py reads too: one fixed port, because a page's
// storage belongs to its origin, port included. 41730 is unassigned in IANA's registry and is no common tool's default.
export const LOCAL_PORT: number = serve.port;
export const SERVE_COMMAND = 'python serve_viewer.py --open';

interface Where { protocol: string; hostname: string; port: string; pathname: string; search?: string }

// what the page needs for passkeys: a secure context with a domain name (https, or http://localhost)
export function pageSupport(where: Where = globalThis.location, scope: any = globalThis): Support {
  const file = decodeURIComponent((where.pathname || '').split('/').pop() || '') || 'anny_viewer.html';
  const local = `http://localhost:${LOCAL_PORT}/${encodeURIComponent(file)}`;
  if (where.protocol === 'file:') {
    return { ok: false, code: 'file-page', url: local, command: SERVE_COMMAND,
      message: 'Passkeys do not work in a page opened as a file. Run this command in the page\'s folder (serve_viewer.py comes with the page in viewer/dist), and it opens ' + local + '.' };
  }
  if (/^(\d+\.){3}\d+$|^\[?[0-9a-f]*:[0-9a-f:.]*\]?$/i.test(where.hostname)) {
    const url = `${where.protocol}//localhost${where.port ? ':' + where.port : ''}${where.pathname}${where.search || ''}`;
    return { ok: false, code: 'ip-address', url, message: 'Passkeys need a name for the computer. Open ' + url + ' instead.' };
  }
  if (!scope.isSecureContext) {
    return { ok: false, code: 'insecure', message: 'Passkeys need a secure page: https, or http://localhost on this computer.' };
  }
  if (typeof scope.PublicKeyCredential === 'undefined' || !scope.navigator?.credentials?.create) {
    return { ok: false, code: 'no-webauthn', message: 'This browser has no passkeys. Use a current Chrome, Edge, Safari or Firefox.' };
  }
  return { ok: true, code: 'ok', message: '' };
}

export type PasskeyErrorCode = 'no-prf' | 'cancelled' | 'exists' | 'unsupported' | 'failed';

export class PasskeyError extends Error {
  code: PasskeyErrorCode;
  constructor(code: PasskeyErrorCode, message: string) { super(message); this.name = 'PasskeyError'; this.code = code; }
}

export const NO_PRF = 'This passkey cannot make an encryption key (it has no PRF support), so it cannot lock the vault. '
  + 'Use a passkey from Windows Hello on Windows 11 with a current browser, from iCloud Keychain, from Google Password Manager, or a recent security key.';

export interface PasskeyResult { credentialId: Bytes; prf: Bytes }

// the vault's view of a passkey; tests give a fake one
export interface PasskeyDriver {
  // a new passkey and its PRF output for this salt
  create(salt: Bytes): Promise<PasskeyResult>;
  // the PRF output of one of these passkeys, each with its own salt
  evaluate(list: { credentialId: Bytes; salt: Bytes }[]): Promise<PasskeyResult>;
  // asks the authenticator to remove a passkey that the vault no longer uses (where the browser can)
  forget?(credentialId: Bytes): Promise<void>;
}

function translate(e: any): PasskeyError {
  if (e instanceof PasskeyError) return e;
  const name = e && e.name;
  if (name === 'NotAllowedError' || name === 'AbortError') return new PasskeyError('cancelled', 'The passkey prompt was closed or timed out.');
  if (name === 'InvalidStateError') return new PasskeyError('exists', 'This authenticator holds the vault\'s passkey already.');
  if (name === 'SecurityError' || name === 'NotSupportedError') return new PasskeyError('unsupported', pageSupport().message || 'This page cannot use passkeys.');
  return new PasskeyError('failed', 'The passkey did not answer: ' + (e && e.message ? e.message : String(e)));
}

const bytes = (b: ArrayBuffer | ArrayBufferView) => new Uint8Array(b instanceof ArrayBuffer ? b.slice(0) : (b as any).buffer.slice((b as any).byteOffset, (b as any).byteOffset + (b as any).byteLength));

// the driver of the page: WebAuthn with the PRF extension
export function webauthnPasskey(rpName = 'OpenSculptBoy'): PasskeyDriver {
  const random = (n: number) => crypto.getRandomValues(new Uint8Array(n));
  const driver: PasskeyDriver = {
    async create(salt) {
      let cred: any;
      try {
        cred = await navigator.credentials.create({ publicKey: {
          rp: { name: rpName },
          user: { id: random(16), name: 'OpenSculptBoy AI keys', displayName: 'OpenSculptBoy AI keys' },
          challenge: random(32),
          pubKeyCredParams: [{ type: 'public-key', alg: -7 }, { type: 'public-key', alg: -257 }],
          authenticatorSelection: { residentKey: 'preferred', userVerification: 'required' },
          timeout: 120000,
          extensions: { prf: { eval: { first: salt } } } as any,
        } });
      } catch (e) { throw translate(e); }
      if (!cred) throw new PasskeyError('cancelled', 'The passkey prompt was closed or timed out.');
      const credentialId = bytes(cred.rawId);
      const prf = cred.getClientExtensionResults()?.prf;
      if (prf?.results?.first) return { credentialId, prf: bytes(prf.results.first) };
      // some authenticators give the PRF output only when the passkey signs in: one more prompt
      if (prf?.enabled !== false) {
        try { return await driver.evaluate([{ credentialId, salt }]); } catch (e) { if (!(e instanceof PasskeyError && e.code === 'no-prf')) throw e; }
      }
      await driver.forget?.(credentialId);
      throw new PasskeyError('no-prf', NO_PRF);
    },
    async evaluate(list) {
      const evalByCredential: Record<string, { first: Bytes }> = {};
      for (const c of list) evalByCredential[b64url(c.credentialId)] = { first: c.salt };
      let a: any;
      try {
        a = await navigator.credentials.get({ publicKey: {
          challenge: random(32),
          allowCredentials: list.map((c) => ({ type: 'public-key', id: c.credentialId })),
          userVerification: 'required',
          timeout: 120000,
          extensions: { prf: { evalByCredential } } as any,
        } });
      } catch (e) { throw translate(e); }
      if (!a) throw new PasskeyError('cancelled', 'The passkey prompt was closed or timed out.');
      const first = a.getClientExtensionResults()?.prf?.results?.first;
      if (!first) throw new PasskeyError('no-prf', NO_PRF);
      return { credentialId: bytes(a.rawId), prf: bytes(first) };
    },
    async forget(credentialId) {
      const signal = (globalThis as any).PublicKeyCredential?.signalUnknownCredential;
      if (!signal) return;
      try { await signal.call((globalThis as any).PublicKeyCredential, { rpId: location.hostname, credentialId: b64url(credentialId) }); } catch (e) { /* the browser may refuse; the passkey then stays unused */ }
    },
  };
  return driver;
}

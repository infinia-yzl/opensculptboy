// OpenSculptBoy
// Apache License, Version 2.0
//
// The key vault and the provider adapters of viewer/src/ai/ without a browser (run by test/test_viewer_ai.py):
//   node viewer/test/ai_vault.mjs
// A fake passkey stands in for WebAuthn: its PRF output is HMAC-SHA-256 of a secret of its own and the salt, as an
// authenticator's is. A fake fetch stands in for the providers. It checks the cryptography, the records in the store
// (no key in the clear), lock, idle lock, unlock, a wrong passkey, a second passkey, forgetting, the page checks, and
// each provider's requests and errors; a failed check throws, so node exits with an error. It prints a JSON summary.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { aad, b64url, deriveKek, fromB64url, newDek, open, randomBytes, seal, unwrapDek } from '../src/ai/crypto.ts';
import { memoryStore } from '../src/ai/store.ts';
import { KeyVault, VaultError, cleanKey } from '../src/ai/vault.ts';
import { NO_PRF, PasskeyError, pageSupport } from '../src/ai/passkey.ts';
import { PROVIDERS, ProviderError, keyHint, scrub } from '../src/ai/providers.ts';

const here = path.dirname(fileURLToPath(import.meta.url));
const subtle = globalThis.crypto.subtle;
const checks = [];
const ok = (name) => checks.push(name);
async function rejects(p, code, name) {
  await assert.rejects(p, (e) => { assert.equal(e.code, code, name + ': ' + e.message); return true; });
  ok(name);
}

// ---------------------------------------------------------------- a fake passkey
async function hmac(secret, salt) {
  const k = await subtle.importKey('raw', secret, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  return new Uint8Array(await subtle.sign('HMAC', k, salt));
}
function fakeAuthenticator() {
  const creds = new Map(); // b64url id -> secret
  const a = { creds, prompts: 0, forgotten: [], choose: null, prf: true };
  a.driver = {
    async create(salt) {
      a.prompts++;
      if (!a.prf) throw new PasskeyError('no-prf', NO_PRF);
      const id = randomBytes(16), secret = randomBytes(32);
      creds.set(b64url(id), secret);
      return { credentialId: id, prf: await hmac(secret, salt) };
    },
    async evaluate(list) {
      a.prompts++;
      const mine = list.filter((c) => creds.has(b64url(c.credentialId)));
      const c = a.choose ? mine.find((x) => b64url(x.credentialId) === a.choose) : mine[0];
      if (!c) throw new PasskeyError('cancelled', 'The passkey prompt was closed or timed out.');
      return { credentialId: c.credentialId, prf: await hmac(creds.get(b64url(c.credentialId)), c.salt) };
    },
    async forget(id) { a.forgotten.push(b64url(id)); creds.delete(b64url(id)); },
  };
  return a;
}
function fakeTimers() {
  const t = { at: 1000, jobs: new Map(), next: 1 };
  t.now = () => t.at;
  t.set = (fn, ms) => { const h = t.next++; t.jobs.set(h, { fn, due: t.at + ms }); return h; };
  t.clear = (h) => t.jobs.delete(h);
  t.advance = (ms) => { t.at += ms; for (const [h, j] of [...t.jobs]) if (j.due <= t.at) { t.jobs.delete(h); j.fn(); } };
  return t;
}
// every byte string of a record, so that a test can look for a key inside it
function haystack(value) {
  const parts = [];
  const walk = (v) => {
    if (v instanceof Uint8Array) parts.push(Buffer.from(v).toString('latin1'));
    else if (v && typeof v === 'object') for (const x of Object.values(v)) walk(x);
    else parts.push(String(v));
  };
  walk(value);
  return parts.join('\n');
}

// ---------------------------------------------------------------- the cryptography
{
  const prf = randomBytes(32), salt = randomBytes(32), id = randomBytes(16);
  const kek = await deriveKek(prf, salt);
  const { wrapped, dek } = await newDek(kek, aad.dek(id));
  assert.equal(dek.extractable, false);
  assert.deepEqual(dek.usages.sort(), ['decrypt', 'encrypt']);
  assert.equal(kek.extractable, false);
  await assert.rejects(subtle.exportKey('raw', dek));
  ok('the data key and the key-wrapping key are non-extractable');
  // the same PRF output and salt give the same KEK: the data key unwraps again and opens what the first copy sealed
  const again = await unwrapDek(wrapped, await deriveKek(prf, salt), aad.dek(id));
  const s = await seal(dek, 'sk-ant-api03-example-key-0123456789', aad.key('anthropic'));
  assert.equal(await open(again, s, aad.key('anthropic')), 'sk-ant-api03-example-key-0123456789');
  ok('one PRF output and salt give one key-wrapping key');
  // another PRF output, another salt or another credential fails
  await assert.rejects(unwrapDek(wrapped, await deriveKek(randomBytes(32), salt), aad.dek(id)));
  await assert.rejects(unwrapDek(wrapped, await deriveKek(prf, randomBytes(32)), aad.dek(id)));
  await assert.rejects(unwrapDek(wrapped, kek, aad.dek(randomBytes(16))));
  ok('a wrong PRF output, salt or credential does not unwrap');
  // the additional data binds a ciphertext to its provider, and the tag guards every byte
  await assert.rejects(open(dek, s, aad.key('openai')));
  const bent = { iv: s.iv, data: s.data.slice() }; bent.data[3] ^= 1;
  await assert.rejects(open(dek, bent, aad.key('anthropic')));
  const s2 = await seal(dek, 'sk-ant-api03-example-key-0123456789', aad.key('anthropic'));
  assert.notDeepEqual(s2.iv, s.iv); assert.notDeepEqual(s2.data, s.data);
  assert.equal(s.iv.length, 12); assert.equal(s.data.length, 'sk-ant-api03-example-key-0123456789'.length + 16);
  ok('a ciphertext opens only for its provider, a changed byte fails, each seal has a new nonce');
  await assert.rejects(deriveKek(randomBytes(16), salt));
  for (const n of [0, 1, 15, 16, 31, 32, 33]) { const b = randomBytes(n); assert.deepEqual(fromB64url(b64url(b)), b); }
  ok('short PRF outputs are refused; base64url round trips');
}

// ---------------------------------------------------------------- the vault
const KEY_A = 'sk-ant-api03-' + 'A1b2C3d4'.repeat(10);
const KEY_O = 'sk-proj-' + 'Z9y8X7w6'.repeat(8);
{
  const store = memoryStore(), auth = fakeAuthenticator(), timers = fakeTimers();
  const v = new KeyVault({ store, passkey: auth.driver, timers, idleMs: 60000 });
  const events = [];
  v.onChange((why) => events.push(why));
  assert.deepEqual(await v.state(), { passkeys: 0, unlocked: false, keys: [] });
  await rejects(v.save('anthropic', KEY_A), 'locked', 'a locked vault refuses a key');
  await rejects(v.unlock(), 'no-passkey', 'unlock without a passkey');
  await v.setup();
  assert.equal(v.unlocked(), true);
  await rejects(v.setup(), 'has-passkey', 'a second setup is refused');
  await v.save('anthropic', '  ' + KEY_A + '\n');
  await v.save('openai', KEY_O);
  const st = await v.state();
  assert.deepEqual(st.keys.map((k) => k.provider), ['anthropic', 'openai']);
  assert.equal(st.passkeys, 1);
  assert.equal(await v.use('anthropic', async (k) => k), KEY_A);
  ok('setup, save (trimmed) and use');
  // the store holds no key, no PRF output and no raw data key
  const all = haystack([...store.tables.keys.values(), ...store.tables.passkeys.values()]);
  for (const k of [KEY_A, KEY_O, KEY_A.slice(13, 33), KEY_O.slice(8, 28)]) assert.ok(!all.includes(k), 'a key in the store');
  for (const r of store.tables.keys.values()) assert.deepEqual(Object.keys(r).sort(), ['added', 'format', 'provider', 'sealed']);
  for (const r of store.tables.passkeys.values()) assert.deepEqual(Object.keys(r).sort(), ['created', 'credentialId', 'format', 'salt', 'wrapped']);
  ok('the store holds ciphertexts, salts and credential IDs only');
  // lock, then unlock with the passkey
  v.lock();
  assert.equal(v.unlocked(), false);
  await rejects(v.use('anthropic', async (k) => k), 'locked', 'a locked vault decrypts nothing');
  const before = auth.prompts;
  await v.unlock();
  assert.equal(auth.prompts, before + 1);
  assert.equal(await v.use('openai', async (k) => k), KEY_O);
  ok('lock and unlock');
  // the idle lock: use restarts the time
  timers.advance(59000);
  assert.equal(v.unlocked(), true);
  await v.use('openai', async () => 1);
  timers.advance(59000);
  assert.equal(v.unlocked(), true);
  timers.advance(1500);
  assert.equal(v.unlocked(), false);
  assert.equal(events.at(-1), 'idle');
  assert.equal(timers.jobs.size, 0);
  ok('the vault locks itself after the idle time, and use restarts it');
  // a passkey of another vault, and a record changed in the store
  await v.unlock();
  v.lock();
  const rec = [...store.tables.passkeys.values()][0];
  const saved = structuredClone(rec);
  rec.wrapped.data[0] ^= 1;
  await rejects(v.unlock(), 'wrong-passkey', 'a changed passkey record does not unlock');
  store.tables.passkeys.set(b64url(saved.credentialId), saved);
  const other = fakeAuthenticator();
  const v2 = new KeyVault({ store, passkey: other.driver, timers });
  await rejects(v2.unlock(), 'cancelled', 'an authenticator without the passkey');
  // a key record moved to another provider does not decrypt
  await v.unlock();
  const a = store.tables.keys.get('anthropic');
  store.tables.keys.set('openrouter', { ...structuredClone(a), provider: 'openrouter' });
  await rejects(v.use('openrouter', async (k) => k), 'wrong-passkey', 'a record moved to another provider');
  await v.forget('openrouter');
  await rejects(v.use('openrouter', async (k) => k), 'no-key', 'a forgotten key');
  // a second passkey opens the same vault
  await v.addPasskey();
  assert.equal((await v.state()).passkeys, 2);
  v.lock();
  auth.choose = [...auth.creds.keys()][1];
  await v.unlock();
  assert.equal(await v.use('anthropic', async (k) => k), KEY_A);
  auth.choose = null;
  ok('a second passkey opens the same vault');
  // check results stay with the record; forget one key, then everything
  await v.mark('anthropic', true, 'The key works: 9 models.');
  assert.deepEqual((await v.state()).keys.find((k) => k.provider === 'anthropic').ok, true);
  await v.forget('openai');
  assert.deepEqual((await v.state()).keys.map((k) => k.provider), ['anthropic']);
  await v.forgetAll();
  assert.deepEqual(await v.state(), { passkeys: 0, unlocked: false, keys: [] });
  assert.equal(auth.forgotten.length, 2);
  assert.ok(events.includes('forget-all'));
  ok('mark, forget and forget all (the authenticator is asked to drop both passkeys)');
  // cleanKey
  for (const bad of ['', '   ', 'sk-ant key with spaces 0123456789', 'short', 'sk-ant-\u00e9' + 'x'.repeat(30)]) assert.throws(() => cleanKey(bad), VaultError);
  ok('keys with spaces, other characters or a wrong length are refused');
  // an authenticator without PRF leaves no record
  const noPrf = fakeAuthenticator(); noPrf.prf = false;
  const v3 = new KeyVault({ store: memoryStore(), passkey: noPrf.driver, timers });
  await rejects(v3.setup(), 'no-prf', 'an authenticator without PRF');
  assert.equal((await v3.state()).passkeys, 0);
}

// ---------------------------------------------------------------- the page checks
{
  const yes = { isSecureContext: true, PublicKeyCredential: function () {}, navigator: { credentials: { create() {} } } };
  const at = (url) => { const u = new URL(url); return { protocol: u.protocol, hostname: u.hostname.replace(/^\[|\]$/g, ''), port: u.port, pathname: u.pathname, search: u.search }; };
  const f = pageSupport(at('file:///srv/My%20Viewer/anny_viewer.html'), { ...yes });
  assert.equal(f.code, 'file-page');
  assert.equal(f.url, 'http://localhost:8765/anny_viewer.html');
  assert.equal(f.command, 'python -m http.server 8765 --bind 127.0.0.1');
  assert.equal(pageSupport(at('http://127.0.0.1:8765/anny_viewer.html?acc=1'), yes).url, 'http://localhost:8765/anny_viewer.html?acc=1');
  assert.equal(pageSupport(at('http://[::1]:8765/p.html'), yes).code, 'ip-address');
  assert.equal(pageSupport(at('http://cafe.example/p.html'), { ...yes, isSecureContext: false }).code, 'insecure');
  assert.equal(pageSupport(at('https://beef/p.html'), { isSecureContext: true, navigator: {} }).code, 'no-webauthn');
  assert.equal(pageSupport(at('http://localhost:8765/anny_viewer.html'), yes).ok, true);
  assert.equal(pageSupport(at('https://example.github.io/opensculptboy/'), yes).ok, true);
  ok('the page checks: file page, an address in numbers, an insecure page, no WebAuthn, localhost, https');
}

// ---------------------------------------------------------------- the providers, with a fake fetch
function fakeFetch(answer) {
  const calls = [];
  const f = async (url, init) => {
    calls.push({ url, init, headers: Object.fromEntries(Object.entries(init.headers || {}).map(([k, v]) => [k.toLowerCase(), v])), body: init.body ? JSON.parse(init.body) : null });
    const a = typeof answer === 'function' ? await answer(url, init) : answer;
    if (a instanceof Error) throw a;
    return new Response(typeof a.body === 'string' ? a.body : JSON.stringify(a.body), { status: a.status || 200, headers: { 'content-type': 'application/json' } });
  };
  f.calls = calls;
  return f;
}
const MSG = [{ role: 'system', content: 'Be brief.' }, { role: 'user', content: 'Name a colour.' }];
{
  const p = PROVIDERS.anthropic;
  let f = fakeFetch({ body: { data: [{ id: 'claude-haiku-5-5' }, { id: 'claude-sonnet-5-5' }] } });
  const r = await p.check(KEY_A, { fetch: f });
  assert.deepEqual([r.ok, r.models], [true, 2]);
  const c = f.calls[0];
  assert.equal(c.url, 'https://api.anthropic.com/v1/models?limit=100');
  assert.equal(c.headers['x-api-key'], KEY_A);
  assert.equal(c.headers['anthropic-version'], '2023-06-01');
  assert.equal(c.headers['anthropic-dangerous-direct-browser-access'], 'true');
  assert.equal(c.init.credentials, 'omit');
  assert.ok(!c.url.includes(KEY_A));
  f = fakeFetch({ body: { content: [{ type: 'text', text: 'Teal' }, { type: 'text', text: '.' }] } });
  assert.equal(await p.complete(KEY_A, MSG, { fetch: f }), 'Teal.');
  const b = f.calls[0].body;
  assert.equal(f.calls[0].url, 'https://api.anthropic.com/v1/messages');
  assert.equal(b.system, 'Be brief.');
  assert.deepEqual(b.messages, [{ role: 'user', content: 'Name a colour.' }]);
  assert.equal(b.model, 'claude-haiku-5-5');
  assert.ok(b.max_tokens > 0);
  ok('Anthropic: the model list with the browser header, and a message with its system text apart');
}
{
  const p = PROVIDERS.openai;
  let f = fakeFetch({ body: { data: [{ id: 'a' }, { id: 'b' }, { id: 'c' }] } });
  assert.equal((await p.check(KEY_O, { fetch: f })).models, 3);
  assert.equal(f.calls[0].url, 'https://api.openai.com/v1/models');
  assert.equal(f.calls[0].headers.authorization, 'Bearer ' + KEY_O);
  f = fakeFetch({ body: { choices: [{ message: { role: 'assistant', content: 'Blue' } }] } });
  assert.equal(await p.complete(KEY_O, MSG, { fetch: f, model: 'gpt-x' }), 'Blue');
  assert.equal(f.calls[0].url, 'https://api.openai.com/v1/chat/completions');
  assert.deepEqual(f.calls[0].body.messages, MSG);
  assert.equal(f.calls[0].body.model, 'gpt-x');
  ok('OpenAI: the model list, and a chat completion');
}
{
  const p = PROVIDERS.openrouter, KEY_R = 'sk-or-v1-' + 'abcdef01'.repeat(8);
  let f = fakeFetch({ body: { data: { label: 'x', limit_remaining: 4.5 } } });
  const r = await p.check(KEY_R, { fetch: f });
  assert.equal(f.calls[0].url, 'https://openrouter.ai/api/v1/key');
  assert.match(r.detail, /4\.50/);
  f = fakeFetch({ body: { choices: [{ message: { content: 'Red' } }] } });
  assert.equal(await p.complete(KEY_R, MSG, { fetch: f }), 'Red');
  assert.equal(f.calls[0].body.model, 'openrouter/auto');
  ok('OpenRouter: the key record, and a chat completion');
}
{
  // errors in plain words, without the key, even when the provider repeats it
  const p = PROVIDERS.openai;
  const cases = [
    [401, { error: { message: 'Incorrect API key provided: ' + KEY_O + '.' } }, 'key'],
    [402, { error: { message: 'Insufficient credits' } }, 'credit'],
    [429, { error: { message: 'Rate limit reached' } }, 'rate'],
    [404, { error: { message: 'The model does not exist' } }, 'model'],
    [400, { error: { message: 'Bad request' } }, 'request'],
    [529, { error: { message: 'Overloaded' } }, 'busy'],
  ];
  for (const [status, body, code] of cases) {
    const f = fakeFetch({ status, body });
    await assert.rejects(p.complete(KEY_O, MSG, { fetch: f }), (e) => {
      assert.ok(e instanceof ProviderError); assert.equal(e.code, code); assert.equal(e.status, status);
      assert.ok(!e.message.includes(KEY_O) && !e.message.includes(KEY_O.slice(8, 24)), e.message);
      assert.ok(e.message.includes('OpenAI '), e.message);
      return true;
    });
  }
  assert.equal(scrub('bad key sk-proj-abc123def456ghi and sk-ant-xyz', ''), 'bad key [key] and sk-ant-xyz');
  await rejects(p.check(KEY_O, { fetch: fakeFetch(new TypeError('Failed to fetch')) }), 'network', 'a network error');
  await rejects(p.check(KEY_O, { fetch: fakeFetch({ body: '<html>' }) }), 'reply', 'a reply that is not JSON');
  await rejects(p.check(KEY_O, { timeoutMs: 30, fetch: (u, init) => new Promise((_, bad) => init.signal.addEventListener('abort', () => bad(new DOMException('aborted', 'AbortError')))) }), 'timeout', 'a provider that does not answer');
  await rejects(PROVIDERS.anthropic.complete(KEY_A, MSG, { fetch: fakeFetch({ body: { content: [] } }) }), 'reply', 'a reply without text');
  ok('errors in plain words, with the key taken out');
  assert.equal(keyHint('openai', KEY_A), 'This looks like an Anthropic key.');
  assert.equal(keyHint('anthropic', KEY_A), '');
  assert.equal(keyHint('anthropic', 'sk-or-v1-abc'), 'This looks like an OpenRouter key.');
}

// ---------------------------------------------------------------- the sources: no log, no web storage, no URL
{
  const dir = path.join(here, '..', 'src', 'ai');
  for (const name of fs.readdirSync(dir).filter((n) => n.endsWith('.ts'))) {
    const code = fs.readFileSync(path.join(dir, name), 'utf8').replace(/\/\/.*$/gm, '');
    for (const word of ['console.', 'localStorage', 'sessionStorage', 'location.hash', 'history.', 'document.cookie', 'postMessage'])
      assert.ok(!code.includes(word), name + ' uses ' + word);
  }
  ok('the vault and the adapters use no console, web storage, cookie, URL or message channel');
}

console.log(JSON.stringify({ ok: true, checks }));

// OpenSculptBoy
// Apache License, Version 2.0
//
// The AI providers that the page calls straight from the browser with a person's own API key (docs/viewer.md,
// "AI providers"): Anthropic, OpenAI and OpenRouter. Each adapter has a key check, which calls a cheap endpoint that
// needs the key, and one plain completion call. Errors come out as ProviderError with a message in plain words; no
// message, error or log line carries the key.
//
// Browser rules (CORS), measured on 2026-10-29: the three APIs answer a browser's preflight with
// Access-Control-Allow-Origin for any origin. Anthropic serves browser calls only with the header
// anthropic-dangerous-direct-browser-access: true.

export type ProviderId = 'anthropic' | 'openai' | 'openrouter';

export interface ChatMessage { role: 'system' | 'user' | 'assistant'; content: string }

export interface CallOptions {
  fetch?: typeof fetch;
  signal?: AbortSignal;
  // milliseconds before the call gives up (default 20 s for a check, 120 s for a completion)
  timeoutMs?: number;
}

export interface CompleteOptions extends CallOptions { model?: string; maxTokens?: number }

export interface CheckResult { ok: true; models: number; detail: string }

export interface Provider {
  id: ProviderId;
  label: string;
  // where a person makes a key
  keysUrl: string;
  // the start of the provider's keys, for a hint when a key of another provider is pasted
  prefix: string;
  defaultModel: string;
  check(key: string, o?: CallOptions): Promise<CheckResult>;
  complete(key: string, messages: ChatMessage[], o?: CompleteOptions): Promise<string>;
}

export type ProviderErrorCode = 'key' | 'credit' | 'rate' | 'model' | 'request' | 'busy' | 'network' | 'timeout' | 'reply';

export class ProviderError extends Error {
  code: ProviderErrorCode;
  status: number;
  constructor(code: ProviderErrorCode, status: number, message: string) { super(message); this.name = 'ProviderError'; this.code = code; this.status = status; }
}

// a provider's own words, with anything that looks like a key taken out
export function scrub(text: string, key: string): string {
  let t = String(text || '');
  if (key) t = t.split(key).join('[key]');
  return t.replace(/\b(sk|pk|rk)-[A-Za-z0-9_\-*.]{8,}/g, '[key]').replace(/\s+/g, ' ').trim().slice(0, 240);
}

function plain(p: Provider, status: number, said: string): ProviderError {
  const tail = said ? ' (' + p.label + ' said: ' + said + ')' : '';
  if (status === 401 || status === 403) return new ProviderError('key', status, p.label + ' refused the key. Check that it is complete and still active.' + tail);
  if (status === 402) return new ProviderError('credit', status, 'The ' + p.label + ' account has no credit left.' + tail);
  if (status === 429) return new ProviderError('rate', status, p.label + ' is limiting requests, or the account has reached its limit. Wait a minute, or check the plan.' + tail);
  if (status === 404) return new ProviderError('model', status, p.label + ' does not offer this model to this key.' + tail);
  if (status === 400 || status === 413 || status === 422) return new ProviderError('request', status, p.label + ' could not use the request.' + tail);
  if (status >= 500) return new ProviderError('busy', status, p.label + ' is busy or down. Try again later.' + tail);
  return new ProviderError('request', status, p.label + ' answered with status ' + status + '.' + tail);
}

// one call: fetch with a time limit, errors in plain words, the reply's JSON
async function call(p: Provider, key: string, url: string, init: RequestInit, o: CallOptions, timeoutMs: number): Promise<any> {
  const f = o.fetch || globalThis.fetch.bind(globalThis);
  const ctl = new AbortController();
  const limit = o.timeoutMs ?? timeoutMs;
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; ctl.abort(); }, limit);
  const outer = () => ctl.abort();
  o.signal?.addEventListener('abort', outer);
  let r: Response;
  try {
    r = await f(url, { ...init, signal: ctl.signal, credentials: 'omit', referrerPolicy: 'no-referrer', cache: 'no-store' });
  } catch (e: any) {
    if (timedOut) throw new ProviderError('timeout', 0, p.label + ' did not answer within ' + Math.round(limit / 1000) + ' s.');
    if (o.signal?.aborted) throw new ProviderError('network', 0, 'The call to ' + p.label + ' was stopped.');
    throw new ProviderError('network', 0, 'Could not reach ' + p.label + '. Check the connection; a firewall or a browser extension may block it.');
  } finally {
    clearTimeout(timer);
    o.signal?.removeEventListener('abort', outer);
  }
  let body: any = null;
  const text = await r.text().catch(() => '');
  try { body = text ? JSON.parse(text) : null; } catch (e) { body = null; }
  if (!r.ok) {
    const said = body?.error?.message || body?.message || (typeof body?.error === 'string' ? body.error : '');
    throw plain(p, r.status, scrub(said, key));
  }
  if (body === null) throw new ProviderError('reply', r.status, p.label + ' sent a reply that is not JSON.');
  return body;
}

const json = (headers: Record<string, string>, body: unknown): RequestInit => ({ method: 'POST', headers: { ...headers, 'content-type': 'application/json' }, body: JSON.stringify(body) });

function splitSystem(messages: ChatMessage[]) {
  const system = messages.filter((m) => m.role === 'system').map((m) => m.content).join('\n\n');
  return { system, rest: messages.filter((m) => m.role !== 'system').map((m) => ({ role: m.role, content: m.content })) };
}

function textOf(p: Provider, t: unknown): string {
  if (typeof t !== 'string') throw new ProviderError('reply', 200, p.label + ' sent a reply without text.');
  return t;
}

const ANTHROPIC = 'https://api.anthropic.com/v1';
export const anthropic: Provider = {
  id: 'anthropic', label: 'Anthropic', keysUrl: 'https://platform.claude.com/settings/keys', prefix: 'sk-ant-', defaultModel: 'claude-haiku-5-5',
  async check(key, o = {}) {
    const body = await call(anthropic, key, ANTHROPIC + '/models?limit=100', { headers: anthropicHeaders(key) }, o, 20000);
    const n = Array.isArray(body.data) ? body.data.length : 0;
    return { ok: true, models: n, detail: 'The key works: ' + n + ' models.' };
  },
  async complete(key, messages, o = {}) {
    const { system, rest } = splitSystem(messages);
    const body = await call(anthropic, key, ANTHROPIC + '/messages', json(anthropicHeaders(key), {
      model: o.model || anthropic.defaultModel, max_tokens: o.maxTokens ?? 1024, ...(system ? { system } : {}), messages: rest,
    }), o, 120000);
    const parts = Array.isArray(body.content) ? body.content.filter((c: any) => c && c.type === 'text').map((c: any) => c.text) : [];
    if (!parts.length) throw new ProviderError('reply', 200, 'Anthropic sent a reply without text.');
    return parts.join('');
  },
};
function anthropicHeaders(key: string): Record<string, string> {
  return { 'x-api-key': key, 'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true' };
}

const OPENAI = 'https://api.openai.com/v1';
export const openai: Provider = {
  id: 'openai', label: 'OpenAI', keysUrl: 'https://platform.openai.com/api-keys', prefix: 'sk-', defaultModel: 'gpt-5-mini',
  async check(key, o = {}) {
    const body = await call(openai, key, OPENAI + '/models', { headers: { authorization: 'Bearer ' + key } }, o, 20000);
    const n = Array.isArray(body.data) ? body.data.length : 0;
    return { ok: true, models: n, detail: 'The key works: ' + n + ' models.' };
  },
  async complete(key, messages, o = {}) {
    const body = await call(openai, key, OPENAI + '/chat/completions', json({ authorization: 'Bearer ' + key }, {
      model: o.model || openai.defaultModel, messages: messages.map((m) => ({ role: m.role, content: m.content })),
      ...(o.maxTokens ? { max_completion_tokens: o.maxTokens } : {}),
    }), o, 120000);
    return textOf(openai, body.choices?.[0]?.message?.content);
  },
};

const OPENROUTER = 'https://openrouter.ai/api/v1';
export const openrouter: Provider = {
  id: 'openrouter', label: 'OpenRouter', keysUrl: 'https://openrouter.ai/settings/keys', prefix: 'sk-or-', defaultModel: 'openrouter/auto',
  // OpenRouter lists its models without a key, so the check reads the key's own record (GET /key)
  async check(key, o = {}) {
    const body = await call(openrouter, key, OPENROUTER + '/key', { headers: { authorization: 'Bearer ' + key } }, o, 20000);
    const d = body.data || {};
    const left = typeof d.limit_remaining === 'number' ? ' Credit left on the key: ' + d.limit_remaining.toFixed(2) + '.' : '';
    return { ok: true, models: 0, detail: 'The key works.' + left };
  },
  async complete(key, messages, o = {}) {
    const body = await call(openrouter, key, OPENROUTER + '/chat/completions', json({ authorization: 'Bearer ' + key, 'x-title': 'OpenSculptBoy' }, {
      model: o.model || openrouter.defaultModel, messages: messages.map((m) => ({ role: m.role, content: m.content })),
      ...(o.maxTokens ? { max_tokens: o.maxTokens } : {}),
    }), o, 120000);
    return textOf(openrouter, body.choices?.[0]?.message?.content);
  },
};

export const PROVIDERS: Record<ProviderId, Provider> = { anthropic, openai, openrouter };

// a hint when a pasted key looks like another provider's
export function keyHint(provider: ProviderId, key: string): string {
  const k = key.trim();
  const owner = (['anthropic', 'openrouter', 'openai'] as ProviderId[]).find((id) => k.startsWith(PROVIDERS[id].prefix));
  if (owner && owner !== provider) return 'This looks like an ' + PROVIDERS[owner].label + ' key.';
  return '';
}

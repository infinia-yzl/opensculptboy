// OpenSculptBoy
// Apache License, Version 2.0
//
// View › Settings in React, with the AI providers (docs/viewer.md, "AI providers"): set up the passkey, unlock, add a
// key, check it, forget it. Where the passkey has no PRF, the panel keeps pasted keys for this visit only
// (src/ai/session.ts). The logic lives in src/ai/. The key field is never part of a form, a saved setting or the URL,
// and it is emptied as soon as the key is taken. The panel reads the vault only once Settings shows, so the page opens
// no database before a person asks for it.

import { useEffect, useRef, useState } from 'react';
import { keyHint, pageSession, pageSupport, pageVault, PROVIDERS, type ProviderId } from '../../ai/index.ts';
import { copyText, saveUi, uiState } from '../../ui/dom.ts';
import { Note, Seg, TextButton } from '../components/controls.tsx';

const day = (t: number) => new Date(t).toLocaleDateString(undefined, { day: 'numeric', month: 'short' });
export const SESSION_LINE = 'This passkey has no PRF, so the page cannot store keys. A pasted key is used for this visit only: it stays in memory and is gone when the page closes.';

interface Info { provider: string; added: number; checked?: number; ok?: boolean; note?: string }

function Unsupported({ support }: { support: ReturnType<typeof pageSupport> }) {
  const [said, setSaid] = useState('');
  return (
    <div className="ai-panel">
      <Note>{support.message}</Note>
      {support.command ? (
        <div className="ai-cmd">
          <code className="ai-code">{support.command.split(' ').map((w, i) => <span key={i}>{i ? ' ' : ''}<span className="ai-word">{w}</span></span>)}</code>
          <TextButton label="Copy" onClick={async () => setSaid((await copyText(support.command!)) ? 'Copied.' : 'Select the command and copy it.')} />
        </div>
      ) : null}
      {support.url && support.code !== 'file-page' ? <a className="note" href={support.url}>{support.url}</a> : null}
      <p className="note ai-status" role="status" aria-live="polite">{said}</p>
    </div>
  );
}

function Vault() {
  const vault = pageVault(), visit = pageSession();
  // the visit mode: after a passkey without PRF; the page remembers that this browser's passkey had none (no key)
  const [session, setSession] = useState(uiState().aiSession === true);
  const [provider, setProvider] = useState<ProviderId>('anthropic');
  const [hint, setHint] = useState(keyHint('anthropic', ''));
  const [status, setStatus] = useState({ text: '', bad: false });
  const [st, setSt] = useState<{ passkeys: number; unlocked: boolean; keys: Info[] } | null>(null);
  const field = useRef<HTMLInputElement>(null);
  const S = useRef(session); S.current = session;
  const say = (text: string, bad = false) => setStatus({ text, bad });
  const render = async () => {
    const s = await vault.state();
    if (s.passkeys && S.current) { S.current = false; setSession(false); }   // a vault with a passkey always wins
    setSt(s);
  };
  useEffect(() => {
    render().catch((e) => say(String(e?.message || e), true));
    return vault.onChange((why) => { if (why === 'lock') { say('Locked.'); render(); } });
  }, []);
  const busy = async (what: string, fn: () => Promise<void>) => {
    say(what);
    try { await fn(); } catch (e: any) {
      if (e && e.name === 'PasskeyError' && e.code === 'no-prf') { S.current = true; setSession(true); saveUi({ aiSession: true }); say(SESSION_LINE); } else say(e && e.message ? e.message : String(e), true);
    }
    await render();
  };
  const check = async (id: ProviderId) => {
    const p = PROVIDERS[id], holder: any = S.current ? visit : vault;
    try {
      const r = await holder.use(id, (key: string) => p.check(key));
      await holder.mark(id, true, r.detail);
      say(p.label + ': ' + r.detail);
    } catch (e: any) {
      const msg = e && e.message ? e.message : String(e);
      if (e && e.name === 'ProviderError') await holder.mark(id, false, msg);
      say(msg, true);
    }
  };
  const save = () => busy('Saving…', async () => {
    const key = field.current!.value;
    field.current!.value = ''; setHint(keyHint(provider, ''));
    if (S.current) visit.set(provider, key); else await vault.save(provider, key);
    say(PROVIDERS[provider].label + (S.current ? ' key kept for this visit. Checking…' : ' key saved. Checking…'));
    await check(provider);
  });
  const row = (k: Info, canCheck: boolean, forget: () => Promise<void> | void, kept: string) => {
    const p = PROVIDERS[k.provider as ProviderId], label = p ? p.label : k.provider;
    const state = k.checked ? (k.ok ? 'works' : 'refused') : 'not checked';
    return (
      <div key={k.provider} className="ai-row" data-provider={k.provider}>
        <div className="ai-what"><span className="ai-name">{label}</span><span className="note" title={k.note || ''}>{kept + ' · ' + state}</span></div>
        {canCheck && p ? <TextButton label="Check" onClick={() => busy('Checking ' + label + '…', () => check(k.provider as ProviderId))} /> : null}
        <TextButton label="Forget" onClick={() => busy('', async () => { await forget(); say(label + ' key forgotten.'); })} />
      </div>
    );
  };
  let intro = '', count = '', rows: any[] = [], adder = false;
  const actions: any[] = [];
  if (session) {
    count = 'this visit'; intro = SESSION_LINE; adder = true;
    rows = visit.list().map((k) => row(k, true, () => visit.forget(k.provider), 'this visit only'));
    actions.push(<TextButton key="again" label="Try a passkey again" onClick={() => { S.current = false; setSession(false); saveUi({ aiSession: false }); visit.clear(); say(''); render(); }} />);
  } else if (st) {
    count = st.passkeys ? (st.unlocked ? 'unlocked' : 'locked') : '';
    rows = st.keys.map((k) => row(k, st.unlocked, () => vault.forget(k.provider), 'added ' + day(k.added)));
    adder = st.unlocked;
    if (!st.passkeys) {
      intro = 'Keep API keys on this device, encrypted, and unlock them with a passkey (Windows Hello, Touch ID, a phone or a security key).';
      actions.push(<TextButton key="setup" label="Set up passkey" icon="key" onClick={() => busy('Waiting for the passkey…', async () => { await vault.setup(); say('Passkey ready. The vault is unlocked.'); })} />);
    } else if (!st.unlocked) {
      intro = st.keys.length ? 'Locked. Unlock with the passkey to add, check or use keys.' : 'Locked. Unlock with the passkey to add a key.';
      actions.push(<TextButton key="unlock" label="Unlock" icon="key" onClick={() => busy('Waiting for the passkey…', async () => { await vault.unlock(); say('Unlocked. The vault locks when the page closes, or with Lock.'); })} />);
    } else {
      intro = 'Unlocked. The keys stay on this device and go only to their provider.';
      actions.push(<TextButton key="lock" label="Lock" icon="lock" onClick={() => { vault.lock(); }} />);
    }
    if (st.passkeys || st.keys.length) {
      actions.push(<TextButton key="all" label="Forget all" className="ai-danger" onClick={() => {
        if (!confirm('Forget every stored key and the vault\'s passkey on this device?')) return;
        busy('', async () => { await vault.forgetAll(); say('Everything forgotten.'); });
      }} />);
    }
  }
  return (
    <div className="ai-panel osb-section">
      <h3 className="osb-eyebrow">AI providers <span className="osb-eyebrow-val">{count}</span></h3>
      <Note>{intro}</Note>
      <div className="ai-list">{rows}</div>
      <div className="ai-add" hidden={!adder}>
        <Seg label="Provider" fill items={(Object.keys(PROVIDERS) as ProviderId[]).map((id) => ({ value: id, label: PROVIDERS[id].label }))} value={provider}
          onPick={(v) => { setProvider(v as ProviderId); setHint(keyHint(v as ProviderId, field.current?.value || '')); }} />
        {/* type password: the key shows as dots; spellcheck off, so no spelling service ever sees it */}
        <input ref={field} className="cs-input ai-key" type="password" autoComplete="off" spellCheck={false} autoCapitalize="off" aria-label="API key" placeholder="Paste the API key"
          onInput={() => setHint(keyHint(provider, field.current!.value))} onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); save(); } }} />
        <Note>{hint}</Note>
        <div className="btn-row"><TextButton label={session ? 'Use for this visit' : 'Save key'} onClick={save} /></div>
      </div>
      <div className="btn-row">{actions}</div>
      <p className={'note ai-status' + (status.bad ? ' ai-bad' : '')} role="status" aria-live="polite">{status.text}</p>
    </div>
  );
}

export function SettingsSection({ shown }: { shown: boolean }) {
  const [seen, setSeen] = useState(false);
  useEffect(() => { if (shown) setSeen(true); }, [shown]);
  const support = pageSupport();
  return (
    <section id="sec-settings" className="sec osb-section-stack" role="tabpanel" aria-labelledby="tool-title" hidden={!shown}>
      <Note>The keys of the AI providers that the page's AI features use. A key stays on this device, and the page sends it only to its provider.</Note>
      {seen ? (support.ok ? <Vault /> : <Unsupported support={support} />) : null}
    </section>
  );
}

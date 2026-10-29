// OpenSculptBoy
// Apache License, Version 2.0
//
// A temporary panel for AI providers in the Scene section (docs/viewer.md, "AI providers"), until the redesigned
// interface takes over: set up the passkey, unlock, add a key, check it, forget it. The logic lives in src/ai/.
// The key field is never part of a form, a saved setting or the URL, and it is emptied as soon as the key is stored.

import { keyHint, pageSupport, pageVault, PROVIDERS, type ProviderId } from '../ai/index.ts';
import { foldGroup, segmented, textButton } from './controls.ts';
import { copyText, h } from './dom.ts';

const day = (t: number) => new Date(t).toLocaleDateString(undefined, { day: 'numeric', month: 'short' });

export function aiPanel(): HTMLElement {
  const fold = foldGroup('AI providers');
  fold.el.classList.add('ai-panel');
  const body = fold.body;
  const status = h('p.note.ai-status', { role: 'status', 'aria-live': 'polite' });
  const say = (text: string, bad = false) => { status.textContent = text; status.classList.toggle('ai-bad', bad); };

  const support = pageSupport();
  if (!support.ok) {
    body.append(h('p.note', { text: support.message }));
    if (support.command) {
      // each word of the command on one line: the lines break at spaces only, never after a hyphen of an option
      const code = h('code.ai-code', {}, ...support.command.split(' ').flatMap((w, i) => [i ? ' ' : '', h('span.ai-word', { text: w })]));
      body.append(h('div.ai-cmd', {}, code, textButton('Copy', async () => say((await copyText(support.command!)) ? 'Copied.' : 'Select the command and copy it.'))));
    }
    if (support.url && support.code !== 'file-page') body.append(h('a.note', { href: support.url, text: support.url }));
    fold.count.textContent = 'off';
    body.append(status);
    return fold.el;
  }

  const vault = pageVault();
  let provider: ProviderId = 'anthropic';
  const pick = segmented('Provider', (Object.keys(PROVIDERS) as ProviderId[]).map((id) => ({ value: id, label: PROVIDERS[id].label })), (v) => { provider = v as ProviderId; pick.set(v); hint(); }, 'seg-fill');
  pick.set(provider);
  // type password: the key shows as dots; spellcheck off, so no spelling service ever sees it
  const field = h('input.cs-input.ai-key', { type: 'password', autocomplete: 'off', spellcheck: 'false', autocapitalize: 'off', 'aria-label': 'API key', placeholder: 'Paste the API key' }) as HTMLInputElement;
  const hintEl = h('p.note');
  const hint = () => { hintEl.textContent = keyHint(provider, field.value); };
  field.addEventListener('input', hint);
  field.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); save(); } });
  const list = h('div.ai-list');
  const intro = h('p.note');
  const actions = h('div.btn-row');
  const adder = h('div.ai-add', {}, pick.el, field, hintEl, h('div.btn-row', {}, textButton('Save key', () => save())));

  const busy = async (what: string, fn: () => Promise<void>) => {
    say(what);
    try { await fn(); } catch (e: any) { say(e && e.message ? e.message : String(e), true); }
    await render();
  };
  const save = () => busy('Saving…', async () => {
    const key = field.value;
    field.value = ''; hint();
    await vault.save(provider, key);
    say(PROVIDERS[provider].label + ' key saved. Checking…');
    await check(provider);
  });
  const check = async (id: ProviderId) => {
    const p = PROVIDERS[id];
    try {
      const r = await vault.use(id, (key) => p.check(key));
      await vault.mark(id, true, r.detail);
      say(p.label + ': ' + r.detail);
    } catch (e: any) {
      const msg = e && e.message ? e.message : String(e);
      if (e && e.name === 'ProviderError') await vault.mark(id, false, msg);
      say(msg, true);
    }
  };

  async function render() {
    const st = await vault.state();
    fold.count.textContent = st.passkeys ? (st.unlocked ? 'unlocked' : 'locked') : '';
    list.replaceChildren(...st.keys.map((k) => {
      const p = PROVIDERS[k.provider as ProviderId];
      const label = p ? p.label : k.provider;
      const state = k.checked ? (k.ok ? 'works' : 'refused') : 'not checked';
      const row = h('div.ai-row', { dataset: { provider: k.provider } }, h('div.ai-what', {},
        h('span.ai-name', { text: label }),
        h('span.note', { text: 'added ' + day(k.added) + ' · ' + state, title: k.note || '' })));
      if (st.unlocked && p) row.append(textButton('Check', () => busy('Checking ' + label + '…', () => check(k.provider as ProviderId))));
      row.append(textButton('Forget', () => busy('', async () => { await vault.forget(k.provider); say(label + ' key forgotten.'); })));
      return row;
    }));
    actions.replaceChildren();
    if (!st.passkeys) {
      intro.textContent = 'Keep API keys on this device, encrypted, and unlock them with a passkey (Windows Hello, Touch ID, a phone or a security key).';
      actions.append(textButton('Set up passkey', () => busy('Waiting for the passkey…', async () => { await vault.setup(); say('Passkey ready. The vault is unlocked.'); })));
    } else if (!st.unlocked) {
      intro.textContent = st.keys.length ? 'Locked. Unlock with the passkey to add, check or use keys.' : 'Locked. Unlock with the passkey to add a key.';
      actions.append(textButton('Unlock', () => busy('Waiting for the passkey…', async () => { await vault.unlock(); say('Unlocked. It locks again after ' + Math.round(vault.idleMs / 60000) + ' minutes without use.'); })));
    } else {
      intro.textContent = 'Unlocked. The keys stay on this device and go only to their provider.';
      actions.append(textButton('Lock', () => { vault.lock(); }));
    }
    if (st.passkeys || st.keys.length) {
      actions.append(textButton('Forget all', () => {
        if (!confirm('Forget every stored key and the vault\'s passkey on this device?')) return;
        busy('', async () => { await vault.forgetAll(); say('Everything forgotten.'); });
      }, undefined, 'ai-danger'));
    }
    adder.hidden = !st.unlocked;
  }

  vault.onChange((why) => {
    if (why === 'lock') say('Locked.');
    if (why === 'idle') say('Locked after ' + Math.round(vault.idleMs / 60000) + ' minutes without use.');
    if (why === 'lock' || why === 'idle') render();
  });
  // the panel renders when the fold opens, so the page opens no database before a person asks for it
  let shown = false;
  fold.el.addEventListener('toggle', () => { if (fold.el.open && !shown) { shown = true; render().catch((e) => say(String(e && e.message || e), true)); } });
  body.append(intro, list, adder, actions, status);
  adder.hidden = true;
  return fold.el;
}

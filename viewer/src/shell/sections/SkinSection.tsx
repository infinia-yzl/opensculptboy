// OpenSculptBoy
// Apache License, Version 2.0
//
// Create's Skin and eyes section in React, with the design system's Swatch: the stone library (Carrara, Travertine,
// Bardiglio, Basalt, Bronze and Verdigris, and a stone of a colour of its own) that makes the character a statue,
// eight skin tones from fair to deep, the undertone as a segmented control (Cool, Neutral, Warm), the eye colours
// drawn as small irises with a custom colour, and the Adjust fold with the tone, the undertone and the stone's polish
// as sliders. A tone makes the character skin again. A click applies at once and Undo takes it back. The chosen
// swatches are React state that changes only when the choice changes; the tones' colours follow the undertone
// through refs, so a slider drag renders nothing in React.

import { useRef, useState } from 'react';
import { useShell } from '../context.ts';
import { useBus } from '../hooks.ts';
import { Slider } from '../components/Slider.tsx';
import { Fold, Note } from '../components/controls.tsx';
import { Icon } from '../components/ui/icon.tsx';
import { RadioGroup, RadioGroupItem } from '../components/ui/radio-group.tsx';
import { ToggleGroup, ToggleGroupItem } from '../components/ui/toggle-group.tsx';
import { Tip } from '../components/ui/tooltip.tsx';
import { STONES, stoneOf, stoneSwatch } from '../../stones.ts';
import { STONE_HAIR } from '../../starters.ts';

// the tones of the swatches, from fair to deep, and the undertones of the segmented control
export const TONES = [0.04, 0.18, 0.35, 0.46, 0.56, 0.68, 0.8, 0.92];
export const UNDERTONES: [string, number, string][] = [['cool', -0.5, 'Cool'], ['neutral', 0, 'Neutral'], ['warm', 0.5, 'Warm']];
const near = (a: number, b: number) => Math.abs(a - b) < 0.005;
const rgb = (c: number[]) => `rgb(${c.map((v) => Math.round(v * 255)).join(',')})`;
const iris = (hex: string) => `radial-gradient(circle, #0d0c0b 0 22%, ${hex} 25% 58%, color-mix(in srgb, ${hex} 60%, #000) 66%, #efe9e1 70%)`;
const STONE_HAIRS = new Set(Object.values(STONE_HAIR));

// a stone for the look: the hair that wore a stone's colour takes the new stone's
export function setStone(n: any, id: string | null, polish?: number) {
  if (!id) { delete n.skin.stone; delete n.skin.polish; return; }
  const st = stoneOf({ stone: id });
  if (!st) return;
  n.skin.stone = st.id; n.skin.polish = polish ?? st.polish;
  if (STONE_HAIRS.has(n.hair.color) && STONE_HAIR[st.id]) n.hair.color = STONE_HAIR[st.id];
}

export function SkinSection({ shown }: { shown: boolean }) {
  const { app, ed } = useShell();
  const toneRow = useRef<HTMLDivElement>(null), skinName = useRef<HTMLSpanElement>(null), eyeName = useRef<HTMLSpanElement>(null);
  const eyeInput = useRef<HTMLInputElement>(null), stoneInput = useRef<HTMLInputElement>(null);
  const [tone, setTone] = useState('');
  const [under, setUnder] = useState('');
  const [eye, setEye] = useState('');
  const [stone, setStoneState] = useState('');
  const [baked, setBaked] = useState(false);
  useBus(['look', 'wardrobe'], () => {
    const L = app.look(), st = stoneOf(L.skin);
    const t = TONES.findIndex((v) => near(v, L.skin.tone)), u = UNDERTONES.find(([, v]) => near(v, L.skin.undertone));
    setStoneState(st ? (STONES.some((x) => x.id === st.id) ? st.id : 'custom') : '');
    setTone(st ? '' : t >= 0 ? 'tone-' + t : 'custom');
    setUnder(st ? '' : u ? u[0] : '');
    const pal = app.palettes.eyes.find(([, hex]) => hex === L.eyes.color);
    setEye(pal ? pal[1] : 'custom');
    setBaked(!!app.cardLook()?.summary()?.skin);
    // the tones in the undertone of the look
    toneRow.current?.querySelectorAll<HTMLElement>('[data-tone]').forEach((b) => b.style.setProperty('--sw', rgb(app.skinBase(+b.dataset.tone!, L.skin.undertone))));
    if (skinName.current) skinName.current.textContent = st ? st.label : (t >= 0 ? `Tone ${t + 1}` : 'Custom') + (u ? `, ${u[2].toLowerCase()}` : '');
    if (eyeName.current) eyeName.current.textContent = st ? 'Stone' : pal ? pal[0] : 'Custom ' + L.eyes.color;
    if (eyeInput.current && document.activeElement !== eyeInput.current) eyeInput.current.value = L.eyes.color;
    if (stoneInput.current && document.activeElement !== stoneInput.current) stoneInput.current.value = st ? st.base : '#e9e5de';
  });
  const pickStone = (v: string) => {
    const st = STONES.find((x) => x.id === v);
    if (st) ed.edit('Skin: ' + st.label, (n) => setStone(n, st.id));
  };
  const pickTone = (v: string) => {
    if (!v.startsWith('tone-')) return;
    const t = TONES[+v.slice(5)];
    ed.edit('Skin tone', (n) => { n.skin.tone = t; setStone(n, null); });
  };
  const pickUnder = (v: string) => {
    const u = UNDERTONES.find(([k]) => k === v);
    if (u) ed.edit('Undertone', (n) => { n.skin.undertone = u[1]; setStone(n, null); });
  };
  const pickEye = (hex: string) => { if (hex !== 'custom') ed.edit('Eye colour', (n) => { n.eyes.color = hex; }); };
  // the colour tracks of the custom skin: the tones at the look's undertone, and the undertones at its tone
  const toneTrack = (inp: HTMLInputElement) => {
    const L = app.look(), stops = [];
    for (let i = 0; i <= 10; i++) stops.push(`${rgb(app.skinBase(i / 10, L.skin.undertone))} ${i * 10}%`);
    inp.style.setProperty('--track', `linear-gradient(90deg, ${stops.join(', ')})`);
    inp.style.setProperty('--thumb-fill', rgb(app.skinBase(L.skin.tone, L.skin.undertone)));
  };
  const underTrack = (inp: HTMLInputElement) => {
    const L = app.look();
    inp.style.setProperty('--track', `linear-gradient(90deg, ${rgb(app.skinBase(L.skin.tone, -1))}, ${rgb(app.skinBase(L.skin.tone, 0))}, ${rgb(app.skinBase(L.skin.tone, 1))})`);
    inp.style.setProperty('--thumb-fill', rgb(app.skinBase(L.skin.tone, L.skin.undertone)));
  };
  return (
    <section id="sec-skin" className="sec osb-section-stack" role="tabpanel" aria-labelledby="tab-skin" hidden={!shown}>
      {baked ? <Note className="osb-override">{stone ? 'A stone of the library shows over the card\'s baked skin. Use own shows the card\'s skin again.' : 'The card\'s baked skin is shown. A stone of the library shows over it.'}</Note> : null}
      <div className="osb-section">
        <h3 className="osb-eyebrow">Skin <span ref={skinName} className="osb-eyebrow-val" /></h3>
        <div className="osb-swatch-line">
          <RadioGroup aria-label="Stone" value={stone === 'custom' ? '' : stone} onValueChange={pickStone}>
            {STONES.map((st) => (
              <Tip key={st.id} label={st.label} side="top"><RadioGroupItem value={st.id} aria-label={st.label} data-stone={st.id} style={{ background: stoneSwatch(st) }} /></Tip>
            ))}
          </RadioGroup>
          <Tip label="A stone of your own colour" side="top">
            <label className="osb-sw osb-sw--custom osb-sw-input" data-state={stone === 'custom' ? 'checked' : 'unchecked'}>
              <Icon name="eyedropper" size={14} />
              <input ref={stoneInput} type="color" aria-label="Custom stone colour" onInput={(e) => { const v = (e.target as HTMLInputElement).value.toLowerCase(); ed.edit('Stone colour', (n) => setStone(n, v, n.skin.polish)); }} />
            </label>
          </Tip>
        </div>
        <RadioGroup ref={toneRow} aria-label="Skin tone" value={tone} onValueChange={pickTone}>
          {TONES.map((t, i) => (
            <Tip key={i} label={`Tone ${i + 1}`} side="top"><RadioGroupItem value={'tone-' + i} data-tone={t} aria-label={`Tone ${i + 1}`} /></Tip>
          ))}
        </RadioGroup>
        <div className="osb-swatch-head">
          <span className="osb-note" style={{ flex: 1 }}>Undertone</span>
          <ToggleGroup type="single" aria-label="Undertone" value={under} onValueChange={pickUnder}>
            {UNDERTONES.map(([k, , label]) => <ToggleGroupItem key={k} value={k}>{label}</ToggleGroupItem>)}
          </ToggleGroup>
        </div>
      </div>
      <div className="osb-section">
        <h3 className="osb-eyebrow">Eyes <span ref={eyeName} className="osb-eyebrow-val" /></h3>
        {stone ? <Note>A statue's eyes are its stone. The colour below shows when the character is skin again.</Note> : null}
        <div className="osb-swatch-line">
          <RadioGroup aria-label="Eye colour" value={eye} onValueChange={pickEye}>
            {app.palettes.eyes.map(([name, hex]) => (
              <Tip key={hex} label={name} side="top"><RadioGroupItem value={hex} aria-label={name} style={{ background: iris(hex) }} /></Tip>
            ))}
          </RadioGroup>
          <Tip label="Custom eye colour" side="top">
            <label className="osb-sw osb-sw--custom osb-sw-input" data-state={eye === 'custom' ? 'checked' : 'unchecked'} aria-checked={eye === 'custom'}>
              <Icon name="eyedropper" size={14} />
              <input ref={eyeInput} type="color" aria-label="Custom eye colour" onInput={(e) => { const v = (e.target as HTMLInputElement).value.toLowerCase(); ed.edit('Eye colour', (n) => { n.eyes.color = v; }); }} />
            </label>
          </Tip>
        </div>
      </div>
      <div className="osb-folds">
        <Fold title="Adjust" count={stone ? 'Polish' : 'Tone, undertone'}>
          <div className="osb-sliders">
            {stone ? (
              <Slider id="ed-polish" label="Polish" min={0} max={1} def={stoneOf(app.look().skin)?.polish ?? 0.5} ends={['Rough', 'Polished']}
                get={() => app.look().skin.polish ?? 0.5} onStart={() => ed.begin('Polish')} onEnd={() => ed.end()} onInput={(v) => ed.queue((n) => { n.skin.polish = v; })} />
            ) : null}
            <Slider id="ed-tone" label="Tone" min={0} max={1} def={0.35} ends={['Fair', 'Deep']} colour paintTrack={toneTrack}
              get={() => app.look().skin.tone} onStart={() => ed.begin('Skin tone')} onEnd={() => ed.end()} onInput={(v) => ed.queue((n) => { n.skin.tone = v; setStone(n, null); })} />
            <Slider id="ed-under" label="Undertone" min={-1} max={1} def={0} ends={['Cool', 'Warm']} colour paintTrack={underTrack}
              get={() => app.look().skin.undertone} onStart={() => ed.begin('Undertone')} onEnd={() => ed.end()} onInput={(v) => ed.queue((n) => { n.skin.undertone = v; setStone(n, null); })} />
          </div>
        </Fold>
      </div>
    </section>
  );
}

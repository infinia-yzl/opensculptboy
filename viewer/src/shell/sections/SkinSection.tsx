// OpenSculptBoy
// Apache License, Version 2.0
//
// Create's Skin and eyes section in React, with the design system's Swatch: eight skin tones from fair to deep, a
// custom swatch that opens the tone and undertone sliders in a popover, the undertone as a segmented control (Cool,
// Neutral, Warm), and the eye colours drawn as small irises with a custom colour. A click applies the colour at once
// and Undo takes it back. The chosen swatch is React state that changes only when the choice changes; the colours of
// the tones follow the undertone through refs, so a slider drag renders nothing in React.

import { useRef, useState } from 'react';
import { useShell } from '../context.ts';
import { useBus } from '../hooks.ts';
import { Slider } from '../components/Slider.tsx';
import { Icon } from '../components/ui/icon.tsx';
import { Popover, PopoverContent, PopoverTrigger } from '../components/ui/popover.tsx';
import { RadioGroup, RadioGroupItem } from '../components/ui/radio-group.tsx';
import { ToggleGroup, ToggleGroupItem } from '../components/ui/toggle-group.tsx';
import { Tip } from '../components/ui/tooltip.tsx';

// the tones of the swatches, from fair to deep, and the undertones of the segmented control
export const TONES = [0.04, 0.18, 0.35, 0.46, 0.56, 0.68, 0.8, 0.92];
export const UNDERTONES: [string, number, string][] = [['cool', -0.5, 'Cool'], ['neutral', 0, 'Neutral'], ['warm', 0.5, 'Warm']];
const near = (a: number, b: number) => Math.abs(a - b) < 0.005;
const rgb = (c: number[]) => `rgb(${c.map((v) => Math.round(v * 255)).join(',')})`;
const iris = (hex: string) => `radial-gradient(circle, #0d0c0b 0 22%, ${hex} 25% 58%, color-mix(in srgb, ${hex} 60%, #000) 66%, #efe9e1 70%)`;

export function SkinSection({ shown }: { shown: boolean }) {
  const { app, ed } = useShell();
  const toneRow = useRef<HTMLDivElement>(null), skinName = useRef<HTMLSpanElement>(null), eyeName = useRef<HTMLSpanElement>(null);
  const eyeInput = useRef<HTMLInputElement>(null);
  const [tone, setTone] = useState('');
  const [under, setUnder] = useState('');
  const [eye, setEye] = useState('');
  useBus('look', () => {
    const L = app.look();
    const t = TONES.findIndex((v) => near(v, L.skin.tone)), u = UNDERTONES.find(([, v]) => near(v, L.skin.undertone));
    setTone(t >= 0 ? 'tone-' + t : 'custom');
    setUnder(u ? u[0] : '');
    const pal = app.palettes.eyes.find(([, hex]) => hex === L.eyes.color);
    setEye(pal ? pal[1] : 'custom');
    // the tones in the undertone of the look
    toneRow.current?.querySelectorAll<HTMLElement>('[data-tone]').forEach((b) => b.style.setProperty('--sw', rgb(app.skinBase(+b.dataset.tone!, L.skin.undertone))));
    const cust = toneRow.current?.querySelector<HTMLElement>('.osb-sw--custom');
    if (cust) cust.style.setProperty('--sw-now', rgb(app.skinBase(L.skin.tone, L.skin.undertone)));
    if (skinName.current) skinName.current.textContent = (t >= 0 ? `Tone ${t + 1}` : 'Custom') + (u ? `, ${u[2].toLowerCase()}` : '');
    if (eyeName.current) eyeName.current.textContent = pal ? pal[0] : 'Custom ' + L.eyes.color;
    if (eyeInput.current && document.activeElement !== eyeInput.current) eyeInput.current.value = L.eyes.color;
  });
  const pickTone = (v: string) => {
    if (!v.startsWith('tone-')) return;
    const t = TONES[+v.slice(5)];
    ed.edit('Skin tone', (n) => { n.skin.tone = t; });
  };
  const pickUnder = (v: string) => {
    const u = UNDERTONES.find(([k]) => k === v);
    if (u) ed.edit('Undertone', (n) => { n.skin.undertone = u[1]; });
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
      <div className="osb-section">
        <h3 className="osb-eyebrow">Skin <span ref={skinName} className="osb-eyebrow-val" /></h3>
        <RadioGroup ref={toneRow} aria-label="Skin tone" value={tone} onValueChange={pickTone}>
          {TONES.map((t, i) => (
            <Tip key={i} label={`Tone ${i + 1}`} side="top"><RadioGroupItem value={'tone-' + i} data-tone={t} aria-label={`Tone ${i + 1}`} /></Tip>
          ))}
        </RadioGroup>
        <div className="osb-swatch-head osb-custom-row">
            <Popover>
              <Tip label="Custom tone and undertone" side="top">
                <PopoverTrigger asChild>
                  <button type="button" className="osb-sw osb-sw--custom" aria-label="Custom skin colour" aria-pressed={tone === 'custom'} data-state={tone === 'custom' ? 'checked' : 'unchecked'}><Icon name="eyedropper" size={14} /></button>
                </PopoverTrigger>
              </Tip>
              <PopoverContent side="right" align="start" className="osb-picker">
                <div className="osb-sliders">
                  <Slider id="ed-tone" label="Tone" min={0} max={1} def={0.35} ends={['Fair', 'Deep']} colour paintTrack={toneTrack}
                    get={() => app.look().skin.tone} onStart={() => ed.begin('Skin tone')} onEnd={() => ed.end()} onInput={(v) => ed.queue((n) => { n.skin.tone = v; })} />
                  <Slider id="ed-under" label="Undertone" min={-1} max={1} def={0} ends={['Cool', 'Warm']} colour paintTrack={underTrack}
                    get={() => app.look().skin.undertone} onStart={() => ed.begin('Undertone')} onEnd={() => ed.end()} onInput={(v) => ed.queue((n) => { n.skin.undertone = v; })} />
                </div>
              </PopoverContent>
            </Popover>
          <span className="osb-note">Custom</span>
        </div>
        <div className="osb-swatch-head">
          <span className="osb-note" style={{ flex: 1 }}>Undertone</span>
          <ToggleGroup type="single" aria-label="Undertone" value={under} onValueChange={pickUnder}>
            {UNDERTONES.map(([k, , label]) => <ToggleGroupItem key={k} value={k}>{label}</ToggleGroupItem>)}
          </ToggleGroup>
        </div>
      </div>
      <div className="osb-section">
        <h3 className="osb-eyebrow">Eyes <span ref={eyeName} className="osb-eyebrow-val" /></h3>
        <RadioGroup aria-label="Eye colour" value={eye} onValueChange={pickEye}>
          {app.palettes.eyes.map(([name, hex]) => (
            <Tip key={hex} label={name} side="top"><RadioGroupItem value={hex} aria-label={name} style={{ background: iris(hex) }} /></Tip>
          ))}
        </RadioGroup>
        <div className="osb-swatch-head osb-custom-row">
            <Tip label="Custom eye colour" side="top">
              <label className="osb-sw osb-sw--custom osb-sw-input" data-state={eye === 'custom' ? 'checked' : 'unchecked'} aria-checked={eye === 'custom'}>
                <Icon name="eyedropper" size={14} />
                <input ref={eyeInput} type="color" aria-label="Custom eye colour" onInput={(e) => { const v = (e.target as HTMLInputElement).value.toLowerCase(); ed.edit('Eye colour', (n) => { n.eyes.color = v; }); }} />
              </label>
            </Tip>
          <span className="osb-note">Custom</span>
        </div>
      </div>
    </section>
  );
}

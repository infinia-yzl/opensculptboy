// OpenSculptBoy
// Apache License, Version 2.0
//
// Create's Hair section in React: the page's 25 styles by length, the shape sliders that the chosen style has, the
// side of the part, the colour swatches with a custom colour, and the switches for the hair and its physics. With a
// card that brings its own hair, a note says so; the page's hair then shows only when the card's is off. The style
// and the family are React state that changes on a click; the sliders follow the look through the bus.

import { useRef, useState } from 'react';
import { useShell } from '../context.ts';
import { useBus } from '../hooks.ts';
import { Slider } from '../components/Slider.tsx';
import { Note, Row, Seg, Switch } from '../components/controls.tsx';
import { Icon } from '../components/ui/icon.tsx';
import { RadioGroup, RadioGroupItem } from '../components/ui/radio-group.tsx';
import { Tip } from '../components/ui/tooltip.tsx';

const FAMILIES = [{ value: 'short', label: 'Short' }, { value: 'medium', label: 'Medium' }, { value: 'long', label: 'Long' }, { value: 'tied', label: 'Tied' }];
const SHAPE = [
  { key: 'length', label: 'Length', ends: ['Shorter', 'Longer'] },
  { key: 'curl', label: 'Curl', ends: ['Straight', 'Curly'] },
  { key: 'volume', label: 'Volume', ends: ['Flat', 'Full'] },
  { key: 'density', label: 'Density', ends: ['Thin', 'Thick'] },
  { key: 'fade', label: 'Fade height', ends: ['Low', 'High'] },
] as const;

export function HairSection({ shown }: { shown: boolean }) {
  const { app, ed } = useShell();
  const specs = app.hairSpecs();
  const [style, setStyle] = useState(app.look().hair.style);
  const [family, setFamily] = useState(specs.find((s) => s.name === style)?.family || 'medium');
  const [hair, setHair] = useState({ visible: app.hairVisible(), physics: app.hairPhysics() });
  const [colour, setColour] = useState(app.look().hair.color);
  const [cardHair, setCardHair] = useState(false);
  const [part, setPart] = useState(app.look().hair.part || 'left');
  const last = useRef(style), name = useRef<HTMLSpanElement>(null), custom = useRef<HTMLInputElement>(null);
  useBus(['look', 'hair', 'wardrobe'], () => {
    const L = app.look();
    if (L.hair.style !== last.current) {
      last.current = L.hair.style; setStyle(L.hair.style);
      const sp = specs.find((s) => s.name === L.hair.style);
      if (sp) setFamily(sp.family);
    }
    setColour(L.hair.color); setPart(L.hair.part || 'left');
    setHair({ visible: app.hairVisible(), physics: app.hairPhysics() });
    setCardHair(!!app.cardLook()?.summary()?.hair);
    const pal = app.palettes.hair.find(([, hex]) => hex === L.hair.color);
    if (name.current) name.current.textContent = pal ? pal[0] : 'Custom ' + L.hair.color;
    if (custom.current && document.activeElement !== custom.current) custom.current.value = L.hair.color;
  });
  const spec = specs.find((s) => s.name === style);
  const d = spec ? app.hairDefaults(spec.name) || {} : {};
  return (
    <section id="sec-hair" className="sec osb-section-stack" role="tabpanel" aria-labelledby="tab-hair" hidden={!shown}>
      {cardHair ? <Note className="osb-override">The card's own hair is shown. The page's styles below show when the card is removed.</Note> : null}
      <div className="osb-section">
        <h3 className="osb-eyebrow">Style <span className="osb-eyebrow-val">{spec?.label || ''}</span></h3>
        <div className="switch-row">
          <Switch label="Show" on={hair.visible} onChange={(on) => app.setHairVisible(on)} tip="Show or hide the hair (Shift H)" />
          <Switch label="Physics" on={hair.physics} onChange={(on) => app.setHairPhysics(on)} tip="The hair follows the motion of the head" />
        </div>
        <Seg label="Hair length" items={FAMILIES} value={family} onPick={setFamily} fill />
        <div className="style-grid osb-style-grid" role="group" aria-label="Hair styles">
          {specs.filter((s) => s.family === family).map((s) => (
            <button key={s.name} type="button" className="style-card" aria-pressed={s.name === style} data-style={s.name} data-family={s.family}
              onClick={() => ed.edit('Hair: ' + s.label, (n) => { n.hair = { color: n.hair.color, style: s.name, part: n.hair.part }; })}><span>{s.label}</span></button>
          ))}
        </div>
      </div>
      {spec ? (
        <div className="osb-section">
          <h3 className="osb-eyebrow">Shape</h3>
          <div className="sliders osb-sliders">
            {SHAPE.map((p) => {
              const range = p.key === 'fade' ? app.fadeRange : spec.controls[p.key];
              const on = p.key === 'fade' ? !!spec.render.fade : range[1] > range[0];
              if (!on) return null;
              return (
                <Slider key={spec.name + p.key} id={'ed-h-' + p.key} label={p.label} min={range[0]} max={range[1]} def={p.key === 'fade' ? 0 : d[p.key] ?? 0} ends={[...p.ends]}
                  digits={p.key === 'fade' ? 0 : 2} step={p.key === 'fade' ? 1 : 0.01} readout={p.key === 'fade' ? (v) => `${v > 0 ? '+' : ''}${v.toFixed(0)}°` : undefined}
                  get={() => (app.look().hair as any)[p.key] ?? 0}
                  onStart={() => ed.begin('Hair ' + p.label.toLowerCase())} onEnd={() => ed.end()} onInput={(v) => ed.queue((n) => { (n.hair as any)[p.key] = v; })} />
              );
            })}
          </div>
          {spec.mirror ? (
            <Row label="Part"><Seg label="Part" items={[{ value: 'left', label: 'Left' }, { value: 'right', label: 'Right' }]} value={part}
              onPick={(v) => ed.edit('Hair part', (n) => { n.hair.part = v; })} /></Row>
          ) : null}
        </div>
      ) : null}
      <div className="osb-section">
        <h3 className="osb-eyebrow">Colour <span ref={name} className="osb-eyebrow-val" /></h3>
        <RadioGroup aria-label="Hair colour" value={app.palettes.hair.some(([, hex]) => hex === colour) ? colour : 'custom'} onValueChange={(hex) => { if (hex !== 'custom') ed.edit('Hair colour', (n) => { n.hair.color = hex; }); }}>
          {app.palettes.hair.map(([nm, hex]) => <Tip key={hex} label={nm} side="top"><RadioGroupItem value={hex} aria-label={nm} data-hex={hex} style={{ background: hex }} /></Tip>)}
        </RadioGroup>
        <div className="osb-swatch-head osb-custom-row">
          <Tip label="Custom hair colour" side="top">
            <label className="osb-sw osb-sw--custom osb-sw-input" data-state={app.palettes.hair.some(([, hex]) => hex === colour) ? 'unchecked' : 'checked'}>
              <Icon name="eyedropper" size={14} />
              <input ref={custom} type="color" aria-label="Custom hair colour" onInput={(e) => { const v = (e.target as HTMLInputElement).value.toLowerCase(); ed.edit('Hair colour', (n) => { n.hair.color = v; }); }} />
            </label>
          </Tip>
          <span className="osb-note">Custom</span>
        </div>
      </div>
    </section>
  );
}

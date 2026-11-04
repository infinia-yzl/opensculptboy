// OpenSculptBoy
// Apache License, Version 2.0
//
// The Controls panel (DESIGN-0014, "Settings"; More › Controls): four speeds from half to twice their default, each
// with its reset, Larger targets, and the Blender middle mouse. The settings stay in this browser (studio/settings.ts).

import { useState } from 'react';
import { Slider } from '../shell/components/Slider.tsx';
import { Switch } from '../shell/components/controls.tsx';
import { DEFAULTS, settings, SPEED_RANGE, SPEEDS } from './settings.ts';
import { useSettings } from './react.ts';

export function ControlsPanel({ shown }: { shown: boolean }) {
  const s = useSettings();
  const [gen, setGen] = useState(0);
  return (
    <section className="sec controls-panel" id="sec-controls" hidden={!shown}>
      {SPEEDS.map((p) => (
        <Slider key={p.key + gen} id={'ctl-' + p.key} label={p.label} title={p.note} min={SPEED_RANGE[0]} max={SPEED_RANGE[1]} step={0.05} def={DEFAULTS[p.key] as number}
          digits={2} ends={['Half', 'Twice']} readout={(v) => `${Math.round(v * 100)} %`} get={() => settings.get()[p.key] as number}
          onInput={(v) => settings.set({ [p.key]: v })} />
      ))}
      <Switch label="Larger targets" on={s.largerTargets} onChange={(on) => settings.set({ largerTargets: on })}
        tip="Points, handles and rings draw larger, and a click picks within 24 pixels instead of 16" />
      <Switch label="Middle mouse turns the camera (Blender)" on={s.blenderMiddle} onChange={(on) => settings.set({ blenderMiddle: on })}
        tip="Middle-drag turns and Shift+middle-drag moves the view, as in Blender" />
      <Switch label="Ask before leaving with unsaved edits" on={s.warnUnsaved} onChange={(on) => settings.set({ warnUnsaved: on })}
        tip="The browser asks before the page closes or reloads while posed points or a pose are not saved" />
      <p className="osb-note">These settings stay in this browser.</p>
      <button type="button" className="osb-btn osb-btn--ghost" onClick={() => { settings.reset(); setGen(gen + 1); }}>Reset the controls</button>
    </section>
  );
}

// OpenSculptBoy
// Apache License, Version 2.0
//
// The Export mode: the files that the page writes today, as cards with an icon, a name and one line, and the files
// that it will write, marked Planned with the command that writes them now.

import { useState } from 'react';
import { copyText } from '../../ui/dom.ts';
import { toast } from '../../ui/toast.ts';
import { useShell } from '../context.ts';
import { actions } from '../state.ts';
import { Icon } from '../components/ui/icon.tsx';

interface Target { id: string; icon: string; name: string; ext: string; line: string; run?: () => void; command?: string }

export function ExportSection({ shown }: { shown: boolean }) {
  const { legacy } = useShell();
  const [copied, setCopied] = useState('');
  const now: Target[] = [
    { id: 'picture', icon: 'picture', name: 'Picture', ext: '.png', line: 'The view as it shows', run: legacy.saveImage },
    { id: 'sheet', icon: 'sheet', name: 'Sheet', ext: '.png', line: 'Turnaround, expressions and poses', run: () => actions.openMode('sheet') },
    { id: 'pose', icon: 'pose', name: 'Pose card', ext: '.json', line: 'The body and its pose', run: legacy.poseCard },
    { id: 'silhouette', icon: 'silhouette', name: 'Silhouette', ext: '.svg', line: 'A flat drawing of the pose', run: legacy.silhouette },
    { id: 'look', icon: 'json', name: 'Page character', ext: '.json', line: 'Body, face, hair and colours; opens again here', run: legacy.downloadLook },
  ];
  const planned: Target[] = [
    { id: 'glb', icon: 'glb', name: 'GLB', ext: '.glb', line: 'Games and 3D apps', command: 'opensculptboy export character.glb --character card.json --tier medium' },
    { id: 'vrm', icon: 'vrm', name: 'VRM', ext: '.vrm', line: 'Avatar apps', command: 'opensculptboy export character.vrm --character card.json --author NAME' },
    { id: 'card', icon: 'zip', name: 'Card', ext: '.zip', line: 'The character card with its files; opens again here' },
  ];
  const card = (t: Target, on: boolean) => (
    <button key={t.id} type="button" className="osb-card osb-export-card" data-target={t.id} disabled={!on || (legacy.hosted && t.id !== 'sheet' && t.id !== 'picture')} onClick={t.run}>
      <span className="osb-export-top"><Icon name={t.icon} size={24} /><span className="osb-tag osb-tag--mono">{t.ext}</span></span>
      <b className="osb-export-name">{t.name}</b>
      <span className="osb-note">{t.line}</span>
      {on ? null : <span className="osb-tag osb-export-planned">Planned</span>}
    </button>
  );
  const copy = async (t: Target) => {
    if (await copyText(t.command!)) { setCopied(t.id); toast('Command copied'); setTimeout(() => setCopied(''), 1500); }
    else toast('Select the command and copy it', 'error');
  };
  return (
    <section id="sec-export" className="sec osb-section-stack" role="tabpanel" aria-labelledby="tab-export" hidden={!shown}>
      <div className="osb-section">
        <h3 className="osb-eyebrow">From this page</h3>
        <div className="osb-export-grid">{now.map((t) => card(t, true))}</div>
        {legacy.hosted ? <p className="osb-note">This copy of the page cannot start downloads: the picture and the sheet open as pictures to save by hand.</p> : null}
      </div>
      <div className="osb-section">
        <h3 className="osb-eyebrow">Planned</h3>
        <div className="osb-export-grid">{planned.map((t) => card(t, false))}</div>
        <p className="osb-note">Until the page writes them, the command line writes GLB and VRM files from a character card:</p>
        <div className="osb-commands">
          {planned.filter((t) => t.command).map((t) => (
            <div key={t.id} className="osb-command">
              <code>{t.command}</code>
              <button type="button" className="osb-ib" aria-label={'Copy the command for ' + t.name} onClick={() => copy(t)}><Icon name={copied === t.id ? 'check' : 'copy'} size={16} /></button>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

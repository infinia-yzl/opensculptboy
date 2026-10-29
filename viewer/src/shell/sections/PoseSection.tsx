// OpenSculptBoy
// Apache License, Version 2.0
//
// The Pose mode's section in React: the animations with a timeline and a speed, the 50 poses of the library in folds
// with a search, From a picture (MediaPipe reads the figure's pose, with sliders for the head's turn when the picture
// leaves it unclear), the pose card and the silhouette, and the soft tissue. The chosen motion is React state that
// changes with the motion (the bus's motion events); the timeline's time and thumb paint through refs on the bus's
// tick, a tenth of a second apart, so a playing clip renders nothing in React.

import { useRef, useState } from 'react';
import { useShell } from '../context.ts';
import { useBus } from '../hooks.ts';
import { Slider } from '../components/Slider.tsx';
import { Chip, Fold, Note, Row, Search, Seg, Sub, Switch, TextButton } from '../components/controls.tsx';
import { Icon } from '../components/ui/icon.tsx';

export function PoseSection({ shown }: { shown: boolean }) {
  const { app, legacy } = useShell();
  const clips = app.clips();
  const loops = clips.filter((c) => c.kind === 'loop'), poses = clips.filter((c) => c.kind === 'pose');
  const groups = [...new Set(poses.map((c) => c.group || 'Poses'))];
  const [m, setM] = useState(() => { const x = app.motion(); return { name: x.cur?.name || '', loop: x.cur?.kind === 'loop', playing: x.playing, speed: x.speed }; });
  const [tissue, setTissue] = useState(app.correctives());
  const [q, setQ] = useState('');
  const [msg, setMsg] = useState<{ text: string; kind: string }>({ text: '', kind: '' });
  const [busy, setBusy] = useState(false);
  const [head, setHead] = useState(app.pictureHead());
  const scrub = useRef<HTMLInputElement>(null), time = useRef<HTMLSpanElement>(null), file = useRef<HTMLInputElement>(null);
  const scrubbing = useRef(false), headVals = useRef({ turn: 0, up: 0, tilt: 0 });
  useBus(['motion', 'look'], () => {
    const x = app.motion();
    const next = { name: x.cur?.name || '', loop: x.cur?.kind === 'loop', playing: x.playing, speed: x.speed };
    setM((o) => (o.name === next.name && o.loop === next.loop && o.playing === next.playing && o.speed === next.speed ? o : next));
    const c = app.correctives();
    setTissue((o) => (o.ready === c.ready && o.on === c.on ? o : c));
  });
  useBus('tick', () => {
    const x = app.motion();
    if (!x.cur || x.cur.kind !== 'loop') { if (time.current) time.current.textContent = ''; return; }
    const d = x.cur.duration, t = ((x.t % d) + d) % d;
    if (!scrubbing.current && scrub.current) scrub.current.value = String(t / d);
    if (time.current) time.current.textContent = `${t.toFixed(1)} / ${d.toFixed(1)} s`;
  });
  const choose = (name: string) => {
    const x = app.motion(), c = clips.find((y) => y.name === name)!;
    if (c.kind === 'loop' && x.cur && x.cur.name === name) app.togglePlay(); else app.setMotion(name);
  };
  const cur = clips.find((c) => c.name === m.name);
  const query = q.trim().toLowerCase();
  const pick = async (f: File) => {
    setBusy(true);
    setMsg({ text: 'Reading the pose. The first picture loads MediaPipe, about 20 MB.', kind: '' });
    try {
      const r = await app.poseFromPicture(f);
      const h = app.pictureHead();
      if (h) headVals.current = { ...h };
      setHead(h);
      setMsg({ text: r.hands === 2 ? 'Posed from the picture, with both hands.' : r.hands === 1 ? 'Posed from the picture, with one hand. The other hand keeps its rest pose.' : 'Posed from the picture. No hands were clear enough, so they keep their rest pose.', kind: '' });
    } catch (e) {
      setMsg({ text: e instanceof Error && e.message.startsWith('No figure') ? 'No figure was found in this picture. Try one where the whole figure shows.' : 'The picture could not be read. MediaPipe loads from the internet the first time.', kind: 'error' });
    } finally { setBusy(false); }
  };
  const headSlider = (key: 'turn' | 'up' | 'tilt', label: string, min: number, max: number, ends: string[]) => (
    <Slider key={key} id={'picture-head-' + key} label={label + ' (°)'} min={min} max={max} step={1} def={0} digits={0} ends={ends} events={['motion']}
      get={() => headVals.current[key]} onInput={(v) => { headVals.current[key] = v; app.setPictureHead({ ...headVals.current }); }} />
  );
  const hosted = legacy.hosted;
  return (
    <section id="sec-pose" className="sec osb-section-stack" role="tabpanel" aria-labelledby="tab-pose" hidden={!shown}>
      <div className="osb-section">
        <h3 className="osb-eyebrow">Animations <span className="osb-eyebrow-val">{cur ? cur.label + (m.loop && !m.playing ? ' · paused' : '') : ''}</span></h3>
        <div className="chips osb-chips" role="group" aria-label="Animations">
          {loops.map((c) => <Chip key={c.name} pressed={c.name === m.name} data-motion={c.name} onClick={() => choose(c.name)}>{c.label}</Chip>)}
        </div>
        <div className="timeline" data-off={m.loop ? undefined : ''}>
          <button type="button" className="play" aria-label={m.loop && m.playing ? 'Pause' : 'Play'} title="Play or pause (Space)" disabled={!m.loop} onClick={() => app.togglePlay()}>
            <Icon name={m.loop && m.playing ? 'pause' : 'play'} size={18} />
          </button>
          <input ref={scrub} className="scrub" type="range" min={0} max={1} step={0.001} defaultValue={0} aria-label="Animation time" disabled={!m.loop}
            onInput={(e) => {
              scrubbing.current = true;
              const x = app.motion(); if (!x.cur || x.cur.kind !== 'loop') return;
              if (x.playing) app.togglePlay();
              app.seek(parseFloat((e.target as HTMLInputElement).value) * x.cur.duration);
            }} onChange={() => { scrubbing.current = false; }} />
          <span ref={time} className="scrub-time" />
        </div>
        <Row label="Speed"><Seg label="Speed" items={[{ value: '0.25', label: '¼×' }, { value: '0.5', label: '½×' }, { value: '1', label: '1×' }]} value={String(m.speed)} onPick={(v) => app.setSpeed(parseFloat(v))} /></Row>
        <Note>The view sharpens while the figure holds still. Pause an animation to see any moment at full quality.</Note>
      </div>
      <div className="osb-section">
        <Sub>Poses</Sub>
        <Search placeholder={`Search ${poses.length} poses`} value={q} onChange={setQ} />
        <div className="folds osb-folds">
          {groups.map((g) => {
            const list = poses.filter((c) => (c.group || 'Poses') === g && (!query || c.label.toLowerCase().includes(query) || g.toLowerCase().includes(query)));
            if (!list.length) return null;
            const holds = list.some((c) => c.name === m.name);
            return (
              <Fold key={g + (query ? ':q' : '') + (holds ? ':h' : '')} title={g} count={list.length} open={query || holds ? true : undefined}>
                <div className="chips osb-chips">{list.map((c) => <Chip key={c.name} pressed={c.name === m.name} data-motion={c.name} onClick={() => choose(c.name)}>{c.label}</Chip>)}</div>
              </Fold>
            );
          })}
        </div>
        {cur?.credit ? <Note className="credit">Pose by {cur.credit}, from the MakeHuman community (CC0).</Note> : null}
      </div>
      <div className="osb-section">
        <Sub>From a picture</Sub>
        <Note>Pose the figure as the figure in a picture: the body, the head and the fingers. The picture stays in this browser.</Note>
        <div className="btn-row"><TextButton label="Choose a picture" icon="upload" disabled={busy} onClick={() => file.current?.click()} /></div>
        <p className="inline-msg" role="status" data-kind={msg.kind}>{msg.text}</p>
        <input ref={file} type="file" accept="image/*" hidden onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ''; if (f) pick(f); }} />
        {head ? (
          <div className="picture-head">
            <Note>When the picture leaves the head unclear, as a drawn or stylised face can, turn the head here. Left and right are the figure's own.</Note>
            {headSlider('turn', 'Head turn', -60, 60, ['Left', 'Right'])}{headSlider('up', 'Head up', -40, 60, ['Down', 'Up'])}{headSlider('tilt', 'Head tilt', -40, 40, ['Left', 'Right'])}
          </div>
        ) : null}
        {hosted ? null : (<>
          <Note>Save the pose shown as a character card for opensculptboy export, or as a silhouette from this view.</Note>
          <div className="btn-row"><TextButton label="Pose card" icon="download" onClick={legacy.poseCard} /><TextButton label="Silhouette" icon="download" onClick={legacy.silhouette} /></div>
        </>)}
      </div>
      {tissue.ready ? (
        <div className="osb-section">
          <Sub>Soft tissue</Sub>
          <Switch label="Soft tissue" on={tissue.on} onChange={(on) => app.setCorrectives(on)} tip="Corrective shapes keep the volume of the joints" />
          <Note>Corrective shapes from a soft-tissue simulation keep the volume of the shoulders, elbows, hips and knees as they bend. Switch them off to compare with plain skinning.</Note>
        </div>
      ) : null}
    </section>
  );
}

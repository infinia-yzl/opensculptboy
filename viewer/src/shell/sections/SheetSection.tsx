// OpenSculptBoy
// Apache License, Version 2.0
//
// The Sheet mode's section in React: Build sheet and Download PNG with the build's progress, the style (the styles
// still to come show as planned), the background, the lights and their exposure, the quality, and the panels, each
// with a switch and its settings: the info block's name and title, the turnaround's pose and views, the expression
// grid's pose, view, framing and expressions, and the pose slots, each opening the pose picker (the library or From
// a picture). The settings, the build and the layout view live in sheet/controller.ts; this section subscribes to it.

import { useState, useSyncExternalStore } from 'react';
import type { SheetController } from '../../sheet/controller.ts';
import { defaultSettings, expressionLabel, MAX_POSES, PANEL_LABELS, RENDER_PRESETS, VIEWS, type PanelId, type PoseSlot, type ViewName } from '../../sheet/settings.ts';
import { STYLES } from '../../sheet/styles.ts';
import { saveUi } from '../../ui/dom.ts';
import { Chip, Fold, Note, Row, Search, Seg, Sub, Switch, TextButton } from '../components/controls.tsx';
import { Icon } from '../components/ui/icon.tsx';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '../components/ui/dialog.tsx';

const PLANNED = ['Toon', 'Ink', 'Flat colour', 'Clay', 'Silhouette'];
const VIEW3: ViewName[] = ['front', 'three', 'side'];
const viewItems = (views: ViewName[]) => views.map((v) => ({ value: v, label: VIEWS[v].label, short: v === 'three' ? '¾' : undefined }));

// the pose picker: the library in folds with a search, and From a picture
function PosePicker({ sheet, title, slot, onPick, onClose }: { sheet: SheetController; title: string; slot: PoseSlot | null; onPick: (s: PoseSlot) => void; onClose: () => void }) {
  const [q, setQ] = useState('');
  const [msg, setMsg] = useState({ text: '', kind: '' });
  const [busy, setBusy] = useState(false);
  const poses = sheet.poses(), groups = [...new Set(poses.map((p) => p.group || 'Poses'))], query = q.trim().toLowerCase();
  return (
    <Dialog open={!!slot} onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent className="cs-picker" aria-describedby="cs-picker-lead">
        <DialogTitle id="cs-picker-title" className="osb-title">{title}</DialogTitle>
        <DialogDescription id="cs-picker-lead" className="osb-note">A pose of the library, or the pose of a figure in a picture.</DialogDescription>
        <div className="cs-picker-body">
          <Search placeholder="Search the poses" value={q} onChange={setQ} />
          <div className="folds osb-folds">
            {groups.map((g) => {
              const list = poses.filter((p) => (p.group || 'Poses') === g && (!query || p.label.toLowerCase().includes(query) || g.toLowerCase().includes(query)));
              if (!list.length) return null;
              const holds = !!slot && !slot.q && list.some((p) => p.name === slot.pose);
              return (
                <Fold key={g + (query ? ':q' : '')} title={g} count={list.length} open={query || holds ? true : undefined}>
                  <div className="chips osb-chips">{list.map((p) => <Chip key={p.name} pressed={!!slot && !slot.q && slot.pose === p.name} data-pose={p.name} onClick={() => onPick({ pose: p.name })}>{p.label}</Chip>)}</div>
                </Fold>
              );
            })}
          </div>
          <Sub>From a picture</Sub>
          <Note>Pose the figure as the figure in a picture. The picture stays in this browser.</Note>
          <label className="tb osb-btn" aria-disabled={busy}>
            <Icon name="upload" size={16} /><span>Choose a picture</span>
            <input type="file" accept="image/*" hidden disabled={busy} onChange={async (e) => {
              const f = e.target.files?.[0]; e.target.value = '';
              if (!f) return;
              setBusy(true); setMsg({ text: 'Reading the pose. The first picture starts MediaPipe, which takes a few seconds.', kind: '' });
              try { onPick(await sheet.poseFromPicture(f)); } catch (err) { setMsg({ text: 'The picture gave no pose. Try one where the whole figure shows.', kind: 'error' }); } finally { setBusy(false); }
            }} />
          </label>
          <p className="inline-msg" role="status" data-kind={msg.kind}>{msg.text}</p>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function SheetSection({ shown, sheet }: { shown: boolean; sheet: SheetController }) {
  // the controller tells of every change; the section reads its settings and its state at render
  const rev = useSyncExternalStore(sheet.subscribe, () => JSON.stringify(sheet.state()) + JSON.stringify(sheet.settings()));
  void rev;
  const S = sheet.settings(), st = sheet.state(), P = S.panels;
  const [picker, setPicker] = useState<{ title: string; slot: PoseSlot; done: (s: PoseSlot) => void } | null>(null);
  const [open, setOpen] = useState<Record<string, boolean>>({ turnaround: true });
  const change = sheet.change;
  const poseButton = (title: string, slot: PoseSlot, done: (s: PoseSlot) => void) => (
    <button type="button" className="cs-pose" title="Change the pose" onClick={() => setPicker({ title, slot, done })}>
      <Icon name="pose" size={16} /><span>{sheet.poseLabel(slot)}</span><Icon name="chevron" size={14} />
    </button>
  );
  const panelFold = (id: PanelId, body: React.ReactNode) => {
    const p: any = P[id];
    return (
      <Fold key={id} title={PANEL_LABELS[id]} data-panel={id} data-off={p.on ? undefined : ''} open={open[id] ?? false} onToggle={(o) => setOpen((x) => ({ ...x, [id]: o }))}
        actions={<Switch label="" on={p.on} className="cs-switch" aria-label={'Show the ' + PANEL_LABELS[id].toLowerCase()} tip="Show this panel on the sheet" onChange={(on) => change((s: any) => { s.panels[id].on = on; })} />}>
        {body}
      </Fold>
    );
  };
  const field = (label: string, value: string, placeholder: string, set: (v: string) => void) => (
    <label className="cs-field"><span>{label}</span>
      <input className="cs-input" type="text" value={value} placeholder={placeholder} autoComplete="off" spellCheck={false} aria-label={label}
        onChange={(e) => change(() => set(e.target.value), { quiet: true })} />
    </label>
  );
  const names = sheet.expressionNames();
  const quality = S.render.scale === RENDER_PRESETS.draft.scale && S.render.samples === RENDER_PRESETS.draft.samples ? 'draft' : S.render.scale === 1 && S.render.samples === 24 ? 'print' : '';
  return (
    <section id="sec-sheet" className="sec osb-section-stack" role="tabpanel" aria-labelledby="tab-sheet" hidden={!shown}>
      <div className="osb-section">
        <h3 className="osb-eyebrow">Character sheet <span className="osb-eyebrow-val">{st.building ? 'Building' : st.built ? 'Built' : ''}</span></h3>
        <Note className="sec-lead">A character sheet of the figure shown: its turnaround, its expressions and its poses on one page, to share or to print.</Note>
        <div className="btn-row">
          <TextButton label="Build sheet" icon="sheet" primary disabled={st.building} onClick={sheet.build} />
          <TextButton label="Download PNG" icon="download" onClick={sheet.downloadPng} />
          {st.building ? <TextButton label="Stop" icon="stop" quiet onClick={sheet.stop} /> : null}
        </div>
        <div className="cs-bar" aria-hidden="true" data-on={st.progress > 0 && st.progress < 1 ? '' : undefined}><span style={{ transform: `scaleX(${st.progress})` }} /></div>
        <p className="inline-msg cs-status" role="status">{st.label}</p>
        {!st.shown ? <TextButton label="Show the page" icon="sheet" quiet onClick={() => sheet.showPage(true)} /> : null}
      </div>
      <div className="osb-section">
        <Sub>Style</Sub>
        <div className="seg osb-seg seg-fill osb-seg--fill cs-styles" role="group" aria-label="Style">
          {STYLES.map((s) => <button key={s.id} type="button" aria-pressed={S.style === s.id} title={s.tip} onClick={() => change((x) => { x.style = s.id; })}><span>{s.label}</span></button>)}
        </div>
        <div className="cs-planned">{PLANNED.map((p) => <span key={p} className="osb-tag" title="Planned">{p} <small>Planned</small></span>)}</div>
        <Sub>Background</Sub>
        <Seg label="Background" fill value={S.background} onPick={(v) => { saveUi({ sheetBackgroundPicked: true }); change((x) => { x.background = v as any; }); }}
          items={[{ value: 'grey', label: 'Grey', tip: 'A warm mid-dark grey: the figures stand out from it' }, { value: 'paper', label: 'Paper', tip: 'Light paper, for print' },
            { value: 'dark', label: 'Dark' }, { value: 'transparent', label: 'Transparent', tip: 'No ground, with dark text' }]} />
        <Sub>Lights</Sub>
        <Seg label="Lights" fill value={S.render.lights} onPick={(v) => change((x) => { x.render.lights = v as any; })}
          items={[{ value: 'studio', label: 'Studio', tip: 'Studio lights: a key, a fill and rims, with the exposure metered on the figure' }, { value: 'page', label: 'The page\'s', tip: 'The lighting preset and the exposure of the Scene panel' }]} />
        {S.render.lights === 'studio' ? (
          <div className="osb-sliders">
            <ExposureSlider sheet={sheet} />
          </div>
        ) : null}
        <Sub>Quality</Sub>
        <Seg label="Quality" fill value={quality} onPick={(v) => change((x) => { x.render = { ...x.render, ...RENDER_PRESETS[v] }; })}
          items={[{ value: 'print', label: 'Print', tip: 'Pictures at print size, 24 samples each' }, { value: 'draft', label: 'Draft', tip: 'Half the size and 8 samples: a quick look' }]} />
      </div>
      <div className="osb-section">
        <Sub>Panels</Sub>
        <div className="folds osb-folds cs-panels">
          {panelFold('info', (<>
            {field('Name', S.name, sheet.characterName(), (v) => { S.name = v; })}
            {field('Title', S.title, 'A line under the name', (v) => { S.title = v; })}
          </>))}
          {panelFold('turnaround', (<>
            <Note>The full figure from four sides, at one scale on one ground line.</Note>
            <Row label="Pose">{poseButton('The turnaround\'s pose', P.turnaround.pose, (s) => change((x) => { x.panels.turnaround.pose = s; }))}</Row>
            <div className="chips osb-chips" role="group" aria-label="Views">
              {(Object.keys(VIEWS) as ViewName[]).map((v) => (
                <Chip key={v} pressed={P.turnaround.views.includes(v)} onClick={() => change((x) => {
                  const on = x.panels.turnaround.views.includes(v);
                  x.panels.turnaround.views = (Object.keys(VIEWS) as ViewName[]).filter((y) => (y === v ? !on : x.panels.turnaround.views.includes(y)));
                })}>{VIEWS[v].label}</Chip>
              ))}
            </div>
          </>))}
          {panelFold('expressions', (<>
            <Note>One picture per expression, framed on the face or on the head and shoulders.</Note>
            <Row label="Pose">{poseButton('The expressions\' pose', P.expressions.pose, (s) => change((x) => { x.panels.expressions.pose = s; }))}</Row>
            <Seg label="View" fill items={viewItems(VIEW3)} value={P.expressions.view} onPick={(v) => change((x) => { x.panels.expressions.view = v as ViewName; })} />
            <Seg label="Framing" fill value={P.expressions.framing} onPick={(v) => change((x) => { x.panels.expressions.framing = v as any; })}
              items={[{ value: 'face', label: 'Face', tip: 'From below the chin to above the head: the face fills most of the picture' }, { value: 'shoulders', label: 'Head and shoulders', tip: 'The head with the shoulders and the top of the chest' }]} />
            <div className="chips osb-chips" role="group" aria-label="Expressions">
              {names.map((n) => (
                <Chip key={n} pressed={P.expressions.items.includes(n)} data-expression={n} onClick={() => change((x) => {
                  const on = x.panels.expressions.items.includes(n);
                  x.panels.expressions.items = on ? x.panels.expressions.items.filter((y) => y !== n) : [...x.panels.expressions.items, n].sort((a, b) => names.indexOf(a) - names.indexOf(b));
                })}>{expressionLabel(n)}</Chip>
              ))}
            </div>
            {sheet.faces() ? null : <Note>This page shows every expression with the neutral face for now.</Note>}
          </>))}
          {panelFold('poses', (<>
            <Note>Full-figure poses that show the character's manner, at one scale.</Note>
            <Seg label="View" fill items={viewItems(VIEW3)} value={P.poses.view} onPick={(v) => change((x) => { x.panels.poses.view = v as ViewName; })} />
            <div className="cs-slots">
              {P.poses.slots.map((slot, i) => (
                <div key={i} className="cs-slot" data-slot={i}>
                  <span className="cs-n">{i + 1}</span>
                  {poseButton(`Pose ${i + 1}`, slot, (s) => change((x) => { x.panels.poses.slots[i] = s; }))}
                  <button type="button" className="osb-ib" aria-label="Remove this pose" onClick={() => change((x) => { x.panels.poses.slots.splice(i, 1); })}><Icon name="close" size={16} /></button>
                </div>
              ))}
            </div>
            <div className="btn-row">
              <TextButton label="Add a pose" icon="plus" quiet disabled={P.poses.slots.length >= MAX_POSES}
                onClick={() => setPicker({ title: `Pose ${P.poses.slots.length + 1}`, slot: { pose: '' }, done: (s) => change((x) => { x.panels.poses.slots.push(s); }) })} />
              <TextButton label="Default poses" icon="undoSmall" quiet onClick={() => change((x) => { x.panels.poses.slots = defaultSettings().panels.poses.slots; })} />
            </div>
          </>))}
        </div>
      </div>
      <PosePicker sheet={sheet} title={picker?.title || ''} slot={picker?.slot || null} onClose={() => setPicker(null)}
        onPick={(s) => { const p = picker; setPicker(null); p?.done(s); }} />
    </section>
  );
}

// the exposure over the meter's: the settings change on each step, the pictures follow when the drag ends
function ExposureSlider({ sheet }: { sheet: SheetController }) {
  const S = sheet.settings();
  return (
    <SliderLite id="cs-exp" label="Exposure (stops)" min={-1.5} max={1.5} step={0.05} value={S.render.exposure}
      onInput={(v) => sheet.change((x) => { x.render.exposure = Math.round(v * 100) / 100; }, { quiet: true })} onEnd={() => sheet.change(() => {})} />
  );
}

// a plain range slider for a setting of the sheet (its value is React state of the controller, not the look)
function SliderLite({ id, label, min, max, step, value, onInput, onEnd }: { id: string; label: string; min: number; max: number; step: number; value: number; onInput: (v: number) => void; onEnd: () => void }) {
  return (
    <div className="sl osb-sl" data-changed={Math.abs(value) > 1e-6 ? '' : undefined}>
      <div className="osb-sl-head"><label htmlFor={id} title="Stops over the exposure that the meter sets for the studio lights">{label}</label><span className="osb-sl-read">{value > 0 ? '+' : ''}{value.toFixed(2)}</span></div>
      <div className="osb-sl-track">
        <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onInput(parseFloat(e.target.value))} onPointerUp={onEnd} onKeyUp={onEnd}
          style={{ ['--lo' as any]: `${Math.min(50, (value - min) / (max - min) * 100)}%`, ['--hi' as any]: `${Math.max(50, (value - min) / (max - min) * 100)}%` }} />
      </div>
      <div className="osb-sl-ends" aria-hidden="true"><span>Darker</span><span>Brighter</span></div>
    </div>
  );
}

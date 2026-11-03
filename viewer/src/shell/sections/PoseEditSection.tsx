// OpenSculptBoy
// Apache License, Version 2.0
//
// The Pose mode's Edit panel (DESIGN-0013, phase 1; artboards 1a to 1f): the scan (shown or hidden, see-through,
// ghost or solid), the reference points (Place points, the points that need a look, the body map and the hand map),
// the selected point (its weight, Use this point, its depth in the stone), contact pins, Grip with Squeeze, the
// balance mark, and Save points. The tools of the view (Select, Move, Turn) float at the view's top left. The panel
// reads the editor's store (pose_edit/editor.ts), which changes on clicks, keys and the ends of drags only.

import { useState, useSyncExternalStore } from 'react';
import { createPortal } from 'react-dom';
import { editStore, PIN_PARTS, poseEditor, type EditState, type PointInfo } from '../../pose_edit/editor.ts';
import { BodyMap, HandMap } from '../../pose_edit/maps.tsx';
import { runAction } from '../../studio/actions.ts';
import { Fold, Note, Seg, Switch, TextButton } from '../components/controls.tsx';
import { Icon } from '../components/ui/icon.tsx';
import { Tip } from '../components/ui/tooltip.tsx';
import { useShell } from '../context.ts';

export function useEditState(): EditState { return useSyncExternalStore(editStore.subscribe, editStore.get, editStore.get); }

const pct = (v: number) => Math.round(v * 100);

// a plain range with its number, for the editor's own values (they change with the store, not with the look)
function Range({ id, label, value, min, max, step, ends, unit, onInput, onStart, onEnd }: { id: string; label: string; value: number; min: number; max: number; step: number;
  ends?: [string, string]; unit?: string; onInput: (v: number) => void; onStart?: () => void; onEnd?: () => void }) {
  const [v, setV] = useState<number | null>(null);
  const shown = v ?? value;
  return (
    <div className="pe-range" data-id={id}>
      <div className="pe-range-head"><label htmlFor={'pe-' + id}>{label}</label><span className="pe-range-num">{Number.isInteger(step) ? shown : shown.toFixed(1)}{unit ? ' ' + unit : ''}</span></div>
      <input id={'pe-' + id} className="pe-range-input" type="range" min={min} max={max} step={step} value={shown}
        onPointerDown={() => onStart?.()} onInput={(e) => { const x = parseFloat((e.target as HTMLInputElement).value); setV(x); onInput(x); }}
        onChange={() => { setV(null); onEnd?.(); }} onPointerUp={() => { setV(null); onEnd?.(); }} />
      {ends ? <div className="pe-range-ends"><span>{ends[0]}</span><span>{ends[1]}</span></div> : null}
    </div>
  );
}

function Legend() {
  return (
    <ul className="pe-legend" aria-label="What the marks mean">
      <li><i className="pe-dot pe-dot--placed" />Placed</li><li><i className="pe-dot pe-dot--look" />Needs a look</li><li><i className="pe-dot pe-dot--off" />Switched off</li>
      <li><i className="pe-dot pe-dot--selected" />Selected</li><li><i className="pe-dot pe-dot--behind" />Behind the stone</li>
    </ul>
  );
}

function PointPanel({ p, st }: { p: PointInfo; st: EditState }) {
  const ed = poseEditor()!;
  const words = p.state === 'unplaced' ? 'Not placed yet. With the Move tool (W), click the scan to place it, or press Place points.'
    : st.dragging ? `Dragging. Anny's point is ${p.distance?.toFixed(0)} mm from the pointer, and the finger turns to meet it.`
      : p.depth > 0 ? 'Hidden by the carving. Pushed into the stone.'
        : p.state === 'look' ? `Needs a look: ${p.reason || 'the placement is unsure'}.` : p.state === 'off' ? 'Switched off: the solve leaves it out, and its finger keeps a natural curl.'
          : `Placed on the scan. Anny's point is ${p.distance === null ? '—' : p.distance.toFixed(1)} mm from it.`;
  return (
    <section className="pe-point" aria-labelledby="pe-point-h">
      <h3 className="osb-eyebrow pe-eyebrow">Point <span className="osb-eyebrow-val">{p.number}</span></h3>
      <p className="pe-point-name" id="pe-point-h">{p.label}</p>
      <Note>{words}</Note>
      {p.state !== 'unplaced' ? <>
        {!st.dragging ? <Range key={p.name + ':w'} id="weight" label="Weight" value={p.weight} min={0.1} max={1} step={0.1} ends={['Light', 'Strong']}
          onStart={() => {}} onInput={(v) => ed.setWeight(p.name, v, false)} onEnd={() => {}} /> : null}
        <Switch label="Use this point" on={p.state !== 'off'} onChange={(on) => ed.usePoint(p.name, on)} />
        {p.depth > 0 ? (
          <div className="pe-depth">
            <label className="pe-depth-row">Depth into the stone
              <input type="number" min={0} max={40} step={0.5} value={Number(p.depth.toFixed(1))} aria-label="Depth into the stone, millimetres"
                onChange={(e) => { const v = parseFloat(e.target.value); if (isFinite(v)) ed.push(p.name, v - p.depth); }} /> mm</label>
            <Tip action="pose.reset-turn" label="Back to the surface"><button type="button" className="tb osb-btn osb-btn--quiet" onClick={() => ed.toSurface(p.name)}>
              <Icon name="reset" size={16} /><span>Back to the surface</span></button></Tip>
          </div>
        ) : (
          <div className="pe-depth pe-depth--surface"><span>Depth <b>On the surface</b></span>
            <Tip action="pose.push" label="Push in"><button type="button" className="tb osb-btn osb-btn--quiet" onClick={() => ed.push(p.name, 1)}><Icon name="arrowDown" size={16} /><span>Push in</span></button></Tip>
          </div>
        )}
      </> : null}
      {st.dragging || st.tool === 'move' ? <>
        <h3 className="osb-eyebrow pe-eyebrow">While dragging</h3>
        <Switch label="Magnifier" on={st.magnifier} onChange={(on) => editStore.set({ magnifier: on })} />
        <Switch label="Show the points behind the stone" on={st.behind} onChange={(on) => editStore.set({ behind: on })} />
      </> : null}
    </section>
  );
}

function ScanBlock({ st }: { st: EditState }) {
  const ed = poseEditor()!, s = st.scan;
  if (!s.available && !s.loading && !s.error) return (
    <div className="osb-section pe-scan"><h3 className="osb-eyebrow">Scan</h3>
      <Note>This character has no scan. A statue starter carries a light copy of its scan; open one from New to place points on it.</Note></div>
  );
  return (
    <div className="osb-section pe-scan">
      <h3 className="osb-eyebrow">Scan</h3>
      {s.loading ? <Note>Reading the scan…</Note> : s.error ? <Note className="pe-error">The scan could not be read: {s.error}</Note> : <>
        <div className="pe-switch-row"><Switch label="Show the scan" on={s.shown} onChange={(on) => ed.setScanLook({ shown: on })} /><kbd className="osb-key">Shift Z</kbd></div>
        <Range key={'op' + s.shown} id="see-through" label="See-through" value={pct(s.opacity)} min={5} max={100} step={5} unit="%" ends={['Faint', 'Solid']}
          onInput={(v) => ed.setScanLook({ opacity: v / 100 })} />
        <Seg label="The scan's look" fill items={[{ value: 'ghost', label: 'Ghost' }, { value: 'solid', label: 'Solid' }]} value={s.look} onPick={(v) => ed.setScanLook({ look: v as any })} />
        {s.credit ? <Note className="credit">{s.credit}</Note> : null}
      </>}
    </div>
  );
}

function ContactFolds({ st }: { st: EditState }) {
  const ed = poseEditor()!;
  const [part, setPart] = useState('palm.L');
  return <>
    <Fold title="Contact pins" count={st.pins.length || 'None'} className="pe-fold">
      {st.pins.map((q, i) => (
        <div key={i} className="pe-pin"><Icon name="pin" size={16} /><span>{q.name}</span>
          <button type="button" className="osb-ib" aria-label={`Remove the pin ${q.name}`} onClick={() => ed.removePin(i)}><Icon name="close" size={14} /></button></div>
      ))}
      {st.pinDraft ? <>
        <Note>Click the surface on the scan, or pin it to the ground.</Note>
        <div className="btn-row"><TextButton label="On the ground" icon="pin" onClick={() => ed.addPin(st.pinDraft!, { kind: 'ground' })} /><TextButton label="Cancel" quiet onClick={() => ed.cancelPin()} /></div>
      </> : <>
        <Seg label="The part to pin" items={PIN_PARTS.map((p) => ({ value: p.id, label: p.label }))} value={part} onPick={setPart} />
        <div className="btn-row"><TextButton label="Add a pin" icon="plus" onClick={() => ed.beginPin(part)} /></div>
        <Note>Pick a palm or a sole, then the surface it rests on. A pinned part stays in touch while the rest of the body moves.</Note>
      </>}
    </Fold>
    <Fold title="Grip" count={st.grip ? `Squeeze ${pct(st.squeeze)} %` : 'Off'} className="pe-fold">
      {st.props.length ? <>
        <Seg label="The hand" items={[{ value: 'R', label: 'Right hand' }, { value: 'L', label: 'Left hand' }]} value={st.grip?.hand || ''} onPick={(v) => ed.setGrip({ hand: v as 'L' | 'R', prop: st.grip?.prop || st.props[0] })} />
        <Seg label="The prop" items={st.props.map((p) => ({ value: p, label: p.split('/').pop() || p }))} value={st.grip?.prop || ''}
          onPick={(v) => ed.setGrip({ hand: st.grip?.hand || 'R', prop: v })} />
        {st.grip ? <Range key="squeeze" id="squeeze" label="Squeeze" value={pct(st.squeeze)} min={0} max={100} step={1} unit="%" ends={['Looser', 'Tighter']}
          onInput={(v) => ed.setSqueeze(v / 100, false)} /> : null}
        <Note>One slider closes or opens every finger of the grip together, each until it touches the prop with 1 mm to spare.</Note>
      </> : <Note>Grip needs a prop that the character holds, shown in Dress.</Note>}
    </Fold>
    <Fold title="Balance" count={st.balance.shown ? 'On' : 'Off'} className="pe-fold">
      <Switch label="Show the balance mark" on={st.balance.shown} onChange={(on) => ed.setBalance(on)} />
      {st.balance.shown && st.balance.inside !== null ? (
        <p className={'pe-balance ' + (st.balance.inside ? 'is-good' : 'is-bad')} role="status"><Icon name={st.balance.inside ? 'check' : 'alert'} size={16} />
          {st.balance.inside ? `Balanced. The centre of mass is over the feet, ${Math.round(st.balance.margin || 0)} mm inside.` : `Off balance by ${Math.round(-(st.balance.margin || 0))} mm.`}</p>
      ) : null}
    </Fold>
  </>;
}

// the tools at the view's top left (artboard 1a): Select (Q), Move (W), Turn (E, with the joint dial)
function ViewTools({ st }: { st: EditState }) {
  const tools = [{ id: 'select', icon: 'cursor', action: 'pose.tool.select' }, { id: 'move', icon: 'move', action: 'pose.tool.move' }, { id: 'turn', icon: 'rotate', action: 'pose.tool.turn' }];
  return createPortal(
    <div className="osb-glass pe-tools" role="toolbar" aria-label="Posing tools">
      {tools.map((t) => (
        <Tip key={t.id} action={t.action} side="right">
          <button type="button" className="osb-ib pe-tool" aria-pressed={st.tool === t.id} aria-label={t.id} onClick={() => runAction(t.action)}>
            <Icon name={t.icon} size={20} /><span className="pe-tool-key" aria-hidden="true">{t.id === 'select' ? 'Q' : t.id === 'move' ? 'W' : 'E'}</span>
          </button>
        </Tip>
      ))}
    </div>, document.body);
}

export function PoseEditSection() {
  const st = useEditState(), ed = poseEditor();
  const { legacy } = useShell();
  if (!ed) return <Note>The pose editor starts with the character.</Note>;
  const hand = st.hand;
  const looks = st.looks;
  return (
    <div className="pose-edit">
      <ViewTools st={st} />
      <ScanBlock st={st} />
      <div className="osb-section pe-points">
        <h3 className="osb-eyebrow">Reference points <span className="osb-eyebrow-val">{st.counts.placed + st.counts.look + st.counts.off ? `${st.counts.placed + st.counts.look + st.counts.off} placed` : 'None placed'}</span></h3>
        {hand ? <>
          <div className="pe-hand-head">
            <button type="button" className="tb osb-btn osb-btn--quiet" onClick={() => ed.selectHand(null, false)}><Icon name="chevronLeft" size={16} /><span>Whole body</span></button>
            <Seg label="The hand" items={[{ value: 'L', label: 'Left' }, { value: 'R', label: 'Right' }]} value={hand} onPick={(v) => ed.selectHand(v as 'L' | 'R')} />
          </div>
          <HandMap side={hand} points={st.hands[hand]} selected={st.selected} onPick={(n) => ed.select(n)} onHover={(n) => ed.hover(n)} />
          <Legend />
        </> : <>
          <Note>Each point sits on the character's skin. Place it on the scan, and the pose follows it.</Note>
          <Seg label="Which points" fill items={[{ value: 'all', label: 'All' }, { value: 'hands', label: 'Hands' }, { value: 'body', label: 'Body' }]} value={st.filter}
            onPick={(v) => editStore.set({ filter: v as any, version: st.version + 1 })} />
          <ul className="pe-counts"><li><i className="pe-dot pe-dot--placed" />{st.counts.placed} placed</li><li><i className="pe-dot pe-dot--look" />{st.counts.look} need a look</li>
            <li><i className="pe-dot pe-dot--off" />{st.counts.off} switched off</li></ul>
          <BodyMap points={st.body} hands={st.hands} selected={st.selected} onPick={(n) => ed.select(n, { frame: true })} onHover={(n) => ed.hover(n)} onHand={(s) => ed.selectHand(s)} />
          {looks.length ? (
            <div className="pe-looks" role="region" aria-label="Points that need a look">
              <p className="pe-looks-head"><Icon name="alert" size={16} /><b>{looks.length} point{looks.length === 1 ? ' needs' : 's need'} a look</b></p>
              <Note>A click on a row frames the point in the view.</Note>
              <ul>{looks.map((p) => (
                <li key={p.name}><button type="button" className="pe-look-row" onClick={() => ed.select(p.name, { frame: true })}>
                  <i className="pe-diamond">{p.number}</i><span><b>{p.label}</b><small>{p.reason}</small></span><Icon name="chevron" size={14} /></button></li>
              ))}</ul>
            </div>
          ) : null}
          {!st.counts.placed && !st.counts.look ? (
            <div className="btn-row pe-place">
              <Tip action="pose.place"><button type="button" className="tb osb-btn osb-btn--primary" disabled={!st.scan.available} onClick={() => ed.placeAll()}><Icon name="sparkle" size={16} /><span>Place points</span></button></Tip>
              <Tip label="Place by hand" desc="Select a point on a map, then click the scan with the Move tool"><button type="button" className="tb osb-btn osb-btn--quiet" disabled={!st.scan.available}
                onClick={() => { runAction('pose.tool.move'); ed.selectHand('R'); }}><Icon name="cursor" size={16} /><span>Place by hand</span></button></Tip>
            </div>
          ) : null}
        </>}
      </div>
      {st.point ? <div className="osb-section"><PointPanel p={st.point} st={st} /></div> : null}
      {!hand ? <div className="osb-section pe-contact"><ContactFolds st={st} /></div> : null}
      <div className="pe-foot">
        {hand ? <TextButton label="Reset the hand" icon="reset" quiet onClick={() => { const s = st.selected || `wrist.${hand}`; ed.resetPoint(s, true); }} />
          : st.counts.placed || st.counts.look ? <TextButton label="Place again" icon="sparkle" quiet disabled={!st.scan.available} onClick={() => ed.placeAll()} /> : <TextButton label="Rest pose" icon="reset" quiet onClick={() => ed.restPose()} />}
        <TextButton label="Save points" icon="saveCard" primary disabled={legacy.hosted} onClick={() => ed.savePoints(legacy.saveCard)} />
      </div>
    </div>
  );
}

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
import { editStore, ENDS, PARTS, PIN_PARTS, poseEditor, type EditState, type PointInfo } from '../../pose_edit/editor.ts';
import { BodyMap, HandMap, JointMap } from '../../pose_edit/maps.tsx';
import { AXES } from '../../pose_edit/joints.ts';
import { FINGER_LABELS, PRESETS } from '../../pose_edit/hands.ts';
import { SECTIONS } from '../../pose_edit/editor.ts';
import { runAction } from '../../studio/actions.ts';
import { Fold, Note, Seg, Switch, TextButton } from '../components/controls.tsx';
import { Icon } from '../components/ui/icon.tsx';
import { Tip } from '../components/ui/tooltip.tsx';
import { useShell } from '../context.ts';

export function useEditState(): EditState { return useSyncExternalStore(editStore.subscribe, editStore.get, editStore.get); }

const pct = (v: number) => Math.round(v * 100);
const TAB_ICONS = ['focus', 'bones', 'hand', 'move', 'save'];
// the Limbs tool (artboard 2d): the hands' and feet's handles in the view, their pins, the pole of the limb chosen and
// Body follows
function LimbsPanel({ st }: { st: EditState }) {
  const ed = poseEditor()!;
  return (
    <div className="osb-section pe-limbs">
      <Note>Drag a gold handle on a hand or a foot: the limb follows, and the elbow or knee bends on its own. The small blue handle sets where the elbow or knee points. A right-click on a handle pins it.</Note>
      <h3 className="osb-eyebrow pe-eyebrow">Pins</h3>
      <div className="pe-handle-opts">
        {ENDS.map((e) => (
          <div key={e.id} className="pe-switch-row">
            <Switch label={e.label} on={st.pinned.includes(e.id)} onChange={(on) => ed.setPinned(e.id, on)} tip={`Pin the ${e.label.toLowerCase()}: it stays where it is while other parts move`} />
            {e.id.startsWith('hand') || e.id.startsWith('foot') ? <button type="button" className="tb osb-btn osb-btn--quiet" aria-pressed={st.limb === e.id}
              onClick={() => ed.selectLimb(st.limb === e.id ? null : e.id)}><span>Pole</span></button> : null}
          </div>
        ))}
      </div>
      <h3 className="osb-eyebrow pe-eyebrow">Body</h3>
      <Switch label="Body follows a hand" on={st.follow} onChange={(on) => ed.setFollow(on)} tip="The spine bends a little towards a far reach" />
      <Note>Off by default: the hips and the spine stay, and only the arm moves.</Note>
    </div>
  );
}

// the Library tool (artboard 2e): saved poses as tiles, applied to the whole body or one part, two poses blended, and
// mirror and copy
function LibraryPanel({ st }: { st: EditState }) {
  const ed = poseEditor()!;
  const [name, setName] = useState('');
  const [a, setA] = useState(''), [b, setB] = useState('');
  const poses = st.poses;
  return (
    <div className="osb-section pe-library-panel">
      <h3 className="osb-eyebrow">Saved poses <span className="osb-eyebrow-val">{poses.length}</span></h3>
      {poses.length ? (
        <div className="pe-library" role="group" aria-label="Saved poses">
          {poses.map((p) => (
            <div key={p.id} className="pe-pose-tile">
              <button type="button" className="tb osb-btn osb-btn--quiet" onClick={() => ed.applyPose(p.id)} title={`Apply ${p.label} to: ${(PARTS.find((x) => x.id === st.applyTo) || PARTS[0]).label}`}>
                <Icon name="pose" size={22} /><span>{p.label}</span></button>
              <button type="button" className="osb-ib" aria-label={`Delete ${p.label}`} onClick={() => ed.deletePose(p.id)}><Icon name="close" size={12} /></button>
            </div>
          ))}
        </div>
      ) : <Note>No saved poses yet. Name the pose shown and save it.</Note>}
      <div className="pe-save-hand">
        <input type="text" className="osb-input" placeholder="A name for this pose" aria-label="A name for this pose" value={name} onChange={(e) => setName(e.target.value)} />
        <TextButton label="Save the pose" icon="plus" disabled={!name.trim()} onClick={() => { ed.savePose(name.trim()); setName(''); }} />
      </div>
      <h3 className="osb-eyebrow pe-eyebrow">Apply to</h3>
      <Seg label="Apply to" items={PARTS.map((p) => ({ value: p.id, label: p.label }))} value={st.applyTo} onPick={(v) => ed.setApplyTo(v)} />
      {poses.length >= 2 ? <>
        <h3 className="osb-eyebrow pe-eyebrow">Blend two poses</h3>
        <div className="pe-blend">
          <select aria-label="First pose" value={a} onChange={(e) => setA(e.target.value)}><option value="">First pose</option>{poses.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}</select>
          <select aria-label="Second pose" value={b} onChange={(e) => setB(e.target.value)}><option value="">Second pose</option>{poses.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}</select>
        </div>
        {a && b ? <Range key={a + b} id="blend" label="Blend" unit="%" value={Math.round((st.blend?.t ?? 0) * 100)} min={0} max={100} step={1}
          ends={[poses.find((p) => p.id === a)!.label, poses.find((p) => p.id === b)!.label]} onInput={(v) => ed.blendPoses(a, b, v / 100, false)}
          onStart={() => (ed as any).record('Blend poses')} /> : null}
      </> : null}
      <h3 className="osb-eyebrow pe-eyebrow">Mirror and copy</h3>
      <div className="btn-row">
        <Tip action="pose.mirror"><button type="button" className="tb osb-btn" onClick={() => ed.mirror()}><Icon name="turntable" size={16} /><span>Mirror the pose</span></button></Tip>
        <Tip action="pose.copy"><button type="button" className="tb osb-btn osb-btn--quiet" onClick={() => runAction('pose.copy')}><Icon name="copy" size={16} /><span>Copy</span></button></Tip>
        <Tip action="pose.paste-mirrored"><button type="button" className="tb osb-btn osb-btn--quiet" onClick={() => runAction('pose.paste-mirrored')}><Icon name="paste" size={16} /><span>Paste mirrored</span></button></Tip>
      </div>
    </div>
  );
}

// a plain range with its number, for the editor's own values (they change with the store, not with the look)
function Range({ id, label, value, min, max, step, ends, unit, onInput, onStart, onEnd, colour, limits, onReset }: { id: string; label: string; value: number; min: number; max: number;
  step: number; ends?: [string, string]; unit?: string; onInput: (v: number) => void; onStart?: () => void; onEnd?: () => void; colour?: string; limits?: [number, number];
  onReset?: () => void }) {
  const [v, setV] = useState<number | null>(null);
  const shown = v ?? value;
  return (
    <div className="pe-range" data-id={id}>
      <div className="pe-range-head"><label htmlFor={'pe-' + id}>{colour ? <i className="pe-slider-axis" style={{ background: colour }} /> : null}{label}</label>
        <span className="pe-range-num">{onReset ? <button type="button" className="osb-ib pe-reset" aria-label={`Reset ${label}`} onClick={onReset}><Icon name="undoSmall" size={14} /></button> : null}
          <input type="number" className="pe-num" aria-label={label} step={step} value={Number(shown.toFixed(step < 1 ? 1 : 0))}
            onChange={(e) => { const x = parseFloat(e.target.value); if (isFinite(x)) { onStart?.(); onInput(Math.min(max, Math.max(min, x))); onEnd?.(); } }} />{unit ? ' ' + unit : ''}</span></div>
      <div className="pe-range-track">
        <input id={'pe-' + id} className={'pe-range-input' + (limits && (shown < limits[0] || shown > limits[1]) ? ' is-out' : '')} type="range" min={min} max={max} step={step} value={shown}
          onPointerDown={() => onStart?.()} onInput={(e) => { const x = parseFloat((e.target as HTMLInputElement).value); setV(x); onInput(x); }}
          onChange={() => { setV(null); onEnd?.(); }} onPointerUp={() => { setV(null); onEnd?.(); }} />
        {limits ? limits.map((l, i) => <span key={i} className="pe-range-limit" style={{ left: `calc(${(l - min) / (max - min) * 100}% - 1px)` }} aria-hidden="true" />) : null}
      </div>
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

// the side view of a pushed point (artboard 1e): the camera's ray, the surface where it meets the stone and the point
// further in, as a diagram with the depth written on it
function DepthInset({ p }: { p: PointInfo }) {
  const d = p.depth, x = 150 + Math.min(150, d * 12);
  return createPortal(
    <aside className="osb-glass pe-inset" aria-label={`Side view of point ${p.number}`}>
      <header className="pe-inset-head"><h2 className="osb-eyebrow">Side view</h2><span className="osb-tag">Point {p.number}</span></header>
      <svg viewBox="0 0 320 180" role="img" aria-label={`The ray from the camera meets the stone at the surface; the point sits ${d.toFixed(1)} mm further in.`}>
        <rect x="0" y="0" width="320" height="180" fill="#15171a" />
        <path d="M150 0 C 132 60, 132 120, 150 180 L 320 180 L 320 0 Z" fill="#d9d3c9" />
        <path d="M150 0 C 132 60, 132 120, 150 180" fill="none" stroke="#d4ae52" strokeWidth="3" />
        <text x="296" y="22" textAnchor="end" fontSize="13" fill="#24201c">Stone</text>
        <circle cx="22" cy="100" r="8" fill="none" stroke="#ffffff" strokeWidth="2" /><circle cx="22" cy="100" r="3" fill="#ffffff" />
        <text x="10" y="130" fontSize="12" fill="#ffffff">Camera</text>
        <line x1="32" y1="100" x2={x - 10} y2="100" stroke="#ffffff" strokeWidth="1.5" strokeDasharray="5 4" />
        <circle cx="137" cy="100" r="4" fill="#ffffff" stroke="#1d1b19" />
        <text x="96" y="86" fontSize="12" fill="#d4ae52">Surface</text>
        <line x1="137" y1="70" x2={x} y2="70" stroke="#1d1b19" strokeWidth="1.5" /><line x1="137" y1="64" x2="137" y2="76" stroke="#1d1b19" strokeWidth="1.5" />
        <line x1={x} y1="64" x2={x} y2="76" stroke="#1d1b19" strokeWidth="1.5" />
        <text x={(137 + x) / 2} y="60" textAnchor="middle" fontSize="13" fontWeight="600" fill="#1d1b19">{d.toFixed(1)} mm</text>
        <circle cx={x} cy="100" r="10" fill="none" stroke="#2f6fd6" strokeWidth="3" />
      </svg>
      <p className="pe-inset-note">The ray from the camera meets the stone at the surface. The point sits {d.toFixed(1)} mm further in, inside the carving.</p>
    </aside>, document.body);
}

// the Joints tool (artboard 2a): the body map of joints, the selected joint's Bend, Twist and Side-Side, the range
// switch and the resets
function JointsPanel({ st }: { st: EditState }) {
  const ed = poseEditor()!, jv = st.jv;
  const started = useState({ on: false })[0];
  const start = (label: string) => { if (!started.on) { (ed as any).record(label); started.on = true; } };
  const end = () => { started.on = false; };
  const ENDS: [string, string][] = [['Down', 'Up'], ['Back', 'Forward'], ['In', 'Out']];
  return (
    <div className="osb-section pe-joints">
      <h3 className="osb-eyebrow">Body map <span className="osb-eyebrow-val">{st.joint ? '' : 'Pick a joint'}</span></h3>
      <JointMap selected={st.joint} onPick={(l) => ed.selectJoint(l)} />
      {jv ? <>
        <h3 className="osb-eyebrow pe-eyebrow">Joint <span className="osb-eyebrow-val">{jv.joint}</span></h3>
        <p className="pe-point-name">{jv.name}</p>
        {[0, 2, 1].map((k) => jv.dofs[k] ? (
          <Range key={jv.label + k} id={'axis' + k} label={AXES[k].label + ' (°)'} colour={AXES[k].colour} value={jv.value[k]}
            min={Math.min(-180, Math.floor(jv.lo[k] - 20))} max={Math.max(180, Math.ceil(jv.hi[k] + 20))} step={1} ends={ENDS[k]} limits={[jv.lo[k], jv.hi[k]]}
            onStart={() => start(AXES[k].label)} onEnd={end} onInput={(v) => ed.setJointValue(k, v, false)}
            onReset={() => ed.setJointValue(k, 0)} />
        ) : null)}
        <Switch label="Keep within the joint's range" on={st.limits} onChange={(on) => ed.setLimits(on)} />
        <div className="btn-row">
          <Tip action="pose.reset-joint" label="Reset the joint"><button type="button" className="tb osb-btn" onClick={() => ed.resetJoint(false)}><Icon name="reset" size={16} /><span>Reset the joint</span></button></Tip>
          <Tip label="Reset the whole limb" keys="Shift+Alt+R"><button type="button" className="tb osb-btn osb-btn--quiet" onClick={() => ed.resetJoint(true)}><Icon name="reset" size={16} /><span>Reset the limb</span></button></Tip>
        </div>
      </> : <Note>Click a joint on the map or on the figure. Its rings show at the joint: drag a ring to turn it, or press R and move the mouse.</Note>}
    </div>
  );
}

// a small picture of a hand shape for the presets: the fingers as strokes bent by their curl
function HandIcon({ curl, spread }: { curl: number[]; spread: number }) {
  const base: [number, number][] = [[10, 26], [14, 16], [18, 14], [22, 15], [26, 17]];
  return (
    <svg viewBox="0 0 36 36" aria-hidden="true">
      <path d="M9 34 L9 24 Q9 18 15 18 L25 18 Q29 18 29 23 L29 34 Z" fill="var(--raised-2)" stroke="currentColor" strokeWidth="1" />
      {base.map(([x, y], k) => {
        const len = k === 0 ? 9 : 12 - Math.abs(k - 2.4) * 1.2, a = (k === 0 ? -130 : -90 + (k - 2.5) * 9 * spread) * Math.PI / 180, c = Math.max(0, Math.min(1, curl[k]));
        const ex = x + Math.cos(a) * len * (1 - 0.75 * c), ey = y + Math.sin(a) * len * (1 - 0.75 * c);
        return <line key={k} x1={x} y1={y} x2={ex} y2={ey} stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" />;
      })}
    </svg>
  );
}

// the gesture pad (artboard 2b): four presets at its corners, a point dragged between them
function GesturePad({ st }: { st: EditState }) {
  const ed = poseEditor()!;
  const [corners, setCorners] = useState(['open', 'pointing', 'fist', 'pinch']);
  const [at, setAt] = useState<[number, number] | null>(null);
  const all = [...PRESETS, ...st.userPresets];
  const shapeOf = (id: string) => (all.find((p) => p.id === id) || PRESETS[0]).shape;
  const move = (e: React.PointerEvent<HTMLDivElement>, first: boolean) => {
    const r = e.currentTarget.getBoundingClientRect(), u = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), v = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
    setAt([u, v]);
    ed.blendPad(corners.map(shapeOf), u, v, first);
  };
  return <>
    <h3 className="osb-eyebrow pe-eyebrow">Gesture pad</h3>
    <div className="pe-pad" role="slider" aria-label="Blend of four hand presets" aria-valuetext={at ? `${Math.round(at[0] * 100)} %, ${Math.round(at[1] * 100)} %` : 'Not used'} tabIndex={0}
      onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); move(e, true); }} onPointerMove={(e) => { if (e.buttons) move(e, false); }}>
      {corners.map((c, i) => (
        <select key={i} className="pe-pad-corner" aria-label={`Corner ${i + 1}`} value={c} onPointerDown={(e) => e.stopPropagation()}
          style={{ [i % 2 ? 'right' : 'left']: 4, [i < 2 ? 'top' : 'bottom']: 4 } as any} onChange={(e) => { const n = corners.slice(); n[i] = e.target.value; setCorners(n); }}>
          {all.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
        </select>
      ))}
      {at ? <span className="pe-pad-dot" style={{ left: `${at[0] * 100}%`, top: `${at[1] * 100}%` }} /> : null}
    </div>
    <Note>Drag between the four hands at the corners. Each corner takes any preset.</Note>
  </>;
}

// the Hands tool (artboard 2b): Fist, Spread, Thumb, Relax; each finger's Curl, with its spread behind a fold; the
// presets, the gesture pad, and Copy to the other hand
function HandsPanel({ st }: { st: EditState }) {
  const ed = poseEditor()!, h = st.handShape, side = st.handSide;
  const started = useState({ on: false })[0];
  const start = (label: string) => { if (!started.on) { (ed as any).record(label); started.on = true; } };
  const end = () => { started.on = false; };
  const pct = (x: number) => Math.round(x * 100);
  const [name, setName] = useState('');
  return (
    <div className="osb-section pe-hands">
      <div className="pe-hand-head"><p className="pe-point-name">{side === 'R' ? 'Right hand' : 'Left hand'}</p>
        <Seg label="The hand" items={[{ value: 'L', label: 'Left' }, { value: 'R', label: 'Right' }]} value={side} onPick={(v) => ed.setHandSide(v as 'L' | 'R')} /></div>
      <h3 className="osb-eyebrow pe-eyebrow">Whole hand</h3>
      <Range key={side + 'fist' + st.version} id="fist" label="Fist" unit="%" value={pct(h.fist)} min={-10} max={110} step={1} ends={['Open', 'Closed']}
        onStart={() => start('Fist')} onEnd={end} onInput={(v) => ed.setHand('fist', 0, v / 100, false)} />
      <Range key={side + 'spread' + st.version} id="spread" label="Spread" unit="%" value={pct(h.spread)} min={-100} max={100} step={1} ends={['Together', 'Apart']}
        onStart={() => start('Spread')} onEnd={end} onInput={(v) => ed.setHand('spread', 0, v / 100, false)} />
      <Range key={side + 'thumb' + st.version} id="thumb" label="Thumb" unit="%" value={pct(h.thumb)} min={-20} max={110} step={1} ends={['Beside', 'Across the palm']}
        onStart={() => start('Thumb')} onEnd={end} onInput={(v) => ed.setHand('thumb', 0, v / 100, false)} />
      <div className="pe-fingers">
        {FINGER_LABELS.map((label, k) => (
          <Fold key={k} title={label} count={`Curl ${pct(h.curl[k])} %`} className="pe-fold">
            <Range key={side + 'curl' + k + st.version} id={'curl' + k} label="Curl" unit="%" value={pct(h.curl[k])} min={-20} max={120} step={1} ends={['Straight', 'Bent']}
              onStart={() => start('Curl')} onEnd={end} onInput={(v) => ed.setHand('curl', k, v / 100, false)} />
            {k > 0 ? <Range key={side + 'fs' + k + st.version} id={'fspread' + k} label="Spread" unit="%" value={pct((ed as any).fingerSpread?.(k) ?? 0)} min={-100} max={100} step={1}
              ends={['Together', 'Apart']} onStart={() => start('Spread')} onEnd={end} onInput={(v) => ed.setHand('fspread', k, v / 100, false)} /> : null}
          </Fold>
        ))}
      </div>
      <h3 className="osb-eyebrow pe-eyebrow">Presets</h3>
      <div className="pe-presets" role="group" aria-label="Hand presets">
        {[...PRESETS, ...st.userPresets].map((p) => (
          <button key={p.id} type="button" className="pe-preset" aria-pressed={st.preset === p.id} onClick={() => ed.applyPreset(p.id)}>
            <HandIcon curl={p.shape.curl} spread={p.shape.spread} /><span>{p.label}</span></button>
        ))}
      </div>
      <div className="pe-save-hand">
        <input type="text" className="osb-input" placeholder="A name for this hand" aria-label="A name for this hand" value={name} onChange={(e) => setName(e.target.value)} />
        <TextButton label="Save this hand" icon="plus" disabled={!name.trim()} onClick={() => { ed.saveHandPreset(name.trim()); setName(''); }} />
      </div>
      <GesturePad st={st} />
      <div className="btn-row">
        <TextButton label="Relax" icon="reset" quiet onClick={() => ed.relax()} />
        <Tip label={`Copy to the ${side === 'R' ? 'left' : 'right'} hand`} desc="The hand's pose mirrored onto the other hand"><button type="button" className="tb osb-btn" onClick={() => ed.copyHand()}>
          <Icon name="copy" size={16} /><span>Copy to the {side === 'R' ? 'left' : 'right'} hand</span></button></Tip>
      </div>
    </div>
  );
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
      <div className="pe-tabs" role="tablist" aria-label="Posing tools">
        {SECTIONS.map((t, i) => (
          <Tip key={t.id} action={`pose.tool.${i + 1}`}><button type="button" role="tab" className="pe-tab" aria-selected={st.section === t.id} onClick={() => runAction(`pose.tool.${i + 1}`)}>
            <Icon name={TAB_ICONS[i]} size={18} /><span>{t.label}</span><span className="pe-tab-key" aria-hidden="true">{i + 1}</span></button></Tip>
        ))}
      </div>
      {st.section === 'joints' ? <JointsPanel st={st} /> : st.section === 'hands' ? <HandsPanel st={st} /> : st.section === 'limbs' ? <LimbsPanel st={st} />
        : st.section === 'library' ? <LibraryPanel st={st} /> : <>
      {st.point && st.point.depth > 0 ? <DepthInset p={st.point} /> : null}
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
      </>}
      <div className="pe-foot">
        {hand ? <TextButton label="Reset the hand" icon="reset" quiet onClick={() => { const s = st.selected || `wrist.${hand}`; ed.resetPoint(s, true); }} />
          : st.counts.placed || st.counts.look ? <TextButton label="Place again" icon="sparkle" quiet disabled={!st.scan.available} onClick={() => ed.placeAll()} /> : <TextButton label="Rest pose" icon="reset" quiet onClick={() => ed.restPose()} />}
        <TextButton label="Save points" icon="saveCard" primary disabled={legacy.hosted} onClick={() => ed.savePoints(legacy.saveCard)} />
      </div>
    </div>
  );
}

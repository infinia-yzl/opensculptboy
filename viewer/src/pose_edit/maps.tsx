// OpenSculptBoy
// Apache License, Version 2.0
//
// The hand map and the body map of the pose editor (DESIGN-0013, "Picking a bone"): flat diagrams in the panel with a
// point per reference point, each at least 24 pixels across, so every finger joint is easy to hit whatever the camera
// shows. The hand map shows one hand seen from the back (the right hand with its thumb on the left, the left hand
// mirrored); the body map shows the figure from the front, its right side on the viewer's left, the left side teal
// and the right side orange as in the Rig view. Hovering a point lights its place in the view; a click selects it.

import type { PointInfo } from './editor.ts';

const SIDE = { L: '#2a9d8f', R: '#e07a2f' };
const COL = { placed: '#2f6fd6', look: '#f0a030', off: '#9e968a', unplaced: '#9e968a', selected: '#d4ae52', outline: '#1d1b19' };

// the right hand's points, seen from the back, in MediaPipe's order (a box of 260 by 270)
const HAND: [number, number][] = [
  [144, 250],
  [104, 222], [76, 192], [54, 162], [34, 132],
  [104, 136], [97, 100], [92, 65], [86, 30],
  [139, 128], [139, 90], [139, 55], [139, 20],
  [172, 135], [178, 99], [184, 64], [189, 30],
  [203, 150], [217, 118], [228, 85], [236, 52],
];
const FINGERS = [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];

function Mark({ x, y, p, selected, onPick, onHover, r = 12 }: { x: number; y: number; p: PointInfo; selected: boolean; onPick: (n: string) => void;
  onHover: (n: string | null) => void; r?: number }) {
  const look = p.state === 'look', hollow = p.state === 'off' || p.state === 'unplaced';
  const d = r * 1.18;
  return (
    <g className="map-point" data-point={p.number} data-name={p.name} role="button" tabIndex={-1} aria-label={`${p.number} ${p.label}`}
      onClick={() => onPick(p.name)} onMouseEnter={() => onHover(p.name)} onMouseLeave={() => onHover(null)} style={{ cursor: 'pointer' }}>
      <circle cx={x} cy={y} r={r + 6} fill="transparent" />
      {selected ? <><circle cx={x} cy={y} r={r + 4.5} fill="none" stroke={COL.outline} strokeWidth={5} /><circle cx={x} cy={y} r={r + 4.5} fill="none" stroke={COL.selected} strokeWidth={2.6} /></> : null}
      {look ? (
        <polygon points={`${x},${y - d} ${x + d},${y} ${x},${y + d} ${x - d},${y}`} fill={COL.look} stroke={COL.outline} strokeWidth={1.5} strokeLinejoin="round" />
      ) : hollow ? (
        <circle cx={x} cy={y} r={r} fill="rgba(255,255,255,0.12)" stroke={COL.off} strokeWidth={1.6} strokeDasharray="3 2.5" />
      ) : (
        <circle cx={x} cy={y} r={r} fill={COL.placed} stroke={COL.outline} strokeWidth={1.5} />
      )}
      {p.depth > 0 ? <circle cx={x} cy={y} r={r - 3.5} fill="none" stroke="#ffffff" strokeWidth={1.4} /> : null}
      <text x={x} y={y + 0.5} textAnchor="middle" dominantBaseline="central" fontSize={11} fontWeight={600} fill={look || hollow ? 'currentColor' : '#ffffff'}>{p.number}</text>
    </g>
  );
}

export function HandMap({ side, points, selected, onPick, onHover }: { side: 'L' | 'R'; points: PointInfo[]; selected: string | null; onPick: (n: string) => void;
  onHover: (n: string | null) => void }) {
  const at = (i: number): [number, number] => { const [x, y] = HAND[i]; return side === 'R' ? [x + 10, y + 12] : [270 - x, y + 12]; };
  return (
    <svg className="pose-map hand-map" viewBox="0 0 280 290" role="group" aria-label={`${side === 'R' ? 'Right' : 'Left'} hand, seen from the back`}>
      <g aria-hidden="true" fill="none" strokeLinecap="round" strokeLinejoin="round" stroke="var(--raised-2)">
        {/* the palm and the fingers as thick strokes */}
        <path d={`M${at(0).join(',')} L${at(1).join(',')} L${at(5).join(',')} L${at(9).join(',')} L${at(13).join(',')} L${at(17).join(',')} Z`} fill="var(--raised-2)" strokeWidth={44} />
        {FINGERS.map((f, k) => <polyline key={k} points={f.map((i) => at(i).join(',')).join(' ')} strokeWidth={k === 0 ? 30 : 26} />)}
      </g>
      <g aria-hidden="true" stroke={SIDE[side]} strokeOpacity={0.5} strokeWidth={1.5} fill="none">
        {FINGERS.map((f, k) => <polyline key={k} points={f.map((i) => at(i).join(',')).join(' ')} />)}
      </g>
      {points.map((p, i) => { const [x, y] = at(i); return <Mark key={p.name} x={x} y={y} p={p} selected={p.name === selected} onPick={onPick} onHover={onHover} />; })}
    </svg>
  );
}

// the body's points from the front (a box of 220 by 320), the figure's right side on the left
const BODY: Record<string, [number, number]> = {
  head_top: [110, 14], 'shoulder.R': [76, 70], 'shoulder.L': [144, 70], 'elbow.R': [60, 124], 'elbow.L': [160, 124], 'hip.R': [92, 162], 'hip.L': [128, 162],
  'knee.R': [90, 226], 'knee.L': [130, 226], 'ankle.R': [88, 280], 'ankle.L': [132, 280], 'heel.R': [76, 300], 'heel.L': [144, 300], 'toe.R': [96, 306], 'toe.L': [124, 306],
};
const HANDS: Record<'L' | 'R', [number, number]> = { R: [44, 176], L: [176, 176] };

export function BodyMap({ points, hands, selected, onPick, onHover, onHand }: { points: PointInfo[]; hands: Record<'L' | 'R', PointInfo[]>; selected: string | null;
  onPick: (n: string) => void; onHover: (n: string | null) => void; onHand: (side: 'L' | 'R') => void }) {
  const limb = (a: string, b: string) => [BODY[a], BODY[b]];
  const bone = (side: 'L' | 'R', list: [number, number][]) => <polyline points={list.map((p) => p.join(',')).join(' ')} stroke={SIDE[side]} />;
  return (
    <svg className="pose-map body-map" viewBox="0 0 220 324" role="group" aria-label="The body's points, the figure seen from the front">
      <g aria-hidden="true" fill="none" strokeLinecap="round" strokeWidth={2} strokeOpacity={0.55}>
        <ellipse cx={110} cy={36} rx={17} ry={22} stroke="var(--line-strong)" />
        <polyline points="110,58 110,150" stroke="var(--line-strong)" />
        <polyline points={`${BODY['shoulder.R'].join(',')} ${BODY['shoulder.L'].join(',')}`} stroke="var(--line-strong)" />
        <polyline points={`${BODY['hip.R'].join(',')} ${BODY['hip.L'].join(',')}`} stroke="var(--line-strong)" />
        {bone('R', [...limb('shoulder.R', 'elbow.R'), HANDS.R])}{bone('L', [...limb('shoulder.L', 'elbow.L'), HANDS.L])}
        {bone('R', [BODY['hip.R'], BODY['knee.R'], BODY['ankle.R'], BODY['toe.R']])}{bone('L', [BODY['hip.L'], BODY['knee.L'], BODY['ankle.L'], BODY['toe.L']])}
      </g>
      <text x={20} y={318} fontSize={11} fill={SIDE.R} fontWeight={600}>R</text>
      <text x={194} y={318} fontSize={11} fill={SIDE.L} fontWeight={600}>L</text>
      {(['R', 'L'] as const).map((s) => {
        const [x, y] = HANDS[s], list = hands[s], look = list.filter((p) => p.state === 'look').length, placed = list.filter((p) => p.state !== 'unplaced').length;
        return (
          <g key={s} className="map-hand" role="button" tabIndex={-1} aria-label={`${s === 'R' ? 'Right' : 'Left'} hand: ${placed} of 21 placed${look ? `, ${look} need a look` : ''}`}
            onClick={() => onHand(s)} style={{ cursor: 'pointer' }}>
            <rect x={x - 17} y={y - 17} width={34} height={34} rx={4} fill="var(--raised-2)" stroke={SIDE[s]} strokeWidth={1.5} />
            <text x={x} y={y - 2} textAnchor="middle" dominantBaseline="central" fontSize={11} fontWeight={600} fill="currentColor">21</text>
            {look ? <polygon points={`${x + 12},${y + 3} ${x + 18},${y + 9} ${x + 12},${y + 15} ${x + 6},${y + 9}`} fill={COL.look} stroke={COL.outline} strokeWidth={1} /> : null}
          </g>
        );
      })}
      {points.map((p) => { const at = BODY[p.name]; return at ? <Mark key={p.name} x={at[0]} y={at[1]} p={p} r={11} selected={p.name === selected} onPick={onPick} onHover={onHover} /> : null; })}
    </svg>
  );
}

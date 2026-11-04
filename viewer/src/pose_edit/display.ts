// OpenSculptBoy
// Apache License, Version 2.0
//
// What Pose › Edit shows: the figure (Anny's body with its hair, eyes, lashes, teeth, wearables and props), the scan,
// or both. The two can never both be hidden: hiding one while the other is hidden shows the other, and a character
// without a scan always shows its figure, since the view would otherwise be empty. Shift+Z swaps them: the figure
// alone becomes the scan alone and back again, and both shown becomes the scan alone. This is a view setting: it stays
// out of the card, the history and the exports.

export interface Shown { figure: boolean; scan: boolean }

// the figure on or off
export function showFigure(s: Shown, on: boolean, hasScan: boolean): Shown {
  if (on || !hasScan) return { figure: true, scan: s.scan };
  return { figure: false, scan: true };
}

// the scan on or off
export function showScan(s: Shown, on: boolean, hasScan: boolean): Shown {
  if (!hasScan) return { figure: true, scan: on };
  if (on) return { figure: s.figure, scan: true };
  return { figure: true, scan: false };
}

// Shift+Z: the figure alone and the scan alone in turn; both shown become the scan alone
export function swapShown(s: Shown, hasScan: boolean): Shown {
  if (!hasScan) return { figure: true, scan: s.scan };
  if (s.figure && s.scan) return { figure: false, scan: true };
  return { figure: !s.figure, scan: !s.scan };
}

// what is shown as the view draws it: without a scan the figure always shows
export function effective(s: Shown, hasScan: boolean): Shown {
  if (!hasScan) return { figure: true, scan: false };
  if (!s.figure && !s.scan) return { figure: true, scan: false };
  return s;
}

// the note at the start of the status line, or '' when both show
export function shownNote(s: Shown, hasScan: boolean): string {
  const e = effective(s, hasScan);
  if (!e.figure) return 'Figure hidden';
  if (hasScan && !e.scan) return 'Scan hidden';
  return '';
}

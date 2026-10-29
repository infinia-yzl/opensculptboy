// Anny
// Copyright (C) 2025 NAVER Corp.
// Apache License, Version 2.0
//
// The icons of the page: Phosphor Icons in the Light weight (icon_paths.ts), drawn as inline SVG so the page needs no
// icon font and no request. The React interface (src/shell) draws the same table.

import { ICON_PATHS } from './icon_paths.ts';

export type IconName = keyof typeof ICON_PATHS | string;

// an <svg> element of the icon (decorative: the button that holds it carries the name)
export function icon(name: IconName, size = 20): SVGSVGElement {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 256 256');
  svg.setAttribute('width', String(size)); svg.setAttribute('height', String(size));
  svg.setAttribute('aria-hidden', 'true'); svg.setAttribute('focusable', 'false');
  svg.setAttribute('class', 'ico osb-ico');
  svg.innerHTML = ICON_PATHS[name] || '';
  return svg;
}

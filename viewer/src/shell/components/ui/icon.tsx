// OpenSculptBoy
// Apache License, Version 2.0
//
// An icon of the page's table (ui/icon_paths.ts: Phosphor Icons, Light weight), for the React interface. Decorative:
// the control that holds it carries the name.

import { ICON_PATHS } from '../../../ui/icon_paths.ts';
import { cn } from '../../lib/utils.ts';

export function Icon({ name, size = 20, className }: { name: string; size?: number; className?: string }) {
  return (
    <svg className={cn('osb-ico', className)} viewBox="0 0 256 256" width={size} height={size} aria-hidden="true" focusable="false"
      dangerouslySetInnerHTML={{ __html: ICON_PATHS[name] || '' }} />
  );
}

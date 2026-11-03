// OpenSculptBoy
// Apache License, Version 2.0
//
// shadcn/ui's Tooltip (Radix Tooltip) in one pattern everywhere (artboard 3a): the name, its key caps, and one line of
// what the control does, after the design system's wait. A control of a registered action names it (`action`), so the
// tooltip shows the registry's label, line and keys. Touch screens show no tooltip.

import * as React from 'react';
import * as TooltipPrimitive from '@radix-ui/react-tooltip';
import { keyText } from '../../../ui/shortcuts.ts';
import { action as actionDef, capsFor, capsOf } from '../../../studio/actions.ts';
import { useActionsVersion } from '../../../studio/react.ts';
import { cn } from '../../lib/utils.ts';

export const TooltipProvider = ({ children }: { children: React.ReactNode }) => (
  <TooltipPrimitive.Provider delayDuration={350} skipDelayDuration={200}>{children}</TooltipPrimitive.Provider>
);
export const TooltipRoot = TooltipPrimitive.Root;
export const TooltipTrigger = TooltipPrimitive.Trigger;

export const TooltipContent = React.forwardRef<React.ElementRef<typeof TooltipPrimitive.Content>, React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>>(
  ({ className, sideOffset = 8, ...props }, ref) => (
    <TooltipPrimitive.Portal>
      <TooltipPrimitive.Content ref={ref} sideOffset={sideOffset} className={cn('osb-tip osb-tooltip', className)} {...props} />
    </TooltipPrimitive.Portal>
  ));
TooltipContent.displayName = 'TooltipContent';

// a control with its tooltip: the label, its keys as key caps, and a line of what it does; `keys` is a chord
// ('Shift+4', 'Ctrl+Shift+E') or caps with spaces ('Ctrl Z'), else the keys of the `action` of the registry
export function Tip({ label, keys, action, desc, side = 'bottom', children }: {
  label?: string; keys?: string; action?: string; desc?: string; side?: 'top' | 'bottom' | 'left' | 'right'; children: React.ReactElement;
}) {
  useActionsVersion();
  const a = action ? actionDef(action) : undefined;
  const title = label ?? a?.label ?? '';
  const caps = keys ? (/\+./.test(keys) ? capsOf(keys) : keys.split(' ').map(keyText)) : action ? capsFor(action) : [];
  const line = desc ?? a?.desc;
  return (
    <TooltipRoot>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent side={side} className={line ? 'osb-tip--long' : undefined}>
        <span className="osb-tip-head"><span className="osb-tip-title">{title}</span>{caps.map((k, i) => <span key={i} className="osb-key">{k}</span>)}</span>
        {line ? <span className="osb-tip-desc">{line}</span> : null}
      </TooltipContent>
    </TooltipRoot>
  );
}

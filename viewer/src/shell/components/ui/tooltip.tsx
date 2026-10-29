// OpenSculptBoy
// Apache License, Version 2.0
//
// shadcn/ui's Tooltip (Radix Tooltip): the action's name, then its key (KeyHint), after the design system's wait.
// Touch screens show no tooltip.

import * as React from 'react';
import * as TooltipPrimitive from '@radix-ui/react-tooltip';
import { keyText } from '../../../ui/shortcuts.ts';
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

// a control with its tooltip: the label and, when it has one, the key as a key cap
export function Tip({ label, keys, side = 'bottom', children }: { label: string; keys?: string; side?: 'top' | 'bottom' | 'left' | 'right'; children: React.ReactElement }) {
  return (
    <TooltipRoot>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent side={side}>
        {label}
        {keys ? keys.split(' ').map((k, i) => <span key={i} className="osb-key">{keyText(k)}</span>) : null}
      </TooltipContent>
    </TooltipRoot>
  );
}

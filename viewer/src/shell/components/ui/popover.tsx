// OpenSculptBoy
// Apache License, Version 2.0
//
// shadcn/ui's Popover (Radix Popover), in the design system's popover (osb-pop).

import * as React from 'react';
import * as PopoverPrimitive from '@radix-ui/react-popover';
import { cn } from '../../lib/utils.ts';

export const Popover = PopoverPrimitive.Root;
export const PopoverTrigger = PopoverPrimitive.Trigger;

export const PopoverContent = React.forwardRef<React.ElementRef<typeof PopoverPrimitive.Content>, React.ComponentPropsWithoutRef<typeof PopoverPrimitive.Content>>(
  ({ className, sideOffset = 8, ...props }, ref) => (
    <PopoverPrimitive.Portal>
      <PopoverPrimitive.Content ref={ref} sideOffset={sideOffset} className={cn('osb-pop osb-pop--pad', className)} {...props} />
    </PopoverPrimitive.Portal>
  ));
PopoverContent.displayName = 'PopoverContent';

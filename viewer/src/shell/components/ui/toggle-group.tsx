// OpenSculptBoy
// Apache License, Version 2.0
//
// shadcn/ui's Toggle Group (Radix Toggle Group) as the design system's Segmented control: one stone, its cells flush,
// the chosen one raised a step.

import * as React from 'react';
import * as ToggleGroupPrimitive from '@radix-ui/react-toggle-group';
import { cn } from '../../lib/utils.ts';

export const ToggleGroup = React.forwardRef<React.ElementRef<typeof ToggleGroupPrimitive.Root>, React.ComponentPropsWithoutRef<typeof ToggleGroupPrimitive.Root> & { fill?: boolean }>(
  ({ className, fill, ...props }, ref) => <ToggleGroupPrimitive.Root ref={ref} className={cn('osb-seg', fill && 'osb-seg--fill', className)} {...(props as any)} />);
ToggleGroup.displayName = 'ToggleGroup';

export const ToggleGroupItem = ToggleGroupPrimitive.Item;

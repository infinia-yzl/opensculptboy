// OpenSculptBoy
// Apache License, Version 2.0
//
// shadcn/ui's Tabs (Radix Tabs): a tab list that the arrow keys move along. The sub-tabs of a mode use the design
// system's seamless button group (osb-seg) with icons over the labels.

import * as React from 'react';
import * as TabsPrimitive from '@radix-ui/react-tabs';
import { cn } from '../../lib/utils.ts';

export const Tabs = TabsPrimitive.Root;
export const TabsList = React.forwardRef<React.ElementRef<typeof TabsPrimitive.List>, React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>>(
  ({ className, ...props }, ref) => <TabsPrimitive.List ref={ref} className={cn('osb-seg osb-seg--fill osb-seg--tall', className)} {...props} />);
TabsList.displayName = 'TabsList';
export const TabsTrigger = React.forwardRef<React.ElementRef<typeof TabsPrimitive.Trigger>, React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>>(
  ({ className, ...props }, ref) => <TabsPrimitive.Trigger ref={ref} className={cn('osb-subtab-seg', className)} {...props} />);
TabsTrigger.displayName = 'TabsTrigger';

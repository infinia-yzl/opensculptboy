// OpenSculptBoy
// Apache License, Version 2.0
//
// shadcn/ui's Radio Group (Radix Radio Group) as the design system's swatches: round stones, the chosen one ringed in
// gold; the arrow keys move between them.

import * as React from 'react';
import * as RadioGroupPrimitive from '@radix-ui/react-radio-group';
import { cn } from '../../lib/utils.ts';

export const RadioGroup = React.forwardRef<React.ElementRef<typeof RadioGroupPrimitive.Root>, React.ComponentPropsWithoutRef<typeof RadioGroupPrimitive.Root>>(
  ({ className, ...props }, ref) => <RadioGroupPrimitive.Root ref={ref} className={cn('osb-swatch-row', className)} {...props} />);
RadioGroup.displayName = 'RadioGroup';

export const RadioGroupItem = React.forwardRef<React.ElementRef<typeof RadioGroupPrimitive.Item>, React.ComponentPropsWithoutRef<typeof RadioGroupPrimitive.Item>>(
  ({ className, ...props }, ref) => <RadioGroupPrimitive.Item ref={ref} className={cn('osb-sw', className)} {...props} />);
RadioGroupItem.displayName = 'RadioGroupItem';

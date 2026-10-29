// OpenSculptBoy
// Apache License, Version 2.0
//
// shadcn/ui's Dialog (Radix Dialog): a panel on the scrim that traps the focus; Esc and a click on the scrim close it.

import * as React from 'react';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { cn } from '../../lib/utils.ts';
import { Icon } from './icon.tsx';

export const Dialog = DialogPrimitive.Root;
export const DialogTitle = DialogPrimitive.Title;
export const DialogDescription = DialogPrimitive.Description;
export const DialogClose = DialogPrimitive.Close;

export const DialogContent = React.forwardRef<React.ElementRef<typeof DialogPrimitive.Content>, React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content>>(
  ({ className, children, ...props }, ref) => (
    <DialogPrimitive.Portal>
      <DialogPrimitive.Overlay className="osb-overlay osb-dialog-overlay" />
      <DialogPrimitive.Content ref={ref} className={cn('osb-dialog osb-dialog-box', className)} {...props}>
        {children}
        <DialogPrimitive.Close className="osb-ib osb-dialog-x" aria-label="Close (Esc)"><Icon name="close" size={16} /></DialogPrimitive.Close>
      </DialogPrimitive.Content>
    </DialogPrimitive.Portal>
  ));
DialogContent.displayName = 'DialogContent';

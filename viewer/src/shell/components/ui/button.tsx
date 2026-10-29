// OpenSculptBoy
// Apache License, Version 2.0
//
// shadcn/ui's Button (Radix Slot for asChild), with the design system's variants: default is the gold tablet
// (osb-btn--primary), outline the stone face (osb-btn), ghost the quiet one (osb-btn--quiet), and size icon the
// square icon button (osb-ib).

import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '../../lib/utils.ts';

const buttonVariants = cva('', {
  variants: {
    variant: {
      default: 'osb-btn osb-btn--primary',
      outline: 'osb-btn',
      ghost: 'osb-btn osb-btn--quiet',
      link: 'osb-link',
    },
    size: { default: '', lg: 'osb-btn--l', icon: '' },
  },
  compoundVariants: [
    { size: 'icon', className: 'osb-ib' },
  ],
  defaultVariants: { variant: 'default', size: 'default' },
});

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant, size, asChild = false, type, ...props }, ref) => {
  const Comp: any = asChild ? Slot : 'button';
  // an icon button is quiet: no tablet, no face
  const cls = size === 'icon' ? 'osb-ib' : buttonVariants({ variant, size });
  return <Comp ref={ref} type={asChild ? undefined : type || 'button'} className={cn(cls, className)} {...props} />;
});
Button.displayName = 'Button';

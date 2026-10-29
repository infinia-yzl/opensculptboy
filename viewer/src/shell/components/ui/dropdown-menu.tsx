// OpenSculptBoy
// Apache License, Version 2.0
//
// shadcn/ui's Dropdown Menu (Radix Dropdown Menu), in the design system's popover (osb-pop) and items (osb-item).

import * as React from 'react';
import * as Menu from '@radix-ui/react-dropdown-menu';
import { cn } from '../../lib/utils.ts';
import { Icon } from './icon.tsx';

export const DropdownMenu = Menu.Root;
export const DropdownMenuTrigger = Menu.Trigger;
export const DropdownMenuGroup = Menu.Group;
export const DropdownMenuRadioGroup = Menu.RadioGroup;

export const DropdownMenuContent = React.forwardRef<React.ElementRef<typeof Menu.Content>, React.ComponentPropsWithoutRef<typeof Menu.Content>>(
  ({ className, sideOffset = 8, ...props }, ref) => (
    <Menu.Portal>
      <Menu.Content ref={ref} sideOffset={sideOffset} className={cn('osb-pop osb-menu', className)} {...props} />
    </Menu.Portal>
  ));
DropdownMenuContent.displayName = 'DropdownMenuContent';

type ItemProps = React.ComponentPropsWithoutRef<typeof Menu.Item> & { icon?: string; shortcut?: string };
export const DropdownMenuItem = React.forwardRef<React.ElementRef<typeof Menu.Item>, ItemProps>(({ className, icon, shortcut, children, ...props }, ref) => (
  <Menu.Item ref={ref} className={cn('osb-item', className)} {...props}>
    {icon ? <Icon name={icon} size={16} /> : null}
    <span>{children}</span>
    {shortcut ? <kbd>{shortcut}</kbd> : null}
  </Menu.Item>
));
DropdownMenuItem.displayName = 'DropdownMenuItem';

type RadioProps = React.ComponentPropsWithoutRef<typeof Menu.RadioItem> & { icon?: string };
export const DropdownMenuRadioItem = React.forwardRef<React.ElementRef<typeof Menu.RadioItem>, RadioProps>(({ className, icon, children, ...props }, ref) => (
  <Menu.RadioItem ref={ref} className={cn('osb-item', className)} {...props}>
    {icon ? <Icon name={icon} size={16} /> : null}
    <span>{children}</span>
    <Menu.ItemIndicator className="osb-item-check"><Icon name="check" size={16} /></Menu.ItemIndicator>
  </Menu.RadioItem>
));
DropdownMenuRadioItem.displayName = 'DropdownMenuRadioItem';

export const DropdownMenuLabel = ({ children }: { children: React.ReactNode }) => <Menu.Label className="osb-pop-label">{children}</Menu.Label>;
export const DropdownMenuSeparator = () => <Menu.Separator className="osb-pop-sep" />;

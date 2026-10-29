// OpenSculptBoy
// Apache License, Version 2.0
//
// shadcn/ui's class helper: clsx for conditional classes, tailwind-merge so a later Tailwind class wins.

import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

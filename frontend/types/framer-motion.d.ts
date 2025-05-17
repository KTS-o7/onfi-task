import type * as React from "react"

declare module "framer-motion" {
  export interface MotionProps {
    children?: React.ReactNode
  }

  export interface AnimatePresenceProps {
    children?: React.ReactNode
  }

  export const motion: {
    [key: string]: React.ForwardRefExoticComponent<any>
  }

  export const AnimatePresence: React.FC<AnimatePresenceProps>
}

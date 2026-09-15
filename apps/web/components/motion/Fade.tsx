import type { ReactNode } from "react";

/**
 * Enter transition for panels that appear in place: the analysis progress panel, API
 * errors, the selected sample. Like every animation here the finished state is the base
 * state, so the panel is readable even if the animation never plays.
 */
export function Fade({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={`fade-in ${className ?? ""}`}>{children}</div>;
}

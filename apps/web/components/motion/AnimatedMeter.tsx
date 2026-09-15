interface AnimatedMeterProps {
  /** Fraction of the track to fill, in [0, 1]. */
  value: number;
  className: string;
}

/**
 * The filled part of a meter. The width attribute is always the measured value: the growth
 * is a CSS animation that scales the bar up from its left edge, so if the animation never
 * runs (reduced motion, a throttled background tab, printing, no JavaScript) the bar is
 * already at the right length. A reader must never see a bar that disagrees with the
 * number printed beside it.
 */
export function AnimatedMeter({ value, className }: AnimatedMeterProps) {
  const width = `${Math.min(Math.max(value, 0), 1) * 100}%`;
  return <div className={`meter-fill ${className}`} style={{ width }} />;
}

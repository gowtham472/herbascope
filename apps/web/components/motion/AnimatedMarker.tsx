interface AnimatedMarkerProps {
  /** Position of the sample on the fixed calibrated scale, as a CSS percentage. */
  left: string;
  className: string;
}

/**
 * The sample marker on the unknown-risk gauge. It is placed at the measured distance and
 * pops into view there; it never travels along the scale, because a marker sliding past
 * the calibrated boundaries would read as the sample moving between them.
 */
export function AnimatedMarker({ left, className }: AnimatedMarkerProps) {
  return <span className={`marker-pop ${className}`} style={{ left }} />;
}

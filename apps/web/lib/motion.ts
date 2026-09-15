/**
 * Motion vocabulary shared by the CSS utilities in `app/globals.css`.
 *
 * Every animation in the app is CSS: the curves are tuned to feel like a spring settling
 * (`--ease-settle` overshoots slightly, `--ease-glide` does not), and the finished state is
 * always the element's base state, so a value is never hidden behind an animation.
 */

/** Seconds added per list position, so a row of cards arrives as a sequence, not a flash. */
export const STAGGER_STEP = 0.05;

/** Cap the stagger: a long history list must not make the last row wait. */
export const MAX_STAGGER = 0.3;

export function staggerDelay(index = 0): number {
  return Math.min(index * STAGGER_STEP, MAX_STAGGER);
}

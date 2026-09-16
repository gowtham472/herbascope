"use client";

import { type ReactNode, useEffect, useRef, useState } from "react";

/**
 * A section that scrolls sideways while the reader scrolls down: the section is taller than
 * the viewport, its contents stick, and vertical progress is mapped to horizontal travel.
 *
 * The un-enhanced markup is an ordinary horizontally scrollable rail with snap points, so
 * without JavaScript, on narrow screens, or with reduced motion the same cards are still
 * there and still reachable by swipe, keyboard and scrollbar.
 */
export function PinnedRail({ heading, children }: { heading: ReactNode; children: ReactNode }) {
  const sectionRef = useRef<HTMLElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const travelRef = useRef(0);
  const [pinned, setPinned] = useState(false);
  const [travel, setTravel] = useState(0);

  useEffect(() => {
    const wideEnough = window.matchMedia("(min-width: 1024px)");
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (!wideEnough.matches || reduced.matches) return;

    const track = trackRef.current;
    const section = sectionRef.current;
    if (!track || !section) return;
    setPinned(true);

    function measure() {
      if (!track) return;
      const distance = Math.max(track.scrollWidth - window.innerWidth + 96, 0);
      travelRef.current = distance;
      setTravel(distance);
    }

    // Apply directly, so the track is correct on mount, on resize and when a background tab
    // becomes visible again; animation frames do not run while a tab is hidden.
    function apply() {
      if (!track || !section) return;
      const scrollable = section.offsetHeight - window.innerHeight;
      const progress = scrollable > 0 ? Math.min(Math.max(-section.getBoundingClientRect().top / scrollable, 0), 1) : 0;
      track.style.transform = `translate3d(${-progress * travelRef.current}px, 0, 0)`;
    }

    let frame = 0;
    function onScroll() {
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        apply();
      });
    }

    function remeasure() {
      measure();
      apply();
    }

    remeasure();
    const observer = new ResizeObserver(remeasure);
    observer.observe(track);
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", remeasure);
    document.addEventListener("visibilitychange", remeasure);

    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", remeasure);
      document.removeEventListener("visibilitychange", remeasure);
      observer.disconnect();
      cancelAnimationFrame(frame);
      track.style.transform = "";
    };
  }, []);

  return (
    <section ref={sectionRef} style={pinned ? { height: `calc(100vh + ${travel}px)` } : undefined} className="relative">
      <div className={pinned ? "sticky top-14 flex h-[calc(100vh-3.5rem)] flex-col justify-center overflow-hidden" : ""}>
        <div className="mx-auto w-full max-w-6xl px-5 sm:px-8">{heading}</div>
        <div
          ref={trackRef}
          className={
            pinned
              ? "mt-10 flex gap-5 pl-[max(1.25rem,calc((100vw-72rem)/2+1.25rem))] will-change-transform"
              : "h-track mt-8 flex snap-x gap-5 overflow-x-auto px-5 pb-4 sm:px-8"
          }
        >
          {children}
        </div>
      </div>
    </section>
  );
}

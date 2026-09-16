"use client";

import { ArrowLeftIcon, ArrowRightIcon } from "@phosphor-icons/react/ssr";
import { type ReactNode, useCallback, useEffect, useRef, useState } from "react";

/**
 * A horizontally scrollable row of cards with snap points and arrow controls: a plain
 * overflow container, so swipe, trackpad, scrollbar and keyboard all work natively. The
 * arrows are an addition for mouse users, and disable themselves at each end.
 */
export function SnapRail({ label, children }: { label: string; children: ReactNode }) {
  const railRef = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ start: true, end: false });

  const sync = useCallback(() => {
    const rail = railRef.current;
    if (!rail) return;
    setEdges({
      start: rail.scrollLeft <= 4,
      end: rail.scrollLeft >= rail.scrollWidth - rail.clientWidth - 4,
    });
  }, []);

  useEffect(() => {
    const rail = railRef.current;
    if (!rail) return;
    sync();
    rail.addEventListener("scroll", sync, { passive: true });
    const observer = new ResizeObserver(sync);
    observer.observe(rail);
    return () => {
      rail.removeEventListener("scroll", sync);
      observer.disconnect();
    };
  }, [sync]);

  function nudge(direction: 1 | -1) {
    const rail = railRef.current;
    if (!rail) return;
    const step = Math.max(rail.clientWidth * 0.8, 280);
    rail.scrollBy({ left: direction * step, behavior: "smooth" });
  }

  return (
    <div className="relative">
      <div
        ref={railRef}
        tabIndex={0}
        role="group"
        aria-label={label}
        className="h-track flex snap-x gap-5 overflow-x-auto px-5 pb-2 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-brand-600 sm:px-8"
      >
        {children}
      </div>

      <div className="mt-5 flex justify-end gap-2 px-5 sm:px-8">
        {([-1, 1] as const).map((direction) => (
          <button
            key={direction}
            type="button"
            onClick={() => nudge(direction)}
            disabled={direction === -1 ? edges.start : edges.end}
            aria-label={direction === -1 ? `Scroll ${label} left` : `Scroll ${label} right`}
            className="press grid size-10 place-items-center rounded-full border border-line bg-surface text-ink hover:bg-canvas disabled:opacity-35 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
          >
            {direction === -1 ? (
              <ArrowLeftIcon aria-hidden="true" weight="bold" className="size-4" />
            ) : (
              <ArrowRightIcon aria-hidden="true" weight="bold" className="size-4" />
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

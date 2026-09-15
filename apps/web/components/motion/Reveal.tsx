"use client";

import { type ElementType, type ReactNode, useEffect, useLayoutEffect, useRef } from "react";

import { staggerDelay } from "@/lib/motion";

/** Layout effect in the browser, plain effect on the server, so nothing flashes on mount. */
const useIsomorphicLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

interface RevealProps {
  children: ReactNode;
  /** Position within a group; later items arrive slightly later. */
  index?: number;
  className?: string;
  /** Element to render: lists need `li`, sections need `section`. */
  as?: ElementType;
}

/**
 * Reveals a block when it first scrolls into view.
 *
 * The finished state is the base state: the markup renders visible, and only after mount
 * does this component hide it and hand it to an IntersectionObserver. If JavaScript never
 * runs, fails, or the reader prefers reduced motion, the content is simply there. Nothing
 * a reader needs is locked behind an animation.
 */
export function Reveal({ children, index = 0, className, as: Tag = "div" }: RevealProps) {
  const ref = useRef<HTMLElement>(null);

  useIsomorphicLayoutEffect(() => {
    const node = ref.current;
    if (!node || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    // Without IntersectionObserver there is no way to know when the block is on screen, so
    // it is never hidden in the first place.
    if (typeof IntersectionObserver === "undefined") return;

    node.classList.add("reveal-pending");
    node.style.setProperty("--reveal-delay", `${staggerDelay(index)}s`);
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        node.classList.remove("reveal-pending");
        node.classList.add("reveal-run");
        observer.disconnect();
      },
      { threshold: 0.15 },
    );
    observer.observe(node);

    return () => {
      observer.disconnect();
      node.classList.remove("reveal-pending");
    };
  }, [index]);

  // `min-w-0`: this wrapper usually sits directly inside a grid or flex container, where the
  // default `min-width: auto` would let wide content (a long filename, a table) push the
  // column past the viewport.
  return (
    <Tag ref={ref} data-reveal className={`min-w-0 ${className ?? ""}`}>
      {children}
    </Tag>
  );
}

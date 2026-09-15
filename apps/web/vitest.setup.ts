import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

/**
 * jsdom implements neither of these. Tests run as a reader with "reduce motion" enabled,
 * which is also the state every animated component must render correctly in: the finished
 * content, no animation. `__tests__/motion.test.tsx` flips the query to cover the animated
 * path.
 */
function matchMedia(query: string): MediaQueryList {
  return {
    matches: query.includes("prefers-reduced-motion"),
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  } as unknown as MediaQueryList;
}

vi.stubGlobal("matchMedia", vi.fn(matchMedia));
vi.stubGlobal(
  "IntersectionObserver",
  class {
    constructor(private readonly callback: IntersectionObserverCallback) {}
    observe(target: Element) {
      this.callback(
        [{ target, isIntersecting: true, intersectionRatio: 1 } as IntersectionObserverEntry],
        this as unknown as IntersectionObserver,
      );
    }
    unobserve() {}
    disconnect() {}
    takeRecords() {
      return [];
    }
  },
);

afterEach(() => {
  cleanup();
});

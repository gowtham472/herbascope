import type { Tone } from "@/lib/format";

/** Literal Tailwind classes per semantic tone (kept literal so Tailwind can detect them). */
export const TONE_CLASSES: Record<Tone, { badge: string; bar: string; panel: string; text: string }> = {
  pass: {
    badge: "bg-pass-50 text-pass-700 ring-pass-200",
    bar: "bg-pass-600",
    panel: "border-pass-200 bg-pass-50",
    text: "text-pass-700",
  },
  review: {
    badge: "bg-review-50 text-review-700 ring-review-200",
    bar: "bg-review-600",
    panel: "border-review-200 bg-review-50",
    text: "text-review-700",
  },
  unknown: {
    badge: "bg-unknown-50 text-unknown-700 ring-unknown-200",
    bar: "bg-unknown-600",
    panel: "border-unknown-200 bg-unknown-50",
    text: "text-unknown-700",
  },
  fail: {
    badge: "bg-fail-50 text-fail-600 ring-fail-600/20",
    bar: "bg-fail-600",
    panel: "border-fail-600/20 bg-fail-50",
    text: "text-fail-600",
  },
  neutral: {
    badge: "bg-canvas text-muted ring-line",
    bar: "bg-brand-600",
    panel: "border-line bg-surface",
    text: "text-muted",
  },
};

import { CheckCircleIcon, InfoIcon, QuestionIcon, WarningIcon, XCircleIcon } from "@phosphor-icons/react/ssr";

import type { Tone } from "@/lib/format";

import { TONE_CLASSES } from "./tone";

const TONE_ICON = {
  pass: CheckCircleIcon,
  review: WarningIcon,
  unknown: QuestionIcon,
  fail: XCircleIcon,
  neutral: InfoIcon,
} as const;

interface StatusBadgeProps {
  tone: Tone;
  label: string;
  size?: "sm" | "lg";
}

export function StatusBadge({ tone, label, size = "sm" }: StatusBadgeProps) {
  const Icon = TONE_ICON[tone];
  const sizing = size === "lg" ? "gap-2 px-3.5 py-1.5 text-base" : "gap-1.5 px-2.5 py-0.5 text-xs";
  return (
    <span className={`inline-flex items-center rounded-full font-semibold ring-1 ring-inset ${sizing} ${TONE_CLASSES[tone].badge}`}>
      <Icon weight="bold" aria-hidden="true" className={size === "lg" ? "size-5" : "size-3.5"} />
      {label}
    </span>
  );
}

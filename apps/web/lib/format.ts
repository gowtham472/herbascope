import type { Agreement, DecisionStatus, QualityStatus, UnknownStatus } from "@/types";

export function percent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function decimal(value: number, digits = 3): string {
  return value.toFixed(digits);
}

/** Dataset class labels use underscores (e.g. `sirih_merah`); show them as words. */
export function classLabel(name: string): string {
  return name.replaceAll("_", " ");
}

export function sentenceCase(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function formatDateTime(iso: string): string {
  return new Intl.DateTimeFormat("en", { dateStyle: "medium", timeStyle: "short" }).format(new Date(iso));
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export type Tone = "pass" | "review" | "unknown" | "neutral" | "fail";

export const DECISION_LABEL: Record<DecisionStatus, string> = {
  PRELIMINARY_PASS: "Preliminary pass",
  REVIEW_REQUIRED: "Review required",
  UNKNOWN: "Unknown",
};

export const DECISION_TONE: Record<DecisionStatus, Tone> = {
  PRELIMINARY_PASS: "pass",
  REVIEW_REQUIRED: "review",
  UNKNOWN: "unknown",
};

export const UNKNOWN_TONE: Record<UnknownStatus, Tone> = {
  KNOWN: "pass",
  UNCERTAIN: "review",
  UNKNOWN: "unknown",
};

export const AGREEMENT_TONE: Record<Agreement, Tone> = {
  HIGH: "pass",
  MEDIUM: "review",
  LOW: "fail",
};

export const QUALITY_TONE: Record<QualityStatus, Tone> = {
  ACCEPTABLE: "pass",
  DEGRADED: "review",
};

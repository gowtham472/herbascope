import { TONE_CLASSES } from "@/components/common/tone";
import type { Tone } from "@/lib/format";

interface ConfidenceBarProps {
  label: string;
  /** Value in [0, 1]. */
  value: number;
  valueText: string;
  tone?: Tone;
  /** Optional calibrated policy threshold in [0, 1], drawn as a marker. */
  threshold?: number;
  thresholdLabel?: string;
}

export function ConfidenceBar({ label, value, valueText, tone = "neutral", threshold, thresholdLabel }: ConfidenceBarProps) {
  const width = `${Math.min(Math.max(value, 0), 1) * 100}%`;
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="text-muted">{label}</span>
        <span className="font-mono font-semibold tabular-nums text-ink">{valueText}</span>
      </div>
      <div
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={Number(value.toFixed(4))}
        aria-valuetext={thresholdLabel ? `${valueText}; ${thresholdLabel}` : valueText}
        className="relative mt-1.5 h-2.5 rounded-full bg-canvas ring-1 ring-inset ring-line"
      >
        <div className={`h-full rounded-full ${TONE_CLASSES[tone].bar}`} style={{ width }} />
        {threshold !== undefined ? (
          <div
            className="absolute -top-1 h-4.5 w-0.5 rounded bg-ink"
            style={{ left: `calc(${Math.min(Math.max(threshold, 0), 1) * 100}% - 1px)` }}
            title={thresholdLabel}
          />
        ) : null}
      </div>
      {thresholdLabel ? <p className="mt-1 text-xs text-muted">{thresholdLabel}</p> : null}
    </div>
  );
}

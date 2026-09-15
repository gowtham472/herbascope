import { QuestionIcon } from "@phosphor-icons/react/ssr";

import { MetricCard } from "@/components/common/MetricCard";
import { Panel } from "@/components/common/Panel";
import { StatusBadge } from "@/components/common/StatusBadge";
import { LABELS } from "@/lib/copy";
import { decimal, percent, UNKNOWN_TONE } from "@/lib/format";
import type { Unknown } from "@/types";

import { ConfidenceBar } from "./ConfidenceBar";

const STATUS_TEXT = {
  KNOWN: "Within the reference distribution",
  UNCERTAIN: "Atypical for validated reference material",
  UNKNOWN: "Outside the reference distribution",
} as const;

interface UnknownRiskCardProps {
  unknown: Unknown;
  maxUnknownRisk: number;
}

/** Position of a distance on the gauge; distances are cosine distances (usually 0–1). */
function gaugePosition(value: number, scaleMax: number): string {
  return `${(Math.min(Math.max(value, 0), scaleMax) / scaleMax) * 100}%`;
}

export function UnknownRiskCard({ unknown, maxUnknownRisk }: UnknownRiskCardProps) {
  const scaleMax = Math.max(1, unknown.distance);
  return (
    <Panel
      title={LABELS.unknown}
      icon={QuestionIcon}
      description="Distance to the nearest reference micrographs, against thresholds calibrated on real data."
      aside={<StatusBadge tone={UNKNOWN_TONE[unknown.status]} label={unknown.status} />}
    >
      <p className="font-medium text-ink">{STATUS_TEXT[unknown.status]}</p>

      <div className="mt-4">
        <p className="text-sm text-muted">Reference distance</p>
        <div className="relative mt-2 h-3 overflow-hidden rounded-full ring-1 ring-inset ring-line" aria-hidden="true">
          <div className="absolute inset-y-0 left-0 bg-pass-200" style={{ width: gaugePosition(unknown.known_boundary, scaleMax) }} />
          <div
            className="absolute inset-y-0 bg-review-200"
            style={{
              left: gaugePosition(unknown.known_boundary, scaleMax),
              width: `calc(${gaugePosition(unknown.threshold, scaleMax)} - ${gaugePosition(unknown.known_boundary, scaleMax)})`,
            }}
          />
          <div className="absolute inset-y-0 right-0 bg-unknown-200" style={{ left: gaugePosition(unknown.threshold, scaleMax) }} />
        </div>
        <div className="relative h-5" aria-hidden="true">
          <span
            className="absolute -top-4.5 size-4 -translate-x-1/2 rotate-45 rounded-sm border-2 border-surface bg-ink shadow"
            style={{ left: gaugePosition(unknown.distance, scaleMax) }}
          />
        </div>
        <div className="flex justify-between text-[11px] text-muted" aria-hidden="true">
          <span>known</span>
          <span>atypical</span>
          <span>unknown</span>
        </div>
        <p className="sr-only">
          Reference distance {decimal(unknown.distance)}; known boundary {decimal(unknown.known_boundary)}; unknown
          threshold {decimal(unknown.threshold)}.
        </p>
      </div>

      <dl className="mt-4 grid grid-cols-3 gap-2">
        <MetricCard label="Distance" value={decimal(unknown.distance)} />
        <MetricCard label="Known boundary" value={decimal(unknown.known_boundary)} />
        <MetricCard label="Unknown above" value={decimal(unknown.threshold)} />
      </dl>

      <div className="mt-4">
        <ConfidenceBar
          label="Unknown risk"
          value={unknown.risk}
          valueText={percent(unknown.risk)}
          tone={unknown.risk <= maxUnknownRisk ? "pass" : "unknown"}
          threshold={maxUnknownRisk}
          thresholdLabel={`Policy maximum ${percent(maxUnknownRisk)} · calibration ${unknown.calibration_version}`}
        />
      </div>
    </Panel>
  );
}

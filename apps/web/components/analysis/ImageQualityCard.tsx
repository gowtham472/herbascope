import { CheckIcon, SealCheckIcon, XIcon } from "@phosphor-icons/react/ssr";

import { Panel } from "@/components/common/Panel";
import { StatusBadge } from "@/components/common/StatusBadge";
import { QUALITY_TONE } from "@/lib/format";
import type { Quality } from "@/types";

function formatValue(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toPrecision(3);
}

export function ImageQualityCard({ quality }: { quality: Quality }) {
  return (
    <Panel
      title="Image Quality"
      icon={SealCheckIcon}
      description="Bounds calibrated on the reference-library training images. Quality affects review, never species identity."
      aside={<StatusBadge tone={QUALITY_TONE[quality.status]} label={quality.status} />}
    >
      <ul className="divide-y divide-line text-sm">
        {quality.checks.map((check) => (
          <li key={check.name} className="flex items-center justify-between gap-3 py-2">
            <span className="flex items-center gap-2">
              {check.passed ? (
                <CheckIcon aria-label="passed" weight="bold" className="size-4 text-pass-600" />
              ) : (
                <XIcon aria-label="failed" weight="bold" className="size-4 text-fail-600" />
              )}
              {check.label}
            </span>
            <span className="font-mono text-xs tabular-nums text-muted">
              <span className="text-ink">{formatValue(check.value)}</span> {check.comparison} {formatValue(check.bound)}
            </span>
          </li>
        ))}
      </ul>
      <p className="mt-2 font-mono text-[11px] text-muted">bounds {quality.version}</p>
    </Panel>
  );
}

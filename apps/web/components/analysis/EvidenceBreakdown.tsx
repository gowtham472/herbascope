import { percent } from "@/lib/format";
import type { StrengthComponents } from "@/types";

import { ConfidenceBar } from "./ConfidenceBar";

interface EvidenceBreakdownProps {
  strength: number;
  components: StrengthComponents;
}

export function EvidenceBreakdown({ strength, components }: EvidenceBreakdownProps) {
  const rows: [string, number][] = [
    ["Classifier confidence", components.classifier_confidence],
    ["Reference support for prediction", components.reference_support],
    ["1 − unknown risk", components.unknown_margin],
    ["Quality checks passed", components.quality_pass_fraction],
  ];
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-ink">Evidence strength</h3>
        <span className="font-mono text-lg font-semibold tabular-nums">{percent(strength)}</span>
      </div>
      <p className="text-xs text-muted">
        Geometric mean of the components below — a descriptive index, not a probability, and not used by the
        decision policy.
      </p>
      <ul className="mt-3 space-y-2">
        {rows.map(([label, value]) => (
          <li key={label}>
            <ConfidenceBar label={label} value={value} valueText={percent(value)} />
          </li>
        ))}
      </ul>
    </div>
  );
}

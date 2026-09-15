import { TargetIcon } from "@phosphor-icons/react/ssr";

import { Panel } from "@/components/common/Panel";
import { LABELS } from "@/lib/copy";
import { classLabel, percent } from "@/lib/format";
import type { DecisionThresholds, Prediction } from "@/types";

import { ConfidenceBar } from "./ConfidenceBar";

interface PredictionCardProps {
  prediction: Prediction;
  thresholds: DecisionThresholds;
}

export function PredictionCard({ prediction, thresholds }: PredictionCardProps) {
  const meetsPolicy = prediction.confidence >= thresholds.min_classifier_confidence;
  return (
    <Panel title={LABELS.prediction} icon={TargetIcon} description="Logistic regression on the DINOv2 embedding.">
      <p className="text-2xl font-semibold capitalize text-ink">{classLabel(prediction.class_name)}</p>
      <p className="font-mono text-xs text-muted">{prediction.class_name}</p>
      <div className="mt-4">
        <ConfidenceBar
          label="Classifier confidence"
          value={prediction.confidence}
          valueText={percent(prediction.confidence)}
          tone={meetsPolicy ? "pass" : "review"}
          threshold={thresholds.min_classifier_confidence}
          thresholdLabel={`Policy minimum ${percent(thresholds.min_classifier_confidence)} (calibrated)`}
        />
      </div>
      <h3 className="mt-5 text-xs font-semibold uppercase tracking-wide text-muted">Class probabilities</h3>
      <ul className="mt-2 space-y-2">
        {prediction.top_k.map((item) => (
          <li key={item.class_name}>
            <ConfidenceBar label={classLabel(item.class_name)} value={item.probability} valueText={percent(item.probability)} />
          </li>
        ))}
      </ul>
      <p className="mt-4 text-xs text-muted">
        A confidence score alone cannot tell whether the sample belongs to any supported class — see the reference
        evidence and unknown risk.
      </p>
    </Panel>
  );
}

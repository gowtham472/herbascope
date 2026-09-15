import { CubeIcon } from "@phosphor-icons/react/ssr";

import { Panel } from "@/components/common/Panel";
import { classLabel } from "@/lib/format";
import type { ModelInfo } from "@/types";

export function ModelMetadata({ model }: { model: ModelInfo }) {
  const rows: [string, string][] = [
    ["Encoder", model.encoder],
    ["Classifier", model.classifier_version],
    ["Reference index", `${model.index_version} · ${model.reference_count} references`],
    ["Unknown calibration", model.unknown_calibration_version],
    ["Quality bounds", model.quality_version],
    ["Decision policy", model.policy_version],
    ["Preprocessing", model.preprocessing_version],
    ["Supported classes", model.classes.map(classLabel).join(", ")],
  ];
  return (
    <Panel title="Model Metadata" icon={CubeIcon} description="Artifact versions that produced this result.">
      <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
        {rows.map(([label, value]) => (
          <div key={label} className="flex justify-between gap-3 border-b border-line/70 pb-2">
            <dt className="text-muted">{label}</dt>
            <dd className="text-right font-mono text-xs leading-5 text-ink">{value}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-3 break-all font-mono text-[11px] text-muted" title="Embedding fingerprint">
        {model.embedding_fingerprint}
      </p>
    </Panel>
  );
}

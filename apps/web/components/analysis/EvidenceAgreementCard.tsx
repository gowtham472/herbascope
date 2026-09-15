import { ArrowsLeftRightIcon, ScalesIcon } from "@phosphor-icons/react/ssr";

import { Panel } from "@/components/common/Panel";
import { StatusBadge } from "@/components/common/StatusBadge";
import { AGREEMENT_TONE, classLabel, percent } from "@/lib/format";
import type { Evidence, Prediction, Retrieval } from "@/types";

import { ConfidenceBar } from "./ConfidenceBar";
import { EvidenceBreakdown } from "./EvidenceBreakdown";

const AGREEMENT_TEXT = {
  HIGH: "Every retrieved reference matches the predicted class.",
  MEDIUM: "Retrieval favours the predicted class, but the references are mixed.",
  LOW: "Retrieval favours a different class than the classifier.",
} as const;

interface EvidenceAgreementCardProps {
  prediction: Prediction;
  retrieval: Retrieval;
  evidence: Evidence;
}

export function EvidenceAgreementCard({ prediction, retrieval, evidence }: EvidenceAgreementCardProps) {
  return (
    <Panel
      title="Evidence Agreement"
      icon={ScalesIcon}
      description="Complementary analyses of the same visual representation — not independent tests."
      aside={<StatusBadge tone={AGREEMENT_TONE[evidence.agreement]} label={evidence.agreement} />}
    >
      <div className="flex items-center justify-between gap-2 rounded-lg bg-canvas px-3 py-2 text-sm">
        <div>
          <p className="text-xs text-muted">Classifier</p>
          <p className="font-semibold capitalize">{classLabel(prediction.class_name)}</p>
        </div>
        <ArrowsLeftRightIcon aria-hidden="true" className="size-4 text-muted" />
        <div className="text-right">
          <p className="text-xs text-muted">Reference retrieval</p>
          <p className="font-semibold capitalize">{classLabel(retrieval.retrieved_class)}</p>
        </div>
      </div>
      <p className="mt-2 text-sm text-muted">{AGREEMENT_TEXT[evidence.agreement]}</p>

      <h3 className="mt-4 text-xs font-semibold uppercase tracking-wide text-muted">
        Similarity-weighted vote over {retrieval.matches.length} references
      </h3>
      <ul className="mt-2 space-y-2">
        {retrieval.class_support.map((support) => (
          <li key={support.class_name}>
            <ConfidenceBar
              label={`${classLabel(support.class_name)} (${support.count})`}
              value={support.weighted_share}
              valueText={percent(support.weighted_share)}
              tone={support.class_name === prediction.class_name ? "pass" : "neutral"}
            />
          </li>
        ))}
      </ul>

      <div className="mt-5 border-t border-line pt-4">
        <EvidenceBreakdown strength={evidence.strength} components={evidence.strength_components} />
      </div>
    </Panel>
  );
}

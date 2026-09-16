import { CheckIcon, GavelIcon, XIcon } from "@phosphor-icons/react/ssr";

import { StatusBadge } from "@/components/common/StatusBadge";
import { TONE_CLASSES } from "@/components/common/tone";
import { LABELS } from "@/lib/copy";
import { DECISION_LABEL, DECISION_TONE } from "@/lib/format";
import type { Decision } from "@/types";

const OUTCOME_TEXT = {
  PRELIMINARY_PASS: "The available visual evidence is strong and consistent enough for a preliminary screening pass.",
  REVIEW_REQUIRED: "The evidence is borderline or conflicting. A trained reviewer should examine this sample.",
  UNKNOWN: "The sample does not resemble the supported reference material closely enough to screen it.",
} as const;

export function DecisionCard({ decision }: { decision: Decision }) {
  const tone = DECISION_TONE[decision.status];
  return (
    <section
      aria-labelledby="decision-heading"
      className={`overflow-hidden rounded-2xl border break-inside-avoid ${TONE_CLASSES[tone].panel}`}
    >
      <div className={`h-1.5 w-full ${TONE_CLASSES[tone].bar}`} aria-hidden="true" />
      <div className="p-5 sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 id="decision-heading" className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-muted">
            <GavelIcon aria-hidden="true" weight="duotone" className="size-5" />
            {LABELS.decision}
          </h2>
          <span className="font-mono text-xs text-muted">policy {decision.policy_version}</span>
        </div>
        <div className="mt-4">
          <StatusBadge tone={tone} label={DECISION_LABEL[decision.status]} size="lg" />
        </div>
        <p className={`mt-4 text-lg font-semibold leading-relaxed ${TONE_CLASSES[tone].text}`}>{OUTCOME_TEXT[decision.status]}</p>

        <h3 className="mt-5 text-xs font-semibold uppercase tracking-[0.18em] text-muted">Decision reason</h3>
        <p className="mt-1.5 text-sm leading-relaxed text-ink">{decision.reason}</p>

        <div className="mt-5 overflow-x-auto rounded-xl border border-line bg-surface">
        <table className="w-full text-sm">
          <caption className="sr-only">Preliminary-pass criteria</caption>
          <thead className="bg-canvas text-left text-xs text-muted">
            <tr>
              <th scope="col" className="px-3 py-2 font-medium">Pass criterion</th>
              <th scope="col" className="px-3 py-2 font-medium">Observed</th>
              <th scope="col" className="px-3 py-2 font-medium">Required</th>
              <th scope="col" className="px-3 py-2 font-medium">Met</th>
            </tr>
          </thead>
          <tbody>
            {decision.checks.map((check) => (
              <tr key={check.name} className="border-t border-line">
                <th scope="row" className="px-3 py-2 text-left font-normal text-ink">{check.label}</th>
                <td className="px-3 py-2 font-mono text-xs">{check.observed}</td>
                <td className="px-3 py-2 font-mono text-xs text-muted">{check.required}</td>
                <td className="px-3 py-2">
                  {check.passed ? (
                    <span className="inline-flex items-center gap-1 text-pass-700">
                      <CheckIcon aria-hidden="true" weight="bold" className="size-4" /> Yes
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-fail-600">
                      <XIcon aria-hidden="true" weight="bold" className="size-4" /> No
                    </span>
                  )}
                </td>
              </tr>
            ))}
            </tbody>
          </table>
        </div>
        {decision.status === "UNKNOWN" ? (
          <p className="mt-3 text-xs text-muted">
            UNKNOWN takes precedence: once the sample is outside the reference distribution the criteria above are
            shown for transparency only.
          </p>
        ) : null}
      </div>
    </section>
  );
}

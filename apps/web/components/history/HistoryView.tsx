"use client";

import { ClockCounterClockwiseIcon, MicroscopeIcon } from "@phosphor-icons/react/ssr";
import Image from "next/image";
import Link from "next/link";
import { useState } from "react";

import { EmptyState } from "@/components/common/EmptyState";
import { ErrorState } from "@/components/common/ErrorState";
import { StatusBadge } from "@/components/common/StatusBadge";
import { Reveal } from "@/components/motion/Reveal";
import { useApiResource } from "@/hooks/useApiResource";
import { apiUrl, listAnalyses } from "@/lib/api";
import { classLabel, DECISION_LABEL, DECISION_TONE, formatDateTime, percent } from "@/lib/format";
import type { DecisionStatus } from "@/types";

type Filter = DecisionStatus | "ALL";

const FILTERS: { value: Filter; label: string }[] = [
  { value: "ALL", label: "All" },
  { value: "PRELIMINARY_PASS", label: DECISION_LABEL.PRELIMINARY_PASS },
  { value: "REVIEW_REQUIRED", label: DECISION_LABEL.REVIEW_REQUIRED },
  { value: "UNKNOWN", label: DECISION_LABEL.UNKNOWN },
];

export function HistoryView() {
  const { state, retry } = useApiResource("history", listAnalyses);
  const [filter, setFilter] = useState<Filter>("ALL");

  if (state.status === "loading") {
    return (
      <div role="status" aria-label="Loading analysis history" className="space-y-2">
        {[0, 1, 2].map((key) => (
          <div key={key} className="h-20 skeleton rounded-xl" />
        ))}
      </div>
    );
  }
  if (state.status === "error") {
    return <ErrorState title="Could not load analysis history" message={state.error.message} onRetry={retry} />;
  }
  if (state.data.items.length === 0) {
    return (
      <EmptyState
        icon={ClockCounterClockwiseIcon}
        title="No analyses yet"
        description="Screening results are stored locally by the API and listed here, newest first."
        action={
          <Link
            href="/analyze"
            className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
          >
            <MicroscopeIcon aria-hidden="true" className="size-4" />
            Analyze a sample
          </Link>
        }
      />
    );
  }

  const counts = state.data.items.reduce<Record<string, number>>(
    (totals, item) => ({ ...totals, [item.decision_status]: (totals[item.decision_status] ?? 0) + 1 }),
    {},
  );
  const visible = filter === "ALL" ? state.data.items : state.data.items.filter((item) => item.decision_status === filter);

  return (
    <>
      <div role="group" aria-label="Filter by decision" className="mb-4 flex flex-wrap gap-2">
        {FILTERS.map(({ value, label }) => {
          const count = value === "ALL" ? state.data.items.length : (counts[value] ?? 0);
          const active = filter === value;
          return (
            <button
              key={value}
              type="button"
              aria-pressed={active}
              onClick={() => setFilter(value)}
              className={`press inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600 ${
                active ? "border-brand-500 bg-brand-50 text-brand-700" : "border-line bg-surface text-muted hover:text-ink"
              }`}
            >
              {label}
              <span className="font-mono text-xs tabular-nums">{count}</span>
            </button>
          );
        })}
      </div>

      {visible.length === 0 ? (
        <p role="status" className="rounded-xl border border-dashed border-line p-6 text-center text-sm text-muted">
          No analyses with this decision.
        </p>
      ) : null}

      <ul className="space-y-2">
        {visible.map((item, index) => (
        <Reveal as="li" key={item.id} index={index}>
          <Link
            href={`/results/${item.id}`}
            className="lift flex items-center gap-4 rounded-xl border border-line bg-surface p-3 shadow-sm hover:border-brand-500/60 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
          >
            <span className="relative size-14 shrink-0 overflow-hidden rounded-lg bg-canvas">
              <Image src={apiUrl(item.image_url)} alt="" fill unoptimized sizes="56px" className="object-cover" />
            </span>
            <span className="min-w-0 flex-1">
              <span className="block truncate font-medium text-ink">{item.filename}</span>
              <span className="block text-sm text-muted">
                <span className="capitalize">{classLabel(item.predicted_class)}</span> · {percent(item.confidence)} confidence
              </span>
              <span className="block font-mono text-xs text-muted">{formatDateTime(item.created_at)}</span>
            </span>
              <StatusBadge tone={DECISION_TONE[item.decision_status]} label={DECISION_LABEL[item.decision_status]} />
            </Link>
          </Reveal>
        ))}
      </ul>
    </>
  );
}

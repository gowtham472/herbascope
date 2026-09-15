"use client";

import { ClockCounterClockwiseIcon, MicroscopeIcon } from "@phosphor-icons/react/ssr";
import Image from "next/image";
import Link from "next/link";

import { EmptyState } from "@/components/common/EmptyState";
import { ErrorState } from "@/components/common/ErrorState";
import { StatusBadge } from "@/components/common/StatusBadge";
import { Reveal } from "@/components/motion/Reveal";
import { useApiResource } from "@/hooks/useApiResource";
import { apiUrl, listAnalyses } from "@/lib/api";
import { classLabel, DECISION_LABEL, DECISION_TONE, formatDateTime, percent } from "@/lib/format";

export function HistoryView() {
  const { state, retry } = useApiResource("history", listAnalyses);

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

  return (
    <ul className="space-y-2">
      {state.data.items.map((item, index) => (
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
  );
}

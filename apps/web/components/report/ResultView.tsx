"use client";

import { MicroscopeIcon } from "@phosphor-icons/react/ssr";
import Link from "next/link";

import { DecisionCard } from "@/components/analysis/DecisionCard";
import { EvidenceAgreementCard } from "@/components/analysis/EvidenceAgreementCard";
import { ImageQualityCard } from "@/components/analysis/ImageQualityCard";
import { PredictionCard } from "@/components/analysis/PredictionCard";
import { ReferenceMatches } from "@/components/analysis/ReferenceMatches";
import { UnknownRiskCard } from "@/components/analysis/UnknownRiskCard";
import { ErrorState } from "@/components/common/ErrorState";
import { Reveal } from "@/components/motion/Reveal";
import { ImagePreview } from "@/components/upload/ImagePreview";
import { useApiResource } from "@/hooks/useApiResource";
import { apiUrl, getAnalysis } from "@/lib/api";
import { formatDateTime } from "@/lib/format";
import type { AnalysisResponse } from "@/types";

import { LimitationsCard } from "./LimitationsCard";
import { ModelMetadata } from "./ModelMetadata";
import { ScreeningSummary } from "./ScreeningSummary";

const NEW_ANALYSIS_LINK = (
  <Link
    href="/analyze"
    className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-brand-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
  >
    <MicroscopeIcon aria-hidden="true" className="size-4" />
    New analysis
  </Link>
);

function LoadingResult() {
  return (
    <div role="status" aria-label="Loading screening result" className="grid gap-4 lg:grid-cols-3">
      <div className="aspect-square skeleton rounded-xl" />
      <div className="skeleton rounded-xl lg:col-span-2" />
      {[0, 1, 2].map((key) => (
        <div key={key} className="h-72 skeleton rounded-xl" />
      ))}
    </div>
  );
}

function ResultReport({ analysis }: { analysis: AnalysisResponse }) {
  const { sample, prediction, retrieval, unknown, evidence, decision, model } = analysis;
  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-brand-600">Microscopic screening result</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink sm:text-3xl">{sample.filename}</h1>
          <p className="mt-1 font-mono text-xs text-muted">
            {formatDateTime(analysis.created_at)} · analysis {analysis.id}
          </p>
        </div>
        <div className="print:hidden">{NEW_ANALYSIS_LINK}</div>
      </header>

      <div className="grid gap-6 lg:grid-cols-[22rem_1fr]">
        <Reveal>
          <ImagePreview
            src={apiUrl(sample.image_url)}
            alt={`Uploaded sample ${sample.filename}`}
            details={[
              ["Format", sample.format],
              ["Dimensions", `${sample.width} × ${sample.height} px`],
            ]}
          />
        </Reveal>
        <Reveal index={1} className="space-y-6">
          <DecisionCard decision={decision} />
          <ScreeningSummary explanation={analysis.explanation} />
        </Reveal>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Reveal>
          <PredictionCard prediction={prediction} thresholds={decision.thresholds} />
        </Reveal>
        <Reveal index={1}>
          <UnknownRiskCard unknown={unknown} maxUnknownRisk={decision.thresholds.max_unknown_risk} />
        </Reveal>
        <Reveal index={2}>
          <EvidenceAgreementCard prediction={prediction} retrieval={retrieval} evidence={evidence} />
        </Reveal>
      </div>

      <Reveal>
        <ReferenceMatches
          retrieval={retrieval}
          predictedClass={prediction.class_name}
          minReferenceSimilarity={decision.thresholds.min_reference_similarity}
        />
      </Reveal>

      <div className="grid gap-6 lg:grid-cols-2">
        <Reveal>
          <ImageQualityCard quality={evidence.quality} />
        </Reveal>
        <Reveal index={1}>
          <ModelMetadata model={model} />
        </Reveal>
      </div>

      <Reveal>
        <LimitationsCard limitations={analysis.limitations} disclaimer={analysis.disclaimer} />
      </Reveal>
    </div>
  );
}

export function ResultView({ id }: { id: string }) {
  const { state, retry } = useApiResource(id, getAnalysis);

  if (state.status === "loading") return <LoadingResult />;
  if (state.status === "error") {
    const notFound = state.error.status === 404 || state.error.status === 422;
    return (
      <ErrorState
        title={notFound ? "Screening result not found" : "Could not load the screening result"}
        message={notFound ? `No analysis exists with id "${id}".` : state.error.message}
        onRetry={notFound ? undefined : retry}
        action={NEW_ANALYSIS_LINK}
      />
    );
  }
  return <ResultReport analysis={state.data} />;
}

"use client";

import { ArrowCounterClockwiseIcon, MicroscopeIcon } from "@phosphor-icons/react/ssr";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { AnalysisProgress } from "@/components/analysis/AnalysisProgress";
import { ErrorState } from "@/components/common/ErrorState";
import { Mascot } from "@/components/common/Mascot";
import { Fade } from "@/components/motion/Fade";
import { useApiResource } from "@/hooks/useApiResource";
import { analyzeImage, ApiError, getHealth } from "@/lib/api";
import { classLabel, formatBytes } from "@/lib/format";

import { ImagePreview } from "./ImagePreview";
import { ImageUploader } from "./ImageUploader";

interface Selection {
  file: File;
  previewUrl: string;
}

export function AnalyzeView() {
  const router = useRouter();
  const { state: health, retry: retryHealth } = useApiResource("health", getHealth);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => () => {
    if (selection) URL.revokeObjectURL(selection.previewUrl);
  }, [selection]);

  function select(file: File) {
    setError(null);
    setSelection({ file, previewUrl: URL.createObjectURL(file) });
  }

  async function submit() {
    if (!selection) return;
    setSubmitting(true);
    setError(null);
    try {
      const result = await analyzeImage(selection.file);
      router.push(`/results/${result.id}`);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught : new ApiError(0, "Unexpected error while analysing the image."));
      setSubmitting(false);
    }
  }

  if (health.status === "error") {
    return <ErrorState title="The screening API is unavailable" message={health.error.message} onRetry={retryHealth} />;
  }
  if (health.status === "success" && health.data.status !== "ok") {
    return (
      <ErrorState
        title="Models are not loaded"
        message={health.data.detail ?? "The API is running without model artifacts."}
        onRetry={retryHealth}
      />
    );
  }

  const model = health.status === "success" ? health.data.model : null;
  const maxBytes = health.status === "success" ? health.data.max_upload_mb * 1024 * 1024 : 0;

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_22rem]">
      <div className="space-y-4">
        {health.status === "loading" ? (
          <div className="h-44 skeleton rounded-xl border border-line" aria-label="Loading upload settings" />
        ) : (
          <ImageUploader maxBytes={maxBytes} disabled={submitting} onSelect={select} />
        )}
        {submitting && model ? (
          <Fade>
            <AnalysisProgress encoder={model.encoder} />
          </Fade>
        ) : null}
        {error ? (
          <Fade>
            <ErrorState
              title={error.status === 0 ? "Could not reach the API" : "The image could not be analysed"}
              message={error.message}
            />
          </Fade>
        ) : null}
        {model ? (
          <p className="text-sm text-muted">
            Screening against the <span className="font-medium text-ink">{model.classes.map(classLabel).join(" / ")}</span>{" "}
            reference library with {model.encoder}. Samples from other species or modalities should be reported as
            unknown, not identified.
          </p>
        ) : null}
      </div>

      <aside aria-label="Selected sample" className="space-y-3">
        {selection ? (
          <Fade key={selection.previewUrl} className="space-y-3">
            <ImagePreview
              src={selection.previewUrl}
              alt={`Selected sample ${selection.file.name}`}
              details={[
                ["File", selection.file.name],
                ["Size", formatBytes(selection.file.size)],
                ["Type", selection.file.type],
              ]}
            />
            <button
              type="button"
              onClick={submit}
              disabled={submitting}
              className="press inline-flex w-full items-center justify-center gap-2 rounded-full bg-brand-500 px-4 py-2.5 font-semibold text-ink-900 shadow-sm hover:bg-brand-600 disabled:cursor-not-allowed disabled:opacity-60 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
            >
              <MicroscopeIcon aria-hidden="true" weight="bold" className="size-5" />
              {submitting ? "Analyzing…" : "Run screening"}
            </button>
            {!submitting ? (
              <p className="flex items-center justify-center gap-1.5 text-xs text-muted">
                <ArrowCounterClockwiseIcon aria-hidden="true" className="size-3.5" />
                Choose or drop another file to replace this sample
              </p>
            ) : null}
          </Fade>
        ) : (
          <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-line p-6 text-center text-sm text-muted">
            <Mascot size="spot" className="w-24 opacity-90" />
            No sample selected yet. The preview and the screening action appear here.
          </div>
        )}
      </aside>
    </div>
  );
}

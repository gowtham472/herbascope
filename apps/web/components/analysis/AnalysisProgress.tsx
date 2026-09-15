"use client";

import { CircleNotchIcon } from "@phosphor-icons/react/ssr";
import { useEffect, useState } from "react";

const STAGES = [
  "Validate and decode the image",
  "Measure image quality",
  "Encode with DINOv2 ViT-S/14",
  "Classify the embedding",
  "Retrieve nearest reference micrographs",
  "Assess unknown risk",
  "Compare evidence and apply the decision policy",
];

/**
 * Indeterminate progress. The API runs the whole pipeline in one request, so the stages are
 * listed for transparency without pretending to know which one is running.
 */
export function AnalysisProgress() {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const started = Date.now();
    const timer = window.setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 250);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <div role="status" aria-live="polite" className="rounded-xl border border-line bg-surface p-5 shadow-sm">
      <div className="flex items-center gap-3">
        <CircleNotchIcon aria-hidden="true" weight="bold" className="size-6 animate-spin text-brand-600" />
        <div>
          <p className="font-semibold text-ink">Analyzing sample…</p>
          <p className="text-sm text-muted">
            Running the complete screening pipeline locally · <span className="font-mono tabular-nums">{seconds}s</span>
          </p>
        </div>
      </div>
      <ol className="mt-4 grid gap-1.5 text-sm text-muted sm:grid-cols-2">
        {STAGES.map((stage, index) => (
          <li key={stage} className="flex gap-2">
            <span className="font-mono text-xs leading-5 text-brand-600">{index + 1}</span>
            {stage}
          </li>
        ))}
      </ol>
    </div>
  );
}

import type { Metadata } from "next";

import { AnalyzeView } from "@/components/upload/AnalyzeView";

export const metadata: Metadata = { title: "Analyze" };

export default function AnalyzePage() {
  return (
    <div className="mx-auto max-w-6xl px-5 py-12 sm:px-8">
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-brand-700">New screening</p>
      <h1 className="mt-3 text-3xl font-bold tracking-tight text-ink sm:text-4xl">Microscopic screening</h1>
      <p className="mt-3 max-w-3xl text-lg leading-relaxed text-muted">
        Upload one micrograph of dried leaf simplicia. The image is analysed on this machine; the result shows the model
        prediction, the reference evidence behind it, the unknown risk and the screening decision.
      </p>
      <div className="mt-10">
        <AnalyzeView />
      </div>
    </div>
  );
}

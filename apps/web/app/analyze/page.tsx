import type { Metadata } from "next";

import { AnalyzeView } from "@/components/upload/AnalyzeView";

export const metadata: Metadata = { title: "Analyze" };

export default function AnalyzePage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
      <h1 className="text-3xl font-semibold tracking-tight text-ink">Microscopic Screening</h1>
      <p className="mt-2 max-w-3xl text-muted">
        Upload one micrograph of dried leaf simplicia. The image is analysed locally; the result shows the model
        prediction, the reference evidence behind it, the unknown risk and the screening decision.
      </p>
      <div className="mt-8">
        <AnalyzeView />
      </div>
    </div>
  );
}

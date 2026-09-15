"use client";

import { FileTextIcon, PrinterIcon } from "@phosphor-icons/react/ssr";

export function ScreeningSummary({ explanation }: { explanation: string }) {
  return (
    <section aria-labelledby="summary-heading" className="rounded-xl border border-line bg-surface p-5 shadow-sm break-inside-avoid">
      <div className="flex items-center justify-between gap-3">
        <h2 id="summary-heading" className="flex items-center gap-2 font-semibold text-ink">
          <FileTextIcon aria-hidden="true" weight="duotone" className="size-5 text-brand-700" />
          Screening summary
        </h2>
        <button
          type="button"
          onClick={() => window.print()}
          className="press inline-flex items-center gap-2 rounded-lg border border-line px-3 py-1.5 text-sm font-medium text-ink hover:bg-canvas focus-visible:outline-2 focus-visible:outline-brand-600 print:hidden"
        >
          <PrinterIcon aria-hidden="true" className="size-4" />
          Print report
        </button>
      </div>
      <p className="mt-3 text-sm leading-6 text-ink">{explanation}</p>
      <p className="mt-3 text-xs text-muted">
        Generated from the structured results by deterministic templates. No language model was used.
      </p>
    </section>
  );
}

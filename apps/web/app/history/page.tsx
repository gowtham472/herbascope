import type { Metadata } from "next";

import { HistoryView } from "@/components/history/HistoryView";

export const metadata: Metadata = { title: "History" };

export default function HistoryPage() {
  return (
    <div className="mx-auto max-w-4xl px-5 py-12 sm:px-8">
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-brand-700">Stored locally</p>
      <h1 className="mt-3 text-3xl font-bold tracking-tight text-ink sm:text-4xl">Analysis history</h1>
      <p className="mt-3 text-lg text-muted">Every screening run on this workstation, with its decision.</p>
      <div className="mt-10">
        <HistoryView />
      </div>
    </div>
  );
}

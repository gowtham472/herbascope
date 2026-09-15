import type { Metadata } from "next";

import { HistoryView } from "@/components/history/HistoryView";

export const metadata: Metadata = { title: "History" };

export default function HistoryPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-10 sm:px-6">
      <h1 className="text-3xl font-semibold tracking-tight text-ink">Analysis history</h1>
      <p className="mt-2 text-muted">Every screening run on this workstation, with its decision.</p>
      <div className="mt-8">
        <HistoryView />
      </div>
    </div>
  );
}

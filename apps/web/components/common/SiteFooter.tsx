import Link from "next/link";

import { DISCLAIMER } from "@/lib/copy";

import { Wordmark } from "./Wordmark";

export function SiteFooter() {
  return (
    <footer className="border-t border-line bg-ink-900 text-white print:hidden">
      <div className="mx-auto grid max-w-6xl gap-8 px-5 py-12 sm:px-8 md:grid-cols-[1.4fr_1fr]">
        <div>
          <Wordmark className="h-6 w-auto brightness-0 invert" />
          <p className="mt-4 max-w-2xl text-sm leading-relaxed text-white/60">{DISCLAIMER}</p>
        </div>
        <div className="flex flex-col gap-2 text-sm text-white/60 md:items-end">
          <Link href="/analyze" className="hover:text-brand-500">
            Analyze a sample
          </Link>
          <Link href="/history" className="hover:text-brand-500">
            Analysis history
          </Link>
          <p className="mt-2 text-xs text-white/40">Local-first · no cloud inference · deterministic decisions</p>
        </div>
      </div>
    </footer>
  );
}

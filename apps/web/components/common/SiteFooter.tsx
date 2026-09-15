import { DISCLAIMER } from "@/lib/copy";

export function SiteFooter() {
  return (
    <footer className="border-t border-line bg-surface print:hidden">
      <div className="mx-auto flex max-w-6xl flex-col gap-2 px-4 py-6 text-xs text-muted sm:px-6 md:flex-row md:items-center md:justify-between">
        <p className="max-w-3xl">{DISCLAIMER}</p>
        <p className="shrink-0">Local-first · no cloud inference · deterministic decisions</p>
      </div>
    </footer>
  );
}

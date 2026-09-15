import type { Icon } from "@phosphor-icons/react";
import type { ReactNode } from "react";

interface PanelProps {
  title: string;
  icon: Icon;
  description?: string;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}

/** Titled card used for every evidence section so the report reads consistently. */
export function Panel({ title, icon: IconComponent, description, aside, children, className = "" }: PanelProps) {
  const headingId = `panel-${title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
  return (
    <section
      aria-labelledby={headingId}
      className={`rounded-xl border border-line bg-surface p-5 shadow-sm break-inside-avoid ${className}`}
    >
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-brand-50 text-brand-700">
            <IconComponent weight="duotone" aria-hidden="true" className="size-5" />
          </span>
          <div>
            <h2 id={headingId} className="text-base font-semibold text-ink">
              {title}
            </h2>
            {description ? <p className="mt-0.5 text-sm text-muted">{description}</p> : null}
          </div>
        </div>
        {aside}
      </header>
      <div className="mt-4">{children}</div>
    </section>
  );
}

import type { Icon } from "@phosphor-icons/react";
import type { ReactNode } from "react";

interface EmptyStateProps {
  icon: Icon;
  title: string;
  description: string;
  action?: ReactNode;
}

export function EmptyState({ icon: IconComponent, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center rounded-xl border border-dashed border-line bg-surface px-6 py-12 text-center">
      <span className="grid size-12 place-items-center rounded-full bg-brand-50 text-brand-700">
        <IconComponent weight="duotone" aria-hidden="true" className="size-6" />
      </span>
      <h2 className="mt-4 font-semibold text-ink">{title}</h2>
      <p className="mt-1 max-w-md text-sm text-muted">{description}</p>
      {action ? <div className="mt-5">{action}</div> : null}
    </div>
  );
}

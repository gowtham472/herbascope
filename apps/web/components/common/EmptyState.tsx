import type { ReactNode } from "react";

import { Mascot } from "./Mascot";

interface EmptyStateProps {
  title: string;
  description: string;
  action?: ReactNode;
}

export function EmptyState({ title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center rounded-2xl border border-dashed border-line bg-surface px-6 py-12 text-center">
      <Mascot variant="spot" className="w-28" />
      <h2 className="mt-4 text-lg font-bold text-ink">{title}</h2>
      <p className="mt-1 max-w-md text-sm leading-relaxed text-muted">{description}</p>
      {action ? <div className="mt-6">{action}</div> : null}
    </div>
  );
}

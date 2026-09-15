import { ArrowsClockwiseIcon, WarningOctagonIcon } from "@phosphor-icons/react/ssr";
import type { ReactNode } from "react";

interface ErrorStateProps {
  title: string;
  message: string;
  onRetry?: () => void;
  action?: ReactNode;
}

export function ErrorState({ title, message, onRetry, action }: ErrorStateProps) {
  return (
    <div role="alert" className="rounded-xl border border-fail-600/20 bg-fail-50 p-5">
      <div className="flex items-start gap-3">
        <WarningOctagonIcon weight="duotone" aria-hidden="true" className="mt-0.5 size-6 shrink-0 text-fail-600" />
        <div className="min-w-0 flex-1">
          <h2 className="font-semibold text-ink">{title}</h2>
          <p className="mt-1 break-words text-sm text-muted">{message}</p>
          {onRetry || action ? (
            <div className="mt-4 flex flex-wrap gap-2">
              {onRetry ? (
                <button
                  type="button"
                  onClick={onRetry}
                  className="inline-flex items-center gap-2 rounded-lg border border-line bg-surface px-3 py-1.5 text-sm font-medium text-ink hover:bg-canvas focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
                >
                  <ArrowsClockwiseIcon aria-hidden="true" className="size-4" />
                  Try again
                </button>
              ) : null}
              {action}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

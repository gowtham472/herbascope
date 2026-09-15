"use client";

import { PulseIcon } from "@phosphor-icons/react/ssr";

import { useApiResource } from "@/hooks/useApiResource";
import { getHealth } from "@/lib/api";
import { classLabel } from "@/lib/format";

import { ErrorState } from "./ErrorState";
import { StatusBadge } from "./StatusBadge";

/** Live API / model status, shown on the landing page. */
export function SystemStatus() {
  const { state, retry } = useApiResource("health", getHealth);

  if (state.status === "error") {
    return <ErrorState title="API unavailable" message={state.error.message} onRetry={retry} />;
  }

  return (
    <section aria-labelledby="system-status" aria-busy={state.status === "loading"} className="rounded-xl border border-line bg-surface p-5 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <h2 id="system-status" className="flex items-center gap-2 text-sm font-semibold text-ink">
          <PulseIcon aria-hidden="true" weight="bold" className="size-4 text-brand-600" />
          System status
        </h2>
        {state.status === "success" ? (
          <StatusBadge
            tone={state.data.status === "ok" ? "pass" : "review"}
            label={state.data.status === "ok" ? "Models loaded" : "Degraded"}
          />
        ) : (
          <span className="text-xs text-muted">Checking…</span>
        )}
      </div>
      {state.status === "loading" ? (
        <div className="mt-4 space-y-2" aria-hidden="true">
          <div className="h-4 w-3/4 animate-pulse rounded bg-canvas" />
          <div className="h-4 w-1/2 animate-pulse rounded bg-canvas" />
        </div>
      ) : state.data.model ? (
        <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
          <div>
            <dt className="text-xs text-muted">Encoder</dt>
            <dd className="font-medium">{state.data.model.encoder}</dd>
          </div>
          <div>
            <dt className="text-xs text-muted">Reference library</dt>
            <dd className="font-medium">{state.data.reference_count} images</dd>
          </div>
          <div className="col-span-2">
            <dt className="text-xs text-muted">Supported classes</dt>
            <dd className="font-medium">{state.data.model.classes.map(classLabel).join(", ")}</dd>
          </div>
          <div className="col-span-2">
            <dt className="text-xs text-muted">Decision policy</dt>
            <dd className="font-mono text-xs">
              {state.data.model.policy_version} · {state.data.model.unknown_calibration_version} · API {state.data.api_version}
            </dd>
          </div>
        </dl>
      ) : (
        <p className="mt-3 text-sm text-muted">{state.data.detail}</p>
      )}
    </section>
  );
}

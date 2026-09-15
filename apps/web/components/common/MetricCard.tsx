interface MetricCardProps {
  label: string;
  value: string;
  hint?: string;
}

export function MetricCard({ label, value, hint }: MetricCardProps) {
  return (
    <div className="rounded-lg border border-line bg-canvas/60 px-3 py-2.5">
      <dt className="text-xs font-medium text-muted">{label}</dt>
      <dd className="mt-0.5 font-mono text-lg font-semibold tabular-nums text-ink">{value}</dd>
      {hint ? <dd className="mt-0.5 text-xs text-muted">{hint}</dd> : null}
    </div>
  );
}

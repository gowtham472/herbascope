import { InfoIcon } from "@phosphor-icons/react/ssr";

import { Panel } from "@/components/common/Panel";

interface LimitationsCardProps {
  limitations: string[];
  disclaimer: string;
}

export function LimitationsCard({ limitations, disclaimer }: LimitationsCardProps) {
  return (
    <Panel title="Limitations" icon={InfoIcon} description="Scope of what this result can and cannot support.">
      <ul className="list-disc space-y-1.5 pl-5 text-sm text-muted marker:text-brand-600">
        {limitations.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
      <p className="mt-4 rounded-lg bg-canvas px-3 py-2 text-sm font-medium text-ink">{disclaimer}</p>
    </Panel>
  );
}

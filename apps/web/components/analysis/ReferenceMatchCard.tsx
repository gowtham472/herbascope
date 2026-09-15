import Image from "next/image";

import { apiUrl } from "@/lib/api";
import { classLabel, decimal, sentenceCase } from "@/lib/format";
import type { ReferenceMatch } from "@/types";

interface ReferenceMatchCardProps {
  match: ReferenceMatch;
  predictedClass: string;
  minSimilarity: number;
}

export function ReferenceMatchCard({ match, predictedClass, minSimilarity }: ReferenceMatchCardProps) {
  const sameClass = match.class_name === predictedClass;
  const similarityWidth = `${Math.min(Math.max(match.similarity, 0), 1) * 100}%`;
  return (
    <li className="overflow-hidden rounded-lg border border-line bg-surface break-inside-avoid">
      <div className="relative aspect-square bg-canvas">
        {/* Served by the local API; see ImagePreview for why optimisation is bypassed. */}
        <Image
          src={apiUrl(match.image_url)}
          alt={`Reference ${match.reference_id}: ${classLabel(match.class_name)}, ${match.fragment_type}`}
          fill
          unoptimized
          sizes="(min-width: 1024px) 200px, 45vw"
          className="object-cover"
        />
        <span className="absolute left-2 top-2 rounded-md bg-ink/80 px-1.5 py-0.5 font-mono text-xs text-white">#{match.rank}</span>
      </div>
      <div className="space-y-1.5 p-3">
        <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1">
          <span className="text-sm font-semibold capitalize">{classLabel(match.class_name)}</span>
          <span
            className={`whitespace-nowrap rounded-full px-2 py-0.5 text-[11px] font-medium ${sameClass ? "bg-pass-50 text-pass-700" : "bg-fail-50 text-fail-600"}`}
          >
            {sameClass ? "same class" : "other class"}
          </span>
        </div>
        <p className="line-clamp-2 min-h-8 text-xs text-muted" title={match.fragment_type}>
          {sentenceCase(match.fragment_type)}
        </p>
        <div className="flex items-center gap-2">
          <div className="h-1.5 flex-1 rounded-full bg-canvas ring-1 ring-inset ring-line" aria-hidden="true">
            <div
              className={`h-full rounded-full ${match.similarity >= minSimilarity ? "bg-pass-600" : "bg-review-600"}`}
              style={{ width: similarityWidth }}
            />
          </div>
          <span className="font-mono text-xs tabular-nums" aria-label={`Cosine similarity ${decimal(match.similarity)}`}>
            {decimal(match.similarity)}
          </span>
        </div>
        <p className="font-mono text-[11px] text-muted">
          {match.reference_id} · {match.dataset}
        </p>
      </div>
    </li>
  );
}

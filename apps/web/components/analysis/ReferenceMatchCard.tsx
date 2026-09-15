import Image from "next/image";

import { AnimatedMeter } from "@/components/motion/AnimatedMeter";
import { Reveal } from "@/components/motion/Reveal";
import { apiUrl } from "@/lib/api";
import { classLabel, decimal, sentenceCase } from "@/lib/format";
import type { ReferenceMatch } from "@/types";

interface ReferenceMatchCardProps {
  match: ReferenceMatch;
  predictedClass: string;
  minSimilarity: number;
  /** Rank position, used to stagger the reveal so the atlas fills in nearest-first. */
  index?: number;
  /** Opens the side-by-side comparison with the uploaded sample. */
  onCompare: () => void;
}

export function ReferenceMatchCard({ match, predictedClass, minSimilarity, index = 0, onCompare }: ReferenceMatchCardProps) {
  const sameClass = match.class_name === predictedClass;
  return (
    <Reveal
      as="li"
      index={index}
      className="lift zoom-hover group overflow-hidden rounded-lg border border-line bg-surface break-inside-avoid"
    >
      <button
        type="button"
        onClick={onCompare}
        aria-label={`Compare the sample with reference ${match.reference_id}`}
        className="relative block aspect-square w-full cursor-zoom-in overflow-hidden bg-canvas focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-brand-600"
      >
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
        <span className="absolute inset-x-0 bottom-0 translate-y-full bg-ink/70 py-1 text-center text-[11px] font-medium text-white transition-transform duration-200 group-hover:translate-y-0 print:hidden">
          Compare
        </span>
      </button>
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
            <AnimatedMeter
              value={match.similarity}
              className={`h-full rounded-full ${match.similarity >= minSimilarity ? "bg-pass-600" : "bg-review-600"}`}
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
    </Reveal>
  );
}

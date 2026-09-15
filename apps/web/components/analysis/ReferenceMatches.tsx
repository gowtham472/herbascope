import { ImagesIcon } from "@phosphor-icons/react/ssr";

import { MetricCard } from "@/components/common/MetricCard";
import { Panel } from "@/components/common/Panel";
import { LABELS } from "@/lib/copy";
import { classLabel, decimal } from "@/lib/format";
import type { Retrieval } from "@/types";

import { ReferenceMatchCard } from "./ReferenceMatchCard";

interface ReferenceMatchesProps {
  retrieval: Retrieval;
  predictedClass: string;
  minReferenceSimilarity: number;
}

export function ReferenceMatches({ retrieval, predictedClass, minReferenceSimilarity }: ReferenceMatchesProps) {
  return (
    <Panel
      title={LABELS.reference}
      icon={ImagesIcon}
      description="Most similar curated reference micrographs (FAISS inner product on L2-normalised DINOv2 embeddings = cosine similarity)."
    >
      <dl className="grid gap-2 sm:grid-cols-3">
        <MetricCard
          label="Top reference similarity"
          value={decimal(retrieval.top_similarity)}
          hint={`Policy minimum ${decimal(minReferenceSimilarity)}`}
        />
        <MetricCard label="Retrieved class" value={classLabel(retrieval.retrieved_class)} hint="Similarity-weighted vote" />
        <MetricCard label="References shown" value={String(retrieval.matches.length)} hint="Nearest neighbours" />
      </dl>
      <ol className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        {retrieval.matches.map((match, index) => (
          <ReferenceMatchCard
            key={match.reference_id}
            match={match}
            predictedClass={predictedClass}
            minSimilarity={minReferenceSimilarity}
            index={index}
          />
        ))}
      </ol>
    </Panel>
  );
}

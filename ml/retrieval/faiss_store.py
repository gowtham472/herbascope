"""Exact inner-product search over L2-normalised reference embeddings (cosine similarity).

IndexFlatIP is exact, deterministic and needs no training; at tens to thousands of
references a brute-force scan is sub-millisecond, so an approximate index would only add
recall loss. Artifacts: models/indexes/references.faiss + reference_metadata.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass(frozen=True)
class ReferenceRecord:
    reference_id: str
    vector_id: int
    image_id: str
    class_name: str
    fragment_type: str
    dataset: str
    source: str
    image_path: str  # relative to DATA_DIR


@dataclass(frozen=True)
class ReferenceMatch:
    rank: int
    similarity: float
    reference: ReferenceRecord


@dataclass(frozen=True)
class ClassSupport:
    class_name: str
    count: int
    weighted_share: float  # share of summed similarity among the top-k matches


@dataclass(frozen=True)
class RetrievalOutput:
    matches: tuple[ReferenceMatch, ...]
    top_similarity: float
    retrieved_class: str
    class_support: tuple[ClassSupport, ...]


def summarize_matches(matches: tuple[ReferenceMatch, ...]) -> RetrievalOutput:
    """Similarity-weighted class vote over the top-k references.

    The retrieved class is the class with the largest summed similarity; ties go to the
    class of the best single match so the result is deterministic.
    """
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for match in matches:
        weight = max(match.similarity, 0.0)
        totals[match.reference.class_name] += weight
        counts[match.reference.class_name] += 1
    grand_total = sum(totals.values())
    top_class = matches[0].reference.class_name
    support = tuple(
        sorted(
            (
                ClassSupport(name, counts[name], totals[name] / grand_total if grand_total > 0 else 0.0)
                for name in totals
            ),
            key=lambda item: (-item.weighted_share, item.class_name != top_class, item.class_name),
        )
    )
    return RetrievalOutput(
        matches=matches,
        top_similarity=matches[0].similarity,
        retrieved_class=support[0].class_name,
        class_support=support,
    )


class ReferenceIndex:
    INDEX_FILE = "references.faiss"
    METADATA_FILE = "reference_metadata.json"

    def __init__(
        self, index: faiss.IndexFlatIP, records: list[ReferenceRecord], version: str, fingerprint: str
    ) -> None:
        if index.ntotal != len(records):
            raise ValueError("FAISS index size does not match reference metadata")
        self._index = index
        self.records = records
        self.version = version
        self.fingerprint = fingerprint

    @classmethod
    def build(
        cls, embeddings: np.ndarray, records: list[ReferenceRecord], version: str, fingerprint: str
    ) -> ReferenceIndex:
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, records, version, fingerprint)

    def search(self, embeddings: np.ndarray, k: int) -> list[tuple[ReferenceMatch, ...]]:
        """Return the k most similar references for each query row, best first."""
        queries = np.ascontiguousarray(np.atleast_2d(embeddings), dtype=np.float32)
        similarities, ids = self._index.search(queries, min(k, len(self.records)))
        return [
            tuple(
                ReferenceMatch(rank=rank + 1, similarity=float(sim), reference=self.records[int(vid)])
                for rank, (sim, vid) in enumerate(zip(row_sims, row_ids, strict=True))
            )
            for row_sims, row_ids in zip(similarities, ids, strict=True)
        ]

    def save(self, directory: Path, provenance: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / self.INDEX_FILE))
        payload = {
            "version": self.version,
            "embedding_fingerprint": self.fingerprint,
            "metric": "inner_product_on_l2_normalized",
            "index_type": "IndexFlatIP",
            **provenance,
            "references": [record.__dict__ for record in self.records],
        }
        (directory / self.METADATA_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> ReferenceIndex:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        records = [ReferenceRecord(**record) for record in payload["references"]]
        return cls(
            faiss.read_index(str(index_path)), records, payload["version"], payload["embedding_fingerprint"]
        )

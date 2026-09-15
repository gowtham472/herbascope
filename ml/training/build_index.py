"""Select the reference library from training images and build the FAISS index.

Reference selection (per class): drop training images whose quality is DEGRADED, run
k-means with k = min(per_class budget, candidates) on their embeddings, and keep the real
image closest to each centroid. This yields representative, visually diverse, good-quality
references — a curated atlas a reviewer can inspect — rather than the entire training set.

The script refuses to index any image whose id or SHA-256 appears in a validation, test,
held-out, probe or OOD split.

Outputs: models/indexes/{references.faiss, reference_metadata.json}, data/references/mikrobat/
Usage:   python -m ml.training.build_index
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime

import numpy as np
from sklearn.cluster import KMeans

from ml import paths
from ml.pipeline_config import load_pipeline_config
from ml.preprocessing.image_io import load_image_file
from ml.preprocessing.quality import ACCEPTABLE, QualityBounds, assess_quality, measure_quality
from ml.retrieval.faiss_store import ReferenceIndex, ReferenceRecord, summarize_matches
from ml.training import datasets as ds


def _assert_no_leakage(reference_ids: set[str], reference_hashes: set[str]) -> None:
    for name in ds.ALL_SPLITS:
        if name == ds.TRAIN:
            continue
        frame = ds.load_split(name).frame
        overlap = reference_ids & set(frame["image_id"]) or reference_hashes & set(frame["sha256"])
        if overlap:
            raise RuntimeError(f"Reference library would leak {len(overlap)} image(s) from split {name!r}")


def main() -> int:
    config = load_pipeline_config()
    train = ds.load_embeddings(ds.TRAIN)
    frame = train.split.frame.reset_index(drop=True)
    bounds = QualityBounds.load(paths.model_config_path(config.quality.version))
    acceptable = np.array(
        [
            assess_quality(measure_quality(load_image_file(path).image, bounds.tile_grid), bounds).status
            == ACCEPTABLE
            for path in train.split.image_paths()
        ]
    )

    selected: list[int] = []
    selection_stats = {}
    for class_name in sorted(frame["class_name"].unique()):
        candidates = np.flatnonzero((frame["class_name"] == class_name).to_numpy() & acceptable)
        k = min(config.references.per_class, candidates.size)
        kmeans = KMeans(n_clusters=k, random_state=config.seed, n_init=10).fit(train.vectors[candidates])
        chosen = []
        for cluster in range(k):
            members = candidates[kmeans.labels_ == cluster]
            centroid = kmeans.cluster_centers_[cluster]
            chosen.append(int(members[np.argmax(train.vectors[members] @ centroid)]))
        selected.extend(sorted(chosen, key=lambda i: frame.loc[i, "image_id"]))
        selection_stats[class_name] = {
            "training_images": int((frame["class_name"] == class_name).sum()),
            "quality_acceptable_candidates": int(candidates.size),
            "references": k,
            "fragment_types": {
                str(t): int(c)
                for t, c in frame.loc[chosen, "fragment_type"].value_counts().sort_index().items()
            },
        }

    _assert_no_leakage(set(frame.loc[selected, "image_id"]), set(frame.loc[selected, "sha256"]))
    source = f"{config.sources.mikrobat.repository}@{config.sources.mikrobat.commit[:12]}"
    reference_root = paths.REFERENCES_DIR / "mikrobat"
    if reference_root.exists():
        shutil.rmtree(reference_root)
    records = []
    for vector_id, row_index in enumerate(selected):
        row = frame.loc[row_index]
        destination = reference_root / row["class_name"] / f"{row['image_id']}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths.DATA_DIR / row["path"], destination)
        records.append(
            ReferenceRecord(
                reference_id=f"REF{vector_id + 1:04d}",
                vector_id=vector_id,
                image_id=row["image_id"],
                class_name=row["class_name"],
                fragment_type=row["fragment_type"],
                dataset="Mikrobat",
                source=source,
                image_path=destination.relative_to(paths.DATA_DIR).as_posix(),
            )
        )

    index = ReferenceIndex.build(
        train.vectors[selected], records, config.references.version, train.fingerprint
    )
    validation = ds.load_embeddings(ds.VALIDATION)
    retrieved = [
        summarize_matches(m).retrieved_class
        for m in index.search(validation.vectors, config.references.top_k)
    ]
    validation_retrieval_accuracy = float(
        np.mean(np.array(retrieved) == validation.split.frame["class_name"].to_numpy())
    )
    index.save(
        paths.INDEXES_DIR,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "source_split": ds.TRAIN,
            "selection": "per-class k-means medoids over quality-acceptable training embeddings",
            "per_class_budget": config.references.per_class,
            "top_k": config.references.top_k,
            "classes": selection_stats,
            "validation_retrieval_accuracy": validation_retrieval_accuracy,
            "pipeline_config_version": config.version,
        },
    )
    print(
        json.dumps(
            {
                "references": len(records),
                "classes": selection_stats,
                "validation_retrieval_accuracy": validation_retrieval_accuracy,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

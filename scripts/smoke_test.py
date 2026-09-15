"""Run one complete local inference on a real image and print the structured outcome.

Loads every artifact exactly as the API does, analyses the first Mikrobat test image (or a
file passed as an argument), and prints prediction, retrieval, unknown, evidence and decision.
Exits non-zero if artifacts are missing or inference fails.

Usage:
  python -m scripts.smoke_test
  python -m scripts.smoke_test path/to/micrograph.png
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from ml import paths
from ml.inference.screening_pipeline import ArtifactPaths, ScreeningPipeline
from ml.preprocessing.image_io import load_image_file
from ml.training import datasets as ds


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    started = time.perf_counter()
    pipeline = ScreeningPipeline.load(ArtifactPaths.from_manifest(paths.MODELS_DIR))
    loaded = time.perf_counter()
    image_path = Path(argv[0]) if argv else ds.load_split(ds.TEST).image_paths()[0]
    result = pipeline.analyze_image(load_image_file(image_path))
    finished = time.perf_counter()
    evidence, decision = result.evidence, result.decision
    print(
        json.dumps(
            {
                "image": str(image_path),
                "prediction": {
                    "class_name": evidence.prediction.class_name,
                    "confidence": evidence.prediction.confidence,
                },
                "retrieval": {
                    "retrieved_class": evidence.retrieval.retrieved_class,
                    "top_similarity": evidence.retrieval.top_similarity,
                    "matches": [m.reference.reference_id for m in evidence.retrieval.matches],
                },
                "unknown": asdict(evidence.unknown),
                "evidence": {
                    "agreement": evidence.agreement,
                    "strength": evidence.strength,
                    "quality": evidence.quality.status,
                },
                "decision": {
                    "status": decision.status,
                    "reason": decision.reason,
                    "policy_version": decision.policy_version,
                },
                "timing_seconds": {
                    "load_artifacts": round(loaded - started, 2),
                    "inference": round(finished - loaded, 3),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

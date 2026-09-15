"""Calibrate the numeric decision policy on the Mikrobat validation split.

  min_classifier_confidence  smallest confidence c whose selective accuracy (accuracy of
                             validation predictions with confidence >= c) meets the target.
                             If no c meets it, 1.0 is stored so PRELIMINARY_PASS is unreachable.
  min_reference_similarity   the (1 - target retention) quantile of the top-1 reference
                             similarity of correctly classified validation samples.
  max_unknown_risk           the calibrated unknown risk at the known boundary, so a PASS
                             requires the sample to sit inside the known-material distance range.

Output: models/configs/<decision version>.json
Usage:  python -m ml.training.calibrate_decision
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import numpy as np

from ml import paths
from ml.classifiers.classifier import EmbeddingClassifier
from ml.pipeline_config import load_pipeline_config
from ml.retrieval.faiss_store import ReferenceIndex
from ml.training import datasets as ds
from ml.uncertainty.unknown_detector import UnknownCalibration


def select_min_confidence(
    confidences: np.ndarray, correct: np.ndarray, target: float
) -> tuple[float, float, float, bool]:
    """Return (threshold, selective accuracy, coverage, objective met)."""
    for threshold in np.unique(confidences):
        kept = confidences >= threshold
        accuracy = float(correct[kept].mean())
        if accuracy >= target:
            return float(threshold), accuracy, float(kept.mean()), True
    return 1.0, float("nan"), 0.0, False


def main() -> int:
    config = load_pipeline_config()
    classifier = EmbeddingClassifier.load(paths.CLASSIFIERS_DIR)
    index = ReferenceIndex.load(
        paths.INDEXES_DIR / ReferenceIndex.INDEX_FILE, paths.INDEXES_DIR / ReferenceIndex.METADATA_FILE
    )
    calibration = UnknownCalibration.load(paths.CLASSIFIERS_DIR / "calibration.json")
    validation = ds.load_embeddings(ds.VALIDATION)
    truth = validation.split.frame["class_name"].to_numpy()

    predictions = classifier.predict_batch(validation.vectors, len(classifier.classes))
    confidences = np.array([p.confidence for p in predictions])
    correct = np.array([p.class_name for p in predictions]) == truth
    top_similarity = np.array([matches[0].similarity for matches in index.search(validation.vectors, 1)])

    target_accuracy = config.decision.target_selective_accuracy
    retention = config.decision.target_known_retention
    min_confidence, selective_accuracy, coverage, met = select_min_confidence(
        confidences, correct, target_accuracy
    )
    min_similarity = float(np.quantile(top_similarity[correct], 1.0 - retention, method="lower"))
    max_risk = calibration.risk(calibration.known_boundary)

    payload = {
        "version": config.decision.version,
        "min_classifier_confidence": min_confidence,
        "min_reference_similarity": min_similarity,
        "max_unknown_risk": max_risk,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "calibrated_on": {"dataset": "Mikrobat", "split": ds.VALIDATION, "samples": int(truth.size)},
        "depends_on": {
            "classifier_version": classifier.version,
            "index_version": index.version,
            "unknown_calibration_version": calibration.version,
        },
        "objectives": {
            "min_classifier_confidence": {
                "target_selective_accuracy": target_accuracy,
                "achieved_selective_accuracy": selective_accuracy,
                "coverage": coverage,
                "objective_met": met,
            },
            "min_reference_similarity": {
                "target_known_retention": retention,
                "population": "correctly classified validation samples",
            },
            "max_unknown_risk": {
                "definition": "unknown risk at the calibrated known boundary",
                "known_boundary_distance": calibration.known_boundary,
            },
        },
        "rules": [
            "unknown status UNKNOWN -> UNKNOWN",
            "all PASS criteria met (confidence, top similarity, unknown risk, HIGH agreement, ACCEPTABLE quality) "
            "-> PRELIMINARY_PASS",
            "otherwise -> REVIEW_REQUIRED",
        ],
        "pipeline_config_version": config.version,
    }
    target = paths.model_config_path(config.decision.version)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "min_classifier_confidence",
                    "min_reference_similarity",
                    "max_unknown_risk",
                    "objectives",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

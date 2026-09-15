"""Calibrate the unknown detector on genuine Mikrobat validation samples vs DIMPSAR far-OOD.

See ml/uncertainty/unknown_detector.py for the signal, objective and status rules.

Output: models/classifiers/calibration.json
Usage:  python -m ml.training.calibrate_unknown
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import numpy as np

from ml import paths
from ml.pipeline_config import load_pipeline_config
from ml.retrieval.faiss_store import ReferenceIndex
from ml.training import datasets as ds
from ml.uncertainty.unknown_detector import fit_risk_model, reference_distance, select_distance_threshold


def split_distances(index: ReferenceIndex, name: str, k: int) -> np.ndarray:
    embeddings = ds.load_embeddings(name)
    if embeddings.fingerprint != index.fingerprint:
        raise ValueError(f"{name} embeddings do not match the reference index embedding space")
    return np.array(
        [
            reference_distance(np.array([m.similarity for m in matches]))
            for matches in index.search(embeddings.vectors, k)
        ]
    )


def _describe(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        **{f"p{p}": float(np.percentile(values, p)) for p in (5, 25, 50, 75, 95)},
    }


def main() -> int:
    config = load_pipeline_config()
    index = ReferenceIndex.load(
        paths.INDEXES_DIR / ReferenceIndex.INDEX_FILE, paths.INDEXES_DIR / ReferenceIndex.METADATA_FILE
    )
    k = config.references.top_k
    known = split_distances(index, ds.VALIDATION, k)
    ood = split_distances(index, ds.OOD_CALIBRATION, k)
    target = config.unknown.target_known_acceptance
    selection = select_distance_threshold(known, ood, target)
    intercept, slope = fit_risk_model(known, ood, config.seed)

    payload = {
        "version": config.unknown.version,
        "method": "reference_similarity",
        "signal": f"distance = 1 - mean cosine similarity to the {k} nearest references",
        "k": k,
        "distance_threshold": selection.threshold,
        "known_boundary": selection.known_boundary,
        "risk_intercept": intercept,
        "risk_slope": slope,
        "risk_model": "logistic P(OOD | distance), balanced classes, standardised feature",
        "embedding_fingerprint": index.fingerprint,
        "index_version": index.version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "calibrated_on": {
            "known_dataset": "Mikrobat",
            "known_split": ds.VALIDATION,
            "ood_dataset": "DIMPSAR",
            "ood_split": ds.OOD_CALIBRATION,
        },
        "known_samples": int(known.size),
        "ood_samples": int(ood.size),
        "objective": "minimize OOD false acceptance while retaining acceptable known-material acceptance",
        "objective_parameters": {
            "target_known_acceptance": target,
            "tie_break": "midpoint of the threshold interval with the optimal false acceptance",
        },
        "calibration_rates": {
            "known_acceptance": selection.known_acceptance,
            "ood_false_acceptance": selection.ood_false_acceptance,
        },
        "distance_distributions": {"known_validation": _describe(known), "ood_calibration": _describe(ood)},
        "limitation": ds.OOD_CALIBRATION_LIMITATION,
        "pipeline_config_version": config.version,
    }
    target_path = paths.CLASSIFIERS_DIR / "calibration.json"
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "distance_threshold",
                    "known_boundary",
                    "calibration_rates",
                    "distance_distributions",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

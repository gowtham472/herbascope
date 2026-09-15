"""Reference-space unknown detector.

Signal
  distance = 1 - mean cosine similarity to the k nearest references. Averaging over k
  neighbours measures how well the sample sits inside the reference distribution, which is
  less brittle than a single best match (that stays in the retrieval evidence).

Calibration (ml/training/calibrate_unknown.py), on genuine Mikrobat validation samples and
a class-disjoint DIMPSAR far-OOD calibration subset:
  * distance_threshold: minimise OOD false acceptance subject to known acceptance >= target,
    placed at the midpoint of the widest threshold interval achieving that optimum.
  * known_boundary: the target-quantile of known validation distances. Samples between
    the boundary and the threshold are atypical for known material but not OOD-like.
  * risk: 1-D logistic model P(OOD | distance) fitted on the same data (balanced classes).

Status: distance > threshold -> UNKNOWN; distance > known_boundary -> UNCERTAIN; else KNOWN.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

KNOWN = "KNOWN"
UNCERTAIN = "UNCERTAIN"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class UnknownCalibration:
    version: str
    method: str
    k: int
    distance_threshold: float
    known_boundary: float
    risk_intercept: float
    risk_slope: float
    embedding_fingerprint: str
    index_version: str
    limitation: str  # scientific scope of what the calibration data can support

    def risk(self, distance: float) -> float:
        logit = self.risk_intercept + self.risk_slope * distance
        return 1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1.0 + math.exp(logit))

    @classmethod
    def load(cls, path: Path) -> UnknownCalibration:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{field: payload[field] for field in cls.__dataclass_fields__})


@dataclass(frozen=True)
class UnknownAssessment:
    status: str
    risk: float
    distance: float
    threshold: float
    known_boundary: float
    calibration_version: str


def reference_distance(similarities: np.ndarray) -> float:
    return float(1.0 - np.mean(similarities))


def assess_unknown(similarities: np.ndarray, calibration: UnknownCalibration) -> UnknownAssessment:
    distance = reference_distance(similarities[: calibration.k])
    if distance > calibration.distance_threshold:
        status = UNKNOWN
    elif distance > calibration.known_boundary:
        status = UNCERTAIN
    else:
        status = KNOWN
    return UnknownAssessment(
        status=status,
        risk=calibration.risk(distance),
        distance=distance,
        threshold=calibration.distance_threshold,
        known_boundary=calibration.known_boundary,
        calibration_version=calibration.version,
    )


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    known_boundary: float
    known_acceptance: float
    ood_false_acceptance: float


def select_distance_threshold(
    known: np.ndarray, ood: np.ndarray, target_known_acceptance: float
) -> ThresholdSelection:
    """Objective: minimise OOD false acceptance while keeping known acceptance >= target.

    Acceptance means distance <= threshold. Both rates are non-decreasing in the threshold,
    so the optimum is the smallest threshold meeting the known-acceptance target (this is
    also the known boundary). The false-acceptance rate stays constant until the next OOD
    distance; the midpoint of that interval maximises the margin on both sides.
    """
    known = np.sort(np.asarray(known, dtype=np.float64))
    ood = np.sort(np.asarray(ood, dtype=np.float64))
    needed = math.ceil(target_known_acceptance * known.size)
    boundary = float(known[needed - 1])
    next_ood = ood[ood > boundary]
    threshold = (boundary + float(next_ood[0])) / 2.0 if next_ood.size else float(known[-1])
    return ThresholdSelection(
        threshold=threshold,
        known_boundary=boundary,
        known_acceptance=float(np.mean(known <= threshold)),
        ood_false_acceptance=float(np.mean(ood <= threshold)),
    )


def fit_risk_model(known: np.ndarray, ood: np.ndarray, seed: int) -> tuple[float, float]:
    """Fit P(OOD | distance); returns (intercept, slope) in raw distance units.

    The feature is standardised before the L2-regularised fit so the penalty does not
    depend on the numeric scale of cosine distances, then mapped back to raw units.
    """
    distances = np.concatenate([known, ood]).astype(np.float64)
    labels = np.concatenate([np.zeros(known.size), np.ones(ood.size)])
    mean, std = float(distances.mean()), float(distances.std())
    standardized = ((distances - mean) / std)[:, None]
    model = LogisticRegression(class_weight="balanced", random_state=seed).fit(standardized, labels)
    slope = float(model.coef_[0, 0]) / std
    intercept = float(model.intercept_[0]) - slope * mean
    return intercept, slope

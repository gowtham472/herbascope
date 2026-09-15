"""Deterministic decision engine (policy file: models/configs/<decision version>.json).

Rules, evaluated in order:
  1. Unknown status UNKNOWN                     -> UNKNOWN (insufficient reference evidence)
  2. Every PASS criterion satisfied             -> PRELIMINARY_PASS
       classifier confidence   >= min_classifier_confidence
       top reference similarity >= min_reference_similarity
       unknown risk            <= max_unknown_risk
       evidence agreement      == HIGH
       image quality           == ACCEPTABLE
  3. Otherwise                                  -> REVIEW_REQUIRED (reason lists failed criteria)

The numeric policy values are calibrated by ml/training/calibrate_decision.py. No model,
LLM or explanation layer can override the result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ml.evidence.evidence_engine import HIGH, Evidence
from ml.preprocessing.quality import ACCEPTABLE
from ml.uncertainty.unknown_detector import UNKNOWN

PRELIMINARY_PASS = "PRELIMINARY_PASS"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
UNKNOWN_DECISION = "UNKNOWN"


@dataclass(frozen=True)
class DecisionPolicy:
    version: str
    min_classifier_confidence: float
    min_reference_similarity: float
    max_unknown_risk: float

    @classmethod
    def load(cls, path: Path) -> DecisionPolicy:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{field: payload[field] for field in cls.__dataclass_fields__})


@dataclass(frozen=True)
class DecisionCheck:
    name: str
    label: str
    passed: bool
    observed: str
    required: str


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str
    policy_version: str
    checks: tuple[DecisionCheck, ...]


def _pass_checks(evidence: Evidence, policy: DecisionPolicy) -> tuple[DecisionCheck, ...]:
    return (
        DecisionCheck(
            "classifier_confidence",
            "Classifier confidence",
            evidence.prediction.confidence >= policy.min_classifier_confidence,
            f"{evidence.prediction.confidence:.3f}",
            f">= {policy.min_classifier_confidence:.3f}",
        ),
        DecisionCheck(
            "reference_similarity",
            "Top reference similarity",
            evidence.retrieval.top_similarity >= policy.min_reference_similarity,
            f"{evidence.retrieval.top_similarity:.3f}",
            f">= {policy.min_reference_similarity:.3f}",
        ),
        DecisionCheck(
            "unknown_risk",
            "Unknown risk",
            evidence.unknown.risk <= policy.max_unknown_risk,
            f"{evidence.unknown.risk:.3f}",
            f"<= {policy.max_unknown_risk:.3f}",
        ),
        DecisionCheck(
            "evidence_agreement",
            "Evidence agreement",
            evidence.agreement == HIGH,
            evidence.agreement,
            HIGH,
        ),
        DecisionCheck(
            "image_quality",
            "Image quality",
            evidence.quality.status == ACCEPTABLE,
            evidence.quality.status,
            ACCEPTABLE,
        ),
    )


def decide(evidence: Evidence, policy: DecisionPolicy) -> Decision:
    checks = _pass_checks(evidence, policy)
    if evidence.unknown.status == UNKNOWN:
        reason = (
            f"Reference distance {evidence.unknown.distance:.3f} exceeds the calibrated unknown "
            f"threshold {evidence.unknown.threshold:.3f}: the sample is outside the supported "
            "reference distribution, so no screening conclusion is drawn."
        )
        return Decision(UNKNOWN_DECISION, reason, policy.version, checks)
    failed = [check for check in checks if not check.passed]
    if not failed:
        reason = (
            f"All preliminary-pass criteria are met for {evidence.prediction.class_name}: "
            "calibrated classifier confidence, reference similarity and unknown risk are within "
            "policy, reference retrieval agrees, and image quality is acceptable."
        )
        return Decision(PRELIMINARY_PASS, reason, policy.version, checks)
    details = "; ".join(
        f"{check.label.lower()} {check.observed} (requires {check.required})" for check in failed
    )
    reason = f"Expert review is required because {len(failed)} criterion(s) were not met: {details}."
    return Decision(REVIEW_REQUIRED, reason, policy.version, checks)

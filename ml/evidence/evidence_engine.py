"""Evidence engine.

The classifier and the reference retrieval are complementary analyses of the same DINOv2
representation — not statistically independent evidence. This module compares them and
summarises how well the available visual evidence supports the model prediction.

Agreement (structural definitions, no tuned numbers):
  HIGH    predicted class == retrieved class and every top-k reference is that class
  MEDIUM  predicted class == retrieved class but the top-k references are mixed
  LOW     predicted class != retrieved class

Evidence strength is a descriptive index in [0, 1]: the geometric mean of classifier
confidence, reference support for the predicted class, (1 - unknown risk) and the fraction
of quality checks passed. A geometric mean is low whenever any single component is low.
It is not a probability, and the decision engine does not use it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.classifiers.classifier import ClassifierOutput
from ml.preprocessing.quality import ACCEPTABLE, QualityAssessment
from ml.retrieval.faiss_store import RetrievalOutput
from ml.uncertainty.unknown_detector import KNOWN, UNKNOWN, UnknownAssessment

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"


@dataclass(frozen=True)
class StrengthComponents:
    classifier_confidence: float
    reference_support: float
    unknown_margin: float
    quality_pass_fraction: float


@dataclass(frozen=True)
class Evidence:
    prediction: ClassifierOutput
    retrieval: RetrievalOutput
    unknown: UnknownAssessment
    quality: QualityAssessment
    agreement: str
    strength: float
    components: StrengthComponents
    summary: str


def _agreement(prediction: ClassifierOutput, retrieval: RetrievalOutput) -> str:
    if prediction.class_name != retrieval.retrieved_class:
        return LOW
    if all(match.reference.class_name == prediction.class_name for match in retrieval.matches):
        return HIGH
    return MEDIUM


def _reference_support(prediction: ClassifierOutput, retrieval: RetrievalOutput) -> float:
    return next(
        (item.weighted_share for item in retrieval.class_support if item.class_name == prediction.class_name),
        0.0,
    )


def _summary(
    prediction: ClassifierOutput,
    retrieval: RetrievalOutput,
    unknown: UnknownAssessment,
    quality: QualityAssessment,
    agreement: str,
) -> str:
    k = len(retrieval.matches)
    same = sum(match.reference.class_name == prediction.class_name for match in retrieval.matches)
    parts = [
        f"The classifier predicts {prediction.class_name} with {prediction.confidence:.0%} confidence.",
        f"{same} of the {k} most similar reference images are {prediction.class_name}; "
        f"the best reference match has cosine similarity {retrieval.top_similarity:.3f}.",
    ]
    if agreement == LOW:
        parts.append(
            f"Reference retrieval favours {retrieval.retrieved_class}, so the two analyses of the "
            "same visual representation disagree."
        )
    else:
        parts.append(
            f"Classifier and reference retrieval agree ({agreement.lower()} agreement); they are "
            "complementary analyses of the same visual representation, not independent tests."
        )
    if unknown.status == UNKNOWN:
        parts.append("The sample lies outside the calibrated reference distribution.")
    elif unknown.status != KNOWN:
        parts.append("The sample is atypical compared with validated reference material.")
    else:
        parts.append("The sample lies within the calibrated reference distribution.")
    if quality.status != ACCEPTABLE:
        failed = ", ".join(check.label.lower() for check in quality.checks if not check.passed)
        parts.append(f"Image quality is degraded ({failed}).")
    return " ".join(parts)


def build_evidence(
    prediction: ClassifierOutput,
    retrieval: RetrievalOutput,
    unknown: UnknownAssessment,
    quality: QualityAssessment,
) -> Evidence:
    agreement = _agreement(prediction, retrieval)
    components = StrengthComponents(
        classifier_confidence=prediction.confidence,
        reference_support=_reference_support(prediction, retrieval),
        unknown_margin=1.0 - unknown.risk,
        quality_pass_fraction=quality.passed_fraction,
    )
    values = np.array(
        [
            components.classifier_confidence,
            components.reference_support,
            components.unknown_margin,
            components.quality_pass_fraction,
        ],
        dtype=np.float64,
    )
    strength = 0.0 if np.any(values <= 0) else float(np.exp(np.mean(np.log(values))))
    return Evidence(
        prediction=prediction,
        retrieval=retrieval,
        unknown=unknown,
        quality=quality,
        agreement=agreement,
        strength=strength,
        components=components,
        summary=_summary(prediction, retrieval, unknown, quality, agreement),
    )

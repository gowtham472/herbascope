"""Small constructors for evidence-layer value objects used by engine unit tests."""

from __future__ import annotations

from ml.classifiers.classifier import ClassifierOutput, ClassProbability
from ml.preprocessing.quality import ACCEPTABLE, DEGRADED, QualityAssessment, QualityCheck
from ml.retrieval.faiss_store import ReferenceMatch, ReferenceRecord, RetrievalOutput, summarize_matches
from ml.uncertainty.unknown_detector import UnknownAssessment


def prediction(class_name: str = "sirih", confidence: float = 0.95) -> ClassifierOutput:
    other = "sirih_merah" if class_name == "sirih" else "sirih"
    return ClassifierOutput(
        class_name,
        confidence,
        (ClassProbability(class_name, confidence), ClassProbability(other, 1 - confidence)),
    )


def retrieval(classes: list[str], similarities: list[float] | None = None) -> RetrievalOutput:
    similarities = similarities or [0.9 - 0.02 * i for i in range(len(classes))]
    matches = tuple(
        ReferenceMatch(
            rank + 1,
            similarity,
            ReferenceRecord(
                f"REF{rank:04d}", rank, f"img-{rank}", name, "fragment", "Mikrobat", "src", "p.png"
            ),
        )
        for rank, (name, similarity) in enumerate(zip(classes, similarities, strict=True))
    )
    return summarize_matches(matches)


def unknown(status: str = "KNOWN", risk: float = 0.02, distance: float = 0.2) -> UnknownAssessment:
    return UnknownAssessment(
        status, risk, distance, threshold=0.56, known_boundary=0.43, calibration_version="u"
    )


def quality(acceptable: bool = True) -> QualityAssessment:
    checks = (
        QualityCheck("resolution", "Resolution (short side, px)", True, 300.0, 224.0, ">="),
        QualityCheck("focus", "Focus (Laplacian variance)", acceptable, 40.0, 11.0, ">="),
    )
    return QualityAssessment(ACCEPTABLE if acceptable else DEGRADED, checks, "q")

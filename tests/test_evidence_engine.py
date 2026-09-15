import math

import pytest

from ml.evidence.evidence_engine import HIGH, LOW, MEDIUM, build_evidence
from tests import builders as b


def test_high_agreement_when_all_references_match_prediction():
    evidence = build_evidence(b.prediction("sirih"), b.retrieval(["sirih"] * 5), b.unknown(), b.quality())
    assert evidence.agreement == HIGH
    assert evidence.components.reference_support == pytest.approx(1.0)


def test_medium_agreement_when_majority_matches_but_mixed():
    evidence = build_evidence(
        b.prediction("sirih"),
        b.retrieval(["sirih", "sirih", "sirih_merah", "sirih", "sirih"]),
        b.unknown(),
        b.quality(),
    )
    assert evidence.agreement == MEDIUM
    assert 0.5 < evidence.components.reference_support < 1.0


def test_low_agreement_when_retrieval_favours_another_class():
    evidence = build_evidence(
        b.prediction("sirih"), b.retrieval(["sirih_merah"] * 4 + ["sirih"]), b.unknown(), b.quality()
    )
    assert evidence.agreement == LOW
    assert "disagree" in evidence.summary


def test_strength_is_geometric_mean_of_components():
    evidence = build_evidence(
        b.prediction(confidence=0.9), b.retrieval(["sirih"] * 5), b.unknown(risk=0.1), b.quality(False)
    )
    expected = math.exp((math.log(0.9) + math.log(1.0) + math.log(0.9) + math.log(0.5)) / 4)
    assert evidence.strength == pytest.approx(expected)


def test_strength_is_zero_when_any_component_is_zero():
    evidence = build_evidence(
        b.prediction("sirih"), b.retrieval(["sirih_merah"] * 5), b.unknown(), b.quality()
    )
    assert evidence.components.reference_support == 0.0
    assert evidence.strength == 0.0


def test_summary_never_claims_independent_evidence_and_reports_status():
    unknown_evidence = build_evidence(
        b.prediction(), b.retrieval(["sirih"] * 5), b.unknown("UNKNOWN", 0.9, 0.8), b.quality(False)
    )
    agree_evidence = build_evidence(b.prediction(), b.retrieval(["sirih"] * 5), b.unknown(), b.quality())
    assert "outside the calibrated reference distribution" in unknown_evidence.summary
    assert "Image quality is degraded (focus (laplacian variance))" in unknown_evidence.summary
    assert "not independent tests" in agree_evidence.summary
    assert "same visual representation" in agree_evidence.summary

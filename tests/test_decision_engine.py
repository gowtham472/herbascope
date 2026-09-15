import json

import pytest

from ml.decision.decision_engine import (
    PRELIMINARY_PASS,
    REVIEW_REQUIRED,
    UNKNOWN_DECISION,
    DecisionPolicy,
    decide,
)
from ml.evidence.evidence_engine import build_evidence
from tests import builders as b

POLICY = DecisionPolicy(
    "decision-test", min_classifier_confidence=0.8, min_reference_similarity=0.65, max_unknown_risk=0.1
)


def _evidence(**kwargs):
    return build_evidence(
        kwargs.get("prediction", b.prediction()),
        kwargs.get("retrieval", b.retrieval(["sirih"] * 5)),
        kwargs.get("unknown", b.unknown()),
        kwargs.get("quality", b.quality()),
    )


def test_preliminary_pass_when_every_criterion_is_met():
    decision = decide(_evidence(), POLICY)
    assert decision.status == PRELIMINARY_PASS
    assert decision.policy_version == "decision-test"
    assert all(check.passed for check in decision.checks)


def test_unknown_status_takes_precedence_over_everything():
    decision = decide(_evidence(unknown=b.unknown("UNKNOWN", risk=0.95, distance=0.8)), POLICY)
    assert decision.status == UNKNOWN_DECISION
    assert "exceeds the calibrated unknown threshold" in decision.reason


@pytest.mark.parametrize(
    ("override", "failed_check"),
    [
        ({"prediction": b.prediction(confidence=0.7)}, "classifier_confidence"),
        ({"retrieval": b.retrieval(["sirih"] * 5, [0.6, 0.6, 0.6, 0.6, 0.6])}, "reference_similarity"),
        ({"unknown": b.unknown("UNCERTAIN", risk=0.3, distance=0.5)}, "unknown_risk"),
        (
            {"retrieval": b.retrieval(["sirih", "sirih", "sirih_merah", "sirih", "sirih"])},
            "evidence_agreement",
        ),
        ({"quality": b.quality(False)}, "image_quality"),
    ],
)
def test_single_failed_criterion_requires_review(override, failed_check):
    decision = decide(_evidence(**override), POLICY)
    assert decision.status == REVIEW_REQUIRED
    assert [check.name for check in decision.checks if not check.passed] == [failed_check]
    assert "1 criterion(s)" in decision.reason


def test_review_reason_lists_every_failed_criterion():
    evidence = _evidence(prediction=b.prediction(confidence=0.6), quality=b.quality(False))
    decision = decide(evidence, POLICY)
    assert decision.status == REVIEW_REQUIRED
    assert "2 criterion(s)" in decision.reason
    assert "classifier confidence 0.600" in decision.reason and "image quality DEGRADED" in decision.reason


def test_decision_is_deterministic():
    evidence = _evidence(prediction=b.prediction(confidence=0.81))
    assert decide(evidence, POLICY) == decide(evidence, POLICY)


def test_policy_load_reads_only_policy_fields(tmp_path):
    path = tmp_path / "decision-v1.json"
    path.write_text(
        json.dumps(
            {
                "version": "decision-test",
                "min_classifier_confidence": 0.8,
                "min_reference_similarity": 0.65,
                "max_unknown_risk": 0.1,
                "objectives": {},
            }
        )
    )
    assert DecisionPolicy.load(path) == POLICY

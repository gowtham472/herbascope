import json
from dataclasses import asdict

import numpy as np
import pytest

from ml.uncertainty.unknown_detector import (
    KNOWN,
    UNCERTAIN,
    UNKNOWN,
    UnknownCalibration,
    assess_unknown,
    fit_risk_model,
    reference_distance,
    select_distance_threshold,
)


def _calibration(**overrides) -> UnknownCalibration:
    values = dict(
        version="unknown-test",
        method="reference_similarity",
        k=3,
        distance_threshold=0.5,
        known_boundary=0.3,
        risk_intercept=-10.0,
        risk_slope=25.0,
        embedding_fingerprint="fp",
        index_version="index-test",
        limitation="test",
    )
    return UnknownCalibration(**{**values, **overrides})


def test_reference_distance_is_one_minus_mean_similarity():
    assert reference_distance(np.array([0.9, 0.7, 0.5])) == pytest.approx(0.3)


def test_threshold_on_separable_distributions_sits_mid_gap():
    known = np.linspace(0.1, 0.3, 20)
    ood = np.linspace(0.7, 0.9, 20)
    selection = select_distance_threshold(known, ood, target_known_acceptance=0.95)
    assert selection.known_boundary == pytest.approx(known[18])  # 19th of 20 accepted = 95 %
    assert selection.threshold == pytest.approx((known[18] + 0.7) / 2)
    assert selection.known_acceptance == 1.0
    assert selection.ood_false_acceptance == 0.0


def test_threshold_on_overlapping_distributions_meets_known_target():
    rng = np.random.default_rng(0)
    known = rng.normal(0.3, 0.1, 200)
    ood = rng.normal(0.45, 0.1, 200)
    selection = select_distance_threshold(known, ood, target_known_acceptance=0.9)
    assert selection.known_acceptance >= 0.9
    assert 0 < selection.ood_false_acceptance < 1
    # No lower threshold could still meet the known-acceptance target.
    assert np.mean(known <= selection.known_boundary - 1e-9) < 0.9


def test_risk_model_is_monotonic_in_distance():
    rng = np.random.default_rng(1)
    intercept, slope = fit_risk_model(rng.normal(0.2, 0.05, 50), rng.normal(0.8, 0.05, 50), seed=0)
    calibration = _calibration(risk_intercept=intercept, risk_slope=slope)
    risks = [calibration.risk(d) for d in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert risks == sorted(risks)
    assert risks[0] < 0.1 and risks[-1] > 0.9


def test_risk_is_numerically_stable_for_extreme_logits():
    calibration = _calibration(risk_intercept=0.0, risk_slope=10_000.0)
    assert calibration.risk(1.0) == pytest.approx(1.0)
    assert calibration.risk(-1.0) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("similarities", "status"),
    [([0.9, 0.8, 0.8], KNOWN), ([0.6, 0.6, 0.6], UNCERTAIN), ([0.3, 0.2, 0.1], UNKNOWN)],
)
def test_assess_unknown_status_bands(similarities, status):
    assessment = assess_unknown(np.array(similarities), _calibration())
    assert assessment.status == status
    assert assessment.threshold == 0.5 and assessment.known_boundary == 0.3


def test_assess_unknown_uses_only_k_neighbours():
    assessment = assess_unknown(np.array([0.9, 0.9, 0.9, 0.0, 0.0]), _calibration(k=3))
    assert assessment.distance == pytest.approx(0.1)


def test_calibration_load_ignores_report_fields(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({**asdict(_calibration()), "objective": "report only"}))
    assert UnknownCalibration.load(path) == _calibration()

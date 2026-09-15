import json

import numpy as np
import pytest

from ml.classifiers.classifier import EmbeddingClassifier, fit_logistic_regression, select_regularization


def _clusters(seed: int, per_class: int = 20, dim: int = 16) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.eye(dim)[:3] * 3
    x = np.concatenate([center + rng.normal(0, 0.5, (per_class, dim)) for center in centers])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float32), np.repeat(np.arange(3), per_class)


def test_predict_batch_returns_sorted_probabilities_that_sum_to_one():
    x, y = _clusters(0)
    model = fit_logistic_regression(x, y, c=10.0, class_weight="balanced", max_iter=1000, seed=0)
    classifier = EmbeddingClassifier(model, ["a", "b", "c"], "v", "fp")
    outputs = classifier.predict_batch(x[:5], top_k=3)
    for output in outputs:
        probabilities = [item.probability for item in output.top_k]
        assert probabilities == sorted(probabilities, reverse=True)
        assert sum(probabilities) == pytest.approx(1.0)
        assert output.class_name == output.top_k[0].class_name
        assert output.confidence == output.top_k[0].probability
    assert all(output.class_name == "a" for output in outputs)


def test_top_k_is_capped():
    x, y = _clusters(1)
    model = fit_logistic_regression(x, y, c=1.0, class_weight="balanced", max_iter=1000, seed=0)
    assert (
        len(EmbeddingClassifier(model, ["a", "b", "c"], "v", "fp").predict_batch(x[:1], top_k=2)[0].top_k)
        == 2
    )


def test_select_regularization_uses_validation_log_loss():
    train, validation = _clusters(2), _clusters(3)
    c_value, scores = select_regularization(train, validation, [0.01, 1.0, 100.0], "balanced", 1000, 0)
    assert c_value in {0.01, 1.0, 100.0}
    assert min(scores, key=lambda row: row["validation_log_loss"])["C"] == c_value


def test_save_and_load_round_trip(tmp_path):
    x, y = _clusters(4)
    model = fit_logistic_regression(x, y, c=1.0, class_weight="balanced", max_iter=1000, seed=0)
    classifier = EmbeddingClassifier(model, ["a", "b", "c"], "classifier-v9", "fp-1")
    classifier.save(tmp_path, {"note": "test"})
    loaded = EmbeddingClassifier.load(tmp_path)
    assert (loaded.version, loaded.fingerprint, loaded.classes) == ("classifier-v9", "fp-1", ["a", "b", "c"])
    assert json.loads((tmp_path / "label_encoder.json").read_text())["classes"] == ["a", "b", "c"]
    assert json.loads((tmp_path / "training_config.json").read_text())["note"] == "test"
    before = [o.confidence for o in classifier.predict_batch(x, 3)]
    after = [o.confidence for o in loaded.predict_batch(x, 3)]
    assert before == pytest.approx(after)


def test_rejects_non_contiguous_class_indices():
    x, y = _clusters(5)
    model = fit_logistic_regression(x, y * 2, c=1.0, class_weight="balanced", max_iter=1000, seed=0)
    with pytest.raises(ValueError, match="contiguous"):
        EmbeddingClassifier(model, ["a", "b", "c"], "v", "fp")

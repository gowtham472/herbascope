import numpy as np
import pytest

from ml import paths
from ml.experiments.run_experiments import Candidate, cross_validate, improves, load_experiments_config
from ml.pipeline_config import load_pipeline_config

LADDERS = sorted((paths.REPO_ROOT / "ml" / "configs").glob("experiments-v*.json"))


@pytest.mark.parametrize("ladder", LADDERS, ids=lambda path: path.stem)
def test_registered_ladders_are_valid_and_ordered(ladder):
    experiments = load_experiments_config(ladder)
    assert experiments.version == ladder.stem
    ids = [experiments.baseline.id] + [rung.id for rung in experiments.ladder]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    config = experiments.baseline.config.model_dump()
    assert config["backbone"] in experiments.backbones
    for rung in experiments.ladder:
        config = Candidate(**{**config, **rung.change}).model_dump()
        assert config["backbone"] in experiments.backbones


@pytest.mark.parametrize(
    ("previous", "adopted", "following"),
    [
        ("experiments-v1", ("E1", "E3"), "experiments-v2"),
        ("experiments-v2", ("F1", "F2"), "experiments-v3"),
        ("experiments-v3", ("G2",), "experiments-v4"),
    ],
)
def test_each_phase_starts_from_the_previous_selection(previous, adopted, following):
    configs = paths.REPO_ROOT / "ml" / "configs"
    earlier = load_experiments_config(configs / f"{previous}.json")
    later = load_experiments_config(configs / f"{following}.json")
    selected = earlier.baseline.config.model_dump()
    for rung_id in adopted:
        rung = next(r for r in earlier.ladder if r.id == rung_id)
        selected = {**selected, **rung.change}
    assert later.baseline.config.model_dump() == selected


def test_ladder_change_must_stay_valid():
    baseline = load_experiments_config(LADDERS[0]).baseline.config.model_dump()
    with pytest.raises(ValueError):
        Candidate(**{**baseline, "input_size": 300})


def _result(accuracy: float, loss: float, ms: float = 100.0) -> dict:
    return {"cv_accuracy": accuracy, "cv_log_loss": loss, "encode_ms_per_image": ms}


def test_adoption_rule_prefers_accuracy_then_log_loss():
    best = _result(0.90, 0.30)
    assert improves(_result(0.91, 0.50), best, 3000)
    assert not improves(_result(0.89, 0.10), best, 3000)
    assert improves(_result(0.90, 0.29), best, 3000)
    assert not improves(_result(0.90, 0.30), best, 3000)


def test_adoption_rule_rejects_rungs_over_the_latency_budget():
    assert not improves(_result(0.99, 0.01, ms=4000), _result(0.90, 0.30), 3000)


def _separable(views: int, per_class: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    labels = np.repeat([0, 1], per_class)
    base = centers[labels] + rng.normal(0, 0.3, (labels.size, 3))
    view_vectors = np.stack([base + rng.normal(0, 0.05, base.shape) for _ in range(views)])
    view_vectors /= np.linalg.norm(view_vectors, axis=-1, keepdims=True)
    groups = np.array([f"g{i // 2}" for i in range(labels.size)])  # near-duplicate pairs
    return view_vectors.astype(np.float32), labels, groups


@pytest.mark.parametrize("train_on_views", [False, True])
def test_cross_validation_on_separable_data(train_on_views):
    view_vectors, labels, groups = _separable(views=4)
    cv = cross_validate(view_vectors, labels, groups, train_on_views, load_pipeline_config(), folds=4)
    assert cv.accuracy >= 0.9
    assert len(cv.fold_accuracies) == 4 and all(0 <= f <= 1 for f in cv.fold_accuracies)
    assert cv.log_loss < 0.4
    assert cv.c in load_pipeline_config().classifier.c_grid
    assert cv.correct.shape == labels.shape and cv.correct.mean() == cv.accuracy

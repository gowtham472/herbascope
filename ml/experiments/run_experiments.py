"""Run the pre-registered model-improvement ladder (ml/configs/experiments.json).

Protocol
  * Selection metric: out-of-fold accuracy from StratifiedGroupKFold cross-validation on the
    development pool (train + validation). Groups are the near-duplicate groups from the dataset
    lock, so near-copies never straddle a fold boundary. C is chosen per rung by pooled
    out-of-fold log-loss.
  * Rungs run in registered order; each applies its change to the best configuration adopted so
    far and is adopted only under the registered rule.
  * Every rung is also scored with the production protocol (fit on train, choose C on
    validation log-loss, evaluate validation / test / held-out). These numbers are reported for
    transparency and never influence adoption.

Outputs: models/experiments/results.json, docs/reports/model_improvement_log.md
Usage:   python -m ml.experiments.run_experiments
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

from ml import paths
from ml.classifiers.classifier import fit_logistic_regression, select_regularization
from ml.experiments.feature_cache import TokenFeatures, load_features
from ml.pipeline_config import Backbone, PipelineConfig, load_pipeline_config
from ml.training import datasets as ds
from ml.training.train_classifier import training_rows

EXPERIMENTS_CONFIG = paths.REPO_ROOT / "ml" / "configs" / "experiments.json"
RESULTS_PATH = paths.MODELS_DIR / "experiments" / "results.json"
REPORT_PATH = paths.DOCS_REPORTS_DIR / "model_improvement_log.md"
SPLITS = (ds.TRAIN, ds.VALIDATION, ds.TEST, ds.HELDOUT_KNOWN)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Candidate(_Strict):
    backbone: str
    input_size: int = Field(gt=0, multiple_of=14)
    pooling: Literal["cls", "cls_patchmean"]
    views: int = Field(ge=1, le=8)
    train_on_views: bool


class Baseline(_Strict):
    id: str
    title: str
    hypothesis: str
    config: Candidate


class Rung(_Strict):
    id: str
    title: str
    hypothesis: str
    change: dict[str, str | int | bool]


class ExperimentsConfig(_Strict):
    version: str
    registered_at: str
    objective: str
    primary_metric: str
    adoption_rule: str
    latency_budget_ms: float = Field(gt=0)
    latency_rationale: str
    cv_folds: int = Field(ge=2)
    feature_views: int = Field(ge=1, le=8)
    backbones: dict[str, Backbone]
    baseline: Baseline
    ladder: list[Rung]


@dataclass(frozen=True)
class RungResult:
    id: str
    title: str
    hypothesis: str
    change: dict
    config: dict
    adopted: bool
    cv_accuracy: float
    cv_fold_accuracies: list[float]
    cv_log_loss: float
    cv_c: float
    holdout_c: float
    validation_accuracy: float
    test_correct: int
    test_total: int
    heldout_correct: int
    heldout_total: int
    test_errors_by_fragment: dict[str, int]
    embedding_dimension: int
    encode_ms_per_image: float


def load_experiments_config() -> ExperimentsConfig:
    return ExperimentsConfig.model_validate_json(EXPERIMENTS_CONFIG.read_text(encoding="utf-8"))


@lru_cache(maxsize=16)
def _features(backbone_key: str, input_size: int, split: str) -> TokenFeatures:
    experiments, pipeline = load_experiments_config(), load_pipeline_config()
    return load_features(
        experiments.backbones[backbone_key],
        input_size,
        split,
        experiments.feature_views,
        pipeline.encoder.batch_size,
    )


def cross_validate(
    view_vectors: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train_on_views: bool,
    pipeline: PipelineConfig,
    folds: int,
) -> tuple[float, list[float], float, float]:
    """Grouped stratified CV; returns (accuracy, fold accuracies, log-loss, C) at the best C."""
    embeddings = training_rows(view_vectors, labels, on_views=False)[0]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=pipeline.seed)
    fold_indices = list(splitter.split(embeddings, labels, groups))
    best = None
    for c in pipeline.classifier.c_grid:
        probabilities = np.zeros((labels.size, int(labels.max()) + 1))
        fold_accuracies = []
        for train_index, test_index in fold_indices:
            x_fit, y_fit = training_rows(view_vectors[:, train_index], labels[train_index], train_on_views)
            model = fit_logistic_regression(
                x_fit, y_fit, c, pipeline.classifier.class_weight, pipeline.classifier.max_iter, pipeline.seed
            )
            probabilities[test_index] = model.predict_proba(embeddings[test_index])
            fold_accuracies.append(float(np.mean(probabilities[test_index].argmax(1) == labels[test_index])))
        loss = float(log_loss(labels, probabilities))
        accuracy = float(np.mean(probabilities.argmax(1) == labels))
        if best is None or (loss, c) < (best[2], best[3]):
            best = (accuracy, fold_accuracies, loss, c)
    return best


def evaluate_candidate(rung_id: str, title: str, hypothesis: str, change: dict, candidate: Candidate) -> dict:
    experiments, pipeline = load_experiments_config(), load_pipeline_config()
    features = {split: _features(candidate.backbone, candidate.input_size, split) for split in SPLITS}
    frames = {split: ds.load_split(split).frame for split in SPLITS}
    classes = sorted(frames[ds.TRAIN]["class_name"].unique())
    index_of = {name: i for i, name in enumerate(classes)}
    labels = {split: frames[split]["class_name"].map(index_of).to_numpy() for split in SPLITS}
    views = {split: features[split].view_embeddings(candidate.pooling, candidate.views) for split in SPLITS}
    final = {split: features[split].embeddings(candidate.pooling, candidate.views) for split in SPLITS}

    dev_views = np.concatenate([views[ds.TRAIN], views[ds.VALIDATION]], axis=1)
    dev_labels = np.concatenate([labels[ds.TRAIN], labels[ds.VALIDATION]])
    dev_groups = np.concatenate([frames[ds.TRAIN]["group_id"], frames[ds.VALIDATION]["group_id"]])
    cv_accuracy, fold_accuracies, cv_loss, cv_c = cross_validate(
        dev_views, dev_labels, dev_groups, candidate.train_on_views, pipeline, experiments.cv_folds
    )

    x_fit, y_fit = training_rows(views[ds.TRAIN], labels[ds.TRAIN], candidate.train_on_views)
    holdout_c, _ = select_regularization(
        (x_fit, y_fit),
        (final[ds.VALIDATION], labels[ds.VALIDATION]),
        pipeline.classifier.c_grid,
        pipeline.classifier.class_weight,
        pipeline.classifier.max_iter,
        pipeline.seed,
    )
    model = fit_logistic_regression(
        x_fit, y_fit, holdout_c, pipeline.classifier.class_weight, pipeline.classifier.max_iter, pipeline.seed
    )
    correct = {split: model.predict(final[split]) == labels[split] for split in SPLITS}
    test_frame = frames[ds.TEST]
    errors = test_frame.loc[~correct[ds.TEST], "fragment_type"].value_counts()
    return {
        "id": rung_id,
        "title": title,
        "hypothesis": hypothesis,
        "change": change,
        "config": candidate.model_dump(),
        "cv_accuracy": cv_accuracy,
        "cv_fold_accuracies": fold_accuracies,
        "cv_log_loss": cv_loss,
        "cv_c": cv_c,
        "holdout_c": holdout_c,
        "validation_accuracy": float(correct[ds.VALIDATION].mean()),
        "test_correct": int(correct[ds.TEST].sum()),
        "test_total": int(correct[ds.TEST].size),
        "heldout_correct": int(correct[ds.HELDOUT_KNOWN].sum()),
        "heldout_total": int(correct[ds.HELDOUT_KNOWN].size),
        "test_errors_by_fragment": {str(k): int(v) for k, v in errors.items()},
        "embedding_dimension": int(final[ds.TRAIN].shape[1]),
        "encode_ms_per_image": 1000.0 * features[ds.TRAIN].seconds_per_view_image * candidate.views,
    }


def improves(result: dict, best: dict, latency_budget_ms: float) -> bool:
    """The registered adoption rule."""
    if result["encode_ms_per_image"] > latency_budget_ms:
        return False
    if result["cv_accuracy"] != best["cv_accuracy"]:
        return result["cv_accuracy"] > best["cv_accuracy"]
    return result["cv_log_loss"] < best["cv_log_loss"]


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def render_log(experiments: ExperimentsConfig, results: list[RungResult], dev_size: int) -> str:
    adopted = [r for r in results if r.adopted]
    best = adopted[-1]
    budget = experiments.latency_budget_ms
    rows = [
        "| Rung | Change | CV accuracy (fold range) | CV log-loss | Adopted | Validation | Test | Held-out | Dim | Encode ms/image |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        change = ", ".join(f"{k}={v}" for k, v in r.change.items()) or "baseline"
        over_budget = " (over budget)" if r.encode_ms_per_image > budget else ""
        rows.append(
            f"| {r.id} {r.title} | `{change}` | {_pct(r.cv_accuracy)} "
            f"({_pct(min(r.cv_fold_accuracies))} to {_pct(max(r.cv_fold_accuracies))}) | {r.cv_log_loss:.3f} | "
            f"{'yes' if r.adopted else 'no'} | {_pct(r.validation_accuracy)} | "
            f"{_pct(r.test_correct / r.test_total)} ({r.test_correct}/{r.test_total}) | "
            f"{_pct(r.heldout_correct / r.heldout_total)} ({r.heldout_correct}/{r.heldout_total}) | "
            f"{r.embedding_dimension} | {r.encode_ms_per_image:.0f}{over_budget} |"
        )
    milestones = []
    for target in (0.90, 0.95):
        reached = next((r for r in adopted if r.cv_accuracy >= target), None)
        milestones.append(
            f"- CV accuracy >= {target:.0%}: "
            + (
                f"first reached by {reached.id} ({_pct(reached.cv_accuracy)} CV; test "
                f"{reached.test_correct}/{reached.test_total})."
                if reached
                else "not reached by the registered ladder."
            )
        )
    details = []
    for r in results:
        errors = ", ".join(f"{k}: {v}" for k, v in r.test_errors_by_fragment.items()) or "none"
        details += [
            f"### {r.id}: {r.title}",
            "",
            f"- Hypothesis: {r.hypothesis}",
            f"- Configuration: `{json.dumps(r.config)}`",
            f"- Cross-validation: accuracy {_pct(r.cv_accuracy)}, folds "
            f"{', '.join(_pct(a) for a in r.cv_fold_accuracies)}, log-loss {r.cv_log_loss:.3f}, C={r.cv_c:g}",
            f"- Production protocol (C={r.holdout_c:g} chosen on validation): validation "
            f"{_pct(r.validation_accuracy)}, test {r.test_correct}/{r.test_total}, held-out "
            f"{r.heldout_correct}/{r.heldout_total}",
            f"- Test errors by fragment type: {errors}",
            f"- Encoding cost: {r.encode_ms_per_image:.0f} ms per image "
            f"({'within' if r.encode_ms_per_image <= budget else 'over'} the {budget:.0f} ms budget)",
            f"- Decision: {'adopted' if r.adopted else 'not adopted'}",
            "",
        ]
    return "\n".join(
        [
            "# Model Improvement Log",
            "",
            f"_Generated by `python -m ml.experiments.run_experiments` at "
            f"{datetime.now(UTC).isoformat(timespec='seconds')} from `ml/configs/experiments.json` "
            f"({experiments.version}, registered {experiments.registered_at}). Do not edit by hand._",
            "",
            "## Protocol",
            "",
            f"- Objective: {experiments.objective}",
            f"- Primary metric: {experiments.primary_metric} Development pool: {dev_size} images; "
            f"one image is {100 / dev_size:.2f} percentage points.",
            f"- Adoption rule: {experiments.adoption_rule}",
            f"- Latency budget: {budget:.0f} ms per image. {experiments.latency_rationale}",
            f"- Test set: {results[0].test_total} images; one image is {100 / results[0].test_total:.2f} "
            "percentage points. Held-out known material contains fragment types absent from training.",
            "",
            "## Ladder",
            "",
            *rows,
            "",
            "## Milestones",
            "",
            *milestones,
            "",
            "## Selected configuration",
            "",
            f"`{json.dumps(best.config)}` (rung {best.id}): CV accuracy {_pct(best.cv_accuracy)}, test "
            f"{best.test_correct}/{best.test_total}, held-out {best.heldout_correct}/{best.heldout_total}.",
            "",
            "## Rung details",
            "",
            *details,
        ]
    )


def main() -> int:
    experiments = load_experiments_config()
    baseline = experiments.baseline
    current = evaluate_candidate(baseline.id, baseline.title, baseline.hypothesis, {}, baseline.config)
    results = [RungResult(**current, adopted=True)]
    best = current
    print(
        f"[{baseline.id}] CV {_pct(current['cv_accuracy'])} test {current['test_correct']}/{current['test_total']}"
    )
    dev_size = sum(len(ds.load_split(split).frame) for split in (ds.TRAIN, ds.VALIDATION))
    _write_outputs(experiments, results, dev_size)
    for rung in experiments.ladder:
        candidate = Candidate(**{**best["config"], **rung.change})
        result = evaluate_candidate(rung.id, rung.title, rung.hypothesis, dict(rung.change), candidate)
        adopted = improves(result, best, experiments.latency_budget_ms)
        results.append(RungResult(**result, adopted=adopted))
        if adopted:
            best = result
        print(
            f"[{rung.id}] CV {_pct(result['cv_accuracy'])} log-loss {result['cv_log_loss']:.3f} "
            f"test {result['test_correct']}/{result['test_total']} "
            f"{result['encode_ms_per_image']:.0f} ms -> {'adopted' if adopted else 'not adopted'}",
            flush=True,
        )
        _write_outputs(experiments, results, dev_size)
    print(f"selected: {best['id']} {json.dumps(best['config'])}")
    return 0


def _write_outputs(experiments: ExperimentsConfig, results: list[RungResult], dev_size: int) -> None:
    """Persist after every rung so long runs leave an up-to-date log."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    REPORT_PATH.write_text(render_log(experiments, results, dev_size), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

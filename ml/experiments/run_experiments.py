"""Run a pre-registered model-improvement ladder (ml/configs/experiments-v*.json).

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

Outputs: models/experiments/<ladder version>.json, docs/reports/model_improvement_<ladder version>.md
Usage:   python -m ml.experiments.run_experiments --config ml/configs/experiments-v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

from ml import paths
from ml.classifiers.classifier import fit_logistic_regression, select_regularization
from ml.encoders.dinov2_encoder import BLOCK_POOLINGS
from ml.experiments.feature_cache import TokenFeatures, load_features
from ml.pipeline_config import Backbone, PipelineConfig, Pooling, load_pipeline_config
from ml.training import datasets as ds
from ml.training.train_classifier import training_rows

SPLITS = (ds.TRAIN, ds.VALIDATION, ds.TEST, ds.HELDOUT_KNOWN)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Candidate(_Strict):
    backbone: str
    input_size: int = Field(gt=0, multiple_of=14)
    pooling: Pooling
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
    cv_errors_by_fragment: dict[str, int]
    dev_images_by_fragment: dict[str, int]
    holdout_c: float
    validation_accuracy: float
    test_correct: int
    test_total: int
    heldout_correct: int
    heldout_total: int
    test_errors_by_fragment: dict[str, int]
    test_images_by_fragment: dict[str, int]
    embedding_dimension: int
    encode_ms_per_image: float


def load_experiments_config(path: Path) -> ExperimentsConfig:
    return ExperimentsConfig.model_validate_json(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=16)
def _features(
    backbone: Backbone, input_size: int, split: str, views: int, need_layers: bool
) -> TokenFeatures:
    return load_features(
        backbone, input_size, split, views, load_pipeline_config().encoder.batch_size, need_layers
    )


@dataclass(frozen=True)
class CrossValidation:
    """Out-of-fold results at the C with the lowest pooled out-of-fold log-loss."""

    accuracy: float
    fold_accuracies: list[float]
    log_loss: float
    c: float
    correct: np.ndarray  # per development image: out-of-fold prediction was correct


def cross_validate(
    view_vectors: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train_on_views: bool,
    pipeline: PipelineConfig,
    folds: int,
) -> CrossValidation:
    """Grouped stratified cross-validation over the configured C grid."""
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
        correct = probabilities.argmax(1) == labels
        if best is None or (loss, c) < (best.log_loss, best.c):
            best = CrossValidation(float(correct.mean()), fold_accuracies, loss, c, correct)
    return best


def _errors_by_fragment(frame: pd.DataFrame, correct: np.ndarray) -> dict[str, int]:
    counts = frame.loc[~correct, "fragment_type"].value_counts()
    return {str(fragment): int(count) for fragment, count in counts.items()}


def _images_by_fragment(frame: pd.DataFrame) -> dict[str, int]:
    return {str(fragment): int(count) for fragment, count in frame["fragment_type"].value_counts().items()}


def _error_summary(errors: dict[str, int], totals: dict[str, int]) -> str:
    return (
        ", ".join(f"{fragment}: {count} of {totals[fragment]}" for fragment, count in errors.items())
        or "none"
    )


def evaluate_candidate(
    experiments: ExperimentsConfig,
    rung_id: str,
    title: str,
    hypothesis: str,
    change: dict,
    candidate: Candidate,
) -> dict:
    pipeline = load_pipeline_config()
    backbone = experiments.backbones[candidate.backbone]
    features = {
        split: _features(
            backbone,
            candidate.input_size,
            split,
            experiments.feature_views,
            need_layers=candidate.pooling in BLOCK_POOLINGS,
        )
        for split in SPLITS
    }
    frames = {split: ds.load_split(split).frame for split in SPLITS}
    classes = sorted(frames[ds.TRAIN]["class_name"].unique())
    index_of = {name: i for i, name in enumerate(classes)}
    labels = {split: frames[split]["class_name"].map(index_of).to_numpy() for split in SPLITS}
    views = {split: features[split].view_embeddings(candidate.pooling, candidate.views) for split in SPLITS}
    final = {split: features[split].embeddings(candidate.pooling, candidate.views) for split in SPLITS}

    dev_frame = pd.concat([frames[ds.TRAIN], frames[ds.VALIDATION]], ignore_index=True)
    dev_views = np.concatenate([views[ds.TRAIN], views[ds.VALIDATION]], axis=1)
    dev_labels = np.concatenate([labels[ds.TRAIN], labels[ds.VALIDATION]])
    cv = cross_validate(
        dev_views,
        dev_labels,
        dev_frame["group_id"].to_numpy(),
        candidate.train_on_views,
        pipeline,
        experiments.cv_folds,
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
    return {
        "id": rung_id,
        "title": title,
        "hypothesis": hypothesis,
        "change": change,
        "config": candidate.model_dump(),
        "cv_accuracy": cv.accuracy,
        "cv_fold_accuracies": cv.fold_accuracies,
        "cv_log_loss": cv.log_loss,
        "cv_c": cv.c,
        "cv_errors_by_fragment": _errors_by_fragment(dev_frame, cv.correct),
        "dev_images_by_fragment": _images_by_fragment(dev_frame),
        "holdout_c": holdout_c,
        "validation_accuracy": float(correct[ds.VALIDATION].mean()),
        "test_correct": int(correct[ds.TEST].sum()),
        "test_total": int(correct[ds.TEST].size),
        "heldout_correct": int(correct[ds.HELDOUT_KNOWN].sum()),
        "heldout_total": int(correct[ds.HELDOUT_KNOWN].size),
        "test_errors_by_fragment": _errors_by_fragment(frames[ds.TEST], correct[ds.TEST]),
        "test_images_by_fragment": _images_by_fragment(frames[ds.TEST]),
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
        errors = _error_summary(r.test_errors_by_fragment, r.test_images_by_fragment)
        cv_errors = _error_summary(r.cv_errors_by_fragment, r.dev_images_by_fragment)
        details += [
            f"### {r.id}: {r.title}",
            "",
            f"- Hypothesis: {r.hypothesis}",
            f"- Configuration: `{json.dumps(r.config)}`",
            f"- Cross-validation: accuracy {_pct(r.cv_accuracy)}, folds "
            f"{', '.join(_pct(a) for a in r.cv_fold_accuracies)}, log-loss {r.cv_log_loss:.3f}, C={r.cv_c:g}",
            f"- Out-of-fold errors by fragment type ({sum(r.cv_errors_by_fragment.values())}): {cv_errors}",
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
            f"{datetime.now(UTC).isoformat(timespec='seconds')} from the ladder "
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
    parser = argparse.ArgumentParser(description="Run a pre-registered experiment ladder.")
    parser.add_argument("--config", type=Path, required=True, help="experiment ladder JSON")
    experiments = load_experiments_config(parser.parse_args().config)
    baseline = experiments.baseline
    current = evaluate_candidate(
        experiments, baseline.id, baseline.title, baseline.hypothesis, {}, baseline.config
    )
    results = [RungResult(**current, adopted=True)]
    best = current
    print(
        f"[{baseline.id}] CV {_pct(current['cv_accuracy'])} test {current['test_correct']}/{current['test_total']}",
        flush=True,
    )
    dev_size = sum(len(ds.load_split(split).frame) for split in (ds.TRAIN, ds.VALIDATION))
    _write_outputs(experiments, results, dev_size)
    for rung in experiments.ladder:
        candidate = Candidate(**{**best["config"], **rung.change})
        result = evaluate_candidate(
            experiments, rung.id, rung.title, rung.hypothesis, dict(rung.change), candidate
        )
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
    """Persist after every rung so long runs leave an up-to-date log (one file pair per ladder version)."""
    results_path = paths.MODELS_DIR / "experiments" / f"{experiments.version}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    report_path = paths.DOCS_REPORTS_DIR / f"model_improvement_{experiments.version}.md"
    report_path.write_text(render_log(experiments, results, dev_size), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

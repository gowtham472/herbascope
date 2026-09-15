"""Evaluate the complete screening pipeline exactly as the API runs it.

Evaluation sets (none used to fit the classifier, the reference library or any threshold):
  test             Mikrobat known material, same fragment types as training   -> expected KNOWN
  heldout_known    Mikrobat fragment types withheld from training/references  -> expected KNOWN
  ood_evaluation   DIMPSAR classes disjoint from the calibration classes      -> expected UNKNOWN
  ambiguity_probe  byte-identical images filed under both Mikrobat classes    -> reported separately

Outputs: models/reports/{evaluation.json, evaluation_samples.csv}, docs/reports/evaluation_report.md
Usage:   python -m ml.evaluation.evaluate
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from ml import paths
from ml.decision.decision_engine import PRELIMINARY_PASS, REVIEW_REQUIRED, UNKNOWN_DECISION
from ml.evidence.evidence_engine import HIGH, LOW, MEDIUM
from ml.inference.screening_pipeline import ArtifactPaths, ScreeningPipeline
from ml.preprocessing.image_io import load_image_file
from ml.training import datasets as ds
from ml.uncertainty.unknown_detector import KNOWN, UNCERTAIN, UNKNOWN

KNOWN_SETS = (ds.TEST, ds.HELDOUT_KNOWN)
DECISIONS = (PRELIMINARY_PASS, REVIEW_REQUIRED, UNKNOWN_DECISION)


def score_split(pipeline: ScreeningPipeline, name: str) -> pd.DataFrame:
    embeddings = ds.load_embeddings(name)
    if embeddings.fingerprint != pipeline.encoder.fingerprint:
        raise ValueError(f"{name} embeddings were produced by a different encoder/preprocessing version")
    qualities = [
        pipeline.assess_image_quality(load_image_file(path)) for path in embeddings.split.image_paths()
    ]
    results = pipeline.analyze_embeddings(embeddings.vectors, qualities)
    rows = []
    for (_, sample), result in zip(embeddings.split.frame.iterrows(), results, strict=True):
        evidence = result.evidence
        reference_classes = [m.reference.class_name for m in evidence.retrieval.matches]
        rows.append(
            {
                "split": name,
                "image_id": sample["image_id"],
                "dataset": sample["dataset"],
                "true_class": sample["class_name"],
                "fragment_type": sample["fragment_type"],
                "path": sample["path"],
                "predicted_class": evidence.prediction.class_name,
                "confidence": evidence.prediction.confidence,
                "retrieved_class": evidence.retrieval.retrieved_class,
                "top_similarity": evidence.retrieval.top_similarity,
                "reference_classes": "|".join(reference_classes),
                "agreement": evidence.agreement,
                "distance": evidence.unknown.distance,
                "unknown_risk": evidence.unknown.risk,
                "unknown_status": evidence.unknown.status,
                "quality_status": evidence.quality.status,
                "strength": evidence.strength,
                "decision": result.decision.status,
            }
        )
    return pd.DataFrame(rows)


def _rate(mask: pd.Series) -> float | None:
    return float(mask.mean()) if len(mask) else None


def classification_metrics(frame: pd.DataFrame, classes: list[str]) -> dict:
    y_true, y_pred = frame["true_class"], frame["predicted_class"]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return {
        "samples": len(frame),
        "accuracy": _rate(y_true == y_pred),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            c: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for c, p, r, f, s in zip(classes, precision, recall, f1, support, strict=True)
        },
        "confusion_matrix": {
            "labels": classes,
            "matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        },
        "top_k_note": f"top-k accuracy for k >= {len(classes)} is 1.0 by construction with {len(classes)} classes",
    }


def retrieval_metrics(frame: pd.DataFrame, k: int) -> dict:
    references = frame["reference_classes"].str.split("|")
    hits = {
        f"top_{n}_hit_rate": _rate(
            pd.Series([t in refs[:n] for t, refs in zip(frame["true_class"], references, strict=True)])
        )
        for n in sorted({1, min(3, k), k})
    }
    return {
        **hits,
        "top_1_reference_class_accuracy": _rate(references.str[0] == frame["true_class"]),
        "retrieved_class_accuracy": _rate(frame["retrieved_class"] == frame["true_class"]),
        "top_similarity": describe(frame["top_similarity"]),
    }


def describe(values: pd.Series) -> dict:
    array = values.to_numpy(dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        **{f"p{p}": float(np.percentile(array, p)) for p in (5, 25, 50, 75, 95)},
    }


def unknown_metrics(scored: dict[str, pd.DataFrame]) -> dict:
    metrics = {}
    for name in KNOWN_SETS:
        status = scored[name]["unknown_status"]
        metrics[name] = {
            "known_acceptance_rate": _rate(status != UNKNOWN),
            "known_rejection_rate": _rate(status == UNKNOWN),
            "uncertain_rate": _rate(status == UNCERTAIN),
            "known_status_rate": _rate(status == KNOWN),
        }
    ood = scored[ds.OOD_EVALUATION]["unknown_status"]
    metrics[ds.OOD_EVALUATION] = {
        "ood_rejection_rate": _rate(ood == UNKNOWN),
        "ood_false_acceptance_rate": _rate(ood != UNKNOWN),
        "uncertain_rate": _rate(ood == UNCERTAIN),
    }
    metrics["distance_distributions"] = {name: describe(frame["distance"]) for name, frame in scored.items()}
    return metrics


def decision_metrics(scored: dict[str, pd.DataFrame]) -> dict:
    table = {
        name: {d: int((frame["decision"] == d).sum()) for d in DECISIONS} for name, frame in scored.items()
    }
    known = pd.concat([scored[name] for name in KNOWN_SETS])
    correct = known["predicted_class"] == known["true_class"]
    ood = scored[ds.OOD_EVALUATION]
    probe = scored[ds.AMBIGUITY_PROBE]
    return {
        "decision_counts": table,
        "known_material": {
            "correct_pass": int(((known["decision"] == PRELIMINARY_PASS) & correct).sum()),
            "false_pass_wrong_class": int(((known["decision"] == PRELIMINARY_PASS) & ~correct).sum()),
            "review_when_classifier_wrong": int(((known["decision"] == REVIEW_REQUIRED) & ~correct).sum()),
            "review_when_classifier_right": int(((known["decision"] == REVIEW_REQUIRED) & correct).sum()),
            "false_unknown": int((known["decision"] == UNKNOWN_DECISION).sum()),
            "samples": len(known),
        },
        "ood": {
            "correct_unknown": int((ood["decision"] == UNKNOWN_DECISION).sum()),
            "false_pass": int((ood["decision"] == PRELIMINARY_PASS).sum()),
            "review": int((ood["decision"] == REVIEW_REQUIRED).sum()),
            "samples": len(ood),
        },
        "ambiguity_probe": {
            "pass": int((probe["decision"] == PRELIMINARY_PASS).sum()),
            "review": int((probe["decision"] == REVIEW_REQUIRED).sum()),
            "unknown": int((probe["decision"] == UNKNOWN_DECISION).sum()),
            "samples": len(probe),
        },
    }


ABLATION_SYSTEMS = {
    "A: classifier only": lambda f: pd.Series(True, index=f.index),
    "B: + retrieval agreement": lambda f: f["agreement"] != LOW,
    "C: + unknown detection": lambda f: (f["agreement"] != LOW) & (f["unknown_status"] != UNKNOWN),
    "D: full evidence + decision (PASS)": lambda f: f["decision"] == PRELIMINARY_PASS,
}


def ablation(scored: dict[str, pd.DataFrame]) -> list[dict]:
    """What each progressively richer system accepts, reported separately per evaluation set."""
    rows = []
    for label, accepts in ABLATION_SYSTEMS.items():
        row = {"system": label}
        for name in KNOWN_SETS:
            frame = scored[name]
            accepted = accepts(frame)
            correct = frame.loc[accepted, "predicted_class"] == frame.loc[accepted, "true_class"]
            row[name] = {
                "acceptance_rate": _rate(accepted),
                "accepted_precision": _rate(correct),
                "wrong_class_accepted": int((~correct).sum()),
            }
        row[ds.OOD_EVALUATION] = {"false_acceptance_rate": _rate(accepts(scored[ds.OOD_EVALUATION]))}
        rows.append(row)
    return rows


def findings(scored: dict[str, pd.DataFrame], reference_fragment_types: set[str]) -> list[str]:
    """Plain-language findings derived only from the scored samples."""
    lines = []
    for name in KNOWN_SETS:
        frame = scored[name]
        correct = frame["predicted_class"] == frame["true_class"]
        passed = frame["decision"] == PRELIMINARY_PASS
        covered = frame["fragment_type"].isin(reference_fragment_types).mean()
        lines.append(
            f"- `{name}`: {len(frame)} samples, {covered:.0%} of them from fragment types present in the reference "
            f"library; classification accuracy {correct.mean():.1%}; {int(passed.sum())} PRELIMINARY_PASS with "
            f"precision {_pct(_rate(correct[passed]))}; {int((~correct & ~passed).sum())} of "
            f"{int((~correct).sum())} misclassified samples were stopped (REVIEW_REQUIRED or UNKNOWN)."
        )
    test_precision = _rate(
        (scored[ds.TEST]["predicted_class"] == scored[ds.TEST]["true_class"])[
            scored[ds.TEST]["decision"] == PRELIMINARY_PASS
        ]
    )
    heldout = scored[ds.HELDOUT_KNOWN]
    heldout_precision = _rate(
        (heldout["predicted_class"] == heldout["true_class"])[heldout["decision"] == PRELIMINARY_PASS]
    )
    if test_precision is not None and heldout_precision is not None and heldout_precision < test_precision:
        lines.append(
            "- PRELIMINARY_PASS precision drops on fragment types absent from the reference library. Classifier and "
            "retrieval share one visual representation, so they can agree confidently on the wrong class for an "
            "unseen fragment type. Screening conclusions only transfer to fragment types represented in the "
            "reference library."
        )
    ood = scored[ds.OOD_EVALUATION]
    lines.append(
        f"- `{ds.OOD_EVALUATION}`: {int((ood['decision'] == UNKNOWN_DECISION).sum())} of {len(ood)} DIMPSAR images "
        f"were UNKNOWN; false PRELIMINARY_PASS: {int((ood['decision'] == PRELIMINARY_PASS).sum())}."
    )
    probe = scored[ds.AMBIGUITY_PROBE]
    probe_types = set(probe["fragment_type"])
    counts = probe["decision"].value_counts()
    lines.append(
        f"- `{ds.AMBIGUITY_PROBE}`: {len(probe)} label-conflict images -> {int(counts.get(PRELIMINARY_PASS, 0))} "
        f"PRELIMINARY_PASS, {int(counts.get(REVIEW_REQUIRED, 0))} REVIEW_REQUIRED, "
        f"{int(counts.get(UNKNOWN_DECISION, 0))} UNKNOWN. Their fragment type(s) "
        f"{'are' if probe_types <= reference_fragment_types else 'are not'} represented in the reference library; "
        "the pipeline has no signal for a label conflict itself, only for visual consistency with references."
    )
    return lines


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _table(header: list[str], rows: list[list]) -> str:
    return "\n".join(
        ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    )


def render_report(report: dict) -> str:
    cal, policy, quality = report["unknown_calibration"], report["decision_policy"], report["quality_bounds"]
    clf = report["classification"]
    lines = [
        "# Evaluation Report",
        "",
        f"_Generated by `python -m ml.evaluation.evaluate` at {report['generated_at']}. Do not edit by hand._",
        "",
        "## Model artifacts",
        "",
        _table(["Artifact", "Version"], [[k, v] for k, v in report["model"].items() if k != "classes"]),
        "",
        "## Classification",
        "",
        _table(
            ["Set", "Samples", "Accuracy", "Macro F1"],
            [[name, m["samples"], _pct(m["accuracy"]), f"{m['macro_f1']:.3f}"] for name, m in clf.items()],
        ),
        "",
    ]
    for name, metrics in clf.items():
        labels = metrics["confusion_matrix"]["labels"]
        lines += [
            f"Confusion matrix - `{name}` (rows = true, columns = predicted)",
            "",
            _table(
                ["true \\ pred", *labels],
                [[labels[i], *row] for i, row in enumerate(metrics["confusion_matrix"]["matrix"])],
            ),
            "",
        ]
    lines += [f"_{next(iter(clf.values()))['top_k_note']}._", "", "## Reference retrieval", ""]
    retrieval = report["retrieval"]
    keys = [k for k in next(iter(retrieval.values())) if k != "top_similarity"]
    lines += [
        _table(
            ["Set", *keys, "median top similarity"],
            [
                [name] + [_pct(m[k]) for k in keys] + [f"{m['top_similarity']['p50']:.3f}"]
                for name, m in retrieval.items()
            ],
        ),
        "",
    ]
    unknown = report["unknown_detection"]
    lines += [
        "## Unknown detection",
        "",
        f"Calibrated on {cal['known_samples']} Mikrobat validation samples and {cal['ood_samples']} DIMPSAR "
        f"calibration samples. Objective: {cal['objective']} (target known acceptance "
        f"{cal['objective_parameters']['target_known_acceptance']:.0%}).",
        "",
        _table(
            ["Parameter", "Value"],
            [
                ["distance threshold (UNKNOWN above)", f"{cal['distance_threshold']:.4f}"],
                ["known boundary (UNCERTAIN above)", f"{cal['known_boundary']:.4f}"],
                ["calibration known acceptance", _pct(cal["calibration_rates"]["known_acceptance"])],
                ["calibration OOD false acceptance", _pct(cal["calibration_rates"]["ood_false_acceptance"])],
            ],
        ),
        "",
        _table(
            [
                "Set",
                "Expected",
                "Known acceptance",
                "Known rejection",
                "OOD rejection",
                "OOD false acceptance",
                "UNCERTAIN",
            ],
            [
                [
                    ds.TEST,
                    "KNOWN",
                    _pct(unknown[ds.TEST]["known_acceptance_rate"]),
                    _pct(unknown[ds.TEST]["known_rejection_rate"]),
                    "",
                    "",
                    _pct(unknown[ds.TEST]["uncertain_rate"]),
                ],
                [
                    ds.HELDOUT_KNOWN,
                    "KNOWN",
                    _pct(unknown[ds.HELDOUT_KNOWN]["known_acceptance_rate"]),
                    _pct(unknown[ds.HELDOUT_KNOWN]["known_rejection_rate"]),
                    "",
                    "",
                    _pct(unknown[ds.HELDOUT_KNOWN]["uncertain_rate"]),
                ],
                [
                    ds.OOD_EVALUATION,
                    "UNKNOWN",
                    "",
                    "",
                    _pct(unknown[ds.OOD_EVALUATION]["ood_rejection_rate"]),
                    _pct(unknown[ds.OOD_EVALUATION]["ood_false_acceptance_rate"]),
                    _pct(unknown[ds.OOD_EVALUATION]["uncertain_rate"]),
                ],
            ],
        ),
        "",
        "Reference-distance distributions (distance = 1 - mean cosine similarity to the k nearest references):",
        "",
        _table(
            ["Set", "n", "min", "p5", "p25", "median", "p75", "p95", "max"],
            [
                [name, d["count"]] + [f"{d[k]:.3f}" for k in ("min", "p5", "p25", "p50", "p75", "p95", "max")]
                for name, d in unknown["distance_distributions"].items()
            ],
        ),
        "",
        f"**Limitation.** {cal['limitation']}",
        "",
        "## Decision policy",
        "",
        _table(
            ["Policy value", "Calibrated value", "Objective"],
            [
                [
                    "min_classifier_confidence",
                    f"{policy['min_classifier_confidence']:.4f}",
                    f"validation selective accuracy >= {policy['objectives']['min_classifier_confidence']['target_selective_accuracy']:.0%} "
                    f"(met: {policy['objectives']['min_classifier_confidence']['objective_met']})",
                ],
                [
                    "min_reference_similarity",
                    f"{policy['min_reference_similarity']:.4f}",
                    f"retain {policy['objectives']['min_reference_similarity']['target_known_retention']:.0%} of correctly classified validation samples",
                ],
                [
                    "max_unknown_risk",
                    f"{policy['max_unknown_risk']:.4f}",
                    policy["objectives"]["max_unknown_risk"]["definition"],
                ],
            ],
        ),
        "",
        "Decision counts per set:",
        "",
        _table(
            ["Set", *list(DECISIONS)],
            [
                [name] + [counts[d] for d in DECISIONS]
                for name, counts in report["decision"]["decision_counts"].items()
            ],
        ),
        "",
        _table(
            ["Known material (test + held-out)", "Count"],
            [[k, v] for k, v in report["decision"]["known_material"].items()],
        ),
        "",
        _table(["DIMPSAR OOD evaluation", "Count"], [[k, v] for k, v in report["decision"]["ood"].items()]),
        "",
        _table(
            ["Ambiguity probe (label conflicts)", "Count"],
            [[k, v] for k, v in report["decision"]["ambiguity_probe"].items()],
        ),
        "",
        "## Ablation",
        "",
        "Acceptance = the system would let the sample through; precision = share of accepted known samples whose "
        "predicted class is correct.",
        "",
        _table(
            [
                "System",
                "test acceptance",
                "test precision",
                "test wrong accepted",
                "held-out acceptance",
                "held-out precision",
                "held-out wrong accepted",
                "OOD false acceptance",
            ],
            [
                [
                    r["system"],
                    _pct(r[ds.TEST]["acceptance_rate"]),
                    _pct(r[ds.TEST]["accepted_precision"]),
                    r[ds.TEST]["wrong_class_accepted"],
                    _pct(r[ds.HELDOUT_KNOWN]["acceptance_rate"]),
                    _pct(r[ds.HELDOUT_KNOWN]["accepted_precision"]),
                    r[ds.HELDOUT_KNOWN]["wrong_class_accepted"],
                    _pct(r[ds.OOD_EVALUATION]["false_acceptance_rate"]),
                ]
                for r in report["ablation"]
            ],
        ),
        "",
        "## Findings",
        "",
        *report["findings"],
        "",
        "## Evidence agreement and image quality",
        "",
        _table(
            ["Set", HIGH, MEDIUM, LOW, "quality DEGRADED"],
            [[name] + [_pct(v) for v in row] for name, row in report["agreement_and_quality"].items()],
        ),
        "",
        "## Image-quality bounds",
        "",
        _table(
            ["Bound", "Value"],
            [[k, f"{v:.4g}" if isinstance(v, float) else v] for k, v in quality["bounds"].items()],
        ),
        "",
        f"Objective: {quality['objective']}.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    artifacts = ArtifactPaths.from_manifest(paths.MODELS_DIR)
    pipeline = ScreeningPipeline.load(artifacts)
    scored = {
        name: score_split(pipeline, name)
        for name in (ds.TEST, ds.HELDOUT_KNOWN, ds.OOD_EVALUATION, ds.AMBIGUITY_PROBE)
    }
    classes = list(pipeline.classifier.classes)
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": {k: v for k, v in pipeline.model_info.__dict__.items()},
        "classification": {name: classification_metrics(scored[name], classes) for name in KNOWN_SETS},
        "retrieval": {
            name: retrieval_metrics(scored[name], pipeline.unknown_calibration.k) for name in KNOWN_SETS
        },
        "unknown_detection": unknown_metrics(scored),
        "decision": decision_metrics(scored),
        "ablation": ablation(scored),
        "findings": findings(scored, {record.fragment_type for record in pipeline.index.records}),
        "agreement_and_quality": {
            name: [
                _rate(f["agreement"] == HIGH),
                _rate(f["agreement"] == MEDIUM),
                _rate(f["agreement"] == LOW),
                _rate(f["quality_status"] != "ACCEPTABLE"),
            ]
            for name, f in scored.items()
        },
        "unknown_calibration": json.loads(artifacts.unknown_calibration_path.read_text(encoding="utf-8")),
        "decision_policy": json.loads(artifacts.decision_policy_path.read_text(encoding="utf-8")),
        "quality_bounds": json.loads(artifacts.quality_bounds_path.read_text(encoding="utf-8")),
    }
    report["model"]["classes"] = list(report["model"]["classes"])
    paths.MODEL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    paths.DOCS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(scored.values()).to_csv(paths.MODEL_REPORTS_DIR / "evaluation_samples.csv", index=False)
    (paths.MODEL_REPORTS_DIR / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (paths.DOCS_REPORTS_DIR / "evaluation_report.md").write_text(render_report(report), encoding="utf-8")
    print("\n".join(report["findings"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Logistic-regression classifier over L2-normalised DINOv2 embeddings.

Artifacts (models/classifiers/):
  classifier.joblib      fitted scikit-learn estimator
  label_encoder.json     class index <-> class name mapping
  training_config.json   version, embedding fingerprint, data provenance, selection metrics
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

CLASSIFIER_FILE = "classifier.joblib"
LABEL_ENCODER_FILE = "label_encoder.json"
TRAINING_CONFIG_FILE = "training_config.json"


@dataclass(frozen=True)
class ClassProbability:
    class_name: str
    probability: float


@dataclass(frozen=True)
class ClassifierOutput:
    class_name: str
    confidence: float
    top_k: tuple[ClassProbability, ...]


def fit_logistic_regression(
    embeddings: np.ndarray, labels: np.ndarray, c: float, class_weight: str, max_iter: int, seed: int
) -> LogisticRegression:
    model = LogisticRegression(C=c, class_weight=class_weight, max_iter=max_iter, random_state=seed)
    return model.fit(embeddings, labels)


def select_regularization(
    train: tuple[np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray],
    c_grid: list[float],
    class_weight: str,
    max_iter: int,
    seed: int,
) -> tuple[float, list[dict]]:
    """Pick C by validation log-loss (a proper scoring rule, so probabilities stay honest)."""
    scores = []
    for c in c_grid:
        model = fit_logistic_regression(*train, c, class_weight, max_iter, seed)
        probabilities = model.predict_proba(validation[0])
        scores.append(
            {
                "C": c,
                "validation_log_loss": float(log_loss(validation[1], probabilities, labels=model.classes_)),
                "validation_accuracy": float(
                    np.mean(model.classes_[probabilities.argmax(1)] == validation[1])
                ),
            }
        )
    best = min(scores, key=lambda row: (row["validation_log_loss"], row["C"]))
    return best["C"], scores


class EmbeddingClassifier:
    def __init__(self, model: LogisticRegression, classes: list[str], version: str, fingerprint: str) -> None:
        if list(model.classes_) != list(range(len(classes))):
            raise ValueError("classifier was not trained on contiguous integer class indices")
        self._model = model
        self.classes = classes
        self.version = version
        self.fingerprint = fingerprint

    def predict_batch(self, embeddings: np.ndarray, top_k: int) -> list[ClassifierOutput]:
        outputs = []
        for row in self._model.predict_proba(embeddings):
            order = np.argsort(-row)[: min(top_k, len(self.classes))]
            ranked = tuple(ClassProbability(self.classes[i], float(row[i])) for i in order)
            outputs.append(ClassifierOutput(ranked[0].class_name, ranked[0].probability, ranked))
        return outputs

    def save(self, directory: Path, training_config: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, directory / CLASSIFIER_FILE)
        label_encoder = {"classes": self.classes}
        (directory / LABEL_ENCODER_FILE).write_text(json.dumps(label_encoder, indent=2), encoding="utf-8")
        config = {"version": self.version, "embedding_fingerprint": self.fingerprint, **training_config}
        (directory / TRAINING_CONFIG_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> EmbeddingClassifier:
        config = json.loads((directory / TRAINING_CONFIG_FILE).read_text(encoding="utf-8"))
        classes = json.loads((directory / LABEL_ENCODER_FILE).read_text(encoding="utf-8"))["classes"]
        model = joblib.load(directory / CLASSIFIER_FILE)
        return cls(model, classes, config["version"], config["embedding_fingerprint"])

"""Train the logistic-regression classifier on Mikrobat training embeddings.

C is chosen by validation log-loss; the final model is fitted on the training split only so
the validation split stays unseen for unknown and decision calibration.

With ``classifier.train_on_views`` every dihedral view of each training image is a training
row (orientation augmentation). Validation and inference always use the view-averaged
embedding, exactly as the API computes it.

Outputs: models/classifiers/{classifier.joblib, label_encoder.json, training_config.json}
Usage:   python -m ml.training.train_classifier
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime

import numpy as np

from ml import paths
from ml.classifiers.classifier import EmbeddingClassifier, fit_logistic_regression, select_regularization
from ml.encoders.dinov2_encoder import l2_normalize
from ml.pipeline_config import load_pipeline_config
from ml.preprocessing.transforms import PREPROCESSING_VERSION
from ml.training import datasets as ds


def _file_sha256(name: str) -> str:
    return hashlib.sha256(ds.split_path(name).read_bytes()).hexdigest()


def training_rows(
    view_vectors: np.ndarray, labels: np.ndarray, on_views: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix from (views, N, D) embeddings: view-averaged rows, or every view as a row."""
    if not on_views:
        return l2_normalize(view_vectors.mean(axis=0)), labels
    return view_vectors.reshape(-1, view_vectors.shape[-1]), np.tile(labels, view_vectors.shape[0])


def main() -> int:
    config = load_pipeline_config()
    train = ds.load_embeddings(ds.TRAIN)
    validation = ds.load_embeddings(ds.VALIDATION)
    if train.fingerprint != validation.fingerprint:
        raise ValueError("train and validation embeddings come from different embedding spaces")

    classes = sorted(train.split.frame["class_name"].unique())
    index_of = {name: i for i, name in enumerate(classes)}
    y_train = train.split.frame["class_name"].map(index_of).to_numpy()
    y_validation = validation.split.frame["class_name"].map(index_of).to_numpy()
    x_fit, y_fit = training_rows(train.view_vectors, y_train, config.classifier.train_on_views)

    c_value, scores = select_regularization(
        (x_fit, y_fit),
        (validation.vectors, y_validation),
        config.classifier.c_grid,
        config.classifier.class_weight,
        config.classifier.max_iter,
        config.seed,
    )
    model = fit_logistic_regression(
        x_fit, y_fit, c_value, config.classifier.class_weight, config.classifier.max_iter, config.seed
    )
    classifier = EmbeddingClassifier(model, classes, config.classifier.version, train.fingerprint)
    validation_accuracy = float(np.mean(model.predict(validation.vectors) == y_validation))
    classifier.save(
        paths.CLASSIFIERS_DIR,
        {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "algorithm": "sklearn.linear_model.LogisticRegression",
            "input": "L2-normalised DINOv2 embeddings (view-averaged at inference)",
            "preprocessing_version": PREPROCESSING_VERSION,
            "encoder": config.encoder.model_dump(),
            "dataset": "Mikrobat",
            "classes": classes,
            "train_images_per_class": {c: int((y_train == i).sum()) for c, i in index_of.items()},
            "validation_images_per_class": {c: int((y_validation == i).sum()) for c, i in index_of.items()},
            "hyperparameters": {
                "C": c_value,
                "class_weight": config.classifier.class_weight,
                "max_iter": config.classifier.max_iter,
                "random_state": config.seed,
                "train_on_views": config.classifier.train_on_views,
                "training_rows": int(x_fit.shape[0]),
            },
            "selection": {"criterion": "minimum validation log-loss", "grid": scores},
            "validation_accuracy": validation_accuracy,
            "split_files_sha256": {name: _file_sha256(name) for name in (ds.TRAIN, ds.VALIDATION)},
            "pipeline_config_version": config.version,
        },
    )
    print(json.dumps({"selected_C": c_value, "validation_accuracy": validation_accuracy}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Test support: a deterministic encoder double and a small synthetic screening pipeline.

The pipeline is built with the real training and calibration functions on synthetic
micrograph-like textures, so tests exercise production code paths without DINOv2 weights.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ml.classifiers.classifier import EmbeddingClassifier, fit_logistic_regression
from ml.decision.decision_engine import DecisionPolicy
from ml.inference.screening_pipeline import ScreeningPipeline
from ml.preprocessing.quality import QualityBounds
from ml.preprocessing.transforms import to_model_input
from ml.retrieval.faiss_store import ReferenceIndex, ReferenceRecord
from ml.uncertainty.unknown_detector import (
    UnknownCalibration,
    fit_risk_model,
    reference_distance,
    select_distance_threshold,
)

FINGERPRINT = "test-encoder|preprocess-v1|dim=64"
CLASSES = ["alpha", "beta"]


class FakeEncoder:
    """Maps the normalised model input to its L2-normalised, mean-centred 8x8 thumbnail."""

    name = "Test encoder"
    fingerprint = FINGERPRINT
    dimension = 64

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        batch = np.stack([to_model_input(image, 224) for image in images])
        channel = batch[:, 0]
        n, h, w = channel.shape
        pooled = channel.reshape(n, 8, h // 8, 8, w // 8).mean(axis=(2, 4)).reshape(n, 64)
        pooled = pooled - pooled.mean(axis=1, keepdims=True)
        return (pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)).astype(
            np.float32
        )


def texture(kind: str, seed: int, size: int = 256) -> Image.Image:
    """Synthetic grayscale textures: 'alpha' stripes, 'beta' checkerboard, 'ood' smooth gradient."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    if kind == "alpha":
        base = 128 + 90 * np.sign(np.sin((y + rng.integers(0, 8)) / 9.0))
    elif kind == "beta":
        base = 128 + 90 * np.sign(np.sin(x / 9.0) * np.sin(y / 9.0 + rng.uniform(0, 0.4)))
    elif kind == "ood":
        base = 40 + 170 * (x / size)
    else:
        raise ValueError(kind)
    noisy = base + rng.normal(0, 12, size=(size, size))
    return Image.fromarray(np.clip(noisy, 1, 254).astype(np.uint8), mode="L")


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass
class SyntheticSetup:
    pipeline: ScreeningPipeline
    data_dir: Path


def build_synthetic_pipeline(data_dir: Path) -> SyntheticSetup:
    encoder = FakeEncoder()
    train_images = {c: [texture(c, seed) for seed in range(12)] for c in CLASSES}
    validation_images = {c: [texture(c, seed) for seed in range(100, 106)] for c in CLASSES}

    train_x = np.concatenate([encoder.encode_images(train_images[c]) for c in CLASSES])
    train_y = np.repeat(np.arange(len(CLASSES)), 12)
    model = fit_logistic_regression(train_x, train_y, c=10.0, class_weight="balanced", max_iter=1000, seed=0)
    classifier = EmbeddingClassifier(model, CLASSES, "classifier-test", FINGERPRINT)

    records, vectors = [], []
    for c in CLASSES:
        for i, image in enumerate(train_images[c][:4]):
            relative = Path("references") / c / f"{c}-{i}.png"
            (data_dir / relative).parent.mkdir(parents=True, exist_ok=True)
            image.save(data_dir / relative)
            records.append(
                ReferenceRecord(
                    reference_id=f"REF{len(records) + 1:04d}",
                    vector_id=len(records),
                    image_id=f"{c}-{i}",
                    class_name=c,
                    fragment_type=f"{c} texture",
                    dataset="Synthetic",
                    source="tests/helpers.py",
                    image_path=relative.as_posix(),
                )
            )
            vectors.append(train_x[CLASSES.index(c) * 12 + i])
    index = ReferenceIndex.build(np.stack(vectors), records, "index-test", FINGERPRINT)

    k = 3
    validation_x = np.concatenate([encoder.encode_images(validation_images[c]) for c in CLASSES])
    ood_x = encoder.encode_images([texture("ood", seed) for seed in range(10)])

    def distances(x: np.ndarray) -> np.ndarray:
        return np.array(
            [reference_distance(np.array([m.similarity for m in row])) for row in index.search(x, k)]
        )

    known, ood = distances(validation_x), distances(ood_x)
    selection = select_distance_threshold(known, ood, 0.95)
    intercept, slope = fit_risk_model(known, ood, seed=0)
    calibration = UnknownCalibration(
        version="unknown-test",
        method="reference_similarity",
        k=k,
        distance_threshold=selection.threshold,
        known_boundary=selection.known_boundary,
        risk_intercept=intercept,
        risk_slope=slope,
        embedding_fingerprint=FINGERPRINT,
        index_version=index.version,
        limitation="Synthetic calibration for tests only.",
    )
    quality = QualityBounds(
        version="quality-test",
        tile_grid=8,
        min_side=224,
        min_sharpness=1.0,
        min_brightness=0.05,
        max_brightness=0.95,
        max_clipped_fraction=0.5,
        tile_detail_floor=1.0,
        min_detail_coverage=0.1,
    )
    policy = DecisionPolicy(
        version="decision-test",
        min_classifier_confidence=0.6,
        min_reference_similarity=0.5,
        max_unknown_risk=calibration.risk(calibration.known_boundary),
    )
    pipeline = ScreeningPipeline(encoder, classifier, index, calibration, quality, policy)
    return SyntheticSetup(pipeline=pipeline, data_dir=data_dir)

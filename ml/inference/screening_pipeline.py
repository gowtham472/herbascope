"""Load every model artifact once and run image -> evidence -> decision.

The API, the evaluation script and the smoke test all call this class, so the behaviour
that is evaluated is exactly the behaviour that is served.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ml.classifiers.classifier import EmbeddingClassifier
from ml.decision.decision_engine import Decision, DecisionPolicy, decide
from ml.encoders.dinov2_encoder import Dinov2Encoder
from ml.evidence.evidence_engine import Evidence, build_evidence
from ml.preprocessing.image_io import DecodedImage
from ml.preprocessing.quality import QualityAssessment, QualityBounds, assess_quality, measure_quality
from ml.preprocessing.transforms import PREPROCESSING_VERSION, to_model_input
from ml.retrieval.faiss_store import ReferenceIndex, summarize_matches
from ml.uncertainty.unknown_detector import UnknownCalibration, assess_unknown

MAX_CLASS_PROBABILITIES = 5


@dataclass(frozen=True)
class ArtifactPaths:
    encoder_dir: Path
    classifier_dir: Path
    index_path: Path
    index_metadata_path: Path
    unknown_calibration_path: Path
    quality_bounds_path: Path
    decision_policy_path: Path

    @classmethod
    def from_layout(cls, model_dir: Path, index_path: Path, index_metadata_path: Path) -> ArtifactPaths:
        return cls(
            encoder_dir=model_dir / "pretrained" / "dinov2_vits14",
            classifier_dir=model_dir / "classifiers",
            index_path=index_path,
            index_metadata_path=index_metadata_path,
            unknown_calibration_path=model_dir / "classifiers" / "calibration.json",
            quality_bounds_path=model_dir / "configs" / "quality-v1.json",
            decision_policy_path=model_dir / "configs" / "decision-v1.json",
        )

    def missing(self) -> list[str]:
        required = [
            self.encoder_dir / "config.json",
            self.encoder_dir / "SOURCE.json",
            self.classifier_dir / "classifier.joblib",
            self.classifier_dir / "label_encoder.json",
            self.classifier_dir / "training_config.json",
            self.index_path,
            self.index_metadata_path,
            self.unknown_calibration_path,
            self.quality_bounds_path,
            self.decision_policy_path,
        ]
        return [str(path) for path in required if not path.is_file()]


@dataclass(frozen=True)
class ModelInfo:
    encoder: str
    embedding_fingerprint: str
    preprocessing_version: str
    classifier_version: str
    index_version: str
    unknown_calibration_version: str
    quality_version: str
    policy_version: str
    classes: tuple[str, ...]
    reference_count: int


@dataclass(frozen=True)
class ScreeningResult:
    evidence: Evidence
    decision: Decision
    explanation: str


class ArtifactMismatchError(RuntimeError):
    """Raised when artifacts were produced for different embedding spaces or indexes."""


def explain(evidence: Evidence, decision: Decision) -> str:
    """Deterministic plain-language explanation built only from structured results."""
    headline = {
        "PRELIMINARY_PASS": "The visual evidence supports a preliminary screening pass.",
        "REVIEW_REQUIRED": "The visual evidence is not strong or consistent enough; expert review is required.",
        "UNKNOWN": "The sample is not sufficiently similar to the reference library to screen it.",
    }[decision.status]
    return f"{headline} {evidence.summary} {decision.reason}"


class Encoder(Protocol):
    """What the pipeline needs from a visual encoder (Dinov2Encoder in production)."""

    name: str
    fingerprint: str

    def encode(self, batch: np.ndarray) -> np.ndarray: ...


class ScreeningPipeline:
    def __init__(
        self,
        encoder: Encoder,
        classifier: EmbeddingClassifier,
        index: ReferenceIndex,
        unknown_calibration: UnknownCalibration,
        quality_bounds: QualityBounds,
        policy: DecisionPolicy,
    ) -> None:
        self.encoder = encoder
        self.classifier = classifier
        self.index = index
        self.unknown_calibration = unknown_calibration
        self.quality_bounds = quality_bounds
        self.policy = policy
        self._verify_consistency()
        self.model_info = ModelInfo(
            encoder=self.encoder.name,
            embedding_fingerprint=self.encoder.fingerprint,
            preprocessing_version=PREPROCESSING_VERSION,
            classifier_version=self.classifier.version,
            index_version=self.index.version,
            unknown_calibration_version=self.unknown_calibration.version,
            quality_version=self.quality_bounds.version,
            policy_version=self.policy.version,
            classes=tuple(self.classifier.classes),
            reference_count=len(self.index.records),
        )

    @classmethod
    def load(cls, artifacts: ArtifactPaths) -> ScreeningPipeline:
        """Load every artifact once (the API calls this at startup)."""
        missing = artifacts.missing()
        if missing:
            raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing))
        return cls(
            encoder=Dinov2Encoder(artifacts.encoder_dir),
            classifier=EmbeddingClassifier.load(artifacts.classifier_dir),
            index=ReferenceIndex.load(artifacts.index_path, artifacts.index_metadata_path),
            unknown_calibration=UnknownCalibration.load(artifacts.unknown_calibration_path),
            quality_bounds=QualityBounds.load(artifacts.quality_bounds_path),
            policy=DecisionPolicy.load(artifacts.decision_policy_path),
        )

    def _verify_consistency(self) -> None:
        fingerprints = {
            "encoder": self.encoder.fingerprint,
            "classifier": self.classifier.fingerprint,
            "index": self.index.fingerprint,
            "unknown calibration": self.unknown_calibration.embedding_fingerprint,
        }
        if len(set(fingerprints.values())) != 1:
            raise ArtifactMismatchError(f"Embedding fingerprints differ: {fingerprints}")
        if self.unknown_calibration.index_version != self.index.version:
            raise ArtifactMismatchError(
                f"Unknown calibration was fitted on index {self.unknown_calibration.index_version}, "
                f"but index {self.index.version} is loaded"
            )
        index_classes = {record.class_name for record in self.index.records}
        if not index_classes <= set(self.classifier.classes):
            raise ArtifactMismatchError("Reference index contains classes the classifier does not know")

    def assess_image_quality(self, decoded: DecodedImage) -> QualityAssessment:
        return assess_quality(
            measure_quality(decoded.image, self.quality_bounds.tile_grid), self.quality_bounds
        )

    def embed(self, decoded: DecodedImage) -> np.ndarray:
        return self.encoder.encode(to_model_input(decoded.image)[None, ...])[0]

    def analyze_embedding(self, embedding: np.ndarray, quality: QualityAssessment) -> ScreeningResult:
        return self.analyze_embeddings(embedding[None, :], [quality])[0]

    def analyze_embeddings(
        self, embeddings: np.ndarray, qualities: list[QualityAssessment]
    ) -> list[ScreeningResult]:
        k = self.unknown_calibration.k
        predictions = self.classifier.predict_batch(embeddings, MAX_CLASS_PROBABILITIES)
        match_sets = self.index.search(embeddings, k)
        results = []
        for prediction, matches, quality in zip(predictions, match_sets, qualities, strict=True):
            retrieval = summarize_matches(matches)
            similarities = np.array([match.similarity for match in matches])
            unknown = assess_unknown(similarities, self.unknown_calibration)
            evidence = build_evidence(prediction, retrieval, unknown, quality)
            decision = decide(evidence, self.policy)
            results.append(ScreeningResult(evidence, decision, explain(evidence, decision)))
        return results

    def analyze_image(self, decoded: DecodedImage) -> ScreeningResult:
        return self.analyze_embedding(self.embed(decoded), self.assess_image_quality(decoded))

"""Upload validation -> screening pipeline -> API response -> persistence."""

from __future__ import annotations

import io
import threading
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime

from app.core.logging import get_logger
from app.schemas import analysis as schema
from app.services.analysis_store import AnalysisStore
from app.services.reference_service import reference_image_url
from ml.inference.screening_pipeline import ModelInfo, ScreeningPipeline, ScreeningResult
from ml.preprocessing.image_io import DecodedImage, InvalidImageError, decode_image

DISCLAIMER = (
    "HerbaScope provides preliminary visual screening support and does not replace laboratory "
    "confirmation or expert botanical authentication."
)
ALLOWED_CONTENT_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/tiff", "image/bmp"})


def to_model_info(info: ModelInfo) -> schema.ModelInfo:
    return schema.ModelInfo(**{**asdict(info), "classes": list(info.classes)})


class UploadRejectedError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def build_limitations(pipeline: ScreeningPipeline) -> list[str]:
    """Scope statements derived from the loaded artifacts, so they stay true when artifacts change."""
    records = pipeline.index.records
    datasets = ", ".join(sorted({record.dataset for record in records}))
    fragments = "; ".join(sorted({record.fragment_type for record in records}))
    return [
        f"Supported classes: {', '.join(pipeline.classifier.classes)} ({datasets} reference library). Any other "
        "material can only be flagged as not matching the references, never identified.",
        f"Conclusions only transfer to fragment types represented in the reference library: {fragments}.",
        pipeline.unknown_calibration.limitation,
        "The classifier and reference retrieval are complementary analyses of the same visual representation, "
        "not independent evidence.",
        "This is reference-consistency screening, not adulteration detection: the reference data contains no "
        "adulterated or substitute material.",
    ]


class InferenceService:
    def __init__(self, pipeline: ScreeningPipeline, store: AnalysisStore, max_upload_bytes: int) -> None:
        self.pipeline = pipeline
        self._store = store
        self._max_upload_bytes = max_upload_bytes
        self._limitations = build_limitations(pipeline)
        # The encoder saturates the CPU on its own; serialising requests avoids thread oversubscription.
        self._lock = threading.Lock()

    def validate(self, filename: str, content_type: str | None, data: bytes) -> DecodedImage:
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise UploadRejectedError(
                415, f"Unsupported media type {content_type!r}; upload PNG, JPEG, WEBP, TIFF or BMP."
            )
        if len(data) > self._max_upload_bytes:
            raise UploadRejectedError(
                413, f"{filename} exceeds the {self._max_upload_bytes // (1024 * 1024)} MB upload limit."
            )
        try:
            return decode_image(data)
        except InvalidImageError as exc:
            raise UploadRejectedError(422, str(exc)) from exc

    def analyze(self, filename: str, content_type: str | None, data: bytes) -> schema.AnalysisResponse:
        decoded = self.validate(filename, content_type, data)
        started = time.perf_counter()
        with self._lock:
            result = self.pipeline.analyze_image(decoded)
        analysis_id = uuid.uuid4().hex
        response = self._to_response(analysis_id, filename, decoded, result)
        self._store.save(response, _normalised_png(decoded))
        get_logger().info(
            "analysis id=%s decision=%s class=%s confidence=%.3f seconds=%.3f",
            analysis_id,
            response.decision.status,
            response.prediction.class_name,
            response.prediction.confidence,
            time.perf_counter() - started,
        )
        return response

    def _to_response(
        self, analysis_id: str, filename: str, decoded: DecodedImage, result: ScreeningResult
    ) -> schema.AnalysisResponse:
        evidence, decision = result.evidence, result.decision
        info = self.pipeline.model_info
        return schema.AnalysisResponse(
            id=analysis_id,
            created_at=datetime.now(UTC).isoformat(timespec="milliseconds"),
            sample=schema.Sample(
                filename=filename,
                format=decoded.format,
                width=decoded.width,
                height=decoded.height,
                image_url=f"/analyses/{analysis_id}/image",
            ),
            prediction=schema.Prediction(
                class_name=evidence.prediction.class_name,
                confidence=evidence.prediction.confidence,
                top_k=[schema.ClassProbability(**asdict(item)) for item in evidence.prediction.top_k],
            ),
            retrieval=schema.Retrieval(
                matches=[
                    schema.ReferenceMatch(
                        rank=match.rank,
                        reference_id=match.reference.reference_id,
                        class_name=match.reference.class_name,
                        fragment_type=match.reference.fragment_type,
                        dataset=match.reference.dataset,
                        similarity=match.similarity,
                        image_url=reference_image_url(match.reference),
                    )
                    for match in evidence.retrieval.matches
                ],
                top_similarity=evidence.retrieval.top_similarity,
                retrieved_class=evidence.retrieval.retrieved_class,
                class_support=[
                    schema.ClassSupport(**asdict(item)) for item in evidence.retrieval.class_support
                ],
            ),
            unknown=schema.Unknown(**asdict(evidence.unknown)),
            evidence=schema.Evidence(
                agreement=evidence.agreement,
                strength=evidence.strength,
                strength_components=schema.StrengthComponents(**asdict(evidence.components)),
                quality=schema.Quality(**evidence.quality.to_dict()),
                summary=evidence.summary,
            ),
            decision=schema.Decision(
                status=decision.status,
                reason=decision.reason,
                policy_version=decision.policy_version,
                thresholds=schema.DecisionThresholds(
                    min_classifier_confidence=self.pipeline.policy.min_classifier_confidence,
                    min_reference_similarity=self.pipeline.policy.min_reference_similarity,
                    max_unknown_risk=self.pipeline.policy.max_unknown_risk,
                ),
                checks=[schema.DecisionCheck(**asdict(check)) for check in decision.checks],
            ),
            explanation=result.explanation,
            model=to_model_info(info),
            limitations=self._limitations,
            disclaimer=DISCLAIMER,
        )


def _normalised_png(decoded: DecodedImage) -> bytes:
    """Re-encode the decoded pixels so stored samples never contain the raw uploaded bytes."""
    buffer = io.BytesIO()
    image = decoded.image if decoded.image.mode in {"L", "RGB", "RGBA"} else decoded.image.convert("RGB")
    image.save(buffer, format="PNG")
    return buffer.getvalue()

"""Response contracts. The `prediction / retrieval / unknown / evidence / decision / model` blocks
follow the frozen HerbaScope schema; other fields add transparency without changing it."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

UnknownStatus = Literal["KNOWN", "UNCERTAIN", "UNKNOWN"]
Agreement = Literal["HIGH", "MEDIUM", "LOW"]
DecisionStatus = Literal["PRELIMINARY_PASS", "REVIEW_REQUIRED", "UNKNOWN"]
QualityStatus = Literal["ACCEPTABLE", "DEGRADED"]


class ClassProbability(BaseModel):
    class_name: str
    probability: float = Field(ge=0, le=1)


class Prediction(BaseModel):
    class_name: str
    confidence: float = Field(ge=0, le=1)
    top_k: list[ClassProbability]


class ReferenceMatch(BaseModel):
    rank: int
    reference_id: str
    class_name: str
    fragment_type: str
    dataset: str
    similarity: float
    image_url: str


class ClassSupport(BaseModel):
    class_name: str
    count: int
    weighted_share: float


class Retrieval(BaseModel):
    matches: list[ReferenceMatch]
    top_similarity: float
    retrieved_class: str
    class_support: list[ClassSupport]


class Unknown(BaseModel):
    risk: float = Field(ge=0, le=1)
    distance: float
    status: UnknownStatus
    threshold: float
    known_boundary: float
    calibration_version: str


class QualityCheck(BaseModel):
    name: str
    label: str
    passed: bool
    value: float
    bound: float
    comparison: Literal[">=", "<="]


class Quality(BaseModel):
    status: QualityStatus
    passed_fraction: float
    checks: list[QualityCheck]
    version: str


class StrengthComponents(BaseModel):
    classifier_confidence: float
    reference_support: float
    unknown_margin: float
    quality_pass_fraction: float


class Evidence(BaseModel):
    agreement: Agreement
    strength: float = Field(ge=0, le=1)
    strength_components: StrengthComponents
    quality: Quality
    summary: str


class DecisionCheck(BaseModel):
    name: str
    label: str
    passed: bool
    observed: str
    required: str


class DecisionThresholds(BaseModel):
    min_classifier_confidence: float
    min_reference_similarity: float
    max_unknown_risk: float


class Decision(BaseModel):
    status: DecisionStatus
    reason: str
    policy_version: str
    thresholds: DecisionThresholds
    checks: list[DecisionCheck]


class ModelInfo(BaseModel):
    encoder: str
    classifier_version: str
    index_version: str
    unknown_calibration_version: str
    quality_version: str
    policy_version: str
    preprocessing_version: str
    embedding_fingerprint: str
    classes: list[str]
    reference_count: int


class Sample(BaseModel):
    filename: str
    format: str
    width: int
    height: int
    image_url: str


class AnalysisResponse(BaseModel):
    id: str
    created_at: str
    sample: Sample
    prediction: Prediction
    retrieval: Retrieval
    unknown: Unknown
    evidence: Evidence
    decision: Decision
    explanation: str
    model: ModelInfo
    limitations: list[str]
    disclaimer: str


class AnalysisSummary(BaseModel):
    id: str
    created_at: str
    filename: str
    decision_status: DecisionStatus
    predicted_class: str
    confidence: float
    image_url: str


class AnalysisList(BaseModel):
    items: list[AnalysisSummary]


class Reference(BaseModel):
    reference_id: str
    image_id: str
    class_name: str
    fragment_type: str
    dataset: str
    source: str
    index_version: str
    image_url: str


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    api_version: str
    model_loaded: bool
    index_loaded: bool
    reference_count: int
    max_upload_mb: float
    model: ModelInfo | None
    detail: str | None


class ErrorResponse(BaseModel):
    detail: str

// Mirrors services/api/app/schemas/analysis.py (frozen /api/v1 contract).

export type UnknownStatus = "KNOWN" | "UNCERTAIN" | "UNKNOWN";
export type Agreement = "HIGH" | "MEDIUM" | "LOW";
export type DecisionStatus = "PRELIMINARY_PASS" | "REVIEW_REQUIRED" | "UNKNOWN";
export type QualityStatus = "ACCEPTABLE" | "DEGRADED";

export interface ClassProbability {
  class_name: string;
  probability: number;
}

export interface Prediction {
  class_name: string;
  confidence: number;
  top_k: ClassProbability[];
}

export interface ReferenceMatch {
  rank: number;
  reference_id: string;
  class_name: string;
  fragment_type: string;
  dataset: string;
  similarity: number;
  /** Path relative to the API base URL. */
  image_url: string;
}

export interface ClassSupport {
  class_name: string;
  count: number;
  weighted_share: number;
}

export interface Retrieval {
  matches: ReferenceMatch[];
  top_similarity: number;
  retrieved_class: string;
  class_support: ClassSupport[];
}

export interface Unknown {
  risk: number;
  distance: number;
  status: UnknownStatus;
  threshold: number;
  known_boundary: number;
  calibration_version: string;
}

export interface QualityCheck {
  name: string;
  label: string;
  passed: boolean;
  value: number;
  bound: number;
  comparison: ">=" | "<=";
}

export interface Quality {
  status: QualityStatus;
  passed_fraction: number;
  checks: QualityCheck[];
  version: string;
}

export interface StrengthComponents {
  classifier_confidence: number;
  reference_support: number;
  unknown_margin: number;
  quality_pass_fraction: number;
}

export interface Evidence {
  agreement: Agreement;
  strength: number;
  strength_components: StrengthComponents;
  quality: Quality;
  summary: string;
}

export interface DecisionCheck {
  name: string;
  label: string;
  passed: boolean;
  observed: string;
  required: string;
}

export interface DecisionThresholds {
  min_classifier_confidence: number;
  min_reference_similarity: number;
  max_unknown_risk: number;
}

export interface Decision {
  status: DecisionStatus;
  reason: string;
  policy_version: string;
  thresholds: DecisionThresholds;
  checks: DecisionCheck[];
}

export interface ModelInfo {
  encoder: string;
  classifier_version: string;
  index_version: string;
  unknown_calibration_version: string;
  quality_version: string;
  policy_version: string;
  preprocessing_version: string;
  embedding_fingerprint: string;
  classes: string[];
  reference_count: number;
}

export interface Sample {
  filename: string;
  format: string;
  width: number;
  height: number;
  image_url: string;
}

export interface AnalysisResponse {
  id: string;
  created_at: string;
  sample: Sample;
  prediction: Prediction;
  retrieval: Retrieval;
  unknown: Unknown;
  evidence: Evidence;
  decision: Decision;
  explanation: string;
  model: ModelInfo;
  limitations: string[];
  disclaimer: string;
}

export interface AnalysisSummary {
  id: string;
  created_at: string;
  filename: string;
  decision_status: DecisionStatus;
  predicted_class: string;
  confidence: number;
  image_url: string;
}

export interface AnalysisList {
  items: AnalysisSummary[];
}

export interface Health {
  status: "ok" | "degraded";
  api_version: string;
  model_loaded: boolean;
  index_loaded: boolean;
  reference_count: number;
  max_upload_mb: number;
  model: ModelInfo | null;
  detail: string | null;
}

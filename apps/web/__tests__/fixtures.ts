// Real API responses captured from a running HerbaScope API (see __tests__/fixtures/*.json).
import analysesJson from "./fixtures/analyses.json";
import reviewJson from "./fixtures/analysis-review.json";
import healthJson from "./fixtures/health.json";

import type { AnalysisList, AnalysisResponse, Health } from "@/types";

export const reviewAnalysis = reviewJson as AnalysisResponse;
export const health = healthJson as Health;
export const analyses = analysesJson as AnalysisList;

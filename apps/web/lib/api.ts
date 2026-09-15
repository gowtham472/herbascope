import type { AnalysisList, AnalysisResponse, Health } from "@/types";

export const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1").replace(
  /\/$/,
  "",
);

/** An API failure with the HTTP status (0 when the API could not be reached at all). */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Absolute URL for an API-relative path such as `/reference/REF0001/image`. */
export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

function detailMessage(body: unknown, status: number): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail.map((item) => (item && typeof item === "object" && "msg" in item ? item.msg : String(item))).join("; ");
    }
  }
  return `Request failed with HTTP ${status}.`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiUrl(path), { ...init, cache: "no-store" });
  } catch {
    throw new ApiError(0, `Cannot reach the HerbaScope X API at ${API_BASE_URL}. Check that the API is running.`);
  }
  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(response.status, detailMessage(body, response.status));
  }
  return body as T;
}

export function analyzeImage(file: File): Promise<AnalysisResponse> {
  const form = new FormData();
  form.append("image", file);
  return request<AnalysisResponse>("/analyze", { method: "POST", body: form });
}

export function getAnalysis(id: string): Promise<AnalysisResponse> {
  return request<AnalysisResponse>(`/analyses/${encodeURIComponent(id)}`);
}

export function listAnalyses(): Promise<AnalysisList> {
  return request<AnalysisList>("/analyses?limit=100");
}

export function getHealth(): Promise<Health> {
  return request<Health>("/health");
}

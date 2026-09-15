import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AnalyzeView } from "@/components/upload/AnalyzeView";
import { analyzeImage, ApiError, getHealth } from "@/lib/api";

import { health, reviewAnalysis } from "./fixtures";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));
vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  getHealth: vi.fn(),
  analyzeImage: vi.fn(),
}));

const sample = new File([new Uint8Array(64)], "sample.png", { type: "image/png" });

beforeEach(() => {
  vi.mocked(getHealth).mockReset().mockResolvedValue(health);
  vi.mocked(analyzeImage).mockReset();
  push.mockReset();
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
});

async function selectSample() {
  const input = await screen.findByLabelText(/upload microscopic sample/i);
  await userEvent.upload(input, sample);
}

describe("AnalyzeView", () => {
  it("shows the selected sample and supported reference library", async () => {
    render(<AnalyzeView />);
    await selectSample();
    expect(screen.getByRole("img", { name: /selected sample sample\.png/i })).toBeInTheDocument();
    expect(screen.getByText(/sirih \/ sirih merah/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /run screening/i })).toBeEnabled();
  });

  it("shows progress while analysing and opens the result on success", async () => {
    let resolve!: (value: typeof reviewAnalysis) => void;
    vi.mocked(analyzeImage).mockReturnValue(new Promise((r) => (resolve = r)));
    render(<AnalyzeView />);
    await selectSample();
    await userEvent.click(screen.getByRole("button", { name: /run screening/i }));

    expect(screen.getByRole("status")).toHaveTextContent("Analyzing sample…");
    expect(screen.getByRole("status")).toHaveTextContent(`Encode every rotation with ${health.model?.encoder}`);
    expect(screen.getByRole("button", { name: /analyzing/i })).toBeDisabled();

    resolve(reviewAnalysis);
    await waitFor(() => expect(push).toHaveBeenCalledWith(`/results/${reviewAnalysis.id}`));
    expect(analyzeImage).toHaveBeenCalledWith(sample);
  });

  it("shows the API error message when analysis is rejected", async () => {
    vi.mocked(analyzeImage).mockRejectedValue(new ApiError(422, "The file is not a readable image."));
    render(<AnalyzeView />);
    await selectSample();
    await userEvent.click(screen.getByRole("button", { name: /run screening/i }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("The image could not be analysed");
    expect(alert).toHaveTextContent("The file is not a readable image.");
    expect(push).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /run screening/i })).toBeEnabled();
  });

  it("explains when the API cannot be reached", async () => {
    vi.mocked(getHealth).mockRejectedValue(new ApiError(0, "Cannot reach the HerbaScope X API."));
    render(<AnalyzeView />);
    expect(await screen.findByRole("alert")).toHaveTextContent("The screening API is unavailable");
    expect(screen.queryByLabelText(/upload microscopic sample/i)).not.toBeInTheDocument();
  });

  it("blocks uploads when the API runs without model artifacts", async () => {
    vi.mocked(getHealth).mockResolvedValue({ ...health, status: "degraded", model: null, detail: "Missing model artifacts" });
    render(<AnalyzeView />);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Models are not loaded");
    expect(alert).toHaveTextContent("Missing model artifacts");
  });
});

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ResultView } from "@/components/report/ResultView";
import { ApiError, getAnalysis } from "@/lib/api";

import { reviewAnalysis } from "./fixtures";

vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  getAnalysis: vi.fn(),
}));

beforeEach(() => {
  vi.mocked(getAnalysis).mockReset();
});

describe("ResultView", () => {
  it("shows a loading state before the result arrives", () => {
    vi.mocked(getAnalysis).mockReturnValue(new Promise(() => {}));
    render(<ResultView id={reviewAnalysis.id} />);
    expect(screen.getByRole("status", { name: /loading screening result/i })).toBeInTheDocument();
  });

  it("renders every result section from the API response", async () => {
    vi.mocked(getAnalysis).mockResolvedValue(reviewAnalysis);
    render(<ResultView id={reviewAnalysis.id} />);

    expect(await screen.findByRole("heading", { name: reviewAnalysis.sample.filename })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /uploaded sample/i })).toBeInTheDocument();

    const decision = screen.getByRole("region", { name: /screening decision/i });
    expect(decision).toHaveTextContent("Review required");
    expect(decision).toHaveTextContent(reviewAnalysis.decision.reason);
    expect(decision).toHaveTextContent("decision-v1");
    const agreementRow = within(decision).getByRole("row", { name: /evidence agreement/i });
    expect(agreementRow).toHaveTextContent("No");

    const prediction = screen.getByRole("region", { name: /model prediction/i });
    expect(prediction).toHaveTextContent("98.2%");
    expect(within(prediction).getAllByRole("meter")[0]).toHaveAttribute("aria-valuetext", expect.stringMatching(/policy minimum/i));

    const references = screen.getByRole("region", { name: /reference evidence/i });
    expect(within(references).getAllByRole("img")).toHaveLength(reviewAnalysis.retrieval.matches.length);
    expect(references).toHaveTextContent("0.763");

    expect(screen.getByRole("region", { name: /unknown risk/i })).toHaveTextContent("KNOWN");
    expect(screen.getByRole("region", { name: /evidence agreement/i })).toHaveTextContent("LOW");
    expect(screen.getByRole("region", { name: /image quality/i })).toHaveTextContent("ACCEPTABLE");
    expect(screen.getByRole("region", { name: /model metadata/i })).toHaveTextContent("DINOv2 ViT-S/14");
    expect(screen.getByRole("region", { name: /limitations/i })).toHaveTextContent(reviewAnalysis.disclaimer);
    expect(screen.getByRole("region", { name: /screening summary/i })).toHaveTextContent(reviewAnalysis.explanation);
  });

  it("compares a reference with the sample side by side", async () => {
    vi.mocked(getAnalysis).mockResolvedValue(reviewAnalysis);
    render(<ResultView id={reviewAnalysis.id} />);
    const first = reviewAnalysis.retrieval.matches[0];

    await userEvent.click(await screen.findByRole("button", { name: new RegExp(`compare the sample with reference ${first.reference_id}`, "i") }));

    const dialog = screen.getByRole("dialog", { name: /sample and reference comparison/i });
    expect(within(dialog).getByRole("heading", { name: new RegExp(first.reference_id) })).toBeInTheDocument();
    expect(within(dialog).getByRole("img", { name: /uploaded sample/i })).toBeInTheDocument();
    expect(within(dialog).getByRole("img", { name: new RegExp(first.reference_id) })).toBeInTheDocument();

    await userEvent.click(within(dialog).getByRole("button", { name: /close comparison/i }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("reports a missing result without offering a pointless retry", async () => {
    vi.mocked(getAnalysis).mockRejectedValue(new ApiError(404, "Analysis not found"));
    render(<ResultView id="0123456789abcdef0123456789abcdef" />);
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Screening result not found");
    expect(within(alert).queryByRole("button", { name: /try again/i })).not.toBeInTheDocument();
    expect(within(alert).getByRole("link", { name: /new analysis/i })).toHaveAttribute("href", "/analyze");
  });

  it("retries after a network error", async () => {
    vi.mocked(getAnalysis).mockRejectedValueOnce(new ApiError(0, "Cannot reach the API")).mockResolvedValueOnce(reviewAnalysis);
    render(<ResultView id={reviewAnalysis.id} />);
    await userEvent.click(await screen.findByRole("button", { name: /try again/i }));
    expect(await screen.findByRole("heading", { name: reviewAnalysis.sample.filename })).toBeInTheDocument();
    expect(getAnalysis).toHaveBeenCalledTimes(2);
  });
});

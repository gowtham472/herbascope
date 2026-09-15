import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { HistoryView } from "@/components/history/HistoryView";
import { ApiError, listAnalyses } from "@/lib/api";

import { analyses } from "./fixtures";

vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  listAnalyses: vi.fn(),
}));

beforeEach(() => {
  vi.mocked(listAnalyses).mockReset();
});

describe("HistoryView", () => {
  it("links every stored analysis to its result page", async () => {
    vi.mocked(listAnalyses).mockResolvedValue(analyses);
    render(<HistoryView />);
    const item = analyses.items[0];
    const link = await screen.findByRole("link", { name: new RegExp(item.filename) });
    expect(link).toHaveAttribute("href", `/results/${item.id}`);
    expect(link).toHaveTextContent("Review required");
  });

  it("shows an empty state when nothing has been analysed", async () => {
    vi.mocked(listAnalyses).mockResolvedValue({ items: [] });
    render(<HistoryView />);
    expect(await screen.findByText("No analyses yet")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /analyze a sample/i })).toHaveAttribute("href", "/analyze");
  });

  it("shows an error state when the API fails", async () => {
    vi.mocked(listAnalyses).mockRejectedValue(new ApiError(0, "Cannot reach the HerbaScope X API."));
    render(<HistoryView />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load analysis history");
  });
});

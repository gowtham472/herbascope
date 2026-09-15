import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { HistoryView } from "@/components/history/HistoryView";
import { ApiError, listAnalyses } from "@/lib/api";
import { DECISION_LABEL, percent } from "@/lib/format";

import { analyses } from "./fixtures";

vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  listAnalyses: vi.fn(),
}));

beforeEach(() => {
  vi.mocked(listAnalyses).mockReset();
});

describe("HistoryView", () => {
  it("links every stored analysis to its result page with its decision", async () => {
    vi.mocked(listAnalyses).mockResolvedValue(analyses);
    render(<HistoryView />);
    await screen.findByRole("link", { name: new RegExp(analyses.items[0].filename) });

    for (const item of analyses.items) {
      const link = screen.getByRole("link", { name: new RegExp(item.filename) });
      expect(link).toHaveAttribute("href", `/results/${item.id}`);
      expect(link).toHaveTextContent(DECISION_LABEL[item.decision_status]);
      expect(link).toHaveTextContent(percent(item.confidence));
    }
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

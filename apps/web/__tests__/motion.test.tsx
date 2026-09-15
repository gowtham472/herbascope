import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ConfidenceBar } from "@/components/analysis/ConfidenceBar";
import { AnimatedMeter } from "@/components/motion/AnimatedMeter";
import { Reveal } from "@/components/motion/Reveal";

/** The suite runs with "prefers-reduced-motion: reduce"; this turns animations on. */
function enableAnimation() {
  vi.stubGlobal(
    "matchMedia",
    vi.fn((query: string) => ({ matches: false, media: query, addEventListener: vi.fn(), removeEventListener: vi.fn() })),
  );
}

describe("animation never hides a measured value", () => {
  it("renders the meter at the measured width, not at zero", () => {
    render(<AnimatedMeter value={0.8312} className="bar" />);
    expect(document.querySelector(".meter-fill")).toHaveStyle({ width: "83.12%" });
  });

  it("prints the exact value next to the bar and exposes it to assistive technology", () => {
    render(
      <ConfidenceBar
        label="Classifier confidence"
        value={0.8312}
        valueText="83.1%"
        threshold={0.6267}
        thresholdLabel="Policy minimum 62.7% (calibrated)"
      />,
    );
    const meter = screen.getByRole("meter", { name: /classifier confidence/i });
    expect(meter).toHaveAttribute("aria-valuenow", "0.8312");
    expect(meter).toHaveAttribute("aria-valuetext", "83.1%; Policy minimum 62.7% (calibrated)");
    expect(screen.getByText("83.1%")).toBeInTheDocument();
  });
});

describe("scroll reveal", () => {
  it("leaves content visible when the reader prefers reduced motion", () => {
    render(<Reveal>reference evidence</Reveal>);
    const block = screen.getByText("reference evidence");
    expect(block).toBeInTheDocument();
    expect(block).not.toHaveClass("reveal-pending");
  });

  it("hides the block only until it enters the viewport", () => {
    enableAnimation();
    render(<Reveal index={2}>reference evidence</Reveal>);
    const block = screen.getByText("reference evidence");
    // The stubbed observer reports the block as visible immediately.
    expect(block).toHaveClass("reveal-run");
    expect(block).not.toHaveClass("reveal-pending");
    expect(block.style.getPropertyValue("--reveal-delay")).toBe("0.1s");
    vi.unstubAllGlobals();
  });
});

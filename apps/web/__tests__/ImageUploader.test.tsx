import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ImageUploader } from "@/components/upload/ImageUploader";

function file(name: string, type: string, bytes: number) {
  return new File([new Uint8Array(bytes)], name, { type });
}

describe("ImageUploader", () => {
  it("accepts a supported image within the size limit", async () => {
    const onSelect = vi.fn();
    render(<ImageUploader maxBytes={1024} onSelect={onSelect} />);
    const sample = file("sample.png", "image/png", 100);
    await userEvent.upload(screen.getByLabelText(/upload microscopic sample/i), sample);
    expect(onSelect).toHaveBeenCalledWith(sample);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("rejects unsupported file types without calling onSelect", async () => {
    const onSelect = vi.fn();
    render(<ImageUploader maxBytes={1024} onSelect={onSelect} />);
    await userEvent.upload(screen.getByLabelText(/upload microscopic sample/i), file("notes.pdf", "application/pdf", 10), {
      applyAccept: false,
    });
    expect(onSelect).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("not a supported image");
  });

  it("rejects files above the API upload limit", async () => {
    const onSelect = vi.fn();
    render(<ImageUploader maxBytes={1024} onSelect={onSelect} />);
    await userEvent.upload(screen.getByLabelText(/upload microscopic sample/i), file("big.png", "image/png", 4096));
    expect(onSelect).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("the limit is 1.0 KB");
  });

  it("is keyboard reachable through a labelled file input", async () => {
    render(<ImageUploader maxBytes={1024} onSelect={vi.fn()} />);
    await userEvent.tab();
    expect(screen.getByLabelText(/upload microscopic sample/i)).toHaveFocus();
  });
});

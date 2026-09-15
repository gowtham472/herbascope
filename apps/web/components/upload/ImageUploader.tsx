"use client";

import { UploadSimpleIcon } from "@phosphor-icons/react/ssr";
import { type ChangeEvent, type DragEvent, useId, useState } from "react";

import { formatBytes } from "@/lib/format";

const ACCEPTED_TYPES =["image/png", "image/jpeg", "image/webp", "image/tiff", "image/bmp"];

interface ImageUploaderProps {
  maxBytes: number;
  disabled?: boolean;
  onSelect: (file: File) => void;
}

/** Drag-and-drop or keyboard file picker. Checks type and size before anything is uploaded. */
export function ImageUploader({ maxBytes, disabled = false, onSelect }: ImageUploaderProps) {
  const inputId = useId();
  const hintId = useId();
  const [dragging, setDragging] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  function accept(file: File | undefined) {
    if (!file) return;
    if (!ACCEPTED_TYPES.includes(file.type)) {
      setProblem(`"${file.name}" is not a supported image. Use PNG, JPEG, WEBP, TIFF or BMP.`);
      return;
    }
    if (file.size > maxBytes) {
      setProblem(`"${file.name}" is ${formatBytes(file.size)}; the limit is ${formatBytes(maxBytes)}.`);
      return;
    }
    setProblem(null);
    onSelect(file);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    if (!disabled) accept(event.dataTransfer.files[0]);
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    accept(event.target.files?.[0]);
    event.target.value = "";
  }

  return (
    <div>
      <label
        htmlFor={inputId}
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-brand-600 ${
          dragging ? "border-brand-500 bg-brand-50" : "border-line bg-surface hover:border-brand-500/60 hover:bg-canvas"
        } ${disabled ? "pointer-events-none opacity-60" : ""}`}
      >
        <span className="grid size-12 place-items-center rounded-full bg-brand-50 text-brand-700">
          <UploadSimpleIcon aria-hidden="true" weight="bold" className="size-6" />
        </span>
        <span className="mt-3 font-semibold text-ink">Upload microscopic sample</span>
        <span id={hintId} className="mt-1 text-sm text-muted">
          Drag an image here or choose a file · PNG, JPEG, WEBP, TIFF, BMP · up to {formatBytes(maxBytes)}
        </span>
        <input
          id={inputId}
          type="file"
          accept={ACCEPTED_TYPES.join(",")}
          aria-describedby={hintId}
          disabled={disabled}
          onChange={handleChange}
          className="sr-only"
        />
      </label>
      {problem ? (
        <p role="alert" className="mt-2 text-sm font-medium text-fail-600">
          {problem}
        </p>
      ) : null}
    </div>
  );
}

"use client";

import { ArrowsLeftRightIcon, XIcon } from "@phosphor-icons/react/ssr";
import Image from "next/image";
import { useEffect, useRef } from "react";

import { apiUrl } from "@/lib/api";
import { classLabel, decimal, sentenceCase } from "@/lib/format";
import type { ReferenceMatch } from "@/types";

interface ReferenceViewerProps {
  match: ReferenceMatch | null;
  /** The uploaded sample, shown beside the reference for comparison. */
  sample: { imageUrl: string; filename: string };
  onClose: () => void;
}

/**
 * Side-by-side comparison of the uploaded sample and one retrieved reference micrograph.
 * Retrieval evidence is only useful if a reviewer can actually look at it, and the small
 * cards in the grid are too small to judge cell walls or crystals.
 *
 * Built on the native `<dialog>` element, which supplies the focus trap, the backdrop and
 * dismissal with Escape without a dependency.
 */
export function ReferenceViewer({ match, sample, onClose }: ReferenceViewerProps) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (match && !dialog.open) dialog.showModal();
    if (!match && dialog.open) dialog.close();
  }, [match]);

  return (
    <dialog
      ref={ref}
      aria-label="Sample and reference comparison"
      onClose={onClose}
      onClick={(event) => {
        // The dialog element itself is the backdrop area around the panel.
        if (event.target === ref.current) onClose();
      }}
      className="dialog-panel m-auto w-[min(56rem,calc(100%-2rem))] max-h-[92dvh] overflow-y-auto rounded-xl border border-line bg-surface p-0 text-ink shadow-xl backdrop:bg-ink/50"
    >
      {match ? (
        <div className="flex flex-col">
          <header className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
            <div>
              <h2 className="text-lg font-semibold">
                Sample compared with {match.reference_id}
              </h2>
              <p className="mt-0.5 text-sm text-muted">
                <span className="capitalize">{classLabel(match.class_name)}</span> ·{" "}
                {sentenceCase(match.fragment_type)} · cosine similarity{" "}
                <span className="font-mono tabular-nums">{decimal(match.similarity)}</span>
              </p>
            </div>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close comparison"
              className="press rounded-lg border border-line p-1.5 text-muted hover:bg-canvas focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
            >
              <XIcon aria-hidden="true" weight="bold" className="size-5" />
            </button>
          </header>

          <div className="grid gap-4 p-5 sm:grid-cols-[1fr_auto_1fr]">
            <figure className="m-0">
              <div className="relative aspect-square overflow-hidden rounded-lg bg-canvas">
                <Image src={sample.imageUrl} alt={`Uploaded sample ${sample.filename}`} fill unoptimized sizes="400px" className="object-contain" />
              </div>
              <figcaption className="mt-2 truncate text-sm text-muted" title={sample.filename}>
                Uploaded sample · {sample.filename}
              </figcaption>
            </figure>

            <div className="hidden items-center justify-center text-muted sm:flex">
              <ArrowsLeftRightIcon aria-hidden="true" weight="bold" className="size-5" />
            </div>

            <figure className="m-0">
              <div className="relative aspect-square overflow-hidden rounded-lg bg-canvas">
                <Image
                  src={apiUrl(match.image_url)}
                  alt={`Reference ${match.reference_id}: ${classLabel(match.class_name)}, ${match.fragment_type}`}
                  fill
                  unoptimized
                  sizes="400px"
                  className="object-contain"
                />
              </div>
              <figcaption className="mt-2 text-sm text-muted">
                Reference #{match.rank} · {match.dataset}
              </figcaption>
            </figure>
          </div>

          <p className="border-t border-line px-5 py-3 text-xs text-muted">
            Reference micrographs are training images of supported material. Visual similarity supports a screening
            decision; it does not authenticate the sample.
          </p>
        </div>
      ) : null}
    </dialog>
  );
}

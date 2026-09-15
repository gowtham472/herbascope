import Image from "next/image";
import type { ReactNode } from "react";

interface ImagePreviewProps {
  src: string;
  alt: string;
  details: [string, string][];
  children?: ReactNode;
}

/**
 * Sample image with its file details. Images come from the local API or a blob URL, so
 * Next image optimisation is bypassed (`unoptimized`): the optimiser's SSRF guard blocks
 * local addresses, and micrographs are already small.
 */
export function ImagePreview({ src, alt, details, children }: ImagePreviewProps) {
  return (
    <figure className="self-start overflow-hidden rounded-xl border border-line bg-surface shadow-sm break-inside-avoid">
      <div className="relative aspect-square bg-[repeating-conic-gradient(#eef1ee_0%_25%,#f8faf8_0%_50%)] bg-[length:20px_20px]">
        {/* The sample is the primary (LCP) image wherever this component is used. */}
        <Image
          src={src}
          alt={alt}
          fill
          unoptimized
          loading="eager"
          sizes="(min-width: 1024px) 360px, 100vw"
          className="object-contain"
        />
      </div>
      <figcaption className="border-t border-line p-4">
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
          {details.map(([label, value]) => (
            <div key={label} className="contents">
              <dt className="text-muted">{label}</dt>
              <dd className="truncate text-right font-medium text-ink" title={value}>
                {value}
              </dd>
            </div>
          ))}
        </dl>
        {children}
      </figcaption>
    </figure>
  );
}

import Image from "next/image";

/** Intrinsic sizes of the two exported files, so the layout reserves space before they load. */
const ARTWORK = {
  hero: { src: "/brand/mascot.webp", width: 1000, height: 1486, sizes: "(min-width: 1024px) 420px, 60vw" },
  spot: { src: "/brand/mascot-sm.webp", width: 420, height: 624, sizes: "180px" },
} as const;

interface MascotProps {
  /** `hero` renders the full-size artwork; `spot` the small version for inline use. */
  size?: "hero" | "spot";
  className?: string;
  /** Mascot artwork is decorative: give it a label only where it carries meaning. */
  alt?: string;
  priority?: boolean;
}

/**
 * Herbie, the HerbaScope mascot. Shipped as two pre-sized WebP files so the page never
 * downloads the 2 MB original.
 */
export function Mascot({ size = "spot", className = "", alt = "", priority = false }: MascotProps) {
  const artwork = ARTWORK[size];
  return (
    <Image
      src={artwork.src}
      width={artwork.width}
      height={artwork.height}
      sizes={artwork.sizes}
      alt={alt}
      aria-hidden={alt === "" ? true : undefined}
      priority={priority}
      className={className}
    />
  );
}

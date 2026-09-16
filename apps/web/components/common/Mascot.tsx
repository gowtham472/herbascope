import Image from "next/image";

/** Intrinsic sizes of the exported files, so the layout reserves space before they load. */
const ARTWORK = {
  hero: { src: "/brand/mascot.webp", width: 1000, height: 1486, sizes: "(min-width: 1024px) 420px, 60vw" },
  spot: { src: "/brand/mascot-sm.webp", width: 420, height: 624, sizes: "180px" },
  "no-sample": { src: "/brand/no-sample.webp", width: 760, height: 563, sizes: "(min-width: 640px) 300px, 70vw" },
} as const;

interface MascotProps {
  /** `hero` is the full-size artwork, `spot` the small one, `no-sample` the empty-state scene. */
  variant?: keyof typeof ARTWORK;
  className?: string;
  /** Mascot artwork is decorative: give it a label only where it carries meaning. */
  alt?: string;
  priority?: boolean;
}

/**
 * Herbie, the HerbaScope mascot. Each variant ships as a pre-sized WebP so a page never
 * downloads the multi-megabyte original artwork.
 */
export function Mascot({ variant = "spot", className = "", alt = "", priority = false }: MascotProps) {
  const artwork = ARTWORK[variant];
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

import Image from "next/image";

import mascot from "@/public/brand/mascot.webp";
import mascotSmall from "@/public/brand/mascot-sm.webp";

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
 * downloads a 2 MB illustration, and imported statically so Next knows its dimensions and
 * reserves the space before it loads.
 */
export function Mascot({ size = "spot", className = "", alt = "", priority = false }: MascotProps) {
  const source = size === "hero" ? mascot : mascotSmall;
  return (
    <Image
      src={source}
      alt={alt}
      aria-hidden={alt === "" ? true : undefined}
      priority={priority}
      sizes={size === "hero" ? "(min-width: 1024px) 420px, 60vw" : "180px"}
      className={className}
    />
  );
}

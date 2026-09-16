import Image from "next/image";

import wordmark from "@/public/brand/wordmark.svg";

/**
 * The supplied HerbaScope wordmark (leaf pair plus lettering), used as-is. The product
 * suffix "X" is set beside it rather than baked into the artwork.
 */
export function Wordmark({ className = "" }: { className?: string }) {
  return <Image src={wordmark} alt="HerbaScope" unoptimized priority className={className} />;
}

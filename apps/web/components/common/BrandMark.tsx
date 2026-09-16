import Image from "next/image";

import mark from "@/public/brand/mark.svg";

/**
 * The HerbaScope leaf mark, served from the supplied logo artwork
 * (`public/brand/mark.svg`, the leaf pair lifted out of the wordmark file with its original
 * gradients). Decorative: the wordmark beside it carries the name.
 */
export function BrandMark({ className = "" }: { className?: string }) {
  return <Image src={mark} alt="" aria-hidden="true" unoptimized className={className} />;
}

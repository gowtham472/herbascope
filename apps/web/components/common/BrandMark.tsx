import Image from "next/image";

/**
 * The HerbaScope leaf mark: the leaf pair lifted out of the supplied wordmark file with its
 * original gradients (`public/brand/mark.svg`). Decorative; a wordmark carries the name.
 */
export function BrandMark({ className = "" }: { className?: string }) {
  return <Image src="/brand/mark.svg" width={166} height={162} alt="" aria-hidden="true" unoptimized className={className} />;
}
